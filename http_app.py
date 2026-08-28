"""
HTTP surface — what Claude Web / cowork connects to as a custom connector.

  POST/GET /mcp   — MCP over Streamable HTTP (remote MCP server = a claude.ai custom connector)
  POST /upload    — large-file side-channel: returns {asset_id} to pass to upload_campaign
  GET  /healthz   — liveness

Run behind a reachable HTTPS URL (localhost → a tunnel such as cloudflared/ngrok) and add it
in Claude Web → Settings → Connectors → Add custom connector, pointing at <public>/mcp.

Auth is the no-op seam (auth.py) in V1; claude.ai connectors will want OAuth — that slots
into auth.authenticate() without moving these call sites.

The StreamableHTTPSessionManager binding (import path, constructor args, run()/handle_request)
is verified against modelcontextprotocol/python-sdk main.
"""
from __future__ import annotations

import contextlib
import uuid

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

import auth
import config
import extract
import store
from mcp_server import app as mcp_app

from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

_sessions = StreamableHTTPSessionManager(app=mcp_app, json_response=False, stateless=True)


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    config.ensure_dirs()
    store.init_db()
    async with _sessions.run():
        yield


api = FastAPI(title="Campaign Intelligence (lean)", lifespan=_lifespan)


async def _principal(request: Request) -> auth.Principal:
    try:
        return await auth.authenticate(request.headers.get("authorization"))
    except auth.AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc))


async def _mcp_asgi(scope, receive, send):
    if scope["type"] == "http":
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        try:
            await auth.authenticate(headers.get("authorization"))
        except auth.AuthError as exc:
            await JSONResponse({"error": str(exc)}, status_code=401)(scope, receive, send)
            return
    await _sessions.handle_request(scope, receive, send)


api.mount("/mcp", _mcp_asgi)


@api.post("/upload")
async def upload(file: UploadFile = File(...), _who: auth.Principal = Depends(_principal)):
    """Stage a large deck; returns {asset_id, filename} to pass as upload_campaign(asset_ref)."""
    data = await file.read()
    if len(data) > config.MAX_ASSET_BYTES:
        raise HTTPException(status_code=413, detail=f"file exceeds {config.MAX_ASSET_BYTES // (1024*1024)} MB")
    mime = extract.guess_mime(file.filename or "asset.bin")
    if mime not in config.ALLOWED_MIME:
        raise HTTPException(status_code=400, detail=f"unsupported type {mime}; POC accepts PDF and PPTX")
    asset_id = uuid.uuid4().hex
    dest_dir = config.UPLOAD_DIR / asset_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    from pathlib import Path
    safe = Path(file.filename or "asset.bin").name
    (dest_dir / safe).write_bytes(data)
    return {"asset_id": asset_id, "filename": safe, "size_bytes": len(data)}


@api.get("/healthz")
async def healthz():
    conn = store.connect()
    try:
        import vectorstore
        return {"status": "ok", "vector_backend": vectorstore.backend_name(conn),
                "embed_provider": config.EMBED_PROVIDER, "auth_mode": config.AUTH_MODE}
    finally:
        conn.close()


def main():
    import uvicorn
    config.ensure_dirs()
    store.init_db()
    uvicorn.run(api, host=config.HTTP_HOST, port=config.HTTP_PORT)


if __name__ == "__main__":
    main()
