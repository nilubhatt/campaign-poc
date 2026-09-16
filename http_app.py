"""
HTTP surface — what Claude Web / cowork connects to as a custom connector.

Built on MCPServer's own Streamable HTTP app (serves /mcp), onto which we add:
  POST /upload   — large-file side-channel: returns {asset_id} to pass to upload_campaign
  GET  /healthz  — liveness + which vector/embed backends are live

Run behind a reachable HTTPS URL (localhost → a tunnel such as cloudflared/ngrok) and add it
in Claude Web → Settings → Connectors → Add custom connector, pointing at <public>/mcp.

Auth is the no-op seam (auth.py) in V1; claude.ai connectors will want OAuth — that slots
into auth.authenticate() without moving these call sites.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

import auth
import clip_embed
import embedding
import config
import core
import extract
import store
import vectorstore
from mcp_server import mcp

app = mcp.streamable_http_app()   # Starlette app already serving /mcp


async def _check_auth(request: Request) -> JSONResponse | None:
    try:
        await auth.authenticate(request.headers.get("authorization"))
        return None
    except auth.AuthError as exc:
        return JSONResponse({"error": str(exc)}, status_code=401)


async def upload(request: Request):
    if (deny := await _check_auth(request)) is not None:
        return deny
    form = await request.form()
    upl = form.get("file")
    if upl is None:
        return JSONResponse({"error": "multipart field 'file' is required"}, status_code=400)
    data = await upl.read()
    if len(data) > config.MAX_ASSET_BYTES:
        return JSONResponse({"error": f"file exceeds {config.MAX_ASSET_BYTES // (1024*1024)} MB"}, status_code=413)
    mime = extract.guess_mime(upl.filename or "asset.bin")
    if mime not in config.ALLOWED_MIME and mime not in config.ALLOWED_IMAGE_MIME:
        return JSONResponse(
            {"error": f"unsupported type {mime}; POC accepts PDF, PPTX, and images"},
            status_code=400,
        )
    asset_id = uuid.uuid4().hex
    dest_dir = config.UPLOAD_DIR / asset_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe = Path(upl.filename or "asset.bin").name
    (dest_dir / safe).write_bytes(data)
    return JSONResponse({"asset_id": asset_id, "filename": safe, "size_bytes": len(data)})


async def healthz(request: Request):
    conn = store.connect()
    try:
        # Whether the vision weights actually resolved is the thing that was invisible in
        # the field — the only way to discover visual search was off was a 60s timeout and
        # a read of the server's source. Reported as a value, not by crashing at boot.
        # The same report the health_check tool and the CLI give, so three surfaces cannot
        # disagree about one machine. probe=False keeps this cheap: /healthz gets polled,
        # and hitting the embedder on every poll would be its own problem.
        report = core.health_check(conn, probe=False)
        report.update({
            "status": "ok" if report["ok"] else "degraded",
            "embed_provider": config.EMBED_PROVIDER,
            "clip_provider": config.CLIP_PROVIDER,
            "auth_provider": config.AUTH_PROVIDER,
        })
        return JSONResponse(report)
    finally:
        conn.close()


app.router.routes.append(Route("/upload", upload, methods=["POST"]))
app.router.routes.append(Route("/healthz", healthz, methods=["GET"]))


def main():
    import uvicorn
    config.ensure_dirs()
    store.init_db()
    # Load the CLIP model now, not on the first tool call — a live MCP request over a
    # tunnel is the wrong place for a first-time model load (risks the client's
    # tool-call timeout; review flagged this).
    clip_embed.warm_up()
    embedding.warm_up()
    # §12.1: the rulebook, before the port opens. The startup load landed in `main.py stdio`
    # alone, so an HTTP deployment discovered a broken rulebook as a tool error mid-request —
    # and `readiness`, the tool whose job is saying what this product cannot do, was the first
    # to crash. A server that cannot state its own limits is the worst one to leave running.
    import rulebook

    print(f"Rulebook: {rulebook.version()}", file=sys.stderr)
    uvicorn.run(app, host=config.HTTP_HOST, port=config.HTTP_PORT)


if __name__ == "__main__":
    main()
