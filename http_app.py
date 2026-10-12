"""
HTTP surface — what Claude Web / cowork connects to as a custom connector.

Built on MCPServer's own Streamable HTTP app (serves /mcp), onto which we add:
  POST /upload          — large-file side-channel: returns {asset_id} to pass to upload_campaign
  POST /api/<tool>      — every MCP tool, for a caller that cannot speak MCP (see openapi.py)
  GET  /openapi.json    — the generated description of /api; GET /llms.txt, the same as prose
  GET  /healthz         — liveness, which backends are live, and who can reach this server

`/mcp`, `/api/<tool>` and `/upload` sit behind one guard (`GUARDED`): Host/Origin checks
against `auth.allowed_hosts()`, a body cap, and `auth.authenticate()`.

Run behind a reachable HTTPS URL (localhost → a tunnel such as cloudflared/ngrok) and add it
in Claude Web → Settings → Connectors → Add custom connector, pointing at <public>/mcp.

Auth is the no-op seam (auth.py) by default; claude.ai connectors will want OAuth — that slots
into auth.authenticate() without moving the guard.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
from mcp.server.transport_security import (
    DEFAULT_MAX_REQUEST_BODY_SIZE,
    RequestBodyLimitMiddleware,
    TransportSecuritySettings,
)
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
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

# Starlette app already serving /mcp. Its own Host/Origin check is turned OFF here and done by
# `GUARDED` instead, for two reasons: left to itself it accepts only a loopback Host — so
# Claude Web reaching it through a tunnel or a public HTTPS name got a 421 — and it guards
# `/mcp` alone, where the same check belongs on every route that reaches the library. Its
# Content-Type rule and 4 MiB body cap on `/mcp` stay on.
app = mcp.streamable_http_app(
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))
# Read from `openapi`, never spelled twice: the prefix appears in the generated document and in
# the route that serves it, and two copies of a URL are two things to keep in step.
import openapi as _openapi

openapi_prefix = _openapi.API_PREFIX


def _guarded(path: str) -> str | None:
    """Which guarded route a path is, or None. Everything else — `/healthz`, `/openapi.json`,
    `/llms.txt` — carries no library data and stays readable without a token: an installer
    polls `/healthz` before anybody has one."""
    if path == "/mcp" or path.startswith("/mcp/"):
        return "mcp"
    if path.startswith(f"{openapi_prefix}/"):
        return "api"
    if path == "/upload":
        return "upload"
    return None


class GUARDED:
    """The one gate in front of every route that reaches the library: `/mcp`, `/api/<tool>`
    and `/upload`. In this order:

    1. **One Host, then Host and Origin** through `auth.host_allowed` — 400 / 421 / 403. DNS rebinding arrives
       with a foreign Host; a page on another site arrives with its Origin. MCP's own check
       does this for `/mcp` but compares strings verbatim — `LOCALHOST`, `name:443` and
       `127.0.0.2` were all refused — so the comparison is `auth`'s, the same one the
       generated document's `servers` URL uses. Every method: `/mcp` GET attaches to the
       notification stream and DELETE ends a session.
    2. **A JSON body must say it is JSON** on `/api` — 400. `text/plain` is a CORS "simple
       request" a browser sends from any page without asking; requiring `application/json`
       forces the preflight this server never answers. An EMPTY body is exempt: there is
       nothing to forge, and plenty of tools take no arguments.
    3. **`auth.authenticate`** — 401 with `WWW-Authenticate: Bearer`. `/mcp` never called it,
       so a configured provider protected only the two side doors. Before the body is read,
       so an unauthenticated caller is refused on its headers alone.
    4. **Body size** on `/api` (MCP's 4 MiB) and `/upload` (the asset cap plus multipart
       overhead) — 413, counted as it streams. `/mcp` carries its own.

    Pure ASGI rather than `BaseHTTPMiddleware`, so `/mcp`'s streamed responses pass through
    untouched.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        route = _guarded(scope.get("path", "")) if scope["type"] == "http" else None
        if route is None:
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        refused = _refuse_host_or_origin(scope, request)
        if refused is None and route == "api" and _has_body(request) and not (
                request.headers.get("content-type", "").lower().startswith("application/json")):
            refused = JSONResponse({"error": "the body must be sent as application/json"},
                                   status_code=400)
        if refused is None:
            try:
                await auth.authenticate(request.headers.get("authorization"))
            except auth.AuthError as exc:
                refused = JSONResponse({"error": _caller_facing(exc)}, status_code=401,
                                       headers={"www-authenticate": "Bearer"})
        if refused is not None:
            await refused(scope, receive, send)
            return
        limit = {"api": DEFAULT_MAX_REQUEST_BODY_SIZE,
                 "upload": config.MAX_ASSET_BYTES + 1024 * 1024}.get(route)
        inner = RequestBodyLimitMiddleware(self.app, limit) if limit else self.app
        await inner(scope, receive, send)


def _refuse_host_or_origin(scope, request: Request):
    # RFC 9110 §7.2: one Host. Two is how a check passes on one name while a front proxy
    # routes on the other — this guard would read the first.
    if sum(1 for key, _ in scope.get("headers", ()) if key.lower() == b"host") > 1:
        return JSONResponse({"error": "a request carries one Host header"}, status_code=400)
    if not auth.host_allowed(request.headers.get("host")):
        return JSONResponse({"error": "this server does not answer to that Host name"},
                            status_code=421)
    origin = request.headers.get("origin")
    if origin is not None:
        # An Origin is `scheme://host[:port]` and nothing else (RFC 6454) — split by hand
        # rather than handed to `urlsplit`, which raises on a malformed bracket. That was a
        # 500 and a traceback before authentication, for any caller. `null`, a path, or a
        # scheme other than http(s) is refused.
        #
        # Accepted limit: the scheme is not compared with the public URL's, so with
        # CAMPAIGN_POC_BASE_URL=https://name an Origin of http://name passes. Exploiting it needs
        # a plaintext page on the same name AND no auth provider — and then the attacker can
        # call the server directly anyway; `/api` still needs a JSON preflight regardless.
        scheme, sep, host = origin.partition("://")
        if not (sep and scheme.lower() in ("http", "https") and auth.host_allowed(host)):
            return JSONResponse({"error": "requests from that origin are not accepted"},
                                status_code=403)
    return None


def _caller_facing(exc: auth.AuthError) -> str:
    """A provider nobody registered is the OPERATOR's problem — which variable, which call —
    and the banner and `/healthz` say so. The caller is told only that it cannot get in."""
    if config.AUTH_PROVIDER not in auth._PROVIDERS:
        return "authentication is not available on this server"
    return str(exc)


def _has_body(request: Request) -> bool:
    return (request.headers.get("content-length", "0") != "0"
            or "transfer-encoding" in request.headers)


async def upload(request: Request):
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
            # WHO CAN REACH THIS, from the one function that decides it. Reported here rather
            # than as a `health_check` component because that report's `ok` is what the
            # installers refuse an install on, and unconfigured auth is a deployment choice
            # rather than a broken build.
            "access": auth.exposure(),
        })
        return JSONResponse(report)
    finally:
        conn.close()


async def call_tool(request: Request):
    """One POST per tool, for a caller that cannot speak MCP (openapi.py says why).

    Dispatched through `mcp.call_tool`, which is the SAME entry point the protocol uses — so a
    tool reached this way runs the identical validation, the identical consent step and the
    identical refusals. A second dispatch path that reached into `core` directly would be two
    implementations of one surface, and it would drift on the first tool that gains an
    argument. Host, Origin, size and auth are `GUARDED`'s, before this runs.
    """
    name = request.path_params["tool"]
    known = {tool.name for tool in await mcp.list_tools()}
    if name not in known:
        # The list, because a 404 that does not say what IS available makes the caller guess.
        return JSONResponse(
            {"error": f"no tool named {name!r}. See /openapi.json for the {len(known)} this "
                      f"server offers."}, status_code=404)
    try:
        arguments = await request.json() if await request.body() else {}
    except ValueError as exc:
        return JSONResponse({"error": f"the body must be a JSON object: {exc}"},
                            status_code=400)
    if not isinstance(arguments, dict):
        return JSONResponse({"error": "the body must be a JSON object of the tool's arguments"},
                            status_code=400)

    # What the PROTOCOL handler does with these, mirrored: `call_tool` raises where
    # `tools/call` returns an `isError` result. A schema rejection (a string where an int
    # belongs, a missing required argument) is the caller's to fix — a 400 carrying the
    # message. A crash is the server's — a 500, still JSON, still carrying it. Uncaught, both
    # were Starlette's plain-text "Internal Server Error".
    try:
        result = await mcp.call_tool(name, arguments)
    except UnexpectedToolError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    except ToolError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    payload = json.loads(result.content[0].text) if result.content else {}
    # A REFUSAL IS A 400. `{"error": ...}` is what every tool in this product returns when it
    # says no — a validation message, an id that is not on file — and an HTTP caller reads that
    # from the status code before it reads the body. The message is unchanged: it was written
    # to be acted on.
    #
    # The KEY's presence, not the key being alone: a refusal carries company. The forgiving-
    # enum refusal adds `field` and `valid` so the caller can be taught rather than guess, and
    # an image search that cannot resolve its asset adds the warnings explaining why. Requiring
    # `error` to be the only key reported all of those as 200 — the first version of this did,
    # and a caller checking the status would have read a refusal as a result. Verified against
    # every `{"error": ...}` return in the tree: they are all refusals.
    refused = isinstance(payload, dict) and "error" in payload
    return JSONResponse(payload, status_code=400 if refused else 200)


async def _requested_tools(request: Request):
    """The tools the caller asked to be described, or all of them.

    `?tools=a,b,c` exists because the whole document is 186KB across 65 operations, and a
    custom GPT handed all of it spends its context on a schema instead of on the work. A
    filtered document is still generated from the registry, so a subset cannot describe a tool
    the server does not have.

    Returns `(tools, refusal)`. An unknown name is refused with the list rather than silently
    dropped: a caller who mistypes one of twelve names and gets eleven back has a document that
    is quietly missing the tool they care about.
    """
    tools = await mcp.list_tools()
    # Every `tools=` the URL carries, not the last one; each name once, in the order asked;
    # and a filter that names nothing (`?tools=`, `?tools=,`) is the whole surface — a URL
    # built from an empty list, not a request for a document describing nothing.
    wanted = list(dict.fromkeys(
        name.strip() for asked in request.query_params.getlist("tools")
        for name in asked.split(",") if name.strip()))
    if not wanted:
        return tools, None
    known = {tool.name: tool for tool in tools}
    unknown = [name for name in wanted if name not in known]
    if unknown:
        return None, JSONResponse(
            {"error": f"no tool named {', '.join(repr(n) for n in unknown)}. "
                      f"Omit `tools` for all {len(known)}, or see /llms.txt for the names."},
            status_code=400)
    return [known[name] for name in wanted], None


async def openapi_json(request: Request):
    """The generated description. No auth: a caller cannot authenticate against a surface it
    has not been allowed to read, and this document contains no library data."""
    import openapi

    tools, refusal = await _requested_tools(request)
    if refusal is not None:
        return refusal
    return JSONResponse(openapi.document(tools, _public_url(request)), headers=_NO_STORE)


async def llms_txt(request: Request):
    """The same, as prose, and filterable the same way."""
    import openapi

    tools, refusal = await _requested_tools(request)
    if refusal is not None:
        return refusal
    return PlainTextResponse(openapi.llms_txt(tools, _public_url(request)), headers=_NO_STORE)


def _public_url(request: Request) -> str:
    """What the caller reached us on, honouring a reverse proxy's headers — a tunnel is the
    ordinary deployment here, and a document that names `localhost` as the server is a
    document nobody else can use.

    Only a name this server answers to (`auth.host_allowed`). These routes are unguarded, and
    `X-Forwarded-Host` is whatever the caller wrote: reflected verbatim, one request naming
    `evil.example` behind a caching proxy would hand every later GPT Action an attacker's
    host.

    So: a `CAMPAIGN_POC_BASE_URL` naming somewhere other than this machine is ALWAYS what is
    advertised — the operator has said what the server is called, and a forwarded loopback
    name, a dead port or a downgraded scheme is a cache away from every later reader. Only
    with no such URL (a laptop, a tunnel nobody named) is a forwarded name reflected, and then
    only one that parses as a name and a port and is on the allowlist — rebuilt from its parts,
    never echoed. Anything else falls back to `CAMPAIGN_POC_BASE_URL`."""
    published, _ = auth.public_base_url()
    if auth.public_name_is_set():
        return published
    forwarded_proto = request.headers.get("x-forwarded-proto")
    forwarded_host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    if forwarded_host and auth.host_allowed(forwarded_host):
        name, port = auth._split_host(forwarded_host)
        # Never `ws`/`wss`: uvicorn's proxy-header middleware accepts those from a trusted
        # loopback peer — where a same-machine tunnel connects from — and rewrites the
        # request's own scheme before it reaches here.
        own = request.url.scheme if request.url.scheme in ("http", "https") else "http"
        scheme = forwarded_proto if forwarded_proto in ("http", "https") else own
        host = (f"[{name}]" if ":" in name else name) + (f":{port}" if port else "")
        return f"{scheme}://{host}"
    # A URL with a problem is never handed out — the startup warning says it is ignored, and
    # `openapi.document` publishes `/` for an empty one.
    return published or ""


# The generated documents vary with the request's name and with configuration, and a cached
# copy is the channel the reflection above defends; neither is worth caching.
_NO_STORE = {"cache-control": "no-store"}


app.router.routes.append(Route("/upload", upload, methods=["POST"]))
app.router.routes.append(Route("/healthz", healthz, methods=["GET"]))
app.router.routes.append(Route("/openapi.json", openapi_json, methods=["GET"]))
app.router.routes.append(Route("/llms.txt", llms_txt, methods=["GET"]))
app.router.routes.append(Route(f"{openapi_prefix}/{{tool}}", call_tool, methods=["POST"]))
app.add_middleware(GUARDED)


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
    # SAID AT THE MOMENT SOMEBODY STARTS IT, not only in a document they may not have read.
    # The same sentence `/healthz` and the OpenAPI preamble carry.
    access = auth.exposure()
    if access["open_to_the_network"]:
        print(f"Access: {access['what_it_means']}", file=sys.stderr)
    else:
        print(f"Access: {access['auth_provider']} auth, bound to {access['bound_to']}",
              file=sys.stderr)
    if (problem := auth.base_url_problem()) is not None:
        print(f"WARNING: {problem}. It is ignored: the server will not answer to it, and the "
              f"generated documents will not advertise it.", file=sys.stderr)
    # Which names it answers to, because a 421 is otherwise the first anybody hears of it.
    print(f"Answers to Host: {', '.join(access['answers_to_hosts'])} "
          f"(add more with CAMPAIGN_POC_ALLOWED_HOSTS)", file=sys.stderr)
    uvicorn.run(app, host=config.HTTP_HOST, port=config.HTTP_PORT)


if __name__ == "__main__":
    main()
