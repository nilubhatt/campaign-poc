"""
The same tool surface, described for a caller that cannot speak MCP.

**Why this exists.** Both transports this product ships speak MCP: `main.py stdio` for a
desktop client, and `/mcp` for a Claude Web connector. A firm that permits neither — no local
desktop client, no custom connectors, only its own agent stack making HTTPS calls — cannot
reach the product at all, however open its network is. The block is a policy about protocols,
not about ports, so a second HTTP endpoint speaking the same protocol answers nothing.

**Why it is GENERATED.** This file writes no paths and describes no tool. It reads the one
registry `mcp_server` already exposes and renders it, because the alternative — REST routes
written beside the MCP tools — is two implementations of one surface, and every P1 the §8.7
review produced was exactly that shape: a question the live path already answered, answered
again somewhere else, differently. A hand-written path is a promise that drifts the first time
a tool gains an argument. A generated one cannot.

So the document has no opinions. One POST per tool, the tool's own JSON Schema as the body,
its docstring as the description, and the server's own INSTRUCTIONS as the preamble — the same
text an MCP client is handed on connect. `llms_txt` is the same content again for an agent that
would rather read prose than a schema.

**What it deliberately does not do.** It does not flatten 65 tools into REST resources. "POST
/campaigns" reads better and would require a hand-written mapping from verbs to tools, which is
the drift this file exists to avoid — and the caller here is a model reading descriptions, not
a developer memorising a URL scheme.
"""
from __future__ import annotations

from typing import Optional

API_PREFIX = "/api"
# The shape every refusal already has, everywhere in this product: `{"error": "..."}` carrying
# the sentence a person can act on. Documented rather than invented, so a caller that reads
# this spec sees the convention the tools have used since §5.1.
_REFUSAL = {
    "description": "The tool refused, and the message says why and what to do instead. This "
                   "is the product's ordinary way of saying no — a validation message, a "
                   "not-found id, a write that needs `confirm=true` first — not a server "
                   "fault.",
    "content": {"application/json": {"schema": {
        "type": "object",
        "properties": {"error": {"type": "string"}},
        "required": ["error"],
    }}},
}


def _summary(description: Optional[str]) -> str:
    """The first sentence of a tool's docstring, for a one-line listing."""
    text = " ".join((description or "").split())
    if not text:
        return ""
    cut = text.find(". ")
    return text if cut < 0 else text[:cut + 1]


def document(tools: list, server_url: Optional[str] = None) -> dict:
    """The OpenAPI 3.1 document, rendered from the tool registry.

    `tools` is passed in rather than fetched. Reading the registry means awaiting it, and a
    generator that owns an event loop cannot be called from inside the request that wants the
    document — which is how the first version of this failed. A pure function of the registry
    is also the version a test can hand two tools and check.

    3.1 rather than 3.0 because the tools' schemas are JSON Schema 2020-12 — `anyOf` with
    `null` for an optional argument, `enum` on the forgiving-input types — and 3.0 cannot
    carry them without a translation step that would have to be written, tested and kept
    honest. The version that already matches the input is the version to emit.
    """
    import config
    import mcp_server

    paths = {}
    for tool in tools:
        paths[f"{API_PREFIX}/{tool.name}"] = {"post": {
            "operationId": tool.name,
            "summary": _summary(tool.description),
            # The docstring in full. It is what an MCP client is given, and it carries the
            # things a schema cannot: when to ask the user first, what a preview means, which
            # argument is a claim rather than a fact.
            "description": tool.description or "",
            "requestBody": {"required": bool(tool.input_schema.get("required")),
                            "content": {"application/json": {"schema": tool.input_schema}}},
            "responses": {
                # No output schema: these tools return a dict shaped by what they found, and
                # inventing one here would be this file forming an opinion about a surface it
                # is supposed to be describing. What every response DOES carry is documented
                # in the preamble instead.
                "200": {"description": "The tool's result.",
                        "content": {"application/json": {"schema": {"type": "object"}}}},
                "400": _REFUSAL,
                "401": {"description": "Authentication failed or was absent."},
            },
        }}

    return {
        "openapi": "3.1.0",
        "info": {
            "title": "Campaign Intelligence",
            "version": config.VERSION,
            "description": _preamble(mcp_server.INSTRUCTIONS),
        },
        "servers": [{"url": server_url or "/"}],
        "paths": paths,
    }


def _preamble(instructions: str) -> str:
    """The server's own instructions, plus the three things an HTTP caller needs that an MCP
    client is told by the protocol."""
    return f"""{instructions}

## Calling this over HTTP

One POST per tool at `{API_PREFIX}/<tool_name>`, with the tool's arguments as a JSON object.
Every path, schema and description in this document is generated from the same tool registry
an MCP client sees, so the two surfaces cannot disagree about what a tool takes.

Three conventions that are not in the schemas:

- **A refusal is a 400 carrying `{{"error": "..."}}`**, and the message is written to be read
  by whoever has to act on it. It is not a fault: a write that needs confirming, an id that is
  not on file, and an enum value the product can teach you about all arrive this way.
- **A write previews unless you confirm it.** Tools that store something take `confirm`, and
  with `confirm` absent or false they return what they WOULD do and store nothing. Show that
  to the person whose data it is, then call again with `confirm: true`.
- **Responses carry their own next step.** `next_actions` is a list of
  `{{label, why, tool, prefilled_args}}` — offers, not instructions. And every claim carries a
  `basis`: `computed` is a fact this server derived, `judged` is a model's opinion, `stated` is
  something a person asserted, `heuristic` is a guess this product is telling you not to trust
  as more than that.

Large files go to `POST /upload` as multipart `file`, which returns an `asset_id` to pass to
the tool that wants it. `GET /healthz` says whether the backends are live.
"""


def llms_txt(tools: list, server_url: Optional[str] = None) -> str:
    """The same surface as prose, for an agent that would rather read than parse.

    Generated for the same reason the document is: a hand-written summary of 65 tools is a
    second description of the product, and the first thing a second description does is go
    out of date.
    """
    import config
    import mcp_server

    lines = [
        "# Campaign Intelligence",
        "",
        f"> Version {config.VERSION}. A campaign-intelligence library that judges marketing "
        f"briefs against what an agency has actually done before, and says what it could not "
        f"check.",
        "",
        "This file is generated from the same tool registry the API and the MCP server expose. "
        "If it disagrees with them, it is a bug in the generator rather than a difference of "
        "opinion.",
        "",
        "## How to drive it",
        "",
        _preamble(mcp_server.INSTRUCTIONS).split("## Calling this over HTTP")[1].strip(),
        "",
        f"The machine-readable description is at `/openapi.json`"
        f"{f' on {server_url}' if server_url else ''}.",
        "",
        "## What the server says about itself",
        "",
        mcp_server.INSTRUCTIONS.strip(),
        "",
        "## The tools",
        "",
    ]
    for tool in tools:
        summary = _summary(tool.description)
        lines.append(f"- `POST {API_PREFIX}/{tool.name}` — {summary}" if summary
                     else f"- `POST {API_PREFIX}/{tool.name}`")
    return "\n".join(lines) + "\n"
