"""
stdio entrypoint — the cleanest LOCAL path for Claude Desktop.

Claude Desktop launches this process directly and talks to it over stdio (pipes): no HTTP
port, no firewall prompt, no Node/mcp-remote, no HTTPS tunnel. Use this when the only client
is Claude Desktop on the same machine. For Claude Web (or remote Desktop), use http_app.py
instead — it serves the same tools over Streamable HTTP.

Claude Desktop config (claude_desktop_config.json):

    {
      "mcpServers": {
        "campaign-intelligence": {
          "command": "<path-to-python>",
          "args": ["<path-to>/stdio_server.py"],
          "env": { "CAMPAIGN_POC_EMBED_PROVIDER": "ollama" }
        }
      }
    }
"""
from __future__ import annotations

import clip_embed
import embedding
import sys

import config
import store
from mcp_server import mcp

def main():
    """The source-checkout entry point, as a function so the sequence can be tested.

    It was a bare `if __name__ == "__main__":` block, which is why nothing noticed when
    `main.py stdio` — the entry point the INSTALLERS wire in — was missing the warm-up this
    file had. That was product-review defect 04, found by a reviewer hitting a 60-second
    timeout rather than by anything in the suite, because neither sequence could be called.
    """
    import rulebook

    config.ensure_dirs()
    store.init_db()
    clip_embed.warm_up()  # load now, not on the first tool call (see http_app.main())
    embedding.warm_up()
    # §12.1: before serving, for the same reason as `main.py stdio` — discovered lazily, a
    # broken rulebook surfaces as a tool error mid-judgment, and the first tool to crash is
    # the one whose job is saying what the product cannot do.
    print(f"Rulebook: {rulebook.version()}", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
