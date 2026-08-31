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

import config
import store
from mcp_server import mcp

if __name__ == "__main__":
    config.ensure_dirs()
    store.init_db()
    mcp.run(transport="stdio")
