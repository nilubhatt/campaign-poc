"""
Unified entry point for the packaged binary.

    campaign-intelligence serve [--host H --port P --embed ollama|hash|voyage]   # HTTP (Web + Desktop)
    campaign-intelligence stdio                                                  # stdio (Claude Desktop, local)

One executable covers both transports so the same bundle serves every client.
"""
from __future__ import annotations

import argparse
import os


def main() -> None:
    p = argparse.ArgumentParser(prog="campaign-intelligence")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("serve", help="run the HTTP server (MCP /mcp + /upload + /healthz)")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--embed", default=None, choices=["ollama", "hash", "voyage"])

    sub.add_parser("stdio", help="run over stdio for a local Claude Desktop connector")

    args = p.parse_args()
    cmd = args.cmd or "serve"

    if cmd == "serve":
        if args.host:  os.environ["CAMPAIGN_POC_HOST"] = args.host
        if args.port:  os.environ["CAMPAIGN_POC_PORT"] = str(args.port)
        if args.embed: os.environ["CAMPAIGN_POC_EMBED_PROVIDER"] = args.embed
        import http_app
        http_app.main()
    elif cmd == "stdio":
        import config
        import store
        from mcp_server import mcp
        config.ensure_dirs()
        store.init_db()
        mcp.run(transport="stdio")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
