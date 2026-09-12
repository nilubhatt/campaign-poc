"""
Unified entry point for the packaged binary.

    campaign-intelligence serve [--host H --port P --embed ollama|hash|voyage]   # HTTP (Web + Desktop)
    campaign-intelligence stdio                                                  # stdio (Claude Desktop, local)

One executable covers both transports so the same bundle serves every client.
"""
from __future__ import annotations

import argparse
import os


def main() -> int:
    p = argparse.ArgumentParser(prog="campaign-intelligence")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("serve", help="run the HTTP server (MCP /mcp + /upload + /healthz)")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--embed", default=None, choices=["ollama", "hash", "voyage"])

    sub.add_parser("stdio", help="run over stdio for a local Claude Desktop connector")

    sub.add_parser("check-weights", help="report whether the CLIP weights resolved (exit 1 if not)")

    c = sub.add_parser("configure-desktop", help="add this server to Claude Desktop's config (merges, backs up)")
    c.add_argument("--http", metavar="URL", default=None,
                   help="wire via mcp-remote to a running HTTP server (e.g. http://localhost:8086/mcp)")

    args = p.parse_args()
    cmd = args.cmd or "serve"

    if cmd == "serve":
        if args.host:  os.environ["CAMPAIGN_POC_HOST"] = args.host
        if args.port:  os.environ["CAMPAIGN_POC_PORT"] = str(args.port)
        if args.embed: os.environ["CAMPAIGN_POC_EMBED_PROVIDER"] = args.embed
        import http_app
        http_app.main()
    elif cmd == "stdio":
        import clip_embed
        import config
        import store
        from mcp_server import mcp
        config.ensure_dirs()
        store.init_db()
        # THE path the installers wire into Claude Desktop (configure-desktop writes
        # args=["stdio"]), so skipping warm-up here meant every installed copy still loaded
        # the vision model inside the first image tool call — the 60s transport timeout the
        # product review hit. stdio_server.py (source checkouts) always did this; the frozen
        # binary's own entry point did not.
        clip_embed.warm_up()
        mcp.run(transport="stdio")
    elif cmd == "check-weights":
        # Exists so the build can verify the PACKAGED product rather than the source tree:
        # CI extracts the archive and runs this, which exercises app_dir() under a frozen
        # binary and the real install layout. Also what the post-install self-test calls.
        import clip_embed
        status = clip_embed.weights_status()
        print(f"source: {status.source}")
        print(f"path:   {status.path or '(not a local file)'}")
        print(f"ok:     {status.ok}")
        if not status.ok:
            print(f"{status.reason} {status.remedy}")
            return 1
    elif cmd == "configure-desktop":
        _configure_desktop(http_url=args.http)
    else:
        p.print_help()
    return 0


def _desktop_config_path():
    """Claude Desktop config location, per OS."""
    import sys
    from pathlib import Path
    if sys.platform == "win32":
        base = Path(os.getenv("APPDATA", Path.home()))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "Claude" / "claude_desktop_config.json"


def _configure_desktop(http_url=None):
    """Merge a 'campaign-intelligence' connector into Claude Desktop's config (backs up first)."""
    import json
    import shutil
    import sys
    from pathlib import Path

    cfg = _desktop_config_path()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if cfg.exists():
        shutil.copy2(cfg, str(cfg) + ".bak")
        try:
            data = json.loads(cfg.read_text() or "{}")
        except json.JSONDecodeError:
            print(f"warning: {cfg} is not valid JSON; leaving it alone.")
            return

    if http_url:
        entry = {"command": "npx", "args": ["mcp-remote", http_url]}
    elif getattr(sys, "frozen", False):
        # packaged binary → launch it directly over stdio
        entry = {"command": sys.executable, "args": ["stdio"]}
    else:
        # source checkout → python + stdio_server.py
        entry = {"command": sys.executable,
                 "args": [str(Path(__file__).resolve().parent / "stdio_server.py")]}

    data.setdefault("mcpServers", {})["campaign-intelligence"] = entry
    cfg.write_text(json.dumps(data, indent=2))
    print(f"Configured Claude Desktop connector in {cfg}")
    print("Restart Claude Desktop (fully quit + reopen) to load it.")


if __name__ == "__main__":
    raise SystemExit(main())
