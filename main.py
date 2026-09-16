"""
Unified entry point for the packaged binary.

    campaign-intelligence serve [--host H --port P --embed ollama|hash|voyage]   # HTTP (Web + Desktop)
    campaign-intelligence stdio                                                  # stdio (Claude Desktop, local)

One executable covers both transports so the same bundle serves every client.
"""
from __future__ import annotations

import argparse
import os
import sys

# Dependency-free by design: --version must answer on a machine where config, the model
# and the database are all broken, which is precisely when somebody asks.
import version


def main() -> int:
    p = argparse.ArgumentParser(prog="campaign-intelligence")
    # The first question support asks, and it has to work on a machine where the product is
    # broken — so it touches no database, no embedder and no model (§3.2, defect 10).
    p.add_argument("--version", action="version",
                   version=f"campaign-intelligence {version.FULL}")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("serve", help="run the HTTP server (MCP /mcp + /upload + /healthz)")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--embed", default=None, choices=["ollama", "hash", "voyage"])

    sub.add_parser("stdio", help="run over stdio for a local Claude Desktop connector")

    sub.add_parser("check-weights", help="report whether the CLIP weights resolved (exit 1 if not)")
    sub.add_parser("disclosure", help="print what this product stores about people, where, "
                                      "for how long, and how to see or remove it")
    sub.add_parser("init", help="create the data directory and database (idempotent); "
                                "run by the installers before the self-test")

    h = sub.add_parser("health-check", help="check every component and the library's coverage "
                                            "(exit 1 if anything is wrong)")
    # An installer parsing prose is an installer that breaks when the prose improves (§4.1).
    h.add_argument("--json", action="store_true",
                   help="emit the report as JSON for a script to read")
    h.add_argument("--wait", type=float, default=0.0, metavar="SECONDS",
                   help="retry while the only failure is a service still starting up "
                        "(used by the installers, which start Ollama moments earlier)")

    c = sub.add_parser("configure-desktop", help="add this server to Claude Desktop's config (merges, backs up)")
    sub.add_parser("unconfigure-desktop",
                   help="remove this server from Claude Desktop's config (used by uninstall)")
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
        import embedding
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
        embedding.warm_up()
        # §12.1: the rulebook, read before the first request rather than on the first
        # judgment. `load()` RAISES on a file that will not parse, and the point of doing it
        # here is WHERE the error lands: discovered lazily it surfaces as a tool error to a
        # model in the middle of judging something, long after whoever edited the file has
        # moved on. Failing at startup puts it where the gesture was. It is deliberately not
        # caught — a server running with the rules quietly absent is the failure this whole
        # item removes, wearing the face of a successful start.
        import rulebook

        print(f"Rulebook: {rulebook.version()}", file=sys.stderr)
        mcp.run(transport="stdio")
    elif cmd == "init":
        # The installers' final self-test opens the database READ-ONLY on purpose (§2.3: a
        # diagnostic must not repair what it is diagnosing), so on a machine where the
        # server has never run there was nothing to open and every fresh install failed its
        # own gate with "FAIL database". Creating it is an install step, not a diagnostic's
        # side effect.
        #
        # It also answers a question worth asking at install time rather than at first use:
        # whether the data location is writable by whoever the installer is running as. On
        # Windows an administrator may be installing for somebody else entirely.
        import config
        import store
        config.ensure_dirs()
        store.init_db()
        import rulebook

        print(f"Data directory: {config.DATA_DIR}")
        print(f"Database:       {config.DB_PATH}")
        # §12.2: where the customer's own rules go, said at the moment somebody is setting
        # this up. §12.1 ships the rulebook EMPTY on purpose — the product owns its judgment
        # discipline and the customer owns the rules about their briefs — and that decision is
        # worth nothing if nobody is ever told where their half lives. §10.6's rule, applied
        # to a file rather than a tool: a capability nobody is told about is one nobody uses.
        print(f"Your rulebook:  {rulebook.overlay_path()}"
              + ("" if rulebook.overlay_path().exists() else "   (not written yet)"))
        # §12.3: and what to copy from. An example nobody is told about is one nobody reads —
        # §10.6's rule, applied to a file.
        example = config.app_dir() / "example-rulebook.yaml"
        print(f"Worked example: {example}"
              + ("" if example.exists() else "   (source checkout: docs/example-rulebook.yaml)"))
        # §11.7/D26: the disclosure, at the defined moment. It existed as a string generator
        # with no call sites — closed by dead code, which is worse than not done, because the
        # tracker said it was finished. A disclosure nobody is ever shown is not a disclosure,
        # and this is the one artefact in the item whose value is entirely in WHEN it appears.
        #
        # Here rather than at first use: `init` is what the installers run, so it is the
        # moment somebody is setting this up and can still decide not to.
        import people
        print()
        print(people.install_disclosure())

    elif cmd == "disclosure":
        # §11.7/D26: reachable without running an install, so the Windows wizard page, the
        # two shell installers and an admin asking "what does this keep?" all read the same
        # text — generated from `retention()`, so it cannot drift from what the product does.
        import people
        print(people.install_disclosure())

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
    elif cmd == "health-check":
        # The same answer the health_check tool gives, for the installer's post-install
        # self-test (item 4.1) and for an admin at a terminal. A non-zero exit is what lets
        # an installer refuse to report success over a broken install.
        import clip_embed
        import core
        # Load the vision model for real: "the weights resolved" is not "the weights work",
        # and a corrupt checkpoint passed the resolved-only check — precisely the case the
        # post-install self-test exists to catch (item 4.1).
        clip_embed.warm_up()
        wait = getattr(args, "wait", 0.0)
        if wait:
            print(f"Waiting up to {wait:g}s for everything to come up...")
        report = core.wait_until_ready(timeout=wait) if wait else core.health_check_cli()
        if getattr(args, "json", False):
            import json
            print(json.dumps(report, indent=2))
            return 0 if report["ok"] else 1
        for name, component in report["components"].items():
            print(f"{'ok ' if component['ok'] else 'FAIL'}  {name}: {component['detail']}")
            if not component["ok"]:
                print(f"      -> {component['remedy']}")
        coverage = report["coverage"]
        print(f"      {coverage['campaigns']} campaigns, {coverage['images']} images; "
              f"{coverage['sections_unindexed']} sections and "
              f"{coverage['images_unindexed']} images not yet searchable")
        if report.get("backlog_remedy"):
            print(f"      -> {report['backlog_remedy']}")
        return 0 if report["ok"] else 1
    elif cmd == "unconfigure-desktop":
        _unconfigure_desktop()

    elif cmd == "configure-desktop":
        _configure_desktop(http_url=args.http)
    else:
        p.print_help()
    return 0


def _unconfigure_desktop():
    """Remove only our own entry from Claude Desktop's config.

    Both Unix uninstallers did this; Windows did not, so an uninstall left Claude Desktop
    pointing at a binary that no longer exists — and the installer's own header claimed
    "uninstall removes all of it". Never fails: an uninstaller that stops because the thing
    it was removing had already gone leaves a half-removed product behind.
    """
    import json

    cfg = _desktop_config_path()
    if not cfg.exists():
        return
    try:
        data = json.loads(cfg.read_text(encoding="utf-8") or "{}")
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        print(f"warning: could not read {cfg} ({exc}); left it alone.")
        return
    servers = data.get("mcpServers") or {}
    if servers.pop("campaign-intelligence", None) is None:
        return
    try:
        cfg.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Removed the Claude Desktop connector entry from {cfg}")
    except OSError as exc:
        print(f"warning: could not update {cfg} ({exc}).")


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
            # encoding="utf-8" explicitly: this file is UTF-8 and Python's default is the
            # locale codec, which on Windows is cp1252 — defect 11's own bug class, running
            # on every Windows and Linux install. Another connector pointing at
            # C:\Users\José came back as C:\Users\JosÃ© and was written back corrupted,
            # and a byte cp1252 cannot decode raised UnicodeDecodeError, which nothing
            # caught.
            data = json.loads(cfg.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError:
            print(f"warning: {cfg} is not valid JSON; leaving it alone.")
            return
        except UnicodeDecodeError as exc:
            print(f"warning: {cfg} could not be read as UTF-8 ({exc}); leaving it alone "
                  f"rather than rewriting it and losing whatever is in there.")
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
    # ensure_ascii=False keeps a non-ASCII path readable in the file rather than escaped,
    # and the explicit encoding is what makes that safe to write.
    cfg.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Configured Claude Desktop connector in {cfg}")
    print("Restart Claude Desktop (fully quit + reopen) to load it.")


if __name__ == "__main__":
    raise SystemExit(main())
