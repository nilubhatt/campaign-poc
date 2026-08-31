# Running on Linux

Two ways: **source install** (a script that mirrors the Windows launcher) or a **prebuilt
bundle** (no Python needed). Works with both **Claude Desktop** and **Claude Web**.

## Option A — source install (`run.sh`)

```bash
cd campaign-poc
./run.sh --setup        # venv + deps + auto-install Ollama (official script) + pull embed model
./run.sh                # ON DEMAND: HTTP server in the foreground (127.0.0.1:8086)
./run.sh --service      # BACKGROUND: systemd --user service, starts now + on login
./run.sh --service --stop   # stop + remove the service
./run.sh --host 0.0.0.0 --port 8086   # bind for a tunnel (Claude Web)
```

Requires `python3` (3.10+) and `curl`. `--setup` needs internet once (Ollama installer + model).
Data lives in `~/campaign-poc-data`. Service logs: `journalctl --user -u campaign-intelligence -f`.

## Option B — prebuilt bundle (no Python)

Grab `campaign-intelligence-linux-x86_64.tar.gz` from the repo's Releases (built by CI), then
either run it directly, or use the bundled **installer** (recommended — it wires Claude Desktop
and Ollama for you):

```bash
tar xzf campaign-intelligence-linux-x86_64.tar.gz
cd campaign-intelligence

./install.sh                 # install to ~/.local + wire Claude Desktop + ensure Ollama
./install.sh --service       # also run in the background (systemd --user)
./uninstall.sh               # remove it (--purge also deletes data)

# ...or run in place without installing:
./campaign-intelligence serve            # HTTP (Web + Desktop)
./campaign-intelligence stdio            # stdio (local Desktop)
```

You still need **Ollama** for real semantic search: `curl -fsSL https://ollama.com/install.sh | sh`
then `ollama pull nomic-embed-text`. (Without it, set `CAMPAIGN_POC_EMBED_PROVIDER=hash` for a
non-semantic smoke test.)

Check health: `curl http://localhost:8086/healthz` — `vector_backend` should read `sqlite-vec`.

## Connect a client

- **Claude Desktop (local, no HTTP):** point its config at the `stdio` command —
  `~/.config/Claude/claude_desktop_config.json`:
  ```json
  { "mcpServers": { "campaign-intelligence": {
      "command": "/abs/path/campaign-intelligence/campaign-intelligence", "args": ["stdio"] } } }
  ```
  (Source install: use `.venv/bin/python` + `stdio_server.py` instead.)
- **Claude Web / cowork:** run bound to `0.0.0.0`, open a tunnel
  (`cloudflared tunnel --url http://localhost:8086`), add `<public-url>/mcp` as a custom connector.

## Building the bundle yourself

`./build.sh` (on Linux) produces `campaign-intelligence-linux-$(arch).tar.gz`. PyInstaller is
native — a Linux binary must be built on Linux. CI (`.github/workflows/build.yml`) builds
Linux, Windows, and macOS bundles on a `v*` tag and attaches them to a GitHub Release.
