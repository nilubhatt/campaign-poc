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
./run.sh --host 0.0.0.0 --port 8086   # bind to the LAN (a tunnel on this machine does not need it)
```

Requires `python3` (3.10+) and `curl`. `--setup` needs internet once (Ollama installer + model).
Data lives in `~/campaign-poc-data`. Service logs: `journalctl --user -u campaign-intelligence -f`.

## Option B — prebuilt bundle (no Python)

Grab `campaign-intelligence-linux-x86_64.tar.gz` from the repo's Releases (built by CI), then
either run it directly, or use the bundled **installer** (recommended — it wires Claude Desktop
for you). The bundle carries its own text embedder (a CPU-only Ollama and `nomic-embed-text`),
so nothing is downloaded and a network that blocks ollama.com is not a problem:

```bash
tar xzf campaign-intelligence-linux-x86_64.tar.gz
cd campaign-intelligence

./install.sh                 # install to ~/.local + wire Claude Desktop (downloads nothing)
./install.sh --service       # also run in the background (systemd --user)
./install.sh --service --base-url https://<name>   # ...reached by Claude Web at that public URL
./uninstall.sh               # remove it (--purge also deletes data)

# ...or run in place without installing:
./campaign-intelligence serve            # HTTP (Web + Desktop)
./campaign-intelligence stdio            # stdio (local Desktop)
```

The product starts the bundled embedder itself, on `127.0.0.1:11435` (an Ollama you run on 11434
is left alone), and `uninstall.sh` stops it. Its log is `~/campaign-poc-data/ollama.log`. Set
`OLLAMA_URL` to use an Ollama of your own instead.

Check health: `curl http://localhost:8086/healthz` — `vector_backend` should read `sqlite-vec`.

## Connect a client

- **Claude Desktop (local, no HTTP):** point its config at the `stdio` command —
  `~/.config/Claude/claude_desktop_config.json`:
  ```json
  { "mcpServers": { "campaign-intelligence": {
      "command": "/abs/path/campaign-intelligence/campaign-intelligence", "args": ["stdio"],
      "env": { "CAMPAIGN_POC_CLIP_WEIGHTS_PATH": "/abs/path/where/you/put/the/weights" } } } }
  ```
  (Source install: use `.venv/bin/python` + `stdio_server.py` instead.)

  `CAMPAIGN_POC_CLIP_WEIGHTS_PATH` tells the server to load the CLIP weights from disk
  instead of fetching them by tag from huggingface.co — set it when that host is blocked on
  your network, or when the weights were supplied out of band. Takes the checkpoint file
  (`open_clip_model.safetensors` / `open_clip_pytorch_model.bin`) or a folder holding it.
  A wrong path doesn't stop the server — text search and uploads work regardless; visual
  similarity is simply off, and the reason is printed at startup (in the client's MCP log
  for a stdio server, or `clip_weights` in `GET /healthz` when running over HTTP).
- **Claude Web / cowork:** open a tunnel on this machine
  (`cloudflared tunnel --url http://localhost:8086`) — the default loopback bind is enough — and
  add `<public-url>/mcp` as a custom connector.
  Start the server with `./run.sh --base-url https://<your-tunnel-name>` — the form is
  `https://name[:port][/path]`, see README § Claude Web (it is also written into
  the `--service` unit, which inherits no shell) — it answers only to
  Host names it knows, and a `421` means the tunnel's is not one of them.

## Building the bundle yourself

`./build.sh` (on Linux) produces `campaign-intelligence-linux-$(arch).tar.gz`. PyInstaller is
native — a Linux binary must be built on Linux. CI (`.github/workflows/build.yml`) builds
Linux, Windows, and macOS bundles on a `v*` tag and attaches them to a GitHub Release.
