# Running on Windows

Phase-1 packaging for developers: a PowerShell launcher (`run.ps1`) that sets everything up
(including auto-installing Ollama), and runs the server either **on demand** or as a
**background service**. Works with both **Claude Desktop** and **Claude Web**.

## Prerequisites

- **Python 3.10+** — install from python.org, tick *"Add python.exe to PATH"*.
- Internet access **once** (first `-Setup` downloads the Ollama installer + the embedding model).
- **Claude Desktop** (for the local stdio path) and/or a Claude Web account with connectors.

## 1. One-time setup

```powershell
cd path\to\campaign-poc
.\run.ps1 -Setup
```

This creates a virtual environment, installs the Python dependencies, **auto-installs Ollama
if it's missing** (silent install of the official Windows package), starts it, and pulls the
`nomic-embed-text` embedding model. Re-run any time to update deps.

> If PowerShell blocks the script: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
> then re-run, or launch with `powershell -ExecutionPolicy Bypass -File .\run.ps1 -Setup`.

## 2. Run it — on demand or as a background service (the switch)

```powershell
.\run.ps1                    # ON DEMAND: runs in the foreground; Ctrl-C to stop
.\run.ps1 -Mode service      # BACKGROUND: registers a Task-Scheduler task, runs at logon, starts now
.\run.ps1 -Mode service -Stop   # stop + remove the background service
```

Default bind is **`127.0.0.1:8086`** (local only — no Windows Firewall prompt). Data lives in
`%LOCALAPPDATA%\CampaignIntelligence`. Check health any time:

```powershell
curl http://localhost:8086/healthz
# {"status":"ok","vector_backend":"sqlite-vec","embed_provider":"ollama",...}
```

If `vector_backend` shows `python-cosine-fallback`, the `sqlite-vec` extension couldn't load
on this Python build — the server still works (brute-force search); fine at this scale.

## 3a. Connect Claude Desktop — local, no HTTP (recommended)

Cleanest local path: Desktop launches the server over **stdio** (no port, no firewall, no
Node). Edit `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "campaign-intelligence": {
      "command": "C:\\path\\to\\campaign-poc\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\campaign-poc\\stdio_server.py"],
      "env": { "CAMPAIGN_POC_EMBED_PROVIDER": "ollama" }
    }
  }
}
```

Fully quit and reopen Claude Desktop. (Ollama must be running — `-Setup` starts it; it also
auto-starts at logon after install.) With stdio you don't run `run.ps1` at all for Desktop —
Desktop starts the process itself.

## 3b. Connect Claude Desktop — via the local HTTP server

If you prefer the running HTTP server (e.g. you also use Web), bridge it with `mcp-remote`
(needs Node/`npx`):

```json
{
  "mcpServers": {
    "campaign-intelligence": {
      "command": "npx",
      "args": ["mcp-remote", "http://localhost:8086/mcp"]
    }
  }
}
```

## 4. Connect Claude Web / cowork

Claude Web reaches your server from the cloud, so it needs a **public HTTPS URL** — run the
server bound to all interfaces and open a tunnel:

```powershell
.\run.ps1 -HostAddr 0.0.0.0 -Port 8086     # (Windows Firewall will prompt once — allow it)
# in another terminal, tunnel it:
cloudflared tunnel --url http://localhost:8086      # or: ngrok http 8086
```

Then Claude Web → **Settings → Connectors → Add custom connector** → `<public-url>/mcp`.

## Notes & next phases

- **Ollama on Windows** runs as a background service on `localhost:11434` and uses your NVIDIA
  GPU if present (CPU otherwise — fine for this small embed model).
- **Hardened Windows Service:** the `-Mode service` switch uses Task Scheduler (simple, runs at
  logon). For a true always-on service independent of login, wrap `python -m http_app` with
  **NSSM** or **WinSW** — a Phase-2 upgrade.
- **Phase 2 (no Python on target):** bundle into a one-folder `.exe` with PyInstaller (must
  include the `sqlite-vec` DLL). **Phase 3:** an Inno Setup installer that also installs Ollama,
  pulls the model on first run, and writes the Claude Desktop connector config.
