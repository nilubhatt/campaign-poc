#!/usr/bin/env bash
# Campaign Intelligence (lean) — Linux/macOS launcher (source install; mirrors run.ps1).
#
#   ./run.sh --setup                 # venv + deps + auto-install Ollama + pull embed model
#   ./run.sh                         # run the HTTP server on demand (foreground)
#   ./run.sh --service               # install + start a systemd --user service (Linux)
#   ./run.sh --service --stop        # stop + remove the service
#   ./run.sh --host 0.0.0.0 --port 8086   # bind for a tunnel (Claude Web); default local-only
set -euo pipefail
cd "$(dirname "$0")"

MODE=ondemand; STOP=0; HOST=127.0.0.1; PORT=8086; EMBED=ollama; SETUP=0
while [ $# -gt 0 ]; do case "$1" in
  --setup) SETUP=1;;
  --service) MODE=service;;
  --stop) STOP=1;;
  --host) HOST="$2"; shift;;
  --port) PORT="$2"; shift;;
  --embed) EMBED="$2"; shift;;
  *) echo "unknown arg: $1"; exit 2;;
esac; shift; done

VENV=.venv; PY="$VENV/bin/python"
SERVICE=campaign-intelligence

ensure_ollama() {
  if command -v ollama >/dev/null 2>&1; then
    echo "Ollama already installed."
  else
    echo "Ollama not found — installing via the official script..."
    curl -fsSL https://ollama.com/install.sh | sh
  fi
  (ollama serve >/dev/null 2>&1 &) || true
  sleep 3
  echo "Pulling embedding model nomic-embed-text..."
  ollama pull nomic-embed-text
}

if [ "$SETUP" = 1 ]; then
  [ -x "$PY" ] || python3 -m venv "$VENV"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r requirements.txt
  ensure_ollama
  echo "Setup complete. Start with:  ./run.sh"
  exit 0
fi

[ -x "$PY" ] || { echo "Not set up. Run: ./run.sh --setup"; exit 1; }

if [ "$MODE" = service ]; then
  UNIT="$HOME/.config/systemd/user/$SERVICE.service"
  if [ "$STOP" = 1 ]; then
    systemctl --user stop "$SERVICE" 2>/dev/null || true
    systemctl --user disable "$SERVICE" 2>/dev/null || true
    rm -f "$UNIT"; systemctl --user daemon-reload 2>/dev/null || true
    echo "Service $SERVICE stopped and removed."; exit 0
  fi
  mkdir -p "$(dirname "$UNIT")"
  cat > "$UNIT" <<EOF
[Unit]
Description=Campaign Intelligence (lean) MCP/HTTP server
After=network.target

[Service]
WorkingDirectory=$(pwd)
Environment=CAMPAIGN_POC_EMBED_PROVIDER=$EMBED
Environment=CAMPAIGN_POC_HOST=$HOST
Environment=CAMPAIGN_POC_PORT=$PORT
ExecStart=$(pwd)/$PY -m http_app
Restart=on-failure

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable --now "$SERVICE"
  echo "Service $SERVICE started (systemd --user). Logs: journalctl --user -u $SERVICE -f"
  exit 0
fi

export CAMPAIGN_POC_EMBED_PROVIDER="$EMBED" CAMPAIGN_POC_HOST="$HOST" CAMPAIGN_POC_PORT="$PORT"
echo "Serving http://$HOST:$PORT  (/mcp, /upload, /healthz)"
exec "$PY" -m http_app
