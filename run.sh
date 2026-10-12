#!/usr/bin/env bash
# Campaign Intelligence (lean) — Linux/macOS launcher (source install; mirrors run.ps1).
#
#   ./run.sh --setup                 # venv + deps + auto-install Ollama + pull embed model
#   ./run.sh                         # run the HTTP server on demand (foreground)
#   ./run.sh --service               # install + start a systemd --user service (Linux)
#   ./run.sh --service --stop        # stop + remove the service
#   ./run.sh --host 0.0.0.0 --port 8086   # bind to the LAN (list the names used in CAMPAIGN_POC_ALLOWED_HOSTS)
#   ./run.sh --base-url https://<tunnel-name>   # Claude Web through a tunnel on this machine: the
#                                    # default bind is enough, but the server must know the name it
#                                    # is reached on (it answers only to known Host names). Also
#                                    # written into the --service unit, which inherits no shell.
set -euo pipefail
# The URL check below uses `[A-Za-z]`, which follows collation order — and so can admit
# non-ASCII letters — in some glibc UTF-8 locales. Everything this script prints is ASCII.
export LC_ALL=C
cd "$(dirname "$0")"

MODE=ondemand; STOP=0; HOST=127.0.0.1; PORT=8086; EMBED=ollama; SETUP=0
BASE_URL="${CAMPAIGN_POC_BASE_URL:-}"; BASE_URL_FROM=CAMPAIGN_POC_BASE_URL
while [ $# -gt 0 ]; do case "$1" in
  --setup) SETUP=1;;
  --service) MODE=service;;
  --stop) STOP=1;;
  --host) HOST="$2"; shift;;
  --port) PORT="$2"; shift;;
  --embed) EMBED="$2"; shift;;
  --base-url) case "${2-}" in ''|-*) [ $# -ge 2 ] && [ -z "${2-}" ] || {
                echo "--base-url needs a value, e.g. --base-url https://briefs.example.com"; exit 2; };; esac
              BASE_URL="$2"; BASE_URL_FROM=--base-url; shift;;
  *) echo "unknown arg: $1"; exit 2;;
esac; shift; done

# Written unquoted into a systemd unit, where a space splits it and `%` is a specifier: a
# value that is not a plain URL is refused here rather than registering a broken service.
# The SAME pattern as auth.BASE_URL_SHAPE (a test checks): what is safe to WRITE here. The
# server also applies DNS rules (label length, no empty or hyphen-edged label, no `..` in the
# path) that a pattern shared with bash and .NET cannot carry; for those, the server's startup
# WARNING has the last word: it names the value and ignores it.
# Not on --stop or --setup: neither uses the URL, and a bad one left in the shell must not
# stand between anybody and removing a service.
BASE_URL_SHAPE='^([Hh][Tt][Tt][Pp][Ss]?://)?([A-Za-z0-9._-]+|\[[0-9A-Fa-f:.]+\])(:[0-9]{1,5})?(/[A-Za-z0-9._~/-]*)?$'
# `--stop` alone is not a service removal — it falls through to a foreground start — so only
# `--service --stop` is exempt. `${BASH_REMATCH[3]:-}`: whether an unmatched group is unset
# under `set -u` is not something to depend on across bash versions.
if ! { [ "$MODE" = service ] && [ "$STOP" = 1 ]; } && [ "$SETUP" != 1 ] && [ -n "$BASE_URL" ]; then
  if ! [[ "$BASE_URL" =~ $BASE_URL_SHAPE ]] || { PORT_PART="${BASH_REMATCH[3]:-}"; [ -n "$PORT_PART" ] &&
       { [ "$((10#${PORT_PART#:}))" -lt 1 ] || [ "$((10#${PORT_PART#:}))" -gt 65535 ]; }; }; then
    echo "$BASE_URL_FROM must look like https://name[:port][/path] (got: $BASE_URL)"; exit 2
  fi
fi

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
${BASE_URL:+Environment=CAMPAIGN_POC_BASE_URL=$BASE_URL}
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
# Empty means none — including `--base-url ''` over a value inherited from the shell, which the
# check above did not look at and the server would otherwise still read.
if [ -n "$BASE_URL" ]; then export CAMPAIGN_POC_BASE_URL="$BASE_URL"; else unset CAMPAIGN_POC_BASE_URL; fi
echo "Serving http://$HOST:$PORT  (/mcp, /api, /upload, /openapi.json, /healthz)"
exec "$PY" -m http_app
