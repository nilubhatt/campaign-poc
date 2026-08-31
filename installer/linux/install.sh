#!/usr/bin/env bash
# End-user Linux installer for Campaign Intelligence (lean). Per-user, no root needed for the
# app itself (Ollama's own installer may use sudo). Run from an extracted bundle directory, or
# it will look for ./dist/campaign-intelligence or ./campaign-intelligence next to this script.
#
#   ./install.sh              # install + wire Claude Desktop + ensure Ollama
#   ./install.sh --service    # also run the server in the background (systemd --user)
#   ./install.sh --no-ollama  # skip the Ollama install/pull step
set -euo pipefail

WITH_SERVICE=0; WITH_OLLAMA=1
for a in "$@"; do case "$a" in
  --service) WITH_SERVICE=1;; --no-ollama) WITH_OLLAMA=0;;
  *) echo "unknown arg: $a"; exit 2;; esac; done

here="$(cd "$(dirname "$0")" && pwd)"
# locate the bundle folder (contains the 'campaign-intelligence' executable)
for cand in "$here" "$here/campaign-intelligence" "$here/dist/campaign-intelligence"; do
  # require a regular executable file — a directory named the same is also -x, so -f matters
  if [ -f "$cand/campaign-intelligence" ] && [ -x "$cand/campaign-intelligence" ]; then BUNDLE="$cand"; break; fi
done
[ -n "${BUNDLE:-}" ] || { echo "Could not find the campaign-intelligence bundle."; exit 1; }

DEST="$HOME/.local/share/campaign-intelligence"
BIN="$HOME/.local/bin"; mkdir -p "$BIN"
echo "Installing to $DEST ..."
rm -rf "$DEST"; mkdir -p "$DEST"; cp -a "$BUNDLE"/. "$DEST"/
ln -sf "$DEST/campaign-intelligence" "$BIN/campaign-intelligence"

if [ "$WITH_OLLAMA" = 1 ]; then
  if command -v ollama >/dev/null 2>&1; then echo "Ollama present."; else
    echo "Installing Ollama..."; curl -fsSL https://ollama.com/install.sh | sh; fi
  (ollama serve >/dev/null 2>&1 &) || true; sleep 3
  ollama pull nomic-embed-text || echo "warning: model pull failed; run 'ollama pull nomic-embed-text' later"
fi

echo "Wiring Claude Desktop..."; "$DEST/campaign-intelligence" configure-desktop || true

if [ "$WITH_SERVICE" = 1 ]; then
  UNIT="$HOME/.config/systemd/user/campaign-intelligence.service"; mkdir -p "$(dirname "$UNIT")"
  cat > "$UNIT" <<EOF
[Unit]
Description=Campaign Intelligence (lean) server
After=network.target
[Service]
ExecStart=$DEST/campaign-intelligence serve
Restart=on-failure
[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload && systemctl --user enable --now campaign-intelligence
  echo "Background service started (systemctl --user status campaign-intelligence)."
fi

echo
echo "Installed. 'campaign-intelligence' is in ~/.local/bin (ensure it's on your PATH)."
echo "Claude Desktop is wired; fully quit and reopen it. Uninstall: ./uninstall.sh"
