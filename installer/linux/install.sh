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

# Verify the shipped CLIP weights actually survived the copy. An interrupted or disk-full
# cp leaves a truncated file that looks "present" to the app and only fails later, inside a
# tool call - the failure mode this whole payload exists to remove.
weights="$DEST/models/open_clip_model.safetensors"
if [ -f "$weights.sha256" ]; then
  echo "Verifying CLIP weights..."
  if command -v sha256sum >/dev/null 2>&1; then
    ( cd "$DEST/models" && sha256sum -c open_clip_model.safetensors.sha256 ) || {
      echo "FAILED: the CLIP weights are corrupt or incomplete. Visual search would be dead." >&2
      echo "Re-run this installer with a complete download." >&2; exit 1; }
  elif command -v shasum >/dev/null 2>&1; then
    ( cd "$DEST/models" && shasum -a 256 -c open_clip_model.safetensors.sha256 ) || {
      echo "FAILED: the CLIP weights are corrupt or incomplete. Visual search would be dead." >&2
      echo "Re-run this installer with a complete download." >&2; exit 1; }
  fi
elif [ ! -f "$weights" ]; then
  echo "WARNING: no CLIP weights in this bundle - visual similarity will be unavailable." >&2
fi
ln -sf "$DEST/campaign-intelligence" "$BIN/campaign-intelligence"

if [ "$WITH_OLLAMA" = 1 ]; then
  if command -v ollama >/dev/null 2>&1; then echo "Ollama present."; else
    echo "Installing Ollama..."; curl -fsSL https://ollama.com/install.sh | sh; fi
  (ollama serve >/dev/null 2>&1 &) || true; sleep 3
  # NOT a warning that the install then ignores: a failed pull leaves text search dead, and
  # the self-test below is what turns that into a refused install rather than a surprise
  # three days later (item 4.3).
  ollama pull nomic-embed-text || echo "Could not pull nomic-embed-text; the self-test will say so." >&2
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

# The gate (item 4.1). Everything above can succeed while the product is unusable - which
# is exactly what happened in the field, where visual search was dead on an installed copy
# and the only way to find out was a 60-second timeout inside a tool call. A non-zero exit
# here fails the install instead of reporting success over it.
echo
echo "Running post-install self-test..."
if ! "$DEST/campaign-intelligence" health-check; then
  echo >&2
  echo "INSTALL FAILED: the self-test above names the component that is not working." >&2
  echo "The files are in $DEST; fix what it names and re-run:" >&2
  echo "  \"$DEST/campaign-intelligence\" health-check" >&2
  exit 1
fi

echo
echo "Installed. 'campaign-intelligence' is in ~/.local/bin (ensure it's on your PATH)."
echo "Claude Desktop is wired; fully quit and reopen it - closing the window is not enough."
echo "Uninstall: ./uninstall.sh"
