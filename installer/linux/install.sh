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
# Staged, not swapped in place. `rm -rf "$DEST"` ran before anything had been checked, so
# a failed UPGRADE left the machine with no working version at all - and the failure
# message's "fix it and run this installer again" was advice about a copy the gate had
# just condemned. Everything below happens in $STAGE; the live directory is only
# replaced once the self-test has passed.
STAGE="$DEST.incoming"
echo "Installing to $DEST ..."
rm -rf "$STAGE"; mkdir -p "$STAGE"; cp -a "$BUNDLE"/. "$STAGE"/

# Verify the shipped CLIP weights actually survived the copy. An interrupted or disk-full
# cp leaves a truncated file that looks "present" to the app and only fails later, inside a
# tool call - the failure mode this whole payload exists to remove.
weights="$STAGE/models/open_clip_model.safetensors"
if [ -f "$weights.sha256" ]; then
  echo "Verifying CLIP weights..."
  if command -v sha256sum >/dev/null 2>&1; then
    ( cd "$STAGE/models" && sha256sum -c open_clip_model.safetensors.sha256 ) || {
      echo "FAILED: the CLIP weights are corrupt or incomplete. Visual search would be dead." >&2
      echo "Re-run this installer with a complete download." >&2; exit 1; }
  elif command -v shasum >/dev/null 2>&1; then
    ( cd "$STAGE/models" && shasum -a 256 -c open_clip_model.safetensors.sha256 ) || {
      echo "FAILED: the CLIP weights are corrupt or incomplete. Visual search would be dead." >&2
      echo "Re-run this installer with a complete download." >&2; exit 1; }
  fi
elif [ ! -f "$weights" ]; then
  echo "WARNING: no CLIP weights in this bundle - visual similarity will be unavailable." >&2
fi


if [ "$WITH_OLLAMA" = 1 ]; then
  if command -v ollama >/dev/null 2>&1; then echo "Ollama present."; else
    echo "Installing Ollama..."; curl -fsSL https://ollama.com/install.sh | sh; fi
  (ollama serve >/dev/null 2>&1 &) || true; sleep 3
  # NOT a warning that the install then ignores: a failed pull leaves text search dead, and
  # the self-test below is what turns that into a refused install rather than a surprise
  # three days later (item 4.3).
  ollama pull nomic-embed-text || echo "Could not pull nomic-embed-text; the self-test will say so." >&2
fi


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
  # `a && b` inside `set -e` does not abort when `a` fails, so a host with no systemd
  # printed "command not found" and then "Background service started" and "Installed."
  if ! command -v systemctl >/dev/null 2>&1; then
    echo "WARNING: no systemctl on this host, so --service did nothing. Start the server" >&2
    echo "         yourself with: $DEST/campaign-intelligence serve" >&2
  elif systemctl --user daemon-reload && systemctl --user enable --now campaign-intelligence; then
    echo "Background service started (systemctl --user status campaign-intelligence)."
  else
    echo "WARNING: the background service could not be started; the product is installed" >&2
    echo "         and usable, but will not run on its own." >&2
  fi
fi

# Create the data directory and database before the gate: the self-test opens the database
# read-only by design, so without this every fresh install failed its own check with
# "FAIL database" and told the marketer to go and start a server.
echo "Preparing the data directory..."
"$STAGE/campaign-intelligence" init || {
  echo "FAILED: could not create the data directory. Check permissions and disk space." >&2
  exit 1; }

# The gate (item 4.1). Everything above can succeed while the product is unusable - which is
# exactly what happened in the field, where visual search was dead on an installed copy and
# the only way to find out was a 60-second timeout inside a tool call.
#
# It runs BEFORE Claude Desktop is wired, deliberately. Wiring first meant a refused install
# still left the marketer's next session pointing at the server the installer had just
# condemned - the review's own defect, with the installer's signature on it.
echo
echo "Running post-install self-test..."
if ! "$STAGE/campaign-intelligence" health-check --wait 60; then
  echo >&2
  echo "INSTALL FAILED: the self-test above names the component that is not working." >&2
  echo "Claude Desktop has NOT been connected, so nothing will try to use this yet." >&2
  echo "Your previous install, if any, is untouched and still working." >&2
  echo "The new files are in $STAGE. Fix what is named above and run this installer again." >&2
  exit 1
fi

# Only now is the live directory replaced: everything above ran against the staged copy.
rm -rf "$DEST"; mv "$STAGE" "$DEST"
ln -sf "$DEST/campaign-intelligence" "$BIN/campaign-intelligence"

echo "Connecting Claude Desktop..."; "$DEST/campaign-intelligence" configure-desktop || true

echo
echo "Installed. 'campaign-intelligence' is in ~/.local/bin (ensure it's on your PATH)."
echo "Claude Desktop is wired; fully quit and reopen it - closing the window is not enough."
echo "Uninstall: ./uninstall.sh"
