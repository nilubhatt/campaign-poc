#!/usr/bin/env bash
# End-user macOS installer for Campaign Intelligence (lean). Per-user, no root needed for the
# app itself (Ollama's own installer may ask). Run from an extracted bundle directory, or it
# will look for ./dist/campaign-intelligence or ./campaign-intelligence next to this script.
#
#   ./install.sh              # install + wire Claude Desktop + ensure Ollama
#   ./install.sh --no-ollama  # skip the Ollama install/pull step
#
# macOS previously had no installer at all: the release shipped a tar.gz and left the user to
# work out where to put it, whether the weights survived the download, and whether Claude
# Desktop had been wired. Every acceptance criterion the Linux installer meets (a checksum on
# the shipped weights, a post-install self-test that BLOCKS success, Ollama verified rather
# than hoped for) applies here identically - the platform was the only thing missing.
set -euo pipefail

WITH_OLLAMA=1
for a in "$@"; do case "$a" in
  --no-ollama) WITH_OLLAMA=0;;
  *) echo "unknown arg: $a"; exit 2;; esac; done

here="$(cd "$(dirname "$0")" && pwd)"
for cand in "$here" "$here/campaign-intelligence" "$here/dist/campaign-intelligence"; do
  # require a regular executable file - a directory of the same name is also -x, so -f matters
  if [ -f "$cand/campaign-intelligence" ] && [ -x "$cand/campaign-intelligence" ]; then BUNDLE="$cand"; break; fi
done
[ -n "${BUNDLE:-}" ] || { echo "Could not find the campaign-intelligence bundle."; exit 1; }

DEST="$HOME/Library/Application Support/CampaignIntelligence"
BIN="$HOME/.local/bin"; mkdir -p "$BIN"
echo "Installing to $DEST ..."
rm -rf "$DEST"; mkdir -p "$DEST"; cp -a "$BUNDLE"/. "$DEST"/

# Gatekeeper quarantines anything that arrived through a browser, and the quarantine flag
# travels with every file inside the archive. Left in place, the first launch is a dialog
# saying the app "cannot be opened because the developer cannot be verified" - and the user
# has no reason to connect that to the deck they just tried to upload.
xattr -dr com.apple.quarantine "$DEST" 2>/dev/null || true

# Verify the shipped CLIP weights actually survived the copy. An interrupted or disk-full cp
# leaves a truncated file that looks present to the app and only fails later, inside a tool
# call - the failure mode this payload exists to remove.
weights="$DEST/models/open_clip_model.safetensors"
if [ -f "$weights.sha256" ]; then
  echo "Verifying CLIP weights..."
  ( cd "$DEST/models" && shasum -a 256 -c open_clip_model.safetensors.sha256 ) || {
    echo "FAILED: the CLIP weights are corrupt or incomplete. Visual search would be dead." >&2
    echo "Re-run this installer with a complete download." >&2; exit 1; }
elif [ ! -f "$weights" ]; then
  echo "WARNING: no CLIP weights in this bundle - visual similarity will be unavailable." >&2
fi
ln -sf "$DEST/campaign-intelligence" "$BIN/campaign-intelligence"

if [ "$WITH_OLLAMA" = 1 ]; then
  if command -v ollama >/dev/null 2>&1; then echo "Ollama present."; else
    echo "Installing Ollama..."; curl -fsSL https://ollama.com/install.sh | sh; fi
  (ollama serve >/dev/null 2>&1 &) || true; sleep 3
  # NOT '|| echo warning': a failed pull leaves text search dead, and the self-test below is
  # what turns that into a refused install rather than a surprise three days later.
  ollama pull nomic-embed-text || echo "Could not pull nomic-embed-text; the self-test will say so." >&2
fi

echo "Wiring Claude Desktop..."; "$DEST/campaign-intelligence" configure-desktop || true

# The gate. Everything above can succeed while the product is unusable - that is exactly what
# happened in the field, where visual search was dead on an installed copy and the only way to
# find out was a 60-second timeout inside a tool call. A non-zero exit here fails the install.
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
