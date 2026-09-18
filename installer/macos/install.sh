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
# No symlink into ~/.local/bin: that is a Linux convention and is not on a default macOS
# PATH, so "ensure it's on your PATH" was advice a marketer could not act on. Claude Desktop
# launches the binary by absolute path (see configure-desktop), so nothing needs it - and the
# absolute command is printed at the end for anyone who wants a terminal.
# Staged, not swapped in place. `rm -rf "$DEST"` ran before anything had been checked, so
# a failed UPGRADE left the machine with no working version at all - and the failure
# message's "fix it and run this installer again" was advice about a copy the gate had
# just condemned. Everything below happens in $STAGE; the live directory is only
# replaced once the self-test has passed.
STAGE="$DEST.incoming"
echo "Installing to $DEST ..."
rm -rf "$STAGE"; mkdir -p "$STAGE"; cp -a "$BUNDLE"/. "$STAGE"/

# Gatekeeper quarantines anything that arrived through a browser. The flag travels to every
# file inside the archive when FINDER expands it - not when `tar xzf` does, which is the path
# LINUX.md documents, and which is why nobody ever reported the dialog this comment predicts.
# It is still worth one line: double-clicking the tarball in Finder is what a marketer does,
# and there the first launch is "cannot be opened because the developer cannot be verified",
# with no reason to connect that to the deck they just tried to upload.
# On the .pkg path this is inert - a pkg payload is never quarantined - and the dialog a
# marketer meets there is on the package itself, which no strip can reach. That is 4.4/D130.
#
# $STAGE, not $DEST (§4.4). This stripped the LIVE directory, and everything the installer
# runs - `init`, the health-check gate - runs out of $STAGE, which is also what `mv` puts in
# place at the end. On a fresh install $DEST does not exist, so the strip was a no-op that
# `|| true` swallowed; on an upgrade it stripped the copy about to be deleted. Measured: the
# installed binary and the CLIP weights beside it were still quarantined after a successful
# install, on both paths. The plan calls this "a legitimate stopgap inside an installer the
# user chose to run" - it is that only if it happens.
xattr -dr com.apple.quarantine "$STAGE" 2>/dev/null || true

# Verify the shipped CLIP weights actually survived the copy. An interrupted or disk-full cp
# leaves a truncated file that looks present to the app and only fails later, inside a tool
# call - the failure mode this payload exists to remove.
weights="$STAGE/models/open_clip_model.safetensors"
if [ -f "$weights.sha256" ]; then
  echo "Verifying CLIP weights..."
  ( cd "$STAGE/models" && shasum -a 256 -c open_clip_model.safetensors.sha256 ) || {
    echo "FAILED: the CLIP weights are corrupt or incomplete. Visual search would be dead." >&2
    echo "Re-run this installer with a complete download." >&2; exit 1; }
elif [ ! -f "$weights" ]; then
  echo "WARNING: no CLIP weights in this bundle - visual similarity will be unavailable." >&2
fi


if [ "$WITH_OLLAMA" = 1 ]; then
  if command -v ollama >/dev/null 2>&1; then echo "Ollama present."; else
    # NOT bare: under `set -euo pipefail` a failed install ends the script here, before the
    # gate, and the marketer gets a bare non-zero with no component named. Ollama's installer
    # writes /Applications and symlinks /usr/local/bin, neither of which a standard account
    # can do without a prompt - and under the .pkg there is no terminal to prompt into. Let it
    # fail and let the self-test say what that cost, which is item 4.3's whole argument.
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh \
      || echo "Could not install Ollama; the self-test will say what that means." >&2; fi
  (ollama serve >/dev/null 2>&1 &) || true; sleep 3
  # NOT '|| echo warning': a failed pull leaves text search dead, and the self-test below is
  # what turns that into a refused install rather than a surprise three days later.
  ollama pull nomic-embed-text || echo "Could not pull nomic-embed-text; the self-test will say so." >&2
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

echo "Connecting Claude Desktop..."; "$DEST/campaign-intelligence" configure-desktop || true

echo
echo "Installed."
echo "To run it from a terminal: \"$DEST/campaign-intelligence\""
echo "Claude Desktop is wired; fully quit and reopen it - closing the window is not enough."
# The absolute path: after a .pkg install there is no such working directory, and the
# relative form sent the reader looking for a file that is not where they are standing.
echo "Uninstall: \"$DEST/uninstall.sh\""
