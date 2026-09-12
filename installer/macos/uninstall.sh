#!/usr/bin/env bash
# Remove Campaign Intelligence (lean) from a macOS machine.
#
# Leaves the DATA alone by default. The library is the thing the product exists to
# accumulate, and an uninstaller that silently deletes a year of campaign history because
# somebody wanted to reinstall is not a trade-off anyone agreed to. --purge is the explicit
# opt-in, and it says what it is about to remove before doing it.
set -euo pipefail

PURGE=0
for a in "$@"; do case "$a" in
  --purge) PURGE=1;;
  *) echo "unknown arg: $a"; exit 2;; esac; done

DEST="$HOME/Library/Application Support/CampaignIntelligence"
BIN="$HOME/.local/bin/campaign-intelligence"
DATA="$HOME/campaign-poc-data"

rm -f "$BIN"
rm -rf "$DEST"
echo "Removed the application."

if [ "$PURGE" = 1 ]; then
  if [ -d "$DATA" ]; then
    echo "About to delete the campaign library at $DATA:"
    du -sh "$DATA" 2>/dev/null || true
    printf "Type DELETE to confirm: "
    read -r answer
    if [ "$answer" = "DELETE" ]; then rm -rf "$DATA"; echo "Library deleted."
    else echo "Left the library in place."; fi
  fi
else
  [ -d "$DATA" ] && echo "Your campaign library is still at $DATA (use --purge to remove it)."
fi

echo "Claude Desktop's config still lists this server; remove the entry there, or re-install."
