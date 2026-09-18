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
    # D28: the rulebook, named separately. It is the one file in there the customer WROTE
    # rather than accumulated — a library of campaigns can be re-uploaded from the decks, and
    # a set of rules somebody sat down and agreed cannot be.
    [ -f "$DATA/rulebook.yaml" ] && echo "  ...including your rulebook ($DATA/rulebook.yaml), which is not rebuildable from your decks. Copy it out first if you want to keep it."
    printf "Type DELETE to confirm: "
    read -r answer
    if [ "$answer" = "DELETE" ]; then rm -rf "$DATA"; echo "Library deleted."
    else echo "Left the library in place."; fi
  fi
else
  [ -d "$DATA" ] && echo "Your campaign library is still at $DATA (use --purge to remove it)."
  [ -f "$DATA/rulebook.yaml" ] && echo "Your rulebook is still at $DATA/rulebook.yaml."
fi

CFG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
if [ -f "$CFG" ] && command -v python3 >/dev/null 2>&1; then
  python3 - "$CFG" <<'PY' || true
# encoding="utf-8" explicitly: Python's default is the locale codec, and under a legacy
# LC_ALL this rewrites every other connector's non-ASCII path (the same trap as defect 11).
import json,sys
p=sys.argv[1]
try: d=json.load(open(p,encoding="utf-8"))
except Exception: sys.exit(0)
d.get("mcpServers",{}).pop("campaign-intelligence",None)
json.dump(d,open(p,"w",encoding="utf-8"),indent=2,ensure_ascii=False)
PY
  echo "Removed the Claude Desktop connector entry."
fi
