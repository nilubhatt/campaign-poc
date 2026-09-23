#!/usr/bin/env bash
# Uninstall the per-user Campaign Intelligence (lean) install. Leaves your data
# (~/campaign-poc-data) untouched unless you pass --purge.
set -euo pipefail
PURGE=0; [ "${1:-}" = "--purge" ] && PURGE=1

systemctl --user disable --now campaign-intelligence 2>/dev/null || true
rm -f "$HOME/.config/systemd/user/campaign-intelligence.service"
systemctl --user daemon-reload 2>/dev/null || true

rm -f  "$HOME/.local/bin/campaign-intelligence"
rm -rf "$HOME/.local/share/campaign-intelligence"

# remove our Claude Desktop connector entry (leave the rest of the config intact)
CFG="$HOME/.config/Claude/claude_desktop_config.json"
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
fi

# D28: what is about to go, named, and a typed confirmation — the same bar the macOS
# uninstaller already held. This deleted a year of campaign history on a bare `--purge` with
# nothing shown and nothing asked, which is the trade-off that script's own comment says
# nobody agreed to. The rulebook is named separately because it is the one file in there the
# customer WROTE rather than accumulated: a library of campaigns can be re-uploaded from the
# decks, and a set of rules somebody sat down and agreed cannot be.
DATA="$HOME/campaign-poc-data"
if [ "$PURGE" = 1 ] && [ -d "$DATA" ]; then
  echo "About to delete the campaign library at $DATA:"
  du -sh "$DATA" 2>/dev/null || true
  [ -f "$DATA/rulebook.yaml" ] && echo "  ...including your rulebook ($DATA/rulebook.yaml), which is not rebuildable from your decks. Copy it out first if you want to keep it."
  printf "Type DELETE to confirm: "
  read -r answer
  if [ "$answer" = "DELETE" ]; then rm -rf "$DATA"; echo "Library deleted."
  else echo "Left the library in place."; PURGE=0; fi
elif [ -d "$DATA" ]; then
  echo "Your campaign library is still at $DATA (use --purge to remove it)."
  [ -f "$DATA/rulebook.yaml" ] && echo "Your rulebook is still at $DATA/rulebook.yaml."
fi
echo "Uninstalled.$([ "$PURGE" = 1 ] && echo ' Data purged.')"
