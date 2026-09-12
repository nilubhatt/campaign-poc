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

[ "$PURGE" = 1 ] && rm -rf "$HOME/campaign-poc-data"
echo "Uninstalled.$([ "$PURGE" = 1 ] && echo ' Data purged.')"
