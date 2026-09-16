#!/usr/bin/env bash
# Build a self-contained bundle for the CURRENT OS (Linux or macOS).
# Stamp the build so two local builds are not indistinguishable (§3.2, defect 10). CI writes
# the same file; without it every locally built binary reports itself identically, and calls
# itself a "source checkout" while being a frozen app.
git rev-parse --short HEAD > /dev/null 2>&1 \
  && echo "$(git rev-parse --short HEAD) $(date -u +%Y-%m-%dT%H:%M:%SZ)" > build_info.txt \
  || echo "local $(date -u +%Y-%m-%dT%H:%M:%SZ)" > build_info.txt

# PyInstaller is native — run this ON the OS you want to target (a Linux binary must be
# built on Linux). For Windows, use build.ps1 on Windows. Output: dist/campaign-intelligence/
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv-build
# shellcheck disable=SC1091
. .venv-build/bin/activate
pip install --upgrade pip
pip install -r requirements.txt pyinstaller typer
pyinstaller --clean --noconfirm campaign-poc.spec

# The CLIP weights ship inside the bundle so an air-gapped install works untouched. ~303MB (fp16),
# verified by SHA-256. Fetched into .weights-cache/ first because PyInstaller wipes dist/
# on every run - fetching straight into it would re-download on every local build.
python3 scripts/fetch_weights.py .weights-cache
mkdir -p dist/campaign-intelligence/models
cp .weights-cache/open_clip_model.safetensors* dist/campaign-intelligence/models/

# §12.1: the rulebook, BESIDE the executable rather than only inside _internal/.
# PyInstaller 6 puts every `datas` entry under the contents directory whatever relative
# destination the spec names, and `config.app_dir()` deliberately resolves to the install
# directory and not `sys._MEIPASS` — because an administrator has to be able to SEE and EDIT
# this file, which is most of what the item is about. Without this copy every frozen install
# read the buried one: edits to the visible file would do nothing, silently.
cp rulebook.yaml dist/campaign-intelligence/rulebook.yaml

os=$(uname -s | tr '[:upper:]' '[:lower:]'); arch=$(uname -m)
out="campaign-intelligence-${os}-${arch}.tar.gz"
tar -C dist -czf "$out" campaign-intelligence
echo "Bundle: $out  ($(du -sh "$out" | cut -f1))"
echo "Run:    dist/campaign-intelligence/campaign-intelligence serve   (or 'stdio')"
