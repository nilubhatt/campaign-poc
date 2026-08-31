#!/usr/bin/env bash
# Build a self-contained bundle for the CURRENT OS (Linux or macOS).
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

os=$(uname -s | tr '[:upper:]' '[:lower:]'); arch=$(uname -m)
out="campaign-intelligence-${os}-${arch}.tar.gz"
tar -C dist -czf "$out" campaign-intelligence
echo "Bundle: $out  ($(du -sh "$out" | cut -f1))"
echo "Run:    dist/campaign-intelligence/campaign-intelligence serve   (or 'stdio')"
