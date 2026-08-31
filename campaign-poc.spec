# PyInstaller spec — one-folder bundle for the campaign-intelligence lean product.
# Build (on the TARGET OS): pyinstaller campaign-poc.spec
# Produces dist/campaign-intelligence/  (a self-contained folder; zip/tar it to distribute).
#
# The load-bearing parts:
#   * sqlite_vec ships a native extension (.so/.dylib/.dll) that must be bundled AND
#     resolvable at runtime, or semantic search silently drops to the pure-Python fallback.
#   * uvicorn / mcp / starlette / anyio pull in submodules PyInstaller's static analysis misses.

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = [], [], []

# Bundle sqlite-vec fully (its compiled extension is a binary + package data).
for pkg in ("sqlite_vec",):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

# Server stack — collect submodules PyInstaller commonly under-detects.
for pkg in ("uvicorn", "mcp", "starlette", "anyio", "fastapi", "pptx", "pypdf"):
    hiddenimports += collect_submodules(pkg)

# Our own modules referenced only via dynamic import (main.py imports them lazily).
hiddenimports += ["http_app", "mcp_server", "store", "core", "vectorstore",
                  "embedding", "extract", "auth", "config"]

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "tkinter", "mcp.cli", "typer"],   # CLI unused; keep the bundle small
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="campaign-intelligence",
    console=True,          # a server/CLI — needs a console
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name="campaign-intelligence",
)
