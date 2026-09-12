# PyInstaller spec — one-folder bundle for the campaign-intelligence lean product.
# Build (on the TARGET OS): pyinstaller campaign-poc.spec
# Produces dist/campaign-intelligence/  (a self-contained folder; zip/tar it to distribute).
#
# The load-bearing parts:
#   * sqlite_vec ships a native extension (.so/.dylib/.dll) that must be bundled AND
#     resolvable at runtime, or semantic search silently drops to the pure-Python fallback.
#   * uvicorn / mcp / starlette / anyio pull in submodules PyInstaller's static analysis misses.
#   * torch/open_clip (§6.6 CLIP layer) — a deliberate, confirmed size tradeoff (~150-250MB
#     of deps + a 605MB checkpoint shipped in the installer payload, see
#     docs/PRODUCTION-ROADMAP.md §6.6 and docs/PRODUCT-REVIEW-PLAN.md item 1.2).
#     NOT excluded: an earlier version of this spec excluded torch to keep the bundle small,
#     predating CLIP being an actual dependency — that would have silently broken
#     find_similar_images/upload_image_asset's CLIP path in every packaged build (review
#     caught this). collect_all is used for torch/open_clip/timm like sqlite_vec, since they
#     ship compiled binaries and data files PyInstaller's static analysis misses.
#     FIXED (2026-09-08), root cause confirmed by direct inspection rather than guessed:
#     torchvision 0.29.0 loads its compiled ops extension by an explicit runtime path
#     (`torch.ops.load_library(...)`, in torchvision/extension.py), not a Python `import` —
#     so PyInstaller's import-graph analysis never sees it, and neither collect_all's
#     collect_dynamic_libs (only picks up torchvision/.dylibs/*, its vendored transitive
#     deps) nor collect_data_files (excludes binary-looking files) picks up the extension
#     itself (`_C_stable.so`, `image_stable.so`, sitting directly in the `torchvision/`
#     package dir). The bundled `_pyinstaller_hooks_contrib` hook for torchvision is stale —
#     it declares `hiddenimports = ['torchvision._C']`, the pre-0.29 name, which is a no-op
#     against a name that no longer exists (hence the build's "Hidden import torchvision._C
#     not found!" warning) and, being a hiddenimport not a binary collection, wouldn't have
#     copied the file even if the name were right. Fixed below by explicitly globbing
#     torchvision's own top-level *.so files into `binaries` at the exact same relative path
#     torchvision's own extension-path lookup expects (`os.path.dirname(__file__)`, i.e.
#     right alongside `torchvision/__init__.py`) — confirmed by an actual built-and-run
#     bundle: `/healthz` responds and find_similar_images produces a real CLIP embedding.

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = [], [], []

# Bundle sqlite-vec fully (its compiled extension is a binary + package data).
for pkg in ("sqlite_vec",):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

# CLIP stack — compiled binaries (torch) + model-config/data files (open_clip, timm).
for pkg in ("torch", "open_clip", "timm"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

# torchvision's own extension modules (_C_stable.*, image_stable.*) are dlopen'd by
# explicit path at runtime, never `import`ed — collect_all/collect_dynamic_libs both miss
# them (see the long comment above). Glob them directly into the same relative path
# torchvision's own lookup expects: right next to torchvision/__init__.py. .so on
# macOS/Linux, .pyd on Windows — only macOS was actually rebuilt-and-run to verify this
# session; Linux/Windows are the same fix by inspection but not yet independently verified.
import torchvision
_tv_dir = Path(torchvision.__file__).parent
for pattern in ("*.so", "*.pyd"):
    for ext_file in _tv_dir.glob(pattern):
        binaries.append((str(ext_file), "torchvision"))

# Server stack — collect submodules PyInstaller commonly under-detects.
for pkg in ("uvicorn", "mcp", "starlette", "anyio", "fastapi", "pptx", "pypdf"):
    hiddenimports += collect_submodules(pkg)

# Our own modules referenced only via dynamic import (main.py imports them lazily).
hiddenimports += ["http_app", "mcp_server", "store", "core", "vectorstore",
                  "embedding", "extract", "auth", "config", "clip_embed", "images"]

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "mcp.cli", "typer"],   # CLI unused; keep the bundle small
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
