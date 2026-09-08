# PyInstaller spec — one-folder bundle for the campaign-intelligence lean product.
# Build (on the TARGET OS): pyinstaller campaign-poc.spec
# Produces dist/campaign-intelligence/  (a self-contained folder; zip/tar it to distribute).
#
# The load-bearing parts:
#   * sqlite_vec ships a native extension (.so/.dylib/.dll) that must be bundled AND
#     resolvable at runtime, or semantic search silently drops to the pure-Python fallback.
#   * uvicorn / mcp / starlette / anyio pull in submodules PyInstaller's static analysis misses.
#   * torch/open_clip (§6.6 CLIP layer) — a deliberate, confirmed size tradeoff (~150-250MB
#     of deps + a ~350MB model download on first use, see docs/PRODUCTION-ROADMAP.md §6.6).
#     NOT excluded: an earlier version of this spec excluded torch to keep the bundle small,
#     predating CLIP being an actual dependency — that would have silently broken
#     find_similar_images/upload_image_asset's CLIP path in every packaged build (review
#     caught this). collect_all is used for torch/open_clip/timm like sqlite_vec, since they
#     ship compiled binaries and data files PyInstaller's static analysis misses.
#     VERIFIED BROKEN (2026-09-08): the build succeeds (717MB bundle, torch/open_clip/timm
#     genuinely present under dist/.../_internal/) but the packaged binary CRASHES on
#     startup: `RuntimeError: operator torchvision::nms does not exist`. Root cause:
#     torchvision 0.29.0 ships its compiled extension as `_C_stable.so` (a newer ABI-stable
#     naming scheme); PyInstaller 6.22.2's bundling hooks don't recognize that name (build
#     log: "Hidden import torchvision._C not found!"), so the extension that registers the
#     `nms` custom op isn't wired up in the frozen build, and open_clip's package init
#     imports coca_model.py -> torchvision.ops -> triggers that registration unconditionally.
#     This is a packaging-only bug — running from source (python -m http_app / run.sh /
#     run.ps1) has CLIP fully working, verified extensively against the real model earlier
#     this session. NOT fixed here — needs dedicated packaging work: pin an older
#     torchvision known to work with PyInstaller's hooks, or a newer PyInstaller with an
#     updated torchvision hook, or a custom hook mapping `_C_stable.so`. Until one of those
#     lands, the shipped bundle's find_similar_images/upload_image_asset's CLIP step will
#     not work — pHash (check_image_provenance) is unaffected (no torch dependency).

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
