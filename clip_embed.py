"""
CLIP visual embeddings — the heavier follow-on to images.py's pHash (§6.6). Catches
aesthetic/regional similarity: same product different photo, same styling, "looks like the
APAC shoot" — not exact/near-duplicate reuse (that's pHash's job). Needs torch + a model
download (hundreds of MB), a deliberate dependency decision (see docs/PRODUCTION-ROADMAP.md
§6.6) — so torch/open_clip are imported lazily, only when the real provider is actually used.

The model loads once per process (global cache) since loading takes real time (~seconds to
tens of seconds) — repeated per-call loading would make ingesting several images painfully
slow.
"""
from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import config

_model = None
_preprocess = None
_model_lock = threading.Lock()
_resolution = None   # WeightsResolution recorded at warm_up()
_load_error = None   # why the last load attempt failed, if it did


def embed_image(path: Path) -> list[float]:
    """Embed a single image into CLIP's visual space with the configured provider."""
    if config.CLIP_PROVIDER == "openclip":
        return _embed_openclip(path)
    if config.CLIP_PROVIDER == "hash":
        return _embed_hash(path)
    raise ValueError(f"unknown CLIP provider {config.CLIP_PROVIDER!r}")


def _load_model():
    """Loads once, guarded by a lock — MCP dispatches sync tools onto worker threads, so
    concurrent first calls could otherwise race into loading the model multiple times at
    once (transiently multiplying memory use and load time for no benefit; review found
    this with 3 concurrent calls each loading their own copy)."""
    global _model, _preprocess, _load_error
    if _model is None:
        # Resolve BEFORE importing open_clip: a config typo should not first pay the
        # multi-second, hundreds-of-MB torch/open_clip import to be told about itself.
        resolution = weights_status()
        if not resolution.ok:
            raise RuntimeError(f"{resolution.reason} {resolution.remedy}")
        with _model_lock:
            if _model is None:  # re-check: another thread may have finished while we waited
                import open_clip
                try:
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        config.CLIP_MODEL_NAME, pretrained=resolution.pretrained
                    )
                except Exception as exc:
                    # Remembered so weights_status() stops claiming the weights are fine
                    # when the model never loaded. Still raised: whoever asked for an
                    # embedding needs to know it did not happen.
                    _load_error = str(exc)
                    raise
                model.eval()
                _model, _preprocess = model, preprocess
                _load_error = None  # a later retry succeeded; stop reporting the old failure
    return _model, _preprocess


# open_clip 3.3.0 publishes and prefers safetensors; the .bin is the older name an admin is
# more likely to already have. Both load from a path (verified against the real library).
_CHECKPOINT_NAMES = ("open_clip_model.safetensors", "open_clip_pytorch_model.bin")

_LOCAL_WEIGHTS_REMEDY = (
    f"Point {{var}} at a CLIP weights file ({' or '.join(_CHECKPOINT_NAMES)}) or the folder "
    f"holding it, or clear {{var}} and reinstall to use the copy shipped with the product."
)


@dataclass(frozen=True)
class WeightsResolution:
    """What the vision weights resolved to, and — when they didn't — what to do about it.

    Deliberately a value, not an exception: resolution is cheap and happens at startup,
    where a vision-weights problem must NOT take down text search, upload or evaluation,
    none of which need CLIP. health_check (plan 2.3) reports this; only an actual attempt
    to embed an image turns a bad resolution into a raised error."""
    ok: bool
    source: str          # "env" | "bundled" | "tag" | "missing" | "none"
    pretrained: str = ""  # what gets handed to open_clip
    path: str = ""        # the resolved checkpoint file, when local
    reason: str = ""
    remedy: str = ""


def resolve_weights() -> WeightsResolution:
    """Decide what to load from, without loading anything and without raising.

    A configured-but-unusable path is NOT silently replaced by the network tag: setting the
    variable is an admin declaring "use this file", and quietly reaching for the Hub instead
    is the exact failure the setting exists to prevent — on the network where this was first
    hit, that fetch cannot succeed at all."""
    if config.CLIP_PROVIDER != "openclip":
        return WeightsResolution(ok=True, source="none")

    configured = (config.CLIP_WEIGHTS_PATH or "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        checkpoint = _find_checkpoint(candidate)
        if checkpoint is None:
            # Deliberately not falling through to the bundled copy: that would hide the
            # admin's typo behind something that happens to work, and they would never
            # learn the path they set is wrong.
            where = f"the folder {candidate}" if candidate.is_dir() else str(candidate)
            return WeightsResolution(
                ok=False, source="env",
                reason=f"Visual search is off: {config.CLIP_WEIGHTS_ENV_VAR} points at "
                       f"{where}, which has no usable CLIP weights file.",
                remedy=_LOCAL_WEIGHTS_REMEDY.format(var=config.CLIP_WEIGHTS_ENV_VAR),
            )
        return WeightsResolution(ok=True, source="env", pretrained=str(checkpoint),
                                 path=str(checkpoint))

    # Shipped with the installer, beside the executable — the path that has to work on a
    # clean, air-gapped machine with nothing configured at all.
    bundled = _find_checkpoint(config.BUNDLED_WEIGHTS_DIR)
    if bundled is not None:
        return WeightsResolution(ok=True, source="bundled", pretrained=str(bundled),
                                 path=str(bundled))

    if config.is_installed():
        # An installed copy ships its weights, so their absence means something removed
        # them — a partial copy, an endpoint filter quarantining a 605MB opaque binary, an
        # admin reclaiming disk. Falling back to the tag here would silently recreate the
        # original defect on the one machine that cannot reach the Hub at all, and would
        # report itself as healthy while doing it.
        return WeightsResolution(
            ok=False, source="missing",
            reason=f"Visual search is off: this installation has no CLIP weights at "
                   f"{config.BUNDLED_WEIGHTS_DIR}, where they ship.",
            remedy=f"Reinstall to restore them, or set {config.CLIP_WEIGHTS_ENV_VAR} to a "
                   f"copy of the weights file.",
        )

    # Source checkout: no bundled copy ever existed and Hub access is the normal developer
    # path. `python scripts/fetch_weights.py models` opts into the offline behaviour.
    return WeightsResolution(ok=True, source="tag", pretrained=config.CLIP_PRETRAINED)


def _find_checkpoint(candidate: Path) -> Path | None:
    """Accept either the checkpoint file itself or a folder holding it — an admin is as
    likely to point at one as the other, and the bundled-weights payload (plan 1.2) is a
    folder, so both share this one resolver."""
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        for name in _CHECKPOINT_NAMES:
            if (candidate / name).is_file():
                return candidate / name
    return None


def weights_status() -> WeightsResolution:
    """What to report about the vision weights right now.

    Two different failures, handled differently. A RESOLUTION failure (the configured path
    has no checkpoint) is re-checked every time: it is one `stat`, and an operator who drops
    the missing file into place should not have to know a restart is needed, nor be told the
    file is absent while it sits there. A LOAD failure (open_clip could not load what was
    resolved — blocked network on the tag path, corrupt or unreadable checkpoint) is
    remembered instead, because re-resolving cannot tell you anything new about it and
    retrying a multi-second load on every status query would be worse than useless."""
    global _resolution
    if _resolution is None or not _resolution.ok:
        _resolution = resolve_weights()
    if _resolution.ok and _load_error:
        return WeightsResolution(
            ok=False, source=_resolution.source, path=_resolution.path,
            reason=f"Visual search is off: the model could not be loaded ({_load_error}).",
            remedy=_LOCAL_WEIGHTS_REMEDY.format(var=config.CLIP_WEIGHTS_ENV_VAR),
        )
    return _resolution


def warm_up() -> None:
    """Load the model now, at server startup, instead of lazily on first tool call — a live
    MCP tool call is the wrong place for a first-time model load, which risks the calling
    client's tool-call timeout (the product review hit exactly that: a 60s transport timeout
    inside upload_image_asset). No-op for the offline `hash` test provider.

    Records the weights resolution and carries on if it failed. It must not raise: the
    vision model is one feature of several, and text search, upload and evaluation do not
    need it — killing the server at boot would take all of them down, hide the reason
    (stdio servers are launched with no visible console), and prevent the post-install
    health check from ever running to report which component is broken."""
    global _resolution
    _resolution = resolve_weights()
    if _resolution.ok and config.CLIP_PROVIDER == "openclip":
        try:
            _load_model()
        except Exception:
            # Resolving successfully says nothing about loading successfully: the tag path
            # resolves fine and then fails inside open_clip on a blocked network, and a
            # corrupt or unreadable checkpoint fails the same way. _load_model has already
            # recorded it; swallow it here. This runs at boot, and the text tools that do
            # not need CLIP must still come up — before this guard existed, adding warm-up
            # to the installed binary's entry point would have turned a blocked network
            # from "images degrade" into "the whole server dies".
            pass
    status = weights_status()
    if not status.ok:
        print(f"[campaign-intelligence] {status.reason} {status.remedy}",
              file=sys.stderr, flush=True)


def _embed_openclip(path: Path) -> list[float]:
    import torch
    from PIL import Image
    model, preprocess = _load_model()
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        vec = model.encode_image(img)
        vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.squeeze(0).tolist()


def _embed_hash(path: Path) -> list[float]:
    """
    Dependency-free deterministic embedding — NO torch, NO model download. A pixel-hash
    vectorizer: NOT semantically meaningful, only for smoke-testing the pipeline offline
    (same role as embedding.py's `hash` text provider). Set
    CAMPAIGN_POC_CLIP_PROVIDER=openclip for real aesthetic/regional similarity.
    """
    import hashlib
    import math
    import numpy as np
    from PIL import Image
    dim = config.CLIP_EMBED_DIM
    vec = [0.0] * dim
    with Image.open(path) as img:
        small = np.asarray(img.convert("RGB").resize((16, 16)))
    for i, px in enumerate(small.reshape(-1, 3).tolist()):
        h = int.from_bytes(hashlib.md5(f"{i}:{px}".encode()).digest()[:4], "big")
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec
