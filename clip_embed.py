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

from pathlib import Path

import config

_model = None
_preprocess = None


def embed_image(path: Path) -> list[float]:
    """Embed a single image into CLIP's visual space with the configured provider."""
    if config.CLIP_PROVIDER == "openclip":
        return _embed_openclip(path)
    if config.CLIP_PROVIDER == "hash":
        return _embed_hash(path)
    raise ValueError(f"unknown CLIP provider {config.CLIP_PROVIDER!r}")


def _load_model():
    global _model, _preprocess
    if _model is None:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED
        )
        model.eval()
        _model, _preprocess = model, preprocess
    return _model, _preprocess


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
