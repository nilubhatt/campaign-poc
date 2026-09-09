"""
Perceptual hashing for creative-reuse detection (§6.6, ship-first layer).

Catches the same/near-same photo — reused, including across regions — even after
resize/recompress/light crop. Cheap, no ML. This is deliberately NOT aesthetic/style
similarity (that needs CLIP embeddings, a much heavier dependency) — pHash only tells you
"this looks like the same photo," which is exactly the SVP's reuse-detection demo.
"""
from __future__ import annotations

from pathlib import Path

import imagehash
from PIL import Image

import config


def phash(path: Path) -> str:
    """Perceptual hash of an image file, as a hex string."""
    with Image.open(path) as img:
        return str(imagehash.phash(img))


def hamming_distance(hash_a: str, hash_b: str) -> int:
    # imagehash returns numpy scalar types; cast to plain int so results (used directly in
    # MCP tool response dicts) JSON-serialize cleanly.
    return int(imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b))


def is_match(hash_a: str, hash_b: str, *, threshold: int | None = None) -> bool:
    threshold = config.PHASH_MATCH_THRESHOLD if threshold is None else threshold
    return bool(hamming_distance(hash_a, hash_b) <= threshold)
