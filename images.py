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


def is_degenerate(hex_hash: str) -> bool:
    """Whether this hash says nothing about the image (§9.2).

    A perceptual hash measures STRUCTURE, and an image with none — a solid fill, a colour
    swatch, a blown-out frame, a black one — hashes to the same value as every other image
    with none. Flat red, flat blue and a near-white gradient all produce `8000000000000000`,
    so a comparison reports them as the same image at distance 0.

    Decks routinely carry solid-fill rectangles and placeholder blocks, and the extractor
    stores them as creative. Without this, a blank wall in a delivered photograph reports a
    colour swatch as having been built.
    """
    bits = bin(int(hex_hash, 16)).count("1")
    return bits <= 1 or bits >= (len(hex_hash) * 4) - 1


def is_match(hash_a: str, hash_b: str, *, threshold: int | None = None) -> bool:
    threshold = config.PHASH_MATCH_THRESHOLD if threshold is None else threshold
    return bool(hamming_distance(hash_a, hash_b) <= threshold)
