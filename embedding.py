"""
Semantic search: embed campaign text, rank prior campaigns by cosine similarity.

Anthropic has no embeddings endpoint, so a separate provider is used (Voyage by default;
Ollama for fully-local). Cosine is pure-Python — no numpy — which is fine at POC scale
(brute force over a few hundred vectors). The provider is the only thing to swap if the
corpus later outgrows brute force (move to a vector index) or you change embedder.
"""
from __future__ import annotations

import math
from typing import Optional

import config


def embed(text: str) -> list[float]:
    """Embed a single text with the configured provider. Empty text → zero-length guard."""
    text = (text or "").strip()
    if not text:
        raise ValueError("cannot embed empty text")
    if config.EMBED_PROVIDER == "ollama":
        return _embed_ollama(text)
    if config.EMBED_PROVIDER == "voyage":
        return _embed_voyage(text)
    if config.EMBED_PROVIDER == "hash":
        return _embed_hash(text)
    raise ValueError(f"unknown embed provider {config.EMBED_PROVIDER!r}")


def _embed_hash(text: str) -> list[float]:
    """
    Dependency-free deterministic embedding — NO external service. A bag-of-words hashing
    vectorizer: NOT semantically meaningful, only for smoke-testing the pipeline offline.
    For real semantic search set CAMPAIGN_POC_EMBED_PROVIDER=ollama (local + free).
    """
    import hashlib
    dim = config.EMBED_DIM
    vec = [0.0] * dim
    for tok in text.lower().split():
        h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "big")
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


def _embed_voyage(text: str) -> list[float]:
    import httpx
    if not config.VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY is not set (or switch CAMPAIGN_POC_EMBED_PROVIDER=ollama)")
    r = httpx.post(
        "https://api.voyageai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {config.VOYAGE_API_KEY}"},
        json={"input": [text], "model": config.VOYAGE_MODEL},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def _embed_ollama(text: str) -> list[float]:
    import httpx
    r = httpx.post(
        f"{config.OLLAMA_URL}/api/embeddings",
        json={"model": config.OLLAMA_EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def rank(query_vec: list[float], candidates: list[tuple[str, list[float]]],
         *, top_k: int = 5, exclude: Optional[set[str]] = None) -> list[tuple[str, float]]:
    """Return [(campaign_id, similarity), ...] sorted desc, top_k, excluding given ids."""
    exclude = exclude or set()
    scored = [
        (cid, cosine(query_vec, vec))
        for cid, vec in candidates
        if cid not in exclude
    ]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:top_k]
