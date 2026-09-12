"""
Semantic search: embed campaign text, rank prior campaigns by cosine similarity.

Anthropic has no embeddings endpoint, so a separate provider is used (Voyage by default;
Ollama for fully-local). Cosine is pure-Python — no numpy — which is fine at POC scale
(brute force over a few hundred vectors). The provider is the only thing to swap if the
corpus later outgrows brute force (move to a vector index) or you change embedder.
"""
from __future__ import annotations

import math
import sys
from typing import Optional

import config


def embed(text: str, timeout: Optional[float] = None) -> list[float]:
    """Embed a single text with the configured provider. Empty text → zero-length guard.

    `timeout` lets a caller grant only the time it actually has left. A handler embeds once
    per chunk against a ceiling it must not exceed, so a fixed per-call timeout bounds when
    a call STARTS but not when the handler ENDS — the last call can begin just inside the
    budget and run for the full timeout on top. Passing the remaining budget makes that
    impossible."""
    text = (text or "").strip()
    if not text:
        raise ValueError("cannot embed empty text")
    if timeout is None:
        timeout = config.EMBED_TIMEOUT_SECONDS
    timeout = max(0.1, min(timeout, config.EMBED_TIMEOUT_SECONDS))
    # Transport failures are translated into ValueError because that is the ONLY exception
    # type the tool layer converts into a message the caller can read; anything else is
    # replaced with a generic "Error executing tool X" and the detail is discarded. A
    # marketer whose Ollama is not running deserves to be told that, not "an error".
    try:
        if config.EMBED_PROVIDER == "ollama":
            return _embed_ollama(text, timeout)
        if config.EMBED_PROVIDER == "voyage":
            return _embed_voyage(text, timeout)
    except Exception as exc:
        raise ValueError(_explain(exc)) from exc
    if config.EMBED_PROVIDER == "hash":
        return _embed_hash(text)
    raise ValueError(f"unknown embed provider {config.EMBED_PROVIDER!r}")


def _explain(exc: Exception) -> str:
    """Turn a transport failure into something an operator can act on."""
    import httpx

    where = config.OLLAMA_URL if config.EMBED_PROVIDER == "ollama" else "the Voyage API"
    what = config.EMBED_PROVIDER
    if isinstance(exc, httpx.TimeoutException):
        return (f"the {what} embedder at {where} timed out after "
                f"{config.EMBED_TIMEOUT_SECONDS:g}s. Text search needs it; check it is "
                f"running and not overloaded.")
    if isinstance(exc, httpx.ConnectError):
        return (f"could not reach the {what} embedder at {where}. Text search needs it; "
                f"check it is running.")
    if isinstance(exc, httpx.HTTPStatusError):
        return (f"the {what} embedder at {where} rejected the request "
                f"({exc.response.status_code}). If this is a long document the model's "
                f"context limit is the usual cause.")
    return f"the {what} embedder at {where} failed: {exc}"


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


def _embed_voyage(text: str, timeout: float = 0) -> list[float]:
    import httpx
    if not config.VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY is not set (or switch CAMPAIGN_POC_EMBED_PROVIDER=ollama)")
    r = httpx.post(
        "https://api.voyageai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {config.VOYAGE_API_KEY}"},
        json={"input": [text], "model": config.VOYAGE_MODEL},
        timeout=timeout or config.EMBED_TIMEOUT_SECONDS,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def _embed_ollama(text: str, timeout: float = 0) -> list[float]:
    import httpx
    r = httpx.post(
        f"{config.OLLAMA_URL}/api/embeddings",
        # keep_alive: warming at startup is pointless if Ollama unloads the model after
        # its default idle window and the next upload pays the load again — inside a
        # handler, which is the defect this item exists to remove.
        json={"model": config.OLLAMA_EMBED_MODEL, "prompt": text,
              "keep_alive": config.OLLAMA_KEEP_ALIVE},
        timeout=timeout or config.EMBED_TIMEOUT_SECONDS,
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


def warm_up() -> None:
    """Load the embedding model now, at startup, rather than inside the first tool call.

    The vision model was already warmed here; the text embedder was not, which meant every
    upload's first chunk paid Ollama's model load inside a handler working against a
    transport ceiling — the exact shape of defect 04, sitting in the code that was meant to
    be sweeping for it. Never raises: a text embedder that cannot warm is a degraded
    server, not a dead one, and the failure surfaces per call where it can be reported."""
    if config.EMBED_PROVIDER not in ("ollama", "voyage"):
        return  # the offline hash provider has nothing to load
    try:
        embed("warm up", timeout=config.EMBED_TIMEOUT_SECONDS)
    except Exception as exc:
        print(f"[campaign-intelligence] text search may be slow on first use: the "
              f"{config.EMBED_PROVIDER} embedder did not respond at startup ({exc}).",
              file=sys.stderr, flush=True)
