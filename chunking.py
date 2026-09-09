"""
Chunk packing for semantic search (§6.1).

Text arrives either as natural units — one per PDF page / PPTX slide, from extract.py — or
as a single flat string when Claude pastes deck_text it already read from an attachment
(the LLM-first path has no page/slide boundaries). Either way it's packed here into pieces
no larger than MAX_CHUNK_CHARS before embedding. A single oversized embed() call on a whole
deck is what silently orphaned long campaigns before: the provider rejects the input, the
campaign is stored but never embedded, invisible in search with no clear error.
"""
from __future__ import annotations

import config


def pack(units: list[str], *, max_chars: int | None = None) -> list[str]:
    """Pack text units into chunks <= max_chars, in order. Adjacent small units merge into
    one chunk; a single unit larger than max_chars is hard-split."""
    max_chars = max_chars or config.MAX_CHUNK_CHARS
    chunks: list[str] = []
    cur = ""
    for unit in units:
        unit = (unit or "").strip()
        if not unit:
            continue
        if len(unit) > max_chars:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(unit[i:i + max_chars] for i in range(0, len(unit), max_chars))
            continue
        candidate = f"{cur}\n\n{unit}" if cur else unit
        if len(candidate) > max_chars:
            chunks.append(cur)
            cur = unit
        else:
            cur = candidate
    if cur:
        chunks.append(cur)
    return chunks
