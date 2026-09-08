"""
Vector store — a real vector DB on SQLite via sqlite-vec, with a pure-Python fallback.

`sqlite-vec` adds a `vec0` virtual table with native KNN (`WHERE embedding MATCH ? ORDER BY
distance`). It's free, pip-installable, and needs no server. When the extension can't load
(not installed, or a sqlite build without extension support), we fall back to brute-force
cosine over the same vectors stored in a plain table — so the product still runs locally,
just without the ANN index. Same `add()` / `search()` interface either way, so swapping in
Postgres+pgvector later is a single-module change.

Vectors are keyed by `vector_id` — a chunk id (§6.1: one vector per chunk, several chunks
per campaign), not a campaign id. Callers roll chunk-level hits up to campaigns themselves
(see core.find_similar) via campaign_chunks.
"""
from __future__ import annotations

import json
import sqlite3
import struct
from typing import Optional

import config

_HAS_VEC: Optional[bool] = None


def _try_load_vec(conn: sqlite3.Connection) -> bool:
    """Attempt to load the sqlite-vec extension onto a connection. Cache the outcome."""
    global _HAS_VEC
    if _HAS_VEC is False:
        return False
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        _HAS_VEC = True
        return True
    except Exception:
        _HAS_VEC = False
        return False


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def init(conn: sqlite3.Connection) -> None:
    """Create the vector table. vec0 when sqlite-vec is present, else a plain fallback table."""
    if _try_load_vec(conn):
        # distance_metric=cosine so `distance` is cosine distance (1 - cosine similarity);
        # the default is L2, under which `1 - distance` below would be meaningless.
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS campaign_vectors USING vec0("
            f"vector_id TEXT PRIMARY KEY, "
            f"embedding float[{config.EMBED_DIM}] distance_metric=cosine)"
        )
    else:
        # Fallback: store the raw vector as JSON; search brute-forces in Python.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS campaign_vectors_fallback ("
            "vector_id TEXT PRIMARY KEY, embedding TEXT NOT NULL)"
        )
    conn.commit()


def add(conn: sqlite3.Connection, vector_id: str, vec: list[float]) -> None:
    if len(vec) != config.EMBED_DIM:
        raise ValueError(f"embedding dim {len(vec)} != configured EMBED_DIM {config.EMBED_DIM}")
    if _try_load_vec(conn):
        conn.execute("DELETE FROM campaign_vectors WHERE vector_id = ?", (vector_id,))
        conn.execute(
            "INSERT INTO campaign_vectors (vector_id, embedding) VALUES (?, ?)",
            (vector_id, _pack(vec)),
        )
    else:
        conn.execute(
            "INSERT OR REPLACE INTO campaign_vectors_fallback (vector_id, embedding) VALUES (?, ?)",
            (vector_id, json.dumps(vec)),
        )
    conn.commit()


def search(conn: sqlite3.Connection, query_vec: list[float], *,
           top_k: int = 5, exclude: Optional[set[str]] = None) -> list[tuple[str, float]]:
    """
    Return [(vector_id, similarity)] best-first. similarity is cosine in [-1, 1]
    (converted from sqlite-vec's cosine *distance* so both backends agree on meaning).
    """
    exclude = exclude or set()
    if _try_load_vec(conn):
        # over-fetch so post-filtering `exclude` still yields top_k
        rows = conn.execute(
            "SELECT vector_id, distance FROM campaign_vectors "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (_pack(query_vec), top_k + len(exclude)),
        ).fetchall()
        out = [(r["vector_id"], 1.0 - r["distance"]) for r in rows if r["vector_id"] not in exclude]
        return out[:top_k]

    # fallback: brute-force cosine
    import embedding as _emb
    rows = conn.execute("SELECT vector_id, embedding FROM campaign_vectors_fallback").fetchall()
    scored = [
        (r["vector_id"], _emb.cosine(query_vec, json.loads(r["embedding"])))
        for r in rows if r["vector_id"] not in exclude
    ]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:top_k]


def get_many(conn: sqlite3.Connection, vector_ids: list[str]) -> dict[str, list[float]]:
    """Fetch raw vectors for specific ids (§6.2: filtered search brute-forces cosine over a
    small pre-filtered candidate set instead of an unrestricted ANN query)."""
    if not vector_ids:
        return {}
    placeholders = ",".join("?" * len(vector_ids))
    if _try_load_vec(conn):
        rows = conn.execute(
            f"SELECT vector_id, embedding FROM campaign_vectors WHERE vector_id IN ({placeholders})",
            vector_ids,
        ).fetchall()
        dim = config.EMBED_DIM
        return {r["vector_id"]: list(struct.unpack(f"{dim}f", r["embedding"])) for r in rows}
    rows = conn.execute(
        f"SELECT vector_id, embedding FROM campaign_vectors_fallback WHERE vector_id IN ({placeholders})",
        vector_ids,
    ).fetchall()
    return {r["vector_id"]: json.loads(r["embedding"]) for r in rows}


def delete_many(conn: sqlite3.Connection, vector_ids: list[str]) -> None:
    """Purge vectors for deleted chunks (§6.4) — otherwise they'd sit as stale rows a search
    could still fetch, relying on the caller-side campaign lookup silently dropping them."""
    if not vector_ids:
        return
    placeholders = ",".join("?" * len(vector_ids))
    if _try_load_vec(conn):
        conn.execute(f"DELETE FROM campaign_vectors WHERE vector_id IN ({placeholders})", vector_ids)
    else:
        conn.execute(
            f"DELETE FROM campaign_vectors_fallback WHERE vector_id IN ({placeholders})", vector_ids
        )
    conn.commit()


def backend_name(conn: sqlite3.Connection) -> str:
    return "sqlite-vec" if _try_load_vec(conn) else "python-cosine-fallback"
