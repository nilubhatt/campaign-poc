"""
Vector store — a real vector DB on SQLite via sqlite-vec, with a pure-Python fallback.

`sqlite-vec` adds a `vec0` virtual table with native KNN (`WHERE embedding MATCH ? ORDER BY
distance`). It's free, pip-installable, and needs no server. When the extension can't load
(not installed, or a sqlite build without extension support), we fall back to brute-force
cosine over the same vectors stored in a plain table — so the product still runs locally,
just without the ANN index. Same `add()` / `search()` interface either way, so swapping in
Postgres+pgvector later is a single-module change.

Vectors are keyed by `vector_id` — a chunk id for text (§6.1: one vector per chunk, several
chunks per campaign) or an asset id for images (§6.6) — not a campaign id. Callers roll
chunk/asset-level hits up to campaigns themselves (see core.find_similar/find_similar_images).

Multiple named `space`s (default "campaign") let unrelated vector kinds coexist without
sharing a table — needed because CLIP image embeddings (512-dim) and text-chunk embeddings
(768-dim, nomic-embed-text) can't live in the same fixed-width vec0 column.
"""
from __future__ import annotations

import json
import sqlite3
import struct
from typing import Optional

import config

_HAS_VEC: Optional[bool] = None

# sqlite-vec's KNN `k` parameter has a hard upper bound (observed: 4096 in this build) — a
# large `exclude` set (top_k + len(exclude), see search()) must be clamped before it crosses
# that limit and crashes the query outright. Clamping means a very large exclude set can
# silently return fewer than top_k results rather than erroring, which is the right tradeoff.
_MAX_ANN_K = 4096


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


def _table_names(space: str) -> tuple[str, str]:
    return f"{space}_vectors", f"{space}_vectors_fallback"


def _default_dim(space: str) -> int:
    """Known spaces default their own dimension so callers don't have to remember to pass
    dim= by hand at every call site (review flagged the repeated-magic-number version of
    this as a footgun — one missed dim= and get_many would misread the vector blob)."""
    # `commitment` is CLIP's space too (§9.3): a commitment vector is the phrase from the
    # brief embedded by the SAME model as the photographs, which is the only reason the two
    # can be compared at all. Sized by the text embedder it would be unstorable.
    return (config.CLIP_EMBED_DIM if space in ("asset", "commitment")
            else config.EMBED_DIM)


def init(conn: sqlite3.Connection, *, space: str = "campaign", dim: Optional[int] = None) -> None:
    """Create the vector table for a space. vec0 when sqlite-vec is present, else a plain
    fallback table. dim defaults to config.EMBED_DIM (the default "campaign" text space);
    other spaces (e.g. "asset" for CLIP's 512-dim) must pass their own dim."""
    dim = dim or _default_dim(space)
    vec_table, fallback_table = _table_names(space)
    if _try_load_vec(conn):
        # distance_metric=cosine so `distance` is cosine distance (1 - cosine similarity);
        # the default is L2, under which `1 - distance` below would be meaningless.
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {vec_table} USING vec0("
            f"vector_id TEXT PRIMARY KEY, "
            f"embedding float[{dim}] distance_metric=cosine)"
        )
    else:
        # Fallback: store the raw vector as JSON; search brute-forces in Python.
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {fallback_table} ("
            f"vector_id TEXT PRIMARY KEY, embedding TEXT NOT NULL)"
        )
    conn.commit()


def add(conn: sqlite3.Connection, vector_id: str, vec: list[float], *,
        space: str = "campaign", dim: Optional[int] = None) -> None:
    dim = dim or _default_dim(space)
    if len(vec) != dim:
        raise ValueError(f"embedding dim {len(vec)} != expected {dim} for space {space!r}")
    vec_table, fallback_table = _table_names(space)
    if _try_load_vec(conn):
        conn.execute(f"DELETE FROM {vec_table} WHERE vector_id = ?", (vector_id,))
        conn.execute(
            f"INSERT INTO {vec_table} (vector_id, embedding) VALUES (?, ?)",
            (vector_id, _pack(vec)),
        )
    else:
        conn.execute(
            f"INSERT OR REPLACE INTO {fallback_table} (vector_id, embedding) VALUES (?, ?)",
            (vector_id, json.dumps(vec)),
        )
    conn.commit()


def search(conn: sqlite3.Connection, query_vec: list[float], *, top_k: int = 5,
           exclude: Optional[set[str]] = None, space: str = "campaign") -> list[tuple[str, float]]:
    """
    Return [(vector_id, similarity)] best-first, within one space. similarity is cosine in
    [-1, 1] (converted from sqlite-vec's cosine *distance* so both backends agree on meaning).
    """
    vec_table, fallback_table = _table_names(space)
    exclude = exclude or set()
    if _try_load_vec(conn):
        # over-fetch so post-filtering `exclude` still yields top_k, clamped to sqlite-vec's
        # own k limit — an unclamped k crashes instead of just returning fewer results.
        k = min(top_k + len(exclude), _MAX_ANN_K)
        rows = conn.execute(
            f"SELECT vector_id, distance FROM {vec_table} "
            f"WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (_pack(query_vec), k),
        ).fetchall()
        out = [(r["vector_id"], 1.0 - r["distance"]) for r in rows if r["vector_id"] not in exclude]
        # §7.2: equal distances come back from the ANN index in whatever order it stored
        # them, which is not stable across machines or across an insert. Sorting by id within
        # a tie makes the same library return the same evidence package every time.
        out.sort(key=lambda t: (-t[1], t[0]))
        return out[:top_k]

    # fallback: brute-force cosine
    import embedding as _emb
    rows = conn.execute(f"SELECT vector_id, embedding FROM {fallback_table}").fetchall()
    scored = [
        (r["vector_id"], _emb.cosine(query_vec, json.loads(r["embedding"])))
        for r in rows if r["vector_id"] not in exclude
    ]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:top_k]


def get_many(conn: sqlite3.Connection, vector_ids: list[str], *, space: str = "campaign",
            dim: Optional[int] = None) -> dict[str, list[float]]:
    """Fetch raw vectors for specific ids (§6.2: filtered search brute-forces cosine over a
    small pre-filtered candidate set instead of an unrestricted ANN query)."""
    if not vector_ids:
        return {}
    dim = dim or _default_dim(space)
    vec_table, fallback_table = _table_names(space)
    placeholders = ",".join("?" * len(vector_ids))
    if _try_load_vec(conn):
        rows = conn.execute(
            f"SELECT vector_id, embedding FROM {vec_table} WHERE vector_id IN ({placeholders})",
            vector_ids,
        ).fetchall()
        return {r["vector_id"]: list(struct.unpack(f"{dim}f", r["embedding"])) for r in rows}
    rows = conn.execute(
        f"SELECT vector_id, embedding FROM {fallback_table} WHERE vector_id IN ({placeholders})",
        vector_ids,
    ).fetchall()
    return {r["vector_id"]: json.loads(r["embedding"]) for r in rows}


def delete_many(conn: sqlite3.Connection, vector_ids: list[str], *, space: str = "campaign") -> None:
    """Purge vectors for deleted chunks/assets (§6.4/6.6) — otherwise they'd sit as stale rows
    a search could still fetch, relying on the caller-side lookup silently dropping them."""
    if not vector_ids:
        return
    vec_table, fallback_table = _table_names(space)
    placeholders = ",".join("?" * len(vector_ids))
    if _try_load_vec(conn):
        conn.execute(f"DELETE FROM {vec_table} WHERE vector_id IN ({placeholders})", vector_ids)
    else:
        conn.execute(f"DELETE FROM {fallback_table} WHERE vector_id IN ({placeholders})", vector_ids)
    conn.commit()


def backend_name(conn: sqlite3.Connection) -> str:
    return "sqlite-vec" if _try_load_vec(conn) else "python-cosine-fallback"


def count_unreadable_vectors(conn) -> int:
    """Vectors sitting in the table this process does NOT read.

    The two backends store into different tables (`{space}_vectors` via sqlite-vec,
    `{space}_vectors_fallback` otherwise). A database written where the extension loaded and
    opened where it does not — a bundle that lost the native library, a copied file, a
    Python without extension support — has every row flagged embedded and every search
    returning nothing. That combination is invisible to a coverage count, which is exactly
    why it is worth asking about explicitly."""
    live_is_vec = _try_load_vec(conn)
    stranded = 0
    for space in ("campaign", "asset"):
        vec_table, fallback_table = _table_names(space)
        unread = fallback_table if live_is_vec else vec_table
        try:
            stranded += conn.execute(f"SELECT COUNT(*) AS n FROM {unread}").fetchone()["n"]
        except Exception:
            continue  # the table may not exist in this database; nothing stranded there
    return stranded
