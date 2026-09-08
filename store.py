"""
SQLite-backed storage. Python owns the database; the MCP tools call in here.

Four record types, all first-class so the memory compounds:
  campaigns        — a past or proposed campaign: freeform detail + extracted deck text
  metrics          — outcomes attached to a campaign (freeform + optional structured)
  evaluations      — Claude's judgment of a campaign, WITH the prior campaigns it cited
  reconciliations  — post-conclusion: Claude's prediction vs. the actual metrics + lesson

Embeddings for semantic search live alongside campaigns (JSON-encoded float lists).
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any, Optional

import config
import vectorstore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    kind          TEXT NOT NULL DEFAULT 'concluded',   -- concluded | proposal
    detail        TEXT,            -- freeform: brief, audience, budget, channel, timeline, anything
    deck_text     TEXT,            -- extracted PDF/PPTX text
    asset_path    TEXT,            -- stored original file (relative to ASSET_DIR)
    embedded      INTEGER NOT NULL DEFAULT 0,  -- 1 once its vector is in the vector store
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS campaign_chunks (
    id            TEXT PRIMARY KEY,
    campaign_id   TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    text          TEXT NOT NULL,
    embedded      INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS metrics (
    id            TEXT PRIMARY KEY,
    campaign_id   TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    detail        TEXT,            -- freeform metrics / learnings, as given
    structured    TEXT,            -- optional JSON {ctr, roi, conversions, ...}
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluations (
    id            TEXT PRIMARY KEY,
    campaign_id   TEXT REFERENCES campaigns(id) ON DELETE SET NULL,  -- may be a not-yet-stored proposal
    subject_title TEXT NOT NULL,   -- what was evaluated
    cited_ids     TEXT,            -- JSON list of campaign ids Claude reasoned from
    analysis      TEXT NOT NULL,   -- Claude's judgment (freeform)
    predictions   TEXT,            -- optional JSON {predicted_ctr_range, predicted_roi_range, recommendation, ...}
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS reconciliations (
    id            TEXT PRIMARY KEY,
    evaluation_id TEXT NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    actual        TEXT,            -- the real post-conclusion metrics (freeform + optional structured)
    comparison    TEXT NOT NULL,   -- Claude's prediction-vs-actual reconciliation + lesson learned
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_campaign_idx   ON campaign_chunks(campaign_id);
CREATE INDEX IF NOT EXISTS metrics_campaign_idx ON metrics(campaign_id);
CREATE INDEX IF NOT EXISTS evals_campaign_idx   ON evaluations(campaign_id);
CREATE INDEX IF NOT EXISTS recon_eval_idx       ON reconciliations(evaluation_id);
"""


def connect() -> sqlite3.Connection:
    config.ensure_dirs()
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    conn = connect()
    conn.executescript(_SCHEMA)
    conn.commit()
    vectorstore.init(conn)   # create the vec0 (or fallback) vector table
    conn.close()


def _now() -> float:
    return time.time()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ── campaigns ────────────────────────────────────────────────────────────────

def insert_campaign(conn, *, title, kind="concluded", detail=None, deck_text=None,
                    asset_path=None) -> str:
    cid = _id("camp")
    now = _now()
    conn.execute(
        """INSERT INTO campaigns (id, title, kind, detail, deck_text, asset_path,
                                  created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (cid, title, kind, detail, deck_text, asset_path, now, now),
    )
    conn.commit()
    return cid


def mark_embedded(conn, campaign_id: str, flag: bool) -> None:
    """Flag whether a campaign has at least one embedded chunk (§6.1: many chunks per
    campaign now hold the actual vectors — this is just the rollup for list/get display)."""
    conn.execute(
        "UPDATE campaigns SET embedded = ?, updated_at = ? WHERE id = ?",
        (1 if flag else 0, _now(), campaign_id),
    )
    conn.commit()


def get_campaign(conn, campaign_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM campaigns WHERE id = ?", (campaign_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["metrics"] = [dict(m) for m in conn.execute(
        "SELECT * FROM metrics WHERE campaign_id = ? ORDER BY created_at", (campaign_id,)
    ).fetchall()]
    chunk_rows = conn.execute(
        "SELECT embedded FROM campaign_chunks WHERE campaign_id = ?", (campaign_id,)
    ).fetchall()
    d["chunks_total"] = len(chunk_rows)
    d["chunks_embedded"] = sum(1 for r in chunk_rows if r["embedded"])
    return d


def list_campaigns(conn, *, kind: Optional[str] = None) -> list[dict]:
    if kind:
        rows = conn.execute("SELECT * FROM campaigns WHERE kind = ? ORDER BY created_at", (kind,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM campaigns ORDER BY created_at").fetchall()
    return [dict(r) for r in rows]


# ── campaign chunks (§6.1: one vector per chunk, not per campaign) ───────────

def insert_chunks(conn, campaign_id: str, texts: list[str]) -> list[str]:
    now = _now()
    ids = []
    for i, text in enumerate(texts):
        chid = _id("chunk")
        conn.execute(
            """INSERT INTO campaign_chunks (id, campaign_id, chunk_index, text, created_at)
               VALUES (?,?,?,?,?)""",
            (chid, campaign_id, i, text, now),
        )
        ids.append(chid)
    conn.commit()
    return ids


def set_chunk_embedded(conn, chunk_id: str) -> None:
    conn.execute("UPDATE campaign_chunks SET embedded = 1 WHERE id = ?", (chunk_id,))
    conn.commit()


def get_chunk(conn, chunk_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM campaign_chunks WHERE id = ?", (chunk_id,)).fetchone()
    return dict(row) if row else None


def get_chunk_ids_for_campaign(conn, campaign_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT id FROM campaign_chunks WHERE campaign_id = ?", (campaign_id,)
    ).fetchall()
    return [r["id"] for r in rows]


def map_chunks_to_campaigns(conn, chunk_ids: list[str]) -> dict[str, str]:
    """Bulk chunk_id -> campaign_id, for rolling up chunk-level search hits."""
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT id, campaign_id FROM campaign_chunks WHERE id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    return {r["id"]: r["campaign_id"] for r in rows}


# ── metrics ──────────────────────────────────────────────────────────────────

def add_metrics(conn, campaign_id: str, *, detail=None, structured=None) -> str:
    mid = _id("met")
    conn.execute(
        "INSERT INTO metrics (id, campaign_id, detail, structured, created_at) VALUES (?,?,?,?,?)",
        (mid, campaign_id, detail, json.dumps(structured) if structured is not None else None, _now()),
    )
    conn.commit()
    return mid


# ── evaluations ──────────────────────────────────────────────────────────────

def insert_evaluation(conn, *, subject_title, analysis, campaign_id=None,
                      cited_ids=None, predictions=None) -> str:
    eid = _id("eval")
    conn.execute(
        """INSERT INTO evaluations (id, campaign_id, subject_title, cited_ids, analysis,
                                    predictions, created_at) VALUES (?,?,?,?,?,?,?)""",
        (eid, campaign_id, subject_title, json.dumps(cited_ids or []), analysis,
         json.dumps(predictions) if predictions is not None else None, _now()),
    )
    conn.commit()
    return eid


def get_evaluation(conn, evaluation_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["reconciliations"] = [dict(r) for r in conn.execute(
        "SELECT * FROM reconciliations WHERE evaluation_id = ? ORDER BY created_at", (evaluation_id,)
    ).fetchall()]
    return d


def list_evaluations(conn) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT id, campaign_id, subject_title, created_at FROM evaluations ORDER BY created_at DESC"
    ).fetchall()]


# ── reconciliations ──────────────────────────────────────────────────────────

def insert_reconciliation(conn, *, evaluation_id, comparison, actual=None) -> str:
    rid = _id("recon")
    conn.execute(
        "INSERT INTO reconciliations (id, evaluation_id, actual, comparison, created_at) VALUES (?,?,?,?,?)",
        (rid, evaluation_id, actual, comparison, _now()),
    )
    conn.commit()
    return rid
