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
    record_type   TEXT NOT NULL DEFAULT 'campaign',   -- campaign | reference | stub
    status        TEXT,            -- proposed | in_flight | concluded (campaigns only)
    tags          TEXT NOT NULL DEFAULT '[]',   -- JSON array of freeform strings, no fixed taxonomy
    region        TEXT,            -- freeform, e.g. "APAC"
    market        TEXT,            -- freeform, e.g. "Philippines"
    supersedes    TEXT,            -- campaign_id this record replaces (§6.4), if any
    superseded_by TEXT,            -- reverse pointer, set automatically when supersedes is used
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

def insert_campaign(conn, *, title, record_type="campaign", status=None, detail=None,
                    deck_text=None, asset_path=None, tags=None, region=None, market=None,
                    supersedes=None) -> str:
    cid = _id("camp")
    now = _now()
    if status is None and record_type == "campaign":
        status = "concluded"  # reference/stub records have no lifecycle status by default
    conn.execute(
        """INSERT INTO campaigns (id, title, record_type, status, tags, region, market,
                                  supersedes, detail, deck_text, asset_path, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cid, title, record_type, status, json.dumps(tags or []), region, market, supersedes,
         detail, deck_text, asset_path, now, now),
    )
    if supersedes:
        # Reverse pointer for cheap "exclude superseded" filtering. A nonexistent target is
        # a harmless no-op (0 rows updated) — supersedes stays on the new row for traceability.
        conn.execute("UPDATE campaigns SET superseded_by = ? WHERE id = ?", (cid, supersedes))
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
    d["tags"] = json.loads(d["tags"]) if d["tags"] else []
    d["metrics"] = [dict(m) for m in conn.execute(
        "SELECT * FROM metrics WHERE campaign_id = ? ORDER BY created_at", (campaign_id,)
    ).fetchall()]
    chunk_rows = conn.execute(
        "SELECT embedded FROM campaign_chunks WHERE campaign_id = ?", (campaign_id,)
    ).fetchall()
    d["chunks_total"] = len(chunk_rows)
    d["chunks_embedded"] = sum(1 for r in chunk_rows if r["embedded"])
    return d


def list_campaigns(conn, *, record_type: Optional[str] = None,
                   status: Optional[str] = None) -> list[dict]:
    clauses, params = [], []
    if record_type:
        clauses.append("record_type = ?")
        params.append(record_type)
    if status:
        clauses.append("status = ?")
        params.append(status)
    sql = "SELECT * FROM campaigns"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at"
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def filter_campaign_ids(conn, *, record_type: Optional[str] = None, status: Optional[str] = None,
                        tags: Optional[list[str]] = None, region: Optional[str] = None,
                        market: Optional[str] = None, exclude_campaign_id: Optional[str] = None
                        ) -> list[str]:
    """
    Structured filtering BEFORE similarity ranking (§6.2) — on a single-brand corpus, pure
    vector search returns noise; narrow to the matching campaigns first (e.g. "concluded
    seeding campaigns in APAC"), then rank what's left by similarity. region/market are
    freeform and matched case-insensitively; tags matches if the campaign has ANY of the
    given tags (no fixed taxonomy — §6.3 decision).
    """
    clauses, params = ["superseded_by IS NULL"], []  # §6.4: never surface a replaced record
    if record_type:
        clauses.append("record_type = ?")
        params.append(record_type)
    if status:
        clauses.append("status = ?")
        params.append(status)
    if region:
        clauses.append("LOWER(region) = LOWER(?)")
        params.append(region)
    if market:
        clauses.append("LOWER(market) = LOWER(?)")
        params.append(market)
    if exclude_campaign_id:
        clauses.append("id != ?")
        params.append(exclude_campaign_id)

    sql = "SELECT id, tags FROM campaigns WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at"
    rows = conn.execute(sql, params).fetchall()

    if tags:
        wanted = {t.lower() for t in tags}
        rows = [r for r in rows if wanted & {t.lower() for t in json.loads(r["tags"] or "[]")}]

    return [r["id"] for r in rows]


def update_campaign(conn, campaign_id: str, *, title=None, detail=None, record_type=None,
                    status=None, tags=None, region=None, market=None) -> bool:
    """Update campaign metadata (§6.4) — NOT deck_text/chunks/embeddings; re-upload (or
    supersede) for content changes. Only given fields change; tags, if given, fully replaces
    the existing list rather than merging. Returns whether the campaign exists."""
    fields, params = [], []
    if title is not None:
        fields.append("title = ?"); params.append(title)
    if detail is not None:
        fields.append("detail = ?"); params.append(detail)
    if record_type is not None:
        fields.append("record_type = ?"); params.append(record_type)
    if status is not None:
        fields.append("status = ?"); params.append(status)
    if tags is not None:
        fields.append("tags = ?"); params.append(json.dumps(tags))
    if region is not None:
        fields.append("region = ?"); params.append(region)
    if market is not None:
        fields.append("market = ?"); params.append(market)

    if not fields:
        return get_campaign(conn, campaign_id) is not None

    fields.append("updated_at = ?")
    params.append(_now())
    params.append(campaign_id)
    cur = conn.execute(f"UPDATE campaigns SET {', '.join(fields)} WHERE id = ?", params)
    conn.commit()
    return cur.rowcount > 0


def delete_campaign(conn, campaign_id: str) -> bool:
    """Delete a campaign and everything that's exclusively its own (§6.4): chunks, vectors,
    metrics (all ON DELETE CASCADE). Evaluations that cited it are kept — they're a record of
    a judgment that was made — but detached (ON DELETE SET NULL). If another record's
    `supersedes` pointed here, that record is restored to active (its `superseded_by`
    cleared), since the thing it replaced no longer exists to be "replaced.\""""
    chunk_ids = get_chunk_ids_for_campaign(conn, campaign_id)
    if chunk_ids:
        vectorstore.delete_many(conn, chunk_ids)
    conn.execute("UPDATE campaigns SET superseded_by = NULL WHERE superseded_by = ?", (campaign_id,))
    cur = conn.execute("DELETE FROM campaigns WHERE id = ?", (campaign_id,))
    conn.commit()
    return cur.rowcount > 0


def get_superseded_campaign_ids(conn) -> set[str]:
    rows = conn.execute("SELECT id FROM campaigns WHERE superseded_by IS NOT NULL").fetchall()
    return {r["id"] for r in rows}


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


def get_chunk_ids_for_campaigns(conn, campaign_ids: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {cid: [] for cid in campaign_ids}
    if not campaign_ids:
        return out
    placeholders = ",".join("?" * len(campaign_ids))
    rows = conn.execute(
        f"SELECT id, campaign_id FROM campaign_chunks WHERE campaign_id IN ({placeholders})",
        campaign_ids,
    ).fetchall()
    for r in rows:
        out[r["campaign_id"]].append(r["id"])
    return out


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
