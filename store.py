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
from typing import Any, Optional, Union

import config
import vectorstore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    record_type   TEXT NOT NULL DEFAULT 'campaign',   -- campaign | reference | stub
    status        TEXT,            -- proposed | in_flight | concluded (campaigns only)
    tags          TEXT NOT NULL DEFAULT '[]',   -- JSON array of {value, source}, no fixed
                                    -- taxonomy; source is verified|stated (§ tag provenance)
    region        TEXT,            -- freeform, single value, e.g. "Malaysia" (exact-match
                                    -- filter - a multi-country activation needs `markets`
                                    -- below too; region alone can't express membership)
    market        TEXT,            -- freeform, single value, e.g. "SEA" (grouping label)
    markets       TEXT NOT NULL DEFAULT '[]',  -- JSON array of strings: every country/sub-
                                    -- market a campaign's activation actually touched,
                                    -- matched by membership (region/market are single-value
                                    -- exact-match and can't represent "ran in MY and ID")
    collection    TEXT,            -- freeform: links market/version variants of the same
                                    -- creative (e.g. "Khloe Q2 2026") - symmetric grouping,
                                    -- unlike supersedes (asymmetric replacement)
    supersedes    TEXT,            -- campaign_id this record replaces (§6.4), if any.
                                    -- "superseded by" is DERIVED (see is_superseded/
                                    -- get_superseded_campaign_ids), not a maintained reverse
                                    -- pointer — a cached pointer breaks on supersession
                                    -- chains and fan-in (two records both superseding one).
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
CREATE TABLE IF NOT EXISTS assets (
    id            TEXT PRIMARY KEY,
    campaign_id   TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    modality      TEXT NOT NULL DEFAULT 'image',   -- image today; video/audio later (§6.7)
    file_path     TEXT NOT NULL,    -- relative to ASSET_DIR
    embedded      INTEGER NOT NULL DEFAULT 0,  -- 1 once its CLIP vector is in the vector store
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS asset_fingerprints (
    asset_id      TEXT PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
    phash         TEXT NOT NULL,
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS metrics (
    id            TEXT PRIMARY KEY,
    campaign_id   TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    metric_type   TEXT NOT NULL DEFAULT 'actual',  -- actual | predicted
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
CREATE INDEX IF NOT EXISTS assets_campaign_idx   ON assets(campaign_id);
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


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Additive migrations for columns added after a release — CREATE TABLE IF NOT EXISTS
    (in _SCHEMA) does nothing for an EXISTING table, so a column added since someone's last
    install (e.g. `collection`, added after v0.2.0 shipped with installers) would otherwise
    make every insert/read against their real database fail outright (reviewed and
    reproduced against a pre-collection schema). Never destructive — only ever adds a column
    (nullable, or NOT NULL with a constant default — SQLite allows the latter on ADD COLUMN,
    backfilling existing rows) to an existing table; a fresh DB already has it via _SCHEMA."""
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(campaigns)").fetchall()}
    if "collection" not in existing:
        conn.execute("ALTER TABLE campaigns ADD COLUMN collection TEXT")
    if "markets" not in existing:
        conn.execute("ALTER TABLE campaigns ADD COLUMN markets TEXT NOT NULL DEFAULT '[]'")
    conn.commit()


def init_db() -> None:
    conn = connect()
    conn.executescript(_SCHEMA)
    conn.commit()
    _migrate_schema(conn)
    vectorstore.init(conn)   # "campaign" space (text chunks, dim=config.EMBED_DIM)
    vectorstore.init(conn, space="asset", dim=config.CLIP_EMBED_DIM)  # CLIP image vectors
    conn.close()


def _now() -> float:
    return time.time()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ── campaigns ────────────────────────────────────────────────────────────────

_VALID_RECORD_TYPES = {"campaign", "reference", "stub"}
_VALID_STATUSES = {"proposed", "in_flight", "concluded"}


def _validate_record_type(value) -> None:
    if value not in _VALID_RECORD_TYPES:
        raise ValueError(f"invalid record_type {value!r}, must be one of {sorted(_VALID_RECORD_TYPES)}")


def _validate_status(value) -> None:
    if value is not None and value not in _VALID_STATUSES:
        raise ValueError(f"invalid status {value!r}, must be one of {sorted(_VALID_STATUSES)} or None")


_VALID_TAG_SOURCES = {"verified", "stated"}


def normalize_tags(tags, *, has_actual_metrics: bool = False) -> list[dict]:
    """
    Normalize tags to [{"value": str, "source": "verified"|"stated"}, ...]. A plain string
    is accepted for ergonomics (LLM-first, conversational intake — §6.9) and defaults to
    'stated': the conservative assumption that a tag is someone's claim, not measured
    evidence, unless explicitly marked 'verified'. This distinction matters specifically for
    performance-outcome tags (performed_well, underperformed, ...) — without it, a stated
    impression reads as evidence and gets weighted as if it were (user feedback: four of
    five performance tags in the real data were impressions, one had real numbers behind
    it). tags overlap and a campaign commonly carries several at once (e.g. a creative-
    reaction tag plus a performance tag) — this is a list, not a single value.

    'verified' REQUIRES has_actual_metrics=True (i.e. the campaign already has at least one
    metric_type='actual' record) — reviewed and found the first version of this let anyone
    claim 'verified' with nothing behind it, defeating the whole point. A brand-new campaign
    can never satisfy this (chicken-and-egg: no metrics exist yet at insert time), so
    'verified' can only be set via update_campaign, after add_metrics/bulk_import_metrics.

    Duplicate values (case/whitespace-insensitive) are deduped, preferring 'verified' over
    'stated' if both appear for the same value.
    """
    if tags is None:
        return []
    if not isinstance(tags, list):
        raise ValueError(f"tags must be a list, got {tags!r}")
    out = []
    for t in tags:
        if isinstance(t, str):
            value, source = t.strip(), "stated"
        elif isinstance(t, dict):
            raw_value = t.get("value")
            if not isinstance(raw_value, str) or not raw_value.strip():
                raise ValueError(f"tag object must have a non-empty string 'value', got {t!r}")
            value = raw_value.strip()
            source = t.get("source", "stated")
            if source not in _VALID_TAG_SOURCES:
                raise ValueError(f"invalid tag source {source!r}, must be one of {sorted(_VALID_TAG_SOURCES)}")
        else:
            raise ValueError(f"tag must be a string or {{value, source}} object, got {t!r}")
        if not value:
            raise ValueError(f"tag value cannot be empty or whitespace-only, got {t!r}")
        if source == "verified" and not has_actual_metrics:
            raise ValueError(
                f"tag {value!r} cannot be marked source='verified' — this campaign has no "
                f"metric_type='actual' record on file yet. Add real metrics first "
                f"(add_metrics/bulk_import_metrics), then update_campaign to mark it verified."
            )
        out.append({"value": value, "source": source})

    deduped: dict[str, dict] = {}
    for entry in out:
        key = entry["value"].lower()
        if key not in deduped or (deduped[key]["source"] != "verified" and entry["source"] == "verified"):
            deduped[key] = entry
    return list(deduped.values())


def normalize_markets(markets) -> list[str]:
    """Normalize markets to a deduped list of stripped strings. Mirrors tags' list-not-a-
    single-value philosophy but with no provenance concept - this is a plain membership list
    (every country/sub-market an activation touched), matched by filter_campaign_ids'
    `markets` param via case-insensitive membership, not exact-match like region/market."""
    if markets is None:
        return []
    if not isinstance(markets, list):
        raise ValueError(f"markets must be a list, got {markets!r}")
    out = []
    seen: set[str] = set()
    for m in markets:
        if not isinstance(m, str) or not m.strip():
            raise ValueError(f"each market must be a non-empty string, got {m!r}")
        value = m.strip()
        key = value.lower()
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _parse_stored_markets(raw: Optional[str]) -> list[str]:
    return json.loads(raw) if raw else []


def _parse_stored_tags(raw: Optional[str]) -> list[dict]:
    """Parse a campaign's stored tags JSON, tolerating legacy plain-string entries (from
    before tags carried provenance) — reviewed and found the strict-dict-only reader crashed
    outright on any pre-existing row with the old shape."""
    entries = json.loads(raw) if raw else []
    out = []
    for e in entries:
        if isinstance(e, str):
            out.append({"value": e, "source": "stated"})
        elif isinstance(e, dict):
            out.append({"value": e.get("value", ""), "source": e.get("source", "stated")})
    return out


def _parse_tag_query(tags) -> list[tuple[str, Optional[str]]]:
    """
    Parse a tags filter into [(value_lower, source_or_None), ...]. Each entry is a plain
    string (match that value regardless of source) or a {value, source} object (match that
    value AND require that specific source) — mirrors the storage shape so a query can mix
    precision per tag: tags=["liked", {"value": "underperformed", "source": "verified"}]
    means "liked, any evidence, AND underperformed, but only if verified." A single global
    verified-only flag can't express this — reviewed and found it forces every queried tag
    (including ones that can never be "verified," like a creative-reaction tag) through the
    same requirement, making the actual quadrant query unanswerable.

    A bare string or {value, source} object (not wrapped in a list) is also accepted for a
    single-tag query — the same ergonomic gap found in testing for `markets`: a caller with
    only one tag to filter on will plausibly pass it unwrapped.
    """
    if tags is None:
        return []
    if isinstance(tags, (str, dict)):
        tags = [tags]
    if not isinstance(tags, list):
        raise ValueError(f"tags filter must be a list (or a single string/object), got {tags!r}")
    out = []
    for t in tags:
        if isinstance(t, str):
            value, source = t.strip(), None
        elif isinstance(t, dict):
            raw_value = t.get("value")
            if not isinstance(raw_value, str) or not raw_value.strip():
                raise ValueError(f"tag query object must have a non-empty string 'value', got {t!r}")
            value = raw_value.strip()
            source = t.get("source")
            if source is not None and source not in _VALID_TAG_SOURCES:
                raise ValueError(f"invalid tag source {source!r} in query, must be one of "
                                 f"{sorted(_VALID_TAG_SOURCES)} or omitted")
        else:
            raise ValueError(f"tag query entry must be a string or {{value, source}} object, got {t!r}")
        if not value:
            raise ValueError(f"tag query value cannot be empty or whitespace-only, got {t!r}")
        out.append((value.lower(), source))
    return out


def insert_campaign(conn, *, title, record_type="campaign", status=None, detail=None,
                    deck_text=None, asset_path=None, tags=None, region=None, market=None,
                    markets=None, collection=None, supersedes=None) -> str:
    _validate_record_type(record_type)
    _validate_status(status)
    tags = normalize_tags(tags, has_actual_metrics=False)  # brand-new: no metrics can exist yet
    markets = normalize_markets(markets)
    cid = _id("camp")
    now = _now()
    if status is None and record_type == "campaign":
        status = "concluded"  # reference/stub records have no lifecycle status by default
    conn.execute(
        """INSERT INTO campaigns (id, title, record_type, status, tags, region, market,
                                  markets, collection, supersedes, detail, deck_text,
                                  asset_path, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cid, title, record_type, status, json.dumps(tags), region, market,
         json.dumps(markets), collection, supersedes, detail, deck_text, asset_path, now, now),
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
    d["tags"] = _parse_stored_tags(d["tags"])
    d["markets"] = _parse_stored_markets(d["markets"])
    d["metrics"] = [dict(m) for m in conn.execute(
        "SELECT * FROM metrics WHERE campaign_id = ? ORDER BY created_at", (campaign_id,)
    ).fetchall()]
    d["has_metrics"] = len(d["metrics"]) > 0
    d["has_evaluations"] = conn.execute(
        "SELECT 1 FROM evaluations WHERE campaign_id = ? LIMIT 1", (campaign_id,)
    ).fetchone() is not None
    d["superseded_by"] = [r["id"] for r in conn.execute(
        "SELECT id FROM campaigns WHERE supersedes = ?", (campaign_id,)
    ).fetchall()]
    d["is_superseded"] = len(d["superseded_by"]) > 0
    # Derived, not maintained — same reasoning as superseded_by/is_superseded above: a
    # freeform collection value on its own isn't usable from a single record without a way
    # to find the other members (design review flagged this as the missing half of the
    # feature).
    d["collection_siblings"] = [r["id"] for r in conn.execute(
        "SELECT id FROM campaigns WHERE collection IS NOT NULL AND LOWER(collection) = LOWER(?) "
        "AND id != ?", (d["collection"], campaign_id)
    ).fetchall()] if d["collection"] else []
    d["assets"] = get_assets_for_campaign(conn, campaign_id)
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

    ids = [r["id"] for r in rows]
    with_metrics: set[str] = set()
    with_evaluations: set[str] = set()
    if ids:
        placeholders = ",".join("?" * len(ids))
        with_metrics = {r["campaign_id"] for r in conn.execute(
            f"SELECT DISTINCT campaign_id FROM metrics WHERE campaign_id IN ({placeholders})", ids
        ).fetchall()}
        with_evaluations = {r["campaign_id"] for r in conn.execute(
            f"SELECT DISTINCT campaign_id FROM evaluations WHERE campaign_id IN ({placeholders})", ids
        ).fetchall()}
    superseded_ids = get_superseded_campaign_ids(conn)

    out = []
    for r in rows:
        d = dict(r)
        d["tags"] = _parse_stored_tags(d["tags"])
        d["markets"] = _parse_stored_markets(d["markets"])
        d["has_metrics"] = d["id"] in with_metrics
        d["is_superseded"] = d["id"] in superseded_ids
        d["has_evaluations"] = d["id"] in with_evaluations
        out.append(d)
    return out


def filter_campaign_ids(conn, *, record_type: Optional[str] = None, status: Optional[str] = None,
                        tags: Optional[Union[str, dict, list]] = None, match_all_tags: bool = False,
                        region: Optional[str] = None, market: Optional[str] = None,
                        markets: Optional[Union[str, list[str]]] = None,
                        collection: Optional[str] = None,
                        exclude_campaign_id: Optional[str] = None) -> list[str]:
    """
    Structured filtering BEFORE similarity ranking (§6.2) — on a single-brand corpus, pure
    vector search returns noise; narrow to the matching campaigns first (e.g. "concluded
    seeding campaigns in APAC"), then rank what's left by similarity. region/market/
    collection are freeform, single-value, and matched case-insensitively (exact match).

    markets is different on purpose: a real activation can span several countries (e.g. a
    SEA launch touching Malaysia, Singapore, and Indonesia), which a single-value region/
    market can't represent — tagging region="Malaysia" makes that same campaign invisible
    to a query for region="Indonesia", even though the activation genuinely included it.
    Pass a single country/sub-market string, or a list of them for ANY-match (e.g.
    markets=["Indonesia", "Thailand"] — matches a campaign whose activation touched EITHER,
    not both); matches any campaign whose stored `markets` list contains at least one of the
    given values (case-insensitive membership, not exact-match on the whole field).

    tags is a list of plain strings (match that value, any source) and/or {value, source}
    objects (match that value AND require that specific source) — mix freely. Defaults to
    ANY-match (has at least one). Pass match_all_tags=True for AND-match (has every one) —
    needed for a quadrant query like tags=["liked", {"value": "underperformed",
    "source": "verified"}]: "liked" (any evidence) AND "underperformed" but only if
    verified. A creative-reaction tag can never itself be "verified" the way a performance
    tag can, so per-tag precision — not one global verified-only flag — is what makes the
    actual quadrant query answerable.
    """
    wanted = _parse_tag_query(tags)
    # §6.4: never surface a replaced record — derived live (see get_campaign's
    # is_superseded/superseded_by) rather than a maintained reverse-pointer column.
    clauses = ["id NOT IN (SELECT supersedes FROM campaigns WHERE supersedes IS NOT NULL)"]
    params: list = []
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
    if collection:
        clauses.append("LOWER(collection) = LOWER(?)")
        params.append(collection)
    if exclude_campaign_id:
        clauses.append("id != ?")
        params.append(exclude_campaign_id)

    sql = "SELECT id, tags, markets FROM campaigns WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at"
    rows = conn.execute(sql, params).fetchall()

    if wanted:
        def one_matches(stored_pairs, value, source) -> bool:
            return any(v == value and (source is None or s == source) for v, s in stored_pairs)

        def campaign_matches(row) -> bool:
            stored_pairs = [(e["value"].lower(), e["source"]) for e in _parse_stored_tags(row["tags"])]
            checks = (one_matches(stored_pairs, v, s) for v, s in wanted)
            return all(checks) if match_all_tags else any(checks)

        rows = [r for r in rows if campaign_matches(r)]

    if markets:
        query_values = [markets] if isinstance(markets, str) else markets
        wanted_markets = {v.strip().lower() for v in query_values}
        rows = [r for r in rows
                if wanted_markets & {m.lower() for m in _parse_stored_markets(r["markets"])}]

    return [r["id"] for r in rows]


def update_campaign(conn, campaign_id: str, *, title=None, detail=None, record_type=None,
                    status=None, tags=None, region=None, market=None, markets=None,
                    collection=None) -> bool:
    """Update campaign metadata (§6.4) — NOT deck_text/chunks/embeddings; re-upload (or
    supersede) for content changes. Only given fields change; tags/markets, if given, fully
    replace the existing list rather than merging. Returns whether the campaign exists."""
    fields, params = [], []
    if title is not None:
        fields.append("title = ?"); params.append(title)
    if detail is not None:
        fields.append("detail = ?"); params.append(detail)
    if record_type is not None:
        _validate_record_type(record_type)
        fields.append("record_type = ?"); params.append(record_type)
    if status is not None:
        _validate_status(status)
        fields.append("status = ?"); params.append(status)
    if tags is not None:
        has_actual = conn.execute(
            "SELECT 1 FROM metrics WHERE campaign_id = ? AND metric_type = 'actual' LIMIT 1",
            (campaign_id,),
        ).fetchone() is not None
        tags = normalize_tags(tags, has_actual_metrics=has_actual)
        fields.append("tags = ?"); params.append(json.dumps(tags))
    if region is not None:
        fields.append("region = ?"); params.append(region)
    if market is not None:
        fields.append("market = ?"); params.append(market)
    if markets is not None:
        fields.append("markets = ?"); params.append(json.dumps(normalize_markets(markets)))
    if collection is not None:
        fields.append("collection = ?"); params.append(collection)

    if not fields:
        return get_campaign(conn, campaign_id) is not None

    fields.append("updated_at = ?")
    params.append(_now())
    params.append(campaign_id)
    cur = conn.execute(f"UPDATE campaigns SET {', '.join(fields)} WHERE id = ?", params)
    conn.commit()
    return cur.rowcount > 0


def delete_campaign(conn, campaign_id: str) -> bool:
    """Delete a campaign and everything that's exclusively its own (§6.4): chunks + their
    text vectors, image assets + their pHash fingerprints, CLIP vectors, and the actual
    files on disk (deck + images) — not just the SQL rows (review found the vectors and
    files were being left behind, a leak and a GDPR-erasure gap). Metrics cascade (FK).
    Evaluations that cited it are kept — they're a record of a judgment that was made — but
    detached (ON DELETE SET NULL). `supersedes`/`is_superseded` are derived live from
    `supersedes`, not a maintained pointer, so deleting a record automatically and correctly
    updates what counts as superseded — nothing to clean up here for that."""
    campaign = get_campaign(conn, campaign_id)
    if not campaign:
        return False

    chunk_ids = get_chunk_ids_for_campaign(conn, campaign_id)
    if chunk_ids:
        vectorstore.delete_many(conn, chunk_ids)

    assets = get_assets_for_campaign(conn, campaign_id)
    asset_ids = [a["id"] for a in assets]
    if asset_ids:
        vectorstore.delete_many(conn, asset_ids, space="asset")

    for file_path in [campaign["asset_path"]] + [a["file_path"] for a in assets]:
        if file_path:
            (config.ASSET_DIR / file_path).unlink(missing_ok=True)

    cur = conn.execute("DELETE FROM campaigns WHERE id = ?", (campaign_id,))
    conn.commit()
    return cur.rowcount > 0


def get_superseded_campaign_ids(conn) -> set[str]:
    rows = conn.execute("SELECT DISTINCT supersedes FROM campaigns WHERE supersedes IS NOT NULL").fetchall()
    return {r["supersedes"] for r in rows}


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


# ── assets (§6.6/6.7: images today, other modalities keyed the same way later) ──

def insert_asset(conn, campaign_id: str, *, file_path: str, modality: str = "image") -> str:
    aid = _id("asset")
    conn.execute(
        "INSERT INTO assets (id, campaign_id, modality, file_path, created_at) VALUES (?,?,?,?,?)",
        (aid, campaign_id, modality, file_path, _now()),
    )
    conn.commit()
    return aid


def set_asset_fingerprint(conn, asset_id: str, phash: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO asset_fingerprints (asset_id, phash, created_at) VALUES (?,?,?)",
        (asset_id, phash, _now()),
    )
    conn.commit()


def get_assets_for_campaign(conn, campaign_id: str) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM assets WHERE campaign_id = ? ORDER BY created_at", (campaign_id,)
    ).fetchall()]


def get_all_fingerprints(conn, *, exclude_campaign_id: Optional[str] = None) -> list[dict]:
    """[{asset_id, campaign_id, phash}] for a provenance check, optionally excluding one
    campaign's own assets (so an image doesn't "match" itself)."""
    sql = """SELECT af.asset_id AS asset_id, a.campaign_id AS campaign_id, af.phash AS phash
             FROM asset_fingerprints af JOIN assets a ON a.id = af.asset_id"""
    params: list = []
    if exclude_campaign_id:
        sql += " WHERE a.campaign_id != ?"
        params.append(exclude_campaign_id)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def mark_asset_embedded(conn, asset_id: str) -> None:
    conn.execute("UPDATE assets SET embedded = 1 WHERE id = ?", (asset_id,))
    conn.commit()


def list_assets(conn, *, campaign_ids: Optional[list[str]] = None,
               exclude_campaign_id: Optional[str] = None) -> list[dict]:
    """[{id, campaign_id, ...}] for CLIP similarity search (§6.6) — either restricted to a
    filtered candidate set of campaigns, or all assets excluding one campaign's own."""
    if campaign_ids is not None:
        if not campaign_ids:
            return []
        placeholders = ",".join("?" * len(campaign_ids))
        rows = conn.execute(
            f"SELECT * FROM assets WHERE campaign_id IN ({placeholders})", campaign_ids
        ).fetchall()
    elif exclude_campaign_id:
        rows = conn.execute(
            "SELECT * FROM assets WHERE campaign_id != ?", (exclude_campaign_id,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM assets").fetchall()
    return [dict(r) for r in rows]


# ── metrics ──────────────────────────────────────────────────────────────────

_VALID_METRIC_TYPES = {"actual", "predicted"}


def add_metrics(conn, campaign_id: str, *, detail=None, structured=None,
                metric_type: str = "actual") -> str:
    metric_type = str(metric_type).strip().lower()
    if metric_type not in _VALID_METRIC_TYPES:
        raise ValueError(f"invalid metric_type {metric_type!r}, "
                         f"must be one of {sorted(_VALID_METRIC_TYPES)}")
    mid = _id("met")
    conn.execute(
        """INSERT INTO metrics (id, campaign_id, metric_type, detail, structured, created_at)
           VALUES (?,?,?,?,?,?)""",
        (mid, campaign_id, metric_type, detail,
         json.dumps(structured) if structured is not None else None, _now()),
    )
    conn.commit()
    return mid


def bulk_import_metrics(conn, rows: list[dict]) -> dict:
    """
    Load a KPI workbook in one call (§6.5) instead of one add_metrics per row. Each row
    identifies its campaign by `campaign_id` (preferred) or `title` (exact, case-insensitive,
    excluding superseded campaigns so a corrected re-upload resolves uniquely — ambiguous or
    missing matches are reported as errors, never guessed) and carries
    `detail`/`structured`/`metric_type` like add_metrics. Every row is processed
    independently: a malformed row (wrong shape, bad metric_type, a nonexistent id) is
    reported in `errors` and never crashes or blocks the rest of the batch.
    """
    imported, errors = 0, []
    for i, row in enumerate(rows):
        try:
            if not isinstance(row, dict):
                errors.append({"row": i, "reason": f"row must be an object, got {type(row).__name__}"})
                continue

            metric_type = str(row.get("metric_type", "actual")).strip().lower()
            if metric_type not in _VALID_METRIC_TYPES:
                errors.append({"row": i, "reason": f"invalid metric_type {metric_type!r}, "
                                                    f"must be one of {sorted(_VALID_METRIC_TYPES)}"})
                continue

            cid = row.get("campaign_id")
            if not cid:
                title = row.get("title")
                if not title:
                    errors.append({"row": i, "reason": "neither campaign_id nor title given"})
                    continue
                matches = conn.execute(
                    "SELECT id FROM campaigns WHERE LOWER(title) = LOWER(?) "
                    "AND id NOT IN (SELECT supersedes FROM campaigns WHERE supersedes IS NOT NULL)",
                    (title,),
                ).fetchall()
                if not matches:
                    errors.append({"row": i, "reason": f"no campaign titled {title!r} found"})
                    continue
                if len(matches) > 1:
                    errors.append({"row": i, "reason": f"title {title!r} is ambiguous ({len(matches)} matches)"})
                    continue
                cid = matches[0]["id"]
            elif get_campaign(conn, cid) is None:
                errors.append({"row": i, "reason": f"campaign_id {cid!r} not found"})
                continue

            add_metrics(conn, cid, detail=row.get("detail"), structured=row.get("structured"),
                       metric_type=metric_type)
            imported += 1
        except Exception as exc:
            errors.append({"row": i, "reason": str(exc)})

    return {"imported": imported, "errors": errors}


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
