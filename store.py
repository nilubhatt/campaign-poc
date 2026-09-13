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
import enums
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
    commentary_checked INTEGER NOT NULL DEFAULT 0,  -- was a file actually read for comments
                                    -- and speaker notes? (§5.3/D17) A warning lives for one
                                    -- response; this is what lets gaps() still report months
                                    -- later that a deck's commentary was never looked at
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
    kind          TEXT NOT NULL DEFAULT 'body',   -- body | commentary (§2.5, defect 08):
                                    -- commentary is the internal reaction to the work, not
                                    -- the work itself, so retrieval can weigh or exclude it
    source        TEXT,            -- JSON {kind, author, date, anchor} for commentary: who
                                    -- said it, when, and which page or slide it sits on -
                                    -- the deck body carries none of those
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
    verdict       TEXT,            -- approve | revise | reject
    summary       TEXT,            -- one line a person can act on (<= 240 chars)
    findings      TEXT,            -- JSON array: severity/category/finding/detail/precedent/fix
    resolved      TEXT,            -- JSON array of {was, now} - what a later version fixed
    closest_precedent TEXT,        -- JSON {campaign_id, layer, quote?, similarity?}
    approve_if    TEXT,            -- the testable change that would flip revise -> approve (§6.5)
    evidence      TEXT,            -- JSON: how much precedent this rests on (§6.6)
    provenance    TEXT,            -- JSON: rulebook/model/server versions behind it (§7.6)
    analysis      TEXT,            -- pre-§2.4 free-text judgment; never written any more,
                                   -- kept so upgraded databases stay readable
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


def _columns(conn: sqlite3.Connection, table: str) -> dict[str, Any]:
    """Column rows by name, or {} for a table that does not exist.

    Every real caller reaches here through init_db, which runs _SCHEMA first, so the tables
    are always present — which is exactly why the assumption went unnoticed until a test
    built a partial legacy database and this crashed on a table the migration has nothing
    to say about. Creating tables is _SCHEMA's job; a migration only ever adjusts one that
    is already there."""
    return {r["name"]: r for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Additive migrations for columns added after a release — CREATE TABLE IF NOT EXISTS
    (in _SCHEMA) does nothing for an EXISTING table, so a column added since someone's last
    install (e.g. `collection`, added after v0.2.0 shipped with installers) would otherwise
    make every insert/read against their real database fail outright (reviewed and
    reproduced against a pre-collection schema). Never destructive — only ever adds a column
    (nullable, or NOT NULL with a constant default — SQLite allows the latter on ADD COLUMN,
    backfilling existing rows) to an existing table; a fresh DB already has it via _SCHEMA."""
    existing = _columns(conn, "campaigns")
    if existing and "collection" not in existing:
        conn.execute("ALTER TABLE campaigns ADD COLUMN collection TEXT")
    if existing and "markets" not in existing:
        conn.execute("ALTER TABLE campaigns ADD COLUMN markets TEXT NOT NULL DEFAULT '[]'")
    evaluation_columns = {r["name"]: r for r in _columns(conn, "evaluations").values()}
    for column in ("verdict", "summary", "findings", "resolved", "closest_precedent",
                   "approve_if", "evidence", "provenance"):
        if evaluation_columns and column not in evaluation_columns:
            conn.execute(f"ALTER TABLE evaluations ADD COLUMN {column} TEXT")
    # The one migration ADD COLUMN cannot do. Before §2.4 a judgment was one required
    # free-text `analysis`; it is now optional and structured, and SQLite has no way to
    # relax a NOT NULL in place — so on every database that already existed, the added
    # columns arrived and then every new save died on `NOT NULL constraint failed:
    # evaluations.analysis`. IntegrityError is not ValueError, so the marketer saw
    # "Error executing tool save_evaluation" with the reason discarded. Both reviewers
    # reproduced it independently; it was invisible to the whole suite because every test
    # starts from a fresh _SCHEMA. Rebuild copies the legacy essays across untouched —
    # they are the record of what the library was told, and the evidence for defect 07.
    if existing and "commentary_checked" not in existing:
        conn.execute("ALTER TABLE campaigns ADD COLUMN commentary_checked INTEGER NOT NULL "
                     "DEFAULT 0")
    chunk_columns = _columns(conn, "campaign_chunks")
    if chunk_columns and "kind" not in chunk_columns:
        # Existing chunks are all deck body — the only kind that existed before §2.5.
        conn.execute("ALTER TABLE campaign_chunks ADD COLUMN kind TEXT NOT NULL "
                     "DEFAULT 'body'")
    if chunk_columns and "source" not in chunk_columns:
        conn.execute("ALTER TABLE campaign_chunks ADD COLUMN source TEXT")
    # After the chunk columns exist, not before: the backfill reads `kind`, which the block
    # above is what adds. Without it every record read since §2.5 reads as never-read — on
    # the reviewer's own database first. Derivable: a campaign with commentary chunks was, by
    # definition, read. A record with a deck and no commentary chunks stays 0, because "read
    # and found none" and "never read" are genuinely indistinguishable from here.
    if _columns(conn, "campaign_chunks") and _columns(conn, "campaigns"):
        conn.execute("""UPDATE campaigns SET commentary_checked = 1
                        WHERE commentary_checked = 0 AND id IN (
                            SELECT DISTINCT campaign_id FROM campaign_chunks
                            WHERE kind = 'commentary')""")
    legacy_analysis = evaluation_columns.get("analysis")
    if legacy_analysis is not None and legacy_analysis["notnull"]:
        _rebuild_evaluations(conn)
    conn.commit()


def _rebuild_evaluations(conn: sqlite3.Connection) -> None:
    """Copy → drop → rename, carrying every existing row. Foreign keys are deferred for the
    swap so `reconciliations.evaluation_id` does not cascade the rows away with the table."""
    columns = [r["name"] for r in conn.execute("PRAGMA table_info(evaluations)").fetchall()]
    # Take the target DDL from a scratch database built by _SCHEMA itself rather than
    # parsing _SCHEMA as text (splitting on ";" broke on a semicolon inside a column
    # comment). This way the rebuilt table is by construction the same table a fresh
    # install gets, which is the invariant that actually matters.
    scratch = sqlite3.connect(":memory:")
    scratch.executescript(_SCHEMA)
    new_schema = scratch.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='evaluations'").fetchone()[0]
    scratch.close()
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("DROP TABLE IF EXISTS evaluations_rebuilt")
        conn.execute(new_schema.replace("evaluations", "evaluations_rebuilt", 1))
        kept = [c for c in columns if c in
                {r["name"] for r in
                 conn.execute("PRAGMA table_info(evaluations_rebuilt)").fetchall()}]
        joined = ", ".join(kept)
        conn.execute(f"INSERT INTO evaluations_rebuilt ({joined}) "
                     f"SELECT {joined} FROM evaluations")
        conn.execute("DROP TABLE evaluations")
        conn.execute("ALTER TABLE evaluations_rebuilt RENAME TO evaluations")
        conn.execute("CREATE INDEX IF NOT EXISTS evals_campaign_idx ON evaluations(campaign_id)")
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


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

# Tuples, not sets: these are shown to a person, and set iteration order is arbitrary, so
# the same error could list the options differently on two runs.
VALID_RECORD_TYPES = ("campaign", "reference", "stub")
VALID_STATUSES = ("proposed", "in_flight", "concluded")
VALID_TAG_SOURCES = ("verified", "stated")
_VALID_RECORD_TYPES = VALID_RECORD_TYPES          # older internal names, kept
_VALID_STATUSES = VALID_STATUSES
_VALID_TAG_SOURCES = VALID_TAG_SOURCES


def _normalise_record_type(value):
    return enums.normalise(value, field="record_type", valid=VALID_RECORD_TYPES,
                           synonyms=enums.RECORD_TYPE_SYNONYMS, allow_none=False)


def _normalise_status(value):
    return enums.normalise(value, field="status", valid=VALID_STATUSES,
                           synonyms=enums.STATUS_SYNONYMS)


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
    # One tag, given as itself. "A bare string where a list was required" was one of the
    # three rejections idea A was written about, and only the FILTER side had been fixed —
    # writing `tags="liked"` still failed with "Input should be a valid list".
    if isinstance(tags, (str, dict)):
        tags = [tags]
    if not isinstance(tags, (list, tuple)):
        raise ValueError(f"tags must be a list of strings or {{value, source}} objects, "
                         f"got {type(tags).__name__}")
    out = []
    for t in tags:
        if isinstance(t, str):
            value, source = t.strip(), "stated"
        elif isinstance(t, dict):
            raw_value = t.get("value")
            if not isinstance(raw_value, str) or not raw_value.strip():
                raise ValueError(f"tag object must have a non-empty string 'value', got {t!r}")
            value = raw_value.strip()
            source = enums.normalise(t.get("source", "stated"), field="tag 'source'",
                                     valid=VALID_TAG_SOURCES,
                                     synonyms=enums.TAG_SOURCE_SYNONYMS,
                                     allow_none=False)
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
            # Normalised exactly as on the way in: a value the library accepted must be
            # usable to search for itself, or the forgiveness is a trap.
            source = enums.normalise(t.get("source"), field="tag 'source'",
                                     valid=VALID_TAG_SOURCES,
                                     synonyms=enums.TAG_SOURCE_SYNONYMS)
        else:
            raise ValueError(f"tag query entry must be a string or {{value, source}} object, got {t!r}")
        if not value:
            raise ValueError(f"tag query value cannot be empty or whitespace-only, got {t!r}")
        out.append((value.lower(), source))
    return out


def insert_campaign(conn, *, title, record_type="campaign", status=None, detail=None,
                    deck_text=None, asset_path=None, tags=None, region=None, market=None,
                    markets=None, collection=None, supersedes=None,
                    commentary_checked=False) -> str:
    record_type = _normalise_record_type(record_type)
    status = _normalise_status(status)
    tags = normalize_tags(tags, has_actual_metrics=False)  # brand-new: no metrics can exist yet
    markets = normalize_markets(markets)
    cid = _id("camp")
    now = _now()
    if status is None and record_type == "campaign":
        status = "concluded"  # reference/stub records have no lifecycle status by default
    conn.execute(
        """INSERT INTO campaigns (id, title, record_type, status, tags, region, market,
                                  markets, collection, supersedes, detail, deck_text,
                                  asset_path, commentary_checked, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cid, title, record_type, status, json.dumps(tags), region, market,
         json.dumps(markets), collection, supersedes, detail, deck_text, asset_path,
         1 if commentary_checked else 0, now, now),
    )
    conn.commit()
    return cid


def mark_embedded(conn, campaign_id: str, flag: bool) -> None:
    """Flag whether a campaign is FULLY embedded, i.e. searchable in its entirety.

    This used to mean "has at least one embedded chunk", which was merely imprecise while
    partial embedding was an accident. Item 2.1 made partial a designed outcome (a time
    budget stops mid-deck), so "at least one" would have reported a deck with 2 of 12
    sections indexed as searchable — the stored-versus-searchable conflation defect 05 is
    about. The counts in list_campaigns tell the partial story."""
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
    d["commentary"] = get_commentary(conn, campaign_id)
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
    # Same counts list_campaigns reports, so a caller does not get a different answer about
    # the same record depending on which tool it asked.
    d["assets_total"] = len(d["assets"])
    d["assets_embedded"] = sum(1 for a in d["assets"] if a["embedded"])
    return d


def list_campaigns(conn, *, record_type: Optional[str] = None,
                   status: Optional[str] = None) -> list[dict]:
    clauses, params = [], []
    if record_type:
        clauses.append("record_type = ?")
        params.append(_normalise_record_type(record_type))
    if status:
        clauses.append("status = ?")
        params.append(_normalise_status(status))
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

    # Counts, not just the rollup flag: "stored" and "searchable" are different states, and
    # a time-budgeted ingest can legitimately leave a campaign between them. A caller that
    # only sees `embedded` cannot tell a fully indexed deck from one with two of twelve
    # sections searchable (defect 05).
    chunk_counts: dict[str, tuple[int, int]] = {}
    asset_counts: dict[str, tuple[int, int]] = {}
    if ids:
        placeholders = ",".join("?" * len(ids))
        for table, sink in (("campaign_chunks", chunk_counts), ("assets", asset_counts)):
            for row in conn.execute(
                f"""SELECT campaign_id, COUNT(*) AS total,
                           SUM(CASE WHEN embedded THEN 1 ELSE 0 END) AS done
                    FROM {table} WHERE campaign_id IN ({placeholders})
                    GROUP BY campaign_id""", ids).fetchall():
                sink[row["campaign_id"]] = (row["total"], row["done"] or 0)

    out = []
    for r in rows:
        d = dict(r)
        d["tags"] = _parse_stored_tags(d["tags"])
        d["markets"] = _parse_stored_markets(d["markets"])
        d["chunks_total"], d["chunks_embedded"] = chunk_counts.get(d["id"], (0, 0))
        d["assets_total"], d["assets_embedded"] = asset_counts.get(d["id"], (0, 0))
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
        params.append(_normalise_record_type(record_type))
    if status:
        clauses.append("status = ?")
        params.append(_normalise_status(status))
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
                    collection=None, supersedes=None) -> bool:
    """Update campaign metadata (§6.4) — NOT deck_text/chunks/embeddings; re-upload (or
    supersede) for content changes. Only given fields change; tags/markets, if given, fully
    replace the existing list rather than merging. Returns whether the campaign exists.

    `supersedes` is here because §6.3 needs it and 5.2's reviewers needed it: supersession
    could only be declared at upload, so somebody who realised afterwards had no way to say
    so, and a supersession declared BY MISTAKE hid a record from every future search and was
    undoable only by deleting the campaign. Passing `""` clears it — the retraction that did
    not exist is the reason the 5.2 offer was called dangerous rather than merely wrong.
    """
    fields, params = [], []
    if title is not None:
        fields.append("title = ?"); params.append(title)
    if detail is not None:
        fields.append("detail = ?"); params.append(detail)
    if record_type is not None:
        fields.append("record_type = ?"); params.append(_normalise_record_type(record_type))
    if status is not None:
        # A blank normalises to None, which means "not saying" — the same as omitting the
        # argument. Appending it anyway would clear a status that was already set, so a
        # spreadsheet row with an empty cell would silently erase one.
        normalised_status = _normalise_status(status)
        if normalised_status is not None:
            fields.append("status = ?"); params.append(normalised_status)
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
    if supersedes is not None:
        fields.append("supersedes = ?")
        params.append(checked_supersedes(conn, campaign_id, supersedes))

    if not fields:
        return get_campaign(conn, campaign_id) is not None

    fields.append("updated_at = ?")
    params.append(_now())
    params.append(campaign_id)
    cur = conn.execute(f"UPDATE campaigns SET {', '.join(fields)} WHERE id = ?", params)
    conn.commit()
    return cur.rowcount > 0


def checked_supersedes(conn, campaign_id: Optional[str], supersedes) -> Optional[str]:
    """The three ways a supersession can be nonsense, refused before it is written.

    All three hide records from search, which is what makes them worth checking rather than
    accepting: a record pointed at nothing is a link nobody can follow back, a record
    superseding itself removes itself from the library, and a cycle removes both ends and
    would make §6.3's chain walk run forever.

    `campaign_id` is None on the upload path, where the record does not have an id yet: only
    the dangling-pointer check applies there, and it is the one that mattered — the upload
    path had no validation at all, so a typo left {"", "  ", "camp_nope"} in the set of
    superseded ids and a link pointing at nothing.
    """
    target = (supersedes or "").strip()
    if not target:
        return None                       # the retraction
    if campaign_id is not None and target == campaign_id:
        raise ValueError("a campaign cannot supersede itself — that would hide it from "
                         "every search, including its own")
    if conn.execute("SELECT 1 FROM campaigns WHERE id = ?", (target,)).fetchone() is None:
        raise ValueError(f"cannot supersede {target!r}: there is no such record. Supersession "
                         f"hides the superseded record from search, so a wrong id here "
                         f"quietly hides nothing and links nothing")
    seen, walk = ({campaign_id} if campaign_id else set()), target
    while walk:
        if walk in seen:
            raise ValueError(f"that would make a supersession cycle through {walk!r}. Both "
                             f"ends of a cycle are hidden from every search and neither can "
                             f"be reached from the other")
        seen.add(walk)
        row = conn.execute("SELECT supersedes FROM campaigns WHERE id = ?",
                           (walk,)).fetchone()
        walk = row["supersedes"] if row else None
    return target


def supersession_chain(conn, campaign_id: str) -> list[str]:
    """Oldest to newest, following `supersedes` back from this record (§6.3 / D64).

    `diff_campaigns` compared adjacent judgments only, so a finding v2 explicitly resolved
    was reported against v3 as "neither resolved nor repeated" — the library forgetting its
    own evidence and then hedging about it. The visited set is belt and braces: cycles are
    refused on write, but a database that predates that check could still hold one.
    """
    chain, seen, walk = [], set(), campaign_id
    while walk and walk not in seen:
        chain.append(walk)
        seen.add(walk)
        row = conn.execute("SELECT supersedes FROM campaigns WHERE id = ?",
                           (walk,)).fetchone()
        walk = row["supersedes"] if row else None
    chain.reverse()
    return chain


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

def insert_chunks(conn, campaign_id: str, texts: list[str], *, kind: str = "body",
                  sources: Optional[list[dict]] = None) -> list[str]:
    """`sources` carries one {kind, author, date, anchor} per text for commentary, so a
    match can say who said it and where, rather than only that the deck mentions it."""
    now = _now()
    ids = []
    start = conn.execute(
        "SELECT COALESCE(MAX(chunk_index), -1) + 1 FROM campaign_chunks WHERE campaign_id = ?",
        (campaign_id,)).fetchone()[0]
    for i, text in enumerate(texts):
        chid = _id("chunk")
        source = (sources or [None] * len(texts))[i]
        conn.execute(
            """INSERT INTO campaign_chunks (id, campaign_id, chunk_index, text, kind,
                                            source, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (chid, campaign_id, start + i, text, kind,
             json.dumps(source) if source else None, now),
        )
        ids.append(chid)
    conn.commit()
    return ids


def get_commentary(conn, campaign_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT text, source FROM campaign_chunks WHERE campaign_id = ? AND kind = "
        "'commentary' ORDER BY chunk_index", (campaign_id,)).fetchall()
    items = []
    for r in rows:
        item = json.loads(r["source"]) if r["source"] else {}
        item["text"] = r["text"]
        items.append(item)
    return items


def text_on_file(conn, campaign_id: str) -> Optional[dict]:
    """Everything this record actually says, split by layer — or None if there is no record.

    §6.1 verifies a finding's quote against this.

    **The body is the row's own columns, not its chunks**, and that is the whole design.
    Chunks are a derived copy, split by `chunking.pack` at 1800 characters — a boundary the
    model cannot see and the document does not have. Verifying against them refused quotes
    that were verbatim and contiguous in the record, told the model "do not paraphrase", and
    could not be satisfied by re-copying more carefully. They are also STALE: `update_campaign`
    rewrites `title`/`detail` without re-chunking, so chunk 0 keeps the old wording and a
    citation of what the record no longer says verified against it. Both were found in review.
    `detail`/`deck_text` are current by definition and contiguous by construction, and every
    body chunk derives from them (`ingest_campaign` sets `deck_text` from the extracted units
    when the caller did not supply it), so nothing quotable is lost.

    **The title is not in it.** A title is text, but it is not evidence: "the record says
    'Always-on Lima Creator Programme 2026'" supports no finding about anything. The first
    version included it and enforced that judgment only for stubs, which meant the same quote
    was evidence or not depending on whether the record happened to have a brief.

    **Metric detail is in it**, for `metric_type='actual'` rows only. "CTR was 3.2 percent,
    well above the benchmark" is the product's most common real citation, `find_similar` shows it to the model as evidence, and
    the first version refused every quote of it — a metrics-only record was told it had
    "title and numbers and nothing else" when the numbers' own words were exactly what was
    being quoted. It is body rather than commentary because it is the record speaking about
    itself, not somebody's remark about it.

    Commentary has no column — the chunks ARE the storage — so it stays chunk-based, with one
    concession: consecutive pieces carrying the SAME attribution are joined, because a comment
    longer than a chunk is split into pieces that all keep the same author, and a quote across
    that split is one person's sentence. Pieces are joined only when the attribution is
    present and equal: an unattributed run used to collapse into one quotable block because
    `None == None`, which let two strangers' remarks be stitched into a single quotation — the
    misattribution the layer rule exists to stop, one level down. Compared as parsed objects,
    not as JSON text, so key order cannot decide it either way.
    """
    # Named columns are not safe to assume here. A database that predates a release is
    # missing whatever that release added, which is the whole reason `_migrate_schema`
    # exists — and this runs on every save, so a column this function names and an upgraded
    # database does not have would fail every judgment on the machine it matters most on.
    columns = _columns(conn, "campaigns")
    if not columns:
        return None
    row = conn.execute("SELECT * FROM campaigns WHERE id = ?", (campaign_id,)).fetchone()
    if not row:
        return None
    body = [row[field] for field in ("detail", "deck_text")
            if field in columns and row[field]]
    if _columns(conn, "metrics"):
        # ACTUAL only. The case for quoting metrics is "what this campaign achieved" — a
        # forecast is not that, and a finding quoting one under `layer: "body"` would read as
        # the record stating an outcome it only predicted, which is the misattribution the
        # layer rule exists to stop, wearing different clothes.
        body += [r["detail"] for r in conn.execute(
            "SELECT detail FROM metrics WHERE campaign_id = ? AND metric_type = 'actual' "
            "ORDER BY created_at", (campaign_id,)).fetchall() if r["detail"]]

    commentary: list[str] = []
    chunk_columns = _columns(conn, "campaign_chunks")
    # No `kind` column means a database from before §2.5, which had no commentary at all —
    # so an empty list is the right answer, not a reason to skip the query and lose the body
    # (the body no longer comes from here anyway).
    if "kind" in chunk_columns:
        last = None
        for chunk in conn.execute(
                "SELECT text, source FROM campaign_chunks WHERE campaign_id = ? AND "
                "kind = 'commentary' ORDER BY chunk_index", (campaign_id,)).fetchall():
            source = json.loads(chunk["source"]) if chunk["source"] else None
            if source is not None and source == last and commentary:
                commentary[-1] += " " + chunk["text"]
            else:
                commentary.append(chunk["text"])
            last = source

    return {"body": body, "commentary": commentary,
            # Whether there is anything here to quote at all. The caller needs the difference
            # to refuse a stub as the right thing ("there is nothing here to quote") rather
            # than the wrong one ("your quote is not in it"), which would send a model off to
            # reword a quote in a loop with no exit.
            "brief": bool(body or commentary)}


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


def get_unembedded_chunks(conn, campaign_id: Optional[str] = None) -> list[dict]:
    """Chunks that exist but have no vector — stored, not searchable.

    This state used to be an accident (an embedder failure mid-upload) and is now also a
    designed outcome (a time budget stopping mid-deck), which is what makes it worth being
    able to find rather than only to regret."""
    sql = "SELECT id, campaign_id, text FROM campaign_chunks WHERE embedded = 0"
    params: list = []
    if campaign_id:
        sql += " AND campaign_id = ?"
        params.append(campaign_id)
    # Newest first: the deck someone just uploaded is the one they are waiting on, not an
    # eight-month-old backlog. A fixed oldest-first order also meant a permanently failing
    # row was retried at the head of every run, able to consume the whole budget and starve
    # everything behind it.
    return [dict(r) for r in conn.execute(sql + " ORDER BY created_at DESC", params).fetchall()]


def get_unembedded_assets(conn, campaign_id: Optional[str] = None) -> list[dict]:
    """Image assets stored (and usually fingerprinted) but never visually embedded — the
    reviewer's two stranded assets, which had no route back short of re-uploading them."""
    sql = "SELECT id, campaign_id, file_path FROM assets WHERE embedded = 0"
    params: list = []
    if campaign_id:
        sql += " AND campaign_id = ?"
        params.append(campaign_id)
    return [dict(r) for r in conn.execute(sql + " ORDER BY created_at DESC", params).fetchall()]


def get_all_campaign_ids_with_chunks(conn) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT DISTINCT campaign_id FROM campaign_chunks").fetchall()]


def chunk_counts(conn, campaign_id: str) -> dict:
    row = conn.execute(
        """SELECT COUNT(*) AS total, SUM(CASE WHEN embedded THEN 1 ELSE 0 END) AS done
           FROM campaign_chunks WHERE campaign_id = ?""", (campaign_id,)).fetchone()
    return {"total": row["total"] or 0, "embedded": row["done"] or 0}


def count_unembedded(conn, campaign_id: Optional[str] = None) -> dict:
    """COUNT(*), not len(fetch-everything): counting a backlog should not drag every
    chunk's full text out of the database."""
    out = {}
    for key, table in (("chunks", "campaign_chunks"), ("assets", "assets")):
        sql = f"SELECT COUNT(*) AS n FROM {table} WHERE embedded = 0"
        params: list = []
        if campaign_id:
            sql += " AND campaign_id = ?"
            params.append(campaign_id)
        out[key] = conn.execute(sql, params).fetchone()["n"]
    return out


def outstanding_by_campaign(conn, campaign_id: Optional[str] = None) -> list[dict]:
    """Which records are unfinished, by name — "212 remaining" means nothing to a marketer;
    "your Mexico deck is done, two older records have 40 sections left" does."""
    sql = """
        SELECT c.id AS campaign_id, c.title AS title,
               (SELECT COUNT(*) FROM campaign_chunks ch
                 WHERE ch.campaign_id = c.id AND ch.embedded = 0) AS sections_left,
               (SELECT COUNT(*) FROM assets a
                 WHERE a.campaign_id = c.id AND a.embedded = 0) AS images_left
        FROM campaigns c
    """
    params: list = []
    if campaign_id:
        sql += " WHERE c.id = ?"
        params.append(campaign_id)
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    return sorted([r for r in rows if r["sections_left"] or r["images_left"]],
                  key=lambda r: -(r["sections_left"] + r["images_left"]))


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

VALID_METRIC_TYPES = ("actual", "predicted")
_VALID_METRIC_TYPES = VALID_METRIC_TYPES


def add_metrics(conn, campaign_id: str, *, detail=None, structured=None,
                metric_type: str = "actual", commit: bool = True) -> str:
    """commit=False lets a bulk import batch many rows into one transaction — a commit per
    row is an fsync per row, which the caller's row count controls."""
    metric_type = enums.normalise(metric_type, field="metric_type",
                                  valid=VALID_METRIC_TYPES,
                                  synonyms=enums.METRIC_TYPE_SYNONYMS, allow_none=False)
    # A row measuring nothing is not a measurement. It mattered because an empty `actual`
    # row satisfies the gate that lets a performance tag be marked `verified` — so a
    # content-free row bought the provenance this library weighs judgments by.
    if not (detail and str(detail).strip()) and not structured:
        raise ValueError("a metric needs `detail` (what was measured, in words) or "
                         "`structured` (the numbers), or both — an empty row records no "
                         "outcome but still counts as one")
    mid = _id("met")
    conn.execute(
        """INSERT INTO metrics (id, campaign_id, metric_type, detail, structured, created_at)
           VALUES (?,?,?,?,?,?)""",
        (mid, campaign_id, metric_type, detail,
         json.dumps(structured) if structured is not None else None, _now()),
    )
    if commit:
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
    # Bounded like every other handler (defect 04's sweep): the caller decides how many rows
    # this is, and every row previously committed on its own — one fsync each, which is
    # cheap on an SSD and much less so on the customer's Windows laptop behind AV scanning.
    # One transaction for the batch, and stop at the same budget everything else respects.
    deadline = time.monotonic() + config.TOOL_TIME_BUDGET_SECONDS
    not_processed = 0
    for i, row in enumerate(rows):
        if time.monotonic() >= deadline:
            not_processed = len(rows) - i
            break
        try:
            if not isinstance(row, dict):
                errors.append({"row": i, "reason": f"row must be an object, got {type(row).__name__}"})
                continue

            # The same normaliser as a single write (§5.1). A spreadsheet column headed
            # "Results" or "Target" is the single most likely place these words arrive, and
            # this was the one path still using the old strict check — the same vocabulary
            # with two behaviours depending on how many rows you sent.
            try:
                metric_type = enums.normalise(row.get("metric_type", "actual"),
                                              field="metric_type",
                                              valid=VALID_METRIC_TYPES,
                                              synonyms=enums.METRIC_TYPE_SYNONYMS,
                                              allow_none=False)
            except ValueError as exc:
                errors.append({"row": i, "reason": str(exc)})
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
                       metric_type=metric_type, commit=False)
            imported += 1
        except Exception as exc:
            errors.append({"row": i, "reason": str(exc)})

    conn.commit()
    result = {"imported": imported, "errors": errors, "not_processed": not_processed}
    if not_processed:
        result["note"] = (
            f"imported {imported} rows before the {config.TOOL_TIME_BUDGET_SECONDS:g}s time "
            f"budget ran out; {not_processed} rows were not processed. Send them again to "
            f"continue — nothing already imported is duplicated by doing so."
        )
    return result


# ── evaluations ──────────────────────────────────────────────────────────────

def next_evaluation_id() -> str:
    """Allocated before the insert so the findings can carry ids derived from it."""
    return _id("eval")


def insert_evaluation(conn, *, subject_title, verdict, summary, findings,
                      evaluation_id=None,
                      resolved=None, closest_precedent=None, approve_if=None,
                      evidence=None, provenance=None, campaign_id=None,
                      cited_ids=None, predictions=None) -> str:
    eid = evaluation_id or _id("eval")
    conn.execute(
        """INSERT INTO evaluations (id, campaign_id, subject_title, cited_ids, verdict,
                                    summary, findings, resolved, closest_precedent,
                                    approve_if, evidence, provenance, predictions,
                                    created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (eid, campaign_id, subject_title, json.dumps(cited_ids or []), verdict, summary,
         json.dumps(findings), json.dumps(resolved or []),
         json.dumps(closest_precedent) if closest_precedent else None, approve_if,
         json.dumps(evidence) if evidence else None,
         json.dumps(provenance) if provenance else None,
         json.dumps(predictions) if predictions is not None else None, _now()),
    )
    conn.commit()
    return eid


def _parse_evaluation(row) -> dict:
    d = dict(row)
    for key, default in (("findings", []), ("resolved", []), ("cited_ids", [])):
        d[key] = json.loads(d[key]) if d.get(key) else default
    for key in ("closest_precedent", "predictions", "evidence", "provenance"):
        d[key] = json.loads(d[key]) if d.get(key) else None
    return d


def get_evaluation(conn, evaluation_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)).fetchone()
    if not row:
        return None
    d = _parse_evaluation(row)
    d["reconciliations"] = [dict(r) for r in conn.execute(
        "SELECT * FROM reconciliations WHERE evaluation_id = ? ORDER BY created_at", (evaluation_id,)
    ).fetchall()]
    return d


def campaigns_with_actual_metrics(conn) -> set:
    """Ids of campaigns with at least one MEASURED outcome.

    Distinct from `has_metrics`, which counts any row including a `predicted` forecast — and
    a forecast is the opposite of a measured outcome, being the thing reconciliation later
    scores against the actuals (§5.3).

    Empty rather than an exception on a database with no `metrics` table. Guarding here
    rather than at each call site because three save-path functions have now crashed on an
    upgraded v0.2.0 database in turn (§6.1's `text_on_file`, §6.4's disconfirming search,
    §6.6's strength line) and §5.3's `missing_input_for_citations` was doing it too, unnoticed,
    because the legacy tests happened to cite nothing. A shared reader is the right place to
    stop a shared failure.
    """
    if not _columns(conn, "metrics"):
        return set()
    return {r["campaign_id"] for r in conn.execute(
        "SELECT DISTINCT campaign_id FROM metrics WHERE metric_type = 'actual'").fetchall()}


def finding_exists(conn, finding_id: str) -> bool:
    """Whether any stored judgment contains a finding with this id.

    Finding ids are `{evaluation_id}#n`, so the evaluation is identifiable from the id
    itself — no scan required.
    """
    evaluation_id = str(finding_id).split("#")[0]
    row = conn.execute("SELECT findings FROM evaluations WHERE id = ?",
                       (evaluation_id,)).fetchone()
    if not row or not row["findings"]:
        return False
    return any(f.get("id") == finding_id for f in json.loads(row["findings"]))


def latest_evaluation_for_campaign(conn, campaign_id: str):
    """The most recent judgment of this campaign, or None.

    A version can be judged more than once — that is what a revised evaluation is — and a
    comparison against an older one reports corrections that were already dealt with (§5.4).
    """
    row = conn.execute(
        "SELECT id FROM evaluations WHERE campaign_id = ? ORDER BY created_at DESC, "
        "rowid DESC LIMIT 1", (campaign_id,)).fetchone()
    return get_evaluation(conn, row["id"]) if row else None


def unreconciled_evaluation_id(conn, campaign_id: str) -> Optional[str]:
    """The most recent judgment about this campaign that has never been compared with its
    results, or None. Used to decide whether recording results is worth offering to close a
    loop with (§5.2) — an offer made when there is no open judgment is the standing kind
    that stops being read."""
    row = conn.execute(
        """SELECT e.id FROM evaluations e
           WHERE e.campaign_id = ?
             AND NOT EXISTS (SELECT 1 FROM reconciliations r WHERE r.evaluation_id = e.id)
           ORDER BY e.created_at DESC LIMIT 1""", (campaign_id,)).fetchone()
    return row["id"] if row else None


def citations(conn) -> list[list[str]]:
    """Every judgment's `cited_ids`, one list per judgment (D65 / §6.6).

    Its own query rather than a wider `list_evaluations`: that function feeds a browsing
    surface and returns what a person reads, and adding a field nothing there displays would
    make it carry two jobs. This is a fact about citation traffic, and only one caller wants
    it.
    """
    out = []
    for row in conn.execute("SELECT cited_ids FROM evaluations").fetchall():
        try:
            cited = json.loads(row["cited_ids"]) if row["cited_ids"] else []
        except (TypeError, ValueError):
            continue
        out.append([c for c in cited if isinstance(c, str)])
    return out


def list_evaluations(conn) -> list[dict]:
    """Titles and dates alone cannot answer "which of these still need work" — which is the
    one question a list of judgments exists to answer. Counts come from the stored findings
    rather than a second column, so they cannot drift from them."""
    rows = []
    for r in conn.execute("SELECT id, campaign_id, subject_title, verdict, findings, "
                          "created_at FROM evaluations ORDER BY created_at DESC").fetchall():
        d = dict(r)
        raw = d.pop("findings")          # popped unconditionally: a conditional pop left
        findings = json.loads(raw) if raw else []   # `findings: null` on every legacy row
        d["counts"] = {level: sum(1 for f in findings if f.get("severity") == level)
                       for level in ("blocking", "should_fix", "note")}
        # A row with no verdict is a judgment written before the structured schema, not a
        # broken one — say which, or a reader has to guess from a null.
        if d["verdict"] is None:
            d["schema"] = "legacy"
        rows.append(d)
    return rows


# ── reconciliations ──────────────────────────────────────────────────────────

def insert_reconciliation(conn, *, evaluation_id, comparison, actual=None) -> str:
    rid = _id("recon")
    conn.execute(
        "INSERT INTO reconciliations (id, evaluation_id, actual, comparison, created_at) VALUES (?,?,?,?,?)",
        (rid, evaluation_id, actual, comparison, _now()),
    )
    conn.commit()
    return rid
