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
    -- §9.1: what this image IS. `proposed` is creative lifted from a brief — what somebody
    -- intends to run; `delivered` is a photograph that came back after the event — evidence
    -- of what ran. "A library that learns from briefs while measuring executions is learning
    -- from the wrong document, and has no way to notice", and this is the field that lets it
    -- notice. Everything in §9.2-9.5 falls out of it.
    --
    -- The DEFAULT is the load-bearing part: every asset on file today came out of a deck, so
    -- an upgraded database has to read as `proposed`. Defaulting the other way would have
    -- §9.2 compare a library of briefs against itself and report zero drift with total
    -- confidence.
    phase         TEXT NOT NULL DEFAULT 'proposed',   -- proposed | delivered
    -- When the photograph was taken, which is the asset's own fact and not the row's: a deck
    -- uploaded in March can carry photos shot in January, and `created_at` records when this
    -- library was told.
    captured_on   TEXT,
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
    metric_type   TEXT NOT NULL DEFAULT 'actual',  -- actual | predicted | target. A target
                                   -- is what somebody aimed at, NOT what this library expects
                                   -- to happen: reconciliation scores itself against its
                                   -- predictions, and scoring it against an ambition instead
                                   -- would make every calibration figure meaningless (§8.1)
    detail        TEXT,            -- freeform metrics / learnings, as given
    structured    TEXT,            -- optional JSON {ctr, roi, conversions, ...}
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS metric_registry (
    canonical     TEXT PRIMARY KEY,      -- §8.1: one name per measure
    display_name  TEXT NOT NULL,
    unit          TEXT,                  -- percent | ratio | currency | count | people
    direction     TEXT,                  -- higher_is_better | lower_is_better | NULL
    aliases       TEXT NOT NULL DEFAULT '[]',  -- JSON: every spelling seen for this measure
    status        TEXT NOT NULL DEFAULT 'provisional',  -- provisional | known | expected
                                                        -- | retired | ignored
    -- `status` is where a measure sits in the VOCABULARY; `expected_in` is which checklists it
    -- is on. Two different questions, and collapsing them is what would make a shipped name
    -- like `cpm` a standing requirement for every brief on day one (§8.3).
    expected_in   TEXT NOT NULL DEFAULT '[]',  -- markets where it graduated; [] = no checklist
    confirmed_by  TEXT,                  -- "confirmed once by a PERSON" — a standing
                                         -- requirement nobody's name is against is one nobody
                                         -- can question later
    confirmed_at  REAL,
    retired_at    REAL,                  -- §8.5 demotes and NEVER deletes: "we used to track
                                         -- this" is an answer, "we never did" is a lie
    surfaced      INTEGER NOT NULL DEFAULT 0,  -- §8.2 asks ONCE; this is what makes that true
    offered       INTEGER NOT NULL DEFAULT 0,  -- §8.3's graduation question, asked once too
    answered      INTEGER NOT NULL DEFAULT 0,  -- somebody decided, so never ask again — even
                                               -- when the decision was "ignore", because
                                               -- re-asking a declined question teaches people
                                               -- to dismiss the product
    first_seen    REAL,
    last_seen     REAL,
    times_seen    INTEGER NOT NULL DEFAULT 0,
    markets       TEXT NOT NULL DEFAULT '[]'   -- which markets it has appeared in (§8.3)
);
CREATE TABLE IF NOT EXISTS execution_drift (
    -- §9.5: the drift figure, stored ON the outcome. "`performed_well` at low drift and
    -- `performed_well` at high drift stop looking identical — the first is evidence the brief
    -- was good, the second is evidence something was good and the brief may not have been it."
    --
    -- REWRITTEN whenever the answer changes: results arriving, photographs arriving, a
    -- classification being made. It is a cache of the current reading, not a snapshot of what a
    -- past judgment rested on — an earlier comment here claimed the latter, which stopped being
    -- true the moment there were three call sites, and a false claim carrying the server's
    -- authority is the failure this product is written against.
    --
    -- A saved verdict therefore does NOT read its precedents' drift from here: `save_evaluation`
    -- stamps the figure for every cited campaign into the evaluation's own `evidence`
    -- (`execution_at_save`), the same way §9.4 stamps `outcome_known` at the moment of a
    -- judgment. Otherwise a verdict written when a cited campaign read `never_checked` is read
    -- back beside a row that now says `drifted`, with nothing recording that it never saw it.
    campaign_id   TEXT PRIMARY KEY REFERENCES campaigns(id) ON DELETE CASCADE,
    score         REAL,                  -- NULL when nothing was ever checked
    relative      REAL,
    counts        TEXT NOT NULL DEFAULT '{}',   -- JSON: as_briefed / never_appeared / new / …
    classified    TEXT NOT NULL DEFAULT '{}',   -- JSON: improvement / neutral / degradation …
    items         TEXT NOT NULL DEFAULT '[]',   -- JSON: the classified items themselves —
                                  -- §9.5 asks for "the score, the counts, and the classified
                                  -- items", and counts alone cannot say WHAT was judged good
    status        TEXT NOT NULL,         -- as_briefed | drifted | never_checked
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS drift_classifications (
    -- §9.4: what somebody made of one piece of execution drift. APPENDED, never updated — a
    -- reading made before the numbers came in and one made after are two judgments about the
    -- same thing, and the pair is worth more than either.
    id            TEXT PRIMARY KEY,
    campaign_id   TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    -- WHAT this is about, as a stable id: the asset a briefed image or delivered photograph
    -- has, or the commitment a promise has. It was the file path, which `_keep_asset` sets to
    -- a uuid on purpose — so the offer read "say whether a3f9c21d88e04b17.png not matching was
    -- a loss", the stored record was unreadable six months later, and a person answering in
    -- their own words created a DIFFERENT item, so the offer re-fired on the same uuid forever.
    subject       TEXT NOT NULL,
    -- And the same thing in words somebody can read: the promise the brief made, or what the
    -- image was. A judgment nobody can read back is one nobody can act on.
    item          TEXT NOT NULL,
    classification TEXT NOT NULL,        -- improvement | neutral | degradation | too_early
    why           TEXT NOT NULL,
    classified_by TEXT NOT NULL,
    -- Whether a measured outcome existed WHEN THIS WAS JUDGED. "Usually needs the outcome to
    -- settle it" — so a reader can weigh "this looked like an improvement" against "this was
    -- an improvement" without reconstructing which one it was.
    outcome_known INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS commitments (
    -- §9.3: the specific, checkable promises a brief makes. "A claw machine loaded with
    -- branded merchandise" is a thing somebody can look for in the photographs; "a premium
    -- feel" is not.
    id            TEXT PRIMARY KEY,
    campaign_id   TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    text          TEXT NOT NULL,         -- the promise, in the brief's own words
    -- The line it was taken from. That a line IS IN THE DECK is a fact and the quote is what
    -- makes it checkable; that the line is a COMMITMENT rather than a passing mention is a
    -- reading, and keeping the source is what lets somebody disagree with the reading.
    source_line   TEXT NOT NULL,
    origin        TEXT NOT NULL DEFAULT 'extracted',   -- extracted | added
    status        TEXT NOT NULL DEFAULT 'open',        -- open | dropped
    why_dropped   TEXT,
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS corrections (
    -- §8.6: client feedback on the same loop as a measure. The lifecycle columns are
    -- deliberately the same names as `metric_registry`'s, because they are the same loop —
    -- `learning.gate` reads both and a second vocabulary here would be a second mechanism.
    id            TEXT PRIMARY KEY,
    text          TEXT NOT NULL,         -- the rule, as somebody said it
    normalised    TEXT NOT NULL,         -- what makes "the same correction" the same one
    status        TEXT NOT NULL DEFAULT 'provisional',
    -- provisional | expected | retired | ignored | merged. `merged` is this table's own: the
    -- row stays because it is somebody's words, and `merged_into` says where its sightings
    -- went.
    merged_into   TEXT,
    expected_in   TEXT NOT NULL DEFAULT '[]',  -- markets where it graduated; [] = no checklist
    confirmed_by  TEXT,
    confirmed_at  REAL,
    retired_at    REAL,
    offered       INTEGER NOT NULL DEFAULT 0,  -- the graduation question, asked once
    asked         INTEGER NOT NULL DEFAULT 0,  -- the "is this the same rule?" question, once
    quiet_asked   INTEGER NOT NULL DEFAULT 0,  -- the "still current?" question, once
    first_seen    REAL,
    last_seen     REAL,
    times_seen    INTEGER NOT NULL DEFAULT 0,
    markets       TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS correction_sightings (
    -- One row per mention, because "Peru (30 influencers, praised); instructed independently
    -- in Australia" is ONE correction with TWO origins, and the second is what makes it more
    -- than a house style. A single provenance column would have to overwrite one of them.
    id            TEXT PRIMARY KEY,
    correction_id TEXT NOT NULL REFERENCES corrections(id) ON DELETE CASCADE,
    campaign_id   TEXT REFERENCES campaigns(id) ON DELETE SET NULL,
    provenance    TEXT NOT NULL,         -- "JD SEA slide 23, named by the client as the
                                         -- standard every brief should follow"
    said_as       TEXT,                  -- this market's OWN words. The correction row keeps
                                         -- the canonical wording; a paraphrase folded into it
                                         -- keeps what was actually said — the same
                                         -- canonical/raw_key split §8.1 made for measures,
                                         -- and what makes the fold checkable afterwards
    noted_at      REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS metric_values (
    id            TEXT PRIMARY KEY,
    campaign_id   TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    metric        TEXT NOT NULL,         -- the canonical name
    raw_key       TEXT NOT NULL,         -- what the caller actually wrote; canonicalising is
                                          -- a claim about what they meant, and keeping this
                                          -- is what makes the claim checkable
    value         REAL NOT NULL,         -- TYPED: "show me every ROAS" needs a number column
    unit          TEXT,
    source        TEXT NOT NULL DEFAULT 'stated',  -- stated | recomputed. §8.1's sharpest
                                          -- finding: a correction had nowhere to live and
                                          -- became a key name
    scope         TEXT,                  -- upper_funnel, organic, … — out of the key
    metric_type   TEXT NOT NULL DEFAULT 'actual',
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
CREATE TABLE IF NOT EXISTS write_refusals (
    id            TEXT PRIMARY KEY,
    reason        TEXT NOT NULL,   -- a stable code, not the message (§7.6 / D81)
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS reconciliations (
    id            TEXT PRIMARY KEY,
    evaluation_id TEXT NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    actual        TEXT,            -- the real post-conclusion metrics (freeform + optional structured)
    comparison    TEXT NOT NULL,   -- Claude's prediction-vs-actual reconciliation + lesson learned
    basis         TEXT,            -- results | superseding_version (D85): §6.3 made the
                                    -- version-based reconciliation the common case, so
                                    -- "v2 shows the structure came back" now lands in the
                                    -- same `actual` column as a CTR figure, and anything
                                    -- computing calibration would read both as outcome data
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS retrievals (
    id            TEXT PRIMARY KEY,
    subject_title TEXT,
    campaign_id   TEXT,            -- the subject record, when there was one
    query         TEXT NOT NULL,   -- subject_record | caller_text: what was embedded (§7.2)
    filters       TEXT,            -- JSON, and where they came from
    top_k         INTEGER NOT NULL,
    campaign_ids  TEXT NOT NULL,   -- JSON list, in the order they were returned
    similarities  TEXT,            -- JSON [[campaign_id, score]] — §7.6 stamps these onto
                                    -- the verdict, so a disagreement can be read off them
    warnings      TEXT,            -- JSON list of codes raised while gathering (D18)
    embedding_model TEXT,          -- so a re-ranking after a model change is visible
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS vector_provenance (
    vector_id     TEXT PRIMARY KEY,
    model         TEXT NOT NULL,   -- §7.2: "a model upgrade is a visible migration rather
                                    -- than a silent re-ranking"
    space         TEXT NOT NULL DEFAULT 'campaign',  -- campaign | asset: CLIP and the text
                                    -- embedder are different models by design, so two
                                    -- entries across spaces is normal, not a mixed index
    created_at    REAL NOT NULL
);
"""


# Split out of `_SCHEMA` deliberately. An index names a COLUMN, and on an upgraded database
# the column may not exist until `_migrate_schema` has added it — so running the two together
# meant `_SCHEMA` failing on an index before the migration that would have made it valid could
# run. Tables first, then columns, then indexes: `upgrade()` is the only correct order and the
# only thing that should ever be called.
_INDEXES = """
CREATE INDEX IF NOT EXISTS chunks_campaign_idx   ON campaign_chunks(campaign_id);
CREATE INDEX IF NOT EXISTS assets_campaign_idx   ON assets(campaign_id);
CREATE INDEX IF NOT EXISTS metrics_campaign_idx ON metrics(campaign_id);
CREATE INDEX IF NOT EXISTS metric_values_metric_idx ON metric_values(metric);
CREATE INDEX IF NOT EXISTS metric_values_campaign_idx ON metric_values(campaign_id);
CREATE INDEX IF NOT EXISTS evals_campaign_idx   ON evaluations(campaign_id);
CREATE INDEX IF NOT EXISTS recon_eval_idx       ON reconciliations(evaluation_id);
"""


def upgrade(conn: sqlite3.Connection) -> None:
    """Bring any database — fresh, or from any earlier release — up to the current schema.

    Three steps in this order and no other. `CREATE TABLE IF NOT EXISTS` creates what is
    missing and does nothing to what exists; `_migrate_schema` adds the columns an existing
    table lacks; only then can an index be built on a column that is guaranteed to be there.
    """
    conn.executescript(_SCHEMA)
    conn.commit()
    _migrate_schema(conn)
    conn.executescript(_INDEXES)
    conn.commit()
    _seed_metric_registry(conn)


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


def _declared_columns(table: str) -> set:
    """Which columns `_SCHEMA` declares for a table, read from `_SCHEMA` itself.

    Derived rather than listed, for the reason §2.4 learned when it had to rebuild the
    evaluations table: a hand-kept list of columns to add is a second copy of the schema, and
    a second copy drifts. `_migrate_schema` had three campaign columns in it while
    `get_campaign` read the whole thing — `tags` and `supersedes` unconditionally, added by
    nothing — so any database predating either broke the most-used reader in the codebase and
    no test would have caught it (D89).
    """
    scratch = sqlite3.connect(":memory:")
    try:
        scratch.executescript(_SCHEMA)
        rows = scratch.execute(f"PRAGMA table_info({table})").fetchall()
        return {r[1] for r in rows}
    finally:
        scratch.close()


def _declared_ddl(table: str) -> dict:
    """`{column: type-and-default}` for a table, as `_SCHEMA` declares it.

    Only what `ALTER TABLE ADD COLUMN` can express: SQLite allows NOT NULL only with a
    constant default, and nothing can add a PRIMARY KEY. A NOT NULL column with no default is
    added WITHOUT the constraint rather than skipped — the column existing and being null on
    old rows is a true statement about those rows, and skipping it leaves a reader crashing on
    a column the schema says is there.
    """
    scratch = sqlite3.connect(":memory:")
    try:
        scratch.executescript(_SCHEMA)
        out = {}
        for row in scratch.execute(f"PRAGMA table_info({table})").fetchall():
            _, name, kind, notnull, default, pk = row
            if pk:
                continue
            clause = kind or "TEXT"
            if notnull and default is not None:
                clause += f" NOT NULL DEFAULT {default}"
            elif notnull:
                # NOT NULL with no default cannot be added to a table that has rows, and the
                # obvious workaround — DEFAULT 0 on a timestamp — backdates every existing
                # record to 1970, which is a false answer rather than a missing one. Added
                # NULLABLE instead: a null `created_at` on a pre-existing row says nobody
                # recorded when, which is true.
                pass
            elif default is not None:
                clause += f" DEFAULT {default}"
            out[name] = clause
        return out
    finally:
        scratch.close()


def _declared_tables() -> list:
    """Every table `_SCHEMA` declares, read from `_SCHEMA` itself.

    Derived for the same reason the columns are, and found the same way: this was a hand-kept
    tuple, and §8.1 added `metric_registry` and `metric_values` without adding them to it — so
    every column §8.3 adds would have reached a fresh install and no upgraded one. That is
    precisely the defect D89 closed "as a class rather than the three instances somebody
    happened to notice", reopened by the one list that was still a second copy of the schema.
    """
    scratch = sqlite3.connect(":memory:")
    try:
        scratch.executescript(_SCHEMA)
        return [r[0] for r in scratch.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()]
    finally:
        scratch.close()


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    """Every column `_SCHEMA` declares that an existing table does not have (D89).

    `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists, so a release
    that adds a column reaches a fresh install and no upgraded one. This closes the class
    rather than the three instances somebody happened to notice.
    """
    for table in _declared_tables():
        existing = _columns(conn, table)
        if not existing:
            continue                       # the table itself is created by `_SCHEMA`
        for column, clause in _declared_ddl(table).items():
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {clause}")


def _seed_metric_registry(conn: sqlite3.Connection) -> None:
    """Put the shipped measures on file as part of bringing a database up to date.

    Seeded HERE rather than on first read, which is where it was. `metrics._registry` seeded
    an empty table, so the first caller to ask a question wrote twelve rows — and §8.7's
    `replay.run` is a REPORT whose whole claim is that it writes nothing. It did, on exactly
    the database a customer runs it against first: an upgraded library, where the registry is
    empty until something seeds it.

    The seed list is a shipped payload like the schema, so it belongs with the schema. The
    late import is the cycle: `metrics` reads `store` everywhere, and this is the one edge
    pointing back.
    """
    import metrics

    if conn.execute("SELECT 1 FROM metric_registry LIMIT 1").fetchone():
        return
    for canonical, (display, unit, direction, aliases) in metrics.SEED_REGISTRY.items():
        # `known`, not `expected`: a shipped name is a recognised measure, not something every
        # brief must carry. See §8.3.
        register_metric(conn, canonical=canonical, display_name=display, unit=unit,
                        direction=direction, aliases=list(aliases), status="known")


def _demote_unconfirmed_seed_metrics(conn: sqlite3.Connection) -> None:
    """Seed measures written as `expected` by §8.1, before the word meant a checklist (§8.3).

    §8.1 registered the twelve shipped measures with `status='expected'` meaning "a recognised
    measure". §8.3 gave the word its real meaning — on the checklist every brief in a market is
    held to — and split the vocabulary state (`status`) from the checklist (`expected_in`).

    On a database seeded by the earlier release those twelve rows still say `expected`, so
    §8.3's first read of them would put all twelve on every checklist, confirmed by nobody. A
    row is only demoted when it has no `expected_in` and no `confirmed_by`: anything a person
    actually graduated has both, and must not be touched.
    """
    if "status" not in _columns(conn, "metric_registry"):
        return
    conn.execute("UPDATE metric_registry SET status = 'known' WHERE status = 'expected' "
                 "AND confirmed_by IS NULL "
                 "AND (expected_in IS NULL OR expected_in IN ('[]', ''))")


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Additive migrations for columns added after a release — CREATE TABLE IF NOT EXISTS
    (in _SCHEMA) does nothing for an EXISTING table, so a column added since someone's last
    install (e.g. `collection`, added after v0.2.0 shipped with installers) would otherwise
    make every insert/read against their real database fail outright (reviewed and
    reproduced against a pre-collection schema). Never destructive — only ever adds a column
    (nullable, or NOT NULL with a constant default — SQLite allows the latter on ADD COLUMN,
    backfilling existing rows) to an existing table; a fresh DB already has it via _SCHEMA."""
    # D89: every declared column, derived from `_SCHEMA`. The named migrations below stay —
    # they carry backfills and a table rebuild that adding a column cannot do — but they no
    # longer have to be the complete list, which is what made them a second copy of the
    # schema.
    _add_missing_columns(conn)
    _demote_unconfirmed_seed_metrics(conn)
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
    # §7.2's two new tables. `_SCHEMA` creates them on a fresh install and on any upgrade
    # that runs `init_db`; this is here so a caller that only migrates still gets them, which
    # is how every legacy test in this suite reaches the save path.
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS retrievals (
            id TEXT PRIMARY KEY, subject_title TEXT, campaign_id TEXT, query TEXT NOT NULL,
            filters TEXT, top_k INTEGER NOT NULL, campaign_ids TEXT NOT NULL,
            similarities TEXT, warnings TEXT,
            embedding_model TEXT, created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS vector_provenance (
            vector_id TEXT PRIMARY KEY, model TEXT NOT NULL, created_at REAL NOT NULL);
    """)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS write_refusals (
            id TEXT PRIMARY KEY, reason TEXT NOT NULL, created_at REAL NOT NULL);
    """)
    recon_columns = _columns(conn, "reconciliations")
    if recon_columns and "basis" not in recon_columns:
        conn.execute("ALTER TABLE reconciliations ADD COLUMN basis TEXT")
    retrieval_columns = _columns(conn, "retrievals")
    for column in ("similarities", "warnings"):
        if retrieval_columns and column not in retrieval_columns:
            conn.execute(f"ALTER TABLE retrievals ADD COLUMN {column} TEXT")
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
    upgrade(conn)
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
    # Separately, because they answer different questions and §8.8 made the difference
    # reachable. `has_metrics` counts any row — a forecast, and now a TARGET. "Has this
    # campaign been measured" is what decides whether to go and ask for its numbers, and a
    # campaign carrying only the figure somebody was aiming at has not been measured at all.
    d["has_actual_metrics"] = any(m["metric_type"] == "actual" for m in d["metrics"])
    # §9.1: whether anything has come back. A concluded campaign whose assets are all
    # `proposed` has never been checked against what actually ran, which is the state the
    # review says the library cannot currently notice.
    d["has_delivered_assets"] = bool(conn.execute(
        "SELECT 1 FROM assets WHERE campaign_id = ? AND phase = 'delivered' LIMIT 1",
        (campaign_id,)).fetchone())
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


def _forget_commitment_vectors(conn, campaign_id: str) -> None:
    """A deleted campaign's commitment vectors go with it (§9.3).

    The rows cascade on the foreign key; the vectors live in their own table keyed by a string
    and nothing reaches them, so they would outlive the campaign as orphans — the same cleanup
    `delete_campaign` already does for chunk and asset vectors.
    """
    if not _columns(conn, "commitments") or not _columns(conn, "commitment_vectors"):
        return
    ids = [f"commitment:{r['id']}" for r in conn.execute(
        "SELECT id FROM commitments WHERE campaign_id = ?", (campaign_id,)).fetchall()]
    for vid in ids:
        conn.execute("DELETE FROM commitment_vectors WHERE vector_id = ?", (vid,))


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

    # §9.3: the commitment rows cascade on the foreign key, but their vectors live in their own
    # table keyed by a string and nothing else reaches them — so they would outlive the
    # campaign as orphans, which is the cleanup the two lines above already do for chunks and
    # assets.
    _forget_commitment_vectors(conn, campaign_id)

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

VALID_ASSET_PHASES = ("proposed", "delivered")


def insert_asset(conn, campaign_id: str, *, file_path: str, modality: str = "image",
                 phase: str = "proposed", captured_on: Optional[str] = None) -> str:
    aid = _id("asset")
    conn.execute(
        "INSERT INTO assets (id, campaign_id, modality, file_path, phase, captured_on, "
        "created_at) VALUES (?,?,?,?,?,?,?)",
        (aid, campaign_id, modality, file_path, phase, captured_on, _now()),
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


def asset_fingerprints(conn, asset_ids: list) -> dict:
    """`{asset_id: phash}` for those that have one (§9.2).

    Missing rather than empty for an asset that was never hashed, so a caller can tell "could
    not be compared" from "compared and matched nothing" — two answers a single empty string
    would collapse into one.
    """
    if not asset_ids:
        return {}
    if not _columns(conn, "asset_fingerprints"):
        return {}
    marks = ",".join("?" * len(asset_ids))
    return {r["asset_id"]: r["phash"] for r in conn.execute(
        f"SELECT asset_id, phash FROM asset_fingerprints WHERE asset_id IN ({marks})",
        list(asset_ids)).fetchall()}


def assets_in_phase(conn, campaign_id: str, phase: str) -> list[dict]:
    """The briefed creative, or the photographs that came back (§9.1)."""
    return [a for a in get_assets_for_campaign(conn, campaign_id) if a["phase"] == phase]


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


def get_all_fingerprints(conn, *, exclude_campaign_id: Optional[str] = None,
                         phase: Optional[str] = None) -> list[dict]:
    """[{asset_id, campaign_id, phash, phase}] for a provenance check, optionally excluding one
    campaign's own assets (so an image doesn't "match" itself).

    `phase` travels on every row (§9.1/D121). This corpus was built when "asset" meant
    "creative", and §9.1 put delivered event photographs into the same table — so a reuse check
    could report "this image is already in the library" about a photograph OF an execution
    without ever saying that is what it was. It is still a real match and still worth
    reporting; what it is not is the same finding.
    """
    sql = """SELECT af.asset_id AS asset_id, a.campaign_id AS campaign_id, af.phash AS phash,
                    a.phase AS phase
             FROM asset_fingerprints af JOIN assets a ON a.id = af.asset_id"""
    where, params = [], []
    if exclude_campaign_id:
        where.append("a.campaign_id != ?")
        params.append(exclude_campaign_id)
    if phase:
        where.append("a.phase = ?")
        params.append(phase)
    if where:
        sql += " WHERE " + " AND ".join(where)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def mark_asset_embedded(conn, asset_id: str) -> None:
    conn.execute("UPDATE assets SET embedded = 1 WHERE id = ?", (asset_id,))
    conn.commit()


def list_assets(conn, *, campaign_ids: Optional[list[str]] = None,
               exclude_campaign_id: Optional[str] = None,
               phase: Optional[str] = None) -> list[dict]:
    """[{id, campaign_id, phase, ...}] for CLIP similarity search (§6.6) — either restricted to
    a filtered candidate set of campaigns, or all assets excluding one campaign's own.

    `phase` narrows it to one side (§9.1/D121): `proposed` is the creative corpus this search
    was written against, `delivered` the photographs of what actually ran. Unfiltered by
    default, because "your new hero shot looks like a photograph of the Bogotá activation" is
    a real answer — it just has to arrive saying which it is.
    """
    where, params = [], []
    if campaign_ids is not None:
        if not campaign_ids:
            return []
        where.append(f"campaign_id IN ({','.join('?' * len(campaign_ids))})")
        params += list(campaign_ids)
    elif exclude_campaign_id:
        where.append("campaign_id != ?")
        params.append(exclude_campaign_id)
    if phase:
        where.append("phase = ?")
        params.append(phase)
    sql = "SELECT * FROM assets" + (" WHERE " + " AND ".join(where) if where else "")
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


# ── metrics ──────────────────────────────────────────────────────────────────

# §8.1/D33: `target` is a value, not a refusal. "Did we hit our number" is the comparison a
# marketer most wants, and it needs a number stored as the thing that was aimed at — separate
# from `predicted`, which is what the library expected and what reconciliation scores itself
# against.
VALID_METRIC_TYPES = ("actual", "predicted", "target")
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


class _Batched:
    """A connection whose `commit()` does nothing, for the duration of one import.

    A commit per row is an fsync per row, which the caller's row count controls — the reason
    `add_metrics` grew `commit=False` in the first place (defect 04's sweep). §8.8 routes the
    batch through `core.add_metrics` so a workbook actually reaches the typed registry, and
    that path commits several times per row on its way through `metrics.record`. Swallowing
    the commits and issuing one at the end keeps both: the registry gets populated, and the
    customer's laptop does one fsync rather than four hundred.

    Per-row atomicity is kept by the SAVEPOINT in the loop, not by the commits — a row that
    fails halfway leaves nothing behind, which a deferred commit alone would not give. Those
    savepoints only nest (rather than each one committing on RELEASE) because the caller opens
    an explicit transaction first.
    """

    def __init__(self, inner):
        self._inner = inner

    def commit(self):
        return None

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _retry_of(exc: Exception) -> dict:
    """The retry a `BadValue` carries, as data (D47/§5.2).

    `str(exc)` alone makes the caller parse "Did you mean…?" out of prose; the attributes are
    what let a surface offer the correction.
    """
    return {k: getattr(exc, k) for k in ("field", "given", "valid", "suggestion")
            if getattr(exc, k, None) is not None}


def _row_problem(conn, row) -> Optional[dict]:
    """Whatever would stop this row importing, found without writing anything."""
    if not isinstance(row, dict):
        return {"reason": f"row must be an object, got {type(row).__name__}"}
    structured = row.get("structured")
    if structured is not None and not isinstance(structured, dict):
        return {"reason": f"`structured` must be an object of column/value pairs, got "
                          f"{type(structured).__name__}"}
    try:
        enums.normalise(row.get("metric_type", "actual"), field="metric_type",
                        valid=VALID_METRIC_TYPES, synonyms=enums.METRIC_TYPE_SYNONYMS,
                        allow_none=False)
    except ValueError as exc:
        return {"reason": str(exc), **_retry_of(exc)}
    cid, title = row.get("campaign_id"), row.get("title")
    if not cid and not title:
        return {"reason": "neither campaign_id nor title given"}
    if cid:
        return (None if get_campaign(conn, cid)
                else {"reason": f"campaign_id {cid!r} not found"})
    matches = conn.execute(
        "SELECT id FROM campaigns WHERE LOWER(title) = LOWER(?) "
        "AND id NOT IN (SELECT supersedes FROM campaigns WHERE supersedes IS NOT NULL)",
        (title,)).fetchall()
    if not matches:
        return {"reason": f"no campaign titled {title!r} found"}
    if len(matches) > 1:
        return {"reason": f"title {title!r} is ambiguous ({len(matches)} matches)"}
    return None


def _import_preview_sentence(rows: list, columns: dict, problems: list) -> str:
    """What the reader is deciding about, in the order it matters."""
    parts = [f"{len(rows)} row{'s' * (len(rows) != 1)} read. Nothing has been stored."]
    if problems:
        parts.append(f"{len(problems)} row{'s' * (len(problems) != 1)} would not import at "
                     f"all — see `errors`; the first is: {problems[0]['reason']}")
    if columns["not_measures"]:
        parts.append(f"{len(columns['not_measures'])} column"
                     f"{'s' * (len(columns['not_measures']) != 1)} look like identifiers or "
                     f"dimensions rather than measures and are skipped: "
                     f"{', '.join(c['column'] for c in columns['not_measures'])}.")
    if columns["cannot_type"]:
        parts.append(f"{len(columns['cannot_type'])} column"
                     f"{'s' * (len(columns['cannot_type']) != 1)} cannot be read as numbers "
                     f"and will not be imported: "
                     f"{', '.join(c['column'] for c in columns['cannot_type'])}.")
    if columns["looks_like"]:
        parts.append(f"{len(columns['looks_like'])} column"
                     f"{'s' * (len(columns['looks_like']) != 1)} may be another name for "
                     f"something already on file — say so and they count as one measure, or "
                     f"leave them and each is its own: "
                     f"{', '.join(c['column'] + ' ~ ' + c['looks_like'] for c in columns['looks_like'])}.")
    if columns["new"]:
        parts.append(f"{len(columns['new'])} column"
                     f"{'s' * (len(columns['new']) != 1)} are new to this library and will be "
                     f"recorded provisionally, then asked about one at a time.")
    if columns["known"]:
        parts.append(f"{len(columns['known'])} already match measures on file.")
    parts.append("Send the same rows with confirm=True to import them.")
    return " ".join(parts)


def bulk_import_metrics(conn, rows: list[dict], *, confirm: bool = False) -> dict:
    """
    Load a KPI workbook in one call (§6.5) instead of one add_metrics per row. Each row
    identifies its campaign by `campaign_id` (preferred) or `title` (exact, case-insensitive,
    excluding superseded campaigns so a corrected re-upload resolves uniquely — ambiguous or
    missing matches are reported as errors, never guessed) and carries
    `detail`/`structured`/`metric_type` like add_metrics. Every row is processed
    independently: a malformed row (wrong shape, bad metric_type, a nonexistent id) is
    reported in `errors` and never crashes or blocks the rest of the batch.

    §8.8: **previews by default.** *"`bulk_import_metrics` should diff incoming columns
    against the registry and report what is new, what it thinks are aliases, and what it
    cannot type — before writing anything. An import that silently accepts 40 new keys is how
    the current drift started."* A workbook carries a COLUMN VOCABULARY, and the moment to
    look at a vocabulary is once, as a vocabulary. The preview is the consent step, the same
    way it is for `upload_campaign` and `add_metrics`.

    The structured values go through `metrics.record`, so a workbook actually populates the
    typed registry — *"that is the moment the registry should be populated properly rather
    than accreted key by key"*. It called `store.add_metrics` directly before, so the import
    the review names as where the registry gets seeded seeded nothing, and forty columns went
    into a JSON blob exactly as the review says they should not.
    """
    import core
    import metrics as metrics_module

    columns = metrics_module.diff_columns(conn, rows)
    if not confirm:
        # The ROWS too, not only the columns. `errors: []` was asserted rather than computed,
        # so a preview saying "nothing has been stored, send with confirm=True" could be
        # followed by three hundred of five hundred rows failing on an unmatched title — it
        # had previewed the wrong half. Identity and metric_type are checked here without
        # writing anything, which is the same check the write does.
        problems = [{"row": i, **p} for i, p in enumerate(_row_problem(conn, r) for r in rows)
                    if p]
        return {"preview": True, "imported": 0, "errors": problems, "rows": len(rows),
                "would_import": len(rows) - len(problems),
                "columns": columns,
                "what_it_means": _import_preview_sentence(rows, columns, problems)}
    imported, errors = 0, []
    # Which campaigns need §9.5's drift figure refreshed once the loop is done.
    touched: set[str] = set()
    # An explicit BEGIN, because without one the per-row SAVEPOINT was the OUTERMOST one — and
    # releasing the outermost savepoint commits. So the batch fsynced once per row after all,
    # exactly as it did before `commit=False` existed, while the comment below claimed
    # otherwise. Measured: a second connection could see row 1 before row 2 started.
    conn.execute("BEGIN")
    # Keyed by measure, because a workbook asks the same question once per column and not once
    # per row — the same reason `diff_columns` classifies a column once.
    asked: dict = {}
    eligible: dict = {}
    retired: dict = {}
    skipped: list = []
    batched = _Batched(conn)
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
                # D47: the structured retry, on the batch path too. It was flattened to a
                # string here, so `field`/`valid`/`suggestion` reached the single write and
                # not the one the plan itself named as the likeliest place "Target" arrives —
                # the same vocabulary behaving differently depending on how many rows you sent.
                errors.append({"row": i, "reason": str(exc), **_retry_of(exc)})
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

            # Through `core.add_metrics`, which is what routes the structured values into the
            # typed registry (§8.1) and asks about an unfamiliar one (§8.2). `confirm=True`
            # because the preview above already WAS the consent step for this batch.
            #
            # SAVEPOINT per row, so a row that fails halfway leaves nothing behind — the old
            # path got that from `store.add_metrics` being a single insert, and this one
            # writes a metrics row and then a value row per column.
            conn.execute("SAVEPOINT bulk_row")
            try:
                written = core.add_metrics(
                    batched, campaign_id=cid, detail=row.get("detail"),
                    structured=row.get("structured"), metric_type=metric_type, confirm=True,
                    # Once per campaign, after the loop — not once per row. §9.5's snapshot
                    # reads every asset and every classification, and a workbook with fifty
                    # rows for one campaign ran that fifty times to keep the last answer.
                    snapshot_drift=False)
            except Exception:
                conn.execute("ROLLBACK TO bulk_row")
                raise
            finally:
                conn.execute("RELEASE bulk_row")
            imported += 1
            if metric_type == "actual":
                touched.add(cid)
            # Everything the write path asked or refused, kept rather than dropped. The return
            # was discarded, and the questions inside it are ONE-SHOT: `metrics.record` marks a
            # measure surfaced and offered as a side effect, so an import consumed §8.2's "is
            # this a new measure?" and §8.3's graduation offer for forty columns at once and
            # showed neither. The item whose headline is "never silently accept" made forty
            # acceptances unaskable, permanently.
            for question in written.get("new_measures") or []:
                asked.setdefault(question["measure"], question)
            for offer in written.get("newly_eligible") or []:
                eligible.setdefault(offer["measure"], offer)
            for gone in written.get("retired") or []:
                retired.setdefault(gone["measure"], gone)
            for bad in written.get("skipped") or []:
                skipped.append({"row": i, **bad})
        except Exception as exc:
            errors.append({"row": i, "reason": str(exc), **_retry_of(exc)})

    # Once per campaign, with every row on file — the figure a later citation carries has to
    # rest on the whole import, not on whichever row happened to be last.
    for cid in touched:
        core._snapshot_execution_drift(conn, cid)
    conn.commit()
    result = {"preview": False, "imported": imported, "errors": errors,
              "not_processed": not_processed, "columns": columns,
              **({"new_measures": list(asked.values())} if asked else {}),
              **({"newly_eligible": list(eligible.values())} if eligible else {}),
              **({"retired_measures": list(retired.values())} if retired else {}),
              **({"skipped": skipped} if skipped else {})}
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


def record_vector_model(conn, vector_id: str, model: str, *, space: str = "campaign") -> None:
    """Which embedding model produced this vector (§7.2).

    "So a model upgrade is a visible migration rather than a silent re-ranking." Without it,
    changing the embedder re-ranks every judgment the library will ever make and nothing says
    so — the similarities from two models are not comparable, and ranking across them is
    arithmetic on incompatible numbers.
    """
    if not _columns(conn, "vector_provenance"):
        return
    conn.execute("INSERT OR REPLACE INTO vector_provenance (vector_id, model, space, "
                 "created_at) VALUES (?,?,?,?)", (vector_id, model, space, _now()))
    conn.commit()


def forget_vector_models(conn, vector_ids: list) -> None:
    """Drop provenance for vectors that no longer exist.

    Without this the rows outlive the vectors, so `embedding_models` reports a model that
    nothing in the index was produced by — and after a real migration (delete, re-embed) the
    old model would be reported forever, leaving a permanent "this library is mixed" notice
    on a library that is not.
    """
    if not vector_ids or not _columns(conn, "vector_provenance"):
        return
    conn.executemany("DELETE FROM vector_provenance WHERE vector_id = ?",
                     [(v,) for v in vector_ids])
    conn.commit()


def set_embedding_model(conn, model: str) -> None:
    """Test seam: pretend the next vectors come from a different model, so the
    mixed-index warning can be exercised without swapping embedders."""
    conn.execute("UPDATE vector_provenance SET model = ?", (model,))
    conn.commit()


def embedding_models(conn, space: str = "campaign") -> set:
    """Every model that produced a vector currently in the library, within one space.

    Per space, because CLIP produces the image vectors and the text embedder the chunk ones:
    two entries across both spaces is the normal state, not a mixed index.
    """
    columns = _columns(conn, "vector_provenance")
    if not columns:
        return set()
    if "space" not in columns:
        return {r["model"] for r in
                conn.execute("SELECT DISTINCT model FROM vector_provenance").fetchall()}
    return {r["model"] for r in conn.execute(
        "SELECT DISTINCT model FROM vector_provenance WHERE space = ?", (space,)).fetchall()}


def insert_retrieval(conn, *, subject_title, campaign_id, query, filters, top_k,
                     campaign_ids, embedding_model, similarities=None, warnings=None) -> str:
    """Write down what the server retrieved (§7.2).

    §6.1 rejected a receipt as the mechanism for verifying a QUOTE (X7): it answers a weaker
    question than "does the cited record contain these words", and it would have put a
    stateful handshake on a stateless tool. That objection stands for 6.1. Here the server is
    already doing the retrieval, so writing it down costs one row — and it answers the
    question 6.1 could not reach: was this record in front of the reasoner at all (D77).
    """
    rid = _id("ret")
    conn.execute(
        "INSERT INTO retrievals (id, subject_title, campaign_id, query, filters, top_k, "
        "campaign_ids, embedding_model, similarities, warnings, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (rid, subject_title, campaign_id, query, json.dumps(filters or {}), top_k,
         json.dumps(list(campaign_ids)), embedding_model,
         json.dumps(similarities or []), json.dumps(warnings or []), _now()))
    conn.commit()
    return rid


def get_retrieval(conn, retrieval_id: str) -> Optional[dict]:
    if not _columns(conn, "retrievals"):
        return None
    row = conn.execute("SELECT * FROM retrievals WHERE id = ?", (retrieval_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["campaign_ids"] = json.loads(d["campaign_ids"] or "[]")
    d["filters"] = json.loads(d["filters"] or "{}")
    d["similarities"] = json.loads(d.get("similarities") or "[]")
    d["warnings"] = json.loads(d.get("warnings") or "[]")
    return d


def metric_registry(conn) -> dict:
    if not _columns(conn, "metric_registry"):
        return {}
    out = {}
    for row in conn.execute("SELECT * FROM metric_registry").fetchall():
        d = dict(row)
        d["aliases"] = json.loads(d["aliases"] or "[]")
        d["markets"] = json.loads(d["markets"] or "[]")
        d["expected_in"] = json.loads(d.get("expected_in") or "[]")
        d["answered"] = bool(d.get("answered"))
        d["surfaced"] = bool(d.get("surfaced"))
        out[d["canonical"]] = d
    return out


def register_metric(conn, *, canonical, display_name, unit, direction, aliases,
                    status="provisional") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO metric_registry (canonical, display_name, unit, direction, "
        "aliases, status) VALUES (?,?,?,?,?,?)",
        (canonical, display_name, unit, direction, json.dumps(list(aliases)), status))
    conn.commit()


def metric_was_surfaced(conn, canonical: str) -> bool:
    row = conn.execute("SELECT surfaced FROM metric_registry WHERE canonical = ?",
                       (canonical,)).fetchone()
    return bool(row and row["surfaced"])


def metric_was_offered(conn, canonical: str) -> bool:
    """Whether §8.3's graduation question has been put once already."""
    row = conn.execute("SELECT offered FROM metric_registry WHERE canonical = ?",
                       (canonical,)).fetchone()
    return bool(row and row["offered"])


def mark_metric_offered(conn, canonical: str) -> None:
    conn.execute("UPDATE metric_registry SET offered = 1 WHERE canonical = ?", (canonical,))
    conn.commit()


def mark_metric_surfaced(conn, canonical: str) -> None:
    conn.execute("UPDATE metric_registry SET surfaced = 1 WHERE canonical = ?", (canonical,))
    conn.commit()


def answer_metric(conn, canonical: str, *, status: str) -> None:
    conn.execute("UPDATE metric_registry SET answered = 1, status = ? WHERE canonical = ?",
                 (status, canonical))
    conn.commit()


def merge_metric(conn, *, provisional: str, into: str) -> None:
    """Fold a provisional measure into an existing one (§8.2's "same thing").

    RETROSPECTIVE. The values already recorded under the provisional name belong to the
    measure it turned out to be, and an answer that fixes the vocabulary while leaving the
    data behind has fixed nothing — the point of saying `crm_reach` is `reach` is being able
    to ask for every reach on file and get both.
    """
    row = conn.execute("SELECT aliases FROM metric_registry WHERE canonical = ?",
                       (into,)).fetchone()
    aliases = json.loads(row["aliases"] or "[]") if row else []
    if provisional not in aliases:
        aliases.append(provisional)
    conn.execute("UPDATE metric_registry SET aliases = ? WHERE canonical = ?",
                 (json.dumps(aliases), into))
    conn.execute("UPDATE metric_values SET metric = ? WHERE metric = ?", (into, provisional))
    conn.execute("DELETE FROM metric_registry WHERE canonical = ?", (provisional,))
    conn.commit()


def touch_metric(conn, canonical: str, *, campaign_id: Optional[str] = None) -> None:
    """Record that this measure was seen again — §8.3's graduation gate and §8.5's retirement
    both read these, and recording them from the first write is what stops the history being
    missing for exactly the measures that arrived before anybody thought about it."""
    row = conn.execute("SELECT first_seen, times_seen, markets FROM metric_registry "
                       "WHERE canonical = ?", (canonical,)).fetchone()
    if not row:
        return
    markets = json.loads(row["markets"] or "[]")
    if campaign_id:
        # Through `markets_of`, and every one of them: a campaign that ran in MX and CO counts
        # towards both, and reading `market or region` alone counted it as zero — the fifth
        # implementation of this question, drifting exactly as D55/D88 say they do.
        record = get_campaign(conn, campaign_id) or {}
        seen = {fold_market(m) for m in markets}
        for where in markets_of(record):
            # Folded, so "LATAM" and "latam" cannot satisfy a two-market gate between them.
            # The first spelling seen is the one kept, because it is what somebody typed.
            if where and fold_market(where) not in seen:
                markets.append(where.strip())
                seen.add(fold_market(where))
    now = _now()
    conn.execute("UPDATE metric_registry SET first_seen = COALESCE(first_seen, ?), "
                 "last_seen = ?, times_seen = times_seen + 1, markets = ? WHERE canonical = ?",
                 (now, now, json.dumps(markets), canonical))
    conn.commit()


def record_execution_drift(conn, campaign_id: str, *, score, relative, counts, classified,
                           status: str, items=None) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO execution_drift (campaign_id, score, relative, counts, "
        "classified, items, status, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (campaign_id, score, relative, json.dumps(counts), json.dumps(classified),
         json.dumps(items or []), status, _now()))
    conn.commit()


def execution_drift_for(conn, campaign_id: str) -> dict:
    """What was known about how faithfully this campaign ran, when its results were filed.

    `never_checked` is a THIRD state and not a low score. Reading "nobody looked" as "it ran as
    briefed" is the assumption that makes the whole of Phase 9 necessary.
    """
    if not _columns(conn, "execution_drift"):
        return _no_drift_on_file()
    row = conn.execute("SELECT * FROM execution_drift WHERE campaign_id = ?",
                       (campaign_id,)).fetchone()
    if not row:
        return _no_drift_on_file()
    d = dict(row)
    d["counts"] = json.loads(d["counts"] or "{}")
    d["classified"] = json.loads(d["classified"] or "{}")
    d["items"] = json.loads(d.get("items") or "[]")
    d["basis"] = "computed"
    return d


def _no_drift_on_file() -> dict:
    return {"score": None, "relative": None, "counts": {}, "classified": {}, "items": [],
            "status": "never_checked", "basis": "computed"}


def insert_drift_classification(conn, *, campaign_id: str, subject: str, item: str,
                                classification: str, why: str, classified_by: str,
                                outcome_known: bool) -> str:
    did = _id("drift")
    conn.execute(
        "INSERT INTO drift_classifications (id, campaign_id, subject, item, classification, "
        "why, classified_by, outcome_known, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (did, campaign_id, subject, item, classification, why, classified_by,
         int(outcome_known), _now()))
    conn.commit()
    return did


def drift_classifications(conn, campaign_id: str, *, subject: Optional[str] = None) -> list:
    """Oldest first — the order a reading changed in is the point of keeping both."""
    if not _columns(conn, "drift_classifications"):
        return []
    sql = "SELECT * FROM drift_classifications WHERE campaign_id = ?"
    params: list = [campaign_id]
    if subject is not None:
        sql += " AND subject = ?"
        params.append(subject)
    return [dict(r) for r in conn.execute(sql + " ORDER BY created_at, id", params).fetchall()]


def latest_drift_classification(conn, campaign_id: str, subject: str) -> Optional[dict]:
    rows = drift_classifications(conn, campaign_id, subject=subject)
    return rows[-1] if rows else None


def insert_commitment(conn, *, campaign_id: str, text: str, source_line: str,
                      origin: str = "extracted") -> str:
    cid = _id("commit")
    conn.execute("INSERT INTO commitments (id, campaign_id, text, source_line, origin, "
                 "created_at) VALUES (?,?,?,?,?,?)",
                 (cid, campaign_id, text, source_line, origin, _now()))
    conn.commit()
    return cid


def commitments_for(conn, campaign_id: str, *, include_dropped: bool = False) -> list:
    if not _columns(conn, "commitments"):
        return []
    sql = "SELECT * FROM commitments WHERE campaign_id = ?"
    if not include_dropped:
        sql += " AND status = 'open'"
    return [dict(r) for r in conn.execute(sql + " ORDER BY created_at, id",
                                          (campaign_id,)).fetchall()]


def get_commitment(conn, commitment_id: str) -> Optional[dict]:
    if not _columns(conn, "commitments"):
        return None
    row = conn.execute("SELECT * FROM commitments WHERE id = ?", (commitment_id,)).fetchone()
    return dict(row) if row else None


def forget_commitment(conn, commitment_id: str) -> None:
    """Remove an EXTRACTED commitment because the brief it was read from has changed.

    Distinct from `drop_commitment`, which records a person's decision and is kept. This is
    the extractor withdrawing its own earlier reading of a document that no longer says that.
    """
    conn.execute("DELETE FROM commitments WHERE id = ? AND origin = 'extracted'",
                 (commitment_id,))
    conn.execute("DELETE FROM commitment_vectors WHERE vector_id = ?"
                 if _columns(conn, "commitment_vectors") else "SELECT 1",
                 (f"commitment:{commitment_id}",) if _columns(conn, "commitment_vectors")
                 else ())
    conn.commit()


def drop_commitment(conn, commitment_id: str, *, why: Optional[str] = None) -> None:
    """Dropped, never deleted. Mechanical extraction will pick up lines that are not promises,
    and the record of what the server thought was one is how somebody later understands why a
    post-mortem said what it said."""
    conn.execute("UPDATE commitments SET status = 'dropped', why_dropped = ? WHERE id = ?",
                 (why, commitment_id))
    conn.commit()


def corrections(conn) -> list:
    if not _columns(conn, "corrections"):
        return []
    out = []
    for row in conn.execute("SELECT * FROM corrections ORDER BY first_seen, id").fetchall():
        d = dict(row)
        d["markets"] = json.loads(d["markets"] or "[]")
        d["expected_in"] = json.loads(d["expected_in"] or "[]")
        out.append(d)
    return out


def get_correction(conn, correction_id: str) -> Optional[dict]:
    return next((c for c in corrections(conn) if c["id"] == correction_id), None)


def live_correction(conn, correction_id: Optional[str]) -> Optional[dict]:
    """Follow `merged_into` to the row that is actually the rule now.

    A merged row is kept because it is somebody's words, which means it is still reachable by
    its own wording — and a later mention of that wording landed on the dead row. The rule it
    was folded into stayed where it was, so the fourth market saying the second market's
    sentence advanced nothing: the review's own three-market scenario, failing on the mention
    after the merge.
    """
    seen: set = set()
    row = get_correction(conn, correction_id) if correction_id else None
    while row and row.get("merged_into") and row["id"] not in seen:
        seen.add(row["id"])
        nxt = get_correction(conn, row["merged_into"])
        if nxt is None or nxt["id"] in seen:
            break                      # a cycle cannot exist (merge refuses one) — belt
        row = nxt
    return row


def correction_by_text(conn, normalised: str) -> Optional[dict]:
    match = next((c for c in corrections(conn) if c["normalised"] == normalised), None)
    return live_correction(conn, match["id"]) if match else None


def insert_correction(conn, *, text: str, normalised: str) -> str:
    cid = _id("corr")
    conn.execute("INSERT INTO corrections (id, text, normalised) VALUES (?,?,?)",
                 (cid, text, normalised))
    conn.commit()
    return cid


def note_correction_sighting(conn, *, correction_id: str, campaign_id: Optional[str],
                             provenance: str, said_as: Optional[str] = None) -> str:
    sid = _id("csight")
    conn.execute("INSERT INTO correction_sightings (id, correction_id, campaign_id, "
                 "provenance, said_as, noted_at) VALUES (?,?,?,?,?,?)",
                 (sid, correction_id, campaign_id, provenance, said_as, _now()))
    conn.commit()
    return sid


def merge_correction(conn, *, absorbed: str, into: str) -> None:
    """Fold one correction into another (§8.6's "same rule").

    RETROSPECTIVE, exactly as §8.2's `merge_metric` is: the sightings already recorded under
    the absorbed wording are sightings of the rule it turned out to be, and an answer that
    fixes the vocabulary while leaving the evidence behind has fixed nothing — the whole point
    of saying these are one rule is that they then COUNT as one rule across three markets.

    Unlike `merge_metric`, the absorbed row is not deleted. That was §8.5's "never delete"
    broken from the direction nothing was watching, and here the row is somebody's words.
    """
    conn.execute("UPDATE correction_sightings SET correction_id = ? WHERE correction_id = ?",
                 (into, absorbed))
    conn.execute("UPDATE corrections SET status = 'merged', merged_into = ? WHERE id = ?",
                 (into, absorbed))
    # The count and the breadth are re-derived from the sightings that just moved, rather than
    # added up — two numbers maintained by arithmetic drift from the rows they describe.
    rows = conn.execute("SELECT campaign_id, noted_at FROM correction_sightings "
                        "WHERE correction_id = ?", (into,)).fetchall()
    markets: list = []
    seen: set = set()
    for row in rows:
        for where in markets_of(get_campaign(conn, row["campaign_id"]) or {}):
            if where and fold_market(where) not in seen:
                seen.add(fold_market(where))
                markets.append(where.strip())
    times = len(rows)
    stamps = [r["noted_at"] for r in rows if r["noted_at"] is not None]
    conn.execute("UPDATE corrections SET times_seen = ?, markets = ?, first_seen = ?, "
                 "last_seen = ? WHERE id = ?",
                 (times, json.dumps(markets), min(stamps or [None], default=None),
                  max(stamps or [None], default=None), into))
    conn.commit()


def set_aside_correction(conn, correction_id: str) -> None:
    """Stop applying and stop asking, without deleting (§8.6).

    `learning.gate` has always had a `set_aside` branch and nothing could reach it, so a
    correction promoted in error blocked approvals with no inverse. The sightings stay: this
    is a decision about the RULE, not about what the client said.
    """
    # `expected_in` is LEFT ALONE. Where a rule used to apply is part of its record, the same
    # way its sightings are — a judgment that cited it while it was standing is only readable
    # if the library can still say where it stood. `status` is what stops it being applied.
    conn.execute("UPDATE corrections SET status = 'ignored', offered = 1 WHERE id = ?",
                 (correction_id,))
    conn.commit()


def correction_sightings(conn, correction_id: str) -> list:
    if not _columns(conn, "correction_sightings"):
        return []
    return [dict(r) for r in conn.execute(
        "SELECT * FROM correction_sightings WHERE correction_id = ? ORDER BY noted_at, id",
        (correction_id,)).fetchall()]


def correction_campaigns(conn, correction_id: str) -> list:
    return sorted({s["campaign_id"] for s in correction_sightings(conn, correction_id)
                   if s["campaign_id"]})


def touch_correction(conn, correction_id: str, *, campaign_id: Optional[str] = None) -> None:
    """Record that a correction was said again — the count and the breadth the gate reads.

    Through `markets_of` and folded, for the reasons §8.3 learned the hard way: a campaign
    running in MX and CO counts towards both, and "SEA"/"sea" is one market.
    """
    row = conn.execute("SELECT markets FROM corrections WHERE id = ?",
                       (correction_id,)).fetchone()
    if not row:
        return
    markets = json.loads(row["markets"] or "[]")
    if campaign_id:
        seen = {fold_market(m) for m in markets}
        for where in markets_of(get_campaign(conn, campaign_id) or {}):
            if where and fold_market(where) not in seen:
                markets.append(where.strip())
                seen.add(fold_market(where))
    now = _now()
    conn.execute("UPDATE corrections SET first_seen = COALESCE(first_seen, ?), last_seen = ?, "
                 "times_seen = times_seen + 1, markets = ? WHERE id = ?",
                 (now, now, json.dumps(markets), correction_id))
    conn.commit()


def graduate_correction(conn, correction_id: str, *, markets: list, confirmed_by: str) -> None:
    # COALESCE, for the reason `graduate_metric` gives: the first confirmation is when this
    # became a rule, and rewriting it makes §8.7 assert that a judgment which CITES the rule
    # was never checked against it.
    conn.execute("UPDATE corrections SET status = 'expected', expected_in = ?, "
                 "confirmed_by = ?, confirmed_at = COALESCE(confirmed_at, ?), "
                 "retired_at = NULL, offered = 1 WHERE id = ?",
                 (json.dumps(sorted(markets)), confirmed_by, _now(), correction_id))
    conn.commit()


def correction_was_offered(conn, correction_id: str) -> bool:
    row = conn.execute("SELECT offered FROM corrections WHERE id = ?",
                       (correction_id,)).fetchone()
    return bool(row and row["offered"])


def reopen_correction(conn, correction_id: str) -> None:
    """Back to provisional — never straight to standing, which is the gate's business."""
    conn.execute("UPDATE corrections SET status = 'provisional', offered = 0, asked = 0 "
                 "WHERE id = ? AND status = 'ignored'", (correction_id,))
    conn.commit()


def correction_quiet_asked(conn, correction_id: str) -> bool:
    row = conn.execute("SELECT quiet_asked FROM corrections WHERE id = ?",
                       (correction_id,)).fetchone()
    return bool(row and row["quiet_asked"])


def mark_correction_quiet_asked(conn, correction_id: str) -> None:
    conn.execute("UPDATE corrections SET quiet_asked = 1 WHERE id = ?", (correction_id,))
    conn.commit()


def correction_was_asked(conn, correction_id: str) -> bool:
    row = conn.execute("SELECT asked FROM corrections WHERE id = ?",
                       (correction_id,)).fetchone()
    return bool(row and row["asked"])


def mark_correction_asked(conn, correction_id: str) -> None:
    conn.execute("UPDATE corrections SET asked = 1 WHERE id = ?", (correction_id,))
    conn.commit()


def mark_correction_offered(conn, correction_id: str) -> None:
    conn.execute("UPDATE corrections SET offered = 1 WHERE id = ?", (correction_id,))
    conn.commit()


def campaigns_that_skipped_correction(conn, correction_id: str, *, since: Optional[float],
                                      markets: Optional[list] = None) -> int:
    """Campaigns that recorded feedback of their own since `since` without repeating this one.

    The measure version of this was wrong in three ways at once (writes not campaigns, every
    campaign not the ones that skipped it, every market not the ones it is expected in). Same
    shape, same three cares — and the shape is why they are two queries rather than one: the
    tables differ, the question does not.
    """
    if not _columns(conn, "correction_sightings") or since is None:
        return 0
    sql = ("SELECT COUNT(DISTINCT s.campaign_id) AS n FROM correction_sightings s "
           "WHERE s.noted_at > ? AND s.campaign_id IS NOT NULL "
           "AND NOT EXISTS (SELECT 1 FROM correction_sightings o "
           "                WHERE o.campaign_id = s.campaign_id AND o.correction_id = ?)")
    params: list = [since, correction_id]
    folded = [f for f in {fold_market(m) for m in (markets or [])} if f]
    if folded:
        clause = " OR ".join(
            ["LOWER(c.market) = ?", "LOWER(c.region) = ?", "LOWER(c.markets) LIKE ?"]
            * len(folded))
        sql += (f" AND EXISTS (SELECT 1 FROM campaigns c WHERE c.id = s.campaign_id "
                f"AND ({clause}))")
        for f in folded:
            params += [f, f, f'%"{f}"%']
    row = conn.execute(sql, params).fetchone()
    return int(row["n"] or 0)


def markets_of(campaign: dict) -> list:
    """Every market a campaign counts towards, or `[None]` when it has none.

    The one implementation. `markets` is the only way to express a multi-country activation,
    so ignoring it reports a real LATAM campaign as covering nothing; a record with no market
    at all is most of a young library, so dropping those describes a library nobody has. Case
    is folded WITHIN a record here — folding it across records is the caller's job, and §5.3
    is where that was got wrong before (D71).
    """
    named = []
    for raw in (campaign.get("market"), campaign.get("region"),
                *(campaign.get("markets") or [])):
        if raw and str(raw).strip():
            value = str(raw).strip()
            if value.lower() not in {m.lower() for m in named}:
                named.append(value)
    return named or [None]


def fold_market(name: Optional[str]) -> Optional[str]:
    """The comparison key for a market name.

    "LATAM" and "latam" are one market. C16 established the fold after exactly this bug, and
    §8.3 reintroduced it in the one place it does the most damage: three spellings of one
    market satisfied a gate whose entire purpose is "seen in at least two markets".
    """
    return name.strip().lower() if name and name.strip() else None


def graduate_metric(conn, canonical: str, *, markets: list, confirmed_by: str) -> None:
    """Promote a measure onto the checklist for the markets it earned (§8.3).

    `expected_in` is the markets it was actually SEEN in, not every market on file. A measure
    that graduated on LATAM and SEA is not a gap in a market nobody has used it in, and
    reporting it as one would make every new market fail a checklist on its first brief.
    """
    # `answered` too: putting a measure on the checklist is a stronger answer than §8.2's
    # question asks for, and a measure that is expected of every brief while still asking "is
    # this a new measure?" is the product asking a question it has already acted on.
    # COALESCE on `confirmed_at`: the first confirmation is when this became a rule, and
    # §8.7 compares it against when each judgment was saved. Overwriting it made a measure
    # graduated, judged against, retired and graduated again look NEWER than the judgment that
    # was checked against it — so the report asserted the judgment had never seen it, which is
    # the confident unfounded claim the report exists to avoid.
    conn.execute("UPDATE metric_registry SET status = 'expected', expected_in = ?, "
                 "answered = 1, surfaced = 1, confirmed_by = ?, "
                 "confirmed_at = COALESCE(confirmed_at, ?), "
                 "retired_at = NULL WHERE canonical = ?",
                 (json.dumps(sorted(markets)), confirmed_by, _now(), canonical))
    conn.commit()


def retire_metric(conn, canonical: str) -> None:
    """Demote, never delete (§8.5).

    `expected_in` and every recorded value are left exactly where they are. The measure stops
    being asked for; the record that it was once asked for survives, because "we used to track
    this" is an answer somebody will need and a deleted row can only say "we never did".
    """
    conn.execute("UPDATE metric_registry SET status = 'retired', retired_at = ? "
                 "WHERE canonical = ?", (_now(), canonical))
    conn.commit()


def revive_metric(conn, canonical: str) -> None:
    """A retired measure that somebody recorded again is expected again (§8.5).

    It graduated once and a person confirmed it. Asking them to confirm it a second time
    because a quarter went by is asking the same question twice, which §8.2 established is how
    a product teaches people to dismiss it.
    """
    conn.execute("UPDATE metric_registry SET status = 'expected', retired_at = NULL "
                 "WHERE canonical = ? AND status = 'retired'", (canonical,))
    conn.commit()


def campaigns_that_skipped(conn, metric: str, *, since: Optional[float],
                           markets: Optional[list] = None) -> int:
    """How many campaigns reported measurements since `since` WITHOUT this one (§8.5).

    Three things this is careful about, and each was wrong in the first version.

    **Campaigns, not writes.** `COUNT(DISTINCT campaign_id)`: one campaign recording ten
    values is one campaign, and counting writes turns "demote after M campaigns" into "demote
    after M numbers".

    **Campaigns that SKIPPED it, not campaigns that exist.** Counting every campaign meant a
    bulk import of 200 historical briefs aged out every measure at once — including, in the
    reviewed scenario, the measure carried by 53 of the 63 campaigns on file. A measure the
    library keeps reporting has not stopped appearing, whatever else was imported alongside it.

    **Only where it is expected.** Ten APAC campaigns retired a measure expected in LATAM, SEA
    and EMEA. A market that never used it cannot be evidence that it fell out of use.
    """
    if not _columns(conn, "metric_values") or since is None:
        return 0
    sql = ("SELECT COUNT(DISTINCT v.campaign_id) AS n FROM metric_values v "
           "WHERE v.created_at > ? "
           "AND NOT EXISTS (SELECT 1 FROM metric_values m "
           "                WHERE m.campaign_id = v.campaign_id AND m.metric = ?)")
    params: list = [since, metric]
    folded = [f for f in {fold_market(m) for m in (markets or [])} if f]
    if folded:
        # The campaign's own market OR region OR anything in its `markets` list — the same
        # question `markets_of` answers, asked in SQL. LIKE on the JSON is deliberate: the
        # list is a JSON array of names and an exact match would miss a multi-market record.
        clause = " OR ".join(
            ["LOWER(c.market) = ?", "LOWER(c.region) = ?", "LOWER(c.markets) LIKE ?"] * len(folded))
        sql += (f" AND EXISTS (SELECT 1 FROM campaigns c WHERE c.id = v.campaign_id "
                f"AND ({clause}))")
        for f in folded:
            params += [f, f, f'%"{f}"%']
    row = conn.execute(sql, params).fetchone()
    return int(row["n"] or 0)


def record_metric_value(conn, *, campaign_id, metric, raw_key, value, unit, source, scope,
                        metric_type) -> str:
    vid = _id("mval")
    conn.execute(
        "INSERT INTO metric_values (id, campaign_id, metric, raw_key, value, unit, source, "
        "scope, metric_type, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (vid, campaign_id, metric, raw_key, value, unit, source, scope, metric_type, _now()))
    conn.commit()
    return vid


def metric_values(conn, *, metric: str, campaign_id: Optional[str] = None) -> list:
    if not _columns(conn, "metric_values"):
        return []
    sql = "SELECT * FROM metric_values WHERE metric = ?"
    params = [metric]
    if campaign_id:
        sql += " AND campaign_id = ?"
        params.append(campaign_id)
    return [dict(r) for r in conn.execute(sql + " ORDER BY created_at", params).fetchall()]


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

def insert_reconciliation(conn, *, evaluation_id, comparison, actual=None,
                          basis: Optional[str] = None) -> str:
    rid = _id("recon")
    conn.execute(
        "INSERT INTO reconciliations (id, evaluation_id, actual, comparison, basis, "
        "created_at) VALUES (?,?,?,?,?,?)",
        (rid, evaluation_id, actual, comparison, basis, _now()),
    )
    conn.commit()
    return rid


def get_reconciliation(conn, reconciliation_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM reconciliations WHERE id = ?",
                       (reconciliation_id,)).fetchone()
    return dict(row) if row else None


def record_refusal(conn, reason: str) -> None:
    """Count a write the server refused, by REASON (§7.6 / D81).

    "Thirty writes were refused" says nothing; which rule refused them is the signal, because
    a rise in citation refusals and a rise in severity downgrades mean different things. The
    worst outcome this makes visible is the quietest one: a real finding dropped because its
    citation would not verify leaves no trace at all today, and the marketer never learns that
    something was left out.

    Best-effort and never raises — a counter that can break a save would be a worse bug than
    the blindness it fixes.
    """
    try:
        if not _columns(conn, "write_refusals"):
            return
        conn.execute("INSERT INTO write_refusals (id, reason, created_at) VALUES (?,?,?)",
                     (_id("refusal"), reason, _now()))
        conn.commit()
    except sqlite3.Error:
        pass


def refusal_counts(conn) -> dict:
    if not _columns(conn, "write_refusals"):
        return {}
    return {r["reason"]: r["n"] for r in conn.execute(
        "SELECT reason, COUNT(*) AS n FROM write_refusals GROUP BY reason").fetchall()}
