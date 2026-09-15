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
    -- §9.6: the campaign's own window. Dates existed only as prose that `facts.py` parses for
    -- contradictions, so there was nothing for a context event's date range to overlap WITH —
    -- this is the field §9.6 turns on, the way `phase` was §9.1's. Nullable on purpose: the
    -- honest default is no window rather than one read out of the deck and stored as though
    -- somebody had entered it. `context.window_of` falls back to the brief's own dates and
    -- labels that reading `heuristic`.
    starts_on     TEXT,            -- ISO date, or NULL
    ends_on       TEXT,            -- ISO date, or NULL
    window_source TEXT,            -- NULL = a person entered it; 'workbook' = read from a
                                   -- spreadsheet's date columns (D119). A KPI workbook is one
                                   -- row per month, so a workbook window WIDENS as rows
                                   -- arrive — but it must never widen over one somebody typed,
                                   -- which is the correction being undone by what it corrected
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
    -- §9.8: WHEN this was measured. A KPI workbook is one row per month, and without this the
    -- campaign's whole window was used — so January's sell-through was confounded by a
    -- September earthquake, on a row that literally carries `month: 2026-01`. It fails hardest
    -- for the customers with the most data. Read from the same date columns §9.6/D119 already
    -- parses; NULL falls back to the campaign window, and the fallback says so.
    period_start  TEXT,
    period_end    TEXT,
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
CREATE TABLE IF NOT EXISTS context_events (
    -- §9.6: what else was going on. "Conflict, natural disaster, regulatory change,
    -- supply-chain or port disruption, platform outage, competitor launch, macro shock, and
    -- fixed calendar events."
    --
    -- There is deliberately NO join table to campaigns. "Any campaign whose window overlaps an
    -- event in its market links automatically — no one should have to remember to connect
    -- them", and a maintained link is exactly the remembering this replaces. Both halves keep
    -- arriving (an earthquake is recorded weeks after the campaigns it overlapped; a campaign
    -- is uploaded months after the event was seeded), so the link is computed from (scope,
    -- range) against (market, window) on every read. A materialised one would be stale in both
    -- directions, and stale here means silently wrong.
    id            TEXT PRIMARY KEY,
    starts_on     TEXT NOT NULL,   -- ISO date
    ends_on       TEXT,            -- NULL = ongoing. A port closure with no announced
                                   -- reopening is the ordinary case, and an invented end date
                                   -- silently stops matching campaigns it still covers
    scope         TEXT NOT NULL,   -- market | region | global
    scope_value   TEXT,            -- the market or region named; NULL for global
    scope_key     TEXT,            -- `scope_value` case-folded, and the column actually
                                   -- MATCHED on. SQLite's COLLATE NOCASE folds ASCII only
                                   -- while Python's .lower() folds Unicode, so the SQL side
                                   -- and the Python side disagreed on "MÉXICO" vs "méxico":
                                   -- recording an event said it reached a campaign and
                                   -- opening that campaign said nothing had. One fold, stored
                                   -- once, is the only way two directions stay one answer
    kind          TEXT NOT NULL,   -- see context.KINDS
    description   TEXT NOT NULL,
    source        TEXT,            -- a link, so this is not a rumour on the record
    -- The STATED impact. The server measured none of this; somebody said it, and §2.4 made
    -- `computed` unwritable from the MCP surface exactly so the word keeps meaning what it says.
    delay_days    INTEGER,
    budget_change_pct REAL,
    channels_disrupted TEXT NOT NULL DEFAULT '[]',   -- JSON array of strings
    recorded_by   TEXT NOT NULL,   -- whose account of it this is
    seeded        INTEGER NOT NULL DEFAULT 0,   -- §9.7: shipped with the product rather than
                                   -- entered by this customer
    recurs_annually INTEGER,       -- §9.8: does this come round every year? NULL/0 means no.
                                   -- The axis §9.8 confounds on, and NOT the same as `kind`:
                                   -- every shipped calendar row is `fixed_calendar`, so keying
                                   -- on kind put a once-in-a-generation home World Cup in the
                                   -- same bucket as Black Friday, and the product raised a
                                   -- finding that a window ran into the tournament and then
                                   -- called the result clean. A yearly date is the BASELINE a
                                   -- year-on-year comparison is made against; a one-off is not
    certainty     TEXT,            -- §9.7, on a SEEDED row: fixed | announced | observed |
                                   -- seasonal. A shipped calendar is a claim about the world
                                   -- and most of these claims are approximate — Ramadan begins
                                   -- on a sighting, a monsoon has an onset that moves by
                                   -- weeks. Shipping those as exact facts is the confident
                                   -- unfounded claim this product is written against
    seed_key      TEXT UNIQUE,     -- §9.7's handle on a row it wrote, e.g. 'ae.ramadan.2026'.
                                   -- A random ctx_… id gives a seeder no way back: it cannot
                                   -- run twice without doubling the calendar, and cannot
                                   -- replace a corrected Ramadan date on upgrade. The
                                   -- precedent is `_seed_metric_registry`, idempotent because
                                   -- `canonical` is its primary key
    -- §8.5's shape: a withdrawn event is KEPT. "We used to think this" is an answer and
    -- deleting is a lie — a judgment saved while the event was on file rested on it. Without
    -- a correction path an event typed with the wrong year attaches itself to every
    -- overlapping campaign forever, which is strictly worse than the hand-maintained join
    -- this replaces: a join table at least lets you unlink.
    withdrawn_at  REAL,
    withdrawn_by  TEXT,
    withdrawn_why TEXT,
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS campaign_notices (
    -- §10.1/D20: a persisted `blocked` or `degraded` notice, per record. "A notice addressed
    -- to a person is by definition 'needs something from a human'", which is 10.1's own
    -- definition of open — and a warning lives for exactly one response, so the thing the
    -- library most wanted somebody to act on was the thing it forgot fastest. C14 persisted
    -- one of these (`commentary_checked`) and left the rest; this is the rest.
    id            TEXT PRIMARY KEY,
    campaign_id   TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    code          TEXT NOT NULL,   -- notices.CODES: the stable name, never the prose
    detail        TEXT,
    created_at    REAL NOT NULL,
    cleared_at    REAL,            -- set when the thing it asked for actually happened
    UNIQUE (campaign_id, code)     -- one open notice per kind per record; re-raising the same
                                   -- condition is the same outstanding job, not a second one
);

CREATE TABLE IF NOT EXISTS library_state (
    -- §10.6/D54: one fact about the library as a whole, remembered between calls.
    --
    -- Exactly one row is written here today: the code of the top-ranked gap as of the last
    -- upload. An upload that CHANGES what matters most is the moment the ranking is worth
    -- reading and the only moment somebody is looking, and "changed" is not a question a
    -- stateless call can answer — the comparison needs the previous answer.
    --
    -- Deliberately not a settings table. Nothing here is configuration and nothing here is
    -- authoritative: every value is a cache of something recomputable, so losing the table
    -- costs one redundant offer and never a wrong answer. That is what makes it safe to be
    -- the only mutable global state in a store that is otherwise append-only.
    key         TEXT PRIMARY KEY,
    value       TEXT,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS answers (
    -- §10.2/D84+D59+D53: where a person's answer to the product's own question goes.
    --
    -- The product asks several questions it could not hear the answer to. "Is this departure
    -- deliberate?" was asked on every judgment and the reply died in a chat window, so the
    -- stored finding kept reading `departure: unexplained` — a FACT about the finding, read
    -- by every later judgment, and false the moment somebody explained it. A question nobody
    -- can answer decays into a wrong answer, which is worse than not asking.
    --
    -- Append-only, like every other `stated` claim here. Changing an answer writes a new row
    -- and the latest wins, so what somebody said in March survives being contradicted in
    -- June and a judgment made between the two stays explicable.
    id            TEXT PRIMARY KEY,
    subject_kind  TEXT NOT NULL,   -- 'finding' | 'gap'
    subject_key   TEXT NOT NULL,   -- finding id, or gap code
    evaluation_id TEXT,            -- for a finding: which judgment it is on
    answer        TEXT NOT NULL,   -- the closed set for that kind; never free text
    note          TEXT NOT NULL,   -- the reason, in their words. "Deliberate" with no reason
                                   -- is the same non-answer as `unexplained`, recorded as
                                   -- though it were one — and it stops the question being
                                   -- asked, which makes it strictly worse.
    said_by       TEXT NOT NULL,   -- every `stated` claim here carries who said it
    said_at       TEXT NOT NULL,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS import_batches (
    -- §10.5/D117+D118: a workbook, staged at preview and quoted by key thereafter.
    --
    -- Two problems, one shape. D117: a 500-row workbook crossed the wire twice, because the
    -- preview held the rows and then threw them away. D118: an import stopped by the time
    -- budget had no resume key, so "send them again to continue" meant resending the file —
    -- which re-imports the prefix, because `add_metrics` appends and nothing deduplicates.
    -- The result reported how many rows were left and not from WHERE.
    --
    -- `imported_through` is the whole of the fix: the count of rows this batch has already
    -- written, which is also the index to resume at. It is updated in the same transaction
    -- as the rows it counts, so a crash between the two is not a state that exists.
    id                TEXT PRIMARY KEY,
    rows              TEXT NOT NULL,   -- the workbook as given, JSON
    row_count         INTEGER NOT NULL,
    imported_through  INTEGER NOT NULL DEFAULT 0,
    created_at        REAL NOT NULL,
    finished_at       REAL             -- set when every row has been attempted
);
CREATE TABLE IF NOT EXISTS feedback_notes (
    -- §10.3: "the free-text box is where 'slide 23 should be the standard' gets captured —
    -- the highest-value sentence in the whole system".
    --
    -- Its OWN table, because the two places it was tried both corrupted something. A metrics
    -- row means "a number about outcomes": prose filed there unlocked §2.3's `verified` gate,
    -- entered §9.9's calibration denominator, was served by `reconcile_evaluation` as the
    -- campaign's actual result, and — as a `predicted` row — re-opened a finished campaign in
    -- the feedback queue as "targets only, no actuals", permanently. A sentence is not a
    -- measurement of any kind.
    id            TEXT PRIMARY KEY,
    campaign_id   TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    note          TEXT NOT NULL,
    said_by       TEXT NOT NULL,   -- §11: whose sentence this is. A client is several people
    said_at       TEXT NOT NULL,   -- and the answer to "do they still hold it" needs a date
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS context_attributions (
    -- §9.8: "A human can add an attribution note, stored as `stated`." The server records that
    -- a metric and an event overlapped; whether the event MOVED the number is exactly the
    -- judgment the review says a model will reach for, so it is a person's, it carries their
    -- name, and there is no path by which the server produces one.
    --
    -- It is also what lets a recurring holiday confound. Every November campaign in the United
    -- States overlaps Black Friday, so a fixed date confounds nothing by itself — a caveat that
    -- travels with every metric in the library is one nobody reads. A row here is somebody
    -- saying this one mattered.
    -- APPENDED, never updated — §9.4's rule, four items earlier, for the same reason: an
    -- account given before the numbers came in and one given after are two judgments about the
    -- same thing, and the pair is worth more than either. "This looked like the earthquake
    -- before the numbers and like our own pricing after" is the highest-value row this table
    -- can hold, and §9.9 is the item that would read it. `outcome_known` is stamped at the
    -- moment of the statement, because the point is to record what the person could see.
    id            TEXT PRIMARY KEY,
    campaign_id   TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    event_id      TEXT NOT NULL REFERENCES context_events(id) ON DELETE CASCADE,
    note          TEXT NOT NULL,
    stated_by     TEXT NOT NULL,
    outcome_known INTEGER NOT NULL DEFAULT 0,
    -- Whether the person says it DID bear on the result. The first version could only record
    -- yes, so the offer built on it — "say whether X is why these numbers came out as they
    -- did" — asked a question whose "no" had nowhere to go. That is the D84 failure this
    -- phase exists to fix, reintroduced in a new offer in the same diff.
    --
    -- A `no` does NOT make the outcome clean. The overlap is a fact the server computed and
    -- somebody being sure it did not matter is a person's claim; both belong in the reading,
    -- which is why a ruled-out event stays visible with the words that ruled it out. What it
    -- does stop is the question being asked again.
    bears_on      INTEGER NOT NULL DEFAULT 1,
    -- §8.5's shape again: an account can be taken back, and is kept when it is. Without this
    -- a wrong attribution could only be replaced by writing another claim — there was no way
    -- to say "I was wrong to say that".
    withdrawn_at  REAL,
    withdrawn_by  TEXT,
    withdrawn_why TEXT,
    created_at    REAL NOT NULL
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
    -- §9.9: the SERVER's tally, kept beside the person's lesson. The split this product is
    -- built on, at the one surface that grades the product: whether 3.4 fell inside 3.0-4.0 is
    -- arithmetic and is stored as arithmetic; what the miss MEANT is the sentence above, and
    -- the two must never be read as one kind of claim. `not_comparable` is stored separately
    -- and never folded into either — a calibration figure that scored unscorable predictions
    -- as passes would be this product awarding itself marks.
    counts        TEXT NOT NULL DEFAULT '{}',   -- JSON: held / missed / not_comparable
    confounded    INTEGER NOT NULL DEFAULT 0,   -- §9.8: was the outcome clean evidence?
    -- §9.9: "produce ONE record". The four assembled columns, kept — three of them are
    -- recomputed live over state that keeps moving (the drift figure is rewritten whenever
    -- the answer changes, a context link is recomputed on every read and an event can be
    -- withdrawn), so a stored lesson beside three integers left a later reader with no way to
    -- see what the lesson was about. §9.5 settled this for itself with `execution_at_save`
    -- and wrote down why; this is the same rule applied to the record that grades the product.
    record        TEXT NOT NULL DEFAULT '{}',   -- JSON: predicted / delivered / actual /
                                   -- context / scored, plus `as_of`
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
-- §9.6: every overlap query filters on the range first, and the link is recomputed on every
-- read by design, so this is the one index that carries the feature.
CREATE INDEX IF NOT EXISTS context_range_idx     ON context_events(starts_on, ends_on);
CREATE INDEX IF NOT EXISTS context_scope_idx     ON context_events(scope, scope_key);
-- SQLite cannot ADD COLUMN ... UNIQUE, so `_add_missing_columns` strips it and an UPGRADED
-- database had no uniqueness on `seed_key` at all — the schema comment promised a guarantee
-- half the installed base did not have, and the seeder's lookup has no ORDER BY, so which of
-- two duplicates got replaced was undefined. A unique INDEX is the form that survives a
-- migration.
CREATE UNIQUE INDEX IF NOT EXISTS context_seed_key_idx ON context_events(seed_key)
    WHERE seed_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS attributions_campaign_idx ON context_attributions(campaign_id);
CREATE INDEX IF NOT EXISTS feedback_notes_campaign_idx ON feedback_notes(campaign_id);
CREATE INDEX IF NOT EXISTS campaign_notices_idx ON campaign_notices(campaign_id, cleared_at);
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
    # §9.7: the shipped calendar, on the same terms as the metric registry — idempotent on a
    # stable key, so it can correct a wrong date on upgrade without doubling the calendar, and
    # it never resurrects a row a customer withdrew.
    import context
    context.seed(conn)


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


def _now_iso() -> str:
    """The same stamp `feedback` writes for a `said_at`, so a date a person is shown reads the
    same wherever it came from."""
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


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

    §10.3/§11: a tag may also carry `said_by` and `said_at` — WHO holds this opinion and when
    they said it. A client is several people with different authority and sometimes different
    opinions, and a tag with no author cannot be weighed against a contradicting one, traced
    back to whoever set it, or removed when that person leaves. Optional, because most tags
    predate the field and a missing author is an honest "nobody recorded one" rather than a
    reason to refuse the write.
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
            value, source, said_by, said_at = t.strip(), "stated", None, None
        elif isinstance(t, dict):
            raw_value = t.get("value")
            if not isinstance(raw_value, str) or not raw_value.strip():
                raise ValueError(f"tag object must have a non-empty string 'value', got {t!r}")
            value = raw_value.strip()
            source = enums.normalise(t.get("source", "stated"), field="tag 'source'",
                                     valid=VALID_TAG_SOURCES,
                                     synonyms=enums.TAG_SOURCE_SYNONYMS,
                                     allow_none=False)
            said_by = (t.get("said_by") or "").strip() or None
            said_at = (t.get("said_at") or "").strip() or None
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
        out.append({"value": value, "source": source,
                    **({"said_by": said_by} if said_by else {}),
                    **({"said_at": said_at} if said_at else {})})

    deduped: dict[str, dict] = {}
    for entry in out:
        key = entry["value"].lower()
        # Later wins on a tie, so re-recording an opinion updates who holds it rather than
        # keeping the first person's name on somebody else's answer.
        if (key not in deduped
                or (deduped[key]["source"] != "verified" and entry["source"] == "verified")
                or (deduped[key]["source"] == entry["source"])):
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
            # Carried through, not rebuilt from two keys. Reconstructing each tag as exactly
            # {value, source} silently dropped §10.3's `said_by`/`said_at` on the way back
            # out — so the author was demanded at the menu, written to the row, and invisible
            # to every reader, while the response said it had been kept.
            out.append({**e, "value": e.get("value", ""),
                        "source": e.get("source", "stated")})
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
                    commentary_checked=False, starts_on=None, ends_on=None) -> str:
    record_type = _normalise_record_type(record_type)
    status = _normalise_status(status)
    tags = normalize_tags(tags, has_actual_metrics=False)  # brand-new: no metrics can exist yet
    markets = normalize_markets(markets)
    cid = _id("camp")
    now = _now()
    if status is None and record_type == "campaign":
        status = "concluded"  # reference/stub records have no lifecycle status by default
    # §9.6: validated here like every other window, because this is the one path that wrote
    # one without going through `update_campaign`.
    starts = _checked_date(starts_on, "starts_on")
    ends = _checked_date(ends_on, "ends_on")
    if starts and ends and ends < starts:
        raise ValueError(
            f"`ends_on` ({ends}) is before `starts_on` ({starts}). A window that closes "
            f"before it opens overlaps nothing, so it would look recorded and match no event.")
    conn.execute(
        """INSERT INTO campaigns (id, title, record_type, status, tags, region, market,
                                  markets, collection, supersedes, detail, deck_text,
                                  asset_path, commentary_checked, starts_on, ends_on,
                                  created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cid, title, record_type, status, json.dumps(tags), region, market,
         json.dumps(markets), collection, supersedes, detail, deck_text, asset_path,
         1 if commentary_checked else 0, starts, ends, now, now),
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


def _overlapping_once(conn, record: dict) -> list:
    """Every event that could touch this record, fetched once per `get_campaign`.

    Memoised on the record dict itself rather than on the connection: it lives exactly as long
    as this one read, so an event recorded a moment later is picked up by the next call. A
    cache that outlived the call would be the stale-link failure §9.6 refused a join table to
    avoid.
    """
    import context

    if "_overlapping" not in record:
        widest = context.widest_window(record)
        record["_overlapping"] = ([] if not widest["starts_on"] else context.overlapping_window(
            conn, starts_on=widest["starts_on"], ends_on=widest["ends_on"],
            markets=sorted(context._market_names(record))))
    return record["_overlapping"]


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
    # §9.8: "the caveat travels with the metric EVERYWHERE it is cited". Attached here, in the
    # one function that loads metrics, rather than at each of the places that display them — a
    # hand-maintained list of call sites is a copy of the codebase, and this project has been
    # bitten by that shape four times. Every reader gets it, including readers nobody has
    # written yet.
    #
    # Guarded on both halves so the ordinary call costs nothing: a campaign with no window
    # cannot overlap anything, and a campaign with no metrics has no outcome to qualify.
    # ACTUAL rows only. A target is what somebody aimed at before anything happened and a
    # prediction is this library's own forecast; neither can be confounded by an event, and
    # calling a target "an outcome that ran through" something is the category error the
    # `metric_type` comment above guards — it would also corrupt §9.9, which scores
    # predictions against actuals.
    measured = [m for m in d["metrics"] if m["metric_type"] == "actual"]
    for metric in d["metrics"]:
        metric.update({"status": "not_an_outcome", "confounded": False, "confounded_by": [],
                       "ran_during": 0, "confounded_basis": "computed"})
    if measured:
        import context

        # Once per RECORD, then narrowed per row in memory. Computed per row it ran the
        # overlap query once per month of a workbook — and `context.record` already fans out
        # across the whole library, so §9.8 was adding two queries per campaign to it.
        for metric in measured:
            metric.update(context.confounders_for(conn, d, metric=metric,
                                                  events=_overlapping_once(conn, d)))
    d["has_metrics"] = len(d["metrics"]) > 0
    # Separately, because they answer different questions and §8.8 made the difference
    # reachable. `has_metrics` counts any row — a forecast, and now a TARGET. "Has this
    # campaign been measured" is what decides whether to go and ask for its numbers, and a
    # campaign carrying only the figure somebody was aiming at has not been measured at all.
    d["has_actual_metrics"] = any(m["metric_type"] == "actual" for m in d["metrics"])
    # §10.3: the sentences people gave about this record, in their words. The most valuable
    # content the library holds, and a table nothing reads is where it goes to die.
    d["feedback_notes"] = feedback_notes(conn, campaign_id)
    d.pop("_overlapping", None)
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
                    collection=None, supersedes=None, starts_on=None, ends_on=None,
                    window_source=None) -> bool:
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
    # §9.6: when it ran. Validated here rather than at the edge because every caller writes
    # through this one function, and a window stored as "March-ish" compares as text against
    # every event range in the library — silently, and wrongly.
    if starts_on is not None or ends_on is not None:
        # Checked against what is STORED, not only against what arrived in this call. Guarded
        # on the two-field case alone, a one-field typo fix wrote 2026-12-01 to 2026-03-31 —
        # and §9.6 then printed that window backwards inside the sentence asserting nothing
        # had been going on. Clearing an end is still allowed: an always-on campaign is
        # ordinary, and refusing a clear because the stored other end is later would make it
        # unsettable.
        current = get_campaign(conn, campaign_id) or {}
        new_start = (_checked_date(starts_on, "starts_on") if starts_on is not None
                     else current.get("starts_on"))
        new_end = (_checked_date(ends_on, "ends_on") if ends_on is not None
                   else current.get("ends_on"))
        if new_start and new_end and new_end < new_start:
            raise ValueError(
                f"that would leave this campaign's window running {new_start} to {new_end}, "
                f"which closes before it opens. A window like that overlaps nothing, so it "
                f"looks recorded and matches no event. Pass both ends to move the whole "
                f"window.")
        if starts_on is not None:
            fields.append("starts_on = ?"); params.append(new_start)
        if ends_on is not None:
            fields.append("ends_on = ?"); params.append(new_end)
        # A window a PERSON set clears the workbook mark, so a later spreadsheet row cannot
        # widen over it. Passing `window_source` explicitly is how the workbook path keeps it.
        fields.append("window_source = ?"); params.append(window_source)

    if not fields:
        return get_campaign(conn, campaign_id) is not None

    fields.append("updated_at = ?")
    params.append(_now())
    params.append(campaign_id)
    cur = conn.execute(f"UPDATE campaigns SET {', '.join(fields)} WHERE id = ?", params)
    conn.commit()
    return cur.rowcount > 0


def _checked_date(value, field: str) -> Optional[str]:
    """An ISO date, or None to clear. §9.6's arithmetic is string comparison on ISO dates —
    which is exactly right for `YYYY-MM-DD` and silently wrong for anything else."""
    import datetime

    text = str(value or "").strip()
    if not text:
        return None
    try:
        return str(datetime.date.fromisoformat(text))
    except ValueError:
        raise ValueError(
            f"`{field}` must be an ISO date (YYYY-MM-DD); got {value!r}. This is compared "
            f"against every context event's range by date arithmetic, and a string that is "
            f"not a date compares as text without failing.") from None


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


def record_notice(conn, *, code: str, campaign_id: str, detail: Optional[str] = None) -> str:
    """Persist a notice against a record (§10.1/D20).

    Re-raising the same condition on the same record updates it rather than stacking: it is
    the same outstanding job, and a queue that showed it four times would be counting how
    often somebody looked rather than what is wrong.
    """
    nid = _id("notice")
    conn.execute(
        "INSERT INTO campaign_notices (id, campaign_id, code, detail, created_at) "
        "VALUES (?,?,?,?,?) ON CONFLICT(campaign_id, code) DO UPDATE SET "
        "detail = excluded.detail, cleared_at = NULL",
        (nid, campaign_id, code, detail, _now()))
    conn.commit()
    row = conn.execute("SELECT id FROM campaign_notices WHERE campaign_id = ? AND code = ?",
                       (campaign_id, code)).fetchone()
    return row["id"]


def clear_notice(conn, notice_id: str) -> None:
    conn.execute("UPDATE campaign_notices SET cleared_at = ? WHERE id = ?",
                 (_now(), notice_id))
    conn.commit()


def open_notices(conn, campaign_id: str) -> list[dict]:
    if not _columns(conn, "campaign_notices"):
        return []
    return [dict(r) for r in conn.execute(
        "SELECT id, code, detail, created_at FROM campaign_notices "
        "WHERE campaign_id = ? AND cleared_at IS NULL ORDER BY created_at, id",
        (campaign_id,)).fetchall()]


def record_answer(conn, *, subject_kind: str, subject_key: str, answer: str, note: str,
                  said_by: str, said_at: Optional[str] = None,
                  evaluation_id: Optional[str] = None) -> str:
    """One answer to one question the product asked (§10.2)."""
    aid = _id("answer")
    conn.execute("INSERT INTO answers (id, subject_kind, subject_key, evaluation_id, answer, "
                 "note, said_by, said_at, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                 (aid, subject_kind, subject_key, evaluation_id, answer, note, said_by,
                  said_at or _now_iso(), _now()))
    conn.commit()
    return aid


def answers_for(conn, subject_kind: str, keys: Optional[list] = None) -> dict:
    """The LATEST answer per subject, keyed by subject (§10.2).

    Append-only underneath, so "the latest" is a read-time decision rather than a destroyed
    history — what somebody said in March survives being contradicted in June, and a judgment
    made between the two stays explicable.
    """
    if not _columns(conn, "answers"):
        return {}
    sql = "SELECT * FROM answers WHERE subject_kind = ?"
    args: list = [subject_kind]
    if keys is not None:
        if not keys:
            return {}
        sql += f" AND subject_key IN ({','.join('?' * len(keys))})"
        args += list(keys)
    latest: dict = {}
    for row in conn.execute(sql + " ORDER BY created_at, id", args).fetchall():
        latest[row["subject_key"]] = {
            "answer": row["answer"], "note": row["note"], "said_by": row["said_by"],
            "said_at": row["said_at"], "evaluation_id": row["evaluation_id"],
            # Always. A reader who cannot tell a person's answer from a computed fact will
            # eventually cite one as the other, which is the whole of §2.4.
            "basis": "stated",
        }
    return latest


def every_answer(conn, *, said_by: Optional[str] = None) -> list[dict]:
    """The whole override log, newest first (§10.2).

    Everything, not the latest per subject — `answers_for` gives that, and it is the wrong
    shape here. The table is append-only precisely so "this looked deliberate in March and
    turned out to be a mistake in June" survives, and a log that shows only the June row has
    thrown away the half that made keeping both worthwhile.
    """
    if not _columns(conn, "answers"):
        return []
    sql = "SELECT * FROM answers"
    args: list = []
    if said_by:
        # Case-folded. A name is not an identifier, and "show me everything Ana answered" is
        # typed by a person who does not know how she typed it last time.
        sql += " WHERE LOWER(said_by) = LOWER(?)"
        args.append(said_by.strip())
    return [{**dict(r), "basis": "stated"}
            for r in conn.execute(sql + " ORDER BY created_at DESC, id DESC",
                                  args).fetchall()]


def get_state(conn, key: str) -> Optional[str]:
    """One remembered fact about the library, or None (§10.6/D54).

    None is returned for a missing TABLE as well as a missing row, so a database written
    before this existed reads as "nothing remembered" rather than raising — the same shape
    every other additive migration here takes.
    """
    if not _columns(conn, "library_state"):
        return None
    row = conn.execute("SELECT value FROM library_state WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_state(conn, key: str, value: Optional[str]) -> None:
    if not _columns(conn, "library_state"):
        return
    conn.execute("INSERT INTO library_state (key, value, updated_at) VALUES (?,?,?) "
                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                 "updated_at = excluded.updated_at", (key, value, _now()))
    conn.commit()


def campaigns_with_open_notices(conn) -> set:
    if not _columns(conn, "campaign_notices"):
        return set()
    return {r["campaign_id"] for r in conn.execute(
        "SELECT DISTINCT campaign_id FROM campaign_notices WHERE cleared_at IS NULL")}


def add_feedback_note(conn, *, campaign_id: str, note: str, said_by: str,
                      said_at: str) -> str:
    nid = _id("note")
    conn.execute("INSERT INTO feedback_notes (id, campaign_id, note, said_by, said_at, "
                 "created_at) VALUES (?,?,?,?,?,?)",
                 (nid, campaign_id, note, said_by, said_at, _now()))
    conn.commit()
    return nid


def feedback_notes(conn, campaign_id: str) -> list[dict]:
    """What people have said about this campaign in their own words (§10.3), oldest first."""
    if not _columns(conn, "feedback_notes"):
        return []
    return [dict(r) for r in conn.execute(
        "SELECT id, note, said_by, said_at FROM feedback_notes WHERE campaign_id = ? "
        "ORDER BY created_at, id", (campaign_id,)).fetchall()]


def link_evaluation(conn, evaluation_id: str, campaign_id: str) -> None:
    conn.execute("UPDATE evaluations SET campaign_id = ? WHERE id = ?",
                 (campaign_id, evaluation_id))
    conn.commit()


def unlinked_judgment_for(conn, subject_title: str) -> Optional[dict]:
    """A judgment about this title that is not attached to any record (§9.9, D40).

    §5.2's "add this to the library" is the moment: the judgment is on screen and the record
    has just been created. Without this the store-then-reconcile sequence could not complete
    from the commonest starting point — a pitch nobody had stored yet.
    """
    row = conn.execute(
        """SELECT id, subject_title FROM evaluations
           WHERE campaign_id IS NULL AND subject_title = ? COLLATE NOCASE
             AND NOT EXISTS (SELECT 1 FROM reconciliations r WHERE r.evaluation_id = id)
           ORDER BY created_at DESC LIMIT 1""", (subject_title,)).fetchone()
    return dict(row) if row else None


def superseded_by(conn, campaign_id: str) -> Optional[str]:
    """Which record replaced this one, or None (§9.9, D120/D127).

    DERIVED, like every other reverse lookup here — `supersedes` points backwards and a cached
    forward pointer breaks on chains and on fan-in, which is the reasoning the `campaigns`
    table already records for `is_superseded`.
    """
    row = conn.execute("SELECT id FROM campaigns WHERE supersedes = ? "
                       "ORDER BY created_at LIMIT 1", (campaign_id,)).fetchone()
    return row["id"] if row else None


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
    # §9.8: the row's OWN measurement period, from the same date columns D119 already reads.
    # A workbook is one row per month, and using the campaign's whole window instead confounded
    # January's sell-through with a September earthquake.
    import context

    period = context.window_from_columns(structured or {})
    conn.execute(
        """INSERT INTO metrics (id, campaign_id, metric_type, detail, structured,
                                period_start, period_end, created_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (mid, campaign_id, metric_type, detail,
         json.dumps(structured) if structured is not None else None,
         (period or {}).get("starts_on"), (period or {}).get("ends_on"), _now()),
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
    # NOT "send the same rows". D117 staged them precisely so they do not cross the wire
    # twice, and the tool docstring now says in bold not to resend — while this sentence,
    # which is the one the model reads aloud, still told them to. The defect survived in
    # the only place the user actually sees.
    parts.append("Confirm with the `preview_id` this preview returned — the rows are held "
                 "here, so there is nothing to resend.")
    return " ".join(parts)


# How many staged workbooks to keep. A staged batch is a copy of the customer's file, and
# the feature exists to stop a 500-row workbook crossing the wire twice — paying for that
# with an unbounded copy of every workbook ever previewed would be the fix costing more than
# the defect. Finished batches go first and unfinished ones are never dropped for a newer
# one: an interrupted import is the case the resume key exists for.
MAX_STAGED_IMPORTS = 20

# How long an UNFINISHED staged workbook is kept. Unfinished batches are never pruned by the
# count rule, and they are subtracted from the cap — so twenty abandoned previews drove the
# limit to zero, deleted every finished batch, and then grew without bound themselves. Each
# is a full copy of a customer KPI workbook sitting in the database with no expiry and no
# route out through `delete_campaign`'s cascade, which is a retention position nobody wrote
# down. A week is long enough for any real resume and short enough not to be an archive.
STAGED_IMPORT_TTL_SECONDS = 7 * 24 * 60 * 60


def stage_import(conn, rows: list) -> str:
    """Keep a previewed workbook so confirming it does not resend it (§10.5/D117)."""
    batch_id = _id("preview")
    conn.execute("INSERT INTO import_batches (id, rows, row_count, created_at) "
                 "VALUES (?,?,?,?)",
                 (batch_id, json.dumps(rows, default=str), len(rows), _now()))
    # Abandoned first: an unfinished batch older than the TTL is not a resume anybody is
    # coming back for, and it is a copy of a customer's workbook. Done before the count rule,
    # because the count rule SUBTRACTS unfinished batches from the cap — so without this,
    # twenty abandoned previews drove the limit to zero, deleted every finished batch, and
    # then grew without bound themselves.
    conn.execute("DELETE FROM import_batches WHERE finished_at IS NULL AND created_at < ?",
                 (_now() - STAGED_IMPORT_TTL_SECONDS,))
    # Then oldest FINISHED, and only finished ones. A recent batch with rows still to write
    # is one somebody holds a resume key for; dropping it loses the half-written import
    # silently, because the offer naming the key lives in a transcript and the key simply
    # stops existing.
    conn.execute(
        "DELETE FROM import_batches WHERE finished_at IS NOT NULL AND id NOT IN ("
        "  SELECT id FROM import_batches WHERE finished_at IS NOT NULL "
        "  ORDER BY created_at DESC, id DESC LIMIT ?)",
        (max(0, MAX_STAGED_IMPORTS - _unfinished_imports(conn)),))
    conn.commit()
    return batch_id


def _unfinished_imports(conn) -> int:
    return conn.execute("SELECT COUNT(*) AS n FROM import_batches "
                        "WHERE finished_at IS NULL").fetchone()["n"]


def staged_import(conn, preview_id: str) -> dict:
    """The staged workbook, or a ValueError naming what went wrong (§10.5/D117).

    Never an empty batch and never a guess. "imported 0 rows" for a key the server does not
    hold reads as a successful import of nothing, which is the `nothing_to_check`-as-a-pass
    failure this project rules out everywhere else.
    """
    if not _columns(conn, "import_batches"):
        raise ValueError(
            f"{preview_id!r} is not a staged workbook — this library predates staged "
            f"imports. Send `rows` with confirm=True instead.")
    row = conn.execute("SELECT * FROM import_batches WHERE id = ?", (preview_id,)).fetchone()
    if not row:
        raise ValueError(
            f"{preview_id!r} is not a staged workbook. Preview the rows again "
            f"(confirm=False) to stage them and get a key.")
    return {**dict(row), "rows": json.loads(row["rows"])}


def _advance_import(conn, preview_id: str, through: int, done: bool, *, was: int) -> bool:
    """Compare-and-set. Returns whether this writer still owned the batch (§10.5/D118).

    `was` is the value the caller STARTED from, and the update only lands if the row still
    holds it. Without that the resume was a TOCTOU: two sessions reading `imported_through=0`
    both imported rows 0..n, and the library ended with two of every figure — the exact
    corruption D118 removed, arriving through the key that removed it. Checking at entry and
    writing at exit is not a check; it is a gap with a write on either side of it.
    """
    changed = conn.execute(
        "UPDATE import_batches SET imported_through = ?, finished_at = ? "
        "WHERE id = ? AND imported_through = ?",
        (through, _now() if done else None, preview_id, was)).rowcount
    return bool(changed)


def bulk_import_metrics(conn, rows: Optional[list] = None, *, confirm: bool = False,
                        preview_id: Optional[str] = None) -> dict:
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
    import actions
    import core
    import metrics as metrics_module

    # §10.5/D117+D118. The key is resolved FIRST, so everything below — the column diff
    # included — sees the workbook that was actually previewed rather than one sent again.
    # Both, and they disagree: the staged rows win silently, so a caller who edited a row and
    # resent it with the old key had their edit discarded with nothing in the result saying
    # so. Two sources for one input is the shape this file keeps hitting; here the quiet one
    # wins, which is the worst arrangement of it.
    if preview_id and rows is not None:
        raise ValueError(
            "send `rows` OR `preview_id`, not both. The key names a workbook this server is "
            "already holding, so the rows you sent would be silently discarded — preview "
            "again (confirm=False) to stage the ones you mean.")
    batch = staged_import(conn, preview_id) if preview_id else None
    start = 0
    if batch:
        if not batch["row_count"]:
            raise ValueError(
                f"{preview_id!r} is an empty workbook — nothing was staged, so there is "
                f"nothing to import. An import of no rows is not a successful import.")
        rows = batch["rows"]
        start = batch["imported_through"]
        if batch["finished_at"] or start >= batch["row_count"]:
            # Accepting an offer twice has to be safe: an offer survives in a transcript and a
            # model re-reading one cannot tell whether the call was already made. Same reason
            # `menu_token` refreshes rather than writing — the difference being that here the
            # safe answer is "that is done", not a fresh menu.
            return {"preview": False, "status": "already_imported", "imported": 0,
                    "errors": [], "not_processed": 0, "preview_id": preview_id,
                    "row_count": batch["row_count"],
                    "what_it_means": (
                        f"This workbook has already been worked through — all "
                        f"{batch['row_count']} row(s) attempted. Nothing was written again. "
                        f"`imported` counts what was WRITTEN on the earlier pass and is not "
                        f"reported here; rows that failed then (an unmatched title, say) "
                        f"failed then and are not retried, because they would fail again.")}
    if rows is None:
        raise ValueError(
            "send either `rows` (the workbook) or `preview_id` (the key a preview handed "
            "back). Neither was given, and an import with no rows is not an empty import.")

    columns = metrics_module.diff_columns(conn, rows)
    if not confirm:
        # The ROWS too, not only the columns. `errors: []` was asserted rather than computed,
        # so a preview saying "nothing has been stored, send with confirm=True" could be
        # followed by three hundred of five hundred rows failing on an unmatched title — it
        # had previewed the wrong half. Identity and metric_type are checked here without
        # writing anything, which is the same check the write does.
        problems = [{"row": i, **p} for i, p in enumerate(_row_problem(conn, r) for r in rows)
                    if p]
        # Staged, so confirming quotes the key rather than the workbook (§10.5/D117). Done
        # AFTER the problems are computed, so a preview that raises stages nothing.
        staged = preview_id or stage_import(conn, rows)
        return {"preview": True, "imported": 0, "errors": problems, "rows": len(rows),
                "would_import": len(rows) - len(problems),
                "columns": columns,
                "preview_id": staged,
                "what_it_means": _import_preview_sentence(rows, columns, problems),
                # §10.6/D116: "send it again with confirm=True" was an instruction in prose,
                # which is the form that gets dropped — and following it meant resending the
                # whole workbook. With the rows staged, accepting really is one step, so
                # there is no `needs` and nothing for the caller to supply.
                "next_actions": [actions.action(
                    f"Import these {len(rows) - len(problems)} row(s)",
                    "bulk_import_metrics",
                    why="Nothing has been written yet — this preview is the consent step. "
                        "The rows are held here, so saying yes does not resend the workbook."
                        + (f" {len(problems)} row(s) would be skipped and reported."
                           if problems else ""),
                    consent="ask", preview_id=staged, confirm=True)]}
    imported, errors = 0, []
    offers: dict = {}
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
    # §10.5/D118: from where this batch stopped, not from the top. `start` is 0 for an
    # unstaged workbook, so the resend path behaves exactly as it always did.
    reached = start
    for i, row in enumerate(rows[start:], start=start):
        if time.monotonic() >= deadline:
            not_processed = len(rows) - i
            break
        reached = i + 1
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
            # §9.9/D116: this loop harvests every other one-shot side effect and dropped
            # `next_actions` — which carries the reconcile offer and §9.4's revisit list. §8.8
            # is how a customer actually loads a workbook, so dropping it here is dropping it
            # on the path that matters. Gathered per campaign rather than per row.
            for offer in written.get("next_actions") or []:
                offers.setdefault((offer["tool"],
                                   str(offer.get("prefilled_args"))), offer)
        except Exception as exc:
            errors.append({"row": i, "reason": str(exc), **_retry_of(exc)})

    # Claimed BEFORE anything that commits. `_snapshot_execution_drift` below writes through
    # a raw connection, which ends the transaction — so a rollback after it had nothing left
    # to undo, and the losing session's duplicate rows survived the check that caught it.
    if batch and not _advance_import(conn, preview_id, reached,
                                     done=not not_processed, was=start):
        # Somebody else resumed this batch while we were working, and their rows are already
        # committed — so ours are the second copy. Roll the whole thing back rather than
        # commit it: `add_metrics` appends and nothing deduplicates, so the alternative is a
        # library holding two of every figure with nothing to say which pass wrote them.
        conn.rollback()
        raise ValueError(
            f"{preview_id!r} was resumed somewhere else while this import was running, so "
            f"nothing here was written — the rows would have been a second copy of ones "
            f"already imported. Call bulk_import_metrics again with the same preview_id to "
            f"continue from wherever that pass reached.")

    # Once per campaign, with every row on file — the figure a later citation carries has to
    # rest on the whole import, not on whichever row happened to be last.
    for cid in touched:
        core._snapshot_execution_drift(conn, cid)
    # §10.5/D118: in the SAME transaction as the rows it counts. A resume key committed
    # separately from the writes it describes is a key that can be wrong in both directions,
    # and the direction nobody audits is the one that re-imports.
    #
    # `reached` counts rows ATTEMPTED, not rows written: a row that failed on an unmatched
    # title is reported in `errors` and will fail identically next time, so retrying it on
    # resume would report the same error twice and never make progress.
    conn.commit()
    result = {"preview": False, "imported": imported, "errors": errors,
              "not_processed": not_processed, "columns": columns,
              **({"preview_id": preview_id} if preview_id else {}),
              **({"new_measures": list(asked.values())} if asked else {}),
              **({"newly_eligible": list(eligible.values())} if eligible else {}),
              **({"retired_measures": list(retired.values())} if retired else {}),
              **({"skipped": skipped} if skipped else {}),
              **({"next_actions": _trimmed(list(offers.values()))} if offers else {})}
    if not_processed and preview_id:
        # §10.5/D118. What this used to say was "send them again to continue — nothing
        # already imported is duplicated by doing so", and nothing enforced it: `add_metrics`
        # appends, there was no key and no dedup, so a caller who resent the workbook — the
        # obvious reading, and the only easy thing to do with a file — imported the prefix
        # twice. A sentence promising idempotency on a path that has none is worse than no
        # sentence, because it names the corrupting move as the safe one.
        result["note"] = (
            f"imported {imported} row(s) before the {config.TOOL_TIME_BUDGET_SECONDS:g}s "
            f"time budget ran out; {not_processed} of {len(rows)} were not reached. Call "
            f"bulk_import_metrics again with preview_id={preview_id!r} and confirm=True to "
            f"continue from row {reached} — it resumes rather than restarting, so do not "
            f"resend the workbook. Keep going until `not_processed` is 0; report once at "
            f"the end."
        )
        result["next_actions"] = _trimmed(
            [actions.action(f"Import the remaining {not_processed} row(s)",
                            "bulk_import_metrics",
                            why=f"The workbook is still staged here and {reached} row(s) are "
                                f"done. Resuming writes only what is left.",
                            consent="do", preview_id=preview_id, confirm=True)]
            + list(offers.values()))
    elif not_processed:
        # The unstaged path, where the honest advice is different: there IS no key, so the
        # caller has to slice the workbook themselves, and resending it whole would duplicate.
        result["note"] = (
            f"imported {imported} row(s) before the {config.TOOL_TIME_BUDGET_SECONDS:g}s "
            f"time budget ran out; the last {not_processed} of {len(rows)} were not reached. "
            f"Send ONLY those rows — the ones from index {reached} on."
            + (f" Resending the whole workbook imports the first {reached} a second time; "
               f"nothing deduplicates them." if reached else "") + " "
            f"Preview with confirm=False first to get a preview_id and this becomes a "
            f"resume rather than a slice."
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
    # §10.2/D84+D59: the answers, attached HERE — the one place a judgment's full findings
    # load, and therefore the only place an answer can be attached once and reach every
    # reader. The same reasoning §9.8 reached about `get_campaign`: a list of call sites is a
    # copy of the codebase, and this project has been bitten by that shape four times.
    #
    # Attached BESIDE `departure`, never over it. The first version rewrote `unexplained` to
    # `explained` on the argument that the stored value becomes false once somebody explains
    # it — which is true, and rewriting it was still wrong, for four reasons review found:
    #
    #   * `explained` is outside the declared vocabulary and readers ENUMERATE it. `_reread`
    #     indexes `_DEPARTURES` positionally, so `diff_campaigns` — the thing a marketer asks
    #     the instant v2 lands — crashed with `tuple.index(x): x not in tuple` BECAUSE the
    #     user had answered the question the product asked them.
    #   * `Departure` is a Literal of the three, so a model could not pass `explained` to
    #     `get_evaluation(departure=...)` at all, and asking for `unexplained` no longer
    #     returned the finding. Somebody auditing unexplained departures was shown none of
    #     the ones that had been explained. The product ate the finding.
    #   * It only worked for one of the three answers: `fixed` and `not_applicable` left the
    #     stored value alone, so the module's own argument was applied to a third of its own
    #     vocabulary.
    #   * `answers` is append-only precisely so that what somebody said in March survives
    #     being contradicted in June — and the derived field was then destructively
    #     overwritten at read time with no record of its prior value. Two opposite
    #     commitments about the same fact.
    #
    # `departure` describes THE DIFFERENCE; `settled` describes the conversation about it.
    # Readers that cared about `unexplained` meaning "still an open question" consult
    # `settled` instead — one predicate, in the few places that decide it.
    settled = answers_for(conn, "finding", [f.get("id") for f in d["findings"] if f.get("id")])
    for finding in d["findings"]:
        answer = settled.get(finding.get("id"))
        # `open` is somebody taking their answer back, so the finding reads as unanswered
        # again — the row stays in `answers` (it is append-only; the history is the point)
        # and simply stops being the current answer. Attaching it would leave every reader
        # with a `settled` block whose presence means "answered" and whose content says the
        # opposite, which is the two-fields-to-check failure this attachment exists to avoid.
        if answer and answer["answer"] != "open":
            finding["settled"] = answer
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


# ── context events (§9.6) ────────────────────────────────────────────────────

def fold(value) -> Optional[str]:
    """The ONE way a market name is compared (§9.6).

    `str.casefold` rather than `.lower()`, and stored rather than applied at query time: the
    forward direction matches in SQL and the reverse in Python, and any difference between the
    two folds means an event that says it reached a campaign the campaign does not list.
    """
    text = str(value or "").strip()
    return text.casefold() or None


def insert_context_event(conn, *, starts_on: str, ends_on, scope: str, scope_value,
                         kind: str, description: str, recorded_by: str, source=None,
                         delay_days=None, budget_change_pct=None,
                         channels_disrupted=None, seeded: bool = False,
                         seed_key=None, certainty=None, recurs_annually=None) -> str:
    """Write one event, or REPLACE the seeded row with this key (§9.6/§9.7).

    Replacing rather than inserting only when a `seed_key` is given, which only the seeder
    supplies: a calendar that doubles every time the product starts is worse than no calendar,
    and a Ramadan date corrected upstream has to be able to reach a database that already has
    the wrong one.
    """
    existing = (conn.execute("SELECT * FROM context_events WHERE seed_key = ?",
                             (seed_key,)).fetchone() if seed_key else None)
    if existing is not None and existing["withdrawn_at"] is not None:
        # A customer withdrew this seeded row — "we do not trade in that market". Re-seeding
        # over it on the next start would undo their correction silently and forever, which
        # is the one thing an upgrade must never do.
        return existing["id"]
    if existing is not None and _same_event(existing, locals()):
        # Nothing changed, so nothing is written. `INSERT OR REPLACE` on every start rewrote
        # `created_at` for all of them, touched the database file on a no-op, and silently
        # reverted any local edit to a seeded row's description or source on restart.
        return existing["id"]
    eid = existing["id"] if existing else _id("ctx")
    conn.execute(
        "INSERT OR REPLACE INTO context_events (id, starts_on, ends_on, scope, scope_value, "
        "scope_key, kind, description, source, delay_days, budget_change_pct, "
        "channels_disrupted, recorded_by, seeded, seed_key, certainty, recurs_annually, "
        "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (eid, starts_on, ends_on, scope, scope_value, fold(scope_value), kind, description,
         source, delay_days, budget_change_pct, json.dumps(list(channels_disrupted or [])),
         recorded_by, int(seeded), seed_key, certainty,
         None if recurs_annually is None else int(recurs_annually), _now()))
    conn.commit()
    return eid


def _same_event(row, values: dict) -> bool:
    """Is this seeded row already exactly what the pack says? (§9.7)"""
    return all(
        (row[column] or None) == (values.get(column) or None)
        for column in ("starts_on", "ends_on", "scope", "scope_value", "kind", "description",
                       "source", "certainty", "recurs_annually")
    ) and json.loads(row["channels_disrupted"] or "[]") == list(
        values.get("channels_disrupted") or [])


def withdraw_context_event(conn, event_id: str, *, why: str, withdrawn_by: str) -> None:
    conn.execute("UPDATE context_events SET withdrawn_at = ?, withdrawn_by = ?, "
                 "withdrawn_why = ? WHERE id = ?", (_now(), withdrawn_by, why, event_id))
    conn.commit()


def _context_row(row) -> dict:
    d = dict(row)
    d["channels_disrupted"] = json.loads(d.get("channels_disrupted") or "[]")
    d["seeded"] = bool(d.get("seeded"))
    d["recurs_annually"] = bool(d.get("recurs_annually"))
    return d


def get_context_event(conn, event_id: str) -> Optional[dict]:
    if not _columns(conn, "context_events"):
        return None  # a lookup by id has no claim to collapse; see `context_events`
    row = conn.execute("SELECT * FROM context_events WHERE id = ?", (event_id,)).fetchone()
    return _context_row(row) if row else None


def context_events(conn, *, scopes=None, starts_on=None, ends_on=None,
                   seeded=None, include_withdrawn: bool = False) -> list[dict]:
    """Events matching a set of scopes, optionally overlapping a window (§9.6).

    The overlap is done in SQL rather than in Python because it is the whole query: a library
    with years of seeded calendar events (§9.7 seeds Ramadan, Golden Week, Black Friday and
    the rest for every market) would otherwise be read into memory on every citation.

    Inclusive at both ends. An off-by-one here drops exactly the events that ran INTO a
    launch, which are the ones anybody cares about. `ends_on IS NULL` is ongoing and covers
    everything after it starts.
    """
    if not _columns(conn, "context_events"):
        # NOT `[]`. An empty list here reaches `for_campaign` as "checked, nothing overlapped"
        # — the exact collapse this module forbids — and the only way to be here is that
        # `init_db` never ran, which is a broken install rather than a quiet answer.
        raise RuntimeError(
            "this database has no `context_events` table, so nothing can be checked against "
            "the calendar. Run the server once to upgrade the schema.")
    where, params = [], []
    if not include_withdrawn:
        where.append("withdrawn_at IS NULL")
    if seeded is not None:
        where.append("seeded = ?")
        params.append(int(seeded))
    if scopes is not None:
        if not scopes:
            return []
        clauses = []
        for scope, value in scopes:
            if value is None:
                clauses.append("(scope = ?)")
                params.append(scope)
            else:
                # `scope_key`, never `scope_value COLLATE NOCASE` — see the column comment.
                clauses.append("(scope = ? AND scope_key = ?)")
                params += [scope, fold(value)]
        where.append("(" + " OR ".join(clauses) + ")")
    if ends_on is not None:
        where.append("starts_on <= ?")
        params.append(ends_on)
    if starts_on is not None:
        where.append("(ends_on IS NULL OR ends_on >= ?)")
        params.append(starts_on)
    sql = "SELECT * FROM context_events"
    if where:
        sql += " WHERE " + " AND ".join(where)
    return [_context_row(r)
            for r in conn.execute(sql + " ORDER BY starts_on, id", params).fetchall()]


def insert_attribution(conn, *, campaign_id: str, event_id: str, note: str,
                       stated_by: str, outcome_known: bool = False,
                       bears_on: bool = True) -> str:
    aid = _id("attr")
    conn.execute(
        "INSERT INTO context_attributions (id, campaign_id, event_id, note, stated_by, "
        "outcome_known, bears_on, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (aid, campaign_id, event_id, note, stated_by, int(outcome_known),
         int(bears_on), _now()))
    conn.commit()
    return aid


def attributions(conn, campaign_id: str, *, event_id=None) -> list[dict]:
    """Every account given about this campaign, oldest first (§9.8)."""
    if not _columns(conn, "context_attributions"):
        # NOT `{}`. An empty answer that means "could not look" reads as "nobody has said
        # anything", which is the collapse `context_events` raises for two functions above.
        raise RuntimeError(
            "this database has no `context_attributions` table, so what anybody said about "
            "these outcomes cannot be read. Run the server once to upgrade the schema.")
    sql = "SELECT * FROM context_attributions WHERE campaign_id = ?"
    params: list = [campaign_id]
    if event_id is not None:
        sql += " AND event_id = ?"
        params.append(event_id)
    # `bears_on` defaulted rather than indexed, so a row written before the column existed
    # reads as "yes, it bore on the result" — which is what every attribution meant when the
    # only thing this table could record was a yes.
    return [{**dict(r), "outcome_known": bool(r["outcome_known"]),
             "bears_on": bool(dict(r).get("bears_on", 1)), "basis": "stated"}
            for r in conn.execute(sql + " ORDER BY created_at, id", params).fetchall()]


def withdraw_attribution(conn, *, campaign_id: str, event_id: str, why: str,
                         withdrawn_by: str) -> None:
    conn.execute("UPDATE context_attributions SET withdrawn_at = ?, withdrawn_by = ?, "
                 "withdrawn_why = ? WHERE campaign_id = ? AND event_id = ? "
                 "AND withdrawn_at IS NULL",
                 (_now(), withdrawn_by, why, campaign_id, event_id))
    conn.commit()


def attributions_for(conn, campaign_id: str) -> dict:
    """{event_id: latest standing account} — with everything, withdrawn or not, still on file."""
    latest: dict = {}
    for row in attributions(conn, campaign_id):
        if row["withdrawn_at"] is None:
            latest[row["event_id"]] = row
        else:
            latest.pop(row["event_id"], None)
    return latest


def set_campaign_window(conn, campaign_id: str, *, starts_on, ends_on) -> None:
    conn.execute("UPDATE campaigns SET starts_on = ?, ends_on = ?, updated_at = ? WHERE id = ?",
                 (starts_on, ends_on, _now(), campaign_id))
    conn.commit()


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
    # A `superseding_version` row is NOT a results check. §6.3 offers one when v2 of a judged
    # brief lands, which is real evidence about the judgment and arrives before any outcome
    # exists — and "no reconciliation exists" closed the results loop permanently on the
    # strength of it. D85 created `basis` for exactly this distinction and nothing read it.
    # The judgment may sit on the record this one REPLACED: results land on the version that
    # ran, and the judgment stays with the brief it was made about (D120/D127).
    row = conn.execute(
        """SELECT e.id FROM evaluations e
           WHERE (e.campaign_id = ?
                  OR e.campaign_id = (SELECT supersedes FROM campaigns WHERE id = ?))
             AND NOT EXISTS (SELECT 1 FROM reconciliations r
                             WHERE r.evaluation_id = e.id
                               AND (r.basis IS NULL OR r.basis != 'superseding_version'))
           ORDER BY e.created_at DESC LIMIT 1""", (campaign_id, campaign_id)).fetchone()
    return row["id"] if row else None


def unreconciled_judgments(conn) -> list[dict]:
    """Every judgment with results on file that nobody has checked against them (§9.9).

    The item's whole diagnosis is that reconciliation "needs somebody to decide to go back and
    nobody does", and without this no surface can say how many are waiting — §9.6 added its own
    gap for precisely that reason.
    """
    return [dict(r) for r in conn.execute(
        """SELECT e.id, e.subject_title, e.campaign_id FROM evaluations e
           WHERE e.campaign_id IS NOT NULL
             AND EXISTS (SELECT 1 FROM metrics m
                         JOIN campaigns c ON c.id = m.campaign_id
                         WHERE m.metric_type = 'actual'
                           AND (c.id = e.campaign_id OR c.supersedes = e.campaign_id))
             AND NOT EXISTS (SELECT 1 FROM reconciliations r
                             WHERE r.evaluation_id = e.id
                               AND (r.basis IS NULL OR r.basis != 'superseding_version'))
           ORDER BY e.created_at""").fetchall()]


def reconcilable_judgments(conn) -> int:
    """How many judgments COULD be reconciled — the denominator `calibration` needs."""
    return conn.execute(
        """SELECT COUNT(*) AS n FROM evaluations e
           WHERE e.campaign_id IS NOT NULL
             AND EXISTS (SELECT 1 FROM metrics m
                         JOIN campaigns c ON c.id = m.campaign_id
                         WHERE m.metric_type = 'actual'
                           AND (c.id = e.campaign_id
                                OR c.supersedes = e.campaign_id))""").fetchone()["n"]


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


def evaluations_for(conn, campaign_id: str) -> list[dict]:
    """Every judgment recorded about this record (§10.1)."""
    return [dict(r) for r in conn.execute(
        "SELECT id, verdict, created_at FROM evaluations WHERE campaign_id = ? "
        "ORDER BY created_at", (campaign_id,)).fetchall()]


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
                          basis: Optional[str] = None, counts: Optional[dict] = None,
                          confounded: bool = False, record: Optional[dict] = None) -> str:
    rid = _id("recon")
    conn.execute(
        "INSERT INTO reconciliations (id, evaluation_id, actual, comparison, basis, counts, "
        "confounded, record, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (rid, evaluation_id, actual, comparison, basis, json.dumps(counts or {}),
         int(confounded), json.dumps(record or {}, default=str), _now()),
    )
    conn.commit()
    return rid


def _trimmed(offers: list) -> list:
    """`actions.trim`, imported late — `actions` is a leaf and store is imported by it."""
    import actions
    return actions.trim(offers)


def _reconciliation_row(row) -> dict:
    d = dict(row)
    d["counts"] = json.loads(d.get("counts") or "{}")
    d["confounded"] = bool(d.get("confounded"))
    d["record"] = json.loads(d.get("record") or "{}")
    return d


def reconciliations(conn) -> list[dict]:
    """Every reconciliation on file, oldest first (§9.9)."""
    if not _columns(conn, "reconciliations"):
        return []
    return [_reconciliation_row(r) for r in conn.execute(
        "SELECT * FROM reconciliations ORDER BY created_at, id").fetchall()]


def reconciliation_for(conn, evaluation_id: str) -> Optional[dict]:
    """The reconciliation of one judgment, or None."""
    row = conn.execute("SELECT * FROM reconciliations WHERE evaluation_id = ? "
                       "ORDER BY created_at DESC LIMIT 1", (evaluation_id,)).fetchone()
    return _reconciliation_row(row) if row else None


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
