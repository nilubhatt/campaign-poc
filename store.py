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

import functools
import json
import pathlib
import re
import sqlite3
import time
import uuid
from typing import Any, Optional, Union

import config
import enums
# §11.2's one guard. Imported at module level: an earlier comment here claimed `identity`
# reads this module and a top-level import would cycle — it does not (it imports `auth` and
# `config` only), and a justification that is not true is how a real constraint stops being
# recognisable when one does appear.
import identity
import vectorstore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    record_type   TEXT NOT NULL DEFAULT 'campaign',   -- campaign | reference | stub
    status        TEXT,            -- `VALID_STATUSES`: proposed | in_flight |
                                   -- concluded | cancelled | paused (campaigns only).
                                   -- §12.4/D38: the last two do NOT count as having
                                   -- run, so neither is ever a missing outcome.
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
    -- §11.6: when this library stamped "we never looked" onto this record's authorship. The
    -- item asks for the import date explicitly, and without it "nobody looked" is undated —
    -- so a record from before §2.5 reads identically to one whose extraction failed today.
    authorship_backfilled_at TEXT,
                                    -- and speaker notes? (§5.3/D17) A warning lives for one
                                    -- response; this is what lets gaps() still report months
                                    -- later that a deck's commentary was never looked at
    asset_path    TEXT,            -- stored original file (relative to ASSET_DIR)
    -- §12.4: WHERE THE WORK LIVES, which `asset_path` is not. That is a file this product
    -- copied; this is the Figma board, the Drive folder, the DAM record — what somebody
    -- reading a judgment six months later opens to look at the thing being judged.
    -- §13.2/D95: `facts.compute` over this record's body, as JSON, with the key it was
    -- computed under. Reading the library re-scanned every deck's text on every call —
    -- measured at 2.41s per `gaps()` over 200 decks of 38,000 characters, essentially all of
    -- it inside the channel regexes — and the answer does not change between two reads.
    --
    -- **The key is the body AND the rulebook**, which is where D95's own premise is wrong.
    -- The row says facts are "deterministic on body text"; they are not. `facts.compute`
    -- reads `rulebook.vocabulary("channels")` for the checklist and `rulebook.rules()` for
    -- the guardrail `watch_for` words, so one unchanged string computes different facts
    -- before and after a customer writes their rules. Keyed on the text alone, every record
    -- stored before the rulebook existed would go on reporting `guardrails: nothing_to_check`
    -- — the product saying there are no rules to check about a library whose rules were just
    -- written, with `nothing_to_check` doing the work of a pass.
    --
    -- A READ-THROUGH cache: the key is checked on every read and a miss recomputes and
    -- rewrites. So it cannot go stale even if a write path forgets to refresh it, which is
    -- this codebase's most repeated defect and not something to leave to having found every
    -- one of them.
    computed_facts      TEXT,
    computed_facts_key  TEXT,
    asset_link    TEXT,
    -- §12.4/D102: what KIND of campaign this is. The review asks for "the checklist for a
    -- campaign TYPE" and renders it "Expected for a store launch"; nothing carried that, so
    -- market stood in — and market is the wrong axis, because it made a store-launch measure
    -- expected of every campaign in that market and of no store launch anywhere else.
    -- Free text with a DECLARED vocabulary behind it (§12.2), which is why it could not be
    -- built before: unfolded, "Store Launch" and `store_launch` are two checklists.
    campaign_type TEXT,
    -- §12.4/D103: who ran it. §8.3's gate is "across at least two PARTNERS or markets" and
    -- nothing recorded the partner, so market stood in for that too. Two campaigns with one
    -- partner in two markets is weaker evidence of a general rule than two partners in one.
    partner       TEXT,
    -- §12.4/D15: the VERDICT on a returned deck, which is the half of `approval_notes` that
    -- was genuinely missing. The NOTES are the tracked comments §2.5 ingests — that is the
    -- decision the item asks for, and it is "yes, they are": a returned deck's comments are
    -- what the client wrote when they sent it back, and they arrive with an author, an anchor
    -- to the slide and a date, which no retyped free-text box could carry.
    --
    -- A second `approval_notes` column would be a parallel store for the same thing, worse in
    -- every way and drifting from the first day. What the comments do NOT say is whether the
    -- deck came back signed off — "the timing is not" is a note, and "approved with changes"
    -- is a different fact that a reader needs first.
    approval      TEXT,            -- approved | approved_with_changes | rejected | withdrawn
    approval_note TEXT,            -- one line: what the sign-off was conditional on
    approval_by   TEXT,            -- §11.2: whose sign-off this is, and it must be a PERSON
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
    -- §12.4/D102: the campaign TYPES it graduated on, when its evidence was type-coherent.
    -- The review's sentence is "expected for a store launch: budget, reach, footfall uplift"
    -- and the checklist could only be keyed on market, so a measure learned entirely from
    -- store launches was expected of every campaign in that market — a seeding brief in Peru
    -- reported as missing footfall uplift — and of no store launch anywhere else. Market was
    -- standing in for a thing it is not.
    --
    -- Filled only when EVERY campaign carrying the measure shares one type. Mixed evidence
    -- says nothing about type, and guessing a type out of it would be a narrower checklist
    -- than the evidence earns. `[]` therefore means "keyed on market", which is what every
    -- row written before this column existed means and is the behaviour it had.
    expected_for_types TEXT NOT NULL DEFAULT '[]',
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

CREATE TABLE IF NOT EXISTS erasures (
    -- §11.7: every time a person's name was removed from this library.
    --
    -- WITHOUT the name. A log recording what it removed would be the personal data again, in
    -- the one table nobody would think to erase — so it holds the pseudonym that replaced it,
    -- which is what a reader needs to understand a record that now says `erased-3f2a1b9c`.
    --
    -- A record silently rewritten is a record nobody can trust: a saved judgment whose
    -- evidence changed under it needs somewhere that says the change happened, on whose
    -- authority, and why.
    id          TEXT PRIMARY KEY,
    pseudonym   TEXT NOT NULL,
    mentions    INTEGER NOT NULL,
    why         TEXT NOT NULL,
    said_by     TEXT NOT NULL,
    erased_at   TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS reactions (
    -- §11.5: what people made of a campaign, APPEND-ONLY, both sides kept.
    --
    -- `campaigns.tags` replaces on write, so R. Vega recording `liked` in March and A. Duarte
    -- recording `not_liked` in June left only June — not superseded, not outvoted, gone, with
    -- nothing saying March was ever there. That is the most expensive thing this library can
    -- lose: a campaign two people disagreed about is worth MORE as evidence than one everybody
    -- liked, because the disagreement is where the client's actual taste lives, and the whole
    -- premise here is reasoning from what this client thinks.
    --
    -- Nothing in this table decides who is right. D10 and §11.5 both say it: authority order
    -- is configured in the rulebook (§12), never inferred. Preferring the newer view, or the
    -- client's, or the one from a grander job title, would be this library inventing an
    -- authority nobody granted it — and §2.5 already refused that once, because a PDF export
    -- turns speaker notes into annotations, so even the FORMAT cannot say whose words weigh
    -- more.
    id           TEXT PRIMARY KEY,
    campaign_id  TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    axis         TEXT NOT NULL,   -- 'reaction' | 'performance': what the value is ABOUT
    value        TEXT NOT NULL,
    source       TEXT NOT NULL,   -- 'stated' | 'verified', as everywhere else
    said_by      TEXT NOT NULL,
    said_at      TEXT NOT NULL,
    role         TEXT,            -- §11.4: as stated at the time
    created_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS authorship (
    -- §11.1–§11.4: who made a decision, and how much that claim is worth.
    --
    -- Nine write paths took a person's name as a free string and the server had no idea what
    -- it was receiving — D105, about the sharpest of them: "nothing distinguishes a real
    -- confirmation from one the model wrote itself." `confirmed_by` promotes a rule to the
    -- checklist where every future brief in its markets is judged against it.
    --
    -- Two facts, kept apart, because they are not the same claim:
    --   on_behalf_of  whose judgment this is. Only a person can say, so it is `stated` and
    --                 required. Deriving it would file every decision under whoever typed.
    --   captured_*    the account that made the call. The server derives it and never
    --                 accepts it, and `captured_method` is what keeps `verified` honest —
    --                 `stdio_local` means "the desktop account this runs as", not "this
    --                 person authenticated".
    --
    -- One row per decision, never updated: §11.5's rule, and the same reason `answers` is
    -- append-only. What somebody's role was AT THE TIME is the fact worth keeping, and a
    -- person who changes team should not retroactively have decided things as their new one.
    id                TEXT PRIMARY KEY,
    subject_kind      TEXT NOT NULL,   -- 'context_event' | 'correction' | 'answer' | ...
    subject_key       TEXT NOT NULL,
    on_behalf_of      TEXT NOT NULL,
    note              TEXT,            -- §12.4: WHY, in the words of whoever decided. Every
                                       -- write on this table demands a reason from its caller
                                       -- and there was nowhere to put one: `update_asset`
                                       -- required `why`, returned it, and dropped it on the
                                       -- floor. "Somebody decided these were the delivered
                                       -- shots" without the reason is the half of the record
                                       -- a reader six months later does not need.
    on_behalf_of_role TEXT,            -- §11.4: as STATED at the time, never looked up later
    captured_source   TEXT NOT NULL,   -- identity.AUTHOR_SOURCES
    captured_method   TEXT NOT NULL,   -- identity.METHODS
    captured_account  TEXT,            -- os_user, or the SSO subject
    captured_host     TEXT,
    captured_display  TEXT,
    channel           TEXT,            -- §11.4: stdio, http, import
    session_id        TEXT,            -- §11.4
    captured_at       TEXT NOT NULL,   -- §11.4
    created_at        REAL NOT NULL
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
    -- §12.3/D108: a correction the CUSTOMER declared in their own rulebook applies in EVERY
    -- market rather than in the ones it was seen in. `expected_in = []` already means "no
    -- checklist", so an empty market list could not carry "everywhere" — nine of the ten
    -- corrections the example ships reached no judgment anywhere while the tool reported them
    -- in force, which is the stored-and-never-applied failure D108 exists to remove.
    --
    -- A column rather than a sentinel in `expected_in`, because the two facts are different:
    -- a LEARNED rule is expected where the evidence put it, which is §8.3's whole anti-capture
    -- argument, and only somebody who wrote the rule down can say "everywhere".
    applies_everywhere INTEGER NOT NULL DEFAULT 0,
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
CREATE TABLE IF NOT EXISTS correction_scope (
    -- WHEN a rule started applying where it applies, which is a different fact from when
    -- somebody confirmed it (§8.7/§12.3). `confirmed_at` is preserved across a re-confirmation
    -- on purpose — it is the audit field, and rewriting it would make the replay claim a
    -- judgment that CITES a rule was never checked against it — so a rule whose SCOPE changed
    -- later was invisible to the replay: the upgrade that put ten house rules in force
    -- everywhere offered "see which judgments this now applies to", and the report was empty
    -- because every one of those rules had been confirmed before the judgments were written.
    --
    -- A history rather than one `scope_changed_at`, because a single timestamp cannot tell
    -- widening from narrowing: dropping SEA from a rule would make every LATAM judgment it had
    -- already been applied to look unchecked. With the history the question is asked exactly —
    -- did this rule reach THIS brief's markets on the day it was judged?
    id            TEXT PRIMARY KEY,
    correction_id TEXT NOT NULL REFERENCES corrections(id) ON DELETE CASCADE,
    changed_at    REAL NOT NULL,
    expected_in   TEXT NOT NULL DEFAULT '[]',
    applies_everywhere INTEGER NOT NULL DEFAULT 0,
    -- 0 when the rule stopped applying ANYWHERE, which is a scope like any other and the one
    -- a table of markets alone would miss: a rule somebody set aside, a brief judged while it
    -- was withdrawn, and the rule then reopened and confirmed again with the SAME markets —
    -- no change to compare, so the history said it had applied throughout and the report said
    -- nothing about the brief it was never applied to. With this column the table is an
    -- IN-FORCE history rather than a scope history, which is the larger and more useful thing.
    standing      INTEGER NOT NULL DEFAULT 1,
    -- 1 when this row was DERIVED rather than recorded: the scope a rule already had on a
    -- database written before this table, assumed to have held since it was confirmed. That
    -- is the assumption this report made before the history existed, and marking it is what
    -- keeps a derived past distinguishable from a recorded one (§2.4's `basis`).
    seeded        INTEGER NOT NULL DEFAULT 0,
    -- The highest `evaluations` rowid at the moment this change was written, which is the one
    -- thing that ORDERS a scope change against a judgment. Two rows in two tables carrying
    -- wall-clock floats have no order between them when the floats are equal — and Windows
    -- measures `time.time()` in whole milliseconds, so equal is reachable. Read through the
    -- timestamp alone, a rule widened in the same tick as a judgment counted as already in
    -- force and the brief it was never checked against vanished from the report. 0 on a row
    -- written before this column, and on a seeded one, both of which mean "older than every
    -- judgment on file" — which is what they are.
    after_evaluation INTEGER NOT NULL DEFAULT 0
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
    space         TEXT NOT NULL DEFAULT 'campaign',  -- vectorstore.SPACES: campaign, asset,
                                    -- commitment. CLIP fills two of them and the text embedder
                                    -- the third, so two model names across spaces is normal
                                    -- and not a mixed index
    created_at    REAL NOT NULL
);

-- §12.4/D106: every occasion a measure was retired or revived, not only the latest.
-- `retired_at` is one timestamp and a measure that cycles in and out has no record of having
-- done so — which is the whole point of demoting rather than deleting: "we used to track
-- this" is an answer somebody needs, and "we have twice" is a different one.
--
-- No `said_by`. This library retires a measure nobody has reported for N campaigns and revives
-- one somebody measures again; there is no person in the loop, and naming one would be the
-- product signing its own name to a decision (§11.2). `why` is what it OBSERVED.
--
-- IN `_SCHEMA`, and this is not a formatting preference. It was declared down in `_INDEXES`
-- beside its own index, where it worked — fresh installs got the table — and was INVISIBLE to
-- `_declared_tables()`, which derives from `_SCHEMA` alone. That is D89's defect exactly: a
-- column added to it later would reach a new install and never an upgraded one, and the two
-- sweeps that read `_SCHEMA` to check every stored name (`test_verdict_stamp`'s upgrade guard,
-- `test_personal_data`'s name-column scan) would walk straight past it. Tables belong here;
-- `_INDEXES` gets the index and nothing else.
CREATE TABLE IF NOT EXISTS measure_history (
    id        TEXT PRIMARY KEY,
    canonical TEXT NOT NULL,
    what      TEXT NOT NULL,      -- retired | revived
    why       TEXT NOT NULL,      -- what the library observed, in its own words
    at        REAL NOT NULL
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
CREATE INDEX IF NOT EXISTS measure_history_idx ON measure_history(canonical, at);
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
    # §12.4/D106: a measure RETIRED before this table existed has no `retired` occasion, so
    # the first thing its history said after the upgrade was "revived" — a record that reads
    # as though the library brought back something it had never demoted. The retirement is
    # derivable: `retired_at` is on the row and is the timestamp it happened at. Seeded once,
    # and only for rows that have no history at all, so an upgrade cannot invent an occasion
    # beside one the new code already recorded.
    if _columns(conn, "measure_history") and _columns(conn, "metric_registry"):
        for row in conn.execute(
                "SELECT canonical, retired_at FROM metric_registry WHERE status = 'retired' "
                "AND retired_at IS NOT NULL AND canonical NOT IN "
                "(SELECT canonical FROM measure_history)").fetchall():
            conn.execute(
                "INSERT INTO measure_history (id, canonical, what, why, at) VALUES (?,?,?,?,?)",
                (_id("mhist"), row["canonical"], "retired",
                 "retired before this library recorded its reasons", row["retired_at"]))

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


# Moved to `store_campaigns.py`, whole and unchanged. Re-exported because `store.X` is what
# every caller and test already says, and a refactor that renames the surface is a
# refactor that changes behaviour.
from store_campaigns import (  # noqa: E402,F401
    AUTHOR_UNKNOWN,
    REACTION_AXES,
    VALID_RECORD_TYPES,
    VALID_STATUSES,
    VALID_TAG_SOURCES,
    _CARRIES_AN_AUTHOR,
    _VALID_RECORD_TYPES,
    _VALID_STATUSES,
    _VALID_TAG_SOURCES,
    _any_spelling,
    _as_authorship,
    _checked_date,
    _declared,
    _folded_name,
    _forget_commitment_vectors,
    _normalise_record_type,
    _normalise_status,
    _overlapping_once,
    _parse_stored_markets,
    _parse_stored_tags,
    _parse_tag_query,
    _short,
    add_feedback_note,
    answers_for,
    attach_deck_to_campaign,
    attributions_on_file,
    author_of,
    authorship_for,
    campaigns_with_open_notices,
    checked_link,
    checked_supersedes,
    clear_notice,
    delete_campaign,
    disagreement_on,
    erasures,
    every_answer,
    feedback_notes,
    filter_campaign_ids,
    fold_campaign_type,
    fold_vocabulary,
    get_campaign,
    get_state,
    get_superseded_campaign_ids,
    insert_campaign,
    link_evaluation,
    list_campaigns,
    mark_authorship_backfilled,
    mark_embedded,
    normalize_markets,
    normalize_tags,
    open_notices,
    reactions_for,
    recent_voices,
    record_answer,
    record_authorship,
    record_erasure,
    record_notice,
    record_reaction,
    records_without_authorship_backfill,
    set_state,
    spellings_of,
    superseded_by,
    supersession_chain,
    unlinked_judgment_for,
    update_campaign,
)

# ── campaign chunks (§6.1: one vector per chunk, not per campaign) ───────────

def insert_chunks(conn, campaign_id: str, texts: list[str], *, kind: str = "body",
                  sources: Optional[list[dict]] = None) -> list[str]:
    """`sources` carries one {kind, author, date, anchor} per text for commentary, so a
    match can say who said it and where, rather than only that the deck mentions it."""
    now = _now()
    ids = []
    # §13.4/D100: PER KIND. Allocated across the whole campaign, a body that gained a
    # position landed after the commentary — measured as `body 0-6, commentary 7, body 8-13`,
    # one layer's sequence interrupted by another's — and a record that gained its commentary
    # BEFORE its body, which is what attaching a deck to something somebody had already
    # annotated looks like, put a client's remark at index 0. "Chunk 0 is the summary" was
    # the invariant, and it was silently gone.
    #
    # Safe to change because every reader orders WITHIN a kind (`body_chunks`,
    # `get_commentary`, `text_on_file`) and nothing treats the index as unique across kinds —
    # so an existing database whose numbering overlaps between layers reads exactly as before.
    # §13.3 depends on the body's RELATIVE order, which per-kind allocation preserves by
    # construction rather than by accident.
    start = conn.execute(
        "SELECT COALESCE(MAX(chunk_index), -1) + 1 FROM campaign_chunks "
        "WHERE campaign_id = ? AND kind = ?", (campaign_id, kind)).fetchone()[0]
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


def body_chunks(conn, campaign_id: str) -> list[dict]:
    """This record's BODY chunks, in order (§13.3/D97).

    In `chunk_index` order because the order is load-bearing: chunk 0 is the title-and-detail
    summary and the deck's sections follow it, which is what lets a rebuild compare position
    by position and re-embed only what changed.
    """
    return [dict(r) for r in conn.execute(
        "SELECT id, chunk_index, text, embedded FROM campaign_chunks "
        "WHERE campaign_id = ? AND kind = 'body' ORDER BY chunk_index", (campaign_id,))]


def set_chunk_text(conn, chunk_id: str, text: str) -> None:
    """Replace one chunk's words, keeping its id and its position (§13.3/D97).

    `embedded` goes back to 0: the words changed, so whatever vector was against this id is
    for text this chunk no longer holds, and leaving the flag set would report the record as
    fully searchable by wording that is not in it.
    """
    conn.execute("UPDATE campaign_chunks SET text = ?, embedded = 0 WHERE id = ?",
                 (text, chunk_id))
    conn.commit()


def delete_chunks(conn, chunk_ids: list) -> None:
    if not chunk_ids:
        return
    conn.execute(f"DELETE FROM campaign_chunks WHERE id IN "
                 f"({','.join('?' * len(chunk_ids))})", list(chunk_ids))
    conn.commit()


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

# §12.4/D15. `withdrawn` is the agency pulling it rather than the client refusing it — a
# different fact about the same deck, and the one a later reader most often needs to explain
# why a promising brief has no results.
VALID_APPROVALS = ("approved", "approved_with_changes", "rejected", "withdrawn")

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


def get_asset(conn, asset_id: str) -> Optional[dict]:
    """One asset, or None (§12.4/D123)."""
    row = conn.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone()
    return dict(row) if row else None


def set_asset_phase(conn, asset_id: str, phase: str) -> bool:
    """Correct what an image IS evidence of (§12.4/D123).

    `phase` decides whether an image is BRIEFED creative or what actually ran, and §9.1 says
    everything falls out of that — so a model that guessed `proposed` on fourteen photographs
    had produced an unrepairable record, because nothing could set it.
    """
    changed = conn.execute("UPDATE assets SET phase = ? WHERE id = ?",
                           (phase, asset_id)).rowcount
    conn.commit()
    return bool(changed)


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


def provenance_knows_spaces(conn) -> bool:
    """Whether `vector_provenance` can say which space a row belongs to.

    It predates its own `space` column. A caller that only needs rows it can name by id is
    fine without it — `vector_models` returns everything and the ids do the filtering — but a
    caller asking "which models made the ASSET vectors" cannot be answered on that schema at
    all, and answering it with every row would report the text embedder as an image model.
    """
    return "space" in _columns(conn, "vector_provenance")


def vector_models(conn, *, space: str) -> dict:
    """`{vector_id: model}` for one space — which weights made each vector (§13.6/D126).

    The schema tolerance is the point, and it is HERE rather than in each caller because there
    are now three of them. `vector_provenance` predates its own `space` column, so a query
    naming that column RAISES on an older install rather than returning nothing — and a caller
    that wraps the query in a bare `except` then reads the exception as "nothing is stale",
    which is the check silently not running on exactly the databases old enough to have lived
    through a model change. `embedding_models` already handled this; a second copy of the
    handling that got it wrong is the shape this file keeps hitting.

    On that older schema there is no space to filter by, so every row comes back and the caller
    filters by the ids it actually asked about.
    """
    columns = _columns(conn, "vector_provenance")
    if not columns:
        return {}
    if "space" not in columns:
        rows = conn.execute("SELECT vector_id, model FROM vector_provenance").fetchall()
    else:
        rows = conn.execute("SELECT vector_id, model FROM vector_provenance WHERE space = ?",
                            (space,)).fetchall()
    return {r["vector_id"]: r["model"] for r in rows}


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
        d = _as_metric(row)
        out[d["canonical"]] = d
    return out


def metric_entry(conn, name: str) -> Optional[dict]:
    """ONE measure, read as one row (§13.3/D107).

    `describe` asked for the whole registry and then `.get(name)` — a full table read for a
    single-row lookup, twice inside one `metrics.record`. Parsed by the same `_as_metric` the
    full read uses, so the two cannot disagree about what an entry is.
    """
    if not _columns(conn, "metric_registry"):
        return None
    row = conn.execute("SELECT * FROM metric_registry WHERE canonical = ?", (name,)).fetchone()
    return _as_metric(row) if row else None


def expected_metrics(conn) -> dict:
    """Only the measures ON a checklist (§13.3/D107).

    `retire_stale` reads the registry to find `expected` ones and skips every other row; the
    database can do that filtering.
    """
    if not _columns(conn, "metric_registry"):
        return {}
    out = {}
    for row in conn.execute(
            "SELECT * FROM metric_registry WHERE status = 'expected'").fetchall():
        d = _as_metric(row)
        out[d["canonical"]] = d
    return out


def _as_metric(row) -> dict:
    """One registry row, parsed. The one implementation, so a narrow read and a full one
    cannot come back shaped differently."""
    d = dict(row)
    d["aliases"] = json.loads(d["aliases"] or "[]")
    d["markets"] = json.loads(d["markets"] or "[]")
    d["expected_in"] = json.loads(d.get("expected_in") or "[]")
    # §12.4/D102. `.get` because an upgraded database reaches this before
    # `_add_missing_columns` has run on the process that opened it, and `[]` is the right
    # answer there: keyed on market, which is what it was.
    d["expected_for_types"] = json.loads(d.get("expected_for_types") or "[]")
    d["answered"] = bool(d.get("answered"))
    d["surfaced"] = bool(d.get("surfaced"))
    return d


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


def _widen_markets(conn, markets: list, campaign_id: Optional[str]) -> list:
    """Every market this campaign counts towards, added to the ones already recorded (§13.5).

    **Through `markets_of`, and every one of them.** A campaign that ran in MX and CO counts
    towards both, and reading `market or region` alone counted it as zero — the fifth
    implementation of a question D55 and D88 both record drifting. Folded, so "LATAM" and
    "latam" cannot satisfy a two-market gate between them; the first spelling seen is the one
    kept, because it is what somebody typed.

    ONE implementation. It stood character-for-character in `touch_metric` and
    `touch_correction`, which is D114's point: the parts that DECIDE were unified long ago,
    and this is the identical half nobody merged — so it is where the sixth implementation
    would have come from.
    """
    import learning

    if not campaign_id:
        return markets
    # Through `learning.fold_markets`, which IS "one name per market, first spelling kept" —
    # the same sentence this function's own docstring says. The first version of this helper
    # open-coded the dedupe, which made it a third copy of the thing §13.5 exists to remove:
    # an extraction that adds a copy of what it is extracting. Review caught it.
    #
    # It also fixes what the hand-rolled version cost on the merge path, which calls this once
    # per sighting: `seen` was rebuilt from every market accumulated so far on every call, and
    # each fold reaches the rulebook vocabulary, so N sightings over M markets was O(N·M)
    # vocabulary walks where the old inline loop built `seen` once.
    return learning.fold_markets(list(markets) + markets_of(get_campaign(conn, campaign_id)
                                                            or {}))


def _market_scope(markets: Optional[list], alias: str) -> tuple:
    """`(sql, params)` restricting a count to campaigns in these markets (§13.5).

    The campaign's own market OR region OR anything in its `markets` list — the same question
    `markets_of` answers, asked in SQL. LIKE on the JSON is deliberate: the list is a JSON
    array of names and an exact match would miss a multi-market record.

    ONE implementation, for the reason above: `campaigns_that_skipped` and
    `campaigns_that_skipped_correction` built this same fragment and packed these same
    parameters, differing only in a table alias — so a change to how a market is matched, and
    §12.2 changed exactly that, had to land twice or the two answers diverged in silence.
    """
    # A LITERAL, never anything a caller was handed. Both call sites pass "v" and "s"; this
    # is the first function here to take a table alias and put it in a query, and the assert
    # is what keeps it the last one that could be given something else.
    assert alias.isidentifier(), f"table alias must be an identifier, got {alias!r}"
    folded = [f for f in {fold_market(m) for m in (markets or [])} if f]
    if not folded:
        return "", []
    clause = " OR ".join(
        ["LOWER(c.market) = ?", "LOWER(c.region) = ?", "LOWER(c.markets) LIKE ?"] * len(folded))
    params: list = []
    for f in folded:
        params += [f, f, f'%"{f}"%']
    return (f" AND EXISTS (SELECT 1 FROM campaigns c WHERE c.id = {alias}.campaign_id "
            f"AND ({clause}))"), params


def touch_metric(conn, canonical: str, *, campaign_id: Optional[str] = None) -> None:
    """Record that this measure was seen again — §8.3's graduation gate and §8.5's retirement
    both read these, and recording them from the first write is what stops the history being
    missing for exactly the measures that arrived before anybody thought about it."""
    row = conn.execute("SELECT first_seen, times_seen, markets FROM metric_registry "
                       "WHERE canonical = ?", (canonical,)).fetchone()
    if not row:
        return
    markets = _widen_markets(conn, json.loads(row["markets"] or "[]"), campaign_id)
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
    if _columns(conn, "commitment_vectors"):
        conn.execute("DELETE FROM commitment_vectors WHERE vector_id = ?",
                     (f"commitment:{commitment_id}",))
    # The same cleanup as the campaign-wide path, for the same reason (§13.6/D126). An
    # extractor withdrawing its own earlier reading leaves the vector AND the row that says
    # which weights made it, or the provenance outlives the thing it is about.
    forget_vector_models(conn, [f"commitment:{commitment_id}"])
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
    # §13.5/D114: the THIRD copy of the market-fold loop, found by mutating the shared one
    # and watching a test that should have felt it stay green. `touch_metric` and
    # `touch_correction` were the two the row names; a merge re-deriving the same breadth had
    # a third, which is exactly the argument for there being one.
    markets: list = []
    for row in rows:
        markets = _widen_markets(conn, markets, row["campaign_id"])
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
    # WHEN it stopped applying, for §8.7: a brief judged while this rule was withdrawn was not
    # checked against it, and if the rule comes back with the same markets there is no scope
    # change to notice — so the report said nothing about the one judgment it should.
    withdraw_correction_scope(conn, correction_id)
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
    markets = _widen_markets(conn, json.loads(row["markets"] or "[]"), campaign_id)
    now = _now()
    conn.execute("UPDATE corrections SET first_seen = COALESCE(first_seen, ?), last_seen = ?, "
                 "times_seen = times_seen + 1, markets = ? WHERE id = ?",
                 (now, now, json.dumps(markets), correction_id))
    conn.commit()


def correction_scope_history(conn, correction_id: str) -> list:
    """Every scope this rule has had, oldest first. Empty for a database written before the
    history existed, which is what makes the caller's fallback necessary rather than tidy."""
    if not _columns(conn, "correction_scope"):
        return []
    return [{"changed_at": row["changed_at"],
             "expected_in": json.loads(row["expected_in"] or "[]"),
             "applies_everywhere": bool(row["applies_everywhere"]),
             # A row from before this column existed is a row written by a graduation, which
             # is the only thing that wrote one: standing.
             "standing": bool(row["standing"]) if "standing" in row.keys() else True,
             "basis": ("heuristic" if ("seeded" in row.keys() and row["seeded"])
                       else "computed"),
             "after_evaluation": (row["after_evaluation"]
                                  if "after_evaluation" in row.keys() else 0)}
            for row in conn.execute(
                # `rowid`, not `id`: the ids are random, and two changes inside one clock tick
                # — which Windows measures in whole milliseconds — would then order
                # arbitrarily. Insertion order is the order they happened in.
                "SELECT * FROM correction_scope WHERE correction_id = ? "
                "ORDER BY changed_at, rowid", (correction_id,))]


def evaluation_order(conn) -> dict:
    """`{evaluation_id: rowid}` — the order judgments were actually written in.

    The one ordering a wall-clock float cannot give when two of them are equal. Fetched once
    per report rather than per judgment: this answers a question about every row at once.
    """
    if not _columns(conn, "evaluations"):
        return {}
    return {row["id"]: row["seq"]
            for row in conn.execute("SELECT id, rowid AS seq FROM evaluations")}


def correction_scope_at(conn, correction_id: str, when: float, *,
                        judgment_seq: Optional[int] = None) -> Optional[dict]:
    """What this rule applied to at `when` — including `standing: False` for withdrawn — or
    None if nothing on file covers that date at all.

    Three pasts, not two, and the caller needs them apart. `None` means nothing was recorded
    by then: the rule was not standing yet. A row with `standing: False` means it HAD been
    standing and somebody had set it aside. A row with markets that do not reach the brief
    means it was standing elsewhere. Collapsed into None, the report said a withdrawn rule's
    "scope changed" — a cause the library cannot back, about a rule whose scope never moved.
    """
    in_effect = None
    for change in correction_scope_history(conn, correction_id):
        if change["changed_at"] > when:
            break
        # The same instant, and `judgment_seq` says which of the two was written first: a
        # change recorded when this judgment already existed came after it, whatever the two
        # floats say.
        if (change["changed_at"] == when and judgment_seq is not None
                and change["after_evaluation"] >= judgment_seq):
            break
        in_effect = change
    return in_effect


def _write_correction_scope(conn, correction_id: str, *, when: float, markets: list,
                            applies_everywhere: bool, standing: bool = True,
                            seeded: bool = False) -> None:
    if not _columns(conn, "correction_scope"):
        return
    # WHICH JUDGMENTS ALREADY EXISTED. A seeded row describes a past older than anything on
    # file, so it claims nothing: 0 sorts before every judgment.
    after = 0 if seeded else (conn.execute(
        "SELECT COALESCE(MAX(rowid), 0) AS n FROM evaluations").fetchone()["n"]
        if _columns(conn, "evaluations") else 0)
    conn.execute(
        "INSERT INTO correction_scope (id, correction_id, changed_at, expected_in, "
        "applies_everywhere, standing, seeded, after_evaluation) VALUES (?,?,?,?,?,?,?,?)",
        (_id("scope"), correction_id, when, json.dumps(sorted(markets)),
         1 if applies_everywhere else 0, 1 if standing else 0, 1 if seeded else 0, after))


def withdraw_correction_scope(conn, correction_id: str) -> None:
    """Record that this rule stopped applying (§8.6's set-aside, §8.7's report).

    Called by `set_aside_correction`, which is the only thing that takes a standing rule out
    of force. `expected_in` is left alone on the row itself — where a rule USED to apply is
    part of its record — and this says WHEN it stopped, which is the fact the replay needs.
    """
    history = correction_scope_history(conn, correction_id)
    if history and not history[-1]["standing"]:
        return
    was = get_correction(conn, correction_id) or {}
    if not history and was.get("confirmed_at"):
        _write_correction_scope(conn, correction_id, when=was["confirmed_at"],
                                markets=was.get("expected_in") or [],
                                applies_everywhere=bool(was.get("applies_everywhere")),
                                seeded=True)
    _write_correction_scope(conn, correction_id, when=_now(),
                            markets=was.get("expected_in") or [],
                            applies_everywhere=bool(was.get("applies_everywhere")),
                            standing=False)


def _record_correction_scope(conn, correction_id: str, *, markets: list,
                             applies_everywhere: bool, was: Optional[dict] = None) -> None:
    """One row per CHANGE, and a row for what came BEFORE the history existed.

    The seeding is the half that is not bookkeeping. A database written before this table has
    no history at all, so the first row written after the upgrade would be the earliest scope
    on file — and a rule standing in LATAM since long before a LATAM judgment would look, to
    anything reading the history, like a rule that started applying today. Every old judgment
    in its markets would come back onto the report as "not checked against", which is the
    false direction: telling somebody they missed something they did not miss. So the scope
    the row ALREADY had is recorded as having held since it was confirmed, which is exactly
    what the report assumed before the history existed.

    One row per change rather than one per confirmation: re-confirming a rule changes nothing
    about what it applies to, and the history is read as a list of changes.
    """
    if not _columns(conn, "correction_scope"):
        return
    history = correction_scope_history(conn, correction_id)
    # `confirmed_at` ALONE, which is the same test its sibling `withdraw_correction_scope`
    # makes. Asking for `status == "expected"` too looked stricter and was wrong on the one
    # path this product offers by name: `reopen` puts a rule back as PROVISIONAL on purpose,
    # so a declared rule somebody had set aside — the likeliest `ignored` row on an upgraded
    # install, and the one the loader offers to reopen — came through here as provisional, the
    # seed was skipped, and the first history row was dated today. Every judgment ever made in
    # that rule's markets then read as unchecked, including ones whose blocking finding QUOTES
    # the rule. A rule with no `confirmed_at` has never been in force and has no past to seed.
    if not history and was and was.get("confirmed_at"):
        _write_correction_scope(conn, correction_id, when=was["confirmed_at"],
                                markets=was.get("expected_in") or [],
                                applies_everywhere=bool(was.get("applies_everywhere")),
                                seeded=True)
        history = correction_scope_history(conn, correction_id)
    now = sorted(markets), bool(applies_everywhere), True
    if history and (sorted(history[-1]["expected_in"]),
                    history[-1]["applies_everywhere"],
                    history[-1]["standing"]) == now:
        return
    _write_correction_scope(conn, correction_id, when=_now(), markets=markets,
                            applies_everywhere=applies_everywhere)


def graduate_correction(conn, correction_id: str, *, markets: list, confirmed_by: str,
                        applies_everywhere: bool = False) -> None:
    # What the row said BEFORE this write, for the scope history below: on a database that
    # predates that table, what it said is what has to be recorded as having held until now.
    was = get_correction(conn, correction_id)
    # COALESCE, for the reason `graduate_metric` gives: the first confirmation is when this
    # became a rule, and rewriting it makes §8.7 assert that a judgment which CITES the rule
    # was never checked against it.
    conn.execute("UPDATE corrections SET status = 'expected', expected_in = ?, "
                 "confirmed_by = ?, confirmed_at = COALESCE(confirmed_at, ?), "
                 "retired_at = NULL, offered = 1, applies_everywhere = ? WHERE id = ?",
                 (json.dumps(sorted(markets)), confirmed_by, _now(),
                  1 if applies_everywhere else 0, correction_id))
    # And WHEN it started applying to that, which `confirmed_at` cannot say once it is
    # preserved across a re-confirmation. Written here because this is the only function that
    # writes a rule's scope, so the history cannot be missed by a second door.
    _record_correction_scope(conn, correction_id, markets=markets,
                             applies_everywhere=applies_everywhere, was=was)
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
    scope, scope_params = _market_scope(markets, "s")
    sql += scope
    params += scope_params
    row = conn.execute(sql, params).fetchone()
    return int(row["n"] or 0)


# SQLite's default, restored after the cache write drops it. Named rather than spelled twice,
# so the two cannot drift into a read that waits longer than the caller expected.
_BUSY_TIMEOUT_MS = 5000


def facts_key(body: str) -> str:
    """What a stored `facts.compute` result was computed UNDER (§13.2/D95).

    THREE things, because the answer depends on all three, and the first version keyed on
    only one and a half of them.

      the BODY          the text the checks ran over.
      the RULEBOOK's    CONTENT, not its self-declared `version:`. A customer who edits
                        `watch_for` and does not bump the version — nothing enforces it, the
                        README only asks — used to be saved by the next restart, because
                        every read recomputed. Cached against their version string, the stale
                        answer became PERMANENT: a `contradicted` guardrail surviving the
                        deletion of the rule that raised it, which is this product asserting
                        a breach that does not exist. Review found it.
      the ALGORITHM     `facts.py` itself. `rulebook.version()`'s `core-1.0` half is the
                        bundled YAML's version, not a code version — `facts.py` has changed
                        six times while that string stood still — so a release that fixes a
                        matcher would have left every stored row answering with the old one.
                        Before this cache, an upgrade healed everything by recomputing; the
                        cache is what made "it heals on restart" stop being true, so it owes
                        the guard.

    Content-addressed throughout, so nothing depends on somebody remembering to bump a
    number — which is the discipline this codebase has watched fail most often.
    """
    import hashlib
    import json as _json

    import rulebook

    body_digest = hashlib.sha256((body or "").encode("utf-8")).hexdigest()[:24]
    try:
        # `load()` is `lru_cache`d, so this is a dict lookup after the first call.
        rules = _json.dumps(rulebook.load(), sort_keys=True, default=str)
    except Exception:                      # noqa: BLE001
        # An unreadable rulebook is `health_check`'s finding and every write path's refusal.
        # Here it means "do not trust any cached answer", which a key nothing matches gives.
        rules = f"unreadable:{_now()}"
    rules_digest = hashlib.sha256(rules.encode("utf-8")).hexdigest()[:16]
    return f"{body_digest}:{rules_digest}:{_facts_algorithm_stamp()}"


@functools.lru_cache(maxsize=1)
def _facts_algorithm_stamp() -> str:
    """A stamp that moves when `facts.py` does (§13.2, round 2).

    The source itself where it can be read, which needs no discipline from anybody. A frozen
    build may have no `.py` to hash, so it falls back to the product version — which changes
    every release, and a release is the only way the algorithm changes in a frozen build.
    """
    import hashlib

    import version

    try:
        source = (pathlib.Path(__file__).parent / "facts.py").read_bytes()
    except OSError:
        return f"v{version.VERSION}"
    return hashlib.sha256(source).hexdigest()[:12]


def stored_facts(conn, campaign_id: str) -> Optional[dict]:
    """The cached `facts.compute` result, or None if there is none for the CURRENT key."""
    if "computed_facts" not in _columns(conn, "campaigns"):
        return None
    row = conn.execute(
        "SELECT computed_facts, computed_facts_key FROM campaigns WHERE id = ?",
        (campaign_id,)).fetchone()
    if not row or not row["computed_facts"]:
        return None
    return _parsed_facts(row["computed_facts"])


def stored_facts_stamp(conn, campaign_id: str) -> Optional[str]:
    """The RULEBOOK half of the key the stored facts were computed under."""
    if "computed_facts_key" not in _columns(conn, "campaigns"):
        return None
    row = conn.execute("SELECT computed_facts_key FROM campaigns WHERE id = ?",
                       (campaign_id,)).fetchone()
    key = (row["computed_facts_key"] if row else None) or ""
    return key.split(":", 1)[1] if ":" in key else None


def facts_if_current(conn, campaign_id: str, body: str) -> Optional[dict]:
    """The stored facts if they were computed under the CURRENT key, else None (§13.2).

    One statement. `stored_facts_are_current` followed by `stored_facts` asked the same row
    twice and `_columns` twice on top — 22 statements per record per report, spent to avoid a
    text scan. Both are kept because tests and `health_check` ask the two questions
    separately; the read path asks them together because that is what it needs.
    """
    if "computed_facts" not in _columns(conn, "campaigns"):
        return None
    row = conn.execute(
        "SELECT computed_facts, computed_facts_key FROM campaigns WHERE id = ?",
        (campaign_id,)).fetchone()
    if not row or row["computed_facts_key"] != facts_key(body):
        return None
    return _parsed_facts(row["computed_facts"])


def _parsed_facts(raw) -> Optional[dict]:
    """A stored fact set, or None if it is not one. SHAPE, not just valid JSON."""
    try:
        stored = json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return None
    # A row holding `[]` or `"text"` parses and then fails in the caller with an
    # `AttributeError`, and a row holding `{}` is returned as a record with no facts at all.
    # The docstring promised a corrupt row costs a scan and never an answer, and checking
    # only the syntax made that false.
    if not isinstance(stored, dict) or "language" not in stored:
        return None
    return stored


def stored_facts_are_current(conn, campaign_id: str, body: str) -> bool:
    if "computed_facts_key" not in _columns(conn, "campaigns"):
        return False
    row = conn.execute("SELECT computed_facts_key FROM campaigns WHERE id = ?",
                       (campaign_id,)).fetchone()
    return bool(row) and row["computed_facts_key"] == facts_key(body)


def keep_facts(conn, campaign_id: str, body: str, computed: dict) -> None:
    """Store a computed result against the key it was computed under (§13.2/D95).

    **Never raises, and never commits somebody else's transaction.** This is reached from a
    READ — `facts.for_campaign` fills the cache on a miss — and a read that fails on a
    read-only database, or that commits a half-finished write the caller meant to roll back,
    is a far worse thing than a slow read. `store.py`'s metrics import documents this exact
    shape ("a rollback after it had nothing left to undo"); review found this one.
    """
    if "computed_facts" not in _columns(conn, "campaigns"):
        return
    # Already inside somebody's transaction: write, and let THEM decide when it commits. The
    # row is correct either way, and if their work is rolled back the cache entry goes with
    # it — which is right, because the body it was computed from goes too.
    theirs = conn.in_transaction
    try:
        # A SHORT wait for the lock, not the default five seconds. A rulebook edit makes every
        # record cold at once, so the first report afterwards tries to fill the whole cache —
        # and with another connection mid-write, each one sat out the full busy timeout inside
        # a read somebody was waiting on. Review measured 5.49s for a single record. Filling a
        # cache is never worth blocking on: if the database is busy, skip it and recompute
        # next time.
        conn.execute("PRAGMA busy_timeout = 50")
        conn.execute("UPDATE campaigns SET computed_facts = ?, computed_facts_key = ? "
                     "WHERE id = ?",
                     (json.dumps(computed), facts_key(body), campaign_id))
        if not theirs:
            conn.commit()
    except Exception:                      # noqa: BLE001 - a cache, never an answer
        return
    finally:
        try:
            conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        except Exception:                  # noqa: BLE001
            pass


def breadth_of(conn, campaign_ids: list) -> dict:
    """How widely a thing has been seen: across partners AND markets (§12.4/D103).

    §8.3's gate is "across at least two PARTNERS or markets" and it could only count markets,
    because nothing in the schema recorded who the partner was. Two campaigns with one partner
    in two markets is weaker evidence of a general rule than two partners in one market — the
    gate exists to stop one partner's house style becoming everybody's standing requirement,
    and it could not see the thing it is named for.

    `learning.gate` is the caller, through `metrics.graduation` and `corrections.graduation`.
    It had none for a round, which is this project's most-repeated defect and the one the
    §12.4 test file opens by naming: a field added because a review asked for it and read by
    nothing leaves the row closed and the defect exactly where it was.
    """
    partners, markets, types = set(), set(), set()
    typed = 0
    for campaign_id in campaign_ids or []:
        record = get_campaign(conn, campaign_id)
        if not record:
            continue
        if (record.get("partner") or "").strip():
            partners.add(record["partner"].strip().casefold())
        markets |= {m for m in (fold_market(m) for m in markets_of(record)) if m}
        kind = fold_campaign_type(record.get("campaign_type"))
        if kind:
            types.add(kind)
            typed += 1
    return {"partners": len(partners), "markets": len(markets),
            "widest": max(len(partners), len(markets)),
            # §12.4/D102: the campaign TYPES, and whether the evidence is coherent about them.
            # `one_type` is the only shape that justifies keying a checklist on type: every
            # campaign carrying this thing was the same kind of campaign, and none was silent
            # about what kind it was. Mixed evidence, or evidence where half the records have
            # no type, says nothing about type at all — and narrowing a checklist on that
            # would remove expectations the evidence never earned the right to remove.
            "types": sorted(types),
            "one_type": sorted(types) if (len(types) == 1 and typed == len(campaign_ids or []))
                        else []}


def markets_of(campaign: dict) -> list:
    """Every market a campaign counts towards, or `[None]` when it has none.

    The one implementation. `markets` is the only way to express a multi-country activation,
    so ignoring it reports a real LATAM campaign as covering nothing; a record with no market
    at all is most of a young library, so dropping those describes a library nobody has. Case
    is folded WITHIN a record here — folding it across records is the caller's job, and §5.3
    is where that was got wrong before.

    **D71 asked whether `region` should feed this at all, or is a different axis.** It is a
    different axis, and now the rulebook can say so: a region is a GROUPING of markets, so a
    campaign whose market is Peru and whose region is LATAM is in ONE market, not two. Counted
    as two it satisfied §8.3's "seen in at least two markets" gate on its own — a gate whose
    entire purpose is that breadth has to be earned — and inflated every coverage cell.

    Only where the rulebook DECLARES the region. Undeclared, this product cannot tell a
    region from a country: `region` is a free-text column that means a continent on one record
    and a country on the next, which is why the row was a question rather than a bug. Guessing
    would drop a real market from a library that never declared anything.
    """
    declared_regions = set()
    try:
        import rulebook

        declared_regions = {(entry.get("region") or "").strip().lower()
                            for entry in rulebook.vocabulary("markets").values()
                            if (entry.get("region") or "").strip()}
    except ValueError:
        declared_regions = set()

    named = []
    for raw in (campaign.get("market"), campaign.get("region"),
                *(campaign.get("markets") or [])):
        if raw and str(raw).strip():
            value = str(raw).strip()
            if value.strip().lower() in declared_regions:
                continue
            named.append(value)
    # §13.5: the dedupe is `learning.fold_markets` — "one name per market, first spelling
    # kept" — which is the FOURTH place that sentence was written out. `[None]` survives it,
    # because "this record has no market" is a sentinel every caller reads and not an empty
    # list: `fold_markets` would drop it, so the fold happens first and the sentinel after.
    import learning

    return learning.fold_markets(named) or [None]


def fold_market(name: Optional[str]) -> Optional[str]:
    """The comparison key for a market name.

    "LATAM" and "latam" are one market. C16 established the fold after exactly this bug, and
    §8.3 reintroduced it in the one place it does the most damage: three spellings of one
    market satisfied a gate whose entire purpose is "seen in at least two markets".

    D72: and "Latin America" is that market too, if the customer's rulebook says so. Case
    folding could never reach that — it is the same stopgap C16 installed, and the row asks
    for a SYNONYM instead. Because it resolves here, in the one comparison key every reader
    already goes through, a declaration reaches rows written years before it and nothing has
    to be rewritten.
    """
    if not name or not str(name).strip():
        return None
    return fold_vocabulary("markets", str(name))


def graduate_metric(conn, canonical: str, *, markets: list, confirmed_by: str,
                    campaign_types: Optional[list] = None) -> None:
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
    fields = ("status = 'expected', expected_in = ?, answered = 1, surfaced = 1, "
              "confirmed_by = ?, confirmed_at = COALESCE(confirmed_at, ?), retired_at = NULL")
    params: list = [json.dumps(sorted(markets)), confirmed_by, _now()]
    # §12.4/D102, and guarded on the column because an upgraded database gets it from
    # `_add_missing_columns` and this runs on every promotion.
    if "expected_for_types" in _columns(conn, "metric_registry"):
        fields += ", expected_for_types = ?"
        params.append(json.dumps(sorted(set(campaign_types or []))))
    conn.execute(f"UPDATE metric_registry SET {fields} WHERE canonical = ?",
                 params + [canonical])
    conn.commit()


def _record_measure_history(conn, canonical: str, what: str, why: str) -> None:
    """One occasion, appended (§12.4/D106)."""
    if not _columns(conn, "measure_history"):
        return
    conn.execute("INSERT INTO measure_history (id, canonical, what, why, at) "
                 "VALUES (?,?,?,?,?)",
                 (_id("mhist"), canonical, what, " ".join((why or "").split()), _now()))


def measure_history(conn, canonical: str) -> list:
    """Every time this measure was retired or revived, oldest first (§12.4/D106)."""
    if not _columns(conn, "measure_history"):
        return []
    return [{"what": row["what"], "why": row["why"], "at": row["at"], "basis": "computed"}
            for row in conn.execute(
                "SELECT what, why, at FROM measure_history WHERE canonical = ? "
                "ORDER BY at, rowid", (canonical,)).fetchall()]


def retire_metric(conn, canonical: str, *, why: str = "") -> None:
    """Demote, never delete (§8.5).

    `expected_in` and every recorded value are left exactly where they are. The measure stops
    being asked for; the record that it was once asked for survives, because "we used to track
    this" is an answer somebody will need and a deleted row can only say "we never did".
    """
    # Guarded on the rowcount, exactly as `revive_metric` is. Unguarded, retiring a name that
    # is not in the registry — or one already retired — appended a `retired` occasion anyway,
    # so the history could assert an event that never happened against a registry with no such
    # measure in it. A history that records things the library did not do is worse than no
    # history, because the whole of D106 is that somebody will read it years later.
    changed = conn.execute(
        "UPDATE metric_registry SET status = 'retired', retired_at = ? "
        "WHERE canonical = ? AND status != 'retired'", (_now(), canonical)).rowcount
    if changed:
        # D106: the occasion, kept. `retired_at` is the LATEST one and overwrites the last.
        _record_measure_history(conn, canonical, "retired",
                                why or "no longer reported by recent campaigns")
    conn.commit()


def revive_metric(conn, canonical: str, *, why: str = "") -> None:
    """A retired measure that somebody recorded again is expected again (§8.5).

    It graduated once and a person confirmed it. Asking them to confirm it a second time
    because a quarter went by is asking the same question twice, which §8.2 established is how
    a product teaches people to dismiss it.
    """
    changed = conn.execute(
        "UPDATE metric_registry SET status = 'expected', retired_at = NULL "
        "WHERE canonical = ? AND status = 'retired'", (canonical,)).rowcount
    if changed:
        _record_measure_history(conn, canonical, "revived",
                                why or "measured again after being retired")
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
    scope, scope_params = _market_scope(markets, "v")
    sql += scope
    params += scope_params
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
