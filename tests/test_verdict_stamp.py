"""
§7.6: stamp every verdict with what produced it.

The review's words: *"Right now two evaluations of the same brief are indistinguishable in the
record, so drift cannot even be detected, let alone diagnosed. Fix: store `rulebook_version`,
`server_version`, `embedding_model`, `model_id` and the retrieved `campaign_ids` with their
similarity scores on every saved evaluation. When two users disagree, the diff of those five
fields usually explains it in seconds — and when they agree, you have evidence the agreement
is real rather than luck."*

The last clause is the one that decides the design. A stamp is not an audit trail for its own
sake: it is what lets **agreement** be read as evidence rather than coincidence, which is the
premise §7.7's golden set rests on. Everything measured there is meaningless without it,
because two runs that agree while differing in embedding model and rulebook version agree
about nothing in particular.

Four of the five are facts the server holds. The fifth, `model_id`, is the one thing here the
server cannot know — the judging model is on the other side of the protocol — and that is
recorded as a gap rather than guessed at, because a stamp with a wrong field in it is worse
than a stamp with a missing one.

**The whole block is the server's** (D4). A provenance record the model writes is a record of
what the model says produced it.
"""
import pytest

import core
import store


def _judged(conn, **over):
    args = dict(subject_title="Colombia v1", verdict="revise", summary="A stretch.",
                approve_if="It is fixed.",
                findings=[{"severity": "should_fix", "kind": "missing_information",
                           "finding": "No end date", "fix": "Add one"}])
    args.update(over)
    return core.save_evaluation(conn, **args)


# ── the five fields ─────────────────────────────────────────────────────────

def test_every_verdict_records_what_produced_it(conn):
    """"Two evaluations of the same brief are indistinguishable in the record, so drift
    cannot even be detected, let alone diagnosed."""
    saved = _judged(conn)
    stamp = store.get_evaluation(conn, saved["evaluation_id"])["provenance"]

    assert stamp["server_version"] == core.version.VERSION
    assert stamp["rulebook_version"] == core.RULEBOOK_VERSION
    assert stamp["embedding_model"] == core.embedding_model_id()
    assert stamp["basis"] == "computed"


def test_the_retrieved_ids_and_their_scores_are_stamped(conn):
    """"The retrieved `campaign_ids` with their similarity scores." Not the cited ones — what
    the server SHOWED, which is what two users must share before agreeing means anything."""
    core.ingest_campaign(conn, title="Peru", detail="A creator-led launch.")
    package = core.prepare_evaluation(conn, subject_title="Colombia v1",
                                      proposal_text="A creator-led launch.")

    saved = _judged(conn, retrieval=package["retrieval"]["receipt"])
    stamp = store.get_evaluation(conn, saved["evaluation_id"])["provenance"]

    assert [r["campaign_id"] for r in stamp["retrieved"]] == \
        [e["campaign_id"] for e in package["evidence"]]
    assert all(isinstance(r["similarity"], float) for r in stamp["retrieved"])


def test_a_judgment_with_no_receipt_stamps_no_retrieval_rather_than_an_empty_one(conn):
    """An empty list reads as "the search returned nothing". "Nobody recorded a search" is a
    different statement, and §6.4 needed three separate codes to keep those apart."""
    stamp = store.get_evaluation(conn, _judged(conn)["evaluation_id"])["provenance"]
    assert stamp["retrieved"] is None


# ── the field the server cannot know ────────────────────────────────────────

def test_the_judging_model_is_recorded_as_unknown_rather_than_guessed(conn):
    """`model_id` is the one field here the server cannot observe: the model doing the judging
    is on the other side of the protocol. A stamp with a wrong field in it is worse than one
    with a missing field, because the diff that is supposed to explain a disagreement would
    then explain it wrongly."""
    stamp = store.get_evaluation(conn, _judged(conn)["evaluation_id"])["provenance"]

    assert stamp["model_id"] is None
    assert "cannot" in stamp["model_id_note"].lower() or "not" in stamp["model_id_note"].lower()


def test_a_caller_that_knows_its_model_may_say_so(conn):
    """The client does know. Accepting it is how the field becomes real, and marking it
    `stated` is how it stays honest next to four computed ones."""
    saved = _judged(conn, model_id="claude-opus-5")
    stamp = store.get_evaluation(conn, saved["evaluation_id"])["provenance"]

    assert stamp["model_id"] == "claude-opus-5"
    assert stamp["model_id_basis"] == "stated"


# ── D18: what went wrong while the evidence was gathered ────────────────────

def test_warnings_raised_while_gathering_the_evidence_are_stamped(conn):
    """D18. A judgment made over a half-indexed library is a different judgment from one made
    over a whole one, and the warning that said so lived for exactly one response."""
    core.ingest_campaign(conn, title="Peru", detail="A creator-led launch.")
    store.set_embedding_model(conn, "some-other-model")
    core.ingest_campaign(conn, title="Chile", detail="A creator-led launch.")
    package = core.prepare_evaluation(conn, subject_title="Colombia v1",
                                      proposal_text="A creator-led launch.")

    saved = _judged(conn, retrieval=package["retrieval"]["receipt"])
    stamp = store.get_evaluation(conn, saved["evaluation_id"])["provenance"]

    assert "mixed_embedding_models" in stamp["warnings_at_retrieval"]


# ── the model does not write its own provenance ─────────────────────────────

def test_the_model_cannot_stamp_the_record(conn):
    """D4. A provenance record the model writes is a record of what the model says produced
    it — the same rule as `basis`, `checked` and `evidence`."""
    with pytest.raises(ValueError) as e:
        _judged(conn, provenance={"server_version": "9.9.9"})
    assert "server" in str(e.value).lower()


# ── it has to be readable, or drift still cannot be diagnosed ───────────────

def test_two_judgments_can_be_compared_field_by_field(conn):
    """"When two users disagree, the diff of those five fields usually explains it in
    seconds." That only works if something will actually produce the diff."""
    import config

    first = _judged(conn)
    # The live embedder changes, not just the provenance rows: the stamp records what
    # produced THIS judgment, so a rewrite of history is not what a reader is diffing.
    original = config.EMBED_PROVIDER
    config.EMBED_PROVIDER = "voyage"
    try:
        second = _judged(conn, subject_title="Colombia v1")
    finally:
        config.EMBED_PROVIDER = original

    diff = core.compare_provenance(conn, first["evaluation_id"], second["evaluation_id"])

    assert diff["differs"] == ["embedding_model"]
    assert diff["same"], "and it says what they had in common, or agreement means nothing"


def test_two_judgments_produced_identically_say_so(conn):
    """"When they agree, you have evidence the agreement is real rather than luck" — which is
    the premise §7.7's golden set rests on, and it is a claim about the STAMP, not about the
    verdicts."""
    diff = core.compare_provenance(conn, _judged(conn)["evaluation_id"],
                                   _judged(conn)["evaluation_id"])
    assert diff["differs"] == []
    assert "same conditions" in diff["what_it_means"].lower()


def test_the_stamp_is_on_the_read_path(conn):
    saved = _judged(conn)
    assert core.get_evaluation(conn, evaluation_id=saved["evaluation_id"])["provenance"]


def test_it_reaches_the_model_over_the_protocol(conn):
    import asyncio
    import json
    import mcp_server

    async def call(name, args):
        return json.loads((await mcp_server.mcp.call_tool(name, args)).content[0].text)

    saved = asyncio.run(call("save_evaluation", {
        "subject_title": "Colombia v1", "verdict": "revise", "summary": "A stretch.",
        "approve_if": "Fixed.", "model_id": "claude-opus-5",
        "findings": [{"severity": "should_fix", "kind": "missing_information",
                      "finding": "No end date", "fix": "Add one"}]}))

    read_back = asyncio.run(call("get_evaluation",
                                 {"evaluation_id": saved["evaluation_id"]}))
    assert read_back["provenance"]["model_id"] == "claude-opus-5"
    assert read_back["provenance"]["server_version"]


# ── D81: what the stamp still cannot explain ────────────────────────────────

def test_a_refused_write_leaves_a_trace(conn):
    """D81. A finding dropped because its quote failed leaves nothing at all, so the kind-drift
    D78 predicts has no instrument and the worst outcome — a real problem omitted because the
    citation would not verify — is invisible from the marketer's side."""
    core.ingest_campaign(conn, title="Peru", detail="Seeded one colourway per creator.")

    before = store.refusal_counts(conn)
    with pytest.raises(ValueError):
        _judged(conn, findings=[{"severity": "should_fix", "kind": "precedent_departure",
                                 "departure": "regression", "finding": "Departs",
                                 "fix": "Change it",
                                 "precedent": {"campaign_id": "camp_nope",
                                               "quote": "a sentence nobody wrote"}}])

    after = store.refusal_counts(conn)
    assert after["precedent_unresolved"] == before.get("precedent_unresolved", 0) + 1


def test_refusals_are_counted_by_kind_not_just_totalled(conn):
    """"Thirty writes were refused" says nothing. Which rule refused them is the signal —
    a rise in citation refusals and a rise in severity downgrades mean different things."""
    with pytest.raises(ValueError):
        _judged(conn, verdict="approve", approve_if="Something", findings=[])

    assert "approve_with_exit_condition" in store.refusal_counts(conn)


# ── D85: which kind of thing a reconciliation was checked against ───────────

def test_a_reconciliation_says_what_it_was_checked_against(conn):
    """D85. §6.3 made the version-based reconciliation the common case, so "v2 shows the
    structure came back" now lands in the same `actual` column as a CTR figure — and anything
    computing calibration later reads both as outcome data."""
    saved = _judged(conn)
    rid = store.insert_reconciliation(conn, evaluation_id=saved["evaluation_id"],
                                      comparison="The prediction held.",
                                      actual="v2 shows the structure came back.",
                                      basis="superseding_version")

    row = store.get_reconciliation(conn, rid)
    assert row["basis"] == "superseding_version"


def test_a_reconciliation_against_measured_results_says_that_instead(conn):
    saved = _judged(conn)
    rid = store.insert_reconciliation(conn, evaluation_id=saved["evaluation_id"],
                                      comparison="CTR came in at 3.4 percent.",
                                      actual="CTR 3.4 percent.", basis="results")
    assert store.get_reconciliation(conn, rid)["basis"] == "results"


# ── D29: the model the stamp names has to be the model that ran ─────────────

def test_the_embedding_model_is_pinned_by_digest_when_one_is_available(conn, monkeypatch):
    """D29. `ollama/nomic-embed-text` names a tag, and a tag moves. CLIP is hash-pinned, so
    any 768-dimension model satisfied the text check — which means the stamp could say two
    judgments used the same embedder while they used different weights, and §7.6's whole
    claim is that the stamp explains a disagreement."""
    import config
    import embedding

    monkeypatch.setattr(config, "EMBED_PROVIDER", "ollama")
    monkeypatch.setattr(config, "OLLAMA_EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setattr(embedding, "model_digest", lambda: "sha256:abc123")

    assert core.embedding_model_id() == "ollama/nomic-embed-text@sha256:abc123"


def test_an_unavailable_digest_is_absent_rather_than_invented(conn, monkeypatch):
    """The digest needs the embedder to answer. When it cannot, the name alone is what is
    known — and a stamp that implies a pin it does not have is the wrong-field failure this
    item is careful about elsewhere."""
    import config
    import embedding

    monkeypatch.setattr(config, "EMBED_PROVIDER", "ollama")
    monkeypatch.setattr(config, "OLLAMA_EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setattr(embedding, "model_digest", lambda: None)

    assert core.embedding_model_id() == "ollama/nomic-embed-text"


# ── D89: the migration covers what the readers assume ──────────────────────

def test_every_column_the_schema_declares_is_added_to_an_existing_database(tmp_path):
    """D89. `_migrate_schema` added three campaign columns by hand while `get_campaign` reads
    the whole schema — `tags` and `supersedes` unconditionally, and nothing added them.
    `CREATE TABLE IF NOT EXISTS` cannot add a column to a table that already exists, so a
    database predating either one broke the most-used reader in the codebase."""
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE campaigns (id TEXT PRIMARY KEY, title TEXT NOT NULL);
        CREATE TABLE evaluations (id TEXT PRIMARY KEY, subject_title TEXT NOT NULL,
                                  created_at REAL NOT NULL);
    """)
    old.execute("INSERT INTO campaigns VALUES ('camp_old', 'Peru launch')")
    old.commit()
    old.close()

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    store.upgrade(conn)

    declared = store._declared_columns("campaigns")
    actual = set(store._columns(conn, "campaigns"))
    assert declared <= actual, f"never added: {sorted(declared - actual)}"
    # And the record survives, which is the whole reason this is additive.
    assert store.get_campaign(conn, "camp_old")["title"] == "Peru launch"


def test_the_migration_covers_every_table_not_just_campaigns(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE campaigns (id TEXT PRIMARY KEY, title TEXT NOT NULL);
        CREATE TABLE metrics (id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL,
                              created_at REAL NOT NULL);
    """)
    old.commit()
    old.close()

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    store.upgrade(conn)

    # Every table the schema declares, not a hand-picked four — a list of tables to check is
    # the same second copy of the schema as a list of columns to add, and it drifts the same
    # way (see the §8.1-era test below for the drift it actually had).
    for table in store._declared_tables():
        declared = store._declared_columns(table)
        actual = set(store._columns(conn, table))
        assert declared <= actual, f"{table} never gained: {sorted(declared - actual)}"


def test_a_table_added_by_a_later_release_still_gains_its_later_columns(tmp_path):
    """The D89 defect, reopened and caught by the mutation pass.

    `_add_missing_columns` iterated a hand-kept TUPLE of tables. §8.1 added `metric_registry`
    and `metric_values` and did not add them to it, so every column §8.3 declares on the
    registry reached a fresh install and no upgraded one — and nothing failed, because the
    tables exist on a fresh database and the tests all ran against fresh databases.

    An old database here therefore has to carry the tables in their EARLIER shape. A test that
    omits them lets `_SCHEMA` create them complete and asserts nothing at all.
    """
    import sqlite3

    path = tmp_path / "eighty_one.db"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE campaigns (id TEXT PRIMARY KEY, title TEXT NOT NULL);
        CREATE TABLE metric_registry (canonical TEXT PRIMARY KEY, display_name TEXT NOT NULL,
                                      unit TEXT, direction TEXT,
                                      aliases TEXT NOT NULL DEFAULT '[]',
                                      status TEXT NOT NULL DEFAULT 'provisional');
        CREATE TABLE metric_values (id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL,
                                    metric TEXT NOT NULL, value REAL NOT NULL,
                                    created_at REAL NOT NULL);
    """)
    old.execute("INSERT INTO metric_registry (canonical, display_name) VALUES ('roas', 'ROAS')")
    old.commit()
    old.close()

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    store.upgrade(conn)

    for table in ("metric_registry", "metric_values"):
        declared = store._declared_columns(table)
        actual = set(store._columns(conn, table))
        assert declared <= actual, f"{table} never gained: {sorted(declared - actual)}"
    # And the reader works, which is the point of adding them at all.
    assert store.metric_registry(conn)["roas"]["expected_in"] == []


def test_seed_measures_written_before_expected_meant_a_checklist_are_demoted(tmp_path):
    """§8.1 registered the twelve shipped measures as `status='expected'`, meaning "a measure
    this product recognises". §8.3 gave the word its real meaning — on the checklist every
    brief in a market is held to.

    On a database seeded by the earlier release those rows still say `expected`, so the first
    §8.3 read of them put all twelve on every checklist, confirmed by nobody. Adding a column
    migrates the shape; this migrates the MEANING, which a column migration cannot see.
    """
    import sqlite3

    import metrics

    path = tmp_path / "eight_one.db"
    old = sqlite3.connect(path)
    old.executescript(store._SCHEMA)
    old.executescript(store._INDEXES)
    old.execute("INSERT INTO metric_registry (canonical, display_name, status) "
                "VALUES ('cpm', 'Cost per mille', 'expected')")
    # One a person actually graduated, which must survive untouched.
    old.execute("INSERT INTO metric_registry (canonical, display_name, status, expected_in, "
                "confirmed_by) VALUES ('footfall_uplift', 'Footfall uplift', 'expected', "
                "'[\"LATAM\"]', 'R. Vega')")
    old.commit()
    old.close()

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    store.upgrade(conn)

    assert store.metric_registry(conn)["cpm"]["status"] == "known"
    assert metrics.expected_for(conn, market="LATAM") == ["footfall_uplift"]
    assert store.metric_registry(conn)["footfall_uplift"]["confirmed_by"] == "R. Vega"
