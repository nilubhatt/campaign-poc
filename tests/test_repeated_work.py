"""
§13.3 — the embedding and scan costs, measured before any of them was touched.

Five rows, each filed as "correct and cheap at the sizes this product has, named so it is a
measured decision later rather than a discovered one". This is that measurement. Counted as
CALLS rather than wall time, because that is the part that does not depend on the provider:
against the bundled hash embedder a repeated embed is microseconds, and against Ollama it is
a network round-trip for a vector already in hand.

    D79   quote verification reads every chunk of the cited record, per finding
          → CONFIRMED: 12 findings, 12 full reads of the same record.
    D90   `save_evaluation` embeds one string twice, `prepare_evaluation` three times
          → HALF STALE, and worse than recorded in the other half. `save_evaluation` already
            embeds once. `prepare_evaluation` embeds the SAME string FOUR times.
    D91   `coverage` calls `store.citations` once per cell, each a full scan
          → CONFIRMED: one call per cell.
    D97   a content edit re-embeds the whole deck when only the summary changed
          → CONFIRMED, and it is the whole deck: a TITLE-only edit re-embedded all 17 chunks
            of an 8-section deck, exactly as many as the original ingest.
    D107  `retire_stale` scans the registry on every metric write
          → CONFIRMED: 5 registry reads per `metrics.record`.

Two of the five rows were therefore wrong about their own subject, in both directions. That is
the argument for the rows existing — and for measuring one before fixing it.

What this file pins is the SHAPE: work that does not grow with the number of findings, cells
or chunks that did not change. Not wall time, which would make these tests a flaky benchmark.
"""
import hashlib

import core
import embedding
import metrics
import pytest
import store


@pytest.fixture()
def counted(monkeypatch):
    """Every embed, with its text — so a test can say "the same string, twice"."""
    seen = []
    real = embedding.embed
    monkeypatch.setattr(embedding, "embed",
                        lambda text, **kw: (seen.append(text), real(text, **kw))[1])
    return seen


def _repeats(seen):
    keys = [hashlib.sha1(t.encode()).hexdigest() for t in seen]
    return len(keys) - len(set(keys))


_PARA = "The Bogota launch will run paid social and out-of-home across the quarter. " * 30
_DECK = "\n\n".join(f"Section {i}. {_PARA}" for i in range(8))


def _deck_record(conn, title="Bogota launch"):
    return core.ingest_campaign(conn, title=title, market="Peru", status="concluded",
                                deck_text=_DECK, confirm=True)["campaign_id"]


# ---------------------------------------------------------------------------------------
# D90 — the same string, embedded again


def test_one_string_is_embedded_once_per_call(conn, counted):
    """D90. `prepare_evaluation` embedded the proposal text FOUR times — the row says three,
    and it had grown since. Against the bundled hash provider that is microseconds; against
    Ollama it is four network round-trips for a vector that was in hand after the first."""
    _deck_record(conn)
    counted.clear()

    core.prepare_evaluation(conn, subject_title="Lima launch",
                            proposal_text="A store opening in Lima.")

    assert _repeats(counted) == 0, (
        f"{_repeats(counted)} of {len(counted)} embeddings were of a string already embedded"
    )


def test_saving_a_judgment_does_not_re_embed_what_it_just_embedded(conn, counted):
    """The half of D90 that was already true when the row was re-read — pinned so it stays
    true, since nothing was stopping it drifting back.

    The first version saved a judgment with NO findings, which performs exactly one embed, so
    "no repeats" held by construction and the test could not fail. The disconfirming search
    runs per verdict pole, so a `revise` with findings is what actually reaches the embedder
    more than once."""
    cid = _deck_record(conn)
    counted.clear()

    core.save_evaluation(
        conn, subject_title="Lima launch", verdict="revise", approve_if="It is fixed.",
        summary="A store opening in Lima, missing its dates.", cited_ids=[cid],
        findings=[{"severity": "should_fix", "kind": "missing_information",
                   "finding": "No dates", "fix": "Add them"}])

    assert len(counted) >= 1, "the fixture has to actually reach the embedder"
    assert _repeats(counted) == 0


def test_the_memo_never_serves_a_vector_from_another_model(conn, monkeypatch):
    """§7.2 spent an item establishing that a model change is "a visible migration rather
    than a silent re-ranking". A memo keyed on the text alone would be that re-ranking with a
    cache in front of it: the same string, embedded under one model, handed to a caller
    working under another."""
    import config
    import embedding

    made = []
    monkeypatch.setattr(embedding, "embed",
                        lambda text, **kw: (made.append(config.EMBED_PROVIDER),
                                            [0.5] * config.EMBED_DIM)[1])

    with embedding.each_string_embedded_once():
        embedding.embed_once("the same sentence")
        monkeypatch.setattr(config, "EMBED_PROVIDER", "voyage")
        embedding.embed_once("the same sentence")

    assert made == ["hash", "voyage"], (
        "the second call was served a vector made by a different model"
    )


def test_the_memo_does_not_outlive_its_call(conn, monkeypatch):
    """The first version was a process-lifetime LRU, and it hid an embedder that had gone
    down between two calls — eight tests of outage behaviour failed, each describing a real
    thing the product is supposed to report. Outside a scope this is plain `embed`."""
    import config
    import embedding

    calls = []
    monkeypatch.setattr(embedding, "embed",
                        lambda text, **kw: (calls.append(text), [0.5] * config.EMBED_DIM)[1])

    with embedding.each_string_embedded_once():
        embedding.embed_once("a sentence")
        embedding.embed_once("a sentence")
    assert len(calls) == 1

    embedding.embed_once("a sentence")
    assert len(calls) == 2, "a vector from a previous call was served in this one"


# ---------------------------------------------------------------------------------------
# D97 — re-embedding a deck nobody edited


def test_editing_a_title_does_not_re_embed_the_deck(conn, counted):
    """D97. The title and detail live in the SUMMARY chunk; the deck's own sections are
    chunked from `deck_text`, which a title edit does not touch. Re-embedding all 17 chunks
    of an 8-section deck to change a title is seventeen network calls for one changed string
    — and it churns every chunk id, which §12.4 found breaks a `commitment_id` the model has
    just been shown."""
    cid = _deck_record(conn)
    before = store.get_campaign(conn, cid)["chunks_total"]
    counted.clear()

    core.update_campaign(conn, campaign_id=cid, title="Bogota launch v2")

    assert len(counted) < before, (
        f"a title edit re-embedded {len(counted)} chunks of {before}"
    )
    assert len(counted) <= 1, "only the summary chunk changed"


def test_editing_the_detail_does_re_embed_what_changed(conn, counted):
    """The other half, and the reason this cannot simply skip the rebuild: `detail` IS part
    of the summary chunk, so it has to be re-embedded — and it is also part of the BODY that
    `facts.compute` reads, which a title is not."""
    cid = _deck_record(conn)
    counted.clear()

    core.update_campaign(conn, campaign_id=cid, detail="A one-line summary, newly written.")

    assert counted, "an edit to the detail did not re-index anything"
    assert any("newly written" in t for t in counted)


def test_an_edit_that_leaves_the_deck_alone_keeps_its_chunk_ids(conn):
    """Reusing an id rather than churning one per edit.

    An earlier version of this docstring justified it by saying chunk ids are cited through
    `precedent.chunk_id` — there is no such field, and review said so. The honest reason is
    narrower: nothing anywhere stores a chunk id as a citation (§6.1 verifies quotes against
    the row's own columns, `insert_evaluation` keeps `cited_ids`), so reuse is SAFE rather
    than required, and what it buys is that the vector store and `vector_provenance` are
    re-keyed in place instead of accumulating a fresh id on every edit."""
    cid = _deck_record(conn)
    before = {r["id"] for r in conn.execute(
        "SELECT id FROM campaign_chunks WHERE campaign_id = ? AND kind = 'body'", (cid,))}

    core.update_campaign(conn, campaign_id=cid, title="Bogota launch v2")

    after = {r["id"] for r in conn.execute(
        "SELECT id FROM campaign_chunks WHERE campaign_id = ? AND kind = 'body'", (cid,))}
    assert len(after & before) >= len(before) - 1, (
        "every chunk id changed because a title did"
    )


def test_the_chunk_that_changed_keeps_its_id_too(conn):
    """Not only the untouched ones. A chunk whose words were edited keeps its id and its
    position: `finish_indexing` offers name chunk ids, and an edit in between would otherwise
    hand back an offer naming something that no longer exists. §6.1 verifies quotes against
    the row's own columns precisely so chunk ids need not be quote anchors, which is what
    makes reusing one safe."""
    cid = _deck_record(conn)
    first = conn.execute(
        "SELECT id, text FROM campaign_chunks WHERE campaign_id = ? AND kind = 'body' "
        "ORDER BY chunk_index", (cid,)).fetchone()

    core.update_campaign(conn, campaign_id=cid, detail="A wholly new summary line.")

    now = conn.execute(
        "SELECT id, text FROM campaign_chunks WHERE campaign_id = ? AND kind = 'body' "
        "ORDER BY chunk_index", (cid,)).fetchone()
    assert now["id"] == first["id"], "the edited chunk was deleted and recreated"
    assert now["text"] != first["text"], "and it does hold the new words"


def test_a_failed_re_embed_never_leaves_the_old_words_behind(conn, monkeypatch):
    """The regression the D97 fix introduced, found in review.

    Writing each chunk's text and embedding it in the same loop means a break exits before
    the remaining positions have been rewritten — so they keep the OLD deck's words while the
    row holds the new ones, with their `embedded` flag still set from before. `finish_indexing`
    then reports the record complete and search goes on matching wording the brief no longer
    contains: D80's defect, reintroduced by the fix for D97, and hidden behind a comment
    claiming the row was correct and only the index behind.

    Text first, vectors second. A partial failure then lands where §2.1 says it should."""
    cid = _deck_record(conn)
    # A replacement sharing NO words with the original, so "did this chunk get rewritten"
    # is decidable. `chunking.pack` splits a long paragraph across chunks and the
    # continuation carries no section header, so looking for a header would call a
    # legitimately-rewritten continuation stale.
    fresh = "\n\n".join(f"Chapter {i}. " + ("entirely different vocabulary here. " * 30)
                         for i in range(8))
    conn.execute("UPDATE campaigns SET deck_text = ? WHERE id = ?", (fresh, cid))
    conn.commit()

    calls = {"n": 0}
    real = embedding.embed

    def dies_after_one(text, *a, **kw):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("the embedder went away mid-rebuild")
        return real(text, *a, **kw)

    monkeypatch.setattr(embedding, "embed", dies_after_one)
    core._rebuild_body_index(conn, cid)

    stored = [r["text"] for r in store.body_chunks(conn, cid)]
    assert not any("Bogota launch will run paid social" in t for t in stored), (
        "chunks still hold the old deck's words while the record holds the new ones"
    )
    # And what could not be embedded says so, rather than reporting itself searchable.
    rows = store.body_chunks(conn, cid)
    assert any(not r["embedded"] for r in rows), (
        "every chunk claims to be searchable after the embedder died mid-rebuild"
    )


def test_a_chunk_that_never_got_a_vector_is_not_counted_as_embedded(conn, monkeypatch):
    """A survivor whose words did not change but which never successfully embedded is still
    owed a vector. Counted as embedded because it survived, `embedded == chunks` would report
    a record as fully searchable on the strength of a chunk that is not in the index."""
    cid = _deck_record(conn)
    conn.execute("UPDATE campaign_chunks SET embedded = 0 WHERE campaign_id = ? "
                 "AND kind = 'body' AND chunk_index > 0", (cid,))
    conn.commit()

    out = core._rebuild_body_index(conn, cid)

    assert out["embedded"] == out["chunks"], "they were re-embedded, so they are searchable"
    assert all(r["embedded"] for r in store.body_chunks(conn, cid))


def test_two_calls_at_once_do_not_share_a_memo(conn):
    """Both memos are per-CALL, and the MCP server runs sync tools on worker threads — so a
    module global is shared by overlapping calls. Interleaved, the nesting logic left a dict
    behind after both scopes exited, and from then on every embed in the process was memoised
    across calls: the process-lifetime cache this was rewritten to remove, arriving by a door
    nobody was watching."""
    import threading

    import config

    calls = []
    real = embedding.embed
    embedding.embed = lambda text, **kw: (calls.append(text), [0.5] * config.EMBED_DIM)[1]
    both_inside = threading.Barrier(2)
    first_out = threading.Event()

    def early():
        with embedding.each_string_embedded_once():
            both_inside.wait(5)
        first_out.set()

    def late():
        with embedding.each_string_embedded_once():
            both_inside.wait(5)
            first_out.wait(5)

    try:
        threads = [threading.Thread(target=early), threading.Thread(target=late)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)

        calls.clear()
        embedding.embed_once("a sentence")
        embedding.embed_once("a sentence")
    finally:
        embedding.embed = real

    assert len(calls) == 2, (
        "a memo outlived the calls that opened it, so every later embed is cached"
    )


# ---------------------------------------------------------------------------------------
# D79 — reading the same record once per finding


def test_verifying_twelve_quotes_reads_the_record_once(conn, monkeypatch):
    """D79. Quote verification reads every chunk of the cited record on each save — twelve
    findings against one long deck is twelve full reads of the same unchanged text, inside
    the call somebody is waiting on."""
    cid = _deck_record(conn)
    reads = []
    real = store.text_on_file
    monkeypatch.setattr(store, "text_on_file",
                        lambda c, campaign_id: (reads.append(campaign_id),
                                                real(c, campaign_id))[1])

    core.save_evaluation(
        conn, subject_title="Lima launch", verdict="revise", approve_if="It is fixed.",
        summary="Several things.", cited_ids=[cid],
        findings=[{"severity": "should_fix", "kind": "missing_information",
                   "finding": f"Finding {i}", "fix": "Add it",
                   "precedent": {"campaign_id": cid,
                                 "quote": f"Section {i}. The Bogota launch will run"}}
                  for i in range(8)])

    assert reads.count(cid) <= 1, (
        f"the same record was read {reads.count(cid)} times to verify eight quotes"
    )


def test_checking_declared_expectations_reads_each_record_once(conn, monkeypatch, tmp_path):
    """D79's shape at a worse exponent, and §13.3 walked past it until review pointed.

    `_expected_inputs_never_supplied` read every record's whole text once PER EXPECTATION,
    and re-joined and re-lowercased each body on every pass — O(expectations x records) full
    reads of data that cannot change during the call."""
    import rulebook

    path = rulebook.overlay_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("version: 'acme-3'\nrules: []\nexpects:\n"
                    + "".join(f"  - id: thing-{i}\n    input: A thing {i}\n"
                              f"    why: Because.\n    looks_like: ['thing{i}']\n"
                              for i in range(4)), encoding="utf-8")
    rulebook.load.cache_clear()
    try:
        for n in range(5):
            core.ingest_campaign(conn, title=f"Peru launch {n}", market="Peru",
                                 status="concluded", deck_text=_DECK, confirm=True)
        reads = []
        real = store.text_on_file
        monkeypatch.setattr(store, "text_on_file",
                            lambda c, cid: (reads.append(cid), real(c, cid))[1])

        core.gaps(conn)

        per_record = {cid: reads.count(cid) for cid in set(reads)}
        assert per_record and max(per_record.values()) <= 1, (
            f"four expectations over five records cost {len(reads)} reads: {per_record}"
        )
    finally:
        rulebook.load.cache_clear()


def test_an_expectation_is_not_satisfied_by_two_records_between_them(conn, monkeypatch):
    """The memo makes it cheap to read each record once; it must not make it tempting to
    fold them into one string. A phrase spanning the end of one brief and the start of the
    next is in NEITHER of them, and reporting the expectation as met would close a real gap
    on evidence nobody wrote."""
    import rulebook

    path = rulebook.overlay_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("version: 'acme-3'\nrules: []\nexpects:\n"
                    "  - id: kpi-workbook\n    input: A KPI workbook\n"
                    "    why: Nothing can be reconciled without one.\n"
                    "    looks_like: ['workbook targets']\n", encoding="utf-8")
    rulebook.load.cache_clear()
    try:
        core.ingest_campaign(conn, title="First", market="Peru", status="concluded",
                             deck_text="The plan ends with the workbook", confirm=True)
        core.ingest_campaign(conn, title="Second", market="Peru", status="concluded",
                             deck_text="targets follow in the appendix", confirm=True)

        codes = {g["code"] for g in core.gaps(conn)["gaps"]}

        assert "expected_input_never_supplied" in codes, (
            "a phrase split across two briefs was counted as one brief carrying it"
        )
    finally:
        rulebook.load.cache_clear()


def test_a_nested_scope_keeps_what_the_outer_one_paid_for(conn, monkeypatch):
    """`save_evaluation` opens both scopes and reaches `prepare_evaluation`'s wrapper, which
    opens one of them again. A nested block that installed a FRESH dict would throw away
    everything the outer had already embedded — the memo quietly doing nothing on precisely
    the path that opens it twice."""
    import config
    import embedding

    calls = []
    monkeypatch.setattr(embedding, "embed",
                        lambda text, **kw: (calls.append(text), [0.5] * config.EMBED_DIM)[1])

    with embedding.each_string_embedded_once():
        embedding.embed_once("a sentence")
        with embedding.each_string_embedded_once():
            embedding.embed_once("a sentence")
        embedding.embed_once("a sentence")

    assert len(calls) == 1, "a nested scope discarded what the outer one had embedded"


def test_an_empty_library_costs_no_citation_scan(conn, monkeypatch):
    """`coverage` returns early on an empty library and used to do NO scan at all. Hoisting
    the citation read above that return made the cheapest path more expensive — a
    performance item taking something from somebody."""
    scans = []
    real = store.citations
    monkeypatch.setattr(store, "citations",
                        lambda *a, **k: (scans.append(1), real(*a, **k))[1])

    core.coverage(conn)

    assert not scans, "an empty library paid for a scan of a table it has no rows in"


# ---------------------------------------------------------------------------------------
# D91 — one scan per coverage cell


def test_coverage_reads_the_citations_once(conn, monkeypatch):
    """D91. `store.citations` is a full scan of `evaluations`, and `coverage` called it once
    per cell — O(cells × evaluations) for a list that is the same on every call."""
    for n in range(6):
        core.ingest_campaign(conn, title=f"Peru launch {n}", market=f"M{n % 3}",
                             status="concluded", deck_text=_DECK, confirm=True)
    scans = []
    real = store.citations
    monkeypatch.setattr(store, "citations",
                        lambda *a, **k: (scans.append(1), real(*a, **k))[1])

    report = core.coverage(conn)

    assert len(report["cells"]) >= 3, "not enough cells to make the point"
    assert len(scans) <= 1, f"{len(scans)} full scans for {len(report['cells'])} cells"


# ---------------------------------------------------------------------------------------
# D107 — scanning the registry on every metric write


def test_recording_a_metric_does_not_scan_the_registry_five_times(conn, monkeypatch):
    """D107. `retire_stale` runs on the write that changes staleness — which is right, §8.5
    argues it — but the registry is read five times over to do it."""
    cid = _deck_record(conn)
    scans = []
    real = store.metric_registry
    monkeypatch.setattr(store, "metric_registry",
                        lambda *a, **k: (scans.append(1), real(*a, **k))[1])

    metrics.record(conn, campaign_id=cid, key="roas", value=3.4, metric_type="actual")

    assert len(scans) <= 2, f"{len(scans)} registry reads to record one number"
