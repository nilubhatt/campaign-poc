"""
§6.6 (idea L): report the strength of the evidence, every time.

The review's words: *"A verdict resting on five concluded campaigns with verified outcomes and
one resting on a single proposed brief currently look identical. Fix: return an evidence line
with every judgment — how many precedents, how many concluded, how many with verified rather
than stated performance, and the top similarity. When one match dominates, say so."*

The load-bearing word is **identical**. The complaint is not that the numbers are missing; it
is that two judgments of very different worth are presented the same way, so a reader has no
signal to weigh them by. That makes this a shape problem rather than a data problem, and it
decides three things about the implementation.

**The server fills it** (D3). It is a fact about what was retrieved and what those records
carry — a model asserting "this rests on five concluded campaigns" is making a claim, and
§7.8's premise (a difference in a computed thing is a bug) only holds when the server counted.

**Counted from `cited_ids`, not from the evidence package.** What a judgment RESTS ON is what
it cited, not what it was handed. A verdict shown eight precedents and citing one rests on
one, and reporting eight would be the overstatement this whole item is about.

**A number nobody reads is not a signal.** Five counts plus a similarity is a row of digits;
"one match is carrying this" is a sentence. So the counts come with a `strength` code and a
line saying what it means — the same discipline §5.5 used for coverage cells and §6.1 for
`checked`, and never a word the counts cannot support.

Kept distinct from §5.3, as the plan records: the evidence line says what a judgment rests on,
`most_valuable_missing_input` says what would most improve it. One describes, the other asks.
"""
import pytest

import core
import store


def _campaign(conn, title, *, concluded=False, verified=None, detail=None):
    cid = core.ingest_campaign(
        conn, title=title, status="concluded" if concluded else "proposed",
        detail=detail or "Creator-led launch with four colourways and a compressed flight."
    )["campaign_id"]
    if verified:
        core.add_metrics(conn, campaign_id=cid, detail="CTR 3.4 percent, above benchmark.")
        store.update_campaign(conn, cid,
                              tags=[{"value": verified, "source": "verified"}])
    elif concluded:
        core.add_metrics(conn, campaign_id=cid, detail="CTR 3.4 percent.")
    return cid


def _evaluation(**over):
    base = dict(subject_title="Colombia v1", verdict="revise", summary="A stretch.",
                approve_if="It is fixed.",
                findings=[{"severity": "should_fix", "kind": "missing_information",
                           "finding": "No end date", "fix": "Add one"}])
    base.update(over)
    return base


# ── the two judgments that used to look identical ───────────────────────────

def test_a_judgment_on_five_measured_campaigns_does_not_look_like_one_on_a_single_brief(conn):
    """The review's own sentence, and the word doing the work in it is "identical". Two
    judgments of very different worth were presented the same way, so nothing told a reader
    which to lean on."""
    strong_ids = [_campaign(conn, f"Measured {n}", concluded=True, verified="performed_well")
                  for n in range(5)]
    thin_id = _campaign(conn, "A proposal")

    strong = core.save_evaluation(conn, cited_ids=strong_ids, **_evaluation())
    thin = core.save_evaluation(conn, cited_ids=[thin_id], **_evaluation())

    assert strong["evidence"]["precedents"] == 5
    assert strong["evidence"]["concluded"] == 5
    assert strong["evidence"]["verified"] == 5
    assert thin["evidence"] == {**thin["evidence"], "precedents": 1, "concluded": 0,
                                "verified": 0}
    assert strong["evidence"]["strength"] != thin["evidence"]["strength"]


def test_a_stated_performance_claim_is_counted_apart_from_a_measured_one(conn):
    """"How many with verified rather than stated performance" — the review asks for the
    split, and the whole tag-provenance apparatus exists because the two are not the same
    kind of thing."""
    measured = _campaign(conn, "Measured", concluded=True, verified="performed_well")
    claimed = _campaign(conn, "Claimed", concluded=True)
    store.update_campaign(conn, claimed, tags=["performed_well"])

    result = core.save_evaluation(conn, cited_ids=[measured, claimed], **_evaluation())

    assert result["evidence"]["precedents"] == 2
    assert result["evidence"]["concluded"] == 2
    assert result["evidence"]["verified"] == 1


# ── what it rests on, not what it was shown ─────────────────────────────────

def test_it_counts_what_the_judgment_cited_not_what_it_was_handed(conn):
    """A verdict shown eight precedents and citing one rests on one. Counting the evidence
    package would be the overstatement this item exists to prevent, made by the field written
    to prevent it."""
    for n in range(8):
        _campaign(conn, f"Shown {n}", concluded=True, verified="performed_well")
    one = _campaign(conn, "The one cited", concluded=True, verified="performed_well")

    result = core.save_evaluation(conn, cited_ids=[one], **_evaluation())
    assert result["evidence"]["precedents"] == 1


def test_a_judgment_that_cites_nothing_says_so(conn):
    """The weakest possible state, and the one most in need of a label: a verdict resting on
    no precedent at all is an opinion, and it used to look like every other verdict."""
    result = core.save_evaluation(conn, cited_ids=[], **_evaluation())

    assert result["evidence"]["precedents"] == 0
    assert result["evidence"]["strength"] == "no_precedent"
    assert "opinion" in result["evidence"]["what_it_means"].lower()


def test_a_citation_that_resolves_to_nothing_is_not_counted_as_precedent(conn):
    """§6.1 verifies the quote on a finding's precedent; `cited_ids` is a separate list and
    nothing checked it. An id that resolves to no record cannot be evidence, and counting it
    would inflate the one number this item exists to make honest."""
    real = _campaign(conn, "Real", concluded=True, verified="performed_well")

    result = core.save_evaluation(conn, cited_ids=[real, "camp_nope"], **_evaluation())

    assert result["evidence"]["precedents"] == 1
    assert result["evidence"]["unresolved_citations"] == ["camp_nope"]


# ── when one match dominates, say so ────────────────────────────────────────

def test_one_match_carrying_the_weight_is_named(conn):
    """The review asks for this in as many words. A judgment resting on six records where one
    is doing all the work is not a judgment resting on six records."""
    dominant = _campaign(conn, "The Peru launch", concluded=True, verified="performed_well",
                         detail="Creator-led launch with four colourways and a compressed "
                                "three-week flight across Lima and Arequipa.")
    others = [_campaign(conn, f"Unrelated {n}", detail="zzz qqq vvv unrelated wording.")
              for n in range(5)]

    result = core.save_evaluation(
        conn, cited_ids=[dominant] + others,
        **_evaluation(summary="Creator-led launch with four colourways and a compressed "
                              "three-week flight across Lima and Arequipa."))

    assert result["evidence"]["dominated_by"] == dominant
    assert "one" in result["evidence"]["what_it_means"].lower()


def test_evenly_weighted_evidence_is_not_reported_as_dominated(conn):
    """The claim only means something if it is sometimes false."""
    ids = [_campaign(conn, f"Alike {n}", concluded=True, verified="performed_well")
           for n in range(4)]

    result = core.save_evaluation(conn, cited_ids=ids, **_evaluation())
    assert result["evidence"]["dominated_by"] is None


def test_the_top_similarity_is_reported(conn):
    close = _campaign(conn, "Close", detail="A compressed three-week flight in Lima.")
    result = core.save_evaluation(
        conn, cited_ids=[close],
        **_evaluation(summary="A compressed three-week flight in Lima."))

    assert 0 < result["evidence"]["top_similarity"] <= 1


# ── a number nobody reads is not a signal ───────────────────────────────────

@pytest.mark.parametrize("strength", ["no_precedent", "single_example", "unmeasured",
                                      "measured"])
def test_every_strength_says_what_it_means(conn, strength):
    """§5.5's discipline for coverage cells and §6.1's for `checked`: a stable code beside a
    sentence, and the sentence never claims more than the counts support."""
    assert core._EVIDENCE_STRENGTH[strength]


def test_the_strength_never_outruns_the_counts(conn):
    """"Measured" on a judgment resting on stated impressions would be the confident
    unfounded claim this whole review was written against."""
    claimed = _campaign(conn, "Claimed", concluded=True)
    store.update_campaign(conn, claimed, tags=["performed_well"])

    result = core.save_evaluation(conn, cited_ids=[claimed], **_evaluation())
    assert result["evidence"]["strength"] == "single_example"


# ── the server's fact, and it survives ──────────────────────────────────────

def test_the_model_cannot_write_it(conn):
    """D3: it is a server fact by definition. A model asserting "this rests on five concluded
    campaigns" is making a claim, and §7.8's premise holds only when the server counted."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, evidence={"precedents": 9}, **_evaluation())
    assert "server" in str(e.value).lower()


def test_it_is_recorded_and_on_the_read_path(conn):
    """A strength line that lives for one response cannot be what a later reader weighs the
    judgment by, and weighing it later is the point."""
    cid = _campaign(conn, "Measured", concluded=True, verified="performed_well")
    saved = core.save_evaluation(conn, cited_ids=[cid], **_evaluation())

    read_back = core.get_evaluation(conn, evaluation_id=saved["evaluation_id"])
    assert read_back["evidence"]["precedents"] == 1
    assert read_back["evidence"]["strength"]


def test_it_reaches_the_model_over_the_protocol(conn):
    import asyncio
    import json

    cid = _campaign(conn, "Measured", concluded=True, verified="performed_well")

    async def call():
        import mcp_server
        result = await mcp_server.mcp.call_tool(
            "save_evaluation", _evaluation(cited_ids=[cid]))
        return json.loads(result.content[0].text)

    saved = asyncio.run(call())
    assert saved["evidence"]["strength"] == "single_example"
    assert "evidence" in saved["note"] or "rests on" in saved["note"]


# ── D65: the same question, one level up ────────────────────────────────────

def test_a_cell_where_one_record_is_cited_by_everything_is_not_six_examples(conn):
    """D65, owed to this item. `coverage` reads a cell of six measured campaigns as
    `measured`, but if every verdict in it cites the same one record, the library's real
    depth there is one — and "one example carrying the weight" is a fact about JUDGMENTS,
    which the schema already holds in `cited_ids` and nothing read."""
    ids = [_campaign(conn, f"Lima {n}", concluded=True, verified="performed_well")
           for n in range(4)]
    for cid in ids:
        store.update_campaign(conn, cid, market="LATAM")
    for n in range(3):
        core.save_evaluation(conn, cited_ids=[ids[0]],
                             **_evaluation(subject_title=f"New brief {n}"))

    cell = [c for c in core.coverage(conn)["cells"] if c["market"] == "LATAM"][0]

    assert cell["campaigns"] == 4
    assert cell["cited_share_top"] == 1.0, "every judgment cited the same record"
    assert sorted(cell["never_cited"]) == sorted(ids[1:])


def test_a_cell_nobody_has_judged_reports_no_concentration_rather_than_total(conn):
    """Zero judgments is not "one record carries everything" — with no citations there is no
    concentration to report, and 1.0 would read as the worst possible state."""
    cid = _campaign(conn, "Lima", concluded=True, verified="performed_well")
    store.update_campaign(conn, cid, market="LATAM")

    cell = [c for c in core.coverage(conn)["cells"] if c["market"] == "LATAM"][0]
    assert cell["cited_share_top"] is None
    assert cell["never_cited"] == [cid]


def test_evenly_cited_records_report_a_low_share(conn):
    ids = [_campaign(conn, f"Lima {n}", concluded=True, verified="performed_well")
           for n in range(3)]
    for cid in ids:
        store.update_campaign(conn, cid, market="LATAM")
        core.save_evaluation(conn, cited_ids=[cid], **_evaluation())

    cell = [c for c in core.coverage(conn)["cells"] if c["market"] == "LATAM"][0]
    assert cell["cited_share_top"] == pytest.approx(1 / 3)
    assert cell["never_cited"] == []


# ── review round ────────────────────────────────────────────────────────────

def test_an_embedder_outage_is_not_a_similarity_of_zero(conn):
    """`Unavailable` is a `ValueError`, and catching it with one reported
    `top_similarity: 0.0` — indistinguishable from "dissimilar" — in the same response where
    §6.4 correctly said `could_not_check`. That is the collapse §6.4 exists to prevent,
    reintroduced 240 lines below it."""
    import embedding

    ids = [_campaign(conn, f"Measured {n}", concluded=True, verified="performed_well")
           for n in range(2)]
    original = embedding.embed
    embedding.embed = lambda *a, **k: (_ for _ in ()).throw(
        embedding.Unavailable("ollama is not running"))
    try:
        result = core.save_evaluation(conn, cited_ids=ids, **_evaluation())
    finally:
        embedding.embed = original

    assert result["evidence"]["similarity_checked"] is False
    assert result["evidence"]["top_similarity"] is None
    assert "could not be checked" in result["evidence"]["what_it_means"]


def test_five_concluded_campaigns_with_results_are_not_reported_as_unmeasured(conn):
    """The ladder counted verified performance TAGS and the sentence claimed measured
    RESULTS, so five concluded campaigns with real metric rows were told "none of them has
    measured results" — while `coverage` on the same five said `measured, with_outcomes 5`.
    Two definitions of measured, contradicting each other in adjacent fields."""
    ids = [_campaign(conn, f"Measured {n}", concluded=True) for n in range(5)]

    result = core.save_evaluation(conn, cited_ids=ids, **_evaluation())

    assert result["evidence"]["with_results"] == 5
    assert result["evidence"]["verified"] == 0, "metrics recorded, no verdict tagged"
    assert result["evidence"]["strength"] == "measured"
    assert "none of them" not in result["evidence"]["what_it_means"].lower()


def test_five_briefs_do_not_read_the_same_as_one_brief(conn):
    """The review's own complaint, one rung down and inside the item written to fix it: five
    unmeasured proposals and a single proposal both came back `single_example`."""
    briefs = [_campaign(conn, f"Proposal {n}") for n in range(5)]

    many = core.save_evaluation(conn, cited_ids=briefs, **_evaluation())
    one = core.save_evaluation(conn, cited_ids=briefs[:1], **_evaluation())

    assert many["evidence"]["strength"] == "unmeasured"
    assert one["evidence"]["strength"] == "single_example"


def test_a_partly_measured_judgment_says_so_rather_than_rounding(conn):
    """Rounding up flatters the judgment and rounding down understates it; both are the
    "looks identical" failure with a different rounding rule."""
    measured = _campaign(conn, "Measured", concluded=True)
    proposal = _campaign(conn, "Proposal")

    result = core.save_evaluation(conn, cited_ids=[measured, proposal], **_evaluation())
    assert result["evidence"]["strength"] == "partly_measured"


def test_a_cited_record_too_far_down_the_library_has_no_similarity_not_zero(conn):
    """A record outside the scan has NO similarity, which is not a similarity of zero — and
    the difference decides whether "one match is carrying this" is a statement about the
    evidence or about the retrieval window."""
    close = _campaign(conn, "Close", detail="A compressed three-week flight in Lima.")
    far = _campaign(conn, "Far", detail="zzz qqq vvv entirely unrelated wording.")

    result = core.save_evaluation(
        conn, cited_ids=[close, far],
        **_evaluation(summary="A compressed three-week flight in Lima."))

    # Both are in a two-record library, so both are scored and the dominance claim is about
    # the evidence. What must never happen is an unscored record standing in at 0.0.
    assert result["evidence"]["top_similarity"] > 0
    assert result["evidence"]["similarity_checked"] is True


def test_the_count_line_reports_both_meanings_of_measured(conn):
    """The review asks for "how many with verified rather than stated performance"; `coverage`
    and `readiness` mean metric rows. Both are real and they are not the same number, so the
    line carries both rather than picking one and using the other's word for it."""
    tagged = _campaign(conn, "Tagged", concluded=True, verified="performed_well")
    metrics_only = _campaign(conn, "Metrics only", concluded=True)

    result = core.save_evaluation(conn, cited_ids=[tagged, metrics_only], **_evaluation())

    assert result["evidence"]["with_results"] == 2
    assert result["evidence"]["verified"] == 1
    assert "2 with recorded results" in result["note"]
    assert "1 with a measured performance verdict" in result["note"]


def test_the_strength_line_survives_a_database_that_predates_it(tmp_path, monkeypatch):
    """Third time in this file: §6.1's `text_on_file` and §6.4's disconfirming search each
    crashed every judgment on an upgraded v0.2.0 database before this. A save path may not
    assume a table or column that a release added, and every one of these runs on save.

    The upgrade runs through `store.init_db()` — the actual entry point, which creates the
    tables that did not exist, sets up the vector store, and then migrates the ones that did.
    Reassembling those steps by hand made the fixture fail on six tables in turn and taught
    nothing, because no installed copy is ever in the state a partial reassembly produces.

    What it DID surface is recorded as D89. `_migrate_schema` adds exactly three campaign
    columns — `collection`, `markets`, `commentary_checked` — while the readers assume the
    whole schema: `tags` and `supersedes` are read unconditionally and are added by nothing.
    `CREATE TABLE IF NOT EXISTS` cannot add a column to a table that already exists, so any
    installed database whose `campaigns` predates either one breaks the most-used reader in
    the codebase, and no test would catch it. Different gap, different item.
    """
    import sqlite3
    import time

    path = tmp_path / "v020.db"
    legacy = sqlite3.connect(path)
    legacy.executescript("""
        CREATE TABLE campaigns (id TEXT PRIMARY KEY, title TEXT NOT NULL, detail TEXT,
                                tags TEXT NOT NULL DEFAULT '[]', status TEXT,
                                region TEXT, market TEXT, supersedes TEXT,
                                deck_text TEXT, asset_path TEXT,
                                embedded INTEGER NOT NULL DEFAULT 0,
                                created_at REAL, updated_at REAL);
        CREATE TABLE evaluations (
            id TEXT PRIMARY KEY, campaign_id TEXT, subject_title TEXT NOT NULL,
            cited_ids TEXT, analysis TEXT NOT NULL, predictions TEXT,
            created_at REAL NOT NULL);
        CREATE TABLE reconciliations (
            id TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL, actual TEXT,
            comparison TEXT NOT NULL, created_at REAL NOT NULL);
    """)
    legacy.execute("INSERT INTO evaluations VALUES ('ev_old', NULL, 'v1', '[]', ?, NULL, ?)",
                   ("An essay.", time.time()))
    legacy.execute("INSERT INTO campaigns (id, title, detail, created_at, updated_at) "
                   "VALUES (?,?,?,?,?)",
                   ("camp_old", "Peru launch", "A brief.", time.time(), time.time()))
    legacy.commit()
    legacy.close()

    import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", path)
    monkeypatch.setattr(config, "ASSET_DIR", tmp_path / "assets")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "EMBED_PROVIDER", "hash")
    store.init_db()
    conn = store.connect()

    saved = core.save_evaluation(conn, cited_ids=["camp_old"], **_evaluation())
    assert saved["evidence"]["precedents"] == 1
    assert saved["evidence"]["with_results"] == 0
    assert saved["evidence"]["strength"] == "single_example"


@pytest.mark.parametrize("scored,expected", [
    # The gap, isolated: a high top score with everything close behind is not domination.
    ([(0.90, "a"), (0.85, "b"), (0.80, "c")], None),
    ([(0.90, "a"), (0.40, "b")], "a"),
    # The floor, isolated: the gap here is 0.19, comfortably over the threshold, and the top
    # match still resembles almost nothing. Domination is a claim that one record is carrying
    # the judgment, which cannot be true of a record that is barely related to the brief.
    # (The first version of this case used a gap of 0.10, so the GAP refused it and the floor
    # was free to be deleted — the two thresholds masking each other one level down.)
    ([(0.19, "a"), (0.00, "b")], None),
    ([(0.21, "a"), (0.00, "b")], "a"),
    ([(0.00, "a"), (0.00, "b"), (0.00, "c")], None),
    # One scored record cannot dominate anything; there is nothing to dominate.
    ([(0.95, "a")], None),
    ([], None),
])
def test_domination_needs_both_a_high_top_and_a_real_gap(scored, expected):
    """Both thresholds survived deletion against the whole suite, because in any realistic
    fixture they mask each other: where the gap is large the top score is usually above the
    floor as well, so a test exercising one leaves the other free to be removed."""
    assert core._dominant(scored) == expected


def test_citing_the_same_record_twice_does_not_make_it_two_precedents(conn):
    """Removing the dedup left the whole suite green. A judgment that names one campaign
    three times rests on one campaign, and counting three would be the inflation this field
    exists to prevent — the same mistake as counting what was shown rather than cited, made
    with the caller's own list."""
    cid = _campaign(conn, "Measured", concluded=True, verified="performed_well")

    result = core.save_evaluation(conn, cited_ids=[cid, cid, cid], **_evaluation())
    assert result["evidence"]["precedents"] == 1
