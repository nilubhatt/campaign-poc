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
