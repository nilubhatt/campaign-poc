"""
§6.3 (idea I): close the prediction loop.

The review's words: *"The v1 Colombia evaluation predicted that the same structure would
return rearranged unless the premise was settled. Colombia v2 proved it right. Nothing in the
system recorded that it had been right, and `reconcile_evaluation` has never once run. Fix:
when a record arrives that supersedes another, surface the prior evaluation's predictions
immediately and ask which held. That is how the library learns whether its own judgment is
any good, and it costs one prompt."*

The load-bearing observation is the second sentence. `reconcile_evaluation` exists, works,
and has never run — because it depends on somebody choosing to go back, and nobody does. The
fix is not another tool; it is a MOMENT. A superseding record arriving is the one instant
where the answer is both known and cheap: the person uploading v2 is looking at the thing
that proves or refutes what was said about v1.

Three tracker rows land here.

**D57** is the item itself: the predictions, surfaced at that moment.

**D41** is the supersession offer, which both reviewers condemned in 5.2 because it was
prefilled from a similarity match — a claim about somebody's INTENT built on a fact about
text. It comes back only on evidence the schema actually holds, and it needs a write path,
because `update_campaign` cannot set `supersedes` at all: today a supersession can only be
declared at upload, and a wrong one is undoable only by deleting the record.

**D64** is the chain: `diff(v1, v3)` compared only adjacent judgments, so a finding resolved
in v2 was reported against v3 as `no_longer_raised` — "neither resolved nor repeated", when
the library holds the record of it being resolved.
"""
import pytest

import core
import store


@pytest.fixture
def peru(conn):
    return core.ingest_campaign(
        conn, title="Peru launch", status="concluded",
        detail="Seeded one colourway per creator, with a posting date per asset."
    )["campaign_id"]


def _judge(conn, campaign_id, peru, **over):
    base = dict(
        subject_title="Colombia v1", verdict="revise",
        summary="The premise is unsettled.",
        campaign_id=campaign_id,
        predictions={"predicted_ctr_range": "1.2-1.8%",
                     "recommendation": "Settle the premise or the same structure returns"},
        findings=[{"severity": "should_fix", "kind": "precedent_departure",
                   "departure": "regression",
                   "finding": "Seeds four colourways where Peru used one",
                   "precedent": {"campaign_id": peru,
                                 "quote": "Seeded one colourway per creator"}}])
    base.update(over)
    return core.save_evaluation(conn, **base)


# ── the moment ──────────────────────────────────────────────────────────────

def test_a_superseding_upload_surfaces_what_was_predicted(conn, peru):
    """The whole item. `reconcile_evaluation` has worked all along and has never once run,
    because it needs somebody to decide to go back. This is the one moment where the answer
    is both known and free: the person uploading v2 is looking at the thing that settles it."""
    v1 = core.ingest_campaign(conn, title="Colombia v1",
                              detail="Seeds four colourways.")["campaign_id"]
    judged = _judge(conn, v1, peru)

    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1,
                              detail="Seeds four colourways, premise unchanged.")

    loop = v2["earlier_judgment"]
    assert loop["campaign_id"] == v1
    assert loop["evaluation_id"] == judged["evaluation_id"]
    assert loop["predictions"]["predicted_ctr_range"] == "1.2-1.8%"
    assert loop["verdict"] == "revise"
    assert "which" in loop["ask"].lower(), "it has to ask, not just display"


def test_an_upload_that_supersedes_nothing_says_nothing(conn, peru):
    """Guidance that is always there is guidance nobody reads, and this one is a question the
    user cannot answer for a record with no earlier version."""
    fresh = core.ingest_campaign(conn, title="Chile v1", detail="A new brief.")
    assert "earlier_judgment" not in fresh


def test_a_superseded_record_that_was_never_judged_offers_nothing_to_check(conn):
    """There is no prediction to test. Saying "which of these held" over an empty list is the
    confident-shape-with-nothing-in-it failure this codebase keeps finding."""
    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1, detail="Seeds four.")
    assert "earlier_judgment" not in v2


def test_a_judgment_with_no_predictions_still_offers_its_verdict_to_check(conn, peru):
    """Predictions are optional and most judgments have none. The verdict and the findings
    are still a claim that either held or did not — surfacing nothing because one field is
    empty would lose the case the review actually described, which was a RECOMMENDATION that
    came true, not a CTR number."""
    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    _judge(conn, v1, peru, predictions=None)
    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1, detail="Seeds four.")

    loop = v2["earlier_judgment"]
    assert loop["predictions"] is None
    assert loop["verdict"] == "revise"
    assert loop["open_findings"], "the findings are the claim, when there is no forecast"


def test_the_loop_is_offered_as_an_action_that_can_actually_be_called(conn, peru):
    """§5.2's rule: an offer whose write path does not exist is worse than no offer. This one
    ends in `save_reconciliation`, which is the tool that records whether the library's own
    judgment was any good."""
    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    judged = _judge(conn, v1, peru)
    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1, detail="Seeds four.")

    offers = [a for a in v2["next_actions"] if a["tool"] == "reconcile_evaluation"]
    assert offers, [a["tool"] for a in v2["next_actions"]]
    assert offers[0]["prefilled_args"]["evaluation_id"] == judged["evaluation_id"]


def test_a_judgment_already_reconciled_is_not_offered_again(conn, peru):
    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    judged = _judge(conn, v1, peru)
    store.insert_reconciliation(conn, evaluation_id=judged["evaluation_id"],
                                comparison="Checked already.", actual="It held.")

    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1, detail="Seeds four.")
    assert "earlier_judgment" not in v2


# ── D41: the write path, and the offer that needs it ────────────────────────

def test_supersession_can_be_declared_after_the_fact(conn):
    """It could only be declared at upload. Somebody who uploads v2 and realises afterwards
    had no way to say so — and `update_campaign`'s docstring lists every other field."""
    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    v2 = core.ingest_campaign(conn, title="Colombia v2", detail="Seeds two.")["campaign_id"]

    assert store.update_campaign(conn, v2, supersedes=v1)
    assert store.get_campaign(conn, v2)["supersedes"] == v1
    assert v1 in store.get_superseded_campaign_ids(conn)


def test_a_supersession_declared_by_mistake_can_be_taken_back(conn):
    """A wrong yes hid a record from every future search and was undoable only by deleting
    the campaign — which is why 5.2's reviewers called the offer dangerous rather than merely
    wrong."""
    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1,
                              detail="Seeds two.")["campaign_id"]

    assert store.update_campaign(conn, v2, supersedes="")
    assert store.get_campaign(conn, v2)["supersedes"] is None
    assert v1 not in store.get_superseded_campaign_ids(conn)


def test_a_record_cannot_supersede_itself_or_something_that_is_not_there(conn):
    v2 = core.ingest_campaign(conn, title="Colombia v2", detail="Seeds two.")["campaign_id"]
    with pytest.raises(ValueError):
        store.update_campaign(conn, v2, supersedes=v2)
    with pytest.raises(ValueError):
        store.update_campaign(conn, v2, supersedes="camp_nonexistent")


def test_a_supersession_cycle_is_refused(conn):
    """v1 supersedes v2 supersedes v1 hides both from every search, and the chain walk below
    would not terminate."""
    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1,
                              detail="Seeds two.")["campaign_id"]
    with pytest.raises(ValueError):
        store.update_campaign(conn, v1, supersedes=v2)


def test_the_offer_comes_back_only_on_evidence_the_schema_holds(conn, peru):
    """5.2 prefilled it from `closest_precedent` — a similarity match, asserted by the model.
    "This replaces that" is a claim about somebody's intent; a similarity score is a fact
    about text. Two judgments of the same brief that a diff calls comparable is evidence of
    the right kind, and it is the only kind offered on."""
    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    v2 = core.ingest_campaign(conn, title="Colombia v2", detail="Seeds two.")["campaign_id"]
    _judge(conn, v1, peru)
    _judge(conn, v2, peru, subject_title="Colombia v2")

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)
    offers = [a for a in diff["next_actions"] if a["tool"] == "update_campaign"]
    assert offers, [a["tool"] for a in diff["next_actions"]]
    assert offers[0]["prefilled_args"] == {"campaign_id": v2, "supersedes": v1}
    # The record it would hide has to be named where the user reads the label, not only in
    # the arguments they never see.
    assert "Colombia v1" in offers[0]["label"]
    assert offers[0]["consent"] == "ask"


def test_the_offer_is_not_made_when_the_link_already_exists(conn, peru):
    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1,
                              detail="Seeds two.")["campaign_id"]
    _judge(conn, v1, peru)
    _judge(conn, v2, peru, subject_title="Colombia v2")

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)
    assert not [a for a in diff["next_actions"] if a["tool"] == "update_campaign"]


# ── D64: the chain ──────────────────────────────────────────────────────────

def test_a_correction_taken_in_v2_is_not_reported_as_dropped_in_v3(conn, peru):
    """Only adjacent judgments were compared, so `diff(v1, v3)` reported a finding v2 had
    explicitly resolved as `no_longer_raised` — "neither resolved nor repeated" — when the
    library holds the record of it being resolved. That is the library forgetting its own
    evidence and then hedging about it."""
    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    first = _judge(conn, v1, peru)
    finding_id = first["findings"][0]["id"]

    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1,
                              detail="Seeds one.")["campaign_id"]
    _judge(conn, v2, peru, subject_title="Colombia v2", verdict="approve",
           summary="Settled.", findings=[],
           resolved=[{"finding_id": finding_id, "was": "Seeds four colourways",
                      "now": "Seeds one"}])

    v3 = core.ingest_campaign(conn, title="Colombia v3", supersedes=v2,
                              detail="Seeds one, dated.")["campaign_id"]
    _judge(conn, v3, peru, subject_title="Colombia v3", verdict="approve",
           summary="Fine.", findings=[])

    diff = core.diff_campaigns(conn, earlier=v1, later=v3)
    assert [f["id"] for f in diff["adopted"]] == [finding_id]
    assert not diff["no_longer_raised"]
    assert diff["through"] == [v1, v2, v3], "say which records the answer was assembled from"


def test_the_chain_is_only_walked_when_the_records_are_actually_linked(conn, peru):
    """Two unrelated records must not be joined by a chain that does not exist."""
    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    other = core.ingest_campaign(conn, title="Chile v1", detail="Seeds two.")["campaign_id"]
    _judge(conn, v1, peru)
    _judge(conn, other, peru, subject_title="Chile v1")

    diff = core.diff_campaigns(conn, earlier=v1, later=other)
    assert diff["through"] == [v1, other]


def test_the_moment_reaches_the_model_over_the_protocol(conn, peru):
    import asyncio
    import json

    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    _judge(conn, v1, peru)

    async def call():
        import mcp_server
        result = await mcp_server.mcp.call_tool("upload_campaign", {
            "title": "Colombia v2", "detail": "Seeds four, premise unchanged.",
            "supersedes": v1, "confirm": True})
        return json.loads(result.content[0].text)

    uploaded = asyncio.run(call())
    assert uploaded["earlier_judgment"]["predictions"]["predicted_ctr_range"] == "1.2-1.8%"
