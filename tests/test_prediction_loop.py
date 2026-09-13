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
        # §6.5: a revise says what would end it, and a finding above a note says what to
        # do about it.
        approve_if="One colourway per creator.",
        findings=[{"severity": "should_fix", "kind": "precedent_departure",
                   "departure": "regression", "fix": "Seed one colourway per creator",
                   "finding": "Seeds four colourways where Peru used one",
                   "precedent": {"campaign_id": peru,
                                 "quote": "Seeded one colourway per creator"}}])
    base.update(over)
    if base.get("verdict") != "revise":
        base.pop("approve_if", None)
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


def test_the_loop_survives_being_accepted(conn, peru):
    """§5.2's rule is not "the tool must exist" — it is that ACCEPTING must work. The first
    version offered `reconcile_evaluation`, which looks up measured results on the record
    being replaced: a brief that never ran. Accepting returned "no actual metrics on file"
    and told the model to record results for a proposal, which is §5.2's own failure one
    field over. So this test CALLS the offer instead of asserting its arguments."""
    import mcp_server

    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    judged = _judge(conn, v1, peru)
    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1, detail="Seeds four.")

    offers = [a for a in v2["next_actions"] if a["tool"] == "save_reconciliation"]
    assert offers, [a["tool"] for a in v2["next_actions"]]
    assert offers[0]["prefilled_args"]["evaluation_id"] == judged["evaluation_id"]
    assert offers[0]["needs"], "the user supplies the answer; say so"

    accepted = getattr(mcp_server, offers[0]["tool"])(
        **offers[0]["prefilled_args"],
        comparison="The premise came back unsettled, so the prediction held.")
    assert "error" not in accepted, accepted


def test_the_ask_names_the_tool_the_offer_actually_calls(conn, peru):
    """The response said "record the answer with save_reconciliation" while `next_actions`
    offered `reconcile_evaluation` — two different instructions in one payload, one of which
    failed."""
    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    _judge(conn, v1, peru)
    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1, detail="Seeds four.")

    tools = {a["tool"] for a in v2["next_actions"]}
    assert "save_reconciliation" in v2["earlier_judgment"]["ask"]
    assert "save_reconciliation" in tools
    assert "reconcile_evaluation" not in tools


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
    with pytest.raises(ValueError) as itself:
        store.update_campaign(conn, v2, supersedes=v2)
    # The MESSAGE, not just the refusal: removing the self-check left the suite green,
    # because the cycle check catches it too — and tells the user their record is in a cycle
    # with itself, which is a worse sentence to have to act on than "it cannot supersede
    # itself".
    assert "itself" in str(itself.value)

    with pytest.raises(ValueError) as missing:
        store.update_campaign(conn, v2, supersedes="camp_nonexistent")
    assert "camp_nonexistent" in str(missing.value)


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
    about text.

    The evidence is a finding id: one judgment naming a finding from the other, or a recorded
    resolution. That is a statement somebody made about these two being versions of one
    brief, which is this module's own rule that a resemblance never outranks a statement."""
    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    v2 = core.ingest_campaign(conn, title="Colombia v2", detail="Seeds two.")["campaign_id"]
    first = _judge(conn, v1, peru)
    _judge(conn, v2, peru, subject_title="Colombia v2", verdict="approve",
           summary="Settled.", findings=[],
           resolved=[{"finding_id": first["findings"][0]["id"],
                      "was": "Seeds four", "now": "Seeds two"}])

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)
    offers = [a for a in diff["next_actions"] if a["tool"] == "update_campaign"]
    assert offers, [a["tool"] for a in diff["next_actions"]]
    assert offers[0]["prefilled_args"] == {"campaign_id": v2, "supersedes": v1}
    # The record it would hide has to be named where the user reads the label, not only in
    # the arguments they never see.
    assert "Colombia v1" in offers[0]["label"]
    assert offers[0]["consent"] == "ask"


def test_two_unrelated_campaigns_are_never_offered_up_to_hide_each_other(conn, peru):
    """The first version of this fix fired whenever a diff of two judged records completed —
    which is not a fact about the records, it is a fact about which two ids the CALLER
    passed. Two unrelated campaigns got an offer to hide one of them, in the same response
    whose warning says the library cannot tell which came first."""
    colombia = core.ingest_campaign(conn, title="Colombia v1",
                                    detail="Seeds four.")["campaign_id"]
    chile = core.ingest_campaign(conn, title="Chile v1", detail="Seeds two.")["campaign_id"]
    _judge(conn, colombia, peru)
    _judge(conn, chile, peru, subject_title="Chile v1")

    diff = core.diff_campaigns(conn, earlier=colombia, later=chile)
    assert any(w["code"] == "version_order_unverified" for w in diff["warnings"])
    assert not [a for a in diff["next_actions"] if a["tool"] == "update_campaign"], \
        "it cannot both say it does not know the order and offer to record one"


def test_a_record_already_replaced_is_not_offered_up_to_be_replaced_again(conn, peru):
    """Reading only the two `supersedes` columns missed fan-in: an earlier record already
    replaced by a THIRD record was still offered, silently making two records claim it."""
    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    first = _judge(conn, v1, peru)
    core.ingest_campaign(conn, title="Colombia v2a", supersedes=v1, detail="Seeds two.")
    v2b = core.ingest_campaign(conn, title="Colombia v2b", detail="Seeds one.")["campaign_id"]
    _judge(conn, v2b, peru, subject_title="Colombia v2b", verdict="approve",
           summary="Settled.", findings=[],
           resolved=[{"finding_id": first["findings"][0]["id"],
                      "was": "Seeds four", "now": "Seeds one"}])

    diff = core.diff_campaigns(conn, earlier=v1, later=v2b)
    assert not [a for a in diff["next_actions"] if a["tool"] == "update_campaign"]


def test_the_offer_is_not_made_when_the_link_already_exists(conn, peru):
    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1,
                              detail="Seeds two.")["campaign_id"]
    first = _judge(conn, v1, peru)
    _judge(conn, v2, peru, subject_title="Colombia v2", verdict="approve",
           summary="Settled.", findings=[],
           resolved=[{"finding_id": first["findings"][0]["id"],
                      "was": "Seeds four", "now": "Seeds two"}])

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


# ── second round ────────────────────────────────────────────────────────────

def test_declaring_the_supersession_late_asks_the_same_question(conn, peru):
    """The loop fired only at upload — and the D41 offer is accepted by calling
    `update_campaign`, so the flow this item constructs (diff two judged versions, offer to
    link, accept) landed on the one path where nothing fired. That user has both judgments on
    screen because they just compared them, which is a better moment, not a worse one."""
    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    judged = _judge(conn, v1, peru)
    v2 = core.ingest_campaign(conn, title="Colombia v2", detail="Seeds two.")["campaign_id"]

    linked = core.update_campaign(conn, v2, supersedes=v1)

    assert linked["earlier_judgment"]["evaluation_id"] == judged["evaluation_id"]
    assert "save_reconciliation" in [a["tool"] for a in linked["next_actions"]]


def test_an_ordinary_edit_does_not_ask_anything(conn, peru):
    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    _judge(conn, v1, peru)
    assert "earlier_judgment" not in core.update_campaign(conn, v1, status="concluded")


def test_the_late_link_reaches_the_model_over_the_protocol(conn, peru):
    import asyncio
    import json

    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    _judge(conn, v1, peru)
    v2 = core.ingest_campaign(conn, title="Colombia v2", detail="Seeds two.")["campaign_id"]

    async def call(name, args):
        import mcp_server
        return json.loads((await mcp_server.mcp.call_tool(name, args)).content[0].text)

    linked = asyncio.run(call("update_campaign", {"campaign_id": v2, "supersedes": v1}))
    assert linked["supersedes"] == v1
    assert linked["earlier_judgment"]["verdict"] == "revise"

    cleared = asyncio.run(call("update_campaign", {"campaign_id": v2, "supersedes": ""}))
    assert cleared["supersedes"] is None
    assert "earlier_judgment" not in cleared


def test_the_upload_path_cannot_create_a_link_to_nothing(conn):
    """It had no validation at all, so a typo left {"", "  ", "camp_nope"} in the set of
    superseded ids and a link pointing at nothing — every state `update_campaign` refuses,
    reachable from the primary declaration path."""
    with pytest.raises(ValueError) as e:
        core.ingest_campaign(conn, title="Colombia v2", detail="Seeds two.",
                             supersedes="camp_nope")
    assert "camp_nope" in str(e.value)
    assert not store.get_superseded_campaign_ids(conn)


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_supersedes_is_no_supersession_rather_than_an_error(conn, blank):
    """Whitespace is not a typo for an id, it is an empty field arriving from a form. It used
    to be STORED, so `get_superseded_campaign_ids` held `"  "` and the record claimed to
    replace something unnameable."""
    v2 = core.ingest_campaign(conn, title="Colombia v2", detail="Seeds two.",
                              supersedes=blank)["campaign_id"]
    assert store.get_campaign(conn, v2)["supersedes"] is None
    assert not store.get_superseded_campaign_ids(conn)


def test_a_chain_diff_is_not_reported_as_an_unlinked_pair(conn, peru):
    """The order check read the two `supersedes` columns and the chain walk read the whole
    chain, so `diff(v1, v3)` warned "no supersedes link" while using the link, and
    `diff(v3, v1)` was answered backwards with the chain never walked."""
    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1,
                              detail="Seeds two.")["campaign_id"]
    v3 = core.ingest_campaign(conn, title="Colombia v3", supersedes=v2,
                              detail="Seeds one.")["campaign_id"]
    for cid, title in ((v1, "Colombia v1"), (v2, "Colombia v2"), (v3, "Colombia v3")):
        _judge(conn, cid, peru, subject_title=title)

    forward = core.diff_campaigns(conn, earlier=v1, later=v3)
    assert forward["order_basis"] == "supersession"
    assert not [w for w in forward["warnings"] if w["code"] == "version_order_unverified"]

    backward = core.diff_campaigns(conn, earlier=v3, later=v1)
    assert backward["arguments_reordered"] is True
    assert backward["through"] == [v1, v2, v3], "the chain is walked either way round"


def test_a_correction_taken_two_versions_ago_says_so(conn, peru):
    """Across a chain, "adopted" means confirmed ONCE and not re-examined since — v3's
    reviewer compared against v2 and never looked at this finding again. Between adjacent
    versions it means the later reviewer confirmed it. Reporting both the same way states the
    weaker claim with the stronger one's authority."""
    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    first = _judge(conn, v1, peru)
    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1,
                              detail="Seeds one.")["campaign_id"]
    _judge(conn, v2, peru, subject_title="Colombia v2", verdict="approve", summary="Settled.",
           findings=[], resolved=[{"finding_id": first["findings"][0]["id"],
                                   "was": "Seeds four", "now": "Seeds one"}])
    v3 = core.ingest_campaign(conn, title="Colombia v3", supersedes=v2,
                              detail="Seeds one, dated.")["campaign_id"]
    _judge(conn, v3, peru, subject_title="Colombia v3", verdict="approve", summary="Fine.",
           findings=[])

    adopted = core.diff_campaigns(conn, earlier=v1, later=v3)["adopted"][0]
    assert adopted["resolved_in"] == v2
    assert "not re-examined" in adopted["caveat"]

    # Adjacent, and there is nothing to caveat: the later reviewer is the one who said so.
    direct = core.diff_campaigns(conn, earlier=v1, later=v2)["adopted"][0]
    assert "caveat" not in direct


def test_the_quiz_is_capped(conn, peru):
    """A twelve-finding judgment turned filing a document into an exam. They are shown so the
    user can see what was said; they get settled one by one when this version is evaluated,
    where each is recorded by id rather than as free text."""
    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    _judge(conn, v1, peru, findings=[
        {"severity": "should_fix", "fix": "Change it", "kind": "missing_information",
         "finding": f"Gap number {n}"} for n in range(9)])
    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1, detail="Seeds two.")

    loop = v2["earlier_judgment"]
    assert len(loop["open_findings"]) == core._MAX_EARLIER_FINDINGS
    assert loop["open_findings_total"] == 9, "say how many there are, do not just truncate"


def test_a_version_in_the_middle_of_a_chain_is_not_offered_a_second_parent(conn, peru):
    """The guard reads BOTH endpoints. Reading only the later one left an earlier record that
    already replaces something free to be re-parented, which is the fan-in case from the
    other side."""
    v0 = core.ingest_campaign(conn, title="Colombia v0", detail="Seeds eight.")["campaign_id"]
    v1 = core.ingest_campaign(conn, title="Colombia v1", supersedes=v0,
                              detail="Seeds four.")["campaign_id"]
    first = _judge(conn, v1, peru)
    other = core.ingest_campaign(conn, title="Colombia alt", detail="Seeds two.")["campaign_id"]
    _judge(conn, other, peru, subject_title="Colombia alt", verdict="approve",
           summary="Settled.", findings=[],
           resolved=[{"finding_id": first["findings"][0]["id"],
                      "was": "Seeds four", "now": "Seeds two"}])

    diff = core.diff_campaigns(conn, earlier=v1, later=other)
    assert not [a for a in diff["next_actions"] if a["tool"] == "update_campaign"]


def test_the_earlier_versions_own_corrections_are_not_read_as_this_ones(conn, peru):
    """The chain is walked from the version AFTER the earlier endpoint. v1's `resolved` rows
    name v0's findings, not v1's — folding them in would report v1's own findings as adopted
    on the strength of a row about a different judgment."""
    v0 = core.ingest_campaign(conn, title="Colombia v0", detail="Seeds eight.")["campaign_id"]
    zeroth = _judge(conn, v0, peru, subject_title="Colombia v0")
    v1 = core.ingest_campaign(conn, title="Colombia v1", supersedes=v0,
                              detail="Seeds four.")["campaign_id"]
    # v1's judgment resolves v0's finding AND raises one of its own, which nobody has closed.
    _judge(conn, v1, peru, subject_title="Colombia v1",
           resolved=[{"finding_id": zeroth["findings"][0]["id"],
                      "was": "Seeds eight", "now": "Seeds four"}])
    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1,
                              detail="Seeds four still.")["campaign_id"]
    _judge(conn, v2, peru, subject_title="Colombia v2", verdict="approve",
           summary="Nothing recorded.", findings=[])

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)
    assert not diff["adopted"], "v1's own finding was never resolved by anybody"
    assert diff["counts"]["no_longer_raised"] == 1
