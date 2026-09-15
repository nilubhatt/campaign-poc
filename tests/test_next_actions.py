"""
Idea B: "Every result should offer its next step."

    "An evaluation finishes and nothing suggests what follows — store the brief as a record,
    mark it as superseding the version it revises, schedule the reconciliation for when
    results land. Those are the only three things anyone does next, and all three are
    currently things you have to know to ask for.

    Fix: return `next_actions` with each result: a short list of {label, tool,
    prefilled_args}. The surface renders them as suggestions; the user says yes."

Two things have to be true for this to be worth anything, and they are what most of these
tests are about:

  * The action has to WORK. A suggestion whose prefilled arguments are wrong, or whose tool
    rejects them, is worse than no suggestion — the user says yes and gets an error they did
    not cause and cannot fix.
  * The action has to be one the user would plausibly want NEXT. An offer that is always
    present is a menu, and a menu is the thing nobody reads.

This also collects two rows the tracker has been carrying: D16 (the `next_step` prose in
notices becomes structured) and D31 (teaching errors carry the retry as an action rather
than as a sentence).
"""
import inspect

import pytest

import actions
import config
import core
import mcp_server
import store


def _evaluation(**over):
    base = {
        "subject_title": "Colombia v2",
        "verdict": "revise",
        "approve_if": "Dates on every deliverable.",   # §6.5
        "summary": "Solid plan, but nothing is dated.",
        "findings": [{"severity": "blocking", "fix": "Change it", "kind": "missing_information",
                      "finding": "No posting dates"}],
    }
    base.update(over)
    if base.get("verdict") != "revise":
        base.pop("approve_if", None)
    return base


# ── the three the review named ──────────────────────────────────────────────

def test_an_evaluation_offers_to_store_the_brief_it_judged(conn):
    """The first of the reviewer's three. A judgment about a proposal that is not itself in
    the library is a judgment with nothing to compare against next time."""
    result = core.save_evaluation(conn, **_evaluation())

    offered = {a["tool"] for a in result["next_actions"]}
    assert "upload_campaign" in offered

    store_it = next(a for a in result["next_actions"] if a["tool"] == "upload_campaign")
    assert store_it["prefilled_args"]["title"] == "Colombia v2"
    assert store_it["label"]


def test_the_second_of_the_three_is_not_offered_on_this_evidence(conn):
    """The review named "mark it as superseding the version it revises" as one of the three,
    and it is NOT offered — see `actions.after_evaluation` for why, at length. In short: the
    only evidence available is a similarity score the model asserted, "replaces" is a claim
    about intent, the user hears only the label, and it cannot be undone.

    Both reviewers reached this independently, and the adversarial one showed it doing real
    harm over the protocol: accepting the offer hid a concluded campaign — the one with the
    metrics in it — from every search."""
    earlier = store.insert_campaign(conn, title="Colombia v1")
    result = core.save_evaluation(conn, **_evaluation(
        closest_precedent={"campaign_id": earlier, "similarity": 0.93}))

    assert not any("supersedes" in a["prefilled_args"] for a in result["next_actions"])


def test_the_third_is_offered_when_it_can_succeed_rather_than_when_it_is_thought_of(conn):
    """The review's third — "schedule the reconciliation for when results land". There is no
    scheduler, and offering it at save time meant the user could accept something that
    immediately failed for want of actuals. It is offered where the numbers arrive."""
    cid = store.insert_campaign(conn, title="Colombia", status="concluded")
    evaluation = core.save_evaluation(conn, campaign_id=cid, **_evaluation())

    recorded = core.add_metrics(conn, cid, detail="CTR 1.4%", confirm=True)

    reconcile = next(a for a in recorded["next_actions"]
                     if a["tool"] == "reconcile_evaluation")
    assert reconcile["prefilled_args"]["evaluation_id"] == evaluation["evaluation_id"]


def test_an_approved_evaluation_is_not_told_to_mark_a_supersession(conn):
    """An approval of a brand-new brief supersedes nothing. Offering it anyway is how a
    suggestion list becomes a menu."""
    result = core.save_evaluation(conn, **_evaluation(
        verdict="approve", findings=[],
        summary="Matches the structure that worked; dates and roster both clean."))

    assert not any(a["prefilled_args"].get("supersedes") for a in result["next_actions"])


# ── the offers have to work ─────────────────────────────────────────────────

def test_every_offered_action_names_a_tool_that_exists(conn):
    """A suggestion pointing at a tool nobody wrote is one the user says yes to and Claude
    cannot carry out."""
    tools = {t.name for t in mcp_server.mcp._tool_manager.list_tools()}

    for result in _every_result_with_actions(conn):
        for action in result["next_actions"]:
            assert action["tool"] in tools, f"{action['tool']} is not a tool"


def test_every_prefilled_argument_is_one_the_tool_accepts(conn):
    """The failure this is really about: an argument name that drifted. The user says yes,
    the call is rejected for an error they did not make and cannot correct."""
    for result in _every_result_with_actions(conn):
        for action in result["next_actions"]:
            signature = inspect.signature(getattr(mcp_server, action["tool"]))
            unknown = set(action["prefilled_args"]) - set(signature.parameters)
            assert not unknown, f"{action['tool']} has no parameter(s) {unknown}"


def test_an_offered_action_actually_runs(conn):
    """Executed rather than type-checked: the arguments are passed to the real function."""
    result = core.save_evaluation(conn, **_evaluation())
    store_it = next(a for a in result["next_actions"] if a["tool"] == "upload_campaign")

    created = core.ingest_campaign(conn, confirm=True, **store_it["prefilled_args"])

    assert created["campaign_id"]


def test_an_action_never_carries_an_argument_the_user_has_not_supplied(conn):
    """Prefilling a guess is how a suggestion quietly writes something nobody said. Only
    values this library already holds may be prefilled."""
    result = core.save_evaluation(conn, **_evaluation())

    for action in result["next_actions"]:
        for key, value in action["prefilled_args"].items():
            assert value not in ("", None), f"{action['tool']}.{key} is a placeholder"


# ── what the tracker has been carrying ──────────────────────────────────────

def test_a_warning_carries_its_next_step_as_an_action(conn, monkeypatch):
    """Tracker D16. `next_step` was prose for Claude to read and act on — which is the shape
    of the thing this item replaces, one layer down."""
    monkeypatch.setattr(config, "TOOL_TIME_BUDGET_SECONDS", 0.0)
    deck = "\n\n".join(f"section {i} " + "word " * 200 for i in range(4))

    result = core.ingest_campaign(conn, title="Cut short", deck_text=deck, confirm=True)

    warning = next(w for w in result["warnings"] if w["code"] == "indexing_incomplete")
    action = warning["next_actions"][0]
    assert action["tool"] == "finish_indexing"
    assert action["prefilled_args"]["campaign_id"] == result["campaign_id"]


def test_a_teaching_error_carries_the_retry_as_an_action(conn):
    """Tracker D31. "Did you mean 'predicted'?" is a sentence somebody has to parse; the
    same thing as an action is one the user can accept."""
    import enums

    with pytest.raises(ValueError) as exc:
        enums.normalise("predicated", field="metric_type", valid=store.VALID_METRIC_TYPES,
                        synonyms=enums.METRIC_TYPE_SYNONYMS)

    assert getattr(exc.value, "suggestion", None) == "predicted"
    assert list(getattr(exc.value, "valid", ())) == list(store.VALID_METRIC_TYPES)


def test_the_structured_error_survives_the_tool_boundary(conn):
    """It is only worth structuring if it reaches the caller structured."""
    cid = store.insert_campaign(conn, title="Colombia")

    result = mcp_server.add_metrics(cid, detail="CTR 2%", metric_type="predicated")

    assert result["suggestion"] == "predicted"
    assert result["valid"] == list(store.VALID_METRIC_TYPES)
    assert result["field"] == "metric_type"


# ── restraint ───────────────────────────────────────────────────────────────

def test_a_result_with_nothing_obvious_to_do_next_offers_nothing(conn):
    """An empty list is a real answer. A suggestion that is always there stops being read,
    and then the one that mattered is not read either."""
    store.insert_campaign(conn, title="Colombia", detail="a concluded campaign")

    result = core.find_similar_with_context(conn, text="colombia")

    assert result.get("next_actions", []) == [] or all(
        a["tool"] != "upload_campaign" for a in result["next_actions"])


def test_the_list_is_short_enough_to_read(conn):
    """"a short list" — three is the number the review named, and a list long enough to
    scan is a list nobody scans."""
    for result in _every_result_with_actions(conn):
        assert len(result["next_actions"]) <= 3, result["next_actions"]


def test_no_action_suggests_the_call_that_just_happened(conn):
    """Offering the thing the user just did is the shape of a loop."""
    result = core.save_evaluation(conn, **_evaluation())

    assert all(a["tool"] != "save_evaluation" for a in result["next_actions"])


# ── helper ──────────────────────────────────────────────────────────────────

def _every_result_with_actions(conn):
    """Each response in the product that carries next_actions, built for real."""
    earlier = store.insert_campaign(conn, title="Colombia v1", detail="the earlier version")
    yield core.save_evaluation(conn, **_evaluation(
        closest_precedent={"campaign_id": earlier, "similarity": 0.9}))
    yield core.ingest_campaign(conn, title="Colombia v2", detail="a brief", confirm=True)
    yield core.find_similar_with_context(conn, text="colombia")


# ══ design review of 5.2 ═════════════════════════════════════════════════════

def test_a_rejected_proposal_never_offers_to_replace_anything(conn):
    """The gate was inverted. `verdict != "approve"` suppressed supersession on approvals
    and PERMITTED it on rejects — so rejecting a proposal offered to file it as the
    replacement of the concluded campaign it merely resembled, and that concluded record,
    the one with the metrics in it, would vanish from every search."""
    earlier = store.insert_campaign(conn, title="Colombia v1", status="concluded")

    result = core.save_evaluation(conn, subject_title="Colombia v2", verdict="reject",
                                  summary="Not viable as written.",
                                  findings=[{"severity": "blocking", "fix": "Change it", "kind": "missing_information",
                                             "finding": "No dates"}],
                                  closest_precedent={"campaign_id": earlier,
                                                     "similarity": 0.93})

    assert not any(a["prefilled_args"].get("supersedes") for a in result["next_actions"])


def test_supersession_is_never_offered_on_a_similarity_score(conn):
    """"Mark this as replacing the version it revises" is a claim about intent. A similarity
    match is a claim about text, asserted by the model rather than computed (tracker D8), and
    the user only ever hears the label — "add this to the library" — while agreeing, unseen,
    to hide another record from every future search. It is not undoable either:
    `update_campaign` cannot clear `supersedes`."""
    earlier = store.insert_campaign(conn, title="Colombia v1")

    for verdict, findings in (("revise", [{"severity": "blocking", "fix": "Change it", "kind": "missing_information",
                                            "finding": "x"}]),
                              ("approve", []),
                              ("reject", [{"severity": "blocking", "fix": "Change it", "kind": "missing_information",
                                            "finding": "x"}])):
        result = core.save_evaluation(
            conn, subject_title="Colombia v2", verdict=verdict,
            summary="A summary long enough to be a real one.", findings=findings,
            # §6.5: required on a revise, forbidden on the other two.
            **({"approve_if": "It is fixed."} if verdict == "revise" else {}),
            closest_precedent={"campaign_id": earlier, "similarity": 0.97})
        assert not any("supersedes" in a["prefilled_args"]
                       for a in result["next_actions"]), verdict


def test_the_offer_for_a_record_already_in_the_library_is_callable(conn):
    """The branch the fixture never reached: with `campaign_id` set, the offer was
    `update_campaign(supersedes=...)` — an argument that tool does not take, so accepting it
    raised a TypeError. This is exactly the failure the item's first rule promises to
    prevent, and it shipped because no test passed a campaign_id."""
    cid = store.insert_campaign(conn, title="Colombia v2")
    earlier = store.insert_campaign(conn, title="Colombia v1")

    result = core.save_evaluation(
        conn, subject_title="Colombia v2", verdict="revise", approve_if="It is fixed.",
        summary="Nothing is dated.", campaign_id=cid,
        findings=[{"severity": "blocking", "fix": "Change it", "kind": "missing_information",
                                             "finding": "No dates"}],
        closest_precedent={"campaign_id": earlier, "similarity": 0.9})

    for action in result["next_actions"]:
        signature = inspect.signature(getattr(mcp_server, action["tool"]))
        assert not set(action["prefilled_args"]) - set(signature.parameters), action


def test_an_offer_is_not_made_before_it_can_possibly_succeed(conn):
    """"When the results land, reconcile this judgment against them" was offered the moment
    the judgment was saved. The user says yes, and the call fails: there are no actuals yet.
    An offer with a temporal precondition is a reminder, and this product has no reminder
    channel — so the offer belongs where the numbers actually arrive."""
    result = core.save_evaluation(conn, **_evaluation())

    assert all(a["tool"] != "reconcile_evaluation" for a in result["next_actions"])


def test_recording_results_offers_to_close_the_judgment_they_answer(conn):
    """Where that offer does belong. This is the moment the precondition is satisfied."""
    cid = store.insert_campaign(conn, title="Colombia", status="concluded")
    evaluation = core.save_evaluation(conn, campaign_id=cid, **_evaluation())

    result = core.add_metrics(conn, cid, detail="CTR 1.4%", confirm=True)

    offer = next(a for a in result["next_actions"] if a["tool"] == "reconcile_evaluation")
    assert offer["prefilled_args"]["evaluation_id"] == evaluation["evaluation_id"]


def test_results_on_a_campaign_with_no_judgment_offer_nothing_to_reconcile(conn):
    cid = store.insert_campaign(conn, title="Colombia", status="concluded")

    result = core.add_metrics(conn, cid, detail="CTR 1.4%", confirm=True)

    assert all(a["tool"] != "reconcile_evaluation" for a in result.get("next_actions", []))


def test_every_action_says_whether_it_needs_permission(conn):
    """Three rules were in play for one action. `next_step` said "act on it; never read it
    out"; `next_actions` said "call the tool only if the user accepts"; and
    `finish_indexing`'s own docstring says "do not stop to ask". Writing a record and
    resuming an interrupted index are not the same kind of act."""
    for result in _every_result_with_actions(conn):
        for action in result["next_actions"]:
            assert action["consent"] in ("ask", "do"), action


def test_finishing_an_interrupted_index_does_not_need_permission(conn, monkeypatch):
    monkeypatch.setattr(config, "TOOL_TIME_BUDGET_SECONDS", 0.0)
    deck = "\n\n".join(f"section {i} " + "word " * 200 for i in range(4))

    result = core.ingest_campaign(conn, title="Cut short", deck_text=deck, confirm=True)

    warning = next(w for w in result["warnings"] if w["code"] == "indexing_incomplete")
    assert warning["next_actions"][0]["consent"] == "do"


def test_writing_a_new_record_does_need_permission(conn):
    result = core.save_evaluation(conn, **_evaluation())

    store_it = next(a for a in result["next_actions"] if a["tool"] == "upload_campaign")
    assert store_it["consent"] == "ask"


def test_an_offer_says_why_it_is_being_made(conn):
    """A label alone gives Claude nothing to say about WHY, and the reason is the library
    fact that triggered it — which is the part that makes an offer persuasive or refusable."""
    cid = store.insert_campaign(conn, title="Colombia", status="concluded")

    result = core.ingest_campaign(conn, title="Colombia 2", detail="d", status="concluded",
                                  confirm=True)

    offer = next(a for a in result["next_actions"] if a["tool"] == "add_metrics")
    assert offer["why"], "no reason given"


# ══ adversarial review of 5.2 ════════════════════════════════════════════════

def test_a_metrics_row_with_nothing_in_it_is_refused(conn):
    """The offer's arguments are `campaign_id` alone, so accepting it with confirm=True
    stored a content-free `actual` row — which then satisfied the gate that lets a
    performance tag be marked `verified`. One yes away from provenance this library is built
    to weigh, for a row that measures nothing."""
    cid = store.insert_campaign(conn, title="Colombia")

    with pytest.raises(ValueError) as exc:
        store.add_metrics(conn, cid, metric_type="actual")

    assert "detail" in str(exc.value) or "structured" in str(exc.value)


def test_an_offer_that_needs_more_from_the_user_says_so(conn):
    """"Accepting is one step, not a form" is only true when the prefilled arguments are
    enough. When they are not, the offer has to name what is still missing — otherwise
    Claude calls it with what it was given and stores an empty row."""
    cid = store.insert_campaign(conn, title="Colombia", status="concluded")

    result = core.ingest_campaign(conn, title="Colombia 2", detail="d", status="concluded",
                                  confirm=True)
    offer = next(a for a in result["next_actions"] if a["tool"] == "add_metrics")

    assert offer["needs"], "nothing says the user still has to supply the numbers"


def test_the_commonest_upload_of_all_gets_its_offer(conn):
    """`after_upload` was given the caller's `status` — None when omitted — while the store
    defaults it to `concluded`. So the most ordinary upload there is, a deck with no status
    stated, was the one path that offered nothing."""
    result = core.ingest_campaign(conn, title="Colombia", detail="a finished campaign",
                                  confirm=True)

    assert store.get_campaign(conn, result["campaign_id"])["status"] == "concluded"
    assert any(a["tool"] == "add_metrics" for a in result["next_actions"])


def test_storing_a_second_record_under_a_used_title_says_so(conn):
    """The offer is what creates the duplicate, and the product's own title lookup then
    breaks on it: `bulk_import_metrics` reports "title 'Colombia v2' is ambiguous"."""
    core.ingest_campaign(conn, title="Colombia v2", detail="the first one", confirm=True)

    result = core.ingest_campaign(conn, title="Colombia v2", detail="again", confirm=True)

    assert any(w["code"] == "duplicate_title" for w in result["warnings"]), result["warnings"]


def test_a_preview_offers_nothing_and_a_label_does_not_promise_a_write(conn):
    """Over the protocol the store offer previewed rather than stored, because
    `upload_campaign` needs confirm=True — while the label said "Add to the library". The
    preview is the consent step for a write, so the label has to describe that, not the
    write it leads to."""
    result = core.save_evaluation(conn, **_evaluation())

    store_it = next(a for a in result["next_actions"] if a["tool"] == "upload_campaign")
    assert "review" in store_it["label"].lower() or "check" in store_it["label"].lower()


def test_trimming_actually_trims():
    """Mutation-proof: making `trim` return everything left the whole suite green, because
    no caller has ever produced more than two offers."""
    many = [actions.action(f"offer {i}", "finish_indexing", campaign_id=f"c{i}")
            for i in range(6)]

    assert len(actions.trim(many)) == actions.MAX_ACTIONS


def test_an_exact_repeat_is_dropped_and_a_different_call_is_not():
    """The docstring used to say "de-duplicate by tool", which is not what the code does —
    and would be wrong if it were: two offers of the same tool with different arguments are
    two different offers."""
    same = actions.action("a", "finish_indexing", campaign_id="c1")
    repeat = actions.action("b", "finish_indexing", campaign_id="c1")
    other = actions.action("c", "finish_indexing", campaign_id="c2")

    assert len(actions.trim([same, repeat, other])) == 2


def test_two_warnings_with_different_actions_do_not_fold_into_one():
    """`collapse` keys on the user-facing text and ignored `next_actions`, so two warnings
    that differ only in which campaign they would repair became one — carrying the first
    one's action and a count of two."""
    import notices

    first = notices.notice("indexing_incomplete", detail="a", affects="same", 
                           next_actions=actions.to_finish_indexing("camp_1"))
    second = notices.notice("indexing_incomplete", detail="b", affects="same",
                            next_actions=actions.to_finish_indexing("camp_2"))

    assert len(notices.collapse([first, second])) == 2
