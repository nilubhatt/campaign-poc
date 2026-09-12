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
        "summary": "Solid plan, but nothing is dated.",
        "findings": [{"severity": "blocking", "finding": "No posting dates"}],
    }
    base.update(over)
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


def test_an_evaluation_offers_to_mark_what_it_supersedes(conn):
    """The second. A v2 that does not supersede its v1 leaves both in the evidence, and the
    older one goes on being cited as though it were current."""
    earlier = store.insert_campaign(conn, title="Colombia v1")
    result = core.save_evaluation(conn, **_evaluation(
        closest_precedent={"campaign_id": earlier, "similarity": 0.93}))

    supersede = next(a for a in result["next_actions"]
                     if a["prefilled_args"].get("supersedes"))
    assert supersede["prefilled_args"]["supersedes"] == earlier


def test_an_evaluation_offers_the_reconciliation_that_closes_it(conn):
    """The third. Nothing schedules anything here, so the honest version is the call that
    closes the loop, prefilled, for when the numbers land."""
    result = core.save_evaluation(conn, **_evaluation())

    reconcile = next(a for a in result["next_actions"]
                     if a["tool"] == "reconcile_evaluation")
    assert reconcile["prefilled_args"]["evaluation_id"] == result["evaluation_id"]
    assert "result" in reconcile["label"].lower() or "land" in reconcile["label"].lower()


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
