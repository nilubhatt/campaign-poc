"""
§6.5 (idea K): state what would flip the verdict.

The review's words: *"Every revise should carry its own exit condition. Today the reader
infers it from the list of problems, which is not the same thing and is not checkable. Fix: a
required `approve_if` field — the specific, testable set of changes that converts this verdict
to approve. It doubles as the note the partner receives."*

§2.4 defined and bounded the field. What is left is D2: required on a revise, forbidden on an
approve, and a `fix` on every finding above a note.

Two things in the review's sentence are doing work beyond "make the field required".

**"Which is not the same thing."** A list of problems and an exit condition are different
statements. Three findings might need two changes, or one finding might need three. Inferring
the second from the first is what the reader does today and what the review says is wrong.

**"Testable", and "the note the partner receives".** A free-text sentence cannot be checked
for testability by a validator — but a CHECKLIST can be ticked off, and that is what the
partner actually needs. So the server composes one from the `fix` lines it already holds, and
`approve_if` is the model's one-line statement of the same thing. The field stays the
sentence; the checklist is the thing somebody can hold the next version against.

**And a reject cannot have one.** If there is a set of changes that converts this to approve,
the verdict is revise. That is what the two words mean, and it is the one part of "testable"
a validator can genuinely enforce.
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


def _finding(peru, **over):
    base = {"severity": "should_fix", "kind": "precedent_departure",
            "departure": "regression",
            "finding": "Seeds four colourways where Peru used one",
            "fix": "Seed one colourway per creator",
            "precedent": {"campaign_id": peru,
                          "quote": "Seeded one colourway per creator"}}
    base.update(over)
    return base


def _evaluation(peru, **over):
    base = dict(subject_title="Colombia v1", verdict="revise",
                summary="One thing to change.",
                approve_if="One colourway per creator, and a posting date on every asset.",
                findings=[_finding(peru)])
    base.update(over)
    return base


# ── required, forbidden, and the pair that means the same thing ─────────────

def test_a_revise_has_to_say_what_would_end_it(conn, peru):
    """"Today the reader infers it from the list of problems, which is not the same thing and
    is not checkable." Three findings might need two changes; one finding might need three."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation(peru, approve_if=None))
    assert "approve_if" in str(e.value)


def test_an_approve_cannot_carry_one(conn, peru):
    """There is nothing to exit. An `approve_if` on an approve is a reservation the verdict
    does not admit to — the hedge §2.4 spent an item removing, in a different field."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation(
            peru, verdict="approve", findings=[],
            approve_if="Actually, also fix the dates."))
    assert "approve" in str(e.value)


def test_a_reject_that_names_its_exit_is_a_revise(conn, peru):
    """The one part of "testable" a validator can genuinely enforce. If there is a set of
    changes that converts this to approve, then by definition the verdict is revise — and
    letting a reject carry one lets the harsher word be used with the softer meaning."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation(
            peru, verdict="reject", findings=[_finding(peru, severity="blocking")],
            approve_if="Seed one colourway and this is fine."))
    assert "revise" in str(e.value)


def test_a_reject_without_one_is_accepted(conn, peru):
    assert core.save_evaluation(conn, **_evaluation(
        peru, verdict="reject", approve_if=None,
        findings=[_finding(peru, severity="blocking")]))["evaluation_id"]


# ── every finding above a note says what to do about it ─────────────────────

@pytest.mark.parametrize("severity", ["blocking", "should_fix"])
def test_a_problem_worth_acting_on_says_what_the_action_is(conn, peru, severity):
    """A finding above a note is a claim that something must change. Without a `fix` the
    reader has the complaint and not the remedy, and the exit condition below cannot be
    composed from the findings at all."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation(
            peru, findings=[_finding(peru, severity=severity, fix=None)]))
    assert "fix" in str(e.value)


def test_a_note_needs_no_fix(conn, peru):
    """"Worth saying once; nobody has to act." Requiring a remedy would make it an action
    item, which is the severity above it."""
    assert core.save_evaluation(conn, **_evaluation(
        peru, verdict="approve", approve_if=None,
        findings=[_finding(peru, severity="note", departure="possible_improvement",
                           fix=None)]))["evaluation_id"]


# ── the note the partner receives ───────────────────────────────────────────

def test_the_exit_condition_comes_back_as_something_you_can_tick_off(conn, peru):
    """"Testable" and "the note the partner receives" are the same requirement: a sentence
    cannot be checked off and a list can. The server composes it from the `fix` lines it
    already holds, so it cannot drift from the findings."""
    result = core.save_evaluation(conn, **_evaluation(peru, findings=[
        _finding(peru, severity="blocking", finding="No posting dates",
                 kind="missing_information", departure=None,
                 fix="Add a posting date to every asset", precedent=None),
        _finding(peru)]))

    checklist = result["exit_checklist"]
    assert [item["fix"] for item in checklist] == [
        "Add a posting date to every asset", "Seed one colourway per creator"]
    assert [item["finding_id"] for item in checklist] == [
        f["id"] for f in store.get_evaluation(conn, result["evaluation_id"])["findings"]
        if f["severity"] in ("blocking", "should_fix")]
    assert result["approve_if"], "the sentence stays; the list is beside it, not instead"


def test_the_checklist_is_ordered_worst_first_like_everything_else(conn, peru):
    result = core.save_evaluation(conn, **_evaluation(peru, findings=[
        _finding(peru, severity="should_fix"),
        _finding(peru, severity="blocking", finding="No posting dates",
                 kind="missing_information", departure=None,
                 fix="Add a posting date to every asset", precedent=None)]))

    assert [item["severity"] for item in result["exit_checklist"]] == [
        "blocking", "should_fix"]


def test_an_approve_has_no_checklist_rather_than_an_empty_one(conn, peru):
    """An empty list beside an approve reads as "nothing left to do, we checked" when the
    truth is that there was never a list. Absent says that; empty does not."""
    result = core.save_evaluation(conn, **_evaluation(
        peru, verdict="approve", approve_if=None, findings=[]))
    assert "exit_checklist" not in result


# ── it survives to be held against the next version ─────────────────────────

def test_the_exit_condition_reaches_the_version_that_is_supposed_to_meet_it(conn, peru):
    """"The specific, testable set of changes that converts this verdict to approve" is worth
    nothing if nobody sees it when the next version arrives. §6.3 built that moment."""
    v1 = core.ingest_campaign(conn, title="Colombia v1", detail="Seeds four.")["campaign_id"]
    core.save_evaluation(conn, campaign_id=v1, **_evaluation(peru))

    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1, detail="Seeds one.")

    assert v2["earlier_judgment"]["approve_if"] == (
        "One colourway per creator, and a posting date on every asset.")
    assert v2["earlier_judgment"]["exit_checklist"]


def test_the_exit_condition_is_on_the_read_path(conn, peru):
    saved = core.save_evaluation(conn, **_evaluation(peru))
    read_back = core.get_evaluation(conn, evaluation_id=saved["evaluation_id"])
    assert read_back["approve_if"]
    assert read_back["exit_checklist"]


def test_it_reaches_the_model_over_the_protocol(conn, peru):
    import asyncio
    import json

    async def call(args):
        import mcp_server
        result = await mcp_server.mcp.call_tool("save_evaluation", args)
        return json.loads(result.content[0].text)

    refused = asyncio.run(call(_evaluation(peru, approve_if=None)))
    assert "approve_if" in refused["error"]

    saved = asyncio.run(call(_evaluation(peru)))
    assert saved["exit_checklist"][0]["fix"] == "Seed one colourway per creator"
