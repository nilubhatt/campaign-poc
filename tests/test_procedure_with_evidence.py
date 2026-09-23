"""
§7.5: ship the procedure with the evidence, not just at connect time.

The review's words: *"Tool descriptions load once, at the start of a session, and compete with
everything the user has done since. The moment that matters is when the model is holding the
evidence package and about to judge. Fix: `prepare_evaluation` already returns a note telling
the model what to do next. Make it carry the full contract: the rulebook at its pinned
version, the computed facts, the required output schema, and the weighting rules (concluded
over proposed, verified over stated, quote or it is an opinion). Restating the rules alongside
the data is cheap and is the highest-leverage prompt real estate the product has."*

§7.4 put the procedure where it loads once. This is the other half, and the argument for it is
about WHEN rather than what: a rule read at connect time has been competing with an hour of
conversation by the moment it matters.

The design question is whether to restate the procedure or summarise it. Summarising would
produce a second, shorter procedure — and §7.4's whole point was that two versions of one
procedure is the drift this project has watched four times. So the note carries the same
constant, and what is added beside it is the part that cannot be static: THIS brief's computed
facts, and the rulebook version in force.

**The rulebook is the honest gap.** §12.1 has not built one, so "the rulebook at its pinned
version" is a version of nothing. The note says which version it is running without one rather
than omitting the line, because a procedure that quietly drops a step reads as though the step
was judged unnecessary.
"""
import core
import mcp_server


def _package(conn, **over):
    args = dict(subject_title="Colombia v1",
                proposal_text="A creator-led launch with four colourways, 40,000 USD budget.")
    args.update(over)
    return core.prepare_evaluation(conn, **args)


# ── the same procedure, not a second shorter one ────────────────────────────

def test_the_note_carries_the_whole_procedure(conn):
    """"Restating the rules alongside the data is cheap and is the highest-leverage prompt
    real estate the product has." A rule read at connect time has been competing with an hour
    of conversation by the moment it matters."""
    assert core.EVALUATION_PROCEDURE in _package(conn)["note"]


def test_it_is_the_same_constant_the_description_uses(conn):
    """Summarising it here would produce a second, shorter procedure — and §7.4's whole point
    was that two versions of one procedure is the drift this project has watched four times."""
    assert mcp_server.EVALUATION_PROCEDURE is core.EVALUATION_PROCEDURE
    assert core.EVALUATION_PROCEDURE in mcp_server._PREPARE_EVALUATION_DESCRIPTION


# ── and beside it, the part that cannot be static ───────────────────────────

def test_the_contract_names_this_briefs_own_computed_facts(conn):
    """The static procedure says "read `computed`". This says what `computed` actually found
    for the brief in front of the model, which is the difference between a rule and a fact."""
    note = _package(conn)["note"]

    assert "budget" in note.lower()
    # Its own line, not a rule the reader has to apply to a field they must go and find.
    assert "present" in note.lower() or "absent" in note.lower()


def test_a_contradicted_date_is_raised_in_the_note_itself(conn):
    """The sharpest of the computed facts, and the one most likely to be missed if it is only
    a key in a dict."""
    note = _package(
        conn, proposal_text="The weekend activation runs Saturday 3 March 2026.")["note"]
    assert "calendar" in note.lower() or "contradict" in note.lower()


def test_the_rulebook_version_is_stated_even_though_there_is_none(conn):
    """§12.1 has not built one, so "the rulebook at its pinned version" is a version of
    nothing. Saying which version it is running without one is not pedantry: a procedure that
    quietly drops a step reads as though the step was judged unnecessary, and a judgment made
    with no rulebook is a different judgment from one made with a rulebook that happened to
    say nothing."""
    note = _package(conn)["note"]
    assert "rulebook" in note.lower()
    assert core.rulebook_version() in note


def test_the_weighting_rules_survive_into_the_note(conn):
    """"Concluded over proposed, verified over stated, quote or it is an opinion" — the
    review's own three, and the ones a model is most likely to have drifted from by the time
    it is holding an evidence package."""
    note = _package(conn)["note"].lower()
    assert "concluded" in note
    assert "verified" in note and "stated" in note
    assert "quote" in note


# ── it must not bury the evidence it accompanies ────────────────────────────

def test_the_note_says_the_facts_before_it_restates_the_rules(conn):
    """A model reads a long field from the top. The part that is about THIS brief goes first;
    the procedure it has already been given at connect time goes last."""
    note = _package(conn)["note"]
    assert note.index("budget") < note.index(core.EVALUATION_PROCEDURE)


def test_it_reaches_the_model_over_the_protocol(conn):
    import asyncio
    import json

    async def call():
        result = await mcp_server.mcp.call_tool(
            "prepare_evaluation",
            {"subject_title": "Colombia v1",
             "proposal_text": "A launch. The weekend activation runs Saturday 3 March 2026."})
        return json.loads(result.content[0].text)

    note = asyncio.run(call())["note"]
    assert core.EVALUATION_PROCEDURE in note
    assert core.rulebook_version() in note
