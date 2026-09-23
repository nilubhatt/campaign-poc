"""
§7.4: tool descriptions are the shared prompt.

The review's words: *"`upload_campaign`'s description carries a whole conversational
procedure. Extend that deliberately to the reasoning tools. Fix: put the evaluation procedure
in `prepare_evaluation`'s description: the order of operations, the scorecard's six criteria,
the requirement to cite and quote, the rule that guardrail breaches and precedent departures
are different classes, the instruction to run a disconfirming search, and the exact output
contract. Also set the MCP server-level `instructions` field, which most servers leave empty —
it is a system prompt the product ships and every client surfaces."*

The phase's subject is consistency across users, and a procedure only produces consistency if
every user gets the same one. So the load-bearing decision here is not what the procedure says
— it is that there is **one** of it. A copy in the server instructions and a copy in the tool
description is two procedures that agree today; this project has now watched a hand-maintained
copy drift four separate times (the Windows installer version, `TOOL_NAMES`, the plan's own
counts, and D75's three definitions of "measured"). One constant, referenced twice.

**Two of the review's six items cannot be shipped as written, and saying so is part of the
item.** The scorecard's six criteria belong to the customer's rulebook, which §12.1 has not
built — hard-coding Fabletics' rubric into a product that ships generic is the thing the
product-owner decision explicitly forbids. And "the instruction to run a disconfirming search"
changed hands: §6.4 decided the SERVER runs it, because a model asked to find evidence against
a verdict it has already reached is marking its own homework. The procedure says to read the
result, which is the instruction that now exists.
"""
import mcp_server


PROCEDURE = mcp_server.EVALUATION_PROCEDURE


# ── one procedure, not two that agree today ─────────────────────────────────

def test_the_server_instructions_and_the_tool_description_are_the_same_text():
    """Not "say the same thing" — literally the same object. Two copies that agree today is
    the hand-maintained copy this project has watched drift four times."""
    assert PROCEDURE in mcp_server.INSTRUCTIONS
    # The DESCRIPTION a client receives, not the Python docstring: the docstring is for
    # somebody reading the file, and what reaches a model is the registered description.
    assert PROCEDURE in mcp_server._PREPARE_EVALUATION_DESCRIPTION


def test_the_procedure_appears_once_in_the_whole_codebase():
    """A literal pasted twice would pass the test above and still drift. This one fails if
    anybody writes the sentences out a second time — anywhere, not just in the file that
    happens to hold the constant today. §7.5 moved it to `core` and a check scoped to one
    module would have said nothing."""
    import pathlib

    root = pathlib.Path(mcp_server.__file__).parent
    opening = PROCEDURE.strip().splitlines()[0].strip()
    written = {f.name: f.read_text().count(opening) for f in root.glob("*.py")}
    assert sum(written.values()) == 1, f"the procedure is written out more than once: {written}"


# ── what the review asked it to carry ───────────────────────────────────────

def test_it_gives_the_order_of_operations():
    lowered = PROCEDURE.lower()
    for step in ("prepare_evaluation", "save_evaluation"):
        assert step in lowered
    assert "1." in PROCEDURE and "2." in PROCEDURE, "an order, not a list of concerns"


def test_it_requires_a_citation_and_a_quote():
    lowered = PROCEDURE.lower()
    assert "quote" in lowered
    assert "cite" in lowered or "citing" in lowered


def test_it_separates_the_two_classes_of_finding():
    """§6.2's whole subject: a breach is a fact about a rule and is not debatable; a departure
    is a judgment about whether a difference matters, and it invites a rationale."""
    lowered = PROCEDURE.lower()
    assert "guardrail_breach" in lowered and "precedent_departure" in lowered
    assert "not debatable" in lowered or "non-negotiable" in lowered


def test_it_says_the_server_runs_the_disconfirming_search():
    """The review asks for "the instruction to run a disconfirming search". §6.4 decided the
    server runs it — a model looking for evidence against its own verdict is marking its own
    homework — so the instruction that exists is to READ it, and to read the code rather than
    the absence of rows."""
    lowered = PROCEDURE.lower()
    assert "disconfirming" in lowered
    assert "server" in lowered


def test_it_states_the_output_contract():
    lowered = PROCEDURE.lower()
    for field in ("verdict", "summary", "severity", "kind", "approve_if", "fix"):
        assert field in lowered, field


def test_it_states_the_weighting_rules():
    """"Concluded over proposed, verified over stated, quote or it is an opinion" — the
    review's own three."""
    lowered = PROCEDURE.lower()
    assert "concluded" in lowered
    assert "verified" in lowered and "stated" in lowered


def test_it_carries_the_layer_rule(conn=None):
    """D12. A commentary chunk is a retrieved chunk, so quoting one is legitimate — but
    quoting it as though the deck said it is a different and false statement, and the rule
    lived only on the browsing tool."""
    lowered = PROCEDURE.lower()
    assert "commentary" in lowered and "body" in lowered


def test_it_says_what_the_product_cannot_yet_do():
    """The scorecard's six criteria belong to the customer's rulebook. Hard-coding one
    customer's rubric into a product that ships generic is what the product-owner decision
    forbids — and a procedure that silently omits a step the review asked for reads as though
    the step is not needed.

    §12.2 moved the scorecard sentence OUT of this constant and into the per-brief contract,
    and that is the point rather than an exception to it: whether a scorecard is declared is a
    fact about one library, and a constant shipped to every customer cannot tell the declared
    case from the undeclared one. Saying "still outstanding" to somebody who has just written
    theirs is the same defect the other way round.

    What this constant still owes is the rulebook itself, and the honest statement of what an
    empty one means."""
    assert "rulebook" in PROCEDURE.lower()
    # Flattened: the procedure is wrapped for reading, and a phrase split across a line break
    # is still the sentence the model receives.
    flat = " ".join(PROCEDURE.lower().split())
    assert "ships empty" in flat
    assert "no rule was checked" in flat


# ── it has to survive the transport ─────────────────────────────────────────

def test_the_instructions_reach_a_client():
    """"It is a system prompt the product ships and every client surfaces." A field the
    server does not actually send is a prompt nobody reads."""
    assert mcp_server.mcp.instructions
    assert PROCEDURE in mcp_server.mcp.instructions


def test_the_description_reaches_a_client_over_the_protocol():
    import asyncio

    async def described():
        tools = await mcp_server.mcp.list_tools()
        return {t.name: (t.description or "") for t in tools}

    described = asyncio.run(described())
    assert PROCEDURE.strip().splitlines()[0].strip() in described["prepare_evaluation"]


def test_the_procedure_is_not_so_long_that_it_displaces_the_evidence():
    """The reason it is repeated with the evidence (§7.5) is that a session's context is
    contested, and a procedure the length of a manual competes with the thing it is meant to
    make readable.

    The ceiling is a budget decision and not a measurement — the review's own position is
    that "restating the rules alongside the data is cheap and is the highest-leverage prompt
    real estate the product has", so this is deliberately generous. What it exists to catch is
    the slow accretion that turns a procedure into a manual: every future addition has to
    displace something, which is the right conversation to be forced into."""
    assert len(PROCEDURE) < 4000, f"{len(PROCEDURE)} characters"


# ── D32: the vocabulary, glossed where the model reads it ───────────────────

def test_every_marketer_facing_enum_is_glossed_with_its_synonyms():
    """D32. §5.1 made the server forgiving about what a marketer types; the gloss is the
    other half — a model that knows "live" means `in_flight` sends the right value first time,
    and the forgiving layer goes back to being a safety net rather than the primary path.

    D37: the words are the PRODUCT's, so the gloss lost "in market" when `enums` did. One
    customer's phrasing in the product's own model-facing text is the same decision
    implemented twice, disagreeing — and this test asserted the copy rather than the source,
    so it held the removed word in place. It reads the synonym table now, which is what the
    gloss is a gloss OF."""
    import pathlib
    import re

    import enums

    source = pathlib.Path(mcp_server.__file__).read_text()
    for name, needed in (("Status", ("live", "running", "finished", "wrapped")),
                         ("MetricType", ("forecast", "measured")),
                         ("RecordType", ("guidelines", "placeholder")),
                         ("TagSource", ("impression", "actual"))):
        block = re.search(rf'{name} = _enum\([^)]*\)\n"""(.*?)"""', source, re.S)
        assert block, f"{name} carries no gloss"
        text = block.group(1).lower()
        for word in needed:
            assert word in text, f"{name} gloss does not mention {word!r}"

    # D37, the direction that actually bit: a word the PRODUCT no longer knows must not still
    # be taught by the gloss. `in_market` moved to the customer's rulebook and the gloss kept
    # it — one decision implemented twice, disagreeing, with the copy a model reads first
    # holding the removed word in place.
    status_gloss = re.search(r'Status = _enum\([^)]*\)\n"""(.*?)"""', source, re.S).group(1)
    for word in re.findall(r"[a-z_]+", status_gloss.lower()):
        shaped = word.replace(" ", "_")
        if shaped in ("proposed", "in_flight", "concluded"):
            continue
        if shaped in enums.STATUS_SYNONYMS or len(shaped) < 4:
            continue
        assert shaped not in ("in_market",), (
            f"the Status gloss teaches {word!r}, which this product no longer knows"
        )


def test_the_instructions_say_a_normalised_value_should_be_spoken():
    """The other half of D32. A silent rewrite is how somebody discovers months later that
    their word meant something else here."""
    lowered = mcp_server.INSTRUCTIONS.lower()
    assert "normalised" in lowered
    assert "say it" in lowered or "say the" in lowered
