"""
§6.2 (idea H): a guardrail breach and a departure from precedent are two classes, and only
one of them is debatable.

The review's words: *"'Uses AI-generated imagery' and 'does not match how Peru seeded'
currently come out of the same machinery as findings of the same kind. The first is binary
and non-negotiable; the second is a judgment that a partner may legitimately argue with. Fix:
two classes with different vocabulary. Guardrail breach cites the rulebook and is not
debatable. Departure from precedent cites a campaign and invites a rationale — the UAE brief's
claw machine was a departure that turned out to be better than the precedent."*

§2.4 defined `kind` and the rule that a breach cannot be a note. §6.1 made a breach cite a
`rule_id` and a departure cite a `campaign_id`, and verified both. What is left is the part
the review actually asked for and that neither of those delivered: the two classes still read
the same, because the only thing separating them is a label a reader has to know how to
interpret.

Two things here.

**`kind` is required.** Optional, it was the cheapest way past every rule attached to it: a
finding with no kind needs no citation, cannot contradict its slot, and is exempt from
everything below. That is not a hypothetical — the 6.1 review found it and it went into the
tracker as the cheaper half of D78.

**A departure has to say which way it departs.** "Does not match how Peru seeded" is not a
finding until somebody says whether different is worse here. Forcing that choice is what makes
the class genuinely different from a breach: a breach is a fact about a rule, and a departure
is a judgment about whether a difference matters — which is precisely what a partner is
entitled to argue with. And it gives the claw machine somewhere to live: a departure the model
thinks may be an IMPROVEMENT is recorded as one rather than filed as a defect, which is what
the product did to the UAE brief.
"""
import pytest

import core
import store


@pytest.fixture
def library(conn):
    """A campaign to depart from and a rulebook to breach, both with real quotable text."""
    peru = core.ingest_campaign(
        conn, title="Peru launch", status="concluded",
        detail="Seeded one colourway per creator, with a content angle and posting date "
               "per asset.")["campaign_id"]
    rules = core.ingest_campaign(
        conn, title="Brand guidelines", record_type="reference",
        detail="No AI-generated imagery in any paid placement.")["campaign_id"]
    return {"peru": peru, "rules": rules}


def _breach(**over):
    base = {"severity": "blocking", "kind": "guardrail_breach",
            "finding": "Uses AI-generated imagery",
            "precedent": {"rule_id": None, "quote": "No AI-generated imagery"}}
    base.update(over)
    return base


def _departure(**over):
    base = {"severity": "should_fix", "kind": "precedent_departure",
            "departure": "unexplained",
            "finding": "Seeds four colourways where Peru used one",
            "precedent": {"campaign_id": None, "quote": "Seeded one colourway per creator"}}
    base.update(over)
    return base


def _evaluation(*findings, **over):
    base = {"subject_title": "Colombia v2", "verdict": "revise",
            "summary": "Two things to settle.", "findings": list(findings)}
    base.update(over)
    return base


def _wire(finding, library):
    precedent = finding.get("precedent") or {}
    if "rule_id" in precedent:
        precedent["rule_id"] = library["rules"]
    if "campaign_id" in precedent:
        precedent["campaign_id"] = library["peru"]
    return finding


# ── kind is required ────────────────────────────────────────────────────────

def test_a_finding_has_to_say_which_kind_it_is(conn, library):
    """Optional, `kind` was the cheapest way past every rule attached to it: no citation
    required, no slot to contradict, no class to be voiced as. The 6.1 review found the
    escape and it went into the tracker as the cheaper half of D78."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation(
            {"severity": "blocking", "finding": "Something is wrong"}))
    message = str(e.value)
    assert "kind" in message
    for kind in ("guardrail_breach", "precedent_departure", "missing_information",
                 "internal_contradiction"):
        assert kind in message, "name the four, or the retry is a guess"


def test_the_requirement_reaches_the_model_over_the_protocol(conn, library):
    import asyncio
    import json

    async def call():
        import mcp_server
        result = await mcp_server.mcp.call_tool("save_evaluation", {
            "subject_title": "Colombia v2", "verdict": "revise", "summary": "Something.",
            "findings": [{"severity": "blocking", "finding": "Something is wrong"}]})
        return json.loads(result.content[0].text)

    refused = asyncio.run(call())
    assert "error" in refused, refused
    assert "kind" in refused["error"]


# ── a departure has to say which way it departs ─────────────────────────────

def test_a_departure_has_to_say_whether_different_is_worse(conn, library):
    """"Does not match how Peru seeded" is not a finding until somebody says whether
    different is worse here. A breach is a fact about a rule; a departure is a judgment about
    whether a difference matters, and that judgment is the thing a partner argues with."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation(
            _wire(_departure(departure=None), library)))
    assert "departure" in str(e.value)
    assert "possible_improvement" in str(e.value)


def test_only_a_departure_may_carry_one(conn, library):
    """A breach is not a matter of degree. Letting it carry "possible_improvement" would put
    the not-debatable class on the same axis as the arguable one."""
    with pytest.raises(ValueError):
        core.save_evaluation(conn, **_evaluation(
            _wire(_breach(departure="possible_improvement"), library)))


def test_an_unknown_departure_value_is_refused(conn, library):
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation(
            _wire(_departure(departure="fine, probably"), library)))
    assert "regression" in str(e.value)


@pytest.mark.parametrize("value", ["regression", "unexplained", "possible_improvement"])
def test_the_three_readings_of_a_difference(conn, library, value):
    severity = "note" if value == "possible_improvement" else "should_fix"
    result = core.save_evaluation(conn, **_evaluation(
        _wire(_departure(departure=value, severity=severity), library),
        # A note alone cannot carry a revise, so give it something that can.
        _wire(_breach(), library)))

    stored = store.get_evaluation(conn, result["evaluation_id"])
    departures = [f for f in stored["findings"] if f["kind"] == "precedent_departure"]
    assert departures[0]["departure"] == value


def test_something_that_may_be_better_is_not_something_to_fix(conn, library):
    """The mirror of "a guardrail breach cannot be a note". If the model thinks the
    difference may be an improvement, asking the marketer to change it back contradicts the
    finding's own reading — and that is exactly what this product did to the UAE brief's
    claw machine, which was a departure that turned out better than the precedent."""
    for severity in ("blocking", "should_fix"):
        with pytest.raises(ValueError) as e:
            core.save_evaluation(conn, **_evaluation(
                _wire(_departure(departure="possible_improvement", severity=severity),
                      library)))
        assert "improvement" in str(e.value).lower()

    assert core.save_evaluation(conn, **_evaluation(
        _wire(_departure(departure="possible_improvement", severity="note"), library),
        _wire(_breach(), library)))["evaluation_id"]


# ── the two classes read differently ────────────────────────────────────────

def test_the_response_separates_what_is_settled_from_what_is_arguable(conn, library):
    """The review's actual complaint: the two come out of the same machinery. Counts by
    severity say how much each matters and nothing about whether it is debatable, so a
    marketer reading "2 blocking" cannot tell a rule they broke from a preference they may
    have been right to depart from."""
    result = core.save_evaluation(conn, **_evaluation(
        _wire(_breach(), library),
        _wire(_departure(), library),
        {"severity": "should_fix", "kind": "missing_information",
         "finding": "No end date on the flighting table"}))

    assert result["by_class"] == {"not_debatable": 1, "debatable": 1, "about_the_brief": 1}


def test_the_note_tells_the_model_to_voice_them_differently(conn, library):
    """A class distinction nothing says out loud is a column in a database. The response is
    where the model learns how to put it, and it must not say the same thing for a library
    that has only breaches as for one that has only departures."""
    breach_only = core.save_evaluation(conn, **_evaluation(_wire(_breach(), library)))
    departure_only = core.save_evaluation(conn, **_evaluation(
        _wire(_departure(), library)))

    assert breach_only["note"] != departure_only["note"]
    assert "a rule they wrote was broken" in breach_only["note"]
    assert "a rule they wrote was broken" not in departure_only["note"]
    # A departure invites a rationale rather than a correction.
    assert "a difference can be right" in departure_only["note"]
    assert "ask whether the difference is deliberate" in departure_only["note"]


def test_each_reading_of_a_difference_is_voiced_as_what_it_is(conn, library):
    """Keyed on the CLASS, a blocking regression — the finding that carries a `revise` when
    there is no breach — was told to "ask whether it is deliberate, do not tell them to
    change it back", in the same response as its own `fix` line. The three readings exist so
    the voicing can differ; two of the three were sharing one voice."""
    worse = core.save_evaluation(conn, **_evaluation(
        _wire(_departure(departure="regression", severity="blocking",
                         fix="Seed one colourway"), library)))
    asked = core.save_evaluation(conn, **_evaluation(
        _wire(_departure(departure="unexplained"), library)))

    assert "not a question" in worse["note"]
    assert "do not tell them to change it back" not in worse["note"].lower()
    assert "ask whether the difference is deliberate" in asked["note"]
    assert "not a question" not in asked["note"]


def test_the_response_carries_which_way_each_departure_departs(conn, library):
    """Without it the model cannot tell "this is worse" from "this may be deliberate" without
    a second round trip — the round trip this fixed response shape exists to avoid."""
    result = core.save_evaluation(conn, **_evaluation(
        _wire(_departure(departure="regression", severity="blocking"), library)))

    assert result["findings"][0]["departure"] == "regression"


def test_the_good_news_is_in_the_response_that_tells_the_model_to_say_it(conn, library):
    """The rule making a possible improvement a `note` also dropped it out of `findings`,
    which keeps only blocking and should_fix — so the response told the model to deliver good
    news whose text it did not have, and the claw machine was the finding least likely to
    reach the marketer."""
    result = core.save_evaluation(conn, **_evaluation(
        _wire(_departure(departure="possible_improvement", severity="note",
                         finding="Adds a claw machine at the grand opening"), library),
        _wire(_breach(), library)))

    assert [f["finding"] for f in result["improvements"]] == [
        "Adds a claw machine at the grand opening"]
    assert "improvements" in result["note"]


def test_a_question_does_not_on_its_own_stop_a_brief(conn, library):
    """The mirror of the improvement rule, which was one-sided. `unexplained` means "the
    brief does not say whether this is deliberate"; `blocking` means "this alone means the
    brief cannot proceed". A question that stops a brief is the same contradiction."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation(
            _wire(_departure(departure="unexplained", severity="blocking"), library)))
    assert "question" in str(e.value)

    assert core.save_evaluation(conn, **_evaluation(
        _wire(_departure(departure="regression", severity="blocking"), library)))


def test_an_improvement_cannot_smuggle_the_reversal_into_the_fix(conn, library):
    """The severity rule refuses "change it back". A `fix` is an instruction to change
    something, so the same reversal arrives in the other field."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation(
            _wire(_departure(departure="possible_improvement", severity="note",
                             fix="Change it back to one colourway"), library)))
    assert "fix" in str(e.value)


def test_the_tailored_refusals_say_what_each_value_means(conn, library):
    """Deleting either `is None` block left the whole suite green, because the membership
    check below it also refuses and also lists the values. What the tailored message adds is
    what each one MEANS — which is the stated reason the check lives in core rather than in
    the type, and it was the part nothing tested."""
    with pytest.raises(ValueError) as no_kind:
        core.save_evaluation(conn, **_evaluation(
            {"severity": "blocking", "finding": "Something is wrong"}))
    assert "a rule was broken" in str(no_kind.value)
    assert "the brief does not say" in str(no_kind.value)

    with pytest.raises(ValueError) as no_departure:
        core.save_evaluation(conn, **_evaluation(
            _wire(_departure(departure=None), library)))
    assert "the difference is worse" in str(no_departure.value)
    assert "it may be better than the precedent" in str(no_departure.value)


def test_every_kind_has_a_class(conn):
    """`_CLASSES[kind]` is looked up AFTER the row is inserted, so a kind added to `_KINDS`
    without a class would persist the evaluation and then raise on the way out — the caller
    sees a crash for a write that succeeded."""
    assert set(core._KINDS) == set(core._CLASSES)


def test_a_judgment_of_departures_alone_does_not_read_as_a_rule_breach(conn, library):
    """The two classes in one evaluation must stay distinguishable in the default response,
    which is the only thing most readers see."""
    result = core.save_evaluation(conn, **_evaluation(
        _wire(_departure(), library),
        _wire(_departure(departure="regression",
                         finding="Runs four weeks where Peru ran six"), library)))

    assert result["by_class"] == {"debatable": 2}
    # The word "rule" is allowed — the departure guidance uses it to say these are NOT rule
    # breaches. What must not appear is the sentence that states one as a fact.
    assert "a rule they wrote was broken" not in result["note"]
    assert "not rule breaches" in result["note"].lower()


def test_the_tool_says_the_two_classes_differently(conn):
    """The description IS the shared prompt (§7.4). If it describes the two classes in the
    same register the model writes them in the same register, whatever the schema says."""
    import mcp_server

    doc = mcp_server.save_evaluation.__doc__
    assert "not debatable" in doc.lower() or "non-negotiable" in doc.lower()
    assert "debatable by design" in doc.lower(), "and the other one said to be so"
    for value in ("regression", "unexplained", "possible_improvement"):
        assert value in doc, "the three readings, or the model has to guess the vocabulary"
    # The claw machine is the review's own example of a departure that beat the precedent,
    # and an abstract rule without a worked case is a rule nobody applies.
    assert "claw machine" in doc.lower()
    assert "better than the precedent" in doc.lower()
    # The registers have to be named, not just the classes.
    assert "state it" in doc.lower()
    assert "ask, do not instruct" in doc.lower()


def test_the_worked_examples_in_the_description_can_actually_be_saved(conn, library):
    """The docstring is the shared prompt (§7.4), and its one worked departure was exactly
    the shape the server refuses — no `departure`, plus a `fix` telling the marketer to change
    back something the same docstring says to ask about four paragraphs above. 6.1's review
    found the same class of defect in the same docstring, so this test checks the examples
    RUN rather than checking that the right words appear in them."""
    import ast
    import mcp_server

    doc = mcp_server.save_evaluation.__doc__
    # Brace-balanced rather than a regex: the examples are multi-line dict literals whose
    # last key varies, and a regex that half-matches would let a broken example through as
    # "no examples found".
    examples = []
    for start in (i for i in range(len(doc)) if doc.startswith('{"severity"', i)):
        depth, end = 0, None
        for i in range(start, len(doc)):
            depth += (doc[i] == "{") - (doc[i] == "}")
            if depth == 0:
                end = i + 1
                break
        assert end, "unbalanced braces in a worked example"
        flat = " ".join(doc[start:end].split())
        flat = flat.replace('"<a campaign_id from your evidence>"', repr(library["peru"]))
        flat = flat.replace('"<its own words, copied \u2014 not written from memory>"',
                            repr("Seeded one colourway per creator"))
        examples.append(ast.literal_eval(flat))
    assert len(examples) >= 2, f"the docstring should carry worked findings, found {examples}"

    for example in examples:
        result = core.save_evaluation(conn, subject_title="Colombia v2", verdict="revise",
                                      summary="From the worked example.",
                                      findings=[example])
        assert result["evaluation_id"], example


def test_the_procedure_note_carries_the_departure_rule_too(conn, library):
    """6.1's own conclusion: a rule a model only meets as a rejection afterwards costs a
    retry every time, so it belongs in `prepare_evaluation`'s note as well as in
    `save_evaluation`'s description."""
    note = core.prepare_evaluation(conn, subject_title="Colombia v2",
                                   proposal_text="A launch in Colombia.")["note"]
    assert "kind` is required" in note
    assert "departure" in note
    for value in ("regression", "unexplained", "possible_improvement"):
        assert value in note


def test_a_judgment_read_back_later_is_voiced_the_way_it_was_written(conn, library):
    """`save_evaluation` translates the class vocabulary into something a marketer can hear.
    Reading the same judgment back a week later returned the raw words with nothing to say
    how to put them — the "worked only inside the session that produced it" failure that
    §2.4 fixed for the findings themselves."""
    saved = core.save_evaluation(conn, **_evaluation(
        _wire(_breach(), library),
        _wire(_departure(departure="possible_improvement", severity="note",
                         finding="Adds a claw machine at the grand opening"), library)))

    read_back = core.get_evaluation(conn, evaluation_id=saved["evaluation_id"])
    assert read_back["by_class"] == {"not_debatable": 1, "debatable": 1}
    assert "a rule they wrote was broken" in read_back["how_to_say_it"]
    assert [f["finding"] for f in read_back["improvements"]] == [
        "Adds a claw machine at the grand opening"]


def test_the_possible_improvements_can_be_asked_for_by_name(conn, library):
    """They are notes by rule, so "show me what we might have got right" was a filter by kind
    followed by a hand-sort."""
    saved = core.save_evaluation(conn, **_evaluation(
        _wire(_breach(), library),
        _wire(_departure(departure="possible_improvement", severity="note",
                         finding="Adds a claw machine at the grand opening"), library)))

    only = core.get_evaluation(conn, evaluation_id=saved["evaluation_id"],
                               departure="possible_improvement")
    assert [f["finding"] for f in only["findings"]] == [
        "Adds a claw machine at the grand opening"]
    # The class summary counts the whole judgment, not the filtered view: a reader who asked
    # for one slice should not be told the rest does not exist.
    assert only["by_class"] == {"not_debatable": 1, "debatable": 1}


def test_a_reader_who_changed_their_mind_is_not_reported_as_ignored_advice(conn, library):
    """A finding raised in both versions reads as a correction not taken. But a departure the
    first review called a `regression` and the second called a `possible_improvement` is the
    library changing its mind — and it is this item's headline case, the claw machine.
    Reported without it, the comparison files exactly that story as a repeat defect."""
    v1 = core.ingest_campaign(conn, title="Colombia v1",
                              detail="Seeds four colourways.")["campaign_id"]
    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1,
                              detail="Seeds four colourways.")["campaign_id"]
    first = core.save_evaluation(conn, campaign_id=v1, **_evaluation(
        _wire(_departure(departure="regression",
                         finding="Seeds four colourways where Peru used one"), library)))
    finding_id = first["findings"][0]["id"]
    core.save_evaluation(conn, campaign_id=v2, **_evaluation(
        _wire(_departure(departure="possible_improvement", severity="note",
                         repeats=finding_id,
                         finding="Seeds four colourways where Peru used one"), library),
        _wire(_breach(), library)))

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)
    again = diff["raised_again"][0]
    assert again["departure"] == "regression"
    assert again["departure_now"] == "possible_improvement"
    assert again["reread"] == "softened"


def test_a_judgment_stored_before_this_rule_still_reads_back(conn, library):
    """`kind` is required for new writes and cannot be for old ones — the findings are a JSON
    blob, so every judgment saved before today holds findings this validator would refuse.
    2.4 and 2.5 both established that the pre-change database gets named and tested rather
    than assumed; 6.2 did neither until this."""
    store.insert_evaluation(
        conn, evaluation_id="eval_before", subject_title="Colombia v0", verdict="revise",
        summary="Written before kind was required.",
        findings=[{"id": "eval_before#1", "severity": "blocking",
                   "finding": "No posting dates", "kind": None, "departure": None}],
        resolved=[], closest_precedent=None, approve_if=None, evidence=None,
        provenance=None, campaign_id=None, cited_ids=[], predictions=None)

    read_back = core.get_evaluation(conn, evaluation_id="eval_before")
    assert read_back["verdict"] == "revise"
    assert [f["finding"] for f in read_back["findings"]] == ["No posting dates"]
    # No class, so no voicing note rather than a wrong one — a kind-less finding cannot be
    # said to be arguable or not.
    assert read_back["by_class"] == {}
    assert "how_to_say_it" not in read_back
    assert read_back["improvements"] == []


@pytest.mark.parametrize("kind", ["missing_information", "internal_contradiction"])
def test_a_claim_about_the_brief_is_neither_settled_nor_arguable(conn, library, kind):
    """The third class is not a leftover. A gap in the brief is not a rule somebody broke and
    not a difference from another record — it is a thing to fill in, and telling the marketer
    to defend it or to argue about it are both wrong. Nothing pinned `internal_contradiction`
    to it, so mapping it to `debatable` passed the whole suite."""
    result = core.save_evaluation(conn, **_evaluation(
        {"severity": "should_fix", "kind": kind,
         "finding": "The brief gives two different end dates"}))

    assert result["by_class"] == {"about_the_brief": 1}
    assert "gaps to fill, not arguments to have" in result["note"]
    assert "a rule they wrote was broken" not in result["note"]
    assert "a difference can be right" not in result["note"]
