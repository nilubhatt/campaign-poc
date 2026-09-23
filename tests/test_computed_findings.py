"""
§7.8: mark findings `computed` vs `judged`.

The premise, stated when §2.4 defined the field: *a computed finding is identical for every
user, so a difference there is a bug.* That only holds if the SERVER computed it — which is
why §2.4 made `basis: computed` unwritable from the MCP surface and left `judged` as the only
value a model could send.

Two things finish it.

**D5: the server has to actually produce some.** §7.1 built the mechanical checks and put them
in `computed` on the evidence package, where they are facts about the brief. A fact is not a
finding: "no date appears in this brief" becomes a finding when somebody says it is a problem,
and until now only the model could do that — which put a `judged` label on the most mechanical
half of the output and made §7.7's agreement figures measure a model's consistency at reading
a regex's result.

**D61: two values were not enough.** §5.4 matches findings across versions by text similarity
and had to stamp them `judged` with a caveat, because they are neither — a similarity score is
not a human judgment, and it is not identical for every user either in the sense that matters,
since it is a heuristic whose threshold somebody chose. The third value says what it is.
"""
import pytest

import core
import store


def _saved(conn, **over):
    args = dict(subject_title="Colombia v1", verdict="revise", summary="A stretch.",
                approve_if="It is fixed.",
                findings=[{"severity": "should_fix", "kind": "missing_information",
                           "finding": "No end date", "fix": "Add one"}])
    args.update(over)
    return core.save_evaluation(conn, **args)


# ── D5: the server produces findings of its own ─────────────────────────────

def test_the_server_raises_the_mechanical_findings_itself(conn):
    """"A computed finding is identical for every user, so a difference there is a bug" — a
    premise that says nothing while every finding is written by a model. §7.1 established the
    facts; this is what turns the ones that are problems into findings."""
    saved = _saved(conn, findings=[], verdict="approve", approve_if=None,
                   subject_text="A creator-led launch with four colourways.")

    computed = [f for f in store.get_evaluation(conn, saved["evaluation_id"])["findings"]
                if f["basis"] == "computed"]
    assert [f["finding"] for f in computed]
    assert all(f["kind"] == "missing_information" for f in computed)


def test_a_computed_finding_carries_the_evidence_it_was_read_from(conn):
    """The same rule §7.1 set for facts: a finding that cannot show what it looked at is an
    assertion, and the server's assertions carry more weight than a model's."""
    saved = _saved(conn, findings=[], verdict="approve", approve_if=None,
                   subject_text="A launch. Creators: @lima_style (180k followers).")

    computed = [f for f in store.get_evaluation(conn, saved["evaluation_id"])["findings"]
                if f["basis"] == "computed"]
    engagement = [f for f in computed if "engagement" in f["finding"].lower()]
    assert engagement
    assert engagement[0]["detail"], "it says what it read"


def test_a_brief_with_nothing_mechanically_wrong_gets_no_computed_findings(conn):
    """A check that always fires is a check nobody reads, and these arrive with the server's
    authority behind them."""
    saved = _saved(conn, findings=[], verdict="approve", approve_if=None,
                   subject_text="Runs 3 March 2026 to 28 March 2026. Budget is 40,000 USD. "
                                "Paid social on Meta ads, a press release, an email push, "
                                "billboards in Lima, in-store at the flagship, organic posts, "
                                "the website, and creators @lima (3.4% ER).")

    computed = [f for f in store.get_evaluation(conn, saved["evaluation_id"])["findings"]
                if f["basis"] == "computed"]
    assert computed == []


def test_a_computed_finding_does_not_change_the_verdict_the_model_gave(conn):
    """The server establishes facts; the verdict is the model's. A server finding that flipped
    an approve would be the product overruling the reasoner on the strength of a regex — and
    §6.4 already decided the server argues with a verdict rather than replacing it."""
    saved = _saved(conn, findings=[], verdict="approve", approve_if=None,
                   subject_text="A launch with no dates at all.")
    assert saved["verdict"] == "approve"


def test_the_model_still_cannot_claim_a_finding_was_computed(conn):
    """Unchanged from §2.4, and the reason the premise holds at all."""
    with pytest.raises(ValueError) as e:
        _saved(conn, findings=[{"severity": "should_fix", "kind": "missing_information",
                                "finding": "No end date", "fix": "Add one",
                                "basis": "computed"}])
    assert "server" in str(e.value).lower()


def test_a_computed_finding_is_never_counted_as_the_models_work(conn):
    """§7.7 measures finding recall against what the product raised. If the server's own
    findings were indistinguishable from the model's, the figure would improve every time a
    regex fired and nobody could tell why."""
    saved = _saved(conn, subject_text="A launch with no dates at all.")
    stored = store.get_evaluation(conn, saved["evaluation_id"])["findings"]

    assert {f["basis"] for f in stored} == {"judged", "computed"}
    assert saved["counts_by_basis"] == {"judged": 1, "computed": len(
        [f for f in stored if f["basis"] == "computed"])}


# ── D61: the third value ────────────────────────────────────────────────────

def test_a_text_similarity_match_is_neither_computed_nor_judged(conn):
    """D61. §5.4 matches findings across versions by character similarity and had to stamp
    them `judged` with a caveat, because they are neither: a similarity score is not a human
    judgment, and it is not identical for every user in the sense that matters, since it is a
    heuristic whose threshold somebody chose."""
    assert "heuristic" in core._BASES


def test_the_diff_uses_it_instead_of_borrowing_judged(conn):
    peru = core.ingest_campaign(conn, title="Peru",
                                detail="Seeded one colourway.")["campaign_id"]
    v1 = core.ingest_campaign(conn, title="Colombia v1",
                              detail="Seeds four.")["campaign_id"]
    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1,
                              detail="Seeds four still.")["campaign_id"]
    finding = {"severity": "should_fix", "kind": "precedent_departure",
               "departure": "regression", "fix": "Seed one",
               "finding": "Seeds four colourways where Peru used one",
               "precedent": {"campaign_id": peru, "quote": "Seeded one colourway"}}
    _saved(conn, campaign_id=v1, findings=[finding])
    _saved(conn, campaign_id=v2, subject_title="Colombia v2", findings=[finding])

    again = core.diff_campaigns(conn, earlier=v1, later=v2)["raised_again"]
    matched_by_text = [e for e in again if e.get("match") == "text"]

    assert matched_by_text
    assert matched_by_text[0]["basis"] == "heuristic"


def test_every_basis_says_what_it_means(conn):
    """Three values, each with a sentence — and the sentences are what stop `heuristic` being
    read as a softer `computed`."""
    for value in core._BASES:
        assert core.BASIS_MEANING[value]
    assert "threshold" in core.BASIS_MEANING["heuristic"].lower()
    assert "every user" in core.BASIS_MEANING["computed"].lower()


def test_the_model_may_not_claim_heuristic_either(conn):
    """It is the server's word for its own inference. A model sending it would be claiming a
    provenance rather than reporting one, which is the whole reason `computed` is refused."""
    with pytest.raises(ValueError):
        _saved(conn, findings=[{"severity": "should_fix", "kind": "missing_information",
                                "finding": "No end date", "fix": "Add one",
                                "basis": "heuristic"}])


def test_it_reaches_the_model_over_the_protocol(conn):
    import asyncio
    import json
    import mcp_server

    async def call(name, args):
        return json.loads((await mcp_server.mcp.call_tool(name, args)).content[0].text)

    saved = asyncio.run(call("save_evaluation", {
        "subject_title": "Colombia v1", "verdict": "revise", "summary": "A stretch.",
        "approve_if": "Fixed.", "subject_text": "A launch with no dates at all.",
        "findings": [{"severity": "should_fix", "kind": "missing_information",
                      "finding": "No end date", "fix": "Add one"}]}))

    assert saved["counts_by_basis"]["computed"] >= 1
    assert "computed" in saved["note"].lower()
