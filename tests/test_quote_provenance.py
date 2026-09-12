"""
§6.1 (idea G): `precedent.quote` required on every finding that makes a claim about a
precedent, and VERIFIED against the record it cites.

The review's complaint was that a finding reading "this departs from what you did in Peru"
is an assertion until somebody can see the Peru brief's own words. §2.4 gave the quote a
field and a cap. What it could not do was check that the quote was real: `_clean_precedent`
never saw the connection, so a citation could name a campaign that does not exist and quote
a sentence nobody ever wrote, and the server stored it with the same authority as a faithful
one. That is worse than no citation at all — an invented quote reads as evidence.

**The plan re-sequenced this behind 7.2 on a premise that turns out to be wrong.** The note
said "drawn from the retrieved chunk" cannot be enforced while `prepare_evaluation` is
stateless, because the server does not retain what it returned, so 6.1 needs 7.2 or a
receipt id. That is true of the literal wording and irrelevant to the defect. The thing
worth preventing is a quote attributed to a record that the record does not contain, and
that is checkable against the CITED RECORD's own stored text, with no receipt, no retained
window, and no dependency on who wrote the retrieval query. "Was it inside the top_k window
of this particular call" is a different and much weaker question, and it belongs with 7.6's
stamp of retrieved ids — a tracker row carries it there.

The layer half is the reason §2.5's comment predicted this item would otherwise "bless the
misattribution with a green tick": a quote lifted from a reviewer's objection on a deck IS
in that record's stored text, so verifying the text alone would confirm "the campaign says
the timeline is unrealistic" — which the campaign never said. Verification is per layer.
"""
import pytest

import core
import store


PERU_BODY = ("Launch runs 3 March to 28 March across Lima and Arequipa. Every asset in the "
             "flighting table carries a content angle, posting date and requirements per "
             "asset, and no creator is booked against a competing brand in the window.")
PERU_NOTE = "I do not think this timeline is realistic for a market this size."


@pytest.fixture
def peru(conn):
    """A real record with real text, body and commentary, so a quote can be checked.

    The commentary goes in through `store.insert_chunks` because the only production path
    that produces it is file extraction, and this test is about what is stored, not how it
    got there."""
    cid = core.ingest_campaign(conn, title="Peru launch", detail=PERU_BODY,
                               status="concluded")["campaign_id"]
    store.insert_chunks(conn, cid, [PERU_NOTE], kind="commentary",
                        sources=[{"kind": "comment", "author": "R. Vega",
                                  "anchor": "slide 4"}])
    return cid


def _evaluation(finding, **over):
    base = {
        "subject_title": "Colombia v2",
        "verdict": "revise",
        "summary": "Nothing is dated.",
        "findings": [finding],
    }
    base.update(over)
    return base


# ── the quote has to be in the record it names ──────────────────────────────

def test_a_quote_the_cited_campaign_never_contained_is_refused(conn, peru):
    """The whole item in one test: an invented quote reads as evidence, which is why it is
    worse than no citation."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "precedent_departure",
            "finding": "Undated deliverables",
            "precedent": {"campaign_id": peru,
                          "quote": "every deliverable must be dated eight weeks ahead"}}))
    assert "quote" in str(e.value).lower()
    assert peru in str(e.value), "name the record it failed against"


def test_a_faithful_quote_is_accepted_and_marked_verified(conn, peru):
    result = core.save_evaluation(conn, **_evaluation({
        "severity": "blocking", "kind": "precedent_departure",
        "finding": "Undated deliverables",
        "precedent": {"campaign_id": peru,
                      "quote": "content angle, posting date and requirements per asset"}}))

    stored = store.get_evaluation(conn, result["evaluation_id"])
    assert stored["findings"][0]["precedent"]["verified"] is True


def test_citing_a_campaign_that_does_not_exist_is_refused(conn, peru):
    """`camp_jdsea` was the placeholder in this codebase's own fixtures and docstrings for
    months, and the server would have stored it against a real user's judgment."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "precedent_departure",
            "finding": "Undated deliverables",
            "precedent": {"campaign_id": "camp_jdsea", "quote": "posting date"}}))
    assert "camp_jdsea" in str(e.value)


# ── the layer half ──────────────────────────────────────────────────────────

def test_a_reviewers_objection_cannot_be_stored_as_what_the_deck_said(conn, peru):
    """The misattribution §2.5 predicted: the text IS in the record, so text-only
    verification would have returned a green tick on a false statement."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation({
            "severity": "should_fix", "kind": "precedent_departure",
            "finding": "Timeline questioned before",
            "precedent": {"campaign_id": peru, "quote": "timeline is realistic",
                          "layer": "body"}}))
    assert "commentary" in str(e.value).lower()


def test_the_same_quote_marked_as_commentary_is_accepted(conn, peru):
    result = core.save_evaluation(conn, **_evaluation({
        "severity": "should_fix", "kind": "precedent_departure",
        "finding": "Timeline questioned before",
        "precedent": {"campaign_id": peru, "quote": "timeline is realistic",
                      "layer": "commentary", "author": "R. Vega", "anchor": "slide 4"}}))

    stored = store.get_evaluation(conn, result["evaluation_id"])
    assert stored["findings"][0]["precedent"]["verified"] is True
    assert stored["findings"][0]["precedent"]["layer"] == "commentary"


def test_a_body_quote_marked_as_commentary_is_also_refused(conn, peru):
    """Both directions. Marking the deck's own words as somebody's opinion lets a finding
    dodge the weight its own citation carries."""
    with pytest.raises(ValueError):
        core.save_evaluation(conn, **_evaluation({
            "severity": "should_fix", "kind": "precedent_departure",
            "finding": "Dates were required before",
            "precedent": {"campaign_id": peru, "quote": "posting date and requirements",
                          "layer": "commentary", "author": "R. Vega"}}))


# ── required-ness ───────────────────────────────────────────────────────────

def test_a_citation_without_a_quote_is_refused(conn, peru):
    """Before this, `quote` was bounded but never required — a finding could name a campaign
    and quote nothing, which is the assertion the item was written about with an id stapled
    to it."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "precedent_departure",
            "finding": "Undated deliverables",
            "precedent": {"campaign_id": peru}}))
    assert "quote" in str(e.value).lower()


@pytest.mark.parametrize("kind", ["precedent_departure", "guardrail_breach"])
def test_a_finding_about_a_precedent_must_cite_one(conn, peru, kind):
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": kind,
            "finding": "Departs from how this is normally done"}))
    assert "precedent" in str(e.value).lower()


@pytest.mark.parametrize("kind", ["missing_information", "internal_contradiction"])
def test_a_finding_about_the_subject_needs_no_precedent(conn, peru, kind):
    """Deliberately NOT required. "The brief gives no end date" is anchored to the subject,
    which is not in the library; demanding a precedent quote there would make the model go
    and find a campaign to quote at, and a citation produced to satisfy a validator is the
    invented evidence this item exists to stop."""
    result = core.save_evaluation(conn, **_evaluation({
        "severity": "blocking", "kind": kind, "finding": "The brief gives no end date"}))
    assert result["evaluation_id"]


# ── faithful, not byte-identical ────────────────────────────────────────────

@pytest.mark.parametrize("quote", [
    "Content Angle, Posting Date and requirements per asset",       # re-cased
    "content angle,  posting date\nand requirements per asset",     # re-wrapped
    "“content angle, posting date”",                      # curly quotes around it
    "content angle … requirements per asset",                  # elided
    "content angle ... requirements per asset",                     # elided, ascii
])
def test_a_faithful_quote_survives_the_ways_models_actually_quote(conn, peru, quote):
    """Refusing these would not improve provenance. It would teach the model that quoting is
    a game it loses, and the way models win that game is by quoting less."""
    assert core.save_evaluation(conn, **_evaluation({
        "severity": "blocking", "kind": "precedent_departure",
        "finding": "Undated deliverables",
        "precedent": {"campaign_id": peru, "quote": quote}}))["evaluation_id"]


def test_an_elision_does_not_let_the_order_be_reversed(conn, peru):
    """"requirements per asset ... content angle" is not what the record says, and an
    elision that matched in any order would make the wildcard a way through the check."""
    with pytest.raises(ValueError):
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "precedent_departure",
            "finding": "Undated deliverables",
            "precedent": {"campaign_id": peru,
                          "quote": "requirements per asset … content angle"}}))


def test_an_elision_cannot_be_stretched_across_two_records(conn, peru):
    """Each segment must land in the same record — and this one exists because the first
    implementation searched a joined blob of every chunk, which is the same hole one level
    down."""
    other = core.ingest_campaign(conn, title="Chile launch",
                                 detail="Budget is fixed at 40,000 USD.")["campaign_id"]
    with pytest.raises(ValueError):
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "precedent_departure",
            "finding": "Budget departs",
            "precedent": {"campaign_id": other,
                          "quote": "Budget is fixed … requirements per asset"}}))


# ── rule_id is not a way round it ───────────────────────────────────────────

def test_a_rule_id_that_resolves_to_nothing_is_refused(conn, peru):
    """Without this the check is decorative: any finding could be saved unverified by
    writing `rule_id` where `campaign_id` would have been checked."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "guardrail_breach",
            "finding": "Breaches the tone rule",
            "precedent": {"rule_id": "rule_tone", "quote": "never use superlatives"}}))
    assert "rule_tone" in str(e.value)


def test_a_rule_id_naming_a_reference_record_verifies_like_any_other(conn):
    rules = core.ingest_campaign(conn, title="Brand guidelines", record_type="reference",
                                 detail="Never use superlatives in paid social.")["campaign_id"]
    result = core.save_evaluation(conn, **_evaluation({
        "severity": "blocking", "kind": "guardrail_breach",
        "finding": "Breaches the tone rule",
        "precedent": {"rule_id": rules, "quote": "Never use superlatives"}}))

    stored = store.get_evaluation(conn, result["evaluation_id"])
    assert stored["findings"][0]["precedent"]["verified"] is True


def test_a_rule_id_pointing_at_an_ordinary_campaign_is_refused(conn, peru):
    """`rule_id` means the rulebook. Letting it name any record would make the two slots
    interchangeable, and then "a rule was broken" could be anchored to somebody's Q3 deck."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "guardrail_breach",
            "finding": "Breaches the tone rule",
            "precedent": {"rule_id": peru, "quote": "posting date"}}))
    assert "reference" in str(e.value).lower()


# ── a record with nothing to check against ──────────────────────────────────

def test_a_record_with_no_text_says_so_rather_than_failing_as_a_bad_quote(conn):
    """A metrics-only stub has nothing to quote. Reporting that as "your quote is not in
    this record" would send the model off to reword a quote that was never the problem."""
    stub = store.insert_campaign(conn, title="Imported KPI row", record_type="stub")
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "precedent_departure",
            "finding": "Departs from it",
            "precedent": {"campaign_id": stub, "quote": "anything at all"}}))
    assert "no text" in str(e.value).lower() or "nothing" in str(e.value).lower()


# ── the server's own findings ───────────────────────────────────────────────

def test_the_servers_computed_findings_are_held_to_the_same_check(conn, peru):
    """`trusted` exempts a finding from the "only the server says computed" rule. It is not
    a way to store an unverifiable citation — §7.1's computed findings quote the subject and
    carry no precedent, so nothing legitimate needs the exemption."""
    with pytest.raises(ValueError):
        core.save_evaluation(conn, trusted=True, **_evaluation({
            "severity": "blocking", "kind": "precedent_departure", "basis": "computed",
            "finding": "Departs from it",
            "precedent": {"campaign_id": peru, "quote": "a sentence nobody wrote"}}))


# ── it reaches the tool ─────────────────────────────────────────────────────

def test_the_refusal_reaches_the_model_over_the_protocol(conn, peru):
    """Every check here is in core. The tool is where the model meets it, and a validation
    error that the transport turns into "an error occurred" is not a retry.

    No connection monkeypatching: the tool opens its own against the same patched DB_PATH,
    which is the real call path."""
    import mcp_server

    refused = mcp_server.save_evaluation(
        subject_title="Colombia v2", verdict="revise", summary="Nothing is dated.",
        findings=[{"severity": "blocking", "kind": "precedent_departure",
                   "finding": "Undated deliverables",
                   "precedent": {"campaign_id": peru,
                                 "quote": "a sentence nobody wrote"}}])

    # `{"error": ...}`, not a raised exception: the MCP framework turns anything that is not
    # its own ToolError into "Error executing tool X" and discards the message, which is why
    # every tool here is wrapped.
    assert "evaluation_id" not in refused, "it must not be stored"
    assert "quote" in refused["error"].lower()
    assert peru in refused["error"]


def test_a_faithful_quote_goes_through_the_tool_too(conn, peru):
    """The other half, and the one a refusal-only test cannot give: a check that refuses
    everything passes every test written about refusals."""
    import mcp_server

    saved = mcp_server.save_evaluation(
        subject_title="Colombia v2", verdict="revise", summary="Nothing is dated.",
        findings=[{"severity": "blocking", "kind": "precedent_departure",
                   "finding": "Undated deliverables",
                   "precedent": {"campaign_id": peru, "quote": "posting date"}}])

    assert saved["evaluation_id"]
