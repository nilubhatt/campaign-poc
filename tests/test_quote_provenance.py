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
    # §6.2 made `kind` required and a `precedent_departure` say which way it departs. These
    # tests are about the QUOTE, so the classification is filled in here rather than repeated
    # in sixty places — a departure whose direction is not the point is `unexplained`.
    finding = dict(finding)
    if finding.get("kind") == "precedent_departure":
        finding.setdefault("departure", "unexplained")
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
    finding = stored["findings"][0]
    # `checked`, and no more. NOT `verified: true` — that word already means a performance
    # claim backed by real metrics in this product, and on a finding the nearer reading is
    # that the FINDING is verified. And NOT `basis: "computed"`: a finding's `basis` says who
    # produced the finding, so the same key one level down would claim the server wrote the
    # citation. It did not; it only checked it.
    assert finding["precedent"]["checked"] == ["record", "layer"]
    assert "basis" not in finding["precedent"], "that key means something else on the finding"
    assert finding["basis"] == "judged", "and it still says the model reached this"


def test_the_check_is_the_servers_to_make_not_the_models(conn, peru):
    """A model-asserted `verified` was silently discarded, which is the one case
    `save_evaluation` refuses outright one level up — a finding cannot claim `basis:
    computed` either."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "precedent_departure",
            "finding": "Undated deliverables",
            "precedent": {"campaign_id": peru, "quote": "posting date and requirements",
                          "verified": True}}))
    assert "server" in str(e.value).lower()


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
    assert stored["findings"][0]["precedent"]["checked"] == ["record", "layer"]
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
    assert stored["findings"][0]["precedent"]["checked"] == ["record", "layer"]


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
                   "departure": "unexplained", "finding": "Undated deliverables",
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
                   "departure": "unexplained", "finding": "Undated deliverables",
                   "precedent": {"campaign_id": peru, "quote": "posting date"}}])

    assert saved["evaluation_id"]


# ── review round: the ways past it, and the ways it blocked honest work ─────

def test_naming_both_a_campaign_and_a_rule_does_not_launder_one_past_the_check(conn, peru):
    """The check was decorative from the other direction too. With both slots filled the
    quote was checked against the RULE and the campaign_id could be anything at all —
    invented included — and was stored beside a passing check."""
    rules = core.ingest_campaign(conn, title="Brand guidelines", record_type="reference",
                                 detail="Never use superlatives in paid social.")["campaign_id"]
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "guardrail_breach",
            "finding": "Breaches the tone rule",
            "precedent": {"campaign_id": "camp_invented", "rule_id": rules,
                          "quote": "Never use superlatives"}}))
    assert "one thing" in str(e.value) or "two findings" in str(e.value)


def test_a_breach_cannot_be_anchored_to_a_campaign_through_the_other_slot(conn, peru):
    """`rule_id` must name reference material — but nothing stopped a guardrail_breach from
    using the `campaign_id` slot instead, which is the "a rule was broken, see somebody's Q3
    deck" the rule_id check was written to prevent."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "guardrail_breach",
            "finding": "Breaches the tone rule",
            "precedent": {"campaign_id": peru, "quote": "posting date and requirements"}}))
    assert "rule_id" in str(e.value)


def test_a_departure_cannot_be_anchored_to_a_rule(conn):
    """The other direction: a rule is not debatable and a departure invites a rationale, so
    the slots are not interchangeable in either direction."""
    rules = core.ingest_campaign(conn, title="Brand guidelines", record_type="reference",
                                 detail="Never use superlatives in paid social.")["campaign_id"]
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "precedent_departure",
            "finding": "Done differently",
            "precedent": {"rule_id": rules, "quote": "Never use superlatives"}}))
    assert "campaign_id" in str(e.value)


@pytest.mark.parametrize("quote", [
    "a", ".", "the", "per asset",                       # too short outright
    "Every asset in the flighting table … per",         # long enough overall, fragment after
    "per … posting date and requirements",              # fragment before
])
def test_a_fragment_is_not_a_quotation(conn, peru, quote):
    """Without a floor, "a" and "." verified against every record in the library. A substring
    test with a one-character floor certifies nothing, and a word like "verified" or
    "computed" beside it is then a claim the check cannot support."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "precedent_departure",
            "finding": "Undated deliverables",
            "precedent": {"campaign_id": peru, "quote": quote}}))
    assert "short" in str(e.value).lower() or "fragment" in str(e.value).lower()


def test_an_elision_cannot_reach_across_the_whole_deck(conn):
    """The per-unit rule was not the protection it was described as. `chunking.pack` merges
    slides up to 1800 characters, so a short deck is ONE unit — and an unbounded elision
    inside it stitched two unrelated slides into a sentence the deck never contained, with
    the meaning inverted."""
    deck = core.ingest_campaign(conn, title="Three slides", deck_text=(
        "Slide 1. Budget is fixed at 40,000 USD for the whole activation.\n\n"
        "Slide 2. " + ("Creator briefing detail. " * 14) + "\n\n"
        "Slide 3. Nobody may post before the embargo lifts on 3 March."))["campaign_id"]
    with pytest.raises(ValueError):
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "precedent_departure",
            "finding": "Budget departs",
            "precedent": {"campaign_id": deck,
                          "quote": "Budget is fixed … may post before the embargo"}}))


def test_a_short_gap_in_one_sentence_is_still_a_quotation(conn, peru):
    """The bound has to leave real elisions working, or it is the same over-refusal one
    level down."""
    assert core.save_evaluation(conn, **_evaluation({
        "severity": "blocking", "kind": "precedent_departure",
        "finding": "Undated deliverables",
        "precedent": {"campaign_id": peru,
                      "quote": "Every asset in the flighting table … posting date"}}))


def test_a_quote_across_a_chunk_boundary_is_not_called_a_paraphrase(conn):
    """`chunking.pack` splits at 1800 characters — a boundary the model cannot see and the
    document does not have. A verbatim, contiguous quote across it was refused with "do not
    paraphrase into a quote", and re-copying more carefully could never satisfy it."""
    long_deck = "\n\n".join(
        f"Slide {n}. " + ("Creator briefing and flighting detail for this market. " * 12)
        for n in range(1, 9))
    cid = core.ingest_campaign(conn, title="Long deck", deck_text=long_deck)["campaign_id"]
    chunks = [r["text"] for r in conn.execute(
        "SELECT text FROM campaign_chunks WHERE campaign_id = ? ORDER BY chunk_index",
        (cid,)).fetchall()]
    assert len(chunks) > 2, "the fixture has to actually straddle a boundary"
    spanning = chunks[1][-45:] + " " + chunks[2][:45]

    assert core.save_evaluation(conn, **_evaluation({
        "severity": "blocking", "kind": "precedent_departure",
        "finding": "Departs from the briefing detail",
        "precedent": {"campaign_id": cid, "quote": spanning[:290]}}))["evaluation_id"]


def test_a_quote_of_what_the_record_no_longer_says_is_refused(conn):
    """`update_campaign` rewrites `detail` without re-chunking, so the old wording survived
    in chunk 0 and a citation of what the marketer had already corrected was stored as
    checked."""
    cid = core.ingest_campaign(
        conn, title="Chile launch", detail="Budget is fixed at 10,000 USD.")["campaign_id"]
    store.update_campaign(conn, cid, detail="Budget is fixed at 90,000 USD.")

    with pytest.raises(ValueError):
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "precedent_departure",
            "finding": "Budget departs",
            "precedent": {"campaign_id": cid, "quote": "Budget is fixed at 10,000 USD"}}))
    assert core.save_evaluation(conn, **_evaluation({
        "severity": "blocking", "kind": "precedent_departure",
        "finding": "Budget departs",
        "precedent": {"campaign_id": cid, "quote": "Budget is fixed at 90,000 USD"}}))


@pytest.mark.parametrize("stored,quoted", [
    # Real extraction output, from review running hand-built files through extract.py.
    ("Full require\u00adments per asset.", "Full requirements per asset"),   # soft hyphen
    ("Posting sched-\nule per creator.", "Posting schedule per creator"),    # line-break hyphen
    ("Caf\u00e9 launch runs 3\u00a0March.", "Caf\u0065\u0301 launch runs 3 March"),  # NFD vs NFC
    ("Deadline is 3\u2010March.", "Deadline is 3-March"),                    # U+2010 hyphen
    ("Full\u200brequirements per asset.", "Fullrequirements per asset"),     # zero-width space
])
def test_extraction_artefacts_do_not_turn_a_verbatim_quote_into_a_paraphrase(
        conn, stored, quoted):
    """Every one of these came out of a real PDF or PPTX and refused a quote a person would
    call verbatim — with a message accusing the model of paraphrasing. That is precisely the
    "quoting is a game it loses" outcome this check was written to avoid."""
    cid = core.ingest_campaign(conn, title="Extracted deck", detail=stored)["campaign_id"]
    assert core.save_evaluation(conn, **_evaluation({
        "severity": "blocking", "kind": "precedent_departure",
        "finding": "Departs from it",
        "precedent": {"campaign_id": cid, "quote": quoted}}))["evaluation_id"]


def test_the_truncation_marker_this_server_adds_is_not_held_against_the_model(conn, peru):
    """`find_similar` trims a long brief and marks it "... [truncated]". A model quoting the
    tail of what it was shown includes the marker, and refusing that is refusing our own
    punctuation."""
    assert core.save_evaluation(conn, **_evaluation({
        "severity": "blocking", "kind": "precedent_departure",
        "finding": "Undated deliverables",
        "precedent": {"campaign_id": peru,
                      "quote": "content angle, posting date... [truncated]"}}))


def test_a_long_comment_split_across_chunks_can_still_be_quoted_whole(conn):
    """One comment longer than a chunk is stored as several pieces that all keep the same
    attribution, so a quote across that split is one person's sentence."""
    cid = core.ingest_campaign(conn, title="Peru launch",
                               detail="Six-week flight.")["campaign_id"]
    long_note = ("The timeline concerns me for a market this size. " * 45)
    import chunking
    pieces = chunking.pack([long_note])
    assert len(pieces) > 1, "the fixture has to actually split"
    store.insert_chunks(conn, cid, pieces, kind="commentary",
                        sources=[{"kind": "comment", "author": "R. Vega"}] * len(pieces))
    spanning = pieces[0][-40:] + " " + pieces[1][:40]

    assert core.save_evaluation(conn, **_evaluation({
        "severity": "should_fix", "kind": "precedent_departure",
        "finding": "Timeline questioned before",
        "precedent": {"campaign_id": cid, "quote": spanning[:290], "layer": "commentary",
                      "author": "R. Vega"}}))["evaluation_id"]


def test_two_peoples_remarks_are_never_joined_into_one_quotation(conn):
    """The concession above is per ATTRIBUTION. Joining everything would let a quotation be
    stitched out of two people's comments, which is the misattribution the layer rule exists
    to stop, one level down."""
    cid = core.ingest_campaign(conn, title="Peru launch",
                               detail="Six-week flight.")["campaign_id"]
    store.insert_chunks(conn, cid, ["The timeline concerns me a great deal."],
                        kind="commentary", sources=[{"author": "R. Vega"}])
    store.insert_chunks(conn, cid, ["The budget is more than generous here."],
                        kind="commentary", sources=[{"author": "K. Mensah"}])
    with pytest.raises(ValueError):
        core.save_evaluation(conn, **_evaluation({
            "severity": "should_fix", "kind": "precedent_departure",
            "finding": "Both were raised",
            "precedent": {"campaign_id": cid, "layer": "commentary", "author": "R. Vega",
                          "quote": "The timeline concerns me … budget is more than generous"}}))


def test_the_refusal_does_not_send_a_citing_finding_round_a_loop(conn, peru):
    """The advice was "drop the citation and say it as an observation" — which a
    precedent_departure is then refused a second time for taking. The only exit left was
    deleting `kind`, which no message mentioned."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "precedent_departure",
            "finding": "Undated deliverables",
            "precedent": {"campaign_id": peru, "quote": "a sentence nobody ever wrote"}}))
    message = str(e.value)
    assert "drop the citation" not in message, "that exit is refused for this kind"
    assert "missing_information" in message or "internal_contradiction" in message


def test_the_rule_refusal_does_not_offer_a_downgrade_as_the_way_out(conn, peru):
    """§2.4's lesson is that any easy exit offered inside a validation message gets taken,
    and "call it a precedent_departure instead" is a downgrade from "not debatable" to
    "arguable" — the severity-downgrade pattern one field across."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "guardrail_breach",
            "finding": "Breaches the tone rule",
            "precedent": {"rule_id": peru, "quote": "posting date and requirements"}}))
    assert "is a precedent_departure" not in str(e.value)


def test_an_ellipsis_on_its_own_is_not_a_quote(conn, peru):
    """It leaves no segments at all, and an empty segment list must never read as "found"."""
    with pytest.raises(ValueError):
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "precedent_departure",
            "finding": "Undated deliverables",
            "precedent": {"campaign_id": peru, "quote": "…"}}))


def test_the_two_layer_mismatches_say_different_things(conn, peru):
    """Both directions are refused, and a test that only asserts `raises` cannot tell whether
    the message sent the model the right way — swapping the two passed."""
    with pytest.raises(ValueError) as body_marked_as_commentary:
        core.save_evaluation(conn, **_evaluation({
            "severity": "should_fix", "kind": "precedent_departure",
            "finding": "Dates were required before",
            "precedent": {"campaign_id": peru, "quote": "posting date and requirements",
                          "layer": "commentary", "author": "R. Vega"}}))
    assert 'layer to "body"' in str(body_marked_as_commentary.value)

    with pytest.raises(ValueError) as commentary_marked_as_body:
        core.save_evaluation(conn, **_evaluation({
            "severity": "should_fix", "kind": "precedent_departure",
            "finding": "Timeline questioned before",
            "precedent": {"campaign_id": peru, "quote": "not think this timeline is realistic",
                          "layer": "body"}}))
    assert 'layer to "commentary"' in str(commentary_marked_as_body.value)


def test_a_title_only_stub_cannot_have_its_own_title_quoted_against_it(conn):
    """"has no brief on file" is checked after the match, so a title quote would verify. The
    floor is what stops that being a way to cite a metrics row as evidence."""
    stub = store.insert_campaign(conn, title="Imported KPI row Q3 Jakarta", record_type="stub")
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "precedent_departure",
            "finding": "Departs from it",
            "precedent": {"campaign_id": stub, "quote": "Imported KPI row Q3 Jakarta"}}))
    assert "no brief" in str(e.value)


def test_the_headline_precedent_is_checked_too(conn, peru):
    """`closest_precedent` is the same id-and-quote shape one field up, and it was outside
    the check — an invented citation at the very top of the judgment, where a summary is
    most likely to read it aloud."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, closest_precedent={"campaign_id": "camp_invented",
                                                      "similarity": 0.91},
                             **_evaluation({"severity": "blocking",
                                            "kind": "missing_information",
                                            "finding": "No end date"}))
    assert "camp_invented" in str(e.value)

    with pytest.raises(ValueError):
        core.save_evaluation(conn, closest_precedent={"campaign_id": peru,
                                                      "quote": "a sentence nobody ever wrote"},
                             **_evaluation({"severity": "blocking",
                                            "kind": "missing_information",
                                            "finding": "No end date"}))

    assert core.save_evaluation(
        conn, closest_precedent={"campaign_id": peru, "similarity": 0.91},
        **_evaluation({"severity": "blocking", "kind": "missing_information",
                       "finding": "No end date"}))["evaluation_id"]


async def _call(tool, arguments):
    import mcp_server
    import json
    result = await mcp_server.mcp.call_tool(tool, arguments)
    return json.loads(result.content[0].text)


def test_the_missing_quote_message_survives_the_transport(conn, peru):
    """Typed as `quote: str`, pydantic refused the call at its own boundary and the caller
    got "Field required" — not the sentence this project wrote about what a citation without
    a quote actually is. That is the exact failure `_enum`'s comment describes, one field
    across: a value that never reaches project code cannot be answered by project code."""
    import asyncio

    refused = asyncio.run(_call("save_evaluation", {
        "subject_title": "Colombia v2", "verdict": "revise", "summary": "Nothing is dated.",
        "findings": [{"severity": "blocking", "kind": "precedent_departure",
                      "departure": "unexplained", "finding": "Undated deliverables",
                      "precedent": {"campaign_id": peru}}]}))

    assert "error" in refused, refused
    assert "assertion" in refused["error"], "core's message, not pydantic's"


def test_a_dash_is_a_dash_however_it_was_typed(conn):
    """Deleting the dash table left the whole suite green. A model retyping "3 March – 28
    March" as a hyphen is not paraphrasing."""
    cid = core.ingest_campaign(
        conn, title="Dashes", detail="Flight runs 3 March \u2013 28 March in Lima.")["campaign_id"]
    assert core.save_evaluation(conn, **_evaluation({
        "severity": "blocking", "kind": "precedent_departure", "finding": "Window departs",
        "precedent": {"campaign_id": cid, "quote": "3 March - 28 March in Lima"}}))


def test_a_record_whose_only_text_is_a_comment_can_still_be_cited(conn):
    """`brief` has to count commentary. Ignoring it left the whole suite green, and the
    refusal would have been the wrong one — "nothing here to quote" about a record with a
    reviewer's remark in it."""
    cid = store.insert_campaign(conn, title="Imported row", record_type="stub")
    store.insert_chunks(conn, cid, ["We agreed never to run this creative in Peru again."],
                        kind="commentary", sources=[{"author": "R. Vega"}])

    assert core.save_evaluation(conn, **_evaluation({
        "severity": "should_fix", "kind": "precedent_departure",
        "finding": "Reruns a creative that was retired",
        "precedent": {"campaign_id": cid, "layer": "commentary", "author": "R. Vega",
                      "quote": "never to run this creative in Peru again"}}))


def test_a_chunk_of_an_unrecognised_kind_is_not_somebodys_comment(conn):
    """Commentary is exactly what was stored as commentary. Treating an unknown kind as
    commentary would let text of unknown provenance be cited as a named person's remark —
    the misattribution the layer rule exists to stop, arriving through the back."""
    cid = core.ingest_campaign(conn, title="Peru launch",
                               detail="Six-week flight across Lima.")["campaign_id"]
    store.insert_chunks(conn, cid, ["Some text of a kind nobody has defined yet."],
                        kind="future_layer")

    with pytest.raises(ValueError):
        core.save_evaluation(conn, **_evaluation({
            "severity": "should_fix", "kind": "precedent_departure",
            "finding": "Said before",
            "precedent": {"campaign_id": cid, "layer": "commentary", "author": "R. Vega",
                          "quote": "a kind nobody has defined yet"}}))


# ── second review round ─────────────────────────────────────────────────────

def test_an_elision_cannot_be_chained_across_a_deck_in_legal_hops(conn):
    """Bounding each HOP did not bound the quote. Ten 130-character hops across a thirty-
    slide deck passed every check, because each one was legal on its own — the total is what
    makes the arithmetic impossible to walk around."""
    deck = core.ingest_campaign(conn, title="Thirty slides", deck_text="\n\n".join(
        f"Slide {n} says point number {n} clearly and briefly." for n in range(1, 31))
    )["campaign_id"]
    with pytest.raises(ValueError):
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "precedent_departure", "finding": "Departs",
            "precedent": {"campaign_id": deck, "quote": " … ".join(
                f"point number {n} clearly" for n in range(1, 30, 3))}}))


def test_a_short_deck_cannot_be_stitched_into_a_sentence_it_never_contained(conn):
    """The original reproduction, with nothing padding it out: four ordinary slides, and a
    "quote" assembled from one phrase in each."""
    deck = core.ingest_campaign(conn, title="Four slides", deck_text=(
        "Budget is fixed at 40,000 USD.\n\nCreators are briefed on Monday.\n\n"
        "Nobody may exceed the cap.\n\nNothing goes live before the embargo."))["campaign_id"]
    with pytest.raises(ValueError):
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "precedent_departure", "finding": "Departs",
            "precedent": {"campaign_id": deck,
                          "quote": "Budget is fixed … Creators are briefed … Nobody may "
                                   "exceed … Nothing goes live"}}))


@pytest.mark.parametrize("stored,quoted", [
    ("Flight runs 3 - 28 March in Lima.", "3-28 March in Lima"),
    ("Flight runs 3 - 28 March in Lima.", "3\u2013 28 March in Lima"),
    ("Flight runs 3\u201328 March in Lima.", "3 - 28 March in Lima"),
    ("The Lima\u2014based roster is fixed.", "The Lima - based roster is fixed"),
    ("Budget\u2014and this is final\u2014is 40,000 USD.", "Budget — and this is final — is 40,000"),
])
def test_a_dash_reads_the_same_however_it_was_spaced(conn, stored, quoted):
    """Deleting every "- " deleted a SPACED dash and left an unspaced one, so the fold was
    asymmetric and a faithful quote came back as "do not paraphrase"."""
    cid = core.ingest_campaign(conn, title="Dashes", detail=stored)["campaign_id"]
    assert core.save_evaluation(conn, **_evaluation({
        "severity": "blocking", "kind": "precedent_departure", "finding": "Departs",
        "precedent": {"campaign_id": cid, "quote": quoted}}))["evaluation_id"]


def test_a_word_split_across_a_line_break_is_still_one_word(conn):
    """The narrower rule still has to do the job it was added for."""
    cid = core.ingest_campaign(
        conn, title="Justified", detail="Posting sched-\nule per creator agreed.")["campaign_id"]
    assert core.save_evaluation(conn, **_evaluation({
        "severity": "blocking", "kind": "precedent_departure", "finding": "Departs",
        "precedent": {"campaign_id": cid, "quote": "Posting schedule per creator"}}))


def test_what_a_campaign_actually_achieved_can_be_quoted(conn):
    """The product's most common real citation, and it was refused outright: metric detail is
    what `find_similar` shows the model as evidence, and a metrics-only record was told it had
    "nothing but a title" while the numbers' own words were what was being quoted."""
    cid = core.ingest_campaign(conn, title="Jakarta launch",
                               detail="Six-week flight.", status="concluded")["campaign_id"]
    core.add_metrics(conn, campaign_id=cid,
                     detail="CTR was 3.2 percent, well above the 1.8 percent benchmark.")

    assert core.save_evaluation(conn, **_evaluation({
        "severity": "should_fix", "kind": "precedent_departure",
        "finding": "Forecast is below what Jakarta actually did",
        "precedent": {"campaign_id": cid,
                      "quote": "well above the 1.8 percent benchmark"}}))["evaluation_id"]


def test_a_campaign_title_is_not_evidence(conn):
    """It was quotable as body text whenever the record had any detail at all, so the same
    quote was evidence or not depending on whether a brief happened to exist."""
    cid = core.ingest_campaign(conn, title="Always-on Lima Creator Programme 2026",
                               detail="Six-week flight.")["campaign_id"]
    with pytest.raises(ValueError):
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "precedent_departure", "finding": "Departs",
            "precedent": {"campaign_id": cid,
                          "quote": "Always-on Lima Creator Programme 2026"}}))


def test_two_unattributed_comments_are_not_one_persons_sentence(conn):
    """`None == None`, so an unattributed run collapsed into one quotable block — two
    strangers' remarks stitched into a single quotation, which is the misattribution the
    layer rule exists to stop."""
    cid = core.ingest_campaign(conn, title="Peru", detail="Six-week flight.")["campaign_id"]
    store.insert_chunks(conn, cid, ["The timeline concerns me a great deal."],
                        kind="commentary")
    store.insert_chunks(conn, cid, ["The budget is more than generous here."],
                        kind="commentary")
    with pytest.raises(ValueError):
        core.save_evaluation(conn, **_evaluation({
            "severity": "should_fix", "kind": "precedent_departure", "finding": "Both raised",
            "precedent": {"campaign_id": cid, "layer": "commentary",
                          "quote": "timeline concerns me … budget is more than generous"}}))


def test_the_same_comment_is_joined_whatever_order_its_keys_were_written_in(conn):
    """Comparing the stored JSON as text made key order decide whether one person's comment
    was one comment."""
    cid = core.ingest_campaign(conn, title="Peru", detail="Six-week flight.")["campaign_id"]
    store.insert_chunks(conn, cid, ["The timeline concerns me a great deal"],
                        kind="commentary", sources=[{"author": "R. Vega", "anchor": "s4"}])
    store.insert_chunks(conn, cid, ["and the budget will not stretch."],
                        kind="commentary", sources=[{"anchor": "s4", "author": "R. Vega"}])

    assert core.save_evaluation(conn, **_evaluation({
        "severity": "should_fix", "kind": "precedent_departure", "finding": "Raised before",
        "precedent": {"campaign_id": cid, "layer": "commentary", "author": "R. Vega",
                      "quote": "concerns me a great deal and the budget"}}))


@pytest.mark.parametrize("quote", [
    "across Lima and Arequipa [...] Every asset in the flighting table",
    "across Lima and Arequipa […] Every asset in the flighting table",
    "across Lima and Arequipa . . . Every asset in the flighting table",
])
def test_the_ways_a_model_actually_writes_an_elision(conn, peru, quote):
    """`[...]` is routine, and refusing it accused the model of paraphrase for using ordinary
    editorial punctuation."""
    assert core.save_evaluation(conn, **_evaluation({
        "severity": "blocking", "kind": "precedent_departure", "finding": "Departs",
        "precedent": {"campaign_id": peru, "quote": quote}}))["evaluation_id"]


def test_a_fraction_survives_being_normalised(conn):
    """NFKC rewrites "⅓" with a FRACTION SLASH, which is not the "/" anybody types."""
    cid = core.ingest_campaign(conn, title="Budgets",
                               detail="Budget is ⅓ of the Peru total spend.")["campaign_id"]
    assert core.save_evaluation(conn, **_evaluation({
        "severity": "blocking", "kind": "precedent_departure", "finding": "Departs",
        "precedent": {"campaign_id": cid, "quote": "1/3 of the Peru total spend"}}))


def test_the_headline_precedent_is_cleaned_not_echoed(conn, peru):
    """It was stored exactly as sent — so a model could assert its own `verified` beside a
    5,000-character junk field, one level up from the finding where those keys are refused."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(
            conn, closest_precedent={"campaign_id": peru, "verified": True},
            **_evaluation({"severity": "blocking", "kind": "missing_information",
                           "finding": "No end date"}))
    assert "server" in str(e.value)

    saved = core.save_evaluation(
        conn, closest_precedent={"id": peru, "similarity": 0.91, "junk": "x" * 5000},
        **_evaluation({"severity": "blocking", "kind": "missing_information",
                       "finding": "No end date"}))
    stored = store.get_evaluation(conn, saved["evaluation_id"])["closest_precedent"]
    assert stored == {"campaign_id": peru, "layer": "body", "similarity": 0.91}


def test_an_unknown_layer_on_the_headline_precedent_reaches_the_model(conn, peru):
    """It went into `on_file[layer]` unvalidated and came back a KeyError — which
    `_catch_value_errors` does not catch, so the caller got "Error executing tool" with the
    message discarded. That decorator exists for exactly this."""
    import asyncio
    refused = asyncio.run(_call("save_evaluation", {
        "subject_title": "Colombia v2", "verdict": "revise", "summary": "Nothing is dated.",
        "closest_precedent": {"campaign_id": peru, "layer": "hearsay",
                              "quote": "content angle, posting date"},
        "findings": [{"severity": "blocking", "kind": "missing_information",
                      "finding": "No end date"}]}))
    assert "error" in refused, refused
    assert "hearsay" in refused["error"]


# ── third round ─────────────────────────────────────────────────────────────

def test_two_legal_gaps_can_still_add_up_to_an_illegal_quote(conn):
    """The bound is on the TOTAL, and nothing pinned that: reverting it to a per-hop check
    left the whole suite green, because every test with more than one gap is refused by the
    gap COUNT before the arithmetic ever runs."""
    # Each gap has to be comfortably UNDER the limit, or a per-hop check refuses it too and
    # the test passes for the wrong reason — which is exactly what the first version did.
    filler = "Ordinary briefing prose that nobody would ever quote in a finding. "
    deck = core.ingest_campaign(conn, title="Two gaps", detail=(
        "alpha bravo charlie " + filler * 2 + " delta echo foxtrot " + filler * 2 +
        " golf hotel india"))["campaign_id"]
    assert len(filler * 2) < 200, "each gap legal on its own"
    assert len(filler * 4) > 200, "the two of them together are not"

    with pytest.raises(ValueError):
        core.save_evaluation(conn, **_evaluation({
            "severity": "blocking", "kind": "precedent_departure", "finding": "Departs",
            "precedent": {"campaign_id": deck,
                          "quote": "alpha bravo charlie … delta echo foxtrot … "
                                   "golf hotel india"}}))


def test_a_quote_is_matched_where_it_fits_not_where_it_first_appears(conn):
    """A recap slide restating the opening is ordinary in a deck. Locking the span to the
    leftmost occurrence of the first phrase measured it across the whole deck and refused a
    quote that is verbatim and contiguous on one slide."""
    deck = core.ingest_campaign(conn, title="With a recap", deck_text=(
        "Budget is fixed at 40,000 USD for the flight.\n\n"
        + "Ordinary briefing prose for this market.\n\n" * 18 +
        "Key takeaways: Budget is fixed at 40,000 USD for the flight. Nothing goes live "
        "before the embargo."))["campaign_id"]

    assert core.save_evaluation(conn, **_evaluation({
        "severity": "blocking", "kind": "precedent_departure", "finding": "Departs",
        "precedent": {"campaign_id": deck,
                      "quote": "Budget is fixed at 40,000 USD … Nothing goes live before "
                               "the embargo"}}))["evaluation_id"]


@pytest.mark.parametrize("stored,quoted", [
    ("A well-\nknown creator roster is fixed for the flight.", "well-known creator roster"),
    ("A well-\nknown creator roster is fixed for the flight.", "well known creator roster"),
    ("Posting sched-\nule per creator agreed.", "Posting schedule per creator"),
    ("Flight runs 3-\n28 March in Lima.", "3-28 March in Lima"),
])
def test_a_hyphen_at_a_line_end_is_read_both_ways(conn, stored, quoted):
    """No regex can tell "sched-\nule" (one word the layout split) from "well-\nknown" (a real
    hyphen that happened to fall at the break) — and typesetting breaks at an existing hyphen
    first, so the second is the commoner of the two. Joining refuses one, not joining refuses
    the other; the stored text is read both ways."""
    cid = core.ingest_campaign(conn, title="Justified", detail=stored)["campaign_id"]
    assert core.save_evaluation(conn, **_evaluation({
        "severity": "blocking", "kind": "precedent_departure", "finding": "Departs",
        "precedent": {"campaign_id": cid, "quote": quoted}}))["evaluation_id"]


def test_a_forecast_is_not_what_the_campaign_achieved(conn):
    """Metric detail is quotable because it is what a campaign DID. A predicted row is not
    that, and quoting one under `layer: body` records the campaign as stating an outcome it
    only forecast."""
    cid = core.ingest_campaign(conn, title="Jakarta launch",
                               detail="Six-week flight.")["campaign_id"]
    core.add_metrics(conn, campaign_id=cid, metric_type="predicted",
                     detail="We forecast CTR around 4 percent for this one.")

    with pytest.raises(ValueError):
        core.save_evaluation(conn, **_evaluation({
            "severity": "should_fix", "kind": "precedent_departure", "finding": "Departs",
            "precedent": {"campaign_id": cid, "quote": "forecast CTR around 4 percent"}}))


@pytest.mark.parametrize("similarity", ["nan", float("inf"), True, "very high"])
def test_a_similarity_has_to_be_a_real_number(conn, peru, similarity):
    """`float("nan")` serialises as a bare `NaN`, which is not JSON — the row reads back
    broken in any strict consumer. `True` would have been stored as a similarity of 1.0 that
    nobody computed."""
    with pytest.raises(ValueError):
        core.save_evaluation(
            conn, closest_precedent={"campaign_id": peru, "similarity": similarity},
            **_evaluation({"severity": "blocking", "kind": "missing_information",
                           "finding": "No end date"}))
