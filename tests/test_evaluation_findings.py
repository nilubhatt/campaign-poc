"""
Product review defect 07 (P1): "Evaluation output is an essay, because the schema asks for
one." Listed in the review's own top five things to do.

The reviewer's diagnosis is the important part, and it is a data-model argument rather than
a prompting one: "A model handed one string field writes prose into it. Two real evaluations
came back at roughly 700 and 900 words of continuous text — correct, well-sourced, and far
more than a marketer will read before deciding what to do next. The verbosity is not a
prompting problem. It is the data model: there is nowhere structured to put a finding, so
every finding becomes a paragraph, and severity, precedent and fix all dissolve into the
same block of text. Nothing downstream can sort, filter, count or collapse it."

So the fix is to change the shape and let the output follow, with two rules the review was
explicit about: cap `finding` and `fix` in the schema so they cannot grow into paragraphs,
and fix severity to three values so counts are sortable and comparable across evaluations.

Enforced server-side, per consistency idea 3: "Reject writes that omit a precedent quote or
exceed the caps, rather than accepting them and hoping. A validation error is a retry; a
silent acceptance is permanent drift."

Clean cutover — no legacy free-text path, decided with the product owner.
"""
import pytest

import core
import store


def _finding(**over):
    base = {
        "severity": "blocking",
        "category": "timeline",
        "finding": "No posting dates on any deliverable",
        "detail": "Every asset in the flighting table is listed without a date, so nothing "
                  "can be sequenced or held to an embargo.",
        "precedent": {"id": "camp_jdsea", "quote": "content angle, posting date and "
                                                   "requirements per asset"},
        "fix": "Add a posting date per asset to the flighting table",
    }
    base.update(over)
    return base


def _evaluation(**over):
    base = {
        "subject_title": "Colombia v2",
        "verdict": "revise",
        "summary": "Solid activation plan, but nothing is dated and the influencer roster "
                   "has an unresolved conflict.",
        "findings": [_finding()],
    }
    base.update(over)
    return base


def test_an_evaluation_is_stored_as_findings_not_prose(conn):
    result = core.save_evaluation(conn, **_evaluation())

    stored = store.get_evaluation(conn, result["evaluation_id"])
    assert stored["verdict"] == "revise"
    assert len(stored["findings"]) == 1
    assert stored["findings"][0]["severity"] == "blocking"
    assert stored["findings"][0]["precedent"]["quote"]


def test_the_default_response_is_a_fixed_shape_not_an_essay(conn):
    """"Then the default response is a fixed shape: verdict, closest precedent, counts by
    severity, and the one-line ask. Everything else is a follow-up question.\""""
    result = core.save_evaluation(conn, **_evaluation(findings=[
        _finding(severity="blocking"),
        _finding(severity="should_fix", finding="No budget figure anywhere"),
        _finding(severity="note", finding="Consider a second colourway"),
    ]))

    assert result["verdict"] == "revise"
    assert result["counts"] == {"blocking": 1, "should_fix": 1, "note": 1}
    assert result["summary"]
    # Design review overturned the stricter version of this: withholding the findings and
    # telling Claude to offer them put a `note` in competition with the request the user
    # actually made, and cost a round trip to learn what the tool already knew. What must
    # stay out is the LONG form - see test_the_default_response_shows_what_has_to_change.
    assert all("detail" not in f for f in result["findings"]), (
        "the reasoning is fetched on demand; only the actionable lines come back"
    )


def test_severity_is_three_values_so_counts_compare_across_evaluations(conn):
    with pytest.raises(ValueError) as exc:
        core.save_evaluation(conn, **_evaluation(findings=[_finding(severity="critical")]))

    assert "blocking" in str(exc.value) and "should_fix" in str(exc.value)


def test_a_verdict_outside_the_vocabulary_is_rejected(conn):
    with pytest.raises(ValueError):
        core.save_evaluation(conn, **_evaluation(verdict="looks good to me"))


def test_a_finding_cannot_grow_into_a_paragraph(conn):
    """The cap is the mechanism. Without it the model writes prose into the short field and
    the shape stops constraining anything."""
    with pytest.raises(ValueError) as exc:
        core.save_evaluation(conn, **_evaluation(findings=[_finding(finding="x" * 200)]))

    assert "120" in str(exc.value)


def test_a_fix_cannot_grow_into_a_paragraph(conn):
    with pytest.raises(ValueError):
        core.save_evaluation(conn, **_evaluation(findings=[_finding(fix="y" * 200)]))


def test_the_summary_is_capped_too(conn):
    with pytest.raises(ValueError) as exc:
        core.save_evaluation(conn, **_evaluation(summary="z" * 400))

    assert "240" in str(exc.value)


def test_the_long_explanation_still_has_somewhere_to_live(conn):
    """Capping the headline is not banning detail - it moves it somewhere a reader can
    choose to open, instead of making them read it to find the next one."""
    paragraph = ("This matters because "
                 + "the same pattern recurred across six briefs. " * 8).strip()
    result = core.save_evaluation(conn, **_evaluation(findings=[_finding(detail=paragraph)]))

    stored = store.get_evaluation(conn, result["evaluation_id"])
    assert stored["findings"][0]["detail"] == paragraph
    assert len(paragraph) > 240, "comfortably past what the headline fields allow"


def test_an_evaluation_with_no_findings_must_not_be_a_revise(conn):
    """A verdict of revise with nothing to revise is not actionable, and is the shape a
    model falls into when it is hedging."""
    with pytest.raises(ValueError) as exc:
        core.save_evaluation(conn, **_evaluation(verdict="revise", findings=[]))

    assert "finding" in str(exc.value).lower()


def test_approve_with_no_findings_is_fine(conn):
    result = core.save_evaluation(conn, **_evaluation(
        verdict="approve", findings=[],
        summary="Matches the JD SEA structure that worked, dates and roster both clean."))

    assert result["verdict"] == "approve"
    assert result["counts"] == {"blocking": 0, "should_fix": 0, "note": 0}


def test_a_blocking_finding_forces_a_verdict_that_reflects_it(conn):
    """Approving something while recording a blocking finding is internally contradictory,
    and the kind of thing prose let slide because nothing could count it."""
    with pytest.raises(ValueError) as exc:
        core.save_evaluation(conn, **_evaluation(verdict="approve",
                                                 findings=[_finding(severity="blocking")]))

    assert "blocking" in str(exc.value).lower()


def test_findings_are_retrievable_and_sortable(conn):
    """The point of the shape: something downstream can sort, filter and count it."""
    result = core.save_evaluation(conn, **_evaluation(findings=[
        _finding(severity="note", finding="Minor: caption tone"),
        _finding(severity="blocking", finding="No dates on deliverables"),
        _finding(severity="should_fix", finding="No budget stated"),
    ]))

    stored = store.get_evaluation(conn, result["evaluation_id"])
    order = [f["severity"] for f in stored["findings"]]
    assert order == ["blocking", "should_fix", "note"], "most severe first, every time"


def test_resolved_items_carry_what_changed(conn):
    """A v2 brief should be able to say what the v1 evaluation asked for and got."""
    result = core.save_evaluation(conn, **_evaluation(
        resolved=[{"was": "No claw machine in the grand opening",
                   "now": "Claw machine and photo booth both committed on slide 25"}]))

    stored = store.get_evaluation(conn, result["evaluation_id"])
    assert stored["resolved"][0]["now"].startswith("Claw machine")


def test_the_closest_precedent_is_recorded(conn):
    result = core.save_evaluation(conn, **_evaluation(
        closest_precedent={"id": "camp_jdsea", "similarity": 0.91}))

    stored = store.get_evaluation(conn, result["evaluation_id"])
    assert stored["closest_precedent"]["similarity"] == 0.91


# ── fields pulled forward so later items don't each migrate stored evaluations ──
#
# This schema has no callers yet, so defining a field costs nothing today and costs a
# migration of every stored judgment later. The BEHAVIOUR of each still belongs to its own
# item; what lands here is the shape.

def test_a_finding_records_what_kind_of_problem_it_is(conn):
    """Item 6.2: "uses AI-generated imagery" and "does not match how Peru seeded" currently
    come out of the same machinery. The first is binary and non-negotiable; the second is a
    judgment a partner may legitimately argue with. A reader cannot tell them apart from
    severity alone, because severity says how much it matters, not whether it is arguable."""
    result = core.save_evaluation(conn, **_evaluation(findings=[
        _finding(kind="guardrail_breach", finding="Uses AI-generated imagery"),
        _finding(severity="should_fix", kind="precedent_departure",
                 finding="Seeds four colourways where Peru used one"),
    ]))

    stored = store.get_evaluation(conn, result["evaluation_id"])
    assert [f["kind"] for f in stored["findings"]] == ["guardrail_breach",
                                                       "precedent_departure"]


def test_an_unknown_kind_is_rejected(conn):
    with pytest.raises(ValueError) as exc:
        core.save_evaluation(conn, **_evaluation(findings=[_finding(kind="vibes")]))

    assert "guardrail_breach" in str(exc.value)


def test_a_finding_records_whether_it_was_computed_or_judged(conn):
    """Item 7.8: a computed finding should be identical for every user, and a regression
    there is a bug; a judged one carries the evidence behind it. Collapsing both into one
    list is how a date check and an opinion end up looking equally authoritative."""
    result = core.save_evaluation(conn, trusted=True, **_evaluation(findings=[
        _finding(basis="computed", finding="No posting dates on any deliverable"),
        _finding(severity="note", basis="judged",
                 finding="The claw machine may outperform the precedent"),
    ]))

    stored = store.get_evaluation(conn, result["evaluation_id"])
    assert [f["basis"] for f in stored["findings"]] == ["computed", "judged"]


def test_basis_defaults_to_judged_rather_than_claiming_to_be_computed(conn):
    """The safe default: anything the model asserted is a judgment unless the server itself
    computed it. Defaulting the other way would let opinions inherit the authority of a
    mechanical check."""
    result = core.save_evaluation(conn, **_evaluation())

    stored = store.get_evaluation(conn, result["evaluation_id"])
    assert stored["findings"][0]["basis"] == "judged"


def test_an_evaluation_can_carry_its_exit_condition(conn):
    """Item 6.5: every revise should say what would flip it to approve - testable, and it
    doubles as the note the partner receives."""
    result = core.save_evaluation(conn, **_evaluation(
        approve_if="Dates added to every deliverable and the two adidas-affiliated "
                   "profiles removed from the roster."))

    stored = store.get_evaluation(conn, result["evaluation_id"])
    assert stored["approve_if"].startswith("Dates added")


def test_an_evaluation_can_carry_the_strength_of_its_evidence(conn):
    """Item 6.6: a verdict resting on five concluded campaigns with verified outcomes and
    one resting on a single proposed brief currently look identical."""
    result = core.save_evaluation(conn, **_evaluation(
        evidence={"precedents": 5, "concluded": 4, "verified_outcomes": 2,
                  "top_similarity": 0.91}))

    stored = store.get_evaluation(conn, result["evaluation_id"])
    assert stored["evidence"]["verified_outcomes"] == 2


def test_an_evaluation_can_carry_what_produced_it(conn):
    """Item 7.6: without version stamps, two evaluations of the same brief are
    indistinguishable in the record, so drift cannot even be detected."""
    result = core.save_evaluation(conn, **_evaluation(
        provenance={"rulebook_version": "2026.09", "embedding_model": "nomic-embed-text",
                    "server_version": "0.2.7"}))

    stored = store.get_evaluation(conn, result["evaluation_id"])
    assert stored["provenance"]["rulebook_version"] == "2026.09"


# ══ adversarial + design review of 2.4 ═══════════════════════════════════════
#
# Both reviewers independently reproduced the same blocker, which is the strongest signal
# either produced: the cutover is fine on a fresh database and impossible on a real one.

def test_an_existing_database_can_still_save_an_evaluation(tmp_path):
    """The clean cutover was agreed on the basis that there are no production customers -
    but v0.2.0 shipped with installers and the reviewer's own machine holds the two essays
    that ARE defect 07. `ALTER TABLE ADD COLUMN` cannot relax the legacy
    `analysis TEXT NOT NULL`, so after migrating, every new save died on an IntegrityError -
    which is not a ValueError, so the marketer saw 'Error executing tool save_evaluation'
    with the reason discarded."""
    import sqlite3
    import time

    path = tmp_path / "v020.db"
    legacy = sqlite3.connect(path)
    legacy.executescript("""
        CREATE TABLE campaigns (id TEXT PRIMARY KEY, title TEXT NOT NULL,
                                collection TEXT, markets TEXT NOT NULL DEFAULT '[]');
        CREATE TABLE evaluations (
            id TEXT PRIMARY KEY, campaign_id TEXT, subject_title TEXT NOT NULL,
            cited_ids TEXT, analysis TEXT NOT NULL, predictions TEXT,
            created_at REAL NOT NULL);
        CREATE TABLE reconciliations (
            id TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL, actual TEXT,
            comparison TEXT NOT NULL, created_at REAL NOT NULL);
    """)
    legacy.execute("INSERT INTO evaluations VALUES ('ev_old', NULL, 'Colombia v1', '[]', ?, "
                   "NULL, ?)", ("A 900-word essay on the Colombia brief. " * 40, time.time()))
    legacy.commit()
    legacy.close()

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    store._migrate_schema(conn)

    result = core.save_evaluation(conn, **_evaluation())
    assert store.get_evaluation(conn, result["evaluation_id"])["verdict"] == "revise"


def test_an_upgrade_keeps_the_judgments_already_on_the_machine(tmp_path):
    """Rebuilding the table to drop a NOT NULL is the one migration that can lose data, so
    it gets its own test: the essays are the evidence for defect 07 and the record of what
    the library was told before the schema changed."""
    import sqlite3
    import time

    path = tmp_path / "v020.db"
    legacy = sqlite3.connect(path)
    legacy.executescript("""
        CREATE TABLE campaigns (id TEXT PRIMARY KEY, title TEXT NOT NULL,
                                collection TEXT, markets TEXT NOT NULL DEFAULT '[]');
        CREATE TABLE evaluations (
            id TEXT PRIMARY KEY, campaign_id TEXT, subject_title TEXT NOT NULL,
            cited_ids TEXT, analysis TEXT NOT NULL, predictions TEXT,
            created_at REAL NOT NULL);
        CREATE TABLE reconciliations (
            id TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL, actual TEXT,
            comparison TEXT NOT NULL, created_at REAL NOT NULL);
    """)
    legacy.execute("INSERT INTO evaluations VALUES ('ev_old', NULL, 'Colombia v1', "
                   "'[\"camp_x\"]', 'the original essay', '{\"ctr\": 1}', ?)", (time.time(),))
    legacy.commit()
    legacy.close()

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    store._migrate_schema(conn)

    row = store.get_evaluation(conn, "ev_old")
    assert row["analysis"] == "the original essay"
    assert row["cited_ids"] == ["camp_x"]
    assert row["predictions"] == {"ctr": 1}


def test_a_pre_cutover_judgment_is_readable_rather_than_silently_empty(conn):
    """A legacy row has no verdict and no findings, so reconciliation ran against an empty
    original and said nothing was originally found - a confident answer built on a record it
    had silently dropped. The essay is still there; hand it back and say what it is."""
    conn.execute("INSERT INTO evaluations (id, subject_title, analysis, created_at) "
                 "VALUES ('ev_legacy', 'Colombia v1', 'the original essay', 1.0)")
    conn.commit()

    out = core.reconcile_evaluation(conn, evaluation_id="ev_legacy", actual="CTR 1.2%")

    assert out["original_analysis"] == "the original essay"
    assert "structured" in out.get("note", "").lower() or out.get("schema") == "legacy"


def test_a_non_string_finding_is_a_readable_error_not_a_crash(conn):
    """Only ValueError reaches the caller as words; an AttributeError from .strip() on a
    list reached the marketer as 'Error executing tool save_evaluation'."""
    for bad in (["a", "b"], 42, {"nested": 1}):
        with pytest.raises(ValueError) as exc:
            core.save_evaluation(conn, **_evaluation(findings=[_finding(finding=bad)]))
        assert "finding" in str(exc.value)


def test_findings_must_be_a_list_not_a_single_finding(conn):
    """A dict was iterated as keys, producing "finding 0 must be an object, got str" - an
    error message about the wrong thing entirely."""
    with pytest.raises(ValueError) as exc:
        core.save_evaluation(conn, **_evaluation(findings=_finding()))

    assert "list" in str(exc.value).lower()


def test_the_model_cannot_claim_a_finding_was_computed(conn):
    """§7.8's premise - "a computed finding is identical for every user, and a difference
    there is a bug" - only holds if the server stamped it. A model that read a missing date
    did not compute it. The column stays; the MCP-facing path may only assert 'judged'."""
    with pytest.raises(ValueError) as exc:
        core.save_evaluation(conn, **_evaluation(findings=[_finding(basis="computed")]))

    assert "server" in str(exc.value).lower()


def test_the_server_can_still_stamp_a_computed_finding(conn):
    """§7.1 will produce these; the path has to exist, it just isn't the model's."""
    result = core.save_evaluation(conn, trusted=True, **_evaluation(findings=[
        _finding(basis="computed", finding="No posting dates on any deliverable")]))

    stored = store.get_evaluation(conn, result["evaluation_id"])
    assert stored["findings"][0]["basis"] == "computed"


def test_the_findings_can_be_read_back_later(conn):
    """"The detail is fetched on demand" was true of the storage and false of the tool
    surface: there was no read path at all, so 'show me the blocking items' worked only
    while the findings were still in the context window that produced them."""
    result = core.save_evaluation(conn, **_evaluation(findings=[
        _finding(severity="blocking"),
        _finding(severity="note", finding="Consider a second colourway"),
    ]))

    got = core.get_evaluation(conn, evaluation_id=result["evaluation_id"],
                              severity="blocking")

    assert [f["severity"] for f in got["findings"]] == ["blocking"]
    assert got["verdict"] == "revise"


def test_reading_back_an_unknown_evaluation_says_so(conn):
    assert "not found" in core.get_evaluation(conn, evaluation_id="ev_nope")["error"]


def test_the_default_response_shows_what_has_to_change(conn):
    """A `note` telling Claude to offer the findings competes with the user's actual request
    and loses; models mirror the shape of a tool result far more reliably than they follow
    instructions inside one. The capped one-liners are bounded by construction, so handing
    back the actionable ones cannot reconstitute the essay."""
    result = core.save_evaluation(conn, **_evaluation(findings=[
        _finding(severity="blocking"),
        _finding(severity="should_fix", finding="No budget figure anywhere"),
        _finding(severity="note", finding="Consider a second colourway"),
    ]))

    shown = result["findings"]
    assert [f["severity"] for f in shown] == ["blocking", "should_fix"], (
        "the two that need action; notes stay in the counts"
    )
    assert "detail" not in shown[0], "the long form is still fetched on demand"


def test_a_precedent_can_cite_a_rule_not_only_a_campaign(conn):
    """A guardrail breach is anchored to a rule in the rulebook (phase 12), not to a
    campaign - so with only a campaign id there was nowhere to cite the thing breached."""
    result = core.save_evaluation(conn, **_evaluation(findings=[
        _finding(kind="guardrail_breach", finding="Uses AI-generated imagery",
                 precedent={"rule_id": "no_ai_imagery",
                            "quote": "No AI-generated imagery in any paid placement"})]))

    stored = store.get_evaluation(conn, result["evaluation_id"])
    assert stored["findings"][0]["precedent"]["rule_id"] == "no_ai_imagery"


def test_a_precedent_that_cites_nothing_is_rejected(conn):
    with pytest.raises(ValueError) as exc:
        core.save_evaluation(conn, **_evaluation(findings=[
            _finding(precedent={"quote": "something someone said"})]))

    assert "campaign_id" in str(exc.value) or "rule_id" in str(exc.value)


def test_a_guardrail_breach_cannot_be_a_note(conn):
    """A rule either applies or it does not. "You broke a rule, but never mind" is the shape
    of a finding written to avoid an argument."""
    with pytest.raises(ValueError) as exc:
        core.save_evaluation(conn, **_evaluation(findings=[
            _finding(severity="note", kind="guardrail_breach")]))

    assert "guardrail" in str(exc.value).lower()


def test_a_revise_needs_something_above_a_note(conn):
    """The original rule counted findings of any severity, so three notes satisfied a
    'revise' - the same hedge one level down."""
    with pytest.raises(ValueError) as exc:
        core.save_evaluation(conn, **_evaluation(verdict="revise", findings=[
            _finding(severity="note", finding="Consider a second colourway")]))

    assert "note" in str(exc.value).lower()


def test_the_approve_conflict_message_does_not_offer_downgrading_as_an_option(conn):
    """"either the verdict is 'revise' or the finding is not blocking" offered both exits as
    equals, and one of them is a single token while the other means rewriting the summary.
    A validation error is a retry only if the honest path is the one it points at."""
    with pytest.raises(ValueError) as exc:
        core.save_evaluation(conn, **_evaluation(verdict="approve",
                                                 findings=[_finding(severity="blocking")]))

    message = str(exc.value).lower()
    assert "revise" in message
    assert "do not lower" in message or "not by lowering" in message


def test_every_free_text_field_is_bounded(conn):
    """Capping the headline and leaving five neighbours unbounded moves the essay, it does
    not prevent it: 900 words fits comfortably as six findings with a 150-word detail."""
    for field, value in (("detail", "d" * 5000), ("category", "c" * 200)):
        with pytest.raises(ValueError, match=r"\d+"):
            core.save_evaluation(conn, **_evaluation(findings=[_finding(**{field: value})]))

    with pytest.raises(ValueError):
        core.save_evaluation(conn, **_evaluation(findings=[
            _finding(precedent={"campaign_id": "camp_x", "quote": "q" * 600})]))
    with pytest.raises(ValueError):
        core.save_evaluation(conn, **_evaluation(approve_if="a" * 600))
    with pytest.raises(ValueError):
        core.save_evaluation(conn, **_evaluation(
            resolved=[{"was": "w" * 400, "now": "fixed"}]))


def test_quantity_cannot_substitute_for_length(conn):
    """Thirty capped findings is an essay assembled from bricks."""
    with pytest.raises(ValueError) as exc:
        core.save_evaluation(conn, **_evaluation(
            findings=[_finding(finding=f"Problem {i}") for i in range(30)]))

    assert "12" in str(exc.value)


def test_resolved_must_say_what_was_and_what_is_now(conn):
    for bad in ("just a string", 42, [{"was": "x"}]):
        with pytest.raises(ValueError):
            core.save_evaluation(conn, **_evaluation(
                resolved=bad if isinstance(bad, list) else [bad]))


def test_resolved_can_point_at_the_finding_it_closed(conn):
    """§5.4 and §6.3 both need to say "this specific earlier finding is now addressed", and
    free text cannot be joined back to one. Ids are free now and a migration later."""
    first = core.save_evaluation(conn, **_evaluation())
    stored = store.get_evaluation(conn, first["evaluation_id"])
    finding_id = stored["findings"][0]["id"]

    second = core.save_evaluation(conn, **_evaluation(
        subject_title="Colombia v3", verdict="approve", findings=[],
        summary="Dates added throughout; the roster conflict is gone.",
        resolved=[{"finding_id": finding_id, "was": "No posting dates",
                   "now": "Dates on all 14 deliverables"}]))

    assert store.get_evaluation(conn, second["evaluation_id"])["resolved"][0]["finding_id"] \
        == finding_id


def test_findings_are_identified_stably_within_an_evaluation(conn):
    result = core.save_evaluation(conn, **_evaluation(findings=[
        _finding(severity="note", finding="Minor"),
        _finding(severity="blocking", finding="Major"),
    ]))

    ids = [f["id"] for f in store.get_evaluation(conn, result["evaluation_id"])["findings"]]
    assert len(set(ids)) == 2
    assert all(i.startswith(result["evaluation_id"]) for i in ids)


def test_listing_evaluations_shows_the_verdict(conn):
    """A list of subject titles and dates cannot answer "which of these still need work",
    which is the whole point of having a verdict."""
    core.save_evaluation(conn, **_evaluation())

    row = store.list_evaluations(conn)[0]
    assert row["verdict"] == "revise"
    assert row["counts"]["blocking"] == 1


def test_the_caps_are_measured_the_same_way_everywhere(conn):
    """`finding` was measured after stripping and `summary` before, so trailing whitespace
    was fatal in one field and free in another."""
    core.save_evaluation(conn, **_evaluation(summary="z" * 240 + "   "))

    with pytest.raises(ValueError):
        core.save_evaluation(conn, **_evaluation(summary="z" * 241))


def test_findings_of_equal_severity_keep_the_order_they_were_written_in(conn):
    """The sort is part of the contract, so its tie-breaking is too: a model that ordered
    two blocking findings deliberately should not have them reshuffled."""
    result = core.save_evaluation(conn, **_evaluation(findings=[
        _finding(severity="blocking", finding="First blocking"),
        _finding(severity="note", finding="A note"),
        _finding(severity="blocking", finding="Second blocking"),
    ]))

    order = [f["finding"] for f in store.get_evaluation(conn, result["evaluation_id"])["findings"]]
    assert order[:2] == ["First blocking", "Second blocking"]
