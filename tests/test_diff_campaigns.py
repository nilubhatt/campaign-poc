"""
Idea D: "Diff two versions of the same brief." The reviewer's own assessment: "the single
most common real task this product faces" and "the highest-value feature not currently on
the list."

    "Briefs arrive as v1, then v2. Working out what actually changed between the Colombia
    versions — which corrections were adopted, which were ignored, which facts went stale —
    was done by hand.

    Fix: `diff_campaigns(a, b)` returning adopted, ignored, newly-introduced and
    carried-stale, computed against the earlier version's evaluation findings."

"Computed against the earlier version's evaluation findings" is the design, and it is what
makes this a computation rather than a second judgment. Once both versions have been
evaluated, the comparison is between two sets of findings:

  adopted           a v1 finding the v2 evaluation explicitly resolved (`finding_id`, §2.4)
  raised_again      a v1 finding the v2 evaluation raised too. NOT called "ignored": that
                    imputes a choice nobody recorded, and where the match is on wording
                    rather than on an id it is a resemblance, not a fact
  newly_introduced  a v2 finding with no counterpart in v1
  no_longer_raised  neither resolved nor re-raised — and that is genuinely ambiguous: it
                    was either fixed without being recorded, or the second review did not
                    look. Reported as its own bucket rather than quietly counted as adopted
  carried_stale     a citation to a record that has since been superseded

Every entry says whether it is `computed` (matched by id) or `judged` (matched on wording). Where a version has never been
evaluated there is nothing to compute, and the tool says so rather than guessing — which is
the one thing a diff must not do, since "you ignored my correction" is an accusation.
"""
import pytest

import core
import store


def _finding(text, **over):
    # §6.2: `kind` is required on every finding. These tests are about matching findings
    # across two versions of a brief, not about classifying them — `missing_information` is
    # the kind that needs no citation, so the fixtures stay about what they are testing.
    base = {"severity": "blocking", "kind": "missing_information",
            "category": "timeline", "finding": text,
            # §6.5: a finding above a note says what to do about it.
            "fix": "Fix it"}
    base.update(over)
    return base


def _version(conn, title, **kw):
    kw.setdefault("detail", f"the {title} brief")
    return core.ingest_campaign(conn, title=title, confirm=True, **kw)["campaign_id"]


def _judge(conn, campaign_id, findings, **over):
    payload = {"subject_title": "a version", "verdict": "revise" if findings else "approve",
               "summary": "A summary of what this version got right and wrong.",
               "campaign_id": campaign_id, "findings": findings}
    # §6.5: a revise says what would end it, an approve may not. Neither is what this file
    # tests, so the helper tracks the verdict rather than every call site restating it.
    if payload["verdict"] == "revise":
        payload["approve_if"] = "The findings above are addressed."
    payload.update(over)
    if payload.get("verdict") != "revise":
        payload.pop("approve_if", None)
    return core.save_evaluation(conn, **payload)


# ── the four the review named ───────────────────────────────────────────────

def test_a_correction_that_was_adopted_is_reported_as_adopted(conn):
    """The mechanism §2.4 built for exactly this: `resolved` carries the `finding_id` of the
    earlier finding it closes, so "they did what we asked" is a fact rather than an
    impression."""
    v1 = _version(conn, "Colombia v1")
    first = _judge(conn, v1, [_finding("No posting dates on any deliverable")])
    finding_id = store.get_evaluation(conn, first["evaluation_id"])["findings"][0]["id"]

    v2 = _version(conn, "Colombia v2")
    _judge(conn, v2, [], verdict="approve",
           resolved=[{"finding_id": finding_id, "was": "No posting dates",
                      "now": "Dates on all 14 deliverables"}])

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)

    assert [f["finding"] for f in diff["adopted"]] == ["No posting dates on any deliverable"]
    assert diff["adopted"][0]["now"] == "Dates on all 14 deliverables"
    assert diff["adopted"][0]["basis"] == "computed"


def test_a_correction_raised_again_is_reported_as_such(conn):
    """The one the reviewer did by hand. Named `raised_again` rather than `ignored`: what the
    record supports is that the later review raised it too, and "ignored" imputes a choice
    nobody recorded — see test_text_similarity_never_earns_the_word_ignored."""
    v1 = _version(conn, "Colombia v1")
    first = _judge(conn, v1, [_finding("No posting dates on any deliverable")])
    finding_id = store.get_evaluation(conn, first["evaluation_id"])["findings"][0]["id"]

    v2 = _version(conn, "Colombia v2")
    _judge(conn, v2, [_finding("No posting dates on any deliverable",
                               repeats=finding_id)])

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)

    assert [f["finding"] for f in diff["raised_again"]] == \
        ["No posting dates on any deliverable"]
    assert diff["raised_again"][0]["basis"] == "computed"


def test_the_same_problem_worded_differently_is_still_the_same_problem(conn):
    """Two reviews of the same defect rarely phrase it identically. Matching on exact text
    would report every ignored correction as a new one, which is the most flattering
    possible error."""
    v1 = _version(conn, "Colombia v1")
    _judge(conn, v1, [_finding("No posting dates on any deliverable")])

    v2 = _version(conn, "Colombia v2")
    _judge(conn, v2, [_finding("No posting date on any of the deliverables")])

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)

    assert len(diff["raised_again"]) == 1
    assert diff["raised_again"][0]["match"] == "text", "wording is a fallback, and says so"
    assert diff["newly_introduced"] == []


def test_a_problem_only_the_later_version_has_is_newly_introduced(conn):
    v1 = _version(conn, "Colombia v1")
    _judge(conn, v1, [_finding("No posting dates on any deliverable")])

    v2 = _version(conn, "Colombia v2")
    _judge(conn, v2, [_finding("Two influencers are adidas-affiliated",
                               category="influencer")])

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)

    assert [f["finding"] for f in diff["newly_introduced"]] == \
        ["Two influencers are adidas-affiliated"]


def test_a_finding_neither_resolved_nor_repeated_is_not_called_adopted(conn):
    """It was either fixed without being recorded, or the second review did not look. Those
    are different things and the record cannot tell them apart, so counting it as adopted
    would credit the brief for work nobody verified."""
    v1 = _version(conn, "Colombia v1")
    _judge(conn, v1, [_finding("No posting dates on any deliverable"),
                      _finding("No budget figure anywhere", category="budget")])

    v2 = _version(conn, "Colombia v2")
    _judge(conn, v2, [_finding("No posting dates on any deliverable")])

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)

    assert diff["adopted"] == []
    assert [f["finding"] for f in diff["no_longer_raised"]] == ["No budget figure anywhere"]
    caveat = diff["no_longer_raised"][0]["caveat"].lower()
    assert "either" in caveat and "cannot tell" in caveat, (
        f"the caveat must say the record cannot distinguish the two cases: {caveat}"
    )


def test_a_citation_to_a_replaced_record_has_gone_stale(conn):
    """"which facts went stale" — computable where it is a citation: a judgment that rested
    on a campaign since replaced rested on something the library no longer treats as
    current."""
    old_precedent = _version(conn, "Peru v1")
    v1 = _version(conn, "Colombia v1")
    _judge(conn, v1, [_finding("No posting dates")], cited_ids=[old_precedent])

    store.insert_campaign(conn, title="Peru v2", supersedes=old_precedent)

    v2 = _version(conn, "Colombia v2")
    _judge(conn, v2, [_finding("No posting dates")])

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)

    stale = diff["carried_stale"]
    assert [s["campaign_id"] for s in stale] == [old_precedent]
    assert stale[0]["superseded_by"]


# ── what it must not do ─────────────────────────────────────────────────────

def test_an_unjudged_version_is_said_to_be_unjudged_rather_than_guessed_at(conn):
    """"You ignored my correction" is an accusation. With no evaluation on one side there is
    nothing to compute, and inventing the comparison from the deck text would be a judgment
    presented as a computation."""
    v1 = _version(conn, "Colombia v1")
    _judge(conn, v1, [_finding("No posting dates on any deliverable")])
    v2 = _version(conn, "Colombia v2")

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)

    assert diff["adopted"] == [] and diff["raised_again"] == []
    assert diff["comparable"] is False
    assert "later" in diff["why_not_comparable"] or "v2" in diff["why_not_comparable"]
    assert diff["next_actions"][0]["tool"] == "prepare_evaluation"


def test_both_versions_unjudged_says_so_once(conn):
    v1 = _version(conn, "Colombia v1")
    v2 = _version(conn, "Colombia v2")

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)

    assert diff["comparable"] is False
    assert diff["next_actions"]


def test_a_missing_campaign_is_an_error_not_an_empty_diff(conn):
    v1 = _version(conn, "Colombia v1")

    assert "not found" in core.diff_campaigns(conn, earlier=v1, later="camp_nope")["error"]


def test_the_order_of_the_arguments_is_not_guessed(conn):
    """Which version is earlier decides what "adopted" means, so getting it backwards
    inverts the whole answer. The record knows: one supersedes the other, or one was created
    first."""
    v1 = _version(conn, "Colombia v1")
    _judge(conn, v1, [_finding("No posting dates on any deliverable")])
    v2 = _version(conn, "Colombia v2", supersedes=v1)
    _judge(conn, v2, [_finding("No posting dates on any deliverable")])

    backwards = core.diff_campaigns(conn, earlier=v2, later=v1)

    assert backwards["earlier"]["campaign_id"] == v1, "corrected from the supersession"
    assert backwards["arguments_reordered"] is True


def test_diffing_a_campaign_against_itself_is_refused(conn):
    v1 = _version(conn, "Colombia v1")

    assert "same" in core.diff_campaigns(conn, earlier=v1, later=v1)["error"].lower()


def test_the_most_recent_judgment_of_each_version_is_the_one_compared(conn):
    """A version can be judged more than once — that is what a revised evaluation is. The
    comparison has to use the latest of each, or it reports corrections that were already
    dealt with."""
    v1 = _version(conn, "Colombia v1")
    _judge(conn, v1, [_finding("No posting dates on any deliverable")])
    _judge(conn, v1, [_finding("No budget figure anywhere", category="budget")])

    v2 = _version(conn, "Colombia v2")
    _judge(conn, v2, [_finding("No budget figure anywhere", category="budget")])

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)

    assert [f["finding"] for f in diff["raised_again"]] == ["No budget figure anywhere"]


def test_the_counts_add_up(conn):
    """Every finding on both sides is accounted for in exactly one bucket, or something has
    been silently dropped."""
    v1 = _version(conn, "Colombia v1")
    _judge(conn, v1, [_finding("No posting dates"), _finding("No budget", category="budget"),
                      _finding("Roster conflict", category="influencer")])
    v2 = _version(conn, "Colombia v2")
    _judge(conn, v2, [_finding("No posting dates"),
                      _finding("AI imagery on slide 9", category="compliance")])

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)

    earlier_accounted = (len(diff["adopted"]) + len(diff["raised_again"])
                         + len(diff["no_longer_raised"]))
    assert earlier_accounted == 3
    assert len(diff["raised_again"]) + len(diff["newly_introduced"]) == 2


# ══ design review of 5.4 ═════════════════════════════════════════════════════

def test_judging_a_new_version_is_handed_the_earlier_versions_findings(conn):
    """The finding that reframes the item: `adopted` had no feeder. It is populated only
    from `resolved[].finding_id`, and nothing in the product ever handed Claude the earlier
    version's finding ids at the moment it was judging v2 — `prepare_evaluation` took only a
    title and some text, and `find_similar` excludes superseded records, so v1's judgment was
    invisible in v2's evidence. Every fixed finding therefore landed in `no_longer_raised`
    and `adopted` was empty in production. The test suite missed it because the fixture reads
    the id out of the store and writes it in by hand."""
    v1 = _version(conn, "Colombia v1")
    _judge(conn, v1, [_finding("No posting dates on any deliverable")])
    v2 = _version(conn, "Colombia v2", supersedes=v1)

    packaged = core.prepare_evaluation(conn, subject_title="Colombia v2",
                                       proposal_text="the revised brief", campaign_id=v2)

    earlier = packaged["earlier_version"]
    assert earlier["campaign_id"] == v1
    assert earlier["findings"][0]["id"]
    assert earlier["findings"][0]["finding"] == "No posting dates on any deliverable"
    assert "finding_id" in packaged["note"] or "resolved" in packaged["note"]


def test_a_version_with_no_earlier_judgment_is_handed_nothing(conn):
    v2 = _version(conn, "Colombia v2")

    packaged = core.prepare_evaluation(conn, subject_title="Colombia v2",
                                       proposal_text="a brief", campaign_id=v2)

    assert packaged["earlier_version"] is None


def test_a_later_finding_can_say_which_earlier_one_it_repeats(conn):
    """The other half of the feeder. "This is the same problem as before" is a claim the
    model can make precisely, and it is the only way `raised_again` can be stated as fact
    rather than guessed from wording."""
    v1 = _version(conn, "Colombia v1")
    first = _judge(conn, v1, [_finding("No posting dates on any deliverable")])
    finding_id = store.get_evaluation(conn, first["evaluation_id"])["findings"][0]["id"]

    v2 = _version(conn, "Colombia v2")
    _judge(conn, v2, [_finding("Deliverables still undated", repeats=finding_id)])

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)

    assert len(diff["raised_again"]) == 1
    assert diff["raised_again"][0]["match"] == "id"


def test_text_similarity_never_earns_the_word_ignored(conn):
    """Measured, not supposed. Character similarity scores "Two influencers are
    adidas-affiliated" against "...Nike-affiliated" at 0.89 and "No budget line for Colombia
    paid media" against "...influencer fees" at 0.77 — different problems, reported as a
    correction somebody ignored, on a surface the marketer carries to their agency. And it
    scores the SAME problem reworded ("Deliverables still undated") at 0.36, which is the
    flattering miss in the other direction.

    So a text match is reported as what was observed — these two read alike — and is marked
    so nothing downstream can state it as fact."""
    v1 = _version(conn, "Colombia v1")
    _judge(conn, v1, [_finding("Two influencers are adidas-affiliated",
                               category="influencer")])
    v2 = _version(conn, "Colombia v2")
    _judge(conn, v2, [_finding("Two influencers are Nike-affiliated",
                               category="influencer")])

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)

    assert "ignored" not in diff, "the accusation is not a computed bucket"
    if diff["raised_again"]:
        entry = diff["raised_again"][0]
        assert entry["match"] == "text"
        assert entry["basis"] == "judged", "a resemblance in wording is not a computed fact"
        assert 0 < entry["similarity"] < 1


def test_an_identified_repeat_is_a_computed_fact(conn):
    v1 = _version(conn, "Colombia v1")
    first = _judge(conn, v1, [_finding("No posting dates on any deliverable")])
    finding_id = store.get_evaluation(conn, first["evaluation_id"])["findings"][0]["id"]
    v2 = _version(conn, "Colombia v2")
    _judge(conn, v2, [_finding("Still nothing dated", repeats=finding_id)])

    entry = core.diff_campaigns(conn, earlier=v1, later=v2)["raised_again"][0]

    assert entry["basis"] == "computed"
    assert entry["match"] == "id"


def test_the_order_is_only_corrected_on_evidence_somebody_recorded(conn):
    """Creation order is UPLOAD order. An organisation seeding its archive uploads v1 after
    v2 as a matter of course, and silently reversing their question — with a JSON field they
    never see as the only tell — answers something they did not ask."""
    v2 = _version(conn, "Colombia v2")
    v1 = _version(conn, "Colombia v1")
    _judge(conn, v1, [_finding("No posting dates")])
    _judge(conn, v2, [_finding("No posting dates")])

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)

    assert diff["earlier"]["campaign_id"] == v1, "the order given is kept"
    assert diff["order_basis"] == "as_given"
    assert any(w["code"] == "version_order_unverified" for w in diff["warnings"])


def test_a_recorded_supersession_still_corrects_the_order(conn):
    v1 = _version(conn, "Colombia v1")
    _judge(conn, v1, [_finding("No posting dates")])
    v2 = _version(conn, "Colombia v2", supersedes=v1)
    _judge(conn, v2, [_finding("No posting dates")])

    diff = core.diff_campaigns(conn, earlier=v2, later=v1)

    assert diff["earlier"]["campaign_id"] == v1
    assert diff["order_basis"] == "supersession"
    assert diff["arguments_reordered"] is True


def test_what_the_record_says_about_each_version_is_compared_too(conn):
    """A v2 that drops a market from a LATAM brief is the largest change a version can carry,
    and it diffed as identical. Same for a performance tag that lost its `verified` source —
    which changes how the record weighs as precedent."""
    v1 = _version(conn, "Colombia v1", markets=["Colombia", "Peru"], status="proposed",
                  tags=[{"value": "liked", "source": "stated"}], collection="Q4 launch")
    _judge(conn, v1, [_finding("No posting dates")])
    v2 = _version(conn, "Colombia v2", markets=["Peru"], status="proposed",
                  tags=[{"value": "mixed_reaction"}], collection="Q4 launch")
    _judge(conn, v2, [_finding("No posting dates")])

    changes = core.diff_campaigns(conn, earlier=v1, later=v2)["record_changes"]

    assert changes["markets"]["removed"] == ["Colombia"]
    assert changes["tags"]["removed"] == ["liked"]
    assert changes["tags"]["added"] == ["mixed_reaction"]
    assert "collection" not in changes, "unchanged fields are not listed"


def test_a_judgment_from_before_the_findings_schema_is_not_comparable(conn):
    """A pre-2.4 evaluation reads back with empty `findings`, so it was treated as comparable
    and every later finding came back as newly introduced — a v2 blamed for everything its
    v1 review had also found."""
    v1 = _version(conn, "Colombia v1")
    conn.execute("INSERT INTO evaluations (id, campaign_id, subject_title, analysis, "
                 "created_at) VALUES ('ev_legacy', ?, 'Colombia v1', 'an essay', 1.0)",
                 (v1,))
    conn.commit()
    v2 = _version(conn, "Colombia v2")
    _judge(conn, v2, [_finding("No posting dates")])

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)

    assert diff["comparable"] is False
    assert diff["newly_introduced"] == []
    assert "structured" in diff["why_not_comparable"].lower()


# ══ adversarial review of 5.4 ════════════════════════════════════════════════

def test_which_correction_is_accused_does_not_depend_on_typing_order(conn):
    """Greedy matching took each earlier finding in turn and gave it the best remaining
    later one — so with two earlier findings competing for one later finding, the one
    processed FIRST won, even at 0.79 against the other's 0.98. Which correction gets
    reported as raised again then depends on the order somebody typed them, or on the
    severity sort that reorders them before ids are assigned. The same two decks must not
    diff differently."""
    later_text = "Budget is absent for paid media entirely"
    weak = _finding("Budget is absent for paid media", category="budget")
    strong = _finding("Budget is absent for paid media entirely!", category="budget")

    results = []
    for order in ([weak, strong], [strong, weak]):
        fresh = _version(conn, f"v1 {len(results)}")
        _judge(conn, fresh, order)
        v2 = _version(conn, f"v2 {len(results)}")
        _judge(conn, v2, [_finding(later_text, category="budget")])
        diff = core.diff_campaigns(conn, earlier=fresh, later=v2)
        results.append([f["finding"] for f in diff["raised_again"]])

    assert results[0] == results[1], results
    assert results[0] == [strong["finding"]], "the best available pairing, not the first"


def test_a_later_finding_is_never_matched_to_two_earlier_ones(conn):
    """Mutation-proof: dropping the bookkeeping that stops a later finding being reused left
    the suite green, because the counts test never checked one-to-one assignment."""
    v1 = _version(conn, "Colombia v1")
    _judge(conn, v1, [_finding("No posting dates on any deliverable"),
                      _finding("No posting dates on any deliverables")])
    v2 = _version(conn, "Colombia v2")
    _judge(conn, v2, [_finding("No posting dates on any deliverable")])

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)

    assert len(diff["raised_again"]) == 1
    assert len(diff["no_longer_raised"]) == 1


def test_agreeing_on_the_category_breaks_a_tie_in_wording(conn):
    """Mutation-proof: removing the category bonus left the suite green. It is the one signal
    that distinguishes two findings whose wording is equally close."""
    v1 = _version(conn, "Colombia v1")
    _judge(conn, v1, [_finding("Dates are missing", category="timeline")])
    v2 = _version(conn, "Colombia v2")
    _judge(conn, v2, [_finding("Dates are missing", category="budget"),
                      _finding("Dates are missing.", category="timeline")])

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)

    assert diff["raised_again"][0]["raised_again_as"] == "Dates are missing."


def test_the_offer_to_evaluate_uses_the_text_there_actually_is(conn):
    """`proposal_text=(detail or title)` — so for a deck-only upload the offer sent the
    eleven characters of the title, and accepting the label "evaluate this version" judged
    nothing. 5.2's rule is that a prefilled argument is never a placeholder; the mechanical
    tests passed because the argument existed and the call succeeded."""
    v1 = _version(conn, "Colombia v1")
    _judge(conn, v1, [_finding("No posting dates")])
    v2 = core.ingest_campaign(conn, title="Colombia v2", deck_text="the full revised brief, "
                              "at length, with all of its sections", confirm=True)

    diff = core.diff_campaigns(conn, earlier=v1, later=v2["campaign_id"])

    offer = diff["next_actions"][0]
    assert "revised brief" in offer["prefilled_args"]["proposal_text"]


def test_a_citation_the_later_judgment_made_can_also_be_stale(conn):
    """`carried_stale` looked only at the earlier evaluation, so a later judgment resting on
    a record that has since been replaced reported nothing — and that is the judgment
    somebody is about to act on."""
    old = _version(conn, "Peru v1")
    v1 = _version(conn, "Colombia v1")
    _judge(conn, v1, [_finding("No posting dates")])
    v2 = _version(conn, "Colombia v2")
    _judge(conn, v2, [_finding("No posting dates")], cited_ids=[old])
    store.insert_campaign(conn, title="Peru v2", supersedes=old)

    stale = core.diff_campaigns(conn, earlier=v1, later=v2)["carried_stale"]

    assert [s["campaign_id"] for s in stale] == [old]
    assert stale[0]["cited_by"] == "later"


def test_the_shape_is_the_same_whether_or_not_it_is_comparable(conn):
    """`counts` was absent on the not-comparable branch, so a caller reading it had to know
    which branch it was in before it could read the result at all."""
    v1 = _version(conn, "Colombia v1")
    _judge(conn, v1, [_finding("No posting dates")])
    v2 = _version(conn, "Colombia v2")

    incomparable = core.diff_campaigns(conn, earlier=v1, later=v2)
    _judge(conn, v2, [_finding("No posting dates")])
    comparable = core.diff_campaigns(conn, earlier=v1, later=v2)

    assert set(incomparable) - {"why_not_comparable", "next_actions"} == \
        set(comparable) - {"why_not_comparable", "next_actions"}


def test_resolving_a_finding_that_does_not_exist_is_refused(conn):
    """A `finding_id` pointing at nothing, or at another campaign's judgment, saved without
    complaint and then vanished from the diff — so "we fixed that" was recorded and silently
    lost, which is the worst possible outcome for the one field `adopted` depends on."""
    v1 = _version(conn, "Colombia v1")
    _judge(conn, v1, [_finding("No posting dates")])
    v2 = _version(conn, "Colombia v2")

    with pytest.raises(ValueError) as exc:
        core.save_evaluation(conn, subject_title="Colombia v2", verdict="approve",
                             summary="All addressed.", campaign_id=v2, findings=[],
                             resolved=[{"finding_id": "eval_nope#1", "was": "x",
                                        "now": "y"}])

    assert "finding_id" in str(exc.value)
