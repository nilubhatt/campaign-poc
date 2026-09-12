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
  ignored           a v1 finding the v2 evaluation raised again
  newly_introduced  a v2 finding with no counterpart in v1
  no_longer_raised  neither resolved nor re-raised — and that is genuinely ambiguous: it
                    was either fixed without being recorded, or the second review did not
                    look. Reported as its own bucket rather than quietly counted as adopted
  carried_stale     a citation to a record that has since been superseded

Every entry says `basis: computed`, because all of it is. Where a version has never been
evaluated there is nothing to compute, and the tool says so rather than guessing — which is
the one thing a diff must not do, since "you ignored my correction" is an accusation.
"""
import pytest

import core
import store


def _finding(text, **over):
    base = {"severity": "blocking", "category": "timeline", "finding": text}
    base.update(over)
    return base


def _version(conn, title, **kw):
    kw.setdefault("detail", f"the {title} brief")
    return core.ingest_campaign(conn, title=title, confirm=True, **kw)["campaign_id"]


def _judge(conn, campaign_id, findings, **over):
    payload = {"subject_title": "a version", "verdict": "revise" if findings else "approve",
               "summary": "A summary of what this version got right and wrong.",
               "campaign_id": campaign_id, "findings": findings}
    payload.update(over)
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


def test_a_correction_raised_again_is_reported_as_ignored(conn):
    """The one the reviewer did by hand. A v1 finding the v2 evaluation raises again is the
    definition of a correction that was not taken."""
    v1 = _version(conn, "Colombia v1")
    _judge(conn, v1, [_finding("No posting dates on any deliverable")])

    v2 = _version(conn, "Colombia v2")
    _judge(conn, v2, [_finding("No posting dates on any deliverable")])

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)

    assert [f["finding"] for f in diff["ignored"]] == ["No posting dates on any deliverable"]
    assert diff["ignored"][0]["basis"] == "computed"


def test_the_same_problem_worded_differently_is_still_the_same_problem(conn):
    """Two reviews of the same defect rarely phrase it identically. Matching on exact text
    would report every ignored correction as a new one, which is the most flattering
    possible error."""
    v1 = _version(conn, "Colombia v1")
    _judge(conn, v1, [_finding("No posting dates on any deliverable")])

    v2 = _version(conn, "Colombia v2")
    _judge(conn, v2, [_finding("No posting date on any of the deliverables")])

    diff = core.diff_campaigns(conn, earlier=v1, later=v2)

    assert len(diff["ignored"]) == 1
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

    assert diff["adopted"] == [] and diff["ignored"] == []
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

    assert [f["finding"] for f in diff["ignored"]] == ["No budget figure anywhere"]


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

    earlier_accounted = (len(diff["adopted"]) + len(diff["ignored"])
                         + len(diff["no_longer_raised"]))
    assert earlier_accounted == 3
    assert len(diff["ignored"]) + len(diff["newly_introduced"]) == 2
