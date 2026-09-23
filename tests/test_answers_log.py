"""
The log of human overrides, readable.

§10.2 gave this product somewhere for a person's answer to go — `answer_finding`,
`answer_gap` — and then nothing to read them back with. For a product whose whole pitch is
that its judgments can be ARGUED WITH, the record of where somebody argued and won is the
single most valuable thing it now holds, and it was write-only.

That is D116's own shape one level down, and it fails the same way: a team lead cannot ask
"what have we marked deliberate this quarter", "who set aside which gaps", or "show me
everything Ana answered" — so nobody reviews the overrides, so a wrong one is permanent in
practice even though `open` makes it reversible in principle.

It also closes the review's other week-one complaint: nothing was ever re-asked. An answer
carries `said_at`, so "set aside eight months ago, by somebody who has left" is a fact the
library holds and never surfaced. It does not schedule anything — this product has no
scheduler and says so — it just stops hiding the age.
"""
import pytest

import core
import store


def _campaign(conn, title="Colombia", **kw):
    kw.setdefault("market", "LATAM")
    kw.setdefault("status", "concluded")
    kw.setdefault("detail", f"A campaign called {title}, which ran in a market.")
    return core.ingest_campaign(conn, title=title, **kw)["campaign_id"]


def _judged(conn, campaign_id):
    return core.save_evaluation(
        conn, subject_title="Colombia", campaign_id=campaign_id, verdict="revise",
        summary="One difference nobody has explained.",
        approve_if="Eight weeks, or say why six.",
        findings=[{"severity": "should_fix", "kind": "precedent_departure",
                   "departure": "unexplained",
                   "finding": "Six weeks where every comparable ran eight.",
                   "fix": "Extend to eight weeks, or say why six is right.",
                   "precedent": {"campaign_id": campaign_id,
                                 "quote": "which ran in a market"}}])


def test_nothing_answered_says_so_rather_than_returning_an_empty_list(conn):
    """`nothing_to_check` is never a pass. An empty list here means "nobody has overridden
    anything", which on a library that has been judged is a finding in itself — and it must
    not read as "we looked and there is nothing to see" when nothing was judged either."""
    answers = core.answers(conn)

    assert answers["status"] == "nothing_to_check"
    assert "Nobody has answered" in answers["what_it_means"]


def test_every_answer_is_readable_with_who_said_it(conn):
    cid = _campaign(conn)
    judged = _judged(conn, cid)
    core.answer_finding(conn, evaluation_id=judged["evaluation_id"],
                        finding_id=judged["findings"][0]["id"], answer="deliberate",
                        note="The client moved the launch date.", said_by="R. Vega")

    answered = core.answers(conn)["answers"]

    assert len(answered) == 1
    assert answered[0]["answer"] == "deliberate"
    assert answered[0]["said_by"] == "R. Vega"
    assert answered[0]["basis"] == "stated"
    assert answered[0]["subject_kind"] == "finding"


def test_an_answer_carries_what_it_was_about(conn):
    """A log of `finding_id` values is a log nobody can read. Whoever is reviewing overrides
    needs the finding's own words and the judgment it belongs to."""
    cid = _campaign(conn)
    judged = _judged(conn, cid)
    core.answer_finding(conn, evaluation_id=judged["evaluation_id"],
                        finding_id=judged["findings"][0]["id"], answer="misread",
                        note="The brief says eight; you read the wrong line.",
                        said_by="R. Vega")

    entry = core.answers(conn)["answers"][0]

    assert "six weeks" in entry["about"].lower()
    assert entry["subject_title"] == "Colombia"


def test_the_history_is_kept_and_the_latest_is_marked(conn):
    """`answers` is append-only so that what somebody said in March survives being
    contradicted in June. That is only worth anything if both are readable — and a reader has
    to be able to tell which one stands now."""
    cid = _campaign(conn)
    judged = _judged(conn, cid)
    for answer, note in (("deliberate", "The client moved the date."),
                         ("fixed", "Actually we went back to eight weeks.")):
        core.answer_finding(conn, evaluation_id=judged["evaluation_id"],
                            finding_id=judged["findings"][0]["id"], answer=answer,
                            note=note, said_by="R. Vega")

    entries = core.answers(conn)["answers"]

    assert [e["answer"] for e in entries] == ["fixed", "deliberate"], "newest first"
    assert entries[0]["stands"] is True
    assert entries[1]["stands"] is False


def test_it_can_be_narrowed_to_one_person(conn):
    """"Show me everything Ana answered" is the question a team lead actually has."""
    cid = _campaign(conn)
    judged = _judged(conn, cid)
    core.answer_finding(conn, evaluation_id=judged["evaluation_id"],
                        finding_id=judged["findings"][0]["id"], answer="deliberate",
                        note="The client moved the date.", said_by="R. Vega")
    second = _judged(conn, _campaign(conn, "Peru"))
    core.answer_finding(conn, evaluation_id=second["evaluation_id"],
                        finding_id=second["findings"][0]["id"], answer="fixed",
                        note="Changed in v2.", said_by="A. Duarte")

    assert [e["said_by"] for e in core.answers(conn, said_by="A. Duarte")["answers"]] \
        == ["A. Duarte"]
    assert [e["said_by"] for e in core.answers(conn, said_by="a. duarte")["answers"]] \
        == ["A. Duarte"], "a name is not a case-sensitive identifier"


def test_gaps_somebody_silenced_are_in_the_same_log(conn):
    """The overrides that matter most are the ones that made the product say LESS."""
    measured = _campaign(conn, "Peru", market="Peru")
    core.add_metrics(conn, campaign_id=measured, structured={"roas": 3.4}, confirm=True)
    _campaign(conn, "An old Chile activation", market="Chile")
    core.answer_gap(conn, code="market_without_outcomes", answer="not_applicable",
                    note="That agency no longer exists.", said_by="R. Vega")

    silencing = [e for e in core.answers(conn)["answers"] if e["subject_kind"] == "gap"]

    assert silencing[0]["about"] == "market_without_outcomes"
    assert silencing[0]["silences"] is True


def test_an_old_answer_says_how_old(conn):
    """"A gap set aside in March against a market that is later revived is a lie the product
    tells forever." There is no scheduler here and this does not invent one — it stops hiding
    the age, so a person reviewing the log can see which answers nobody has revisited."""
    measured = _campaign(conn, "Peru", market="Peru")
    core.add_metrics(conn, campaign_id=measured, structured={"roas": 3.4}, confirm=True)
    _campaign(conn, "An old Chile activation", market="Chile")
    core.answer_gap(conn, code="market_without_outcomes", answer="not_applicable",
                    note="That agency no longer exists.", said_by="R. Vega")
    conn.execute("UPDATE answers SET said_at = ?, created_at = ?",
                 ("2025-01-04T09:00:00+00:00", store._now() - 400 * 86400))
    conn.commit()

    entry = core.answers(conn)["answers"][0]

    assert entry["days_ago"] >= 365
    assert entry["stale"] is True
    assert "nobody has revisited" in core.answers(conn)["what_it_means"]


def test_an_unreviewed_silencing_answer_is_surfaced_at_session_start(conn):
    """The queue's founding insight is that a question asked once and never again is a
    question nobody answered — and it was not applied to the ANSWERS. A gap written off in
    March is still written off in November, nothing re-asks, and the surface whose job is to
    say what the library cannot do reported a rosier picture with no hint why."""
    measured = _campaign(conn, "Peru", market="Peru")
    core.add_metrics(conn, campaign_id=measured, structured={"roas": 3.4}, confirm=True)
    _campaign(conn, "An old Chile activation", market="Chile")
    core.answer_gap(conn, code="market_without_outcomes", answer="not_applicable",
                    note="That agency no longer exists.", said_by="R. Vega")
    conn.execute("UPDATE answers SET created_at = ?", (store._now() - 400 * 86400,))
    conn.commit()

    assert "answers" in [a["tool"] for a in core.readiness(conn)["next_actions"]]


def test_a_recent_answer_is_not_nagged_about(conn):
    """The silence half. An answer given last week is one somebody remembers giving, and
    asking them to review it is the always-present offer that teaches a reader to skip the
    list."""
    measured = _campaign(conn, "Peru", market="Peru")
    core.add_metrics(conn, campaign_id=measured, structured={"roas": 3.4}, confirm=True)
    _campaign(conn, "An old Chile activation", market="Chile")
    core.answer_gap(conn, code="market_without_outcomes", answer="not_applicable",
                    note="That agency no longer exists.", said_by="R. Vega")

    assert "answers" not in [a["tool"] for a in core.readiness(conn).get("next_actions") or []]


def test_an_old_answer_that_silences_nothing_is_not_nagged_about(conn):
    """`open` takes an answer back; a log full of those is a tidy log, not an outstanding
    review. Gating on age alone would turn this into a footer."""
    cid = _campaign(conn)
    judged = _judged(conn, cid)
    core.answer_finding(conn, evaluation_id=judged["evaluation_id"],
                        finding_id=judged["findings"][0]["id"], answer="fixed",
                        note="We went back to eight weeks.", said_by="R. Vega")
    conn.execute("UPDATE answers SET created_at = ?", (store._now() - 400 * 86400,))
    conn.commit()

    assert "answers" not in [a["tool"] for a in core.readiness(conn).get("next_actions") or []]
