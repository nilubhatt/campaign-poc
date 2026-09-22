"""
§8.7: replay as a report.

The review: *"Because every verdict is stamped with its rulebook version, adding an expected
metric can be applied backwards as a question rather than a rewrite. Two answers worth having
immediately: which stored campaigns now fail the new expectation — that is the backlog of
things to go and ask partners for — and which past verdicts would change under the current
rules. Old evaluations are never silently rewritten; the replay is a report."*

**"The replay is a report"** is the whole design. Nothing is written, nothing is marked, and
running it twice changes nothing — the answer is derived from what is on file every time it is
asked, so there is no second copy of it to go stale and no stored judgment carrying a flag
somebody has to trust.

**"Which past verdicts would change" is a claim the server cannot honestly make.** It cannot
re-run the model, and saying a verdict "would change" is the confident unfounded assertion this
whole review is written against. What the server knows exactly is what the judgment was NOT
CHECKED AGAINST: which measures became expected after it was saved, which rules became standing
after it was saved, and whether a rule it actually rested on has since been set aside. That is
computed, identical for every reader, and it is the question a person can act on — the answer
to "would it change" is re-running it, which is the offer rather than the report.

The two halves are different in kind and are kept apart. The **backlog** is about records: a
campaign missing an expected measure is a thing to go and ask a partner for. The **judgments**
half is about evaluations, and every row is a question, never a correction.
"""
import pytest

import corrections
import core
import metrics
import replay
import store

SEEDING = "Seed a single colourway across all recipients."


def _campaign(conn, title, market="LATAM", status="concluded", **kw):
    return core.ingest_campaign(conn, title=title, market=market, status=status,
                                detail="A launch.", **kw)["campaign_id"]


def _expect_measure(conn, key="footfall_uplift_pct"):
    """Graduate a measure the honest way — through the gate."""
    for market in ("LATAM", "SEA", "EMEA"):
        metrics.record(conn, campaign_id=_campaign(conn, f"seen-{market}", market=market),
                       key=key, value=1.0)
    name = metrics.canonical(conn, key)
    metrics.graduate(conn, name, confirmed_by="R. Vega")
    return name


def _standing_rule(conn, text=SEEDING):
    for market in ("LATAM", "SEA", "EMEA"):
        corrections.note(conn, text=text, provenance=f"{market} slide 4",
                         campaign_id=_campaign(conn, f"said-{market}", market=market))
    cid = corrections.find(conn, text)["correction_id"]
    corrections.graduate(conn, cid, confirmed_by="R. Vega")
    return cid


def _judged(conn, campaign_id, title="Colombia v1", **over):
    args = dict(subject_title=title, campaign_id=campaign_id, verdict="approve",
                summary="Looks sound.", findings=[])
    args.update(over)
    return core.save_evaluation(conn, **args)["evaluation_id"]


# ── the backlog: things to go and ask partners for ──────────────────────────

def test_it_lists_the_campaigns_that_now_fail_a_new_expectation(conn):
    """"Which stored campaigns now fail the new expectation — that is the backlog of things to
    go and ask partners for." The point is that it is actionable: a list of partners to email,
    not a count."""
    subject = _campaign(conn, "Colombia v1")
    name = _expect_measure(conn)

    report = replay.run(conn)
    group = next(g for g in report["backlog"]["groups"] if g["measure"] == name)
    assert group["market"] == "LATAM"
    assert group["campaigns_missing_it"] == 1
    assert [c["title"] for c in group["campaigns"]] == ["Colombia v1"]


def test_a_campaign_that_carries_it_is_not_in_the_backlog(conn):
    subject = _campaign(conn, "Colombia v1")
    _expect_measure(conn)
    metrics.record(conn, campaign_id=subject, key="footfall_uplift_pct", value=9.0)

    assert replay.run(conn)["backlog"]["groups"] == []


def test_a_campaign_in_another_market_is_not_in_the_backlog(conn):
    """A measure that graduated on LATAM, SEA and EMEA is not something to go and ask a
    Japanese partner for — they never agreed to it, and a backlog that lists them is a list
    nobody works through."""
    elsewhere = _campaign(conn, "Tokyo v1", market="JAPAN")
    _expect_measure(conn)

    assert replay.run(conn)["backlog"]["groups"] == []


def test_a_reference_record_is_not_in_the_backlog(conn):
    ref = core.ingest_campaign(conn, title="Brand guidelines", record_type="reference",
                               market="LATAM", detail="Guidelines.")["campaign_id"]
    _expect_measure(conn)

    assert replay.run(conn)["backlog"]["groups"] == []


def test_an_empty_backlog_says_so_rather_than_returning_nothing(conn):
    """A bare empty list reads as "the report found nothing" and as "the report did not run".
    §7.1's rule: `nothing_to_check` is not a clean result."""
    report = replay.run(conn)
    assert report["backlog"]["groups"] == []
    assert "nothing is expected" in report["what_it_means"].lower()


def test_a_campaign_that_has_not_run_is_not_something_to_ask_a_partner_for(conn):
    """"The backlog of things to go and ask partners for." You cannot ask for numbers that do
    not exist yet — a campaign still in flight has no measured anything, so listing it as
    missing everything fills the list with work nobody can do."""
    _campaign(conn, "Not yet run", status="proposed")
    _expect_measure(conn)

    backlog = replay.run(conn)["backlog"]
    assert backlog["groups"] == []
    assert backlog["campaigns_still_running"] == 1, \
        "counted, not dropped silently — 'nothing to chase' and 'I did not look' differ"


def test_the_backlog_is_grouped_by_who_to_ask_and_what_for(conn):
    """One conversation per partner per measure is the unit of work. An ungrouped list of
    four hundred rows ordered by creation date is neither a count nor a morning's work."""
    for n in range(3):
        _campaign(conn, f"Colombia v{n}")
    _campaign(conn, "Lima v1", market="SEA")
    _expect_measure(conn)

    groups = replay.run(conn)["backlog"]["groups"]
    assert [(g["market"], g["campaigns_missing_it"]) for g in groups] == \
        [("LATAM", 3), ("SEA", 1)], "biggest conversation first"
    assert "one conversation to have" in groups[0]["what_it_means"]


def test_the_backlog_names_only_a_few_and_says_how_many_more(conn):
    """Every model-facing list in this codebase is bounded, and this phase has now closed the
    same defect twice."""
    for n in range(12):
        _campaign(conn, f"Colombia v{n}")
    _expect_measure(conn)

    group = replay.run(conn)["backlog"]["groups"][0]
    assert len(group["campaigns"]) == replay._MAX_NAMED
    assert group["campaigns_missing_it"] == 12
    assert group["more"] == 12 - replay._MAX_NAMED


# ── the judgments half: a question, never a correction ──────────────────────

def test_it_names_what_a_past_judgment_was_never_checked_against(conn):
    """"Which past verdicts would change" is a claim the server cannot make — it cannot re-run
    the model. What it knows exactly is what the judgment was not checked against, which is
    computed, identical for every reader, and the thing a person can act on."""
    subject = _campaign(conn, "Colombia v1")
    eid = _judged(conn, subject)
    name = _expect_measure(conn)
    cid = _standing_rule(conn)

    row = next(r for r in replay.run(conn)["judgments"] if r["evaluation_id"] == eid)
    assert row["not_checked_against"]["measures"] == [name]
    assert [c["correction_id"] for c in row["not_checked_against"]["corrections"]] == [cid]


def test_it_does_not_claim_the_verdict_would_change(conn):
    """Saying so would be the confident unfounded assertion this whole review is about. The
    answer to "would it change" is re-running it, which is an offer and not a report."""
    subject = _campaign(conn, "Colombia v1")
    _judged(conn, subject)
    _expect_measure(conn)

    report = replay.run(conn)
    said = report["what_it_means"] + report["judgments"][0]["what_it_means"]
    assert "would change" not in said.lower()
    assert "not a verdict" in said.lower()


def test_a_judgment_whose_subject_already_carries_it_is_not_listed(conn):
    """A check that would pass is not a question. Listing it beside a real gap, with the same
    sentence, is what turns this report into noise — and the function that answers it
    (`expected_check`) was already there and was not being asked."""
    subject = _campaign(conn, "Colombia v1")
    eid = _judged(conn, subject)
    _expect_measure(conn)
    metrics.record(conn, campaign_id=subject, key="footfall_uplift_pct", value=9.0)

    assert [r for r in replay.run(conn)["judgments"] if r["evaluation_id"] == eid] == []


def test_a_gap_that_would_appear_says_so_as_a_fact_about_the_evidence(conn):
    subject = _campaign(conn, "Colombia v1")
    _judged(conn, subject)
    name = _expect_measure(conn)

    row = replay.run(conn)["judgments"][0]
    assert row["consequence"] == "gap_appears"
    assert row["gaps_that_would_appear"] == [name]
    assert "not a verdict" in row["what_it_means"]


def test_a_verdict_resting_entirely_on_a_withdrawn_rule_is_ranked_first(conn):
    """The product already encodes "this verdict rests ONLY on rules that were broken" — it is
    why such a verdict skips the disconfirming search. When every one of those rules has since
    been withdrawn, the whole stated basis of the verdict is gone, and reported as a flat list
    that read exactly like one note among five."""
    cid = _standing_rule(conn)
    sole = _campaign(conn, "Colombia v1")
    eid = core.save_evaluation(
        conn, subject_title="Colombia v1", campaign_id=sole, verdict="reject",
        summary="Seeds four.",
        findings=[{"severity": "blocking", "kind": "guardrail_breach",
                   "finding": "Seeds four colourways", "fix": "Seed one",
                   "precedent": {"correction_id": cid, "quote": SEEDING}}])["evaluation_id"]
    other = _campaign(conn, "Peru v1")
    _judged(conn, other, title="Peru v1")
    _expect_measure(conn)
    corrections.set_aside(conn, cid)

    rows = replay.run(conn)["judgments"]
    assert rows[0]["evaluation_id"] == eid
    assert rows[0]["consequence"] == "stated_basis_withdrawn"
    assert "whole stated basis" in rows[0]["what_it_means"]
    assert rows[1]["consequence"] == "gap_appears"


def test_a_verdict_that_merely_mentions_a_withdrawn_rule_is_not_that(conn):
    """One finding among several is a different situation from the verdict's whole basis, and
    saying so is the point of separating them."""
    cid = _standing_rule(conn)
    subject = _campaign(conn, "Colombia v1")
    core.save_evaluation(
        conn, subject_title="Colombia v1", campaign_id=subject, verdict="revise",
        summary="A stretch.", approve_if="Fix the dates.",
        findings=[{"severity": "blocking", "kind": "guardrail_breach",
                   "finding": "Seeds four colourways", "fix": "Seed one",
                   "precedent": {"correction_id": cid, "quote": SEEDING}},
                  {"severity": "blocking", "kind": "missing_information",
                   "finding": "No end date", "fix": "Add one"}])
    corrections.set_aside(conn, cid)

    row = replay.run(conn)["judgments"][0]
    assert row["consequence"] == "rests_on_withdrawn"
    assert "The rest of it stands" in row["what_it_means"]


def test_a_rule_that_became_standing_is_not_guessed_at(conn):
    """Whether a brief breaks a rule is a judgment. The server says the rule arrived after the
    judgment and stops — guessing would be the thing this whole item refuses."""
    subject = _campaign(conn, "Colombia v1")
    _judged(conn, subject)
    _standing_rule(conn)

    row = replay.run(conn)["judgments"][0]
    assert row["consequence"] == "rule_not_applied"
    assert "is a judgment, so the library does not guess" in row["what_it_means"]


def test_it_offers_to_judge_it_again_rather_than_doing_it(conn):
    subject = _campaign(conn, "Colombia v1")
    _judged(conn, subject)
    _expect_measure(conn)

    row = replay.run(conn)["judgments"][0]
    assert row["next_actions"][0]["tool"] == "prepare_evaluation"
    assert row["next_actions"][0]["prefilled_args"]["campaign_id"] == subject


def test_a_judgment_made_after_the_rule_graduated_is_not_listed(conn):
    """It was checked against it. Listing it would make the report a list of every judgment
    ever saved, which is a report nobody reads twice."""
    _expect_measure(conn)
    subject = _campaign(conn, "Colombia v1")
    eid = _judged(conn, subject)

    assert [r for r in replay.run(conn)["judgments"] if r["evaluation_id"] == eid] == []


def test_a_judgment_resting_on_a_rule_since_set_aside_is_surfaced(conn):
    """The other direction, and the more urgent one: a finding that was not debatable rests on
    a rule that is no longer a rule. Nothing rewrites the judgment — but somebody has to know
    it is standing on something that was withdrawn."""
    cid = _standing_rule(conn)
    subject = _campaign(conn, "Colombia v1")
    eid = core.save_evaluation(
        conn, subject_title="Colombia v1", campaign_id=subject, verdict="revise",
        summary="Seeds four.", approve_if="Seed one colourway.",
        findings=[{"severity": "blocking", "kind": "guardrail_breach",
                   "finding": "Seeds four colourways", "fix": "Seed one",
                   "precedent": {"correction_id": cid, "quote": SEEDING}}])["evaluation_id"]
    corrections.set_aside(conn, cid)

    row = next(r for r in replay.run(conn)["judgments"] if r["evaluation_id"] == eid)
    assert [c["correction_id"] for c in row["rests_on_withdrawn"]] == [cid]
    assert "no longer" in row["what_it_means"].lower()


def test_a_judgment_about_another_market_is_not_listed(conn):
    """A judgment about a Japanese brief was not "unchecked" against a rule that graduated on
    LATAM and SEA — that rule does not apply to it. Listing it would put every judgment in the
    library on the report the first time anything graduated anywhere."""
    elsewhere = _campaign(conn, "Tokyo v1", market="JAPAN")
    eid = _judged(conn, elsewhere, title="Tokyo v1")
    _expect_measure(conn)
    _standing_rule(conn)

    assert [r for r in replay.run(conn)["judgments"] if r["evaluation_id"] == eid] == []


def test_a_judgment_with_no_subject_record_claims_nothing(conn):
    """No record means no market, and a rule belongs to one. It is listed only for a rule it
    actually cited being withdrawn, which is a fact about the judgment rather than a market."""
    eid = core.save_evaluation(conn, subject_title="A loose pitch", verdict="approve",
                               summary="Fine.", findings=[])["evaluation_id"]
    _expect_measure(conn)
    _standing_rule(conn)

    assert [r for r in replay.run(conn)["judgments"] if r["evaluation_id"] == eid] == []


def test_a_judgment_made_after_a_rule_became_standing_is_not_listed(conn):
    """The corrections half of the same comparison — measures had this and rules did not."""
    _standing_rule(conn)
    subject = _campaign(conn, "Colombia v1")
    eid = _judged(conn, subject)

    assert [r for r in replay.run(conn)["judgments"] if r["evaluation_id"] == eid] == []


def test_the_sentence_names_what_was_missed(conn):
    """A sentence that says "something changed" is a sentence nobody can act on."""
    subject = _campaign(conn, "Colombia v1")
    _judged(conn, subject)
    name = _expect_measure(conn)

    assert name in replay.run(conn)["judgments"][0]["what_it_means"]


def test_nothing_is_ever_written(conn):
    """"Old evaluations are never silently rewritten; the replay is a report." Running it twice
    gives the same answer and changes nothing — so there is no second copy to go stale, and no
    stored judgment carrying a flag somebody has to trust."""
    subject = _campaign(conn, "Colombia v1")
    eid = _judged(conn, subject)
    _expect_measure(conn)

    before = store.get_evaluation(conn, eid)
    first = replay.run(conn)
    second = replay.run(conn)

    assert store.get_evaluation(conn, eid) == before
    assert first["judgments"] == second["judgments"]
    assert first["backlog"] == second["backlog"]


def test_running_the_report_writes_nothing_at_all_on_a_fresh_library(conn):
    """Not just "the evaluation is unchanged" — the DATABASE is unchanged. The registry used
    to seed itself on first read, so the first call to a report whose whole claim is that it
    writes nothing wrote twelve rows, on exactly the database a customer meets first."""
    import hashlib

    _campaign(conn, "Colombia v1")
    before = hashlib.sha256("\n".join(conn.iterdump()).encode()).hexdigest()

    replay.run(conn)

    assert hashlib.sha256("\n".join(conn.iterdump()).encode()).hexdigest() == before


def test_asking_a_gate_writes_nothing_either(conn):
    """The gate now carries the blast radius, which scans the library. Scanning is reading."""
    import hashlib

    for market in ("LATAM", "SEA", "EMEA"):
        metrics.record(conn, campaign_id=_campaign(conn, f"seen-{market}", market=market),
                       key="footfall_uplift_pct", value=1.0)
    before = hashlib.sha256("\n".join(conn.iterdump()).encode()).hexdigest()

    metrics.graduation(conn, "footfall_uplift")

    assert hashlib.sha256("\n".join(conn.iterdump()).encode()).hexdigest() == before


def test_a_judgment_that_cites_a_rule_is_never_said_to_have_missed_it(conn):
    """Re-confirming a rule rewrote `confirmed_at`, so a rule graduated, cited by a judgment,
    set aside and re-graduated looked NEWER than the judgment citing it — and the report
    asserted the judgment had never seen it, with the server's authority behind it."""
    cid = _standing_rule(conn)
    subject = _campaign(conn, "Colombia v1")
    eid = core.save_evaluation(
        conn, subject_title="Colombia v1", campaign_id=subject, verdict="revise",
        summary="Seeds four.", approve_if="Seed one colourway.",
        findings=[{"severity": "blocking", "kind": "guardrail_breach",
                   "finding": "Seeds four colourways", "fix": "Seed one",
                   "precedent": {"correction_id": cid, "quote": SEEDING}}])["evaluation_id"]

    corrections.set_aside(conn, cid)
    corrections.reopen(conn, cid)
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    listed = [r for r in replay.run(conn)["judgments"] if r["evaluation_id"] == eid]
    assert listed == [], "it cited the rule; there is nothing to report"


def test_the_first_confirmation_is_the_one_that_counts(conn):
    """What makes the citation case hold at all: the comparison is against when this BECAME a
    rule, and re-confirming must not move that."""
    name = _expect_measure(conn)
    first = metrics.describe(conn, name)["confirmed_at"]
    store.retire_metric(conn, name)
    metrics.graduate(conn, name, confirmed_by="A. Duarte")

    assert metrics.describe(conn, name)["confirmed_at"] == first


def test_the_same_holds_for_a_rule(conn):
    cid = _standing_rule(conn)
    first = corrections.describe(conn, cid)["confirmed_at"]
    corrections.set_aside(conn, cid)
    corrections.reopen(conn, cid)
    corrections.graduate(conn, cid, confirmed_by="A. Duarte")

    assert corrections.describe(conn, cid)["confirmed_at"] == first


def test_a_measure_no_longer_expected_is_not_reported_as_missed(conn):
    """It carries a `confirmed_at` forever — it was confirmed once. A judgment is not
    "unchecked" against something nothing is checked against any more."""
    subject = _campaign(conn, "Colombia v1")
    eid = _judged(conn, subject)
    name = _expect_measure(conn)
    store.retire_metric(conn, name)

    assert [r for r in replay.run(conn)["judgments"] if r["evaluation_id"] == eid] == []


def test_a_rule_no_longer_standing_is_not_reported_as_missed_either(conn):
    subject = _campaign(conn, "Colombia v1")
    eid = _judged(conn, subject)
    cid = _standing_rule(conn)
    corrections.set_aside(conn, cid)

    listed = [r for r in replay.run(conn)["judgments"] if r["evaluation_id"] == eid]
    assert listed == [], "this judgment neither cited it nor missed a rule that applies"


def test_a_judgment_with_no_recorded_date_is_left_alone(tmp_path):
    """A judgment from a database that predates the column cannot be placed on the timeline,
    and guessing a date for it would make the report assert something about a judgment nobody
    can place. `created_at` is NOT NULL on a fresh schema, so this is only reachable the way a
    customer would reach it: an upgrade, where the column is added nullable (D89)."""
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE campaigns (id TEXT PRIMARY KEY, title TEXT NOT NULL, market TEXT,
                                status TEXT, record_type TEXT);
        CREATE TABLE evaluations (id TEXT PRIMARY KEY, subject_title TEXT NOT NULL,
                                  campaign_id TEXT);
    """)
    old.execute("INSERT INTO campaigns VALUES ('camp_old','Colombia v1','LATAM','concluded',"
                "'campaign')")
    old.execute("INSERT INTO evaluations VALUES ('eval_old','Colombia v1','camp_old')")
    old.commit()
    old.close()

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    store.upgrade(conn)
    assert store.list_evaluations(conn)[0]["created_at"] is None

    for market in ("LATAM", "SEA", "EMEA"):
        cid = core.ingest_campaign(conn, title=f"s{market}", market=market,
                                   status="concluded", detail="x")["campaign_id"]
        metrics.record(conn, campaign_id=cid, key="footfall_uplift_pct", value=1.0)
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")

    assert [r for r in replay.run(conn)["judgments"]
            if r["evaluation_id"] == "eval_old"] == []


def test_the_number_of_groups_is_bounded_too(conn):
    """`_MAX_NAMED` bounds the names inside one group; without `_MAX_GROUPS` a library with
    many markets returns as many groups as it has market-measure pairs."""
    markets = [f"MARKET{n}" for n in range(replay._MAX_GROUPS + 4)]
    for market in markets:
        metrics.record(conn, campaign_id=_campaign(conn, f"seen-{market}", market=market),
                       key="footfall_uplift_pct", value=1.0)
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")
    for market in markets:
        _campaign(conn, f"Brief in {market}", market=market)

    report = replay.run(conn)
    assert report["backlog"]["groups_total"] == len(markets)
    assert len(report["backlog"]["groups"]) == replay._MAX_GROUPS


def test_a_rule_reopened_but_not_re_confirmed_still_shows_as_withdrawn(conn):
    """Keyed on `ignored` alone, a judgment whose blocking finding cites the rule vanished from
    the report the moment somebody reopened it — and a reopened rule is `provisional`, applied
    to nothing."""
    cid = _standing_rule(conn)
    subject = _campaign(conn, "Colombia v1")
    eid = core.save_evaluation(
        conn, subject_title="Colombia v1", campaign_id=subject, verdict="reject",
        summary="Seeds four.",
        findings=[{"severity": "blocking", "kind": "guardrail_breach",
                   "finding": "Seeds four colourways", "fix": "Seed one",
                   "precedent": {"correction_id": cid, "quote": SEEDING}}])["evaluation_id"]
    corrections.set_aside(conn, cid)
    corrections.reopen(conn, cid)

    row = next(r for r in replay.run(conn)["judgments"] if r["evaluation_id"] == eid)
    assert [c["correction_id"] for c in row["rests_on_withdrawn"]] == [cid]


def test_a_superseded_version_is_not_a_second_partner_to_chase(conn):
    """v1 and v2 of one brief were both listed, so one conversation appeared as two."""
    v1 = _campaign(conn, "Colombia v1")
    core.ingest_campaign(conn, title="Colombia v2", market="LATAM", status="concluded",
                         supersedes=v1, detail="A launch.")
    _expect_measure(conn)

    group = replay.run(conn)["backlog"]["groups"][0]
    assert [c["title"] for c in group["campaigns"]] == ["Colombia v2"]


def test_scoping_the_report_scopes_both_halves(conn):
    """Filtering only the backlog made the summary add up two different populations."""
    lat = _campaign(conn, "Colombia v1")
    _judged(conn, lat)
    sea = _campaign(conn, "Lima v1", market="SEA")
    _judged(conn, sea, title="Lima v1")
    _expect_measure(conn)

    only = replay.run(conn, market="SEA")
    assert [g["market"] for g in only["backlog"]["groups"]] == ["SEA"]
    assert [j["subject_title"] for j in only["judgments"]] == ["Lima v1"]


def test_the_report_says_which_rulebook_it_was_run_under(conn):
    """The review's own premise: "every verdict is stamped with its rulebook version". A report
    about what changed has to say what it is comparing against."""
    assert replay.run(conn)["rulebook_version"] == core.rulebook_version()


# ── D104: the blast radius, before somebody confirms ────────────────────────

def test_graduating_a_measure_says_how_many_records_it_affects(conn):
    """D104. "This makes 14 LATAM records show as missing it" — and the person confirming is
    exactly who needs to know that before they confirm, not after."""
    for n in range(4):
        _campaign(conn, f"Old {n}")
    for market in ("LATAM", "SEA", "EMEA"):
        metrics.record(conn, campaign_id=_campaign(conn, f"seen-{market}", market=market),
                       key="footfall_uplift_pct", value=1.0)

    ahead = replay.if_graduated(conn, measure="footfall_uplift")
    assert ahead["campaigns_affected"] == 4
    assert "4" in ahead["what_it_means"]


def test_the_blast_radius_is_on_the_gate_the_model_reads(conn):
    for n in range(4):
        _campaign(conn, f"Old {n}")
    for market in ("LATAM", "SEA", "EMEA"):
        metrics.record(conn, campaign_id=_campaign(conn, f"seen-{market}", market=market),
                       key="footfall_uplift_pct", value=1.0)

    gate = metrics.graduation(conn, "footfall_uplift")
    assert gate["if_confirmed"]["campaigns_affected"] == 4


def test_the_blast_radius_counts_only_records_it_would_apply_to(conn):
    """A reference record is not a brief, a campaign in another market never agreed to the
    measure, and a campaign holding only a TARGET has not measured it. Each one inflates the
    number the person is deciding on."""
    core.ingest_campaign(conn, title="Brand guidelines", record_type="reference",
                         market="LATAM", detail="Guidelines.")
    _campaign(conn, "Tokyo v1", market="JAPAN")
    target_only = _campaign(conn, "Target only")
    real = _campaign(conn, "Colombia v1")
    for market in ("LATAM", "SEA", "EMEA"):
        metrics.record(conn, campaign_id=_campaign(conn, f"seen-{market}", market=market),
                       key="footfall_uplift_pct", value=1.0)
    metrics.record(conn, campaign_id=target_only, key="footfall_uplift_pct", value=9.0,
                   metric_type="target")

    ahead = replay.if_graduated(conn, measure="footfall_uplift")
    assert ahead["campaigns_affected"] == 2, [e["title"] for e in ahead["examples"]]
    assert {e["campaign_id"] for e in ahead["examples"]} == {target_only, real}


def test_a_gate_with_nothing_to_confirm_carries_no_blast_radius(conn):
    """It is attached because there is a decision to make. On an already-expected or
    ineligible measure there is none, and computing it is work nobody asked for."""
    assert "if_confirmed" not in metrics.graduation(conn, "cpm")
    _expect_measure(conn)
    assert "if_confirmed" not in metrics.graduation(conn, "footfall_uplift")


def test_the_blast_radius_is_on_the_offer_a_person_actually_answers(conn):
    """D104's point is "before they confirm", and the confirm decision is made from the offer
    riding on the write — not from `measure_status`, which §8.4 described as a tool a user
    would have to know exists, think to call, and name by canonical stem to use."""
    for n in range(4):
        _campaign(conn, f"Old {n}")
    for market in ("LATAM", "SEA"):
        metrics.record(conn, campaign_id=_campaign(conn, f"seen-{market}", market=market),
                       key="footfall_uplift_pct", value=1.0)
    third = metrics.record(conn, campaign_id=_campaign(conn, "seen-EMEA", market="EMEA"),
                           key="footfall_uplift_pct", value=1.0)

    assert third["newly_eligible"]["if_confirmed"]["campaigns_affected"] == 4


def test_the_same_holds_for_a_standing_rule(conn):
    subject = _campaign(conn, "Colombia v1")
    _judged(conn, subject)
    for market in ("LATAM", "SEA"):
        corrections.note(conn, text=SEEDING, provenance=f"{market} slide 4",
                         campaign_id=_campaign(conn, f"said-{market}", market=market))
    third = corrections.note(conn, text=SEEDING, provenance="EMEA slide 4",
                             campaign_id=_campaign(conn, "said-EMEA", market="EMEA"))

    assert third["newly_eligible"]["if_confirmed"]["judgments_affected"] == 1


def test_graduating_points_at_the_report_it_just_made_non_empty(conn):
    """`replay_rules` was referenced nowhere in the product but its own definition — the third
    time this phase shipped a tool the model would have to know existed and spontaneously
    call. Graduation is the exact instant the report stops being empty."""
    for n in range(3):
        _campaign(conn, f"Old {n}")
    _expect_measure(conn)

    # A second measure, so the freshly-returned result is the one being read.
    for market in ("LATAM", "SEA", "EMEA"):
        metrics.record(conn, campaign_id=_campaign(conn, f"r-{market}", market=market),
                       key="dwell_time_s", value=1.0)
    result = metrics.graduate(conn, "dwell_time", confirmed_by="R. Vega")

    assert result["next_actions"][0]["tool"] == "replay_rules"
    assert "rewritten" in result["next_actions"][0]["why"]


def test_a_standing_rule_points_at_it_too(conn):
    cid = _standing_rule(conn)
    other = "Photo time goes before the workout."
    for market in ("LATAM", "SEA", "EMEA"):
        corrections.note(conn, text=other, provenance="s33",
                         campaign_id=_campaign(conn, f"p-{market}", market=market))
    result = corrections.graduate(conn, corrections.find(conn, other)["correction_id"],
                                  confirmed_by="R. Vega")

    assert result["next_actions"][0]["tool"] == "replay_rules"


def test_the_report_can_be_scoped_to_one_market(conn):
    """A dead parameter on a model-facing seam is the same "offer that cannot be made" class
    this codebase already guards — and scoping is how a big library stays readable."""
    _campaign(conn, "Colombia v1")
    _campaign(conn, "Lima v1", market="SEA")
    _expect_measure(conn)

    only = replay.run(conn, market="SEA")["backlog"]["groups"]
    assert [g["market"] for g in only] == ["SEA"]


def test_asking_what_would_happen_does_not_make_it_happen(conn):
    for market in ("LATAM", "SEA", "EMEA"):
        metrics.record(conn, campaign_id=_campaign(conn, f"seen-{market}", market=market),
                       key="footfall_uplift_pct", value=1.0)

    replay.if_graduated(conn, measure="footfall_uplift")
    assert metrics.describe(conn, "footfall_uplift")["status"] == "provisional"


def test_it_reaches_the_model_over_the_protocol(conn):
    import asyncio
    import json

    import mcp_server

    async def call(name, args):
        return json.loads((await mcp_server.mcp.call_tool(name, args)).content[0].text)

    subject = _campaign(conn, "Colombia v1")
    _judged(conn, subject)
    name = _expect_measure(conn)

    report = asyncio.run(call("replay_rules", {}))
    assert [g for g in report["backlog"]["groups"] if g["measure"] == name]
    assert report["judgments"][0]["not_checked_against"]["measures"] == [name]


# ── a rule that applies EVERYWHERE ──────────────────────────────────────────
#
# §12.3/D108: a correction the customer DECLARED, which has graduated in no market because it
# was seen in none. `expected_in = []` and `applies_everywhere = 1`, and that flag is the only
# thing that says the rule is in force at all. `prepare_evaluation` reads it. This report did
# not, and the two disagreeing about the same rule is what D114 is about: the tool that offers
# the replay is the one that puts the rule in force.


def _declared_everywhere(conn, text="No price promises in client-facing creative."):
    """A house rule, promoted the way the rulebook loader promotes one."""
    cid = corrections.note(conn, text=text, campaign_id=None,
                           provenance="the house rules (declared in rulebook fab-1.0)",
                           )["correction_id"]
    corrections.graduate(conn, cid, confirmed_by="R. Vega", from_rulebook="fab-1.0",
                         everywhere=True, markets=None)
    return cid


def test_a_judgment_is_listed_against_a_rule_that_applies_everywhere(conn):
    """The rule graduated in no market, so a market list can never match it. Read through
    `expected_in` alone, every judgment ever saved vanished from this report the moment a
    house rule was loaded — and the graduation had just offered this very report."""
    subject = _campaign(conn, "Peru v1", market="Peru")
    _judged(conn, subject)
    _declared_everywhere(conn)

    rows = replay.run(conn)["judgments"]
    assert rows, "a house rule in force everywhere reached no saved judgment at all"
    assert rows[0]["consequence"] == "rule_not_applied"
    assert [c["text"] for c in rows[0]["not_checked_against"]["corrections"]] == [
        "No price promises in client-facing creative."]


def test_a_rule_that_applies_everywhere_reaches_a_market_it_never_graduated_in(conn):
    """The point of `everywhere`, on the axis this report gets wrong: no overlap with the
    market a rule was seen in, because there is no market it was seen in."""
    _judged(conn, _campaign(conn, "Japan v1", market="Japan"), title="Japan v1")
    _judged(conn, _campaign(conn, "Peru v1", market="Peru"), title="Peru v1")
    _declared_everywhere(conn)

    assert sorted(r["subject_title"] for r in replay.run(conn)["judgments"]) == [
        "Japan v1", "Peru v1"]


def test_a_scoped_rule_still_reaches_only_its_own_markets(conn):
    """The other half, and the one `everywhere` must not quietly swallow: a rule the library
    INFERRED is expected only where the evidence put it (§8.3). A judgment about a Japanese
    brief was not "unchecked" against a rule that graduated on LATAM."""
    _judged(conn, _campaign(conn, "Japan v1", market="Japan"), title="Japan v1")
    _standing_rule(conn)               # LATAM, SEA, EMEA — not Japan

    assert replay.run(conn)["judgments"] == []


def test_a_judgment_with_no_record_is_not_listed_against_a_rule_that_applies_everywhere(conn):
    """"Everywhere" is every MARKET, and a judgment about a brief nobody stored has none. The
    live side refuses the same way — `standing_for` will not apply a house rule to a brief
    with no market rather than applying all of them — so listing one here would be the two
    surfaces disagreeing again, in the direction that puts every unstored judgment in the
    library on every house rule's list. Mutation found nothing was watching this."""
    _judged(conn, None, title="A brief that was never stored")
    _declared_everywhere(conn)

    assert replay.run(conn)["judgments"] == []


def test_a_rule_that_widens_to_a_new_market_reaches_that_market_s_judgments(conn):
    """A rule standing in LATAM when a Japanese brief was judged did not apply to it, so the
    judgment is not listed. Widen the rule to Japan and it is — the rule reaches that brief
    now and did not then, which is what this report is for. `confirmed_at` cannot see that:
    it says when somebody first confirmed the rule, which was before the judgment either way.
    """
    subject = _campaign(conn, "Japan v1", market="Japan")
    cid = _standing_rule(conn)                       # LATAM, SEA, EMEA
    _judged(conn, subject, title="Japan v1")
    assert replay.run(conn)["judgments"] == []

    # `store.graduate_correction` is the one function that writes a rule's scope, and the
    # only way a standing rule's scope moves: `graduate` refuses a rule already in force.
    store.graduate_correction(conn, cid, markets=["LATAM", "SEA", "EMEA", "Japan"],
                              confirmed_by="R. Vega")

    rows = replay.run(conn)["judgments"]
    assert [r["subject_title"] for r in rows] == ["Japan v1"]
    assert rows[0]["consequence"] == "rule_not_applied"


def test_a_rule_that_narrows_does_not_relist_the_markets_it_kept(conn):
    """The other direction, and the one a timestamp alone gets wrong: a LATAM brief judged
    under a rule that covered LATAM and SEA was checked against it. Dropping SEA changes
    nothing about that judgment, and saying "not checked against" would be false."""
    subject = _campaign(conn, "Colombia v1", market="LATAM")
    cid = _standing_rule(conn)                       # LATAM, SEA, EMEA
    _judged(conn, subject, title="Colombia v1")

    store.graduate_correction(conn, cid, markets=["LATAM"], confirmed_by="R. Vega")

    assert replay.run(conn)["judgments"] == [], (
        "narrowing a rule made a judgment it had already been applied to look unchecked")


def test_a_database_written_before_the_scope_history_answers_as_it_used_to(conn):
    """Nobody's library has this history for rules confirmed under an earlier release, and a
    report that reads an empty history as "nothing applied then" would put every judgment ever
    saved on the list. Without a history the old question is the best answer there is: was the
    rule confirmed after this judgment."""
    already = _campaign(conn, "Colombia v1")
    cid = _standing_rule(conn)
    _judged(conn, already, title="Colombia v1")
    later = _campaign(conn, "Colombia v2")
    _judged(conn, later, title="Colombia v2")
    conn.execute("DELETE FROM correction_scope")       # as an older release left it
    conn.commit()

    assert replay.run(conn)["judgments"] == [], (
        "a rule standing before these judgments came back as never applied")

    # And the other half of that fallback still works: confirmed AFTER the judgment is listed.
    conn.execute("UPDATE corrections SET confirmed_at = ? WHERE id = ?",
                 (store._now(), cid))
    conn.commit()
    assert len(replay.run(conn)["judgments"]) == 2


def test_re_confirming_a_rule_that_has_not_moved_relists_nothing(conn):
    """The history is a list of CHANGES. A confirmation that changes no scope is not one, and
    recording it would say the rule started applying today — putting every judgment in its
    markets back on the report, on an upgrade that changed nothing."""
    subject = _campaign(conn, "Colombia v1")
    cid = _standing_rule(conn)
    _judged(conn, subject, title="Colombia v1")

    before = len(store.correction_scope_history(conn, cid))
    store.graduate_correction(conn, cid, markets=["LATAM", "SEA", "EMEA"],
                              confirmed_by="R. Vega")

    assert len(store.correction_scope_history(conn, cid)) == before, (
        "a re-confirmation with the same scope was recorded as a scope change")
    assert replay.run(conn)["judgments"] == []


def test_a_rule_that_predates_the_history_and_is_then_re_confirmed_relists_nothing(conn):
    """The two halves together, which is the upgrade an install actually performs: rules with
    no history, and then something re-confirms one. The first row written afterwards would be
    the earliest scope on file, so a rule standing since long before a judgment would read as
    having started applying today. What it already said is recorded as having held since it
    was confirmed — the same assumption the report made before this table existed."""
    subject = _campaign(conn, "Colombia v1")
    cid = _standing_rule(conn)
    _judged(conn, subject, title="Colombia v1")
    conn.execute("DELETE FROM correction_scope")
    conn.commit()

    store.graduate_correction(conn, cid, markets=["LATAM", "SEA", "EMEA"],
                              confirmed_by="R. Vega")

    assert replay.run(conn)["judgments"] == [], (
        "an upgrade that changed nothing put an old judgment back on the report")
    assert [h["expected_in"] for h in store.correction_scope_history(conn, cid)] == [
        ["EMEA", "LATAM", "SEA"]], "the scope it already had was not carried forward"


def test_a_judgment_made_while_a_rule_was_withdrawn_is_listed_when_it_comes_back(conn):
    """Set aside, judged, reopened, confirmed again with the SAME markets. There is no scope
    change to compare, so a history of markets alone said the rule had applied throughout and
    this report said nothing about the one judgment it was never applied to. Withdrawn is a
    scope like any other: it is the scope of nothing."""
    cid = _standing_rule(conn)
    corrections.set_aside(conn, cid, why="paused while the client rethinks")
    subject = _campaign(conn, "Colombia v1")
    _judged(conn, subject, title="Colombia v1")
    assert replay.run(conn)["judgments"] == [], "it was not standing, so nothing is claimed"

    corrections.reopen(conn, cid)
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    rows = replay.run(conn)["judgments"]
    assert [r["subject_title"] for r in rows] == ["Colombia v1"], (
        "the rule is back in force and the brief judged without it is not on the report")
    assert rows[0]["consequence"] == "rule_not_applied"


def test_a_judgment_made_before_a_rule_was_withdrawn_is_not_relisted(conn):
    """The other direction. This brief WAS checked against the rule, then somebody set the
    rule aside and put it back. Nothing about that judgment changed, and saying it was never
    checked would be the false claim that matters."""
    subject = _campaign(conn, "Colombia v1")
    cid = _standing_rule(conn)
    _judged(conn, subject, title="Colombia v1")

    corrections.set_aside(conn, cid, why="paused")
    corrections.reopen(conn, cid)
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    assert replay.run(conn)["judgments"] == []


def test_setting_a_rule_aside_twice_records_when_it_stopped_once(conn):
    """`set_aside` takes no view of the status it found — it is idempotent at the row — so it
    can be called on a rule already set aside. The history says WHEN the rule stopped
    applying, and that is the first time, not the last time somebody clicked."""
    cid = _standing_rule(conn)
    corrections.set_aside(conn, cid, why="paused")
    after_first = store.correction_scope_history(conn, cid)

    corrections.set_aside(conn, cid, why="still paused")

    assert store.correction_scope_history(conn, cid) == after_first, (
        "the second set-aside moved the date the rule stopped applying")


def test_a_rule_withdrawn_on_a_database_with_no_history_keeps_what_came_before(conn):
    """The upgrade case for withdrawal. Without the scope it already had recorded as having
    held until then, the earliest row in the history is "withdrawn" — so a brief judged years
    earlier, against the rule, WHILE it was standing, has no scope at its own date and reads
    as never checked once the rule comes back."""
    subject = _campaign(conn, "Colombia v1")
    cid = _standing_rule(conn)
    _judged(conn, subject, title="Colombia v1")
    conn.execute("DELETE FROM correction_scope")        # as an older release left it
    conn.commit()

    corrections.set_aside(conn, cid, why="paused")
    corrections.reopen(conn, cid)
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    assert replay.run(conn)["judgments"] == [], (
        "a judgment this rule HAD been applied to came back as never checked")


def test_two_scope_changes_in_one_clock_tick_keep_the_order_they_happened_in(conn):
    """Windows measures `time.time()` in whole milliseconds, so two changes to one rule can
    share a timestamp. Ordered by the random row id, which of them came last is a coin toss —
    and the last one is the scope that is in force. Ordered by insertion, it is the truth.

    The ids here are deliberately arranged to sort against insertion order, which is what
    makes this test able to fail rather than able to pass twice."""
    cid = _standing_rule(conn)
    conn.execute("DELETE FROM correction_scope")
    same_tick = 1_700_000_000.0
    for row_id, markets in (("zzz", '["LATAM"]'), ("aaa", '["LATAM", "SEA"]')):
        conn.execute("INSERT INTO correction_scope (id, correction_id, changed_at, "
                     "expected_in, applies_everywhere, standing) VALUES (?,?,?,?,0,1)",
                     (row_id, cid, same_tick, markets))
    conn.commit()

    history = store.correction_scope_history(conn, cid)
    assert [h["expected_in"] for h in history] == [["LATAM"], ["LATAM", "SEA"]]
    assert store.correction_scope_at(conn, cid, same_tick)["expected_in"] == ["LATAM", "SEA"]


def test_a_rule_reopened_on_an_upgraded_database_does_not_accuse_an_old_judgment(conn):
    """The offer this product makes for a declared rule somebody set aside is
    `reopen_correction`, and `reopen` leaves the rule PROVISIONAL by design — so the seed that
    records what a row already applied to was skipped on exactly that path. The first history
    row was then dated today, and every judgment ever made in the rule's markets read as
    unchecked, including ones whose blocking finding CITES the rule.

    The unrecorded set-aside window stays invisible, which is the honest answer: nothing on
    file says when it happened, and silence is the safe direction."""
    subject = _campaign(conn, "Colombia v1")
    cid = _standing_rule(conn)
    eid = core.save_evaluation(
        conn, subject_title="Colombia v1", campaign_id=subject, verdict="revise",
        summary="Seeds four.", approve_if="Seed one colourway.",
        findings=[{"severity": "blocking", "kind": "guardrail_breach",
                   "finding": "Seeds four colourways", "fix": "Seed one",
                   "precedent": {"correction_id": cid, "quote": SEEDING}}])["evaluation_id"]
    corrections.set_aside(conn, cid, why="paused while the client rethinks")
    conn.execute("DELETE FROM correction_scope")     # an install from before the history
    conn.commit()

    corrections.reopen(conn, cid)
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    listed = [r for r in replay.run(conn)["judgments"] if r["evaluation_id"] == eid]
    assert not [r for r in listed
                if cid in [c["correction_id"] for c in
                           r["not_checked_against"]["corrections"]]], (
        "the report says this judgment was never checked against the rule its own blocking "
        "finding quotes")


def test_a_rule_a_judgment_cites_is_never_reported_as_unchecked(conn):
    """The guard that came back. It was removed as unreachable, and it WAS — under the old
    comparison, where a rule confirmed before its own judgment could not sort after it. The
    comparison is the in-force history now, and `save_evaluation` accepts a judgment citing a
    rule standing in another market: cited here, out of scope then, in scope after a widening,
    and the report puts the rule its own blocking finding quotes on its "never checked
    against" list. Whatever the history says, a judgment that quotes a rule was checked
    against it — that is what citing it means."""
    for market in ("SEA", "EMEA", "APAC"):
        corrections.note(conn, text=SEEDING, provenance=f"{market} slide 4",
                         campaign_id=_campaign(conn, f"said-{market}", market=market))
    cid = corrections.find(conn, SEEDING)["correction_id"]
    corrections.graduate(conn, cid, confirmed_by="R. Vega")      # SEA, EMEA, APAC

    subject = _campaign(conn, "Colombia v1", market="LATAM")
    eid = core.save_evaluation(
        conn, subject_title="Colombia v1", campaign_id=subject, verdict="revise",
        summary="Seeds four.", approve_if="Seed one colourway.",
        findings=[{"severity": "blocking", "kind": "guardrail_breach",
                   "finding": "Seeds four colourways", "fix": "Seed one",
                   "precedent": {"correction_id": cid, "quote": SEEDING}}])["evaluation_id"]

    store.graduate_correction(conn, cid, markets=["SEA", "EMEA", "APAC", "LATAM"],
                              confirmed_by="R. Vega")

    rows = [r for r in replay.run(conn)["judgments"] if r["evaluation_id"] == eid]
    assert not [c for r in rows for c in r["not_checked_against"]["corrections"]
                if c["correction_id"] == cid], (
        "a judgment's own cited rule is on its 'never checked against' list")


def test_the_report_says_which_of_the_two_things_happened(conn):
    """A reader told a rule "became standing after this was judged" who then looks it up and
    finds a confirmation date from BEFORE the judgment concludes the report is broken — which
    is what the reviewer who found the missing judgments concluded. Both rules below were
    never applied to this brief, and for different reasons; the report names which."""
    subject = _campaign(conn, "Japan v1", market="Japan")
    moved = _standing_rule(conn)                       # LATAM, SEA, EMEA — not Japan
    _judged(conn, subject, title="Japan v1")
    fresh = _declared_everywhere(conn, "No price promises in client-facing creative.")
    store.graduate_correction(conn, moved, markets=["LATAM", "SEA", "EMEA", "Japan"],
                              confirmed_by="R. Vega")

    row = replay.run(conn)["judgments"][0]
    since = {c["correction_id"]: c["since"] for c in row["not_checked_against"]["corrections"]}
    assert since == {moved: "scope_changed", fresh: "became_standing"}, since
    assert "1 became standing here after this was judged" in row["what_it_means"]
    assert ("1 was already standing and did not apply to this brief until its scope changed"
            in row["what_it_means"]), row["what_it_means"]


def test_a_reconstructed_past_is_marked_as_one(conn):
    """The seeded row is an ASSUMPTION — the scope a rule already had, taken to have held
    since it was confirmed, because nothing on a database written before this table recorded
    when it started applying. It is the honest assumption and it is still an assumption, so it
    does not read like a recorded change (§2.4: a computed claim and a stated one must never
    read alike)."""
    cid = _standing_rule(conn)
    conn.execute("DELETE FROM correction_scope")       # as an older release left it
    conn.commit()

    store.graduate_correction(conn, cid, markets=["LATAM", "Japan"], confirmed_by="R. Vega")

    history = store.correction_scope_history(conn, cid)
    assert [h["basis"] for h in history] == ["heuristic", "computed"], history
    assert history[0]["expected_in"] == ["EMEA", "LATAM", "SEA"], "the past it assumed"
    assert history[1]["expected_in"] == ["Japan", "LATAM"], "the change it recorded"


def test_a_rule_put_back_after_being_set_aside_says_so_rather_than_blaming_its_scope(conn):
    """Its scope never moved. "Already standing and did not apply to this brief until its
    scope changed" is a cause the library cannot back — the third of three pasts, and the
    report had two words for them."""
    cid = _standing_rule(conn)
    corrections.set_aside(conn, cid, why="paused while the client rethinks")
    subject = _campaign(conn, "Colombia v1")
    _judged(conn, subject, title="Colombia v1")
    corrections.reopen(conn, cid)
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    row = replay.run(conn)["judgments"][0]
    assert [c["since"] for c in row["not_checked_against"]["corrections"]] == ["was_withdrawn"]
    assert ("1 was set aside when this was judged and has since been put back"
            in row["what_it_means"]), row["what_it_means"]
    assert "scope changed" not in row["what_it_means"]
    # A rule is a sentence and ends in a period; the report used to add a second one.
    assert ".. Whether" not in row["what_it_means"]


def test_a_judgment_written_after_a_repair_is_not_on_the_report(conn):
    """The flow this whole change ships for, from the other side. Nowhere → everywhere is the
    one scope change with an identical market list, so an equality test that compares markets
    alone reads it as no change at all — and then a brief judged AFTER the repair, which was
    checked against the rule, is reported as never checked. Mutation found nothing watching
    this: every other test judges before the repair."""
    cid = _standing_rule(conn)
    conn.execute("DELETE FROM correction_scope")       # an install from before the history
    conn.commit()
    store.graduate_correction(conn, cid, markets=[], confirmed_by="R. Vega",
                              applies_everywhere=False)    # in force nowhere, as it shipped
    store.graduate_correction(conn, cid, markets=[], confirmed_by="R. Vega",
                              applies_everywhere=True)     # the repair

    subject = _campaign(conn, "Peru v1", market="Peru")
    _judged(conn, subject, title="Peru v1")

    assert replay.run(conn)["judgments"] == [], (
        "a brief judged AFTER the repair was checked against these rules")


def test_a_judgment_written_after_a_rule_came_back_is_not_on_the_report(conn):
    """The same shape for a withdrawal: set aside, put back with the SAME markets, and only
    then judged. Nothing about that judgment was unchecked."""
    cid = _standing_rule(conn)
    corrections.set_aside(conn, cid, why="paused")
    corrections.reopen(conn, cid)
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    subject = _campaign(conn, "Colombia v1")
    _judged(conn, subject, title="Colombia v1")

    assert replay.run(conn)["judgments"] == []


def test_a_reconstructed_past_keeps_whether_the_rule_applied_everywhere(conn):
    """The seeded row carries `applies_everywhere`, not only the market list. Dropped, a
    global rule's reconstructed past reads as a rule scoped to nothing — so narrowing it
    afterwards would report every brief it HAD applied to as never checked."""
    cid = _standing_rule(conn)
    store.graduate_correction(conn, cid, markets=[], confirmed_by="R. Vega",
                              applies_everywhere=True)
    subject = _campaign(conn, "Peru v1", market="Peru")
    _judged(conn, subject, title="Peru v1")
    conn.execute("DELETE FROM correction_scope")       # as an older release left it
    conn.commit()

    store.graduate_correction(conn, cid, markets=["Peru"], confirmed_by="R. Vega")

    seeded = store.correction_scope_history(conn, cid)[0]
    assert seeded["applies_everywhere"] and seeded["basis"] == "heuristic"
    assert replay.run(conn)["judgments"] == [], (
        "a brief this rule already applied to came back as never checked")


def test_a_reopened_rule_does_not_accuse_a_judgment_that_never_cited_it(conn):
    """The same defect as the citing version above, with the citation removed — because the
    citation guard was masking it. A judgment that quotes the rule is excluded from this
    report whatever the history says, so the test that proved the seed stopped proving it the
    moment that guard came back, and the mutation pass said so.

    Plain judgment, rule standing when it was written, set aside, upgraded from a release with
    no history, reopened through the offer the loader makes: nothing was unchecked here."""
    subject = _campaign(conn, "Colombia v1")
    cid = _standing_rule(conn)
    _judged(conn, subject, title="Colombia v1")
    corrections.set_aside(conn, cid, why="paused while the client rethinks")
    conn.execute("DELETE FROM correction_scope")       # an install from before the history
    conn.commit()

    corrections.reopen(conn, cid)
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    assert replay.run(conn)["judgments"] == [], (
        "a rule this judgment WAS checked against came back as never applied")


def test_a_rule_widened_in_the_same_tick_as_a_judgment_does_not_hide_it(conn):
    """Two tables, two wall-clock floats, and no order between them when the floats are equal
    — which Windows makes reachable, measuring `time.time()` in whole milliseconds. Read
    through the timestamp alone, a rule widened in the same tick as a judgment counted as
    already in force, and the brief it was never checked against vanished from the report.

    The scope row records which judgments already existed when it was written, so the two are
    ordered by what happened rather than by a float comparison that cannot decide."""
    cid = _standing_rule(conn)                       # LATAM, SEA, EMEA
    subject = _campaign(conn, "Japan v1", market="Japan")
    eid = _judged(conn, subject, title="Japan v1")
    judged_at = store.get_evaluation(conn, eid)["created_at"]

    # The widening lands on the judgment's own timestamp, and is written after it.
    store.graduate_correction(conn, cid, markets=["LATAM", "SEA", "EMEA", "Japan"],
                              confirmed_by="R. Vega")
    conn.execute("UPDATE correction_scope SET changed_at = ? "
                 "WHERE correction_id = ? AND changed_at = "
                 "(SELECT MAX(changed_at) FROM correction_scope WHERE correction_id = ?)",
                 (judged_at, cid, cid))
    conn.commit()

    rows = replay.run(conn)["judgments"]
    assert [r["subject_title"] for r in rows] == ["Japan v1"], (
        "a brief judged before the widening, in the same tick, is not on the report")


def test_a_judgment_written_after_a_same_tick_change_is_not_listed(conn):
    """The other side of the same tie: the change was written FIRST and the judgment landed on
    the same float. It was checked, and saying otherwise is the accusing direction."""
    cid = _standing_rule(conn)
    store.graduate_correction(conn, cid, markets=["LATAM", "SEA", "EMEA", "Japan"],
                              confirmed_by="R. Vega")
    subject = _campaign(conn, "Japan v1", market="Japan")
    eid = _judged(conn, subject, title="Japan v1")
    changed_at = store.correction_scope_history(conn, cid)[-1]["changed_at"]
    conn.execute("UPDATE evaluations SET created_at = ? WHERE id = ?", (changed_at, eid))
    conn.commit()

    assert replay.run(conn)["judgments"] == []


def test_a_reconstructed_past_is_older_than_every_judgment_on_file(conn):
    """A seeded row describes a past, not a change somebody just made — so it cannot be
    "written after" a judgment, even one saved in the same tick as the confirmation it is
    dated from. Given the ordering of the write that triggered it, the rule would read as
    having started applying after a brief it had already been applied to."""
    subject = _campaign(conn, "Colombia v1")
    cid = _standing_rule(conn)
    eid = _judged(conn, subject, title="Colombia v1")
    judged_at = store.get_evaluation(conn, eid)["created_at"]
    # The rule was confirmed in the same tick the judgment was saved, which is all the tie
    # rule needs — and the seed is dated from `confirmed_at`.
    conn.execute("UPDATE corrections SET confirmed_at = ? WHERE id = ?", (judged_at, cid))
    conn.execute("DELETE FROM correction_scope")       # an install from before the history
    conn.commit()

    store.graduate_correction(conn, cid, markets=["LATAM", "SEA", "EMEA", "Japan"],
                              confirmed_by="R. Vega")

    seeded = store.correction_scope_history(conn, cid)[0]
    assert seeded["basis"] == "heuristic" and seeded["after_evaluation"] == 0
    assert replay.run(conn)["judgments"] == [], (
        "a brief this rule had already been applied to came back as never checked")


def test_a_measure_confirmed_in_the_judgment_s_tick_is_not_hidden(conn):
    """The rules half got the cross-table ordering and the measures half did not, so the two
    answered differently about the same instant. A measure confirmed after a judgment — in the
    same clock tick — is a gap that judgment was never checked for."""
    subject = _campaign(conn, "Colombia v1")
    eid = _judged(conn, subject, title="Colombia v1")
    name = _expect_measure(conn)
    judged_at = store.get_evaluation(conn, eid)["created_at"]
    conn.execute("UPDATE metric_registry SET confirmed_at = ? WHERE canonical = ?",
                 (judged_at, name))
    conn.commit()

    rows = [r for r in replay.run(conn)["judgments"] if r["evaluation_id"] == eid]
    assert rows, "a measure confirmed after this judgment, in its tick, is not on the report"
    assert name in rows[0]["not_checked_against"]["measures"]


def test_a_measure_confirmed_before_a_judgment_in_one_tick_stays_off_the_report(conn):
    """The other direction of the same tie: confirmed first, judged in the same tick. That
    brief WAS checked for it."""
    name = _expect_measure(conn)
    subject = _campaign(conn, "Colombia v1")
    eid = _judged(conn, subject, title="Colombia v1")
    confirmed_at = store.metric_registry(conn)[name]["confirmed_at"]
    conn.execute("UPDATE evaluations SET created_at = ? WHERE id = ?", (confirmed_at, eid))
    conn.commit()

    assert [r for r in replay.run(conn)["judgments"] if r["evaluation_id"] == eid] == []


def test_a_rule_first_confirmed_in_the_judgment_s_tick_is_not_blamed_on_its_scope(conn):
    """The rule is on the report for the right reason and described with the wrong one: it
    had no scope to change, because nobody had confirmed it yet. `confirmed_at > judged_at` is
    false on equality, so the one comparison that decides WHICH sentence a reader gets was
    still deciding it on a float."""
    subject = _campaign(conn, "Colombia v1")
    eid = _judged(conn, subject, title="Colombia v1")
    cid = _standing_rule(conn)
    judged_at = store.get_evaluation(conn, eid)["created_at"]
    conn.execute("UPDATE corrections SET confirmed_at = ? WHERE id = ?", (judged_at, cid))
    conn.execute("UPDATE correction_scope SET changed_at = ? WHERE correction_id = ?",
                 (judged_at, cid))
    conn.commit()

    row = [r for r in replay.run(conn)["judgments"] if r["evaluation_id"] == eid][0]
    assert [c["since"] for c in row["not_checked_against"]["corrections"]] == [
        "became_standing"], "a rule that had no scope yet was reported as having changed one"
    assert "scope changed" not in row["what_it_means"]
