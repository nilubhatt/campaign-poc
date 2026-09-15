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
    assert replay.run(conn)["rulebook_version"] == core.RULEBOOK_VERSION


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
