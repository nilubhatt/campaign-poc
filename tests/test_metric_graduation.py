"""
§8.3–8.5: graduation, a fixed template, and retirement.

The review on graduation: *"A measure becomes part of the checklist for a campaign type once
it has been seen in N campaigns, across at least two partners or markets, and confirmed once
by a person. Count alone is not enough — one partner's house metric should never quietly
become a standing requirement for everyone. Then `prepare_evaluation` renders the expected set
as computed facts automatically: 'Expected for a store launch: budget, reach, footfall uplift,
sell-through at 60 days. This brief carries none of the four.' No prompt was edited to make
that appear."*

**"Count alone is not enough"** is the sentence that shapes the gate. A measure seen fifteen
times in one market is one partner's habit, and promoting it makes every other market fail a
checklist it never agreed to. Three conditions, all required, and the third is a person —
§8.2's question was "the only human step in the loop" and this is where it is spent.

On the template (§8.4): *"Grow the data the prompt renders — never the prompt."* The last
sentence of the graduation paragraph is the test: **no prompt was edited to make that appear.**
The expected set is rendered from the registry at call time, so a measure graduating changes
what a brief is checked against without anybody touching a string.

On retirement (§8.5): *never delete.* A measure that stops appearing stops being expected, and
the record of it stays — because "we used to track this" is an answer, and a deleted row can
only say "we never did".
"""
import pytest

import metrics


def _campaign(conn, title, market="LATAM"):
    import core
    return core.ingest_campaign(conn, title=title, market=market, status="concluded",
                                detail="A launch.")["campaign_id"]


def _seen(conn, key, *, markets, times=1):
    for n, market in enumerate(markets):
        for t in range(times):
            metrics.record(conn, campaign_id=_campaign(conn, f"{key}-{market}-{n}-{t}",
                                                       market=market),
                           key=key, value=1.0 + t)


# ── the gate ────────────────────────────────────────────────────────────────

def test_a_measure_seen_widely_enough_can_graduate(conn):
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.resolve(conn, "footfall_uplift", decision="different_measure")

    assert metrics.graduation(conn, "footfall_uplift")["eligible"] is True
    metrics.graduate(conn, "footfall_uplift", confirmed_by="the product owner")
    assert metrics.describe(conn, "footfall_uplift")["status"] == "expected"


def test_count_alone_is_not_enough(conn):
    """"One partner's house metric should never quietly become a standing requirement for
    everyone." Fifteen sightings in one market is a habit, not a standard."""
    _seen(conn, "footfall_uplift_pct", markets=["LATAM"], times=15)
    metrics.resolve(conn, "footfall_uplift", decision="different_measure")

    gate = metrics.graduation(conn, "footfall_uplift")
    assert gate["eligible"] is False
    assert gate["markets"] == 1
    assert "market" in gate["what_it_means"].lower()


def test_breadth_alone_is_not_enough_either(conn):
    """Two markets and two sightings is two people trying something once."""
    _seen(conn, "dwell_time_s", markets=["LATAM", "SEA"])
    metrics.resolve(conn, "dwell_time", decision="different_measure")

    gate = metrics.graduation(conn, "dwell_time")
    assert gate["eligible"] is False
    assert gate["campaigns"] < metrics.GRADUATION_CAMPAIGNS


def test_the_gate_counts_campaigns_not_writes(conn):
    """"Seen in N campaigns." A brief that logs a stated figure and a recomputed correction has
    written the measure twice and carried it once — and `times_seen` counts writes, so a gate
    reading that would promote on one brief's opinion recorded three times."""
    cid = _campaign(conn, "Peru")
    for source in ("stated", "recomputed", "verified"):
        metrics.record(conn, campaign_id=cid, key=f"dwell_time_s_{source}", value=1.0)

    assert metrics.describe(conn, "dwell_time")["times_seen"] == 3
    assert metrics.graduation(conn, "dwell_time")["campaigns"] == 1


def test_a_measure_somebody_set_aside_does_not_graduate(conn):
    """"Ignore" was a decision about the vocabulary. Promoting it to a standing requirement
    over the top of that decision would make the answer meaningless."""
    _seen(conn, "junk_key", markets=["LATAM", "SEA", "EMEA"])
    metrics.resolve(conn, "junk_key", decision="ignore")

    assert metrics.graduation(conn, "junk_key")["eligible"] is False
    with pytest.raises(ValueError):
        metrics.graduate(conn, "junk_key", confirmed_by="R. Vega")


def test_a_promotion_with_nobody_behind_it_is_refused(conn):
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    with pytest.raises(ValueError) as e:
        metrics.graduate(conn, "footfall_uplift", confirmed_by="   ")
    assert "person" in str(e.value).lower()


def test_a_shipped_name_is_not_a_standing_requirement_on_day_one(conn):
    """The registry ships twelve measures. If being in it meant being expected, a brand-new
    install would fail every brief against a checklist nobody's library had earned — and a
    check that fires on everything is one nobody reads."""
    assert metrics.expected_for(conn, market="LATAM") == []
    assert metrics.describe(conn, "cpm")["status"] == "known"


def test_a_person_has_to_confirm_it(conn):
    """"Confirmed once by a person." §8.2's question was "the only human step in the loop", and
    this is where it is spent — an automatic promotion is how a checklist grows teeth nobody
    agreed to."""
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.resolve(conn, "footfall_uplift", decision="different_measure")

    assert metrics.describe(conn, "footfall_uplift")["status"] == "provisional"
    assert metrics.graduation(conn, "footfall_uplift")["eligible"] is True, \
        "eligible is not the same as promoted"


def test_graduating_an_ineligible_measure_is_refused_with_the_reason(conn):
    _seen(conn, "footfall_uplift_pct", markets=["LATAM"])
    with pytest.raises(ValueError) as e:
        metrics.graduate(conn, "footfall_uplift", confirmed_by="somebody")
    assert "market" in str(e.value).lower()


def test_graduating_a_measure_stops_it_asking_to_be_identified(conn):
    """Promoting a measure to the checklist is a stronger answer than §8.2's question wants.
    One that is expected of every brief while still asking "is this a new measure?" is the
    product asking a question it has already acted on."""
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")

    assert metrics.unanswered(conn) == []
    assert metrics.describe(conn, "footfall_uplift")["answered"] is True


def test_who_confirmed_it_is_recorded(conn):
    """A standing requirement that nobody's name is against is one nobody can question later."""
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.resolve(conn, "footfall_uplift", decision="different_measure")
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")

    assert metrics.describe(conn, "footfall_uplift")["confirmed_by"] == "R. Vega"


# ── §8.4: the data grows, the prompt does not ───────────────────────────────

def test_the_expected_set_is_rendered_not_written(conn):
    """"No prompt was edited to make that appear." The expected set comes from the registry at
    call time, so a measure graduating changes what a brief is checked against without
    anybody touching a string."""
    package_before = metrics.expected_for(conn, market="LATAM")
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.resolve(conn, "footfall_uplift", decision="different_measure")
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")

    assert "footfall_uplift" not in package_before
    assert "footfall_uplift" in metrics.expected_for(conn, market="LATAM")


def test_a_brief_is_told_which_expected_measures_it_carries(conn):
    """"Expected for a store launch: budget, reach, footfall uplift, sell-through at 60 days.
    This brief carries none of the four."""
    for key in ("footfall_uplift_pct", "sell_through_pct"):
        _seen(conn, key, markets=["LATAM", "SEA", "EMEA"])
        metrics.graduate(conn, metrics.canonical(conn, key), confirmed_by="R. Vega")
    subject = _campaign(conn, "Colombia v1")

    report = metrics.expected_check(conn, subject)
    assert sorted(report["missing"]) == ["footfall_uplift", "sell_through"]
    assert report["carried"] == []
    assert "none of the two" in report["what_it_means"].lower()


def test_the_count_is_written_the_way_a_person_writes_it(conn):
    """The review's own sentence is "none of the four", and a server that renders "none of
    the 1" has written something nobody would say — which is what this did until a real
    protocol round-trip printed it."""
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")
    subject = _campaign(conn, "Colombia v1")

    said = metrics.expected_check(conn, subject)["what_it_means"]
    assert "none of the 1" not in said
    assert said.endswith("This brief does not carry it.")


def test_a_brief_that_carries_them_is_told_that_too(conn):
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.resolve(conn, "footfall_uplift", decision="different_measure")
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")
    subject = _campaign(conn, "Colombia v1")
    metrics.record(conn, campaign_id=subject, key="footfall_uplift_pct", value=9.0)

    report = metrics.expected_check(conn, subject)
    assert report["carried"] == ["footfall_uplift"]
    assert report["missing"] == []


def test_a_measure_expected_only_in_another_market_is_not_missing_here(conn):
    """The review's own framing is "the checklist for a campaign TYPE". A measure that
    graduated on the strength of LATAM and SEA is not a gap in a market nobody has used it
    in — and reporting it would make every new market fail a checklist on day one."""
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.resolve(conn, "footfall_uplift", decision="different_measure")
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")

    elsewhere = _campaign(conn, "Tokyo v1", market="APAC")
    assert metrics.expected_check(conn, elsewhere)["missing"] == []


# ── §8.5: retirement, never deletion ────────────────────────────────────────

def test_a_measure_nobody_has_recorded_lately_is_demoted(conn):
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.resolve(conn, "footfall_uplift", decision="different_measure")
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")

    last = None
    for n in range(metrics.RETIREMENT_AFTER):
        last = metrics.record(conn, campaign_id=_campaign(conn, f"Later {n}"), key="reach",
                              value=1.0)

    # On the write that makes it true, not on whoever happens to read next.
    assert [r["measure"] for r in last["retired"]] == ["footfall_uplift"]
    assert metrics.describe(conn, "footfall_uplift")["status"] == "retired"


def test_a_retired_measure_is_kept_not_deleted(conn):
    """"Never delete." "We used to track this" is an answer; a deleted row can only say "we
    never did" — and the values recorded against it are somebody's measurements."""
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.resolve(conn, "footfall_uplift", decision="different_measure")
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")
    for n in range(metrics.RETIREMENT_AFTER):
        metrics.record(conn, campaign_id=_campaign(conn, f"Later {n}"), key="reach", value=1.0)
    metrics.retire_stale(conn)

    assert metrics.describe(conn, "footfall_uplift")
    assert metrics.across_library(conn, "footfall_uplift")


def test_a_retired_measure_stops_being_expected(conn):
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.resolve(conn, "footfall_uplift", decision="different_measure")
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")
    for n in range(metrics.RETIREMENT_AFTER):
        metrics.record(conn, campaign_id=_campaign(conn, f"Later {n}"), key="reach", value=1.0)
    metrics.retire_stale(conn)

    assert "footfall_uplift" not in metrics.expected_for(conn, market="LATAM")


def test_a_measure_never_recorded_has_not_gone_stale(conn):
    """"Stopped appearing" needs a start. A measure with no sighting to count from must not be
    retired by an arithmetic accident — and the guard that said so was a branch no test could
    reach, so this pins the behaviour where it actually lives."""
    import store

    for n in range(RETIREMENT_SPAN := metrics.RETIREMENT_AFTER + 2):
        metrics.record(conn, campaign_id=_campaign(conn, f"A{n}"), key="reach", value=1.0)

    # Asked AFTER the library has campaigns in it: on an empty table this returns 0 whatever
    # it does with `when`, so the version of this that ran first asserted nothing — treating a
    # missing `last_seen` as the epoch would have passed it.
    assert store.campaigns_that_skipped(conn, "cpm", since=None) == 0
    assert store.campaigns_that_skipped(conn, "cpm", since=0) == RETIREMENT_SPAN
    assert [r["measure"] for r in metrics.retire_stale(conn)] == []


def test_a_measure_still_in_use_is_not_retired(conn):
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.resolve(conn, "footfall_uplift", decision="different_measure")
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")

    assert metrics.retire_stale(conn) == []


def test_a_retired_measure_that_comes_back_is_expected_again(conn):
    """It graduated once and a person confirmed it. Making them confirm it again because a
    quarter went by would be asking the same question twice."""
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.resolve(conn, "footfall_uplift", decision="different_measure")
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")
    for n in range(metrics.RETIREMENT_AFTER):
        metrics.record(conn, campaign_id=_campaign(conn, f"Later {n}"), key="reach", value=1.0)
    metrics.retire_stale(conn)

    metrics.record(conn, campaign_id=_campaign(conn, "Back"), key="footfall_uplift_pct",
                   value=3.0)
    assert metrics.describe(conn, "footfall_uplift")["status"] == "expected"


def test_the_note_tells_the_model_what_the_gap_is_and_is_not(conn):
    """A block the model is handed with no instruction is one it will read however it likes.
    And the instruction has to say MISSING rather than wrong: a brief that omits a measure the
    market usually reports may be incomplete or may be a kind of campaign where the measure is
    meaningless, and the server cannot tell which (§6.5)."""
    import core

    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")
    subject = _campaign(conn, "Colombia v1")

    note = core.prepare_evaluation(conn, subject_title="Colombia v1",
                                   proposal_text="A launch.", campaign_id=subject)["note"]
    assert "footfall_uplift" in note
    assert "not a verdict" in note


def test_the_tool_description_tells_the_model_what_the_block_is(conn):
    """A field in the package that no description mentions is one the model reads however it
    likes — most plausibly by turning every miss into an uncited `missing_information` finding,
    which D78 already records as under evidentiary pressure. Documenting the field is not
    editing the prompt that generates its contents, so §8.4's claim survives."""
    import mcp_server

    described = mcp_server._PREPARE_EVALUATION_DESCRIPTION
    assert "expected_measures" in described
    assert "not a verdict on it" in described
    assert "nothing_to_check" in described


def test_the_note_says_nothing_when_there_is_no_checklist(conn):
    """On a new install nothing has graduated. Explaining a checklist mechanism to a model
    that has just been handed an empty checklist is a note that fires on everything."""
    import core

    subject = _campaign(conn, "Colombia v1")
    note = core.prepare_evaluation(conn, subject_title="Colombia v1",
                                   proposal_text="A launch.", campaign_id=subject)["note"]
    assert "expected_measures" not in note


# ── what two independent reviews found ──────────────────────────────────────

def test_two_markets_is_enough(conn):
    """"At least TWO" is the requirement, and every test here used three — so a gate quietly
    demanding a third passed the whole suite. The boundary is the number the rule is about."""
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA"])
    metrics.record(conn, campaign_id=_campaign(conn, "third", market="SEA"),
                   key="footfall_uplift_pct", value=2.0)

    gate = metrics.graduation(conn, "footfall_uplift")
    assert (gate["campaigns"], gate["markets"]) == (3, 2)
    assert gate["eligible"] is True


def test_one_market_spelled_three_ways_is_one_market(conn):
    """The gate exists to stop one partner's house metric becoming everyone's requirement, and
    it was satisfied by "SEA", "sea" and "Sea". Every other market comparison in the codebase
    folds case (C16); this one did not."""
    _seen(conn, "footfall_uplift_pct", markets=["SEA", "sea", "Sea"])

    gate = metrics.graduation(conn, "footfall_uplift")
    assert gate["markets"] == 1
    assert gate["eligible"] is False


def test_a_checklist_applies_whatever_the_market_is_typed_as(conn):
    """The same fold, from the other side: a brief filed as `latam` was silently exempt from
    the LATAM checklist, which is the failure being invisible rather than loud."""
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")

    lower = _campaign(conn, "Colombia v1", market="latam")
    assert metrics.expected_check(conn, lower)["missing"] == ["footfall_uplift"]


def test_a_campaign_that_ran_in_two_markets_counts_as_two(conn):
    """`markets` is the only way to express a multi-country activation. Reading `market or
    region` alone reported a campaign that literally ran in MX and CO as covering none — the
    fifth implementation of one question, drifting exactly as D55 and D88 describe."""
    import core

    for n in range(3):
        cid = core.ingest_campaign(conn, title=f"Andes {n}", markets=["MX", "CO"],
                                   status="concluded", detail="A launch.")["campaign_id"]
        metrics.record(conn, campaign_id=cid, key="footfall_uplift_pct", value=1.0)

    assert metrics.graduation(conn, "footfall_uplift")["markets"] == 2


def test_three_versions_of_one_brief_are_one_brief(conn):
    """"Seen in N campaigns" means N different campaigns. v1/v2/v3 of one partner's deck is
    one brief revised twice, and counting the records let a single deck satisfy a gate that
    means three campaigns carried it — the write-counting mistake one level up."""
    import core

    prev = None
    for n in range(3):
        prev = core.ingest_campaign(conn, title=f"Colombia v{n + 1}", market="LATAM",
                                    supersedes=prev, status="concluded",
                                    detail="A launch.")["campaign_id"]
        metrics.record(conn, campaign_id=prev, key="footfall_uplift_pct", value=1.0)

    assert metrics.graduation(conn, "footfall_uplift")["campaigns"] == 1


def test_a_brief_with_no_market_is_not_held_to_every_checklist(conn):
    """A record with no market is most of a young library, and it was being measured against
    every expectation anybody had earned anywhere — the check that fires on everything,
    arrived at through an `if market and ...` that read as a convenience."""
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")

    import core
    loose = core.ingest_campaign(conn, title="Somewhere", status="planned",
                                 detail="A launch.")["campaign_id"]
    report = metrics.expected_check(conn, loose)
    assert report["missing"] == []
    assert report["status"] == "nothing_to_check", "an empty list is not a pass"
    assert report["code"] == "no_market"


def test_a_reference_record_is_not_a_brief(conn):
    """Brand guidelines were being told they were missing footfall uplift. §7.1 was careful
    about which TEXT a check reads; this was not careful about which RECORD it runs on."""
    import core

    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")
    ref = core.ingest_campaign(conn, title="Brand guidelines", record_type="reference",
                               market="LATAM", detail="Guidelines.")["campaign_id"]

    assert metrics.expected_check(conn, ref)["code"] == "not_a_campaign"


def test_a_target_is_not_a_result(conn):
    """A concluded campaign holding nothing but targets was told it "carries all of them" — a
    clean bill of health for a record with no results at all, which is §5.3's mistake in a new
    place."""
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")
    subject = _campaign(conn, "Colombia v1")
    metrics.record(conn, campaign_id=subject, key="footfall_uplift_pct", value=9.0,
                   metric_type="target")

    assert metrics.expected_check(conn, subject)["missing"] == ["footfall_uplift"]


def test_the_gate_answers_to_the_name_the_user_actually_has(conn):
    """`footfall_uplift` is an artefact of decoration-stripping, surfaced once inside §8.2's
    question and possibly months ago. The raw key is what is in the spreadsheet, and a gate
    that only answers to the stem is one nobody can reach."""
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])

    assert metrics.graduation(conn, "footfall_uplift_pct")["measure"] == "footfall_uplift"
    assert metrics.graduation(conn, "crm_reach")["measure"] == "reach"


def test_an_already_expected_measure_says_so_instead_of_asking_for_more_data(conn):
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")

    gate = metrics.graduation(conn, "footfall_uplift")
    assert gate["code"] == "already_expected"
    assert gate["eligible"] is False
    assert "R. Vega" in gate["what_it_means"]


def test_confirming_twice_cannot_erase_who_confirmed_it(conn):
    """The one audit field the whole gate exists to create, reached by following the tool's own
    advice: a second `graduate` overwrote the name of the person who actually confirmed it."""
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")

    with pytest.raises(ValueError):
        metrics.graduate(conn, "footfall_uplift", confirmed_by="somebody else")
    assert metrics.describe(conn, "footfall_uplift")["confirmed_by"] == "R. Vega"


def test_a_set_aside_measure_is_not_told_to_collect_more_data(conn):
    """Asked counts-first, an ignored measure read "seen in 1 campaign, needs 3" — telling the
    user to keep recording something that will be refused forever, and re-asking a question
    they had declined. §8.2 named that failure one item ago."""
    metrics.record(conn, campaign_id=_campaign(conn, "A"), key="junk_key", value=1.0)
    metrics.resolve(conn, "junk_key", decision="ignore")

    gate = metrics.graduation(conn, "junk_key")
    assert gate["code"] == "set_aside"
    assert "needs" not in gate["what_it_means"]


def test_answering_the_new_measure_question_cannot_undo_a_graduation(conn):
    """The realistic route in is §8.2's offers, emitted at first sighting, being accepted
    months later. `different_measure` would drop it off every checklist and `same_thing` runs
    `merge_metric`, which DELETES the row — taking `confirmed_by` and `expected_in` with it and
    breaking §8.5's "never delete" from the one direction nothing was watching."""
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")

    for decision, extra in (("different_measure", {}), ("ignore", {}),
                            ("same_thing", {"same_as": "reach"})):
        with pytest.raises(ValueError) as e:
            metrics.resolve(conn, "footfall_uplift", decision=decision, **extra)
        assert "already expected" in str(e.value)
    assert metrics.describe(conn, "footfall_uplift")["confirmed_by"] == "R. Vega"


# ── the human step is reachable ─────────────────────────────────────────────

def test_the_write_that_makes_a_measure_eligible_asks(conn):
    """The gate's third condition is a person, so somebody has to be ASKED. Nothing surfaced
    eligibility, which left a tool the user would have to know exists, think to call, and name
    by its canonical stem — so no measure would ever graduate and §8.4's sentence could never
    fire. §8.2 solved the same problem the same way: the question rides on the write."""
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA"])
    third = metrics.record(conn, campaign_id=_campaign(conn, "third", market="EMEA"),
                           key="footfall_uplift_pct", value=3.0)

    asked = third["newly_eligible"]
    assert asked["measure"] == "footfall_uplift"
    assert asked["next_actions"][0]["tool"] == "graduate_measure"
    assert "somebody has to say so" in asked["what_it_means"]


def test_the_offer_does_not_supply_the_confirmation_itself(conn):
    """`confirmed_by` is the one argument the server must not fill in. Prefilled with a name
    nobody gave, the offer would manufacture the very confirmation the gate exists to require
    (§6.5) — and left out silently it would simply fail when accepted (§5.2). `needs` is how
    this codebase already says "one more thing, and it is yours to supply"."""
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA"])
    third = metrics.record(conn, campaign_id=_campaign(conn, "third", market="EMEA"),
                           key="footfall_uplift_pct", value=3.0)

    offer = third["newly_eligible"]["next_actions"][0]
    assert "confirmed_by" not in offer["prefilled_args"]
    assert any("confirmed_by" in n for n in offer["needs"])


def test_every_offered_argument_is_one_the_tool_accepts(conn):
    """§5.2's standing check: an offer whose write path does not exist is worse than no offer."""
    import inspect

    import mcp_server

    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA"])
    third = metrics.record(conn, campaign_id=_campaign(conn, "third", market="EMEA"),
                           key="footfall_uplift_pct", value=3.0)

    for offer in third["newly_eligible"]["next_actions"]:
        signature = inspect.signature(getattr(mcp_server, offer["tool"]))
        assert set(offer["prefilled_args"]) <= set(signature.parameters)
        # And what `needs` names is a real argument too, or it is advice nobody can act on.
        for need in offer.get("needs", []):
            assert need.split()[0].strip(" —:") in signature.parameters


def test_it_asks_once(conn):
    """§8.2's rule, and the same flag. A question on every subsequent write is one nobody
    reads, and this one arrives on an ordinary metrics upload."""
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    again = metrics.record(conn, campaign_id=_campaign(conn, "fourth", market="APAC"),
                           key="footfall_uplift_pct", value=4.0)

    assert "newly_eligible" not in again


def test_an_ineligible_measure_asks_nothing(conn):
    one = metrics.record(conn, campaign_id=_campaign(conn, "A"), key="footfall_uplift_pct",
                         value=1.0)
    assert "newly_eligible" not in one


def test_the_offer_reaches_the_model_through_add_metrics(conn):
    import core

    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA"])
    third = _campaign(conn, "third", market="EMEA")
    added = core.add_metrics(conn, campaign_id=third,
                             structured={"footfall_uplift_pct": 3.0}, confirm=True)

    assert added["newly_eligible"][0]["measure"] == "footfall_uplift"


# ── retirement, counted the way the sentence means it ───────────────────────

def test_retirement_counts_campaigns_that_skipped_it(conn):
    """A bulk import of historical briefs aged out every measure at once — including, in the
    reviewed scenario, the measure carried by 53 of the 63 campaigns on file. A measure the
    library keeps reporting has not stopped appearing, whatever was imported alongside it."""
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")

    for n in range(metrics.RETIREMENT_AFTER + 5):
        cid = _campaign(conn, f"Import {n}")
        metrics.record(conn, campaign_id=cid, key="reach", value=1.0)
        metrics.record(conn, campaign_id=cid, key="footfall_uplift_pct", value=2.0)

    assert metrics.describe(conn, "footfall_uplift")["status"] == "expected"


def test_one_campaign_reporting_many_numbers_is_one_campaign(conn):
    """"Demote after M campaigns", not after M numbers."""
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")

    busy = _campaign(conn, "One busy brief")
    for key in ("reach", "impressions", "roas", "ctr", "cpa", "cpm", "budget",
                "conversions", "views", "traffic", "sell_through_pct", "er_pct"):
        metrics.record(conn, campaign_id=busy, key=key, value=1.0)

    assert metrics.describe(conn, "footfall_uplift")["status"] == "expected"


def test_a_market_that_never_used_it_is_not_evidence_it_fell_out_of_use(conn):
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")

    for n in range(metrics.RETIREMENT_AFTER + 2):
        metrics.record(conn, campaign_id=_campaign(conn, f"Tokyo {n}", market="APAC"),
                       key="reach", value=1.0)

    assert metrics.describe(conn, "footfall_uplift")["status"] == "expected"


def test_a_campaign_topping_up_its_numbers_has_not_skipped_the_measure(conn):
    """"Campaigns that SKIPPED it" is a question about the campaign, not about the write. A
    brief that reported footfall in March and adds its reach figure in June has not stopped
    reporting footfall — counting it as an absence retires a measure on the strength of
    campaigns that carry it."""
    import store

    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    early = [_campaign(conn, f"Early {n}") for n in range(metrics.RETIREMENT_AFTER + 2)]
    for cid in early:
        metrics.record(conn, campaign_id=cid, key="footfall_uplift_pct", value=1.0)
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")
    since = metrics.describe(conn, "footfall_uplift")["last_seen"]

    # Every one of them adds a different figure later. None has skipped footfall.
    for cid in early:
        metrics.record(conn, campaign_id=cid, key="reach", value=1.0)

    assert store.campaigns_that_skipped(conn, "footfall_uplift", since=since,
                                        markets=["LATAM"]) == 0
    assert metrics.describe(conn, "footfall_uplift")["status"] == "expected"


def test_a_registry_row_written_before_the_fold_still_counts_one_market(conn):
    """`touch_metric` folds on the way in now, so new rows cannot hold "LATAM" and "latam"
    together — but rows written by the release that did not fold can, and the gate has to read
    them correctly or an upgraded database graduates on a spelling."""
    import json

    import store

    conn.execute("INSERT INTO metric_registry (canonical, display_name, times_seen, markets) "
                 "VALUES ('legacy_measure', 'Legacy', 3, ?)",
                 (json.dumps(["SEA", "sea", "Sea"]),))
    conn.commit()

    assert metrics.graduation(conn, "legacy_measure")["markets"] == 1


def test_the_expected_set_is_never_asked_for_without_a_market(conn):
    """`expected_for` with no market returned the union of every checklist. `expected_check`
    no longer reaches it that way, but it is a function anything can call, and the union is
    the wrong answer however you arrive at it."""
    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")

    assert metrics.expected_for(conn) == []
    assert metrics.expected_for(conn, markets=[]) == []
    assert metrics.expected_for(conn, market="LATAM") == ["footfall_uplift"]


def test_only_sightings_after_the_last_one_count(conn):
    """"Since last seen." Counting every campaign in the library instead would retire a measure
    reported yesterday as soon as the library held ten briefs."""
    import store

    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    for n in range(metrics.RETIREMENT_AFTER + 2):
        metrics.record(conn, campaign_id=_campaign(conn, f"Before {n}"), key="reach", value=1.0)
    # Seen again LAST, so nothing has skipped it since.
    metrics.record(conn, campaign_id=_campaign(conn, "Latest"), key="footfall_uplift_pct",
                   value=5.0)
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")

    entry = metrics.describe(conn, "footfall_uplift")
    assert store.campaigns_that_skipped(conn, "footfall_uplift", since=entry["last_seen"],
                                        markets=entry["expected_in"]) == 0
    assert metrics.retire_stale(conn) == []


def test_a_retirement_is_reported_not_silent(conn):
    """It ran inside `expected_check` first, so `prepare_evaluation` quietly mutated the
    registry, the returned list was discarded, and *who read* decided *what was demoted*.
    §2.1's rule is that partial state is never silent."""
    import core

    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")
    for n in range(metrics.RETIREMENT_AFTER - 1):
        metrics.record(conn, campaign_id=_campaign(conn, f"Later {n}"), key="reach", value=1.0)

    told = core.add_metrics(conn, campaign_id=_campaign(conn, "The last one"),
                            structured={"reach": 1.0}, confirm=True)
    assert told["retired_measures"][0]["measure"] == "footfall_uplift"
    assert "still on file" in told["retired_measures"][0]["what_it_means"]


def test_preparing_an_evaluation_does_not_mutate_the_registry(conn):
    import core

    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")
    for n in range(metrics.RETIREMENT_AFTER + 3):
        metrics.record(conn, campaign_id=_campaign(conn, f"Later {n}"), key="reach", value=1.0)
    before = metrics.describe(conn, "footfall_uplift")["status"]

    subject = _campaign(conn, "Colombia v1")
    core.prepare_evaluation(conn, subject_title="Colombia v1", proposal_text="A launch.",
                            campaign_id=subject)

    assert metrics.describe(conn, "footfall_uplift")["status"] == before


def test_it_reaches_the_model_over_the_protocol(conn):
    import asyncio
    import json
    import mcp_server

    async def call(name, args):
        return json.loads((await mcp_server.mcp.call_tool(name, args)).content[0].text)

    _seen(conn, "footfall_uplift_pct", markets=["LATAM", "SEA", "EMEA"])
    metrics.resolve(conn, "footfall_uplift", decision="different_measure")

    promoted = asyncio.run(call("graduate_measure",
                                {"measure": "footfall_uplift", "confirmed_by": "R. Vega"}))
    assert promoted["status"] == "expected"

    subject = _campaign(conn, "Colombia v1")
    package = asyncio.run(call("prepare_evaluation",
                               {"subject_title": "Colombia v1", "proposal_text": "A launch.",
                                "campaign_id": subject}))
    assert "footfall_uplift" in package["expected_measures"]["missing"]
