"""
§7.1: compute what can be computed. Reason only about the rest.

The review's words: *"Of the findings produced in two real evaluations today, roughly two
thirds were mechanical: no dated flighting, no engagement rate on any profile, no budget
figure, missing 360 channels, three internal date contradictions, a 'weekend' falling on a
Tuesday. None of that needs a language model, and anything a model decides is something a
model can decide differently tomorrow. Fix: run these as code in `prepare_evaluation` and
return them as computed facts with the evidence attached — date coverage, ER presence per
profile row, budget detected yes/no, channel checklist hit/miss, internal date consistency,
guardrail keyword hits. The model then reasons only about what is genuinely judgment."*

The sentence that decides the design is **"anything a model decides is something a model can
decide differently tomorrow"**. This is Phase 7's subject — consistency across users — and the
mechanism is not a better prompt, it is removing the question. Two thirds of the output stops
being generated at all.

Three things follow.

**Body layer only** (D11). A reviewer's comment saying "make sure we never mention adidas"
would otherwise register as the brief mentioning adidas: the check would read somebody's
warning as the thing they were warning about. §2.5 built the layer separation for exactly this
and §7.1 is the first consumer.

**Evidence attached, or it is an assertion in a different coat.** A computed fact that says
"no budget" without showing what it looked at cannot be argued with, and the whole point is
that a person can check it. Every present fact carries the text it found.

**Absent is a finding; unknown is not.** "This brief carries no dates" is a mechanical fact.
"This brief carries no dates that I could parse" is a fact about the parser, and reporting the
second as the first is the confident-unfounded-claim failure this review is built around.
"""
import pytest

import facts


# ── the six checks, on the review's own examples ────────────────────────────

def test_a_brief_with_no_dated_flighting_says_so():
    """"No dated flighting" — the review's first mechanical finding."""
    found = facts.compute("Flighting: hero asset, three creator posts, one story set. "
                          "Assets delivered to the agency ahead of launch.")

    dates = found["date_coverage"]
    assert dates["status"] == "absent"
    assert dates["dates_found"] == 0


@pytest.mark.parametrize("text,count", [
    ("Launch runs 3 March to 28 March across Lima.", 2),
    ("Posting dates: 2026-03-03, 2026-03-10 and 2026-03-17.", 3),
    ("Go live 3rd of March, wrap on 28th March.", 2),
])
def test_dates_are_found_however_they_are_written(text, count):
    assert facts.compute(text)["date_coverage"]["dates_found"] == count


def test_no_engagement_rate_on_any_profile_is_reported():
    """"No engagement rate on any profile" — and the check has to know a profile row when it
    sees one, or an absent ER on a brief with no creators reads as a defect."""
    found = facts.compute(
        "Creators: @lima_style (180k followers), @andes_fit (95k followers).")

    er = found["engagement_rate"]
    assert er["status"] == "absent"
    assert er["profiles_found"] == 2
    assert er["profiles_with_rate"] == 0


def test_a_brief_with_engagement_rates_is_not_flagged():
    found = facts.compute(
        "Creators: @lima_style (180k followers, 3.4% ER), "
        "@andes_fit (95k followers, engagement rate 2.8%).")

    assert found["engagement_rate"]["status"] == "present"
    assert found["engagement_rate"]["profiles_with_rate"] == 2


def test_a_brief_with_no_creators_is_not_missing_their_engagement_rates():
    """An absent ER on a brief with no profiles is not a gap; it is a category that does not
    apply. Reporting it would be the complaint nobody can answer that §5.3 wrote out."""
    found = facts.compute("An out-of-home campaign across three Lima metro stations.")
    assert found["engagement_rate"]["status"] == "not_applicable"


@pytest.mark.parametrize("text,detected", [
    ("Budget is fixed at 40,000 USD for the whole activation.", True),
    ("Total spend: $125,000.", True),
    ("Budget: TBC.", False),
    ("A creator-led launch with three colourways.", False),
])
def test_budget_detected_yes_or_no(text, detected):
    """"Budget detected yes/no" — deliberately not "is the budget adequate", which is a
    judgment. The mechanical part is whether a figure is there at all."""
    assert facts.compute(text)["budget"]["status"] == ("present" if detected else "absent")


def test_the_channel_checklist_says_which_are_missing():
    """"Missing 360 channels" — hit/miss against a fixed list, so two users judging the same
    brief cannot disagree about which channels it covers."""
    found = facts.compute(
        "Paid social across Meta and TikTok, organic social on the brand handles, "
        "and an email push to the CRM list.")

    channels = found["channels"]
    assert channels["status"] == "partial"
    assert set(channels["present"]) >= {"paid_social", "organic_social", "email"}
    assert "out_of_home" in channels["missing"]
    assert "retail" in channels["missing"]


def test_a_weekend_falling_on_a_tuesday_is_caught():
    """The review's own example, and the sharpest of them: a claim inside the brief that
    contradicts the calendar. Nothing about it needs a language model."""
    found = facts.compute("The weekend activation runs Saturday 3 March 2026 in Lima.")

    consistency = found["date_consistency"]
    assert consistency["status"] == "contradicted"
    assert "3 March 2026" in consistency["contradictions"][0]["evidence"]
    assert "tuesday" in consistency["contradictions"][0]["detail"].lower()


def test_the_word_weekend_alone_is_enough_to_check(conn=None):
    """The test above says "Saturday 3 March", so the WEEKDAY branch catches it and the
    weekend branch was never exercised — deleting it left the whole suite green. The review's
    example is "a 'weekend' falling on a Tuesday", and a brief often says only that."""
    found = facts.compute("The weekend of 3 March 2026 is the activation window in Lima.")

    consistency = found["date_consistency"]
    assert consistency["status"] == "contradicted"
    assert "weekend" in consistency["contradictions"][0]["detail"].lower()


def test_a_weekend_that_really_is_a_weekend_is_not_flagged():
    found = facts.compute("The weekend of 7 March 2026 is the activation window in Lima.")
    assert found["date_consistency"]["status"] == "consistent"


def test_a_range_that_ends_before_it_starts_is_caught():
    found = facts.compute("Flight runs 28 March 2026 to 3 March 2026.")
    assert found["date_consistency"]["status"] == "contradicted"


def test_a_consistent_brief_is_reported_as_consistent():
    found = facts.compute("The launch runs Tuesday 3 March 2026 to Saturday 28 March 2026.")
    assert found["date_consistency"]["status"] == "consistent"


def test_a_brief_with_no_checkable_dates_is_unknown_not_consistent():
    """"Consistent" is a claim that the dates agree. With no weekday and no year there is
    nothing to check, and saying "consistent" would be a green tick on an unread page —
    the same distinction §6.4 needed three codes for."""
    found = facts.compute("Flight runs 3 March to 28 March.")
    assert found["date_consistency"]["status"] == "nothing_to_check"


# ── evidence, or it is an assertion in a different coat ─────────────────────

def test_every_fact_that_found_something_shows_what_it_found():
    """A computed fact without evidence cannot be argued with, and the point of computing it
    is that a person can check it."""
    found = facts.compute(
        "Budget is 40,000 USD. Launch runs 3 March 2026. Paid social on Meta. "
        "Creators: @lima_style (3.4% ER).")

    for code, fact in found.items():
        if fact["status"] in ("present", "partial", "contradicted"):
            assert fact["evidence"], f"{code} claims something and shows nothing"


def test_every_fact_carries_a_stable_code_and_a_sentence():
    found = facts.compute("A brief.")
    for code, fact in found.items():
        assert fact["code"] == code
        assert fact["what_it_means"], code
        assert fact["basis"] == "computed"


# ── D11: the body layer, and why ────────────────────────────────────────────

def test_a_warning_not_to_mention_a_competitor_is_not_a_mention_of_it(conn):
    """D11, in the review's own words. A reviewer's comment saying "make sure we never
    mention adidas" would otherwise register as the brief mentioning adidas — the check
    reading somebody's warning as the thing they were warning about. §2.5 built the layer
    separation for this, and §7.1 is its first consumer."""
    import core
    import store

    cid = core.ingest_campaign(
        conn, title="Colombia launch",
        detail="A creator-led launch. Budget is 40,000 USD.")["campaign_id"]
    store.insert_chunks(conn, cid, ["Make sure we never mention a competitor budget of "
                                    "90,000 USD in this deck."],
                        kind="commentary", sources=[{"author": "R. Vega"}])

    found = facts.for_campaign(conn, cid)
    assert found["budget"]["status"] == "present"
    assert "40,000" in found["budget"]["evidence"][0]
    assert not any("90,000" in e for e in found["budget"]["evidence"]), \
        "that figure is a reviewer's warning, not the brief's budget"


def test_the_layer_it_read_is_stated(conn):
    import core
    cid = core.ingest_campaign(conn, title="X", detail="Budget is 40,000 USD.")["campaign_id"]
    assert facts.for_campaign(conn, cid)["budget"]["layer"] == "body"


# ── it reaches the reasoning, which is the point ────────────────────────────

def test_the_evidence_package_carries_the_computed_facts(conn):
    """"Run these as code in `prepare_evaluation`." The model then reasons only about what is
    genuinely judgment, and two thirds of the output stops being generated at all."""
    import core

    package = core.prepare_evaluation(
        conn, subject_title="Colombia v1",
        proposal_text="A creator-led launch. Creators: @lima_style (180k followers).")

    computed = package["computed"]
    assert computed["budget"]["status"] == "absent"
    assert computed["engagement_rate"]["status"] == "absent"
    assert "computed" in package["note"].lower()


def test_the_note_tells_the_model_not_to_re_derive_them(conn):
    """The variance the review is removing comes back the moment a model recomputes a fact
    the server already established — and models do, because the brief is right there."""
    import core

    note = core.prepare_evaluation(conn, subject_title="X",
                                   proposal_text="A brief.")["note"]
    assert "already" in note.lower() or "do not" in note.lower()


def test_a_stored_subject_is_read_from_the_record_not_the_prose(conn):
    """When the subject is a record, the facts come from the record — the same reasoning §6.4
    reached about which text a check should run against."""
    import core

    cid = core.ingest_campaign(
        conn, title="Colombia v1",
        detail="Budget is 40,000 USD. Paid social on Meta.")["campaign_id"]

    package = core.prepare_evaluation(conn, subject_title="Colombia v1",
                                      proposal_text="A short summary.", campaign_id=cid)
    assert package["computed"]["budget"]["status"] == "present"


# ── D49: a field missing from every record in a cell ────────────────────────

def test_a_market_where_nothing_has_ever_carried_a_budget_is_a_gap(conn):
    """D49, and one of the review's own three examples of a gap: "no LATAM store launch has
    ever carried a budget". It is about an absent FIELD inside records, not an absent record,
    which is why it needed 7.1 first — nothing could detect a budget."""
    import core

    for n in range(3):
        core.ingest_campaign(conn, title=f"LATAM launch {n}", market="LATAM",
                             status="concluded",
                             detail="A retail store launch with creator seeding.")
    core.ingest_campaign(conn, title="APAC launch", market="APAC", status="concluded",
                         detail="A retail store launch. Budget is 40,000 USD.")

    blanks = [g for g in core.gaps(conn)["gaps"] if g["code"] == "field_never_recorded"]
    budget = [g for g in blanks if g["field"] == "budget"]

    assert budget, [g["code"] for g in core.gaps(conn)["gaps"]]
    assert budget[0]["market"] == "LATAM"
    assert budget[0]["campaigns"] == 3
    assert "APAC" not in str(budget[0]), "APAC has one, so it is not the gap"


def test_a_market_where_one_record_carries_the_field_is_not_a_gap(conn):
    """"Never" is the claim. One record with a budget makes it false, and reporting the cell
    anyway would be the overstatement the whole `gaps` surface is written against."""
    import core

    core.ingest_campaign(conn, title="LATAM one", market="LATAM", status="concluded",
                         detail="A launch. Budget is 40,000 USD.")
    core.ingest_campaign(conn, title="LATAM two", market="LATAM", status="concluded",
                         detail="A launch with no figures at all.")

    blanks = [g for g in core.gaps(conn)["gaps"]
              if g["code"] == "field_never_recorded" and g["field"] == "budget"]
    assert not blanks


def test_a_single_record_market_is_not_a_pattern(conn):
    """One record missing a budget is a fact about that record, not about the market. Calling
    it a coverage gap would turn every new market into a complaint on the day it is added."""
    import core

    core.ingest_campaign(conn, title="Solo", market="LATAM", status="concluded",
                         detail="A launch with no figures at all.")

    assert not [g for g in core.gaps(conn)["gaps"] if g["code"] == "field_never_recorded"]


# ── D58: a fact that went stale between versions ────────────────────────────

def test_a_budget_that_disappeared_between_versions_is_reported(conn):
    """D58. `diff_campaigns` compared structured fields and citations; the brief's own
    content changing — a budget dropped, a channel gone, a date contradiction introduced —
    was invisible, because nothing could read those facts until 7.1."""
    import core

    v1 = core.ingest_campaign(conn, title="Colombia v1",
                              detail="A launch. Budget is 40,000 USD. Paid social on Meta."
                              )["campaign_id"]
    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1,
                              detail="A launch. Paid social on Meta.")["campaign_id"]

    changes = core.diff_campaigns(conn, earlier=v1, later=v2)["fact_changes"]
    budget = [c for c in changes if c["code"] == "budget"]

    assert budget, changes
    assert budget[0]["was"] == "present"
    assert budget[0]["now"] == "absent"


def test_a_contradiction_introduced_by_a_later_version_is_reported(conn):
    import core

    v1 = core.ingest_campaign(conn, title="Colombia v1",
                              detail="Runs Tuesday 3 March 2026.")["campaign_id"]
    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1,
                              detail="Runs Saturday 3 March 2026.")["campaign_id"]

    changes = core.diff_campaigns(conn, earlier=v1, later=v2)["fact_changes"]
    assert [c["now"] for c in changes if c["code"] == "date_consistency"] == ["contradicted"]


def test_facts_that_did_not_change_are_not_reported(conn):
    """A diff that lists everything is a diff nobody reads."""
    import core

    v1 = core.ingest_campaign(conn, title="Colombia v1",
                              detail="A launch. Budget is 40,000 USD.")["campaign_id"]
    v2 = core.ingest_campaign(conn, title="Colombia v2", supersedes=v1,
                              detail="A launch. Budget is 50,000 USD.")["campaign_id"]

    changes = core.diff_campaigns(conn, earlier=v1, later=v2)["fact_changes"]
    assert [c["code"] for c in changes] == [], "both carry a budget; the amount is not status"


def test_a_market_of_briefs_with_no_creators_is_not_missing_their_rates(conn):
    """`not_applicable` is not a miss. Counting it made the gap unclosable — a market of
    out-of-home campaigns would be told forever that none of its records carries an
    engagement rate, which is true and is not a gap."""
    import core

    for n in range(3):
        core.ingest_campaign(conn, title=f"OOH {n}", market="LATAM", status="concluded",
                             detail="An out-of-home campaign across metro stations. "
                                    "Budget is 40,000 USD. Runs 3 March 2026.")

    blanks = [g for g in core.gaps(conn)["gaps"]
              if g["code"] == "field_never_recorded" and g["field"] == "engagement_rate"]
    assert not blanks


def test_a_record_the_field_cannot_apply_to_does_not_hide_the_gap(conn):
    """The sharper half, and the one that pins the filter. A market where two briefs carry
    creators with no engagement rates and one is out-of-home: the OOH record cannot be
    missing an ER, so counting it as a record that HAS one would hide a real gap behind an
    irrelevance. Both this and the test above are needed — the all-not-applicable case leaves
    the filter free to be deleted, because the answer comes out the same either way."""
    import core

    for n in range(2):
        core.ingest_campaign(conn, title=f"Creator {n}", market="LATAM", status="concluded",
                             detail="Creators: @lima_style (180k followers).")
    core.ingest_campaign(conn, title="OOH", market="LATAM", status="concluded",
                         detail="An out-of-home campaign across metro stations.")

    blanks = [g for g in core.gaps(conn)["gaps"]
              if g["code"] == "field_never_recorded" and g["field"] == "engagement_rate"]
    assert blanks, "two rosters with no rates is a gap; the OOH brief is not evidence against"
    assert blanks[0]["campaigns"] == 2, "and it is counted over the records it applies to"
