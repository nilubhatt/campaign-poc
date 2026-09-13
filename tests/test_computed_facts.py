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


# ── review round: every one of these was a confident false fact ─────────────

@pytest.mark.parametrize("text", [
    "Contact maria@brand.com and press@agency.co.uk for assets.",
    "Send to mailto:jose@x.com before the launch.",
])
def test_an_email_address_is_not_a_creator(text):
    """Almost every brief carries a contact address, so the commonest real brief got an
    authoritative "none of your creators has an engagement rate" about people who do not
    exist — and it poisoned the D49 market gap on top."""
    assert facts.compute(text)["engagement_rate"]["status"] == "not_applicable"


@pytest.mark.parametrize("text", [
    "Express delivery on all orders.",
    "500k impressions across the flight.",
    "Impress the board with the numbers.",
    "See https://x.com/pr/launch for the deck.",
])
def test_the_word_impressions_does_not_mean_public_relations(text):
    """`press` was an unanchored substring, so every brief with a reach KPI reported PR as
    covered — the checklist saying a channel is handled because another word contains it."""
    assert "pr" not in facts.compute(text)["channels"]["present"]


def test_press_release_still_counts():
    assert "pr" in facts.compute("A press release goes out on launch day.")["channels"]["present"]


def test_a_correct_brief_is_not_told_its_dates_contradict_the_calendar():
    """The sharpest of them. A fixed lookback read across neighbouring dates, so a brief that
    is right — 7 March 2026 really is a Saturday — was told it calls 12 March a Saturday. It
    produced the very finding this item was sold on, out of nothing, with the server's
    authority behind it."""
    found = facts.compute("Saturday 7 March 2026 to 12 March 2026 in Lima.")
    assert found["date_consistency"]["status"] == "consistent"


@pytest.mark.parametrize("text", [
    "Review on Monday. 7 March 2026 is launch.",
    "The weekend of 7 March 2026, then 10 March 2026 workshop.",
])
def test_a_weekday_in_a_different_clause_is_not_a_claim_about_this_date(text):
    assert facts.compute(text)["date_consistency"]["status"] == "consistent"


@pytest.mark.parametrize("text,dates", [
    ("Budget line 3 may be deferred.", 0),
    ("We may 3 reduce spend.", 0),
    ("Order 24 Dec-branded hoodies.", 0),
    ("Reference 15 Jan-Feb split.", 0),
    ("The 3rd of May is launch day.", 1),
    ("Launch 3 May 2026.", 1),
])
def test_a_month_name_doing_ordinary_work_is_not_a_date(text, dates):
    """`may` is a modal verb before it is a month, and a hyphenated abbreviation names a
    product line. Both read as dates, and a date the brief does not contain is a fact about
    the parser presented as a fact about the brief."""
    assert facts.compute(text)["date_coverage"]["dates_found"] == dates


@pytest.mark.parametrize("text", ["Go live March 2026, wrap May 2026.", "Runs across June 2026."])
def test_a_month_and_a_year_is_a_date(text):
    """"No date appears anywhere in this brief" was returned for "Go live March 2026" — the
    parser-failure-as-absence the module docstring forbids, committed by the module."""
    assert facts.compute(text)["date_coverage"]["status"] == "present"


@pytest.mark.parametrize("text,detected", [
    ("RRP $49.99 per unit; retails at €30.", False),
    ("Ticket price $12 on the door.", False),
    ("Budget is fixed at 40,000 USD.", True),
    ("Total spend: $125,000.", True),
    ("12 pen and paper, 20 cop cars.", False),
])
def test_a_price_is_not_a_budget(text, detected):
    """A retail price is money and is not what the review means by "budget detected" — and
    lower-cased `pen` and `cop` are ordinary words, so "12 pen and paper" was a budget."""
    assert facts.compute(text)["budget"]["status"] == ("present" if detected else "absent")


def test_a_brief_with_prices_and_no_budget_says_which(conn=None):
    """"No money at all" and "money, but none of it a budget" are different states, and the
    absent message has to say which — the same distinction every other check here makes."""
    budget = facts.compute("RRP $49.99 per unit.")["budget"]
    assert budget["money_figures"] == 1
    assert "none of them near a word like budget" in budget["what_it_means"]


def test_a_channel_the_brief_rules_out_is_not_a_channel_it_covers():
    """"No paid social" reported paid social as present, which inverts the finding."""
    found = facts.compute(
        "No paid social. We will not use influencers. Email is out of scope.")
    assert found["channels"]["present"] == []
    assert set(found["channels"]["ruled_out"]) == {"paid_social", "influencer", "email"}


@pytest.mark.parametrize("text,channel", [
    ("Organic cotton tees in three colourways.", "organic_social"),
    ("Community guidelines apply to all posts.", "organic_social"),
    ("Metro Manila is the lead market.", "out_of_home"),
    ("An on-site activation at the flagship.", "web"),
    ("Meta description for the landing copy.", "paid_social"),
    ("Ecomsoft is the vendor.", "web"),
])
def test_a_word_that_merely_contains_a_channel_name_is_not_that_channel(text, channel):
    assert channel not in facts.compute(text)["channels"]["present"]


def test_each_named_channel_shows_the_words_it_was_named_by():
    """A flat evidence list capped at three left five of eight channels with nothing to check
    them against — and unlabelled, so a reader could not tell which snippet was which."""
    found = facts.compute(
        "Paid social on Meta ads, a press release, an email push, and billboards in Lima.")
    by_channel = found["channels"]["evidence_by_channel"]
    assert set(by_channel) == set(found["channels"]["present"])
    assert all(by_channel.values())


def test_the_checklist_says_named_rather_than_covered():
    """The check reads words. Whether a named channel is actually planned is a judgment, and
    saying "covered" would be the server claiming what it did not establish."""
    channels = facts.compute("Paid social on Meta ads.")["channels"]
    assert "NAMED" in channels["what_it_means"]
    assert "covered" not in channels["what_it_means"].lower()


@pytest.mark.parametrize("text", ["Tue 7 March 2026 is launch.", "Weds 7 March 2026 is launch."])
def test_an_abbreviated_weekday_is_still_a_claim(text):
    """Slides abbreviate, and a check that only knows "Tuesday" silently passes the rest —
    7 March 2026 was a Saturday."""
    assert facts.compute(text)["date_consistency"]["status"] == "contradicted"


def test_a_rate_belongs_to_the_profile_it_sits_beside():
    """The 80-character window ran past the next handle, so one creator was credited with
    another's engagement rate and the roster read as fully rated."""
    found = facts.compute("@lucia.rios and @pedro.g are on the roster; "
                          "@ana.p 3.1% ER is the only one measured.")
    assert found["engagement_rate"]["profiles_with_rate"] == 1
    assert found["engagement_rate"]["status"] == "partial"


def test_a_missing_record_is_not_a_diff_where_everything_changed(conn):
    """`for_campaign` returns `{}` for an unknown id, and `_fact_changes` then reported every
    fact as having changed from nothing."""
    import core

    v1 = core.ingest_campaign(conn, title="Colombia v1",
                              detail="Budget is 40,000 USD.")["campaign_id"]
    assert core._fact_changes(conn, v1, "camp_nope") == []
    assert core._fact_changes(conn, "camp_nope", v1) == []


def test_the_model_is_told_what_to_do_when_a_computed_fact_is_wrong(conn):
    """"Do not contradict them" removes the last check on a wrong fact — the model looking at
    the evidence and noticing the creator is an email address. Every check here is a regex,
    and the module's own premise is that a wrong computed fact is worse than a guess because
    it carries the server's authority."""
    import core

    note = core.prepare_evaluation(conn, subject_title="X",
                                   proposal_text="A brief.")["note"]
    assert "evidence" in note.lower()
    assert "computed_fact_disputed" in note


def test_a_weekday_belonging_to_the_previous_date_does_not_reach_this_one():
    """The sentence-boundary cut hides this on its own: "Saturday 7 March to 12 March" is cut
    at " to ". Separated by a comma there is no boundary word, and only stopping the window at
    the PREVIOUS DATE keeps the claim where it belongs."""
    found = facts.compute("Key dates: Saturday 7 March 2026, 12 March 2026, 19 March 2026.")
    assert found["date_consistency"]["status"] == "consistent"


def test_a_weekday_further_back_in_the_same_clause_is_not_a_claim_about_this_date():
    """A weekday has to be the last thing before the date apart from filler like "the" or
    "of". Without that rule any weekday within forty characters became a claim, and no
    boundary word separates these two."""
    found = facts.compute("Monday briefing covers 7 March 2026 in detail.")
    assert found["date_consistency"]["status"] == "consistent"

    # And the rule still lets a real claim through.
    assert facts.compute("Launch is Monday the 7 March 2026."
                         )["date_consistency"]["status"] == "contradicted"


def test_a_lowercase_currency_code_beside_a_budget_word_is_still_not_money():
    """`pen` and `cop` are ordinary words. The budget-proximity rule hides this for prices,
    but a brief that says "Budget: 20 cop cars" puts an ordinary word next to a budget word,
    and only requiring uppercase codes keeps it out."""
    assert facts.compute("Budget: 20 cop cars and 12 pen sets."
                         )["budget"]["status"] == "absent"
    assert facts.compute("Budget: 20000 COP for the market."
                         )["budget"]["status"] == "present"


def test_the_escape_hatch_says_what_to_do_and_not_only_what_to_call_it(conn):
    """Naming the code is not the instruction. The model has to be told to quote the evidence
    it is disputing, or "computed_fact_disputed" is a label with no argument attached."""
    import core

    note = core.prepare_evaluation(conn, subject_title="X",
                                   proposal_text="A brief.")["note"]
    assert "quoting the evidence" in note
    assert "do not defer to it against the evidence" in note
