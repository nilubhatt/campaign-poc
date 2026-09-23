"""
§9.6: a context events table, scoped by market and date.

The review: *"Conflict, natural disaster, regulatory change, supply-chain or port disruption,
platform outage, competitor launch, macro shock, and fixed calendar events. Fields: date range,
scope (market, region or global), kind, description, source link, and a stated impact — delay
in days, budget change, channels disrupted. Any campaign whose window overlaps an event in its
market links automatically. **No one should have to remember to connect them.**"*

**"No one should have to remember to connect them" is the whole item.** A join table somebody
maintains by hand is the thing this replaces, not the thing it is. The link is computed from
(scope, date range) against (market, window) every time it is read, because both halves keep
arriving: an earthquake is recorded weeks after the campaigns it overlapped, and a campaign is
uploaded months after the event was seeded. A materialised link would be stale in both
directions and stale here means silently wrong.

**A campaign had no window, and that is the load-bearing gap.** Dates existed only as prose
that `facts.py` parses for contradictions. So `starts_on`/`ends_on` are the field this item
turns on, exactly as `phase` was §9.1's — and the honest default is NOT to invent one. Where a
brief's own dates are the only evidence, the window is `heuristic` and says which dates it was
read from; where somebody typed it, it is `stated`. A campaign with no window at all reports
`nothing_to_check`, never "no events overlapped" — the second is a claim, and it is the claim
this library most wants to be able to make.

**The link is an overlap and nothing more.** §9.8 owns `confounded` and the caveat that travels
with a metric. What 9.6 must not do is get there early: "ran during" is a fact, "affected by"
is an assertion, and a model handed the second will explain a disappointing number with it.
"""
import pytest

import context
import core
import store


@pytest.fixture(autouse=True)
def _only_what_this_file_records(conn):
    """Clear the shipped calendar (§9.7) before every test in this file.

    Not a convenience. §9.6 is the LINKING MECHANISM and these tests count what they recorded;
    §9.7 is the pack of dates the product ships with, and its own file checks that those reach
    a proposal by exactly this arithmetic. Leaving both in one file would mean every count here
    silently depended on which holidays happen to fall in the window a test picked — and a test
    whose meaning changes when somebody edits a calendar entry is not testing what it says.
    """
    conn.execute("DELETE FROM context_events WHERE seeded = 1")
    conn.commit()


def _campaign(conn, title="Mexico launch", market="LATAM", **kw):
    return core.ingest_campaign(conn, title=title, market=market, status="concluded",
                                detail=kw.pop("detail", "A launch."), **kw)["campaign_id"]


def _quake(conn, **kw):
    kw.setdefault("starts_on", "2026-03-10")
    kw.setdefault("ends_on", "2026-03-12")
    kw.setdefault("scope", "market")
    kw.setdefault("scope_value", "LATAM")
    kw.setdefault("kind", "natural_disaster")
    kw.setdefault("description", "Magnitude 7.1 earthquake; Mexico City retail shut for 3 days.")
    kw.setdefault("recorded_by", "R. Vega")
    return context.record(conn, **kw)


# ── the table ───────────────────────────────────────────────────────────────

def test_an_event_is_recorded_with_its_range_scope_and_kind(conn):
    saved = _quake(conn)

    assert saved["starts_on"] == "2026-03-10"
    assert saved["ends_on"] == "2026-03-12"
    assert saved["scope"] == "market"
    assert saved["kind"] == "natural_disaster"
    assert saved["basis"] == "stated", "an event is something a person put on the record"


def test_every_kind_the_review_names_is_expressible(conn):
    """"Conflict, natural disaster, regulatory change, supply-chain or port disruption,
    platform outage, competitor launch, macro shock, and fixed calendar events." A vocabulary
    missing one of them sends that event to whichever word is nearest, and the nearest word to
    "port strike" is whatever the person typing is thinking of that day."""
    for kind in ("conflict", "natural_disaster", "regulatory_change", "supply_chain",
                 "platform_outage", "competitor_launch", "macro_shock", "fixed_calendar"):
        assert kind in context.KINDS


def test_a_kind_outside_the_vocabulary_is_refused_with_it(conn):
    with pytest.raises(ValueError) as e:
        _quake(conn, kind="bad_weather")
    assert "natural_disaster" in str(e.value)


def test_an_event_carries_its_source_and_stated_impact(conn):
    """"Source link, and a stated impact — delay in days, budget change, channels
    disrupted." Without the source it is a rumour on the record; without the impact it is a
    date nobody can act on."""
    saved = _quake(conn, source="https://example.gov/quake-2026-03",
                   delay_days=3, budget_change_pct=-12.0,
                   channels_disrupted=["retail", "ooh"])

    assert saved["source"].startswith("https://")
    assert saved["delay_days"] == 3
    assert saved["budget_change_pct"] == -12.0
    assert saved["channels_disrupted"] == ["retail", "ooh"]


def test_the_impact_is_stated_never_computed(conn):
    """The server did not measure a three-day delay; somebody said so. §2.4 made `computed`
    unwritable from the MCP surface exactly so the word keeps meaning what it says."""
    saved = _quake(conn, delay_days=3)

    assert saved["basis"] == "stated"
    assert "computed" not in saved["basis"]


def test_an_event_needs_a_person(conn):
    with pytest.raises(ValueError) as e:
        _quake(conn, recorded_by="")
    assert "person" in str(e.value).lower()


def test_an_event_that_ends_before_it_starts_is_refused(conn):
    with pytest.raises(ValueError) as e:
        _quake(conn, starts_on="2026-03-12", ends_on="2026-03-10")
    assert "before" in str(e.value).lower()


def test_an_ongoing_event_has_no_end(conn):
    """A port closure with no announced reopening is the ordinary case. Forcing an end date
    invents one, and an invented end date silently stops matching campaigns it still covers."""
    saved = _quake(conn, kind="supply_chain", ends_on=None,
                   description="Port of Manzanillo closed indefinitely.")

    assert saved["ends_on"] is None
    assert "ongoing" in saved["what_it_means"].lower()


# ── the window a campaign has to have ───────────────────────────────────────

def test_a_campaign_can_be_given_a_window(conn):
    cid = _campaign(conn)

    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")

    window = context.window_of(conn, cid)
    assert window["starts_on"] == "2026-03-01"
    assert window["ends_on"] == "2026-03-31"
    assert window["basis"] == "stated"


def test_a_window_read_out_of_the_brief_says_it_was_read_out_of_the_brief(conn):
    """A window nobody typed is an inference from prose, and the one thing it must not do is
    arrive looking like a fact somebody entered."""
    cid = _campaign(conn, detail="Flight runs 2 March 2026 to 28 March 2026 across LATAM.")

    window = context.window_of(conn, cid)

    assert window["basis"] == "heuristic"
    assert window["starts_on"] == "2026-03-02"
    assert window["ends_on"] == "2026-03-28"
    assert "read from" in window["what_it_means"].lower()


def test_a_campaign_with_no_dates_has_no_window_rather_than_a_guessed_one(conn):
    cid = _campaign(conn, detail="A launch with no dates in it at all.")

    window = context.window_of(conn, cid)

    assert window["starts_on"] is None
    assert window["basis"] is None


def test_a_stated_window_beats_the_prose(conn):
    """Somebody typing the dates is the correction; continuing to read them out of the deck
    would make the correction unrecordable."""
    cid = _campaign(conn, detail="Flight runs 2 March 2026 to 28 March 2026.")

    core.update_campaign(conn, campaign_id=cid, starts_on="2026-04-01", ends_on="2026-04-30")

    assert context.window_of(conn, cid)["starts_on"] == "2026-04-01"


# ── the link nobody has to remember ─────────────────────────────────────────

def test_a_campaign_whose_window_overlaps_an_event_in_its_market_links(conn):
    """"Any campaign whose window overlaps an event in its market links automatically. No one
    should have to remember to connect them.\""""
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")
    _quake(conn)

    linked = context.for_campaign(conn, cid)

    assert len(linked["events"]) == 1
    assert linked["events"][0]["kind"] == "natural_disaster"
    assert linked["basis"] == "computed", "the overlap is arithmetic on two date ranges"


def test_the_event_can_arrive_after_the_campaign_and_still_link(conn):
    """Both halves keep arriving: an earthquake is recorded weeks after the campaigns it
    overlapped. A link written at upload time would never see it."""
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")
    assert context.for_campaign(conn, cid)["events"] == []

    _quake(conn)

    assert len(context.for_campaign(conn, cid)["events"]) == 1


def test_the_campaign_can_arrive_after_the_event_and_still_link(conn):
    _quake(conn)
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")

    assert len(context.for_campaign(conn, cid)["events"]) == 1


def test_an_event_in_another_market_does_not_link(conn):
    cid = _campaign(conn, market="EMEA")
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")
    _quake(conn, scope_value="LATAM")

    assert context.for_campaign(conn, cid)["events"] == []


def test_a_global_event_links_everywhere(conn):
    cid = _campaign(conn, market="APAC")
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")
    _quake(conn, scope="global", scope_value=None, kind="macro_shock",
           description="Global ad-market repricing after the March rate decision.")

    assert len(context.for_campaign(conn, cid)["events"]) == 1


def test_a_market_the_activation_touched_counts_as_its_market(conn):
    """`markets` is the membership field — `market` is a single grouping label and cannot say
    "ran in MY and ID". An event in Indonesia has to reach a campaign that ran there."""
    cid = _campaign(conn, market="SEA", markets=["Malaysia", "Indonesia"])
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")
    _quake(conn, scope="market", scope_value="Indonesia")

    assert len(context.for_campaign(conn, cid)["events"]) == 1


def test_an_event_that_ends_the_day_the_campaign_starts_overlaps(conn):
    """Inclusive at both ends. An off-by-one here drops exactly the events that ran into a
    launch, which are the ones anybody cares about."""
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-12", ends_on="2026-03-31")
    _quake(conn, starts_on="2026-03-01", ends_on="2026-03-12")

    assert len(context.for_campaign(conn, cid)["events"]) == 1


def test_an_event_that_ends_the_day_before_does_not(conn):
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-12", ends_on="2026-03-31")
    _quake(conn, starts_on="2026-03-01", ends_on="2026-03-11")

    assert context.for_campaign(conn, cid)["events"] == []


def test_an_ongoing_event_covers_everything_after_it_starts(conn):
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-06-01", ends_on="2026-06-30")
    _quake(conn, kind="supply_chain", starts_on="2026-03-01", ends_on=None,
           description="Port of Manzanillo closed indefinitely.")

    assert len(context.for_campaign(conn, cid)["events"]) == 1


# ── what it refuses to say ──────────────────────────────────────────────────

def test_a_campaign_with_no_window_says_so_rather_than_no_events(conn):
    """"No events overlapped this campaign" is a claim, and it is the claim this table most
    wants to be able to make. A campaign with no window has not earned it — and a `[]` that
    means "could not check" reads identically to one that means "checked, nothing there"."""
    cid = _campaign(conn, detail="A launch with no dates in it at all.")
    _quake(conn)

    linked = context.for_campaign(conn, cid)

    assert linked["status"] == "nothing_to_check"
    assert "no window" in linked["what_it_means"].lower()
    assert linked["next_actions"][0]["tool"] == "update_campaign"


def test_a_checked_campaign_with_nothing_overlapping_says_that_instead(conn):
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-08-01", ends_on="2026-08-31")
    _quake(conn)

    linked = context.for_campaign(conn, cid)

    assert linked["status"] == "checked"
    assert linked["events"] == []


def test_an_overlap_found_on_a_heuristic_window_says_which_window_it_used(conn):
    """A window read out of prose can be wrong, and the overlap it produces inherits that. It
    still gets reported — it is the most useful thing here — but never as though somebody had
    entered the dates."""
    cid = _campaign(conn, detail="Flight runs 2 March 2026 to 28 March 2026 across LATAM.")
    _quake(conn)

    linked = context.for_campaign(conn, cid)

    assert linked["events"]
    assert linked["window"]["basis"] == "heuristic"
    assert "read from the brief" in linked["what_it_means"].lower()


def test_it_records_the_overlap_and_never_the_cause(conn):
    """"Sell-through was down and there was an earthquake" is not evidence the earthquake
    caused it. A model asked to explain a disappointing number will reach for whatever is
    nearby, and this is the nearest thing there will ever be."""
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")
    _quake(conn)

    said = (context.for_campaign(conn, cid)["what_it_means"] + " "
            + context.for_campaign(conn, cid)["events"][0]["what_it_means"]).lower()

    for word in ("caused", "because of", "due to", "affected by", "explains",
                 "responsible for", "blame"):
        assert word not in said, word
    assert "ran during" in said or "overlaps" in said


# ── D116: a tool nothing offers is a tool nobody calls ──────────────────────

def test_recording_an_event_offers_the_campaigns_it_now_reaches(conn):
    """The link is automatic, which makes it invisible: nothing on screen changes when an
    event lands, so the person who recorded it has no reason to believe it reached anything.
    The offer is how "it connected to these" gets said."""
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")

    saved = _quake(conn)

    assert saved["now_overlaps"] == 1
    offer = saved["next_actions"][0]
    assert offer["tool"] == "campaign_context"
    assert offer["prefilled_args"]["campaign_id"] == cid


def test_an_event_reaching_nothing_says_so_rather_than_offering_a_campaign(conn):
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-08-01", ends_on="2026-08-31")

    saved = _quake(conn)

    assert saved["now_overlaps"] == 0
    assert "no campaign" in saved["what_it_means"].lower()


def test_setting_a_window_offers_the_check_it_just_made_possible(conn):
    cid = _campaign(conn)
    _quake(conn)

    out = core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01",
                               ends_on="2026-03-31")

    assert any(a["tool"] == "campaign_context" for a in out.get("next_actions") or [])


def test_a_campaign_with_nothing_recorded_around_it_offers_the_other_direction(conn):
    """"Checked, and nothing was going on" is only as true as the calendar behind it. On a
    library where nobody has recorded anything, that sentence is about the library."""
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")

    linked = context.for_campaign(conn, cid)

    assert linked["events"] == []
    assert linked["next_actions"][0]["tool"] == "record_context_event"


def test_a_campaign_that_already_has_context_is_not_nagged(conn):
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")
    _quake(conn)

    linked = context.for_campaign(conn, cid)

    assert linked["events"]
    assert not linked.get("next_actions")


def test_both_tools_reach_the_model_over_the_protocol(conn):
    import asyncio
    import json

    import mcp_server

    async def call(name, args):
        return json.loads((await mcp_server.mcp.call_tool(name, args)).content[0].text)

    cid = _campaign(conn)
    asyncio.run(call("update_campaign", {"campaign_id": cid, "starts_on": "2026-03-01",
                                         "ends_on": "2026-03-31"}))
    saved = asyncio.run(call("record_context_event", {
        "starts_on": "2026-03-10", "ends_on": "2026-03-12", "scope": "market",
        "scope_value": "LATAM", "kind": "natural_disaster",
        "description": "Magnitude 7.1 earthquake; Mexico City retail shut for 3 days.",
        "recorded_by": "R. Vega"}))
    assert saved["basis"] == "stated"

    linked = asyncio.run(call("campaign_context", {"campaign_id": cid}))
    assert linked["status"] == "checked"
    assert len(linked["events"]) == 1


def test_a_window_that_closes_before_it_opens_is_refused(conn):
    cid = _campaign(conn)

    with pytest.raises(ValueError) as e:
        core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-31",
                             ends_on="2026-03-01")
    assert "before" in str(e.value).lower()


def test_a_window_that_is_not_a_date_is_refused(conn):
    cid = _campaign(conn)

    with pytest.raises(ValueError) as e:
        core.update_campaign(conn, campaign_id=cid, starts_on="March-ish")
    assert "YYYY-MM-DD" in str(e.value)


# ── the boundaries and refusals the mutation pass found unguarded ───────────

def test_an_event_that_starts_the_day_the_campaign_ends_overlaps(conn):
    """The other end of the same inclusive boundary. Only the lower one was tested, so
    `starts_on <= window_end` could be weakened to `<` and every test still passed — dropping
    exactly the events that began on a launch's final day."""
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-10")
    _quake(conn, starts_on="2026-03-10", ends_on="2026-03-14")

    assert len(context.for_campaign(conn, cid)["events"]) == 1


def test_an_event_that_starts_the_day_after_does_not(conn):
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-10")
    _quake(conn, starts_on="2026-03-11", ends_on="2026-03-14")

    assert context.for_campaign(conn, cid)["events"] == []


def test_a_date_with_no_year_is_not_a_window(conn):
    """`facts._parsed_dates` stands a yearless date up in the year 2000 to keep its spans
    comparable — it exists to find CONTRADICTIONS, not to bound a flight. Inherited as a
    window, "launch on 3 March" produces 2000-03-03 to 2000-03-03: a window that overlaps
    nothing, looks like a real answer while doing it, and reports `heuristic` rather than the
    truth, which is that this brief does not say what year it ran."""
    cid = _campaign(conn, detail="Launch on 3 March, with a follow-up on 9 March.")

    window = context.window_of(conn, cid)

    assert window["basis"] is None
    assert window["starts_on"] is None


def test_an_event_in_another_market_reaches_nothing_even_when_the_dates_line_up(conn):
    """The two directions are separate implementations of "does this event reach this
    campaign" — `_scopes_of` on the campaign side, `_in_scope` on the event side. Only the
    campaign side was tested for scope, so the event side could ignore scope entirely and
    report that an earthquake in LATAM had reached a campaign in EMEA."""
    cid = _campaign(conn, market="EMEA")
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")

    saved = _quake(conn, scope="market", scope_value="LATAM")

    assert saved["now_overlaps"] == 0


def test_the_two_directions_agree(conn):
    """One arithmetic, run both ways. They were written as separate functions, so nothing
    stopped them drifting apart — and a disagreement would mean an event that says it reached
    a campaign the campaign does not list, which is worse than either answer alone."""
    reaching = _campaign(conn, "In market and in window", market="LATAM")
    core.update_campaign(conn, campaign_id=reaching, starts_on="2026-03-01",
                         ends_on="2026-03-31")
    wrong_market = _campaign(conn, "Out of market", market="EMEA")
    core.update_campaign(conn, campaign_id=wrong_market, starts_on="2026-03-01",
                         ends_on="2026-03-31")
    wrong_window = _campaign(conn, "Out of window", market="LATAM")
    core.update_campaign(conn, campaign_id=wrong_window, starts_on="2026-08-01",
                         ends_on="2026-08-31")
    by_membership = _campaign(conn, "Reached by markets", market="SEA",
                              markets=["Mexico", "Colombia"])
    core.update_campaign(conn, campaign_id=by_membership, starts_on="2026-03-01",
                         ends_on="2026-03-31")
    _quake(conn, scope="market", scope_value="Mexico")
    event = context.record(conn, starts_on="2026-03-10", ends_on="2026-03-12", scope="market",
                           scope_value="LATAM", kind="natural_disaster",
                           description="Magnitude 7.1 earthquake.", recorded_by="R. Vega")

    from_the_event = {c["campaign_id"] for c in context.campaigns_overlapping(conn, event)}
    from_the_campaigns = {cid for cid in (reaching, wrong_market, wrong_window, by_membership)
                          if any(e["id"] == event["id"]
                                 for e in context.for_campaign(conn, cid)["events"])}

    assert from_the_event == from_the_campaigns
    assert from_the_event == {reaching}


def test_a_market_event_with_no_market_named_is_refused(conn):
    """It would link to nothing and sit on the record looking recorded. `global` is the value
    for something that applies everywhere."""
    with pytest.raises(ValueError) as e:
        _quake(conn, scope="market", scope_value=None)
    assert "global" in str(e.value)


def test_a_global_event_does_not_quietly_keep_a_market(conn):
    """"Global" plus a market is a contradiction somebody probably did not mean, and keeping
    the market silently would make it match one market while reading as global everywhere."""
    saved = _quake(conn, scope="global", scope_value="LATAM", kind="macro_shock")

    assert saved["scope_value"] is None


def test_a_long_list_of_events_is_capped_but_the_count_is_not(conn):
    """A campaign running a quarter in a seeded market genuinely overlaps a dozen holidays,
    and a list that long stops being read. The COUNT is the finding — "twelve things were
    going on" — so it stays complete.

    A literal, not `MAX_EVENTS_SHOWN + n`: sized from the constant, the fixture grows when the
    cap does and the assertion can never fail."""
    assert context.MAX_EVENTS_SHOWN < 11, "this fixture must exceed the cap; resize it"
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")
    for day in range(1, 12):
        _quake(conn, kind="fixed_calendar", starts_on=f"2026-03-{day:02d}",
               ends_on=f"2026-03-{day:02d}", description=f"Public holiday {day}.")

    linked = context.for_campaign(conn, cid)

    assert len(linked["events"]) == context.MAX_EVENTS_SHOWN
    assert linked["events_total"] == 11
    assert linked["events_truncated"] is True
    assert "11 recorded event" in linked["what_it_means"]


# ── D119: the date column a workbook carries and nothing read ───────────────

def test_a_workbook_date_column_sets_the_window_instead_of_being_dropped(conn):
    """§8.8 skips a `Month` or `Launch Date` column as "not a measurement", which was right
    against the alternative it replaced — making them provisional KPI measures that could
    graduate. But a date column is real information, and §9.6 is the field it belongs in: the
    period is exactly what a campaign had no way to record."""
    cid = _campaign(conn)

    core.add_metrics(conn, campaign_id=cid, confirm=True,
                     structured={"launch_date": "2026-03-01", "roas": 3.2})

    window = context.window_of(conn, cid)
    assert window["starts_on"] == "2026-03-01"
    assert window["basis"] == "stated"


def test_a_date_column_is_reported_rather_than_silently_used(conn):
    """A skip nobody is told about is the silent acceptance this phase is written against —
    and so is a WRITE nobody is told about. Setting a campaign's window from a spreadsheet
    cell changes which context events reach it."""
    cid = _campaign(conn)

    out = core.add_metrics(conn, campaign_id=cid, confirm=True,
                           structured={"launch_date": "2026-03-01", "roas": 3.2})

    assert out["window_set"]["starts_on"] == "2026-03-01"
    assert "launch_date" in out["window_set"]["from"]
    assert not any(s["key"] == "launch_date" for s in out.get("skipped") or []), (
        "it was used, so reporting it as skipped would be a false statement about the import")


def test_a_window_already_entered_is_not_overwritten_by_a_spreadsheet(conn):
    """Somebody typing the dates is the more reliable claim. A workbook cell silently
    replacing it is the correction being undone by the thing it corrected."""
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-04-01", ends_on="2026-04-30")

    core.add_metrics(conn, campaign_id=cid, confirm=True,
                     structured={"launch_date": "2026-03-01", "roas": 3.2})

    assert context.window_of(conn, cid)["starts_on"] == "2026-04-01"


def test_a_start_and_end_column_set_both_ends(conn):
    cid = _campaign(conn)

    core.add_metrics(conn, campaign_id=cid, confirm=True,
                     structured={"start_date": "2026-03-01", "end_date": "2026-03-31",
                                 "roas": 3.2})

    window = context.window_of(conn, cid)
    assert (window["starts_on"], window["ends_on"]) == ("2026-03-01", "2026-03-31")


def test_a_month_column_becomes_the_whole_month(conn):
    """"March 2026" names a period, not a day. Reading it as the 1st would make a campaign
    that ran all month look like a one-day activation, and the overlap arithmetic would then
    miss every event in the other thirty days."""
    cid = _campaign(conn)

    core.add_metrics(conn, campaign_id=cid, confirm=True,
                     structured={"month": "2026-03", "roas": 3.2})

    window = context.window_of(conn, cid)
    assert (window["starts_on"], window["ends_on"]) == ("2026-03-01", "2026-03-31")


def test_a_column_that_is_not_a_date_is_still_skipped_and_said(conn):
    cid = _campaign(conn)

    out = core.add_metrics(conn, campaign_id=cid, confirm=True,
                           structured={"store_number": 14, "roas": 3.2})

    assert any(s["key"] == "store_number" for s in out["skipped"])
    assert "window_set" not in out


# ── the three routes to a false clean bill ──────────────────────────────────

def test_a_heuristic_window_never_produces_a_clean_bill(conn):
    """The `nothing_to_check`/`checked` split exists so an empty list never reads as a claim,
    and the heuristic window is the mechanism that defeats it. A brief whose only year-bearing
    date is an asset-spec version stamp gets a window of that one day, finds nothing, and the
    product says "Nothing on record was going on. That is a real answer." Without the
    heuristic that record would have been honest.

    A window read out of prose may still produce a positive MATCH — that is the useful half —
    but it can never support the negative claim."""
    cid = _campaign(conn, detail="Khloe MX launch. Launch window TBC. Asset spec v2 dated "
                                 "2019-04-01.")

    linked = context.for_campaign(conn, cid)

    assert linked["window"]["basis"] == "heuristic"
    assert linked["events"] == []
    assert linked["status"] == "nothing_to_check"
    assert "real answer" not in linked["what_it_means"]


def test_a_heuristic_window_that_does_match_still_reports_the_match(conn):
    cid = _campaign(conn, detail="Flight runs 2 March 2026 to 28 March 2026 across LATAM.")
    _quake(conn)

    linked = context.for_campaign(conn, cid)

    assert linked["status"] == "checked"
    assert len(linked["events"]) == 1


def test_a_campaign_with_no_market_is_not_given_a_clean_bill(conn):
    """Nothing market-scoped COULD reach it — only global events are searched. Reporting
    "nothing was going on in this market" about a record that names no market is a claim about
    a search that never ran."""
    cid = _campaign(conn, market="")
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")
    _quake(conn)

    linked = context.for_campaign(conn, cid)

    assert linked["status"] == "nothing_to_check"
    assert "no market" in linked["what_it_means"].lower()


def test_the_answer_says_which_scopes_it_searched(conn):
    """The empty sentence named one market while the search covered market, region, every
    market the activation touched, and global. A reader cannot tell a narrow search from a
    complete one unless it says."""
    cid = _campaign(conn, market="SEA", markets=["Malaysia", "Indonesia"])
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")

    linked = context.for_campaign(conn, cid)

    assert set(linked["searched"]) == {"SEA", "Malaysia", "Indonesia", "global"}
    for name in ("Malaysia", "Indonesia", "global"):
        assert name in linked["what_it_means"]


def test_a_window_inverted_across_two_calls_is_refused(conn):
    """Validated only when both ends arrived in the same call, a one-field typo fix stored
    2026-12-01 to 2026-03-31 — and the product printed that window backwards inside the
    sentence asserting nothing was going on."""
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")

    with pytest.raises(ValueError) as e:
        core.update_campaign(conn, campaign_id=cid, starts_on="2026-12-01")

    assert "before" in str(e.value).lower()
    assert context.window_of(conn, cid)["starts_on"] == "2026-03-01", "and nothing was written"


def test_moving_one_end_past_the_other_is_refused_from_either_side(conn):
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")

    with pytest.raises(ValueError):
        core.update_campaign(conn, campaign_id=cid, ends_on="2026-01-01")


def test_clearing_one_end_is_still_allowed(conn):
    """An open-ended campaign is ordinary — always-on retail has no end date. Refusing a
    clear because the stored other end is later would make it unsettable."""
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")

    core.update_campaign(conn, campaign_id=cid, ends_on="")

    assert context.window_of(conn, cid)["ends_on"] is None


# ── F3: "no one should have to remember" must not become "remember a window" ─

def test_a_window_can_be_given_at_upload(conn):
    """The only route in was a second, deliberate `update_campaign` call. Trading a
    hand-maintained join for one date pair is a good trade only if the date pair is asked for
    where the campaign is created."""
    cid = core.ingest_campaign(conn, title="Mexico launch", market="LATAM",
                               status="concluded", detail="A launch.",
                               starts_on="2026-03-01", ends_on="2026-03-31")["campaign_id"]

    window = context.window_of(conn, cid)
    assert (window["starts_on"], window["basis"]) == ("2026-03-01", "stated")


def test_a_concluded_campaign_with_no_window_is_offered_one_at_upload(conn):
    """`after_upload` already offers `add_metrics` for a concluded campaign with no results —
    the same shape. Without this the absence is invisible at the one moment somebody is
    present and thinking about the campaign."""
    out = core.ingest_campaign(conn, title="Mexico launch", market="LATAM",
                               status="concluded", detail="A launch.")

    offers = [a for a in out.get("next_actions") or []
              if a["tool"] == "update_campaign" and "starts_on" in str(a)]
    assert offers, out.get("next_actions")


def test_a_campaign_that_already_has_a_window_is_not_offered_one(conn):
    out = core.ingest_campaign(conn, title="Mexico launch", market="LATAM",
                               status="concluded", detail="A launch.",
                               starts_on="2026-03-01", ends_on="2026-03-31")

    assert not [a for a in out.get("next_actions") or []
                if a["tool"] == "update_campaign" and "starts_on" in str(a)]


def test_a_library_of_windowless_campaigns_is_a_gap(conn):
    """§9.5 gave itself `execution_never_checked` for exactly this reason. Without a gap, no
    surface ever says "41 of your 60 concluded campaigns have no window, so none of them can
    be checked against a calendar" — and the absence stays invisible forever."""
    for n in range(3):
        _campaign(conn, f"Mexico {n}")
    with_window = _campaign(conn, "Mexico dated")
    core.update_campaign(conn, campaign_id=with_window, starts_on="2026-03-01",
                         ends_on="2026-03-31")

    found = [g for g in core.gaps(conn)["gaps"] if g["code"] == "no_window"]

    assert found, core.gaps(conn)["gaps"]
    assert found[0]["counts"]["campaigns"] == 3
    assert found[0]["next_actions"][0]["tool"] == "update_campaign"


def test_the_gap_closes_when_the_windows_are_entered(conn):
    cid = _campaign(conn)
    assert [g for g in core.gaps(conn)["gaps"] if g["code"] == "no_window"]

    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")

    assert not [g for g in core.gaps(conn)["gaps"] if g["code"] == "no_window"]


# ── the two directions are one arithmetic, or they are two answers ──────────

def test_the_directions_agree_on_non_ascii_markets(conn):
    """SQL `COLLATE NOCASE` folds ASCII only; Python `.lower()` folds Unicode. So
    `market='MÉXICO'` against `scope_value='méxico'` matched one way and not the other:
    recording the event said "it overlaps 1 campaign" and offered a link, and following that
    link said nothing overlapped."""
    cid = _campaign(conn, market="MÉXICO")
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")

    saved = _quake(conn, scope_value="méxico")

    assert saved["now_overlaps"] == 1
    assert len(context.for_campaign(conn, cid)["events"]) == 1


def test_the_directions_agree_on_the_lower_boundary(conn):
    """Only the forward direction had boundary tests, so the reverse could drop an event that
    ended on a campaign's first day and nothing noticed."""
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-12", ends_on="2026-03-31")

    saved = _quake(conn, starts_on="2026-03-01", ends_on="2026-03-12")

    assert saved["now_overlaps"] == 1
    assert len(context.for_campaign(conn, cid)["events"]) == 1


def test_the_directions_agree_on_the_upper_boundary(conn):
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-10")

    saved = _quake(conn, starts_on="2026-03-10", ends_on="2026-03-14")

    assert saved["now_overlaps"] == 1
    assert len(context.for_campaign(conn, cid)["events"]) == 1


def test_the_directions_agree_on_markets_membership(conn):
    """The earlier agreement test put its membership campaign outside the event's market, so
    the branch it was written for was never exercised."""
    cid = _campaign(conn, market="SEA", markets=["Malaysia", "Indonesia"])
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")

    saved = _quake(conn, scope_value="Indonesia")

    assert saved["now_overlaps"] == 1
    assert len(context.for_campaign(conn, cid)["events"]) == 1


def test_the_directions_agree_on_region_scope(conn):
    cid = _campaign(conn, market="Mexico")
    core.update_campaign(conn, campaign_id=cid, region="LATAM", starts_on="2026-03-01",
                         ends_on="2026-03-31")

    saved = _quake(conn, scope="region", scope_value="LATAM")

    assert saved["now_overlaps"] == 1
    assert len(context.for_campaign(conn, cid)["events"]) == 1


def test_a_superseded_record_is_not_a_campaign_that_ran(conn):
    """Superseded records are excluded from every search, so counting one as a campaign an
    event reached describes something no judgment can see — and the offer pointed at it,
    because it is the older of the two."""
    old = _campaign(conn, "OLD v1")
    core.update_campaign(conn, campaign_id=old, starts_on="2026-03-01", ends_on="2026-03-31")
    new = _campaign(conn, "NEW v2")
    core.update_campaign(conn, campaign_id=new, starts_on="2026-03-01", ends_on="2026-03-31",
                         supersedes=old)

    saved = _quake(conn)

    assert saved["now_overlaps"] == 1
    assert saved["next_actions"][0]["prefilled_args"]["campaign_id"] == new


def test_reference_material_is_not_a_campaign_that_ran(conn):
    """A brand-guidelines PDF saying "published 2026-03-05" is not a campaign that ran through
    an earthquake. The rulebook is not precedent, and nothing else here treats it as any."""
    core.ingest_campaign(conn, title="Brand guidelines", record_type="reference",
                         market="LATAM", detail="Published 2026-03-05. Colour and type.")

    saved = _quake(conn)

    assert saved["now_overlaps"] == 0


# ── what truncation must never hide ─────────────────────────────────────────

def test_the_event_with_a_stated_impact_survives_a_calendar(conn):
    """Ordered by date and cut at eight, "earliest eight" after §9.7 seeds a market's calendar
    means "eight holidays" — and the flood with a stated 14-day delay, the one event §9.8
    needs linked to a confounded outcome, is the one that disappears. A model handed eight
    holiday rows and the number 27 will summarise "it ran through the usual holidays"."""
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-01-01", ends_on="2026-12-31")
    for day in range(1, 11):
        _quake(conn, kind="fixed_calendar", starts_on=f"2026-01-{day:02d}",
               ends_on=f"2026-01-{day:02d}", description=f"Public holiday {day}.",
               seeded=True)
    _quake(conn, kind="natural_disaster", starts_on="2026-11-02", ends_on="2026-11-16",
           description="Flooding closed the Pacific corridor.", delay_days=14,
           budget_change_pct=-12.0)

    linked = context.for_campaign(conn, cid)

    kinds = [e["kind"] for e in linked["events"]]
    assert "natural_disaster" in kinds, kinds
    assert linked["events"][0]["kind"] == "natural_disaster", "and it comes first"


def test_the_kinds_survive_truncation_even_when_the_rows_do_not(conn):
    """The count alone cannot say WHAT was going on, which is the half a reader acts on."""
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-01-01", ends_on="2026-12-31")
    for day in range(1, 12):
        _quake(conn, kind="fixed_calendar", starts_on=f"2026-01-{day:02d}",
               ends_on=f"2026-01-{day:02d}", description=f"Public holiday {day}.",
               seeded=True)
    _quake(conn, kind="competitor_launch", starts_on="2026-06-01", ends_on="2026-06-30",
           description="A rival launched into the same shelf.")

    linked = context.for_campaign(conn, cid)

    assert linked["by_kind"] == {"fixed_calendar": 11, "competitor_launch": 1}
    assert "competitor_launch" in linked["what_it_means"]


def test_a_seeded_calendar_does_not_crowd_out_what_somebody_recorded(conn):
    """A row this customer entered is about their market; a seeded holiday is about the
    calendar. Both are real, and only one of them was worth somebody's time."""
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-01-01", ends_on="2026-12-31")
    for day in range(1, 11):
        _quake(conn, kind="supply_chain", starts_on=f"2026-01-{day:02d}",
               ends_on=f"2026-01-{day:02d}", description=f"Seeded disruption {day}.",
               seeded=True)
    _quake(conn, kind="supply_chain", starts_on="2026-09-01", ends_on=None,
           description="Port of Manzanillo closed indefinitely.")

    shown = context.for_campaign(conn, cid)["events"]

    assert any("Manzanillo" in e["description"] for e in shown)


# ── D119's monthly workbook ─────────────────────────────────────────────────

def test_twelve_monthly_rows_do_not_leave_a_january_campaign(conn):
    """A KPI workbook is one row per month, and the first row's month became the whole
    campaign's window — labelled `stated`, so nothing downstream doubted it. A March event
    then missed a campaign that ran all year, and the product said "checked, nothing
    overlapped" about two-thirds of it."""
    cid = _campaign(conn)
    for month in range(1, 13):
        core.add_metrics(conn, campaign_id=cid, confirm=True,
                         structured={"month": f"2026-{month:02d}", "roas": 3.0})

    window = context.window_of(conn, cid)

    assert (window["starts_on"], window["ends_on"]) == ("2026-01-01", "2026-12-31")


def test_a_later_row_that_widens_the_window_says_so(conn):
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, confirm=True,
                     structured={"month": "2026-01", "roas": 3.0})

    out = core.add_metrics(conn, campaign_id=cid, confirm=True,
                           structured={"month": "2026-03", "roas": 3.1})

    assert out["window_set"]["ends_on"] == "2026-03-31"
    assert not any(s["key"] == "month" for s in out.get("skipped") or []), (
        "it was used, so calling it skipped is a false statement about the import")


def test_a_row_that_widens_nothing_reports_no_window_change(conn):
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, confirm=True,
                     structured={"month": "2026-01", "roas": 3.0})

    out = core.add_metrics(conn, campaign_id=cid, confirm=True,
                           structured={"month": "2026-01", "roas": 3.1})

    assert "window_set" not in out
    assert not any(s["key"] == "month" for s in out.get("skipped") or [])


def test_a_workbook_never_widens_a_window_somebody_typed(conn):
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-04-01", ends_on="2026-04-30")

    core.add_metrics(conn, campaign_id=cid, confirm=True,
                     structured={"month": "2026-01", "roas": 3.0})

    window = context.window_of(conn, cid)
    assert (window["starts_on"], window["ends_on"]) == ("2026-04-01", "2026-04-30")


def test_a_workbook_window_is_offered_the_check_it_made_possible(conn):
    """D116: `update_campaign` got the offer and `add_metrics` — the other write that sets a
    window — did not."""
    _quake(conn)
    cid = _campaign(conn)

    out = core.add_metrics(conn, campaign_id=cid, confirm=True,
                           structured={"month": "2026-03", "roas": 3.0})

    assert any(a["tool"] == "campaign_context" for a in out.get("next_actions") or [])


def test_an_inverted_window_from_a_spreadsheet_is_refused(conn):
    """`set_campaign_window` bypassed every check `update_campaign` makes."""
    cid = _campaign(conn)

    with pytest.raises(ValueError):
        core.add_metrics(conn, campaign_id=cid, confirm=True,
                         structured={"start_date": "2026-03-31", "end_date": "2026-03-01",
                                     "roas": 3.0})


# ── a wrong event is permanent and contagious without this ──────────────────

def test_an_event_can_be_withdrawn(conn):
    """Its siblings in this phase all have a correction path — `commitments.drop`,
    `drift.classify`'s never-overwritten history. An event typed with the wrong year attaches
    itself silently to every overlapping campaign forever, and after §9.8 will stamp
    `confounded` on their outcomes. That is strictly worse than the hand-maintained join this
    replaces: a join table at least lets you unlink."""
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")
    saved = _quake(conn)
    assert len(context.for_campaign(conn, cid)["events"]) == 1

    context.withdraw(conn, event_id=saved["id"], why="Wrong year — this was 2025.",
                     withdrawn_by="R. Vega")

    assert context.for_campaign(conn, cid)["events"] == []


def test_a_withdrawn_event_is_kept_rather_than_deleted(conn):
    """§8.5's rule: "we used to think this" is an answer, and deleting is a lie. A judgment
    saved while the event was on file rested on it."""
    saved = _quake(conn)

    context.withdraw(conn, event_id=saved["id"], why="Wrong year — this was 2025.",
                     withdrawn_by="R. Vega")

    gone = store.get_context_event(conn, saved["id"])
    assert gone["withdrawn_why"] == "Wrong year — this was 2025."
    assert gone["withdrawn_by"] == "R. Vega"


def test_withdrawing_needs_a_reason_and_a_person(conn):
    saved = _quake(conn)

    with pytest.raises(ValueError) as e:
        context.withdraw(conn, event_id=saved["id"], why="", withdrawn_by="R. Vega")
    assert "why" in str(e.value).lower()
    with pytest.raises(ValueError) as e:
        context.withdraw(conn, event_id=saved["id"], why="Wrong year.", withdrawn_by="")
    assert "person" in str(e.value).lower()


def test_withdrawing_something_that_is_not_there_is_refused(conn):
    with pytest.raises(ValueError):
        context.withdraw(conn, event_id="ctx_nope", why="Wrong year.", withdrawn_by="R. Vega")


def test_a_withdrawn_event_says_what_it_used_to_reach(conn):
    """Withdrawing is as consequential as recording — it removes a caveat from every campaign
    that was carrying it, and silence about that is the same failure as silence about the
    link."""
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")
    saved = _quake(conn)

    out = context.withdraw(conn, event_id=saved["id"], why="Wrong year — this was 2025.",
                           withdrawn_by="R. Vega")

    assert out["no_longer_overlaps"] == 1
    assert "Mexico launch" in out["what_it_means"]


# ── the seam 9.7 and 9.8 need ──────────────────────────────────────────────

def test_a_window_can_be_checked_without_a_stored_campaign(conn):
    """§9.7 checks a PROPOSED window at review time, and a proposal may not be stored yet —
    `prepare_evaluation` already handles that shape with `facts.compute(proposal_text)`.
    Without this seam it would reach around into `store` for raw rows and rebuild the wording,
    which is how two surfaces come to describe the same library differently."""
    _quake(conn)

    out = context.for_window(conn, starts_on="2026-03-01", ends_on="2026-03-31",
                             markets=["LATAM"])

    assert out["status"] == "checked"
    assert len(out["events"]) == 1
    assert out["basis"] == "computed"


def test_a_proposed_window_with_no_market_is_not_given_a_clean_bill(conn):
    _quake(conn)

    out = context.for_window(conn, starts_on="2026-03-01", ends_on="2026-03-31", markets=[])

    assert out["status"] == "nothing_to_check"


def test_a_narrower_window_than_the_campaign_can_be_asked_about(conn):
    """§9.8 marks an OUTCOME confounded, and a metric's measurement period is usually much
    narrower than an always-on campaign's year. Confounding every metric on a record because
    of a three-day June quake would make the caveat always-on, which is a caveat nobody
    reads."""
    _quake(conn, starts_on="2026-06-01", ends_on="2026-06-03")

    june = context.for_window(conn, starts_on="2026-06-01", ends_on="2026-06-30",
                              markets=["LATAM"])
    december = context.for_window(conn, starts_on="2026-12-01", ends_on="2026-12-31",
                                  markets=["LATAM"])

    assert len(june["events"]) == 1
    assert december["events"] == []
    assert december["status"] == "checked"


def test_a_seeded_event_can_be_found_again(conn):
    """§9.7 seeds a calendar per market and must be able to run twice without doubling it, and
    to replace a corrected Ramadan date on upgrade. `_seed_metric_registry` is idempotent
    because `canonical` is its primary key; a random `ctx_…` id gives a seeder no way back to
    the row it wrote."""
    first = context.record(conn, starts_on="2026-02-18", ends_on="2026-03-19", scope="market",
                           scope_value="UAE", kind="fixed_calendar", description="Ramadan.",
                           recorded_by="seed", seeded=True, seed_key="ae.ramadan.2026")

    again = context.record(conn, starts_on="2026-02-17", ends_on="2026-03-18", scope="market",
                           scope_value="UAE", kind="fixed_calendar",
                           description="Ramadan (corrected).", recorded_by="seed",
                           seeded=True, seed_key="ae.ramadan.2026")

    assert again["id"] == first["id"], "the same key is the same row"
    assert again["starts_on"] == "2026-02-17"
    assert len(store.context_events(conn)) == 1


def test_seeded_events_can_be_told_from_recorded_ones(conn):
    context.record(conn, starts_on="2026-02-18", ends_on="2026-03-19", scope="market",
                   scope_value="UAE", kind="fixed_calendar", description="Ramadan.",
                   recorded_by="seed", seeded=True, seed_key="ae.ramadan.2026")
    _quake(conn)

    assert len(store.context_events(conn, seeded=True)) == 1
    assert len(store.context_events(conn, seeded=False)) == 1


def test_a_person_never_overwrites_a_seeded_row_by_accident(conn):
    """A `seed_key` is the seeder's handle, not a field a person fills in."""
    with pytest.raises(ValueError) as e:
        _quake(conn, seed_key="ae.ramadan.2026")
    assert "seeded" in str(e.value).lower()


# ── the heuristic window's remaining over-reach ─────────────────────────────

def test_a_heuristic_window_never_says_the_campaign_ran_during_anything(conn):
    """"RAN DURING" is a fact-claim, and on the heuristic path the window is a reading of
    prose the docstring itself calls "can be wrong". A brief naming a competitor's launch and
    last year's activation produced a seventeen-month window and the sentence asserted the
    campaign ran through events four months before it started."""
    cid = _campaign(conn, detail="Flight 2-28 March 2026. Benchmark: a rival ran their push "
                                 "on 15 November 2025.")
    _quake(conn, starts_on="2025-11-15", ends_on="2025-11-20",
           description="A rival's push.", kind="competitor_launch")

    said = context.for_campaign(conn, cid)["what_it_means"]

    assert "RAN DURING" not in said
    assert "may have run during" in said.lower()


def test_an_absurdly_long_heuristic_window_is_not_a_window(conn):
    """A four-week flight read as seventeen months is not a conservative reading of a brief;
    it is a different and worse object. A typo — "3 March 2062" — produced a thirty-six-year
    window that would overlap every event the library will ever hold."""
    cid = _campaign(conn, detail="Flight runs 2 March 2026 to 28 March 2026. Our 2024 "
                                 "programme ran 1 June 2024 to 30 June 2024 for comparison. "
                                 "Deck exported 3 March 2062.")

    window = context.window_of(conn, cid)

    assert window["basis"] is None
    assert "too far apart" in window["what_it_means"].lower()


def test_a_plausible_heuristic_window_still_works(conn):
    cid = _campaign(conn, detail="Flight runs 2 March 2026 to 28 March 2026 across LATAM.")

    assert context.window_of(conn, cid)["basis"] == "heuristic"


# ── what the impact is about ────────────────────────────────────────────────

def test_a_stated_impact_says_whose_it_is(conn):
    """A market-wide event's impact is displayed inside one campaign's context list with no
    subject. Nothing said whether −12% was the market's budget or this campaign's, and a model
    reconciling that campaign's numbers now has a −12% figure sitting beside them."""
    saved = _quake(conn, delay_days=3, budget_change_pct=-12.0,
                   channels_disrupted=["retail"])

    assert "on the market" in saved["what_it_means"]
    assert "this campaign" not in saved["what_it_means"].lower()


def test_no_delay_is_a_stated_impact_and_not_a_silence(conn):
    """Zero is something somebody said: "it did not delay anything". Rendering it as "no
    impact was stated" throws away the one reading a reader most wants — that somebody looked
    and found none."""
    saved = _quake(conn, delay_days=0)

    assert "no impact was stated" not in saved["what_it_means"].lower()
    assert "no delay" in saved["what_it_means"].lower()


def test_a_negative_delay_is_refused(conn):
    with pytest.raises(ValueError) as e:
        _quake(conn, delay_days=-5)
    assert "delay" in str(e.value).lower()


def test_an_impossible_budget_change_is_refused(conn):
    """−500% is not a budget cut. A figure nobody can act on is worse than an absent one,
    because it travels."""
    with pytest.raises(ValueError) as e:
        _quake(conn, budget_change_pct=-500.0)
    assert "budget" in str(e.value).lower()


def test_channels_are_tidied_rather_than_stored_as_typed(conn):
    """§9.8 will reconcile these against the channel vocabulary the rest of the library uses,
    and " Retail " and "retail" being two disrupted channels is a difference that means
    nothing."""
    saved = _quake(conn, channels_disrupted=["Retail ", "retail", "TikTok", ""])

    assert saved["channels_disrupted"] == ["retail", "tiktok"]


def test_a_source_that_is_not_a_link_is_refused(conn):
    """The review says "source link". A sentence in the source field is a rumour wearing a
    citation's clothes."""
    with pytest.raises(ValueError) as e:
        _quake(conn, source="somebody mentioned it")
    assert "link" in str(e.value).lower()


def test_a_global_event_says_its_market_was_dropped(conn):
    """Dropping it silently is right; not SAYING so is not. The recorder typed a market and
    got an event that matches everywhere."""
    saved = _quake(conn, scope="global", scope_value="LATAM", kind="macro_shock")

    assert saved["scope_value"] is None
    assert "global" in saved["what_it_means"].lower()
    assert saved["warnings"], "the market they typed went somewhere they should know about"


# ── the caveat has to reach a reader who is not looking for it ──────────────

def test_a_cited_campaign_carries_what_was_going_on(conn):
    """§9.5's whole claim is that the caveat travels WITH the citation — `_execution_note` is
    wired into four read paths. §9.6 was wired into none: a reader looking at an evidence row
    saw nothing about the port closure that ran through half the flight, and would only find
    it by independently calling one tool that nothing recommends."""
    cited = _campaign(conn, "Mexico launch", market="LATAM")
    core.update_campaign(conn, campaign_id=cited, starts_on="2026-03-01",
                         ends_on="2026-03-31")
    _quake(conn)

    pkg = core.prepare_evaluation(conn, subject_title="Mexico v2",
                                  proposal_text="A retail activation in Mexico.")

    row = next(e for e in pkg["evidence"] if e["campaign_id"] == cited)
    assert row["context"]["status"] == "checked"
    assert row["context"]["events_total"] == 1
    assert row["context"]["by_kind"] == {"natural_disaster": 1}


def test_a_cited_campaign_with_no_window_says_so_in_the_evidence(conn):
    """`nothing_to_check` has to survive the trip. Absent, a reader cannot tell "we looked and
    nothing was going on" from "this campaign cannot be checked at all"."""
    cited = _campaign(conn, "Mexico launch", market="LATAM")
    _quake(conn)

    pkg = core.prepare_evaluation(conn, subject_title="Mexico v2",
                                  proposal_text="A retail activation in Mexico.")

    row = next(e for e in pkg["evidence"] if e["campaign_id"] == cited)
    assert row["context"]["status"] == "nothing_to_check"


def test_the_model_is_told_what_the_context_key_is_for(conn):
    cited = _campaign(conn, "Mexico launch", market="LATAM")
    core.update_campaign(conn, campaign_id=cited, starts_on="2026-03-01",
                         ends_on="2026-03-31")
    _quake(conn)

    note = core.prepare_evaluation(conn, subject_title="Mexico v2",
                                   proposal_text="A retail activation in Mexico.")["note"]

    assert "context" in note
    assert "not a cause" in note.lower() or "never a cause" in note.lower()


def test_the_note_says_nothing_when_no_cited_campaign_has_any(conn):
    """A standing paragraph about context events on a library holding none is the note that
    fires on everything — §9.5's own rule, one item over."""
    _campaign(conn, "Mexico launch", market="LATAM")

    note = core.prepare_evaluation(conn, subject_title="Mexico v2",
                                   proposal_text="A retail activation in Mexico.")["note"]

    assert "context events" not in note.lower()


# ── the last six the mutation pass found unguarded ─────────────────────────

def test_the_folds_agree_where_lower_and_casefold_disagree(conn):
    """"MÉXICO" vs "méxico" is not enough to prove one fold: `.lower()` handles it too. German
    ß is the case that separates them — `.casefold()` gives "ss", `.lower()` leaves "ß" — so
    this is the test that actually holds the two directions to the same function."""
    cid = _campaign(conn, market="GROSSMARKT")
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")

    saved = _quake(conn, scope_value="Großmarkt")

    assert saved["now_overlaps"] == 1
    assert len(context.for_campaign(conn, cid)["events"]) == 1


def test_the_cap_is_eight(conn):
    """A literal. `len(events) == MAX_EVENTS_SHOWN` is true of every cap, so setting the
    constant to 1 passed — the assertion measured the constant against itself."""
    assert context.MAX_EVENTS_SHOWN == 8
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-01-01", ends_on="2026-12-31")
    for day in range(1, 12):
        _quake(conn, starts_on=f"2026-01-{day:02d}", ends_on=f"2026-01-{day:02d}",
               description=f"Disruption {day}.")

    assert len(context.for_campaign(conn, cid)["events"]) == 8


def test_a_one_day_campaign_is_a_window(conn):
    """A single-day activation is ordinary — a launch party, a drop. Refusing it as "closes
    before it opens" would make the commonest short campaign unrecordable."""
    cid = _campaign(conn)

    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-10", ends_on="2026-03-10")

    assert context.window_of(conn, cid)["starts_on"] == "2026-03-10"
    _quake(conn)
    assert len(context.for_campaign(conn, cid)["events"]) == 1


def test_events_of_equal_salience_are_ordered_by_date(conn):
    """Salience decides the bands; inside a band the order still has to be readable, and it is
    what truncation cuts from."""
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-01-01", ends_on="2026-12-31")
    for month in (9, 3, 6):
        _quake(conn, starts_on=f"2026-{month:02d}-01", ends_on=f"2026-{month:02d}-05",
               description=f"Disruption in month {month}.")

    dates = [e["starts_on"] for e in context.for_campaign(conn, cid)["events"]]

    assert dates == ["2026-03-01", "2026-06-01", "2026-09-01"]


def test_the_offer_names_a_campaign_the_event_actually_reached(conn):
    """It picks one of several, and which one is not arbitrary — the offer is what the
    recorder clicks, so it has to land on a record they would recognise."""
    first = _campaign(conn, "Mexico launch")
    core.update_campaign(conn, campaign_id=first, starts_on="2026-03-01", ends_on="2026-03-31")
    second = _campaign(conn, "Colombia launch")
    core.update_campaign(conn, campaign_id=second, starts_on="2026-03-05",
                         ends_on="2026-03-20")

    saved = _quake(conn)

    assert saved["now_overlaps"] == 2
    assert saved["next_actions"][0]["prefilled_args"]["campaign_id"] == first
    assert "Mexico launch" in saved["next_actions"][0]["label"]


def test_a_database_without_the_table_refuses_rather_than_answering(conn):
    """Returning `[]` from a missing table reaches `for_campaign` as "checked, nothing
    overlapped" — the exact collapse this module forbids, arriving from a broken install
    rather than from a mistake. The only way here is that `init_db` never ran."""
    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, starts_on="2026-03-01", ends_on="2026-03-31")
    conn.execute("DROP TABLE context_events")
    conn.commit()

    with pytest.raises(RuntimeError) as e:
        context.for_campaign(conn, cid)
    assert "upgrade" in str(e.value).lower()
