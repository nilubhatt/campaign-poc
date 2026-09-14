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
