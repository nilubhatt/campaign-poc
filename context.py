"""
§9.6: a context events table, scoped by market and date.

The review: *"Conflict, natural disaster, regulatory change, supply-chain or port disruption,
platform outage, competitor launch, macro shock, and fixed calendar events. Fields: date range,
scope (market, region or global), kind, description, source link, and a stated impact — delay
in days, budget change, channels disrupted. Any campaign whose window overlaps an event in its
market links automatically. **No one should have to remember to connect them.**"*

**"No one should have to remember to connect them" is the item.** A join table somebody
maintains by hand is the thing this replaces, not the thing it is — so there is no link table.
The connection is computed from (scope, range) against (market, window) on every read, because
both halves keep arriving: an earthquake is recorded weeks after the campaigns it overlapped,
and a campaign is uploaded months after its market's calendar was seeded. A link written at
upload time would miss the first; one written when the event lands would miss the second.

**A campaign had no window, and that was the load-bearing gap.** Dates existed only as prose
`facts.py` parses for contradictions, so there was nothing for a date range to overlap with.
`starts_on`/`ends_on` is the field this turns on — and the honest default is NOT to invent one.
Where somebody typed the dates the window is `stated`; where the brief's own dates are the only
evidence it is `heuristic` and says which dates it was read from; where there are none there is
no window, and the answer is `nothing_to_check` rather than "no events overlapped". The second
is a claim, and it is the claim this table most wants to be able to make.

**The link is an overlap and nothing more.** §9.8 owns `confounded` and the caveat that travels
with a metric. What this must not do is get there early — *"'Sell-through was down and there
was an earthquake' is not evidence the earthquake caused it. A model asked to explain a
disappointing number will reach for whatever is nearby"*, and this is the nearest thing there
will ever be. "Ran during" is a fact; "affected by" is an assertion this module never makes.
"""
from __future__ import annotations

import datetime
from typing import Optional

import identity

# The review's own list, one value each. A vocabulary missing one sends that event to whichever
# word is nearest, and the nearest word to "port strike" is whatever the person typing happens
# to be thinking of — which is how a taxonomy stops being one.
KINDS = ("conflict", "natural_disaster", "regulatory_change", "supply_chain",
         "platform_outage", "competitor_launch", "macro_shock", "fixed_calendar")

SCOPES = ("market", "region", "global")

_KIND_WORD = {
    "conflict": "conflict",
    "natural_disaster": "a natural disaster",
    "regulatory_change": "a regulatory change",
    "supply_chain": "a supply-chain or port disruption",
    "platform_outage": "a platform outage",
    "competitor_launch": "a competitor launch",
    "macro_shock": "a macroeconomic shock",
    "fixed_calendar": "a fixed calendar event",
}

# How many events one campaign's answer carries. A campaign running through a quarter in a
# seeded market genuinely overlaps a dozen holidays, and a list that long stops being read —
# but the COUNT stays complete, because "twelve things were going on" is the finding.
MAX_EVENTS_SHOWN = 8

# The longest span this will read out of a brief's prose. `facts._parsed_dates` is careful
# about whether a TOKEN is a date and says nothing about whether a date is a flight boundary —
# and `min`/`max` is the least conservative aggregation there is, so one outlier sets the
# bound. Measured on ordinary brief prose: a four-week flight mentioning a competitor's launch
# and last year's activation became a seventeen-month window, and a typo ("3 March 2062") a
# thirty-six-year one that would overlap every event this library will ever hold. Past this,
# the dates are not describing one campaign's flight and the honest answer is no window.
MAX_HEURISTIC_DAYS = 200


def record(conn, *, starts_on: str, scope: str, kind: str, description: str,
           recorded_by: str, ends_on: Optional[str] = None,
           scope_value: Optional[str] = None, source: Optional[str] = None,
           delay_days: Optional[int] = None, budget_change_pct: Optional[float] = None,
           channels_disrupted: Optional[list] = None, seeded: bool = False,
           seed_key: Optional[str] = None, certainty: Optional[str] = None,
           recurs_annually: bool = False, role: Optional[str] = None) -> dict:
    """Put an event on the record (§9.6).

    `basis` is `stated` and never anything else. The server did not measure a three-day delay
    or a twelve-percent budget cut; somebody said so, and §2.4 made `computed` unwritable from
    the MCP surface precisely so the word keeps meaning "the server worked this out".
    """
    import enums
    import store

    kind = enums.normalise(kind, field="kind", valid=KINDS, allow_none=False)
    scope = enums.normalise(scope, field="scope", valid=SCOPES, allow_none=False)
    starts = _a_date(starts_on, "starts_on")
    ends = _a_date(ends_on, "ends_on") if (ends_on or "").strip() else None
    if ends and ends < starts:
        raise ValueError(
            f"`ends_on` ({ends}) is before `starts_on` ({starts}). An event that ends before "
            f"it starts matches no window at all, so it would sit on the record looking "
            f"recorded and reach nothing.")
    warnings = []
    if scope == "global":
        # Not an error: "global" plus a market is a contradiction somebody probably did not
        # mean, and silently keeping the market would make the event match one market only
        # while reading as global everywhere it is shown. Dropping it is right; dropping it
        # WITHOUT SAYING SO is not — the recorder typed a market and got something that
        # matches everywhere.
        if (scope_value or "").strip():
            import notices
            warnings.append(notices.notice(
                "scope_value_ignored",
                affects=f"this event is scoped `global`, so the market you named "
                        f"({scope_value.strip()}) was not stored — it matches campaigns in "
                        f"every market. Record it with scope `market` if it was local.",
                detail=f"scope=global, scope_value={scope_value.strip()!r}"))
        scope_value = None
    elif not (scope_value or "").strip():
        raise ValueError(
            f"`scope_value` is required for scope {scope!r}: name the {scope} this applies to. "
            f"An event scoped to a market with no market named links to nothing, and `global` "
            f"is the value for something that applies everywhere.")
    if not (description or "").strip():
        raise ValueError(
            "`description` is required: a date range and a kind cannot tell a reader what "
            "happened, and this is read beside a campaign's results months afterwards.")
    # §11.1/§11.2: one guard, and the seeded calendar is the only thing exempt — it is
    # shipped with the product and says so, which is a true statement about authorship rather
    # than a person's name borrowed by a machine.
    if not seeded:
        identity.person(recorded_by, field="recorded_by")
    if delay_days is not None and delay_days < 0:
        raise ValueError(
            f"`delay_days` is {delay_days}: a delay cannot be negative. Zero is the value for "
            f"“it delayed nothing”, which is a real finding and worth recording.")
    if budget_change_pct is not None and not (-100.0 <= budget_change_pct <= 1000.0):
        raise ValueError(
            f"`budget_change_pct` is {budget_change_pct}: a budget cannot fall by more than "
            f"100%. A figure nobody can act on is worse than an absent one, because it "
            f"travels with the event everywhere it is cited.")
    if (source or "").strip() and not str(source).strip().lower().startswith(
            ("http://", "https://", "file:", "mailto:")):
        raise ValueError(
            f"`source` is a LINK — {source!r} is a sentence. Somewhere a reader can go and "
            f"check is what separates this from a rumour on the record; put an account of "
            f"what happened in `description` instead.")
    if seed_key and not seeded:
        # The seeder's handle on its own rows, not a field a person fills in — passing one by
        # hand would silently REPLACE a shipped calendar entry rather than adding an event.
        raise ValueError(
            "`seed_key` belongs to the seeded calendar (§9.7) and replaces the row that "
            "already holds it. Recording something that happened does not take one.")

    eid = store.insert_context_event(
        conn, starts_on=str(starts), ends_on=str(ends) if ends else None, scope=scope,
        scope_value=(scope_value or "").strip() or None, kind=kind,
        description=description.strip(), source=(source or "").strip() or None,
        delay_days=delay_days, budget_change_pct=budget_change_pct,
        channels_disrupted=_tidy_channels(channels_disrupted),
        recorded_by=recorded_by.strip(), seeded=seeded, seed_key=seed_key,
        certainty=certainty, recurs_annually=recurs_annually)
    if not seeded:
        store.record_authorship(conn, subject_kind="context_event", subject_key=eid,
                                on_behalf_of=recorded_by, role=role)
    saved = _public(store.get_context_event(conn, eid))
    if warnings:
        saved = {**saved, "warnings": warnings}
    # D116: the link is automatic, which makes it INVISIBLE. Nothing on screen changes when an
    # event lands, so the person who recorded it has no way to know whether it reached
    # anything — and a feature whose whole promise is "no one should have to remember to
    # connect them" has to say that it connected.
    return {**saved, **_what_it_reached(conn, saved)}


def _what_it_reached(conn, event: dict) -> dict:
    import actions

    reached = campaigns_overlapping(conn, event)
    if not reached:
        return {"now_overlaps": 0,
                "what_it_means": saved_nothing_reached(event)}
    first = reached[0]
    return {
        "now_overlaps": len(reached),
        "what_it_means": (
            f"{event['what_it_means']} It overlaps {len(reached)} campaign(s) already in this "
            f"library, starting with “{first['title']}” — they picked it up automatically and "
            f"there is nothing to link. That is an overlap in time and nothing more."),
        "next_actions": actions.trim([actions.action(
            f"See everything that was going on while “{first['title'][:36]}” ran",
            "campaign_context",
            why="This event now overlaps it. Reading them together is what the record is for "
                "— and the overlap is a fact about timing, never about cause.",
            consent="ask", campaign_id=first["campaign_id"])]),
    }


def saved_nothing_reached(event: dict) -> str:
    return (f"{event['what_it_means']} It overlaps no campaign in this library yet — either "
            f"none ran in that market at that time, or the ones that did have no window on "
            f"file. Anything uploaded later that overlaps it will pick it up automatically.")


def campaigns_overlapping(conn, event: dict) -> list:
    """Which campaigns this event reaches (§9.6), the same arithmetic run the other way.

    Computed rather than stored, for the reason in the module docstring — and computed HERE
    rather than duplicated, so the two directions cannot disagree about what an overlap is.
    """
    import store

    # The same exclusions every search in this library makes. A superseded record is hidden
    # from retrieval, so counting one as a campaign this event reached describes something no
    # judgment can see — and the offer pointed straight at it, being the older of the pair. A
    # `reference` record is the rulebook: a brand-guidelines PDF dated "published 2026-03-05"
    # is not a campaign that ran through an earthquake.
    superseded = store.get_superseded_campaign_ids(conn)
    reached = []
    for record in store.list_campaigns(conn):
        if record["id"] in superseded or record.get("record_type") == "reference":
            continue
        window = window_of(conn, record["id"])
        if not window["starts_on"]:
            continue
        if not _in_scope(record, event):
            continue
        if event["starts_on"] > (window["ends_on"] or str(datetime.date.max)):
            continue
        if event["ends_on"] is not None and event["ends_on"] < window["starts_on"]:
            continue
        reached.append({"campaign_id": record["id"], "title": record["title"]})
    return reached


def _in_scope(record: dict, event: dict) -> bool:
    """The reverse direction's half of the same question, folded the same way (§9.6).

    `store.fold` and not a second `.lower()` here: the forward direction matches a stored
    folded key in SQL, and two folds that differ mean an event reporting it reached a campaign
    that does not list it — which is worse than either answer on its own.
    """
    import store

    if event["scope"] == "global":
        return True
    return store.fold(event["scope_value"]) in {store.fold(v) for v in _market_names(record)}


def _tidy_channels(values) -> list:
    """Folded and de-duplicated. §9.8 reconciles these against the channel vocabulary the rest
    of the library uses, and " Retail " and "retail" being two disrupted channels is a
    difference that means nothing to anybody."""
    seen, out = set(), []
    for value in values or []:
        text = str(value).strip().casefold()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _a_date(value, field: str) -> datetime.date:
    try:
        return datetime.date.fromisoformat(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(
            f"`{field}` must be an ISO date (YYYY-MM-DD); got {value!r}. Everything here is "
            f"compared against a campaign's window by date arithmetic, and a string that is "
            f"not a date would silently compare as text.") from None


def _public(row: dict) -> dict:
    ongoing = row["ends_on"] is None
    when = (f"from {row['starts_on']}, ongoing" if ongoing
            else f"{row['starts_on']} to {row['ends_on']}")
    said = (f"{_KIND_WORD[row['kind']]} — {row['description']} ({when}"
            f"{'' if row['scope'] == 'global' else ', ' + str(row['scope_value'])}"
            f"{', global' if row['scope'] == 'global' else ''}). "
            f"Recorded by {row['recorded_by']}.")
    impact = _impact_sentence(row)
    return {
        **row,
        # Never `computed`. Every field here is somebody's account, including the impact.
        "basis": "stated",
        "what_it_means": said + impact + _how_well_it_is_known(row),
    }


def _how_well_it_is_known(row: dict) -> str:
    """Whether this date is ours to be wrong about, and how wrong it could be (§9.7).

    A shipped calendar is a claim about the world. Ramadan begins on a sighting and starts a
    day apart in neighbouring countries; a monsoon has an onset that moves by weeks. Presenting
    those with the same confidence as Singles' Day would be the confident unfounded claim this
    product is written against — and a customer correcting one has to know it was ours.
    """
    if not row.get("seeded"):
        return ""
    import calendar_seed

    return (" This came from the calendar shipped with this product rather than from anybody "
            "here. " + calendar_seed.note_for(row.get("certainty"))
            + " Withdraw it if it does not apply to your market.")


def _impact_sentence(row: dict) -> str:
    """What somebody said this did — to the MARKET, not to any one campaign.

    The subject is the fix. A market-wide event's impact is displayed inside one campaign's
    context list, and with no subject a model reconciling that campaign's numbers finds a
    "-12%" sitting beside them with nothing saying whose budget it was. The existing hedge
    guards provenance ("recorded, not measured"); this guards the subject.
    """
    parts = []
    if row.get("delay_days") is not None:
        # Zero is a stated value: somebody looked and found no delay. Rendered as silence it
        # threw away the one reading a reader most wants.
        parts.append("no delay" if row["delay_days"] == 0
                     else f"a delay of {row['delay_days']} day(s)")
    if row.get("budget_change_pct") is not None:
        parts.append("no budget change" if row["budget_change_pct"] == 0
                     else f"a budget change of {row['budget_change_pct']:+g}%")
    if row.get("channels_disrupted"):
        parts.append("disruption to " + ", ".join(row["channels_disrupted"]))
    if not parts:
        return " No impact was stated."
    return (" Stated impact on the market: " + "; ".join(parts) + ". That is what somebody "
            "recorded about the market, not something measured here and not a figure for any "
            "one campaign.")


def window_of(conn, campaign_id: str) -> dict:
    """When this campaign ran, and how well that is known (§9.6).

    Three answers, and collapsing any two of them is the defect this shape exists to prevent.
    `stated` is dates somebody entered. `heuristic` is dates read out of the brief's own prose,
    which can be wrong and must never arrive looking like the first. `None` is no window at
    all — not a guess, and not zero.
    """
    import store

    record = store.get_campaign(conn, campaign_id)
    if record is None:
        raise ValueError(f"{campaign_id!r} is not a record in this library.")
    if record.get("starts_on"):
        return {"starts_on": record["starts_on"], "ends_on": record.get("ends_on"),
                "basis": "stated",
                "what_it_means": (
                    f"This campaign's window was entered as "
                    f"{record['starts_on']} to {record.get('ends_on') or 'open-ended'}.")}
    return _window_from_the_brief(record)


def _window_from_the_brief(record: dict) -> dict:
    """The earliest and latest dates the brief itself names (§9.6).

    `facts._parsed_dates` is deliberately not a general date parser — a wrong date read out of
    a brief is worse than an unread one — so this inherits that conservatism rather than
    loosening it. Dates with no year are dropped: `_parsed_dates` stands them up in year 2000
    to keep its spans comparable, and a window running from 2000 would overlap nothing and
    look like a real answer while doing it.
    """
    import facts

    text = " ".join(filter(None, [record.get("detail"), record.get("deck_text")]))
    dated = [value for _, value, has_year in facts._parsed_dates(text) if has_year]
    if not dated:
        return {"starts_on": None, "ends_on": None, "basis": None,
                "what_it_means": (
                    "This campaign has no window: no dates were entered, and none that this "
                    "reader recognises appear in the brief. That is not the same as it having "
                    "run at no particular time — nothing here can be checked against a "
                    "calendar until somebody says when it ran.")}
    first, last = min(dated), max(dated)
    if (last - first).days > MAX_HEURISTIC_DAYS:
        return {"starts_on": None, "ends_on": None, "basis": None,
                "what_it_means": (
                    f"This campaign has no window. No dates were entered, and the ones in the "
                    f"brief ({first} to {last}) are too far apart to be one campaign's "
                    f"flight — a competitor's launch date, last year's recap or a typo reads "
                    f"to a parser exactly like a flighting date. Enter the dates to settle "
                    f"it.")}
    first, last = str(first), str(last)
    return {
        "starts_on": first, "ends_on": last, "basis": "heuristic",
        "what_it_means": (
            f"No window was entered, so this one was READ FROM the brief's own dates — the "
            f"earliest ({first}) and the latest ({last}) it mentions. That is a reading of "
            f"prose and can be wrong: a date in a competitor note or a past campaign's recap "
            f"counts the same as a flighting date to a parser. Enter the window to settle it."),
    }


def for_campaign(conn, campaign_id: str) -> dict:
    """What else was going on while this campaign ran (§9.6).

    Computed on every read, never stored. See the module docstring: both halves keep arriving,
    and a link written at either moment is wrong about the other.
    """
    import actions
    import store

    record = store.get_campaign(conn, campaign_id)
    if record is None:
        raise ValueError(f"{campaign_id!r} is not a record in this library.")
    window = window_of(conn, campaign_id)
    if not window["starts_on"]:
        return {
            "campaign_id": campaign_id, "title": record["title"],
            "status": "nothing_to_check", "basis": "computed",
            "window": window, "events": [], "events_total": 0,
            # NEVER `events: []` with a `checked` status. "No events overlapped this campaign"
            # is a claim, and a campaign with no window has not earned it — an empty list that
            # means "could not check" reads identically to one that means "checked, nothing
            # there", which is the collapse this whole product is written against.
            "what_it_means": (
                "This campaign has no window, so nothing could be checked against the "
                "calendar. That is not the same as nothing having been going on."),
            "next_actions": actions.trim([actions.action(
                f"Say when “{record['title'][:40]}” ran", "update_campaign",
                why="Without a window, no context event can be matched to it — and an outcome "
                    "read with no idea what else was happening that month is the reading this "
                    "library exists to improve.",
                consent="ask",
                needs=["starts_on — the first day it ran (YYYY-MM-DD)",
                       "ends_on — the last day (YYYY-MM-DD)"],
                campaign_id=campaign_id)]),
        }

    scopes = _scopes_of(record)
    searched = [value or "global" for _, value in scopes]
    events = [_public(e) for e in store.context_events(
        conn, scopes=scopes, starts_on=window["starts_on"],
        ends_on=window["ends_on"] or str(datetime.date.max))]
    shown = sorted(events, key=_salience)[:MAX_EVENTS_SHOWN]
    # THREE routes to a clean bill, and two of them are false. An empty answer is only a claim
    # about the market when the search that produced it could have found something: a window
    # read out of prose can be a stray asset-spec date, and a record naming no market is
    # searched against global events alone. Both returned "Nothing on record was going on.
    # That is a real answer." — the exact sentence the `nothing_to_check` split exists to
    # prevent, reintroduced by the two mechanisms that widen coverage.
    #
    # A positive MATCH from either still stands. Finding something is not weakened by the
    # window being a reading; only the negative claim is.
    unearned = [] if events else _why_the_empty_answer_is_not_a_claim(record, window)
    if unearned:
        return {
            "campaign_id": campaign_id, "title": record["title"],
            "status": "nothing_to_check", "basis": "computed",
            "window": window, "events": [], "events_total": 0, "searched": searched,
            "what_it_means": (
                "Nothing overlapping was found, but this campaign has not earned the "
                "answer \u201cnothing was going on\u201d: " + " ".join(unearned)),
            "next_actions": actions.trim([actions.action(
                f"Say when and where \u201c{record['title'][:36]}\u201d ran",
                "update_campaign",
                why="Until then, an empty result here means the search was too narrow to "
                    "find anything — not that nothing was happening.",
                consent="ask",
                needs=["starts_on — the first day it ran (YYYY-MM-DD)",
                       "ends_on — the last day (YYYY-MM-DD)",
                       "market — which market it ran in"],
                campaign_id=campaign_id)]),
        }
    return {
        "campaign_id": campaign_id, "title": record["title"],
        "status": "checked",
        "searched": searched,
        # D116: offered only when the answer is empty. "Checked, and nothing was going on" is
        # only as true as the calendar behind it, and on a library where nobody has recorded
        # anything that sentence is about the library rather than the market. Where events DID
        # come back the question is answered, and an offer that is always there stops being
        # read — which is the failure mode D116 catalogues, not a second instance of it.
        **({"next_actions": actions.trim([actions.action(
            f"Record what was going on in "
            f"{record.get('market') or record.get('region') or 'this market'} "
            f"while “{record['title'][:36]}” ran",
            "record_context_event",
            why="Nothing on record overlapped this campaign, which is a fact about what has "
                "been RECORDED and not about the market. An outcome read with no idea what "
                "else was happening that month is the reading this library exists to improve.",
            consent="ask",
            needs=["starts_on — when it began (YYYY-MM-DD)",
                   "kind — conflict, natural_disaster, regulatory_change, supply_chain, "
                   "platform_outage, competitor_launch, macro_shock or fixed_calendar",
                   "description — what happened",
                   "recorded_by — whose account this is"],
            scope="market",
            scope_value=record.get("market") or record.get("region") or None)])}
           if not events else {}),
        # The OVERLAP is computed — it is arithmetic on two date ranges. Each event's own
        # contents are `stated`, and they say so individually.
        "basis": "computed",
        "window": window,
        "events": shown,
        "events_total": len(events),
        # The COUNT cannot say what was going on, which is the half a reader acts on — so the
        # kinds survive truncation even when the rows do not.
        "by_kind": {kind: sum(1 for e in events if e["kind"] == kind)
                    for kind in sorted({e["kind"] for e in events})},
        **({"events_truncated": True} if len(events) > len(shown) else {}),
        "what_it_means": _sentence(record, window, events, shown,
                                    ", ".join(searched)),
    }


def _scopes_of(record: dict) -> list:
    """Which events can reach this campaign (§9.6).

    `markets` as well as `market`, because `market` is a single grouping label that cannot say
    "ran in MY and ID" — membership is what `markets` is for, and an event in Indonesia has to
    reach a campaign that ran there. `region` too: the review's scope vocabulary is "market,
    region or global" and a region event that only matched the `region` column would miss a
    campaign that recorded its region as a market.
    """
    scopes = [("global", None)]
    for value in sorted(_market_names(record)):
        scopes.append(("market", value))
        scopes.append(("region", value))
    return scopes


def _salience(event: dict) -> tuple:
    """Which events survive the cap (§9.6).

    Ordered by date and cut at eight, "the earliest eight" becomes "eight holidays" the
    moment §9.7 seeds a market's calendar — and the flood with a stated fourteen-day delay,
    which is precisely the event §9.8 needs linked to a confounded outcome, is the one that
    disappears. A model handed eight holiday rows and the number 27 will summarise "it ran
    through the usual holidays".

    So: something somebody sat down and RECORDED before something the product shipped; an
    event with a stated impact before one without; a disruption before a date on a calendar.
    Date last, so the order is still stable and readable inside each band.
    """
    stated_impact = (event.get("delay_days") is not None
                     or event.get("budget_change_pct") is not None
                     or bool(event.get("channels_disrupted")))
    return (bool(event.get("seeded")),
            event["kind"] == "fixed_calendar",
            not stated_impact,
            event["starts_on"], event["id"])


def _why_the_empty_answer_is_not_a_claim(record: dict, window: dict) -> list:
    """Why an empty result here is about the SEARCH rather than about the market (§9.6)."""
    reasons = []
    if window["basis"] == "heuristic":
        reasons.append(
            f"its window was read out of the brief's own prose ({window['starts_on']} to "
            f"{window['ends_on']}) rather than entered, and a stray date in a competitor note "
            f"or an asset spec produces a window like this one. Enter the dates and ask again.")
    if not _market_names(record):
        reasons.append(
            "it names no market, region or markets, so only GLOBAL events could reach it — "
            "nothing market-scoped was searched at all.")
    return reasons


def _market_names(record: dict) -> set:
    values = {(record.get("market") or "").strip(), (record.get("region") or "").strip()}
    values |= {str(m).strip() for m in (record.get("markets") or [])}
    return {v for v in values if v}


def _sentence(record: dict, window: dict, events: list, shown: list, searched: str) -> str:
    where = record.get("market") or record.get("region") or "this market"
    when = f"{window['starts_on']} to {window['ends_on'] or 'open-ended'}"
    read = ("" if window["basis"] != "heuristic" else
            " The window was READ FROM THE BRIEF rather than entered, so this match inherits "
            "that reading — enter the dates to settle it.")
    if not events:
        return (f"Nothing on record was going on between {when}, searched against "
                f"{searched}. That is a real answer and only as complete as the calendar "
                f"behind it: events nobody has recorded cannot be found.{read}")
    listed = "; ".join(f"{_KIND_WORD[e['kind']]} ({e['starts_on']})" for e in shown)
    by_kind = {kind: sum(1 for e in events if e["kind"] == kind)
               for kind in sorted({e["kind"] for e in events})}
    more = ("" if len(events) == len(shown) else
            f" {len(events) - len(shown)} more are not listed; across all "
            f"{len(events)} the kinds are "
            + ", ".join(f"{kind} x{count}" for kind, count in by_kind.items()) + ".")
    # "RAN DURING" is a fact-claim, and on the heuristic path the window is a reading of prose
    # this module's own docstring calls "can be wrong". The claim has to match the evidence.
    ran = ("RAN DURING" if window["basis"] == "stated"
           else "may have run during")
    return (f"This campaign {ran} {len(events)} recorded event(s) in {where}: {listed}."
            f"{more} That is an overlap in time and nothing more — it is not a cause, and "
            f"nothing here says any of it changed the result.{read}")


# ── D119: the dates a workbook carries (§9.6, owed by §8.8) ──────────────────

# A KPI workbook's `Month`, `Launch Date` or `Start`/`End` column is skipped by §8.8 as "not a
# measurement" — right against the alternative it replaced, which made each one a provisional
# measure that accrued sightings and could graduate. But it is real information, and §9.6 is
# the field it belongs in: a campaign's period was exactly what nothing could record.
_START_KEYS = ("launch_date", "start_date", "start", "starts_on", "live_date", "go_live",
               "flight_start", "campaign_start")
_END_KEYS = ("end_date", "end", "ends_on", "finish_date", "flight_end", "campaign_end")
_PERIOD_KEYS = ("month", "period", "year_month")

# Every column name this module understands as a date. A caller needs it to tell "read and it
# added nothing" from "not understood at all" — the two look identical from the outside and
# only one of them is a skip.
DATE_COLUMNS = frozenset(_START_KEYS + _END_KEYS + _PERIOD_KEYS)


def widen(existing: Optional[dict], stated: dict) -> Optional[dict]:
    """The window a second spreadsheet row implies, folded into the first (§9.6/D119).

    A KPI workbook is one row per MONTH. Taking the first row and refusing to look again made
    a twelve-month campaign a January campaign — labelled `stated`, so nothing downstream
    doubted it — and a March event then missed it while the product said "checked, nothing
    overlapped" about two-thirds of the year.

    None when the row adds nothing, so the caller can tell "this changed the window" from
    "this row was inside it already" and report only the first.
    """
    if not existing or not existing.get("starts_on"):
        return stated
    starts = min(existing["starts_on"], stated["starts_on"])
    ends = (None if existing.get("ends_on") is None or stated.get("ends_on") is None
            else max(existing["ends_on"], stated["ends_on"]))
    if starts == existing["starts_on"] and ends == existing.get("ends_on"):
        return None
    return {"starts_on": starts, "ends_on": ends, "from": stated["from"]}


def window_from_columns(values: dict) -> Optional[dict]:
    """The window a spreadsheet row states, or None (§9.6/D119).

    Returns what it FOUND and which columns it came from, so the caller can say so. A window
    set silently from a cell changes which context events reach a campaign, and a write nobody
    is told about is the same failure as a skip nobody is told about, wearing the other face.
    """
    found: dict = {}
    for key, value in (values or {}).items():
        stem = str(key).strip().lower().replace(" ", "_").replace("-", "_")
        text = str(value or "").strip()
        if not text:
            continue
        if stem in _PERIOD_KEYS:
            span = _whole_month(text)
            if span:
                # A month names a PERIOD, not a day. Read as the 1st, a campaign that ran all
                # month looks like a one-day activation and the overlap arithmetic misses
                # every event in the other thirty days.
                found.setdefault("starts_on", (span[0], key))
                found.setdefault("ends_on", (span[1], key))
            continue
        if stem in _START_KEYS or stem in _END_KEYS:
            try:
                day = str(datetime.date.fromisoformat(text))
            except ValueError:
                continue
            found["starts_on" if stem in _START_KEYS else "ends_on"] = (day, key)
    if "starts_on" not in found:
        # An end with no start bounds nothing: the overlap needs a lower edge, and half a
        # window stored as a whole one would read as "ran until March" — a claim about when it
        # began that the spreadsheet never made.
        return None
    return {
        "starts_on": found["starts_on"][0],
        "ends_on": found["ends_on"][0] if "ends_on" in found else None,
        "from": sorted({str(k) for _, k in found.values()}),
    }


def _whole_month(text: str) -> Optional[tuple]:
    import calendar

    try:
        year, month = (int(p) for p in text.replace("/", "-").split("-")[:2])
        return (str(datetime.date(year, month, 1)),
                str(datetime.date(year, month, calendar.monthrange(year, month)[1])))
    except (ValueError, TypeError):
        return None


def withdraw(conn, *, event_id: str, why: str, withdrawn_by: str) -> dict:
    """Take an event back off the record (§9.6, §8.5's shape).

    KEPT, not deleted: "we used to think this" is an answer, and a judgment saved while the
    event was on file rested on it. Withdrawing is as consequential as recording — it removes
    a caveat from every campaign carrying it — so it says what it used to reach, for the same
    reason recording says what it now reaches.
    """
    import store

    event = store.get_context_event(conn, event_id)
    if event is None:
        raise ValueError(
            f"{event_id!r} is not an event on this record. `campaign_context` names the id of "
            f"every event it lists.")
    if not (why or "").strip():
        raise ValueError(
            "`why` is required: this event may already be carried as a caveat on several "
            "campaigns' outcomes, and removing it without saying why leaves nobody able to "
            "tell a correction from a mistake.")
    # §11.1: this checked only that the string was non-empty, so `withdrawn_by="the system"`
    # took an event off the record and recorded that a person had decided to.
    withdrawn_by = identity.person(withdrawn_by, field="withdrawn_by")

    was_reaching = campaigns_overlapping(conn, _public(event))
    store.withdraw_context_event(conn, event_id, why=why.strip(),
                                 withdrawn_by=withdrawn_by.strip())
    names = ", ".join(f"\u201c{c['title']}\u201d" for c in was_reaching[:_MAX_NAMED])
    return {
        "id": event_id, "status": "withdrawn", "basis": "stated",
        "no_longer_overlaps": len(was_reaching),
        "what_it_means": (
            f"Withdrawn by {withdrawn_by.strip()}: {why.strip()} It is kept on the record "
            f"rather than deleted, because anything judged while it was on file rested on it."
            + (f" It no longer reaches {len(was_reaching)} campaign(s): {names}."
               if was_reaching else " It was reaching no campaign.")),
    }


_MAX_NAMED = 5


def overlapping_window(conn, *, starts_on: str, ends_on: Optional[str],
                       markets: list) -> list:
    """Every event overlapping this window in these markets — UNCAPPED (§9.7).

    `for_window` is a DISPLAY answer and caps at `MAX_EVENTS_SHOWN` by salience, which sorts
    seeded rows last: right for §9.6, where a customer's recorded flood should outrank a
    shipped holiday in a list somebody reads, and exactly wrong for §9.7, where the shipped
    rows are the whole point. Reusing the display list meant eight customer records silently
    deleted the World Cup, Ramadan and Buen Fin from the check — and the fact then stated
    "8 thing(s)" as a count of what it had seen.
    """
    import store

    scopes = [("global", None)]
    for value in sorted({str(m).strip() for m in (markets or []) if str(m).strip()}):
        scopes.append(("market", value))
        scopes.append(("region", value))
    return [_public(e) for e in store.context_events(
        conn, scopes=scopes, starts_on=starts_on,
        ends_on=ends_on or str(datetime.date.max))]


def for_window(conn, *, starts_on: str, ends_on: Optional[str], markets: list) -> dict:
    """What was going on in these markets over this window (§9.6), with no campaign involved.

    The seam §9.7 and §9.8 need, exposed here rather than left for them to reach around into
    `store` for raw rows and rebuild the sentence layer — which is how two surfaces come to
    describe the same library differently.

    §9.7 checks a PROPOSED window at review time and a proposal may not be stored yet, the
    same shape `prepare_evaluation` already handles with `facts.compute(proposal_text)`. §9.8
    marks an OUTCOME confounded, and a metric's measurement period is usually far narrower
    than an always-on campaign's year — confounding every metric on a record because of a
    three-day June quake makes the caveat always-on, which is a caveat nobody reads.
    """
    import store

    first = _a_date(starts_on, "starts_on")
    last = _a_date(ends_on, "ends_on") if (ends_on or "").strip() else None
    if last and last < first:
        raise ValueError(f"`ends_on` ({last}) is before `starts_on` ({first}).")
    named = [str(m).strip() for m in (markets or []) if str(m).strip()]
    scopes = [("global", None)]
    for value in sorted(set(named)):
        scopes.append(("market", value))
        scopes.append(("region", value))
    searched = [value or "global" for _, value in scopes]
    events = [_public(e) for e in store.context_events(
        conn, scopes=scopes, starts_on=str(first),
        ends_on=str(last) if last else str(datetime.date.max))]
    if not events and not named:
        # Same rule as a campaign with no market: only global events could reach this, so an
        # empty answer is about the search rather than about the market.
        return {"status": "nothing_to_check", "basis": "computed", "events": [],
                "events_total": 0, "by_kind": {}, "searched": searched,
                "what_it_means": (
                    "Nothing overlapping was found, but no market was named — so only GLOBAL "
                    "events could reach this window and nothing market-scoped was searched.")}
    shown = sorted(events, key=_salience)[:MAX_EVENTS_SHOWN]
    by_kind = {kind: sum(1 for e in events if e["kind"] == kind)
               for kind in sorted({e["kind"] for e in events})}
    when = f"{first} to {last or 'open-ended'}"
    return {
        "status": "checked", "basis": "computed",
        "events": shown, "events_total": len(events), "by_kind": by_kind,
        "searched": searched,
        **({"events_truncated": True} if len(events) > len(shown) else {}),
        "what_it_means": (
            f"{len(events)} recorded event(s) overlap {when}, searched against "
            f"{', '.join(searched)}: "
            + ", ".join(f"{kind} x{count}" for kind, count in by_kind.items())
            + ". That is an overlap in time and nothing more — it is not a cause."
            if events else
            f"Nothing on record was going on between {when}, searched against "
            f"{', '.join(searched)}. That is a real answer and only as complete as the "
            f"calendar behind it: events nobody has recorded cannot be found."),
    }


# ── §9.7: the calendar this product ships with ───────────────────────────────

def seed(conn) -> int:
    """Put the shipped calendar on the record (§9.7).

    Through `store.insert_context_event` rather than `record`, deliberately: `record` fans out
    over the whole library to report what it reached, which is right for one event somebody
    typed and wrong for a loop — measured at roughly 25 seconds for a few hundred rows on a
    large library. Nothing here needs the fan-out, because the link is computed on read.

    Idempotent on `seed_key`, and it will not resurrect a row a customer withdrew.
    """
    import calendar_seed
    import store

    for row in calendar_seed.rows():
        store.insert_context_event(
            conn, starts_on=row["starts_on"], ends_on=row["ends_on"], scope=row["scope"],
            scope_value=row["scope_value"], kind=row["kind"], description=row["description"],
            recorded_by="shipped with this product", seeded=True, seed_key=row["seed_key"],
            certainty=row["certainty"], recurs_annually=row["recurs_annually"])
    return len(calendar_seed.rows())


def clash_check(conn, *, markets: list, text: str, starts_on=None, ends_on=None,
                window: Optional[dict] = None, from_a_record: bool = False) -> dict:
    """Does this proposed window run into anything already on the calendar? (§9.7)

    A COMPUTED fact, in `facts.py`'s shape, because "impossible to drop" is what the review
    asked for: §2.4 made `computed` unwritable from the MCP surface and §7.1 tells the model to
    treat computed facts as established rather than re-deriving them. A clash that arrives as
    prose is a sentence a model can decline to repeat; one that arrives in `computed` it has to
    carry or dispute out loud.

    The useful finding is not "does it clash" — Mexico's deck DID flag the World Cup. It is
    that the plan clashes with something it never mentions, which is what "and then never
    addressed it" describes.

    Nothing here says a clash is bad. Launching into Black Friday is the point of some
    campaigns and the ruin of others and this library cannot tell which; §9.4's lesson, that
    treating every deviation as a defect teaches it to punish improvement, applies unchanged.
    """
    window = window or _window_for_the_check(starts_on, ends_on, text)
    named = [str(m).strip() for m in (markets or []) if str(m).strip()]
    if not window["starts_on"] or not named:
        return _clash_fact("nothing_to_check", window, [], [], missing_market=not named,
                           from_a_record=from_a_record)
    import calendar_seed

    searched = ["global"] + sorted({str(m).strip() for m in named})
    # Uncapped: this is a CHECK, not a list somebody reads. See `overlapping_window`.
    events = overlapping_window(conn, starts_on=window["starts_on"],
                                ends_on=window["ends_on"], markets=named)
    covered = calendar_seed.covered_markets()
    uncovered = sorted(m for m in named if m.casefold() not in covered)
    expired = _outside_the_shipped_span(window)
    if not events:
        # Three different empty answers, and collapsing them is the clean bill this item is
        # most at risk of giving. "Nothing was happening" is a claim about the MARKET; "we
        # have no rows for Brazil" and "the shipped calendar stops at 2026" are claims about
        # the PRODUCT, and its own gap must not be reported as the customer's.
        if len(uncovered) == len(named):
            return _clash_fact("not_covered", window, [], [], searched=searched,
                               uncovered=uncovered)
        if expired:
            return _clash_fact("calendar_expired", window, [], [], searched=searched,
                               uncovered=uncovered, expired=True)
        return _clash_fact("absent", window, [], [], searched=searched, uncovered=uncovered)
    # What the plan itself already names. A plan discusses a thing in its OWN words, so the
    # handles carry aliases and match on word boundaries — "we pre-bought FIFA inventory
    # around the fixtures" addresses the World Cup, and "Eidos Media" is not Eid.
    unaddressed, unchecked = [], []
    for subject in dict.fromkeys(_subject_of(e) for e in events):
        if subject is None:
            unchecked.append(True)
        elif not _mentions(text, subject):
            unaddressed.append(subject)
    return _clash_fact("present", window, events, unaddressed, searched=searched,
                       uncovered=uncovered, expired=expired,
                       unchecked=len([u for u in unchecked if u]))


def _outside_the_shipped_span(window: dict) -> bool:
    import calendar_seed

    first, last = calendar_seed.COVERS
    return bool(window["starts_on"] > last or (window["ends_on"] or window["starts_on"]) < first)


def _mentions(text: str, subject: str) -> bool:
    """Does this plan name the thing, in any of the words a plan would use? (§9.7)

    Word boundaries and aliases, because the failure modes are not symmetric. A false positive
    lands as a neutral question — "whether that is deliberate" — which is safe. A false
    negative used to land as an affirmative claim that the plan named everything, which is the
    review's own complaint restated as an endorsement.
    """
    import calendar_seed
    import re
    import unicodedata

    said = _plain(text)
    for alias in calendar_seed.ALIASES.get(subject, (subject,)):
        if re.search(rf"(?<!\w){re.escape(_plain(alias))}(?!\w)", said):
            return True
    return False


def _plain(text: str) -> str:
    """Case- and accent-folded, because a plan writes "Dia de Muertos" as often as "Día" and
    "Ramadán" as often as "Ramadan". An accent is not a different subject."""
    import unicodedata

    stripped = unicodedata.normalize("NFD", str(text or ""))
    return "".join(c for c in stripped if not unicodedata.combining(c)).casefold()


def _window_for_the_check(starts_on, ends_on, text: str) -> dict:
    """The window to check, entered or read (§9.7).

    A pitch usually names its own flight, so reading it is worth doing — under the same bound
    and the same label as §9.6's, because a proposal deck carries competitor dates and last
    year's recaps exactly like a wrap deck does.
    """
    if (starts_on or "").strip():
        first = _a_date(starts_on, "starts_on")
        last = _a_date(ends_on, "ends_on") if (ends_on or "").strip() else None
        if last and last < first:
            raise ValueError(f"`ends_on` ({last}) is before `starts_on` ({first}).")
        return {"starts_on": str(first), "ends_on": str(last) if last else None,
                "basis": "stated"}
    return _window_from_the_brief({"detail": text or ""})


def _subject_of(event: dict) -> Optional[str]:
    """What a plan would call this, if a plan mentioned it.

    Only seeded events have one: a customer's own "New labelling rules take effect" has no
    short handle, and inventing one would produce an `unaddressed` entry naming words nobody
    used. Those events still COUNT as clashes; they are simply not checked for silence — and
    the caller has to carry that distinction rather than reading an empty `unaddressed` as
    "the plan named everything", which was an affirmative claim about a plan nobody had read.
    """
    import calendar_seed

    return calendar_seed.subject_for(event.get("seed_key") or "") or None


def _clash_fact(status: str, window: dict, events: list, unaddressed: list,
                missing_market: bool = False, searched: Optional[list] = None,
                uncovered: Optional[list] = None, expired: bool = False,
                unchecked: int = 0, from_a_record: bool = False) -> dict:
    import calendar_seed

    listed = "; ".join(
        f"{e['description'].split('.')[0]} ({e['starts_on']}"
        + ("" if e["ends_on"] in (None, e["starts_on"]) else f" to {e['ends_on']}") + ")"
        for e in events)
    when = (f"{window['starts_on']} to {window['ends_on'] or 'open-ended'}"
            if window["starts_on"] else "an unstated window")
    read = ("" if window.get("basis") != "heuristic" else
            " The window was READ FROM the proposal's own dates rather than entered, so this "
            "rests on that reading.")
    covers = " to ".join(calendar_seed.COVERS)
    if status == "nothing_to_check":
        why = ("no window was given and none could be read from the proposal"
               if not window["starts_on"] else "no market was named")
        if missing_market and window["starts_on"]:
            # Named differently depending on WHO could have named it. On the subject-record
            # path the caller cannot pass a market at all — telling them "no market was named"
            # is true of nobody and points nowhere.
            why = ("this record names no market, region or markets, so only global events "
                   "could be searched — `update_campaign` is where that is fixed"
                   if from_a_record else
                   "no market was named, so only global events could be searched")
        said = (f"This proposal was NOT checked against the calendar: {why}. That is not the "
                f"same as it clashing with nothing — \u201cthis launch clashes with nothing\u201d "
                f"is a claim, and this one has not been earned.")
    elif status == "not_covered":
        # The product's gap, said as the product's. Reported as "nothing was happening" it
        # reads as a checked market, under the heading that tells the model not to re-derive.
        said = (f"The calendar shipped with this product has no entries for "
                f"{', '.join(uncovered or [])}, so this window was checked against global "
                f"events and this customer's own records only — and neither had anything in "
                f"it. That is a gap in what was SHIPPED, not a finding about the market. "
                f"Record what was going on there and it becomes checkable.")
    elif status == "calendar_expired":
        said = (f"The calendar shipped with this product covers {covers}, and this window "
                f"({when}) falls outside it — so only this customer's own records could be "
                f"checked, and they had nothing in this window. That is a gap in what was "
                f"SHIPPED, not a finding about the market.")
    elif status == "absent":
        said = (f"Nothing on record was happening between {when}, searched against "
                f"{', '.join(searched or [])}. That is a real answer and only as complete as "
                f"the calendar behind it: events nobody has recorded cannot be found.{read}")
        if uncovered:
            said += (f" The shipped calendar has no entries for {', '.join(uncovered)}, so "
                     f"{'that market was' if len(uncovered) == 1 else 'those markets were'} "
                     f"covered only by this customer's own records.")
    else:
        said = (f"This window ({when}) runs into {len(events)} thing(s) already on the "
                f"calendar: {listed}.{read}")
        if unaddressed:
            said += (f" The proposal does not name {', '.join(unaddressed)}. Whether that is "
                     f"deliberate is a question for the people who wrote it — overlapping a "
                     f"fixed date is the point of some campaigns and the ruin of others, and "
                     f"nothing here says which this is.")
        elif unchecked and not events[0].get("seeded"):
            # NEVER "the proposal names all of them" over events that were never checked for
            # silence. A customer's own "New labelling rules take effect" has no handle a plan
            # would use, so its silence is unknowable — and asserting the plan named it is a
            # claim about a document nobody read, carrying `basis: computed`.
            said += (f" {unchecked} of these are events this customer recorded, which have no "
                     f"short name a plan would use — so whether the proposal addresses them "
                     f"is not something this check can tell you. Read them.")
        else:
            said += (" The proposal names them all by name; whether it ADDRESSES them is a "
                     "question for the reader.")
        if unchecked and unaddressed:
            said += (f" A further {unchecked} are events this customer recorded, with no "
                     f"short name to look for — read those rather than relying on this.")
        if expired:
            said += (f" Note that the shipped calendar covers {covers} and this window falls "
                     f"outside it, so anything found here came from this customer's own "
                     f"records.")
        if uncovered:
            said += (f" The shipped calendar has no entries for {', '.join(uncovered)}.")
    return {
        "code": "calendar_clash", "status": status, "basis": "computed", "layer": "body",
        "evidence": [e["description"][:120] for e in events[:3]],
        "window": window,
        "searched": searched or [],
        "markets_not_covered": uncovered or [],
        "shipped_calendar_covers": list(calendar_seed.COVERS),
        "clashes": [{"description": e["description"], "starts_on": e["starts_on"],
                     "ends_on": e["ends_on"], "seeded": e.get("seeded", False),
                     "certainty": e.get("certainty"),
                     # The hedge, ON the row. It reached the reader on §9.6's path and was
                     # dropped here, so a `seasonal` monsoon whose onset moves by weeks and a
                     # `fixed` Singles' Day arrived as two ranges differing by one unexplained
                     # word, rendered with identical confidence.
                     "certainty_note": calendar_seed.note_for(e.get("certainty"))
                     if e.get("seeded") else ""}
                    for e in events],
        "unaddressed": unaddressed,
        "not_checked_for_silence": unchecked,
        "what_it_means": said,
    }


# ── §9.8: record the overlap, never the cause ────────────────────────────────

# Which kinds confound an outcome on their own. The thing that was NOT supposed to happen: an
# earthquake, a port closure, a regulatory change, a platform going down, a macro shock.
#
# `competitor_launch` is deliberately absent. Competitor launches are continuous background in
# any real market, so a diligent customer recording them would turn every outcome confounded —
# which is the wallpaper this rule exists to avoid, arriving through the customer's own
# diligence. It confounds when somebody says it mattered.
#
# `fixed_calendar` is absent too, but the axis that actually decides it is RECURRENCE, not
# kind: every row in the shipped calendar is `fixed_calendar`, so keying on kind put a
# once-in-a-generation home World Cup in the same bucket as Black Friday — and the product then
# raised a finding that a window ran into the tournament and, once the numbers arrived, called
# the result clean. A date that comes round every year is the BASELINE a year-on-year
# comparison is made against; anything that does not is a thing that happened.
#
# A recurring date can still confound — when a person says it did (`attribute`), or when
# whoever recorded it stated an impact, which is already somebody saying it changed what
# happened. "Never quoted as clean evidence" only means anything while most evidence still is.
CONFOUNDING_KINDS = ("conflict", "natural_disaster", "regulatory_change", "supply_chain",
                     "platform_outage", "macro_shock")


def attribute(conn, *, campaign_id: str, event_id: str, note: str, stated_by: str,
              bears_on: bool = True) -> dict:
    """Somebody's account of what an event did to a campaign's numbers (§9.8).

    `stated`, always. "Sell-through was down and there was an earthquake" is not evidence the
    earthquake caused it, and a model asked to explain a disappointing number will reach for
    whatever is nearby — so the server records that the two overlapped and a PERSON records
    what they made of it, with their name on it.

    `bears_on=False` is "we looked at this and it is not why", which the first version could
    not express at all. The offer built on this asks "is X why these numbers came out as they
    did" — a question with a `no`, and the `no` had nowhere to go, which is the D84 failure
    this phase exists to fix arriving in new code. A ruled-out event is still reported as
    having run through the window, with the words that ruled it out beside it: the overlap is
    a computed fact and the ruling-out is somebody's claim, and a reader is owed both.
    """
    import store

    record = store.get_campaign(conn, campaign_id)
    if record is None:
        raise ValueError(f"{campaign_id!r} is not a record in this library.")
    if record.get("record_type") == "reference":
        # The rulebook is not precedent, and `campaigns_overlapping` already excludes it —
        # accepting one here made the two directions disagree about the same record.
        raise ValueError(
            f"{campaign_id!r} is a reference record, not a campaign. Reference material has no "
            f"outcome for an event to bear on.")
    event = store.get_context_event(conn, event_id)
    if event is None:
        raise ValueError(f"{event_id!r} is not an event on this record.")
    if not (note or "").strip():
        raise ValueError(
            "`note` is required: this is the whole content of the attribution — what you "
            "think the event did to these numbers, in your words. Without it the row says "
            "only that somebody thought something.")
    # §11.2: `identity.person` is the rule everywhere. In the one item whose premise is that a
    # model asked to explain a disappointing number will reach for whatever is nearby, a name
    # that reads as the library itself is the model laundering its own guess into the record —
    # which is why this path had its own check long before the others did, and why it is the
    # one that must not drift from them.
    stated_by = identity.person(stated_by, field="stated_by")
    window = window_of(conn, campaign_id)
    if not window["starts_on"]:
        raise ValueError(
            f"this campaign has no window, so nothing can be said to overlap it. Give it one "
            f"with `update_campaign` (starts_on / ends_on) and the events that ran through it "
            f"follow automatically.")
    # Against EVERY overlap, never `for_campaign`'s list — that one is capped at
    # `MAX_EVENTS_SHOWN` by salience, which sorts seeded rows LAST. So the only events it can
    # cut are precisely the recurring ones that need an attribution to confound, and this
    # feature's entire escape hatch closed as soon as a library got rich. The same bug shape
    # cost §9.6 and §9.7 a review round each.
    overlapping = overlapping_window(conn, starts_on=window["starts_on"],
                                     ends_on=window["ends_on"],
                                     markets=sorted(_market_names(record)))
    if event_id not in {e["id"] for e in overlapping}:
        # A note explaining a number by an event that ran in another market or another year is
        # invented evidence arriving through the one field that accepts free text.
        raise ValueError(
            f"{event_id!r} does not overlap this campaign — it did not run in its market, or "
            f"not in its window, so it cannot be what moved its numbers. `campaign_context` "
            f"lists the events that did overlap.")
    # §9.4's stamp: what the person could see when they said it. An account given before the
    # numbers arrived and one given after are two judgments, and which was which is the thing
    # a later reader most needs.
    outcome_known = bool(record.get("has_actual_metrics"))
    store.insert_attribution(conn, campaign_id=campaign_id, event_id=event_id,
                             note=note.strip(), stated_by=stated_by.strip(),
                             outcome_known=outcome_known, bears_on=bears_on)
    return {"campaign_id": campaign_id, "event_id": event_id, "basis": "stated",
            "note": note.strip(), "stated_by": stated_by.strip(),
            "outcome_known": outcome_known, "bears_on": bears_on,
            "what_it_means": (
                f"{stated_by.strip()} states that this event bears on the campaign's results: "
                f"\u201c{note.strip()}\u201d. That is their account, not a measurement — the "
                f"library records that the two overlapped and does not work out what caused "
                f"what. Its outcomes read as confounded, which means they are still evidence "
                f"and are not clean evidence."
                if bears_on else
                f"{stated_by.strip()} states that this event is NOT why the numbers came out "
                f"as they did: \u201c{note.strip()}\u201d. The overlap is still reported — it "
                f"is a fact the library computed, and their reading of it is a claim — so "
                f"anyone citing these results still sees that the two coincided, with this "
                f"beside it. What it stops is the question being asked again.")}


def attribution_history(conn, *, campaign_id: str, event_id: str) -> list:
    """Every account given about one event on one campaign, oldest first (§9.8).

    Nothing is overwritten. "This looked like the earthquake before the numbers and like our
    own pricing after" is the most interesting thing this table holds, and an update in place
    destroys it — §9.4's rule, four items over.
    """
    import store
    return store.attributions(conn, campaign_id, event_id=event_id)


def widest_window(record: dict) -> dict:
    """The outer bound of everything this record's outcomes could overlap (§9.8).

    A metric's own period always sits inside the campaign's window where both exist, so one
    fetch over the campaign window covers every row — and where a row's period falls outside
    it, the union keeps it. One query per read rather than one per month of a workbook.
    """
    starts = [record["starts_on"]] if record.get("starts_on") else []
    ends = [record.get("ends_on")] if record.get("starts_on") else []
    for metric in record.get("metrics") or []:
        if metric.get("period_start"):
            starts.append(metric["period_start"])
            ends.append(metric.get("period_end"))
    if not starts:
        read = _window_from_the_brief(record)
        return {"starts_on": read["starts_on"], "ends_on": read["ends_on"]}
    return {"starts_on": min(starts),
            "ends_on": None if any(e is None for e in ends) else max(e for e in ends)}


def confounders_for(conn, record: dict, *, metric: Optional[dict] = None,
                    events: Optional[list] = None) -> dict:
    """What ran through this campaign, and which of it confounds its outcomes (§9.8).

    Takes the RECORD rather than an id, and reads nothing back: it is called from inside
    `store.get_campaign`, which is the one place metrics load and therefore the only place a
    caveat can be attached once and reach every reader. Going back through `get_campaign` for
    the window recursed forever.

    Computed on every read, like every other link in §9.6: both halves keep arriving, and the
    ordinary order is results in October and the September earthquake recorded in November.
    """
    import store

    campaign_id = record["id"]
    # The METRIC's own period where it has one, and the campaign window where it does not —
    # carried visibly, so a reader can tell which they got. A workbook row for January is not
    # confounded by a September earthquake, and a single wrap-up figure for the whole campaign
    # genuinely is.
    period = _period_of(record, metric)
    window = period
    # FOUR states, not one boolean. `confounded: false` collapsed "we looked and nothing was
    # going on" into the same word as "this cannot be checked at all" — the exact collapse
    # `_execution_note` and `_context_note`, its two neighbours on every evidence row, were
    # each written to prevent.
    if not window["starts_on"]:
        return _nothing_to_check(
            window, "this campaign has no window, so nothing could be matched to it. That is "
                    "not the same as nothing having been going on.")
    if events is None:
        events = overlapping_window(conn, starts_on=window["starts_on"],
                                    ends_on=window["ends_on"],
                                    markets=sorted(_market_names(record)))
    else:
        # Narrowed in memory from the record-wide fetch: a January row is not confounded by a
        # September earthquake, and re-querying per row is what made a twelve-month workbook
        # twelve overlap queries.
        last = window["ends_on"] or str(datetime.date.max)
        events = [e for e in events
                  if e["starts_on"] <= last
                  and (e["ends_on"] is None or e["ends_on"] >= window["starts_on"])]
    # No early return on a missing market. `_scopes_of` always includes `("global", None)`, so
    # a global macro shock — the paradigm confounder — reaches a record that names no market,
    # and dropping it here made this surface disagree with `campaign_context` about the same
    # campaign.
    if not events and not _market_names(record):
        return _nothing_to_check(
            window, "this campaign names no market, region or markets, so only global events "
                    "could reach it — nothing market-scoped was searched at all.")
    stated = store.attributions_for(conn, campaign_id)
    confounding, alongside = [], []
    for event in events:
        attribution = stated.get(event["id"])
        # Somebody who looked at this overlap and said it is not why. Their `no` is a claim,
        # not a measurement, so it cannot make a confounded outcome clean — but it is the
        # most informed claim anyone has, so it is carried on the row and it stops the
        # question being asked again. An event that confounds BY ITSELF (an earthquake is not
        # part of a normal year) stays in the confounding list with the words that ruled it
        # out beside it; one that was only confounding BECAUSE somebody attributed it moves
        # back alongside, since the reason it was there has been withdrawn.
        ruled_out = attribution is not None and not attribution["bears_on"]
        if (attribution and not ruled_out) or _confounds_by_itself(event):
            confounding.append({**_slim(event), "attribution": attribution,
                                **({"ruled_out": True} if ruled_out else {}),
                                "why": (
                                    f"it overlapped, and {attribution['stated_by']} states it "
                                    f"is NOT why: “{attribution['note']}”. The overlap is "
                                    f"computed; that reading is theirs."
                                    if ruled_out else
                                    "somebody stated that it bears on these results"
                                    if attribution else
                                    "a stated impact was recorded against it"
                                    if _has_stated_impact(event)
                                    else f"a {event['kind'].replace('_', ' ')} is not "
                                         f"part of a normal year")})
        else:
            alongside.append({**_slim(event),
                              **({"ruled_out": True, "attribution": attribution}
                                 if ruled_out else {})})
    # Ordered before it is cut, so which five survive is not an accident of insertion: what
    # somebody attributed first, then what a person recorded, then by date.
    confounding.sort(key=lambda e: (e["attribution"] is None, e.get("seeded", False),
                                    e["starts_on"], e["id"]))
    shown = confounding[:MAX_CONFOUNDERS_SHOWN]
    return {
        "status": "confounded" if confounding else "checked_clean",
        "confounded": bool(confounding),
        "confounded_by": shown,
        "confounded_by_total": len(confounding),
        **({"confounded_by_truncated": True} if len(confounding) > len(shown) else {}),
        # A COUNT, not the events. "It ran through Black Friday" is worth knowing and is not a
        # reason to doubt the number, so it does not need to carry a paragraph on every row of
        # a twelve-month workbook.
        "ran_during": len(alongside),
        "period": window,
        # Attached either way. Carried only when confounded, an outcome that ran through three
        # recurring dates arrived with an empty `confounded_by` and no words at all — which
        # reads as "nothing was going on", the claim §9.6 spent a round refusing to make.
        "what_it_means": say_the_outcome(
            {"status": "confounded" if confounding else "checked_clean",
             "confounded": bool(confounding), "confounded_by": shown,
             "confounded_by_total": len(confounding), "ran_during": len(alongside)}),
        # `confounded_basis`, not `basis`: at the metric level a bare `basis: computed` reads
        # as "this measurement is computed", which is false — it is the OVERLAP that was
        # worked out, and each event inside keeps its own `stated`.
        "confounded_basis": "computed",
    }


def _period_of(record: dict, metric: Optional[dict]) -> dict:
    if metric and metric.get("period_start"):
        return {"starts_on": metric["period_start"], "ends_on": metric.get("period_end"),
                "basis": "metric"}
    if record.get("starts_on"):
        return {"starts_on": record["starts_on"], "ends_on": record.get("ends_on"),
                "basis": "campaign_window"}
    read = _window_from_the_brief(record)
    return {**read, "basis": "campaign_window" if read["starts_on"] else None}


def _nothing_to_check(window: dict, why: str) -> dict:
    return {"status": "nothing_to_check", "confounded": False, "confounded_by": [],
            "ran_during": 0, "period": window, "confounded_basis": "computed",
            "what_it_means": (
                f"Whether anything else was going on around this outcome could not be "
                f"checked: {why}")}


def _why_it_confounds(event: dict, attribution) -> str:
    if attribution:
        return f"{attribution['stated_by']} stated that it bears on these results"
    if _has_stated_impact(event):
        return "whoever recorded it stated an impact, which is somebody saying it changed things"
    if event["kind"] == "fixed_calendar":
        return "it is a one-off rather than a date that comes round every year"
    return f"a {event['kind'].replace('_', ' ')} is not part of a normal year"


def _confounds_by_itself(event: dict) -> bool:
    """Does this overlap cast doubt on a number all by itself? (§9.8)

    Recurrence is the axis, but only where recurrence MEANS anything — which is inside
    `fixed_calendar`, the kind that holds both Black Friday and a once-in-a-generation home
    World Cup. A calendar entry confounds when it is a one-off: a customer typing "our
    flagship was shut for a refit" means a thing that happened, not a thing that happens every
    year. Everything else is decided by its kind, because "a competitor launched" does not
    become more or less doubt-casting for happening annually.
    """
    if _has_stated_impact(event):
        return True
    if event["kind"] == "fixed_calendar":
        return not event.get("recurs_annually", False)
    return event["kind"] in CONFOUNDING_KINDS


def _has_stated_impact(event: dict) -> bool:
    """Whoever recorded it said it changed something, which is already a person saying so.

    A ZERO is not that. `_impact_sentence` in this same module renders `delay_days=0` as "no
    delay" — somebody looked and found none — so reading it here as "somebody said it changed
    things" made the two functions contradict each other, in the direction that adds a caveat
    to a number a person had just cleared.
    """
    return bool(event.get("delay_days") or event.get("budget_change_pct")
                or event.get("channels_disrupted"))


# How many confounders one outcome lists. The COUNT stays complete — "this ran through
# twenty-five recorded events" is the finding — but the rows behind it are trimmed like every
# other list here. An always-on record in a busy market genuinely overlaps dozens, and carrying
# each one in full on every metric row is how a twelve-month workbook became a 742 KB response.
MAX_CONFOUNDERS_SHOWN = 5


def _slim(event: dict) -> dict:
    """The confounder as a metric row carries it.

    `what_it_means` is deliberately absent: it is §9.6's full sentence about the event, it
    repeats the description, and it was the bulk of the payload — the `description` says what
    happened and `campaign_context` has the rest. `basis` stays, because every field on a
    context event is somebody's account (§9.6's `_public` sets `stated` on purpose) and
    stripping it put an entirely-stated event inside a block labelled `computed`.
    """
    return {k: event[k] for k in ("id", "starts_on", "ends_on", "kind", "description",
                                  "seeded", "certainty", "basis")}


def _short(text: str, limit: int = 90) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip(" ,;") + "\u2026"


def say_the_outcome(confounders: dict) -> str:
    """What a reader is told about a number that ran through something (§9.8)."""
    if confounders["status"] == "nothing_to_check":
        return confounders["what_it_means"]
    if not confounders["confounded"]:
        if confounders["ran_during"]:
            # NOT "read it as a clean result". A campaign with no overlaps was silent while one
            # with two was actively certified clean — so the more the library knew was going
            # on, the more confidently it said nothing was. That is the false clean bill,
            # inverted, and §9.6 spent a round removing exactly this shape.
            return (f"Ran through {confounders['ran_during']} recorded event(s), all dates "
                    f"that come round every year, and nobody has said any of them bears on "
                    f"this result. On the record either way.")
        return ""
    # NOT `split(".")`: "Magnitude 7.1 earthquake; …" became "Magnitude 7" and "U.S. tariffs
    # of 25% took effect" became "U". A decimal point and an abbreviation are not sentence ends.
    named = "; ".join(_short(e["description"]).rstrip(".")
                      for e in confounders["confounded_by"])
    total = confounders.get("confounded_by_total", len(confounders["confounded_by"]))
    if total > len(confounders["confounded_by"]):
        named += f" (and {total - len(confounders['confounded_by'])} more)"
    # SHORT, because it rides on every measured row. The full argument — still counts, never a
    # cause, do not quote it as clean — is `core._say_the_confounded`, said once per package;
    # repeating it on each of a workbook's twelve rows was most of a 742 KB response and told
    # a reader nothing the twelfth time it appeared.
    return (f"CONFOUNDED: ran through {named}. Still counts as a measured result, but not as "
            f"CLEAN evidence — and nothing here says the event moved the number.")


# Words that mean "not a person". Shared: §9.8 records who says an event moved a number and
# §10.3 records whose opinion a tag is, and two lists of these drift apart — the one that
# drifts being the one nobody looks at.
def _reads_as_the_product(name: str) -> bool:
    """§11.1: `identity`'s list, not a second copy. Two lists of words meaning "not a person"
    drift apart, and the one that drifts is the one nobody looks at.

    The list itself is gone from this file — it was still sitting here unused after the guard
    moved, which is the copy that would have drifted: somebody adding a word to one of them
    would have had no way to know the other existed.
    """
    return identity.reads_as_the_product(_plain(name))


def withdraw_attribution(conn, *, campaign_id: str, event_id: str, why: str,
                         withdrawn_by: str) -> dict:
    """Take back an account of what an event did (§9.8, §8.5's shape).

    Kept, not deleted: a judgment saved while the attribution stood rested on it, and "we used
    to think this" is an answer. Withdrawing may also un-confound the outcome, where the
    attribution was the only reason it was marked — which is as consequential as adding one.
    """
    import store

    standing = store.attributions_for(conn, campaign_id).get(event_id)
    if standing is None:
        raise ValueError(
            f"there is no standing account of {event_id!r} on this campaign to withdraw. "
            f"`get_campaign` lists what is on each outcome.")
    if not (why or "").strip():
        raise ValueError(
            "`why` is required: this may be the only reason an outcome is marked confounded, "
            "and removing it without saying why leaves nobody able to tell a correction from "
            "a mistake.")
    # §11.2: `identity.person` is the rule. This carried its own pair of checks after the
    # sweep that claimed to leave one implementation, which is how a sweep leaves two.
    withdrawn_by = identity.person(withdrawn_by, field="withdrawn_by")
    store.withdraw_attribution(conn, campaign_id=campaign_id, event_id=event_id,
                               why=why.strip(), withdrawn_by=withdrawn_by.strip())
    return {"campaign_id": campaign_id, "event_id": event_id, "status": "withdrawn",
            "basis": "stated",
            "what_it_means": (
                f"{withdrawn_by.strip()} withdrew the account given by "
                f"{standing['stated_by']}: {why.strip()} It is kept on the record rather than "
                f"deleted, because anything judged while it stood rested on it.")}
