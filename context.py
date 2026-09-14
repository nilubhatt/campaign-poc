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
           seed_key: Optional[str] = None) -> dict:
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
    if not (recorded_by or "").strip():
        raise ValueError(
            "`recorded_by` is required: what was going on in a market is an account a PERSON "
            "gives, and one nobody's name is against is one nobody can question later.")
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
        recorded_by=recorded_by.strip(), seeded=seeded, seed_key=seed_key)
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
        "what_it_means": said + impact,
    }


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
    if not (withdrawn_by or "").strip():
        raise ValueError(
            "`withdrawn_by` is required: taking something off the record is a PERSON's "
            "decision, and one nobody's name is against is one nobody can question later.")

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
