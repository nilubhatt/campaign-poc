"""
§8.1: the metric registry, and typed storage for the values.

The review's evidence is two campaigns and 25 distinct keys: `reach` / `crm_reach` /
`stated_combined_influencer_reach`; `local_influencer_egr_pct` / `er_social_pct`;
`impressions` / `impressions_upper_funnel`; `total_budget_mxn` / `total_budget_usd`;
`planned_roas_july_stated` / `planned_roas_july_recomputed`. *"At fifty campaigns this is
unusable. Worse, `structured` is stored as a JSON string, not an object. The library cannot
answer 'which campaigns have sell-through data' or 'show me every ROAS on file' — the one
question a metric store exists to answer."*

Three things fall out of reading that list closely, and only the first is obvious.

**The keys collide.** A registry with aliases turns three spellings of reach into one measure
with three names on record. `raw_key` is kept beside the canonical one, because canonicalising
is a CLAIM about what somebody meant and keeping what they wrote is what makes it checkable.

**Some of what is in those keys is not a name at all.** `_mxn` is a unit and `_upper_funnel`
is a scope. Folding them into the key is what made two budgets unaddable, so they are pulled
out into their own columns rather than being spelled away.

**And the last pair is not a naming problem.** `planned_roas_july_stated` versus `_recomputed`
is a correction with nowhere to live — "the partner's figure is wrong and here is the right
one" became a key name because the schema had no room for it. A registry that only
canonicalised names would rename both halves to `roas` and lose exactly the thing that
distinguishes them. So a value row carries its own `source`, and a correction is a second
VALUE for one measure rather than a second measure.

The seed registry below is the product's, not a customer's: canonical names for the measures
this review found, and §12.1 will let a rulebook extend it. §8.2 makes an unknown key arrive
as provisional rather than being rejected or silently accepted.
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional

# The measured cutoff from §5.1: below this a suggestion is noise, and a wrong alias merges
# two measures that are not the same thing.
_SUGGESTION_CUTOFF = 0.75

# ── the shipped registry ─────────────────────────────────────────────────────
# `direction` is here because a store that does not know which way a measure runs cannot say
# whether a campaign did well — "cost per acquisition went up" is good news to anything that
# assumes higher is better.
SEED_REGISTRY = {
    "reach": ("Reach", "people", "higher_is_better",
              ("crm_reach", "stated_combined_influencer_reach", "combined_reach",
               "influencer_reach", "total_reach")),
    "impressions": ("Impressions", "count", "higher_is_better",
                    ("impressions_upper_funnel", "impressions_lower_funnel",
                     "total_impressions", "paid_impressions")),
    "engagement_rate": ("Engagement rate", "percent", "higher_is_better",
                        ("er", "er_social_pct", "local_influencer_egr_pct", "egr",
                         "engagement_pct", "er_pct")),
    "roas": ("Return on ad spend", "ratio", "higher_is_better",
             ("planned_roas", "return_on_ad_spend", "roi")),
    "ctr": ("Click-through rate", "percent", "higher_is_better",
            ("click_through_rate", "ctr_pct")),
    "cpa": ("Cost per acquisition", "currency", "lower_is_better",
            ("cost_per_acquisition", "cpa_usd", "cost_per_acq")),
    "cpm": ("Cost per mille", "currency", "lower_is_better", ("cost_per_mille",)),
    "budget": ("Budget", "currency", None,
               ("total_budget", "spend", "total_spend", "media_spend")),
    "sell_through": ("Sell-through", "percent", "higher_is_better",
                     ("sell_through_pct", "sellthrough", "sell_thru")),
    "conversions": ("Conversions", "count", "higher_is_better",
                    ("total_conversions", "orders")),
    "views": ("Views", "count", "higher_is_better", ("video_views", "total_views")),
    "traffic": ("Traffic", "count", "higher_is_better", ("sessions", "site_traffic")),
}

# Currency codes that appear IN keys, which is where the review found them. Pulled out rather
# than spelled away: `total_budget_mxn` and `total_budget_usd` are one measure in two
# currencies, and a unit inside a key means the library can neither add nor compare them.
_CURRENCIES = ("usd", "eur", "gbp", "mxn", "cop", "pen", "myr", "idr", "brl", "sgd")
# A unit spelled into the key, exactly like a currency. `footfall_uplift_pct` and
# `footfall_uplift_percent` are one measure written two ways, and leaving the suffix in the
# stem would make them two — the collision this whole item exists to stop, reintroduced by the
# thing meant to prevent it.
_UNIT_SUFFIXES = {"pct": "percent", "percent": "percent", "usd": "USD",
                  "s": None, "sec": "seconds", "seconds": "seconds", "ms": "milliseconds"}
# Scope words that appear in keys the same way. `impressions_upper_funnel` is impressions.
_SCOPES = ("upper_funnel", "lower_funnel", "mid_funnel", "organic", "paid", "social",
           "local", "combined", "total", "stated", "planned", "actual")
# Whose figure this is. The review's clearest schema failure, given its own column.
_SOURCES = {"stated": "stated", "recomputed": "recomputed", "reported": "stated",
            "corrected": "recomputed", "verified": "recomputed"}
_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december",
           "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec")


def _tokens(key: str) -> list:
    return [t for t in re.split(r"[^a-z0-9]+", (key or "").lower()) if t]


def _strip_decoration(key: str) -> tuple:
    """Pull the unit, scope and source OUT of a key, leaving the measure.

    These are the three things the review found buried in key names, and each one was doing
    real damage: a currency in the key made two budgets incomparable, a scope in the key made
    impressions unqueryable, and a source in the key turned a correction into a new measure.
    """
    unit = source = None
    scope = []
    kept = []
    for token in _tokens(key):
        if token in _CURRENCIES:
            unit = token.upper()
        elif token in _UNIT_SUFFIXES and kept:
            # Only once something is already in `kept`: a key that IS just "pct" is not a
            # measure with its unit stripped, it is a key nobody should have written.
            unit = unit or _UNIT_SUFFIXES[token]
        elif token in _SOURCES:
            source = _SOURCES[token]
        elif token in _SCOPES:
            scope.append(token)
        elif token in _MONTHS or token.isdigit():
            continue                       # a period, not a measure
        else:
            kept.append(token)
    return "_".join(kept), unit, source, "_".join(scope) or None


def _lookup(registry: dict, stem: str, raw: str) -> Optional[str]:
    """Which canonical measure this key names, if any."""
    candidates = {stem, "_".join(_tokens(raw))}
    for canonical, entry in registry.items():
        names = {canonical, *entry["aliases"]}
        if candidates & names:
            return canonical
        # A key that ENDS in an alias after decoration: `stated_combined_influencer_reach`
        # decorates down to `influencer_reach`, which is one.
        for name in names:
            if stem == name or stem.endswith("_" + name) or stem.startswith(name + "_"):
                return canonical
    return None


def _registry(conn) -> dict:
    """The registry as it stands, seeded on first use.

    Seeded into the DATABASE rather than read from the constant, because §8.2 adds provisional
    entries and §8.3 graduates them — a registry half in code and half in a table would be two
    registries, which is the drift this project has now watched five times.
    """
    import store

    rows = store.metric_registry(conn)
    if not rows:
        for canonical, (display, unit, direction, aliases) in SEED_REGISTRY.items():
            # `known`, not `expected`. A shipped name is a recognised measure with no question
            # pending against it — it is not something every brief must carry. Collapsing the
            # two would put twelve measures on every checklist the day the product is
            # installed, and a check that fires on everything is one nobody reads (§8.3).
            store.register_metric(conn, canonical=canonical, display_name=display, unit=unit,
                                  direction=direction, aliases=list(aliases),
                                  status="known")
        rows = store.metric_registry(conn)
    return rows


def canonical(conn, key: str) -> Optional[str]:
    """Which measure this key names. None when nothing in the registry claims it (§8.2)."""
    stem, _unit, _source, _scope = _strip_decoration(key)
    return _lookup(_registry(conn), stem, key)


def unit_of(conn, key: str) -> Optional[str]:
    """The unit this particular key carries, which may be more specific than the measure's.

    `budget` is a currency; `total_budget_mxn` is MXN. The registry knows the kind and the key
    knows the instance, and losing the second is what made two budgets incomparable.
    """
    _stem, unit, _source, _scope = _strip_decoration(key)
    if unit:
        return unit
    name = canonical(conn, key)
    return _registry(conn)[name]["unit"] if name else None


def describe(conn, name: str) -> Optional[dict]:
    return _registry(conn).get(name)


DECISIONS = ("same_thing", "different_measure", "ignore")


def _suggestion(conn, stem: str) -> Optional[str]:
    """The measure this unfamiliar key most resembles, or None (§8.2).

    "Looks similar to: retail_traffic_uplift_pct" is what turns the question from a chore into
    a decision somebody can make in a second. None when nothing is close, because §5.1 settled
    that a wrong suggestion is worse than none — a bad alias silently merges two measures that
    are not the same thing, and the answer is one click away.
    """
    import difflib

    names = [n for n in _registry(conn) if n != stem]
    if not names:
        return None
    close = difflib.get_close_matches(stem, names, n=1, cutoff=_SUGGESTION_CUTOFF)
    return close[0] if close else None


def resolve(conn, measure: str, *, decision: str, same_as: Optional[str] = None) -> dict:
    """Answer the one question §8.2 asks. Three answers, and they do different things.

    `same_thing` folds the key into the measure it turned out to be — RETROSPECTIVELY, because
    the values already recorded under the provisional name belong to that measure, and an
    answer that fixes the vocabulary while losing the data has fixed nothing.

    `different_measure` keeps it and stops asking. It stays `provisional`: becoming part of
    what a brief is expected to carry is §8.3's gate, and one partner's house metric must not
    turn into a standing requirement for everyone just because somebody said it was real.

    `ignore` is a decision about the VOCABULARY, not the data. The values stay — a dismissive
    click must not destroy a measurement somebody recorded.
    """
    import store

    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {list(DECISIONS)}, got {decision!r}")
    measure = canonical(conn, measure) or measure
    entry = describe(conn, measure)
    if not entry:
        raise ValueError(f"{measure!r} is not a measure on file")
    # §8.2's question is about a measure nobody has decided on. Answering it against one that
    # has graduated is a different act entirely: `different_measure` would drop it off every
    # checklist, `ignore` would set aside something a person confirmed, and `same_thing` runs
    # `merge_metric`, which DELETES the registry row — taking `confirmed_by` and `expected_in`
    # with it and breaking §8.5's "never delete" from the one direction nothing was watching.
    # The realistic route in is the offers emitted at first sight being accepted months later.
    if entry["status"] == "expected":
        raise ValueError(
            f"{measure!r} is already expected of briefs in "
            f"{', '.join(entry['expected_in']) or 'no market'}"
            + (f", confirmed by {entry['confirmed_by']}" if entry["confirmed_by"] else "")
            + ". Answering the new-measure question about it would quietly undo that. If it "
              "should no longer be expected, that is a retirement, not a naming decision.")
    if decision == "same_thing":
        if not same_as:
            raise ValueError(
                f"`same_as` is required with 'same_thing': saying {measure!r} is the same as "
                f"something means naming the something. Use 'different_measure' if it is its "
                f"own thing.")
        if not describe(conn, same_as):
            raise ValueError(f"{same_as!r} is not a measure on file")
        store.merge_metric(conn, provisional=measure, into=same_as)
        return {"measure": same_as, "status": "merged", "absorbed": measure}
    store.answer_metric(conn, measure,
                        status="ignored" if decision == "ignore" else "provisional")
    return {"measure": measure, "status": "ignored" if decision == "ignore" else "provisional",
            "answered": True}


def unanswered(conn) -> list:
    """Measures nobody has answered the question about.

    The question is asked ONCE, so something has to hold the ones nobody answered — otherwise
    "surface once" quietly becomes "surface once and lose".
    """
    return [entry for entry in _registry(conn).values()
            if entry["status"] == "provisional" and not entry["answered"]]


def record(conn, *, campaign_id: str, key: str, value, metric_type: str = "actual") -> dict:
    """Store one measurement, typed and canonicalised.

    Typed because "show me every ROAS on file" is the question a metric store exists to answer
    and a JSON string cannot be asked it. Canonicalised because three spellings of reach are
    one measure. And `source` is its own column because a correction is a second value, not a
    second measure.
    """
    import store

    stem, unit, source, scope = _strip_decoration(key)
    name = _lookup(_registry(conn), stem, key)
    asked = None
    if name is None:
        # §8.2: never rejected, never silently accepted. Rejecting loses the measurement — the
        # marketer has the number and the product refuses it, so it goes into prose where
        # nothing can compare it. Silently accepting is how two campaigns produced 25 keys.
        name = stem or "_".join(_tokens(key))
        looks_like = _suggestion(conn, name)
        store.register_metric(conn, canonical=name, display_name=key, unit=unit,
                              direction=None, aliases=[], status="provisional")
        record_row = store.get_campaign(conn, campaign_id) or {}
        asked = {
            "raw_key": key,
            "measure": name,
            # A key on its own is not a question anybody can answer — they need to know which
            # brief brought it in, which is why the review's own mock-up names both.
            "campaign_id": campaign_id,
            "campaign_title": record_row.get("title"),
            "seen_at": time.strftime("%Y-%m-%d"),
            "looks_like": looks_like,
            "next_actions": _resolution_offers(name, looks_like),
        }
    number = _as_number(value, key)
    store.record_metric_value(
        conn, campaign_id=campaign_id, metric=name, raw_key=key, value=number,
        unit=unit or (describe(conn, name) or {}).get("unit"),
        source=source or "stated", scope=scope, metric_type=metric_type)
    store.touch_metric(conn, name, campaign_id=campaign_id)
    # §8.5: a retired measure somebody has recorded again is expected again. It graduated once
    # and a person confirmed it; asking them a second time because a quarter went by is asking
    # the same question twice.
    store.revive_metric(conn, name)
    result = {"metric": name, "value": number, "unit": unit, "source": source or "stated"}
    # §8.5, on the write that changes staleness rather than on a read. A campaign reporting its
    # measures is exactly the event that can make another measure's absence a pattern, and it
    # is the moment somebody is present to be told.
    retired = retire_stale(conn)
    if retired:
        result["retired"] = retired
    # §8.3: the gate is three conditions and the third is a person, so somebody has to be
    # ASKED. Nothing surfaced eligibility, which left `measure_status` a tool a user would
    # have to know exists, think to call, and name the measure by its canonical stem to use —
    # so no measure would ever graduate and §8.4's sentence could never fire. §8.2 solved the
    # same problem the same way: the question rides on the write that creates it.
    newly = _newly_eligible(conn, name)
    if newly:
        result["newly_eligible"] = newly
    # Asked ONCE. The same key arriving in ten campaigns asks once, not ten times: a prompt on
    # every write is a prompt nobody reads, and one nobody reads is one nobody answers.
    if asked and not store.metric_was_surfaced(conn, name):
        store.mark_metric_surfaced(conn, name)
        result["new_measure"] = asked
    return result


def _resolution_offers(name: str, looks_like: Optional[str]) -> list:
    """[same thing] [different measure] [ignore], as offers that can actually be called."""
    import actions

    offers = []
    if looks_like:
        offers.append(actions.action(
            f"\u201c{name}\u201d is the same thing as \u201c{looks_like}\u201d",
            "resolve_measure",
            why="Two names for one measure is how a library ends up unable to answer "
                "\u201cshow me every one of these\u201d.",
            consent="ask", measure=name, decision="same_thing", same_as=looks_like))
    offers.append(actions.action(
        f"\u201c{name}\u201d is its own measure", "resolve_measure",
        why="It stays on file and stops asking. Whether briefs should be EXPECTED to carry "
            "it is a separate question, asked once it has been seen more widely.",
        consent="ask", measure=name, decision="different_measure"))
    offers.append(actions.action(
        f"Stop asking about \u201c{name}\u201d", "resolve_measure",
        why="The values already recorded are kept — this is a decision about the vocabulary, "
            "not about the data.",
        consent="ask", measure=name, decision="ignore"))
    return actions.trim(offers)


def _as_number(value, key: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"metric {key!r} got {value!r}, which is not a measurement")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "").rstrip("%")
    try:
        return float(text)
    except ValueError:
        raise ValueError(
            f"metric {key!r} got {value!r}, which is not a number. A measurement has to be "
            f"one — put the words in the metrics `detail`, where freeform text belongs, and "
            f"the figure here where it can be compared.")


def target_for(conn, campaign_id: str, name: str) -> Optional[dict]:
    """The number somebody said they were aiming at (D33).

    "`detail` is freeform and can never be reconciled, yet 'did we hit our number' is the
    comparison a marketer most wants." A target written into prose is a number nothing can
    compare against, which is why the refusal text existed at all. A `metric_type` rather than
    a separate table: it is the same measure, and the only difference is whether it had
    happened yet.
    """
    rows = [r for r in values_for(conn, campaign_id, name) if r["metric_type"] == "target"]
    return rows[-1] if rows else None


def against_target(conn, campaign_id: str, name: str) -> dict:
    """Did we hit our number (D33).

    Three answers, not two. `met` is None when the question cannot be asked — a campaign with
    no result has not missed anything, and a measure with no `direction` cannot be judged at
    all: under budget is not automatically good and over is not automatically bad. Reporting a
    verdict on either would be the confident unfounded claim this review is built around.
    """
    target = target_for(conn, campaign_id, name)
    actual = next((r for r in reversed(values_for(conn, campaign_id, name))
                   if r["metric_type"] == "actual"), None)
    entry = describe(conn, name) or {}
    base = {"metric": name, "target": target["value"] if target else None,
            "actual": actual["value"] if actual else None,
            "direction": entry.get("direction")}
    if not target:
        return {**base, "met": None, "code": "no_target",
                "what_it_means": f"Nobody recorded a target for {name}."}
    if not actual:
        return {**base, "met": None, "code": "no_result_yet",
                "what_it_means": (f"There is a target for {name} and no measured result yet. "
                                  f"A campaign that has not concluded has not missed its "
                                  f"number.")}
    if not entry.get("direction"):
        return {**base, "met": None, "code": "no_direction",
                "what_it_means": (f"{name} has no recorded direction, so whether "
                                  f"{actual['value']} against a target of {target['value']} "
                                  f"is good or bad is not something the library knows. Under "
                                  f"budget is not automatically good.")}
    met = (actual["value"] >= target["value"] if entry["direction"] == "higher_is_better"
           else actual["value"] <= target["value"])
    return {**base, "met": met, "code": "compared",
            "what_it_means": (f"{name} came in at {actual['value']} against a target of "
                              f"{target['value']}, and {name} is "
                              f"{entry['direction'].replace('_', ' ')}.")}


def values_for(conn, campaign_id: str, name: str) -> list:
    import store
    return store.metric_values(conn, campaign_id=campaign_id, metric=name)


def best_value(conn, campaign_id: str, name: str) -> Optional[dict]:
    """The value to weigh when a measure has more than one.

    A correction exists because somebody decided the partner's figure was wrong. Keeping both
    and saying nothing records a disagreement without resolving it, which is worse than either
    answer: the reader has to know the convention to know which number is meant.
    """
    rows = values_for(conn, campaign_id, name)
    if not rows:
        return None
    # Results only. A target is not a result, and weighing one as if it were is how a
    # library starts reporting what somebody hoped for as what happened.
    results = [r for r in rows if r["metric_type"] == "actual"]
    if not results:
        return None
    recomputed = [r for r in results if r["source"] == "recomputed"]
    return (recomputed or results)[-1]


def across_library(conn, name: str) -> list:
    """Every value of one measure, anywhere. The question the review says a store exists for."""
    import store
    return store.metric_values(conn, metric=name)


def campaigns_with(conn, name: str) -> list:
    return sorted({r["campaign_id"] for r in across_library(conn, name)})


# ── §8.3: the graduation gate ────────────────────────────────────────────────
# "Seen in N campaigns, across at least two partners or markets, and confirmed once by a
# person." Three conditions, all required. The second is the one doing the work: *"count alone
# is not enough — one partner's house metric should never quietly become a standing requirement
# for everyone."* Fifteen sightings in one market is a habit, not a standard, and promoting it
# makes every other market fail a checklist it never agreed to.
GRADUATION_CAMPAIGNS = 3
GRADUATION_MARKETS = 2
# §8.5: how many campaigns may record measurements without this one appearing before it stops
# being asked for. Campaigns, not months — see `store.campaigns_recording_metrics_since`.
RETIREMENT_AFTER = 10


def _distinct_briefs(conn, campaign_ids: list) -> int:
    """How many separate briefs these records represent (§8.3).

    Three versions of one brief are one brief. `supersedes` already says so, and counting the
    records instead let v1, v2 and v3 of a single partner's deck satisfy a gate that means
    "three different campaigns carried this" — the same mistake as counting writes, one level
    up. Each record is folded onto the root of its supersession chain.
    """
    import store

    roots = set()
    for cid in campaign_ids:
        chain = store.supersession_chain(conn, cid)
        roots.add(chain[0] if chain else cid)
    return len(roots)


def graduation(conn, name: str) -> dict:
    """Whether a measure has earned a place on the checklist, and what is still missing.

    CAMPAIGNS, not writes. `times_seen` counts every value recorded, so a campaign that logs a
    stated figure and a recomputed correction would count twice towards a gate that is supposed
    to mean "three different briefs carried this" — the same key twice in one brief is one
    brief's opinion, recorded twice.
    """
    import store

    # The raw key is what the caller has. `footfall_uplift` is an artefact of
    # `_strip_decoration`, surfaced once inside §8.2's question and possibly months ago, so a
    # gate that only answers to it is one nobody can reach — the tool refused `crm_reach` and
    # `footfall_uplift_pct`, which are the only names anybody actually types.
    name = canonical(conn, name) or name
    entry = describe(conn, name)
    if not entry:
        raise ValueError(f"{name!r} is not a measure on file")
    campaigns = _distinct_briefs(conn, campaigns_with(conn, name))
    # Folded. Three spellings of one market satisfied a gate whose whole purpose is "seen in at
    # least two markets" — one partner's house metric graduating on one market typed three
    # ways is precisely what the requirement forbids (C16).
    markets, seen = [], set()
    for raw in entry["markets"]:
        key = store.fold_market(raw)
        if key and key not in seen:
            seen.add(key)
            markets.append(raw.strip())
    base = {"measure": name, "campaigns": campaigns, "markets": len(markets),
            "seen_in": markets, "status": entry["status"],
            "confirmed_by": entry.get("confirmed_by")}

    # Status FIRST, before the counts. Asked in the other order, a measure somebody set aside
    # read "seen in 1 campaign, needs 3" — telling the user to keep recording something that
    # will be refused forever, and re-asking a question they had declined. §8.2 named that
    # failure one item ago.
    if entry["status"] == "expected":
        return {**base, "eligible": False, "code": "already_expected",
                "what_it_means": (
                    f"{name} is already expected of briefs in "
                    f"{', '.join(entry['expected_in']) or 'no market'}"
                    + (f", confirmed by {entry['confirmed_by']}." if entry["confirmed_by"]
                       else "."))}
    if entry["status"] == "ignored":
        return {**base, "eligible": False, "code": "set_aside",
                "what_it_means": (f"{name} was set aside, so it will not be asked for. "
                                  f"Recording more of it will not change that — reopen it "
                                  f"with resolve_measure if that was wrong.")}

    missing = []
    if campaigns < GRADUATION_CAMPAIGNS:
        missing.append(f"seen in {campaigns} campaign{'s' * (campaigns != 1)}, "
                       f"needs {GRADUATION_CAMPAIGNS}")
    if len(markets) < GRADUATION_MARKETS:
        missing.append(f"seen in {len(markets)} market{'s' * (len(markets) != 1)} "
                       f"({', '.join(markets) or 'none recorded'}), "
                       f"needs {GRADUATION_MARKETS} — one partner's house metric should not "
                       f"become a standing requirement for everyone")
    if missing:
        return {**base, "eligible": False, "code": "not_yet",
                "what_it_means": f"{name} is not ready to be expected of a brief: "
                                 + "; ".join(missing) + "."}
    return {**base, "eligible": True, "code": "eligible",
            "what_it_means": (f"{name} can be added to the checklist for "
                              f"{', '.join(markets)}, once a person confirms it.")}


def _newly_eligible(conn, name: str) -> Optional[dict]:
    """The graduation question, asked once, on the write that made it askable (§8.3).

    Surfaced once and never again, on §8.2's flag: a question about the same measure on every
    subsequent write is a question nobody reads. Declining it costs nothing — the measure stays
    exactly where it is, and `measure_status` still answers for anyone who comes back to it.
    """
    import actions
    import store

    gate = graduation(conn, name)
    if not gate["eligible"] or store.metric_was_offered(conn, name):
        return None
    store.mark_metric_offered(conn, name)
    return {
        "measure": name,
        "campaigns": gate["campaigns"],
        "seen_in": gate["seen_in"],
        "what_it_means": (
            f"{name} has now been reported by {gate['campaigns']} campaigns across "
            f"{', '.join(gate['seen_in'])}. It can become part of what briefs in those "
            f"markets are checked for — which is a standing requirement, so somebody has to "
            f"say so rather than the library deciding on its own."),
        # `needs`, because `confirmed_by` is the one argument the server must not supply. The
        # gate's third condition is a person, and an offer that arrived prefilled with a name
        # nobody gave would manufacture exactly the confirmation it exists to require — §6.5's
        # finding. Without saying so the offer would simply fail when accepted, which is §5.2's.
        "next_actions": actions.trim([actions.action(
            f"Expect “{name}” of briefs in {', '.join(gate['seen_in'])}",
            "graduate_measure",
            why="Briefs that do not report it will be shown as missing it. Nothing is deleted "
                "and it stops being asked for if it falls out of use.",
            consent="ask", needs=["confirmed_by — who is confirming this; ask, do not assume"],
            measure=name)]),
    }


def graduate(conn, name: str, *, confirmed_by: str) -> dict:
    """Promote a measure onto the checklist. Requires the gate AND a person (§8.3).

    Eligible is not promoted. The third condition is deliberately not automatable: a checklist
    that grows teeth on its own is one nobody agreed to, and §8.2 spent the loop's only human
    step on exactly this question.
    """
    import store

    if not (confirmed_by or "").strip():
        raise ValueError(
            "`confirmed_by` is required: the gate is 'confirmed once by a person', and a "
            "promotion with nobody's name against it is a standing requirement nobody can "
            "question later.")
    gate = graduation(conn, name)
    if not gate["eligible"]:
        # Already expected is not a failure, and saying "not ready" about something that has
        # already been promoted would send the caller off to collect data it does not need.
        # Refusing rather than re-running it is what protects `confirmed_by`: a second call
        # overwrote the name of the person who actually confirmed it, which is the one audit
        # field this whole gate exists to create.
        raise ValueError(gate["what_it_means"])
    store.graduate_metric(conn, gate["measure"], markets=gate["seen_in"],
                          confirmed_by=confirmed_by.strip())
    entry = describe(conn, gate["measure"])
    return {**entry, "graduated": True,
            "what_it_means": (
                f"Briefs in {', '.join(entry['expected_in'])} are now checked for "
                f"{gate['measure']}, on {confirmed_by.strip()}'s confirmation. Ones that do "
                f"not report it will be shown as missing it — a gap to consider, not a "
                f"verdict. It stops being asked for if it falls out of use.")}


# ── §8.4: the data grows, the prompt does not ────────────────────────────────

def expected_for(conn, *, market: Optional[str] = None, markets: Optional[list] = None) -> list:
    """The measures a brief in these markets is expected to carry.

    Read from the registry at call time. *"No prompt was edited to make that appear."* A
    measure graduating changes what every subsequent brief is checked against without anybody
    touching a string, which is the whole of §8.4: grow the data the template renders, never
    the template.

    A market is REQUIRED. With neither argument this returned the union of every market's
    checklist, so a record with no market — most of a young library — was measured against
    every expectation anybody had ever earned anywhere. That is the check that fires on
    everything, arrived at by an `if market and ...` that read as a convenience.
    """
    import store

    wanted = {store.fold_market(m) for m in (markets or ([market] if market else []))}
    wanted.discard(None)
    if not wanted:
        return []
    out = []
    for name, entry in sorted(_registry(conn).items()):
        if entry["status"] != "expected":
            continue               # retired, provisional, ignored and known are not checklists
        if not wanted & {store.fold_market(m) for m in entry["expected_in"]}:
            continue
        out.append(name)
    return out


# A checklist is for a brief that is going to run. A reference record is brand guidelines and a
# stub is a placeholder; telling either one it is missing footfall uplift is the check firing
# on everything, and §7.1/D11 was careful about which TEXT a check reads while this was not
# careful about which RECORD it runs against.
_CHECKABLE_RECORDS = ("campaign", None)


def expected_check(conn, campaign_id: str) -> dict:
    """Which expected measures this brief carries and which it does not (§8.3/§8.4).

    *"Expected for a store launch: budget, reach, footfall uplift, sell-through at 60 days.
    This brief carries none of the four."* Computed, not judged: it is a set membership test
    over the registry, identical for every user.

    A `carried` measure has a MEASURED value. A target is what somebody is aiming at, and
    counting one as carried told a concluded campaign holding nothing but targets that it
    "carries all of them" — a clean bill of health for a record with no results at all, which
    is §5.3's mistake in a new place.
    """
    import store

    record = store.get_campaign(conn, campaign_id) or {}
    if record.get("record_type") not in _CHECKABLE_RECORDS:
        return _not_a_brief(record)
    named = [m for m in store.markets_of(record) if m]
    if not named:
        return _no_market_to_check()
    expected = expected_for(conn, markets=named)
    carried, missing = [], []
    for name in expected:
        measured = [r for r in values_for(conn, campaign_id, name)
                    if r["metric_type"] == "actual"]
        (carried if measured else missing).append(name)
    return {
        "market": ", ".join(named),
        "markets": named,
        "expected": expected,
        "carried": carried,
        "missing": missing,
        "basis": "computed",
        "code": "checked" if expected else "none_expected",
        "status": "checked" if expected else "nothing_to_check",
        "what_it_means": _expected_sentence(", ".join(named), expected, carried, missing),
    }


def _unchecked(code: str, what: str) -> dict:
    """The shape §7.1 settled on, so a reader can tell an unchecked result from a clean one.

    `status: nothing_to_check` is not a pass. Every one of these returns an empty `missing`,
    and an empty `missing` beside a missing `status` reads as "this brief carries everything
    expected of it" — a clean bill of health the server has no basis for.
    """
    return {"market": None, "markets": [], "expected": [], "carried": [], "missing": [],
            "basis": "computed", "code": code, "status": "nothing_to_check",
            "what_it_means": what}


def _no_market_to_check() -> dict:
    return _unchecked("no_market", (
        "This record names no market, and a checklist belongs to one — so there is nothing to "
        "check it against. This is not a pass: add a market to see what briefs like it "
        "usually carry."))


def _not_a_brief(record: dict) -> dict:
    return _unchecked("not_a_campaign", (
        f"This is a {record.get('record_type')} record, not a campaign brief, so the "
        f"checklist does not apply to it."))


# "This brief carries none of the four" is the review's own sentence, and a server that
# renders "none of the 1" instead has written something no person would. Counted out to ten,
# and a bare numeral past that, because "none of the seventeen" is where the word stops helping.
_COUNT_WORDS = ("", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten")


def _count_word(n: int) -> str:
    return _COUNT_WORDS[n] if n < len(_COUNT_WORDS) else str(n)


def _expected_sentence(market, expected, carried, missing) -> str:
    if not expected:
        return ("Nothing is expected of a brief in this market yet. A measure joins the "
                "checklist once it has been seen across several campaigns and markets and a "
                "person has confirmed it.")
    head = f"Expected for a campaign in {market}: " if market else "Expected: "
    head += ", ".join(expected) + "."
    if not missing:
        return head + " This brief carries all of them."
    if not carried:
        if len(expected) == 1:
            return head + " This brief does not carry it."
        return head + f" This brief carries none of the {_count_word(len(expected))}."
    return head + f" This brief is missing {', '.join(missing)}."


# ── §8.5: retirement, never deletion ─────────────────────────────────────────

def retire_stale(conn) -> list:
    """Demote measures nobody has recorded in a long while. Never deletes (§8.5).

    A measure that has stopped appearing has stopped being a standing expectation, and leaving
    it on the checklist turns the report into a list of things the business no longer does. The
    row and every value under it stay: "we used to track this" is an answer somebody will need,
    and a deleted row can only say "we never did".

    Only measures that were actually SEEN can go stale, and that falls out of the count rather
    than being asserted here: a measure with no `last_seen` has nothing to count campaigns
    since, so `campaigns_that_skipped` returns 0 and it is never retired. An extra guard on
    `last_seen` here was a branch nothing could reach — a measure only reaches `expected` by
    being seen in three campaigns — and the mutation pass caught it as a condition no test
    could make false.

    Called from the WRITE that changes staleness, never from a read. It ran inside
    `expected_check` first, which meant `prepare_evaluation` quietly mutated the registry, the
    returned list was discarded so nobody was ever told the checklist had shrunk, and *who
    read* decided *what was demoted*. §2.1's rule is that partial state is never silent.
    """
    import store

    retired = []
    for name, entry in _registry(conn).items():
        if entry["status"] != "expected":
            continue
        skipped = store.campaigns_that_skipped(conn, name, since=entry["last_seen"],
                                               markets=entry["expected_in"])
        if skipped >= RETIREMENT_AFTER:
            store.retire_metric(conn, name)
            retired.append({
                "measure": name,
                "was_expected_in": entry["expected_in"],
                "campaigns_without_it": skipped,
                "what_it_means": (
                    f"{name} has not been reported by the last {skipped} campaigns in "
                    f"{', '.join(entry['expected_in']) or 'its markets'}, so briefs are no "
                    f"longer checked for it. Nothing was deleted — every value recorded "
                    f"against it is still on file, and recording it again makes it expected "
                    f"again."),
            })
    return sorted(retired, key=lambda r: r["measure"])
