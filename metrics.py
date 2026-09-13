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
            store.register_metric(conn, canonical=canonical, display_name=display, unit=unit,
                                  direction=direction, aliases=list(aliases),
                                  status="expected")
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
    entry = describe(conn, measure)
    if not entry:
        raise ValueError(f"{measure!r} is not a measure on file")
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
    result = {"metric": name, "value": number, "unit": unit, "source": source or "stated"}
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
