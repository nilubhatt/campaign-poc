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
    if name is None:
        # §8.2 owns what happens next; until then an unknown key is registered provisionally
        # rather than dropped, because dropping it is how 25 keys became invisible.
        name = stem or "_".join(_tokens(key))
        store.register_metric(conn, canonical=name, display_name=key, unit=unit,
                              direction=None, aliases=[], status="provisional")
    number = _as_number(value, key)
    store.record_metric_value(
        conn, campaign_id=campaign_id, metric=name, raw_key=key, value=number,
        unit=unit or (describe(conn, name) or {}).get("unit"),
        source=source or "stated", scope=scope, metric_type=metric_type)
    store.touch_metric(conn, name, campaign_id=campaign_id)
    return {"metric": name, "value": number, "unit": unit, "source": source or "stated"}


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
    recomputed = [r for r in rows if r["source"] == "recomputed"]
    return (recomputed or rows)[-1]


def across_library(conn, name: str) -> list:
    """Every value of one measure, anywhere. The question the review says a store exists for."""
    import store
    return store.metric_values(conn, metric=name)


def campaigns_with(conn, name: str) -> list:
    return sorted({r["campaign_id"] for r in across_library(conn, name)})
