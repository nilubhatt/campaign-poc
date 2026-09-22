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

import actions
import learning

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
           "local", "combined", "total", "stated")
# A METRIC TYPE spelled into the column name, which is the ordinary shape of a KPI workbook:
# "Target Reach" beside "Actual Reach". Read as a type rather than left in the stem, because
# `target_reach` decorates to a stem ending in `_reach` and was therefore filed as a measured
# reach — a target counted as a result, which is §8.1's founding distinction, arriving through
# the door §8.8 opened. It is also the only way the sibling-column shape can be expressed at
# all, since a row carries one `metric_type` for every column in it.
_KEY_METRIC_TYPES = {"target": "target", "goal": "target", "objective": "target",
                     "actual": "actual", "achieved": "actual", "delivered": "actual",
                     "predicted": "predicted", "forecast": "predicted",
                     "projected": "predicted"}
# Whose figure this is. The review's clearest schema failure, given its own column.
_SOURCES = {"stated": "stated", "recomputed": "recomputed", "reported": "stated",
            "corrected": "recomputed", "verified": "recomputed"}
_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december",
           "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec")


# Column names that are not measurements. A KPI workbook carries identifiers and dimensions
# beside its measures — Month, Week, Store #, Campaign — and every one of them is numeric or
# short, so each became a provisional KPI measure that accrued sightings and could graduate.
# That is the "forty new keys" drift, produced by the tool built to stop it.
_NOT_A_MEASUREMENT = re.compile(
    r"^(id|ids|no|number|code|ref|reference|key|row|index|store|shop|site|branch|region|"
    r"market|country|city|campaign|brand|partner|owner|month|week|day|year|quarter|date|"
    r"period|start|end|launch|q[1-4]|h[12]|fy)(_.*)?$"
    r"|.*_(id|no|number|code|ref|date)$")


def _tokens(key: str) -> list:
    return [t for t in re.split(r"[^a-z0-9]+", (key or "").lower()) if t]


def _strip_decoration(key: str) -> tuple:
    """Pull the unit, scope and source OUT of a key, leaving the measure.

    These are the three things the review found buried in key names, and each one was doing
    real damage: a currency in the key made two budgets incomparable, a scope in the key made
    impressions unqueryable, and a source in the key turned a correction into a new measure.
    """
    unit = source = metric_type = None
    scope = []
    kept = []
    for token in _tokens(key):
        if token in _KEY_METRIC_TYPES and not metric_type:
            metric_type = _KEY_METRIC_TYPES[token]
        elif token in _CURRENCIES:
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
    return "_".join(kept), unit, source, "_".join(scope) or None, metric_type


def _lookup(registry: dict, stem: str, raw: str) -> Optional[str]:
    """Which canonical measure this key names, if any."""
    candidates = {stem, "_".join(_tokens(raw))}
    for canonical, entry in registry.items():
        names = {canonical, *entry["aliases"]}
        if candidates & names:
            return canonical
        # A key that ENDS in an alias after decoration: `stated_combined_influencer_reach`
        # decorates down to `influencer_reach`, which is one.
        #
        # NOT against a bare provisional entry, though. A provisional measure is one nobody
        # has confirmed, and treating it as an alias ROOT asserts a relationship nobody agreed
        # to: `dwell` and `queue_dwell` arriving in one workbook were silently filed as one
        # measure, with dict ordering deciding which — the silent alias merge §5.1, §8.2 and
        # §8.8 all refuse. An exact match still holds, because that is the same name.
        if entry["status"] == "provisional" and not entry["aliases"]:
            continue
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

    # No seeding here. This is read by `canonical`, `describe`, `expected_for` and everything
    # downstream of them — including §8.7's replay, whose entire claim is that it writes
    # nothing — and seeding on first read made every one of those a write on a fresh database.
    # `store.upgrade` seeds it now, which is where a shipped payload belongs.
    return store.metric_registry(conn)


def canonical(conn, key: str) -> Optional[str]:
    """Which measure this key names. None when nothing in the registry claims it (§8.2)."""
    stem, _unit, _source, _scope, _type = _strip_decoration(key)
    return _lookup(_registry(conn), stem, key)


def unit_of(conn, key: str) -> Optional[str]:
    """The unit this particular key carries, which may be more specific than the measure's.

    `budget` is a currency; `total_budget_mxn` is MXN. The registry knows the kind and the key
    knows the instance, and losing the second is what made two budgets incomparable.
    """
    _stem, unit, _source, _scope, _type = _strip_decoration(key)
    if unit:
        return unit
    name = canonical(conn, key)
    return _registry(conn)[name]["unit"] if name else None


def describe(conn, name: str) -> Optional[dict]:
    """One measure, read as one row (§13.3/D107).

    This read the WHOLE registry and then `.get(name)` — a full table read for a single-name
    lookup, and `metrics.record` did it twice. `store.metric_entry` parses through the same
    `_as_metric` the full read uses, so a narrow read and a wide one cannot disagree.
    """
    import store

    return store.metric_entry(conn, name)


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


def is_a_measurement(key: str) -> bool:
    """Whether a column name is a measure at all, rather than an identifier or a dimension."""
    stem, _u, _s, _sc, _t = _strip_decoration(key)
    return not _NOT_A_MEASUREMENT.match(stem or "_".join(_tokens(key)))


def record(conn, *, campaign_id: str, key: str, value, metric_type: str = "actual") -> dict:
    """Store one measurement, typed and canonicalised.

    Typed because "show me every ROAS on file" is the question a metric store exists to answer
    and a JSON string cannot be asked it. Canonicalised because three spellings of reach are
    one measure. And `source` is its own column because a correction is a second value, not a
    second measure.
    """
    import store

    stem, unit, source, scope, in_key_type = _strip_decoration(key)
    # A type spelled into the column wins over the row's default, and CONFLICTS with an
    # explicit one rather than being quietly overridden — "Target Reach" in a row marked
    # actual is two claims about the same number, and picking one silently is how a target
    # becomes a result.
    if in_key_type and in_key_type != metric_type:
        if metric_type not in ("actual", None) :
            raise ValueError(
                f"{key!r} says {in_key_type!r} and the row says {metric_type!r}. One number "
                f"cannot be both — split them into separate rows, or drop the word from the "
                f"column name.")
        metric_type = in_key_type
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
    # §8.5: a retired measure somebody has MEASURED again is expected again. It graduated once
    # and a person confirmed it; asking them a second time because a quarter went by is asking
    # the same question twice. A target is not a sighting of the measure being used — it is
    # somebody writing down what they hope for, and reviving a standing requirement on that is
    # the same mistake the gate made one function up.
    if metric_type == "actual":
        # The occasion in its own words (§12.4/D106), for the same reason the retirement
        # carries one: "measured again" is true of every revival.
        store.revive_metric(conn, name, why=(
            f"measured again on {campaign_id}" if campaign_id else "measured again"))
    result = {"metric": name, "value": number, "unit": unit, "source": source or "stated"}
    # §12.4/D113, the third of the three things this product calls a correction. A value whose
    # `source` is `recomputed` IS one — "the partner's figure is wrong and here is the right
    # one" — and it said so nowhere, so a reader meeting it beside a standing correction had
    # nothing to tell them apart. They carry completely different weight: this is a data fix
    # on one campaign, and a standing correction judges every future brief in a market.
    if (source or "stated") == "recomputed":
        result["what_a_correction_means_here"] = _WHAT_A_RECOMPUTED_VALUE_IS
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


def diff_columns(conn, rows: list) -> dict:
    """Classify a workbook's COLUMN VOCABULARY against the registry, writing nothing (§8.8).

    *"`bulk_import_metrics` should diff incoming columns against the registry and report what
    is new, what it thinks are aliases, and what it cannot type — before writing anything. An
    import that silently accepts 40 new keys is how the current drift started."*

    Four answers, and each one is a different decision for the reader:

      • **known** — it canonicalises to a measure already on file. Nothing to decide, and
        saying which measure is what lets somebody check the claim.
      • **looks_like** — unknown, but it resembles something. A SUGGESTION, never a merge:
        §5.1 and §8.2 both settled that a wrong alias silently merges two measures that are
        not the same, and a workbook is the worst place to get that wrong because it arrives
        forty columns at a time.
      • **new** — unknown and resembling nothing. It will be recorded provisionally and asked
        about, exactly as a single unfamiliar key is.
      • **cannot_type** — the values are not numbers. Otherwise this is discovered row by row
        as the import half-fails, which is the reading this item replaces.

    Once per COLUMN, however many rows carry it. A vocabulary reported forty times is the
    row-at-a-time view the review is describing.
    """
    registry = _registry(conn)
    seen: dict = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        structured = row.get("structured") or {}
        if not isinstance(structured, dict):
            # One malformed row killed the whole preview AND the whole import with an
            # AttributeError — not a ValueError, so over the protocol it arrived as "Error
            # executing tool" with no row index, against a docstring promising that a
            # malformed row "never crashes or blocks the rest of the batch".
            continue
        for key, value in structured.items():
            entry = seen.setdefault(key, {"column": key, "rows": 0, "bad": None})
            entry["rows"] += 1
            if entry["bad"] is None:
                try:
                    _as_number(value, key)
                except ValueError:
                    entry["bad"] = value

    # What the batch itself is about to register, so a column can be compared against the ones
    # beside it and not only against the registry. Without this, `dwell` and `queue_dwell` in
    # one workbook were both classified `new` and then SILENTLY MERGED on write — `_lookup`'s
    # suffix rule matching the second against a provisional entry the first had created
    # seconds earlier, with dict ordering deciding which. A silent alias merge is the one
    # thing §5.1, §8.2 and this item all refuse.
    batch_stems: dict = {}
    for key in seen:
        stem, _u, _s, _sc, _t = _strip_decoration(key)
        batch_stems.setdefault(stem or "_".join(_tokens(key)), key)

    known, maybe, fresh, cannot, not_measures = [], [], [], [], []
    for key, entry in seen.items():
        stem, unit, _source, _scope, in_key_type = _strip_decoration(key)
        name = _lookup(registry, stem, key)
        provisional = stem or "_".join(_tokens(key))
        if name is None and _NOT_A_MEASUREMENT.match(provisional):
            # A KPI workbook carries identifiers and dimensions beside its measures — Month,
            # Week, Store #, Quarter. They are numeric, so every one of them became a
            # provisional KPI measure that accrued sightings and could graduate: the "forty
            # new keys" drift, produced by the tool built to stop it.
            not_measures.append({
                "column": key, "rows": entry["rows"],
                "what_it_means": (
                    f"{key} looks like a dimension or an identifier rather than something "
                    f"measured, so it is not imported as a measure. If it really is a KPI, "
                    f"rename the column.")})
            continue
        if entry["bad"] is not None:
            cannot.append({"column": key, "rows": entry["rows"],
                           "example": repr(entry["bad"]),
                           "what_it_means": (
                               f"{key} carries {entry['bad']!r}, which is not a number. The "
                               f"other figures in those rows are still imported; this column "
                               f"is not. Put the words in `detail`, where freeform text "
                               f"belongs.")})
        elif name:
            known.append({"column": key, "rows": entry["rows"], "measure": name,
                          "unit": unit or registry[name]["unit"]})
        else:
            # Against the batch as well as the registry.
            sibling = next((other for other_stem, other in batch_stems.items()
                            if other != key and other_stem != provisional
                            and (provisional.endswith("_" + other_stem)
                                 or other_stem.endswith("_" + provisional))), None)
            resembles = _suggestion(conn, provisional) or (
                _strip_decoration(sibling)[0] if sibling else None)
            if resembles:
                maybe.append({"column": key, "rows": entry["rows"],
                              "measure": provisional, "looks_like": resembles,
                              "next_actions": _resolution_offers(provisional, resembles)})
            else:
                fresh.append({"column": key, "rows": entry["rows"], "measure": provisional})
    return {"known": known, "looks_like": maybe, "new": fresh, "cannot_type": cannot,
            "not_measures": not_measures}


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


def campaigns_with(conn, name: str, *, measured_only: bool = False) -> list:
    """Which campaigns have this measure on file.

    `measured_only` is what the GATE asks. §8.3 counts campaigns that CARRIED a measure, and a
    campaign that recorded the figure it was aiming at has not measured anything — so three
    target rows graduated a measure nobody had ever measured, and `expected_check` (which
    correctly counts actuals only) then reported those same three campaigns as missing it. The
    two halves of one item disagreeing about what counts.
    """
    rows = across_library(conn, name)
    if measured_only:
        rows = [r for r in rows if r["metric_type"] == "actual"]
    return sorted({r["campaign_id"] for r in rows})


# ── §8.3: the graduation gate ────────────────────────────────────────────────
# The gate itself lives in `learning`, because §8.6 puts standing corrections on the same loop
# and *"one learning mechanism for both"* is the requirement — a second implementation shaped
# like this one is two mechanisms that agree today and drift by the next item. Read here, never
# re-declared: a constant beside the one it mirrors is the copy that drifts.
GRADUATION_CAMPAIGNS = learning.GRADUATION_CAMPAIGNS
GRADUATION_MARKETS = learning.GRADUATION_MARKETS
RETIREMENT_AFTER = learning.RETIREMENT_AFTER


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
    measured_on = campaigns_with(conn, name, measured_only=True)
    gate = learning.gate(
        name=name, noun="metric",
        campaigns=learning.distinct_briefs(conn, measured_on),
        # §12.4/D103: the PARTNERS half of §8.3's gate, which is the half it is named for and
        # the half nothing could count until `campaigns.partner` existed.
        partners=store.breadth_of(conn, measured_on)["partners"],
        markets=entry["markets"], status=entry["status"],
        expected_in=entry["expected_in"], confirmed_by=entry.get("confirmed_by"))
    out = {**gate, "measure": gate["name"]}
    # §12.4/D106, ON the surface that answers "where does this measure stand". The table was
    # written by two paths and read by nothing — no tool exposed it, and a history nobody can
    # reach is the "stored and never applied" defect this project keeps hitting, one table
    # along. This is the surface it belongs on: "retired in March, revived in June, retired
    # again in October" is the answer to a different question from `status`, and `status` is
    # the one that cannot give it. Only when there IS one, so an ordinary measure that has
    # never been demoted says nothing rather than reporting an empty list as a fact.
    comings_and_goings = store.measure_history(conn, name)
    if comings_and_goings:
        out["history"] = comings_and_goings
        out["history_means"] = (
            f"{name} has been demoted or brought back {len(comings_and_goings)} time(s). A "
            f"measure that cycles is one the library keeps deciding about and nobody has "
            f"settled — worth a person's attention in a way a stable one is not. These are "
            f"the library's own observations, not anybody's decision: nothing here was "
            f"somebody's call, which is why no name is against them.")
    # D104/§8.7: the blast radius, ON the gate rather than in a separate tool nobody would
    # think to call. The person confirming needs "this shows 14 stored campaigns as missing
    # it" BEFORE they confirm; afterwards it is a surprise rather than a decision. Only when
    # there is a decision to make — computing it for an already-expected measure is work
    # nobody asked for.
    if out["eligible"]:
        import replay
        # WHAT CONFIRMING WOULD ACTUALLY KEY ON, computed the way `graduate` computes it — the
        # preview asked its own market question and so described a different graduation from
        # the one the button performs.
        out["if_confirmed"] = replay.if_graduated(
            conn, measure=name, markets=out["seen_in"],
            campaign_types=store.breadth_of(
                conn, campaigns_with(conn, name, measured_only=True))["one_type"])
    return out


def _newly_eligible(conn, name: str) -> Optional[dict]:
    """The graduation question, asked once, on the write that made it askable (§8.3).

    Surfaced once and never again, on §8.2's flag: a question about the same measure on every
    subsequent write is a question nobody reads. Declining it costs nothing — the measure stays
    exactly where it is, and `measure_status` still answers for anyone who comes back to it.
    """
    import actions
    import store

    # The offered check FIRST, so the blast radius — a full library scan — is computed only
    # when it is about to be shown, rather than on every write touching an eligible measure.
    if store.metric_was_offered(conn, name):
        return None
    gate = graduation(conn, name)
    if not gate["eligible"]:
        return None
    store.mark_metric_offered(conn, name)
    return {
        "measure": name,
        "campaigns": gate["campaigns"],
        "seen_in": gate["seen_in"],
        # D104, on the surface where the decision is actually made. It was on `graduation()`,
        # which the model reaches through `measure_status` — the tool §8.4 described as one a
        # user would have to know exists and think to call. The offer riding on the write is
        # the route §8.4 built after finding the gate unreachable, and the number belongs on it.
        "if_confirmed": gate.get("if_confirmed"),
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

    confirmed_by = learning.require_a_person(confirmed_by)
    gate = graduation(conn, name)
    if not gate["eligible"]:
        # Already expected is not a failure, and saying "not ready" about something that has
        # already been promoted would send the caller off to collect data it does not need.
        # Refusing rather than re-running it is what protects `confirmed_by`: a second call
        # overwrote the name of the person who actually confirmed it, which is the one audit
        # field this whole gate exists to create.
        raise ValueError(gate["what_it_means"])
    # §12.4/D102: the campaign TYPE this measure earned, when every campaign carrying it was
    # the same kind of campaign. That makes the checklist key "store launch" rather than
    # "Peru" — the review's own sentence — so it reaches store launches in every market and
    # stops reaching seeding briefs that never had footfall to uplift.
    # The same question the preview asked, and it has to be the same answer: `if_confirmed`
    # told somebody what confirming would do, and this is the confirming.
    on_type = store.breadth_of(
        conn, campaigns_with(conn, gate["measure"], measured_only=True))["one_type"]
    store.graduate_metric(conn, gate["measure"], markets=gate["seen_in"],
                          confirmed_by=confirmed_by, campaign_types=on_type)
    # §11.1/§13.5: the account beside the name, on the write that puts a measure in front of
    # every future brief in its markets. The correction twin has recorded this since §11.1 and
    # this did not — so a graduated MEASURE kept only `confirmed_by`, a free-text name, while
    # a graduated RULE kept the account the call was made from. That is not a table-shaped
    # difference between the two paths, which is what D114's row claims they are reduced to;
    # it is a decision about whether a write is audited, made one way here and the other way
    # there. Review found it while checking that claim.
    store.record_authorship(conn, subject_kind="metric", subject_key=gate["measure"],
                            on_behalf_of=confirmed_by)
    entry = describe(conn, gate["measure"])
    return {**entry, "graduated": True,
            # §8.7: the moment the replay becomes non-empty is the moment to point at it.
            "next_actions": actions.after_graduation(what=gate["measure"],
                                                     markets=entry["expected_in"]),
            "what_it_means": (
                (f"Every {', '.join(on_type)} brief is now checked for {gate['measure']}, in "
                 f"any market, on {confirmed_by.strip()}'s confirmation — every campaign that "
                 f"carried it was one, so that is what makes it the right question rather "
                 f"than where it happened. Briefs of other kinds are not checked for it."
                 if on_type else
                 f"Briefs in {', '.join(entry['expected_in'])} are now checked for "
                 f"{gate['measure']}, on {confirmed_by.strip()}'s confirmation.")
                + " Ones that do not report it will be shown as missing it — a gap to "
                  "consider, not a verdict. It stops being asked for if it falls out of use.")}


# ── §8.4: the data grows, the prompt does not ────────────────────────────────

def expected_for(conn, *, market: Optional[str] = None, markets: Optional[list] = None,
                 campaign_type: Optional[str] = None) -> list:
    """The measures a brief is expected to carry, keyed on campaign type or on market.

    Read from the registry at call time. *"No prompt was edited to make that appear."* A
    measure graduating changes what every subsequent brief is checked against without anybody
    touching a string, which is the whole of §8.4: grow the data the template renders, never
    the template.

    **§12.4/D102 — which key.** The review's own sentence is *"expected for a store LAUNCH:
    budget, reach, footfall uplift, sell-through at 60 days"*, and market was standing in for
    the word "launch". So a measure learned entirely from store launches in Peru was expected
    of a seeding brief in Peru, which never had footfall to uplift, and was expected of no
    store launch in Mexico, which had exactly the same reason to report it. Both halves are
    the same mistake: the checklist was keyed on the wrong thing.

    A measure whose evidence was type-coherent is keyed on its TYPE and reaches every market,
    because what makes footfall uplift the right question is that it is a store launch, not
    that it is in Peru. A measure whose evidence spanned types keeps the market key — that is
    every row written before D102 and every measure genuinely about a place.

    A market is REQUIRED for the market-keyed ones. With neither argument this returned the
    union of every market's checklist, so a record with no market — most of a young library —
    was measured against every expectation anybody had ever earned anywhere. That is the check
    that fires on everything, arrived at by an `if market and ...` that read as a convenience.
    A type-keyed measure is not subject to that: it has a key of its own, and an untyped brief
    simply does not match it.
    """
    import store

    wanted = {store.fold_market(m) for m in (markets or ([market] if market else []))}
    wanted.discard(None)
    return sorted(name for name, entry in _registry(conn).items()
                  # retired, provisional, ignored and known are not checklists
                  if entry["status"] == "expected"
                  and keyed_on_reaches(expected_in=entry["expected_in"],
                                       expected_for_types=entry.get("expected_for_types"),
                                       markets=wanted, campaign_type=campaign_type))


def keyed_on_reaches(*, expected_in, expected_for_types, markets, campaign_type) -> bool:
    """Whether a checklist entry scoped this way reaches a brief like this (§12.4/D102).

    The two keys a checklist can hang on, and which one WINS. A measure that graduated when
    every campaign carrying it was the same kind of campaign is keyed on the TYPE — "that
    makes the checklist key 'store launch' rather than 'Peru'", the review's own sentence — so
    it reaches store launches in every market, including markets it has never been seen in,
    and stops reaching other kinds of brief in the markets it has.

    A FUNCTION, because the graduation preview needs the same answer about a measure that has
    not graduated yet, and asked it with its own market test instead. So the sentence somebody
    reads at the moment they confirm named the one record the graduation guarantees it will
    never check, and said "this would change nothing" about a library holding a brief it was
    about to flag. §12.4's own sentence, inverted, on both halves at once.

    No `everywhere` in the market branch, and that is the distinction rather than an omission:
    a measure is a thing this library WATCHED recur, so it is expected where the evidence put
    it, and there is no such thing as a declared measure for a customer to scope to all
    markets. The day there is, this is the line that has to learn about it.
    """
    import store

    by_type = {store.fold_campaign_type(t) for t in expected_for_types or []} - {None}
    if by_type:
        return store.fold_campaign_type(campaign_type) in by_type
    return learning.in_force(expected_in, markets, everywhere=False)


def measured(rows) -> list:
    """The RESULTS among a record's metric rows — what happened, not what was aimed at.

    Same distinction as `carries`, asked of a list that is already in hand. Written out at
    each surface that needed it, which is how "this record has results" came to be answered
    from four places.
    """
    return [row for row in rows or [] if row["metric_type"] == "actual"]


def carries(conn, campaign_id: str, name: str) -> bool:
    """Whether this record holds a MEASURED value for this measure (§8.1).

    A target is what somebody is aiming at, not what happened, and counting one as carried
    told a concluded campaign holding nothing but targets that it "carries all of them" — a
    clean bill of health for a record with no results at all. That distinction is this
    module's founding one, and the test for it was written out in four places: here, the
    replay's gap list, the graduation preview, and the evidence `missing` weighs. One of them
    drifting is a record that has results on one surface and none on another.
    """
    return any(row["metric_type"] == "actual" for row in values_for(conn, campaign_id, name))


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
    # The three reasons a record has no checklist are `learning`'s, because §8.6 needs the same
    # three and writing them twice is how a reference record came to be told it was missing
    # something in one place and not the other.
    import store

    named, refusal = learning.subject_markets(conn, campaign_id)
    if refusal:
        return {**refusal, "expected": [], "carried": [], "missing": []}
    # §12.4/D102: the brief's own kind, which is the other key a checklist can hang on.
    record = store.get_campaign(conn, campaign_id) or {}
    kind = record.get("campaign_type")
    expected = expected_for(conn, markets=named, campaign_type=kind)
    carried, missing = [], []
    for name in expected:
        (carried if carries(conn, campaign_id, name) else missing).append(name)
    return {
        "market": ", ".join(named),
        "markets": named,
        "campaign_type": kind,
        "expected": expected,
        "carried": carried,
        "missing": missing,
        "basis": "computed",
        "code": "checked" if expected else "none_expected",
        "status": "checked" if expected else "nothing_to_check",
        "what_it_means": _expected_sentence(", ".join(named), expected, carried, missing,
                                            campaign_type=kind),
    }


# "This brief carries none of the four" is the review's own sentence, and a server that
# renders "none of the 1" instead has written something no person would. Counted out to ten,
# and a bare numeral past that, because "none of the seventeen" is where the word stops helping.
_COUNT_WORDS = ("", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten")


def _count_word(n: int) -> str:
    return _COUNT_WORDS[n] if n < len(_COUNT_WORDS) else str(n)


def _expected_sentence(market, expected, carried, missing, campaign_type=None) -> str:
    if not expected:
        return ("Nothing is expected of a brief in this market yet. A measure joins the "
                "checklist once it has been seen across several campaigns and markets and a "
                "person has confirmed it.")
    # §12.4/D102: the review's own sentence is "Expected for a store LAUNCH", and this said
    # "for a campaign in Peru" because market was the only key a checklist could hang on.
    # Naming the type where there is one is not decoration: it is what tells a reader why the
    # measure is being asked for, and therefore whether the answer "this brief is missing
    # footfall uplift" is a real gap or a checklist pointed at the wrong kind of campaign.
    if campaign_type and str(campaign_type).strip():
        head = f"Expected for a {str(campaign_type).strip()}: "
    else:
        head = f"Expected for a campaign in {market}: " if market else "Expected: "
    head += ", ".join(expected) + "."
    if not missing:
        return head + " This brief carries all of them."
    if not carried:
        if len(expected) == 1:
            return head + " This brief does not carry it."
        return head + f" This brief carries none of the {_count_word(len(expected))}."
    return head + f" This brief is missing {', '.join(missing)}."


# §12.4/D113. The stored vocabulary cannot be renamed — every saved row and every tool
# argument is written in it — so each surface says which kind it means. `corrections._public`
# and `core.diff_campaigns` carry the other two.
_WHAT_A_RECOMPUTED_VALUE_IS = (
    "`source: recomputed` is a CORRECTED NUMBER: somebody checked the figure that was "
    "reported and this is what it actually was. Both values stay on file, because which "
    "figure a judgment was made against is part of that judgment. It is not a STANDING "
    "correction — a rule this library watched recur until somebody confirmed it, which then "
    "judges every brief in its markets — and it is not a finding a later deck answered. This "
    "product calls all three corrections and they carry completely different weight.")


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
    # §13.3/D107: only the measures ON a checklist. This read every row and skipped all but
    # the `expected` ones, on every metric write.
    for name, entry in store.expected_metrics(conn).items():
        skipped = store.campaigns_that_skipped(conn, name, since=entry["last_seen"],
                                               markets=entry["expected_in"])
        if skipped >= RETIREMENT_AFTER:
            # §12.4/D106: the OBSERVATION, not the default. This computed the specific
            # sentence four lines down and passed nothing, so every history row in real use
            # read "no longer reported by recent campaigns" — a sentence that is true of every
            # retirement and therefore says nothing about any of them. The whole of D106 is
            # that a reader years later can tell one occasion from another, and "skipped by 5
            # campaigns in Peru, Mexico" is the part that does that.
            store.retire_metric(conn, name, why=(
                f"skipped by the last {skipped} campaign(s) in "
                f"{', '.join(entry['expected_in']) or 'its markets'}"))
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
