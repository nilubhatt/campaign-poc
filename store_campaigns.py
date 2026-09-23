"""
Campaign records — storage only.

Lifted out of `store` unchanged: the campaigns table, its listing and filtering, the markets
and tags a record carries, and deleting one along with everything keyed to it.

`store` re-exports every name here, so `store.get_campaign`, `store.list_campaigns` and the
rest keep working for every caller and test that already says them. `store` is imported INSIDE
the functions that need its base — `_columns`, `_now`, the id helper — because `store` imports
this module and a module-level import back would be a cycle.

Two things to know before moving anything else out of `store`. A re-export is a separate
binding, so `monkeypatch.setattr(store, "f", ...)` does NOT change what a function in here
calls; four names are patched that way across the suite — `text_on_file`, `citations`,
`metric_registry`, `_facts_algorithm_stamp` — and none is in this module, which is why this
block moved first. And a local variable may share a module-level name: rewriting one of those
into `store.x` assigns over the real thing on the module.
"""
from __future__ import annotations

from typing import Optional
from typing import Union
import config
import enums
import identity
import json
import re
import time
import vectorstore

# ── campaigns ────────────────────────────────────────────────────────────────

# Tuples, not sets: these are shown to a person, and set iteration order is arbitrary, so
# the same error could list the options differently on two runs.
VALID_RECORD_TYPES = ("campaign", "reference", "stub")
# §12.4/D38: `cancelled` and `paused` were REFUSED, with an error explaining there was no
# home for them — and a brief that was cancelled is exactly the record a library about what
# works most needs to keep. "We stopped this one" is an outcome, and it is one of the more
# useful things this library can say about a kind of brief.
#
# `on_hold` is NOT a fourth value: it is the same fact as `paused` in somebody else's words,
# and two states nothing distinguishes are two checklists, two cells and two gap reports for
# one situation. It is a synonym.
VALID_STATUSES = ("proposed", "in_flight", "concluded", "cancelled", "paused")
VALID_TAG_SOURCES = ("verified", "stated")
_VALID_RECORD_TYPES = VALID_RECORD_TYPES          # older internal names, kept
_VALID_STATUSES = VALID_STATUSES
_VALID_TAG_SOURCES = VALID_TAG_SOURCES


def _normalise_record_type(value):
    return enums.normalise(value, field="record_type", valid=VALID_RECORD_TYPES,
                           synonyms=enums.RECORD_TYPE_SYNONYMS, allow_none=False)


def _declared(key: str) -> dict:
    """The customer's spellings for one vocabulary, as a synonym table (§12.2/D36).

    Built from the rulebook rather than kept here, so there is one place a spelling is
    declared. `enums.normalise` already takes `synonyms=` as a parameter, which is why this
    reaches the whole product by changing two wrappers — the note in `enums` said so.
    """
    import rulebook

    out = {}
    for canonical, entry in rulebook.vocabulary(key).items():
        for spelling in entry["also"]:
            # `enums.canonical_shape`, because `enums.normalise` looks the value up in THAT
            # shape. Keyed by the rulebook's own fold (spaces) every multi-word spelling a
            # customer declared sat in the table unreachable: "out the door" was declared and
            # "out_the_door" was looked up. Two folds for one lookup, which is this codebase's
            # signature defect arriving inside the fix for it.
            out[enums.canonical_shape(spelling)] = canonical
    return out


def _normalise_status(value):
    # D36: a stage name is CUSTOMER vocabulary. An agency that says `shipped` or `in_the_wild`
    # is describing their own process, and the product refusing it is the product telling them
    # how to talk about their work. `verified` and `actual` stay product-owned, because those
    # are this library's claims about evidence rather than words for a stage — the rulebook
    # loader refuses an overlay that tries to declare them.
    return enums.normalise(value, field="status", valid=VALID_STATUSES,
                           synonyms={**enums.STATUS_SYNONYMS, **_declared("statuses")})


def _folded_name(name: str) -> str:
    """One spelling of a name, for comparison only — `people._folded` without the import.

    Kept here rather than calling `people` because `people` reads this module for every table
    it scans, and this is two lines of the same rule: NFC then casefold then collapse, so
    "R.  Vega" and "r. vega" are one person on both sides of the comparison.
    """
    import unicodedata

    return " ".join(unicodedata.normalize("NFC", name or "").split()).casefold()


def attributions_on_file(tags) -> set:
    """Every (tag value, `said_by`) pair already stored on a record, folded.

    What makes a re-send distinguishable from a new claim. The VALUE is part of the key on
    purpose: a legacy `liked` held by "the client" is not a licence to file a new
    `performed_well` under the same non-name — that is the claim the rule is about, arriving
    on a record that happens to carry an old one. Read from the STORED tags, so a caller
    cannot grandfather anything by asserting it was already there.
    """
    return {(t["value"].strip().lower(), _folded_name(t["said_by"])) for t in (tags or [])
            if isinstance(t, dict) and (t.get("said_by") or "").strip()
            and isinstance(t.get("value"), str)}


def _short(text: str, limit: int = 60) -> str:
    """A value, cut to something a refusal can carry.

    A tag value is caller-supplied and unbounded: interpolating one whole turned a 100,000-
    character tag into a 100,198-character exception. An error nobody can read is a different
    way of not saying what went wrong.
    """
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "\u2026"


def normalize_tags(tags, *, has_actual_metrics: bool = False,
                   already_said_by=()) -> list[dict]:
    """
    Normalize tags to [{"value": str, "source": "verified"|"stated"}, ...]. A plain string
    is accepted for ergonomics (LLM-first, conversational intake — §6.9) and defaults to
    'stated': the conservative assumption that a tag is someone's claim, not measured
    evidence, unless explicitly marked 'verified'. This distinction matters specifically for
    performance-outcome tags (performed_well, underperformed, ...) — without it, a stated
    impression reads as evidence and gets weighted as if it were (user feedback: four of
    five performance tags in the real data were impressions, one had real numbers behind
    it). tags overlap and a campaign commonly carries several at once (e.g. a creative-
    reaction tag plus a performance tag) — this is a list, not a single value.

    'verified' REQUIRES has_actual_metrics=True (i.e. the campaign already has at least one
    metric_type='actual' record) — reviewed and found the first version of this let anyone
    claim 'verified' with nothing behind it, defeating the whole point. A brand-new campaign
    can never satisfy this (chicken-and-egg: no metrics exist yet at insert time), so
    'verified' can only be set via update_campaign, after add_metrics/bulk_import_metrics.

    Duplicate values (case/whitespace-insensitive) are deduped, preferring 'verified' over
    'stated' if both appear for the same value.

    §10.3/§11: a tag may also carry `said_by` and `said_at` — WHO holds this opinion and when
    they said it. A client is several people with different authority and sometimes different
    opinions, and a tag with no author cannot be weighed against a contradicting one, traced
    back to whoever set it, or removed when that person leaves. Optional, because most tags
    predate the field and a missing author is an honest "nobody recorded one" rather than a
    reason to refuse the write.

    But a `said_by` that is PRESENT has to name a person (§11.2). This is the door tags come
    through, and it had no such check: `_keep_the_view` refused `said_by="the team"` from the
    append-only record and this function wrote it onto the campaign anyway, where
    `get_campaign` hands it to every reader. The surface people read said the team held an
    opinion; the record built to be the authority on who holds which opinion had never heard
    of it. One rule, in the one place both paths pass through.
    """
    if tags is None:
        return []
    # One tag, given as itself. "A bare string where a list was required" was one of the
    # three rejections idea A was written about, and only the FILTER side had been fixed —
    # writing `tags="liked"` still failed with "Input should be a valid list".
    if isinstance(tags, (str, dict)):
        tags = [tags]
    if not isinstance(tags, (list, tuple)):
        raise ValueError(f"tags must be a list of strings or {{value, source}} objects, "
                         f"got {type(tags).__name__}")
    out = []
    for t in tags:
        if isinstance(t, str):
            value, source, said_by, said_at = t.strip(), "stated", None, None
            unresolved = None
        elif isinstance(t, dict):
            raw_value = t.get("value")
            if not isinstance(raw_value, str) or not raw_value.strip():
                raise ValueError(f"tag object must have a non-empty string 'value', got {t!r}")
            value = raw_value.strip()
            source = enums.normalise(t.get("source", "stated"), field="tag 'source'",
                                     valid=VALID_TAG_SOURCES,
                                     synonyms=enums.TAG_SOURCE_SYNONYMS,
                                     allow_none=False)
            said_by = (t.get("said_by") or "").strip() or None
            # Carried through rather than dropped: it is what is left of an attribution this
            # library should never have accepted, and the next round-trip would lose it.
            unresolved = (t.get("said_by_unresolved") or "").strip() or None
            if said_by:
                try:
                    said_by = identity.person(said_by, field="said_by")
                except ValueError as bad:
                    if (value.lower(), _folded_name(said_by)) in already_said_by:
                        # ALREADY ON FILE. Tags replace, so every caller re-sends the ones it
                        # is keeping — `feedback.record` re-sends the whole stored list on
                        # every menu answer — and applying the rule blind made one legacy
                        # `said_by: "the client"` (v0.2.0 wrote them, past a name list that
                        # did not yet have the placeholders on it) permanently unwritable:
                        # every answer refused, naming a tag the caller never sent, with no
                        # remedy in the message. A guard that bricks a record to enforce a
                        # naming rule has cost more than the record it objected to.
                        #
                        # Grandfathered is not endorsed. The opinion survives — losing it is
                        # the §11.5 loss — but it stops being recorded as HELD by somebody,
                        # because nothing can find that person, erase them on request, or
                        # weigh their view against another. The text stays where a person can
                        # still act on it.
                        said_by, unresolved = None, unresolved or said_by
                    else:
                        # Named, because `normalize_tags` refuses whole writes and a caller
                        # sending five tags cannot resend four of them without knowing which
                        # one this was. The same shape as the `verified` refusal below.
                        raise ValueError(f"tag {_short(raw_value.strip())!r}: {bad}") from None
            said_at = (t.get("said_at") or "").strip() or None
            if unresolved and not said_by:
                said_at = None   # a time nobody is attached to says nothing
        else:
            raise ValueError(f"tag must be a string or {{value, source}} object, got {t!r}")
        if not value:
            raise ValueError(f"tag value cannot be empty or whitespace-only, got {t!r}")
        # The product's OWN vocabulary, canonicalised. `REACTION_AXES` is lowercase and tag
        # values were never folded, so `"Liked"` — the likelier spelling for a model writing
        # prose — was stored on the campaign and dropped by `_keep_the_view` as not a
        # reaction: the campaign saying R. Vega liked it and the append-only record never
        # having heard of it, which then costs §11.5's `contested_precedent` finding.
        # Only these seven words: a marketer's "Back-to-School" is their words, and
        # lowercasing every tag to fix seven of them would rewrite the library.
        if value.lower() in REACTION_AXES:
            value = value.lower()
        if source == "verified" and not has_actual_metrics:
            raise ValueError(
                f"tag {value!r} cannot be marked source='verified' — this campaign has no "
                f"metric_type='actual' record on file yet. Add real metrics first "
                f"(add_metrics/bulk_import_metrics), then update_campaign to mark it verified."
            )
        out.append({"value": value, "source": source,
                    **({"said_by": said_by} if said_by else {}),
                    **({"said_by_unresolved": unresolved}
                       if unresolved and not said_by else {}),
                    **({"said_at": said_at} if said_at else {})})

    deduped: dict[tuple, dict] = {}
    for entry in out:
        # WHO is part of the key. It used to be the value alone, so a second person recording
        # `liked` replaced the first — §11.5's loss arriving through the de-duplication
        # instead of the replace, and invisible because the raw-list walk still filed both
        # reactions. Agreement is corroboration: two people liking something is the evidence,
        # not a duplicate of it.
        key = (entry["value"].lower(), _folded_name(entry.get("said_by") or ""))
        # Later wins on a tie, so re-recording an opinion updates who holds it rather than
        # keeping the first person's name on somebody else's answer.
        if (key not in deduped
                or (deduped[key]["source"] != "verified" and entry["source"] == "verified")
                or (deduped[key]["source"] == entry["source"])):
            deduped[key] = entry
    return list(deduped.values())


def fold_vocabulary(key: str, value):
    """The declared spelling of a value, for COMPARISON only (D72/D73).

    **Resolved when two values are compared, never written to the row.** D72 asks for
    "canonical market names from the rulebook, so `latam` is a SYNONYM rather than a fold",
    and a synonym is a statement about when two words mean the same thing — not an
    instruction to rewrite one into the other.

    The first version canonicalised on write, and it made a declaration actively harmful: rows
    written before it kept the old spelling, rows after got the new one, and a query in either
    found half of them. Two spellings that had at least folded to one cell became two cells,
    each with its own thin evidence — the precise harm the feature exists to remove, inflicted
    by the feature, on the libraries that already had data. Nothing migrated and nothing said
    so.

    Resolving on comparison has none of that. Every row ever written joins its cell the moment
    the declaration lands, rows added tomorrow from a spreadsheet that still says the old word
    are found too, and withdrawing the declaration puts everything back. Nothing a customer
    exported yesterday disagrees with what the product holds today.
    """
    import rulebook

    if not isinstance(value, str) or not value.strip():
        return (value or "").strip().lower() if isinstance(value, str) else value
    try:
        declared = rulebook.canonical(key, value)
    except ValueError:
        declared = None
    return (declared or value).strip().lower()


def normalize_markets(markets) -> list[str]:
    """Normalize markets to a deduped list of stripped strings. Mirrors tags' list-not-a-
    single-value philosophy but with no provenance concept - this is a plain membership list
    (every country/sub-market an activation touched), matched by filter_campaign_ids'
    `markets` param via case-insensitive membership, not exact-match like region/market."""
    if markets is None:
        return []
    if not isinstance(markets, list):
        raise ValueError(f"markets must be a list, got {markets!r}")
    out = []
    seen: set[str] = set()
    for m in markets:
        if not isinstance(m, str) or not m.strip():
            raise ValueError(f"each market must be a non-empty string, got {m!r}")
        value = m.strip()
        # D72: two spellings of one market dedupe to a single entry, WITHOUT the stored
        # spelling changing — the caller's own words go in the row, and the declaration says
        # which of them mean the same thing.
        key = fold_vocabulary("markets", value)
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _parse_stored_markets(raw: Optional[str]) -> list[str]:
    return json.loads(raw) if raw else []


def _parse_stored_tags(raw: Optional[str]) -> list[dict]:
    """Parse a campaign's stored tags JSON, tolerating legacy plain-string entries (from
    before tags carried provenance) — reviewed and found the strict-dict-only reader crashed
    outright on any pre-existing row with the old shape."""
    entries = json.loads(raw) if raw else []
    out = []
    for e in entries:
        if isinstance(e, str):
            out.append({"value": e, "source": "stated"})
        elif isinstance(e, dict):
            # Carried through, not rebuilt from two keys. Reconstructing each tag as exactly
            # {value, source} silently dropped §10.3's `said_by`/`said_at` on the way back
            # out — so the author was demanded at the menu, written to the row, and invisible
            # to every reader, while the response said it had been kept.
            out.append({**e, "value": e.get("value", ""),
                        "source": e.get("source", "stated")})
    return out


def _parse_tag_query(tags) -> list[tuple[str, Optional[str]]]:
    """
    Parse a tags filter into [(value_lower, source_or_None), ...]. Each entry is a plain
    string (match that value regardless of source) or a {value, source} object (match that
    value AND require that specific source) — mirrors the storage shape so a query can mix
    precision per tag: tags=["liked", {"value": "underperformed", "source": "verified"}]
    means "liked, any evidence, AND underperformed, but only if verified." A single global
    verified-only flag can't express this — reviewed and found it forces every queried tag
    (including ones that can never be "verified," like a creative-reaction tag) through the
    same requirement, making the actual quadrant query unanswerable.

    A bare string or {value, source} object (not wrapped in a list) is also accepted for a
    single-tag query — the same ergonomic gap found in testing for `markets`: a caller with
    only one tag to filter on will plausibly pass it unwrapped.
    """
    if tags is None:
        return []
    if isinstance(tags, (str, dict)):
        tags = [tags]
    if not isinstance(tags, list):
        raise ValueError(f"tags filter must be a list (or a single string/object), got {tags!r}")
    out = []
    for t in tags:
        if isinstance(t, str):
            value, source = t.strip(), None
        elif isinstance(t, dict):
            raw_value = t.get("value")
            if not isinstance(raw_value, str) or not raw_value.strip():
                raise ValueError(f"tag query object must have a non-empty string 'value', got {t!r}")
            value = raw_value.strip()
            # Normalised exactly as on the way in: a value the library accepted must be
            # usable to search for itself, or the forgiveness is a trap.
            source = enums.normalise(t.get("source"), field="tag 'source'",
                                     valid=VALID_TAG_SOURCES,
                                     synonyms=enums.TAG_SOURCE_SYNONYMS)
        else:
            raise ValueError(f"tag query entry must be a string or {{value, source}} object, got {t!r}")
        if not value:
            raise ValueError(f"tag query value cannot be empty or whitespace-only, got {t!r}")
        out.append((value.lower(), source))
    return out


def checked_link(value):
    """A link somebody will follow, or a refusal (§12.4).

    "ask Dana" in a field called `asset_link` is a field that LOOKS like a link and is not
    one, and the reader finds that out by clicking. The check is deliberately shallow — a
    scheme and something after it — because this product cannot tell a live Figma board from a
    dead one and pretending otherwise would be a different false claim.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if not re.match(r"^(https?|s3|gs|smb|file)://\S+$", text, re.IGNORECASE):
        raise ValueError(
            f"`asset_link` must be a link somebody can open — it is where the work lives, and "
            f"a reader follows it. Got {text!r}. Use the full URL (https://…), or leave it out "
            f"and put the note in `detail`.")
    return text


def fold_campaign_type(value):
    """The comparison key for a campaign type (§12.4/D102).

    Through the declared vocabulary, like every other one: unfolded, "Store Launch" and
    `store_launch` are two checklists, which is C16's failure and the reason D102 could not be
    built until §12.2 gave a customer somewhere to declare their own words.
    """
    if not value or not str(value).strip():
        return None
    return fold_vocabulary("campaign_types", str(value))


def insert_campaign(conn, *, title, record_type="campaign", status=None, detail=None,
                    deck_text=None, asset_path=None, tags=None, region=None, market=None,
                    markets=None, collection=None, supersedes=None,
                    commentary_checked=False, starts_on=None, ends_on=None,
                    asset_link=None, campaign_type=None, partner=None) -> str:
    import store

    record_type = _normalise_record_type(record_type)
    status = _normalise_status(status)
    tags = normalize_tags(tags, has_actual_metrics=False)  # brand-new: no metrics can exist yet
    markets = normalize_markets(markets)
    cid = store._id("camp")
    now = store._now()
    if status is None and record_type == "campaign":
        status = "concluded"  # reference/stub records have no lifecycle status by default
    # §9.6: validated here like every other window, because this is the one path that wrote
    # one without going through `update_campaign`.
    starts = _checked_date(starts_on, "starts_on")
    ends = _checked_date(ends_on, "ends_on")
    if starts and ends and ends < starts:
        raise ValueError(
            f"`ends_on` ({ends}) is before `starts_on` ({starts}). A window that closes "
            f"before it opens overlaps nothing, so it would look recorded and match no event.")
    conn.execute(
        """INSERT INTO campaigns (id, title, record_type, status, tags, region, market,
                                  markets, collection, supersedes, detail, deck_text,
                                  asset_path, asset_link, campaign_type, partner,
                                  commentary_checked, starts_on, ends_on,
                                  created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cid, title, record_type, status, json.dumps(tags), region, market,
         json.dumps(markets), collection, supersedes, detail, deck_text, asset_path,
         checked_link(asset_link), str(campaign_type).strip() if campaign_type else None,
         str(partner).strip() if partner else None,
         1 if commentary_checked else 0, starts, ends, now, now),
    )
    conn.commit()
    return cid


def mark_embedded(conn, campaign_id: str, flag: bool) -> None:
    """Flag whether a campaign is FULLY embedded, i.e. searchable in its entirety.

    This used to mean "has at least one embedded chunk", which was merely imprecise while
    partial embedding was an accident. Item 2.1 made partial a designed outcome (a time
    budget stops mid-deck), so "at least one" would have reported a deck with 2 of 12
    sections indexed as searchable — the stored-versus-searchable conflation defect 05 is
    about. The counts in list_campaigns tell the partial story."""
    import store

    conn.execute(
        "UPDATE campaigns SET embedded = ?, updated_at = ? WHERE id = ?",
        (1 if flag else 0, store._now(), campaign_id),
    )
    conn.commit()


def _overlapping_once(conn, record: dict) -> list:
    """Every event that could touch this record, fetched once per `get_campaign`.

    Memoised on the record dict itself rather than on the connection: it lives exactly as long
    as this one read, so an event recorded a moment later is picked up by the next call. A
    cache that outlived the call would be the stale-link failure §9.6 refused a join table to
    avoid.
    """
    import context

    if "_overlapping" not in record:
        widest = context.widest_window(record)
        record["_overlapping"] = ([] if not widest["starts_on"] else context.overlapping_window(
            conn, starts_on=widest["starts_on"], ends_on=widest["ends_on"],
            markets=sorted(context._market_names(record))))
    return record["_overlapping"]


def get_campaign(conn, campaign_id: str) -> Optional[dict]:
    import store

    row = conn.execute("SELECT * FROM campaigns WHERE id = ?", (campaign_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["tags"] = _parse_stored_tags(d["tags"])
    d["markets"] = _parse_stored_markets(d["markets"])
    d["metrics"] = [dict(m) for m in conn.execute(
        "SELECT * FROM metrics WHERE campaign_id = ? ORDER BY created_at", (campaign_id,)
    ).fetchall()]
    # §9.8: "the caveat travels with the metric EVERYWHERE it is cited". Attached here, in the
    # one function that loads metrics, rather than at each of the places that display them — a
    # hand-maintained list of call sites is a copy of the codebase, and this project has been
    # bitten by that shape four times. Every reader gets it, including readers nobody has
    # written yet.
    #
    # Guarded on both halves so the ordinary call costs nothing: a campaign with no window
    # cannot overlap anything, and a campaign with no metrics has no outcome to qualify.
    # ACTUAL rows only. A target is what somebody aimed at before anything happened and a
    # prediction is this library's own forecast; neither can be confounded by an event, and
    # calling a target "an outcome that ran through" something is the category error the
    # `metric_type` comment above guards — it would also corrupt §9.9, which scores
    # predictions against actuals.
    import metrics as metrics_module
    measured = metrics_module.measured(d["metrics"])
    for metric in d["metrics"]:
        metric.update({"status": "not_an_outcome", "confounded": False, "confounded_by": [],
                       "ran_during": 0, "confounded_basis": "computed"})
    if measured:
        import context

        # Once per RECORD, then narrowed per row in memory. Computed per row it ran the
        # overlap query once per month of a workbook — and `context.record` already fans out
        # across the whole library, so §9.8 was adding two queries per campaign to it.
        for metric in measured:
            metric.update(context.confounders_for(conn, d, metric=metric,
                                                  events=_overlapping_once(conn, d)))
    d["has_metrics"] = len(d["metrics"]) > 0
    # Separately, because they answer different questions and §8.8 made the difference
    # reachable. `has_metrics` counts any row — a forecast, and now a TARGET. "Has this
    # campaign been measured" is what decides whether to go and ask for its numbers, and a
    # campaign carrying only the figure somebody was aiming at has not been measured at all.
    d["has_actual_metrics"] = bool(metrics_module.measured(d["metrics"]))
    # §10.3: the sentences people gave about this record, in their words. The most valuable
    # content the library holds, and a table nothing reads is where it goes to die.
    d["feedback_notes"] = feedback_notes(conn, campaign_id)
    # §11.5, attached HERE — the one place a record loads, and therefore the only place the
    # split can be attached once and reach every reader, including retrieval. The same
    # reasoning §9.8 reached about confounders: a list of call sites is a copy of the
    # codebase, and this project has been bitten by that shape six times.
    d["reactions"] = reactions_for(conn, campaign_id)
    split = disagreement_on(conn, campaign_id, rows=d["reactions"])
    if split:
        d["disagreement"] = split
    d.pop("_overlapping", None)
    # §9.1: whether anything has come back. A concluded campaign whose assets are all
    # `proposed` has never been checked against what actually ran, which is the state the
    # review says the library cannot currently notice.
    d["has_delivered_assets"] = bool(conn.execute(
        "SELECT 1 FROM assets WHERE campaign_id = ? AND phase = 'delivered' LIMIT 1",
        (campaign_id,)).fetchone())
    d["has_evaluations"] = conn.execute(
        "SELECT 1 FROM evaluations WHERE campaign_id = ? LIMIT 1", (campaign_id,)
    ).fetchone() is not None
    d["superseded_by"] = [r["id"] for r in conn.execute(
        "SELECT id FROM campaigns WHERE supersedes = ?", (campaign_id,)
    ).fetchall()]
    d["is_superseded"] = len(d["superseded_by"]) > 0
    d["commentary"] = store.get_commentary(conn, campaign_id)
    # Derived, not maintained — same reasoning as superseded_by/is_superseded above: a
    # freeform collection value on its own isn't usable from a single record without a way
    # to find the other members (design review flagged this as the missing half of the
    # feature).
    d["collection_siblings"] = [r["id"] for r in conn.execute(
        "SELECT id FROM campaigns WHERE collection IS NOT NULL AND LOWER(collection) = LOWER(?) "
        "AND id != ?", (d["collection"], campaign_id)
    ).fetchall()] if d["collection"] else []
    d["assets"] = store.get_assets_for_campaign(conn, campaign_id)
    chunk_rows = conn.execute(
        "SELECT embedded FROM campaign_chunks WHERE campaign_id = ?", (campaign_id,)
    ).fetchall()
    d["chunks_total"] = len(chunk_rows)
    d["chunks_embedded"] = sum(1 for r in chunk_rows if r["embedded"])
    # Same counts list_campaigns reports, so a caller does not get a different answer about
    # the same record depending on which tool it asked.
    d["assets_total"] = len(d["assets"])
    d["assets_embedded"] = sum(1 for a in d["assets"] if a["embedded"])
    return d


def list_campaigns(conn, *, record_type: Optional[str] = None,
                   status: Optional[str] = None) -> list[dict]:
    import store

    clauses, params = [], []
    if record_type:
        clauses.append("record_type = ?")
        params.append(_normalise_record_type(record_type))
    if status:
        clauses.append("status = ?")
        params.append(_normalise_status(status))
    sql = "SELECT * FROM campaigns"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at"
    rows = conn.execute(sql, params).fetchall()

    ids = [r["id"] for r in rows]
    with_metrics: set[str] = set()
    with_actuals: set[str] = set()
    with_evaluations: set[str] = set()
    # The `metrics` table may not exist, and this guard is DEFENSIVE rather than a fix for a
    # reachable crash — the distinction matters, because an overstated one is the kind of
    # claim this project keeps finding in its own comments. `upgrade()` runs `_SCHEMA`'s
    # `CREATE TABLE IF NOT EXISTS` for every table before anything else, and every entry
    # point calls `init_db()`, so a live install cannot reach here without the table; review
    # drove a v0.2.0 fixture through `init_db` → `save_evaluation` and it was fine. What
    # actually breaks an upgraded database is missing COLUMNS (D89), which this does not
    # touch. It is kept because `campaigns_with_actual_metrics` carries the same guard for
    # the same reason — a shared reader is the right place to stop a shared failure — and
    # because §13.1 put this function on the save path, where a raise is unrecoverable.
    if ids and store._columns(conn, "metrics"):
        placeholders = ",".join("?" * len(ids))
        with_metrics = {r["campaign_id"] for r in conn.execute(
            f"SELECT DISTINCT campaign_id FROM metrics WHERE campaign_id IN ({placeholders})", ids
        ).fetchall()}
        # §13.1: MEASURED rows, separately. `get_campaign` carries both flags and this
        # carried only the first, so a record read through the listing had no
        # `has_actual_metrics` key at all — and `core.has_results`, reading it with `.get`,
        # answered False for every campaign in the library without erroring. Two shapes of one
        # record, one of them missing the field the shared predicate reads, is how a shared
        # predicate becomes a silent wrong answer rather than a shared one.
        with_actuals = {r["campaign_id"] for r in conn.execute(
            f"SELECT DISTINCT campaign_id FROM metrics WHERE metric_type = 'actual' "
            f"AND campaign_id IN ({placeholders})", ids).fetchall()}
    if ids and store._columns(conn, "evaluations"):
        placeholders = ",".join("?" * len(ids))
        with_evaluations = {r["campaign_id"] for r in conn.execute(
            f"SELECT DISTINCT campaign_id FROM evaluations WHERE campaign_id IN ({placeholders})", ids
        ).fetchall()}
    superseded_ids = get_superseded_campaign_ids(conn)

    # Counts, not just the rollup flag: "stored" and "searchable" are different states, and
    # a time-budgeted ingest can legitimately leave a campaign between them. A caller that
    # only sees `embedded` cannot tell a fully indexed deck from one with two of twelve
    # sections searchable (defect 05).
    chunk_counts: dict[str, tuple[int, int]] = {}
    asset_counts: dict[str, tuple[int, int]] = {}
    if ids:
        placeholders = ",".join("?" * len(ids))
        for table, sink in (("campaign_chunks", chunk_counts), ("assets", asset_counts)):
            # Same guard, same reason: every table this function reaches for is one an
            # upgraded database may not have, and a listing that raises is a listing no
            # migration can be run from. A missing table means "nothing recorded", which is
            # the truth about a database that never had it.
            if not store._columns(conn, table):
                continue
            for row in conn.execute(
                f"""SELECT campaign_id, COUNT(*) AS total,
                           SUM(CASE WHEN embedded THEN 1 ELSE 0 END) AS done
                    FROM {table} WHERE campaign_id IN ({placeholders})
                    GROUP BY campaign_id""", ids).fetchall():
                sink[row["campaign_id"]] = (row["total"], row["done"] or 0)

    out = []
    for r in rows:
        d = dict(r)
        d["tags"] = _parse_stored_tags(d["tags"])
        d["markets"] = _parse_stored_markets(d["markets"])
        d["chunks_total"], d["chunks_embedded"] = chunk_counts.get(d["id"], (0, 0))
        d["assets_total"], d["assets_embedded"] = asset_counts.get(d["id"], (0, 0))
        # `has_metrics` counts ANY row, a forecast or a target included, and is published
        # raw to clients. `has_actual_metrics` is the one that means "this was measured" —
        # §5.3 spent an item on the distinction and the listing carried only the looser half.
        d["has_metrics"] = d["id"] in with_metrics
        d["has_actual_metrics"] = d["id"] in with_actuals
        d["is_superseded"] = d["id"] in superseded_ids
        d["has_evaluations"] = d["id"] in with_evaluations
        out.append(d)
    return out


def spellings_of(key: str, value: str) -> list:
    """Every declared spelling of one value, folded — itself included.

    What makes a comparison see a synonym. With nothing declared this is just the folded
    value, which is what every comparison did before.
    """
    import rulebook

    folded = fold_vocabulary(key, value)
    try:
        name = rulebook.canonical(key, value)
        entry = rulebook.vocabulary(key).get(name) if name else None
    except ValueError:
        name, entry = None, None
    if not entry:
        return [folded]
    return sorted({folded, (name or "").strip().lower(),
                   *(w.strip().lower() for w in entry["also"] if w.strip())})


def _any_spelling(column: str, value: str, params: list) -> str:
    """A WHERE clause matching any declared spelling of `value` in `column`."""
    key = {"market": "markets", "region": "markets",
           "collection": "collections"}.get(column, column)
    words = spellings_of(key, value)
    params.extend(words)
    return "(" + " OR ".join(f"LOWER({column}) = ?" for _ in words) + ")"   # noqa: S608


def filter_campaign_ids(conn, *, record_type: Optional[str] = None, status: Optional[str] = None,
                        tags: Optional[Union[str, dict, list]] = None, match_all_tags: bool = False,
                        region: Optional[str] = None, market: Optional[str] = None,
                        markets: Optional[Union[str, list[str]]] = None,
                        collection: Optional[str] = None,
                        exclude_campaign_id: Optional[str] = None) -> list[str]:
    """
    Structured filtering BEFORE similarity ranking (§6.2) — on a single-brand corpus, pure
    vector search returns noise; narrow to the matching campaigns first (e.g. "concluded
    seeding campaigns in APAC"), then rank what's left by similarity. region/market/
    collection are freeform, single-value, and matched case-insensitively (exact match).

    markets is different on purpose: a real activation can span several countries (e.g. a
    SEA launch touching Malaysia, Singapore, and Indonesia), which a single-value region/
    market can't represent — tagging region="Malaysia" makes that same campaign invisible
    to a query for region="Indonesia", even though the activation genuinely included it.
    Pass a single country/sub-market string, or a list of them for ANY-match (e.g.
    markets=["Indonesia", "Thailand"] — matches a campaign whose activation touched EITHER,
    not both); matches any campaign whose stored `markets` list contains at least one of the
    given values (case-insensitive membership, not exact-match on the whole field).

    tags is a list of plain strings (match that value, any source) and/or {value, source}
    objects (match that value AND require that specific source) — mix freely. Defaults to
    ANY-match (has at least one). Pass match_all_tags=True for AND-match (has every one) —
    needed for a quadrant query like tags=["liked", {"value": "underperformed",
    "source": "verified"}]: "liked" (any evidence) AND "underperformed" but only if
    verified. A creative-reaction tag can never itself be "verified" the way a performance
    tag can, so per-tag precision — not one global verified-only flag — is what makes the
    actual quadrant query answerable.
    """
    wanted = _parse_tag_query(tags)
    # §6.4: never surface a replaced record — derived live (see get_campaign's
    # is_superseded/superseded_by) rather than a maintained reverse-pointer column.
    clauses = ["id NOT IN (SELECT supersedes FROM campaigns WHERE supersedes IS NOT NULL)"]
    params: list = []
    if record_type:
        clauses.append("record_type = ?")
        params.append(_normalise_record_type(record_type))
    if status:
        clauses.append("status = ?")
        params.append(_normalise_status(status))
    # D72/D73: the DECLARED spellings of what was asked for, so a query in any of them finds
    # every row in any of them — including rows written before the declaration existed. The
    # first version canonicalised on write instead, which found only the rows written since
    # and split the library in two without saying so.
    if region:
        clauses.append(_any_spelling("region", region, params))
    if market:
        clauses.append(_any_spelling("market", market, params))
    if collection:
        clauses.append(_any_spelling("collection", collection, params))
    if exclude_campaign_id:
        clauses.append("id != ?")
        params.append(exclude_campaign_id)

    sql = "SELECT id, tags, markets FROM campaigns WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at"
    rows = conn.execute(sql, params).fetchall()

    if wanted:
        def one_matches(stored_pairs, value, source) -> bool:
            # D73: the stored spelling and the queried one are compared through the declared
            # vocabulary, so "went down badly" and `not_liked` are one value — including on
            # rows written before anybody declared them. The first version rewrote the value
            # on WRITE instead, which reached only rows written since.
            spellings = set(spellings_of("tags", value))
            return any((v in spellings or fold_vocabulary("tags", v) in spellings)
                       and (source is None or s == source) for v, s in stored_pairs)

        def campaign_matches(row) -> bool:
            stored_pairs = [(e["value"].lower(), e["source"]) for e in _parse_stored_tags(row["tags"])]
            checks = (one_matches(stored_pairs, v, s) for v, s in wanted)
            return all(checks) if match_all_tags else any(checks)

        rows = [r for r in rows if campaign_matches(r)]

    if markets:
        query_values = [markets] if isinstance(markets, str) else markets
        # Through the declared vocabulary on both sides, for the same reason as `market`.
        wanted_markets = {word for v in query_values for word in spellings_of("markets", v)}
        rows = [r for r in rows
                if wanted_markets & {word for m in _parse_stored_markets(r["markets"])
                                     for word in spellings_of("markets", m)}]

    return [r["id"] for r in rows]


def attach_deck_to_campaign(conn, campaign_id: str, *, deck_text: str, asset_path: str,
                            commentary_checked: bool) -> bool:
    """Put a deck on a record that already exists (§12.4/D39). False if it already had one.

    Separate from `update_campaign`, which deliberately does not accept `deck_text` — content
    that large is a re-upload, and its docstring has always said so. This is the other case:
    a record that has NO deck getting its first one, which was previously possible only by
    uploading a duplicate campaign.

    **The "has no deck yet" test is in the WHERE clause, and that is the point of this
    function.** `core.attach_deck` checks it too, and has to, because it owes a sentence
    explaining supersession rather than a bare False. But a check in the caller is a
    check-then-act: two attaches racing past it both passed, and the row ended up with one
    deck's text carrying the other deck's comments — a record that says one thing and is
    annotated with remarks about another, which is precisely the misattribution this product
    is built against. The condition that decides has to be the one the database applies.
    """
    import store

    changed = conn.execute(
        "UPDATE campaigns SET deck_text = ?, asset_path = ?, "
        "commentary_checked = ?, updated_at = ? WHERE id = ? "
        "AND COALESCE(TRIM(deck_text), '') = '' AND asset_path IS NULL",
        (deck_text, asset_path, 1 if commentary_checked else 0, store._now(),
         campaign_id)).rowcount
    conn.commit()
    return bool(changed)


def update_campaign(conn, campaign_id: str, *, title=None, detail=None, record_type=None,
                    status=None, tags=None, region=None, market=None, markets=None,
                    collection=None, supersedes=None, starts_on=None, ends_on=None,
                    window_source=None, asset_link=None, campaign_type=None,
                    partner=None, approval=None, approval_note=None,
                    approval_by=None) -> bool:
    """Update campaign metadata (§6.4) — NOT deck_text/chunks/embeddings; re-upload (or
    supersede) for content changes. Only given fields change; tags/markets, if given, fully
    replace the existing list rather than merging. Returns whether the campaign exists.

    `supersedes` is here because §6.3 needs it and 5.2's reviewers needed it: supersession
    could only be declared at upload, so somebody who realised afterwards had no way to say
    so, and a supersession declared BY MISTAKE hid a record from every future search and was
    undoable only by deleting the campaign. Passing `""` clears it — the retraction that did
    not exist is the reason the 5.2 offer was called dangerous rather than merely wrong.
    """
    import store

    fields, params = [], []
    if title is not None:
        fields.append("title = ?"); params.append(title)
    if detail is not None:
        fields.append("detail = ?"); params.append(detail)
    if record_type is not None:
        fields.append("record_type = ?"); params.append(_normalise_record_type(record_type))
    if status is not None:
        # A blank normalises to None, which means "not saying" — the same as omitting the
        # argument. Appending it anyway would clear a status that was already set, so a
        # spreadsheet row with an empty cell would silently erase one.
        normalised_status = _normalise_status(status)
        if normalised_status is not None:
            fields.append("status = ?"); params.append(normalised_status)
    if tags is not None:
        has_actual = conn.execute(
            "SELECT 1 FROM metrics WHERE campaign_id = ? AND metric_type = 'actual' LIMIT 1",
            (campaign_id,),
        ).fetchone() is not None
        # The names already stored on THIS record, so a re-send of a legacy attribution is
        # distinguishable from somebody typing one today. Read from the row rather than taken
        # from the caller: otherwise a caller could grandfather any name by claiming it was
        # already there.
        stored = conn.execute("SELECT tags FROM campaigns WHERE id = ?",
                              (campaign_id,)).fetchone()
        tags = normalize_tags(
            tags, has_actual_metrics=has_actual,
            already_said_by=attributions_on_file(json.loads(stored["tags"]) if stored
                                                 and stored["tags"] else []))
        fields.append("tags = ?"); params.append(json.dumps(tags))
    if region is not None:
        fields.append("region = ?"); params.append(region)
    if market is not None:
        fields.append("market = ?"); params.append(market)
    if markets is not None:
        fields.append("markets = ?"); params.append(json.dumps(normalize_markets(markets)))
    if collection is not None:
        fields.append("collection = ?"); params.append(collection)
    # §12.4: the three the review named. A record that predates a field is the ordinary case
    # for anything added later, so each is settable on an existing record and not only at
    # upload.
    if asset_link is not None:
        fields.append("asset_link = ?"); params.append(checked_link(asset_link))
    if campaign_type is not None:
        fields.append("campaign_type = ?"); params.append(str(campaign_type).strip() or None)
    if partner is not None:
        fields.append("partner = ?"); params.append(str(partner).strip() or None)
    if approval is not None:
        fields.append("approval = ?")
        params.append(enums.normalise(approval, field="approval", valid=store.VALID_APPROVALS,
                                      allow_none=False))
    if approval_note is not None:
        fields.append("approval_note = ?")
        params.append(" ".join(str(approval_note).split()) or None)
    if approval_by is not None:
        fields.append("approval_by = ?"); params.append(str(approval_by).strip() or None)
    if supersedes is not None:
        fields.append("supersedes = ?")
        params.append(checked_supersedes(conn, campaign_id, supersedes))
    # §9.6: when it ran. Validated here rather than at the edge because every caller writes
    # through this one function, and a window stored as "March-ish" compares as text against
    # every event range in the library — silently, and wrongly.
    if starts_on is not None or ends_on is not None:
        # Checked against what is STORED, not only against what arrived in this call. Guarded
        # on the two-field case alone, a one-field typo fix wrote 2026-12-01 to 2026-03-31 —
        # and §9.6 then printed that window backwards inside the sentence asserting nothing
        # had been going on. Clearing an end is still allowed: an always-on campaign is
        # ordinary, and refusing a clear because the stored other end is later would make it
        # unsettable.
        current = get_campaign(conn, campaign_id) or {}
        new_start = (_checked_date(starts_on, "starts_on") if starts_on is not None
                     else current.get("starts_on"))
        new_end = (_checked_date(ends_on, "ends_on") if ends_on is not None
                   else current.get("ends_on"))
        if new_start and new_end and new_end < new_start:
            raise ValueError(
                f"that would leave this campaign's window running {new_start} to {new_end}, "
                f"which closes before it opens. A window like that overlaps nothing, so it "
                f"looks recorded and matches no event. Pass both ends to move the whole "
                f"window.")
        if starts_on is not None:
            fields.append("starts_on = ?"); params.append(new_start)
        if ends_on is not None:
            fields.append("ends_on = ?"); params.append(new_end)
        # A window a PERSON set clears the workbook mark, so a later spreadsheet row cannot
        # widen over it. Passing `window_source` explicitly is how the workbook path keeps it.
        fields.append("window_source = ?"); params.append(window_source)

    if not fields:
        return get_campaign(conn, campaign_id) is not None

    fields.append("updated_at = ?")
    params.append(store._now())
    params.append(campaign_id)
    cur = conn.execute(f"UPDATE campaigns SET {', '.join(fields)} WHERE id = ?", params)
    conn.commit()
    return cur.rowcount > 0


def _checked_date(value, field: str) -> Optional[str]:
    """An ISO date, or None to clear. §9.6's arithmetic is string comparison on ISO dates —
    which is exactly right for `YYYY-MM-DD` and silently wrong for anything else."""
    import datetime

    text = str(value or "").strip()
    if not text:
        return None
    try:
        return str(datetime.date.fromisoformat(text))
    except ValueError:
        raise ValueError(
            f"`{field}` must be an ISO date (YYYY-MM-DD); got {value!r}. This is compared "
            f"against every context event's range by date arithmetic, and a string that is "
            f"not a date compares as text without failing.") from None


def checked_supersedes(conn, campaign_id: Optional[str], supersedes) -> Optional[str]:
    """The three ways a supersession can be nonsense, refused before it is written.

    All three hide records from search, which is what makes them worth checking rather than
    accepting: a record pointed at nothing is a link nobody can follow back, a record
    superseding itself removes itself from the library, and a cycle removes both ends and
    would make §6.3's chain walk run forever.

    `campaign_id` is None on the upload path, where the record does not have an id yet: only
    the dangling-pointer check applies there, and it is the one that mattered — the upload
    path had no validation at all, so a typo left {"", "  ", "camp_nope"} in the set of
    superseded ids and a link pointing at nothing.
    """
    target = (supersedes or "").strip()
    if not target:
        return None                       # the retraction
    if campaign_id is not None and target == campaign_id:
        raise ValueError("a campaign cannot supersede itself — that would hide it from "
                         "every search, including its own")
    if conn.execute("SELECT 1 FROM campaigns WHERE id = ?", (target,)).fetchone() is None:
        raise ValueError(f"cannot supersede {target!r}: there is no such record. Supersession "
                         f"hides the superseded record from search, so a wrong id here "
                         f"quietly hides nothing and links nothing")
    seen, walk = ({campaign_id} if campaign_id else set()), target
    while walk:
        if walk in seen:
            raise ValueError(f"that would make a supersession cycle through {walk!r}. Both "
                             f"ends of a cycle are hidden from every search and neither can "
                             f"be reached from the other")
        seen.add(walk)
        row = conn.execute("SELECT supersedes FROM campaigns WHERE id = ?",
                           (walk,)).fetchone()
        walk = row["supersedes"] if row else None
    return target


def supersession_chain(conn, campaign_id: str) -> list[str]:
    """Oldest to newest, following `supersedes` back from this record (§6.3 / D64).

    `diff_campaigns` compared adjacent judgments only, so a finding v2 explicitly resolved
    was reported against v3 as "neither resolved nor repeated" — the library forgetting its
    own evidence and then hedging about it. The visited set is belt and braces: cycles are
    refused on write, but a database that predates that check could still hold one.
    """
    chain, seen, walk = [], set(), campaign_id
    while walk and walk not in seen:
        chain.append(walk)
        seen.add(walk)
        row = conn.execute("SELECT supersedes FROM campaigns WHERE id = ?",
                           (walk,)).fetchone()
        walk = row["supersedes"] if row else None
    chain.reverse()
    return chain


def _forget_commitment_vectors(conn, campaign_id: str) -> None:
    """A deleted campaign's commitment vectors go with it (§9.3).

    The rows cascade on the foreign key; the vectors live in their own table keyed by a string
    and nothing reaches them, so they would outlive the campaign as orphans — the same cleanup
    `delete_campaign` already does for chunk and asset vectors.
    """
    import store

    if not store._columns(conn, "commitments") or not store._columns(conn, "commitment_vectors"):
        return
    ids = [f"commitment:{r['id']}" for r in conn.execute(
        "SELECT id FROM commitments WHERE campaign_id = ?", (campaign_id,)).fetchall()]
    for vid in ids:
        conn.execute("DELETE FROM commitment_vectors WHERE vector_id = ?", (vid,))
    # §13.6/D126: and the row saying which model made them. While commitment vectors carried
    # no provenance this was nothing to clean up; now they do, and leaving it is precisely the
    # leak `forget_vector_models` exists to prevent — `embedding_models` reporting a model that
    # produced nothing still in the index, and `stale_vectors` naming a vector that is gone.
    store.forget_vector_models(conn, ids)


def delete_campaign(conn, campaign_id: str) -> bool:
    """Delete a campaign and everything that's exclusively its own (§6.4): chunks + their
    text vectors, image assets + their pHash fingerprints, CLIP vectors, and the actual
    files on disk (deck + images) — not just the SQL rows (review found the vectors and
    files were being left behind, a leak and a GDPR-erasure gap). Metrics cascade (FK).
    Evaluations that cited it are kept — they're a record of a judgment that was made — but
    detached (ON DELETE SET NULL). `supersedes`/`is_superseded` are derived live from
    `supersedes`, not a maintained pointer, so deleting a record automatically and correctly
    updates what counts as superseded — nothing to clean up here for that."""
    import store

    campaign = get_campaign(conn, campaign_id)
    if not campaign:
        return False

    chunk_ids = store.get_chunk_ids_for_campaign(conn, campaign_id)
    if chunk_ids:
        vectorstore.delete_many(conn, chunk_ids)

    assets = store.get_assets_for_campaign(conn, campaign_id)
    asset_ids = [a["id"] for a in assets]
    if asset_ids:
        vectorstore.delete_many(conn, asset_ids, space="asset")

    # §9.3: the commitment rows cascade on the foreign key, but their vectors live in their own
    # table keyed by a string and nothing else reaches them — so they would outlive the
    # campaign as orphans, which is the cleanup the two lines above already do for chunks and
    # assets.
    _forget_commitment_vectors(conn, campaign_id)

    for file_path in [campaign["asset_path"]] + [a["file_path"] for a in assets]:
        if file_path:
            (config.ASSET_DIR / file_path).unlink(missing_ok=True)

    cur = conn.execute("DELETE FROM campaigns WHERE id = ?", (campaign_id,))
    conn.commit()
    return cur.rowcount > 0


def record_notice(conn, *, code: str, campaign_id: str, detail: Optional[str] = None) -> str:
    """Persist a notice against a record (§10.1/D20).

    Re-raising the same condition on the same record updates it rather than stacking: it is
    the same outstanding job, and a queue that showed it four times would be counting how
    often somebody looked rather than what is wrong.
    """
    import store

    nid = store._id("notice")
    conn.execute(
        "INSERT INTO campaign_notices (id, campaign_id, code, detail, created_at) "
        "VALUES (?,?,?,?,?) ON CONFLICT(campaign_id, code) DO UPDATE SET "
        "detail = excluded.detail, cleared_at = NULL",
        (nid, campaign_id, code, detail, store._now()))
    conn.commit()
    row = conn.execute("SELECT id FROM campaign_notices WHERE campaign_id = ? AND code = ?",
                       (campaign_id, code)).fetchone()
    return row["id"]


def clear_notice(conn, notice_id: str) -> None:
    import store

    conn.execute("UPDATE campaign_notices SET cleared_at = ? WHERE id = ?",
                 (store._now(), notice_id))
    conn.commit()


def open_notices(conn, campaign_id: str) -> list[dict]:
    import store

    if not store._columns(conn, "campaign_notices"):
        return []
    return [dict(r) for r in conn.execute(
        "SELECT id, code, detail, created_at FROM campaign_notices "
        "WHERE campaign_id = ? AND cleared_at IS NULL ORDER BY created_at, id",
        (campaign_id,)).fetchall()]


# Which axis a value belongs to. Two people saying "liked" and "underperformed" have not
# disagreed — that is the most ordinary finding in marketing — so the axis is what makes a
# contradiction a contradiction.
REACTION_AXES = {
    "liked": "reaction", "not_liked": "reaction", "mixed_reaction": "reaction",
    "performed_well": "performance", "underperformed": "performance",
    "performed_as_expected": "performance", "no_data_yet": "performance",
}


# §11.6/D13: the four different reasons this library cannot name an author. A null said all
# four at once, and they have completely different consequences for a judgment citing the
# words — §2.5's whole point was that a deck's own speaker note and a client's objection are
# not the same thing, and `author: null` on both erases exactly that.
AUTHOR_UNKNOWN = {
    # A PowerPoint notesSlide has no author FIELD. Nobody withheld anything and the words are
    # almost certainly the deck's own authors: the agency talking to itself.
    "format_carries_none":
        "This kind of commentary does not carry an author at all — a speaker note has no such "
        "field — so nobody withheld a name. These are almost certainly the deck authors' own "
        "words, which is a different thing from a client's comment and carries different "
        "weight.",
    # A PDF annotation CAN carry `/T` and this one does not: a fact about the person, not the
    # format.
    "file_did_not_say":
        "This kind of commentary can carry an author and this one does not. Somebody "
        "commented without a name, or their reader was not configured with one — so the "
        "words are somebody's and the library cannot say whose.",
    # §11.6's backfill: a fact about US, not about the customer's deck.
    "not_captured":
        "This record was stored before this library read commentary at all, so the file may "
        "well have said who wrote it and nobody looked. That is a gap in what was captured, "
        "not silence in the document.",
    # Body text is not a comment and has no author question to answer.
    "never_claimed":
        "Nothing about this text claims an author. It is the brief itself rather than a "
        "comment on it, so there is no missing name here.",
}

# Which commentary kinds can carry an author at all. §2.5 stores the FORMAT the words arrived
# in, and this is the one thing that format legitimately tells you.
_CARRIES_AN_AUTHOR = ("comment", "annotation")


def mark_authorship_backfilled(conn, campaign_id: str, when: str) -> None:
    conn.execute("UPDATE campaigns SET authorship_backfilled_at = ? WHERE id = ?",
                 (when, campaign_id))
    conn.commit()


def records_without_authorship_backfill(conn) -> list[dict]:
    """Records stored before this library read commentary, and not yet stamped (§11.6)."""
    import store

    if "authorship_backfilled_at" not in store._columns(conn, "campaigns"):
        return []
    return [dict(r) for r in conn.execute(
        "SELECT id, title FROM campaigns WHERE authorship_backfilled_at IS NULL "
        "AND COALESCE(commentary_checked, 0) = 0 ORDER BY created_at, id").fetchall()]


def author_of(source: Optional[dict], *, commentary_checked: bool = True) -> dict:
    """Who wrote this, or WHICH KIND of unknown it is (§11.6/D13).

    Never a bare null. "The file did not say", "the format cannot say", "we never looked" and
    "nothing here claims an author" are four different statements, and a reader given one null
    for all four cannot tell a client's anonymous objection from the deck talking to itself.
    """
    source = source or {}
    named = (source.get("author") or "").strip()
    if named:
        return {"author": named, "basis": "stated"}
    kind = source.get("kind") or "body"
    if kind not in _CARRIES_AN_AUTHOR and kind != "speaker_note":
        why = "never_claimed"
    elif not commentary_checked:
        why = "not_captured"
    elif kind == "speaker_note":
        why = "format_carries_none"
    else:
        why = "file_did_not_say"
    return {"author": "unknown", "why": why, "basis": "computed",
            "what_it_means": AUTHOR_UNKNOWN[why]}


def record_erasure(conn, *, pseudonym: str, why: str, said_by: str, mentions: int) -> str:
    """Put an erasure on the record, without the name it removed (§11.7)."""
    import store

    eid = store._id("erase")
    conn.execute(
        "INSERT INTO erasures (id, pseudonym, mentions, why, said_by, erased_at, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (eid, pseudonym, mentions, why, said_by, store._now_iso(), store._now()))
    conn.commit()
    return eid


def erasures(conn) -> list[dict]:
    import store

    if not store._columns(conn, "erasures"):
        return []
    return [dict(r) for r in conn.execute(
        "SELECT pseudonym, mentions, why, said_by, erased_at FROM erasures "
        "ORDER BY created_at DESC, id DESC").fetchall()]


def record_reaction(conn, *, campaign_id: str, value: str, said_by: str, source: str,
                    said_at: str, role=None) -> str:
    """One person's view, appended (§11.5). Never an update: see the table comment."""
    import store

    rid = store._id("react")
    conn.execute(
        "INSERT INTO reactions (id, campaign_id, axis, value, source, said_by, said_at, "
        "role, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (rid, campaign_id, REACTION_AXES.get(value, "reaction"), value, source,
         said_by, said_at, role, store._now()))
    conn.commit()
    return rid


def reactions_for(conn, campaign_id: str) -> list[dict]:
    """Every view on file, oldest first (§11.5)."""
    import store

    if not store._columns(conn, "reactions"):
        return []
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM reactions WHERE campaign_id = ? ORDER BY created_at, id",
        (campaign_id,)).fetchall()]
    seen: dict = {}
    for row in rows:
        key = (row["said_by"].strip().casefold(), row["axis"])
        # Somebody revisiting their own view is a SEQUENCE, not a split — and the pair is the
        # interesting part: "this looked fine before the numbers came in" is the most
        # informative row this table holds, which is §9.8's argument about attributions.
        row["supersedes_own_earlier_view"] = key in seen
        seen[key] = row["id"]
    return rows


def recent_voices(conn, *, limit: int = 4) -> list:
    """The people who have most recently recorded a view (§11.3).

    So the menu can number them. Most recent rather than most frequent: an agency's queue is
    worked in bursts about whoever is live this month, and the person who answered most often
    two years ago is not the likely answer today.
    """
    import store

    if not store._columns(conn, "reactions"):
        return []
    return [r["said_by"] for r in conn.execute(
        "SELECT said_by, MAX(created_at) AS last FROM reactions GROUP BY said_by "
        "ORDER BY last DESC LIMIT ?", (limit,)).fetchall()]


def disagreement_on(conn, campaign_id: str, rows=None) -> Optional[dict]:
    """Where two different PEOPLE hold different views on one axis (§11.5).

    Two voices, not two values: one person changing their mind is a revision, and counting it
    as a split would report a disagreement that never happened to every judgment citing the
    record. Per axis, because "they liked it and it underperformed" is not a contradiction.
    """
    rows = reactions_for(conn, campaign_id) if rows is None else rows
    by_axis: dict = {}
    for row in rows:
        by_axis.setdefault(row["axis"], []).append(row)
    split = {}
    for axis, voices in by_axis.items():
        # Collapsed per PERSON first, keeping their latest — and that collapse is what makes
        # this about two people rather than two values. Somebody changing their own mind
        # leaves one entry holding one value, so it cannot read as a split. An explicit
        # `len(standing) > 1` beside this was redundant: mutation showed removing it changed
        # nothing, because the collapse had already decided.
        standing: dict = {}
        for voice in voices:
            standing[voice["said_by"].strip().casefold()] = voice
        values = sorted({v["value"] for v in standing.values()})
        if len(values) > 1:
            split[axis] = {
                "values": values,
                # No `winner`, no `standing`. That is the whole of D10.
                "voices": [{"said_by": v["said_by"], "value": v["value"], "role": v["role"],
                            "said_at": v["said_at"], "source": v["source"]}
                           for v in sorted(standing.values(), key=lambda v: v["created_at"])],
            }
    if not split:
        return None
    return {
        **split, "basis": "computed",
        "what_it_means": (
            "Two people recorded different views about this campaign and both are on file. "
            "This library will NOT say which one stands: authority order is configured in "
            "the rulebook and is not configured here, and preferring the later view, or the "
            "client's, or somebody's job title would be authority nobody granted it. Cite "
            "the split rather than a side — a campaign people disagreed about is stronger "
            "evidence about this client's taste than one everybody liked, and quoting it as "
            "unanimous throws that away."),
    }


def record_authorship(conn, *, subject_kind: str, subject_key: str, on_behalf_of: str,
                      role: Optional[str] = None, note: Optional[str] = None) -> dict:
    """Record who decided this, and the account that made the call (§11.1–§11.4).

    `captured_*` is derived here and never accepted from a caller — that is the whole point.
    A parameter would make it a second thing the model can write, and the one fact this
    server can establish for itself would become one more claim it has to take on trust.
    """
    import store

    import identity

    who = identity.who_said_it(on_behalf_of=on_behalf_of)
    captured = who["captured_by"]
    row = store._id("who")
    conn.execute(
        "INSERT INTO authorship (id, subject_kind, subject_key, on_behalf_of, "
        "on_behalf_of_role, note, captured_source, captured_method, captured_account, "
        "captured_host, captured_display, channel, session_id, captured_at, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (row, subject_kind, subject_key, who["on_behalf_of"]["name"], role,
         " ".join(str(note or "").split()) or None,
         captured["source"], captured["method"],
         captured.get("os_user") or captured.get("subject"), captured.get("host"),
         captured.get("display_name"), identity.channel(), identity.session_id(),
         store._now_iso(), store._now()))
    conn.commit()
    return who


def authorship_for(conn, subject_kind: str, subject_key: str) -> Optional[dict]:
    """Who decided this, or None. The LATEST, with the history still on file."""
    import store

    if not store._columns(conn, "authorship"):
        return None
    row = conn.execute(
        "SELECT * FROM authorship WHERE subject_kind = ? AND subject_key = ? "
        "ORDER BY created_at DESC, id DESC LIMIT 1", (subject_kind, subject_key)).fetchone()
    return _as_authorship(row) if row else None


def _as_authorship(row) -> dict:
    import identity

    # The sentence, rebuilt from the stored method. It was dropped here — it lives on the live
    # dict `identity.captured_by()` returns and was never persisted — so `answers()`, the
    # surface built to review human overrides, reported `source: verified` by a named person
    # with the caveat that makes it honest stripped off. §11.2 says in writing that this is
    # the thing it refuses.
    captured = {"source": row["captured_source"], "method": row["captured_method"],
                "what_it_means": identity.METHOD_MEANS.get(
                    row["captured_method"], "This account was not recorded.")}
    for field, key in (("captured_account", "account"), ("captured_host", "host"),
                       ("captured_display", "display_name")):
        if row[field]:
            captured[key] = row[field]
    said = {"name": row["on_behalf_of"], "source": "stated"}
    # `note` is what they SAID, so it travels with the name and not with the derived account.
    if "note" in row.keys() and row["note"]:
        said["why"] = row["note"]
    if row["on_behalf_of_role"]:
        said["role"] = row["on_behalf_of_role"]
        said["role_basis"] = "as stated at the time"
    # §11.3: computed from the two names rather than stored, so it cannot disagree with them.
    # This is the distinction "most products miss" and the reason they miss it: on the day the
    # note is written everybody knows who was in the room, and two years on the record says a
    # name with nothing to say whether that person held the view or merely typed it.
    #
    # None, not False, when the account has no name to compare — `unattributed` and `import`
    # know nothing about who was there, and answering "no, somebody else" would be a claim
    # made out of an absence.
    mine = (row["captured_display"] or "").strip().casefold()
    speaking = (mine == row["on_behalf_of"].strip().casefold()) if mine else None
    return {
        "on_behalf_of": said, "captured_by": captured,
        "speaking_for_themselves": speaking,
        "captured_at": row["captured_at"],
        **({"channel": row["channel"]} if row["channel"] else {}),
        **({"session_id": row["session_id"]} if row["session_id"] else {}),
    }


def record_answer(conn, *, subject_kind: str, subject_key: str, answer: str, note: str,
                  said_by: str, said_at: Optional[str] = None,
                  evaluation_id: Optional[str] = None) -> str:
    """One answer to one question the product asked (§10.2)."""
    import store

    aid = store._id("answer")
    conn.execute("INSERT INTO answers (id, subject_kind, subject_key, evaluation_id, answer, "
                 "note, said_by, said_at, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                 (aid, subject_kind, subject_key, evaluation_id, answer, note, said_by,
                  said_at or store._now_iso(), store._now()))
    conn.commit()
    return aid


def answers_for(conn, subject_kind: str, keys: Optional[list] = None) -> dict:
    """The LATEST answer per subject, keyed by subject (§10.2).

    Append-only underneath, so "the latest" is a read-time decision rather than a destroyed
    history — what somebody said in March survives being contradicted in June, and a judgment
    made between the two stays explicable.
    """
    import store

    if not store._columns(conn, "answers"):
        return {}
    sql = "SELECT * FROM answers WHERE subject_kind = ?"
    args: list = [subject_kind]
    if keys is not None:
        if not keys:
            return {}
        sql += f" AND subject_key IN ({','.join('?' * len(keys))})"
        args += list(keys)
    latest: dict = {}
    for row in conn.execute(sql + " ORDER BY created_at, id", args).fetchall():
        latest[row["subject_key"]] = {
            "answer": row["answer"], "note": row["note"], "said_by": row["said_by"],
            "said_at": row["said_at"], "evaluation_id": row["evaluation_id"],
            # The server's own clock, beside the caller-supplied `said_at`. §10.2's set-aside
            # is a statement about the records that existed WHEN it was made — "these decks
            # are gone" says nothing about one uploaded next week — and deciding that needs a
            # timestamp the caller did not write. `said_at` is a string somebody passed in.
            "recorded_at": row["created_at"],
            # Always. A reader who cannot tell a person's answer from a computed fact will
            # eventually cite one as the other, which is the whole of §2.4.
            "basis": "stated",
        }
    return latest


def every_answer(conn, *, said_by: Optional[str] = None) -> list[dict]:
    """The whole override log, newest first (§10.2).

    Everything, not the latest per subject — `answers_for` gives that, and it is the wrong
    shape here. The table is append-only precisely so "this looked deliberate in March and
    turned out to be a mistake in June" survives, and a log that shows only the June row has
    thrown away the half that made keeping both worthwhile.
    """
    import store

    if not store._columns(conn, "answers"):
        return []
    sql = "SELECT * FROM answers"
    args: list = []
    if said_by:
        # Case-folded. A name is not an identifier, and "show me everything Ana answered" is
        # typed by a person who does not know how she typed it last time.
        sql += " WHERE LOWER(said_by) = LOWER(?)"
        args.append(said_by.strip())
    return [{**dict(r), "basis": "stated"}
            for r in conn.execute(sql + " ORDER BY created_at DESC, id DESC",
                                  args).fetchall()]


def get_state(conn, key: str) -> Optional[str]:
    """One remembered fact about the library, or None (§10.6/D54).

    None is returned for a missing TABLE as well as a missing row, so a database written
    before this existed reads as "nothing remembered" rather than raising — the same shape
    every other additive migration here takes.
    """
    import store

    if not store._columns(conn, "library_state"):
        return None
    row = conn.execute("SELECT value FROM library_state WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_state(conn, key: str, value: Optional[str]) -> None:
    import store

    if not store._columns(conn, "library_state"):
        return
    conn.execute("INSERT INTO library_state (key, value, updated_at) VALUES (?,?,?) "
                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                 "updated_at = excluded.updated_at", (key, value, store._now()))
    conn.commit()


def campaigns_with_open_notices(conn) -> set:
    import store

    if not store._columns(conn, "campaign_notices"):
        return set()
    return {r["campaign_id"] for r in conn.execute(
        "SELECT DISTINCT campaign_id FROM campaign_notices WHERE cleared_at IS NULL")}


def add_feedback_note(conn, *, campaign_id: str, note: str, said_by: str,
                      said_at: str) -> str:
    import store

    nid = store._id("note")
    conn.execute("INSERT INTO feedback_notes (id, campaign_id, note, said_by, said_at, "
                 "created_at) VALUES (?,?,?,?,?,?)",
                 (nid, campaign_id, note, said_by, said_at, store._now()))
    conn.commit()
    return nid


def feedback_notes(conn, campaign_id: str) -> list[dict]:
    """What people have said about this campaign in their own words (§10.3), oldest first."""
    import store

    if not store._columns(conn, "feedback_notes"):
        return []
    return [dict(r) for r in conn.execute(
        "SELECT id, note, said_by, said_at FROM feedback_notes WHERE campaign_id = ? "
        "ORDER BY created_at, id", (campaign_id,)).fetchall()]


def link_evaluation(conn, evaluation_id: str, campaign_id: str) -> None:
    conn.execute("UPDATE evaluations SET campaign_id = ? WHERE id = ?",
                 (campaign_id, evaluation_id))
    conn.commit()


def unlinked_judgment_for(conn, subject_title: str) -> Optional[dict]:
    """A judgment about this title that is not attached to any record (§9.9, D40).

    §5.2's "add this to the library" is the moment: the judgment is on screen and the record
    has just been created. Without this the store-then-reconcile sequence could not complete
    from the commonest starting point — a pitch nobody had stored yet.
    """
    row = conn.execute(
        """SELECT id, subject_title FROM evaluations
           WHERE campaign_id IS NULL AND subject_title = ? COLLATE NOCASE
             AND NOT EXISTS (SELECT 1 FROM reconciliations r WHERE r.evaluation_id = id)
           ORDER BY created_at DESC LIMIT 1""", (subject_title,)).fetchone()
    return dict(row) if row else None


def superseded_by(conn, campaign_id: str) -> Optional[str]:
    """Which record replaced this one, or None (§9.9, D120/D127).

    DERIVED, like every other reverse lookup here — `supersedes` points backwards and a cached
    forward pointer breaks on chains and on fan-in, which is the reasoning the `campaigns`
    table already records for `is_superseded`.
    """
    row = conn.execute("SELECT id FROM campaigns WHERE supersedes = ? "
                       "ORDER BY created_at LIMIT 1", (campaign_id,)).fetchone()
    return row["id"] if row else None


def get_superseded_campaign_ids(conn) -> set[str]:
    rows = conn.execute("SELECT DISTINCT supersedes FROM campaigns WHERE supersedes IS NOT NULL").fetchall()
    return {r["supersedes"] for r in rows}
