"""
§9.4: drift is not a synonym for failure — classify it.

The review: *"The UAE brief's claw machine was a departure from precedent that turned out
better than precedent. Treating every deviation as a defect would teach the library to punish
improvement. Fix: classify each drift item as improvement, neutral or degradation, and mark
that classification **judged** rather than **computed**. Presence and absence are facts;
whether a change was good is a judgment, and usually needs the outcome to settle it."*

**"Presence and absence are facts; whether a change was good is a judgment"** draws a line the
server must not cross. §9.2 computes which briefed images came back; §9.3 looks for the named
promises. Neither can say whether a thing not appearing was a loss — a claw machine replaced by
something better and a claw machine that never turned up produce identical numbers.

So this module RECORDS a classification and never makes one. No default, no inference from the
drift score, and nothing anywhere that reads a high score as a bad sign; that inference is
precisely what "would teach the library to punish improvement" describes.

**"Usually needs the outcome to settle it"** gets a value rather than a caveat. `too_early` is
the honest classification before results exist, and every classification is stamped with
whether the outcome was known WHEN IT WAS MADE — so a later reader can weigh "this looked like
an improvement" against "this was an improvement" without reconstructing which it was. Nothing
is ever overwritten: a reading that changed when the numbers arrived is the most interesting
thing this record can hold.
"""
from __future__ import annotations

from typing import Optional

# The review's own three, plus the one its own sentence requires, plus the one the INSTRUMENT
# requires. `neutral` is "this changed and it did not matter", which is a real finding and not
# an absence of one.
#
# `not_drift` is the correction, and without it every other value here presupposes that the
# difference is real. §9.2 says in its own comments that a photograph of a built claw machine
# will not fingerprint-match the briefed render — so a campaign that ran EXACTLY as written is
# stamped `drifted` on the strength of an instrument that cannot see the case, and the four
# judgments on offer all begin "the execution moved away from the brief". A person looking at
# the photographs had no way to say "it did not; you just couldn't tell", and the computed
# downgrade stood forever. It is the one classification that disputes a fact rather than
# weighing one, which is exactly why it has to be a person's and has to be visible as such.
CLASSIFICATIONS = ("improvement", "neutral", "degradation", "too_early", "not_drift")

# The counts stay complete; the worked examples behind them are trimmed like every other list
# here. `compare_execution` carries this dict, so an uncapped one is paid for in the caller's
# context on every single comparison.
MAX_ITEMS_SHOWN = 12

_MEANING = {
    "improvement": "the execution moved away from the brief and moved somewhere better",
    "neutral": "the execution moved away from the brief and it made no difference",
    "degradation": "the execution moved away from the brief and that cost something",
    "too_early": "whether it mattered cannot be said yet",
    "not_drift": ("the execution did NOT move away from the brief here — the comparison could "
                  "not see that it matched"),
}


def classify(conn, *, campaign_id: str, subject: str, classification: str, why: str,
             classified_by: str, about: Optional[str] = None) -> dict:
    """Record what somebody made of one piece of drift (§9.4).

    `subject` is a STABLE id — the asset, or the commitment — and has to resolve. It was a free
    string keyed on the file path, which is a uuid by design, so the offer read "say whether
    a3f9c21d88e04b17.png not matching was a loss", the record was unreadable months later, and
    a person answering in their own words created a different item: the offer re-fired on the
    same uuid forever and the unjudged count never fell.

    Appends. A classification made before the numbers came in and one made after are two
    different judgments about the same thing, and the pair is worth more than either.
    """
    import enums
    import store

    record = store.get_campaign(conn, campaign_id)
    if record is None:
        raise ValueError(f"{campaign_id!r} is not a record in this library.")
    about = (about or "").strip() or _describe_subject(conn, campaign_id, subject)
    if about is None:
        raise ValueError(
            f"{subject!r} is not something this campaign drifted on. Pass the `asset_id` of a "
            f"briefed image or delivered photograph, or the `commitment_id` of a promise — "
            f"compare_execution and check_commitments both name them. A classification "
            f"against nothing cannot be read back or counted.")
    classification = enums.normalise(classification, field="classification",
                                     valid=CLASSIFICATIONS, allow_none=False)
    if not (why or "").strip():
        raise ValueError(
            "`why` is required: this is read months later by somebody deciding whether to "
            "repeat the change, and a judgment with no reason cannot be weighed against "
            "anything.")
    # §11.2: `identity.person` is the rule. This checked only for an empty string, so
    # `classified_by="the system"` recorded that a person had judged whether a departure from
    # the brief was an improvement — which is the library marking its own homework, in the
    # field §9.4 built to stop exactly that.
    import identity

    classified_by = identity.person(classified_by, field="classified_by")

    # Stamped at the moment of the judgment, not read back later — the point is to record what
    # the person could see when they said it.
    outcome_known = bool(record.get("has_actual_metrics"))
    store.insert_drift_classification(
        conn, campaign_id=campaign_id, subject=subject.strip(), item=about,
        classification=classification, why=why.strip(),
        classified_by=classified_by.strip(), outcome_known=outcome_known)
    # §11.1: the account beside the name. Judging whether a departure from the brief was an
    # improvement decides what every later citation of this campaign's results carries.
    store.record_authorship(conn, subject_kind="drift_classification",
                            subject_key=f"{campaign_id}:{subject.strip()}",
                            on_behalf_of=classified_by)
    # §9.5 stores the classified counts beside the outcome, so a judgment changes what a later
    # citation carries — and a snapshot that never sees the classification is the stale half of
    # the same figure.
    import core
    core._snapshot_execution_drift(conn, campaign_id)
    return _public(store.latest_drift_classification(conn, campaign_id, subject.strip()))


def _describe_subject(conn, campaign_id: str, subject: str) -> Optional[str]:
    """What this drift item IS, in words a person can read back in six months.

    None when the subject is not something this campaign drifted on — a classification against
    nothing cannot be read back, cannot be counted, and silently decrements the unjudged count.
    """
    import store

    promise = store.get_commitment(conn, subject or "")
    if promise and promise["campaign_id"] == campaign_id:
        return promise["text"]
    for asset in store.get_assets_for_campaign(conn, campaign_id):
        if asset["id"] == subject:
            return (f"the delivered photograph taken {asset['captured_on']}"
                    if asset["phase"] == "delivered"
                    else "a briefed image that did not come back")
    return None


def _public(row: dict) -> dict:
    known = bool(row["outcome_known"])
    return {
        **row,
        "outcome_known": known,
        # Never `computed`. §2.4 made that word unwritable from the MCP surface precisely so it
        # keeps meaning "the server worked this out", and whether a change was good is the
        # clearest possible case of something it did not.
        "basis": "judged",
        "what_it_means": (
            f"{row['classified_by']} read this as {row['classification']} — "
            f"{_MEANING[row['classification']]}."
            + ("" if known else
               " It was judged with no measured result on file, so it is a reading of what "
               "happened rather than of what it cost. Worth revisiting when the numbers "
               "arrive.")
            # `too_early` with a result already on file is not a contradiction — the campaign's
            # numbers can be in while nothing measures THIS difference — but the server must
            # not write "nothing has been measured yet" over the top of metrics it holds. That
            # sentence would be the server asserting something false about its own contents.
            + (" There is a measured result on file for this campaign; the reading is that "
               "nothing in it settles this particular difference."
               if known and row["classification"] == "too_early" else "")),
    }


def history(conn, *, campaign_id: str, subject: str) -> list:
    """Every reading of one piece of drift, oldest first (§9.4).

    Nothing is overwritten: "this looked like an improvement before the numbers came in and
    like a degradation after" is the most interesting thing this table holds, and an update in
    place destroys it — §8.7's rule, one item over.
    """
    import store
    return [_public(r) for r in store.drift_classifications(conn, campaign_id,
                                                            subject=subject)]


def to_revisit(conn, campaign_id: str) -> list:
    """Drift somebody judged `too_early`, now that there is a result (§9.4).

    The classification offer fires when the photographs land, which is normally BEFORE the
    numbers — so at that moment `too_early` is the only honest answer, and without asking again
    it is a one-way sink that absorbs exactly the answer this feature exists to collect.
    """
    import actions

    waiting = [i for i in for_campaign(conn, campaign_id)["items"]
               if i["classification"] == "too_early"]
    if not waiting:
        return []
    first = waiting[0]
    return [{
        "subject": first["subject"], "about": first["about"],
        "was_said": first["why"],
        "what_it_means": (
            f"{len(waiting)} difference{'s' * (len(waiting) != 1)} between this brief and what "
            f"ran {'were' if len(waiting) != 1 else 'was'} judged too early to call, before "
            f"there was a result. There is one now."),
        "next_actions": actions.trim([actions.action(
            f"Now the numbers are in, say whether {first['about']} mattered",
            "classify_drift",
            why=f"It was left as too_early because nothing had been measured — "
                f"\u201c{first['was_said'] if 'was_said' in first else first['why']}\u201d.",
            consent="ask",
            needs=["classification — improvement, neutral or degradation",
                   "why — what makes it that", "classified_by — whose judgment this is"],
            campaign_id=campaign_id, subject=first["subject"])]),
    }]


def for_campaign(conn, campaign_id: str) -> dict:
    """What has been made of this campaign's drift, latest reading per item (§9.4)."""
    import store

    latest: dict = {}
    changed = []
    for row in store.drift_classifications(conn, campaign_id):
        previous = latest.get(row["subject"])
        if previous and previous["classification"] != row["classification"]:
            changed.append((row["item"], previous["classification"], row["classification"]))
        latest[row["subject"]] = row

    items = [{**_public(r), "about": r["item"]} for r in latest.values()]
    counts = {value: sum(1 for i in items if i["classification"] == value)
              for value in CLASSIFICATIONS}
    # Every list in this project is trimmed and this one was not: `compare_execution` embeds it,
    # so five hundred classifications made the comparison a 330 KB response and the caller's
    # context the place it landed. The COUNTS are the figure a reader acts on and they stay
    # complete; the items are the worked examples behind them.
    shown = items[:MAX_ITEMS_SHOWN]
    return {
        "campaign_id": campaign_id,
        "basis": "judged",
        "counts": counts,
        "items": shown,
        "items_total": len(items),
        **({"items_truncated": True} if len(items) > len(shown) else {}),
        "what_it_means": _sentence(counts, changed),
    }


def _sentence(counts: dict, changed: list) -> str:
    if not any(counts.values()):
        return ("Nothing about this campaign's drift has been judged yet. What appeared and "
                "what did not are facts the library can establish; whether any of it was a "
                "loss is not.")
    parts = [f"{count} {value}" for value, count in counts.items() if count]
    said = ("Read as: " + ", ".join(parts) + ". These are judgments, not measurements — the "
            "library records them and does not make them.")
    for item, was, now in changed:
        said += (f" One reading changed: “{item[:40]}” was read as {was} and is now read as "
                 f"{now}. Both are on file with what was known at the time, because that pair "
                 f"is worth more than either on its own.")
    return said
