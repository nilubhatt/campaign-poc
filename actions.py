"""
What to offer next (§5.2, idea B).

    "An evaluation finishes and nothing suggests what follows — store the brief as a record,
    mark it as superseding the version it revises, schedule the reconciliation for when
    results land. Those are the only three things anyone does next, and all three are
    currently things you have to know to ask for.

    Fix: return `next_actions` with each result: a short list of {label, tool,
    prefilled_args}. The surface renders them as suggestions; the user says yes."

  label           what a person is being offered, in their words.
  why             the library fact that prompted it. An offer with no reason is one nobody
                  can refuse intelligently.
  tool            what Claude calls if they accept.
  consent         `ask` before writing a record; `do` for recovery that needs no permission.
                  Three rules were in play for one action before this existed: `next_step`
                  said "act on it, never read it out", `next_actions` said "only if the user
                  accepts", and finish_indexing's own docstring says "do not stop to ask".
                  Writing a new record and resuming an interrupted index are not the same
                  kind of act.
  prefilled_args  arguments this library already holds. Never a guess and never a
                  placeholder: the user says yes to the label, so anything filled in here is
                  something they agreed to without being shown it.
  needs           what the user must still supply. Present only when accepting is not in
                  fact one step.

Two rules, and most of the tests are about them:

**An offer has to work.** A suggestion whose argument name has drifted is one the user
accepts and Claude then cannot carry out — an error they did not cause and cannot fix. The
tests check every offered tool exists and every prefilled argument is one it takes.

**An offer has to be worth making.** A list that is always there stops being read, and then
the one that mattered is not read either. An empty list is a real answer. Three is the
ceiling, which is the number the review itself named.

Nothing here schedules anything, because the product has no scheduler. "Schedule the
reconciliation" becomes the reconciliation call itself, prefilled, labelled for when the
numbers arrive — which is the honest version of the same offer.
"""
from __future__ import annotations

import json

from typing import Optional

# "a short list". Past three, a reader is choosing from a menu rather than being offered a
# next step, and the useful one is no more visible than it was with none at all.
MAX_ACTIONS = 3


def action(label: str, tool: str, *, why: str = "", consent: str = "ask",
           needs: Optional[list] = None, **prefilled_args) -> dict:
    """One offer. Arguments whose value is unknown are dropped rather than sent empty: a
    prefilled blank is a placeholder the user cannot see and did not agree to."""
    assert consent in ("ask", "do"), consent
    offer = {
        "label": label,
        "why": why,
        "tool": tool,
        "consent": consent,
        "prefilled_args": {k: v for k, v in prefilled_args.items()
                           if v is not None and v != ""},
    }
    # "Accepting is one step, not a form" holds only while the prefilled arguments are
    # enough. Where they are not, say what is still missing — otherwise the call goes out
    # with what it was given, which is how `add_metrics(campaign_id=...)` came to store a
    # content-free row that then satisfied the gate for a `verified` tag.
    if needs:
        offer["needs"] = list(needs)
    return offer


def trim(offers: list[dict]) -> list[dict]:
    """Drop the empties, drop exact repeats, and keep the list short enough to read.

    Repeats are keyed on the tool AND its arguments, so two offers of the same tool with
    different arguments both survive — the earlier docstring said "de-duplicate by tool",
    which is not what the code does and would have been wrong if it were.
    """
    seen: set[str] = set()
    kept = []
    for offer in offers:
        if not offer:
            continue
        # Serialised, not tupled. A prefilled argument can be a LIST — `tags=[{...}]` is one
        # this module builds itself, two functions down — and a tuple containing a list is
        # unhashable, so an offer carrying one crashed the deduplication rather than being
        # deduplicated. It survived because nothing had yet produced two offers alongside one.
        key = json.dumps([offer["tool"], offer.get("prefilled_args") or {}],
                         sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        kept.append(offer)
    return kept[:MAX_ACTIONS]


# ── after a judgment ────────────────────────────────────────────────────────

def after_evaluation(*, subject_title: str, evaluation_id: str, verdict: str,
                     campaign_id: Optional[str] = None) -> list[dict]:
    """
    One offer, not three, and the two that were dropped were both wrong.

    **Supersession is not offered at all.** It was prefilled from
    `closest_precedent.campaign_id` — a similarity match, asserted by the model rather than
    computed by the server. "Mark this as replacing the version it revises" is a claim about
    somebody's INTENT; a similarity score is a fact about text. The user hears only the
    label ("add this to the library") while agreeing, unseen, to hide another record from
    every future search — and `update_campaign` cannot clear `supersedes`, so a wrong yes is
    undoable only by deleting the campaign. The gate was also inverted: it suppressed the
    offer on approvals and permitted it on REJECTS, so rejecting a proposal offered to file
    it as the replacement of the concluded campaign it happened to resemble, metrics and
    all. §6.3 revisits it on evidence the schema actually has — a non-empty `resolved`, or a
    record that already carries `supersedes` (tracker D41).

    **Reconciliation is not offered here either.** "When the results land, reconcile this"
    was offered the instant the judgment was saved: the user says yes and the call fails,
    because there are no actuals yet. An offer with a temporal precondition is a reminder,
    and this product has no reminder channel. It is offered where the numbers actually
    arrive instead — see `after_metrics`.
    """
    if campaign_id:
        return []
    # "Review and add", not "Add": `upload_campaign` previews unless confirm=True, and the
    # preview IS the consent step for a write. A label promising storage describes the call
    # after the one being offered.
    return trim([action(
        f"Review and add \u201c{subject_title}\u201d to the library, so later briefs can "
        f"be compared against it",
        "upload_campaign",
        why="This judgment was made about a brief that is not itself a record, so nothing "
            "later can be compared against it.",
        consent="ask", title=subject_title)])


def offer_the_queue(waiting: int) -> list[dict]:
    """§10.6: "the menu is worthless if the user has to know it exists".

    Surfaced wherever it is cheap — after any upload or judgment, and on the first interaction
    of a session. Same instinct as `gaps()`: the library knows what it is missing and should
    say so rather than being interrogated.

    Silent at zero, because an offer that is always there stops being read — the rule every
    other offer in this product follows.
    """
    if waiting < 1:
        return []
    return [action(
        f"{waiting} campaign{'s' * (waiting != 1)} "
        f"{'are' if waiting != 1 else 'is'} waiting on feedback — clear a few?",
        "feedback_queue",
        why="Feedback is the only input that makes this library worth anything, and it is "
            "the hardest thing to give: the user has to remember what is outstanding, name "
            "it, and compose prose. The queue makes it a numbered choice.",
        consent="ask")]


def after_metrics(*, campaign_id: str, open_evaluation_id: Optional[str]) -> list[dict]:
    """Results have just been recorded. If a judgment about this campaign is still open, the
    precondition for reconciling it is now satisfied — which is what makes this the right
    moment for the offer, and the save-time version wrong."""
    if not open_evaluation_id:
        return []
    return trim([action(
        "Check the judgment made about this campaign against what actually happened",
        "reconcile_evaluation",
        why="A judgment was recorded for this campaign and has never been compared with "
            "its results.",
        consent="ask", evaluation_id=open_evaluation_id)])


def after_upload(*, campaign_id: str, status: Optional[str],
                 has_metrics: bool, earlier_judgment: Optional[dict] = None,
                 commentary: Optional[list] = None, title: str = "",
                 has_window: bool = True,
                 unlinked_judgment: Optional[dict] = None,
                 waiting: int = 0) -> list[dict]:
    """After a record lands.

    One thing is worth offering, and only sometimes: a concluded campaign with no outcome
    data is the largest gap this library can have, because every judgment that later cites
    it rests on nothing measured.

    There is deliberately no "send the file to capture its comments" offer, even though
    `commentary_checked: false` says exactly when it would help. No tool attaches a deck to
    an EXISTING record — `update_campaign` takes no `asset_ref`, and `upload_image_asset`
    takes an image — so the only route is a second upload, which duplicates the campaign.
    Offering a call that cannot be made is the failure this module's tests exist to catch;
    the missing tool is tracked instead (D39).

    §8.6 added the MIRROR of that failure, which had no test: a call that can be made and is
    never offered. `note_correction` was referenced nowhere in the product but its own
    definition — a tool the model would have to know existed and spontaneously decide to call,
    which is §8.3's unreachable human step one stage earlier and strictly worse, because
    nothing would ever be RECORDED for the gate to act on. The input was already in hand:
    §2.5 ingests tracked client comments with author, anchor and date, which is the whole of
    what a correction needs.
    """
    offers = []
    # §6.3, and FIRST: a record that replaces a judged one is the one moment where "was our
    # judgment any good?" can actually be answered, and the answer is gone as soon as the
    # person moves on. It ends in `save_reconciliation` by way of `reconcile_evaluation`,
    # which is a tool that exists and works and has never once been called.
    if earlier_judgment:
        # `save_reconciliation`, NOT `reconcile_evaluation` — and this was wrong in the first
        # version in the exact way §5.2 records one field over. `reconcile_evaluation` looks
        # up `metric_type='actual'` rows on the record being replaced, which is a BRIEF that
        # never ran, so accepting the offer returned "no actual metrics on file" and told the
        # model to record results for a proposal. An offer whose `why` says "this is the
        # moment somebody can say" and which then refuses to hear the answer is worse than no
        # offer. Everything `reconcile_evaluation` would have fetched is already in
        # `earlier_judgment`, so the step it adds is the step that breaks.
        offers.append(action(
            f"Record whether the library was right about "
            f"\u201c{earlier_judgment['title']}\u201d, now that this version shows",
            "save_reconciliation",
            why="This replaces a record the library already judged, and nothing has yet "
                "recorded whether that judgment was right. This is the moment somebody can "
                "say.",
            consent="ask",
            needs=["what the user says actually happened — which of the predictions and "
                   "findings held, in their words"],
            # D85: this offer exists because a LATER VERSION arrived, not because results
            # came in. Prefilling the basis keeps that distinction in the record — otherwise
            # "v2 shows the structure came back" sits in the same column as a CTR figure.
            evaluation_id=earlier_judgment["evaluation_id"],
            basis="superseding_version"))
    if status == "concluded" and not has_metrics:
        offers.append(action(
            "Record what this campaign actually achieved, so later judgments can weigh it",
            "add_metrics",
            why="This campaign is finished and has no results on file, so every judgment "
                "that cites it rests on nothing measured.",
            consent="ask", needs=["the results themselves — CTR, ROI, conversions, or "
                                  "whatever was measured"],
            campaign_id=campaign_id))
    # §9.6: the item's headline is "no one should have to remember to connect them", and it
    # is true — the link is computed. What it traded for is a WINDOW, and a window nothing
    # asks for is the unreachable human step §8.3 and §8.6 each hit once already. This is the
    # moment somebody is present and thinking about the campaign; a second deliberate
    # `update_campaign` call months later is not.
    if status == "concluded" and not has_window:
        offers.append(action(
            "Say when this campaign ran, so it can be checked against what else was going on",
            "update_campaign",
            why="Without a window nothing can be matched to the calendar, so “this "
                "launch overlapped Ramadan” and “the port was shut for half of "
                "it” are findings the library cannot produce about it — ever, silently.",
            consent="ask",
            needs=["starts_on — the first day it ran (YYYY-MM-DD)",
                   "ends_on — the last day (YYYY-MM-DD)"],
            campaign_id=campaign_id))
    # §9.9/D40: a judgment was made about this title before it was a record, and nothing could
    # join them — so "judge the pitch, then store it, then check what happened" could not
    # complete from the commonest starting point there is.
    if unlinked_judgment:
        offers.append(action(
            f"Attach the judgment already made about \u201c{title[:40]}\u201d to this record",
            "link_evaluation",
            why="That judgment is about a proposal nobody had stored, so nothing can ever "
                "check it against results. Attaching it is what makes the loop closable.",
            consent="ask",
            needs=["linked_by — whose call it is that these are the same thing"],
            evaluation_id=unlinked_judgment["id"], campaign_id=campaign_id))
    offers += _note_what_the_client_said(campaign_id, commentary, title)
    # §10.6: last, because it is about the LIBRARY rather than about this record — the things
    # above are what this upload specifically needs.
    offers += offer_the_queue(waiting)
    return trim(offers)


# A speaker note is the agency talking to itself; a tracked comment or an annotation is
# somebody reviewing the work, and that is where a standing correction comes from. Filtering
# rather than offering everything, because a deck's own presenter notes would fill the list
# with the agency's own words and teach the model that corrections mean "anything written
# anywhere in the file".
_CLIENT_KINDS = ("comment", "annotation")
# At most this many, however annotated the deck. `trim` caps the whole list anyway, and a
# returned deck can carry forty comments — offering forty is a menu nobody reads, and it would
# crowd out the metrics offer above, which is the larger gap.
_MAX_CORRECTION_OFFERS = 2


def _note_what_the_client_said(campaign_id: str, commentary: Optional[list],
                               title: str) -> list[dict]:
    """Offer to record a tracked client comment as a correction in the making (§8.6).

    The provenance is assembled here and not left to the model: author, anchor and deck are
    all on the row already, and a provenance the model composes is one it can compose wrongly.
    The TEXT stays the client's own words — `needs` says so, because generalising "this looks
    like a rule" into a rule they did not state is the invented evidence this product exists
    to stop.
    """
    out = []
    for row in (commentary or []):
        if len(out) >= _MAX_CORRECTION_OFFERS:
            break
        if (row.get("kind") or "") not in _CLIENT_KINDS:
            continue
        said = (row.get("text") or "").strip()
        if not said:
            continue
        where = ", ".join(part for part in (
            row.get("author"), row.get("anchor"), f"on “{title}”" if title else None,
            row.get("date")) if part)
        out.append(action(
            f"Record “{said[:70]}” as a standing correction in the making",
            "note_correction",
            why="Client feedback that recurs across markets becomes a rule every brief is "
                "judged against — but only once it recurs and somebody confirms it. Recording "
                "it now is what lets that be noticed later. Nothing is applied to any brief "
                "yet.",
            consent="ask",
            needs=["text — the rule in the client's own words, not a generalisation of them"],
            campaign_id=campaign_id, provenance=where or "tracked comment"))
    return out


def after_graduation(*, what: str, markets: list) -> list[dict]:
    """After a measure or a rule joins the checklist (§8.7).

    This is the exact instant the replay becomes non-empty, and it was the only instant
    nothing pointed at it. `replay_rules` was referenced nowhere in the product but its own
    definition — the third time this phase has shipped a tool the model would have to know
    existed and spontaneously call (§8.3's gate, §8.6's `note_correction`, and this).
    """
    where = ", ".join(markets) or "these markets"
    return trim([action(
        f"See which stored campaigns and judgments \u201c{what[:50]}\u201d now applies to",
        "replay_rules",
        why=f"Briefs in {where} are checked against it from here on. Nothing already on file "
            f"is rewritten — this reports which records are missing it and which judgments "
            f"were written before it, so they can be chased or judged again.",
        consent="ask")])


def after_delivered_asset(*, campaign_id: str, briefed: int, promises: int = 0) -> list[dict]:
    """A photograph of what ran has just landed (§9.2/§9.3).

    The moment BOTH checks stop being empty, and the moment nothing pointed at either — the
    fourth and fifth times this project would have shipped a tool the model has to know exists
    and spontaneously call (D116). Each offered only when there is something to compare
    against, because a comparison with one side missing is not a comparison.
    """
    offers = []
    if promises:
        # First: the review calls this "the richest signal", and it is the one a marketer can
        # act on line by line.
        offers.append(action(
            "Check what the brief promised against the photographs",
            "check_commitments",
            why=f"The brief names {promises} specific thing{'s' * (promises != 1)} — this "
                f"says which of them are visible in the photographs that came back, and which "
                f"are not visible in them.",
            consent="ask", campaign_id=campaign_id))
    if briefed:
        offers.append(action(
            "See what actually ran against what was briefed",
            "compare_execution",
            why=f"This campaign has {briefed} briefed image{'s' * (briefed != 1)} and now has "
                f"photographs of what happened. The comparison says which briefed elements "
                f"appeared, which did not, and which arrived unbriefed.",
            consent="ask", campaign_id=campaign_id))
    return trim(offers)


def to_first_upload() -> list[dict]:
    """The one thing an empty library needs. In one place because `gaps()` and `coverage()`
    both offer it and had the sentence written out twice — two sources of truth for one
    offer, which is how they drift.

    `needs` names the title as well as the deck: accepting the offer literally, with only the
    prefilled arguments, raised `TypeError: missing 'title'` — and §5.2's promise is that
    accepting is one step, so where it is not, `needs` has to be complete.
    """
    return trim([action(
        "Add a campaign you were happy with, and one you were not",
        "upload_campaign",
        why="Two contrasting records is the smallest library that can produce a useful "
            "judgment.",
        consent="ask",
        needs=["what the campaign is called",
               "its deck, or a description of what it was"])])


def to_finish_indexing(campaign_id: Optional[str]) -> list[dict]:
    """The offer behind every "this is only partly searchable" warning (tracker D16).

    `consent: "do"` — this is recovery, not a new decision. It re-embeds rows that already
    exist, changes nothing the user did not already ask for, and `finish_indexing`'s own
    docstring says to keep going without stopping to ask.
    """
    return trim([action("Finish indexing this campaign — nothing needs re-uploading",
                        "finish_indexing",
                        why="Part of this upload is stored but not yet searchable.",
                        consent="do", campaign_id=campaign_id)])
