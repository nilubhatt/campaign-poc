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
    seen: set[tuple] = set()
    kept = []
    for offer in offers:
        if not offer:
            continue
        key = (offer["tool"], tuple(sorted(offer["prefilled_args"].items(),
                                           key=lambda kv: kv[0])))
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
                 has_metrics: bool, earlier_judgment: Optional[dict] = None) -> list[dict]:
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
