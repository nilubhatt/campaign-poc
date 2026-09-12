"""
What to offer next (§5.2, idea B).

    "An evaluation finishes and nothing suggests what follows — store the brief as a record,
    mark it as superseding the version it revises, schedule the reconciliation for when
    results land. Those are the only three things anyone does next, and all three are
    currently things you have to know to ask for.

    Fix: return `next_actions` with each result: a short list of {label, tool,
    prefilled_args}. The surface renders them as suggestions; the user says yes."

  label           what a person is being offered, in their words.
  tool            what Claude calls if they accept.
  prefilled_args  arguments this library already holds. Never a guess and never a
                  placeholder: the user says yes to the label, so anything filled in here is
                  something they agreed to without being shown it.

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


def action(label: str, tool: str, **prefilled_args) -> dict:
    """One offer. Arguments whose value is unknown are dropped rather than sent empty: a
    prefilled blank is a placeholder the user cannot see and did not agree to."""
    return {
        "label": label,
        "tool": tool,
        "prefilled_args": {k: v for k, v in prefilled_args.items()
                           if v is not None and v != ""},
    }


def trim(offers: list[dict]) -> list[dict]:
    """Drop the empties, de-duplicate by tool, and keep the list short enough to read."""
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


# ── the three the review named, after a judgment ────────────────────────────

def after_evaluation(*, subject_title: str, evaluation_id: str, verdict: str,
                     campaign_id: Optional[str] = None,
                     supersedes: Optional[str] = None) -> list[dict]:
    """
    `campaign_id` is set when the thing judged is already a record; `supersedes` is the
    earlier version this one replaces, when there is one.

    The offers differ by verdict on purpose. An approval of a new brief supersedes nothing,
    and offering it anyway is how a suggestion list turns into a menu.
    """
    offers = []
    if not campaign_id:
        offers.append(action(
            f"Add “{subject_title}” to the library so later briefs can be "
            f"compared against it",
            "upload_campaign", title=subject_title,
            supersedes=supersedes if verdict != "approve" or supersedes else None))
    elif supersedes:
        offers.append(action(
            "Mark this as replacing the version it revises, so the older one stops being "
            "cited as current",
            "update_campaign", campaign_id=campaign_id, supersedes=supersedes))
    offers.append(action(
        "When the results land, reconcile this judgment against them",
        "reconcile_evaluation", evaluation_id=evaluation_id))
    return trim(offers)


def after_upload(*, campaign_id: str, status: Optional[str],
                 has_metrics: bool) -> list[dict]:
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
    if status == "concluded" and not has_metrics:
        offers.append(action(
            "Add what this campaign actually achieved, so later judgments can weigh it",
            "add_metrics", campaign_id=campaign_id))
    return trim(offers)


def to_finish_indexing(campaign_id: Optional[str]) -> list[dict]:
    """The offer behind every "this is only partly searchable" warning (tracker D16)."""
    return trim([action("Finish indexing this campaign — nothing needs re-uploading",
                        "finish_indexing", campaign_id=campaign_id)])
