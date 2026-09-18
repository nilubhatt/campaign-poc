"""
Creative: the images, what they show, and how far execution moved from the brief.

Three sections lifted out of `core` unchanged — image assets and creative-reuse detection
(§6.6), what shipped and how far it moved (§9.2), and the secondary asset-resolution path.
They were already one cluster: almost every name below is called by another name below, and
between them they reached back into the rest of `core` for three things.

`core` re-exports every name here, so `core.compare_execution`, `core._snapshot_execution_drift`
and the rest keep working for the callers and tests that already say them. `core` is imported
INSIDE the functions that need it, because `core` imports this module and a module-level import
back would be a cycle.
"""
from __future__ import annotations

from diffing import diff_campaigns
from pathlib import Path
from typing import Optional
import base64
import binascii
import re
import sys
import time

import actions
import clip_embed
import commitments
import config
import drift
import embedding
import enums
import extract
import facts
import identity
import images
import notices
import store
import vectorstore
import version

# ── image assets / creative-reuse detection (§6.6) ───────────────────────────

def ingest_image_assets(conn, *, campaign_id: str, asset_refs: list,
                        phase: str = "proposed",
                        captured_on: Optional[str] = None) -> dict:
    """Several images in one call (§9.2).

    Fourteen photographs from an event were fourteen uploads and fourteen tool calls, each
    needing its own `phase="delivered"` — and `ingest_campaign`'s own comment already records
    that a manual call per image "isn't a workflow anyone would actually use". The obvious
    workaround is worse: dropping the photos into a wrap deck files all fourteen as `proposed`,
    which inflates the brief and drags the comparison toward zero drift by construction.

    Each image is independent: one that cannot be resolved is reported and does not stop the
    rest, which is `bulk_import_metrics`' rule for the same reason.
    """
    stored, failed = [], []
    for ref in asset_refs or []:
        result = ingest_image_asset(conn, campaign_id=campaign_id, asset_ref=ref,
                                    phase=phase, captured_on=captured_on)
        (failed if result.get("error") else stored).append(result)
    offers = (actions.after_delivered_asset(
        campaign_id=campaign_id,
        briefed=len(store.assets_in_phase(conn, campaign_id, "proposed")),
        promises=len(store.commitments_for(conn, campaign_id)))
        if stored and phase == "delivered" else [])
    return {"campaign_id": campaign_id, "phase": phase, "captured_on": captured_on,
            "stored": len(stored), "assets": stored,
            **({"failed": failed} if failed else {}),
            "next_actions": offers,
            "what_it_means": (
                f"{len(stored)} image{'s' * (len(stored) != 1)} attached as {phase}"
                + (f"; {len(failed)} could not be read" if failed else "") + ".")}


def ingest_image_asset(conn, *, campaign_id: str, asset_ref: dict,
                       phase: str = "proposed", captured_on: Optional[str] = None) -> dict:
    """Attach an image to a campaign and process it two ways: a perceptual hash (exact/
    near-duplicate reuse detection, §6.6) and a CLIP visual embedding (aesthetic/regional
    similarity, the heavier follow-on). Storage always succeeds even if one or both
    processing steps fail (e.g. a corrupt image, or CLIP unavailable) — failures are
    reported per-step, not swallowed, and don't block each other.

    §9.1: `phase` says what this image IS — `proposed` creative lifted from a brief, or a
    `delivered` photograph that came back after the event. `captured_on` is when the
    photograph was taken, which belongs to `delivered` and nothing else.
    """
    if store.get_campaign(conn, campaign_id) is None:
        return {"error": f"campaign {campaign_id} not found"}

    phase = enums.normalise(phase, field="phase", valid=store.VALID_ASSET_PHASES,
                            synonyms=enums.ASSET_PHASE_SYNONYMS, allow_none=False)
    captured_on = _asset_date(captured_on, phase)

    path, warnings = _resolve_asset(asset_ref)
    if not path:
        return {"error": "; ".join(w["detail"] for w in warnings)
                         or "could not resolve asset_ref",
                "warnings": notices.collapse(warnings)}

    stored_name = _keep_asset(path)
    result = _store_and_fingerprint_image(conn, campaign_id, stored_name, phase=phase,
                                          captured_on=captured_on)
    if phase == "delivered":
        # The photographs are the other half of the answer, and they usually arrive after the
        # results (§9.5).
        _snapshot_execution_drift(conn, campaign_id)
    result["warnings"] = warnings + result["warnings"]
    return result


def _asset_date(captured_on: Optional[str], phase: str) -> Optional[str]:
    """When the photograph was taken (§9.1).

    Refused on a `proposed` asset rather than ignored: a capture date on briefed creative says
    a photograph exists of something that has not happened, and a stored date nothing reads is
    a fact somebody will later believe.
    """
    if not (captured_on or "").strip():
        return None
    if phase != "delivered":
        raise ValueError(
            f"`captured_on` belongs to a delivered asset — it says when the photograph was "
            f"taken, and this one is {phase!r}, which is creative from a brief. Either it "
            f"came back after the event (pass phase='delivered') or it has no capture date.")
    import datetime

    text = captured_on.strip()
    try:
        parsed = datetime.date.fromisoformat(text)
    except ValueError:
        raise ValueError(
            f"`captured_on` must be a date as YYYY-MM-DD; got {text!r}. A date this library "
            f"cannot place on a calendar cannot be compared with a campaign's window, which "
            f"is the only thing it is for.")
    # A photograph cannot have been taken tomorrow, and a date in the future is a typo that
    # would otherwise sit in the record looking like a fact.
    if parsed > datetime.date.today():
        raise ValueError(
            f"`captured_on` is {text}, which is in the future — a photograph of what ran "
            f"cannot have been taken yet. Check the year.")
    # Stored normalised, because `fromisoformat` also accepts `20260314` and `2026-W11-5`, and
    # a later comparison against a campaign window is a string comparison.
    return parsed.isoformat()


def _store_and_fingerprint_image(conn, campaign_id: str, stored_name: str, *,
                                 phase: str = "proposed",
                                 captured_on: Optional[str] = None) -> dict:
    """The part of ingest_image_asset that runs once the image file is already saved under
    ASSET_DIR. NOT used by the deck-embedded-image path in ingest_campaign — that path also
    computes reuse_flags inline (via the shared _phash_matches below) as part of the same
    loop, which this helper doesn't do, so it re-implements the fingerprint/CLIP-embed steps
    rather than call this and bolt reuse-checking on after."""
    import core

    full_path = config.ASSET_DIR / stored_name
    warnings: list[dict] = []
    aid = store.insert_asset(conn, campaign_id, file_path=stored_name, phase=phase,
                             captured_on=captured_on)

    fingerprinted = False
    try:
        h = images.phash(full_path)
        store.set_asset_fingerprint(conn, aid, h)
        fingerprinted = True
    except Exception as exc:
        warnings.append(notices.notice(
            "image_not_fingerprinted",
            detail=f"asset stored but not fingerprinted (reuse detection will miss it): "
                   f"{exc}"))

    visually_embedded = False
    try:
        vec = clip_embed.embed_image(full_path)
        core._add_vector(conn, aid, vec, space="asset")
        store.mark_asset_embedded(conn, aid)
        visually_embedded = True
    except Exception as exc:
        # Which warning this is depends on WHY: the model being absent from the machine is
        # the operator's problem to fix once, while one image failing on a working model is
        # this record's problem. Telling a marketer to go and find an administrator over a
        # single corrupt PNG would be the same mistake in the other direction.
        warnings.append(core._vision_notice(
            f"asset stored but not visually embedded (aesthetic similarity will miss it): "
            f"{exc}. Run finish_indexing to complete it — the image is saved, so it does "
            f"not need uploading again."))

    return {"asset_id": aid, "campaign_id": campaign_id, "fingerprinted": fingerprinted,
            "visually_embedded": visually_embedded, "phase": phase,
            "captured_on": captured_on,
            # §9.2, offered where it becomes answerable rather than left to be discovered.
            "next_actions": actions.after_delivered_asset(
                campaign_id=campaign_id,
                briefed=len(store.assets_in_phase(conn, campaign_id, "proposed")),
                promises=len(store.commitments_for(conn, campaign_id)))
            if phase == "delivered" else [],
            "warnings": notices.collapse(warnings)}



# ── §9.2: what shipped, and how far it moved ────────────────────────────────

def compare_execution(conn, *, campaign_id: str) -> dict:
    """What actually ran against what was briefed (§9.2).

    *"Fingerprints answer what shipped: delivered assets that match a proposed one are as
    briefed; proposed assets with no match never appeared; delivered assets matching nothing
    are new. Visual similarity answers how far it moved: the distance between the proposed and
    delivered creative sets is a drift score, not a vibe."*

    **Two instruments, two questions, kept apart.** A fingerprint is an IDENTITY claim — this
    photograph is that render — and it is exact enough to state. A CLIP distance is a claim
    about resemblance, which is a number and not a verdict. Reporting one as the other is how
    "the creative changed" becomes an assertion nobody can check.

    The plan's signature is `compare_execution(campaign_id, delivered_assets[])`, and this
    takes only the campaign. §9.1 said everything falls out of the phase field, and it does:
    once an asset carries its phase, which images are the brief and which came back is already
    on file, and a second inline upload path would be a second way to attach an image with its
    own resolution rules and its own bugs.

    What it will not say is what the drift MEANS. Improvement, neutral or degradation is
    §9.4's question and is marked `judged` for a reason — a cosine cannot know which, and a
    server that guessed would be asserting a verdict from a distance.
    """
    record = store.get_campaign(conn, campaign_id)
    if record is None:
        raise ValueError(f"{campaign_id!r} is not a record in this library.")

    briefed = store.assets_in_phase(conn, campaign_id, "proposed")
    delivered = store.assets_in_phase(conn, campaign_id, "delivered")
    warnings: list = []

    if not delivered:
        return _nothing_to_compare(
            campaign_id, briefed, delivered,
            f"{len(briefed)} briefed image{'s' * (len(briefed) != 1)} on file and nothing "
            f"has come back, so what ran has never been checked against what was asked for.",
            title=record["title"],
            offers=actions.trim([actions.action(
                f"Add the photographs from \u201c{record['title']}\u201d",
                "upload_image_asset",
                why="With them on file this comparison says which briefed elements appeared, "
                    "which did not, and which arrived unbriefed.",
                consent="ask", needs=["the photographs themselves"],
                campaign_id=campaign_id, phase="delivered")]))
    if not briefed:
        # §12.4/D123: this is also what a batch filed with the wrong PHASE looks like — every
        # image on the record came in as `delivered`, so the comparison has nothing on the
        # other side. Offered here rather than as a footer on every upload, because this is
        # the moment the mistake becomes visible: a comparison with an empty half.
        return _nothing_to_compare(
            campaign_id, briefed, delivered,
            f"{len(delivered)} delivered image{'s' * (len(delivered) != 1)} on file and "
            f"nothing briefed to compare them against — there was never a brief to drift "
            f"from, so calling them all new would say more than is known.",
            title=record["title"],
            offers=actions.trim([actions.action(
                "Correct the phase, if some of these are the briefed creative",
                "update_asset",
                why="Every image on this record is filed as what RAN. If some of them came "
                    "out of the deck, they are the thing execution is compared against — and "
                    "with all of them on one side there is nothing to compare.",
                consent="ask",
                needs=["which images are the briefed creative", "who is correcting it",
                       "why"],
                phase="proposed")]))

    prints = store.asset_fingerprints(conn, [a["id"] for a in briefed + delivered])
    # A hash with no structure in it matches every other image with no structure, at distance
    # zero — so a solid-fill rectangle lifted out of a deck would be reported as built on the
    # strength of a blank wall in a delivered photograph. Dropped from the comparison and SAID,
    # rather than silently carried.
    flat = {aid for aid, value in prints.items() if images.is_degenerate(value)}
    prints = {aid: value for aid, value in prints.items() if aid not in flat}
    missing = [a for a in briefed + delivered if a["id"] not in prints]
    if missing:
        # §2.1: an image that could not be hashed cannot be matched, and leaving it out of all
        # three lists silently would make the totals lie about what was examined.
        no_print = [a for a in missing if a["id"] not in flat]
        reasons = []
        if no_print:
            reasons.append(f"{len(no_print)} {'have' if len(no_print) != 1 else 'has'} no "
                           f"fingerprint (run finish_indexing)")
        if flat:
            reasons.append(f"{len(flat)} {'are' if len(flat) != 1 else 'is'} a flat fill or "
                           f"blank frame, which carries no structure to match on")
        warnings.append(notices.notice(
            "assets_not_fingerprinted",
            detail=f"{len(missing)} image{'s' * (len(missing) != 1)} could not be compared: "
                   + "; ".join(reasons) + "."))

    drift_score, drift_warnings = _drift_score(conn, briefed, delivered)
    warnings += drift_warnings
    judged = drift.for_campaign(conn, campaign_id)
    as_briefed, never_appeared, new, another_view = _match_by_fingerprint(
        [a for a in briefed if a["id"] in prints],
        [a for a in delivered if a["id"] in prints], prints)
    # A fingerprint answers "is this the same FILE, resized or recompressed" — that is what
    # `images.py`'s own docstring scopes it to, and what its regression baseline verified. A
    # photograph of a physical execution, shot on site at an angle under different light, will
    # not land within Hamming 8 of the briefed render. So on the review's own motivating case —
    # a claw machine that was actually built — pHash alone reports `0 as_briefed, 3 never
    # appeared, 14 new` and stamps `computed` on it.
    #
    # The second pass says what resemblance can say and labels it as what it is: `heuristic`,
    # §7.8's third basis, which exists for exactly this — a threshold somebody chose, neither
    # a server fact nor a model's judgment.
    _suggest_visual_matches(conn, never_appeared, new)
    unjudged = _unjudged_drift(conn, campaign_id, never_appeared, new, judged)

    return {
        "campaign_id": campaign_id,
        "title": record["title"],
        "status": "checked",
        "basis": "computed",
        "briefed_count": len(briefed),
        "delivered_count": len(delivered),
        # D60: the same words `diff_campaigns` uses, so "what changed" means one thing at both
        # levels — a `counts` dict keyed by the bucket names, and a per-item `match` saying
        # what the classification rests on.
        "counts": {"as_briefed": len(as_briefed), "never_appeared": len(never_appeared),
                   "new": len(new), "another_view": len(another_view),
                   "not_compared": len(missing)},
        "as_briefed": as_briefed,
        "never_appeared": never_appeared,
        "new": new,
        "another_view": another_view,
        "not_compared": [{"asset_id": a["id"], "file": a["file_path"], "phase": a["phase"]}
                         for a in missing],
        "drift": drift_score,
        # §9.4: what has been made of this drift, and how much of it nobody has judged yet.
        # The server records classifications and makes none — "presence and absence are facts;
        # whether a change was good is a judgment".
        "classified": judged,
        # DIFFERENCES, not assets, and the same list the offer below is drawn from — so
        # answering the question the server asks moves the number it shows. The pair "this was
        # briefed and did not come back / this came back and was not briefed" is one difference
        # a person would judge once.
        "unclassified": len(unjudged),
        "unclassified_items": unjudged,
        "what_it_means": _execution_sentence(as_briefed, never_appeared, new, another_view),
        "next_actions": _drift_offers(conn, campaign_id, unjudged),
        "warnings": notices.collapse(warnings),
    }


def _snapshot_execution_drift(conn, campaign_id: str) -> None:
    """Record how faithfully this campaign ran, beside the result it produced (§9.5).

    Run at every moment the answer can CHANGE — results arriving, photographs arriving, a
    classification being made — because the ordinary sequence is results first and the wrap
    deck weeks later. Snapshotting only on `add_metrics` froze every real campaign at
    `never_checked`: the photographs turned up afterwards and nothing looked again, so §9.5's
    whole point never fired for the common case.

    A snapshot rather than a read-time computation, because it is a fact about the evidence AS
    IT STOOD, and the stored figure is what a judgment saved at the time rested on.
    """
    import core

    record = store.get_campaign(conn, campaign_id)
    # Nothing to say until there is a result for the drift to qualify. This also keeps a bulk
    # import of five hundred rows from running five hundred comparisons: the campaigns in a KPI
    # workbook almost never have delivered photographs.
    if not record or not core.has_results(record):
        return
    try:
        out = compare_execution(conn, campaign_id=campaign_id)
    except Exception:                    # noqa: BLE001 — a result must still file
        # A result filing is more important than its drift figure, so this does not raise. It
        # does not vanish either: swallowed, a comparison that broke left the campaign reading
        # `never_checked` forever, which is indistinguishable from nobody having looked — and
        # §9.5's whole point is that those two are different. The stale figure stays, and the
        # failure is on the record.
        # stderr, not stdout: stdout is the MCP protocol channel on the transport that matters.
        import traceback
        print(f"[campaign-intelligence] execution drift snapshot failed for {campaign_id}; "
              f"the stored figure is unchanged.\n{traceback.format_exc()}",
              file=sys.stderr, flush=True)
        return
    relative = (out.get("drift") or {}).get("relative_to_brief_spread")
    # Determined by the COUNTS, not by the score. A threshold on the relative figure was in
    # here too and the mutation pass showed it was unreachable: every briefed image coming back
    # by fingerprint means the files are the same files, so the sets cannot also be far apart.
    # An untestable branch with a chosen constant in it is exactly the thing that quietly
    # becomes wrong, and "something briefed did not come back, or something unbriefed did" is
    # the fact this status rests on anyway.
    if out["status"] == "nothing_to_check":
        status = "never_checked"
    elif out["counts"].get("never_appeared") or out["counts"].get("new"):
        status = "drifted"
    else:
        status = "as_briefed"
    judged = out.get("classified") or {}
    store.record_execution_drift(
        conn, campaign_id, score=(out.get("drift") or {}).get("score"), relative=relative,
        # `unclassified` beside the image counts, because a reader of the stored figure needs
        # to know how much of this drift nobody has looked at. Without it, a campaign whose
        # single corrected difference sat beside four unexamined ones read as fully corrected.
        counts={**(out.get("counts") or {}), "unclassified": out.get("unclassified", 0)},
        classified=judged.get("counts") or {},
        # §9.5 verbatim: "the score, the counts, and the CLASSIFIED ITEMS". Counts alone cannot
        # say what was judged good, which is the half a reader needs.
        items=[{"about": i.get("about") or i.get("item"),
                "classification": i["classification"], "why": i["why"],
                "classified_by": i["classified_by"], "outcome_known": i["outcome_known"]}
               for i in judged.get("items") or []],
        status=status)


_EXECUTION_MEANING = {
    "as_briefed": ("This campaign ran as it was briefed, so its result is evidence about the "
                   "BRIEF."),
    "drifted": ("This campaign's execution moved away from its brief, so its result is "
                "evidence that something worked and the brief may not have been it. Weigh it "
                "as precedent accordingly — it is not disqualified, and the change may be "
                "exactly why it worked."),
    "never_checked": ("Nobody has checked what this campaign actually ran against what it "
                      "briefed, so whether its result is evidence about the brief is unknown. "
                      "That is not the same as it having run faithfully."),
}


def _execution_note(conn, campaign_id: str) -> dict:
    """The caveat that travels with a citation (§9.5, D19's shape).

    It reads the CLASSIFICATIONS, not only the status. Keyed on status alone, a drift somebody
    named had judged an improvement carried the identical "the brief may not have been it"
    caveat as one judged a degradation — so the UAE claw machine, judged better than the plan
    by a person, was still discounted at the one layer where drift changes a verdict. §9.4
    exists to stop the library punishing improvement, and that is where it was punished.
    """
    stored = store.execution_drift_for(conn, campaign_id)
    # Only the non-zero ones. The counts dict carries a key per classification whatever the
    # numbers are, so `if judged:` was true of a campaign nobody had judged at all.
    judged = {value: count for value, count in (stored["classified"] or {}).items() if count}
    said = _EXECUTION_MEANING[stored["status"]]
    if stored["status"] == "drifted" and judged:
        said = _judged_drift_sentence(judged, stored)
    return {"status": stored["status"], "score": stored["score"],
            "relative_to_brief_spread": stored["relative"],
            # The image counts AND how much of the drift nobody has looked at — a reader
            # weighing "somebody says this matched" has to be able to see what else is sitting
            # unexamined beside it.
            "counts": stored.get("counts") or {},
            "classified": judged, "classified_items": stored.get("items") or [],
            "basis": "computed", "what_it_means": said}


def _reads_as_briefed(note: dict) -> bool:
    """Whether this campaign's result is evidence about its BRIEF (§9.5).

    Two ways to be: the fingerprints matched, or somebody who looked at the photographs said
    the fingerprints were wrong. §9.2's own comments say a photograph of a built claw machine
    will not match the briefed render, so the second route is not an escape hatch — it is the
    only route the motivating case has.
    """
    if note["status"] == "as_briefed":
        return True
    judged = {v: c for v, c in (note.get("classified") or {}).items() if c}
    if not judged or set(judged) != {"not_drift"}:
        return False
    # And nothing left over. One corrected difference out of two does not make the campaign
    # faithful — the second is still a difference the comparison saw and nobody disputed, and
    # counting the campaign as faithful on the strength of the first is the same over-reach in
    # the opposite direction. A missing count is unknown, not zero.
    return (note.get("counts") or {}).get("unclassified") == 0


def _judged_drift_sentence(judged: dict, stored: dict) -> str:
    """What a drift somebody has actually looked at means for a citation (§9.4/§9.5)."""
    improvement = judged.get("improvement", 0)
    degradation = judged.get("degradation", 0)
    neutral = judged.get("neutral", 0)
    too_early = judged.get("too_early", 0)
    not_drift = judged.get("not_drift", 0)
    named = ", ".join(f"{count} {value}" for value, count in judged.items() if count)
    if not_drift and len(judged) == 1:
        return (f"The comparison could not match this campaign's delivered photographs to its "
                f"briefed images, and somebody who looked said they DO match ({named}) — a "
                f"photograph of a thing that was built rarely fingerprints like the render of "
                f"it. Read this as having run as briefed: its result is evidence about the "
                f"BRIEF, and the difference the comparison reported is an artefact of the "
                f"instrument rather than a fact about the campaign.")
    if not_drift:
        return (f"Part of what the comparison read as drift here was disputed by somebody who "
                f"looked — they say it matched and the instrument could not see it ({named}). "
                f"Read the items: some of this campaign's result is evidence about the brief "
                f"as written, and the rest about what was done differently.")
    if improvement and not degradation:
        return (f"This campaign's execution moved away from its brief, and somebody who "
                f"looked read the difference as an improvement ({named}). Its result is "
                f"evidence that the change worked — which may be the most useful thing here, "
                f"and is not a reason to discount it.")
    if degradation and not improvement:
        return (f"This campaign's execution moved away from its brief, and somebody who "
                f"looked read the difference as a degradation ({named}). Its result was "
                f"achieved despite that, so the brief may have been better than the outcome "
                f"suggests.")
    if neutral and not (improvement or degradation):
        return (f"This campaign's execution moved away from its brief and somebody who "
                f"looked read the difference as making no difference ({named}), so its "
                f"result is still largely evidence about the brief.")
    if too_early and len(judged) == 1:
        return (f"This campaign's execution moved away from its brief, and the difference was "
                f"looked at before any result existed ({named}) — so nobody has yet said "
                f"whether it mattered. Worth revisiting now the numbers are in.")
    return (f"This campaign's execution moved away from its brief, and what was made of the "
            f"difference is mixed ({named}). Read the items rather than the headline: part of "
            f"this result may be evidence about the brief and part about the change.")


def _drift_events(never_appeared, new) -> list:
    """The differences a PERSON would count, not the assets they are made of.

    One conceptual event — the claw machine was replaced by something else — appears twice in
    the lists: as a briefed image that did not come back, and as a delivered photograph
    matching nothing. Counting assets asked for seventeen decisions where a human sees three
    or four, and §9.2's own analysis says a photograph of a built thing will rarely
    fingerprint-match its render, so `new` approximates "every wrap photo you uploaded".
    Paired by the visual resemblance the second pass already nominated.
    """
    paired, events = set(), []
    for item in never_appeared:
        looks_like = (item.get("looks_like") or {}).get("asset_id")
        if looks_like:
            paired.add(looks_like)
        events.append({"subject": item["asset_id"],
                       "about": "a briefed image that did not come back",
                       "replaced_by": looks_like})
    events += [{"subject": item["asset_id"],
                "about": "a delivered photograph matching nothing that was briefed",
                "replaced_by": None}
               for item in new if item["asset_id"] not in paired]
    return events


def _unjudged_drift(conn, campaign_id, never_appeared, new, judged) -> list:
    """Every difference between this brief and what ran that nobody has judged yet (§9.4).

    ONE list, feeding both the displayed count and the offer. They were built separately — the
    count over asset-keyed events, the offer over commitments — so the two never referred to the
    same things: answering the question the server asked left its own `unclassified` figure
    exactly where it was, and the next call re-asked it with the same prefilled subject. A nag
    that cannot be satisfied is worse than no offer at all, because the count beside it reads as
    a running tally of unexplained failure.

    Two kinds of difference, because they are found two different ways and a person judges each
    on its own: a promise the deck named that is not visible in the photographs (§9.3), and a
    briefed image that did not come back or a delivered one matching nothing (§9.2).
    """
    already = {i["subject"] for i in judged["items"]}
    unjudged = [
        {"subject": promise["commitment_id"],
         "about": f"“{promise['text'][:60]}” was promised and is not visible",
         "replaced_by": None}
        for promise in _promises_not_visible(conn, campaign_id)
        if promise["commitment_id"] not in already]
    unjudged += [e for e in _drift_events(never_appeared, new) if e["subject"] not in already]
    return unjudged


def _promises_not_visible(conn, campaign_id: str) -> list:
    """The deck's own promises the visual pass could not find (§9.3), as drift to judge.

    `not_visible` only. Offering to judge a promise that WAS visible asks somebody to explain a
    difference that is not there, and a question with no true answer gets a made-up one.
    """
    try:
        report = commitments.check(conn, campaign_id=campaign_id)
    except Exception:                        # noqa: BLE001 — the comparison must still return
        return []
    return [i for i in report.get("items") or [] if i.get("verdict") == "not_visible"]


def _drift_offers(conn, campaign_id, unjudged) -> list:
    """Offer to judge what the server will not (§9.4).

    The library can say a briefed element did not come back. Whether that was a loss is the
    thing it must not guess, and the offer is how the question reaches somebody who can answer
    it — rather than leaving a count of unexplained differences that quietly reads as failure.

    A PROMISE first where there is one, because "a claw machine loaded with branded
    merchandise" is a subject somebody can still read in six months and an asset id is not.
    `_unjudged_drift` already orders them that way.
    """
    if not unjudged:
        return []
    return actions.trim([actions.action(
        f"Say whether {unjudged[0]['about']} was a loss", "classify_drift",
        why="The library can see the execution differed from the brief. Whether that was an "
            "improvement, made no difference or cost something is a judgment it will not "
            "make — and treating every difference as a defect teaches it to punish "
            "anything that went better than planned.",
        consent="ask",
        needs=["classification — improvement, neutral, degradation or too_early",
               "why — what makes it that",
               "classified_by — whose judgment this is"],
        campaign_id=campaign_id, subject=unjudged[0]["subject"])])

def _nothing_to_compare(campaign_id, briefed, delivered, said, *, title="",
                        offers=None) -> dict:
    """One half of the comparison is missing, so there is no comparison (§9.2).

    `score: None` and `nothing_to_check`, never zero. A drift score computed over an empty set
    reads as "the execution matched the brief perfectly" — the confident unfounded claim in its
    purest form, arriving from an empty list rather than from a mistake.
    """
    return {
        "campaign_id": campaign_id, "title": title,
        "status": "nothing_to_check", "basis": "computed",
        # The instruction as an OFFER, not as prose. C12 replaced exactly this shape once
        # already, and this is the best place in the product to make it: the user is looking
        # at the answer "nobody has checked what ran" at the moment they could fix it.
        "next_actions": offers or [],
        "briefed_count": len(briefed), "delivered_count": len(delivered),
        "as_briefed": [], "never_appeared": [], "new": [], "another_view": [],
        "not_compared": [], "classified": {"counts": {}, "items": []}, "unclassified": 0,
        "drift": {"score": None, "code": "nothing_to_compare", "basis": "computed",
                  "what_it_means": "There is nothing to measure a distance between."},
        "what_it_means": said, "warnings": [],
    }


def _match_by_fingerprint(briefed, delivered, prints) -> tuple:
    """The three lists, by pHash. Each item carries the match it rests on.

    **Briefed images are clustered first.** A deck puts the hero visual on the cover and again
    on a detail slide, and `extract` dedupes by sha256 — two different files, one image. Treated
    as two elements, a photograph of it matched one and the other was reported as NEVER
    APPEARED: the list whose entire claim is identity, saying a briefed element was missing when
    it demonstrably was not.

    **And the assignment is global, not greedy in upload order.** Taking each photograph's
    closest unclaimed render in turn made the answer depend on which photo was uploaded first —
    the same four images produced opposite verdicts on the same briefed element. Pairs are
    sorted by distance across the whole set and assigned from the closest, so the result is a
    property of the images rather than of the upload sequence.
    """
    elements = _cluster_briefed(briefed, prints)
    pairs = sorted(
        (images.hamming_distance(prints[element[0]["id"]], prints[shot["id"]]), e, d)
        for e, element in enumerate(elements) for d, shot in enumerate(delivered))

    claimed_element, claimed_shot = {}, {}
    for distance, e, d in pairs:
        if distance > config.PHASH_MATCH_THRESHOLD:
            break
        if e in claimed_element or d in claimed_shot:
            continue
        claimed_element[e] = (d, distance)
        claimed_shot[d] = (e, distance)

    as_briefed, new, another_view = [], [], []
    for d, shot in enumerate(delivered):
        if d in claimed_shot:
            e, distance = claimed_shot[d]
            first = elements[e][0]
            as_briefed.append({
                "asset_id": shot["id"], "file": shot["file_path"],
                "captured_on": shot["captured_on"],
                "matched_asset_id": first["id"], "matched_file": first["file_path"],
                "also_briefed_as": [a["file_path"] for a in elements[e][1:]],
                "distance": distance, "threshold": config.PHASH_MATCH_THRESHOLD,
                "match": "fingerprint",
                "what_it_means": (
                    f"Matches the briefed image {first['file_path']} at a perceptual distance "
                    f"of {distance} (anything up to {config.PHASH_MATCH_THRESHOLD} is the "
                    f"same image).")})
            continue
        echo = _already_matched(elements, claimed_element, prints, shot)
        if echo:
            # A second photograph of something already matched is not NEW — "matches nothing
            # that was briefed" would be false of it. Without this, three photos of one claw
            # machine against a brief promising three things read as "three briefed elements
            # appeared", when one did.
            another_view.append({
                "asset_id": shot["id"], "file": shot["file_path"],
                "captured_on": shot["captured_on"], "matched_asset_id": echo[0],
                "distance": echo[1], "match": "fingerprint",
                "what_it_means": (f"Another photograph of the briefed image {echo[2]}, which "
                                  f"is already accounted for.")})
            continue
        new.append({"asset_id": shot["id"], "file": shot["file_path"],
                    "captured_on": shot["captured_on"],
                    "nearest": _nearest_miss(shot, briefed, prints),
                    "threshold": config.PHASH_MATCH_THRESHOLD, "match": "none",
                    "what_it_means": "Came back, and matches nothing that was briefed."})

    # The nearest miss, on both lists. "§9.2: with the matching evidence attached per item" —
    # `as_briefed` carried its match and these two carried a bare sentence, while the matcher
    # had already computed every distance and thrown the near-misses away. A negative with no
    # number behind it is the assertion this product refuses everywhere else.
    never_appeared = [
        {"asset_id": element[0]["id"], "file": element[0]["file_path"],
         "also_briefed_as": [a["file_path"] for a in element[1:]],
         "nearest": _nearest_miss(element[0], delivered, prints),
         "threshold": config.PHASH_MATCH_THRESHOLD, "match": "none",
         "what_it_means": "Was briefed and nothing that came back matches it."}
        for e, element in enumerate(elements) if e not in claimed_element]
    return as_briefed, never_appeared, new, another_view


def _cluster_briefed(briefed, prints) -> list:
    """Briefed images that are the SAME image, grouped into one element.

    The hero on the cover slide and on a detail slide are one thing the brief promises, and
    counting them twice makes a delivered photograph of it satisfy one and leave the other
    reading as missing.
    """
    elements: list = []
    for asset in briefed:
        for element in elements:
            if images.hamming_distance(prints[element[0]["id"]],
                                       prints[asset["id"]]) <= config.PHASH_MATCH_THRESHOLD:
                element.append(asset)
                break
        else:
            elements.append([asset])
    return elements


def _already_matched(elements, claimed_element, prints, shot):
    """The briefed element this photograph is another view of, if any — the CLOSEST one."""
    scored = [(images.hamming_distance(prints[element[0]["id"]], prints[shot["id"]]), element)
              for e, element in enumerate(elements) if e in claimed_element]
    within = [(d, el) for d, el in scored if d <= config.PHASH_MATCH_THRESHOLD]
    if not within:
        return None
    distance, element = min(within, key=lambda pair: pair[0])
    return element[0]["id"], distance, element[0]["file_path"]


# How close two images have to look before the server will say they MIGHT be the same thing.
# Deliberately high: this is offered beside a `never appeared` verdict, and a wrong suggestion
# there tells a marketer their claw machine turned up when it did not.
_VISUAL_MATCH_CEILING = 0.12


def _nearest_miss(asset, others, prints) -> Optional[dict]:
    """The closest thing this did NOT match, and by how much."""
    scored = [(images.hamming_distance(prints[asset["id"]], prints[o["id"]]), o)
              for o in others if o["id"] in prints and o["id"] != asset["id"]]
    if not scored:
        return None
    distance, closest = min(scored, key=lambda pair: pair[0])
    return {"asset_id": closest["id"], "file": closest["file_path"], "distance": distance}


def _suggest_visual_matches(conn, never_appeared, new) -> None:
    """Pair up what the fingerprints could not, by resemblance, as a SUGGESTION (§9.2).

    Mutates both lists in place, adding `looks_like` where something resembles something. It
    never moves an item between lists: a resemblance is not an identity claim, and promoting
    one to `as_briefed` would put the server's `computed` authority behind a guess about
    whether a thing was built.
    """
    if not never_appeared or not new:
        return
    briefed_vecs, _ = _asset_vectors(conn, [{"id": item["asset_id"]}
                                            for item in never_appeared])
    shot_vecs, _ = _asset_vectors(conn, [{"id": item["asset_id"]} for item in new])
    if not briefed_vecs or not shot_vecs:
        return
    pairs = []
    for i, left in enumerate(briefed_vecs):
        for j, right in enumerate(shot_vecs):
            pairs.append(((1.0 - embedding.cosine(left, right)) / 2.0, i, j))
    taken_left, taken_right = set(), set()
    for distance, i, j in sorted(pairs):
        if distance > _VISUAL_MATCH_CEILING or i in taken_left or j in taken_right:
            continue
        taken_left.add(i)
        taken_right.add(j)
        brief_item, shot_item = never_appeared[i], new[j]
        brief_item["match"] = "visual"
        shot_item["match"] = "visual"
        brief_item["looks_like"] = {
            "asset_id": shot_item["asset_id"], "file": shot_item["file"],
            "distance": round(distance, 6), "basis": "heuristic",
            "what_it_means": (f"No delivered photograph is the same FILE as this briefed "
                              f"image, but {shot_item['file']} looks like it. A fingerprint "
                              f"answers 'same image'; this only answers 'looks similar', so "
                              f"it is a question for somebody who can recognise the thing.")}
        shot_item["looks_like"] = {
            "asset_id": brief_item["asset_id"], "file": brief_item["file"],
            "distance": round(distance, 6), "basis": "heuristic",
            "what_it_means": (f"Matches no briefed image by fingerprint, but resembles the "
                              f"briefed {brief_item['file']}.")}


def _drift_score(conn, briefed, delivered) -> tuple:
    """How far the delivered creative sits from the briefed creative (§9.2).

    The MEAN PAIRWISE distance between the two sets, not the distance between their centroids.
    A centroid contracts toward the mean as a set grows, so the centroid version halved — 0.84
    to 0.47, measured — purely because somebody uploaded eight photographs of one shoot instead
    of one. A score that falls when you supply more evidence is not a measurement.

    Pairwise also makes the numerator and the denominator the same kind of quantity: the
    yardstick is the brief's own mean pairwise distance, so the ratio compares like with like.
    Against the brief's internal spread because the raw number is unreadable alone — visual
    embeddings of real photographs sit close together, and two unrelated images measured 0.0009
    apart in a protocol run, which a marketer reads as "no drift" about creative sharing
    nothing.

    Returns `(drift, warnings)`. What it will not say is whether moving that far was good;
    that is §9.4's question and is marked `judged` there for this reason.
    """
    import core

    briefed_vecs, briefed_missing = _asset_vectors(conn, briefed)
    delivered_vecs, delivered_missing = _asset_vectors(conn, delivered)
    missing = briefed_missing + delivered_missing
    warnings: list = []

    if not briefed_vecs or not delivered_vecs:
        # §6.4's lesson: an outage that reads as a clean result is worse than one that says so.
        # A missing score is not a zero score. D19: the shared notice, not a private code —
        # otherwise it escapes `notices.collapse`, the remedy registry and the server-wide
        # {code, severity, remedy} contract every other degradation goes through.
        return ({"score": None, "code": "not_visually_indexed", "basis": "computed",
                 "what_it_means": ("None of these images is visually indexed, so how far the "
                                   "creative moved could not be measured. This is not a small "
                                   "distance — it is no measurement.")},
                [core._vision_notice("how far the delivered creative moved from the brief could "
                                "not be measured: the images are not visually indexed. Run "
                                "finish_indexing — the files are stored.")])
    if missing:
        # PARTIAL is its own answer. Firing only when a side is entirely unindexed meant a
        # half-indexed set computed a score from whatever subset had vectors and labelled it
        # `measured`, with a count buried in `compared` as the only tell.
        warnings.append(core._vision_notice(
            f"{len(missing)} of these images {'are' if len(missing) != 1 else 'is'} not "
            f"visually indexed, so the drift figure is measured over the rest. Run "
            f"finish_indexing — the files are stored."))

    score = _mean_pairwise(briefed_vecs, delivered_vecs)
    spread = _internal_spread(briefed_vecs)
    relative = _relative_drift(score, spread)
    furthest = _furthest_delivered(delivered, delivered_vecs, briefed_vecs)
    return ({
        "score": round(score, 6), "code": "measured" if not missing else "partly_measured",
        "basis": "computed",
        "relative_to_brief_spread": relative["value"],
        "brief_spread": round(spread, 6) if spread is not None else None,
        "compared": {"briefed": len(briefed_vecs), "delivered": len(delivered_vecs),
                     "not_indexed": len(missing)},
        "furthest_delivered": furthest,
        "what_it_means": (
            f"How far the delivered creative sits from the briefed creative, as the mean "
            f"distance between every briefed image and every delivered one. The raw figure is "
            f"not a percentage and is only readable against something: " + relative["said"]
            + f" It is a distance and not a verdict — whether moving that far was an "
              f"improvement, a neutral change or a degradation is a judgment about this "
              f"campaign."),
    }, warnings)


def _asset_vectors(conn, assets) -> tuple:
    """`(vectors, ids_without_one)`. Tolerates the vector table not existing at all.

    A library where no image was ever embedded has no `asset_vectors` table, and reading it
    raised `OperationalError` — not a `ValueError`, so it reached the model as "Error executing
    tool" with the reason discarded, on exactly the install where CLIP never ran.
    """
    try:
        found = vectorstore.get_many(conn, [a["id"] for a in assets], space="asset")
    except Exception:
        found = {}
    return [found[a["id"]] for a in assets if a["id"] in found], \
        [a["id"] for a in assets if a["id"] not in found]


def _mean_pairwise(left: list, right: list) -> float:
    pairs = [(1.0 - embedding.cosine(a, b)) / 2.0 for a in left for b in right]
    return max(0.0, min(1.0, sum(pairs) / len(pairs)))


def _relative_drift(score: float, spread) -> dict:
    """The drift in units of the brief's own internal spread, where that means anything.

    Three different answers, and collapsing them was wrong twice over. With no spread there is
    one briefed image. With a spread of ZERO the briefed images are identical — a real
    measurement, and the sentence said "there is only one briefed image", which was reproduced
    with two. And where the spread is vanishingly small the ratio explodes: a brief with two
    near-identical renders drove it to 442, a number that looks like precision and is noise.
    """
    if spread is None:
        return {"value": None,
                "said": ("there is only one briefed image, so there is no internal spread to "
                         "compare it against and the raw figure stands alone.")}
    if spread < _SPREAD_FLOOR:
        return {"value": None,
                "said": ("every briefed image looks essentially the same as the others, so "
                         "the brief provides no range to measure against — any delivered "
                         "image would look far away. The raw figure stands alone.")}
    return {"value": round(score / spread, 2),
            "said": (f"here it is {round(score / spread, 2)}\u00d7 the distance the briefed "
                     f"images sit from EACH OTHER, so around 1 is within the brief's own "
                     f"range of looks and well above it is creative that moved outside it.")}


# Below this, the brief's images are the same image as far as the yardstick is concerned, and
# dividing by it produces a number that looks like precision and is noise.
_SPREAD_FLOOR = 0.001


def _internal_spread(vectors: list) -> Optional[float]:
    """Mean pairwise distance within one set — the yardstick the brief provides itself.

    None when there is nothing to spread: one image has no internal distance, and returning
    0.0 for it would say "every briefed image looks identical", which is a measurement about a
    set that does not exist.
    """
    if len(vectors) < 2:
        return None
    pairs = [(1.0 - embedding.cosine(vectors[i], vectors[j])) / 2.0
             for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
    return sum(pairs) / len(pairs)


def _furthest_delivered(delivered, delivered_vecs, briefed_vecs) -> Optional[dict]:
    """Which delivered image sits furthest from the brief — the actionable half of a mean."""
    if not delivered_vecs:
        return None
    scored = [(_mean_pairwise([vec], briefed_vecs), asset)
              for asset, vec in zip([a for a in delivered if a], delivered_vecs)]
    distance, asset = max(scored, key=lambda pair: pair[0])
    return {"asset_id": asset["id"], "file": asset["file_path"],
            "distance": round(distance, 6),
            "what_it_means": "The delivered image sitting furthest from the briefed set."}


def _execution_sentence(as_briefed, never_appeared, new, another_view) -> str:
    parts = [f"{len(as_briefed)} briefed image"
             f"{'s' * (len(as_briefed) != 1)} came back as briefed"]
    if never_appeared:
        parts.append(f"{len(never_appeared)} was briefed and did not appear"
                     if len(never_appeared) == 1 else
                     f"{len(never_appeared)} were briefed and did not appear")
    if new:
        parts.append(f"{len(new)} came back that {'was' if len(new) == 1 else 'were'} never "
                     f"briefed")
    if another_view:
        parts.append(f"{len(another_view)} {'is' if len(another_view) == 1 else 'are'} "
                     f"further photographs of something already counted")
    return ("; ".join(parts) + ". These are matches between images, not a judgment about the "
            "campaign — a briefed image that did not appear may have been replaced by "
            "something better.")


def check_image_provenance(conn, *, asset_ref: dict, campaign_id: Optional[str] = None,
                           threshold: Optional[int] = None,
                           phase: Optional[str] = None) -> dict:
    """
    Check whether an image matches one already in the memory (§6.6 — the SVP's creative-reuse
    question). Works on an image that isn't stored yet — call this before upload_campaign's
    asset, or before ingest_image_asset, to flag reuse up front. If campaign_id is given
    (the campaign this image is headed for), its own assets are excluded from matching, and
    a region mismatch against a matched campaign is flagged explicitly — the point isn't just
    "this image exists," it's "this image's look belongs to a *different* region."
    """
    current = store.get_campaign(conn, campaign_id) if campaign_id else None
    if campaign_id and current is None:
        return {"error": f"campaign {campaign_id} not found"}

    path, warnings = _resolve_asset(asset_ref)
    if not path:
        return {"error": "; ".join(warnings) or "could not resolve asset_ref"}

    try:
        query_hash = images.phash(path)
    except Exception as exc:
        return {"error": f"could not process image: {exc}"}
    matches = _phash_matches(conn, query_hash, exclude_campaign_id=campaign_id,
                             current=current, threshold=threshold, phase=phase)
    return {"query_hash": query_hash, "matches": matches, "warnings": warnings}


# §9.1/D121: what a matched asset IS, said rather than left to the caller. This corpus was
# built when every asset was creative; §9.1 put photographs of executions in the same table,
# and without this a reuse check answered "this image is already in the library" about a
# photograph OF an event exactly as it would about a reused hero render. Both are real matches
# and they are not the same finding.
_PHASE_MEANING = {
    "proposed": "a briefed image — creative that was planned",
    "delivered": "a photograph of what actually ran",
}


def _what_the_match_is(phase: Optional[str]) -> str:
    """The fallback cannot be reached from the database — `assets.phase` is NOT NULL with a
    `proposed` default, which is §9.1's load-bearing decision. It is here for the next value
    somebody adds to the vocabulary: an unrecognised phase described as "a briefed image"
    would invent provenance, and naming it is the one safe thing to say about it.
    """
    return _PHASE_MEANING.get(phase or "", f"an asset recorded as {phase!r} — this reader "
                                           f"does not know what that phase means")


def _phash_matches(conn, query_hash: str, *, exclude_campaign_id: Optional[str],
                   current: Optional[dict], threshold: Optional[int] = None,
                   phase: Optional[str] = None) -> list[dict]:
    """Shared by check_image_provenance and the automatic deck-embedded-image path in
    ingest_campaign — same pHash-match + region-flag logic either way."""
    threshold = config.PHASH_MATCH_THRESHOLD if threshold is None else threshold
    superseded_ids = store.get_superseded_campaign_ids(conn)

    matches = []
    for cand in store.get_all_fingerprints(conn, exclude_campaign_id=exclude_campaign_id,
                                           phase=phase):
        if cand["campaign_id"] in superseded_ids:
            continue
        dist = images.hamming_distance(query_hash, cand["phash"])
        if dist > threshold:
            continue
        c = store.get_campaign(conn, cand["campaign_id"])
        if not c:
            continue
        matches.append({
            "campaign_id": cand["campaign_id"], "title": c["title"], "region": c["region"],
            "asset_id": cand["asset_id"], "hamming_distance": dist,
            "phase": cand.get("phase"),
            "what_it_is": _what_the_match_is(cand.get("phase")),
            "flag": _region_mismatch_flag(current, c),
        })
    matches.sort(key=lambda m: m["hamming_distance"])
    return matches


def find_similar_images(conn, *, asset_ref: dict, campaign_id: Optional[str] = None,
                        top_k: int = 5, region: Optional[str] = None,
                        phase: Optional[str] = None) -> dict:
    """
    Aesthetic/regional visual similarity (CLIP) — catches "same product, different photo,"
    "looks like the APAC shoot," NOT exact/near-duplicate reuse (that's
    check_image_provenance's pHash job). Works on an image that isn't stored yet. Pass
    region to filter to that region FIRST (§6.2's pattern, applied to images); pass
    campaign_id (the campaign this image is headed for) to exclude its own assets and get a
    region-mismatch flag on matches from elsewhere.
    """
    import core

    current = store.get_campaign(conn, campaign_id) if campaign_id else None
    if campaign_id and current is None:
        return {"error": f"campaign {campaign_id} not found"}

    path, warnings = _resolve_asset(asset_ref)
    if not path:
        return {"error": "; ".join(warnings) or "could not resolve asset_ref"}

    try:
        qvec = clip_embed.embed_image(path)
    except Exception as exc:
        # A capability outage is not the caller's input mistake, and `error` is the shape
        # reserved for the latter. This path was returning the review's own quoted string —
        # "Failed to download weights for tag 'openai'" — straight to a marketer, on a
        # sibling tool of the one defect 09 was reported against.
        notice = core._vision_notice(f"visual search could not process the image: {exc}")
        return {"error": notice["affects"], "warnings": [notice], "matches": []}

    if region:
        # filter_campaign_ids already excludes superseded campaigns.
        candidate_ids = store.filter_campaign_ids(conn, region=region, exclude_campaign_id=campaign_id)
        assets = store.list_assets(conn, campaign_ids=candidate_ids, phase=phase)
    else:
        superseded_ids = store.get_superseded_campaign_ids(conn)
        assets = [a for a in store.list_assets(conn, exclude_campaign_id=campaign_id,
                                               phase=phase)
                 if a["campaign_id"] not in superseded_ids]

    asset_to_campaign = {a["id"]: a["campaign_id"] for a in assets}
    asset_phase = {a["id"]: a.get("phase") for a in assets}
    vecs = vectorstore.get_many(conn, list(asset_to_campaign), space="asset")
    hits = embedding.rank(qvec, list(vecs.items()), top_k=top_k)

    matches = []
    for asset_id, sim in hits:
        cid = asset_to_campaign.get(asset_id)
        c = store.get_campaign(conn, cid) if cid else None
        if not c:
            continue
        matches.append({
            "campaign_id": cid, "title": c["title"], "region": c["region"],
            "asset_id": asset_id, "similarity": round(sim, 4),
            # §9.1/D121. "Looks like the APAC shoot" and "looks like a photograph of the
            # Bogotá activation" are different answers to the same query, and this corpus
            # could not tell them apart because it predates the field.
            "phase": asset_phase.get(asset_id),
            "what_it_is": _what_the_match_is(asset_phase.get(asset_id)),
            "flag": _region_mismatch_flag(current, c),
        })
    return {"matches": matches, "warnings": warnings}


def _region_mismatch_flag(current: Optional[dict], other: dict) -> Optional[str]:
    """Shared by check_image_provenance and find_similar_images: the point isn't "this image
    exists elsewhere," it's "this image/look belongs to a *different* region" (§6.6)."""
    if current and current["region"] and other["region"] and current["region"].lower() != other["region"].lower():
        return f"used in a different region ({other['region']} vs {current['region']})"
    return None


# ── asset resolution (secondary path) ────────────────────────────────────────

def _unreadable(detail: str) -> tuple:
    """Every failure here returns a NOTICE, not a bare string.

    They were bare strings, and `notices.collapse` reads dicts — so an ordinary wrong path
    crashed `upload_campaign` with `TypeError: string indices must be integers`. That is not a
    `ValueError`, so `_catch_value_errors` did not catch it and the model got "Error executing
    tool" with the reason discarded: the one case where saying "that path does not exist" is
    the entire job.
    """
    return None, [notices.notice("asset_unreadable", detail=detail)]


def _resolve_asset(ref: dict) -> tuple[Optional[Path], list]:
    """Resolve an asset reference to a local file. Accepts {path} | {asset_id} | {filename,base64}."""
    if not isinstance(ref, dict):
        return _unreadable("asset_ref must be an object")
    if ref.get("path"):
        p = Path(ref["path"])
        return (p, []) if p.is_file() else _unreadable(f"local path not found: {ref['path']}")
    if ref.get("asset_id"):
        d = config.UPLOAD_DIR / ref["asset_id"]
        files = list(d.iterdir()) if d.is_dir() else []
        return (files[0], []) if files else _unreadable(
            f"unknown asset_id {ref['asset_id']}")
    if ref.get("base64") is not None:
        try:
            data = base64.b64decode(ref["base64"], validate=True)
        except (binascii.Error, ValueError) as exc:
            return _unreadable(f"invalid base64: {exc}")
        if len(data) > config.MAX_INLINE_BYTES:
            return _unreadable("inline asset too large; use POST /upload")
        name = Path(ref.get("filename") or "asset.bin").name
        dest = config.UPLOAD_DIR / f"inline_{name}"
        config.ensure_dirs()
        dest.write_bytes(data)
        return dest, []
    return _unreadable("asset_ref needs path, asset_id, or base64")


def _keep_asset(src: Path) -> str:
    """Copy an ingested original into the asset store under a unique name; return its
    stored name. NOT the original filename — campaigns commonly reuse generic names
    (hero.png, IMG_1234.jpg), which let two campaigns silently overwrite each other's stored
    original on disk (found in review)."""
    import shutil
    import uuid
    config.ensure_dirs()
    dest = config.ASSET_DIR / f"{uuid.uuid4().hex[:16]}{src.suffix}"
    shutil.copy2(src, dest)
    return dest.name


def _keep_asset_bytes(data: bytes, ext: str) -> str:
    """Write image bytes directly into the asset store under a unique name; return the
    stored name. Companion to _keep_asset — used for images extracted from a deck (already
    in memory, no source file to copy from) rather than a direct upload."""
    import uuid
    config.ensure_dirs()
    dest = config.ASSET_DIR / f"{uuid.uuid4().hex[:16]}{ext}"
    dest.write_bytes(data)
    return dest.name
