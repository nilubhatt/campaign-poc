"""
Orchestration the MCP tools call — LLM-first: every entry point accepts what Claude has in
a conversation (text + freeform detail), with file references as a secondary path.

The division of labor:
  * Python owns files, storage, and semantic retrieval.
  * Claude owns judgment — ingest/prepare hand it evidence; it reasons and calls save_*.
"""
from __future__ import annotations

import base64
import binascii
import re
import sqlite3
import unicodedata
import time
from pathlib import Path
from typing import Optional, Union

import actions
import chunking
import clip_embed
import config
import embedding
import enums
import extract
import images
import notices
import store
import vectorstore

# Over-fetch factor for the unfiltered ANN path only (§6.1) — the filtered path ranks its
# full candidate set directly, no cap needed. Several chunks from the same campaign can rank
# highly, so fetch more than top_k chunks to still surface top_k *distinct* campaigns; a
# large deck with many highly-ranked chunks can otherwise starve every other campaign out of
# the result entirely (review found this at the old value of 4 — bumped, and still well
# under sqlite-vec's k limit for realistic top_k).
_SEARCH_OVERFETCH = 20


# ── ingest ───────────────────────────────────────────────────────────────────

def ingest_campaign(conn, *, title: str, detail: Optional[str] = None,
                    deck_text: Optional[str] = None, record_type: str = "campaign",
                    status: Optional[str] = None, tags: Optional[list] = None,
                    region: Optional[str] = None, market: Optional[str] = None,
                    markets: Optional[list] = None, collection: Optional[str] = None,
                    supersedes: Optional[str] = None,
                    asset_ref: Optional[dict] = None, confirm: bool = True) -> dict:
    """
    Store a past/proposed campaign, chunk it, and embed each chunk for search (§6.1).

    record_type distinguishes an actual campaign from background reference material or a
    placeholder stub (§6.3); status tracks a campaign's own lifecycle. tags/region/market
    are structured fields used to filter BEFORE similarity ranking (§6.2). collection links
    market/version variants of the same creative (e.g. multiple regional launches of "Khloe
    Q2 2026") — a symmetric grouping, unlike supersedes (asymmetric replacement).

    LLM-first: from Claude Web, pass `deck_text` (the text Claude already read from the
    attached PDF/PPTX) plus freeform `detail`. Alternatively pass `asset_ref`
    ({asset_id} from POST /upload, {path} local, or {filename,base64} inline) and the
    server extracts the text itself — that path also preserves per-page/slide boundaries
    as the natural chunks. Returns the stored campaign summary.

    §6.9: confirm=False previews the fields as given WITHOUT writing anything — for a
    conversational, non-technical intake flow: ask the guided questions (or parse a free-text
    answer into these fields), show the user the preview, then call again with confirm=True
    once they've reviewed/edited it. Default is True (write immediately) for backward
    compatibility with direct/programmatic callers that already know what they want stored.
    """
    # Normalise BEFORE the preview is built. The preview is the correction screen — the only
    # moment a marketer can catch a synonym that guessed wrong — and it was showing the word
    # they typed while a different value went into the database. Worse, the default status
    # below was computed from the RAW record_type while store computed it from the normalised
    # one, so record_type="Campaign" previewed `status: None` and committed `concluded`.
    given_record_type, given_status = record_type, status
    record_type = store._normalise_record_type(record_type)
    status = store._normalise_status(status)
    # Shape changes are lossless and not worth saying; a SYNONYM is a guess, and a guess
    # nobody hears about is one nobody can correct.
    changed = [
        {"field": field, "given": given, "stored_as": stored}
        for field, given, stored in (("record_type", given_record_type, record_type),
                                     ("status", given_status, status))
        if given is not None and stored is not None
        and enums.canonical_shape(str(given)) != enums.canonical_shape(str(stored))
    ]

    if not confirm:
        # Normalize (and validate — raise the same errors confirm=True would) so the
        # preview shows what would ACTUALLY be stored, not the raw input. Reviewed: an
        # unnormalized echo could look fine and then fail differently on confirm=True,
        # defeating the point of a preview. has_actual_metrics=False always here: a
        # not-yet-created campaign can't have metrics on file, so a 'verified' tag
        # correctly raises at preview time too, not just on commit.
        normalized_tags = store.normalize_tags(tags, has_actual_metrics=False)
        normalized_markets = store.normalize_markets(markets)
        return {
            "preview": True, "title": title, "record_type": record_type,
            "status": status if status is not None else ("concluded" if record_type == "campaign" else None),
            "tags": normalized_tags, "region": region, "market": market,
            "markets": normalized_markets,
            "collection": collection, "supersedes": supersedes, "detail": detail,
            "normalised": changed,
            "note": "Nothing has been stored yet. Show this to the user for confirmation or "
                    "edits, then call upload_campaign again with confirm=True to save it."
                    + (" Say what was interpreted: " + "; ".join(
                        f"{c['field']} {c['given']!r} recorded as {c['stored_as']!r}"
                        for c in changed) if changed else ""),
        }

    warnings: list[dict] = []
    stored_path = None
    units: list[str] = []  # natural per-page/slide units, when extraction ran
    commentary: list[dict] = []  # comments, annotations and speaker notes (§2.5)
    commentary_checked = False   # did we read a file, or is 0 just "we never looked"?
    asset_path_for_images: Optional[Path] = None  # only set when a real file reached us
    # ONE clock for the whole handler, not one per loop: this call embeds up to 20 images
    # and then every text chunk, and two independent budgets would let it take twice the
    # transport ceiling while believing itself inside it. The image loop runs first, so it
    # spends from the same allowance the text loop later measures against.
    deadline = time.monotonic() + config.TOOL_TIME_BUDGET_SECONDS

    if asset_ref:
        path, w = _resolve_asset(asset_ref)
        warnings += w
        if path:
            # A real file reached us — store it and extract its embedded images regardless
            # of whether deck_text was also given (docstring promises "instead of/alongside
            # deck_text"; the realistic case is Claude passing both: text it already read,
            # plus the file itself via asset_ref for storage/image extraction).
            stored_path = _keep_asset(path)
            asset_path_for_images = path
            # NOT gated on `not deck_text`. Commentary is a different layer from the body,
            # so "Claude already read the deck and passed deck_text" says nothing about
            # whether the notes were read — and the realistic case is both being passed.
            # (The same gate silently skipped image extraction on the documented demo flow.)
            try:
                commentary, cw = extract.extract_commentary(path)
                warnings += cw
                # Same distinction images_checked exists to make: an empty list says nothing
                # about whether anyone looked, and a calling LLM reading 0 as "this deck has
                # no comments" is making a claim about a file nobody opened. Only true when a
                # type we know how to read was actually read.
                commentary_checked = extract.guess_mime(path.name) in (
                    config.PPTX_MIME, "application/pdf")
            except Exception as exc:              # noqa: BLE001
                commentary = []
                warnings.append(notices.notice(
                    "commentary_unreadable",
                    detail=f"comments and notes could not be read ({exc}); the deck itself "
                           f"was ingested normally"))
            if not deck_text:
                units, w2 = extract.extract_units(path)
                warnings += w2
                deck_text = "\n\n".join(units)

    # The store offer (§5.2) is the thing that creates duplicates, and the product's own
    # title lookup then breaks on them — bulk_import_metrics reports the title as ambiguous
    # and refuses the row. Warn at the moment it happens, while somebody can still say which
    # one they meant.
    duplicates = [c for c in store.list_campaigns(conn)
                  if (c["title"] or "").strip().lower() == (title or "").strip().lower()]
    if duplicates:
        warnings.append(notices.notice(
            "duplicate_title",
            detail=f"{len(duplicates)} existing campaign(s) already titled {title!r}",
            next_actions=[]))
    # The upload path could create every state `update_campaign` refuses: a dangling id, a
    # blank string, whitespace. `get_superseded_campaign_ids` then held {"", "  ",
    # "camp_nope"} and the link pointed at nothing, silently. Self-reference and cycles are
    # impossible here — the id does not exist yet — but "there is no such record" is not.
    supersedes = store.checked_supersedes(conn, None, supersedes) if supersedes else None
    cid = store.insert_campaign(
        conn, title=title, record_type=record_type, status=status, tags=tags, region=region,
        market=market, markets=markets, collection=collection, supersedes=supersedes,
        detail=detail, deck_text=deck_text, asset_path=stored_path,
        # Recorded on the row, not only in this response's warnings: a warning lives for one
        # call, and "nobody ever read this deck's comments" is a fact somebody needs months
        # later, when they are wondering why a search misses what they remember writing.
        commentary_checked=commentary_checked,
    )

    # Images embedded IN the deck, extracted and processed automatically — a separate
    # manual upload_image_asset call per image isn't a workflow anyone would actually use
    # (product feedback). Only possible when a real file reached us (asset_ref); the
    # LLM-first deck_text-only path has no file to extract images from.
    image_assets: list[dict] = []
    # Distinguishes "checked, found nothing/nothing to flag" from "never checked" (e.g. the
    # deck_text-only path with no file, an unsupported file type like legacy .ppt, or
    # extraction itself failing) - all otherwise look identical as an empty image_assets
    # list, which a calling LLM can't tell apart. Tied to the file type actually being one we
    # know how to check, not just to the per-image loop completing without error below (a
    # single image's storage failure shouldn't retroactively make an otherwise-successful
    # check report itself as "never happened").
    images_checked = False
    images_embedded = 0
    current_campaign = None
    if asset_path_for_images is not None:
        image_mime = extract.guess_mime(asset_path_for_images.name)
        try:
            found_images, img_warnings = extract.extract_images(asset_path_for_images, mime=image_mime)
            warnings += img_warnings
        except Exception as exc:
            found_images = []
            warnings.append(notices.notice(
                "images_unreadable",
                detail=f"image extraction failed (deck images will not be searchable or "
                       f"reuse-checked): {exc}"))
        else:
            images_checked = image_mime in (config.PPTX_MIME, "application/pdf")

        if found_images:
            current_campaign = store.get_campaign(conn, cid)

        # PASS 1 — store, fingerprint, reuse-check. Deliberately NOT budgeted: writing a
        # file and hashing it is milliseconds, and reuse detection is the question this
        # product exists to answer ("this hero image is identical to one used in Mexico").
        # Abandoning that to save a fraction of a second would cut the wrong thing. It also
        # guarantees every image leaves a row, so an interrupted run can be finished later
        # rather than being silently lost the way an unextracted image would be.
        for img_bytes, ext, location in found_images:
            stored_name = None
            try:
                stored_name = _keep_asset_bytes(img_bytes, ext)
                image_full_path = config.ASSET_DIR / stored_name
                aid = store.insert_asset(conn, cid, file_path=stored_name)
            except Exception as exc:
                if stored_name:
                    (config.ASSET_DIR / stored_name).unlink(missing_ok=True)
                warnings.append(notices.notice(
                    "image_not_stored",
                    detail=f"deck image on slide/page {location} could not be stored: {exc}"))
                continue
            entry = {"asset_id": aid, "location": location, "fingerprinted": False,
                     "visually_embedded": False, "reuse_flags": [],
                     "_path": image_full_path}
            try:
                h = images.phash(image_full_path)
                store.set_asset_fingerprint(conn, aid, h)
                entry["fingerprinted"] = True
            except Exception as exc:
                warnings.append(notices.notice(
                    "image_not_fingerprinted",
                    detail=f"deck image {aid} not fingerprinted (reuse detection will miss "
                           f"it): {exc}"))
            else:
                try:
                    entry["reuse_flags"] = _phash_matches(
                        conn, h, exclude_campaign_id=cid, current=current_campaign)
                except Exception as exc:
                    warnings.append(notices.notice(
                        "reuse_check_failed",
                        detail=f"deck image {aid} fingerprinted but reuse check failed: "
                               f"{exc}"))
            image_assets.append(entry)

        # PASS 2 — the expensive half. This is what yields when time runs out; the images
        # are already stored and reuse-checked, so stopping here costs only visual
        # similarity, and every skipped one has a row waiting to be finished.
        for entry in image_assets:
            if time.monotonic() >= deadline:
                warnings.append(notices.notice(
                    "indexing_incomplete",
                    affects=f"{images_embedded} of {len(image_assets)} images in this deck "
                            f"are in visual search so far. All of them were stored and "
                            f"checked for reuse, so nothing is lost.",
                    next_actions=actions.to_finish_indexing(cid),
                    detail=f"visually embedded {images_embedded} of {len(image_assets)} "
                           f"deck images before the "
                           f"{config.TOOL_TIME_BUDGET_SECONDS:g}s time budget ran out"))
                break
            try:
                vec = clip_embed.embed_image(entry["_path"])
                vectorstore.add(conn, entry["asset_id"], vec, space="asset")
                store.mark_asset_embedded(conn, entry["asset_id"])
                entry["visually_embedded"] = True
                images_embedded += 1
            except Exception as exc:
                warnings.append(_vision_notice(
                    f"deck image {entry['asset_id']} not visually embedded: {exc}"))

        for entry in image_assets:
            entry.pop("_path", None)

    if not units and deck_text:
        # LLM-first path: Claude passed one flat string with no page/slide boundaries —
        # split on blank lines so chunking.pack() has units to work with.
        units = deck_text.split("\n\n")

    summary = "\n\n".join(p for p in (title, detail) if p).strip()
    chunk_texts = chunking.pack(([summary] if summary else []) + units)

    # Commentary counts as something to embed. The early return below used to fire first,
    # so a deck whose body extracted to nothing reported the notes as found and then threw
    # them away — the count described something that no longer existed.
    if not chunk_texts and not commentary:
        warnings.append(notices.notice(
            "nothing_to_embed", detail="nothing to embed (no title/detail/deck_text)"))
        return {"campaign_id": cid, "title": title, "record_type": record_type,
                "embedded": False, "chunks_total": 0, "chunks_embedded": 0,
                "image_assets": image_assets, "images_checked": images_checked,
                "images_total": len(image_assets), "images_embedded": images_embedded,
                "commentary_found": len(commentary),
                "commentary_checked": commentary_checked,
                "warnings": notices.collapse(warnings)}

    chunk_ids = store.insert_chunks(conn, cid, chunk_texts)

    # Commentary is embedded alongside the deck chunks but stored as its own kind, so
    # retrieval can weigh it or leave it out (defect 08). One chunk per comment, never
    # merged: two notes packed together would share one author and one anchor, and the
    # anchor is half of what makes a comment worth keeping. A note longer than a chunk is
    # split, with every piece keeping the same attribution.
    for item in commentary:
        # The position as a NUMBER as well as a display string: the plan and the review both
        # specify {page, author, date, text, kind}, and only the string survived, so nothing
        # downstream could sort or group by where in the deck a remark sits.
        source = {k: item.get(k) for k in
                  ("kind", "author", "date", "anchor", "page", "slide", "reply_to")}
        source = {k: v for k, v in source.items() if v is not None}
        pieces = chunking.pack([item["text"]])
        ids = store.insert_chunks(conn, cid, pieces, kind="commentary",
                                  sources=[source] * len(pieces))
        chunk_ids += ids
        chunk_texts += pieces

    embedded_count = 0
    # One embed call per chunk, against an embedder that can be slow or wedged. Without a
    # wall-clock budget a long deck simply outlives the transport: the client reports "did
    # not respond", the row is already written, and nobody can tell how much got done
    # (defect 04 — the reviewer hit this failure class inside upload_image_asset). Stopping early and
    # saying so beats being cut off mid-loop.
    for chunk_id, text in zip(chunk_ids, chunk_texts):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            warnings.append(notices.notice(
                "indexing_incomplete",
                affects=f"The campaign is saved and {embedded_count} of "
                        f"{len(chunk_texts)} sections are searchable so far.",
                next_actions=actions.to_finish_indexing(cid),
                detail=f"embedded {embedded_count} of {len(chunk_texts)} sections before "
                       f"the {config.TOOL_TIME_BUDGET_SECONDS:g}s time budget ran out"))
            break
        try:
            # Only the time that is actually left, so no single call can push the handler
            # past its own deadline.
            vec = embedding.embed(text, timeout=remaining)
            vectorstore.add(conn, chunk_id, vec)
            store.set_chunk_embedded(conn, chunk_id)
            embedded_count += 1
        except Exception as exc:
            # The image path splits "the model is missing from this machine" from "this one
            # item failed"; the text path did not, so a dead Ollama produced one line per
            # chunk, each advising finish_indexing — which then fails every item and says
            # calling again will not help. One outage, said once.
            if embedding.is_unreachable(exc):
                warnings.append(notices.notice("text_search_offline", detail=str(exc),
                                               cause="embedder_unreachable"))
                break
            warnings.append(notices.notice(
                "chunk_not_embedded",
                detail=f"chunk {chunk_id} not embedded (search will miss it): {exc}"))

    # "embedded" means SEARCHABLE, not "we managed at least one". Since 2.1 made a partial
    # result a designed outcome, `> 0` would report a deck with 2 of 12 sections indexed as
    # fully embedded — in the same response whose warning says 2 of 12. That is exactly the
    # stored-versus-searchable conflation defect 05 opened with.
    store.mark_embedded(conn, cid, embedded_count == len(chunk_texts))
    current = store.get_campaign(conn, cid)
    earlier_judgment = _judgment_to_check(conn, supersedes)
    return {
        "campaign_id": cid, "title": title, "record_type": record_type,
        "embedded": embedded_count == len(chunk_texts),
        "chunks_total": len(chunk_texts), "chunks_embedded": embedded_count,
        "image_assets": image_assets, "images_checked": images_checked,
        "images_total": len(image_assets), "images_embedded": images_embedded,
        "commentary_found": len(commentary),
        "commentary_checked": commentary_checked,
        "normalised": changed,
        "next_actions": actions.after_upload(
            campaign_id=cid, status=current["status"],
            has_metrics=bool(current["metrics"]),
            earlier_judgment=earlier_judgment),
        "warnings": notices.collapse(warnings),
        # §6.3: the one moment where "was our judgment any good?" is both answerable and
        # free. Attached only when there IS an unreconciled judgment on the record this one
        # replaces — see `_judgment_to_check`.
        **({"earlier_judgment": earlier_judgment} if earlier_judgment else {}),
    }


def add_metrics(conn, campaign_id: str, *, detail: Optional[str] = None,
                structured: Optional[dict] = None, metric_type: str = "actual",
                confirm: bool = True) -> dict:
    """
    Record an outcome/metric on a campaign (§6.5), with the same §6.9 confirm-before-write
    gate as ingest_campaign: confirm=False previews what would be recorded — for a
    conversational feedback flow (which campaign, how did it go, what metrics) where a
    free-text answer gets parsed into detail/structured and shown back before saving.
    """
    if store.get_campaign(conn, campaign_id) is None:
        return {"error": f"campaign {campaign_id} not found"}
    # Validate and normalise BEFORE previewing. The preview used to return first, so a
    # marketer was shown "metric_type: target" as if it were about to be saved and the
    # rejection arrived only after they said yes — and a normalised value ("Results" ->
    # "actual") was hidden from the one screen that exists for them to correct it.
    metric_type = enums.normalise(metric_type, field="metric_type",
                                  valid=store.VALID_METRIC_TYPES,
                                  synonyms=enums.METRIC_TYPE_SYNONYMS, allow_none=False)
    if not confirm:
        return {
            "preview": True, "campaign_id": campaign_id, "metric_type": metric_type,
            "detail": detail, "structured": structured,
            "note": "Nothing has been stored yet. Show this to the user, then call "
                    "add_metrics again with confirm=True to save it.",
        }
    mid = store.add_metrics(conn, campaign_id, detail=detail, structured=structured,
                           metric_type=metric_type)
    return {"metrics_id": mid, "campaign_id": campaign_id, "status": "stored",
            # The moment the precondition for reconciling is satisfied. Offered at
            # save_evaluation time it simply failed: there were no actuals yet (§5.2 review).
            "next_actions": actions.after_metrics(
                campaign_id=campaign_id,
                open_evaluation_id=store.unreconciled_evaluation_id(conn, campaign_id)
                if metric_type == "actual" else None)}


# ── retrieval / evidence for Claude ──────────────────────────────────────────

def _vision_notice(detail: str) -> dict:
    """An image failed to embed — say whether the model is missing from this machine or this
    one image failed on a working model, and carry the remedy the component itself worked
    out.

    `WeightsResolution` already knows the right instruction for each cause: the bundled copy
    is absent (reinstall), the configured path is wrong (fix the variable), the checkpoint
    would not load (same). The first version of the registry threw that away and substituted
    "ask IT to run setup", naming a gesture that exists nowhere — there is no setup script,
    and nothing in run.sh or run.ps1 fetches the vision weights.
    """
    status = clip_embed.weights_status()
    if status.ok:
        return notices.notice("image_not_embedded", detail=detail)
    # `cause` carries health_check's code for the same condition. The two vocabularies are
    # on different axes on purpose — a notice code names what the USER loses, a health code
    # names what is WRONG — but without the link, support reading an upload warning and an
    # installer gating on health_check had no way to know they were looking at one problem.
    return notices.notice("visual_search_offline", detail=detail,
                          cause=f"clip_weights_{status.source or 'unknown'}",
                          remedy=status.remedy or notices.remedy_for("visual_search_offline"))


class _WrongDimension(RuntimeError):
    """The embedder answered, but with vectors this library cannot store."""


# The review's vocabulary, fixed at three values so counts are sortable and comparable
# across evaluations - "critical/major/minor/nit" drifts per run and cannot be summed.
_SEVERITIES = ("blocking", "should_fix", "note")
_VERDICTS = ("approve", "revise", "reject")
# Caps are the mechanism, not decoration: handed an unbounded string a model writes prose
# into it, and the shape stops constraining anything. The long version has its own field.
# What KIND of problem this is, which severity cannot express: severity says how much it
# matters, not whether it is arguable. A guardrail breach cites the rulebook and is not
# debatable; a departure from precedent cites a campaign and invites a rationale — the UAE
# claw machine was a departure that turned out better than the precedent. (§6.2 gives these
# their different vocabulary; the field is defined here so that item does not have to
# migrate every stored judgment.)
_KINDS = ("guardrail_breach", "precedent_departure", "missing_information",
          "internal_contradiction")
# Whether a finding came from code or from judgment (§7.8). Computed findings should be
# identical for every user and a difference there is a bug; judged ones carry the evidence
# behind them. Defaults to "judged": anything the model asserted is a judgment unless the
# server itself worked it out, and defaulting the other way would let an opinion inherit the
# authority of a mechanical check.
_BASES = ("computed", "judged")
_MAX_SUMMARY = 240
_MAX_FINDING = 120
_MAX_FIX = 120
# Capping the headline and leaving its neighbours unbounded moves the essay rather than
# preventing it — 900 words fits comfortably as six findings with a 150-word `detail`, and
# the marketer is no better off. Every field a model can write into is bounded; `detail` is
# the long form, so it gets a paragraph rather than a line.
_MAX_DETAIL = 600
_MAX_QUOTE = 300          # longer than this is not a quote
_MAX_CATEGORY = 40        # an enum in waiting
_MAX_APPROVE_IF = 240     # it is a second summary
_MAX_RESOLVED = 120
# ...and quantity cannot substitute for length: thirty capped findings is an essay built
# out of bricks.
_MAX_FINDINGS = 12
# How many of an earlier judgment's findings to put in front of somebody who is uploading its
# replacement (§6.3). They are shown so the user can see what was said; they get SETTLED one
# by one when this version is evaluated. Unfiltered, filing a document became a twelve-item
# quiz.
_MAX_EARLIER_FINDINGS = 3
# Which layer of a deck a quote was taken from (§2.5): what the brief says, or what somebody
# said about it.
_LAYERS = ("body", "commentary")
# The two kinds that assert something about another record, and so must cite it (§6.1). The
# other two are claims about the subject, which is not in the library.
_CITING_KINDS = ("guardrail_breach", "precedent_departure")
# §6.2: which way a departure departs. "Does not match how Peru seeded" is not a finding
# until somebody says whether different is WORSE here — and forcing that choice is what makes
# the class genuinely different from a breach. A breach is a fact about a rule; a departure is
# a judgment about whether a difference matters, which is precisely the thing a partner is
# entitled to argue with. It also gives the review's own example somewhere to live: the UAE
# brief's claw machine was a departure that turned out BETTER than the precedent, and the
# product filed it as a defect because there was nowhere to say so.
_DEPARTURES = ("regression", "unexplained", "possible_improvement")
# The two classes the review asked for, plus the one that is neither: a claim about the brief
# in front of you rather than about another record. Derived from `kind` rather than stored, so
# there is one place the mapping lives and no second field to drift from the first.
_CLASSES = {
    "guardrail_breach": "not_debatable",
    "precedent_departure": "debatable",
    "missing_information": "about_the_brief",
    "internal_contradiction": "about_the_brief",
}


def _bounded(value, field: str, limit: int, *, where: str = "") -> Optional[str]:
    """One measuring rule for every capped field. `finding` used to be measured after
    stripping and `summary` before it, so trailing whitespace was fatal in one and free in
    the other."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{where}{field} must be text, got {type(value).__name__}")
    text = value.strip()
    if len(text) > limit:
        raise ValueError(f"{where}{field} is {len(text)} characters; the limit is {limit}."
                         + (" Put the explanation in 'detail'." if limit <= 120 else ""))
    return text or None


def _category(value, where: str) -> Optional[str]:
    """Lower-cased so it can be counted: "Timeline", "timeline " and "timeline" are one
    category, and a field meant to be grouped by cannot be case-sensitive."""
    text = _bounded(value, "'category'", _MAX_CATEGORY, where=where)
    return text.lower() if text else None


# A quote is faithful when it says what the record says, not when it matches byte for byte.
# Refusing a re-cased or re-wrapped quote would not improve provenance; it would teach the
# model that quoting is a game it loses, and the way a model wins that game is by quoting
# less. So the comparison folds everything that changes how text LOOKS and nothing that
# changes a word.
#
# The list below is not decorative. Review ran real PDF and PPTX files through
# `extract.extract_units` and back: a hand-built PDF extracts "requirements" as
# "require\u00adments" (a soft hyphen), justified text arrives hyphenated across the line
# break as "sched-\nule", python-pptx emits U+000B for a soft line break and U+00A0 for a
# non-breaking space, and an accented word can arrive decomposed. Every one of those refused
# a quote a person would call verbatim, with a message accusing the model of paraphrasing.
_QUOTE_EQUIVALENTS = {
    "\u2018": "'", "\u2019": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201f": '"',
    "\u2010": "-", "\u2011": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    # NFKC turns "\u2153" into "1\u20443" with a FRACTION SLASH, which is not the "/" anybody types.
    "\u2044": "/",
}
# Characters that carry no word and no space: a soft hyphen marks where a word MAY break, a
# zero-width space marks where a line may. Both are invisible, and a model retyping what it
# read never reproduces them.
_QUOTE_INVISIBLES = dict.fromkeys(
    "\u00ad\u200b\u200c\u200d\u2060\ufeff", "")
# Hyphenation at a LINE BREAK: "sched-\nule" is one word the layout split, not two. Anchored
# to an actual line break, not to any whitespace — the first version deleted every "- " and
# so deleted a spaced dash while leaving an unspaced one, which made the fold asymmetric:
# "3 - 28 March" folded to "3 28" while the quote "3-28 March" folded to "3-28", and a
# faithful quote was refused as paraphrase.
_LINE_HYPHEN_RE = re.compile(r"(?<=\w)-[^\S\r\n]*[\r\n\v\f]\s*(?=\w)")
# Everything else spelled with a dash reads the same however it was spaced, so a dash and the
# space around it both collapse to one separator on both sides. "3-28", "3 - 28" and
# "3 \u2013 28" are one date range written three ways, and refusing two of them would only teach
# the model to stop quoting.
_DASH_SPACING_RE = re.compile(r"\s*-\s*")
# What a model writes when it leaves the middle out. Treated as "and then, further on in the
# same stored unit" — never as "and somewhere else in the record".
_ELISION_RE = re.compile(r"\s*(?:\[\s*(?:\u2026|\.\s*\.\s*\.)\s*\]|\u2026|\.\s*\.\s*\.)\s*")
# `find_similar` marks a trimmed brief with this. A model quoting the tail of what it was
# shown includes the marker; refusing that would be refusing our own punctuation.
_TRUNCATION_MARKER = "[truncated]"
# A quote has to be a quotation. Without a floor, "a" and "." both verified against every
# record in the library — a substring test with a one-character floor certifies nothing, and
# the word "verified" beside it is then a claim the check cannot support.
_MIN_QUOTE = 12
_MIN_QUOTE_SEGMENT = 8
# How much a quote may leave out IN TOTAL, across all of its elisions. Unbounded,
# "Budget \u2026 cannot \u2026 post before the embargo" stitched three unrelated slides into one
# sentence the deck never contained and the meaning inverted. The per-unit rule alone does not
# stop it: `chunking.pack` merges slides up to 1800 characters, and body text is now read from
# the row's columns, so a whole deck is one unit.
#
# Bounded PER HOP it did not stop it either — review chained ten 130-character hops across a
# thirty-slide deck and the check passed every one of them, because each hop was legal on its
# own. A total makes the arithmetic impossible to walk around: leave out more than this and it
# is not an elision, it is two quotes, and they belong to two findings.
_MAX_ELIDED = 200
# And how many times. A character budget alone does not catch a SHORT deck: four ordinary
# slides assembled into "Budget is fixed \u2026 Creators are briefed \u2026 Nobody may exceed \u2026
# Nothing goes live" leaves out barely fifty characters, and it is still a sentence nobody
# wrote. A quotation has one gap in it, occasionally two. Four is not a quotation with
# elisions; it is a composition, and what it composes is deniable.
_MAX_ELISIONS = 2
# How many starting positions to try before giving up on a unit. A quote whose first phrase
# appears twenty times in one record is not being let through by the twenty-first.
_MAX_ALIGNMENTS = 20


def _fold_for_quote_match(text: str, *, join_line_hyphens: bool = True) -> str:
    # NFKC first: it folds ligatures, decomposed accents and several width variants, so the
    # explicit tables below only have to carry what it leaves alone.
    text = unicodedata.normalize("NFKC", text)
    for raw, plain in {**_QUOTE_INVISIBLES, **_QUOTE_EQUIVALENTS}.items():
        text = text.replace(raw, plain)
    if join_line_hyphens:
        text = _LINE_HYPHEN_RE.sub("", text)
    text = _DASH_SPACING_RE.sub(" ", text)
    return " ".join(text.split()).casefold()


def _haystacks(unit: str) -> list:
    """Both readings of a hyphen at the end of a line, because no regex can tell them apart.

    "sched-\nule" is one word the layout split; "well-\nknown" is a real hyphen that happens
    to fall at the break — and typesetting breaks at an existing hyphen first, so the second
    is the commoner of the two. Joining refuses "well-known"; not joining refuses "schedule".
    The stored text is read both ways and a quote matching either is faithful, because the
    only thing in question is where the layout put a line end.
    """
    joined = _fold_for_quote_match(unit)
    apart = _fold_for_quote_match(unit, join_line_hyphens=False)
    return [joined] if joined == apart else [joined, apart]


def _quote_segments(quote: str, where: str) -> list:
    """The pieces of a quote either side of its elisions, or a refusal saying why not.

    Raises rather than returning empty, because "this is too short to be a quotation" and
    "these words are not in the record" are different problems with different fixes, and
    telling a model to reword a quote that was never long enough is a loop with no exit.
    """
    folded = _fold_for_quote_match(quote)
    if folded.endswith(_TRUNCATION_MARKER):
        folded = folded[:-len(_TRUNCATION_MARKER)].rstrip(" .")
    segments = [s.strip(" \"'") for s in _ELISION_RE.split(folded)]
    segments = [s for s in segments if s]
    if sum(len(s) for s in segments) < _MIN_QUOTE:
        raise ValueError(
            f"{where}precedent.quote is too short to be a quotation ({_MIN_QUOTE} characters "
            f"minimum, excluding anything elided). A word or two appears in almost any "
            f"record, so matching it certifies nothing — quote the phrase that makes the "
            f"point.")
    if len(segments) > _MAX_ELISIONS + 1:
        raise ValueError(
            f"{where}precedent.quote leaves out {len(segments) - 1} separate passages. That "
            f"is an assembly rather than a quotation — pick the one passage the finding "
            f"rests on, or make it two findings.")
    short = [s for s in segments if len(s) < _MIN_QUOTE_SEGMENT]
    if short:
        raise ValueError(
            f"{where}precedent.quote has a fragment of {len(short[0])} characters either "
            f"side of an elision; each piece needs {_MIN_QUOTE_SEGMENT}. Fragments that "
            f"short match by accident, and an elision between them asserts a sentence the "
            f"record may not contain.")
    return segments


def _quote_is_in(segments: list, texts: list) -> bool:
    """Are these segments present, in order and close together, in any ONE of these units?

    Per unit, not across the joined list: a quote spanning two units that were never adjacent
    is a sentence the record does not contain. And within a unit the gap is bounded, because
    `chunking.pack` merges slides up to 1800 characters — so "one unit" can be a whole short
    deck, and an unbounded elision inside it can stitch two unrelated slides together.
    """
    quoted = sum(len(s) for s in segments)
    for unit in texts:
        for haystack in _haystacks(unit):
            # Every place the first segment occurs, not only the first. Locking `start` to
            # the leftmost match refused a quote that was verbatim on a recap slide, because
            # the opening slide also carried its first phrase and the span was then measured
            # across the whole deck. Recaps that restate the opening are ordinary in decks.
            # Later segments keep the nearest match, which for a fixed start is optimal.
            start = haystack.find(segments[0])
            tries = 0
            while start >= 0 and tries < _MAX_ALIGNMENTS:
                tries += 1
                at = start
                for segment in segments:
                    found = haystack.find(segment, at)
                    if found < 0 or (found + len(segment) - start) - quoted > _MAX_ELIDED:
                        break
                    at = found + len(segment)
                else:
                    return True
                start = haystack.find(segments[0], start + 1)
    return False


def _verify_quote(conn, cited: str, segments: list, layer: str, where: str, *,
                  is_rule: bool, kind=None) -> None:
    """Refuse a citation the cited record does not support (§6.1).

    Refusal rather than a `verified: false` flag, for the reason consistency idea 3 gives
    about every other check here: a validation error costs a retry, while storing an
    unverified quote beside verified ones is permanent drift, and nothing downstream would
    be able to tell them apart once a summary quoted either.
    """
    on_file = store.text_on_file(conn, cited)
    # What to say INSTEAD of quoting, which depends on the kind: a finding that is a claim
    # about another record cannot simply drop its citation, because `_CITING_KINDS` refuses
    # it a second time. Offering that exit to a precedent_departure sent the model round a
    # loop whose only exit was deleting `kind` — review walked it.
    instead = ("or, if this is not really a claim about that record, say it as a "
               "missing_information or internal_contradiction finding about the brief itself"
               if kind in _CITING_KINDS else
               "or drop the citation and say it as an observation")
    if on_file is None:
        raise ValueError(
            f"{where}precedent cites {cited!r}, which is not a record in this library. Cite "
            f"a campaign_id from the evidence you were given, {instead}.")
    if is_rule:
        row = conn.execute("SELECT record_type FROM campaigns WHERE id = ?",
                           (cited,)).fetchone()
        record_type = row["record_type"] if row else None
        # `rule_id` means the rulebook. Left interchangeable with `campaign_id`, "a rule was
        # broken" could be anchored to somebody's Q3 deck, and the one class of finding the
        # product says is not debatable would rest on a campaign that merely did it that way.
        # Deliberately NOT phrased as "so call it a precedent_departure instead": §2.4's
        # lesson is that any easy exit offered inside a validation message gets taken, and
        # that one is a downgrade from "not debatable" to "arguable".
        if record_type != "reference":
            raise ValueError(
                f"{where}precedent cites {cited!r} as a rule_id, but that record is stored "
                f"as {record_type!r}, not as reference material. A guardrail breach cites "
                f"the guidelines. If no rulebook is on file, then this library cannot "
                f"support a breach finding at all — say what you can support instead of "
                f"anchoring a rule to a campaign.")
    if not on_file["brief"]:
        # Before the match, not after, so the message is the right one: telling a model to
        # reword a quote against a record that has nothing to quote is a loop with no exit.
        # A stub from a KPI workbook with no metric detail is the common case — and the
        # title is deliberately not counted, because "the record says 'Imported KPI row Q3
        # Jakarta'" supports no finding about anything.
        raise ValueError(
            f"{where}precedent cites {cited!r}, which has nothing on file to quote — no "
            f"brief, no recorded results, no comments: a title and nothing else. Cite a "
            f"record with words in it, {instead}.")
    if _quote_is_in(segments, on_file[layer]):
        return
    other = "commentary" if layer == "body" else "body"
    if _quote_is_in(segments, on_file[other]):
        # The misattribution §2.5 predicted this item would otherwise bless: the text IS in
        # the record, so verifying the text alone returns a green tick on a false statement.
        if other == "commentary":
            raise ValueError(
                f"{where}that quote is from the commentary on {cited!r} — a note or comment "
                f"somebody left about it — not from the campaign itself. Set "
                f"precedent.layer to \"commentary\" with the author, or the finding is "
                f"stored as something that deck claimed.")
        raise ValueError(
            f"{where}that quote is the campaign's own text in {cited!r}, not commentary on "
            f"it. Set precedent.layer to \"body\".")
    raise ValueError(
        f"{where}that quote is not in {cited!r}. Quote the evidence you were given — the "
        f"words as they are written, an elision (…) for a short gap — {instead}. Do not "
        f"paraphrase into a quote.")


def _clean_precedent(conn, value, where: str, *, kind=None) -> Optional[dict]:
    """A finding cites either a campaign that did it differently or a rule it breached.
    Before this there was only `id`, meaning a campaign — so a guardrail breach, which is
    anchored to the rulebook rather than to any campaign, had nowhere to cite the thing it
    breached (§12 writes the rules; the slot is defined here so it does not migrate)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{where}precedent must be an object with a quote and either a "
                         f"campaign_id or a rule_id, got {type(value).__name__}")
    # §6.1: the server's own verdict on this citation, never the model's. `basis` is the
    # vocabulary this codebase already uses for the difference (a computed thing is a server
    # fact; a judged one is an assertion), and letting the model write it would be the exact
    # case `save_evaluation` refuses one level up.
    asserted = [key for key in ("basis", "checked", "verified") if key in value]
    if asserted:
        raise ValueError(f"{where}precedent.{asserted[0]} is set by the server, not by you — "
                         f"it records what the server checked about your quote. Send the "
                         f"quote and the id; the check is not yours to assert.")
    campaign_id = value.get("campaign_id") or value.get("id")
    rule_id = value.get("rule_id")
    if not campaign_id and not rule_id:
        raise ValueError(f"{where}precedent must name what it cites — a campaign_id for a "
                         f"departure from precedent, or a rule_id for a guardrail breach")
    # Both slots at once made the campaign_id decorative: the quote was checked against the
    # rule and the campaign could then be anything, invented included, and still be stored
    # beside a passing check. A finding is anchored to ONE thing; two anchors are two
    # findings.
    if campaign_id and rule_id:
        raise ValueError(f"{where}precedent names both a campaign_id ({campaign_id!r}) and a "
                         f"rule_id ({rule_id!r}). A finding is anchored to one thing: the "
                         f"rule it breaches, or the campaign it departs from. If both are "
                         f"true, they are two findings.")
    quote = _bounded(value.get("quote"), "precedent.quote", _MAX_QUOTE, where=where)
    # §6.1: required, not merely bounded. A citation naming a campaign and quoting nothing is
    # the assertion this whole item was written about with an id stapled to it — and it was
    # the shape the caps alone happily accepted.
    if not quote:
        raise ValueError(f"{where}precedent needs a quote — the words from "
                         f"{campaign_id or rule_id!r} that the finding rests on. An id "
                         f"without a quote is an assertion with a reference attached.")
    # Which LAYER the quote came from (§2.5). A commentary chunk is a retrieved chunk, so
    # "I do not think the timeline is realistic" was a perfectly compliant citation against
    # the campaign — and once stored it read forever as something that campaign's own deck
    # said. Defaulting to "body" is the safe reading: a citation that silently BECAME
    # commentary would be the misattribution this field exists to prevent. §6.1 will verify
    # quotes against retrieved chunks, and without the layer it would have verified the
    # misattribution too.
    layer = value.get("layer") or "body"
    if layer not in _LAYERS:
        raise ValueError(f"{where}precedent layer must be one of {list(_LAYERS)} — 'body' "
                         f"is what the deck says, 'commentary' is what somebody said about "
                         f"it; got {layer!r}")
    # §6.1: the quote has to be IN the record it names, at the layer it claims. Without the
    # connection this function could only check that a quote was short — so a finding could
    # cite a campaign that does not exist and quote a sentence nobody ever wrote, and the
    # server stored it with exactly the authority of a faithful one.
    segments = _quote_segments(quote, where)
    _verify_quote(conn, rule_id or campaign_id, segments, layer, where,
                  is_rule=bool(rule_id), kind=kind)
    # `checked`, and nothing else. Not `verified: true`: "verified" already means something
    # exact in this product — a performance claim backed by real metric data — and on a
    # finding the nearer reading is that the FINDING is verified, which is not what was
    # established. Not `basis: "computed"` either, which the first fix used: a finding's
    # `basis` says who produced the FINDING, and the same key one level down would say the
    # citation was computed by the server. It was not. The model wrote the quote and the id;
    # only the CHECK is the server's, and `checked` says exactly that and no more. It is also
    # the shape that can grow — 7.6 adds "window" to the list without a schema change.
    cleaned = {"quote": quote, "layer": layer, "checked": ["record", "layer"]}
    if campaign_id:
        cleaned["campaign_id"] = campaign_id
    if rule_id:
        cleaned["rule_id"] = rule_id
    if layer == "commentary":
        # Who said it and where, so the finding can be read back as "their reviewer said X"
        # rather than "the deck says X".
        for field, limit in (("author", 120), ("anchor", 40), ("date", 40)):
            kept = _bounded(value.get(field), f"precedent.{field}", limit, where=where)
            if kept:
                cleaned[field] = kept
    return cleaned


def _how_to_say_it(by_class: dict, findings: list) -> str:
    """The same judgment, voiced differently depending on what is actually in it (§6.2).

    A class distinction that nothing says out loud is a column in a database. This is where
    the model learns how to put it, and it has to differ — a library of rule breaches and a
    library of departures are not the same news, and saying both as "here is what has to
    change" is exactly the machinery the review said flattens them.
    """
    parts = ["Give the user the verdict and the one-line summary."]
    if by_class.get("not_debatable"):
        parts.append(
            "State the guardrail breach(es) plainly: a rule they wrote was broken, and that "
            "is not a matter of opinion. Name the rule.")
    if by_class.get("debatable"):
        parts.append(
            "The departures are NOT rule breaches — they are places this differs from a "
            "campaign on file, and a difference can be right.")
        # Keyed on the DEPARTURE, not the class. Keyed on the class, a blocking regression —
        # the finding that carries a `revise` when there is no breach — was told to "ask
        # whether it is deliberate, do not tell them to change it back", in the same response
        # as its own `fix` line. The three readings exist so the voicing can differ; two of
        # the three were sharing one voice.
        found = {value: [f for f in findings if f.get("departure") == value]
                 for value in _DEPARTURES}
        if found["regression"]:
            parts.append(
                "The ones marked `regression` are a judgment, not a question: say what the "
                "precedent did and why this is worse. They may disagree, and a rebuttal is a "
                "legitimate answer — but do not soften it into a query.")
        if found["unexplained"]:
            parts.append(
                "The ones marked `unexplained` ARE a question: ask whether the difference is "
                "deliberate, and do not tell them to change it back until they have "
                "answered. Say plainly that their answer cannot yet be recorded against this "
                "judgment, so the same question will come back next time.")
        if found["possible_improvement"]:
            parts.append(
                f"{len(found['possible_improvement'])} departure(s) may be an IMPROVEMENT on "
                f"the precedent — they are in `improvements`, not in `findings`. Say so as "
                f"good news, not as a problem: a departure that turns out better than what "
                f"it departs from is the most valuable thing this library can notice.")
    if by_class.get("about_the_brief"):
        parts.append(
            "The rest are about the brief itself — something it does not say, or something "
            "it says twice differently. Those are gaps to fill, not arguments to have.")
    parts.append(
        "The reasoning behind any finding is in get_evaluation, not here. `next_actions` are "
        "offers — say them in your own words and act on the one the user picks; do not call "
        "them unasked.")
    return " ".join(parts)


def _say_the_disconfirming_check(check: dict) -> str:
    """§6.4 makes overconfidence visible, which it can only do if somebody is told."""
    if check["code"] == "contradicting_precedent":
        return (" The server searched for precedent that CONTRADICTS this verdict and found "
                f"{len(check['uncited'])} the judgment did not cite — see `disconfirming`. "
                "Say so before the verdict: a campaign that looked like this and went the "
                "other way is the one thing most likely to change what the marketer does.")
    if check["code"] == "nothing_to_check_against":
        return (" The server could not check this verdict against the other side: nothing in "
                "the library has measured results pointing that way. Say that the check was "
                "not possible — do NOT say nothing contradicted it.")
    return (" The server searched for precedent contradicting this verdict and found none. "
            "That is worth one sentence: the verdict was argued against and it held.")


# §6.4: which way a verdict has to be argued with. A negative verdict is contradicted by
# precedent that looked like this and WORKED — Mexico, the library's own warning, a brief
# rated weak that performed. A positive verdict is contradicted by precedent that looked like
# this and did not.
_DISCONFIRMING = {
    "revise": ("precedent_that_worked", "performed_well"),
    "reject": ("precedent_that_worked", "performed_well"),
    "approve": ("precedent_that_failed", "underperformed"),
}
_MAX_DISCONFIRMING = 3


def _both_poles(evidence: list) -> dict:
    """The same retrieved evidence, split by what actually happened (§6.4).

    Only `verified` performance tags count. A `performed_well` somebody typed is an
    impression, and weighing an impression as the counterweight to a verdict is the failure
    tag provenance exists to prevent. A record with no verified performance tag is in
    neither pole — `unknown` is the honest third bucket, and it is usually the biggest.
    """
    poles: dict = {"worked": [], "did_not_work": [], "unknown": []}
    for row in evidence:
        verified = {t.get("value") for t in (row.get("tags") or [])
                    if t.get("source") == "verified"}
        where = ("worked" if "performed_well" in verified
                 else "did_not_work" if "underperformed" in verified
                 else "unknown")
        poles[where].append({"campaign_id": row["campaign_id"], "title": row["title"],
                             "similarity": row["similarity"]})
    return poles


def _disconfirming_search(conn, *, verdict: str, text: str,
                          cited_ids: Optional[list]) -> dict:
    """One query for precedent that contradicts this verdict, run by the SERVER (§6.4).

    The review asked for the search and for the result to be recorded, "including nothing".
    It did not say who runs it, and that is the decision worth writing down: a model that has
    already reached a verdict, asked to go and find evidence against itself, is marking its
    own homework — and the failure the review names, drifting toward confirming the first
    strong match, is exactly what would shape the query. This codebase settled the same
    question twice already, in `basis` and in `precedent.checked`: a check is only worth
    anything if a difference in it is a bug, which holds only when the server did it.

    Restricted to `verified` performance tags — a tag somebody typed is an impression, and
    weighing an impression as the counterweight to a verdict is the unverified-evidence
    failure tag provenance exists to prevent.
    """
    looked_for, tag = _DISCONFIRMING[verdict]
    # This runs on EVERY save, so it has to survive a database that predates the columns it
    # reads. An upgraded v0.2.0 schema has no `tags`, and the first version of this crashed
    # every judgment on exactly the machine the review was gathered on — the same lesson
    # §6.1's `text_on_file` learned, one function over.
    if not {"tags", "markets"} <= set(store._columns(conn, "campaigns")):
        return {
            "looked_for": looked_for, "found": [], "uncited": [], "basis": "computed",
            "code": "nothing_to_check_against",
            "what_it_means": ("This library predates performance tagging, so nothing in it "
                              "carries a measured verdict that could contradict this one. "
                              "That is a fact about the library, NOT a check this judgment "
                              "passed."),
        }
    # Whether the library COULD argue back at all, asked before asking whether it did. An
    # empty shelf reported as a clean check turns an absence of evidence into a supporting
    # vote, and those are opposite conclusions about the same judgment.
    possible = store.filter_campaign_ids(conn, tags=[{"value": tag, "source": "verified"}])
    if not possible:
        return {
            "looked_for": looked_for, "found": [], "uncited": [], "basis": "computed",
            "code": "nothing_to_check_against",
            "what_it_means": (
                f"Nothing in the library is tagged {tag!r} with measured results behind it, "
                f"so there is no precedent that could contradict this verdict. That is a "
                f"fact about the library, NOT a check this judgment passed."),
        }
    try:
        matches = find_similar(conn, text=text, top_k=_MAX_DISCONFIRMING,
                               tags=[{"value": tag, "source": "verified"}],
                               full_detail=False)
    except ValueError:
        matches = []
    already = set(cited_ids or [])
    found = [{"campaign_id": m["campaign_id"], "title": m["title"],
              "similarity": m["similarity"], "cited": m["campaign_id"] in already}
             for m in matches]
    uncited = [row["campaign_id"] for row in found if not row["cited"]]
    return {
        "looked_for": looked_for,
        "found": found,
        # The point is not the record, it is the noticing: a contradicting campaign the
        # judgment never cited is the Mexico case — the reasoner never retrieved it, so it
        # never had to explain it away. One it DID cite has already been weighed, and
        # throwing that back would train the reader to skip the field.
        "uncited": uncited,
        "basis": "computed",
        "code": "contradicting_precedent" if uncited else "nothing_contradicted_it",
        "what_it_means": (
            f"{len(uncited)} campaign(s) resembling this one are tagged {tag!r} with measured "
            f"results, and this judgment did not cite them. Say so."
            if uncited else
            f"Nothing in the library resembling this one is tagged {tag!r} with measured "
            f"results behind it. The verdict was checked against the case for the other side "
            f"and nothing came back."),
    }


def _clean_closest_precedent(conn, value) -> Optional[dict]:
    """The same id-and-quote shape as a finding's precedent, and it was outside the check.

    Review stored `{"campaign_id": "camp_nonexistent", "quote": "made up"}` here verbatim —
    an invented citation at the top of the judgment, where a summary is most likely to read
    it aloud. It then echoed the raw dict back into the database, so a model could assert its
    own `verified`/`checked` beside a 5,000-character junk field, one level up from the
    finding where exactly those keys are refused.

    So it is rebuilt rather than passed through, from known keys only. It is still
    MODEL-asserted rather than computed: D8 owns making the server pick it, which needs 7.2's
    server-owned retrieval.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"closest_precedent must be an object with a campaign_id, got "
                         f"{type(value).__name__}")
    where = "closest_precedent: "
    asserted = [key for key in ("basis", "checked", "verified") if key in value]
    if asserted:
        raise ValueError(f"{where}{asserted[0]} is set by the server, not by you.")
    cited = value.get("campaign_id") or value.get("id")
    if not cited:
        raise ValueError("closest_precedent must name the campaign it points at "
                         "(campaign_id)")
    if store.text_on_file(conn, cited) is None:
        raise ValueError(f"closest_precedent names {cited!r}, which is not a record in this "
                         f"library. It is the id a summary quotes first — an invented one "
                         f"there is the most visible wrong citation the product can make.")
    # Validated, not passed through: an unknown layer reached `on_file[layer]` and came back
    # as a KeyError, which `_catch_value_errors` does not catch, so the caller got "Error
    # executing tool" with the message discarded — the very failure that decorator exists
    # for. The finding-level precedent had validated this all along.
    layer = value.get("layer") or "body"
    if layer not in _LAYERS:
        raise ValueError(f"{where}layer must be one of {list(_LAYERS)}, got {layer!r}")
    cleaned = {"campaign_id": cited, "layer": layer}
    quote = _bounded(value.get("quote"), "closest_precedent.quote", _MAX_QUOTE, where=where)
    if quote:
        _verify_quote(conn, cited, _quote_segments(quote, where), layer, where,
                      is_rule=False)
        cleaned["quote"] = quote
        cleaned["checked"] = ["record", "layer"]
    similarity = value.get("similarity")
    if similarity is not None:
        try:
            # `bool` is an `int` in Python, so True would have been stored as 1.0 — a
            # similarity nobody computed. NaN and infinity are worse: `json.dumps` emits them
            # as bare `NaN`/`Infinity`, which is not JSON, so the row reads back broken in
            # any strict consumer.
            if isinstance(similarity, bool):
                raise TypeError
            similarity = float(similarity)
            if similarity != similarity or similarity in (float("inf"), float("-inf")):
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError(f"{where}similarity must be a real number between 0 and 1, got "
                             f"{similarity!r}")
        cleaned["similarity"] = similarity
    return cleaned


def _clean_resolved(value, conn=None) -> list:
    """`[{was, now}]` was accepted as literally anything — a bare string, a number, a dict
    of unrelated keys — so the one field that records whether the library's own advice was
    taken could hold something nothing could read back."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"resolved must be a list of {{was, now}} items, got "
                         f"{type(value).__name__}")
    cleaned = []
    for i, item in enumerate(value):
        where = f"resolved {i}: "
        if not isinstance(item, dict):
            raise ValueError(f"{where}must be an object with 'was' and 'now', got "
                             f"{type(item).__name__}")
        was = _bounded(item.get("was"), "'was'", _MAX_RESOLVED, where=where)
        now = _bounded(item.get("now"), "'now'", _MAX_RESOLVED, where=where)
        if not was or not now:
            raise ValueError(f"{where}needs both 'was' (what the earlier evaluation asked "
                             f"for) and 'now' (what changed)")
        entry = {"was": was, "now": now}
        # A `finding_id` pointing at nothing, or at another campaign's judgment, used to save
        # without complaint and then vanish from the diff — so "we fixed that" was recorded
        # and silently lost, on the one field `adopted` depends on.
        if item.get("finding_id") and conn is not None:
            if not store.finding_exists(conn, item["finding_id"]):
                raise ValueError(
                    f"{where}finding_id {item['finding_id']!r} does not match any finding "
                    f"on record. Take it from get_evaluation, or from the `earlier_version` "
                    f"block prepare_evaluation returns.")
        # §5.4 and §6.3 both need "this specific earlier finding is now addressed", and
        # free text cannot be joined back to one.
        if item.get("finding_id"):
            entry["finding_id"] = item["finding_id"]
        cleaned.append(entry)
    return cleaned


def get_evaluation(conn, *, evaluation_id: str, severity: Optional[str] = None,
                   kind: Optional[str] = None, departure: Optional[str] = None) -> dict:
    """Read a stored judgment back, optionally narrowed to one severity, kind or departure.

    "The detail is fetched on demand" was true of the storage and false of the surface:
    there was no read path at all, so "show me the blocking items" worked only while the
    findings were still in the context window that produced them. The next session, another
    person, and any later comparison had no way to reach them.

    §6.2's class summary and voicing note come back too. Without them the read path had the
    same failure one level up: `save_evaluation` translated `not_debatable` and `debatable`
    into something a marketer can hear, and reading the judgment back a week later returned
    the raw vocabulary with nothing to say how to put it.
    """
    ev = store.get_evaluation(conn, evaluation_id)
    if not ev:
        return {"error": f"evaluation {evaluation_id} not found"}
    findings = ev.get("findings") or []
    stored = list(findings)
    if severity:
        findings = [f for f in findings if f.get("severity") == severity]
    if kind:
        findings = [f for f in findings if f.get("kind") == kind]
    if departure:
        findings = [f for f in findings if f.get("departure") == departure]
    # Counted over everything stored, not over the filtered view: "how much of this judgment
    # is arguable" is a fact about the judgment, and a reader who asked for the blocking
    # items should not be told the rest does not exist.
    by_class: dict = {}
    for finding in stored:
        klass = _CLASSES.get(finding.get("kind"))
        if klass:
            by_class[klass] = by_class.get(klass, 0) + 1
    return {
        "evaluation_id": ev["id"],
        "subject_title": ev["subject_title"],
        "verdict": ev["verdict"],
        "summary": ev["summary"],
        "approve_if": ev.get("approve_if"),
        "closest_precedent": ev.get("closest_precedent"),
        "by_class": by_class,
        "findings": findings,
        "improvements": [f for f in stored
                         if f.get("departure") == "possible_improvement"],
        **({"how_to_say_it": _how_to_say_it(by_class, stored)} if by_class else {}),
        "resolved": ev.get("resolved") or [],
        # A judgment written before §2.4 has no verdict and no findings, only the essay.
        # Returning it as an empty structured evaluation would be a confident answer built
        # on a record silently dropped.
        **({"original_analysis": ev["analysis"],
            "schema": "legacy",
            "note": "This judgment predates the structured schema, so it has no verdict or "
                    "findings — only the original free text, returned as it was written."}
           if ev.get("analysis") and not ev.get("verdict") else {}),
    }


def save_evaluation(conn, *, subject_title: str, verdict: str, summary: str,
                    findings: Optional[list] = None, resolved: Optional[list] = None,
                    closest_precedent: Optional[dict] = None,
                    approve_if: Optional[str] = None, evidence: Optional[dict] = None,
                    provenance: Optional[dict] = None,
                    campaign_id: Optional[str] = None, cited_ids: Optional[list] = None,
                    predictions: Optional[dict] = None, trusted: bool = False) -> dict:
    """Record a judgment as structured findings rather than an essay (defect 07).

    The review's diagnosis was a data-model one, not a prompting one: handed a single
    free-text field a model writes prose into it, so severity, precedent and fix all
    dissolve into one block that nothing downstream can sort, filter, count or collapse.
    Two real evaluations came back at 700 and 900 words — correct, well-sourced, and far
    more than a marketer reads before deciding what to do.

    Validation is server-side on purpose (consistency idea 3): rejecting a write that
    exceeds the caps costs a retry, while accepting it is permanent drift, and the caps are
    what stop the short fields growing back into paragraphs.

    `trusted` is for the server's own findings (§7.1), which are the only ones allowed to
    claim they were computed. It is never set from the MCP surface."""
    findings = [] if findings is None else findings
    if isinstance(findings, dict) or not isinstance(findings, (list, tuple)):
        raise ValueError(f"findings must be a list of findings, got "
                         f"{type(findings).__name__}")
    if len(findings) > _MAX_FINDINGS:
        raise ValueError(f"{len(findings)} findings; the limit is {_MAX_FINDINGS}. Merge "
                         f"the related ones — a list this long is an essay in fragments.")
    if verdict not in _VERDICTS:
        raise ValueError(f"verdict must be one of {list(_VERDICTS)}, got {verdict!r}")
    summary = _bounded(summary, "summary", _MAX_SUMMARY)
    if not summary:
        raise ValueError("summary is required — one line a person can act on")
    approve_if = _bounded(approve_if, "approve_if", _MAX_APPROVE_IF)
    closest_precedent = _clean_closest_precedent(conn, closest_precedent)
    # §6.4 is a server fact for the same reason `basis` and `precedent.checked` are: a check
    # the model reports is a claim, and a model asked to find evidence against a verdict it
    # has already reached is marking its own homework.
    if isinstance(evidence, dict) and "disconfirming" in evidence:
        raise ValueError("evidence.disconfirming is written by the server, not by you — it "
                         "records the search the server ran for precedent contradicting your "
                         "verdict. A check you report on yourself is not a check.")
    resolved = _clean_resolved(resolved, conn)

    cleaned = []
    for i, finding in enumerate(findings):
        if not isinstance(finding, dict):
            raise ValueError(f"finding {i} must be an object, got {type(finding).__name__}")
        severity = finding.get("severity")
        if severity not in _SEVERITIES:
            raise ValueError(f"finding {i}: severity must be one of {list(_SEVERITIES)}, "
                             f"got {severity!r}")
        where = f"finding {i}: "
        text = _bounded(finding.get("finding"), "'finding'", _MAX_FINDING, where=where)
        if not text:
            raise ValueError(f"{where}'finding' is required — one line naming the problem")
        # §5.4: which earlier finding this repeats, when the model knows. This is the feeder
        # `adopted`/`raised_again` needed: without it the only signal was how alike two
        # sentences read, which scored "adidas-affiliated" against "Nike-affiliated" at 0.89
        # and the same problem reworded at 0.36.
        repeats = _bounded(finding.get("repeats"), "'repeats'", 80, where=where)
        kind = finding.get("kind")
        # §6.2: required. Optional, it was the cheapest way past every rule attached to it —
        # a finding with no kind needs no citation, cannot contradict its slot, and is exempt
        # from the departure rule below. The 6.1 review found the escape; this closes it.
        if kind is None:
            raise ValueError(
                f"{where}kind is required — it is what says whether the finding is arguable "
                f"at all, which severity cannot. 'guardrail_breach' (a rule was broken; cite "
                f"the rule), 'precedent_departure' (done differently from a campaign; cite "
                f"it, and say whether different is worse), 'missing_information' (the brief "
                f"does not say), 'internal_contradiction' (the brief contradicts itself).")
        if kind not in _KINDS:
            raise ValueError(f"{where}kind must be one of {list(_KINDS)}, got {kind!r}")
        departure = finding.get("departure")
        if kind == "precedent_departure":
            if departure is None:
                raise ValueError(
                    f"{where}a precedent_departure must say which way it departs: "
                    f"'regression' (the difference is worse), 'unexplained' (it may be "
                    f"deliberate and the brief does not say), or 'possible_improvement' (it "
                    f"may be better than the precedent). Without it, \u201cthis does not match "
                    f"what Peru did\u201d is an observation the reader has to interpret.")
            if departure not in _DEPARTURES:
                raise ValueError(f"{where}departure must be one of {list(_DEPARTURES)}, got "
                                 f"{departure!r}")
            # The mirror of "a guardrail_breach cannot be a note". Asking the marketer to
            # change something back while recording that it may be better contradicts the
            # finding's own reading — and that is what this product did to the UAE brief's
            # claw machine.
            if departure == "possible_improvement" and severity != "note":
                raise ValueError(
                    f"{where}a departure you think may be an improvement cannot be "
                    f"{severity!r} — asking for it to be changed back contradicts your own "
                    f"reading of it. Record it as a note, or say plainly that it is worse.")
            # The same one-sided rule, made symmetric. `unexplained` means "it may well be
            # deliberate; the brief does not say" — an open question. `blocking` means "this
            # alone means the brief cannot proceed as written". A question that by itself
            # stops the brief is the same contradiction as an improvement you want undone.
            if departure == "unexplained" and severity == "blocking":
                raise ValueError(
                    f"{where}a departure you cannot yet say is worse cannot be blocking — "
                    f"\u201cthe brief does not say whether this is deliberate\u201d is a question, and "
                    f"a question does not on its own stop a brief. Say it is a regression if "
                    f"you can say why, or lower it to should_fix and ask.")
            # A fix is an instruction to change something. Attaching one to a departure you
            # have just called a possible improvement is the reversal the severity rule above
            # refuses, arriving in the other field.
            if departure == "possible_improvement" and finding.get("fix"):
                raise ValueError(
                    f"{where}a departure you think may be an improvement cannot carry a "
                    f"`fix` — a fix is an instruction to change it back. Put what you would "
                    f"want to know in `detail` instead.")
        elif departure is not None:
            raise ValueError(
                f"{where}only a precedent_departure carries `departure`; {kind!r} does not. "
                f"A rule is not a matter of degree, and a gap in the brief is not a "
                f"difference from anything.")
        # A rule either applies or it does not. "You broke a rule, but never mind" is the
        # shape of a finding written to avoid an argument.
        if kind == "guardrail_breach" and severity == "note":
            raise ValueError(f"{where}a guardrail_breach cannot be a note — if the rule "
                             f"applies the finding is blocking, and if it does not apply "
                             f"this is not a guardrail breach")
        precedent = _clean_precedent(conn, finding.get("precedent"), where, kind=kind)
        # §6.1/6.2: a finding that makes a claim ABOUT a precedent has to cite one. The other
        # two kinds are anchored to the subject — "the brief gives no end date" has no
        # precedent to quote, and demanding one there would send the model looking for a
        # campaign to quote at, which is the invented evidence this item exists to stop.
        wanted = "rule_id" if kind == "guardrail_breach" else "campaign_id"
        if kind in _CITING_KINDS and not precedent:
            raise ValueError(
                f"{where}a {kind} has to cite what it departs from: precedent with a "
                f"{wanted} and a quote. Without one it is an opinion in the vocabulary of a "
                f"citation.")
        # The SLOT has to match the kind, or the rule_id check is only half a check: the
        # comment on `_verify_quote` says a rule must not be anchored to "somebody's Q3
        # deck", and without this a guardrail_breach could cite exactly that by using the
        # campaign_id slot instead.
        if kind in _CITING_KINDS and precedent and wanted not in precedent:
            raise ValueError(
                f"{where}a {kind} cites a {wanted}, and this precedent has a "
                f"{'campaign_id' if wanted == 'rule_id' else 'rule_id'}. A rule in your "
                f"guidelines and a campaign that did it differently are not "
                f"interchangeable — one is not debatable and the other invites a rationale.")
        basis = finding.get("basis") or "judged"
        if basis not in _BASES:
            raise ValueError(f"{where}basis must be one of {list(_BASES)}, got {basis!r}")
        # §7.8 rests on "a computed finding is identical for every user, so a difference
        # there is a bug" — which only holds if the SERVER computed it. A model that read a
        # missing date did not compute it, and letting it say so would let an opinion
        # inherit the authority of a mechanical check.
        if basis == "computed" and not trusted:
            raise ValueError(f"{where}only the server sets basis 'computed'; a finding you "
                             f"reached yourself is 'judged', however certain it is")
        cleaned.append({
            "severity": severity,
            "kind": kind,
            "departure": departure,
            "basis": basis,
            "category": _category(finding.get("category"), where),
            "finding": text,
            "repeats": repeats,
            "detail": _bounded(finding.get("detail"), "'detail'", _MAX_DETAIL, where=where),
            "precedent": precedent,
            "fix": _bounded(finding.get("fix"), "'fix'", _MAX_FIX, where=where),
        })

    counts = {level: sum(1 for f in cleaned if f["severity"] == level)
              for level in _SEVERITIES}

    # Internal consistency the prose version could not enforce, because nothing could count
    # the findings: a "revise" with nothing to revise is a hedge, and an "approve" carrying
    # a blocking finding contradicts itself.
    # Three notes satisfied the original rule, which is the same hedge one level down.
    if verdict in ("revise", "reject") and not (counts["blocking"] or counts["should_fix"]):
        raise ValueError(f"a verdict of {verdict!r} needs at least one finding above a "
                         f"note saying what has to change — notes alone are an approve "
                         f"with reservations")
    # The old wording ("either the verdict is 'revise' or the finding is not blocking")
    # offered both exits as equals, and one of them is a single token while the other means
    # rewriting the summary. A validation error is a retry only while the honest path is
    # the one it points at.
    if verdict == "approve" and counts["blocking"]:
        raise ValueError(f"cannot approve with {counts['blocking']} blocking finding(s): if "
                         f"the brief genuinely cannot proceed as written the verdict is "
                         f"'revise'. Do not lower the severity to make the write succeed.")

    # Most severe first, always: the order is part of the contract, so two evaluations of
    # the same brief can be compared without re-reading them. The sort is stable, so two
    # findings of equal severity keep the order they were written in — a model that ordered
    # them deliberately is not second-guessed.
    cleaned.sort(key=lambda f: _SEVERITIES.index(f["severity"]))

    # Identify the findings so a later version can say which one it closed, and so two
    # evaluations of the same brief can be compared by reference rather than by string
    # match. Assigned after the sort, so an id also reads as a position.
    # §5.3: the single thing that would most change THIS verdict, computed here rather than
    # carried from prepare_evaluation, and stored so get_evaluation and §7.6's stamp can
    # recover it.
    missing = missing_input_for_citations(conn, cited_ids)
    if missing:
        evidence = dict(evidence or {})
        evidence["most_valuable_missing_input"] = missing

    # §6.4, and recorded rather than only returned: a check that lives for one response is a
    # check nobody can audit, and the review asked for this precisely so overconfidence stays
    # visible afterwards.
    disconfirming = _disconfirming_search(
        conn, verdict=verdict,
        text="\n".join([summary] + [f["finding"] for f in cleaned]),
        cited_ids=cited_ids)
    evidence = dict(evidence or {})
    evidence["disconfirming"] = disconfirming

    eid = store.next_evaluation_id()
    for n, finding in enumerate(cleaned, start=1):
        finding["id"] = f"{eid}#{n}"

    store.insert_evaluation(
        conn, evaluation_id=eid, subject_title=subject_title, verdict=verdict, summary=summary,
        findings=cleaned, resolved=resolved, closest_precedent=closest_precedent,
        approve_if=approve_if, evidence=evidence, provenance=provenance,
        campaign_id=campaign_id, cited_ids=cited_ids, predictions=predictions)

    # The fixed default shape the review asked for: verdict, closest precedent, counts by
    # severity and the one-line ask. The findings themselves are a follow-up question,
    # answered from the same record without re-reasoning.
    # Hand back what has to change, not an instruction to go and ask for it. A `note`
    # telling Claude to offer the findings competes with the request the user actually made
    # and loses — models mirror the shape of a tool result far more reliably than they
    # follow instructions inside one — and it costs the marketer a round trip to learn what
    # the tool already knows. The caps make this bounded by construction: at most twelve
    # findings of a capped line and a capped fix, with `detail` still fetched on demand.
    by_class: dict = {}
    for finding in cleaned:
        klass = _CLASSES[finding["kind"]]
        by_class[klass] = by_class.get(klass, 0) + 1

    return {
        "evaluation_id": eid,
        "verdict": verdict,
        "summary": summary,
        "counts": counts,
        # §6.2: counts by severity say how much each finding matters and nothing about
        # whether it is arguable, so "2 blocking" reads the same for a rule somebody broke
        # and a preference they may have been right to depart from. That collapse is the
        # review's complaint in one line: the two "come out of the same machinery".
        "by_class": by_class,
        "closest_precedent": closest_precedent,
        # `departure` is in the projection: without it the model cannot tell "this is worse"
        # from "this may be deliberate" without a second round trip — the round trip this
        # fixed shape exists to avoid.
        "findings": [{k: f[k] for k in ("id", "severity", "kind", "departure", "finding",
                                        "fix")}
                     for f in cleaned if f["severity"] in ("blocking", "should_fix")],
        # Carried separately, because the rule making a possible improvement a `note` also
        # dropped it out of `findings` — so the response told the model to deliver good news
        # whose text it did not have, and the claw machine, this item's own headline example,
        # was the finding least likely to reach the marketer.
        "improvements": [{k: f[k] for k in ("id", "finding", "detail", "precedent")}
                         for f in cleaned
                         if f.get("departure") == "possible_improvement"],
        "approve_if": approve_if,
        "most_valuable_missing_input": missing,
        "disconfirming": disconfirming,
        # §5.2: the three things anyone actually does after a judgment, prefilled. The
        # supersession offer only appears when there IS an earlier version — an approval of
        # a new brief supersedes nothing, and an offer that is always there stops being read.
        "next_actions": actions.after_evaluation(
            subject_title=subject_title, evaluation_id=eid, verdict=verdict,
            campaign_id=campaign_id),
        "note": _how_to_say_it(by_class, cleaned) + _say_the_disconfirming_check(disconfirming),
    }


def health_check_cli() -> dict:
    """health_check for a terminal or an installer, opening the database READ-ONLY.

    A diagnostic must not repair what it is diagnosing: going through the normal startup
    path created a fresh database when the real one was missing and then reported "0
    records, healthy", and a corrupt file crashed with a raw traceback — on the surface
    built so nobody has to read tracebacks."""
    import sqlite3

    conn = None
    try:
        conn = sqlite3.connect(f"file:{config.DB_PATH}?mode=rw", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("SELECT 1 FROM campaigns LIMIT 1")
    except Exception as exc:
        if conn is not None:
            conn.close()
        missing = not Path(config.DB_PATH).exists()
        report = health_check(_NoDatabase(), probe=True)
        report["components"]["database"] = {
            "ok": False,
            "code": "db_missing" if missing else "db_unreadable",
            "detail": (f"no database at {config.DB_PATH}" if missing
                       else f"the file at {config.DB_PATH} is not a readable database: {exc}"),
            "affects": "Everything — nothing can be saved, searched or reported.",
            "remedy": ("Nothing has been uploaded yet, or the data directory was moved or "
                       "deleted. Start the server once to create it, or restore the file."),
        }
        report["ok"] = False
        report["headline"] = f"Not working: database. {report['headline']}"
        return report
    try:
        return health_check(conn)
    finally:
        conn.close()


class _NoDatabase:
    """Stands in for an unusable connection so the other components can still be reported —
    knowing text search and the vision model are fine narrows the problem."""

    def execute(self, *args, **kwargs):
        raise sqlite3.OperationalError("no usable database")


def health_check(conn, *, probe: bool = True) -> dict:
    """Answer "is this thing actually working" in one call, in seconds (defect 06).

    Establishing that half the product was dead previously took a 60-second timeout, a
    second upload to capture a warning string, and a read of the server's source. None of
    that is available to a marketer, and barely to an admin.

    Three properties are load-bearing, more than the field list:
      * it never hangs — probes are bounded far below a normal call's budget, because a
        diagnostic that inherits the timeout it exists to diagnose is useless;
      * it never raises — a health check that crashes leaves you where you started;
      * every unhealthy component carries a stable `code` (for installers and later
        automation), an `affects` line in the user's terms, and a `remedy` for whoever
        administers the machine. Those are three different audiences and one string cannot
        serve all of them.

    `ok` is about LIVENESS only. Whether the library is fully indexed is a different
    question with a different answer (`coverage.complete`) — folding them together would
    fail an installer gate on a working machine simply because someone uploaded a big deck
    a minute earlier.

    Components are reported separately because they fail independently: vision being
    unavailable is not the server being down, and collapsing that would hide which half is
    broken — the thing the reviewer had to read source to discover.

    probe=False skips the live embedder call, for cheap repeated liveness (e.g. an HTTP
    healthz poll) that should not hammer Ollama on every request."""
    components: dict[str, dict] = {}

    try:
        campaigns = conn.execute("SELECT COUNT(*) AS n FROM campaigns").fetchone()["n"]
        backend = vectorstore.backend_name(conn)
        components["database"] = {
            "ok": True,
            "detail": f"{campaigns} records in {config.DB_PATH}; vector_index={backend}",
        }
        stranded = vectorstore.count_unreadable_vectors(conn)
        if stranded:
            components["database"] = {
                "ok": False, "code": "vector_index_mismatch",
                "detail": f"{stranded} stored vectors are in a table this build cannot "
                          f"read (vector_index={backend}). Records look indexed but "
                          f"searches will return nothing.",
                "affects": "Searching the library — it will come back empty or short.",
                "remedy": "This database was written by a build with a different vector "
                          "backend. Reinstall the matching version, or re-run "
                          "finish_indexing after removing the stale vectors.",
            }
        elif "fallback" in backend or "python" in backend:
            # Not fatal, but it is a real degradation and it was invisible here while
            # /healthz reported it — exactly the asymmetry the reviewer kept hitting.
            components["database"]["degraded"] = (
                "sqlite-vec is not loaded, so search is using the slower pure-Python "
                "fallback. Results are the same; large libraries will be slower.")
    except Exception as exc:
        campaigns = 0
        components["database"] = {
            "ok": False, "code": "db_unreadable",
            "detail": f"cannot read {config.DB_PATH}: {exc}",
            "affects": "Nothing can be saved or searched.",
            "remedy": "The database file is missing or unreadable. Check the install "
                      "directory exists and the account running this can write to it.",
        }

    probe_timeout = min(config.HEALTH_PROBE_SECONDS, config.EMBED_TIMEOUT_SECONDS)
    if not probe:
        components["text_search"] = {
            "ok": True, "detail": f"{config.EMBED_PROVIDER} (not probed)"}
    else:
        try:
            started = time.monotonic()
            vec = embedding.embed("health check", timeout=probe_timeout)
            if len(vec) != config.EMBED_DIM:
                # Answering is not the same as answering usefully: a different model
                # returns a different width, and every upload then fails deep inside the
                # vector store while this reported everything fine.
                raise _WrongDimension(
                    f"{config.EMBED_PROVIDER} returned {len(vec)}-dimension vectors but "
                    f"this library stores {config.EMBED_DIM}")
            model = (f" ({config.OLLAMA_EMBED_MODEL})"
                     if config.EMBED_PROVIDER == "ollama" else "")
            components["text_search"] = {
                "ok": True,
                "detail": f"{config.EMBED_PROVIDER} responded in "
                          f"{time.monotonic() - started:.2f}s{model}",
            }
        except _WrongDimension as exc:
            components["text_search"] = {
                "ok": False, "code": "embedder_wrong_dimension", "detail": str(exc),
                "affects": "Uploading decks and searching — new uploads will fail outright.",
                "remedy": (f"The embedding model does not match this library. Set "
                           f"CAMPAIGN_POC_OLLAMA_MODEL back to one producing "
                           f"{config.EMBED_DIM}-dimension vectors (default: "
                           f"{config.OLLAMA_EMBED_MODEL}), or start a new library."),
            }
        except Exception as exc:
            message = str(exc).lower()
            slow = "timed out" in message or "timeout" in type(exc).__name__.lower()
            missing_model = "404" in message or "not found" in message
            components["text_search"] = {
                "ok": False,
                "code": ("embedder_slow" if slow else
                         "embedder_model_missing" if missing_model else
                         "embedder_unreachable"),
                # The probe's own timeout, not the embed timeout: reporting a number it
                # never waited is the original complaint in miniature.
                "detail": (f"{config.EMBED_PROVIDER} did not answer within "
                           f"{probe_timeout:g}s" if slow
                           else f"the model {config.OLLAMA_EMBED_MODEL!r} is not installed"
                           if missing_model
                           else f"{config.EMBED_PROVIDER} embedder: {exc}"),
                "affects": "Searching the library, uploading decks and evaluations.",
                "remedy": (f"If this is Ollama, check it is running at {config.OLLAMA_URL} "
                           f"and that `ollama pull {config.OLLAMA_EMBED_MODEL}` has been "
                           f"run. If it was just started it may still be loading the "
                           f"model — try again in a minute."),
            }

    try:
        weights = clip_embed.weights_status()
        components["visual_search"] = (
            {"ok": True,
             "detail": f"weights from {weights.source}"
                       + (f" ({weights.path})" if weights.path else "")}
            if weights.ok else
            {"ok": False, "code": f"clip_weights_{weights.source or 'unknown'}",
             "detail": weights.reason,
             # Not "and flagging reused images": exact-reuse detection is perceptual
             # hashing and does not touch the vision model. Review caught the same false
             # claim in the upload warning; it was here too.
             "affects": "Finding visually similar creative. Exact-reuse detection is "
                        "unaffected.",
             "remedy": weights.remedy}
        )
    except Exception as exc:
        components["visual_search"] = {
            "ok": False, "code": "clip_load_failed",
            "detail": f"could not determine the vision model's state: {exc}",
            "affects": "Finding visually similar creative. Exact-reuse detection is "
                       "unaffected.",
            "remedy": "Reinstall to restore the shipped weights.",
        }

    try:
        outstanding = store.count_unembedded(conn)
        assets = conn.execute("SELECT COUNT(*) AS n FROM assets").fetchone()["n"]
        by_campaign = store.outstanding_by_campaign(conn)[:5]
    except Exception:
        outstanding, assets, by_campaign = {"chunks": 0, "assets": 0}, 0, []

    backlog = outstanding["chunks"] + outstanding["assets"]
    coverage = {
        "campaigns": campaigns,
        "images": assets,
        "sections_unindexed": outstanding["chunks"],
        "images_unindexed": outstanding["assets"],
        "complete": backlog == 0,
        # Names, not just a number: "212 outstanding" means nothing to a marketer, "your
        # Mexico deck is fine, two older records are not" is the actual answer.
        "outstanding": by_campaign,
    }

    live = all(c["ok"] for c in components.values())
    broken = [name.replace("_", " ") for name, c in components.items() if not c["ok"]]
    if not live:
        headline = f"Not working: {', '.join(broken)}. Everything else is fine."
    elif not coverage["complete"]:
        headline = (f"All components working. {backlog} item(s) across "
                    f"{len(by_campaign)} record(s) are stored but not yet searchable.")
    else:
        headline = f"Everything is working; {campaigns} record(s) fully searchable."

    report = {"ok": live, "headline": headline, "version": config.VERSION,
              "build": config.VERSION_FULL, "components": components,
              "coverage": coverage, "tools": published_tool_parameters()}
    if backlog:
        report["backlog_remedy"] = (
            f"{backlog} items are stored but not searchable, so results will be incomplete. "
            f"Run finish_indexing to complete them — nothing needs re-uploading."
        )
    return report


# Component failures that a bounded wait can plausibly clear: a daemon started seconds ago
# and still warming up. Everything else is settled — a missing checkpoint does not appear by
# waiting for it, and waiting 60 seconds to repeat that is worse than saying it at once.
_TRANSIENT_CODES = ("embedder_unreachable", "embedder_slow")


def wait_until_ready(timeout: float = 60.0, interval: float = 3.0) -> dict:
    """Run the self-test, retrying while the only thing wrong is something still starting.

    The installers start Ollama and sleep 3, then gate on it. That is one shot at a daemon
    three seconds old: a cold model load measured 0.7s here, and nobody has measured it on
    the Windows laptop the review was written against, with antivirus scanning a freshly
    written 274MB file. Refusing an install because a service was slow to warm up is a
    refusal the user can do nothing useful with — but only for the codes where waiting is
    the answer.
    """
    deadline = time.monotonic() + timeout
    while True:
        report = health_check_cli()
        if report["ok"]:
            return report
        failing = [c for c in report["components"].values() if not c.get("ok")]
        if any(c.get("code") not in _TRANSIENT_CODES for c in failing):
            return report
        if time.monotonic() + interval >= deadline:
            return report
        time.sleep(interval)


# ── what is missing (§5.3, idea C) ──────────────────────────────────────────
#
# "The library knows it holds one campaign with real outcome data. It knows no LATAM store
# launch has ever carried a budget. It knows the KPI workbook named in its own rubric has
# never been supplied. It says none of this unless directly interrogated."
#
# Ranked, lowest number first, and the ranking is the design. An unordered list of everything
# absent is the thing nobody reads — which is the state being replaced. The order is by how
# much closing the gap would change what this library can answer:
#
#   1  it is empty, so nothing else is worth saying
#   2  nothing has measured outcomes, so no judgment rests on evidence
#   3  a whole market has none, so judgments about that market rest on nothing local
#   4  records are stored but not searchable, so the evidence exists and cannot be found
#   5  decks whose commentary was never read — real content, never ingested
# How many names a gap may list before it is a paragraph rather than a sentence.
_MAX_NAMED = 5

_GAP_RANK = {
    "library_is_empty": 1,
    "few_verified_outcomes": 2,
    "market_without_outcomes": 3,
    "partly_indexed": 4,
    # "commentary_never_read" is recorded but not reported — see gaps() for why.
}


def gaps(conn) -> dict:
    """What this library is missing, ranked, with what would close each one.

    The counterpart to `most_valuable_missing_input` on a judgment: this is about the
    LIBRARY, that is about one verdict, and they routinely disagree — a library that is 90%
    measured can still produce a judgment resting entirely on the unmeasured tenth.
    """
    # Superseded records are excluded from every search, so counting one as the library's
    # measured evidence describes something no judgment can reach.
    superseded = store.get_superseded_campaign_ids(conn)
    campaigns = [c for c in store.list_campaigns(conn)
                 if c.get("record_type") != "reference" and c["id"] not in superseded]
    # A campaign that has not run cannot be missing its results, and asking for them is a
    # request nobody can satisfy — the permanent-complaint failure, on the highest-ranked
    # gap. `after_upload` already drew this line; this did not.
    ran = [c for c in campaigns if c.get("status") == "concluded"]
    found: list[dict] = []

    if not campaigns:
        # Every gap is present in an empty library, which makes the list useless. A new
        # install has exactly one thing to do, and it is not "fix your LATAM coverage".
        found.append({
            "code": "library_is_empty",
            "what": "There are no campaigns in the library yet.",
            "why_it_matters": "Every judgment this product makes is a comparison against "
                              "what you have already run, so with nothing stored there is "
                              "nothing to compare against.",
            "counts": {"campaigns": 0},
            "next_actions": actions.to_first_upload(),
        })
        return _ranked(found)

    # `has_metrics` counts any metric row, so a PREDICTED figure silenced this gap — and a
    # forecast is the opposite of a measured outcome; it is the thing reconciliation later
    # scores against the actuals.
    measured_ids = store.campaigns_with_actual_metrics(conn)
    with_outcomes = [c for c in ran if c["id"] in measured_ids]
    if ran and len(with_outcomes) < len(ran):
        found.append({
            "code": "few_verified_outcomes",
            "what": f"{len(with_outcomes)} of {len(ran)} finished campaigns have measured "
                    f"results on file.",
            "why_it_matters": "A campaign with no outcome data can be cited as a precedent "
                              "but cannot show whether it worked, so a judgment resting on "
                              "it rests on somebody's impression.",
            "counts": {"campaigns": len(ran), "with_outcomes": len(with_outcomes)},
            "next_actions": actions.trim([actions.action(
                "Record what one of these campaigns actually achieved",
                "add_metrics",
                why=f"{len(ran) - len(with_outcomes)} finished campaigns have no results.",
                consent="ask", needs=["which campaign, and the numbers"],
                campaign_id=next(c["id"] for c in ran if c["id"] not in measured_ids))]),
        })

    # A gap located in a SLICE rather than in the total. A marketer asking about Colombia
    # does not care that the library is 60% measured overall if the LATAM part is 0%.
    # Grouped by §5.5's `_markets_of`, not by a second implementation here. This grew its
    # own market-or-region grouping and the two surfaces then described the same library
    # differently — one of them ignoring the `markets` list entirely (tracker D55).
    by_market: dict[str, list] = {}
    display: dict[str, str] = {}
    for campaign in ran:
        for raw in _markets_of(campaign):
            if raw is None:
                continue
            key = raw.strip().lower()
            display.setdefault(key, raw.strip())
            if campaign not in by_market.setdefault(key, []):
                by_market[key].append(campaign)

    # Ordered by how many campaigns are affected, not alphabetically: the one place magnitude
    # decided anything was deciding it by the alphabet, offering Andorra's single campaign
    # ahead of LATAM's twenty. And no `len(barren) < len(by_market)` guard — it was meant to
    # avoid noise and instead silenced the review's own example, since "no LATAM store launch
    # has ever carried a budget" in a library holding only LATAM campaigns said nothing.
    barren = sorted((m for m, rows in by_market.items()
                     if not any(c["id"] in measured_ids for c in rows)),
                    key=lambda m: (-len(by_market[m]), m))
    if barren:
        # Capped: 35 markets produced a 3,185-character sentence inside a tool result
        # somebody has to read.
        shown = [display[m] for m in barren[:_MAX_NAMED]]
        more = f" (and {len(barren) - _MAX_NAMED} more)" if len(barren) > _MAX_NAMED else ""
        # Not a campaign an earlier gap already sent them to: the oldest unmeasured record is
        # usually also the barren market's first, and accepting both offers appended two
        # rows to it.
        already = {a["prefilled_args"].get("campaign_id")
                   for g in found for a in g["next_actions"]}
        target = next((c for m in barren for c in by_market[m]
                       if c["id"] not in already), by_market[barren[0]][0])
        found.append({
            "code": "market_without_outcomes",
            "what": f"No campaign in {', '.join(shown)}{more} has measured results.",
            "why_it_matters": "Judgments about these markets can only be compared against "
                              "campaigns elsewhere, which is a weaker comparison than it "
                              "looks.",
            "counts": {"markets": shown, "markets_total": len(barren)},
            "next_actions": actions.trim([actions.action(
                f"Record results for a campaign in {display[barren[0]]}",
                "add_metrics",
                why=f"{display[barren[0]]} has campaigns on file but nothing measured.",
                consent="ask", needs=["which campaign, and the numbers"],
                campaign_id=target["id"])]),
        })

    outstanding = store.count_unembedded(conn)
    if outstanding["chunks"] + outstanding["assets"]:
        by_campaign = store.outstanding_by_campaign(conn)
        found.append({
            "code": "partly_indexed",
            "what": f"{len(by_campaign)} record(s) are stored but not fully searchable.",
            "why_it_matters": "The content is here and searches cannot find it, so it is "
                              "absent from evidence without being absent from the library.",
            "counts": {"records": len(by_campaign),
                       "items": outstanding["chunks"] + outstanding["assets"]},
            "next_actions": actions.to_finish_indexing(
                by_campaign[0]["campaign_id"] if by_campaign else None),
        })

    # DELIBERATELY not reported, though it is recorded (§5.2's D39). The only offer available
    # is `upload_campaign`, which creates a SECOND record and fires `duplicate_title` — item
    # 5.2 refused exactly this offer in writing, one commit before this one asked for it. No
    # tool attaches a deck to an existing campaign. A gap whose only action makes things
    # worse is a complaint, which this item's own rule forbids; `commentary_checked` is kept
    # so it can be reported the moment there is something that closes it (D51).

    return _ranked(found)


def _ranked(found: list[dict]) -> dict:
    for gap in found:
        gap["rank"] = _GAP_RANK[gap["code"]]
    found.sort(key=lambda g: g["rank"])

    # Two gaps can be closed by one act — a library whose single unmeasured campaign is also
    # its only LATAM campaign has two true facts and one thing to do. Offering the same call
    # twice is how somebody accepts both and appends two identical metric rows to the same
    # record. The later gap keeps the fact and points at what closes it.
    seen: dict[tuple, str] = {}
    for gap in found:
        kept = []
        for offer in gap["next_actions"]:
            key = (offer["tool"], tuple(sorted(offer["prefilled_args"].items())))
            if key in seen:
                gap["closed_by"] = seen[key]
                continue
            seen[key] = gap["code"]
            kept.append(offer)
        gap["next_actions"] = kept

    return {"gaps": found, "most_valuable": found[0]["code"] if found else None}


def missing_input_for_citations(conn, cited_ids: Optional[list]) -> Optional[dict]:
    """The standing line, recomputed at save time from what the judgment actually cited.

    It lived only on `prepare_evaluation` — two calls before the verdict a user hears, with a
    note asking for it to be repeated. This project's own stated principle is that models
    mirror the shape of a tool result far more reliably than they follow instructions inside
    one, so a line delivered earlier and asked to be carried forward is the thing that gets
    dropped.

    Computed by the server rather than accepted from the model, for the same reason
    `evidence` and `provenance` are: it is a fact about what the library holds, and a model
    asserting "nothing is missing" would be asserting it about records it cannot see.
    """
    if not cited_ids:
        return most_valuable_missing_input([])
    measured = store.campaigns_with_actual_metrics(conn)
    cited = [store.get_campaign(conn, cid) for cid in cited_ids]
    cited = [c for c in cited if c]
    if not cited:
        return most_valuable_missing_input([])
    return most_valuable_missing_input([
        {"campaign_id": c["id"], "title": c["title"],
         "metrics": [m for m in c["metrics"] if m["metric_type"] == "actual"]}
        for c in cited])


def most_valuable_missing_input(evidence: list[dict]) -> Optional[dict]:
    """Of everything missing, the one thing that would most change THIS judgment.

    The review's own phrasing: "this verdict rests on zero verified outcomes; the Q2 results
    workbook would change that." One thing, not a list — several things are always missing,
    and naming them all is the behaviour being replaced. None when nothing is missing, which
    is what makes the line worth reading when it appears.
    """
    if not evidence:
        return {
            "code": "no_precedent",
            "what": "Nothing in the library is similar enough to compare this against.",
            "why_it_matters": "Without a precedent this is an opinion rather than a "
                              "judgment from your own record.",
            "next_actions": actions.trim([actions.action(
                "Add a past campaign of this kind, so there is something to judge against",
                "upload_campaign",
                why="No stored campaign resembles this proposal.",
                consent="ask", needs=["a comparable campaign"])]),
        }
    measured = [e for e in evidence if e.get("metrics")]
    if not measured:
        return {
            "code": "no_measured_precedent",
            "what": f"None of the {len(evidence)} campaigns this rests on has measured "
                    f"results on file.",
            "why_it_matters": "The comparison is to what was planned, not to what happened, "
                              "so the verdict cannot say whether any of it worked.",
            "next_actions": actions.trim([actions.action(
                f"Record what \u201c{evidence[0]['title']}\u201d actually achieved",
                "add_metrics",
                why="It is the closest precedent and has no results on file.",
                consent="ask", needs=["the numbers"],
                campaign_id=evidence[0]["campaign_id"])]),
        }
    return None


# ── comparing two versions of a brief (§5.4, idea D) ────────────────────────
#
# "Working out what actually changed between the Colombia versions — which corrections were
# adopted, which were ignored, which facts went stale — was done by hand, and it is the
# single most common real task this product faces."
#
# "Computed against the earlier version's evaluation findings" is what makes this a
# computation rather than a second judgment. Once both versions have been evaluated the
# comparison is between two sets of findings, and every bucket below is a fact about the
# record rather than an opinion about the decks.
#
# Two reviews of one defect rarely word it identically, so matching on exact text would
# report every ignored correction as a new one — the most flattering error available.

# How alike two findings must read before they are the same finding. Deliberately generous:
# the cost of splitting one problem into two is a brief credited for a correction it never
# made, and the cost of merging two is one line a reader can see is wrong.
_SAME_FINDING = 0.62


def _looks_like(a: str, b: str) -> float:
    import difflib

    return difflib.SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def _pair_score(earlier: dict, later: dict) -> float:
    """How likely these two describe the same problem.

    Category is a hint rather than a gate: the same defect gets filed under different
    categories by two reviewers, but when both agree on one it is the only signal available
    to separate two findings whose wording is equally close.
    """
    score = _looks_like(earlier.get("finding"), later.get("finding"))
    if earlier.get("category") and earlier["category"] == later.get("category"):
        score += 0.1
    return score


def _pair_up(earlier: list[dict], later: list[dict]) -> dict:
    """Best-first assignment over ALL candidate pairs, one earlier to one later.

    The first version walked the earlier findings in order and gave each the best REMAINING
    later one, so with two earlier findings competing for one later finding the one processed
    first won — even at 0.79 against the other's 0.98. Which correction was reported as
    raised again therefore depended on the order somebody typed them, or on the severity sort
    that reorders them before ids are assigned. The same two decks must not diff differently.

    Scoring every pair and taking them best-first makes the result a property of the two sets
    rather than of their order. Ties break on the findings' own text, so it is deterministic
    rather than merely consistent.
    """
    pairs = sorted(
        ((_pair_score(e, l), (e.get("finding") or ""), (l.get("finding") or ""), i, j)
         for i, e in enumerate(earlier) for j, l in enumerate(later)),
        key=lambda t: (-t[0], t[1], t[2]))
    taken_earlier: set = set()
    taken_later: set = set()
    assignment: dict = {}
    for score, _et, _lt, i, j in pairs:
        if score < _SAME_FINDING:
            break
        if i in taken_earlier or j in taken_later:
            continue
        taken_earlier.add(i)
        taken_later.add(j)
        assignment[i] = (j, score)
    return assignment


def diff_campaigns(conn, *, earlier: str, later: str) -> dict:
    """What changed between two versions of the same brief.

    Returns the four the review named — `adopted`, `ignored`, `newly_introduced`,
    `carried_stale` — plus `no_longer_raised`, which the review folded into "adopted" and
    which is not the same thing: a finding neither resolved nor repeated was either fixed
    without being recorded or missed by the second review, and the record cannot tell those
    apart. Counting it as adopted would credit a brief for work nobody verified.

    Nothing is guessed. Where a version has not been evaluated there is nothing to compute,
    and `comparable` is false — because "you ignored my correction" is an accusation, and
    inferring it from deck text would be a judgment dressed as a computation.
    """
    first = store.get_campaign(conn, earlier)
    second = store.get_campaign(conn, later)
    for cid, record in ((earlier, first), (later, second)):
        if record is None:
            return {"error": f"campaign {cid} not found"}
    if earlier == later:
        return {"error": "those are the same campaign; pass the two versions you want "
                         "compared"}

    # Which version is earlier decides what "adopted" means, so getting it backwards inverts
    # the entire answer — which is why it is corrected only on evidence somebody RECORDED.
    # Creation order is UPLOAD order: an organisation seeding its archive uploads v1 after
    # v2 as a matter of course, and silently reversing their question, with a JSON field they
    # never see as the only tell, answers something they did not ask.
    reordered = False
    order_basis = "as_given"
    warnings: list[dict] = []
    # Adjacency is not the question — ANCESTRY is. Checking only the two `supersedes` columns
    # meant `diff(v1, v3)` across a real chain still warned "no supersedes link", while the
    # chain walk below happily used the link; and `diff(v3, v1)` was silently answered
    # backwards with the chain never walked, though the library holds the whole thing. The
    # order check and the chain walk disagreed about what "linked" means.
    if earlier in store.supersession_chain(conn, later):
        order_basis = "supersession"
    elif later in store.supersession_chain(conn, earlier):
        first, second = second, first
        earlier, later = later, earlier
        reordered = True
        order_basis = "supersession"
    else:
        warnings.append(notices.notice(
            "version_order_unverified",
            detail=f"no supersedes link between {earlier} and {later}",
            next_actions=[]))

    # D64: every version BETWEEN the two, when they are actually linked. Comparing only the
    # two endpoints reported a finding v2 explicitly resolved as `no_longer_raised` against
    # v3 — "neither resolved nor repeated", when the library holds the record of it being
    # resolved. That is the library forgetting its own evidence and then hedging about it.
    chain = store.supersession_chain(conn, later)
    through = chain[chain.index(earlier):] if earlier in chain else [earlier, later]

    before = store.latest_evaluation_for_campaign(conn, earlier)
    after = store.latest_evaluation_for_campaign(conn, later)
    # What every judgment along the way recorded as resolved, by finding id. A correction
    # taken at any point in the chain was taken.
    resolved_along_chain: dict = {}
    # `[1:]` — everything AFTER the earlier endpoint. Worth saying that this slice cannot
    # currently change the answer and is kept for intent: a judgment's `resolved` rows name
    # findings from the version BEFORE it, never its own, so folding in the earlier
    # endpoint's rows adds ids that are not in the set being matched against. Review's
    # mutation of it survives the suite and no honest test kills it, because the two forms
    # are equivalent on every input the schema can produce. Left explicit rather than
    # simplified, so that if `resolved` ever names an arbitrary id the slice is already
    # saying which versions are in scope.
    for link in through[1:]:
        judgment = store.latest_evaluation_for_campaign(conn, link)
        for row in ((judgment or {}).get("resolved") or []):
            if row.get("finding_id"):
                resolved_along_chain[row["finding_id"]] = {**row, "in_version": link}

    # `comparable` means both sides carry a STRUCTURED judgment. A pre-§2.4 evaluation reads
    # back with no verdict and an empty findings list, and treating it as comparable made
    # every later finding "newly introduced" — a v2 blamed for everything its own v1 review
    # had also found.
    #
    # The marker is the VERDICT, not a non-empty findings list: an approve with nothing wrong
    # is a complete structured judgment that happens to have no findings, and the whole point
    # of this tool is to recognise the version where everything got fixed. (`get_evaluation`
    # draws the legacy line the same way.)
    structured = [bool(e and e.get("verdict")) for e in (before, after)]

    result = {
        "earlier": {"campaign_id": earlier, "title": first["title"]},
        "later": {"campaign_id": later, "title": second["title"]},
        "arguments_reordered": reordered,
        "order_basis": order_basis,
        # Which records the answer was assembled from. A diff across a chain is a different
        # claim from a diff between two adjacent versions, and a reader cannot tell without
        # being told.
        "through": through,
        "adopted": [], "raised_again": [], "newly_introduced": [], "no_longer_raised": [],
        # BOTH judgments' citations. Only the earlier one was checked, so a later judgment
        # resting on a record since replaced reported nothing — and the later judgment is
        # the one somebody is about to act on.
        "carried_stale": (_stale_citations(conn, before, "earlier")
                          + _stale_citations(conn, after, "later")),
        "record_changes": _record_changes(first, second),
        "comparable": all(structured),
        "warnings": warnings,
    }

    result["counts"] = {name: len(result[name]) for name in
                        ("adopted", "raised_again", "newly_introduced", "no_longer_raised",
                         "carried_stale")}

    if not all(structured):
        if not (before and after):
            missing = "earlier" if not before else "later"
            result["why_not_comparable"] = (
                f"The {missing} version has never been evaluated, so there are no findings "
                f"to compare. What changed between the decks is readable, but which "
                f"corrections were taken is not — that is a judgment about the earlier "
                f"review, and guessing it would mean telling somebody they ignored advice "
                f"nobody checked.")
        else:
            side = "earlier" if not structured[0] else "later"
            result["why_not_comparable"] = (
                f"The {side} version's judgment predates the structured findings schema, so "
                f"it has no findings to compare — only the original free text. Treating it "
                f"as comparable would report every finding in the other version as newly "
                f"introduced.")
            return result
        unjudged = earlier if not before else later
        unjudged_record = store.get_campaign(conn, unjudged)
        result["next_actions"] = actions.trim([actions.action(
            f"Evaluate \u201c{store.get_campaign(conn, unjudged)['title']}\u201d, so the "
            f"two versions can be compared",
            "prepare_evaluation",
            why="A version with no judgment on file cannot be compared with one that has.",
            # Whatever text the record actually has. `detail or title` sent the title alone
            # for a deck-only upload, so accepting "evaluate this version" judged eleven
            # characters — a prefilled placeholder, which §5.2 forbids.
            consent="ask", subject_title=unjudged_record["title"],
            campaign_id=unjudged,
            proposal_text="\n\n".join(
                part for part in (unjudged_record.get("detail"),
                                  unjudged_record.get("deck_text"))
                if part) or unjudged_record["title"])])
        return result

    resolved_by_id = {r["finding_id"]: r for r in (after.get("resolved") or [])
                      if r.get("finding_id")}
    # A correction recorded anywhere along the chain counts, not only in the final judgment.
    resolved_by_id = {**resolved_along_chain, **resolved_by_id}
    later_findings = list(after.get("findings") or [])
    repeats_by_id = {f["repeats"]: f for f in later_findings if f.get("repeats")}
    matched_later: list = []
    later_categories = {f.get("category") for f in later_findings if f.get("category")}

    # Everything identity already settles is taken out before wording is consulted at all,
    # so a resemblance can never outrank a statement somebody made.
    earlier_findings = list(before.get("findings") or [])
    settled = {f.get("id") for f in earlier_findings
               if f.get("id") in resolved_by_id or f.get("id") in repeats_by_id}
    by_text = _pair_up(
        [f for f in earlier_findings if f.get("id") not in settled],
        [f for f in later_findings if not f.get("repeats")])
    text_match = {}
    unsettled = [f for f in earlier_findings if f.get("id") not in settled]
    candidates = [f for f in later_findings if not f.get("repeats")]
    for i, (j, score) in by_text.items():
        text_match[id(unsettled[i])] = (candidates[j], score)

    for finding in earlier_findings:
        entry = {k: finding.get(k) for k in ("id", "severity", "kind", "departure",
                                             "category",
                                             "finding")}
        if finding.get("id") in resolved_by_id:
            row = resolved_by_id[finding["id"]]
            entry["basis"] = "computed"
            entry["now"] = row.get("now")
            # Which version actually recorded it, when that was not the one being compared
            # against. Without it, a v1→v3 diff reported a correction as adopted with `now`
            # text written by v2's reviewer and nothing said so — and the caveat below is
            # only true of the chain case, so a reader has to be able to tell them apart.
            if row.get("in_version") and row["in_version"] != later:
                entry["resolved_in"] = row["in_version"]
                # Between adjacent versions, "adopted" means the later reviewer confirmed it.
                # Across a chain it means confirmed ONCE and not re-examined since: v3's
                # reviewer compared against v2 and never looked at this finding again, so a
                # regression between v2 and v3 would not show here. This module caveats
                # weaker claims than that one.
                entry["caveat"] = (
                    f"Recorded as resolved in {row['in_version']}, not re-examined in "
                    f"{later}. Adopted once, not confirmed since.")
            result["adopted"].append(entry)
            continue

        # Identity first. A later finding that names the earlier one it repeats is a claim
        # somebody made, and the only basis on which "this correction was not taken" can be
        # stated as fact.
        named = repeats_by_id.get(finding.get("id"))
        if named is not None:
            matched_later.append(named)
            result["raised_again"].append({**entry, "basis": "computed", "match": "id",
                                           "raised_again_as": named.get("finding"),
                                           **_reread(finding, named)})
            continue

        # Wording, as a fallback that is labelled as one. Character similarity scored
        # "adidas-affiliated" against "Nike-affiliated" at 0.89 and the same problem reworded
        # at 0.36 — it measures phrasing, not meaning, so it is reported as a resemblance
        # and marked `judged` so nothing downstream states it as fact.
        paired = text_match.get(id(finding))
        if paired is not None:
            again, _score = paired
            matched_later.append(again)
            result["raised_again"].append({
                **entry, "basis": "judged", "match": "text", **_reread(finding, again),
                "similarity": round(_looks_like(finding.get("finding"),
                                                again.get("finding")), 2),
                "raised_again_as": again.get("finding"),
                "caveat": "Matched on how alike the two read, not on either review saying "
                          "they are the same problem. Check both before telling anyone a "
                          "correction was not taken.",
            })
            continue

        entry["basis"] = "computed"
        # The caveat is split by what the later review actually covered, because "we cannot
        # tell" is not equally true in both cases.
        if after.get("verdict") == "approve" and not later_findings:
            entry["caveat"] = ("The later review approved this version with no findings at "
                               "all, which is a positive statement that nothing blocks it — "
                               "so this was most likely addressed, though no one recorded "
                               "it against this finding.")
        elif finding.get("category") and finding["category"] in later_categories:
            entry["caveat"] = (f"The later review did raise other {finding['category']} "
                               f"findings, so it looked at this area and did not raise this "
                               f"one — more likely fixed than overlooked, though nobody "
                               f"recorded it.")
        else:
            entry["caveat"] = ("Neither recorded as resolved nor raised again, and the later "
                               "review raised nothing in this area at all — so it was either "
                               "fixed without being recorded or not looked at. The record "
                               "cannot tell which.")
        result["no_longer_raised"].append(entry)

    for finding in later_findings:
        if finding in matched_later:
            continue
        entry = {k: finding.get(k) for k in ("id", "severity", "kind", "departure",
                                             "category",
                                             "finding")}
        entry["basis"] = "computed"
        result["newly_introduced"].append(entry)

    result["counts"] = {name: len(result[name]) for name in
                        ("adopted", "raised_again", "newly_introduced", "no_longer_raised",
                         "carried_stale")}
    result["next_actions"] = _offer_to_link_versions(first, second, result)
    return result


def _offer_to_link_versions(first: dict, second: dict, diff: dict) -> list[dict]:
    """Offer to record that one version replaces the other — D41, and only here.

    5.2 offered this prefilled from `closest_precedent`: a similarity match, ASSERTED by the
    model. "This replaces that" is a claim about somebody's intent and a similarity score is
    a fact about text, and the two are not the same kind of thing — both reviewers condemned
    it independently. What was wrong was never the offer; it was the evidence.

    **The first version of this fix got the evidence wrong too, and review said so in the
    same words.** It fired whenever a diff of two judged records completed — which is not a
    fact about the records, it is a fact about which two ids the CALLER passed, and the
    caller is the model 5.2 condemned. Two unrelated campaigns got an offer to hide one of
    them, in the same response whose warning said the library cannot tell which came first.

    So the gate is a finding id: a later finding whose `repeats` names an earlier one, or a
    `resolved` row naming an earlier finding. Both are statements somebody made about these
    two records being versions of one brief — which is this module's own rule, that a
    resemblance never outranks a statement. Unrelated campaigns share no finding ids and get
    nothing.

    The offer names the record it would hide, in the label, because that is the consequence
    the user is agreeing to and it is invisible in the arguments. §5.2's other lesson: this
    is offered, never taken — accepting hides a record from every search, and `supersedes`
    can now be cleared, which is what makes offering it defensible at all.
    """
    # Either endpoint already linked, in either direction. Reading only the two `supersedes`
    # columns missed fan-in: an earlier record already replaced by a THIRD record was still
    # offered up to be replaced again, silently making two records claim the same one.
    if (second.get("supersedes") or first.get("supersedes")
            or second.get("is_superseded") or first.get("is_superseded")):
        return []
    # A statement, not a resemblance: an id-matched repeat, or a recorded resolution.
    linked = any(entry.get("match") == "id" for entry in diff["raised_again"])
    linked = linked or bool(diff["adopted"])
    if not linked:
        return []
    return actions.trim([actions.action(
        f"Record that “{second['title']}” replaces “{first['title']}”, "
        f"which removes “{first['title']}” from future search results",
        "update_campaign",
        why="A judgment of one names a finding from the judgment of the other by id, so "
            "somebody has already treated these as versions of one brief — but nothing links "
            "the records, so searches still return both and the older can be cited as "
            "precedent for the newer.",
        consent="ask", campaign_id=second["id"], supersedes=first["id"])])


def _reread(earlier: dict, later: dict) -> dict:
    """Did the second reader read the same difference differently? (§6.2)

    A finding raised in both versions is reported as "raised again", which reads as a
    correction not taken. But a departure the first review called a `regression` and the
    second called a `possible_improvement` is not an ignored correction — it is the library
    changing its mind, and it is the review's own headline case: the UAE brief's claw machine
    was a departure that turned out better than the precedent. Reported without this, the
    comparison files exactly that story as a repeat defect.
    """
    before, after = earlier.get("departure"), later.get("departure")
    if not before or not after or before == after:
        return {}
    return {
        "departure_now": after,
        # A stable code, not prose: `softened` is the claw machine, `hardened` is a second
        # reader who decided the difference was worse than the first thought.
        "reread": ("softened" if _DEPARTURES.index(after) > _DEPARTURES.index(before)
                   else "hardened"),
    }


def _record_changes(first: dict, second: dict) -> dict:
    """What the two records themselves say differently.

    A v2 that drops a market from a LATAM brief is the largest change a version can carry,
    and it diffed as identical — the comparison only ever looked at findings. Same for a
    performance tag that lost its `verified` source, which changes how the record weighs as
    precedent. Set differences on structured fields, so all of it is genuinely computed.

    Unchanged fields are absent rather than listed as empty: a diff that lists everything it
    checked is a diff nobody reads.
    """
    changes: dict = {}

    for field in ("markets",):
        before, after = set(first.get(field) or []), set(second.get(field) or [])
        if before != after:
            changes[field] = {"removed": sorted(before - after),
                              "added": sorted(after - before)}

    def tag_set(record):
        return {(t.get("value"), t.get("source")) for t in (record.get("tags") or [])}

    before_tags, after_tags = tag_set(first), tag_set(second)
    if before_tags != after_tags:
        before_values = {v for v, _ in before_tags}
        after_values = {v for v, _ in after_tags}
        entry = {"removed": sorted(before_values - after_values),
                 "added": sorted(after_values - before_values)}
        # A tag whose value survived but whose source did not: a performance claim that was
        # backed by measurement and now is not weighs differently as precedent.
        downgraded = sorted(v for v, source in before_tags
                            if source == "verified" and (v, "verified") not in after_tags
                            and v in after_values)
        if downgraded:
            entry["no_longer_verified"] = downgraded
        changes["tags"] = entry

    for field in ("status", "collection", "market", "region"):
        if (first.get(field) or None) != (second.get(field) or None):
            changes[field] = {"was": first.get(field), "now": second.get(field)}

    return changes


def _stale_citations(conn, evaluation, cited_by: str) -> list[dict]:
    """Citations the earlier judgment rested on that the library no longer treats as current.

    The computable half of "which facts went stale": a verdict that leaned on a campaign
    since replaced leaned on something no search would now return.
    """
    if not evaluation:
        return []
    superseded = store.get_superseded_campaign_ids(conn)
    stale = []
    for cited in evaluation.get("cited_ids") or []:
        if cited not in superseded:
            continue
        record = store.get_campaign(conn, cited)
        if not record:
            continue
        stale.append({
            "campaign_id": cited,
            "cited_by": cited_by,
            "title": record["title"],
            "superseded_by": record["superseded_by"],
            "basis": "computed",
            "why_it_matters": "The earlier judgment cited this record, and it has since "
                              "been replaced — so that part of the reasoning rests on "
                              "something no search would return today.",
        })
    return stale


# ── coverage (§5.5, idea E) ─────────────────────────────────────────────────
#
# "list_campaigns returns eight rows. The question a marketer actually has is where the holes
# are: which markets, collections and launch types are represented, which have outcomes,
# which have only one example carrying all the weight."
#
# The third question is the one nothing else in the product answers. A cell carried by a
# single campaign produces judgments that are really that one campaign's opinion, and the
# similarity score looks identical whether it came from one precedent or nine.
#
# Distinct from gaps() (§5.3), which is "what do I fix first" — one ranked list with actions.
# This is "what do I have", a matrix. §5.3's market gap is computed from these cells so the
# two surfaces cannot describe the same library differently (tracker D55).

# Market x collection x stage is multiplicative, and this goes inside a tool result somebody
# has to read.
MAX_COVERAGE_CELLS = 25

# Worst first. `no_outcomes` outranks `single_example` because a cell with two campaigns and
# nothing measured compares a proposal against what was planned, which is weaker than one
# measured example.
_EVIDENCE_ORDER = ("no_outcomes", "single_example", "measured", "not_yet_run")


def coverage(conn) -> dict:
    """Where the library is thick and where it is thin, by market, collection and stage.

    Cells OVERLAP by construction: a campaign that ran in three markets belongs to three
    cells, because "what do I have in Colombia" is asked per market. So the counts do not sum
    to the number of campaigns, and `campaigns_total` plus the note say so rather than
    leaving somebody to add them up.
    """
    superseded = store.get_superseded_campaign_ids(conn)
    # A `stub` is "a placeholder record" by the product's own definition, so counting one as
    # evidence a judgment can lean on describes content that is not there — and it arrives
    # with no status, producing a cell whose stage nothing could explain.
    campaigns = [c for c in store.list_campaigns(conn)
                 if c.get("record_type") not in ("reference", "stub")
                 and c["id"] not in superseded]
    measured = store.campaigns_with_actual_metrics(conn)

    if not campaigns:
        return {
            "cells": [], "cells_total": 0, "thin": [], "campaigns_total": 0,
            "markets": [], "collections": [], "stages": [],
            "note": "The library is empty, so there is nothing to have coverage of.",
            "next_actions": actions.to_first_upload(),
        }

    # Keyed case-insensitively, displayed as first seen. The bucket key used the raw
    # spelling while `_markets_of` folded case only WITHIN a record, so "LATAM" and "latam"
    # became two cells — one reading `no_outcomes` and the other `single_example`, for a
    # library that `filter_campaign_ids` treats as one market of two. `collection` had the
    # same split, and it matches on `LOWER(collection)` in the store. This is the §5.3 bug
    # one file over, and it made this item's "the two cannot disagree" claim false.
    buckets: dict[tuple, list] = {}
    display: dict[str, str] = {}

    def fold(value):
        if value is None or not str(value).strip():
            return None
        text = str(value).strip()
        display.setdefault(text.lower(), text)
        return text.lower()

    for campaign in campaigns:
        for market in _markets_of(campaign):
            key = (fold(market), fold(campaign.get("collection")),
                   campaign.get("status") or None)
            buckets.setdefault(key, []).append(campaign)

    cells = []
    for (market, collection, stage), rows in buckets.items():
        with_outcomes = sum(1 for c in rows if c["id"] in measured)
        cells.append({
            "market": display.get(market) if market else None,
            "collection": display.get(collection) if collection else None,
            "stage": stage,
            "campaigns": len(rows),
            "with_outcomes": with_outcomes,
            # A campaign that has not run cannot be missing its results, and saying so is
            # the complaint nobody can answer that §5.3 wrote out. Beyond that, both markers
            # can be true at once and the one that costs more wins — the counts above are
            # there so nothing hides behind it.
            "evidence": ("not_yet_run" if stage != "concluded"
                         else "no_outcomes" if not with_outcomes
                         else "single_example" if len(rows) == 1
                         else "measured"),
            "campaign_ids": [c["id"] for c in rows][:5],
        })

    # `cells` is for BROWSING, so it is ordered the way somebody reads a table. `thin` below
    # carries the ranking. When both were sorted worst-first and then truncated, a library
    # with 25 weak cells returned the two lists byte-identical: no thick cell was shown, ten
    # were hidden with no count of what, and `single_example` — this item's own headline —
    # never appeared because `no_outcomes` filled the list.
    cells.sort(key=lambda c: (c["market"] or "~", c["collection"] or "~",
                              c["stage"] or "~"))
    ranked = sorted(cells, key=lambda c: (_EVIDENCE_ORDER.index(c["evidence"]),
                                          -c["campaigns"], c["market"] or ""))
    thin = [c for c in ranked if c["evidence"] in ("no_outcomes", "single_example")]

    # D66: when nothing in the library is strong, listing every weak cell is the matrix
    # again — and the answer at that size is not "fix LATAM", it is the first-run guidance.
    # "Nothing here is measured yet" is about the LIBRARY, not about the cell markers: five
    # cells of one measured campaign each read `single_example` rather than `measured`, and
    # that is a library with real evidence in it, not one to hand back to the first-run
    # guidance.
    everything_is_thin = bool(campaigns) and not measured

    # D68: the same campaign sits in one cell per market, so a campaign that ran in three
    # markets had the same fix offered three times. The grid is right for "what do I have in
    # Colombia"; the ACTION is per campaign.
    unmeasured: dict[str, dict] = {}
    for cell in thin:
        for campaign in [c for c in campaigns if c["id"] in cell["campaign_ids"]]:
            if campaign["id"] in measured:
                continue
            entry = unmeasured.setdefault(campaign["id"], {
                "campaign_id": campaign["id"], "title": campaign["title"], "markets": []})
            if cell["market"] and cell["market"] not in entry["markets"]:
                entry["markets"].append(cell["market"])

    shown = cells[:MAX_COVERAGE_CELLS]
    hidden: dict = {}
    for cell in cells[MAX_COVERAGE_CELLS:]:
        hidden[cell["evidence"]] = hidden.get(cell["evidence"], 0) + 1

    return {
        "cells": shown,
        "cells_total": len(cells),
        "hidden": hidden,
        # Counts across EVERY cell, truncated or not. The list can lose detail; the shape of
        # the library must not depend on where the cut fell — a report showing 25 weak cells
        # out of 35 said nothing about the 5 measured ones it had dropped.
        # Every kind, including the zeros: "measured: 0" is the most informative line in the
        # report, and omitting it makes its absence indistinguishable from truncation.
        "evidence_summary": {kind: sum(1 for c in cells if c["evidence"] == kind)
                             for kind in _EVIDENCE_ORDER},
        # A matrix is something to browse; the answer is which cells are weak. Worst first,
        # so the first line is the one that matters.
        "thin": [] if everything_is_thin else thin[:MAX_COVERAGE_CELLS],
        # Not the total of a list that is not being shown: `thin: []` beside
        # `thin_total: 3` is an inconsistent pair for a reader.
        "thin_total": 0 if everything_is_thin else len(thin),
        "thin_summary": (f"Nothing in this library is measured yet: {len(campaigns)} "
                         f"campaign(s) across {len({c['market'] for c in cells})} cell(s), "
                         f"none with results on file. The place to start is not a "
                         f"particular market."
                         if everything_is_thin else None),
        "unmeasured_campaigns": sorted(unmeasured.values(),
                                       key=lambda c: (-len(c["markets"]), c["title"]))[:10],
        # The measurement offer, not the first-run path: the path never mentions results, so
        # once liked/not_liked/rulebook existed the summary named measurement as the problem
        # and offered nothing at all — while `gaps()` on the same library offered add_metrics.
        "next_actions": (_offer_to_measure(conn, unmeasured) if everything_is_thin else []),
        "campaigns_total": len(campaigns),
        "markets": sorted({c["market"] for c in cells if c["market"]}),
        "collections": sorted({c["collection"] for c in cells if c["collection"]}),
        "stages": sorted({c["stage"] for c in cells if c["stage"]}),
        "note": ("A campaign that ran in more than one market appears in a cell for each, so "
                 "the cell counts overlap and do not add up to campaigns_total. `cells` is "
                 "ordered for reading; `thin` is the answer, ordered worst first: "
                 "`no_outcomes` means nothing in that cell was ever measured, "
                 "`single_example` means one campaign is carrying every judgment about it. "
                 "`not_yet_run` is neither — a campaign that has not concluded cannot have "
                 "results yet."),
    }


def _offer_to_measure(conn, unmeasured: dict) -> list[dict]:
    """The single thing that would lift a wholly unmeasured library out of it.

    Falls back to the first-run path only when there is nothing to measure yet — a library
    with no concluded campaign cannot record results for one.
    """
    candidates = sorted(unmeasured.values(), key=lambda c: (-len(c["markets"]), c["title"]))
    if not candidates:
        return readiness(conn)["shortest_path"]
    first = candidates[0]
    return actions.trim([actions.action(
        f"Record what \u201c{first['title']}\u201d actually achieved",
        "add_metrics",
        why="Nothing in the library has measured results, so every judgment compares a "
            "proposal to what was planned rather than to what happened.",
        consent="ask", needs=["the numbers, or what happened in words"],
        campaign_id=first["campaign_id"])])


def _markets_of(campaign: dict) -> list:
    """Every market this campaign counts towards, or `[None]` when it has none.

    The `markets` list is the only way to express a multi-country activation, so ignoring it
    would report a real LATAM campaign as covering nothing. And a record with no market at
    all is most of a young library — dropping those would describe a library nobody has.
    """
    named = []
    for raw in (campaign.get("market"), campaign.get("region"),
                *(campaign.get("markets") or [])):
        if raw and str(raw).strip():
            value = str(raw).strip()
            if value.lower() not in {m.lower() for m in named}:
                named.append(value)
    return named or [None]


# ── the first run (§5.6, idea F) ────────────────────────────────────────────
#
# "A fresh install has no campaigns and therefore no opinions, and nothing tells a new user
# how many records it takes before judgments become useful, or which ones to add first."
#
# "How many records" has no honest numeric answer, and giving one would be the kind of
# confident number this whole review was written against. Usefulness depends on WHAT is in
# the library: two contrasting briefs make the liked/not-liked comparison work at two
# records, and a hundred concluded campaigns with nothing measured still cannot say whether
# any of it worked. So readiness is expressed as what the product can and cannot do given
# what is actually present, and every limit names the record that would lift it — a
# capability statement nobody can act on is a disclaimer.


def readiness(conn) -> dict:
    """What this library can and cannot do yet, and the shortest path to more.

    The path is the review's own prescription, in its order, because it is a path and not a
    menu: one brief you liked, one you did not, the rulebook. The contrast is the point —
    two briefs somebody liked teach nothing about the axis they are asking the product to
    judge on.
    """
    superseded = store.get_superseded_campaign_ids(conn)
    records = [c for c in store.list_campaigns(conn) if c["id"] not in superseded]
    campaigns = [c for c in records if c.get("record_type") not in ("reference", "stub")]
    measured = store.campaigns_with_actual_metrics(conn)

    has_rulebook = any(c.get("record_type") == "reference" for c in records)
    # Case-folded, because tags are freeform and stored as typed while the store folds them
    # when filtering. "Liked" left a marketer who had done exactly what the path asked being
    # told forever to add a campaign they liked.
    # Per RECORD, not pooled: a single campaign tagged both `liked` and `not_liked` used to
    # satisfy the axis on its own, and "the library holds both" was then technically true and
    # substantively false. The contrast this product reasons from is between records, and one
    # record cannot be the counter-example to itself.
    def reactions_of(campaign):
        return {str(t.get("value") or "").strip().lower()
                for t in (campaign.get("tags") or [])}

    liked_records = [c for c in campaigns if "liked" in reactions_of(c)]
    disliked_records = [c for c in campaigns
                        if reactions_of(c) & {"not_liked", "not liked"}]
    reactions = {r for c in campaigns for r in reactions_of(c)}
    liked = "liked" in reactions
    # `mixed_reaction` is deliberately NOT a dislike. The item's own rationale asks for "one
    # you did not like", and a mixed reaction is not that contrast — counting it would tell
    # the user the axis works when it does not.
    disliked = "not_liked" in reactions or "not liked" in reactions
    with_outcomes = [c for c in campaigns if c["id"] in measured]
    concluded = [c for c in campaigns if c.get("status") == "concluded"]

    can: list[dict] = []
    cannot: list[dict] = []

    if len(campaigns) >= 2:
        # "on file", not "run before": `gaps()` draws the line that a campaign which has not
        # concluded cannot have results, and this said "what you have run" for a library of
        # proposals.
        can.append({"code": "compare_to_precedent",
                    "what": "Compare a new brief against what you already have on file, and "
                            "say where it departs from it."})
    else:
        cannot.append({
            "code": "compare_to_precedent",
            "what": "Compare a new brief against anything — with fewer than two campaigns "
                    "there is nothing to compare against, so a judgment would be an opinion "
                    "rather than a reading of your own record.",
            "needs": "at least two past campaigns",
        })

    contrasting = bool({c["id"] for c in liked_records}
                       - {c["id"] for c in disliked_records}) and bool(disliked_records)
    if contrasting:
        can.append({"code": "weigh_reactions",
                    "what": "Weigh what you liked against what you did not, because the "
                            "library holds both."})
    else:
        cannot.append({
            "code": "weigh_reactions",
            "what": "Tell what you like from what you do not — every record here reads the "
                    "same way on that axis, so it cannot be used to judge a new brief.",
            "needs": "a campaign you were unhappy with, tagged as such"
                     if liked else "one campaign you liked and one you did not",
        })

    # One code, one list. It used to appear on BOTH for any partly measured library, so a
    # reader keying on the code could not tell which side won.
    if with_outcomes and len(with_outcomes) == len(campaigns):
        can.append({"code": "say_what_worked",
                    "what": f"Say whether something worked: all "
                            f"{len(with_outcomes)} campaign(s) here have measured results."})
    elif with_outcomes:
        can.append({"code": "say_what_worked",
                    "what": f"Say whether something worked for the {len(with_outcomes)} of "
                            f"{len(campaigns)} campaign(s) with measured results. For the "
                            f"rest, comparisons are to what was planned, not what happened."})
    else:
        cannot.append({
            "code": "say_what_worked",
            "what": "Say whether anything worked — nothing here has measured results, so "
                    "every comparison is to what was planned rather than what happened.",
            # Asking for "a concluded campaign's results" when nothing has concluded is a
            # request nobody can satisfy.
            "needs": ("the results of a campaign that has concluded" if concluded
                      else "a campaign that has concluded, and its results"),
        })

    if has_rulebook:
        # NOT "check a brief against a rule". `has_rulebook` is "some reference record
        # exists", and nothing pins, fetches or checks against it — `prepare_evaluation` is
        # similarity retrieval, so the rulebook reaches the evidence only if it happens to
        # rank. Promising the "this breaks your own rule" finding would be exactly the
        # confident, unfounded claim this item exists to prevent (§7.5/§12.1 make it true).
        can.append({"code": "rulebook_on_file",
                    "what": "Cite your guidelines when they happen to be among the most "
                            "similar records retrieved for a brief."})
    cannot.append({
        "code": "check_against_rules",
        "what": "Check a brief against a rule reliably. Guidelines on file are retrieved by "
                "similarity like anything else, so a guardrail that is not retrieved is not "
                "a guardrail — it can say \u201cthis differs from what you did in Peru\u201d, "
                "which invites an argument, but not \u201cthis breaks your own rule\u201d, "
                "which does not.",
        "needs": ("the rulebook to be pinned rather than retrieved, which is planned work"
                  if has_rulebook else "your brand guidelines, uploaded as reference "
                                       "material"),
    })

    path = _shortest_path(liked, disliked, has_rulebook)
    if not campaigns:
        stage = "empty"
        can = []
    elif len(campaigns) < 2:
        stage = "first_records"
    elif not with_outcomes or path:
        # `working` means the path is walked AND something is measured. Measured-alone
        # reported `working` with all three steps outstanding, so the guidance stopped being
        # attached exactly while it was still needed.
        stage = "thin"
    else:
        stage = "working"

    return {
        "stage": stage,
        "campaigns": len(campaigns),
        "with_outcomes": len(with_outcomes),
        "has_rulebook": has_rulebook,
        "can": can,
        "cannot": cannot,
        "shortest_path": path,
        "note": ("Say the stage and what it cannot do yet before giving any judgment from a "
                 "library this size — a confident, evidence-free verdict is the thing a new "
                 "user will believe. `shortest_path` is ordered: it is a path, not a menu."),
    }


def _shortest_path(liked: bool, disliked: bool, has_rulebook: bool) -> list[dict]:
    """The review's three, in its order, minus what is already done.

    Every one of these is an offer whose arguments only the user has — there is nothing in an
    empty library to prefill from — so `needs` is what keeps "accepting is one step" honest
    (tracker D43).
    """
    steps = []
    if not liked:
        steps.append(actions.action(
            "Add one campaign you were happy with",
            "upload_campaign",
            why="The library has nothing it knows you liked, so it has no positive example "
                "to reason from.",
            consent="ask",
            needs=["what it was called", "its deck or a description of it",
                   "that you were happy with it"],
            tags=[{"value": "liked"}]))
    if not disliked:
        steps.append(actions.action(
            "Add one campaign you were not happy with",
            "upload_campaign",
            why="Without a contrast every record reads the same way, and the comparison the "
                "product exists to make cannot be made.",
            consent="ask",
            needs=["what it was called", "its deck or a description of it",
                   "what you did not like about it"],
            tags=[{"value": "not_liked"}]))
    if not has_rulebook:
        steps.append(actions.action(
            "Add your brand guidelines as the rulebook",
            "upload_campaign",
            why="A rule that is not on file can only be reported as a departure from "
                "precedent, which invites an argument.",
            consent="ask",
            needs=["the guidelines document, or the rules in your own words"],
            record_type="reference"))
    return steps


def readiness_for_listing(conn) -> Optional[dict]:
    """The guidance to attach to a listing, or None once the library works.

    Replaces a `list_campaigns_with_readiness` that returned its own differently-shaped rows
    and that no tool ever called — so the guidance reached nobody, which is the exact failure
    the tracker row behind it was written about. This returns only the readiness part, and
    the tool keeps its own field projection.

    Attached only while the library is NOT working: guidance that never stops appearing is
    the thing nobody reads.
    """
    state = readiness(conn)
    return state if state["stage"] != "working" else None


def published_tool_parameters() -> dict[str, list[str]]:
    """What each tool actually takes, right now, read off the functions themselves (§3.2).

    The version is the half the review asked for, and the weaker half: a version string says
    the server changed, not that the schema in your hand is missing a parameter — and the
    reported symptom was silent absence. "Several parameters that were live and working —
    markets, status, tag source, match_all_tags, confirm — were absent from the schemas in
    use, and had to be rediscovered by trial and error against a server that already
    supported them."

    This is the list a caller holding a cached schema can compare against, so the gap can be
    named instead of guessed at. Generated from the signatures rather than maintained by
    hand, because a hand-written copy would drift from the tools exactly the way the client's
    cache did — the defect reproduced inside its own fix.
    """
    import inspect

    try:
        import mcp_server
    except Exception:                              # noqa: BLE001
        # health_check has to answer on a machine where things are broken — that is when
        # somebody runs it. An import failure here must cost the tool inventory, never the
        # report that says which component is down.
        return {}

    # From the server's own registry, not a list beside it. The first version iterated a
    # hand-typed TOOL_NAMES — the hand-maintained copy this function's docstring says it
    # refuses to have — and a tool registered without editing that tuple was silently
    # omitted. Review deleted a name from it and the entire suite stayed green.
    published: dict[str, list[str]] = {}
    for tool in mcp_server.mcp._tool_manager.list_tools():
        fn = getattr(tool, "fn", None) or getattr(mcp_server, tool.name, None)
        if fn is None:
            continue
        published[tool.name] = [p for p in inspect.signature(fn).parameters
                                if p not in ("self", "ctx")]
    return published


def stale_schema_parameters(published: dict, cached: dict) -> dict[str, list[str]]:
    """Per tool, the parameters this server accepts that the caller's schema does not list.

    Empty means the caller is current. Anything else is the answer to "why did passing
    markets= do nothing" without a round of trial and error — and the cue to tell them to
    fully quit and reopen the host app, since closing the window leaves the server running.
    """
    missing = {}
    for tool, params in published.items():
        if tool not in cached:
            missing[tool] = list(params)
            continue
        gap = [p for p in params if p not in set(cached[tool])]
        if gap:
            missing[tool] = gap
    return missing


def finish_indexing(conn, *, campaign_id: Optional[str] = None) -> dict:
    """Finish records that are stored but not yet searchable (defect 05).

    The review's complaint was that there was no route back: two image assets sat
    fingerprinted with no vector, and "the only recovery is to upload the images again".
    Text had the same hole. Item 2.1 then made partial state a DESIGNED outcome (a time
    budget stops mid-deck rather than outliving the transport), so a way to finish the job
    stopped being optional.

    Named for what the user wants rather than the mechanism: nothing here is *re*-embedded,
    these rows were never embedded at all. `reembed` stays free for the genuinely different
    operation 7.6 needs — re-embedding everything when the model identity changes.

    Works on both modalities because both leave a row: chunks are inserted before they are
    embedded, and deck images are stored and fingerprinted in a first pass precisely so an
    interrupted run leaves something to come back to.

    Resumable, but only claims to be when continuing would actually help. An embedder that
    is down fails every item instantly; advising "call again" there invites a loop against a
    component that is not coming back on its own."""
    if campaign_id is not None:
        if not campaign_id.strip():
            raise ValueError("campaign_id cannot be empty — omit it to finish everything")
        if store.get_campaign(conn, campaign_id) is None:
            raise ValueError(f"campaign {campaign_id} not found")

    deadline = time.monotonic() + config.TOOL_TIME_BUDGET_SECONDS
    failures: dict[str, int] = {}
    unfixable = 0
    sections_indexed = images_indexed = 0
    touched: set[str] = set()
    ran_out_of_time = False

    def note(exc: Exception) -> bool:
        """Record a failure, collapsed by cause. Returns whether it can never succeed."""
        failures[str(exc)] = failures.get(str(exc), 0) + 1
        return isinstance(exc, FileNotFoundError)

    def as_notice(reason: str, count: int) -> dict:
        """A failure reason as a §3.1 notice, so this surface speaks the same language as
        every other. These were raw `str(exc)` until review pointed out that the sibling of
        the tool defect 09 was reported against was still handing a marketer
        `HTTPConnectionPool(host='localhost', port=11434)`."""
        lowered = reason.lower()
        if "vision" in lowered or "weights" in lowered or "clip" in lowered:
            return notices.notice("visual_search_offline", detail=reason, count=count,
                                  remedy=vision.remedy or
                                  notices.remedy_for("visual_search_offline"))
        if "embedder" in lowered or "could not reach" in lowered or "timed out" in lowered:
            return notices.notice("text_search_offline", detail=reason, count=count,
                                  cause="embedder_unreachable")
        if "no such file" in lowered or "not found" in lowered:
            return notices.notice("image_not_stored", detail=reason, count=count)
        return notices.notice("chunk_not_embedded", detail=reason, count=count)

    for chunk in store.get_unembedded_chunks(conn, campaign_id):
        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0:
            ran_out_of_time = True
            break
        try:
            vec = embedding.embed(chunk["text"], timeout=remaining_time)
            vectorstore.add(conn, chunk["id"], vec)
            store.set_chunk_embedded(conn, chunk["id"])
            sections_indexed += 1
            touched.add(chunk["campaign_id"])
        except Exception as exc:
            if note(exc):
                unfixable += 1

    # Vision needs its model; if that is what is broken there is no point spending the
    # budget discovering it once per asset, and a cold load inside this call is exactly the
    # first-use cost item 2.1 exists to keep out of handlers.
    vision = clip_embed.weights_status()
    if not vision.ok:
        assets = store.get_unembedded_assets(conn, campaign_id)
        if assets:
            failures[f"{vision.reason} {vision.remedy}"] = len(assets)
    else:
        for asset in store.get_unembedded_assets(conn, campaign_id):
            if time.monotonic() >= deadline:
                ran_out_of_time = True
                break
            path = config.ASSET_DIR / asset["file_path"]
            try:
                if not path.is_file():
                    raise FileNotFoundError(
                        f"the stored image file is missing, so it cannot be indexed; "
                        f"re-upload the deck if its visual matches matter")
                vec = clip_embed.embed_image(path)
                vectorstore.add(conn, asset["id"], vec, space="asset")
                store.mark_asset_embedded(conn, asset["id"])
                images_indexed += 1
                touched.add(asset["campaign_id"])
            except Exception as exc:
                if note(exc):
                    unfixable += 1

    # Only campaigns this run actually changed: refreshing every campaign's rollup rewrote
    # updated_at on untouched records, showing a modification that never happened.
    for cid in touched:
        counts = store.chunk_counts(conn, cid)
        if counts["total"]:
            store.mark_embedded(conn, cid, counts["embedded"] == counts["total"])

    left = store.count_unembedded(conn, campaign_id)
    # Items that fail identically every run are NOT "remaining": counting them there means
    # `complete` never becomes true, and a caller following the advice loops forever on
    # something no amount of retrying can fix.
    remaining = max(0, left["chunks"] + left["assets"] - unfixable)
    errors = [{"reason": reason, "count": count}
              for reason, count in sorted(failures.items(), key=lambda kv: -kv[1])]
    # The same failures in the shape every other surface uses (§3.1). `errors` keeps the raw
    # reasons for support and for the tests that read them.
    warnings = notices.collapse([as_notice(reason, count)
                                 for reason, count in failures.items()])

    indexed = sections_indexed + images_indexed
    result = {
        "indexed": indexed,
        "sections_indexed": sections_indexed,
        "images_indexed": images_indexed,
        "remaining": remaining,
        "failed": unfixable,
        "complete": remaining == 0,
        "errors": errors,
        "warnings": warnings,
        "outstanding": store.outstanding_by_campaign(conn, campaign_id)[:5],
    }

    if remaining and indexed and ran_out_of_time:
        result["note"] = (
            f"indexed {indexed} items; {remaining} still to go. Call finish_indexing again "
            f"to continue — it resumes where it stopped. Keep going without stopping to ask, "
            f"unless the user is waiting on something else; report once at the end."
        )
    elif remaining and not indexed:
        # The remedy, not the traceback: this line used to interpolate str(exc) and hand a
        # marketer an HTTPConnectionPool repr.
        cause = (warnings[0]["affects"] if warnings
                 else "the indexer did not respond")
        result["note"] = (
            f"nothing could be indexed. {cause} Calling again will not help until that is "
            f"fixed — say what is wrong, and what would fix it (see warnings), instead of "
            f"retrying."
        )
    elif remaining:
        result["note"] = (
            f"indexed {indexed} items; {remaining} still to go, and the ones attempted this "
            f"time failed. Check the errors before calling again."
        )
    elif unfixable:
        result["note"] = (
            f"indexed {indexed} items. {unfixable} cannot be indexed at all (see errors) and "
            f"were skipped; nothing else is outstanding."
        )
    return result


def find_similar_with_context(conn, **kwargs) -> dict:
    """find_similar, plus a warning when the library itself is incompletely indexed.

    The review's second complaint about partial state was that a half-indexed record is
    "silently invisible to every search while still appearing in list_campaigns". Listing it
    honestly fixes the listing; the moment it actually misleads someone is here — asking
    "what have we run in Mexico" and getting a confident answer that quietly omits the deck
    still waiting to be indexed. An absent result cannot announce itself, so the search has
    to."""
    matches = find_similar(conn, **kwargs)
    # Deliberately empty. Searching is not a step in a sequence — what somebody does after
    # reading results depends entirely on why they searched, and an offer that is always
    # present is the one nobody reads. The key is still here so every surface has one shape.
    return {"matches": matches, "next_actions": [],
            "warnings": _incompleteness_warnings(conn)}


# The three ways somebody's words end up attached to a deck. `speaker_note` is what the
# author wrote to themselves; `comment` is what a reviewer left on it; `annotation` is a PDF
# mark-up, which is either of those depending on how the file was made — a PDF export turns
# speaker notes into annotations, so the kind records the FORMAT, not the authority. Weighing
# by authority is §11.5's job, configured in the rulebook rather than inferred here.
_COMMENTARY_KINDS = ("speaker_note", "comment", "annotation")


def _wanted_commentary_kinds(include_commentary) -> Optional[set]:
    """None = everything (no layer filter). A set = body chunks plus commentary of those
    kinds; the empty set is body only.

    A boolean could only ask "did anyone write anything anywhere". "What did the CLIENT say"
    is a different question, and the finer kind was stored where nothing could filter on it,
    so the field was carried without being usable.
    """
    if include_commentary is True:
        return None
    if include_commentary is False:
        return set()
    if isinstance(include_commentary, str):
        wanted = {include_commentary}
    elif isinstance(include_commentary, (list, tuple, set)):
        wanted = set(include_commentary)
    else:
        raise ValueError(f"include_commentary must be true, false, or a list of "
                         f"{list(_COMMENTARY_KINDS)}; got {include_commentary!r}")
    unknown = wanted - set(_COMMENTARY_KINDS)
    if unknown:
        raise ValueError(f"include_commentary must be true, false, or a list of "
                         f"{list(_COMMENTARY_KINDS)}; got {sorted(unknown)}")
    return wanted


def _chunk_is_wanted(conn, chunk_id: str, wanted_kinds: set) -> bool:
    """A body chunk always survives the commentary filter; a commentary chunk survives only
    if its kind was asked for."""
    import json as _json

    chunk = store.get_chunk(conn, chunk_id)
    if not chunk or chunk.get("kind") != "commentary":
        return True
    if not wanted_kinds:
        return False
    source = _json.loads(chunk["source"]) if chunk.get("source") else {}
    return source.get("kind") in wanted_kinds


def _matched_source(matched: Optional[dict]) -> dict:
    """Attribution for a commentary match: who said it, when, and which page or slide. A
    body chunk has none of these, so it contributes nothing rather than a row of nulls."""
    import json as _json

    raw = (matched or {}).get("source")
    if not raw:
        return {}
    source = _json.loads(raw) if isinstance(raw, str) else raw
    return {"matched_anchor": source.get("anchor"),
            "matched_author": source.get("author"),
            "matched_date": source.get("date"),
            "matched_commentary_kind": source.get("kind")}


def find_similar(conn, *, text: Optional[str] = None, campaign_id: Optional[str] = None,
                 top_k: int = 5, record_type: Optional[str] = None, status: Optional[str] = None,
                 tags: Optional[Union[str, dict, list]] = None, match_all_tags: bool = False,
                 region: Optional[str] = None, market: Optional[str] = None,
                 markets: Optional[Union[str, list]] = None, collection: Optional[str] = None,
                 full_detail: bool = False,
                 include_commentary: Union[bool, list, str] = True) -> list[dict]:
    """
    Rank prior campaigns by semantic similarity to `text` (or to an existing campaign's
    own content). Searches at chunk level (§6.1 — one vector per slide/section) and rolls
    up to the best-matching chunk per campaign, so a long deck can still match on the one
    section that's actually relevant. Excludes the query campaign's own chunks.

    §6.2: if any of record_type/status/tags/region/market/markets/collection is given,
    campaigns are filtered to that structured criteria FIRST, then ranked by similarity only
    within that set — otherwise pure vector search on a single-brand corpus returns
    everything as "similar." markets differs from region/market: a single country/sub-market
    string (or list, ANY-match), matched by membership against a campaign's full activation
    list, for when region/market's single exact-match value can't represent a multi-country
    campaign. tags is a list of plain strings (match that value, any source) and/or
    {value, source} objects (match that value AND require that specific source) — mix
    freely, e.g. tags=["liked", {"value": "underperformed", "source": "verified"}] with
    match_all_tags=True for the actual quadrant query (a creative-reaction tag can never
    itself be "verified" the way a performance tag can, so a single global verified-only
    flag can't express this — per-tag precision can).

    §6.8: each row's freeform `detail` is trimmed to config.EVIDENCE_DETAIL_SUMMARY_CHARS by
    default (`detail_truncated` flags when that happened) — a similarity scan over several
    campaigns shouldn't pay for every full brief. Pass full_detail=True for the untrimmed
    text, or call get_campaign(campaign_id) for the full record.

    Returns evidence rows — id, title, similarity, detail, the matched excerpt, and
    metrics — for Claude to reason over.
    """
    if campaign_id:
        row = conn.execute("SELECT detail, deck_text FROM campaigns WHERE id = ?", (campaign_id,)).fetchone()
        if not row:
            raise ValueError(f"campaign {campaign_id} not found")
        text = "\n\n".join(p for p in (row["detail"], row["deck_text"]) if p)
    if not text or not text.strip():
        raise ValueError("provide text (or a campaign_id that has content) to search by")

    qvec = embedding.embed(text)
    wanted_kinds = _wanted_commentary_kinds(include_commentary)
    filters_given = any([record_type, status, tags, region, market, markets, collection])
    superseded_ids = store.get_superseded_campaign_ids(conn)

    # Narrowing by layer is a FILTER, and has to happen where the other filters happen.
    # Applied after the ANN over-fetch window instead, it re-created the starvation bug the
    # branch below already carries a comment about, one layer down: a deck whose 25 speaker
    # notes filled the window lost its body chunk before the filter ever saw it, and the
    # campaign vanished from a search it plainly matched. Excluding a layer must narrow what
    # can MATCH, never delete a record.
    if filters_given or wanted_kinds is not None:
        # filter_campaign_ids already excludes superseded + self; the full candidate set is
        # already in memory, so rank all of it (no per-chunk over-fetch cap) rather than
        # truncating before the rollup below — a chunk-heavy campaign truncating the field
        # before rollup is exactly the starvation bug review found (one deck filling every
        # slot, starving every other campaign out of the result entirely).
        candidate_ids = store.filter_campaign_ids(
            conn, record_type=record_type, status=status, tags=tags,
            match_all_tags=match_all_tags, region=region, market=market, markets=markets,
            collection=collection, exclude_campaign_id=campaign_id,
        )
        if not candidate_ids:
            return []
        chunk_map = store.get_chunk_ids_for_campaigns(conn, candidate_ids)
        candidate_chunk_ids = [chid for chids in chunk_map.values() for chid in chids]
        if wanted_kinds is not None:
            candidate_chunk_ids = [chid for chid in candidate_chunk_ids
                                   if _chunk_is_wanted(conn, chid, wanted_kinds)]
        vecs = vectorstore.get_many(conn, candidate_chunk_ids)
        hits = embedding.rank(qvec, list(vecs.items()), top_k=len(vecs))
    else:
        # Only self-exclusion goes into the ANN `exclude` set — folding every superseded
        # campaign's chunks in here inflates sqlite-vec's `k` parameter and can crash past
        # its limit as the corpus grows (review found this). Superseded campaigns are
        # filtered out below, after fetching, alongside the rollup instead.
        exclude_chunks = set(store.get_chunk_ids_for_campaign(conn, campaign_id)) if campaign_id else set()
        hits = vectorstore.search(conn, qvec, top_k=top_k * _SEARCH_OVERFETCH, exclude=exclude_chunks)

    chunk_to_campaign = store.map_chunks_to_campaigns(conn, [chunk_id for chunk_id, _ in hits])

    # Commentary is indexed alongside the body but is a different kind of evidence — the
    # internal reaction to the work rather than the work itself (defect 08). Callers who are
    # asking "what does the brief say", not "what did someone think of it", can leave it
    # out; the default keeps it, because a client's tracked comment is exactly the feedback
    # this library exists to remember.
    best: dict[str, tuple[float, str]] = {}
    for chunk_id, sim in hits:
        cid = chunk_to_campaign.get(chunk_id)
        if cid is None or cid in superseded_ids or (cid in best and sim <= best[cid][0]):
            continue
        best[cid] = (sim, chunk_id)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k]

    evidence = []
    for cid, (sim, chunk_id) in ranked:
        c = store.get_campaign(conn, cid)
        if not c:
            continue
        matched = store.get_chunk(conn, chunk_id)
        detail = c["detail"] or ""
        truncated = not full_detail and len(detail) > config.EVIDENCE_DETAIL_SUMMARY_CHARS
        if truncated:
            detail = detail[:config.EVIDENCE_DETAIL_SUMMARY_CHARS] + "... [truncated]"
        # Most decision-relevant first: real outcomes before forecasts, most recent within
        # each — trimming below then keeps the rows worth keeping. (Reviewed: trimming used
        # to keep insertion order, i.e. the OLDEST rows — a campaign predicted early then
        # measured later, the normal lifecycle, had its real actuals silently dropped from a
        # browsing scan while stale predictions survived.)
        sorted_metrics = sorted(c["metrics"], key=lambda m: (m["metric_type"] != "actual", -m["created_at"]))
        all_metrics = [{"metric_type": m["metric_type"], "detail": m["detail"],
                        "structured": m["structured"]} for m in sorted_metrics]
        # §6.8 (extended): a match with many metrics rows (e.g. bulk-imported) was returning
        # all of them unconditionally even in a light similarity scan — heavy at top_k=5.
        # Same knob as detail: full_detail=True (prepare_evaluation's default) keeps every
        # row for real judgment; find_similar_campaigns' browsing default trims.
        metrics_truncated = not full_detail and len(all_metrics) > config.EVIDENCE_METRICS_MAX
        metrics = all_metrics[:config.EVIDENCE_METRICS_MAX] if metrics_truncated else all_metrics
        evidence.append({
            "campaign_id": cid,
            "title": c["title"],
            "record_type": c["record_type"],
            "status": c["status"],
            "tags": c["tags"],
            "region": c["region"],
            "market": c["market"],
            "markets": c["markets"],
            "collection": c["collection"],
            "similarity": round(sim, 4),
            "detail": detail,
            "detail_truncated": truncated,
            "matched_excerpt": matched["text"] if matched else "",
            # Which layer matched, and its attribution — a judgment that cites a reviewer's
            # objection as though the brief itself claimed it is citing the wrong thing.
            "matched_kind": (matched or {}).get("kind") or "body",
            **_matched_source(matched),
            "metrics": metrics,
            "metrics_total": len(all_metrics),
            "metrics_truncated": metrics_truncated,
        })
    return evidence


def _incompleteness_warnings(conn) -> list[dict]:
    """"Some of the library is not searchable yet", as a warning, for any surface that needs
    to say so. Shared by find_similar_with_context and prepare_evaluation rather than written
    twice, since the two disagreeing about the same library is its own defect."""
    left = store.count_unembedded(conn)
    outstanding = left["chunks"] + left["assets"]
    if not outstanding:
        return []
    # One query, not two: the same list was being fetched twice, once for its length and
    # once for its head.
    by_campaign = store.outstanding_by_campaign(conn)
    records = len(by_campaign)
    return [notices.notice(
        "results_may_be_incomplete",
        affects=f"{records} record(s) are only partly searchable ({outstanding} items "
                f"still to index), so these results may be incomplete.",
        # D45: a single tool sits behind this advice, so it is an action rather than prose.
        next_actions=actions.to_finish_indexing(
            by_campaign[0]["campaign_id"] if by_campaign else None),
        detail=f"{outstanding} unembedded items across {records} campaigns")]


def update_campaign(conn, campaign_id: str, **fields) -> dict:
    """`store.update_campaign`, plus §6.3's moment when a supersession is declared LATE.

    The first version fired the prediction loop only at upload — and the D41 offer is
    accepted by calling THIS, so the flow the item constructs (diff two judged versions,
    offer to link, accept) landed on the one path where the loop never fired. That user is
    the better case, not the worse one: they have both judgments on screen because they just
    compared them.
    """
    if not store.update_campaign(conn, campaign_id, **fields):
        return {"error": f"campaign {campaign_id} not found"}
    record = store.get_campaign(conn, campaign_id)
    earlier_judgment = _judgment_to_check(conn, fields.get("supersedes") or None)
    if not earlier_judgment:
        return record
    return {**record, "earlier_judgment": earlier_judgment,
            "next_actions": actions.after_upload(
                campaign_id=campaign_id, status=record["status"],
                has_metrics=bool(record["metrics"]),
                earlier_judgment=earlier_judgment)}


def _judgment_to_check(conn, superseded: Optional[str]) -> Optional[dict]:
    """The prior judgment this new record is about to settle, or None (§6.3).

    The review's observation is that `reconcile_evaluation` works, and has never once run,
    because it needs somebody to decide to go back and nobody does. So the fix is not a tool
    but a MOMENT: the person uploading v2 is looking at the thing that proves or refutes what
    was said about v1, and asking then costs one prompt.

    Returned only when there is something to check. A record nobody judged has no claim to
    test, and a judgment already reconciled has been tested — asking again would be the
    always-present prompt that stops being read.

    Note what is NOT required: `predictions`. Most judgments have none, and the case the
    review actually described was a RECOMMENDATION that came true ("the same structure
    returns rearranged unless the premise is settled"), not a CTR range. Gating on the
    forecast field would have dropped the example the item exists for.
    """
    if not superseded:
        return None
    judgment = store.latest_evaluation_for_campaign(conn, superseded)
    if not judgment:
        return None
    if store.unreconciled_evaluation_id(conn, superseded) != judgment["id"]:
        return None
    record = store.get_campaign(conn, superseded)
    # Capped, and the cap is the point. The user is filing a document, not sitting an exam:
    # unfiltered, a twelve-finding judgment turned an upload into a quiz. The findings get
    # settled ONE BY ONE in the evaluation of this new version, where 5.4 already records
    # each as resolved or raised again — structurally, by id, rather than as free text. What
    # only THIS moment can catch is the forecast-shaped claim, which is why the verdict and
    # the predictions are not capped.
    all_findings = judgment.get("findings") or []
    open_findings = [{k: f.get(k) for k in ("id", "severity", "kind", "departure", "finding")}
                     for f in all_findings[:_MAX_EARLIER_FINDINGS]]
    return {
        "campaign_id": superseded,
        "title": record["title"] if record else None,
        "evaluation_id": judgment["id"],
        "verdict": judgment.get("verdict"),
        "summary": judgment.get("summary"),
        "predictions": judgment.get("predictions"),
        "open_findings": open_findings,
        "open_findings_total": len(all_findings),
        "ask": ("This replaces a record the library has already judged. Ask which of the "
                "predictions and the verdict held — that is the only way the library learns "
                "whether its own judgment is worth anything, and this is the moment somebody "
                "can actually answer. Record their answer with save_reconciliation, in their "
                "words, as `comparison`. Do NOT call reconcile_evaluation: it looks for "
                "measured results on the record being replaced, which is a brief that never "
                "ran. The findings are listed so the user can see what was said; the place "
                "they get settled one by one is the evaluation of THIS version, which "
                "records each as resolved or raised again."),
    }


def _earlier_version_findings(conn, campaign_id: Optional[str]) -> Optional[dict]:
    """The judgment of the version this one replaces, with its finding ids.

    Without this, `diff_campaigns`'s `adopted` bucket had no feeder at all: it is populated
    only from `resolved[].finding_id`, and nothing ever put those ids in front of the model
    at the moment it was judging the new version. `find_similar` excludes superseded records
    by design, so the earlier judgment was invisible in the evidence too — every fixed
    finding landed in `no_longer_raised` and `adopted` was empty in production. The test
    suite missed it because the fixture wrote the id in by hand.
    """
    if not campaign_id:
        return None
    record = store.get_campaign(conn, campaign_id)
    if not record:
        return None
    previous = record.get("supersedes")
    if not previous:
        return None
    judgment = store.latest_evaluation_for_campaign(conn, previous)
    if not judgment or not judgment.get("findings"):
        return None
    earlier = store.get_campaign(conn, previous)
    return {
        "campaign_id": previous,
        "title": earlier["title"] if earlier else None,
        "evaluation_id": judgment["id"],
        "verdict": judgment.get("verdict"),
        "findings": [{k: f.get(k) for k in ("id", "severity", "kind", "departure",
                                            "category",
                                            "finding", "fix")}
                     for f in judgment["findings"]],
    }


def prepare_evaluation(conn, *, subject_title: str, proposal_text: str, top_k: int = 5,
                       record_type: Optional[str] = None, status: Optional[str] = None,
                       tags: Optional[Union[str, dict, list]] = None, match_all_tags: bool = False,
                       region: Optional[str] = None, market: Optional[str] = None,
                       markets: Optional[Union[str, list]] = None, collection: Optional[str] = None,
                       full_detail: bool = True,
                       campaign_id: Optional[str] = None) -> dict:
    """
    Package the evidence Claude needs to judge a new proposal: the most similar prior
    campaigns WITH their outcomes. full_detail defaults to True here (unlike find_similar) —
    an actual judgment over a short evidence list shouldn't be working from trimmed briefs.
    Claude reads this, works out its findings citing specific
    priors, then calls save_evaluation. This tool does NOT itself judge. Optionally narrow
    to structured criteria first (§6.2), e.g. region="APAC" to only weigh APAC precedent.
    Pass a {"value": ..., "source": "verified"} tag to weigh only precedent whose matching
    performance claim is backed by real metric data, not a stated impression.
    """
    # find_similar, not find_similar_with_context, so the "your library is only partly
    # indexed" warning is added explicitly below rather than inherited — see the note there.
    evidence = find_similar(conn, text=proposal_text, top_k=top_k, record_type=record_type,
                            status=status, tags=tags, match_all_tags=match_all_tags,
                            region=region, market=market, markets=markets,
                            collection=collection, full_detail=full_detail)
    concluded = [e for e in evidence if e["metrics"]]
    return {
        "subject_title": subject_title,
        "evidence_count": len(evidence),
        "evidence": evidence,
        "note": (
            # The vocabulary here has to be the vocabulary save_evaluation accepts. This
            # said "proceed/revise/reject" while the enum takes "approve" — so the prompt
            # that shapes the judgment taught a word the next tool rejects.
            ("When `earlier_version` is present this proposal revises a judged record. For "
             "each of its findings: if this version addresses it, put it in `resolved` with "
             "its `finding_id`; if the problem is still there, raise it as your own finding "
             "carrying `repeats` set to that id. That is what lets diff_campaigns state "
             "which corrections were taken instead of guessing from how alike two sentences "
             "read. "
             if _earlier_version_findings(conn, campaign_id) else "") +
            "`outcomes` splits this evidence into what WORKED and what did NOT, by measured "
            "result rather than by impression. Read both before deciding: resemblance to a "
            "strong performer is not evidence, and a brief that looks like something that "
            "failed may be the thing that worked. Say which side you weighed. "
            "Reason over this evidence, then call save_evaluation with a verdict "
            "(approve / revise / reject), a one-line summary, and one short finding per "
            "problem — each with its severity, its kind, and a quote from the campaign or "
            "rule it is anchored to, CITING specific campaign_ids above. `kind` is required; "
            "a precedent_departure must also say which way it departs (`departure`: "
            "regression / unexplained / possible_improvement), because \u201cthis is not how Peru "
            "did it\u201d is not a finding until you say whether different is worse here. "
            "Predicted CTR/ROI "
            "ranges go in `predictions`. Weight concluded campaigns (those with metrics) "
            "most. "
            # Said here as well as in save_evaluation's description, because this is the
            # message in front of the model while it is deciding what to write down. A rule
            # it only meets as a rejection afterwards costs a retry every time.
            "Quotes are checked against the record you cite and the write is refused if the "
            "words are not there, so copy them from the evidence above rather than writing "
            "them from memory — … for anything you leave out. A point you cannot quote is "
            "an observation, and saying it as one is better than a citation that fails."
        ),
        "campaigns_with_outcomes": [e["campaign_id"] for e in concluded],
        # §6.4's other half, and the half that can actually change a verdict. The save-time
        # search RECORDS overconfidence; by then the judgment is written. What changes the
        # reasoning is seeing both sides while reasoning — so the evidence that worked and
        # the evidence that did not are separated here, instead of arriving as one ranked
        # list in which the strongest match sets the tone. `verified` only: a performance tag
        # somebody typed is an impression, and an impression cannot be the counterweight.
        "outcomes": _both_poles(evidence),
        # §5.3: the single thing that would most change THIS verdict, or None when nothing
        # is. About the evidence cited, not about the library — a library that is 90%
        # measured can still produce a judgment resting entirely on the unmeasured tenth.
        # Filtered to MEASURED outcomes: `find_similar` includes predicted rows in
        # `metrics`, so the unfiltered version reported a judgment resting entirely on a
        # forecast as complete. The save_evaluation path was fixed and this one was not.
        # §5.4: the findings this version is a revision OF, so each can be resolved by id
        # or repeated by id rather than left to be matched by wording later.
        "earlier_version": _earlier_version_findings(conn, campaign_id),
        "most_valuable_missing_input": most_valuable_missing_input([
            {**e, "metrics": [m for m in (e.get("metrics") or [])
                              if m.get("metric_type") == "actual"]}
            for e in evidence]),
        # The one surface where a half-indexed library matters most was the one that said
        # nothing about it: find_similar_campaigns warns, and this — a verdict about to be
        # saved against this evidence — did not. "How many precedents did this rest on" is
        # the wrong number when records are missing from the search entirely (§6.6).
        "warnings": _incompleteness_warnings(conn),
    }


def reconcile_evaluation(conn, *, evaluation_id: str, actual: Optional[str] = None) -> dict:
    """
    Start closing the loop on a past judgment (§6.5: this is what makes reconciliation
    actually functional). If `actual` isn't given, pull it automatically from the
    campaign's own metric_type='actual' metrics on file — the user shouldn't have to retype
    numbers that were already recorded via add_metrics/bulk_import_metrics. Errors if
    neither is available (only predicted metrics on file don't count as "actual").
    """
    import json
    ev = store.get_evaluation(conn, evaluation_id)
    if not ev:
        return {"error": f"evaluation {evaluation_id} not found"}

    if actual is None:
        campaign = store.get_campaign(conn, ev["campaign_id"]) if ev["campaign_id"] else None
        actual_metrics = [m for m in (campaign["metrics"] if campaign else [])
                          if m["metric_type"] == "actual"]
        if not actual_metrics:
            return {"error": "no actual metrics on file for this campaign — pass actual= "
                              "or record them first with add_metrics/bulk_import_metrics"}
        # A metrics row can carry `structured` with no freeform `detail` at all (exactly
        # what bulk_import_metrics produces from a KPI workbook) — include both, not just
        # detail, or a structured-only row silently contributed nothing (review found this).
        parts = []
        for m in actual_metrics:
            if m["detail"]:
                parts.append(m["detail"])
            if m["structured"]:
                parts.append(m["structured"])  # already a JSON string from the DB row
        actual = "\n".join(parts)
        if not actual:
            return {"error": "actual metrics on file for this campaign have no readable "
                              "detail or structured data"}

    # A judgment written before §2.4 has only the essay: no verdict, no findings. Returning
    # it as an empty structured evaluation meant reconciling against an original the code
    # had silently dropped - a confident comparison with nothing on one side of it.
    legacy = bool(ev.get("analysis")) and not ev.get("verdict")

    return {
        "evaluation_id": ev["id"],
        "subject_title": ev["subject_title"],
        "original_verdict": ev["verdict"],
        "original_summary": ev["summary"],
        "original_findings": ev["findings"],
        **({"original_analysis": ev["analysis"], "schema": "legacy"} if legacy else {}),
        # Already decoded by store._parse_evaluation - decoding again would be parsing a
        # dict as JSON.
        "predictions": ev["predictions"],
        "cited_ids": ev["cited_ids"],
        "actual": actual,
        "note": ("This judgment predates the structured schema, so there is no verdict or "
                 "findings to compare against — only `original_analysis`, the free text as "
                 "it was written. Read it before comparing."
                 if legacy else
                 "Compare the original findings and predictions to actual, then call "
                 "save_reconciliation with the lesson."),
    }


# ── image assets / creative-reuse detection (§6.6) ───────────────────────────

def ingest_image_asset(conn, *, campaign_id: str, asset_ref: dict) -> dict:
    """Attach an image to a campaign and process it two ways: a perceptual hash (exact/
    near-duplicate reuse detection, §6.6) and a CLIP visual embedding (aesthetic/regional
    similarity, the heavier follow-on). Storage always succeeds even if one or both
    processing steps fail (e.g. a corrupt image, or CLIP unavailable) — failures are
    reported per-step, not swallowed, and don't block each other."""
    if store.get_campaign(conn, campaign_id) is None:
        return {"error": f"campaign {campaign_id} not found"}

    path, warnings = _resolve_asset(asset_ref)
    if not path:
        return {"error": "; ".join(warnings) or "could not resolve asset_ref"}

    stored_name = _keep_asset(path)
    result = _store_and_fingerprint_image(conn, campaign_id, stored_name)
    result["warnings"] = warnings + result["warnings"]
    return result


def _store_and_fingerprint_image(conn, campaign_id: str, stored_name: str) -> dict:
    """The part of ingest_image_asset that runs once the image file is already saved under
    ASSET_DIR. NOT used by the deck-embedded-image path in ingest_campaign — that path also
    computes reuse_flags inline (via the shared _phash_matches below) as part of the same
    loop, which this helper doesn't do, so it re-implements the fingerprint/CLIP-embed steps
    rather than call this and bolt reuse-checking on after."""
    full_path = config.ASSET_DIR / stored_name
    warnings: list[dict] = []
    aid = store.insert_asset(conn, campaign_id, file_path=stored_name)

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
        vectorstore.add(conn, aid, vec, space="asset")
        store.mark_asset_embedded(conn, aid)
        visually_embedded = True
    except Exception as exc:
        # Which warning this is depends on WHY: the model being absent from the machine is
        # the operator's problem to fix once, while one image failing on a working model is
        # this record's problem. Telling a marketer to go and find an administrator over a
        # single corrupt PNG would be the same mistake in the other direction.
        warnings.append(_vision_notice(
            f"asset stored but not visually embedded (aesthetic similarity will miss it): "
            f"{exc}. Run finish_indexing to complete it — the image is saved, so it does "
            f"not need uploading again."))

    return {"asset_id": aid, "campaign_id": campaign_id, "fingerprinted": fingerprinted,
            "visually_embedded": visually_embedded,
            "warnings": notices.collapse(warnings)}


def check_image_provenance(conn, *, asset_ref: dict, campaign_id: Optional[str] = None,
                           threshold: Optional[int] = None) -> dict:
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
                             current=current, threshold=threshold)
    return {"query_hash": query_hash, "matches": matches, "warnings": warnings}


def _phash_matches(conn, query_hash: str, *, exclude_campaign_id: Optional[str],
                   current: Optional[dict], threshold: Optional[int] = None) -> list[dict]:
    """Shared by check_image_provenance and the automatic deck-embedded-image path in
    ingest_campaign — same pHash-match + region-flag logic either way."""
    threshold = config.PHASH_MATCH_THRESHOLD if threshold is None else threshold
    superseded_ids = store.get_superseded_campaign_ids(conn)

    matches = []
    for cand in store.get_all_fingerprints(conn, exclude_campaign_id=exclude_campaign_id):
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
            "flag": _region_mismatch_flag(current, c),
        })
    matches.sort(key=lambda m: m["hamming_distance"])
    return matches


def find_similar_images(conn, *, asset_ref: dict, campaign_id: Optional[str] = None,
                        top_k: int = 5, region: Optional[str] = None) -> dict:
    """
    Aesthetic/regional visual similarity (CLIP) — catches "same product, different photo,"
    "looks like the APAC shoot," NOT exact/near-duplicate reuse (that's
    check_image_provenance's pHash job). Works on an image that isn't stored yet. Pass
    region to filter to that region FIRST (§6.2's pattern, applied to images); pass
    campaign_id (the campaign this image is headed for) to exclude its own assets and get a
    region-mismatch flag on matches from elsewhere.
    """
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
        notice = _vision_notice(f"visual search could not process the image: {exc}")
        return {"error": notice["affects"], "warnings": [notice], "matches": []}

    if region:
        # filter_campaign_ids already excludes superseded campaigns.
        candidate_ids = store.filter_campaign_ids(conn, region=region, exclude_campaign_id=campaign_id)
        assets = store.list_assets(conn, campaign_ids=candidate_ids)
    else:
        superseded_ids = store.get_superseded_campaign_ids(conn)
        assets = [a for a in store.list_assets(conn, exclude_campaign_id=campaign_id)
                 if a["campaign_id"] not in superseded_ids]

    asset_to_campaign = {a["id"]: a["campaign_id"] for a in assets}
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

def _resolve_asset(ref: dict) -> tuple[Optional[Path], list[str]]:
    """Resolve an asset reference to a local file. Accepts {path} | {asset_id} | {filename,base64}."""
    if not isinstance(ref, dict):
        return None, ["asset_ref must be an object"]
    if ref.get("path"):
        p = Path(ref["path"])
        return (p, []) if p.is_file() else (None, [f"local path not found: {ref['path']}"])
    if ref.get("asset_id"):
        d = config.UPLOAD_DIR / ref["asset_id"]
        files = list(d.iterdir()) if d.is_dir() else []
        return (files[0], []) if files else (None, [f"unknown asset_id {ref['asset_id']}"])
    if ref.get("base64") is not None:
        try:
            data = base64.b64decode(ref["base64"], validate=True)
        except (binascii.Error, ValueError) as exc:
            return None, [f"invalid base64: {exc}"]
        if len(data) > config.MAX_INLINE_BYTES:
            return None, ["inline asset too large; use POST /upload"]
        name = Path(ref.get("filename") or "asset.bin").name
        dest = config.UPLOAD_DIR / f"inline_{name}"
        config.ensure_dirs()
        dest.write_bytes(data)
        return dest, []
    return None, ["asset_ref needs path, asset_id, or base64"]


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
