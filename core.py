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
import json
import re
import sqlite3
import sys
import unicodedata
import time
from pathlib import Path
from typing import Optional, Union

import actions
import chunking
import clip_embed
import commitments
import config
import context
import corrections
import drift
import embedding
import enums
import extract
import facts
import identity
import feedback
import images
import learning
import metrics
import notices
import rulebook
import store
import vectorstore
import version

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
                    asset_ref: Optional[dict] = None, confirm: bool = True,
                    starts_on: Optional[str] = None, ends_on: Optional[str] = None,
                    asset_link: Optional[str] = None,
                    campaign_type: Optional[str] = None,
                    partner: Optional[str] = None) -> dict:
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

    promised: dict = {}
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
        # §9.6: when it ran, at the moment the record is created. The only route in was a
        # second deliberate `update_campaign` call, which is the unreachable human step this
        # project has now hit three times.
        starts_on=starts_on, ends_on=ends_on,
        # §12.4: where the work lives, what kind of campaign it is, and who ran it — the three
        # the review named as having no field. Validated on the way in (`checked_link`) rather
        # than on the way out, because a link nobody can open is worth catching while the
        # person who typed it is still here.
        asset_link=asset_link, campaign_type=campaign_type, partner=partner,
    )

    # §11.5, at the door most people arrive by. A tag may carry `said_by`, and `upload_campaign`
    # accepts one — but only `update_campaign` walked them, so an opinion recorded at UPLOAD
    # sat on the campaign and was absent from the table that is supposed to be the authority
    # on opinions. Reading the stored list rather than the argument, for the same reason as
    # there: what was stored is the normalised form, and the raw one disagrees with it.
    _keep_the_view(conn, cid, store.get_campaign(conn, cid).get("tags"))

    # Images embedded IN the deck, extracted and processed automatically — a separate
    # manual upload_image_asset call per image isn't a workflow anyone would actually use
    # (product feedback). Only possible when a real file reached us (asset_ref); the
    # LLM-first deck_text-only path has no file to extract images from.
    found = _index_deck_images(conn, cid, asset_path_for_images, deadline=deadline)
    image_assets = found["image_assets"]
    images_checked = found["images_checked"]
    images_embedded = found["images_embedded"]
    warnings += found["warnings"]

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
            _add_vector(conn, chunk_id, vec)
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
    # §9.3: the promises the brief makes, taken from the list the deck almost always already
    # has. Extracted at upload because that is when the text is in hand — and recorded with the
    # line each came from, so the reading can be disagreed with.
    #
    # Campaigns only. Brand guidelines are full of bulleted lists and none of them is a promise
    # this campaign made — §9.2 learned the same thing about which RECORD a check runs on, one
    # item ago.
    if record_type in learning.CHECKABLE_RECORDS:
        commitments.extract(conn, campaign_id=cid, text="\n".join(
            filter(None, [detail, deck_text])))
        promised = commitments.summary_for(conn, cid)
    current = store.get_campaign(conn, cid)
    earlier_judgment = _judgment_to_check(conn, supersedes)
    # §10.6/D54: computed before the offers so the result can be consulted after them.
    gap_moved = _gap_moved(conn)
    offers = actions.after_upload(
        campaign_id=cid, status=current["status"],
        # A record being ingested has no metrics yet, so this is always False here — it
        # reads `has_actual_metrics` anyway so the two call sites cannot answer the same
        # question two ways, which is how they drift.
        has_metrics=current["has_actual_metrics"],
        earlier_judgment=earlier_judgment,
        # §8.6: a tracked client comment IS client feedback, and it arrives with its
        # provenance already assembled. Without this the correction loop had no input at
        # all — `note_correction` was a tool nothing in the product ever mentioned.
        commentary=commentary, title=title,
        # §9.9/D40: a judgment about this very title that is attached to nothing. This is
        # §5.2's moment — the judgment is on screen because they just asked for it.
        unlinked_judgment=store.unlinked_judgment_for(conn, title),
        # §10.6: said where it is cheap, rather than waited for.
        waiting=feedback.waiting(conn, apart_from=cid),
        # §9.6: the window is what makes this record checkable against a calendar at all,
        # and this is the moment somebody is present and thinking about the campaign.
        has_window=bool(current.get("starts_on")),
        gap_moved=bool(gap_moved))
    # Only once the offer has survived `trim`. Marking it shown before that dropped the
    # offer and recorded the change as announced, so it was never made at all.
    _the_gap_was_announced(conn, offers, gap_moved)
    return {
        "campaign_id": cid, "title": title, "record_type": record_type,
        "embedded": embedded_count == len(chunk_texts),
        "chunks_total": len(chunk_texts), "chunks_embedded": embedded_count,
        "image_assets": image_assets, "images_checked": images_checked,
        "images_total": len(image_assets), "images_embedded": images_embedded,
        # §9.3/D116 (the sixth time): a mechanically-extracted list first reached a human as
        # verdicts in a post-mortem, after the junk entries had already said something about a
        # supplier. The correction belongs before the accusation.
        **({"commitments": promised} if promised else {}),
        "commentary_found": len(commentary),
        "commentary_checked": commentary_checked,
        "normalised": changed,
        "next_actions": offers,
        "warnings": _persisted(conn, cid, notices.collapse(warnings)),
        # §6.3: the one moment where "was our judgment any good?" is both answerable and
        # free. Attached only when there IS an unreconciled judgment on the record this one
        # replaces — see `_judgment_to_check`.
        **({"earlier_judgment": earlier_judgment} if earlier_judgment else {}),
    }


def _index_deck_images(conn, cid: str, asset_path_for_images, *, deadline: float) -> dict:
    """Every image inside a deck: stored, fingerprinted, reuse-checked, visually embedded.

    Lifted out of `ingest_campaign` so that `attach_deck` runs the SAME one (§12.4/D39). It
    did not, and a record repaired the way `commentary_never_read` recommends had nothing on
    the briefed side of `compare_execution` — the comparison the whole of Phase 9 is about,
    reading empty on exactly the records the gap had just told somebody to fix. Copying the
    ninety lines into the new path would have been this codebase's own most-repeated defect,
    two implementations of one rule, on the hot path.
    """
    warnings: list[dict] = []
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
                _add_vector(conn, entry["asset_id"], vec, space="asset")
                store.mark_asset_embedded(conn, entry["asset_id"])
                entry["visually_embedded"] = True
                images_embedded += 1
            except Exception as exc:
                warnings.append(_vision_notice(
                    f"deck image {entry['asset_id']} not visually embedded: {exc}"))

        for entry in image_assets:
            entry.pop("_path", None)

    return {"image_assets": image_assets, "images_checked": images_checked,
            "images_embedded": images_embedded, "warnings": warnings}


def add_metrics(conn, campaign_id: str, *, detail: Optional[str] = None,
                structured: Optional[dict] = None, metric_type: str = "actual",
                confirm: bool = True, snapshot_drift: bool = True) -> dict:
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
    # §9.5: the drift, ON the outcome. `snapshot_drift=False` is for a bulk import, which
    # writes many rows against the same campaign and re-ran the whole comparison for each —
    # fifty workbook rows for one campaign meant fifty identical comparisons and forty-nine
    # discarded snapshots. It takes it once, after the loop.
    if metric_type == "actual" and snapshot_drift:
        _snapshot_execution_drift(conn, campaign_id)
    # §8.1/§8.2: the structured values also go into the registry's TYPED storage, canonicalised
    # — the JSON blob above stays as the record of what was sent, and this is what makes "show
    # me every ROAS on file" answerable. An unfamiliar key asks once rather than being rejected
    # (which loses the number) or silently accepted (which is how 25 keys happened).
    asked, eligible, retired, skipped = [], [], [], []
    for key, value in (structured or {}).items():
        # A workbook's Month, Store # and Campaign columns are numeric and are not KPIs.
        # Recording them made each one a provisional measure that accrued sightings and could
        # graduate — the drift this phase exists to stop, arriving through the import.
        if not metrics.is_a_measurement(key):
            skipped.append({"key": key, "value": value,
                            "reason": f"{key!r} looks like an identifier or a dimension "
                                      f"rather than something measured, so it is not "
                                      f"recorded as a measure."})
            continue
        try:
            written = metrics.record(conn, campaign_id=campaign_id, key=key, value=value,
                                     metric_type=metric_type)
        except ValueError as exc:
            # A value that is not a number is still in the JSON blob and in `detail`. Failing
            # the whole write over one unparseable figure would lose the other nine.
            #
            # REPORTED, though. Silently continuing meant a column the §8.8 preview said would
            # not be imported was skipped without a word, and a contradiction between a
            # column's own metric_type and the row's was swallowed as though it were an
            # unreadable cell. A skip nobody is told about is the silent acceptance this whole
            # phase is written against, wearing the other face.
            skipped.append({"key": key, "value": value, "reason": str(exc)})
            continue
        if written.get("new_measure"):
            asked.append(written["new_measure"])
        # §8.3: the write that makes a measure eligible is the moment somebody is present to
        # be asked, and the gate's third condition is a person. Without this the only route to
        # the gate was a tool the user would have to know exists and name the measure by its
        # canonical stem to call — so nothing would ever graduate.
        eligible += [written["newly_eligible"]] if written.get("newly_eligible") else []
        # §8.5/§2.1: a checklist that shrank silently is partial state nobody was told about.
        retired += written.get("retired") or []
    # D119: a date column is not a measurement and is not rubbish either. §9.6 gave campaigns
    # a window, which is the field it belongs in — and setting one changes which context
    # events reach this campaign, so it is REPORTED rather than done quietly.
    window_set = _window_from_the_workbook(conn, campaign_id, structured)
    # D116: `update_campaign` offered the check a window makes possible and this — the other
    # write that sets one — did not. The house rule is the write that makes a tool's output
    # non-empty, and a workbook's date column is exactly that write.
    context_offer = (_context_offer(conn, campaign_id, {"starts_on": True})
                     if window_set else {})
    if context.window_from_columns(structured or {}):
        # A column this READ is not a column it skipped, whether or not it moved the window —
        # a workbook's second January row adds nothing and was still understood. Saying it was
        # skipped is a false statement about the import in the one field a reader checks to
        # find out what the library did with their spreadsheet.
        skipped = [s for s in skipped if s["key"] not in _date_columns(structured)]
    revisit = drift.to_revisit(conn, campaign_id) if metric_type == "actual" else []
    return {"metrics_id": mid, "campaign_id": campaign_id, "status": "stored",
            **({"window_set": window_set} if window_set else {}),
            **({"context_events_found": context_offer["context_events_found"]}
               if context_offer else {}),
            **({"new_measures": asked} if asked else {}),
            **({"skipped": skipped} if skipped else {}),
            **({"newly_eligible": eligible} if eligible else {}),
            **({"retired_measures": retired} if retired else {}),
            # §9.4: "usually needs the outcome to settle it" — and the classification offer
            # fires when the photographs land, which is normally BEFORE the numbers. Without
            # asking again here, `too_early` is a one-way sink that absorbs the answer the
            # feature exists to collect.
            **({"drift_to_revisit": revisit} if revisit else {}),
            # §10.6: recording a target opens a row in the feedback queue, and a write that
            # opens a row should name it — the queue is only worth building if the writes that
            # fill it also say so.
            **({"waiting_on_feedback": _waiting_elsewhere(conn, campaign_id)}
               if _waiting_elsewhere(conn, campaign_id) else {}),
            # The moment the precondition for reconciling is satisfied. Offered at
            # save_evaluation time it simply failed: there were no actuals yet (§5.2 review).
            # D116: the workbook's date column is a write that makes `campaign_context`
            # answerable, so its offer joins the list rather than being overwritten by it.
            "next_actions": actions.trim(
                actions.offer_the_queue(_waiting_elsewhere(conn, campaign_id))
                + actions.after_metrics(
                    campaign_id=campaign_id,
                    open_evaluation_id=store.unreconciled_evaluation_id(conn, campaign_id)
                    if metric_type == "actual" else None)
                + (context_offer.get("next_actions") or [])
                + (_attribution_offer(conn, campaign_id)
                   if metric_type == "actual" else []))}


def _say_the_settled(earlier: Optional[dict]) -> str:
    """Tell the model which of the earlier findings a PERSON has already answered (§10.2/D84).

    This is the predicate that moved when the `departure` rewrite came out. `unexplained`
    means "the brief does not say whether this is deliberate" — an open question — and once
    somebody has said, it is no longer open. Rewriting the stored value said that by
    destroying it; saying it here says the same thing to the reader who acts on it, and
    leaves the model's own word on the record where an audit can still find it.

    Works for all three answers, which the rewrite did not: `fixed` and `not_applicable` are
    equally settled and equally worth not re-raising.
    """
    answered = [f for f in (earlier or {}).get("findings") or [] if f.get("settled")]
    if not answered:
        return ""
    said = "; ".join(
        f"“{f['finding'][:60]}” — {f['settled']['said_by']}: "
        f"“{f['settled']['note'][:90]}”" for f in answered[:3])
    return (f"{len(answered)} of those findings has already been answered by somebody: "
            f"{said}. Do not raise those again as though nobody had said anything. If the "
            f"answer settles it, leave it alone; if you think the answer is wrong, say so "
            f"explicitly and quote what they said — disagreeing with a person on the record "
            f"is legitimate, and silently re-asking is what this was built to stop. ")


def _offer_to_settle(evaluation_id: str, findings: list) -> list[dict]:
    """Where the answer to "is this deliberate?" goes (§10.2/D84).

    Offered only for an `unexplained` departure, which is the one finding kind that is
    literally a question — its own validation says so: "a departure you cannot yet say is
    worse cannot be blocking; 'the brief does not say whether this is deliberate' IS a
    question". Every other finding is a statement, and offering to settle one invites
    somebody to wave away a guardrail breach with a sentence.
    """
    asking = [f for f in findings if f.get("departure") == "unexplained"]
    if not asking:
        return []
    first = asking[0]
    # `answer` is NOT prefilled, and that was the first version's mistake. `actions.action`
    # says it in writing: prefilled arguments are "arguments this library already holds.
    # Never a guess and never a placeholder: the user says yes to the label, so anything
    # filled in here is something they agreed to without being shown it." The answer is the
    # ONE field only a person can supply — the entire premise of the feature — and prefilling
    # `deliberate` meant a user saying "we fixed that in v2" could have "we kept it on
    # purpose" recorded under their own name, destroying the distinction `FINDING_ANSWERS`
    # exists to preserve. Accepting this is not one step, and saying otherwise was the lie.
    return [actions.action(
        "Record the user's answer if they say why that difference was there",
        "answer_finding",
        why=f"{len(asking)} finding(s) here ask whether a difference was on purpose. Without "
            f"an answer on file the next judgment of this brief is asked the same question "
            f"again, and the user has already answered it.",
        consent="ask",
        needs=["answer — `fixed` (we changed it), `deliberate` (we kept it, and here is "
               "why), `not_applicable` (it does not apply here) or `open` (never mind)",
               "note — why, in the user's own words",
               "said_by — whose answer it is"],
        evaluation_id=evaluation_id, finding_id=first["id"])]


def _attribution_offer(conn, campaign_id: str) -> list[dict]:
    """Ask the person the one thing the server cannot work out (§10.6/D116, §9.8).

    The server can say "the port was shut for nine of the thirty days this ran". Only a person
    can say whether that is WHY the number is what it is — §9.8 is explicit that the machine
    lines the two up and never decides between them. `attribute_outcome` is the sentence where
    that answer goes, and nothing in the product had ever pointed at it, so the item's entire
    human half was reachable only by somebody who had read the tool list.

    Offered at the moment the outcome lands, which is the only moment both halves are on
    screen — and only for an event nobody has answered about yet, because re-asking a settled
    question is how an offer list stops being read.

    NOT offered when nothing overlapped. An invitation to attribute an outcome to an event
    that did not run through it is an invitation to invent a cause, which is worse than
    silence: it would be this product manufacturing exactly the confident unfounded claim it
    exists to refuse.
    """
    record = store.get_campaign(conn, campaign_id) or {}
    unanswered = [event
                  for metric in record.get("metrics") or []
                  if metric.get("metric_type") == "actual"
                  for event in metric.get("confounded_by") or []
                  if not event.get("attribution")]
    if not unanswered:
        return []
    first = unanswered[0]
    # The label no longer frames this as a yes/no, and the `why` no longer promises something
    # this call cannot do. Both were wrong in the first version, and wrong in ways the tool
    # itself contradicted four lines into its own docstring: it said "without an answer the
    # outcome stays flagged as confounded, which weakens every judgment that cites it", while
    # `attribute_outcome` says in bold that a confounded outcome STILL COUNTS and nothing
    # down-weights it. Attributing does not clear the flag either — it can only confirm or
    # add one. Two user-facing strings in one product, one asserting what the other denies.
    return [actions.action(
        f"Record what somebody thinks “"
        f"{(first.get('description') or first.get('subject') or '')[:44]}” did to these "
        f"numbers",
        "attribute_outcome",
        why="The library can see it ran through this campaign's window; whether it bears on "
            "the result is a judgment only somebody who was there can make, and nothing is "
            "recorded until they say so. “It is not why” is an answer too — pass "
            "bears_on=false. A confounded outcome still counts; it stops being quotable as "
            "clean evidence.",
        consent="ask",
        needs=["stated_by — whose reading of it this is",
               "note — what they think it did, or why it is not why, in their words",
               "bears_on — false if they say this is not what moved the numbers"],
        campaign_id=campaign_id, event_id=first["id"])]


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
_BASES = ("computed", "judged", "heuristic")
BASIS_MEANING = {
    "computed": "The server worked this out by reading the brief. It is identical for every "
                "user, so a difference in it is a bug rather than a disagreement.",
    "judged": "Somebody reasoned their way to this. Two readers may legitimately differ, and "
              "the evidence behind it is what makes it arguable rather than arbitrary.",
    # D61. §5.4 matched findings across versions by character similarity and had to call the
    # result `judged` with a caveat attached, because it is neither: a similarity score is not
    # a human judgment, and it is not identical for every user in the sense `computed` means,
    # since it is a heuristic whose threshold somebody picked. Reading `heuristic` as a softer
    # `computed` is the mistake this sentence exists to prevent.
    "heuristic": "The server inferred this from a similarity score against a threshold "
                 "somebody chose. It is repeatable but it is not a fact — a different "
                 "threshold would give a different answer, and it should never be stated as "
                 "something the record says.",
}
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


def _verify_rule_quote(rule: dict, segments: list, where: str) -> None:
    """The §6.1 guarantee, applied to a rule instead of a record (§12.1).

    The same reason as everywhere else: a breach finding quoting a rule that does not say that
    is worse than an uncited one, because a citation moves a reader from "this is an
    assertion" to "this is established". And a rule is SHORT, so a paraphrase is not a
    convenience here — it is somebody's rule rewritten into what the model thought it said.
    """
    # `_quote_is_in`, the same matcher the record path uses, rather than a comparison written
    # here. A second implementation of "is this quote really in that text" would differ from
    # the first on exactly the cases §6.1 spent three rounds getting right — folding,
    # elisions, and how far apart two segments may be.
    if _quote_is_in(segments, [rule["rule"]]):
        return
    raise ValueError(
        f"{where}precedent quotes rule {rule['id']!r}, but that rule does not contain the "
        f"quoted words. The rule reads: {rule['rule']!r}. Quote it as written — a breach "
        f"finding citing a rule that does not say that reads as established when it is not.")


def _verify_quote(conn, cited: str, segments: list, layer: str, where: str, *,
                  is_rule: bool, kind=None) -> None:
    """Refuse a citation the cited record does not support (§6.1).

    Refusal rather than a `verified: false` flag, for the reason consistency idea 3 gives
    about every other check here: a validation error costs a retry, while storing an
    unverified quote beside verified ones is permanent drift, and nothing downstream would
    be able to tell them apart once a summary quoted either.
    """
    # §12.1: a `rule_id` means THE RULEBOOK, and now there is one. This resolved every
    # `rule_id` through `store.text_on_file` — a lookup in the CAMPAIGNS table — so the
    # product told the model "a finding that a rule is breached cites its id", handed it six
    # rule ids, and then refused every one of them as "not a record in this library", while
    # accepting any `reference` row as a rule. The single most important thing this item
    # claims was the one thing the code forbade.
    if is_rule:
        rule = rulebook.by_id(cited)
        if rule is not None:
            _verify_rule_quote(rule, segments, where)
            return
        known = [r["id"] for r in rulebook.rules()]
        raise ValueError(
            f"{where}precedent cites {cited!r} as a rule_id, and rulebook "
            f"{rulebook.version()} contains no such rule. "
            + (f"It contains: {', '.join(repr(r) for r in known[:8])}"
               + (f" and {len(known) - 8} more" if len(known) > 8 else "") + ". "
               if known else
               "No rules are written in it at all, so this library cannot support a breach "
               "finding — nobody has written the rule down. ")
            + "A guardrail breach cites a rule somebody wrote, not a campaign that happened "
              "to do it that way.")

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
    # §8.6: a STANDING CORRECTION is the third thing a finding can rest on, and the reason the
    # corrections carry provenance at all — *"so a judgment can cite where the rule came
    # from"*. Without this slot a graduated correction was shown to the model as a rule and
    # could not be cited by it: `guardrail_breach` demands a `rule_id` pointing at a
    # `reference` record, and a correction is neither. That is §5.2's failure — an offer whose
    # write path does not exist — arriving one item after it was named.
    correction_id = value.get("correction_id")
    named = [k for k, v in (("campaign_id", campaign_id), ("rule_id", rule_id),
                            ("correction_id", correction_id)) if v]
    if not named:
        raise ValueError(f"{where}precedent must name what it cites — a campaign_id for a "
                         f"departure from precedent, or a rule_id or correction_id for a "
                         f"guardrail breach")
    # More than one slot at once made the campaign_id decorative: the quote was checked
    # against the rule and the campaign could then be anything, invented included, and still
    # be stored beside a passing check. A finding is anchored to ONE thing; two anchors are
    # two findings.
    if len(named) > 1:
        raise ValueError(f"{where}precedent names {' and '.join(named)}. A finding is "
                         f"anchored to one thing: the rule it breaches, the standing "
                         f"correction it breaches, or the campaign it departs from. If more "
                         f"than one is true, they are more than one finding.")
    quote = _bounded(value.get("quote"), "precedent.quote", _MAX_QUOTE, where=where)
    # §6.1: required, not merely bounded. A citation naming a campaign and quoting nothing is
    # the assertion this whole item was written about with an id stapled to it — and it was
    # the shape the caps alone happily accepted.
    if not quote:
        raise ValueError(f"{where}precedent needs a quote — the words from "
                         f"{campaign_id or rule_id or correction_id!r} that the finding rests "
                         f"on. An id without a quote is an assertion with a reference "
                         f"attached.")
    if correction_id:
        # Checked against the CORRECTION's own text, the same way a rule is checked against
        # the rulebook record: a citation the cited thing does not support is the assertion
        # §6.1 was written about, with an id stapled to it. The provenance travels with it so
        # the reader can follow the rule back to the deck it was learned from.
        return _checked_correction(conn, correction_id, quote, where)
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
        # WHICH rulebook the rule came from, on the finding itself. `source` was carried on
        # every rule from the first version of the loader and read by nothing — a field whose
        # stated purpose ("so a judgment can say whether a breach was of the product's rule or
        # the customer's") no caller could achieve. It matters most once §12.2's overlays
        # exist: two judgments citing `no-ai-imagery` may be citing two different rules, and
        # the provenance stamp is one scalar.
        rule = rulebook.by_id(rule_id)
        if rule:
            cleaned["rule_source"] = rule["source"]
            cleaned["rule_severity"] = rule["severity"]
            cleaned["checked"] = ["rulebook", "wording"]
    if layer == "commentary":
        # Who said it and where, so the finding can be read back as "their reviewer said X"
        # rather than "the deck says X".
        for field, limit in (("author", 120), ("anchor", 40), ("date", 40)):
            kept = _bounded(value.get(field), f"precedent.{field}", limit, where=where)
            if kept:
                cleaned[field] = kept
    return cleaned


# A floor under "quoted a fragment", not the negation check — see `_checked_correction` for
# why a length ratio cannot catch an inversion. Not 100%, because trailing punctuation and a
# closing clause naming an exception are legitimate to leave off.
_MIN_RULE_QUOTE = 0.6


def _checked_correction(conn, correction_id: str, quote: str, where: str) -> dict:
    """Verify a citation of a standing correction (§8.6).

    Three things have to be true, and each one exists because its absence would let a finding
    claim the authority of a rule nobody set. The correction has to EXIST; it has to be
    STANDING, because a provisional one is exactly what the gate withheld and citing it would
    promote it by the back door; and the quote has to be the correction's own words, checked
    the same way §6.1 checks every other citation.

    The provenance comes back with it. That is the whole reason corrections carry it: a reader
    following the finding gets to the deck the rule was learned from, rather than to the
    library asserting it.
    """
    entry = corrections.describe(conn, correction_id)
    if entry is None:
        raise ValueError(
            f"{where}precedent cites correction {correction_id!r}, which is not a standing "
            f"correction in this library. Cite one from `standing_corrections`, or say it as "
            f"a missing_information or internal_contradiction finding about the brief itself.")
    if entry["status"] != "expected":
        raise ValueError(
            f"{where}precedent cites correction {correction_id!r}, which is {entry['status']}, "
            f"not standing. A guardrail breach is not debatable, and a rule nobody has "
            f"confirmed cannot carry that. Raise it as a precedent_departure against the "
            f"campaign it came from instead.")
    # STRICTER than §6.1's check on a deck, and the difference is the length of the thing
    # quoted. §6.1's bounds — 2 gaps, 200 elided characters — are measured against a unit that
    # can be a whole packed slide; a rule is one sentence, so both bounds sit far outside it
    # and an elision can invert the rule outright: "Never use AI imagery … approves it in
    # writing" verified against "Never use AI imagery unless the client approves it in
    # writing." A plain substring drops a negation the same way, since "seed more than one
    # colourway" is inside "Do not seed more than one colourway".
    #
    # So: no elision at all, and the quote has to be most of the rule. A guardrail breach is
    # the one class of finding this product calls not debatable, and a quote that leaves out
    # the half that reverses the meaning is the assertion §6.1 exists to refuse, carrying more
    # authority than any other citation could.
    segments = _quote_segments(quote, where)
    if len(segments) > 1:
        raise ValueError(
            f"{where}precedent quotes correction {correction_id!r} with an elision. Quote the "
            f"rule as it reads — it is one sentence, and leaving words out of it can reverse "
            f"what it says. The rule on file reads: {entry['text']!r}.")
    if not _quote_is_in(segments, [entry["text"]]):
        raise ValueError(
            f"{where}precedent quotes {quote!r}, which is not what correction "
            f"{correction_id!r} says. The rule on file reads: {entry['text']!r}.")
    # The check that actually catches the inversion. A LENGTH ratio does not: dropping "Do
    # not " is seven characters out of fifty and reverses the rule completely, so the quote
    # still measures 84% of it. What matters is whether the quote carries the rule's negation,
    # and "seed more than one colourway per recipient" is a faithful substring of "Do not seed
    # more than one colourway per recipient" that says the opposite of it.
    if corrections.negated(entry["text"]) and not corrections.negated(quote):
        raise ValueError(
            f"{where}precedent quotes correction {correction_id!r} without the part that makes "
            f"it a prohibition, so the quote says the opposite of the rule. Quote it as it "
            f"reads: {entry['text']!r}.")
    if len(quote.strip()) < _MIN_RULE_QUOTE * len(entry["text"].strip()):
        raise ValueError(
            f"{where}precedent quotes only a fragment of correction {correction_id!r}, and a "
            f"fragment of a rule is not the rule. Quote it as it reads: {entry['text']!r}.")
    return {
        "correction_id": correction_id,
        "quote": quote,
        "layer": "rule",
        "checked": ["record", "standing"],
        "provenance": "; ".join(s["provenance"] for s in
                                corrections.sightings(conn, correction_id)),
        "confirmed_by": entry["confirmed_by"],
    }


def _how_to_say_it(by_class: dict, findings: list) -> str:
    """The same judgment, voiced differently depending on what is actually in it (§6.2).

    A class distinction that nothing says out loud is a column in a database. This is where
    the model learns how to put it, and it has to differ — a library of rule breaches and a
    library of departures are not the same news, and saying both as "here is what has to
    change" is exactly the machinery the review said flattens them.
    """
    parts = ["Give the user the verdict and the one-line summary."]
    if by_class.get("not_debatable"):
        # §8.6: "a rule they wrote" is false of a standing correction. Nobody wrote it — the
        # library inferred it from repetition across markets and one person confirmed it. It
        # is still not a matter of opinion, because a person stood behind it, but stating an
        # inference back to the customer as their own authored rule is the confident unfounded
        # claim this whole review is about, arriving in the voicing rather than in a finding.
        learned = [f for f in findings
                   if (f.get("precedent") or {}).get("correction_id")]
        if learned and len(learned) >= by_class["not_debatable"]:
            parts.append(
                "State the breach(es) plainly, and say where the rule came from: this is a "
                "rule the library learned from their own repeated feedback and somebody "
                "confirmed — not one they wrote down. Name it and name its provenance.")
        elif learned:
            parts.append(
                "State the breach(es) plainly and name each rule. Some are rules they wrote "
                "and some the library learned from their repeated feedback and somebody "
                "confirmed — say which is which; the provenance is on each citation.")
        else:
            parts.append(
                "State the guardrail breach(es) plainly: a rule they wrote was broken, and "
                "that is not a matter of opinion. Name the rule.")
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
                "answered. When they answer, RECORD IT with answer_finding — it used to be "
                "true that their answer had nowhere to go and the same question came back "
                "next time, and that is what answer_finding exists to stop.")
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
    computed = [f for f in findings if f.get("basis") == "computed"]
    if computed:
        parts.append(
            f"{len(computed)} of these findings are COMPUTED — the server read them out of "
            f"the brief, they are the same for every user, and each carries what it read in "
            f"`detail`. State them as facts; they are not yours and they are not opinions.")
    parts.append(
        "The reasoning behind any finding is in get_evaluation, not here. `next_actions` are "
        "offers — say them in your own words and act on the one the user picks; do not call "
        "them unasked.")
    return " ".join(parts)


# §6.6. A stable code beside a sentence, and never a word the counts cannot support — the
# discipline §5.5 used for coverage cells and §6.1 for `checked`.
_EVIDENCE_STRENGTH = {
    "no_precedent": "It cites no past campaign at all, so it is an opinion about the brief "
                    "and should be said as one.",
    "single_example": "One example is a story, not a pattern.",
    "unmeasured": "None of them has recorded results, so every comparison is to what was "
                  "planned rather than what happened.",
    "partly_measured": "Some of them have recorded results and some do not, so the judgment "
                       "is part evidence and part proposal.",
    "measured": "All of them have recorded results behind them.",
}


def _dominant(scored: list) -> Optional[str]:
    """Is one cited record carrying this judgment? (§6.6)

    "When one match dominates, say so." A judgment resting on six records where one is doing
    all the work is not a judgment resting on six records — and the arithmetic has to be a
    GAP rather than a rank, or the top of every list "dominates".

    Its own function because the two thresholds mask each other in any realistic fixture: a
    library where the gap is large enough to matter usually has a top score above the floor
    too, so a test that exercises one leaves the other free to be deleted. Both mutations
    survived the whole suite until this was pulled out and given explicit numbers.

    `scored` is (similarity, campaign_id) worst-last, and holds ONLY records that were
    actually scored — a record outside the scan has no similarity, and letting it stand in at
    zero would make "far nearer than anything else cited" a statement about the retrieval
    window rather than about the evidence.
    """
    if len(scored) < 2:
        return None
    # The floor: a top match that resembles nothing cannot dominate. Four records all scoring
    # zero have a gap of zero between them, but the reason to say nothing there is that none
    # of them is close, not that they are close to each other.
    if scored[0][0] < _DOMINANCE_FLOOR:
        return None
    if (scored[0][0] - scored[1][0]) < _DOMINANCE_GAP:
        return None
    return scored[0][1]


def _say_the_expected_measures(expected: dict) -> str:
    """What the model is told about the checklist, and only when there is one (§8.3/§8.4).

    Silent when nothing has graduated, which on a new install is always. A line explaining a
    checklist mechanism to a model that has just been handed an empty checklist is the note
    that fires on everything — and this package already carries more standing instruction than
    any one judgment needs.

    It says MISSING, not wrong. A brief that does not carry a measure the market usually
    reports may be incomplete or may be a kind of campaign where that measure is meaningless,
    and the server cannot tell which. §6.5 settled the shape: the server establishes the fact
    and the model decides whether it is a problem.
    """
    if not expected.get("expected"):
        return ""
    if not expected["missing"]:
        return (f"`expected_measures` lists what briefs in this market usually report, and "
                f"this one carries all of them. ")
    return (f"`expected_measures` lists what briefs in this market usually report — the list "
            f"is built from what the library has actually seen, not from a rule somebody "
            f"wrote. This brief does not carry: {', '.join(expected['missing'])}. That is a "
            f"gap in the brief, not a verdict on it: the measure may be meaningless for this "
            f"kind of campaign, and only you can tell. Raise it as `missing_information` if "
            f"it matters here, and say nothing if it does not. ")


# How many rules the note itself spells out. The rest stay in `standing_corrections`, where
# the model can read them — this bounds the STANDING INSTRUCTION, not the data.
_MAX_STANDING_SHOWN = 5


def _say_the_execution_drift(evidence: list) -> str:
    """What the model is told about how faithfully the cited campaigns ran (§9.5).

    Silent when nobody has ever checked one, which on a library with no delivered photographs
    is always — a standing paragraph about execution drift there is the note that fires on
    everything.

    It says WEAKER, never disqualified. Dropping a high-drift campaign would throw away the
    evidence this phase most wants kept: the UAE claw machine departed from precedent and beat
    it, and a library that hides those has learned to punish improvement.
    """
    checked = [e for e in evidence if e["execution"]["status"] != "never_checked"]
    if not checked:
        return ""
    # `_reads_as_briefed`, not the raw status: a campaign whose photographs somebody looked at
    # and confirmed match the brief belongs on the faithful side, whatever the fingerprints
    # made of it. Keyed on the status alone, the one case §9.2 says its instrument cannot see
    # was also the one case a person could not correct.
    faithful = [e for e in checked if _reads_as_briefed(e["execution"])]
    drifted = [e for e in checked
               if e["execution"]["status"] == "drifted" and not _reads_as_briefed(
                   e["execution"])]
    said = ("`execution` on each piece of evidence says whether that campaign RAN AS BRIEFED. "
            "It changes what its result is evidence OF, and not whether the result is true. ")
    if faithful:
        said += (f"{len(faithful)} of these ran as briefed, so their outcomes are evidence "
                 f"about the brief. ")
    if drifted:
        said += (f"{len(drifted)} drifted from the brief, so an outcome there says something "
                 f"worked and the brief may not have been it — weigh it as precedent "
                 f"accordingly, and say so if you rest on it. It is not disqualified: a "
                 f"change that beat the plan is the most useful thing a library can hold. ")
    said += ("Anything marked `never_checked` has not been compared at all, which is not the "
             "same as having run faithfully. ")
    return said


def _say_the_standing_corrections(standing: dict) -> str:
    """What the model is told about the rules this client has actually repeated (§8.6).

    Silent when there are none, for the same reason as the measures: explaining a mechanism to
    a model handed an empty list is the note that fires on everything.

    Unlike a missing measure, a standing correction IS a rule — it recurred across markets and
    a person confirmed it — so this says `guardrail_breach` rather than "consider". What it
    does not do is decide that the rule was broken; that is the judgment, and the server has
    only established what the rules are.
    """
    if not standing.get("standing"):
        return ""
    # Bounded, most-repeated first. The example overlay §12.3 ships carries ten rules, and ten
    # rules rendered into every note — each of which can only be expressed as a blocking
    # breach, against a 12-finding cap — is the checklist that fires on everything.
    ranked = sorted(standing["standing"], key=lambda c: -c.get("times_seen", 0))
    shown = ranked[:_MAX_STANDING_SHOWN]
    rules = "; ".join(f"[{c['correction_id']}] {c['text']}" for c in shown)
    more = (f" ({len(ranked) - len(shown)} more are in `standing_corrections`)"
            if len(ranked) > len(shown) else "")
    # WHERE THEY CAME FROM, and the two answers are different claims. This said "learned
    # from their own feedback, NOT WRITTEN BY THEM" about all of them — false for every rule a
    # customer declared in their own rulebook, which they did write and which recurred
    # nowhere. `load_declared_corrections` is careful that a judgment citing one must not read
    # identically to one citing an inferred rule, and the only place that survived was inside
    # a provenance string, while this sentence asserted the inferred story over the top.
    declared = [c for c in ranked if "(declared in rulebook " in (c.get("provenance") or "")]
    where = ("rules this client WROTE DOWN in their own rulebook, plus rules this library "
             "watched recur across their decks until somebody confirmed them"
             if declared and len(declared) < len(ranked) else
             "rules this client WROTE DOWN in their own rulebook"
             if declared else
             "rules this client has repeated across markets and somebody has confirmed — "
             "learned from their own feedback, not written by them")
    return (f"`standing_corrections` are {where}, and not invented by the product. Each "
            f"carries its provenance, which says which it is. The most-repeated: {rules}.{more} Check the "
            f"brief against each one that applies. Where one is broken, that is a "
            f"`guardrail_breach` citing `precedent: {{correction_id, quote}}` with the quote "
            f"taken from the rule's own words; the server attaches the provenance so the "
            f"reader can see which deck it was learned from, and you should name it. A breach "
            f"is blocking only where the rule plainly applies to this kind of brief — where "
            f"it does not, say nothing, because a rule applied regardless is how a checklist "
            f"stops being read. ")


def _no_subject_to_check() -> dict:
    """The checklist for a proposal that is not a record yet (§8.3).

    An empty `missing` here would read as "this brief carries everything expected of it", which
    is a clean bill of health the server has no basis for — the checklist is per market, and a
    loose block of text has no market. Saying the check did not run is the honest answer, and
    it is §7.1's rule: `nothing_to_check` is not a clean result, it is an unchecked one.
    """
    return {"market": None, "expected": [], "carried": [], "missing": [],
            "basis": "computed", "status": "nothing_to_check",
            "what_it_means": ("This proposal is not a stored record, so there is no market to "
                              "read a checklist for. Store it and pass `campaign_id` to see "
                              "which expected measures it carries.")}


def _contract_for_this_brief(computed: dict) -> str:
    """The part of the contract that cannot be static (§7.5).

    §7.4 put the procedure where it loads once. The review's argument for repeating it here
    is about WHEN rather than what: "tool descriptions load once, at the start of a session,
    and compete with everything the user has done since. The moment that matters is when the
    model is holding the evidence package and about to judge."

    What goes FIRST is what is specific to this brief, because a model reads a long field from
    the top and the procedure is the half it has already been given. §12.1 put the RULEBOOK
    first of all, for the same reason at one more remove: the rules are the frame the rest is
    judged inside, and a rule read after the evidence is a rule applied to a verdict that has
    already formed.
    """
    lines = [rulebook.as_contract()]
    said = []
    for code, fact in computed.items():
        if fact["status"] in ("absent", "contradicted", "partial", "present"):
            said.append(f"  {code}: {fact['status']} — {fact['what_it_means']}")
    if said:
        lines.append("WHAT THE SERVER ALREADY ESTABLISHED ABOUT THIS BRIEF (do not re-derive; "
                     "dispute one only by quoting the `evidence` it carries):")
        lines += said
    return "\n".join(lines) + "\n\n"


def _mixed_model_warning(conn) -> list:
    """Two embedding models in one index (§7.2).

    "Record the embedding model version on every stored vector so a model upgrade is a
    visible migration rather than a silent re-ranking." This is the visible part: similarities
    produced by two different models are not comparable, so a ranking across them is
    arithmetic on incompatible numbers — and the whole evidence package is that ranking.
    """
    models = store.embedding_models(conn)
    if len(models) < 2:
        return []
    return [notices.notice(
        "mixed_embedding_models",
        detail=f"vectors in this library were produced by {len(models)} different embedding "
               f"models ({', '.join(sorted(models))})",
        affects="this evidence package and every similarity in it",
        remedy="re-index the library so every vector comes from one model",
        # NOT "run reembed": there is no such tool. `finish_indexing` is the one that exists,
        # and naming a gesture the user cannot perform is the failure L5 records about the
        # Windows installer — a remedy nobody can follow is worse than none, because it moves
        # the blame to them.
        next_step="there is no one-step re-index yet; re-upload the affected records, or "
                  "call finish_indexing after clearing the index")]


def _stamp(conn, *, receipt: Optional[dict], model_id: Optional[str]) -> dict:
    """What produced this judgment (§7.6, D4).

    The review's last clause is the one that decides the design: "when they agree, you have
    evidence the agreement is real rather than luck". A stamp is not an audit trail for its
    own sake — it is what lets AGREEMENT be read as evidence, which is the premise §7.7's
    golden set rests on. Two runs that agree while differing in embedding model and rulebook
    version agree about nothing in particular.

    Four of the five fields are facts the server holds. `model_id` is the one it cannot
    observe: the judging model is on the other side of the protocol. It is recorded as unknown
    unless the caller says, and marked `stated` when they do — a stamp with a wrong field in
    it is worse than one with a missing field, because the diff that is supposed to explain a
    disagreement would then explain it wrongly.
    """
    retrieved = None
    if receipt:
        scores = dict(receipt.get("similarities") or [])
        retrieved = [{"campaign_id": cid, "similarity": scores.get(cid)}
                     for cid in receipt["campaign_ids"]]
    return {
        "server_version": version.VERSION,
        "rulebook_version": rulebook_version(),
        "embedding_model": (receipt or {}).get("embedding_model") or embedding_model_id(),
        # None rather than [] without a receipt: an empty list reads as "the search returned
        # nothing", and "nobody recorded a search" is a different statement.
        "retrieved": retrieved,
        "model_id": model_id,
        "model_id_basis": "stated" if model_id else None,
        "model_id_note": ("the server cannot observe which model produced a judgment; pass "
                          "`model_id` to record it"),
        # D18: a judgment made over a half-indexed library is a different judgment from one
        # made over a whole one, and the warning that said so lived for exactly one response.
        "warnings_at_retrieval": (receipt or {}).get("warnings") or [],
        "basis": "computed",
    }


def compare_provenance(conn, first: str, second: str) -> dict:
    """Why two judgments might differ, field by field (§7.6).

    "When two users disagree, the diff of those five fields usually explains it in seconds."
    That only works if something actually produces the diff — a stamp nobody can compare is
    an audit trail, and the review asked for a diagnosis.
    """
    a = (store.get_evaluation(conn, first) or {}).get("provenance") or {}
    b = (store.get_evaluation(conn, second) or {}).get("provenance") or {}
    if not a or not b:
        return {"error": "one of those judgments carries no provenance stamp"}
    fields = ("server_version", "rulebook_version", "embedding_model", "model_id")
    differs = [f for f in fields if a.get(f) != b.get(f)]
    same = [f for f in fields if a.get(f) == b.get(f)]
    if (a.get("retrieved") or []) != (b.get("retrieved") or []):
        differs.append("retrieved")
    else:
        same.append("retrieved")
    return {
        "differs": differs,
        "same": same,
        "first": a, "second": b,
        "what_it_means": (
            "These two judgments were produced under the same conditions, so a difference "
            "between their verdicts is a difference in reasoning rather than in setup — and "
            "an agreement between them is evidence rather than luck."
            if not differs else
            f"These two judgments differ in {', '.join(differs)}. Explain any disagreement "
            f"between their verdicts from that before reading anything into the reasoning."),
    }


def _window_check(conn, retrieval: Optional[str], cited_ids: Optional[list], *,
                  subject_title: str, campaign_id: Optional[str]) -> dict:
    """Was each cited record in the evidence this judgment was actually given? (D77, §7.2)

    §6.1 checks that the record contains the quote. This checks that the record was in front
    of the reasoner at all, which is a different failure and the one Phase 7 needs: two users
    judging the same brief should rest on the same evidence, and "the words are somewhere in
    that record" says nothing about that. It also catches what §6.1's record check cannot —
    cherry-picking a record retrieval never surfaced, and a record remembered from an earlier
    session.

    `not_recorded` rather than a clean result when there is no receipt. An absent check reads
    as a passed one, which is the collapse §6.4 needed separate codes to avoid.
    """
    if not retrieval:
        return {"window": "not_recorded"}
    receipt = store.get_retrieval(conn, retrieval)
    if receipt is None:
        raise ValueError(
            f"retrieval {retrieval!r} is not a receipt this server issued. It comes back from "
            f"prepare_evaluation as `retrieval.receipt`; pass that value or omit it.")
    # Bound to its subject. Unbound, a receipt taken for one brief laundered any citation
    # into `from_the_window` for a different brief — and stamped a server-`computed` closest
    # precedent onto a subject it was never about. The receipt's whole claim is "this evidence
    # was in front of the reasoner FOR THIS BRIEF", and half of that was unchecked.
    if campaign_id and receipt.get("campaign_id") and receipt["campaign_id"] != campaign_id:
        raise ValueError(
            f"retrieval {retrieval!r} was taken for campaign {receipt['campaign_id']!r} and "
            f"this judgment is about {campaign_id!r}. A receipt says which evidence was in "
            f"front of you for a particular brief; using another brief's receipt says "
            f"nothing about this one.")
    if not campaign_id and receipt.get("subject_title") != subject_title:
        raise ValueError(
            f"retrieval {retrieval!r} was taken for {receipt.get('subject_title')!r} and this "
            f"judgment is about {subject_title!r}. Call prepare_evaluation for the brief you "
            f"are judging and pass the receipt it returns.")

    shown = set(receipt["campaign_ids"])
    cited = list(dict.fromkeys(cited_ids or []))
    # A snapshot ages. A record superseded or deleted since the window was taken was in front
    # of the reasoner and is no longer evidence anybody can reach, and reporting it as plain
    # `from_the_window` would let the server assert as current a precedent its own search
    # would not return.
    superseded = store.get_superseded_campaign_ids(conn)
    gone = [c for c in cited if c in shown and store.get_campaign(conn, c) is None]
    stale = [c for c in cited if c in shown and c in superseded]
    return {
        "window": "recorded",
        "retrieval_id": retrieval,
        "from_the_window": [c for c in cited if c in shown],
        "outside_the_window": [c for c in cited if c not in shown],
        **({"superseded_since_retrieval": stale} if stale else {}),
        **({"deleted_since_retrieval": gone} if gone else {}),
    }


# The words a verdict would use to say a number is not clean. Matched on word boundaries by
# §9.7's `_mentions` machinery, so "unconfounded" does not count as saying "confounded".
_SAID_IT_WAS_CONFOUNDED = ("confounded", "confounding", "not clean evidence",
                           "not a clean result", "ran through", "coincided", "overlapped",
                           "context event", "caveat")


# §11.5's equivalent of `_SAID_IT_WAS_CONFOUNDED`. A verdict that names the split has said
# what there was to say.
_SAID_IT_WAS_CONTESTED = ("disagree", "disagreed", "disagreement", "contested", "split",
                          "divided", "two views", "both views", "not unanimous",
                          "one of them", "mixed view")


def _contested_silence(conn, cited_ids, written: str) -> dict:
    """Which cited campaigns people disagreed about that the verdict never mentions (§11.5).

    Byte for byte the shape of `_confounded_silence`, and for the reason review named: §9.8
    built a CHECK that reads what the model actually wrote, and §11.5 built only a sentence in
    a note carrying fifteen other instructions. This product has twice learned that an
    instruction in a payload gets dropped — so a judgment citing a two-to-one split as though
    it were unanimous produced no finding, no flag, nothing, which is the exact loss §11.5
    exists to prevent and undetectable from the outside.

    The finding is the SILENCE, not the split: a verdict that says the precedent was contested
    is doing the right thing, and raising one anyway is the wallpaper that teaches a reader to
    skip server findings.
    """
    said = context._plain(written)
    if any(context._plain(phrase) in said for phrase in _SAID_IT_WAS_CONTESTED):
        return {"status": "absent", "basis": "computed", "unaddressed": [],
                "what_it_means": "The verdict says the evidence it rests on was contested."}
    unaddressed = []
    for cid in list(dict.fromkeys(cited_ids or [])):
        record = store.get_campaign(conn, cid)
        if record and record.get("disagreement"):
            unaddressed.append(record["title"])
    if not unaddressed:
        return {"status": "absent", "basis": "computed", "unaddressed": [],
                "what_it_means": "Nothing this verdict cites was disagreed about."}
    return {
        "status": "present", "basis": "computed", "unaddressed": unaddressed,
        "what_it_means": (
            f"This verdict rests on {', '.join(unaddressed)}, which two people recorded "
            f"different views about, and does not say so. A contested precedent is stronger "
            f"evidence about this client's taste than an uncontested one — and quoting it as "
            f"though everybody agreed throws that away."),
    }


def _confounded_silence(conn, cited_ids, written: str) -> dict:
    """Which cited campaigns' confounded outcomes the verdict never mentions (§9.8).

    A phrase match, and it degrades the safe way round: a false positive lands as one extra
    `should_fix` on a judgment that did hedge in words the matcher missed, and a false
    negative just means no finding — which is where this was before. §9.7 accepted the same
    trade on the same machinery.
    """
    said = context._plain(written)
    if any(context._plain(phrase) in said for phrase in _SAID_IT_WAS_CONFOUNDED):
        return {"status": "absent", "basis": "computed", "unaddressed": [],
                "what_it_means": "The verdict says the evidence it rests on is not clean."}
    unaddressed = []
    for cid in list(dict.fromkeys(cited_ids or [])):
        record = store.get_campaign(conn, cid)
        if not record:
            continue
        actual = [m for m in record.get("metrics") or [] if m["metric_type"] == "actual"]
        if any(m.get("confounded") for m in actual):
            unaddressed.append(record["title"])
    if not unaddressed:
        return {"status": "absent", "basis": "computed", "unaddressed": [],
                "what_it_means": "Nothing this verdict cites carries a confounded outcome."}
    return {
        "status": "present", "basis": "computed", "unaddressed": unaddressed,
        "what_it_means": (
            f"This verdict rests on {', '.join(unaddressed)}, whose measured outcome ran "
            f"through something else that was going on — and says nothing about it. The "
            f"outcome still counts; quoting it as though nothing else was happening "
            f"overstates it. Say what it ran through, or say why it does not matter here."),
    }


def _reconciliation_context(conn, evaluation: dict, actual_metrics: list) -> dict:
    """What else was going on, for the record §9.9 is built around (§9.8).

    The SUBJECT's confounders, not the evidence's — `_confounded_at_save` stamps what the
    cited precedent was confounded by, which is a useful provenance record and the wrong
    subject here. And computed at reconcile time rather than read from that stamp, because the
    subject's actuals arrive AFTER the verdict: a save-time figure is by construction the wrong
    moment for it.
    """
    confounded = [m for m in actual_metrics if m.get("confounded")]
    evidence = evaluation.get("evidence") or {}
    clash = evidence.get("calendar_clash") or {}
    # De-duplicated. Twelve workbook rows confounded by the same six events produced seventy
    # entries — §9.8 fixed exactly this shape one item earlier and this reintroduced it.
    linked = {}
    for m in confounded:
        for e in m["confounded_by"]:
            linked.setdefault(e["id"], {"id": e["id"], "description": e["description"],
                                        "why": e["why"], "attribution": e["attribution"]})
    # What was KNOWN when the verdict was written. §9.5's `execution_at_save` and §9.8's
    # `confounded` both name §9.9 in their docstrings — "it cannot do that against a caveat
    # nobody stored" — and both landed in the evaluation's evidence, and this read neither.
    # The delta is the most interesting thing in the record: we predicted 55%, we shipped as
    # briefed, we got 41%, and a port was shut for nine days that nobody knew about then.
    known = {
        "confounded_evidence": [e["campaign_id"] for e in evidence.get("confounded") or []],
        "execution": [{"campaign_id": e["campaign_id"], "status": e["status"]}
                      for e in evidence.get("execution_at_save") or []],
        **({"calendar_clash": clash["status"],
            "what_it_means": clash.get("what_it_means", "")} if clash else {}),
    }
    return {
        "basis": "computed",
        "confounded": bool(confounded),
        "confounded_by": list(linked.values()),
        "known_when_predicted": known,
        # Did the world move after the verdict? A confounder nobody knew about when the call
        # was made is a different lesson from one that was on the record and ignored.
        "learned_since": bool(confounded) and not evidence.get("confounded"),
    }


def _say_the_reconciliation_context(conn, evaluation: dict, actual_metrics: list) -> str:
    confounded = [m for m in actual_metrics if m.get("confounded")]
    if not confounded:
        return ""
    since = not (evaluation.get("evidence") or {}).get("confounded")
    return ((" What confounded this outcome WAS NOT KNOWN when the judgment was written — it "
             "was recorded afterwards. A call made without that information is not the same "
             "mistake as one made in spite of it, and the lesson should say which."
             if since else "")
            + " `context` says something else was going on while this ran, so the actual figures "
            "are not clean evidence of whether the judgment was right. Say so in the lesson: "
            "a prediction that missed because a port was shut is a different lesson from one "
            "that missed because the reasoning was wrong, and recording them the same way is "
            "how a calibration figure stops meaning anything. It is an overlap in time and "
            "never a cause.")


def _confounded_at_save(conn, cited_ids: Optional[list]) -> list:
    """Which cited outcomes were confounded when the verdict was written (§9.8).

    §9.9 lines up predicted against delivered against actual against CONTEXT, and it cannot do
    that against a caveat nobody stored. Same reasoning as §9.5's `execution_at_save`: the
    overlap is recomputed on every read, so a judgment written before an earthquake was
    recorded has to say that it was.
    """
    marked = []
    for cid in list(dict.fromkeys(cited_ids or [])):
        record = store.get_campaign(conn, cid)
        if not record:
            continue
        actual = [m for m in record.get("metrics") or [] if m["metric_type"] == "actual"]
        if actual and actual[0].get("confounded"):
            marked.append({"campaign_id": cid,
                           "events": [{"id": e["id"], "description": e["description"],
                                       "starts_on": e["starts_on"], "kind": e["kind"],
                                       "attribution": e["attribution"]}
                                      for e in actual[0]["confounded_by"]]})
    return marked


def _execution_at_save(conn, cited_ids: Optional[list]) -> dict:
    """How faithfully each cited campaign had been shown to run, at the moment of the verdict.

    Absent rather than empty when nothing was cited: an `execution_at_save: []` beside a
    verdict reads as "we checked the precedent's execution and found none of it remarkable",
    when the truth is there was no precedent to check. Every other empty list in this file is
    settled the same way.
    """
    cited = list(dict.fromkeys(cited_ids or []))
    stamped = []
    for cid in cited:
        if store.get_campaign(conn, cid) is None:
            continue
        note = _execution_note(conn, cid)
        stamped.append({"campaign_id": cid, "status": note["status"], "score": note["score"],
                        "classified": note["classified"],
                        "what_it_means": note["what_it_means"]})
    return {"execution_at_save": stamped} if stamped else {}


def _evidence_strength(conn, *, cited_ids: Optional[list], text: str) -> dict:
    """What this judgment rests on, counted by the server (§6.6, D3).

    The review's complaint is that "a verdict resting on five concluded campaigns with
    verified outcomes and one resting on a single proposed brief currently look identical" —
    and the load-bearing word is IDENTICAL. Two judgments of very different worth were
    presented the same way, so a reader had no signal to weigh them by. That makes this a
    shape problem, not a data problem, which decides two things.

    Counted from `cited_ids`, NOT from the evidence package. What a judgment rests on is what
    it cited; a verdict shown eight precedents and citing one rests on one, and reporting
    eight would be the overstatement this field exists to prevent, made by the field written
    to prevent it.

    And a number nobody reads is not a signal. Five counts and a similarity are a row of
    digits; "one match is carrying this" is a sentence. Hence `strength` and
    `what_it_means`, which never claim more than the counts support.
    """
    cited = list(dict.fromkeys(cited_ids or []))
    records, unresolved = [], []
    for cid in cited:
        record = store.get_campaign(conn, cid)
        (records if record else unresolved).append(record or cid)
    concluded = [r for r in records if r.get("status") == "concluded"]
    # TWO different things, and conflating them is what made this field contradict itself out
    # loud. `with_results` is actual metric rows, which is what `coverage` and `readiness`
    # mean by measured. `verified` is a performance VERDICT somebody stood behind, which is
    # what the review means by "verified rather than stated performance". Five concluded
    # campaigns with metrics and no performance tags were reported as "none of them has
    # measured results" while `coverage` called the same five `measured` — D88's drift, and
    # the strength ladder is now built on the first while still reporting the second.
    # `campaigns_with_actual_metrics` returns an empty set on a database with no `metrics`
    # table. Guarded there rather than here: this is the THIRD save-path function to crash on
    # an upgraded v0.2.0 database, and §5.3 was doing it too — unnoticed, because the legacy
    # tests happened to cite nothing.
    measured_ids = store.campaigns_with_actual_metrics(conn)
    with_results = [r for r in records if r["id"] in measured_ids]
    verified = [r for r in records
                if {t.get("value") for t in (r.get("tags") or [])
                    if t.get("source") == "verified"} & {"performed_well", "underperformed"}]

    dominated_by, top_similarity, similarity_checked = None, None, True
    if records:
        try:
            # Wide enough that a cited record is scored unless it is genuinely far away. The
            # first version asked for `len(records) + 5` of the WHOLE library, so a cited
            # record ranked eleventh scored 0.0 — reported as a real similarity, and standing
            # in for the runner-up in the dominance gap, which then measured "far nearer than
            # anything else cited" against records that were simply never retrieved.
            ranked = find_similar(conn, text=text, top_k=_SIMILARITY_SCAN, full_detail=False)
        except embedding.Unavailable:
            # NOT swallowed into a score. `Unavailable` is a `ValueError`, and catching it
            # with one reported `top_similarity: 0.0` — indistinguishable from "dissimilar" —
            # in the same response where §6.4 correctly said `could_not_check`. That is the
            # collapse §6.4 exists to prevent, reintroduced 240 lines below it.
            ranked, similarity_checked = [], False
        except ValueError:
            ranked = []
        scores = {m["campaign_id"]: m["similarity"] for m in ranked}
        # Only records that were actually scored. A record outside the scan has NO similarity,
        # which is not the same as a similarity of zero, and the difference decides whether
        # "one match is carrying this" is a statement about the evidence or about the window.
        mine = sorted(((scores[r["id"]], r["id"]) for r in records if r["id"] in scores),
                      reverse=True)
        top_similarity = mine[0][0] if mine else None
        dominated_by = _dominant(mine)

    if not records:
        strength = "no_precedent"
    elif len(records) < 2:
        strength = "single_example"
    elif not with_results:
        strength = "unmeasured"
    elif len(with_results) < len(records):
        strength = "partly_measured"
    else:
        strength = "measured"
    what = _EVIDENCE_STRENGTH[strength]
    if dominated_by:
        what += (" And one of them is carrying it: the closest match is far nearer this brief "
                 "than anything else cited, so the judgment is effectively resting on that "
                 "one record.")
    if not similarity_checked:
        what += (" How close any of them is to this brief could not be checked — the "
                 "similarity search did not run.")
    return {
        "precedents": len(records),
        "concluded": len(concluded),
        "with_results": len(with_results),
        "verified": len(verified),
        "top_similarity": top_similarity,
        "similarity_checked": similarity_checked,
        "dominated_by": dominated_by,
        # An id that resolves to no record cannot be evidence, and counting it would inflate
        # the one number this item exists to make honest. §6.1 verifies the precedent on a
        # finding; `cited_ids` is a separate list and nothing checked it.
        "unresolved_citations": unresolved,
        "strength": strength,
        "basis": "computed",
        "what_it_means": what,
    }


def _exit_checklist(verdict: Optional[str], findings: list) -> list:
    """The exit condition as something somebody can tick off (§6.5).

    **Only for a revise**, and the first version got this wrong in the way that matters: it
    composed a checklist for every verdict, so an approve carrying a `should_fix` came back
    with a list of things to change — the exact "reservation the verdict does not admit to"
    that the `approve_if` rule two functions up refuses. The server was manufacturing it from
    the `fix` it had just demanded. A reject got one too, while being refused an exit
    condition on the grounds that having one makes it a revise. The rule was enforced on the
    sentence and contradicted by the list.

    The fixes are still on the findings for an approve or a reject. What they are not is an
    exit condition, because neither verdict has an exit.

    The review asks for an exit condition that is "testable" and that "doubles as the note
    the partner receives", and those are the same requirement: a sentence cannot be checked
    off and a list can. Composed from the `fix` lines the findings already carry, so it
    cannot drift from them — a hand-written checklist beside a findings array is two sources
    of truth for one thing, which is the failure this project has now hit four times.

    Notes are left out. "Worth saying once; nobody has to act" is the severity's own
    definition, and an item on an exit checklist is by definition something to act on.
    """
    if verdict != "revise":
        return []
    # `.get`, like every other read-path consumer here: a stored judgment from before the
    # `fix` rule has no such key, and `_exit_checklist` was the one place that indexed it —
    # so `get_evaluation` and §6.3's moment raised KeyError on a row nothing else minded.
    # `departure` rides along because §6.2 distinguishes a fix that CHANGES something from
    # one that answers a question, and a checklist stripped of it cannot show the difference
    # to the person ticking it off.
    return [{"finding_id": f.get("id"), "severity": f.get("severity"),
             "departure": f.get("departure"), "fix": f.get("fix")}
            for f in findings
            if f.get("severity") in ("blocking", "should_fix") and f.get("fix")]


def _say_the_evidence_strength(evidence: dict) -> str:
    """§6.6, said rather than counted at.

    The review's word is "identical": two judgments of very different worth looked the same.
    Returning the counts fixes the data and not the presentation, and this project's own note
    on the response shape says models mirror what a result CONTAINS far more reliably than
    they follow instructions inside one — so the sentence goes in the note and the counts go
    in the field, and the note says to give the strength BEFORE the verdict, because after it
    the verdict has already landed.
    """
    outside = evidence.get("outside_the_window") or []
    line = (f" This judgment rests on {evidence['precedents']} cited campaign(s), "
            f"{evidence['concluded']} concluded, {evidence['with_results']} with recorded "
            f"results, {evidence['verified']} with a measured performance verdict behind "
            f"them. {evidence['what_it_means']}")
    if (evidence["strength"] in ("no_precedent", "single_example", "unmeasured")
            or evidence["dominated_by"] or not evidence["similarity_checked"]):
        line += (" Say how much this rests on BEFORE giving the verdict — afterwards the "
                 "verdict has already landed and the caveat reads as hedging.")
    if outside:
        line += (f" {len(outside)} of the campaigns cited were NOT in the evidence the server "
                 f"retrieved for this judgment — see `outside_the_window`. Say where they "
                 f"came from: a record nobody was shown is not shared evidence, and the next "
                 f"person judging this brief will not see it.")
    if evidence["unresolved_citations"]:
        line += (f" {len(evidence['unresolved_citations'])} of the ids cited resolve to no "
                 f"record and were not counted.")
    return line


def _say_the_disconfirming_check(check: dict) -> str:
    """§6.4 makes overconfidence visible, which it can only do if somebody is told.

    One sentence per code, and there are five because five different things can have
    happened. The first version had three, two of which covered states meaning the opposite
    of each other — an embedder outage came back as "the verdict was argued against and it
    held", which is the collapse this item exists to prevent, on the failure that actually
    happens.
    """
    code = check["code"]
    if code == "contradicting_precedent":
        return (" The server searched for precedent that CONTRADICTS this verdict and found "
                f"{len(check['uncited'])} the judgment did not cite — see `disconfirming`, "
                "which carries the measured result behind each. Say so before the verdict: a "
                "campaign that looked like this and went the other way is the one thing most "
                "likely to change what the marketer does.")
    if code == "could_not_check":
        return (" The search for contradicting precedent COULD NOT RUN. Do not report this "
                "verdict as unchallenged — nothing was checked.")
    if code == "nothing_to_check_against":
        return (" This verdict could not be checked against the other side: no campaign in "
                "the library carries a measured performance verdict. Say the check was not "
                "possible — do NOT say nothing contradicted it. `could_be_checked_if` names "
                "the campaigns that would make it possible.")
    if code == "verdict_rests_on_a_rule":
        return (" No disconfirming search was run: this verdict rests only on rules that were "
                "broken, and a rule is not a matter of precedent.")
    # nothing_ranked — checked, and nothing close enough came back.
    return (" The server searched for precedent contradicting this verdict. Campaigns with "
            "measured results pointing the other way exist, and none of them resembles this "
            "brief closely enough to argue with it. Worth one sentence, and no more than "
            "one: what was checked is similarity, not the merits.")


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
# Below this, "resembling this one" is not a sentence anybody can defend. `find_similar` has
# no cutoff — with a tag filter it ranks every candidate and slices — so the first version
# reported a single-market print campaign at similarity 0.0 as precedent resembling a
# creator-led launch, and told the model to raise it before the verdict. Once a library holds
# one measured campaign per pole, that fires on every save: the always-on field nobody reads.
_DISCONFIRMING_FLOOR = 0.25
# "When one match dominates, say so" — a gap, not a rank, or the top of every list dominates.
_DOMINANCE_FLOOR = 0.2
_DOMINANCE_GAP = 0.15
# How far down the library to look when scoring the records a judgment cited. Wide, because
# the question is "how close is THIS cited record", not "what are the nearest records" — and a
# cited record that falls outside the scan is reported as unscored rather than as zero.
_SIMILARITY_SCAN = 200


# §7.4/§7.5. The evaluation procedure, written ONCE and referenced twice: in the server-level
# `instructions` (a system prompt every client surfaces, which most servers leave empty) and
# in `prepare_evaluation`'s own description. Not two copies that agree today — this project
# has watched a hand-maintained copy drift four separate times, and a procedure whose two
# halves disagree is worse than one that lives in a single place, because each reader is
# confident and they are not reading the same thing.
#
# What it does NOT contain is as deliberate as what it does. The review asks for "the
# scorecard's six criteria"; those belong to the customer's rulebook, and hard-coding one
# customer's rubric into a product that ships generic is the thing the product owner ruled
# out. §12.1 built the file they go in and deliberately shipped it empty — the scorecard is
# §12.2's, and this says so rather than omitting it silently.
#
# It also does not restate the rules the RULEBOOK carries, and that is the same decision from
# the other side. The first rulebook shipped six rules about judgment discipline, four of
# which are written here word for word — two statements of one rule in one payload, which is
# what §7.4 exists to prevent. The procedure keeps the product's own discipline; the rulebook
# keeps the customer's rules about their briefs; neither repeats the other.
EVALUATION_PROCEDURE = """\
HOW TO JUDGE A BRIEF. Every user of this library gets this same procedure; following it is
what makes two people's judgments of one brief comparable.

1. Call `prepare_evaluation`, passing `campaign_id` if the brief is already a record — the
   server then derives the query and filters from the record, so the same subject retrieves
   the same evidence however you describe it. Read it before writing anything. `computed`
   holds facts the server established by reading the brief (dates, budget, engagement rates,
   channels named, calendar contradictions): do not re-derive them, and dispute one only by
   quoting the `evidence` it carries. `outcomes` splits precedent into what WORKED and what
   did not, by measured result.
2. Reason about what is genuinely judgment: precedent fit, premise disagreements, whether a
   difference is an improvement. Weight concluded campaigns over proposed, and `verified`
   performance over `stated` — a stated claim is somebody's impression.
3. Call `save_evaluation`, passing `retrieval` (the receipt from step 1) so the server can
   record which of your citations it had actually shown you.

EVERY FINDING CITES SOMETHING IT CAN QUOTE. A precedent carries a `quote` from the record it
names and the server checks it is really there; paraphrase is refused. Mark the LAYER: `body`
is what the deck says, `commentary` is what somebody said ABOUT it. Quoting a reviewer's
objection is often the best evidence there is — storing it unmarked says the deck claimed it,
which is false.

TWO CLASSES OF FINDING, AND THEY ARE NOT THE SAME KIND OF STATEMENT.
  `guardrail_breach`     — a rule the customer wrote was broken. Cite the rule. NOT DEBATABLE:
                           state it, never soften it into a question, and never a `note`.
  `precedent_departure`  — done differently from a campaign on file. Cite the campaign, and
                           say which way it departs: `regression`, `unexplained`, or
                           `possible_improvement`. Debatable by design — ask, do not instruct.
  `missing_information` / `internal_contradiction` — claims about the brief in front of you.
                           They need no citation; do not go looking for one to satisfy a shape.

THE SERVER ARGUES WITH YOU. After you save it searches for precedent CONTRADICTING your
verdict and returns `disconfirming`. Read the `code`, not the absence of rows: "could not be
checked" and "nothing came back" are opposite conclusions. If it found something you did not
cite, say so before the verdict. `evidence` counts what the judgment rests on — give that
before the verdict too, because afterwards a caveat reads as hedging.

OUTPUT CONTRACT. `verdict` (approve / revise / reject), a one-line `summary`, one short
finding per problem with `severity`, `kind`, and a `fix` if above a note. A `revise` carries
`approve_if`: the change that would make it an approve. An `approve` and a `reject` may not.
Say the verdict, the summary and what has to change; the reasoning is in `get_evaluation`.

THE RULEBOOK. Every rule written in it is at the top of the note, in full, whatever the brief
is about — never retrieved by similarity, so a rule applies to a brief that resembles nothing.
A `guardrail_breach` cites one by id and quotes it as written. It ships EMPTY: guidelines
uploaded as records are not rules, and where none are written the honest answer is that no
rule was checked."""


# The rulebook this judgment was made under (§12.1). A function reading the file, not a
# constant beside it: a constant and a file that can disagree is the hand-maintained-copy
# failure this project has hit four times, and here it would make §7.6's stamp worthless — a
# judgment labelled with a version it was not made under is worse than an unlabelled one.
#
# Called rather than resolved at import, because `rulebook.load()` RAISES on a broken file and
# that has to reach whoever is starting the server. Swallowed at import time it would leave a
# half-started process running with no rules in it, which is the silent failure the whole item
# exists to remove.
def rulebook_version() -> str:
    import rulebook

    return rulebook.version()


# §7.2. The caller does not choose how much evidence a judgment rests on: "a caller who asks
# for twenty gets a different evidence package from one who asks for three, and neither of
# them chose the brief".
_PINNED_TOP_K = 5
# Which filters say WHICH BRIEF this is. The subject record answers these, and a caller
# passing one alongside `campaign_id` is describing a different subject. Everything else
# narrows within the subject and stays the caller's, recorded on the receipt.
# `markets` is here as well as `market`: a subject whose activation spanned several countries
# carries the list and not the single value, and deriving nothing for it meant two records
# describing one brief retrieved different evidence depending on which field was filled in.
# Which brief this IS — the record answers all of these, so passing one alongside a
# `campaign_id` asks for evidence about a different subject (§7.2). §9.7's `starts_on`/
# `ends_on` are the same kind of claim and get the same refusal, but are not in here because
# they are not retrieval filters: see `caller_window` in `prepare_evaluation`.
_SUBJECT_FILTERS = ("market", "region", "collection", "markets")


def embedding_model_id(space: str = "campaign") -> str:
    """Which model is producing vectors right now, as one string.

    Recorded on every vector so that changing the embedder is a visible migration rather than
    a silent re-ranking of every judgment the library will ever make — the similarities from
    two models are not comparable, and ranking across them is arithmetic on incompatible
    numbers.
    """
    if space == "asset":
        return (f"clip/{config.CLIP_MODEL_NAME}" if config.CLIP_PROVIDER != "hash"
                else "hash")
    provider = config.EMBED_PROVIDER
    if provider == "ollama":
        # D29: the tag names a moving target. The digest pins it when the embedder will say
        # what it loaded, and is simply absent when it will not — see `embedding.model_digest`.
        digest = embedding.model_digest()
        return (f"ollama/{config.OLLAMA_EMBED_MODEL}@{digest}" if digest
                else f"ollama/{config.OLLAMA_EMBED_MODEL}")
    if provider == "voyage":
        return f"voyage/{getattr(config, 'VOYAGE_EMBED_MODEL', 'default')}"
    return provider


def _add_vector(conn, vector_id: str, vec: list, *, space: str = "campaign") -> None:
    """`vectorstore.add`, plus which model made it. One function so the two cannot drift.

    The model recorded is the one for THIS space. An asset vector comes from CLIP and a chunk
    vector from the text embedder, and stamping the text model on both made the provenance
    false for every image — and would have reported "mixed models" the moment the text
    embedder changed, on the strength of asset rows that had nothing to do with it.
    """
    vectorstore.add(conn, vector_id, vec, space=space)
    store.record_vector_model(conn, vector_id, embedding_model_id(space), space=space)


def _pole_search(conn, *, tag: str, text: Optional[str] = None,
                 campaign_id: Optional[str] = None, top_k: int,
                 exclude: Optional[set] = None) -> list:
    """The most similar records with a MEASURED verdict of one kind, above the floor.

    `verified` only: a performance tag somebody typed is an impression, and an impression
    cannot be the counterweight to a judgment. `reference` records are dropped — the rulebook
    is not precedent, and nothing else here treats it as any.
    """
    exclude = exclude or set()
    try:
        matches = find_similar(conn, text=text, campaign_id=campaign_id,
                               top_k=top_k + len(exclude) + _MAX_DISCONFIRMING,
                               tags=[{"value": tag, "source": "verified"}],
                               full_detail=False)
    except embedding.Unavailable:
        raise
    except ValueError:
        return []
    kept = [m for m in matches
            if m["campaign_id"] not in exclude
            and m.get("record_type") != "reference"
            and (m.get("similarity") or 0) >= _DISCONFIRMING_FLOOR]
    return kept[:top_k]


def _disconfirming_search(conn, *, verdict: str, subject_text: str, query_basis: str,
                          cited_ids: Optional[list], by_class: dict,
                          findings: Optional[list] = None,
                          subject_campaign_id: Optional[str] = None) -> dict:
    """One query for precedent that contradicts this verdict, run by the SERVER (§6.4).

    The review asked for the search and for the result to be recorded, "including nothing".
    It did not say who runs it, and that is the decision worth writing down: a model that has
    already reached a verdict, asked to go and find evidence against itself, is marking its
    own homework — and the failure the review names, drifting toward confirming the first
    strong match, is exactly what would shape the query. This codebase settled the same
    question twice already, in `basis` and in `precedent.checked`: a check is only worth
    anything if a difference in it is a bug, which holds only when the server did it.

    Every branch returns a `code`, and there are five of them because there are five
    genuinely different things that can have happened. The first version had three, and
    review found two of them covering states that mean opposite things.
    """
    def outcome(code, what_it_means, **extra):
        return {"looked_for": looked_for, "query": query_basis, "basis": "computed",
                "found": [], "uncited": [], "code": code,
                "what_it_means": what_it_means, **extra}

    looked_for, tag = _DISCONFIRMING[verdict]
    # A verdict resting only on broken rules is not open to this argument, and §6.2 spent a
    # whole item saying so. "A campaign that broke the rule and performed anyway" is real
    # information — for whoever owns the rulebook (§12.1), not as a reason to reconsider the
    # breach. Voicing it as verdict-changing would undo 6.2 from the next field over.
    # §8.6: this exemption is right for a rule the customer WROTE and can edit — arguing with
    # it is a question about the rule, and the rulebook is where that belongs. It is backwards
    # for a rule the library INFERRED. A past campaign that did the opposite and performed
    # well is the single best evidence that the inference is wrong, and it is the only channel
    # by which a wrongly-graduated correction could ever be caught. Extending a rulebook-shaped
    # exemption to inferred content would have switched off the one check that watches it.
    rests_on_learned = any((f.get("precedent") or {}).get("correction_id")
                           for f in (findings or []))
    if (verdict != "approve" and by_class and set(by_class) == {"not_debatable"}
            and not rests_on_learned):
        return outcome(
            "verdict_rests_on_a_rule",
            "This verdict rests only on rules that were broken, which is not a matter of "
            "precedent. Whether some past campaign broke the same rule and did well is a "
            "question about the rule, not about this brief.")
    # This runs on EVERY save, so it has to survive a database that predates the columns it
    # reads. An upgraded v0.2.0 schema has no `tags`, and the first version of this crashed
    # every judgment on exactly the machine the review was gathered on.
    if not {"tags", "markets"} <= set(store._columns(conn, "campaigns")):
        return outcome(
            "nothing_to_check_against",
            "This library predates performance tagging, so nothing in it carries a measured "
            "verdict that could contradict this one. That is a fact about the library, NOT a "
            "check this judgment passed.")
    # Whether the library COULD argue back, asked before whether it did. An empty shelf
    # reported as a clean check turns an absence of evidence into a supporting vote.
    possible = [cid for cid in store.filter_campaign_ids(
        conn, tags=[{"value": tag, "source": "verified"}])
        if (store.get_campaign(conn, cid) or {}).get("record_type") != "reference"]
    if not possible:
        return outcome(
            "nothing_to_check_against",
            f"No campaign in the library is tagged {tag!r} with measured results behind it, "
            f"so there is no precedent that could contradict this verdict. That is a fact "
            f"about the library, NOT a check this judgment passed.",
            could_be_checked_if=_campaigns_that_could_be_tagged(conn))
    already = set(cited_ids or [])
    try:
        # The cited ones are excluded from the CANDIDATES, not filtered out afterwards.
        # Filtering after the slice meant three cited campaigns filled `top_k` and a fourth,
        # genuinely uncited and contradicting, was never looked at — and the answer came back
        # "nothing contradicted it".
        uncited_matches = _pole_search(
            conn, tag=tag, top_k=_MAX_DISCONFIRMING, exclude=already | {subject_campaign_id},
            **({"campaign_id": subject_campaign_id} if subject_campaign_id
               else {"text": subject_text}))
    except embedding.Unavailable as exc:
        # A fourth outcome, and the most likely real one: Ollama is not running. `Unavailable`
        # is a ValueError, so the first version swallowed it into "nothing came back" and
        # then said the verdict "was argued against and it held" — the exact collapse this
        # item was written to prevent, on the failure that actually happens.
        return outcome(
            "could_not_check",
            f"The search for contradicting precedent could not run ({exc}). Nothing was "
            f"checked — do not report this verdict as unchallenged.")
    found = [{"campaign_id": m["campaign_id"], "title": m["title"],
              "similarity": m["similarity"], "tag": tag,
              "why_it_contradicts": _why_it_contradicts(conn, m["campaign_id"], tag)}
             for m in uncited_matches]
    if not found:
        return outcome(
            "nothing_ranked",
            f"{len(possible)} campaign(s) are tagged {tag!r} with measured results, and none "
            f"the judgment had not already cited resembles this one closely enough to argue "
            f"with it.")
    return {
        "looked_for": looked_for, "query": query_basis, "basis": "computed",
        "found": found,
        # Rich rows, not bare ids: a model mirroring the shape of a result says what the shape
        # contains, and "camp_8701e96a" is not the thing most likely to change what the
        # marketer does. "Mexico launch, performed_well, CTR 3.4% vs 1.8% benchmark" is.
        "uncited": [row["campaign_id"] for row in found],
        "code": "contradicting_precedent",
        "what_it_means": (
            f"{len(found)} campaign(s) resembling this one are tagged {tag!r} with measured "
            f"results, and this judgment did not cite them. Say so."),
    }


def _both_poles(conn, *, evidence: list, text: Optional[str],
                campaign_id: Optional[str]) -> dict:
    """What worked and what did not, each retrieved in its OWN right (§6.4).

    The first version partitioned the evidence list that had already been retrieved — and
    review's objection is exact: on any real library the five nearest by similarity are five
    records nobody has tagged, both poles come back empty, and the reasoner has seen nothing
    contradictory. That is the state the review describes, dressed as a fix for it. A
    partition of a ranked list cannot reach past the ranking; only a filtered search can.

    So each pole is a search with the tag as a FILTER, which runs before ranking. `unknown`
    stays a partition of what was retrieved, because that is exactly what it is: the rest of
    the evidence, whose outcome nobody recorded.
    """
    ranked = {row["campaign_id"] for row in evidence}
    poles: dict = {}
    for pole, tag in (("worked", "performed_well"), ("did_not_work", "underperformed")):
        try:
            matches = _pole_search(conn, tag=tag, text=text, campaign_id=campaign_id,
                                   top_k=_MAX_DISCONFIRMING,
                                   exclude={campaign_id} if campaign_id else set())
        except embedding.Unavailable:
            matches = []
        poles[pole] = [{"campaign_id": m["campaign_id"], "title": m["title"],
                        "similarity": m["similarity"],
                        # §9.5 again, and this is the pole where it matters most: "this one
                        # worked" is the sentence a reasoner leans on hardest, and it is the
                        # one place the caveat was not stated. A `performed_well` campaign
                        # that drifted is evidence that SOMETHING worked and the brief may not
                        # have been it — reaching a reader here as a bare endorsement of the
                        # brief is the misreading §9.5 exists to prevent.
                        "execution": _execution_note(conn, m["campaign_id"]),
                        # §9.8, on the same pole and for the same reason §9.5 found its own
                        # caveat missing here: "this one worked" is the line a reasoner leans
                        # on hardest, and a result that ran through an earthquake is not a
                        # clean one.
                        **_confounded_note(conn, m["campaign_id"]),
                        # Whether the reasoner would have seen it anyway. A pole entry that is
                        # NOT in the ranked evidence is the one this search exists for.
                        "in_evidence": m["campaign_id"] in ranked}
                       for m in matches]
    measured = {row["campaign_id"] for pole in poles.values() for row in pole}
    poles["unknown"] = [{"campaign_id": row["campaign_id"], "title": row["title"],
                         "similarity": row["similarity"]}
                        for row in evidence if row["campaign_id"] not in measured]
    return poles


def _ran_as_briefed(conn, *, evidence: list, text: Optional[str],
                    campaign_id: Optional[str]) -> dict:
    """Precedent that actually ran the way it was written down (§9.5), retrieved in its own
    right.

    `_both_poles`' argument, one axis over. §9.5 attaches an `execution` note to every cited
    row, which tells a reader what the ranked evidence is worth — but it cannot tell them the
    library holds a faithfully-executed precedent the ranking did not reach. On a library where
    the five nearest all drifted, partitioning those five reports "nothing here ran as briefed",
    which is a statement about the ranking dressed as a statement about the library. Only a
    filtered pass can tell the two apart.

    Filtered on the STORED snapshot, not recomputed: §9.5's figure is a fact about the evidence
    as it stood, and a reader comparing this list against a citation's own `execution` must be
    reading the same number.
    """
    ranked = {row["campaign_id"] for row in evidence}
    try:
        matches = find_similar(conn, text=text, campaign_id=campaign_id,
                               top_k=_MAX_DISCONFIRMING * 4, full_detail=False)
    except (embedding.Unavailable, ValueError):
        matches = []
    faithful = []
    for match in matches:
        if match["campaign_id"] == campaign_id or match.get("record_type") == "reference":
            continue
        note = _execution_note(conn, match["campaign_id"])
        if not _reads_as_briefed(note):
            continue
        faithful.append({"campaign_id": match["campaign_id"], "title": match["title"],
                         "similarity": match["similarity"], "execution": note,
                         "in_evidence": match["campaign_id"] in ranked})
        if len(faithful) == _MAX_DISCONFIRMING:
            break
    return {
        "basis": "computed",
        "found": faithful,
        # Never silent on empty. "No precedent here has been checked against what it ran" and
        # "the precedent that was checked all drifted" are different facts about the library,
        # and both are different again from the ranking simply not reaching one.
        "what_it_means": (
            f"{len(faithful)} campaign(s) like this one have been compared against their own "
            f"brief and ran as briefed, so their outcomes are evidence about the BRIEF. "
            + ("" if all(f["in_evidence"] for f in faithful) else
               "Some are not in the ranked evidence above — they are here because they ran "
               "faithfully, not because they ranked. ")
            if faithful else
            "No campaign like this one has been compared against its own brief and found to "
            "have run as briefed. That is a fact about what has been CHECKED, not a finding "
            "about execution: an unchecked campaign is not a faithful one, and it is not a "
            "drifted one either."),
    }


def _why_it_contradicts(conn, campaign_id: str, tag: str) -> Optional[str]:
    """The measured line behind the tag, so the contradiction can be stated rather than
    pointed at. Without it the model can only name an id."""
    record = store.get_campaign(conn, campaign_id) or {}
    for metric in record.get("metrics") or []:
        if metric["metric_type"] == "actual" and metric.get("detail"):
            return _bounded(metric["detail"], "detail", 200)
    return None


def _campaigns_that_could_be_tagged(conn) -> list:
    """Which records would make the check possible, when it is not (§6.4 / §5.6's rule).

    Every other `cannot` in this codebase names the record that would lift it, because a
    capability statement nobody can act on is a disclaimer. This one said "nothing here has a
    measured verdict" and stopped — while the library may be full of concluded campaigns with
    real metrics that nobody has tagged. That is one `update_campaign` away.
    """
    measured = store.campaigns_with_actual_metrics(conn)
    waiting = []
    for cid in measured:
        record = store.get_campaign(conn, cid)
        if not record or record.get("record_type") == "reference":
            continue
        verified = {t.get("value") for t in (record.get("tags") or [])
                    if t.get("source") == "verified"}
        if not verified & {"performed_well", "underperformed"}:
            waiting.append({"campaign_id": cid, "title": record["title"]})
    return waiting[:_MAX_DISCONFIRMING]


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
    # Composed from everything STORED, not from the filtered view: a reader who asked for the
    # blocking items should not be handed a shorter exit condition as a side effect, which is
    # the same promise the class summary above makes.
    read_checklist = _exit_checklist(ev.get("verdict"), stored)
    return {
        "evaluation_id": ev["id"],
        "subject_title": ev["subject_title"],
        "verdict": ev["verdict"],
        "summary": ev["summary"],
        "approve_if": ev.get("approve_if"),
        **({"exit_checklist": read_checklist} if read_checklist else {}),
        "closest_precedent": ev.get("closest_precedent"),
        "by_class": by_class,
        "findings": findings,
        "improvements": [f for f in stored
                         if f.get("departure") == "possible_improvement"],
        # §6.4, on the read path. "Recorded so it can be audited afterwards" was true of the
        # table and false of the surface: without this the check lived for exactly one
        # response, which is the failure §2.4 fixed for the findings themselves.
        **({"disconfirming": (ev.get("evidence") or {})["disconfirming"]}
           if (ev.get("evidence") or {}).get("disconfirming") else {}),
        # §6.6, on the read path: a strength line that lives for one response cannot be what a
        # later reader weighs the judgment by, and weighing it later is the point.
        **({"evidence": ev["evidence"]} if ev.get("evidence") else {}),
        # §7.6 on the read path: a stamp nobody can read back cannot explain a disagreement,
        # which is the only thing it is for.
        **({"provenance": ev["provenance"]} if ev.get("provenance") else {}),
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


# D81: the stable code for each rule that can refuse a write, so a drift in WHICH rule is
# refusing is visible. A total says nothing — a rise in citation refusals and a rise in
# severity downgrades mean different things, and the quietest outcome of all is a real finding
# dropped because its citation would not verify, which today leaves no trace.
_REFUSAL_CODES = (
    ("is not a record in this library", "precedent_unresolved"),
    ("that quote is not in", "quote_not_found"),
    ("precedent needs a quote", "precedent_without_quote"),
    ("has to cite what it departs from", "citing_kind_without_precedent"),
    ("cannot approve with", "approve_with_blocking"),
    ("cannot carry `approve_if`", "approve_with_exit_condition"),
    ("needs `approve_if`", "revise_without_exit_condition"),
    ("needs a `fix`", "finding_without_fix"),
    ("kind is required", "finding_without_kind"),
    ("must say which way it departs", "departure_without_direction"),
    ("cannot be a note", "breach_as_note"),
    ("needs at least one finding above a note", "verdict_without_finding"),
)


def _count_refusal(conn, message: str) -> None:
    for needle, code in _REFUSAL_CODES:
        if needle in message:
            store.record_refusal(conn, code)
            return
    store.record_refusal(conn, "other")


def save_evaluation(conn, *args, **kwargs) -> dict:
    """`_save_evaluation`, counting what it refuses (§7.6 / D81).

    A wrapper rather than a `try` around each check, because there are twelve of them and the
    one that matters most is whichever one nobody thought to instrument.
    """
    try:
        return _save_evaluation(conn, *args, **kwargs)
    except ValueError as exc:
        _count_refusal(conn, str(exc))
        raise


# §7.8 / D5. Which computed facts become FINDINGS rather than staying facts, and what to call
# them. A fact is not a finding — "no date appears in this brief" becomes one when somebody
# says it is a problem — and until this existed only the model could make that step, which put
# a `judged` label on the most mechanical half of the output.
#
# Deliberately not every fact. `channels: partial` is the normal state of every brief, and a
# finding that fires on everything is one nobody reads. These three are the review's own
# examples of mechanical findings, and each is a thing that is simply absent or simply wrong.
_COMPUTED_FINDINGS = {
    "date_coverage": ("absent", "should_fix",
                      "No date appears anywhere in this brief"),
    "engagement_rate": ("absent", "should_fix",
                        "No creator profile carries an engagement rate"),
    "date_consistency": ("contradicted", "blocking",
                         "A statement about dates contradicts the calendar"),
    # §9.7: "that flag should have come from the system and been IMPOSSIBLE TO DROP". Landing
    # in `computed` buys §2.4's write protection — the model cannot forge one — and not this,
    # which is what the review was actually asking for: a finding appended after the model's
    # list, exempt from the caps, that an `approve` with no findings cannot make disappear.
    #
    # `should_fix`, never `blocking`. The instruction everywhere else in this item is "do not
    # turn it into a blocking finding on its own", and the server has to hold itself to the
    # rule it gives the model — overlapping a fixed date is the point of some campaigns.
    "calendar_clash": ("unaddressed", "should_fix",
                       "The launch window overlaps a fixed date the plan never names"),
    # §9.8, on §9.7's pattern and for the same reason. Every §9.8 signal was an INPUT — the
    # evidence row, the pole, the standing note — and none was an output the model had to
    # carry, so an `approve` leaning on a confounded number and never mentioning it was
    # accepted and left no trace. "Never quoted as clean evidence" was hoped for rather than
    # true.
    #
    # The finding is the SILENCE, exactly as §9.7 framed it: a verdict that says the number is
    # confounded has said what there was to say, and raising one anyway is the wallpaper this
    # item's whole confounding rule exists to avoid.
    "confounded_evidence": ("unaddressed", "should_fix",
                            "The verdict rests on an outcome that is not clean evidence"),
    # §11.5, the same machinery for the same reason: the sentence in the note is an
    # instruction, and this is the check. `should_fix` — a contested precedent is weaker
    # support, not a reason the brief cannot proceed.
    "contested_precedent": ("unaddressed", "should_fix",
                            "The verdict rests on a precedent people disagreed about"),
}


def _elide(text: str, limit: int) -> str:
    """Shorten the SERVER's own prose to fit a field. Never used on a caller's text, where
    exceeding a limit is something the caller should be told about rather than hidden."""
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "\u2026"


def _computed_findings(subject_text: Optional[str], computed: Optional[dict]) -> list:
    """The mechanical findings, raised by the server (§7.8 / D5).

    §2.4's premise — "a computed finding is identical for every user, so a difference there is
    a bug" — says nothing while every finding is written by a model. §7.1 established the
    facts; this is what turns the ones that are problems into findings, so §7.7 measures a
    model's consistency at reasoning rather than its consistency at reading a regex's output.

    They never change the verdict. The server establishes facts and the reasoner judges; a
    server finding that flipped an approve would be the product overruling the reasoner on the
    strength of a regex, and §6.4 already settled that the server ARGUES with a verdict rather
    than replacing it.
    """
    if computed is None:
        if not subject_text:
            return []
        computed = facts.compute(subject_text)
    raised = []
    for code, (status, severity, headline) in _COMPUTED_FINDINGS.items():
        fact = computed.get(code) or {}
        if code == "confounded_evidence":
            if fact.get("status") != "present" or not fact.get("unaddressed"):
                continue
            headline = (f"The verdict rests on {', '.join(fact['unaddressed'])}, whose result "
                        f"is confounded, without saying so")
        elif code == "contested_precedent":
            if fact.get("status") != "present" or not fact.get("unaddressed"):
                continue
            headline = (f"The verdict rests on {', '.join(fact['unaddressed'])}, which two "
                        f"people recorded different views about, without saying so")
        elif code == "calendar_clash":
            # A clash is not a finding; the SILENCE is. "Mexico's own deck flagged that it
            # clashed with the World Cup and then never addressed it" — a plan that names what
            # it runs into has said what there was to say, and raising one anyway is the
            # nuisance that teaches a reader to skip server findings.
            if fact.get("status") != "present" or not fact.get("unaddressed"):
                continue
            headline = (f"The launch window overlaps "
                        f"{', '.join(fact['unaddressed'])}, which the plan never names")
        elif fact.get("status") != status:
            continue
        raised.append({
            "severity": severity,
            "kind": "missing_information" if code != "date_consistency"
                    else "internal_contradiction",
            "departure": None,
            "basis": "computed",
            "category": code,
            "finding": headline,
            "repeats": None,
            # The evidence the check read, for the same reason §7.1 attaches it to a fact: a
            # finding that cannot show what it looked at is an assertion, and the server's
            # assertions carry more weight than a model's.
            # Truncated rather than refused. `_bounded` RAISES past the limit, which is right
            # for a model's own text — a finding somebody wrote over the cap is a finding they
            # should shorten — and wrong for the server's, where it would turn a long computed
            # sentence into a failed write of the whole judgment.
            "detail": _elide(fact.get("what_it_means", "") + (
                "  Read from: " + " / ".join(fact.get("evidence") or [])
                if fact.get("evidence") else ""), _MAX_DETAIL),
            "precedent": None,
            "fix": None,
        })
    return raised


def _save_evaluation(conn, *, subject_title: str, verdict: str, summary: str,
                    findings: Optional[list] = None, resolved: Optional[list] = None,
                    closest_precedent: Optional[dict] = None,
                    approve_if: Optional[str] = None, evidence: Optional[dict] = None,
                    provenance: Optional[dict] = None,
                    campaign_id: Optional[str] = None, cited_ids: Optional[list] = None,
                    predictions: Optional[dict] = None, retrieval: Optional[str] = None,
                    model_id: Optional[str] = None, subject_text: Optional[str] = None,
                    trusted: bool = False, markets: Optional[list] = None) -> dict:
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
    # Looked up once, here, because three separate things below need it: the window check,
    # the closest precedent, and which subject the disconfirming search should run against.
    receipt = store.get_retrieval(conn, retrieval) if retrieval else None
    # §6.4 is a server fact for the same reason `basis` and `precedent.checked` are: a check
    # the model reports is a claim, and a model asked to find evidence against a verdict it
    # has already reached is marking its own homework.
    # D3/§6.6: the whole `evidence` block is the server's. It is a count of what this
    # judgment cites and what those records carry — a model asserting "this rests on five
    # concluded campaigns" is making a claim, and §7.8's premise (a difference in a computed
    # thing is a bug) holds only when the server counted.
    if provenance:
        # D4, and the same rule as `basis`, `checked` and `evidence`: a provenance record the
        # model writes is a record of what the model SAYS produced it. `model_id` is the one
        # field the server cannot observe, so it has its own argument.
        raise ValueError("`provenance` is written by the server, not by you: it records what "
                         "produced this judgment, and a record of what you say produced it "
                         "explains nothing when two judgments disagree. Pass `model_id` if "
                         "you know which model you are.")
    if evidence:
        raise ValueError("`evidence` is written by the server, not by you: it counts what "
                         "this judgment cites and what those records actually carry, and "
                         "records the search for precedent contradicting your verdict. A "
                         "measure of your own evidence, reported by you, is not a measure.")
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
        # §8.6: a guardrail breach rests on a RULE, and the library now holds rules in two
        # places — the rulebook, and the standing corrections it has learned and had confirmed.
        # Either is a rule somebody set; neither is "a campaign that did it that way".
        wanted = (("rule_id", "correction_id") if kind == "guardrail_breach"
                  else ("campaign_id",))
        naming = " or a ".join(wanted)
        if kind in _CITING_KINDS and not precedent:
            raise ValueError(
                f"{where}a {kind} has to cite what it departs from: precedent with a "
                f"{naming} and a quote. Without one it is an opinion in the vocabulary of a "
                f"citation.")
        # The SLOT has to match the kind, or the rule_id check is only half a check: the
        # comment on `_verify_quote` says a rule must not be anchored to "somebody's Q3
        # deck", and without this a guardrail_breach could cite exactly that by using the
        # campaign_id slot instead.
        if kind in _CITING_KINDS and precedent and not any(w in precedent for w in wanted):
            raise ValueError(
                f"{where}a {kind} cites a {naming}, and this precedent does not have one. A "
                f"rule — in your guidelines or as a standing correction — and a campaign that "
                f"did it differently are not interchangeable: one is not debatable and the "
                f"other invites a rationale.")
        basis = finding.get("basis") or "judged"
        if basis not in _BASES:
            raise ValueError(f"{where}basis must be one of {list(_BASES)}, got {basis!r}")
        # §7.8 rests on "a computed finding is identical for every user, so a difference
        # there is a bug" — which only holds if the SERVER computed it. A model that read a
        # missing date did not compute it, and letting it say so would let an opinion
        # inherit the authority of a mechanical check.
        if basis in ("computed", "heuristic") and not trusted:
            raise ValueError(
                f"{where}only the server sets basis {basis!r}: {BASIS_MEANING[basis]} A "
                f"finding you reached yourself is 'judged', however certain it is — claiming "
                f"a provenance is not the same as having one.")
        # §6.5 / D2: a finding above a note is a claim that something must change, and
        # without a `fix` the reader has the complaint and not the remedy. It is also what
        # the exit checklist is composed from, so a missing one leaves a hole in the list
        # somebody is meant to tick off.
        fix = _bounded(finding.get("fix"), "'fix'", _MAX_FIX, where=where)
        if severity in ("blocking", "should_fix") and not fix:
            raise ValueError(
                f"{where}a {severity!r} finding needs a `fix` — one line saying what to "
                f"change. A problem worth acting on that does not say what the action is "
                f"leaves the reader with the complaint and not the remedy, and it leaves a "
                f"hole in the exit condition, which is built from these.")
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
            "fix": fix,
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

    # §6.5. The review: "Every revise should carry its own exit condition. Today the reader
    # infers it from the list of problems, which is not the same thing and is not checkable."
    if verdict == "revise" and not approve_if:
        raise ValueError(
            "a 'revise' needs `approve_if`: the specific change that would make this an "
            "approve, stated so somebody could check it against the next version. The list "
            "of findings is not the same thing — three findings may need two changes, and "
            "one may need three.")
    if verdict == "approve" and approve_if:
        raise ValueError(
            "an 'approve' cannot carry `approve_if` — there is nothing to exit. A condition "
            "attached to an approval is a reservation the verdict does not admit to; if "
            "something still has to change, the verdict is 'revise'.")
    # The one part of "testable" a validator can genuinely enforce. If there IS a set of
    # changes that converts this to approve, then that is what 'revise' means — and letting a
    # reject carry one lets the harsher word be used with the softer meaning.
    if verdict == "reject" and approve_if:
        raise ValueError(
            "a 'reject' cannot carry `approve_if`. If naming a set of changes would make "
            "this approvable, the verdict is 'revise' — that is the difference between the "
            "two words. Reject is for a brief that cannot get there from here.")

    # Most severe first, always: the order is part of the contract, so two evaluations of
    # the same brief can be compared without re-reading them. The sort is stable, so two
    # findings of equal severity keep the order they were written in — a model that ordered
    # them deliberately is not second-guessed.
    # The server's own findings join the list before ids are assigned, so they are addressable
    # like any other — a later version can resolve one by id. They are appended rather than
    # merged into the caps: the model's twelve are its own, and refusing a mechanical finding
    # because the model filled the list would hide the half nobody is guessing at.
    established = (facts.for_campaign(conn, campaign_id) if campaign_id
                   else (facts.compute(subject_text) if subject_text else {}))
    # §9.7, recomputed here rather than trusted from `prepare_evaluation`: a check is only
    # worth anything if a difference in it is a bug, which holds only when the server did it —
    # the same reasoning §6.4 reached about who runs the disconfirming search.
    clash = _calendar_clash(conn, campaign_id=campaign_id, proposal_text=subject_text,
                            starts_on=None, ends_on=None, markets=list(markets or []))
    established["calendar_clash"] = clash
    # §9.8's half of the same machinery: does the verdict lean on a confounded outcome and
    # never mention it? Computed over what the model actually WROTE, which is the only place
    # the silence can be seen.
    written = " ".join(filter(None, [summary, approve_if] + [
        " ".join(str(f.get(k) or "") for k in ("finding", "detail", "fix"))
        for f in (findings or [])]))
    established["confounded_evidence"] = _confounded_silence(conn, cited_ids, written)
    # §11.5's half, on the same written text and for the same reason.
    established["contested_precedent"] = _contested_silence(conn, cited_ids, written)
    cleaned += _computed_findings(subject_text, established or None)
    # Counted AFTER the server's own findings join the list — counting before it meant the
    # one figure that says which half of the output is the model's did not include the other
    # half at all.
    counts_by_basis: dict = {}
    for finding in cleaned:
        counts_by_basis[finding["basis"]] = counts_by_basis.get(finding["basis"], 0) + 1
    cleaned.sort(key=lambda f: _SEVERITIES.index(f["severity"]))

    # Identify the findings so a later version can say which one it closed, and so two
    # evaluations of the same brief can be compared by reference rather than by string
    # match. Assigned after the sort, so an id also reads as a position.
    # §5.3: the single thing that would most change THIS verdict, computed here rather than
    # carried from prepare_evaluation, and stored so get_evaluation and §7.6's stamp can
    # recover it.
    missing = missing_input_for_citations(conn, cited_ids)

    by_class: dict = {}
    for finding in cleaned:
        klass = _CLASSES[finding["kind"]]
        by_class[klass] = by_class.get(klass, 0) + 1

    # §6.4, and recorded rather than only returned: a check that lives for one response is a
    # check nobody can audit, and the review asked for this precisely so overconfidence stays
    # visible afterwards.
    #
    # The QUERY is the subject where the subject is knowable. The first version searched on
    # the judgment's own prose — summary plus finding lines — and then said the results
    # "resemble this one", which was a claim about the brief made from a search over the
    # complaint about it. A verdict reading "budget is thin" about a Mexico-shaped brief does
    # not retrieve Mexico. When the subject is a stored record the record's own text is used;
    # when it is not, `query` says so and the wording does not overreach. Making the brief
    # knowable at save time in every case is 7.2's job (D86).
    # D86: the receipt records which subject the evidence was gathered for, so the check can
    # use it even when the subject is not a stored record. Without it this searched on the
    # judgment's own prose and then called the results "precedent resembling this one" — a
    # claim about the brief made from a search over the complaint about it.
    subject_record = store.get_campaign(conn, campaign_id) if campaign_id else None
    if subject_record:
        subject_text, query_basis = None, "subject_record"
    elif receipt and receipt.get("campaign_id"):
        subject_record = store.get_campaign(conn, receipt["campaign_id"])
        subject_text, query_basis = None, "retrieval_receipt"
    elif receipt:
        # No record, but the receipt holds the subject the window was built from.
        subject_text, query_basis = receipt["subject_title"], "retrieval_receipt"
    else:
        subject_text = "\n".join([subject_title, summary])
        query_basis = "judgment_text"
    disconfirming = _disconfirming_search(
        conn, verdict=verdict,
        subject_text=subject_text if subject_text is not None else "",
        query_basis=query_basis, cited_ids=cited_ids, by_class=by_class, findings=cleaned,
        subject_campaign_id=(campaign_id if campaign_id and subject_record
                             else (receipt or {}).get("campaign_id")))
    window = _window_check(conn, retrieval, cited_ids, subject_title=subject_title,
                           campaign_id=campaign_id)
    # D8/§7.2: whichever record the server ranked first IS the closest precedent, and it was
    # previously whatever the model asserted — a claim about which record is nearest, made by
    # the party that did not do the ranking. With a receipt the server knows; without one it
    # cannot compute what it did not retrieve, so the model may still name one and it is not
    # marked computed.
    live = [c for c in (receipt["campaign_ids"] if receipt else [])
            if store.get_campaign(conn, c) is not None
            and c not in store.get_superseded_campaign_ids(conn)]
    if receipt and live:
        if closest_precedent:
            raise ValueError(
                "closest_precedent is the server's when you pass a `retrieval` receipt: it "
                "is the top of the window the server ranked, and a claim about which record "
                "is nearest, made by the party that did not do the ranking, is not a "
                "measure. Send the receipt and it is filled in.")
        # The first record from the window that a search would still return today. Taking
        # `[0]` unconditionally let the server assert as a computed fact a precedent that had
        # since been deleted or superseded — while `unresolved_citations` in the same evidence
        # block named the same id as missing.
        closest_precedent = {"campaign_id": live[0], "basis": "computed",
                             "from": "retrieval_window"}
    else:
        closest_precedent = _clean_closest_precedent(conn, closest_precedent)
    # Once. It was called twice — one to test the truthiness and one to use it — and each
    # call walks every cited campaign.
    confounded_now = _confounded_at_save(conn, cited_ids)
    evidence = {
        **_evidence_strength(conn, cited_ids=cited_ids,
                             text="\n".join([subject_title, summary])),
        "disconfirming": disconfirming,
        **window,
        # §9.7: what the calendar said at the moment of the verdict. §9.9 reconciles predicted
        # against actual against CONTEXT, and it cannot do that against a check nobody stored —
        # the same reasoning as §9.5's `execution_at_save`.
        **({"confounded": confounded_now} if confounded_now else {}),
        **({"calendar_clash": {k: clash[k] for k in
                               ("status", "window", "unaddressed", "searched",
                                "markets_not_covered", "what_it_means")}}
           if clash.get("status") not in (None, "nothing_to_check") else {}),
        # §9.5's figure, STAMPED — the live `execution_drift` row is rewritten whenever the
        # answer changes (results arriving, photographs arriving, a classification being
        # made), which is right for the current reading and wrong for a saved one. A verdict
        # written when a cited campaign read `never_checked` is read back beside a row that
        # now says `drifted`, with nothing saying the judgment never saw it. Same reasoning as
        # §9.4's `outcome_known`: record what the judgment could see at the moment it was made.
        **_execution_at_save(conn, cited_ids),
    }
    if missing:
        # §5.3 lives alongside it, and the two are deliberately different questions: this says
        # what the judgment RESTS ON, that says what would most improve it. One describes, the
        # other asks.
        evidence["most_valuable_missing_input"] = missing

    eid = store.next_evaluation_id()
    for n, finding in enumerate(cleaned, start=1):
        finding["id"] = f"{eid}#{n}"
    # After the ids exist, because each item points at the finding it came from — that is
    # what lets the next version's judgment close them by id rather than by wording.
    exit_checklist = _exit_checklist(verdict, cleaned)

    provenance = _stamp(conn, receipt=receipt, model_id=model_id)
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
    return {
        "evaluation_id": eid,
        "verdict": verdict,
        "summary": summary,
        "counts": counts,
        # §7.8: which half of the output is the model's. §7.7 measures finding recall against
        # what the product raised, and without this the figure would improve every time a
        # regex fired and nobody could tell why.
        "counts_by_basis": counts_by_basis,
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
        # Absent rather than empty on an approve: an empty list beside an approval reads as
        # "nothing left to do, we checked", when the truth is there was never a list.
        **({"exit_checklist": exit_checklist} if exit_checklist else {}),
        "most_valuable_missing_input": missing,
        # §6.6: two judgments of very different worth used to be presented identically.
        "evidence": {k: v for k, v in evidence.items() if k != "most_valuable_missing_input"},
        "disconfirming": disconfirming,
        # §5.2: the three things anyone actually does after a judgment, prefilled. The
        # supersession offer only appears when there IS an earlier version — an approval of
        # a new brief supersedes nothing, and an offer that is always there stops being read.
        "next_actions": actions.trim(
            actions.after_evaluation(
                subject_title=subject_title, evaluation_id=eid, verdict=verdict,
                campaign_id=campaign_id)
            # §10.2/D84. An `unexplained` departure says in writing "the brief does not say
            # whether this is deliberate" — and the marketer very often answers on the spot,
            # in passing. Until now there was nowhere for that to go, so the answer was lost
            # and the same question came back on the next version.
            #
            # Second, not first: `after_evaluation` offers to store a judged brief that is
            # not yet a record, which is the commonest case and is about whether the judgment
            # can ever be checked at all. The comment here used to claim FIRST and the code
            # never gave it that, which is the doc-code drift this project guards against.
            + _offer_to_settle(eid, cleaned)
            # §10.6: a judgment is one of the two moments the review names.
            # `apart_from`: a write announcing itself is the always-present offer that
            # teaches a reader to skip the list.
            + actions.offer_the_queue(feedback.waiting(conn, apart_from=campaign_id))),
        "note": (_how_to_say_it(by_class, cleaned)
                 + _say_the_evidence_strength(evidence)
                 + _say_the_disconfirming_check(disconfirming)),
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

    # D27: the rulebook is a shipped payload like the CLIP weights, and it fails the same
    # way — the server answers, the database is fine, and every judgment it makes is missing
    # its rules. Invisible here, an install broken in the one way §12.1 cares about passed the
    # installers' own post-install gate.
    #
    # Inside a `try` for the reason the whole function is: a diagnostic that crashes on the
    # thing it is diagnosing leaves the administrator where they started.
    try:
        loaded = rulebook.load()
        components["rulebook"] = {
            "ok": True,
            # BOTH files, always. It named the install path even when the rules came from the
            # overlay, so an administrator reading "0 rule(s)" with their own rulebook open in
            # front of them had no lead at all.
            "detail": (f"{loaded['version']}: {len(loaded['rules'])} rule(s). "
                       f"Product: {rulebook._bundled()}. "
                       + (f"Yours: {rulebook.overlay_path()} ({loaded['overlay_version']})."
                          if loaded["overlay_version"] else
                          f"You have not written one: {rulebook.overlay_path()}.")),
        }
        if rulebook.overlay_path().exists() and not loaded["overlay_version"]:
            # Written since the server started. The rulebook is read once per process — which
            # is right, because a judgment must not depend on when in a session it was made —
            # but the overlay is the file the product invites the customer to create, under a
            # connector that stays up for days. The stamp stays honest; what was missing is
            # that nothing told them their file was not in force.
            components["rulebook"]["degraded"] = (
                f"There is a rulebook at {rulebook.overlay_path()} that is NOT in force: it "
                f"was written or changed after this server started, and the rules are read "
                f"once so that a judgment cannot depend on when in a session it was made. "
                f"Restart the server to apply it.")
        if not rulebook.is_the_editable_copy():
            # Working, so not a failure — but the file in force is the one collected inside
            # the bundle rather than the one beside the executable. A silent fallback is its
            # own defect: an administrator editing the file they can SEE would change nothing
            # and be told nothing, which looks like it worked and is worse than either half.
            components["rulebook"]["degraded"] = (
                f"This is the copy packaged inside the application, not the one beside the "
                f"executable — so there is no {rulebook.BUNDLED_NAME} anybody can edit. "
                f"Copy it to {config.app_dir()} to change the rules, or re-run the "
                f"installer.")
    except Exception as exc:
        components["rulebook"] = {
            "ok": False, "code": "rulebook_unreadable",
            "detail": str(exc),
            "affects": "Every judgment. The rules that are supposed to apply to each brief "
                       "are not being applied, and nothing in a judgment says so.",
            # WHICH file, because there are two and only one of them is the customer's.
            # This said "re-run the installer, which restores the shipped copy" — false for an
            # overlay, which lives in the data directory no installer touches, and a remedy
            # that cannot work costs somebody a reinstall before they look further.
            "remedy": (f"Fix {rulebook.overlay_path()} and restart the server — the message "
                       f"above names the line."
                       if rulebook.overlay_path().exists() else
                       f"Restore {rulebook._bundled()} and restart the server. Re-running the "
                       f"installer replaces the shipped copy."),
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
#   6  campaigns judged on a brief with nothing showing what actually ran (§9.1)
# How many names a gap may list before it is a paragraph rather than a sentence.
_MAX_NAMED = 5

_GAP_RANK = {
    "library_is_empty": 1,
    "few_verified_outcomes": 2,
    "market_without_outcomes": 3,
    "partly_indexed": 4,
    # §9.1's state: "a library that learns from briefs while measuring executions is learning
    # from the wrong document, and has no way to notice." Ranked below the others because a
    # brief with no measured outcome is a bigger hole than a measured outcome nobody has
    # checked against what ran — but above nothing, because it is the difference between
    # "this campaign worked" and "something worked, and we do not know if it was this".
    "execution_never_checked": 5,
    # Below execution drift: not knowing what ran is worse than not knowing what else was
    # going on while it ran, and this one is cheap to close (two dates) where that one needs
    # photographs.
    "no_window": 4,
    # Below the gaps about MISSING data, because this one is about data that is all present
    # and simply has not been looked at — and the looking is a few minutes rather than a
    # workbook somebody has to go and find.
    "judgments_never_reconciled": 3,
    # §12.4/D51. LOWER than every gap about missing evidence (the number is a position, and a
    # smaller one ranks higher): the deck is usually already on file and the remedy is one
    # call, where `few_verified_outcomes` is the hole this product was built around. Above
    # nothing, though — a client's recorded objection is often the most useful precedent the
    # library could have, and these records have none. Recorded since §2.5 and NOT reported
    # until D39 gave it an action that did not create a duplicate record.
    "commentary_never_read": 6,
}


# Which computed facts are worth reporting as never-recorded across a whole market (D49).
# Not every check: `date_consistency` has no "never" reading, and `channels` is partial by
# design for almost every brief.
_FIELD_GAPS = ("budget", "date_coverage", "engagement_rate")
# Below this a market is not a pattern. One record missing a budget is a fact about that
# record; calling it a coverage gap would turn every new market into a complaint on the day
# it is added.
_MIN_FOR_A_PATTERN = 2


def _fields_never_recorded(conn, campaigns: list) -> list:
    """Markets where no record has ever carried a field (D49, §7.1).

    "No LATAM store launch has ever carried a budget" is one of the review's own three
    examples of a gap, and the only one about an absent FIELD inside records rather than an
    absent record. It needed §7.1 first, because until then nothing could detect a budget.

    `never` is the claim, so one record carrying the field makes it false — reporting the
    market anyway would be the overstatement this whole surface is written against.
    """
    by_market: dict = {}
    for campaign in campaigns:
        by_market.setdefault(campaign.get("market") or campaign.get("region"),
                             []).append(campaign)
    found = []
    for market, rows in sorted(by_market.items(), key=lambda kv: kv[0] or ""):
        if not market or len(rows) < _MIN_FOR_A_PATTERN:
            continue
        computed = [facts.for_campaign(conn, c["id"]) for c in rows]
        for field in _FIELD_GAPS:
            statuses = [c.get(field, {}).get("status") for c in computed]
            # "not_applicable" is not a miss — a brief with no creators is not missing their
            # engagement rates, and counting it would make the gap unclosable.
            #
            # D94: neither is "unchecked", and for a sharper reason. It is not evidence the
            # field is absent, so it cannot support the claim; and it is not evidence the
            # field is present, so it must not REFUTE the claim either. Left in, one
            # non-English deck falsified the `all()` and the whole gap disappeared — the
            # customer had been told about it for weeks, uploaded one Spanish deck, and the
            # product stopped saying it with nothing to explain why. That is this phase's own
            # defect with the sign flipped: an unknown must not be reported as a finding, and
            # it must not silently cancel one that is real.
            unreadable = sum(1 for s in statuses if s == "unchecked")
            relevant = [s for s in statuses
                        if s not in ("not_applicable", "unchecked")]
            if relevant and all(s == "absent" for s in relevant):
                found.append({
                    "code": "field_never_recorded",
                    "field": field,
                    "market": market,
                    "campaigns": len(relevant),
                    # Said, not hidden: the count is a claim about the records this library
                    # could READ, and a reader who is not told that will take it for a claim
                    # about the market.
                    "unreadable": unreadable,
                    "what": f"No campaign in {market} has ever recorded a "
                            f"{field.replace('_', ' ')} "
                            f"({len(relevant)} record(s) checked, all missing it"
                            + (f"; {unreadable} more could not be read — not in English"
                               if unreadable else "") + ").",
                    "why_it_matters": "A field missing from every record in a market is a gap "
                                      "no single upload reveals — every judgment there is "
                                      "made without it and nothing says so.",
                    "basis": "computed",
                })
    return found


def _expected_inputs_never_supplied(conn) -> list:
    """Inputs the rulebook declares a brief must carry, that nothing on file carries (D50).

    Reported per LIBRARY rather than per record, like the other field gaps beside it: "no
    brief has ever carried a KPI workbook" is the finding, and firing it once per campaign
    would bury every other gap under one fact repeated.

    `looks_like` is what makes it closable. Without the words that would show the input HAD
    arrived, the gap can only be reported forever — a gap nobody can close is one everybody
    learns to ignore, which is §8.2's own lesson about re-asking a declined question.
    """
    import rulebook

    try:
        declared = rulebook.expects()
    except ValueError:
        # A broken rulebook is `health_check`'s finding, loudly. It is not this function's,
        # and raising here would take out `gaps()` — a report about the library — over a
        # configuration file.
        return []
    if not declared:
        return []

    superseded = store.get_superseded_campaign_ids(conn)
    records = [c for c in store.list_campaigns(conn)
               if c.get("record_type") != "reference" and c["id"] not in superseded]
    if not records:
        return []

    found = []
    for expected in declared:
        if not expected["looks_like"]:
            continue
        carrying = 0
        for record in records:
            # The BODY only, never the commentary — §7.1's rule, and it is the same reason
            # here as there: a reviewer's note asking "where is the KPI workbook?" is not the
            # workbook, and counting it would close the gap with the complaint about it.
            on_file = store.text_on_file(conn, record["id"]) or {}
            text = " ".join(str(part) for part in (on_file.get("body") or [])).lower()
            if any(word in text for word in expected["looks_like"]):
                carrying += 1
        if carrying:
            continue
        found.append({
            "code": "expected_input_never_supplied",
            "expectation": expected["id"],
            "field": expected["input"],
            "campaigns": len(records),
            "what": f"Your rulebook says a brief should carry {expected['input']}, and no "
                    f"record on file mentions one ({len(records)} checked).",
            "what_it_means": f"{expected['input']} — {expected['why']} This is YOUR rule, "
                             f"from rulebook {rulebook.version()}, reported back rather than "
                             f"invented here.",
            "why_it_matters": expected["why"],
            "basis": "computed",
            "evidence": f"searched every brief's body text for "
                        f"{', '.join(repr(w) for w in expected['looks_like'])}",
        })
    return found


def gaps(conn) -> dict:
    """What this library is missing, ranked, with what would close each one.

    The counterpart to `most_valuable_missing_input` on a judgment: this is about the
    LIBRARY, that is about one verdict, and they routinely disagree — a library that is 90%
    measured can still produce a judgment resting entirely on the unmeasured tenth.

    §10.6/D76 folded in the first-steps path, which `readiness` also shows. It is the same
    ordered list from the same function, not a second copy — and it is empty once the path
    is walked, because guidance that never stops appearing is guidance nobody reads.
    """
    ranked = _ranked_gaps(conn)
    # Appended after ranking, not among the ranked gaps: these are patterns across a market
    # rather than one missing thing, and they carry no offer, because closing one means
    # editing several records. Ranking an unofferable gap above an offerable one would put
    # the thing nobody can act on first.
    superseded = store.get_superseded_campaign_ids(conn)
    field_gaps = _fields_never_recorded(conn, [
        c for c in store.list_campaigns(conn)
        if c.get("record_type") != "reference" and c["id"] not in superseded])
    # Carrying `order` too, past every ranked gap. They are appended rather than sorted in,
    # but a list where some rows have the key the list is ordered on and some do not is one
    # nobody can check the order of — and "appended last" is a position, so it should say so
    # in the same field as every other position rather than only by where it sits.
    # D50: an input the RULEBOOK declares a brief must carry, which no brief on file carries.
    # The review's third gap case, and it needed the rulebook to exist: an expectation nobody
    # wrote down cannot be reported as missing without the product inventing a requirement on
    # the customer's behalf. The product declares none, so this is empty until a customer
    # declares one in their overlay — which is the point. It is their rule, reported back.
    field_gaps += _expected_inputs_never_supplied(conn)
    last = max((g["order"] for g in ranked["gaps"]), default=0) + 1
    for gap in field_gaps:
        gap.update({"order": last, "rank": last, "affects": gap["campaigns"],
                    "share": 0.0, "outweighs_its_kind": False, "rank_basis": "computed"})
    ranked["gaps"] += field_gaps
    ranked["shortest_path"] = first_steps(conn)
    return ranked


def top_gap(conn) -> Optional[str]:
    """The code of the gap worth fixing first, or None (§10.6/D54).

    Separate from `gaps()` because it has to be cheap enough to call on every upload:
    `_fields_never_recorded` walks `facts.for_campaign` per record and cannot change
    `most_valuable` — it is appended after the ranking — so the expensive half is skipped.
    """
    return _ranked_gaps(conn)["most_valuable"]


_TOP_GAP_KEY = "top_gap"


def _gap_moved(conn) -> Optional[str]:
    """Has the library's worst problem changed since anybody was told? (§10.6/D54.)

    The second half of D54, and the harder half. "Offer the ranking at session start" is a
    condition a stateless call can evaluate; "offer it when an upload changes the top gap" is
    not — `changed` needs the previous answer, so one is remembered.

    Reads only. The first version also WROTE the new value here, before the offer it enables
    had reached `trim` — and the offer is appended last, so on any upload with three other
    things to say it was dropped AND the change was recorded as already announced. It
    therefore fired ZERO times per change on exactly the uploads that matter most: a v2
    replacing a judged record, or a concluded campaign with no results and no dates. The
    docstring claimed "once per change"; it was none. `_the_gap_was_announced` is now called
    by the caller, after the offer survives.

    A library with nothing remembered — a fresh install, or a database from before this
    existed — reads as moved, and it has: the ranking went from meaning nothing to meaning
    something. Losing the row costs one redundant offer, never a wrong answer.
    """
    now = top_gap(conn)
    # `library_is_empty` cannot be the NEW top gap after an upload, but it can be the
    # remembered one, and moving off it is the most meaningful move there is.
    return now if now and now != store.get_state(conn, _TOP_GAP_KEY) else None


def _the_gap_was_announced(conn, offers: list, moved: Optional[str]) -> None:
    """Remember the top gap only once somebody has actually been shown it (§10.6/D54).

    The whole of the fix for the burned offer: `trim` is the last word on what a user sees,
    so it is the only place that can say whether the announcement happened.
    """
    if moved and any(offer["tool"] == "gaps" for offer in offers):
        store.set_state(conn, _TOP_GAP_KEY, moved)


# ── answering back (§10.2, D84 / D59 / D53) ────────────────────────────────
#
# The four rows the tracker points at 10.2 are one thing: a place for a person's answer to
# go. This product asks several questions on surfaces with no return path — "is this
# departure deliberate?", "did you fix that?", "is this gap ever going to close?" — and the
# reply goes into a chat window and dies.
#
# That is worse than not asking, twice over. The question comes back, so the product looks
# like it is not listening, because it is not. And the UN-answer is stored: `departure:
# unexplained` is a fact every later judgment reads, and it is false the moment somebody
# explains it. A question nobody can answer decays into a wrong answer.

# What a person can say about a stored finding. Three, and the distinction between the first
# two is the one D84 insists on in writing: `resolved` "records 'we changed it', not 'we kept
# it, and here is why'". A library that cannot tell a brief that was corrected from one that
# was defended has lost what the whole correction loop reasons from.
FINDING_ANSWERS = {
    "fixed": "The problem was real and we addressed it.",
    "deliberate": "We kept it on purpose, and here is why.",
    # `not_applicable` used to carry both of these, and they are not the same answer. "That
    # rule does not apply to this brief" accepts the finding and disputes its scope; "you
    # have misread the brief" disputes the finding itself. Collapsing them loses the
    # disagreement — and a disagreement with a judgment is the single most valuable thing
    # this library can be told, because it is the only signal that the product got something
    # wrong. The whole review that started this began with "impossible to argue with".
    "does_not_apply": "The point is right in general and does not apply to this brief.",
    "misread": "The finding is wrong about what the brief says.",
    # The way back, which `GAP_ANSWERS` had from the start and this did not. A finding
    # answered wrongly could be re-answered but never returned to unanswered, so the first
    # mis-click was permanent — and `_offer_to_settle` was prefilling `deliberate`, which
    # made a wrong first answer likely rather than hypothetical. The whole point of a
    # person's answer is that a person can change it.
    "open": "Never mind — treat it as unanswered again.",
}

# How each answer reads when `diff_campaigns` renders it. `misread` is deliberately blunt:
# a person saying the judgment was wrong is the most valuable thing this library gets told,
# and softening it into "not applicable" is how a product stops hearing that it is wrong.
_SETTLED_AS = {
    "fixed": "addressed",
    "does_not_apply": "right in general but not applicable to this brief",
    "misread": "wrong about what the brief says",
}

# What a person can say about a gap they cannot close. `known_not_yet` stays ranked, because
# it is still the thing to fix; `not_applicable` leaves the ranking, because a market whose
# agency no longer exists will never have its results and reporting it forever is the
# permanent complaint this surface is written against. One state for both would either nag
# somebody who has answered or silence a gap that is real.
GAP_ANSWERS = {
    "known_not_yet": "Known, and not being done yet.",
    "not_applicable": "This will never be true here.",
    "open": "Never mind — treat it as outstanding again.",
}


# Past this, an answer is old enough that nobody can reasonably say it still reflects what
# they think. Not a deadline and not a scheduler — this product has no scheduler and says so
# in `after_evaluation` — just the point at which the log stops letting the age go unsaid.
_STALE_ANSWER_DAYS = 180


def answers(conn, *, said_by: Optional[str] = None) -> dict:
    """Every answer a person has given this library, newest first (§10.2).

    The read path the write paths needed. §10.2 gave this product somewhere for a person's
    answer to go and nothing to read them back with — and for a product whose whole pitch is
    that its judgments can be ARGUED WITH, the record of where somebody argued and won is the
    most valuable thing it holds. It was write-only, which is D116's shape one level down: a
    team lead could not ask "what have we marked deliberate", "who set aside which gaps" or
    "show me everything Ana answered", so nobody reviewed the overrides, so a wrong one was
    permanent in practice even though `open` makes it reversible in principle.

    Every row, not the standing one per subject. The table is append-only so that "this
    looked deliberate in March and turned out to be a mistake in June" survives, and a log
    showing only June has discarded the half that made keeping both worth doing. `stands`
    marks which one is current.
    """
    rows = store.every_answer(conn, said_by=said_by)
    if not rows:
        return {
            "status": "nothing_to_check", "basis": "computed", "answers": [],
            "what_it_means": (
                f"Nobody has answered anything{f' — nothing from {said_by}' if said_by else ''}"
                f". That is not the same as nothing having been asked: a judgment that raised "
                f"an unexplained departure asked one, and a gap somebody could not close "
                f"asked another. It means no answer is on file, so every such question is "
                f"still open and will be put again."),
        }
    seen: set = set()
    now = time.time()
    out = []
    for row in rows:
        subject = (row["subject_kind"], row["subject_key"])
        days = int((now - row["created_at"]) / 86400)
        entry = {
            "answer": row["answer"], "note": row["note"], "said_by": row["said_by"],
            "said_at": row["said_at"], "basis": "stated",
            "subject_kind": row["subject_kind"],
            "about": _what_the_answer_was_about(conn, row),
            # Newest first, so the first time a subject is seen is the answer that stands.
            "stands": subject not in seen,
            "days_ago": days,
            "stale": days >= _STALE_ANSWER_DAYS,
            # The overrides that matter most are the ones that made the product say LESS.
            "silences": row["answer"] in ("not_applicable", "deliberate", "misread"),
        }
        # §11.1: the account beside the name. `answers` is the surface built to review
        # overrides, so it is the surface where "who was actually at the keyboard" belongs.
        who = store.authorship_for(conn, "answer", row["subject_key"])
        entry["captured_by"] = (who or {}).get("captured_by") or {
            "method": "unattributed", "source": "stated",
            "what_it_means": "Recorded before this library kept the account."}
        if row["subject_kind"] == "finding":
            entry["evaluation_id"] = row["evaluation_id"]
            entry["finding_id"] = row["subject_key"]
            judgment = store.get_evaluation(conn, row["evaluation_id"] or "")
            entry["subject_title"] = (judgment or {}).get("subject_title")
        seen.add(subject)
        out.append(entry)
    stale = [e for e in out if e["stands"] and e["stale"]]
    silencing = [e for e in out if e["stands"] and e["silences"]]
    return {
        "status": "checked", "basis": "computed", "answers": out,
        "counts": {"answers": len(out), "standing": len(seen), "silencing": len(silencing)},
        "what_it_means": (
            f"{len(seen)} question(s) this library asked have an answer on file, from "
            f"{len(out)} statement(s) — the older ones are kept, because what somebody said "
            f"before the numbers came in and what they said after are two judgments. "
            f"`stands` marks the one in force. {len(silencing)} of them make this product "
            f"say LESS than it otherwise would, which is the half worth reading first."
            + (f" {len(stale)} were given over {_STALE_ANSWER_DAYS} days ago and nobody has "
               f"revisited them; a market written off in March may have been revived since, "
               f"and nothing here re-asks."
               if stale else "")),
    }


def _what_the_answer_was_about(conn, row: dict) -> str:
    """The question in its own words. A log of `finding_id` values is a log nobody can read,
    and the person reviewing overrides is not the person who made them."""
    if row["subject_kind"] != "finding":
        return row["subject_key"]
    judgment = store.get_evaluation(conn, row["evaluation_id"] or "")
    for finding in (judgment or {}).get("findings") or []:
        if finding.get("id") == row["subject_key"]:
            return finding.get("finding") or row["subject_key"]
    return row["subject_key"]


def backfill_author_unknown(conn) -> dict:
    """Stamp records stored before this library read commentary, with the date (§11.6).

    What it stamps is a fact about US: "this record predates commentary extraction, so the
    file may well have said who wrote it and nobody looked." Reporting that as the document's
    silence would blame a customer's deck for a gap in what this product captured.

    It never touches a name. A record whose commentary WAS read carries real people, and
    writing `unknown` over them would destroy the personal data §11.7 has to be able to show
    and erase, while making the library look as though it had never known.

    Runs once per record. A second stamp is not wrong so much as untrue: it moves the import
    date onto the day somebody happened to re-run this.
    """
    when = store._now_iso()
    stamped = store.records_without_authorship_backfill(conn)
    for record in stamped:
        store.mark_authorship_backfilled(conn, record["id"], when)
    return {
        "records": len(stamped), "backfilled_at": when, "basis": "computed",
        "titles": [r["title"] for r in stamped[:10]],
        "what_it_means": (
            f"{len(stamped)} record(s) were stored before this library read commentary at "
            f"all, and are now marked as such with today's date. Their authorship reads "
            f"`unknown` because nobody looked — not because the files were silent — and a "
            f"judgment citing their words is told which of those it is looking at."
            if stamped else
            "Every record already says why it can or cannot name an author. Nothing needed "
            "stamping, which is the answer on a library whose records were all read."),
    }


def record_reaction(conn, *, campaign_id: str, value: str, said_by: str,
                    role: Optional[str] = None, source: str = "stated") -> dict:
    """Record one person's view of a campaign, keeping everyone else's (§11.5).

    The write `update_campaign`'s tag list could not be: it REPLACES, so a second opinion
    erased the first. Both are kept here, and neither is ranked — see `store.disagreement_on`
    and D10 for why this library refuses to say who is right.
    """
    if store.get_campaign(conn, campaign_id) is None:
        raise ValueError(f"{campaign_id!r} is not a record in this library.")
    if value not in store.REACTION_AXES:
        raise ValueError(
            f"{value!r} is not something this library records a view about. The closed sets "
            f"are what make two people comparable: {sorted(store.REACTION_AXES)}.")
    said_by = identity.person(said_by, field="said_by")
    if source not in identity.AUTHOR_SOURCES:
        raise ValueError(f"`source` must be one of {list(identity.AUTHOR_SOURCES)}.")
    said_at = store._now_iso()
    store.record_reaction(conn, campaign_id=campaign_id, value=value, said_by=said_by,
                          source=source, said_at=said_at, role=role)
    store.record_authorship(conn, subject_kind="reaction", subject_key=campaign_id,
                            on_behalf_of=said_by, role=role)
    split = store.disagreement_on(conn, campaign_id)
    return {
        "campaign_id": campaign_id, "value": value, "said_by": said_by, "said_at": said_at,
        "source": source, "basis": "stated", "status": "recorded",
        **({"role": role} if role else {}),
        **({"disagreement": split} if split else {}),
        "what_it_means": (
            f"{said_by}'s view is on file. Everything anybody else said is still on file too "
            f"— this never replaces."
            + (" They and somebody else have recorded different views, which is worth citing "
               "as a split rather than as a verdict."
               if split else "")),
    }


def answer_finding(conn, *, evaluation_id: str, finding_id: str, answer: str, note: str,
                   said_by: str) -> dict:
    """Record what a person says about one finding on a saved judgment (§10.2/D84+D59).

    The write path both rows needed. Every check here exists because the alternative is an
    answer that is worse than the silence it replaced:

    `note` is required, because "deliberate" with no reason is the same non-answer as
    `unexplained` — recorded as though it were an answer, and therefore stopping the question
    being asked. That is strictly worse than the state it replaces.

    `said_by` is required, because this overrides a stored finding, and every `stated` claim
    in this product carries who made it. An anonymous override of a computed finding is the
    one thing this library refuses everywhere.

    The finding has to be ON the judgment, because a settled finding nobody can find is an
    answer filed against nothing — and it would return `status: settled` while settling
    nothing, which is the shape of every defect this review found.
    """
    if answer not in FINDING_ANSWERS:
        raise ValueError(
            f"answer must be one of {list(FINDING_ANSWERS)}, got {answer!r}. "
            + " ".join(f"`{k}`: {v}" for k, v in FINDING_ANSWERS.items()))
    if not (note or "").strip():
        raise ValueError(
            "`note` is required — say why, in the words somebody used. An answer with no "
            "reason is the same non-answer as `unexplained`, except that it stops the "
            "question being asked, which makes it worse than saying nothing.")
    # §11.1/§11.2: `identity.person` is the one guard, so every path that names somebody
    # refuses the same things. It also refuses the product naming itself, which this check
    # did not — and this is the field that overrides a stored judgment.
    said_by = identity.person(said_by, field="said_by")
    judgment = store.get_evaluation(conn, evaluation_id)
    if not judgment:
        raise ValueError(f"{evaluation_id!r} is not a judgment on file")
    finding = next((f for f in judgment.get("findings") or []
                    if f.get("id") == finding_id), None)
    if not finding:
        raise ValueError(
            f"{finding_id!r} is not a finding on {evaluation_id}. Its findings are: "
            f"{[f.get('id') for f in judgment.get('findings') or []]}")

    store.record_answer(conn, subject_kind="finding", subject_key=finding_id,
                        evaluation_id=evaluation_id, answer=answer, note=note.strip(),
                        said_by=said_by.strip())
    store.record_authorship(conn, subject_kind="answer", subject_key=finding_id,
                            on_behalf_of=said_by)
    return {
        "status": "settled",
        "evaluation_id": evaluation_id,
        "finding_id": finding_id,
        "answer": answer,
        "note": note.strip(),
        "said_by": said_by.strip(),
        "basis": "stated",
        "what_it_means": (
            f"Recorded against this finding, and it travels with it: every later judgment "
            f"that retrieves this record reads the answer rather than the question. "
            + "The next judgment of this brief is shown your answer alongside the finding, "
              "and told not to raise it again as though nobody had said anything. That is "
              "the strongest claim this can make: the model still writes the findings, so "
              "it is being informed rather than constrained."),
    }


def answer_gap(conn, *, code: str, answer: str, note: str, said_by: str) -> dict:
    """Record what a person says about a gap they cannot close (§10.2/D53).

    Silenced in the RANKING, never deleted from the record. A gap somebody set aside is still
    a fact about the library's evidence, and a judgment resting on that evidence is still
    weaker for it — hiding it entirely would make the product overstate its own coverage,
    which is what this whole review is about.
    """
    if answer not in GAP_ANSWERS:
        raise ValueError(
            f"answer must be one of {list(GAP_ANSWERS)}, got {answer!r}. "
            + " ".join(f"`{k}`: {v}" for k, v in GAP_ANSWERS.items()))
    if not (note or "").strip():
        raise ValueError(
            "`note` is required — say why this gap is not going to close, or when it will. "
            "A gap set aside with no reason is one nobody can reopen intelligently.")
    said_by = identity.person(said_by, field="said_by")
    # Against the gaps this library ACTUALLY has, not against the registry of codes. A
    # silent no-op for a gap nobody is reporting reads as done and is not — and `_GAP_RANK`
    # would happily accept `no_window` for a library where every record has dates.
    ranked = [g["code"] for g in _ranked_gaps(conn)["gaps"]]
    aside = store.answers_for(conn, "gap")
    if code not in ranked and code not in aside:
        raise ValueError(
            f"{code!r} is not a gap this library is reporting. Call gaps() to see what it "
            f"has: {ranked}")
    # ONE rule, not two. The offer was gated on `_CAN_BE_SET_ASIDE` and the write was not, so
    # the whitelist that keeps somebody from silencing "finished campaigns have no results on
    # file" guarded only the suggestion — `answer_gap(code="library_is_empty",
    # answer="not_applicable")` was accepted on an empty library and `gaps()` then returned
    # `[]`, which is the product saying nothing is missing about a library holding nothing.
    # Two implementations of one rule is the shape this file has now hit five times, and this
    # is the version where the two disagree about something that matters.
    if answer != "open" and code not in _CAN_BE_SET_ASIDE:
        raise ValueError(
            f"{code!r} cannot be set aside. “This will never be true here” is honest about a "
            f"market whose agency no longer exists — {list(_CAN_BE_SET_ASIDE)} — and not "
            f"about a gap that describes work nobody has done yet. Silencing "
            f"`few_verified_outcomes` hides the gap this product was built around, and "
            f"silencing `judgments_never_reconciled` makes its own calibration figure "
            f"unfalsifiable. Close it, or leave it reported.")

    store.record_answer(conn, subject_kind="gap", subject_key=code, answer=answer,
                        note=note.strip(), said_by=said_by.strip())
    store.record_authorship(conn, subject_kind="answer", subject_key=code,
                            on_behalf_of=said_by)
    return {
        "status": "recorded", "code": code, "answer": answer, "note": note.strip(),
        "said_by": said_by.strip(), "basis": "stated",
        "what_it_means": {
            "known_not_yet": "It stays in the ranking, because it is still the thing to fix — "
                            "but the note travels with it, so nobody has to work out whether "
                            "anyone has looked at it.",
            "not_applicable": "It leaves the ranking and stays on the record under "
                              "`set_aside`. Judgments resting on this evidence are still "
                              "weaker for it, and still say so.",
            "open": "Back in the ranking, as though nothing had been said.",
        }[answer],
    }


def authorship_backfill_offer(conn) -> list[dict]:
    """Offer the §11.6 stamp when records actually need it.

    The moment is a library holding records from before commentary extraction: their search
    hits say `unknown` for a reason nobody has recorded, so a reader cannot tell "the file was
    silent" from "we never looked". Silent otherwise — a library whose records were all read
    has nothing to stamp, and offering it anyway is the footer this product keeps refusing.
    """
    waiting = store.records_without_authorship_backfill(conn)
    if not waiting:
        return []
    return [actions.action(
        f"Record that {len(waiting)} older record(s) predate commentary extraction",
        "backfill_author_unknown",
        why=f"{len(waiting)} record(s) were stored before this library read commentary, so "
            f"their authorship reads `unknown` with nothing saying whether the file was "
            f"silent or nobody looked. A judgment citing their words cannot tell a client's "
            f"objection from the deck talking to itself.",
        consent="do")]


def declared_corrections_offer(conn) -> list[dict]:
    """Offer to put the customer's own standing corrections in force (§12.3/D116).

    A customer writes ten corrections into their rulebook, restarts, and nothing happens: the
    rules sit in the file and the library never mentions them. That is D108's "stored and
    never applied" arriving one level up — through nobody being told there was anything to do.
    This product has hit that shape eight times, and §10.6's answer is that a capability
    nobody offers is a capability nobody calls.

    Silent once they are in force, because guidance that never stops appearing is guidance
    nobody reads.
    """
    import corrections
    import rulebook

    try:
        declared = rulebook.corrections()
    except ValueError:
        return []   # a broken rulebook is `health_check`'s finding, loudly, not an offer
    if not declared:
        return []
    waiting = [entry for entry in declared if not corrections.find(conn, entry["text"])]
    if not waiting:
        return []
    return [actions.action(
        f"Put the {len(waiting)} standing correction(s) from your rulebook in force",
        "load_rulebook_corrections",
        why=f"Your rulebook ({rulebook.overlay() or rulebook.version()}) declares "
            f"{len(waiting)} standing correction(s) that no brief is being judged against "
            f"yet. They do not have to be seen in three campaigns first — that test is for a "
            f"rule this library inferred — but somebody has to put their name to them.",
        consent="ask",
        # NOT prefilled. The one thing the server cannot work out is whose confirmation this
        # is, and filling it in would put a rule in front of every future brief under a name
        # nobody gave.
        needs=["who is confirming these — ask, do not assume"])]


def stale_answers_offer(conn) -> list[dict]:
    """Offer the override log when something that SILENCES this product has gone unreviewed.

    The honest moment for it, and the only one. Offering "read the log" right after somebody
    writes to it is noise — they have just been told what they said. What nobody is ever told
    is that a gap written off months ago is still written off: the queue's founding insight is
    that a question asked once and never again is a question nobody answered, and until now
    that insight was not applied to the ANSWERS. A market written off in March may have been
    revived since, and nothing here re-asks.

    Gated on `stale` AND `silences`, so it fires on the answers that made the library say
    less — never on a tidy log of things somebody explained.
    """
    silenced = [a for a in answers(conn).get("answers") or []
                if a["stands"] and a["stale"] and a["silences"]]
    if not silenced:
        return []
    return [actions.action(
        f"Review the {len(silenced)} answer(s) nobody has revisited",
        "answers",
        why=f"{len(silenced)} standing answer(s) make this library report less than it "
            f"otherwise would, and the most recent is {min(a['days_ago'] for a in silenced)} "
            f"days old. Nothing re-asks; a market written off last year may have come back.",
        consent="do")]


def gaps_offer(conn) -> list[dict]:
    """Offer the ranked gaps, when there is a ranking worth reading (§10.6/D54).

    `gaps`'s own docstring says "call it when the user asks how good their library is" — an
    instruction inside a tool result, which is the form this project's stated principle says
    models drop. The library knows it holds one measured campaign and has never said so.

    Not on an empty library: `library_is_empty` is the only gap there, `shortest_path` is
    already the answer to it on the same response, and an offer to enumerate the gaps of an
    empty library is the permanent complaint this surface is written against.
    """
    top = top_gap(conn)
    if not top or top == "library_is_empty":
        return []
    return [actions.action(
        "Show what this library is missing, worst first",
        "gaps",
        why=_GAP_HEADLINE.get(top, "Something in the library is holding its judgments back."),
        consent="do")]


# The one-line reason for offering the ranking, per top gap. Written here rather than read
# off the gap's own `what`, because that sentence carries counts and names and this one has
# to fit on a line beside two other offers — and because an offer whose `why` is the answer
# it is offering to fetch is an offer nobody needs to accept.
_GAP_HEADLINE = {
    "few_verified_outcomes": "Finished campaigns here have no results on file, so judgments "
                             "citing them rest on somebody's impression.",
    "market_without_outcomes": "A whole market has nothing measured, so judgments about it "
                               "compare against campaigns somewhere else.",
    "partly_indexed": "Records are stored and not searchable, so their content is absent "
                      "from evidence without being absent from the library.",
    "execution_never_checked": "Outcomes are being read as though the brief caused them, "
                               "with nothing on file showing what actually ran.",
    "no_window": "Finished campaigns have no dates, so nothing can be checked against what "
                 "else was going on at the time.",
    "judgments_never_reconciled": "Judgments with results on file have never been checked "
                                  "against what happened, so the calibration figure is "
                                  "about whoever bothered.",
}


def coverage_offer(conn) -> list[dict]:
    """Offer the coverage matrix, when the library has a shape to show (§10.6/D67).

    `coverage`'s own docstring says `list_campaigns` "answers 'what have I got' one record at
    a time. This answers the question a marketer actually has" — which is the tool admitting
    the listing is the moment for it, in a sentence the listing never carried.

    Gated on `thin` being non-empty, which is narrower than "there are cells". A wholly
    unmeasured library reports `thin: []` with a `thin_summary` saying the place to start is
    not a particular market — and `gaps` says that better, with an offer that closes it. So
    the matrix is offered when there IS a contrast to see: some cells measured, some not.
    """
    shape = coverage(conn)
    thin = shape.get("thin") or []
    if not thin or shape.get("cells_total", 0) < 2:
        return []
    # `thin[0]`, not a second grouping computed here — D67's own words. The first version of
    # `gaps` grew its own market-or-region grouping beside §5.5's and the two then described
    # the same library differently (D55); an offer POINTING at coverage that counted the
    # library itself would be that failure inside the fix for it.
    worst = thin[0]
    where = worst.get("market") or worst.get("collection") or worst.get("stage") or "one cell"
    return [actions.action(
        "Show where this library is thick and where it is thin",
        "coverage",
        # "the largest gap", NOT "the thinnest". `thin` is ranked by `(_EVIDENCE_ORDER,
        # -campaigns)`, so `thin[0]` is the worst-evidence cell with the MOST campaigns in
        # it — the biggest hole, not the smallest cell. Calling it the thinnest reported a
        # cell holding nine unmeasured campaigns as thinner than one holding a single
        # unmeasured campaign, which is backwards in the field that decides where to look.
        why=f"{shape['thin_total']} of {shape['cells_total']} cells are weak — "
            f"{where} is the largest of them, and the similarity score looks the same "
            f"whether a judgment came from one precedent there or nine.",
        consent="do")]


def _ranked_gaps(conn) -> dict:
    """The ranked gaps alone: everything that can decide `most_valuable`."""
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
        #
        # §10.6/D76: the FIRST step of the ordered path, not a separate list of two of its
        # three. `to_first_upload()` said "add a campaign" where the path says "add one you
        # were happy with", and the difference is the whole point of the path — a library of
        # things somebody liked cannot judge on the axis it is being asked about.
        found.append({
            "code": "library_is_empty",
            "what": "There are no campaigns in the library yet.",
            "why_it_matters": "Every judgment this product makes is a comparison against "
                              "what you have already run, so with nothing stored there is "
                              "nothing to compare against.",
            "counts": {"campaigns": 0}, "affects": 0,
            "next_actions": first_steps(conn)[:1],
        })
        return _what_people_said_about_the_gaps(conn, _ranked(found, library_size=0))

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
            "affects": len(ran) - len(with_outcomes),
            "next_actions": actions.trim([actions.action(
                "Record what one of these campaigns actually achieved",
                "add_metrics",
                why=f"{len(ran) - len(with_outcomes)} finished campaigns have no results.",
                consent="ask", needs=["which campaign, and the numbers"],
                campaign_id=next(c["id"] for c in ran if c["id"] not in measured_ids))]),
        })

    # §9.1/§9.2: a concluded campaign with measured results, briefed creative, and nothing
    # showing what actually ran. The review's third possibility — "that what ran was not what
    # was briefed" — which the library could not previously notice at all.
    unchecked = [c for c in with_outcomes
                 if store.assets_in_phase(conn, c["id"], "proposed")
                 and not store.assets_in_phase(conn, c["id"], "delivered")]
    if unchecked:
        found.append({
            "code": "execution_never_checked",
            "what": f"{len(unchecked)} finished campaign"
                    f"{'s' * (len(unchecked) != 1)} with results on file "
                    f"{'have' if len(unchecked) != 1 else 'has'} briefed creative and no "
                    f"photographs of what actually ran: "
                    f"{', '.join(c['title'] for c in unchecked[:_MAX_NAMED])}"
                    + (f" (and {len(unchecked) - _MAX_NAMED} more)"
                       if len(unchecked) > _MAX_NAMED else "") + ".",
            "why_it_matters": ("Every outcome on those campaigns is being read as though the "
                               "brief caused it. If what ran was not what was briefed, the "
                               "library is learning from the wrong document and has no way "
                               "to notice."),
            "counts": {"campaigns": len(unchecked)}, "affects": len(unchecked),
            "next_actions": actions.trim([actions.action(
                f"Add the photographs from \u201c{unchecked[0]['title']}\u201d",
                "upload_image_asset",
                why="With the delivered photographs on file, compare_execution says which "
                    "briefed elements appeared, which did not, and which arrived unbriefed.",
                consent="ask",
                needs=["the photographs themselves"],
                campaign_id=unchecked[0]["id"], phase="delivered")]),
        })

    # §12.4/D51: records whose deck's comments were never read. Recorded since §2.5 and NOT
    # reported until now, for a good reason: the only action available was `upload_campaign`,
    # which creates a SECOND record and fires `duplicate_title` — the offer §5.2 refused in
    # writing. "A gap whose only action makes things worse is a complaint."
    #
    # D39 built `attach_deck`, so the gap has an action that works and can be reported.
    # RECORDS WITH NO DECK AT ALL, which is the population `attach_deck` can actually repair.
    # Without the deck test this fired on every record whose brief arrived as typed text and
    # every upload whose file type carries no comments this product can read — and for those
    # the gap's own sentence ("not because their decks were silent, but because no file was
    # ever read") is false, while the action it offers is the one `attach_deck` refuses. A gap
    # whose only remedy is rejected by the tool it names is the complaint §5.2 spent an item
    # removing, arriving back through the door the fix for it opened.
    #
    # It is not a gap this product could honestly report for the others either: it cannot read
    # a .docx's comments, so it cannot know whether there are any, and "N records may have
    # unread comments" is the confident claim from an absence that §2.4 exists to refuse.
    unread = [c for c in store.list_campaigns(conn)
              if c.get("record_type") == "campaign"
              and not c.get("commentary_checked")
              and not (c.get("deck_text") or "").strip()
              and not c.get("asset_path")
              and c["id"] not in store.get_superseded_campaign_ids(conn)]
    if unread:
        found.append({
            "code": "commentary_never_read",
            "what": f"{len(unread)} record{'s' * (len(unread) != 1)} "
                    f"{'have' if len(unread) != 1 else 'has'} never had a deck read for "
                    f"comments: "
                    f"{', '.join(c['title'] for c in unread[:_MAX_NAMED])}"
                    + (f" (and {len(unread) - _MAX_NAMED} more)"
                       if len(unread) > _MAX_NAMED else "") + ".",
            "why_it_matters": ("A client's recorded objection is often the most useful "
                               "precedent in the library, and these records have none — not "
                               "because their decks were silent, but because no file was ever "
                               "read. A judgment citing them cannot tell those apart."),
            "counts": {"records": len(unread)},
            "affects": len(unread),
            "basis": "computed",
            # The newest record this gap counts, so a set-aside made about the records on
            # file then does not silence it for records that arrive later (§10.2).
            "since": max((c.get("created_at") or 0) for c in unread),
            # NOT `upload_campaign`: that is the duplicate §5.2 refused in writing, and
            # offering it here is what kept this gap unreported for seven phases.
            "next_actions": actions.trim([actions.action(
                f"Attach the deck for \u201c{unread[0]['title']}\u201d",
                "attach_deck",
                why="The deck's comments are read when it is attached, so the client's own "
                    "objections become precedent a judgment can cite — on the SAME record "
                    "rather than a second copy of it.",
                consent="ask",
                needs=["the deck file itself"],
                campaign_id=unread[0]["id"])]),
        })

    # §9.9: the item's whole diagnosis is that reconciliation "needs somebody to decide to go
    # back and nobody does" — and it gave itself no gap, so the number waiting was invisible on
    # every reporting surface. `calibration` meanwhile reported a perfect record beside them.
    waiting = store.unreconciled_judgments(conn)
    if waiting:
        found.append({
            "code": "judgments_never_reconciled",
            "what": f"{len(waiting)} judgment"
                    f"{'s' * (len(waiting) != 1)} with results on file "
                    f"{'have' if len(waiting) != 1 else 'has'} never been checked against "
                    f"what actually happened: "
                    f"{', '.join(j['subject_title'] for j in waiting[:_MAX_NAMED])}"
                    + (f" (and {len(waiting) - _MAX_NAMED} more)"
                       if len(waiting) > _MAX_NAMED else "") + ".",
            "why_it_matters": ("Reconciliation is the only thing in this product that says "
                               "whether its own judgments are any good, and it needs somebody "
                               "to go back. Until these are checked, the calibration figure "
                               "is about whoever bothered."),
            "counts": {"judgments": len(waiting),
                       "campaigns": len({j["campaign_id"] for j in waiting})},
            # RECORDS, like every other gap's `affects`. It counted JUDGMENTS, and `share`
            # then divided judgments by campaigns: one campaign carrying three unreconciled
            # judgments reported `share: 3.0` — not a share of anything — and, being above
            # the threshold, claimed that single record "outweighs its kind" and moved up a
            # rank. A ratio of two different units is a number that cannot be wrong, because
            # it does not mean anything. The judgment count is still reported, in `counts`,
            # where it is labelled.
            "affects": len({j["campaign_id"] for j in waiting}),
            "next_actions": actions.trim([actions.action(
                f"Check what the library said about \u201c{waiting[0]['subject_title'][:36]}\u201d "
                f"against what happened",
                "reconcile_evaluation",
                why="The results are on file and the judgment has never been tested against "
                    "them. This is the one surface that grades the product.",
                consent="ask", evaluation_id=waiting[0]["id"])]),
        })

    # §9.6: a concluded campaign with no window is one the calendar can never reach. §9.5 gave
    # itself `execution_never_checked` for exactly this reason — without a gap, the absence is
    # invisible on every reporting surface and nothing ever says "none of these can be checked
    # against what else was going on".
    windowless = [c for c in ran if not c.get("starts_on")]
    if windowless:
        found.append({
            "code": "no_window",
            "what": f"{len(windowless)} finished campaign"
                    f"{'s' * (len(windowless) != 1)} "
                    f"{'have' if len(windowless) != 1 else 'has'} no dates on file, so "
                    f"nothing can be matched to what else was happening at the time: "
                    f"{', '.join(c['title'] for c in windowless[:_MAX_NAMED])}"
                    + (f" (and {len(windowless) - _MAX_NAMED} more)"
                       if len(windowless) > _MAX_NAMED else "") + ".",
            "why_it_matters": ("“This launch overlapped Ramadan” and “the port "
                               "was shut for half of it” are findings no model needs to "
                               "be clever to produce — but only for a campaign that says when "
                               "it ran. Without a window every outcome is read as though "
                               "nothing else was going on."),
            "counts": {"campaigns": len(windowless)}, "affects": len(windowless),
            "since": max((c.get("created_at") or 0) for c in windowless),
            "next_actions": actions.trim([actions.action(
                f"Say when “{windowless[0]['title'][:36]}” ran",
                "update_campaign",
                why="A window is what lets recorded events reach it — and they reach it "
                    "automatically once it has one, with nothing to link by hand.",
                consent="ask",
                needs=["starts_on — the first day it ran (YYYY-MM-DD)",
                       "ends_on — the last day (YYYY-MM-DD)"],
                campaign_id=windowless[0]["id"])]),
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
            "affects": len({c["id"] for m in barren for c in by_market[m]}),
            "since": max((c.get("created_at") or 0)
                         for m in barren for c in by_market[m]),
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
            "affects": len(by_campaign),
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

    return _what_people_said_about_the_gaps(
        conn, _ranked(found, library_size=len(campaigns)))


# The only gaps where "this will never be true here" is an honest thing for somebody to say.
#
# A whitelist, after review found the blacklist let through the two that matter most. The
# honest case for D53 is a market whose agency no longer exists and whose numbers are gone —
# a fact that CANNOT become true. It is not `few_verified_outcomes` ("finished campaigns have
# no results on file"), which is the gap behind the original customer complaint, and it is
# not `judgments_never_reconciled`, where silencing it makes the product's own self-
# assessment unfalsifiable — the calibration figure would then be about whoever bothered,
# with nothing left to say so. Those are about work nobody has DONE, not facts that cannot
# exist, and a blacklist cannot tell the difference because the difference is not syntactic.
#
# `execution_never_checked` is out for the same reason: photographs of what ran are work.
# §12.4/D51: `commentary_never_read` joins them, and it has to. A brief that only ever
# existed as pasted text has no file whose comments could be read, so without a way to say
# "there is no deck" the gap is permanent and unclosable — which is a gap everybody learns to
# ignore, and D51's own reason for not reporting it at all. Now it closes two ways: attach the
# deck, or say there is not one.
_CAN_BE_SET_ASIDE = ("market_without_outcomes", "no_window", "commentary_never_read")

# One per response, ever. Measured at 3 of 7 offers in a single `gaps()` reply — one per gap,
# because `trim` caps each gap's own list and nothing capped the response — which made "shall
# we stop mentioning this?" the most repeated sentence in the product's report on its own
# evidence. "Last in the gap's own list" controlled position and did nothing about frequency.
_MAX_SET_ASIDE_OFFERS = 1


def _what_people_said_about_the_gaps(conn, ranked: dict) -> dict:
    """Apply D53's answers to the ranking (§10.2).

    Inside `_ranked_gaps` rather than inside `gaps()`, so `top_gap` sees the same list. With
    it one level up, the session-start offer fired on a gap somebody had set aside while
    `gaps()` did not show it — two surfaces describing the same library differently, which is
    D55's shape and the one this file has now hit four times.
    """
    said = store.answers_for(conn, "gap")
    kept = []
    offered_to_set_aside = 0
    for gap in ranked["gaps"]:
        answer = said.get(gap["code"])
        if answer and answer["answer"] == "not_applicable":
            # §10.2/D53 and §12.4: the set-aside is keyed on the gap's CODE, and the evidence
            # underneath it moves. "This will never be true here" is a statement about the
            # records somebody was looking at — a market whose agency no longer exists, decks
            # that are genuinely lost — and it says nothing whatever about a record uploaded
            # afterwards. Suppressed by code alone, one decision silenced the gap permanently,
            # including for evidence that did not exist when it was made: the product agreeing
            # to stop mentioning something it had not yet seen. `since` is the newest thing
            # this gap counts; when it postdates the answer, the gap comes back and says why.
            since = gap.get("since")
            recorded = answer.get("recorded_at")
            if not (since and recorded and since > recorded):
                continue
            gap["answered"] = answer
            gap["reopened"] = {
                "basis": "computed",
                "what_it_means": (
                    f"This was set aside by {answer['said_by']}, and records counted by it "
                    f"have arrived since. What was said then was about what was on file "
                    f"then; it is reported again because this evidence is new. Setting it "
                    f"aside again covers these too."),
            }
            kept.append(gap)
            continue
        if answer and answer["answer"] == "known_not_yet":
            gap["answered"] = answer
        elif (not answer and gap["code"] in _CAN_BE_SET_ASIDE
                and offered_to_set_aside < _MAX_SET_ASIDE_OFFERS
                and not gap.get("closed_by") and gap.get("next_actions")):
            offered_to_set_aside += 1
            # NOT on a gap whose own offer was de-duplicated away. `_ranked` drops an offer
            # another gap already makes and leaves `closed_by` pointing at it — so appending
            # the silencing offer afterwards made it the ONLY thing on that gap, and the
            # product's single suggestion about a real hole in the evidence was "shall we
            # stop mentioning this?". Reproduced on an ordinary five-record library: the
            # `market_without_outcomes` row offered nothing else at all.
            #
            # §10.2/D53, LAST in the gap's own list and never first: the offer that closes a
            # gap is worth more than the offer that silences it, and a product whose opening
            # suggestion is "shall we stop mentioning this?" is one that teaches people to
            # dismiss its findings. Offered at all because some gaps genuinely never close,
            # and reporting one of those forever is the permanent complaint this surface is
            # written against.
            gap["next_actions"] = actions.trim((gap.get("next_actions") or []) + [
                actions.action(
                    "Set this aside if it is never going to close",
                    "answer_gap",
                    why="Some gaps are true and permanent — a market whose agency no longer "
                        "exists will never have its results. It stays on the record either "
                        "way; it stops being ranked.",
                    consent="ask",
                    needs=["answer — `known_not_yet` (known, not yet) or `not_applicable` "
                           "(never going to be true here)",
                           "note — why", "said_by — whose call it is"],
                    code=gap["code"])])
        kept.append(gap)
    return {
        **ranked,
        "gaps": kept,
        # Recomputed, because the head of the list may have just left it. `most_valuable`
        # naming a gap that is not in `gaps` is the inconsistent pair `coverage` was fixed for.
        "most_valuable": kept[0]["code"] if kept else None,
        # Visible, not deleted. A gap somebody set aside is still a fact about this library's
        # evidence, and a judgment resting on it is still weaker for it — dropping it
        # entirely would make the product overstate its own coverage, which is the failure
        # this whole review is about.
        # Only gaps this library STILL has. A gap set aside in March and then genuinely
        # closed in June is not "set aside" any more, it is closed — and listing it says the
        # library is choosing not to look at something it no longer has, which is a worse
        # misreading than not mentioning it. `ranked["gaps"]` is the list before the answers
        # were applied, so it is the honest test of "is this still true".
        "set_aside": [{"code": code, **answer} for code, answer in sorted(said.items())
                      if answer["answer"] == "not_applicable"
                      and code in {g["code"] for g in ranked["gaps"]}],
    }


# D52: how much of the library a gap has to be about before its SIZE moves it up a step past
# a gap of a worse kind. A share, not a count, because a threshold of "ten records" is the
# wrong threshold for a library of eight and for a library of two thousand, and the ranking
# matters most on the first day, when neither the traffic nor the record count exists yet.
#
# Half. A gap about most of the library is the library's problem; a gap about a quarter of it
# is a problem in the library, and the kind ordering — which says what each gap COSTS — is
# still the better guide to which of those to fix first.
_MAGNITUDE_SHARE = 0.5


def _ranked(found: list[dict], *, library_size: int = 0) -> dict:
    """Order the gaps (§5.3), by kind and then by how much of the library each is about.

    D52: *"Gap ranking by judgments affected rather than a fixed rank per kind — one
    unindexed record currently outranks forty."* The fixed table cannot see magnitude at all,
    so the order of two gaps was decided entirely by which KIND they were — and `gaps()`'s own
    docstring says the ranking IS the design: "an unordered list of everything absent is the
    thing nobody reads". A ranking blind to size is an unordered list wearing numbers.

    Not "by judgments affected", which is what the row asks for and is not available: a
    library needs a great many saved judgments before that number means anything, so a
    ranking resting on it is wrong for every new customer — which is every customer, on the
    day the ranking matters most. RECORDS affected is what the review's own example counts,
    and it is a number this library always has.

    Kind stays the primary key and size is allowed to override exactly one step of it. Both
    extremes are wrong: kind-only is the defect, and size-only would say a market with nothing
    measured matters less than a half-indexed deck the moment one more record was affected.
    """
    for gap in found:
        gap["rank"] = _GAP_RANK[gap["code"]]
        gap["affects"] = gap.get("affects", 0)
        # Records over records. Every `affects` counts RECORDS for exactly this reason: the
        # moment one gap counts something else, `share` stops being a share and the magnitude
        # rule starts acting on a number with no meaning.
        # Deliberately NOT clamped to 1.0. A clamp would have hidden the defect this comment
        # is about — `judgments_never_reconciled` reporting `share: 3.0` — by turning a
        # meaningless number into a plausible one, which is the failure mode this whole
        # product is written against. If a share ever exceeds 1 again, something is counting
        # the wrong unit and the tests should say so rather than the code smoothing it over.
        gap["share"] = (round(gap["affects"] / library_size, 3) if library_size else 0.0)
        # One step, and it is named in the output rather than folded silently into `rank`: a
        # number that decides the order and does not appear is one nobody can argue with,
        # which is the review's complaint about this product in one sentence.
        gap["outweighs_its_kind"] = gap["share"] >= _MAGNITUDE_SHARE and gap["affects"] > 1
        # The number this list is actually SORTED by, beside the kind's own rank rather than
        # folded into it. A reader who checks that the output is ordered has to have the key
        # it is ordered on: with only `rank` published, a correctly-ordered list looked
        # unsorted, which is how a sort key nobody can see becomes a sort nobody can audit.
        gap["order"] = gap["rank"] - int(gap["outweighs_its_kind"])
        gap["rank_basis"] = "computed"
    # `-affects` second, so two gaps of the same `order` are ordered by size rather than by
    # insertion — `partly_indexed` and `no_window` have tied at 4 since they were written,
    # and the tie was broken by which line came first in this function.
    found.sort(key=lambda g: (g["order"], -g["affects"], g["code"]))

    # Two gaps can be closed by one act — a library whose single unmeasured campaign is also
    # its only LATAM campaign has two true facts and one thing to do. Offering the same call
    # twice is how somebody accepts both and appends two identical metric rows to the same
    # record. The later gap keeps the fact and points at what closes it.
    seen: dict[tuple, str] = {}
    for gap in found:
        kept = []
        for offer in gap["next_actions"]:
            key = actions.identity(offer)
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


# How many passages a diff shows per side. Two long decks with nothing in common would
# otherwise return both in full, in a reply somebody has to read.
_MAX_PASSAGES = 10


# A single passage's ceiling in the reply. `_MAX_PASSAGES` bounds the COUNT and nothing
# bounded the size: a deck with no blank lines in it is one paragraph, so one "passage" was
# 78 KB on each side and a diff of two such decks was a 156 KB tool result. Truncated with the
# cut marked, because a quotation that has been shortened must never look complete.
_MAX_PASSAGE_CHARS = 1200


def _folded_passage(text: str) -> str:
    """The form two passages are compared on: whitespace, case and Unicode composition folded.

    NFC is not a nicety. "Café" is one codepoint on a Windows machine and "e" plus a combining
    accent on a Mac, the two are the same sentence to every reader, and Python compares them
    unequal — so a deck that had been through both reported every accented paragraph as
    removed AND added, which is the diff saying the brief changed when nobody touched it.
    """
    import unicodedata

    return unicodedata.normalize("NFC", " ".join(str(text or "").split())).casefold()


def _short_passage(text: str) -> str:
    """One passage, bounded, with the cut visible."""
    whole = str(text or "").strip()
    if len(whole) <= _MAX_PASSAGE_CHARS:
        return whole
    return whole[:_MAX_PASSAGE_CHARS].rstrip() + f" … [{len(whole)} characters in all]"


def _passages_changed(conn, earlier: str, later: str) -> dict:
    """Added and removed passages between two versions, by layer (§12.4/D63).

    From what is ALREADY STORED for each version: nothing is extracted again and no file is
    re-read, which is the item's framing. The two layers come from different places, and the
    comment inside `by_layer` says why — the body from the record's own `deck_text`, the
    commentary from its chunks. D63's wording is "from the chunks already stored"; taking that
    literally made every version differ on its own title, because the first chunk carries the
    title and detail as a summary. The departure is deliberate; claiming both was the error.

    Compared on a FOLDED form so that whitespace and case do not register as a change — a diff
    that reports every paragraph because the chunker packed them differently is one nobody
    reads twice. Unicode is normalised in the same fold: "Café" typed as one codepoint and as
    "e" plus an accent are the same sentence to every reader and different bytes to Python, so
    a deck round-tripped through a Mac reported its whole text as replaced.

    By layer, and the layer is load-bearing. A paragraph gone from the BODY is the brief
    changing; a comment gone is the client's remark being dropped, which is a different and
    usually more interesting event. One list would make a deleted objection look like an edit.
    That is also why commentary is filled FIRST when the reply has to be trimmed: body first,
    a fifteen-paragraph rewrite pushed out every dropped client comment, so the layer the
    docstring calls more interesting was the layer truncation removed.
    """
    def by_layer(campaign_id: str) -> dict:
        out: dict = {"body": {}, "commentary": {}}
        record = store.get_campaign(conn, campaign_id) or {}
        # The BODY from `deck_text`'s own paragraphs, not from the chunks. A chunk is not a
        # passage: `chunking.pack` merges units up to 1800 characters, and the first chunk
        # carries the title and detail as a summary — so every version differed on its own
        # title and the whole deck read as replaced. §6.1 made the same call for quotes, for
        # the same reason: the chunk boundary is one the document does not have.
        for unit in (record.get("deck_text") or "").split("\n\n"):
            folded = _folded_passage(unit)
            if folded:
                out["body"].setdefault(folded, _short_passage(unit))
        # Commentary IS one chunk per comment and never merged, so its chunks are its
        # passages — and they carry the author and anchor nothing else can reconstruct.
        for row in conn.execute(
                "SELECT text FROM campaign_chunks WHERE campaign_id = ? AND kind = "
                "'commentary'", (campaign_id,)).fetchall():
            folded = _folded_passage(row["text"] or "")
            if folded:
                out["commentary"].setdefault(folded, _short_passage(row["text"] or ""))
        return out

    before, after = by_layer(earlier), by_layer(later)
    added, removed, more = [], [], 0
    # COMMENTARY FIRST, and this ordering is the whole of the layer rule at the point where it
    # costs something. Body first, a deck with fifteen rewritten paragraphs and two dropped
    # client comments returned ten body passages and ZERO commentary, with the comments folded
    # into an undifferentiated `more: 12` — the layer this function's own docstring calls "the
    # more interesting of the two" was the one the cap removed, every time, because there is
    # always more body than commentary.
    for layer in ("commentary", "body"):
        gone = [before[layer][k] for k in before[layer] if k not in after[layer]]
        new = [after[layer][k] for k in after[layer] if k not in before[layer]]
        more += max(0, len(gone) - _MAX_PASSAGES) + max(0, len(new) - _MAX_PASSAGES)
        removed += [{"layer": layer, "text": t} for t in gone[:_MAX_PASSAGES]]
        added += [{"layer": layer, "text": t} for t in new[:_MAX_PASSAGES]]

    return {
        "added": added[:_MAX_PASSAGES], "removed": removed[:_MAX_PASSAGES],
        "more": more + max(0, len(added) - _MAX_PASSAGES) + max(0, len(removed) - _MAX_PASSAGES),
        "basis": "computed",
        "what_it_means": (
            "What the two decks say differently, from the passages already stored for each. "
            "`layer` matters: a passage gone from `body` is the brief changing, and one gone "
            "from `commentary` is a remark somebody made about it being dropped — which is "
            "usually the more interesting of the two. Whether a change ANSWERS anything is a "
            "reading of both decks and is not asserted here."
            if added or removed else
            "The two decks say the same things, passage for passage. Any difference between "
            "them is in the structured fields rather than the text."),
    }


_MAX_CLIENT_ASKS = 8

# The commentary kinds that are SOMEBODY ELSE writing on the deck (§12.4/D62). `extract` reads
# three, and the third is not one of these: a `speaker_note` is the notes slide, which is the
# deck author's own presenter script. All three are worth storing and searching — the notes
# slide often says what the brief means — but only these two are an ASK somebody made of the
# work. Read through a list with the kind stripped off, "Presenter: remember to smile" came
# back as what the client wrote when they sent the deck back.
_SOMEBODY_ELSES_REMARK = ("comment", "annotation")

# What the earlier deck's VERDICT does to the weight of its asks (§12.4/D15 read beside D62).
# One sentence each, and they are genuinely different readings of the same quotation — which
# is the argument for recording the verdict at all.
_WHAT_THE_VERDICT_DOES_TO_THE_ASKS = {
    "rejected": ("The earlier deck was REJECTED, so these are the objections it did not "
                 "survive. A version answering them is answering the reason it was refused."),
    "approved_with_changes": ("The earlier deck was signed off CONDITIONALLY, so these are "
                              "what it was conditional on. `approval_note` says which."),
    "approved": ("The earlier deck was signed off as it stood, so these are remarks made "
                 "alongside a yes rather than conditions on one — worth reading and not "
                 "worth treating as a blocker."),
    "withdrawn": ("The earlier deck was taken back before a verdict, so nobody said no to "
                  "it. These are the remarks it had collected by then."),
}


def _what_the_client_asked_for(conn, earlier: str) -> dict:
    """The comments on the earlier deck, as the asks a new version is answering (§12.4/D62).

    This is the §12.4/D15 decision put to work: a returned deck's tracked comments ARE the
    approval notes, so they are also the corrections the next version is responding to. The
    product was reading them as searchable text and as nothing else.

    It does NOT decide whether each one was addressed. Whether "drop the third colourway" was
    done is a judgment about two documents, and asserting it from a keyword overlap is the
    confident-unfounded claim this product exists to avoid — `adopted` says what the LIBRARY's
    findings did, from finding ids, which is a fact. This says what was asked, quotes it, and
    leaves the reading to whoever is looking at both decks.
    """
    # `get_commentary`, not `text_on_file`. The latter strips `kind`, and the three kinds are
    # not the same claim: a `comment` and a PDF `annotation` are somebody writing ON the deck,
    # and a `speaker_note` is the DECK AUTHOR'S OWN presenter script. Read through the stripped
    # list, "Presenter: remember to smile, and skip slide 4" was reported to the model as "what
    # the client wrote when they sent it back" — the agency's own note, filed as the client's
    # ask, by a product whose entire thesis is that it can say where a claim came from.
    everything = store.get_commentary(conn, earlier)
    rows = [row for row in everything
            if row.get("kind") in _SOMEBODY_ELSES_REMARK and (row.get("text") or "").strip()]
    own_notes = len(everything) - len(rows)
    items = []
    for row in rows[:_MAX_CLIENT_ASKS]:
        item = {"text": " ".join((row.get("text") or "").split())}
        # The author where the file carried one, because "who asked for this" is half of what
        # makes an ask worth answering — and because an unattributed remark must not be
        # narrated as the client's either.
        for field in ("author", "slide", "page"):
            if row.get(field):
                item[field] = row[field]
        items.append(item)
    if not items:
        return {"count": 0, "items": [], "basis": "computed",
                "what_it_means": (
                    "The earlier deck carries no remarks anybody wrote ON it"
                    + (f" — only {own_notes} speaker note(s), which are the deck author's own "
                       f"script rather than anything a reviewer asked for."
                       if own_notes else
                       " — either nobody wrote any, or the file was never read for them, "
                       "which `commentary_never_read` in `gaps` tells apart.")),
                }
    named = sorted({str(row["author"]).strip() for row in rows if (row.get("author") or "").strip()})
    # §12.4/D15's verdict, BESIDE the asks, because the two are one fact read apart. "Drop the
    # third colourway" under `rejected` is why the deck did not proceed; the same sentence
    # under `approved_with_changes` is what the sign-off was conditional on; under `approved`
    # it is a remark somebody made in passing. The verdict was stored and read by nothing, and
    # this is the surface where not having it changes what the asks MEAN.
    verdict = (store.get_campaign(conn, earlier) or {}).get("approval")
    said_so = (store.get_campaign(conn, earlier) or {}).get("approval_by")
    return {
        "count": len(rows), "items": items, "basis": "computed",
        **({"approval": verdict, "approval_by": said_so,
            "approval_means": _WHAT_THE_VERDICT_DOES_TO_THE_ASKS[verdict]}
           if verdict in _WHAT_THE_VERDICT_DOES_TO_THE_ASKS else {}),
        # `some_unattributed` rather than a sentence, so a caller can act on it: a reader who
        # cannot tell "R. Vega asked for this" from "somebody did" will eventually quote the
        # second as the first.
        "authors": named, "some_unattributed": any(not (r.get("author") or "").strip()
                                                   for r in rows),
        "what_it_means": (
            f"{len(rows)} remark(s) written on the earlier deck by "
            + (f"{', '.join(named)}" if named else "somebody the file does not name")
            + f": the asks this version is answering, quoted rather than judged — whether "
              f"each was addressed is a reading of both decks, and this product does not "
              f"assert it from word overlap. `adopted` beside this is a different claim: what "
              f"the LIBRARY's own findings did, from their ids."
            + (f" {own_notes} speaker note(s) on the deck are NOT here: those are the deck "
               f"author's own script, not a reviewer's ask." if own_notes else "")),
    }


# §12.4/D113. The stored vocabulary cannot be renamed — every saved row and every tool
# argument is written in it — so each surface says which kind it means, in the words a reader
# uses. `corrections._public` carries the standing-rule version of this sentence.
_WHAT_A_DIFF_CORRECTION_IS = (
    "A correction HERE is an observation about two documents: a finding raised on the earlier "
    "version that the later one answered. It binds nothing and judges nothing else. It is not "
    "a STANDING correction — a rule this library watched recur until somebody confirmed it, "
    "which then judges every brief in its markets — and it is not a RECOMPUTED number, which "
    "is a corrected value for one measure on one campaign. This product calls all three "
    "corrections and they carry completely different weight.")


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
        # §12.4/D113, set HERE rather than at the end of the function: this returns early for
        # records with no structured judgment on both sides, so a sentence appended at the
        # bottom reached only half the callers — and the half it missed is the ordinary one.
        "what_a_correction_means_here": _WHAT_A_DIFF_CORRECTION_IS,
        # D58: what the BRIEFS say differently, not just what their structured fields do. A
        # budget dropped, a channel gone, a date contradiction introduced — all invisible
        # until §7.1 could read them, and all things a reader of a v1→v2 diff is asking about.
        "fact_changes": _fact_changes(conn, earlier, later),
        # §12.4/D63: what the two decks actually SAY differently, passage by passage and by
        # layer. `fact_changes` says a budget moved and `record_changes` says a field did;
        # neither says the third-colourway paragraph is gone, which is what somebody comparing
        # two versions is looking at. The chunks were already stored per version and nothing
        # had read them against each other.
        "passages": _passages_changed(conn, earlier, later),
        # §12.4/D62: what the CLIENT asked for, from the comments on the earlier deck.
        # "In the Colombia case 'corrections' meant the client's tracked comments at least as
        # much as the library's findings" — and a diff that reports only which of its OWN
        # findings were addressed is reporting on itself. What the client actually wrote when
        # they sent the deck back was being read as nothing.
        #
        # Reported beside `adopted` and never merged into it: one is what this product
        # decided, the other is what the client instructed, and blending them would let the
        # library take credit for somebody else's sentence.
        "what_the_client_asked_for": _what_the_client_asked_for(conn, earlier),
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
                # D61/§7.8: `heuristic`, not `judged`. Nobody judged this — a threshold did.
                **entry, "basis": "heuristic", "match": "text", **_reread(finding, again),
                "similarity": round(_looks_like(finding.get("finding"),
                                                again.get("finding")), 2),
                "raised_again_as": again.get("finding"),
                "caveat": "Matched on how alike the two read, not on either review saying "
                          "they are the same problem. Check both before telling anyone a "
                          "correction was not taken.",
            })
            continue

        # §10.2/D59: somebody ANSWERED this one. `no_longer_raised` is the "we cannot tell"
        # bucket and all three of its caveats below end in "though nobody recorded it" —
        # which is D59's complaint exactly, and the row asks for a way to record it. So an
        # answered finding must leave the bucket, or `answer_finding` writes a field nothing
        # reads: a tool whose output nobody consumes cannot fail, which is D116's shape one
        # level down.
        #
        # `stated`, never `computed`. Every other entry in `adopted` is the SERVER matching a
        # `resolved` row by id; this is somebody's word, and a reader who cannot tell those
        # apart will eventually cite one as the other.
        settled_answer = (finding.get("settled") or {}).get("answer")
        if settled_answer == "deliberate":
            # Not `adopted` — nothing was adopted; they kept it and said why. And not the
            # "we cannot tell" bucket either, whose three caveats below all end "though
            # nobody recorded it", which this answer disproves in the user's own words.
            result["no_longer_raised"].append({
                **entry, "basis": "stated", "settled": finding["settled"],
                "caveat": (
                    f"Not raised again, and it would not be: {finding['settled']['said_by']} "
                    f"recorded that this was deliberate — “{finding['settled']['note']}”. "
                    f"That is their word, not a comparison of the two versions."),
            })
            continue
        if settled_answer in ("fixed", "does_not_apply", "misread"):
            result["adopted"].append({
                **entry, "basis": "stated", "settled": finding["settled"],
                "now": finding["settled"]["note"],
                "caveat": (
                    f"{finding['settled']['said_by']} recorded this as "
                    f"{_SETTLED_AS[settled_answer]} "
                    f"directly, rather than it being matched to anything in the later "
                    f"judgment. It is their word, not a comparison of the two versions."),
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
    # `computed` only. §10.2/D59 put a second kind of entry in `adopted` — a finding somebody
    # settled DIRECTLY with `answer_finding` — and that is a statement about one record's
    # finding, not about the two records being versions of each other. Counting it here would
    # make this offer's own `why` false ("a judgment of one names a finding from the judgment
    # of the other by id") and would hide a record from every future search on evidence that
    # never mentioned the other record.
    linked = linked or any(entry.get("basis") == "computed" for entry in diff["adopted"])
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


def _fact_changes(conn, earlier: str, later: str) -> list:
    """Computed facts whose STATUS changed between two versions (D58, §7.1).

    Status, not value: "the budget went from 40,000 to 50,000" is a change in the brief that
    a reader can see for themselves, while "the budget disappeared" is a change in what the
    brief can be judged on. The first is content and the second is coverage, and only the
    second is what a diff of computed facts is for.
    """
    before, after = facts.for_campaign(conn, earlier), facts.for_campaign(conn, later)
    # A missing record is not a version where every fact changed. `for_campaign` returns `{}`
    # for an id that resolves to nothing, and comparing that against a real record reported
    # the whole checklist as having appeared out of nowhere.
    if not before or not after:
        return []
    changed = []
    for code in sorted(set(before) | set(after)):
        was = before.get(code, {}).get("status")
        now = after.get(code, {}).get("status")
        if was == now:
            continue
        changed.append({
            "code": code, "was": was, "now": now, "basis": "computed",
            "what_it_means": (after.get(code) or before.get(code) or {}).get("what_it_means"),
        })
    return changed


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
# §12.4/D38: `never_ran` sits beside `not_yet_run` and is a different statement. A cancelled
# campaign has not "not yet run" — it will never run, and the difference is whether anybody
# should ever come back for its results. Ranked last with it, because neither is a hole in the
# evidence somebody can close.
_EVIDENCE_ORDER = ("no_outcomes", "single_example", "measured", "not_yet_run", "never_ran")


def _citation_concentration(conn, campaign_ids: list) -> dict:
    """How concentrated the citations across these records are (D65).

    `cited_share_top` is the share of citing judgments that named the single most-cited
    record — None when nothing has been judged, because zero judgments is not "one record
    carries everything" and 1.0 would read as the worst possible state when the truth is
    that there is no state yet. That distinction is the same one §6.4 needed three codes for.
    """
    if not campaign_ids:
        return {"cited_share_top": None, "never_cited": []}
    citations: dict = {cid: 0 for cid in campaign_ids}
    total = 0
    for cited_ids in store.citations(conn):
        cited = set(cited_ids) & set(campaign_ids)
        if not cited:
            continue
        total += 1
        for cid in cited:
            citations[cid] += 1
    if not total:
        return {"cited_share_top": None, "never_cited": list(campaign_ids)}
    return {
        "cited_share_top": max(citations.values()) / total,
        "never_cited": [cid for cid, n in citations.items() if not n],
    }


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
            # §12.4/D38 added two statuses and this reader was not swept, so a CANCELLED
            # campaign was reported as `not_yet_run` — a false statement on a surface built
            # to say what the library's evidence is worth, and one that invites somebody to
            # go and collect results for a campaign that was called off.
            "evidence": ("never_ran" if stage == "cancelled"
                         else "not_yet_run" if stage != "concluded"
                         else "no_outcomes" if not with_outcomes
                         else "single_example" if len(rows) == 1
                         else "measured"),
            "campaign_ids": [c["id"] for c in rows][:5],
            # D65, owed to §6.6. `evidence` above counts what the cell HOLDS; these two count
            # what judgments have actually leaned on. A cell of six measured campaigns where
            # every verdict cites the same one reads `measured` and has a real depth of one —
            # "one example carrying the weight" is a fact about judgments, and the schema has
            # held it in `cited_ids` all along with nothing reading it.
            **_citation_concentration(conn, [c["id"] for c in rows]),
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
                 "results yet — and `never_ran` is a campaign that was cancelled, which "
                 "nobody should go looking for results for at all."),
    }


def _offer_to_measure(conn, unmeasured: dict) -> list[dict]:
    """The single thing that would lift a wholly unmeasured library out of it.

    Falls back to the first-run path only when there is nothing to measure yet — a library
    with no concluded campaign cannot record results for one.
    """
    candidates = sorted(unmeasured.values(), key=lambda c: (-len(c["markets"]), c["title"]))
    if not candidates:
        # `first_steps`, not `readiness(conn)["shortest_path"]`. Same list, and readiness now
        # offers coverage (D67), so going back through it would be coverage → readiness →
        # coverage without end. Reading the shared path directly is both the D76 rule and
        # what keeps this a tree.
        return first_steps(conn)
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

    Lives in `store` now, because §8.3 needed the same answer and `metrics` cannot import
    `core` (cycle) — so it was about to become the fifth implementation of one question, which
    D55 and D88 both record as how these drift. It had already drifted once: §8.3's first
    version read `market or region` and reported a campaign that literally ran in MX and CO as
    covering zero markets.
    """
    return store.markets_of(campaign)


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


def load_declared_corrections(conn, *, confirmed_by: str) -> dict:
    """Put the standing corrections from the customer's own rulebook in force (§12.3/D108).

    **Why they need a route at all.** They arrive with provenance — a deck and a slide — and no
    `campaign_id` in this library, so `learning.gate` refused them forever: seen in 0
    campaigns, needs 3. The item built to give them "somewhere to live and grow" gave them
    somewhere they could be STORED and never applied, which is invisible from the row itself.

    The counting gate is for a rule this library INFERRED from what it watched recur — breadth
    across markets is what makes that guess safe. A rule the customer wrote in their own file
    is not a guess, and no number of campaigns makes their own rule truer. What it still needs
    is a person's name against promoting it, because §8.2 spent this loop's only human step on
    "who says so" and a file is not somebody.

    Idempotent, because the rulebook is read at every start and a customer edits it: a loader
    that appended would turn one rule into five over a week of restarts, each looking like
    independent confirmation of the same thing.
    """
    import corrections
    import rulebook

    who = learning.require_a_person(confirmed_by)
    try:
        declared = rulebook.corrections()
    except ValueError as bad:
        return {"loaded": 0, "already": 0, "basis": "computed",
                "what_it_means": f"No corrections were loaded: {bad}"}
    if not declared:
        return {"loaded": 0, "already": 0, "basis": "computed",
                "what_it_means": (
                    f"Your rulebook declares no standing corrections, so none were loaded. "
                    f"They go in {rulebook.overlay_path()} under `corrections:`, each with "
                    f"the rule itself and where it came from.")}

    version = rulebook.overlay() or rulebook.version()
    loaded, already, follow_on = [], 0, []
    for entry in declared:
        existing = corrections.find(conn, entry["text"])
        if existing and existing["status"] == "expected":
            already += 1
            continue
        if existing:
            # ALREADY HEARD, NOT YET IN FORCE — the likeliest case for a house rule, and it
            # failed silently. The library had recorded the same rule provisionally from a
            # deck, the loader counted it "already on file" and never promoted it, and the
            # offer went quiet because it used the same lookup. True sentence, rule never
            # applied, nothing said.
            correction_id = existing["correction_id"]
        else:
            correction_id = corrections.note(
                conn, text=entry["text"], campaign_id=None,
                # WHERE it came from, naming the file. A judgment citing this must not read
                # identically to one citing a rule the library inferred and a person confirmed
                # after three campaigns: those are different claims about how much is known.
                provenance=f"{entry['provenance']} (declared in rulebook {version})",
            )["correction_id"]
        promoted = corrections.graduate(
            conn, correction_id, confirmed_by=who, from_rulebook=version,
            # No market list means EVERYWHERE, which is what a house rule is.
            everywhere=not entry["markets"])
        follow_on.extend(promoted.get("next_actions") or [])
        loaded.append(correction_id)

    # DRIFT: rules that stand under a rulebook that no longer declares them. A customer
    # edits the wording or deletes a line, and the old rule keeps applying — eleven standing
    # for ten declared, neither wrong on its face, and the offer goes quiet because everything
    # declared is on file. Reported rather than retired automatically: a typo in a file must
    # not silently withdraw a rule that saved judgments already cite.
    declared_now = {entry["text"].casefold() for entry in declared}
    stale = [row["text"] for row in corrections.all_of_them(conn)
             if row.get("status") == "expected"
             and row["text"].casefold() not in declared_now
             and any("(declared in rulebook " in (sighting.get("provenance") or "")
                     for sighting in corrections.sightings(conn, row["id"]))]

    return {
        "loaded": len(loaded), "already": already, "basis": "computed",
        "rulebook": version, "confirmed_by": who,
        **({"no_longer_declared": stale} if stale else {}),
        # What promoting a rule offers next. Going around `corrections.graduate` dropped
        # these silently, along with the replay entry it writes.
        **({"next_actions": actions.trim(follow_on)} if follow_on else {}),
        "what_it_means": (
            f"{len(loaded)} standing correction(s) from your rulebook ({version}) are now in "
            f"force, on {who}'s confirmation"
            + (f"; {already} were already on file." if already else ".")
            + (f" {len(stale)} rule(s) that came from a rulebook are still in force and are "
               f"NO LONGER IN YOUR RULEBOOK — edited or deleted since they were loaded: "
               f"{'; '.join(repr(t) for t in stale[:3])}"
               + (f" and {len(stale) - 3} more" if len(stale) > 3 else "")
               + ". They were not withdrawn automatically, because judgments already cite "
                 "them and a typo in a file should not silently remove a rule. Retire each "
                 "one you meant to drop."
               if stale else "")
            + " They did not go through the three-campaign gate: that test is for a rule this "
              "library inferred from what it watched recur, and these are rules you wrote "
              "down. Each carries your rulebook as its provenance, so a judgment citing one "
              "says where it came from."),
    }


def readiness(conn) -> dict:
    """What this library can and cannot do yet, and the shortest path to more.

    The path is the review's own prescription, in its order, because it is a path and not a
    menu: one brief you liked, one you did not, your guidelines as reference material. The
    contrast is the point — two briefs somebody liked teach nothing about the axis they are
    asking the product to judge on.

    D74: the third step used to say "as the rulebook", and ticked itself when ANY `reference`
    record existed — so a competitor's deck filed as reference material told the customer they
    had put their rules in force. The rulebook is a file now, and the two are separate facts.
    """
    facts_about_the_axis = _axis_facts(conn)
    records = facts_about_the_axis["records"]
    campaigns = facts_about_the_axis["campaigns"]
    measured = store.campaigns_with_actual_metrics(conn)

    has_reference_material = facts_about_the_axis["has_reference_material"]
    liked_records = facts_about_the_axis["liked_records"]
    disliked_records = facts_about_the_axis["disliked_records"]
    liked = facts_about_the_axis["liked"]
    disliked = facts_about_the_axis["disliked"]
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

    if has_reference_material:
        # Uploaded guidelines are STILL retrieved by similarity — §12.1 pinned the rulebook
        # FILE, not every reference record in the library. Saying otherwise here would move
        # the old overclaim rather than remove it.
        can.append({"code": "rulebook_on_file",
                    "what": "Cite your uploaded guidelines when they happen to be among the "
                            "most similar records retrieved for a brief. (The rules in the "
                            "rulebook file are a separate thing and always apply — see "
                            "`check_against_rules`.)"})
    # §12.1. This row lived on `cannot` and named this item by number as the work that would
    # fix it. The rules in the rulebook are now put in front of every judgment in full,
    # whatever the brief is about, so "this breaks your own rule" is a claim this product can
    # make — about the rules it HOLDS.
    #
    # The boundary is the whole point of the wording. A guideline the customer never wrote
    # down is not in the rulebook, and a product implying otherwise would have replaced one
    # overclaim with another, in the tool whose entire job is saying what it cannot do.
    applied = rulebook.applied()
    if applied["rules_applied"]:
        can.append({
            "code": "check_against_rules",
            "what": f"Check a brief against the {applied['rules_applied']} rule(s) in the "
                    f"rulebook ({applied['version']}) and say \u201cthis breaks your own "
                    f"rule\u201d, citing the rule by its id. They are put in front of every "
                    f"judgment in full rather than retrieved by similarity, so a rule applies "
                    f"whether or not the brief resembles it.",
            "bounded_by": "Only rules written down in the rulebook. A guideline nobody has "
                          "written down is not one this can check, and an uploaded guidelines "
                          "DOCUMENT is retrieved by similarity like any other record.",
        })
    else:
        # THE MECHANISM SHIPPING IS NOT THE CAPABILITY EXISTING. This row went onto `can` the
        # moment the rulebook file existed, so a fresh install advertised "I can say this
        # breaks your own rule" while holding no rules — the confident first-run claim this
        # whole tool exists to prevent, arriving through the item meant to remove it.
        cannot.append({
            "code": "check_against_rules",
            "what": f"Check a brief against a rule. The rulebook ({applied['version']}) is in "
                    f"force and applies to every judgment, and no rules are written in it. "
                    f"Guidelines uploaded as records are retrieved by similarity like "
                    f"anything else, so a guardrail that is not retrieved is not a guardrail "
                    f"\u2014 it can say \u201cthis differs from what you did in Peru\u201d, "
                    f"which invites an argument, but not \u201cthis breaks your own "
                    f"rule\u201d, which does not.",
            # The OVERLAY's path, not the shipped file's. This named `rulebook.yaml` with no
            # directory — and the shipped file's own header says DO NOT EDIT THIS FILE,
            # because the next installer replaces it. The one instruction the product gave
            # would have sent a customer to write rules they lose on an upgrade.
            "needs": f"your rules written into {rulebook.overlay_path()}, each with an id, a "
                     f"severity and a reason. Every one of them then reaches every judgment "
                     f"whatever the brief is about. Restart the server to load them.",
        })

    path = _shortest_path(liked, disliked, has_reference_material)
    if not campaigns:
        stage = "empty"
        # Everything else on `can` is a claim about what the LIBRARY holds, and an empty
        # library holds nothing. The rulebook is the exception and the reason §12.1 put it in
        # a file: it ships with the product, so its rules apply to the very first brief
        # somebody sends — which is the one they are most likely to be judging alone.
        can = [row for row in can if row["code"] == "check_against_rules"]
    elif len(campaigns) < 2:
        stage = "first_records"
    elif not with_outcomes or path:
        # `working` means the path is walked AND something is measured. Measured-alone
        # reported `working` with all three steps outstanding, so the guidance stopped being
        # attached exactly while it was still needed.
        stage = "thin"
    else:
        stage = "working"

    # §10.6: "the first interaction of a session" is the review's own second moment for the
    # queue, and this is that surface. D54 adds the other thing worth saying here — what is
    # missing — for the same reason: it was documented as "call it when the user asks",
    # which is a tool nobody calls.
    #
    # `coverage` is NOT offered here, though D67 asked for it at session start too. Review
    # measured what a mature library's session start then looked like: three offers, two of
    # them read-only reports, with the one act that would actually improve the library two
    # steps away inside `gaps()`. Coverage is the picture BEHIND the gaps — the less
    # actionable of two views of the same fact — so spending a scarce slot on it at the
    # moment somebody opens a session is spending it on the wrong one. It stays offered from
    # `list_campaigns`, which is where D67's own sentence put it.
    waiting = feedback.waiting(conn)
    set_aside = _ranked_gaps(conn).get("set_aside") or []
    offers = actions.trim(actions.offer_the_queue(waiting) + gaps_offer(conn)
                          + stale_answers_offer(conn)
                          + declared_corrections_offer(conn)
                          + authorship_backfill_offer(conn))
    return {
        "stage": stage,
        "campaigns": len(campaigns),
        "with_outcomes": len(with_outcomes),
        # The FILE, which ships with the product — so this is true of a fresh install, and
        # the question "are the customer's own rules in force" is `rulebook`, below.
        "has_rulebook": True,
        "rulebook": rulebook.applied(),
        # D74: reference material on file, which is a different fact and was reported as this
        # one. A competitor teardown is reference material and is not anybody's guidelines.
        "has_reference_material": has_reference_material,
        "can": can,
        "cannot": cannot,
        "shortest_path": path,
        "waiting_on_feedback": waiting,
        # §10.2/D53: what somebody has told this library to stop ranking. HERE as well as on
        # `gaps()`, because this is the surface whose whole job is to say what the library
        # cannot do — reporting a rosier top gap with no hint that a worse one was silenced
        # is the product overstating its own coverage, which is the failure the review that
        # started all of this was written about, arriving through the door D53 opened.
        "set_aside": set_aside,
        **({"next_actions": offers} if offers else {}),
        "note": ("Say the stage and what it cannot do yet before giving any judgment from a "
                 "library this size — a confident, evidence-free verdict is the thing a new "
                 "user will believe. `shortest_path` is ordered: it is a path, not a menu."
                 + (f" {waiting} campaign(s) are waiting on feedback; say so and offer the "
                    f"queue rather than waiting to be asked for it."
                    if waiting else "")),
    }


def _axis_facts(conn) -> dict:
    """What the library holds on the axis the first-steps path is about (§10.6/D76).

    One definition, read by both surfaces that answer "what do I add first". They each had
    their own before: `readiness` computed this and ordered the answer, `gaps` carried
    `to_first_upload()` — an unordered copy of the first two steps — on `library_is_empty`.
    Two implementations of one question is how D55 and D88 both begin, and here the two
    could disagree about the SAME library while both were internally consistent, which is
    the version of that failure nobody notices.
    """
    superseded = store.get_superseded_campaign_ids(conn)
    records = [c for c in store.list_campaigns(conn) if c["id"] not in superseded]
    campaigns = [c for c in records if c.get("record_type") not in ("reference", "stub")]

    # Case-folded, because tags are freeform and stored as typed while the store folds them
    # when filtering. "Liked" left a marketer who had done exactly what the path asked being
    # told forever to add a campaign they liked.
    def reactions_of(campaign):
        return {str(t.get("value") or "").strip().lower()
                for t in (campaign.get("tags") or [])}

    reactions = {r for c in campaigns for r in reactions_of(c)}
    return {
        "records": records,
        "campaigns": campaigns,
        # D74: this counts REFERENCE RECORDS, and it used to be called `has_rulebook` — so a
        # competitor's deck filed as reference material made the product report that the
        # customer's guidelines were on file, and `readiness` then ticked the rulebook step of
        # its own shortest path. §12.1 makes the two separable: the rulebook is a declared
        # artefact (a file), and no record is ever mistaken for it.
        "has_reference_material": any(c.get("record_type") == "reference" for c in records),
        # Per RECORD, not pooled: a single campaign tagged both `liked` and `not_liked` used
        # to satisfy the axis on its own, and "the library holds both" was then technically
        # true and substantively false. The contrast this product reasons from is between
        # records, and one record cannot be the counter-example to itself.
        "liked_records": [c for c in campaigns if "liked" in reactions_of(c)],
        "disliked_records": [c for c in campaigns
                             if reactions_of(c) & {"not_liked", "not liked"}],
        "liked": "liked" in reactions,
        # `mixed_reaction` is deliberately NOT a dislike. The item's own rationale asks for
        # "one you did not like", and a mixed reaction is not that contrast — counting it
        # would tell the user the axis works when it does not.
        "disliked": "not_liked" in reactions or "not liked" in reactions,
    }


def first_steps(conn) -> list[dict]:
    """The ordered path, computed once and read by everything that shows it (§10.6/D76)."""
    axis = _axis_facts(conn)
    return _shortest_path(axis["liked"], axis["disliked"], axis["has_reference_material"])


def _shortest_path(liked: bool, disliked: bool, has_reference_material: bool) -> list[dict]:
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
    if not has_reference_material:
        steps.append(actions.action(
            "Add your brand guidelines as reference material",
            "upload_campaign",
            # D74: it no longer says "as the rulebook". The rulebook is a file that ships with
            # the product and already applies; a guidelines DOCUMENT is an ordinary record,
            # retrieved by similarity, and worth uploading for what it actually buys — a
            # judgment can quote it when it ranks. Calling it the rulebook told people they
            # had done the step that puts rules in force, which they had not.
            why="A judgment can then quote your guidelines when they are among the records "
                "retrieved for a brief. (Rules that must apply to EVERY brief go in the "
                "rulebook file, which already ships and already applies.)",
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
            _add_vector(conn, chunk["id"], vec)
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
                _add_vector(conn, asset["id"], vec, space="asset")
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
        _settle_indexing_notice(conn, cid)

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
    # §11.6: never a bare null. `matched_author: null` is what a judgment actually sees, and
    # it reads as "nobody" for four different situations — a format that carries no author, a
    # person who withheld one, a record nobody looked at, and text that is not a comment at
    # all. §2.5's distinction between the deck talking to itself and a client objecting is
    # exactly what the null erased.
    said = store.author_of(source)
    return {"matched_anchor": source.get("anchor"),
            "matched_author": said["author"],
            **({"matched_author_why": said["why"]} if said.get("why") else {}),
            **({"matched_author_means": said["what_it_means"]} if said.get("why") else {}),
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
    # §7.2: the campaign-level rollup re-sorts, so a stable tie-break in the vector search
    # is undone here unless it is repeated. Equal similarity breaks on the campaign id, which
    # is the same rule at both levels and stable across machines and across an insert.
    ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))[:top_k]

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
                        "structured": m["structured"],
                        # §9.8: the caveat travels WITH the metric. A projection that drops it
                        # here would mean the one surface a judgment is actually written from
                        # is the one surface that never carries it.
                        "confounded": m.get("confounded", False),
                        **({"confounded_by": [
                                {"id": e["id"], "description": e["description"],
                                 "why": e["why"], "attribution": e["attribution"]}
                                for e in m["confounded_by"]],
                            "what_it_means": m.get("what_it_means", "")}
                           if m.get("confounded") else {})}
                       for m in sorted_metrics]
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
            # §9.5: the caveat travels WITH the citation, not somewhere a reader has to go and
            # look. A result from a campaign that drifted is evidence that something worked and
            # the brief may not have been it — and a result from one nobody checked is neither.
            "execution": _execution_note(conn, c["id"]),
            # §9.6, on the same principle and for the same reason. A reader looking at this row
            # would otherwise see nothing about the port closure that ran through half the
            # flight, and could only find it by independently calling a tool nothing
            # recommends. Counts and kinds rather than the events themselves: an evidence row
            # is already long, and "one natural disaster" is what changes how the number is
            # read. §9.8 changes what this key CONTAINS rather than having to establish, across
            # four call sites, that it exists.
            "context": _context_note(conn, c["id"]),
            # §11.5, on the same principle: the caveat travels with the citation. A judgment
            # weighing this precedent needs to know it was contested — told only "liked", it
            # cites a two-to-one split as though it were unanimous, which is the whole of what
            # append-only reactions are for.
            **({"disagreement": c["disagreement"]} if c.get("disagreement") else {}),
        })
    return evidence


def _date_columns(structured) -> set:
    """Which of these columns this library READ as dates, moved the window or not."""
    return {key for key in (structured or {})
            if context.window_from_columns({key: (structured or {})[key]})
            or key.strip().lower().replace(" ", "_") in context.DATE_COLUMNS}


def _persisted(conn, campaign_id: str, warnings: list) -> list:
    """Keep the warnings that need a PERSON, against the record (§10.1/D20).

    A warning lives for exactly one response, so the thing the library most wanted somebody to
    act on was the thing it forgot fastest — and §10.1's definition of open is precisely "the
    library is missing something a human has to supply". `blocked` and `degraded` are the two
    severities that mean somebody has to do something; the rest are information.

    Never lets a notice break the write it is attached to: a warning about a degraded upload
    that prevented the upload would be a worse failure than the one it describes.
    """
    for warning in warnings or []:
        if warning.get("severity") not in ("blocked", "degraded"):
            continue
        try:
            store.record_notice(conn, code=warning["code"], campaign_id=campaign_id,
                                detail=warning.get("affects") or warning.get("detail"))
        except Exception:                    # noqa: BLE001
            pass
    return warnings


def _waiting_elsewhere(conn, campaign_id: Optional[str]) -> int:
    """How many OTHER campaigns are waiting on feedback (§10.6)."""
    return feedback.waiting(conn, apart_from=campaign_id)


def _window_from_the_workbook(conn, campaign_id: str, structured) -> Optional[dict]:
    """Set the campaign's window from a spreadsheet's own date columns (§9.6/D119).

    Never over a window somebody entered. Typed dates are the more reliable claim, and a
    workbook cell replacing them is the correction being undone by the thing it corrected.
    """
    stated = context.window_from_columns(structured or {})
    if not stated:
        return None
    record = store.get_campaign(conn, campaign_id) or {}
    # A window somebody TYPED is the more reliable claim, and a spreadsheet cell replacing it
    # is the correction being undone by the thing it corrected. `window_source` records which
    # kind is on file so this can tell them apart — a window this function wrote is one it may
    # widen, and one a person wrote is not.
    if record.get("starts_on") and record.get("window_source") != "workbook":
        return None
    widened = context.widen(
        {"starts_on": record.get("starts_on"), "ends_on": record.get("ends_on")}
        if record.get("starts_on") else None, stated)
    if not widened:
        # The row sat inside the window already. Reporting a change that did not happen is as
        # false as reporting a skip that did not happen.
        return None
    # Through `update_campaign`, which is where a window is validated — `set_campaign_window`
    # bypassed every check, so a spreadsheet could store 2026-03-31 to 2026-03-01 and the
    # product would print that backwards inside "nothing was going on".
    store.update_campaign(conn, campaign_id, starts_on=widened["starts_on"],
                          ends_on=widened["ends_on"] or "", window_source="workbook")
    return {**widened,
            "basis": "stated",
            "what_it_means": (
                f"This campaign's window is now {widened['starts_on']} to "
                f"{widened['ends_on'] or 'open-ended'}, from the "
                f"{', '.join(widened['from'])} column(s) in what you sent. A workbook is "
                f"normally one row per month, so it widens as the rows arrive. That is what "
                f"decides which recorded events overlap it — say so if it is wrong.")}


def _calendar_clash(conn, *, campaign_id, proposal_text, starts_on, ends_on,
                    markets: list) -> dict:
    """§9.7's check, against the subject whichever way it arrived.

    With a stored record, its own window and its own markets — §7.2's split, where the record
    answers "which brief is this". Without one, what the caller named, falling back to the
    proposal's own dates.
    """
    record = store.get_campaign(conn, campaign_id) if campaign_id else None
    if record:
        # The record's OWN window, basis and all. Passing only the dates threw the basis away
        # and `_window_for_the_check` then labelled a window read out of a competitor's launch
        # date `stated` — so `campaign_context` hedged ("may have run during") while
        # `prepare_evaluation` asserted a clash on the same record with the same dates. Two
        # surfaces describing one record differently is the defect; mislabelling the basis is
        # the house rule it breaks.
        window = context.window_of(conn, campaign_id)
        return context.clash_check(
            conn, window=window, markets=sorted(context._market_names(record)),
            from_a_record=True,
            # `deck_text` too: a plan that addresses the World Cup does it in the deck, and
            # the silence test reading only `detail` would call that plan silent.
            text=" ".join(filter(None, [record.get("detail"), record.get("deck_text")])))
    return context.clash_check(conn, starts_on=starts_on, ends_on=ends_on, markets=markets,
                               text=proposal_text)


def _say_the_calendar(clash: dict) -> str:
    """What the model is told about a clash (§9.7).

    Silent unless there is one. It ASKS rather than judging: overlapping a fixed date is the
    point of some campaigns and the ruin of others, and §9.4's lesson — that treating every
    deviation as a defect teaches this library to punish improvement — applies unchanged one
    item later.
    """
    if (clash or {}).get("status") != "present":
        return ""
    said = ("`calendar_clash` in `computed` says this window runs into fixed dates already on "
            "the calendar. It is a fact about timing and NOT a criticism: launching into "
            "Black Friday is the point of some campaigns and the ruin of others, and this "
            "library cannot tell which. ")
    if clash.get("unaddressed"):
        said += (f"What IS worth raising is the silence — the proposal does not mention "
                 f"{', '.join(clash['unaddressed'])}. Ask whether that is deliberate rather "
                 f"than assuming it is an oversight, and do not turn it into a blocking "
                 f"finding on its own. ")
    said += ("Any clash marked `seeded` came from the calendar shipped with this product, not "
             "from this customer — say so if you rest on one, because a shipped date can be "
             "wrong for their market. Each carries a `certainty` and a `certainty_note`: "
             "`fixed` does not move, `announced` is published and does get changed, "
             "`observed` is set by sighting and differs between neighbouring countries, and "
             "`seasonal` is not a date at all but an onset that moves by weeks. Do not quote "
             "a `seasonal` or `observed` range as though it were a fixed date. ")
    if clash.get("not_checked_for_silence"):
        said += (f"{clash['not_checked_for_silence']} of these are events this customer "
                 f"recorded, which have no short name to look for — whether the proposal "
                 f"addresses those is not something the check can tell you, so read them. ")
    if clash.get("markets_not_covered"):
        said += (f"The shipped calendar has no entries for "
                 f"{', '.join(clash['markets_not_covered'])}, so anything found there came "
                 f"from this customer's own records. ")
    return said


def _confounded_note(conn, campaign_id: str) -> dict:
    """Whether this campaign's measured outcomes ran through anything (§9.8)."""
    record = store.get_campaign(conn, campaign_id) or {}
    actual = [m for m in record.get("metrics") or [] if m["metric_type"] == "actual"]
    if not actual or not actual[0].get("confounded"):
        return {"confounded": False}
    return {"confounded": True,
            "confounded_by": [{"id": e["id"], "description": e["description"],
                               "attribution": e["attribution"]}
                              for e in actual[0]["confounded_by"]]}


def _say_the_language(computed: dict) -> str:
    """Tell the model the checks did not run, and why (D94).

    §7.1's own instruction is that the model treat computed facts as established and not
    re-derive them. Handed five `unchecked` with no sentence, it does exactly what it did
    before the language signal existed — reports the brief as missing a budget — which is the
    finding this exists to stop, arriving through the instruction meant to prevent it.
    """
    language = computed.get("language") or {}
    if language.get("checks_apply", True):
        return ""
    return (f"THE MECHANICAL CHECKS DID NOT RUN: this brief is not in English (it reads as "
            f"{language.get('language')!r}) and every one of them is an English pattern. "
            f"They are reported as `unchecked`, which is NOT a finding that the brief lacks "
            f"anything — do not say it has no budget, no dates or no channels on that basis. "
            f"Read the brief yourself for those facts, and say plainly that the automatic "
            f"checks could not be applied. ")


def _say_the_disagreement(evidence: list) -> str:
    """What the model is told about precedents people disagreed about (§11.5).

    Structure alone does not survive a summary — §7.1's lesson, and the reason every other
    computed fact in this note has a sentence beside it. A `disagreement` block the model has
    to infer the significance of is one it flattens into "they liked it", which is the exact
    loss this item exists to prevent.

    Silent when there are none, for §9.5's reason: a standing paragraph about disagreement on
    a library with none is the note that fires on everything and is read on nothing.
    """
    split = [e for e in evidence if e.get("disagreement")]
    if not split:
        return ""
    return (f"{len(split)} of these campaigns are ones people DISAGREED about, and both views "
            f"are on file. Cite the split, not a side: a campaign two people saw differently "
            f"is stronger evidence about this client's taste than one everybody liked, and "
            f"quoting it as unanimous throws that away. This library will not tell you who "
            f"was right — authority order would be configured in the rulebook and nothing is "
            f"configured yet, so preferring the later view or the grander job title would be "
            f"authority nobody granted it. ")


def _say_the_confounded(evidence: list) -> str:
    """What the model is told about outcomes that ran through something (§9.8).

    Silent when none are, for §9.5's reason three items over: a standing paragraph about
    confounding on a library with none is the note that fires on everything.
    """
    marked = [e for e in evidence
              if any(m.get("confounded") for m in e.get("metrics") or [])]
    if not marked:
        return ""
    return (f"{len(marked)} of these campaigns carry outcomes marked CONFOUNDED: something "
            f"was going on in that market at the same time. They STILL COUNT — they are real "
            f"measured results and nothing here down-weights them — but they are not CLEAN "
            f"evidence, so do not quote one as though nothing else was happening. It is an "
            f"overlap in time and NEVER a cause: \u201csell-through was down and there was an "
            f"earthquake\u201d is not evidence the earthquake did it. If a person has "
            f"attributed it, their note is on the row and is theirs, not a measurement. ")


def _context_note(conn, campaign_id: str) -> dict:
    """What else was going on, as a citation carries it (§9.6).

    The same shape decision as `_execution_note`: `status` first, so `nothing_to_check`
    survives the trip and a reader can tell "we looked and nothing was going on" from "this
    campaign cannot be checked at all".
    """
    try:
        linked = context.for_campaign(conn, campaign_id)
    except (ValueError, RuntimeError):
        return {"status": "nothing_to_check", "basis": "computed", "events_total": 0,
                "by_kind": {}, "what_it_means": "This campaign's context could not be read."}
    return {"status": linked["status"], "basis": "computed",
            "events_total": linked.get("events_total", 0),
            "by_kind": linked.get("by_kind", {}),
            "what_it_means": linked["what_it_means"]}


def _say_the_context(evidence: list) -> str:
    """What the model is told about `context` on each citation (§9.6).

    Silent when no cited campaign has any, for §9.5's reason one item over: a standing
    paragraph about context events on a library holding none is the note that fires on
    everything.
    """
    carrying = [e for e in evidence if (e.get("context") or {}).get("events_total")]
    if not carrying:
        return ""
    total = sum(e["context"]["events_total"] for e in carrying)
    return (f"`context` on each piece of evidence says what ELSE was going on in that "
            f"market while it ran — {total} recorded event(s) across {len(carrying)} of "
            f"these campaigns. It is an overlap in time and NEVER a cause: \u201csell-through "
            f"was down and there was an earthquake\u201d is not evidence the earthquake did "
            f"it, and a disappointing number will go looking for the nearest event to "
            f"explain it. Say what ran during what. A `nothing_to_check` there means the "
            f"campaign has no window or no market, so nothing could be matched — not that "
            f"nothing was happening. ")


def _context_offer(conn, campaign_id: str, fields: dict) -> dict:
    """"You just made this answerable, and here is what it says" (§9.6, D116's rule).

    Only on a write that SET a window, and only when something overlaps. The link is computed
    rather than stored, so nothing on screen changes when the dates go in — without this, a
    marketer who has just recorded when a campaign ran has no way to discover that the library
    already knew a port was shut through half of it.
    """
    if not (fields.get("starts_on") or fields.get("ends_on")):
        return {}
    try:
        linked = context.for_campaign(conn, campaign_id)
    except ValueError:
        return {}
    if linked["status"] != "checked" or not linked["events"]:
        return {}
    return {
        "context_events_found": linked["events_total"],
        "next_actions": actions.trim([actions.action(
            f"See the {linked['events_total']} recorded event(s) this campaign ran through",
            "campaign_context",
            why="Now it has a window, it matches what was already on record for its market. "
                "Nothing was linked by hand and nothing needs to be — and an overlap is a fact "
                "about timing, never about cause.",
            consent="ask", campaign_id=campaign_id)]),
    }


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


# Editing either of these changes what the record SAYS, so the index has to be rebuilt.
# Everything else on a record is metadata: a status or a tag is not a content change, and
# re-embedding for one would turn every bulk edit into a re-index of the library.
_CONTENT_FIELDS = ("title", "detail")


def _rebuild_body_index(conn, campaign_id: str) -> dict:
    """Re-chunk and re-embed a record's BODY from what is stored on the row.

    One implementation, used by an edit that changed the text (D80) and by a deck attached to
    a record that already existed (§12.4/D39). Two copies of "rebuild the index" would agree
    on the day they were written and drift by the next item — which is the failure this
    codebase names most often.

    Commentary is a different layer and is never rebuilt here: it is not derived from the row
    columns, and its chunks carry an author and an anchor that nothing else can reconstruct.
    """
    record = store.get_campaign(conn, campaign_id)
    summary = "\n\n".join(p for p in (record["title"], record.get("detail")) if p)
    units = (record.get("deck_text") or "").split("\n\n")
    texts = chunking.pack(([summary] if summary else []) + [u for u in units if u.strip()])

    old_ids = [r["id"] for r in conn.execute(
        "SELECT id FROM campaign_chunks WHERE campaign_id = ? AND kind = 'body'",
        (campaign_id,)).fetchall()]
    vectorstore.delete_many(conn, old_ids)
    store.forget_vector_models(conn, old_ids)
    conn.execute("DELETE FROM campaign_chunks WHERE campaign_id = ? AND kind = 'body'",
                 (campaign_id,))
    conn.commit()
    if not texts:
        store.mark_embedded(conn, campaign_id, False)
        return {"chunks": 0, "embedded": 0}

    new_ids = store.insert_chunks(conn, campaign_id, texts)
    embedded = 0
    for chunk_id, text in zip(new_ids, texts):
        try:
            _add_vector(conn, chunk_id, embedding.embed(text))
            store.set_chunk_embedded(conn, chunk_id)
            embedded += 1
        except Exception:                      # noqa: BLE001
            # Partial state is a designed outcome here as everywhere else (§2.1): the row is
            # correct, the index is behind, and `finish_indexing` closes it.
            break
    store.mark_embedded(conn, campaign_id, embedded == len(texts))
    return {"chunks": len(texts), "embedded": embedded}


def _reindex(conn, campaign_id: str, *, commentary: list) -> dict:
    """The body, plus commentary chunks that arrived with a newly attached deck (§12.4/D39).

    The commentary half is `ingest_campaign`'s, kept in the same shape: one chunk per comment,
    never merged, each keeping its author and anchor — two notes packed together would share
    one attribution, and the anchor is half of what makes a comment worth keeping.
    """
    built = _rebuild_body_index(conn, campaign_id)
    embedded = built["embedded"]
    for item in commentary:
        source = {k: item.get(k) for k in
                  ("kind", "author", "date", "anchor", "page", "slide", "reply_to")}
        source = {k: v for k, v in source.items() if v is not None}
        pieces = chunking.pack([item["text"]])
        ids = store.insert_chunks(conn, campaign_id, pieces, kind="commentary",
                                  sources=[source] * len(pieces))
        for chunk_id, text in zip(ids, pieces):
            try:
                _add_vector(conn, chunk_id, embedding.embed(text))
                store.set_chunk_embedded(conn, chunk_id)
                embedded += 1
            except Exception:                  # noqa: BLE001
                break
    return {"chunks": built["chunks"] + len(commentary), "embedded": embedded}


def update_asset(conn, *, asset_id: str, phase: str, why: str, said_by: str) -> dict:
    """Correct an asset's phase (§12.4/D123).

    `phase` is what an image IS evidence of — briefed creative, or what actually ran — and
    §9.1 says everything falls out of that: the reuse check, the execution comparison, which
    corpus a match is drawn from. A model that guessed `proposed` on fourteen event
    photographs had produced an unrepairable record, because nothing could set it afterwards.

    It takes a reason and a person, like every other consequential write here: changing this
    changes what a past judgment's evidence MEANT, and "somebody decided these were the
    delivered shots" is the part a reader needs six months later.
    """
    who = identity.person(said_by, field="said_by")
    if not (why or "").strip():
        raise ValueError(
            "`why` is required: this changes what the image is evidence OF — briefed creative "
            "or what actually ran — and every reuse check and execution comparison reads it. "
            "A correction nobody can account for later is one nobody can trust.")
    phase = enums.normalise(phase, field="phase", valid=store.VALID_ASSET_PHASES,
                            allow_none=False)

    before = store.get_asset(conn, asset_id)
    if not before:
        raise ValueError(f"{asset_id!r} is not an asset on file.")
    # The account DERIVED before anything is written. `record_authorship` ran after the phase
    # had already been committed, so an identity failure left the correction made and nobody's
    # name against it — a changed reading of what an image is evidence of, with no account of
    # who changed it, which is the one thing §11.1 exists to prevent. Deriving it first turns
    # that into a refusal: the asset stays as it was and the caller is told why. This is a
    # read with no side effects, and `record_authorship` still derives its own, so there is
    # one implementation of who-is-speaking rather than a second copy here.
    identity.who_said_it(on_behalf_of=who)
    if not store.set_asset_phase(conn, asset_id, phase):
        raise ValueError(f"{asset_id!r} could not be updated.")
    # §11.1: the account beside the name, on a write that changes what evidence means — and
    # WITH the reason. `why` was required, returned in the response, and stored nowhere: the
    # docstring above calls it "the part a reader needs six months later" and it was lost the
    # moment the call returned. A required argument the product throws away is worse than no
    # argument, because the caller believes they have recorded something.
    store.record_authorship(conn, subject_kind="asset", subject_key=asset_id,
                            on_behalf_of=who, note=why.strip())
    return {
        "asset_id": asset_id, "phase": phase, "was": before["phase"],
        "why": why.strip(), "said_by": who, "basis": "stated",
        "what_it_means": (
            f"This image now counts as {phase!r} rather than {before['phase']!r}. "
            + ("It is what RAN, so it is evidence for the execution comparison and is "
               "searched as delivered creative."
               if phase == "delivered" else
               "It is BRIEFED creative, so it is what execution is compared against rather "
               "than evidence of what happened.")
            + " Judgments saved before this correction were made against the old reading."),
    }


def attach_deck(conn, *, campaign_id: str, asset_ref: dict) -> dict:
    """Attach a deck to a record that already exists (§12.4/D39).

    **The gap this fills is not a convenience.** `update_campaign` takes no `asset_ref` and
    `upload_image_asset` takes an image, so a record whose `commentary_checked` is false could
    only gain its comments by being UPLOADED AGAIN AS A DUPLICATE — which is the offer §5.2
    refused in writing, and D51's gap was blocked behind it because the only remedy available
    was the thing the product tells people not to do.

    It was found by 5.2's own test that every offered action must name arguments its tool
    accepts: the obvious offer could not be made, because there was no tool to make it with.

    REPLACING a deck is not what this does. A record that already has one gets a new version
    through §6.3's supersession, which keeps both — silently swapping the text would throw
    away what every saved judgment was made against, and a quote verified against the old
    deck would then fail against a record that never said it.
    """
    record = store.get_campaign(conn, campaign_id)
    if not record:
        return {"error": f"campaign {campaign_id} not found"}
    if (record.get("deck_text") or "").strip() or record.get("asset_path"):
        raise ValueError(
            f"{record['title']!r} already has a deck. Attaching another would replace the "
            f"text every saved judgment about this record was made against, and a quote "
            f"verified against the old one would then fail against a record that never said "
            f"it. To record a NEW version, upload it and mark it as superseding this one "
            f"(`supersedes={campaign_id!r}`), which keeps both.")

    path, warnings = _resolve_asset(asset_ref)
    if not path:
        raise ValueError(
            "No readable file reached the server. `asset_ref` takes a path this machine can "
            "open, or the bytes themselves — see `upload_campaign` for the shapes it accepts.")

    # EXTRACTED BEFORE ANYTHING IS KEPT OR WRITTEN. The first version copied the file into
    # the asset directory first, so a corrupt .pptx raised out of `extract_units` having
    # already left an orphan behind, and — worse — a file this product cannot read at all
    # (`notes.txt`, a `.png`) was written onto the row as `asset_path` with an empty
    # `deck_text`. That record then had "a deck" for every purpose: the refusal above fired
    # on the next attempt, `commentary_never_read` still counted it, and the only remedy the
    # product offered was one it now rejected. A record you cannot repair is worse than one
    # that never got its deck.
    commentary, commentary_checked = [], False
    try:
        commentary, extra = extract.extract_commentary(path)
        warnings += extra
        commentary_checked = extract.guess_mime(path.name) in (config.PPTX_MIME,
                                                               "application/pdf")
    except Exception as exc:              # noqa: BLE001
        warnings.append(notices.notice(
            "commentary_unreadable",
            detail=f"comments and notes could not be read ({exc}); the deck itself was "
                   f"attached normally"))
    try:
        units, extra = extract.extract_units(path)
    except Exception as exc:              # noqa: BLE001
        raise ValueError(
            f"{path.name!r} could not be read as a deck ({exc}), so nothing was attached and "
            f"{record['title']!r} is exactly as it was. A zero-byte or truncated file reads "
            f"like this. Attaching it anyway would leave the record holding an unreadable "
            f"file and refusing the real deck afterwards.") from exc
    warnings += extra
    deck_text = "\n\n".join(units)

    if not deck_text.strip() and not commentary:
        raise ValueError(
            f"Nothing could be read out of {path.name!r} — no text and no comments — so "
            f"nothing was attached and {record['title']!r} is exactly as it was. This "
            f"product reads .pptx, .pdf, .docx and plain text as decks; an image belongs on "
            f"`upload_image_asset`. Attaching a file it cannot read would mark the record as "
            f"having a deck, which refuses the real one when it turns up.")

    stored_path = _keep_asset(path)
    # The row's own condition decides, not the check at the top of this function: two attaches
    # racing both passed that check and the second overwrote the first's text while the first's
    # comments stayed. False here means somebody else got there in between.
    if not store.attach_deck_to_campaign(conn, campaign_id, deck_text=deck_text,
                                         asset_path=stored_path,
                                         commentary_checked=commentary_checked):
        (config.ASSET_DIR / stored_path).unlink(missing_ok=True)
        raise ValueError(
            f"{record['title']!r} gained a deck while this one was being read, so nothing "
            f"was attached — two decks on one record would leave its text saying one thing "
            f"and its comments remarking on another. To record this one as a NEW version, "
            f"upload it with `supersedes={campaign_id!r}`.")
    # Re-chunked and re-embedded through the SAME path an edit takes, because a deck attached
    # and not indexed is a file on disk: the point of attaching it is that the record can then
    # be found by what the deck says.
    indexed = _reindex(conn, campaign_id, commentary=commentary)

    # §9.3, and the same rule the EDIT path applies: a deck that arrives is a deck whose
    # promises this library has not read. Omitted here, a record repaired exactly as
    # `commentary_never_read` recommends had no commitments at all, so `check_commitments`
    # and the briefed half of `compare_execution` read empty on the records the product had
    # just told somebody to fix. Campaigns only — brand guidelines are full of bulleted lists
    # and none of them is a promise this campaign made.
    promised = None
    if record.get("record_type") in learning.CHECKABLE_RECORDS:
        commitments.extract(conn, campaign_id=campaign_id, text="\n".join(
            filter(None, [record.get("detail"), deck_text])))
        promised = commitments.summary_for(conn, campaign_id)
    # §9.1: the images inside the deck, through `ingest_campaign`'s own helper rather than a
    # second copy of it. Without this the briefed side of every execution comparison was empty
    # on an attached deck, which is the half of Phase 9 that says what was MEANT to run.
    found = _index_deck_images(conn, campaign_id, path,
                               deadline=time.monotonic() + config.TOOL_TIME_BUDGET_SECONDS)
    warnings += found["warnings"]

    out = {
        "campaign_id": campaign_id, "title": record["title"],
        "commentary_checked": commentary_checked,
        "commentary_found": len(commentary),
        "image_assets": found["image_assets"],
        "images_checked": found["images_checked"],
        "images_total": len(found["image_assets"]),
        "images_embedded": found["images_embedded"],
        "basis": "computed",
        **indexed,
        "what_it_means": (
            f"The deck is now on {record['title']!r} — the same record, not a copy. Its text "
            f"is searchable and "
            + (f"{len(commentary)} comment(s) were read from it."
               if commentary_checked else
               "this file type carries no comments this product can read, which is recorded "
               "so a later reader can tell that from a deck that had none.")),
    }
    if promised is not None:
        out["promised"] = promised
    # §2.1/D98: partial state is never silent, and this path was. A deck attached with two of
    # nine sections embedded returned `chunks: 9, embedded: 2` into a field nothing tells the
    # model to read, with no warning and no offer — so the observable result of a half-failed
    # attach was a successful one, and the record stayed unfindable by the very text that was
    # just attached to make it findable. The edit path raises this notice; so does ingest.
    behind = indexed["chunks"] - indexed["embedded"]
    if behind > 0:
        warnings.append(notices.notice(
            "chunk_not_embedded",
            detail=f"{behind} of {indexed['chunks']} section(s) of the attached deck could "
                   f"not be indexed, so searches will not match what they say.",
            affects="This record will not come back in searches for the deck just attached.",
            count=behind,
            next_actions=actions.to_finish_indexing(campaign_id)))
    out["warnings"] = notices.collapse(warnings)
    return out


def update_campaign(conn, campaign_id: str, **fields) -> dict:
    """`store.update_campaign`, plus §6.3's moment when a supersession is declared LATE.

    The first version fired the prediction loop only at upload — and the D41 offer is
    accepted by calling THIS, so the flow the item constructs (diff two judged versions,
    offer to link, accept) landed on the one path where the loop never fired. That user is
    the better case, not the worse one: they have both judgments on screen because they just
    compared them.
    """
    # §12.4/D15: an approval is somebody's sign-off, so it takes a name — "the client
    # approved it" with nobody's name against it is an opinion this library holds and cannot
    # attribute, which is the whole of §11.2. Taken as `said_by` and stored as `approval_by`,
    # so the caller uses the same word everywhere else in this product does.
    said_by = fields.pop("said_by", None)
    if said_by is not None:
        fields["approval_by"] = identity.person(said_by, field="said_by")
    # And REFUSED without one. Validating `said_by` when it happens to arrive is not the same
    # rule as requiring it, and the gap between the two is a row reading `approval: approved`
    # with `approval_by` empty — which is this library asserting that somebody signed the deck
    # off while being unable to say who. Every one of the four values is somebody's act; there
    # is no value here meaning "nobody has decided yet", because that is the field being unset.
    elif str(fields.get("approval") or "").strip():
        raise ValueError(
            "`approval` needs `said_by`: it is a person's sign-off on a returned deck, and "
            "recording the verdict without the name leaves this library asserting that the "
            "work was approved with no way to say by whom. Pass the person who gave it.")
    # Dropped when nothing was passed, so a wire call that omits them does not write NULLs
    # over what is already on the row.
    for empty in [k for k, v in list(fields.items()) if v is None]:
        fields.pop(empty)

    if not store.update_campaign(conn, campaign_id, **fields):
        return {"error": f"campaign {campaign_id} not found"}
    # §11.5: a tag carrying a person's name IS an opinion, however it arrived. Without this
    # the table would hold only what came through the menu, and `update_campaign` — which the
    # docstring itself calls the way to upgrade a tag's provenance — would be a second door
    # into the same fact, with the library losing every view that came through it.
    # What was STORED, not what the caller sent. The two differ on exactly the fields this
    # walk gates on: `" liked "` is stored as `liked` and was skipped here as not a reaction
    # axis, and `"client stated"` is stored as `stated` and was copied in raw — the campaign
    # and the append-only record disagreeing about one opinion, which is the split this walk
    # exists to prevent. Gated on the caller having SENT tags, because tags replace: reading
    # the stored list on a title-only edit would re-walk opinions this call never touched.
    if fields.get("tags") is not None:
        _keep_the_view(conn, campaign_id, store.get_campaign(conn, campaign_id).get("tags"))
    reindexed = _reindex_if_content_changed(conn, campaign_id, fields)
    record = store.get_campaign(conn, campaign_id)
    earlier_judgment = _judgment_to_check(conn, fields.get("supersedes") or None)
    # §10.6/D98: the re-index's warnings, persisted and carried onto the response. `_persisted`
    # is what makes one survive past this reply — a `degraded` notice that lives for exactly
    # one response is the thing the library most wanted somebody to act on and forgot fastest.
    warnings = _persisted(conn, campaign_id,
                          notices.collapse(reindexed.pop("warnings", [])))
    # And retract it when this edit re-indexed cleanly. Without this, a partial edit followed
    # by a successful one left the record wholly searchable with the notice still open — so
    # the queue kept it under `needs_attention` forever.
    if not warnings:
        _settle_indexing_notice(conn, campaign_id)
    reindex_offers = [a for w in warnings for a in (w.get("next_actions") or [])]
    record = {**record, **reindexed, **({"warnings": warnings} if warnings else {})}
    # §9.6/D116: giving a campaign a window is the write that makes `campaign_context`
    # answerable at all — before it, every call returns `nothing_to_check`. Offered only when
    # something actually overlaps, because "here is a tool that will tell you nothing" is the
    # offer that teaches a reader to skip the list.
    window_offer = _context_offer(conn, campaign_id, fields)
    if not earlier_judgment:
        # The re-index offer FIRST: an index that is behind makes this record unfindable, and
        # the window offer is about a check it cannot yet be part of.
        merged = actions.trim(reindex_offers + (window_offer.get("next_actions") or []))
        rest = {k: v for k, v in window_offer.items() if k != "next_actions"}
        return {**record, **rest, **({"next_actions": merged} if merged else {})}
    return {**record, "earlier_judgment": earlier_judgment,
            **{k: v for k, v in window_offer.items() if k != "next_actions"},
            "next_actions": actions.trim(reindex_offers + actions.after_upload(
                campaign_id=campaign_id, status=record["status"],
                # ACTUAL metrics. A campaign holding only a target has not been measured, and
                # the offer this gates is "record what this campaign actually achieved" — §8.8
                # made a target storable, so "has a row" and "has a result" became different
                # questions on the one path that can see both.
                has_metrics=record["has_actual_metrics"],
                has_window=bool(record.get("starts_on")),
                earlier_judgment=earlier_judgment))}


def _settle_indexing_notice(conn, campaign_id: str) -> None:
    """Retract `chunk_not_embedded` once the record really is wholly searchable (§10.6/D98).

    Called from EVERY path that can change how much of a record is indexed, not only from
    `finish_indexing`. It lived there alone at first, so a second edit that re-indexed
    cleanly left the notice standing — the record was wholly searchable and the queue still
    listed it under `needs_attention`, which is the top-ranked reason a campaign waits on a
    person. A notice nothing retracts accumulates until the queue is made of them, and then
    the most urgent row in the product is permanently a job that finished last week.

    Only when the record is WHOLLY indexed. A run that closed four of nine sections has not
    made it findable by its new wording, and saying so would be the partial-state-reported-as-
    success failure this notice exists to fix.
    """
    outstanding = store.count_unembedded(conn, campaign_id)
    if outstanding["chunks"] + outstanding["assets"]:
        return
    for notice_row in store.open_notices(conn, campaign_id):
        if notice_row["code"] == "chunk_not_embedded":
            store.clear_notice(conn, notice_row["id"])


def _keep_the_view(conn, campaign_id: str, tags) -> None:
    """Append any opinion carried in a tag list to the append-only record (§11.5).

    Only tags that name somebody: a `liked` with no `said_by` is the library's own bookkeeping
    or an import, and filing it as a person's view would put an opinion on the record with
    nobody's name against it — which is the one thing `reactions` exists to prevent.
    """
    # Read, never re-validated. `store.update_campaign` normalises these with the record's own
    # `has_actual_metrics` in hand; calling `normalize_tags` again here without it re-ran the
    # `verified` gate blind and refused a tag the write had just accepted — a second copy of a
    # rule, disagreeing with the first because it was given less to work with.
    for tag in (tags if isinstance(tags, list) else [tags]) if tags is not None else []:
        if not isinstance(tag, dict):
            continue
        who = (tag.get("said_by") or "").strip()
        if not who or tag.get("value") not in store.REACTION_AXES:
            continue
        # §11.2 is NOT re-checked here. It used to be — as a `try`/`except ValueError` that
        # skipped the tag — and mutation proved the branch could not be reached: deleting the
        # guard changed nothing any test could see. That was the tell. `store.update_campaign`
        # has already run these same tags through `normalize_tags`, which refuses a `said_by`
        # that does not name a person and fails the whole write, so anything arriving here has
        # passed the rule. Re-running it was a second copy of one rule, and the skip made this
        # path DISAGREE with the stored record rather than agree with it: the tag was written
        # and the opinion was dropped.
        # Not a second copy of a view already on file. `store.update_campaign` REPLACES tags,
        # so every caller re-sends the ones it is keeping — and each re-send appended the same
        # opinion again, marking the copy as a revision of itself. A phantom disagreement with
        # nobody on the other side of it.
        # A STATED time makes this a particular occasion; without one, "again" means nothing.
        # The dedupe compared `said_at` for exact equality against a stamp taken from the
        # clock, so a re-send one second later read as R. Vega revising their own view — C95
        # closed for calls landing inside the same second, which is not what the row claimed.
        # Tags REPLACE, so any second edit that touches them re-sends them, and §11.5 calls a
        # revision pair the most informative row this table holds: manufacturing a false one
        # costs more than the duplicate it was avoiding.
        said_at = (tag.get("said_at") or "").strip()
        already = any(
            v["said_by"] == who and v["value"] == tag["value"]
            and (v["said_at"] == said_at if said_at else True)
            for v in store.reactions_for(conn, campaign_id))
        if already:
            continue
        said_at = said_at or store._now_iso()
        store.record_reaction(conn, campaign_id=campaign_id, value=tag["value"],
                              source=tag.get("source") or "stated", said_by=who,
                              said_at=said_at)
        # §11.1: the account beside the name, on this door too.
        store.record_authorship(conn, subject_kind="reaction", subject_key=campaign_id,
                                on_behalf_of=who)


def _reindex_if_content_changed(conn, campaign_id: str, fields: dict) -> dict:
    """Rebuild the chunks and vectors when an edit changed what the record says (D80, §7.2).

    `update_campaign` rewrote `title` and `detail` and left the index alone, so search kept
    matching the old wording — "same deck in, same chunks out" is false the moment a deck
    changes and the index does not. §6.1 stopped the stale text being QUOTABLE by verifying
    against the row columns; nothing had fixed the index itself, and a marketer who corrects a
    brief and then cannot find it is looking at the same bug from the other end.

    Deliberately not for `deck_text`, which `update_campaign` does not accept: content that
    large is a re-upload, and the docstring has always said so.
    """
    changed = [f for f in _CONTENT_FIELDS if fields.get(f) is not None]
    if not changed:
        return {}
    record = store.get_campaign(conn, campaign_id)
    summary = "\n\n".join(p for p in (record["title"], record.get("detail")) if p)
    # §9.3: the promises too. An edited brief promises something different, and checking the
    # new deck against the old deck's commitments is the "learning from the wrong document"
    # failure this phase exists to stop — arriving inside the fix for it. What a person decided
    # survives: added commitments stay, dropped ones stay dropped.
    # Only when the CONTENT changed. A title typo re-ran extraction, which deletes the open
    # extracted rows and reinserts them with new ids — so every `commitment_id` in a list the
    # model had just shown went stale, and `drop_commitment` on one failed with "not a
    # commitment on file". The promises are in the body, not the title.
    if (fields.get("detail") is not None
            and record.get("record_type") in learning.CHECKABLE_RECORDS):
        commitments.extract(conn, campaign_id=campaign_id, text="\n".join(
            filter(None, [record.get("detail"), record.get("deck_text")])))
    built = _rebuild_body_index(conn, campaign_id)
    if not built["chunks"]:
        return {"reindexed": {"fields": changed, "chunks": 0, "embedded": 0}}
    embedded = built["embedded"]
    texts = [None] * built["chunks"]
    out = {"reindexed": {"fields": changed, "chunks": len(texts), "embedded": embedded}}
    if embedded == len(texts):
        return out
    # §10.6/D98: every other partial-state path in this codebase raises a notice naming the
    # fix (§2.1's obligation). This one reported `embedded: 2` of `chunks: 9` into a field
    # nothing told the model to read, so the observable result of an edit that half-failed
    # was a successful edit — and the record was then unfindable by its NEW wording while
    # `update_campaign` returned the new wording. That is the §6.1 defect from the other end,
    # and silent: search does not report what it failed to consider.
    behind = len(texts) - embedded
    out["warnings"] = [notices.notice(
        "chunk_not_embedded",
        detail=f"The edit rewrote {' and '.join(changed)} and {behind} of {len(texts)} "
               f"section(s) could not be re-indexed, so searches will still match the OLD "
               f"wording of those sections and not the new.",
        affects="This record will not come back in searches for its new wording.",
        count=behind,
        next_actions=actions.to_finish_indexing(campaign_id))]
    return out


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
    handoff_checklist = _exit_checklist(judgment.get("verdict"), all_findings)
    open_findings = [{k: f.get(k) for k in ("id", "severity", "kind", "departure", "finding")}
                     for f in all_findings[:_MAX_EARLIER_FINDINGS]]
    return {
        "campaign_id": superseded,
        "title": record["title"] if record else None,
        "evaluation_id": judgment["id"],
        "verdict": judgment.get("verdict"),
        "summary": judgment.get("summary"),
        "predictions": judgment.get("predictions"),
        # §6.5: the exit condition, in front of the version that is supposed to meet it. "The
        # specific, testable set of changes that converts this verdict to approve" is worth
        # nothing if nobody sees it when the next version arrives, and §6.3 built the moment.
        # Absent rather than empty, and capped like `open_findings` beside it — the cap is
        # there because a twelve-item list turned filing a document into a quiz, and a second
        # uncapped copy of the same findings under another key puts it straight back.
        **({"approve_if": judgment["approve_if"]} if judgment.get("approve_if") else {}),
        **({"exit_checklist": handoff_checklist[:_MAX_EARLIER_FINDINGS],
            "exit_checklist_total": len(handoff_checklist)} if handoff_checklist else {}),
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
        # §10.2/D84: `settled` travels. This is the exact surface the row is about — "the
        # same question returns next time" — and a hand-written key list is how the answer
        # would have been dropped one function after being attached at the chokepoint that
        # exists so it reaches every reader. Carried only when there IS one, so an unanswered
        # finding does not gain an empty field that reads as an answer.
        "findings": [{**{k: f.get(k) for k in ("id", "severity", "kind", "departure",
                                               "category", "finding", "fix")},
                      **({"settled": f["settled"]} if f.get("settled") else {})}
                     for f in judgment["findings"]],
    }


def prepare_evaluation(conn, *, subject_title: str, proposal_text: str, top_k: Optional[int] = None,
                       record_type: Optional[str] = None, status: Optional[str] = None,
                       tags: Optional[Union[str, dict, list]] = None, match_all_tags: bool = False,
                       region: Optional[str] = None, market: Optional[str] = None,
                       markets: Optional[Union[str, list]] = None, collection: Optional[str] = None,
                       full_detail: bool = True,
                       campaign_id: Optional[str] = None,
                       starts_on: Optional[str] = None,
                       ends_on: Optional[str] = None) -> dict:
    """
    Package the evidence Claude needs to judge a new proposal: the most similar prior
    campaigns WITH their outcomes. full_detail defaults to True here (unlike find_similar) —
    an actual judgment over a short evidence list shouldn't be working from trimmed briefs.
    Claude reads this, works out its findings citing specific
    priors, then calls save_evaluation. This tool does NOT itself judge.

    §7.2 split the filters in two. `market`, `region`, `collection` and `markets` say WHICH
    BRIEF this is: with a `campaign_id` the record answers them and passing one is refused,
    because it asks for evidence about a different subject. `tags`, `status` and `record_type`
    narrow WITHIN the subject and stay yours — a {"value": ..., "source": "verified"} tag
    still weighs only precedent backed by real metric data rather than a stated impression.
    Every filter used is recorded on the receipt, so the choice is reproducible.
    """
    # §7.2. "The server extracts the query from the source file, not from the conversation.
    # Same deck in, same chunks out."
    caller_filters = {k: v for k, v in (("record_type", record_type), ("status", status),
                                        ("tags", tags), ("region", region),
                                        ("market", market), ("markets", markets),
                                        ("collection", collection)) if v}
    # §9.7: WHEN it runs is the same kind of claim as WHERE — it says which brief this is, and
    # a record answers it. Kept out of `caller_filters` because it is not a retrieval filter;
    # it is checked for the same refusal, because silently dropping it left a caller who passed
    # dates believing those dates were what the calendar checked.
    caller_window = {k: v for k, v in (("starts_on", starts_on), ("ends_on", ends_on)) if v}
    if top_k is not None:
        raise ValueError(
            f"top_k is not the caller's to choose: a judgment resting on 3 precedents and "
            f"one resting on 20 are different judgments, and neither number is a fact about "
            f"the brief. The server pins it at {_PINNED_TOP_K}.")
    subject = store.get_campaign(conn, campaign_id) if campaign_id else None
    if subject:
        # Refusing EVERY caller filter was too broad, and it removed a capability the
        # docstring documents: "weigh only precedent whose performance claim is verified" is a
        # deliberate narrowing somebody asks for, not the model quietly choosing a scope. The
        # line is what the filter DOES. `market`, `region` and `collection` say which brief
        # this is, and the record already answers that — a caller overriding them is
        # describing a different subject. `tags`, `status` and `record_type` narrow within it,
        # and every one of them is recorded on the receipt, so the choice is reproducible
        # rather than invisible, which was the actual complaint.
        overriding = sorted((set(caller_filters) & set(_SUBJECT_FILTERS)) | set(caller_window))
        if overriding:
            raise ValueError(
                f"{', '.join(overriding)} describes which brief this is, and the subject "
                f"record already answers that — passing it alongside `campaign_id` asks for "
                f"evidence about a different subject. Narrow within the subject with `tags`, "
                f"`status` or `record_type` instead, or omit `campaign_id` and own the whole "
                f"choice.")
        # The record's OWN attributes: the same subject retrieves the same evidence whoever
        # is describing it, which is the acceptance test for this whole item.
        filters = {k: subject[k] for k in _SUBJECT_FILTERS if subject.get(k)}
        narrowing = {k: v for k, v in caller_filters.items() if k not in _SUBJECT_FILTERS}
        filters.update(narrowing)
        filters_from = "subject_record+caller" if narrowing else "subject_record"
        query_basis = "subject_record"
        query_text = None
    else:
        filters, filters_from = caller_filters, ("caller" if caller_filters else "none")
        query_basis = "caller_text"
        query_text = proposal_text

    # §7.1's checks, hoisted out of the return dict because §7.5's contract reads them: the
    # note has to say what was found for THIS brief, not restate the rule for finding it.
    computed = (facts.for_campaign(conn, campaign_id) if campaign_id
                else facts.compute(proposal_text))
    # §9.7: "have prepare_evaluation check every proposed window against it as a COMPUTED
    # fact". It joins `computed` rather than sitting beside it, because that is the whole of
    # "impossible to drop" — §7.1 tells the model to treat these as established rather than
    # re-deriving them, and a clash arriving as prose is a sentence it can decline to repeat.
    computed["calendar_clash"] = _calendar_clash(
        conn, campaign_id=campaign_id, proposal_text=proposal_text,
        starts_on=starts_on, ends_on=ends_on,
        markets=[m for m in ([market, region]
                             + ([markets] if isinstance(markets, str)
                                else list(markets or []))) if m])
    # §8.3/§8.4, hoisted for the same reason: the note has to say what THIS brief is missing,
    # not restate the rule that produces the list.
    expected_now = (metrics.expected_check(conn, campaign_id) if campaign_id
                    else _no_subject_to_check())
    # With a record, its own markets; without one, the markets the CALLER named. A proposal
    # that is not stored yet is the "judge this new pitch" flow, and leaving the client's own
    # standing rules out of it left them out of the judgment they most obviously apply to.
    standing_now = corrections.standing_for(
        conn, campaign_id if subject else None,
        markets=None if subject else [m for m in ([market, region]
                                                  + ([markets] if isinstance(markets, str)
                                                     else list(markets or []))) if m])
    # Gathered before the receipt is written, because D18 stamps them onto the verdict: a
    # judgment made over a half-indexed library is a different judgment from one made over a
    # whole one, and the warning that said so lived for exactly one response.
    warnings_now = _incompleteness_warnings(conn) + _mixed_model_warning(conn)

    # find_similar, not find_similar_with_context, so the "your library is only partly
    # indexed" warning is added explicitly below rather than inherited — see the note there.
    evidence = find_similar(conn, text=query_text, campaign_id=campaign_id if subject else None,
                            top_k=_PINNED_TOP_K, full_detail=full_detail, **filters)
    concluded = [e for e in evidence if e["metrics"]]
    return {
        "subject_title": subject_title,
        "evidence_count": len(evidence),
        "evidence": evidence,
        # §7.2: what was retrieved, how, and a receipt so a later judgment can be checked
        # against the window it was actually given (D77).
        "retrieval": {
            "query": query_basis,
            "filters": filters,
            "filters_from": filters_from,
            "top_k": _PINNED_TOP_K,
            "embedding_model": embedding_model_id(),
            "receipt": store.insert_retrieval(
                conn, subject_title=subject_title,
                campaign_id=campaign_id if subject else None, query=query_basis,
                filters=filters, top_k=_PINNED_TOP_K,
                campaign_ids=[e["campaign_id"] for e in evidence],
                similarities=[[e["campaign_id"], e["similarity"]] for e in evidence],
                warnings=[w["code"] for w in warnings_now],
                embedding_model=embedding_model_id()),
            "what_it_means": (
                "The query was this record's own text, so the same subject retrieves the "
                "same evidence however it is described."
                if query_basis == "subject_record" else
                "The query was the text you passed, so a different description of the same "
                "brief would retrieve different evidence. Pass `campaign_id` once the "
                "subject is a record and the server will derive the query and the filters "
                "from it."),
        },
        # §12.1: what the rulebook actually did to THIS call, as a fact the caller can read
        # rather than a claim the model makes about itself. `basis: computed` because the
        # server put the rules in front of the judgment; "the model says it considered the
        # rulebook" would be worth nothing.
        "rulebook": rulebook.applied(),
        "note": _contract_for_this_brief(computed) + (
            # The vocabulary here has to be the vocabulary save_evaluation accepts. This
            # said "proceed/revise/reject" while the enum takes "approve" — so the prompt
            # that shapes the judgment taught a word the next tool rejects.
            ("When `earlier_version` is present this proposal revises a judged record. For "
             "each of its findings: if this version addresses it, put it in `resolved` with "
             "its `finding_id`; if the problem is still there, raise it as your own finding "
             "carrying `repeats` set to that id. That is what lets diff_campaigns state "
             "which corrections were taken instead of guessing from how alike two sentences "
             "read. "
             + _say_the_settled(_earlier_version_findings(conn, campaign_id))
             if _earlier_version_findings(conn, campaign_id) else "") +
            "`computed` holds the facts the SERVER established about this brief by reading "
            "it — dates, budget, creator engagement rates, which channels are NAMED, and "
            "whether any statement about dates contradicts the calendar. Treat them as "
            "established and do not re-derive them: a model deciding them again is the "
            "variance this exists to remove. Read each `status` before using it — "
            "`nothing_to_check` and `not_applicable` are not clean results, they are "
            "unchecked ones. "
            # The escape hatch, and it is not a hedge. Every one of these is a regex, and a
            # wrong computed fact is worse than a model's guess precisely because it carries
            # the server's authority — so the one remaining check is the model reading the
            # attached evidence and seeing that it does not support the claim. Telling it
            # never to contradict them removes that check.
            "Each fact carries the `evidence` it was read from. If the evidence plainly does "
            "not support the fact — the \u201ccreator\u201d is an email address, the "
            "\u201cbudget\u201d is a retail price — say so as a `computed_fact_disputed` "
            "observation naming the code and quoting the evidence, and reason from what you "
            "can see. Do not silently re-derive it, and do not defer to it against the "
            "evidence in front of you. "
            + _say_the_language(computed)
            + _say_the_expected_measures(expected_now)
            + _say_the_standing_corrections(standing_now)
            + _say_the_execution_drift(evidence)
            + _say_the_context(evidence)
            + _say_the_calendar(computed.get("calendar_clash"))
            + _say_the_confounded(evidence)
            + _say_the_disagreement(evidence) +
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
            "most, and a `verified` performance claim over a `stated` one — a stated claim is "
            "somebody's impression, and a finding you cannot quote is an opinion. "
            # Said here as well as in save_evaluation's description, because this is the
            # message in front of the model while it is deciding what to write down. A rule
            # it only meets as a rejection afterwards costs a retry every time.
            "Quotes are checked against the record you cite and the write is refused if the "
            "words are not there, so copy them from the evidence above rather than writing "
            "them from memory — … for anything you leave out. A point you cannot quote is "
            "an observation, and saying it as one is better than a citation that fails."
        ) + "\n\n" + EVALUATION_PROCEDURE,
        "campaigns_with_outcomes": [e["campaign_id"] for e in concluded],
        # §7.1: the two thirds of a real evaluation that were mechanical, run as code so they
        # stop being generated at all. From the RECORD when the subject is one — the same
        # reasoning §6.4 reached about which text a check should run against — and from the
        # body layer either way (D11), because a reviewer's note saying "never mention a
        # competitor budget of 90,000" would otherwise be read as the brief's budget.
        "computed": computed,
        # §8.3/§8.4: the checklist for this market, rendered from the metric registry at call
        # time. "Expected for a store launch: budget, reach, footfall uplift, sell-through at
        # 60 days. This brief carries none of the four." — and no prompt was edited to make it
        # appear. A measure graduating changes what every subsequent brief is checked against
        # without anybody touching a string, which is the whole of §8.4.
        "expected_measures": expected_now,
        # §8.6: the same loop's other half. These are what this client has actually repeated
        # across markets, each with the provenance it was learned from — *"the most valuable
        # content the library holds, because they are the things this client repeats"*. A rule
        # nothing reads is a row in a table, which is the "frozen" the review is describing.
        "standing_corrections": standing_now,
        # §6.4's other half, and the half that can actually change a verdict. The save-time
        # search RECORDS overconfidence; by then the judgment is written. What changes the
        # reasoning is seeing both sides while reasoning — so the evidence that worked and
        # the evidence that did not are separated here, instead of arriving as one ranked
        # list in which the strongest match sets the tone. `verified` only: a performance tag
        # somebody typed is an impression, and an impression cannot be the counterweight.
        "outcomes": _both_poles(conn, evidence=evidence,
                                text=None if campaign_id else proposal_text,
                                campaign_id=campaign_id),
        # §9.5, retrieved rather than partitioned. The `execution` note on each citation says
        # what THAT row is worth; this says whether the library holds a precedent that ran the
        # way it was written, which the ranked five cannot answer for it.
        "ran_as_briefed": _ran_as_briefed(conn, evidence=evidence,
                                          text=None if campaign_id else proposal_text,
                                          campaign_id=campaign_id),
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
        "warnings": warnings_now,
        # §10.6/D116. The `note` above ends "then call save_evaluation with a verdict" — an
        # instruction inside prose, which is the form this project's own principle says gets
        # dropped, and it is the one instruction the whole second half of the product rests
        # on: nothing can be reconciled, calibrated, superseded or linked to a later version
        # unless the judgment was SAVED. An unsaved verdict is a sentence in a chat window.
        #
        # `needs` carries the whole judgment, because accepting this is not one step and
        # saying otherwise would be the prefilled-blank failure `action` refuses. Only what
        # this call already holds is filled in: the title, and the record when there is one.
        "next_actions": [actions.action(
            f"Save the judgment about “{subject_title[:40]}” once you have made it",
            "save_evaluation",
            why="A verdict that is not saved cannot be reconciled against results, cited by "
                "a later brief, or checked when the next version lands — the whole "
                "second half of this product starts at the saved judgment.",
            consent="ask",
            needs=["verdict — approve, revise or reject",
                   "summary — one line",
                   "findings — one per problem, each with severity, kind and a quote",
                   "predictions — any CTR/ROI ranges, so they can be scored later"],
            subject_title=subject_title,
            campaign_id=campaign_id if subject else None)],
    }


def reconcile_evaluation(conn, *, evaluation_id: str, actual: Optional[str] = None) -> dict:
    """
    Line up what was said against what shipped, what it did, and what else was going on (§9.9).

    The review: *"The tool exists and has never run. This is its job: line up predicted against
    delivered against actual against context, and produce one record — what we said would
    happen, what we actually shipped, what it did, and what else was going on. That record is
    the only thing in the entire product capable of telling you whether its own judgment is any
    good."*

    All four columns already existed and none of them had ever been put beside each other. That
    independence is most of the value here: a reconciliation assembled out of one subsystem's
    opinion of itself proves nothing, and these were each built for their own reasons —
    §2.4 recorded what was said, §9.2/§9.3 what shipped, §6.5 what it did, §9.6–§9.8 what else
    was happening.

    The server LINES UP and scores arithmetic; it does not mark its own homework. Whether 3.4
    falls inside 3.0–4.0 is computed. Whether "the six-week timeline is unrealistic" held is a
    judgment, and it comes back `yours_to_judge` rather than guessed — `not_comparable` never
    counts as a pass, because a calibration figure built from unscorable rows quietly scored as
    held would be the system awarding itself marks.
    """
    ev = store.get_evaluation(conn, evaluation_id)
    if not ev:
        return {"error": f"evaluation {evaluation_id} not found"}

    # D120/D127: the results of a re-briefed campaign land on the record that RAN, and the
    # judgment sits on the one it replaced. Reconciling against v1 reported "no measured
    # result" and offered to record results against a brief that never ran — the exact failure
    # `actions.after_upload` describes in writing, recreated on this item's own surface.
    # "Predicted vs delivered vs actual vs SUPERSEDED" is the review's own phrase for it.
    campaign, measured_on, superseded_by = _the_record_that_ran(conn, ev.get("campaign_id"))
    actual_metrics = [m for m in ((campaign or {}).get("metrics") or [])
                      if m["metric_type"] == "actual"]
    given = (actual or "").strip()
    if not actual_metrics and not given:
        # `nothing_to_check`, never an empty comparison. A reconciliation with no outcome on
        # either side would be the product grading itself against nothing.
        return {
            "evaluation_id": evaluation_id, "subject_title": ev["subject_title"],
            "status": "nothing_to_check", "basis": "computed",
            "what_it_means": (
                "This judgment has no measured result to be checked against: the campaign "
                "carries no actual metrics and none were passed in. That is not the same as "
                "the judgment having been right."),
            "next_actions": actions.trim([actions.action(
                f"Record what \u201c{(campaign or {}).get('title') or ev['subject_title']}\u201d "
                f"actually achieved",
                "add_metrics",
                why="Until there are results, nothing can say whether this judgment was any "
                    "good — and that is the only thing in this product that grades it.",
                consent="ask",
                needs=["the results themselves — CTR, ROI, conversions, or whatever was "
                       "measured"],
                # The record that RAN, not the one that was judged.
                campaign_id=(campaign or {}).get("id") or ev.get("campaign_id"))]),
        }

    legacy = bool(ev.get("analysis")) and not ev.get("verdict")
    values, periods = _measured_values(conn, actual_metrics)
    scored = _score_predictions(conn, ev.get("predictions") or {}, values, periods)
    counts = {verdict: sum(1 for s in scored if s["verdict"] == verdict)
              for verdict in ("held", "missed", "not_comparable")}
    context_half = _reconciliation_context(conn, ev, actual_metrics)
    return {
        "evaluation_id": ev["id"],
        "subject_title": ev["subject_title"],
        "status": "checked",
        # The LINING UP is computed; every judgment inside it stays whoever's it was.
        "basis": "computed",
        # 1. What we said would happen.
        "predicted": {
            "basis": "judged",
            "verdict": ev["verdict"],
            "summary": ev["summary"],
            "approve_if": ev.get("approve_if"),
            "predictions": ev.get("predictions") or {},
            # Each finding, with the one honest verdict the server can give it.
            "findings": [{**{k: f.get(k) for k in ("id", "severity", "kind", "finding",
                                                   "fix")},
                          "verdict": "yours_to_judge"}
                         for f in (ev.get("findings") or [])],
            **({"original_analysis": ev["analysis"], "schema": "legacy"} if legacy else {}),
        },
        # 2. What we actually shipped — §9.2/§9.3, not the brief. A reconciliation that reads
        #    the brief as the execution is measuring the wrong document, which is the whole of
        #    Phase 9.
        "delivered": _delivered_half(conn, (campaign or {}).get("id")),
        # 3. What it did.
        "actual": {
            # TWO bases, because there are two things here. The sentence is the caller's when
            # they pass one; the numbers always come off the record. Labelling both `computed`
            # put a stated sentence and a computed figure that contradict each other under one
            # word — and a corrected `actual=` text was silently scored against the old rows.
            "detail": given or _joined_detail(actual_metrics),
            "detail_basis": "stated" if given else "computed",
            "values": values, "values_basis": "computed",
            "measured_rows": len(actual_metrics),
            "from_campaign_id": measured_on,
            "what_it_means": (
                "The sentence here is yours and the numbers are the campaign's own metric "
                "rows — the scoring below uses the numbers, which do not come from what you "
                "typed. If they disagree, record the correction with add_metrics."
                if given else
                f"{len(actual_metrics)} measured row(s) on this campaign.")},
        # 4. What else was going on.
        "context": context_half,
        "scored": scored,
        "counts": counts,
        "cited_ids": ev["cited_ids"],
        "what_it_means": _reconciliation_sentence(counts, context_half, ev, superseded_by),
        "note": ("This judgment predates the structured schema, so there is no verdict or "
                 "findings to compare against — only `original_analysis`, the free text as "
                 "it was written. Read it before comparing."
                 if legacy else
                 "The server has scored what is arithmetic. Everything marked "
                 "`yours_to_judge` or `not_comparable` is yours: say whether the finding "
                 "held, in your words, and call save_reconciliation with the lesson.")
        + _say_the_reconciliation_context(conn, ev, actual_metrics),
        "next_actions": actions.trim([actions.action(
            f"Record the lesson from \u201c{ev['subject_title'][:40]}\u201d",
            "save_reconciliation",
            why="The tally above is the server's. What it MEANT — whether the reasoning was "
                "wrong or the world was — is yours, and it is the only thing here that "
                "improves the next judgment.",
            consent="ask",
            needs=["comparison — what you make of it, in your words"],
            evaluation_id=evaluation_id)]),
    }


def link_evaluation(conn, *, evaluation_id: str, campaign_id: str,
                    linked_by: str) -> dict:
    """Attach a judgment made before the record existed (§9.9, D40).

    A judgment about a pitch nobody had stored carries a null `campaign_id`, §5.2 offers to
    add it to the library, and nothing could then join the two — so the loop this whole item
    exists to close was unreachable from the commonest starting point.

    Refused where one is already set. Re-pointing a judgment at a different campaign would
    silently change what it was about, and every citation of it with it.
    """
    ev = store.get_evaluation(conn, evaluation_id)
    if ev is None:
        raise ValueError(f"{evaluation_id!r} is not a judgment in this library.")
    if ev.get("campaign_id"):
        raise ValueError(
            f"this judgment is already about {ev['campaign_id']!r}. Re-pointing it at another "
            f"record would change what it was a judgment OF, and everything that cites it "
            f"with it. Judge the other record on its own.")
    if store.get_campaign(conn, campaign_id) is None:
        raise ValueError(f"{campaign_id!r} is not a record in this library.")
    # §11.1: empty-only, so `linked_by="Claude"` attached a judgment to a record and recorded
    # that a person had said they were the same thing — on the write that decides what a
    # judgment was a judgment OF.
    linked_by = identity.person(linked_by, field="linked_by")
    store.link_evaluation(conn, evaluation_id, campaign_id)
    return {"evaluation_id": evaluation_id, "campaign_id": campaign_id, "status": "linked",
            "basis": "stated", "linked_by": linked_by.strip(),
            "what_it_means": (
                f"{linked_by.strip()} attached this judgment to the record. Once its results "
                f"are on file, reconcile_evaluation can say whether the judgment was any "
                f"good — which it could not do while the two were unconnected.")}


def _the_record_that_ran(conn, campaign_id: Optional[str]) -> tuple:
    """The judged record, or whatever superseded it (§9.9, D120/D127).

    A judgment is about a brief; the results land on whichever version actually ran. Walking
    the chain is what `diff_campaigns` already does (C25) and what this needs for the same
    reason — otherwise the one surface that grades the product grades it against a document
    nobody executed.

    Stops at the first version carrying measured results, so a chain of three re-briefs
    reconciles against the one that has numbers rather than the newest empty one.
    """
    if not campaign_id:
        return None, None, None
    judged = store.get_campaign(conn, campaign_id)
    if judged is None:
        return None, None, None
    if any(m["metric_type"] == "actual" for m in judged.get("metrics") or []):
        return judged, judged["id"], None
    seen, walking = {campaign_id}, campaign_id
    while True:
        later = store.superseded_by(conn, walking)
        if not later or later in seen:
            return judged, judged["id"], None
        seen.add(later)
        record = store.get_campaign(conn, later)
        if record is None:
            return judged, judged["id"], None
        if any(m["metric_type"] == "actual" for m in record.get("metrics") or []):
            return record, record["id"], judged["id"]
        walking = later


def _joined_detail(actual_metrics: list) -> str:
    """The measured rows as text. A row can carry `structured` with no freeform `detail` at all
    — exactly what a KPI workbook produces — so a detail-only join silently contributed
    nothing for the commonest import path."""
    parts = []
    for m in actual_metrics:
        if m["detail"]:
            parts.append(m["detail"])
        if m["structured"]:
            parts.append(m["structured"] if isinstance(m["structured"], str)
                         else json.dumps(m["structured"]))
    return "\n".join(parts)


def _as_number(value) -> Optional[float]:
    """A workbook cell arriving as "3.4" is a measurement. Saying "no ROAS was measured" about
    it is a false statement about the library's own contents."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def _measured_values(conn, actual_metrics: list) -> tuple:
    """Every number on the measured rows, canonicalised through the registry, latest wins.

    Through §8.1's registry rather than the raw key, because "roas", "ROAS" and "roi" are one
    measure and a prediction scored against one spelling and not another is a calibration
    figure built on a coin toss. A key the registry does not claim keeps its own name — it
    simply will not match a prediction, which is `not_comparable` and honest.
    """
    values: dict = {}
    periods: dict = {}
    for m in actual_metrics:
        raw = m["structured"]
        if not raw:
            continue
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(parsed, dict):
            continue
        for key, value in parsed.items():
            # §8.2 already refuses a workbook's Month and Store # on the way in; reading them
            # back out as measurements here undid that, and `month: 12` read as somebody
            # having measured twelve of something.
            if not metrics.is_a_measurement(key):
                continue
            number = _as_number(value)
            if number is not None:
                name = metrics.canonical(conn, key) or key
                values[name] = number
                # How many DISTINCT periods this measure was recorded for. "Latest wins" was
                # insertion order, so a campaign-level prediction was scored against whichever
                # month the spreadsheet happened to end with — and the calibration figure was
                # built on it. §9.8 gave metrics their own period; this reads it.
                periods.setdefault(name, set()).add(
                    m.get("period_start") or m.get("created_at"))
    return values, {name: len(seen) for name, seen in periods.items()}


# `predicted_roi_range` -> `roi` -> whatever §8.1's registry says that measure is called.
_PREDICTION_KEY = re.compile(r"^(?:predicted_)?(?P<measure>.+?)(?:_range)?$")


def _measure_predicted(conn, key: str) -> Optional[str]:
    """Which measured value answers this prediction, per the REGISTRY (§9.9).

    Derived rather than hand-mapped. A table here would be a second opinion about what a
    measure is called, sitting beside §8.1's registry — and it drifts the moment somebody adds
    a synonym there: the first version mapped `predicted_conversion_range` to
    `conversion_rate`, which the registry does not recognise, so that entry could never match
    anything and nothing would have said so.

    It also settles ROI versus ROAS the only defensible way. They are arguably different
    measures, but the registry already canonicalises `roi` to `roas`, and a calibration figure
    that disagreed with the library's own naming about which number answers which prediction
    would be wrong in a way nobody could see.
    """
    match = _PREDICTION_KEY.match(str(key or "").strip().strip("_").lower())
    if not match:
        return None
    stem = match.group("measure")
    name = metrics.canonical(conn, stem)
    if name and stem != name and stem.startswith(name + "_"):
        # A LIFT is not a LEVEL. The registry's prefix rule is right for storing a value —
        # `ctr_lift` belongs to the CTR family — and wrong for deciding which measured number
        # answers a prediction: "predicted a 10-20% CTR lift" scored against a measured CTR of
        # 2.3 reads "missed, below by 7.7", which is arithmetic on two different quantities
        # wearing the server's authority. A prediction scored against the wrong measure is
        # worse than one not scored at all, because nobody can see it.
        return None
    return name


def _score_predictions(conn, predictions: dict, values: dict,
                       periods: Optional[dict] = None) -> list:
    """Whether each numeric prediction landed (§9.9).

    Arithmetic only. A range and a number are comparable and the server does it — a model
    re-deriving that is the variance §7.1 exists to remove. Anything else is `not_comparable`,
    which is a THIRD answer and never a quiet pass.
    """
    scored = []
    if predictions and not isinstance(predictions, dict):
        # `save_evaluation` accepts whatever the model sent, and `.items()` on a list is an
        # AttributeError on the surface that grades the product.
        return []
    for key, stated in sorted((predictions or {}).items()):
        measure = _measure_predicted(conn, key)
        bounds = _range_of(stated)
        if measure is None or bounds is None:
            scored.append({
                "predicted": key, "stated": stated, "kind": "prediction",
                "verdict": "not_comparable", "basis": "computed", "actual": None,
                "what_it_means": (
                    _why_not_a_range(stated)
                    if bounds is None else
                    f"No measure in this library is called {key!r}, so nothing on file "
                    f"answers it and it was not scored. That is not the same as it having "
                    f"held.")})
            continue
        if measure not in values:
            scored.append({
                "predicted": key, "stated": stated, "kind": "prediction",
                "verdict": "not_comparable", "basis": "computed", "actual": None,
                "measure": measure,
                "what_it_means": (
                    f"No {measure.upper()} was measured on this campaign, so this prediction "
                    f"cannot be checked. That is not the same as it having held.")})
            continue
        spans = (periods or {}).get(measure, 1)
        if spans > 1:
            # A prediction about a campaign, and a measure recorded for twelve months. Which
            # month it was about is not something the server can know, and picking one is how
            # a calibration figure gets built on a coin toss.
            scored.append({
                "predicted": key, "stated": stated, "kind": "prediction",
                "verdict": "not_comparable", "basis": "computed", "actual": None,
                "measure": measure, "periods": spans,
                "what_it_means": (
                    f"{measure.upper()} was recorded for {spans} periods on this campaign, so "
                    f"which of them this prediction was about is yours to say. Scoring it "
                    f"against one of them would be arithmetic on a guess.")})
            continue
        low, high = bounds
        got = values[measure]
        held = low <= got <= high
        scored.append({
            "predicted": key, "stated": stated, "kind": "prediction", "measure": measure,
            "actual": got, "verdict": "held" if held else "missed", "basis": "computed",
            "what_it_means": (
                f"Predicted {low:g}\u2013{high:g}; measured {got:g}. "
                + ("Inside the range." if held else
                   f"Outside it, {'above' if got > high else 'below'} by "
                   f"{(got - high) if got > high else (low - got):g}."))})
    return scored


# A number: optional sign, thousands groups of exactly three, optional decimal. Written
# strictly so "3,5" (a European decimal) does not read as three-thousand-and-something, and
# "5,000" does.
_NUMBER = r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d*\.\d+|[-+]?\d+"
# Two of them with a separator between, and anything unit-ish allowed in the gaps: "2%-3%",
# "3x to 4x", "300-400bps".
_RANGE = re.compile(
    rf"(?P<low>{_NUMBER})\s*[%xX]?\s*(?:bps|pts?|pp)?\s*(?:-|\u2013|\u2014|to|\.\.)\s*"
    rf"(?P<high>{_NUMBER})\s*[%xX]?\s*(?:bps|pts?|pp)?",
    re.IGNORECASE)
# An open-ended claim is not a range this can check, and reading one as a point turns "up to
# 4" into "exactly 4" — a prediction that was never made.
_OPEN_ENDED = re.compile(r"(up to|at least|no more than|no less than|under|over|below|above|"
                         r"minimum|maximum|min\.?|max\.?|[<>\u2264\u2265])", re.IGNORECASE)
# A comma that is not a thousands separator: "3,5" is a decimal in half of Europe and this
# cannot tell which half wrote the brief.
_AMBIGUOUS_COMMA = re.compile(r"\d,\d{1,2}(?!\d)")


def _range_of(stated) -> Optional[tuple]:
    """The low and high a prediction actually stated, or None when it is not a range (§9.9).

    Conservative on purpose. This decides whether the library's own judgment gets a pass, so
    every ambiguity resolves to `not_comparable` — a prediction nobody scored is honest, and a
    prediction scored against a number parsed out of a quarter label is the product grading
    itself on noise. Measured failures that drove each rule:

      "2%-3%"            the hyphen read as a minus once a unit sat before it, giving (-3, 2)
      "Q3 2026: 3.0-4.0" the year became the upper bound, so 1500 "held"
      "3,5-4,0"          a European decimal read as (3, 5)
      "5,000-6,000"      thousands separators split into (0, 5)
      "up to 4"          an open bound read as the point 4
      "3.4"              a point estimate that can hold only on exact equality
    """
    text = str(stated or "").strip()
    if not text or _OPEN_ENDED.search(text) or _AMBIGUOUS_COMMA.search(text):
        return None
    matches = _RANGE.findall(text)
    if len(matches) != 1:
        # None at all, or two — "3.0-4.0 in Q1, 5.0-6.0 in Q2" is two predictions and picking
        # one is a guess.
        return None
    low, high = (float(n.replace(",", "")) for n in matches[0])
    # Any number NOT inside the range is something this did not understand — a quarter, a
    # year, a footnote marker. The whole string has to be the range.
    leftover = _RANGE.sub(" ", text, count=1)
    if re.search(r"\d", leftover):
        return None
    return (min(low, high), max(low, high))


def _why_not_a_range(stated) -> str:
    """Why this prediction could not be checked, specifically enough to fix."""
    text = str(stated or "").strip()
    if _OPEN_ENDED.search(text):
        return (f"\u201c{text}\u201d is open-ended, so there is no upper or lower bound to check a "
                f"number against. Whether it held is yours to say — and a two-sided range "
                f"would make the next one checkable.")
    if _AMBIGUOUS_COMMA.search(text):
        return (f"\u201c{text}\u201d uses a comma this cannot read safely: 3,5 is a decimal in "
                f"half of Europe and three-and-a-half-thousand in the other half, and "
                f"guessing would score the judgment against a number nobody predicted.")
    if re.search(r"\d", text) and not _RANGE.search(text):
        return (f"\u201c{text}\u201d is a single figure rather than a range, so it can only be "
                f"checked by exact equality — which no real measurement ever satisfies. "
                f"Whether it held is yours to say.")
    return (f"\u201c{text}\u201d is not a range this server can check against a measured number, "
            f"so whether it held is yours to say.")


def _delivered_half(conn, campaign_id: Optional[str]) -> dict:
    """What actually shipped (§9.2/§9.3), for the reconciliation record."""
    if not campaign_id:
        return {"status": "nothing_to_check", "basis": "computed",
                "what_it_means": (
                    "This judgment was about a proposal that is not a record in the library, "
                    "so there is nothing to compare what shipped against.")}
    execution = _execution_note(conn, campaign_id)
    try:
        promised = commitments.summary_for(conn, campaign_id) or {}
    except Exception as exc:                 # noqa: BLE001 — the record must still assemble
        # "Could not check" is not "nothing to check". Every other surface in Phase 9 refuses
        # that collapse, and this one made it silently.
        return {"basis": "computed", "execution": execution,
                "commitments": {"status": "could_not_check", "detail": str(exc)[:120],
                                "what_it_means": (
                                    "The promises in this brief could not be checked, so what "
                                    "shipped against them is unknown — which is not the same "
                                    "as the brief having promised nothing.")},
                "what_it_means": execution["what_it_means"]}
    if not promised:
        # `{}` is what `summary_for` returns for a brief that promised nothing specific, and an
        # empty dict in a column headed "what we shipped" reads as an answer. A reader cannot
        # tell "this brief named no checkable promises" from "the check did not run".
        promised = {
            "status": "nothing_to_check", "count": 0,
            "what_it_means": ("This brief named no specific, checkable promises, so there is "
                              "nothing to hold the execution to on that axis."),
        }
    else:
        promised = {**promised, "status": "checked"}
    return {"basis": "computed", "execution": execution, "commitments": promised,
            "what_it_means": (
                execution["what_it_means"] + " " + promised["what_it_means"])}


def _reconciliation_sentence(counts: dict, context_half: dict, ev: dict,
                             superseded_by: Optional[str] = None) -> str:
    scorable = counts["held"] + counts["missed"]
    if not isinstance(ev.get("predictions") or {}, dict):
        said = ("This judgment's `predictions` are not a set of named predictions, so nothing "
                "could be scored. Whether it was right is entirely yours to say.")
    elif not scorable:
        said = ("None of this judgment's predictions could be checked against a measured "
                "number, so the server has scored nothing. Whether it was right is entirely "
                "yours to say.")
    else:
        said = (f"{counts['held']} of {scorable} checkable prediction(s) landed inside the "
                f"range they were given.")
    if counts["not_comparable"]:
        said += (f" {counts['not_comparable']} could not be checked at all — that is a third "
                 f"answer and not a pass.")
    if superseded_by:
        said += (" These results are from the record that SUPERSEDED the one this judgment was "
                 "about: the brief was re-briefed and the version that ran is the one with "
                 "numbers on it. Read the predictions against a plan that changed after they "
                 "were made.")
    if context_half.get("confounded"):
        said += (" Something else was going on while this ran, so read the misses twice: a "
                 "prediction that missed because the world moved is a different lesson from "
                 "one that missed because the reasoning was wrong.")
    return said


def save_reconciliation(conn, *, evaluation_id: str, comparison: str,
                        actual: Optional[str] = None,
                        basis: Optional[str] = None) -> dict:
    """Record what a person made of the comparison, with the server's tally beside it (§9.9).

    The split this product is built on, at the one surface that grades the product. The
    LESSON is `stated` and stays in their words; the TALLY is recomputed here rather than
    accepted from the caller, for the reason §6.4 settled about the disconfirming search — a
    check is only worth anything if a difference in it is a bug, which holds only when the
    server did it.
    """
    _check_the_lesson(comparison)
    if store.get_evaluation(conn, evaluation_id) is None:
        raise ValueError(
            f"{evaluation_id!r} is not a judgment in this library. `list_evaluations` names "
            f"the ones on file.")
    lined_up = reconcile_evaluation(conn, evaluation_id=evaluation_id, actual=actual)
    if lined_up.get("error"):
        raise ValueError(lined_up["error"])
    standing = store.reconciliation_for(conn, evaluation_id)
    if standing is not None and (standing.get("basis") or "results") != "superseding_version":
        # Re-reconciling the same judgment added its tally to `calibration` again. Three goes
        # at one judgment read as three judgments landing, which is the product inflating its
        # own record by the simple expedient of being asked twice.
        raise ValueError(
            f"this judgment has already been reconciled against its results — "
            f"\u201c{standing['comparison'][:80]}\u201d. Reconciling again would count the same "
            f"judgment twice in the calibration figure. Read it with get_reconciliation, "
            f"which also says whether anything has changed since.")
    counts = lined_up.get("counts")
    if counts is None:
        # A `nothing_to_check` line-up — normally the §6.3 version check, which arrives before
        # any outcome exists. Storing `{}` said "zero not checkable" about a judgment carrying
        # predictions nobody could check, which is a false statement of fact and quietly
        # removed those predictions from the calibration figure altogether.
        ev = store.get_evaluation(conn, evaluation_id) or {}
        counts = {"held": 0, "missed": 0,
                  "not_comparable": len(ev.get("predictions") or {})}
    rid = store.insert_reconciliation(
        conn, evaluation_id=evaluation_id, comparison=comparison.strip(), actual=actual,
        basis=basis,
        counts=counts,
        confounded=bool((lined_up.get("context") or {}).get("confounded")),
        # The assembled columns, kept. See the `record` column's comment: three of the four
        # are recomputed live over state that keeps moving, so a lesson beside three integers
        # leaves a later reader unable to see what the lesson was about.
        record={**{k: lined_up[k] for k in
                   ("predicted", "delivered", "actual", "context", "scored", "counts",
                    "what_it_means") if k in lined_up},
                "as_of": _now_iso()})
    return {
        "reconciliation_id": rid, "status": "saved",
        # The lesson is somebody's account; the counts beside it are arithmetic. Labelling the
        # whole row one way or the other would make one of them unreadable.
        "basis": "stated",
        "counts": counts,
        "counts_basis": "computed",
        "what_it_means": (
            f"Recorded. The server's tally for this judgment stands beside your lesson: "
            f"{counts.get('held', 0)} held, {counts.get('missed', 0)} missed, "
            f"{counts.get('not_comparable', 0)} not checkable. "
            f"`calibration` reads every one of these together."),
    }


# Below this a "lesson" is an acknowledgement. The product's only evidence about its own
# judgment, and its own fixtures were writing "Recorded." into it.
_MIN_LESSON = 25
# Everything the tally already says. What is left after removing all of it has to be a
# sentence — a regex matching the whole string could not see "1 held, 0 missed. 1 held."
_TALLY_WORDS = re.compile(r"\b(held|missed|not[\s_]?comparable|comparable|checkable|"
                          r"prediction|predictions|of|and|the|server|tally|out)\b",
                          re.IGNORECASE)


def _check_the_lesson(comparison: str) -> None:
    """The half that is a person's, and it has to say something (§9.9).

    §9.8 one item earlier refuses a `stated_by` that reads as the product and refuses a zero
    stated impact read as an attribution. This accepted "ok", "." and an emoji — and the
    repo's own fixtures were writing "Recorded." into the one field that carries what the
    product learned about itself.
    """
    text = (comparison or "").strip()
    if len(text) < _MIN_LESSON:
        raise ValueError(
            f"`comparison` is the LESSON, and {text!r} is an acknowledgement. The tally is "
            f"the server's and already on the record; this is the half that is yours — what "
            f"it meant, whether the reasoning was wrong or the world was. It is the only "
            f"thing here that improves the next judgment, so it needs a sentence.")
    if not _TALLY_WORDS.sub("", text).strip(" \t\n.,;:%/()-0123456789"):
        raise ValueError(
            "`comparison` restates the tally, which is arithmetic the server already did and "
            "stored. What belongs here is what it MEANT — why the call was right, or what "
            "the library got wrong about this kind of brief.")


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def get_reconciliation(conn, *, evaluation_id: str) -> dict:
    """The record as it stood when the lesson was written (§9.9).

    And whether the live answer has MOVED since — an earthquake recorded afterwards, a drift
    figure rewritten, a context event withdrawn. Both halves matter: the record says what the
    lesson rested on, and the delta says whether to go back.
    """
    stored = store.reconciliation_for(conn, evaluation_id)
    if stored is None:
        return {
            "evaluation_id": evaluation_id, "status": "nothing_to_check", "basis": "computed",
            "what_it_means": ("This judgment has never been checked against what actually "
                              "happened, so there is no record to read."),
            "next_actions": actions.trim([actions.action(
                "Check this judgment against what actually happened", "reconcile_evaluation",
                why="Nothing in this library says whether this judgment was any good.",
                consent="ask", evaluation_id=evaluation_id)]),
        }
    live = reconcile_evaluation(conn, evaluation_id=evaluation_id)
    kept = stored.get("record") or {}
    now = live.get("counts") or {}
    changed = {
        "context": bool((kept.get("context") or {}).get("confounded"))
                   != bool((live.get("context") or {}).get("confounded")),
        "delivered": ((kept.get("delivered") or {}).get("execution") or {}).get("status")
                     != ((live.get("delivered") or {}).get("execution") or {}).get("status"),
        "counts": bool(now) and now != (stored.get("counts") or {}),
    }
    return {
        "evaluation_id": evaluation_id, "status": "checked",
        "reconciliation_id": stored["id"],
        # The lesson is a person's; the record beside it is what the server assembled.
        "basis": "stated",
        "comparison": stored["comparison"],
        "against": stored.get("basis") or "results",
        # What the lesson rested on, and what the same check says today. A reader needs both
        # and needs to be able to tell them apart.
        "counts": stored["counts"], "counts_now": now, "counts_basis": "computed",
        "record": kept,
        "changed_since": changed,
        "what_it_means": (
            f"Recorded {kept.get('as_of', 'at an unknown time')}. "
            + ("Nothing about this has changed since."
               if not any(changed.values()) else
               "The live answer has moved since this was written ("
               + ", ".join(k for k, v in changed.items() if v)
               + "), so the lesson may be worth revisiting — the record above is what it "
                 "actually rested on.")),
    }


def calibration(conn) -> dict:
    """How often this library's own judgments turned out to be right (§9.9).

    "The only thing in the entire product capable of telling you whether its own judgment is
    any good." One reconciliation proves nothing; the tally across them is the point — and it
    is only as honest as its refusal to count an unscorable prediction as a pass.
    """
    rows = store.reconciliations(conn)
    if not rows:
        return {"status": "nothing_to_check", "basis": "computed",
                "counts": {"held": 0, "missed": 0, "not_comparable": 0},
                "reconciled": 0, "reconcilable": store.reconcilable_judgments(conn),
                "against_a_later_version": 0, "confounded": 0,
                "what_it_means": (
                    "No judgment in this library has ever been reconciled against what "
                    "actually happened, so nothing can say whether its judgments are any "
                    "good. That is not a score of zero; it is the absence of one.")}
    # A brief-versus-brief check is not a measured outcome. `store.py`'s own comment on the
    # `basis` column says why the column exists: "anything computing calibration would read
    # both as outcome data" — and this did.
    against_version = [r for r in rows if r.get("basis") == "superseding_version"]
    rows = [r for r in rows if r.get("basis") != "superseding_version"]
    reconcilable = store.reconcilable_judgments(conn)
    counts = {"held": 0, "missed": 0, "not_comparable": 0}
    confounded = moved = 0
    for row in rows:
        # RECOMPUTED, not read off the stored row. The tally is a save-time snapshot and the
        # inputs keep moving: a corrected figure turns a held prediction into a missed one, an
        # earthquake recorded afterwards confounds an outcome that read clean. Reading the
        # stamp left the product still reporting the pass — flattering in exactly the cases
        # where a correction lowered a figure, which is the direction nobody audits.
        #
        # The stored row is not wrong and is not discarded: it is what the LESSON rested on,
        # and `get_reconciliation` shows both. This figure is about the library as it stands.
        live = reconcile_evaluation(conn, evaluation_id=row["evaluation_id"])
        now = live.get("counts") or {}
        for key in counts:
            counts[key] += now.get(key, (row.get("counts") or {}).get(key, 0))
        confounded += 1 if (live.get("context") or {}).get("confounded") else 0
        if now and now != (row.get("counts") or {}):
            moved += 1
    scorable = counts["held"] + counts["missed"]
    said = (f"{counts['held']} of {scorable} checkable prediction(s) across "
            f"{len(rows)} reconciled judgment(s) landed."
            if scorable else
            f"{len(rows)} judgment(s) have been reconciled and none of their predictions "
            f"could be checked against a measured number.")
    # The denominator is whoever bothered, and saying so is the difference between a figure
    # and a boast: fourteen unchecked judgments beside one reconciled reads as a hundred
    # per cent.
    said += (f" This covers {len(rows)} of {reconcilable} judgment(s) that COULD be "
             f"reconciled, so it is a figure about the ones somebody went back to.")
    if len(rows) < _MIN_FOR_A_PATTERN:
        said += (" That is too few to be a record of anything — one judgment landing is a "
                 "fact about that judgment.")
    if counts["not_comparable"]:
        said += (f" A further {counts['not_comparable']} could not be checked — counted "
                 f"separately, because a figure that scored those as passes would be this "
                 f"product awarding itself marks.")
    if confounded:
        said += (f" {confounded} of these ran through something else that was going on and "
                 f"are marked confounded: they still count, and a reader weighing this "
                 f"figure should know how much of it is about the weather.")
    if moved:
        said += (f" {moved} of them have CHANGED SINCE the lesson was recorded — a corrected "
                 f"figure, or something recorded afterwards. This tally is the library as it "
                 f"stands; `get_reconciliation` shows what each lesson actually rested on.")
    if against_version:
        said += (f" A further {len(against_version)} judgment(s) were checked against a later "
                 f"VERSION of the brief rather than against results; they are counted "
                 f"separately because that is evidence about the judgment and not an outcome.")
    return {"status": "checked", "basis": "computed", "counts": counts,
            "moved_since_recorded": moved,
            "reconciled": len(rows), "reconcilable": reconcilable,
            "against_a_later_version": len(against_version),
            "confounded": confounded, "what_it_means": said}


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
        _add_vector(conn, aid, vec, space="asset")
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
    record = store.get_campaign(conn, campaign_id)
    # Nothing to say until there is a result for the drift to qualify. This also keeps a bulk
    # import of five hundred rows from running five hundred comparisons: the campaigns in a KPI
    # workbook almost never have delivered photographs.
    if not record or not record.get("has_actual_metrics"):
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
                [_vision_notice("how far the delivered creative moved from the brief could "
                                "not be measured: the images are not visually indexed. Run "
                                "finish_indexing — the files are stored.")])
    if missing:
        # PARTIAL is its own answer. Firing only when a side is entirely unindexed meant a
        # half-indexed set computed a score from whatever subset had vectors and labelled it
        # `measured`, with a count buried in `compared` as the only tell.
        warnings.append(_vision_notice(
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
