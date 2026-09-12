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
import sqlite3
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
    cid = store.insert_campaign(
        conn, title=title, record_type=record_type, status=status, tags=tags, region=region,
        market=market, markets=markets, collection=collection, supersedes=supersedes,
        detail=detail, deck_text=deck_text, asset_path=stored_path,
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
            has_metrics=bool(current["metrics"])),
        "warnings": notices.collapse(warnings),
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
# Which layer of a deck a quote was taken from (§2.5): what the brief says, or what somebody
# said about it.
_LAYERS = ("body", "commentary")


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


def _clean_precedent(value, where: str) -> Optional[dict]:
    """A finding cites either a campaign that did it differently or a rule it breached.
    Before this there was only `id`, meaning a campaign — so a guardrail breach, which is
    anchored to the rulebook rather than to any campaign, had nowhere to cite the thing it
    breached (§12 writes the rules; the slot is defined here so it does not migrate)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{where}precedent must be an object with a quote and either a "
                         f"campaign_id or a rule_id, got {type(value).__name__}")
    campaign_id = value.get("campaign_id") or value.get("id")
    rule_id = value.get("rule_id")
    if not campaign_id and not rule_id:
        raise ValueError(f"{where}precedent must name what it cites — a campaign_id for a "
                         f"departure from precedent, or a rule_id for a guardrail breach")
    quote = _bounded(value.get("quote"), "precedent.quote", _MAX_QUOTE, where=where)
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
    cleaned = {"quote": quote, "layer": layer}
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


def _clean_resolved(value) -> list:
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
        # §5.4 and §6.3 both need "this specific earlier finding is now addressed", and
        # free text cannot be joined back to one.
        if item.get("finding_id"):
            entry["finding_id"] = item["finding_id"]
        cleaned.append(entry)
    return cleaned


def get_evaluation(conn, *, evaluation_id: str, severity: Optional[str] = None,
                   kind: Optional[str] = None) -> dict:
    """Read a stored judgment back, optionally narrowed to one severity or kind.

    "The detail is fetched on demand" was true of the storage and false of the surface:
    there was no read path at all, so "show me the blocking items" worked only while the
    findings were still in the context window that produced them. The next session, another
    person, and any later comparison had no way to reach them."""
    ev = store.get_evaluation(conn, evaluation_id)
    if not ev:
        return {"error": f"evaluation {evaluation_id} not found"}
    findings = ev.get("findings") or []
    if severity:
        findings = [f for f in findings if f.get("severity") == severity]
    if kind:
        findings = [f for f in findings if f.get("kind") == kind]
    return {
        "evaluation_id": ev["id"],
        "subject_title": ev["subject_title"],
        "verdict": ev["verdict"],
        "summary": ev["summary"],
        "approve_if": ev.get("approve_if"),
        "closest_precedent": ev.get("closest_precedent"),
        "findings": findings,
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
    resolved = _clean_resolved(resolved)

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
        kind = finding.get("kind")
        if kind is not None and kind not in _KINDS:
            raise ValueError(f"{where}kind must be one of {list(_KINDS)}, got {kind!r}")
        # A rule either applies or it does not. "You broke a rule, but never mind" is the
        # shape of a finding written to avoid an argument.
        if kind == "guardrail_breach" and severity == "note":
            raise ValueError(f"{where}a guardrail_breach cannot be a note — if the rule "
                             f"applies the finding is blocking, and if it does not apply "
                             f"this is not a guardrail breach")
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
            "basis": basis,
            "category": _category(finding.get("category"), where),
            "finding": text,
            "detail": _bounded(finding.get("detail"), "'detail'", _MAX_DETAIL, where=where),
            "precedent": _clean_precedent(finding.get("precedent"), where),
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
    return {
        "evaluation_id": eid,
        "verdict": verdict,
        "summary": summary,
        "counts": counts,
        "closest_precedent": closest_precedent,
        "findings": [{k: f[k] for k in ("id", "severity", "kind", "finding", "fix")}
                     for f in cleaned if f["severity"] in ("blocking", "should_fix")],
        "approve_if": approve_if,
        # §5.2: the three things anyone actually does after a judgment, prefilled. The
        # supersession offer only appears when there IS an earlier version — an approval of
        # a new brief supersedes nothing, and an offer that is always there stops being read.
        "next_actions": actions.after_evaluation(
            subject_title=subject_title, evaluation_id=eid, verdict=verdict,
            campaign_id=campaign_id),
        "note": "Give the user the verdict, the one-line summary and what has to change. "
                "The reasoning behind any finding is in get_evaluation, not here. "
                "`next_actions` are offers — say them in your own words and act on the one "
                "the user picks; do not call them unasked.",
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
    records = len(store.outstanding_by_campaign(conn))
    return [notices.notice(
        "results_may_be_incomplete",
        affects=f"{records} record(s) are only partly searchable ({outstanding} items "
                f"still to index), so these results may be incomplete.",
        detail=f"{outstanding} unembedded items across {records} campaigns")]


def prepare_evaluation(conn, *, subject_title: str, proposal_text: str, top_k: int = 5,
                       record_type: Optional[str] = None, status: Optional[str] = None,
                       tags: Optional[Union[str, dict, list]] = None, match_all_tags: bool = False,
                       region: Optional[str] = None, market: Optional[str] = None,
                       markets: Optional[Union[str, list]] = None, collection: Optional[str] = None,
                       full_detail: bool = True) -> dict:
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
            "Reason over this evidence, then call save_evaluation with a verdict "
            "(approve / revise / reject), a one-line summary, and one short finding per "
            "problem — each with its severity, its kind, and a quote from the campaign or "
            "rule it is anchored to, CITING specific campaign_ids above. Predicted CTR/ROI "
            "ranges go in `predictions`. Weight concluded campaigns (those with metrics) "
            "most."
        ),
        "campaigns_with_outcomes": [e["campaign_id"] for e in concluded],
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
