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
from pathlib import Path
from typing import Optional

import chunking
import config
import embedding
import extract
import images
import store
import vectorstore

# Over-fetch factor for chunk-level search before rolling up to campaigns (§6.1): several
# chunks from the same campaign can rank highly, so fetch more than top_k chunks to still
# surface top_k *distinct* campaigns.
_SEARCH_OVERFETCH = 4


# ── ingest ───────────────────────────────────────────────────────────────────

def ingest_campaign(conn, *, title: str, detail: Optional[str] = None,
                    deck_text: Optional[str] = None, record_type: str = "campaign",
                    status: Optional[str] = None, tags: Optional[list[str]] = None,
                    region: Optional[str] = None, market: Optional[str] = None,
                    supersedes: Optional[str] = None, asset_ref: Optional[dict] = None) -> dict:
    """
    Store a past/proposed campaign, chunk it, and embed each chunk for search (§6.1).

    record_type distinguishes an actual campaign from background reference material or a
    placeholder stub (§6.3); status tracks a campaign's own lifecycle. tags/region/market
    are structured fields used to filter BEFORE similarity ranking (§6.2).

    LLM-first: from Claude Web, pass `deck_text` (the text Claude already read from the
    attached PDF/PPTX) plus freeform `detail`. Alternatively pass `asset_ref`
    ({asset_id} from POST /upload, {path} local, or {filename,base64} inline) and the
    server extracts the text itself — that path also preserves per-page/slide boundaries
    as the natural chunks. Returns the stored campaign summary.
    """
    warnings: list[str] = []
    stored_path = None
    units: list[str] = []  # natural per-page/slide units, when extraction ran

    if asset_ref and not deck_text:
        path, w = _resolve_asset(asset_ref)
        warnings += w
        if path:
            units, w2 = extract.extract_units(path)
            warnings += w2
            deck_text = "\n\n".join(units)
            stored_path = _keep_asset(path)

    cid = store.insert_campaign(
        conn, title=title, record_type=record_type, status=status, tags=tags, region=region,
        market=market, supersedes=supersedes, detail=detail, deck_text=deck_text,
        asset_path=stored_path,
    )

    if not units and deck_text:
        # LLM-first path: Claude passed one flat string with no page/slide boundaries —
        # split on blank lines so chunking.pack() has units to work with.
        units = deck_text.split("\n\n")

    summary = "\n\n".join(p for p in (title, detail) if p).strip()
    chunk_texts = chunking.pack(([summary] if summary else []) + units)

    if not chunk_texts:
        warnings.append("nothing to embed (no title/detail/deck_text)")
        return {"campaign_id": cid, "title": title, "record_type": record_type,
                "embedded": False, "chunks_total": 0, "chunks_embedded": 0, "warnings": warnings}

    chunk_ids = store.insert_chunks(conn, cid, chunk_texts)
    embedded_count = 0
    for chunk_id, text in zip(chunk_ids, chunk_texts):
        try:
            vec = embedding.embed(text)
            vectorstore.add(conn, chunk_id, vec)
            store.set_chunk_embedded(conn, chunk_id)
            embedded_count += 1
        except Exception as exc:
            warnings.append(f"chunk {chunk_id} not embedded (search will miss it): {exc}")

    store.mark_embedded(conn, cid, embedded_count > 0)
    return {
        "campaign_id": cid, "title": title, "record_type": record_type,
        "embedded": embedded_count > 0,
        "chunks_total": len(chunk_texts), "chunks_embedded": embedded_count,
        "warnings": warnings,
    }


# ── retrieval / evidence for Claude ──────────────────────────────────────────

def find_similar(conn, *, text: Optional[str] = None, campaign_id: Optional[str] = None,
                 top_k: int = 5, record_type: Optional[str] = None, status: Optional[str] = None,
                 tags: Optional[list[str]] = None, region: Optional[str] = None,
                 market: Optional[str] = None) -> list[dict]:
    """
    Rank prior campaigns by semantic similarity to `text` (or to an existing campaign's
    own content). Searches at chunk level (§6.1 — one vector per slide/section) and rolls
    up to the best-matching chunk per campaign, so a long deck can still match on the one
    section that's actually relevant. Excludes the query campaign's own chunks.

    §6.2: if any of record_type/status/tags/region/market is given, campaigns are filtered
    to that structured criteria FIRST, then ranked by similarity only within that set —
    otherwise pure vector search on a single-brand corpus returns everything as "similar."
    Returns evidence rows — id, title, similarity, detail, the matched excerpt, and
    metrics — for Claude to reason over.
    """
    if campaign_id:
        row = conn.execute("SELECT detail, deck_text FROM campaigns WHERE id = ?", (campaign_id,)).fetchone()
        text = "\n\n".join(p for p in ((row["detail"] if row else None),
                                       (row["deck_text"] if row else None)) if p)
    if not text or not text.strip():
        raise ValueError("provide text (or a campaign_id that has content) to search by")

    qvec = embedding.embed(text)
    filters_given = any([record_type, status, tags, region, market])

    if filters_given:
        candidate_ids = store.filter_campaign_ids(
            conn, record_type=record_type, status=status, tags=tags, region=region,
            market=market, exclude_campaign_id=campaign_id,
        )
        if not candidate_ids:
            return []
        chunk_map = store.get_chunk_ids_for_campaigns(conn, candidate_ids)
        candidate_chunk_ids = [chid for chids in chunk_map.values() for chid in chids]
        vecs = vectorstore.get_many(conn, candidate_chunk_ids)
        hits = embedding.rank(qvec, list(vecs.items()), top_k=top_k * _SEARCH_OVERFETCH)
    else:
        exclude_chunks = set(store.get_chunk_ids_for_campaign(conn, campaign_id)) if campaign_id else set()
        superseded_ids = store.get_superseded_campaign_ids(conn)
        if superseded_ids:
            for chids in store.get_chunk_ids_for_campaigns(conn, list(superseded_ids)).values():
                exclude_chunks.update(chids)
        hits = vectorstore.search(conn, qvec, top_k=top_k * _SEARCH_OVERFETCH, exclude=exclude_chunks)

    chunk_to_campaign = store.map_chunks_to_campaigns(conn, [chunk_id for chunk_id, _ in hits])

    best: dict[str, tuple[float, str]] = {}
    for chunk_id, sim in hits:
        cid = chunk_to_campaign.get(chunk_id)
        if cid is None or (cid in best and sim <= best[cid][0]):
            continue
        best[cid] = (sim, chunk_id)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k]

    evidence = []
    for cid, (sim, chunk_id) in ranked:
        c = store.get_campaign(conn, cid)
        if not c:
            continue
        matched = store.get_chunk(conn, chunk_id)
        evidence.append({
            "campaign_id": cid,
            "title": c["title"],
            "record_type": c["record_type"],
            "status": c["status"],
            "tags": c["tags"],
            "region": c["region"],
            "market": c["market"],
            "similarity": round(sim, 4),
            "detail": c["detail"],
            "matched_excerpt": matched["text"] if matched else "",
            "metrics": [{"detail": m["detail"], "structured": m["structured"]} for m in c["metrics"]],
        })
    return evidence


def prepare_evaluation(conn, *, subject_title: str, proposal_text: str, top_k: int = 5,
                       record_type: Optional[str] = None, status: Optional[str] = None,
                       tags: Optional[list[str]] = None, region: Optional[str] = None,
                       market: Optional[str] = None) -> dict:
    """
    Package the evidence Claude needs to judge a new proposal: the most similar prior
    campaigns WITH their outcomes. Claude reads this, produces its analysis citing specific
    priors, then calls save_evaluation. This tool does NOT itself judge. Optionally narrow
    to structured criteria first (§6.2), e.g. region="APAC" to only weigh APAC precedent.
    """
    evidence = find_similar(conn, text=proposal_text, top_k=top_k, record_type=record_type,
                            status=status, tags=tags, region=region, market=market)
    concluded = [e for e in evidence if e["metrics"]]
    return {
        "subject_title": subject_title,
        "evidence_count": len(evidence),
        "evidence": evidence,
        "note": (
            "Reason over this evidence and produce your judgment: predicted CTR/ROI ranges, "
            "risks, and proceed/revise/reject — CITING specific campaign_ids above. "
            "Weight concluded campaigns (those with metrics) most. Then call save_evaluation."
        ),
        "campaigns_with_outcomes": [e["campaign_id"] for e in concluded],
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
        actual = "\n".join(m["detail"] or "" for m in actual_metrics if m["detail"])

    return {
        "evaluation_id": ev["id"],
        "subject_title": ev["subject_title"],
        "original_analysis": ev["analysis"],
        "predictions": json.loads(ev["predictions"]) if ev["predictions"] else None,
        "cited_ids": json.loads(ev["cited_ids"]) if ev["cited_ids"] else [],
        "actual": actual,
        "note": "Compare predictions to actual, then call save_reconciliation with the lesson.",
    }


# ── image assets / creative-reuse detection (§6.6) ───────────────────────────

def ingest_image_asset(conn, *, campaign_id: str, asset_ref: dict) -> dict:
    """Attach an image to a campaign and fingerprint it (perceptual hash) for creative-reuse
    detection. Storage still succeeds even if fingerprinting fails (e.g. a corrupt image) —
    the failure is reported, not swallowed."""
    path, warnings = _resolve_asset(asset_ref)
    if not path:
        return {"error": "; ".join(warnings) or "could not resolve asset_ref"}

    stored_name = _keep_asset(path)
    aid = store.insert_asset(conn, campaign_id, file_path=stored_name)
    try:
        h = images.phash(config.ASSET_DIR / stored_name)
        store.set_asset_fingerprint(conn, aid, h)
        return {"asset_id": aid, "campaign_id": campaign_id, "fingerprinted": True,
                "warnings": warnings}
    except Exception as exc:
        warnings.append(f"asset stored but not fingerprinted (reuse detection will miss it): {exc}")
        return {"asset_id": aid, "campaign_id": campaign_id, "fingerprinted": False,
                "warnings": warnings}


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
    path, warnings = _resolve_asset(asset_ref)
    if not path:
        return {"error": "; ".join(warnings) or "could not resolve asset_ref"}

    query_hash = images.phash(path)
    threshold = config.PHASH_MATCH_THRESHOLD if threshold is None else threshold
    current = store.get_campaign(conn, campaign_id) if campaign_id else None

    matches = []
    for cand in store.get_all_fingerprints(conn, exclude_campaign_id=campaign_id):
        dist = images.hamming_distance(query_hash, cand["phash"])
        if dist > threshold:
            continue
        c = store.get_campaign(conn, cand["campaign_id"])
        if not c:
            continue
        flag = None
        if current and current["region"] and c["region"] and current["region"].lower() != c["region"].lower():
            flag = f"used in a different region ({c['region']} vs {current['region']})"
        matches.append({
            "campaign_id": cand["campaign_id"], "title": c["title"], "region": c["region"],
            "asset_id": cand["asset_id"], "hamming_distance": dist, "flag": flag,
        })
    matches.sort(key=lambda m: m["hamming_distance"])
    return {"query_hash": query_hash, "matches": matches, "warnings": warnings}


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
    """Copy an ingested original into the asset store; return its stored name."""
    import shutil
    config.ensure_dirs()
    dest = config.ASSET_DIR / src.name
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest.name
