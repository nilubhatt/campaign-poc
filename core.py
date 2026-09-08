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
import store
import vectorstore

# Over-fetch factor for chunk-level search before rolling up to campaigns (§6.1): several
# chunks from the same campaign can rank highly, so fetch more than top_k chunks to still
# surface top_k *distinct* campaigns.
_SEARCH_OVERFETCH = 4


# ── ingest ───────────────────────────────────────────────────────────────────

def ingest_campaign(conn, *, title: str, detail: Optional[str] = None,
                    deck_text: Optional[str] = None, kind: str = "concluded",
                    asset_ref: Optional[dict] = None) -> dict:
    """
    Store a past/proposed campaign, chunk it, and embed each chunk for search (§6.1).

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
        conn, title=title, kind=kind, detail=detail, deck_text=deck_text, asset_path=stored_path
    )

    if not units and deck_text:
        # LLM-first path: Claude passed one flat string with no page/slide boundaries —
        # split on blank lines so chunking.pack() has units to work with.
        units = deck_text.split("\n\n")

    summary = "\n\n".join(p for p in (title, detail) if p).strip()
    chunk_texts = chunking.pack(([summary] if summary else []) + units)

    if not chunk_texts:
        warnings.append("nothing to embed (no title/detail/deck_text)")
        return {"campaign_id": cid, "title": title, "kind": kind, "embedded": False,
                "chunks_total": 0, "chunks_embedded": 0, "warnings": warnings}

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
        "campaign_id": cid, "title": title, "kind": kind,
        "embedded": embedded_count > 0,
        "chunks_total": len(chunk_texts), "chunks_embedded": embedded_count,
        "warnings": warnings,
    }


# ── retrieval / evidence for Claude ──────────────────────────────────────────

def find_similar(conn, *, text: Optional[str] = None, campaign_id: Optional[str] = None,
                 top_k: int = 5) -> list[dict]:
    """
    Rank prior campaigns by semantic similarity to `text` (or to an existing campaign's
    own content). Searches at chunk level (§6.1 — one vector per slide/section) and rolls
    up to the best-matching chunk per campaign, so a long deck can still match on the one
    section that's actually relevant. Returns evidence rows — id, title, similarity,
    detail, the matched excerpt, and metrics — for Claude to reason over. Excludes the
    query campaign's own chunks.
    """
    exclude_chunks: set[str] = set()
    if campaign_id:
        row = conn.execute("SELECT detail, deck_text FROM campaigns WHERE id = ?", (campaign_id,)).fetchone()
        text = "\n\n".join(p for p in ((row["detail"] if row else None),
                                       (row["deck_text"] if row else None)) if p)
        exclude_chunks = set(store.get_chunk_ids_for_campaign(conn, campaign_id))
    if not text or not text.strip():
        raise ValueError("provide text (or a campaign_id that has content) to search by")

    qvec = embedding.embed(text)
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
            "kind": c["kind"],
            "similarity": round(sim, 4),
            "detail": c["detail"],
            "matched_excerpt": matched["text"] if matched else "",
            "metrics": [{"detail": m["detail"], "structured": m["structured"]} for m in c["metrics"]],
        })
    return evidence


def prepare_evaluation(conn, *, subject_title: str, proposal_text: str, top_k: int = 5) -> dict:
    """
    Package the evidence Claude needs to judge a new proposal: the most similar prior
    campaigns WITH their outcomes. Claude reads this, produces its analysis citing specific
    priors, then calls save_evaluation. This tool does NOT itself judge.
    """
    evidence = find_similar(conn, text=proposal_text, top_k=top_k)
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
