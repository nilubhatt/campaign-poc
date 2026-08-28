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

import config
import embedding
import extract
import store
import vectorstore


# ── ingest ───────────────────────────────────────────────────────────────────

def ingest_campaign(conn, *, title: str, detail: Optional[str] = None,
                    deck_text: Optional[str] = None, kind: str = "concluded",
                    asset_ref: Optional[dict] = None) -> dict:
    """
    Store a past/proposed campaign and embed it for search.

    LLM-first: from Claude Web, pass `deck_text` (the text Claude already read from the
    attached PDF/PPTX) plus freeform `detail`. Alternatively pass `asset_ref`
    ({asset_id} from POST /upload, {path} local, or {filename,base64} inline) and the
    server extracts the text itself. Returns the stored campaign summary.
    """
    warnings: list[str] = []
    stored_path = None

    if asset_ref and not deck_text:
        path, w = _resolve_asset(asset_ref)
        warnings += w
        if path:
            text, w2 = extract.extract_text(path)
            warnings += w2
            deck_text = text
            stored_path = _keep_asset(path)

    cid = store.insert_campaign(
        conn, title=title, kind=kind, detail=detail, deck_text=deck_text, asset_path=stored_path
    )

    embed_source = "\n\n".join(p for p in (title, detail, deck_text) if p).strip()
    if embed_source:
        try:
            store.set_embedding(conn, cid, embedding.embed(embed_source))
        except Exception as exc:
            warnings.append(f"campaign stored but not embedded (search will miss it): {exc}")

    return {"campaign_id": cid, "title": title, "kind": kind,
            "embedded": bool(embed_source and not any("not embedded" in w for w in warnings)),
            "warnings": warnings}


# ── retrieval / evidence for Claude ──────────────────────────────────────────

def find_similar(conn, *, text: Optional[str] = None, campaign_id: Optional[str] = None,
                 top_k: int = 5) -> list[dict]:
    """
    Rank prior campaigns by semantic similarity to `text` (or to an existing campaign's
    own vector). Returns evidence rows — id, title, similarity, detail, and metrics — for
    Claude to reason over. Excludes the query campaign itself.
    """
    exclude = {campaign_id} if campaign_id else set()
    if campaign_id:
        row = conn.execute("SELECT detail, deck_text FROM campaigns WHERE id = ?", (campaign_id,)).fetchone()
        text = "\n\n".join(p for p in ((row["detail"] if row else None),
                                       (row["deck_text"] if row else None)) if p)
    if not text or not text.strip():
        raise ValueError("provide text (or a campaign_id that has content) to search by")

    qvec = embedding.embed(text)
    hits = vectorstore.search(conn, qvec, top_k=top_k, exclude=exclude)
    evidence = []
    for cid, sim in hits:
        c = store.get_campaign(conn, cid)
        if not c:
            continue
        evidence.append({
            "campaign_id": cid,
            "title": c["title"],
            "kind": c["kind"],
            "similarity": round(sim, 4),
            "detail": c["detail"],
            "deck_excerpt": (c["deck_text"] or "")[:1500],
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
