"""
MCP server (mcp 2.x MCPServer) — the API Claude calls. LLM-first: tools are typed Python
functions Claude drives with text/fields it has in a conversation. Claude orchestrates the
whole memory → judge → reconcile flow.

The HTTP surface is MCPServer's built-in Streamable HTTP app (see http_app.py) — what a
Claude Web / cowork custom connector points at.
"""
from __future__ import annotations

from typing import Any, Optional

from mcp.server.mcpserver import MCPServer

import core
import store

mcp = MCPServer("campaign-intelligence")


@mcp.tool()
def upload_campaign(title: str, detail: Optional[str] = None, deck_text: Optional[str] = None,
                    record_type: str = "campaign", status: Optional[str] = None,
                    tags: Optional[list] = None, region: Optional[str] = None,
                    market: Optional[str] = None, supersedes: Optional[str] = None,
                    asset_ref: Optional[dict] = None) -> dict:
    """Store a past or proposed campaign in the memory. From Claude Web, pass deck_text (the
    text you read from the attached PDF/PPTX) plus any freeform detail you have (brief,
    audience, budget, channel, timeline). The server chunks and embeds it per slide/section
    for search.

    record_type is 'campaign' (default), 'reference' (background material, not itself a
    campaign), or 'stub' (a placeholder record). status is 'proposed', 'in_flight', or
    'concluded' — defaults to 'concluded' for record_type='campaign', otherwise unset.
    tags is a freeform list of strings (no fixed taxonomy); region/market are freeform too
    (e.g. region='APAC', market='Philippines'). These structured fields let
    find_similar_campaigns / prepare_evaluation filter before ranking by similarity.

    Pass supersedes=<campaign_id> if this record replaces an existing one (e.g. a corrected
    deck) — the old record is then excluded from future search evidence, so it stops
    confusing retrieval, without being deleted.

    Add results later with add_metrics. Returns the campaign_id plus
    chunks_total/chunks_embedded (partial embedding failures are reported per-chunk in
    warnings, not silently)."""
    conn = store.connect()
    try:
        return core.ingest_campaign(conn, title=title, detail=detail, deck_text=deck_text,
                                    record_type=record_type, status=status, tags=tags,
                                    region=region, market=market, supersedes=supersedes,
                                    asset_ref=asset_ref)
    finally:
        conn.close()


@mcp.tool()
def update_campaign(campaign_id: str, title: Optional[str] = None, detail: Optional[str] = None,
                    record_type: Optional[str] = None, status: Optional[str] = None,
                    tags: Optional[list] = None, region: Optional[str] = None,
                    market: Optional[str] = None) -> dict:
    """Edit a campaign's metadata (title, detail, record_type, status, tags, region, market).
    Only the fields you pass change. tags, if given, fully REPLACES the existing list (not a
    merge) — pass the complete new list. Does NOT change deck_text/chunks/embeddings; for
    content changes, upload a new record and pass supersedes=campaign_id instead."""
    conn = store.connect()
    try:
        ok = store.update_campaign(conn, campaign_id, title=title, detail=detail,
                                   record_type=record_type, status=status, tags=tags,
                                   region=region, market=market)
        if not ok:
            return {"error": f"campaign {campaign_id} not found"}
        return store.get_campaign(conn, campaign_id)
    finally:
        conn.close()


@mcp.tool()
def delete_campaign(campaign_id: str) -> dict:
    """Permanently delete a campaign and its chunks/vectors/metrics. Evaluations that cited
    it are kept but detached. If this record superseded another one, that older record is
    restored to active (no longer excluded from search)."""
    conn = store.connect()
    try:
        ok = store.delete_campaign(conn, campaign_id)
        if not ok:
            return {"error": f"campaign {campaign_id} not found"}
        return {"campaign_id": campaign_id, "status": "deleted"}
    finally:
        conn.close()


@mcp.tool()
def upload_image_asset(campaign_id: str, asset_ref: dict) -> dict:
    """Attach an image (hero shot, creative asset) to a campaign and fingerprint it
    (perceptual hash) for creative-reuse detection. asset_ref is {asset_id} from POST
    /upload, {path} local, or {filename, base64} inline. Consider calling
    check_image_provenance first if you want to flag reuse before attaching it."""
    conn = store.connect()
    try:
        return core.ingest_image_asset(conn, campaign_id=campaign_id, asset_ref=asset_ref)
    finally:
        conn.close()


@mcp.tool()
def check_image_provenance(asset_ref: dict, campaign_id: Optional[str] = None) -> dict:
    """Check whether an image matches one already in the memory — same/near-same photo,
    even after resize/recompress/light crop (perceptual hashing; catches exact reuse, not
    aesthetic similarity). Works before the image is stored. Pass campaign_id (the campaign
    this image is headed for) to exclude that campaign's own assets and get a flag when a
    match comes from a *different* region — the real question is usually not "does this image
    exist" but "does this image belong to a different region than where it's being used.\""""
    conn = store.connect()
    try:
        return core.check_image_provenance(conn, asset_ref=asset_ref, campaign_id=campaign_id)
    finally:
        conn.close()


@mcp.tool()
def add_metrics(campaign_id: str, detail: str, structured: Optional[dict] = None,
                metric_type: str = "actual") -> dict:
    """Attach outcomes to a campaign. Pass whatever you have as freeform detail (CTR, ROI,
    conversions, qualitative learnings) and optionally a structured object for
    machine-readable numbers. metric_type is 'actual' (post-conclusion results, the default)
    or 'predicted' (a forecast/target set before launch) — reconcile_evaluation only pulls
    'actual' metrics automatically."""
    conn = store.connect()
    try:
        mid = store.add_metrics(conn, campaign_id, detail=detail, structured=structured,
                               metric_type=metric_type)
        return {"metrics_id": mid, "campaign_id": campaign_id, "status": "stored"}
    finally:
        conn.close()


@mcp.tool()
def bulk_import_metrics(rows: list) -> dict:
    """Load a KPI workbook in one call instead of one add_metrics per row. Each row is an
    object identifying its campaign by campaign_id (preferred) or title (exact,
    case-insensitive — ambiguous or unmatched titles are reported as errors, never guessed),
    plus detail/structured/metric_type like add_metrics. Read the workbook yourself (CSV,
    pasted table, whatever you have) and pass the rows here. Returns {imported, errors} —
    valid rows import even if others fail."""
    conn = store.connect()
    try:
        return store.bulk_import_metrics(conn, rows)
    finally:
        conn.close()


@mcp.tool()
def list_campaigns(record_type: Optional[str] = None, status: Optional[str] = None) -> dict:
    """List records in the memory. Optionally filter by record_type ('campaign', 'reference',
    'stub') and/or status ('proposed', 'in_flight', 'concluded')."""
    conn = store.connect()
    try:
        rows = store.list_campaigns(conn, record_type=record_type, status=status)
        return {"count": len(rows), "campaigns": [
            {"campaign_id": r["id"], "title": r["title"], "record_type": r["record_type"],
             "status": r["status"], "region": r["region"], "market": r["market"],
             "embedded": r["embedded"], "has_metrics": r["has_metrics"],
             "has_evaluations": r["has_evaluations"]}
            for r in rows]}
    finally:
        conn.close()


@mcp.tool()
def get_campaign(campaign_id: str) -> dict:
    """Full detail + all metrics for one campaign by id."""
    conn = store.connect()
    try:
        c = store.get_campaign(conn, campaign_id)
        return c or {"error": f"campaign {campaign_id} not found"}
    finally:
        conn.close()


@mcp.tool()
def find_similar_campaigns(text: Optional[str] = None, campaign_id: Optional[str] = None,
                           top_k: int = 5, record_type: Optional[str] = None,
                           status: Optional[str] = None, tags: Optional[list] = None,
                           region: Optional[str] = None, market: Optional[str] = None,
                           full_detail: bool = False) -> dict:
    """Semantic search: find prior campaigns most similar to a description (text) or to an
    existing campaign (campaign_id). Matches at the slide/section level and rolls up to the
    best-matching campaign, so long decks match on the relevant part.

    Pass record_type/status/tags/region/market to filter to that criteria FIRST, then rank
    by similarity within it — e.g. status='concluded', region='APAC' to only weigh concluded
    APAC precedent instead of everything in the memory. Returns ranked evidence — title,
    status/tags/region/market, similarity, detail, the matched excerpt, and metrics — for
    you to reason over. detail is trimmed by default (detail_truncated flags it) — pass
    full_detail=True, or call get_campaign, for the untrimmed brief."""
    conn = store.connect()
    try:
        return {"matches": core.find_similar(conn, text=text, campaign_id=campaign_id,
                                             top_k=top_k, record_type=record_type,
                                             status=status, tags=tags, region=region,
                                             market=market, full_detail=full_detail)}
    finally:
        conn.close()


@mcp.tool()
def prepare_evaluation(subject_title: str, proposal_text: str, top_k: int = 5,
                       record_type: Optional[str] = None, status: Optional[str] = None,
                       tags: Optional[list] = None, region: Optional[str] = None,
                       market: Optional[str] = None, full_detail: bool = True) -> dict:
    """Evaluate a NEW campaign proposal against the memory. Returns the most similar prior
    campaigns WITH their outcomes as an evidence package (full detail by default — this is
    for judging, not browsing). Optionally narrow to structured criteria first (e.g.
    region='APAC') so only relevant precedent is weighed. Read it, then produce your
    judgment (predicted CTR/ROI ranges, risks, proceed/revise/reject) CITING specific
    campaign_ids, and call save_evaluation. This tool gathers evidence; the judgment is
    yours."""
    conn = store.connect()
    try:
        return core.prepare_evaluation(conn, subject_title=subject_title,
                                       proposal_text=proposal_text, top_k=top_k,
                                       record_type=record_type, status=status, tags=tags,
                                       region=region, market=market, full_detail=full_detail)
    finally:
        conn.close()


@mcp.tool()
def save_evaluation(subject_title: str, analysis: str, cited_ids: Optional[list] = None,
                    predictions: Optional[dict] = None, campaign_id: Optional[str] = None) -> dict:
    """Persist your judgment of a campaign so it becomes memory. Include the specific
    campaign_ids you cited and, if given, structured predictions — so a later
    reconcile_evaluation can score prediction vs. actual. Returns the evaluation_id."""
    conn = store.connect()
    try:
        eid = store.insert_evaluation(conn, subject_title=subject_title, analysis=analysis,
                                      campaign_id=campaign_id, cited_ids=cited_ids,
                                      predictions=predictions)
        return {"evaluation_id": eid, "status": "saved"}
    finally:
        conn.close()


@mcp.tool()
def reconcile_evaluation(evaluation_id: str, actual: Optional[str] = None) -> dict:
    """Start closing the loop on a past judgment. If actual metrics are already on file for
    this campaign (via add_metrics/bulk_import_metrics), they're pulled automatically —
    otherwise pass actual= with the real post-campaign metrics yourself. Returns your
    original analysis + predictions alongside the actuals. Compare them, then call
    save_reconciliation with the lesson."""
    conn = store.connect()
    try:
        return core.reconcile_evaluation(conn, evaluation_id=evaluation_id, actual=actual)
    finally:
        conn.close()


@mcp.tool()
def save_reconciliation(evaluation_id: str, comparison: str, actual: Optional[str] = None) -> dict:
    """Persist your prediction-vs-actual comparison and the lesson learned, so future
    evaluations are better calibrated. Returns the reconciliation id."""
    conn = store.connect()
    try:
        rid = store.insert_reconciliation(conn, evaluation_id=evaluation_id,
                                          comparison=comparison, actual=actual)
        return {"reconciliation_id": rid, "status": "saved"}
    finally:
        conn.close()
