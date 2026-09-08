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
                    kind: str = "concluded", asset_ref: Optional[dict] = None) -> dict:
    """Store a past or proposed campaign in the memory. From Claude Web, pass deck_text (the
    text you read from the attached PDF/PPTX) plus any freeform detail you have (brief,
    audience, budget, channel, timeline). The server chunks and embeds it per slide/section
    for search. Add results later with add_metrics. kind is 'concluded' or 'proposal'.
    Returns the campaign_id plus chunks_total/chunks_embedded (partial embedding failures
    are reported per-chunk in warnings, not silently)."""
    conn = store.connect()
    try:
        return core.ingest_campaign(conn, title=title, detail=detail, deck_text=deck_text,
                                    kind=kind, asset_ref=asset_ref)
    finally:
        conn.close()


@mcp.tool()
def add_metrics(campaign_id: str, detail: str, structured: Optional[dict] = None) -> dict:
    """Attach post-conclusion outcomes to a campaign. Pass whatever you have as freeform
    detail (CTR, ROI, conversions, qualitative learnings) and optionally a structured object
    for machine-readable numbers."""
    conn = store.connect()
    try:
        mid = store.add_metrics(conn, campaign_id, detail=detail, structured=structured)
        return {"metrics_id": mid, "campaign_id": campaign_id, "status": "stored"}
    finally:
        conn.close()


@mcp.tool()
def list_campaigns(kind: Optional[str] = None) -> dict:
    """List campaigns in the memory. Optionally filter by kind ('concluded' or 'proposal')."""
    conn = store.connect()
    try:
        rows = store.list_campaigns(conn, kind=kind)
        return {"count": len(rows), "campaigns": [
            {"campaign_id": r["id"], "title": r["title"], "kind": r["kind"], "embedded": r["embedded"]}
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
                           top_k: int = 5) -> dict:
    """Semantic search: find prior campaigns most similar to a description (text) or to an
    existing campaign (campaign_id). Matches at the slide/section level and rolls up to the
    best-matching campaign, so long decks match on the relevant part. Returns ranked
    evidence — title, similarity, detail, the matched excerpt, and metrics — for you to
    reason over."""
    conn = store.connect()
    try:
        return {"matches": core.find_similar(conn, text=text, campaign_id=campaign_id, top_k=top_k)}
    finally:
        conn.close()


@mcp.tool()
def prepare_evaluation(subject_title: str, proposal_text: str, top_k: int = 5) -> dict:
    """Evaluate a NEW campaign proposal against the memory. Returns the most similar prior
    campaigns WITH their outcomes as an evidence package. Read it, then produce your judgment
    (predicted CTR/ROI ranges, risks, proceed/revise/reject) CITING specific campaign_ids, and
    call save_evaluation. This tool gathers evidence; the judgment is yours."""
    conn = store.connect()
    try:
        return core.prepare_evaluation(conn, subject_title=subject_title,
                                       proposal_text=proposal_text, top_k=top_k)
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
def reconcile_evaluation(evaluation_id: str, actual: str) -> dict:
    """Start closing the loop on a past judgment. Give the evaluation_id and the real
    post-campaign metrics; returns your original analysis + predictions alongside the actuals.
    Compare them, then call save_reconciliation with the lesson."""
    import json
    conn = store.connect()
    try:
        ev = store.get_evaluation(conn, evaluation_id)
        if not ev:
            return {"error": f"evaluation {evaluation_id} not found"}
        return {
            "evaluation_id": ev["id"],
            "subject_title": ev["subject_title"],
            "original_analysis": ev["analysis"],
            "predictions": json.loads(ev["predictions"]) if ev["predictions"] else None,
            "cited_ids": json.loads(ev["cited_ids"]) if ev["cited_ids"] else [],
            "actual": actual,
            "note": "Compare predictions to actual, then call save_reconciliation with the lesson.",
        }
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
