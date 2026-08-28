"""
MCP server — the API Claude calls. LLM-first: tools are driven by text/fields Claude has in
a conversation. Claude orchestrates the whole memory → judge → reconcile flow.

Tool map:
  upload_campaign   — store a past/proposed campaign (deck_text Claude read, + freeform detail)
  add_metrics       — attach post-conclusion outcomes to a campaign
  list_campaigns    — browse the memory
  get_campaign      — full detail + metrics for one campaign
  find_similar_campaigns — semantic search: prior campaigns like a description/campaign
  prepare_evaluation — evidence package (similar priors + outcomes) for Claude to judge
  save_evaluation   — persist Claude's judgment + the priors it cited + its predictions
  reconcile_evaluation — fetch a past judgment + accept the real metrics, for Claude to compare
  save_reconciliation — persist Claude's prediction-vs-actual lesson
"""
from __future__ import annotations

import json
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

import core
import store

app = Server("campaign-intelligence")

_ASSET_REF = {
    "type": "object",
    "description": "Optional file reference if not passing deck_text: {path} local, {asset_id} from POST /upload, or {filename,base64} inline",
    "properties": {
        "path": {"type": "string"},
        "asset_id": {"type": "string"},
        "filename": {"type": "string"},
        "base64": {"type": "string"},
    },
}


def _ok(data: Any) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(data, default=str, indent=2))]


def _err(msg: str) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps({"error": msg}))]


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="upload_campaign",
            description=(
                "Store a past or proposed marketing campaign in the memory. From Claude Web, "
                "pass deck_text (the text you read from the attached PDF/PPTX) plus any freeform "
                "detail you have (brief, audience, budget, channel, timeline). For concluded "
                "campaigns, add their results later with add_metrics. Returns the campaign_id."
            ),
            inputSchema={
                "type": "object",
                "required": ["title"],
                "properties": {
                    "title": {"type": "string"},
                    "detail": {"type": "string", "description": "Freeform — whatever detail you have; more is better"},
                    "deck_text": {"type": "string", "description": "Text of the deck/brief (what you read from the file)"},
                    "kind": {"type": "string", "enum": ["concluded", "proposal"], "default": "concluded"},
                    "asset_ref": _ASSET_REF,
                },
            },
        ),
        Tool(
            name="add_metrics",
            description=(
                "Attach post-conclusion outcomes/metrics to an existing campaign. Pass whatever "
                "you have as freeform `detail` (CTR, ROI, conversions, qualitative learnings), and "
                "optionally a structured object for the numbers you want machine-readable."
            ),
            inputSchema={
                "type": "object",
                "required": ["campaign_id", "detail"],
                "properties": {
                    "campaign_id": {"type": "string"},
                    "detail": {"type": "string", "description": "Freeform metrics + learnings"},
                    "structured": {"type": "object", "description": "Optional: {ctr, roi, conversions, ...}"},
                },
            },
        ),
        Tool(
            name="list_campaigns",
            description="List campaigns in the memory. Optionally filter by kind (concluded|proposal).",
            inputSchema={
                "type": "object",
                "properties": {"kind": {"type": "string", "enum": ["concluded", "proposal"]}},
            },
        ),
        Tool(
            name="get_campaign",
            description="Full detail + all metrics for one campaign by id.",
            inputSchema={"type": "object", "required": ["campaign_id"],
                         "properties": {"campaign_id": {"type": "string"}}},
        ),
        Tool(
            name="find_similar_campaigns",
            description=(
                "Semantic search: find prior campaigns most similar to a description (`text`) or to "
                "an existing campaign (`campaign_id`). Returns ranked evidence — title, similarity, "
                "detail, and metrics — for you to reason over."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Describe the campaign to match"},
                    "campaign_id": {"type": "string", "description": "Or match an existing campaign by id"},
                    "top_k": {"type": "integer", "default": 5},
                },
            },
        ),
        Tool(
            name="prepare_evaluation",
            description=(
                "Evaluate a NEW campaign proposal against the memory. Returns the most similar prior "
                "campaigns WITH their outcomes as an evidence package. Read it, then produce your "
                "judgment (predicted CTR/ROI ranges, risks, proceed/revise/reject) CITING specific "
                "campaign_ids, and call save_evaluation. This tool gathers evidence; the judgment is yours."
            ),
            inputSchema={
                "type": "object",
                "required": ["subject_title", "proposal_text"],
                "properties": {
                    "subject_title": {"type": "string"},
                    "proposal_text": {"type": "string", "description": "The new proposal's brief/deck text"},
                    "top_k": {"type": "integer", "default": 5},
                },
            },
        ),
        Tool(
            name="save_evaluation",
            description=(
                "Persist your judgment of a campaign so it becomes memory. Include the specific "
                "campaign_ids you cited and, if you gave them, structured predictions — so a later "
                "reconcile_evaluation can score prediction vs. actual. Returns the evaluation_id."
            ),
            inputSchema={
                "type": "object",
                "required": ["subject_title", "analysis"],
                "properties": {
                    "subject_title": {"type": "string"},
                    "analysis": {"type": "string", "description": "Your full judgment/reasoning"},
                    "cited_ids": {"type": "array", "items": {"type": "string"}, "description": "campaign_ids you reasoned from"},
                    "predictions": {"type": "object", "description": "Optional: {predicted_ctr_range, predicted_roi_range, recommendation}"},
                    "campaign_id": {"type": "string", "description": "Optional: id if this proposal is also stored as a campaign"},
                },
            },
        ),
        Tool(
            name="reconcile_evaluation",
            description=(
                "Start closing the loop on a past judgment. Give the evaluation_id and the real "
                "post-campaign metrics; returns your original analysis + predictions alongside the "
                "actuals. Compare them, then call save_reconciliation with the lesson."
            ),
            inputSchema={
                "type": "object",
                "required": ["evaluation_id", "actual"],
                "properties": {
                    "evaluation_id": {"type": "string"},
                    "actual": {"type": "string", "description": "The real metrics/outcomes, freeform"},
                },
            },
        ),
        Tool(
            name="save_reconciliation",
            description=(
                "Persist your prediction-vs-actual comparison and the lesson learned, so future "
                "evaluations are better calibrated. Returns the reconciliation id."
            ),
            inputSchema={
                "type": "object",
                "required": ["evaluation_id", "comparison"],
                "properties": {
                    "evaluation_id": {"type": "string"},
                    "comparison": {"type": "string", "description": "What you predicted vs. what happened + the lesson"},
                    "actual": {"type": "string", "description": "Optional: the actual metrics, for the record"},
                },
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    conn = store.connect()
    try:
        if name == "upload_campaign":
            return _ok(core.ingest_campaign(conn, **arguments))

        if name == "add_metrics":
            mid = store.add_metrics(conn, arguments["campaign_id"],
                                    detail=arguments.get("detail"),
                                    structured=arguments.get("structured"))
            return _ok({"metrics_id": mid, "campaign_id": arguments["campaign_id"], "status": "stored"})

        if name == "list_campaigns":
            rows = store.list_campaigns(conn, kind=arguments.get("kind"))
            slim = [{"campaign_id": r["id"], "title": r["title"], "kind": r["kind"],
                     "embedded": r["embedded"]} for r in rows]
            return _ok({"count": len(slim), "campaigns": slim})

        if name == "get_campaign":
            c = store.get_campaign(conn, arguments["campaign_id"])
            return _ok(c) if c else _err(f"campaign {arguments['campaign_id']} not found")

        if name == "find_similar_campaigns":
            return _ok({"matches": core.find_similar(conn, **arguments)})

        if name == "prepare_evaluation":
            return _ok(core.prepare_evaluation(conn, **arguments))

        if name == "save_evaluation":
            eid = store.insert_evaluation(
                conn, subject_title=arguments["subject_title"], analysis=arguments["analysis"],
                campaign_id=arguments.get("campaign_id"), cited_ids=arguments.get("cited_ids"),
                predictions=arguments.get("predictions"))
            return _ok({"evaluation_id": eid, "status": "saved"})

        if name == "reconcile_evaluation":
            ev = store.get_evaluation(conn, arguments["evaluation_id"])
            if not ev:
                return _err(f"evaluation {arguments['evaluation_id']} not found")
            return _ok({
                "evaluation_id": ev["id"],
                "subject_title": ev["subject_title"],
                "original_analysis": ev["analysis"],
                "predictions": json.loads(ev["predictions"]) if ev["predictions"] else None,
                "cited_ids": json.loads(ev["cited_ids"]) if ev["cited_ids"] else [],
                "actual": arguments["actual"],
                "note": "Compare predictions to actual, then call save_reconciliation with the lesson.",
            })

        if name == "save_reconciliation":
            rid = store.insert_reconciliation(
                conn, evaluation_id=arguments["evaluation_id"],
                comparison=arguments["comparison"], actual=arguments.get("actual"))
            return _ok({"reconciliation_id": rid, "status": "saved"})

        return _err(f"unknown tool: {name}")
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}")
    finally:
        conn.close()
