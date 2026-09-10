"""
MCP server (mcp 2.x MCPServer) — the API Claude calls. LLM-first: tools are typed Python
functions Claude drives with text/fields it has in a conversation. Claude orchestrates the
whole memory → judge → reconcile flow.

The HTTP surface is MCPServer's built-in Streamable HTTP app (see http_app.py) — what a
Claude Web / cowork custom connector points at.
"""
from __future__ import annotations

import functools
from typing import Literal, NotRequired, Optional, TypedDict, Union

from mcp.server.mcpserver import MCPServer

import core
import store

mcp = MCPServer("campaign-intelligence")


def _catch_value_errors(fn):
    """A validation ValueError raised from inside a tool body (store.py/core.py's input
    validation — invalid enum values, a 'verified' tag with no metrics behind it, a
    not-found id via find_similar, etc.) must not propagate as a raw exception: the MCP
    framework converts any exception other than its own ToolError into a generic "Error
    executing tool X" and discards the original message entirely (reviewed and reproduced
    over a real MCP client — none of the carefully-written validation messages in this
    codebase were reaching the caller). Convert to the same {"error": str(exc)} convention
    already used for the not-found cases, so the actual guidance reaches Claude instead of
    being silently swallowed."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as exc:
            return {"error": str(exc)}
    return wrapper

# Constrains the JSON schema the LLM sees for these params, instead of relying on prose in
# a docstring alone — a typo ("inflight") is now a schema-validation error, not a silent
# value that never matches any filter (review flagged this as the cheapest correctness win
# available; store.py enforces the same values server-side regardless).
RecordType = Literal["campaign", "reference", "stub"]
Status = Literal["proposed", "in_flight", "concluded"]
MetricType = Literal["actual", "predicted"]

# Suggested tag vocabulary (not enforced — tags stay freeform, this is guidance for the
# conversational intake). Two independent axes that commonly co-occur on the same campaign
# (a creative reaction AND a performance verdict): did we like it, and did it work. The
# "missing quadrant" — liked but underperformed, or disliked but performed well — is where
# the real lessons are; querying it needs tags=["liked","underperformed"],
# match_all_tags=True (see find_similar_campaigns).
SUGGESTED_TAGS = {
    "creative reaction": ["liked", "not_liked", "mixed_reaction"],
    "performance": ["performed_well", "underperformed", "performed_as_expected", "no_data_yet"],
}

# Each tag is a plain string (defaults to source='stated') or a {value, source} object —
# put in the actual JSON schema (not just prose) so the shape is self-documenting.
# "verified" means backed by real metric_type='actual' data (e.g. via
# add_metrics/bulk_import_metrics); "stated" means someone's claim with no data behind it.
# Mark a performance tag "verified" ONLY when real numbers actually back it — an unverified
# impression tagged as if it were evidence is worse than not tagging it at all, since the
# agent will weight it as if it were measured.
class TagObject(TypedDict):
    value: str
    source: NotRequired[Literal["verified", "stated"]]  # omitted -> defaults to 'stated'

TagInput = Union[str, TagObject]


@mcp.tool()
@_catch_value_errors
def upload_campaign(title: str, detail: Optional[str] = None, deck_text: Optional[str] = None,
                    record_type: RecordType = "campaign", status: Optional[Status] = None,
                    tags: Optional[list[TagInput]] = None, region: Optional[str] = None,
                    market: Optional[str] = None, collection: Optional[str] = None,
                    supersedes: Optional[str] = None, asset_ref: Optional[dict] = None,
                    confirm: bool = False) -> dict:
    """Store a past or proposed campaign in the memory.

    The user is a non-technical marketer, not someone filling out a form — have a
    conversation, don't demand structured fields. Ask things like: is this a *finished
    campaign or a future/proposed one* (record_type/status)? What do you *like* about it,
    what don't you like, what are you trying to *achieve* (fold into detail)? Where does it
    run (region/market)? Is it a market/version variant of something already in the memory
    (collection)? Any tags that fit — two independent axes that commonly BOTH apply to the
    same campaign: creative reaction (liked / not_liked / mixed_reaction) and performance
    (performed_well / underperformed / performed_as_expected / no_data_yet). If they answer
    in one free-text paragraph instead of field-by-field, parse it into these fields
    yourself rather than asking again.

    Then call this tool with confirm=False (the default) to get a PREVIEW — nothing is
    stored yet. Show the user the breakdown you parsed ("Here's what I got: type=future,
    region=APAC, ... — anything to fix?"), let them correct it, then call again with
    confirm=True (same or corrected fields) to actually save. Never go straight to
    confirm=True from a free-text answer without showing the breakdown first.

    From Claude Web, pass deck_text (the text you read from the attached PDF/PPTX) plus any
    freeform detail you have (brief, audience, budget, channel, timeline). The server chunks
    and embeds it per slide/section for search.

    Passing deck_text alone does NOT check images — you also need asset_ref (a reference to
    the actual file: POST /upload first to get one, or a local path in stdio mode). Prefer
    passing asset_ref whenever you have the file, alongside deck_text if you already read it
    (deck_text you pass is kept as-is, not overwritten by server-side extraction) — this is
    the only way to get automatic creative-reuse detection, and it's what most users actually
    want when they attach a deck.

    When asset_ref resolves to a real file, the server ALSO extracts every image embedded in
    the deck automatically, fingerprints and visually embeds each one, and checks it against
    every other campaign's images for reuse — no separate upload_image_asset call needed per
    image (nobody would actually do that for every slide). Check the response's
    images_checked field first: True means image reuse was actually checked (image_assets may
    still be empty if the deck simply had no images); False means it was NOT checked at all
    (no file reached the server, or extraction itself failed) — do not tell the user "no
    reuse found" when images_checked is False. When True, each image_assets entry's
    reuse_flags is a list of prior campaigns whose image matched this one (by content, not by
    look) — a NON-EMPTY list means that image was reused, period, even if its `flag` field is
    null (null `flag` = reused within the same region, not itself suspicious; a `flag` string
    = reused across a different region, the signal worth calling out). Mention every non-empty
    reuse_flags entry to the user, not just ones with a `flag` string.

    record_type is 'campaign' (default), 'reference' (background material, not itself a
    campaign), or 'stub' (a placeholder record). status is 'proposed', 'in_flight', or
    'concluded' — defaults to 'concluded' for record_type='campaign', otherwise unset.
    tags is a list of strings or {"value","source"} objects (source: 'verified' if backed by
    real data — this requires the campaign to already have a metric_type='actual' record on
    file, so it can only be set via update_campaign after add_metrics, never here on a
    brand-new campaign — or 'stated' if it's someone's claim; defaults to 'stated', the
    conservative assumption. Not restricted to the creative-reaction/performance vocabulary
    above, but that's the vocabulary the "missing quadrant" analysis needs. region/market are freeform too (e.g.
    region='APAC', market='Philippines'). collection links market/version variants of the
    SAME creative — a symmetric grouping (e.g. all regional launches of one collection share
    a collection value), unlike supersedes below (asymmetric replacement). These structured
    fields let find_similar_campaigns / prepare_evaluation filter before ranking by
    similarity.

    Pass supersedes=<campaign_id> if this record replaces an existing one (e.g. a corrected
    deck) — the old record is then excluded from future search evidence, so it stops
    confusing retrieval, without being deleted.

    Add results later with add_metrics. On confirm=True, returns the campaign_id plus
    chunks_total/chunks_embedded (partial embedding failures are reported per-chunk in
    warnings, not silently), and images_checked/image_assets (see above)."""
    conn = store.connect()
    try:
        return core.ingest_campaign(conn, title=title, detail=detail, deck_text=deck_text,
                                    record_type=record_type, status=status, tags=tags,
                                    region=region, market=market, collection=collection,
                                    supersedes=supersedes, asset_ref=asset_ref, confirm=confirm)
    finally:
        conn.close()


@mcp.tool()
@_catch_value_errors
def update_campaign(campaign_id: str, title: Optional[str] = None, detail: Optional[str] = None,
                    record_type: Optional[RecordType] = None, status: Optional[Status] = None,
                    tags: Optional[list[TagInput]] = None, region: Optional[str] = None,
                    market: Optional[str] = None, collection: Optional[str] = None) -> dict:
    """Edit a campaign's metadata (title, detail, record_type, status, tags, region, market,
    collection). Only the fields you pass change. tags, if given, fully REPLACES the
    existing list (not a merge) — pass the complete new list, including any you're keeping.
    This is also how you upgrade a tag's provenance once real data comes in — e.g. re-save
    tags with {"value": "performed_well", "source": "verified"} instead of the plain string
    once add_metrics has real numbers on file, so it stops reading as an unverified
    impression. Does NOT change deck_text/chunks/embeddings; for content changes, upload a
    new record and pass supersedes=campaign_id instead."""
    conn = store.connect()
    try:
        ok = store.update_campaign(conn, campaign_id, title=title, detail=detail,
                                   record_type=record_type, status=status, tags=tags,
                                   region=region, market=market, collection=collection)
        if not ok:
            return {"error": f"campaign {campaign_id} not found"}
        return store.get_campaign(conn, campaign_id)
    finally:
        conn.close()


@mcp.tool()
@_catch_value_errors
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
@_catch_value_errors
def upload_image_asset(campaign_id: str, asset_ref: dict) -> dict:
    """Attach an image (hero shot, creative asset) to a campaign. Processed two ways: a
    perceptual hash (exact/near-duplicate reuse — check_image_provenance) and a CLIP visual
    embedding (aesthetic/regional similarity — find_similar_images). asset_ref is {asset_id}
    from POST /upload, {path} local, or {filename, base64} inline. Consider calling
    check_image_provenance and/or find_similar_images first if you want to flag reuse or
    similarity before attaching it."""
    conn = store.connect()
    try:
        return core.ingest_image_asset(conn, campaign_id=campaign_id, asset_ref=asset_ref)
    finally:
        conn.close()


@mcp.tool()
@_catch_value_errors
def check_image_provenance(asset_ref: dict, campaign_id: Optional[str] = None) -> dict:
    """Check whether an image matches one already in the memory — same/near-same photo,
    even after resize/recompress/light crop (perceptual hashing; catches exact reuse, NOT
    aesthetic similarity — use find_similar_images for that). Works before the image is
    stored. Pass campaign_id (the campaign this image is headed for) to exclude that
    campaign's own assets and get a flag when a match comes from a *different* region — the
    real question is usually not "does this image exist" but "does this image belong to a
    different region than where it's being used.\""""
    conn = store.connect()
    try:
        return core.check_image_provenance(conn, asset_ref=asset_ref, campaign_id=campaign_id)
    finally:
        conn.close()


@mcp.tool()
@_catch_value_errors
def find_similar_images(asset_ref: dict, campaign_id: Optional[str] = None, top_k: int = 5,
                        region: Optional[str] = None) -> dict:
    """Aesthetic/regional visual similarity via CLIP — catches "same product, different
    photo," "looks like the APAC shoot" — NOT exact reuse (use check_image_provenance for
    that). Works before the image is stored. Pass region to weigh only that region's assets
    first (mirrors find_similar_campaigns' filter-before-rank pattern); pass campaign_id (the
    campaign this image is headed for) to exclude its own assets and flag matches from a
    different region."""
    conn = store.connect()
    try:
        return core.find_similar_images(conn, asset_ref=asset_ref, campaign_id=campaign_id,
                                        top_k=top_k, region=region)
    finally:
        conn.close()


@mcp.tool()
@_catch_value_errors
def add_metrics(campaign_id: str, detail: Optional[str] = None,
                structured: Optional[dict] = None, metric_type: MetricType = "actual",
                confirm: bool = False) -> dict:
    """Record feedback/outcomes for a campaign — this is the feedback conversation, not a
    form. Ask: *which campaign* (look it up with find_similar_campaigns/list_campaigns if the
    user doesn't give an exact id/title — disambiguate rather than guessing), *how did it
    go*, *how was the response*, *what metrics do you have* — impressions, likes/engagement,
    footfall, sales, whatever they tracked. If they give you one free-text paragraph, break
    it down into detail/structured yourself instead of asking again field-by-field.

    Call with confirm=False (the default) first — this PREVIEWS the breakdown without
    storing anything. Show it to the user ("Here's what I got: ... — anything to add or
    fix?"), let them fine-tune it, then call again with confirm=True to actually save.

    detail is freeform (CTR, ROI, conversions, qualitative learnings, or just what the user
    said); structured is an optional machine-readable object for numbers you extracted.
    metric_type is 'actual' (post-conclusion results, the default) or 'predicted' (a
    forecast/target set before launch) — reconcile_evaluation only pulls 'actual' metrics
    automatically."""
    conn = store.connect()
    try:
        return core.add_metrics(conn, campaign_id, detail=detail, structured=structured,
                                metric_type=metric_type, confirm=confirm)
    finally:
        conn.close()


@mcp.tool()
@_catch_value_errors
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
@_catch_value_errors
def list_campaigns(record_type: Optional[RecordType] = None, status: Optional[Status] = None) -> dict:
    """List records in the memory. Optionally filter by record_type ('campaign', 'reference',
    'stub') and/or status ('proposed', 'in_flight', 'concluded'). is_superseded/supersedes
    show whether a record has been replaced by a corrected/later one (and by what) — check
    these before treating two similarly-titled records as both live."""
    conn = store.connect()
    try:
        rows = store.list_campaigns(conn, record_type=record_type, status=status)
        return {"count": len(rows), "campaigns": [
            {"campaign_id": r["id"], "title": r["title"], "record_type": r["record_type"],
             "status": r["status"], "tags": r["tags"], "region": r["region"],
             "market": r["market"], "collection": r["collection"], "embedded": r["embedded"],
             "has_metrics": r["has_metrics"], "has_evaluations": r["has_evaluations"],
             "supersedes": r["supersedes"], "is_superseded": r["is_superseded"]}
            for r in rows]}
    finally:
        conn.close()


@mcp.tool()
@_catch_value_errors
def get_campaign(campaign_id: str) -> dict:
    """Full detail + all metrics for one campaign by id."""
    conn = store.connect()
    try:
        c = store.get_campaign(conn, campaign_id)
        return c or {"error": f"campaign {campaign_id} not found"}
    finally:
        conn.close()


@mcp.tool()
@_catch_value_errors
def find_similar_campaigns(text: Optional[str] = None, campaign_id: Optional[str] = None,
                           top_k: int = 5, record_type: Optional[RecordType] = None,
                           status: Optional[Status] = None, tags: Optional[list[TagInput]] = None,
                           match_all_tags: bool = False, region: Optional[str] = None,
                           market: Optional[str] = None, collection: Optional[str] = None,
                           full_detail: bool = False) -> dict:
    """Semantic search: find prior campaigns most similar to a description (text) or to an
    existing campaign (campaign_id). Matches at the slide/section level and rolls up to the
    best-matching campaign, so long decks match on the relevant part.

    Pass record_type/status/tags/region/market/collection to filter to that criteria FIRST,
    then rank by similarity within it — e.g. status='concluded', region='APAC' to only weigh
    concluded APAC precedent instead of everything in the memory.

    tags is a list of plain strings (match that value, any source) and/or {value, source}
    objects (match that value AND require that specific source) — mix freely. Defaults to
    ANY-match. For a quadrant query like "which campaigns were liked but VERIFIED
    underperformed," pass tags=["liked", {"value": "underperformed", "source": "verified"}]
    with match_all_tags=True (otherwise you'd get anything matching EITHER tag, not the
    co-occurrence). A creative-reaction tag like "liked" can never itself be "verified" the
    way a performance tag can — that's why source is per-tag, not one global flag: it lets
    you require verification on just the performance tag while leaving the reaction tag
    open to any source.

    Returns ranked evidence — title, status/tags/region/market/collection, similarity,
    detail, the matched excerpt, and metrics (each tag shows its value AND source) — for you
    to reason over. detail and metrics are trimmed by default (detail_truncated/
    metrics_truncated flag it) — pass full_detail=True, or call get_campaign, for the
    untrimmed record."""
    conn = store.connect()
    try:
        return {"matches": core.find_similar(conn, text=text, campaign_id=campaign_id,
                                             top_k=top_k, record_type=record_type,
                                             status=status, tags=tags,
                                             match_all_tags=match_all_tags, region=region,
                                             market=market, collection=collection,
                                             full_detail=full_detail)}
    finally:
        conn.close()


@mcp.tool()
@_catch_value_errors
def prepare_evaluation(subject_title: str, proposal_text: str, top_k: int = 5,
                       record_type: Optional[RecordType] = None, status: Optional[Status] = None,
                       tags: Optional[list[TagInput]] = None, match_all_tags: bool = False,
                       region: Optional[str] = None, market: Optional[str] = None,
                       collection: Optional[str] = None, full_detail: bool = True) -> dict:
    """Evaluate a NEW campaign proposal against the memory. Returns the most similar prior
    campaigns WITH their outcomes as an evidence package (full detail by default — this is
    for judging, not browsing). Optionally narrow to structured criteria first (e.g.
    region='APAC') so only relevant precedent is weighed.

    Pass a tag as {"value": "underperformed", "source": "verified"} to weigh only precedent
    whose matching performance claim is backed by real metric data, not someone's stated
    impression — a performance claim with no measurement behind it should carry less
    weight in your judgment than one with real numbers. Read the evidence's tags for each
    match's source either way before treating a performance tag as fact.

    Read it, then produce your judgment (predicted CTR/ROI ranges, risks,
    proceed/revise/reject) CITING specific campaign_ids, and call save_evaluation. This tool
    gathers evidence; the judgment is yours."""
    conn = store.connect()
    try:
        return core.prepare_evaluation(conn, subject_title=subject_title,
                                       proposal_text=proposal_text, top_k=top_k,
                                       record_type=record_type, status=status, tags=tags,
                                       match_all_tags=match_all_tags, region=region,
                                       market=market, collection=collection,
                                       full_detail=full_detail)
    finally:
        conn.close()


@mcp.tool()
@_catch_value_errors
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
@_catch_value_errors
def list_evaluations() -> dict:
    """List past evaluations (id, campaign_id, subject_title, created_at) — use this to find
    an evaluation_id when the user refers to a judgment by name rather than id (e.g.
    "reconcile the APAC campaign evaluation") before calling reconcile_evaluation."""
    conn = store.connect()
    try:
        rows = store.list_evaluations(conn)
        return {"count": len(rows), "evaluations": rows}
    finally:
        conn.close()


@mcp.tool()
@_catch_value_errors
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
@_catch_value_errors
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
