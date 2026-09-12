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

# The evaluation vocabulary, in the schema rather than only in prose — a typo becomes a
# validation error the model can retry, instead of a value that quietly means nothing.
Verdict = Literal["approve", "revise", "reject"]
Severity = Literal["blocking", "should_fix", "note"]
# Severity says how much a finding matters; KIND says whether it is arguable at all. A
# guardrail breach cites the rulebook and is not open to debate; a departure from precedent
# cites a campaign and invites a rationale — one real departure turned out better than the
# precedent it departed from.
FindingKind = Literal["guardrail_breach", "precedent_departure", "missing_information",
                      "internal_contradiction"]
# Whether the server worked this out or the model judged it (§7.8). Only "judged" is
# writable here: the premise that a computed finding is identical for every user, so a
# difference is a bug, holds only if the SERVER computed it. §7.1 stamps the other one.
Basis = Literal["judged"]


class Precedent(TypedDict):
    """What a finding is anchored to: a campaign that did it differently, or a rule it
    breached. `quote` is text from the retrieved chunk, not written fresh — a finding that
    cannot quote its source is a judgment call, not a citation."""
    quote: NotRequired[str]           # <= 300 chars; longer than that is not a quote
    campaign_id: NotRequired[str]     # for a departure from precedent
    rule_id: NotRequired[str]         # for a guardrail breach — a rule, not a campaign
    # Which layer the quote came from. Default "body" = the deck itself. Set "commentary"
    # whenever the excerpt you are quoting arrived with matched_kind "commentary", and name
    # who said it — otherwise the finding records somebody's objection as a claim the deck
    # made.
    layer: NotRequired[Literal["body", "commentary"]]
    author: NotRequired[str]          # commentary only: who said it
    anchor: NotRequired[str]          # commentary only: "slide 4", "page 2", "deck"
    date: NotRequired[str]            # commentary only: when they said it


class Finding(TypedDict):
    """One problem. The short fields are capped server-side so they cannot grow back into
    the paragraph this shape exists to replace."""
    severity: Severity
    finding: str                      # <= 120 chars, names the problem
    kind: NotRequired[FindingKind]    # is it a rule broken, or a precedent departed from?
    basis: NotRequired[Basis]         # computed by the server, or judged (default: judged)
    category: NotRequired[str]        # timeline | influencer | compliance | budget | ...
    detail: NotRequired[str]          # the paragraph, read on demand
    precedent: NotRequired[Precedent]
    fix: NotRequired[str]             # <= 120 chars, what to change


@mcp.tool()
@_catch_value_errors
def upload_campaign(title: str, detail: Optional[str] = None, deck_text: Optional[str] = None,
                    record_type: RecordType = "campaign", status: Optional[Status] = None,
                    tags: Optional[list[TagInput]] = None, region: Optional[str] = None,
                    market: Optional[str] = None, markets: Optional[list[str]] = None,
                    collection: Optional[str] = None,
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

    Whether a campaign genuinely belongs to a collection is a judgment call for the
    marketer, not you — "features the same product/collection" and "IS one of that
    collection's own launches" are different things (e.g. a store launch that promotes a
    collection's products is not itself a collection launch), and guessing wrong either way
    quietly corrupts the collection_siblings set every later query relies on. If it's not
    obvious which one this is from what they've told you, ask directly ("is this itself
    part of the Khloe Q2 2026 launch, or just related to it?") rather than picking one — do
    not leave this to be silently decided outside the conversation.

    Then call this tool with confirm=False (the default) to get a PREVIEW — nothing is
    stored yet. Show the user the breakdown you parsed ("Here's what I got: type=future,
    region=APAC, ... — anything to fix?"), let them correct it, then call again with
    confirm=True (same or corrected fields) to actually save. Never go straight to
    confirm=True from a free-text answer without showing the breakdown first.

    From Claude Web, pass deck_text (the text you read from the attached PDF/PPTX) plus any
    freeform detail you have (brief, audience, budget, channel, timeline). The server chunks
    and embeds it per slide/section for search.

    Pass asset_ref (the file itself) whenever you have it, even alongside deck_text: comments,
    annotations and speaker notes can only be read from the file, and a partner deck returned
    with tracked client comments is the feedback this library most wants to remember.
    `commentary_found` says how many were FOUND (searchable once chunks_embedded reaches
    chunks_total), and `commentary_checked` says whether anything was read at all. When it is
    false — a deck_text-only upload, an unsupported file, or a comments part that would not
    parse — do NOT tell the user the deck has no comments; nobody opened a file. Say that
    comments need the file itself and offer to re-upload with it.

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
    above, but that's the vocabulary the "missing quadrant" analysis needs. region/market are
    freeform, SINGLE-value fields (e.g. region='Malaysia', market='SEA' as the grouping
    label) — exact-match on filter, so pick region as the one country/market this record is
    primarily about. If the activation actually ran in more than one country (e.g. a SEA
    campaign covering Malaysia, Singapore, and Indonesia), ALSO pass markets=["Malaysia",
    "Singapore", "Indonesia"] — a list, matched by membership, not exact-match — otherwise
    that campaign is invisible to a query for any country besides the one in region (a real
    gap found in testing: tagging region alone made an Indonesia-inclusive campaign
    unreachable by region='Indonesia', which reads as "no such campaign" rather than "field
    can't represent this"). collection links market/version variants of the SAME creative —
    a symmetric grouping (e.g. all regional launches of one collection share a collection
    value), unlike supersedes below (asymmetric replacement). These structured fields let
    find_similar_campaigns / prepare_evaluation filter before ranking by similarity.

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
                                    region=region, market=market, markets=markets,
                                    collection=collection, supersedes=supersedes,
                                    asset_ref=asset_ref, confirm=confirm)
    finally:
        conn.close()


@mcp.tool()
@_catch_value_errors
def update_campaign(campaign_id: str, title: Optional[str] = None, detail: Optional[str] = None,
                    record_type: Optional[RecordType] = None, status: Optional[Status] = None,
                    tags: Optional[list[TagInput]] = None, region: Optional[str] = None,
                    market: Optional[str] = None, markets: Optional[list[str]] = None,
                    collection: Optional[str] = None) -> dict:
    """Edit a campaign's metadata (title, detail, record_type, status, tags, region, market,
    markets, collection). Only the fields you pass change. tags/markets, if given, fully
    REPLACE the existing list (not a merge) — pass the complete new list, including any
    you're keeping. This is also how you upgrade a tag's provenance once real data comes in
    — e.g. re-save tags with {"value": "performed_well", "source": "verified"} instead of
    the plain string once add_metrics has real numbers on file, so it stops reading as an
    unverified impression. It's also how you fix a campaign tagged with only a single
    region when its activation actually spanned more (a real gap found in testing) — pass
    markets=["Malaysia", "Indonesia", ...] to make it reachable by any of those countries,
    without touching region/market. Setting/changing collection is a judgment call for the
    marketer, not you — see upload_campaign's docstring on asking rather than guessing when
    it's unclear whether this record IS a collection launch versus merely related to one.
    Does NOT change deck_text/chunks/embeddings; for content changes, upload a new record
    and pass supersedes=campaign_id instead."""
    conn = store.connect()
    try:
        ok = store.update_campaign(conn, campaign_id, title=title, detail=detail,
                                   record_type=record_type, status=status, tags=tags,
                                   region=region, market=market, markets=markets,
                                   collection=collection)
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
def health_check() -> dict:
    """Check whether this server's parts are actually working, and whether the library is
    fully searchable. Answers in about a second.

    Call it when something seems wrong — a search returning less than expected, an upload
    warning, an image tool failing — and before any demo or important session, rather than
    inferring health from a tool call that times out.

    Each component reports separately because they fail independently: visual search being
    unavailable does not stop text search, uploads or evaluations, and saying "the server is
    down" when only half is would be wrong. Translate for the user — "visual similarity is
    unavailable because the image model is missing; everything else works" beats relaying
    component names — and pass on the `remedy` verbatim enough that an admin can act on it.

    `coverage` answers the different question of whether what they uploaded is usable: a
    non-zero `sections_unindexed`/`images_unindexed` means searches will be incomplete until
    finish_indexing is run."""
    conn = store.connect()
    try:
        return core.health_check(conn)
    finally:
        conn.close()


@mcp.tool()
@_catch_value_errors
def finish_indexing(campaign_id: Optional[str] = None) -> dict:
    """Finish records that are stored but not yet searchable, without re-uploading anything.

    Use this when an upload reported it ran out of time, when list_campaigns shows
    chunks_embedded below chunks_total (or assets_embedded below assets_total), or after the
    embedder was down while uploads went in. Pass a campaign_id for one record, or nothing
    to work through everything outstanding.

    Bounded by the same time budget as any other call, so a large backlog takes several
    passes. **If `complete` is false and `indexed` was above zero, just call it again
    straight away** — do not stop to ask each time; the user wants the job done, not a
    progress meeting. Report once at the end. Only stop and ask if there is a lot left
    (`remaining` in the hundreds) or the user is waiting on something else.

    **If `indexed` is zero, do NOT call it again** — nothing was achieved and nothing will
    be until the cause in `errors` is fixed. Tell the user what is broken instead.
    `failed` counts items that can never be indexed (their file is gone); those are skipped
    rather than retried forever, which is why `complete` can be true with failures present.
    `outstanding` names the records still waiting, so you can say "your Mexico deck is done,
    two older records still have 40 sections to go" rather than reciting numbers."""
    conn = store.connect()
    try:
        return core.finish_indexing(conn, campaign_id=campaign_id)
    finally:
        conn.close()


@mcp.tool()
@_catch_value_errors
def list_campaigns(record_type: Optional[RecordType] = None, status: Optional[Status] = None) -> dict:
    """List records in the memory. Optionally filter by record_type ('campaign', 'reference',
    'stub') and/or status ('proposed', 'in_flight', 'concluded'). is_superseded/supersedes
    show whether a record has been replaced by a corrected/later one (and by what) — check
    these before treating two similarly-titled records as both live.

    `embedded` means FULLY searchable. When it is false, chunks_embedded/chunks_total (and
    assets_embedded/assets_total) say how much of the record search can actually find —
    "stored" and "searchable" are different states, and an upload that ran out of time sits
    between them. Anything short of complete can be finished with finish_indexing, without
    the user re-uploading anything; say so rather than leaving them to wonder why a deck they
    uploaded isn't coming back in results."""
    conn = store.connect()
    try:
        rows = store.list_campaigns(conn, record_type=record_type, status=status)
        return {"count": len(rows), "campaigns": [
            {"campaign_id": r["id"], "title": r["title"], "record_type": r["record_type"],
             "status": r["status"], "tags": r["tags"], "region": r["region"],
             "market": r["market"], "collection": r["collection"], "embedded": r["embedded"],
             "chunks_total": r["chunks_total"], "chunks_embedded": r["chunks_embedded"],
             "assets_total": r["assets_total"], "assets_embedded": r["assets_embedded"],
             "has_metrics": r["has_metrics"], "has_evaluations": r["has_evaluations"],
             "supersedes": r["supersedes"], "is_superseded": r["is_superseded"]}
            for r in rows]}
    finally:
        conn.close()


@mcp.tool()
@_catch_value_errors
def get_campaign(campaign_id: str) -> dict:
    """Full detail + all metrics for one campaign by id, plus its `commentary`: the speaker
    notes, annotations and tracked reviewer comments its deck carried, with the author, date
    and position wherever the file recorded them — a speaker note carries no author at all,
    and an annotation often has none, so a missing author means the file did not say, not
    that nobody said it. That layer is what people said ABOUT the work and is kept separate
    from the deck body deliberately — do not read it back as the brief's own content."""
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
                           status: Optional[Status] = None,
                           tags: Optional[Union[TagInput, list[TagInput]]] = None,
                           match_all_tags: bool = False, region: Optional[str] = None,
                           market: Optional[str] = None, markets: Optional[Union[str, list[str]]] = None,
                           collection: Optional[str] = None,
                           full_detail: bool = False,
                           include_commentary: Union[bool, list[str]] = True) -> dict:
    """Semantic search: find prior campaigns most similar to a description (text) or to an
    existing campaign (campaign_id). Matches at the slide/section level and rolls up to the
    best-matching campaign, so long decks match on the relevant part.

    Pass record_type/status/tags/region/market/collection to filter to that criteria FIRST,
    then rank by similarity within it — e.g. status='concluded', region='APAC' to only weigh
    concluded APAC precedent instead of everything in the memory. region/market are exact-
    match on a SINGLE value, which can't find a campaign whose activation spanned several
    countries if it's only tagged with one of them as region — pass markets='Indonesia' (or
    markets=["Indonesia","Thailand"] for ANY-match across several) instead to match any
    campaign whose stored markets list includes at least one, regardless of what its
    region/market fields say.

    tags is a list of plain strings (match that value, any source) and/or {value, source}
    objects (match that value AND require that specific source) — mix freely. Defaults to
    ANY-match. For a quadrant query like "which campaigns were liked but VERIFIED
    underperformed," pass tags=["liked", {"value": "underperformed", "source": "verified"}]
    with match_all_tags=True (otherwise you'd get anything matching EITHER tag, not the
    co-occurrence). A creative-reaction tag like "liked" can never itself be "verified" the
    way a performance tag can — that's why source is per-tag, not one global flag: it lets
    you require verification on just the performance tag while leaving the reaction tag
    open to any source.

    Decks are indexed in two layers. The BODY is what the deck says; COMMENTARY is what
    people said about it — speaker notes, PDF annotations and tracked reviewer comments,
    carrying author, date and position wherever the file recorded them (speaker notes carry
    no author; a null one means the file did not say, not that nobody said it). Both are
    searched by default, and `matched_kind` on every hit says which one matched: a
    `commentary` hit is somebody's opinion of the work, not a claim the brief made, and
    citing it as the latter attributes a reviewer's objection to the deck.
    include_commentary=False answers "what does the brief say"; a list of kinds narrows to
    who was speaking — ["comment"] is what reviewers left on the deck, as opposed to
    ["speaker_note"], which the deck's own author wrote to themselves. Note that a PDF
    export turns speaker notes into annotations, so the kind records the format the words
    arrived in, not how much authority they carry.

    Returns ranked evidence — title, status/tags/region/market/collection, similarity,
    detail, the matched excerpt with its matched_kind (and matched_author/matched_anchor/
    matched_date when commentary matched), and metrics (each tag shows its value AND source)
    — for you to reason over. detail and metrics are trimmed by default (detail_truncated/
    metrics_truncated flag it) — pass full_detail=True, or call get_campaign, for the
    untrimmed record."""
    conn = store.connect()
    try:
        return core.find_similar_with_context(conn, text=text, campaign_id=campaign_id,
                                             top_k=top_k, record_type=record_type,
                                             status=status, tags=tags,
                                             match_all_tags=match_all_tags, region=region,
                                             market=market, markets=markets,
                                             collection=collection,
                                             full_detail=full_detail,
                                             include_commentary=include_commentary)
    finally:
        conn.close()


@mcp.tool()
@_catch_value_errors
def prepare_evaluation(subject_title: str, proposal_text: str, top_k: int = 5,
                       record_type: Optional[RecordType] = None, status: Optional[Status] = None,
                       tags: Optional[Union[TagInput, list[TagInput]]] = None,
                       match_all_tags: bool = False,
                       region: Optional[str] = None, market: Optional[str] = None,
                       markets: Optional[Union[str, list[str]]] = None, collection: Optional[str] = None,
                       full_detail: bool = True) -> dict:
    """Evaluate a NEW campaign proposal against the memory. Returns the most similar prior
    campaigns WITH their outcomes as an evidence package (full detail by default — this is
    for judging, not browsing). Optionally narrow to structured criteria first (e.g.
    region='APAC') so only relevant precedent is weighed. region/market are single-value
    exact-match; pass markets='Indonesia' (or a list for ANY-match) instead to reach a
    campaign whose multi-country activation only has ONE of those countries in region/market.

    Pass a tag as {"value": "underperformed", "source": "verified"} to weigh only precedent
    whose matching performance claim is backed by real metric data, not someone's stated
    impression — a performance claim with no measurement behind it should carry less
    weight in your judgment than one with real numbers. Read the evidence's tags for each
    match's source either way before treating a performance tag as fact.

    Evidence rows come from two layers and say which in `matched_kind`: `body` is what a
    deck says, `commentary` is what somebody said about it (speaker notes, annotations,
    tracked client comments, with `matched_author` and `matched_anchor`). Weigh both — a
    client's recorded objection is often the most useful precedent in the library — but
    never blur them: quoting a commentary row as though the deck itself claimed it is a
    false statement about that campaign.

    Read it, then call save_evaluation with a verdict (approve / revise / reject), a
    one-line summary and one short finding per problem, CITING specific campaign_ids. When a
    finding quotes a `commentary` row, set that precedent's `layer: "commentary"` and carry
    its author and anchor across. Predicted CTR/ROI ranges go in `predictions`. This tool
    gathers evidence; the judgment is yours."""
    conn = store.connect()
    try:
        return core.prepare_evaluation(conn, subject_title=subject_title,
                                       proposal_text=proposal_text, top_k=top_k,
                                       record_type=record_type, status=status, tags=tags,
                                       match_all_tags=match_all_tags, region=region,
                                       market=market, markets=markets, collection=collection,
                                       full_detail=full_detail)
    finally:
        conn.close()


@mcp.tool()
@_catch_value_errors
def save_evaluation(subject_title: str, verdict: Verdict, summary: str,
                    findings: Optional[list[Finding]] = None,
                    resolved: Optional[list[dict]] = None,
                    closest_precedent: Optional[dict] = None,
                    approve_if: Optional[str] = None,
                    cited_ids: Optional[list] = None,
                    predictions: Optional[dict] = None,
                    campaign_id: Optional[str] = None) -> dict:
    """Persist your judgment as structured findings, not prose.

    Write ONE finding per problem. Each is a short line naming the problem (<=120 chars),
    with the explanation in `detail` where a reader can open it if they want it — not a
    paragraph in `finding`. `summary` is <=240 chars: the one line a marketer acts on.

    `severity` says what the finding does to the verdict, so counts mean the same thing in
    every evaluation:
      • `blocking`   — this alone means the brief cannot proceed as written.
      • `should_fix` — proceeding is defensible, but it will cost something.
      • `note`       — worth saying once; nobody has to act.
    A `revise` or `reject` needs at least one finding above a note, and you cannot `approve`
    while recording a blocking one. Both are rejected rather than saved — and the honest fix
    for the second is the verdict, never a quieter severity.

    `kind` says whether the finding is arguable at all, which severity cannot express:
      • `guardrail_breach`      — a rule was broken. Cite it: `precedent: {rule_id, quote}`.
        Not debatable, so never a `note`.
      • `precedent_departure`   — done differently from a campaign that worked. Cite it:
        `precedent: {campaign_id, quote}`. A departure can be an improvement; say so.
      • `missing_information`   — the brief does not say.
      • `internal_contradiction`— the brief contradicts itself.

    Anchor findings to evidence: `quote` is text from the retrieved chunk, not written
    fresh. A finding that cannot quote its source is a judgment call and reads as one.

    Check which LAYER your quote came from. Evidence rows carry `matched_kind`: `body` is
    what the deck says, `commentary` is what somebody said ABOUT it — a speaker note, a PDF
    annotation, or a tracked comment a client left on a returned deck. Quoting commentary is
    legitimate and often the best evidence there is, but you must mark it: set
    `precedent.layer: "commentary"` with the `author` and `anchor` from that row. Left
    unmarked it is stored as something the campaign's own deck claimed, which is a different
    and false statement.

    A worked finding:
      {"severity": "blocking", "kind": "missing_information", "category": "timeline",
       "finding": "No posting dates on any deliverable",
       "detail": "All 14 assets in the flighting table are undated, so nothing can be
                  sequenced or held to the embargo.",
       "precedent": {"campaign_id": "camp_jdsea",
                     "quote": "content angle, posting date and requirements per asset"},
       "fix": "Add a posting date per asset to the flighting table"}

    `approve_if` is what would flip a `revise` to `approve`, stated so someone could check
    it: "dates on every deliverable and the two conflicted profiles removed". It doubles as
    the note the partner receives.

    `resolved` is for a later version of a brief: `[{was, now}]` records what an earlier
    evaluation asked for and what changed — that is how the library learns whether its own
    advice was taken.

    At most 12 findings. The response gives back the verdict, the summary, the counts, and
    the blocking and should_fix lines themselves — give the user those. The reasoning behind
    any finding is in `get_evaluation`, which also filters by severity or kind."""
    conn = store.connect()
    try:
        return core.save_evaluation(
            conn, subject_title=subject_title, verdict=verdict, summary=summary,
            findings=findings, resolved=resolved, closest_precedent=closest_precedent,
            approve_if=approve_if, campaign_id=campaign_id, cited_ids=cited_ids,
            predictions=predictions)
    finally:
        conn.close()


@mcp.tool()
@_catch_value_errors
def get_evaluation(evaluation_id: str, severity: Optional[Severity] = None,
                   kind: Optional[FindingKind] = None) -> dict:
    """Read a stored judgment back in full, with the reasoning `save_evaluation` left out.

    This is where "show me the blocking items" is answered — including in a session that did
    not produce the evaluation. Narrow with `severity` ("blocking") or `kind`
    ("guardrail_breach") rather than fetching everything and filtering in the reply.

    Find the id with list_evaluations if the user names the judgment rather than its id."""
    conn = store.connect()
    try:
        return core.get_evaluation(conn, evaluation_id=evaluation_id, severity=severity,
                                   kind=kind)
    finally:
        conn.close()


@mcp.tool()
@_catch_value_errors
def list_evaluations() -> dict:
    """List past evaluations — id, campaign_id, subject_title, verdict, counts by severity,
    created_at. The verdict and counts are there so "which of these still need work" can be
    answered from the list itself. Use it to find an evaluation_id when the user refers to a
    judgment by name rather than id (e.g. "reconcile the APAC campaign evaluation"), then
    call get_evaluation for the findings or reconcile_evaluation to close the loop."""
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
    otherwise pass actual= with the real post-campaign metrics yourself. Returns the
    original verdict, summary and findings (or, for a judgment written before the structured
    schema, `original_analysis` — the free text as it was written) alongside the actuals.
    Compare them, then call save_reconciliation with the lesson."""
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
