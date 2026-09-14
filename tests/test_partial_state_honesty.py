"""
Follow-through on defect 05 ("failed embeddings leave unrepairable partial records") after
item 2.1 made partial embedding a DESIGNED outcome rather than an accident.

The review's original complaint was that a half-embedded campaign was "silently invisible to
every search while still appearing in list_campaigns". Item 2.1 introduced a time budget that
stops mid-deck on purpose — which means the `embedded` boolean now lies by design: two of
twelve chunks embedded still reported `embedded: True`, in the same response whose warning
said "2 of 12 sections are searchable so far".

A boolean cannot express "partly". These tests pin the counts instead.
"""
import time

import pytest

import config
import core
import embedding
import store


def _slow_embed(seconds=0.12):
    def stub(text, timeout=None):
        time.sleep(seconds)
        return [0.0] * config.EMBED_DIM
    return stub


def test_a_partly_embedded_campaign_does_not_claim_to_be_embedded(conn, monkeypatch):
    monkeypatch.setattr(config, "TOOL_TIME_BUDGET_SECONDS", 0.2)
    monkeypatch.setattr(embedding, "embed", _slow_embed())

    deck = "\n\n".join(f"section {i} " + "word " * 200 for i in range(12))
    result = core.ingest_campaign(conn, title="Cut short", deck_text=deck, confirm=True)

    assert 0 < result["chunks_embedded"] < result["chunks_total"]
    assert result["embedded"] is False, (
        "'embedded' must mean searchable, not 'we managed at least one' - the boolean is "
        "what list_campaigns shows, and claiming True here is the stored-vs-searchable "
        "confusion defect 05 opened with"
    )


def test_a_fully_embedded_campaign_still_reports_embedded(conn):
    result = core.ingest_campaign(conn, title="Complete", deck_text="a short brief",
                                  confirm=True)

    assert result["chunks_embedded"] == result["chunks_total"]
    assert result["embedded"] is True


def test_list_campaigns_shows_how_much_is_searchable(conn, monkeypatch):
    """"Stored" and "searchable" must never be conflated - the review asked for this
    explicitly, as a count rather than a flag."""
    monkeypatch.setattr(config, "TOOL_TIME_BUDGET_SECONDS", 0.2)
    monkeypatch.setattr(embedding, "embed", _slow_embed())

    deck = "\n\n".join(f"section {i} " + "word " * 200 for i in range(12))
    created = core.ingest_campaign(conn, title="Cut short", deck_text=deck, confirm=True)

    row = next(r for r in store.list_campaigns(conn) if r["id"] == created["campaign_id"])

    assert row["chunks_total"] == created["chunks_total"]
    assert row["chunks_embedded"] == created["chunks_embedded"]
    assert row["chunks_embedded"] < row["chunks_total"]


def test_list_campaigns_counts_unembedded_images_too(conn, tmp_path, monkeypatch):
    import clip_embed
    import numpy as np
    from PIL import Image
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    for i in range(3):
        img = tmp_path / f"i{i}.png"
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (8, 8, 3), dtype="uint8"),
                        mode="RGB").resize((120, 120), Image.BICUBIC).save(img)
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.shapes.add_picture(str(img), Inches(1), Inches(1), width=Inches(2))
    deck = tmp_path / "deck.pptx"
    prs.save(str(deck))

    monkeypatch.setattr(config, "TOOL_TIME_BUDGET_SECONDS", 0.0)
    monkeypatch.setattr(clip_embed, "embed_image", lambda p: pytest.fail("no budget"))

    created = core.ingest_campaign(conn, title="Images", asset_ref={"path": str(deck)},
                                   confirm=True)
    row = next(r for r in store.list_campaigns(conn) if r["id"] == created["campaign_id"])

    assert row["assets_total"] == 3
    assert row["assets_embedded"] == 0


# ── the budget's own test was weaker than its name (adversarial review) ──────

def test_each_call_is_granted_strictly_less_time_as_the_budget_drains(conn, monkeypatch):
    """A constant timeout passes an assertion that only checks an upper bound, so the
    original version of this test would have accepted an implementation that never shrank
    the grant at all. What must hold is that the grant tracks the time actually left."""
    monkeypatch.setattr(config, "TOOL_TIME_BUDGET_SECONDS", 2.0)
    monkeypatch.setattr(config, "EMBED_TIMEOUT_SECONDS", 5.0)

    started = time.monotonic()
    observed = []

    def record(text, timeout=None):
        observed.append((time.monotonic() - started, timeout))
        time.sleep(0.05)
        return [0.0] * config.EMBED_DIM

    monkeypatch.setattr(embedding, "embed", record)

    deck = "\n\n".join(f"section {i} " + "word " * 200 for i in range(6))
    core.ingest_campaign(conn, title="Draining", deck_text=deck, confirm=True)

    assert len(observed) >= 3, "need several calls to see the trend"
    grants = [t for _, t in observed]
    assert grants == sorted(grants, reverse=True), f"grants must shrink: {grants}"
    for elapsed, granted in observed:
        assert granted <= 2.0 - elapsed + 0.05, (
            "a call may never be granted more than the budget actually has left"
        )


# ── the remaining unbounded handler (adversarial review, audit miss) ─────────

def test_bulk_import_commits_once_not_once_per_row(conn, monkeypatch):
    """Every other handler is now bounded, but bulk_import_metrics loops over
    caller-supplied rows committing each one - an fsync per row. Measured at ~2ms/row on an
    SSD; on the customer's Windows laptop with AV scanning every commit is far slower, and
    the caller controls how many rows there are."""
    cid = store.insert_campaign(conn, title="Target")
    commits = {"n": 0}

    class CountingConn:
        """sqlite3.Connection is a C type, so its methods cannot be patched - wrap it."""

        def __init__(self, inner):
            self._inner = inner

        def commit(self):
            commits["n"] += 1
            return self._inner.commit()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    rows = [{"campaign_id": cid, "detail": f"row {i}"} for i in range(50)]
    result = store.bulk_import_metrics(CountingConn(conn), rows, confirm=True)

    assert result["imported"] == 50
    assert commits["n"] <= 2, f"one commit for the batch, not {commits['n']}"


def test_bulk_import_stops_at_the_time_budget_and_says_where_it_stopped(conn, monkeypatch):
    """The only handler whose wall time was still proportional to caller input with nothing
    bounding it."""
    cid = store.insert_campaign(conn, title="Target")
    monkeypatch.setattr(config, "TOOL_TIME_BUDGET_SECONDS", 0.0)

    rows = [{"campaign_id": cid, "detail": f"row {i}"} for i in range(20)]
    result = store.bulk_import_metrics(conn, rows, confirm=True)

    assert result["imported"] < 20
    assert result["not_processed"] == 20 - result["imported"]


def test_bulk_import_still_reports_bad_rows_individually(conn):
    """No regression: a malformed row is reported, never crashes the batch."""
    cid = store.insert_campaign(conn, title="Target")
    rows = [{"campaign_id": cid, "detail": "fine"},
            {"campaign_id": "nope", "detail": "bad id"},
            "not even a dict"]

    result = store.bulk_import_metrics(conn, rows, confirm=True)

    assert result["imported"] == 1
    assert len(result["errors"]) == 2


# ── query-side failures must stay legible (adversarial review) ──────────────

def test_an_embedder_timeout_surfaces_as_a_readable_error(monkeypatch):
    """mcp_server only converts ValueError into a message the caller can read; anything
    else is replaced by a generic "Error executing tool X" with the detail discarded. An
    httpx timeout is not a ValueError, so a slow or unreachable embedder reached the
    marketer as an opaque failure with nothing to act on."""
    import httpx

    monkeypatch.setattr(config, "EMBED_PROVIDER", "ollama")

    def timeout(*a, **k):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(httpx, "post", timeout)

    with pytest.raises(ValueError) as exc:
        embedding.embed("anything")

    message = str(exc.value).lower()
    assert "ollama" in message, "name the component that failed"
    assert "timed out" in message or "not responding" in message


def test_an_unreachable_embedder_says_so_in_words(monkeypatch):
    import httpx

    monkeypatch.setattr(config, "EMBED_PROVIDER", "ollama")

    def refused(*a, **k):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", refused)

    with pytest.raises(ValueError) as exc:
        embedding.embed("anything")

    assert config.OLLAMA_URL in str(exc.value), "say where it tried to reach"


def test_a_nonsense_budget_is_rejected_at_startup(monkeypatch):
    """"nan" defeats every comparison at once - a deadline is never reached and every grant
    collapses to the floor - so the protection is silently gone rather than loudly wrong."""
    import importlib

    for bad in ("nan", "0", "-5", "soon"):
        monkeypatch.setenv("CAMPAIGN_POC_TOOL_BUDGET_SECONDS", bad)
        with pytest.raises(ValueError):
            importlib.reload(config)

    monkeypatch.delenv("CAMPAIGN_POC_TOOL_BUDGET_SECONDS")
    importlib.reload(config)   # restore a sane module for the rest of the suite


def test_the_budget_stays_clear_of_the_transport_ceiling():
    assert config.TOOL_TIME_BUDGET_SECONDS < config.TRANSPORT_CEILING_SECONDS, (
        "a budget at or above the ceiling cannot stop the handler being cut off"
    )
    assert config.TRANSPORT_CEILING_SECONDS - config.TOOL_TIME_BUDGET_SECONDS >= 10, (
        "leave room for extraction, database writes and the transport itself"
    )
