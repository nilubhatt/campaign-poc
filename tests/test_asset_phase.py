"""
§9.1: phase every asset — `proposed` or `delivered`, plus a date.

The review: *"The library stores a plan and, eventually, a result, with nothing in between.
Two things belong in that gap: how far the delivered campaign drifted from the brief, and what
was going on in the market at the time. Without them, every outcome is read as if the brief
caused it."*

And on why it is load-bearing rather than a nice-to-have: *"Mexico is tagged not_liked +
performed_well and is the library's standing warning about confusing the two. But there is a
third possibility nobody can currently test: that what ran was not what was briefed. A library
that learns from briefs while measuring executions is learning from the wrong document, and has
no way to notice."*

*"Images already carry a fingerprint and a visual embedding. Add a `phase` — `proposed` for
creative lifted from the brief, `delivered` for photos that come back after the event — and a
date. **Everything below falls out of that one field.**"*

**The default is the load-bearing decision.** Creative lifted from a deck is `proposed`, and
that is what every asset in the library today is — so the default has to be `proposed` and an
upgraded database has to read that way, or §9.2 would compare a library of briefs against
itself and report zero drift with total confidence.

**`delivered` is a claim about when, so it carries the date.** A photo that came back after the
event is evidence about what ran; the same file with no date is evidence about nothing in
particular. The date is the asset's own, not the row's `created_at`, because a deck uploaded in
March can carry photos shot in January.
"""
import pytest

import core
import store


def _campaign(conn, title="Colombia v1"):
    return core.ingest_campaign(conn, title=title, market="LATAM", status="concluded",
                                detail="A launch.")["campaign_id"]


def _png(tmp_path, name="a.png", colour=(200, 30, 30)):
    from PIL import Image

    path = tmp_path / name
    Image.new("RGB", (64, 64), colour).save(path)
    return str(path)


# ── the field itself ────────────────────────────────────────────────────────

def test_an_asset_lifted_from_a_brief_is_proposed(conn, tmp_path):
    """Creative in a deck is what somebody INTENDS to run. Reading it as evidence of what ran
    is the confusion this whole phase exists to make visible."""
    cid = _campaign(conn)
    result = core.ingest_image_asset(conn, campaign_id=cid,
                                     asset_ref={"path": _png(tmp_path)})

    assert result["phase"] == "proposed"


def test_a_photo_that_came_back_is_delivered_and_says_when(conn, tmp_path):
    cid = _campaign(conn)
    result = core.ingest_image_asset(conn, campaign_id=cid,
                                     asset_ref={"path": _png(tmp_path)},
                                     phase="delivered", captured_on="2026-03-14")

    assert result["phase"] == "delivered"
    assert store.get_assets_for_campaign(conn, cid)[0]["captured_on"] == "2026-03-14"


def test_the_date_is_the_assets_own_not_the_rows(conn, tmp_path):
    """A deck uploaded in March can carry photos shot in January, and `created_at` records
    when this library was told rather than when the thing happened."""
    cid = _campaign(conn)
    core.ingest_image_asset(conn, campaign_id=cid, asset_ref={"path": _png(tmp_path)},
                            phase="delivered", captured_on="2026-01-09")

    asset = store.get_assets_for_campaign(conn, cid)[0]
    assert asset["captured_on"] == "2026-01-09"
    assert asset["created_at"] > 1_700_000_000, "which is a different fact"


def test_a_date_that_is_not_a_date_is_refused(conn, tmp_path):
    cid = _campaign(conn)
    with pytest.raises(ValueError) as e:
        core.ingest_image_asset(conn, campaign_id=cid, asset_ref={"path": _png(tmp_path)},
                                phase="delivered", captured_on="last spring")
    assert "captured_on" in str(e.value)


def test_the_vocabulary_is_forgiving_the_way_every_other_one_is(conn, tmp_path):
    """§5.1's rule. "Shot", "final" and "as delivered" are what a marketer types."""
    cid = _campaign(conn)
    for said in ("Delivered", "as delivered", "shot", "final"):
        result = core.ingest_image_asset(conn, campaign_id=cid,
                                         asset_ref={"path": _png(tmp_path, f"{said[:3]}.png")},
                                         phase=said)
        assert result["phase"] == "delivered", said


def test_a_phase_nobody_has_a_home_for_is_refused_with_the_valid_set(conn, tmp_path):
    cid = _campaign(conn)
    with pytest.raises(ValueError) as e:
        core.ingest_image_asset(conn, campaign_id=cid, asset_ref={"path": _png(tmp_path)},
                                phase="maybe")
    assert "proposed" in str(e.value) and "delivered" in str(e.value)


# ── the default, which is the load-bearing part ─────────────────────────────

def test_everything_already_in_the_library_reads_as_proposed(conn, tmp_path):
    """Every asset on file today came out of a deck. If an upgraded database read them as
    `delivered`, §9.2 would compare a library of briefs against itself and report zero drift
    with total confidence — the confident unfounded claim, from a schema default."""
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE campaigns (id TEXT PRIMARY KEY, title TEXT NOT NULL);
        CREATE TABLE assets (id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL,
                             modality TEXT, file_path TEXT NOT NULL, embedded INTEGER,
                             created_at REAL NOT NULL);
    """)
    old.execute("INSERT INTO campaigns VALUES ('camp_old', 'Peru')")
    old.execute("INSERT INTO assets VALUES ('asset_old','camp_old','image','a.png',1,1.0)")
    old.commit()
    old.close()

    upgraded = sqlite3.connect(path)
    upgraded.row_factory = sqlite3.Row
    store.upgrade(upgraded)

    asset = store.get_assets_for_campaign(upgraded, "camp_old")[0]
    assert asset["phase"] == "proposed"
    assert asset["captured_on"] is None, "nobody said when, and a guess would be worse"


def test_an_asset_extracted_from_a_deck_is_proposed(conn, tmp_path):
    """The other route in. Images pulled out of an uploaded deck are the deck's creative."""
    from PIL import Image

    deck = tmp_path / "deck.pdf"
    Image.new("RGB", (80, 80), (10, 120, 200)).save(tmp_path / "slide.png")
    # A PDF with one embedded image is more machinery than this needs; the claim is about the
    # default on the extraction path, which `insert_asset` owns.
    cid = _campaign(conn)
    aid = store.insert_asset(conn, cid, file_path="slide.png")

    assert store.get_assets_for_campaign(conn, cid)[0]["phase"] == "proposed"


# ── what the field is for ───────────────────────────────────────────────────

def test_the_two_phases_can_be_told_apart(conn, tmp_path):
    """§9.2 needs exactly this: which assets were briefed and which came back."""
    cid = _campaign(conn)
    core.ingest_image_asset(conn, campaign_id=cid, asset_ref={"path": _png(tmp_path, "p.png")})
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": _png(tmp_path, "d.png", (30, 200, 30))},
                            phase="delivered", captured_on="2026-03-14")

    assert len(store.assets_in_phase(conn, cid, "proposed")) == 1
    assert len(store.assets_in_phase(conn, cid, "delivered")) == 1


def test_a_campaign_says_whether_anything_has_come_back(conn, tmp_path):
    """"A library that learns from briefs while measuring executions is learning from the wrong
    document, and has no way to notice." This is the noticing: a concluded campaign whose
    assets are all `proposed` has never been checked against what ran."""
    cid = _campaign(conn)
    core.ingest_image_asset(conn, campaign_id=cid, asset_ref={"path": _png(tmp_path)})

    record = store.get_campaign(conn, cid)
    assert record["has_delivered_assets"] is False

    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": _png(tmp_path, "d.png", (30, 200, 30))},
                            phase="delivered", captured_on="2026-03-14")
    assert store.get_campaign(conn, cid)["has_delivered_assets"] is True


def test_a_proposed_asset_cannot_claim_a_capture_date(conn, tmp_path):
    """A date on a proposed asset says a photograph exists of something that has not happened.
    Refused rather than ignored, because a stored date nothing reads is a fact somebody will
    later believe."""
    cid = _campaign(conn)
    with pytest.raises(ValueError) as e:
        core.ingest_image_asset(conn, campaign_id=cid, asset_ref={"path": _png(tmp_path)},
                                phase="proposed", captured_on="2026-03-14")
    assert "delivered" in str(e.value)


def test_it_reaches_the_model_over_the_protocol(conn, tmp_path):
    import asyncio
    import json

    import mcp_server

    async def call(name, args):
        return json.loads((await mcp_server.mcp.call_tool(name, args)).content[0].text)

    cid = _campaign(conn)
    got = asyncio.run(call("upload_image_asset", {
        "campaign_id": cid, "asset_ref": {"path": _png(tmp_path)},
        "phase": "delivered", "captured_on": "2026-03-14"}))

    assert got["phase"] == "delivered"
    assert got["captured_on"] == "2026-03-14"


def test_a_date_in_the_future_is_refused(conn, tmp_path):
    """A photograph of what ran cannot have been taken tomorrow, and a typo'd year would
    otherwise sit in the record looking like a fact."""
    cid = _campaign(conn)
    with pytest.raises(ValueError) as e:
        core.ingest_image_asset(conn, campaign_id=cid, asset_ref={"path": _png(tmp_path)},
                                phase="delivered", captured_on="2099-01-01")
    assert "future" in str(e.value)


def test_a_date_is_stored_normalised(conn, tmp_path):
    """`fromisoformat` also accepts `20260314` and `2026-W11-5`, and a later comparison
    against a campaign's window is a string comparison — so the stored form has to be one
    form."""
    cid = _campaign(conn)
    core.ingest_image_asset(conn, campaign_id=cid, asset_ref={"path": _png(tmp_path)},
                            phase="delivered", captured_on="20260314")

    assert store.get_assets_for_campaign(conn, cid)[0]["captured_on"] == "2026-03-14"
