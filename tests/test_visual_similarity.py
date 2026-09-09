"""
§6.6 (CLIP layer): aesthetic/regional visual similarity — same product different photo, same
styling, "looks like the APAC shoot" — distinct from images.py's pHash exact-reuse detection.
Uses the offline `hash` CLIP provider (conftest sets it for every test) so no torch/model
download is needed for the automated suite.
"""
import numpy as np
from PIL import Image

import core
import store


def _make_image(path, seed, size=(64, 64)):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(8, 8, 3), dtype="uint8")
    Image.fromarray(small, mode="RGB").resize(size, Image.BICUBIC).save(path)


def test_ingest_image_asset_also_computes_visual_embedding(conn, tmp_path):
    cid = store.insert_campaign(conn, title="X")
    img = tmp_path / "hero.png"
    _make_image(img, seed=1)

    r = core.ingest_image_asset(conn, campaign_id=cid, asset_ref={"path": str(img)})
    assert r["fingerprinted"] is True
    assert r["visually_embedded"] is True
    assets = store.get_assets_for_campaign(conn, cid)
    assert assets[0]["embedded"] == 1


def test_find_similar_images_ranks_by_visual_similarity(conn, tmp_path):
    query = tmp_path / "query.png"
    _make_image(query, seed=1)

    exact = store.insert_campaign(conn, title="Exact visual match")
    core.ingest_image_asset(conn, campaign_id=exact, asset_ref={"path": str(query)})

    unrelated = store.insert_campaign(conn, title="Unrelated")
    other = tmp_path / "other.png"
    _make_image(other, seed=2)
    core.ingest_image_asset(conn, campaign_id=unrelated, asset_ref={"path": str(other)})

    result = core.find_similar_images(conn, asset_ref={"path": str(query)})
    assert result["matches"][0]["campaign_id"] == exact
    assert result["matches"][0]["similarity"] > result["matches"][1]["similarity"]


def test_find_similar_images_excludes_the_campaigns_own_assets(conn, tmp_path):
    img = tmp_path / "hero.png"
    _make_image(img, seed=1)
    cid = store.insert_campaign(conn, title="X")
    core.ingest_image_asset(conn, campaign_id=cid, asset_ref={"path": str(img)})

    result = core.find_similar_images(conn, asset_ref={"path": str(img)}, campaign_id=cid)
    assert result["matches"] == []


def test_find_similar_images_filters_by_region_first(conn, tmp_path):
    img = tmp_path / "hero.png"
    _make_image(img, seed=1)

    apac = store.insert_campaign(conn, title="APAC shoot", region="APAC")
    core.ingest_image_asset(conn, campaign_id=apac, asset_ref={"path": str(img)})

    latam = store.insert_campaign(conn, title="LATAM shoot", region="LATAM")
    core.ingest_image_asset(conn, campaign_id=latam, asset_ref={"path": str(img)})

    result = core.find_similar_images(conn, asset_ref={"path": str(img)}, region="APAC")
    ids = [m["campaign_id"] for m in result["matches"]]
    assert apac in ids
    assert latam not in ids


def test_find_similar_images_flags_a_different_region_match(conn, tmp_path):
    img = tmp_path / "hero.png"
    _make_image(img, seed=1)

    mexico = store.insert_campaign(conn, title="Mexico shoot", region="LATAM")
    core.ingest_image_asset(conn, campaign_id=mexico, asset_ref={"path": str(img)})

    apac = store.insert_campaign(conn, title="APAC proposal", region="APAC")
    result = core.find_similar_images(conn, asset_ref={"path": str(img)}, campaign_id=apac)

    match = next(m for m in result["matches"] if m["campaign_id"] == mexico)
    assert match["flag"] is not None
    assert "LATAM" in match["flag"] and "APAC" in match["flag"]


def test_find_similar_images_no_assets_returns_empty(conn, tmp_path):
    img = tmp_path / "hero.png"
    _make_image(img, seed=1)
    result = core.find_similar_images(conn, asset_ref={"path": str(img)})
    assert result["matches"] == []
