"""
§6.6 (ship-first layer): perceptual hashing catches same/near-same images reused across
campaigns, even after resize/recompress — cheap, no ML. CLIP (aesthetic similarity) is a
separate, heavier follow-on not built here.
"""
import numpy as np
from PIL import Image

import core
import images
import store


def _make_image(path, seed, size=(64, 64)):
    """A photo-like textured image — pHash needs spatial variation to be meaningful (flat
    color blocks all hash identically regardless of color) but also needs real low-frequency
    structure to survive resizing, unlike pure per-pixel noise: generate small-scale noise,
    then upscale with interpolation, the way a real photo has correlated neighboring pixels."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    Image.fromarray(small, mode="RGB").resize(size, Image.BICUBIC).save(path)


def test_phash_identical_images_have_zero_distance(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    _make_image(a, seed=1)
    _make_image(b, seed=1)
    assert images.hamming_distance(images.phash(a), images.phash(b)) == 0


def test_phash_survives_resize(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    _make_image(a, seed=1, size=(256, 256))
    _make_image(b, seed=1, size=(64, 64))
    assert images.hamming_distance(images.phash(a), images.phash(b)) <= 4


def test_phash_different_images_have_large_distance(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    _make_image(a, seed=1)
    _make_image(b, seed=2)
    assert images.hamming_distance(images.phash(a), images.phash(b)) > 10


def test_is_match_respects_threshold():
    assert images.is_match("0000000000000000", "0000000000000000", threshold=0) is True
    assert images.is_match("0000000000000000", "ffffffffffffffff", threshold=0) is False


def test_ingest_image_asset_stores_and_fingerprints(conn, tmp_path):
    cid = store.insert_campaign(conn, title="X")
    img = tmp_path / "hero.png"
    _make_image(img, seed=1)

    r = core.ingest_image_asset(conn, campaign_id=cid, asset_ref={"path": str(img)})
    assert r["fingerprinted"] is True
    assets = store.get_assets_for_campaign(conn, cid)
    assert len(assets) == 1
    assert assets[0]["modality"] == "image"


def test_check_image_provenance_flags_reuse_in_a_different_region(conn, tmp_path):
    hero = tmp_path / "hero.png"
    _make_image(hero, seed=1)

    mexico = store.insert_campaign(conn, title="Mexico Launch", region="LATAM")
    core.ingest_image_asset(conn, campaign_id=mexico, asset_ref={"path": str(hero)})

    apac = store.insert_campaign(conn, title="APAC Launch", region="APAC")
    result = core.check_image_provenance(conn, asset_ref={"path": str(hero)}, campaign_id=apac)

    assert len(result["matches"]) == 1
    m = result["matches"][0]
    assert m["campaign_id"] == mexico
    assert m["hamming_distance"] == 0
    assert "different region" in m["flag"]
    assert "LATAM" in m["flag"] and "APAC" in m["flag"]


def test_check_image_provenance_no_flag_when_same_region(conn, tmp_path):
    hero = tmp_path / "hero.png"
    _make_image(hero, seed=1)

    first = store.insert_campaign(conn, title="A", region="APAC")
    core.ingest_image_asset(conn, campaign_id=first, asset_ref={"path": str(hero)})

    second = store.insert_campaign(conn, title="B", region="APAC")
    result = core.check_image_provenance(conn, asset_ref={"path": str(hero)}, campaign_id=second)

    assert result["matches"][0]["flag"] is None


def test_check_image_provenance_excludes_the_campaigns_own_assets(conn, tmp_path):
    hero = tmp_path / "hero.png"
    _make_image(hero, seed=1)
    cid = store.insert_campaign(conn, title="X")
    core.ingest_image_asset(conn, campaign_id=cid, asset_ref={"path": str(hero)})

    result = core.check_image_provenance(conn, asset_ref={"path": str(hero)}, campaign_id=cid)
    assert result["matches"] == []


def test_check_image_provenance_no_match_returns_empty(conn, tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    _make_image(a, seed=1)
    _make_image(b, seed=2)
    cid = store.insert_campaign(conn, title="X")
    core.ingest_image_asset(conn, campaign_id=cid, asset_ref={"path": str(a)})

    result = core.check_image_provenance(conn, asset_ref={"path": str(b)})
    assert result["matches"] == []


def test_check_image_provenance_ranks_closest_match_first(conn, tmp_path):
    query = tmp_path / "query.png"
    _make_image(query, seed=1)
    unrelated = tmp_path / "unrelated.png"
    _make_image(unrelated, seed=2)

    c1 = store.insert_campaign(conn, title="Unrelated")
    core.ingest_image_asset(conn, campaign_id=c1, asset_ref={"path": str(unrelated)})
    c2 = store.insert_campaign(conn, title="Exact match")
    core.ingest_image_asset(conn, campaign_id=c2, asset_ref={"path": str(query)})

    result = core.check_image_provenance(conn, asset_ref={"path": str(query)})
    assert result["matches"][0]["campaign_id"] == c2  # exact match ranks first
    assert result["matches"][0]["hamming_distance"] == 0
