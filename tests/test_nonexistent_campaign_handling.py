"""
Adversarial review finding: a bogus campaign_id was handled inconsistently across tools -
some crashed after a side effect (a file copied to disk before the FK insert failed), some
silently returned misleading data (a "same region" flag for a campaign that doesn't exist),
some raised a confusing unrelated error message. Every entry point should fail the same way:
a clean {"error": ...} (or a clear exception for find_similar, which raises rather than
returning a dict), with no side effects.
"""
from pathlib import Path

import pytest

import config
import core
import store


def _make_image(path, seed=1):
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(8, 8, 3), dtype="uint8")
    Image.fromarray(small, mode="RGB").resize((64, 64), Image.BICUBIC).save(path)


def test_ingest_image_asset_rejects_nonexistent_campaign_without_copying_the_file(conn, tmp_path):
    img = tmp_path / "hero.png"
    _make_image(img)

    result = core.ingest_image_asset(conn, campaign_id="does-not-exist", asset_ref={"path": str(img)})

    assert "error" in result
    # no orphan file left behind in ASSET_DIR
    assert list(config.ASSET_DIR.glob("*")) == [] if config.ASSET_DIR.exists() else True


def test_check_image_provenance_rejects_nonexistent_campaign_id(conn, tmp_path):
    img = tmp_path / "hero.png"
    _make_image(img)
    result = core.check_image_provenance(conn, asset_ref={"path": str(img)}, campaign_id="does-not-exist")
    assert "error" in result


def test_find_similar_images_rejects_nonexistent_campaign_id(conn, tmp_path):
    img = tmp_path / "hero.png"
    _make_image(img)
    result = core.find_similar_images(conn, asset_ref={"path": str(img)}, campaign_id="does-not-exist")
    assert "error" in result


def test_add_metrics_preview_rejects_nonexistent_campaign(conn):
    result = core.add_metrics(conn, "does-not-exist", detail="x", confirm=False)
    assert "error" in result


def test_add_metrics_confirm_rejects_nonexistent_campaign_cleanly(conn):
    result = core.add_metrics(conn, "does-not-exist", detail="x", confirm=True)
    assert "error" in result


def test_find_similar_rejects_nonexistent_campaign_id_with_a_clear_error(conn):
    with pytest.raises(ValueError, match="not found"):
        core.find_similar(conn, campaign_id="does-not-exist", top_k=5)


def test_check_image_provenance_rejects_a_non_image_file_cleanly(conn, tmp_path):
    not_an_image = tmp_path / "notes.txt"
    not_an_image.write_text("this is not an image")
    result = core.check_image_provenance(conn, asset_ref={"path": str(not_an_image)})
    assert "error" in result


def test_find_similar_images_rejects_a_non_image_file_cleanly(conn, tmp_path):
    not_an_image = tmp_path / "notes.txt"
    not_an_image.write_text("this is not an image")
    result = core.find_similar_images(conn, asset_ref={"path": str(not_an_image)})
    assert "error" in result


def test_check_image_provenance_excludes_superseded_campaigns(conn, tmp_path):
    img = tmp_path / "hero.png"
    _make_image(img)
    old = store.insert_campaign(conn, title="Old")
    core.ingest_image_asset(conn, campaign_id=old, asset_ref={"path": str(img)})
    store.insert_campaign(conn, title="New", supersedes=old)

    result = core.check_image_provenance(conn, asset_ref={"path": str(img)})
    assert result["matches"] == []


def test_find_similar_images_excludes_superseded_campaigns_unfiltered_path(conn, tmp_path):
    img = tmp_path / "hero.png"
    _make_image(img)
    old = store.insert_campaign(conn, title="Old")
    core.ingest_image_asset(conn, campaign_id=old, asset_ref={"path": str(img)})
    store.insert_campaign(conn, title="New", supersedes=old)

    result = core.find_similar_images(conn, asset_ref={"path": str(img)})
    assert result["matches"] == []
