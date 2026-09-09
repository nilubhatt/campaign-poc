"""Adversarial review finding: _keep_asset used the bare original filename as the stored
name, so two campaigns uploading files with the same common name (hero.png, IMG_1234.jpg)
silently overwrote each other's stored original on disk."""
import pytest

import config
import core


@pytest.fixture(autouse=True)
def isolated_asset_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ASSET_DIR", tmp_path / "assets")


def test_keep_asset_does_not_collide_on_same_filename(tmp_path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    a_file = a_dir / "hero.png"
    b_file = b_dir / "hero.png"
    a_file.write_bytes(b"AAAA")
    b_file.write_bytes(b"BBBB")

    stored_a = core._keep_asset(a_file)
    stored_b = core._keep_asset(b_file)

    assert stored_a != stored_b
    assert (config.ASSET_DIR / stored_a).read_bytes() == b"AAAA"
    assert (config.ASSET_DIR / stored_b).read_bytes() == b"BBBB"


def test_keep_asset_preserves_extension(tmp_path):
    src = tmp_path / "deck.pptx"
    src.write_bytes(b"data")
    stored = core._keep_asset(src)
    assert stored.endswith(".pptx")
