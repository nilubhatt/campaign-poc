"""
Shared fixtures. Every test gets an isolated data dir and the offline `hash` embed
provider (no Ollama/Voyage dependency) — deterministic, no network.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import store


@pytest.fixture(autouse=True)
def _never_the_real_data_directory(tmp_path, monkeypatch):
    """§12.2: no test may read the developer's own rulebook.

    `rulebook.overlay_path()` is `config.DATA_DIR / "rulebook.yaml"`, and eighteen tests load
    the rulebook WITHOUT the `conn` fixture — so `DATA_DIR` was the real
    `~/campaign-poc-data`. It is empty today, so they passed. The moment anybody does what
    §12.2 tells them to do and writes a rulebook there, those tests would start failing
    against a file that has nothing to do with them, and the failure would point at the
    product rather than at the file.

    Autouse and unconditional: the `conn` fixture already redirects `DATA_DIR`, and this
    covers every test that does not use it.
    """
    # NOT created. Two `test_fetch_weights` tests assert `tmp_path` is empty after a failed
    # download, and a directory this fixture made would be the thing that "survived". Nothing
    # needs it to exist: `overlay_path().exists()` is False either way, and the tests that
    # write an overlay make it themselves.
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data", raising=False)
    try:
        import rulebook

        rulebook.load.cache_clear()
        yield
        rulebook.load.cache_clear()
    except ImportError:
        yield


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "campaigns.db")
    monkeypatch.setattr(config, "ASSET_DIR", tmp_path / "assets")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "EMBED_PROVIDER", "hash")
    monkeypatch.setattr(config, "CLIP_PROVIDER", "hash")

    store.init_db()
    c = store.connect()
    yield c
    c.close()
