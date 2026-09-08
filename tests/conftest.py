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


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "campaigns.db")
    monkeypatch.setattr(config, "ASSET_DIR", tmp_path / "assets")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "EMBED_PROVIDER", "hash")

    store.init_db()
    c = store.connect()
    yield c
    c.close()
