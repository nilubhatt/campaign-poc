"""
§6.6/6.7: CLIP visual embeddings are a different dimensionality (512) than text-chunk
embeddings (768, nomic-embed-text) and can't share one vec0 table (fixed column width).
vectorstore gains a `space` param so the same module serves both vector spaces without
duplicating the sqlite-vec/fallback logic - default space="campaign" must stay 100%
backward compatible with every existing caller/test that doesn't pass it.
"""
import sqlite3

import pytest

import config
import vectorstore


def _conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    return c


@pytest.fixture(autouse=True)
def small_embed_dim(monkeypatch):
    monkeypatch.setattr(config, "EMBED_DIM", 3)


def test_default_space_still_works_unchanged():
    conn = _conn()
    vectorstore.init(conn)
    vectorstore.add(conn, "a", [1.0, 0.0, 0.0])
    hits = vectorstore.search(conn, [1.0, 0.0, 0.0], top_k=5)
    assert hits[0][0] == "a"


def test_second_space_has_independent_storage_and_dimension():
    conn = _conn()
    vectorstore.init(conn)  # space="campaign", dim=3 (via EMBED_DIM)
    vectorstore.init(conn, space="asset", dim=5)

    vectorstore.add(conn, "chunk-1", [1.0, 0.0, 0.0])
    vectorstore.add(conn, "asset-1", [1.0, 0.0, 0.0, 0.0, 0.0], space="asset", dim=5)

    campaign_hits = vectorstore.search(conn, [1.0, 0.0, 0.0], top_k=5)
    asset_hits = vectorstore.search(conn, [1.0, 0.0, 0.0, 0.0, 0.0], top_k=5, space="asset")

    assert [h[0] for h in campaign_hits] == ["chunk-1"]
    assert [h[0] for h in asset_hits] == ["asset-1"]


def test_wrong_dimension_for_a_space_is_rejected():
    conn = _conn()
    vectorstore.init(conn, space="asset", dim=5)
    with pytest.raises(ValueError):
        vectorstore.add(conn, "bad", [1.0, 2.0, 3.0], space="asset", dim=5)


def test_get_many_respects_space_and_dim():
    conn = _conn()
    vectorstore.init(conn)  # space="campaign", dim=3 (via EMBED_DIM)
    vectorstore.init(conn, space="asset", dim=4)
    vectorstore.add(conn, "a", [1.0, 2.0, 3.0, 4.0], space="asset", dim=4)
    vectorstore.add(conn, "a", [9.0, 9.0, 9.0], space="campaign")  # same id, other space

    out = vectorstore.get_many(conn, ["a"], space="asset", dim=4)
    assert out == {"a": [1.0, 2.0, 3.0, 4.0]}


def test_delete_many_respects_space():
    conn = _conn()
    vectorstore.init(conn)  # space="campaign", dim=3 (via EMBED_DIM)
    vectorstore.init(conn, space="asset", dim=4)
    vectorstore.add(conn, "a", [1.0, 2.0, 3.0, 4.0], space="asset", dim=4)
    vectorstore.add(conn, "a", [9.0, 9.0, 9.0], space="campaign")

    vectorstore.delete_many(conn, ["a"], space="asset")

    assert vectorstore.get_many(conn, ["a"], space="asset", dim=4) == {}
    assert vectorstore.get_many(conn, ["a"], space="campaign") == {"a": [9.0, 9.0, 9.0]}


def test_search_excludes_within_the_same_space_only():
    conn = _conn()
    vectorstore.init(conn, space="asset", dim=3)
    vectorstore.add(conn, "a", [1.0, 0.0, 0.0], space="asset", dim=3)
    vectorstore.add(conn, "b", [0.9, 0.1, 0.0], space="asset", dim=3)

    hits = vectorstore.search(conn, [1.0, 0.0, 0.0], top_k=5, exclude={"a"}, space="asset")
    assert [h[0] for h in hits] == ["b"]
