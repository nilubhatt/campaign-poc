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
    # Real config.EMBED_DIM is 768 (nomic-embed-text); use small test vectors instead.
    monkeypatch.setattr(config, "EMBED_DIM", 3)


def test_init_add_search_roundtrip():
    conn = _conn()
    vectorstore.init(conn)
    vectorstore.add(conn, "chunk-a", [1.0, 0.0, 0.0])
    vectorstore.add(conn, "chunk-b", [0.0, 1.0, 0.0])

    hits = vectorstore.search(conn, [1.0, 0.0, 0.0], top_k=5)
    assert hits[0][0] == "chunk-a"
    assert hits[0][1] > hits[1][1]


def test_search_excludes_given_ids():
    conn = _conn()
    vectorstore.init(conn)
    vectorstore.add(conn, "a", [1.0, 0.0, 0.0])
    vectorstore.add(conn, "b", [0.9, 0.1, 0.0])
    hits = vectorstore.search(conn, [1.0, 0.0, 0.0], top_k=5, exclude={"a"})
    assert [h[0] for h in hits] == ["b"]


def test_add_rejects_wrong_dimension():
    conn = _conn()
    vectorstore.init(conn)
    with pytest.raises(ValueError):
        vectorstore.add(conn, "bad", [1.0, 2.0])  # dim 2 != configured EMBED_DIM 3


def test_add_same_id_twice_replaces_not_duplicates():
    conn = _conn()
    vectorstore.init(conn)
    vectorstore.add(conn, "a", [1.0, 0.0, 0.0])
    vectorstore.add(conn, "a", [0.0, 1.0, 0.0])
    hits = vectorstore.search(conn, [0.0, 1.0, 0.0], top_k=5)
    assert len(hits) == 1
    assert hits[0][1] > 0.99


def test_get_many_roundtrips_and_skips_missing():
    conn = _conn()
    vectorstore.init(conn)
    vectorstore.add(conn, "a", [1.0, 2.0, 3.0])
    vectorstore.add(conn, "b", [4.0, 5.0, 6.0])

    out = vectorstore.get_many(conn, ["a", "b", "missing"])
    assert out == {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}


def test_get_many_empty_input_returns_empty_dict():
    conn = _conn()
    vectorstore.init(conn)
    assert vectorstore.get_many(conn, []) == {}


def test_backend_name_reports_something_sane():
    conn = _conn()
    vectorstore.init(conn)
    assert vectorstore.backend_name(conn) in ("sqlite-vec", "python-cosine-fallback")
