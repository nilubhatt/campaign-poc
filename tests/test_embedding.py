import math

import pytest

import embedding


def test_cosine_identical_vectors_is_one():
    v = [1.0, 2.0, 3.0]
    assert embedding.cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_is_zero():
    assert embedding.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors_is_negative_one():
    assert embedding.cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_handles_zero_vector_without_dividing_by_zero():
    assert embedding.cosine([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert embedding.cosine([], []) == 0.0
    assert embedding.cosine([1.0], [1.0, 2.0]) == 0.0  # mismatched length


def test_rank_orders_by_similarity_desc_and_respects_top_k():
    query = [1.0, 0.0]
    candidates = [
        ("a", [1.0, 0.0]),    # sim 1.0
        ("b", [0.0, 1.0]),    # sim 0.0
        ("c", [0.9, 0.1]),    # sim ~0.99
    ]
    ranked = embedding.rank(query, candidates, top_k=2)
    assert [cid for cid, _ in ranked] == ["a", "c"]


def test_rank_excludes_given_ids():
    query = [1.0, 0.0]
    candidates = [("a", [1.0, 0.0]), ("b", [0.9, 0.1])]
    ranked = embedding.rank(query, candidates, top_k=5, exclude={"a"})
    assert [cid for cid, _ in ranked] == ["b"]


def test_embed_hash_is_deterministic_and_normalized(monkeypatch):
    import config
    monkeypatch.setattr(config, "EMBED_PROVIDER", "hash")
    v1 = embedding.embed("audience targeting social campaign")
    v2 = embedding.embed("audience targeting social campaign")
    assert v1 == v2
    norm = math.sqrt(sum(x * x for x in v1))
    assert norm == pytest.approx(1.0)


def test_embed_hash_different_text_gives_different_vector(monkeypatch):
    import config
    monkeypatch.setattr(config, "EMBED_PROVIDER", "hash")
    v1 = embedding.embed("apac summer launch")
    v2 = embedding.embed("completely unrelated enterprise renewal")
    assert v1 != v2


def test_embed_rejects_empty_text(monkeypatch):
    import config
    monkeypatch.setattr(config, "EMBED_PROVIDER", "hash")
    with pytest.raises(ValueError):
        embedding.embed("   ")


def test_embed_rejects_unknown_provider(monkeypatch):
    import config
    monkeypatch.setattr(config, "EMBED_PROVIDER", "not-a-real-provider")
    with pytest.raises(ValueError):
        embedding.embed("some text")
