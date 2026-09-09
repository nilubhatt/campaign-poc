"""
§6.6 (CLIP layer): aesthetic/regional visual similarity, not exact reuse (images.py's pHash
job). The 'hash' provider is dependency-free (no torch, no model download) so the automated
suite never touches torch - same role as embedding.py's 'hash' text provider. The real
'openclip' provider is exercised only by manual smoke test (like Ollama for text), not CI.
"""
import math

import pytest

import clip_embed
import config


@pytest.fixture(autouse=True)
def hash_provider(monkeypatch):
    monkeypatch.setattr(config, "CLIP_PROVIDER", "hash")


def _make_image(path, seed):
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(8, 8, 3), dtype="uint8")
    Image.fromarray(small, mode="RGB").resize((64, 64), Image.BICUBIC).save(path)


def test_embed_image_hash_provider_is_deterministic(tmp_path):
    img = tmp_path / "a.png"
    _make_image(img, seed=1)
    v1 = clip_embed.embed_image(img)
    v2 = clip_embed.embed_image(img)
    assert v1 == v2


def test_embed_image_hash_provider_has_configured_dim(tmp_path):
    img = tmp_path / "a.png"
    _make_image(img, seed=1)
    vec = clip_embed.embed_image(img)
    assert len(vec) == config.CLIP_EMBED_DIM


def test_embed_image_hash_provider_is_normalized(tmp_path):
    img = tmp_path / "a.png"
    _make_image(img, seed=1)
    vec = clip_embed.embed_image(img)
    norm = math.sqrt(sum(x * x for x in vec))
    assert norm == pytest.approx(1.0)


def test_embed_image_hash_provider_different_images_differ(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    _make_image(a, seed=1)
    _make_image(b, seed=2)
    assert clip_embed.embed_image(a) != clip_embed.embed_image(b)


def test_embed_image_unknown_provider_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CLIP_PROVIDER", "not-a-real-provider")
    img = tmp_path / "a.png"
    _make_image(img, seed=1)
    with pytest.raises(ValueError):
        clip_embed.embed_image(img)
