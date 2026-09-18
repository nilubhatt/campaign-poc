"""
Product review defect 03 (P0): huggingface.co is blocked in the customer environment.

Two independent TLS stacks failed against it on the target machine - Python/OpenSSL with
"[SSL: WRONG_VERSION_NUMBER]" and .NET/Schannel with "handshake failed due to an unexpected
packet format" - both meaning something answered :443 in plaintext, i.e. an interception
appliance returning a block page. pip and github.com worked on the same host, so TLS was
healthy and this host was filtered specifically.

Items 1.1 and 1.2 made local weights possible and then shipped them. This item makes "no
Hub dependency at runtime" a PROPERTY rather than a claim:

* `HF_HUB_OFFLINE=1` is set before open_clip is imported whenever the weights resolved
  locally, so the library cannot reach for the network even if some future code path asks
  it to. A test asserting "we didn't call the network" only covers the paths it exercises;
  the env var covers the ones nobody thought of.
* A cache-gated integration test blocks sockets outright and loads a real checkpoint, so
  the claim is demonstrated against the real library rather than a stub.
"""
import os
import socket
from pathlib import Path

import pytest

import clip_embed
import config


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    clip_embed._model = None
    clip_embed._preprocess = None
    clip_embed._resolution = None
    clip_embed._load_error = None
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    yield
    clip_embed._model = None
    clip_embed._preprocess = None
    clip_embed._resolution = None
    clip_embed._load_error = None


class _FakeModel:
    def eval(self):
        return self


@pytest.fixture
def fake_open_clip(monkeypatch):
    seen = {}

    def stub(model_name, pretrained=None, **kwargs):
        # Captured at the moment open_clip is used, which is what matters: the variable has
        # to be set BEFORE the library reads it, not merely at some point afterwards.
        seen["hf_hub_offline"] = os.environ.get("HF_HUB_OFFLINE")
        seen["pretrained"] = pretrained
        return (_FakeModel(), None, "preprocess")

    monkeypatch.setattr("open_clip.create_model_and_transforms", stub)
    return seen


def _local_checkpoint() -> Path | None:
    """A real checkpoint, if this machine has one. Looked up rather than downloaded: a
    300MB fetch does not belong in a unit-test run."""
    bundled = config.BUNDLED_WEIGHTS_DIR
    if bundled.is_dir():
        for name in clip_embed._CHECKPOINT_NAMES:
            if (bundled / name).is_file():
                return bundled / name
    # The snapshots/ tree, not blobs/: blobs are named by hash with no extension, and
    # open_clip picks its loader from the suffix (.safetensors vs torch.load), so a blob
    # path fails to load even though the bytes are right.
    cache = Path.home() / ".cache/huggingface/hub"
    if cache.is_dir():
        for name in clip_embed._CHECKPOINT_NAMES:
            for found in cache.glob(
                    f"models--timm--vit_base_patch32_clip_224.openai/snapshots/*/{name}"):
                if found.is_file():
                    return found
    return None


# ── the property: offline mode is on whenever weights are local ──────────────

def test_local_weights_put_the_hub_client_in_offline_mode(monkeypatch, tmp_path, fake_open_clip):
    weights = tmp_path / "open_clip_model.safetensors"
    weights.write_bytes(b"checkpoint")
    monkeypatch.setattr(config, "CLIP_PROVIDER", "openclip")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", str(weights))

    clip_embed._load_model()

    assert fake_open_clip["hf_hub_offline"] == "1", (
        "with a local checkpoint the Hub client must be offline before it is imported - "
        "otherwise a future code path can still reach for a host the customer blocks"
    )


def test_bundled_weights_also_force_offline_mode(monkeypatch, tmp_path, fake_open_clip):
    models = tmp_path / "models"
    models.mkdir()
    (models / "open_clip_model.safetensors").write_bytes(b"checkpoint")
    monkeypatch.setattr(config, "CLIP_PROVIDER", "openclip")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", "")
    monkeypatch.setattr(config, "BUNDLED_WEIGHTS_DIR", models)

    clip_embed._load_model()

    assert fake_open_clip["hf_hub_offline"] == "1"


def test_the_tag_path_is_left_online(monkeypatch, fake_open_clip):
    """A source checkout resolving the tag NEEDS the Hub; forcing it offline would turn a
    working developer setup into a confusing failure."""
    monkeypatch.setattr(config, "CLIP_PROVIDER", "openclip")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", "")
    monkeypatch.setattr(config, "BUNDLED_WEIGHTS_DIR", Path("/nonexistent"))
    monkeypatch.setattr(config, "is_installed", lambda: False)

    clip_embed._load_model()

    assert fake_open_clip["pretrained"] == config.CLIP_PRETRAINED
    assert fake_open_clip["hf_hub_offline"] != "1"


def test_an_operators_explicit_offline_setting_is_not_overwritten(monkeypatch, fake_open_clip):
    """If someone deliberately set HF_HUB_OFFLINE=0 we should not silently flip it."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(config, "CLIP_PROVIDER", "openclip")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", "")
    monkeypatch.setattr(config, "BUNDLED_WEIGHTS_DIR", Path("/nonexistent"))
    monkeypatch.setattr(config, "is_installed", lambda: False)

    clip_embed._load_model()

    assert os.environ["HF_HUB_OFFLINE"] == "0"


# ── the demonstration: a real checkpoint, with the network actually blocked ───

@pytest.mark.skipif(_local_checkpoint() is None,
                    reason="no local CLIP checkpoint on this machine (run "
                           "`python scripts/fetch_weights.py models` to enable)")
def test_a_real_checkpoint_loads_and_embeds_with_the_network_blocked(monkeypatch, tmp_path):
    """The claim defect 03 needs: with weights on disk, loading the real library makes no
    network calls at all. Stubs cannot show this - only the real open_clip can."""
    import numpy as np
    from PIL import Image

    checkpoint = _local_checkpoint()
    monkeypatch.setattr(config, "CLIP_PROVIDER", "openclip")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", str(checkpoint))

    attempts = []

    def blocked(*args, **kwargs):
        attempts.append(args[:1])
        raise AssertionError(f"network access attempted: {args[:1]}")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", blocked)

    img = tmp_path / "probe.png"
    Image.fromarray(
        np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype="uint8")).save(img)

    vector = clip_embed.embed_image(img)

    assert attempts == [], "a local checkpoint must not touch the network"
    assert len(vector) == config.CLIP_EMBED_DIM
    assert abs(sum(x * x for x in vector) ** 0.5 - 1.0) < 1e-5


# ── preprocess parity: the silent way vectors stop being comparable ──────────

def test_a_local_checkpoint_gets_the_same_preprocessing_as_the_tag():
    """Loading by PATH skips the tag's preprocess metadata and falls back to the model's
    own defaults. For this model they are identical - but "identical today" is exactly the
    kind of thing that changes under you, and if it ever diverges the failure is silent:
    vectors keep computing, they just stop being comparable with every vector already in
    the database. So pin it rather than assume it."""
    import open_clip
    from open_clip.pretrained import get_pretrained_cfg

    tag_cfg = get_pretrained_cfg(config.CLIP_MODEL_NAME, config.CLIP_PRETRAINED)
    _, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL_NAME, pretrained=None)   # the path-style code path

    normalize = next(t for t in preprocess.transforms if type(t).__name__ == "Normalize")
    assert tuple(normalize.mean) == pytest.approx(tag_cfg["mean"])
    assert tuple(normalize.std) == pytest.approx(tag_cfg["std"])

    resize = next(t for t in preprocess.transforms if type(t).__name__ == "Resize")
    assert tag_cfg["interpolation"] in str(resize.interpolation).lower()
