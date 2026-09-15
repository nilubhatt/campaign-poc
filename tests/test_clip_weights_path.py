"""
Product review defect 02 (P0): "No supported way to point at local weights."

The reviewer held the correct .bin file and had no supported way to tell the application
where it was - the workaround was hand-fabricating a Hugging Face cache entry (blobs/,
snapshots/<rev>/, refs/main) so the library would serve the file from cache after its
network call failed. open_clip accepts a filesystem path anywhere it accepts a tag, so this
is a pass-through, not a new loading path.

Two design points, both from review of this item:

* Resolution is SEPARATE from loading. `resolve_weights()` is cheap, never raises, and
  returns what happened (source, path, ok, remedy) so startup can record it and
  health_check (plan 2.3) can report it. Only an actual attempt to embed an image raises.
* A misconfigured path must NOT take the server down at boot. Text search, upload and
  evaluation do not need CLIP; killing them because the vision weights are missing would
  break the "graceful degradation" the review verified as working. Startup records the
  problem; the image path reports it per call.
"""
import pytest

import clip_embed
import config


class _FakeModel:
    def eval(self):
        return self


@pytest.fixture(autouse=True)
def reset_model_cache():
    clip_embed._model = None
    clip_embed._preprocess = None
    clip_embed._resolution = None
    clip_embed._load_error = None
    yield
    clip_embed._model = None
    clip_embed._preprocess = None
    clip_embed._resolution = None
    clip_embed._load_error = None


@pytest.fixture
def captured(monkeypatch):
    """Capture what gets passed to open_clip without loading a real model."""
    seen = {}

    def stub(model_name, pretrained=None, **kwargs):
        seen["model_name"] = model_name
        seen["pretrained"] = pretrained
        return (_FakeModel(), None, "preprocess")

    monkeypatch.setattr("open_clip.create_model_and_transforms", stub)
    return seen


# ── resolution: cheap, never raises, describes what happened ──────────────────

def test_resolve_reports_the_tag_when_no_local_weights_configured(monkeypatch):
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", "")

    res = clip_embed.resolve_weights()

    assert res.ok is True
    assert res.source == "tag"
    assert res.pretrained == config.CLIP_PRETRAINED


def test_resolve_accepts_a_checkpoint_file(monkeypatch, tmp_path):
    weights = tmp_path / "open_clip_pytorch_model.bin"
    weights.write_bytes(b"not a real checkpoint, just has to exist")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", str(weights))

    res = clip_embed.resolve_weights()

    assert res.ok is True
    assert res.source == "env"
    assert res.pretrained == str(weights)


def test_resolve_accepts_a_directory_containing_the_checkpoint(monkeypatch, tmp_path):
    """A directory is as natural a thing for an admin to point at as a file - and the
    bundled-weights work (plan 1.2) installs a directory, so both share one resolver."""
    weights = tmp_path / "open_clip_pytorch_model.bin"
    weights.write_bytes(b"checkpoint")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", str(tmp_path))

    res = clip_embed.resolve_weights()

    assert res.ok is True
    assert res.pretrained == str(weights)


def test_resolve_reports_a_missing_path_without_raising(monkeypatch, tmp_path):
    missing = tmp_path / "nope" / "open_clip_pytorch_model.bin"
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", str(missing))

    res = clip_embed.resolve_weights()  # must not raise

    assert res.ok is False
    assert str(missing) in res.reason
    assert res.remedy, "an operator needs to be told what to do about it"


def test_resolve_does_not_tell_the_operator_to_use_the_network(monkeypatch, tmp_path):
    """Defect 03: huggingface.co is blocked on the target network. Advising 'unset it and
    we will download instead' is advice that cannot work there."""
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", str(tmp_path / "missing.bin"))

    res = clip_embed.resolve_weights()

    lowered = (res.remedy + " " + res.reason).lower()
    assert "download" not in lowered
    assert "network" not in lowered
    assert "huggingface" not in lowered


def test_resolve_reports_an_empty_directory_as_not_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", str(tmp_path))

    res = clip_embed.resolve_weights()

    assert res.ok is False
    assert res.remedy


# ── loading: consumes the resolution ─────────────────────────────────────────

def test_configured_weights_are_passed_through_to_open_clip(monkeypatch, tmp_path, captured):
    weights = tmp_path / "open_clip_pytorch_model.bin"
    weights.write_bytes(b"checkpoint")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", str(weights))

    clip_embed._load_model()

    assert captured["pretrained"] == str(weights)
    assert captured["model_name"] == config.CLIP_MODEL_NAME


def test_without_configured_weights_the_pretrained_tag_is_used(monkeypatch, captured):
    """No regression: unset means the existing tag-based resolution, exactly as before."""
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", "")

    clip_embed._load_model()

    assert captured["pretrained"] == config.CLIP_PRETRAINED


def test_embedding_with_bad_weights_config_raises_rather_than_fetching(monkeypatch, tmp_path, captured):
    """Someone is actually trying to embed an image and there is no way to do it - that has
    to surface, not silently fall back to the network fetch the setting exists to avoid."""
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", str(tmp_path / "missing.bin"))

    with pytest.raises(RuntimeError) as exc:
        clip_embed._load_model()

    assert "CLIP_WEIGHTS_PATH" in str(exc.value)
    assert "pretrained" not in captured, "must not reach open_clip when the config is bad"


# ── startup: records, never kills the server ─────────────────────────────────

def test_warm_up_does_not_raise_when_weights_are_misconfigured(monkeypatch, tmp_path, captured):
    """Text search, upload and evaluation do not need CLIP. A bad vision config must not
    take the whole server down at boot - it must be visible, not fatal."""
    monkeypatch.setattr(config, "CLIP_PROVIDER", "openclip")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", str(tmp_path / "missing.bin"))

    clip_embed.warm_up()  # must not raise

    assert clip_embed.weights_status().ok is False


def test_warm_up_records_a_good_resolution(monkeypatch, tmp_path, captured):
    weights = tmp_path / "open_clip_pytorch_model.bin"
    weights.write_bytes(b"checkpoint")
    monkeypatch.setattr(config, "CLIP_PROVIDER", "openclip")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", str(weights))

    clip_embed.warm_up()

    status = clip_embed.weights_status()
    assert status.ok is True
    assert status.source == "env"


def test_warm_up_skips_loading_when_resolution_failed(monkeypatch, tmp_path, captured):
    monkeypatch.setattr(config, "CLIP_PROVIDER", "openclip")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", str(tmp_path / "missing.bin"))

    clip_embed.warm_up()

    assert "pretrained" not in captured, "no point attempting a load that cannot succeed"


def test_weights_status_for_the_hash_provider_is_not_an_error(monkeypatch):
    """The offline test provider needs no weights at all - it should not report as broken."""
    monkeypatch.setattr(config, "CLIP_PROVIDER", "hash")

    clip_embed.warm_up()

    assert clip_embed.weights_status().ok is True


# ── env var plumbing (adversarial review: config.py:49 had zero coverage) ─────

def _reload_config(monkeypatch, **env):
    """config reads env at import time, so precedence has to be tested via reimport."""
    import importlib
    for var in ("CAMPAIGN_POC_CLIP_WEIGHTS_PATH", "CLIP_WEIGHTS_PATH"):
        monkeypatch.delenv(var, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import config as config_module
    return importlib.reload(config_module)


def test_prefixed_env_var_is_authoritative(monkeypatch):
    """Every other setting in config.py uses the CAMPAIGN_POC_ prefix; when both are set the
    prefixed one must win, and the error must name the variable that was actually used."""
    reloaded = _reload_config(monkeypatch,
                              CAMPAIGN_POC_CLIP_WEIGHTS_PATH="/prefixed/weights.bin",
                              CLIP_WEIGHTS_PATH="/bare/weights.bin")
    try:
        assert reloaded.CLIP_WEIGHTS_PATH == "/prefixed/weights.bin"
        assert reloaded.CLIP_WEIGHTS_ENV_VAR == "CAMPAIGN_POC_CLIP_WEIGHTS_PATH"
    finally:
        _reload_config(monkeypatch)


def test_bare_env_var_is_honoured_when_it_is_the_only_one_set(monkeypatch):
    """It is the name an admin would guess, and no library reads it."""
    reloaded = _reload_config(monkeypatch, CLIP_WEIGHTS_PATH="/bare/weights.bin")
    try:
        assert reloaded.CLIP_WEIGHTS_PATH == "/bare/weights.bin"
        assert reloaded.CLIP_WEIGHTS_ENV_VAR == "CLIP_WEIGHTS_PATH"
    finally:
        _reload_config(monkeypatch)


def test_error_names_whichever_variable_the_admin_actually_set(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", str(tmp_path / "missing.bin"))
    monkeypatch.setattr(config, "CLIP_WEIGHTS_ENV_VAR", "CLIP_WEIGHTS_PATH")

    res = clip_embed.resolve_weights()

    assert "CLIP_WEIGHTS_PATH" in res.reason
    assert "CAMPAIGN_POC_CLIP_WEIGHTS_PATH" not in res.reason


def test_home_relative_path_is_expanded(monkeypatch, tmp_path):
    """~/weights.bin is a thing an admin will type; Path.is_file() on a literal '~' is False."""
    weights = tmp_path / "open_clip_pytorch_model.bin"
    weights.write_bytes(b"checkpoint")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", "~/open_clip_pytorch_model.bin")

    res = clip_embed.resolve_weights()

    assert res.ok is True
    assert res.pretrained == str(weights)


def test_healthz_reports_the_weights_status(monkeypatch, tmp_path):
    """Defect 06 in miniature: establishing that visual search was off took a 60s timeout
    and a read of the source. The status has to be askable."""
    from starlette.testclient import TestClient
    import http_app

    monkeypatch.setattr(config, "CLIP_PROVIDER", "openclip")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", str(tmp_path / "missing.bin"))
    clip_embed._resolution = None
    clip_embed._load_error = None

    body = TestClient(http_app.app).get("/healthz").json()

    # /healthz now returns the same component report as the health_check tool and CLI, so
    # three surfaces cannot disagree about one machine (item 2.3).
    assert body["components"]["visual_search"]["ok"] is False
    assert body["components"]["visual_search"]["remedy"]
    assert body["status"] == "degraded"


# ── load failures, not just resolution failures (verification-round findings) ─

def test_warm_up_survives_a_load_failure(monkeypatch):
    """Resolution succeeding says nothing about the load succeeding. The tag path on a
    blocked network resolves fine and then raises inside open_clip - and since the installed
    binary now warms up at boot, an unguarded load would kill the whole server exactly where
    the product review was run. Text tools don't need CLIP; they must survive this."""
    monkeypatch.setattr(config, "CLIP_PROVIDER", "openclip")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", "")

    def boom(*a, **k):
        raise RuntimeError("no route to huggingface.co")
    monkeypatch.setattr("open_clip.create_model_and_transforms", boom)

    clip_embed.warm_up()  # must not raise

    assert clip_embed.weights_status().ok is False


def test_status_reflects_a_load_failure_not_just_resolution(monkeypatch):
    """/healthz must not report the weights as fine when the model never loaded."""
    monkeypatch.setattr(config, "CLIP_PROVIDER", "openclip")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", "")
    monkeypatch.setattr("open_clip.create_model_and_transforms",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("corrupt checkpoint")))

    clip_embed.warm_up()

    status = clip_embed.weights_status()
    assert status.ok is False
    assert "corrupt checkpoint" in status.reason
    assert status.remedy


def test_a_failed_status_is_rechecked_rather_than_cached_forever(monkeypatch, tmp_path, captured):
    """An operator who drops the missing file into place shouldn't have to guess that a
    restart is required - and shouldn't be told the file is absent while it sits there."""
    weights = tmp_path / "open_clip_model.safetensors"
    monkeypatch.setattr(config, "CLIP_PROVIDER", "openclip")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", str(weights))

    clip_embed.warm_up()
    assert clip_embed.weights_status().ok is False

    weights.write_bytes(b"checkpoint")  # operator fixes it, server still running

    assert clip_embed.weights_status().ok is True
    clip_embed._load_model()
    assert captured["pretrained"] == str(weights)
