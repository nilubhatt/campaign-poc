"""
Issue #23, part 1: `check-weights` printed `ok: True` for a weights file truncated to 1,000
bytes, because it only resolved the path. README and the release build both use it as the
weights check, so a damaged payload passed it while `health-check` on the same file said
`FAIL visual_search`. It now compares the file with the `.sha256` shipped beside it and loads
it, and exits 1 if either fails.
"""
import hashlib
import sys

import pytest

import clip_embed
import config
import main

NAME = "open_clip_model.safetensors"


@pytest.fixture
def weights(tmp_path, monkeypatch):
    """A weights file the env var points at, with module state reset around each test."""
    path = tmp_path / NAME
    path.write_bytes(b"not really a checkpoint, but bytes to hash" * 100)
    monkeypatch.setattr(config, "CLIP_PROVIDER", "openclip")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", str(path))
    monkeypatch.setattr(clip_embed, "_resolution", None)
    monkeypatch.setattr(clip_embed, "_load_error", None)
    monkeypatch.setattr(clip_embed, "_model", None)
    monkeypatch.setattr(sys, "argv", ["campaign-intelligence", "check-weights"])
    return path


def _sidecar(path, digest):
    path.with_name(NAME + ".sha256").write_text(f"{digest}  {NAME}\n", encoding="utf-8")


def _loads(monkeypatch):
    loaded = []
    monkeypatch.setattr(clip_embed, "_load_model", lambda: loaded.append(1) or ("model", "pre"))
    return loaded


def _fails_to_load(monkeypatch, message="Error while deserializing header: invalid header length"):
    def boom():
        raise RuntimeError(message)
    monkeypatch.setattr(clip_embed, "_load_model", boom)


def test_a_file_that_does_not_match_its_checksum_fails(weights, monkeypatch, capsys):
    _sidecar(weights, "0" * 64)
    loaded = _loads(monkeypatch)

    assert main.main() == 1

    out = capsys.readouterr().out
    assert "ok:     False" in out
    assert "does not match" in out
    # A mismatch is decided by the hash alone; loading a file already known to be wrong
    # would only add seconds.
    assert loaded == []


def test_a_matching_file_that_will_not_load_fails(weights, monkeypatch, capsys):
    _sidecar(weights, hashlib.sha256(weights.read_bytes()).hexdigest())
    _fails_to_load(monkeypatch)

    assert main.main() == 1

    out = capsys.readouterr().out
    assert "ok:     False" in out
    assert "invalid header length" in out


def test_without_a_checksum_file_a_damaged_file_still_fails(weights, monkeypatch, capsys):
    # The case the issue reproduced: a file named by the env var, no .sha256 beside it.
    _fails_to_load(monkeypatch)

    assert main.main() == 1

    out = capsys.readouterr().out
    assert "checksum: no .sha256 file beside it" in out
    assert "ok:     False" in out


def test_a_matching_file_that_loads_passes(weights, monkeypatch, capsys):
    _sidecar(weights, hashlib.sha256(weights.read_bytes()).hexdigest().upper())
    loaded = _loads(monkeypatch)

    assert main.main() in (None, 0)

    out = capsys.readouterr().out
    assert "checksum: matches" in out
    assert "ok:     True" in out
    assert loaded == [1]


def test_an_empty_checksum_file_is_a_mismatch_not_a_crash(weights, monkeypatch, capsys):
    weights.with_name(NAME + ".sha256").write_text("", encoding="utf-8")
    _loads(monkeypatch)

    assert main.main() == 1
    assert "does not match" in capsys.readouterr().out


def test_an_unresolved_path_still_fails_without_hashing_or_loading(tmp_path, weights, monkeypatch, capsys):
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", str(tmp_path / "missing" / NAME))
    loaded = _loads(monkeypatch)

    assert main.main() == 1

    out = capsys.readouterr().out
    assert "ok:     False" in out
    assert loaded == []
