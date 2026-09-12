"""
Build-time helper for product review defect 01: the weights have to be IN the installer
payload, which means CI fetches them once and verifies them before they are shipped.

The verification is the point, not a formality. The review's own warning: a filtered
network hands back an HTML block page with a 200, and without a hash check that page gets
written to disk as if it were the model - producing an installer that passes every build
step and ships a broken product. Same failure shape as the one that started all this.
"""
import hashlib

import pytest

from scripts import fetch_weights


def _write(path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_verify_accepts_a_matching_file(tmp_path):
    target = tmp_path / "open_clip_model.safetensors"
    digest = _write(target, b"pretend checkpoint")

    fetch_weights.verify(target, digest)  # must not raise


def test_verify_rejects_a_corrupt_file(tmp_path):
    target = tmp_path / "open_clip_model.safetensors"
    _write(target, b"pretend checkpoint")

    with pytest.raises(fetch_weights.WeightsError) as exc:
        fetch_weights.verify(target, "0" * 64)

    assert "sha256" in str(exc.value).lower()


def test_verify_rejects_an_html_block_page(tmp_path):
    """The exact failure the customer's endpoint filter produces: 200 OK, HTML body."""
    target = tmp_path / "open_clip_model.safetensors"
    _write(target, b"<html><body>Access Denied by Corporate Policy</body></html>")

    with pytest.raises(fetch_weights.WeightsError) as exc:
        fetch_weights.verify(target, "a" * 64)

    assert "sha256" in str(exc.value).lower()


def test_verify_reports_a_missing_file_clearly(tmp_path):
    with pytest.raises(fetch_weights.WeightsError) as exc:
        fetch_weights.verify(tmp_path / "nope.safetensors", "a" * 64)

    assert "not" in str(exc.value).lower()


def test_already_present_and_valid_is_not_redownloaded(tmp_path, monkeypatch):
    """CI caches the payload between runs; re-fetching 600MB on every build is waste."""
    target = tmp_path / "open_clip_model.safetensors"
    digest = _write(target, b"pretend checkpoint")

    def fail(*a, **k):
        raise AssertionError("must not download when a valid copy is already present")
    monkeypatch.setattr(fetch_weights, "_download", fail)

    result = fetch_weights.ensure_weights(tmp_path, sha256=digest)

    assert result == target


def test_a_present_but_corrupt_copy_is_replaced(tmp_path, monkeypatch):
    """A half-written or block-page copy from a previous run must not be trusted."""
    target = tmp_path / "open_clip_model.safetensors"
    _write(target, b"truncated junk")
    good = b"pretend checkpoint"
    digest = hashlib.sha256(good).hexdigest()

    downloaded = {}

    def fake_download(url, dest):
        downloaded["url"] = url
        dest.write_bytes(good)
    monkeypatch.setattr(fetch_weights, "_download", fake_download)

    result = fetch_weights.ensure_weights(tmp_path, sha256=digest)

    assert result == target
    assert target.read_bytes() == good
    assert downloaded["url"], "a corrupt copy should trigger a real fetch"


def test_a_bad_download_raises_rather_than_shipping(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_weights, "_download",
                        lambda url, dest: dest.write_bytes(b"<html>blocked</html>"))

    with pytest.raises(fetch_weights.WeightsError):
        fetch_weights.ensure_weights(tmp_path, sha256="b" * 64)


def test_the_pinned_defaults_name_the_real_checkpoint():
    """The filename must be one the app actually looks for, or the bundle is invisible."""
    import clip_embed

    assert fetch_weights.CHECKPOINT_NAME in clip_embed._CHECKPOINT_NAMES
    assert len(fetch_weights.SHA256) == 64
    assert fetch_weights.URL.startswith("https://")


def test_the_download_path_writes_atomically_via_a_part_file(tmp_path, monkeypatch):
    """Every other test stubs _download, so the .part -> final rename was never executed.
    It matters: a half-written file under the real name resolves as "present" and fails
    later, at load."""
    seen = {}

    class _FakeResponse:
        def __init__(self):
            self._chunks = [b"pretend ", b"checkpoint"]
            self.headers = {}

        def read(self, _size):
            seen["part_existed_mid_transfer"] = (
                seen.get("part_existed_mid_transfer")
                or (tmp_path / "open_clip_model.safetensors.part").exists())
            return self._chunks.pop(0) if self._chunks else b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=None: _FakeResponse())

    fetch_weights._download("https://example.invalid/w", tmp_path / "open_clip_model.safetensors")

    assert (tmp_path / "open_clip_model.safetensors").read_bytes() == b"pretend checkpoint"
    assert seen["part_existed_mid_transfer"], "should stage through a .part file"
    assert not (tmp_path / "open_clip_model.safetensors.part").exists()


def test_a_failed_download_leaves_no_part_file_behind(tmp_path, monkeypatch):
    class _Boom:
        headers = {}

        def read(self, _size):
            raise OSError("connection reset")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=None: _Boom())

    with pytest.raises(OSError):
        fetch_weights._download("https://example.invalid/w", tmp_path / "open_clip_model.safetensors")

    assert list(tmp_path.iterdir()) == [], "a stale .part confuses the next run"


def test_a_wrong_content_length_fails_before_downloading_600mb(tmp_path, monkeypatch):
    """A block page announces a small body; no reason to stream it to disk and hash it."""
    class _BlockPage:
        headers = {"Content-Length": "1274"}

        def read(self, _size):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=None: _BlockPage())

    with pytest.raises(fetch_weights.WeightsError) as exc:
        fetch_weights._download("https://example.invalid/w", tmp_path / "open_clip_model.safetensors")

    assert "1274" in str(exc.value)


def test_a_sidecar_hash_is_written_for_the_installer_to_check(tmp_path, monkeypatch):
    good = b"pretend checkpoint"
    digest = hashlib.sha256(good).hexdigest()
    monkeypatch.setattr(fetch_weights, "_download", lambda url, dest: dest.write_bytes(good))

    fetch_weights.ensure_weights(tmp_path, sha256=digest)

    sidecar = tmp_path / "open_clip_model.safetensors.sha256"
    assert sidecar.exists()
    assert sidecar.read_text().split()[0] == digest
    assert sidecar.read_text().split()[1] == "open_clip_model.safetensors"
