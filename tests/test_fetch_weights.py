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

    def fake_download(url, dest, expected_size=0):
        downloaded["url"] = url
        dest.write_bytes(good)
    monkeypatch.setattr(fetch_weights, "_download", fake_download)

    result = fetch_weights.ensure_weights(tmp_path, sha256=digest)

    assert result == target
    assert target.read_bytes() == good
    assert downloaded["url"], "a corrupt copy should trigger a real fetch"


def test_a_bad_download_raises_rather_than_shipping(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_weights, "_download",
                        lambda url, dest, size=0: dest.write_bytes(b"<html>blocked</html>"))

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

    # A transport failure surfaces as WeightsError so ensure_weights can try the next
    # source; the raw OSError escaping is what used to kill a build on a stalled stream.
    with pytest.raises(fetch_weights.WeightsError):
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
        fetch_weights._download("https://example.invalid/w",
                                tmp_path / "open_clip_model.safetensors",
                                expected_size=605_143_284)

    assert "1274" in str(exc.value)


def test_a_sidecar_hash_is_written_for_the_installer_to_check(tmp_path, monkeypatch):
    good = b"pretend checkpoint"
    digest = hashlib.sha256(good).hexdigest()
    monkeypatch.setattr(fetch_weights, "_download",
                        lambda url, dest, size=0: dest.write_bytes(good))

    fetch_weights.ensure_weights(tmp_path, sha256=digest)

    sidecar = tmp_path / "open_clip_model.safetensors.sha256"
    assert sidecar.exists()
    assert sidecar.read_text().split()[0] == digest
    assert sidecar.read_text().split()[1] == "open_clip_model.safetensors"


# ── mirror-first sourcing (plan 1.2b) ────────────────────────────────────────

def test_sources_are_tried_in_order_with_their_own_hashes(tmp_path, monkeypatch):
    """Two different files (our fp16 mirror, upstream's fp32), so each source carries its
    own expected hash - a single pinned hash could not describe both."""
    assert len(fetch_weights.SOURCES) >= 2
    for source in fetch_weights.SOURCES:
        assert source.url.startswith("https://")
        assert len(source.sha256) == 64
    assert "github.com" in fetch_weights.SOURCES[0].url, "our mirror should be tried first"
    assert "huggingface" in fetch_weights.SOURCES[-1].url, "upstream stays as a fallback"


def test_falls_through_to_the_next_source_when_the_first_is_unreachable(tmp_path, monkeypatch):
    """A deleted release or a blocked host must not fail the build while another source
    still works."""
    good = b"second source payload"
    digest = hashlib.sha256(good).hexdigest()
    attempted = []

    def fake_download(url, dest, expected_size=0):
        attempted.append(url)
        if "github.com" in url:
            raise fetch_weights.WeightsError("mirror unreachable")
        dest.write_bytes(good)

    monkeypatch.setattr(fetch_weights, "_download", fake_download)
    monkeypatch.setattr(fetch_weights, "SOURCES", [
        fetch_weights.Source("https://github.com/x/y", "0" * 64),
        fetch_weights.Source("https://huggingface.co/x/y", digest),
    ])

    result = fetch_weights.ensure_weights(tmp_path)

    assert result.read_bytes() == good
    assert len(attempted) == 2, "should have tried the mirror before upstream"


def test_an_existing_copy_matching_any_source_is_accepted(tmp_path, monkeypatch):
    """The bundle may already hold either the fp16 mirror or an fp32 upstream copy."""
    payload = b"upstream copy"
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / "open_clip_model.safetensors").write_bytes(payload)
    monkeypatch.setattr(fetch_weights, "SOURCES", [
        fetch_weights.Source("https://github.com/x/y", "0" * 64),
        fetch_weights.Source("https://huggingface.co/x/y", digest),
    ])
    monkeypatch.setattr(fetch_weights, "_download",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not download")))

    assert fetch_weights.ensure_weights(tmp_path).read_bytes() == payload


def test_all_sources_failing_raises_with_every_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_weights, "_download",
                        lambda url, dest, size=0: (_ for _ in ()).throw(
                            fetch_weights.WeightsError(f"nope: {url}")))

    with pytest.raises(fetch_weights.WeightsError) as exc:
        fetch_weights.ensure_weights(tmp_path)

    assert "github.com" in str(exc.value) and "huggingface" in str(exc.value)


def test_a_mid_transfer_failure_falls_through_to_the_next_source(tmp_path, monkeypatch):
    """The most likely mirror failure on a 300MB transfer is a stall or a cut stream, not a
    clean connect error - and those raise TimeoutError/IncompleteRead/ConnectionResetError,
    none of which are URLError. Only converting URLError meant the single most probable
    failure killed the build instead of degrading to upstream. Exercises the REAL _download,
    not a stub, because a stub is exactly what hid this."""
    good = b"upstream payload"
    digest = hashlib.sha256(good).hexdigest()
    attempted = []

    class _StallsMidStream:
        headers = {}

        def read(self, _size):
            raise TimeoutError("the read timed out")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Works:
        headers = {}

        def __init__(self):
            self._chunks = [good]

        def read(self, _size):
            return self._chunks.pop(0) if self._chunks else b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(url, timeout=None):
        attempted.append(url)
        return _StallsMidStream() if "github.com" in url else _Works()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(fetch_weights, "SOURCES", [
        fetch_weights.Source("https://github.com/x/y", "0" * 64, note="mirror"),
        fetch_weights.Source("https://huggingface.co/x/y", digest, note="upstream"),
    ])

    result = fetch_weights.ensure_weights(tmp_path)

    assert result.read_bytes() == good
    assert len(attempted) == 2, "a stalled mirror must degrade to upstream, not fail the build"


def test_a_failed_verify_leaves_no_misleading_file_or_sidecar(tmp_path, monkeypatch):
    """A wrong file left next to a sidecar describing a DIFFERENT file is how someone
    inspecting the cache directory gets misled about what is about to ship."""
    monkeypatch.setattr(fetch_weights, "SOURCES", [
        fetch_weights.Source("https://example.invalid/a", "0" * 64, note="one"),
    ])
    monkeypatch.setattr(fetch_weights, "_download",
                        lambda url, dest, size=0: dest.write_bytes(b"wrong content"))

    with pytest.raises(fetch_weights.WeightsError):
        fetch_weights.ensure_weights(tmp_path)

    assert list(tmp_path.iterdir()) == [], "nothing unverified should survive"


def test_a_stale_sidecar_is_removed_with_the_file_it_described(tmp_path, monkeypatch):
    (tmp_path / "open_clip_model.safetensors").write_bytes(b"stale junk")
    (tmp_path / "open_clip_model.safetensors.sha256").write_text("deadbeef  open_clip_model.safetensors\n")
    good = b"fresh payload"
    digest = hashlib.sha256(good).hexdigest()
    monkeypatch.setattr(fetch_weights, "SOURCES",
                        [fetch_weights.Source("https://example.invalid/a", digest)])
    monkeypatch.setattr(fetch_weights, "_download",
                        lambda url, dest, size=0: dest.write_bytes(good))

    fetch_weights.ensure_weights(tmp_path)

    assert (tmp_path / "open_clip_model.safetensors.sha256").read_text().split()[0] == digest


def test_require_mirror_refuses_to_fall_back(tmp_path, monkeypatch):
    """A release build must ship a KNOWN artifact. Silently falling back to upstream would
    change which model ships - 300MB larger, different vectors - with nothing going red."""
    monkeypatch.setattr(fetch_weights, "_download",
                        lambda url, dest, size=0: (_ for _ in ()).throw(
                            fetch_weights.WeightsError("mirror gone")))

    with pytest.raises(fetch_weights.WeightsError) as exc:
        fetch_weights.ensure_weights(tmp_path, require_mirror=True)

    assert "huggingface" not in str(exc.value), "must not have tried upstream"
