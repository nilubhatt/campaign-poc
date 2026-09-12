"""
Product review defect 01 (P0): "CLIP weights are fetched at first use, not shipped."

The installer placed a 56.8MB executable and declared success; the ~600MB the vision
feature actually needs was never fetched, and on that host never could be (defect 03).
The acceptance criterion is "zero outbound network required" on a clean machine - which
means the default path, with NOTHING configured, has to work offline. So the weights ship
in the payload and the app finds them on its own.

Where: beside the executable, OUTSIDE PyInstaller's _internal/. Resolved from
sys.executable when frozen, the source tree otherwise - never __file__ in a frozen app,
where it points into the bundle rather than the install directory.
"""
import sys
from pathlib import Path

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


@pytest.fixture(autouse=True)
def openclip_provider(monkeypatch):
    monkeypatch.setattr(config, "CLIP_PROVIDER", "openclip")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", "")


def _bundle(tmp_path, name="open_clip_model.safetensors"):
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    (models / name).write_bytes(b"checkpoint")
    return models / name


# ── where the app looks ──────────────────────────────────────────────────────

def test_app_dir_is_next_to_the_executable_when_frozen(monkeypatch, tmp_path):
    """PyInstaller onedir: <app>/campaign-intelligence.exe + <app>/_internal/. Data shipped
    beside the exe lands in <app>/, which is what the installers copy into Program Files."""
    fake_exe = tmp_path / "campaign-intelligence"
    fake_exe.write_bytes(b"")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe))

    assert config.app_dir() == tmp_path


def test_app_dir_is_the_source_tree_when_not_frozen(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)

    assert config.app_dir() == Path(config.__file__).resolve().parent


def test_app_dir_does_not_point_inside_pyinstallers_internal(monkeypatch, tmp_path):
    """_MEIPASS is _internal/; shipping 600MB of weights in there would put them inside the
    bundle rather than beside it, where an admin cannot see or replace them."""
    fake_exe = tmp_path / "campaign-intelligence"
    fake_exe.write_bytes(b"")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe))
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "_internal"), raising=False)

    assert "_internal" not in str(config.app_dir())


# ── resolution order: env, then bundled, then the network tag ────────────────

def test_bundled_weights_are_found_with_nothing_configured(monkeypatch, tmp_path):
    """The whole point: a clean install works offline without the admin setting anything."""
    weights = _bundle(tmp_path)
    monkeypatch.setattr(config, "BUNDLED_WEIGHTS_DIR", tmp_path / "models")

    res = clip_embed.resolve_weights()

    assert res.ok is True
    assert res.source == "bundled"
    assert res.pretrained == str(weights)


def test_explicit_setting_beats_the_bundled_copy(monkeypatch, tmp_path):
    """An admin who supplied weights out of band means it."""
    _bundle(tmp_path)
    override = tmp_path / "override" / "open_clip_pytorch_model.bin"
    override.parent.mkdir()
    override.write_bytes(b"checkpoint")
    monkeypatch.setattr(config, "BUNDLED_WEIGHTS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", str(override))

    res = clip_embed.resolve_weights()

    assert res.source == "env"
    assert res.pretrained == str(override)


def test_falls_back_to_the_tag_when_nothing_is_bundled(monkeypatch, tmp_path):
    """Source checkouts and anyone with Hub access keep working exactly as before. Scoped
    to a checkout deliberately: for an INSTALLED copy this same situation is an error, not
    a fallback (see test_installed_copy_with_no_bundled_weights_reports_missing)."""
    monkeypatch.setattr(config, "BUNDLED_WEIGHTS_DIR", tmp_path / "models")  # absent
    monkeypatch.setattr(config, "is_installed", lambda: False)

    res = clip_embed.resolve_weights()

    assert res.ok is True
    assert res.source == "tag"
    assert res.pretrained == config.CLIP_PRETRAINED


def test_a_broken_explicit_setting_does_not_silently_use_the_bundled_copy(monkeypatch, tmp_path):
    """Falling back would hide the admin's mistake and make a typo look like it worked."""
    _bundle(tmp_path)
    monkeypatch.setattr(config, "BUNDLED_WEIGHTS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", str(tmp_path / "typo.bin"))

    res = clip_embed.resolve_weights()

    assert res.ok is False
    assert "typo.bin" in res.reason


def test_bundled_safetensors_is_preferred_over_the_bin(monkeypatch, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "open_clip_pytorch_model.bin").write_bytes(b"old")
    (models / "open_clip_model.safetensors").write_bytes(b"new")
    monkeypatch.setattr(config, "BUNDLED_WEIGHTS_DIR", models)

    res = clip_embed.resolve_weights()

    assert res.pretrained.endswith("open_clip_model.safetensors")


def test_the_remedy_mentions_reinstalling_once_weights_are_bundled(monkeypatch, tmp_path):
    """Now that a bundled copy exists, 'reinstall' is real advice rather than a dead end -
    which is why item 1.1 deliberately left it out."""
    monkeypatch.setattr(config, "BUNDLED_WEIGHTS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", str(tmp_path / "missing.bin"))

    res = clip_embed.resolve_weights()

    assert "reinstall" in res.remedy.lower()


# ── the packaging paths must carry the folder, on every platform ─────────────

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_windows_installer_copies_subdirectories():
    """Inno Setup only carries <app>/models/ if the [Files] entry recurses. Without this
    flag the weights are silently dropped and the installer still reports success - the
    exact class of failure the review opened with."""
    iss = (_repo_root() / "installer/windows/campaign-intelligence.iss").read_text()
    files_line = next(line for line in iss.splitlines()
                      if line.strip().startswith("Source:") and "dist" in line)
    assert "recursesubdirs" in files_line, files_line


def test_linux_installer_copies_the_whole_bundle():
    sh = (_repo_root() / "installer/linux/install.sh").read_text()
    assert 'cp -a "$BUNDLE"/. "$DEST"/' in sh, "install.sh must copy the bundle recursively"


def test_every_build_path_stages_the_weights():
    """CI, build.sh and build.ps1 must agree - a bundle built locally should match what
    ships, or 'works on my machine' means something different from 'works installed'."""
    root = _repo_root()
    for name in ("build.sh", "build.ps1", ".github/workflows/build.yml"):
        text = (root / name).read_text()
        assert "fetch_weights.py" in text, f"{name} does not stage the CLIP weights"


# ── an installed copy must never fall back to the network ────────────────────

def test_installed_copy_with_no_bundled_weights_reports_missing(monkeypatch, tmp_path):
    """The plan's own rule: "never silently fall back to a network fetch". If an installed
    copy has lost its models/ folder - partial copy, AV quarantine, an admin tidying up -
    resolving the tag would reproduce defects 01 and 03 exactly: a TLS error or a 60s hang
    inside a tool call, on the machine that cannot reach the Hub at all."""
    monkeypatch.setattr(config, "BUNDLED_WEIGHTS_DIR", tmp_path / "models")  # absent
    monkeypatch.setattr(config, "is_installed", lambda: True)

    res = clip_embed.resolve_weights()

    assert res.ok is False
    assert res.source == "missing"
    assert "reinstall" in res.remedy.lower()
    assert str(tmp_path / "models") in res.reason


def test_source_checkout_with_no_bundled_weights_still_uses_the_tag(monkeypatch, tmp_path):
    """Developers have no bundled copy and Hub access is normal for them - this must not
    become an error for a source checkout."""
    monkeypatch.setattr(config, "BUNDLED_WEIGHTS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "is_installed", lambda: False)

    res = clip_embed.resolve_weights()

    assert res.ok is True
    assert res.source == "tag"


def test_installed_copy_with_bundled_weights_is_fine(monkeypatch, tmp_path):
    _bundle(tmp_path)
    monkeypatch.setattr(config, "BUNDLED_WEIGHTS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "is_installed", lambda: True)

    res = clip_embed.resolve_weights()

    assert res.ok is True
    assert res.source == "bundled"


def test_bundled_dir_is_absolute_and_derived_from_the_app_dir():
    """Every other test overrides BUNDLED_WEIGHTS_DIR, so nothing pinned how it is built.
    A relative Path("models") would pass all of them and then break in the field: Claude
    Desktop launches the server with an arbitrary working directory."""
    assert config.BUNDLED_WEIGHTS_DIR.is_absolute()
    assert config.BUNDLED_WEIGHTS_DIR == config.app_dir() / "models"


def test_app_dir_resolves_through_a_symlinked_launcher(monkeypatch, tmp_path):
    """installer/linux/install.sh puts a symlink on PATH pointing into the install dir. If
    sys.executable is reported as that symlink, the weights must still be found."""
    install_dir = tmp_path / "install"
    install_dir.mkdir()
    real_exe = install_dir / "campaign-intelligence"
    real_exe.write_bytes(b"")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    link = bin_dir / "campaign-intelligence"
    link.symlink_to(real_exe)

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(link))

    assert config.app_dir() == install_dir
