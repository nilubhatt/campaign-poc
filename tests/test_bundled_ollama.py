"""
The embedder ships inside the bundle. The customer's network blocks ollama.com and the model
registry, so the installers' "install Ollama, then `ollama pull nomic-embed-text`" could
never complete there and text search, uploads and evaluations were dead on arrival.

These pin the product half (where the bundled runtime is, when it is used, how it is started,
reused, replaced and stopped, and that loopback calls ignore a corporate HTTP_PROXY), the
build half (scripts/fetch_ollama.py), and the installers and CI that depend on both.
"""
import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import httpx
import pytest

import bundled_ollama
import config
import core
import embedding

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_ollama  # noqa: E402


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _lay_out_bundle(app: Path) -> Path:
    """An install directory carrying a runtime and a model store, in this platform's layout."""
    binary = config.bundled_ollama_binary()
    binary.parent.mkdir(parents=True, exist_ok=True)
    (app / "ollama" / "models" / "manifests").mkdir(parents=True, exist_ok=True)
    return binary


@pytest.fixture
def app(tmp_path, monkeypatch):
    """A fake install directory, with the data directory beside it."""
    app = tmp_path / "app"
    app.mkdir()
    monkeypatch.setattr(config, "app_dir", lambda: app)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    return app


# ── where it is and when it is used ──────────────────────────────────────────────────────

def test_the_binary_is_where_each_platforms_release_puts_it(app, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert config.bundled_ollama_binary() == app / "ollama" / "bin" / "ollama"
    monkeypatch.setattr(sys, "platform", "darwin")
    assert config.bundled_ollama_binary() == app / "ollama" / "ollama"
    # Each name matches the executable fetch_ollama stages for that platform.
    assert fetch_ollama.ASSETS["linux"].binary == "bin/ollama"
    assert fetch_ollama.ASSETS["darwin"].binary == "ollama"
    assert fetch_ollama.ASSETS["win32"].binary == "ollama.exe"


def test_a_bundle_with_its_own_ollama_uses_it_on_a_private_port(app, monkeypatch):
    assert config._default_ollama_url() == "http://localhost:11434"
    _lay_out_bundle(app).write_text("")
    assert config._default_ollama_url() == "http://127.0.0.1:11435"
    # Never the user's port: their Ollama, their models, their version.
    assert config.BUNDLED_OLLAMA_PORT != 11434


def test_ollama_url_set_by_the_user_still_wins(app, monkeypatch):
    _lay_out_bundle(app).write_text("")
    monkeypatch.setenv("OLLAMA_URL", "http://gpu-box:11434")
    assert config._default_ollama_url() == "http://gpu-box:11434"


def test_a_runtime_without_its_model_store_is_not_used(app):
    binary = _lay_out_bundle(app)
    binary.write_text("")
    (app / "ollama" / "models" / "manifests").rmdir()
    assert config._default_ollama_url() == "http://localhost:11434"


def _use_bundled(app, monkeypatch):
    _lay_out_bundle(app).write_text("")
    monkeypatch.setattr(config, "EMBED_PROVIDER", "ollama")
    monkeypatch.setattr(config, "OLLAMA_URL", bundled_ollama.url())


def test_in_use_only_when_the_product_talks_to_the_bundled_one(app, monkeypatch):
    _use_bundled(app, monkeypatch)
    assert bundled_ollama.in_use()
    monkeypatch.setattr(config, "OLLAMA_URL", "http://localhost:11434")
    assert not bundled_ollama.in_use()
    monkeypatch.setattr(config, "OLLAMA_URL", bundled_ollama.url())
    monkeypatch.setattr(config, "EMBED_PROVIDER", "hash")
    assert not bundled_ollama.in_use()


def test_not_in_use_means_nothing_is_started(app, monkeypatch):
    monkeypatch.setattr(bundled_ollama, "_spawn",
                        lambda: pytest.fail("started an embedder this process does not use"))
    assert bundled_ollama.ensure_running(timeout=0.1) is True


# ── starting, reusing, replacing and stopping it ─────────────────────────────────────────

FAKE_OLLAMA = textwrap.dedent('''\
    #!{python}
    # Stands in for `ollama serve`: answers /api/version on OLLAMA_HOST and records the
    # environment it was started with.
    import json, os, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    assert sys.argv[1:] == ["serve"], sys.argv
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env.json"), "w") as fh:
        json.dump({{k: v for k, v in os.environ.items() if k.startswith("OLLAMA_")}}, fh)
    host, port = os.environ["OLLAMA_HOST"].split(":")
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'{{"version":"x"}}')
        def log_message(self, *a): pass
    HTTPServer((host, int(port)), H).serve_forever()
''')


@pytest.fixture
def fake_runtime(app, monkeypatch):
    if os.name == "nt":
        pytest.skip("the fake runtime is a script with a shebang")
    binary = _lay_out_bundle(app)
    binary.write_text(FAKE_OLLAMA.format(python=sys.executable))
    binary.chmod(0o755)
    monkeypatch.setattr(config, "EMBED_PROVIDER", "ollama")
    monkeypatch.setattr(config, "BUNDLED_OLLAMA_PORT", _free_port())
    monkeypatch.setattr(config, "OLLAMA_URL", bundled_ollama.url())
    # The real check reads the process name, which for this script is the interpreter.
    monkeypatch.setattr(bundled_ollama, "_is_ollama", _alive)
    yield binary
    bundled_ollama.stop()


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # A zombie still answers kill(0); `ps` says whether it is really running.
    state = subprocess.run(["ps", "-p", str(pid), "-o", "stat="],
                           capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith("Z")


def test_the_first_process_that_needs_it_starts_it(fake_runtime, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11434")
    monkeypatch.setenv("OLLAMA_MODELS", "/somebody/elses/models")

    assert bundled_ollama.ensure_running(timeout=15)

    assert bundled_ollama._answering()
    info = json.loads((config.DATA_DIR / "ollama.pid").read_text())
    assert info["binary"] == str(fake_runtime) and _alive(info["pid"])
    env = json.loads((fake_runtime.parent / "env.json").read_text())
    # Its own port and its own model store, never the user's, whatever they exported.
    assert env["OLLAMA_HOST"] == f"127.0.0.1:{config.BUNDLED_OLLAMA_PORT}"
    assert env["OLLAMA_MODELS"] == str(config.app_dir() / "ollama" / "models")
    assert env["OLLAMA_NOPRUNE"] == "1"
    assert (config.DATA_DIR / "ollama.log").exists()


def test_a_running_one_is_reused_not_started_twice(fake_runtime, monkeypatch):
    assert bundled_ollama.ensure_running(timeout=15)
    first = json.loads((config.DATA_DIR / "ollama.pid").read_text())["pid"]
    monkeypatch.setattr(bundled_ollama, "_spawn", lambda: pytest.fail("started a second one"))

    assert bundled_ollama.ensure_running(timeout=15)

    assert json.loads((config.DATA_DIR / "ollama.pid").read_text())["pid"] == first


def test_one_left_running_by_another_install_directory_is_replaced(fake_runtime):
    # An upgrade, or the installer's self-test from its staging directory: answering, but
    # serving from files that are about to stop existing.
    assert bundled_ollama.ensure_running(timeout=15)
    old = json.loads((config.DATA_DIR / "ollama.pid").read_text())
    (config.DATA_DIR / "ollama.pid").write_text(
        json.dumps({"pid": old["pid"], "binary": "/old/install/ollama/ollama"}))

    assert bundled_ollama.ensure_running(timeout=15)

    new = json.loads((config.DATA_DIR / "ollama.pid").read_text())
    assert new["pid"] != old["pid"] and new["binary"] == str(fake_runtime)
    for _ in range(50):
        if not _alive(old["pid"]):
            break
        time.sleep(0.1)
    assert not _alive(old["pid"]), "the previous install's embedder was left running"


def test_stop_stops_it_and_says_so(fake_runtime):
    assert bundled_ollama.ensure_running(timeout=15)
    pid = json.loads((config.DATA_DIR / "ollama.pid").read_text())["pid"]

    assert bundled_ollama.stop() is True

    assert not _alive(pid) and not bundled_ollama._answering()
    assert not (config.DATA_DIR / "ollama.pid").exists()
    assert bundled_ollama.stop() is False


def test_stop_never_kills_a_process_that_is_not_ollama(app):
    # A pid outlives nothing: after a reboot the recorded number can be anything. This
    # process is python, so the real name check must refuse it.
    config.DATA_DIR.mkdir(parents=True)
    (config.DATA_DIR / "ollama.pid").write_text(
        json.dumps({"pid": os.getpid(), "binary": "/x/ollama"}))

    assert bundled_ollama.stop() is False
    assert not (config.DATA_DIR / "ollama.pid").exists()


def test_a_runtime_that_will_not_start_is_reported_not_raised(app, monkeypatch, capsys):
    _use_bundled(app, monkeypatch)
    monkeypatch.setattr(config, "BUNDLED_OLLAMA_PORT", _free_port())
    monkeypatch.setattr(config, "OLLAMA_URL", bundled_ollama.url())
    if os.name == "nt":
        pytest.skip("execute permission is a POSIX notion")
    config.bundled_ollama_binary().write_text("not a program")
    config.bundled_ollama_binary().chmod(0o644)     # present, but cannot be executed

    assert bundled_ollama.ensure_running(timeout=0.5) is False
    assert "could not start the bundled embedder" in capsys.readouterr().err


# ── the embedder starts it when it is gone ───────────────────────────────────────────────

def test_an_embed_that_finds_it_gone_starts_it_and_retries(app, monkeypatch):
    _use_bundled(app, monkeypatch)
    calls, started = [], []

    def post(text, timeout=0):
        calls.append(timeout)
        if len(calls) == 1:
            raise httpx.ConnectError("refused")
        return [0.1] * config.EMBED_DIM

    monkeypatch.setattr(embedding, "_post_ollama", post)
    monkeypatch.setattr(bundled_ollama, "ensure_running",
                        lambda timeout=20.0: started.append(timeout) or True)

    assert len(embedding.embed("hello", timeout=10)) == config.EMBED_DIM
    assert len(calls) == 2 and started
    # Within the caller's budget, not a fresh one on top of it.
    assert started[0] <= 10 and calls[1] <= 10


def test_an_ollama_the_user_runs_is_not_started_by_the_product(app, monkeypatch):
    monkeypatch.setattr(config, "EMBED_PROVIDER", "ollama")
    monkeypatch.setattr(config, "OLLAMA_URL", "http://localhost:11434")
    monkeypatch.setattr(embedding, "_post_ollama",
                        lambda text, timeout=0: (_ for _ in ()).throw(httpx.ConnectError("x")))
    monkeypatch.setattr(bundled_ollama, "ensure_running",
                        lambda timeout=20.0: pytest.fail("tried to start an embedder"))

    with pytest.raises(embedding.Unavailable):
        embedding.embed("hello", timeout=5)


def test_a_failed_start_is_still_reported_as_unavailable(app, monkeypatch):
    _use_bundled(app, monkeypatch)
    monkeypatch.setattr(embedding, "_post_ollama",
                        lambda text, timeout=0: (_ for _ in ()).throw(httpx.ConnectError("x")))
    monkeypatch.setattr(bundled_ollama, "ensure_running", lambda timeout=20.0: False)

    with pytest.raises(embedding.Unavailable):
        embedding.embed("hello", timeout=5)


# ── a corporate HTTP_PROXY does not swallow loopback calls ───────────────────────────────

@pytest.mark.parametrize("url,local", [
    ("http://127.0.0.1:11435", True), ("http://localhost:11434", True),
    ("http://[::1]:11434", True), ("http://gpu-box:11434", False),
    ("https://ollama.example.com", False),
])
def test_loopback_is_recognised(url, local):
    assert embedding.is_loopback(url) is local


@pytest.mark.parametrize("url,trust_env", [
    ("http://127.0.0.1:11435", False), ("http://gpu-box:11434", True)])
def test_the_embed_call_ignores_proxies_only_for_loopback(monkeypatch, url, trust_env):
    seen = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"embedding": [0.0] * config.EMBED_DIM}

    monkeypatch.setattr(config, "OLLAMA_URL", url)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: seen.update(k) or _Resp())
    embedding._post_ollama("hello", 5)
    assert seen["trust_env"] is trust_env


def test_a_real_proxy_setting_does_not_reach_the_loopback_embedder(monkeypatch):
    """End to end through httpx: every proxied request goes to a closed port, and the
    loopback embedder still answers."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({"embedding": [0.5] * config.EMBED_DIM}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "all_proxy"):
            monkeypatch.setenv(var, "http://127.0.0.1:9")
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        monkeypatch.setattr(config, "OLLAMA_URL", f"http://127.0.0.1:{server.server_port}")
        assert embedding._post_ollama("hello", 5) == [0.5] * config.EMBED_DIM
    finally:
        server.shutdown()


# ── the health check's advice ────────────────────────────────────────────────────────────

def test_the_remedy_for_the_bundled_embedder_is_not_ollama_pull(app, monkeypatch):
    _use_bundled(app, monkeypatch)
    remedy = core._bundled_embedder_remedy()
    assert "ollama pull" not in remedy
    assert str(bundled_ollama.log_file()) in remedy and "reinstall" in remedy.lower()


# ── the build half: scripts/fetch_ollama.py ──────────────────────────────────────────────

@pytest.mark.parametrize("member,gpu", [
    ("lib/ollama/cuda_v12/libcublasLt.so.12", True), ("lib/ollama/cuda_v13/x.so", True),
    ("lib/ollama/vulkan/libggml-vulkan.so", True), ("mlx_metal_v4/mlx.metallib", True),
    ("lib/ollama/rocm/x.so", True), ("bin/ollama", False),
    ("lib/ollama/libggml-cpu-haswell.so", False), ("lib/ollama/llama-server", False),
    ("ollama.exe", False),
])
def test_only_gpu_back_ends_are_left_out(member, gpu):
    assert fetch_ollama.is_gpu_path(member) is gpu


def test_every_platform_is_pinned_to_a_published_checksum():
    assert set(fetch_ollama.ASSETS) == {"linux", "win32", "darwin"}
    for asset in fetch_ollama.ASSETS.values():
        assert len(asset.sha256) == 64 and int(asset.sha256, 16) >= 0
        assert fetch_ollama.OLLAMA_VERSION in ("v" + "0.40.5",)  # bump both together
    assert fetch_ollama.EMBED_DIM == config.EMBED_DIM
    assert fetch_ollama.MODEL == config.OLLAMA_EMBED_MODEL


def _models(tmp_path, blobs: dict, layers: list):
    models = tmp_path / "models"
    (models / "blobs").mkdir(parents=True)
    for content in blobs.values():
        digest = fetch_ollama.hashlib.sha256(content).hexdigest()
        (models / "blobs" / f"sha256-{digest}").write_bytes(content)
    manifest = models / "manifests" / "registry.ollama.ai" / "library" / "m" / "latest"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"config": layers[0], "layers": layers[1:]}))
    return models


def _digest(content):
    return fetch_ollama.hashlib.sha256(content).hexdigest()


def test_the_staged_model_must_be_the_pinned_one(tmp_path, monkeypatch):
    blobs = {"manifest": b"registry manifest", "weights": b"gguf weights", "cfg": b"{}"}
    layers = [{"digest": f"sha256:{_digest(c)}"} for c in blobs.values()]
    models = _models(tmp_path, blobs, layers)
    monkeypatch.setattr(fetch_ollama, "MODEL_MANIFEST_DIGEST", _digest(b"registry manifest"))
    monkeypatch.setattr(fetch_ollama, "MODEL_WEIGHTS_DIGEST", _digest(b"gguf weights"))
    fetch_ollama.verify_models(models)

    # `latest` moved: different weights, which would mean different vectors.
    monkeypatch.setattr(fetch_ollama, "MODEL_WEIGHTS_DIGEST", "0" * 64)
    with pytest.raises(SystemExit, match="not the pinned build"):
        fetch_ollama.verify_models(models)


def test_a_damaged_blob_fails_the_build(tmp_path, monkeypatch):
    blobs = {"manifest": b"registry manifest", "weights": b"gguf weights"}
    models = _models(tmp_path, blobs, [{"digest": f"sha256:{_digest(c)}"} for c in blobs.values()])
    monkeypatch.setattr(fetch_ollama, "MODEL_MANIFEST_DIGEST", _digest(b"registry manifest"))
    monkeypatch.setattr(fetch_ollama, "MODEL_WEIGHTS_DIGEST", _digest(b"gguf weights"))
    (models / "blobs" / f"sha256-{_digest(b'gguf weights')}").write_bytes(b"gguf weigh")

    with pytest.raises(SystemExit, match="hashes to"):
        fetch_ollama.verify_models(models)


def test_a_manifest_naming_a_missing_blob_fails_the_build(tmp_path, monkeypatch):
    blobs = {"manifest": b"registry manifest", "weights": b"gguf weights"}
    layers = [{"digest": f"sha256:{_digest(c)}"} for c in blobs.values()]
    layers.append({"digest": "sha256:" + "f" * 64})
    models = _models(tmp_path, blobs, layers)
    monkeypatch.setattr(fetch_ollama, "MODEL_MANIFEST_DIGEST", _digest(b"registry manifest"))
    monkeypatch.setattr(fetch_ollama, "MODEL_WEIGHTS_DIGEST", _digest(b"gguf weights"))

    with pytest.raises(SystemExit, match="not in"):
        fetch_ollama.verify_models(models)


def test_a_download_that_does_not_match_its_pin_is_refused(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.write_bytes(b"an HTML block page with a 200")
    with pytest.raises(SystemExit, match="is not the pinned"):
        fetch_ollama._download(src.as_uri(), tmp_path / "out" / "a.tgz", "0" * 64)
    assert not (tmp_path / "out" / "a.tgz").exists()
    assert not (tmp_path / "out" / "a.tgz.partial").exists()


# ── the installers ───────────────────────────────────────────────────────────────────────

MAC = ROOT / "installer" / "macos" / "install.sh"
LINUX = ROOT / "installer" / "linux" / "install.sh"


@pytest.mark.parametrize("installer", [LINUX, MAC], ids=lambda p: p.parent.name)
def test_the_unix_installers_stop_the_embedder_around_every_replacement(installer):
    text = installer.read_text(encoding="utf-8")
    copy = text.index('cp -a "$BUNDLE"/. "$STAGE"/')
    first_stop = text.index("stop-embedder")
    assert first_stop < copy, "the previous install's embedder is still running when its files go"
    assert 'for prev in "$DEST" "$STAGE"' in text

    gate = text.index('health-check --wait 60; then')
    failed = text[gate:text.index("exit 1", gate)]
    assert "stop-embedder" in failed, "a refused install leaves its embedder running"

    move = text.index('mv "$STAGE" "$DEST"')
    assert "stop-embedder" in text[gate:move].split("INSTALL FAILED")[-1], (
        "the self-test's embedder keeps serving from $STAGE after it is moved")


@pytest.mark.parametrize("installer", [LINUX, MAC], ids=lambda p: p.parent.name)
def test_a_bundle_with_its_own_embedder_downloads_nothing(installer):
    text = installer.read_text(encoding="utf-8")
    bundled = text.index('if [ -d "$STAGE/ollama/models" ]; then')
    download = text.index("ollama.com/install.sh")
    assert bundled < text.index('elif [ "$WITH_OLLAMA" = 1 ]; then') < download


@pytest.mark.parametrize("platform", ["linux", "macos"])
def test_the_unix_uninstallers_stop_the_embedder_before_deleting_it(platform):
    text = (ROOT / "installer" / platform / "uninstall.sh").read_text(encoding="utf-8")
    remove = ('rm -rf "$DEST"' if platform == "macos"
              else 'rm -rf "$HOME/.local/share/campaign-intelligence"')
    assert text.index("stop-embedder") < text.index(remove)


def test_the_windows_installer_stops_the_embedder_and_ships_its_model_uncompressed():
    iss = (ROOT / "installer" / "windows" / "campaign-intelligence.iss").read_text(encoding="utf-8")
    assert "function PrepareToInstall" in iss
    prepare = iss[iss.index("function PrepareToInstall"):]
    assert "stop-embedder" in prepare[:prepare.index("end;")]

    uninstall = iss[iss.index("\n[UninstallRun]"):iss.index("\n[Code]")]
    entries = [l for l in uninstall.splitlines() if l.startswith("Filename:")]
    assert "stop-embedder" in entries[0], "files are deleted under a running embedder"

    assert 'Excludes: "models\\*,ollama\\models\\*"' in iss
    model_line = next(l for l in iss.splitlines() if l.startswith('Source: "{#SourceDir}\\ollama\\models'))
    assert "nocompression" in model_line

    install = iss[iss.index("procedure InstallOllama"):]
    install = install[:install.index("\nend;")]
    assert install.index("ollama\\models") < install.index("OllamaSetup.exe")


def test_stop_embedder_is_a_command():
    out = subprocess.run([sys.executable, str(ROOT / "main.py"), "--help"],
                         capture_output=True, text=True, cwd=ROOT).stdout
    assert "stop-embedder" in out


# ── CI ───────────────────────────────────────────────────────────────────────────────────

def _steps():
    import yaml
    return yaml.safe_load((ROOT / ".github" / "workflows" / "build.yml")
                          .read_text(encoding="utf-8"))["jobs"]["build"]["steps"]


def test_ci_stages_the_embedder_before_packaging_on_every_platform():
    names = [s.get("name", "") for s in _steps()]
    stage = next(s for s in _steps() if "fetch_ollama.py" in s.get("run", ""))
    assert "if" not in stage, "some platform ships without the embedder"
    assert names.index(stage["name"]) > names.index("Stage weights into the bundle")
    assert names.index(stage["name"]) < names.index("Package (Linux/macOS) — bundle + install scripts")
    assert names.index(stage["name"]) < names.index("Package (Windows) — zip bundle")


def test_ci_proves_the_packaged_embedder_with_no_ollama_on_the_runner():
    checks = [s for s in _steps() if "embeds text with its own Ollama" in s.get("name", "")]
    assert {s["if"] for s in checks} == {"runner.os != 'Windows'", "runner.os == 'Windows'"}
    for step in checks:
        run = step["run"]
        assert "text_search" in run and "stop-embedder" in run
        assert "ollama" in run.lower() and ("command -v ollama" in run or "Get-Command ollama" in run)


def test_ci_offline_checks_cover_text_search_too():
    offline = [s for s in _steps() if "needs no network" in s.get("name", "")]
    assert len(offline) == 3
    for step in offline:
        assert "text_search" in step["run"], f"{step['name']} only proves the vision model"
        assert "stop-embedder" in step["run"]
    linux = next(s for s in offline if "Linux" in s["name"])
    assert "ip link set lo up" in linux["run"]
