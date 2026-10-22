"""
installer/macos/install-macos.sh: the whole Mac install in one command, so nobody copies
commands out of a guide. These run the real script against a fake bundle served over HTTP
from this process, with `uname` stubbed so they run on any machine (the CI runner is Linux).
The real bundle is exercised by the release build and was run end to end on a Mac.
"""
import hashlib
import http.server
import json
import os
import shutil
import subprocess
import tarfile
import threading
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.skipif(
    not (shutil.which("curl") and shutil.which("shasum")),
    reason="the script needs curl and shasum, which every Mac (and the CI runner) has")

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "installer" / "macos" / "install-macos.sh"
ASSET = "campaign-intelligence-macos-arm64.tar.gz"

FAKE_INSTALL = """#!/usr/bin/env bash
# Records what it was given, as the bundle's install.sh would be run.
printf '{"args": "%s", "dir": "%s"}' "$*" "$(cd "$(dirname "$0")" && pwd)" > "$RECORD"
"""


@pytest.fixture
def release(tmp_path):
    """A release directory holding a fake bundle archive and its checksum, served over HTTP."""
    served = tmp_path / "release"
    served.mkdir()
    bundle = tmp_path / "build" / "campaign-intelligence"
    bundle.mkdir(parents=True)
    (bundle / "install.sh").write_text(FAKE_INSTALL)
    (bundle / "install.sh").chmod(0o755)
    with tarfile.open(served / ASSET, "w:gz") as tar:
        tar.add(bundle, arcname="campaign-intelligence")
    digest = hashlib.sha256((served / ASSET).read_bytes()).hexdigest()
    (served / f"{ASSET}.sha256").write_text(f"{digest}  {ASSET}\n")

    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(served), **k)
    http.server.SimpleHTTPRequestHandler.log_message = lambda *a: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield served, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def _shims(tmp_path, system="Darwin", machine="arm64", hardware=None):
    shims = tmp_path / "shims"
    shims.mkdir(exist_ok=True)
    (shims / "uname").write_text(
        f'#!/bin/sh\ncase "$1" in -s) echo {system};; -m) echo {machine};; *) echo {system};; esac\n')
    (shims / "uname").chmod(0o755)
    # The hardware question the script asks first; a Rosetta shell answers uname -m wrongly.
    arm = "1" if (hardware or machine) == "arm64" else "0"
    (shims / "sysctl").write_text(f"#!/bin/sh\necho {arm}\n")
    (shims / "sysctl").chmod(0o755)
    return shims


def _run(tmp_path, base, *args, system="Darwin", machine="arm64", piped=False, script=None,
         hardware=None, extra_env=None):
    record = tmp_path / "record.json"
    env = {
        "PATH": f"{_shims(tmp_path, system, machine, hardware)}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path / "home"),
        "TMPDIR": str(tmp_path / "tmp"),
        "CAMPAIGN_INTELLIGENCE_RELEASE_URL": base,
        "RECORD": str(record),
    }
    (tmp_path / "tmp").mkdir(exist_ok=True)
    for var in ("NO_PROXY", "no_proxy"):
        env[var] = "127.0.0.1"
    env.update(extra_env or {})
    if piped:   # exactly as published: curl ... | bash
        result = subprocess.run(["bash", "-s", "--", *args], input=script or SCRIPT.read_text(),
                                capture_output=True, text=True, env=env, timeout=120)
    else:
        result = subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True,
                                env=env, timeout=120)
    return result, (json.loads(record.read_text()) if record.exists() else None)


def test_it_downloads_checks_unpacks_and_runs_the_bundles_installer(tmp_path, release):
    _, base = release
    result, ran = _run(tmp_path, base, "--no-ollama")
    assert result.returncode == 0, result.stderr
    assert ran is not None, "the bundle's install.sh never ran"
    assert ran["args"] == "--no-ollama", "arguments are not passed through"
    assert ran["dir"].endswith("/campaign-intelligence")
    # Nothing left behind: the installer copies the bundle where it belongs.
    assert list((tmp_path / "tmp").iterdir()) == []


def test_it_works_as_published_piped_into_bash(tmp_path, release):
    _, base = release
    result, ran = _run(tmp_path, base, piped=True)
    assert result.returncode == 0, result.stderr
    assert ran is not None, "piped into bash, the bundle's install.sh never ran"


def test_a_damaged_download_is_refused_and_nothing_is_installed(tmp_path, release):
    served, base = release
    with open(served / ASSET, "ab") as fh:
        fh.write(b"one byte too many")
    result, ran = _run(tmp_path, base)
    assert result.returncode != 0
    assert ran is None, "a damaged download was installed"
    assert "checksum does not match" in result.stderr
    assert list((tmp_path / "tmp").iterdir()) == []


def test_a_download_with_no_checksum_is_refused(tmp_path, release):
    served, base = release
    (served / f"{ASSET}.sha256").unlink()
    result, ran = _run(tmp_path, base)
    assert result.returncode != 0 and ran is None


def test_a_failed_download_says_so(tmp_path, release):
    served, base = release
    (served / ASSET).unlink()
    result, ran = _run(tmp_path, base)
    assert result.returncode != 0 and ran is None
    assert "download failed" in result.stderr


@pytest.mark.parametrize("system,machine,says", [
    ("Darwin", "x86_64", "Apple silicon"),
    ("Linux", "x86_64", "for macOS"),
])
def test_a_machine_it_cannot_serve_is_told_before_anything_downloads(tmp_path, release, system, machine, says):
    _, base = release
    result, ran = _run(tmp_path, base, system=system, machine=machine)
    assert result.returncode != 0 and ran is None
    assert says in result.stderr


def test_a_rosetta_terminal_on_apple_silicon_is_not_turned_away(tmp_path, release):
    # A shell running under Rosetta says x86_64; the hardware says arm64, and that decides.
    _, base = release
    result, ran = _run(tmp_path, base, machine="x86_64", hardware="arm64")
    assert result.returncode == 0 and ran is not None, result.stderr


def test_a_release_that_does_not_exist_is_named_not_blamed_on_the_network(tmp_path, release):
    served, base = release
    (served / f"{ASSET}.sha256").unlink()
    result, ran = _run(tmp_path, base, extra_env={"CAMPAIGN_INTELLIGENCE_VERSION": "0.5.0"})
    assert result.returncode != 0 and ran is None
    assert "no release 0.5.0" in result.stderr and "internet" not in result.stderr


def test_the_release_publishes_the_script_and_a_checksum_beside_every_download():
    steps = yaml.safe_load((ROOT / ".github" / "workflows" / "build.yml")
                           .read_text(encoding="utf-8"))["jobs"]["build"]["steps"]
    attach = next(s for s in steps if s.get("name") == "Attach to release")
    files = attach["with"]["files"]
    assert "install-macos.sh" in files and "*.sha256" in files
    names = [s.get("name", "") for s in steps]
    checksums = next(s for s in steps if "Checksums for the release files" in s.get("name", ""))
    assert names.index(checksums["name"]) < names.index("Attach to release")
    assert "installer/macos/install-macos.sh" in checksums["run"]
    # The name the script downloads is the name the release publishes.
    matrix = yaml.safe_load((ROOT / ".github" / "workflows" / "build.yml").read_text(
        encoding="utf-8"))["jobs"]["build"]["strategy"]["matrix"]["include"]
    assert ASSET in [m["asset"] for m in matrix]
    assert f'asset="{ASSET}"' in SCRIPT.read_text()


def test_the_script_is_all_or_nothing_when_cut_short(tmp_path, release):
    # A download cut short by the network, at any point, runs nothing.
    _, base = release
    text = SCRIPT.read_text()
    for cut in range(200, len(text) - 1, max(1, len(text) // 25)):
        result, ran = _run(tmp_path, base, piped=True, script=text[:cut])
        assert ran is None, f"a script cut at byte {cut} of {len(text)} installed something"
    result, ran = _run(tmp_path, base, piped=True, script=text)
    assert ran is not None


def test_the_digest_is_compared_whatever_file_the_checksum_line_names(tmp_path, release):
    # `shasum -c` checks the file a line NAMES; a line naming another file proved nothing
    # about the download. The script compares the digest of the file it downloaded.
    served, base = release
    other = hashlib.sha256(b"some other file").hexdigest()
    (served / f"{ASSET}.sha256").write_text(f"{other}  /etc/hosts\n")
    result, ran = _run(tmp_path, base)
    assert result.returncode != 0 and ran is None
    assert "checksum does not match" in result.stderr


def test_a_checksum_with_windows_line_endings_is_still_read(tmp_path, release):
    served, base = release
    line = (served / f"{ASSET}.sha256").read_text().strip()
    (served / f"{ASSET}.sha256").write_bytes((line + "\r\n").encode())
    result, ran = _run(tmp_path, base)
    assert result.returncode == 0 and ran is not None, result.stderr


def test_a_checksum_file_that_is_not_a_sha256_stops_before_the_download(tmp_path, release):
    served, base = release
    (served / f"{ASSET}.sha256").write_text("<html>blocked by policy</html>\n")
    result, ran = _run(tmp_path, base)
    assert result.returncode != 0 and ran is None
    assert "not a SHA-256" in result.stderr
    assert "Downloading" not in result.stdout, "800 MB fetched before the checksum was read"
