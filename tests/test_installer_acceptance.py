"""
Phase 4 — installer acceptance criteria.

These are the criteria the product review's three P0s reduce to. The reviewer's machine had
the product installed and visual search silently dead, and the only way to find out was a
60-second timeout and a read of the server's source. An installer that reports success over
that is the defect: "install fails loudly if weights are absent or corrupt" is worth nothing
if the failure is discovered by a marketer three days later.

  4.1  the post-install self-test runs health_check and a non-green result BLOCKS the
       success screen, naming the failing component, on all three platforms
  4.2  zero egress at install, actually tested rather than "degrades gracefully"
  4.3  Ollama verified to the same standard as the vision model — daemon reachable and
       nomic-embed-text present

Tests of a shell script are tests of its text, which is weak. So the parts that can be
executed are: the CLI contract every installer depends on (exit codes, component names), and
the scripts' actual behaviour where bash can run them.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LINUX = ROOT / "installer" / "linux" / "install.sh"
MACOS = ROOT / "installer" / "macos" / "install.sh"
WINDOWS = ROOT / "installer" / "windows" / "campaign-intelligence.iss"

INSTALLERS = [p for p in (LINUX, MACOS, WINDOWS)]


def _run(*args, **kw):
    return subprocess.run([sys.executable, "main.py", *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=180, **kw)


# ── 4.1 the seam the installers call ────────────────────────────────────────

def test_the_self_test_exits_non_zero_when_a_component_is_down(tmp_path):
    """The whole mechanism. An installer can only refuse to report success if the check it
    runs says no in a way a shell can read — a printed warning is not a gate."""
    env = {**os.environ,
           "CAMPAIGN_POC_DB": str(tmp_path / "c.db"),
           "CAMPAIGN_POC_CLIP_WEIGHTS_PATH": str(tmp_path / "absent.safetensors"),
           "HF_HOME": str(tmp_path / "hf"),
           "HF_HUB_OFFLINE": "1"}

    result = _run("health-check", env=env)

    assert result.returncode != 0, result.stdout
    assert "visual_search" in result.stdout


def test_the_self_test_names_the_component_and_what_to_do(tmp_path):
    """"Naming the failing component" is the criterion. An installer that says "self-test
    failed" has moved the diagnosis to whoever reads the screen."""
    env = {**os.environ,
           "CAMPAIGN_POC_DB": str(tmp_path / "c.db"),
           "CAMPAIGN_POC_CLIP_WEIGHTS_PATH": str(tmp_path / "absent.safetensors"),
           "HF_HOME": str(tmp_path / "hf"),
           "HF_HUB_OFFLINE": "1"}

    out = _run("health-check", env=env).stdout

    assert re.search(r"FAIL\s+visual_search", out), out
    assert "->" in out or "remedy" in out.lower(), "say what would fix it"


def test_the_self_test_is_machine_readable_for_an_installer(tmp_path):
    """A shell script parsing prose is a script that breaks when the prose improves."""
    env = {**os.environ, "CAMPAIGN_POC_DB": str(tmp_path / "c.db")}

    result = _run("health-check", "--json", env=env)

    import json
    report = json.loads(result.stdout)
    assert "components" in report and "ok" in report
    assert set(report["components"]) >= {"database", "text_search", "visual_search"}


@pytest.mark.parametrize("installer", INSTALLERS, ids=lambda p: p.parent.name)
def test_every_platform_runs_the_self_test_before_claiming_success(installer):
    """All three platforms, which is the part that was outstanding: Linux verified the
    weights checksum and nothing else, Windows verified nothing, and macOS had no installer
    at all — it shipped a tarball."""
    assert installer.exists(), f"{installer} is missing"
    text = installer.read_text(encoding="utf-8")

    assert "health-check" in text, (
        f"{installer.name} never runs the self-test, so it cannot know whether what it just "
        f"installed works"
    )


@pytest.mark.parametrize("installer", [LINUX, MACOS], ids=lambda p: p.parent.name)
def test_a_failed_self_test_stops_the_unix_installers(installer):
    """`set -e` does not cover a command in an `if`, a pipeline, or one whose failure is
    swallowed by `|| true` — which is how a gate becomes a log line."""
    text = installer.read_text(encoding="utf-8")
    line = next(l for l in text.splitlines() if "health-check" in l and not
                l.strip().startswith("#"))

    assert "|| true" not in line, f"the self-test's result is discarded: {line}"


def test_the_linux_installer_fails_the_install_when_the_self_test_fails(tmp_path):
    """Executed, not read: a stub binary that exits non-zero on health-check must stop the
    script and leave a non-zero status."""
    # The installer ships inside the bundle and locates the binary relative to its own
    # path, so the test has to lay the bundle out the way a release actually does.
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    script = bundle / "install.sh"
    script.write_bytes(LINUX.read_bytes())
    stub = bundle / "campaign-intelligence"
    stub.write_text("#!/usr/bin/env bash\n"
                    'case "$1" in\n'
                    '  health-check) echo \'FAIL  visual_search: no weights\'; exit 1;;\n'
                    '  *) exit 0;;\n'
                    'esac\n')
    stub.chmod(0o755)

    result = subprocess.run(["bash", str(script), "--no-ollama"], cwd=bundle,
                            capture_output=True, text=True, timeout=120,
                            env={**os.environ, "HOME": str(tmp_path / "home")})

    assert result.returncode != 0, (
        f"the installer reported success over a failing self-test:\n{result.stdout}"
    )
    assert "visual_search" in (result.stdout + result.stderr)


def test_the_windows_installer_blocks_its_success_screen(tmp_path):
    """Inno Setup's own gate: the self-test has to run in a step that can abort the install,
    not in a `postinstall` run entry, which fires after the success page is already shown."""
    text = WINDOWS.read_text(encoding="utf-8")

    assert "[Code]" in text, "no [Code] section, so nothing can stop the install"
    body = text[text.index("[Code]"):]

    # The gate is an exception, not a message box: Inno ignores a [Run] entry's exit code,
    # and a MsgBox is dismissed and the success page shown anyway.
    assert "RaiseException" in body, "a failed self-test does not stop setup"
    self_test = body[body.index("procedure SelfTest"):]
    assert "health-check" in self_test and "RaiseException" in self_test, (
        "the self-test's result is not what raises"
    )

    # And it has to run against the finished machine. ssPostInstall precedes [Run], so a
    # self-test there would check a box before the installer had finished setting it up.
    step = body[body.index("procedure CurStepChanged"):]
    assert "ssPostInstall" in step and "SelfTest()" in step
    assert step.index("InstallOllama()") < step.index("SelfTest()"), (
        "the self-test runs before Ollama is installed, so it would fail on a machine the "
        "installer had not finished preparing"
    )

    runs = text[text.index("[Run]"):text.index("[UninstallRun]")]
    assert "health-check" not in runs, (
        "a [Run] entry cannot block anything — Inno ignores its exit code"
    )


# ── 4.3 Ollama held to the same standard as the vision model ────────────────

def test_the_self_test_checks_the_text_model_not_just_the_daemon(tmp_path):
    """"Reachable" is not "usable": a running Ollama with no nomic-embed-text pulled answers
    on the socket and 404s every embed, which is how text search was dead while everything
    looked fine. health_check already distinguishes them; this pins it."""
    import core

    report = core.health_check_cli()
    text = report["components"]["text_search"]

    assert "detail" in text
    if not text["ok"]:
        assert text["code"] in {"embedder_unreachable", "embedder_model_missing",
                                "embedder_slow"}, text


@pytest.mark.parametrize("installer", [LINUX, MACOS], ids=lambda p: p.parent.name)
def test_the_unix_installers_verify_the_model_was_actually_pulled(installer):
    """`ollama pull` failing was logged as a warning and the install carried on — the same
    "degrades gracefully into silently broken" the review was written about."""
    lines = installer.read_text(encoding="utf-8").splitlines()

    assert any("nomic-embed-text" in l for l in lines)
    pull = next(i for i, l in enumerate(lines) if "ollama pull" in l)
    gate = next(i for i, l in enumerate(lines)
                if "health-check" in l and l.strip().startswith("if !"))

    # A failed pull may be reported and carried past — what it must NOT do is end the
    # install. The self-test is what turns "the model never arrived" into a refused
    # install, so it has to run after the pull, not before it.
    assert gate > pull, (
        "the self-test runs before the model is pulled, so a failed pull is never caught"
    )


# ── 4.2 zero egress ─────────────────────────────────────────────────────────

def test_the_self_test_makes_no_outbound_connection(tmp_path):
    """The install-time criterion, tested rather than assumed. Anything the installer runs
    on a customer machine must work with egress disabled — the reviewer's network blocked
    huggingface.co, which is how all three P0s were discovered at once."""
    weights = os.getenv("CAMPAIGN_POC_TEST_WEIGHTS")
    if not weights:
        pytest.skip("needs a local weights file; CI sets CAMPAIGN_POC_TEST_WEIGHTS")

    # The guard has to be installed inside the subprocess, before anything imports a
    # network library — so it runs as the program, and calls the CLI itself. Blocking the
    # socket rather than trusting a flag is the point: "degrades gracefully" was the claim
    # the review disproved.
    runner = tmp_path / "run_offline.py"
    runner.write_text(
        "import socket, sys\n"
        "_real = socket.socket.connect\n"
        "def _blocked(self, address):\n"
        "    host = address[0] if isinstance(address, tuple) else str(address)\n"
        "    if host not in ('127.0.0.1', '::1', 'localhost'):\n"
        "        raise AssertionError('outbound connection attempted: %r' % (address,))\n"
        "    return _real(self, address)\n"
        "socket.socket.connect = _blocked\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        # health-check, not check-weights: the weaker one only stats a path, and the claim
        # being tested is that the whole install-time gate — including actually LOADING the
        # vision model, which is where open_clip reaches for the Hub — needs no network.
        "sys.argv = ['campaign-intelligence', 'health-check']\n"
        "import main\n"
        "raise SystemExit(main.main())\n"
    )
    env = {**os.environ,
           "CAMPAIGN_POC_DB": str(tmp_path / "c.db"),
           "CAMPAIGN_POC_CLIP_WEIGHTS_PATH": weights,
           "HF_HUB_OFFLINE": "1",
           "HF_HOME": str(tmp_path / "hf")}

    result = subprocess.run([sys.executable, str(runner)], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=300)

    assert "outbound connection attempted" not in (result.stdout + result.stderr), (
        result.stdout + result.stderr
    )
    assert "visual_search" in result.stdout, result.stdout + result.stderr
    assert "FAIL  visual_search" not in result.stdout, (
        "the vision model could not be loaded without a network, which is the install this "
        "product is supposed to support:\n" + result.stdout
    )


def test_the_readme_states_the_offline_install_position():
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()

    assert "offline" in readme or "egress" in readme
