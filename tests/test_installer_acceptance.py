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
import pathlib
import re
import shutil
import subprocess
import sys
import time
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

    # This test originally asserted `RaiseException`, which encoded a belief that turned out
    # to be false — Inno catches it at ssPostInstall and carries on. The mechanism that does
    # work is recording the failure and rewriting the Finished page, plus refusing to wire
    # Claude Desktop. See test_the_windows_gate_uses_a_mechanism_that_actually_stops_the_
    # success_page for the source citation.
    assert "GSelfTestFailure" in body, "a failed self-test leaves no trace to act on"
    self_test = body[body.index("procedure SelfTest"):]
    assert "health-check" in self_test and "GSelfTestFailure :=" in self_test, (
        "the self-test's result is not what is recorded"
    )
    wiring = body[body.index("procedure ConfigureDesktop"):]
    wiring = wiring[:wiring.index("\nprocedure ")] if "\nprocedure " in wiring else wiring
    assert "GSelfTestFailure" in wiring, (
        "Claude Desktop is wired regardless of whether the product works"
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

    # The first version of this only asserted inside `if not text["ok"]`, so on a machine
    # with a healthy Ollama it asserted nothing at all — and a do-nothing implementation
    # that always reported ok would have passed it.
    report = core.health_check_cli()
    assert "detail" in report["components"]["text_search"]

    # The distinction itself, forced rather than hoped for. "Reachable" is not "usable": a
    # running Ollama with nothing pulled answers on the socket and 404s every embed, which
    # is how text search was dead while everything looked fine.
    import config
    import httpx

    class _Response:
        status_code = 404
        text = "model 'nomic-embed-text' not found"

        def raise_for_status(self):
            raise httpx.HTTPStatusError("404", request=httpx.Request("POST", "http://x"),
                                        response=httpx.Response(404))

        def json(self):
            return {}

    original = httpx.post
    try:
        httpx.post = lambda *a, **k: _Response()
        missing = core.health_check_cli()
    finally:
        httpx.post = original

    component = missing["components"]["text_search"]
    assert component["ok"] is False
    assert component["code"] == "embedder_model_missing", (
        f"a reachable daemon with no model pulled must not read as healthy: {component}"
    )


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


# ══ review of Phase 4 ════════════════════════════════════════════════════════

def test_a_fresh_install_passes_its_own_gate(tmp_path):
    """The one case nobody tested. Every Phase 4 test asserted the gate REFUSES when
    something is broken; none asserted it ACCEPTS when everything works — and on a machine
    where the server has never run there is no database, `health_check_cli` opens read-only
    by design (2.3: a diagnostic must not repair what it is diagnosing), and the gate fails.

    So the first thing a marketer saw after installing was "INSTALL FAILED: database", with
    a remedy telling them to start a server."""
    data = tmp_path / "fresh"
    env = {**os.environ, "CAMPAIGN_POC_DATA": str(data),
           "CAMPAIGN_POC_DB": str(data / "campaigns.db")}

    _run("init", env=env)
    result = _run("health-check", env=env)

    assert "FAIL  database" not in result.stdout, result.stdout
    assert result.returncode == 0 or "text_search" in result.stdout, (
        f"a fresh install must not fail on an empty library:\n{result.stdout}"
    )


def test_init_creates_what_the_product_needs_and_is_safe_to_repeat(tmp_path):
    """It also proves the data location is writable by whoever the installer is running as
    — which on Windows, where an administrator may be installing for somebody else, is worth
    knowing at install time rather than at first use."""
    data = tmp_path / "fresh"
    env = {**os.environ, "CAMPAIGN_POC_DATA": str(data),
           "CAMPAIGN_POC_DB": str(data / "campaigns.db")}

    first = _run("init", env=env)
    assert first.returncode == 0, first.stdout + first.stderr
    assert (data / "campaigns.db").exists()

    second = _run("init", env=env)
    assert second.returncode == 0, "an installer that cannot be re-run is not an installer"


@pytest.mark.parametrize("installer", [LINUX, MACOS], ids=lambda p: p.parent.name)
def test_the_installer_creates_the_database_before_testing_it(installer):
    lines = installer.read_text(encoding="utf-8").splitlines()
    init = next(i for i, l in enumerate(lines) if '" init' in l or "' init" in l)
    gate = next(i for i, l in enumerate(lines)
                if "health-check" in l and l.strip().startswith("if !"))

    assert init < gate, "the self-test runs before the database it checks for exists"


@pytest.mark.parametrize("installer", [LINUX, MACOS], ids=lambda p: p.parent.name)
def test_claude_desktop_is_not_wired_to_a_product_the_gate_rejected(installer):
    """The finding that matters most. The installer certified the product broken and left
    Claude Desktop pointing at it — which is the state the product review was written about,
    now with an installer that said so and did it anyway. The self-test never reads the
    Claude config, so there was no reason the wiring came first."""
    lines = installer.read_text(encoding="utf-8").splitlines()
    gate = next(i for i, l in enumerate(lines)
                if "health-check" in l and l.strip().startswith("if !"))
    # Skip comments: a comment explaining the ordering mentions the command too, and
    # matching it made this test read the explanation as the thing being explained.
    wire = next(i for i, l in enumerate(lines)
                if "configure-desktop" in l and not l.strip().startswith("#"))

    assert gate < wire, (
        "Claude Desktop is wired before the gate, so a refused install still leaves the "
        "marketer's next session pointing at the server the installer just condemned"
    )


def test_the_windows_installer_wires_the_desktop_only_after_the_gate():
    body = WINDOWS.read_text(encoding="utf-8")
    step = body[body.index("procedure CurStepChanged"):]

    assert step.index("SelfTest()") < step.index("ConfigureDesktop()"), (
        "same defect on Windows: the connector is wired before the product is verified"
    )


def test_the_windows_installer_says_what_it_is_doing_during_a_long_download():
    """Moving Ollama out of [Run] lost its StatusMsg. The wizard now sits on a finished
    progress bar with an unchanged caption for the length of a 600MB download, and the
    user's move against an installer that looks hung is Cancel — which lands them in a
    half-installed state with Ollama's own installer partly run."""
    body = WINDOWS.read_text(encoding="utf-8")

    assert "StatusLabel" in body or "WizardForm" in body, (
        "nothing tells the user what is happening during the Ollama download"
    )


@pytest.mark.parametrize("installer", [LINUX, MACOS], ids=lambda p: p.parent.name)
def test_a_failed_install_says_what_it_did_and_did_not_do(installer):
    """"INSTALL FAILED" over a machine that has had files copied, a symlink made and Ollama
    installed is only honest if it says so. The user has to know what state they are in."""
    text = installer.read_text(encoding="utf-8")
    failure = text[text.index("INSTALL FAILED"):]

    assert "not wired" in failure or "not connected" in failure or "Claude" in failure, (
        "the failure message never mentions Claude Desktop, which the user will assume is "
        "connected because the installer normally connects it"
    )


def test_uninstalling_removes_the_claude_desktop_entry_on_every_platform(tmp_path,
                                                                        monkeypatch):
    """Both Unix uninstallers strip the connector; Windows did not, so an uninstall left
    Claude Desktop pointing at a binary that no longer exists — and the .iss header claimed
    "uninstall removes all of it"."""
    import json

    import main

    cfg = tmp_path / "claude_desktop_config.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "campaign-intelligence": {"command": "/gone/campaign-intelligence"},
        "notes": {"command": "/still/here"},
    }}), encoding="utf-8")
    monkeypatch.setattr(main, "_desktop_config_path", lambda: cfg)

    main._unconfigure_desktop()

    after = json.loads(cfg.read_text(encoding="utf-8"))
    assert "campaign-intelligence" not in after["mcpServers"]
    assert after["mcpServers"]["notes"], "somebody else's connector is not ours to remove"


def test_removing_the_entry_when_there_is_no_config_is_not_an_error(tmp_path, monkeypatch):
    """An uninstaller that fails because the thing it removes was already gone is an
    uninstaller that leaves a half-removed product."""
    import main

    monkeypatch.setattr(main, "_desktop_config_path", lambda: tmp_path / "nope.json")

    main._unconfigure_desktop()      # must not raise


def test_the_windows_install_belongs_to_the_person_who_will_use_it():
    """With PrivilegesRequired=admin and plain Exec, every step ran as whoever answered the
    UAC prompt. In the case the product is built for — IT installing for a marketer —
    %LOCALAPPDATA% and %APPDATA% then resolve to the ADMIN's profile: the embedding model is
    pulled into the admin's Ollama, the admin's Claude Desktop is wired, the data directory
    is created under the admin's account, and the self-test passes against an environment
    the marketer will never see. The product is per-user everywhere else (config.py's
    DATA_DIR, main.py's desktop config path)."""
    text = WINDOWS.read_text(encoding="ascii")

    privileges = next(l for l in text.splitlines() if l.startswith("PrivilegesRequired"))
    if "admin" in privileges:
        body = text[text.index("[Code]"):]
        for step in ("InitData", "ConfigureDesktop", "InstallOllama"):
            procedure = body[body.index(f"procedure {step}"):]
            procedure = procedure[:procedure.index("end;")]
            assert "ExecAsOriginalUser" in procedure, (
                f"{step} runs elevated, so it acts on the administrator's profile rather "
                f"than the profile of whoever will actually use this"
            )


@pytest.mark.parametrize("installer", [LINUX, MACOS], ids=lambda p: p.parent.name)
def test_an_upgrade_does_not_destroy_the_working_copy_before_verifying_the_new_one(installer):
    """`rm -rf "$DEST"` ran before anything was checked, so a failed upgrade left no working
    version at all — and "fix what it names and run this installer again" is advice about a
    copy the gate has just condemned. Stage, verify, then swap."""
    lines = installer.read_text(encoding="utf-8").splitlines()
    destroy = next((i for i, l in enumerate(lines)
                    if 'rm -rf "$DEST"' in l and not l.strip().startswith("#")), None)

    if destroy is None:
        return      # staged install: nothing is destroyed up front
    gate = next(i for i, l in enumerate(lines)
                if "health-check" in l and l.strip().startswith("if !"))
    assert destroy > gate, (
        "the existing install is deleted before the new one is verified, so a failed "
        "upgrade leaves the machine with neither"
    )


def test_the_gate_waits_for_a_daemon_it_started_three_seconds_ago(tmp_path, monkeypatch):
    """The installers start Ollama and sleep 3, then give the gate exactly one shot at it.
    A cold model load measured 0.7s on this machine — but on the Windows laptop the review
    was written against, with antivirus scanning a freshly written 274MB file, nobody has
    measured it. Refusing the install because a daemon was slow to warm up is a refusal the
    user can do nothing useful with.

    Only the transient codes are retried. A missing checkpoint is settled: waiting 60
    seconds to say so again is worse than saying it at once."""
    import core

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            return {"ok": False, "headline": "text search not ready",
                    "components": {"text_search": {"ok": False, "code": "embedder_unreachable",
                                                   "detail": "connection refused"}},
                    "coverage": {}}
        return {"ok": True, "headline": "fine", "components": {}, "coverage": {}}

    monkeypatch.setattr(core, "health_check_cli", flaky)
    monkeypatch.setattr(core.time, "sleep", lambda s: None)

    report = core.wait_until_ready(timeout=30)

    assert report["ok"] is True
    assert calls["n"] == 3, "it retried until the daemon answered"


def test_the_gate_does_not_wait_for_something_that_will_never_change(monkeypatch):
    import core

    calls = {"n": 0}

    def settled():
        calls["n"] += 1
        return {"ok": False, "headline": "no weights",
                "components": {"visual_search": {"ok": False, "code": "clip_weights_missing",
                                                 "detail": "not there"}},
                "coverage": {}}

    monkeypatch.setattr(core, "health_check_cli", settled)
    monkeypatch.setattr(core.time, "sleep", lambda s: None)

    report = core.wait_until_ready(timeout=30)

    assert report["ok"] is False
    assert calls["n"] == 1, "a missing checkpoint will not appear by waiting for it"


@pytest.mark.parametrize("installer", [LINUX, MACOS], ids=lambda p: p.parent.name)
def test_the_unix_installers_give_the_daemon_time(installer):
    gate = next(l for l in installer.read_text(encoding="utf-8").splitlines()
                if "health-check" in l and l.strip().startswith("if !"))

    assert "--wait" in gate, f"one shot at a daemon started seconds earlier: {gate}"


def test_the_windows_gate_uses_a_mechanism_that_actually_stops_the_success_page():
    """`RaiseException` at `ssPostInstall` does NOT stop anything. From Inno Setup's own
    source, `Setup.MainForm.pas`:

        SetStep(ssPostInstall, True);        // HandleExceptions = True
        ...
        except
          if HandleExceptions then begin
            Log('CurStepChanged raised an exception.');
            Application.HandleException(Self);   // shows it, swallows it
          end

    Execution continues to the Finished page and the exit code stays 0 — which is precisely
    the "message dismissed, success screen shown anyway" behaviour this was supposed to
    replace. Only `ssInstall` re-raises, and an exception there rolls the files back.

    So the Finished page has to be rewritten instead, and the plan must not claim an abort
    it cannot perform."""
    body = WINDOWS.read_text(encoding="ascii")

    assert "CurPageChanged" in body and "wpFinished" in body, (
        "nothing rewrites the Finished page, so a failed self-test still shows success"
    )
    self_test = body[body.index("procedure SelfTest"):]
    self_test = self_test[:self_test.index("\nprocedure ") if "\nprocedure " in self_test
                          else len(self_test)]
    assert "RaiseException" not in self_test, (
        "RaiseException at ssPostInstall is caught by Inno and discarded"
    )


def test_the_windows_self_test_command_is_quoted_so_it_actually_runs():
    """The redirect was inside the quoted region, so cmd passed the whole thing as one
    argument and the exe exited 2 with no log file written — `LoadStringsFromFile` then
    returned False without raising, so the dialog named nothing at all.

        /C ""...exe" health-check" > "...txt" 2>&1
              ^ first quote stripped        ^ last quote stripped

    The outer quote has to close after the redirect, not before it."""
    body = WINDOWS.read_text(encoding="ascii")
    line = next(l for l in body.splitlines() if "'/C" in l)

    assert line.rstrip().endswith("2>&1\"',") or "2>&1\"'" in line, (
        f"the outer quote closes before the redirect, so nothing is captured: {line}"
    )


def test_no_line_of_the_inno_script_starts_with_a_hash():
    """Inno's preprocessor reads a line beginning with `#` as a directive, so a wrapped
    string continuation starting with `#13#10` fails to compile with "Unknown preprocessor
    directive" — on line 221, which nothing in this suite could see. Caught the first time
    the script was actually compiled in CI. This keeps the shape that breaks it out of the
    file between compiles."""
    directives = ("define", "undef", "if", "ifdef", "ifndef", "elif", "else", "endif",
                  "include", "emit", "error", "pragma", "expr", "insert", "sub", "endsub",
                  "for", "dim", "file", "sethostname")

    for number, line in enumerate(WINDOWS.read_text(encoding="ascii").splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        word = stripped[1:].split(None, 1)[0].split("(")[0].lower() if len(stripped) > 1 else ""
        assert word in directives, (
            f"line {number} begins with # but is not a directive, so Inno's preprocessor "
            f"rejects it: {stripped[:60]}"
        )


# ── 4.4 macOS: the quarantine flag the installer believes it removes ────────────

@pytest.mark.skipif(sys.platform != "darwin", reason="xattr semantics are macOS's")
def test_the_macos_installer_actually_removes_the_quarantine_flag(tmp_path):
    """§4.4, measured before the change and it does not work.

    `xattr -dr com.apple.quarantine "$DEST"` strips the LIVE directory — but everything the
    installer runs (`init`, the `health-check` gate) runs out of `$STAGE`, and `$STAGE` is
    what `mv` then puts in place. On a fresh install `$DEST` does not exist, so the strip is a
    no-op swallowed by `|| true`; on an upgrade it strips the copy that is about to be deleted.
    Either way the binary Claude Desktop launches is still quarantined, along with the CLIP
    weights beside it.

    The plan calls the strip "a legitimate stopgap inside an installer the user chose to run".
    It is that only if it happens; today it does not, which is a worse position than the one
    written down — and on macOS 15+ an unsigned quarantined binary launched by Claude Desktop
    is blocked outright rather than offering "open anyway"."""
    bundle = tmp_path / "bundle"
    (bundle / "models").mkdir(parents=True)
    script = bundle / "install.sh"
    script.write_bytes(MACOS.read_bytes())
    stub = bundle / "campaign-intelligence"
    stub.write_text("#!/usr/bin/env bash\nexit 0\n")
    stub.chmod(0o755)
    (bundle / "models" / "open_clip_model.safetensors").write_text("weights")

    # What a browser download leaves on every file in the archive.
    for path in (stub, bundle / "models" / "open_clip_model.safetensors"):
        subprocess.run(["xattr", "-w", "com.apple.quarantine", "0083;00000000;Safari;",
                        str(path)], check=True)

    home = tmp_path / "home"
    result = subprocess.run(["bash", str(script), "--no-ollama"], cwd=bundle,
                            capture_output=True, text=True, timeout=180,
                            env={**os.environ, "HOME": str(home)})
    assert result.returncode == 0, result.stdout + result.stderr

    dest = home / "Library" / "Application Support" / "CampaignIntelligence"
    installed = dest / "campaign-intelligence"
    assert installed.is_file(), f"the installer did not place the binary: {result.stdout}"

    def quarantined(path):
        got = subprocess.run(["xattr", str(path)], capture_output=True, text=True)
        return "com.apple.quarantine" in got.stdout

    assert not quarantined(installed), (
        "the installed binary is still quarantined — the strip ran against a directory that "
        "did not exist yet, so Claude Desktop launches a quarantined binary"
    )
    assert not quarantined(dest / "models" / "open_clip_model.safetensors"), (
        "the CLIP weights are still quarantined"
    )

    # And the UPGRADE, which the first version of this argued rather than measured. It is the
    # path where the old code looked most plausible — `$DEST` exists, so the strip appeared to
    # do something — and it stripped the copy `mv` was about to delete, leaving the NEW files
    # quarantined exactly as on a fresh install.
    for path in (stub, bundle / "models" / "open_clip_model.safetensors"):
        subprocess.run(["xattr", "-w", "com.apple.quarantine", "0083;00000000;Safari;",
                        str(path)], check=True)
    again = subprocess.run(["bash", str(script), "--no-ollama"], cwd=bundle,
                           capture_output=True, text=True, timeout=180,
                           env={**os.environ, "HOME": str(home)})
    assert again.returncode == 0, again.stdout + again.stderr
    assert not quarantined(installed), (
        "an upgrade left the replacement binary quarantined — the strip ran against the copy "
        "it was about to delete"
    )


# ── 4.4 macOS: the .pkg a marketer can actually double-click ────────────────────

PKG = ROOT / "installer" / "macos" / "pkg"
POSTINSTALL = PKG / "scripts" / "postinstall"


def _shell_code(path: Path) -> str:
    """A shell script with its comment lines removed.

    Review demonstrated the need: moving the `sudo -u … HOME=…` line into a `#` comment left
    the test that asserts on it green. A check that reads prose is checking that somebody
    once described the behaviour, which is the opposite of what it claims."""
    return "\n".join(line for line in path.read_text().splitlines()
                      if not line.lstrip().startswith("#"))


def _staged(tmp_path, install_sh: str):
    """A staging directory shaped like the .pkg payload, with install.sh replaced by a stub
    that reports how it was invoked. The postinstall's whole contract is how it calls that
    script, so the stub is the instrument, not a shortcut."""
    staging = tmp_path / "staging"
    (staging / "models").mkdir(parents=True)
    (staging / "campaign-intelligence").write_text("#!/bin/bash\nexit 0\n")
    (staging / "campaign-intelligence").chmod(0o755)
    script = staging / "install.sh"
    script.write_text(install_sh)
    script.chmod(0o755)
    return staging


def _postinstall(staging, *args, **env):
    # HOME inside the staging directory's parent, never the real one. Run as a non-root user
    # the postinstall writes its install log to `$HOME/Library/Logs` — which is correct
    # behaviour and has no business happening on the machine running the tests.
    home = pathlib.Path(staging).parent / "home"
    home.mkdir(parents=True, exist_ok=True)
    return subprocess.run(["bash", str(POSTINSTALL), *args], capture_output=True, text=True,
                          timeout=120,
                          env={**os.environ, "CAMPAIGN_POC_PKG_STAGING": str(staging),
                               "CAMPAIGN_POC_PKG_USER": os.environ.get("USER", "runner"),
                               "HOME": str(home), **env})


def test_the_pkg_postinstall_runs_the_installer_rather_than_repeating_it():
    """The gate — weights checksum, `init`, `health-check` as a blocking self-test, Claude
    Desktop wired only afterwards — is `install.sh`'s, and the .pkg has to RUN it. A
    postinstall that repeated any of it would be the second implementation of one rule this
    codebase has been bitten by repeatedly, and the copy that drifts is always the one nobody
    runs by hand."""
    code = _shell_code(POSTINSTALL)

    assert "install.sh" in code, "the postinstall does not run the installer at all"
    for repeated in ("health-check", "configure-desktop", "shasum", "com.apple.quarantine"):
        assert repeated not in code, (
            f"the postinstall repeats {repeated!r}, which install.sh already does"
        )


@pytest.mark.skipif(sys.platform != "darwin", reason="pkg scripts are macOS's")
def test_a_failed_gate_fails_the_pkg_install(tmp_path):
    """Executed. Installer.app shows a success screen on exit 0, so a postinstall that
    swallows the gate's status ships the review's original defect wearing a nicer wrapper —
    a marketer told everything worked, over a product whose vision half is dead."""
    staging = _staged(tmp_path, "#!/bin/bash\necho 'FAIL  visual_search: no weights' >&2\n"
                                "exit 1\n")

    result = _postinstall(staging, "--no-ollama")

    assert result.returncode != 0, (
        f"the package reported success over a failing self-test:\n{result.stdout}"
    )
    assert "visual_search" in (result.stdout + result.stderr)
    assert "NOT been connected" in result.stderr


@pytest.mark.skipif(sys.platform != "darwin", reason="pkg scripts are macOS's")
def test_the_pkg_installs_for_the_person_who_double_clicked_not_for_root(tmp_path):
    """`installer` runs scripts as root, and this product installs per-user: into the user's
    Application Support directory, wiring that user's Claude Desktop. Run with root's
    environment it would land in /var/root and configure a Claude Desktop nothing reads, while
    reporting success — the install equivalent of writing to the wrong database."""
    staging = _staged(tmp_path, "#!/bin/bash\necho \"ran_as=$(id -un) home=$HOME\"\n")

    result = _postinstall(staging, "--no-ollama")

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"ran_as={os.environ.get('USER')}" in result.stdout, result.stdout
    assert "home=/var/root" not in result.stdout
    assert f"home={tmp_path / 'home'}" in result.stdout, (
        f"install.sh ran with the wrong HOME: {result.stdout}"
    )
    # The run above cannot tell whether the drop-to-user happened or was simply unnecessary:
    # pytest is not root, so `install.sh` would land on the right user either way. Under
    # `installer` it IS root, which is the case that matters and the one no test here can
    # enter without becoming root on somebody's machine. So the MECHANISM is asserted
    # directly — and `HOME=` with it, because `sudo -u` alone keeps root's environment, which
    # is how a per-user installer writes to /var/root while reporting success.
    code = _shell_code(POSTINSTALL)
    assert 'sudo -u "$target_user"' in code
    assert 'HOME="$target_home"' in code
    assert "/dev/console" in code, (
        "nothing works out who is actually logged in, so root is whoever installed it"
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="pkg scripts are macOS's")
def test_the_staged_payload_does_not_outlive_the_install(tmp_path):
    """The payload stages through /usr/local so the postinstall has something to run. Left
    there it is a second copy of the binary and its 2GB of weights that no uninstaller knows
    about — and on a FAILED install, 2GB kept as a souvenir of a product that does not work."""
    for install_sh, label in (("#!/bin/bash\nexit 0\n", "a successful install"),
                              ("#!/bin/bash\nexit 1\n", "a failed install")):
        staging = _staged(tmp_path / label.replace(" ", "_"), install_sh)
        _postinstall(staging, "--no-ollama")
        assert not staging.exists(), f"{label} left the staged payload behind"


@pytest.mark.skipif(sys.platform != "darwin", reason="pkg scripts are macOS's")
def test_the_installers_own_arguments_are_not_passed_on_as_options(tmp_path):
    """`installer` hands a postinstall four positional parameters — the package path, the
    destination, the destination volume and the root of the System folder — and `install.sh`
    rejects anything it does not recognise with `exit 2`. Forwarded, they produced

        unknown arg: /Volumes/.../CampaignIntelligence.pkg

    so every .pkg install failed, always, before the gate had run — and the failure message
    said the self-test had named a failing component, which had not happened. The arguments
    describe the PACKAGE; they say nothing about how to install."""
    staging = _staged(tmp_path, '#!/bin/bash\n'
                                'for a in "$@"; do case "$a" in\n'
                                '  --no-ollama) ;;\n'
                                '  *) echo "unknown arg: $a"; exit 2;; esac; done\n'
                                'echo "ran with $# argument(s)"\n')

    result = _postinstall(staging, "/Volumes/x/CampaignIntelligence.pkg", "/", "/", "/")

    assert result.returncode == 0, (
        f"the arguments `installer` passes reached the option parser:"
        f"\n{result.stdout}{result.stderr}"
    )
    assert "ran with 0 argument(s)" in result.stdout, result.stdout


@pytest.mark.skipif(sys.platform != "darwin", reason="pkg scripts are macOS's")
def test_a_failure_before_the_gate_is_not_reported_as_a_failed_gate(tmp_path):
    """The message said "the self-test above names the component that is not working" whatever
    went wrong. The first thing that went wrong was argument parsing, before the self-test ran
    — so it named a cause that had not happened and pointed at output that did not exist,
    which is the confident unfounded claim this product is built against, in its own
    installer."""
    staging = _staged(tmp_path, "#!/bin/bash\necho 'could not create the data directory' >&2\n"
                                "exit 1\n")

    result = _postinstall(staging, "--no-ollama")

    assert result.returncode != 0
    assert "the self-test above names" not in result.stderr, result.stderr
    assert "could not create the data directory" in (
        (pathlib.Path(staging).parent / "home" / "Library" / "Logs"
         / "CampaignIntelligence-install.log").read_text()), (
        "what actually failed was not written down anywhere"
    )
    assert "See " in result.stderr and "install.log" in result.stderr, (
        f"the message names no cause and points at nothing: {result.stderr}"
    )


def test_the_package_cannot_be_built_without_the_installer_it_runs():
    """A payload missing install.sh gives a package whose postinstall has nothing to run —
    and the failure would land on the marketer's machine, at the only moment they are paying
    attention."""
    import tempfile

    # Executed. The phrase alone survived the guard being deleted, because the phrase was
    # still in the comment above it.
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "bundle"
        bundle.mkdir()
        (bundle / "campaign-intelligence").write_text("#!/bin/bash\nexit 0\n")
        (bundle / "campaign-intelligence").chmod(0o755)

        result = subprocess.run(["bash", str(PKG / "build-pkg.sh"), str(bundle), "0.0.0",
                                 str(Path(tmp) / "out.pkg")],
                                capture_output=True, text=True, timeout=120)

    assert result.returncode != 0, (
        "a package was built whose postinstall has nothing to run — the failure would land "
        "on the marketer's machine, at the only moment they are paying attention"
    )
    assert "install.sh" in result.stderr


def test_the_installer_gate_cannot_be_skipped_by_the_package():
    """`require-scripts` — without it a payload-only install is a supported option, and the
    self-test that BLOCKS success becomes advisory."""
    import xml.etree.ElementTree as ET

    # PARSED. Moving the attribute into an XML comment left the string check green, which is
    # a test asserting that somebody once wrote the word down.
    root = ET.parse(PKG / "distribution.xml").getroot()
    options = root.find("options")
    assert options is not None, "the package declares no options element at all"
    assert options.get("require-scripts") == "true", options.attrib
    # And the domains that decide whether a marketer without an admin password can install.
    domains = root.find("domains")
    assert domains is not None and domains.get("enable_currentUserHome") == "true", (
        "a system-only package demands an admin password for a per-user product — on an "
        "IT-managed Mac the marketer cannot install it at all, where the tarball worked"
    )


def test_signing_covers_the_binary_and_not_only_the_package():
    """A signed .pkg around an unsigned binary notarizes and then fails on launch: the
    package's signature says nothing about what is inside it, and Gatekeeper checks the thing
    that runs. And `--options runtime`, without which notarization is refused."""
    build = _shell_code(PKG / "build-pkg.sh")
    assert "codesign" in build and "campaign-intelligence" in build
    # By what the file IS. `-name "*.so" -o -name "*.dylib"` misses the Mach-O executables a
    # PyInstaller bundle carries with no extension at all — torch ships `bin/protoc` and
    # `bin/torch_shm_manager` as DATA — and notarization refuses a payload containing an
    # unsigned Mach-O.
    assert "Mach-O" in build, (
        "signing selects files by extension, so an unextensioned Mach-O ships unsigned"
    )
    # Shell continuations joined, so each `codesign` is one command and the check cannot be
    # satisfied by a neighbouring call's flags. Per call matters: `--options runtime` on one
    # of two invocations still leaves an unhardened Mach-O in the payload, and notarization
    # refuses the whole package for it.
    joined, buffer = [], ""
    for line in build.splitlines():
        code = line.split("#", 1)[0].rstrip()
        if code.endswith("\\"):
            buffer += code[:-1] + " "
            continue
        joined.append(buffer + code)
        buffer = ""
    signing = [c for c in joined if "codesign" in c and "--verify" not in c]
    assert len(signing) >= 2, (
        f"a PyInstaller bundle is a directory of Mach-O files; only {len(signing)} signed"
    )
    for command in signing:
        assert "--options runtime" in command, (
            f"signs without the hardened runtime, which notarization refuses: {command.strip()}"
        )
    assert "stapler staple" in build, (
        "notarization without stapling asks Apple at first launch, on a network §4.2 says "
        "may not allow it"
    )


def test_the_package_builds_unsigned_and_says_so():
    """Signing is conditional on credentials this repository does not hold. It must still
    produce a package — an unsigned .pkg warns and lets the user proceed, which is strictly
    more than the tarball ever allowed — and it must not claim to be signed."""
    build = _shell_code(PKG / "build-pkg.sh")
    assert "UNSIGNED" in build
    assert "MACOS_SIGN_IDENTITY" in build


# ── D25: an installer that is compiled is not an installer that has run ─────────

BUILD_WORKFLOW = ROOT / ".github" / "workflows" / "build.yml"


def _ci_code() -> str:
    """The workflow with its comment lines removed.

    Twice now a check of this kind has passed on a comment: the prose explaining why a flag
    matters contains the flag, so deleting the flag from the command left the assertion green.
    A step's comments are worth having and are not the step."""
    return "\n".join(line for line in BUILD_WORKFLOW.read_text().splitlines()
                      if not line.lstrip().startswith("#"))


@pytest.mark.parametrize("platform,run_marker", [
    ("windows", "CampaignIntelligence-Setup.exe"),
    ("macos", "installer -pkg CampaignIntelligence.pkg"),
])
def test_the_release_build_runs_each_installer_rather_than_only_building_it(platform,
                                                                           run_marker):
    """D25, owed since §4.1: "the `[Code]` path has still never been executed anywhere;
    compiling it is new but is not the same thing". Inno's `[Code]` section holds the gate that
    must block the success page, and running ISCC proves the Pascal compiled. The same was true
    of macOS one step earlier — there was no package at all."""
    ci = _ci_code()

    assert run_marker in ci, (
        f"the {platform} installer is built but never executed in CI"
    )
    if platform == "windows":
        # Naming the exe is not running it. `/VERYSILENT` is what makes it install
        # unattended, and `/SUPPRESSMSGBOXES` is what stops a modal hanging the runner until
        # the job times out — which reads as a slow build rather than a broken installer.
        assert "/VERYSILENT" in ci and "/SUPPRESSMSGBOXES" in ci
        assert "/LOG=install.log" in ci, "a failed install leaves nothing to read"


def test_the_release_build_requires_a_broken_bundle_to_be_refused():
    """The half D25 names explicitly. An installer that has only ever met a good bundle has
    never had its gate exercised — and the gate is the one acceptance criterion §4.1 turns on.
    Both platforms corrupt the weights the way an interrupted download does and require the
    install to fail, and to say why."""
    ci = _ci_code()

    assert ci.count("truncated") >= 2, (
        "only one platform tests its installer against a deliberately broken bundle"
    )
    # What a refusal LOOKS like differs by platform, and assuming it did not was a defect of
    # its own. On macOS `install.sh` exits non-zero and the package fails. On Windows a
    # refused install exits ZERO by design — the .iss explains at length that Inno discards an
    # exception at ssPostInstall and that raising at ssInstall would roll back the files the
    # user is told to fix and re-run against. So the shared, observable promise is the one the
    # .iss states: "Never wire a product the self-test just rejected."
    assert ci.count("rejected was wired to Claude Desktop") >= 2, (
        "only one platform checks that a refused product is left unwired"
    )
    assert ci.count("was not wired to Claude Desktop") >= 2, (
        "without the positive control, an installer that never wires anything also passes"
    )
    assert "weights are corrupt" in ci, (
        "the macOS refusal is matched on the wrapper's own INSTALL FAILED line, which it "
        "prints whatever went wrong — an assertion that cannot fail"
    )


def test_the_windows_installer_can_be_pointed_at_the_bundle_under_test():
    """The broken-bundle run needs an installer built around a DIFFERENT payload, and the
    script hard-coded `..\\..\\dist\\campaign-intelligence`. Defaulted so an ordinary build is
    unchanged, overridable so the refusal can be tested at all."""
    iss = WINDOWS.read_text(encoding="utf-8")

    assert "#ifndef SourceDir" in iss and "#define SourceDir" in iss
    assert "{#SourceDir}\\*" in iss
    assert "..\\..\\dist\\campaign-intelligence\\*" not in iss, (
        "a Source line still hard-codes the bundle, so it ignores the override"
    )


def test_the_installed_product_is_required_to_pass_its_own_self_test_in_ci():
    """Installing is not the claim. §4.1's criterion is that the product WORKS afterwards, and
    the reviewer's machine had it installed with visual search silently dead. Both end-to-end
    runs execute the INSTALLED copy — not the build tree — and require every component green."""
    ci = _ci_code()

    assert ci.count("fails its own self-test") >= 2
    assert "Library/Application Support/CampaignIntelligence" in ci
    assert "Programs\\CampaignIntelligence" in ci


def test_the_release_ships_the_package_it_builds():
    """A .pkg built in CI and left on the runner is a .pkg nobody can install. The tarball
    stays — it is what a terminal user and every script expects — but the thing a marketer
    double-clicks has to be attached to the release."""
    ci = _ci_code()

    def step(name):
        """One step's own body. Splitting on a heading and reading to the END of the file
        makes every later step's text count as this one's — an artefact list that dropped the
        package still "contained" it, because the release step below did."""
        after = ci.split(name, 1)[1]
        rest = after.split("\n      - name:", 1)
        return rest[0]

    assert "CampaignIntelligence.pkg" in step("Attach to release"), (
        "the .pkg is built and then not released"
    )
    assert "CampaignIntelligence.pkg" in step("Upload build artifacts"), (
        "a failed release leaves no .pkg to inspect either"
    )


def test_the_macos_only_tests_are_run_somewhere():
    """Every test in this file that EXECUTES a macOS installer script is guarded by
    `skipif(sys.platform != "darwin")`, and the test workflow ran pytest on ubuntu alone — so
    the only tests here that do more than read text were skipped, silently, everywhere. That is
    the gap the Windows compile job was added to close, one platform over."""
    ci = (ROOT / ".github" / "workflows" / "test.yml").read_text()

    assert "macos-latest" in ci, (
        "nothing runs this file on macOS, so every executed test in it is skipped in CI"
    )
    assert "test_installer_acceptance.py" in ci.split("macos-latest", 1)[1]
    assert "-rs" in ci, (
        "a run that skips them on macOS would look identical to one that ran them"
    )


def test_the_readme_says_the_macos_package_is_unsigned_and_what_to_do():
    """§4.2's offline limit and §4.3's Ollama limit are both stated to USERS in README. §4.4's
    was stated only in the plan and the tracker — documents a marketer will never open — while
    the thing they actually meet is a Gatekeeper refusal on the first double-click.

    And the plan's own phrasing was optimistic: "Installer.app warns and the user can proceed"
    is true only if you know that on macOS 15+ the Control-click override is gone and the way
    through is System Settings → Privacy & Security → Open Anyway."""
    # Whitespace collapsed: prose wraps, and a check that a sentence appears on one line is a
    # check about the editor rather than about the documentation.
    readme = re.sub(r"\s+", " ", (ROOT / "README.md").read_text())

    assert "CampaignIntelligence.pkg" in readme, (
        "the release ships a .pkg that appears in no user-facing document"
    )
    assert "unidentified developer" in readme
    assert "Privacy & Security" in readme and "Open Anyway" in readme, (
        "it says the package is unsigned without saying how to get past it"
    )
    assert "CampaignIntelligence-install.log" in readme, (
        "Installer.app shows none of the gate's output, and nothing says where it went"
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="the macOS installer's Ollama path")
def test_a_failed_ollama_install_reaches_the_gate_rather_than_killing_the_script(tmp_path):
    """§4.3's argument: whether a failed dependency step actually broke anything "is the
    self-test's question to answer, and it answers it about the machine rather than about
    whether one command returned zero".

    Under `set -euo pipefail` a bare `curl … | sh` ends the script where it stands. The
    marketer then gets a non-zero status with no component named — and under the .pkg, where
    Ollama's installer writes `/Applications` and symlinks `/usr/local/bin` as a standard user
    with no terminal to prompt into, that is the likely path rather than the exotic one."""
    bundle = tmp_path / "bundle"
    (bundle / "models").mkdir(parents=True)
    script = bundle / "install.sh"
    script.write_bytes(MACOS.read_bytes())
    stub = bundle / "campaign-intelligence"
    stub.write_text("#!/usr/bin/env bash\n"
                    'case "$1" in\n'
                    "  health-check) echo 'FAIL  text_search: the embedding model is not "
                    "installed'; exit 1;;\n"
                    "  *) exit 0;;\n"
                    "esac\n")
    stub.chmod(0o755)

    # No ollama on PATH, and a curl that fails the way a standard account's install fails.
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "curl").write_text("#!/bin/bash\necho 'curl: permission denied' >&2\nexit 1\n")
    (fake / "curl").chmod(0o755)

    result = subprocess.run(["bash", str(script)], cwd=bundle, capture_output=True, text=True,
                            timeout=180,
                            env={**os.environ, "HOME": str(tmp_path / "home"),
                                 "PATH": f"{fake}:/usr/bin:/bin"})

    assert result.returncode != 0, "the install reported success with no embedding model"
    assert "text_search" in (result.stdout + result.stderr), (
        "the script died at the Ollama step, so the gate never ran and nothing named a "
        f"component:\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="pkg scripts are macOS's")
def test_running_the_tests_does_not_put_a_dialog_on_anybodys_screen(tmp_path):
    """The failure dialog exists so the gate's words reach somebody — Installer.app shows none
    of a postinstall's output. Ungated it also reached whoever ran the test suite: a modal
    appeared on their desktop, once per failure-path test, telling them a product they had
    never installed could not be installed. A test run must not draw on somebody's screen.

    `INSTALL_PKG_SESSION_ID` and `PACKAGE_PATH` are set by `installer` and Installer.app and by
    nothing else, which is exactly the question being asked: is this a real install?

    Exercised on a COPY whose `osascript` resolves through PATH. The shipped script calls it by
    absolute path — correct, since PATH under `installer` is minimal — and a stub could
    therefore never intercept the real one, so a test written against the shipped path would
    have reported "no dialog" whatever the code did."""
    copy = tmp_path / "postinstall"
    copy.write_text(POSTINSTALL.read_text().replace("/usr/bin/osascript", "osascript"))
    staging_src = _staged(tmp_path, "#!/bin/bash\necho 'the weights are corrupt' >&2\nexit 1\n")
    fake = tmp_path / "bin"
    fake.mkdir()
    marker = tmp_path / "osascript-was-called"
    (fake / "osascript").write_text(f"#!/bin/bash\ntouch {marker}\n")
    (fake / "osascript").chmod(0o755)

    def run(**extra):
        if not staging_src.exists():          # the postinstall removes it either way
            shutil.copytree(_staged(tmp_path / "again", "#!/bin/bash\nexit 1\n"), staging_src)
        marker.unlink(missing_ok=True)
        subprocess.run(["bash", str(copy)], capture_output=True, text=True, timeout=120,
                       env={**os.environ, "CAMPAIGN_POC_PKG_STAGING": str(staging_src),
                            "CAMPAIGN_POC_PKG_USER": os.environ.get("USER", "runner"),
                            "HOME": str(tmp_path / "home"),
                            "PATH": f"{fake}:/usr/bin:/bin", **extra})
        time.sleep(1)                          # it is backgrounded, so give it a moment
        return marker.exists()

    assert run(INSTALL_PKG_SESSION_ID="abc123"), (
        "no dialog inside a real Installer session — the gate's words reach nobody, since "
        "Installer.app shows none of a postinstall's output"
    )
    assert not run(), (
        "running the test suite put a dialog on the screen of whoever ran it"
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="pkg scripts are macOS's")
def test_a_system_install_with_no_gui_session_falls_back_to_who_ran_it(tmp_path):
    """On a machine with no GUI session — an SSH install, a CI runner, an MDM push —
    `/dev/console` is owned by root. Refusing there was defensible and wrong: the admin who
    typed `sudo installer` is a better answer than no answer, and `SUDO_USER` is set by sudo
    itself so it cannot invent a user who is not there.

    Asserted on the code, because entering the case needs to BE root on somebody's machine."""
    code = _shell_code(POSTINSTALL)

    assert "SUDO_USER" in code, (
        "a system install with no console user refuses outright, which is every headless "
        "install — including the one in this repository's own CI"
    )
    console = code.index("/dev/console")
    sudo_user = code.index("SUDO_USER")
    assert console < sudo_user, (
        "SUDO_USER is consulted before the console user, so a real logged-in session loses "
        "to whoever happened to type sudo"
    )
    assert code.index("no home directory to install into") > sudo_user, (
        "it still refuses before trying the fallback"
    )


def test_the_refusal_check_reads_this_installs_behaviour_not_the_last_ones():
    """A run measured it: the good-path step installs and wires Claude Desktop, then the
    broken-bundle step asserts "the gate wired nothing" — and found the PREVIOUS install's
    entry still sitting there. The gate had behaved perfectly, refusing and naming the corrupt
    weights, and the assertion failed anyway.

    Both platforms clear the config before the broken install now. The Windows half always
    did; the macOS half did not, and that asymmetry was the whole bug."""
    ci = _ci_code()

    for platform, marker in (("macOS", "Library/Application Support/Claude"),
                             ("Windows", "Claude\\claude_desktop_config.json")):
        broken = ci.split("A broken bundle must fail", 1)[1] if platform == "macOS" else ci
        assert marker in broken, f"the {platform} broken-bundle step never names the config"

    # Removed, on both, before the install whose behaviour is being read.
    assert ci.count('rm -f "$cfg"') >= 1, "the macOS step does not clear the config first"
    assert "Remove-Item $cfg" in ci, "the Windows step does not clear the config first"


def test_ci_does_not_run_a_second_installer_inside_the_silent_install():
    """Measured on a runner: with the `ollama` task selected, the silent Windows install ran
    OllamaSetup.exe, which installed and launched the Ollama app and never returned. The step
    timed out at 25 minutes having printed nothing, and job cleanup killed two orphan `ollama`
    processes. It happened with Ollama ALREADY provisioned and the model pulled, because the
    installer could not see it.

    So CI provisions Ollama itself and deselects the task. The gate still gets a real
    `text_search` — which is also the commonest state of a customer's machine, one that already
    has Ollama."""
    ci = _ci_code()

    assert "/TASKS=desktop" in ci, (
        "the silent install still runs the installer's own Ollama task, which is the hang"
    )
    assert ci.count("/TASKS=desktop") >= 2, "only one of the two Windows installs deselects it"
    assert "Provision Ollama" in ci, "nothing provisions Ollama, so the gate cannot pass"


def test_an_expected_non_zero_self_test_is_not_the_steps_verdict():
    """Measured: the Windows broken-bundle step printed its own success line — "refused: the
    self-test failed and nothing was wired" — and then exited 1.

    `health-check` exits non-zero when a component is down, which is exactly what this step
    is checking for, and pwsh hands the step its last exit code. So a step whose every
    assertion passed reported failure, on the strength of a command behaving correctly."""
    ci = _ci_code()
    broken = ci.split("A broken bundle must fail the Windows install", 1)[1]

    assert "$global:LASTEXITCODE = 0" in broken, (
        "the expected non-zero from health-check still becomes the step's exit code"
    )
    assert "exit 0" in broken, "the step does not end on a verdict of its own"
