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
