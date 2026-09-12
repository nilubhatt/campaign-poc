"""
Product review defect 10 (P2): "No version surface; client caches stale tool schemas."

    "The client caches tool schemas at connect time. After the server was rebuilt
    mid-session, several parameters that were live and working — `markets`, `status`, tag
    `source`, `match_all_tags`, `confirm` — were absent from the schemas in use, and had to
    be rediscovered by trial and error against a server that already supported them."

    Fix: "Expose a build version in the server description so drift is visible at a glance.
    Document 'fully quit and reopen the host app after rebuilding' in the developer README —
    closing the window is not enough, the server keeps running."

The version is the asked-for half. It is also the weaker half: a version string tells you
the server changed, not that the schema in your hand is missing a parameter — and the
reviewer's symptom was silent absence, which nothing announces. So the server also publishes
what its tools actually take, which is the one thing a caller holding a stale schema can
compare against and notice.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import config
import core
import mcp_server

ROOT = Path(__file__).resolve().parent.parent


def test_the_server_knows_its_own_version():
    assert config.VERSION, "a build with no version cannot be told apart from any other"


def test_the_version_is_visible_without_calling_a_tool():
    """"Visible at a glance" means in the server's own description, which a host shows next
    to the connector — not behind a call the user has to know to make."""
    assert config.VERSION in mcp_server.mcp.instructions


def test_the_health_check_reports_the_version(conn):
    assert core.health_check(conn)["version"] == config.VERSION


def test_the_binary_can_say_its_version_without_a_database():
    """The first question support asks. It must work on a machine where the product is
    broken, so it cannot need the database, the embedder or the model."""
    result = subprocess.run([sys.executable, "main.py", "--version"],
                            cwd=ROOT, capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    assert config.VERSION in result.stdout


def test_the_server_publishes_what_its_tools_actually_take(conn):
    """The reviewer's real symptom: parameters that were live and working were absent from
    the cached schema, and the only way to find them was trial and error. A version string
    does not fix that — it says the server changed, not which of your parameters are
    missing. This is the list a caller can compare its own schema against."""
    report = core.health_check(conn)

    tools = report["tools"]
    assert tools["find_similar_campaigns"], "the tool the reviewer was fighting with"
    for parameter in ("markets", "status", "match_all_tags", "tags"):
        assert parameter in tools["find_similar_campaigns"], parameter
    assert "confirm" in tools["upload_campaign"]


def test_every_published_parameter_really_exists_on_the_tool(conn):
    """The earlier version of this test was tautological — it compared the registry against
    a set derived from the registry, so it could not fail. This compares against the
    functions themselves. (Whether the tool LIST is complete is
    test_the_inventory_covers_every_tool_the_server_actually_advertises.)"""
    import inspect

    report = core.health_check(conn)

    for tool, params in report["tools"].items():
        fn = getattr(mcp_server, tool, None)
        assert fn is not None, f"{tool} is published but is not a function on mcp_server"
        real = set(inspect.signature(fn).parameters)
        assert set(params) <= real, f"{tool}: published {set(params) - real} which do not exist"


def test_a_stale_client_can_be_told_what_it_is_missing(conn):
    """The point of publishing the inventory: a caller holding a schema from before a rebuild
    can name the gap instead of rediscovering it by trial and error."""
    report = core.health_check(conn)

    cached = {"find_similar_campaigns": ["text", "top_k"]}      # what the reviewer had
    missing = core.stale_schema_parameters(report["tools"], cached)

    assert "markets" in missing["find_similar_campaigns"]
    assert "match_all_tags" in missing["find_similar_campaigns"]


def test_nothing_is_reported_missing_when_the_client_is_current(conn):
    report = core.health_check(conn)

    assert core.stale_schema_parameters(report["tools"], report["tools"]) == {}


def test_the_readme_tells_a_developer_how_to_pick_up_a_rebuild():
    """The other half of the review's fix, and the half a test can actually pin: "closing the
    window is not enough, the server keeps running"."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()

    assert "quit" in readme and "reopen" in readme, "the restart instruction is missing"
    assert "closing the window" in readme, (
        "say why quitting is different from closing — that is the part nobody guesses"
    )


def test_the_version_is_not_hand_edited_in_two_places():
    """Two copies of a version number disagree the first time somebody is in a hurry."""
    sources = []
    for path in (ROOT / "config.py", ROOT / "version.py", ROOT / "pyproject.toml"):
        if path.exists() and "VERSION" in path.read_text(encoding="utf-8").upper():
            sources.append(path.name)

    assert len(sources) <= 2, f"version appears to be defined in {sources}"


# ══ review of 3.2 ════════════════════════════════════════════════════════════

def test_the_inventory_covers_every_tool_the_server_actually_advertises():
    """The first version iterated a hand-typed `TOOL_NAMES` tuple — the hand-maintained copy
    this fix's own commit message said it refused to have. It matched on the day it was
    written; the day somebody adds a tool without editing the tuple, health_check omits it
    silently, which is defect 10 reproduced inside its own fix.

    And the test that was supposed to catch that could not: it asserted
    `registered == actual or registered >= {two known names}`, with `actual` derived from
    `registered`. Deleting a name from the tuple left the whole suite green."""
    import core
    import mcp_server

    advertised = {t.name for t in mcp_server.mcp._tool_manager.list_tools()}

    assert set(core.published_tool_parameters()) == advertised


def test_adding_a_tool_does_not_need_a_second_edit_to_be_published():
    """The property, rather than today's count: register one and it appears."""
    import core
    import mcp_server

    @mcp_server.mcp.tool()
    def _probe_tool(only_here_for_the_test: str = "x") -> dict:
        """A tool registered at test time."""
        return {}

    try:
        published = core.published_tool_parameters()
        assert "_probe_tool" in published
        assert published["_probe_tool"] == ["only_here_for_the_test"]
    finally:
        mcp_server.mcp.remove_tool("_probe_tool")


def test_a_locally_built_binary_is_not_indistinguishable_from_every_other():
    """"Drift visible at a glance" fails in the one scenario the review reported: a local
    rebuild. `build.sh` and `build.ps1` never stamped, so before and after a rebuild every
    surface said exactly the same thing — and called a frozen PyInstaller build a "source
    checkout", which it is not."""
    for script in (ROOT / "build.sh", ROOT / "build.ps1"):
        text = script.read_text(encoding="utf-8")
        assert "build_info.txt" in text, (
            f"{script.name} does not stamp the build, so two different local builds report "
            f"the same version"
        )
        assert "rev-parse" in text, f"{script.name} stamps nothing identifying"


def test_the_windows_installer_version_comes_from_the_one_source():
    """AppVersion was a second hand-edited copy and had already drifted — version.py said
    0.3.0 while the installer said 0.2.7, so Add/Remove Programs, the uninstall entry and
    upgrade detection would all disagree with the binary they installed."""
    iss = (ROOT / "installer" / "windows" / "campaign-intelligence.iss").read_text(
        encoding="ascii")

    version_line = next(l for l in iss.splitlines() if l.startswith("AppVersion"))
    assert "{#" in version_line, (
        f"the installer hard-codes a version: {version_line}. Pass it in from version.py."
    )

    workflow = (ROOT / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")
    assert "/DAppVersion" in workflow, "nothing supplies the version to ISCC"


def test_the_desktop_config_is_read_and_written_as_utf8():
    """Defect 11's own bug class, still shipped. `configure-desktop` read and wrote Claude
    Desktop's config with the locale codec — cp1252 on a Windows machine, where the file is
    UTF-8. Another connector pointing at C:\\Users\\José became C:\\Users\\JosÃ©, and a byte
    the codec cannot decode raised UnicodeDecodeError, which is not caught. It runs on every
    Windows and Linux install."""
    import ast

    tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_configure_desktop")

    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr in ("read_text", "write_text", "open")]
    assert calls, "nothing reads or writes the config?"
    for call in calls:
        assert any(kw.arg == "encoding" for kw in call.keywords), (
            f"{call.func.attr}() on line {call.lineno} uses the locale codec; on Windows "
            f"that is cp1252 and the file is UTF-8"
        )


def test_reading_a_desktop_config_with_a_non_ascii_path_does_not_corrupt_it(tmp_path,
                                                                           monkeypatch):
    """Executed rather than grepped: a config naming another connector under a non-ASCII
    path must come back byte-identical in that entry."""
    import json

    import main

    cfg = tmp_path / "claude_desktop_config.json"
    original = {"mcpServers": {"notes": {"command": "C:\\Users\\José\\notes.exe",
                                         "args": ["--flag"]}}}
    cfg.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(main, "_desktop_config_path", lambda: cfg)

    main._configure_desktop()

    after = json.loads(cfg.read_text(encoding="utf-8"))
    assert after["mcpServers"]["notes"]["command"] == "C:\\Users\\José\\notes.exe"
    assert "campaign-intelligence" in after["mcpServers"]


def test_the_readme_lists_every_tool_the_server_publishes():
    """It has drifted twice — once missing `finish_indexing`/`health_check`, once missing
    `diff_campaigns`/`gaps`. A hand-maintained list beside a generated one is the failure
    this project keeps rediscovering, so this is the cheap version of the guard."""
    import re

    import mcp_server

    text = (ROOT / "README.md").read_text(encoding="utf-8")
    section = text[text.index("## Tools (what Claude calls)"):text.index("## Files")]
    listed = set(re.findall(r"`([a-z_]+)`", section))
    published = {t.name for t in mcp_server.mcp._tool_manager.list_tools()}

    assert published - listed == set(), f"README omits {sorted(published - listed)}"
    assert listed - published == set(), f"README lists non-tools {sorted(listed - published)}"
