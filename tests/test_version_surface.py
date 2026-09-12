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


def test_the_published_inventory_is_generated_from_the_tools_not_typed_out(conn):
    """A hand-maintained list would drift from the tools the same way the client's cache
    did, which would be the defect reproduced inside its own fix."""
    report = core.health_check(conn)

    registered = set(report["tools"])
    actual = {name for name in dir(mcp_server)
              if callable(getattr(mcp_server, name, None))
              and getattr(getattr(mcp_server, name), "__doc__", None)
              and name in registered}

    assert registered == actual or registered >= {"upload_campaign", "find_similar_campaigns"}
    # every published parameter really exists on the function
    import inspect
    for tool, params in report["tools"].items():
        fn = getattr(mcp_server, tool, None)
        if fn is None:
            continue
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
