"""
Issue #23, part 2: every v0.4.0 macOS health check said `vector_index=python-cosine-fallback`.
vec0.dylib was bundled, but the build's Python (actions/setup-python on macOS) has no
`sqlite3.Connection.enable_load_extension`, and vectorstore falls back quietly, so nothing
failed. The build now uses Homebrew's Python on macOS and asserts the backend of the
packaged product on every platform. These tests keep both in the workflow.
"""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / ".github" / "workflows" / "build.yml"


def _steps():
    return yaml.safe_load(BUILD.read_text(encoding="utf-8"))["jobs"]["build"]["steps"]


def test_setup_python_is_not_used_on_macos():
    for step in _steps():
        if str(step.get("uses", "")).startswith("actions/setup-python"):
            assert step.get("if") == "runner.os != 'macOS'", (
                "setup-python's macOS CPython cannot load SQLite extensions; the macOS bundle "
                "must be built with Homebrew's python@3.12"
            )


def test_macos_python_is_proved_able_to_load_extensions_before_the_build():
    names = [s.get("name", "") for s in _steps()]
    mac = next(s for s in _steps() if s.get("name") == "Python with loadable SQLite extensions (macOS)")
    assert mac["if"] == "runner.os == 'macOS'"
    assert "python@3.12" in mac["run"]
    assert "enable_load_extension(True)" in mac["run"]
    assert "GITHUB_PATH" in mac["run"]
    assert names.index(mac["name"]) < names.index("Install dependencies")


def test_every_platform_asserts_the_packaged_product_uses_sqlite_vec():
    checks = [s for s in _steps() if "loads sqlite-vec" in s.get("name", "")]
    conditions = {s["if"] for s in checks}
    assert conditions == {"runner.os != 'Windows'", "runner.os == 'Windows'"}, (
        "Linux, macOS and Windows must each check the packaged product's vector index"
    )
    for step in checks:
        assert "init" in step["run"], "health-check needs a database to report the backend"
        assert "CAMPAIGN_POC_DATA" in step.get("env", {}), "use a fresh data directory"
        assert re.search(r"vector_index=sqlite-vec", step["run"])
