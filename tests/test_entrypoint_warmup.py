"""
Design-review finding on plan item 1.1, and the root cause of product-review defect 04
("Model init inside the request handler exceeds the MCP timeout").

`warm_up()` existed and was called from stdio_server.py and http_app.main() - but NOT from
main.py's `stdio` subcommand, which is the entry point the installers actually wire into
Claude Desktop (`configure-desktop` writes {"command": <exe>, "args": ["stdio"]}). So on
every installed copy the model was still loaded lazily inside the first image tool call,
which is exactly the 60-second transport timeout the reviewer hit. The fix is one line; this
test is what stops it regressing, since the source-checkout path it is easy to test by hand
was never the broken one.
"""
import sys
import types

import pytest

import main


@pytest.fixture
def stub_runtime(monkeypatch):
    """Stub everything main() touches so we can assert on ordering without a real server."""
    calls = []

    monkeypatch.setattr("config.ensure_dirs", lambda: calls.append("ensure_dirs"))
    monkeypatch.setattr("store.init_db", lambda: calls.append("init_db"))
    monkeypatch.setattr("clip_embed.warm_up", lambda: calls.append("warm_up"))
    monkeypatch.setattr("embedding.warm_up", lambda: calls.append("embed_warm_up"))

    fake_mcp = types.SimpleNamespace(run=lambda transport=None: calls.append(f"run:{transport}"))
    monkeypatch.setattr("mcp_server.mcp", fake_mcp)
    return calls


def test_stdio_subcommand_warms_the_model_before_serving(monkeypatch, stub_runtime):
    monkeypatch.setattr(sys, "argv", ["campaign-intelligence", "stdio"])

    main.main()

    assert "warm_up" in stub_runtime, (
        "the installed binary's stdio entry point must warm the vision model at startup - "
        "otherwise the first image tool call pays the model load and blows the MCP timeout"
    )
    assert stub_runtime.index("warm_up") < stub_runtime.index("run:stdio"), (
        "warm-up has to finish before the transport starts accepting tool calls"
    )
    # Both models, not just the vision one: the text embedder unloads after Ollama's idle
    # window, so without this the first chunk of every upload pays the reload inside a
    # handler - the same defect, one model over.
    assert "embed_warm_up" in stub_runtime
    assert stub_runtime.index("embed_warm_up") < stub_runtime.index("run:stdio")


def test_stdio_subcommand_still_initialises_dirs_and_db(monkeypatch, stub_runtime):
    monkeypatch.setattr(sys, "argv", ["campaign-intelligence", "stdio"])

    main.main()

    assert "ensure_dirs" in stub_runtime
    assert "init_db" in stub_runtime
