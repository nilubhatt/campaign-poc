"""
The same tool surface, for a caller that cannot speak MCP.

Both transports this product shipped speak MCP: `stdio` for a desktop client and `/mcp` for a
Claude Web connector. A firm that permits neither — no desktop client, no custom connectors,
only its own agent stack making HTTPS calls — could not reach the product at all, however open
its network was. The block is a policy about protocols rather than about ports, so a second
endpoint speaking the same protocol answered nothing.

**The whole design is that the description is GENERATED.** Hand-written REST paths beside the
MCP tools are two implementations of one surface, which is the shape every P1 of the §8.7
review had. So the first test here is the one that matters: every tool the protocol offers has
a path, generated from the same registry, and a tool added tomorrow appears without anybody
remembering to add it.
"""
import json

import pytest
from starlette.testclient import TestClient

import config
import openapi


@pytest.fixture
def client(conn):
    """The HTTP app, against the test database the `conn` fixture built.

    Two deliberate details. `http_app` is imported inside the fixture, because it builds the
    Starlette app at import time and doing that at module scope would happen before `conn` has
    pointed `config` at a temp directory.

    And the client is NOT used as a context manager, which is what runs the app's lifespan.
    The lifespan starts MCP's streamable-HTTP session manager, and that can be started once
    per process — a second test would fail on the manager rather than on anything it was
    testing. Nothing here needs it: these routes dispatch `mcp.call_tool` directly, which is
    the same entry point the protocol uses and the reason the two surfaces cannot disagree.
    """
    import http_app

    return TestClient(http_app.app)


def _tools(client):
    return sorted(client.get("/openapi.json").json()["paths"])


# ── the surface cannot drift from the protocol ──────────────────────────────

def test_every_tool_the_protocol_offers_has_a_path(client):
    """The anti-drift guard, and the reason this is generated rather than written. A tool added
    to `mcp_server` appears here with no second edit; a path written by hand would not, and the
    first person to notice would be a customer whose agent could not call it."""
    import asyncio

    import mcp_server

    offered = {tool.name for tool in asyncio.run(mcp_server.mcp.list_tools())}
    documented = {path.rsplit("/", 1)[-1] for path in _tools(client)}

    assert documented == offered, {
        "in the protocol and not the document": sorted(offered - documented),
        "in the document and not the protocol": sorted(documented - offered),
    }


def test_a_tool_is_described_by_its_own_docstring_and_schema(client):
    """Not by a summary written here. The docstring carries what a schema cannot — when to ask
    the user first, what a preview means, which argument is a claim rather than a fact — and an
    agent reading this document is the audience it was written for."""
    import asyncio

    import mcp_server

    tools = {t.name: t for t in asyncio.run(mcp_server.mcp.list_tools())}
    spec = client.get("/openapi.json").json()
    operation = spec["paths"]["/api/add_metrics"]["post"]

    assert operation["description"] == tools["add_metrics"].description
    assert operation["requestBody"]["content"]["application/json"]["schema"] == (
        tools["add_metrics"].input_schema)
    assert operation["operationId"] == "add_metrics"


def test_the_document_says_which_version_it_describes(client):
    spec = client.get("/openapi.json").json()

    assert spec["openapi"] == "3.1.0", (
        "the tools' schemas are JSON Schema 2020-12 — anyOf with null, enums — and 3.0 cannot "
        "carry them without a translation nobody would maintain")
    assert spec["info"]["version"] == config.VERSION


def test_the_preamble_carries_the_conventions_a_schema_cannot(client):
    """Three things an MCP client learns from the protocol and an HTTP caller has to be told:
    a refusal is a 400 with a readable message, a write previews until it is confirmed, and
    every response carries its own next step and the basis of each claim."""
    description = client.get("/openapi.json").json()["info"]["description"]

    for convention in ("confirm", "next_actions", "basis", "computed", "heuristic"):
        assert convention in description, convention
    # And the server's own instructions, rather than a second description of the product.
    import mcp_server

    assert mcp_server.INSTRUCTIONS.strip()[:200] in description


# ── calling a tool without MCP ──────────────────────────────────────────────

def test_a_tool_runs_and_stores_through_the_plain_http_path(client):
    made = client.post("/api/upload_campaign", json={
        "title": "Lima flagship", "market": "Peru", "status": "concluded",
        "detail": "A store opening.", "confirm": True})

    assert made.status_code == 200
    campaign_id = made.json()["campaign_id"]
    got = client.post("/api/get_campaign", json={"campaign_id": campaign_id})
    assert got.json()["title"] == "Lima flagship"


def test_a_write_still_previews_until_it_is_confirmed(client):
    """The consent step is the product's, not the protocol's, and it survives the change of
    transport because this route dispatches through the same entry point MCP uses. A façade
    that reached into `core` directly is exactly where that guarantee would have been lost."""
    made = client.post("/api/upload_campaign", json={
        "title": "Lima flagship", "market": "Peru", "status": "concluded",
        "detail": "A store opening.", "confirm": True}).json()

    preview = client.post("/api/add_metrics", json={
        "campaign_id": made["campaign_id"], "structured": {"footfall_uplift_pct": 3.1}})

    assert preview.status_code == 200
    assert preview.json()["preview"] is True
    assert not client.post("/api/get_campaign",
                           json={"campaign_id": made["campaign_id"]}).json()["metrics"]


def test_a_refusal_is_a_400_that_still_teaches(client):
    """`{"error": ...}` is how every tool here says no, and an HTTP caller reads the status
    before the body. The message is unchanged — and the forgiving-enum refusal keeps the
    fields that make it teach rather than merely fail."""
    refused = client.post("/api/upload_campaign", json={"title": "X", "status": "nonsense"})

    assert refused.status_code == 400
    body = refused.json()
    assert "must be one of" in body["error"]
    assert body["field"] == "status"
    assert "concluded" in body["valid"], (
        "a refusal that carries company is still a refusal — requiring `error` to be the only "
        "key reported this as a 200")


def test_a_refusal_with_no_company_is_also_a_400(client):
    refused = client.post("/api/get_campaign", json={"campaign_id": "corr_nope"})

    assert refused.status_code == 400
    assert list(refused.json()) == ["error"]


def test_an_unknown_tool_says_what_is_available(client):
    missing = client.post("/api/no_such_tool", json={})

    assert missing.status_code == 404
    assert "/openapi.json" in missing.json()["error"], (
        "a 404 that does not say where the list is makes the caller guess")


def test_a_body_that_is_not_an_object_is_refused_readably(client):
    assert client.post("/api/list_campaigns", content=b"[1, 2, 3]",
                       headers={"content-type": "application/json"}).status_code == 400
    assert client.post("/api/list_campaigns", content=b"{oops",
                       headers={"content-type": "application/json"}).status_code == 400
    # And no body at all is an empty argument list, not an error: plenty of tools take none.
    assert client.post("/api/list_campaigns").status_code == 200


# ── llms.txt ───────────────────────────────────────────────────────────────

def test_llms_txt_lists_every_tool_and_is_generated(client):
    """A hand-written summary of 65 tools is a second description of the product, and the first
    thing a second description does is go out of date."""
    text = client.get("/llms.txt").text
    listed = {line.split("`")[1].rsplit("/", 1)[-1] for line in text.splitlines()
              if line.startswith("- `POST")}

    assert listed == {path.rsplit("/", 1)[-1] for path in _tools(client)}
    assert "/openapi.json" in text
    assert "confirm" in text and "next_actions" in text


def test_the_generators_are_pure_functions_of_the_registry(conn):
    """No event loop, no server, no database: hand them two tools and check the output. The
    first version fetched the registry itself, which made it unusable from inside the request
    that wanted it — `asyncio.run()` cannot be called from a running loop."""
    import asyncio

    import mcp_server

    two = asyncio.run(mcp_server.mcp.list_tools())[:2]

    document = openapi.document(two, "https://example.test")
    assert sorted(document["paths"]) == sorted(f"/api/{t.name}" for t in two)
    assert document["servers"] == [{"url": "https://example.test"}]
    assert openapi.llms_txt(two).count("- `POST") == 2


def test_the_document_names_the_url_it_was_reached_on(client):
    """A tunnel is the ordinary deployment here, and a document naming `localhost` as the
    server is one nobody else can use."""
    spec = client.get("/openapi.json",
                      headers={"x-forwarded-proto": "https",
                               "x-forwarded-host": "briefs.example.test"}).json()

    assert spec["servers"] == [{"url": "https://briefs.example.test"}]


def test_the_description_is_readable_where_it_is_not_json(client):
    """`/llms.txt` is text. Served as JSON it renders as one escaped string, which is the
    opposite of the point."""
    response = client.get("/llms.txt")

    assert response.headers["content-type"].startswith("text/plain")
    with pytest.raises(json.JSONDecodeError):
        json.loads(response.text)
