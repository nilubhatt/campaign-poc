"""
What stands in front of every tool reached over HTTP: which Host and Origin a request may
carry, how large its body may be, who it must authenticate as, and what a caller is told when
the tool itself says no.

Found by the review of the OpenAPI surface (aa7efb6, c045147), and every one of them was a
place where `/api/<tool>` claimed to be the same surface as `/mcp` and was not:

- a schema rejection over `/mcp` arrives as a readable refusal; over `/api` it was a bare 500;
- `/mcp` refuses a request from another web page (Host/Origin, and a JSON content type that
  forces a browser preflight); `/api` accepted a `text/plain` POST from any page the user had
  open, which on a laptop bound to loopback is a write from the internet;
- `/mcp` caps a body at 4 MiB; `/api` buffered whatever it was sent;
- and `/mcp` never called the auth hook at all, while `exposure()` reported a configured
  provider as covering the server.

And one the review did not find, because it was already on main: `/mcp` accepted ONLY a
loopback Host, so the documented deployment — Claude Web through a tunnel or a public HTTPS
name — was refused with a 421 the moment the proxy forwarded the name it was reached on.
"""
import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

import auth
import config

# The Host a laptop install is reached on. Every client here uses it, because the default
# allowlist is loopback and TestClient's own default (`testserver`) is, correctly, refused.
LOCAL = "http://localhost:8086"

INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "t", "version": "1"}}}
MCP_HEADERS = {"accept": "application/json, text/event-stream"}


@pytest.fixture
def client(conn, monkeypatch):
    import http_app

    monkeypatch.setattr(config, "ALLOWED_HOSTS", ())
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", "http://localhost:8086")
    monkeypatch.setattr(config, "AUTH_PROVIDER", "none")
    return TestClient(http_app.app, base_url=LOCAL, raise_server_exceptions=False)


class _Refuse:
    async def authenticate(self, authorization_header):
        raise auth.AuthError("no token")


class _OnlyBearerOk:
    async def authenticate(self, authorization_header):
        if authorization_header != "Bearer ok":
            raise auth.AuthError("bad token")
        return auth.Principal(subject="u1", anonymous=False)


@pytest.fixture
def refusing_provider(monkeypatch):
    monkeypatch.setitem(auth._PROVIDERS, "refuse", _Refuse())
    monkeypatch.setattr(config, "AUTH_PROVIDER", "refuse")


# ── a refusal is a refusal, whoever does the refusing ──────────────────────

def test_an_argument_of_the_wrong_type_is_a_400_with_the_reason(client):
    """Over `/mcp` this is an `isError` result carrying the validation message. The REST route
    let `ToolError` escape and Starlette answered `Internal Server Error` — no message, a
    stack trace in the log, and a retry by any client that treats 5xx as transient."""
    refused = client.post("/api/get_campaign", json={"campaign_id": 123})

    assert refused.status_code == 400
    assert "campaign_id" in refused.json()["error"]


def test_a_missing_required_argument_is_a_400_with_the_reason(client):
    refused = client.post("/api/get_campaign", json={})

    assert refused.status_code == 400
    assert "campaign_id" in refused.json()["error"]


def test_a_tool_that_crashes_is_a_500_that_still_says_so(client, monkeypatch):
    """A crash is the server's fault and keeps its 500 — but as JSON carrying the message MCP
    would have carried, not a plain-text page a caller cannot parse. And no more than MCP
    carries: the protocol keeps the internal cause out of the reply, and a route that is
    unauthenticated by default is the last place to start handing it out."""
    import store

    def boom(*args, **kwargs):
        raise RuntimeError("the disk went away")

    monkeypatch.setattr(store, "get_campaign", boom)

    crashed = client.post("/api/get_campaign", json={"campaign_id": "camp_x"})

    assert crashed.status_code == 500
    assert crashed.json()["error"] == "Error executing tool get_campaign"
    assert "the disk went away" not in crashed.text


# ── a request from another web page ────────────────────────────────────────

def test_a_body_that_is_not_declared_json_is_refused(client):
    """`text/plain` is a CORS "simple request": a browser sends it from any page without
    asking first. Requiring `application/json` is what makes a cross-site write need a
    preflight this server never answers — the same rule `/mcp` applies."""
    sneaky = client.post("/api/upload_campaign",
                         content=json.dumps({"title": "csrf", "confirm": True}),
                         headers={"content-type": "text/plain"})

    assert sneaky.status_code == 400
    assert client.post("/api/list_campaigns").status_code == 200, (
        "no body at all is still an empty argument list — there is nothing to forge")


@pytest.mark.parametrize("path, kwargs", [
    ("/api/list_campaigns", {"json": {}}),
    ("/upload", {"files": {"file": ("a.png", b"x", "image/png")}}),
    ("/mcp", {"json": INIT, "headers": MCP_HEADERS}),
])
def test_another_sites_origin_is_refused_everywhere_a_write_can_land(client, path, kwargs):
    headers = {**kwargs.pop("headers", {}), "origin": "https://evil.example"}

    assert client.post(path, headers=headers, **kwargs).status_code == 403


@pytest.mark.parametrize("path, kwargs", [
    ("/api/list_campaigns", {"json": {}}),
    ("/upload", {"files": {"file": ("a.png", b"x", "image/png")}}),
    ("/mcp", {"json": INIT, "headers": MCP_HEADERS}),
])
def test_a_host_nobody_configured_is_refused_everywhere(client, path, kwargs):
    """DNS rebinding: a page on `evil.example` whose name now resolves to 127.0.0.1 arrives
    with `Host: evil.example`. Loopback is the default allowlist."""
    headers = {**kwargs.pop("headers", {}), "host": "evil.example"}

    assert client.post(path, headers=headers, **kwargs).status_code == 421


def test_the_same_machines_own_pages_are_not_refused(client):
    assert client.post("/api/list_campaigns", json={},
                       headers={"origin": "http://localhost:3000"}).status_code == 200


# ── the deployment the product documents ───────────────────────────────────

def test_the_public_name_in_base_url_is_accepted(client, monkeypatch):
    """`CAMPAIGN_POC_BASE_URL` already says what name this server is reached on. A second
    setting repeating it would be a second thing to keep in step."""
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", "https://briefs.example.com")

    reached = client.post("/api/list_campaigns", json={},
                          headers={"host": "briefs.example.com",
                                   "origin": "https://briefs.example.com"})

    assert reached.status_code == 200


def test_an_extra_name_can_be_allowed(client, monkeypatch):
    """A private-network install reached as `10.1.2.3:8086`, or a second DNS name."""
    monkeypatch.setattr(config, "ALLOWED_HOSTS", ("10.1.2.3:8086",))

    assert client.post("/api/list_campaigns", json={},
                       headers={"host": "10.1.2.3:8086"}).status_code == 200
    assert client.post("/api/list_campaigns", json={},
                       headers={"host": "10.1.2.4:8086"}).status_code == 421


def test_mcp_answers_on_the_public_name(conn, monkeypatch):
    """The one that was broken on main: `/mcp` refused every Host but loopback, so Claude Web
    through a tunnel or Front Door got a 421 the moment the proxy forwarded the public name.

    The only test that runs the lifespan, because MCP's session manager can be started once
    per process."""
    import http_app

    monkeypatch.setattr(config, "ALLOWED_HOSTS", ())
    monkeypatch.setattr(config, "AUTH_PROVIDER", "none")
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", "https://briefs.example.com")

    with TestClient(http_app.app, base_url="https://briefs.example.com") as client:
        reached = client.post("/mcp", json=INIT, headers=MCP_HEADERS)
        refused = client.post("/mcp", json=INIT,
                              headers={**MCP_HEADERS, "host": "elsewhere.example"})

    assert reached.status_code == 200
    assert refused.status_code == 421


# ── size ───────────────────────────────────────────────────────────────────

def test_an_oversized_body_is_refused_before_it_is_read(client):
    """`/mcp` caps a body at 4 MiB. `/api` buffered whatever it was sent, unauthenticated by
    default."""
    huge = b'{"x": "' + b"a" * (4 * 1024 * 1024 + 1) + b'"}'

    refused = client.post("/api/list_campaigns", content=huge,
                          headers={"content-type": "application/json"})

    assert refused.status_code == 413


# ── authentication is on every door, including the one that matters most ───

@pytest.mark.parametrize("path, kwargs", [
    ("/api/list_campaigns", {"json": {}}),
    ("/upload", {"files": {"file": ("a.png", b"x", "image/png")}}),
    ("/mcp", {"json": INIT, "headers": MCP_HEADERS}),
])
def test_a_rejecting_provider_rejects_every_route_that_reaches_the_library(
        client, refusing_provider, path, kwargs):
    """`/mcp` is the product's primary surface and it never called `auth.authenticate`, so a
    configured provider protected only the two side doors."""
    denied = client.post(path, **kwargs)

    assert denied.status_code == 401
    assert denied.json()["error"] == "no token"
    assert denied.headers["www-authenticate"].lower().startswith("bearer")


@pytest.mark.parametrize("path", ["/healthz", "/openapi.json", "/llms.txt"])
def test_what_carries_no_library_data_stays_readable(client, refusing_provider, path):
    """A caller cannot authenticate against a surface it has not been allowed to read, and an
    installer polls `/healthz` before anybody has a token."""
    assert client.get(path).status_code == 200


def test_an_accepting_provider_lets_the_caller_through(client, monkeypatch):
    monkeypatch.setitem(auth._PROVIDERS, "bearer", _OnlyBearerOk())
    monkeypatch.setattr(config, "AUTH_PROVIDER", "bearer")

    assert client.post("/api/list_campaigns", json={},
                       headers={"authorization": "Bearer ok"}).status_code == 200
    assert client.post("/api/list_campaigns", json={},
                       headers={"authorization": "Bearer no"}).status_code == 401


# ── the access statement tells the truth ───────────────────────────────────

@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "[::1]", "localhost",
                                  "LOCALHOST"])
def test_every_loopback_spelling_is_one_machine(monkeypatch, host):
    """Calling a loopback bind exposed is the boy who cried wolf."""
    monkeypatch.setattr(config, "AUTH_PROVIDER", "none")
    monkeypatch.setattr(config, "HTTP_HOST", host)

    assert auth.exposure()["open_to_the_network"] is False


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.1.2.3", "briefs.internal"])
def test_anything_else_is_a_network(monkeypatch, host):
    monkeypatch.setattr(config, "AUTH_PROVIDER", "none")
    monkeypatch.setattr(config, "HTTP_HOST", host)

    assert auth.exposure()["open_to_the_network"] is True


def test_the_statement_says_a_tunnel_changes_it(monkeypatch):
    """The bind address is all this function can see, and the documented way to reach Claude
    Web is a tunnel in front of a loopback bind — which "only this machine" would contradict."""
    monkeypatch.setattr(config, "AUTH_PROVIDER", "none")
    monkeypatch.setattr(config, "HTTP_HOST", "127.0.0.1")

    said = auth.exposure()["what_it_means"]

    assert "tunnel" in said and "reverse proxy" in said


def test_the_statement_names_the_hosts_it_answers_to(monkeypatch):
    monkeypatch.setattr(config, "ALLOWED_HOSTS", ("10.1.2.3:8086",))
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", "https://briefs.example.com")

    answers_to = auth.exposure()["answers_to_hosts"]

    assert "10.1.2.3:8086" in answers_to
    assert "briefs.example.com" in answers_to
    assert "localhost:*" in answers_to


def test_a_provider_that_is_not_registered_is_said_to_refuse_everything(monkeypatch):
    """Selecting a provider nobody registered does not open the server — every request that
    reaches the library is refused — and the statement used to say it was "handled"."""
    monkeypatch.setattr(config, "AUTH_PROVIDER", "entra")
    monkeypatch.setattr(config, "HTTP_HOST", "0.0.0.0")

    access = auth.exposure()

    assert access["open_to_the_network"] is False
    assert "not registered" in access["what_it_means"]
    assert "refused" in access["what_it_means"]


def test_a_registered_provider_is_said_to_cover_mcp(monkeypatch):
    monkeypatch.setitem(auth._PROVIDERS, "refuse", _Refuse())
    monkeypatch.setattr(config, "AUTH_PROVIDER", "refuse")

    said = auth.exposure()["what_it_means"]

    assert "/mcp" in said and "/api" in said and "/upload" in said


# ── the subset filter ──────────────────────────────────────────────────────

def test_a_filter_of_only_commas_is_the_whole_surface(client):
    """`?tools=,` is a URL built from an empty list, exactly like `?tools=`. It described
    zero operations."""
    assert len(client.get("/openapi.json?tools=,").json()["paths"]) == 65


def test_a_name_asked_for_twice_is_described_once_in_both_documents(client):
    text = client.get("/llms.txt?tools=gaps,gaps").text

    assert text.count("- `POST") == 1


def test_a_repeated_parameter_is_not_silently_narrowed_to_the_last(client):
    paths = client.get("/openapi.json?tools=gaps&tools=coverage").json()["paths"]

    assert sorted(paths) == ["/api/coverage", "/api/gaps"]


# ── round 2: every method, every spelling, every proxy ──────────────────────

@pytest.mark.parametrize("method", ["get", "delete"])
def test_mcp_session_methods_are_guarded_too(client, refusing_provider, method):
    """`/mcp` is not only POST: GET attaches to the server's notification stream and DELETE
    ends a session. A guard scoped to writes would leave both open — a page on any origin
    could end somebody else's session — and every test above uses POST."""
    assert getattr(client, method)("/mcp", headers=MCP_HEADERS).status_code == 401


@pytest.mark.parametrize("method", ["get", "delete"])
def test_mcp_session_methods_check_the_host(client, method):
    assert getattr(client, method)("/mcp", headers={**MCP_HEADERS,
                                                    "host": "evil.example"}).status_code == 421


@pytest.mark.parametrize("host", ["LOCALHOST:8086", "Briefs.Example.COM", "briefs.example.com.",
                                  "briefs.example.com:443"])
def test_a_host_is_compared_the_way_dns_compares_it(client, monkeypatch, host):
    """Case-insensitive (RFC 9110 §4.2.3), a trailing dot is the same name, and the default
    port is the same address whether or not it is spelled."""
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", "https://briefs.example.com")

    assert client.post("/api/list_campaigns", json={}, headers={"host": host}).status_code == 200


@pytest.mark.parametrize("base_url", ["https://Briefs.Example.com:443", "briefs.example.com"])
def test_the_public_name_is_understood_however_it_was_written(client, monkeypatch, base_url):
    """`https://name:443` is the name every browser sends as `Host: name`, and a URL typed
    without its scheme is still a name — it used to add nothing at all, and every request got
    a 421."""
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", base_url)

    assert client.post("/api/list_campaigns", json={},
                       headers={"host": "briefs.example.com"}).status_code == 200


@pytest.mark.parametrize("host", ["127.0.0.2:8086", "[0:0:0:0:0:0:0:1]:8086",
                                  "[::ffff:127.0.0.1]:8086"])
def test_every_loopback_spelling_is_answered(client, host):
    """`exposure()` already calls these one machine; the guard refused them."""
    assert client.post("/api/list_campaigns", json={}, headers={"host": host}).status_code == 200


def test_a_specific_bind_address_is_answered_on(client, monkeypatch):
    """Bound to one LAN address, the server is reached on that address. The wildcard binds
    (`0.0.0.0`, `::`) are not a name anybody can use, so they add nothing."""
    monkeypatch.setattr(config, "HTTP_HOST", "10.1.2.3")
    assert client.post("/api/list_campaigns", json={},
                       headers={"host": "10.1.2.3:8086"}).status_code == 200

    monkeypatch.setattr(config, "HTTP_HOST", "0.0.0.0")
    assert client.post("/api/list_campaigns", json={},
                       headers={"host": "0.0.0.0:8086"}).status_code == 421


def test_a_network_bind_says_how_to_name_itself(monkeypatch):
    """The supported private-network deployment answers only to names it was told; the
    statement that calls it open is the place an operator is reading when they need that."""
    monkeypatch.setattr(config, "AUTH_PROVIDER", "none")
    monkeypatch.setattr(config, "HTTP_HOST", "0.0.0.0")

    assert "CAMPAIGN_POC_ALLOWED_HOSTS" in auth.exposure()["what_it_means"]


def test_an_unregistered_provider_does_not_tell_the_caller_how_the_server_is_built(
        client, monkeypatch):
    """Operator guidance — which variable, which function to call — is for the log and the
    banner, not for an anonymous caller on a public name."""
    monkeypatch.setattr(config, "AUTH_PROVIDER", "entra")

    denied = client.post("/api/list_campaigns", json={})

    assert denied.status_code == 401
    assert "register" not in denied.text.lower()
    assert "CAMPAIGN_POC" not in denied.text


def test_an_oversized_upload_is_refused_by_the_guard(client, monkeypatch):
    """The handler's own check reads the whole file first; the guard's cap is what stops an
    upload ten times the size from being buffered at all."""
    monkeypatch.setattr(config, "MAX_ASSET_BYTES", 1024)

    big = client.post("/upload", files={"file": ("a.png", b"x" * (2 * 1024 * 1024), "image/png")})

    assert big.status_code == 413
    assert big.text != "" and "file exceeds" not in big.text, "refused before the handler ran"


def test_a_chunked_body_must_still_say_it_is_json(client):
    """No Content-Length is not no body."""
    def chunks():
        yield b'{"title": "csrf", "confirm": true}'

    sneaky = client.post("/api/upload_campaign", content=chunks(),
                         headers={"content-type": "text/plain"})

    assert sneaky.status_code == 400


def test_the_document_does_not_name_a_server_it_was_told_by_a_stranger(client, monkeypatch):
    """`servers[].url` came from `X-Forwarded-Host` verbatim. Behind a caching proxy, one
    request naming `evil.example` would hand every later GPT Action an attacker's host."""
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", "https://briefs.example.com")

    spec = client.get("/openapi.json", headers={"x-forwarded-host": "evil.example"}).json()

    assert spec["servers"] == [{"url": "https://briefs.example.com"}]


# ── round 3: a Host is a name and a port, nothing else ──────────────────────

@pytest.mark.parametrize("forged", ["localhost:8086@evil.example",
                                    "briefs.example.com:443@evil.example",
                                    "[::1]evil.example", "localhost:8086/evil",
                                    "localhost:abc"])
def test_a_host_that_is_not_a_name_and_a_port_is_refused(client, monkeypatch, forged):
    """`localhost:8086@evil.example` split at its one colon was `localhost` with port
    `8086@evil.example` — loopback, so allowed — and read back as a URL it is userinfo
    `localhost:8086` on the host `evil.example`."""
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", "https://briefs.example.com")

    assert not auth.host_allowed(forged)
    assert client.post("/api/list_campaigns", json={},
                       headers={"host": forged}).status_code == 421


@pytest.mark.parametrize("forged", ["localhost:8086@evil.example", "127.0.0.1:1",
                                    "briefs.example.com:1"])
def test_a_configured_public_url_is_the_only_one_advertised(client, monkeypatch, forged):
    """With `CAMPAIGN_POC_BASE_URL` set, the operator has said what the server is called.
    Reflecting a forwarded name instead — loopback, a dead port, a forged one — is a cache
    away from every later reader's document."""
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", "https://briefs.example.com")

    for path in ("/openapi.json", "/llms.txt"):
        got = client.get(path, headers={"x-forwarded-host": forged,
                                        "x-forwarded-proto": "http"})
        assert "evil" not in got.text and "127.0.0.1:1" not in got.text
        assert ":1\"" not in got.text and "briefs.example.com:1" not in got.text
    spec = client.get("/openapi.json", headers={"x-forwarded-host": forged}).json()
    assert spec["servers"] == [{"url": "https://briefs.example.com"}]


def test_the_documents_are_not_cached(client):
    for path in ("/openapi.json", "/llms.txt"):
        assert "no-store" in client.get(path).headers.get("cache-control", "")


def test_an_explicit_default_port_on_an_allowed_name_is_that_port_only(monkeypatch):
    """`intra.corp:443` in the list promised that port; it was normalised to the bare name,
    which answers on any."""
    monkeypatch.setattr(config, "ALLOWED_HOSTS", ("intra.corp:443",))

    assert auth.host_allowed("intra.corp")
    assert auth.host_allowed("intra.corp:443")
    assert not auth.host_allowed("intra.corp:9999")


def test_two_host_headers_are_refused(client):
    """RFC 9110 §7.2. A front proxy and this guard picking different ones is how a check is
    passed with one name and the request routed on the other."""
    import httpx

    request = client.build_request("POST", "/api/list_campaigns", json={})
    request.headers = httpx.Headers([*[(k, v) for k, v in request.headers.multi_items()
                                       if k.lower() != "host"],
                                     ("host", "localhost:8086"), ("host", "evil.example")])
    assert client.send(request).status_code == 400


@pytest.mark.parametrize("launcher, into_the_service, into_the_foreground", [
    ("run.sh", "Environment=CAMPAIGN_POC_BASE_URL=$BASE_URL", 'export CAMPAIGN_POC_BASE_URL="$BASE_URL"'),
    ("run.ps1", "set CAMPAIGN_POC_BASE_URL=$BaseUrl&& ", "$env:CAMPAIGN_POC_BASE_URL = $BaseUrl"),
])
def test_a_background_service_can_carry_the_public_url(launcher, into_the_service,
                                                       into_the_foreground):
    """A systemd --user unit and a scheduled task do not inherit the shell, so the variable a
    tunnel deployment needs has to be something the launcher writes into the service — the
    lines that do it, not a mention in a comment."""
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / launcher).read_text()
    assert into_the_service in text
    assert into_the_foreground in text


@pytest.mark.parametrize("forged", ["evil@localhost", "[::1]x", "[::1@evil]:80", "local host"])
def test_the_parts_of_a_host_are_checked_one_by_one(forged):
    """Each check stands alone: a name that is not a name, junk after an IPv6 literal, an
    IPv6 literal that is not one."""
    assert auth._split_host(forged) == ("", None)
    assert not auth.host_allowed(forged)


def test_a_reflected_name_is_rebuilt_from_its_parts(client, monkeypatch):
    """With no public URL configured the name a caller reached is reflected — as the name and
    port it parsed to, never as the bytes it arrived as."""
    monkeypatch.setattr(config, "ALLOWED_HOSTS", ("briefs.example.test",))

    spec = client.get("/openapi.json", headers={"x-forwarded-host": "Briefs.Example.Test.:443",
                                                "x-forwarded-proto": "https"}).json()

    assert spec["servers"] == [{"url": "https://briefs.example.test"}]


# ── round 4: a port is a port, a URL is a URL ───────────────────────────────

@pytest.mark.parametrize("host", ["localhost:0", "localhost:99999", "briefs.example.com:٨٠"])
def test_a_port_outside_the_range_is_not_a_port(host):
    assert auth._split_host(host) == ("", None)


# ── round 5: one shape for the public URL, everywhere it is read ────────────
#
# The launchers and the server each had their own idea of a valid CAMPAIGN_POC_BASE_URL, and
# the server published the raw string — so `http://` (which config's rstrip turns into
# `http:`) passed as a host named `http`, credentials in `https://u:p@host` were published, and
# a URL one launcher accepted the other refused. Now: one pattern (`auth.BASE_URL_SHAPE`,
# carried verbatim by both launchers), and the server publishes the URL rebuilt from its parts.

@pytest.mark.parametrize("base_url", [
    "http:",                            # what `CAMPAIGN_POC_BASE_URL=http://` becomes in config
    "https:/briefs.example.com",        # one slash short
    "ftp://briefs.example.com",
    "https://briefs.example.com:abc",
    "https://a, https://b",
    "https://u:p@briefs.example.com",   # credentials would be published to anyone
    "https://briefs.example.com?x=1",
    "https://briefs.example.com#f",
    "https://briefs.example.com:99999",
    "https://brief s.example.com",
])
def test_a_base_url_that_is_not_one_is_named_at_startup(monkeypatch, base_url):
    """The operator believes they named the server; every public request is a 421 or the
    document advertises `http:`. Said where they are looking — not left to be inferred from
    a name missing from a list."""
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", base_url)
    monkeypatch.setattr(config, "ALLOWED_HOSTS", ())
    monkeypatch.setattr(config, "HTTP_HOST", "0.0.0.0")

    assert auth.base_url_problem() is not None
    assert not auth.public_name_is_set()
    assert auth.allowed_hosts() == auth.LOOPBACK_HOSTS, "and nothing it half-names is answered"


@pytest.mark.parametrize("base_url, published", [
    ("http://localhost:8086", "http://localhost:8086"),
    ("https://briefs.example.com", "https://briefs.example.com"),
    ("https://briefs.example.com:8443/", "https://briefs.example.com:8443"),
    ("HTTPS://Briefs.Example.COM:443/", "https://briefs.example.com"),
    ("http://briefs.example.com:443", "http://briefs.example.com:443"),
    ("https://briefs.example.com:08443", "https://briefs.example.com:8443"),
    ("https://briefs.example.com/prefix/", "https://briefs.example.com/prefix"),
    ("https://[2001:db8::1]:8443", "https://[2001:db8::1]:8443"),
    ("briefs", "https://briefs"),       # a single-label intranet name, typed without a scheme
    ("http://10.1.2.3:8086", "http://10.1.2.3:8086"),
])
def test_a_real_base_url_is_published_rebuilt_from_its_parts(client, monkeypatch, base_url,
                                                            published):
    """Never the raw string: `briefs` published as-is is a relative reference no client can
    resolve, and `HTTPS://Briefs…:443/` is four spellings of one URL."""
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", base_url)

    assert auth.base_url_problem() is None
    assert auth.public_base_url() == (published, None)
    if auth.public_name_is_set():
        assert client.get("/openapi.json").json()["servers"] == [{"url": published}]


def test_a_url_with_a_problem_is_never_advertised(client, monkeypatch):
    """The startup warning promises it is ignored; the fallback for a foreign Host used to
    hand it out anyway."""
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", "ftp://x")

    spec = client.get("/openapi.json", headers={"host": "evil.example"}).json()

    assert spec["servers"] == [{"url": "/"}]


def test_a_port_is_compared_as_a_number(monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", "https://briefs.example.com:08086")

    assert auth.host_allowed("briefs.example.com:8086")
    assert auth.host_allowed("briefs.example.com:08086")


@pytest.mark.parametrize("launcher", ["run.sh", "run.ps1"])
def test_the_launchers_carry_the_servers_own_shape(launcher):
    """One pattern, three readers. A launcher that accepts what the server rejects registers a
    service that starts and ignores the URL; one that rejects what the server accepts blocks a
    working deployment."""
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / launcher).read_text()
    assert auth.BASE_URL_SHAPE in text


def _run_sh(tmp_path, *args, env_base_url=None):
    """run.sh in an empty directory: no venv, so a value the check lets through stops at
    "Not set up" (exit 1) and nothing is installed, started or removed. HOME is the temp dir
    so the --stop path cannot touch a real unit."""
    import os
    import shutil
    import subprocess
    from pathlib import Path

    if shutil.which("bash") is None:
        pytest.skip("no bash on this platform (run.ps1 cannot be executed here either)")
    shutil.copy(Path(__file__).resolve().parent.parent / "run.sh", tmp_path / "run.sh")
    env = {k: v for k, v in os.environ.items() if k != "CAMPAIGN_POC_BASE_URL"}
    env["HOME"] = str(tmp_path)
    if env_base_url is not None:
        env["CAMPAIGN_POC_BASE_URL"] = env_base_url
    return subprocess.run(["bash", str(tmp_path / "run.sh"), *args], env=env, cwd=tmp_path,
                          capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("bad", ["https://a b", "https://x.example/?a=1&b=2",
                                 "https://briefs.example.com:99999", "https://u:p@host"])
def test_run_sh_refuses_a_base_url_before_writing_anything(tmp_path, bad):
    ran = _run_sh(tmp_path, "--base-url", bad)

    assert ran.returncode == 2, ran.stdout + ran.stderr
    assert "--base-url" in ran.stdout


@pytest.mark.parametrize("good", ["https://briefs.example.com", "HTTPS://host:8443/prefix",
                                  "https://[2001:db8::1]:8443", "briefs"])
def test_run_sh_accepts_what_the_server_accepts(tmp_path, monkeypatch, good):
    ran = _run_sh(tmp_path, "--base-url", good)
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", good)

    assert ran.returncode == 1 and "Not set up" in ran.stdout, ran.stdout + ran.stderr
    assert auth.base_url_problem() is None


def test_removing_a_service_is_not_blocked_by_a_bad_url_in_the_shell(tmp_path):
    """Removing a service needs no URL, and a malformed one left in the environment must not
    stand between the user and `--stop`."""
    ran = _run_sh(tmp_path, "--service", "--stop", env_base_url="https://host/?bad")

    assert ran.returncode != 2, ran.stdout


def test_a_bad_url_from_the_environment_is_named_as_such(tmp_path):
    ran = _run_sh(tmp_path, env_base_url="https://a b")

    assert ran.returncode == 2
    assert "CAMPAIGN_POC_BASE_URL" in ran.stdout, "not blamed on a flag the user did not pass"


# ── round 6: what the pattern lets through, the server still has to understand ──

@pytest.mark.parametrize("base_url", ["https://[:]", "https://[1::2::3]", "https://[1:2]",
                                      "https://[12345::1]", "https://[2001:db8:::1]",
                                      "https://-", "https://a..b", "https://-host.example",
                                      "https://host-.example", "https://host/../x",
                                      "https://host//double", " https://host", "https://host "])
def test_a_shape_that_parses_but_names_nothing_is_a_problem_not_a_crash(client, monkeypatch,
                                                                       base_url):
    """`[1::2::3]` matches the bracket class and is no address: Python 3.12's `urlsplit`
    raises on it, so the server crashed at startup — and on every non-loopback request — where
    it promised a WARNING. Whitespace is refused here as the launchers refuse it: one rule."""
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", base_url)

    assert auth.base_url_problem() is not None
    auth.allowed_hosts()                                    # does not raise
    assert client.get("/healthz").status_code == 200
    assert client.post("/api/list_campaigns", json={},
                       headers={"host": "briefs.example.com"}).status_code == 421


def test_stop_without_service_does_not_start_a_server_with_an_unchecked_url(tmp_path):
    ran = _run_sh(tmp_path, "--stop", "--base-url", "https://a b")

    assert ran.returncode == 2, ran.stdout


def test_an_empty_flag_does_not_wave_through_the_environment(tmp_path):
    """`--base-url ''` skipped the check and the export, and the server still inherited the
    caller's CAMPAIGN_POC_BASE_URL. Empty means none: it is unset for the child."""
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "run.sh").read_text()
    assert "unset CAMPAIGN_POC_BASE_URL" in text
    ps1 = (Path(__file__).resolve().parent.parent / "run.ps1").read_text()
    assert "Remove-Item Env:CAMPAIGN_POC_BASE_URL" in ps1


def test_run_sh_does_not_depend_on_the_callers_locale_or_bash_version():
    """glibc ranges like `[A-Za-z]` follow collation order in a UTF-8 locale, and whether an
    unmatched BASH_REMATCH group is unset under `set -u` varies by bash version."""
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "run.sh").read_text()
    assert "export LC_ALL=C" in text
    assert "${BASH_REMATCH[3]:-}" in text


def test_run_ps1_refuses_a_line_break_dotnet_dollar_would_allow():
    """In .NET `$` also matches before a final newline, and the value is written into a
    `cmd /c` line."""
    from pathlib import Path

    assert "[\\r\\n]" in (Path(__file__).resolve().parent.parent / "run.ps1").read_text()


# ── round 7: the last caller input that reached urlsplit ────────────────────

@pytest.mark.parametrize("origin", ["http://[", "http://]", "http://[1::2::3]", "http://[::1]x",
                                    "http://[1.2.3.4]", "javascript:alert(1)",
                                    "http://localhost:3000/path", "ftp://localhost:3000"])
def test_an_origin_that_is_not_one_is_a_403_not_a_crash(client, origin):
    """`urlsplit` raises on a malformed bracket — before authentication, so any caller could
    turn the promised 403 into a 500 and a logged traceback."""
    assert client.post("/api/list_campaigns", json={},
                       headers={"origin": origin}).status_code == 403


@pytest.mark.parametrize("base_url", ["https://" + "a" * 64 + ".example",
                                      "https://" + ".".join(["abcdefghi"] * 26)])
def test_a_name_longer_than_dns_allows_is_a_problem(monkeypatch, base_url):
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", base_url)

    assert auth.base_url_problem() is not None


def test_the_launchers_say_the_server_has_the_last_word():
    """The launchers filter for what is safe to WRITE into a unit or a cmd line; the server
    decides what is usable. Claiming they agree exactly was false the moment the server
    learned DNS label rules, so the comments say which is which."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for launcher in ("run.sh", "run.ps1"):
        assert "WARNING has the last word" in (root / launcher).read_text()


# ── round 8 ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("byte", ["\xa0", "\x85", "\r"])
def test_only_http_whitespace_is_trimmed_from_a_host(byte):
    """`str.strip()` also strips Unicode whitespace — NBSP and NEL, which a latin-1 decoded
    header can carry — so `localhost\\xa0` was loopback. HTTP's own whitespace is space and
    tab."""
    assert not auth.host_allowed("localhost" + byte)
    assert auth.host_allowed(" localhost\t")


def test_a_non_ascii_public_name_is_told_to_use_punycode(monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", "https://bücher.example")

    assert "punycode" in auth.base_url_problem()


# ── the installers' services can carry the public URL too ───────────────────
#
# A systemd unit and a scheduled task inherit no shell, and `schtasks /TR` cannot set an
# environment variable without a `cmd /c` wrapper whose quoting is its own hazard. So the URL
# is an ARGUMENT to `serve`, and each installer appends it to the service command.

def test_serve_takes_the_public_url_as_an_argument(monkeypatch):
    import os
    import sys
    import types

    import main

    seen = {}
    fake = types.ModuleType("http_app")
    fake.main = lambda: seen.update(url=os.environ.get("CAMPAIGN_POC_BASE_URL"))
    monkeypatch.setitem(sys.modules, "http_app", fake)
    monkeypatch.setenv("CAMPAIGN_POC_BASE_URL", "https://from-the-shell.example")   # restored
    monkeypatch.setattr(sys, "argv", ["campaign-intelligence", "serve",
                                      "--base-url", "https://briefs.example.com"])

    main.main()

    assert seen["url"] == "https://briefs.example.com", "the argument wins over the shell"


def _linux_bundle(tmp_path):
    """The Linux installer laid out the way a release ships it, around a stub binary that
    passes every step, with a stub `systemctl` first on PATH so no real user session is
    touched."""
    import os

    root = Path(__file__).resolve().parent.parent
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "install.sh").write_bytes((root / "installer/linux/install.sh").read_bytes())
    stub = bundle / "campaign-intelligence"
    stub.write_text("#!/usr/bin/env bash\nexit 0\n")
    stub.chmod(0o755)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    systemctl = fake_bin / "systemctl"
    systemctl.write_text("#!/usr/bin/env bash\nexit 0\n")
    systemctl.chmod(0o755)
    env = {**os.environ, "HOME": str(tmp_path / "home"),
           "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}
    return bundle, env


def _install(tmp_path, *args):
    import shutil
    import subprocess

    if shutil.which("bash") is None:
        pytest.skip("no bash on this platform")
    bundle, env = _linux_bundle(tmp_path)
    return subprocess.run(["bash", str(bundle / "install.sh"), "--no-ollama", *args],
                          cwd=bundle, env=env, capture_output=True, text=True, timeout=120)


def test_the_linux_installer_writes_the_public_url_into_its_service(tmp_path):
    ran = _install(tmp_path, "--service", "--base-url", "https://briefs.example.com")

    assert ran.returncode == 0, ran.stdout + ran.stderr
    unit = (tmp_path / "home/.config/systemd/user/campaign-intelligence.service").read_text()
    assert "serve --base-url https://briefs.example.com\n" in unit


def test_the_linux_installer_writes_no_url_when_none_was_given(tmp_path):
    ran = _install(tmp_path, "--service")

    assert ran.returncode == 0, ran.stdout + ran.stderr
    unit = (tmp_path / "home/.config/systemd/user/campaign-intelligence.service").read_text()
    assert "--base-url" not in unit


@pytest.mark.parametrize("bad", ["https://a b", "https://x/?a=1&b=2", "https://h:99999",
                                 "https://u:p@h"])
def test_the_linux_installer_refuses_a_bad_url_before_installing_anything(tmp_path, bad):
    ran = _install(tmp_path, "--service", "--base-url", bad)

    assert ran.returncode == 2, ran.stdout + ran.stderr
    assert "must look like" in ran.stdout, "refused for its shape, not as an unknown flag"
    assert not (tmp_path / "home/.local/share/campaign-intelligence").exists()


def test_the_linux_installer_says_a_url_without_a_service_does_nothing(tmp_path):
    ran = _install(tmp_path, "--base-url", "https://briefs.example.com")

    assert ran.returncode == 2
    assert "--service" in ran.stdout


def test_the_linux_installer_carries_the_servers_own_shape():
    root = Path(__file__).resolve().parent.parent
    assert auth.BASE_URL_SHAPE in (root / "installer/linux/install.sh").read_text()


def test_the_windows_installer_can_carry_the_public_url():
    """Not compilable here (CI's Windows job compiles it): the URL is asked for on a page
    shown only when the background service is chosen, taken from /BASEURL= on a silent
    install, checked character by character (Pascal Script has no regex), and appended to
    the scheduled task's `serve` command."""
    root = Path(__file__).resolve().parent.parent
    iss = (root / "installer/windows/campaign-intelligence.iss").read_text()

    assert "/BASEURL=" in iss or "'BASEURL'" in iss
    assert "{code:BaseUrlArg}" in iss
    assert "function SafeBaseUrl" in iss
    assert "--base-url" in iss


# ── round 9 ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("args", [("--service", "--base-url"), ("--base-url",),
                                  ("--service", "--base-url", "--no-ollama"),
                                  ("--service", "--base-url", "-x")])
def test_the_linux_installer_names_a_missing_url_value(tmp_path, args):
    """`--base-url` last made `shift` fail under `set -e`: exit 1, no message. And a value
    starting with `-` is the next flag swallowed — it matched the pattern and registered a
    service whose `serve --base-url --no-ollama` argparse refuses on every restart."""
    ran = _install(tmp_path, *args)

    assert ran.returncode == 2, ran.stdout + ran.stderr
    assert "--base-url needs a value" in ran.stdout
    assert not (tmp_path / "home/.local/share/campaign-intelligence").exists()


def test_run_sh_names_a_missing_url_value(tmp_path):
    ran = _run_sh(tmp_path, "--base-url", "--service")

    assert ran.returncode == 2
    assert "--base-url needs a value" in ran.stdout


def test_an_empty_base_url_argument_means_none(monkeypatch):
    """As `run.sh --base-url ''` already does: empty is "none", not a URL to warn about."""
    import os
    import sys
    import types

    import main

    seen = {}
    fake = types.ModuleType("http_app")
    fake.main = lambda: seen.update(url=os.environ.get("CAMPAIGN_POC_BASE_URL"))
    monkeypatch.setitem(sys.modules, "http_app", fake)
    monkeypatch.setenv("CAMPAIGN_POC_BASE_URL", "https://from-the-shell.example")
    monkeypatch.setattr(sys, "argv", ["campaign-intelligence", "serve", "--base-url", ""])

    main.main()

    assert seen["url"] is None


def test_the_windows_installer_never_shows_a_dialog_on_a_silent_install():
    """Inno runs NextButtonClick on a silent install too (ClickThroughPages), and a [Code]
    MsgBox is shown under /SILENT — so a bad /BASEURL= stopped an unattended install on a
    modal dialog. Silent: log it and refuse, with no dialog."""
    iss = (Path(__file__).resolve().parent.parent
           / "installer/windows/campaign-intelligence.iss").read_text()
    click = iss[iss.index("function NextButtonClick"):]
    click = click[:click.index("\nend;") + 5]

    assert "WizardSilent" in click
    assert click.index("if WizardSilent") < click.index("MsgBox(")
    assert "A silent install never shows the page" not in iss, "that comment was false"


def test_the_windows_installer_caps_the_url_for_schtasks():
    """`schtasks /TR` is limited to 261 characters and [Run] ignores its exit code, so a long
    URL registered no task and said nothing. The cap is measured from the install path, which
    holds the user's name: a fixed 150 overflowed for a 13-character username."""
    iss = (Path(__file__).resolve().parent.parent
           / "installer/windows/campaign-intelligence.iss").read_text()
    safe = iss[iss.index("function SafeBaseUrl"):]
    safe = safe[:safe.index("\nend;") + 5]
    cap = iss[iss.index("function MaxBaseUrlLength"):]
    cap = cap[:cap.index("\nend;") + 5]

    assert "Length(Value) <= MaxBaseUrlLength" in safe
    assert "261 - Length(ExpandConstant('{app}\\{#AppExe}')) - 20" in cap
    # The 20: the quote, the path, the quote, then ` serve --base-url ` before the URL.
    assert len("'" + "'" + " serve --base-url ") == 20
