"""
Pluggable authentication.

`authenticate()` is the single hook every network entry point calls: `http_app.GUARDED`
runs it in front of `/mcp`, `/api/<tool>` and `/upload` alike.
Which provider handles it is selected by CAMPAIGN_POC_AUTH_PROVIDER (default 'none' → allow
all, anonymous — fine for local/single-user).

The provider is an abstraction on purpose: this server should be deployable for any
organization without touching call sites. To add an identity provider (Azure AD / Entra ID,
Okta, Google Workspace, or any OIDC), implement AuthProvider and register it:

    class EntraProvider:
        async def authenticate(self, authorization_header):
            token = _bearer(authorization_header)
            claims = verify_entra_jwt(token)          # validate signature (JWKS), iss, aud, exp
            return Principal(subject=claims["oid"], anonymous=False,
                             email=claims.get("preferred_username"),
                             groups=tuple(claims.get("groups", ())))
    register("entra", EntraProvider())

then set CAMPAIGN_POC_AUTH_PROVIDER=entra. No other file changes. The Principal's `groups`
field is meant to carry IdP group claims so access scoping (per region/team) can key off them.

See docs/PRODUCTION-ROADMAP.md § Authentication for the full design and the Entra specifics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import config


@dataclass(frozen=True)
class Principal:
    subject: str
    anonymous: bool = True
    email: Optional[str] = None
    groups: tuple[str, ...] = ()   # IdP group claims → drive access scope later


ANONYMOUS = Principal(subject="anonymous", anonymous=True)


class AuthError(Exception):
    pass


@runtime_checkable
class AuthProvider(Protocol):
    """Any auth backend. Return a Principal on success; raise AuthError to reject.

    The AuthError's message is sent to the CALLER as the 401's `error` — say what they must
    fix ("token expired"), never how the server is built or why a signature check failed."""
    async def authenticate(self, authorization_header: Optional[str]) -> Principal: ...


class NoAuthProvider:
    """Default provider: no enforcement — every request is anonymous."""
    async def authenticate(self, authorization_header: Optional[str]) -> Principal:
        return ANONYMOUS


# Provider registry — name → instance. Add your own with register().
_PROVIDERS: dict[str, AuthProvider] = {"none": NoAuthProvider()}


def register(name: str, provider: AuthProvider) -> None:
    """Register an AuthProvider under a name selectable via CAMPAIGN_POC_AUTH_PROVIDER."""
    _PROVIDERS[name.lower()] = provider


def get_provider() -> AuthProvider:
    name = config.AUTH_PROVIDER
    provider = _PROVIDERS.get(name)
    if provider is None:
        raise AuthError(
            f"auth provider {name!r} is not registered. Register one with "
            f"auth.register({name!r}, <provider>), or set CAMPAIGN_POC_AUTH_PROVIDER=none. "
            f"See docs/PRODUCTION-ROADMAP.md."
        )
    return provider


async def authenticate(authorization_header: Optional[str]) -> Principal:
    """The single auth hook. Delegates to the configured provider (default: anonymous)."""
    return await get_provider().authenticate(authorization_header)


def _bearer(authorization_header: Optional[str]) -> str:
    """Helper for token providers: extract a Bearer token or raise AuthError."""
    if not authorization_header or not authorization_header.lower().startswith("bearer "):
        raise AuthError("missing or malformed Authorization: Bearer header")
    return authorization_header.split(" ", 1)[1].strip()


# Names that reach only this machine. Anything else is a network, and on a network with no
# provider configured "anonymous" stops meaning "the person sitting here". Compared through
# `ipaddress` for anything numeric, so every spelling of loopback — `127.0.0.2`, `::1`,
# `[::1]` — counts, rather than the three strings somebody remembered to list.
_LOOPBACK_NAMES = ("localhost",)


def is_loopback(host: str) -> bool:
    import ipaddress

    bare = host.strip().lower().strip("[]").rstrip(".")
    if bare in _LOOPBACK_NAMES:
        return True
    try:
        address = ipaddress.ip_address(bare)
    except ValueError:
        return False
    # ::ffff:127.0.0.1 is 127.0.0.1. Python 3.13+ says so itself; 3.12 — what CI and the
    # shipped builds run — does not, so this is not redundant there.
    mapped = getattr(address, "ipv4_mapped", None)
    return address.is_loopback or bool(mapped and mapped.is_loopback)


# Shown in `answers_to_hosts`; matched by `is_loopback`, so every spelling counts, not these.
LOOPBACK_HOSTS = ("localhost:*", "127.0.0.1:*", "[::1]:*")
# Binds that are not a name anybody reaches the server on.
_WILDCARD_BINDS = ("0.0.0.0", "::", "")
_DEFAULT_PORTS = ("80", "443")


_NAME = re.compile(r"[a-z0-9_.-]+")          # a DNS name or a dotted IPv4 address
_V6 = re.compile(r"[0-9a-f:.]+(%[a-z0-9_.-]+)?")   # an IPv6 literal, optional zone
_PORT = re.compile(r"[0-9]{1,5}")


def _is_v6(name: str) -> bool:
    """`_V6` is a character class, and `1::2::3` or `:` is made of the right characters
    without being an address — `urlsplit` raises on those (Python 3.11.4+), which crashed the
    server where it promised a warning. The class is the cheap filter; this is the check."""
    import ipaddress

    if not _V6.fullmatch(name):
        return False
    try:
        ipaddress.IPv6Address(name)
    except ValueError:
        return False
    return True             # ASCII: `\d` also matches ٨٠ and ８０


def _split_host(value: str, *, entry: bool = False) -> tuple[str, str | None]:
    """`Host`-header spelling → (name, port), or `("", None)` when it is not one.

    Compared the way DNS compares names: case-insensitive (RFC 9110 §4.2.3), a trailing dot
    the same name. A Host is a name and a port and NOTHING else: `localhost:8086@evil.example`
    split at its one colon was loopback with port `8086@evil.example`, and read back as a URL
    it is userinfo on `evil.example` — so the name must be a name, the port digits.

    For a request (`entry=False`) the default ports are the same address whether or not they
    are written. For an allowlist entry (`entry=True`) a port written is a port promised —
    `intra.corp:443` answers to `intra.corp` and `intra.corp:443`, not to `:9999` — and `*`
    means any."""
    value = value.strip(" \t").lower()   # HTTP's whitespace only: `.strip()` also eats NBSP
    if value.startswith("["):
        name, closed, rest = value[1:].partition("]")
        if not closed or (rest and not rest.startswith(":")) or not _is_v6(name):
            return "", None
        port = rest[1:] if rest else None
    elif value.count(":") == 1:
        name, port = value.split(":")
    elif value.count(":") > 1 and _is_v6(value):   # an unbracketed IPv6 literal
        name, port = value, None
    else:
        name, port = value, None
    name = name.rstrip(".")
    if port == "":
        port = None
    if port is not None and not (entry and port == "*") and not (
            _PORT.fullmatch(port) and 1 <= int(port) <= 65535):
        return "", None
    if not name or not (_NAME.fullmatch(name) or _is_v6(name)):
        return "", None
    if port is not None and port != "*":
        port = str(int(port))              # `08086` is port 8086, compared as a number
    if port in _DEFAULT_PORTS:
        port = "default" if entry else None
    return name, port


# THE shape of CAMPAIGN_POC_BASE_URL — carried verbatim by run.sh and run.ps1, which a test
# checks, because three readers with three ideas of a valid URL is how a launcher came to
# register a service the server then ignored. An optional http(s) scheme (a URL typed without
# one is https), a name or a bracketed IPv6 literal, an optional port, an optional path of
# plain characters. Nothing else: no userinfo (it would be published), no query or fragment,
# no whitespace, `%`, `&`, `^` or quotes (the launchers write it unquoted into a systemd unit
# and a `cmd /c` line). Plain ERE, so bash, .NET and Python read it the same.
BASE_URL_SHAPE = r"^([Hh][Tt][Tt][Pp][Ss]?://)?([A-Za-z0-9._-]+|\[[0-9A-Fa-f:.]+\])(:[0-9]{1,5})?(/[A-Za-z0-9._~/-]*)?$"


def public_base_url() -> tuple[str | None, str | None]:
    """`CAMPAIGN_POC_BASE_URL` → `(url, None)` rebuilt from its parts, or `(None, problem)`.

    Rebuilt, never echoed: `briefs` published as typed is a relative reference no client can
    resolve, and `HTTPS://Briefs.Example.COM:443/` is four spellings of `https://briefs.example.com`.
    Every reader — the allowlist, the generated documents, the startup warning — goes through
    this one function."""
    import config

    # Not stripped: the launchers refuse surrounding whitespace, and one rule means the server
    # does too rather than quietly accepting what they would not write.
    raw = config.PUBLIC_BASE_URL
    shape = re.fullmatch(BASE_URL_SHAPE[1:-1], raw)
    if shape is None:
        hint = ("; write an international name in its punycode form (xn--...), which is "
                "what browsers send" if any(ord(c) > 127 for c in raw) else "")
        return None, (f"CAMPAIGN_POC_BASE_URL={raw!r} is not a URL of the form "
                      f"https://name[:port][/path]{hint}")
    scheme_part, host_part, port_part, path_part = shape.groups()
    scheme = scheme_part[:-3].lower() if scheme_part else "https"
    name, _ = _split_host(host_part)
    port = port_part[1:] if port_part else None
    labels = [] if ":" in name else name.split(".")
    too_long = len(name) > 253 or any(len(label) > 63 for label in labels)   # RFC 1035
    segments = (path_part or "").split("/")[1:]
    if (not name or too_long or (port is not None and not 1 <= int(port) <= 65535)
            or any(not label or label[0] == "-" or label[-1] == "-" for label in labels)
            or any(seg in (".", "..") for seg in segments) or "" in segments[:-1]):
        return None, (f"CAMPAIGN_POC_BASE_URL={raw!r} does not name a usable host, port "
                      f"(1-65535) and path")
    if port is not None:
        port = str(int(port))
        if (scheme, port) in (("http", "80"), ("https", "443")):
            port = None
    host = f"[{name}]" if ":" in name else name
    return f"{scheme}://{host}{f':{port}' if port else ''}{(path_part or '').rstrip('/')}", None


def _public_name() -> str:
    """The host[:port] of `CAMPAIGN_POC_BASE_URL`, or "" when it has a problem."""
    from urllib.parse import urlsplit

    url, _ = public_base_url()
    return urlsplit(url).netloc if url else ""


def allowed_hosts() -> tuple[str, ...]:
    """Every Host the HTTP surface answers to: loopback in any spelling, the name in
    `CAMPAIGN_POC_BASE_URL` (the one the deployment already declares it is reached on), a
    specific bind address, and anything in `CAMPAIGN_POC_ALLOWED_HOSTS`. A name written
    without a port answers on any port; `name:port` on that one. Read at request time, so it
    is one list for the guard, `/healthz` and the banner."""
    import config

    hosts = list(LOOPBACK_HOSTS)
    bind = config.HTTP_HOST.strip()
    extra = (_public_name(), "" if bind in _WILDCARD_BINDS or is_loopback(bind) else bind,
             *config.ALLOWED_HOSTS)
    for entry in extra:
        name, port = _split_host(entry, entry=True) if entry else ("", None)
        if not name:
            continue
        written = entry.strip().lower().rsplit(":", 1)[-1] if port == "default" else port
        spelled = (f"[{name}]" if ":" in name else name) + (f":{written}" if written else "")
        if spelled not in hosts:
            hosts.append(spelled)
    return tuple(hosts)


def host_allowed(value: str | None) -> bool:
    """Whether a `Host` (or an Origin's host) is one this server answers to. The one
    comparison the guard, the Origin check and the generated document's `servers` all use."""
    if not value:
        return False
    name, port = _split_host(value)
    if not name:
        return False
    if is_loopback(name):
        return True
    for entry in allowed_hosts():
        allowed_name, allowed_port = _split_host(entry, entry=True)
        if allowed_name != name:
            continue
        if allowed_port in (None, "*", port) or (allowed_port == "default" and port is None):
            return True
    return False


def base_url_problem() -> str | None:
    """Why `CAMPAIGN_POC_BASE_URL` is not a usable URL, or None. Printed at startup: a value
    that names nothing (`http:`, `https://a, https://b`, `…:abc`, `u:p@host`) otherwise fails
    silently — every public request a 421, or a document advertising `http:` — and the only
    clue is a name missing from the `Answers to Host` list."""
    return public_base_url()[1]


def public_name_is_set() -> bool:
    """Whether `CAMPAIGN_POC_BASE_URL` names somewhere other than this machine — i.e. the
    operator has said what the server is called, and nothing a request carries overrides it."""
    name, _ = _split_host(_public_name())
    return bool(name) and not is_loopback(name)


# The routes a provider protects. Named once, because the statement below and the guard in
# `http_app` have to agree about them.
PROTECTED = ("/mcp", "/api/<tool>", "/upload")


def exposure() -> dict:
    """WHO CAN REACH THIS SERVER, as a fact rather than a warning.

    `authenticate` defaults to anonymous-allow, which is the right default for the product's
    ordinary shape: a marketer's own laptop, one user, nothing listening. It becomes a
    different claim the moment the HTTP surface binds to an address other than loopback —
    `CAMPAIGN_POC_HOST` defaults to `0.0.0.0`, so "anonymous" then means anyone who can route
    to the port, which on a corporate network is a lot of people.

    Computed from the BIND ADDRESS, and it says so: a tunnel or reverse proxy in front of a
    loopback bind is the documented way to reach Claude Web, and nothing in this process can
    see one. The statement names it rather than claiming "only this machine" for a server the
    internet can reach.

    Computed once here, and read by every surface that should say it: `/healthz`, the startup
    banner, and the generated OpenAPI preamble. A sentence about access written three times is
    three sentences to keep in step, and this one has to stay true when somebody sets a
    provider.

    NOT a `health_check` component, deliberately. That report's `ok` is the gate the installers
    refuse an install on — `ok = all(components)` — so a component that is False whenever auth
    is unconfigured would block every install of the default configuration. This is a statement
    about deployment, not a fault in the build.
    """
    import config

    provider = config.AUTH_PROVIDER
    anonymous = provider == "none"
    registered = provider in _PROVIDERS
    local_only = is_loopback(config.HTTP_HOST)
    open_to_network = anonymous and not local_only
    routes = ", ".join(PROTECTED)
    proxied = (" This is computed from the bind address: a tunnel or reverse proxy in front of "
               "it makes it reachable from wherever that proxy is, the internet included.")
    if open_to_network:
        said = (
            f"No authentication is configured and this server is bound to "
            f"{config.HTTP_HOST}, so anyone who can reach port {config.HTTP_PORT} can read "
            f"and write this library. That is a supported way to run it on a private "
            f"network — an internal address behind the firm's own perimeter — and it is not "
            f"safe on anything reachable from the internet. Restrict it to one machine with "
            f"CAMPAIGN_POC_HOST=127.0.0.1, or configure a provider with "
            f"CAMPAIGN_POC_AUTH_PROVIDER (see docs/PRODUCTION-ROADMAP.md § Authentication). "
            f"It answers only to the Host names listed under answers_to_hosts — a client "
            f"reaching it by another (a LAN address, a DNS name) gets a 421 until that name is "
            f"added to CAMPAIGN_POC_ALLOWED_HOSTS.")
    elif anonymous:
        said = (f"No authentication is configured, and this server is bound to "
                f"{config.HTTP_HOST}, so only this machine can reach it directly.{proxied} "
                f"Anything that can reach it through one can read and write this library.")
    elif not registered:
        said = (f"CAMPAIGN_POC_AUTH_PROVIDER names {provider!r}, which is not registered, so "
                f"every request to {routes} is refused until it is. Register it with "
                f"auth.register({provider!r}, <provider>) or set the variable to 'none'.")
    else:
        said = (f"Authentication is handled by the {provider!r} provider on {routes}, and "
                f"this server is bound to {config.HTTP_HOST}.{proxied}")
    return {
        "auth_provider": provider,
        "bound_to": config.HTTP_HOST,
        "open_to_the_network": open_to_network,
        "answers_to_hosts": list(allowed_hosts()),
        "basis": "computed",
        "what_it_means": said,
    }
