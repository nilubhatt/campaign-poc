"""
Pluggable authentication.

`authenticate()` is the single hook every network entry point calls (HTTP + the MCP mount).
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
    """Any auth backend. Return a Principal on success; raise AuthError to reject."""
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
