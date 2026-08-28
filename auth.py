"""
Authentication seam — no-op in V1, the single hook OAuth slots into later.

Claude Web / cowork custom connectors authenticate remote MCP servers via OAuth. That's not
implemented yet; every entry point already calls authenticate() so wiring OAuth in touches
only this file. CAMPAIGN_POC_AUTH_MODE=none (default) allows all requests as anonymous.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import config


@dataclass(frozen=True)
class Principal:
    subject: str
    anonymous: bool = True


ANONYMOUS = Principal(subject="anonymous", anonymous=True)


class AuthError(Exception):
    pass


async def authenticate(authorization_header: Optional[str]) -> Principal:
    if config.AUTH_MODE == "none":
        return ANONYMOUS
    # OAuth verification goes here (not implemented in V1).
    raise AuthError("CAMPAIGN_POC_AUTH_MODE=oauth set but OAuth is not implemented; use 'none' for now")
