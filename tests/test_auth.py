import pytest

import auth
import config


@pytest.mark.anyio
async def test_default_provider_is_anonymous(monkeypatch):
    monkeypatch.setattr(config, "AUTH_PROVIDER", "none")
    p = await auth.authenticate(None)
    assert p.anonymous is True
    assert p.subject == "anonymous"


@pytest.mark.anyio
async def test_unregistered_provider_raises_auth_error(monkeypatch):
    monkeypatch.setattr(config, "AUTH_PROVIDER", "does-not-exist")
    with pytest.raises(auth.AuthError):
        await auth.authenticate("Bearer whatever")


@pytest.mark.anyio
async def test_register_custom_provider_and_select_it(monkeypatch):
    class FakeProvider:
        async def authenticate(self, authorization_header):
            if authorization_header != "Bearer good-token":
                raise auth.AuthError("bad token")
            return auth.Principal(subject="user-1", anonymous=False, email="u@example.com",
                                  groups=("region-apac",))

    auth.register("fake", FakeProvider())
    monkeypatch.setattr(config, "AUTH_PROVIDER", "fake")

    p = await auth.authenticate("Bearer good-token")
    assert p.subject == "user-1"
    assert p.anonymous is False
    assert "region-apac" in p.groups

    with pytest.raises(auth.AuthError):
        await auth.authenticate("Bearer wrong-token")


def test_bearer_extracts_token():
    assert auth._bearer("Bearer abc123") == "abc123"


def test_bearer_rejects_missing_or_malformed_header():
    with pytest.raises(auth.AuthError):
        auth._bearer(None)
    with pytest.raises(auth.AuthError):
        auth._bearer("Basic abc123")


@pytest.fixture
def anyio_backend():
    return "asyncio"
