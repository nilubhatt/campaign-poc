"""
Adversarial review finding: upload_image_asset's docstring says asset_ref can be {asset_id}
from POST /upload, but /upload's ALLOWED_MIME was deck-only (PDF/PPTX/PPT) - the documented
route could never actually work for images. From Claude Web (no local paths) inline base64
was the only working input for an image.
"""
import io

import pytest
from starlette.testclient import TestClient

import config
import http_app
import store


@pytest.fixture
def client(conn):
    # `conn` fixture already isolates config.DATA_DIR/UPLOAD_DIR/etc. via monkeypatch
    store.init_db()
    return TestClient(http_app.app)


def test_upload_accepts_a_png_image(client):
    files = {"file": ("hero.png", io.BytesIO(b"not a real png but a valid upload payload"), "image/png")}
    r = client.post("/upload", files=files)
    assert r.status_code == 200
    assert "asset_id" in r.json()


def test_upload_accepts_a_jpeg_image(client):
    files = {"file": ("hero.jpg", io.BytesIO(b"jpeg-ish bytes"), "image/jpeg")}
    r = client.post("/upload", files=files)
    assert r.status_code == 200


def test_upload_still_rejects_unsupported_types(client):
    files = {"file": ("notes.txt", io.BytesIO(b"plain text"), "text/plain")}
    r = client.post("/upload", files=files)
    assert r.status_code == 400
