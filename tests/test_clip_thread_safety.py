"""
Adversarial review finding, reproduced with a stub: mcp dispatches sync tools via
anyio.to_thread.run_sync, so concurrent first calls to a tool that embeds an image can race
into _load_model() simultaneously. Without a lock, 3 concurrent first calls loaded the model
3 times (transiently ~3x memory, 3x load time, last writer wins).
"""
import threading
import time

import pytest

import clip_embed


class _FakeModel:
    def eval(self):
        return self


@pytest.fixture(autouse=True)
def reset_model_cache():
    clip_embed._model = None
    clip_embed._preprocess = None
    yield
    clip_embed._model = None
    clip_embed._preprocess = None


def test_load_model_is_thread_safe_and_loads_only_once(monkeypatch):
    call_count = {"n": 0}

    def slow_stub(*args, **kwargs):
        call_count["n"] += 1
        time.sleep(0.1)  # simulate a slow model load, wide enough for threads to race
        return (_FakeModel(), None, "preprocess")

    monkeypatch.setattr("open_clip.create_model_and_transforms", slow_stub)

    threads = [threading.Thread(target=clip_embed._load_model) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert call_count["n"] == 1
