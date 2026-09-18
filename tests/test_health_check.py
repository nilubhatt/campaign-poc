"""
Product review defect 06 (P1): "No health check — failures are only discoverable by timeout."

The reviewer's account: "Establishing that half the product was non-functional took a
60-second timeout, a second upload to capture the warning string, and a read of the server's
source behaviour. There is no way to ask the server whether its dependencies are live."

So the bar is not "expose some JSON". It is that a person who suspects something is wrong
can ask once, get an answer in seconds, and be told what to do about it — and that the same
answer is available to the installer as a post-install self-test (item 4.1) rather than
being a thing only an engineer reading source can determine.

Three properties matter more than the field list:
  * it must never hang — a diagnostic that inherits the timeout it exists to diagnose is
    worthless, so every probe is bounded well under a normal call;
  * it must never raise — a health check that crashes tells you nothing at all;
  * every unhealthy component must carry a remedy a non-engineer can act on.
"""
import time

import pytest

import clip_embed
import config
import core
import embedding
import store


def test_reports_every_component_and_an_overall_verdict(conn):
    report = core.health_check(conn)

    assert set(report["components"]) >= {"database", "text_search", "visual_search"}
    assert isinstance(report["ok"], bool)
    for name, component in report["components"].items():
        assert isinstance(component["ok"], bool), name


def test_a_healthy_system_says_so(conn, monkeypatch):
    monkeypatch.setattr(embedding, "embed", lambda text, timeout=None: [0.0] * config.EMBED_DIM)

    report = core.health_check(conn)

    assert report["ok"] is True
    assert report["components"]["text_search"]["ok"] is True
    assert report["components"]["database"]["ok"] is True


def test_an_unreachable_embedder_is_reported_with_a_remedy(conn, monkeypatch):
    """The literal situation the reviewer was in - half the product dead, discoverable only
    by waiting for a timeout."""
    monkeypatch.setattr(embedding, "embed",
                        lambda text, timeout=None: (_ for _ in ()).throw(
                            ValueError("could not reach the ollama embedder at "
                                       "http://localhost:11434")))

    report = core.health_check(conn)

    text_search = report["components"]["text_search"]
    assert report["ok"] is False
    assert text_search["ok"] is False
    assert text_search["remedy"], "an operator must be told what to do"
    assert "ollama" in (text_search["detail"] + text_search["remedy"]).lower()


def test_missing_vision_weights_are_reported_without_breaking_the_rest(conn, monkeypatch):
    """The whole point of phase 1's design: vision being down is not the server being down.
    The health check has to express that distinction rather than collapsing to one flag."""
    monkeypatch.setattr(config, "CLIP_PROVIDER", "openclip")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", "/nowhere/at/all")
    monkeypatch.setattr(embedding, "embed", lambda text, timeout=None: [0.0] * config.EMBED_DIM)
    clip_embed._resolution = None

    report = core.health_check(conn)

    assert report["components"]["visual_search"]["ok"] is False
    assert report["components"]["visual_search"]["remedy"]
    assert report["components"]["text_search"]["ok"] is True, "text is unaffected"
    assert report["components"]["database"]["ok"] is True


def test_it_never_raises_however_broken_things_are(conn, monkeypatch):
    """A diagnostic that crashes is worse than none: it tells you nothing and costs a
    support call to interpret."""
    monkeypatch.setattr(embedding, "embed",
                        lambda text, timeout=None: (_ for _ in ()).throw(
                            RuntimeError("something nobody anticipated")))
    monkeypatch.setattr(clip_embed, "weights_status",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    report = core.health_check(conn)   # must not raise

    assert report["ok"] is False


def test_it_is_fast_even_when_everything_is_down(conn, monkeypatch):
    """It must not inherit the very timeout it exists to diagnose. The reviewer waited 60
    seconds to learn something this should answer in about one."""
    def hang(text, timeout=None):
        time.sleep(timeout if timeout else 30)
        raise ValueError("timed out")

    monkeypatch.setattr(embedding, "embed", hang)

    started = time.monotonic()
    report = core.health_check(conn)
    elapsed = time.monotonic() - started

    assert elapsed < 5, f"a health check took {elapsed:.1f}s - that is the defect, not the fix"
    assert report["components"]["text_search"]["ok"] is False


def test_it_reports_data_coverage_not_just_liveness(conn, monkeypatch):
    """The review asked for counts alongside component status: "campaigns: 8,
    unembedded_text: 0, assets: 2, unembedded_assets: 2". Liveness answers "is it running";
    coverage answers "is what I uploaded actually usable", which is the question behind it."""
    monkeypatch.setattr(config, "TOOL_TIME_BUDGET_SECONDS", 0.0)
    core.ingest_campaign(conn, title="Half done",
                         deck_text="\n\n".join(f"s{i} " + "word " * 200 for i in range(4)),
                         confirm=True)
    monkeypatch.setattr(config, "TOOL_TIME_BUDGET_SECONDS", 45.0)

    report = core.health_check(conn)

    coverage = report["coverage"]
    assert coverage["campaigns"] == 1
    assert coverage["sections_unindexed"] > 0
    # NOT `ok is False`: an earlier version folded coverage into liveness, which would fail
    # an installer gate on a perfectly working machine because someone had just uploaded a
    # big deck. Liveness and coverage are separate verdicts for separate audiences.
    assert coverage["complete"] is False
    assert any("finish_indexing" in str(v) for v in report.values()), \
        "say how to fix the backlog"


def test_a_fully_indexed_library_reports_no_backlog(conn, monkeypatch):
    monkeypatch.setattr(embedding, "embed", lambda text, timeout=None: [0.0] * config.EMBED_DIM)
    core.ingest_campaign(conn, title="Complete", deck_text="a brief", confirm=True)

    report = core.health_check(conn)

    assert report["coverage"]["sections_unindexed"] == 0
    assert report["coverage"]["images_unindexed"] == 0


def test_the_database_component_names_the_file_it_is_using(conn):
    """"Which database am I actually looking at" is the first question in any support call
    on a local-first product."""
    report = core.health_check(conn)

    assert str(config.DB_PATH) in report["components"]["database"]["detail"]


# ── a regression the health check itself found on its first real run ────────

def test_keep_alive_is_sent_in_a_form_ollama_accepts(monkeypatch):
    """Item 2.1 added `keep_alive` to stop Ollama unloading the model between calls, and
    sent it as the string "-1". Ollama rejects that with
    `time: missing unit in duration "-1"` — it wants a NUMBER of seconds, or a duration
    string like "10m". So every embed call returned 400 and text search was dead.

    No unit test caught it: they all use the offline provider or stub httpx, so nothing
    exercised the real request shape. The health check found it within a second of first
    running against a live Ollama, which is the argument for the health check.
    """
    sent = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"embedding": [0.0] * config.EMBED_DIM}

    import httpx
    monkeypatch.setattr(httpx, "post",
                        lambda url, json=None, timeout=None: (sent.update(json or {}), _Resp())[1])

    embedding._embed_ollama("hello", 5.0)

    keep_alive = sent["keep_alive"]
    assert isinstance(keep_alive, (int, float)) or (
        isinstance(keep_alive, str) and keep_alive[-1].isalpha()), (
        f"keep_alive={keep_alive!r} is the shape Ollama rejects: send a number of seconds "
        f"or a duration string with a unit"
    )


# ── second review round: the shape has three audiences, not one ─────────────

def test_liveness_and_coverage_are_separate_verdicts(conn, monkeypatch):
    """Folding "is my library fully indexed" into "are my dependencies alive" means a
    perfectly healthy server reports not-ok because someone uploaded a big deck a minute
    ago - and an installer gating on that flag would fail a working install. They are
    different questions with different audiences."""
    monkeypatch.setattr(embedding, "embed", lambda text, timeout=None: [0.0] * config.EMBED_DIM)
    monkeypatch.setattr(config, "TOOL_TIME_BUDGET_SECONDS", 0.0)
    core.ingest_campaign(conn, title="Backlog",
                         deck_text="\n\n".join(f"s{i} " + "word " * 200 for i in range(4)),
                         confirm=True)
    monkeypatch.setattr(config, "TOOL_TIME_BUDGET_SECONDS", 45.0)

    report = core.health_check(conn)

    assert report["ok"] is True, "every component is alive - that is what ok means"
    assert report["coverage"]["complete"] is False, "but the library is not fully searchable"
    assert report["coverage"]["outstanding"], "and it says which records are waiting"
    assert report["coverage"]["outstanding"][0]["title"] == "Backlog"


def test_a_failing_component_reports_the_timeout_it_actually_waited(conn, monkeypatch):
    """The probe waits HEALTH_PROBE_SECONDS, not EMBED_TIMEOUT_SECONDS. Reporting the wrong
    number on the one surface built to answer "how long did it wait" is the original
    complaint in miniature."""
    monkeypatch.setattr(config, "HEALTH_PROBE_SECONDS", 2.0)
    monkeypatch.setattr(config, "EMBED_TIMEOUT_SECONDS", 15.0)

    import httpx
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **k: (_ for _ in ()).throw(httpx.ReadTimeout("slow")))
    monkeypatch.setattr(config, "EMBED_PROVIDER", "ollama")

    report = core.health_check(conn)

    detail = report["components"]["text_search"]["detail"]
    assert "2" in detail and "15s" not in detail, detail


def test_every_failure_carries_a_stable_code(conn, monkeypatch):
    """The installer (4.1) must gate on something better than string-matching prose, and
    3.1 is going to want exactly this shape. Choosing it now costs nothing - this surface
    has no callers yet."""
    monkeypatch.setattr(config, "CLIP_PROVIDER", "openclip")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", "/nowhere")
    import clip_embed as ce
    ce._resolution = None
    monkeypatch.setattr(embedding, "embed",
                        lambda text, timeout=None: (_ for _ in ()).throw(ValueError("down")))

    report = core.health_check(conn)

    for name, component in report["components"].items():
        if not component["ok"]:
            assert component.get("code"), f"{name} has no machine-readable code"
            assert component["code"].islower()


def test_an_unavailable_component_says_what_the_user_cannot_do(conn, monkeypatch):
    """The person in the chat is a marketer. "Set CAMPAIGN_POC_CLIP_WEIGHTS_PATH" is for an
    admin; she needs to know which capability is gone, in her words."""
    monkeypatch.setattr(config, "CLIP_PROVIDER", "openclip")
    monkeypatch.setattr(config, "CLIP_WEIGHTS_PATH", "/nowhere")
    import clip_embed as ce
    ce._resolution = None

    report = core.health_check(conn)

    visual = report["components"]["visual_search"]
    assert visual["affects"], "say what stops working, not only how to fix it"
    assert "remedy" in visual, "and keep the admin instruction separate"


def test_there_is_a_headline_in_every_state(conn, monkeypatch):
    """A sentence a person can read, present whether things are fine or not - not an
    instruction to Claude that only appears on failure."""
    monkeypatch.setattr(embedding, "embed", lambda text, timeout=None: [0.0] * config.EMBED_DIM)

    healthy = core.health_check(conn)
    assert healthy["headline"]

    monkeypatch.setattr(embedding, "embed",
                        lambda text, timeout=None: (_ for _ in ()).throw(ValueError("down")))
    broken = core.health_check(conn)
    assert broken["headline"] and broken["headline"] != healthy["headline"]


def test_the_vector_backend_is_reported(conn):
    """The reviewer named sqlite-vec explicitly. Silently falling back to pure-Python cosine
    is a real degradation and was invisible here while /healthz reported it."""
    report = core.health_check(conn)

    assert "vector_index" in report["components"]["database"]["detail"] or \
           report["components"].get("vector_index")


def test_the_text_detail_does_not_name_the_ollama_model_for_other_providers(conn, monkeypatch):
    monkeypatch.setattr(config, "EMBED_PROVIDER", "hash")

    report = core.health_check(conn)

    assert config.OLLAMA_EMBED_MODEL not in report["components"]["text_search"]["detail"]


# ── third round: the ways it reported green on a broken system ──────────────

def test_vectors_stranded_in_the_wrong_backend_are_detected(conn, monkeypatch):
    """The worst failure a health check can have: reporting green while search returns
    nothing. Vectors written with sqlite-vec loaded live in a different table from the
    fallback, so a database written on one machine and opened where the extension is
    unavailable has every chunk flagged embedded, zero backlog, and no results at all."""
    import vectorstore

    monkeypatch.setattr(embedding, "embed", lambda text, timeout=None: [0.0] * config.EMBED_DIM)
    core.ingest_campaign(conn, title="Indexed", deck_text="mexico launch", confirm=True)

    # Now pretend the extension is unavailable, as on a bundle that lost the native lib.
    monkeypatch.setattr(vectorstore, "_try_load_vec", lambda c: False)

    report = core.health_check(conn)

    assert report["ok"] is False, "search cannot work; that is not healthy"
    codes = [c.get("code") for c in report["components"].values()]
    assert "vector_index_mismatch" in codes, report["components"]


def test_an_embedder_returning_the_wrong_size_is_caught(conn, monkeypatch):
    """A user who points CAMPAIGN_POC_OLLAMA_MODEL at a different model gets vectors of the
    wrong width: every upload then fails deep inside the vector store, while a health check
    that only asks "did it answer" says everything is fine."""
    monkeypatch.setattr(embedding, "embed",
                        lambda text, timeout=None: [0.0] * (config.EMBED_DIM + 256))

    report = core.health_check(conn)

    assert report["components"]["text_search"]["ok"] is False
    assert report["components"]["text_search"]["code"] == "embedder_wrong_dimension"
    assert str(config.EMBED_DIM) in report["components"]["text_search"]["detail"]


def test_a_missing_database_is_reported_not_created(conn, monkeypatch, tmp_path):
    """A diagnostic must not repair what it is diagnosing. Creating the database and then
    reporting "0 records, healthy" hides that the real one is gone."""
    missing = tmp_path / "nope" / "campaigns.db"
    monkeypatch.setattr(config, "DB_PATH", missing)

    report = core.health_check_cli()

    assert report["components"]["database"]["ok"] is False
    assert report["components"]["database"]["code"] == "db_missing"
    assert not missing.exists(), "the health check must not have created it"


def test_a_corrupt_database_reports_rather_than_crashing(monkeypatch, tmp_path):
    """Verified in review: the CLI died with a raw sqlite3.DatabaseError traceback, which
    breaks the "never raises" property on the surface built to replace reading a traceback."""
    corrupt = tmp_path / "campaigns.db"
    corrupt.write_bytes(b"this is definitely not a database")
    monkeypatch.setattr(config, "DB_PATH", corrupt)

    report = core.health_check_cli()   # must not raise

    assert report["ok"] is False
    assert report["components"]["database"]["ok"] is False


def test_a_model_that_was_never_pulled_is_explained_correctly(conn, monkeypatch):
    """Ollama answers 404 "model not found, try pulling it first". Rendering that as "the
    context limit is the usual cause" sends an operator in exactly the wrong direction."""
    import httpx

    monkeypatch.setattr(config, "EMBED_PROVIDER", "ollama")

    class _Resp:
        status_code = 404

        def raise_for_status(self):
            raise httpx.HTTPStatusError("404", request=None, response=self)

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp())

    report = core.health_check(conn)

    text = report["components"]["text_search"]
    assert text["code"] == "embedder_model_missing"
    assert "pull" in text["remedy"].lower()
    assert "context limit" not in text["detail"].lower()
