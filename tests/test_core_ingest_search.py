import core
import embedding
import store


def test_ingest_short_campaign_yields_one_chunk(conn):
    r = core.ingest_campaign(conn, title="Short Brief", status="proposed",
                             detail="A small proposal with no deck.")
    assert r["chunks_total"] == 1
    assert r["chunks_embedded"] == 1
    assert r["embedded"] is True
    assert r["warnings"] == []


def test_ingest_empty_campaign_embeds_nothing(conn):
    r = core.ingest_campaign(conn, title="", status="proposed")
    assert r["chunks_total"] == 0
    assert r["chunks_embedded"] == 0
    assert r["embedded"] is False
    assert "nothing to embed" in r["warnings"][0]


def test_ingest_long_flat_deck_text_splits_into_multiple_chunks(conn):
    # No blank-line boundaries within each paragraph, but multiple paragraphs -> multiple
    # chunk-worthy units once packed past MAX_CHUNK_CHARS.
    paragraph = "word " * 500  # ~2500 chars, one paragraph
    deck_text = "\n\n".join([paragraph] * 4)
    r = core.ingest_campaign(conn, title="Long Deck", status="concluded", deck_text=deck_text)
    assert r["chunks_total"] > 1
    assert r["chunks_embedded"] == r["chunks_total"]
    chunks = store.get_chunk_ids_for_campaign(conn, r["campaign_id"])
    assert len(chunks) == r["chunks_total"]


def test_partial_embedding_failure_is_reported_per_chunk_not_swallowed(conn, monkeypatch):
    real_embed = embedding.embed
    calls = {"n": 0}

    def flaky_embed(text, timeout=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("provider rejected input")
        return real_embed(text)

    monkeypatch.setattr(embedding, "embed", flaky_embed)

    paragraph = "word " * 500
    deck_text = "\n\n".join([paragraph] * 4)
    r = core.ingest_campaign(conn, title="Flaky Deck", status="concluded", deck_text=deck_text)

    assert r["chunks_total"] > 1
    assert r["chunks_embedded"] == r["chunks_total"] - 1
    # `embedded` means FULLY searchable. This assertion used to read "at least one chunk
    # embedded -> still searchable", which made sense while partial embedding was an
    # accident; item 2.1 made it a designed outcome (a time budget stops mid-deck on
    # purpose), so a flag that says True at 1-of-12 is the stored-versus-searchable
    # conflation defect 05 complained about. The partial story is told by the counts.
    assert r["embedded"] is False
    assert 0 < r["chunks_embedded"] < r["chunks_total"]
    assert any("not embedded" in w for w in r["warnings"])

    c = store.get_campaign(conn, r["campaign_id"])
    assert c["chunks_embedded"] == r["chunks_embedded"]


def test_find_similar_excludes_query_campaigns_own_chunks(conn):
    r1 = core.ingest_campaign(conn, title="APAC Summer Launch", status="concluded",
                              detail="Region APAC, social + OOH, awareness goal.")
    core.ingest_campaign(conn, title="Mexico Spring Push", status="concluded",
                         detail="Region LATAM, social, awareness goal.")

    hits = core.find_similar(conn, campaign_id=r1["campaign_id"], top_k=5)
    assert all(e["campaign_id"] != r1["campaign_id"] for e in hits)


def test_find_similar_dedupes_multi_chunk_campaign_to_best_match(conn):
    paragraph = "APAC summer awareness social OOH launch targeting young adults. " * 40
    deck_text = "\n\n".join([paragraph] * 5)
    r1 = core.ingest_campaign(conn, title="APAC Summer Launch", status="concluded",
                              deck_text=deck_text)
    assert r1["chunks_total"] > 1  # multiple chunks from the same campaign

    core.ingest_campaign(conn, title="Unrelated Winter Sale", status="concluded",
                         detail="Completely different: B2B enterprise software renewal push.")

    hits = core.find_similar(conn, text="APAC summer awareness social OOH launch", top_k=5)
    campaign_ids = [e["campaign_id"] for e in hits]
    assert campaign_ids.count(r1["campaign_id"]) == 1  # never duplicated across its own chunks


def test_get_campaign_reports_chunk_counts(conn):
    r = core.ingest_campaign(conn, title="X", status="proposed", detail="Some detail text.")
    c = store.get_campaign(conn, r["campaign_id"])
    assert c["chunks_total"] == r["chunks_total"]
    assert c["chunks_embedded"] == r["chunks_embedded"]
