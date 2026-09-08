"""
Adversarial review findings, both reproduced with real repro scripts:
  (1) over-fetch happens BEFORE the per-campaign rollup, so one deck with many chunks that
      all score highly can fill every slot, starving every other campaign out of the results
      entirely (top_k=5 returning just 1 campaign).
  (2) core.find_similar folded every superseded campaign's chunks into vectorstore.search's
      `exclude` set, which inflates the `k` sent to sqlite-vec's KNN query - past ~4096
      excluded chunks (a few hundred superseded decks) this crashes outright.
Uses hand-inserted vectors (bypassing the text embedder) for exact control over similarity.
"""
import config
import core
import embedding
import store
import vectorstore


def test_find_similar_unfiltered_does_not_starve_on_one_chunky_campaign(conn):
    qvec = embedding.embed("shared query text")

    big = store.insert_campaign(conn, title="Big Deck")
    chunk_ids = store.insert_chunks(conn, big, [f"chunk {i}" for i in range(30)])
    for chid in chunk_ids:
        vectorstore.add(conn, chid, qvec)  # 30 chunks, all a perfect match

    small_ids = []
    for i in range(5):
        sid = store.insert_campaign(conn, title=f"Small {i}")
        (chid,) = store.insert_chunks(conn, sid, [f"small {i}"])
        vectorstore.add(conn, chid, embedding.embed(f"small {i} shared query text"))
        small_ids.append(sid)

    hits = core.find_similar(conn, text="shared query text", top_k=5)
    campaign_ids = {h["campaign_id"] for h in hits}
    assert len(campaign_ids) == 5  # not starved down to just the one chunky campaign
    assert big in campaign_ids


def test_find_similar_filtered_does_not_starve_on_one_chunky_campaign(conn):
    qvec = embedding.embed("shared query text")

    big = store.insert_campaign(conn, title="Big Deck", region="APAC")
    chunk_ids = store.insert_chunks(conn, big, [f"chunk {i}" for i in range(30)])
    for chid in chunk_ids:
        vectorstore.add(conn, chid, qvec)

    small_ids = []
    for i in range(5):
        sid = store.insert_campaign(conn, title=f"Small {i}", region="APAC")
        (chid,) = store.insert_chunks(conn, sid, [f"small {i}"])
        vectorstore.add(conn, chid, embedding.embed(f"small {i} shared query text"))
        small_ids.append(sid)

    hits = core.find_similar(conn, text="shared query text", top_k=5, region="APAC")
    campaign_ids = {h["campaign_id"] for h in hits}
    assert len(campaign_ids) == 5
    assert big in campaign_ids


def test_vectorstore_search_clamps_k_to_avoid_sqlite_vec_limit(conn, monkeypatch):
    """sqlite-vec's KNN `k` parameter has a hard upper bound (observed: 4096) - a large
    `exclude` set must not cross it and crash the query."""
    monkeypatch.setattr(vectorstore, "_MAX_ANN_K", 3)
    qvec = embedding.embed("query")
    excluded = []
    for i in range(10):
        cid = store.insert_campaign(conn, title=f"C{i}")
        (chid,) = store.insert_chunks(conn, cid, [f"chunk {i}"])
        vectorstore.add(conn, chid, embedding.embed(f"chunk {i}"))
        excluded.append(chid)

    # top_k(1) + len(exclude)(9) = 10, well past the monkeypatched cap of 3 - must not raise
    hits = vectorstore.search(conn, qvec, top_k=1, exclude=set(excluded[:9]))
    assert isinstance(hits, list)
