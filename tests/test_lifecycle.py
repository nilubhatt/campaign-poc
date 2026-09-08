"""
§6.4: CRUD + supersede. Prior mistakes were permanent (a dead deck stayed indistinguishable
from the live one). update_campaign edits metadata (not deck_text/chunks - re-upload for
content changes); delete_campaign removes a record + its chunks/vectors; supersedes links a
new record to the one it replaces, and the replaced one is excluded from search evidence.
"""
import core
import store
import vectorstore


def test_update_campaign_changes_only_given_fields(conn):
    cid = store.insert_campaign(conn, title="Old Title", detail="old detail", region="APAC")
    ok = store.update_campaign(conn, cid, title="New Title")
    assert ok is True

    c = store.get_campaign(conn, cid)
    assert c["title"] == "New Title"
    assert c["detail"] == "old detail"  # untouched
    assert c["region"] == "APAC"        # untouched


def test_update_campaign_tags_fully_replaces_not_merges(conn):
    cid = store.insert_campaign(conn, title="X", tags=["a", "b"])
    store.update_campaign(conn, cid, tags=["c"])
    assert store.get_campaign(conn, cid)["tags"] == ["c"]


def test_update_campaign_missing_id_returns_false(conn):
    assert store.update_campaign(conn, "nope", title="X") is False


def test_update_campaign_no_fields_given_is_a_noop_that_reports_existence(conn):
    cid = store.insert_campaign(conn, title="X")
    assert store.update_campaign(conn, cid) is True
    assert store.update_campaign(conn, "nope") is False


def test_delete_campaign_removes_record(conn):
    cid = store.insert_campaign(conn, title="X")
    assert store.delete_campaign(conn, cid) is True
    assert store.get_campaign(conn, cid) is None


def test_delete_campaign_missing_id_returns_false(conn):
    assert store.delete_campaign(conn, "nope") is False


def test_delete_campaign_cascades_chunks_and_vectors(conn):
    r = core.ingest_campaign(conn, title="X", detail="some detail to embed")
    chunk_ids = store.get_chunk_ids_for_campaign(conn, r["campaign_id"])
    assert chunk_ids  # sanity: chunks exist
    assert vectorstore.get_many(conn, chunk_ids) != {}  # sanity: vectors exist

    store.delete_campaign(conn, r["campaign_id"])

    assert store.get_chunk_ids_for_campaign(conn, r["campaign_id"]) == []
    assert vectorstore.get_many(conn, chunk_ids) == {}  # vectors purged, not orphaned


def test_delete_campaign_cascades_metrics_and_detaches_evaluations(conn):
    cid = store.insert_campaign(conn, title="X")
    store.add_metrics(conn, cid, detail="did well")
    eid = store.insert_evaluation(conn, subject_title="X", analysis="looks good", campaign_id=cid)

    store.delete_campaign(conn, cid)

    ev = store.get_evaluation(conn, eid)
    assert ev is not None  # evaluation itself survives (it's a record of a judgment made)
    assert ev["campaign_id"] is None  # FK ON DELETE SET NULL


def test_insert_with_supersedes_sets_reverse_pointer(conn):
    old = store.insert_campaign(conn, title="Mexico Push (draft)")
    new = store.insert_campaign(conn, title="Mexico Push (final)", supersedes=old)

    assert store.get_campaign(conn, new)["supersedes"] == old
    assert store.get_campaign(conn, old)["superseded_by"] == new


def test_insert_with_supersedes_nonexistent_target_is_a_harmless_noop(conn):
    new = store.insert_campaign(conn, title="X", supersedes="does-not-exist")
    assert store.get_campaign(conn, new)["supersedes"] == "does-not-exist"  # kept for traceability


def test_delete_campaign_clears_superseded_by_on_whatever_pointed_to_it(conn):
    old = store.insert_campaign(conn, title="Old")
    new = store.insert_campaign(conn, title="New", supersedes=old)

    store.delete_campaign(conn, new)  # deleting the *newer* record un-supersedes the old one

    assert store.get_campaign(conn, old)["superseded_by"] is None


def test_get_superseded_campaign_ids(conn):
    old = store.insert_campaign(conn, title="Old")
    store.insert_campaign(conn, title="New", supersedes=old)
    other = store.insert_campaign(conn, title="Unrelated")

    superseded = store.get_superseded_campaign_ids(conn)
    assert superseded == {old}
    assert other not in superseded


def test_find_similar_excludes_superseded_records_unfiltered_path(conn):
    old = core.ingest_campaign(conn, title="APAC Push (draft)", detail="audience targeting brief")
    new = core.ingest_campaign(conn, title="APAC Push (final)", detail="audience targeting brief",
                               supersedes=old["campaign_id"])

    hits = core.find_similar(conn, text="audience targeting brief", top_k=5)
    ids = [h["campaign_id"] for h in hits]
    assert old["campaign_id"] not in ids
    assert new["campaign_id"] in ids


def test_find_similar_excludes_superseded_records_filtered_path(conn):
    old = core.ingest_campaign(conn, title="APAC Push (draft)", region="APAC",
                               detail="audience targeting brief")
    new = core.ingest_campaign(conn, title="APAC Push (final)", region="APAC",
                               detail="audience targeting brief", supersedes=old["campaign_id"])

    hits = core.find_similar(conn, text="audience targeting brief", region="APAC", top_k=5)
    ids = [h["campaign_id"] for h in hits]
    assert old["campaign_id"] not in ids
    assert new["campaign_id"] in ids


def test_vectorstore_delete_many(conn):
    import sqlite3
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    vectorstore.init(c)
    vectorstore.add(c, "a", [1.0] + [0.0] * 767)
    vectorstore.add(c, "b", [0.0] * 768)
    vectorstore.delete_many(c, ["a"])
    assert vectorstore.get_many(c, ["a", "b"]) == {"b": [0.0] * 768}


def test_vectorstore_delete_many_empty_input_is_a_noop():
    import sqlite3
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    vectorstore.init(c)
    vectorstore.delete_many(c, [])  # must not raise
