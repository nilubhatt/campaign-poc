"""Direct coverage of store.py CRUD, independent of core.py's orchestration."""
import store


def test_insert_and_get_campaign_roundtrip(conn):
    cid = store.insert_campaign(conn, title="X", status="proposed", detail="d", deck_text="dt")
    c = store.get_campaign(conn, cid)
    assert c["title"] == "X"
    assert c["record_type"] == "campaign"
    assert c["status"] == "proposed"
    assert c["detail"] == "d"
    assert c["metrics"] == []
    assert c["chunks_total"] == 0  # no chunks inserted via this low-level call


def test_get_campaign_missing_returns_none(conn):
    assert store.get_campaign(conn, "does-not-exist") is None


def test_list_campaigns_filters_by_record_type_and_status(conn):
    store.insert_campaign(conn, title="A", status="proposed")
    store.insert_campaign(conn, title="B", status="concluded")
    store.insert_campaign(conn, title="C", status="concluded")
    store.insert_campaign(conn, title="D", record_type="reference")

    assert len(store.list_campaigns(conn)) == 4
    assert len(store.list_campaigns(conn, status="concluded")) == 2
    assert len(store.list_campaigns(conn, status="proposed")) == 1
    assert len(store.list_campaigns(conn, record_type="campaign")) == 3
    assert len(store.list_campaigns(conn, record_type="reference")) == 1
    assert len(store.list_campaigns(conn, record_type="campaign", status="concluded")) == 2


def test_list_campaigns_shows_is_superseded(conn):
    """Design review finding: list_campaigns hid supersedes/is_superseded, so a dead and a
    live version of the same deck were indistinguishable in a listing - the exact bug §6.4
    was written to fix, just moved from get_campaign to list_campaigns."""
    old = store.insert_campaign(conn, title="Mexico Push (draft)")
    new = store.insert_campaign(conn, title="Mexico Push (final)", supersedes=old)

    rows = {r["id"]: r for r in store.list_campaigns(conn)}
    assert rows[old]["is_superseded"] is True
    assert rows[old]["supersedes"] is None
    assert rows[new]["is_superseded"] is False
    assert rows[new]["supersedes"] == old


def test_reference_and_stub_records_have_no_status_by_default(conn):
    ref = store.insert_campaign(conn, title="Brand guidelines", record_type="reference")
    stub = store.insert_campaign(conn, title="Placeholder", record_type="stub")
    campaign = store.insert_campaign(conn, title="Real campaign")  # default record_type

    assert store.get_campaign(conn, ref)["status"] is None
    assert store.get_campaign(conn, stub)["status"] is None
    assert store.get_campaign(conn, campaign)["status"] == "concluded"


def test_explicit_status_overrides_default_even_for_non_campaign_record_type(conn):
    cid = store.insert_campaign(conn, title="X", record_type="reference", status="proposed")
    assert store.get_campaign(conn, cid)["status"] == "proposed"


def test_mark_embedded_flips_flag(conn):
    cid = store.insert_campaign(conn, title="X")
    assert store.get_campaign(conn, cid)["embedded"] == 0
    store.mark_embedded(conn, cid, True)
    assert store.get_campaign(conn, cid)["embedded"] == 1
    store.mark_embedded(conn, cid, False)
    assert store.get_campaign(conn, cid)["embedded"] == 0


def test_chunks_insert_and_lookup(conn):
    cid = store.insert_campaign(conn, title="X")
    ids = store.insert_chunks(conn, cid, ["chunk one", "chunk two"])
    assert len(ids) == 2
    assert store.get_chunk_ids_for_campaign(conn, cid) == ids

    c0 = store.get_chunk(conn, ids[0])
    assert c0["text"] == "chunk one"
    assert c0["embedded"] == 0
    store.set_chunk_embedded(conn, ids[0])
    assert store.get_chunk(conn, ids[0])["embedded"] == 1

    assert store.map_chunks_to_campaigns(conn, ids) == {ids[0]: cid, ids[1]: cid}
    assert store.map_chunks_to_campaigns(conn, []) == {}


def test_add_metrics_stores_freeform_and_structured(conn):
    cid = store.insert_campaign(conn, title="X")
    store.add_metrics(conn, cid, detail="went well", structured={"ctr": 0.04})
    c = store.get_campaign(conn, cid)
    assert len(c["metrics"]) == 1
    assert c["metrics"][0]["detail"] == "went well"
    import json
    assert json.loads(c["metrics"][0]["structured"]) == {"ctr": 0.04}


def test_evaluation_and_reconciliation_roundtrip(conn):
    cid = store.insert_campaign(conn, title="X")
    eid = store.insert_evaluation(conn, subject_title="X proposal", analysis="looks strong",
                                  campaign_id=cid, cited_ids=[cid],
                                  predictions={"roi_range": [1.1, 1.5]})
    ev = store.get_evaluation(conn, eid)
    assert ev["subject_title"] == "X proposal"
    assert ev["reconciliations"] == []

    rid = store.insert_reconciliation(conn, evaluation_id=eid, comparison="beat prediction",
                                      actual="roi 1.8")
    ev2 = store.get_evaluation(conn, eid)
    assert len(ev2["reconciliations"]) == 1
    assert ev2["reconciliations"][0]["id"] == rid

    listed = store.list_evaluations(conn)
    assert any(e["id"] == eid for e in listed)


def test_get_evaluation_missing_returns_none(conn):
    assert store.get_evaluation(conn, "nope") is None
