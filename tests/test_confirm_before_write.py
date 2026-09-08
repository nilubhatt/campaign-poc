"""
§6.9: confirm-before-write, made structural rather than relying purely on Claude
remembering to ask first. confirm=False (the first call, after a guided conversation)
previews what WOULD be stored without writing anything; only confirm=True (a second,
explicit call) commits. Scoped to the two write tools fed by free-text conversational
intake - upload_campaign and add_metrics - not bulk_import_metrics (already-structured
tabular data, not a prose answer to parse) or save_evaluation/reconciliation (Claude's own
judgment, not a user's free-text answer).
"""
import core
import store


def test_ingest_campaign_confirm_false_does_not_write_anything(conn):
    result = core.ingest_campaign(conn, title="X", detail="some brief", confirm=False)
    assert result["preview"] is True
    assert store.list_campaigns(conn) == []


def test_ingest_campaign_confirm_false_echoes_the_fields(conn):
    result = core.ingest_campaign(conn, title="APAC Push", detail="brief text",
                                  record_type="campaign", status="proposed",
                                  tags=["seeding"], region="APAC", market="Philippines",
                                  confirm=False)
    assert result["title"] == "APAC Push"
    assert result["status"] == "proposed"
    assert result["tags"] == ["seeding"]
    assert result["region"] == "APAC"
    assert result["market"] == "Philippines"


def test_ingest_campaign_confirm_false_still_defaults_status_for_campaigns(conn):
    result = core.ingest_campaign(conn, title="X", confirm=False)
    assert result["status"] == "concluded"


def test_ingest_campaign_confirm_true_actually_writes(conn):
    result = core.ingest_campaign(conn, title="X", detail="brief", confirm=True)
    assert "preview" not in result
    assert result["campaign_id"]
    assert len(store.list_campaigns(conn)) == 1


def test_ingest_campaign_default_is_confirm_true_backward_compatible(conn):
    # No confirm= given at all - existing callers (and all prior tests) expect immediate
    # storage, unchanged.
    result = core.ingest_campaign(conn, title="X", detail="brief")
    assert "preview" not in result
    assert len(store.list_campaigns(conn)) == 1


def test_add_metrics_confirm_false_does_not_write_anything(conn):
    cid = store.insert_campaign(conn, title="X")
    result = core.add_metrics(conn, cid, detail="CTR 4%", confirm=False)
    assert result["preview"] is True
    assert store.get_campaign(conn, cid)["metrics"] == []


def test_add_metrics_confirm_false_echoes_the_fields(conn):
    cid = store.insert_campaign(conn, title="X")
    result = core.add_metrics(conn, cid, detail="great results", structured={"ctr": 0.05},
                              metric_type="actual", confirm=False)
    assert result["detail"] == "great results"
    assert result["structured"] == {"ctr": 0.05}
    assert result["metric_type"] == "actual"


def test_add_metrics_confirm_true_actually_writes(conn):
    cid = store.insert_campaign(conn, title="X")
    result = core.add_metrics(conn, cid, detail="great results", confirm=True)
    assert "preview" not in result
    assert result["metrics_id"]
    assert len(store.get_campaign(conn, cid)["metrics"]) == 1


def test_add_metrics_default_is_confirm_true_backward_compatible(conn):
    cid = store.insert_campaign(conn, title="X")
    result = core.add_metrics(conn, cid, detail="great results")
    assert "preview" not in result
    assert len(store.get_campaign(conn, cid)["metrics"]) == 1
