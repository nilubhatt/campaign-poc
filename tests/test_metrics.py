"""
§6.5: metrics as first-class. metric_type distinguishes predicted vs actual outcomes;
list_campaigns/get_campaign report whether a record has metrics/evaluations; bulk import
loads a KPI workbook in one call; reconcile_evaluation pulls actual metrics automatically
once they're on file, instead of requiring them retyped every time.
"""
import core
import store


def test_add_metrics_defaults_to_actual(conn):
    cid = store.insert_campaign(conn, title="X")
    store.add_metrics(conn, cid, detail="great results")
    m = store.get_campaign(conn, cid)["metrics"][0]
    assert m["metric_type"] == "actual"


def test_add_metrics_can_record_a_prediction(conn):
    cid = store.insert_campaign(conn, title="X")
    store.add_metrics(conn, cid, detail="forecast", metric_type="predicted")
    m = store.get_campaign(conn, cid)["metrics"][0]
    assert m["metric_type"] == "predicted"


def test_get_campaign_reports_has_metrics_and_has_evaluations(conn):
    cid = store.insert_campaign(conn, title="X")
    c0 = store.get_campaign(conn, cid)
    assert c0["has_metrics"] is False
    assert c0["has_evaluations"] is False

    store.add_metrics(conn, cid, detail="results")
    store.insert_evaluation(conn, subject_title="X", verdict="approve", summary="judged", findings=[], campaign_id=cid)

    c1 = store.get_campaign(conn, cid)
    assert c1["has_metrics"] is True
    assert c1["has_evaluations"] is True


def test_list_campaigns_reports_has_metrics_and_has_evaluations(conn):
    with_metrics = store.insert_campaign(conn, title="A")
    store.add_metrics(conn, with_metrics, detail="results")
    without = store.insert_campaign(conn, title="B")

    rows = {r["id"]: r for r in store.list_campaigns(conn)}
    assert rows[with_metrics]["has_metrics"] is True
    assert rows[without]["has_metrics"] is False


def test_bulk_import_metrics_by_campaign_id(conn):
    cid = store.insert_campaign(conn, title="X")
    result = store.bulk_import_metrics(conn, [
        {"campaign_id": cid, "detail": "CTR 4%", "structured": {"ctr": 0.04}},
    ], confirm=True)
    assert result["imported"] == 1
    assert result["errors"] == []
    assert store.get_campaign(conn, cid)["has_metrics"] is True


def test_bulk_import_metrics_by_title_exact_case_insensitive(conn):
    cid = store.insert_campaign(conn, title="APAC Summer Launch")
    result = store.bulk_import_metrics(conn, [
        {"title": "apac summer launch", "detail": "sales up 12%"},
    ], confirm=True)
    assert result["imported"] == 1
    assert store.get_campaign(conn, cid)["metrics"][0]["detail"] == "sales up 12%"


def test_bulk_import_metrics_reports_per_row_errors_not_swallowed(conn):
    store.insert_campaign(conn, title="A")
    store.insert_campaign(conn, title="Duplicate")
    store.insert_campaign(conn, title="Duplicate")

    result = store.bulk_import_metrics(conn, [
        {"campaign_id": "does-not-exist", "detail": "x"},
        {"title": "No Such Campaign", "detail": "x"},
        {"title": "Duplicate", "detail": "x"},        # ambiguous - two matches
        {"detail": "x"},                               # neither campaign_id nor title given
        {"title": "A", "detail": "good row"},          # this one succeeds
    ], confirm=True)
    assert result["imported"] == 1
    assert len(result["errors"]) == 4
    reasons = " ".join(e["reason"] for e in result["errors"])
    assert "not found" in reasons
    assert "ambiguous" in reasons or "matches" in reasons


def test_bulk_import_metrics_supports_metric_type(conn):
    cid = store.insert_campaign(conn, title="X")
    store.bulk_import_metrics(conn, [
        {"campaign_id": cid, "detail": "forecast 5%", "metric_type": "predicted"},
    ], confirm=True)
    assert store.get_campaign(conn, cid)["metrics"][0]["metric_type"] == "predicted"


def test_bulk_import_metrics_normalizes_metric_type_case_and_whitespace(conn):
    """Adversarial review finding: 'Actual'/'predicted ' inserted fine but were silently
    excluded from reconcile_evaluation's exact `== 'actual'` check - normalize instead."""
    cid = store.insert_campaign(conn, title="X")
    store.bulk_import_metrics(conn, [
        {"campaign_id": cid, "detail": "x", "metric_type": " Actual "},
    ], confirm=True)
    assert store.get_campaign(conn, cid)["metrics"][0]["metric_type"] == "actual"


def test_bulk_import_metrics_rejects_invalid_metric_type_as_a_row_error(conn):
    cid = store.insert_campaign(conn, title="X")
    result = store.bulk_import_metrics(conn, [
        {"campaign_id": cid, "detail": "x", "metric_type": "forecasted"},
    ], confirm=True)
    assert result["imported"] == 0
    assert "metric_type" in result["errors"][0]["reason"]


def test_bulk_import_metrics_one_bad_row_does_not_crash_the_batch(conn):
    """Adversarial review finding: a non-dict row raised AttributeError and crashed the
    whole call, including rows already processed before it in the same batch."""
    cid = store.insert_campaign(conn, title="X")
    result = store.bulk_import_metrics(conn, [
        {"campaign_id": cid, "detail": "good row one"},
        "not a dict",
        {"campaign_id": cid, "detail": "good row two"},
    ], confirm=True)
    assert result["imported"] == 2
    assert len(result["errors"]) == 1
    assert result["errors"][0]["row"] == 1


def test_bulk_import_metrics_title_match_excludes_superseded_campaigns(conn):
    """A re-uploaded corrected deck (supersedes=old) shouldn't make every subsequent
    title-based import ambiguous forever - the live one should resolve uniquely."""
    old = store.insert_campaign(conn, title="Mexico Push")
    store.insert_campaign(conn, title="Mexico Push", supersedes=old)

    result = store.bulk_import_metrics(conn, [{"title": "Mexico Push", "detail": "results"}], confirm=True)
    assert result["imported"] == 1
    assert result["errors"] == []


def test_reconcile_evaluation_pulls_actual_metrics_automatically(conn):
    cid = store.insert_campaign(conn, title="X")
    eid = store.insert_evaluation(conn, subject_title="X", verdict="approve", summary="predicted strong ROI", findings=[],
                                  campaign_id=cid, predictions={"roi_range": [1.2, 1.6]})
    store.add_metrics(conn, cid, detail="actual ROI came in at 1.8", metric_type="actual")

    result = core.reconcile_evaluation(conn, evaluation_id=eid)
    assert "actual ROI came in at 1.8" in result["actual"]
    assert result["predictions"] == {"roi_range": [1.2, 1.6]}


def test_reconcile_evaluation_explicit_actual_overrides_stored_metrics(conn):
    cid = store.insert_campaign(conn, title="X")
    eid = store.insert_evaluation(conn, subject_title="X", verdict="approve", summary="predicted", findings=[], campaign_id=cid)
    store.add_metrics(conn, cid, detail="stored actual", metric_type="actual")

    result = core.reconcile_evaluation(conn, evaluation_id=eid, actual="manually provided actual")
    assert result["actual"] == "manually provided actual"


def test_reconcile_evaluation_errors_when_no_actual_available(conn):
    cid = store.insert_campaign(conn, title="X")
    eid = store.insert_evaluation(conn, subject_title="X", verdict="approve", summary="predicted", findings=[], campaign_id=cid)

    result = core.reconcile_evaluation(conn, evaluation_id=eid)
    assert "error" in result


def test_reconcile_evaluation_includes_structured_only_actuals(conn):
    """Adversarial review finding: a bulk-imported KPI row with only `structured` (no
    freeform `detail` text — exactly what a workbook import produces) was silently dropped,
    leaving `actual` an empty string with no error, instead of surfacing the numbers."""
    cid = store.insert_campaign(conn, title="X")
    eid = store.insert_evaluation(conn, subject_title="X", verdict="approve", summary="predicted", findings=[], campaign_id=cid)
    store.add_metrics(conn, cid, structured={"ctr": 0.05, "roi": 1.8}, metric_type="actual")

    result = core.reconcile_evaluation(conn, evaluation_id=eid)
    assert "error" not in result
    assert "0.05" in result["actual"] or "ctr" in result["actual"]


def test_reconcile_evaluation_ignores_predicted_metrics_when_auto_pulling(conn):
    cid = store.insert_campaign(conn, title="X")
    eid = store.insert_evaluation(conn, subject_title="X", verdict="approve", summary="predicted", findings=[], campaign_id=cid)
    store.add_metrics(conn, cid, detail="forecast only", metric_type="predicted")

    result = core.reconcile_evaluation(conn, evaluation_id=eid)
    assert "error" in result  # no *actual* metrics on file yet
