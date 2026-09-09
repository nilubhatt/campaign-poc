"""§6.8 (minor): find_similar returns a trimmed detail by default (full campaign briefs can
be long, and every evidence row was paying that cost even for a quick similarity scan) -
full detail is opt-in via full_detail=True; get_campaign is always there for the full record."""
import config
import core


def test_find_similar_trims_long_detail_by_default(conn):
    long_detail = "x" * 5000
    core.ingest_campaign(conn, title="X", detail=long_detail)
    hits = core.find_similar(conn, text=long_detail, top_k=5)
    assert len(hits[0]["detail"]) <= config.EVIDENCE_DETAIL_SUMMARY_CHARS + len("... [truncated]")
    assert hits[0]["detail_truncated"] is True


def test_find_similar_short_detail_is_not_marked_truncated(conn):
    core.ingest_campaign(conn, title="X", detail="short brief")
    hits = core.find_similar(conn, text="short brief", top_k=5)
    assert hits[0]["detail"] == "short brief"
    assert hits[0]["detail_truncated"] is False


def test_find_similar_full_detail_true_returns_untrimmed(conn):
    long_detail = "x" * 5000
    core.ingest_campaign(conn, title="X", detail=long_detail)
    hits = core.find_similar(conn, text=long_detail, top_k=5, full_detail=True)
    assert hits[0]["detail"] == long_detail
    assert hits[0]["detail_truncated"] is False


def test_find_similar_metrics_trimming_prefers_actual_over_predicted(conn):
    """Adversarial review finding: trimming kept the OLDEST rows by insertion order, so a
    campaign predicted early then measured later (the normal lifecycle) had its real actual
    metrics silently dropped from a browsing-mode scan while stale predictions survived."""
    import config
    import store

    cid = core.ingest_campaign(conn, title="X", detail="brief")["campaign_id"]
    for i in range(config.EVIDENCE_METRICS_MAX):
        store.add_metrics(conn, cid, detail=f"predicted {i}", metric_type="predicted")
    store.add_metrics(conn, cid, detail="actual result", metric_type="actual")

    hits = core.find_similar(conn, text="brief", top_k=5)  # full_detail=False (browsing default)
    kept_details = [m["detail"] for m in hits[0]["metrics"]]
    assert "actual result" in kept_details
    assert hits[0]["metrics_total"] == config.EVIDENCE_METRICS_MAX + 1
    assert hits[0]["metrics_truncated"] is True


def test_find_similar_metrics_trimming_most_recent_within_same_type(conn):
    import config
    import store

    cid = core.ingest_campaign(conn, title="X", detail="brief")["campaign_id"]
    for i in range(config.EVIDENCE_METRICS_MAX + 2):
        store.add_metrics(conn, cid, detail=f"actual {i}", metric_type="actual")

    hits = core.find_similar(conn, text="brief", top_k=5)
    kept_details = {m["detail"] for m in hits[0]["metrics"]}
    # the most recently added ones survive, not the earliest
    assert f"actual {config.EVIDENCE_METRICS_MAX + 1}" in kept_details
    assert "actual 0" not in kept_details
