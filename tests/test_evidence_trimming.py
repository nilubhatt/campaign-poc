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
