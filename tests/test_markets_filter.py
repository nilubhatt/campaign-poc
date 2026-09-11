"""
Real-usage finding from live testing (2026-09-10): region/market are single exact-match
strings. A campaign whose activation actually spans several countries (e.g. a SEA launch
covering Malaysia, Singapore, Thailand, Philippines, Indonesia) can only be tagged with ONE
of them as region - querying region="Indonesia" for a campaign tagged region="Malaysia"
returns nothing, even though the campaign's activation genuinely included Indonesia. That
reads as "no such campaign," not "field can't represent this," which is worse than no filter.
`markets` is a new, additive list field (mirrors tags' shape/philosophy) alongside the
existing scalar region/market: region stays the single primary geo tag, market stays the
freeform grouping label (e.g. "SEA"), and markets carries the full list of countries/sub-
markets a campaign's activation actually touched, matched by membership.
"""
import core
import store


def test_insert_campaign_stores_markets(conn):
    cid = store.insert_campaign(conn, title="Khloe SEA", markets=["Malaysia", "Indonesia"])
    assert store.get_campaign(conn, cid)["markets"] == ["Malaysia", "Indonesia"]


def test_markets_defaults_to_empty_list(conn):
    cid = store.insert_campaign(conn, title="X")
    assert store.get_campaign(conn, cid)["markets"] == []


def test_markets_rejects_non_list(conn):
    import pytest
    with pytest.raises(ValueError):
        store.insert_campaign(conn, title="X", markets="Indonesia")


def test_markets_rejects_non_string_entries(conn):
    import pytest
    with pytest.raises(ValueError):
        store.insert_campaign(conn, title="X", markets=["Indonesia", 123])


def test_markets_strips_whitespace_and_dedupes_case_insensitively(conn):
    cid = store.insert_campaign(conn, title="X", markets=[" Indonesia ", "indonesia", "Malaysia"])
    assert store.get_campaign(conn, cid)["markets"] == ["Indonesia", "Malaysia"]


def test_markets_rejects_empty_string_entry(conn):
    import pytest
    with pytest.raises(ValueError):
        store.insert_campaign(conn, title="X", markets=["Indonesia", "  "])


def test_update_campaign_can_set_markets(conn):
    cid = store.insert_campaign(conn, title="X")
    store.update_campaign(conn, cid, markets=["Malaysia", "Singapore"])
    assert store.get_campaign(conn, cid)["markets"] == ["Malaysia", "Singapore"]


def test_update_campaign_markets_replaces_not_merges(conn):
    cid = store.insert_campaign(conn, title="X", markets=["Malaysia"])
    store.update_campaign(conn, cid, markets=["Indonesia"])
    assert store.get_campaign(conn, cid)["markets"] == ["Indonesia"]


def test_list_campaigns_shows_markets(conn):
    store.insert_campaign(conn, title="SEA", markets=["Malaysia", "Indonesia"])
    rows = store.list_campaigns(conn)
    assert rows[0]["markets"] == ["Malaysia", "Indonesia"]


def test_filter_campaign_ids_by_markets_membership_case_insensitive(conn):
    a = store.insert_campaign(conn, title="SEA", region="Malaysia",
                              markets=["Malaysia", "Singapore", "Indonesia"])
    b = store.insert_campaign(conn, title="LATAM", region="Mexico", markets=["Mexico"])

    ids = store.filter_campaign_ids(conn, markets="indonesia")
    assert ids == [a]

    ids2 = store.filter_campaign_ids(conn, markets="Mexico")
    assert ids2 == [b]


def test_filter_campaign_ids_markets_finds_nothing_for_absent_country(conn):
    store.insert_campaign(conn, title="SEA", markets=["Malaysia", "Singapore"])
    assert store.filter_campaign_ids(conn, markets="Thailand") == []


def test_filter_campaign_ids_markets_accepts_a_list_query_any_match(conn):
    """The write side takes a list (markets=["Malaysia","Indonesia"]) - the query side must
    too, with ANY-match semantics, or an LLM that just wrote a list will plausibly query
    with one too and hit a type error mid-demo. This is also how "Indonesia OR Thailand" -
    otherwise unreachable in one call - becomes answerable."""
    a = store.insert_campaign(conn, title="SEA", markets=["Malaysia", "Indonesia"])
    b = store.insert_campaign(conn, title="Other SEA", markets=["Thailand"])
    store.insert_campaign(conn, title="LATAM", markets=["Mexico"])

    ids = store.filter_campaign_ids(conn, markets=["Indonesia", "Thailand"])
    assert set(ids) == {a, b}

    # a single string still works exactly as before
    assert store.filter_campaign_ids(conn, markets="Indonesia") == [a]


def test_filter_campaign_ids_combines_region_and_markets(conn):
    """The real scenario: region is the single country a record is primarily tagged with,
    markets is the fuller activation list - both should be usable as independent filters,
    together or apart."""
    a = store.insert_campaign(conn, title="SEA", region="Malaysia",
                              markets=["Malaysia", "Indonesia"])
    store.insert_campaign(conn, title="Other SEA", region="Singapore",
                          markets=["Singapore", "Indonesia"])

    # region alone still only finds the Malaysia-tagged record
    assert store.filter_campaign_ids(conn, region="Malaysia") == [a]
    # markets alone finds BOTH records that touched Indonesia, regardless of region
    ids = store.filter_campaign_ids(conn, markets="Indonesia")
    assert len(ids) == 2


def test_find_similar_can_filter_by_markets(conn):
    a = core.ingest_campaign(conn, title="Khloe SEA", detail="activation seeding",
                             region="Malaysia", markets=["Malaysia", "Indonesia"])
    core.ingest_campaign(conn, title="Unrelated LATAM", detail="activation seeding",
                         region="Mexico", markets=["Mexico"])

    hits = core.find_similar(conn, text="activation seeding", markets="Indonesia")
    ids = [h["campaign_id"] for h in hits]
    assert ids == [a["campaign_id"]]


def test_evidence_rows_surface_markets(conn):
    core.ingest_campaign(conn, title="Khloe SEA", detail="seeding brief",
                         markets=["Malaysia", "Indonesia"])
    hits = core.find_similar(conn, text="seeding brief", top_k=5)
    assert hits[0]["markets"] == ["Malaysia", "Indonesia"]


def test_prepare_evaluation_can_filter_by_markets(conn):
    a = core.ingest_campaign(conn, title="Khloe SEA", detail="seeding brief",
                             markets=["Malaysia", "Indonesia"])
    core.ingest_campaign(conn, title="Unrelated", detail="seeding brief", markets=["Mexico"])

    result = core.prepare_evaluation(conn, subject_title="New SEA proposal",
                                     proposal_text="seeding brief", markets="Indonesia")
    ids = [e["campaign_id"] for e in result["evidence"]]
    assert ids == [a["campaign_id"]]
