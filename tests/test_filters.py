"""
§6.2/§6.3: structured fields (record_type, status, tags, region, market) replace the old
binary `kind`, and filtering happens BEFORE similarity ranking so a single-brand corpus can
actually discriminate ("concluded campaigns in APAC" -> then rank by similarity).
"""
import core
import store


def test_insert_campaign_stores_structured_fields(conn):
    cid = store.insert_campaign(conn, title="APAC Launch", record_type="campaign",
                                status="concluded", tags=["seeding", "awareness"],
                                region="APAC", market="Philippines")
    c = store.get_campaign(conn, cid)
    assert c["record_type"] == "campaign"
    assert c["status"] == "concluded"
    assert c["tags"] == ["seeding", "awareness"]
    assert c["region"] == "APAC"
    assert c["market"] == "Philippines"


def test_insert_campaign_defaults_record_type_and_tags(conn):
    cid = store.insert_campaign(conn, title="X")
    c = store.get_campaign(conn, cid)
    assert c["record_type"] == "campaign"
    assert c["status"] == "concluded"
    assert c["tags"] == []
    assert c["region"] is None
    assert c["market"] is None


def test_filter_campaign_ids_by_record_type(conn):
    a = store.insert_campaign(conn, title="A", record_type="campaign")
    b = store.insert_campaign(conn, title="B", record_type="reference")
    c = store.insert_campaign(conn, title="C", record_type="stub")

    assert store.filter_campaign_ids(conn, record_type="campaign") == [a]
    assert set(store.filter_campaign_ids(conn, record_type=None)) == {a, b, c}


def test_filter_campaign_ids_by_status(conn):
    a = store.insert_campaign(conn, title="A", status="concluded")
    b = store.insert_campaign(conn, title="B", status="proposed")
    store.insert_campaign(conn, title="C", status="in_flight")

    assert store.filter_campaign_ids(conn, status="concluded") == [a]
    assert store.filter_campaign_ids(conn, status="proposed") == [b]


def test_filter_campaign_ids_by_region_and_market_case_insensitive(conn):
    a = store.insert_campaign(conn, title="A", region="APAC", market="Philippines")
    store.insert_campaign(conn, title="B", region="LATAM", market="Mexico")

    assert store.filter_campaign_ids(conn, region="apac") == [a]
    assert store.filter_campaign_ids(conn, market="philippines") == [a]
    assert store.filter_campaign_ids(conn, region="EMEA") == []


def test_filter_campaign_ids_by_tags_any_match(conn):
    a = store.insert_campaign(conn, title="A", tags=["seeding", "awareness"])
    b = store.insert_campaign(conn, title="B", tags=["performance"])
    store.insert_campaign(conn, title="C", tags=[])

    assert store.filter_campaign_ids(conn, tags=["seeding"]) == [a]
    assert set(store.filter_campaign_ids(conn, tags=["awareness", "performance"])) == {a, b}
    assert store.filter_campaign_ids(conn, tags=["nonexistent"]) == []


def test_filter_campaign_ids_combines_all_filters(conn):
    a = store.insert_campaign(conn, title="A", record_type="campaign", status="concluded",
                              region="APAC", tags=["seeding"])
    store.insert_campaign(conn, title="B", record_type="campaign", status="concluded",
                          region="LATAM", tags=["seeding"])
    store.insert_campaign(conn, title="C", record_type="campaign", status="proposed",
                          region="APAC", tags=["seeding"])

    assert store.filter_campaign_ids(conn, status="concluded", region="APAC",
                                     tags=["seeding"]) == [a]


def test_filter_campaign_ids_excludes_given_id(conn):
    a = store.insert_campaign(conn, title="A", region="APAC")
    b = store.insert_campaign(conn, title="B", region="APAC")
    assert store.filter_campaign_ids(conn, region="APAC", exclude_campaign_id=a) == [b]


def test_find_similar_filters_before_ranking(conn):
    apac = core.ingest_campaign(conn, title="APAC Seeding Push", record_type="campaign",
                                status="concluded", region="APAC", tags=["seeding"],
                                detail="Influencer seeding across APAC social channels.")
    latam = core.ingest_campaign(conn, title="LATAM Seeding Push", record_type="campaign",
                                 status="concluded", region="LATAM", tags=["seeding"],
                                 detail="Influencer seeding across LATAM social channels.")

    # LATAM text is closer in wording, but region filter should exclude it entirely.
    hits = core.find_similar(conn, text="Influencer seeding across LATAM social channels.",
                             region="APAC", top_k=5)
    ids = [h["campaign_id"] for h in hits]
    assert ids == [apac["campaign_id"]]
    assert latam["campaign_id"] not in ids


def test_find_similar_returns_empty_when_no_campaign_matches_filters(conn):
    core.ingest_campaign(conn, title="X", region="APAC", detail="some brief")
    hits = core.find_similar(conn, text="some brief", region="EMEA", top_k=5)
    assert hits == []


def test_find_similar_without_filters_behaves_as_before(conn):
    core.ingest_campaign(conn, title="A", region="APAC", detail="audience targeting brief")
    core.ingest_campaign(conn, title="B", region="LATAM", detail="audience targeting brief")
    hits = core.find_similar(conn, text="audience targeting brief", top_k=5)
    assert len(hits) == 2
