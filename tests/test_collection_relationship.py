"""
User feedback (original review that drove §6.1-6.9): "No relationships. Same source deck,
same collection different market, v1->v2 - all buried in prose." supersedes (§6.4) covers
v1->v2 (replacement, asymmetric). This covers the other case: market variants of the same
collection (e.g. "Khloe Q2 2026" launched in SEA, APAC, LATAM) - a symmetric grouping, not a
replacement. `collection` is freeform text like region/market, filterable the same way, so
siblings are explicitly linked instead of only discoverable by title-guessing.
"""
import core
import store


def test_insert_campaign_stores_collection(conn):
    cid = store.insert_campaign(conn, title="Khloe SEA Launch", collection="Khloe Q2 2026",
                                region="SEA")
    assert store.get_campaign(conn, cid)["collection"] == "Khloe Q2 2026"


def test_collection_defaults_to_none(conn):
    cid = store.insert_campaign(conn, title="X")
    assert store.get_campaign(conn, cid)["collection"] is None


def test_update_campaign_can_set_collection(conn):
    cid = store.insert_campaign(conn, title="X")
    store.update_campaign(conn, cid, collection="Khloe Q2 2026")
    assert store.get_campaign(conn, cid)["collection"] == "Khloe Q2 2026"


def test_filter_campaign_ids_by_collection_case_insensitive(conn):
    a = store.insert_campaign(conn, title="SEA", collection="Khloe Q2 2026")
    b = store.insert_campaign(conn, title="APAC", collection="khloe q2 2026")
    store.insert_campaign(conn, title="Unrelated", collection="Keke SS26")

    siblings = store.filter_campaign_ids(conn, collection="Khloe Q2 2026")
    assert set(siblings) == {a, b}


def test_list_campaigns_shows_collection(conn):
    store.insert_campaign(conn, title="SEA", collection="Khloe Q2 2026")
    rows = store.list_campaigns(conn)
    assert rows[0]["collection"] == "Khloe Q2 2026"


def test_find_similar_can_filter_by_collection(conn):
    a = core.ingest_campaign(conn, title="Khloe SEA", detail="influencer seeding",
                             collection="Khloe Q2 2026", region="SEA")
    core.ingest_campaign(conn, title="Unrelated", detail="influencer seeding",
                         collection="Keke SS26")

    hits = core.find_similar(conn, text="influencer seeding", collection="Khloe Q2 2026")
    ids = [h["campaign_id"] for h in hits]
    assert ids == [a["campaign_id"]]


def test_evidence_rows_surface_collection(conn):
    core.ingest_campaign(conn, title="Khloe SEA", detail="seeding brief",
                         collection="Khloe Q2 2026")
    hits = core.find_similar(conn, text="seeding brief", top_k=5)
    assert hits[0]["collection"] == "Khloe Q2 2026"
