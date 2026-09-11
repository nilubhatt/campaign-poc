"""
Adversarial review finding: `CREATE TABLE IF NOT EXISTS` does nothing for a column added to
an EXISTING table — a database from before `collection` existed (e.g. anyone who installed
the already-shipped v0.2.0 release) would fail on every insert/read. init_db() must migrate
existing databases additively, not just create fresh ones.
"""
import store


def test_init_db_adds_collection_column_to_a_pre_collection_database(conn):
    """Simulate an existing DB from before `collection` existed, then verify init_db()
    upgrades it in place rather than requiring a delete-and-reload."""
    store_conn = store.connect()
    store_conn.execute("DROP TABLE IF EXISTS campaigns")
    store_conn.execute("""
        CREATE TABLE campaigns (
            id TEXT PRIMARY KEY, title TEXT NOT NULL,
            record_type TEXT NOT NULL DEFAULT 'campaign', status TEXT,
            tags TEXT NOT NULL DEFAULT '[]', region TEXT, market TEXT,
            supersedes TEXT, superseded_by TEXT, detail TEXT, deck_text TEXT,
            asset_path TEXT, embedded INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL, updated_at REAL NOT NULL
        )
    """)
    store_conn.execute(
        "INSERT INTO campaigns (id, title, created_at, updated_at) VALUES ('camp_old', 'Old', 0, 0)"
    )
    store_conn.commit()

    columns_before = {r["name"] for r in store_conn.execute("PRAGMA table_info(campaigns)").fetchall()}
    assert "collection" not in columns_before
    store_conn.close()

    store.init_db()  # must not raise, must add the missing column

    conn2 = store.connect()
    columns_after = {r["name"] for r in conn2.execute("PRAGMA table_info(campaigns)").fetchall()}
    assert "collection" in columns_after

    # the pre-existing row survives and new operations work against it
    old = store.get_campaign(conn2, "camp_old")
    assert old is not None
    assert old["collection"] is None

    new_id = store.insert_campaign(conn2, title="New", collection="Test Collection")
    assert store.get_campaign(conn2, new_id)["collection"] == "Test Collection"
    conn2.close()


def test_init_db_is_a_noop_on_a_database_that_already_has_collection(conn):
    store.init_db()
    store.init_db()  # must not raise when called twice / column already present


def test_init_db_adds_markets_column_to_a_pre_markets_database(conn):
    """Same scenario, one column later: a v0.2.4-and-earlier database (has `collection`,
    not yet `markets`) must upgrade in place too - this is the exact real-world case anyone
    reinstalling over an existing database tonight will hit."""
    store_conn = store.connect()
    store_conn.execute("DROP TABLE IF EXISTS campaigns")
    store_conn.execute("""
        CREATE TABLE campaigns (
            id TEXT PRIMARY KEY, title TEXT NOT NULL,
            record_type TEXT NOT NULL DEFAULT 'campaign', status TEXT,
            tags TEXT NOT NULL DEFAULT '[]', region TEXT, market TEXT, collection TEXT,
            supersedes TEXT, detail TEXT, deck_text TEXT,
            asset_path TEXT, embedded INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL, updated_at REAL NOT NULL
        )
    """)
    store_conn.execute(
        "INSERT INTO campaigns (id, title, created_at, updated_at) VALUES ('camp_old', 'Old', 0, 0)"
    )
    store_conn.commit()

    columns_before = {r["name"] for r in store_conn.execute("PRAGMA table_info(campaigns)").fetchall()}
    assert "markets" not in columns_before
    store_conn.close()

    store.init_db()

    conn2 = store.connect()
    columns_after = {r["name"] for r in conn2.execute("PRAGMA table_info(campaigns)").fetchall()}
    assert "markets" in columns_after

    old = store.get_campaign(conn2, "camp_old")
    assert old is not None
    assert old["markets"] == []

    new_id = store.insert_campaign(conn2, title="New", markets=["Malaysia", "Indonesia"])
    assert store.get_campaign(conn2, new_id)["markets"] == ["Malaysia", "Indonesia"]
    conn2.close()
