"""
Design review finding: record_type/status/metric_type were free text at every layer with no
validation - an LLM typo ("inflight" instead of "in_flight") silently succeeds and the
record becomes unfindable by any status filter, with no error anywhere.
"""
import pytest

import store


def test_insert_campaign_rejects_invalid_record_type(conn):
    with pytest.raises(ValueError, match="record_type"):
        store.insert_campaign(conn, title="X", record_type="not-a-real-type")


def test_insert_campaign_rejects_invalid_status(conn):
    with pytest.raises(ValueError, match="status"):
        store.insert_campaign(conn, title="X", status="inflight")  # missing underscore


def test_insert_campaign_accepts_valid_values(conn):
    cid = store.insert_campaign(conn, title="X", record_type="reference", status=None)
    assert store.get_campaign(conn, cid)["record_type"] == "reference"


def test_insert_campaign_rejects_non_string_tags(conn):
    with pytest.raises(ValueError, match="tags"):
        store.insert_campaign(conn, title="X", tags=["ok", 42])


def test_insert_campaign_rejects_tags_that_is_not_a_list(conn):
    with pytest.raises(ValueError, match="tags"):
        store.insert_campaign(conn, title="X", tags="not-a-list")


def test_update_campaign_rejects_invalid_record_type(conn):
    cid = store.insert_campaign(conn, title="X")
    with pytest.raises(ValueError, match="record_type"):
        store.update_campaign(conn, cid, record_type="bogus")


def test_update_campaign_rejects_invalid_status(conn):
    cid = store.insert_campaign(conn, title="X")
    with pytest.raises(ValueError, match="status"):
        store.update_campaign(conn, cid, status="bogus")


def test_update_campaign_rejects_non_string_tags(conn):
    cid = store.insert_campaign(conn, title="X")
    with pytest.raises(ValueError, match="tags"):
        store.update_campaign(conn, cid, tags=[1, 2, 3])


def test_filter_campaign_ids_rejects_non_string_tags_in_query(conn):
    store.insert_campaign(conn, title="X", tags=["seeding"])
    with pytest.raises(ValueError, match="tags"):
        store.filter_campaign_ids(conn, tags=[42])
