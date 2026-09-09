"""
User feedback: "Four of our five performable tags are currently impression, not
measurements. Without that distinction, the tag reads as evidence when it isn't, and the
agent will weight it as if it were." Tags now carry a `source`: 'verified' (backed by real
metric_type='actual' data) or 'stated' (someone's claim, no data behind it) - and default
to 'stated', the conservative assumption, unless explicitly marked otherwise. A plain string
is accepted for ergonomics (LLM-first, non-technical users) and normalizes to
{"value": str, "source": "stated"}.
"""
import pytest

import core
import store


def test_plain_string_tag_defaults_to_stated(conn):
    cid = store.insert_campaign(conn, title="X", tags=["liked"])
    tags = store.get_campaign(conn, cid)["tags"]
    assert tags == [{"value": "liked", "source": "stated"}]


def test_explicit_verified_tag_is_preserved(conn):
    cid = store.insert_campaign(conn, title="X",
                                tags=[{"value": "performed_well", "source": "verified"}])
    tags = store.get_campaign(conn, cid)["tags"]
    assert tags == [{"value": "performed_well", "source": "verified"}]


def test_mixed_plain_and_object_tags(conn):
    cid = store.insert_campaign(conn, title="X", tags=[
        "liked",
        {"value": "performed_well", "source": "verified"},
    ])
    tags = store.get_campaign(conn, cid)["tags"]
    assert {"value": "liked", "source": "stated"} in tags
    assert {"value": "performed_well", "source": "verified"} in tags


def test_invalid_tag_source_rejected(conn):
    with pytest.raises(ValueError, match="source"):
        store.insert_campaign(conn, title="X",
                              tags=[{"value": "performed_well", "source": "definitely"}])


def test_tag_object_missing_value_rejected(conn):
    with pytest.raises(ValueError, match="value"):
        store.insert_campaign(conn, title="X", tags=[{"source": "verified"}])


def test_tag_neither_string_nor_object_rejected(conn):
    with pytest.raises(ValueError):
        store.insert_campaign(conn, title="X", tags=[42])


def test_update_campaign_tags_also_normalizes(conn):
    cid = store.insert_campaign(conn, title="X")
    store.update_campaign(conn, cid, tags=["not_liked", {"value": "underperformed", "source": "verified"}])
    tags = store.get_campaign(conn, cid)["tags"]
    assert {"value": "not_liked", "source": "stated"} in tags
    assert {"value": "underperformed", "source": "verified"} in tags


def test_filter_campaign_ids_by_tag_value_matches_regardless_of_source(conn):
    a = store.insert_campaign(conn, title="A", tags=["liked"])
    b = store.insert_campaign(conn, title="B", tags=[{"value": "liked", "source": "verified"}])
    store.insert_campaign(conn, title="C", tags=["not_liked"])

    matches = store.filter_campaign_ids(conn, tags=["liked"])
    assert set(matches) == {a, b}


def test_filter_campaign_ids_verified_tags_only(conn):
    stated = store.insert_campaign(conn, title="Stated", tags=["performed_well"])
    verified = store.insert_campaign(conn, title="Verified",
                                     tags=[{"value": "performed_well", "source": "verified"}])

    all_matches = store.filter_campaign_ids(conn, tags=["performed_well"])
    assert set(all_matches) == {stated, verified}

    verified_only = store.filter_campaign_ids(conn, tags=["performed_well"], verified_tags_only=True)
    assert verified_only == [verified]


def test_the_missing_quadrant_query_liked_and_underperformed(conn):
    """The actual query this feature exists to answer."""
    target = store.insert_campaign(conn, title="Mexico Record", tags=[
        "liked", {"value": "underperformed", "source": "verified"},
    ])
    store.insert_campaign(conn, title="Liked and did well", tags=["liked", "performed_well"])
    store.insert_campaign(conn, title="Disliked and underperformed",
                          tags=["not_liked", "underperformed"])

    quadrant = store.filter_campaign_ids(conn, tags=["liked", "underperformed"], match_all_tags=True)
    assert quadrant == [target]


def test_evidence_rows_surface_tag_source(conn):
    core.ingest_campaign(conn, title="X", detail="brief",
                         tags=[{"value": "performed_well", "source": "verified"}])
    hits = core.find_similar(conn, text="brief", top_k=5)
    assert hits[0]["tags"] == [{"value": "performed_well", "source": "verified"}]
