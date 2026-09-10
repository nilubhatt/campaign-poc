"""
User feedback: "Four of our five performable tags are currently impression, not
measurements. Without that distinction, the tag reads as evidence when it isn't, and the
agent will weight it as if it were." Tags now carry a `source`: 'verified' (backed by real
metric_type='actual' data) or 'stated' (someone's claim, no data behind it) - and default
to 'stated', the conservative assumption, unless explicitly marked otherwise. A plain string
is accepted for ergonomics (LLM-first, non-technical users) and normalizes to
{"value": str, "source": "stated"}.

Two independent reviews (adversarial + design) found the first version of this incomplete:
'verified' had no structural connection to actual data (anyone could claim it), and the
global verified_tags_only flag couldn't express the actual quadrant query (a creative-
reaction tag like "liked" can never itself be "verified" - only a performance tag can be).
Both are fixed here: 'verified' now requires a metric_type='actual' record to already exist
on the campaign (enforced in store.py, not just documented), and tag queries carry
per-tag source instead of one global flag - tags=["liked", {"value": "underperformed",
"source": "verified"}] expresses "liked (any evidence) AND underperformed (verified only)"
precisely, which a blanket flag could never do.
"""
import pytest

import core
import store


def test_plain_string_tag_defaults_to_stated(conn):
    cid = store.insert_campaign(conn, title="X", tags=["liked"])
    tags = store.get_campaign(conn, cid)["tags"]
    assert tags == [{"value": "liked", "source": "stated"}]


def test_verified_tag_rejected_at_insert_time_no_metrics_can_exist_yet(conn):
    """A brand-new campaign cannot have metrics yet (chicken-and-egg) - verified must be
    earned via update_campaign after add_metrics, never claimed at creation."""
    with pytest.raises(ValueError, match="verified"):
        store.insert_campaign(conn, title="X",
                              tags=[{"value": "performed_well", "source": "verified"}])


def test_verified_tag_rejected_on_update_without_actual_metrics(conn):
    cid = store.insert_campaign(conn, title="X")
    with pytest.raises(ValueError, match="verified"):
        store.update_campaign(conn, cid, tags=[{"value": "performed_well", "source": "verified"}])


def test_verified_tag_accepted_once_actual_metrics_exist(conn):
    cid = store.insert_campaign(conn, title="X")
    store.add_metrics(conn, cid, detail="sell-through data", metric_type="actual")
    store.update_campaign(conn, cid, tags=[{"value": "performed_well", "source": "verified"}])
    tags = store.get_campaign(conn, cid)["tags"]
    assert tags == [{"value": "performed_well", "source": "verified"}]


def test_predicted_only_metrics_do_not_unlock_verified(conn):
    """Predicted metrics are a forecast, not evidence - only 'actual' unlocks 'verified'."""
    cid = store.insert_campaign(conn, title="X")
    store.add_metrics(conn, cid, detail="forecast", metric_type="predicted")
    with pytest.raises(ValueError, match="verified"):
        store.update_campaign(conn, cid, tags=[{"value": "performed_well", "source": "verified"}])


def test_mixed_plain_and_object_tags(conn):
    cid = store.insert_campaign(conn, title="X")
    store.add_metrics(conn, cid, detail="data", metric_type="actual")
    store.update_campaign(conn, cid, tags=[
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
        store.insert_campaign(conn, title="X", tags=[{"source": "stated"}])


def test_tag_neither_string_nor_object_rejected(conn):
    with pytest.raises(ValueError):
        store.insert_campaign(conn, title="X", tags=[42])


def test_empty_or_whitespace_only_tag_value_rejected_both_shapes(conn):
    with pytest.raises(ValueError):
        store.insert_campaign(conn, title="X", tags=["   "])
    with pytest.raises(ValueError):
        store.insert_campaign(conn, title="X", tags=[{"value": "  "}])


def test_tag_value_whitespace_is_stripped(conn):
    cid = store.insert_campaign(conn, title="X", tags=["  liked  "])
    assert store.get_campaign(conn, cid)["tags"] == [{"value": "liked", "source": "stated"}]


def test_duplicate_tag_value_deduped_preferring_verified(conn):
    cid = store.insert_campaign(conn, title="X", tags=["liked"])
    store.add_metrics(conn, cid, detail="data", metric_type="actual")
    # re-adding the same value as verified should replace, not duplicate, the stated one
    store.update_campaign(conn, cid, tags=["liked", "liked",
                                           {"value": "liked", "source": "verified"}])
    tags = store.get_campaign(conn, cid)["tags"]
    assert tags == [{"value": "liked", "source": "verified"}]


def test_filter_campaign_ids_by_tag_value_matches_regardless_of_source(conn):
    a = store.insert_campaign(conn, title="A", tags=["liked"])
    b = store.insert_campaign(conn, title="B")
    store.add_metrics(conn, b, detail="data", metric_type="actual")
    store.update_campaign(conn, b, tags=[{"value": "liked", "source": "verified"}])
    store.insert_campaign(conn, title="C", tags=["not_liked"])

    matches = store.filter_campaign_ids(conn, tags=["liked"])
    assert set(matches) == {a, b}


def test_filter_campaign_ids_per_tag_source_requires_that_specific_source(conn):
    stated = store.insert_campaign(conn, title="Stated", tags=["performed_well"])
    verified = store.insert_campaign(conn, title="Verified")
    store.add_metrics(conn, verified, detail="data", metric_type="actual")
    store.update_campaign(conn, verified, tags=[{"value": "performed_well", "source": "verified"}])

    any_source = store.filter_campaign_ids(conn, tags=["performed_well"])
    assert set(any_source) == {stated, verified}

    verified_only = store.filter_campaign_ids(
        conn, tags=[{"value": "performed_well", "source": "verified"}])
    assert verified_only == [verified]


def test_the_missing_quadrant_query_liked_and_underperformed(conn):
    """The actual query this feature exists to answer - a per-tag source constraint
    (underperformed must be verified) combined with a source-agnostic one (liked can be
    stated, since a creative reaction is never "verified" the way a performance claim is).
    A single global verified-only flag can't express this; per-tag source can."""
    target = store.insert_campaign(conn, title="Mexico Record", tags=["liked"])
    store.add_metrics(conn, target, detail="sell-through", metric_type="actual")
    store.update_campaign(conn, target, tags=[
        "liked", {"value": "underperformed", "source": "verified"},
    ])

    liked_and_did_well = store.insert_campaign(conn, title="Liked and did well",
                                               tags=["liked", "performed_well"])
    disliked_flop = store.insert_campaign(conn, title="Disliked and underperformed",
                                          tags=["not_liked", "underperformed"])

    quadrant = store.filter_campaign_ids(
        conn, tags=["liked", {"value": "underperformed", "source": "verified"}],
        match_all_tags=True,
    )
    assert quadrant == [target]
    assert liked_and_did_well not in quadrant
    assert disliked_flop not in quadrant  # "underperformed" here is stated, not verified


def test_tag_query_rejects_invalid_source(conn):
    store.insert_campaign(conn, title="X", tags=["liked"])
    with pytest.raises(ValueError, match="source"):
        store.filter_campaign_ids(conn, tags=[{"value": "liked", "source": "definitely"}])


def test_evidence_rows_surface_tag_source(conn):
    cid = core.ingest_campaign(conn, title="X", detail="brief")["campaign_id"]
    store.add_metrics(conn, cid, detail="data", metric_type="actual")
    store.update_campaign(conn, cid, tags=[{"value": "performed_well", "source": "verified"}])
    hits = core.find_similar(conn, text="brief", top_k=5)
    assert hits[0]["tags"] == [{"value": "performed_well", "source": "verified"}]
