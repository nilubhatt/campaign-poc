"""
Idea A: "Make the tools forgiving, and make errors teach."

    "Three enum values were rejected before the right one was found: `client_stated` for a
    tag source (valid: verified, stated), `target` for a metric type (valid: actual,
    predicted), and a bare string where a list was required. Each rejection said what was
    wrong, none said what was right.

    Fix: every enum error returns the valid set and the closest match. Better still,
    normalise on the way in — "client stated", "stated by client" and "client_stated" all
    mean stated. The intended user is a marketer in a conversation, not an engineer reading
    a schema."

Reproduced before changing anything. The bare-string case is already fixed. The other two
are rejected by pydantic at the schema boundary, before any of this project's code runs, so
the message is pydantic's: it names the valid set but never the closest match, and it cannot
normalise because it never sees the value. The tag-source case is worse than reported — the
union of `str | {value, source}` reports only the first branch's failure:

    tags.0.str
      Input should be a valid string [input_value={'value': 'liked', 'source': 'client_stated'}]

which tells a marketer their object should be a string, and never mentions `source` at all.

So the schema keeps advertising the enum (it is what stops a well-behaved caller guessing),
while the Python types accept a superset and this code does the normalising and the teaching.
"""
import json

import pytest

import enums
import store


# ── normalising on the way in ───────────────────────────────────────────────

@pytest.mark.parametrize("given", [
    "stated", "Stated", " stated ", "client_stated", "client stated",
    "stated by client", "client-stated", "self reported", "CLIENT STATED",
])
def test_every_way_of_saying_stated_means_stated(given):
    assert enums.normalise(given, field="tag 'source'", valid=store.VALID_TAG_SOURCES,
                           synonyms=enums.TAG_SOURCE_SYNONYMS) == "stated"


@pytest.mark.parametrize("given", ["verified", "Verified", "measured", "from metrics",
                                   "backed by data"])
def test_every_way_of_saying_verified_means_verified(given):
    assert enums.normalise(given, field="tag 'source'", valid=store.VALID_TAG_SOURCES,
                           synonyms=enums.TAG_SOURCE_SYNONYMS) == "verified"


@pytest.mark.parametrize("given,expected", [
    ("in flight", "in_flight"), ("in-flight", "in_flight"), ("live", "in_flight"),
    ("running", "in_flight"), ("concluded", "concluded"), ("finished", "concluded"),
    ("done", "concluded"), ("proposed", "proposed"), ("draft", "proposed"),
    ("a pitch", "proposed"),
])
def test_status_is_normalised_the_way_a_marketer_says_it(given, expected):
    assert enums.normalise(given, field="status", valid=store.VALID_STATUSES,
                           synonyms=enums.STATUS_SYNONYMS) == expected


def test_the_shape_of_the_word_is_enough_without_a_synonym_entry():
    """Case, spacing and punctuation are not meaning. A synonym table that has to list
    "In Flight" as well as "in flight" is a table nobody can keep complete."""
    assert enums.normalise("In_Flight", field="status", valid=store.VALID_STATUSES,
                           synonyms={}) == "in_flight"


# ── teaching when it genuinely cannot be guessed ────────────────────────────

def test_an_unknown_value_is_told_the_valid_set_and_the_closest_match():
    with pytest.raises(ValueError) as exc:
        enums.normalise("predicated", field="metric_type",
                        valid=store.VALID_METRIC_TYPES, synonyms=enums.METRIC_TYPE_SYNONYMS)

    message = str(exc.value)
    assert "actual" in message and "predicted" in message, "the valid set"
    assert "predicted" in message.split("closest")[-1] or "did you mean" in message.lower()


def test_a_value_with_no_near_match_still_gets_the_valid_set():
    """A suggestion that is not actually close is worse than none — it sends somebody to
    retype a word that will also be rejected."""
    with pytest.raises(ValueError) as exc:
        enums.normalise("bananas", field="status", valid=store.VALID_STATUSES,
                        synonyms=enums.STATUS_SYNONYMS)

    message = str(exc.value)
    assert "proposed" in message and "in_flight" in message and "concluded" in message
    assert "did you mean" not in message.lower(), "nothing here is close to 'bananas'"


def test_target_is_not_silently_recorded_as_a_prediction():
    """The reviewer's own example, and the one place normalising would be wrong. A target is
    what somebody wants to happen; a prediction is what this library expects to happen.
    Filing one as the other corrupts every later reconciliation — which compares what was
    predicted against what occurred — so this teaches instead of guessing."""
    with pytest.raises(ValueError) as exc:
        enums.normalise("target", field="metric_type", valid=store.VALID_METRIC_TYPES,
                        synonyms=enums.METRIC_TYPE_SYNONYMS)

    message = str(exc.value).lower()
    assert "actual" in message and "predicted" in message
    assert "target" in message and "aim" in message or "want" in message or "goal" in message


# ── through the real call paths ─────────────────────────────────────────────

def test_a_tag_source_a_marketer_typed_is_accepted(conn):
    cid = store.insert_campaign(conn, title="Colombia",
                                tags=[{"value": "liked", "source": "client stated"}])

    stored = store.get_campaign(conn, cid)["tags"]
    assert stored[0]["source"] == "stated"


def test_a_status_a_marketer_typed_is_accepted(conn):
    cid = store.insert_campaign(conn, title="Colombia", status="in flight")

    assert store.get_campaign(conn, cid)["status"] == "in_flight"


def test_a_metric_type_a_marketer_typed_is_accepted(conn):
    cid = store.insert_campaign(conn, title="Colombia")
    store.add_metrics(conn, cid, metric_type="Actual", detail="CTR 1.2%")

    assert store.get_campaign(conn, cid)["metrics"][0]["metric_type"] == "actual"


def test_a_bad_tag_source_names_the_field_and_not_the_union(conn):
    """The reproduced failure: `str | {value, source}` reported the first branch, so the
    message was "Input should be a valid string" about a dict, and `source` was never
    mentioned."""
    with pytest.raises(ValueError) as exc:
        store.insert_campaign(conn, title="Colombia",
                              tags=[{"value": "liked", "source": "hearsay"}])

    message = str(exc.value)
    assert "source" in message
    assert "verified" in message and "stated" in message
    assert "valid string" not in message


def test_a_filter_normalises_the_same_way_as_a_write(conn):
    """Otherwise a value accepted on the way in cannot be used to search for itself."""
    store.insert_campaign(conn, title="Colombia", status="in flight",
                          tags=[{"value": "liked", "source": "client stated"}])

    assert store.filter_campaign_ids(conn, status="In Flight")
    assert store.filter_campaign_ids(conn, tags=[{"value": "liked",
                                                  "source": "stated by client"}])


def test_record_type_is_forgiving_too(conn):
    cid = store.insert_campaign(conn, title="A rulebook", record_type="Reference")

    assert store.get_campaign(conn, cid)["record_type"] == "reference"


# ── the schema still teaches, which is what stops the guess happening ───────

def test_the_published_schema_still_advertises_the_valid_values():
    """Accepting a superset must not cost the caller the list. The enum in the schema is
    what stops a well-behaved client guessing in the first place; the normalising is for
    when it guesses anyway."""
    import mcp_server

    tools = {t.name: t.parameters for t in mcp_server.mcp._tool_manager.list_tools()}

    status = json.dumps(tools["upload_campaign"]["properties"]["status"])
    assert "in_flight" in status and "concluded" in status

    metric = json.dumps(tools["add_metrics"]["properties"]["metric_type"])
    assert "actual" in metric and "predicted" in metric


def test_a_preview_validates_what_it_is_previewing(conn):
    """Found over the real protocol while checking idea A: `add_metrics(confirm=False)`
    returned before any validation, so a marketer was shown "metric_type: target" as though
    it were about to be saved — and the rejection arrived only after they said yes. A
    preview of something that cannot be stored is worse than no preview."""
    cid = store.insert_campaign(conn, title="Colombia")

    with pytest.raises(ValueError) as exc:
        core.add_metrics(conn, cid, metric_type="target", detail="CTR 2%", confirm=False)

    assert "predicted" in str(exc.value)


def test_a_preview_shows_the_value_it_would_actually_store(conn):
    """The other half: a preview that shows what the user typed rather than what will be
    recorded hides the normalisation, so nobody can correct it if it guessed wrong."""
    cid = store.insert_campaign(conn, title="Colombia")

    preview = core.add_metrics(conn, cid, metric_type="Results", detail="CTR 1.2%",
                               confirm=False)

    assert preview["metric_type"] == "actual"


import core  # noqa: E402  (used by the two tests above)
