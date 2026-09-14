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
normalise because it never sees the value.

A correction, from review: the first version of this file claimed the `str | {value, source}`
union reported only its first branch and never named `source`. It names both. The probe
printed "2 validation errors" and the output was truncated at 300 characters before the
second was read. What loosening the type actually buys is a 66-character message naming one
field, rather than a two-branch dump whose first line tells a marketer their object should be
a string.

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


def test_target_is_still_never_a_prediction():
    """The reviewer's own example, and the distinction has not changed even though the answer
    has. A target is what somebody WANTS to happen; a prediction is what this library expects
    to happen, and reconciliation scores the library against its predictions. Filing one as the
    other would score it against somebody's ambition.

    §8.1/D33 built the place a target belongs — a `metric_type` of its own, comparable against
    the actual — so it is accepted now. What it must never be is folded into `predicted`."""
    assert enums.normalise("target", field="metric_type", valid=store.VALID_METRIC_TYPES,
                           synonyms=enums.METRIC_TYPE_SYNONYMS) == "target"
    assert enums.normalise("goal", field="metric_type", valid=store.VALID_METRIC_TYPES,
                           synonyms=enums.METRIC_TYPE_SYNONYMS) == "target"
    assert "target" not in [v for k, v in enums.METRIC_TYPE_SYNONYMS.items()
                            if v == "predicted"]
    assert enums.METRIC_TYPE_SYNONYMS["forecast"] == "predicted"


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
    """Pydantic named `source` too (see the module docstring's correction); what it could not
    do is say it in one line, or normalise. This pins the one-line version."""
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
        core.add_metrics(conn, cid, metric_type="Targett", detail="CTR 2%", confirm=False)

    assert "predicted" in str(exc.value)


def test_a_preview_shows_the_value_it_would_actually_store(conn):
    """The other half: a preview that shows what the user typed rather than what will be
    recorded hides the normalisation, so nobody can correct it if it guessed wrong."""
    cid = store.insert_campaign(conn, title="Colombia")

    preview = core.add_metrics(conn, cid, metric_type="Results", detail="CTR 1.2%",
                               confirm=False)

    assert preview["metric_type"] == "actual"


import core  # noqa: E402  (used by the two tests above)


# ══ design review of 5.1 ═════════════════════════════════════════════════════

def test_the_preview_shows_what_will_actually_be_stored(conn):
    """The preview IS the correction screen — it is the only moment a marketer can catch a
    synonym that guessed wrong. It was showing the value they typed, so "live" looked like
    what would be recorded and `in_flight` went in silently. The same defect was fixed for
    add_metrics in this item and left in the flagship flow."""
    preview = core.ingest_campaign(conn, title="Colombia", detail="d",
                                   record_type="Campaign", status="live", confirm=False)

    assert preview["record_type"] == "campaign"
    assert preview["status"] == "in_flight"


def test_the_preview_applies_the_same_default_as_the_write(conn):
    """Worse than a cosmetic mismatch: the default status was computed against the RAW
    record_type while the write computed it against the normalised one, so
    record_type="Campaign" previewed `status: None` and committed `concluded`. The user
    agreed to one record and got another."""
    preview = core.ingest_campaign(conn, title="Colombia", detail="d",
                                   record_type="Campaign", confirm=False)
    stored = store.get_campaign(conn, core.ingest_campaign(
        conn, title="Colombia", detail="d", record_type="Campaign",
        confirm=True)["campaign_id"])

    assert preview["status"] == stored["status"]


def test_a_write_says_when_it_changed_what_you_gave_it(conn):
    """Silent normalisation is only acceptable if every write says what it stored. Shape
    changes are lossless and not worth mentioning; a SYNONYM is a guess, and a guess the
    marketer never hears about is one they can never correct."""
    result = core.ingest_campaign(conn, title="Colombia", detail="d", status="live",
                                  confirm=True)

    changed = {n["field"]: n for n in result["normalised"]}
    assert changed["status"]["given"] == "live"
    assert changed["status"]["stored_as"] == "in_flight"


def test_a_write_does_not_announce_a_change_it_did_not_make(conn):
    """"Recording this as in flight" said about somebody who typed `in_flight` is noise, and
    noise is how the real ones stop being read."""
    result = core.ingest_campaign(conn, title="Colombia", detail="d", status="in_flight",
                                  confirm=True)

    assert result.get("normalised") == []


def test_bulk_import_is_as_forgiving_as_a_single_write(conn):
    """A spreadsheet column headed "Results" is the single most likely place these words
    arrive, and it was the one path still using the old strict check — the same vocabulary
    with two behaviours."""
    cid = store.insert_campaign(conn, title="Colombia")

    result = store.bulk_import_metrics(conn, [
        {"campaign_id": cid, "metric_type": "Results", "detail": "CTR 1.2%"}], confirm=True)

    assert result["imported"] == 1
    assert store.get_campaign(conn, cid)["metrics"][0]["metric_type"] == "actual"


def test_bulk_import_teaches_the_same_way_too(conn):
    """"Targett" rather than "target": the latter is a value now (§8.1/D33). A typo still has
    to teach, and on the batch path it has to carry the retry as DATA — D47, which is the one
    thing this path had that the single write did not."""
    cid = store.insert_campaign(conn, title="Colombia")

    result = store.bulk_import_metrics(conn, [
        {"campaign_id": cid, "metric_type": "Targett", "detail": "CTR 2%"}], confirm=True)

    assert result["imported"] == 0
    assert result["errors"][0]["field"] == "metric_type"
    assert result["errors"][0]["valid"] == list(store.VALID_METRIC_TYPES)


def test_bulk_import_accepts_a_target_now_that_there_is_somewhere_for_it(conn):
    cid = store.insert_campaign(conn, title="Colombia")

    result = store.bulk_import_metrics(conn, [
        {"campaign_id": cid, "metric_type": "Target", "structured": {"roas": 4.0}}],
        confirm=True)

    assert result["errors"] == []
    assert store.get_campaign(conn, cid)["metrics"][0]["metric_type"] == "target"


def test_past_is_not_treated_as_concluded():
    """Temporal, not lifecycle: a cancelled campaign is also "past". This was the wrong-guess
    case the echo above exists to catch, baked into the table as a certainty."""
    with pytest.raises(ValueError):
        enums.normalise("past", field="status", valid=store.VALID_STATUSES,
                        synonyms=enums.STATUS_SYNONYMS)


def test_confirmed_is_not_treated_as_verified():
    """`verified` has a hard definition here — backed by a metric_type='actual' row, enforced
    on write. "The client confirmed it worked" is a stated claim. The write side would have
    caught it when no metrics existed; the FILTER side has no such guard, so a query for
    "confirmed" silently narrowed to measured evidence."""
    with pytest.raises(ValueError):
        enums.normalise("confirmed", field="tag 'source'", valid=store.VALID_TAG_SOURCES,
                        synonyms=enums.TAG_SOURCE_SYNONYMS)


def test_a_status_the_library_has_no_home_for_is_not_guessed_at():
    """difflib measures string similarity, not meaning: "cancelled" is three edits from
    "concluded" and the opposite of it. Suggesting it would file an abandoned campaign as a
    finished one, and every later "what worked" query would count it."""
    with pytest.raises(ValueError) as exc:
        enums.normalise("cancelled", field="status", valid=store.VALID_STATUSES,
                        synonyms=enums.STATUS_SYNONYMS)

    message = str(exc.value)
    assert "Did you mean 'concluded'" not in message
    assert "cancel" in message.lower()


def test_nothing_still_sends_a_target_into_freeform_prose(conn):
    """The refusal used to say "it belongs in the campaign's detail", with careful advice about
    `update_campaign` replacing rather than appending. §8.1/D33 then built the place it
    actually belongs, and the refusal outlived it by two items — telling people to put a
    number into text that nothing can compare, which is the drift this file exists to prevent,
    in this file.

    A vocabulary's advice has to be retired when the thing it routed around gets built."""
    import re

    source = open("enums.py", encoding="utf-8").read()
    explain = source[source.index("_EXPLAIN = {"):source.index("def _teach")]
    assert not re.search(r'\("metric_type", "(target|goal)"\)', explain)


def test_the_docstring_does_not_teach_the_word_the_server_refuses():
    """The shared prompt said "'predicted' (a forecast/target set before launch)". Claude
    would map "our target is 2% CTR" to `predicted` on that authority and never reach the
    teaching error — the string-literal drift pattern, where the prose outlives the rule it
    described."""
    import re
    from pathlib import Path

    source = Path("mcp_server.py").read_text(encoding="utf-8")
    body = source[source.index("def add_metrics"):]
    docstring = body[:body.index('"""', body.index('"""') + 3)]

    assert not re.search(r"forecast/target|target set before", docstring), (
        "the docstring still equates a target with a prediction"
    )


# ══ adversarial review of 5.1 ════════════════════════════════════════════════

def test_the_error_message_never_suggests_the_opposite_of_what_was_typed():
    """The worst possible suggestion, and difflib rated it 0.89: `unverified` is three
    letters from `verified` and its exact denial. A model retrying with the suggestion would
    mark an unverified claim as measured evidence — a provenance upgrade delivered BY the
    error message, on the one field this library weighs judgments by."""
    for given in ("unverified", "not verified", "un-verified", "not_verified"):
        with pytest.raises(ValueError) as exc:
            enums.normalise(given, field="tag 'source'", valid=store.VALID_TAG_SOURCES,
                            synonyms=enums.TAG_SOURCE_SYNONYMS)
        assert "did you mean" not in str(exc.value).lower(), given


def test_a_coincidence_of_letters_is_not_a_suggestion():
    """`approved` scores 0.625 against `proposed` and `cancelled` 0.667 against `concluded`
    — both above the old cutoff, both meaning something else. A real typo scores 0.82 and up,
    so the line goes between them."""
    for given, wrong in (("approved", "proposed"), ("cancelled", "concluded")):
        with pytest.raises(ValueError) as exc:
            enums.normalise(given, field="status", valid=store.VALID_STATUSES,
                            synonyms=enums.STATUS_SYNONYMS)
        assert f"Did you mean {wrong!r}" not in str(exc.value), given


@pytest.mark.parametrize("typo,expected", [
    ("predicated", "predicted"), ("in_flite", "in_flight"), ("conclude", "concluded"),
])
def test_a_real_typo_still_gets_its_suggestion(typo, expected):
    """The other half: tightening the cutoff must not cost the case the feature exists for."""
    valid = store.VALID_METRIC_TYPES if expected == "predicted" else store.VALID_STATUSES
    with pytest.raises(ValueError) as exc:
        enums.normalise(typo, field="f", valid=valid, synonyms={})

    assert f"Did you mean {expected!r}" in str(exc.value)


def test_the_suggestion_half_is_actually_asserted():
    """The earlier version of this check read
        `"predicted" in message.split("closest")[-1] or "did you mean" in message.lower()`
    and with no "closest" in the message, `split` returns the whole message — in which
    "predicted" appears anyway, as part of the valid set. Deleting the suggestion branch
    entirely left the suite green."""
    with pytest.raises(ValueError) as exc:
        enums.normalise("predicated", field="metric_type", valid=store.VALID_METRIC_TYPES,
                        synonyms=enums.METRIC_TYPE_SYNONYMS)

    assert "Did you mean 'predicted'?" in str(exc.value)


def test_the_explanations_are_keyed_to_field_names_that_are_actually_used():
    """`_EXPLAIN` is keyed by the `field` string, so a caller passing "metric type" instead
    of "metric_type" would silently lose every explanation and fall back to a bare list."""
    import re
    from pathlib import Path

    used = set(re.findall(r'field="([^"]+)"', Path("store.py").read_text(encoding="utf-8")))
    used |= set(re.findall(r"field='([^']+)'", Path("core.py").read_text(encoding="utf-8")))
    used |= set(re.findall(r'field="([^"]+)"', Path("core.py").read_text(encoding="utf-8")))

    for (field, _value) in enums._EXPLAIN:
        assert field in used, (
            f"_EXPLAIN is keyed on {field!r}, which no call site passes — the explanation is "
            f"unreachable"
        )


def test_a_bare_tag_is_accepted_when_writing_as_well_as_when_filtering(conn):
    """"A bare string where a list was required" was one of the reviewer's three rejections.
    Only the FILTER had been fixed: `upload_campaign(tags="liked")` still failed with
    "Input should be a valid list"."""
    cid = store.insert_campaign(conn, title="Colombia", tags="liked")

    assert store.get_campaign(conn, cid)["tags"][0]["value"] == "liked"


def test_a_blank_value_means_not_saying_rather_than_something_else(conn):
    """Blank used to raise everywhere. It now means "unset", which is defensible — but it
    has to mean the same thing on both paths, and a spreadsheet import hits it constantly."""
    created = store.insert_campaign(conn, title="Colombia", status="")
    omitted = store.insert_campaign(conn, title="Colombia 2")

    assert store.get_campaign(conn, created)["status"] == \
        store.get_campaign(conn, omitted)["status"]

    store.update_campaign(conn, created, status="   ")
    assert store.get_campaign(conn, created)["status"] == \
        store.get_campaign(conn, omitted)["status"], "blank must not clear a set value"


def test_no_tool_description_still_says_a_target_is_rejected(conn):
    """§5.1's own defect — "the shared prompt was teaching the word the server refuses" —
    inverted. `add_metrics`'s docstring told the model a target is "rejected on purpose" and
    belongs in the campaign's prose, for two items after §8.1 built the place it goes. The
    prompt is the model's authority: it overrides the teaching error, so a stale one is worse
    than a stale comment."""
    import mcp_server

    for text in (mcp_server.add_metrics.__doc__,
                 mcp_server.bulk_import_metrics.__doc__,
                 str(enums._EXPLAIN.get(("metric_type", "benchmark"), ""))):
        assert "rejected on purpose" not in (text or "")
        assert "goal belongs in the campaign" not in (text or "")
    assert "target" in mcp_server.add_metrics.__doc__
