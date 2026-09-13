"""
§8.2: an unknown metric becomes provisional and asks — never rejected, never silently accepted.

The review: *"An unfamiliar key on ingest should not fail, and should not be silently accepted
either. Create a provisional registry entry and surface it once, in plain language:*

    New measure seen: footfall_uplift_pct (Colombia, 22 Sept)
    Looks similar to: retail_traffic_uplift_pct  [same thing] [different measure] [ignore]
    Ask for it in future store launches? [yes] [not yet]

*That one question is what keeps the vocabulary clean, and it is the only human step in the
loop."*

Both halves of "never fail, never silently accept" are load-bearing, and they fail in opposite
directions. **Rejecting** an unfamiliar key loses the measurement — the marketer has the number
and the product refuses it, so it goes into freeform prose and nothing can ever compare it.
**Silently accepting** is how two campaigns produced 25 keys: every one was accepted, and
nobody was ever asked whether `crm_reach` was the `reach` already on file.

**"Surface it once"** is the constraint that makes the loop bearable. A prompt on every write
is a prompt nobody reads, and the same unknown key arriving in ten campaigns must ask once, not
ten times. Once answered, it never asks again — including when the answer was "ignore", because
re-asking a question somebody has declined is how a product teaches people to dismiss it.

**The suggestion is the part that keeps the vocabulary clean.** "Looks similar to
`retail_traffic_uplift_pct`" is what turns the question from a chore into a decision somebody
can make in a second — and §5.1 already established that a suggestion must never be the
default, because a wrong alias silently merges two measures that are not the same.
"""
import pytest

import metrics
import store


def _campaign(conn, title="Colombia launch", **kw):
    import core
    return core.ingest_campaign(conn, title=title, status="concluded",
                                detail="A launch.", **kw)["campaign_id"]


# ── never fail, never silently accept ───────────────────────────────────────

def test_an_unfamiliar_key_is_not_rejected(conn):
    """Rejecting loses the measurement. The marketer has the number and the product refuses
    it, so it goes into freeform prose where nothing can ever compare it."""
    cid = _campaign(conn)
    result = metrics.record(conn, campaign_id=cid, key="footfall_uplift_pct", value=12.5)

    assert result["value"] == 12.5
    assert metrics.values_for(conn, cid, result["metric"])[0]["value"] == 12.5


def test_an_unfamiliar_key_is_not_silently_accepted(conn):
    """Silent acceptance is how two campaigns produced 25 keys: every one was accepted, and
    nobody was ever asked whether `crm_reach` was the `reach` already on file."""
    cid = _campaign(conn)
    result = metrics.record(conn, campaign_id=cid, key="footfall_uplift_pct", value=12.5)

    assert result["new_measure"]
    assert result["new_measure"]["raw_key"] == "footfall_uplift_pct"
    assert metrics.describe(conn, result["metric"])["status"] == "provisional"


def test_a_known_key_asks_nothing(conn):
    cid = _campaign(conn)
    assert "new_measure" not in metrics.record(conn, campaign_id=cid, key="crm_reach",
                                              value=180000)


# ── the question, in the review's own shape ─────────────────────────────────

def test_the_question_names_the_measure_the_campaign_and_when(conn):
    """"New measure seen: footfall_uplift_pct (Colombia, 22 Sept)." A key on its own is not a
    question somebody can answer — they need to know which brief brought it in."""
    cid = _campaign(conn, "Colombia launch")
    asked = metrics.record(conn, campaign_id=cid, key="footfall_uplift_pct",
                           value=12.5)["new_measure"]

    assert asked["raw_key"] == "footfall_uplift_pct"
    assert asked["campaign_title"] == "Colombia launch"
    assert asked["seen_at"]


def test_it_suggests_the_measure_it_most_resembles(conn):
    """"Looks similar to: retail_traffic_uplift_pct." This is what turns the question from a
    chore into a decision somebody can make in a second."""
    cid = _campaign(conn)
    metrics.record(conn, campaign_id=cid, key="retail_traffic_uplift_pct", value=8.0)

    asked = metrics.record(conn, campaign_id=_campaign(conn, "Peru"),
                           key="retail_traffic_growth_pct", value=9.0)["new_measure"]
    assert asked["looks_like"] == "retail_traffic_uplift"


def test_a_key_that_resembles_nothing_suggests_nothing(conn):
    """A suggestion that is always present is a suggestion nobody trusts — and §5.1 settled
    that a wrong alias is worse than none, because it silently merges two measures that are
    not the same thing."""
    cid = _campaign(conn)
    asked = metrics.record(conn, campaign_id=cid, key="zzz_qqq_vvv",
                           value=1.0)["new_measure"]
    assert asked["looks_like"] is None


def test_the_three_choices_are_offered_as_callable_actions(conn):
    """"[same thing] [different measure] [ignore]" — §5.2's rule is that an offer whose write
    path does not exist is worse than no offer."""
    import mcp_server

    cid = _campaign(conn)
    metrics.record(conn, campaign_id=cid, key="retail_traffic_uplift_pct", value=8.0)
    asked = metrics.record(conn, campaign_id=_campaign(conn, "Peru"),
                           key="retail_traffic_growth_pct", value=9.0)["new_measure"]

    tools = {a["tool"] for a in asked["next_actions"]}
    assert tools == {"resolve_measure"}
    assert {a["prefilled_args"]["decision"] for a in asked["next_actions"]} == \
        {"same_thing", "different_measure", "ignore"}
    for action in asked["next_actions"]:
        assert hasattr(mcp_server, action["tool"])


# ── surfaced once ───────────────────────────────────────────────────────────

def test_the_same_unknown_key_asks_only_once(conn):
    """"Surface it once." The same key arriving in ten campaigns must ask once, not ten
    times — a prompt on every write is a prompt nobody reads."""
    first = metrics.record(conn, campaign_id=_campaign(conn, "A"),
                           key="footfall_uplift_pct", value=1.0)
    second = metrics.record(conn, campaign_id=_campaign(conn, "B"),
                            key="footfall_uplift_pct", value=2.0)

    assert first["new_measure"]
    assert "new_measure" not in second


def test_a_measure_that_was_answered_never_asks_again(conn):
    cid = _campaign(conn)
    metrics.record(conn, campaign_id=cid, key="footfall_uplift_pct", value=1.0)
    metrics.resolve(conn, "footfall_uplift", decision="different_measure")

    assert "new_measure" not in metrics.record(conn, campaign_id=_campaign(conn, "B"),
                                               key="footfall_uplift_pct", value=2.0)


def test_a_measure_somebody_declined_never_asks_again(conn):
    """Including when the answer was "ignore". Re-asking a question somebody has declined is
    how a product teaches people to dismiss it."""
    cid = _campaign(conn)
    metrics.record(conn, campaign_id=cid, key="junk_key", value=1.0)
    metrics.resolve(conn, "junk_key", decision="ignore")

    assert "new_measure" not in metrics.record(conn, campaign_id=cid, key="junk_key",
                                               value=2.0)


def test_what_is_still_unanswered_can_be_listed(conn):
    """The question is asked once, so something has to hold the ones nobody answered — or
    "surface once" becomes "surface once and lose"."""
    metrics.record(conn, campaign_id=_campaign(conn), key="footfall_uplift_pct", value=1.0)
    metrics.record(conn, campaign_id=_campaign(conn, "B"), key="dwell_time_s", value=2.0)

    assert sorted(m["canonical"] for m in metrics.unanswered(conn)) == \
        ["dwell_time", "footfall_uplift"]


# ── the three answers ───────────────────────────────────────────────────────

def test_same_thing_folds_the_key_into_the_existing_measure(conn):
    """And retrospectively: the values already recorded under the provisional name belong to
    the measure it turned out to be, or the answer fixes the vocabulary and loses the data."""
    cid = _campaign(conn)
    metrics.record(conn, campaign_id=cid, key="reach", value=100.0)
    metrics.record(conn, campaign_id=cid, key="audience_size", value=200.0)

    metrics.resolve(conn, "audience_size", decision="same_thing", same_as="reach")

    assert metrics.canonical(conn, "audience_size") == "reach"
    assert sorted(r["value"] for r in metrics.values_for(conn, cid, "reach")) == [100.0, 200.0]
    assert metrics.describe(conn, "audience_size") is None


def test_same_thing_needs_to_say_which_thing(conn):
    cid = _campaign(conn)
    metrics.record(conn, campaign_id=cid, key="audience_size", value=200.0)
    with pytest.raises(ValueError) as e:
        metrics.resolve(conn, "audience_size", decision="same_thing")
    assert "same_as" in str(e.value)


def test_different_measure_keeps_it_and_stops_asking(conn):
    cid = _campaign(conn)
    metrics.record(conn, campaign_id=cid, key="footfall_uplift_pct", value=1.0)
    metrics.resolve(conn, "footfall_uplift", decision="different_measure")

    entry = metrics.describe(conn, "footfall_uplift")
    assert entry["status"] == "provisional", "still not expected — that gate is §8.3's"
    assert entry["answered"] is True


def test_ignore_keeps_the_values_and_stops_asking(conn):
    """"Ignore" is a decision about the VOCABULARY, not about the data. Deleting the values
    would make a dismissive click destroy a measurement somebody recorded."""
    cid = _campaign(conn)
    metrics.record(conn, campaign_id=cid, key="junk_key", value=7.0)
    metrics.resolve(conn, "junk_key", decision="ignore")

    assert metrics.values_for(conn, cid, "junk_key")[0]["value"] == 7.0
    assert metrics.describe(conn, "junk_key")["status"] == "ignored"


def test_an_unknown_decision_is_refused(conn):
    cid = _campaign(conn)
    metrics.record(conn, campaign_id=cid, key="junk_key", value=7.0)
    with pytest.raises(ValueError) as e:
        metrics.resolve(conn, "junk_key", decision="maybe")
    assert "same_thing" in str(e.value)


def test_it_reaches_the_model_over_the_protocol(conn):
    import asyncio
    import json
    import mcp_server

    async def call(name, args):
        return json.loads((await mcp_server.mcp.call_tool(name, args)).content[0].text)

    cid = _campaign(conn)
    added = asyncio.run(call("add_metrics", {
        "campaign_id": cid, "structured": {"footfall_uplift_pct": 12.5}, "confirm": True}))

    assert added["new_measures"][0]["raw_key"] == "footfall_uplift_pct"
    resolved = asyncio.run(call("resolve_measure",
                                {"measure": "footfall_uplift", "decision": "ignore"}))
    assert resolved["status"] == "ignored"
