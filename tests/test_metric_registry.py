"""
§8.1: a metric registry — canonical key, display name, unit, direction, aliases, first seen,
times seen, campaign types, status. Values in typed storage, not an opaque JSON string.

The review's evidence: *"Two campaigns carry structured metrics. Between them they have
introduced 25 distinct keys, with no registry, no canonical names and no units. The collisions
are already visible: `reach` / `crm_reach` / `stated_combined_influencer_reach` — three
'reach'; `local_influencer_egr_pct` / `er_social_pct` — EGR vs ER; `impressions` /
`impressions_upper_funnel` — scope in the key; `total_budget_mxn` / `total_budget_usd` —
currency in the key; `planned_roas_july_stated` / `planned_roas_july_recomputed` — a correction
as a key. That last pair is the clearest signal that the schema is failing: 'the partner's
figure is wrong and here is the right one' had nowhere to go, so it became a key name. At
fifty campaigns this is unusable. Worse, `structured` is stored as a JSON string, not an
object. The library cannot answer 'which campaigns have sell-through data' or 'show me every
ROAS on file' — the one question a metric store exists to answer."*

Two failures, and they are different in kind.

**The keys collide**, because nothing canonicalises them. A registry with aliases is what turns
three spellings of reach into one measure with three names on record.

**The values cannot be queried**, because a JSON string is not storage. "Show me every ROAS"
is the question a metric store exists to answer, and today it requires reading every row and
parsing it.

The last collision is the one worth dwelling on: `planned_roas_july_stated` versus
`_recomputed` is not a naming problem at all. It is a correction with nowhere to live, and a
registry that only canonicalises names would rename both halves and lose the point. So the
value row carries its own provenance — whose figure this is — and a correction is a second
value for the same measure rather than a second measure.
"""
import pytest

import metrics
import store


def _campaign(conn, title="Peru launch", **kw):
    import core
    return core.ingest_campaign(conn, title=title, status="concluded",
                                detail="A launch.", **kw)["campaign_id"]


# ── the collisions the review found ─────────────────────────────────────────

@pytest.mark.parametrize("spelling", ["reach", "crm_reach",
                                      "stated_combined_influencer_reach"])
def test_three_spellings_of_reach_are_one_measure(conn, spelling):
    """The review's first collision. Three keys, one thing being counted."""
    assert metrics.canonical(conn, spelling) == "reach"


def test_egr_and_er_are_one_measure(conn):
    assert metrics.canonical(conn, "local_influencer_egr_pct") == \
        metrics.canonical(conn, "er_social_pct") == "engagement_rate"


def test_scope_in_the_key_does_not_make_a_new_measure(conn):
    """`impressions` and `impressions_upper_funnel` are impressions. The scope is a fact about
    the row, not about the measure, and putting it in the key made it unqueryable."""
    assert metrics.canonical(conn, "impressions_upper_funnel") == "impressions"


def test_currency_in_the_key_does_not_make_a_new_measure(conn):
    """`total_budget_mxn` and `total_budget_usd` are both budget. Currency is a UNIT, and a
    unit in the key means the library cannot add two budgets or compare them."""
    assert metrics.canonical(conn, "total_budget_mxn") == "budget"
    assert metrics.unit_of(conn, "total_budget_mxn") == "MXN"
    assert metrics.unit_of(conn, "total_budget_usd") == "USD"


# ── the collision that is not a naming problem ──────────────────────────────

def test_a_correction_is_a_second_value_not_a_second_measure(conn):
    """"`planned_roas_july_stated` / `planned_roas_july_recomputed` — a correction as a key.
    That last pair is the clearest signal that the schema is failing."

    A registry that only canonicalised names would rename both halves to `roas` and lose the
    thing that distinguishes them. The correction needs somewhere to live."""
    cid = _campaign(conn)
    metrics.record(conn, campaign_id=cid, key="planned_roas_july_stated", value=4.2)
    metrics.record(conn, campaign_id=cid, key="planned_roas_july_recomputed", value=2.8)

    rows = metrics.values_for(conn, cid, "roas")
    assert [r["value"] for r in rows] == [4.2, 2.8]
    assert [r["source"] for r in rows] == ["stated", "recomputed"]


def test_the_recomputed_figure_is_the_one_to_weigh(conn):
    """The correction exists because somebody decided the partner's figure was wrong. A store
    that keeps both and says nothing has recorded a disagreement without resolving it."""
    cid = _campaign(conn)
    metrics.record(conn, campaign_id=cid, key="planned_roas_july_stated", value=4.2)
    metrics.record(conn, campaign_id=cid, key="planned_roas_july_recomputed", value=2.8)

    assert metrics.best_value(conn, cid, "roas")["value"] == 2.8


# ── the question a metric store exists to answer ────────────────────────────

def test_every_roas_on_file_can_be_listed(conn):
    """"The library cannot answer 'which campaigns have sell-through data' or 'show me every
    ROAS on file' — the one question a metric store exists to answer." A JSON string is not
    storage."""
    a, b = _campaign(conn, "Peru"), _campaign(conn, "Chile")
    metrics.record(conn, campaign_id=a, key="roas", value=3.1)
    metrics.record(conn, campaign_id=b, key="planned_roas_july_recomputed", value=2.8)
    metrics.record(conn, campaign_id=a, key="reach", value=180000)

    every = metrics.across_library(conn, "roas")
    assert {r["campaign_id"] for r in every} == {a, b}
    assert sorted(r["value"] for r in every) == [2.8, 3.1]


def test_which_campaigns_have_a_measure_can_be_asked(conn):
    a, b = _campaign(conn, "Peru"), _campaign(conn, "Chile")
    metrics.record(conn, campaign_id=a, key="sell_through_pct", value=62.0)

    assert metrics.campaigns_with(conn, "sell_through") == [a]
    assert b not in metrics.campaigns_with(conn, "sell_through")


# ── what the registry holds about each measure ──────────────────────────────

def test_a_registered_measure_carries_what_a_reader_needs(conn):
    entry = metrics.describe(conn, "roas")

    assert entry["display_name"]
    assert entry["unit"] == "ratio"
    assert entry["direction"] == "higher_is_better"
    assert "planned_roas_july_stated" in entry["aliases"] or entry["aliases"]


def test_direction_is_recorded_because_higher_is_not_always_better(conn):
    """A store that does not know which way a measure runs cannot say whether a campaign did
    well, and "cost per acquisition went up" is good news to anything that assumes higher is
    better."""
    assert metrics.describe(conn, "cpa")["direction"] == "lower_is_better"


def test_first_seen_and_times_seen_are_tracked(conn):
    """§8.3's graduation gate needs both, and §8.5's retirement needs the last. Recorded from
    the first write rather than added later, so the history is not missing for exactly the
    measures that arrived before somebody thought about it."""
    cid = _campaign(conn)
    metrics.record(conn, campaign_id=cid, key="roas", value=3.1)
    metrics.record(conn, campaign_id=_campaign(conn, "Chile"), key="roas", value=2.2)

    entry = metrics.describe(conn, "roas")
    assert entry["times_seen"] == 2
    assert entry["first_seen"] and entry["last_seen"]


def test_the_campaign_types_a_measure_appears_on_are_tracked(conn):
    """"Campaign types" in the review's list. A measure that only ever appears on retail
    launches is not a measure every brief is missing."""
    cid = _campaign(conn, "Peru", market="LATAM")
    metrics.record(conn, campaign_id=cid, key="sell_through_pct", value=62.0)

    assert metrics.describe(conn, "sell_through")["markets"] == ["LATAM"]


# ── typed storage, and what that buys ───────────────────────────────────────

def test_a_value_is_stored_as_a_number(conn):
    cid = _campaign(conn)
    metrics.record(conn, campaign_id=cid, key="reach", value="180,000")

    assert metrics.values_for(conn, cid, "reach")[0]["value"] == 180000.0


def test_a_value_that_is_not_a_number_is_refused_with_its_own_message(conn):
    cid = _campaign(conn)
    with pytest.raises(ValueError) as e:
        metrics.record(conn, campaign_id=cid, key="reach", value="lots")
    assert "reach" in str(e.value) and "lots" in str(e.value)


def test_the_raw_key_is_kept_beside_the_canonical_one(conn):
    """Canonicalising is a claim about what somebody meant. Keeping what they wrote is what
    makes the claim checkable, and what lets a wrong alias be found later."""
    cid = _campaign(conn)
    metrics.record(conn, campaign_id=cid, key="stated_combined_influencer_reach", value=5.0)

    row = metrics.values_for(conn, cid, "reach")[0]
    assert row["raw_key"] == "stated_combined_influencer_reach"
    assert row["metric"] == "reach"
