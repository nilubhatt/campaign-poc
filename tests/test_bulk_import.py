"""
§8.8: the workbook import is where the registry gets seeded.

The review: *"The rubric names a companion workbook as the source of truth for KPI results, and
it has never arrived. When it does, it will carry a column vocabulary — that is the moment the
registry should be populated properly rather than accreted key by key. Fix:
`bulk_import_metrics` should diff incoming columns against the registry and report what is new,
what it thinks are aliases, and what it cannot type — before writing anything. An import that
silently accepts 40 new keys is how the current drift started."*

**"Before writing anything" is the requirement, and it is the whole shape of this item.** A
workbook carries a COLUMN VOCABULARY, and the moment to look at a vocabulary is once, as a
vocabulary — not forty times, one key at a time, after the writes have already happened. So the
import previews by default and writes on `confirm`, which is the pattern `upload_campaign` and
`add_metrics` already use for exactly this reason: the preview IS the consent step.

**Four answers, because the review asks for four.** What is already known; what it *thinks* are
aliases (a suggestion, never a merge — §5.1 and §8.2 both settled that); what is genuinely new;
and what it CANNOT TYPE, which is the one a reader would otherwise discover row by row as the
import half-failed.

*Found while reading the code:* the bulk path called `store.add_metrics` directly, so a
workbook import never reached `metrics.record` and never touched the registry at all. The item
the review describes as "where the registry gets seeded" seeded nothing, and forty columns went
into a JSON blob exactly as the review says they should not.

D47 also lands here: the batch path flattened `BadValue` to a string, so the structured retry
was available on the single-write path and not on the one the plan itself named as the likeliest
place "Target" arrives.
"""
import pytest

import core
import metrics
import store


# Forty distinct column names. Numbered ones would not do: `_strip_decoration` reads a
# trailing digit as a period (`planned_roas_july_2024`), so `house_metric_0` and
# `house_metric_1` are correctly one measure — which is the right rule and the wrong fixture.
_FORTY = ("footfall uplift queue basket dwell greet scan loyalty refund exchange "
          "voucher sample demo fitting mirror kiosk locker signage window aisle "
          "endcap shelf plinth mannequin lighting music scent seating counter till "
          "bagging entrance exit escalator lift stair ramp door canopy awning").split()


def _campaign(conn, title, market="LATAM"):
    return core.ingest_campaign(conn, title=title, market=market, status="concluded",
                                detail="A launch.")["campaign_id"]


def _rows(conn, **structured):
    return [{"campaign_id": _campaign(conn, "Peru launch"), "structured": dict(structured)}]


# ── before writing anything ─────────────────────────────────────────────────

def test_it_previews_the_column_vocabulary_and_writes_nothing(conn):
    """"Before writing anything." A workbook carries a vocabulary, and the moment to look at a
    vocabulary is once — not forty times, one key at a time, after the fact."""
    rows = _rows(conn, reach=180000, footfall_uplift_pct=12.5)

    preview = store.bulk_import_metrics(conn, rows)

    assert preview["preview"] is True
    assert preview["imported"] == 0
    assert metrics.across_library(conn, "reach") == []


def test_confirming_writes(conn):
    rows = _rows(conn, reach=180000)
    store.bulk_import_metrics(conn, rows)

    done = store.bulk_import_metrics(conn, rows, confirm=True)

    assert done["imported"] == 1
    assert [r["value"] for r in metrics.across_library(conn, "reach")] == [180000.0]


def test_the_import_reaches_the_registry_at_all(conn):
    """The bulk path called `store.add_metrics` directly, so a workbook import never reached
    `metrics.record`. The item the review calls "where the registry gets seeded" seeded
    nothing, and the columns went into a JSON blob exactly as the review says they must not."""
    cid = _campaign(conn, "Peru launch")
    store.bulk_import_metrics(conn, [{"campaign_id": cid,
                                      "structured": {"crm_reach": 180000}}], confirm=True)

    row = metrics.values_for(conn, cid, "reach")[0]
    assert row["value"] == 180000.0
    assert row["raw_key"] == "crm_reach", "canonicalised, and what they wrote is kept"


# ── the four answers ────────────────────────────────────────────────────────

def test_it_says_which_columns_are_already_known(conn):
    preview = store.bulk_import_metrics(conn, _rows(conn, crm_reach=1, total_budget_usd=2))

    known = {c["column"]: c["measure"] for c in preview["columns"]["known"]}
    assert known == {"crm_reach": "reach", "total_budget_usd": "budget"}


def test_it_says_what_it_thinks_are_aliases_without_merging_them(conn):
    """"What it thinks are aliases" — a suggestion, never a merge. §5.1 settled that a wrong
    alias silently merges two measures that are not the same thing, and a workbook is the worst
    place to get that wrong because it arrives forty columns at a time."""
    cid = _campaign(conn, "Peru launch")
    metrics.record(conn, campaign_id=cid, key="retail_traffic_uplift_pct", value=8.0)

    preview = store.bulk_import_metrics(
        conn, [{"campaign_id": cid, "structured": {"retail_traffic_growth_pct": 9.0}}])

    maybe = preview["columns"]["looks_like"]
    assert maybe[0]["column"] == "retail_traffic_growth_pct"
    assert maybe[0]["looks_like"] == "retail_traffic_uplift"
    assert maybe[0]["next_actions"], "offered, not applied"


def test_it_says_what_is_genuinely_new(conn):
    preview = store.bulk_import_metrics(conn, _rows(conn, zzz_qqq_vvv=1))

    assert [c["column"] for c in preview["columns"]["new"]] == ["zzz_qqq_vvv"]


def test_it_says_what_it_cannot_type(conn):
    """The answer a reader would otherwise discover row by row, as the import half-failed."""
    preview = store.bulk_import_metrics(conn, _rows(conn, reach="lots", roas=3.1))

    cannot = preview["columns"]["cannot_type"]
    assert [c["column"] for c in cannot] == ["reach"]
    assert "lots" in cannot[0]["example"]


def test_a_column_is_classified_once_however_many_rows_carry_it(conn):
    """It is a COLUMN vocabulary. Forty rows of the same workbook are one vocabulary, and
    reporting a column forty times is the row-at-a-time reading this item exists to replace."""
    rows = [{"campaign_id": _campaign(conn, f"Campaign {n}"),
             "structured": {"footfall_uplift_pct": n}} for n in range(1, 6)]

    preview = store.bulk_import_metrics(conn, rows)

    assert len(preview["columns"]["new"]) == 1
    assert preview["columns"]["new"][0]["rows"] == 5


def test_the_preview_says_what_would_happen(conn):
    preview = store.bulk_import_metrics(conn, _rows(conn, reach=1, zzz_qqq_vvv=2))

    assert "nothing has been stored" in preview["what_it_means"].lower()
    assert "confirm" in preview["what_it_means"].lower()


# ── an import that silently accepts 40 new keys ─────────────────────────────

def test_a_workbook_of_unknown_columns_is_not_quietly_absorbed(conn):
    """"An import that silently accepts 40 new keys is how the current drift started." The
    preview is what makes forty a number somebody sees before it is forty rows in a table."""
    cid = _campaign(conn, "Peru launch")
    wide = {f"house_{w}": n for n, w in enumerate(_FORTY)}

    preview = store.bulk_import_metrics(conn, [{"campaign_id": cid, "structured": wide}])

    assert len(preview["columns"]["new"]) == 40
    assert "40" in preview["what_it_means"]


def test_confirming_a_wide_import_still_records_each_as_provisional(conn):
    """Confirmed, they are still provisional measures nobody has answered for — §8.2's gate
    does not weaken because forty arrived at once."""
    cid = _campaign(conn, "Peru launch")
    wide = {"dwell_time_s": 1, "queue_length": 2, "basket_size": 3}
    store.bulk_import_metrics(conn, [{"campaign_id": cid, "structured": wide}], confirm=True)

    assert {m["canonical"] for m in metrics.unanswered(conn)} == \
        {"dwell_time", "queue_length", "basket_size"}


# ── D47: the structured retry on the batch path too ─────────────────────────

def test_a_rejected_enum_carries_its_retry_on_the_batch_path(conn):
    """D47. The batch path flattened `BadValue` to a string, so the structured retry —
    field, valid, suggestion — was available on the single write and not on the one the plan
    itself named as the likeliest place "Target" arrives."""
    rows = [{"campaign_id": _campaign(conn, "Peru launch"), "metric_type": "Targett",
             "structured": {"roas": 4.0}}]

    result = store.bulk_import_metrics(conn, rows, confirm=True)

    bad = result["errors"][0]
    assert bad["field"] == "metric_type"
    assert bad["given"] == "Targett"
    assert "target" in [v.lower() for v in bad["valid"]] or bad["suggestion"]


def test_a_recognised_synonym_is_still_accepted(conn):
    """The retry exists because the vocabulary is forgiving, not instead of it."""
    rows = [{"campaign_id": _campaign(conn, "Peru launch"), "metric_type": "Target",
             "structured": {"roas": 4.0}}]

    result = store.bulk_import_metrics(conn, rows, confirm=True)

    assert result["errors"] == []
    assert metrics.target_for(conn, rows[0]["campaign_id"], "roas")["value"] == 4.0


def test_a_bad_row_does_not_stop_the_batch(conn):
    good = _campaign(conn, "Peru launch")
    rows = [{"campaign_id": "camp_invented", "structured": {"roas": 1.0}},
            {"campaign_id": good, "structured": {"roas": 2.0}}]

    result = store.bulk_import_metrics(conn, rows, confirm=True)

    assert result["imported"] == 1
    assert result["errors"][0]["row"] == 0


def test_a_value_that_is_not_a_number_does_not_lose_the_rest_of_the_row(conn):
    """The single-write path keeps the other nine figures when one is unparseable, and the
    batch path has to behave the same way or the same workbook means two different things."""
    cid = _campaign(conn, "Peru launch")
    store.bulk_import_metrics(conn, [{"campaign_id": cid,
                                      "structured": {"reach": "lots", "roas": 3.1}}],
                              confirm=True)

    assert metrics.values_for(conn, cid, "roas")[0]["value"] == 3.1


def test_it_reaches_the_model_over_the_protocol(conn):
    import asyncio
    import json

    import mcp_server

    async def call(name, args):
        return json.loads((await mcp_server.mcp.call_tool(name, args)).content[0].text)

    cid = _campaign(conn, "Peru launch")
    rows = [{"campaign_id": cid, "structured": {"crm_reach": 180000,
                                                "footfall_uplift_pct": 12.5}}]

    preview = asyncio.run(call("bulk_import_metrics", {"rows": rows}))
    assert preview["preview"] is True
    assert [c["measure"] for c in preview["columns"]["known"]] == ["reach"]
    assert [c["column"] for c in preview["columns"]["new"]] == ["footfall_uplift_pct"]

    done = asyncio.run(call("bulk_import_metrics", {"rows": rows, "confirm": True}))
    assert done["imported"] == 1


# ── what making `target` a value reaches ────────────────────────────────────

def test_a_campaign_holding_only_a_target_has_not_been_measured(conn):
    """`target` was unstorable before §8.8, so "has a metrics row" and "has a result" were the
    same question everywhere. They are not, and the offer this gates is literally "record what
    this campaign actually achieved" — asking a campaign that has only said what it hopes for
    to stop being asked is the §8.1 mistake ("a target is not a result") one layer out."""
    cid = _campaign(conn, "Peru launch")
    store.bulk_import_metrics(conn, [{"campaign_id": cid, "metric_type": "target",
                                      "structured": {"roas": 4.0}}], confirm=True)

    record = store.get_campaign(conn, cid)
    assert record["has_metrics"] is True, "there is a row"
    assert record["has_actual_metrics"] is False, "and nothing has been measured"


def test_it_is_still_asked_for_its_results(conn):
    """Through the real path, not through `after_upload` directly — the call site is where the
    two readings of "has metrics" diverge, and asserting on the helper would pass whichever
    one the caller passed it."""
    v1 = _campaign(conn, "Colombia v1")
    core.save_evaluation(conn, subject_title="Colombia v1", campaign_id=v1, verdict="approve",
                         summary="Looks sound.", findings=[])
    v2 = _campaign(conn, "Colombia v2")
    store.bulk_import_metrics(conn, [{"campaign_id": v2, "metric_type": "target",
                                      "structured": {"roas": 4.0}}], confirm=True)

    out = core.update_campaign(conn, campaign_id=v2, supersedes=v1)
    assert "add_metrics" in [o["tool"] for o in out["next_actions"]]


def test_and_stops_being_asked_once_it_has_them(conn):
    v1 = _campaign(conn, "Colombia v1")
    core.save_evaluation(conn, subject_title="Colombia v1", campaign_id=v1, verdict="approve",
                         summary="Looks sound.", findings=[])
    v2 = _campaign(conn, "Colombia v2")
    store.bulk_import_metrics(conn, [{"campaign_id": v2, "metric_type": "actual",
                                      "structured": {"roas": 3.1}}], confirm=True)

    out = core.update_campaign(conn, campaign_id=v2, supersedes=v1)
    assert "add_metrics" not in [o["tool"] for o in out["next_actions"]]


def test_a_target_never_counts_as_evidence_a_claim_is_verified(conn):
    """The gate this library weighs judgments by: `verified` means backed by a measured
    outcome. A target satisfying it would buy that provenance with an ambition."""
    cid = _campaign(conn, "Peru launch")
    store.bulk_import_metrics(conn, [{"campaign_id": cid, "metric_type": "target",
                                      "structured": {"roas": 4.0}}], confirm=True)

    with pytest.raises(ValueError) as e:
        core.update_campaign(conn, campaign_id=cid,
                             tags=[{"value": "overperformed", "source": "verified"}])
    assert "actual" in str(e.value)


def test_a_target_is_not_weighed_as_a_result_anywhere(conn):
    """§8.1 already settled this inside `metrics`; §8.8 made it reachable from the surface."""
    cid = _campaign(conn, "Peru launch")
    store.bulk_import_metrics(conn, [
        {"campaign_id": cid, "metric_type": "target", "structured": {"roas": 4.0}},
        {"campaign_id": cid, "metric_type": "actual", "structured": {"roas": 3.1}}],
        confirm=True)

    assert metrics.best_value(conn, cid, "roas")["value"] == 3.1
    assert metrics.against_target(conn, cid, "roas")["met"] is False
    assert cid in store.campaigns_with_actual_metrics(conn)


def test_a_target_column_is_stored_as_a_target(conn):
    """"Target Reach" beside "Actual Reach" is the ordinary shape of a KPI workbook, and a row
    carries ONE metric_type for every column in it — so without reading the type out of the
    column name the sibling shape cannot be expressed at all, and `target_reach` decorated to a
    stem ending in `_reach` and was filed as a measured reach."""
    cid = _campaign(conn, "Peru launch")
    store.bulk_import_metrics(conn, [{"campaign_id": cid, "structured": {
        "Target Reach": 200000, "Actual Reach": 180000}}], confirm=True)

    by_type = {r["metric_type"]: r["value"] for r in metrics.values_for(conn, cid, "reach")}
    assert by_type == {"target": 200000.0, "actual": 180000.0}
    assert metrics.against_target(conn, cid, "reach")["met"] is False


def test_a_column_that_contradicts_its_row_is_refused(conn):
    """Two claims about one number. Picking one silently is how a target becomes a result."""
    cid = _campaign(conn, "Peru launch")
    result = store.bulk_import_metrics(conn, [{"campaign_id": cid, "metric_type": "predicted",
                                               "structured": {"Target Reach": 1}}],
                                       confirm=True)
    assert "cannot be both" in result["skipped"][0]["reason"]
    assert metrics.values_for(conn, cid, "reach") == []


def test_the_gate_counts_campaigns_that_measured_it_not_ones_that_hoped(conn):
    """Three target rows graduated a measure nobody had ever measured — and `expected_check`,
    which correctly counts actuals only, then reported those same three campaigns as missing
    it. Two halves of one item disagreeing about what counts."""
    for market in ("MX", "CO", "PE"):
        store.bulk_import_metrics(conn, [{
            "campaign_id": _campaign(conn, f"t{market}", market=market),
            "metric_type": "target", "structured": {"sell_through_pct": 60}}], confirm=True)

    gate = metrics.graduation(conn, "sell_through")
    assert gate["campaigns"] == 0
    assert gate["eligible"] is False


def test_a_target_does_not_revive_a_retired_measure(conn):
    """Reviving a standing requirement on somebody writing down what they hope for is the same
    mistake the gate made one function up."""
    for market in ("MX", "CO", "PE"):
        metrics.record(conn, campaign_id=_campaign(conn, f"m{market}", market=market),
                       key="sell_through_pct", value=60)
    metrics.graduate(conn, "sell_through", confirmed_by="R. Vega")
    store.retire_metric(conn, "sell_through")

    metrics.record(conn, campaign_id=_campaign(conn, "Later"), key="sell_through_pct",
                   value=70, metric_type="target")
    assert metrics.describe(conn, "sell_through")["status"] == "retired"

    metrics.record(conn, campaign_id=_campaign(conn, "Measured"), key="sell_through_pct",
                   value=70)
    assert metrics.describe(conn, "sell_through")["status"] == "expected"


# ── the questions the import creates ────────────────────────────────────────

def test_the_questions_the_import_raises_come_back_with_it(conn):
    """`metrics.record` marks a measure surfaced and offered as a SIDE EFFECT, so discarding
    the return consumed §8.2's "is this a new measure?" for every column at once and showed
    none of them. The item whose headline is "never silently accept" made forty acceptances
    unaskable, permanently — the only remaining route being a tool §8.4 describes as one a
    user would have to know exists and name by canonical stem to use."""
    cid = _campaign(conn, "Peru launch")
    result = store.bulk_import_metrics(
        conn, [{"campaign_id": cid, "structured": {"zz_house": 1, "reach": 2}}], confirm=True)

    assert [q["raw_key"] for q in result["new_measures"]] == ["zz_house"]
    assert result["new_measures"][0]["next_actions"]


def test_a_question_is_asked_once_per_column_not_once_per_row(conn):
    rows = [{"campaign_id": _campaign(conn, f"C{n}"), "structured": {"zz_house": n}}
            for n in range(4)]
    result = store.bulk_import_metrics(conn, rows, confirm=True)

    assert len(result["new_measures"]) == 1


def test_a_measure_a_workbook_makes_eligible_is_offered(conn):
    """§8.3's human step. A workbook is exactly where a measure crosses the gate, and burning
    the one-shot offer there means it can never be made."""
    rows = [{"campaign_id": _campaign(conn, f"C{m}", market=m),
             "structured": {"footfall_uplift_pct": 9.0}}
            for m in ("LATAM", "SEA", "EMEA")]

    result = store.bulk_import_metrics(conn, rows, confirm=True)

    assert [o["measure"] for o in result["newly_eligible"]] == ["footfall_uplift"]
    assert result["newly_eligible"][0]["if_confirmed"]


def test_a_skipped_cell_is_reported_rather_than_dropped(conn):
    """A skip nobody is told about is the silent acceptance this phase is written against,
    wearing the other face — and the preview promised this column would not be imported."""
    cid = _campaign(conn, "Peru launch")
    result = store.bulk_import_metrics(
        conn, [{"campaign_id": cid, "structured": {"reach": "lots", "roas": 3.1}}],
        confirm=True)

    assert result["skipped"][0]["key"] == "reach"
    assert result["skipped"][0]["row"] == 0
    assert metrics.values_for(conn, cid, "roas")[0]["value"] == 3.1


# ── the preview has to preview the failures too ─────────────────────────────

def test_the_preview_reports_rows_that_would_not_import(conn):
    """`errors: []` was asserted, not computed — so a preview saying "nothing has been stored,
    send with confirm=True" could be followed by three hundred of five hundred rows failing on
    an unmatched title. It had previewed the wrong half."""
    good = _campaign(conn, "Peru launch")
    rows = [{"campaign_id": good, "structured": {"roas": 1.0}},
            {"title": "No such campaign", "structured": {"roas": 2.0}},
            {"campaign_id": good, "metric_type": "Targett", "structured": {"roas": 3.0}},
            {"campaign_id": good, "structured": "not an object"}]

    preview = store.bulk_import_metrics(conn, rows)

    assert preview["would_import"] == 1
    assert [e["row"] for e in preview["errors"]] == [1, 2, 3]
    bad_type = next(e for e in preview["errors"] if e["row"] == 2)
    assert bad_type["field"] == "metric_type", "the retry, at preview time too"
    assert bad_type["suggestion"] == "target"
    assert "would not import" in preview["what_it_means"]


def test_the_preview_finds_them_without_writing_anything(conn):
    import hashlib

    good = _campaign(conn, "Peru launch")
    before = hashlib.sha256("\n".join(conn.iterdump()).encode()).hexdigest()

    store.bulk_import_metrics(conn, [{"campaign_id": good, "structured": {"zz_new": 1}},
                                     {"title": "Nope", "structured": {"roas": 2.0}}])

    assert hashlib.sha256("\n".join(conn.iterdump()).encode()).hexdigest() == before


def test_a_malformed_row_does_not_kill_the_whole_batch(conn):
    """`structured` that is not a dict raised AttributeError — not a ValueError, so over the
    protocol it arrived as "Error executing tool" with no row index, against a docstring
    promising a malformed row "never crashes or blocks the rest of the batch"."""
    good = _campaign(conn, "Peru launch")
    rows = [{"campaign_id": good, "structured": "12"},
            {"campaign_id": good, "structured": {"roas": 3.1}}]

    result = store.bulk_import_metrics(conn, rows, confirm=True)

    assert result["imported"] == 1
    assert result["errors"][0]["row"] == 0
    assert metrics.values_for(conn, good, "roas")[0]["value"] == 3.1


# ── a workbook's dimensions are not KPIs ────────────────────────────────────

def test_an_identifier_column_is_not_a_measure(conn):
    """A KPI workbook carries Month, Week, Store # and Campaign beside its measures. Every one
    is numeric, so each became a provisional measure that accrued sightings and could
    graduate — the "forty new keys" drift, produced by the tool built to stop it."""
    cid = _campaign(conn, "Peru launch")
    rows = [{"campaign_id": cid, "structured": {"Month": 3, "Store #": 41, "reach": 180000}}]

    preview = store.bulk_import_metrics(conn, rows)
    assert sorted(c["column"] for c in preview["columns"]["not_measures"]) == \
        ["Month", "Store #"]

    store.bulk_import_metrics(conn, rows, confirm=True)
    assert [m["canonical"] for m in metrics.unanswered(conn)] == []
    assert metrics.values_for(conn, cid, "reach")[0]["value"] == 180000.0


def test_two_columns_that_would_be_folded_together_are_asked_about(conn):
    """`dwell` and `queue_dwell` in one workbook were both classified `new` and then silently
    merged on write — the suffix rule matching the second against a provisional entry the
    first had created seconds earlier, with dict ordering deciding which."""
    cid = _campaign(conn, "Peru launch")
    rows = [{"campaign_id": cid, "structured": {"dwell": 1, "queue_dwell": 2}}]

    preview = store.bulk_import_metrics(conn, rows)
    assert sorted(c["column"] for c in preview["columns"]["looks_like"]) == \
        ["dwell", "queue_dwell"]

    store.bulk_import_metrics(conn, rows, confirm=True)
    assert {m["canonical"] for m in metrics.unanswered(conn)} == {"dwell", "queue_dwell"}


def test_an_established_alias_still_folds(conn):
    """The suffix rule is for aliases of measures the product knows. Turning it off entirely
    would lose `stated_combined_influencer_reach`, which §8.1 exists to fold."""
    assert metrics.canonical(conn, "stated_combined_influencer_reach") == "reach"


def test_the_batch_really_is_one_transaction(conn, tmp_path):
    """Without an explicit BEGIN the per-row SAVEPOINT was the OUTERMOST one, and releasing
    the outermost savepoint COMMITS — so the batch fsynced once per row after all, exactly as
    it did before `commit=False` existed, while the comment claimed otherwise. A claim about
    cost that is false is worse than no claim, because it stops anybody measuring."""
    import sqlite3

    import config

    cid = _campaign(conn, "Peru launch")
    watcher = sqlite3.connect(config.DB_PATH)
    seen = []

    rows = [{"campaign_id": cid, "structured": {"roas": float(n)}} for n in range(4)]
    original = core.add_metrics

    def watched(*a, **kw):
        seen.append(watcher.execute("SELECT COUNT(*) FROM metric_values").fetchone()[0])
        return original(*a, **kw)

    core.add_metrics = watched
    try:
        store.bulk_import_metrics(conn, rows, confirm=True)
    finally:
        core.add_metrics = original
        watcher.close()

    assert seen == [0, 0, 0, 0], f"another connection saw rows mid-batch: {seen}"
    assert len(metrics.values_for(conn, cid, "roas")) == 4
