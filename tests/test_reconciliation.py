"""
§9.9: make `reconcile_evaluation` do its job.

The review: *"The tool exists and has never run. This is its job: line up **predicted** against
**delivered** against **actual** against **context**, and produce one record — what we said
would happen, what we actually shipped, what it did, and what else was going on. That record is
the only thing in the entire product capable of telling you whether its own judgment is any
good. Everything else is the system talking about briefs; this is the system being held to
account."*

**Four columns, and three of them already exist.** §2.4 records what was said, §9.2/§9.3 record
what shipped, §6.5 records what it did, §9.6–9.8 record what else was going on. Every one was
built for its own reasons and none of them were ever put beside each other. This item is the
join, and almost all of its value is that the columns were computed independently — a
reconciliation assembled out of one subsystem's opinion of itself proves nothing.

**The server lines up; it does not mark its own homework.** Where a prediction is a number and
the outcome is a number, whether it landed is arithmetic and the server does it. Where a
finding is "the timeline is unrealistic", whether it held is a judgment, and the server says
`not_comparable` rather than guessing. `not_comparable` must never read as `held` — a
calibration figure built from unscorable rows silently scored as passes is worse than no figure
at all, because it would be the system awarding itself marks.

**A confounded outcome does not invalidate the reconciliation; it changes the lesson.** A
prediction that missed because a port was shut for nine days is a different lesson from one
that missed because the reasoning was wrong, and recording them identically is how a
calibration number stops meaning anything. §9.8's caveat therefore reaches this record, and the
note says what to do with it.
"""
import pytest

import context
import core
import store


def _campaign(conn, title="Mexico launch", **kw):
    kw.setdefault("market", "Mexico")
    kw.setdefault("detail", "A retail activation across Mexico City.")
    kw.setdefault("starts_on", "2026-09-01")
    kw.setdefault("ends_on", "2026-09-30")
    return core.ingest_campaign(conn, title=title, status="concluded", **kw)["campaign_id"]


def _judged(conn, campaign_id, **kw):
    kw.setdefault("verdict", "approve")
    kw.setdefault("summary", "Looks like Peru, which worked.")
    kw.setdefault("findings", [])
    kw.setdefault("predictions", {"predicted_ctr_range": "2.0-2.5",
                                  "predicted_roi_range": "3.0-4.0"})
    return core.save_evaluation(conn, subject_title="Mexico launch",
                                campaign_id=campaign_id, **kw)["evaluation_id"]


def _reconcile(conn, evaluation_id, **kw):
    return core.reconcile_evaluation(conn, evaluation_id=evaluation_id, **kw)


# ── four columns, side by side ─────────────────────────────────────────────

def test_it_lines_up_all_four(conn):
    """"Predicted against delivered against actual against context, and produce ONE record."
    Every one of the four already existed and none of them had ever been put beside each
    other."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid)

    out = _reconcile(conn, eid)

    for column in ("predicted", "delivered", "actual", "context"):
        assert column in out, column
    assert out["basis"] == "computed"


def test_predicted_is_what_was_said_at_the_time(conn):
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid, summary="The timeline is tight but the channel mix is right.")

    predicted = _reconcile(conn, eid)["predicted"]

    assert predicted["verdict"] == "approve"
    assert "channel mix" in predicted["summary"]
    assert predicted["predictions"]["predicted_roi_range"] == "3.0-4.0"


def test_delivered_is_what_actually_shipped(conn, tmp_path):
    """§9.2's half. "What we actually shipped" is not the brief — that is the whole of Phase
    9, and a reconciliation that reads the brief as the execution is measuring the wrong
    document."""
    cid = _campaign(conn)
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": _png(tmp_path, "brief.png", 3)})
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": _png(tmp_path, "shot.png", 91)},
                            phase="delivered", captured_on="2026-09-10")
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid)

    delivered = _reconcile(conn, eid)["delivered"]

    assert delivered["execution"]["status"] == "drifted"
    assert "commitments" in delivered


def test_a_campaign_nobody_checked_says_so_in_delivered(conn):
    """`never_checked` has to survive the trip, or a reconciliation reads "we shipped the
    brief" off a campaign nobody ever compared."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid)

    assert _reconcile(conn, eid)["delivered"]["execution"]["status"] == "never_checked"


def test_actual_is_the_measured_result(conn):
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4},
                     detail="ROAS came in at 3.4.", confirm=True)
    eid = _judged(conn, cid)

    actual = _reconcile(conn, eid)["actual"]

    assert actual["values"]["roas"] == 3.4
    assert "3.4" in actual["detail"]


def test_context_is_what_else_was_going_on(conn):
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    context.record(conn, starts_on="2026-09-10", ends_on="2026-09-12", scope="market",
                   scope_value="Mexico", kind="natural_disaster",
                   description="Magnitude 7.1 earthquake; retail shut 3 days.",
                   recorded_by="R. Vega")
    eid = _judged(conn, cid)

    out = _reconcile(conn, eid)

    assert out["context"]["confounded"] is True
    assert "earthquake" in str(out["context"]["confounded_by"]).lower()


# ── the server scores arithmetic and nothing else ──────────────────────────

def test_a_numeric_prediction_is_scored_by_the_server(conn):
    """Whether 3.4 falls inside 3.0-4.0 is arithmetic, and a model re-deriving it is the
    variance §7.1 exists to remove."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0"})

    scored = _reconcile(conn, eid)["scored"]

    roi = next(s for s in scored if s["predicted"] == "predicted_roi_range")
    assert roi["verdict"] == "held"
    assert roi["basis"] == "computed"
    assert roi["actual"] == 3.4


def test_a_prediction_that_missed_says_so(conn):
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 1.2}, confirm=True)
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0"})

    roi = next(s for s in _reconcile(conn, eid)["scored"]
               if s["predicted"] == "predicted_roi_range")

    assert roi["verdict"] == "missed"
    assert "1.2" in roi["what_it_means"]
    assert "3" in roi["what_it_means"] and "4" in roi["what_it_means"]


def test_a_prediction_with_no_matching_measurement_is_not_scored(conn):
    """The row that must never read as a pass. A calibration figure built out of unscorable
    rows silently counted as held would be the system awarding itself marks."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid, predictions={"predicted_ctr_range": "2.0-2.5"})

    ctr = next(s for s in _reconcile(conn, eid)["scored"]
               if s["predicted"] == "predicted_ctr_range")

    assert ctr["verdict"] == "not_comparable"
    assert "no ctr" in ctr["what_it_means"].lower()


def test_a_finding_is_never_scored_by_the_server(conn):
    """"The timeline is unrealistic" against a ROAS of 3.4 is a judgment, and the server has
    no way to make it. It lines the two up and says whose call it is."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid, verdict="revise", approve_if="The timeline is extended.",
                  findings=[{"severity": "should_fix", "kind": "missing_information",
                             "finding": "The six-week timeline is unrealistic",
                             "fix": "Extend it"}])

    out = _reconcile(conn, eid)

    assert out["predicted"]["findings"][0]["verdict"] == "yours_to_judge"
    assert not any(s.get("verdict") in ("held", "missed")
                   for s in out["scored"] if s.get("kind") == "finding")


def test_the_tally_never_counts_an_unscorable_row_as_a_pass(conn):
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0",
                                          "predicted_ctr_range": "2.0-2.5"})

    counts = _reconcile(conn, eid)["counts"]

    assert counts == {"held": 1, "missed": 0, "not_comparable": 1}


# ── a confounded outcome changes the lesson, not the record ────────────────

def test_a_confounded_outcome_is_still_reconciled(conn):
    """It does not invalidate anything — the prediction still missed or held. What changes is
    what the miss MEANS."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 1.2}, confirm=True)
    context.record(conn, starts_on="2026-09-10", ends_on="2026-09-19", scope="market",
                   scope_value="Mexico", kind="supply_chain",
                   description="The port was shut for nine days.", recorded_by="R. Vega")
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0"})

    out = _reconcile(conn, eid)

    assert out["scored"][0]["verdict"] == "missed"
    assert out["context"]["confounded"] is True
    said = out["note"].lower()
    assert "port" in said or "confounded" in said
    assert "different lesson" in said


def test_a_clean_outcome_says_nothing_about_confounding(conn):
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid)

    assert "confounded" not in _reconcile(conn, eid)["note"].lower()


# ── the record, and the loop closing ───────────────────────────────────────

def test_the_lesson_is_a_persons_and_the_tally_is_the_servers(conn):
    """The split this whole product is built on, at the one surface that grades the product."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 1.2}, confirm=True)
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0"})
    out = _reconcile(conn, eid)

    saved = core.save_reconciliation(
        conn, evaluation_id=eid,
        comparison="We were too optimistic about a new market with no precedent.")

    assert saved["basis"] == "stated"
    assert out["counts"]["missed"] == 1
    stored = store.reconciliation_for(conn, eid)
    assert stored["counts"]["missed"] == 1, "the server's tally is kept beside the lesson"


def test_a_reconciled_judgment_stops_being_offered(conn):
    """"A judgment already reconciled has been tested — asking again would be the
    always-present prompt that stops being read.\""""
    cid = _campaign(conn)
    eid = _judged(conn, cid)
    out = core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    assert any(a["tool"] == "reconcile_evaluation" for a in out["next_actions"])

    core.save_reconciliation(conn, evaluation_id=eid, comparison="The ROI call was right for a reason we can name: the channel mix.")

    again = core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.5}, confirm=True)
    assert not any(a["tool"] == "reconcile_evaluation"
                   for a in again.get("next_actions") or [])


def test_it_offers_the_lesson_rather_than_ending(conn):
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid)

    out = _reconcile(conn, eid)

    assert out["next_actions"][0]["tool"] == "save_reconciliation"
    assert out["next_actions"][0]["prefilled_args"]["evaluation_id"] == eid


def test_a_judgment_with_nothing_to_reconcile_says_so(conn):
    cid = _campaign(conn)
    eid = _judged(conn, cid)

    out = _reconcile(conn, eid)

    assert out["status"] == "nothing_to_check"
    assert "no measured result" in out["what_it_means"].lower()
    assert out["next_actions"][0]["tool"] == "add_metrics"


# ── what the library learns about itself ───────────────────────────────────

def test_the_library_can_say_how_often_it_was_right(conn):
    """"The only thing in the entire product capable of telling you whether its own judgment is
    any good." One record proves nothing; the point is the tally across them."""
    for n, (roas, low, high) in enumerate([(3.4, "3.0", "4.0"), (1.2, "3.0", "4.0"),
                                           (3.9, "3.0", "4.0")]):
        cid = _campaign(conn, f"Campaign {n}")
        core.add_metrics(conn, campaign_id=cid, structured={"roas": roas}, confirm=True)
        eid = _judged(conn, cid, predictions={"predicted_roi_range": f"{low}-{high}"})
        core.save_reconciliation(conn, evaluation_id=eid, comparison="The range was wide enough to be nearly unfalsifiable.")

    calibration = core.calibration(conn)

    assert calibration["counts"] == {"held": 2, "missed": 1, "not_comparable": 0}
    assert calibration["basis"] == "computed"
    assert "2 of 3" in calibration["what_it_means"]


def test_calibration_says_how_much_of_it_is_confounded(conn):
    """A calibration figure built mostly on confounded outcomes is a figure about the weather.
    It still counts — §9.8's rule — and a reader has to be able to see it."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 1.2}, confirm=True)
    context.record(conn, starts_on="2026-09-10", ends_on="2026-09-19", scope="market",
                   scope_value="Mexico", kind="supply_chain",
                   description="The port was shut for nine days.", recorded_by="R. Vega")
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0"})
    core.save_reconciliation(conn, evaluation_id=eid, comparison="The range was wide enough to be nearly unfalsifiable.")

    calibration = core.calibration(conn)

    assert calibration["confounded"] == 1
    assert "confounded" in calibration["what_it_means"].lower()


def test_an_empty_library_does_not_claim_a_score(conn):
    """Zero of zero is not a hundred per cent and it is not nought per cent."""
    calibration = core.calibration(conn)

    assert calibration["status"] == "nothing_to_check"
    assert "ever been reconciled" in calibration["what_it_means"].lower()
    assert "not a score of zero" in calibration["what_it_means"].lower()


def test_it_reaches_the_model_over_the_protocol(conn):
    import asyncio
    import json

    import mcp_server

    async def call(name, args):
        return json.loads((await mcp_server.mcp.call_tool(name, args)).content[0].text)

    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid)

    out = asyncio.run(call("reconcile_evaluation", {"evaluation_id": eid}))

    assert out["counts"]["held"] == 1
    assert out["delivered"]["execution"]["status"] == "never_checked"


def _png(tmp_path, name, seed):
    import random

    from PIL import Image

    rng = random.Random(seed)
    img = Image.new("RGB", (64, 64), (250, 250, 250))
    px = img.load()
    for _ in range(700):
        x, y = rng.randrange(64), rng.randrange(64)
        c = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        for dx in range(4):
            for dy in range(4):
                if x + dx < 64 and y + dy < 64:
                    px[x + dx, y + dy] = c
    path = tmp_path / name
    img.save(path)
    return str(path)


def test_a_hyphen_between_two_numbers_is_a_range_not_a_minus_sign(conn):
    """"3.0-4.0" parsed as 3.0 and MINUS 4.0, so the range became -4 to 3 and a ROAS of 3.4
    scored as a miss. Every prediction written with the commonest separator was wrong — in the
    direction nobody checks, because a product reporting its own judgments as worse than they
    were reads as modesty."""
    assert core._range_of("3.0-4.0") == (3.0, 4.0)
    assert core._range_of("2-3%") == (2.0, 3.0)
    assert core._range_of("3.0 to 4.0") == (3.0, 4.0)
    # A single number is NOT a range: read as a point it can only hold on exact equality,
    # which no real measurement satisfies. Refused rather than scored — see the parser test
    # below, which pins every case a later review found wrong.
    assert core._range_of("3.4") is None
    # And a genuine negative still reads as one.
    assert core._range_of("-5 to -2") == (-5.0, -2.0)
    assert core._range_of("a qualitative lift") is None


def test_which_measure_answers_a_prediction_comes_from_the_registry(conn):
    """A hand-written map here would be a second opinion about what a measure is called,
    sitting beside §8.1's registry and drifting the moment somebody adds a synonym. The first
    version mapped `predicted_conversion_range` to `conversion_rate`, which the registry does
    not recognise — an entry that could never match and nothing to say so."""
    assert core._measure_predicted(conn, "predicted_roi_range") == "roas"
    assert core._measure_predicted(conn, "predicted_roas_range") == "roas"
    assert core._measure_predicted(conn, "predicted_ctr_range") == "ctr"
    assert core._measure_predicted(conn, "predicted_click_through_rate_range") == "ctr"
    assert core._measure_predicted(conn, "predicted_unicorns_range") is None


def test_a_prediction_naming_no_known_measure_says_that(conn):
    """"Not scored" has to distinguish "we do not know what you predicted" from "we know and
    nobody measured it" — they are fixed by different people."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid, predictions={"predicted_unicorns_range": "3-4"})

    row = _reconcile(conn, eid)["scored"][0]

    assert row["verdict"] == "not_comparable"
    assert "no measure in this library is called" in row["what_it_means"].lower()


def test_a_prediction_in_a_synonym_still_scores(conn):
    """ROI and ROAS are one measure to this library — §8.1's registry says so, and a
    calibration figure that disagreed with the library's own naming about which number answers
    which prediction would be wrong in a way nobody could see."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"return_on_ad_spend": 3.4},
                     confirm=True)
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0"})

    assert _reconcile(conn, eid)["scored"][0]["verdict"] == "held"


# ── the routes in, which are the whole diagnosis ───────────────────────────

def test_a_version_check_does_not_close_the_results_loop(conn):
    """§6.3 offers `save_reconciliation(basis="superseding_version")` when v2 of a judged brief
    lands — a real check against a later BRIEF, and not an outcome. It wrote a row, and
    `unreconciled_evaluation_id` is "no reconciliation exists", so the results loop closed
    permanently before any results existed. D85 created `basis` for exactly this distinction
    and nothing read it."""
    cid = _campaign(conn)
    eid = _judged(conn, cid)

    core.save_reconciliation(conn, evaluation_id=eid, basis="superseding_version",
                             comparison="v2 shows the structure came back exactly as the judgment predicted it would.")
    out = core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)

    assert any(a["tool"] == "reconcile_evaluation" for a in out["next_actions"]), (
        "the results have never been checked against this judgment")


def test_a_results_check_does_close_it(conn):
    cid = _campaign(conn)
    eid = _judged(conn, cid)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)

    core.save_reconciliation(conn, evaluation_id=eid, comparison="The ROI call was right; the channel mix carried it, as predicted.")

    again = core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.5}, confirm=True)
    assert not any(a["tool"] == "reconcile_evaluation"
                   for a in again.get("next_actions") or [])


def test_a_judgment_checked_against_a_later_version_still_counts_its_predictions(conn):
    """The stored tally said "0 not checkable" about a judgment carrying a numeric prediction
    nobody could check — a false statement of fact, and the prediction then vanished from
    calibration entirely. §9.9's rule is that `not_comparable` never counts as a pass; it was
    satisfied here by the row disappearing instead."""
    cid = _campaign(conn)
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0"})

    saved = core.save_reconciliation(conn, evaluation_id=eid, basis="superseding_version",
                                     comparison="v2 brought the structure back, which is what the judgment said would have to happen.")

    assert saved["counts"] == {"held": 0, "missed": 0, "not_comparable": 1}


def test_a_bulk_import_offers_the_loop_it_just_made_closable(conn):
    """§8.8 is how a customer actually loads a KPI workbook, and the loop harvested every
    other one-shot side effect and dropped `next_actions` — which carries this offer and
    §9.4's. D116's shape, on the surface D116 was written about."""
    cid = _campaign(conn)
    eid = _judged(conn, cid)

    out = store.bulk_import_metrics(
        conn, [{"campaign_id": cid, "structured": {"roas": 3.4}}], confirm=True)

    assert any(a["tool"] == "reconcile_evaluation"
               for a in out.get("next_actions") or []), out.get("next_actions")
    assert store.unreconciled_evaluation_id(conn, cid) == eid


def test_unreconciled_judgments_are_a_gap(conn):
    """The item whose entire diagnosis is "it needs somebody to decide to go back and nobody
    does" gave itself no gap. §9.6 added `no_window` precisely because without one the absence
    is invisible on every reporting surface."""
    for n in range(3):
        cid = _campaign(conn, f"Campaign {n}")
        core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
        _judged(conn, cid)

    found = [g for g in core.gaps(conn)["gaps"] if g["code"] == "judgments_never_reconciled"]

    assert found, [g["code"] for g in core.gaps(conn)["gaps"]]
    assert found[0]["counts"]["judgments"] == 3
    assert found[0]["next_actions"][0]["tool"] == "reconcile_evaluation"


def test_the_gap_closes_as_they_are_checked(conn):
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid)
    assert [g for g in core.gaps(conn)["gaps"] if g["code"] == "judgments_never_reconciled"]

    core.save_reconciliation(conn, evaluation_id=eid, comparison="The ROI call was right; the channel mix carried it, as predicted.")

    assert not [g for g in core.gaps(conn)["gaps"]
                if g["code"] == "judgments_never_reconciled"]


# ── a score whose denominator is whoever bothered ──────────────────────────

def test_calibration_says_how_much_of_the_library_it_has_seen(conn):
    """A bare ratio over a self-selected denominator is the most dangerous version of this
    number: fourteen unchecked judgments and one reconciled reads as a hundred per cent."""
    for n in range(4):
        cid = _campaign(conn, f"Campaign {n}")
        core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
        eid = _judged(conn, cid)
        if n == 0:
            core.save_reconciliation(conn, evaluation_id=eid, comparison="We were right, and on one campaign that is luck rather than skill.")

    cal = core.calibration(conn)

    assert cal["reconciled"] == 1
    assert cal["reconcilable"] == 4
    assert "1 of 4" in cal["what_it_means"]


def test_calibration_will_not_call_one_judgment_a_record(conn):
    """`_MIN_FOR_A_PATTERN` exists four hundred lines away for this reasoning: one record
    missing a budget is a fact about that record, not a coverage gap."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0"})
    core.save_reconciliation(conn, evaluation_id=eid, comparison="We were right, and on one campaign that is luck rather than skill.")

    cal = core.calibration(conn)

    assert cal["counts"]["held"] == 1
    assert "too few" in cal["what_it_means"].lower()


def test_calibration_keeps_version_checks_out_of_the_outcome_score(conn):
    """store.py's own comment on the `basis` column: "anything computing calibration would
    read both as outcome data"."""
    cid = _campaign(conn)
    eid = _judged(conn, cid)
    core.save_reconciliation(conn, evaluation_id=eid, basis="superseding_version",
                             comparison="v2 brought the structure back, which is what the judgment said would have to happen.")

    cal = core.calibration(conn)

    assert cal["reconciled"] == 0, "a brief-versus-brief check is not a measured outcome"
    assert cal["against_a_later_version"] == 1


# ── "produce ONE record" ───────────────────────────────────────────────────

def test_the_assembled_record_is_kept(conn):
    """The four columns were assembled and thrown away: the stored row held a lesson, three
    integers and one bit. Six months later a reader gets that, and re-running gives a
    different answer with nothing saying so."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid)
    core.save_reconciliation(conn, evaluation_id=eid, comparison="The ROI call was right; the channel mix carried it, as predicted.")

    kept = core.get_reconciliation(conn, evaluation_id=eid)

    for column in ("predicted", "delivered", "actual", "context"):
        assert column in kept["record"], column
    assert kept["record"]["as_of"], "and when it was true"
    assert kept["comparison"].startswith("The ROI call was right")


def test_the_kept_record_does_not_move_under_the_reader(conn):
    """Three of the four columns are recomputed live over mutable state — the drift figure is
    rewritten whenever the answer changes, context links are recomputed on every read and an
    event can be withdrawn. §9.5 solved this for itself with `execution_at_save` and wrote
    down why; §9.9 did not apply the rule to itself."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid)
    core.save_reconciliation(conn, evaluation_id=eid, comparison="The ROI call was right; the channel mix carried it, as predicted.")
    assert core.get_reconciliation(conn, evaluation_id=eid)[
        "record"]["context"]["confounded"] is False

    context.record(conn, starts_on="2026-09-10", ends_on="2026-09-19", scope="market",
                   scope_value="Mexico", kind="supply_chain",
                   description="The port was shut for nine days.", recorded_by="R. Vega")

    kept = core.get_reconciliation(conn, evaluation_id=eid)
    assert kept["record"]["context"]["confounded"] is False, "what it rested on, unchanged"
    assert kept["changed_since"]["context"] is True, "and it says the live answer has moved"


def test_a_judgment_never_reconciled_has_no_record(conn):
    cid = _campaign(conn)
    eid = _judged(conn, cid)

    out = core.get_reconciliation(conn, evaluation_id=eid)

    assert out["status"] == "nothing_to_check"
    assert out["next_actions"][0]["tool"] == "reconcile_evaluation"


# ── what §9.5 and §9.8 stamped FOR this item ───────────────────────────────

def test_it_reads_what_was_known_when_the_verdict_was_written(conn):
    """`_confounded_at_save` and `_execution_at_save` both name §9.9 in their docstrings —
    "it cannot do that against a caveat nobody stored". Both land in the evaluation's evidence
    and §9.9 read neither. The then-versus-now delta is the most interesting thing here."""
    cited = _campaign(conn, "Peru launch")
    core.add_metrics(conn, campaign_id=cited, structured={"roas": 3.9}, confirm=True)
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid, cited_ids=[cited])

    # The earthquake is recorded after the verdict, which is the ordinary order.
    context.record(conn, starts_on="2026-09-10", ends_on="2026-09-19", scope="market",
                   scope_value="Mexico", kind="supply_chain",
                   description="The port was shut for nine days.", recorded_by="R. Vega")

    known = _reconcile(conn, eid)["context"]["known_when_predicted"]
    assert known["confounded_evidence"] == [], "nothing was confounded when this was written"
    assert "execution" in known


def test_the_delta_is_said_when_the_world_moved_afterwards(conn):
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 1.2}, confirm=True)
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0"})
    context.record(conn, starts_on="2026-09-10", ends_on="2026-09-19", scope="market",
                   scope_value="Mexico", kind="supply_chain",
                   description="The port was shut for nine days.", recorded_by="R. Vega")

    out = _reconcile(conn, eid)

    assert out["context"]["confounded"] is True
    assert out["context"]["learned_since"] is True
    assert "was not known" in out["note"].lower()


# ── a prediction about a year, scored against one month ────────────────────

def test_a_measure_recorded_for_many_periods_is_not_scored_against_one(conn):
    """"Latest wins" was insertion order, so a campaign-level prediction was scored against
    whichever month the spreadsheet happened to end with — and calibration was built on it."""
    cid = _campaign(conn, starts_on="2026-01-01", ends_on="2026-12-31")
    for month, roas in ((1, 4.4), (2, 3.5), (3, 2.2)):
        core.add_metrics(conn, campaign_id=cid, confirm=True,
                         structured={"month": f"2026-{month:02d}", "roas": roas})
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0"})

    row = _reconcile(conn, eid)["scored"][0]

    assert row["verdict"] == "not_comparable"
    assert "3 period" in row["what_it_means"]
    assert "yours to say" in row["what_it_means"].lower()


def test_one_period_is_still_scored(conn):
    cid = _campaign(conn, starts_on="2026-01-01", ends_on="2026-12-31")
    core.add_metrics(conn, campaign_id=cid, confirm=True,
                     structured={"month": "2026-01", "roas": 3.4})
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0"})

    assert _reconcile(conn, eid)["scored"][0]["verdict"] == "held"


# ── the lesson has to say something ────────────────────────────────────────

def test_a_content_free_lesson_is_refused(conn):
    """The product's only evidence about its own judgment, and its own test fixtures wrote
    "Recorded." into it. §9.8 one item earlier refuses a `stated_by` that reads as the
    product; adjacent items, opposite rigour."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid)

    for empty in ("It held.", "ok", "Recorded.", ".", "👍"):
        with pytest.raises(ValueError) as e:
            core.save_reconciliation(conn, evaluation_id=eid, comparison=empty)
        assert "lesson" in str(e.value).lower() or "what it meant" in str(e.value).lower()


def test_restating_the_tally_is_refused(conn):
    """The docstring already says "do not restate it: that half is arithmetic and already on
    the record". Nothing enforced it."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid)

    with pytest.raises(ValueError) as e:
        core.save_reconciliation(conn, evaluation_id=eid,
                                 comparison="1 held, 0 missed, 1 not comparable. 1 held.")
    assert "arithmetic" in str(e.value).lower()


def test_a_real_lesson_is_accepted(conn):
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid)

    saved = core.save_reconciliation(
        conn, evaluation_id=eid,
        comparison="We were right about ROI and never measured CTR, so half of this judgment "
                   "is still untested.")

    assert saved["status"] == "saved"


# ── the boundaries, which are where a grade flips ──────────────────────────

def test_an_actual_exactly_on_the_low_bound_held(conn):
    """No test anywhere placed an actual on a boundary, so either edge could flip silently —
    on the one surface whose whole job is grading the product."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.0}, confirm=True)
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0"})

    assert _reconcile(conn, eid)["scored"][0]["verdict"] == "held"


def test_an_actual_exactly_on_the_high_bound_held(conn):
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 4.0}, confirm=True)
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0"})

    assert _reconcile(conn, eid)["scored"][0]["verdict"] == "held"


def test_a_hair_outside_either_bound_missed(conn):
    for roas in (2.99, 4.01):
        cid = _campaign(conn, f"Campaign {roas}")
        core.add_metrics(conn, campaign_id=cid, structured={"roas": roas}, confirm=True)
        eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0"})
        assert _reconcile(conn, eid)["scored"][0]["verdict"] == "missed", roas


# ── the sentence, and the denominator inside it ────────────────────────────

def test_the_record_sentence_counts_the_misses_in_its_denominator(conn):
    """A record that held one and missed three read "1 of 1 checkable prediction(s) landed" —
    the flattering direction, and no test asserted the denominator."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid,
                     structured={"roas": 1.0, "ctr": 9.0}, confirm=True)
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0",
                                          "predicted_ctr_range": "2.0-2.5"})

    said = _reconcile(conn, eid)["what_it_means"]

    assert "0 of 2" in said, said


def test_the_record_sentence_keeps_the_confounded_caveat(conn):
    """The tests checked `note` and not `what_it_means`, so the "read the misses twice" line
    could be removed from the record's own sentence with nothing noticing."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 1.2}, confirm=True)
    context.record(conn, starts_on="2026-09-10", ends_on="2026-09-19", scope="market",
                   scope_value="Mexico", kind="supply_chain",
                   description="The port was shut for nine days.", recorded_by="R. Vega")
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0"})

    said = _reconcile(conn, eid)["what_it_means"].lower()

    assert "read the misses twice" in said
    assert "different lesson" in said


def test_the_record_says_how_many_rows_the_number_came_from(conn):
    """`measured_rows` is the only signal that a single figure collapsed several rows, and
    nothing pinned it."""
    cid = _campaign(conn)
    for roas in (3.1, 3.4):
        core.add_metrics(conn, campaign_id=cid, structured={"roas": roas}, confirm=True)

    assert _reconcile(conn, _judged(conn, cid))["actual"]["measured_rows"] == 2


def test_what_the_calendar_said_at_the_time_is_carried(conn):
    """§9.7 stamps its check onto the verdict so §9.9 can show the then-versus-now delta — the
    docstring calls it the interesting record, and it could vanish untested."""
    cid = _campaign(conn, starts_on="2026-06-15", ends_on="2026-07-10")
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid)

    known = _reconcile(conn, eid)["context"]["known_when_predicted"]

    assert known["calendar_clash"] == "present"
    assert "World Cup" in known["what_it_means"]


def test_the_delivered_half_is_never_silently_empty(conn):
    """`"commitments" in delivered` is satisfied by a `None`, so the whole §9.3 half could be
    nulled with no test noticing."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)

    delivered = _reconcile(conn, _judged(conn, cid))["delivered"]

    assert delivered["commitments"]["what_it_means"]
    assert delivered["execution"]["status"]
    assert delivered["what_it_means"]


def test_reconciling_a_judgment_that_does_not_exist_is_refused_readably(conn):
    """A bogus id reached the insert and came back as a database IntegrityError at the MCP
    surface — the FK held, so nothing was corrupted, but the message is for a marketer."""
    with pytest.raises(ValueError) as e:
        core.save_reconciliation(conn, evaluation_id="eval_nope",
                                 comparison="We were too optimistic about a new market.")
    assert "eval_nope" in str(e.value)


def test_a_miss_says_which_way_it_missed(conn):
    """The sentence a person reads to decide whether the library was optimistic or
    pessimistic — and it could report a miss below the range as "above" with nothing
    noticing."""
    low = _campaign(conn, "Low")
    core.add_metrics(conn, campaign_id=low, structured={"roas": 1.2}, confirm=True)
    high = _campaign(conn, "High")
    core.add_metrics(conn, campaign_id=high, structured={"roas": 9.5}, confirm=True)

    said_low = _reconcile(conn, _judged(conn, low, predictions={
        "predicted_roi_range": "3.0-4.0"}))["scored"][0]["what_it_means"]
    said_high = _reconcile(conn, _judged(conn, high, predictions={
        "predicted_roi_range": "3.0-4.0"}))["scored"][0]["what_it_means"]

    assert "below by 1.8" in said_low
    assert "above by 5.5" in said_high


def test_a_prediction_key_in_a_different_case_is_the_same_prediction(conn):
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid, predictions={"  PREDICTED_ROI_RANGE  ": "3.0-4.0"})

    assert _reconcile(conn, eid)["scored"][0]["verdict"] == "held"


def test_a_judgment_with_nothing_measured_is_labelled_computed(conn):
    """A house-rule label with no test behind it. `nothing_to_check` is a fact the server
    established about its own records, and mislabelling it is the one thing §2.4 exists to
    stop."""
    out = _reconcile(conn, _judged(conn, _campaign(conn)))

    assert out["status"] == "nothing_to_check"
    assert out["basis"] == "computed"


def test_a_lift_is_never_scored_against_a_level(conn):
    """The registry's prefix rule routes `ctr_lift` into the CTR family, which is right for
    STORING it and wrong for deciding which measured number answers a prediction. A predicted
    10–20% CTR lift scored against a measured CTR of 2.3 reads "missed, below by 7.7" — two
    different quantities, wearing the server's authority."""
    assert core._measure_predicted(conn, "predicted_ctr_lift_range") is None
    assert core._measure_predicted(conn, "predicted_roas_uplift_range") is None
    assert core._measure_predicted(conn, "predicted_ctr_range") == "ctr"
    assert core._measure_predicted(conn, "predicted_click_through_rate_range") == "ctr"

    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"ctr": 2.3}, confirm=True)
    eid = _judged(conn, cid, predictions={"predicted_ctr_lift_range": "10-20"})

    assert _reconcile(conn, eid)["scored"][0]["verdict"] == "not_comparable"


def test_the_same_judgment_cannot_be_counted_twice(conn):
    """Three goes at one judgment read as three judgments landing — the product inflating its
    own record by being asked twice."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0"})
    core.save_reconciliation(conn, evaluation_id=eid,
                             comparison="The channel mix carried it, exactly as predicted.")

    with pytest.raises(ValueError) as e:
        core.save_reconciliation(conn, evaluation_id=eid,
                                 comparison="Saying the same thing a second time over.")

    assert "already been reconciled" in str(e.value)
    assert core.calibration(conn)["counts"]["held"] == 1


def test_a_version_check_does_not_block_the_results_check(conn):
    """They are different evidence about the same judgment — D85's whole point."""
    cid = _campaign(conn)
    eid = _judged(conn, cid)
    core.save_reconciliation(conn, evaluation_id=eid, basis="superseding_version",
                             comparison="v2 brought the structure back as the judgment said.")
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)

    saved = core.save_reconciliation(
        conn, evaluation_id=eid,
        comparison="And the numbers bore it out: the channel mix carried the ROI.")

    assert saved["status"] == "saved"


# ── the parser that decides whether the product passes ─────────────────────

def test_the_range_parser_refuses_everything_it_cannot_read_safely(conn):
    """Each of these was measured wrong, and every one of them decides whether the library's
    own judgment gets a pass. The rule is that every ambiguity resolves to `not_comparable`: a
    prediction nobody scored is honest, and one scored against a number parsed out of a
    quarter label is the product grading itself on noise."""
    # Read correctly.
    assert core._range_of("3.0-4.0") == (3.0, 4.0)
    assert core._range_of("3.0 to 4.0") == (3.0, 4.0)
    assert core._range_of("3.0–4.0") == (3.0, 4.0), "en-dash"
    assert core._range_of("2%-3%") == (2.0, 3.0), "a unit before the hyphen is not a minus"
    assert core._range_of("3x-4x") == (3.0, 4.0)
    assert core._range_of("300-400bps") == (300.0, 400.0)
    assert core._range_of("5,000-6,000") == (5000.0, 6000.0), "thousands separators"
    assert core._range_of("-5 to -2") == (-5.0, -2.0), "and a real negative still reads as one"
    assert core._range_of(".5-.8") == (0.5, 0.8)

    # Refused, each for its own reason.
    assert core._range_of("Q3 2026: 3.0-4.0") is None, "the year became the upper bound"
    assert core._range_of("3,5-4,0") is None, "a European decimal, unreadable either way"
    assert core._range_of("up to 4") is None, "an open bound read as the point 4"
    assert core._range_of("at least 3") is None
    assert core._range_of(">3") is None
    assert core._range_of("3.4") is None, "a point estimate holds only on exact equality"
    assert core._range_of("1e3") is None
    assert core._range_of("3.0-4.0 in Q1, 5.0-6.0 in Q2") is None, "two predictions"
    assert core._range_of("a qualitative lift") is None


def test_an_unreadable_prediction_says_which_kind_of_unreadable(conn):
    """"Not scored" has to be specific enough to fix: an open bound, an ambiguous comma and a
    point estimate are three different mistakes with three different corrections."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)

    for stated, expected in (("up to 4", "open-ended"),
                             ("3,5-4,0", "comma"),
                             ("3.4", "single figure")):
        eid = _judged(conn, cid, predictions={"predicted_roi_range": stated})
        row = _reconcile(conn, eid)["scored"][0]
        assert row["verdict"] == "not_comparable", stated
        assert expected in row["what_it_means"], (stated, row["what_it_means"])


# ── a figure that does not quietly go stale ────────────────────────────────

def test_calibration_reflects_a_correction_to_the_numbers(conn):
    """The stored tally is a save-time snapshot, and `calibration` read it — so a corrected
    figure that turns a held prediction into a missed one left the product still reporting the
    pass. `_reconciliation_context`'s own docstring says why a save-time figure is the wrong
    moment; the tally was doing exactly that."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, confirm=True,
                     structured={"month": "2026-09", "roas": 3.4})
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0"})
    core.save_reconciliation(conn, evaluation_id=eid,
                             comparison="The channel mix carried the ROI, as predicted.")
    assert core.calibration(conn)["counts"] == {"held": 1, "missed": 0, "not_comparable": 0}

    # A correction restates the SAME period. Two undated rows would be genuinely ambiguous —
    # see the test below.
    core.add_metrics(conn, campaign_id=cid, confirm=True,
                     detail="Corrected: the earlier figure double-counted.",
                     structured={"month": "2026-09", "roas": 1.0})

    cal = core.calibration(conn)
    assert cal["counts"] == {"held": 0, "missed": 1, "not_comparable": 0}
    assert cal["moved_since_recorded"] == 1
    assert "changed since" in cal["what_it_means"].lower()


def test_calibration_reflects_a_confounder_recorded_afterwards(conn):
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 1.2}, confirm=True)
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0"})
    core.save_reconciliation(conn, evaluation_id=eid,
                             comparison="We were too optimistic about an untested market.")
    assert core.calibration(conn)["confounded"] == 0

    context.record(conn, starts_on="2026-09-10", ends_on="2026-09-19", scope="market",
                   scope_value="Mexico", kind="supply_chain",
                   description="The port was shut for nine days.", recorded_by="R. Vega")

    assert core.calibration(conn)["confounded"] == 1


def test_the_stored_record_still_says_what_the_lesson_rested_on(conn):
    """Both halves. The figure is current; the record is what was true when somebody wrote the
    lesson, and a reader needs to be able to tell them apart."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, confirm=True,
                     structured={"month": "2026-09", "roas": 3.4})
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0"})
    core.save_reconciliation(conn, evaluation_id=eid,
                             comparison="The channel mix carried the ROI, as predicted.")

    core.add_metrics(conn, campaign_id=cid, confirm=True, detail="Corrected.",
                     structured={"month": "2026-09", "roas": 1.0})

    kept = core.get_reconciliation(conn, evaluation_id=eid)
    assert kept["counts"] == {"held": 1, "missed": 0, "not_comparable": 0}, "as recorded"
    assert kept["counts_now"] == {"held": 0, "missed": 1, "not_comparable": 0}, "as it stands"
    assert kept["changed_since"]["counts"] is True


def test_two_undated_figures_for_one_measure_are_not_scored(conn):
    """Two ROAS rows with no period could be two months or one correction, and the library
    cannot tell. Picking the later one is arithmetic on a guess — and it was, silently, by
    insertion order."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 1.0}, confirm=True)
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0"})

    row = _reconcile(conn, eid)["scored"][0]

    assert row["verdict"] == "not_comparable"
    assert "2 periods" in row["what_it_means"]


# ── the remaining house-rule breaks ────────────────────────────────────────

def test_caller_supplied_actual_is_labelled_stated(conn):
    """`actual.basis` was always `computed` — but `detail` is the caller's own sentence when
    they pass one, and the numbers beside it come from the record. A stated sentence and a
    computed number contradicting each other, both labelled computed."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 1.0}, confirm=True)
    eid = _judged(conn, cid)

    out = _reconcile(conn, eid, actual="Corrected: ROAS was 3.5.")

    assert out["actual"]["detail_basis"] == "stated"
    assert out["actual"]["values_basis"] == "computed"
    assert "do not come from what you typed" in out["actual"]["what_it_means"]


def test_the_record_says_where_its_numbers_came_from(conn):
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)

    out = _reconcile(conn, _judged(conn, cid))

    assert out["actual"]["detail_basis"] == "computed"


def test_a_broken_commitment_check_is_not_reported_as_nothing_to_check(conn, monkeypatch):
    """"Could not check" rendered as "nothing to check" is the collapse every other surface in
    Phase 9 refuses."""
    import commitments

    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid)
    monkeypatch.setattr(commitments, "summary_for",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db locked")))

    delivered = _reconcile(conn, eid)["delivered"]

    assert delivered["commitments"]["status"] == "could_not_check"
    assert "could not" in delivered["commitments"]["what_it_means"].lower()


def test_predictions_in_the_wrong_shape_are_refused_not_crashed(conn):
    """`save_evaluation` accepts a list, and `_score_predictions` called `.items()` on it."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    eid = _judged(conn, cid, predictions=["3.0-4.0"])

    out = _reconcile(conn, eid)

    assert out["scored"] == []
    assert "not a set of named predictions" in out["what_it_means"].lower()


def test_a_number_recorded_as_text_is_still_a_number(conn):
    """A workbook cell arriving as "3.4" said "No ROAS was measured on this campaign", which
    is false — it was measured, as a string."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": "3.4"}, confirm=True)
    eid = _judged(conn, cid, predictions={"predicted_roi_range": "3.0-4.0"})

    assert _reconcile(conn, eid)["scored"][0]["verdict"] == "held"


def test_a_dimension_column_is_not_a_measurement(conn):
    """`month: 12` leaked into the measured values as though somebody had measured twelve of
    something — §8.2 already refuses these on the way in."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, confirm=True,
                     structured={"month": "2026-12", "store_number": 14, "roas": 3.4})

    values = _reconcile(conn, _judged(conn, cid))["actual"]["values"]

    assert "roas" in values
    assert not {k for k in values if "store" in k or "month" in k}, values


# ── supersession: D120, D127 and D40, all owed to this item ────────────────

def test_a_judgment_is_reconciled_against_the_record_that_actually_ran(conn):
    """D120/D127 were re-pointed here because "predicted vs delivered vs actual vs superseded"
    is this item's subject. A judgment on v1 whose results landed on v2 reported
    `nothing_to_check` and offered to record results against v1 — a brief that never ran, and
    the exact failure `actions.after_upload` describes in writing."""
    v1 = _campaign(conn, "Mexico v1")
    eid = _judged(conn, v1, predictions={"predicted_roi_range": "3.0-4.0"})
    v2 = core.ingest_campaign(conn, title="Mexico v2", market="Mexico", status="concluded",
                              detail="The revised activation.", supersedes=v1,
                              starts_on="2026-09-01", ends_on="2026-09-30")["campaign_id"]
    core.add_metrics(conn, campaign_id=v2, structured={"roas": 3.4}, confirm=True)

    out = _reconcile(conn, eid)

    assert out["status"] == "checked"
    assert out["scored"][0]["verdict"] == "held"
    assert out["actual"]["from_campaign_id"] == v2
    assert "superseded" in out["what_it_means"].lower()


def test_results_on_a_later_version_offer_the_earlier_judgment(conn):
    v1 = _campaign(conn, "Mexico v1")
    eid = _judged(conn, v1)
    v2 = core.ingest_campaign(conn, title="Mexico v2", market="Mexico", status="concluded",
                              detail="The revised activation.", supersedes=v1,
                              starts_on="2026-09-01", ends_on="2026-09-30")["campaign_id"]

    out = core.add_metrics(conn, campaign_id=v2, structured={"roas": 3.4}, confirm=True)

    offers = [a for a in out.get("next_actions") or []
              if a["tool"] == "reconcile_evaluation"]
    assert offers, out.get("next_actions")
    assert offers[0]["prefilled_args"]["evaluation_id"] == eid


def test_a_judgment_across_a_re_brief_is_a_gap_too(conn):
    v1 = _campaign(conn, "Mexico v1")
    _judged(conn, v1)
    v2 = core.ingest_campaign(conn, title="Mexico v2", market="Mexico", status="concluded",
                              detail="The revised activation.", supersedes=v1,
                              starts_on="2026-09-01", ends_on="2026-09-30")["campaign_id"]
    core.add_metrics(conn, campaign_id=v2, structured={"roas": 3.4}, confirm=True)

    assert [g for g in core.gaps(conn)["gaps"]
            if g["code"] == "judgments_never_reconciled"]


# ── D40: a judgment made before the record existed ─────────────────────────

def test_a_judgment_on_an_unstored_proposal_can_be_linked_afterwards(conn):
    """D40: the store-then-reconcile sequence could not complete. A judgment on a pitch that
    is not yet a record carries a null `campaign_id`, §5.2 offers "add this to the library",
    and nothing could then attach the judgment to the record — so the loop this whole item
    exists to close was unreachable from the commonest starting point."""
    saved = core.save_evaluation(conn, subject_title="Mexico launch", verdict="approve",
                                 summary="A new pitch, not in the library yet.", findings=[],
                                 predictions={"predicted_roi_range": "3.0-4.0"})
    eid = saved["evaluation_id"]
    cid = _campaign(conn)
    assert _reconcile(conn, eid)["status"] == "nothing_to_check"

    core.link_evaluation(conn, evaluation_id=eid, campaign_id=cid,
                         linked_by="R. Vega")
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)

    out = _reconcile(conn, eid)
    assert out["status"] == "checked"
    assert out["scored"][0]["verdict"] == "held"


def test_linking_a_judgment_that_is_already_linked_is_refused(conn):
    """Re-pointing a judgment at a different campaign would silently move what it was about,
    and every citation of it with it."""
    cid = _campaign(conn)
    eid = _judged(conn, cid)
    other = _campaign(conn, "Somewhere else")

    with pytest.raises(ValueError) as e:
        core.link_evaluation(conn, evaluation_id=eid, campaign_id=other, linked_by="R. Vega")
    assert "already" in str(e.value).lower()


def test_linking_needs_a_person_and_a_real_record(conn):
    saved = core.save_evaluation(conn, subject_title="Mexico launch", verdict="approve",
                                 summary="A new pitch.", findings=[])

    with pytest.raises(ValueError):
        core.link_evaluation(conn, evaluation_id=saved["evaluation_id"],
                             campaign_id="camp_nope", linked_by="R. Vega")
    with pytest.raises(ValueError) as e:
        core.link_evaluation(conn, evaluation_id=saved["evaluation_id"],
                             campaign_id=_campaign(conn), linked_by="")
    assert "person" in str(e.value).lower()


def test_the_upload_that_follows_a_judgment_offers_the_link(conn):
    """§5.2's "add this to the library" is the moment — the judgment is on screen and the
    record has just been created."""
    saved = core.save_evaluation(conn, subject_title="Mexico launch", verdict="approve",
                                 summary="A new pitch, not in the library yet.", findings=[])

    out = core.ingest_campaign(conn, title="Mexico launch", market="Mexico",
                               status="concluded", detail="A retail activation.")

    offers = [a for a in out.get("next_actions") or [] if a["tool"] == "link_evaluation"]
    assert offers, out.get("next_actions")
    assert offers[0]["prefilled_args"]["evaluation_id"] == saved["evaluation_id"]
