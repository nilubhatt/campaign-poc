"""
§9.5: attach drift to the outcome, so performance is never read naked.

The review: *"Store `execution_drift` on the outcome record: the score, the counts, and the
classified items. Then `performed_well` at low drift and `performed_well` at high drift stop
looking identical — the first is evidence the brief was good, the second is evidence something
was good and the brief may not have been it. Retrieval should carry this forward: a high-drift
campaign is weaker precedent for judging a brief, and `prepare_evaluation` should say so rather
than ranking it purely on similarity."*

**"Evidence something was good and the brief may not have been it"** is the whole claim, and it
is about what a result can be USED for rather than about whether it is true. A campaign that
performed well after drifting a long way from its brief really did perform well. What it cannot
do is tell you whether the brief in front of you is a good one — and today the library cites it
as though it could.

**Weaker precedent, not excluded precedent.** Dropping high-drift campaigns would throw away the
evidence the review most wants kept: the UAE claw machine departed from precedent and beat it.
The caveat travels with the citation instead, which is D19's shape — the same rule §9.8 will
apply to confounded outcomes.

**And the caveat is only as good as what is behind it.** A campaign nobody has checked has no
drift figure, and that is a third state — not low drift. Reading "unchecked" as "faithful" is
the assumption that makes this whole phase necessary.
"""
import pytest

import core
import drift
import store


def _campaign(conn, title, market="LATAM"):
    return core.ingest_campaign(conn, title=title, market=market, status="concluded",
                                detail="Plan:\n- A claw machine\n- A photo booth\n")[
        "campaign_id"]


def _claw(conn, campaign_id):
    """The commitment id for the claw machine — a subject somebody can read back, which is the
    point of keying on one."""
    import commitments
    return next(c["id"] for c in commitments.for_campaign(conn, campaign_id)
                if "claw" in c["text"].lower())


def _booth(conn, campaign_id):
    import commitments
    return next(c["id"] for c in commitments.for_campaign(conn, campaign_id)
                if "booth" in c["text"].lower())


def _png(tmp_path, name, seed=1):
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


def _ran_as_briefed(conn, cid, tmp_path, tag=""):
    same = _png(tmp_path, f"same{tag}.png", seed=11)
    core.ingest_image_asset(conn, campaign_id=cid, asset_ref={"path": same})
    core.ingest_image_asset(conn, campaign_id=cid, asset_ref={"path": same},
                            phase="delivered", captured_on="2026-03-14")


def _ran_differently(conn, cid, tmp_path, tag=""):
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": _png(tmp_path, f"b{tag}.png", seed=3)})
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": _png(tmp_path, f"d{tag}.png", seed=91)},
                            phase="delivered", captured_on="2026-03-14")


# ── the outcome record carries it ───────────────────────────────────────────

def test_recording_results_stores_the_drift_alongside_them(conn, tmp_path):
    """"Store `execution_drift` on the outcome record: the score, the counts, and the
    classified items.\""""
    cid = _campaign(conn, "Colombia")
    _ran_differently(conn, cid, tmp_path)

    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.1}, confirm=True)

    stored = store.execution_drift_for(conn, cid)
    assert stored["score"] is not None
    assert stored["counts"]["never_appeared"] == 1
    assert stored["basis"] == "computed"


def test_it_carries_the_classifications_too(conn, tmp_path):
    cid = _campaign(conn, "UAE", market="EMEA")
    _ran_differently(conn, cid, tmp_path)
    drift.classify(conn, campaign_id=cid, subject=_claw(conn, cid),
                   classification="improvement", why="It drew the queue.",
                   classified_by="R. Vega")

    core.add_metrics(conn, campaign_id=cid, structured={"roas": 4.2}, confirm=True)

    assert store.execution_drift_for(conn, cid)["classified"]["improvement"] == 1


def test_a_campaign_nobody_checked_has_no_drift_not_low_drift(conn):
    """The third state, and the assumption that makes this whole phase necessary. "Nobody
    looked" and "it ran as briefed" are not the same evidence."""
    cid = _campaign(conn, "Unchecked")

    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.1}, confirm=True)

    stored = store.execution_drift_for(conn, cid)
    assert stored["score"] is None
    assert stored["status"] == "never_checked"


# ── performance is never read naked ─────────────────────────────────────────

def test_two_good_results_stop_looking_identical(conn, tmp_path):
    """"`performed_well` at low drift and `performed_well` at high drift stop looking
    identical — the first is evidence the brief was good, the second is evidence something was
    good and the brief may not have been it.\""""
    faithful = _campaign(conn, "Faithful")
    _ran_as_briefed(conn, faithful, tmp_path, "f")
    core.add_metrics(conn, campaign_id=faithful, structured={"roas": 4.0}, confirm=True)

    drifted = _campaign(conn, "Drifted")
    _ran_differently(conn, drifted, tmp_path, "d")
    core.add_metrics(conn, campaign_id=drifted, structured={"roas": 4.0}, confirm=True)

    assert store.execution_drift_for(conn, faithful)["score"] < \
        store.execution_drift_for(conn, drifted)["score"]


def test_the_caveat_travels_with_the_evidence(conn, tmp_path):
    """D19's shape. A high-drift campaign cited as precedent has to say so where it is cited,
    not somewhere a reader has to go and look."""
    drifted = _campaign(conn, "Colombia drifted")
    _ran_differently(conn, drifted, tmp_path, "x")
    core.add_metrics(conn, campaign_id=drifted, structured={"roas": 4.0}, confirm=True)
    subject = _campaign(conn, "Colombia v3")

    package = core.prepare_evaluation(conn, subject_title="Colombia v3",
                                      proposal_text="A launch.", campaign_id=subject)
    cited = next(e for e in package["evidence"] if e["campaign_id"] == drifted)

    assert cited["execution"]["status"] == "drifted"
    assert "may not have been" in cited["execution"]["what_it_means"]


def test_a_faithful_campaign_says_that_too(conn, tmp_path):
    """The positive half matters as much: "this ran as briefed" is what makes its result
    evidence about the BRIEF, and a caveat that only ever appears as a warning teaches nothing
    about the other case."""
    faithful = _campaign(conn, "Peru faithful")
    _ran_as_briefed(conn, faithful, tmp_path, "p")
    core.add_metrics(conn, campaign_id=faithful, structured={"roas": 4.0}, confirm=True)
    subject = _campaign(conn, "Colombia v3")

    package = core.prepare_evaluation(conn, subject_title="Colombia v3",
                                      proposal_text="A launch.", campaign_id=subject)
    cited = next(e for e in package["evidence"] if e["campaign_id"] == faithful)

    assert cited["execution"]["status"] == "as_briefed"


def test_a_campaign_nobody_checked_says_nobody_checked(conn, tmp_path):
    unchecked = _campaign(conn, "Never checked")
    core.add_metrics(conn, campaign_id=unchecked, structured={"roas": 4.0}, confirm=True)
    subject = _campaign(conn, "Colombia v3")

    package = core.prepare_evaluation(conn, subject_title="Colombia v3",
                                      proposal_text="A launch.", campaign_id=subject)
    cited = next(e for e in package["evidence"] if e["campaign_id"] == unchecked)

    assert cited["execution"]["status"] == "never_checked"
    assert "nobody has checked" in cited["execution"]["what_it_means"].lower()


def test_high_drift_is_weaker_precedent_not_excluded_precedent(conn, tmp_path):
    """Dropping it would throw away the evidence the review most wants kept: the UAE claw
    machine departed from precedent and beat it."""
    drifted = _campaign(conn, "Colombia drifted")
    _ran_differently(conn, drifted, tmp_path, "y")
    core.add_metrics(conn, campaign_id=drifted, structured={"roas": 4.0}, confirm=True)
    subject = _campaign(conn, "Colombia v3")

    package = core.prepare_evaluation(conn, subject_title="Colombia v3",
                                      proposal_text="A launch.", campaign_id=subject)

    assert drifted in [e["campaign_id"] for e in package["evidence"]]


def test_the_note_tells_the_model_what_to_do_with_it(conn, tmp_path):
    drifted = _campaign(conn, "Colombia drifted")
    _ran_differently(conn, drifted, tmp_path, "z")
    core.add_metrics(conn, campaign_id=drifted, structured={"roas": 4.0}, confirm=True)
    subject = _campaign(conn, "Colombia v3")

    note = core.prepare_evaluation(conn, subject_title="Colombia v3",
                                   proposal_text="A launch.",
                                   campaign_id=subject)["note"]

    assert "drift" in note.lower()
    assert "ran as briefed" in note.lower()


def test_the_note_says_nothing_when_nothing_has_been_checked(conn):
    """A standing paragraph about execution drift, in a library where nobody has ever uploaded
    a delivered photograph, is the note that fires on everything."""
    subject = _campaign(conn, "Colombia v3")
    _campaign(conn, "Peru")

    note = core.prepare_evaluation(conn, subject_title="Colombia v3",
                                   proposal_text="A launch.",
                                   campaign_id=subject)["note"]
    assert "execution" not in note.lower()


def test_it_reaches_the_model_over_the_protocol(conn, tmp_path):
    import asyncio
    import json

    import mcp_server

    async def call(name, args):
        return json.loads((await mcp_server.mcp.call_tool(name, args)).content[0].text)

    drifted = _campaign(conn, "Colombia drifted")
    _ran_differently(conn, drifted, tmp_path, "w")
    core.add_metrics(conn, campaign_id=drifted, structured={"roas": 4.0}, confirm=True)
    subject = _campaign(conn, "Colombia v3")

    package = asyncio.run(call("prepare_evaluation", {
        "subject_title": "Colombia v3", "proposal_text": "A launch.",
        "campaign_id": subject}))
    cited = next(e for e in package["evidence"] if e["campaign_id"] == drifted)

    assert cited["execution"]["status"] == "drifted"


def test_photographs_arriving_after_the_results_still_change_the_answer(conn, tmp_path):
    """The ORDINARY sequence: results come in, the wrap deck weeks later. Snapshotting only on
    `add_metrics` froze every real campaign at `never_checked` — the photographs turned up
    afterwards and nothing looked again, so §9.5's whole point never fired."""
    cid = _campaign(conn, "Colombia")
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": _png(tmp_path, "brief.png", seed=3)})
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.1}, confirm=True)
    assert store.execution_drift_for(conn, cid)["status"] == "never_checked"

    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": _png(tmp_path, "shot.png", seed=91)},
                            phase="delivered", captured_on="2026-03-14")

    assert store.execution_drift_for(conn, cid)["status"] == "drifted"


def test_a_classification_updates_what_a_citation_carries(conn, tmp_path):
    """§9.5 stores the classified counts beside the outcome, so a snapshot that never sees a
    classification is the stale half of the same figure."""
    cid = _campaign(conn, "UAE", market="EMEA")
    _ran_differently(conn, cid, tmp_path, "q")
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 4.2}, confirm=True)
    assert store.execution_drift_for(conn, cid)["classified"].get("improvement", 0) == 0

    drift.classify(conn, campaign_id=cid, subject=_claw(conn, cid),
                   classification="improvement", why="It drew the queue.",
                   classified_by="R. Vega")

    assert store.execution_drift_for(conn, cid)["classified"]["improvement"] == 1


def test_a_campaign_with_no_results_is_not_snapshotted(conn, tmp_path):
    """Nothing to qualify yet — and it keeps a bulk import of five hundred rows from running
    five hundred comparisons, since workbook campaigns almost never have photographs."""
    cid = _campaign(conn, "Not yet")
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": _png(tmp_path, "shot.png", seed=91)},
                            phase="delivered", captured_on="2026-03-14")

    assert store.execution_drift_for(conn, cid)["status"] == "never_checked"


def test_the_three_statuses_do_not_say_the_same_thing(conn):
    """The distinction is the entire item. Three statuses sharing one sentence would carry the
    field and lose the meaning."""
    said = {status: core._EXECUTION_MEANING[status]
            for status in ("as_briefed", "drifted", "never_checked")}

    assert len(set(said.values())) == 3
    assert "evidence about the BRIEF" in said["as_briefed"]
    assert "may not have been it" in said["drifted"]
    assert "not the same as it having run faithfully" in said["never_checked"]


def test_drift_is_not_recorded_before_there_is_a_result_to_qualify(conn, tmp_path):
    """The guard is what keeps a bulk import of five hundred workbook rows from running five
    hundred comparisons — and a drift figure with no outcome beside it qualifies nothing."""
    cid = _campaign(conn, "No results yet")
    _ran_differently(conn, cid, tmp_path, "n")

    assert store.execution_drift_for(conn, cid)["status"] == "never_checked"

    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.1}, confirm=True)
    assert store.execution_drift_for(conn, cid)["status"] == "drifted"


# ── the computed downgrade is correctable ───────────────────────────────────

def test_a_person_can_say_the_comparison_was_wrong(conn, tmp_path):
    """§9.2 says in its own comments that a photograph of a built claw machine will not
    fingerprint-match the briefed render. So every campaign with site photography is stamped
    `drifted`, `_say_the_execution_drift` tells the model to weigh it as weaker precedent, and
    the four judgments on offer all began "the execution moved away from the brief" — there was
    no way to record "it did not; you just couldn't tell", and the downgrade stood forever."""
    cid = _campaign(conn, "UAE", market="EMEA")
    _ran_differently(conn, cid, tmp_path)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 4.2}, confirm=True)
    assert store.execution_drift_for(conn, cid)["status"] == "drifted"

    for item in core.compare_execution(conn, campaign_id=cid)["unclassified_items"]:
        drift.classify(conn, campaign_id=cid, subject=item["subject"],
                       classification="not_drift",
                       why="The photographs show the claw machine that was briefed.",
                       classified_by="R. Vega")

    note = core._execution_note(conn, cid)
    assert core._reads_as_briefed(note)
    assert "evidence about the BRIEF" in note["what_it_means"]
    assert "weigh it as precedent accordingly" not in note["what_it_means"]


def test_the_correction_is_a_judgment_and_says_so(conn, tmp_path):
    """It disputes a computed fact, which is more than the other four do — so the record has to
    make plain that a person did it and which person."""
    cid = _campaign(conn, "UAE", market="EMEA")
    _ran_differently(conn, cid, tmp_path)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 4.2}, confirm=True)

    saved = drift.classify(conn, campaign_id=cid, subject=_claw(conn, cid),
                           classification="not_drift", why="It is in frame in shot 3.",
                           classified_by="R. Vega")

    assert saved["basis"] == "judged"
    assert "R. Vega" in saved["what_it_means"]
    assert store.execution_drift_for(conn, cid)["status"] == "drifted", (
        "the computed status is what the instrument saw, and stays what it saw")
    # One of the two differences corrected. The other is still one the comparison saw and
    # nobody disputed, so the campaign is not yet faithful — reading it as faithful on the
    # strength of the first correction is the same over-reach in the other direction.
    assert core._reads_as_briefed(core._execution_note(conn, cid)) is False


def test_a_corrected_campaign_counts_as_faithful_to_the_model(conn, tmp_path):
    cid = _campaign(conn, "UAE", market="EMEA")
    _ran_differently(conn, cid, tmp_path)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 4.2}, confirm=True)
    for item in core.compare_execution(conn, campaign_id=cid)["unclassified_items"]:
        drift.classify(conn, campaign_id=cid, subject=item["subject"],
                       classification="not_drift", why="Both are in the wrap photographs.",
                       classified_by="R. Vega")

    said = core._say_the_execution_drift(
        [{"execution": core._execution_note(conn, cid)}])

    assert "1 of these ran as briefed" in said
    assert "1 drifted" not in said


# ── the faithful precedent is retrieved, not partitioned ────────────────────

def test_precedent_that_ran_as_briefed_is_searched_for_in_its_own_right(conn, tmp_path):
    """`_both_poles`' argument, one axis over: partitioning the ranked five reports "nothing
    here ran as briefed", which is a statement about the RANKING dressed as one about the
    library."""
    faithful = _campaign(conn, "Peru", market="LATAM")
    _ran_as_briefed(conn, faithful, tmp_path, tag="p")
    core.add_metrics(conn, campaign_id=faithful, structured={"roas": 3.9}, confirm=True)

    out = core._ran_as_briefed(conn, evidence=[], text="A launch activation.",
                               campaign_id=None)

    assert [f["campaign_id"] for f in out["found"]] == [faithful]
    assert out["found"][0]["in_evidence"] is False


def test_a_library_nobody_checked_says_that_rather_than_reporting_drift(conn, tmp_path):
    _campaign(conn, "Peru", market="LATAM")

    out = core._ran_as_briefed(conn, evidence=[], text="A launch activation.",
                               campaign_id=None)

    assert out["found"] == []
    assert "has been CHECKED" in out["what_it_means"]
    assert "an unchecked campaign is not a faithful one" in out["what_it_means"].lower()


def test_the_package_carries_it(conn, tmp_path):
    faithful = _campaign(conn, "Peru", market="LATAM")
    _ran_as_briefed(conn, faithful, tmp_path, tag="p")
    core.add_metrics(conn, campaign_id=faithful, structured={"roas": 3.9}, confirm=True)

    pkg = core.prepare_evaluation(conn, subject_title="Dubai v2",
                                  proposal_text="A launch activation.")

    assert pkg["ran_as_briefed"]["basis"] == "computed"
    assert faithful in {f["campaign_id"] for f in pkg["ran_as_briefed"]["found"]}


def test_what_worked_carries_the_caveat_too(conn, tmp_path):
    """§9.5's sentence is about `performed_well`, and that pole is the one place it was not
    stated — "this one worked" is the line a reasoner leans on hardest."""
    cid = _campaign(conn, "UAE", market="EMEA")
    _ran_differently(conn, cid, tmp_path)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 4.2}, confirm=True)
    core.update_campaign(conn, campaign_id=cid,
                         tags=[{"value": "performed_well", "source": "verified"}])

    poles = core._both_poles(conn, evidence=[], text="A launch activation.", campaign_id=None)

    worked = [row for row in poles["worked"] if row["campaign_id"] == cid]
    assert worked, poles
    assert worked[0]["execution"]["status"] == "drifted"


# ── the negative cases the mutation pass found unguarded ────────────────────

def test_a_blank_classification_is_refused_rather_than_defaulted(conn, tmp_path):
    """The central claim of §9.4 is that the server never classifies. A default — any default —
    is the server classifying, arriving through the quietest possible door."""
    cid = _campaign(conn, "UAE", market="EMEA")

    with pytest.raises(ValueError):
        drift.classify(conn, campaign_id=cid, subject=_claw(conn, cid), classification="  ",
                       why="It drew a queue.", classified_by="R. Vega")

    assert drift.for_campaign(conn, cid)["counts"] == {v: 0 for v in drift.CLASSIFICATIONS}


def test_a_campaign_that_does_not_exist_is_refused(conn):
    with pytest.raises(ValueError) as e:
        drift.classify(conn, campaign_id="nope", subject="anything",
                       classification="improvement", why="It drew a queue.",
                       classified_by="R. Vega")
    assert "nope" in str(e.value)


def test_a_blank_subject_is_refused_with_what_to_pass(conn):
    cid = _campaign(conn, "UAE", market="EMEA")

    with pytest.raises(ValueError) as e:
        drift.classify(conn, campaign_id=cid, subject="   ", classification="improvement",
                       why="It drew a queue.", classified_by="R. Vega")
    assert "commitment_id" in str(e.value)


def test_the_execution_note_is_computed_not_judged(conn, tmp_path):
    """The status is what the comparison saw. Stamped `judged`, the same word would read as a
    person's verdict on the campaign — and §2.4 made `basis` mean something exactly so that
    cannot happen by accident."""
    cid = _campaign(conn, "UAE", market="EMEA")
    _ran_differently(conn, cid, tmp_path)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 4.2}, confirm=True)
    drift.classify(conn, campaign_id=cid, subject=_claw(conn, cid),
                   classification="improvement", why="It drew the queue.",
                   classified_by="R. Vega")

    assert core._execution_note(conn, cid)["basis"] == "computed"


def test_too_early_after_the_numbers_never_claims_nothing_was_measured(conn, tmp_path):
    """The campaign's numbers can be in while nothing in them settles THIS difference — but the
    server must not write "nothing has been measured yet" over metrics it is holding."""
    cid = _campaign(conn, "UAE", market="EMEA")
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 4.2}, confirm=True)

    saved = drift.classify(conn, campaign_id=cid, subject=_claw(conn, cid),
                           classification="too_early",
                           why="Footfall was never counted at the machine.",
                           classified_by="R. Vega")

    assert "nothing has been measured yet" not in saved["what_it_means"].lower()
    assert "measured result on file" in saved["what_it_means"]


def test_a_bulk_import_snapshots_each_campaign_once(conn, tmp_path):
    """Fifty workbook rows for one campaign ran the whole comparison fifty times and kept the
    last answer."""
    cid = _campaign(conn, "Colombia")
    _ran_differently(conn, cid, tmp_path)
    calls = []
    real = core._snapshot_execution_drift
    core._snapshot_execution_drift = lambda c, i: (calls.append(i), real(c, i))[1]
    try:
        store.bulk_import_metrics(
            conn, [{"campaign_id": cid, "structured": {"roas": 3.0 + n / 10}}
                   for n in range(8)], confirm=True)
    finally:
        core._snapshot_execution_drift = real

    assert calls == [cid], calls
    assert store.execution_drift_for(conn, cid)["status"] == "drifted"


def test_the_classified_items_are_capped(conn, tmp_path):
    """Every other list here is trimmed; uncapped, five hundred classifications made
    compare_execution a 330 KB response that landed in the caller's context.

    The subjects have to be genuinely DISTINCT — `for_campaign` keeps the latest reading per
    subject, so classifying the same two things sixteen times leaves two items and a cap of
    twelve that never binds. Sizing the loop from `MAX_ITEMS_SHOWN` itself has the same
    problem in the other direction: raise the constant and the test raises with it.
    """
    # A LITERAL, not `MAX_ITEMS_SHOWN + 3`. Sized from the constant, the test scales with it:
    # raise the cap and the fixture grows to match, so the cap is never exceeded and the
    # assertion below cannot fail. (Found by mutating the constant, which hung the suite
    # building a hundred thousand fixtures rather than failing.)
    assert drift.MAX_ITEMS_SHOWN < 15, "this fixture must exceed the cap; resize it"
    cid = _campaign(conn, "UAE", market="EMEA")
    for n in range(15):
        core.ingest_image_asset(conn, campaign_id=cid,
                                asset_ref={"path": _png(tmp_path, f"brief{n}.png", seed=n + 5)})
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": _png(tmp_path, "shot.png", seed=999)},
                            phase="delivered", captured_on="2026-03-14")
    subjects = [i["subject"] for i in core.compare_execution(
        conn, campaign_id=cid)["unclassified_items"]]
    assert len(subjects) > drift.MAX_ITEMS_SHOWN, (
        f"only {len(subjects)} distinct subjects — the cap would never bind")
    for n, subject in enumerate(subjects):
        drift.classify(conn, campaign_id=cid, subject=subject, classification="neutral",
                       why="It made no difference at all.", classified_by="R. Vega")

    summary = drift.for_campaign(conn, cid)

    assert len(summary["items"]) == drift.MAX_ITEMS_SHOWN
    assert summary["items_truncated"] is True
    assert summary["items_total"] == len(subjects)
    assert summary["counts"]["neutral"] == len(subjects), "the counts stay complete"
