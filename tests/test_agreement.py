"""
§7.7: build a golden set and measure agreement.

The review's words: *"Everything above is a hypothesis until it is measured. Consistency is a
number, not a feeling. Fix: keep ten to twenty briefs with client-agreed verdicts and expected
blocking findings. Run them on every release and report two figures: verdict agreement (does
the Approve/Revise/Reject match) and finding recall (what fraction of the expected blocking
findings were raised). Three runs per brief."*

**"Everything above is a hypothesis until it is measured"** is the sentence that places this
item. Phases 6 and 7 are twenty-odd changes made on the argument that they reduce variance,
and not one of them has been measured. This is the instrument, and until it exists every claim
in those phases is an argument rather than a result.

**What ships is the harness, not the briefs.** The plan records that decision: the golden set
is ten to twenty real briefs with client-agreed verdicts, and only the product owner can
supply those. A harness with invented briefs would measure agreement with my own guesses,
which is worse than measuring nothing because it produces a number.

**The judging is supplied too.** This server cannot run a model; the model is on the other
side of the protocol. So the harness takes a callable and measures what comes back, which is
also what makes it testable here — a stub judge with known behaviour is the only way to check
that the measurement itself is right.
"""
import pytest

import agreement


def _brief(**over):
    base = {"id": "colombia-v1", "subject_title": "Colombia v1",
            "proposal_text": "A creator-led launch with four colourways.",
            "expected_verdict": "revise",
            "expected_blocking": ["no posting dates", "conflicted influencer"]}
    base.update(over)
    return base


def _judge(verdict="revise", findings=("no posting dates", "conflicted influencer")):
    def judge(brief, run):
        return {"verdict": verdict,
                "findings": [{"severity": "blocking", "finding": f} for f in findings]}
    return judge


# ── the two figures the review asks for ─────────────────────────────────────

def test_verdict_agreement_is_reported(conn):
    report = agreement.run(conn, [_brief()], judge=_judge(), runs=3)
    assert report["verdict_agreement"] == 1.0


def test_a_disagreeing_verdict_lowers_it(conn):
    report = agreement.run(conn, [_brief()], judge=_judge(verdict="approve"), runs=3)
    assert report["verdict_agreement"] == 0.0


def test_finding_recall_is_reported(conn):
    """"What fraction of the expected blocking findings were raised." Recall, not precision:
    a judgment that raises the two expected problems and one more has not failed."""
    report = agreement.run(conn, [_brief()],
                           judge=_judge(findings=("no posting dates",)), runs=3)
    assert report["finding_recall"] == 0.5


def test_an_extra_finding_does_not_count_against_recall(conn):
    report = agreement.run(
        conn, [_brief()],
        judge=_judge(findings=("no posting dates", "conflicted influencer", "budget thin")),
        runs=3)
    assert report["finding_recall"] == 1.0


# ── three runs per brief, and what that is for ──────────────────────────────

def test_it_runs_each_brief_the_number_of_times_asked(conn):
    seen = []

    def judge(brief, run):
        seen.append((brief["id"], run))
        return {"verdict": "revise", "findings": []}

    agreement.run(conn, [_brief()], judge=judge, runs=3)
    assert seen == [("colombia-v1", 1), ("colombia-v1", 2), ("colombia-v1", 3)]


def test_a_brief_that_answers_differently_across_runs_is_reported_as_unstable(conn):
    """Three runs per brief measures something the other two figures cannot: whether the
    product answers the SAME question the same way twice. A brief that agrees with the client
    two runs in three is not a brief the product judges correctly."""
    answers = iter(["revise", "revise", "approve"])

    def judge(brief, run):
        return {"verdict": next(answers), "findings": []}

    report = agreement.run(conn, [_brief(expected_blocking=[])], judge=judge, runs=3)

    assert report["self_consistency"] == pytest.approx(2 / 3)
    assert report["briefs"][0]["stable"] is False


def test_a_brief_that_answers_identically_is_stable(conn):
    report = agreement.run(conn, [_brief(expected_blocking=[])], judge=_judge(), runs=3)
    assert report["self_consistency"] == 1.0
    assert report["briefs"][0]["stable"] is True


# ── the numbers have to be attributable, or they diagnose nothing ───────────

def test_the_report_stamps_what_produced_it(conn):
    """§7.6 exists so that agreement can be read as evidence rather than luck. A figure with
    no record of the conditions it was measured under is a number nobody can act on: it cannot
    be compared with last release's, and a change in it cannot be attributed."""
    import core

    report = agreement.run(conn, [_brief()], judge=_judge(), runs=3)

    assert report["provenance"]["server_version"] == core.version.VERSION
    assert report["provenance"]["embedding_model"] == core.embedding_model_id()
    assert report["provenance"]["rulebook_version"] == core.rulebook_version()


def test_each_brief_reports_its_own_figures(conn):
    """An aggregate hides which brief moved. "Verdict agreement fell to 0.8" is not actionable;
    "Colombia v1 started answering approve" is."""
    report = agreement.run(
        conn, [_brief(), _brief(id="peru-v2", expected_verdict="approve")],
        judge=_judge(), runs=3)

    by_id = {b["id"]: b for b in report["briefs"]}
    assert by_id["colombia-v1"]["verdict_agreement"] == 1.0
    assert by_id["peru-v2"]["verdict_agreement"] == 0.0
    assert by_id["peru-v2"]["expected_verdict"] == "approve"
    assert by_id["peru-v2"]["verdicts"] == ["revise", "revise", "revise"]


# ── an empty golden set is not a passing one ────────────────────────────────

def test_an_empty_golden_set_reports_no_figures_rather_than_perfect_ones(conn):
    """Zero briefs divided by zero briefs is not 100% agreement. A harness that reports a
    perfect score on an empty set is the confident-shape-with-nothing-in-it failure this
    project keeps finding, and it would be reported on every release until somebody supplied
    the briefs."""
    report = agreement.run(conn, [], judge=_judge(), runs=3)

    assert report["verdict_agreement"] is None
    assert report["finding_recall"] is None
    assert report["code"] == "no_golden_set"
    assert "product owner" in report["what_it_means"].lower()


def test_the_shipped_golden_set_is_empty_and_says_why(conn):
    """The plan's own decision: ten to twenty real briefs with client-agreed verdicts, and
    only the product owner can supply them. Inventing briefs would measure agreement with my
    own guesses — worse than measuring nothing, because it produces a number."""
    briefs = agreement.load()
    assert briefs == []


# ── a malformed brief is a fault in the set, not a failed judgment ──────────

def test_a_brief_missing_its_expected_verdict_is_refused(conn):
    with pytest.raises(ValueError) as e:
        agreement.run(conn, [{"id": "x", "subject_title": "x", "proposal_text": "x"}],
                      judge=_judge(), runs=3)
    assert "expected_verdict" in str(e.value)


def test_a_judge_that_raises_is_reported_as_an_error_not_a_disagreement(conn):
    """A model that failed to answer did not disagree with the client. Counting a crash as a
    wrong verdict would make the product look inconsistent when it was unavailable."""
    def judge(brief, run):
        raise RuntimeError("the model timed out")

    report = agreement.run(conn, [_brief()], judge=judge, runs=3)

    assert report["briefs"][0]["errors"] == 3
    assert report["verdict_agreement"] is None
    assert report["code"] == "no_usable_runs"


# ── a release has to be able to run it ──────────────────────────────────────

def test_the_measurement_is_reachable_from_the_command_line():
    """"Run them on every release." A harness nobody can invoke is a module, and this is the
    one item whose value is entirely in being run repeatedly."""
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-m", "agreement"],
                            capture_output=True, text=True, cwd=".")
    assert result.returncode == 0, result.stderr
    assert "no_golden_set" in result.stdout


def test_the_command_line_says_what_to_do_about_an_empty_set():
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-m", "agreement"],
                         capture_output=True, text=True, cwd=".").stdout
    assert "product owner" in out.lower()
    assert "golden_set.json" in out


# ── where the agreement holds, not only how high it averages ────────────────
#
# D56 and D69 ask two questions a mean cannot answer, and the tracker said of both that "the
# harness exists and runs; what is missing is the input". It ran, and returned a flat figure:
# ready for the headline and for neither stratification. So the golden set arriving would not
# have closed the rows that were waiting for it.


def _stub_context(fired, evidence):
    return lambda conn, brief: {"missing_input_fired": fired(brief),
                                "evidence_count": evidence(brief)}


def test_agreement_is_reported_where_the_library_is_thin_and_where_it_is_not(conn):
    """D69, in its own words: "does verdict agreement drop where the library is thin?" A
    product that is excellent on well-covered briefs and guesses on thin ones reports the same
    headline as one that is mediocre everywhere — and only the first is worth shipping while
    the library fills up."""
    briefs = [{"id": "thick", "subject_title": "Lima", "proposal_text": "A push.",
               "expected_verdict": "approve"},
              {"id": "thin", "subject_title": "Tokyo", "proposal_text": "A push.",
               "expected_verdict": "approve"}]

    def judge(brief, run_number):
        # Agrees where there is evidence; guesses where there is none.
        if brief["id"] == "thick":
            return {"verdict": "approve"}
        return {"verdict": "approve" if run_number == 1 else "revise"}

    report = agreement.run(conn, briefs, judge=judge, runs=3,
                           context=_stub_context(lambda b: False,
                                                 lambda b: 5 if b["id"] == "thick" else 0))

    by_evidence = report["verdict_agreement_by"]["evidence"]
    assert by_evidence["several"]["verdict_agreement"] == 1.0
    assert by_evidence["none"]["verdict_agreement"] < 0.5, by_evidence
    assert 0.5 < report["verdict_agreement"] < 1.0, (
        "and the flat figure hides both, which is why the split exists")


def test_agreement_is_reported_for_the_briefs_the_missing_input_line_fired_on(conn):
    """D56. The line says a brief is missing something the library would need; the question is
    whether the verdicts on those briefs agree less with the client's. Averaged together, the
    answer is unavailable in either direction."""
    briefs = [{"id": f"b{n}", "subject_title": f"Brief {n}", "proposal_text": "A push.",
               "expected_verdict": "approve"} for n in range(2)]

    def judge(brief, run_number):
        return {"verdict": "approve" if brief["id"] == "b0" else "revise"}

    report = agreement.run(conn, briefs, judge=judge, runs=2,
                           context=_stub_context(lambda b: b["id"] == "b1", lambda b: 3))

    by_line = report["verdict_agreement_by"]["missing_input_line"]
    assert by_line["did_not_fire"]["verdict_agreement"] == 1.0
    assert by_line["fired"]["verdict_agreement"] == 0.0
    assert by_line["fired"]["briefs"] == 1


def test_every_stratified_figure_says_how_many_briefs_it_rests_on(conn):
    """"1.0 agreement" over one brief and over fourteen are different claims, and a stratified
    report is exactly where one brief can look like a finding."""
    briefs = [{"id": "only", "subject_title": "Lima", "proposal_text": "A push.",
               "expected_verdict": "approve"}]
    report = agreement.run(conn, briefs, judge=lambda b, n: {"verdict": "approve"}, runs=3,
                           context=_stub_context(lambda b: True, lambda b: 1))

    for split in report["verdict_agreement_by"].values():
        assert all("briefs" in figure for figure in split.values()), split
    assert report["verdict_agreement_by"]["evidence"]["thin"]["briefs"] == 1


def test_what_the_server_refused_during_the_run_is_counted(conn):
    """D9's second half: "count rule-2 rejections to detect systematic severity downgrading".
    A guardrail breach filed as a note is refused by the validator, so it never reaches a
    verdict and no agreement figure can see it — a model quietly downgrading severities looks
    like a model that agrees. The counter is the only witness.

    Counted across the run rather than read from the database, so the figure belongs to this
    measurement and not to everything the library has ever refused."""
    import core

    try:                                   # refused, and BEFORE the run starts
        core.save_evaluation(conn, subject_title="Earlier", campaign_id=None,
                             verdict="revise", summary="x", approve_if="y",
                             findings=[{"severity": "note", "kind": "guardrail_breach",
                                        "finding": "before the run", "fix": "z"}])
    except ValueError:
        pass
    briefs = [{"id": "b0", "subject_title": "Lima", "proposal_text": "A push.",
               "expected_verdict": "revise"}]

    def judge(brief, run_number):
        try:
            core.save_evaluation(conn, subject_title=brief["subject_title"], campaign_id=None,
                                 verdict="revise", summary="Seeds four.", approve_if="Seed one.",
                                 findings=[{"severity": "note", "kind": "guardrail_breach",
                                            "finding": "Seeds four colourways", "fix": "Seed one"}])
        except ValueError:
            pass                      # the refusal is the thing being counted
        return {"verdict": "revise"}

    report = agreement.run(conn, briefs, judge=judge, runs=2,
                           context=_stub_context(lambda b: False, lambda b: 2))

    assert report["refusals_during_the_run"].get("breach_as_note") == 2, (
        report["refusals_during_the_run"])
