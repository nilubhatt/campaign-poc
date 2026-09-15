"""
§7.7: the instrument. Consistency is a number, not a feeling.

The review: *"Everything above is a hypothesis until it is measured. Fix: keep ten to twenty
briefs with client-agreed verdicts and expected blocking findings. Run them on every release
and report two figures: verdict agreement (does the Approve/Revise/Reject match) and finding
recall (what fraction of the expected blocking findings were raised). Three runs per brief."*

That first sentence is what places this item. Phases 6 and 7 are twenty-odd changes made on
the argument that they reduce variance, and not one has been measured — every claim in them is
an argument until this exists.

**What ships is the harness, not the briefs.** Ten to twenty briefs with CLIENT-AGREED
verdicts is the definition, and only the product owner can supply those. Inventing them would
measure agreement with my own guesses, which is worse than measuring nothing because it
produces a number somebody will quote.

**The judging is supplied too.** This server cannot run a model — the model is on the other
side of the protocol — so `run` takes a callable and measures what comes back. That is also
what makes the measurement testable: a stub judge with known behaviour is the only way to
check that the arithmetic is right.

**Three figures, not two.** The review asks for verdict agreement and finding recall. Three
runs per brief measures a third thing neither can see: whether the product answers the same
question the same way twice. A brief that matches the client two runs in three is not a brief
the product judges correctly, and the two headline figures would both report it as a partial
success rather than as instability.
"""
from __future__ import annotations

import json
import pathlib
from typing import Callable, Optional

import core

# Where a customer's golden set lives. Empty in the repository by design — see the module
# docstring, and §7.7's entry in the plan.
GOLDEN_SET_PATH = pathlib.Path(__file__).parent / "golden_set.json"

REQUIRED_FIELDS = ("id", "subject_title", "proposal_text", "expected_verdict")


def load(path: Optional[pathlib.Path] = None) -> list:
    """The golden set, or an empty list.

    Empty is the shipped state and not a failure: the briefs are the customer's, and a harness
    that invented them would report a number measured against nothing.
    """
    path = path or GOLDEN_SET_PATH
    if not path.exists():
        return []
    briefs = json.loads(path.read_text())
    return briefs if isinstance(briefs, list) else []


def _normalise(text: str) -> str:
    return " ".join((text or "").lower().split())


def _recall(expected: list, raised: list) -> Optional[float]:
    """What fraction of the expected blocking findings were raised.

    RECALL, not precision, exactly as the review asks: a judgment that raises the two expected
    problems and one more has not failed. Matched on containment either way round, because the
    expected finding is a short phrase a person wrote and the raised one is a sentence a model
    wrote, and requiring them to be equal would measure phrasing.
    """
    if not expected:
        return None
    raised_text = [_normalise(f.get("finding", "")) for f in raised
                   if f.get("severity") == "blocking"]
    hit = sum(1 for want in expected
              if any(_normalise(want) in got or got in _normalise(want)
                     for got in raised_text))
    return hit / len(expected)


def run(conn, briefs: list, *, judge: Callable, runs: int = 3) -> dict:
    """Measure the three figures over a golden set.

    `judge(brief, run_number)` returns a saved-evaluation-shaped dict: at minimum a `verdict`
    and a `findings` list. It is supplied by the caller because the model is on the other side
    of the protocol; in tests it is a stub, and on a release it is a real client.
    """
    for brief in briefs:
        missing = [f for f in REQUIRED_FIELDS if not brief.get(f)]
        if missing:
            raise ValueError(
                f"golden-set brief {brief.get('id', '?')!r} is missing {', '.join(missing)}. "
                f"A brief with no client-agreed verdict cannot measure agreement with one — "
                f"that is a fault in the set, not a judgment the product failed.")

    measured = []
    for brief in briefs:
        verdicts, recalls, errors = [], [], 0
        for n in range(1, runs + 1):
            try:
                answer = judge(brief, n) or {}
            except Exception:                      # noqa: BLE001
                # A model that failed to answer did not disagree with the client. Counting a
                # crash as a wrong verdict would make the product look inconsistent when it
                # was merely unavailable.
                errors += 1
                continue
            verdicts.append(answer.get("verdict"))
            recall = _recall(brief.get("expected_blocking") or [],
                             answer.get("findings") or [])
            if recall is not None:
                recalls.append(recall)
        agreed = [v for v in verdicts if v == brief["expected_verdict"]]
        measured.append({
            "id": brief["id"],
            "expected_verdict": brief["expected_verdict"],
            "verdicts": verdicts,
            "verdict_agreement": (len(agreed) / len(verdicts)) if verdicts else None,
            "finding_recall": (sum(recalls) / len(recalls)) if recalls else None,
            # The third figure. Not "did it agree with the client" but "did it agree with
            # itself" — a brief answered two ways in three runs is unstable whatever the
            # headline figures say about it.
            "stable": (len(set(verdicts)) == 1) if verdicts else None,
            "errors": errors,
        })

    usable = [b for b in measured if b["verdict_agreement"] is not None]
    with_recall = [b for b in measured if b["finding_recall"] is not None]
    report = {
        "briefs": measured,
        "runs_per_brief": runs,
        "verdict_agreement": (sum(b["verdict_agreement"] for b in usable) / len(usable)
                              if usable else None),
        "finding_recall": (sum(b["finding_recall"] for b in with_recall) / len(with_recall)
                           if with_recall else None),
        "self_consistency": _self_consistency(measured),
        # §7.6's stamp, on the measurement itself. A figure with no record of the conditions
        # it was measured under cannot be compared with last release's, and a change in it
        # cannot be attributed to anything.
        "provenance": {
            "server_version": core.version.VERSION,
            "rulebook_version": core.RULEBOOK_VERSION,
            "embedding_model": core.embedding_model_id(),
        },
    }
    report.update(_verdict(measured, usable))
    return report


def _self_consistency(measured: list) -> Optional[float]:
    """How often the most common answer was the answer, across every run of every brief.

    Deliberately not "how many briefs were stable": a brief answered 2-1 and one answered 1-1-1
    are both unstable, and the second is worse.
    """
    runs = [v for b in measured for v in b["verdicts"]]
    if not runs:
        return None
    total = 0.0
    for b in measured:
        if not b["verdicts"]:
            continue
        commonest = max(set(b["verdicts"]), key=b["verdicts"].count)
        total += b["verdicts"].count(commonest)
    return total / len(runs)


def _verdict(measured: list, usable: list) -> dict:
    """What the report MEANS, in the same shape every other check in this codebase uses.

    An empty golden set is the case worth being careful about: zero briefs agreeing out of
    zero is not a perfect score, and a harness reporting 100% on an empty set would say so on
    every release until somebody noticed.
    """
    if not measured:
        return {"code": "no_golden_set",
                "what_it_means": (
                    "No golden set is present, so nothing was measured. This is not a passing "
                    "score — it is an absent one. The set is ten to twenty real briefs with "
                    "client-agreed verdicts and expected blocking findings, and only the "
                    "product owner can supply them.")}
    if not usable:
        return {"code": "no_usable_runs",
                "what_it_means": (
                    "Every run failed, so nothing was measured. The product did not disagree "
                    "with anybody; it did not answer.")}
    unstable = [b["id"] for b in measured if b["stable"] is False]
    if unstable:
        return {"code": "unstable",
                "what_it_means": (
                    f"{len(unstable)} brief(s) were answered more than one way across runs: "
                    f"{', '.join(unstable)}. Read that before the agreement figures — a "
                    f"product that answers the same question differently twice has not agreed "
                    f"with anybody, it has guessed consistently enough to look like it.")}
    return {"code": "measured",
            "what_it_means": (
                "Every brief was answered the same way on every run, so the agreement figures "
                "describe the product's judgment rather than the spread of its guesses.")}


def _cli() -> int:
    """`python -m agreement` — the thing a release runs.

    Prints the report and exits 0 even with no golden set, because an absent measurement is
    not a failing build; it is an absent measurement, and failing CI over it would make the
    first person to add a brief the person who broke the pipeline.
    """
    import store

    briefs = load()
    conn = store.connect()
    try:
        report = run(conn, briefs, judge=_no_judge, runs=3)
    finally:
        conn.close()
    print(json.dumps({k: v for k, v in report.items() if k != "briefs"}, indent=2))
    if report["code"] == "no_golden_set":
        print(f"\nNo golden set at {GOLDEN_SET_PATH}. It is ten to twenty real briefs with "
              f"client-agreed verdicts and expected blocking findings, which only the product "
              f"owner can supply — see docs/PRODUCT-REVIEW-PLAN.md §7.7. Each entry needs "
              f"{', '.join(REQUIRED_FIELDS)} and optionally `expected_blocking`.")
    else:
        print(f"\n{report['what_it_means']}")
    return 0


def _no_judge(brief, run_number):
    """The placeholder judge for a run with no client attached.

    It raises, so the report says every run errored rather than inventing verdicts — the CLI
    exists to prove the harness runs and to print what is missing, not to produce figures out
    of nothing.
    """
    raise RuntimeError("no judging client is attached to this run")


if __name__ == "__main__":
    raise SystemExit(_cli())
