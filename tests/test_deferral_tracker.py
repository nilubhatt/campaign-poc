"""
The deferral tracker has to stay true, or it is worse than not having one.

`docs/DEFERRALS.md` records every piece of work consciously not done and where it went
instead. It is a hand-maintained list pointing at another hand-maintained list — which is
precisely the shape that has already failed three times in this project: the Windows
installer's `AppVersion` drifted from `version.py`, `TOOL_NAMES` drifted from the registered
tools, and the plan's own text drifted from the code it described. Each time the copy was
right on the day it was written.

So these check the two things that can rot silently: a deferral pointing at an item that does
not exist, and a deferral still listed as owed against an item already marked done.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TRACKER = ROOT / "docs" / "DEFERRALS.md"
PLAN = ROOT / "docs" / "PRODUCT-REVIEW-PLAN.md"

ITEM = re.compile(r"\b(\d{1,2}\.\d{1,2})\b")


def _tracker_rows(section: str) -> list[tuple[str, str]]:
    """(row id, whole row) for one `## ...` section of the tracker."""
    text = TRACKER.read_text(encoding="utf-8")
    start = text.index(f"## {section}")
    rest = text[start + 1:]
    end = rest.index("\n## ") if "\n## " in rest else len(rest)
    rows = []
    for line in rest[:end].splitlines():
        if not line.startswith("| ") or line.startswith("| #") or set(line) <= set("| -"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append((cells[0], line))
    return rows


def _plan_items() -> dict[str, bool]:
    """Every numbered plan item -> whether it is marked done."""
    items = {}
    for line in PLAN.read_text(encoding="utf-8").splitlines():
        match = re.match(r"- \[([ x])\] \*\*(\d{1,2}\.\d{1,2})", line)
        if match:
            items[match.group(2)] = match.group(1) == "x"
    return items


def test_the_tracker_exists_and_has_rows():
    """An empty tracker and no tracker are the same artefact."""
    assert TRACKER.exists()
    assert _tracker_rows("Open — deferred"), "no open deferrals listed"


def test_every_deferral_points_at_an_item_that_exists():
    """A deferral aimed at an item number nobody ever wrote is a deferral that will never be
    picked up, and it reads as tracked."""
    items = _plan_items()

    for row_id, row in _tracker_rows("Open — deferred"):
        # Columns: | id | deferred | From | To | why |  -> index 4 is "To". Index 3 is
        # "From", and using it made every row look stale against its own source item.
        targets = ITEM.findall(row.split("|")[4])
        assert targets, f"{row_id} names no destination item"
        for target in targets:
            assert target in items, (
                f"{row_id} defers to item {target}, which does not exist in the plan"
            )


def test_nothing_is_still_deferred_to_an_item_already_finished():
    """The failure mode that makes a tracker actively misleading: the item shipped, the
    deferred work was not done, and the row still says it is owed to that item. Either the
    work landed (move the row to Closed) or it did not (the item is not done)."""
    items = _plan_items()

    stale = []
    for row_id, row in _tracker_rows("Open — deferred"):
        for target in ITEM.findall(row.split("|")[4]):
            if items.get(target):
                stale.append(f"{row_id} -> {target}")

    assert not stale, (
        f"these are owed to items already marked done: {stale}. Either close the row or "
        f"un-tick the item."
    )


def test_a_rejected_decision_carries_its_reason():
    """"Decided against" with no reason is indistinguishable from "forgotten", and the whole
    point of recording it is that it is not re-litigated by the next person."""
    for row_id, row in _tracker_rows("Open — decided against"):
        reason = row.split("|")[4].strip()
        assert len(reason) > 40, f"{row_id} rejects something without saying why: {reason!r}"


def test_an_accepted_limit_says_why_it_stays():
    for row_id, row in _tracker_rows("Accepted limits"):
        reason = row.split("|")[3].strip()
        assert len(reason) > 40, f"{row_id} states a limit without justifying it"


def test_row_ids_are_unique():
    for section, prefix in (("Open — deferred", "D"), ("Open — decided against", "X"),
                            ("Accepted limits", "L"), ("Closed", "C")):
        ids = [row_id for row_id, _ in _tracker_rows(section)]
        assert len(ids) == len(set(ids)), f"duplicate ids in {section}: {ids}"
        assert all(i.startswith(prefix) for i in ids), (
            f"{section} should use the {prefix} prefix: {ids}"
        )


def test_the_plan_and_the_tracker_agree_on_what_is_done():
    """Sanity on the other direction: the plan's done-count is what the tracker's staleness
    check runs against, so a malformed plan line would silently weaken every test above."""
    items = _plan_items()

    assert len(items) > 40, f"only parsed {len(items)} plan items — has the format changed?"
    assert any(items.values()), "no completed items parsed"
    assert not all(items.values()), "no outstanding items parsed"
