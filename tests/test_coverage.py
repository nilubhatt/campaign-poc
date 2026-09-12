"""
Idea E: "Show coverage, not just a list."

    "`list_campaigns` returns eight rows. The question a marketer actually has is where the
    holes are: which markets, collections and launch types are represented, which have
    outcomes, which have only one example carrying all the weight.

    Fix: a coverage view — market by collection by stage, with counts and an
    evidence-quality marker per cell."

Three questions, and the third is the one nothing else in the product answers: **which have
only one example carrying all the weight**. A cell with a single campaign in it produces
judgments that are really one campaign's opinion, and until now nothing said so — the
similarity score looked identical whether it came from one precedent or nine.

Cells OVERLAP by construction: a campaign that ran in three markets belongs to three cells,
because the question "what do I have in Colombia" is asked per market. So the cell counts
deliberately do not sum to the number of campaigns, and the report says so rather than
letting somebody add them up.

Distinct from `gaps()` (§5.3), which is "what do I fix first" — one ranked list with actions.
This is "what do I have", a matrix to read. §5.3's market gap is now computed from these
cells (tracker D55), so the two cannot disagree about the same library.
"""
import pytest

import core
import store


def _campaign(conn, title, *, outcomes=False, **kw):
    kw.setdefault("detail", f"the {title} brief")
    kw.setdefault("status", "concluded")
    cid = core.ingest_campaign(conn, title=title, confirm=True, **kw)["campaign_id"]
    if outcomes:
        store.add_metrics(conn, cid, metric_type="actual", detail="CTR 1.2%")
    return cid


# ── the three questions the review asks ─────────────────────────────────────

def test_what_is_represented_comes_back_as_cells_not_rows(conn):
    """The first question. `list_campaigns` answers "what have I got" one record at a time;
    this answers it by market, collection and stage at once."""
    _campaign(conn, "Bogota launch", market="LATAM", collection="Q4 launch")
    _campaign(conn, "Jakarta launch", market="SEA", collection="Q4 launch")

    report = core.coverage(conn)

    cells = {(c["market"], c["collection"], c["stage"]) for c in report["cells"]}
    assert ("LATAM", "Q4 launch", "concluded") in cells
    assert ("SEA", "Q4 launch", "concluded") in cells
    assert report["markets"] == ["LATAM", "SEA"]
    assert report["collections"] == ["Q4 launch"]


def test_which_have_outcomes_is_marked_per_cell(conn):
    """The second question. A cell with campaigns and no measured results produces judgments
    that compare a proposal against what was planned, never against what happened."""
    _campaign(conn, "Bogota", market="LATAM", outcomes=True)
    _campaign(conn, "Lima", market="LATAM", outcomes=True)
    _campaign(conn, "Jakarta", market="SEA")

    by_market = {c["market"]: c for c in core.coverage(conn)["cells"]}

    assert by_market["LATAM"]["with_outcomes"] == 2
    assert by_market["SEA"]["with_outcomes"] == 0
    assert by_market["SEA"]["evidence"] == "no_outcomes"


def test_a_cell_carried_by_one_campaign_says_so(conn):
    """The third question, and the one nothing else answers: "which have only one example
    carrying all the weight". Every judgment about that market is really one campaign's
    opinion, and the similarity score looks identical whether it came from one precedent or
    nine."""
    _campaign(conn, "Bogota", market="LATAM", outcomes=True)
    _campaign(conn, "Jakarta", market="SEA", outcomes=True)
    _campaign(conn, "Manila", market="SEA", outcomes=True)

    by_market = {c["market"]: c for c in core.coverage(conn)["cells"]}

    assert by_market["LATAM"]["evidence"] == "single_example"
    assert by_market["SEA"]["evidence"] == "measured"


def test_a_cell_with_one_campaign_and_no_outcomes_reports_the_worse_of_the_two(conn):
    """Both are true; the marker names the one that costs more. Nothing is hidden — the
    counts are there to be read."""
    _campaign(conn, "Bogota", market="LATAM")
    _campaign(conn, "Jakarta", market="SEA", outcomes=True)

    latam = next(c for c in core.coverage(conn)["cells"] if c["market"] == "LATAM")

    assert latam["evidence"] == "no_outcomes"
    assert latam["campaigns"] == 1
    assert latam["with_outcomes"] == 0


# ── how the library is actually shaped ──────────────────────────────────────

def test_a_campaign_that_ran_in_several_markets_is_in_several_cells(conn):
    """The `markets` list is the only way to express a multi-country activation, and a
    coverage view that ignored it would report a LATAM campaign as covering nothing."""
    _campaign(conn, "SEA rollout", markets=["Malaysia", "Singapore"], outcomes=True)

    report = core.coverage(conn)

    assert report["markets"] == ["Malaysia", "Singapore"]
    assert all(c["campaigns"] == 1 for c in report["cells"])


def test_the_report_says_its_cells_overlap(conn):
    """Otherwise somebody adds the counts up. One campaign in three markets is three cells
    of one, not three campaigns."""
    _campaign(conn, "SEA rollout", markets=["Malaysia", "Singapore", "Thailand"])

    report = core.coverage(conn)

    assert sum(c["campaigns"] for c in report["cells"]) == 3
    assert report["campaigns_total"] == 1
    assert "overlap" in report["note"].lower() or "more than one" in report["note"].lower()


def test_a_campaign_with_no_market_is_not_silently_dropped(conn):
    """Most of a young library has no market set. Leaving those records out of the coverage
    view would describe a library nobody has."""
    _campaign(conn, "Global brand film", outcomes=True)

    report = core.coverage(conn)

    unplaced = next(c for c in report["cells"] if c["market"] is None)
    assert unplaced["campaigns"] == 1


def test_superseded_and_reference_records_are_not_coverage(conn):
    """A superseded record is excluded from every search, so counting it as coverage
    describes evidence no judgment can reach — the same correction §5.3 needed."""
    old = _campaign(conn, "Bogota v1", market="LATAM", outcomes=True)
    store.insert_campaign(conn, title="Bogota v2", status="concluded", supersedes=old)
    core.ingest_campaign(conn, title="The rulebook", detail="guidelines",
                         record_type="reference", confirm=True)

    report = core.coverage(conn)

    assert report["campaigns_total"] == 1
    assert all(c["evidence"] != "measured" for c in report["cells"])


def test_stage_is_the_lifecycle_the_record_is_in(conn):
    _campaign(conn, "Next quarter", market="LATAM", status="proposed")
    _campaign(conn, "Bogota", market="LATAM", status="concluded", outcomes=True)

    stages = {c["stage"] for c in core.coverage(conn)["cells"]}

    assert stages == {"proposed", "concluded"}


# ── restraint ───────────────────────────────────────────────────────────────

def test_the_thin_cells_are_called_out_so_the_matrix_does_not_have_to_be_read(conn):
    """A matrix is something to browse; the answer is which cells are weak. Ordered worst
    first, so the first line is the one that matters."""
    for i in range(3):
        _campaign(conn, f"SEA {i}", market="SEA", outcomes=True)
    _campaign(conn, "Bogota", market="LATAM")
    _campaign(conn, "Cairo", market="MENA", outcomes=True)

    report = core.coverage(conn)

    assert [t["market"] for t in report["thin"]] == ["LATAM", "MENA"]
    assert report["thin"][0]["evidence"] == "no_outcomes"


def test_a_wide_library_returns_a_bounded_report(conn):
    """Market × collection × stage is multiplicative, and this goes inside a tool result
    somebody has to read."""
    for i in range(40):
        _campaign(conn, f"Campaign {i}", market=f"Market {i:02d}",
                  collection=f"Collection {i:02d}")

    report = core.coverage(conn)

    assert len(report["cells"]) <= core.MAX_COVERAGE_CELLS
    assert report["cells_total"] == 40
    assert len(report["thin"]) <= core.MAX_COVERAGE_CELLS


def test_an_empty_library_says_so_rather_than_returning_an_empty_matrix(conn):
    report = core.coverage(conn)

    assert report["cells"] == []
    assert report["campaigns_total"] == 0
    assert report["next_actions"], "a new library has one thing to do"


# ── the two views must not disagree ─────────────────────────────────────────

def test_the_gap_about_markets_is_computed_from_these_cells(conn):
    """Tracker D55. `gaps()` grew its own market-or-region grouping, so the two surfaces
    could describe the same library differently — and one of them ignored the `markets`
    list entirely."""
    _campaign(conn, "SEA rollout", markets=["Malaysia", "Singapore"])
    _campaign(conn, "Bogota", market="LATAM", outcomes=True)

    gap = next(g for g in core.gaps(conn)["gaps"]
               if g["code"] == "market_without_outcomes")
    thin = {t["market"] for t in core.coverage(conn)["thin"]
            if t["evidence"] == "no_outcomes"}

    assert set(gap["counts"]["markets"]) <= thin
    assert "Malaysia" in thin and "Singapore" in thin
