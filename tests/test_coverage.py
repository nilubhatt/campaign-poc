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


# ══ review of 5.5 ════════════════════════════════════════════════════════════

def test_one_market_spelled_two_ways_is_one_cell(conn):
    """The 5.3 bug, one file over, and it made this item's own claim false. `_markets_of`
    folded case WITHIN a record while the bucket key used the raw spelling, so two records
    spelled differently became two cells — one reading `no_outcomes` and the other
    `single_example`, for a library that search treats as a single market of two campaigns
    with one measured. `gaps()` folded correctly and reported no market gap at all, so the
    two surfaces contradicted each other on the same library."""
    a = _campaign(conn, "Bogota", market="LATAM", outcomes=True)
    _campaign(conn, "Lima", market="latam")

    report = core.coverage(conn)

    assert report["markets"] == ["LATAM"], report["markets"]
    cell = next(c for c in report["cells"] if c["market"] == "LATAM")
    assert cell["campaigns"] == 2 and cell["with_outcomes"] == 1
    assert len(store.filter_campaign_ids(conn, market="latam")) == cell["campaigns"]


def test_one_collection_spelled_two_ways_is_one_cell(conn):
    """`filter_campaign_ids` matches `LOWER(collection)`, so one collection became two cells
    and the headline marker was wrong because of a capital letter."""
    _campaign(conn, "Bogota", market="LATAM", collection="Q4 launch", outcomes=True)
    _campaign(conn, "Lima", market="LATAM", collection="q4 launch", outcomes=True)

    report = core.coverage(conn)

    assert report["collections"] == ["Q4 launch"]
    assert len(report["cells"]) == 1


def test_the_two_surfaces_agree_about_a_case_split_library(conn):
    """What C16 claimed and did not deliver."""
    _campaign(conn, "Bogota", market="LATAM")
    _campaign(conn, "Lima", market="latam")

    thin = {t["market"] for t in core.coverage(conn)["thin"]}
    gap = [g for g in core.gaps(conn)["gaps"] if g["code"] == "market_without_outcomes"]

    assert thin == {"LATAM"}
    assert gap and set(gap[0]["counts"]["markets"]) == {"LATAM"}


def test_a_forecast_is_not_an_outcome_here_either(conn):
    """Mutation-proof: swapping `campaigns_with_actual_metrics` for any-metric-row left the
    suite green. It is the §5.3 bug, and nothing pinned it on this surface."""
    cid = _campaign(conn, "Bogota", market="LATAM")
    store.add_metrics(conn, cid, metric_type="predicted", detail="expect CTR 2%")

    cell = next(c for c in core.coverage(conn)["cells"] if c["market"] == "LATAM")

    assert cell["with_outcomes"] == 0
    assert cell["evidence"] == "no_outcomes"


def test_a_campaign_that_has_not_run_is_not_missing_its_outcomes(conn):
    """§5.3 wrote this lesson out and this surface repeated it: a proposed campaign cannot
    have results, so marking its cell `no_outcomes` is a complaint nobody can answer."""
    _campaign(conn, "Next quarter", market="LATAM", status="proposed")

    cell = next(c for c in core.coverage(conn)["cells"] if c["market"] == "LATAM")

    assert cell["evidence"] == "not_yet_run"
    assert cell not in core.coverage(conn)["thin"]


def test_a_placeholder_is_not_coverage(conn):
    """A `stub` is "a placeholder record" by the product's own definition, so counting it as
    evidence a judgment can lean on describes content that is not there — and it arrived with
    a null stage that nothing explained."""
    core.ingest_campaign(conn, title="Placeholder", detail="to fill in later",
                         record_type="stub", confirm=True)
    _campaign(conn, "Bogota", market="LATAM", outcomes=True)

    report = core.coverage(conn)

    assert report["campaigns_total"] == 1
    assert all(c["stage"] is not None for c in report["cells"])


def test_a_library_where_everything_is_thin_gets_one_line_not_the_matrix_twice(conn):
    """With 25 thin cells, `cells` and `thin` came back byte-identical: no thick cell was
    shown, ten cells were hidden with no count of what, and `single_example` — this item's
    own headline — never appeared because `no_outcomes` filled the list. "The matrix is the
    evidence" failed exactly where it was needed."""
    for i in range(30):
        _campaign(conn, f"Unmeasured {i}", market=f"Market {i:02d}")
    for i in range(5):
        _campaign(conn, f"Measured {i}", market=f"Strong {i}", outcomes=True)

    report = core.coverage(conn)

    assert sum(report["hidden"].values()) == report["cells_total"] - len(report["cells"])
    # The list can lose detail; the SHAPE cannot depend on where the cut fell. A report
    # showing 25 weak cells out of 35 said nothing at all about the 5 measured ones.
    assert report["evidence_summary"]["measured"] == 0
    assert report["evidence_summary"]["single_example"] == 5
    assert report["evidence_summary"]["no_outcomes"] == 30
    assert report["thin_total"] == 35 and len(report["thin"]) == 25


def test_the_cells_are_ordered_for_browsing_and_thin_carries_the_ranking(conn):
    """Mutation-proof: two mutations of the sort survived, because the only ordering test
    used two cells whose alphabetical order happened to agree with their ranking."""
    _campaign(conn, "Zulu", market="ZZ", outcomes=True)
    _campaign(conn, "Zulu 2", market="ZZ", outcomes=True)
    _campaign(conn, "Alpha", market="AA")

    report = core.coverage(conn)

    assert [c["market"] for c in report["cells"]] == ["AA", "ZZ"], "browsing order"
    assert [t["market"] for t in report["thin"]] == ["AA"], "ranked, worst first"


def test_a_region_counts_as_a_market_when_nothing_else_does(conn):
    """Mutation-proof: ignoring `region` in `_markets_of` left the suite green, though
    `region` is how a single-country activation is recorded."""
    _campaign(conn, "Jakarta", region="Indonesia", outcomes=True)

    assert core.coverage(conn)["markets"] == ["Indonesia"]


def test_an_offer_that_cannot_be_accepted_as_it_stands_says_how_many_things_it_needs(conn):
    """Accepting it literally — calling with only the prefilled arguments — raised
    `TypeError: missing 'title'`, and `needs` named only the deck. §5.2's promise is that
    accepting is one step; where it is not, `needs` has to account for every required
    argument that is not prefilled.

    Asserted structurally rather than by looking for the word "title" in the prose, which is
    the wording-coupling §5.1 was written about."""
    import inspect

    import mcp_server

    offer = core.coverage(conn)["next_actions"][0]
    signature = inspect.signature(getattr(mcp_server, offer["tool"]))
    required = {name for name, param in signature.parameters.items()
                if param.default is inspect.Parameter.empty}
    unsatisfied = required - set(offer["prefilled_args"])

    assert len(offer.get("needs", [])) >= len(unsatisfied), (
        f"{offer['tool']} still needs {unsatisfied} and `needs` lists "
        f"{offer.get('needs')}"
    )
