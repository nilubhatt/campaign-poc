"""
Idea C: "Ask for what is missing, instead of waiting to be asked."

    "The library knows it holds one campaign with real outcome data. It knows no LATAM store
    launch has ever carried a budget. It knows the KPI workbook named in its own rubric has
    never been supplied. It says none of this unless directly interrogated.

    Fix: a `gaps()` call, and a standing line on every evaluation naming the single most
    valuable missing input for that judgment — 'this verdict rests on zero verified
    outcomes; the Q2 results workbook would change that'."

Two halves, and they answer different questions.

`gaps()` is about the LIBRARY: what is missing across everything, ranked, so somebody can go
and fix the biggest hole. The standing line is about ONE judgment: of everything missing,
the single thing that would most change *this* verdict — which is usually not the library's
biggest hole at all.

The word "single" is the design. A list of everything absent is a list nobody reads, and the
review's own example names one thing and says what supplying it would do.

Kept deliberately distinct from §6.6, the evidence-strength line. That says what a judgment
RESTS ON — how many precedents, how many concluded, top similarity. This says what would
most improve it. One describes, the other asks.
"""
import pytest

import core
import store


def _concluded(conn, title, **kw):
    """Through the real ingest path, so the record is embedded and findable — a campaign
    inserted straight into the table is invisible to every search."""
    kw.setdefault("detail", f"a campaign called {title}")
    return core.ingest_campaign(conn, title=title, status="concluded", confirm=True,
                                **kw)["campaign_id"]


def _with_results(conn, title, **kw):
    cid = _concluded(conn, title, **kw)
    store.add_metrics(conn, cid, metric_type="actual", detail="CTR 1.2%")
    return cid


# ── the library view ────────────────────────────────────────────────────────

def test_a_library_with_one_measured_campaign_says_so(conn):
    """The review's first example, verbatim: "It knows it holds one campaign with real
    outcome data." Everything a judgment cites rests on what was actually measured, so this
    is the number that decides how much any verdict is worth."""
    _with_results(conn, "Colombia")
    for title in ("Peru", "Mexico", "Chile"):
        _concluded(conn, title)

    report = core.gaps(conn)

    outcomes = next(g for g in report["gaps"] if g["code"] == "few_verified_outcomes")
    assert outcomes["counts"] == {"campaigns": 4, "with_outcomes": 1}


def test_a_library_where_everything_is_measured_does_not_raise_it(conn):
    _with_results(conn, "Colombia")

    assert all(g["code"] != "few_verified_outcomes" for g in core.gaps(conn)["gaps"])


def test_a_market_with_no_measured_campaign_at_all_is_named(conn):
    """A gap located in a *slice* of the library rather than in its total: a marketer asking
    about Colombia does not care that the library is 60% measured overall if the LATAM part
    is 0%.

    NOT the review's budget example, which this docstring used to claim. "No LATAM store
    launch has ever carried a budget" is about a missing FIELD within records; nothing
    detects a budget, and that is D49."""
    _with_results(conn, "Jakarta launch", market="SEA")
    _concluded(conn, "Bogota launch", market="LATAM")
    _concluded(conn, "Lima launch", market="LATAM")

    report = core.gaps(conn)

    unmeasured = next(g for g in report["gaps"] if g["code"] == "market_without_outcomes")
    assert "LATAM" in unmeasured["what"]
    assert "SEA" not in unmeasured["what"]


def test_records_that_are_stored_but_not_searchable_are_a_gap(conn, monkeypatch):
    """Not "missing input" in the review's sense, but the same class of thing: the library
    holds it and cannot use it, and says nothing."""
    import config

    monkeypatch.setattr(config, "TOOL_TIME_BUDGET_SECONDS", 0.0)
    deck = "\n\n".join(f"section {i} " + "word " * 200 for i in range(4))
    core.ingest_campaign(conn, title="Cut short", deck_text=deck, confirm=True)

    report = core.gaps(conn)

    assert any(g["code"] == "partly_indexed" for g in report["gaps"])


def test_every_gap_says_what_would_close_it(conn):
    """A gap with no action is a complaint. The review's own phrasing — "the Q2 results
    workbook would change that" — names the thing that closes it."""
    _concluded(conn, "Colombia")
    _concluded(conn, "Peru")

    for gap in core.gaps(conn)["gaps"]:
        assert gap["what"], gap
        assert gap["why_it_matters"], gap
        assert gap["next_actions"] or gap.get("closed_by"), (
            f"{gap['code']} names nothing that would close it"
        )


def test_the_gaps_are_ranked_so_the_first_one_is_the_one_to_fix(conn):
    """"Ranked" is the whole value. An unordered list of everything absent is the thing
    nobody reads — which is the state the review was describing."""
    import config

    _concluded(conn, "Colombia")
    report = core.gaps(conn)

    assert report["gaps"] == sorted(report["gaps"], key=lambda g: g["rank"])
    assert report["most_valuable"] == report["gaps"][0]["code"]


def test_an_empty_library_says_it_is_empty_rather_than_listing_holes(conn):
    """Every gap is present in an empty library, which makes the list useless. A new install
    has one thing to do and it is not "fix your LATAM coverage"."""
    report = core.gaps(conn)

    assert report["most_valuable"] == "library_is_empty"
    assert len(report["gaps"]) == 1


def test_a_complete_library_says_nothing_is_missing(conn):
    _with_results(conn, "Colombia", market="LATAM")

    report = core.gaps(conn)

    assert report["gaps"] == []
    assert report["most_valuable"] is None


# ── the standing line on a judgment ─────────────────────────────────────────

def test_a_judgment_says_what_would_most_change_it(conn):
    """The review's own sentence: "this verdict rests on zero verified outcomes; the Q2
    results workbook would change that"."""
    _concluded(conn, "Peru seeding", detail="a seeding campaign in Peru")
    _concluded(conn, "Chile seeding", detail="a seeding campaign in Chile")

    packaged = core.prepare_evaluation(conn, subject_title="Colombia seeding",
                                       proposal_text="a seeding campaign in Colombia")

    missing = packaged["most_valuable_missing_input"]
    assert missing["code"] == "no_measured_precedent"
    assert "outcome" in missing["what"].lower() or "result" in missing["what"].lower()
    assert missing["next_actions"]


def test_the_line_is_about_the_evidence_cited_not_the_library(conn):
    """A library that is 90% measured can still produce a judgment resting entirely on the
    unmeasured 10%. The line is about THIS verdict."""
    for i in range(9):
        _with_results(conn, f"Measured {i}", detail="soap opera product placement")
    _concluded(conn, "Unmeasured seeding", detail="influencer seeding in Colombia")

    packaged = core.prepare_evaluation(conn, subject_title="New seeding",
                                       proposal_text="influencer seeding in Colombia",
                                       top_k=1)

    assert packaged["most_valuable_missing_input"]["code"] == "no_measured_precedent"


def test_a_judgment_on_measured_precedent_asks_for_nothing(conn):
    """Silence is the right answer when nothing is missing, and it is what makes the line
    worth reading when it appears."""
    _with_results(conn, "Peru seeding", detail="influencer seeding in Peru")

    packaged = core.prepare_evaluation(conn, subject_title="Colombia seeding",
                                       proposal_text="influencer seeding in Peru")

    assert packaged["most_valuable_missing_input"] is None


def test_a_judgment_with_no_precedent_at_all_says_that_first(conn):
    """Worse than unmeasured precedent, and a different ask: there is nothing to compare
    against, so the verdict is an opinion rather than a judgment from the record."""
    packaged = core.prepare_evaluation(conn, subject_title="Colombia seeding",
                                       proposal_text="influencer seeding in Colombia")

    assert packaged["most_valuable_missing_input"]["code"] == "no_precedent"


def test_the_missing_input_is_one_thing_not_a_list(conn):
    """"the single most valuable missing input". Several things are always missing; naming
    them all is the behaviour this replaces."""
    _concluded(conn, "Peru", detail="seeding")

    packaged = core.prepare_evaluation(conn, subject_title="Colombia",
                                       proposal_text="seeding")

    assert isinstance(packaged["most_valuable_missing_input"], dict)


# ── what the tracker was carrying ───────────────────────────────────────────

def test_a_deck_whose_comments_were_never_read_is_recorded_for_later(conn):
    """Tracker D17. Warnings are ephemeral — they exist for the length of one response — so
    "this deck's comments were never read" was unrecoverable the moment the upload returned.

    Recorded, but deliberately not REPORTED as a gap: see
    test_a_gap_is_not_reported_when_nothing_can_close_it. The fact is what D17 was about;
    reporting it needs D39 first."""
    created = core.ingest_campaign(conn, title="Pasted in", deck_text="a brief, as text",
                                   confirm=True)

    assert store.get_campaign(conn, created["campaign_id"])["commentary_checked"] == 0


def test_the_incompleteness_warning_offers_the_tool_behind_it(conn, monkeypatch):
    """Tracker D45. `results_may_be_incomplete` carried `finish_indexing` as prose while a
    single tool sat behind it, and `prepare_evaluation` returned no actions at all."""
    import config

    monkeypatch.setattr(config, "TOOL_TIME_BUDGET_SECONDS", 0.0)
    core.ingest_campaign(conn, title="Cut short", deck_text="\n\n".join(
        f"section {i} " + "word " * 200 for i in range(4)), confirm=True)

    packaged = core.prepare_evaluation(conn, subject_title="X", proposal_text="section")
    warning = next(w for w in packaged["warnings"]
                   if w["code"] == "results_may_be_incomplete")

    assert warning["next_actions"][0]["tool"] == "finish_indexing"
    assert warning["next_actions"][0]["consent"] == "do"


# ══ design review of 5.3 ═════════════════════════════════════════════════════

def test_a_forecast_is_not_a_measured_outcome(conn):
    """`has_metrics` counts any metric row, so adding a `predicted` figure silenced both the
    library gap and the judgment line. The review says "verified outcomes"; a forecast is the
    opposite of one — it is the thing reconciliation later scores AGAINST the actuals."""
    cid = _concluded(conn, "Colombia")
    store.add_metrics(conn, cid, metric_type="predicted", detail="expect CTR 2%")

    report = core.gaps(conn)

    assert any(g["code"] == "few_verified_outcomes" for g in report["gaps"])


def test_a_campaign_that_has_not_run_is_not_missing_its_results(conn):
    """A proposed campaign has no outcomes by definition. Counting it as a gap, and
    prefilling it into "record what this achieved", is a request nobody can satisfy — the
    permanent-complaint failure, on the highest-ranked gap. `after_upload` already gates on
    `concluded`; this did not."""
    core.ingest_campaign(conn, title="Next quarter", detail="a proposal", status="proposed",
                         confirm=True)
    _with_results(conn, "Colombia")

    report = core.gaps(conn)

    assert all(g["code"] != "few_verified_outcomes" for g in report["gaps"]), report["gaps"]


def test_the_market_offered_first_is_the_one_that_matters_most(conn):
    """`barren[0]` was alphabetical, so the single place magnitude decided anything decided
    it by the alphabet: Andorra with one campaign offered ahead of LATAM with twenty."""
    _with_results(conn, "Jakarta", market="SEA")
    _concluded(conn, "Andorra one", market="Andorra")
    for i in range(4):
        _concluded(conn, f"LATAM {i}", market="LATAM")

    gap = next(g for g in core.gaps(conn)["gaps"] if g["code"] == "market_without_outcomes")

    assert "LATAM" in gap["next_actions"][0]["label"]


def test_a_gap_is_not_reported_when_nothing_can_close_it(conn):
    """`commentary_never_read` offered `upload_campaign` — which creates a SECOND record and
    fires `duplicate_title`. Item 5.2 refused exactly this offer, in writing, one commit
    earlier: no tool attaches a deck to an existing record (D39). A gap whose only action
    makes things worse is a complaint, which this item's own rule forbids."""
    core.ingest_campaign(conn, title="Pasted in", deck_text="a brief, as text", confirm=True)

    report = core.gaps(conn)

    assert all(g["code"] != "commentary_never_read" for g in report["gaps"])


def test_the_fact_is_still_recorded_even_though_it_is_not_reported(conn):
    """The column stays — D17's reasoning was right, and the gap can be reported the moment
    D39 gives it an action that works."""
    created = core.ingest_campaign(conn, title="Pasted in", deck_text="text", confirm=True)

    assert store.get_campaign(conn, created["campaign_id"])["commentary_checked"] == 0


def test_an_upgraded_library_does_not_invent_gaps_for_records_it_already_read(conn):
    """The migration adds `commentary_checked` with DEFAULT 0, so every record whose file WAS
    read before this column existed reads as never-read. The backfill is derivable: a record
    with commentary chunks was, by definition, read."""
    import sqlite3

    cid = store.insert_campaign(conn, title="Read before the column existed",
                                deck_text="body")
    store.insert_chunks(conn, cid, ["a speaker note"], kind="commentary",
                        sources=[{"kind": "speaker_note", "anchor": "slide 1"}])
    conn.execute("UPDATE campaigns SET commentary_checked = 0 WHERE id = ?", (cid,))
    conn.commit()

    store._migrate_schema(conn)

    assert store.get_campaign(conn, cid)["commentary_checked"] == 1


def test_the_verdict_itself_says_what_would_most_change_it(conn):
    """The line was only on `prepare_evaluation` — two calls before the verdict the user
    actually hears, and this project's own stated principle is that models mirror the shape
    of a tool result far more reliably than they follow instructions inside one. So the line
    delivered earlier, with a note asking for it to be repeated later, is the thing that gets
    dropped."""
    cid = _concluded(conn, "Peru seeding", detail="influencer seeding")

    result = core.save_evaluation(
        conn, subject_title="Colombia seeding", verdict="revise",
        summary="Nothing is dated.", cited_ids=[cid],
        findings=[{"severity": "blocking", "finding": "No dates"}])

    missing = result["most_valuable_missing_input"]
    assert missing["code"] == "no_measured_precedent"


def test_the_line_is_computed_by_the_server_not_accepted_from_the_model(conn):
    """Like `evidence` and `provenance`: a fact about what the library holds is the server's
    to state. A model asserting "nothing is missing" would be asserting it about records it
    cannot see."""
    cid = _with_results(conn, "Peru seeding", detail="influencer seeding")

    result = core.save_evaluation(
        conn, subject_title="Colombia", verdict="approve", findings=[],
        summary="Matches a measured precedent.", cited_ids=[cid])

    assert result["most_valuable_missing_input"] is None


def test_the_line_survives_to_be_read_back_later(conn):
    """`get_evaluation` and the §7.6 stamp both need it, and neither can recover something
    that was never stored."""
    cid = _concluded(conn, "Peru seeding", detail="influencer seeding")

    saved = core.save_evaluation(
        conn, subject_title="Colombia", verdict="revise", summary="Nothing is dated.",
        cited_ids=[cid], findings=[{"severity": "blocking", "finding": "No dates"}])

    stored = store.get_evaluation(conn, saved["evaluation_id"])
    assert stored["evidence"]["most_valuable_missing_input"]["code"] == "no_measured_precedent"


# ══ adversarial review of 5.3 ════════════════════════════════════════════════

def test_the_reviews_own_example_works_in_a_library_of_one_market(conn):
    """`len(barren) < len(by_market)` suppressed the gap whenever EVERY market was barren —
    so "no LATAM store launch has ever carried a budget", in a library that holds only LATAM
    campaigns, said nothing at all. The guard was meant to avoid noise and instead silenced
    the review's own example."""
    _concluded(conn, "Bogota launch", market="LATAM")
    _concluded(conn, "Lima launch", market="LATAM")

    codes = [g["code"] for g in core.gaps(conn)["gaps"]]

    assert "market_without_outcomes" in codes


def test_a_forecast_only_precedent_is_missing_its_outcomes_on_the_judgment_path_too(conn):
    """The fix round filtered `save_evaluation` and left `prepare_evaluation` reading the
    raw evidence, where `find_similar` includes predicted rows in `metrics`. So the earlier
    of the two calls still reported a judgment resting entirely on a forecast as complete."""
    cid = _concluded(conn, "Peru seeding", detail="influencer seeding")
    store.add_metrics(conn, cid, metric_type="predicted", detail="expect CTR 2%")

    packaged = core.prepare_evaluation(conn, subject_title="Colombia",
                                       proposal_text="influencer seeding")

    assert packaged["most_valuable_missing_input"]["code"] == "no_measured_precedent"


def test_markets_are_grouped_the_way_the_rest_of_the_product_groups_them(conn):
    """Three disagreements with the product's own semantics, all reproduced. A campaign
    reached only through the `markets` list was invisible to this gap. A whitespace market
    was reported as a market — "No campaign in    , NA has measured results." And `LATAM`
    was called barren while `latam` had results, though `filter_campaign_ids` matches them
    case-insensitively and returns both."""
    _concluded(conn, "Manila", markets=["Philippines"])
    _concluded(conn, "Bogota", market="LATAM")
    measured = _concluded(conn, "Lima", market="latam")
    store.add_metrics(conn, measured, metric_type="actual", detail="CTR 1%")
    _concluded(conn, "Nowhere", market="   ")

    gap = next(g for g in core.gaps(conn)["gaps"] if g["code"] == "market_without_outcomes")

    assert "Philippines" in gap["what"], "a markets-list campaign is still in a market"
    assert "LATAM" not in gap["what"].upper().replace("PHILIPPINES", ""), \
        "latam has measured results, and the product treats the two spellings as one"
    assert "   " not in gap["what"]


def test_a_superseded_record_is_not_counted_as_library_evidence(conn):
    """A measured v1 that v2 replaces can never be cited — `find_similar` excludes it — so
    counting it as one of the library's measured campaigns describes evidence no judgment
    can reach."""
    v1 = _with_results(conn, "Colombia v1")
    store.insert_campaign(conn, title="Colombia v2", status="concluded", supersedes=v1)

    report = core.gaps(conn)

    outcomes = next(g for g in report["gaps"] if g["code"] == "few_verified_outcomes")
    assert outcomes["counts"] == {"campaigns": 1, "with_outcomes": 0}


def test_a_wide_library_does_not_produce_a_paragraph(conn):
    """35 markets produced a 3,185-character sentence and an uncapped list, inside a tool
    result somebody has to read."""
    for i in range(35):
        _concluded(conn, f"Campaign {i}", market=f"Market {i:02d}")
    measured = _concluded(conn, "Measured", market="Measured market")
    store.add_metrics(conn, measured, metric_type="actual", detail="CTR 1%")

    gap = next(g for g in core.gaps(conn)["gaps"] if g["code"] == "market_without_outcomes")

    assert len(gap["what"]) < 300, len(gap["what"])
    assert len(gap["counts"]["markets"]) <= 5


def test_two_gaps_do_not_send_the_user_to_the_same_campaign_twice(conn):
    """The oldest unmeasured campaign is usually also the barren market's first, so accepting
    both offers appended two `actual` rows to the same record."""
    _with_results(conn, "Jakarta", market="SEA")
    _concluded(conn, "Bogota", market="LATAM")

    report = core.gaps(conn)
    targets = [a["prefilled_args"].get("campaign_id")
               for g in report["gaps"] for a in g["next_actions"]
               if a["tool"] == "add_metrics"]

    assert len(targets) == len(set(targets)), targets
    # And the gap that lost its offer says which one closes it, rather than going silent.
    market = next(g for g in report["gaps"] if g["code"] == "market_without_outcomes")
    assert market["closed_by"] == "few_verified_outcomes"


def test_the_ranking_is_asserted_against_a_library_that_has_several_gaps(conn, monkeypatch):
    """Mutation-proof: reversing the sort left the suite green, because the ranking test
    built a library with exactly one gap in it."""
    import config

    monkeypatch.setattr(config, "TOOL_TIME_BUDGET_SECONDS", 0.0)
    core.ingest_campaign(conn, title="Cut short", status="concluded", deck_text="\n\n".join(
        f"section {i} " + "word " * 200 for i in range(4)), confirm=True)
    _concluded(conn, "Bogota", market="LATAM")

    report = core.gaps(conn)

    assert len(report["gaps"]) >= 3, [g["code"] for g in report["gaps"]]
    assert [g["rank"] for g in report["gaps"]] == sorted(g["rank"] for g in report["gaps"])
    assert report["gaps"][0]["code"] == "few_verified_outcomes"


def test_an_empty_library_reports_only_that_even_when_other_branches_would_fire(conn,
                                                                               monkeypatch):
    """Mutation-proof: deleting the early return left the suite green, because on an empty
    library every later branch is naturally silent anyway. This makes the early return carry
    its own weight."""
    import config

    monkeypatch.setattr(config, "TOOL_TIME_BUDGET_SECONDS", 0.0)
    core.ingest_campaign(conn, title="Reference only", record_type="reference",
                         deck_text="\n\n".join(f"section {i} " + "word " * 200
                                               for i in range(4)), confirm=True)

    report = core.gaps(conn)

    assert [g["code"] for g in report["gaps"]] == ["library_is_empty"], report["gaps"]


def test_every_gap_code_is_reached_by_the_test_that_checks_them_all(conn, monkeypatch):
    """`test_every_gap_says_what_would_close_it` was reaching one code of three."""
    import config

    monkeypatch.setattr(config, "TOOL_TIME_BUDGET_SECONDS", 0.0)
    core.ingest_campaign(conn, title="Cut short", status="concluded", market="LATAM",
                         deck_text="\n\n".join(f"section {i} " + "word " * 200
                                               for i in range(4)), confirm=True)
    measured = _concluded(conn, "Jakarta", market="SEA")
    store.add_metrics(conn, measured, metric_type="actual", detail="CTR 1%")

    codes = {g["code"] for g in core.gaps(conn)["gaps"]}

    assert codes == {"few_verified_outcomes", "market_without_outcomes", "partly_indexed"}
