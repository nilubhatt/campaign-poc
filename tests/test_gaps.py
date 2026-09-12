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
    """The review's second: "no LATAM store launch has ever carried a budget" — a gap
    located in a *slice* of the library rather than in its total. A marketer asking about
    Colombia does not care that the library is 60% measured overall if the LATAM part is 0%."""
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
        assert gap["next_actions"], f"{gap['code']} names nothing that would close it"


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

def test_a_deck_whose_comments_were_never_read_is_still_reportable_later(conn):
    """Tracker D17. Warnings are ephemeral — they exist for the length of one response — so
    "this deck's comments were never read" was unrecoverable the moment the upload returned,
    and `gaps()` could never report it."""
    core.ingest_campaign(conn, title="Pasted in", deck_text="a brief, as text", confirm=True)

    report = core.gaps(conn)

    assert any(g["code"] == "commentary_never_read" for g in report["gaps"])


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
