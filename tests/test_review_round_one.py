"""
What the design review found after the suite was green at 2032.

Every one of these is a defect the tests could not see, and three of them are the shape this
project keeps rediscovering: **a new thing quietly switching off an old one**, with nothing
that notices because each half works perfectly on its own.

  1  A single pair of similarly-worded corrections silently turned off EVERY proactive queue
     offer in the product — permanently, and invisibly, because the counter's defensive
     `except` turned the crash into "nothing is waiting".
  2  `diff_campaigns` — §5.4's whole reason to exist, and the thing a marketer asks the
     instant v2 lands — crashed BECAUSE the user had answered the question the product asked.
  3  A settled finding became unreachable by `get_evaluation(departure="unexplained")`, the
     only filter that would find it, and the value it was rewritten to could not be passed
     through the MCP schema at all. The product ate the finding.
  4  The "what this library now needs most" offer wrote its own state before reaching
     `trim`, so on any upload with three other things to say it was dropped AND burned: the
     change it was announcing was recorded as already announced.
  5  `answer_gap` — an offer whose effect is "show me less" — was the most repeated
     sentence in a `gaps()` response, and it was offered on the two gaps that are about work
     nobody has done rather than facts that cannot exist.
"""
import pytest

import actions
import core
import corrections
import feedback
import store


def _campaign(conn, title="Colombia", **kw):
    kw.setdefault("market", "LATAM")
    kw.setdefault("status", "concluded")
    kw.setdefault("detail", f"A campaign called {title}, which ran in a market.")
    return core.ingest_campaign(conn, title=title, **kw)["campaign_id"]


def _two_similar_rules(conn, campaign_id):
    corrections.note(conn, text="Never put the logo on a dark background.",
                     provenance="client email, 4 March", campaign_id=campaign_id)
    corrections.note(conn, text="Never place the logo on a dark background.",
                     provenance="client call, 9 March", campaign_id=campaign_id)


def _judged_with_a_question(conn, campaign_id, title="Colombia"):
    return core.save_evaluation(
        conn, subject_title=title, campaign_id=campaign_id, verdict="revise",
        summary="Close to Peru, with one difference nobody has explained.",
        approve_if="The timeline runs eight weeks, or the brief says why six is right.",
        findings=[{
            "severity": "should_fix", "kind": "precedent_departure",
            "departure": "unexplained",
            "finding": "The timeline is six weeks where every comparable ran eight.",
            "fix": "Extend to eight weeks, or say why six is right here.",
            "precedent": {"campaign_id": campaign_id, "quote": "which ran in a market"},
        }])


# ── 1: one rule pair turned off every proactive offer in the product ───────

def test_a_rule_question_does_not_silence_the_waiting_count(conn):
    """`waiting()` filtered on `r["campaign_id"]`, and a rule-question row has no campaign at
    all — so it raised KeyError, the bare `except` returned 0, and "N campaigns are waiting on
    feedback" vanished from every upload, every judgment and every session start.

    Permanently: a library containing two similarly-worded client corrections is the ORDINARY
    state of an agency library, and it is the precondition D110 was written about. 10.6's
    entire headline was switched off by 10.2's own new row."""
    for title in ("Bogota", "Jakarta", "Lima"):
        cid = _campaign(conn, title)
    before = feedback.waiting(conn)
    assert before == 3, "the fixture has to have campaigns actually waiting"

    _two_similar_rules(conn, cid)

    assert feedback.waiting(conn) == before, (
        "one pair of similar corrections turned the whole queue offer off"
    )


def test_the_waiting_count_is_about_campaigns(conn):
    """`offer_the_queue` renders "N campaign(s) are waiting on feedback", and the queue's own
    summary says the same. Counting a rule-vocabulary question as a campaign makes both
    sentences false — a library with three open campaigns reported four."""
    cid = _campaign(conn, "Bogota")
    _two_similar_rules(conn, cid)

    assert feedback.waiting(conn) == 1
    assert "1 campaign" in feedback.queue(conn)["what_it_means"]


def test_the_counter_does_not_swallow_a_real_failure(conn, monkeypatch):
    """The `except` that hid this. An exception handler that turns a crash into "there is
    nothing waiting" is the same defect class as the silent partial re-index this phase just
    fixed: an error the product cannot have, reported as a clean result."""
    def broken(*a, **kw):
        raise RuntimeError("something in the queue is wrong")

    monkeypatch.setattr(feedback, "_open_rows", broken)

    with pytest.raises(RuntimeError):
        feedback.waiting(conn)


# ── 2 and 3: answering the product must not break the product ──────────────

def test_the_comparison_survives_somebody_answering_a_departure(conn):
    """The user says "that was deliberate, the client moved the date"; the answer is recorded;
    v2 is judged and repeats the finding by id; and `diff_campaigns` raised
    `ValueError: tuple.index(x): x not in tuple`.

    `_reread` enumerates `_DEPARTURES` positionally, and `explained` was deliberately not in
    it. So "what changed between v1 and v2" — the thing a marketer asks the instant v2 lands —
    failed because they had answered the question the product asked them."""
    one = _campaign(conn, "Colombia v1")
    judged = _judged_with_a_question(conn, one, "Colombia v1")
    finding_id = judged["findings"][0]["id"]
    core.answer_finding(conn, evaluation_id=judged["evaluation_id"], finding_id=finding_id,
                        answer="deliberate", note="The client moved the launch date.",
                        said_by="R. Vega")
    two = core.ingest_campaign(conn, title="Colombia v2", market="LATAM", status="concluded",
                               detail="The second version, still six weeks.",
                               supersedes=one)["campaign_id"]
    core.save_evaluation(
        conn, subject_title="Colombia v2", campaign_id=two, verdict="revise",
        summary="Same difference, and now it costs something.",
        approve_if="The timeline runs eight weeks.",
        findings=[{"severity": "should_fix", "kind": "precedent_departure",
                   "departure": "regression", "repeats": finding_id,
                   "finding": "Still six weeks, and the market has moved.",
                   "fix": "Extend to eight weeks.",
                   "precedent": {"campaign_id": one, "quote": "which ran in a market"}}])

    diffed = core.diff_campaigns(conn, earlier=one, later=two)

    assert diffed["raised_again"][0]["id"] == finding_id


def test_a_settled_finding_is_still_found_by_the_filter_that_would_find_it(conn):
    """The rewrite made the stored `departure` read `explained` — a value `Departure` cannot
    express, so a model calling `get_evaluation(departure=...)` could not ask for it at all,
    and asking for `unexplained` returned nothing. Somebody looking for the departures nobody
    had explained was shown none of the ones that HAD been, which is the opposite of an
    audit trail."""
    cid = _campaign(conn)
    judged = _judged_with_a_question(conn, cid)
    finding_id = judged["findings"][0]["id"]
    core.answer_finding(conn, evaluation_id=judged["evaluation_id"], finding_id=finding_id,
                        answer="deliberate", note="The client moved the launch date.",
                        said_by="R. Vega")

    found = core.get_evaluation(conn, evaluation_id=judged["evaluation_id"],
                                departure="unexplained")

    assert [f["id"] for f in found["findings"]] == [finding_id]
    assert found["findings"][0]["settled"]["answer"] == "deliberate"


def test_the_stored_departure_keeps_the_word_the_model_wrote(conn):
    """`departure` describes THE DIFFERENCE — regression, unexplained, possible_improvement.
    "Explained" describes the conversation about the difference, which is a different thing
    and belongs in a different field. And `answers` is append-only precisely so that what
    somebody said in March survives being contradicted in June; destructively overwriting the
    derived field at read time, with no record of its prior value, is the opposite
    commitment."""
    cid = _campaign(conn)
    judged = _judged_with_a_question(conn, cid)
    finding_id = judged["findings"][0]["id"]
    core.answer_finding(conn, evaluation_id=judged["evaluation_id"], finding_id=finding_id,
                        answer="deliberate", note="The client moved the launch date.",
                        said_by="R. Vega")

    stored = core.get_evaluation(conn, evaluation_id=judged["evaluation_id"])["findings"][0]

    assert stored["departure"] == "unexplained"
    assert stored["settled"]["answer"] == "deliberate"


def test_a_judgment_reading_the_finding_is_told_it_was_answered(conn):
    """Dropping the rewrite only works if the readers that CARED about `unexplained` learn to
    consult the answer instead. `unexplained` means "the brief does not say whether this is
    deliberate" — an open question — and once somebody has said, it is no longer open. This
    is the predicate that has to move, not the stored value."""
    cid = _campaign(conn)
    judged = _judged_with_a_question(conn, cid)
    finding_id = judged["findings"][0]["id"]
    core.answer_finding(conn, evaluation_id=judged["evaluation_id"], finding_id=finding_id,
                        answer="deliberate", note="The client moved the launch date.",
                        said_by="R. Vega")
    v2 = core.ingest_campaign(conn, title="Colombia v2", market="LATAM", status="proposed",
                              detail="The second version.", supersedes=cid)["campaign_id"]

    prepared = core.prepare_evaluation(conn, subject_title="Colombia v2",
                                       proposal_text="The second version.", campaign_id=v2)

    assert "already been answered" in prepared["note"]


# ── 4: the offer that burned the change it was announcing ──────────────────

def test_the_ranking_offer_is_not_burned_when_it_is_trimmed(conn):
    """`_gap_moved` wrote the new top gap BEFORE the offer reached `trim`, and the offer was
    appended last. On an upload with three higher offers it was dropped — and the next upload
    then saw "nothing changed", so it never came back.

    Which means it fired ZERO times per change on exactly the uploads that matter most: a v2
    replacing a judged record, or a concluded campaign with no results and no dates."""
    # What was last ANNOUNCED. Seeded rather than reached through a sequence of uploads,
    # because `few_verified_outcomes` is both the commonest top gap and the thing that
    # crowds the offer out, so the natural paths keep returning to the value they started
    # from. Any value here is reachable; the memo simply records what somebody was last told.
    store.set_state(conn, core._TOP_GAP_KEY, "judgments_never_reconciled")
    for title in ("Bogota", "Jakarta"):
        measured = core.ingest_campaign(conn, title=title, market="LATAM",
                                        status="concluded", starts_on="2026-01-01",
                                        ends_on="2026-02-01",
                                        detail=f"{title} ran.")["campaign_id"]
        core.add_metrics(conn, campaign_id=measured, structured={"roas": 3.4}, confirm=True)
    store.set_state(conn, core._TOP_GAP_KEY, "judgments_never_reconciled")

    # Three offers about the record itself, so the ranking offer cannot fit.
    crowded = core.ingest_campaign(conn, title="Lima", market="Peru", status="concluded",
                                   detail="A concluded record with no results and no dates.")
    offered = [a["tool"] for a in crowded["next_actions"]]
    assert len(offered) == actions.MAX_ACTIONS and "gaps" not in offered, (
        f"the fixture has to actually crowd the ranking offer out: {offered}"
    )
    assert core.top_gap(conn) != "judgments_never_reconciled", (
        "and the ranking has to have actually moved, or there was nothing to announce"
    )

    # The change it was announcing is still outstanding, so the next chance must take it.
    quiet = core.ingest_campaign(conn, title="Brazil", market="LATAM", status="proposed",
                                 detail="A proposal, which needs nothing said about it.")

    assert "gaps" in [a["tool"] for a in quiet.get("next_actions") or []], (
        "the ranking moved, the offer was trimmed away, and the move was recorded as "
        "already announced — so it was never made at all"
    )


def test_the_memo_records_what_was_shown_not_what_was_computed(conn):
    """The fix, at the mechanism. `trim` is the last word on what a user sees, so it is the
    only thing that can say whether the announcement happened — and the state write has to
    come after it, not before."""
    core.ingest_campaign(conn, title="Peru", market="LATAM", status="proposed",
                         detail="A first record, which makes the library non-empty.")
    store.set_state(conn, core._TOP_GAP_KEY, "something_else")

    core._the_gap_was_announced(conn, [{"tool": "add_metrics"}], "few_verified_outcomes")
    assert store.get_state(conn, core._TOP_GAP_KEY) == "something_else", (
        "an offer that never reached the user is not an announcement"
    )

    core._the_gap_was_announced(conn, [{"tool": "gaps"}], "few_verified_outcomes")
    assert store.get_state(conn, core._TOP_GAP_KEY) == "few_verified_outcomes"


def test_the_ranking_offer_stops_once_it_has_been_made(conn):
    """The other half. An offer that is never marked as shown is one that appears on every
    upload forever, which is the footer `MAX_ACTIONS` exists to keep out of the three slots."""
    core.ingest_campaign(conn, title="Peru", market="LATAM", status="proposed",
                         detail="A first record, which makes the library non-empty.")

    second = core.ingest_campaign(conn, title="Chile", market="LATAM", status="proposed",
                                  detail="A second record in the same shape.")

    assert "gaps" not in [a["tool"] for a in second.get("next_actions") or []]


# ── 5: an offer whose effect is "show me less" ─────────────────────────────

def test_the_gaps_a_person_can_honestly_set_aside_are_a_short_list(conn):
    """"Never going to be true here" is an honest thing to say about a market whose agency no
    longer exists. It is not an honest thing to say about "finished campaigns have no results
    on file" — which is THE gap behind the original customer complaint — or about "judgments
    have never been checked against what happened", where silencing it makes the product's
    own self-assessment unfalsifiable.

    Those are about work nobody has done, not facts that cannot exist. A blacklist let both
    through; only a whitelist can be argued with."""
    ids = [_campaign(conn, f"Campaign {n}", market=m)
           for n, m in enumerate(["LATAM", "EMEA", "APAC", "LATAM"])]
    core.add_metrics(conn, campaign_id=ids[0], structured={"roas": 3.4}, confirm=True)

    silenceable = {g["code"] for g in core.gaps(conn)["gaps"]
                   if "answer_gap" in [a["tool"] for a in g.get("next_actions") or []]}

    assert "few_verified_outcomes" not in silenceable
    assert "judgments_never_reconciled" not in silenceable
    assert silenceable, "something has to be honestly set-asideable, or D53 is unreachable"


def test_the_response_does_not_repeat_the_silencing_offer(conn):
    """Measured at 3 of 7 offers in one response, one per gap — making "shall we stop
    mentioning this?" the most repeated sentence in the product's report on its own evidence.
    `trim` caps each gap's own list and nothing caps the response, so "last in the gap's own
    list" controlled position and did nothing about frequency."""
    ids = [_campaign(conn, f"Campaign {n}", market=m)
           for n, m in enumerate(["LATAM", "EMEA", "APAC", "LATAM"])]
    core.add_metrics(conn, campaign_id=ids[0], structured={"roas": 3.4}, confirm=True)

    offers = [a["tool"] for g in core.gaps(conn)["gaps"]
              for a in g.get("next_actions") or []]

    assert offers.count("answer_gap") <= 1, offers


def test_a_gap_somebody_set_aside_is_visible_where_the_library_describes_itself(conn):
    """`readiness` is the session-start surface — the one that exists to say what the library
    CANNOT do. Reporting a rosier top gap with no hint that a worse one was silenced is the
    product overstating its own coverage, which is the failure this whole review is about,
    arriving through the door D53 opened."""
    for title in ("Peru", "Lima"):
        cid = _campaign(conn, title, market="Peru")
        core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    top = core.gaps(conn)["most_valuable"]
    assert top == "no_window", top

    core.answer_gap(conn, code=top, answer="not_applicable",
                    note="Nobody recorded dates for these and the decks are gone.",
                    said_by="R. Vega")

    assert core.readiness(conn)["set_aside"][0]["code"] == top


def test_a_half_migrated_database_reports_an_empty_queue_rather_than_crashing(conn, tmp_path):
    """Removing the blanket `except` was right and it exposed what the blanket was covering:
    a database written by an older version has no `metrics` table, and `_open_rows` reads it.

    The replacement is a NAMED condition, not a wider catch. "This database has not been
    upgraded yet, so nothing is waiting on feedback" is a true statement — `store.init_db`
    creates the tables on the next start — where "any exception means zero" was not, and that
    is the whole difference between this and the bug it replaced."""
    import sqlite3

    path = tmp_path / "v020.db"
    legacy = sqlite3.connect(path)
    legacy.executescript(
        "CREATE TABLE campaigns (id TEXT PRIMARY KEY, title TEXT NOT NULL, detail TEXT,"
        " collection TEXT, markets TEXT NOT NULL DEFAULT '[]');"
        "CREATE TABLE evaluations (id TEXT PRIMARY KEY, campaign_id TEXT,"
        " subject_title TEXT NOT NULL, cited_ids TEXT, analysis TEXT NOT NULL,"
        " predictions TEXT, created_at REAL NOT NULL);")
    legacy.commit()
    legacy.close()
    old = sqlite3.connect(path)
    old.row_factory = sqlite3.Row

    assert feedback.waiting(old) == 0

    old.close()


def test_the_open_rows_are_all_campaigns(conn):
    """The invariant `waiting()` and `_sentence` now rest on, stated once where it can be
    checked. §10.2/D110's questions were mixed into this list at first and both of those
    surfaces grew a `campaign_id` filter to cope; the rows moved to their own scope, and a
    filter kept afterwards guards a case that cannot arise — which reads as a live check and
    is not one. This is the check, in the one place the rule actually lives."""
    cid = _campaign(conn, "Bogota")
    _two_similar_rules(conn, cid)

    rows = feedback._open_rows(conn)

    assert rows, "the fixture has to produce open rows"
    assert all(r.get("campaign_id") for r in rows)
    assert feedback._rule_questions(conn), (
        "and there have to be rule questions in play, or this proves nothing"
    )


def test_the_queue_says_where_the_rule_questions_went(conn):
    """A campaign count that silently excludes them is accurate and unhelpful: somebody
    reading "1 campaign is waiting" beside a row 12 they have not noticed has been told a true
    thing and left with the wrong picture."""
    cid = _campaign(conn, "Bogota")
    _two_similar_rules(conn, cid)

    said = feedback.queue(conn)["what_it_means"]

    assert "1 campaign(s)" in said
    assert f"Row {feedback.RULES_ROW}" in said
    assert "1 pair(s)" in said
