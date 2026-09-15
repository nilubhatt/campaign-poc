"""
§11.6 — `author: unknown` as an explicit convention, not a null.

    "Backfill existing records as `author: unknown` with import date, explicitly. *Applies to
     `campaign_chunks.source.author`: a speaker note carries no author field at all and a PDF
     annotation often has no `/T`, so a null there means "the file did not say", which is a
     different statement from "nobody said it".*"

## Why a null is not good enough

`matched_author: null` on a search hit is read by whoever sees it, and it reads as "nobody".
There are at least four different reasons this library can fail to name an author, and they
have completely different consequences for a judgment citing the words:

  the file did not say       A PowerPoint speaker note has no author FIELD. Nobody withheld
                             anything; the format does not carry one. The words are almost
                             certainly the deck's own authors — the agency talking to itself.
  the file said nothing      A PDF annotation with no `/T`. The format CAN carry a name and
                             this one does not, so somebody commented anonymously, or their
                             reader was not configured with a name.
  stored before we kept it   Records imported before §2.5 extracted commentary at all. The
                             file may well have said; this library did not look.
  nobody has said            No claim about authorship was ever made, because nothing about
                             this text is a comment in the first place.

A judgment that cites "a reviewer objected" needs to know which of those it is looking at.
Told `null`, it cannot tell the deck's own speaker note from a client's anonymous objection —
and §2.5's whole point was that those are different things with different authority.

This is also §11.7's business, which is why the convention comes first: you cannot answer
"show me everything this person wrote" until "we do not know who wrote it" is one thing rather
than four.
"""
import pytest

import core
import store


def _campaign(conn, title="Colombia", **kw):
    kw.setdefault("market", "LATAM")
    kw.setdefault("status", "concluded")
    kw.setdefault("detail", f"A campaign called {title}, which ran in a market.")
    return core.ingest_campaign(conn, title=title, **kw)["campaign_id"]


def test_the_four_reasons_are_four_different_answers():
    """Not one null. Each says what a reader may conclude, and they conclude different
    things — a format that cannot carry a name and a person who withheld one are not the
    same fact about a comment."""
    assert set(store.AUTHOR_UNKNOWN) == {
        "format_carries_none", "file_did_not_say", "not_captured", "never_claimed"}


def test_a_speaker_note_says_the_format_carries_no_author(conn):
    """A PowerPoint notesSlide has no author FIELD. Nobody withheld anything, and the words
    are almost certainly the deck's own authors — the agency talking to itself, which §2.5
    says must never be read as a client's objection."""
    said = store.author_of({"kind": "speaker_note", "author": None})

    assert said["author"] == "unknown"
    assert said["why"] == "format_carries_none"
    assert "does not carry" in said["what_it_means"]


def test_an_anonymous_annotation_says_the_file_did_not_say(conn):
    """A PDF annotation CAN carry `/T` and this one does not — so somebody commented without
    a name, which is a fact about the person rather than about the format."""
    said = store.author_of({"kind": "annotation", "author": None})

    assert said["author"] == "unknown"
    assert said["why"] == "file_did_not_say"


def test_a_named_author_is_left_alone(conn):
    said = store.author_of({"kind": "comment", "author": "R. Vega"})

    assert said["author"] == "R. Vega"
    assert "why" not in said


def test_body_text_never_claimed_an_author(conn):
    """A chunk of the brief itself is not a comment and has no author question to answer.
    Reporting `unknown` there would invent a missing fact about every record in the
    library."""
    said = store.author_of({"kind": "body", "author": None})

    assert said["why"] == "never_claimed"


def test_a_record_stored_before_commentary_was_read_says_so(conn):
    """"Backfill existing records… with import date, explicitly." The file may well have said
    who wrote it; this library did not look. That is a fact about US, and reporting it as the
    file's silence would blame the customer's deck for our own gap."""
    cid = _campaign(conn)
    conn.execute("UPDATE campaigns SET commentary_checked = 0 WHERE id = ?", (cid,))
    conn.commit()

    said = store.author_of({"kind": "comment", "author": None}, commentary_checked=False)

    assert said["why"] == "not_captured"
    assert "nobody looked" in said["what_it_means"]


def test_the_backfill_records_when_it_was_imported(conn):
    """"with import date" — the item's own words. Without it "we never looked" is undated,
    so nobody can tell a record from before §2.5 from one whose extraction failed today."""
    cid = _campaign(conn)
    conn.execute("UPDATE campaigns SET commentary_checked = 0 WHERE id = ?", (cid,))
    conn.commit()

    backfilled = core.backfill_author_unknown(conn)

    assert backfilled["records"] >= 1
    assert backfilled["basis"] == "computed"
    marked = store.get_campaign(conn, cid)
    assert marked["authorship_backfilled_at"]


def test_the_backfill_never_overwrites_a_name(conn):
    """The one thing it must not do. A record whose commentary WAS read carries real names,
    and stamping `unknown` over them would destroy the personal data §11.7 has to be able to
    show and erase — while making the library look like it never knew."""
    cid = _campaign(conn)
    store.insert_chunks(conn, cid, ["A reviewer objected to the timeline."],
                        kind="commentary",
                        sources=[{"kind": "comment", "author": "R. Vega",
                                  "anchor": "slide 3"}])

    core.backfill_author_unknown(conn)

    rows = conn.execute("SELECT source FROM campaign_chunks WHERE campaign_id = ? "
                        "AND kind = 'commentary'", (cid,)).fetchall()
    assert "R. Vega" in (rows[0]["source"] or "")


def test_the_backfill_runs_once(conn):
    """Twice is not wrong but it is a lie about the library's history: a second stamp moves
    the import date onto the day somebody re-ran it."""
    cid = _campaign(conn)
    conn.execute("UPDATE campaigns SET commentary_checked = 0 WHERE id = ?", (cid,))
    conn.commit()
    first = core.backfill_author_unknown(conn)

    again = core.backfill_author_unknown(conn)

    assert first["records"] >= 1
    assert again["records"] == 0
    assert (store.get_campaign(conn, cid)["authorship_backfilled_at"]
            == first["backfilled_at"])


# ── it has to reach the reader, or it is a convention nobody applies ───────

def test_a_search_hit_says_why_it_cannot_name_the_author(conn):
    """`matched_author: null` is what a judgment actually sees, and a judgment citing "a
    reviewer objected" needs to know whether that was a client or the deck talking to
    itself — §2.5's distinction, which a null erases."""
    cid = _campaign(conn, "Peru launch")
    store.insert_chunks(conn, cid, ["The timeline here looks short for a launch."],
                        kind="commentary",
                        sources=[{"kind": "speaker_note", "author": None,
                                  "anchor": "slide 4"}])
    core.finish_indexing(conn, campaign_id=cid)

    found = core.find_similar(conn, text="the timeline looks short for a launch")

    row = next(e for e in found if e["campaign_id"] == cid)
    assert row["matched_author"] == "unknown"
    assert row["matched_author_why"] == "format_carries_none"


def test_a_named_author_still_reaches_the_reader(conn):
    cid = _campaign(conn, "Peru launch")
    store.insert_chunks(conn, cid, ["The timeline here looks short for a launch."],
                        kind="commentary",
                        sources=[{"kind": "comment", "author": "R. Vega",
                                  "anchor": "slide 4"}])
    core.finish_indexing(conn, campaign_id=cid)

    found = core.find_similar(conn, text="the timeline looks short for a launch")

    row = next(e for e in found if e["campaign_id"] == cid)
    assert row["matched_author"] == "R. Vega"
    assert "matched_author_why" not in row


def test_a_library_with_older_records_is_offered_the_stamp(conn):
    """D116's rule: the write that makes a tool's output non-empty has to name it. Here the
    "write" is a library that already holds records from before commentary extraction — they
    are what the tool is for, and nothing would otherwise mention it exists."""
    cid = _campaign(conn)
    conn.execute("UPDATE campaigns SET commentary_checked = 0 WHERE id = ?", (cid,))
    conn.commit()

    assert "backfill_author_unknown" in [
        a["tool"] for a in core.readiness(conn).get("next_actions") or []]


def test_a_library_with_nothing_to_stamp_is_not_offered_it(conn):
    """The silence half. A library whose records were all read has nothing to say here, and
    an offer that is always present is the footer this product keeps refusing."""
    cid = _campaign(conn)
    conn.execute("UPDATE campaigns SET commentary_checked = 1 WHERE id = ?", (cid,))
    conn.commit()

    assert "backfill_author_unknown" not in [
        a["tool"] for a in core.readiness(conn).get("next_actions") or []]
