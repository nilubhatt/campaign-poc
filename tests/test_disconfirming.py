"""
§6.4 (idea J): require a search for the disconfirming case.

The review's words: *"Mexico is the library's warning: a brief rated weak that performed well.
Resemblance to a strong performer is not evidence. A reasoner that only retrieves supporting
precedent will drift toward confirming whatever the first strong match suggests. Fix: before a
verdict is saved, run one query for precedent that contradicts it, and record what came back —
including 'nothing'. Cheap, and it makes overconfidence visible."*

**The server runs it, not the model.** A required `disconfirming` field the model fills in is
a field the model fills in — and this codebase already decided that question twice, in §2.4's
`basis` rule and §7.8's premise: a check is only worth anything if a difference in it is a
bug, which holds only when the server did it. A model that has already reached a verdict
asked to go and look for evidence against itself is being asked to mark its own homework, and
the failure mode the review names — drifting toward confirming the first strong match — is
exactly the failure mode that would shape the query.

**What "contradicting" means depends on the verdict**, which is why this cannot happen at
retrieval time and has to happen at save time:

  a negative verdict (revise / reject)  → precedent that looks like this and WORKED anyway
  a positive verdict (approve)          → precedent that looks like this and did NOT work

Mexico is the first row: the brief was rated weak, and it performed. Nothing in the library
was asked whether that had happened before.

**And "nothing" is a result.** The review says so explicitly, and it is the more common one:
a library with two campaigns in it cannot contradict anything, and a judgment that rests on a
library too thin to argue back is a different thing from one nothing argued back against. The
two must not read the same.
"""
import pytest

import core
import store


@pytest.fixture
def library(conn):
    """One campaign that looks like the subject and worked, one that looked like it and did
    not, and a rulebook. Both are `verified` — backed by real metric rows — because the whole
    question is whether the library can argue with a judgment using evidence."""
    # Metrics first, then the tag: the store refuses `source: verified` on a campaign with no
    # measured row, which is the provenance rule this item leans on.
    ids = {}
    for key, title, verdict, numbers in (
            ("worked", "Mexico launch", "performed_well",
             "CTR 3.4 percent, well above the 1.8 percent benchmark."),
            ("failed", "Chile launch", "underperformed", "CTR 0.6 percent, well below.")):
        ids[key] = core.ingest_campaign(
            conn, title=title, status="concluded",
            detail="Creator-led launch with four colourways and a compressed three-week "
                   "flight.")["campaign_id"]
        core.add_metrics(conn, campaign_id=ids[key], detail=numbers)
        store.update_campaign(conn, ids[key],
                              tags=[{"value": verdict, "source": "verified"}])
    ids["rules"] = core.ingest_campaign(
        conn, title="Brand guidelines", record_type="reference",
        detail="No AI-generated imagery in any paid placement.")["campaign_id"]
    return ids


def _evaluation(library, **over):
    base = dict(
        subject_title="Colombia v1", verdict="revise",
        summary="A compressed flight with four colourways is a stretch.",
        cited_ids=[library["failed"]],
        findings=[{"severity": "should_fix", "kind": "precedent_departure",
                   "departure": "regression",
                   "finding": "Compresses the flight to three weeks",
                   "precedent": {"campaign_id": library["failed"],
                                 "quote": "compressed three-week flight"}}])
    base.update(over)
    return base


# ── the server looks, and the direction depends on the verdict ──────────────

def test_a_negative_verdict_is_checked_against_precedent_that_worked(conn, library):
    """Mexico, exactly: a brief rated weak, and a campaign that looks like it performed well.
    Nothing in the library had ever been asked whether that had happened before."""
    result = core.save_evaluation(conn, **_evaluation(library))

    check = result["disconfirming"]
    assert check["looked_for"] == "precedent_that_worked"
    assert library["worked"] in [row["campaign_id"] for row in check["found"]]
    assert check["basis"] == "computed", "the server looked; the model did not report it"


def test_a_positive_verdict_is_checked_against_precedent_that_failed(conn, library):
    """The other direction, and the one an approving reasoner never goes looking for."""
    result = core.save_evaluation(conn, **_evaluation(
        library, verdict="approve", summary="Looks fine.", findings=[]))

    check = result["disconfirming"]
    assert check["looked_for"] == "precedent_that_failed"
    assert library["failed"] in [row["campaign_id"] for row in check["found"]]


def test_the_search_is_recorded_on_the_judgment_not_only_returned(conn, library):
    """"Record what came back" — a check that lives for one response is a check nobody can
    audit later, and the review asked for this precisely so overconfidence stays visible."""
    result = core.save_evaluation(conn, **_evaluation(library))

    stored = store.get_evaluation(conn, result["evaluation_id"])
    assert stored["evidence"]["disconfirming"]["looked_for"] == "precedent_that_worked"
    assert stored["evidence"]["disconfirming"]["found"]


# ── "nothing" is a result, and not the same as "we could not look" ──────────

def test_nothing_found_is_recorded_as_nothing_found(conn, library):
    """The review says so in as many words. A judgment nothing argued back against is a
    stronger judgment, and it can only be read as one if the search is on the record."""
    lonely = core.ingest_campaign(
        conn, title="Peru launch", status="concluded",
        detail="A single-market print campaign with no creators at all.")["campaign_id"]
    core.add_metrics(conn, campaign_id=lonely, detail="Nothing measured worth reporting.")
    store.update_campaign(conn, lonely,
                          tags=[{"value": "underperformed", "source": "verified"}])

    result = core.save_evaluation(conn, **_evaluation(
        library, verdict="approve", summary="Fine.", findings=[],
        subject_title="Something else entirely"))
    assert result["disconfirming"]["looked_for"] == "precedent_that_failed"
    assert isinstance(result["disconfirming"]["found"], list)


def test_a_library_that_cannot_argue_back_says_so(conn):
    """Different from "nothing contradicted it". A library with no measured outcomes cannot
    contradict anything, and reporting that as a clean check would turn an empty shelf into
    a supporting vote — the confident-shape-with-nothing-in-it failure again."""
    core.ingest_campaign(conn, title="Only brief", detail="A proposal with no results.")

    result = core.save_evaluation(
        conn, subject_title="Colombia v1", verdict="revise", summary="A stretch.",
        findings=[{"severity": "should_fix", "kind": "missing_information",
                   "finding": "No end date"}])

    check = result["disconfirming"]
    assert check["found"] == []
    assert check["code"] == "nothing_to_check_against"
    assert "measured" in check["what_it_means"]


def test_the_three_outcomes_of_the_check_are_three_different_things(conn, library):
    """"It could not be checked", "it was checked and nothing came back" and "it was checked
    and something did" are three conclusions, and collapsing any two is how an absence of
    evidence becomes evidence of absence. The first is covered above, on a library with
    nothing measured; these are the other two, which differ only in what the search found."""
    contradicted = core.save_evaluation(conn, **_evaluation(library))
    assert contradicted["disconfirming"]["code"] == "contradicting_precedent"

    # The same library, and a judgment whose contradicting precedent was already weighed.
    weighed = core.save_evaluation(conn, **_evaluation(
        library, cited_ids=[library["failed"], library["worked"]]))
    assert weighed["disconfirming"]["code"] == "nothing_contradicted_it"
    assert weighed["disconfirming"]["found"], "it looked and it found; nothing was UNCITED"

    assert "not possible" in core._say_the_disconfirming_check(
        {"code": "nothing_to_check_against", "uncited": []})
    assert "held" in core._say_the_disconfirming_check(
        {"code": "nothing_contradicted_it", "uncited": []})


# ── it has to be visible, or it is a column nobody reads ────────────────────

def test_precedent_the_judgment_never_cited_is_called_out(conn, library):
    """The point is not the record, it is the noticing. A contradicting campaign the judgment
    did not cite is the case the review describes — the reasoner never retrieved it, so it
    never had to explain it away."""
    result = core.save_evaluation(conn, **_evaluation(library))

    check = result["disconfirming"]
    assert check["uncited"], "Mexico was never cited and it is the whole point"
    assert library["worked"] in check["uncited"]
    assert "disconfirming" in result["note"] or "contradict" in result["note"].lower()


def test_precedent_the_judgment_already_weighed_is_not_thrown_back_at_it(conn, library):
    """A judgment that cited the contradicting campaign has already done the work. Reporting
    it as an unexamined contradiction would train the reader to ignore the field."""
    result = core.save_evaluation(conn, **_evaluation(
        library, cited_ids=[library["failed"], library["worked"]]))

    assert library["worked"] not in result["disconfirming"]["uncited"]


def test_the_model_cannot_write_the_check_itself(conn, library):
    """Same rule as `basis` and `precedent.checked`: a check the model reports is a claim,
    and §7.8's premise — a difference in a computed thing is a bug — holds only if the server
    did it."""
    with pytest.raises(ValueError) as e:
        core.save_evaluation(conn, **_evaluation(
            library, evidence={"disconfirming": {"found": [], "looked_for": "nothing"}}))
    assert "server" in str(e.value).lower()


def test_it_reaches_the_model_over_the_protocol(conn, library):
    import asyncio
    import json

    async def call():
        import mcp_server
        result = await mcp_server.mcp.call_tool("save_evaluation", _evaluation(library))
        return json.loads(result.content[0].text)

    saved = asyncio.run(call())
    assert saved["disconfirming"]["looked_for"] == "precedent_that_worked"
    assert library["worked"] in saved["disconfirming"]["uncited"]


# ── both poles, before the verdict exists ───────────────────────────────────

def test_the_evidence_package_carries_both_poles(conn, library):
    """The save-time check records overconfidence; it cannot prevent it, because by then the
    verdict is written. What changes the reasoning is seeing both sides while reasoning — so
    `prepare_evaluation` separates the precedent that worked from the precedent that did not,
    instead of handing over one ranked list in which the strongest match sets the tone."""
    package = core.prepare_evaluation(
        conn, subject_title="Colombia v1",
        proposal_text="Creator-led launch with four colourways and a compressed flight.")

    poles = package["outcomes"]
    assert library["worked"] in [r["campaign_id"] for r in poles["worked"]]
    assert library["failed"] in [r["campaign_id"] for r in poles["did_not_work"]]
    assert "both" in package["note"].lower() or "contradict" in package["note"].lower()


def test_the_poles_only_count_measured_outcomes(conn, library):
    """A `performed_well` tag somebody typed is an impression. Weighing it as the
    counterweight to a verdict would be the unverified-impression-as-evidence failure that
    tag provenance exists to prevent."""
    stated = core.ingest_campaign(
        conn, title="Guess launch", status="concluded",
        detail="Creator-led launch with four colourways and a compressed three-week flight.",
        tags=["performed_well"])["campaign_id"]

    package = core.prepare_evaluation(
        conn, subject_title="Colombia v1",
        proposal_text="Creator-led launch with four colourways and a compressed flight.")

    assert stated not in [r["campaign_id"] for r in package["outcomes"]["worked"]]


def test_the_check_survives_a_database_that_predates_it(tmp_path):
    """It runs on every save, so a column it names and an upgraded database does not have
    would crash every judgment on the machine that already holds data — which is the machine
    the review was gathered on. §6.1's `text_on_file` learned this one function over."""
    import sqlite3
    import time

    path = tmp_path / "v020.db"
    legacy = sqlite3.connect(path)
    legacy.executescript("""
        CREATE TABLE campaigns (id TEXT PRIMARY KEY, title TEXT NOT NULL, detail TEXT);
        CREATE TABLE evaluations (
            id TEXT PRIMARY KEY, campaign_id TEXT, subject_title TEXT NOT NULL,
            cited_ids TEXT, analysis TEXT NOT NULL, predictions TEXT,
            created_at REAL NOT NULL);
        CREATE TABLE reconciliations (
            id TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL, actual TEXT,
            comparison TEXT NOT NULL, created_at REAL NOT NULL);
    """)
    legacy.execute("INSERT INTO evaluations VALUES ('ev_old', NULL, 'v1', '[]', ?, NULL, ?)",
                   ("An essay.", time.time()))
    legacy.commit()
    legacy.close()

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    store._migrate_schema(conn)

    saved = core.save_evaluation(
        conn, subject_title="Colombia v1", verdict="revise", summary="A stretch.",
        findings=[{"severity": "should_fix", "kind": "missing_information",
                   "finding": "No end date"}])
    assert saved["disconfirming"]["code"] == "nothing_to_check_against"
