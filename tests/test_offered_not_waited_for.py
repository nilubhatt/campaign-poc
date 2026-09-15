"""
§10.6 — "offer it, do not wait to be asked" — applied to the rest of the surface.

The item's own sentence is about the feedback menu: *"The menu is worthless if the user has
to know it exists."* That is true of the menu and it is not only true of the menu. Five
tracker rows say the same thing about five other tools, and they are one piece of work:

  D54   `gaps()` offered at session start and after an upload that changes the top gap,
        rather than by a sentence in a docstring.
  D67   `coverage()` offered from `list_campaigns` and at session start.
  D76   Three tools answer "what is missing" and should be two: `gaps()` and `readiness`
        each carry their own copy of the first steps, unordered in one and ordered in the
        other, and they can disagree.
  D98   A partial re-index reports `reindexed.embedded` and nothing else — no warning, no
        notice, and `update_campaign`'s description does not tell the model to read it.
        Every other partial-state path in this codebase raises a notice naming the fix.
  D116  Every model-facing tool must be named in a `next_actions` from the write that makes
        its output non-empty. Seven occurrences before this file existed.

The failure is the same each time and it is not a small one: a tool nobody offers is a tool
nobody calls, and a feature nobody calls cannot fail, so nothing reports it broken. §8.6 is
the proof — `note_correction` was named nowhere but its own definition, so the gate built to
act on corrections had nothing to act on, and the whole item was inert while every test
passed.

What is deliberately NOT here: an offer on every response. `MAX_ACTIONS` is three and the
reason is written into `actions`: a list that is always present stops being read, and then
the one that mattered is not read either. So each offer below is gated on the library fact
that makes it worth making, and there are tests for the silence as well as for the offer.
"""
import pytest

import actions
import context
import core
import corrections
import feedback
import store


def _campaign(conn, title="Colombia", **kw):
    kw.setdefault("market", "LATAM")
    kw.setdefault("status", "concluded")
    kw.setdefault("detail", f"A campaign called {title}, which ran in a market.")
    return core.ingest_campaign(conn, title=title, **kw)["campaign_id"]


def _tools(result) -> list:
    return [a["tool"] for a in (result.get("next_actions") or [])]


# ── D76: one source for "what to add first" ────────────────────────────────

def test_the_two_surfaces_agree_on_the_first_steps(conn):
    """`gaps()` and `readiness()` are the same question asked twice. Today the empty-library
    gap carries `to_first_upload()` and readiness carries `_shortest_path()`, and the two are
    written separately — so "what do I add first" has two answers that drift apart with no
    test able to notice."""
    gaps = core.gaps(conn)
    ready = core.readiness(conn)

    assert gaps["shortest_path"] == ready["shortest_path"], (
        "the ordered path has two implementations and they have diverged"
    )


def test_the_empty_library_is_sent_down_the_ordered_path(conn):
    """"one brief you liked, one you did not, the rulebook" — a path, not a menu. The empty
    gap's own offer was an unordered copy of the first two steps, so a user who followed the
    gap and a user who followed `getting_started` were given different first moves."""
    gap = core.gaps(conn)["gaps"][0]

    assert gap["code"] == "library_is_empty"
    assert gap["next_actions"] == core.readiness(conn)["shortest_path"][:1], (
        "the empty library has exactly one first step and it is the path's first step"
    )


def test_a_working_library_carries_no_path(conn):
    """Guidance that never stops appearing is guidance nobody reads. `readiness` already
    draws this line with `readiness_for_listing`; the folded version has to draw it too."""
    liked = _campaign(conn, "Peru — the one we liked",
                      tags=[{"value": "liked", "source": "stated"}])
    _campaign(conn, "Chile — the one we did not",
              tags=[{"value": "not_liked", "source": "stated"}])
    core.ingest_campaign(conn, title="Brand guidelines", record_type="reference",
                         detail="The rules, such as they are.")
    core.add_metrics(conn, campaign_id=liked, structured={"roas": 3.4}, confirm=True)

    assert core.gaps(conn).get("shortest_path") == []


# ── D54: gaps offered, not documented ──────────────────────────────────────

def test_the_session_start_offers_the_ranked_gaps(conn):
    """"Call it when the user asks how good their library is" is an instruction in a
    docstring, which is the form this project's own principle says models drop. A library
    with something ranked and nothing said about it is the state the item describes."""
    _campaign(conn)

    assert "gaps" in _tools(core.readiness(conn))


def test_an_empty_library_is_not_offered_its_gaps(conn):
    """Nothing is ranked, and the one thing to do is already the whole of `shortest_path`.
    An offer to enumerate the gaps of an empty library is the permanent-complaint failure."""
    assert "gaps" not in _tools(core.readiness(conn))


def test_an_upload_that_changes_the_top_gap_says_so(conn):
    """The second half of D54, and the harder half: an upload that moves what matters most is
    the moment the ranking is worth reading, and it is also the only moment somebody is
    looking. A library whose worst problem just changed and does not mention it has made the
    ranking a thing you have to remember to go and ask for.

    Both records carry a window on purpose. A concluded record with no results AND no dates
    fills all three of `trim`'s slots with offers about ITSELF, so a fourth offer is dropped
    whether or not it fired — and the silence half of this test then passed with `_gap_moved`
    hard-wired to True. Mutation found that; the fixture is the fix."""
    first = core.ingest_campaign(conn, title="Peru", market="LATAM", status="concluded",
                                 starts_on="2026-01-01", ends_on="2026-02-01",
                                 detail="The first record, which makes the library non-empty.")

    assert "gaps" in _tools(first), "the first upload changes the top gap from `empty`"

    second = core.ingest_campaign(conn, title="Chile", market="LATAM", status="concluded",
                                  starts_on="2026-01-01", ends_on="2026-02-01",
                                  detail="A second record in the same shape as the first.")

    assert "gaps" not in _tools(second), (
        "the top gap is unchanged, so saying it again is noise on every upload"
    )
    assert len(_tools(second)) < actions.MAX_ACTIONS, (
        f"there is a free slot, so `gaps` is absent because it did not fire rather than "
        f"because it was trimmed: {_tools(second)}"
    )


# ── D67: coverage offered, and consuming its own thin cell ─────────────────

def _library_with_a_shape(conn):
    """Three markets, one of them measured — so `thin` is a contrast rather than the whole
    library. A wholly unmeasured library reports `thin: []` on purpose (its answer is "the
    place to start is not a particular market"), which is `gaps`'s business, not this one's.
    """
    for market in ("LATAM", "EMEA", "APAC"):
        cid = _campaign(conn, f"{market} launch", market=market)
        if market == "LATAM":
            core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
            _campaign(conn, "LATAM second", market=market)
    return conn


def test_a_listing_offers_the_shape_of_the_library(conn):
    """"`list_campaigns` answers 'what have I got' one record at a time. This answers the
    question a marketer actually has" — which is its own docstring admitting the listing is
    the moment for it."""
    _library_with_a_shape(conn)

    offered = core.coverage_offer(conn)

    assert offered and offered[0]["tool"] == "coverage"


def test_the_offer_names_the_thinnest_cell_rather_than_counting_again(conn):
    """"5.6 consumes `thin[0]` rather than growing a third grouping." An offer whose `why` is
    computed by a second, private grouping is the D55 defect — two surfaces describing the
    same library differently — reintroduced inside the offer that points at one of them."""
    _library_with_a_shape(conn)

    why = core.coverage_offer(conn)[0]["why"]
    worst = core.coverage(conn)["thin"][0]

    # `market` alone. The first version allowed `... or (thin[0].get("collection") or "") in
    # why`, and `collection` is None for these fixtures — so the fallback was `"" in why`,
    # which is True of every string. Mutating `thin[0]` to `thin[-1]` left it green: an
    # assertion satisfied by the empty string is not an assertion.
    assert worst["market"] and worst["market"] in why
    assert core.coverage(conn)["thin"][-1]["market"] not in why or len(
        core.coverage(conn)["thin"]) == 1, (
        "the fixture has to have a distinguishable worst cell, or picking any cell passes"
    )


def test_a_wholly_unmeasured_library_is_not_offered_the_matrix(conn):
    """`coverage` reports `thin: []` for this library on purpose — its own answer is that the
    place to start is not a particular market. Offering a matrix whose every cell reads the
    same way is offering to show somebody a blank grid."""
    for market in ("LATAM", "EMEA", "APAC"):
        _campaign(conn, f"{market} launch", market=market)

    assert not core.coverage_offer(conn)


def test_a_library_with_one_market_is_not_offered_a_matrix(conn):
    """A grouping with one cell is a list of everything with a heading on it."""
    cid = _campaign(conn, "Colombia")
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)

    assert not core.coverage_offer(conn)


# ── D98: the partial re-index that reported nothing ────────────────────────

def test_a_partial_reindex_raises_a_notice(conn, monkeypatch):
    """"Every other partial-state path in this codebase raises a notice naming the fix." An
    edit that leaves the index behind makes the record unfindable by its NEW wording while
    `update_campaign` returns success — the §6.1 defect from the other end, and silent."""
    cid = _campaign(conn, "Colombia", detail="The original wording of this brief.",
                    # Long enough that each section is its own chunk: `pack` merges adjacent
                    # small units, so eight short paragraphs are one chunk and the fixture
                    # would be testing a total failure while claiming to test a partial one.
                    deck_text="\n\n".join(f"Section {n}. " + ("words " * 400)
                                          for n in range(4)))

    # Half the sections re-index and half do not, which is the state the field name
    # `embedded` describes and nothing else reported.
    import embedding
    calls = {"n": 0}
    real = embedding.embed

    def flaky(text, *a, **kw):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("the embedding service went away mid-edit")
        return real(text, *a, **kw)

    monkeypatch.setattr(embedding, "embed", flaky)
    edited = core.update_campaign(conn, campaign_id=cid,
                                  detail="Entirely new wording for this brief.")

    assert edited["reindexed"]["embedded"] < edited["reindexed"]["chunks"], (
        "the fixture has to actually produce a PARTIAL re-index, not a total failure"
    )

    codes = [w["code"] for w in (edited.get("warnings") or [])]
    assert "chunk_not_embedded" in codes, (
        f"a partial re-index said only {edited.get('reindexed')} — no warning, no notice"
    )
    assert "finish_indexing" in _tools(edited)


def test_the_notice_survives_the_response_it_was_raised_in(conn, monkeypatch):
    """D20's rule: a warning that needs a person lives for exactly one response, so the thing
    the library most wants somebody to act on is the thing it forgets fastest."""
    cid = _campaign(conn, "Colombia", detail="The original wording.")

    import embedding
    monkeypatch.setattr(embedding, "embed",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("gone")))
    core.update_campaign(conn, campaign_id=cid,
                         detail="New wording.\n\n" + "Another paragraph.\n\n" * 6)

    assert "chunk_not_embedded" in [n["code"] for n in store.open_notices(conn, cid)]


def test_a_complete_reindex_raises_nothing(conn):
    """The silence half. A notice on every successful edit is a notice nobody reads."""
    cid = _campaign(conn, "Colombia", detail="The original wording.")

    edited = core.update_campaign(conn, campaign_id=cid, detail="New wording entirely.")

    assert "chunk_not_embedded" not in [w["code"] for w in (edited.get("warnings") or [])]
    assert not store.open_notices(conn, cid)


def test_a_later_clean_edit_clears_the_notice(conn, monkeypatch):
    """The notice was retracted only by `finish_indexing`, so an edit that half-failed and a
    second edit that succeeded left the record WHOLLY searchable with the notice still open —
    and the queue then listed it under `needs_attention`, which is the top-ranked reason a
    campaign waits on a person. The most urgent row in the product, permanently occupied by a
    job that finished last week."""
    cid = _campaign(conn, "Colombia", detail="The original wording.")

    import embedding
    real = embedding.embed
    monkeypatch.setattr(embedding, "embed",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("gone")))
    core.update_campaign(conn, campaign_id=cid,
                         detail="New wording.\n\n" + "Another paragraph.\n\n" * 6)
    assert store.open_notices(conn, cid), "the fixture has to raise the notice first"
    monkeypatch.setattr(embedding, "embed", real)

    core.update_campaign(conn, campaign_id=cid, detail="Wording that indexes cleanly.")

    assert not store.open_notices(conn, cid)
    assert not [r for r in feedback.queue(conn)["rows"]
                if r.get("campaign_id") == cid and "needs_attention" in (r.get("needs") or [])]


def test_finishing_the_index_clears_the_notice(conn, monkeypatch):
    """A notice nothing retracts is one that accumulates until the queue is all of them —
    which is how the top-ranked queue reason becomes permanently occupied by a job that was
    done last week."""
    cid = _campaign(conn, "Colombia", detail="The original wording.")

    import embedding
    real = embedding.embed
    monkeypatch.setattr(embedding, "embed",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("gone")))
    core.update_campaign(conn, campaign_id=cid,
                         detail="New wording.\n\n" + "Another paragraph.\n\n" * 6)
    monkeypatch.setattr(embedding, "embed", real)

    core.finish_indexing(conn, campaign_id=cid)

    assert not store.open_notices(conn, cid)


# ── D116: the tools that had no route to them ──────────────────────────────

def test_a_prepared_brief_offers_the_judgment_that_saves_it(conn):
    """`prepare_evaluation` packages the evidence and then says nothing about what to do with
    the verdict. `save_evaluation` is the only thing that persists it, and the whole
    reconciliation half of the product rests on the judgment having been saved."""
    _campaign(conn, "Peru")

    prepared = core.prepare_evaluation(conn, subject_title="A new Colombia brief",
                                       proposal_text="A store launch in Bogota.")

    assert "save_evaluation" in _tools(prepared)


def test_a_recorded_correction_offers_the_tool_that_says_where_it_stands(conn):
    """§8.6's own defect, one stage on: `note_correction` is now offered, and what it writes
    is still only readable by a tool nobody is told about."""
    cid = _campaign(conn, "Colombia")

    noted = corrections.note(conn, text="Never put the logo on a dark background.",
                                 provenance="client email, 4 March", campaign_id=cid)

    assert "correction_status" in _tools(noted)


def test_a_set_aside_rule_offers_the_way_back(conn):
    """"Undo a set-aside" is only reachable by somebody who knows `reopen_correction` exists.
    The moment it becomes worth knowing is the moment the rule is set aside."""
    cid = _campaign(conn, "Colombia")
    noted = corrections.note(conn, text="Never put the logo on a dark background.",
                                 provenance="client email, 4 March", campaign_id=cid)

    aside = corrections.set_aside(conn, noted["correction_id"], why="Not a rule, one client's taste.")

    assert "reopen_correction" in _tools(aside)


def test_a_confounded_campaign_offers_the_person_the_last_word(conn):
    """§9.8: the server can say "the port was shut for nine of the thirty days"; only a person
    can say whether that is why. `attribute_outcome` is that sentence, and nothing has ever
    pointed at it — so the item's whole human half was unreachable."""
    cid = _campaign(conn, "Mexico launch", market="Mexico",
                    starts_on="2026-09-01", ends_on="2026-09-30")
    context.record(conn, starts_on="2026-09-05", ends_on="2026-09-14",
                              scope="market", scope_value="Mexico", kind="supply_chain",
                              description="The port was shut for nine days.",
                              recorded_by="R. Vega")

    filed = core.add_metrics(conn, campaign_id=cid, structured={"roas": 1.1}, confirm=True)

    assert "attribute_outcome" in _tools(filed)


def test_an_unconfounded_campaign_is_not_asked_to_explain_itself(conn):
    """The silence half, and it matters more than the offer: an invitation to attribute an
    outcome to an event that did not overlap it is an invitation to invent a cause."""
    cid = _campaign(conn, "Mexico launch", market="Mexico",
                    starts_on="2026-09-01", ends_on="2026-09-30")

    filed = core.add_metrics(conn, campaign_id=cid, structured={"roas": 1.1}, confirm=True)

    assert "attribute_outcome" not in _tools(filed)
