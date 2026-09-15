"""
Phase 10: capturing feedback without making anyone type.

The review: *"Feedback is the only input that makes this library worth anything, and it is
currently the hardest thing to give: the user has to remember which campaigns are outstanding,
name one, and compose prose. **It should be a numbered choice.**"*

    Which campaign do you want to give feedback on?
      1  Colombia - Calle 82 Bogota  (v2 received)   no verdict on v2
      2  UAE - Dubai Mall store opening              no verdict yet
      ...
     10  Add feedback to a concluded campaign
     11  Not here - show more

*"Every row says **why it is open**, because 'what does this want from me' is the question the
user actually has. Nothing requires typing beyond a number."*

Six items, and they are one feature:

**10.1 — "open" means NEEDS SOMETHING FROM YOU**, not a status field. *"A campaign is open when
the library is missing something a human has to supply: no creative-reaction tag, no
performance tag, predicted metrics with no actuals, an evaluation whose predictions were never
reconciled, or a superseding version received with no verdict."* Ordered by what is most
valuable to capture — *"a rejected brief whose v2 has just landed is worth more than a stub
nobody has briefed."*

**10.2 — fixed numbers.** *"Campaigns fill 1–9. 10 is always 'add feedback to a concluded
campaign' and 11 is always 'show more', whether there are three open campaigns or nine.
Floating positions break the habit the menu exists to create, and a small gap in the numbering
is a cheaper price than a moving target."*

**10.3 — numbered all the way down.** The follow-ups are closed sets too, so they are asked the
same way, and free text comes last and optional: *"the free-text box is where 'slide 23 should
be the standard' gets captured — the highest-value sentence in the whole system — so it should
be invited, never required."*

**10.4 — the SERVER builds the menu.** *"If the model composes the list, two users see
different orderings and different numbering for the same library — the exact class of variance
the consistency work is trying to remove."*

**10.5 — numbers are not identifiers.** *"If anything changes between rendering the menu and
the user answering, '3' silently means a different campaign and feedback lands on the wrong
record."* A stale token returns the refreshed menu instead of writing.

**10.6 — offer it, do not wait to be asked.** *"The menu is worthless if the user has to know it
exists."*
"""
import pytest

import core
import feedback
import store


def _campaign(conn, title="Colombia", **kw):
    kw.setdefault("market", "LATAM")
    kw.setdefault("status", "concluded")
    kw.setdefault("detail", f"A campaign called {title}.")
    return core.ingest_campaign(conn, title=title, **kw)["campaign_id"]


def _queue(conn, **kw):
    return feedback.queue(conn, **kw)


def _rows(queue):
    return {row["number"]: row for row in queue["rows"]}


# ── 10.1: "open" means needs something from a human ────────────────────────

def test_a_campaign_with_no_reaction_tag_is_open(conn):
    """"The library is missing something a human has to supply." Whether the work was liked is
    not derivable from anything on file."""
    cid = _campaign(conn)

    row = next(r for r in _queue(conn)["rows"] if r.get("campaign_id") == cid)

    assert "what they made of the work" in row["why"].lower()


def test_a_campaign_with_everything_recorded_is_not_open(conn):
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    core.update_campaign(conn, campaign_id=cid, tags=[
        {"value": "liked", "source": "stated"},
        {"value": "performed_well", "source": "verified"}])

    assert not [r for r in _queue(conn)["rows"] if r.get("campaign_id") == cid]


def test_a_target_with_no_actual_is_open(conn):
    """"Predicted metrics with no actuals." A number somebody aimed at is not a result."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 4.0},
                     metric_type="target", confirm=True)
    core.update_campaign(conn, campaign_id=cid,
                         tags=[{"value": "liked", "source": "stated"}])

    row = next(r for r in _queue(conn)["rows"] if r.get("campaign_id") == cid)

    assert "actuals never arrived" in row["why"].lower()


def test_an_unreconciled_judgment_is_open(conn):
    """"An evaluation whose predictions were never reconciled" — §9.9's loop, surfaced where
    somebody is already giving feedback."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    core.update_campaign(conn, campaign_id=cid, tags=[
        {"value": "liked", "source": "stated"},
        {"value": "performed_well", "source": "verified"}])
    core.save_evaluation(conn, subject_title="Colombia", campaign_id=cid, verdict="approve",
                         summary="It should work.", findings=[])

    row = next(r for r in _queue(conn)["rows"] if r.get("campaign_id") == cid)

    assert "never been checked" in row["why"].lower() or "reconcil" in row["why"].lower()


def test_a_superseding_version_with_no_verdict_is_open(conn):
    """"A superseding version received with no verdict" — the review's own top row, and the
    most valuable thing in the list."""
    v1 = _campaign(conn, "Colombia v1")
    core.save_evaluation(conn, subject_title="Colombia v1", campaign_id=v1, verdict="revise",
                         summary="The timeline is unrealistic.",
                         approve_if="The timeline is extended to ten weeks.",
                         findings=[{"severity": "blocking", "kind": "missing_information",
                                    "finding": "The six-week timeline is unrealistic",
                                    "fix": "Extend it"}])
    v2 = core.ingest_campaign(conn, title="Colombia v2", market="LATAM", status="proposed",
                              detail="The revised plan.", supersedes=v1)["campaign_id"]

    row = next(r for r in _queue(conn)["rows"] if r.get("campaign_id") == v2)

    assert "nobody has said whether it fixed anything" in row["why"].lower()


def test_the_most_valuable_row_comes_first(conn):
    """"Order by what is most valuable to capture, not alphabetically or by date — a rejected
    brief whose v2 has just landed is worth more than a stub nobody has briefed.\""""
    stub = core.ingest_campaign(conn, title="A stub", record_type="stub",
                                market="LATAM")["campaign_id"]
    plain = _campaign(conn, "An ordinary campaign")
    v1 = _campaign(conn, "Colombia v1", status="proposed")
    core.save_evaluation(conn, subject_title="Colombia v1", campaign_id=v1, verdict="revise",
                         summary="The timeline is unrealistic.",
                         approve_if="The timeline is extended to ten weeks.",
                         findings=[{"severity": "blocking", "kind": "missing_information",
                                    "finding": "The six-week timeline is unrealistic",
                                    "fix": "Extend it"}])
    v2 = core.ingest_campaign(conn, title="Colombia v2", market="LATAM", status="proposed",
                              detail="The revised plan.", supersedes=v1)["campaign_id"]

    ordered = [r.get("campaign_id") for r in _queue(conn)["rows"]]

    assert ordered[0] == v2, "the re-brief awaiting a verdict"
    assert ordered.index(plain) < ordered.index(stub), "a real campaign beats a stub"


def test_every_row_says_why_it_is_open(conn):
    """"Every row says why it is open, because 'what does this want from me' is the question
    the user actually has.\""""
    for n in range(3):
        _campaign(conn, f"Campaign {n}")

    for row in _queue(conn)["rows"]:
        if row.get("campaign_id"):
            assert row["why"], row


def test_an_empty_library_says_so_rather_than_showing_an_empty_menu(conn):
    out = _queue(conn)

    assert out["status"] == "nothing_to_check"
    assert "nothing is waiting" in out["what_it_means"].lower()


# ── 10.2: the numbers do not move ──────────────────────────────────────────

def test_campaigns_fill_one_to_nine(conn):
    for n in range(12):
        _campaign(conn, f"Campaign {n}")

    rows = _rows(_queue(conn))

    assert [n for n in rows if rows[n].get("campaign_id")] == list(range(1, 10))


def test_ten_is_always_the_concluded_menu_and_eleven_is_always_more(conn):
    """"Whether there are three open campaigns or nine. Floating positions break the habit the
    menu exists to create, and a small gap in the numbering is a cheaper price than a moving
    target.\""""
    for count in (1, 3, 9, 15):
        fresh = store.connect()
        store.upgrade(fresh)
        for n in range(count):
            _campaign(fresh, f"Campaign {n}")
        rows = _rows(_queue(fresh))
        assert rows[10]["action"] == "concluded", count
        assert rows[11]["action"] == "more", count
        fresh.close()


def test_the_gap_is_left_rather_than_closed(conn):
    """Three open campaigns and the menu still puts "concluded" at 10, not at 4."""
    for n in range(3):
        _campaign(conn, f"Campaign {n}")

    rows = _rows(_queue(conn))

    assert set(rows) == {1, 2, 3, 10, 11}


def test_show_more_pages_without_reshuffling(conn):
    """"Option 11 pages, keeping its state so going back and forth does not reshuffle
    anything.\""""
    for n in range(15):
        _campaign(conn, f"Campaign {n:02d}")
    first = _queue(conn)
    page_two = _queue(conn, page=2)

    assert [r["campaign_id"] for r in page_two["rows"] if r.get("campaign_id")] != \
           [r["campaign_id"] for r in first["rows"] if r.get("campaign_id")]
    assert _queue(conn)["rows"] == first["rows"], "going back shows the same page"


def test_the_concluded_menu_is_the_same_menu_over_concluded_records(conn):
    """"Option 10 opens the same menu over concluded records.\""""
    done = _campaign(conn, "Finished")
    core.add_metrics(conn, campaign_id=done, structured={"roas": 3.4}, confirm=True)
    core.update_campaign(conn, campaign_id=done, tags=[
        {"value": "liked", "source": "stated"},
        {"value": "performed_well", "source": "verified"}])

    out = _queue(conn, scope="concluded")

    assert done in [r.get("campaign_id") for r in out["rows"]]


# ── 10.4: the server builds it ─────────────────────────────────────────────

def test_the_same_library_gives_the_same_menu(conn):
    """"If the model composes the list, two users see different orderings and different
    numbering for the same library — the exact class of variance the consistency work is
    trying to remove.\""""
    for n in range(5):
        _campaign(conn, f"Campaign {n}")

    first, second = _queue(conn), _queue(conn)

    assert [r["number"] for r in first["rows"]] == [r["number"] for r in second["rows"]]
    assert [r.get("campaign_id") for r in first["rows"]] == \
           [r.get("campaign_id") for r in second["rows"]]
    assert first["menu_token"] == second["menu_token"]


def test_the_rows_carry_what_the_model_needs_to_render_them(conn):
    cid = _campaign(conn, "Colombia - Calle 82 Bogota")

    row = next(r for r in _queue(conn)["rows"] if r.get("campaign_id") == cid)

    assert row["title"] == "Colombia - Calle 82 Bogota"
    assert row["number"] == 1
    assert row["why"]


# ── 10.5: a number is not an identifier ────────────────────────────────────

def test_a_stale_token_refreshes_instead_of_writing(conn):
    """"If anything changes between rendering the menu and the user answering, '3' silently
    means a different campaign and feedback lands on the wrong record." The whole class of
    silent data corruption, made impossible."""
    first = _campaign(conn, "First")
    menu = _queue(conn)
    _campaign(conn, "Colombia v2 — just landed", status="proposed")

    out = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    assert out["status"] == "refreshed"
    assert out["menu_token"] != menu["menu_token"]
    assert "changed" in out["what_it_means"].lower()
    assert store.get_campaign(conn, first)["tags"] == [], "and nothing was written"


def test_a_current_token_selects_the_row_the_user_saw(conn):
    cid = _campaign(conn, "Colombia")
    menu = _queue(conn)

    out = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    assert out["status"] == "asking"
    assert out["campaign_id"] == cid


def test_a_number_nobody_offered_is_refused(conn):
    _campaign(conn, "Colombia")
    menu = _queue(conn)

    with pytest.raises(ValueError) as e:
        feedback.choose(conn, menu_token=menu["menu_token"], choice=7)
    assert "7" in str(e.value)


def test_a_token_from_nowhere_is_refused(conn):
    _campaign(conn, "Colombia")

    out = feedback.choose(conn, menu_token="not-a-token", choice=1)

    assert out["status"] == "refreshed"


# ── 10.3: numbered all the way down ────────────────────────────────────────

def test_choosing_a_campaign_asks_the_next_question_as_numbers(conn):
    """"Once a campaign is chosen, the follow-ups are already closed sets — so ask them the
    same way.\""""
    _campaign(conn, "Colombia")
    menu = _queue(conn)

    out = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    reaction = out["questions"][0]
    assert reaction["options"] == {1: "liked", 2: "not_liked", 3: "mixed_reaction"}
    performance = out["questions"][1]
    assert performance["options"] == {1: "performed_well", 2: "underperformed",
                                      3: "performed_as_expected", 4: "no_data_yet"}


def test_only_the_questions_this_campaign_needs_are_asked(conn):
    """A campaign that already carries a reaction is not asked for one again — the menu exists
    to collect what is missing."""
    cid = _campaign(conn, "Colombia")
    core.update_campaign(conn, campaign_id=cid,
                         tags=[{"value": "liked", "source": "stated"}])
    menu = _queue(conn)

    out = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    assert [q["field"] for q in out["questions"]] == ["performance"]


def test_answers_are_recorded_as_tags(conn):
    cid = _campaign(conn, "Colombia")
    menu = _queue(conn)
    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    saved = feedback.record(conn, menu_token=asked["menu_token"], choice=1,
                            reaction=2, performance=4, said_by="R. Vega")

    values = {t["value"] for t in store.get_campaign(conn, cid)["tags"]}
    assert values == {"not_liked", "no_data_yet"}
    assert saved["status"] == "recorded"


def test_free_text_is_invited_and_never_required(conn):
    """"The free-text box is where 'slide 23 should be the standard' gets captured — the
    highest-value sentence in the whole system — so it should be invited, never required.\""""
    cid = _campaign(conn, "Colombia")
    menu = _queue(conn)
    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    assert asked["free_text"]["required"] is False
    assert "skip" in asked["free_text"]["prompt"].lower()

    saved = feedback.record(conn, menu_token=asked["menu_token"], choice=1,
                            reaction=1, performance=1, said_by="R. Vega")
    assert saved["status"] == "recorded"


def test_the_sentence_somebody_typed_is_kept_as_theirs(conn):
    cid = _campaign(conn, "Colombia")
    menu = _queue(conn)
    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    feedback.record(conn, menu_token=asked["menu_token"], choice=1, reaction=1,
                    performance=1, said_by="R. Vega",
                    note="Slide 23 should be the standard for every market.")

    record = store.get_campaign(conn, cid)
    assert "Slide 23" in json_of(record)


def json_of(record) -> str:
    import json
    return json.dumps(record, default=str)


def test_a_performance_tag_without_measurements_is_not_verified(conn):
    """§2.3's rule reaches the menu too: a claim somebody typed is `stated` until numbers back
    it, and the menu is the easiest possible place to type one."""
    cid = _campaign(conn, "Colombia")
    menu = _queue(conn)
    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    feedback.record(conn, menu_token=asked["menu_token"], choice=1, reaction=1,
                    performance=1, said_by="R. Vega")

    tags = {t["value"]: t["source"] for t in store.get_campaign(conn, cid)["tags"]}
    assert tags["performed_well"] == "stated"


def test_recording_needs_a_person(conn):
    cid = _campaign(conn, "Colombia")
    menu = _queue(conn)
    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    with pytest.raises(ValueError) as e:
        feedback.record(conn, menu_token=asked["menu_token"], choice=1,
                        reaction=1, performance=1, said_by="")
    assert "people" in str(e.value).lower() or "person" in str(e.value).lower()


def test_a_stale_token_never_writes_feedback(conn):
    cid = _campaign(conn, "Colombia")
    menu = _queue(conn)
    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)
    _campaign(conn, "Something else arrived")

    out = feedback.record(conn, menu_token=asked["menu_token"], choice=1,
                          reaction=1, performance=1, said_by="R. Vega")

    assert out["status"] == "refreshed"
    assert store.get_campaign(conn, cid)["tags"] == []


# ── 10.6: offered, not waited for ──────────────────────────────────────────

def test_an_upload_says_how_many_are_waiting(conn):
    """"The menu is worthless if the user has to know it exists." Same instinct as `gaps()`:
    the library knows what it is missing and should say so."""
    for n in range(3):
        _campaign(conn, f"Campaign {n}")

    out = core.ingest_campaign(conn, title="A new one", market="LATAM", status="concluded",
                               detail="Something new.")

    offers = [a for a in out.get("next_actions") or [] if a["tool"] == "feedback_queue"]
    assert offers, out.get("next_actions")
    assert "waiting" in offers[0]["label"].lower()


def test_a_judgment_says_it_too(conn):
    for n in range(3):
        _campaign(conn, f"Campaign {n}")
    cid = _campaign(conn, "Judged")

    out = core.save_evaluation(conn, subject_title="Judged", campaign_id=cid,
                               verdict="approve", summary="Fine.", findings=[])

    assert any(a["tool"] == "feedback_queue" for a in out.get("next_actions") or [])


def test_the_first_thing_a_session_sees_mentions_it(conn):
    for n in range(7):
        _campaign(conn, f"Campaign {n}")

    out = core.readiness(conn)

    assert "7 campaign" in str(out)
    assert any(a["tool"] == "feedback_queue" for a in out.get("next_actions") or [])


def test_a_library_with_nothing_waiting_offers_nothing(conn):
    """An offer that is always there stops being read — the rule every other offer in this
    product follows."""
    out = core.ingest_campaign(conn, title="The only one", market="LATAM",
                               status="concluded", detail="Something.")

    assert not [a for a in out.get("next_actions") or [] if a["tool"] == "feedback_queue"]


def test_it_reaches_the_model_over_the_protocol(conn):
    import asyncio
    import json

    import mcp_server

    async def call(name, args):
        return json.loads((await mcp_server.mcp.call_tool(name, args)).content[0].text)

    cid = _campaign(conn, "Colombia")
    menu = asyncio.run(call("feedback_queue", {}))
    assert menu["rows"][0]["campaign_id"] == cid

    asked = asyncio.run(call("feedback_choose", {"menu_token": menu["menu_token"],
                                                 "choice": 1}))
    assert asked["status"] == "asking"

    saved = asyncio.run(call("feedback_record", {
        "menu_token": asked["menu_token"], "choice": 1, "reaction": 1,
        "performance": 4, "said_by": "R. Vega"}))
    assert saved["status"] == "recorded"


# ── the promises that were stated and not kept ─────────────────────────────

def test_the_name_it_demands_is_actually_kept(conn):
    """`said_by` was required with a paragraph about why, then dropped on the dominant path —
    two numbers and no note — while the response said "attributed to R. Vega". A false
    provenance claim in a product whose whole thesis is that unfounded claims are the enemy."""
    cid = _campaign(conn, "Colombia")
    menu = _queue(conn)
    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    feedback.record(conn, menu_token=asked["menu_token"], choice=1,
                    reaction=1, performance=2, said_by="Dana Okafor")

    tags = store.get_campaign(conn, cid)["tags"]
    assert all(t["said_by"] == "Dana Okafor" for t in tags), tags
    assert all(t["said_at"] for t in tags)


def test_the_library_cannot_sign_feedback_as_itself(conn):
    """§9.8 built this guard one phase earlier and Phase 10 did not reuse it — while telling
    the model in the same payload that "answering should never need typing", which is an
    invitation to fill the field in."""
    _campaign(conn, "Colombia")
    menu = _queue(conn)
    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    for name in ("the library", "Campaign Intelligence", "Claude", "n/a", "the system"):
        with pytest.raises(ValueError) as e:
            feedback.record(conn, menu_token=asked["menu_token"], choice=1,
                            reaction=1, performance=2, said_by=name)
        assert "person" in str(e.value).lower(), name


def test_a_write_cannot_land_on_a_row_the_user_never_saw(conn):
    """The token proved the MENU was fresh and proved nothing about the ROW. A valid token
    plus any campaign id the model had lying about from another tool wrote feedback to a
    record nobody chose, and returned "recorded"."""
    _campaign(conn, "Chosen")
    elsewhere = _campaign(conn, "Never chosen")
    menu = _queue(conn)
    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    # The write takes the NUMBER, not an id — there is no way to name a row off the menu.
    import inspect
    assert "campaign_id" not in inspect.signature(feedback.record).parameters

    feedback.record(conn, menu_token=asked["menu_token"], choice=1, reaction=1,
                    performance=2, said_by="R. Vega")
    assert store.get_campaign(conn, elsewhere)["tags"] == []


def test_a_number_off_this_menu_is_refused_at_the_write_too(conn):
    _campaign(conn, "Colombia")
    menu = _queue(conn)
    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    with pytest.raises(ValueError):
        feedback.record(conn, menu_token=asked["menu_token"], choice=8, reaction=1,
                        performance=2, said_by="R. Vega")


def test_a_note_never_turns_into_a_measured_outcome(conn):
    """`performance=4` is "no data yet" — and 4 is truthy, so the sentence was filed as an
    ACTUAL metric row. One call wrote "no data yet" and "this campaign has measured results"
    onto the same record, which then closed it out of the queue, told `readiness` the library
    was measured, and opened §2.3's `verified` gate. All from somebody typing a sentence."""
    cid = _campaign(conn, "Colombia")
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 4.0},
                     metric_type="target", confirm=True)
    menu = _queue(conn)
    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    feedback.record(conn, menu_token=asked["menu_token"], choice=1, reaction=3,
                    performance=4, said_by="R. Vega",
                    note="Slide 23 should be the standard for every market.")

    record = store.get_campaign(conn, cid)
    assert record["has_actual_metrics"] is False, "a sentence is not a measurement"
    assert any("actuals never arrived" in r["why"]
               for r in _queue(conn)["rows"] if r.get("campaign_id") == cid), (
        "and the queue keeps asking for the thing it was built to ask for")


# ── the menu the review actually drew ──────────────────────────────────────

def test_the_server_builds_the_line_not_just_the_fields(conn):
    """If the server builds the menu it should build the LINE. A model handed a `market`
    field, a title and 140 characters of semicolon-joined prose, told to "read the rows out",
    will compose — and composing is what §10.4 forbids."""
    cid = _campaign(conn, "Calle 82 Bogota", market="Colombia")

    row = next(r for r in _queue(conn)["rows"] if r.get("campaign_id") == cid)

    assert row["line"].startswith("Colombia - Calle 82 Bogota")
    assert len(row["line"]) < 80, row["line"]


def test_a_new_version_says_so_on_its_own_line(conn):
    """The review's row 1 reads "Colombia - Calle 82 Bogota (v2 received)". `supersedes` is in
    hand when the row is built and was dropped."""
    v1 = _campaign(conn, "Calle 82 Bogota", market="Colombia", status="proposed")
    v2 = core.ingest_campaign(conn, title="Calle 82 Bogota", market="Colombia",
                              status="proposed", detail="The revision.",
                              supersedes=v1)["campaign_id"]

    row = next(r for r in _queue(conn)["rows"] if r.get("campaign_id") == v2)

    assert "(v2 received)" in row["line"]


def test_the_reason_is_as_short_as_the_review_wrote_it(conn):
    """"no outcome recorded" is nineteen characters. Two reasons concatenated with a semicolon
    is not a menu column."""
    cid = _campaign(conn, "Keke Palmer seeding event", market="Australia")

    row = next(r for r in _queue(conn)["rows"] if r.get("campaign_id") == cid)

    assert row["reason"] == "no outcome recorded"
    assert ";" not in row["reason"]


# ── the questions a row actually needs ─────────────────────────────────────

def test_a_stub_is_not_asked_how_it_performed(conn):
    """The row says nobody briefed it and the next sentence asked how it performed. `_ask_about`
    recomputed from tags and never read the row's own `needs`."""
    core.ingest_campaign(conn, title="A stub", record_type="stub", market="LATAM")
    menu = _queue(conn)

    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    assert asked["questions"] == []
    assert asked["next_actions"][0]["tool"] == "upload_campaign"


def test_a_campaign_that_has_not_run_is_not_asked_how_it_performed(conn):
    """A store opening that has not happened cannot have performed. `readiness` draws this
    line explicitly; the queue did not draw it at all."""
    _campaign(conn, "Dubai Mall store opening", status="proposed")
    menu = _queue(conn)

    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    assert [q["field"] for q in asked["questions"]] == ["reaction"]


def test_an_unrun_campaign_is_not_listed_as_missing_its_outcome(conn):
    cid = _campaign(conn, "Dubai Mall store opening", status="proposed")

    row = next(r for r in _queue(conn)["rows"] if r.get("campaign_id") == cid)

    assert "performed" not in row["why"]


# ── the sentence, and where it should go ───────────────────────────────────

def test_the_sentence_is_offered_to_the_corrections_loop(conn):
    """§8.6 turns "slide 23 should be the standard" into a standing rule once it recurs and
    somebody confirms it. The product offers that for words it SCRAPED out of a deck, and did
    not offer it for the one place a person is deliberately asked for the sentence — where the
    provenance is better than the scraped path ever gets."""
    cid = _campaign(conn, "Colombia")
    menu = _queue(conn)
    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    saved = feedback.record(conn, menu_token=asked["menu_token"], choice=1, reaction=1,
                            performance=2, said_by="R. Vega",
                            note="Slide 23 should be the standard for every market.")

    offers = [a for a in saved.get("next_actions") or [] if a["tool"] == "note_correction"]
    assert offers, saved.get("next_actions")
    assert offers[0]["prefilled_args"]["text"] == (
        "Slide 23 should be the standard for every market.")
    assert "R. Vega" in offers[0]["prefilled_args"]["provenance"]


def test_no_sentence_means_no_offer(conn):
    cid = _campaign(conn, "Colombia")
    menu = _queue(conn)
    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    saved = feedback.record(conn, menu_token=asked["menu_token"], choice=1, reaction=1,
                            performance=2, said_by="R. Vega")

    assert not [a for a in saved.get("next_actions") or []
                if a["tool"] == "note_correction"]


# ── option 11 has to go somewhere ──────────────────────────────────────────

def test_the_last_page_wraps_rather_than_dead_ending(conn):
    """Row 11 reads "start again" on the last page and did not: `page + 1` clamped back to the
    same page, so a user at the end had no way home from inside the menu."""
    for n in range(12):
        _campaign(conn, f"Campaign {n:02d}")
    last = _queue(conn, page=2)
    assert last["page"] == last["pages"]

    out = feedback.choose(conn, menu_token=last["menu_token"], choice=11)

    assert out["page"] == 1


def test_the_concluded_menu_leads_with_what_it_was_opened_for(conn):
    """Its own docstring argues it is a different population because somebody picking 10 wants
    to talk about something that has already run — usually to add a sentence to a campaign
    that is otherwise complete. Sorting those to the bottom answered the other question."""
    done = _campaign(conn, "Fully recorded")
    core.add_metrics(conn, campaign_id=done, structured={"roas": 3.4}, confirm=True)
    core.update_campaign(conn, campaign_id=done, tags=[
        {"value": "liked", "source": "stated"},
        {"value": "performed_well", "source": "verified"}])
    _campaign(conn, "Still missing things")

    rows = _queue(conn, scope="concluded")["rows"]

    assert rows[0]["campaign_id"] == done


# ── the sentence needs its own home ────────────────────────────────────────

def test_a_note_does_not_reopen_a_finished_campaign(conn):
    """Filed as a metrics row the note IS the target, so a complete campaign given one
    sentence came back as "targets only, no actuals" — permanently, because only real actuals
    clear it. The highest-value sentence in the system re-opened the record it was praising."""
    cid = _campaign(conn, "Finished")
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    core.update_campaign(conn, campaign_id=cid, tags=[
        {"value": "liked", "source": "stated"},
        {"value": "performed_well", "source": "verified"}])
    menu = _queue(conn, scope="concluded")

    feedback.record(conn, menu_token=menu["menu_token"], choice=1, said_by="R. Vega",
                    note="Slide 23 should be the standard for every market.")

    assert _queue(conn)["status"] == "nothing_to_check"
    assert store.get_campaign(conn, cid)["has_actual_metrics"] is True


def test_a_note_is_not_a_metric_of_any_kind(conn):
    """A metrics row means "a number about outcomes". Prose with `structured: NULL` in that
    table unlocked §2.3's `verified` gate, entered §9.9's calibration denominator, and was
    served by `reconcile_evaluation` as the campaign's actual result."""
    cid = _campaign(conn, "Colombia")
    menu = _queue(conn)
    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    feedback.record(conn, menu_token=asked["menu_token"], choice=1, reaction=1,
                    performance=1, said_by="R. Vega",
                    note="Slide 23 should be the standard.")

    record = store.get_campaign(conn, cid)
    assert record["metrics"] == []
    assert record["has_actual_metrics"] is False
    notes = store.feedback_notes(conn, cid)
    assert notes[0]["note"] == "Slide 23 should be the standard."
    assert notes[0]["said_by"] == "R. Vega"
    assert notes[0]["said_at"]


def test_the_note_is_visible_on_the_record(conn):
    """It is the most valuable sentence in the system; a table nothing reads is where it goes
    to die."""
    cid = _campaign(conn, "Colombia")
    menu = _queue(conn)
    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)
    feedback.record(conn, menu_token=asked["menu_token"], choice=1, reaction=1,
                    performance=1, said_by="R. Vega", note="Slide 23 is the standard.")

    assert store.get_campaign(conn, cid)["feedback_notes"][0]["said_by"] == "R. Vega"


# ── a menu that has gone stale in a way the token could not see ────────────

def test_a_row_that_gained_a_tag_invalidates_the_menu(conn):
    """The hash covered [number, campaign_id, action] only, so a row that gained a tag but
    stayed open kept its token — and an answer composed before that landed wrote `liked` on a
    campaign somebody had just marked `not_liked`."""
    cid = _campaign(conn, "Colombia")
    menu = _queue(conn)

    core.update_campaign(conn, campaign_id=cid,
                         tags=[{"value": "liked", "source": "stated"}])

    out = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)
    assert out["status"] == "refreshed"


def test_an_answer_to_a_question_nobody_asked_is_refused(conn):
    """`choose` asked only for a reaction; `record` accepted a performance answer anyway and
    wrote `no_data_yet` beside a `performed_well` that measurements had earned."""
    cid = _campaign(conn, "Colombia")
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    core.update_campaign(conn, campaign_id=cid, tags=[
        {"value": "performed_well", "source": "verified"}])
    menu = _queue(conn)
    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)
    assert [q["field"] for q in asked["questions"]] == ["reaction"]

    with pytest.raises(ValueError) as e:
        feedback.record(conn, menu_token=asked["menu_token"], choice=1, reaction=1,
                        performance=4, said_by="R. Vega")
    assert "not asked" in str(e.value).lower()


def test_a_second_opinion_replaces_the_first_within_its_family(conn):
    """Appending produced `liked` AND `not_liked` on one record, which the queue then treated
    as closed forever. One person's answer to one question is one answer."""
    cid = _campaign(conn, "Colombia")
    menu = _queue(conn)
    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)
    feedback.record(conn, menu_token=asked["menu_token"], choice=1, reaction=1,
                    performance=1, said_by="R. Vega")

    concluded = _queue(conn, scope="concluded")
    feedback.record(conn, menu_token=concluded["menu_token"], choice=1, reaction=2,
                    said_by="J. Lin")

    values = {t["value"]: t for t in store.get_campaign(conn, cid)["tags"]}
    assert "liked" not in values
    assert values["not_liked"]["said_by"] == "J. Lin"


# ── the menu names the tool that closes each row ───────────────────────────

def test_a_row_the_numbers_cannot_close_offers_the_tool_that_can(conn):
    """Ranks 1, 2 and 5 need `save_evaluation`, `reconcile_evaluation` and `add_metrics` —
    tools the menu never named. The top-ranked reason in the whole queue could not be cleared
    from the queue, which is D116 arriving through the queue's own rows."""
    v1 = _campaign(conn, "Colombia v1", status="proposed")
    v2 = core.ingest_campaign(conn, title="Colombia v2", market="LATAM", status="proposed",
                              detail="The revision.", supersedes=v1)["campaign_id"]
    menu = _queue(conn)

    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    assert asked["campaign_id"] == v2
    tools = [a["tool"] for a in asked.get("next_actions") or []]
    assert "prepare_evaluation" in tools or "save_evaluation" in tools


def test_a_targets_only_row_offers_the_numbers_it_wants(conn):
    cid = _campaign(conn, "Colombia")
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 4.0},
                     metric_type="target", confirm=True)
    core.update_campaign(conn, campaign_id=cid, tags=[
        {"value": "liked", "source": "stated"},
        {"value": "performed_well", "source": "stated"}])
    menu = _queue(conn)

    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    assert any(a["tool"] == "add_metrics" for a in asked.get("next_actions") or [])


# ── the token, and the cost of checking it ─────────────────────────────────

def test_a_token_names_its_own_page_so_checking_it_is_one_lookup(conn):
    """Validation re-derived every page of every scope: 8.8s on 500 campaigns, and the STALE
    path — the common one, since any change invalidates every open menu — was the slowest.
    Page 50 and beyond were unreachable entirely, and a valid token there was reported as
    "the library has changed"."""
    for n in range(60):
        _campaign(conn, f"Campaign {n:02d}")
    last = _queue(conn, page=6)
    assert last["pages"] >= 6

    out = feedback.choose(conn, menu_token=last["menu_token"], choice=1)

    assert out["status"] == "asking"


def test_a_page_beyond_the_old_ceiling_still_works(conn):
    for n in range(95):
        _campaign(conn, f"Campaign {n:02d}")
    deep = _queue(conn, page=10)

    assert feedback.choose(conn, menu_token=deep["menu_token"], choice=1)["status"] == "asking"


# ── the remaining small honesty breaks ─────────────────────────────────────

def test_an_empty_answer_is_not_reported_as_a_write(conn):
    cid = _campaign(conn, "Colombia")
    menu = _queue(conn)
    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    with pytest.raises(ValueError) as e:
        feedback.record(conn, menu_token=asked["menu_token"], choice=1, said_by="R. Vega")
    assert "nothing" in str(e.value).lower()


def test_choosing_ten_on_an_empty_concluded_list_says_nothing_to_check(conn):
    _campaign(conn, "Colombia", status="proposed")
    menu = _queue(conn)

    out = feedback.choose(conn, menu_token=menu["menu_token"], choice=10)

    assert out["status"] == "nothing_to_check"


def test_a_judgment_does_not_advertise_the_record_it_just_judged(conn):
    """`apart_from` exists to stop a write announcing itself, and the judgment path did not
    pass it."""
    cid = _campaign(conn, "The only one")
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    core.update_campaign(conn, campaign_id=cid, tags=[
        {"value": "liked", "source": "stated"},
        {"value": "performed_well", "source": "verified"}])

    out = core.save_evaluation(conn, subject_title="The only one", campaign_id=cid,
                               verdict="approve", summary="Fine.", findings=[])

    assert not [a for a in out.get("next_actions") or [] if a["tool"] == "feedback_queue"]


def test_a_write_that_opens_a_row_offers_the_queue(conn):
    """Recording a target opens a row and offered nothing; so did tagging. The queue is only
    worth building if the writes that fill it also name it."""
    for n in range(2):
        _campaign(conn, f"Other {n}")
    cid = _campaign(conn, "Colombia")
    core.update_campaign(conn, campaign_id=cid, tags=[
        {"value": "liked", "source": "stated"},
        {"value": "performed_well", "source": "stated"}])

    out = core.add_metrics(conn, campaign_id=cid, structured={"roas": 4.0},
                           metric_type="target", confirm=True)

    assert any(a["tool"] == "feedback_queue" for a in out.get("next_actions") or [])


def test_rows_of_equal_worth_are_ordered_by_name(conn):
    """Falling back to the campaign id leaves rows in uuid order — deterministic, and
    unreadable. Within a rank the user is scanning for a name they recognise."""
    for title in ("Zanzibar", "Acapulco", "Mumbai"):
        _campaign(conn, title)

    listed = [r["title"] for r in _queue(conn)["rows"] if r.get("campaign_id")]

    assert listed == ["Acapulco", "Mumbai", "Zanzibar"]


def test_the_sentence_is_kept_even_when_no_numbers_are_given(conn):
    """Option 10's commonest use: nothing outstanding, and somebody has a sentence."""
    cid = _campaign(conn, "Finished")
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    core.update_campaign(conn, campaign_id=cid, tags=[
        {"value": "liked", "source": "stated"},
        {"value": "performed_well", "source": "verified"}])
    menu = _queue(conn, scope="concluded")

    saved = feedback.record(conn, menu_token=menu["menu_token"], choice=1,
                            said_by="R. Vega",
                            note="Slide 23 should be the standard for every market.")

    assert saved["note_kept"] is True
    assert store.feedback_notes(conn, cid)[0]["said_by"] == "R. Vega"


# ── D20 / D70 / D87: what else "needs something from a human" means ─────────

def test_a_notice_addressed_to_a_person_is_a_queue_row(conn):
    """D20: a persisted `blocked` or `degraded` notice is by definition something needing a
    human — that is 10.1's own definition of open, and the notices were sitting in their own
    table where nobody looked."""
    cid = _campaign(conn, "Colombia")
    store.record_notice(conn, code="visual_search_offline", campaign_id=cid,
                        detail="CLIP weights are missing on this machine.")

    row = next(r for r in _queue(conn)["rows"] if r.get("campaign_id") == cid)

    assert "needs_attention" in row["needs"]
    assert row["reason"] == "something needs attention"


def test_a_notice_outranks_a_missing_tag(conn):
    """Something is broken, and a missing reaction tag is not."""
    quiet = _campaign(conn, "A quiet one")
    broken = _campaign(conn, "Zebra — but broken")
    store.record_notice(conn, code="visual_search_offline", campaign_id=broken,
                        detail="CLIP weights are missing.")

    ordered = [r.get("campaign_id") for r in _queue(conn)["rows"]]

    assert ordered.index(broken) < ordered.index(quiet)


def test_a_cleared_notice_leaves_the_queue(conn):
    cid = _campaign(conn, "Colombia")
    nid = store.record_notice(conn, code="visual_search_offline", campaign_id=cid,
                              detail="CLIP weights are missing.")
    core.update_campaign(conn, campaign_id=cid, tags=[
        {"value": "liked", "source": "stated"},
        {"value": "performed_well", "source": "stated"}])
    assert [r for r in _queue(conn)["rows"] if r.get("campaign_id") == cid]

    store.clear_notice(conn, nid)

    assert not [r for r in _queue(conn)["rows"] if r.get("campaign_id") == cid]


def test_a_measured_campaign_whose_claim_is_unverified_is_open(conn):
    """D87: nothing guided a marketer to mark a measured campaign `verified`, so §6.4 reported
    "could not be checked" on libraries that held the evidence. The numbers are there and only
    a person can say the claim rests on them."""
    cid = _campaign(conn, "Colombia")
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    core.update_campaign(conn, campaign_id=cid, tags=[
        {"value": "liked", "source": "stated"},
        {"value": "performed_well", "source": "stated"}])

    row = next(r for r in _queue(conn)["rows"] if r.get("campaign_id") == cid)

    assert "unverified_claim" in row["needs"]
    assert row["reason"] == "results on file, claim not verified"


def test_confirming_the_claim_closes_that_row(conn):
    cid = _campaign(conn, "Colombia")
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    core.update_campaign(conn, campaign_id=cid, tags=[
        {"value": "liked", "source": "stated"},
        {"value": "performed_well", "source": "stated"}])
    menu = _queue(conn)

    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=1)

    assert any(a["tool"] == "update_campaign" and
               "verified" in str(a.get("prefilled_args") or a)
               for a in asked.get("next_actions") or []), asked.get("next_actions")


def test_a_market_resting_on_one_campaign_is_open(conn):
    """D70: a `single_example` cell needs something from a human — another campaign in that
    market — and the similarity score looks the same whether it came from one example or ten."""
    for n in range(4):
        other = _campaign(conn, f"LATAM {n}", market="LATAM")
        core.add_metrics(conn, campaign_id=other, structured={"roas": 3.4}, confirm=True)
        core.update_campaign(conn, campaign_id=other, tags=[
            {"value": "liked", "source": "stated"},
            {"value": "performed_well", "source": "verified"}])
    only = _campaign(conn, "The only APAC one", market="APAC")
    core.add_metrics(conn, campaign_id=only, structured={"roas": 3.4}, confirm=True)
    core.update_campaign(conn, campaign_id=only, tags=[
        {"value": "liked", "source": "stated"},
        {"value": "performed_well", "source": "verified"}])

    row = next(r for r in _queue(conn)["rows"] if r.get("campaign_id") == only)

    assert "single_example" in row["needs"]
    assert row["reason"] == "this market rests on it alone"


def test_a_market_with_two_campaigns_is_not(conn):
    for n in range(5):
        cid = _campaign(conn, f"APAC {n}", market="APAC")
        core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
        core.update_campaign(conn, campaign_id=cid, tags=[
            {"value": "liked", "source": "stated"},
            {"value": "performed_well", "source": "verified"}])

    assert _queue(conn)["status"] == "nothing_to_check"


def test_a_degraded_upload_stays_on_the_record(conn, tmp_path, monkeypatch):
    """A warning lives for exactly one response, so the thing the library most wanted somebody
    to act on was the thing it forgot fastest. D20's whole point."""
    import clip_embed

    monkeypatch.setattr(clip_embed, "embed_image",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no weights")))
    from PIL import Image
    path = tmp_path / "hero.png"
    Image.new("RGB", (32, 32), (200, 30, 30)).save(path)

    out = core.ingest_campaign(conn, title="Colombia", market="LATAM", status="concluded",
                               detail="A launch.", asset_ref={"path": str(path)})

    notices = store.open_notices(conn, out["campaign_id"])
    assert notices, out.get("warnings")
    row = next(r for r in _queue(conn)["rows"]
               if r.get("campaign_id") == out["campaign_id"])
    assert "needs_attention" in row["needs"]
