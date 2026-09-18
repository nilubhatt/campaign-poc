"""
§11.1 and §11.2 — who actually said this, and how much that claim is worth.

**11.1 Environment-derived author** — `os_user`, `host`, config display name.
**11.2 Reuse `verified` vs `stated`** for author rather than inventing a vocabulary; add
`method` (stdio_local | sso | import).

They are one mechanism and ship together, because 11.1 without 11.2 is the defect this
product exists to refuse: a claim about a person with nothing saying where it came from.

## What is wrong today

Nine write paths take a person's name as a free string — `said_by`, `recorded_by`,
`stated_by`, `withdrawn_by`, `confirmed_by`, `linked_by` — and the server has no idea what it
is receiving. D105 says it plainly, about `confirmed_by`: *"nothing distinguishes a real
confirmation from one the model wrote itself."*

That is not a small gap, because of what these fields DO. `confirmed_by` promotes a rule to
the checklist, where it is applied to every future brief in its markets. `said_by` turns an
impression into a `stated` tag that later judgments weigh. `withdrawn_by` takes a finding off
the record. Every one of them is a claim that somebody made a decision, and the library
records it with the same confidence whether a marketer typed their name or a model filled in
a plausible one.

## The distinction that fixes it

The server knows exactly one thing for certain: which machine account made the call. It knows
nothing at all about whose opinion is being recorded. Those are different facts and the
product already has the vocabulary for that difference — `verified` for what it established
itself, `stated` for what somebody told it — so 11.2's instruction is to reuse it rather than
invent a second scheme.

  `captured_by`     the account that made the call. The SERVER derives it: `os_user` from the
                    operating system, `host`, and a display name if the operator configured
                    one. `verified`, with a `method` saying how — and `stdio_local` is a weak
                    verification that says so: it means "this is the desktop account running
                    the server", not "this person authenticated".

  `on_behalf_of`    whose opinion this is. Only a person can say, so it stays `stated`, and
                    it stays REQUIRED — deriving it would be the product inventing the one
                    fact it cannot know.

D35 is explicit that `TAG_SOURCE_SYNONYMS` must not be reused for this: "measured" and "from
metrics" are meaningless as identity claims, and a synonym table that accepts them would let
`source: measured` through on a statement about a person.
"""
import pytest

import core
import corrections
import identity
import store


def test_the_server_knows_which_account_made_the_call():
    """"Environment-derived author — `os_user`, `host`, config display name." Nothing here is
    typed by anybody, which is the entire point: it is the one fact about authorship the
    server can establish without being told."""
    who = identity.captured_by()

    assert who["os_user"]
    assert who["host"]
    assert who["source"] == "verified"


def test_what_the_server_verified_is_the_account_not_the_person():
    """The claim has to be exactly as strong as the evidence. `stdio_local` means "this is the
    desktop account the server is running as" — it is not an authentication, nobody proved
    anything, and a field reading `verified` with no method beside it would overstate that
    into "we checked who this is"."""
    who = identity.captured_by()

    assert who["method"] == "stdio_local"
    assert "not an authentication" in who["what_it_means"]


def test_a_configured_display_name_is_used_when_the_operator_set_one(monkeypatch):
    """An `os_user` of `nbhatt` is an account, not a name a colleague would recognise. The
    operator can say who that is, and then the library can too."""
    monkeypatch.setattr(identity.config, "OPERATOR_NAME", "Nilanjan Bhattacharya")

    who = identity.captured_by()

    assert who["display_name"] == "Nilanjan Bhattacharya"
    assert who["os_user"], "and the account is still recorded, not replaced by the label"


def test_a_display_name_nobody_configured_is_absent_rather_than_guessed(monkeypatch):
    """Not derived from `os_user` by title-casing it. "Nbhatt" is not a name, and a field that
    sometimes holds a real name and sometimes a prettied-up login is one nobody can read."""
    monkeypatch.setattr(identity.config, "OPERATOR_NAME", "")

    assert "display_name" not in identity.captured_by()


def test_the_method_says_how_over_http(monkeypatch):
    """`method` (stdio_local | sso | import) — the vocabulary 11.2 names. Over HTTP with a
    real identity provider the claim is genuinely stronger, and the field has to be able to
    say so, or the strongest and weakest cases read identically."""
    monkeypatch.setattr(identity, "_principal_for_this_call",
                        lambda: identity.auth.Principal(subject="r.vega@agency.example",
                                                        anonymous=False,
                                                        email="r.vega@agency.example"))

    who = identity.captured_by()

    assert who["method"] == "sso"
    assert who["subject"] == "r.vega@agency.example"


def test_an_anonymous_http_caller_is_not_reported_as_the_desktop_account(monkeypatch):
    """The failure this would otherwise have: the server running over HTTP reports the OS
    account it was STARTED as for every caller, so nine anonymous users on one deployment all
    record their decisions under the name of whoever launched it."""
    monkeypatch.setattr(identity, "_over_http", lambda: True)
    monkeypatch.setattr(identity, "_principal_for_this_call", lambda: identity.auth.ANONYMOUS)

    who = identity.captured_by()

    assert who["method"] == "unattributed"
    assert "os_user" not in who
    assert who["source"] == "stated", (
        "nothing was verified about this caller, so the field cannot say `verified`"
    )


# ── 11.2: the vocabulary, and what it refuses ──────────────────────────────

def test_who_said_it_carries_both_halves():
    """`captured_by` is the account; `on_behalf_of` is whose opinion it is. Conflating them is
    the defect: "R. Vega thinks the port closure mattered" and "the operator typed R. Vega"
    are different claims and the library recorded both as one string."""
    said = identity.who_said_it(on_behalf_of="R. Vega")

    assert said["on_behalf_of"]["name"] == "R. Vega"
    assert said["on_behalf_of"]["source"] == "stated"
    assert said["captured_by"]["source"] in ("verified", "stated")


def test_whose_opinion_it_is_is_never_derived():
    """The one fact the server cannot know. Filling `on_behalf_of` from `os_user` would make
    every decision look like the operator's — and the operator is usually the person typing,
    not the person whose judgment is being recorded."""
    with pytest.raises(ValueError, match="on_behalf_of"):
        identity.who_said_it(on_behalf_of="")


def test_the_product_cannot_be_the_person():
    """§9.8's guard, applied to every path rather than one. A model writing "the system" or
    "Claude" into a field that promotes a rule to the checklist is the library confirming its
    own suggestion and recording that a person did."""
    for name in ("Claude", "the system", "automatic", "unknown"):
        with pytest.raises(ValueError, match="must be a PERSON"):
            identity.who_said_it(on_behalf_of=name)


def test_the_author_vocabulary_is_its_own(monkeypatch):
    """D35: `TAG_SOURCE_SYNONYMS` must NOT be reused — "measured" and "from metrics" are
    meaningless as identity claims, and a synonym table accepting them would let
    `source: measured` through on a statement about a person."""
    import enums

    assert "measured" in enums.TAG_SOURCE_SYNONYMS
    assert "measured" not in identity.AUTHOR_SOURCES
    assert set(identity.AUTHOR_SOURCES) == {"verified", "stated"}


def test_the_methods_are_the_three_the_item_names():
    """`method` (stdio_local | sso | import), plus the one the item did not foresee: a caller
    nobody authenticated at all, which must not borrow `stdio_local`'s meaning."""
    assert set(identity.METHODS) == {"stdio_local", "sso", "import", "unattributed"}


def test_an_import_says_so(monkeypatch):
    """A workbook loaded by a script is not a person making a decision at a keyboard, and a
    row that reads the same as one is how a bulk load becomes evidence that somebody
    confirmed something."""
    with identity.importing():
        who = identity.captured_by()

    assert who["method"] == "import"
    assert identity.captured_by()["method"] != "import", "and it does not leak past the load"


# ── wired in: the nine paths that take a person's name ─────────────────────
#
# A module nobody calls is the defect D116 exists to catch, and it would be a particularly
# poor one here — the whole point is that every write recording a decision records who made
# it. These are the paths, and the check is that each one stores BOTH halves.

def _campaign(conn, title="Colombia", **kw):
    kw.setdefault("market", "LATAM")
    kw.setdefault("status", "concluded")
    kw.setdefault("detail", f"A campaign called {title}, which ran in a market.")
    return core.ingest_campaign(conn, title=title, **kw)["campaign_id"]


def test_recording_a_context_event_records_who_captured_it(conn):
    import context

    event = context.record(conn, starts_on="2026-09-05", ends_on="2026-09-14",
                           scope="market", scope_value="Mexico", kind="supply_chain",
                           description="The port was shut for nine days.",
                           recorded_by="R. Vega")

    said = store.authorship_for(conn, "context_event", event["id"])
    assert said["on_behalf_of"]["name"] == "R. Vega"
    assert said["captured_by"]["method"] in identity.METHODS


def test_answering_a_finding_records_who_captured_it(conn):
    import core

    cid = _campaign(conn)
    judged = core.save_evaluation(
        conn, subject_title="Colombia", campaign_id=cid, verdict="revise",
        summary="One difference.", approve_if="Eight weeks, or say why six.",
        findings=[{"severity": "should_fix", "kind": "precedent_departure",
                   "departure": "unexplained", "finding": "Six weeks where Peru ran eight.",
                   "fix": "Extend to eight weeks.",
                   "precedent": {"campaign_id": cid, "quote": "which ran in a market"}}])
    core.answer_finding(conn, evaluation_id=judged["evaluation_id"],
                        finding_id=judged["findings"][0]["id"], answer="deliberate",
                        note="The client moved the date.", said_by="R. Vega")

    entry = core.answers(conn)["answers"][0]
    assert entry["said_by"] == "R. Vega"
    assert entry["captured_by"]["method"] in identity.METHODS


def test_promoting_a_rule_records_who_captured_it(conn):
    """The sharpest case, and D105's own: `confirmed_by` puts a rule on the checklist where it
    is applied to every future brief in its markets. "Nothing distinguishes a real
    confirmation from one the model wrote itself" — now something does: the account is on the
    record beside the name, so a confirmation nobody was present for is visible as one."""
    # The gate needs three campaigns across two markets before anybody can confirm anything.
    noted = None
    for n, market in enumerate(("LATAM", "EMEA", "APAC")):
        cid = _campaign(conn, f"Campaign {n}", market=market)
        noted = corrections.note(conn, text="Never put the logo on a dark background.",
                                 provenance=f"client email, item {n}", campaign_id=cid)

    promoted = corrections.graduate(conn, noted["correction_id"], confirmed_by="R. Vega")

    said = store.authorship_for(conn, "correction", noted["correction_id"])
    assert said["on_behalf_of"]["name"] == "R. Vega"
    assert said["captured_by"]["method"] in identity.METHODS
    assert promoted["confirmed_by"] == "R. Vega"


def test_every_path_that_names_a_person_refuses_the_product(conn):
    """One guard, reached from every one of them. §9.8 had it on two paths and the other seven
    accepted "the system" — so the weakest door decided what the library would believe about
    who made a decision."""
    import context

    cid = _campaign(conn)
    with pytest.raises(ValueError, match="must be a PERSON"):
        context.record(conn, starts_on="2026-09-05", ends_on="2026-09-14", scope="market",
                       scope_value="Mexico", kind="supply_chain",
                       description="The port was shut.", recorded_by="Claude")


def test_the_account_is_recorded_even_when_it_is_the_same_person(conn):
    """Not skipped when `display_name` matches `said_by`. "R. Vega recorded this, and R. Vega
    was at the keyboard" is a stronger claim than "R. Vega recorded this", and dropping the
    account where the two agree throws away exactly the corroboration worth keeping."""
    import context

    event = context.record(conn, starts_on="2026-09-05", ends_on="2026-09-14",
                           scope="market", scope_value="Mexico", kind="supply_chain",
                           description="The port was shut for nine days.",
                           recorded_by="R. Vega")

    assert store.authorship_for(conn, "context_event", event["id"])["captured_by"]


# ── 11.3: captured_by vs on_behalf_of, asked as one numbered question ──────
#
# "The field most products miss." They miss it because it only matters later: on the day the
# note is written everybody knows who was in the room, and two years on the record says a
# name with nothing to say whether that person held the view or merely typed it.
#
# §10.3's rule applies — numbered, never typed — so the menu offers the operator by name from
# `captured_by` and asks for a name only when the answer is somebody else.

def test_the_menu_offers_the_operator_rather_than_asking_them_to_type(monkeypatch, conn):
    """§10.3: "answering should never need typing." The operator is the commonest answer by
    far and the server already knows who they are, so making them spell their own name is the
    one bit of typing this menu had left."""
    import feedback

    monkeypatch.setattr(identity.config, "OPERATOR_NAME", "Nilanjan Bhattacharya")
    cid = _campaign(conn)
    menu = feedback.queue(conn)
    row = next(r for r in menu["rows"] if r.get("campaign_id") == cid)

    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=row["number"])

    whose = next(q for q in asked["questions"] if q["field"] == "said_by")
    assert whose["options"][1] == "Nilanjan Bhattacharya"
    assert "someone else" in whose["options"][2].lower()


def test_the_operator_question_is_absent_when_nobody_configured_a_name(monkeypatch, conn):
    """An `os_user` of `nbhatt` is an account, not a name to file an opinion under. Offering
    "1 = nbhatt" would make the easy answer the wrong one."""
    import feedback

    monkeypatch.setattr(identity.config, "OPERATOR_NAME", "")
    cid = _campaign(conn)
    menu = feedback.queue(conn)
    row = next(r for r in menu["rows"] if r.get("campaign_id") == cid)

    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=row["number"])

    assert not [q for q in asked["questions"] if q["field"] == "said_by"]


def test_recording_says_whether_the_operator_was_speaking_for_themselves(monkeypatch, conn):
    """The distinction, on the record. "R. Vega said this and R. Vega was at the keyboard" and
    "R. Vega said this, relayed by somebody else" are different claims, and the second is the
    one that needs checking two years later."""
    import feedback

    monkeypatch.setattr(identity.config, "OPERATOR_NAME", "Nilanjan Bhattacharya")
    cid = _campaign(conn)
    menu = feedback.queue(conn)
    row = next(r for r in menu["rows"] if r.get("campaign_id") == cid)

    feedback.record(conn, menu_token=menu["menu_token"], choice=row["number"],
                    said_by="R. Vega", reaction=1)

    who = store.authorship_for(conn, "feedback", cid)
    assert who["on_behalf_of"]["name"] == "R. Vega"
    assert who["captured_by"].get("display_name") == "Nilanjan Bhattacharya"
    assert who["speaking_for_themselves"] is False


def test_the_operator_speaking_for_themselves_is_recorded_as_such(monkeypatch, conn):
    import feedback

    monkeypatch.setattr(identity.config, "OPERATOR_NAME", "Nilanjan Bhattacharya")
    cid = _campaign(conn)
    menu = feedback.queue(conn)
    row = next(r for r in menu["rows"] if r.get("campaign_id") == cid)

    feedback.record(conn, menu_token=menu["menu_token"], choice=row["number"],
                    said_by="Nilanjan Bhattacharya", reaction=1)

    assert store.authorship_for(conn, "feedback", cid)["speaking_for_themselves"] is True


# ── 11.4: the context, as stated at the time ───────────────────────────────

def test_every_decision_records_when_and_how_it_arrived(conn):
    """`captured_at`, channel, session id. Not decoration: "these eleven decisions were made
    in one sitting" and "these eleven were made over a month" are different facts about how
    much thought each got, and the library could not tell them apart."""
    import context

    event = context.record(conn, starts_on="2026-09-05", ends_on="2026-09-14",
                           scope="market", scope_value="Mexico", kind="supply_chain",
                           description="The port was shut for nine days.",
                           recorded_by="R. Vega")

    who = store.authorship_for(conn, "context_event", event["id"])
    assert who["captured_at"]
    assert who["channel"] == "stdio"
    assert who["session_id"]


def test_two_decisions_in_one_sitting_share_a_session(conn):
    import context

    first = context.record(conn, starts_on="2026-09-05", ends_on="2026-09-14",
                           scope="market", scope_value="Mexico", kind="supply_chain",
                           description="The port was shut for nine days.",
                           recorded_by="R. Vega")
    second = context.record(conn, starts_on="2026-10-01", ends_on="2026-10-03",
                            scope="market", scope_value="Mexico", kind="competitor_launch",
                            description="A competitor launched the same week.",
                            recorded_by="R. Vega")

    assert (store.authorship_for(conn, "context_event", first["id"])["session_id"]
            == store.authorship_for(conn, "context_event", second["id"])["session_id"])


def test_a_role_is_kept_as_stated_at_the_time(conn):
    """"role **as stated at the time**" — the item's own emphasis. A role looked up later is
    the role somebody holds NOW, so a planner who becomes head of strategy retroactively made
    every past decision as head of strategy, and the record quietly gains authority nobody
    granted it."""
    import context

    event = context.record(conn, starts_on="2026-09-05", ends_on="2026-09-14",
                           scope="market", scope_value="Mexico", kind="supply_chain",
                           description="The port was shut for nine days.",
                           recorded_by="R. Vega", role="Regional planner, LATAM")

    said = store.authorship_for(conn, "context_event", event["id"])["on_behalf_of"]
    assert said["role"] == "Regional planner, LATAM"
    assert said["role_basis"] == "as stated at the time"


def test_a_role_nobody_gave_is_absent_rather_than_looked_up(conn):
    import context

    event = context.record(conn, starts_on="2026-09-05", ends_on="2026-09-14",
                           scope="market", scope_value="Mexico", kind="supply_chain",
                           description="The port was shut for nine days.",
                           recorded_by="R. Vega")

    assert "role" not in store.authorship_for(conn, "context_event", event["id"])["on_behalf_of"]


def test_a_tag_carrying_a_name_survives_the_tool_boundary():
    """The defect the stdio probe found and every unit test missed: `TagObject` had no
    `said_by`, so pydantic stripped it at the MCP boundary. `store.normalize_tags` has carried
    these since §10.3 and the schema did not — so every reaction recorded through
    `update_campaign` over the wire arrived anonymous, and §11.5's whole append-only mechanism
    silently recorded nothing attributable.

    Green everywhere, because the tests call `core.update_campaign` directly and the loss
    happens one layer above it. This is that layer, asserted."""
    import mcp_server

    assert "said_by" in mcp_server.TagObject.__annotations__
    assert "said_at" in mcp_server.TagObject.__annotations__


def test_the_schema_carries_every_field_the_normaliser_reads():
    """The general form, so the next field added to `normalize_tags` cannot go missing the
    same way. A schema and a normaliser are two implementations of one contract, and this
    codebase has been bitten by that shape seven times now."""
    import mcp_server

    # The fields `store.normalize_tags` reads off a tag object, by inspection of its source.
    import inspect
    import store

    source = inspect.getsource(store.normalize_tags)
    read = {m for m in ("value", "source", "said_by", "said_at")
            if f't.get("{m}")' in source or f'"{m}"' in source}
    assert read <= set(mcp_server.TagObject.__annotations__), (
        f"`normalize_tags` reads {sorted(read - set(mcp_server.TagObject.__annotations__))} "
        f"and the MCP schema drops it before the code ever sees it"
    )


def test_an_initial_nobody_can_resolve_is_not_a_name():
    """The rule `feedback` had and the other four paths did not — moved into `identity.person`
    so it applies everywhere. Mutation found that moving it left it untested: the test that
    covered it was about `feedback`'s own message, and the rule now lives somewhere else.

    "R" is not somebody whose view can be weighed against another one later, which is the
    entire reason this field is required."""
    for initial in ("R", "A", " x "):
        with pytest.raises(ValueError, match="not a name"):
            identity.person(initial, field="said_by")


def test_the_length_rule_reaches_the_menu_it_came_from(conn):
    """And it still applies where it started. A rule moved to a shared place has to be checked
    at the surface it was moved AWAY from, or "one guard everywhere" is a claim about the
    refactor rather than about the product."""
    import feedback

    cid = _campaign(conn)
    menu = feedback.queue(conn)
    row = next(r for r in menu["rows"] if r.get("campaign_id") == cid)

    with pytest.raises(ValueError, match="not a name"):
        feedback.record(conn, menu_token=menu["menu_token"], choice=row["number"],
                        said_by="R", reaction=1)


def test_only_one_thing_decides_whether_a_name_is_a_person():
    """The claim §11.2 makes, checked rather than asserted.

    The first sweep gathered four implementations and left `feedback._check_the_name` with its
    own copy; the second gathered that and left `context.attribute` and
    `context.withdraw_attribution` with theirs, plus a `_NOT_A_PERSON` list sitting unused in
    a file that no longer read it — the copy that would have drifted, because somebody adding
    a word to one of them had no way to know the other existed.

    Two sweeps, each believing it had finished. So this is the check that says so."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    offenders = {}
    for path in sorted(root.glob("*.py")):
        if path.name == "identity.py":
            continue
        text = path.read_text(encoding="utf-8")
        # A file may DELEGATE (calling identity) — what it may not do is decide for itself.
        for marker in ('"server", "system"', "is not a PERSON. This records",
                       "must be a PERSON:"):
            if marker in text:
                offenders.setdefault(path.name, []).append(marker)

    assert not offenders, (
        f"these decide for themselves whether a name is a person, instead of asking "
        f"`identity.person`: {offenders}"
    )


def test_promoting_a_rule_refuses_the_product_by_name(conn):
    """D105's own path, and the sharpest: `confirmed_by` puts a rule on the checklist where
    every future brief in its markets is judged against it. `learning.require_a_person` checked
    only for an empty string, so the library could confirm its own suggestion and record that
    a person had — and mutation showed no test would have noticed the guard being removed."""
    import corrections

    noted = None
    for n, market in enumerate(("LATAM", "EMEA", "APAC")):
        cid = _campaign(conn, f"Campaign {n}", market=market)
        noted = corrections.note(conn, text="Never put the logo on a dark background.",
                                 provenance=f"client email, item {n}", campaign_id=cid)

    for name in ("the system", "Claude", "automatic", ""):
        with pytest.raises(ValueError):
            corrections.graduate(conn, noted["correction_id"], confirmed_by=name)

    assert corrections.describe(conn, noted["correction_id"])["status"] != "expected", (
        "and none of those refusals left the rule half-promoted"
    )


def test_speaking_for_themselves_is_unknown_rather_than_no(conn):
    """§11.3. With no configured display name the server has nothing to compare, so "was this
    person at the keyboard?" has no answer — and `False` would be a claim made out of an
    absence, which is exactly the `nothing_to_check`-as-a-pass failure one field over."""
    import context

    monkey = identity.config.OPERATOR_NAME
    identity.config.OPERATOR_NAME = ""
    try:
        event = context.record(conn, starts_on="2026-09-05", ends_on="2026-09-14",
                               scope="market", scope_value="Mexico", kind="supply_chain",
                               description="The port was shut.", recorded_by="R. Vega")
    finally:
        identity.config.OPERATOR_NAME = monkey

    said = store.authorship_for(conn, "context_event", event["id"])
    assert said["speaking_for_themselves"] is None


def test_judging_a_drift_refuses_the_product_and_records_the_account(conn):
    """§9.4's field, which decides whether a departure from the brief was an improvement —
    and therefore what every later citation of this campaign's results carries. It checked
    only for an empty string, so the library could mark its own homework and record that a
    person had."""
    import drift

    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)

    with pytest.raises(ValueError, match="must be a PERSON"):
        drift.classify(conn, campaign_id=cid, subject="a swapped image", about="a swapped image",
                       classification="improvement", why="It read better.",
                       classified_by="the system")

    # The refusal has to arrive BEFORE the write, not from somewhere downstream. Deleting the
    # guard left this test green: `record_authorship` runs `identity.person` too, so the
    # ValueError still came — after `insert_drift_classification` had filed the judgment under
    # "the system". A raise that happens after the row exists is not a refusal, it is a
    # half-written record with a matching exception. Mutation found that.
    assert conn.execute(
        "SELECT COUNT(*) FROM drift_classifications WHERE campaign_id = ?", (cid,)
    ).fetchone()[0] == 0


def test_re_sending_the_same_tags_does_not_invent_a_disagreement(conn):
    """`store.update_campaign` REPLACES tags, so every caller re-sends the ones it is keeping
    — and each re-send appended the same opinion again and marked the copy a revision of
    itself. A phantom disagreement with nobody on the other side of it, on a path the model
    takes whenever it edits anything."""
    import store as store_module

    cid = _campaign(conn)
    tags = [{"value": "liked", "source": "stated", "said_by": "R. Vega",
             "said_at": "2026-09-20T09:00:00+00:00"}]
    core.update_campaign(conn, campaign_id=cid, tags=tags)
    core.update_campaign(conn, campaign_id=cid, tags=tags)
    core.update_campaign(conn, campaign_id=cid, title="Colombia, renamed", tags=tags)

    voices = store_module.reactions_for(conn, cid)
    assert len(voices) == 1, [(v["said_by"], v["value"], v["said_at"]) for v in voices]
    assert voices[0]["supersedes_own_earlier_view"] is False
    assert "disagreement" not in store_module.get_campaign(conn, cid)


def test_an_opinion_through_a_tag_edit_carries_the_account(conn):
    """The `update_campaign` door was the one write path that recorded a view with no
    `captured_*` beside it — so an opinion arriving over the wire had a name and nothing
    saying who was at the keyboard."""
    import store as store_module

    cid = _campaign(conn)
    core.update_campaign(conn, campaign_id=cid, tags=[
        {"value": "liked", "source": "stated", "said_by": "R. Vega"}])

    said = store_module.authorship_for(conn, "reaction", cid)
    assert said["on_behalf_of"]["name"] == "R. Vega"
    assert said["captured_by"]["method"] in identity.METHODS


def test_the_override_log_keeps_the_caveat_that_makes_verified_honest(conn):
    """§11.2, verbatim: reporting `verified` "with nothing beside it would inflate a fact
    about a process into a fact about a person". The sentence lived only on the live dict and
    was never persisted — so `answers()`, the surface built to review human overrides, showed
    `source: verified` by a named person with the caveat stripped off."""
    cid = _campaign(conn)
    judged = core.save_evaluation(
        conn, subject_title="Colombia", campaign_id=cid, verdict="revise",
        summary="One difference.", approve_if="Eight weeks, or say why six.",
        findings=[{"severity": "should_fix", "kind": "precedent_departure",
                   "departure": "unexplained", "finding": "Six weeks where Peru ran eight.",
                   "fix": "Extend to eight weeks.",
                   "precedent": {"campaign_id": cid, "quote": "which ran in a market"}}])
    core.answer_finding(conn, evaluation_id=judged["evaluation_id"],
                        finding_id=judged["findings"][0]["id"], answer="deliberate",
                        note="The client moved the date.", said_by="R. Vega")

    captured = core.answers(conn)["answers"][0]["captured_by"]

    assert captured["source"] == "verified"
    assert "not an authentication" in captured["what_it_means"], (
        "a reader of the log sees a VERIFIED decision by a named person; the sentence saying "
        "anyone with access to this machine writes under this account is the whole caveat"
    )


def test_the_menu_does_not_accept_its_own_option_label_as_a_name(conn):
    """§11.3 offers "2 = Someone else — who?" and `said_by` is a free string on the tool, so
    the model has to translate the number itself. `"Someone else — who?"` was accepted
    VERBATIM as a person, and so were "a colleague", "the client" and "the team" — names that
    `person_on_file` can never find and `pseudonymise_person` can never erase.

    The menu's own label being a valid answer is the sharpest form of it."""
    import feedback

    cid = _campaign(conn)
    menu = feedback.queue(conn)
    row = next(r for r in menu["rows"] if r.get("campaign_id") == cid)

    for placeholder in ("Someone else — who?", "someone else", "a colleague", "the client",
                        "the team", "they"):
        with pytest.raises(ValueError, match="must be a PERSON"):
            feedback.record(conn, menu_token=menu["menu_token"], choice=row["number"],
                            said_by=placeholder, reaction=1)


def test_the_question_says_this_one_is_not_a_number(conn, monkeypatch):
    """Every other question on this menu is answered with a number the SERVER resolves; this
    one is a free string on the tool. The payload's own summary says "offering the numbers",
    so nothing told the model that picking 2 means asking for a name and sending that."""
    import feedback

    monkeypatch.setattr(identity.config, "OPERATOR_NAME", "N. Bhattacharya")
    cid = _campaign(conn)
    menu = feedback.queue(conn)
    row = next(r for r in menu["rows"] if r.get("campaign_id") == cid)

    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=row["number"])

    whose = next(q for q in asked["questions"] if q["field"] == "said_by")
    assert whose["answer_with"] == "a name"
    assert "not the number" in whose["why"]


def test_the_people_already_on_file_are_offered_too(conn, monkeypatch):
    """§11.3's premise is that the person typing usually is NOT the person whose view it is —
    and the menu made the typist the one-keystroke answer and everybody else a typing task,
    optimising the ergonomics of the failure mode.

    The people already in the library are known, so they can be numbered too: less typing than
    before for the colleague case, and the operator is no longer the cheap default."""
    import feedback

    monkeypatch.setattr(identity.config, "OPERATOR_NAME", "N. Bhattacharya")
    first = _campaign(conn, "Peru")
    core.record_reaction(conn, campaign_id=first, value="liked", said_by="R. Vega")
    core.record_reaction(conn, campaign_id=first, value="liked", said_by="A. Duarte")
    cid = _campaign(conn, "Bogota")
    menu = feedback.queue(conn)
    row = next(r for r in menu["rows"] if r.get("campaign_id") == cid)

    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=row["number"])

    offered = list(next(q for q in asked["questions"]
                        if q["field"] == "said_by")["options"].values())
    assert "R. Vega" in offered and "A. Duarte" in offered
    assert "N. Bhattacharya" in offered


def test_a_tag_cannot_attribute_an_opinion_to_something_that_is_not_a_person(conn):
    """§11.2 at the door tags actually come through.

    `_keep_the_view` refused a non-person and SKIPPED the tag — but `normalize_tags`, which is
    what decides what gets stored, had no such check. So `said_by="the team"` was refused from
    the append-only record and written onto the campaign anyway, where `get_campaign` hands it
    to every reader. Two implementations of one rule, disagreeing: the surface people read
    said an opinion was held by the team, and the record built to be the authority on who
    holds which opinion had never heard of it.

    Refused rather than quietly dropped, because the caller believes they attributed it, and a
    silently unattributed tag is the state §11.5 exists to make impossible."""
    cid = _campaign(conn)

    with pytest.raises(ValueError, match="must be a PERSON"):
        core.update_campaign(conn, campaign_id=cid,
                             tags=[{"value": "liked", "source": "stated",
                                    "said_by": "the team"}])

    assert store.get_campaign(conn, cid).get("tags") in (None, [], ())
    assert store.reactions_for(conn, cid) == []


def test_the_refusal_names_the_tag_so_the_rest_of_the_write_can_be_resent(conn):
    """A whole-write refusal is only usable if it says which tag. `normalize_tags` already
    refuses a bad `value`, a bad `source` and a `verified` with no metrics the same way, so
    this is that function's existing contract rather than a new failure mode for callers."""
    cid = _campaign(conn)

    with pytest.raises(ValueError) as raised:
        core.update_campaign(conn, campaign_id=cid,
                             tags=[{"value": "liked", "source": "stated", "said_by": "R. Vega"},
                                   {"value": "not_liked", "source": "stated",
                                    "said_by": "somebody else"}])

    assert "not_liked" in str(raised.value), "which tag, or the caller has to guess"


def test_a_tag_with_a_real_name_still_reaches_the_append_only_record(conn):
    """The other half: the guard must not cost the thing it is guarding."""
    cid = _campaign(conn)

    core.update_campaign(conn, campaign_id=cid,
                         tags=[{"value": "liked", "source": "stated", "said_by": "R. Vega"}])

    assert [v["said_by"] for v in store.reactions_for(conn, cid)] == ["R. Vega"]


# The product-name check was a substring scan, so every name CONTAINING one of its words was
# refused — "Themba Nkosi" because of "them", "Automne" because of "auto", "Claudette" because
# of "claude", "Sautoy" and "Lautoka" and "Matthey" and "Serverin" and "Anthem Lee". Refusing a
# real person's name is not a safe failure: it is a hard block on recording their view, it
# falls hardest on names that are not Anglo, and the remedy the message offers — "give another
# spelling" — does not exist for somebody's own name.
_REAL_PEOPLE = ("Themba Nkosi", "Thembi Dlamini", "Thembekile Mandela", "Claude Monet",
                "Claudette Colbert", "Automne Girard", "Anthem Lee", "Marcus Autolycus",
                "Serverin Kowalski", "Matthey Laurent", "Sautoy Bernard", "Lautoka Vakatawa")

# And the other half: these still have to be refused, or the check has been deleted rather
# than fixed. Both lists are asserted, because a guard loosened until it passes everything is
# the commonest way this kind of fix goes wrong.
_NOT_PEOPLE = ("Claude", "claude", "the system", "the model", "campaign-poc", "server",
               "Campaign Intelligence", "automatic", "n/a", "unknown", "anonymous", "nobody",
               "someone else", "a colleague", "the client", "the team", "they", "tbd",
               "the  System  ")


@pytest.mark.parametrize("name", _REAL_PEOPLE)
def test_a_real_name_containing_a_disqualifying_word_is_a_person(name):
    import identity

    assert identity.person(name, field="said_by") == name.strip()


@pytest.mark.parametrize("name", _NOT_PEOPLE)
def test_a_name_that_is_only_disqualifying_words_is_refused(name):
    import identity

    with pytest.raises(ValueError, match="must be a PERSON"):
        identity.person(name, field="said_by")


def test_a_tag_whose_value_needed_normalising_still_reaches_the_record(conn):
    """`_keep_the_view` was handed the caller's RAW tag list while the campaign stored the
    NORMALISED one, so the two disagreed on exactly the fields the walk gates on. A value of
    `" liked "` is stored as `liked` and was skipped here, because the raw string is not in
    `REACTION_AXES` — the campaign saying R. Vega liked it and the append-only record never
    having heard of it, which is the §11.5 split this walk exists to prevent.

    The guard against re-validating (which once re-ran the `verified` gate blind, without the
    record's `has_actual_metrics`) is kept: this reads what was STORED rather than
    normalising a second time, so there is still one implementation of the rule."""
    cid = _campaign(conn)

    core.update_campaign(conn, campaign_id=cid,
                         tags=[{"value": " liked ", "source": "client stated",
                                "said_by": "R. Vega"}])

    voices = store.reactions_for(conn, cid)
    assert [(v["said_by"], v["value"]) for v in voices] == [("R. Vega", "liked")]
    assert voices[0]["source"] == "stated", (
        "`client stated` normalises to `stated` on the campaign; storing the raw spelling "
        "here makes two records of one opinion disagree about its provenance"
    )


def test_a_tag_sent_at_upload_reaches_the_record_too(conn):
    """`upload_campaign` accepts `said_by` on a tag and never called this walk, so an opinion
    recorded at upload was on the campaign and absent from the table that is supposed to be
    the authority on opinions — the same split, through the door most people arrive by."""
    cid = core.ingest_campaign(
        conn, title="Chile launch", market="LATAM", status="concluded",
        detail="A campaign called Chile launch, which ran in a market.",
        tags=[{"value": "liked", "source": "stated", "said_by": "A. Duarte"}],
        confirm=True)["campaign_id"]

    assert [v["said_by"] for v in store.reactions_for(conn, cid)] == ["A. Duarte"]


def _legacy_tag(conn, cid, said_by="the client"):
    """A tag as v0.2.0 stored it: `said_by` written server-side by `feedback.record`, past a
    name list that did not yet contain "the client" or "the team". Written straight into the
    blob, because the only way to produce one now is to have produced it then."""
    import json

    conn.execute("UPDATE campaigns SET tags = ? WHERE id = ?",
                 (json.dumps([{"value": "liked", "source": "stated", "said_by": said_by,
                               "said_at": "2026-03-01T09:00:00+00:00"}]), cid))
    conn.commit()


def test_a_name_already_on_file_does_not_lock_the_record(conn):
    """The refusal must apply to the claim being MADE, not to one already stored.

    Tags REPLACE, so every caller re-sends the ones it is keeping — `feedback.record` re-sends
    the whole stored list on every menu answer. With the new rule applied blind, one legacy
    `said_by: "the client"` made the campaign permanently unwritable: every answer refused,
    naming a tag the caller never sent, with no remedy in the message. A guard that bricks a
    record to enforce a naming rule has done more damage than the record it objected to."""
    cid = _campaign(conn)
    _legacy_tag(conn, cid)

    stored = store.get_campaign(conn, cid)["tags"]
    core.update_campaign(conn, campaign_id=cid, tags=stored + [
        {"value": "performed_well", "source": "stated", "said_by": "R. Vega"}])

    values = {t["value"] for t in store.get_campaign(conn, cid)["tags"]}
    assert values == {"liked", "performed_well"}


def test_the_legacy_attribution_is_not_kept_as_if_it_named_somebody(conn):
    """Grandfathered is not endorsed. The opinion stays — losing it is the §11.5 loss — but
    it is no longer recorded as being HELD by "the client", because nothing can find that
    person, erase them at their request, or weigh their view against another. It reads as
    what it is: an opinion with nobody's name against it, with the original text kept where
    somebody can still act on it."""
    cid = _campaign(conn)
    _legacy_tag(conn, cid)

    core.update_campaign(conn, campaign_id=cid, tags=store.get_campaign(conn, cid)["tags"])

    tag = next(t for t in store.get_campaign(conn, cid)["tags"] if t["value"] == "liked")
    assert "said_by" not in tag
    assert tag["said_by_unresolved"] == "the client"
    assert not [v for v in store.reactions_for(conn, cid) if v["said_by"] == "the client"], (
        "and it must not be copied into the append-only record on the way past"
    )


def test_a_new_tag_naming_the_same_non_person_is_still_refused(conn):
    """The grandfathering is per stored value, not an amnesty. Somebody typing "the client"
    today is making the claim the rule is about."""
    cid = _campaign(conn)
    _legacy_tag(conn, cid)

    with pytest.raises(ValueError, match="must be a PERSON"):
        core.update_campaign(conn, campaign_id=cid, tags=[
            {"value": "performed_well", "source": "stated", "said_by": "the client"}])


def test_the_refusal_says_the_right_thing_for_a_placeholder():
    """One sentence covered two different mistakes and only fitted one. A model writing
    "Claude" is the library confirming its own suggestion; a model writing "the team" is
    relaying a real human answer it could not turn into a name. Telling the second one it is
    "naming itself" describes nothing that happened and suggests no fix."""
    import identity

    with pytest.raises(ValueError) as product:
        identity.person("Claude", field="said_by")
    assert "naming itself" in str(product.value)

    with pytest.raises(ValueError) as placeholder:
        identity.person("the team", field="said_by")
    said = str(placeholder.value)
    assert "naming itself" not in said
    assert "must be a PERSON" in said
    assert "ask" in said.lower() or "who" in said.lower(), (
        "the fix is to ask whose view it is — the message has to say so"
    )


def test_a_refusal_does_not_echo_an_unbounded_tag_value(conn):
    """The tag value is caller-supplied and unbounded, and it is interpolated into the
    refusal: a 100,000-character tag produced a 100,198-character exception. An error nobody
    can read is a different way of not saying what went wrong."""
    cid = _campaign(conn)

    with pytest.raises(ValueError) as raised:
        core.update_campaign(conn, campaign_id=cid,
                             tags=[{"value": "x" * 100_000, "source": "stated",
                                    "said_by": "the team"}])

    assert len(str(raised.value)) < 500
    assert "xxxx" in str(raised.value), "it still has to say which tag"


def test_a_capitalised_reaction_value_still_reaches_the_record(conn):
    """`REACTION_AXES` is lowercase and tag values were never folded, so `"Liked"` — the
    likelier spelling for a model writing prose — was stored on the campaign and dropped
    here. C99's sentence again, one capital letter later.

    The consequence is not cosmetic: no reaction row means `disagreement_on` sees one voice,
    which means `_contested_silence` finds nothing, which means §11.5's `contested_precedent`
    never fires — the finding C93 put in `_COMPUTED_FINDINGS` so that an `approve` with no
    findings could not drop it. A capital letter dropped it."""
    cid = _campaign(conn)

    core.update_campaign(conn, campaign_id=cid, tags=[
        {"value": "Liked", "source": "stated", "said_by": "R. Vega"}])

    assert [(v["said_by"], v["value"]) for v in store.reactions_for(conn, cid)] == [
        ("R. Vega", "liked")]
    assert [t["value"] for t in store.get_campaign(conn, cid)["tags"]] == ["liked"], (
        "and the two surfaces say the same word, not two spellings of it"
    )


def test_a_tag_that_is_not_a_reaction_keeps_the_spelling_it_was_given(conn):
    """Only the product's OWN vocabulary is canonicalised. A marketer's "Back-to-School" is
    their words, and lowercasing every tag to fix seven of them would rewrite the library."""
    cid = _campaign(conn)

    core.update_campaign(conn, campaign_id=cid, tags=["Back-to-School", "Gen-Z"])

    assert {t["value"] for t in store.get_campaign(conn, cid)["tags"]} == {
        "Back-to-School", "Gen-Z"}


def test_two_people_holding_the_same_view_are_both_kept(conn):
    """The dedupe key was the tag VALUE alone, so a second person recording `liked` replaced
    the first — §11.5's loss, arriving through the de-duplication rather than the replace.

    It reached the append-only record before only because `_keep_the_view` read the caller's
    raw list; making the two surfaces agree would have made them agree on having lost a
    voice. Agreement is corroboration: two people liking something is the evidence, not a
    duplicate of it."""
    cid = _campaign(conn)

    core.update_campaign(conn, campaign_id=cid, tags=[
        {"value": "liked", "source": "stated", "said_by": "R. Vega"},
        {"value": "liked", "source": "stated", "said_by": "A. Duarte"}])

    assert {t.get("said_by") for t in store.get_campaign(conn, cid)["tags"]} == {
        "R. Vega", "A. Duarte"}
    assert {v["said_by"] for v in store.reactions_for(conn, cid)} == {"R. Vega", "A. Duarte"}


def test_re_sending_a_tag_a_second_later_still_invents_no_disagreement(conn):
    """C95 closed the phantom-revision bug and closed it for calls landing in the same
    SECOND. The dedupe compared `said_at` for exact equality, and `normalize_tags` only stores
    `said_at` when the caller supplies one — so a re-send got a fresh stamp and the pair read
    as R. Vega revising their own view.

    Tags REPLACE, so any second edit that touches them re-sends them, and two edits a second
    apart are ordinary. §11.5 calls a revision pair the most informative row this table holds,
    which is exactly why manufacturing a false one costs more than a duplicate would."""
    tags = [{"value": "liked", "source": "stated", "said_by": "R. Vega"}]
    cid = _campaign(conn)

    # The clock MOVES between the calls. The earlier version of this test made all three in
    # the same second, so it passed against the bug — which is why it is written this way:
    # the defect is entirely about what happens when the wall clock ticks.
    ticks = iter(["2026-09-20T09:00:01+00:00", "2026-09-20T09:00:02+00:00",
                  "2026-09-20T09:00:03+00:00"])
    real = store._now_iso
    store._now_iso = lambda: next(ticks, "2026-09-20T09:00:09+00:00")
    try:
        core.update_campaign(conn, campaign_id=cid, tags=tags)
        core.update_campaign(conn, campaign_id=cid,
                             tags=store.get_campaign(conn, cid)["tags"])
        core.update_campaign(conn, campaign_id=cid, title="Colombia, renamed",
                             tags=store.get_campaign(conn, cid)["tags"])
    finally:
        store._now_iso = real

    voices = store.reactions_for(conn, cid)
    assert len(voices) == 1, [v["said_at"] for v in voices]
    assert store.disagreement_on(conn, cid) in (None, {}, [])


def test_the_same_person_saying_it_again_at_a_stated_later_time_is_a_revision(conn):
    """And the other half, or the fix has deleted the mechanism rather than the bug: a
    caller who SAYS when is recording a second occasion, and that pair is the row §11.5 is
    built around."""
    cid = _campaign(conn)

    core.update_campaign(conn, campaign_id=cid, tags=[
        {"value": "liked", "source": "stated", "said_by": "R. Vega",
         "said_at": "2026-03-01T09:00:00+00:00"}])
    core.update_campaign(conn, campaign_id=cid, tags=[
        {"value": "not_liked", "source": "stated", "said_by": "R. Vega",
         "said_at": "2026-06-01T09:00:00+00:00"}])

    assert len(store.reactions_for(conn, cid)) == 2


# Every parameter name this product uses for "whose judgment this is". A write taking one of
# these is making an attribution, and §11.2 says exactly one thing decides whether the value
# names a person.
_NAMES_A_PERSON = ("said_by", "recorded_by", "classified_by", "confirmed_by", "withdrawn_by",
                   "linked_by", "on_behalf_of", "answered_by", "attributed_by")

# Layers that may take one of these and not check it, with the reason. `store` is persistence:
# it writes what the domain layer decided, and putting the guard there as well as above would
# be the second copy the test above forbids. `mcp_server` and `http_app` are transport, and
# hand straight through to a `core`/`people`/`drift` function that does check.
#
# `store.normalize_tags` is the exception, and it is here as a NAMED one rather than by being
# in a skipped file: tags are the one attribution the domain layer never sees as a parameter,
# because they arrive inside a list of dicts. That is how the sixth door got in.
# `store*` rather than `store.py`: the split into `store_campaigns.py` and friends moved these
# writes without changing what they are. The entry is about a ROLE — this layer hands the name
# on to whoever records it, rather than recording it itself — and the role travelled with the
# code. A new module outside the family still has to be added here deliberately, which is the
# review this table exists to force.
_MAY_DELEGATE = {"mcp_server.py", "http_app.py", "stdio_server.py", "replay.py"}
_MAY_DELEGATE_PREFIXES = ("store",)

# Functions that take one of these names and do not RECORD it, with the reason written beside
# them — the §10.6 pattern, where an exemption has to be stated rather than arrived at. A new
# door cannot get in by resembling one of these; it has to be added here, which is the review.
# Keyed by the file the function LIVES in, so moving code moves the entry — which is a small
# visible cost and the thing that keeps this table reviewable: a reader can open the file named
# beside a name and check the reason still holds. `answers` moved from core.py to missing.py
# when the gap surface was split out, and this was the only place that had to follow it.
_DOES_NOT_RECORD_IT = {
    # A read. `answers(said_by=...)` filters the override log by whose answers to show, and
    # filtering on a name asserts nothing about who decided anything.
    "missing.py": {"answers"},
    # Decides ELIGIBILITY and echoes the name back in its explanation; the write that follows
    # goes through `require_a_person` in this same module, which is the guard. Checking here
    # as well would be the second copy the test above forbids.
    "learning.py": {"gate"},
}


def test_every_write_that_takes_a_name_reaches_the_one_guard():
    """The half the marker sweep above cannot see.

    That test catches a DUPLICATE implementation — a file deciding for itself. It cannot
    catch a door with NO check, which is exactly how `store.normalize_tags` accepted
    `said_by: "the team"` for a whole release while §11.2 presented the marker sweep as the
    thing that proved the rule had one implementation. A missing check leaves no marker to
    find."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    trees = {path.name: ast.parse(path.read_text(encoding="utf-8"))
             for path in sorted(root.glob("*.py"))}

    # Every function ANYWHERE in the product that itself reaches `identity.person`. Across
    # modules rather than within one: `corrections.graduate` and `metrics.graduate` both
    # guard by calling `learning.require_a_person`, and requiring the call to be literal in
    # every function would be asking for the duplicate the sweep above forbids.
    guards = set()
    for tree in trees.values():
        for top in ast.walk(tree):
            # `record_authorship` is deliberately NOT a guard, though it does reach
            # `identity.person` through `who_said_it`. It runs after the row is written, so a
            # caller relying on it alone raises the right error having already stored the
            # thing — which is exactly what `drift.classify` did: the refusal arrived, and the
            # judgment was on file under "the system". A raise after the write is not a
            # refusal, it is a half-written record with a matching exception.
            if top.__class__.__name__ not in ("FunctionDef", "AsyncFunctionDef"):
                continue
            if top.name == "record_authorship":
                continue
            if any((isinstance(c.func, ast.Attribute)
                    and c.func.attr in ("person", "who_said_it"))
                   or (isinstance(c.func, ast.Name) and c.func.id in ("person",
                                                                      "who_said_it"))
                   for c in ast.walk(top) if isinstance(c, ast.Call)):
                guards.add(top.name)

    unguarded = {}
    for name, tree in trees.items():
        if (name in _MAY_DELEGATE or name == "identity.py"
                or name.removesuffix(".py").startswith(_MAY_DELEGATE_PREFIXES)):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            takes = ({a.arg for a in node.args.args + node.args.kwonlyargs}
                     & set(_NAMES_A_PERSON))
            if not takes or node.name in guards:
                continue
            calls = {c.func.attr if isinstance(c.func, ast.Attribute) else c.func.id
                     for c in ast.walk(node) if isinstance(c, ast.Call)
                     and isinstance(c.func, (ast.Attribute, ast.Name))}
            if calls & guards:
                continue
            if node.name in _DOES_NOT_RECORD_IT.get(name, set()):
                continue
            unguarded.setdefault(name, []).append(
                f"{node.name}({', '.join(sorted(takes))})")

    assert not unguarded, (
        f"these accept a name and never reach `identity.person`, so whichever of them a "
        f"model finds decides what this library believes about who decided things: "
        f"{unguarded}"
    )
