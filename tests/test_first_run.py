"""
Idea F: "Make the empty state a guided first run."

    "A fresh install has no campaigns and therefore no opinions, and nothing tells a new user
    how many records it takes before judgments become useful, or which ones to add first.

    Fix: on an empty or thin library, lead with the shortest path to value — one brief you
    liked, one you did not, and the rulebook — and say plainly what the system can and
    cannot do at that size."

Two halves. The **shortest path** is the review's own prescription and is ordered, not a
menu: one brief you liked, one you did not, the rulebook. The **capability statement** is the
harder half, because "how many records before judgments become useful" has no honest numeric
answer — usefulness depends on what is in the library, not how much. Two contrasting briefs
make the liked/not-liked axis work at two records; a hundred concluded campaigns with no
measured outcomes still cannot say whether anything worked.

So readiness is expressed as what the product **can** and **cannot** do given what is
actually present, and each `cannot` names the thing that would change it. A capability
statement nobody can act on is a disclaimer.

Collects three tracker rows: D43 (an offer whose arguments the user must supply — this item
is made almost entirely of them), D66 (`coverage`'s `thin` list collapsing when everything is
thin) and D68 (one campaign in three thin cells offered once, not three times).
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


# ── the shortest path, in order ─────────────────────────────────────────────

def test_an_empty_library_is_given_the_reviews_own_three_steps(conn):
    """"one brief you liked, one you did not, and the rulebook" — in that order, because it
    is a path rather than a menu. The contrast is what makes two records useful; two briefs
    somebody liked teach the library nothing about the axis it is being asked to judge on."""
    report = core.readiness(conn)

    # Asserted on what each step would WRITE, not on its wording — the label is for the user
    # and will be rewritten, which is the coupling §5.1 was about.
    steps = report["shortest_path"]
    assert len(steps) == 3
    assert [s["prefilled_args"].get("tags", [{}])[0].get("value") for s in steps[:2]] == \
        ["liked", "not_liked"], "the contrast, and in that order"
    assert steps[2]["prefilled_args"]["record_type"] == "reference", "the rulebook, last"


def test_each_step_says_what_the_user_has_to_supply(conn):
    """Tracker D43. Every step here is an offer whose arguments only the user has — there is
    nothing in an empty library to prefill from — so "accepting is one step" is false and
    `needs` is what makes that honest."""
    for step in core.readiness(conn)["shortest_path"]:
        assert step["needs"], step["label"]
        assert step["consent"] == "ask"


def test_a_step_already_taken_drops_off_the_path(conn):
    """A path that still lists what you have done is a checklist nobody believes."""
    _campaign(conn, "Bogota", tags=[{"value": "liked"}])

    steps = core.readiness(conn)["shortest_path"]

    written = [s["prefilled_args"].get("tags", [{}])[0].get("value") for s in steps]
    assert "liked" not in written
    assert any(s["prefilled_args"].get("record_type") == "reference" for s in steps)


def test_the_path_is_empty_once_all_three_are_done(conn):
    _campaign(conn, "Bogota", tags=[{"value": "liked"}])
    _campaign(conn, "Lima", tags=[{"value": "not_liked"}])
    core.ingest_campaign(conn, title="Brand guidelines", detail="the rulebook",
                         record_type="reference", confirm=True)

    assert core.readiness(conn)["shortest_path"] == []


# ── what it can and cannot do at this size ──────────────────────────────────

def test_an_empty_library_says_it_has_no_opinions(conn):
    """"A fresh install has no campaigns and therefore no opinions." Said plainly, because a
    marketer whose first judgment comes back confident and evidence-free will believe it.

    §12.1 put ONE row on `can` for an empty library, and it is worth being careful about why
    that is not a softening. Every other `can` row is a claim about what the library holds,
    and an empty library holds nothing. The rulebook is not in the library: it ships with the
    product, so its rules apply to the very first brief somebody sends — which is exactly the
    brief they are most likely to be judging alone. Saying "I can check this against your
    written rules" on day one is true, and withholding it would be the same product
    understating itself with the same authority it used to overstate."""
    report = core.readiness(conn)

    assert report["stage"] == "empty"
    assert report["can"] == [], (
        "the product ships no rules, so even the rulebook row is a `cannot` here — the "
        "mechanism shipping is not the capability existing"
    )
    assert report["cannot"], "an empty library cannot do anything, and should say so"
    # And the instruction the test is named for is still the one a reader gets.
    assert "cannot do" in report["note"] and "evidence-free" in report["note"]


def test_every_limit_names_the_thing_that_would_lift_it(conn):
    """A capability statement nobody can act on is a disclaimer. Each `cannot` carries the
    record that would change it."""
    _campaign(conn, "Bogota")

    for limit in core.readiness(conn)["cannot"]:
        assert limit["code"], limit
        assert limit["what"], limit
        assert limit["needs"], f"{limit['code']} names nothing that would change it"


def test_two_contrasting_briefs_unlock_the_axis_they_contrast_on(conn):
    """Usefulness is about what is in the library, not how much. Two records with opposite
    reactions make the liked/not-liked comparison work; two the marketer liked do not."""
    _campaign(conn, "Bogota", tags=[{"value": "liked"}])
    _campaign(conn, "Lima", tags=[{"value": "not_liked"}])

    report = core.readiness(conn)

    assert "weigh_reactions" in {c["code"] for c in report["can"]}


def test_a_library_with_no_measured_outcome_says_it_cannot_tell_you_what_worked(conn):
    """The limit that survives any amount of uploading. A hundred concluded campaigns with
    nothing measured still cannot say whether any of it worked."""
    for i in range(6):
        _campaign(conn, f"Campaign {i}", tags=[{"value": "liked"}])

    report = core.readiness(conn)

    assert "say_what_worked" in {c["code"] for c in report["cannot"]}
    assert report["stage"] != "working"


def test_a_measured_library_is_reported_as_working(conn):
    """`working` needs the path walked as well as something measured — the rulebook is the
    third step, and without it the guidance still has something to say."""
    _campaign(conn, "Bogota", outcomes=True, tags=[{"value": "liked"}])
    _campaign(conn, "Lima", outcomes=True, tags=[{"value": "not_liked"}])
    core.ingest_campaign(conn, title="Brand guidelines", detail="the rules",
                         record_type="reference", confirm=True)

    report = core.readiness(conn)

    assert report["stage"] == "working"
    assert "say_what_worked" in {c["code"] for c in report["can"]}


def test_uploading_guidelines_is_still_not_the_same_as_a_rule(conn):
    """The distinction this test was written for, which SURVIVES §12.1 and is now the only
    thing separating two rows.

    The first version asserted `check_against_rules` moved into `can` once a reference record
    existed — the false claim review caught. §12.1 lifted the limit, but only for rules in the
    rulebook FILE: a guidelines PDF uploaded into the library is still an ordinary record,
    still retrieved by similarity, and still absent from the briefs least like it. So
    `rulebook_on_file` turns on when you upload one and says what it is worth, and
    `check_against_rules` is about the file, is true of an empty library, and does not move
    when a document is uploaded."""
    _campaign(conn, "Bogota", outcomes=True)

    before = core.readiness(conn)
    core.ingest_campaign(conn, title="Brand guidelines", detail="never use AI imagery",
                         record_type="reference", confirm=True)
    after = core.readiness(conn)

    assert "rulebook_on_file" not in {c["code"] for c in before["can"]}
    assert "rulebook_on_file" in {c["code"] for c in after["can"]}
    # Unmoved by the upload, in both directions: uploading a document neither grants the
    # rule-checking capability nor takes it away. (It is a `cannot` here because no rules are
    # written in the rulebook — which is the point: a guidelines PDF is not a rule.)
    assert "check_against_rules" in {c["code"] for c in before["cannot"]}
    assert "check_against_rules" in {c["code"] for c in after["cannot"]}
    cited = next(c for c in after["can"] if c["code"] == "rulebook_on_file")
    assert "similar" in cited["what"], (
        "and it still has to say that an uploaded document arrives only if it ranks"
    )


# ── it has to reach somebody ────────────────────────────────────────────────

def test_the_guidance_is_attached_while_the_library_is_not_working(conn):
    """Tracker D67, at the core level; the tool-level version is below and is the one that
    matters, since the first attempt at this was a function no tool called."""
    _campaign(conn, "Bogota")

    assert core.readiness_for_listing(conn)["stage"] in ("first_records", "thin")


def test_the_guidance_stops_once_the_library_works(conn):
    """Guidance that never stops appearing is the thing nobody reads."""
    _campaign(conn, "Bogota", outcomes=True, tags=[{"value": "liked"}])
    _campaign(conn, "Lima", outcomes=True, tags=[{"value": "not_liked"}])
    core.ingest_campaign(conn, title="Brand guidelines", detail="the rules",
                         record_type="reference", confirm=True)

    assert core.readiness_for_listing(conn) is None


# ── what the tracker was carrying ───────────────────────────────────────────

def test_a_library_where_every_cell_is_thin_gets_one_line_not_a_list(conn):
    """Tracker D66. Listing 25 weak cells in a library where all of them are weak is the
    matrix again, and the answer at that size is not "fix LATAM" — it is the readiness
    guidance."""
    for i in range(8):
        _campaign(conn, f"Campaign {i}", market=f"Market {i}")

    report = core.coverage(conn)

    assert report["thin"] == []
    assert report["thin_summary"]
    assert report["next_actions"], "the guidance takes over from the list"


def test_a_library_with_some_strength_still_lists_its_weak_cells(conn):
    _campaign(conn, "Bogota", market="LATAM", outcomes=True)
    _campaign(conn, "Lima", market="LATAM", outcomes=True)
    _campaign(conn, "Cairo", market="MENA")

    report = core.coverage(conn)

    assert [t["market"] for t in report["thin"]] == ["MENA"]


def test_one_campaign_in_several_thin_cells_is_offered_once(conn):
    """Tracker D68. A campaign that ran in three markets sits in three thin cells, and the
    same fix was offered three times — the grid is right for "what do I have in Colombia",
    but the action is per campaign."""
    _campaign(conn, "SEA rollout", markets=["Malaysia", "Singapore", "Thailand"])
    _campaign(conn, "Bogota", market="LATAM", outcomes=True)
    _campaign(conn, "Lima", market="LATAM", outcomes=True)

    report = core.coverage(conn)

    unmeasured = report["unmeasured_campaigns"]
    assert len(unmeasured) == 1
    assert sorted(unmeasured[0]["markets"]) == ["Malaysia", "Singapore", "Thailand"]


# ══ review of 5.6 ════════════════════════════════════════════════════════════

def test_the_list_campaigns_TOOL_carries_the_guidance(conn, monkeypatch):
    """The claim was false. `core.list_campaigns_with_readiness` existed, was tested, and no
    tool ever called it — so the guidance reached nobody, which is exactly the failure D67
    was written about. The commit message, the plan and the docstring all said otherwise."""
    import mcp_server

    _campaign(conn, "Bogota")
    monkeypatch.setattr(mcp_server.store, "connect", lambda: _Borrowed(conn))

    listed = mcp_server.list_campaigns()

    assert listed["readiness"]["stage"] in ("first_records", "thin")
    assert listed["campaigns"][0]["campaign_id"], "the tool's own field projection survives"


def test_the_tool_stops_carrying_it_once_the_library_works(conn, monkeypatch):
    import mcp_server

    _campaign(conn, "Bogota", outcomes=True, tags=[{"value": "liked"}])
    _campaign(conn, "Lima", outcomes=True, tags=[{"value": "not_liked"}])
    core.ingest_campaign(conn, title="Brand guidelines", detail="the rules",
                         record_type="reference", confirm=True)
    monkeypatch.setattr(mcp_server.store, "connect", lambda: _Borrowed(conn))

    assert "readiness" not in mcp_server.list_campaigns()


def test_a_tag_typed_the_way_a_person_types_it_still_counts(conn):
    """Tags are freeform and stored as typed, and the store folds case when filtering — this
    did not. So a marketer who did exactly what the path asked was told forever to add a
    campaign they liked: the permanent complaint this codebase names three files over."""
    _campaign(conn, "Bogota", tags=[{"value": "Liked"}])
    _campaign(conn, "Lima", tags=[{"value": "Not_Liked"}])

    report = core.readiness(conn)

    assert "weigh_reactions" in {c["code"] for c in report["can"]}
    written = [st["prefilled_args"].get("tags", [{}])[0].get("value")
               for st in report["shortest_path"]]
    assert written == [None], "both tag steps are done; only the rulebook remains"


def test_the_new_claim_is_bounded_by_what_is_written_down(conn, tmp_path, monkeypatch):
    """§12.1 makes "this breaks your own rule" sayable, and the danger of shipping it is
    replacing one overclaim with another in the tool whose entire job is saying what this
    product cannot do.

    So the row has to carry its boundary: only the rules actually written in the rulebook. A
    guideline nobody has written down is not checkable by anything, and an uploaded guidelines
    document is retrieved by similarity like any other record."""
    _campaign(conn, "Bogota", outcomes=True)
    core.ingest_campaign(conn, title="Brand guidelines", detail="never use AI imagery",
                         record_type="reference", confirm=True)

    import rulebook

    monkeypatch_target = tmp_path / "rulebook.yaml"
    monkeypatch_target.write_text(
        "version: 'acme-2'\nrules:\n  - id: no-ai\n    rule: No AI imagery.\n"
        "    severity: blocking\n    why: Two clients asked in writing.\n")
    monkeypatch.setattr(rulebook, "_bundled", lambda: monkeypatch_target)
    rulebook.load.cache_clear()
    try:
        report = core.readiness(conn)
        can = {c["code"]: c for c in report["can"]}

        assert "check_against_rules" in can
        bounded = can["check_against_rules"]["bounded_by"]
        assert "written down" in bounded
        assert "similarity" in bounded, (
            "the uploaded-document case is the one somebody will assume is covered"
        )
    finally:
        rulebook.load.cache_clear()


def test_a_wholly_unmeasured_library_is_offered_the_thing_that_would_fix_it(conn):
    """The collapse handed over to `shortest_path`, which never mentions measurement — so
    once liked/not_liked/rulebook existed, the summary said the problem was measurement and
    offered nothing at all, while `gaps()` on the same library offered `add_metrics`."""
    _campaign(conn, "Bogota", tags=[{"value": "liked"}])
    _campaign(conn, "Lima", tags=[{"value": "not_liked"}])
    core.ingest_campaign(conn, title="Brand guidelines", detail="the rules",
                         record_type="reference", confirm=True)

    report = core.coverage(conn)

    assert report["thin_summary"]
    assert report["next_actions"], "the summary named a problem and offered nothing"
    assert report["next_actions"][0]["tool"] == "add_metrics"


def test_the_collapsed_list_does_not_report_a_total_it_is_hiding(conn):
    """`thin: []` beside `thin_total: 3` is an inconsistent pair for a reader."""
    _campaign(conn, "Bogota", market="LATAM")
    _campaign(conn, "Lima", market="SEA")

    report = core.coverage(conn)

    assert report["thin"] == []
    assert report["thin_total"] == 0 or report["thin_summary"]


def test_a_library_of_proposals_is_not_described_as_things_you_have_run(conn):
    """`gaps()` draws this line explicitly — a campaign that has not concluded cannot have
    results — and `readiness`, written beside it, did not. It asserted "compare against what
    you have run before" for two things that never ran, and asked for the results of a
    concluded campaign when none had concluded."""
    _campaign(conn, "Next quarter", status="proposed", tags=[{"value": "liked"}])
    _campaign(conn, "Also next", status="proposed", tags=[{"value": "not_liked"}])

    report = core.readiness(conn)

    compare = next(c for c in report["can"] if c["code"] == "compare_to_precedent")
    assert "run" not in compare["what"].lower(), compare["what"]

    worked = next(c for c in report["cannot"] if c["code"] == "say_what_worked")
    assert "conclude" in worked["needs"].lower(), (
        f"nothing has concluded, so asking for a concluded campaign's results is a request "
        f"nobody can satisfy: {worked['needs']}"
    )


def test_a_capability_is_never_on_both_lists_at_once(conn):
    """`say_what_worked` appeared in `can` and `cannot` simultaneously for any partly measured
    library, so a reader keying on the code could not tell which side won."""
    _campaign(conn, "Bogota", outcomes=True)
    _campaign(conn, "Lima")

    report = core.readiness(conn)

    assert not ({c["code"] for c in report["can"]}
                & {c["code"] for c in report["cannot"]})


def test_a_mixed_reaction_is_not_a_dislike(conn):
    """Mutation-proof: dropping `mixed_reaction` from the dislike set left all 103 tests
    green, so nothing pinned the choice. And the item's own rationale asks for "one you did
    NOT like" — a mixed reaction is not that contrast, and counting it would tell the user
    the axis works when it does not."""
    _campaign(conn, "Bogota", tags=[{"value": "liked"}])
    _campaign(conn, "Lima", tags=[{"value": "mixed_reaction"}])

    report = core.readiness(conn)

    assert "weigh_reactions" in {c["code"] for c in report["cannot"]}


class _Borrowed:
    """Lends the test's connection to a tool without letting the tool close it.

    `sqlite3.Connection.close` is read-only, so it cannot be patched out — and every tool
    closes the connection it was handed in a `finally`.
    """

    def __init__(self, inner):
        self._inner = inner

    def close(self):
        pass

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_the_guidance_does_not_vanish_before_the_path_is_walked(conn):
    """Two campaigns, one measured, no tags, no rulebook reported `working` — so
    `list_campaigns` stopped attaching the guidance while all three steps were still
    outstanding and the library could neither weigh reactions nor cite a rule. "Working" has
    to mean the path is done AND something is measured, or the guidance disappears exactly
    when it is still needed."""
    measured = _campaign(conn, "A", outcomes=True)
    _campaign(conn, "B")

    report = core.readiness(conn)

    assert report["shortest_path"], "precondition: the path is unwalked"
    assert report["stage"] != "working"
    assert core.readiness_for_listing(conn) is not None


def test_one_record_cannot_contrast_with_itself(conn):
    """A single campaign tagged both `liked` and `not_liked` satisfied the axis on its own.
    "The library holds both" was technically true and substantively false — the contrast the
    product reasons from is between records, and one record cannot be the counter-example to
    itself."""
    _campaign(conn, "A", tags=[{"value": "liked"}, {"value": "not_liked"}])

    report = core.readiness(conn)

    assert "weigh_reactions" in {c["code"] for c in report["cannot"]}
