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
    marketer whose first judgment comes back confident and evidence-free will believe it."""
    report = core.readiness(conn)

    assert report["stage"] == "empty"
    assert report["can"] == []
    assert report["cannot"], "an empty library cannot do anything, and should say so"


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
    _campaign(conn, "Bogota", outcomes=True, tags=[{"value": "liked"}])
    _campaign(conn, "Lima", outcomes=True, tags=[{"value": "not_liked"}])

    report = core.readiness(conn)

    assert report["stage"] == "working"
    assert "say_what_worked" in {c["code"] for c in report["can"]}


def test_the_rulebook_is_what_makes_a_guardrail_checkable(conn):
    """Without it the product can compare a proposal against precedent and cannot check it
    against a rule, which is a different kind of finding entirely."""
    _campaign(conn, "Bogota", outcomes=True)

    before = core.readiness(conn)
    core.ingest_campaign(conn, title="Brand guidelines", detail="never use AI imagery",
                         record_type="reference", confirm=True)
    after = core.readiness(conn)

    assert "check_against_rules" in {c["code"] for c in before["cannot"]}
    assert "check_against_rules" in {c["code"] for c in after["can"]}


# ── it has to reach somebody ────────────────────────────────────────────────

def test_listing_a_thin_library_carries_the_guidance(conn):
    """Tracker D67. A tool nobody calls is still waiting to be asked, and `list_campaigns`
    is the surface the review named — eight rows with no sense of whether eight is enough."""
    _campaign(conn, "Bogota")

    listed = core.list_campaigns_with_readiness(conn)

    assert listed["readiness"]["stage"] in ("first_records", "thin")
    assert listed["readiness"]["shortest_path"]


def test_listing_a_working_library_does_not_lecture(conn):
    """Guidance that never stops appearing is the thing nobody reads."""
    _campaign(conn, "Bogota", outcomes=True, tags=[{"value": "liked"}])
    _campaign(conn, "Lima", outcomes=True, tags=[{"value": "not_liked"}])
    core.ingest_campaign(conn, title="Brand guidelines", detail="the rules",
                         record_type="reference", confirm=True)

    listed = core.list_campaigns_with_readiness(conn)

    assert "readiness" not in listed or listed["readiness"] is None


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
