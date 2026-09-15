"""
D116: every model-facing tool must be named in a `next_actions` from the write that makes its
output non-empty.

Not a defect in any one item. The same omission was found by review in §8.3's gate, §8.6's
`note_correction`, §8.7's `replay_rules`, §9.2's comparison, §9.4's classification, §9.6's two
context tools and §9.9's reconciliation — seven times, each caught after the fact by somebody
reading the code rather than by anything in it. A process gap that recurs seven times is not
seven mistakes; it is a missing check.

**Why it matters more than it sounds.** A tool the model would have to know exists and
spontaneously decide to call is, in practice, a tool nobody calls. §8.3 hit the sharpest
version: `note_correction` was referenced nowhere in the product but its own definition, so
nothing was ever RECORDED for the gate to act on — the feature could not fail, because it
could not start.

This is the checklist, executed. A tool that is genuinely never offered has to say so here,
in words, with the reason — which is a cheaper conversation than the eighth review finding.
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _registered_tools() -> set:
    """Every `@mcp.tool()` in the server, by name."""
    tree = ast.parse((ROOT / "mcp_server.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            call = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(call, ast.Attribute) and call.attr == "tool":
                names.add(node.name)
    return names


def _offered_tools() -> set:
    """Every tool named as the target of an offer, anywhere in the product."""
    sources = "".join(path.read_text(encoding="utf-8")
                      for path in sorted(ROOT.glob("*.py")))
    # `actions.action(label, "tool_name", ...)` is the only way an offer is built.
    return set(re.findall(r'action\(\s*(?:f?"(?:[^"\\]|\\.)*"|f?\'[^\']*\'|[^,]+),\s*'
                          r'"([a-z_]+)"', sources, re.S))


# Tools that are genuinely never offered, each with the reason. A short list on purpose: this
# is the escape hatch, and an escape hatch nobody has to justify is not one.
NEVER_OFFERED = {
    # READS. The model reaches for these because the USER asked a question, not because a
    # write made them answerable. Offering one is offering to answer a question nobody asked,
    # and three of those in a `next_actions` is how the list stops being read.
    "list_campaigns", "get_campaign", "find_similar_campaigns", "diff_campaigns",
    "get_evaluation", "list_evaluations", "find_similar_images", "check_image_provenance",
    "drift_readings", "measure_status", "list_corrections", "calibration",
    "get_reconciliation",
    # DIAGNOSTICS AND LIFECYCLE, which a person invokes deliberately. `delete_campaign` in
    # particular: a product that suggests deleting your records is not one to trust with them.
    "health_check", "getting_started", "delete_campaign",
    # §11.7. `person_on_file` and `personal_data_position` answer a question somebody asked —
    # nothing this library WRITES makes them newly answerable, because the personal data is
    # there from the first upload. `pseudonymise_person` is the sharpest never-offer in the product:
    # it is irreversible, it rewrites records that saved judgments rest on, and a product that
    # SUGGESTS erasing a person is one nobody should trust with their colleagues' names.
    "person_on_file", "personal_data_position", "pseudonymise_person",
    "correct_person_name",
    # THE NUMBERED MENU IS THE OFFER. §10.2's whole complaint is that a floating set of
    # affordances breaks the habit the menu exists to create — so `feedback_queue` returns
    # numbered rows and a `next_actions` beside them would be a second, competing way to
    # answer the same question.
    "feedback_choose", "feedback_record",
    # THE PLURAL OF AN OFFERED TOOL. `upload_image_asset` is offered from the
    # `execution_never_checked` gap; offering both is two labels for one act, and the plural's
    # `asset_refs` is the one argument that cannot be prefilled from anything on file.
    "upload_image_assets",
    # UNDO. Offering somebody the way to take back what they have just deliberately done
    # reads as the product doubting them. `reopen_correction` is the deliberate exception and
    # earns it: setting a rule aside is the one act here that silently changes every future
    # judgment in a market, so the way back is named once, at the moment it is made.
    "withdraw_context_event", "withdraw_attribution",
}


def test_every_write_making_a_tool_useful_offers_it():
    """The checklist D116 exists to replace. A tool that is neither offered nor listed above
    is one somebody has to already know about — which is the state §8.6 was in when nothing
    it could act on was ever recorded."""
    registered = _registered_tools()
    assert len(registered) > 30, f"only found {len(registered)} tools — has the shape changed?"

    unoffered = sorted(registered - _offered_tools() - NEVER_OFFERED)

    assert not unoffered, (
        f"these tools are never offered anywhere and are not on the never-offered list: "
        f"{unoffered}. Either offer each from the write that makes its output non-empty, or "
        f"add it to NEVER_OFFERED with the reason."
    )


def test_the_never_offered_list_has_no_ghosts():
    """A name on the escape-hatch list that is not a tool any more is an exemption nobody can
    see the shape of — the same copy-drift the deferral tracker and the README list are each
    guarded against."""
    registered = _registered_tools()

    ghosts = sorted(NEVER_OFFERED - registered)

    assert not ghosts, f"these are on the never-offered list and are not tools: {ghosts}"


def test_the_never_offered_list_stays_short():
    """It is the escape hatch. If most of the surface is on it, the check is decoration."""
    registered = _registered_tools()

    assert len(NEVER_OFFERED) < len(registered) * 0.75, (
        f"{len(NEVER_OFFERED)} of {len(registered)} tools are exempt — that is not a "
        f"checklist, it is a list of things nobody offers."
    )


# ── what a user actually receives, not what the source mentions ────────────
#
# Review's objection to everything above, and it is right: the checks are regexes over source
# text. They prove a tool's name appears inside an `actions.action(...)` call — not that any
# user ever receives the offer. A tool offered from an unreachable branch passes. A tool
# offered from a branch whose offer is ALWAYS trimmed away passes.
#
# That is not hypothetical: D54's ranking offer was "offered" from `after_upload` and, on the
# uploads that mattered, emitted zero times, because it was appended last and `trim` keeps
# three. The source-text guard was green throughout.
#
# So these run the real surfaces against a real library and read what comes back. Narrow on
# purpose — a fixture cannot cover every branch, and pretending otherwise would be the same
# false comfort one level up. What it does cover is that the commonest path through the
# product emits the offers the phase was built to add.

def test_the_offers_a_new_library_actually_receives(conn):
    """One upload into an empty library: the surface every new customer meets first."""
    import core

    landed = core.ingest_campaign(conn, title="Peru launch", market="Peru",
                                  status="concluded",
                                  detail="A retail activation in Lima.")

    assert [a["tool"] for a in landed["next_actions"]] == [
        "add_metrics", "update_campaign", "gaps"], (
        "the first upload's three: record the results, say when it ran, and read the "
        "ranking that just started meaning something"
    )


def test_the_offers_a_working_library_actually_receives(conn):
    """The judging flow, end to end, reading only what each call returns."""
    import core

    peru = core.ingest_campaign(conn, title="Peru launch", market="Peru", status="concluded",
                                starts_on="2026-01-01", ends_on="2026-02-01",
                                detail="A retail activation in Lima.")["campaign_id"]
    core.add_metrics(conn, campaign_id=peru, structured={"roas": 3.4}, confirm=True)

    prepared = core.prepare_evaluation(conn, subject_title="Bogota brief",
                                       proposal_text="A store launch on Calle 82.")
    assert "save_evaluation" in [a["tool"] for a in prepared["next_actions"]], (
        "the call the whole second half of the product starts at"
    )

    judged = core.save_evaluation(
        conn, subject_title="Bogota brief", campaign_id=None, verdict="revise",
        summary="Close to Peru, one difference unexplained.",
        approve_if="Eight weeks, or say why six.",
        findings=[{"severity": "should_fix", "kind": "precedent_departure",
                   "departure": "unexplained",
                   "finding": "Six weeks where Peru ran eight.",
                   "fix": "Extend to eight weeks, or say why six is right.",
                   "precedent": {"campaign_id": peru,
                                 "quote": "A retail activation in Lima."}}])
    assert "answer_finding" in [a["tool"] for a in judged["next_actions"]], (
        "this judgment asks a question; the place its answer goes has to be named"
    )


def test_the_offers_a_confounded_outcome_actually_receives(conn):
    """§9.8's human half, which nothing pointed at until D116."""
    import context
    import core

    mex = core.ingest_campaign(conn, title="Mexico launch", market="Mexico",
                               status="concluded", starts_on="2026-09-01",
                               ends_on="2026-09-30",
                               detail="A launch in Mexico City.")["campaign_id"]
    context.record(conn, starts_on="2026-09-05", ends_on="2026-09-14", scope="market",
                   scope_value="Mexico", kind="supply_chain",
                   description="The port was shut for nine days.", recorded_by="R. Vega")

    filed = core.add_metrics(conn, campaign_id=mex, structured={"roas": 1.1}, confirm=True)

    assert "attribute_outcome" in [a["tool"] for a in filed["next_actions"]]


def test_every_offer_a_real_library_emits_can_actually_be_carried_out(conn):
    """"An offer has to work" — checked against offers that were EMITTED rather than against
    offers the source mentions. An argument name that has drifted is an error the user did
    not cause and cannot fix, and it reaches them only through an offer they actually saw."""
    import inspect

    import core
    import mcp_server

    peru = core.ingest_campaign(conn, title="Peru launch", market="Peru", status="concluded",
                                detail="A retail activation in Lima.")["campaign_id"]
    core.add_metrics(conn, campaign_id=peru, structured={"roas": 3.4}, confirm=True)
    emitted = (
        (core.ingest_campaign(conn, title="Bogota", market="Colombia", status="concluded",
                              detail="A store launch.").get("next_actions") or [])
        + (core.readiness(conn).get("next_actions") or [])
        + [a for g in core.gaps(conn)["gaps"] for a in g.get("next_actions") or []]
        + core.coverage_offer(conn)
        + (core.prepare_evaluation(conn, subject_title="X",
                                   proposal_text="A brief.").get("next_actions") or []))

    assert len(emitted) > 6, f"the fixture has to emit a real spread of offers: {len(emitted)}"
    published = {t.name for t in mcp_server.mcp._tool_manager.list_tools()}
    for offer in emitted:
        assert offer["tool"] in published, f"{offer['tool']} is not a tool"
        takes = set(inspect.signature(getattr(mcp_server, offer["tool"])).parameters)
        extra = set(offer.get("prefilled_args") or {}) - takes
        assert not extra, f"{offer['tool']} does not take {sorted(extra)}"
