"""
§13.1 — one definition of "measured", and the two facts it was standing for.

D75 and D88. `readiness`, `gaps`, `coverage` and `disconfirming` each decided separately what
a measured library is, and by the fourth implementation two of them disagreed OUT LOUD: on an
ordinary library the product says *"all 3 campaigns here have measured results"* and, in the
very next response, *"no campaign in the library is tagged 'underperformed' with measured
results behind it"*. Reproduced before anything was changed; both sentences are below.

**The fix is not one predicate.** Reading those two sentences carefully, they are not the same
claim made twice — they are two different claims wearing one word:

  HAS RESULTS   numbers are on file for it. `add_metrics` put them there.
  HAS A VERDICT somebody has said whether those numbers were good, and that judgment is
                backed by them — a `performed_well` / `underperformed` tag with
                `source: verified`.

A campaign can have every number anybody asked for and no verdict at all; that is the ordinary
state of a library nobody has been back to. Collapsing the two would make the product either
claim a verdict it does not have or deny results it does. So what §13.1 removes is the four
private implementations and the shared WORD — one helper answers both questions, every surface
asks it, and each says which of the two it means.

What this file pins is therefore: the two facts are computed in ONE place, they are computed
the same way for every caller (status, record type and supersession included), and no surface
uses the bare word "measured" for a fact it has not established.
"""
import pytest

import core
import store

# `conn` is `tests/conftest.py`'s: it patches `config`'s paths, which are read at import
# and cached, so a fixture that only set the environment variables handed every test in
# this file the SAME database. The records leaked forwards and the first version of
# `test_the_two_facts_are_computed_in_one_place` failed on the previous test's campaigns.


def _a_png(conn, tmp_path):
    from PIL import Image

    path = tmp_path / "shot.png"
    Image.new("RGB", (32, 32), "navy").save(path)
    return path


def _measured(conn, title, *, market="Peru", status="concluded", value=3.4, **kwargs):
    """A campaign with real numbers on file and nobody's verdict about them."""
    cid = core.ingest_campaign(conn, title=title, market=market, status=status,
                               detail="A store opening.", confirm=True, **kwargs)["campaign_id"]
    core.add_metrics(conn, campaign_id=cid, detail=f"ROAS came in at {value}.",
                     structured={"roas": value}, metric_type="actual", confirm=True)
    return cid


def _with_a_verdict(conn, title, *, tag="performed_well", **kwargs):
    """The same, plus somebody's verified judgment that the numbers were good or bad."""
    cid = _measured(conn, title, **kwargs)
    core.update_campaign(conn, campaign_id=cid,
                         tags=[{"value": tag, "source": "verified"}])
    return cid


# ---------------------------------------------------------------------------------------
# The disagreement, stated as the product states it


def test_the_product_does_not_say_a_library_has_and_has_not_got_measured_results(conn):
    """D88's own sentence: "a library can be told all its campaigns have measured results and,
    in the next response, that nothing in it has a measured verdict."

    Both halves are true of this library. What was wrong is that both were said in the same
    words, so a reader has no way to tell a second FACT from a contradiction — and a product
    that appears to contradict itself about its own evidence is one nobody checks twice."""
    for n in range(3):
        _measured(conn, f"Peru launch {n}", value=3.0 + n)

    said_it_can = next(c for c in core.readiness(conn)["can"]
                       if c["code"] == "say_what_worked")
    judgment = core.save_evaluation(conn, subject_title="Lima launch", verdict="approve",
                                    approve_if="", summary="Looks strong.",
                                    cited_ids=[], findings=[])
    said_it_cannot = judgment["disconfirming"]["what_it_means"]

    assert "measured results" in said_it_can["what"]
    assert "no campaign in the library" not in said_it_cannot.lower(), said_it_cannot
    # It has to name the fact it is actually missing, which is the VERDICT and not the
    # results — the results are on file and the sentence above just said so.
    assert "verdict" in said_it_cannot.lower()


def test_the_surface_that_needs_a_verdict_says_what_would_supply_one(conn):
    """§5.6's rule, which this surface was the last to break: a capability statement nobody
    can act on is a disclaimer. "Nothing here has a measured verdict" over a library of three
    campaigns with real numbers is one `update_campaign` away from being false."""
    for n in range(3):
        _measured(conn, f"Peru launch {n}", value=3.0 + n)

    judgment = core.save_evaluation(conn, subject_title="Lima launch", verdict="approve",
                                    approve_if="", summary="Looks strong.",
                                    cited_ids=[], findings=[])

    waiting = judgment["disconfirming"]["could_be_checked_if"]
    assert len(waiting) == 3, waiting
    assert all(w["campaign_id"] and w["title"] for w in waiting)


def test_a_verdict_is_a_performance_tag_and_not_any_tag(conn):
    """"Has a verdict" means somebody said whether it WORKED. A reaction tag says whether
    somebody liked it, which is the other axis entirely — §2.4 keeps them apart everywhere
    else, and a `liked` tag counted as a measured verdict would let an opinion stand in for a
    result on the one surface built to argue with a verdict using results."""
    cid = _measured(conn, "Peru launch")
    core.update_campaign(conn, campaign_id=cid,
                         tags=[{"value": "liked", "source": "verified"}])

    state = core.library_state(conn)

    assert cid in state["with_results"]
    assert cid not in state["with_verdicts"], "a reaction is not a performance verdict"


def test_both_directions_of_a_verdict_count(conn):
    """A campaign that UNDERPERFORMED is the more valuable of the two here — the disconfirming
    search exists to find precedent that argues against a verdict, and a library where only
    the successes count as verdicts can only ever agree with an approval."""
    good = _with_a_verdict(conn, "Peru launch", tag="performed_well")
    bad = _with_a_verdict(conn, "Lima launch", tag="underperformed")

    assert core.library_state(conn)["with_verdicts"] == {good, bad}


def test_a_stated_performance_claim_is_not_a_verdict(conn):
    """`source: verified` is the whole of it: an impression somebody typed is not a measured
    verdict, and the tag provenance §2.4 built exists to keep the two apart."""
    cid = _measured(conn, "Peru launch")
    core.update_campaign(conn, campaign_id=cid,
                         tags=[{"value": "performed_well", "source": "stated"}])

    assert cid not in core.library_state(conn)["with_verdicts"]


def test_the_remedy_names_only_the_records_that_would_change_the_answer(conn):
    """`could_be_checked_if` is §5.6's "name the record that would lift this". A list that
    includes records already carrying a verdict is naming work that changes nothing — and the
    first version iterated every measured campaign, so on a library where the check was
    already possible it offered the tagged ones back."""
    plain = _measured(conn, "Peru launch")
    _with_a_verdict(conn, "Lima launch", tag="performed_well")

    waiting, total = core._campaigns_that_could_be_tagged(conn)

    assert [w["campaign_id"] for w in waiting] == [plain]
    assert total == 1


def test_the_count_beside_the_remedy_is_not_the_truncated_one(conn):
    """`coverage` states the rule in writing — "the list can lose detail; the shape of the
    library must not depend on where the cut fell" — and this broke it in the other
    direction: the sentence read its count off the CAPPED list, so a library with seven
    campaigns waiting was told three."""
    for n in range(7):
        _measured(conn, f"Peru launch {n}")

    judgment = core.save_evaluation(conn, subject_title="Lima launch", verdict="approve",
                                    approve_if="", summary="Looks strong.",
                                    cited_ids=[], findings=[])
    check = judgment["disconfirming"]

    assert check["waiting_total"] == 7
    assert len(check["could_be_checked_if"]) == core._MAX_DISCONFIRMING
    assert "7 campaign(s)" in check["what_it_means"], check["what_it_means"]


def test_a_library_of_verdicts_of_the_other_kind_is_not_a_library_with_none(conn):
    """The state the first fix got backwards. Every campaign tagged `performed_well`, and the
    search is for `underperformed`: the product said "nothing in the library carries a
    measured verdict" while two campaigns carried one — and the string it replaced was TRUE
    here. A fix that makes the sentence false in a state where it used to be true is worse
    than the defect."""
    _with_a_verdict(conn, "Peru launch", tag="performed_well")
    _with_a_verdict(conn, "Lima launch", tag="performed_well")

    judgment = core.save_evaluation(conn, subject_title="Bogota launch", verdict="approve",
                                    approve_if="", summary="Looks strong.",
                                    cited_ids=[], findings=[])
    check = judgment["disconfirming"]

    assert check["why_not"] == "none_of_this_kind"
    said = check["what_it_means"]
    assert "2 campaign(s) here carry a measured verdict" in said, said
    assert "nothing in the library carries a measured verdict" not in said.lower()
    # And the sentence the MODEL says, which is the one the user hears and the one the first
    # round left on the old wording.
    spoken = core._say_the_disconfirming_check(check)
    assert "none of them is the kind" in spoken, spoken


def test_an_empty_library_says_nothing_has_been_measured(conn):
    """The third state, and the only one where the original sentence was right."""
    core.ingest_campaign(conn, title="Nobody went back to it", market="Peru",
                         status="concluded", detail="A store opening.", confirm=True)

    check = core.save_evaluation(conn, subject_title="X", verdict="approve", approve_if="",
                                 summary="s", cited_ids=[], findings=[])["disconfirming"]

    assert check["why_not"] == "nothing_measured_at_all"
    assert "nothing in this library has been measured" in check["what_it_means"].lower()
    assert check["could_be_checked_if"] == []


def test_a_cited_superseded_version_still_counts_as_evidence_with_numbers(conn):
    """The evidence ladder weighs the records a judgment CITED, and the library-wide sets drop
    superseded records by design. Asking them here answered a question about the library when
    the question is about somebody's citations: a cited v1 with real numbers on it read as
    unmeasured, and the strength ladder is built on that count."""
    first = _measured(conn, "Peru launch v1")
    core.ingest_campaign(conn, title="Peru launch v2", market="Peru", status="concluded",
                         detail="Revised.", supersedes=first, confirm=True)

    assert first not in core.library_state(conn)["with_results"], "not library evidence"

    strength = core._evidence_strength(conn, cited_ids=[first], text="A store opening.")

    assert strength["with_results"] == 1, (
        "a citation of a superseded version is still a citation of something measured"
    )


def test_the_verdict_words_are_a_named_subset_of_the_performance_words():
    """`feedback.PERFORMANCE` has four values and `core._A_PERFORMANCE_VERDICT` has two, and
    that difference is deliberate: `feedback` asks "has anybody answered the performance
    question", which `performed_as_expected` and `no_data_yet` both do, and this asks "is
    there a verdict that could CONTRADICT a judgment", which only a directional one can.

    Two questions, not two vocabularies. What would be a defect is the two drifting without
    anybody deciding — a fifth performance word added to the queue and silently absent from
    the verdict set, which is §13.1's own failure shape one list along."""
    import feedback

    assert set(core._A_PERFORMANCE_VERDICT) < set(feedback.PERFORMANCE.values())
    assert set(feedback.PERFORMANCE.values()) - set(core._A_PERFORMANCE_VERDICT) == {
        "performed_as_expected", "no_data_yet"}, (
        "a performance word appeared that is neither directional nor an explicit absence — "
        "decide which kind it is before the two lists disagree in silence"
    )


def test_a_neutral_performance_tag_is_not_a_verdict_that_can_contradict(conn):
    """`performed_as_expected` answers the performance question and points nowhere, so it is
    not precedent against anything. Counted as a verdict it would make the disconfirming
    search report a check it could not have run."""
    cid = _measured(conn, "Peru launch")
    core.update_campaign(conn, campaign_id=cid,
                         tags=[{"value": "performed_as_expected", "source": "verified"}])

    assert cid not in core.library_state(conn)["with_verdicts"]


def test_the_remedy_does_not_offer_a_campaign_that_has_not_run(conn):
    """`could_be_checked_if` names records that would make the disconfirming check possible.
    A campaign still in flight cannot supply a verdict — `store.normalize_tags` would take
    the tag, but "it performed well" about something still running is a claim nobody can
    stand behind, and the whole point of a verified tag is that somebody does."""
    running = core.ingest_campaign(conn, title="Still going", market="Peru",
                                   status="in_flight", detail="A store opening.",
                                   confirm=True)["campaign_id"]
    core.add_metrics(conn, campaign_id=running, detail="ROAS 3.4 so far.",
                     structured={"roas": 3.4}, metric_type="actual", confirm=True)
    done = _measured(conn, "Finished")

    waiting, total = core._campaigns_that_could_be_tagged(conn)

    assert [w["campaign_id"] for w in waiting] == [done]
    assert total == 1


@pytest.fixture()
def declaring(request):
    """A customer rulebook in force for one test, and cleared afterwards."""
    import rulebook

    def write(yaml):
        path = rulebook.overlay_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml, encoding="utf-8")
        rulebook.load.cache_clear()

    yield write
    rulebook.load.cache_clear()


def test_a_verdict_written_in_the_customers_own_word_is_a_verdict(conn, declaring):
    """§12.2/D73: a vocabulary is a synonym resolved WHEN TWO VALUES ARE COMPARED, never a
    rewrite of what was stored. `store.filter_campaign_ids` folds tags at read time and
    `has_a_verdict` matched the exact string, so the store and core disagreed about the same
    record — and the cost was not cosmetic. The disconfirming search HID a contradicting
    precedent that the code before §13.1 found, and said "no verdict recorded against them"
    about a record whose verdict was written in the agency's own word."""
    declaring("version: 'acme-3'\nrules: []\nvocabulary:\n  tags:\n"
              "    underperformed: ['flopped']\n")
    cid = _measured(conn, "Peru launch flop", value=0.9)
    core.update_campaign(conn, campaign_id=cid,
                         tags=[{"value": "flopped", "source": "verified"}])

    # The store already counted it. That is the disagreement.
    assert cid in set(store.filter_campaign_ids(
        conn, tags=[{"value": "underperformed", "source": "verified"}]))
    assert cid in core.library_state(conn)["with_verdicts"]
    assert core.carries_verdict(store.get_campaign(conn, cid), "underperformed")

    check = core.save_evaluation(conn, subject_title="Lima launch", verdict="approve",
                                 approve_if="", summary="A store opening in Peru.",
                                 cited_ids=[], findings=[])["disconfirming"]

    assert check["code"] != "nothing_to_check_against", check["what_it_means"]


def test_a_neutral_verdict_is_not_reported_as_no_verdict_recorded(conn):
    """`performed_as_expected` is an ANSWER — the feedback queue writes it as `verified` and
    stops asking — so it is rightly not precedent against anything. What is not right is
    calling that record one with "no verdict recorded against it" and offering it as the
    remedy: tagging it `underperformed` on the strength of that offer would contradict what
    somebody already recorded."""
    neutral = _measured(conn, "Peru launch")
    core.update_campaign(conn, campaign_id=neutral,
                         tags=[{"value": "performed_as_expected", "source": "verified"}])
    unanswered = _measured(conn, "Lima launch")

    state = core.library_state(conn)

    assert neutral not in state["with_verdicts"], "it points nowhere, so it is not precedent"
    assert neutral in state["with_neutral_verdicts"]
    assert state["without_verdicts"] == {unanswered}, (
        "a campaign somebody HAS judged was reported as one nobody has"
    )

    waiting, total = core._campaigns_that_could_be_tagged(conn)
    assert [w["campaign_id"] for w in waiting] == [unanswered] and total == 1


def test_a_verdict_on_something_that_never_ran_is_not_precedent(conn):
    """The helper said in one field that a cancelled campaign never ran and in the next that
    its verdict could contradict a judgment. One status rule, applied to both."""
    cancelled = core.ingest_campaign(conn, title="Cusco launch", market="Peru",
                                     status="cancelled", detail="The site fell through.",
                                     confirm=True)["campaign_id"]
    core.add_metrics(conn, campaign_id=cancelled, detail="Spend to date 400.",
                     structured={"roas": 0.4}, metric_type="actual", confirm=True)
    core.update_campaign(conn, campaign_id=cancelled,
                         tags=[{"value": "underperformed", "source": "verified"}])

    state = core.library_state(conn)

    assert cancelled not in state["ran"]
    assert cancelled not in state["with_verdicts"], (
        "its verdict is offered as precedent about a campaign the helper says never ran"
    )


def test_the_remedy_is_ordered_by_when_the_record_arrived(conn):
    """The list is capped at `_MAX_DISCONFIRMING`, so WHICH records it names matters. Sorted
    by id it named three uuids at random — and a different three if the same records were
    re-imported. Oldest first: those are the results somebody is least likely to remember."""
    ids = [_measured(conn, f"Peru launch {n}") for n in range(5)]

    waiting, total = core._campaigns_that_could_be_tagged(conn)

    assert total == 5
    assert [w["campaign_id"] for w in waiting] == ids[:len(waiting)]


def test_the_pole_search_and_the_helper_cannot_disagree(conn):
    """Two routes to one answer, proved rather than assumed.

    `_pole_search` reaches the verdict question through `find_similar`'s tag filter and
    `library_state` reaches it through `has_a_verdict`. They agree on every input the schema
    can currently produce — a first draft of this test asserted that the intersection
    excluded a superseded record, and it passed because `find_similar` already excludes those,
    so it could not have failed. What is worth pinning is the EQUIVALENCE: every record the
    tag-filtered search can return is one the helper calls verdict-holding. If either record
    line moves, this is what says so."""
    _with_a_verdict(conn, "Peru launch", tag="underperformed")
    _with_a_verdict(conn, "Lima launch", tag="performed_well")
    _measured(conn, "Cusco launch")
    superseded = _with_a_verdict(conn, "Bogota v1", tag="underperformed")
    core.ingest_campaign(conn, title="Bogota v2", market="Peru", status="concluded",
                         detail="Revised.", supersedes=superseded, confirm=True)
    reference = core.ingest_campaign(conn, title="Guidelines", record_type="reference",
                                     detail="How we talk.", confirm=True)["campaign_id"]

    holds = core.library_state(conn)["with_verdicts"]
    assert superseded not in holds and reference not in holds

    for tag in ("underperformed", "performed_well"):
        ranked = core.find_similar(conn, text="A store opening.", top_k=20,
                                   tags=[{"value": tag, "source": "verified"}],
                                   full_detail=False)
        returned = {m["campaign_id"] for m in ranked}
        assert returned <= holds, (
            f"the search returns {returned - holds} as carrying a {tag!r} verdict and the "
            f"helper does not — two record lines for one question"
        )


def test_a_library_with_no_numbers_at_all_says_so_plainly(conn):
    """The other branch of coverage's summary, and the one the original sentence was written
    for. "None with results on file" is TRUE here and must still be said — the second branch
    exists for the library that has results and has not finished, not instead of this one."""
    for n in range(2):
        core.ingest_campaign(conn, title=f"Peru launch {n}", market="Peru",
                             status="concluded", detail="Nobody went back to it.",
                             confirm=True)

    report = core.coverage(conn)

    assert "none with results on file" in report["thin_summary"]
    assert "have not concluded" not in report["thin_summary"]
    assert {a["tool"] for a in report["next_actions"]} == {"add_metrics"}


def test_drift_reads_the_shared_predicate_for_a_known_outcome(conn, tmp_path):
    """`drift.py` and `context.py` each held a byte-identical copy of
    `bool(record.get("has_actual_metrics"))`, which is `core.has_results` a third time. Three
    expressions of one rule is how they drift — the whole of D75 — and the cost here is a
    forecast being read as a known outcome, which decides whether a departure from the brief
    can be classified at all."""
    import drift

    cid = core.ingest_campaign(conn, title="Next quarter", market="Peru", status="concluded",
                               detail="A plan.", confirm=True)["campaign_id"]
    core.add_metrics(conn, campaign_id=cid, detail="We expect ROAS of 4.",
                     structured={"roas": 4.0}, metric_type="predicted", confirm=True)
    record = store.get_campaign(conn, cid)
    assert record["has_metrics"] is True and core.has_results(record) is False

    asset = core.ingest_image_asset(
        conn, campaign_id=cid, phase="delivered",
        asset_ref={"path": str(_a_png(conn, tmp_path))})["asset_id"]
    drift.classify(conn, campaign_id=cid, subject=asset, classification="neutral",
                   why="It went wider than briefed.", classified_by="R. Vega")

    stamped = store.drift_classifications(conn, cid)[0]
    assert not stamped["outcome_known"], (
        "a forecast was read as a known outcome, so a departure was stamped as judged "
        "against a result nobody has"
    )


# ---------------------------------------------------------------------------------------
# One implementation, asked by everybody


def test_the_library_wide_query_and_the_per_record_flag_cannot_disagree(conn):
    """The two-expressions-of-one-rule shape, handled the only way it can honestly be.

    "Has results" is asked two ways because the callers need two shapes: one SQL query for
    the library, and a per-record flag for a subset somebody cited. That is exactly the split
    this codebase has got wrong five times — so rather than trust that they agree, this walks
    every record and proves they cannot. If a future change to either one makes them differ,
    this is what says so."""
    _measured(conn, "Peru launch")
    _with_a_verdict(conn, "Lima launch")
    core.ingest_campaign(conn, title="No numbers", market="Peru", status="concluded",
                         detail="Nobody went back.", confirm=True)
    core.ingest_campaign(conn, title="Guidelines", record_type="reference",
                         detail="How we talk.", confirm=True)
    forecast = core.ingest_campaign(conn, title="Next quarter", market="Peru",
                                    status="proposed", detail="A plan.",
                                    confirm=True)["campaign_id"]
    core.add_metrics(conn, campaign_id=forecast, detail="We expect ROAS of 4.",
                     structured={"roas": 4.0}, metric_type="predicted", confirm=True)

    by_query = store.campaigns_with_actual_metrics(conn)

    for record in store.list_campaigns(conn):
        assert core.has_results(record) == (record["id"] in by_query), record["title"]


def test_a_forecast_is_not_a_measured_result(conn):
    """The distinction underneath both expressions, and the one §5.3 spent an item on: a
    predicted figure is the OPPOSITE of a measured outcome — it is the thing reconciliation
    later scores against the actuals — and counting it silenced the gap that asks for real
    numbers."""
    cid = core.ingest_campaign(conn, title="Next quarter", market="Peru", status="concluded",
                               detail="A plan.", confirm=True)["campaign_id"]
    core.add_metrics(conn, campaign_id=cid, detail="We expect ROAS of 4.",
                     structured={"roas": 4.0}, metric_type="predicted", confirm=True)

    state = core.library_state(conn)

    assert cid not in state["with_results"]
    assert cid in state["without_results"], "and it is still a finished campaign owing one"


def test_the_two_facts_are_computed_in_one_place(conn):
    """D75's ask. Four surfaces computing this separately is how they drift, and C16 was that
    failure once already. The helper answers both questions; nothing computes either
    privately."""
    plain = _measured(conn, "Peru launch")
    judged = _with_a_verdict(conn, "Lima launch")
    core.ingest_campaign(conn, title="No numbers at all", market="Peru", status="concluded",
                         detail="Nobody went back to it.", confirm=True)

    state = core.library_state(conn)

    assert state["with_results"] == {plain, judged}
    assert state["with_verdicts"] == {judged}
    assert state["basis"] == "computed"


def test_every_surface_asks_the_same_helper(conn):
    """The guard that makes the other tests mean something: a fifth private implementation
    added later re-creates the defect, and the only thing that catches that is a test which
    reads the code. Four surfaces are named in the item; this pins all of them plus every
    other reader, so the next one has to go through the helper too."""
    import inspect
    import re

    source = inspect.getsource(core)
    owners = {"library_state", "has_results", "has_a_verdict", "has_a_neutral_verdict",
              "carries_verdict"}
    bodies = _top_level_functions(source)

    # THE SCAN MUST MATCH ITS OWN EXEMPLARS. A first version keyed on a literal
    # `performed_well|underperformed` beside a `source == "verified"` check — which does not
    # appear in `has_a_verdict`'s own body, because that reads `_A_PERFORMANCE_VERDICT`. So a
    # verbatim copy of the real implementation pasted into another function passed the test
    # written to forbid exactly that. A structural test that cannot see the thing it models is
    # not a test, so this asserts the patterns fire on the owners before trusting their
    # silence about anybody else.
    def looks_private(body):
        hits = []
        if "campaigns_with_actual_metrics" in body or '"has_actual_metrics"' in body:
            hits.append("decides which campaigns have results")
        if re.search(r"(_A_PERFORMANCE_VERDICT|performed_well|underperformed)", body) \
                and re.search(r"""source["']\s*\)?\s*==\s*["']verified["']""", body):
            hits.append("decides which campaigns have a verdict")
        if re.search(r"filter_campaign_ids\([^)]*verified", body, re.S):
            hits.append("asks the store for verdict-tagged records")
        return hits

    for owner in ("has_results", "has_a_verdict"):
        assert looks_private(bodies[owner]), (
            f"the scan does not match {owner}, the implementation it exists to forbid "
            f"copies of — so a verbatim copy elsewhere would pass"
        )

    private = [f"{name} {hit}" for name, body in bodies.items() if name not in owners
               for hit in looks_private(body)]

    assert not private, (
        f"these compute one of §13.1's facts privately: {private}. Four separate "
        f"implementations is what D75 and D88 are; a fifth is the same defect."
    )


def _top_level_functions(source: str) -> dict:
    """`{name: body}` for every top-level def, so a test can ask which one a line is in."""
    import ast

    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = "".join(lines[node.lineno - 1:node.end_lineno])
    return out


# ---------------------------------------------------------------------------------------
# And computed the same way for everybody


def test_a_superseded_record_is_not_counted_as_evidence_by_one_surface_and_not_another(conn):
    """The drift a shared helper exists to stop, and the one most likely to have happened
    already: a replaced version is still a row with metrics on it. Whether it counts is a
    real decision — it is NOT independent evidence, because it is the same campaign — and the
    point is that it must be the same decision everywhere."""
    first = _measured(conn, "Peru launch v1")
    second = core.ingest_campaign(conn, title="Peru launch v2", market="Peru",
                                  status="concluded", detail="A store opening, revised.",
                                  supersedes=first, confirm=True)["campaign_id"]
    core.add_metrics(conn, campaign_id=second, detail="ROAS came in at 4.1.",
                     structured={"roas": 4.1}, metric_type="actual", confirm=True)

    state = core.library_state(conn)

    assert first not in state["with_results"], (
        "a superseded version is the same campaign counted twice"
    )
    assert second in state["with_results"]


def test_a_stub_is_results_and_counts_as_such(conn):
    """`library_state`'s docstring argues stubs belong in `with_results` — a stub is "a
    placeholder with results but no brief, e.g. a row imported from a KPI workbook", so it is
    the one record type that is results by definition — and nothing pinned it. Review mutated
    `holds_results` to drop stubs and the whole suite stayed green, which means the argument
    was in a comment and not in the product."""
    stub = core.ingest_campaign(conn, title="From the KPI workbook", record_type="stub",
                                market="Peru", status="concluded",
                                detail="A row somebody imported.", confirm=True)["campaign_id"]
    core.add_metrics(conn, campaign_id=stub, detail="ROAS 3.4.",
                     structured={"roas": 3.4}, metric_type="actual", confirm=True)

    state = core.library_state(conn)

    assert stub in state["with_results"] and stub in state["measured"]
    # And it is NOT one of the campaigns the library describes: a stub has no brief, so
    # nothing can learn a judgment from it. The two populations are both in the helper and
    # each surface names the one it means.
    assert stub not in {c["id"] for c in state["campaigns"]}


def test_a_stub_without_a_status_is_not_counted_as_having_run(conn):
    """A stub arrives with no status when it is imported, and `measured` is the intersection
    with `ran`. So the KPI row counts as having results and not as a finished campaign — the
    honest reading, because nobody recorded that it concluded."""
    stub = core.ingest_campaign(conn, title="From the KPI workbook", record_type="stub",
                                market="Peru", detail="A row somebody imported.",
                                confirm=True)["campaign_id"]
    core.add_metrics(conn, campaign_id=stub, detail="ROAS 3.4.",
                     structured={"roas": 3.4}, metric_type="actual", confirm=True)

    state = core.library_state(conn)

    assert stub in state["with_results"]
    assert stub not in state["ran"] and stub not in state["measured"]


def test_a_reference_record_never_counts_as_a_measured_campaign(conn):
    """Brand guidelines do not have results. `_campaigns_that_could_be_tagged` excluded
    reference records and the other three readers did not, so a library of guidelines with a
    stray number on one of them read as partly measured."""
    cid = core.ingest_campaign(conn, title="Brand guidelines", record_type="reference",
                               detail="How we talk about the brand.", confirm=True,
                               market="Peru")["campaign_id"]
    core.add_metrics(conn, campaign_id=cid, detail="Referenced 12 times.",
                     structured={"references": 12}, metric_type="actual", confirm=True)

    state = core.library_state(conn)

    assert cid not in state["with_results"]
    assert cid not in state["with_verdicts"]


def test_coverage_does_not_report_no_measured_cells_and_a_measured_library(conn):
    """Coverage held two answers inside one function. A cell checks the STATUS first — a
    campaign that has not concluded reads `not_yet_run` however many numbers are on it — and
    the library-level "is everything thin" test had no status test at all.

    So a library whose only measured campaign is still proposed reported zero measured cells
    and, in the same response, refused to say the library was unmeasured: `thin: []`,
    `thin_total: 0`, no offer, nothing anybody could act on."""
    cid = core.ingest_campaign(conn, title="Next quarter", market="Peru", status="proposed",
                               detail="A plan, with last year's numbers attached.",
                               confirm=True)["campaign_id"]
    core.add_metrics(conn, campaign_id=cid, detail="ROAS came in at 3.4.",
                     structured={"roas": 3.4}, metric_type="actual", confirm=True)

    report = core.coverage(conn)

    assert not any(c["evidence"] == "measured" for c in report["cells"])
    assert report["thin_summary"], (
        "no cell is measured and the library is reported as not thin, so nothing is offered"
    )
    # And it says WHY, rather than "none with results on file" — which is false here: the
    # campaign has results, it has not concluded. The two reasons want different things done.
    assert "already carry results but have not concluded" in report["thin_summary"]
    assert "none with results on file" not in report["thin_summary"]
    # The offer follows from that. `_offer_to_measure` would ask for numbers that are already
    # on file, or fall through to the first-run path on a library that is not new.
    offered = {a["tool"] for a in report["next_actions"]}
    assert offered == {"update_campaign"}, offered
    assert report["next_actions"][0]["prefilled_args"]["status"] == "concluded"


def test_readiness_and_coverage_agree_about_a_library_that_has_not_finished(conn):
    """The contradiction the FIRST round of §13.1 introduced while fixing the other one.

    Giving `coverage` a status test and not `readiness` left them saying of one library "all
    3 campaign(s) here have measured results" and "none with results on file" — D88's sentence
    again, with both halves now about the SAME fact, so the two-facts distinction cannot
    excuse it. Membership is decided in one place or it is decided four times."""
    for n in range(3):
        cid = core.ingest_campaign(conn, title=f"Plan {n}", market="Peru", status="proposed",
                                   detail="A plan.", confirm=True)["campaign_id"]
        core.add_metrics(conn, campaign_id=cid, detail="ROAS 3.4.",
                         structured={"roas": 3.4}, metric_type="actual", confirm=True)

    ready = core.readiness(conn)
    report = core.coverage(conn)

    can_say = [c for c in ready["can"] if c["code"] == "say_what_worked"]
    assert not can_say, (
        "readiness says it can say what worked about campaigns that have not run, while "
        "coverage says the library holds nothing measured"
    )
    assert report["thin_summary"]


def test_a_listing_still_works_on_a_database_that_predates_the_metrics_table(conn):
    """The FOURTH function on a save path to reach an upgraded v0.2.0 database, and it got
    there because §13.1 routed the disconfirming search through the shared library state —
    nothing had called `list_campaigns` on that path before, so the missing guard had never
    been reachable. A listing that raises is a listing no migration can be run from."""
    store.insert_campaign(conn, title="From an older release", detail="Still here.")
    for table in ("metrics", "evaluations", "campaign_chunks", "assets"):
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()

    listed = store.list_campaigns(conn)

    assert [c["title"] for c in listed] == ["From an older release"]
    assert listed[0]["has_metrics"] is False
    assert listed[0]["has_actual_metrics"] is False
    assert core.has_results(listed[0]) is False


def test_the_wire_publishes_the_measured_flag_and_not_only_the_loose_one(conn):
    """`has_metrics` counts ANY metric row and was the only one published, so a campaign
    carrying nothing but what somebody HOPED for reported `has_metrics: true` to the client.
    §5.3 spent an item establishing that a forecast is the opposite of a measured outcome."""
    cid = core.ingest_campaign(conn, title="Next quarter", market="Peru", status="concluded",
                               detail="A plan.", confirm=True)["campaign_id"]
    core.add_metrics(conn, campaign_id=cid, detail="We expect ROAS of 4.",
                     structured={"roas": 4.0}, metric_type="predicted", confirm=True)

    row = next(c for c in store.list_campaigns(conn) if c["id"] == cid)

    assert row["has_metrics"] is True
    assert row["has_actual_metrics"] is False

    import inspect

    import mcp_server

    listing = inspect.getsource(mcp_server.list_campaigns.fn
                               if hasattr(mcp_server.list_campaigns, "fn")
                               else mcp_server.list_campaigns)
    assert "has_actual_metrics" in listing, "the client is told only the looser of the two"


def test_a_cancelled_campaign_is_not_a_library_missing_its_results(conn):
    """§12.4/D38 landed a status that means "it did not run", and every reader of "measured"
    is a reader of that too. A campaign that was called off is not a hole in the evidence and
    must never be counted as one — that is a gap nobody can close."""
    cancelled = core.ingest_campaign(conn, title="Cusco launch", market="Peru",
                                     status="cancelled", detail="The site fell through.",
                                     confirm=True)["campaign_id"]
    _measured(conn, "Peru launch")

    state = core.library_state(conn)

    assert cancelled not in state["ran"], "a cancelled campaign never ran"
    assert cancelled not in state["without_results"], (
        "and is therefore never missing its results"
    )
