"""
§10.2 — the questions this product asks and cannot hear the answer to.

10.2 is about the menu's fixed numbering, and the tracker points four rows at it that are
really one thing: *a place for a person's answer to go.* The numbered queue is where an
unanswered question waits; it is worth building only if answering one is possible.

  D84   A departure's rationale has nowhere to live. The response asks "is this deliberate?",
        and when the marketer answers, nothing records it — so the stored finding still reads
        `unexplained` after it has been explained, and the same question returns next time.
  D59   A human-answerable path for `no_longer_raised`: "yes, we fixed that" can only be
        recorded by uploading a whole new version and having it judged.
  D53   An "known_not_yet" or "not applicable" state for a gap the user cannot close.
  D110  Deciding two rules are the same is a person's answer, so a library where nobody
        answers accumulates near-duplicates that each stay at one market and never graduate.
        The question is asked only on the write that raised it.

The shape is the same every time, and it is the worst shape this product has: **a question
asked on a surface with no return path.** It is worse than not asking, for two reasons.

The first is that the question comes back. A brief judged today asks "is the six-week
timeline deliberate?"; the marketer says "yes, the client moved the date" — into a chat
window, where it dies. The next version is judged and asks again. The product looks like it
is not listening, because it is not.

The second is that the un-answer is STORED. `departure: unexplained` is a fact about a
finding, and after somebody explains it, that fact is false and the library keeps asserting
it — to every later judgment that retrieves the record, in the field a model reads to decide
severity. A question nobody can answer decays into a wrong answer.

So: one table, two writes. A finding can be settled (`fixed`, `deliberate`, `not_applicable`)
and a gap can be acknowledged. D110 needs neither — `resolve_correction` already exists — it
needs to be ASKED again, which is the queue's job.
"""
import pytest

import core
import corrections
import feedback
import store


def _campaign(conn, title="Colombia", **kw):
    kw.setdefault("market", "LATAM")
    kw.setdefault("status", "concluded")
    kw.setdefault("detail", f"A campaign called {title}, which ran in a market.")
    return core.ingest_campaign(conn, title=title, **kw)["campaign_id"]


def _judged(conn, campaign_id, *, departure="unexplained", severity="should_fix"):
    """A saved judgment carrying one precedent_departure — the finding D84 is about."""
    return core.save_evaluation(
        conn, subject_title="Colombia", campaign_id=campaign_id, verdict="revise",
        summary="Close to Peru, with one difference nobody has explained.",
        approve_if="The timeline runs eight weeks, or the brief says why six is right.",
        findings=[{
            "severity": severity, "kind": "precedent_departure", "departure": departure,
            "finding": "The timeline is six weeks where every comparable ran eight.",
            "fix": "Extend to eight weeks, or say why six is right here.",
            "precedent": {"campaign_id": campaign_id,
                          "quote": "which ran in a market"},
        }])


# ── D84: the rationale that had nowhere to live ────────────────────────────

def test_a_departure_can_be_explained(conn):
    """"The response asks 'is this deliberate?', and when the marketer answers, nothing
    records it." This is the answer landing somewhere."""
    cid = _campaign(conn)
    judged = _judged(conn, cid)
    finding_id = judged["findings"][0]["id"]

    settled = core.answer_finding(conn, evaluation_id=judged["evaluation_id"],
                                  finding_id=finding_id, answer="deliberate",
                                  note="The client moved the launch date; six weeks was the "
                                       "brief we were given.",
                                  said_by="R. Vega")

    assert settled["answer"] == "deliberate"
    assert settled["said_by"] == "R. Vega"


def test_the_answer_is_attached_to_the_stored_finding(conn):
    """The half that matters more: the answer has to reach every reader of the judgment, not
    only the caller who recorded it. Attached at `store.get_evaluation`, the one place a
    judgment's full findings load.

    BESIDE `departure`, not over it — see `test_the_stored_departure_keeps_the_word_the_model
    _wrote` in test_review_round_one.py, where rewriting it crashed `diff_campaigns` and made
    the finding unreachable by the filter that would have found it."""
    cid = _campaign(conn)
    judged = _judged(conn, cid)
    finding_id = judged["findings"][0]["id"]

    core.answer_finding(conn, evaluation_id=judged["evaluation_id"], finding_id=finding_id,
                        answer="deliberate", note="The client moved the launch date.",
                        said_by="R. Vega")

    stored = next(f for f in core.get_evaluation(
        conn, evaluation_id=judged["evaluation_id"])["findings"] if f["id"] == finding_id)
    assert stored["settled"]["note"] == "The client moved the launch date."
    assert stored["settled"]["basis"] == "stated", (
        "a person said this; the library did not work it out"
    )


def test_the_explanation_travels_to_the_next_judgment(conn):
    """"the same question returns next time" — which is the symptom a marketer actually
    reports. The rationale has to reach the brief that would otherwise ask again."""
    cid = _campaign(conn)
    judged = _judged(conn, cid)
    core.answer_finding(conn, evaluation_id=judged["evaluation_id"],
                        finding_id=judged["findings"][0]["id"], answer="deliberate",
                        note="The client moved the launch date.", said_by="R. Vega")

    v2 = core.ingest_campaign(conn, title="Colombia v2", market="LATAM", status="proposed",
                              detail="The second version of the Colombia brief.",
                              supersedes=cid)["campaign_id"]
    prepared = core.prepare_evaluation(conn, subject_title="Colombia v2",
                                       proposal_text="The second version.", campaign_id=v2)

    carried = prepared["earlier_version"]["findings"][0]
    assert carried["settled"]["note"] == "The client moved the launch date."


# ── D59: "yes, we fixed that", said directly ───────────────────────────────

def test_a_finding_can_be_recorded_as_fixed(conn):
    """"A human-answerable path for `no_longer_raised`." Today the only way to record that a
    problem was addressed is to upload a whole new version and have it judged — which is a
    lot of work to say one word, and nobody does it."""
    cid = _campaign(conn)
    judged = _judged(conn, cid)

    settled = core.answer_finding(conn, evaluation_id=judged["evaluation_id"],
                                  finding_id=judged["findings"][0]["id"], answer="fixed",
                                  note="We went back to eight weeks.", said_by="R. Vega")

    assert settled["answer"] == "fixed"
    assert settled["status"] == "settled"


def test_fixed_and_deliberate_are_not_the_same_answer(conn):
    """D84's row says so in writing: "`resolved` is not it: it records 'we changed it', not
    'we kept it, and here is why'." Collapsing them would make the library unable to tell a
    brief that was corrected from one that was defended, which is the distinction the whole
    correction loop is built on."""
    cid = _campaign(conn)
    one, two = _judged(conn, cid), _judged(conn, _campaign(conn, "Peru"))

    core.answer_finding(conn, evaluation_id=one["evaluation_id"],
                        finding_id=one["findings"][0]["id"], answer="fixed",
                        note="We changed it back to eight weeks.", said_by="R. Vega")
    core.answer_finding(conn, evaluation_id=two["evaluation_id"],
                        finding_id=two["findings"][0]["id"], answer="deliberate",
                        note="We kept six weeks; the client moved the date.", said_by="R. Vega")

    answers = {core.get_evaluation(conn, evaluation_id=e["evaluation_id"])["findings"][0]
               ["settled"]["answer"] for e in (one, two)}
    assert answers == {"fixed", "deliberate"}


def test_an_answer_needs_somebody_to_have_said_it(conn):
    """Every `stated` claim in this product carries who said it. An anonymous explanation is
    an unattributed override of a computed finding, which is the one thing this library
    refuses everywhere else."""
    cid = _campaign(conn)
    judged = _judged(conn, cid)

    with pytest.raises(ValueError, match="said_by"):
        core.answer_finding(conn, evaluation_id=judged["evaluation_id"],
                            finding_id=judged["findings"][0]["id"], answer="deliberate",
                            note="Because.", said_by="")


def test_an_explanation_needs_to_explain_something(conn):
    """"Deliberate" with no reason is the same non-answer as `unexplained`, recorded as
    though it were an answer — strictly worse, because it stops the question being asked."""
    cid = _campaign(conn)
    judged = _judged(conn, cid)

    with pytest.raises(ValueError, match="note"):
        core.answer_finding(conn, evaluation_id=judged["evaluation_id"],
                            finding_id=judged["findings"][0]["id"], answer="deliberate",
                            note="", said_by="R. Vega")


def test_a_finding_that_is_not_on_the_judgment_is_refused(conn):
    cid = _campaign(conn)
    judged = _judged(conn, cid)

    with pytest.raises(ValueError, match="not a finding"):
        core.answer_finding(conn, evaluation_id=judged["evaluation_id"],
                            finding_id="find_nonesuch", answer="fixed",
                            note="Done.", said_by="R. Vega")


def test_a_finding_somebody_fixed_stops_reading_as_we_cannot_tell(conn):
    """The half of D59 that makes the write worth making. `no_longer_raised` is the "we
    cannot tell" bucket, and all three of its caveats end in *"though nobody recorded it"* —
    which is the whole row's complaint. If `answer='fixed'` does not move the finding out of
    that bucket, the answer is a field nothing reads, which is D116's shape one level down:
    a value written by a tool nobody consumes cannot fail, so nothing reports it broken."""
    one = _campaign(conn, "Colombia v1")
    judged = _judged(conn, one)
    core.answer_finding(conn, evaluation_id=judged["evaluation_id"],
                        finding_id=judged["findings"][0]["id"], answer="fixed",
                        note="We went back to eight weeks.", said_by="R. Vega")
    two = core.ingest_campaign(conn, title="Colombia v2", market="LATAM", status="concluded",
                               detail="The second version, at eight weeks.",
                               supersedes=one)["campaign_id"]
    core.save_evaluation(conn, subject_title="Colombia v2", campaign_id=two,
                         verdict="approve", summary="Fine now.", findings=[])

    diffed = core.diff_campaigns(conn, earlier=one, later=two)

    adopted = {f["id"] for f in diffed["adopted"]}
    assert judged["findings"][0]["id"] in adopted, (
        f"a finding somebody explicitly recorded as fixed is still in "
        f"{[k for k in diffed if diffed.get(k) and k != 'adopted']}"
    )


def test_a_person_saying_it_is_fixed_is_stated_not_computed(conn):
    """The rest of `adopted` is `computed` — a `resolved` entry on the later judgment, which
    the SERVER matched by id. This one is somebody's word, and a reader who cannot tell them
    apart will eventually cite one as the other. §2.4, on the surface §2.4 is about."""
    one = _campaign(conn, "Colombia v1")
    judged = _judged(conn, one)
    core.answer_finding(conn, evaluation_id=judged["evaluation_id"],
                        finding_id=judged["findings"][0]["id"], answer="fixed",
                        note="We went back to eight weeks.", said_by="R. Vega")
    two = core.ingest_campaign(conn, title="Colombia v2", market="LATAM", status="concluded",
                               detail="The second version.", supersedes=one)["campaign_id"]
    core.save_evaluation(conn, subject_title="Colombia v2", campaign_id=two,
                         verdict="approve", summary="Fine now.", findings=[])

    entry = core.diff_campaigns(conn, earlier=one, later=two)["adopted"][0]

    assert entry["basis"] == "stated"
    assert entry["settled"]["said_by"] == "R. Vega"


def test_settling_a_finding_does_not_offer_to_hide_the_other_record(conn):
    """Putting `answer_finding` entries into `adopted` reaches `_offer_to_link`, whose rule is
    "a statement, not a resemblance" and whose `why` says *"a judgment of one names a finding
    from the judgment of the other by id"*. Somebody settling a finding on v1 says nothing
    about v2 at all — so counting it would make that sentence false, on the one offer whose
    acceptance removes a record from every future search."""
    one = _campaign(conn, "Colombia v1")
    judged = _judged(conn, one)
    core.answer_finding(conn, evaluation_id=judged["evaluation_id"],
                        finding_id=judged["findings"][0]["id"], answer="fixed",
                        note="We went back to eight weeks.", said_by="R. Vega")
    two = _campaign(conn, "Colombia, a separate brief")
    core.save_evaluation(conn, subject_title="Colombia, a separate brief", campaign_id=two,
                         verdict="approve", summary="Unrelated, and fine.", findings=[])

    diffed = core.diff_campaigns(conn, earlier=one, later=two)

    assert diffed["adopted"], "the fixture has to actually produce a settled adoption"
    assert "update_campaign" not in [a["tool"] for a in diffed.get("next_actions") or []]


def test_a_finding_nobody_answered_still_says_nobody_recorded_it(conn):
    """The silence half, and the one that matters: `no_longer_raised`'s honesty is the point
    of it. A change that moved every unmatched finding into `adopted` would be the product
    crediting itself for work nobody did — which is what the bucket exists to refuse."""
    one = _campaign(conn, "Colombia v1")
    judged = _judged(conn, one)
    two = core.ingest_campaign(conn, title="Colombia v2", market="LATAM", status="concluded",
                               detail="The second version.", supersedes=one)["campaign_id"]
    core.save_evaluation(conn, subject_title="Colombia v2", campaign_id=two,
                         verdict="approve", summary="Fine now.", findings=[])

    diffed = core.diff_campaigns(conn, earlier=one, later=two)

    assert [f["id"] for f in diffed["no_longer_raised"]] == [judged["findings"][0]["id"]]
    assert not diffed["adopted"]


# ── D53: a gap the user cannot close ───────────────────────────────────────

def _a_market_nobody_can_measure(conn):
    """The case D53 exists for, and one `_CAN_BE_SET_ASIDE` actually admits: a market whose
    campaigns were run by an agency that no longer exists, whose numbers are gone."""
    measured = _campaign(conn, "Peru", market="Peru")
    core.add_metrics(conn, campaign_id=measured, structured={"roas": 3.4}, confirm=True)
    _campaign(conn, "An old Chile activation", market="Chile")
    return "market_without_outcomes"


def _a_library_whose_worst_gap_can_be_set_aside(conn):
    """Everything measured, nothing dated. The only shape where a set-asideable gap is also
    the TOP one — `few_verified_outcomes` and `market_without_outcomes` both need something
    unmeasured, and both outrank `no_window`."""
    for title in ("Peru", "Lima"):
        cid = _campaign(conn, title, market="Peru")
        core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    return "no_window"


def test_a_gap_can_be_set_aside(conn):
    """"A `known_not_yet` or `not_applicable` state for a gap the user cannot close." Some
    gaps are true and permanent: a market whose campaigns were run by an agency that no
    longer exists will never have its results. Reporting it every time is a complaint."""
    code = _a_market_nobody_can_measure(conn)

    core.answer_gap(conn, code=code, answer="not_applicable",
                    note="That agency no longer exists; the numbers went with it.",
                    said_by="R. Vega")

    assert code not in [g["code"] for g in core.gaps(conn)["gaps"]]


def test_the_gap_behind_the_whole_review_cannot_be_silenced(conn):
    """One rule, not two. `_CAN_BE_SET_ASIDE` gated the OFFER and the write path took
    anything — so `answer_gap(code="library_is_empty", answer="not_applicable")` was accepted
    on an empty library and `gaps()` then returned `[]`: the product saying nothing is
    missing about a library holding nothing.

    "This will never be true here" is honest about a market whose agency is gone. It is not
    honest about work nobody has done — and `few_verified_outcomes` is the gap this whole
    product was built around, while silencing `judgments_never_reconciled` would make its own
    calibration figure unfalsifiable."""
    cid = _campaign(conn, "Peru", market="Peru")
    core.save_evaluation(conn, subject_title="Peru", campaign_id=cid, verdict="approve",
                         summary="Fine.", findings=[],
                         predictions={"predicted_roi_range": "3.0 to 4.0"})
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    _campaign(conn, "Bogota", market="Colombia")
    reported = {g["code"] for g in core.gaps(conn)["gaps"]}
    assert {"few_verified_outcomes", "judgments_never_reconciled"} <= reported, reported

    for code in ("few_verified_outcomes", "judgments_never_reconciled"):
        with pytest.raises(ValueError, match="cannot be set aside"):
            core.answer_gap(conn, code=code, answer="not_applicable",
                            note="We would rather not hear about this.", said_by="R. Vega")


def test_a_gap_set_aside_is_still_true_and_still_says_so(conn):
    """Silenced in the ranking, not deleted from the record. A gap somebody set aside is
    still a fact about the library's evidence, and a judgment resting on that evidence is
    still weaker for it — hiding it entirely would make the product overstate its own
    coverage, which is the failure the whole review is about."""
    code = _a_market_nobody_can_measure(conn)
    core.answer_gap(conn, code=code, answer="not_applicable",
                    note="That agency no longer exists; the numbers went with it.",
                    said_by="R. Vega")

    setting = core.gaps(conn)["set_aside"]

    assert setting[0]["code"] == code
    assert setting[0]["said_by"] == "R. Vega"
    assert setting[0]["basis"] == "stated"


def test_a_gap_that_closed_on_its_own_stops_reading_as_set_aside(conn):
    """A gap set aside in March and genuinely closed in June is not set aside any more, it is
    closed — and still listing it says the library is choosing not to look at something it no
    longer has, which is a worse misreading than not mentioning it at all."""
    code = _a_market_nobody_can_measure(conn)
    core.answer_gap(conn, code=code, answer="not_applicable",
                    note="That agency no longer exists.", said_by="R. Vega")
    assert [g["code"] for g in core.gaps(conn)["set_aside"]] == [code]

    for record in store.list_campaigns(conn):
        core.add_metrics(conn, campaign_id=record["id"], structured={"roas": 2.0},
                         confirm=True)

    assert not core.gaps(conn)["set_aside"]


def test_known_not_yet_is_not_the_same_as_not_applicable(conn):
    """`known_not_yet` means "I know, and I am not doing it yet" — it stays ranked, because it
    is still the thing to fix. `not_applicable` means "this will never be true here". Making
    them one state would either nag somebody who has answered or silence a real gap."""
    code = _a_market_nobody_can_measure(conn)

    core.answer_gap(conn, code=code, answer="known_not_yet",
                    note="Chasing the client for the numbers this quarter.",
                    said_by="R. Vega")

    gap = next(g for g in core.gaps(conn)["gaps"] if g["code"] == code)
    assert gap["answered"]["note"] == "Chasing the client for the numbers this quarter."


def test_the_only_offer_on_a_gap_is_never_to_silence_it(conn):
    """`_ranked` drops an offer that another gap already makes and leaves `closed_by` pointing
    at the gap that makes it — so appending the set-aside offer afterwards made it the ONLY
    thing on that row, and the product's single suggestion about a real hole in its evidence
    was "shall we stop mentioning this?".

    Two campaigns, one measured, is the smallest library that reproduces it: the barren
    market's only candidate IS the campaign `few_verified_outcomes` already offered, so
    `market_without_outcomes` falls back to it and the offer is de-duplicated away."""
    measured = _campaign(conn, "Peru", market="Peru")
    _campaign(conn, "Chile", market="Chile")
    core.add_metrics(conn, campaign_id=measured, structured={"roas": 3.4}, confirm=True)

    gaps = core.gaps(conn)["gaps"]

    silenced = [g["code"] for g in gaps
                if [a["tool"] for a in g.get("next_actions") or []] == ["answer_gap"]]
    assert not silenced, (
        f"the only thing offered about {silenced} is to stop reporting it"
    )
    assert any(g.get("closed_by") for g in gaps), (
        "the fixture has to actually produce a gap whose own offer was de-duplicated, or "
        "this test is checking a case that never arises"
    )


def test_a_gap_the_user_cannot_close_still_offers_the_way_out(conn):
    """The other side of the same rule. Suppressing the set-aside offer everywhere would make
    D53 unreachable, and the case it exists for is real: a market whose agency no longer
    exists will never have its results, and reporting that forever is a complaint.

    `_CAN_BE_SET_ASIDE` is a whitelist, so this asserts that the whitelist is not empty in
    practice — a tool reachable only in theory is the state D116 was written about."""
    ids = [_campaign(conn, f"Campaign {n}", market=m)
           for n, m in enumerate(["LATAM", "EMEA", "APAC", "LATAM", "LATAM"])]
    core.add_metrics(conn, campaign_id=ids[0], structured={"roas": 3.4}, confirm=True)

    offered = {g["code"] for g in core.gaps(conn)["gaps"]
               if "answer_gap" in [a["tool"] for a in g.get("next_actions") or []]}

    assert offered and offered <= set(core._CAN_BE_SET_ASIDE)
    assert "library_is_empty" not in offered


def test_setting_aside_the_top_gap_moves_most_valuable(conn):
    """`most_valuable` is recomputed after the answers are applied, and it has to be: naming a
    gap that is no longer in `gaps` is the inconsistent pair `coverage` was fixed for, and it
    is the field `gaps_offer` and every caller read to decide what to say first."""
    code = _a_library_whose_worst_gap_can_be_set_aside(conn)
    top = core.gaps(conn)["most_valuable"]
    assert top == code, f"the fixture has to make the set-asideable gap the top one: {top}"

    core.answer_gap(conn, code=code, answer="not_applicable",
                    note="Nobody recorded dates for these and the decks are gone.",
                    said_by="R. Vega")

    after = core.gaps(conn)
    assert after["most_valuable"] != top
    # `most_valuable` names the top RANKED gap, and `gaps` also carries the field gaps that
    # are appended after ranking and carry no offer — so `None` beside a non-empty list is
    # the honest answer here: nothing left is something this library can hand you an action
    # for. What must never differ is `most_valuable` and what the offer surface acts on.
    # §12.4/D51 added `commentary_never_read`, which IS offerable and IS set-asideable — so
    # the honest answer after setting the window gap aside is the next offerable gap rather
    # than None. What must never differ is `most_valuable` and what the offer surface acts on,
    # which is the assertion below.
    assert after["most_valuable"] == "commentary_never_read"
    assert all(g["code"] in ("field_never_recorded", "commentary_never_read")
               for g in after["gaps"]), after["gaps"]
    assert core.top_gap(conn) == after["most_valuable"], (
        "the session-start offer would fire on a gap the ranking no longer shows"
    )


def test_a_gap_set_aside_with_no_reason_is_refused(conn):
    """Same rule as a finding: a gap set aside with no reason is one nobody can reopen
    intelligently, and it silences a real hole in the evidence permanently on one call."""
    code = _a_market_nobody_can_measure(conn)

    with pytest.raises(ValueError, match="note"):
        core.answer_gap(conn, code=code, answer="not_applicable", note="   ",
                        said_by="R. Vega")


def test_a_one_record_library_claims_nothing_outweighs_its_kind(conn):
    """`affects > 1` in the magnitude rule. With one record every gap is about 100% of the
    library, so `outweighs_its_kind` would be true of all of them — a published claim that
    says nothing, on the library where the ranking is least informative anyway.

    It cannot change the ORDER there (a uniform boost is no boost), which is exactly why it
    needs its own test: the field is the only thing it protects, and a test that checked the
    ordering would pass with the guard deleted."""
    _campaign(conn, "The only one")

    gaps = core.gaps(conn)["gaps"]

    assert gaps, "the fixture has to produce at least one gap"
    assert not any(g["outweighs_its_kind"] for g in gaps), [
        (g["code"], g["affects"], g["share"]) for g in gaps]


def test_a_gap_nobody_has_is_refused(conn):
    """Never a silent no-op: "known_not_yet" for a code the library does not rank reads as
    done and is not."""
    _campaign(conn)

    with pytest.raises(ValueError, match="not a gap"):
        core.answer_gap(conn, code="nonesuch", answer="known_not_yet",
                             note="A real note.", said_by="R. Vega")


def test_acknowledging_can_be_undone(conn):
    """The whole point of a person's answer is that a person can change it — and a permanent
    silence bought with one call is how a library stops reporting a gap it really has."""
    code = _a_market_nobody_can_measure(conn)
    core.answer_gap(conn, code=code, answer="not_applicable",
                    note="That agency no longer exists.", said_by="R. Vega")

    core.answer_gap(conn, code=code, answer="open",
                    note="The old agency found the numbers after all.", said_by="R. Vega")

    assert code in [g["code"] for g in core.gaps(conn)["gaps"]]


# ── D110: a question asked once is a question nobody answers ───────────────

def test_two_rules_that_may_be_the_same_wait_in_the_queue(conn):
    """"a library where nobody answers accumulates near-duplicates that each stay at one
    market and never graduate." The question is asked on the write that raised it and nowhere
    else — so it is asked exactly once, to whoever happened to be uploading, and a `no` is
    indistinguishable from nobody having seen it.

    Row 12, not a row among the campaigns: the menu's header asks which CAMPAIGN, and this
    has none. See `feedback.RULES_ROW`."""
    _two_similar_rules(conn)

    menu = feedback.queue(conn)
    gateway = next(r for r in menu["rows"] if r["number"] == feedback.RULES_ROW)
    row = next(r for r in feedback.queue(conn, scope="rules")["rows"]
               if r.get("action") == "resolve_correction")

    assert "1 pair(s)" in gateway["title"]
    # "Every row says why it is open" — and for this one, why it is open is also what it
    # costs: each wording counting separately is exactly why neither ever graduates.
    assert "one rule" in row["why"]
    assert "recurs" in row["why"]


def test_the_queue_row_carries_both_wordings(conn):
    """A numbered row reading "two rules may be the same" that does not say WHICH two is a
    question nobody can answer from the menu — which is the state it is being lifted out of."""
    _two_similar_rules(conn)

    row = next(r for r in feedback.queue(conn, scope="rules")["rows"]
               if r.get("action") == "resolve_correction")

    # BOTH wordings, in full. They matched because their openings are near-identical, so a
    # truncated pair showed the user two visually identical strings and asked if they were
    # the same thing.
    assert row["title"].count("dark background") == 2
    assert row["needs"]


def _two_similar_rules(conn):
    cid = _campaign(conn)
    first = corrections.note(conn, text="Never put the logo on a dark background.",
                             provenance="client email, 4 March", campaign_id=cid)
    second = corrections.note(conn, text="Never place the logo on a dark background.",
                              provenance="client call, 9 March", campaign_id=cid)
    return first, second


def test_choosing_the_row_asks_the_question(conn):
    """Every other row on this menu is about a campaign, and `_ask_about` reaches straight for
    `row["campaign_id"]`. The first version of this branch shipped with `menu["token"]` where
    the key is `menu_token`, so picking the number raised a KeyError and reached the model as
    "Error executing tool" — and every test above passed, because they all READ the row and
    none of them picked it. A menu row nothing selects is a row nobody has tried."""
    _two_similar_rules(conn)
    menu = feedback.queue(conn, scope="rules")
    row = next(r for r in menu["rows"] if r.get("action") == "resolve_correction")

    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=row["number"])

    assert asked["ask"] == "Are these the same rule?"
    assert sorted(asked["texts"]) == ["Never place the logo on a dark background.",
                                      "Never put the logo on a dark background."]
    assert {a["prefilled_args"]["decision"] for a in asked["next_actions"]} == {
        "same_rule", "different_rule", "set_aside"}


def test_the_answer_offered_can_actually_be_carried_out(conn):
    """"An offer has to work" is this project's own rule for `next_actions`, and it is the
    rule a menu makes easiest to break: the user says a number, then yes, and an offer whose
    arguments have drifted fails as an error they did not cause and cannot fix."""
    _two_similar_rules(conn)
    menu = feedback.queue(conn, scope="rules")
    row = next(r for r in menu["rows"] if r.get("action") == "resolve_correction")
    asked = feedback.choose(conn, menu_token=menu["menu_token"], choice=row["number"])
    same = next(a for a in asked["next_actions"]
                if a["prefilled_args"]["decision"] == "same_rule")

    folded = corrections.resolve(conn, **same["prefilled_args"])

    assert folded["status"] == "merged"
    assert not [r for r in feedback.queue(conn, scope="rules")["rows"]
                if r.get("action") == "resolve_correction"]


def test_choosing_a_pair_somebody_answered_meanwhile_refreshes(conn):
    """§10.5's rule, on the one row that is not about a campaign. Two people working the
    queue is the ordinary case, and the second must not write against a question the first
    has already settled.

    It is the TOKEN that catches this, not a check inside the branch: the token is computed
    over the rows, and an answered pair is not a row any more. Worth a test anyway, because
    "the token covers it" is an argument about a row shape that did not exist when the token
    was written — a question row carries no `campaign_id`, and `_token` reads that field."""
    first, second = _two_similar_rules(conn)
    menu = feedback.queue(conn, scope="rules")
    row = next(r for r in menu["rows"] if r.get("action") == "resolve_correction")
    corrections.resolve(conn, second["correction_id"], decision="same_rule",
                        same_as=first["correction_id"])

    out = feedback.choose(conn, menu_token=menu["menu_token"], choice=row["number"])

    assert out["status"] == "refreshed"


def test_the_token_notices_a_question_row_appearing(conn):
    """The token's own premise, checked against the new row shape. `_token` hashes
    `campaign_id`, `action` and `needs` per row — a question row has no `campaign_id`, so if
    those three did not distinguish it, a menu with a new question in it would carry the
    token of the menu without it, and "3" would mean a different thing behind an unchanged
    key. That is the exact silent corruption §10.5 exists to make impossible."""
    _campaign(conn)
    before = feedback.queue(conn)["menu_token"]

    _two_similar_rules(conn)

    assert feedback.queue(conn)["menu_token"] != before


def test_one_row_per_pair_not_one_per_wording(conn):
    """Both wordings resemble each other, so the loop meets the same question twice with the
    two texts swapped — and an unguarded version renders it as two separate problems, which
    is the near-duplicate defect D110 is about reproduced inside the fix for it."""
    _two_similar_rules(conn)

    asking = [r for r in feedback.queue(conn, scope="rules")["rows"]
              if r.get("action") == "resolve_correction"]

    assert len(asking) == 1, [r["line"] for r in asking]


def test_a_pair_answered_as_two_rules_stops_being_asked(conn):
    """"It is its own rule" is one of §8.2's three answers and the only one that leaves BOTH
    corrections live and provisional — so it is the answer the status check cannot see, and
    the `asked` flag is the only thing that stops the question returning. A queue that keeps
    asking a settled question teaches people to stop reading it, and then the unsettled one
    goes unread with it."""
    _, second = _two_similar_rules(conn)

    answered = corrections.resolve(conn, second["correction_id"], decision="different_rule")

    assert answered["status"] != "ignored", (
        "this answer has to leave the correction LIVE, or the status check would catch it "
        "and `asked` would never be the thing under test"
    )
    assert not [r for r in feedback.queue(conn, scope="rules")["rows"]
                if r.get("action") == "resolve_correction"]


def test_an_answered_pair_stops_being_asked(conn):
    """A queue that keeps asking a settled question teaches people to stop reading it, and
    then the unsettled one goes unread with it."""
    first, second = _two_similar_rules(conn)

    corrections.resolve(conn, second["correction_id"], decision="same_rule",
                        same_as=first["correction_id"])

    assert not [r for r in feedback.queue(conn, scope="rules")["rows"]
                if r.get("action") == "resolve_correction"]


def test_a_library_with_no_near_duplicates_asks_nothing(conn):
    """The silence half. A queue row per correction would be a queue made of corrections."""
    cid = _campaign(conn)
    corrections.note(conn, text="Never put the logo on a dark background.",
                     provenance="client email, 4 March", campaign_id=cid)
    corrections.note(conn, text="Seed a single colourway per market.",
                     provenance="client call, 9 March", campaign_id=cid)

    assert not [r for r in feedback.queue(conn, scope="rules")["rows"]
                if r.get("action") == "resolve_correction"]


def test_a_different_pair_on_the_same_number_changes_the_token(conn):
    """§10.5's whole contract, on the row shape it was not written for. `_token` fingerprinted
    rows by `campaign_id`/`action`/`needs`, and a rule question has no campaign — so pair
    (A,B) being settled and pair (C,D) arriving in its place left the token UNCHANGED, and
    "3" silently meant a different question. That is the exact silent misfiling the token
    exists to make impossible, reappearing because the row shape changed and the fingerprint
    did not."""
    cid = _campaign(conn)
    first = corrections.note(conn, text="Never put the logo on a dark background.",
                             provenance="client email, 4 March", campaign_id=cid)
    second = corrections.note(conn, text="Never place the logo on a dark background.",
                              provenance="client call, 9 March", campaign_id=cid)
    before = feedback.queue(conn, scope="rules")["menu_token"]

    corrections.resolve(conn, second["correction_id"], decision="same_rule",
                        same_as=first["correction_id"])
    corrections.note(conn, text="Seed a single colourway per market.",
                     provenance="client email, 2 April", campaign_id=cid)
    corrections.note(conn, text="Seed one single colourway per market.",
                     provenance="client call, 8 April", campaign_id=cid)
    after = feedback.queue(conn, scope="rules")

    assert [r["number"] for r in after["rows"] if r.get("action") == "resolve_correction"] \
        == [1], "the new pair has to land on the number the old one had"
    assert after["menu_token"] != before
    assert feedback.choose(conn, menu_token=before, choice=1)["status"] == "refreshed"


def test_the_rule_question_does_not_make_every_upload_slow(conn):
    """`_looks_like` re-reads and re-tokenises every correction on file, so calling it once
    per correction was O(n²) over the whole table — and `ingest_campaign` calls `waiting()` on
    every upload, which calls this.

    Measured before the fix: 100 corrections 0.37s, 200 1.74s, 400 8.39s, for a single upload.
    At 400 the offer alone spent a third of the tool's whole time budget, on a library that is
    not large: an agency accumulates client corrections faster than it accumulates campaigns.
    """
    import time

    cid = _campaign(conn)
    for n in range(300):
        corrections.note(conn, text=f"Rule number {n}: never use the {n} treatment on a deck.",
                         provenance=f"client email, item {n}", campaign_id=cid)

    started = time.monotonic()
    feedback.waiting(conn)
    elapsed = time.monotonic() - started

    assert elapsed < 1.5, (
        f"one upload spent {elapsed:.1f}s counting what is waiting, on 300 corrections"
    )


def test_taking_an_answer_back_leaves_the_finding_unanswered(conn):
    """`open` is somebody withdrawing what they said, so the finding has to read as unanswered
    again — every reader treats the PRESENCE of `settled` as "answered", so attaching an
    `open` row would leave them with a block whose presence says one thing and whose content
    says the opposite. That is the two-fields-to-check failure this attachment exists to
    avoid, arriving through the escape hatch."""
    cid = _campaign(conn)
    judged = _judged(conn, cid)
    finding_id = judged["findings"][0]["id"]
    core.answer_finding(conn, evaluation_id=judged["evaluation_id"], finding_id=finding_id,
                        answer="deliberate", note="The client moved the date.",
                        said_by="R. Vega")
    assert "settled" in core.get_evaluation(
        conn, evaluation_id=judged["evaluation_id"])["findings"][0]

    core.answer_finding(conn, evaluation_id=judged["evaluation_id"], finding_id=finding_id,
                        answer="open", note="Misfiled — that was a different brief.",
                        said_by="R. Vega")

    stored = core.get_evaluation(conn, evaluation_id=judged["evaluation_id"])["findings"][0]
    assert "settled" not in stored
    assert len(core.answers(conn)["answers"]) == 2, "and the history is still kept"


def test_row_twelve_leads_to_the_rules_menu(conn):
    """The navigation itself, which only the protocol probe was exercising. Row 12 is the
    only row whose action is neither a campaign nor one of the two the menu has always had —
    so if it falls through, `_ask_about` reaches for `row["campaign_id"]` on a row that has
    none, and the model gets "Error executing tool" for pressing a number it was offered."""
    cid = _campaign(conn)
    _two_similar_rules(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    menu = feedback.queue(conn)
    gateway = next(r for r in menu["rows"] if r["number"] == feedback.RULES_ROW)
    assert gateway["action"] == "rules"

    opened = feedback.choose(conn, menu_token=menu["menu_token"],
                             choice=feedback.RULES_ROW)

    assert opened["scope"] == "rules"
    assert [r["number"] for r in opened["rows"] if r.get("action") == "resolve_correction"] \
        == [1]


def test_the_rules_menu_leads_back(conn):
    """Its last row is the way home, not "show more" — the rules menu has one page. Without
    the branch, `more` recomputed `page + 1` inside a one-page scope and clamped back to
    itself, so somebody who opened row 12 had no way out from inside the menu."""
    _two_similar_rules(conn)
    rules = feedback.queue(conn, scope="rules")
    back = next(r for r in rules["rows"] if r.get("action") == "more")
    assert "Back" in back["title"]

    home = feedback.choose(conn, menu_token=rules["menu_token"], choice=back["number"])

    assert home["scope"] == "open"


def test_a_prohibition_and_its_permission_are_never_one_pair(conn):
    """The rule that lives in `corrections.resembles` and that the queue's pairing must obey:
    "never use AI imagery" and "use AI imagery with approval" share every content word, so
    the similarity score alone puts them at 1.0. Suggesting they are one rule would put the
    OPPOSITE of a client's instruction on the checklist under their provenance.

    Worth its own test here because §10.2/D110 added a second caller of that comparison, and
    the first version of it re-derived the check rather than sharing it — mutation showed
    nothing would have noticed if that copy were deleted."""
    cid = _campaign(conn)
    corrections.note(conn, text="Never use AI imagery in a seeding box.",
                     provenance="client email, 4 March", campaign_id=cid)
    corrections.note(conn, text="Use AI imagery in a seeding box.",
                     provenance="client call, 9 March", campaign_id=cid)

    assert not feedback._rule_questions(conn)
    assert not [r for r in feedback.queue(conn, scope="rules")["rows"]
                if r.get("action") == "resolve_correction"]


def test_two_wordings_of_one_rule_are_still_paired(conn):
    """The other side. A check that refused every pair would pass the test above and make
    D110 unreachable."""
    cid = _campaign(conn)
    corrections.note(conn, text="Never use AI imagery in a seeding box.",
                     provenance="client email, 4 March", campaign_id=cid)
    corrections.note(conn, text="Never use AI imagery inside a seeding box.",
                     provenance="client call, 9 March", campaign_id=cid)

    assert len(feedback._rule_questions(conn)) == 1


def test_a_rule_too_short_to_compare_is_not_paired_with_anything(conn):
    """`_MIN_CONTENT` — "below this many content words there is not enough to compare, and a
    two-word rule matches half the library on one word."

    "Avoid red" against "Avoid red tones" scores 1.0 on the raw comparison, because every word
    of the shorter rule is in the longer one. That is not two wordings of one rule; it is one
    rule and a fragment, and offering to fold them puts the fragment's provenance onto the
    real rule. The guard is in `_reading`, so it holds for both callers rather than for
    whichever one remembered it."""
    cid = _campaign(conn)
    corrections.note(conn, text="Avoid red.", provenance="client email, 4 March",
                     campaign_id=cid)
    corrections.note(conn, text="Avoid red tones in any seeding box imagery.",
                     provenance="client call, 9 March", campaign_id=cid)

    assert not feedback._rule_questions(conn)
    assert corrections._looks_like(conn, "Avoid red.") is None
