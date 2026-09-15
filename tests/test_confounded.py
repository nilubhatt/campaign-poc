"""
§9.8: record the overlap. Do not assert the cause.

The review: *"**This is where a learning system most easily goes wrong.** 'Sell-through was down
and there was an earthquake' is not evidence the earthquake caused it. A model asked to explain
a disappointing number will reach for whatever is nearby. Fix: the system records the overlap as
a fact and marks the outcome `confounded`, with the event linked. It does not compute
attribution. A human can add an attribution note, stored as `stated`. **Confounded outcomes
still count** — they are just never quoted as clean evidence, and the caveat travels with the
metric everywhere it is cited."*

**"Still count" is the sentence that decides the design.** Dropping a confounded outcome would
throw away most of what a real library holds — every campaign that ran through a holiday, a
port closure or an election. So nothing here filters, excludes, down-ranks or reweights. The
number is the number; what changes is that a reader can no longer be told it was clean.

**"Everywhere it is cited" means one choke point, not a list of call sites.** A hand-maintained
list of places to attach a caveat is a copy of the codebase, and this project has been bitten by
that shape four times. Metrics load in exactly one function, so the caveat is attached there and
every reader gets it — including readers nobody has written yet.

**Not every overlap is a confounder, and this is the difference between a caveat and
wallpaper.** Every November campaign in the United States overlaps Black Friday; every Ramadan
campaign overlaps Ramadan. A recurring fixed date is the BASELINE a year-on-year comparison is
made against, not a confounder of it. So a `fixed_calendar` event confounds only when somebody
says it did, and what confounds by default is the thing that was not supposed to happen — an
earthquake, a port closure, a regulatory change, or any event whose recorder stated an impact.

**Attribution is a person's, always.** The server records that a metric and an event overlapped.
Whether the event moved the number is exactly the judgment the review says a model will reach
for, so it is `stated`, it needs a name against it, and there is no path by which the server
produces one.
"""
import pytest

import context
import core
import store


def _campaign(conn, title="Mexico launch", market="Mexico", **kw):
    kw.setdefault("detail", "A retail activation across Mexico City.")
    kw.setdefault("starts_on", "2026-09-01")
    kw.setdefault("ends_on", "2026-09-30")
    return core.ingest_campaign(conn, title=title, market=market, status="concluded",
                                **kw)["campaign_id"]


def _quake(conn, **kw):
    kw.setdefault("starts_on", "2026-09-10")
    kw.setdefault("ends_on", "2026-09-12")
    kw.setdefault("scope", "market")
    kw.setdefault("scope_value", "Mexico")
    kw.setdefault("kind", "natural_disaster")
    kw.setdefault("description", "Magnitude 7.1 earthquake; Mexico City retail shut 3 days.")
    kw.setdefault("recorded_by", "R. Vega")
    return context.record(conn, **kw)


def _metric(conn, campaign_id, **kw):
    kw.setdefault("structured", {"sell_through_pct": 41.0})
    kw.setdefault("confirm", True)
    return core.add_metrics(conn, campaign_id=campaign_id, **kw)


def _only_metric(conn, campaign_id):
    return store.get_campaign(conn, campaign_id)["metrics"][0]


# ── the mark ────────────────────────────────────────────────────────────────

def test_an_outcome_that_ran_through_an_event_is_marked_confounded(conn):
    cid = _campaign(conn)
    _quake(conn)

    _metric(conn, cid)

    metric = _only_metric(conn, cid)
    assert metric["confounded"] is True
    # `confounded_basis`, not `basis`: at the metric level a bare `basis: computed` would read
    # as "this measurement is computed", which is false — it is the OVERLAP that was worked
    # out, and each event inside keeps its own `stated`.
    assert metric["confounded_basis"] == "computed"
    assert metric["confounded_by"][0]["basis"] == "stated"


def test_the_event_is_linked(conn):
    """"Marks the outcome `confounded`, WITH THE EVENT LINKED." A caveat that cannot say what
    it is about is a warning label with no text."""
    cid = _campaign(conn)
    quake = _quake(conn)

    _metric(conn, cid)

    metric = _only_metric(conn, cid)
    assert [e["id"] for e in metric["confounded_by"]] == [quake["id"]]
    assert "earthquake" in metric["confounded_by"][0]["description"]


def test_an_outcome_with_nothing_around_it_is_not_marked(conn):
    cid = _campaign(conn)

    _metric(conn, cid)

    metric = _only_metric(conn, cid)
    assert metric["confounded"] is False
    assert metric["confounded_by"] == []


def test_the_event_can_arrive_after_the_numbers(conn):
    """The ordinary order: the results are filed in October and somebody records the
    September earthquake in November. A mark written when the metric was stored would never
    see it."""
    cid = _campaign(conn)
    _metric(conn, cid)
    assert _only_metric(conn, cid)["confounded"] is False

    _quake(conn)

    assert _only_metric(conn, cid)["confounded"] is True


def test_withdrawing_the_event_unmarks_the_outcome(conn):
    """An event recorded with the wrong year confounded everything it touched. §9.6 made it
    withdrawable; the mark has to follow, or the correction is half a correction."""
    cid = _campaign(conn)
    quake = _quake(conn)
    assert _only_metric(conn, _campaign_with_metric(conn, cid))["confounded"] is True

    context.withdraw(conn, event_id=quake["id"], why="Wrong year — this was 2025.",
                     withdrawn_by="R. Vega")

    assert _only_metric(conn, cid)["confounded"] is False


def _campaign_with_metric(conn, campaign_id):
    _metric(conn, campaign_id)
    return campaign_id


# ── a caveat that fires on everything is not a caveat ───────────────────────

def test_a_recurring_holiday_does_not_confound_by_itself(conn):
    """Every November campaign in the United States overlaps Black Friday and every Ramadan
    campaign overlaps Ramadan. A recurring fixed date is the BASELINE a year-on-year
    comparison is made against, not a confounder of it — and a caveat that travels with every
    metric in the library is one nobody reads."""
    cid = _campaign(conn, "US launch", market="United States",
                    starts_on="2026-11-20", ends_on="2026-12-05")

    _metric(conn, cid)

    metric = _only_metric(conn, cid)
    assert metric["confounded"] is False
    assert metric["ran_during"] >= 1 >= 1, (
        "the overlap is still counted — it is just not a confounder")


def test_a_holiday_confounds_when_somebody_says_it_did(conn):
    """"A human can add an attribution note." The server will not decide that Black Friday
    moved this number; a person who was there can."""
    cid = _campaign(conn, "US launch", market="United States",
                    starts_on="2026-11-20", ends_on="2026-12-05")
    _metric(conn, cid)
    event = next(e for e in context.for_campaign(conn, cid)["events"]
                 if "Black Friday" in e["description"])

    context.attribute(conn, campaign_id=cid, event_id=event["id"],
                      note="Our whole Q4 spend moved into the Black Friday window, so the "
                           "sell-through figure is not comparable with last year.",
                      stated_by="R. Vega")

    metric = _only_metric(conn, cid)
    assert metric["confounded"] is True
    assert metric["confounded_by"][0]["attribution"]["stated_by"] == "R. Vega"


def test_an_event_with_a_stated_impact_confounds(conn):
    """Somebody recorded that it delayed the campaign by three days or cut the budget. That is
    a person saying it changed what happened, which is the whole condition."""
    cid = _campaign(conn, "US launch", market="United States",
                    starts_on="2026-11-20", ends_on="2026-12-05")
    _metric(conn, cid)
    assert _only_metric(conn, cid)["confounded"] is False

    context.record(conn, starts_on="2026-11-22", ends_on="2026-11-24", scope="market",
                   scope_value="United States", kind="fixed_calendar",
                   description="A regional storm closed stores over the weekend.",
                   delay_days=2, recorded_by="R. Vega")

    assert _only_metric(conn, cid)["confounded"] is True


def test_a_disruption_confounds_without_anybody_saying_so(conn):
    """An earthquake, a port closure, a regulatory change: the thing that was not supposed to
    happen. Waiting for somebody to confirm each one would mean the mark never appears on the
    outcomes that most need it."""
    for kind in ("natural_disaster", "conflict", "supply_chain", "platform_outage",
                 "regulatory_change", "macro_shock"):
        assert kind in context.CONFOUNDING_KINDS
    # Not these two: a recurring date is the baseline, and competitor launches are continuous
    # background that a diligent customer would turn into wallpaper.
    assert "fixed_calendar" not in context.CONFOUNDING_KINDS
    assert "competitor_launch" not in context.CONFOUNDING_KINDS


# ── it never says what caused what ──────────────────────────────────────────

def test_the_server_never_attributes(conn):
    cid = _campaign(conn)
    _quake(conn)
    _metric(conn, cid)

    metric = _only_metric(conn, cid)
    said = (metric["what_it_means"] + " "
            + metric["confounded_by"][0]["description"] + " "
            + metric["confounded_by"][0]["why"]).lower()

    # The disclaimers are removed BEFORE the sweep, because "will not work out what caused
    # what" contains the word and is the opposite of a causal claim. Checking for the token
    # alone would have to be satisfied by deleting the sentence that refuses to attribute.
    # The row carries the label and the refusal; the full argument is said once per package
    # by `core._say_the_confounded`, not repeated on every metric row.
    assert "nothing here says the event moved the number" in said
    claims = said.replace("nothing here says the event moved the number", "")
    for word in ("caused", "because of", "due to", "explains", "responsible for",
                 "the earthquake hit", "blame", "as a result"):
        assert word not in claims, word
    assert "ran through" in said or "overlapped" in said


def test_an_attribution_is_stated_and_needs_a_person(conn):
    cid = _campaign(conn)
    quake = _quake(conn)
    _metric(conn, cid)

    note = context.attribute(conn, campaign_id=cid, event_id=quake["id"],
                             note="Three days of closures inside a four-week flight.",
                             stated_by="R. Vega")

    assert note["basis"] == "stated"
    with pytest.raises(ValueError) as e:
        context.attribute(conn, campaign_id=cid, event_id=quake["id"],
                          note="It hurt us.", stated_by="")
    assert "person" in str(e.value).lower()


def test_an_attribution_against_an_event_that_never_overlapped_is_refused(conn):
    """A note explaining a number by an event that ran in another market or another year is
    the invented evidence this whole product is written against, arriving in the one field
    that accepts free text."""
    cid = _campaign(conn)
    elsewhere = _quake(conn, scope_value="Brazil",
                       description="Flooding in São Paulo.")
    _metric(conn, cid)

    with pytest.raises(ValueError) as e:
        context.attribute(conn, campaign_id=cid, event_id=elsewhere["id"],
                          note="This is why the number is low.", stated_by="R. Vega")
    assert "overlap" in str(e.value).lower()


# ── still counts ────────────────────────────────────────────────────────────

def test_a_confounded_outcome_is_still_retrieved_as_evidence(conn):
    """"Confounded outcomes STILL COUNT." Dropping them would throw away most of what a real
    library holds — every campaign that ran through a holiday, a port closure or an election."""
    cid = _campaign(conn)
    _quake(conn)
    _metric(conn, cid)

    package = core.prepare_evaluation(conn, subject_title="Mexico v2",
                                      proposal_text="A retail activation in Mexico City.")

    row = next(e for e in package["evidence"] if e["campaign_id"] == cid)
    assert row["metrics"], "the outcome is still there"
    assert row["metrics"][0]["confounded"] is True


def test_a_confounded_outcome_still_counts_as_a_measured_campaign(conn):
    cid = _campaign(conn)
    _quake(conn)
    _metric(conn, cid)

    assert cid in store.campaigns_with_actual_metrics(conn)
    assert store.get_campaign(conn, cid)["has_actual_metrics"] is True


def test_a_confounded_outcome_is_not_ranked_lower(conn):
    """Nothing here filters, excludes, down-ranks or reweights. The number is the number; what
    changes is that a reader can no longer be told it was clean."""
    clean = _campaign(conn, "Peru launch", market="Peru", detail="A retail activation.")
    dirty = _campaign(conn, "Mexico launch", detail="A retail activation.")
    _quake(conn)
    _metric(conn, clean)
    _metric(conn, dirty)

    titles = [e["title"] for e in core.find_similar(conn, text="A retail activation.")]

    assert {"Peru launch", "Mexico launch"} <= set(titles)


# ── the caveat travels ──────────────────────────────────────────────────────

def test_the_caveat_reaches_a_citation(conn):
    cid = _campaign(conn)
    _quake(conn)
    _metric(conn, cid)

    package = core.prepare_evaluation(conn, subject_title="Mexico v2",
                                      proposal_text="A retail activation in Mexico City.")

    row = next(e for e in package["evidence"] if e["campaign_id"] == cid)
    assert "confounded" in row["metrics"][0]["what_it_means"].lower()


def test_the_model_is_told_what_confounded_means(conn):
    cid = _campaign(conn)
    _quake(conn)
    _metric(conn, cid)

    note = core.prepare_evaluation(conn, subject_title="Mexico v2",
                                   proposal_text="A retail activation in Mexico City.")["note"]

    assert "confounded" in note.lower()
    assert "still count" in note.lower()
    assert "not a cause" in note.lower() or "never a cause" in note.lower()


def test_the_note_is_silent_when_nothing_is_confounded(conn):
    """A standing paragraph about confounding on a library with none is the note that fires on
    everything — §9.5's rule, three items over."""
    cid = _campaign(conn)
    _metric(conn, cid)

    note = core.prepare_evaluation(conn, subject_title="Mexico v2",
                                   proposal_text="A retail activation in Mexico City.")["note"]

    assert "confounded" not in note.lower()


def test_a_verified_performance_tag_carries_it_too(conn):
    """"This one worked" is the line a reasoner leans on hardest — §9.5 found the same pole
    was the one place its own caveat was missing."""
    cid = _campaign(conn)
    _quake(conn)
    _metric(conn, cid)
    core.update_campaign(conn, campaign_id=cid,
                         tags=[{"value": "performed_well", "source": "verified"}])

    poles = core._both_poles(conn, evidence=[], text="A retail activation in Mexico City.",
                             campaign_id=None)

    worked = [r for r in poles["worked"] if r["campaign_id"] == cid]
    assert worked and worked[0]["confounded"] is True


def test_the_saved_judgment_records_what_was_confounded(conn):
    """§9.9 lines up predicted against actual against CONTEXT, and cannot do that against a
    caveat nobody stored — the same reasoning as §9.5's `execution_at_save`."""
    cid = _campaign(conn)
    _quake(conn)
    _metric(conn, cid)

    saved = core.save_evaluation(conn, subject_title="Mexico v2", verdict="approve",
                                 summary="Rests on Mexico, which ran through an earthquake.",
                                 cited_ids=[cid], findings=[])

    stamped = store.get_evaluation(conn, saved["evaluation_id"])["evidence"]["confounded"]
    assert stamped[0]["campaign_id"] == cid
    event = stamped[0]["events"][0]
    # Ids and dates, not just prose. §9.9 reconciles against this and cannot look up a string.
    assert event["id"]
    assert event["starts_on"] == "2026-09-10"
    assert "earthquake" in event["description"].lower()


def test_it_reaches_the_model_over_the_protocol(conn):
    import asyncio
    import json

    import mcp_server

    async def call(name, args):
        return json.loads((await mcp_server.mcp.call_tool(name, args)).content[0].text)

    cid = _campaign(conn)
    _quake(conn)
    _metric(conn, cid)

    record = asyncio.run(call("get_campaign", {"campaign_id": cid}))

    assert record["metrics"][0]["confounded"] is True
    assert record["metrics"][0]["confounded_by"]


# ── the truncated-display-list bug, for the third item running ─────────────

def test_an_attribution_is_checked_against_every_overlap_not_the_shown_eight(conn):
    """`for_campaign`'s `events` is the DISPLAY list — capped at eight by salience, which sorts
    seeded rows last. So the only events that could be cut are exactly the ones that need an
    attribution to confound, and the escape hatch this item's whole design rests on closed as
    soon as a library got rich. §9.6 and §9.7 each had a round on this same bug shape."""
    cid = _campaign(conn, "US launch", market="United States",
                    starts_on="2026-11-20", ends_on="2026-12-05")
    for n in range(9):
        context.record(conn, starts_on="2026-11-21", ends_on="2026-11-30", scope="market",
                       scope_value="United States", kind="competitor_launch",
                       description=f"A rival launched, number {n}.", recorded_by="R. Vega")
    _metric(conn, cid)
    shown = context.for_campaign(conn, cid)
    assert shown["events_total"] > len(shown["events"]), "the fixture must truncate"
    black_friday = next(e for e in context.overlapping_window(
        conn, starts_on="2026-11-20", ends_on="2026-12-05", markets=["United States"])
        if "Black Friday" in e["description"])
    assert black_friday["id"] not in {e["id"] for e in shown["events"]}, "and cut that one"

    saved = context.attribute(conn, campaign_id=cid, event_id=black_friday["id"],
                              note="Our whole Q4 spend moved into that window.",
                              stated_by="R. Vega")

    assert saved["basis"] == "stated"


def test_a_campaign_with_no_window_is_told_that_and_not_something_false(conn):
    """"It did not run in its market, or not in its window" is false about a campaign that has
    no window at all, and points nowhere."""
    cid = core.ingest_campaign(conn, title="Undated", market="Mexico", status="concluded",
                               detail="A retail activation.")["campaign_id"]
    quake = _quake(conn)

    with pytest.raises(ValueError) as e:
        context.attribute(conn, campaign_id=cid, event_id=quake["id"],
                          note="This is why.", stated_by="R. Vega")
    assert "no window" in str(e.value).lower()
    assert "update_campaign" in str(e.value)


# ── recurrence, not kind ───────────────────────────────────────────────────

def test_the_product_does_not_contradict_itself_about_the_world_cup(conn):
    """§9.7 raises a `should_fix` that the window runs into the World Cup. §9.8 then told the
    reasoner to read the result as clean — on the review's own headline example. Every shipped
    calendar row is `fixed_calendar`, so keying on kind put a once-in-a-generation home
    tournament in the same bucket as Black Friday."""
    cid = _campaign(conn, "Mexico launch", market="Mexico",
                    starts_on="2026-06-15", ends_on="2026-07-15")

    _metric(conn, cid)

    metric = _only_metric(conn, cid)
    assert metric["confounded"] is True
    assert any("World Cup" in e["description"] for e in metric["confounded_by"])


def test_a_yearly_date_is_still_the_baseline(conn):
    cid = _campaign(conn, "US launch", market="United States",
                    starts_on="2026-11-20", ends_on="2026-12-05")

    _metric(conn, cid)

    assert _only_metric(conn, cid)["confounded"] is False


def test_a_competitor_launch_does_not_confound_on_its_own(conn):
    """Competitor launches are continuous background in any real market. A diligent customer
    recording them would turn every outcome confounded, which is the wallpaper this rule
    exists to avoid."""
    cid = _campaign(conn)
    context.record(conn, starts_on="2026-09-10", ends_on="2026-09-12", scope="market",
                   scope_value="Mexico", kind="competitor_launch",
                   description="A rival launched into the same shelf.", recorded_by="R. Vega")

    _metric(conn, cid)

    metric = _only_metric(conn, cid)
    assert metric["confounded"] is False
    assert metric["ran_during"]


def test_a_one_off_a_customer_records_confounds(conn):
    """A customer typing a one-off store closure as `fixed_calendar` means a thing that
    happened, not a thing that happens every year."""
    cid = _campaign(conn)
    context.record(conn, starts_on="2026-09-10", ends_on="2026-09-12", scope="market",
                   scope_value="Mexico", kind="fixed_calendar",
                   description="Our flagship was shut for a refit.", recorded_by="R. Vega")

    _metric(conn, cid)

    assert _only_metric(conn, cid)["confounded"] is True


def test_nothing_says_read_it_as_clean(conn):
    """A campaign with zero overlaps was silent while one with two overlaps was told to "read
    it as a clean result" — the more the library knew was going on, the more affirmatively it
    certified cleanliness. That is the false clean bill, inverted."""
    cid = _campaign(conn, "US launch", market="United States",
                    starts_on="2026-11-20", ends_on="2026-12-05")
    _metric(conn, cid)

    said = (_only_metric(conn, cid).get("what_it_means") or "").lower()
    assert "clean result" not in said
    assert "unless you know otherwise" not in said


# ── a target is not an outcome ─────────────────────────────────────────────

def test_only_a_measured_result_can_be_confounded(conn):
    """A target is what somebody aimed at before anything happened; a prediction is this
    library's own forecast. Neither can be confounded by an event, and calling a target an
    "outcome that ran through" something is the category error the metrics schema guards. It
    also corrupts §9.9, which scores predictions against actuals."""
    cid = _campaign(conn)
    _quake(conn)
    _metric(conn, cid, structured={"roas": 3.5}, metric_type="predicted")
    _metric(conn, cid, structured={"roas": 4.0}, metric_type="target")
    _metric(conn, cid)

    by_type = {m["metric_type"]: m for m in store.get_campaign(conn, cid)["metrics"]}
    assert by_type["actual"]["confounded"] is True
    assert by_type["predicted"]["confounded"] is False
    assert by_type["target"]["confounded"] is False
    assert not by_type["target"].get("what_it_means")


# ── four states, not one boolean ───────────────────────────────────────────

def test_a_campaign_that_could_not_be_checked_says_so(conn):
    """`confounded: false` collapsed four different provenances into one word — and its two
    neighbours on the same evidence row, `execution` and `context`, each have a
    `nothing_to_check` precisely so a reader can tell "we looked and nothing was going on"
    from "this cannot be checked at all"."""
    cid = core.ingest_campaign(conn, title="Undated", market="Mexico", status="concluded",
                               detail="A retail activation.")["campaign_id"]
    _quake(conn)
    _metric(conn, cid)

    metric = _only_metric(conn, cid)
    assert metric["status"] == "nothing_to_check"
    assert metric["confounded"] is False
    assert "no window" in metric["what_it_means"].lower()


def test_a_checked_campaign_with_nothing_around_it_says_that_instead(conn):
    cid = _campaign(conn)

    _metric(conn, cid)

    assert _only_metric(conn, cid)["status"] == "checked_clean"


def test_a_confounded_campaign_says_confounded(conn):
    cid = _campaign(conn)
    _quake(conn)

    _metric(conn, cid)

    assert _only_metric(conn, cid)["status"] == "confounded"


def test_a_window_read_from_the_brief_still_finds_a_confounder(conn):
    """§9.6's rule: a positive MATCH from a heuristic window still stands — finding something
    is not weakened by the window being a reading; only the negative claim is. §9.8 discarded
    the match and kept the claim."""
    cid = core.ingest_campaign(
        conn, title="Mexico launch", market="Mexico", status="concluded",
        detail="The activation ran 1 September 2026 to 30 September 2026.")["campaign_id"]
    _quake(conn)

    _metric(conn, cid)

    metric = _only_metric(conn, cid)
    assert metric["confounded"] is True
    assert metric["period"]["basis"] == "campaign_window"


def test_a_global_event_reaches_a_record_with_no_market(conn):
    """A global macro shock is the paradigm confounder, and it was dropped for any record with
    no market — while `campaign_context` found it on the same record. Two surfaces, two
    answers about one campaign."""
    cid = core.ingest_campaign(conn, title="Unlabelled", status="concluded",
                               detail="A retail activation.", starts_on="2026-09-01",
                               ends_on="2026-09-30")["campaign_id"]
    context.record(conn, starts_on="2026-09-10", ends_on="2026-09-12", scope="global",
                   kind="macro_shock", description="A global credit shock.",
                   recorded_by="R. Vega")

    _metric(conn, cid)

    assert _only_metric(conn, cid)["confounded"] is True


# ── the attribution has to reach the reader ────────────────────────────────

def test_the_persons_note_reaches_a_citation(conn):
    """The standing note tells the model "if a person has attributed it, their note is on the
    row and is theirs, not a measurement". It was not on the row: the projection flattened
    every confounder to its description, so the single most decision-relevant sentence on the
    record reached nothing — and the server was making a false statement about its own output
    in the paragraph telling the model where to look."""
    cid = _campaign(conn)
    quake = _quake(conn)
    _metric(conn, cid)
    context.attribute(conn, campaign_id=cid, event_id=quake["id"],
                      note="Three days of closures inside a four-week flight.",
                      stated_by="R. Vega")

    package = core.prepare_evaluation(conn, subject_title="Mexico v2",
                                      proposal_text="A retail activation in Mexico City.")

    row = next(e for e in package["evidence"] if e["campaign_id"] == cid)
    carried = row["metrics"][0]["confounded_by"][0]
    assert carried["id"] == quake["id"], "and the event is linked, not just described"
    assert carried["attribution"]["note"].startswith("Three days")
    assert carried["attribution"]["stated_by"] == "R. Vega"
    assert carried["why"]


def test_the_sentence_is_not_cut_at_the_first_full_stop(conn):
    """`description.split(".")[0]` turned "Magnitude 7.1 earthquake; Mexico City retail shut 3
    days." into "Magnitude 7", and "U.S. tariffs of 25%…" into "U"."""
    cid = _campaign(conn)
    _quake(conn)
    _metric(conn, cid)

    said = _only_metric(conn, cid)["what_it_means"]
    assert "Magnitude 7." not in said.replace("Magnitude 7.1", "")
    assert "7.1 earthquake" in said


# ── an attribution is a judgment, and judgments are kept ───────────────────

def test_a_second_attribution_does_not_erase_the_first(conn):
    """§9.4 settled this four items earlier and wrote it into the schema: a reading made before
    the numbers came in and one made after are two judgments about the same thing, and the pair
    is worth more than either. "This looked like the earthquake before the numbers and like our
    own pricing after" is the highest-value record in the item — and §9.9 is the calibration
    item that would read it."""
    cid = _campaign(conn)
    quake = _quake(conn)
    _metric(conn, cid)
    context.attribute(conn, campaign_id=cid, event_id=quake["id"],
                      note="Three days of closures inside a four-week flight.",
                      stated_by="R. Vega")

    context.attribute(conn, campaign_id=cid, event_id=quake["id"],
                      note="Actually the dip started before the quake.", stated_by="J. Lin")

    history = context.attribution_history(conn, campaign_id=cid, event_id=quake["id"])
    assert [h["stated_by"] for h in history] == ["R. Vega", "J. Lin"]
    assert _only_metric(conn, cid)["confounded_by"][0]["attribution"]["stated_by"] == "J. Lin"


def test_an_attribution_records_whether_the_numbers_were_in(conn):
    """§9.4's `outcome_known`, for the same reason: the point is to record what the person
    could see when they said it."""
    cid = _campaign(conn)
    quake = _quake(conn)

    before = context.attribute(conn, campaign_id=cid, event_id=quake["id"],
                               note="It will have cost us three days.", stated_by="R. Vega")
    _metric(conn, cid)
    after = context.attribute(conn, campaign_id=cid, event_id=quake["id"],
                              note="Sell-through came in at 41%.", stated_by="R. Vega")

    assert before["outcome_known"] is False
    assert after["outcome_known"] is True


# ── a metric has its own period ────────────────────────────────────────────

def test_a_monthly_row_is_confounded_by_its_own_month(conn):
    """The campaign's whole window was used, so January's sell-through was confounded by a
    September earthquake — on a row that literally carries `month: 2026-01`, the same column
    the importer already parses. §9.6's own write-up promised §9.8 "a measurement period
    narrower than an always-on campaign's year"; nothing delivered it, and it failed hardest
    for the customers with the most data."""
    cid = _campaign(conn, starts_on="2026-01-01", ends_on="2026-12-31")
    _quake(conn)
    for month in range(1, 13):
        _metric(conn, cid, structured={"month": f"2026-{month:02d}",
                                       "sell_through_pct": 40.0 + month})

    rows = [m for m in store.get_campaign(conn, cid)["metrics"]
            if m["metric_type"] == "actual"]
    by_month = {m["structured"]: m for m in rows}
    september = next(m for k, m in by_month.items() if "2026-09" in k)
    january = next(m for k, m in by_month.items() if "2026-01" in k)
    assert september["confounded"] is True
    assert january["confounded"] is False
    assert january["period"]["basis"] == "metric"


def test_a_row_with_no_period_falls_back_to_the_campaign_window_and_says_so(conn):
    cid = _campaign(conn)
    _quake(conn)

    _metric(conn, cid)

    metric = _only_metric(conn, cid)
    assert metric["confounded"] is True
    assert metric["period"]["basis"] == "campaign_window"


def test_the_caveat_is_not_duplicated_onto_every_row(conn):
    """A 120-row workbook campaign produced a 742 KB `get_campaign` response — the same event
    list and the same 858-character sentence repeated onto every row. §6.8 exists in this
    codebase because a match with many metrics rows was heavy at top_k=5; this re-created it
    by another route."""
    import json

    cid = _campaign(conn, starts_on="2026-01-01", ends_on="2026-12-31")
    for n in range(6):
        context.record(conn, starts_on=f"2026-0{n + 1}-05", ends_on=f"2026-0{n + 1}-09",
                       scope="market", scope_value="Mexico", kind="supply_chain",
                       description=f"Port disruption {n}.", recorded_by="R. Vega")
    for month in range(1, 13):
        _metric(conn, cid, structured={"month": f"2026-{month:02d}", "roas": 3.0})

    size = len(json.dumps(store.get_campaign(conn, cid)["metrics"], default=str))

    assert size < 30_000, f"{size} bytes of metrics for twelve rows"


# ── the surface §9.9 is named after ────────────────────────────────────────

def test_reconciliation_carries_the_caveat(conn):
    """"Line up predicted against delivered against actual against CONTEXT." Every other
    number-bearing surface carried the caveat; the one the review names did not."""
    cid = _campaign(conn)
    _quake(conn)
    _metric(conn, cid, structured={"roas": 3.0}, metric_type="predicted")
    _metric(conn, cid)
    saved = core.save_evaluation(conn, subject_title="Mexico launch", verdict="approve",
                                 summary="It should work.", campaign_id=cid, findings=[],
                                 predictions={"predicted_roi_range": "3-4"})

    out = core.reconcile_evaluation(conn, evaluation_id=saved["evaluation_id"])

    assert out["context"]["confounded"] is True
    assert "earthquake" in str(out["context"]["confounded_by"]).lower()


# ── "never quoted as clean evidence" ───────────────────────────────────────

def test_citing_a_confounded_outcome_without_saying_so_is_a_finding(conn):
    """Every §9.8 signal was an input and none was an output the model had to carry. §9.7
    settled the same question: the finding is the SILENCE. A verdict that leans on a
    confounded number and never mentions it is the case this item exists for."""
    cid = _campaign(conn)
    _quake(conn)
    _metric(conn, cid)

    saved = core.save_evaluation(
        conn, subject_title="Mexico v2", verdict="approve",
        summary="Mexico delivered 41% sell-through, so this works.",
        cited_ids=[cid], findings=[])

    stored = store.get_evaluation(conn, saved["evaluation_id"])
    raised = [f for f in stored["findings"] if f.get("category") == "confounded_evidence"]
    assert raised, stored["findings"]
    assert raised[0]["severity"] == "should_fix"
    assert saved["verdict"] == "approve", "the server argues with a verdict, never replaces it"


def test_a_verdict_that_says_so_raises_nothing(conn):
    cid = _campaign(conn)
    _quake(conn)
    _metric(conn, cid)

    saved = core.save_evaluation(
        conn, subject_title="Mexico v2", verdict="approve",
        summary="Mexico delivered 41% sell-through, though that result is confounded by the "
                "earthquake and is not clean evidence.",
        cited_ids=[cid], findings=[])

    stored = store.get_evaluation(conn, saved["evaluation_id"])
    assert not [f for f in stored["findings"] if f.get("category") == "confounded_evidence"]


def test_a_verdict_citing_nothing_confounded_raises_nothing(conn):
    cid = _campaign(conn)
    _metric(conn, cid)

    saved = core.save_evaluation(conn, subject_title="Mexico v2", verdict="approve",
                                 summary="Mexico delivered 41% sell-through.",
                                 cited_ids=[cid], findings=[])

    stored = store.get_evaluation(conn, saved["evaluation_id"])
    assert not [f for f in stored["findings"] if f.get("category") == "confounded_evidence"]


def test_reconciliation_carries_it_when_the_numbers_are_passed_in(conn):
    """`actual_metrics` was bound on only one branch, so passing the figures by hand raised a
    NameError on the surface §9.9 is named after."""
    cid = _campaign(conn)
    _quake(conn)
    _metric(conn, cid)
    saved = core.save_evaluation(conn, subject_title="Mexico launch", verdict="approve",
                                 summary="It should work.", campaign_id=cid, findings=[])

    out = core.reconcile_evaluation(conn, evaluation_id=saved["evaluation_id"],
                                    actual="Sell-through 41%.")

    assert out["actual"] == "Sell-through 41%."
    assert out["context"]["confounded"] is True


# ── the remaining edges both reviews walked ────────────────────────────────

def test_a_stated_impact_of_zero_is_not_an_impact(conn):
    """`_has_stated_impact` used `is not None` while `_impact_sentence`, in the same module,
    renders a zero as "no delay" — somebody looked and found none. The two contradicted each
    other, and the one that decided confounding read a person's "it changed nothing" as
    "somebody said it changed things"."""
    cid = _campaign(conn, "US launch", market="United States",
                    starts_on="2026-11-20", ends_on="2026-12-05")
    context.record(conn, starts_on="2026-11-25", ends_on="2026-11-26", scope="market",
                   scope_value="United States", kind="fixed_calendar",
                   description="A holiday that changed nothing for us.", delay_days=0,
                   budget_change_pct=0.0, recorded_by="R. Vega", recurs_annually=True)

    _metric(conn, cid)

    assert _only_metric(conn, cid)["confounded"] is False


def test_an_attribution_can_be_withdrawn(conn):
    """Its siblings all have a correction path. An attribution today could only be replaced by
    writing another claim — there was no way to say "I was wrong to say that"."""
    cid = _campaign(conn)
    quake = _quake(conn)
    _metric(conn, cid)
    context.attribute(conn, campaign_id=cid, event_id=quake["id"],
                      note="Three days of closures.", stated_by="R. Vega")

    context.withdraw_attribution(conn, campaign_id=cid, event_id=quake["id"],
                                 why="I was thinking of a different campaign.",
                                 withdrawn_by="R. Vega")

    history = context.attribution_history(conn, campaign_id=cid, event_id=quake["id"])
    assert history[-1]["withdrawn_at"] is not None, "kept, not deleted"
    assert _only_metric(conn, cid)["confounded_by"][0]["attribution"] is None


def test_the_server_cannot_sign_an_attribution_as_itself(conn):
    """`stated_by="campaign-poc server (computed)"` was accepted and rendered as though the
    library had worked it out — in the one item whose premise is that a model asked to explain
    a disappointing number will reach for whatever is nearby."""
    cid = _campaign(conn)
    quake = _quake(conn)
    _metric(conn, cid)

    for name in ("campaign-poc server", "the system", "computed", "server (computed)"):
        with pytest.raises(ValueError) as e:
            context.attribute(conn, campaign_id=cid, event_id=quake["id"],
                              note="It cost us three days.", stated_by=name)
        assert "person" in str(e.value).lower()


def test_reference_material_cannot_be_confounded(conn):
    """`campaigns_overlapping` excludes reference records — the rulebook is not precedent. A
    brand-guidelines PDF accepting an attribution made the two directions disagree, which is
    the shape §9.6 spent a round fixing."""
    ref = core.ingest_campaign(conn, title="Brand guidelines", record_type="reference",
                               market="Mexico", detail="Colour and type.",
                               starts_on="2026-09-01", ends_on="2026-09-30")["campaign_id"]
    quake = _quake(conn)

    with pytest.raises(ValueError) as e:
        context.attribute(conn, campaign_id=ref, event_id=quake["id"],
                          note="This is why.", stated_by="R. Vega")
    assert "reference" in str(e.value).lower()


def test_the_caveat_is_computed_once_per_campaign_not_once_per_row(conn):
    """A twelve-row workbook ran the overlap query twelve times. `context.record` already
    fans out across the whole library, and §9.8 added two queries per campaign to it."""
    cid = _campaign(conn, starts_on="2026-01-01", ends_on="2026-12-31")
    _quake(conn)
    for month in range(1, 13):
        _metric(conn, cid, structured={"month": f"2026-{month:02d}", "roas": 3.0})

    calls = []
    real = store.context_events
    store.context_events = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        store.get_campaign(conn, cid)
    finally:
        store.context_events = real

    assert len(calls) <= 2, f"{len(calls)} overlap queries for one record"


def test_a_missing_table_refuses_rather_than_answering(conn):
    """`attributions_for` returned `{}` from a missing table while `context_events` raises for
    precisely this collapse — an empty answer that means "could not look"."""
    cid = _campaign(conn)
    _metric(conn, cid)
    conn.execute("DROP TABLE context_attributions")
    conn.commit()

    with pytest.raises(RuntimeError):
        store.attributions(conn, cid)


# ── the mutations that survived the adversarial pass ───────────────────────

def test_the_caveat_says_the_outcome_still_counts(conn):
    """"Confounded outcomes STILL COUNT" is half the review's sentence, and removing it from
    the caveat passed the suite — leaving a warning a reader would reasonably act on by
    discarding the number."""
    cid = _campaign(conn)
    _quake(conn)
    _metric(conn, cid)

    said = _only_metric(conn, cid)["what_it_means"].lower()
    assert "still counts" in said
    assert "ignore" not in said
    assert "discount" not in said
    assert "disregard" not in said


def test_the_note_says_it_is_never_a_cause_in_its_own_words(conn):
    """The old test was satisfied by §9.6's context paragraph, so removing this sentence
    entirely passed — the one line telling the model not to explain a number by whatever was
    nearby was tested by another item's text."""
    cid = _campaign(conn)
    _quake(conn)
    _metric(conn, cid)
    package = core.prepare_evaluation(conn, subject_title="Mexico v2",
                                      proposal_text="A retail activation in Mexico City.")

    said = core._say_the_confounded(package["evidence"])
    assert said, "the function under test, not another item's prose"
    assert "never a cause" in said.lower()
    assert "still count" in said.lower()
    assert "earthquake" in said.lower(), "the review's own example, kept"


def test_an_event_ending_on_the_first_day_of_the_period_confounds(conn):
    """No boundary-day test existed, so making the overlap exclusive passed."""
    cid = _campaign(conn, starts_on="2026-09-10", ends_on="2026-09-30")
    _quake(conn, starts_on="2026-09-01", ends_on="2026-09-10")

    _metric(conn, cid)

    assert _only_metric(conn, cid)["confounded"] is True


def test_an_event_ending_the_day_before_does_not(conn):
    cid = _campaign(conn, starts_on="2026-09-10", ends_on="2026-09-30")
    _quake(conn, starts_on="2026-09-01", ends_on="2026-09-09")

    _metric(conn, cid)

    assert _only_metric(conn, cid)["confounded"] is False


def test_every_measured_row_is_marked_not_just_the_first(conn):
    """Marking only `metrics[0]` passed, because no test had more than one row."""
    cid = _campaign(conn)
    _quake(conn)
    _metric(conn, cid, structured={"roas": 3.0})
    _metric(conn, cid, structured={"sell_through_pct": 41.0})

    rows = [m for m in store.get_campaign(conn, cid)["metrics"]
            if m["metric_type"] == "actual"]
    assert len(rows) == 2
    assert all(m["confounded"] for m in rows)


def test_a_confounded_outcome_keeps_its_place_in_the_ranking(conn):
    """The old test checked set membership, so sorting confounded rows last passed. "Still
    count" means the ORDER is untouched, not merely that the row survives."""
    clean = _campaign(conn, "Mexico clean", detail="An identical retail activation.")
    dirty = _campaign(conn, "Mexico dirty", detail="An identical retail activation.")
    before = [e["title"] for e in core.find_similar(
        conn, text="An identical retail activation.")]

    _quake(conn)
    _metric(conn, clean)
    _metric(conn, dirty)

    after = [e["title"] for e in core.find_similar(
        conn, text="An identical retail activation.")]
    assert after == before, "confounding must not move anything"


def test_a_confounded_outcome_still_counts_toward_evidence_strength(conn):
    """§6.6's strength ladder is built on `with_results`. Excluding confounded rows there
    passed the suite and would have made a confounded precedent worth less — a down-weight by
    the back door, which is the one thing "still count" forbids."""
    cid = _campaign(conn)
    _quake(conn)
    _metric(conn, cid)

    strength = core._evidence_strength(conn, cited_ids=[cid], text="A retail activation.")

    assert strength["with_results"] == 1


def test_the_ran_during_line_survives(conn):
    """Removing the sentence for a campaign that overlapped only recurring dates passed — and
    that sentence is what stops an empty `confounded_by` reading as "nothing was going on"."""
    cid = _campaign(conn, "US launch", market="United States",
                    starts_on="2026-11-20", ends_on="2026-12-05")

    _metric(conn, cid)

    said = _only_metric(conn, cid).get("what_it_means") or ""
    assert "come round every year" in said
    assert "on the record either way" in said.lower()


def test_an_attribution_read_back_is_still_stated(conn):
    """`attributions_for` returning `basis: computed` passed the suite — the house rule is
    never to mislabel, and this is a person's account of a cause."""
    cid = _campaign(conn)
    quake = _quake(conn)
    _metric(conn, cid)
    context.attribute(conn, campaign_id=cid, event_id=quake["id"],
                      note="Three days of closures.", stated_by="R. Vega")

    assert store.attributions_for(conn, cid)[quake["id"]]["basis"] == "stated"
    assert _only_metric(conn, cid)["confounded_by"][0]["attribution"]["basis"] == "stated"


def test_a_busy_market_does_not_put_a_paragraph_per_event_on_every_row(conn):
    """An always-on record in a busy market genuinely overlaps dozens of events. The count is
    the finding; carrying each one in full on every metric row is how a twelve-month workbook
    became a 742 KB response."""
    import json

    cid = _campaign(conn, starts_on="2026-01-01", ends_on="2026-12-31")
    for n in range(25):
        context.record(conn, starts_on="2026-03-01", ends_on="2026-03-28", scope="market",
                       scope_value="Mexico", kind="supply_chain",
                       description=f"Port disruption number {n}, with a long description.",
                       recorded_by="R. Vega")
    for month in range(1, 13):
        _metric(conn, cid, structured={"month": f"2026-{month:02d}", "roas": 3.0})

    rows = [m for m in store.get_campaign(conn, cid)["metrics"]
            if m["metric_type"] == "actual"]
    march = next(m for m in rows if "2026-03" in m["structured"])

    assert march["confounded_by_total"] == 25, "the count stays complete"
    assert len(march["confounded_by"]) == context.MAX_CONFOUNDERS_SHOWN
    assert march["confounded_by_truncated"] is True
    assert "and 20 more" in march["what_it_means"]
    assert len(json.dumps(rows, default=str)) < 20_000


def test_an_event_starting_on_the_last_day_of_the_period_confounds(conn):
    """The in-memory narrowing has its own boundary, separate from the SQL one — and every
    boundary test hit the SQL path, so making this half exclusive passed."""
    cid = _campaign(conn, starts_on="2026-01-01", ends_on="2026-12-31")
    _quake(conn, starts_on="2026-03-31", ends_on="2026-04-05")
    _metric(conn, cid, structured={"month": "2026-03", "roas": 3.0})

    assert _only_metric(conn, cid)["confounded"] is True


def test_the_record_carries_no_private_working(conn):
    """The overlap memo lives on the record dict for the duration of one read. Left there it
    reaches every caller and every serialised response as a key nobody documented — and a
    cache that outlives the call is the stale link §9.6 refused a join table to avoid."""
    cid = _campaign(conn)
    _quake(conn)
    _metric(conn, cid)

    record = store.get_campaign(conn, cid)

    assert not [k for k in record if k.startswith("_")], sorted(record)
