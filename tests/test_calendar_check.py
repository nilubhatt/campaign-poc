"""
§9.7: the same table works forwards — seed it, and check every proposed window against it.

The review: *"This is not only a post-mortem device. The UAE brief names a Ramadan content
series and never places it on a calendar — a fixed, immovable date the plan could have been
checked against at review time. Fix: seed the table with known fixed events per market —
Ramadan, national holidays, Golden Week, Black Friday, monsoon and rainy seasons, major
sporting fixtures — and have `prepare_evaluation` check every proposed window against it as a
computed fact. **"Launch window overlaps Ramadan (18 Feb – 19 Mar)" is a finding no model needs
to be clever to produce.** Mexico's own deck flagged that it clashed with the World Cup and
then never addressed it; that flag should have come from the system and been impossible to
drop."*

**"Impossible to drop" is what makes this a COMPUTED fact rather than a note.** §2.4 made
`computed` unwritable from the MCP surface and §7.1 made the model treat computed facts as
established rather than re-deriving them. A calendar clash that arrives as prose is a sentence
the model can decline to repeat; one that arrives in `computed` is a fact it has to either
carry or dispute out loud.

**A shipped calendar is a claim about the world, and the honest ones are hedged.** Ramadan
begins on a sighting and differs by country; a monsoon has no start date at all. Shipping those
as exact facts would be the confident unfounded claim this product is written against, so a
seeded event says it is a shipped approximation and names what would settle it. What is NOT
hedged is the overlap arithmetic: given a window and a date range, whether they overlap is
computed, and that is the half the review calls "no model needs to be clever to produce".

**Nothing here says a clash is bad.** Launching into Black Friday is the entire point of some
campaigns and the ruin of others, and the library has no way to tell which. It reports the
overlap and the plan's silence about it; §9.4's lesson — that treating every deviation as a
defect teaches the library to punish improvement — applies unchanged.
"""
import json

import pytest

import context
import core
import store


def _proposal(conn, **kw):
    kw.setdefault("subject_title", "Mexico v2")
    kw.setdefault("proposal_text", "A retail activation across Mexico City.")
    return core.prepare_evaluation(conn, **kw)


def _clash(package):
    return package["computed"]["calendar_clash"]


# ── the calendar the product ships ──────────────────────────────────────────

def test_a_fresh_library_already_knows_about_ramadan(conn):
    """"Seed the table with known fixed events per market." A calendar every customer has to
    type in is a calendar no customer has."""
    seeded = store.context_events(conn, seeded=True)

    assert seeded, "nothing was seeded"
    assert any("ramadan" in e["description"].lower() for e in seeded)


def test_the_review_s_own_list_is_covered(conn):
    """"Ramadan, national holidays, Golden Week, Black Friday, monsoon and rainy seasons,
    major sporting fixtures.\""""
    described = " ".join(e["description"].lower()
                         for e in store.context_events(conn, seeded=True))

    for expected in ("ramadan", "golden week", "black friday", "monsoon", "world cup"):
        assert expected in described, expected


def test_every_seeded_event_is_marked_seeded_and_keyed(conn):
    """`seeded` is how a customer tells what they recorded from what the product assumed, and
    `seed_key` is how the seeder finds its own row again — without it a calendar doubles on
    every start and a corrected date can never replace a wrong one."""
    for event in store.context_events(conn, seeded=True):
        assert event["seeded"] is True
        assert event["seed_key"], event["description"]


def test_seeding_twice_does_not_double_the_calendar(conn):
    before = len(store.context_events(conn, seeded=True))

    context.seed(conn)
    context.seed(conn)

    assert len(store.context_events(conn, seeded=True)) == before


def test_a_corrected_seed_date_replaces_the_wrong_one(conn):
    """A shipped date that turns out to be wrong has to be able to reach a database that
    already has it. `_seed_metric_registry` gets this for free because `canonical` is its
    primary key; this needs `seed_key` to do the same job."""
    key = "test.correctable.2026"
    context.record(conn, starts_on="2026-02-18", ends_on="2026-03-19", scope="market",
                   scope_value="UAE", kind="fixed_calendar", description="Ramadan (est).",
                   recorded_by="seed", seeded=True, seed_key=key)

    context.record(conn, starts_on="2026-02-17", ends_on="2026-03-19", scope="market",
                   scope_value="UAE", kind="fixed_calendar",
                   description="Ramadan (corrected).", recorded_by="seed", seeded=True,
                   seed_key=key)

    rows = [e for e in store.context_events(conn) if e["seed_key"] == key]
    assert len(rows) == 1
    assert rows[0]["starts_on"] == "2026-02-17"


def test_a_seeded_date_says_it_is_an_approximation(conn):
    """Ramadan begins on a sighting and differs by country; a monsoon has no start date at
    all. Shipping those as exact facts is the confident unfounded claim this product exists to
    stop — and a customer correcting one has to know it was ours to be wrong about."""
    ramadan = next(e for e in store.context_events(conn, seeded=True)
                   if "ramadan" in e["description"].lower())

    said = context._public(ramadan)["what_it_means"].lower()
    assert "approximate" in said or "varies" in said
    assert "shipped with this product" in said


def test_a_seeded_event_can_be_withdrawn_like_any_other(conn):
    """A customer whose market does not observe a shipped holiday must be able to say so."""
    seeded = store.context_events(conn, seeded=True)[0]

    context.withdraw(conn, event_id=seeded["id"], why="We do not trade in that market.",
                     withdrawn_by="R. Vega")

    assert seeded["id"] not in {e["id"] for e in store.context_events(conn)}


# ── the check at review time ────────────────────────────────────────────────

def test_a_proposed_window_overlapping_ramadan_is_a_computed_fact(conn):
    """The review's own sentence: "Launch window overlaps Ramadan (18 Feb – 19 Mar)" is a
    finding no model needs to be clever to produce."""
    package = _proposal(conn, subject_title="UAE launch",
                        proposal_text="A Ramadan content series across the UAE.",
                        market="UAE", starts_on="2026-03-01", ends_on="2026-03-31")

    clash = _clash(package)
    assert clash["basis"] == "computed"
    assert clash["status"] == "present"
    assert "Ramadan" in clash["what_it_means"]
    assert "2026-03" in clash["what_it_means"]


def test_the_mexico_world_cup_clash_comes_from_the_system(conn):
    """"Mexico's own deck flagged that it clashed with the World Cup and then never addressed
    it; that flag should have come from the system and been impossible to drop.\""""
    package = _proposal(conn, subject_title="Mexico launch",
                        proposal_text="A retail activation across Mexico City.",
                        market="Mexico", starts_on="2026-06-15", ends_on="2026-07-10")

    clash = _clash(package)
    assert clash["status"] == "present"
    assert "World Cup" in clash["what_it_means"]


def test_a_window_with_nothing_in_it_says_so(conn):
    package = _proposal(conn, market="Mexico", starts_on="2026-09-02", ends_on="2026-09-10")

    clash = _clash(package)
    assert clash["status"] == "absent"
    assert "nothing on record" in clash["what_it_means"].lower()


def test_a_proposal_with_no_window_is_not_given_a_clean_bill(conn):
    """The whole §9.6 rule, arriving one item later. "This launch clashes with nothing" is a
    claim, and a proposal that never says when it runs has not earned it."""
    package = _proposal(conn, market="Mexico")

    clash = _clash(package)
    assert clash["status"] == "nothing_to_check"
    assert "no window" in clash["what_it_means"].lower()


def test_a_proposal_with_no_market_is_not_given_a_clean_bill(conn):
    package = _proposal(conn, starts_on="2026-03-01", ends_on="2026-03-31")

    assert _clash(package)["status"] == "nothing_to_check"


def test_a_stored_subject_uses_its_own_window_and_market(conn):
    """With a `campaign_id`, the record answers "which brief is this" — §7.2's split, and
    passing the filters alongside one is already refused."""
    cid = core.ingest_campaign(conn, title="UAE launch", market="UAE", status="proposed",
                               detail="A Ramadan content series.", starts_on="2026-03-01",
                               ends_on="2026-03-31")["campaign_id"]

    package = _proposal(conn, subject_title="UAE launch", proposal_text="A Ramadan series.",
                        campaign_id=cid)

    assert _clash(package)["status"] == "present"


def test_a_window_read_out_of_the_proposal_is_used_and_labelled(conn):
    """A pitch usually names its own flight. Reading it is worth doing — and saying it was
    read rather than entered is what stops the reading being mistaken for the dates."""
    package = _proposal(conn, subject_title="UAE launch",
                        proposal_text="Flight runs 1 March 2026 to 20 March 2026 in the UAE.",
                        market="UAE")

    clash = _clash(package)
    assert clash["status"] == "present"
    assert clash["window"]["basis"] == "heuristic"
    assert "read from" in clash["what_it_means"].lower()


# ── what it refuses to say ──────────────────────────────────────────────────

def test_it_never_says_a_clash_is_a_problem(conn):
    """Launching into Black Friday is the point of some campaigns and the ruin of others, and
    the library cannot tell which. §9.4's lesson applies unchanged: treating every deviation
    as a defect teaches it to punish anything that went better than planned."""
    package = _proposal(conn, subject_title="Mexico launch",
                        proposal_text="A retail push across Mexico City.",
                        market="Mexico", starts_on="2026-06-15", ends_on="2026-07-10")

    said = _clash(package)["what_it_means"].lower()
    for word in ("avoid", "risk", "problem", "conflict", "should not", "bad time",
                 "reschedule", "caused", "will hurt"):
        assert word not in said, word


def test_the_model_is_told_to_ask_rather_than_to_judge(conn):
    package = _proposal(conn, subject_title="Mexico launch",
                        proposal_text="A retail push across Mexico City.",
                        market="Mexico", starts_on="2026-06-15", ends_on="2026-07-10")

    assert "whether that is deliberate" in package["note"].lower()


def test_the_plan_s_own_silence_is_the_finding(conn):
    """Mexico's deck DID flag the World Cup. What it never did was address it — so the useful
    distinction is not "does it clash" but "does the plan mention the thing it clashes with"."""
    silent = _proposal(conn, subject_title="Mexico launch",
                       proposal_text="A retail push across Mexico City.",
                       market="Mexico", starts_on="2026-06-15", ends_on="2026-07-10")
    speaks = _proposal(conn, subject_title="Mexico launch",
                       proposal_text="A retail push across Mexico City, timed to the World "
                                     "Cup and built around the fixtures.",
                       market="Mexico", starts_on="2026-06-15", ends_on="2026-07-10")

    assert _clash(silent)["unaddressed"] == ["World Cup"]
    assert _clash(speaks)["unaddressed"] == []
    assert "does not name" in _clash(silent)["what_it_means"].lower()


def test_a_customer_event_clashes_the_same_way_a_seeded_one_does(conn):
    """The seeded calendar is a starting point, not the mechanism. An event this customer
    recorded has to reach a proposal exactly the same way."""
    context.record(conn, starts_on="2026-09-01", ends_on="2026-09-30", scope="market",
                   scope_value="Mexico", kind="regulatory_change",
                   description="New labelling rules take effect.", recorded_by="R. Vega")

    package = _proposal(conn, market="Mexico", starts_on="2026-09-02", ends_on="2026-09-10")

    assert _clash(package)["status"] == "present"
    assert "labelling" in _clash(package)["what_it_means"]


def test_it_reaches_the_model_over_the_protocol(conn):
    import asyncio
    import json

    import mcp_server

    async def call(name, args):
        return json.loads((await mcp_server.mcp.call_tool(name, args)).content[0].text)

    package = asyncio.run(call("prepare_evaluation", {
        "subject_title": "Mexico launch",
        "proposal_text": "A retail activation across Mexico City.",
        "market": "Mexico", "starts_on": "2026-06-15", "ends_on": "2026-07-10"}))

    clash = package["computed"]["calendar_clash"]
    assert clash["basis"] == "computed"
    assert "World Cup" in clash["what_it_means"]


# ── what the shipped calendar does NOT cover ────────────────────────────────

def test_a_market_the_shipped_calendar_never_covered_is_not_a_clean_bill(conn):
    """Thirteen events is not the wrong size — it is the wrong CLAIM. A customer in Brazil,
    Nigeria and Poland gets "nothing on record was happening" on every window, promoted into
    the block headed "what the server already established", with a hedge that blames them:
    "events nobody has recorded". The gap is the product's."""
    package = _proposal(conn, subject_title="Rio launch",
                        proposal_text="A retail push across Brazil.",
                        market="Brazil", starts_on="2026-11-20", ends_on="2026-12-05")

    clash = _clash(package)
    assert clash["status"] == "not_covered"
    assert "Brazil" in clash["what_it_means"]
    assert "shipped" in clash["what_it_means"].lower()
    assert clash["markets_not_covered"] == ["Brazil"]


def test_a_covered_market_with_a_quiet_window_still_says_absent(conn):
    package = _proposal(conn, market="Mexico", starts_on="2026-09-02", ends_on="2026-09-10")

    assert _clash(package)["status"] == "absent"


def test_the_fact_says_which_markets_it_searched(conn):
    package = _proposal(conn, market="Mexico", starts_on="2026-09-02", ends_on="2026-09-10")

    assert "Mexico" in _clash(package)["searched"]
    assert "global" in _clash(package)["searched"]


def test_a_customer_event_makes_an_uncovered_market_checkable(conn):
    """The shipped calendar is a starting point. Once this customer has recorded something in
    Brazil, a Brazilian window IS being checked against a real calendar."""
    context.record(conn, starts_on="2026-11-25", ends_on="2026-11-30", scope="market",
                   scope_value="Brazil", kind="fixed_calendar",
                   description="Black Friday Brasil.", recorded_by="R. Vega")

    package = _proposal(conn, market="Brazil", starts_on="2026-11-20", ends_on="2026-12-05")

    assert _clash(package)["status"] == "present"


def test_the_united_states_is_reachable_by_the_names_people_type(conn):
    """Matching is exact-string against one canonical spelling, and `market` is freeform
    everywhere in this library. Three of four spellings of one country got a clean bill."""
    for spelling in ("United States", "USA", "US"):
        package = _proposal(conn, subject_title="US launch",
                            proposal_text="A retail push.", market=spelling,
                            starts_on="2026-11-25", ends_on="2026-11-30")
        assert _clash(package)["status"] == "present", spelling


def test_a_south_east_asian_market_reaches_the_rainy_season(conn):
    """One `region`-scoped row that only somebody typing `SEA` could reach is a dead row —
    and `APAC`, the region value this tool's own description uses as its example, matched
    nothing at all."""
    package = _proposal(conn, subject_title="Jakarta launch",
                        proposal_text="An outdoor activation.", market="Indonesia",
                        starts_on="2026-12-01", ends_on="2026-12-20")

    assert _clash(package)["status"] == "present"
    assert "rainy season" in _clash(package)["what_it_means"].lower()


# ── Ramadan is not a global event ───────────────────────────────────────────

def test_a_polish_campaign_is_not_told_it_failed_to_mention_ramadan(conn):
    """The row's own description says the effect is "in Muslim-majority markets" while the row
    was scoped to every market on earth — so a Polish spring push was told, as an ESTABLISHED
    computed fact, that its plan fails to mention Ramadan. A nuisance list is how a check stops
    being read."""
    context.record(conn, starts_on="2026-03-05", ends_on="2026-03-08", scope="market",
                   scope_value="Poland", kind="competitor_launch",
                   description="A rival launched.", recorded_by="R. Vega")

    package = _proposal(conn, subject_title="Warsaw launch",
                        proposal_text="A retail push across Poland.",
                        market="Poland", starts_on="2026-03-02", ends_on="2026-03-20")

    said = _clash(package)["what_it_means"]
    assert "Ramadan" not in said
    assert _clash(package)["unaddressed"] == []


def test_ramadan_still_reaches_the_markets_it_is_about(conn):
    package = _proposal(conn, subject_title="UAE launch",
                        proposal_text="A retail push across the UAE.",
                        market="UAE", starts_on="2026-03-02", ends_on="2026-03-10")

    assert "Ramadan" in _clash(package)["what_it_means"]


# ── a calendar of 2026 dates, read in 2027 ──────────────────────────────────

def test_a_window_past_the_shipped_calendar_says_the_calendar_ran_out(conn):
    """Every row is a 2026 literal. On 1 January 2027 the product starts giving a clean bill
    to every window on earth, with `absent` promoted into "what the server already
    established", and the only clue is worded to blame the customer."""
    package = _proposal(conn, market="Mexico", starts_on="2027-06-15", ends_on="2027-07-10")

    clash = _clash(package)
    assert clash["status"] == "calendar_expired"
    assert "2026" in clash["what_it_means"]


def test_a_window_inside_the_shipped_span_is_checked_normally(conn):
    package = _proposal(conn, market="Mexico", starts_on="2026-06-15", ends_on="2026-07-10")

    assert _clash(package)["status"] == "present"


def test_an_expired_window_still_reports_the_customer_s_own_events(conn):
    """The shipped calendar running out does not stop this customer's own records applying —
    saying "we cannot check" over the top of events they recorded would be its own falsehood."""
    context.record(conn, starts_on="2027-06-20", ends_on="2027-06-25", scope="market",
                   scope_value="Mexico", kind="regulatory_change",
                   description="New labelling rules take effect.", recorded_by="R. Vega")

    package = _proposal(conn, market="Mexico", starts_on="2027-06-15", ends_on="2027-07-10")

    clash = _clash(package)
    assert clash["status"] == "present"
    assert "labelling" in clash["what_it_means"]
    assert "2026" in clash["what_it_means"], "and it still says the shipped calendar ran out"


# ── the silence test, and what it must never assert ─────────────────────────

def test_a_plan_discussing_it_in_its_own_words_is_not_reported_silent(conn):
    """A plan discusses a thing in ITS words, not in the library's. "We pre-bought FIFA
    inventory around the fixtures" addresses the World Cup."""
    package = _proposal(conn, subject_title="Mexico launch",
                        proposal_text="Media pricing in Mexico spikes for the tournament; we "
                                      "have pre-bought FIFA inventory around the fixtures.",
                        market="Mexico", starts_on="2026-06-15", ends_on="2026-07-10")

    assert _clash(package)["unaddressed"] == []


def test_an_incidental_substring_does_not_silence_an_event(conn):
    """"Our partner agency Eidos Media" silenced Eid. Word boundaries, not substrings."""
    package = _proposal(conn, subject_title="UAE launch",
                        proposal_text="Our partner agency Eidos Media runs the buy.",
                        market="UAE", starts_on="2026-03-20", ends_on="2026-03-22")

    assert "Eid" in _clash(package)["unaddressed"]


def test_an_unaccented_spelling_counts(conn):
    package = _proposal(conn, subject_title="Mexico launch",
                        proposal_text="A campaign across Mexico for Dia de Muertos.",
                        market="Mexico", starts_on="2026-11-01", ends_on="2026-11-02")

    assert _clash(package)["unaddressed"] == []


def test_it_never_claims_a_plan_addressed_an_event_it_could_not_check(conn):
    """The worst sentence this feature could emit, and it emitted it: "The proposal names all
    of them, so this is a timing it chose", over a clash set made entirely of customer records
    that have no short name to look for — an affirmative claim about a document nobody read,
    carrying `basis: computed`. It fires for the whole of §9.6's contribution."""
    context.record(conn, starts_on="2026-09-01", ends_on="2026-09-30", scope="market",
                   scope_value="Mexico", kind="regulatory_change",
                   description="New labelling rules take effect.", recorded_by="R. Vega")

    package = _proposal(conn, subject_title="Mexico launch", proposal_text="A retail push.",
                        market="Mexico", starts_on="2026-09-02", ends_on="2026-09-10")

    clash = _clash(package)
    said = clash["what_it_means"].lower()
    assert "names all of them" not in said
    assert "timing it chose" not in said
    assert clash["not_checked_for_silence"] == 1
    assert "no short name" in said


def test_a_plan_naming_every_shipped_event_says_only_that(conn):
    """Naming a thing is not addressing it — that is the whole of "flagged it and then never
    addressed it". The sentence may say the first and must not claim the second."""
    package = _proposal(conn, subject_title="Mexico launch",
                        proposal_text="A retail push timed to the World Cup.",
                        market="Mexico", starts_on="2026-06-15", ends_on="2026-07-10")

    said = _clash(package)["what_it_means"].lower()
    assert "names them all by name" in said
    assert "addresses them is a question for the reader" in said


# ── truncation must not delete the check ────────────────────────────────────

def test_a_busy_customer_log_does_not_delete_the_shipped_events(conn):
    """The display list caps at eight by salience, which sorts seeded rows LAST — right for
    §9.6, where a customer's recorded flood should outrank a shipped holiday in a list
    somebody reads, and exactly wrong here, where the shipped rows are the point. Eight
    customer records silently deleted the World Cup, Ramadan and Buen Fin from the check, and
    the fact then stated "8 thing(s)" as a count of what it had seen."""
    for n in range(8):
        context.record(conn, starts_on=f"2026-0{n % 8 + 1}-01", ends_on=f"2026-0{n % 8 + 1}-28",
                       scope="market", scope_value="Mexico", kind="regulatory_change",
                       description=f"Regulatory change number {n}.", recorded_by="R. Vega")

    package = _proposal(conn, subject_title="Mexico launch", proposal_text="A retail push.",
                        market="Mexico", starts_on="2026-01-01", ends_on="2026-12-31")

    clash = _clash(package)
    assert "World Cup" in clash["unaddressed"]
    assert "Buen Fin" in clash["unaddressed"]
    assert len(clash["clashes"]) > 8, "the check is not a display list"


# ── the hedge has to reach the reader on this path too ──────────────────────

def test_a_seasonal_range_says_it_is_a_season(conn):
    """A `seasonal` monsoon whose onset moves by weeks and a `fixed` Singles' Day arrived as
    two date ranges differing by one unexplained token."""
    package = _proposal(conn, subject_title="India launch",
                        proposal_text="An outdoor activation.", market="India",
                        starts_on="2026-07-01", ends_on="2026-07-20")

    monsoon = next(c for c in _clash(package)["clashes"] if "monsoon" in c["description"])
    assert monsoon["certainty"] == "seasonal"
    assert "season rather than a date" in monsoon["certainty_note"]


def test_a_customer_event_carries_no_shipped_hedge(conn):
    """The note says "this came from the calendar shipped with this product". Attaching it to
    something the customer recorded would be a false statement about its provenance."""
    context.record(conn, starts_on="2026-09-01", ends_on="2026-09-30", scope="market",
                   scope_value="Mexico", kind="regulatory_change",
                   description="New labelling rules take effect.", recorded_by="R. Vega")

    package = _proposal(conn, market="Mexico", starts_on="2026-09-02", ends_on="2026-09-10")

    theirs = _clash(package)["clashes"][0]
    assert theirs["seeded"] is False
    assert theirs["certainty_note"] == ""


def test_the_model_is_told_what_certainty_means(conn):
    package = _proposal(conn, subject_title="India launch",
                        proposal_text="An outdoor activation.", market="India",
                        starts_on="2026-07-01", ends_on="2026-07-20")

    assert "certainty" in package["note"].lower()


# ── "impossible to drop" ────────────────────────────────────────────────────

def test_a_clash_the_plan_never_names_becomes_a_server_finding(conn):
    """"That flag should have come from the system and been IMPOSSIBLE TO DROP." Landing in
    `computed` gets §2.4's write protection — the model cannot forge one — and not §7.8's
    persistence: the model can ignore it. `save_evaluation(verdict="approve", findings=[])`
    was accepted and the World Cup appeared nowhere in the stored judgment, so a later reader
    could not tell it from one made before §9.7 shipped."""
    cid = core.ingest_campaign(conn, title="Mexico launch", market="Mexico", status="proposed",
                               detail="A retail activation across Mexico City.",
                               starts_on="2026-06-15", ends_on="2026-07-10")["campaign_id"]

    saved = core.save_evaluation(conn, subject_title="Mexico launch", verdict="approve",
                                 summary="Looks like Peru, which worked.", campaign_id=cid,
                                 findings=[])

    stored = store.get_evaluation(conn, saved["evaluation_id"])
    raised = [f for f in stored["findings"] if f["basis"] == "computed"]
    assert any("World Cup" in f["finding"] for f in raised), raised
    assert "World Cup" in json.dumps(stored)


def test_the_server_finding_is_not_blocking(conn):
    """"Do not turn it into a blocking finding on its own" is the instruction, and the server
    has to hold itself to it: overlapping a fixed date is the point of some campaigns."""
    cid = core.ingest_campaign(conn, title="Mexico launch", market="Mexico", status="proposed",
                               detail="A retail activation across Mexico City.",
                               starts_on="2026-06-15", ends_on="2026-07-10")["campaign_id"]

    saved = core.save_evaluation(conn, subject_title="Mexico launch", verdict="approve",
                                 summary="Looks like Peru, which worked.", campaign_id=cid,
                                 findings=[])

    stored = store.get_evaluation(conn, saved["evaluation_id"])
    clash = next(f for f in stored["findings"] if f.get("category") == "calendar_clash")
    assert clash["severity"] == "should_fix"
    assert saved["verdict"] == "approve", "the server argues with a verdict, never replaces it"


def test_a_plan_that_names_the_clash_raises_nothing(conn):
    """The finding is the SILENCE. A plan that names what it runs into has said what there was
    to say, and a finding raised anyway is the nuisance that teaches a reader to skip them."""
    cid = core.ingest_campaign(conn, title="Mexico launch", market="Mexico", status="proposed",
                               detail="A retail push timed to the World Cup.",
                               starts_on="2026-06-15", ends_on="2026-07-10")["campaign_id"]

    saved = core.save_evaluation(conn, subject_title="Mexico launch", verdict="approve",
                                 summary="Looks like Peru.", campaign_id=cid, findings=[])

    stored = store.get_evaluation(conn, saved["evaluation_id"])
    assert not [f for f in stored["findings"] if f.get("category") == "calendar_clash"]


def test_a_proposal_with_no_record_still_raises_it(conn):
    """The "judge this new pitch" flow has no stored record, and it is the flow this item is
    most about — a proposal being reviewed is exactly when a fixed date could still be moved
    around."""
    saved = core.save_evaluation(
        conn, subject_title="Mexico launch", verdict="approve",
        summary="A retail activation across Mexico City in June.",
        subject_text="A retail activation across Mexico City, 15 June 2026 to 10 July 2026.",
        markets=["Mexico"], findings=[])

    stored = store.get_evaluation(conn, saved["evaluation_id"])
    assert any(f.get("category") == "calendar_clash" for f in stored["findings"])


def test_the_saved_judgment_records_what_the_calendar_said(conn):
    """§9.9 reconciles predicted against actual against context, and it cannot do that against
    a check nobody stored. The same reasoning as §9.5's `execution_at_save`."""
    cid = core.ingest_campaign(conn, title="Mexico launch", market="Mexico", status="proposed",
                               detail="A retail activation across Mexico City.",
                               starts_on="2026-06-15", ends_on="2026-07-10")["campaign_id"]

    saved = core.save_evaluation(conn, subject_title="Mexico launch", verdict="approve",
                                 summary="Looks like Peru.", campaign_id=cid, findings=[])

    stamped = store.get_evaluation(conn, saved["evaluation_id"])["evidence"]["calendar_clash"]
    assert stamped["status"] == "present"
    assert stamped["unaddressed"] == ["World Cup"]


def test_a_stated_window_counts_as_a_date_in_the_brief(conn):
    """Two adjacent lines of the block the model is told to treat as established: "No date
    appears anywhere in this brief" and "This window (2026-06-15 to 2026-07-10) runs into…".
    The brief was formally faulted for carrying no dates in the same response where the server
    named its window twice."""
    cid = core.ingest_campaign(conn, title="Mexico launch", market="Mexico", status="proposed",
                               detail="A retail activation across Mexico City.",
                               starts_on="2026-06-15", ends_on="2026-07-10")["campaign_id"]

    package = core.prepare_evaluation(conn, subject_title="Mexico launch",
                                      proposal_text="A retail activation.", campaign_id=cid)

    assert package["computed"]["date_coverage"]["status"] == "present"
    assert "window" in package["computed"]["date_coverage"]["what_it_means"].lower()


def test_a_brief_with_no_dates_and_no_window_is_still_faulted(conn):
    cid = core.ingest_campaign(conn, title="Mexico launch", market="Mexico", status="proposed",
                               detail="A retail activation across Mexico City.")["campaign_id"]

    package = core.prepare_evaluation(conn, subject_title="Mexico launch",
                                      proposal_text="A retail activation.", campaign_id=cid)

    assert package["computed"]["date_coverage"]["status"] == "absent"


def test_a_heuristic_window_on_a_record_stays_heuristic(conn):
    """The record's basis was thrown away and any non-empty date relabelled `stated` — so
    `campaign_context` hedged "may have run during" while `prepare_evaluation` asserted a
    clash, on the same record, from the same dates, with `basis: computed` over a window built
    out of a competitor's launch date."""
    cid = core.ingest_campaign(
        conn, title="Gulf push", market="UAE", status="proposed",
        detail="Gulf push. Benchmark: a rival launched 1 February 2026. Asset spec due "
               "20 March 2026.")["campaign_id"]
    assert context.window_of(conn, cid)["basis"] == "heuristic"

    clash = _clash(core.prepare_evaluation(conn, subject_title="Gulf push",
                                           proposal_text="A Gulf push.", campaign_id=cid))

    assert clash["window"]["basis"] == "heuristic"
    assert "read from" in clash["what_it_means"].lower()


def test_the_deck_counts_as_the_plan_for_the_silence_test(conn):
    """A plan addresses a thing in the deck, not in the one-line summary — reading only
    `detail` would call a deck that discusses the World Cup for four slides silent."""
    cid = core.ingest_campaign(
        conn, title="Mexico launch", market="Mexico", status="proposed",
        detail="A retail activation.",
        deck_text="Timing: we are building the whole push around the World Cup fixtures.",
        starts_on="2026-06-15", ends_on="2026-07-10")["campaign_id"]

    clash = _clash(core.prepare_evaluation(conn, subject_title="Mexico launch",
                                           proposal_text="A retail activation.",
                                           campaign_id=cid))

    assert clash["unaddressed"] == []


# ── seeding hygiene ─────────────────────────────────────────────────────────

def test_re_seeding_an_unchanged_calendar_writes_nothing(conn):
    """`INSERT OR REPLACE` on every start rewrote `created_at` for all of them, touched the
    database file on a no-op, and silently reverted any local edit to a seeded row on the next
    restart."""
    before = {e["id"]: e["created_at"] for e in store.context_events(conn, seeded=True)}

    context.seed(conn)

    after = {e["id"]: e["created_at"] for e in store.context_events(conn, seeded=True)}
    assert after == before


def test_a_withdrawn_seed_row_is_not_resurrected_by_the_next_start(conn):
    """The one guarantee `seed`'s docstring makes, and nothing tested it — the existing
    withdrawal test never called `seed()` afterwards."""
    ramadan = next(e for e in store.context_events(conn, seeded=True)
                   if "Ramadan" in e["description"] and e["scope_value"] == "UAE")
    context.withdraw(conn, event_id=ramadan["id"], why="We do not trade there.",
                     withdrawn_by="R. Vega")

    context.seed(conn)

    assert ramadan["id"] not in {e["id"] for e in store.context_events(conn)}


def test_the_seed_key_is_unique_on_an_upgraded_database(conn):
    """SQLite cannot `ADD COLUMN ... UNIQUE`, so the migration stripped it and an upgraded
    database had no uniqueness at all — the schema comment promised a guarantee half the
    installed base did not have."""
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO context_events (id, starts_on, scope, kind, description, "
            "recorded_by, seed_key, created_at) VALUES ('ctx_x','2026-01-01','global',"
            "'fixed_calendar','dupe','seed','ramadan.2026@UAE',1.0)")


def test_withdrawing_one_market_leaves_the_others(conn):
    """One row per market, so a customer who does not trade in Turkey can say so without
    removing Ramadan from the UAE."""
    turkey = next(e for e in store.context_events(conn, seeded=True)
                  if "Ramadan" in e["description"] and e["scope_value"] == "Turkey")

    context.withdraw(conn, event_id=turkey["id"], why="We do not trade in Turkey.",
                     withdrawn_by="R. Vega")

    still = [e for e in store.context_events(conn, seeded=True)
             if "Ramadan" in e["description"]]
    assert any(e["scope_value"] == "UAE" for e in still)


# ── the dates themselves ────────────────────────────────────────────────────

def test_the_shipped_dates_are_the_dates(conn):
    """A shipped date carries the product's authority, and five of the first thirteen were
    wrong — found by review, not by this suite: shifting the World Cup by seven weeks, Ramadan
    by a week and Black Friday by a week all passed. A calendar with no test is a claim
    nobody checked.

    Each of these is pinned deliberately. Changing one should be a decision somebody makes and
    writes down, not something a refactor can do quietly."""
    import calendar_seed

    expected = {
        # 1 Ramadan 1447 by the Umm al-Qura calendar, and the date the product review itself
        # quotes ("18 Feb – 19 Mar"). Observed: it moves by a day between countries.
        "ramadan.2026": ("2026-02-18", "2026-03-19"),
        "eid-al-fitr.2026": ("2026-03-20", "2026-03-22"),
        # The published State Council holiday, which starts at New Year's Eve rather than at
        # the new year itself — the row's own description promises the official dates.
        "chinese-new-year.2026": ("2026-02-15", "2026-02-23"),
        "golden-week-cn.2026": ("2026-10-01", "2026-10-07"),
        # 3 May 2026 is a Sunday, so 6 May is the substitute holiday.
        "golden-week-jp.2026": ("2026-04-29", "2026-05-06"),
        "fifa-world-cup.2026": ("2026-06-11", "2026-07-19"),
        # US Thanksgiving 2026 is 26 November.
        "black-friday.2026": ("2026-11-27", "2026-11-30"),
        "singles-day.2026": ("2026-11-11", "2026-11-11"),
        # The 2026 edition runs five days.
        "buen-fin.2026": ("2026-11-13", "2026-11-17"),
        "dia-de-muertos.2026": ("2026-11-01", "2026-11-02"),
        "diwali.2026": ("2026-10-25", "2026-11-10"),
        "monsoon-in.2026": ("2026-06-01", "2026-09-30"),
        # Mainland South-East Asia is WET May to October and dry November to March; maritime
        # is the opposite half of the year. One row for both was wrong for most of the region.
        "rainy-season-mainland-sea.2026": ("2026-05-15", "2026-10-31"),
        "rainy-season-maritime-sea.2026": ("2026-11-01", "2027-03-31"),
    }
    shipped = {key: (starts, ends)
               for key, _m, _k, starts, ends, *_rest in calendar_seed.EVENTS}

    assert shipped == expected


def test_a_bangkok_campaign_is_dry_in_december_and_wet_in_july(conn):
    """One rainy-season row for all of South-East Asia fired in exactly the wrong half of the
    year for the mainland: a Bangkok outdoor campaign in December was told it clashed with the
    rainy season, and one in July was told nothing was on record."""
    december = _proposal(conn, subject_title="Bangkok", proposal_text="An outdoor activation.",
                         market="Thailand", starts_on="2026-12-01", ends_on="2026-12-20")
    july = _proposal(conn, subject_title="Bangkok", proposal_text="An outdoor activation.",
                     market="Thailand", starts_on="2026-07-01", ends_on="2026-07-20")

    assert "rainy" not in _clash(december)["what_it_means"].lower()
    assert "rainy" in _clash(july)["what_it_means"].lower()


def test_an_indian_campaign_in_late_october_meets_diwali(conn):
    """A one-day row told a campaign running through the largest gifting build-up of the
    Indian year that nothing was on."""
    package = _proposal(conn, subject_title="India launch",
                        proposal_text="A gifting push.", market="India",
                        starts_on="2026-10-26", ends_on="2026-11-07")

    assert "Diwali" in _clash(package)["what_it_means"]


def test_every_shipped_event_carries_a_certainty_and_a_note(conn):
    """`certainty` dropped at seed, and `note_for` returning "", both passed the suite — so
    the hedge that makes an approximate date honest was untested end to end."""
    import calendar_seed

    for event in store.context_events(conn, seeded=True):
        assert event["certainty"] in ("fixed", "announced", "observed", "seasonal"), event
        assert len(calendar_seed.note_for(event["certainty"])) > 20, event["certainty"]


def test_every_shipped_event_says_who_it_came_from(conn):
    for event in store.context_events(conn, seeded=True):
        assert "shipped with this product" in event["recorded_by"]


# ── the note the model actually reads ───────────────────────────────────────

def test_the_note_carries_the_calendar_instruction(conn):
    """Silencing `_say_the_calendar` entirely passed the suite: the phrase the old test looked
    for was reaching the note from the FACT's own sentence, not from the function under test.
    The one piece of text telling the model not to turn a clash into a blocking finding had no
    test at all."""
    package = _proposal(conn, subject_title="Mexico launch",
                        proposal_text="A retail push.", market="Mexico",
                        starts_on="2026-06-15", ends_on="2026-07-10")

    said = core._say_the_calendar(_clash(package))
    assert "calendar_clash" in said
    assert "not a criticism" in said.lower()
    assert "blocking finding" in said.lower()
    assert said in package["note"]


def test_the_note_names_what_the_plan_was_silent_about(conn):
    package = _proposal(conn, subject_title="Mexico launch",
                        proposal_text="A retail push.", market="Mexico",
                        starts_on="2026-06-15", ends_on="2026-07-10")

    assert "World Cup" in core._say_the_calendar(_clash(package))


def test_the_note_warns_that_a_shipped_date_can_be_wrong(conn):
    package = _proposal(conn, subject_title="Mexico launch",
                        proposal_text="A retail push.", market="Mexico",
                        starts_on="2026-06-15", ends_on="2026-07-10")

    said = core._say_the_calendar(_clash(package)).lower()
    assert "shipped with this product" in said
    assert "can be wrong" in said


def test_the_note_is_silent_when_there_is_no_clash(conn):
    """A standing paragraph about the calendar on every judgment is the note that fires on
    everything — §9.5's rule, two items over."""
    package = _proposal(conn, market="Mexico", starts_on="2026-09-02", ends_on="2026-09-10")

    assert core._say_the_calendar(_clash(package)) == ""


# ── the edges the adversarial pass walked ──────────────────────────────────

def test_dates_passed_alongside_a_campaign_id_are_refused(conn):
    """§7.2's rule: with a `campaign_id` the record answers "which brief is this", and passing
    `market` alongside one is already REFUSED. Dates are the same kind of claim and were
    silently dropped instead — documented, which is not the same as safe."""
    cid = core.ingest_campaign(conn, title="Mexico launch", market="Mexico", status="proposed",
                               detail="A retail activation.", starts_on="2026-06-15",
                               ends_on="2026-07-10")["campaign_id"]

    with pytest.raises(ValueError) as e:
        core.prepare_evaluation(conn, subject_title="Mexico launch",
                                proposal_text="A retail activation.", campaign_id=cid,
                                starts_on="2026-01-01", ends_on="2026-01-31")
    assert "starts_on" in str(e.value)


def test_a_record_with_no_market_says_which_record(conn):
    """"No market was named" is true of the caller and false of this path, where the caller
    cannot name one — and it points nowhere."""
    cid = core.ingest_campaign(conn, title="Mexico launch", status="proposed",
                               detail="A retail activation.", starts_on="2026-06-15",
                               ends_on="2026-07-10")["campaign_id"]

    clash = _clash(core.prepare_evaluation(conn, subject_title="Mexico launch",
                                           proposal_text="A retail activation.",
                                           campaign_id=cid))

    assert clash["status"] == "nothing_to_check"
    assert "update_campaign" in clash["what_it_means"]


def test_the_fact_carries_the_evidence_it_read(conn):
    """Every computed fact in this library attaches what it looked at — a check that cannot
    show its working is an assertion, and the server's assertions carry more weight than a
    model's."""
    package = _proposal(conn, subject_title="Mexico launch",
                        proposal_text="A retail push.", market="Mexico",
                        starts_on="2026-06-15", ends_on="2026-07-10")

    clash = _clash(package)
    assert clash["evidence"]
    assert "World Cup" in clash["evidence"][0]
    assert clash["clashes"][0]["seeded"] is True


def test_a_window_ending_on_the_day_an_event_starts_clashes(conn):
    """Every test window started inside its event, so "the end is ignored" and "an inverted
    window is accepted" both passed unnoticed."""
    package = _proposal(conn, subject_title="Mexico launch", proposal_text="A retail push.",
                        market="Mexico", starts_on="2026-05-01", ends_on="2026-06-11")

    assert _clash(package)["status"] == "present"


def test_a_window_ending_the_day_before_does_not(conn):
    package = _proposal(conn, subject_title="Mexico launch", proposal_text="A retail push.",
                        market="Mexico", starts_on="2026-05-01", ends_on="2026-06-10")

    assert _clash(package)["status"] == "absent"


def test_an_inverted_proposed_window_is_refused(conn):
    with pytest.raises(ValueError) as e:
        _proposal(conn, market="Mexico", starts_on="2026-07-10", ends_on="2026-06-15")
    assert "before" in str(e.value).lower()


def test_the_region_the_caller_names_is_searched(conn):
    """`region` is one of the two fields §7.2 lets a caller use to say which brief this is, and
    the clash check was reading `market` only."""
    package = _proposal(conn, subject_title="SEA launch", proposal_text="An outdoor push.",
                        region="SEA", starts_on="2026-12-01", ends_on="2026-12-20")

    assert _clash(package)["status"] == "present"
    assert "rainy" in _clash(package)["what_it_means"].lower()


def test_an_accented_spelling_counts(conn):
    """The earlier test used "Dia de Muertos" against an alias that is already unaccented, so
    stripping accents was never exercised. A plan writes "Ramadán" as readily as "Ramadan"."""
    package = _proposal(conn, subject_title="Morocco launch",
                        proposal_text="Timed to Ramadán across Morocco.",
                        market="Morocco", starts_on="2026-02-20", ends_on="2026-03-10")

    assert _clash(package)["status"] == "present"
    assert "Ramadan" not in _clash(package)["unaddressed"]


def test_a_withdrawn_seed_row_is_not_resurrected_by_a_corrected_date(conn):
    """The earlier test re-seeded an UNCHANGED pack, so the no-op guard answered before the
    withdrawal guard was reached and the one that matters was never exercised. The case is a
    customer who withdrew a row and an upgrade that then corrects its date."""
    ramadan = next(e for e in store.context_events(conn, seeded=True)
                   if "Ramadan" in e["description"] and e["scope_value"] == "UAE")
    context.withdraw(conn, event_id=ramadan["id"], why="We do not trade there.",
                     withdrawn_by="R. Vega")

    store.insert_context_event(
        conn, starts_on="2026-02-19", ends_on="2026-03-20", scope="market",
        scope_value="UAE", kind="fixed_calendar", description="Ramadan (corrected upstream).",
        recorded_by="shipped with this product", seeded=True, seed_key=ramadan["seed_key"],
        certainty="observed")

    assert ramadan["id"] not in {e["id"] for e in store.context_events(conn)}
    back = store.get_context_event(conn, ramadan["id"])
    assert back["withdrawn_at"] is not None
    assert back["starts_on"] == ramadan["starts_on"], "and the correction did not land either"


def test_the_seed_key_stays_unique_on_a_database_built_before_the_column(conn, tmp_path):
    """SQLite cannot `ADD COLUMN ... UNIQUE`, so the migration strips it — the fresh-install
    test passes on the table's own constraint and says nothing about the upgraded path, which
    is half the installed base."""
    import sqlite3

    legacy = sqlite3.connect(tmp_path / "legacy.db")
    legacy.row_factory = sqlite3.Row
    legacy.execute("CREATE TABLE context_events (id TEXT PRIMARY KEY, starts_on TEXT NOT NULL,"
                   " ends_on TEXT, scope TEXT NOT NULL, scope_value TEXT, kind TEXT NOT NULL,"
                   " description TEXT NOT NULL, recorded_by TEXT NOT NULL,"
                   " channels_disrupted TEXT NOT NULL DEFAULT '[]',"
                   " seeded INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL)")
    legacy.commit()
    store.upgrade(legacy)

    with pytest.raises(sqlite3.IntegrityError):
        legacy.execute(
            "INSERT INTO context_events (id, starts_on, scope, kind, description, "
            "recorded_by, seed_key, created_at) VALUES ('ctx_x','2026-01-01','global',"
            "'fixed_calendar','dupe','seed','ramadan.2026@UAE',1.0)")
    legacy.close()
