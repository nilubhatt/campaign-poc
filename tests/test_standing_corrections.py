"""
§8.6: the same loop for standing corrections.

The review: *"New client feedback is the same shape of event as a new metric. 'We want the
seeding box to carry one colourway' should travel the identical path: provisional on first
mention, counted, promoted to the checklist once it recurs across markets and a person confirms
it, with provenance attached. One learning mechanism for both keeps the rulebook coherent, and
means the ten standing corrections already extracted have somewhere to live and grow rather
than being frozen at whatever was true the day they were written."*

**"One learning mechanism" is the requirement, not "a second one shaped like the first."** A
parallel implementation with its own thresholds and its own gate is two mechanisms that agree
today and drift by the next item — which is the failure this codebase has now found five times
(D55, D75, D88, and twice in §8.3 alone). So the loop lives in `learning.py` and both callers
use it, and the tests below run the SAME scenarios against both kinds. Every defect the §8.3
reviews found — one market typed three ways, versions of one brief counted separately, a brief
with no market held to every checklist — must be fixed for corrections by construction, because
there is only one place it could be wrong.

**What differs is the data, not the loop.** A measure is a key with numeric values; a
correction is a sentence. The sentence needs `provenance` — *"Each needs its provenance shipped
with it so a judgment can cite where the rule came from"* — and a correction with none is not a
standing correction, it is an opinion in a text field.

The ten corrections themselves are NOT seeded here. They are one customer's rules, and §12.3
ships them as an example overlay; seeding them into the product is the thing this project has
refused since §5.1.
"""
import pytest

import core
import corrections
import learning
import metrics

SEEDING = "Seed a single colourway across all recipients."
PHOTO = "Photo time goes before the workout."


def _campaign(conn, title, market="LATAM", **kw):
    import core
    return core.ingest_campaign(conn, title=title, market=market, status="concluded",
                                detail="A launch.", **kw)["campaign_id"]


def _noted(conn, text, *, markets, provenance="Slide 23"):
    for n, market in enumerate(markets):
        corrections.note(conn, text=text, campaign_id=_campaign(conn, f"{text[:6]}-{market}-{n}",
                                                                market=market),
                         provenance=provenance)


# ── one mechanism, not two ──────────────────────────────────────────────────

def test_both_kinds_run_through_the_same_gate(conn):
    """"One learning mechanism for both keeps the rulebook coherent." Two implementations that
    agree today drift by the next item — this codebase has found that five times."""
    assert corrections.GRADUATION_CAMPAIGNS is learning.GRADUATION_CAMPAIGNS
    assert metrics.GRADUATION_CAMPAIGNS is learning.GRADUATION_CAMPAIGNS
    assert corrections.GRADUATION_MARKETS is learning.GRADUATION_MARKETS
    assert metrics.GRADUATION_MARKETS is learning.GRADUATION_MARKETS
    assert corrections.RETIREMENT_AFTER is learning.RETIREMENT_AFTER
    assert metrics.RETIREMENT_AFTER is learning.RETIREMENT_AFTER


def test_the_thresholds_are_not_copied_into_either_caller(conn):
    """A constant re-declared beside the one it mirrors is the copy that drifts. Reading them
    from `learning` is what makes "the identical path" checkable rather than aspirational."""
    import re

    for module in ("metrics.py", "corrections.py"):
        source = open(module, encoding="utf-8").read()
        assert not re.search(r"^GRADUATION_CAMPAIGNS\s*=\s*\d", source, re.M), module
        assert not re.search(r"^GRADUATION_MARKETS\s*=\s*\d", source, re.M), module
        assert not re.search(r"^RETIREMENT_AFTER\s*=\s*\d", source, re.M), module


# ── provisional on first mention ────────────────────────────────────────────

def test_a_new_correction_is_recorded_not_rejected(conn):
    """The §8.2 half that applies here: refusing it loses the feedback, and the client said it
    once whether or not the library was ready to hear it."""
    cid = _campaign(conn, "Peru")
    noted = corrections.note(conn, text=SEEDING, campaign_id=cid, provenance="Peru slide 12")

    assert noted["status"] == "provisional"
    assert corrections.describe(conn, noted["correction_id"])["text"] == SEEDING


def test_a_new_correction_is_not_silently_accepted(conn):
    """A rule that arrives on one deck and is never questioned is how one person's preference
    becomes everyone's requirement — the §8.3 capture, in the softer half of the library."""
    noted = corrections.note(conn, text=SEEDING, campaign_id=_campaign(conn, "Peru"),
                             provenance="Peru slide 12")
    assert noted["new_correction"]
    assert noted["new_correction"]["next_actions"]


def test_provenance_is_required(conn):
    """"Each needs its provenance shipped with it so a judgment can cite where the rule came
    from." A correction with none is not a standing correction, it is an opinion in a text
    field — and a judgment citing it could say only "the library says so"."""
    with pytest.raises(ValueError) as e:
        corrections.note(conn, text=SEEDING, campaign_id=_campaign(conn, "Peru"),
                         provenance="  ")
    assert "provenance" in str(e.value).lower()


def test_the_same_correction_mentioned_twice_is_one_correction(conn):
    """"Provisional on first mention, COUNTED." Two rows for one rule counts nothing, and the
    count is the whole basis of the gate."""
    first = corrections.note(conn, text=SEEDING, campaign_id=_campaign(conn, "Peru"),
                             provenance="Peru slide 12")
    again = corrections.note(conn, text="Seed a single colourway across all recipients",
                             campaign_id=_campaign(conn, "Australia", market="APAC"),
                             provenance="AU brief")

    assert again["correction_id"] == first["correction_id"]
    assert corrections.describe(conn, first["correction_id"])["times_seen"] == 2


def test_every_sighting_keeps_its_own_provenance(conn):
    """"Peru (30 influencers, praised); instructed independently in Australia" is ONE
    correction with two origins, and the second is what makes it more than a house style. A
    single provenance field would have to overwrite one of them."""
    first = corrections.note(conn, text=SEEDING, campaign_id=_campaign(conn, "Peru"),
                             provenance="Peru slide 12, praised")
    corrections.note(conn, text=SEEDING, campaign_id=_campaign(conn, "Australia",
                                                               market="APAC"),
                     provenance="AU brief, instructed independently")

    where = [s["provenance"] for s in corrections.sightings(conn, first["correction_id"])]
    assert where == ["Peru slide 12, praised", "AU brief, instructed independently"]


# ── the identical gate ──────────────────────────────────────────────────────

def test_it_is_promoted_once_it_recurs_across_markets_and_a_person_confirms(conn):
    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    cid = corrections.find(conn, SEEDING)["correction_id"]

    assert corrections.graduation(conn, cid)["eligible"] is True
    corrections.graduate(conn, cid, confirmed_by="R. Vega")
    assert corrections.describe(conn, cid)["status"] == "expected"


def test_one_market_is_not_enough_for_a_correction_either(conn):
    """"Recurs across MARKETS." One client contact repeating themselves on five decks is one
    opinion stated five times, and promoting it makes it everybody's rule."""
    _noted(conn, SEEDING, markets=["LATAM"] * 5)
    cid = corrections.find(conn, SEEDING)["correction_id"]

    gate = corrections.graduation(conn, cid)
    assert gate["eligible"] is False
    assert gate["markets"] == 1


def test_a_person_still_has_to_confirm_it(conn):
    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    cid = corrections.find(conn, SEEDING)["correction_id"]

    assert corrections.describe(conn, cid)["status"] == "provisional"
    with pytest.raises(ValueError) as e:
        corrections.graduate(conn, cid, confirmed_by="  ")
    assert "person" in str(e.value).lower()


def test_who_confirmed_it_is_recorded(conn):
    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    cid = corrections.find(conn, SEEDING)["correction_id"]
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    assert corrections.describe(conn, cid)["confirmed_by"] == "R. Vega"


# ── the defects §8.3's reviews found, which sharing must fix for free ───────

def test_one_market_spelled_three_ways_is_one_market(conn):
    _noted(conn, SEEDING, markets=["SEA", "sea", "Sea"])
    cid = corrections.find(conn, SEEDING)["correction_id"]

    assert corrections.graduation(conn, cid)["markets"] == 1
    assert corrections.graduation(conn, cid)["eligible"] is False


def test_three_versions_of_one_brief_are_one_brief(conn):
    import core

    prev = None
    for n in range(3):
        prev = core.ingest_campaign(conn, title=f"Colombia v{n + 1}", market="LATAM",
                                    supersedes=prev, status="concluded",
                                    detail="A launch.")["campaign_id"]
        corrections.note(conn, text=SEEDING, campaign_id=prev, provenance=f"v{n + 1} slide 6")

    cid = corrections.find(conn, SEEDING)["correction_id"]
    assert corrections.graduation(conn, cid)["campaigns"] == 1


def test_a_brief_with_no_market_is_not_held_to_every_correction(conn):
    import core

    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    corrections.graduate(conn, corrections.find(conn, SEEDING)["correction_id"],
                         confirmed_by="R. Vega")

    loose = core.ingest_campaign(conn, title="Somewhere", status="planned",
                                 detail="A launch.")["campaign_id"]
    report = corrections.standing_for(conn, loose)
    assert report["standing"] == []
    assert report["status"] == "nothing_to_check"


def test_a_market_that_never_raised_it_is_not_judged_against_it(conn):
    """The §8.3 rule, and the reason the checklist is scoped at all: a rule that earned its
    place on LATAM, APAC and EMEA is not a standard a market nobody raised it in agreed to, and
    applying it there makes every new market fail on its first brief."""
    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    corrections.graduate(conn, corrections.find(conn, SEEDING)["correction_id"],
                         confirmed_by="R. Vega")

    elsewhere = _campaign(conn, "Tokyo v1", market="JAPAN")
    assert corrections.standing_for(conn, elsewhere)["standing"] == []
    assert corrections.standing_for(conn, _campaign(conn, "Bogota"))["standing"]


def test_the_markets_it_was_raised_in_are_folded_on_the_way_in(conn):
    """Stored unfolded, the list grows a row per spelling — and anything reading it without
    folding again (the gate does; `expected_in` at graduation comes from the gate) sees three
    markets where there is one."""
    _noted(conn, SEEDING, markets=["SEA", "sea", "Sea"])
    entry = corrections.find(conn, SEEDING)

    assert entry["markets"] == ["SEA"]


def test_it_asks_once(conn):
    """§8.2's rule and §8.3's flag. A question repeated on every subsequent mention is one
    nobody reads — and this one arrives on an ordinary note."""
    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    again = corrections.note(conn, text=SEEDING, campaign_id=_campaign(conn, "Chile"),
                             provenance="Chile slide 2")

    assert "newly_eligible" not in again


def test_a_reference_record_is_not_a_brief(conn):
    import core

    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    corrections.graduate(conn, corrections.find(conn, SEEDING)["correction_id"],
                         confirmed_by="R. Vega")
    ref = core.ingest_campaign(conn, title="Brand guidelines", record_type="reference",
                               market="LATAM", detail="Guidelines.")["campaign_id"]

    assert corrections.standing_for(conn, ref)["code"] == "not_a_campaign"


def test_confirming_twice_cannot_erase_who_confirmed_it(conn):
    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    cid = corrections.find(conn, SEEDING)["correction_id"]
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    with pytest.raises(ValueError):
        corrections.graduate(conn, cid, confirmed_by="somebody else")
    assert corrections.describe(conn, cid)["confirmed_by"] == "R. Vega"


def test_the_write_that_makes_it_eligible_asks(conn):
    """§8.3's lesson, one item later: a gate whose third condition is a person needs somebody
    to be asked, and nothing surfaced eligibility until the offer rode on the write."""
    _noted(conn, SEEDING, markets=["LATAM", "APAC"])
    third = corrections.note(conn, text=SEEDING,
                             campaign_id=_campaign(conn, "Dubai", market="EMEA"),
                             provenance="UAE brief")

    assert third["newly_eligible"]["next_actions"][0]["tool"] == "graduate_correction"
    assert "confirmed_by" not in third["newly_eligible"]["next_actions"][0]["prefilled_args"]


# ── it reaches the judgment ─────────────────────────────────────────────────

def test_a_standing_correction_reaches_the_brief_being_judged(conn):
    """"Somewhere to live and grow rather than being frozen." A rule nothing reads is a row in
    a table, and the whole argument for extracting them was that they are what this client
    repeats."""
    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    corrections.graduate(conn, corrections.find(conn, SEEDING)["correction_id"],
                         confirmed_by="R. Vega")

    subject = _campaign(conn, "Colombia v1")
    standing = corrections.standing_for(conn, subject)
    assert [c["text"] for c in standing["standing"]] == [SEEDING]
    assert standing["standing"][0]["provenance"]


def test_it_carries_its_provenance_so_a_judgment_can_cite_it(conn):
    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"], provenance="Peru slide 12")
    corrections.graduate(conn, corrections.find(conn, SEEDING)["correction_id"],
                         confirmed_by="R. Vega")

    standing = corrections.standing_for(conn, _campaign(conn, "Colombia v1"))
    assert "Peru slide 12" in standing["standing"][0]["provenance"]
    assert standing["standing"][0]["confirmed_by"] == "R. Vega"


def test_a_provisional_correction_does_not_reach_the_judgment(conn):
    """It has not been confirmed by anybody. Putting it in front of a judgment as a standing
    rule is the promotion the gate exists to withhold."""
    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    assert corrections.standing_for(conn, _campaign(conn, "Colombia v1"))["standing"] == []


# ── retirement, the same way ────────────────────────────────────────────────

def test_a_rule_nobody_repeats_is_asked_about_never_demoted(conn):
    """The one place this loop deliberately differs from §8.5's, because the same signal means
    opposite things. For a MEASURE, absence is disuse — nobody tracks it any more. For a RULE,
    absence of repetition is usually COMPLIANCE: a client stops restating "seed a single
    colourway" exactly when the agency starts doing it. Retiring on silence demotes precisely
    the rules that are working and keeps the ones being ignored, and no threshold fixes that."""
    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    cid = corrections.find(conn, SEEDING)["correction_id"]
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    last = None
    for n in range(learning.RETIREMENT_AFTER):
        last = corrections.note(conn, text=PHOTO, campaign_id=_campaign(conn, f"Later {n}"),
                                provenance="slide 33")

    asked = last["gone_quiet"]
    assert [q["correction_id"] for q in asked] == [cid]
    assert "cannot tell which" in asked[0]["what_it_means"]
    assert {a["tool"] for a in asked[0]["next_actions"]} == {"set_aside_correction",
                                                             "keep_correction"}
    assert corrections.describe(conn, cid)["status"] == "expected", "asked, not demoted"
    assert corrections.standing_for(conn, _campaign(conn, "Still"))["standing"], \
        "still applied until somebody says otherwise"


def test_it_asks_about_a_quiet_rule_once(conn):
    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    corrections.graduate(conn, corrections.find(conn, SEEDING)["correction_id"],
                         confirmed_by="R. Vega")
    for n in range(learning.RETIREMENT_AFTER):
        corrections.note(conn, text=PHOTO, campaign_id=_campaign(conn, f"Later {n}"),
                         provenance="slide 33")

    again = corrections.note(conn, text=PHOTO, campaign_id=_campaign(conn, "One more"),
                             provenance="slide 33")
    assert "gone_quiet" not in again


def test_keeping_a_quiet_rule_leaves_it_standing(conn):
    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    cid = corrections.find(conn, SEEDING)["correction_id"]
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    corrections.keep(conn, cid)
    assert corrections.describe(conn, cid)["status"] == "expected"
    assert corrections.standing_for(conn, _campaign(conn, "Next"))["standing"]


def test_a_rule_still_being_repeated_is_not_asked_about(conn):
    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    corrections.graduate(conn, corrections.find(conn, SEEDING)["correction_id"],
                         confirmed_by="R. Vega")

    last = corrections.note(conn, text=SEEDING, campaign_id=_campaign(conn, "Again"),
                            provenance="slide 1")
    assert "gone_quiet" not in last


def test_a_rule_somebody_set_aside_does_not_come_back_on_its_own(conn):
    """§8.5 revives a measure that is recorded again, because nothing DECIDED to demote it —
    it aged out. A correction only ever leaves the checklist because a person set it aside, and
    putting it back because the client mentioned it again overrules them without asking."""
    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    cid = corrections.find(conn, SEEDING)["correction_id"]
    corrections.graduate(conn, cid, confirmed_by="R. Vega")
    corrections.set_aside(conn, cid)

    corrections.note(conn, text=SEEDING, campaign_id=_campaign(conn, "Back"),
                     provenance="Chile slide 4")
    assert corrections.describe(conn, cid)["status"] == "ignored"
    assert corrections.standing_for(conn, _campaign(conn, "Next"))["standing"] == []


def test_one_brief_raising_many_rules_is_one_brief(conn):
    """"After M campaigns", not after M notes. One deck returned with twelve comments is one
    deck's worth of evidence that some other rule has gone quiet."""
    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    cid = corrections.find(conn, SEEDING)["correction_id"]
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    busy = _campaign(conn, "One heavily annotated deck")
    # Every result, not just the last: the question is asked ONCE, so a wrong count that fires
    # on note 11 leaves note 14 looking clean.
    asked = [corrections.note(conn, text=f"Rule number {n} about the deck layout.",
                              campaign_id=busy, provenance=f"slide {n}").get("gone_quiet")
             for n in range(learning.RETIREMENT_AFTER + 4)]

    assert not any(asked), asked


def test_a_market_that_never_raised_it_is_not_evidence_it_stopped(conn):
    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    cid = corrections.find(conn, SEEDING)["correction_id"]
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    asked = [corrections.note(conn, text=PHOTO, provenance="slide 33",
                              campaign_id=_campaign(conn, f"Tokyo {n}",
                                                    market="JAPAN")).get("gone_quiet")
             for n in range(learning.RETIREMENT_AFTER + 2)]
    assert not any(asked), asked


def test_only_briefs_since_the_last_mention_count(conn):
    """"Since it was last raised." Counting every brief in the library instead would ask about
    a rule the client restated yesterday as soon as ten decks were on file."""
    import store

    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    cid = corrections.find(conn, SEEDING)["correction_id"]
    for n in range(learning.RETIREMENT_AFTER + 2):
        corrections.note(conn, text=PHOTO, campaign_id=_campaign(conn, f"Before {n}"),
                         provenance="slide 33")
    corrections.note(conn, text=SEEDING, campaign_id=_campaign(conn, "Latest"),
                     provenance="slide 1")
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    entry = corrections.describe(conn, cid)
    assert store.campaigns_that_skipped_correction(
        conn, cid, since=entry["last_seen"], markets=entry["expected_in"]) == 0
    assert corrections.gone_quiet(conn) == []


def test_a_brief_that_raised_it_earlier_has_not_skipped_it(conn):
    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    cid = corrections.find(conn, SEEDING)["correction_id"]
    early = [_campaign(conn, f"Early {n}") for n in range(learning.RETIREMENT_AFTER + 2)]
    for c in early:
        corrections.note(conn, text=SEEDING, campaign_id=c, provenance="slide 1")
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    asked = [corrections.note(conn, text=PHOTO, campaign_id=c,
                              provenance="slide 33").get("gone_quiet") for c in early]
    assert not any(asked), asked


def _standing(conn):
    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"], provenance="Peru slide 12")
    cid = corrections.find(conn, SEEDING)["correction_id"]
    corrections.graduate(conn, cid, confirmed_by="R. Vega")
    return cid


def _save(conn, precedent, **over):
    import core

    args = dict(subject_title="Colombia v1", verdict="revise", summary="A stretch.",
                approve_if="Seed one colourway.",
                findings=[{"severity": "blocking", "kind": "guardrail_breach",
                           "finding": "Seeds four colourways", "fix": "Seed one",
                           "precedent": precedent}])
    args.update(over)
    return core.save_evaluation(conn, **args)


def test_a_finding_can_cite_a_standing_correction(conn):
    """The whole purpose of shipping provenance: *"so a judgment can cite where the rule came
    from"*. Before this, a graduated correction was shown to the model as a rule and could not
    be cited by it — `guardrail_breach` demanded a `rule_id` pointing at a `reference` record,
    and a correction is neither. §5.2's failure, one item after it was named."""
    import store

    cid = _standing(conn)
    saved = _save(conn, {"correction_id": cid, "quote": SEEDING})

    cited = store.get_evaluation(conn, saved["evaluation_id"])["findings"][0]["precedent"]
    assert cited["correction_id"] == cid
    assert "Peru slide 12" in cited["provenance"]
    assert cited["confirmed_by"] == "R. Vega"


def test_the_quote_has_to_be_what_the_rule_actually_says(conn):
    """§6.1's rule, and there is no reason a rule should be checked less carefully than a
    deck — a fabricated quote carries MORE weight here, because a guardrail breach is the one
    class of finding the product says is not debatable."""
    cid = _standing(conn)
    with pytest.raises(ValueError) as e:
        _save(conn, {"correction_id": cid, "quote": "Seed every colourway you have"})
    assert "not what correction" in str(e.value)


def test_a_provisional_correction_cannot_be_cited_as_a_rule(conn):
    """It is exactly what the gate withheld. Citing it as not-debatable would promote it by the
    back door, with nobody's confirmation behind it."""
    corrections.note(conn, text=SEEDING, campaign_id=_campaign(conn, "Peru"),
                     provenance="Peru slide 12")
    cid = corrections.find(conn, SEEDING)["correction_id"]

    with pytest.raises(ValueError) as e:
        _save(conn, {"correction_id": cid, "quote": SEEDING})
    assert "not standing" in str(e.value)


def test_a_quote_cannot_drop_the_half_that_reverses_the_rule(conn):
    """§6.1's bounds are measured against a unit that can be a whole packed slide; a rule is
    one sentence, so both sit far outside it. "Seed more than one colourway per recipient" is
    a faithful substring of "Do not seed more than one colourway per recipient" — and a
    guardrail breach is the one class of finding this product calls not debatable."""
    _noted(conn, "Do not seed more than one colourway per recipient.",
           markets=["LATAM", "APAC", "EMEA"])
    cid = corrections.find(conn,
                           "Do not seed more than one colourway per recipient.")["correction_id"]
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    with pytest.raises(ValueError) as e:
        _save(conn, {"correction_id": cid,
                     "quote": "seed more than one colourway per recipient"})
    assert "opposite of the rule" in str(e.value)


def test_a_length_ratio_would_not_have_caught_that(conn):
    """Recorded because the first attempt used one: "Do not " is seven characters out of
    fifty, so the inverted quote still measures 84% of the rule. The check has to be about the
    negation, not about the length."""
    rule = "Do not seed more than one colourway per recipient."
    quote = "seed more than one colourway per recipient"
    assert len(quote) / len(rule) > core._MIN_RULE_QUOTE
    assert corrections.negated(rule) and not corrections.negated(quote)


def test_a_quote_of_a_rule_cannot_elide(conn):
    """"Never use AI imagery … approves it in writing" verifies against "Never use AI imagery
    unless the client approves it in writing." One sentence has no room for an ellipsis that
    is not changing what it says."""
    rule = "Never use AI imagery unless the client approves it in writing."
    _noted(conn, rule, markets=["LATAM", "APAC", "EMEA"])
    cid = corrections.find(conn, rule)["correction_id"]
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    with pytest.raises(ValueError) as e:
        _save(conn, {"correction_id": cid,
                     "quote": "Never use AI imagery … approves it in writing."})
    assert "elision" in str(e.value)


def test_a_fragment_of_a_rule_is_not_the_rule(conn):
    """The negation check catches the inversion; this catches the other half. "A content
    angle" is a faithful quotation of a rule about four things and is not the rule — and a
    guardrail breach citing it reads as though the rule said only that."""
    rule = "Every deliverable needs a date, a content angle, and a collab handle."
    _noted(conn, rule, markets=["LATAM", "APAC", "EMEA"])
    cid = corrections.find(conn, rule)["correction_id"]
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    assert not corrections.negated(rule), "not the negation check doing the work"
    with pytest.raises(ValueError) as e:
        _save(conn, {"correction_id": cid, "quote": "a content angle"})
    assert "fragment" in str(e.value)


def test_a_faithful_whole_quote_is_accepted(conn):
    """The bound has to let the legitimate citation through, or it is a refusal not a check."""
    cid = _standing(conn)
    saved = _save(conn, {"correction_id": cid, "quote": SEEDING})
    assert saved["evaluation_id"]


def test_a_correction_that_does_not_exist_cannot_be_cited(conn):
    with pytest.raises(ValueError) as e:
        _save(conn, {"correction_id": "corr_invented", "quote": SEEDING})
    assert "not a standing correction" in str(e.value)


def test_a_finding_is_still_anchored_to_exactly_one_thing(conn):
    """Two anchors are two findings — and with three slots the old two-slot check would have
    let `correction_id` ride alongside a `campaign_id` as decoration."""
    cid = _standing(conn)
    peru = _campaign(conn, "Peru precedent")
    with pytest.raises(ValueError) as e:
        _save(conn, {"correction_id": cid, "campaign_id": peru,
                     "quote": SEEDING})
    assert "one thing" in str(e.value)


def test_a_precedent_departure_still_cannot_cite_a_rule(conn):
    """The slot has to match the kind. A departure invites a rationale and a breach does not,
    and making them interchangeable collapses the distinction §6.2 exists to draw."""
    cid = _standing(conn)
    with pytest.raises(ValueError) as e:
        _save(conn, {"correction_id": cid, "quote": SEEDING},
              findings=[{"severity": "should_fix", "kind": "precedent_departure",
                         "departure": "regression", "finding": "Seeds four", "fix": "Seed one",
                         "precedent": {"correction_id": cid,
                                       "quote": SEEDING}}])
    assert "campaign_id" in str(e.value)


def test_the_model_is_told_the_rules_and_how_to_cite_them(conn):
    import core

    cid = _standing(conn)
    subject = _campaign(conn, "Colombia v1")
    note = core.prepare_evaluation(conn, subject_title="Colombia v1",
                                   proposal_text="A launch.", campaign_id=subject)["note"]

    assert cid in note
    assert "guardrail_breach" in note
    assert "correction_id" in note


def test_the_note_is_silent_when_nothing_is_standing(conn):
    import core

    subject = _campaign(conn, "Colombia v1")
    note = core.prepare_evaluation(conn, subject_title="Colombia v1",
                                   proposal_text="A launch.", campaign_id=subject)["note"]
    assert "standing_corrections" not in note


# ── what the reviews found ──────────────────────────────────────────────────

def test_the_same_rule_worded_three_ways_can_still_become_standing(conn):
    """The defect that made the whole item inert. Three markets independently stating one rule
    made THREE corrections, each stuck at one campaign in one market — so the gate was
    satisfiable only by the identical string arriving three times, which means one person
    copy-pasting: exactly the single-source case the two-market rule exists to reject."""
    said = [("Seed a single colourway across all recipients.", "LATAM"),
            ("Seeding boxes should carry one colourway.", "APAC"),
            ("Only one colourway per seeding box, please.", "EMEA")]
    first = None
    for text, market in said:
        noted = corrections.note(conn, text=text, provenance="slide 1",
                                 campaign_id=_campaign(conn, f"c-{market}", market=market))
        first = first or noted["correction_id"]
        if noted["correction_id"] != first:
            corrections.resolve(conn, noted["correction_id"], decision="same_rule",
                                same_as=first)

    gate = corrections.graduation(conn, first)
    assert (gate["campaigns"], gate["markets"]) == (3, 3)
    assert gate["eligible"] is True


def test_a_paraphrase_is_suggested_never_merged(conn):
    """§5.1 and §8.2 both settled this: SUGGEST, never auto-merge. Keeping only the refusal
    half is what made the loop inert; merging automatically would put a rule nobody agreed to
    on the checklist with somebody else's provenance attached."""
    corrections.note(conn, text=SEEDING, campaign_id=_campaign(conn, "Peru"),
                     provenance="Peru slide 12")
    noted = corrections.note(conn, text="Seeding boxes should carry one colourway.",
                             campaign_id=_campaign(conn, "Sydney", market="APAC"),
                             provenance="AU brief")

    assert noted["new_correction"]["looks_like"]["text"] == SEEDING
    assert noted["correction_id"] != corrections.find(conn, SEEDING)["correction_id"], \
        "suggested, not merged"
    assert {a["prefilled_args"]["decision"] for a in noted["new_correction"]["next_actions"]} \
        == {"same_rule", "different_rule", "set_aside"}


def test_a_rule_that_resembles_nothing_suggests_nothing(conn):
    corrections.note(conn, text=SEEDING, campaign_id=_campaign(conn, "Peru"),
                     provenance="Peru slide 12")
    noted = corrections.note(conn, text="State duration and placement for any billboard.",
                             campaign_id=_campaign(conn, "Mexico"), provenance="MX slide 9")
    assert noted["new_correction"]["looks_like"] is None


def test_a_prohibition_is_never_suggested_as_the_same_rule_as_its_permission(conn):
    """With `not`/`no`/`never` treated as noise words, "Do not use AI imagery" and "Use AI
    imagery only with approval" scored 1.0 — the suggester inverted on precisely the words
    that invert a rule, offering to merge a prohibition with its permission."""
    corrections.note(conn, text="Do not use AI imagery in any asset.",
                     campaign_id=_campaign(conn, "Peru"), provenance="Peru slide 4")
    noted = corrections.note(conn, text="Use AI imagery only with written approval.",
                             campaign_id=_campaign(conn, "Chile"), provenance="CL slide 4")

    assert noted["new_correction"]["looks_like"] is None


def test_punctuation_does_not_merge_two_different_numbers(conn):
    """Deleting punctuation rather than replacing it made "Post 3-4 times per week" and "Post
    34 times per week" the same rule — an automatic merge of two different instructions, which
    is the one thing exact matching is supposed to be too narrow to do."""
    first = corrections.note(conn, text="Post 3-4 times per week.",
                             campaign_id=_campaign(conn, "Peru"), provenance="p1")
    second = corrections.note(conn, text="Post 34 times per week.",
                              campaign_id=_campaign(conn, "Chile"), provenance="p2")

    assert second["correction_id"] != first["correction_id"]


def test_a_rule_that_merely_shares_a_word_suggests_nothing(conn):
    """The cutoff has to do work, not just the zero case. "Photograph the seeding boxes" and
    "seed a single colourway" are both about seeding boxes and are not the same rule — and a
    suggestion that fires on one shared word is the wrong alias §5.1 refused."""
    corrections.note(conn, text=SEEDING, campaign_id=_campaign(conn, "Peru"),
                     provenance="Peru slide 12")
    noted = corrections.note(conn, text="Photograph the seeding boxes before they ship.",
                             campaign_id=_campaign(conn, "Chile"), provenance="CL slide 3")

    resemblance = corrections._resembles(
        corrections._content(SEEDING),
        corrections._content("Photograph the seeding boxes before they ship."))
    assert 0 < resemblance < corrections._LOOKS_LIKE_CUTOFF, resemblance
    assert noted["new_correction"]["looks_like"] is None


def test_merging_moves_everywhere_it_was_said(conn):
    """Retrospective, exactly as §8.2's metric merge is: an answer that fixes the vocabulary
    while leaving the evidence behind has fixed nothing."""
    a = corrections.note(conn, text=SEEDING, campaign_id=_campaign(conn, "Peru"),
                         provenance="Peru slide 12")["correction_id"]
    b = corrections.note(conn, text="Seeding boxes should carry one colourway.",
                         campaign_id=_campaign(conn, "Sydney", market="APAC"),
                         provenance="AU brief")["correction_id"]

    corrections.resolve(conn, b, decision="same_rule", same_as=a)

    where = [s["provenance"] for s in corrections.sightings(conn, a)]
    assert where == ["Peru slide 12", "AU brief"]
    assert corrections.describe(conn, a)["times_seen"] == 2
    assert corrections.describe(conn, a)["markets"] == ["LATAM", "APAC"]


def test_a_later_mention_of_a_folded_wording_lands_on_the_live_rule(conn):
    """A merged row is kept because it is somebody's words, which means it is still reachable
    by its own wording — and the mention after the merge landed on the dead row. The rule it
    was folded into stayed where it was, so the review's own three-market scenario failed on
    the fourth mention, advancing nothing and asking nothing."""
    a = corrections.note(conn, text=SEEDING, campaign_id=_campaign(conn, "Peru"),
                         provenance="p1")["correction_id"]
    b = corrections.note(conn, text="Seeding boxes should carry one colourway.",
                         campaign_id=_campaign(conn, "Sydney", market="APAC"),
                         provenance="p2")["correction_id"]
    corrections.resolve(conn, b, decision="same_rule", same_as=a)

    third = corrections.note(conn, text="Seeding boxes should carry one colourway.",
                             campaign_id=_campaign(conn, "Dubai", market="EMEA"),
                             provenance="p3")

    assert third["correction_id"] == a
    gate = corrections.graduation(conn, a)
    assert (gate["campaigns"], gate["markets"]) == (3, 3)
    assert gate["eligible"] is True


def test_two_rules_cannot_be_folded_into_each_other(conn):
    """A→B then B→A leaves both rows `merged`, no live row at all, and the rule gone from
    every reader with its sightings intact and unreachable."""
    a = corrections.note(conn, text=SEEDING, campaign_id=_campaign(conn, "Peru"),
                         provenance="p1")["correction_id"]
    b = corrections.note(conn, text="Seeding boxes should carry one colourway.",
                         campaign_id=_campaign(conn, "Sydney", market="APAC"),
                         provenance="p2")["correction_id"]
    corrections.resolve(conn, b, decision="same_rule", same_as=a)

    with pytest.raises(ValueError) as e:
        corrections.resolve(conn, a, decision="same_rule", same_as=b)
    assert "already one" in str(e.value)
    assert corrections.describe(conn, a)["status"] == "provisional"


def test_folding_into_an_already_folded_rule_reaches_the_live_one(conn):
    """C merged into an already-merged B put C's sightings on a row nothing reads."""
    a = corrections.note(conn, text=SEEDING, campaign_id=_campaign(conn, "Peru"),
                         provenance="p1")["correction_id"]
    b = corrections.note(conn, text="Seeding boxes should carry one colourway.",
                         campaign_id=_campaign(conn, "Sydney", market="APAC"),
                         provenance="p2")["correction_id"]
    c = corrections.note(conn, text="Only one colourway per seeding box, please.",
                         campaign_id=_campaign(conn, "Dubai", market="EMEA"),
                         provenance="p3")["correction_id"]
    corrections.resolve(conn, b, decision="same_rule", same_as=a)
    result = corrections.resolve(conn, c, decision="same_rule", same_as=b)

    assert result["correction_id"] == a
    assert corrections.describe(conn, a)["times_seen"] == 3


def test_a_rule_cannot_be_folded_into_one_somebody_set_aside(conn):
    a = corrections.note(conn, text=SEEDING, campaign_id=_campaign(conn, "Peru"),
                         provenance="p1")["correction_id"]
    corrections.set_aside(conn, a)
    b = corrections.note(conn, text="Seeding boxes should carry one colourway.",
                         campaign_id=_campaign(conn, "Sydney", market="APAC"),
                         provenance="p2")["correction_id"]

    with pytest.raises(ValueError) as e:
        corrections.resolve(conn, b, decision="same_rule", same_as=a)
    assert "set aside" in str(e.value)


def test_a_merged_rule_is_kept_not_deleted(conn):
    """§8.2's metric merge DELETES the absorbed row. Here the row is somebody's words."""
    a = corrections.note(conn, text=SEEDING, campaign_id=_campaign(conn, "Peru"),
                         provenance="Peru slide 12")["correction_id"]
    b = corrections.note(conn, text="Seeding boxes should carry one colourway.",
                         campaign_id=_campaign(conn, "Sydney", market="APAC"),
                         provenance="AU brief")["correction_id"]
    corrections.resolve(conn, b, decision="same_rule", same_as=a)

    absorbed = corrections.describe(conn, b)
    assert absorbed["text"] == "Seeding boxes should carry one colourway."
    assert absorbed["status"] == "merged"
    assert absorbed["merged_into"] == a


def test_answering_cannot_undo_a_standing_rule(conn):
    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    cid = corrections.find(conn, SEEDING)["correction_id"]
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    for decision, extra in (("different_rule", {}), ("set_aside", {}),
                            ("same_rule", {"same_as": cid})):
        with pytest.raises(ValueError) as e:
            corrections.resolve(conn, cid, decision=decision, **extra)
        assert "already a standing correction" in str(e.value)


def test_a_standing_rule_can_be_set_aside(conn):
    """The inverse `graduate` had none. A correction promoted in error is a blocking finding on
    every brief in its markets, and a guardrail_breach cannot be softened to a note — so
    without this there was no way back except waiting for it to fall out of use."""
    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    cid = corrections.find(conn, SEEDING)["correction_id"]
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    corrections.set_aside(conn, cid, why="It was never really their rule.")

    assert corrections.describe(conn, cid)["status"] == "ignored"
    assert corrections.describe(conn, cid)["expected_in"] == ["APAC", "EMEA", "LATAM"], \
        "where it used to apply is part of its record"
    assert corrections.standing_for(conn, _campaign(conn, "Colombia"))["standing"] == []
    assert corrections.sightings(conn, cid), "what was said is kept"


def test_a_campaign_id_that_does_not_exist_is_refused_before_anything_is_written(conn):
    """The sighting carries a foreign key and the correction row is committed first, so an id
    that does not exist raised IntegrityError — not a ValueError, so the caller saw "Error
    executing tool" with the reason discarded — and left a correction with no sightings and no
    provenance behind. That is an opinion in a text field, which is what this module refuses.
    The model supplying the id is now the ordinary path, because `after_upload` prefills it."""
    with pytest.raises(ValueError) as e:
        corrections.note(conn, text=SEEDING, campaign_id="camp_invented",
                         provenance="somewhere")
    assert "not a record in this library" in str(e.value)
    assert corrections.all_of_them(conn) == [], "nothing half-written"


def test_a_rule_longer_than_a_rule_is_refused(conn):
    """Every other model-authored field in this codebase is measured, and this one travels into
    the evidence package and then into saved findings."""
    with pytest.raises(ValueError) as e:
        corrections.note(conn, text="Seed one colourway. " * 40,
                         campaign_id=_campaign(conn, "Peru"), provenance="p")
    assert "record them separately" in str(e.value)


def test_what_each_market_actually_said_is_kept(conn):
    """After a merge the canonical row carries one wording and the sightings carry the others —
    §8.1's canonical/raw_key split, and what makes the fold checkable rather than a claim
    nobody can audit."""
    a = corrections.note(conn, text=SEEDING, campaign_id=_campaign(conn, "Peru"),
                         provenance="p1")["correction_id"]
    b = corrections.note(conn, text="Seeding boxes should carry one colourway.",
                         campaign_id=_campaign(conn, "Sydney", market="APAC"),
                         provenance="p2")["correction_id"]
    corrections.resolve(conn, b, decision="same_rule", same_as=a)

    assert [s["said_as"] for s in corrections.sightings(conn, a)] == \
        [SEEDING, "Seeding boxes should carry one colourway."]


def test_a_rule_set_aside_by_mistake_can_be_reopened(conn):
    """Setting aside is a decision, and decisions are sometimes wrong. §8.2's
    `different_measure` reopens an ignored measure; a correction with no way back at all was
    the drift."""
    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    cid = corrections.find(conn, SEEDING)["correction_id"]
    corrections.graduate(conn, cid, confirmed_by="R. Vega")
    corrections.set_aside(conn, cid)

    corrections.reopen(conn, cid)

    assert corrections.describe(conn, cid)["status"] == "provisional", \
        "not straight back to standing — that is the gate's question and a person's answer"
    assert corrections.standing_for(conn, _campaign(conn, "Next"))["standing"] == []
    assert corrections.graduation(conn, cid)["eligible"] is True


def test_a_mention_with_no_campaign_says_it_counts_towards_nothing(conn):
    """`times_seen` was the only number on this surface and it is not the number the gate
    reads: five mentions with no campaign showed "times_seen: 5" while the gate said "seen in
    0 campaigns". §8.3 wrote a paragraph on exactly this distinction."""
    noted = corrections.note(conn, text=SEEDING, campaign_id=None, provenance="somewhere")

    assert noted["times_seen"] == 1
    assert noted["campaigns"] == 0
    assert "counts towards nothing" in noted["what_it_means"]


def test_the_upload_offers_to_record_what_the_client_said(conn):
    """`note_correction` was referenced nowhere in the product but its own definition — a tool
    the model would have to know existed and spontaneously call. That is §8.3's unreachable
    human step one stage earlier and strictly worse: nothing would ever be RECORDED for the
    gate to act on. §2.5 already ingests tracked client comments, with provenance assembled."""
    import actions

    offers = [o for o in actions.after_upload(
        campaign_id="camp_1", status="planned", has_metrics=False, title="Colombia v1",
        commentary=[{"kind": "comment", "author": "R. Vega", "anchor": "slide 6",
                     "text": "We want the seeding box to carry one colourway."},
                    {"kind": "speaker_note", "author": "Us", "anchor": "slide 2",
                     "text": "Remember to pause here."}])
        if o["tool"] == "note_correction"]
    assert len(offers) == 1, "a speaker note is the agency talking to itself, not the client"
    assert offers[0]["prefilled_args"]["campaign_id"] == "camp_1"
    assert "R. Vega" in offers[0]["prefilled_args"]["provenance"]
    assert "slide 6" in offers[0]["prefilled_args"]["provenance"]
    assert any("own words" in n for n in offers[0]["needs"])


def test_the_upload_offer_is_one_the_tool_accepts(conn):
    """§5.2's standing check, and the mirror of the failure `after_upload`'s docstring already
    guards: offering a call that cannot be made. This module had no test for the other
    direction — a call that can be made and is never offered."""
    import inspect

    import actions
    import mcp_server

    offer = next(o for o in actions.after_upload(
        campaign_id="camp_1", status="planned", has_metrics=False, title="Colombia v1",
        commentary=[{"kind": "comment", "author": "R. Vega", "anchor": "slide 6",
                     "text": "One colourway per box."}]) if o["tool"] == "note_correction")
    signature = inspect.signature(getattr(mcp_server, offer["tool"]))

    assert set(offer["prefilled_args"]) <= set(signature.parameters)
    for need in offer.get("needs", []):
        assert need.split()[0].strip(" —:") in signature.parameters


def test_a_deck_with_no_client_comments_offers_nothing(conn):
    import core

    up = core.ingest_campaign(conn, title="Colombia v1", market="LATAM", detail="A launch.")
    assert [o for o in up["next_actions"] if o["tool"] == "note_correction"] == []


def test_a_proposal_that_is_not_a_record_still_sees_the_rules(conn):
    """"Judge this new pitch" is the flow the review cares about most, and the caller NAMED the
    market. Returning nothing left the client's own standing rules out of the one judgment they
    most obviously apply to."""
    import core

    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    corrections.graduate(conn, corrections.find(conn, SEEDING)["correction_id"],
                         confirmed_by="R. Vega")

    package = core.prepare_evaluation(conn, subject_title="A new pitch",
                                      proposal_text="Seeds four colourways.", market="LATAM")
    assert [c["text"] for c in package["standing_corrections"]["standing"]] == [SEEDING]


def test_a_proposal_with_no_market_given_is_told_so(conn):
    import core

    _noted(conn, SEEDING, markets=["LATAM", "APAC", "EMEA"])
    corrections.graduate(conn, corrections.find(conn, SEEDING)["correction_id"],
                         confirmed_by="R. Vega")

    package = core.prepare_evaluation(conn, subject_title="A new pitch",
                                      proposal_text="Seeds four colourways.")
    block = package["standing_corrections"]
    assert block["standing"] == []
    assert block["status"] == "nothing_to_check"
    assert block["code"] == "no_market"


def test_a_learned_rule_is_not_called_a_rule_they_wrote(conn):
    """Nobody wrote a standing correction. The library inferred it from repetition and one
    person confirmed it — stating that back to the customer as their own authored rule is this
    review's central failure, arriving in the voicing rather than in a finding."""
    cid = _standing(conn)
    saved = _save(conn, {"correction_id": cid, "quote": SEEDING})

    said = saved["note"]
    assert "a rule they wrote was broken" not in said, \
        "nobody wrote a standing correction"
    assert "learned from their own repeated feedback" in said
    assert "somebody confirmed" in said


def test_the_server_still_argues_with_a_verdict_resting_on_a_learned_rule(conn):
    """The exemption is right for a rule the customer WROTE and can edit. It is backwards for
    one the library inferred: a past campaign that did the opposite and did well is the only
    evidence that could ever catch a wrong inference, and suppressing the search switches off
    the one check watching it."""
    cid = _standing(conn)
    saved = _save(conn, {"correction_id": cid, "quote": SEEDING})

    assert saved["disconfirming"]["code"] != "verdict_rests_on_a_rule"


def test_the_provenance_the_model_is_shown_is_bounded(conn):
    """Every other model-facing field here is bounded. A rule stated in twenty briefs carried
    twenty provenances into the evidence package and then into the stored judgment."""
    for n in range(12):
        corrections.note(conn, text=SEEDING, provenance=f"Deck {n} slide {n}",
                         campaign_id=_campaign(conn, f"c{n}", market=["LATAM", "APAC"][n % 2]))
    cid = corrections.find(conn, SEEDING)["correction_id"]
    corrections.graduate(conn, cid, confirmed_by="R. Vega")

    shown = corrections.standing_for(conn, _campaign(conn, "Next"))["standing"][0]
    assert shown["provenance"].count(";") == corrections._MAX_PROVENANCE - 1
    assert "Deck 11" not in shown["provenance"], "truncated, not merely annotated"
    assert "and 8 more" in shown["provenance"]


def test_the_note_spells_out_only_a_few_rules(conn):
    """The example overlay ships ten. Ten rules rendered into every note — each expressible
    only as a blocking breach, against a 12-finding cap — is the checklist that fires on
    everything."""
    import core

    for n in range(8):
        text = f"Standing rule number {n} about the deck layout."
        for market in ("LATAM", "APAC", "EMEA"):
            corrections.note(conn, text=text, provenance=f"slide {n}",
                             campaign_id=_campaign(conn, f"c{n}{market}", market=market))
        corrections.graduate(conn, corrections.find(conn, text)["correction_id"],
                             confirmed_by="R. Vega")

    package = core.prepare_evaluation(conn, subject_title="X", proposal_text="A launch.",
                                      campaign_id=_campaign(conn, "Subject"))
    assert len(package["standing_corrections"]["standing"]) == 8, "the data is all there"
    assert package["note"].count("Standing rule number") == core._MAX_STANDING_SHOWN
    assert "3 more are in `standing_corrections`" in package["note"]


def test_a_bad_status_filter_is_refused_with_the_valid_set(conn):
    import asyncio
    import json

    import mcp_server

    out = json.loads(asyncio.run(
        mcp_server.mcp.call_tool("list_corrections", {"status": "expectd"})).content[0].text)
    assert out.get("error"), "a typo silently returned an empty list"
    assert "expected" in str(out)


def test_the_product_ships_no_corrections_of_its_own(conn):
    """The ten are one customer's rules. §12.3 ships them as an example overlay, and seeding
    them into the product is what this project has refused since §5.1."""
    assert corrections.all_of_them(conn) == []


def test_it_reaches_the_model_over_the_protocol(conn):
    import asyncio
    import json

    import mcp_server

    async def call(name, args):
        return json.loads((await mcp_server.mcp.call_tool(name, args)).content[0].text)

    _noted(conn, SEEDING, markets=["LATAM", "APAC"])
    third = _campaign(conn, "Dubai", market="EMEA")
    noted = asyncio.run(call("note_correction", {
        "text": SEEDING, "campaign_id": third, "provenance": "UAE brief"}))
    cid = noted["correction_id"]
    assert noted["newly_eligible"]

    promoted = asyncio.run(call("graduate_correction",
                                {"correction_id": cid, "confirmed_by": "R. Vega"}))
    assert promoted["status"] == "expected"

    subject = _campaign(conn, "Colombia v1")
    package = asyncio.run(call("prepare_evaluation",
                               {"subject_title": "Colombia v1", "proposal_text": "A launch.",
                                "campaign_id": subject}))
    assert package["standing_corrections"]["standing"][0]["text"] == SEEDING
