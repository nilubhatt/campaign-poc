"""
D94 — the fact checks are English-only, and report "absent" where they mean "unchecked".

    "The fact checks are English-only and regex-based: a brief in Spanish reports no channels,
     no budget and no creators. The library already holds LATAM and SEA campaigns whose decks
     may not be in English, and every check would report absent rather than unchecked — the
     exact 'unknown reported as a finding' this item is careful about elsewhere. Needs a
     language signal before it can be honest."

This is the product's own central failure, in the one place it had not been looked for. §7.1
made the checks computed so a model would stop re-deriving them and treat them as established
— and §9.6 spent a whole round on the difference between "we looked and found nothing" and
"we could not look". `absent` on a Spanish brief is the second wearing the first's clothes,
with the server's authority attached, feeding a `missing_information` finding that tells a
marketer their deck has no budget when page four says `Presupuesto: 45.000 €`.

The honest fix is not to teach the regexes Spanish. It is to know when they do not apply,
and say `unchecked` — a word this product already has, and already distinguishes from a pass.
"""
import pytest

import core
import facts


SPANISH = """
Lanzamiento de tienda en Bogotá — Calle 82.
Presupuesto: 45.000 € para el trimestre.
El equipo trabajará con creadores locales durante seis semanas, con publicaciones
en redes sociales y en televisión. La campaña empieza el 3 de marzo de 2026.
"""

ENGLISH = """
Store launch in Bogota — Calle 82.
Budget: EUR 45,000 for the quarter.
The team will work with local creators over six weeks, posting on social and on TV.
The campaign starts on 3 March 2026.
"""


def test_english_is_recognised_as_english():
    assert facts.language_of(ENGLISH)["language"] == "en"
    assert facts.language_of(ENGLISH)["basis"] == "heuristic"


def test_a_spanish_brief_is_not_reported_as_english():
    """The signal the row says is needed before any of this can be honest."""
    said = facts.language_of(SPANISH)

    assert said["language"] != "en"
    assert said["checks_apply"] is False


def test_the_signal_is_a_heuristic_and_says_so():
    """It is word-list matching, not a language model. A confident `language: es` would be a
    computed-looking claim resting on counting stopwords — and this product's whole complaint
    is confident claims that cannot be argued with. `heuristic` is the honest basis, and D61
    already established that a threshold deciding something is not a judgment."""
    said = facts.language_of(SPANISH)

    assert said["basis"] == "heuristic"
    assert said["what_it_means"]


def test_a_brief_too_short_to_tell_is_read_normally():
    """Three words is not a language sample — and the absence of a sample is not evidence of
    another language. Suppressing negatives here would turn every short brief into a shrug:
    "Store launch." genuinely carries no budget, and this product's users write English.

    What suppresses a negative is EVIDENCE — another language clearly ahead, or enough text
    to expect English words with none appearing. Not the absence of either."""
    said = facts.language_of("Store launch.")

    assert said["language"] == "unknown"
    assert said["reads_as_english"] is True
    assert facts.compute("Store launch.")["budget"]["status"] == "absent"


# ── what the checks do about it ───────────────────────────────────────────

def test_a_spanish_brief_is_not_told_it_has_no_budget():
    """The defect, exactly: `Presupuesto: 45.000 EUR` is a budget, and the library told a
    marketer their deck had none — with the server's authority attached, which is worse than
    a model guessing it.

    It now FINDS it, which is better than declaring it unchecked: the amount was always
    language-neutral and the English word beside it was the whole of the problem. What stays
    `unchecked` is the check that genuinely could not read the text."""
    computed = facts.compute(SPANISH)

    assert computed["budget"]["status"] == "present"
    assert computed["channels"]["status"] == "unchecked"


def test_an_english_brief_is_still_checked():
    """The silence half, and the one that matters: a language signal that suppressed the
    checks too eagerly would turn every real finding into an `unchecked`, which is the same
    loss from the other direction."""
    computed = facts.compute(ENGLISH)

    assert computed["budget"]["status"] == "present"
    # RAN, rather than any particular verdict: `channels` is `partial` for almost every brief
    # by design, and asserting `present` would be testing the channel regex rather than the
    # language gate.
    assert computed["channels"]["status"] != "unchecked"


def test_unchecked_says_why_and_what_would_fix_it():
    """`nothing_to_check` is never a pass — and neither is this. A reader has to be able to
    tell "your deck lists no channels" from "we cannot read your deck"."""
    said = facts.compute(SPANISH)["channels"]

    assert said["status"] == "unchecked"
    assert "English" in said["what_it_means"]
    assert said["basis"] == "computed"


def test_the_language_is_reported_beside_the_checks():
    """A reader who cannot see WHY everything says `unchecked` will read it as the product
    being broken. The signal that suppressed them has to travel with them."""
    computed = facts.compute(SPANISH)

    assert computed["language"]["language"] != "en"
    assert computed["language"]["checks_apply"] is False


# ── it has to reach the model, or the finding comes back anyway ───────────

def test_the_model_is_told_the_checks_did_not_run(conn):
    """§7.1's own lesson: the model is told to treat computed facts as established and not
    re-derive them. Handed five `unchecked` with no sentence, it will do what it did before —
    report the brief as missing a budget, which is the finding this exists to stop."""
    prepared = core.prepare_evaluation(conn, subject_title="Bogota launch",
                                       proposal_text=SPANISH)

    assert "not in English" in prepared["note"]
    assert "unchecked" in prepared["note"]


def test_an_english_brief_gets_no_such_paragraph(conn):
    prepared = core.prepare_evaluation(conn, subject_title="Bogota launch",
                                       proposal_text=ENGLISH)

    assert "not in English" not in prepared["note"]


def test_a_stored_spanish_record_is_not_reported_as_missing_things(conn):
    """`for_campaign` is what `gaps()` and the market-wide field checks read. Left alone, a
    LATAM library of Spanish decks would report "no campaign in Colombia has ever recorded a
    budget" — a gap the customer cannot close because it is not true."""
    cid = core.ingest_campaign(conn, title="Bogota launch", market="Colombia",
                               status="concluded", detail=SPANISH)["campaign_id"]

    computed = facts.for_campaign(conn, cid)

    assert computed["channels"]["status"] == "unchecked"
    assert computed["budget"]["status"] == "present", (
        "and what CAN be read is read: the customer is not told their deck is empty"
    )


def test_a_market_of_spanish_decks_is_not_a_field_gap(conn):
    """The consequence in the surface that would have shipped it to a customer: §5.3's
    `field_never_recorded` says "no campaign in X has ever recorded a budget", and an
    `unchecked` must not count as an `absent` — reporting a gap nobody can close is the
    permanent complaint this product refuses everywhere else."""
    for n in range(2):
        core.ingest_campaign(conn, title=f"Bogota {n}", market="Colombia",
                             status="concluded", detail=SPANISH)

    codes = [(g["code"], g.get("field")) for g in core.gaps(conn)["gaps"]]

    assert ("field_never_recorded", "budget") not in codes


def test_english_carrying_foreign_words_is_still_english():
    """`_CLEARLY_AHEAD` is what stops a stray word deciding. English marketing copy is full of
    them — place names, brand names, a market spelled the local way — and a rule that fired on
    any non-English stopword would suppress every check on a perfectly readable brief.

    Mutation found this: relaxing the threshold to "any foreign word at all" left every test
    green, because none of them used English that mentions Spain."""
    english = ("Store launch for the Rio de Janeiro and Los Angeles markets. The team will "
               "work with local creators over six weeks, and the budget is EUR 45,000 for "
               "the quarter with posting on social and on television.")

    said = facts.language_of(english)

    assert said["language"] == "en", said["counts"]
    assert facts.compute(english)["budget"]["status"] == "present", (
        "and the checks run, which is the thing the threshold protects"
    )


def test_a_narrow_lead_is_not_a_language():
    """A deck is a mix: a Spanish brief carries English channel names and an English one
    carries a market name. A one-word lead is noise, and treating it as a language means the
    checks stand down on evidence nobody would call evidence."""
    mixed = ("The de facto launch plan for the quarter: work with local creators over six "
             "weeks, with a budget of EUR 45,000 and posting on social and television.")

    assert facts.language_of(mixed)["language"] == "en"


def test_a_clearly_spanish_brief_still_trips_it():
    """The other side, or the threshold is just "never fire". The gap between the two has to
    be wide enough for real copy to land on both sides of it."""
    assert facts.language_of(SPANISH)["checks_apply"] is False


def test_portuguese_and_spanish_are_not_confused_for_english():
    """They share most of their stopwords, so the WINNER between them is unreliable — and it
    does not need to be reliable. What matters is "not English", because that is the only
    thing the checks act on. The reported language is a label on a heuristic, and the code
    says so."""
    portuguese = ("Lançamento de loja em São Paulo. O orçamento é de 45.000 EUR para o "
                  "trimestre, com criadores locais durante seis semanas e publicações nas "
                  "redes sociais e na televisão.")

    said = facts.language_of(portuguese)

    assert said["checks_apply"] is False
    assert said["language"] in ("pt", "es"), (
        "either is fine — the checks act on 'not English', not on which one"
    )
    assert said["basis"] == "heuristic"


# ── found by review: the heuristic misfires both ways ─────────────────────

BULLETS_EN = """Store launch Q3
- Los Angeles flagship
- Las Vegas popup
- Budget: EUR 45,000 for the quarter
- Six weeks, posting on social and TV
"""

BULLETS_ES = """Lanzamiento Q3
- Bogota flagship
- Presupuesto: 45.000 EUR para el trimestre
- Seis semanas
"""

VIETNAMESE = ("Ra mat cua hang tai Ho Chi Minh. Ngan sach 45.000 EUR cho quy nay voi cac "
              "nha sang tao dia phuong trong sau tuan.")


def test_an_english_bullet_deck_is_still_checked():
    """A bullet deck has almost no stopwords — and it is the commonest deck shape there is.
    Two market names ("Los Angeles", "Las Vegas") outvoted zero English words and suppressed
    every check on a perfectly readable English brief."""
    computed = facts.compute(BULLETS_EN)

    assert computed["budget"]["status"] == "present", computed["language"]["counts"]


def test_a_spanish_bullet_deck_is_not_reported_as_empty():
    """D94's own headline example, in the shape that defeated the stopword signal: no Spanish
    stopwords either, so the language read as `unknown`, the checks ran, and
    `Presupuesto: 45.000 EUR` was reported as no budget at all."""
    said = facts.compute(BULLETS_ES)["budget"]

    assert said["status"] != "absent", said
    assert said["status"] == "present", (
        "and it should be FOUND: the amount is language-neutral and the word is now known"
    )


def test_a_language_with_no_stopword_list_does_not_read_as_english():
    """Vietnamese and Thai are SEA markets D94 names explicitly, and neither has a stopword
    list here. With none, the counts are all zero, the text read as English, and every check
    reported `absent` — the exact defect, for the markets the row was written about."""
    computed = facts.compute(VIETNAMESE)

    assert computed["channels"]["status"] == "unchecked"


def test_finding_something_is_trusted_whatever_the_language():
    """The rule that makes the signal safe to be wrong: a check that FOUND its evidence has
    read the text, whatever a stopword count guessed. Only a NEGATIVE depends on knowing we
    could read it — which is the asymmetry that was missing."""
    said = facts.compute(BULLETS_ES)

    assert said["language"]["language"] != "en" or said["language"]["counts"] == {}
    assert said["budget"]["status"] == "present"


def test_a_negative_is_only_reported_where_the_text_could_be_read():
    """The other half. "No budget appears in this brief" is a finding; it is only true if the
    brief could be read, and saying it about a Vietnamese deck is the unknown-as-a-finding
    failure this product exists to refuse."""
    no_budget_en = ("Store launch in Bogota over six weeks, with the team posting on social "
                    "and on television, and creators briefed for the quarter.")

    assert facts.compute(no_budget_en)["budget"]["status"] == "absent"
    assert facts.compute(VIETNAMESE)["budget"]["status"] in ("present", "unchecked")


def test_one_unreadable_deck_does_not_delete_a_real_gap(conn):
    """§5.3's `field_never_recorded` says "no campaign in X has ever recorded a budget", and
    it required EVERY record to read `absent`. The deck here is Vietnamese — a language with
    no keyword list, so the check genuinely cannot read it, which is the case that matters. An `unchecked` is neither `absent` nor
    `not_applicable`, so a single non-English deck falsified the `all()` and the gap vanished
    — the customer had been told about it for weeks, uploads one Spanish deck, and the
    product stops saying it with nothing to explain why.

    That is D94's own defect with the sign flipped: §7.1 established that an unknown must not
    be REPORTED as a finding, and this let an unknown silently CANCEL one that was real."""
    for n in range(3):
        core.ingest_campaign(conn, title=f"Lima {n}", market="Peru", status="concluded",
                             detail=("Store launch in Lima over six weeks, with the team "
                                     "posting on social and television for the quarter."))
    before = {(g["code"], g.get("field")) for g in core.gaps(conn)["gaps"]}
    assert ("field_never_recorded", "budget") in before, before

    core.ingest_campaign(conn, title="Ho Chi Minh", market="Peru", status="concluded",
                         detail=VIETNAMESE)

    after = {(g["code"], g.get("field")) for g in core.gaps(conn)["gaps"]}
    assert ("field_never_recorded", "budget") in after, (
        "three English decks still have no budget; one unreadable deck does not change that"
    )


def test_the_gap_says_how_many_records_it_could_actually_read(conn):
    """And it has to say so, or the count is a claim about the whole market when it is a claim
    about the part of it this library can read."""
    for n in range(3):
        core.ingest_campaign(conn, title=f"Lima {n}", market="Peru", status="concluded",
                             detail=("Store launch in Lima over six weeks, with the team "
                                     "posting on social and television for the quarter."))
    core.ingest_campaign(conn, title="Ho Chi Minh", market="Peru", status="concluded",
                         detail=VIETNAMESE)

    gap = next(g for g in core.gaps(conn)["gaps"]
               if g["code"] == "field_never_recorded" and g["field"] == "budget")

    assert gap["campaigns"] == 3
    assert gap["unreadable"] == 1
    assert "could not be read" in gap["what"]
