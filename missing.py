"""
What the library is missing, and answering back about it (§5.3 idea C, §10.2).

Two sections lifted out of `core` unchanged: the gap surface — what is absent, what it costs,
and an offer that would close it — and the machinery by which somebody sets a gap aside, says a
finding is settled, or reopens either.

They are one cluster: the answers decide what the ranking shows, and the ranking decides what
there is to answer. Between them they reach back into the rest of `core` for six things, and
`core` calls back into ten of the names here — so both directions are explicit rather than
tangled, which is the point of the split.

`core` re-exports every name below, so `core.gaps`, `core.answer_gap` and the private helpers
tests already reach for keep working. `core` is imported INSIDE the functions that need it,
because `core` imports this module.
"""
from __future__ import annotations

from creative import _execution_note
from creative import _snapshot_execution_drift
from creative import compare_execution
from diffing import diff_campaigns
from typing import Optional
import re
import time

import actions
import corrections
import drift
import facts
import identity
import images
import learning
import metrics
import rulebook
import store
import version

# ── what is missing (§5.3, idea C) ──────────────────────────────────────────
#
# "The library knows it holds one campaign with real outcome data. It knows no LATAM store
# launch has ever carried a budget. It knows the KPI workbook named in its own rubric has
# never been supplied. It says none of this unless directly interrogated."
#
# Ranked, lowest number first, and the ranking is the design. An unordered list of everything
# absent is the thing nobody reads — which is the state being replaced. The order is by how
# much closing the gap would change what this library can answer:
#
#   1  it is empty, so nothing else is worth saying
#   2  nothing has measured outcomes, so no judgment rests on evidence
#   3  a whole market has none, so judgments about that market rest on nothing local
#   4  records are stored but not searchable, so the evidence exists and cannot be found
#   5  decks whose commentary was never read — real content, never ingested
#   6  campaigns judged on a brief with nothing showing what actually ran (§9.1)
# How many names a gap may list before it is a paragraph rather than a sentence.
_MAX_NAMED = 5

_GAP_RANK = {
    "library_is_empty": 1,
    "few_verified_outcomes": 2,
    "market_without_outcomes": 3,
    "partly_indexed": 4,
    # §9.1's state: "a library that learns from briefs while measuring executions is learning
    # from the wrong document, and has no way to notice." Ranked below the others because a
    # brief with no measured outcome is a bigger hole than a measured outcome nobody has
    # checked against what ran — but above nothing, because it is the difference between
    # "this campaign worked" and "something worked, and we do not know if it was this".
    "execution_never_checked": 5,
    # Below execution drift: not knowing what ran is worse than not knowing what else was
    # going on while it ran, and this one is cheap to close (two dates) where that one needs
    # photographs.
    "no_window": 4,
    # Below the gaps about MISSING data, because this one is about data that is all present
    # and simply has not been looked at — and the looking is a few minutes rather than a
    # workbook somebody has to go and find.
    "judgments_never_reconciled": 3,
    # §12.4/D51. LOWER than every gap about missing evidence (the number is a position, and a
    # smaller one ranks higher): the deck is usually already on file and the remedy is one
    # call, where `few_verified_outcomes` is the hole this product was built around. Above
    # nothing, though — a client's recorded objection is often the most useful precedent the
    # library could have, and these records have none. Recorded since §2.5 and NOT reported
    # until D39 gave it an action that did not create a duplicate record.
    "commentary_never_read": 6,
}


# Which computed facts are worth reporting as never-recorded across a whole market (D49).
# Not every check: `date_consistency` has no "never" reading, and `channels` is partial by
# design for almost every brief.
_FIELD_GAPS = ("budget", "date_coverage", "engagement_rate")
# Below this a market is not a pattern. One record missing a budget is a fact about that
# record; calling it a coverage gap would turn every new market into a complaint on the day
# it is added.
_MIN_FOR_A_PATTERN = 2


def _fields_never_recorded(conn, campaigns: list) -> list:
    """Markets where no record has ever carried a field (D49, §7.1).

    "No LATAM store launch has ever carried a budget" is one of the review's own three
    examples of a gap, and the only one about an absent FIELD inside records rather than an
    absent record. It needed §7.1 first, because until then nothing could detect a budget.

    `never` is the claim, so one record carrying the field makes it false — reporting the
    market anyway would be the overstatement this whole surface is written against.
    """
    by_market: dict = {}
    for campaign in campaigns:
        by_market.setdefault(campaign.get("market") or campaign.get("region"),
                             []).append(campaign)
    found = []
    for market, rows in sorted(by_market.items(), key=lambda kv: kv[0] or ""):
        if not market or len(rows) < _MIN_FOR_A_PATTERN:
            continue
        computed = [facts.for_campaign(conn, c["id"]) for c in rows]
        for field in _FIELD_GAPS:
            statuses = [c.get(field, {}).get("status") for c in computed]
            # "not_applicable" is not a miss — a brief with no creators is not missing their
            # engagement rates, and counting it would make the gap unclosable.
            #
            # D94: neither is "unchecked", and for a sharper reason. It is not evidence the
            # field is absent, so it cannot support the claim; and it is not evidence the
            # field is present, so it must not REFUTE the claim either. Left in, one
            # non-English deck falsified the `all()` and the whole gap disappeared — the
            # customer had been told about it for weeks, uploaded one Spanish deck, and the
            # product stopped saying it with nothing to explain why. That is this phase's own
            # defect with the sign flipped: an unknown must not be reported as a finding, and
            # it must not silently cancel one that is real.
            unreadable = sum(1 for s in statuses if s == "unchecked")
            relevant = [s for s in statuses
                        if s not in ("not_applicable", "unchecked")]
            if relevant and all(s == "absent" for s in relevant):
                found.append({
                    "code": "field_never_recorded",
                    "field": field,
                    "market": market,
                    "campaigns": len(relevant),
                    # Said, not hidden: the count is a claim about the records this library
                    # could READ, and a reader who is not told that will take it for a claim
                    # about the market.
                    "unreadable": unreadable,
                    "what": f"No campaign in {market} has ever recorded a "
                            f"{field.replace('_', ' ')} "
                            f"({len(relevant)} record(s) checked, all missing it"
                            + (f"; {unreadable} more could not be read — not in English"
                               if unreadable else "") + ").",
                    "why_it_matters": "A field missing from every record in a market is a gap "
                                      "no single upload reveals — every judgment there is "
                                      "made without it and nothing says so.",
                    "basis": "computed",
                })
    return found


def _expected_inputs_never_supplied(conn) -> list:
    """Inputs the rulebook declares a brief must carry, that nothing on file carries (D50).

    Reported per LIBRARY rather than per record, like the other field gaps beside it: "no
    brief has ever carried a KPI workbook" is the finding, and firing it once per campaign
    would bury every other gap under one fact repeated.

    `looks_like` is what makes it closable. Without the words that would show the input HAD
    arrived, the gap can only be reported forever — a gap nobody can close is one everybody
    learns to ignore, which is §8.2's own lesson about re-asking a declined question.
    """
    import core

    import rulebook

    try:
        declared = rulebook.expects()
    except ValueError:
        # A broken rulebook is `health_check`'s finding, loudly. It is not this function's,
        # and raising here would take out `gaps()` — a report about the library — over a
        # configuration file.
        return []
    if not declared:
        return []

    superseded = store.get_superseded_campaign_ids(conn)
    records = [c for c in store.list_campaigns(conn)
               if c.get("record_type") != "reference" and c["id"] not in superseded]
    if not records:
        return []

    # §13.3: read each record ONCE, and fold its body once. This sat inside
    # `for expected: for record:` — a full read of every record per expectation, and a
    # re-join and re-lowercase of every body on each pass. That is D79's shape at a worse
    # exponent, in the same file as the memo that fixes it, and §13.3 named five rows and
    # walked past this one until review pointed at it.
    #
    # The BODY only, never the commentary — §7.1's rule, and it is the same reason here as
    # there: a reviewer's note asking "where is the KPI workbook?" is not the workbook, and
    # counting it would close the gap with the complaint about it.
    bodies = []
    for record in records:
        on_file = core._text_on_file(conn, record["id"]) or {}
        bodies.append(" ".join(str(part) for part in (on_file.get("body") or [])).lower())

    found = []
    for expected in declared:
        if not expected["looks_like"]:
            continue
        if any(word in text for text in bodies for word in expected["looks_like"]):
            continue
        found.append({
            "code": "expected_input_never_supplied",
            "expectation": expected["id"],
            "field": expected["input"],
            "campaigns": len(records),
            "what": f"Your rulebook says a brief should carry {expected['input']}, and no "
                    f"record on file mentions one ({len(records)} checked).",
            "what_it_means": f"{expected['input']} — {expected['why']} This is YOUR rule, "
                             f"from rulebook {rulebook.version()}, reported back rather than "
                             f"invented here.",
            "why_it_matters": expected["why"],
            "basis": "computed",
            "evidence": f"searched every brief's body text for "
                        f"{', '.join(repr(w) for w in expected['looks_like'])}",
        })
    return found


def gaps(conn) -> dict:
    """`_gaps`, reading each record at most once for the whole report (§13.3/D79).

    Two of this report's parts read every record's text — the declared-expectation check and
    the never-recorded-field walk, which reaches it through `facts.for_campaign` — so a
    scope around each of them separately still read everything twice. The report is one
    question about one library at one moment; a record's text cannot change inside it.
    """
    import core

    with core._each_record_read_once():
        return _gaps(conn)


def _gaps(conn) -> dict:
    """What this library is missing, ranked, with what would close each one.

    The counterpart to `most_valuable_missing_input` on a judgment: this is about the
    LIBRARY, that is about one verdict, and they routinely disagree — a library that is 90%
    measured can still produce a judgment resting entirely on the unmeasured tenth.

    §10.6/D76 folded in the first-steps path, which `readiness` also shows. It is the same
    ordered list from the same function, not a second copy — and it is empty once the path
    is walked, because guidance that never stops appearing is guidance nobody reads.
    """
    import core

    ranked = _ranked_gaps(conn)
    # Appended after ranking, not among the ranked gaps: these are patterns across a market
    # rather than one missing thing, and they carry no offer, because closing one means
    # editing several records. Ranking an unofferable gap above an offerable one would put
    # the thing nobody can act on first.
    superseded = store.get_superseded_campaign_ids(conn)
    field_gaps = _fields_never_recorded(conn, [
        c for c in store.list_campaigns(conn)
        if c.get("record_type") != "reference" and c["id"] not in superseded])
    # Carrying `order` too, past every ranked gap. They are appended rather than sorted in,
    # but a list where some rows have the key the list is ordered on and some do not is one
    # nobody can check the order of — and "appended last" is a position, so it should say so
    # in the same field as every other position rather than only by where it sits.
    # D50: an input the RULEBOOK declares a brief must carry, which no brief on file carries.
    # The review's third gap case, and it needed the rulebook to exist: an expectation nobody
    # wrote down cannot be reported as missing without the product inventing a requirement on
    # the customer's behalf. The product declares none, so this is empty until a customer
    # declares one in their overlay — which is the point. It is their rule, reported back.
    field_gaps += _expected_inputs_never_supplied(conn)
    last = max((g["order"] for g in ranked["gaps"]), default=0) + 1
    for gap in field_gaps:
        gap.update({"order": last, "rank": last, "affects": gap["campaigns"],
                    "share": 0.0, "outweighs_its_kind": False, "rank_basis": "computed"})
    ranked["gaps"] += field_gaps
    ranked["shortest_path"] = core.first_steps(conn)
    return ranked


def top_gap(conn) -> Optional[str]:
    """The code of the gap worth fixing first, or None (§10.6/D54).

    Separate from `gaps()` because it has to be cheap enough to call on every upload:
    `_fields_never_recorded` walks `facts.for_campaign` per record and cannot change
    `most_valuable` — it is appended after the ranking — so the expensive half is skipped.
    """
    return _ranked_gaps(conn)["most_valuable"]


_TOP_GAP_KEY = "top_gap"


def _gap_moved(conn) -> Optional[str]:
    """Has the library's worst problem changed since anybody was told? (§10.6/D54.)

    The second half of D54, and the harder half. "Offer the ranking at session start" is a
    condition a stateless call can evaluate; "offer it when an upload changes the top gap" is
    not — `changed` needs the previous answer, so one is remembered.

    Reads only. The first version also WROTE the new value here, before the offer it enables
    had reached `trim` — and the offer is appended last, so on any upload with three other
    things to say it was dropped AND the change was recorded as already announced. It
    therefore fired ZERO times per change on exactly the uploads that matter most: a v2
    replacing a judged record, or a concluded campaign with no results and no dates. The
    docstring claimed "once per change"; it was none. `_the_gap_was_announced` is now called
    by the caller, after the offer survives.

    A library with nothing remembered — a fresh install, or a database from before this
    existed — reads as moved, and it has: the ranking went from meaning nothing to meaning
    something. Losing the row costs one redundant offer, never a wrong answer.
    """
    now = top_gap(conn)
    # `library_is_empty` cannot be the NEW top gap after an upload, but it can be the
    # remembered one, and moving off it is the most meaningful move there is.
    return now if now and now != store.get_state(conn, _TOP_GAP_KEY) else None


def _the_gap_was_announced(conn, offers: list, moved: Optional[str]) -> None:
    """Remember the top gap only once somebody has actually been shown it (§10.6/D54).

    The whole of the fix for the burned offer: `trim` is the last word on what a user sees,
    so it is the only place that can say whether the announcement happened.
    """
    if moved and any(offer["tool"] == "gaps" for offer in offers):
        store.set_state(conn, _TOP_GAP_KEY, moved)


# ── answering back (§10.2, D84 / D59 / D53) ────────────────────────────────
#
# The four rows the tracker points at 10.2 are one thing: a place for a person's answer to
# go. This product asks several questions on surfaces with no return path — "is this
# departure deliberate?", "did you fix that?", "is this gap ever going to close?" — and the
# reply goes into a chat window and dies.
#
# That is worse than not asking, twice over. The question comes back, so the product looks
# like it is not listening, because it is not. And the UN-answer is stored: `departure:
# unexplained` is a fact every later judgment reads, and it is false the moment somebody
# explains it. A question nobody can answer decays into a wrong answer.

# What a person can say about a stored finding. Three, and the distinction between the first
# two is the one D84 insists on in writing: `resolved` "records 'we changed it', not 'we kept
# it, and here is why'". A library that cannot tell a brief that was corrected from one that
# was defended has lost what the whole correction loop reasons from.
FINDING_ANSWERS = {
    "fixed": "The problem was real and we addressed it.",
    "deliberate": "We kept it on purpose, and here is why.",
    # `not_applicable` used to carry both of these, and they are not the same answer. "That
    # rule does not apply to this brief" accepts the finding and disputes its scope; "you
    # have misread the brief" disputes the finding itself. Collapsing them loses the
    # disagreement — and a disagreement with a judgment is the single most valuable thing
    # this library can be told, because it is the only signal that the product got something
    # wrong. The whole review that started this began with "impossible to argue with".
    "does_not_apply": "The point is right in general and does not apply to this brief.",
    "misread": "The finding is wrong about what the brief says.",
    # The way back, which `GAP_ANSWERS` had from the start and this did not. A finding
    # answered wrongly could be re-answered but never returned to unanswered, so the first
    # mis-click was permanent — and `_offer_to_settle` was prefilling `deliberate`, which
    # made a wrong first answer likely rather than hypothetical. The whole point of a
    # person's answer is that a person can change it.
    "open": "Never mind — treat it as unanswered again.",
}

# How each answer reads when `diff_campaigns` renders it. `misread` is deliberately blunt:
# a person saying the judgment was wrong is the most valuable thing this library gets told,
# and softening it into "not applicable" is how a product stops hearing that it is wrong.
_SETTLED_AS = {
    "fixed": "addressed",
    "does_not_apply": "right in general but not applicable to this brief",
    "misread": "wrong about what the brief says",
}

# What a person can say about a gap they cannot close. `known_not_yet` stays ranked, because
# it is still the thing to fix; `not_applicable` leaves the ranking, because a market whose
# agency no longer exists will never have its results and reporting it forever is the
# permanent complaint this surface is written against. One state for both would either nag
# somebody who has answered or silence a gap that is real.
GAP_ANSWERS = {
    "known_not_yet": "Known, and not being done yet.",
    "not_applicable": "This will never be true here.",
    "open": "Never mind — treat it as outstanding again.",
}


# Past this, an answer is old enough that nobody can reasonably say it still reflects what
# they think. Not a deadline and not a scheduler — this product has no scheduler and says so
# in `after_evaluation` — just the point at which the log stops letting the age go unsaid.
_STALE_ANSWER_DAYS = 180


def answers(conn, *, said_by: Optional[str] = None) -> dict:
    """Every answer a person has given this library, newest first (§10.2).

    The read path the write paths needed. §10.2 gave this product somewhere for a person's
    answer to go and nothing to read them back with — and for a product whose whole pitch is
    that its judgments can be ARGUED WITH, the record of where somebody argued and won is the
    most valuable thing it holds. It was write-only, which is D116's shape one level down: a
    team lead could not ask "what have we marked deliberate", "who set aside which gaps" or
    "show me everything Ana answered", so nobody reviewed the overrides, so a wrong one was
    permanent in practice even though `open` makes it reversible in principle.

    Every row, not the standing one per subject. The table is append-only so that "this
    looked deliberate in March and turned out to be a mistake in June" survives, and a log
    showing only June has discarded the half that made keeping both worth doing. `stands`
    marks which one is current.
    """
    rows = store.every_answer(conn, said_by=said_by)
    if not rows:
        return {
            "status": "nothing_to_check", "basis": "computed", "answers": [],
            "what_it_means": (
                f"Nobody has answered anything{f' — nothing from {said_by}' if said_by else ''}"
                f". That is not the same as nothing having been asked: a judgment that raised "
                f"an unexplained departure asked one, and a gap somebody could not close "
                f"asked another. It means no answer is on file, so every such question is "
                f"still open and will be put again."),
        }
    seen: set = set()
    now = time.time()
    out = []
    for row in rows:
        subject = (row["subject_kind"], row["subject_key"])
        days = int((now - row["created_at"]) / 86400)
        entry = {
            "answer": row["answer"], "note": row["note"], "said_by": row["said_by"],
            "said_at": row["said_at"], "basis": "stated",
            "subject_kind": row["subject_kind"],
            "about": _what_the_answer_was_about(conn, row),
            # Newest first, so the first time a subject is seen is the answer that stands.
            "stands": subject not in seen,
            "days_ago": days,
            "stale": days >= _STALE_ANSWER_DAYS,
            # The overrides that matter most are the ones that made the product say LESS.
            "silences": row["answer"] in ("not_applicable", "deliberate", "misread"),
        }
        # §11.1: the account beside the name. `answers` is the surface built to review
        # overrides, so it is the surface where "who was actually at the keyboard" belongs.
        who = store.authorship_for(conn, "answer", row["subject_key"])
        entry["captured_by"] = (who or {}).get("captured_by") or {
            "method": "unattributed", "source": "stated",
            "what_it_means": "Recorded before this library kept the account."}
        if row["subject_kind"] == "finding":
            entry["evaluation_id"] = row["evaluation_id"]
            entry["finding_id"] = row["subject_key"]
            judgment = store.get_evaluation(conn, row["evaluation_id"] or "")
            entry["subject_title"] = (judgment or {}).get("subject_title")
        seen.add(subject)
        out.append(entry)
    stale = [e for e in out if e["stands"] and e["stale"]]
    silencing = [e for e in out if e["stands"] and e["silences"]]
    return {
        "status": "checked", "basis": "computed", "answers": out,
        "counts": {"answers": len(out), "standing": len(seen), "silencing": len(silencing)},
        "what_it_means": (
            f"{len(seen)} question(s) this library asked have an answer on file, from "
            f"{len(out)} statement(s) — the older ones are kept, because what somebody said "
            f"before the numbers came in and what they said after are two judgments. "
            f"`stands` marks the one in force. {len(silencing)} of them make this product "
            f"say LESS than it otherwise would, which is the half worth reading first."
            + (f" {len(stale)} were given over {_STALE_ANSWER_DAYS} days ago and nobody has "
               f"revisited them; a market written off in March may have been revived since, "
               f"and nothing here re-asks."
               if stale else "")),
    }


def _what_the_answer_was_about(conn, row: dict) -> str:
    """The question in its own words. A log of `finding_id` values is a log nobody can read,
    and the person reviewing overrides is not the person who made them."""
    if row["subject_kind"] != "finding":
        return row["subject_key"]
    judgment = store.get_evaluation(conn, row["evaluation_id"] or "")
    for finding in (judgment or {}).get("findings") or []:
        if finding.get("id") == row["subject_key"]:
            return finding.get("finding") or row["subject_key"]
    return row["subject_key"]


def backfill_author_unknown(conn) -> dict:
    """Stamp records stored before this library read commentary, with the date (§11.6).

    What it stamps is a fact about US: "this record predates commentary extraction, so the
    file may well have said who wrote it and nobody looked." Reporting that as the document's
    silence would blame a customer's deck for a gap in what this product captured.

    It never touches a name. A record whose commentary WAS read carries real people, and
    writing `unknown` over them would destroy the personal data §11.7 has to be able to show
    and erase, while making the library look as though it had never known.

    Runs once per record. A second stamp is not wrong so much as untrue: it moves the import
    date onto the day somebody happened to re-run this.
    """
    when = store._now_iso()
    stamped = store.records_without_authorship_backfill(conn)
    for record in stamped:
        store.mark_authorship_backfilled(conn, record["id"], when)
    return {
        "records": len(stamped), "backfilled_at": when, "basis": "computed",
        "titles": [r["title"] for r in stamped[:10]],
        "what_it_means": (
            f"{len(stamped)} record(s) were stored before this library read commentary at "
            f"all, and are now marked as such with today's date. Their authorship reads "
            f"`unknown` because nobody looked — not because the files were silent — and a "
            f"judgment citing their words is told which of those it is looking at."
            if stamped else
            "Every record already says why it can or cannot name an author. Nothing needed "
            "stamping, which is the answer on a library whose records were all read."),
    }


def record_reaction(conn, *, campaign_id: str, value: str, said_by: str,
                    role: Optional[str] = None, source: str = "stated") -> dict:
    """Record one person's view of a campaign, keeping everyone else's (§11.5).

    The write `update_campaign`'s tag list could not be: it REPLACES, so a second opinion
    erased the first. Both are kept here, and neither is ranked — see `store.disagreement_on`
    and D10 for why this library refuses to say who is right.
    """
    if store.get_campaign(conn, campaign_id) is None:
        raise ValueError(f"{campaign_id!r} is not a record in this library.")
    if value not in store.REACTION_AXES:
        raise ValueError(
            f"{value!r} is not something this library records a view about. The closed sets "
            f"are what make two people comparable: {sorted(store.REACTION_AXES)}.")
    said_by = identity.person(said_by, field="said_by")
    if source not in identity.AUTHOR_SOURCES:
        raise ValueError(f"`source` must be one of {list(identity.AUTHOR_SOURCES)}.")
    said_at = store._now_iso()
    store.record_reaction(conn, campaign_id=campaign_id, value=value, said_by=said_by,
                          source=source, said_at=said_at, role=role)
    store.record_authorship(conn, subject_kind="reaction", subject_key=campaign_id,
                            on_behalf_of=said_by, role=role)
    split = store.disagreement_on(conn, campaign_id)
    return {
        "campaign_id": campaign_id, "value": value, "said_by": said_by, "said_at": said_at,
        "source": source, "basis": "stated", "status": "recorded",
        **({"role": role} if role else {}),
        **({"disagreement": split} if split else {}),
        "what_it_means": (
            f"{said_by}'s view is on file. Everything anybody else said is still on file too "
            f"— this never replaces."
            + (" They and somebody else have recorded different views, which is worth citing "
               "as a split rather than as a verdict."
               if split else "")),
    }


def answer_finding(conn, *, evaluation_id: str, finding_id: str, answer: str, note: str,
                   said_by: str) -> dict:
    """Record what a person says about one finding on a saved judgment (§10.2/D84+D59).

    The write path both rows needed. Every check here exists because the alternative is an
    answer that is worse than the silence it replaced:

    `note` is required, because "deliberate" with no reason is the same non-answer as
    `unexplained` — recorded as though it were an answer, and therefore stopping the question
    being asked. That is strictly worse than the state it replaces.

    `said_by` is required, because this overrides a stored finding, and every `stated` claim
    in this product carries who made it. An anonymous override of a computed finding is the
    one thing this library refuses everywhere.

    The finding has to be ON the judgment, because a settled finding nobody can find is an
    answer filed against nothing — and it would return `status: settled` while settling
    nothing, which is the shape of every defect this review found.
    """
    if answer not in FINDING_ANSWERS:
        raise ValueError(
            f"answer must be one of {list(FINDING_ANSWERS)}, got {answer!r}. "
            + " ".join(f"`{k}`: {v}" for k, v in FINDING_ANSWERS.items()))
    if not (note or "").strip():
        raise ValueError(
            "`note` is required — say why, in the words somebody used. An answer with no "
            "reason is the same non-answer as `unexplained`, except that it stops the "
            "question being asked, which makes it worse than saying nothing.")
    # §11.1/§11.2: `identity.person` is the one guard, so every path that names somebody
    # refuses the same things. It also refuses the product naming itself, which this check
    # did not — and this is the field that overrides a stored judgment.
    said_by = identity.person(said_by, field="said_by")
    judgment = store.get_evaluation(conn, evaluation_id)
    if not judgment:
        raise ValueError(f"{evaluation_id!r} is not a judgment on file")
    finding = next((f for f in judgment.get("findings") or []
                    if f.get("id") == finding_id), None)
    if not finding:
        raise ValueError(
            f"{finding_id!r} is not a finding on {evaluation_id}. Its findings are: "
            f"{[f.get('id') for f in judgment.get('findings') or []]}")

    store.record_answer(conn, subject_kind="finding", subject_key=finding_id,
                        evaluation_id=evaluation_id, answer=answer, note=note.strip(),
                        said_by=said_by.strip())
    store.record_authorship(conn, subject_kind="answer", subject_key=finding_id,
                            on_behalf_of=said_by)
    return {
        "status": "settled",
        "evaluation_id": evaluation_id,
        "finding_id": finding_id,
        "answer": answer,
        "note": note.strip(),
        "said_by": said_by.strip(),
        "basis": "stated",
        "what_it_means": (
            f"Recorded against this finding, and it travels with it: every later judgment "
            f"that retrieves this record reads the answer rather than the question. "
            + "The next judgment of this brief is shown your answer alongside the finding, "
              "and told not to raise it again as though nobody had said anything. That is "
              "the strongest claim this can make: the model still writes the findings, so "
              "it is being informed rather than constrained."),
    }


def answer_gap(conn, *, code: str, answer: str, note: str, said_by: str) -> dict:
    """Record what a person says about a gap they cannot close (§10.2/D53).

    Silenced in the RANKING, never deleted from the record. A gap somebody set aside is still
    a fact about the library's evidence, and a judgment resting on that evidence is still
    weaker for it — hiding it entirely would make the product overstate its own coverage,
    which is what this whole review is about.
    """
    import core

    if answer not in GAP_ANSWERS:
        raise ValueError(
            f"answer must be one of {list(GAP_ANSWERS)}, got {answer!r}. "
            + " ".join(f"`{k}`: {v}" for k, v in GAP_ANSWERS.items()))
    if not (note or "").strip():
        raise ValueError(
            "`note` is required — say why this gap is not going to close, or when it will. "
            "A gap set aside with no reason is one nobody can reopen intelligently.")
    said_by = identity.person(said_by, field="said_by")
    # Against the gaps this library ACTUALLY has, not against the registry of codes. A
    # silent no-op for a gap nobody is reporting reads as done and is not — and `_GAP_RANK`
    # would happily accept `no_window` for a library where every record has dates.
    ranked = [g["code"] for g in _ranked_gaps(conn)["gaps"]]
    aside = store.answers_for(conn, "gap")
    if code not in ranked and code not in aside:
        raise ValueError(
            f"{code!r} is not a gap this library is reporting. Call gaps() to see what it "
            f"has: {ranked}")
    # ONE rule, not two. The offer was gated on `_CAN_BE_SET_ASIDE` and the write was not, so
    # the whitelist that keeps somebody from silencing "finished campaigns have no results on
    # file" guarded only the suggestion — `answer_gap(code="library_is_empty",
    # answer="not_applicable")` was accepted on an empty library and `gaps()` then returned
    # `[]`, which is the product saying nothing is missing about a library holding nothing.
    # Two implementations of one rule is the shape this file has now hit five times, and this
    # is the version where the two disagree about something that matters.
    if answer != "open" and code not in _CAN_BE_SET_ASIDE:
        raise ValueError(
            f"{code!r} cannot be set aside. “This will never be true here” is honest about a "
            f"market whose agency no longer exists — {list(_CAN_BE_SET_ASIDE)} — and not "
            f"about a gap that describes work nobody has done yet. Silencing "
            f"`few_verified_outcomes` hides the gap this product was built around, and "
            f"silencing `judgments_never_reconciled` makes its own calibration figure "
            f"unfalsifiable. Close it, or leave it reported.")

    store.record_answer(conn, subject_kind="gap", subject_key=code, answer=answer,
                        note=note.strip(), said_by=said_by.strip())
    store.record_authorship(conn, subject_kind="answer", subject_key=code,
                            on_behalf_of=said_by)
    return {
        "status": "recorded", "code": code, "answer": answer, "note": note.strip(),
        "said_by": said_by.strip(), "basis": "stated",
        "what_it_means": {
            "known_not_yet": "It stays in the ranking, because it is still the thing to fix — "
                            "but the note travels with it, so nobody has to work out whether "
                            "anyone has looked at it.",
            "not_applicable": "It leaves the ranking and stays on the record under "
                              "`set_aside`. Judgments resting on this evidence are still "
                              "weaker for it, and still say so.",
            "open": "Back in the ranking, as though nothing had been said.",
        }[answer],
    }


def authorship_backfill_offer(conn) -> list[dict]:
    """Offer the §11.6 stamp when records actually need it.

    The moment is a library holding records from before commentary extraction: their search
    hits say `unknown` for a reason nobody has recorded, so a reader cannot tell "the file was
    silent" from "we never looked". Silent otherwise — a library whose records were all read
    has nothing to stamp, and offering it anyway is the footer this product keeps refusing.
    """
    waiting = store.records_without_authorship_backfill(conn)
    if not waiting:
        return []
    return [actions.action(
        f"Record that {len(waiting)} older record(s) predate commentary extraction",
        "backfill_author_unknown",
        why=f"{len(waiting)} record(s) were stored before this library read commentary, so "
            f"their authorship reads `unknown` with nothing saying whether the file was "
            f"silent or nobody looked. A judgment citing their words cannot tell a client's "
            f"objection from the deck talking to itself.",
        consent="do")]


def declared_corrections_offer(conn) -> list[dict]:
    """Offer to put the customer's own standing corrections in force (§12.3/D116).

    A customer writes ten corrections into their rulebook, restarts, and nothing happens: the
    rules sit in the file and the library never mentions them. That is D108's "stored and
    never applied" arriving one level up — through nobody being told there was anything to do.
    This product has hit that shape eight times, and §10.6's answer is that a capability
    nobody offers is a capability nobody calls.

    Silent once they are in force, because guidance that never stops appearing is guidance
    nobody reads.
    """
    import corrections
    import rulebook

    try:
        declared = rulebook.corrections()
    except ValueError:
        return []   # a broken rulebook is `health_check`'s finding, loudly, not an offer
    if not declared:
        return []
    waiting = [entry for entry in declared if not corrections.find(conn, entry["text"])]
    if not waiting:
        return []
    return [actions.action(
        f"Put the {len(waiting)} standing correction(s) from your rulebook in force",
        "load_rulebook_corrections",
        why=f"Your rulebook ({rulebook.overlay() or rulebook.version()}) declares "
            f"{len(waiting)} standing correction(s) that no brief is being judged against "
            f"yet. They do not have to be seen in three campaigns first — that test is for a "
            f"rule this library inferred — but somebody has to put their name to them.",
        consent="ask",
        # NOT prefilled. The one thing the server cannot work out is whose confirmation this
        # is, and filling it in would put a rule in front of every future brief under a name
        # nobody gave.
        needs=["who is confirming these — ask, do not assume"])]


def stale_answers_offer(conn) -> list[dict]:
    """Offer the override log when something that SILENCES this product has gone unreviewed.

    The honest moment for it, and the only one. Offering "read the log" right after somebody
    writes to it is noise — they have just been told what they said. What nobody is ever told
    is that a gap written off months ago is still written off: the queue's founding insight is
    that a question asked once and never again is a question nobody answered, and until now
    that insight was not applied to the ANSWERS. A market written off in March may have been
    revived since, and nothing here re-asks.

    Gated on `stale` AND `silences`, so it fires on the answers that made the library say
    less — never on a tidy log of things somebody explained.
    """
    silenced = [a for a in answers(conn).get("answers") or []
                if a["stands"] and a["stale"] and a["silences"]]
    if not silenced:
        return []
    return [actions.action(
        f"Review the {len(silenced)} answer(s) nobody has revisited",
        "answers",
        why=f"{len(silenced)} standing answer(s) make this library report less than it "
            f"otherwise would, and the most recent is {min(a['days_ago'] for a in silenced)} "
            f"days old. Nothing re-asks; a market written off last year may have come back.",
        consent="do")]


def gaps_offer(conn) -> list[dict]:
    """Offer the ranked gaps, when there is a ranking worth reading (§10.6/D54).

    `gaps`'s own docstring says "call it when the user asks how good their library is" — an
    instruction inside a tool result, which is the form this project's stated principle says
    models drop. The library knows it holds one measured campaign and has never said so.

    Not on an empty library: `library_is_empty` is the only gap there, `shortest_path` is
    already the answer to it on the same response, and an offer to enumerate the gaps of an
    empty library is the permanent complaint this surface is written against.
    """
    top = top_gap(conn)
    if not top or top == "library_is_empty":
        return []
    return [actions.action(
        "Show what this library is missing, worst first",
        "gaps",
        why=_GAP_HEADLINE.get(top, "Something in the library is holding its judgments back."),
        consent="do")]


# The one-line reason for offering the ranking, per top gap. Written here rather than read
# off the gap's own `what`, because that sentence carries counts and names and this one has
# to fit on a line beside two other offers — and because an offer whose `why` is the answer
# it is offering to fetch is an offer nobody needs to accept.
_GAP_HEADLINE = {
    "few_verified_outcomes": "Finished campaigns here have no results on file, so judgments "
                             "citing them rest on somebody's impression.",
    "market_without_outcomes": "A whole market has nothing measured, so judgments about it "
                               "compare against campaigns somewhere else.",
    "partly_indexed": "Records are stored and not searchable, so their content is absent "
                      "from evidence without being absent from the library.",
    "execution_never_checked": "Outcomes are being read as though the brief caused them, "
                               "with nothing on file showing what actually ran.",
    "no_window": "Finished campaigns have no dates, so nothing can be checked against what "
                 "else was going on at the time.",
    "judgments_never_reconciled": "Judgments with results on file have never been checked "
                                  "against what happened, so the calibration figure is "
                                  "about whoever bothered.",
}


def coverage_offer(conn) -> list[dict]:
    """Offer the coverage matrix, when the library has a shape to show (§10.6/D67).

    `coverage`'s own docstring says `list_campaigns` "answers 'what have I got' one record at
    a time. This answers the question a marketer actually has" — which is the tool admitting
    the listing is the moment for it, in a sentence the listing never carried.

    Gated on `thin` being non-empty, which is narrower than "there are cells". A wholly
    unmeasured library reports `thin: []` with a `thin_summary` saying the place to start is
    not a particular market — and `gaps` says that better, with an offer that closes it. So
    the matrix is offered when there IS a contrast to see: some cells measured, some not.
    """
    import core

    shape = core.coverage(conn)
    thin = shape.get("thin") or []
    if not thin or shape.get("cells_total", 0) < 2:
        return []
    # `thin[0]`, not a second grouping computed here — D67's own words. The first version of
    # `gaps` grew its own market-or-region grouping beside §5.5's and the two then described
    # the same library differently (D55); an offer POINTING at coverage that counted the
    # library itself would be that failure inside the fix for it.
    worst = thin[0]
    where = worst.get("market") or worst.get("collection") or worst.get("stage") or "one cell"
    return [actions.action(
        "Show where this library is thick and where it is thin",
        "coverage",
        # "the largest gap", NOT "the thinnest". `thin` is ranked by `(_EVIDENCE_ORDER,
        # -campaigns)`, so `thin[0]` is the worst-evidence cell with the MOST campaigns in
        # it — the biggest hole, not the smallest cell. Calling it the thinnest reported a
        # cell holding nine unmeasured campaigns as thinner than one holding a single
        # unmeasured campaign, which is backwards in the field that decides where to look.
        why=f"{shape['thin_total']} of {shape['cells_total']} cells are weak — "
            f"{where} is the largest of them, and the similarity score looks the same "
            f"whether a judgment came from one precedent there or nine.",
        consent="do")]


def _ranked_gaps(conn) -> dict:
    """The ranked gaps alone: everything that can decide `most_valuable`."""
    import core

    # ONE read, and the shared record line. Superseded records are excluded from every
    # search, so counting one as the library's measured evidence describes something no
    # judgment can reach; a `stub` IS results and belongs in the denominator of "finished
    # campaigns with nothing measured", which is why this population is wider than the one
    # `coverage` and `readiness` describe. `library_state` holds both and says which is which.
    state = core.library_state(conn)
    campaigns = [c for c in state["records"] if c.get("record_type") != "reference"]
    # A campaign that has not run cannot be missing its results, and asking for them is a
    # request nobody can satisfy — the permanent-complaint failure, on the highest-ranked
    # gap. `after_upload` already drew this line; this did not. Taken from the shared state
    # rather than recomputed: it was worked out here AND in `library_state`, where the
    # docstring claimed nothing had to subtract it by hand.
    ran = [c for c in campaigns if c["id"] in state["ran"]]
    found: list[dict] = []

    if not campaigns:
        # Every gap is present in an empty library, which makes the list useless. A new
        # install has exactly one thing to do, and it is not "fix your LATAM coverage".
        #
        # §10.6/D76: the FIRST step of the ordered path, not a separate list of two of its
        # three. `to_first_upload()` said "add a campaign" where the path says "add one you
        # were happy with", and the difference is the whole point of the path — a library of
        # things somebody liked cannot judge on the axis it is being asked about.
        found.append({
            "code": "library_is_empty",
            "what": "There are no campaigns in the library yet.",
            "why_it_matters": "Every judgment this product makes is a comparison against "
                              "what you have already run, so with nothing stored there is "
                              "nothing to compare against.",
            "counts": {"campaigns": 0}, "affects": 0,
            "next_actions": core.first_steps(conn)[:1],
        })
        return _what_people_said_about_the_gaps(conn, _ranked(found, library_size=0))

    # `has_metrics` counts any metric row, so a PREDICTED figure silenced this gap — and a
    # forecast is the opposite of a measured outcome; it is the thing reconciliation later
    # scores against the actuals.
    measured_ids = state["measured"]
    with_outcomes = [c for c in ran if c["id"] in measured_ids]
    if ran and len(with_outcomes) < len(ran):
        found.append({
            "code": "few_verified_outcomes",
            "what": f"{len(with_outcomes)} of {len(ran)} finished campaigns have measured "
                    f"results on file.",
            "why_it_matters": "A campaign with no outcome data can be cited as a precedent "
                              "but cannot show whether it worked, so a judgment resting on "
                              "it rests on somebody's impression.",
            "counts": {"campaigns": len(ran), "with_outcomes": len(with_outcomes)},
            "affects": len(ran) - len(with_outcomes),
            "next_actions": actions.trim([actions.action(
                "Record what one of these campaigns actually achieved",
                "add_metrics",
                why=f"{len(ran) - len(with_outcomes)} finished campaigns have no results.",
                consent="ask", needs=["which campaign, and the numbers"],
                campaign_id=next(c["id"] for c in ran if c["id"] not in measured_ids))]),
        })

    # §9.1/§9.2: a concluded campaign with measured results and nothing showing whether what
    # ran was what was briefed. The review's third possibility — "that what ran was not what
    # was briefed" — which the library could not previously notice at all.
    #
    # §13.6/D122. The condition was "no delivered creative, and briefed creative on file", and
    # both halves were wrong. It missed the record uploaded as text with nothing at either
    # phase — the one with the LEAST evidence about what ran. And its replacement, "no
    # delivered creative", produced an OFFER THAT DID NOT WORK: asked for the photographs, a
    # record with nothing briefed got a comparison with the other half empty, which
    # `compare_execution` refuses in as many words — and the upload then CLOSED the gap while
    # every citation of that campaign went on saying nobody had checked what it ran. Two
    # surfaces describing one record differently, produced by the product's own suggestion.
    #
    # So the condition is now the STORED STATUS the citations read, and not a second opinion
    # about it. `_snapshot_execution_drift` runs at every moment the answer can change and
    # writes `never_checked` when `compare_execution` could not answer; `_execution_note`
    # stamps that on every citation. Reading the same row here makes the gap and the caveat
    # one claim by construction — the gap is reported exactly while the caveat is being
    # attached, and it closes exactly when the caveat stops.
    unchecked = [c for c in with_outcomes
                 if store.execution_drift_for(conn, c["id"])["status"] == "never_checked"]
    if unchecked:
        # WHICH HALF IS MISSING, because that decides what to ask for and the previous version
        # asked for the wrong one. A record with boards and no photographs needs the
        # photographs; a record with photographs and nothing briefed needs the brief; a record
        # with neither needs the brief first, because the photographs alone would produce the
        # same empty-half refusal. A record with BOTH and no comparison on file needs neither —
        # it needs the comparison run.
        def _has(campaign, phase):
            return bool(store.assets_in_phase(conn, campaign["id"], phase))

        nothing_at_all = [c for c in unchecked
                          if not _has(c, "proposed") and not _has(c, "delivered")]
        nothing_briefed = [c for c in unchecked
                           if not _has(c, "proposed") and _has(c, "delivered")]
        nothing_delivered = [c for c in unchecked
                             if _has(c, "proposed") and not _has(c, "delivered")]
        never_run = [c for c in unchecked if _has(c, "proposed") and _has(c, "delivered")]
        # Least evidence first, and ONE ordering: `what`, `campaign_ids` and the offer all read
        # position 0 of this same list. Reading the target out of a different ordering put an
        # id in the arguments that appeared in neither — with seven records the sentence named
        # five boards-only campaigns and the offer said "add the photographs from" a sixth.
        order = {c["id"]: n for n, group in enumerate(
            (nothing_at_all, nothing_briefed, nothing_delivered, never_run)) for c in group}
        unchecked.sort(key=lambda c: order[c["id"]])
        target = unchecked[0]
        # An offer §5.2 would accept: one the tool takes AND the record can use. Each of these
        # leaves the record able to answer something it could not answer before.
        if target in never_run:
            offer = actions.action(
                f"Compare what ran on “{target['title']}” against its brief",
                "compare_execution",
                why="Both halves are on file and nothing has compared them, so every citation "
                    "of this campaign still says nobody checked what it ran.",
                consent="ask", campaign_id=target["id"])
        elif target in nothing_delivered:
            offer = actions.action(
                f"Add the photographs from “{target['title']}”",
                "upload_image_asset",
                why="The brief is on file and nothing shows what ran. With the delivered "
                    "photographs, compare_execution says which briefed elements appeared, "
                    "which did not, and which arrived unbriefed.",
                consent="ask", needs=["the photographs themselves"],
                campaign_id=target["id"], phase="delivered")
        else:
            # Bare, or photographs with nothing briefed. Both need the BRIEFED half, and
            # asking a bare record for the photographs instead was the false offer: it
            # produced "1 delivered image on file and nothing briefed to compare them
            # against", and closed the gap on the way.
            offer = actions.action(
                f"Add the briefed creative for “{target['title']}”",
                "upload_image_asset",
                why=("Nothing on file says what this campaign was supposed to look like, so "
                     "there is nothing for what ran to be compared against. The boards are "
                     "the half to add first — the photographs on their own would produce the "
                     "same empty comparison from the other side."),
                consent="ask", needs=["the briefed boards or brief images"],
                campaign_id=target["id"], phase="proposed")
        # The composition, named rather than counted twice. One record described as "1 of them"
        # is a record stated as a fraction of itself, and the version that suppressed the
        # briefed-creative clause whenever any record was bare never told the reader which
        # state the others were in.
        parts = [f"{len(group)} with {phrase}" for group, phrase in (
            (nothing_at_all, "no creative on file at all"),
            (nothing_briefed, "photographs but nothing briefed to compare them against"),
            (nothing_delivered, "briefed creative but no photographs of what ran"),
            (never_run, "both halves on file and no comparison ever run")) if group]
        composition = (f" — {parts[0].split(' with ', 1)[1]}" if len(unchecked) == 1
                       else " (" + "; ".join(parts) + ")")
        found.append({
            "code": "execution_never_checked",
            "what": f"{len(unchecked)} finished campaign"
                    f"{'s' * (len(unchecked) != 1)} with results on file "
                    f"{'have' if len(unchecked) != 1 else 'has'} never been checked against "
                    f"what actually ran" + composition + ": "
                    f"{', '.join(c['title'] for c in unchecked[:_MAX_NAMED])}"
                    + (f" (and {len(unchecked) - _MAX_NAMED} more)"
                       if len(unchecked) > _MAX_NAMED else "") + ".",
            "why_it_matters": ("Every outcome on those campaigns is being read as though the "
                               "brief caused it. If what ran was not what was briefed, the "
                               "library is learning from the wrong document and has no way "
                               "to notice."),
            "basis": "computed",
            "counts": {"campaigns": len(unchecked),
                       "no_creative_at_all": len(nothing_at_all),
                       "nothing_briefed": len(nothing_briefed),
                       "nothing_delivered": len(nothing_delivered),
                       "never_compared": len(never_run),
                       # `campaign_ids` is capped, and a caller reading a list of five beside
                       # a count of seven has to work out for itself whether it is looking at
                       # all of them. Say it.
                       "named": min(len(unchecked), _MAX_NAMED)},
            "affects": len(unchecked),
            "campaign_ids": [c["id"] for c in unchecked][:_MAX_NAMED],
            # The newest record this gap counts, so a set-aside made about the records on file
            # then does not silence it for records that arrive later (§10.2). Missing from the
            # first version of this row, which made `execution_never_checked` the one
            # set-aside-able code with no way back — "the product agreeing to stop mentioning
            # something it had not yet seen", in the one place the mechanism was written to
            # prevent it.
            "since": max((c.get("created_at") or 0) for c in unchecked),
            "next_actions": actions.trim([offer]),
        })

    # §12.4/D51: records whose deck's comments were never read. Recorded since §2.5 and NOT
    # reported until now, for a good reason: the only action available was `upload_campaign`,
    # which creates a SECOND record and fires `duplicate_title` — the offer §5.2 refused in
    # writing. "A gap whose only action makes things worse is a complaint."
    #
    # D39 built `attach_deck`, so the gap has an action that works and can be reported.
    # RECORDS WITH NO DECK AT ALL, which is the population `attach_deck` can actually repair.
    # Without the deck test this fired on every record whose brief arrived as typed text and
    # every upload whose file type carries no comments this product can read — and for those
    # the gap's own sentence ("not because their decks were silent, but because no file was
    # ever read") is false, while the action it offers is the one `attach_deck` refuses. A gap
    # whose only remedy is rejected by the tool it names is the complaint §5.2 spent an item
    # removing, arriving back through the door the fix for it opened.
    #
    # It is not a gap this product could honestly report for the others either: it cannot read
    # a .docx's comments, so it cannot know whether there are any, and "N records may have
    # unread comments" is the confident claim from an absence that §2.4 exists to refuse.
    unread = [c for c in store.list_campaigns(conn)
              if c.get("record_type") == "campaign"
              and not c.get("commentary_checked")
              and not (c.get("deck_text") or "").strip()
              and not c.get("asset_path")
              and c["id"] not in store.get_superseded_campaign_ids(conn)]
    if unread:
        found.append({
            "code": "commentary_never_read",
            "what": f"{len(unread)} record{'s' * (len(unread) != 1)} "
                    f"{'have' if len(unread) != 1 else 'has'} never had a deck read for "
                    f"comments: "
                    f"{', '.join(c['title'] for c in unread[:_MAX_NAMED])}"
                    + (f" (and {len(unread) - _MAX_NAMED} more)"
                       if len(unread) > _MAX_NAMED else "") + ".",
            "why_it_matters": ("A client's recorded objection is often the most useful "
                               "precedent in the library, and these records have none — not "
                               "because their decks were silent, but because no file was ever "
                               "read. A judgment citing them cannot tell those apart."),
            "counts": {"records": len(unread)},
            "affects": len(unread),
            "basis": "computed",
            # The newest record this gap counts, so a set-aside made about the records on
            # file then does not silence it for records that arrive later (§10.2).
            "since": max((c.get("created_at") or 0) for c in unread),
            # NOT `upload_campaign`: that is the duplicate §5.2 refused in writing, and
            # offering it here is what kept this gap unreported for seven phases.
            "next_actions": actions.trim([actions.action(
                f"Attach the deck for \u201c{unread[0]['title']}\u201d",
                "attach_deck",
                why="The deck's comments are read when it is attached, so the client's own "
                    "objections become precedent a judgment can cite — on the SAME record "
                    "rather than a second copy of it.",
                consent="ask",
                needs=["the deck file itself"],
                campaign_id=unread[0]["id"])]),
        })

    # §9.9: the item's whole diagnosis is that reconciliation "needs somebody to decide to go
    # back and nobody does" — and it gave itself no gap, so the number waiting was invisible on
    # every reporting surface. `calibration` meanwhile reported a perfect record beside them.
    waiting = store.unreconciled_judgments(conn)
    if waiting:
        found.append({
            "code": "judgments_never_reconciled",
            "what": f"{len(waiting)} judgment"
                    f"{'s' * (len(waiting) != 1)} with results on file "
                    f"{'have' if len(waiting) != 1 else 'has'} never been checked against "
                    f"what actually happened: "
                    f"{', '.join(j['subject_title'] for j in waiting[:_MAX_NAMED])}"
                    + (f" (and {len(waiting) - _MAX_NAMED} more)"
                       if len(waiting) > _MAX_NAMED else "") + ".",
            "why_it_matters": ("Reconciliation is the only thing in this product that says "
                               "whether its own judgments are any good, and it needs somebody "
                               "to go back. Until these are checked, the calibration figure "
                               "is about whoever bothered."),
            "counts": {"judgments": len(waiting),
                       "campaigns": len({j["campaign_id"] for j in waiting})},
            # RECORDS, like every other gap's `affects`. It counted JUDGMENTS, and `share`
            # then divided judgments by campaigns: one campaign carrying three unreconciled
            # judgments reported `share: 3.0` — not a share of anything — and, being above
            # the threshold, claimed that single record "outweighs its kind" and moved up a
            # rank. A ratio of two different units is a number that cannot be wrong, because
            # it does not mean anything. The judgment count is still reported, in `counts`,
            # where it is labelled.
            "affects": len({j["campaign_id"] for j in waiting}),
            "next_actions": actions.trim([actions.action(
                f"Check what the library said about \u201c{waiting[0]['subject_title'][:36]}\u201d "
                f"against what happened",
                "reconcile_evaluation",
                why="The results are on file and the judgment has never been tested against "
                    "them. This is the one surface that grades the product.",
                consent="ask", evaluation_id=waiting[0]["id"])]),
        })

    # §9.6: a concluded campaign with no window is one the calendar can never reach. §9.5 gave
    # itself `execution_never_checked` for exactly this reason — without a gap, the absence is
    # invisible on every reporting surface and nothing ever says "none of these can be checked
    # against what else was going on".
    windowless = [c for c in ran if not c.get("starts_on")]
    if windowless:
        found.append({
            "code": "no_window",
            "what": f"{len(windowless)} finished campaign"
                    f"{'s' * (len(windowless) != 1)} "
                    f"{'have' if len(windowless) != 1 else 'has'} no dates on file, so "
                    f"nothing can be matched to what else was happening at the time: "
                    f"{', '.join(c['title'] for c in windowless[:_MAX_NAMED])}"
                    + (f" (and {len(windowless) - _MAX_NAMED} more)"
                       if len(windowless) > _MAX_NAMED else "") + ".",
            "why_it_matters": ("“This launch overlapped Ramadan” and “the port "
                               "was shut for half of it” are findings no model needs to "
                               "be clever to produce — but only for a campaign that says when "
                               "it ran. Without a window every outcome is read as though "
                               "nothing else was going on."),
            "counts": {"campaigns": len(windowless)}, "affects": len(windowless),
            "since": max((c.get("created_at") or 0) for c in windowless),
            "next_actions": actions.trim([actions.action(
                f"Say when “{windowless[0]['title'][:36]}” ran",
                "update_campaign",
                why="A window is what lets recorded events reach it — and they reach it "
                    "automatically once it has one, with nothing to link by hand.",
                consent="ask",
                needs=["starts_on — the first day it ran (YYYY-MM-DD)",
                       "ends_on — the last day (YYYY-MM-DD)"],
                campaign_id=windowless[0]["id"])]),
        })

    # A gap located in a SLICE rather than in the total. A marketer asking about Colombia
    # does not care that the library is 60% measured overall if the LATAM part is 0%.
    # Grouped by §5.5's `_markets_of`, not by a second implementation here. This grew its
    # own market-or-region grouping and the two surfaces then described the same library
    # differently — one of them ignoring the `markets` list entirely (tracker D55).
    by_market: dict[str, list] = {}
    display: dict[str, str] = {}
    for campaign in ran:
        for raw in core._markets_of(campaign):
            if raw is None:
                continue
            key = raw.strip().lower()
            display.setdefault(key, raw.strip())
            if campaign not in by_market.setdefault(key, []):
                by_market[key].append(campaign)

    # Ordered by how many campaigns are affected, not alphabetically: the one place magnitude
    # decided anything was deciding it by the alphabet, offering Andorra's single campaign
    # ahead of LATAM's twenty. And no `len(barren) < len(by_market)` guard — it was meant to
    # avoid noise and instead silenced the review's own example, since "no LATAM store launch
    # has ever carried a budget" in a library holding only LATAM campaigns said nothing.
    barren = sorted((m for m, rows in by_market.items()
                     if not any(c["id"] in measured_ids for c in rows)),
                    key=lambda m: (-len(by_market[m]), m))
    if barren:
        # Capped: 35 markets produced a 3,185-character sentence inside a tool result
        # somebody has to read.
        shown = [display[m] for m in barren[:_MAX_NAMED]]
        more = f" (and {len(barren) - _MAX_NAMED} more)" if len(barren) > _MAX_NAMED else ""
        # Not a campaign an earlier gap already sent them to: the oldest unmeasured record is
        # usually also the barren market's first, and accepting both offers appended two
        # rows to it.
        already = {a["prefilled_args"].get("campaign_id")
                   for g in found for a in g["next_actions"]}
        target = next((c for m in barren for c in by_market[m]
                       if c["id"] not in already), by_market[barren[0]][0])
        found.append({
            "code": "market_without_outcomes",
            "what": f"No campaign in {', '.join(shown)}{more} has measured results.",
            "why_it_matters": "Judgments about these markets can only be compared against "
                              "campaigns elsewhere, which is a weaker comparison than it "
                              "looks.",
            "counts": {"markets": shown, "markets_total": len(barren)},
            "affects": len({c["id"] for m in barren for c in by_market[m]}),
            "since": max((c.get("created_at") or 0)
                         for m in barren for c in by_market[m]),
            "next_actions": actions.trim([actions.action(
                f"Record results for a campaign in {display[barren[0]]}",
                "add_metrics",
                why=f"{display[barren[0]]} has campaigns on file but nothing measured.",
                consent="ask", needs=["which campaign, and the numbers"],
                campaign_id=target["id"])]),
        })

    outstanding = store.count_unembedded(conn)
    if outstanding["chunks"] + outstanding["assets"]:
        by_campaign = store.outstanding_by_campaign(conn)
        found.append({
            "code": "partly_indexed",
            "what": f"{len(by_campaign)} record(s) are stored but not fully searchable.",
            "why_it_matters": "The content is here and searches cannot find it, so it is "
                              "absent from evidence without being absent from the library.",
            "affects": len(by_campaign),
            "counts": {"records": len(by_campaign),
                       "items": outstanding["chunks"] + outstanding["assets"]},
            "next_actions": actions.to_finish_indexing(
                by_campaign[0]["campaign_id"] if by_campaign else None),
        })

    # DELIBERATELY not reported, though it is recorded (§5.2's D39). The only offer available
    # is `upload_campaign`, which creates a SECOND record and fires `duplicate_title` — item
    # 5.2 refused exactly this offer in writing, one commit before this one asked for it. No
    # tool attaches a deck to an existing campaign. A gap whose only action makes things
    # worse is a complaint, which this item's own rule forbids; `commentary_checked` is kept
    # so it can be reported the moment there is something that closes it (D51).

    return _what_people_said_about_the_gaps(
        conn, _ranked(found, library_size=len(campaigns)))


# The only gaps where "this will never be true here" is an honest thing for somebody to say.
#
# A whitelist, after review found the blacklist let through the two that matter most. The
# honest case for D53 is a market whose agency no longer exists and whose numbers are gone —
# a fact that CANNOT become true. It is not `few_verified_outcomes` ("finished campaigns have
# no results on file"), which is the gap behind the original customer complaint, and it is
# not `judgments_never_reconciled`, where silencing it makes the product's own self-
# assessment unfalsifiable — the calibration figure would then be about whoever bothered,
# with nothing left to say so. Those are about work nobody has DONE, not facts that cannot
# exist, and a blacklist cannot tell the difference because the difference is not syntactic.
#
# `execution_never_checked` is out for the same reason: photographs of what ran are work.
# §12.4/D51: `commentary_never_read` joins them, and it has to. A brief that only ever
# existed as pasted text has no file whose comments could be read, so without a way to say
# "there is no deck" the gap is permanent and unclosable — which is a gap everybody learns to
# ignore, and D51's own reason for not reporting it at all. Now it closes two ways: attach the
# deck, or say there is not one.
# §13.6/D122 added `execution_never_checked`, and it had to be re-answered rather than
# inherited: while the gap required briefed creative to already exist, every record it reached
# had somebody who photographs, so "there will never be photographs here" was never true of one.
# Widened to records with no creative at either phase, it reaches the shop that works from
# descriptions — and for them it is permanent, which is exactly what this whitelist is for. The
# same trade `commentary_never_read` already took: the answer is per CODE, so a library that
# does photograph can silence it too.
# What keeps THAT honest is `_execution_note`, not the set-aside record. An earlier version of
# this comment said the gap "stays on the record under `set_aside`" and that judgments resting
# on the evidence still say so — the first half is true and the second does not follow from it.
# Nothing that builds a judgment reads `set_aside`; its only readers are `gaps` and `readiness`.
# The caveat that actually travels is `_execution_note`, attached per CITED RECORD, which
# stamps "nobody has checked what this campaign actually ran… that is not the same as it having
# run faithfully" on every citation regardless of what anyone set aside. Saying otherwise would
# tell the next maintainer there is a backstop under the per-citation note, which is exactly
# the belief that would let somebody weaken it.
_CAN_BE_SET_ASIDE = ("market_without_outcomes", "no_window", "commentary_never_read",
                     "execution_never_checked")

# One per response, ever. Measured at 3 of 7 offers in a single `gaps()` reply — one per gap,
# because `trim` caps each gap's own list and nothing capped the response — which made "shall
# we stop mentioning this?" the most repeated sentence in the product's report on its own
# evidence. "Last in the gap's own list" controlled position and did nothing about frequency.
_MAX_SET_ASIDE_OFFERS = 1


def _what_people_said_about_the_gaps(conn, ranked: dict) -> dict:
    """Apply D53's answers to the ranking (§10.2).

    Inside `_ranked_gaps` rather than inside `gaps()`, so `top_gap` sees the same list. With
    it one level up, the session-start offer fired on a gap somebody had set aside while
    `gaps()` did not show it — two surfaces describing the same library differently, which is
    D55's shape and the one this file has now hit four times.
    """
    import core

    said = store.answers_for(conn, "gap")
    kept = []
    offered_to_set_aside = 0
    for gap in ranked["gaps"]:
        answer = said.get(gap["code"])
        if answer and answer["answer"] == "not_applicable":
            # §10.2/D53 and §12.4: the set-aside is keyed on the gap's CODE, and the evidence
            # underneath it moves. "This will never be true here" is a statement about the
            # records somebody was looking at — a market whose agency no longer exists, decks
            # that are genuinely lost — and it says nothing whatever about a record uploaded
            # afterwards. Suppressed by code alone, one decision silenced the gap permanently,
            # including for evidence that did not exist when it was made: the product agreeing
            # to stop mentioning something it had not yet seen. `since` is the newest thing
            # this gap counts; when it postdates the answer, the gap comes back and says why.
            since = gap.get("since")
            recorded = answer.get("recorded_at")
            if not (since and recorded and since > recorded):
                continue
            gap["answered"] = answer
            gap["reopened"] = {
                "basis": "computed",
                "what_it_means": (
                    f"This was set aside by {answer['said_by']}, and records counted by it "
                    f"have arrived since. What was said then was about what was on file "
                    f"then; it is reported again because this evidence is new. Setting it "
                    f"aside again covers these too."),
            }
            kept.append(gap)
            continue
        if answer and answer["answer"] == "known_not_yet":
            gap["answered"] = answer
        elif (not answer and gap["code"] in _CAN_BE_SET_ASIDE
                and offered_to_set_aside < _MAX_SET_ASIDE_OFFERS
                and not gap.get("closed_by") and gap.get("next_actions")):
            offered_to_set_aside += 1
            # NOT on a gap whose own offer was de-duplicated away. `_ranked` drops an offer
            # another gap already makes and leaves `closed_by` pointing at it — so appending
            # the silencing offer afterwards made it the ONLY thing on that gap, and the
            # product's single suggestion about a real hole in the evidence was "shall we
            # stop mentioning this?". Reproduced on an ordinary five-record library: the
            # `market_without_outcomes` row offered nothing else at all.
            #
            # §10.2/D53, LAST in the gap's own list and never first: the offer that closes a
            # gap is worth more than the offer that silences it, and a product whose opening
            # suggestion is "shall we stop mentioning this?" is one that teaches people to
            # dismiss its findings. Offered at all because some gaps genuinely never close,
            # and reporting one of those forever is the permanent complaint this surface is
            # written against.
            gap["next_actions"] = actions.trim((gap.get("next_actions") or []) + [
                actions.action(
                    "Set this aside if it is never going to close",
                    "answer_gap",
                    why="Some gaps are true and permanent — a market whose agency no longer "
                        "exists will never have its results. It stays on the record either "
                        "way; it stops being ranked.",
                    consent="ask",
                    needs=["answer — `known_not_yet` (known, not yet) or `not_applicable` "
                           "(never going to be true here)",
                           "note — why", "said_by — whose call it is"],
                    code=gap["code"])])
        kept.append(gap)
    return {
        **ranked,
        "gaps": kept,
        # Recomputed, because the head of the list may have just left it. `most_valuable`
        # naming a gap that is not in `gaps` is the inconsistent pair `coverage` was fixed for.
        "most_valuable": kept[0]["code"] if kept else None,
        # Visible, not deleted. A gap somebody set aside is still a fact about this library's
        # evidence, and a judgment resting on it is still weaker for it — dropping it
        # entirely would make the product overstate its own coverage, which is the failure
        # this whole review is about.
        # Only gaps this library STILL has. A gap set aside in March and then genuinely
        # closed in June is not "set aside" any more, it is closed — and listing it says the
        # library is choosing not to look at something it no longer has, which is a worse
        # misreading than not mentioning it. `ranked["gaps"]` is the list before the answers
        # were applied, so it is the honest test of "is this still true".
        "set_aside": [{"code": code, **answer} for code, answer in sorted(said.items())
                      if answer["answer"] == "not_applicable"
                      and code in {g["code"] for g in ranked["gaps"]}],
    }


# D52: how much of the library a gap has to be about before its SIZE moves it up a step past
# a gap of a worse kind. A share, not a count, because a threshold of "ten records" is the
# wrong threshold for a library of eight and for a library of two thousand, and the ranking
# matters most on the first day, when neither the traffic nor the record count exists yet.
#
# Half. A gap about most of the library is the library's problem; a gap about a quarter of it
# is a problem in the library, and the kind ordering — which says what each gap COSTS — is
# still the better guide to which of those to fix first.
_MAGNITUDE_SHARE = 0.5


def _ranked(found: list[dict], *, library_size: int = 0) -> dict:
    """Order the gaps (§5.3), by kind and then by how much of the library each is about.

    D52: *"Gap ranking by judgments affected rather than a fixed rank per kind — one
    unindexed record currently outranks forty."* The fixed table cannot see magnitude at all,
    so the order of two gaps was decided entirely by which KIND they were — and `gaps()`'s own
    docstring says the ranking IS the design: "an unordered list of everything absent is the
    thing nobody reads". A ranking blind to size is an unordered list wearing numbers.

    Not "by judgments affected", which is what the row asks for and is not available: a
    library needs a great many saved judgments before that number means anything, so a
    ranking resting on it is wrong for every new customer — which is every customer, on the
    day the ranking matters most. RECORDS affected is what the review's own example counts,
    and it is a number this library always has.

    Kind stays the primary key and size is allowed to override exactly one step of it. Both
    extremes are wrong: kind-only is the defect, and size-only would say a market with nothing
    measured matters less than a half-indexed deck the moment one more record was affected.
    """
    for gap in found:
        gap["rank"] = _GAP_RANK[gap["code"]]
        gap["affects"] = gap.get("affects", 0)
        # Records over records. Every `affects` counts RECORDS for exactly this reason: the
        # moment one gap counts something else, `share` stops being a share and the magnitude
        # rule starts acting on a number with no meaning.
        # Deliberately NOT clamped to 1.0. A clamp would have hidden the defect this comment
        # is about — `judgments_never_reconciled` reporting `share: 3.0` — by turning a
        # meaningless number into a plausible one, which is the failure mode this whole
        # product is written against. If a share ever exceeds 1 again, something is counting
        # the wrong unit and the tests should say so rather than the code smoothing it over.
        gap["share"] = (round(gap["affects"] / library_size, 3) if library_size else 0.0)
        # One step, and it is named in the output rather than folded silently into `rank`: a
        # number that decides the order and does not appear is one nobody can argue with,
        # which is the review's complaint about this product in one sentence.
        gap["outweighs_its_kind"] = gap["share"] >= _MAGNITUDE_SHARE and gap["affects"] > 1
        # The number this list is actually SORTED by, beside the kind's own rank rather than
        # folded into it. A reader who checks that the output is ordered has to have the key
        # it is ordered on: with only `rank` published, a correctly-ordered list looked
        # unsorted, which is how a sort key nobody can see becomes a sort nobody can audit.
        gap["order"] = gap["rank"] - int(gap["outweighs_its_kind"])
        gap["rank_basis"] = "computed"
    # `-affects` second, so two gaps of the same `order` are ordered by size rather than by
    # insertion — `partly_indexed` and `no_window` have tied at 4 since they were written,
    # and the tie was broken by which line came first in this function.
    found.sort(key=lambda g: (g["order"], -g["affects"], g["code"]))

    # Two gaps can be closed by one act — a library whose single unmeasured campaign is also
    # its only LATAM campaign has two true facts and one thing to do. Offering the same call
    # twice is how somebody accepts both and appends two identical metric rows to the same
    # record. The later gap keeps the fact and points at what closes it.
    seen: dict[tuple, str] = {}
    for gap in found:
        kept = []
        for offer in gap["next_actions"]:
            key = actions.identity(offer)
            if key in seen:
                gap["closed_by"] = seen[key]
                continue
            seen[key] = gap["code"]
            kept.append(offer)
        gap["next_actions"] = kept

    return {"gaps": found, "most_valuable": found[0]["code"] if found else None}


def missing_input_for_citations(conn, cited_ids: Optional[list]) -> Optional[dict]:
    """The standing line, recomputed at save time from what the judgment actually cited.

    It lived only on `prepare_evaluation` — two calls before the verdict a user hears, with a
    note asking for it to be repeated. This project's own stated principle is that models
    mirror the shape of a tool result far more reliably than they follow instructions inside
    one, so a line delivered earlier and asked to be carried forward is the thing that gets
    dropped.

    Computed by the server rather than accepted from the model, for the same reason
    `evidence` and `provenance` are: it is a fact about what the library holds, and a model
    asserting "nothing is missing" would be asserting it about records it cannot see.
    """
    if not cited_ids:
        return most_valuable_missing_input([])
    # No library-wide read here. A `measured` set was computed on this line and never used —
    # the per-record `metric_type == "actual"` filter four lines down is the real predicate,
    # and it is about the CITED records rather than the library. A dead call to a shared
    # helper reads as though this surface agrees with the others about something it never
    # asks. (§13.1 review.)
    cited = [store.get_campaign(conn, cid) for cid in cited_ids]
    cited = [c for c in cited if c]
    if not cited:
        return most_valuable_missing_input([])
    return most_valuable_missing_input([
        {"campaign_id": c["id"], "title": c["title"],
         "metrics": metrics.measured(c["metrics"])}
        for c in cited])


def most_valuable_missing_input(evidence: list[dict]) -> Optional[dict]:
    """Of everything missing, the one thing that would most change THIS judgment.

    The review's own phrasing: "this verdict rests on zero verified outcomes; the Q2 results
    workbook would change that." One thing, not a list — several things are always missing,
    and naming them all is the behaviour being replaced. None when nothing is missing, which
    is what makes the line worth reading when it appears.
    """
    if not evidence:
        return {
            "code": "no_precedent",
            "what": "Nothing in the library is similar enough to compare this against.",
            "why_it_matters": "Without a precedent this is an opinion rather than a "
                              "judgment from your own record.",
            "next_actions": actions.trim([actions.action(
                "Add a past campaign of this kind, so there is something to judge against",
                "upload_campaign",
                why="No stored campaign resembles this proposal.",
                consent="ask", needs=["a comparable campaign"])]),
        }
    measured = [e for e in evidence if e.get("metrics")]
    if not measured:
        return {
            "code": "no_measured_precedent",
            "what": f"None of the {len(evidence)} campaigns this rests on has measured "
                    f"results on file.",
            "why_it_matters": "The comparison is to what was planned, not to what happened, "
                              "so the verdict cannot say whether any of it worked.",
            "next_actions": actions.trim([actions.action(
                f"Record what \u201c{evidence[0]['title']}\u201d actually achieved",
                "add_metrics",
                why="It is the closest precedent and has no results on file.",
                consent="ask", needs=["the numbers"],
                campaign_id=evidence[0]["campaign_id"])]),
        }
    return None
