"""
Phase 10: capturing feedback without making anyone type.

The review: *"Feedback is the only input that makes this library worth anything, and it is
currently the hardest thing to give: the user has to remember which campaigns are outstanding,
name one, and compose prose. **It should be a numbered choice.**"*

**"Open" means NEEDS SOMETHING FROM YOU** (§10.1), not a status field. A campaign is open when
the library is missing something only a person can supply — and every row says WHICH thing,
because *"what does this want from me"* is the question the user actually has.

**The numbers do not move** (§10.2). Campaigns fill 1–9, 10 is always the concluded menu and 11
is always "show more", whether three campaigns are open or nine. *"Floating positions break the
habit the menu exists to create, and a small gap in the numbering is a cheaper price than a
moving target."*

**The SERVER builds it** (§10.4). *"If the model composes the list, two users see different
orderings and different numbering for the same library — the exact class of variance the
consistency work is trying to remove."*

**A number is not an identifier** (§10.5). Between rendering a menu and answering it, a v2 can
land and "3" silently becomes a different campaign. Every selection carries the token of the
menu it was read from, and a stale one returns the refreshed menu INSTEAD OF WRITING — the
whole class of silent misfiling, made impossible rather than made unlikely.

**Numbered all the way down** (§10.3), with free text last and optional: *"the free-text box is
where 'slide 23 should be the standard' gets captured — the highest-value sentence in the whole
system — so it should be invited, never required."*
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

# Campaigns fill these; 10 and 11 are fixed whatever the count. The gap between the last
# campaign and 10 is deliberate — see the module docstring.
FIRST_ROW, LAST_ROW = 1, 9
CONCLUDED_ROW, MORE_ROW = 10, 11
# §10.2/D110: a fixed row of its own, and NOT a row among the campaigns.
#
# The rule questions were merged into the campaign list first, and that was wrong on four
# counts review put in writing. The menu's header is "which CAMPAIGN do you want to give
# feedback on" and this row has no campaign. `_row`'s contract is "name, then reason" and the
# rule row inverted the columns, so in a numbered list column one was a campaign name for
# every row but that one. It outranked `no_outcome`, so a marketer's session opened with a
# question about correction wording rather than about any of their work. And it consumed one
# of the nine slots §10.2 fixes for campaigns.
#
# Fixed numbering is what makes the menu a habit, so the answer is another fixed position
# rather than a floating row: 12 is always the rulebook questions, present or absent — and a
# row saying "0 pairs to rule on" is still the answer to "is there anything".
RULES_ROW = 12
# Every scope a menu token can name. `_address` validated against a hard-coded pair, so
# adding `rules` without adding it here would have made every rules token unparseable — and
# an unparseable token refreshes rather than erroring, so row 12 would have bounced silently
# back to the campaign menu with nothing to say why.
SCOPES = ("open", "concluded", "rules")
PER_PAGE = LAST_ROW - FIRST_ROW + 1

# Below this the library has barely started, and "this market rests on one campaign" is true
# of every row — which is `readiness`' advice, not a queue item.
_ENOUGH_TO_COMPARE = 4

# What each answer means, as the tag vocabulary this library already uses. Numbered because the
# whole point is that nobody has to recall a word, and closed because these ARE closed sets.
REACTIONS = {1: "liked", 2: "not_liked", 3: "mixed_reaction"}
PERFORMANCE = {1: "performed_well", 2: "underperformed", 3: "performed_as_expected",
               4: "no_data_yet"}

# Why a row is open, and what each is worth. Ordered by what is most valuable to CAPTURE —
# "a rejected brief whose v2 has just landed is worth more than a stub nobody has briefed" —
# rather than alphabetically or by date, which is what a list sorted for the machine's
# convenience looks like.
# (code, rank, the column the user reads, the sentence behind it). The SHORT one is the
# menu's; the review's own column is "no outcome recorded" — nineteen characters — and a
# semicolon-joined pair of clauses is not a menu column. A model handed 140 characters and
# told to read the rows out will compress them, and compressing is the model composing the
# list, which §10.4 forbids.
_REASONS = (
    # D20: something is BROKEN and a person has to act. Above every missing tag, because a
    # missing tag is a library that knows less and this is a library that cannot do what it
    # says. A warning lives for one response; the thing most wanting somebody's attention was
    # the thing the product forgot fastest.
    ("needs_attention", 0, "something needs attention",
     "a warning about this record is still outstanding"),
    ("no_verdict_on_new_version", 1, "no verdict on this version",
     "a new version arrived and nobody has said whether it fixed anything"),
    ("judgment_never_reconciled", 2, "judgment never checked",
     "the library judged this and has never been checked against what happened"),
    # D110: two wordings that may be one rule. Not about a campaign at all — the only row here
    # that is not — and it sits with the two above it because all three are questions the
    # library asked and nobody answered. Its cost is the highest of the three and the least
    # visible: a rule stated three ways in three markets counts as three rules, so each stays
    # at one market and NONE of them ever recurs. The library quietly cannot learn the thing
    # the client has said three times.
    ("rules_may_be_the_same", 3, "two rules may be the same",
     "two corrections are worded alike, and until somebody says whether they are one rule "
     "each counts separately and neither ever recurs across markets"),
    ("no_outcome", 4, "no outcome recorded",
     "nobody has said what they made of the work or how it performed"),
    ("no_performance_tag", 5, "no performance recorded", "nobody has said how it performed"),
    ("targets_only", 6, "targets only, no actuals",
     "targets are on file and the actuals never arrived"),
    ("no_reaction_tag", 7, "no reaction recorded",
     "nobody has said what they made of the work"),
    ("never_briefed", 8, "stub, never briefed", "a stub nobody has briefed"),
    # D87: the numbers are on file and the claim beside them is still somebody's impression,
    # so §6.4 reports "could not be checked" on a library that holds the evidence. Only a
    # person can say the claim rests on those numbers.
    ("unverified_claim", 9, "results on file, claim not verified",
     "measurements are on file and the performance claim is still an impression"),
    # D70: one campaign carrying every judgment about a market. The similarity score looks the
    # same whether it came from one example or ten, and only a person can supply the second.
    ("single_example", 10, "this market rests on it alone",
     "this market has one campaign in it, so every judgment about the market rests on it"),
)
_RANK = {code: rank for code, rank, _short, _why in _REASONS}
_SHORT = {code: short for code, _rank, short, _why in _REASONS}
_WHY = {code: why for code, _rank, _short, why in _REASONS}


def queue(conn, *, scope: str = "open", page: int = 1) -> dict:
    """The numbered menu (§10.1, §10.2, §10.4).

    `scope="concluded"` is option 10 — *"the same menu over concluded records"* — and `page`
    is option 11, which keeps its position so going back and forth does not reshuffle anything.
    """
    import store

    if scope == "rules":
        return _rules_menu(conn)
    rows = _open_rows(conn) if scope == "open" else _concluded_rows(conn)
    # §10.2/D110: counted, not mixed in. They belong to row 12.
    pairs = len(_rule_questions(conn)) if scope == "open" else 0
    pages = max(1, -(-len(rows) // PER_PAGE))
    page = max(1, min(page, pages))
    shown = rows[(page - 1) * PER_PAGE:page * PER_PAGE]
    numbered = [{"number": FIRST_ROW + n, **row} for n, row in enumerate(shown)]
    if not numbered and not pairs:
        return {
            "status": "nothing_to_check", "basis": "computed", "scope": scope,
            "rows": [], "waiting": 0, "page": page, "pages": pages,
            "what_it_means": (
                "Nothing is waiting on you. Every campaign in the library carries what a "
                "person has to supply — how the work landed, how it performed, and the "
                "results behind it."
                if scope == "open" else
                "No concluded campaign is missing anything a person has to supply."),
        }
    numbered += [
        {"number": CONCLUDED_ROW, "action": "concluded",
         "title": "Add feedback to a concluded campaign"},
        {"number": MORE_ROW, "action": "more",
         "title": ("Not here — show more" if pages > page else "Not here — start again")},
    ]
    if pairs:
        numbered.append({
            "number": RULES_ROW, "action": "rules",
            "title": f"Two rules may be the same — {pairs} pair(s) to rule on"})
    return {
        "status": "checked", "basis": "computed", "scope": scope,
        "rows": numbered, "waiting": len(rows), "page": page, "pages": pages,
        # §10.5. The token is over the ROWS as rendered, so anything that would change what a
        # number means changes the token — a new campaign, a tag recorded elsewhere, a verdict
        # saved in another window.
        "menu_token": _token(conn, scope, page, numbered),
        "what_it_means": _sentence(rows, numbered, scope, page, pages, rules=pairs),
    }


def _sentence(rows: list, numbered: list, scope: str, page: int, pages: int,
              rules: int = 0) -> str:
    # `rows` is campaigns, all of it — see `_open_rows`. The first version of §10.2/D110 mixed
    # rule questions in here and this sentence then announced "4 campaign(s) are waiting" for
    # three campaigns and one rule pair. Splitting the count fixed the symptom; moving the
    # rows to their own scope fixed the cause, and a split kept afterwards would be a filter
    # guarding a case that can no longer arise — which reads as a live guard and is not one.
    said = (f"{len(rows)} campaign(s) are waiting on something only a person can supply. "
            f"Read each row's `why` aloud with it — that is the question the user actually "
            f"has. Offer the numbers and nothing else: answering should never need typing."
            if scope == "open" else
            f"{len(rows)} concluded campaign(s) are missing something a person has to "
            f"supply.")
    if rules:
        said += (f" Row {RULES_ROW} is separate: {rules} pair(s) of corrections are worded "
                 f"alike, and whether each pair is one rule is a question only somebody who "
                 f"knows this client's vocabulary can answer.")
    if pages > 1:
        said += f" Showing {page} of {pages} pages; {MORE_ROW} pages on."
    return said


def _open_rows(conn) -> list:
    """Every campaign needing something from a human, most valuable first (§10.1)."""
    import store

    superseded = store.get_superseded_campaign_ids(conn)
    rows = []
    for record in store.list_campaigns(conn):
        if record["id"] in superseded or record.get("record_type") == "reference":
            continue
        full = store.get_campaign(conn, record["id"])
        reasons = _what_it_needs(conn, full)
        if not reasons:
            continue
        rows.append(_row(full, reasons))
    # §10.2/D110's rows are NOT merged here — they are row 12's, and `queue` builds that.
    # Merging them put a row with no campaign into a list whose header asks which campaign,
    # inverted the column order for that one row, and let it outrank every real campaign.
    # Rank first, then title — a stable second key, so two rows of equal worth do not swap
    # places between two renderings of the same library (§10.4).
    rows.sort(key=lambda r: (r["_rank"], r["title"].casefold(), r["campaign_id"]))
    return [{k: v for k, v in row.items() if k != "_rank"} for row in rows]


def _rules_menu(conn) -> dict:
    """Row 12: the rulebook questions, numbered the same way (§10.2/D110).

    A second SCOPE, not a second list beside the first — reached by a number from the menu
    everyone already has, the way row 10 reaches the concluded records. That keeps "what is
    waiting on me" one entry point while keeping the campaign menu about campaigns.

    The full wordings, not the 40-character prefixes the merged version showed. Both texts
    matched BECAUSE their first forty characters are near-identical, so a truncated pair
    showed the user two visually identical strings and asked whether they were the same.
    """
    rows = _rule_questions(conn)
    numbered = [{"number": FIRST_ROW + n, **row} for n, row in enumerate(rows[:PER_PAGE])]
    if not numbered:
        return {
            "status": "nothing_to_check", "basis": "computed", "scope": "rules",
            "rows": [], "waiting": 0, "page": 1, "pages": 1,
            "what_it_means": (
                "No two corrections on file are worded alike enough to be worth asking "
                "about. That is a real answer: it does not mean the rulebook is tidy, only "
                "that nothing currently looks like a duplicate."),
        }
    numbered.append({"number": MORE_ROW, "action": "more",
                     "title": "Back to the campaigns waiting on you"})
    return {
        "status": "checked", "basis": "computed", "scope": "rules",
        "rows": numbered, "waiting": len(rows), "page": 1, "pages": 1,
        "menu_token": _token(conn, "rules", 1, numbered),
        "what_it_means": (
            f"{len(rows)} pair(s) of corrections are worded alike. Only somebody who knows "
            f"this client's vocabulary can say whether each pair is one rule — and until "
            f"they do, each wording counts separately, so a rule the client has stated "
            f"twice never recurs across markets and never becomes standing. Read both "
            f"wordings out in full: they matched because their openings are near-identical, "
            f"so a summary of either is a summary of both."),
    }


def _rule_questions(conn) -> list:
    """Near-duplicate corrections nobody has ruled on (§10.2/D110).

    The question is asked today by `note_correction`, on the write that raised it, and
    NOWHERE else — so it is put once, to whoever happened to be uploading a deck, and a
    silence is indistinguishable from a "no". A library where nobody was looking at that
    moment accumulates near-duplicates that each stay at one market and never graduate, and
    nothing ever says so.

    `asked` is the flag `resolve_correction` sets, so a pair somebody has ruled on drops out
    — a queue that keeps asking a settled question teaches people to stop reading it, and the
    unsettled question then goes unread with it.
    """
    import corrections as corrections_module
    import store

    # ONE pass over the corrections, pairing in memory. `_looks_like` re-reads and re-tokenises
    # every correction on file, so calling it once per correction was O(n²) over the WHOLE
    # table — and `ingest_campaign` calls `waiting()` on every upload, which calls this.
    # Measured: 100 corrections 0.37s, 200 corrections 1.74s, 400 corrections 8.39s, for a
    # single upload. At 400 the offer alone spent a third of `TOOL_TIME_BUDGET_SECONDS`, on a
    # library that is not large — an agency accumulates corrections faster than campaigns.
    live = [row for row in store.corrections(conn)
            if not row.get("asked")
            and row["status"] not in ("merged", "ignored", "expected")]
    # Read once per correction rather than once per comparison — that is the whole of the
    # speed fix. WHAT a reading is, and what makes two of them close, stays in `corrections`:
    # the first version of this re-derived the "a prohibition and its permission are not the
    # same rule" check here, which is two implementations of one rule, and mutation showed
    # nothing would have noticed if this copy had been deleted.
    readings = {row["id"]: corrections_module._reading(row["text"]) for row in live}

    def resembling(entry):
        best, score = None, 0.0
        for other in live:
            if other["id"] == entry["id"]:
                continue
            overlap = corrections_module.resembles(readings[entry["id"]],
                                                   readings[other["id"]])
            if overlap > score:
                best, score = other, overlap
        return best if corrections_module.is_close(score) else None

    rows = []
    seen: set = set()
    answered = {row["id"] for row in store.corrections(conn) if row.get("asked")}
    for entry in live:
        similar = resembling(entry)
        if not similar:
            continue
        # EITHER side having been asked settles the pair, because the question is symmetric.
        # `different_rule` — the one answer of §8.2's three that leaves both corrections live
        # and provisional — marks only the correction it was called on, so checking just this
        # side asked the same question straight back from the other wording. Answering it did
        # not make it go away, which is D110's own complaint reproduced inside its fix.
        if similar["id"] in answered:
            continue
        # One row per PAIR. Both wordings resemble each other, so an unguarded loop asks the
        # same question twice with the two texts swapped — which reads as two problems.
        pair = tuple(sorted((entry["id"], similar["id"])))
        if pair in seen:
            continue
        seen.add(pair)
        rows.append({
            "action": "resolve_correction",
            "correction_id": entry["id"],
            "same_as": similar["id"],
            # Both wordings, because a row saying "two rules may be the same" without saying
            # WHICH two is a question nobody can answer from the menu — the state it is being
            # lifted out of.
            "title": f"“{entry['text']}” / “{similar['text']}”",
            "needs": ["rules_may_be_the_same"],
            "reason": _SHORT["rules_may_be_the_same"],
            "why": _WHY["rules_may_be_the_same"],
            # FULL wordings, not 40-character prefixes. They matched BECAUSE their openings
            # are near-identical, so truncating showed the user two visually identical
            # strings and asked whether they were the same thing. This row is on its own
            # menu now, so it has the width.
            "line": f"“{entry['text']}”  /  “{similar['text']}”",
            "_rank": _RANK["rules_may_be_the_same"],
        })
    return rows


# How many people already on file to offer. Past a handful this stops being a numbered
# choice and becomes a directory somebody has to read.
_RECENT_VOICES = 4


def _whose_opinion(conn) -> Optional[dict]:
    """§11.3's numbered question, or None when there is nobody to offer.

    The PEOPLE ALREADY IN THE LIBRARY, not just the operator. The first version offered the
    operator alone as option 1 with "someone else" as a typing task — and §11.3's whole
    premise is that the person typing is usually NOT the person whose view it is, so that
    optimised the ergonomics of the failure mode. In an agency the queue is worked by one
    person recording what clients and colleagues said.

    Everyone who has recorded a view is known, so they can be numbered too: fewer keystrokes
    than before for the colleague case, and the operator stops being the cheap default.
    """
    import identity
    import store

    names: list = []
    for voice in store.recent_voices(conn, limit=_RECENT_VOICES):
        if voice not in names:
            names.append(voice)
    operator = (identity.captured_by().get("display_name") or "").strip()
    if operator and operator not in names:
        names.append(operator)
    if not names:
        return None
    options = {n: name for n, name in enumerate(names, start=1)}
    options[len(options) + 1] = "Someone else — who?"
    return {
        "field": "said_by", "ask": "Whose view is this?",
        "options": options,
        # Every other question here is answered with a number the SERVER resolves. This one is
        # a free string on the tool, and the payload's own summary says "offering the
        # numbers" — so nothing told the model that picking the last option means asking for
        # a name and sending THAT.
        "answer_with": "a name",
        "why": ("Send the NAME, not the number — this is the one question on this menu the "
                "server cannot resolve for you. Recorded either way; the difference is "
                "whether the person who holds this view is the person entering it, and two "
                "years on that is the thing nobody can reconstruct."),
    }


def _what_would_close_it(record: dict, row: dict) -> list:
    """The tool that answers a reason no number can (§10.3, D116).

    A reaction and a performance tag are numbers. A missing verdict, an unreconciled judgment
    and absent actuals are not — they need `prepare_evaluation`, `reconcile_evaluation` and
    `add_metrics`, and a menu that lists them without naming those is a list of complaints.
    """
    import actions

    needs = set(row.get("needs") or [])
    offers = []
    if "no_verdict_on_new_version" in needs:
        offers.append(actions.action(
            f"Judge \u201c{record['title'][:36]}\u201d against what came before it",
            "prepare_evaluation",
            why="A new version arrived and nobody has said whether it fixed what was wrong. "
                "No number on this menu can answer that.",
            consent="ask", subject_title=record["title"], campaign_id=record["id"]))
    if "judgment_never_reconciled" in needs:
        offers.append(actions.action(
            f"Check what the library said about \u201c{record['title'][:36]}\u201d against what "
            f"happened",
            "reconcile_evaluation",
            why="The results are in and the judgment has never been tested against them.",
            consent="ask"))
    if "unverified_claim" in needs:
        offers.append(actions.action(
            f"Mark \u201c{record['title'][:36]}\u201d's performance claim as verified",
            "update_campaign",
            why="The measurements are on file and the claim beside them still reads as "
                "somebody's impression, so every check that weighs verified evidence reports "
                "that it could not be checked.",
            consent="ask", campaign_id=record["id"],
            tags=[{"value": v, "source": "verified"}
                  for v in sorted({t["value"] for t in record.get("tags") or []
                                   if t.get("value") in PERFORMANCE.values()})]))
    if "single_example" in needs:
        offers.append(actions.action(
            f"Add a second campaign from {record.get('market') or record.get('region')}",
            "upload_campaign",
            why="Every judgment about this market rests on one campaign, and a similarity "
                "score looks the same whether it came from one example or ten.",
            consent="ask", needs=["another campaign that ran in that market"],
            market=record.get("market") or record.get("region")))
    if "targets_only" in needs:
        offers.append(actions.action(
            f"Record what \u201c{record['title'][:36]}\u201d actually achieved",
            "add_metrics",
            why="Targets are on file and the actuals never arrived, so nothing here is a "
                "result yet.",
            consent="ask", needs=["the results themselves"], campaign_id=record["id"]))
    return actions.trim(offers)


def _row(record: dict, reasons: list) -> dict:
    """One line of the menu, rendered by the SERVER (§10.2, §10.4).

    The `line` is the whole point: if the server builds the menu it should build the line, not
    hand the model a title, a market and a paragraph and hope two operators format them the
    same way. The structured fields stay for anyone who wants them.
    """
    codes = [code for code, _ in reasons]
    where = (record.get("market") or record.get("region") or "").strip()
    name = f"{where} - {record['title']}" if where else record["title"]
    # "(v2 received)" — the annotation the review's own row 1 carries, and `supersedes` is in
    # hand right here.
    if record.get("supersedes"):
        name += " (v2 received)"
    reason = _SHORT[min(codes, key=lambda c: _RANK[c])]
    return {
        "campaign_id": record["id"], "title": record["title"], "market": where or None,
        "needs": codes,
        "reason": reason,
        "why": "; ".join(_WHY[code] for code in codes),
        "line": f"{name}   {reason}",
        "_rank": min(_RANK[code] for code in codes),
    }


def _concluded_rows(conn) -> list:
    """Option 10: the same menu over records that have already CONCLUDED.

    A different population, not a filter on the first: the open menu is ordered by what is
    most valuable to capture and leads with re-briefs awaiting a verdict, which are by
    definition not concluded. Somebody who picks 10 is saying "I want to talk about something
    that has already run", and handing them the same rows in the same order would be ignoring
    the question.
    """
    import store

    superseded = store.get_superseded_campaign_ids(conn)
    open_by_id = {row["campaign_id"]: row for row in _open_rows(conn)}
    rows = []
    for record in store.list_campaigns(conn):
        if (record.get("status") != "concluded" or record["id"] in superseded
                or record.get("record_type") == "reference"):
            continue
        # EVERY concluded record, not only the ones missing something. Somebody choosing 10 is
        # saying "I want to talk about something that has already run" — often to add the
        # sentence that is worth more than any tag, on a campaign that is otherwise complete.
        # Filtering to the open ones would answer a question they did not ask.
        where = (record.get("market") or record.get("region") or "").strip()
        name = f"{where} - {record['title']}" if where else record["title"]
        rows.append(open_by_id.get(record["id"]) or {
            "campaign_id": record["id"], "title": record["title"], "market": where or None,
            "needs": [], "reason": "nothing outstanding — add a note",
            "why": "nothing outstanding — add a note if you have one",
            "line": f"{name}   nothing outstanding — add a note",
        })
    # Complete records FIRST. Somebody picking 10 is usually here to add the sentence that is
    # worth more than any tag, on a campaign that is otherwise finished — sorting those to the
    # bottom answers the question they did not ask.
    rows.sort(key=lambda r: (bool(r["needs"]), r["title"].casefold(), r["campaign_id"]))
    return rows


def _what_it_needs(conn, record: dict) -> list:
    """What this campaign is missing that a PERSON has to supply (§10.1).

    Not a status field. Every entry here is something no amount of computation can produce:
    whether the work was liked, whether it performed, what the numbers actually were, whether
    a revision fixed what was wrong.
    """
    import store

    needs = []
    values = {t.get("value") for t in record.get("tags") or []}
    if store.open_notices(conn, record["id"]):
        needs.append(("needs_attention", _WHY["needs_attention"]))
    if record.get("supersedes") and not store.evaluations_for(conn, record["id"]):
        needs.append(("no_verdict_on_new_version", _WHY["no_verdict_on_new_version"]))
    if record.get("record_type") == "stub":
        needs.append(("never_briefed", _WHY["never_briefed"]))
        return needs
    # A campaign that has not run cannot have performed, and asking is a question with no true
    # answer. `readiness` draws this line explicitly and the queue was not drawing it at all —
    # so a store opening that had not happened was listed as missing its outcome.
    has_run = record.get("status") == "concluded"
    wants_reaction = not values & set(REACTIONS.values())
    wants_performance = has_run and not values & set(PERFORMANCE.values())
    if wants_reaction and wants_performance:
        # One reason, the way the review's own column reads it.
        needs.append(("no_outcome", _WHY["no_outcome"]))
    elif wants_reaction:
        needs.append(("no_reaction_tag", _WHY["no_reaction_tag"]))
    elif wants_performance:
        needs.append(("no_performance_tag", _WHY["no_performance_tag"]))
    has_actual = record.get("has_actual_metrics")
    has_target = any(m["metric_type"] in ("target", "predicted")
                     for m in record.get("metrics") or [])
    if has_target and not has_actual:
        needs.append(("targets_only", _WHY["targets_only"]))
    if store.unreconciled_evaluation_id(conn, record["id"]) and has_actual:
        needs.append(("judgment_never_reconciled", _WHY["judgment_never_reconciled"]))
    if has_actual and any(t.get("source") == "stated" for t in record.get("tags") or []
                          if t.get("value") in PERFORMANCE.values()):
        needs.append(("unverified_claim", _WHY["unverified_claim"]))
    # Only when nothing else about this record is outstanding, and only once the library is
    # past its first few records. A brand-new library has one campaign per market BY
    # DEFINITION, so raising it on every row would make the whole queue read "add more
    # campaigns" — advice `readiness` already gives better, and the wallpaper this ordering
    # exists to avoid.
    if not needs and _alone_in_its_market(conn, record):
        needs.append(("single_example", _WHY["single_example"]))
    return needs


def _alone_in_its_market(conn, record: dict) -> bool:
    """One campaign carrying every judgment about a market (§10.1/D70)."""
    import store

    where = (record.get("market") or record.get("region") or "").strip().casefold()
    if not where:
        return False
    superseded = store.get_superseded_campaign_ids(conn)
    everything = [c for c in store.list_campaigns(conn)
                  if c["id"] not in superseded
                  and c.get("record_type") not in ("reference", "stub")]
    if len(everything) < _ENOUGH_TO_COMPARE:
        return False
    siblings = [c for c in everything
                if (c.get("market") or c.get("region") or "").strip().casefold() == where]
    return len(siblings) == 1


def _token(conn, scope: str, page: int, numbered: list) -> str:
    """A fingerprint of the menu AS RENDERED, carrying its own address (§10.5).

    Two things it must do. It has to change whenever anything that would alter what "3" means
    alters — including what a row is ASKING FOR: the first version hashed only
    [number, campaign_id, action], so a row that gained a tag but stayed open kept its token,
    and an answer composed before that landed wrote `liked` over somebody's `not_liked`.

    And it has to say which menu it is. Validation used to re-derive every page of every scope
    looking for a match — 8.8 seconds on five hundred campaigns, with the STALE path the
    slowest of all, which is the common path because any change invalidates every open menu.
    Pages past the fiftieth were unreachable and a valid token there was reported as "the
    library has changed", which is a false statement of fact.
    """
    # `correction_id`/`same_as` too, for §10.2/D110's rows. Those carry no `campaign_id`, so
    # two DIFFERENT pairs landing on the same number produced the same token — one settled, the
    # next arriving in its place, and "3" silently meaning a different question with the key
    # that is supposed to make that impossible. The row shape changed and the fingerprint did
    # not; a token that identifies rows by a field one row shape does not have identifies
    # nothing about that row.
    payload = json.dumps(
        [scope, page, [[r["number"], r.get("campaign_id"), r.get("action"),
                        r.get("correction_id"), r.get("same_as"),
                        sorted(r.get("needs") or [])] for r in numbered]],
        sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"menu_{scope}_{page}_{digest}"


def _address(menu_token: str) -> Optional[tuple]:
    """The scope and page a token names, so checking it is one derivation rather than a
    search over every page of every scope."""
    parts = str(menu_token or "").split("_")
    if len(parts) != 4 or parts[0] != "menu" or parts[1] not in SCOPES:
        return None
    try:
        return parts[1], int(parts[2])
    except ValueError:
        return None


def choose(conn, *, menu_token: str, choice: int) -> dict:
    """Take a number from the menu the user was actually looking at (§10.5, §10.3).

    A stale token returns the refreshed menu INSTEAD OF WRITING. Ordinal selection is fragile —
    if anything changed between rendering and answering, "3" silently means a different
    campaign and the feedback lands on the wrong record. Cheap to check, and it makes a whole
    class of silent corruption impossible rather than unlikely.
    """
    menu = _menu_for(conn, menu_token)
    if menu is None:
        return _refreshed(conn, "the library has changed since that menu was shown")
    return _ask_about(conn, menu, choice)


def _ask_about(conn, menu: dict, choice: int) -> dict:
    import actions
    import store

    row = next((r for r in menu["rows"] if r["number"] == choice), None)
    if row is None:
        raise ValueError(
            f"{choice} is not on this menu. The numbers offered were "
            f"{', '.join(str(r['number']) for r in menu['rows'])} — read them back rather "
            f"than guessing, because a number nobody offered cannot be matched to a record.")
    if row.get("action") == "concluded":
        # Not forced to `checked`: an empty concluded list IS `nothing_to_check`, and
        # overwriting the status put an empty menu behind a word that means "we looked and
        # here it is".
        return queue(conn, scope="concluded")
    if row.get("action") == "rules":
        # §10.2/D110: row 12, a scope of its own rather than a row among the campaigns.
        return queue(conn, scope="rules")
    if row.get("action") == "more":
        if menu["scope"] == "rules":
            # The rules menu has one page and its last row is the way back, not "show more".
            return queue(conn, scope="open")
        # Wraps. On the last page the row already reads "start again" and did not: `page + 1`
        # clamped back to the same page, so somebody at the end of a long list had no way home
        # from inside the menu.
        following = menu["page"] + 1 if menu["page"] < menu["pages"] else 1
        return queue(conn, scope=menu["scope"], page=following)
    if row.get("action") == "resolve_correction":
        # §10.2/D110. Not a reaction question — the closed set here is §8.2's three answers,
        # and `corrections` already builds them. Rebuilding them beside it is how one surface
        # comes to offer two answers and the other three, which is the drift this file has
        # been bitten by four times.
        import corrections as corrections_module

        entry = corrections_module.describe(conn, row["correction_id"])
        similar = corrections_module.describe(conn, row["same_as"])
        # The token already catches somebody answering this pair in another window — it is
        # computed over the rows, and an answered pair is not a row any more. This catches
        # what the token cannot: a correction that is GONE, which `delete_campaign`'s cascade
        # can do without changing anybody else's row. Narrow on purpose, and second.
        if not entry or not similar:
            return _refreshed(conn, "one of those rules is no longer on file")
        return {
            "status": "asking", "scope": menu["scope"], "menu_token": menu["menu_token"],
            "correction_id": entry["id"], "same_as": similar["id"],
            "texts": [entry["text"], similar["text"]],
            "ask": "Are these the same rule?",
            "next_actions": corrections_module._offers(entry["id"], entry["text"], similar),
            "what_it_means": (
                "Only somebody who knows the client's vocabulary can answer this, which is "
                "why it is asked rather than decided. Folded together, both markets count "
                "towards one rule; left apart, each wording is its own rule and neither ever "
                "recurs across markets, so a rule the client has stated twice never becomes "
                "standing."),
        }

    record = store.get_campaign(conn, row["campaign_id"])
    # From the ROW's own reasons, not recomputed from tags. Recomputing asked a stub how it
    # performed one sentence after saying nobody had briefed it, and asked an unrun store
    # opening how it went.
    needs = set(row.get("needs") or [])
    # A row with nothing outstanding can only have come from option 10, which exists to add
    # feedback to a campaign that has already run — often a second person's view, or the
    # sentence that is worth more than any tag. Asking nothing there would answer a question
    # they did not ask. (§11.5 will keep BOTH sides of a disagreement properly; until then the
    # later answer stands and carries its author's name.)
    if not needs:
        needs = {"no_outcome"}
    questions = []
    # §11.3, FIRST: whose opinion is this? "The field most products miss", and they miss it
    # because it only matters later — on the day the note is written everybody knows who was
    # in the room, and two years on the record says a name with nothing to say whether that
    # person held the view or merely typed it.
    #
    # Numbered, because §10.3 says answering should never need typing and the operator is by
    # far the commonest answer — the server already knows their name, so making them spell it
    # was the one bit of typing this menu had left. Offered only when the operator CONFIGURED
    # a name: `os_user` is an account, and "1 = nbhatt" would make the easy answer wrong.
    whose = _whose_opinion(conn)
    if whose:
        questions.append(whose)
    if {"no_outcome", "no_reaction_tag"} & needs:
        questions.append({"field": "reaction", "ask": "How did the work land?",
                          "options": dict(REACTIONS)})
    if {"no_outcome", "no_performance_tag"} & needs:
        questions.append({"field": "performance", "ask": "How did it perform?",
                          "options": dict(PERFORMANCE)})
    if "never_briefed" in needs:
        # What this record needs is a BRIEF, and no number supplies one. Offering 1/2/3 here
        # would be asking a question whose answer cannot be given.
        return {
            "status": "asking", "basis": "computed", "campaign_id": record["id"],
            "title": record["title"], "why_it_is_open": row.get("why", ""),
            "questions": [], "menu_token": menu["menu_token"], "choice": choice,
            "free_text": {"field": "note", "required": False,
                          "prompt": "Anything specific? (type it, or skip)",
                          "why": "Even without a brief, what somebody knows about it is "
                                 "worth keeping."},
            "what_it_means": (
                f"\u201c{record['title']}\u201d is a stub nobody has briefed, so there is nothing "
                f"to react to yet. What it needs is the brief itself."),
            "next_actions": actions.trim([actions.action(
                f"Add the brief for \u201c{record['title'][:40]}\u201d",
                "upload_campaign",
                why="A stub is a name with no content: nothing can be compared against it, "
                    "and no feedback about the work exists until the work does.",
                consent="ask", needs=["its deck, or the brief in your own words"],
                title=record["title"])]),
        }
    return {
        "status": "asking", "basis": "computed",
        "campaign_id": record["id"], "title": record["title"],
        "why_it_is_open": row.get("why", ""),
        "questions": questions,
        # Last, and optional. This is where "slide 23 should be the standard" gets captured —
        # the highest-value sentence in the whole system — so it is invited and never demanded.
        "free_text": {
            "field": "note", "required": False,
            "prompt": "Anything specific? (type it, or skip)",
            "why": ("A sentence like “slide 23 should be the standard” is worth more "
                    "than any tag, and it only ever arrives if somebody is asked for it "
                    "without being made to type."),
        },
        # The token AND the number travel with the answer: the write takes the number back,
        # never an id, so there is no way to name a row that was not on this menu.
        "menu_token": menu["menu_token"], "choice": choice,
        # D116, from the queue's own rows. Ranks 1, 2 and 5 are closed by tools the menu never
        # named — so the highest-ranked reason in the whole library could not be cleared from
        # the surface built to clear things.
        **({"next_actions": _what_would_close_it(record, row)}
           if _what_would_close_it(record, row) else {}),
        "what_it_means": (
            f"Ask these in order, offering the numbers. {row.get('why', '')}. Then invite the "
            f"free text and accept a skip without pressing."),
    }


def record(conn, *, menu_token: str, choice: int, said_by: str,
           reaction: Optional[int] = None, performance: Optional[int] = None,
           note: Optional[str] = None, role: Optional[str] = None) -> dict:
    """Write down what somebody said (§10.3, §10.5).

    Takes the NUMBER the user pressed, resolved server-side against the menu that token names.
    It used to take a `campaign_id` — a free string the model supplied from its own context —
    and check only that the token matched SOME current menu. So a valid token plus any id the
    model had lying about wrote feedback to a record nobody had chosen, and returned
    "recorded". The token proved the menu was fresh and proved nothing about the row, which is
    the exact silent misfiling §10.5 exists to make impossible.
    """
    import store

    # Checked before the token, because a missing name is the caller's mistake to fix and a
    # stale token is the library's — sending them to re-read the menu when the real problem is
    # a blank field is a loop that cannot close.
    _check_the_name(said_by)
    menu = _menu_for(conn, menu_token)
    if menu is None:
        return _refreshed(conn, "the library changed while this was being answered, so "
                                "nothing was written")
    row = next((r for r in menu["rows"] if r["number"] == choice), None)
    if row is None or not row.get("campaign_id"):
        raise ValueError(
            f"{choice} is not a campaign on that menu. The campaigns offered were "
            f"{', '.join(str(r['number']) for r in menu['rows'] if r.get('campaign_id'))}.")
    record = store.get_campaign(conn, row["campaign_id"])
    if record is None:
        return _refreshed(conn, "that record is no longer in the library")
    if reaction is None and performance is None and not (note or "").strip():
        raise ValueError(
            "nothing was given to record. Ask the questions the menu returned and pass the "
            "numbers back, or pass a note — a call with neither is not feedback.")
    # Only what this row actually asked. Accepting a performance answer on a row that asked
    # only for a reaction wrote `no_data_yet` beside a `performed_well` that measurements had
    # earned, and there is no reading of the conversation in which the user said both.
    asked_for = set(row.get("needs") or []) or {"no_outcome"}
    if performance is not None and not {"no_outcome", "no_performance_tag"} & asked_for:
        raise ValueError(
            f"\u201c{record['title']}\u201d was not asked how it performed — that is already on "
            f"file, or it has not run. Pass only the answers to the questions the menu "
            f"returned.")
    if reaction is not None and not {"no_outcome", "no_reaction_tag"} & asked_for:
        raise ValueError(
            f"\u201c{record['title']}\u201d was not asked how the work landed — that is already "
            f"on file. Pass only the answers to the questions the menu returned.")

    said_at = _now_iso()
    tags = list(record.get("tags") or [])
    added = []
    for answer, vocabulary in ((reaction, REACTIONS), (performance, PERFORMANCE)):
        if answer is None:
            continue
        if answer not in vocabulary:
            raise ValueError(
                f"{answer} is not one of the numbers offered "
                f"({', '.join(f'{n} {v}' for n, v in vocabulary.items())}).")
        value = vocabulary[answer]
        # REPLACES within its family. Appending produced `liked` AND `not_liked` on one
        # record — which the queue then read as answered and closed forever — and there is no
        # reading of one person's answer to one question in which both are true.
        #
        # §11.5 is now the other half, and it is a different thing: two PEOPLE disagreeing is
        # not a contradiction, it is the most informative row this library can hold. The tag
        # blob stays what it is — the latest view per axis, which is what filtering and
        # search need — and `reactions` below keeps everyone's, so `get_campaign` can say
        # they disagreed rather than reporting whichever was written last as the answer.
        tags = [t for t in tags if t.get("value") not in set(vocabulary.values())]
        # `stated`, always — §2.3's rule reaches the menu too, and this is the easiest place
        # in the product to type an impression. WITH the name: a tag nobody's name is against
        # cannot be weighed against a contradicting one later, and the field was being
        # demanded and then dropped while the response claimed it had been kept.
        tags.append({"value": value, "source": "stated",
                     "said_by": said_by.strip(), "said_at": said_at})
        added.append(value)
    if added:
        store.update_campaign(conn, record["id"], tags=tags)
    # §11.5: appended, never replaced. This is where opinions actually arrive, so a reaction
    # that reached the blob and not the table would be a view the library silently lost.
    for value in added:
        store.record_reaction(conn, campaign_id=record["id"], value=value, source="stated",
                              said_by=said_by.strip(), said_at=said_at, role=role)
    # §11.1/§11.3: the account beside the name, on the surface built to make answering easy —
    # which is exactly where "whoever was at the keyboard" and "whose view this is" are most
    # likely to be the same string and most likely to be different people.
    store.record_authorship(conn, subject_kind="feedback", subject_key=record["id"],
                            on_behalf_of=said_by, role=role)
    offers = []
    if (note or "").strip():
        # `predicted`, ALWAYS. Keyed off `performance` it made "no data yet" — a truthy 4 —
        # write an ACTUAL metric row, so one sentence told the library the campaign had
        # measured results: it left the queue, `readiness` called the library measured, and
        # §2.3's `verified` gate opened. A sentence is not a measurement.
        store.add_feedback_note(conn, campaign_id=record["id"], note=note.strip(),
                                said_by=said_by.strip(), said_at=said_at)
        # §8.6 is what turns "slide 23 should be the standard" into a standing rule once it
        # recurs and somebody confirms it. The product already offers that for words it
        # SCRAPED out of a deck; this is the one place a person is deliberately asked for the
        # sentence, and the provenance here is better than the scraped path ever gets.
        # Offered, never written: §8.6 is explicit that the text stays the client's own words.
        import actions
        offers = actions.trim([actions.action(
            f"Record \u201c{note.strip()[:44]}\u201d as a standing correction",
            "note_correction",
            why="A sentence like this is the most valuable thing this library holds. Recorded "
                "as a correction it can recur across markets and become a rule; left here it "
                "is a note on one campaign.",
            consent="ask",
            text=note.strip(),
            provenance=f"{said_by.strip()} on \u201c{record['title']}\u201d, {said_at[:10]}",
            campaign_id=record["id"])])
    return {
        "status": "recorded", "basis": "stated", "campaign_id": record["id"],
        "recorded": added, "said_by": said_by.strip(), "said_at": said_at,
        "note_kept": bool((note or "").strip()),
        "what_it_means": (
            f"Recorded against {record['title']}, attributed to {said_by.strip()}. "
            + (f"Tagged {', '.join(added)} — `stated`, because somebody said so rather than "
               f"anything having been measured. " if added else "")
            + ("Their sentence is on the record in their words. " if (note or "").strip()
               else "")
            + "The menu below is what is still waiting."),
        **({"next_actions": offers} if offers else {}),
        "next": queue(conn),
    }


def _check_the_name(said_by: str) -> None:
    """Whose opinion this is, and that it is a person's (§10.3, §11.2).

    `identity.person` IS the rule now. This function kept its own copy of the length and
    product-name checks after §11.2 gathered the other four, so the sweep that claimed to
    leave one implementation left two — and mutation showed the shared one's length rule was
    untested, because the test covering it was about THIS function's message.

    It stays as a named function because this is the easiest place in the product to abuse the
    field: the payload the model just read says "answering should never need typing", which
    reads as an invitation to fill it in.
    """
    import identity

    identity.person(said_by, field="said_by")


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _menu_for(conn, menu_token: str) -> Optional[dict]:
    """The menu this token names, or None if it no longer matches.

    ONE derivation: the token says which scope and page it came from, so this rebuilds that
    menu and compares. The search it replaces was quadratic and slowest on the path it is
    taken on most.
    """
    address = _address(menu_token)
    if address is None:
        return None
    scope, page = address
    menu = queue(conn, scope=scope, page=page)
    return menu if menu.get("menu_token") == menu_token else None


def _refreshed(conn, why: str) -> dict:
    return {**queue(conn), "status": "refreshed",
            "what_it_means": (
                f"Nothing was written: {why}, so the number you chose may no longer mean the "
                f"campaign you were looking at. Here is the menu as it stands — read it out "
                f"and ask again.")}


def waiting(conn, *, apart_from: Optional[str] = None) -> int:
    """How many campaigns are waiting, for the surfaces that offer the queue (§10.6).

    `apart_from` is the record the caller has just written. Counting it made every first
    upload answer itself — "1 campaign is waiting on feedback" about the campaign still on
    screen — which is the always-present offer that teaches a reader to skip the list.

    CAMPAIGNS, which is what every sentence built from this number says. §10.2/D110 added a
    row that is not about a campaign, and this filtered on `r["campaign_id"]` — a key that
    row does not have. The KeyError went into a bare `except` and came back as 0, so one pair
    of similarly-worded client corrections — the ordinary state of an agency library, and the
    precondition D110 was written about — silently turned off every proactive queue offer in
    the product. Permanently, and with nothing to see: 10.6's entire headline switched off by
    10.2's own new row.

    The `except` is gone with it. An exception handler that turns a crash into "there is
    nothing waiting" is the same defect class as the silent partial re-index D98 just fixed:
    a failure the product cannot have, reported as a clean result. `_open_rows` reads what
    every other surface reads; if it throws, that is not a fact about the feedback queue.
    """
    import store

    # A database that has not been upgraded yet has nothing waiting on feedback, and that is
    # a true statement rather than a swallowed error — `store.init_db` creates these on the
    # next start. NAMED, because the blanket `except Exception: return 0` this replaces made
    # every failure indistinguishable from an empty queue, which is how one pair of similar
    # corrections switched off every proactive offer in the product with nothing to see.
    if not all(store._columns(conn, table)
               for table in ("campaigns", "metrics", "evaluations")):
        return 0
    # Every row here is a campaign — §10.2/D110's questions live in their own scope, and
    # `test_the_open_rows_are_all_campaigns` is what keeps that true. A `r.get("campaign_id")`
    # guard was here while they were mixed in; keeping it now would read as a live check on a
    # case that cannot arise, which is how a dead guard outlives the bug it was written for.
    return len([r for r in _open_rows(conn) if r["campaign_id"] != apart_from])
