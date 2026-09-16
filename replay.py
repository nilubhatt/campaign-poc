"""
§8.7: replay as a report.

The review: *"Because every verdict is stamped with its rulebook version, adding an expected
metric can be applied backwards as a question rather than a rewrite. Two answers worth having
immediately: which stored campaigns now fail the new expectation — that is the backlog of
things to go and ask partners for — and which past verdicts would change under the current
rules. Old evaluations are never silently rewritten; the replay is a report."*

**Nothing here writes.** Not a flag on a judgment, not a cached answer, not a "stale" column.
The report is derived from what is on file every time it is asked, which is what makes "never
silently rewritten" true by construction rather than by discipline — there is no second copy to
drift, and no stored row carrying a verdict about itself that a reader has to trust.

**"Which past verdicts would change" is a claim this server cannot honestly make.** It cannot
re-run the model, and a model asked the same question twice does not always answer the same way
— which is the variance §7.7 exists to measure. Saying a verdict "would change" would be the
confident unfounded assertion this whole review is written against, with the server's authority
behind it.

What the server knows exactly is what a judgment was **not checked against**: which measures
became expected after it was saved, which rules became standing after it was saved, and whether
a rule it actually rested on has since been withdrawn. All three are `computed` in §2.4's sense
— identical for every reader, so a difference is a bug. And they are the thing a person can act
on, because the answer to "would it change" is *judge it again*, which is an offer.

The two halves are deliberately different in kind. The **backlog** is about records — a campaign
missing an expected measure is a partner to email. The **judgments** half is about evaluations,
and every row of it is a question.
"""
from __future__ import annotations

from typing import Optional

import learning

# Bounded, because this phase has now closed the same defect twice: §8.4's rendered checklist
# and §8.6's rule list and provenance join were all unbounded. A library of five hundred
# campaigns produces a backlog of hundreds, and the tool's own description tells the model to
# say the names.
_MAX_GROUPS = 8            # market × measure groups spelled out
_MAX_NAMED = 5             # campaigns named inside one group
_MAX_JUDGMENTS = 20        # judgment rows returned, most consequential first

# What judging this again would MECHANICALLY meet that the saved judgment did not. Not a
# prediction about the verdict — each of these is a fact about the evidence package, computed
# the same way for every reader. Ordered most consequential first, which is the order the rows
# come back in.
CONSEQUENCES = ("stated_basis_withdrawn", "rests_on_withdrawn", "gap_appears",
                "rule_not_applied")


def run(conn, *, market: Optional[str] = None) -> dict:
    """The replay. Reads everything, writes nothing (§8.7)."""
    import core

    backlog = _backlog(conn, market=market)
    # Scoped too. Filtering only the backlog made `what_it_means` add up two different
    # populations — "0 stored campaigns, 1 saved judgment" from a report asked about one
    # market.
    judgments = _judgments(conn, market=market)
    return {
        "rulebook_version": core.rulebook_version(),
        "rulebook_note": (
            "A judgment's stamp says which rulebook was in force, and it is the same on "
            "every judgment made between two edits of that file — so this report is built "
            "from WHEN each measure and rule was confirmed, not from the stamp. That does "
            "not change when the rulebook does: the stamp tells two rulebooks apart, and "
            "this report is about what was known on a given day."),
        "basis": "computed",
        "backlog": backlog,
        "judgments": judgments[:_MAX_JUDGMENTS],
        "judgments_total": len(judgments),
        "what_it_means": _sentence(conn, backlog, judgments),
    }


def _backlog(conn, *, market: Optional[str] = None) -> list:
    """Stored campaigns that do not carry what is now expected of them.

    The review's own framing is what makes this useful: *"that is the backlog of things to go
    and ask partners for"*. So it is a list of records with names on them, not a count — a
    count is a fact and a list is a morning's work.
    """
    import metrics
    import store

    groups: dict = {}
    not_yet = 0
    for record in store.list_campaigns(conn):
        check = metrics.expected_check(conn, record["id"])
        # `missing` alone is the test. Every `nothing_to_check` answer — no market, not a
        # campaign, no record — already returns an empty `missing`, so a status check beside it
        # was a second condition that could never be the one doing the work.
        if not check["missing"]:
            continue
        if market and store.fold_market(market) not in {store.fold_market(m)
                                                        for m in check["markets"]}:
            continue
        # A campaign that has not finished cannot have a measured result, so it is not
        # something to go and ask a partner for — you cannot ask for numbers that do not exist
        # yet. Counted rather than dropped silently: "nothing to chase here" and "I did not
        # look at the ones still running" are different answers (§7.1).
        if record.get("status") != "concluded":
            not_yet += 1
            continue
        # A superseded version is not a partner to chase — v1 and v2 of one brief were both
        # listed, so one conversation appeared as two. The later version is the one that ran.
        if record.get("is_superseded"):
            continue
        for name in check["missing"]:
            for where in check["markets"]:
                groups.setdefault((store.fold_market(where), name),
                                  {"market": where, "measure": name, "campaigns": []}
                                  )["campaigns"].append(
                    {"campaign_id": record["id"], "title": record["title"]})

    # One conversation per partner per measure is the unit of work the review describes, so
    # that is the shape: "LATAM: 14 campaigns missing footfall uplift", with names.
    out = []
    for group in sorted(groups.values(), key=lambda g: (-len(g["campaigns"]), g["market"],
                                                        g["measure"])):
        total = len(group["campaigns"])
        out.append({
            "market": group["market"],
            "measure": group["measure"],
            "campaigns_missing_it": total,
            "campaigns": group["campaigns"][:_MAX_NAMED],
            "more": max(0, total - _MAX_NAMED),
            "what_it_means": (
                f"{total} concluded campaign{'s' * (total != 1)} in {group['market']} "
                f"{'have' if total != 1 else 'has'} no measured {group['measure']} on file. "
                f"That is one conversation to have with whoever ran "
                f"{'them' if total != 1 else 'it'}."),
        })
    return {"groups": out[:_MAX_GROUPS],
            "groups_total": len(out),
            "campaigns_still_running": not_yet}


def _judgments(conn, *, market: Optional[str] = None) -> list:
    """Saved judgments and what has changed under them since (§8.7).

    Each row carries a `consequence` — what judging this again would MECHANICALLY meet that
    the saved judgment did not. That is not a prediction about the verdict, which the server
    cannot make; it is a fact about the evidence package, computed the same way for every
    reader. The distinction matters because "was not checked against X" flattens three very
    different situations into one sentence:

      • the subject DOES carry the measure, so the new check passes and nothing about a
        re-judgment can differ on that axis — a non-event, and listing it as though it were a
        gap is what turns this report into noise;
      • the subject does NOT carry it, so judging it again meets a computed gap;
      • the verdict's whole stated basis was a rule somebody has since withdrawn, which is the
        most urgent row this report can produce and read identically to the least urgent.
    """
    import actions
    import store

    measures = {name: entry for name, entry in metrics_registry(conn).items()
                if entry["status"] == "expected" and entry.get("confirmed_at")}
    all_rules = store.corrections(conn)
    rules = [c for c in all_rules if c["status"] == "expected" and c.get("confirmed_at")]
    # Anything that is not standing. A rule set aside and then REOPENED is `provisional` —
    # still applied to nothing — and keying on `ignored` alone made a judgment whose blocking
    # finding cites it vanish from the report entirely at the moment somebody reopened it.
    withdrawn = {c["id"]: c for c in all_rules if c["status"] != "expected"}

    out = []
    for row in store.list_evaluations(conn):
        judged_at = row.get("created_at")
        if judged_at is None:
            continue                   # a legacy row with no timestamp cannot be compared
        subject = row.get("campaign_id")
        markets = _markets_of_subject(conn, subject)
        if market and store.fold_market(market) not in {store.fold_market(m)
                                                        for m in markets}:
            continue
        after_measures = sorted(
            name for name, entry in measures.items()
            if entry["confirmed_at"] > judged_at and _applies(entry["expected_in"], markets))
        after_rules = [
            {"correction_id": c["id"], "text": c["text"]}
            for c in rules
            if c["confirmed_at"] > judged_at and _applies(c["expected_in"], markets)]
        saved = store.get_evaluation(conn, row["id"]) or {}
        rested_on, sole = _withdrawn_rules_cited(saved, withdrawn)
        # No "exclude the rules this judgment cites" guard here, though the first fix added
        # one: a judgment can only cite a rule that was already standing when it was saved
        # (core refuses a provisional one), so with `confirmed_at` preserved across a
        # re-confirmation the comparison can never put a cited rule after its own judgment.
        # The mutation pass found the guard unreachable, which is what it is for.
        # Only the measures the subject does not actually carry. The rest are checks that
        # would pass, and a report that lists a passing check beside a real gap is one nobody
        # reads twice.
        gaps = _not_carried(conn, subject, after_measures)

        consequence = _consequence(sole, rested_on, gaps, after_rules)
        if consequence is None:
            continue
        out.append({
            "evaluation_id": row["id"],
            "subject_title": row["subject_title"],
            "campaign_id": subject,
            "verdict": row.get("verdict"),
            "judged_at": judged_at,
            "consequence": consequence,
            "not_checked_against": {"measures": after_measures, "corrections": after_rules},
            "gaps_that_would_appear": gaps,
            "rests_on_withdrawn": rested_on,
            "basis": "computed",
            "what_it_means": _judgment_sentence(consequence, gaps, after_measures,
                                                after_rules, rested_on),
            # An OFFER. The answer to "would this verdict change" is judging it again, and
            # judging it again is the model's work, not the server's — so the report stops at
            # the question and hands over the one call that answers it.
            "next_actions": actions.trim([actions.action(
                f"Judge \u201c{row['subject_title']}\u201d again under the current rules",
                "prepare_evaluation",
                why="Nothing about the saved judgment changes. This gathers the evidence "
                    "again, including what has become expected since, so a new judgment can "
                    "be written beside the old one.",
                consent="ask",
                subject_title=row["subject_title"],
                campaign_id=subject)]) if subject else [],
        })
    return sorted(out, key=lambda r: (CONSEQUENCES.index(r["consequence"]), -r["judged_at"]))


def _consequence(sole: bool, rested_on: list, gaps: list, after_rules: list):
    """Which of the four, or None when judging it again would meet nothing new."""
    if sole:
        return "stated_basis_withdrawn"
    if rested_on:
        return "rests_on_withdrawn"
    if gaps:
        return "gap_appears"
    if after_rules:
        return "rule_not_applied"
    return None


def _not_carried(conn, campaign_id: Optional[str], names: list) -> list:
    """Of these measures, the ones the subject has no MEASURED value for."""
    import metrics

    if not campaign_id:
        return []
    return [name for name in names
            if not [v for v in metrics.values_for(conn, campaign_id, name)
                    if v["metric_type"] == "actual"]]


def metrics_registry(conn) -> dict:
    import store
    return store.metric_registry(conn)


def _markets_of_subject(conn, campaign_id: Optional[str]) -> list:
    import store

    if not campaign_id:
        return []
    record = store.get_campaign(conn, campaign_id) or {}
    return [m for m in store.markets_of(record) if m]


def _applies(expected_in, markets) -> bool:
    """Whether a rule that graduated in `expected_in` reaches a subject in `markets`.

    A judgment about a Japanese brief was not "unchecked" against a rule that graduated on
    LATAM and SEA — that rule does not apply to it at all, and saying otherwise would put every
    judgment in the library on the list the first time anything graduated anywhere.

    A judgment with no subject record has no market, so nothing is claimed about it: it is
    listed only for a rule it actually cited being withdrawn, which is a fact about the
    judgment rather than about a market.
    """
    import store

    if not markets:
        return False
    return bool({store.fold_market(m) for m in markets}
                & {store.fold_market(m) for m in expected_in})


def _withdrawn_rules_cited(saved: dict, withdrawn: dict) -> tuple:
    """Standing corrections this judgment rested on that have since been set aside.

    Returns `(rows, sole)`. `sole` is the one §8.7 most needs to say out loud: the product
    already encodes "this verdict rests ONLY on rules that were broken" — it is why a verdict
    like that skips the disconfirming search — so when every not-debatable finding in a
    non-approve verdict cites a rule that has since been withdrawn, the server can say
    mechanically that the whole stated basis of the verdict is gone. Reported as a flat list,
    that read exactly like one note among five.
    """
    out, seen = [], set()
    findings = saved.get("findings") or []
    for finding in findings:
        cited = (finding.get("precedent") or {}).get("correction_id")
        if cited in withdrawn and cited not in seen:
            seen.add(cited)
            out.append({"correction_id": cited, "text": withdrawn[cited]["text"],
                        "finding": finding.get("finding")})
    # The MODEL's findings only. §7.8's computed ones are the server's own facts about the
    # brief — they are not part of what the judgment argued, so counting them would mean a
    # verdict could never rest solely on anything.
    argued = [f for f in findings if (f.get("basis") or "judged") == "judged"]
    breaches = [f for f in argued if f.get("kind") == "guardrail_breach"]
    sole = bool(
        out and breaches and saved.get("verdict") != "approve"
        # Every finding that carries the verdict is a breach, and every breach rests on a
        # rule that has been withdrawn.
        and all(f.get("kind") == "guardrail_breach" for f in argued
                if f.get("severity") in ("blocking", "should_fix"))
        and all((f.get("precedent") or {}).get("correction_id") in withdrawn
                for f in breaches))
    return out, sole


def _judgment_sentence(consequence: str, gaps: list, measures: list, rules: list,
                       withdrawn: list) -> str:
    """One sentence per consequence, and none of them says the verdict would change."""
    if consequence == "stated_basis_withdrawn":
        return (f"Everything this verdict rests on is a rule that has since been set aside: "
                f"{'; '.join(c['text'] for c in withdrawn)}. The judgment is unchanged and "
                f"nobody has revisited it — but its whole stated basis is no longer a rule "
                f"here.")
    if consequence == "rests_on_withdrawn":
        return (f"This judgment cites {len(withdrawn)} standing correction"
                f"{'s' * (len(withdrawn) != 1)} that "
                f"{'have' if len(withdrawn) != 1 else 'has'} since been set aside, so "
                f"{'those findings are' if len(withdrawn) != 1 else 'that finding is'} "
                f"anchored to something that is no longer a rule. The rest of it stands.")
    if consequence == "gap_appears":
        return (f"The subject has no measured {', '.join(gaps)} on file, and "
                f"{'those became' if len(gaps) != 1 else 'that became'} expected here after "
                f"this was judged. Judging it again would see a gap this judgment did not. "
                f"That is a difference in the evidence, not a verdict — whether it changes "
                f"anything is the judgment's to make.")
    named = [c["text"] for c in rules]
    return (f"{len(named)} rule{'s' * (len(named) != 1)} became standing here after this was "
            f"judged, so it was never checked against "
            f"{'them' if len(named) != 1 else 'it'}: {'; '.join(named[:2])}"
            f"{' and others' if len(named) > 2 else ''}. Whether this brief breaks "
            f"{'any of them' if len(named) != 1 else 'it'} is a judgment, so the library does "
            f"not guess — judge it again to find out.")


def _sentence(conn, backlog: dict, judgments: list) -> str:
    import store

    anything_expected = any(e["status"] == "expected"
                            for e in store.metric_registry(conn).values())
    anything_standing = any(c["status"] == "expected" for c in store.corrections(conn))
    if not (anything_expected or anything_standing):
        return ("Nothing is expected of a brief yet and no rule is standing, so there is "
                "nothing to replay. A measure or a rule joins once it has been seen across "
                "markets and a person has confirmed it.")

    chase = sum(g["campaigns_missing_it"] for g in backlog["groups"])
    still = backlog["campaigns_still_running"]
    urgent = [j for j in judgments if j["consequence"] == "stated_basis_withdrawn"]
    return (
        f"{backlog['groups_total']} thing"
        f"{'s' * (backlog['groups_total'] != 1)} to go and ask partners for, covering "
        f"{chase} concluded campaign{'s' * (chase != 1)}"
        + (f" ({still} still running, so nothing to ask for yet)" if still else "")
        + f". {len(judgments)} saved judgment{'s' * (len(judgments) != 1)} would meet "
          f"something new if judged again"
        + (f", and {len(urgent)} rest{'s' * (len(urgent) == 1)} entirely on a rule that has "
           f"since been withdrawn" if urgent else "")
        + ". Nothing has been changed: this is a report, and judging any of them again writes "
          "a new judgment beside the old one rather than replacing it.")


# ── D104: the blast radius, asked before somebody confirms ──────────────────

def if_graduated(conn, *, measure: Optional[str] = None,
                 correction_id: Optional[str] = None,
                 markets: Optional[list] = None) -> dict:
    """What promoting this would do to the library, without promoting it (§8.7, D104).

    *"This makes 14 LATAM records show as missing it."* The person confirming is exactly who
    needs that number, and needs it BEFORE they confirm — afterwards it is a surprise rather
    than a decision. Asking costs nothing and changes nothing.
    """
    import corrections
    import metrics
    import store

    if measure:
        name = metrics.canonical(conn, measure) or measure
        entry = metrics.describe(conn, name)
        if not entry:
            raise ValueError(f"{name!r} is not a measure on file")
        where = markets if markets is not None else metrics.graduation(conn, name)["seen_in"]
        affected = [r for r in store.list_campaigns(conn)
                    if _would_miss(conn, r, where, name)]
        return {
            "measure": name,
            "markets": where,
            "campaigns_affected": len(affected),
            "examples": [{"campaign_id": r["id"], "title": r["title"]} for r in affected[:5]],
            "basis": "computed",
            "what_it_means": (
                f"Confirming this would show {len(affected)} stored "
                f"campaign{'s' * (len(affected) != 1)} in "
                f"{', '.join(where) or 'no market'} as missing {name}, and every "
                f"brief there afterwards would be checked for it. Nothing is deleted and it "
                f"can be retired later."
                if affected else
                f"Confirming this would change nothing about what is already on file — no "
                f"stored campaign in {', '.join(where) or 'no market'} is missing "
                f"{name}. Briefs from here on would be checked for it."),
        }

    entry = corrections.describe(conn, correction_id or "")
    if not entry:
        raise ValueError(f"{correction_id!r} is not a correction on file")
    where = (markets if markets is not None
             else corrections.graduation(conn, entry["correction_id"])["seen_in"])
    # A rule is not a set-membership test — whether a brief breaks it is a judgment. So the
    # honest number is not "how many fail it" but how many were judged WITHOUT it, which is
    # what a person is deciding to leave unexamined.
    affected = [r for r in _judgments_before(conn)
                if _applies(where, _markets_of_subject(conn, r.get("campaign_id")))]
    return {
        "correction_id": entry["correction_id"],
        "text": entry["text"],
        "markets": where,
        "judgments_affected": len(affected),
        "examples": [{"evaluation_id": r["id"], "subject_title": r["subject_title"]}
                     for r in affected[:5]],
        "basis": "computed",
        "what_it_means": (
            f"Confirming this would make every brief in "
            f"{', '.join(where) or 'no market'} judged against it from here on. "
            f"{len(affected)} saved judgment{'s' * (len(affected) != 1)} "
            f"{'were' if len(affected) != 1 else 'was'} written without it and would show in "
            f"the replay as not checked against it — none of them is rewritten."),
    }


def _would_miss(conn, record: dict, markets: list, name: str) -> bool:
    import metrics
    import store

    if record.get("record_type") not in learning.CHECKABLE_RECORDS:
        return False
    where = [m for m in store.markets_of(record) if m]
    if not _applies(markets, where):
        return False
    return not [v for v in metrics.values_for(conn, record["id"], name)
                if v["metric_type"] == "actual"]


def _judgments_before(conn) -> list:
    import store
    return [r for r in store.list_evaluations(conn) if r.get("created_at") is not None]
