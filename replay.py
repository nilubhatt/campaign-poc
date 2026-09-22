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
and rules APPLY TO IT NOW and did not on the day it was written — a measure can be confirmed
later, or have been retired while the brief was judged; a rule can become standing later, be
standing and reach this brief's markets only after its scope changes, or have been set aside at
the time — and whether a rule it actually rested on is no longer standing. All three are `computed` in §2.4's sense
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
# WHAT EACH CONSEQUENCE IS, most urgent first — and this order is the report's sort. One
# tuple, because the names were typed out here and the decision written out again below, with
# `CONSEQUENCES.index` raising on any name the two disagreed about: a rename in one place
# would have turned every report into a crash.
_CONSEQUENCES = (
    # `sole`: every not-debatable finding cites a rule that is no longer standing.
    ("stated_basis_withdrawn", lambda sole, rested, gaps, rules: sole),
    ("rests_on_withdrawn", lambda sole, rested, gaps, rules: bool(rested)),
    ("gap_appears", lambda sole, rested, gaps, rules: bool(gaps)),
    ("rule_not_applied", lambda sole, rested, gaps, rules: bool(rules)),
)
CONSEQUENCES = tuple(name for name, _ in _CONSEQUENCES)


def run(conn, *, market: Optional[str] = None) -> dict:
    """The replay. Reads everything, writes nothing (§8.7)."""
    import core

    backlog = _backlog(conn, market=market)
    # Scoped too. Filtering only the backlog made `what_it_means` add up two different
    # populations — "0 stored campaigns, 1 saved judgment" from a report asked about one
    # market.
    judgments = _judgments(conn, market=market)
    # D115: whether THE RULES THEMSELVES changed under these judgments.
    #
    # The replay is built from `confirmed_at` on registry and correction rows, which is the
    # right way to ask "what was known on the day" — but a rulebook's rules carry no
    # `confirmed_at`, so a v1→v2 rulebook change produced a replay reporting that nothing had
    # changed. A silent false negative on precisely the axis §8.7 exists for.
    #
    # It could not be built until there were rulebook versions to compare, and §12.2's stamp
    # is what makes them comparable: before it, every judgment carried one literal string.
    stamps = _rulebooks_seen(conn, market)
    return {
        "rulebook_version": core.rulebook_version(),
        "rulebooks_seen": stamps,
        "rulebook_changed": len(stamps) > 1,
        "rulebook_note": (
            ("The rules THEMSELVES changed under these judgments — they were made under "
             f"{len(stamps)} different rulebooks ({', '.join(stamps)}), so a judgment that "
             "reads differently from a later one may be answering a different rulebook "
             "rather than a different library. That is a bigger difference than anything "
             "below and nothing else here can see it."
             if len(stamps) > 1 else
             "One rulebook throughout, so nothing below is explained by the rules having "
             "moved.")
            + " Everything else here is built from what was IN FORCE on the day each "
              "judgment was written rather than from the stamp — for a measure, whether it "
              "had been confirmed and not retired by then, and for a rule, whether it had "
              "started applying to that brief's markets. Both are different dates from the "
              "one a measure or a rule shows as its confirmation, whenever it has been "
              "retired, set aside or re-scoped since."),
        "basis": "computed",
        "backlog": backlog,
        "judgments": judgments[:_MAX_JUDGMENTS],
        "judgments_total": len(judgments),
        "what_it_means": _sentence(conn, backlog, judgments),
    }


def _rulebooks_seen(conn, market: Optional[str]) -> list:
    """The rulebook versions the judgments IN SCOPE were made under (§12.2).

    Scoped, because everything else in the report is and this sentence claims to outrank all
    of it — "a bigger difference than anything below and nothing else here can see it". Asked
    unscoped, a report about Peru announced that the rules themselves had changed under these
    judgments on the strength of a Japanese judgment stamped with another rulebook, about a
    market whose every judgment was made under one.
    """
    import store

    seen = {}
    for row in store.list_evaluations(conn):
        if market and not learning.reaches(_where_it_ran(conn, row.get("campaign_id")),
                                           market):
            continue
        saved = store.get_evaluation(conn, row["id"]) or {}
        stamp = (saved.get("provenance") or {}).get("rulebook_version")
        if stamp:
            seen.setdefault(stamp, row.get("created_at") or 0)
    # By when each was first seen rather than by name: `v0.10` sorts before `v0.9` as a
    # string, and the order a reader wants is the order they happened in.
    return [stamp for stamp, _ in sorted(seen.items(), key=lambda pair: pair[1])]


def _backlog(conn, *, market: Optional[str] = None) -> dict:
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
        if market and not learning.reaches(check["markets"], market):
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

    # One conversation per MARKET per measure, which is the key below — the review describes
    # the unit of work as one per partner, and `campaigns.partner` exists, but a market is
    # what every campaign carries and a partner is not. Named rather than left to be read off
    # the key, because the two are easy to confuse and one of them is a promise.
    out = []
    for group in sorted(groups.values(), key=lambda g: (-len(g["campaigns"]), g["market"],
                                                        g["measure"])):
        total = len(group["campaigns"])
        out.append({
            "market": group["market"],
            "measure": group["measure"],
            "campaigns_missing_it": total,
            # Every one of them, for counting; `campaigns` is the named few a reader sees.
            "campaigns_all": group["campaigns"],
            "campaigns": group["campaigns"][:_MAX_NAMED],
            "more": max(0, total - _MAX_NAMED),
            "what_it_means": (
                f"{total} concluded campaign{'s' * (total != 1)} in {group['market']} "
                f"{'have' if total != 1 else 'has'} no measured {group['measure']} on file. "
                f"That is one conversation to have with whoever ran "
                f"{'them' if total != 1 else 'it'}."),
        })
    shown = [{k: v for k, v in group.items() if k != "campaigns_all"}
             for group in out[:_MAX_GROUPS]]
    return {"groups": shown,
            "groups_total": len(out),
            # THE CAMPAIGNS, counted once each and over every group rather than the truncated
            # eight. Summing `campaigns_missing_it` counts a campaign once per measure it is
            # missing AND once per market it ran in, so the sentence read "12 things covering
            # 84 concluded campaigns" about a library holding 15 such campaigns and 33 in
            # total — a per-market-per-measure overcount, truncated, presented as a count of
            # campaigns. It is the defect this module's own comment says it closed when
            # `what_it_means` added up two populations, one key along.
            "campaigns_total": len({c["campaign_id"] for group in out
                                    for c in group["campaigns_all"]}),
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
    import metrics
    import store

    # Which judgment was written first, for the one comparison two wall-clock floats cannot
    # settle between them. Once per report.
    order = store.evaluation_order(conn)
    all_rules = store.corrections(conn)
    # The same selection `corrections.standing_for` makes, and no stricter. Requiring
    # `confirmed_at` too made this report's idea of "standing" narrower than the live one: on a
    # row without it the live path applies the rule to the brief and this returned an empty
    # report. No write produces that row today, which is exactly how a second copy of a
    # selection waits.
    rules = [c for c in all_rules if c["status"] == "expected"]
    # Anything that is not standing. A rule set aside and then REOPENED is `provisional` —
    # still applied to nothing — and keying on `ignored` alone made a judgment whose blocking
    # finding cites it vanish from the report entirely at the moment somebody reopened it.
    withdrawn = {c["id"]: c for c in all_rules if c["status"] != "expected"}

    out = []
    judgments_on_file = store.list_evaluations(conn)
    for row in judgments_on_file:
        judged_at = row.get("created_at")
        if judged_at is None:
            continue                   # a legacy row with no timestamp cannot be compared
        subject = row.get("campaign_id")
        # TWO DIFFERENT QUESTIONS, and collapsing them silenced the most urgent row this
        # report has. "Is this judgment one the reader asked about" is about where the record
        # RAN — a Peru reference record is a Peru row whatever any checklist thinks of it —
        # while "what is this brief checked against" goes through the live gate, which refuses
        # a reference record entirely. Filtered on the gated answer, a judgment about Peru
        # brand guidelines whose whole stated basis had been withdrawn appeared in the
        # unfiltered report and vanished from `market="Peru"`.
        markets, checklist = _subject_checklist(conn, subject)
        if market and not learning.reaches(_where_it_ran(conn, subject), market):
            continue
        # WHAT THIS BRIEF IS CHECKED FOR NOW, asked of the function that decides it for a live
        # judgment. This file kept its own market test and never learned about the other half
        # of §12.4: a measure that graduated on store launches is keyed on the TYPE, not on
        # the markets it happened to be seen in. So the report flagged a Peru seeding brief
        # for a store-launch measure it is not expected to carry, and stayed silent about a
        # Vietnam store launch that is — two wrong answers from one missing condition, and the
        # live surface had them both right.
        # THE MEASURES HALF NEEDS THE REFUSAL ITSELF, not just an empty market list, and the
        # rules half does not — which is not an inconsistency but §12.4: a measure keyed on
        # the campaign TYPE matches with no market at all, so nothing else would stop it
        # reaching a record the checklist refuses. A rule is scoped to markets or declared for
        # every market, and neither reaches a subject that has none.
        after_measures = [] if not checklist else _newly_expected(
            conn, metrics.expected_for(
                conn, markets=markets, campaign_type=_type_of_subject(conn, subject)),
            judged_at, order.get(row["id"]))
        saved = store.get_evaluation(conn, row["id"]) or {}
        rested_on, sole = _withdrawn_rules_cited(saved, withdrawn)
        # THE RULES THIS JUDGMENT ACTUALLY CITES, excluded — the guard that came back.
        #
        # It was removed as unreachable and it was: under the old comparison a judgment could
        # only cite a rule already standing when it was saved, and with `confirmed_at`
        # preserved across a re-confirmation, a rule confirmed before its own judgment could
        # never sort after it. That property belonged to the comparison, and the comparison is
        # the in-force history now. A rule can leave a brief's markets and come back — and a
        # judgment can cite a rule that was standing SOMEWHERE ELSE, which `save_evaluation`
        # accepts — so a rule whose own blocking finding quotes it appeared on that judgment's
        # "never checked against" list. Whatever the history says, a judgment that quotes a
        # rule was checked against it: that is what citing it means.
        cited = _rules_cited(saved)
        after_rules = []
        for c in rules:
            if c["id"] in cited:
                continue
            since = _newly_reaches(conn, c, markets, judged_at,
                                   judgment_seq=order.get(row["id"]))
            if since:
                after_rules.append({"correction_id": c["id"], "text": c["text"],
                                    "since": since})
        # Only the measures the subject does not actually carry. The rest are checks that
        # would pass, and a report that lists a passing check beside a real gap is one nobody
        # reads twice.
        gaps = _not_carried(conn, subject, [m["measure"] for m in after_measures])

        consequence = _consequence(sole, rested_on, gaps, after_rules)
        if consequence is None:
            continue
        # LATER, settled by the order things were written rather than by two equal floats —
        # the comparison this file spent two rounds replacing everywhere else, written out
        # again here. A judgment's own place in that order is one past the judgments that
        # existed when it was saved, which is what `after_judgment` takes.
        revisited = bool(subject) and any(
            other.get("campaign_id") == subject
            and other["id"] != row["id"]
            and store.after_judgment(other.get("created_at"),
                                     (order.get(other["id"]) or 1) - 1,
                                     judged_at, order.get(row["id"]))
            for other in judgments_on_file)
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
            # WHETHER SOMEBODY HAS JUDGED THIS SUBJECT AGAIN — the fact, not a conclusion
            # drawn from it. "Nobody has revisited it" was asserted on a row stamped
            # `computed` with nothing behind it; this is computed, from the order things were
            # written in. What it does NOT say is whether that later judgment met what this
            # row reports, because the report cannot see that without knowing when each change
            # happened, and it got that wrong in the withdrawn-basis case the first time.
            "revisited": revisited,
            "what_it_means": _judgment_sentence(consequence, gaps, after_measures,
                                                after_rules, rested_on, revisited),
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
    # NO INFERENCE FROM "a later judgment is not on this report" TO "it met what this row
    # reports". That reading was wrong for the row that matters most: a judgment written
    # BEFORE somebody withdrew a rule is absent from the report because it cites nothing
    # withdrawn, and it cannot have met a withdrawal that had not happened. Making the test
    # right needs the date of every change each row reports — more machinery, on a report
    # whose only job here is to hand over one offer.
    #
    # So the report states the fact it has (`revisited`, and the sentence that goes with it)
    # and leaves the conclusion to the reader, who knows what they judged and when. An offer
    # nobody needs costs a glance; an offer withdrawn because the server inferred something
    # false costs the one row §8.7 exists for.
    return sorted(out, key=lambda r: (CONSEQUENCES.index(r["consequence"]), -r["judged_at"]))


def _consequence(sole: bool, rested_on: list, gaps: list, after_rules: list):
    """Which of the four, or None when judging it again would meet nothing new. In the order
    they are declared, which is the order they are reported in."""
    return next((name for name, holds in _CONSEQUENCES
                 if holds(sole, rested_on, gaps, after_rules)), None)


def _not_carried(conn, campaign_id: Optional[str], names: list) -> list:
    """Of these measures, the ones the subject has no MEASURED value for."""
    import metrics

    if not campaign_id:
        return []
    return [name for name in names if not metrics.carries(conn, campaign_id, name)]


def metrics_registry(conn) -> dict:
    import store
    return store.metric_registry(conn)


def _subject_checklist(conn, campaign_id: Optional[str]) -> tuple:
    """`(markets, whether a checklist applies to this subject at all)`.

    `learning.subject_markets`' three answers, which are the live path's: no such record, not
    a campaign, no market named — or here are the markets. Both live surfaces go through it,
    `expected_check` for measures and `standing_for` for rules, and this report read the
    record's markets directly. So it claimed a reference record's judgment had missed a
    store-launch measure that the live checklist refuses to apply to a reference record at
    all, and a market-less record the same: a type-keyed measure needs no market to match, so
    nothing stopped it.

    A refusal is not an empty market list. A checklist that does not apply cannot be something
    a judgment failed to check against, and saying otherwise is this report accusing somebody
    of missing a check the product would never have made.
    """
    if not campaign_id:
        return [], False
    named, refusal = learning.subject_markets(conn, campaign_id)
    return ([], False) if refusal else (named, True)


def _type_of_subject(conn, campaign_id: Optional[str]) -> Optional[str]:
    """The kind of campaign this judgment was about, for §12.4's type-keyed checklist."""
    import store

    record = store.get_campaign(conn, campaign_id) if campaign_id else None
    return (record or {}).get("campaign_type")


def _where_it_ran(conn, campaign_id: Optional[str]) -> list:
    """Where the subject RAN, for scoping a report — not for deciding what it is checked
    against. `_subject_checklist` answers that, and answers it differently: a reference record
    has markets and has no checklist."""
    import store

    # `or {}` on BOTH counts, and the guard is the point rather than defensive habit. A
    # judgment can carry no `campaign_id` at all — §8.6's "judge this new pitch", a proposal
    # that is not a record yet, which `save_evaluation` accepts deliberately — and a judgment
    # can outlive the record it was about. `markets_of(None)` raises `AttributeError`, which
    # reaches a marketer as "Error executing tool replay_rules" with the reason discarded, and
    # only when they scope the report to a market.
    record = (store.get_campaign(conn, campaign_id) if campaign_id else None) or {}
    return [m for m in store.markets_of(record) if m]


def _applies(expected_in, markets, *, everywhere: bool) -> bool:
    """Whether a rule that graduated in `expected_in` reaches a subject in `markets`.

    A judgment about a Japanese brief was not "unchecked" against a rule that graduated on
    LATAM and SEA — that rule does not apply to it at all, and saying otherwise would put every
    judgment in the library on the list the first time anything graduated anywhere.

    A judgment with no subject record has no market, so nothing is claimed about it: it is
    listed only for a rule it actually cited being withdrawn, which is a fact about the
    judgment rather than about a market.

    Both of those, and `everywhere`, are `learning.in_force`'s to decide: this file held the
    second and third copies of that test, and the copy here was the one that never learned
    about a rule the customer declares applies everywhere.
    """
    return learning.in_force(expected_in, markets, everywhere=everywhere)


def _cited_by(finding: dict) -> Optional[str]:
    """`corrections.cited_by`, which is where the definition lives — the two other readers of
    this are the ones that decide how a verdict is voiced."""
    import corrections

    return corrections.cited_by(finding)


def _rules_cited(saved: dict) -> set:
    """Every correction a saved judgment's findings quote as precedent."""
    return {_cited_by(finding) for finding in (saved.get("findings") or [])} - {None}


def _newly_expected(conn, names: list, judged_at: float,
                    judgment_seq: Optional[int] = None) -> list:
    """The measures this brief is checked for now and was not when it was judged, and WHY.

    The rules half has carried its reason since the round that split "became standing" from
    "its scope changed"; this half got the mechanism — `measure_in_force_at`, which walks
    §12.4's retirements and revivals — and kept the old sentence. So a brief judged while a
    measure was retired was told the measure "became expected here after this was judged",
    about a measure confirmed months before it. That is the reading whose own docstring says a
    reader "concludes the report is broken. Which is what happened."

    Dicts rather than names, which is what made the explanation impossible to carry: the
    corrections half of the same key has been a list of dicts since it learned to say why.
    """
    import store

    out = []
    for name in names:
        if store.measure_in_force_at(conn, name, judged_at, judgment_seq):
            continue
        entry = store.metric_entry(conn, name) or {}
        out.append({
            "measure": name,
            "since": ("became_expected"
                      if store.after_judgment(entry.get("confirmed_at"),
                                              entry.get("after_evaluation"),
                                              judged_at, judgment_seq)
                      else "was_retired")})
    return sorted(out, key=lambda m: m["measure"])


def _newly_reaches(conn, rule: dict, markets: list, judged_at: float, *,
                   judgment_seq: Optional[int] = None) -> Optional[str]:
    """WHY this rule reaches this brief now and did not on the day it was judged, or None.

    Three answers, not one, because they are three different things for a reader to do
    something about: `became_standing` is a rule nobody had confirmed yet, `scope_changed` is
    a rule they confirmed months ago that did not apply HERE until its scope moved, and
    `was_withdrawn` is a rule that was standing, was set aside, and has since been put back.
    Reported as one sentence — "became standing after this was judged" — the other two read as
    mistakes, and the reader who checks `correction_status` finds a confirmation date from
    before the judgment and concludes the report is broken. Which is what happened. Reported
    as "its scope changed", a withdrawal states a cause the library cannot back: that rule's
    scope never moved.

    Asked of the scope history rather than of `confirmed_at`, because those are two different
    facts and only the first is the one this report is about. `confirmed_at` is preserved
    across a re-confirmation — deliberately, it is the audit field — so a rule that was
    standing but in force NOWHERE when a brief was judged, and in force everywhere afterwards,
    looked to this report like a rule that had been applied all along. That is precisely the
    upgraded install: ten house rules reaching nothing, briefs judged without them, the repair
    offering "see which judgments this now applies to", and an empty report.

    The history also keeps the other direction honest, which a single "scope changed at"
    timestamp could not: narrowing a rule from LATAM+SEA to LATAM changes nothing about a
    LATAM judgment that WAS checked against it, and re-listing it would be a false claim in
    the more damaging direction — the report saying somebody missed something they did not.

    A database written before the history existed has none, and then the old question is the
    best available answer: was it confirmed after this judgment.
    """
    import store

    if not _applies(rule["expected_in"], markets,
                    everywhere=bool(rule.get("applies_everywhere"))):
        return None                       # it does not reach this brief even now
    history = store.correction_scope_history(conn, rule["id"])
    if not history:
        return ("became_standing"
                if store.after_judgment(rule["confirmed_at"], None, judged_at, judgment_seq)
                else None)
    # WHEN IT BECAME STANDING, from the history rather than from `confirmed_at`, so the tie
    # is settled by the same ordering the rest of this function uses. Compared on the
    # timestamp alone, a rule first confirmed in the judgment's own tick read as already
    # standing — and the report then blamed its SCOPE for a rule that had none yet.
    became = next((change for change in history if change["standing"]), None)
    confirmed_after = bool(became) and store.after_judgment(
        became["changed_at"], became["after_evaluation"], judged_at, judgment_seq)
    then = store.correction_scope_at(conn, rule["id"], judged_at,
                                     judgment_seq=judgment_seq)
    if (then and then["standing"]
            and _applies(then["expected_in"], markets,
                         everywhere=then["applies_everywhere"])):
        return None                       # it reached this brief on the day it was judged
    if confirmed_after:
        return "became_standing"          # nobody had confirmed it yet
    if then and not then["standing"]:
        return "was_withdrawn"            # standing, set aside, and since put back
    return "scope_changed"                # standing, and not here


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
        cited = _cited_by(finding)
        if cited in withdrawn and cited not in seen:
            seen.add(cited)
            out.append({"correction_id": cited, "text": withdrawn[cited]["text"],
                        # WHICH not-standing state, because the two are different things to
                        # tell somebody. A rule that is `provisional` was set aside and then
                        # REOPENED — the user clicked the product's own "put this rule back
                        # if that was not what you meant" — and reading the row as "since been
                        # set aside" tells them the opposite of what they just did.
                        "status": withdrawn[cited].get("status"),
                        "finding": finding.get("finding")})
    # The MODEL's findings only. §7.8's computed ones are the server's own facts about the
    # brief — they are not part of what the judgment argued, so counting them would mean a
    # verdict could never rest solely on anything.
    #
    # And this rests on an invariant written elsewhere: `core.save_evaluation` refuses a
    # `note`-severity `guardrail_breach` ("a guardrail breach cannot be a note"), so every
    # breach is one of the severities tested below. Named here because a change there would
    # make `sole` quietly unreachable, and nothing would fail.
    argued = [f for f in findings if (f.get("basis") or "judged") == "judged"]
    breaches = [f for f in argued if f.get("kind") == "guardrail_breach"]
    sole = bool(
        out and breaches and saved.get("verdict") != "approve"
        # Every finding that carries the verdict is a breach, and every breach rests on a
        # rule that has been withdrawn.
        and all(f.get("kind") == "guardrail_breach" for f in argued
                if f.get("severity") in ("blocking", "should_fix"))
        and all(_cited_by(f) in withdrawn for f in breaches))
    return out, sole


def _how_withdrawn(withdrawn: list) -> str:
    """Set aside, or set aside and put back but not confirmed again — one sentence for two
    states told somebody the opposite of what they had just done."""
    states = {c.get("status") for c in withdrawn}
    if states == {"provisional"}:
        return ("reopened, and applies to nothing until somebody confirms it"
                if len(withdrawn) == 1 else
                "reopened, and applying to nothing until somebody confirms them")
    if "provisional" in states:
        return "some set aside, some reopened and not confirmed again"
    return "set aside" if len(withdrawn) == 1 else "set aside"


def _and_a_stop(text: str) -> str:
    """A rule is a sentence and usually ends in a period; a second one reads as a typo."""
    return text if text.rstrip().endswith((".", "!", "?")) else f"{text}."


def _judgment_sentence(consequence: str, gaps: list, measures: list, rules: list,
                       withdrawn: list, revisited: bool = False) -> str:
    """One sentence per consequence, and none of them says the verdict would change."""
    if consequence == "stated_basis_withdrawn":
        return (f"Everything this verdict rests on is a rule that is no longer standing "
                f"({_how_withdrawn(withdrawn)}): "
                f"{_and_a_stop('; '.join(c['text'] for c in withdrawn))} The judgment is "
                f"unchanged, but its whole stated basis is no longer a rule here."
                + ("" if revisited else " Nobody has judged this subject again since."))
    if consequence == "rests_on_withdrawn":
        return (f"This judgment cites {len(withdrawn)} standing correction"
                f"{'s' * (len(withdrawn) != 1)} that "
                f"{'are' if len(withdrawn) != 1 else 'is'} no longer standing "
                f"({_how_withdrawn(withdrawn)}), so "
                f"{'those findings are' if len(withdrawn) != 1 else 'that finding is'} "
                f"anchored to something that is no longer a rule. The rest of it stands.")
    if consequence == "gap_appears":
        # WHY each gap is new, for the reason the rules half says it: a measure confirmed long
        # before this judgment, retired while it was made and revived since, is not one that
        # "became expected after this was judged" — and a reader who checks the date concludes
        # the report is broken.
        since = {m["measure"]: m["since"] for m in measures}
        fresh = [g for g in gaps if since.get(g) != "was_retired"]
        back = [g for g in gaps if since.get(g) == "was_retired"]
        how = []
        if fresh:
            how.append(f"{', '.join(fresh)} became expected here after this was judged")
        if back:
            how.append(f"{', '.join(back)} {'was' if len(back) == 1 else 'were'} retired when "
                       f"this was judged and {'has' if len(back) == 1 else 'have'} since been "
                       f"put back")
        return (f"The subject has no measured {', '.join(gaps)} on file, and "
                f"{' and '.join(how)}. Judging it again would see a gap this judgment did "
                f"not. That is a difference in the evidence, not a verdict — whether it "
                f"changes anything is the judgment's to make.")
    named = [c["text"] for c in rules]
    # WHY each one is newly here, because a reader who is told a rule "became standing after
    # this was judged" and then finds a confirmation date from before it concludes the report
    # is broken. Both sentences describe the same fact about the judgment — it was not checked
    # against this rule — and they are different facts about the RULE.
    moved = [c["text"] for c in rules if c.get("since") == "scope_changed"]
    back = [c["text"] for c in rules if c.get("since") == "was_withdrawn"]
    new_here = [c["text"] for c in rules
                if c.get("since") not in ("scope_changed", "was_withdrawn")]
    how = []
    if new_here:
        how.append(f"{len(new_here)} became standing here after this was judged")
    if moved:
        how.append(f"{len(moved)} {'was' if len(moved) == 1 else 'were'} already standing and "
                   f"did not apply to this brief until its scope changed")
    if back:
        how.append(f"{len(back)} {'was' if len(back) == 1 else 'were'} set aside when this was "
                   f"judged and {'has' if len(back) == 1 else 'have'} since been put back")
    listed = "; ".join(named[:2])
    # "and others" ends the list when there are more than two, so the stop belongs after THAT
    # — measured on the rendered string rather than on the last rule, which is what made the
    # guard wrong for exactly the case it was written for: the upgraded install, where ten
    # house rules become standing at once ("…creative. and others Whether…").
    return (f"{' and '.join(how)}, so it was never checked against "
            f"{'them' if len(named) != 1 else 'it'}: "
            f"{_and_a_stop(listed + (' and others' if len(named) > 2 else ''))}"
            f" Whether this brief breaks "
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

    chase = backlog["campaigns_total"]
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
                 markets: Optional[list] = None,
                 everywhere: bool = False,
                 campaign_types: Optional[list] = None) -> dict:
    """What promoting this would do to the library, without promoting it (§8.7, D104).

    *"This makes 14 LATAM records show as missing it."* The person confirming is exactly who
    needs that number, and needs it BEFORE they confirm — afterwards it is a surprise rather
    than a decision. Asking costs nothing and changes nothing.

    `everywhere` is §12.3's house rule, and this is the sentence somebody reads at the moment
    they decide. Computed from the market list alone it said confirming a rule the customer
    declares EVERYWHERE would affect the briefs of "no market" and 0 saved judgments — and the
    report it opens afterwards listed every judgment in the library. That is the same
    disagreement this file's `_applies` had, one surface earlier, pointing the other way.
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
                    if _would_miss(conn, r, where, name, campaign_types=campaign_types)]
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
                # The CHECKLIST question — would this rule reach that judgment's subject —
                # so it goes through the gate rather than through where the record ran. The
                # refusal is read rather than discarded: a gated subject comes back with no
                # markets, and relying on `in_force` to answer False for an empty list would
                # make this correct by the order of two lines in another module.
                if _reaches_subject(conn, r.get("campaign_id"), where, everywhere)]
    return {
        "correction_id": entry["correction_id"],
        "text": entry["text"],
        # WHAT WOULD BE IN FORCE, and separately where the rule has been seen. One key
        # carrying both is how a rule in force everywhere came to report a single market as
        # its scope — the shape this whole round is about, in a response body.
        "markets": [] if everywhere else where,
        # WHERE IT WAS SEEN, from the column that records that — not from `where`, which is
        # what would be IN FORCE and, for a rule the file scopes to markets, is the
        # declaration rather than an observation. Repeating the conflation this round removed,
        # in a key added by this round, one function along. The same expression
        # `standing_for` uses, and it cannot call the gate: the gate calls this.
        "seen_in": learning.fold_markets(entry.get("markets")),
        "applies_everywhere": everywhere,
        "judgments_affected": len(affected),
        "examples": [{"evaluation_id": r["id"], "subject_title": r["subject_title"]}
                     for r in affected[:5]],
        "basis": "computed",
        "what_it_means": (
            f"Confirming this would make every brief in "
            f"{'every market' if everywhere else (', '.join(where) or 'no market')} "
            f"judged against it from here on. "
            f"{len(affected)} saved judgment{'s' * (len(affected) != 1)} "
            f"{'were' if len(affected) != 1 else 'was'} written without it and would show in "
            f"the replay as not checked against it — none of them is rewritten."),
    }


def _would_miss(conn, record: dict, markets: list, name: str,
                campaign_types: Optional[list] = None) -> bool:
    """Whether confirming this measure would show this record as missing it.

    Through the same two functions a live checklist goes through: `_subject_checklist` for
    whether the record has one at all, and `metrics.keyed_on_reaches` for whether an entry
    scoped this way reaches it. This hand-rolled both — a record-type test of its own and a
    market test of its own — and never heard of §12.4's other key, so the preview described a
    graduation keyed on markets while the graduation it previews keys on the TYPE.
    """
    import metrics

    where, checklist = _subject_checklist(conn, record.get("id"))
    if not checklist:
        return False
    if not metrics.keyed_on_reaches(expected_in=markets, expected_for_types=campaign_types,
                                    markets=where,
                                    campaign_type=record.get("campaign_type")):
        return False
    return not metrics.carries(conn, record["id"], name)


def _reaches_subject(conn, campaign_id: Optional[str], where: list,
                     everywhere: bool) -> bool:
    """Whether a rule scoped like this would reach that judgment's subject."""
    markets, checklist = _subject_checklist(conn, campaign_id)
    return checklist and _applies(where, markets, everywhere=everywhere)


def _judgments_before(conn) -> list:
    import store
    return [r for r in store.list_evaluations(conn) if r.get("created_at") is not None]
