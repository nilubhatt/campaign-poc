"""
Comparing two versions of a brief (§5.4, idea D).

Lifted out of `core` unchanged. It was a true leaf there — 783 lines that nothing else in
`core` called into, reaching back only for two constants — and `core` re-exports every name
below, so `core.diff_campaigns` and the private helpers other modules and tests already reach
for keep working exactly as before.

`core` is imported INSIDE the two functions that need it rather than at module load: `core`
imports this module, so a module-level import back would be a cycle. That is the pattern this
codebase already uses wherever two modules need each other.
"""
from __future__ import annotations

import re
import time
import unicodedata

import actions
import chunking
import corrections
import extract
import facts
import identity
import notices
import store
import version

# ── comparing two versions of a brief (§5.4, idea D) ────────────────────────
#
# "Working out what actually changed between the Colombia versions — which corrections were
# adopted, which were ignored, which facts went stale — was done by hand, and it is the
# single most common real task this product faces."
#
# "Computed against the earlier version's evaluation findings" is what makes this a
# computation rather than a second judgment. Once both versions have been evaluated the
# comparison is between two sets of findings, and every bucket below is a fact about the
# record rather than an opinion about the decks.
#
# Two reviews of one defect rarely word it identically, so matching on exact text would
# report every ignored correction as a new one — the most flattering error available.

# How alike two findings must read before they are the same finding. Deliberately generous:
# the cost of splitting one problem into two is a brief credited for a correction it never
# made, and the cost of merging two is one line a reader can see is wrong.
_SAME_FINDING = 0.62


def _looks_like(a: str, b: str) -> float:
    import difflib

    return difflib.SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def _pair_score(earlier: dict, later: dict) -> float:
    """How likely these two describe the same problem.

    Category is a hint rather than a gate: the same defect gets filed under different
    categories by two reviewers, but when both agree on one it is the only signal available
    to separate two findings whose wording is equally close.
    """
    score = _looks_like(earlier.get("finding"), later.get("finding"))
    if earlier.get("category") and earlier["category"] == later.get("category"):
        score += 0.1
    return score


def _pair_up(earlier: list[dict], later: list[dict]) -> dict:
    """Best-first assignment over ALL candidate pairs, one earlier to one later.

    The first version walked the earlier findings in order and gave each the best REMAINING
    later one, so with two earlier findings competing for one later finding the one processed
    first won — even at 0.79 against the other's 0.98. Which correction was reported as
    raised again therefore depended on the order somebody typed them, or on the severity sort
    that reorders them before ids are assigned. The same two decks must not diff differently.

    Scoring every pair and taking them best-first makes the result a property of the two sets
    rather than of their order. Ties break on the findings' own text, so it is deterministic
    rather than merely consistent.
    """
    pairs = sorted(
        ((_pair_score(e, l), (e.get("finding") or ""), (l.get("finding") or ""), i, j)
         for i, e in enumerate(earlier) for j, l in enumerate(later)),
        key=lambda t: (-t[0], t[1], t[2]))
    taken_earlier: set = set()
    taken_later: set = set()
    assignment: dict = {}
    for score, _et, _lt, i, j in pairs:
        if score < _SAME_FINDING:
            break
        if i in taken_earlier or j in taken_later:
            continue
        taken_earlier.add(i)
        taken_later.add(j)
        assignment[i] = (j, score)
    return assignment


# How many passages a diff shows per side. Two long decks with nothing in common would
# otherwise return both in full, in a reply somebody has to read.
_MAX_PASSAGES = 10


# A single passage's ceiling in the reply. `_MAX_PASSAGES` bounds the COUNT and nothing
# bounded the size: a deck with no blank lines in it is one paragraph, so one "passage" was
# 78 KB on each side and a diff of two such decks was a 156 KB tool result. Truncated with the
# cut marked, because a quotation that has been shortened must never look complete.
_MAX_PASSAGE_CHARS = 1200


def _folded_passage(text: str) -> str:
    """The form two passages are compared on: whitespace, case and Unicode composition folded.

    NFC is not a nicety. "Café" is one codepoint on a Windows machine and "e" plus a combining
    accent on a Mac, the two are the same sentence to every reader, and Python compares them
    unequal — so a deck that had been through both reported every accented paragraph as
    removed AND added, which is the diff saying the brief changed when nobody touched it.
    """
    import unicodedata

    return unicodedata.normalize("NFC", " ".join(str(text or "").split())).casefold()


def _short_passage(text: str) -> str:
    """One passage, bounded, with the cut visible."""
    whole = str(text or "").strip()
    if len(whole) <= _MAX_PASSAGE_CHARS:
        return whole
    return whole[:_MAX_PASSAGE_CHARS].rstrip() + f" … [{len(whole)} characters in all]"


def _passages_changed(conn, earlier: str, later: str) -> dict:
    """Added and removed passages between two versions, by layer (§12.4/D63).

    From what is ALREADY STORED for each version: nothing is extracted again and no file is
    re-read, which is the item's framing. The two layers come from different places, and the
    comment inside `by_layer` says why — the body from the record's own `deck_text`, the
    commentary from its chunks. D63's wording is "from the chunks already stored"; taking that
    literally made every version differ on its own title, because the first chunk carries the
    title and detail as a summary. The departure is deliberate; claiming both was the error.

    Compared on a FOLDED form so that whitespace and case do not register as a change — a diff
    that reports every paragraph because the chunker packed them differently is one nobody
    reads twice. Unicode is normalised in the same fold: "Café" typed as one codepoint and as
    "e" plus an accent are the same sentence to every reader and different bytes to Python, so
    a deck round-tripped through a Mac reported its whole text as replaced.

    By layer, and the layer is load-bearing. A paragraph gone from the BODY is the brief
    changing; a comment gone is the client's remark being dropped, which is a different and
    usually more interesting event. One list would make a deleted objection look like an edit.
    That is also why commentary is filled FIRST when the reply has to be trimmed: body first,
    a fifteen-paragraph rewrite pushed out every dropped client comment, so the layer the
    docstring calls more interesting was the layer truncation removed.
    """
    def by_layer(campaign_id: str) -> dict:
        out: dict = {"body": {}, "commentary": {}}
        record = store.get_campaign(conn, campaign_id) or {}
        # The BODY from `deck_text`'s own paragraphs, not from the chunks. A chunk is not a
        # passage: `chunking.pack` merges units up to 1800 characters, and the first chunk
        # carries the title and detail as a summary — so every version differed on its own
        # title and the whole deck read as replaced. §6.1 made the same call for quotes, for
        # the same reason: the chunk boundary is one the document does not have.
        for unit in (record.get("deck_text") or "").split("\n\n"):
            folded = _folded_passage(unit)
            if folded:
                out["body"].setdefault(folded, _short_passage(unit))
        # Commentary IS one chunk per comment and never merged, so its chunks are its
        # passages — and they carry the author and anchor nothing else can reconstruct.
        for row in conn.execute(
                "SELECT text FROM campaign_chunks WHERE campaign_id = ? AND kind = "
                "'commentary'", (campaign_id,)).fetchall():
            folded = _folded_passage(row["text"] or "")
            if folded:
                out["commentary"].setdefault(folded, _short_passage(row["text"] or ""))
        return out

    before, after = by_layer(earlier), by_layer(later)
    added, removed, more = [], [], 0
    # COMMENTARY FIRST, and this ordering is the whole of the layer rule at the point where it
    # costs something. Body first, a deck with fifteen rewritten paragraphs and two dropped
    # client comments returned ten body passages and ZERO commentary, with the comments folded
    # into an undifferentiated `more: 12` — the layer this function's own docstring calls "the
    # more interesting of the two" was the one the cap removed, every time, because there is
    # always more body than commentary.
    for layer in ("commentary", "body"):
        gone = [before[layer][k] for k in before[layer] if k not in after[layer]]
        new = [after[layer][k] for k in after[layer] if k not in before[layer]]
        more += max(0, len(gone) - _MAX_PASSAGES) + max(0, len(new) - _MAX_PASSAGES)
        removed += [{"layer": layer, "text": t} for t in gone[:_MAX_PASSAGES]]
        added += [{"layer": layer, "text": t} for t in new[:_MAX_PASSAGES]]

    return {
        "added": added[:_MAX_PASSAGES], "removed": removed[:_MAX_PASSAGES],
        "more": more + max(0, len(added) - _MAX_PASSAGES) + max(0, len(removed) - _MAX_PASSAGES),
        "basis": "computed",
        "what_it_means": (
            "What the two decks say differently, from the passages already stored for each. "
            "`layer` matters: a passage gone from `body` is the brief changing, and one gone "
            "from `commentary` is a remark somebody made about it being dropped — which is "
            "usually the more interesting of the two. Whether a change ANSWERS anything is a "
            "reading of both decks and is not asserted here."
            if added or removed else
            "The two decks say the same things, passage for passage. Any difference between "
            "them is in the structured fields rather than the text."),
    }


_MAX_CLIENT_ASKS = 8

# The commentary kinds that are SOMEBODY ELSE writing on the deck (§12.4/D62). `extract` reads
# three, and the third is not one of these: a `speaker_note` is the notes slide, which is the
# deck author's own presenter script. All three are worth storing and searching — the notes
# slide often says what the brief means — but only these two are an ASK somebody made of the
# work. Read through a list with the kind stripped off, "Presenter: remember to smile" came
# back as what the client wrote when they sent the deck back.
_SOMEBODY_ELSES_REMARK = ("comment", "annotation")

# What the earlier deck's VERDICT does to the weight of its asks (§12.4/D15 read beside D62).
# One sentence each, and they are genuinely different readings of the same quotation — which
# is the argument for recording the verdict at all.
_WHAT_THE_VERDICT_DOES_TO_THE_ASKS = {
    "rejected": ("The earlier deck was REJECTED, so these are the objections it did not "
                 "survive. A version answering them is answering the reason it was refused."),
    "approved_with_changes": ("The earlier deck was signed off CONDITIONALLY, so these are "
                              "what it was conditional on. `approval_note` says which."),
    "approved": ("The earlier deck was signed off as it stood, so these are remarks made "
                 "alongside a yes rather than conditions on one — worth reading and not "
                 "worth treating as a blocker."),
    "withdrawn": ("The earlier deck was taken back before a verdict, so nobody said no to "
                  "it. These are the remarks it had collected by then."),
}


def _what_the_client_asked_for(conn, earlier: str) -> dict:
    """The comments on the earlier deck, as the asks a new version is answering (§12.4/D62).

    This is the §12.4/D15 decision put to work: a returned deck's tracked comments ARE the
    approval notes, so they are also the corrections the next version is responding to. The
    product was reading them as searchable text and as nothing else.

    It does NOT decide whether each one was addressed. Whether "drop the third colourway" was
    done is a judgment about two documents, and asserting it from a keyword overlap is the
    confident-unfounded claim this product exists to avoid — `adopted` says what the LIBRARY's
    findings did, from finding ids, which is a fact. This says what was asked, quotes it, and
    leaves the reading to whoever is looking at both decks.
    """
    # `get_commentary`, not `text_on_file`. The latter strips `kind`, and the three kinds are
    # not the same claim: a `comment` and a PDF `annotation` are somebody writing ON the deck,
    # and a `speaker_note` is the DECK AUTHOR'S OWN presenter script. Read through the stripped
    # list, "Presenter: remember to smile, and skip slide 4" was reported to the model as "what
    # the client wrote when they sent it back" — the agency's own note, filed as the client's
    # ask, by a product whose entire thesis is that it can say where a claim came from.
    everything = store.get_commentary(conn, earlier)
    rows = [row for row in everything
            if row.get("kind") in _SOMEBODY_ELSES_REMARK and (row.get("text") or "").strip()]
    own_notes = len(everything) - len(rows)
    items = []
    for row in rows[:_MAX_CLIENT_ASKS]:
        item = {"text": " ".join((row.get("text") or "").split())}
        # The author where the file carried one, because "who asked for this" is half of what
        # makes an ask worth answering — and because an unattributed remark must not be
        # narrated as the client's either.
        for field in ("author", "slide", "page"):
            if row.get(field):
                item[field] = row[field]
        items.append(item)
    if not items:
        return {"count": 0, "items": [], "basis": "computed",
                "what_it_means": (
                    "The earlier deck carries no remarks anybody wrote ON it"
                    + (f" — only {own_notes} speaker note(s), which are the deck author's own "
                       f"script rather than anything a reviewer asked for."
                       if own_notes else
                       " — either nobody wrote any, or the file was never read for them, "
                       "which `commentary_never_read` in `gaps` tells apart.")),
                }
    named = sorted({str(row["author"]).strip() for row in rows if (row.get("author") or "").strip()})
    # §12.4/D15's verdict, BESIDE the asks, because the two are one fact read apart. "Drop the
    # third colourway" under `rejected` is why the deck did not proceed; the same sentence
    # under `approved_with_changes` is what the sign-off was conditional on; under `approved`
    # it is a remark somebody made in passing. The verdict was stored and read by nothing, and
    # this is the surface where not having it changes what the asks MEAN.
    verdict = (store.get_campaign(conn, earlier) or {}).get("approval")
    said_so = (store.get_campaign(conn, earlier) or {}).get("approval_by")
    return {
        "count": len(rows), "items": items, "basis": "computed",
        **({"approval": verdict, "approval_by": said_so,
            "approval_means": _WHAT_THE_VERDICT_DOES_TO_THE_ASKS[verdict]}
           if verdict in _WHAT_THE_VERDICT_DOES_TO_THE_ASKS else {}),
        # `some_unattributed` rather than a sentence, so a caller can act on it: a reader who
        # cannot tell "R. Vega asked for this" from "somebody did" will eventually quote the
        # second as the first.
        "authors": named, "some_unattributed": any(not (r.get("author") or "").strip()
                                                   for r in rows),
        "what_it_means": (
            f"{len(rows)} remark(s) written on the earlier deck by "
            + (f"{', '.join(named)}" if named else "somebody the file does not name")
            + f": the asks this version is answering, quoted rather than judged — whether "
              f"each was addressed is a reading of both decks, and this product does not "
              f"assert it from word overlap. `adopted` beside this is a different claim: what "
              f"the LIBRARY's own findings did, from their ids."
            + (f" {own_notes} speaker note(s) on the deck are NOT here: those are the deck "
               f"author's own script, not a reviewer's ask." if own_notes else "")),
    }


# §12.4/D113. The stored vocabulary cannot be renamed — every saved row and every tool
# argument is written in it — so each surface says which kind it means, in the words a reader
# uses. `corrections._public` carries the standing-rule version of this sentence.
_WHAT_A_DIFF_CORRECTION_IS = (
    "A correction HERE is an observation about two documents: a finding raised on the earlier "
    "version that the later one answered. It binds nothing and judges nothing else. It is not "
    "a STANDING correction — a rule this library watched recur until somebody confirmed it, "
    "which then judges every brief in its markets — and it is not a RECOMPUTED number, which "
    "is a corrected value for one measure on one campaign. This product calls all three "
    "corrections and they carry completely different weight.")


def diff_campaigns(conn, *, earlier: str, later: str) -> dict:
    """What changed between two versions of the same brief.

    Returns the four the review named — `adopted`, `ignored`, `newly_introduced`,
    `carried_stale` — plus `no_longer_raised`, which the review folded into "adopted" and
    which is not the same thing: a finding neither resolved nor repeated was either fixed
    without being recorded or missed by the second review, and the record cannot tell those
    apart. Counting it as adopted would credit a brief for work nobody verified.

    Nothing is guessed. Where a version has not been evaluated there is nothing to compute,
    and `comparable` is false — because "you ignored my correction" is an accusation, and
    inferring it from deck text would be a judgment dressed as a computation.
    """
    import core

    first = store.get_campaign(conn, earlier)
    second = store.get_campaign(conn, later)
    for cid, record in ((earlier, first), (later, second)):
        if record is None:
            return {"error": f"campaign {cid} not found"}
    if earlier == later:
        return {"error": "those are the same campaign; pass the two versions you want "
                         "compared"}

    # Which version is earlier decides what "adopted" means, so getting it backwards inverts
    # the entire answer — which is why it is corrected only on evidence somebody RECORDED.
    # Creation order is UPLOAD order: an organisation seeding its archive uploads v1 after
    # v2 as a matter of course, and silently reversing their question, with a JSON field they
    # never see as the only tell, answers something they did not ask.
    reordered = False
    order_basis = "as_given"
    warnings: list[dict] = []
    # Adjacency is not the question — ANCESTRY is. Checking only the two `supersedes` columns
    # meant `diff(v1, v3)` across a real chain still warned "no supersedes link", while the
    # chain walk below happily used the link; and `diff(v3, v1)` was silently answered
    # backwards with the chain never walked, though the library holds the whole thing. The
    # order check and the chain walk disagreed about what "linked" means.
    if earlier in store.supersession_chain(conn, later):
        order_basis = "supersession"
    elif later in store.supersession_chain(conn, earlier):
        first, second = second, first
        earlier, later = later, earlier
        reordered = True
        order_basis = "supersession"
    else:
        warnings.append(notices.notice(
            "version_order_unverified",
            detail=f"no supersedes link between {earlier} and {later}",
            next_actions=[]))

    # D64: every version BETWEEN the two, when they are actually linked. Comparing only the
    # two endpoints reported a finding v2 explicitly resolved as `no_longer_raised` against
    # v3 — "neither resolved nor repeated", when the library holds the record of it being
    # resolved. That is the library forgetting its own evidence and then hedging about it.
    chain = store.supersession_chain(conn, later)
    through = chain[chain.index(earlier):] if earlier in chain else [earlier, later]

    before = store.latest_evaluation_for_campaign(conn, earlier)
    after = store.latest_evaluation_for_campaign(conn, later)
    # What every judgment along the way recorded as resolved, by finding id. A correction
    # taken at any point in the chain was taken.
    resolved_along_chain: dict = {}
    # `[1:]` — everything AFTER the earlier endpoint. Worth saying that this slice cannot
    # currently change the answer and is kept for intent: a judgment's `resolved` rows name
    # findings from the version BEFORE it, never its own, so folding in the earlier
    # endpoint's rows adds ids that are not in the set being matched against. Review's
    # mutation of it survives the suite and no honest test kills it, because the two forms
    # are equivalent on every input the schema can produce. Left explicit rather than
    # simplified, so that if `resolved` ever names an arbitrary id the slice is already
    # saying which versions are in scope.
    for link in through[1:]:
        judgment = store.latest_evaluation_for_campaign(conn, link)
        for row in ((judgment or {}).get("resolved") or []):
            if row.get("finding_id"):
                resolved_along_chain[row["finding_id"]] = {**row, "in_version": link}

    # `comparable` means both sides carry a STRUCTURED judgment. A pre-§2.4 evaluation reads
    # back with no verdict and an empty findings list, and treating it as comparable made
    # every later finding "newly introduced" — a v2 blamed for everything its own v1 review
    # had also found.
    #
    # The marker is the VERDICT, not a non-empty findings list: an approve with nothing wrong
    # is a complete structured judgment that happens to have no findings, and the whole point
    # of this tool is to recognise the version where everything got fixed. (`get_evaluation`
    # draws the legacy line the same way.)
    structured = [bool(e and e.get("verdict")) for e in (before, after)]

    result = {
        "earlier": {"campaign_id": earlier, "title": first["title"]},
        "later": {"campaign_id": later, "title": second["title"]},
        "arguments_reordered": reordered,
        "order_basis": order_basis,
        # Which records the answer was assembled from. A diff across a chain is a different
        # claim from a diff between two adjacent versions, and a reader cannot tell without
        # being told.
        "through": through,
        "adopted": [], "raised_again": [], "newly_introduced": [], "no_longer_raised": [],
        # BOTH judgments' citations. Only the earlier one was checked, so a later judgment
        # resting on a record since replaced reported nothing — and the later judgment is
        # the one somebody is about to act on.
        "carried_stale": (_stale_citations(conn, before, "earlier")
                          + _stale_citations(conn, after, "later")),
        "record_changes": _record_changes(first, second),
        # §12.4/D113, set HERE rather than at the end of the function: this returns early for
        # records with no structured judgment on both sides, so a sentence appended at the
        # bottom reached only half the callers — and the half it missed is the ordinary one.
        "what_a_correction_means_here": _WHAT_A_DIFF_CORRECTION_IS,
        # D58: what the BRIEFS say differently, not just what their structured fields do. A
        # budget dropped, a channel gone, a date contradiction introduced — all invisible
        # until §7.1 could read them, and all things a reader of a v1→v2 diff is asking about.
        "fact_changes": _fact_changes(conn, earlier, later),
        # §12.4/D63: what the two decks actually SAY differently, passage by passage and by
        # layer. `fact_changes` says a budget moved and `record_changes` says a field did;
        # neither says the third-colourway paragraph is gone, which is what somebody comparing
        # two versions is looking at. The chunks were already stored per version and nothing
        # had read them against each other.
        "passages": _passages_changed(conn, earlier, later),
        # §12.4/D62: what the CLIENT asked for, from the comments on the earlier deck.
        # "In the Colombia case 'corrections' meant the client's tracked comments at least as
        # much as the library's findings" — and a diff that reports only which of its OWN
        # findings were addressed is reporting on itself. What the client actually wrote when
        # they sent the deck back was being read as nothing.
        #
        # Reported beside `adopted` and never merged into it: one is what this product
        # decided, the other is what the client instructed, and blending them would let the
        # library take credit for somebody else's sentence.
        "what_the_client_asked_for": _what_the_client_asked_for(conn, earlier),
        "comparable": all(structured),
        "warnings": warnings,
    }

    result["counts"] = {name: len(result[name]) for name in
                        ("adopted", "raised_again", "newly_introduced", "no_longer_raised",
                         "carried_stale")}

    if not all(structured):
        if not (before and after):
            missing = "earlier" if not before else "later"
            result["why_not_comparable"] = (
                f"The {missing} version has never been evaluated, so there are no findings "
                f"to compare. What changed between the decks is readable, but which "
                f"corrections were taken is not — that is a judgment about the earlier "
                f"review, and guessing it would mean telling somebody they ignored advice "
                f"nobody checked.")
        else:
            side = "earlier" if not structured[0] else "later"
            result["why_not_comparable"] = (
                f"The {side} version's judgment predates the structured findings schema, so "
                f"it has no findings to compare — only the original free text. Treating it "
                f"as comparable would report every finding in the other version as newly "
                f"introduced.")
            return result
        unjudged = earlier if not before else later
        unjudged_record = store.get_campaign(conn, unjudged)
        result["next_actions"] = actions.trim([actions.action(
            f"Evaluate \u201c{store.get_campaign(conn, unjudged)['title']}\u201d, so the "
            f"two versions can be compared",
            "prepare_evaluation",
            why="A version with no judgment on file cannot be compared with one that has.",
            # Whatever text the record actually has. `detail or title` sent the title alone
            # for a deck-only upload, so accepting "evaluate this version" judged eleven
            # characters — a prefilled placeholder, which §5.2 forbids.
            consent="ask", subject_title=unjudged_record["title"],
            campaign_id=unjudged,
            proposal_text="\n\n".join(
                part for part in (unjudged_record.get("detail"),
                                  unjudged_record.get("deck_text"))
                if part) or unjudged_record["title"])])
        return result

    resolved_by_id = {r["finding_id"]: r for r in (after.get("resolved") or [])
                      if r.get("finding_id")}
    # A correction recorded anywhere along the chain counts, not only in the final judgment.
    resolved_by_id = {**resolved_along_chain, **resolved_by_id}
    later_findings = list(after.get("findings") or [])
    repeats_by_id = {f["repeats"]: f for f in later_findings if f.get("repeats")}
    matched_later: list = []
    later_categories = {f.get("category") for f in later_findings if f.get("category")}

    # Everything identity already settles is taken out before wording is consulted at all,
    # so a resemblance can never outrank a statement somebody made.
    earlier_findings = list(before.get("findings") or [])
    settled = {f.get("id") for f in earlier_findings
               if f.get("id") in resolved_by_id or f.get("id") in repeats_by_id}
    by_text = _pair_up(
        [f for f in earlier_findings if f.get("id") not in settled],
        [f for f in later_findings if not f.get("repeats")])
    text_match = {}
    unsettled = [f for f in earlier_findings if f.get("id") not in settled]
    candidates = [f for f in later_findings if not f.get("repeats")]
    for i, (j, score) in by_text.items():
        text_match[id(unsettled[i])] = (candidates[j], score)

    for finding in earlier_findings:
        entry = {k: finding.get(k) for k in ("id", "severity", "kind", "departure",
                                             "category",
                                             "finding")}
        if finding.get("id") in resolved_by_id:
            row = resolved_by_id[finding["id"]]
            entry["basis"] = "computed"
            entry["now"] = row.get("now")
            # Which version actually recorded it, when that was not the one being compared
            # against. Without it, a v1→v3 diff reported a correction as adopted with `now`
            # text written by v2's reviewer and nothing said so — and the caveat below is
            # only true of the chain case, so a reader has to be able to tell them apart.
            if row.get("in_version") and row["in_version"] != later:
                entry["resolved_in"] = row["in_version"]
                # Between adjacent versions, "adopted" means the later reviewer confirmed it.
                # Across a chain it means confirmed ONCE and not re-examined since: v3's
                # reviewer compared against v2 and never looked at this finding again, so a
                # regression between v2 and v3 would not show here. This module caveats
                # weaker claims than that one.
                entry["caveat"] = (
                    f"Recorded as resolved in {row['in_version']}, not re-examined in "
                    f"{later}. Adopted once, not confirmed since.")
            result["adopted"].append(entry)
            continue

        # Identity first. A later finding that names the earlier one it repeats is a claim
        # somebody made, and the only basis on which "this correction was not taken" can be
        # stated as fact.
        named = repeats_by_id.get(finding.get("id"))
        if named is not None:
            matched_later.append(named)
            result["raised_again"].append({**entry, "basis": "computed", "match": "id",
                                           "raised_again_as": named.get("finding"),
                                           **_reread(finding, named)})
            continue

        # Wording, as a fallback that is labelled as one. Character similarity scored
        # "adidas-affiliated" against "Nike-affiliated" at 0.89 and the same problem reworded
        # at 0.36 — it measures phrasing, not meaning, so it is reported as a resemblance
        # and marked `judged` so nothing downstream states it as fact.
        paired = text_match.get(id(finding))
        if paired is not None:
            again, _score = paired
            matched_later.append(again)
            result["raised_again"].append({
                # D61/§7.8: `heuristic`, not `judged`. Nobody judged this — a threshold did.
                **entry, "basis": "heuristic", "match": "text", **_reread(finding, again),
                "similarity": round(_looks_like(finding.get("finding"),
                                                again.get("finding")), 2),
                "raised_again_as": again.get("finding"),
                "caveat": "Matched on how alike the two read, not on either review saying "
                          "they are the same problem. Check both before telling anyone a "
                          "correction was not taken.",
            })
            continue

        # §10.2/D59: somebody ANSWERED this one. `no_longer_raised` is the "we cannot tell"
        # bucket and all three of its caveats below end in "though nobody recorded it" —
        # which is D59's complaint exactly, and the row asks for a way to record it. So an
        # answered finding must leave the bucket, or `answer_finding` writes a field nothing
        # reads: a tool whose output nobody consumes cannot fail, which is D116's shape one
        # level down.
        #
        # `stated`, never `computed`. Every other entry in `adopted` is the SERVER matching a
        # `resolved` row by id; this is somebody's word, and a reader who cannot tell those
        # apart will eventually cite one as the other.
        settled_answer = (finding.get("settled") or {}).get("answer")
        if settled_answer == "deliberate":
            # Not `adopted` — nothing was adopted; they kept it and said why. And not the
            # "we cannot tell" bucket either, whose three caveats below all end "though
            # nobody recorded it", which this answer disproves in the user's own words.
            result["no_longer_raised"].append({
                **entry, "basis": "stated", "settled": finding["settled"],
                "caveat": (
                    f"Not raised again, and it would not be: {finding['settled']['said_by']} "
                    f"recorded that this was deliberate — “{finding['settled']['note']}”. "
                    f"That is their word, not a comparison of the two versions."),
            })
            continue
        if settled_answer in ("fixed", "does_not_apply", "misread"):
            result["adopted"].append({
                **entry, "basis": "stated", "settled": finding["settled"],
                "now": finding["settled"]["note"],
                "caveat": (
                    f"{finding['settled']['said_by']} recorded this as "
                    f"{core._SETTLED_AS[settled_answer]} "
                    f"directly, rather than it being matched to anything in the later "
                    f"judgment. It is their word, not a comparison of the two versions."),
            })
            continue

        entry["basis"] = "computed"
        # The caveat is split by what the later review actually covered, because "we cannot
        # tell" is not equally true in both cases.
        if after.get("verdict") == "approve" and not later_findings:
            entry["caveat"] = ("The later review approved this version with no findings at "
                               "all, which is a positive statement that nothing blocks it — "
                               "so this was most likely addressed, though no one recorded "
                               "it against this finding.")
        elif finding.get("category") and finding["category"] in later_categories:
            entry["caveat"] = (f"The later review did raise other {finding['category']} "
                               f"findings, so it looked at this area and did not raise this "
                               f"one — more likely fixed than overlooked, though nobody "
                               f"recorded it.")
        else:
            entry["caveat"] = ("Neither recorded as resolved nor raised again, and the later "
                               "review raised nothing in this area at all — so it was either "
                               "fixed without being recorded or not looked at. The record "
                               "cannot tell which.")
        result["no_longer_raised"].append(entry)

    for finding in later_findings:
        if finding in matched_later:
            continue
        entry = {k: finding.get(k) for k in ("id", "severity", "kind", "departure",
                                             "category",
                                             "finding")}
        entry["basis"] = "computed"
        result["newly_introduced"].append(entry)

    result["counts"] = {name: len(result[name]) for name in
                        ("adopted", "raised_again", "newly_introduced", "no_longer_raised",
                         "carried_stale")}
    result["next_actions"] = _offer_to_link_versions(first, second, result)
    return result


def _offer_to_link_versions(first: dict, second: dict, diff: dict) -> list[dict]:
    """Offer to record that one version replaces the other — D41, and only here.

    5.2 offered this prefilled from `closest_precedent`: a similarity match, ASSERTED by the
    model. "This replaces that" is a claim about somebody's intent and a similarity score is
    a fact about text, and the two are not the same kind of thing — both reviewers condemned
    it independently. What was wrong was never the offer; it was the evidence.

    **The first version of this fix got the evidence wrong too, and review said so in the
    same words.** It fired whenever a diff of two judged records completed — which is not a
    fact about the records, it is a fact about which two ids the CALLER passed, and the
    caller is the model 5.2 condemned. Two unrelated campaigns got an offer to hide one of
    them, in the same response whose warning said the library cannot tell which came first.

    So the gate is a finding id: a later finding whose `repeats` names an earlier one, or a
    `resolved` row naming an earlier finding. Both are statements somebody made about these
    two records being versions of one brief — which is this module's own rule, that a
    resemblance never outranks a statement. Unrelated campaigns share no finding ids and get
    nothing.

    The offer names the record it would hide, in the label, because that is the consequence
    the user is agreeing to and it is invisible in the arguments. §5.2's other lesson: this
    is offered, never taken — accepting hides a record from every search, and `supersedes`
    can now be cleared, which is what makes offering it defensible at all.
    """
    # Either endpoint already linked, in either direction. Reading only the two `supersedes`
    # columns missed fan-in: an earlier record already replaced by a THIRD record was still
    # offered up to be replaced again, silently making two records claim the same one.
    if (second.get("supersedes") or first.get("supersedes")
            or second.get("is_superseded") or first.get("is_superseded")):
        return []
    # A statement, not a resemblance: an id-matched repeat, or a recorded resolution.
    linked = any(entry.get("match") == "id" for entry in diff["raised_again"])
    # `computed` only. §10.2/D59 put a second kind of entry in `adopted` — a finding somebody
    # settled DIRECTLY with `answer_finding` — and that is a statement about one record's
    # finding, not about the two records being versions of each other. Counting it here would
    # make this offer's own `why` false ("a judgment of one names a finding from the judgment
    # of the other by id") and would hide a record from every future search on evidence that
    # never mentioned the other record.
    linked = linked or any(entry.get("basis") == "computed" for entry in diff["adopted"])
    if not linked:
        return []
    return actions.trim([actions.action(
        f"Record that “{second['title']}” replaces “{first['title']}”, "
        f"which removes “{first['title']}” from future search results",
        "update_campaign",
        why="A judgment of one names a finding from the judgment of the other by id, so "
            "somebody has already treated these as versions of one brief — but nothing links "
            "the records, so searches still return both and the older can be cited as "
            "precedent for the newer.",
        consent="ask", campaign_id=second["id"], supersedes=first["id"])])


def _reread(earlier: dict, later: dict) -> dict:
    """Did the second reader read the same difference differently? (§6.2)

    A finding raised in both versions is reported as "raised again", which reads as a
    correction not taken. But a departure the first review called a `regression` and the
    second called a `possible_improvement` is not an ignored correction — it is the library
    changing its mind, and it is the review's own headline case: the UAE brief's claw machine
    was a departure that turned out better than the precedent. Reported without this, the
    comparison files exactly that story as a repeat defect.
    """
    import core

    before, after = earlier.get("departure"), later.get("departure")
    if not before or not after or before == after:
        return {}
    return {
        "departure_now": after,
        # A stable code, not prose: `softened` is the claw machine, `hardened` is a second
        # reader who decided the difference was worse than the first thought.
        "reread": ("softened" if core._DEPARTURES.index(after) > core._DEPARTURES.index(before)
                   else "hardened"),
    }


def _fact_changes(conn, earlier: str, later: str) -> list:
    """Computed facts whose STATUS changed between two versions (D58, §7.1).

    Status, not value: "the budget went from 40,000 to 50,000" is a change in the brief that
    a reader can see for themselves, while "the budget disappeared" is a change in what the
    brief can be judged on. The first is content and the second is coverage, and only the
    second is what a diff of computed facts is for.
    """
    before, after = facts.for_campaign(conn, earlier), facts.for_campaign(conn, later)
    # A missing record is not a version where every fact changed. `for_campaign` returns `{}`
    # for an id that resolves to nothing, and comparing that against a real record reported
    # the whole checklist as having appeared out of nowhere.
    if not before or not after:
        return []
    changed = []
    for code in sorted(set(before) | set(after)):
        was = before.get(code, {}).get("status")
        now = after.get(code, {}).get("status")
        if was == now:
            continue
        changed.append({
            "code": code, "was": was, "now": now, "basis": "computed",
            "what_it_means": (after.get(code) or before.get(code) or {}).get("what_it_means"),
        })
    return changed


def _record_changes(first: dict, second: dict) -> dict:
    """What the two records themselves say differently.

    A v2 that drops a market from a LATAM brief is the largest change a version can carry,
    and it diffed as identical — the comparison only ever looked at findings. Same for a
    performance tag that lost its `verified` source, which changes how the record weighs as
    precedent. Set differences on structured fields, so all of it is genuinely computed.

    Unchanged fields are absent rather than listed as empty: a diff that lists everything it
    checked is a diff nobody reads.
    """
    changes: dict = {}

    for field in ("markets",):
        before, after = set(first.get(field) or []), set(second.get(field) or [])
        if before != after:
            changes[field] = {"removed": sorted(before - after),
                              "added": sorted(after - before)}

    def tag_set(record):
        return {(t.get("value"), t.get("source")) for t in (record.get("tags") or [])}

    before_tags, after_tags = tag_set(first), tag_set(second)
    if before_tags != after_tags:
        before_values = {v for v, _ in before_tags}
        after_values = {v for v, _ in after_tags}
        entry = {"removed": sorted(before_values - after_values),
                 "added": sorted(after_values - before_values)}
        # A tag whose value survived but whose source did not: a performance claim that was
        # backed by measurement and now is not weighs differently as precedent.
        downgraded = sorted(v for v, source in before_tags
                            if source == "verified" and (v, "verified") not in after_tags
                            and v in after_values)
        if downgraded:
            entry["no_longer_verified"] = downgraded
        changes["tags"] = entry

    for field in ("status", "collection", "market", "region"):
        if (first.get(field) or None) != (second.get(field) or None):
            changes[field] = {"was": first.get(field), "now": second.get(field)}

    return changes


def _stale_citations(conn, evaluation, cited_by: str) -> list[dict]:
    """Citations the earlier judgment rested on that the library no longer treats as current.

    The computable half of "which facts went stale": a verdict that leaned on a campaign
    since replaced leaned on something no search would now return.
    """
    if not evaluation:
        return []
    superseded = store.get_superseded_campaign_ids(conn)
    stale = []
    for cited in evaluation.get("cited_ids") or []:
        if cited not in superseded:
            continue
        record = store.get_campaign(conn, cited)
        if not record:
            continue
        stale.append({
            "campaign_id": cited,
            "cited_by": cited_by,
            "title": record["title"],
            "superseded_by": record["superseded_by"],
            "basis": "computed",
            "why_it_matters": "The earlier judgment cited this record, and it has since "
                              "been replaced — so that part of the reasoning rests on "
                              "something no search would return today.",
        })
    return stale


