"""
§8.6: standing corrections, on the same loop as measures.

The review: *"New client feedback is the same shape of event as a new metric. 'We want the
seeding box to carry one colourway' should travel the identical path: provisional on first
mention, counted, promoted to the checklist once it recurs across markets and a person confirms
it, with provenance attached. One learning mechanism for both keeps the rulebook coherent, and
means the ten standing corrections already extracted have somewhere to live and grow rather
than being frozen at whatever was true the day they were written."*

Every decision in the loop is `learning`'s, not this module's — the thresholds, the gate, the
order status is asked in, the fold on markets, the version folding, the three reasons a record
has no checklist. What is here is the part that is genuinely about corrections: they are
sentences rather than keys, "the same correction" needs deciding, and every mention carries its
own provenance.

**Provenance is required.** *"Each needs its provenance shipped with it so a judgment can cite
where the rule came from."* A correction with none is not a standing correction, it is an
opinion in a text field, and a judgment resting on it could say only "the library says so".

**The ten corrections themselves are not here.** They are one customer's rules; §12.3 ships them
as an example overlay. Seeding them into the product is what this project has refused since
§5.1, and it is also what the review means by "frozen at whatever was true the day they were
written" — the point is the loop, so they can grow.
"""
from __future__ import annotations

import re
from typing import Optional

import actions
import learning

# Read from `learning`, never re-declared. A constant beside the one it mirrors is the copy
# that drifts, and "one learning mechanism" has to be checkable rather than aspirational.
GRADUATION_CAMPAIGNS = learning.GRADUATION_CAMPAIGNS
GRADUATION_MARKETS = learning.GRADUATION_MARKETS
RETIREMENT_AFTER = learning.RETIREMENT_AFTER

_TRAILING = ".,;:!?\"'“”‘’ \t"


def _normalise(text: str) -> str:
    """What makes two mentions AUTOMATICALLY the same correction.

    Case, punctuation and whitespace only — a deliberately narrow test, because merging two
    client rules that are not the same puts a rule nobody agreed to on the checklist with
    somebody else's provenance attached.

    Narrow is not the whole answer, though, and the first version stopped here and was
    inverted as a result. Three markets independently saying "seed a single colourway",
    "seeding boxes should carry one colourway" and "only one colourway per box" made THREE
    corrections, each stuck at one campaign in one market — so the gate was satisfiable only
    by the identical string arriving three times, which in practice means one person
    copy-pasting: exactly the single-source case the two-market rule exists to reject. The
    mechanism was easy to satisfy illegitimately and impossible to satisfy legitimately.

    So `_looks_like` suggests and a person answers, which is what §5.1 and §8.2 actually
    settled — *suggest, never auto-merge*. Keeping only the refusal half was a misreading of
    both.
    """
    # Punctuation becomes a SPACE, not nothing. Deleting it made "Post 3-4 times per week"
    # and "Post 34 times per week" the same rule, and "Seed 1-2 colourways" the same as "Seed
    # 12 colourways" — an automatic merge of two genuinely different instructions, which is
    # the one thing this function is supposed to be too narrow to do.
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", (text or "").lower())).strip()


# Measured on different material from §8.2's 0.75, and with a different instrument. A metric
# key is two or three tokens where one character matters; a rule is a sentence two people will
# phrase differently on purpose. Character similarity — difflib, which §8.2 uses — measures
# grammar and word order, which is exactly what differs between two statements of one rule and
# exactly what does not matter.
#
# So: content words, stemmed, compared by how much of the SHORTER rule the longer one covers.
# Jaccard was the first attempt and scored the review's own example at 0.125, because a terse
# restatement of a long rule shares few words with it in total while sharing most of its own.
_LOOKS_LIKE_CUTOFF = 0.55
# Below this many content words there is not enough to compare, and a two-word rule matches
# half the library on one word.
_MIN_CONTENT = 3
_STOPWORDS = frozenset((
    "a an the this that these those and or but if then so as of to in on at for with from by "
    "is are was were be been being do does did should must can could would will shall may "
    "we you they it its our your their please need needs needed want wants all any each per "
    "every more most").split())
# NOT stopwords, and the reason is the whole point. With `not`/`no`/`never` dropped, "Do not
# use AI imagery" and "Use AI imagery only with approval" scored 1.0 — the suggester was
# inverted on precisely the words that invert a rule, and offering "same rule as" there merges
# a prohibition with its permission.
_NEGATIONS = frozenset("not no never dont doesnt cant cannot without avoid neither nor".split())
# Small numbers written as words. "a single colourway" and "one colourway" are the same
# instruction, and nothing else in this comparison would ever see that.
_NUMBER_WORDS = {"one": "1", "single": "1", "sole": "1", "two": "2", "double": "2",
                 "three": "3", "four": "4", "five": "5", "six": "6"}
_SUFFIXES = ("ings", "ing", "ies", "ied", "ers", "er", "ed", "es", "s", "ly")


def _stem(word: str) -> str:
    """Crude and deliberately so. "seed"/"seeding" and "carry"/"carries" are one word here.

    Not a real stemmer: this decides which rule to SUGGEST, and a person answers. Over-matching
    costs one declined suggestion; §5.1's rule is that a wrong suggestion is worse than none,
    and a suggestion nobody is forced to take is the version of "wrong" that costs least.
    """
    word = _NUMBER_WORDS.get(word, word)
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[:-len(suffix)]
    return word


def _content(text: str) -> set:
    return {_stem(w) for w in _normalise(text).split() if w and w not in _STOPWORDS}


def negated(text: str) -> bool:
    """Whether this rule is phrased as a prohibition.

    Crude — one negation anywhere flips it — and deliberately used only to REFUSE a
    suggestion, never to make one. A missed suggestion costs an unmerged rule somebody can
    still merge by hand; a suggested merge of "never use AI imagery" with "use AI imagery"
    puts the opposite of a client's rule on the checklist under their provenance.
    """
    return bool(_NEGATIONS & {_normalise(w) for w in _normalise(text).split()})


def _resembles(a: set, b: set) -> float:
    """How much of the shorter rule the longer one covers."""
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def _reading(text: str) -> Optional[tuple]:
    """What comparing two rules needs from one of them, or None when it is too short to say.

    One place, because two things compare corrections now — `_looks_like` for the single
    write, and `feedback._rule_questions` for the whole table — and the rule that a
    prohibition and its permission are NOT the same rule lived only in the first. The second
    copied it, which is the shape this codebase has been bitten by five times: two
    implementations that agree until one of them is edited.
    """
    content = _content(text)
    return (content, negated(text)) if len(content) >= _MIN_CONTENT else None


def resembles(a: Optional[tuple], b: Optional[tuple]) -> float:
    """How alike two readings are, 0.0 when they cannot be compared at all."""
    if a is None or b is None:
        return 0.0
    # A prohibition and its permission are not the same rule, however alike they read —
    # "never seed a single colourway" and "always seed a single colourway" share every word.
    if a[1] != b[1]:
        return 0.0
    return _resembles(a[0], b[0])


def is_close(score: float) -> bool:
    return score >= _LOOKS_LIKE_CUTOFF


def _looks_like(conn, text: str, *, exclude: Optional[str] = None) -> Optional[dict]:
    """The correction on file this one most resembles, or None (§8.6, mirroring §8.2).

    None when nothing is close, because §5.1 settled that a wrong suggestion is worse than
    none — and none is the honest answer far more often here than for a metric key, since two
    rules about seeding boxes may be two genuinely different rules about seeding boxes.
    """
    import store

    mine = _reading(text)
    best, score = None, 0.0
    for row in store.corrections(conn):
        if row["id"] == exclude or row["status"] in ("merged", "ignored"):
            continue
        overlap = resembles(mine, _reading(row["text"]))
        if overlap > score:
            best, score = row, overlap
    return best if is_close(score) else None


# Every other model-authored field in this codebase is measured. A 400 KB rule and a 200 KB
# provenance were both accepted and stored, and they travel into the evidence package and then
# into saved findings.
_MAX_TEXT = 400
_MAX_PROVENANCE_CHARS = 300


def _bounded(value: str, field: str, limit: int) -> str:
    value = (value or "").strip()
    if len(value) > limit:
        raise ValueError(
            f"{field} is {len(value)} characters; the limit is {limit}. A standing correction "
            f"is one rule somebody can act on — if this is several, record them separately so "
            f"each can be counted, confirmed and cited on its own.")
    return value


DECISIONS = ("same_rule", "different_rule", "set_aside")


def resolve(conn, correction_id: str, *, decision: str,
            same_as: Optional[str] = None) -> dict:
    """Answer the one question §8.6 asks about a new correction. §8.2's three answers, exactly.

    `same_rule` folds it in RETROSPECTIVELY — the sightings already recorded belong to the
    rule it turned out to be, and an answer that fixes the vocabulary while leaving the
    evidence behind has fixed nothing. `different_rule` keeps it and stops asking; it stays
    provisional, because becoming a standing rule is the gate's business. `set_aside` stops it
    being asked about or applied, and keeps what was said — a dismissive click must not
    destroy somebody's recorded feedback.
    """
    import store

    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {list(DECISIONS)}, got {decision!r}")
    entry = describe(conn, correction_id)
    if not entry:
        raise ValueError(f"{correction_id!r} is not a correction on file")
    if entry["status"] == "expected":
        raise ValueError(
            f"{correction_id!r} is already a standing correction in "
            f"{', '.join(entry['expected_in']) or 'no market'}"
            + (f", confirmed by {entry['confirmed_by']}" if entry["confirmed_by"] else "")
            + ". Answering the new-rule question about it would quietly undo that. Use "
              "set_aside_correction if it should no longer be applied.")
    if decision == "same_rule":
        if not same_as:
            raise ValueError(
                "`same_as` is required with 'same_rule': saying two rules are one means "
                "naming which one they are. Use 'different_rule' if it stands on its own.")
        target = describe(conn, same_as)
        if not target:
            raise ValueError(f"{same_as!r} is not a correction on file")
        if target["id"] == correction_id:
            raise ValueError("A correction cannot be the same rule as itself.")
        # Onto the LIVE head of the target's chain, never onto a row that was itself folded
        # away: C merged into an already-merged B put C's sightings on a row nothing reads.
        target = store.live_correction(conn, same_as) or target
        if target["id"] == correction_id:
            # A→B then B→A. Both rows end `merged`, there is no live row left, and the rule
            # disappears from every reader — the sightings intact and unreachable.
            raise ValueError(
                f"{same_as!r} was already folded into {correction_id!r}, so merging the other "
                f"way would leave neither of them as the rule. They are already one.")
        if target["status"] == "ignored":
            raise ValueError(
                f"{same_as!r} was set aside, so folding this into it would put what was said "
                f"somewhere nothing reads. Use 'different_rule', or reopen that one first.")
        store.merge_correction(conn, absorbed=correction_id, into=target["id"])
        return {"correction_id": target["id"], "status": "merged", "absorbed": correction_id,
                "what_it_means": (
                    "Folded in, with everywhere it was said. The rule now counts every market "
                    "that stated it, which is what the gate reads.")}
    if decision == "set_aside":
        store.set_aside_correction(conn, correction_id)
        return {"correction_id": correction_id, "status": "ignored", "answered": True,
                "what_it_means": ("It will not be asked about or applied. What was said is "
                                  "kept — this is a decision about the rule, not the record.")}
    store.mark_correction_asked(conn, correction_id)
    return {"correction_id": correction_id, "status": entry["status"], "answered": True,
            "what_it_means": ("Kept on its own. Whether briefs are judged against it is a "
                              "separate question, asked once it recurs across markets.")}


def _public(row: Optional[dict]) -> Optional[dict]:
    """`correction_id` alongside the row's own `id`.

    One name for this thing everywhere it is read — the tools take `correction_id`, `note`
    returns `correction_id`, and a reader having to know that `find` says `id` instead is the
    small drift that turns into a wrong argument at a call site.
    """
    return {**row, "correction_id": row["id"]} if row else None


def find(conn, text: str) -> Optional[dict]:
    import store
    return _public(store.correction_by_text(conn, _normalise(text)))


def describe(conn, correction_id: str) -> Optional[dict]:
    import store
    return _public(store.get_correction(conn, correction_id))


def sightings(conn, correction_id: str) -> list:
    import store
    return store.correction_sightings(conn, correction_id)


def all_of_them(conn) -> list:
    import store
    return store.corrections(conn)


def note(conn, *, text: str, campaign_id: Optional[str], provenance: str) -> dict:
    """Record that somebody said this. Provisional on first mention, counted after (§8.6)."""
    import store

    if not (text or "").strip():
        raise ValueError("A standing correction needs the rule itself, as somebody said it.")
    if not (provenance or "").strip():
        raise ValueError(
            "`provenance` is required: a correction has to say where it came from — the deck "
            "and slide, or who asked for it — or a judgment citing it can say only “the "
            "library says so”, which is the unfounded confident claim this product is "
            "built against.")
    text = _bounded(text, "text", _MAX_TEXT)
    provenance = _bounded(provenance, "provenance", _MAX_PROVENANCE_CHARS)
    # Before anything is written. The sighting carries a foreign key to `campaigns` and the
    # correction row is committed first, so an id that does not exist raised IntegrityError —
    # not a ValueError, so the reason was discarded and the caller saw "Error executing tool"
    # — and left a correction with no sightings and no provenance behind: an opinion in a text
    # field, which is the one thing this module says it refuses. The model supplying the id is
    # now the ordinary path, because `after_upload` prefills it.
    if campaign_id and not store.get_campaign(conn, campaign_id):
        raise ValueError(
            f"campaign_id {campaign_id!r} is not a record in this library. Pass the id of the "
            f"brief the feedback came from, or omit it — though a correction with no campaign "
            f"counts towards nothing, because the gate is about which campaigns and markets "
            f"raised a rule.")

    normalised = _normalise(text)
    existing = store.correction_by_text(conn, normalised)
    fresh = existing is None
    correction_id = (existing or {}).get("id") or store.insert_correction(
        conn, text=text.strip(), normalised=normalised)
    # `said_as` is what THIS mention actually said. After a merge the canonical row carries
    # one wording and the sightings carry the others, which is the `canonical`/`raw_key` split
    # §8.1 made for measures — and it is what makes the fold checkable afterwards rather than
    # a claim nobody can audit.
    store.note_correction_sighting(conn, correction_id=correction_id, campaign_id=campaign_id,
                                   provenance=provenance, said_as=text)
    store.touch_correction(conn, correction_id, campaign_id=campaign_id)
    # No revival path, deliberately. §8.5 has one because a measure is demoted AUTOMATICALLY —
    # nothing decided it, so a sighting can undo it. A correction only ever leaves the
    # checklist because a person set it aside, and quietly putting it back because the client
    # mentioned it again would overrule them without asking, which is §8.2's lesson exactly.

    entry = describe(conn, correction_id)
    gate = graduation(conn, correction_id)
    result = {"correction_id": correction_id, "text": entry["text"],
              "status": entry["status"],
              "times_seen": entry["times_seen"],
              # CAMPAIGNS, beside the mention count and not instead of it. `times_seen` was
              # the only number on this surface, and it is not the number the gate reads —
              # five mentions with no campaign showed "times_seen: 5" while the gate said
              # "seen in 0 campaigns, needs 3". §8.3 wrote a paragraph on exactly this
              # distinction and the return shape then showed the wrong half of it.
              "campaigns": gate["campaigns"]}
    if campaign_id is None:
        result["what_it_means"] = (
            "Recorded, but not attached to a campaign — so it counts towards nothing. The "
            "gate is about which campaigns and markets raised a rule, and a mention with no "
            "campaign has neither. Pass `campaign_id` to have it count.")
    else:
        # §10.6/D116. `correction_status` answers the question this write creates — how far
        # off standing is it, and what is missing — and it was named nowhere but its own
        # definition. That is §8.6's own defect one stage on: `note_correction` was made
        # reachable and what it writes stayed unreadable, so the gate a rule has to pass was
        # a number nobody could see. `do`, not `ask`: it reads and writes nothing.
        import actions
        result["next_actions"] = [actions.action(
            "Show how far this rule is from standing",
            "correction_status",
            why=f"It has been raised in {gate['campaigns']} campaign(s). Whether that is "
                f"enough, and what is missing if it is not, is a count nothing else reports.",
            consent="do", correction_id=correction_id)]
    if fresh:
        # Never rejected — the client said it, whether or not the library was ready. Never
        # silently accepted either: a rule that arrives on one deck and is never questioned is
        # how one person's preference becomes everybody's requirement.
        similar = _looks_like(conn, text, exclude=correction_id)
        result["new_correction"] = {
            "correction_id": correction_id,
            "text": entry["text"],
            "provenance": provenance.strip(),
            "looks_like": ({"correction_id": similar["id"], "text": similar["text"]}
                           if similar else None),
            "what_it_means": (
                "Recorded as a provisional correction. It is not applied to any brief yet — "
                "that takes it recurring across markets and somebody confirming it."
                + (f" It closely resembles one already on file: “{similar['text']}”. If they "
                   f"are the same rule, say so — a rule stated three ways in three markets "
                   f"counts as three separate rules until somebody folds them together, and "
                   f"then none of them ever recurs."
                   if similar else "")),
            "next_actions": _offers(correction_id, entry["text"], similar),
        }
    # §8.6: asked, never demoted — see `gone_quiet` for why silence means the opposite thing
    # for a rule than it does for a measure.
    quiet = gone_quiet(conn)
    if quiet:
        result["gone_quiet"] = quiet
    newly = _newly_eligible(conn, correction_id)
    if newly:
        result["newly_eligible"] = newly
    return result


def _offers(correction_id: str, text: str, similar: Optional[dict]) -> list:
    """§8.2's three answers, in this item's vocabulary. Offered, never chosen for the user:
    which of two wordings is the rule is a judgment about their vocabulary, and getting it
    wrong merges two rules that are not the same."""
    import actions

    offers = []
    if similar:
        offers.append(actions.action(
            f"Same rule as “{similar['text'][:50]}”", "resolve_correction",
            why="Folded together, both markets count towards the same rule. Left apart, each "
                "wording is its own rule and neither ever recurs — so a rule three markets "
                "actually agree on never becomes standing.",
            consent="ask", correction_id=correction_id, decision="same_rule",
            same_as=similar["id"]))
    offers.append(actions.action(
        "It is its own rule", "resolve_correction",
        why="It stays on file and stops asking. Whether briefs are judged against it is a "
            "separate question, asked once it recurs across markets.",
        consent="ask", correction_id=correction_id, decision="different_rule"))
    offers.append(actions.action(
        "Stop asking about this one", "resolve_correction",
        why="What was said is kept — this is a decision about the rule, not about the record.",
        consent="ask", correction_id=correction_id, decision="set_aside"))
    return actions.trim(offers)


def graduation(conn, correction_id: str) -> dict:
    """Whether this correction has earned a place. `learning`'s gate, not a second one."""
    import store

    entry = describe(conn, correction_id)
    if not entry:
        raise ValueError(f"{correction_id!r} is not a correction on file")
    gate = {**learning.gate(
        name=entry["text"], noun="rule",
        campaigns=learning.distinct_briefs(conn, store.correction_campaigns(conn,
                                                                            correction_id)),
        markets=entry["markets"], status=entry["status"],
        expected_in=entry["expected_in"], confirmed_by=entry["confirmed_by"]),
        "correction_id": correction_id}
    if gate["eligible"]:
        import replay
        gate["if_confirmed"] = replay.if_graduated(conn, correction_id=correction_id,
                                                   markets=gate["seen_in"])
    return gate


def graduate(conn, correction_id: str, *, confirmed_by: str) -> dict:
    """Promote a correction to standing. Requires the gate AND a person (§8.3's gate, §8.6's
    subject)."""
    import store

    who = learning.require_a_person(confirmed_by)
    gate = graduation(conn, correction_id)
    if not gate["eligible"]:
        raise ValueError(gate["what_it_means"])
    store.graduate_correction(conn, correction_id, markets=gate["seen_in"], confirmed_by=who)
    entry = describe(conn, correction_id)
    return {**entry, "graduated": True,
            "next_actions": actions.after_graduation(what=entry["text"],
                                                     markets=entry["expected_in"]),
            "what_it_means": (
                f"Briefs in {', '.join(entry['expected_in'])} are now judged against this, on "
                f"{who}'s confirmation. It is shown with the judgment as a standing correction "
                f"with its provenance, so a finding can cite where the rule came from.")}


def _newly_eligible(conn, correction_id: str) -> Optional[dict]:
    """The graduation question, asked once, on the write that made it askable.

    §8.3 shipped without this and the gate was unreachable: nothing surfaced eligibility, so
    the only route was a tool a user would have to know exists and think to call. The same
    mistake here would leave the ten corrections exactly as frozen as the review says they are.
    """
    import actions
    import store

    if store.correction_was_offered(conn, correction_id):
        return None
    gate = graduation(conn, correction_id)
    if not gate["eligible"]:
        return None
    store.mark_correction_offered(conn, correction_id)
    return {
        "correction_id": correction_id,
        "text": gate["name"],
        "campaigns": gate["campaigns"],
        "seen_in": gate["seen_in"],
        "if_confirmed": gate.get("if_confirmed"),
        "what_it_means": (
            f"This has now come up in {gate['campaigns']} campaigns across "
            f"{', '.join(gate['seen_in'])}. It can become a standing correction every brief in "
            f"those markets is judged against — which is a rule, so somebody has to say so "
            f"rather than the library deciding on its own."),
        # `confirmed_by` deliberately not prefilled: filled in with a name nobody gave, the
        # offer would manufacture the confirmation the gate exists to require (§6.5).
        "next_actions": actions.trim([actions.action(
            f"Make “{gate['name'][:60]}” a standing correction",
            "graduate_correction",
            why="Briefs in those markets will be judged against it, with its provenance "
                "attached so a finding can cite where it came from. It stops being applied if "
                "it falls out of use, and nothing is deleted.",
            consent="ask", needs=["confirmed_by — who is confirming this; ask, do not assume"],
            correction_id=correction_id)]),
    }


# Every other model-facing field in this codebase is bounded. A rule stated in twenty briefs
# carries twenty provenances, and joining all of them puts a paragraph into the evidence
# package and then into the stored judgment.
_MAX_PROVENANCE = 4


def _provenance_of(conn, correction_id: str) -> str:
    """Where a rule was learned, bounded.

    Every origin matters — "praised in Peru; instructed independently in Australia" is what
    makes a rule more than one client's house style, and a judgment that can cite only one of
    them has lost the stronger half. But "every" and "unbounded" are not the same thing.
    """
    import store

    where = [s["provenance"] for s in store.correction_sightings(conn, correction_id)]
    shown = "; ".join(where[:_MAX_PROVENANCE])
    extra = len(where) - _MAX_PROVENANCE
    return shown + (f" (and {extra} more)" if extra > 0 else "")


def standing_for(conn, campaign_id: str, *, markets: Optional[list] = None) -> dict:
    """The corrections a brief in this market is judged against (§8.6).

    Provisional ones are excluded. One that nobody has confirmed is exactly what the gate
    exists to withhold, and putting it in front of a judgment as a standing rule would promote
    it by the back door.
    """
    import store

    if campaign_id:
        markets, refusal = learning.subject_markets(conn, campaign_id)
        if refusal:
            return {**refusal, "standing": []}
    else:
        # §8.6: a proposal that is not a record yet still has a market — the CALLER named it,
        # and "judge this new pitch" is the flow the review cares about most. Returning
        # nothing there left the client's own standing rules out of the one judgment they
        # most obviously apply to. Measures can defend the record-only reading (a loose block
        # of text has no market); a caller who passed `market="LATAM"` has told us otherwise.
        markets = [m for m in (markets or []) if m]
        if not markets:
            return {**learning.unchecked("no_market", (
                "No market was given for this proposal, and standing corrections belong to "
                "one — so none were applied. This is not a pass: pass `market`, or store the "
                "brief and pass `campaign_id`.")), "standing": []}

    wanted = {store.fold_market(m) for m in markets}
    standing = []
    for entry in store.corrections(conn):
        if entry["status"] != "expected":
            continue
        if not wanted & {store.fold_market(m) for m in entry["expected_in"]}:
            continue
        standing.append({
            "correction_id": entry["id"],
            "text": entry["text"],
            "provenance": _provenance_of(conn, entry["id"]),
            "confirmed_by": entry["confirmed_by"],
            "seen_in": entry["expected_in"],
            "times_seen": entry["times_seen"],
        })
    return {
        "market": ", ".join(markets),
        "markets": markets,
        "standing": standing,
        "basis": "computed",
        "code": "checked" if standing else "none_standing",
        "status": "checked" if standing else "nothing_to_check",
        "what_it_means": (
            f"{len(standing)} standing correction{'s' * (len(standing) != 1)} apply to briefs "
            f"in {', '.join(markets)}. Each carries the provenance it was learned from — cite "
            f"that when a finding rests on one."
            if standing else
            "Nothing has become a standing correction for this market yet. A rule joins once "
            "it has recurred across markets and a person has confirmed it."),
    }


def set_aside(conn, correction_id: str, *, why: Optional[str] = None) -> dict:
    """Stop applying a standing correction (§8.6).

    The inverse `graduate` had none. `learning.gate` has carried a `set_aside` branch since
    §8.3 and nothing could reach it for a correction, so one promoted in error was a blocking
    finding on every brief in its markets with no way back short of waiting for it to fall out
    of use — and a `guardrail_breach` cannot be softened to a note.
    """
    import store

    entry = describe(conn, correction_id)
    if not entry:
        raise ValueError(f"{correction_id!r} is not a correction on file")
    import actions

    store.set_aside_correction(conn, correction_id)
    return {"correction_id": correction_id, "text": entry["text"], "status": "ignored",
            "why": why,
            "what_it_means": (
                "No brief is judged against this any more, and it will not be asked about "
                "again. What was said and where it was said are still on file, so judgments "
                "that already cited it stay explicable."),
            # §10.6/D116: the way back, named at the one moment it becomes worth knowing.
            # `reopen_correction` exists precisely because a rule can be set aside in error,
            # and it was reachable only by somebody who already knew it existed — so the
            # escape hatch for a mistake was itself behind a thing you had to not make a
            # mistake about. It is `ask`, not `do`: putting a rule back changes every future
            # judgment in its markets, which is what this call just decided not to do.
            "next_actions": [actions.action(
                "Put this rule back if that was not what you meant",
                "reopen_correction",
                why="Setting a rule aside is the only way it leaves the checklist, and "
                    "nothing puts it back on its own — silence is read as compliance here, "
                    "not as disuse.",
                consent="ask", correction_id=correction_id)]}


def gone_quiet(conn) -> list:
    """Standing rules nobody has repeated lately — ASKED about, never demoted (§8.6).

    This is the one place the loop deliberately differs from §8.5's, and the reason is that
    the same signal means opposite things.

    For a MEASURE, absence is evidence of disuse: nobody is tracking it any more, and demoting
    it is right. For a RULE, absence of repetition is evidence of COMPLIANCE — a client stops
    restating "seed a single colourway" exactly when the agency has started doing it. Retiring
    on silence therefore demotes precisely the rules that are working, and keeps the ones that
    are being ignored, which is backwards and no threshold fixes it.

    So the question is surfaced and a person answers, exactly as every other promotion and
    demotion in this loop is answered. A rule leaving the checklist changes every future
    judgment, and that is not a thing to do unattended — which was doubly true when the
    demotion was reported only to whoever happened to write next.
    """
    import actions
    import store

    quiet = []
    for entry in store.corrections(conn):
        if entry["status"] != "expected":
            continue
        skipped = store.campaigns_that_skipped_correction(
            conn, entry["id"], since=entry["last_seen"], markets=entry["expected_in"])
        if skipped < RETIREMENT_AFTER or store.correction_quiet_asked(conn, entry["id"]):
            continue
        store.mark_correction_quiet_asked(conn, entry["id"])
        quiet.append({
            "correction_id": entry["id"],
            "text": entry["text"],
            "campaigns_without_it": skipped,
            "what_it_means": (
                f"This has not come up in the last {skipped} campaigns in "
                f"{', '.join(entry['expected_in']) or 'its markets'}. That may mean it is "
                f"being followed, or that it stopped mattering — the library cannot tell "
                f"which, so it is still being applied until somebody says otherwise."),
            "next_actions": actions.trim([
                actions.action(
                    f"Stop applying “{entry['text'][:50]}”", "set_aside_correction",
                    why="Briefs will no longer be judged against it. Nothing is deleted and "
                        "judgments that already cited it stay explicable.",
                    consent="ask", correction_id=entry["id"]),
                actions.action(
                    f"Keep applying “{entry['text'][:50]}”", "keep_correction",
                    why="It stays standing, and the library stops asking about this one.",
                    consent="ask", correction_id=entry["id"]),
            ]),
        })
    return sorted(quiet, key=lambda r: r["correction_id"])


def reopen(conn, correction_id: str) -> dict:
    """Undo a set-aside (§8.6).

    Back to `provisional`, never straight to standing: whether briefs are judged against a rule
    is the gate's question and a person's confirmation, and jumping over both would let a
    reopen do what a graduation is for. §8.2's `different_measure` reopens an ignored measure
    the same way, and the asymmetry — a correction with no way back at all — was the drift.
    """
    import store

    entry = describe(conn, correction_id)
    if not entry:
        raise ValueError(f"{correction_id!r} is not a correction on file")
    if entry["status"] != "ignored":
        raise ValueError(f"{correction_id!r} is {entry['status']}, not set aside.")
    store.reopen_correction(conn, correction_id)
    return {"correction_id": correction_id, "text": entry["text"], "status": "provisional",
            "what_it_means": (
                "Back on file and countable again. It is not applied to any brief — that "
                "still takes it recurring across markets and somebody confirming it.")}


def keep(conn, correction_id: str) -> dict:
    """"Still current" — stop asking about this one (§8.6)."""
    import store

    entry = describe(conn, correction_id)
    if not entry:
        raise ValueError(f"{correction_id!r} is not a correction on file")
    store.mark_correction_quiet_asked(conn, correction_id)
    return {"correction_id": correction_id, "text": entry["text"], "status": entry["status"],
            "what_it_means": "Still applied, and the library will not ask about it again."}
