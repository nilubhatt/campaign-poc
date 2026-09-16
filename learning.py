"""
§8.6: one learning mechanism, for measures and for standing corrections.

The review: *"New client feedback is the same shape of event as a new metric. 'We want the
seeding box to carry one colourway' should travel the identical path: provisional on first
mention, counted, promoted to the checklist once it recurs across markets and a person confirms
it, with provenance attached. One learning mechanism for both keeps the rulebook coherent."*

**"One learning mechanism" is the requirement, and a second one shaped like the first does not
satisfy it.** Two implementations agree on the day they are written and drift by the next item.
This codebase has now found that five separate times — D55 and D88 on "which markets is this
campaign in", and twice inside §8.3 alone, where the market question had drifted into a fifth
copy that read `market or region` and reported a campaign running in MX and CO as covering
none. So the loop lives here, once, and both callers pass their counts into it.

What is genuinely different stays with each caller: a measure is a key with numeric values and
a correction is a sentence with provenance. What is the same is the whole of the loop —
provisional on first sight, counted by CAMPAIGN, gated on breadth as well as count, confirmed
by a person, and never deleted once recorded.

The one place the two deliberately part company is what happens when something stops coming
up. For a measure, absence is disuse and §8.5 demotes it. For a rule, absence of repetition is
usually COMPLIANCE — a client stops restating a rule when the agency starts following it — so
§8.6 asks instead of demoting. Same gate, different signal, and `set_aside` is the status that
branch has always had.

The three thresholds live here for the same reason. A constant re-declared beside the one it
mirrors is the copy that drifts, and `metrics` and `corrections` both read these.
"""
from __future__ import annotations

from typing import Optional

# "Seen in N campaigns, across at least two partners or markets, and confirmed once by a
# person." Three conditions, all required. The second is the one doing the work: *"count alone
# is not enough — one partner's house metric should never quietly become a standing requirement
# for everyone."* Fifteen sightings in one market is a habit, not a standard.
GRADUATION_CAMPAIGNS = 3
GRADUATION_MARKETS = 2
# §8.5: how many campaigns may report their own learnings without this one appearing before it
# stops being asked for. Campaigns, not months — a library nobody has opened for six months has
# not retired anything.
RETIREMENT_AFTER = 10

# Where a learned thing sits in the vocabulary, which is NOT the same question as which
# checklists it is on. Collapsing the two is what would have put twelve shipped measures on
# every checklist the day the product is installed (§8.3).
STATUSES = ("provisional", "known", "expected", "retired", "ignored")


def distinct_briefs(conn, campaign_ids) -> int:
    """How many separate briefs these records represent.

    Three versions of one brief are one brief. `supersedes` already says so, and counting the
    records let v1, v2 and v3 of a single partner's deck satisfy a gate that means "three
    different campaigns carried this" — the write-counting mistake one level up.
    """
    import store

    roots = set()
    for cid in campaign_ids:
        chain = store.supersession_chain(conn, cid)
        roots.add(chain[0] if chain else cid)
    return len(roots)


def fold_markets(raw_markets) -> list:
    """One name per market, first spelling kept.

    "SEA", "sea" and "Sea" are one market, and reading them as three satisfied a gate whose
    entire purpose is "seen in at least two". C16 established the fold after exactly this bug;
    §8.3 reintroduced it in the place it does the most damage.
    """
    import store

    out, seen = [], set()
    for raw in raw_markets or []:
        key = store.fold_market(raw)
        if key and key not in seen:
            seen.add(key)
            out.append(raw.strip())
    return out


def gate(*, name: str, noun: str, campaigns: int, markets: list, status: str,
         expected_in=(), confirmed_by: Optional[str] = None,
         from_rulebook: Optional[str] = None) -> dict:
    """Whether a learned thing has earned a place on the checklist, and what is missing if not.

    Status is asked BEFORE the counts, and the order matters. Asked the other way round, a
    thing somebody had set aside read "seen in 1 campaign, needs 3" — telling the user to keep
    recording something that will be refused forever, and re-asking a question they had
    declined, which §8.2 named as how a product teaches people to dismiss it.
    """
    markets = fold_markets(markets)
    base = {"name": name, "campaigns": campaigns, "markets": len(markets),
            "seen_in": markets, "status": status, "confirmed_by": confirmed_by}

    # D108: a rule that came out of the CUSTOMER'S OWN RULEBOOK is confirmed by definition —
    # it is in a file they wrote. The ten standing corrections §12.3 ships have provenance (a
    # deck and a slide) and no `campaign_id` in this library, so the counting gate refused
    # them forever: seen in 0 campaigns, needs 3. The item built to give them "somewhere to
    # live and grow" gave them somewhere they could be stored and never applied.
    #
    # This is a different question from the one the gate asks, not an exemption from it. The
    # gate exists to stop the LIBRARY promoting a rule it INFERRED from one partner's house
    # style — breadth has to be earned because the library is guessing. A customer writing a
    # rule down is not guessing, and nothing about three campaigns in two markets makes their
    # own rule truer.
    #
    # What it does NOT skip is `require_a_person`: a rule arriving from a file is not somebody
    # confirming it, and §8.2 spent this loop's only human step on "who says so".
    if from_rulebook and status not in ("expected", "ignored"):
        return {**base, "eligible": True, "code": "declared_by_the_customer",
                # WHY it skipped the counting, on the row. Without this the audit trail reads
                # identically to a rule that met three campaigns in two markets, and nobody
                # later can tell a customer's declaration from the library's inference.
                "from_rulebook": from_rulebook,
                "basis": "stated",
                "what_it_means": (
                    f"{name} comes from your own rulebook ({from_rulebook}), so it does not "
                    f"have to be seen in three campaigns first — that test is for a rule this "
                    f"library INFERRED, where breadth is what makes the guess safe. It still "
                    f"needs a person's name against promoting it.")}

    if status == "expected":
        # Not a failure, and "not ready" would send the caller off to collect data it does not
        # need. Refusing rather than re-running is also what protects `confirmed_by`: a second
        # confirmation overwrote the name of the person who actually gave the first, which is
        # the one audit field this whole gate exists to create.
        return {**base, "eligible": False, "code": "already_expected",
                "what_it_means": (
                    f"{name} is already expected of briefs in "
                    f"{', '.join(expected_in) or 'no market'}"
                    + (f", confirmed by {confirmed_by}." if confirmed_by else "."))}
    if status == "ignored":
        return {**base, "eligible": False, "code": "set_aside",
                "what_it_means": (f"{name} was set aside, so it will not be asked for. "
                                  f"Recording more of it will not change that.")}

    missing = []
    if campaigns < GRADUATION_CAMPAIGNS:
        missing.append(f"seen in {campaigns} campaign{'s' * (campaigns != 1)}, "
                       f"needs {GRADUATION_CAMPAIGNS}")
    if len(markets) < GRADUATION_MARKETS:
        missing.append(f"seen in {len(markets)} market{'s' * (len(markets) != 1)} "
                       f"({', '.join(markets) or 'none recorded'}), "
                       f"needs {GRADUATION_MARKETS} — one partner's house {noun} should not "
                       f"become a standing requirement for everyone")
    if missing:
        return {**base, "eligible": False, "code": "not_yet",
                "what_it_means": f"{name} is not ready to be expected of a brief: "
                                 + "; ".join(missing) + "."}
    return {**base, "eligible": True, "code": "eligible",
            "what_it_means": (f"{name} can be added to the checklist for "
                              f"{', '.join(markets)}, once a person confirms it.")}


def require_a_person(confirmed_by: Optional[str]) -> str:
    """The gate's third condition, which is deliberately not automatable.

    A promotion with nobody's name against it is a standing requirement nobody can question
    later, and §8.2 spent the loop's only human step on exactly this question.

    §11.1/§11.2: `identity.person` is the rule, not a fourth copy of it. This one checked only
    that the string was non-empty, so `confirmed_by="the system"` promoted a rule to the
    checklist — the library confirming its own suggestion and recording that a person had,
    on the one path D105 names as the sharpest. Four implementations of one rule and the
    weakest of them deciding what the library believes about who decided things.
    """
    import identity

    return identity.person(confirmed_by, field="confirmed_by")


# A checklist is for a brief that is going to run. A reference record is brand guidelines and a
# stub is a placeholder; telling either it is missing something is the check firing on
# everything — §7.1/D11 was careful about which TEXT a check reads, and this is the same care
# about which RECORD it runs against.
CHECKABLE_RECORDS = ("campaign", None)


def unchecked(code: str, what: str, **extra) -> dict:
    """The shape §7.1 settled on, so a reader can tell an unchecked result from a clean one.

    `status: nothing_to_check` is not a pass. Every one of these returns an empty list, and an
    empty list beside a missing `status` reads as "this brief carries everything expected of
    it" — a clean bill of health the server has no basis for.
    """
    return {"market": None, "markets": [], "basis": "computed", "code": code,
            "status": "nothing_to_check", "what_it_means": what, **extra}


def subject_markets(conn, campaign_id: str):
    """The markets a checklist should be read for, or a reason there are none.

    Returns `(markets, refusal)` — exactly one of them is meaningful. Both callers need the
    same three answers (not a campaign / no market / here they are), and writing that twice is
    how the reference record came to be told it was missing footfall uplift in one place and
    not the other.
    """
    import store

    record = store.get_campaign(conn, campaign_id) or {}
    if not record:
        return None, unchecked("no_such_record",
                               "No record with that id, so there is nothing to check.")
    if record.get("record_type") not in CHECKABLE_RECORDS:
        return None, unchecked("not_a_campaign",
                               f"This is a {record.get('record_type')} record, not a campaign "
                               f"brief, so the checklist does not apply to it.")
    named = [m for m in store.markets_of(record) if m]
    if not named:
        return None, unchecked("no_market", (
            "This record names no market, and a checklist belongs to one — so there is nothing "
            "to check it against. This is not a pass: add a market to see what briefs like it "
            "usually carry."))
    return named, None
