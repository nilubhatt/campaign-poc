"""
D52 — ranking the gaps by what they cost, not by what kind they are.

    "Gap ranking by judgments affected rather than a fixed rank per kind — one unindexed
     record currently outranks forty."

`_GAP_RANK` is a constant per code, so the order of two gaps is decided entirely by which
KIND they are and never by how much of the library each one is about. `partly_indexed` sits
at rank 4 and `execution_never_checked` at 5, so a library with one half-embedded deck and
forty concluded campaigns whose execution nobody ever checked is told, first and in bold, to
finish indexing one record.

The ranking is `gaps()`'s entire product — its own docstring says so: *"Ranked, lowest number
first, and the ranking is the design. An unordered list of everything absent is the thing
nobody reads."* A ranking that cannot see magnitude is an unordered list wearing numbers.

**What this does NOT do.** The tracker row says "by judgments affected", and that is not
available: there is not enough evaluation traffic in any real library to rank by it, and a
ranking that needs a hundred saved judgments before it works is one that is wrong for every
new customer — which is every customer, on the day the ranking matters most.

What IS available is how many RECORDS each gap is about, which is the thing the review's own
example is counting ("one unindexed record" against "forty"). So the kind stays the primary
key — a market with nothing measured really is a worse hole than a record that has not been
indexed, at equal size — and magnitude breaks ties and, past a threshold, overrides one step
of kind. Kind-only and size-only are both wrong; the fix is that size is allowed to speak.
"""
import core


def _campaign(conn, title, **kw):
    kw.setdefault("market", "LATAM")
    kw.setdefault("status", "concluded")
    kw.setdefault("detail", f"A campaign called {title}, which ran in a market.")
    return core.ingest_campaign(conn, title=title, **kw)["campaign_id"]


def _codes(conn):
    return [g["code"] for g in core.gaps(conn)["gaps"]]


def _judged_and_measured(conn, title):
    """A campaign with results on file and a judgment nobody ever checked against them —
    `judgments_never_reconciled`, which the fixed table ranks BELOW `few_verified_outcomes`.
    """
    cid = _campaign(conn, title)
    core.save_evaluation(conn, subject_title=title, campaign_id=cid, verdict="approve",
                         summary="Looks like the others, which worked.", findings=[],
                         predictions={"predicted_roi_range": "3.0 to 4.0"})
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)
    return cid


def test_a_big_gap_outranks_a_small_one_of_a_worse_kind(conn):
    """The review's own example, in the shape this library can actually build: twelve
    judgments nobody has ever checked against the results sitting beside them, and one
    finished campaign with no results at all.

    The fixed table puts `few_verified_outcomes` at 2 and `judgments_never_reconciled` at 3,
    so the product leads with the one record and buries the twelve — which is exactly "one
    unindexed record outranks forty" with different nouns."""
    for n in range(12):
        _judged_and_measured(conn, f"Judged {n}")
    _campaign(conn, "The one nobody measured")

    codes = _codes(conn)

    assert codes.index("judgments_never_reconciled") < codes.index("few_verified_outcomes"), (
        f"one unmeasured campaign still outranks twelve unchecked judgments: {codes}"
    )


def test_kind_still_decides_between_gaps_of_similar_size(conn):
    """Size does not become the only key. At comparable size a campaign with NO results is a
    worse hole than a judgment nobody has re-read — the first means a citation rests on
    somebody's impression, the second that a judgment is untested — and a pure size ordering
    would flip that as soon as one more record was affected."""
    for n in range(3):
        _judged_and_measured(conn, f"Judged {n}")
    for n in range(3):
        _campaign(conn, f"Unmeasured {n}")

    codes = _codes(conn)

    assert codes.index("few_verified_outcomes") < codes.index("judgments_never_reconciled")


def test_the_ranking_is_stable_for_the_same_library(conn):
    """§10.4's rule, which applies here as much as to the menu: two readings of one library
    must not reshuffle. A magnitude key with ties broken by nothing is a ranking that depends
    on dictionary order."""
    for n in range(4):
        _campaign(conn, f"Campaign {n}")

    assert _codes(conn) == _codes(conn)


def test_what_the_ranking_rests_on_is_visible(conn):
    """A number that decides the order and does not appear in the output is one nobody can
    argue with — and the review's whole complaint about this product is answers that cannot
    be argued with."""
    for n in range(4):
        _campaign(conn, f"Campaign {n}")

    gap = core.gaps(conn)["gaps"][0]

    assert gap["affects"] >= 1
    assert gap["rank_basis"] == "computed"


def test_share_is_a_share(conn):
    """`affects` counts RECORDS for every gap, because `share` divides it by the record count.
    `judgments_never_reconciled` counted JUDGMENTS, so one campaign carrying three unchecked
    judgments reported `share: 3.0` — not a share of anything — and, being over the threshold,
    claimed that single record outweighed its kind and moved up a rank.

    A ratio of two different units is a number that cannot be wrong, because it does not mean
    anything; and this one was published beside `rank_basis: computed`."""
    cid = _campaign(conn, "Peru")
    for n in range(3):
        core.save_evaluation(conn, subject_title=f"Peru take {n}", campaign_id=cid,
                             verdict="approve", summary="Fine.", findings=[],
                             predictions={"predicted_roi_range": "3.0 to 4.0"})
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)

    gap = next(g for g in core.gaps(conn)["gaps"]
               if g["code"] == "judgments_never_reconciled")

    assert gap["counts"]["judgments"] == 3
    assert gap["affects"] == 1
    assert gap["share"] <= 1.0


def test_no_gap_claims_a_share_above_one(conn):
    """The general form. Any `affects` that counts something other than records reappears
    here as a share greater than 1, whatever the gap."""
    for n in range(3):
        cid = _campaign(conn, f"Campaign {n}", market=f"Market {n}")
        for take in range(2):
            core.save_evaluation(conn, subject_title=f"C{n} take {take}", campaign_id=cid,
                                 verdict="approve", summary="Fine.", findings=[],
                                 predictions={"predicted_roi_range": "3.0 to 4.0"})
        core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.4}, confirm=True)

    for gap in core.gaps(conn)["gaps"]:
        assert 0.0 <= gap["share"] <= 1.0, (gap["code"], gap["affects"], gap["share"])
