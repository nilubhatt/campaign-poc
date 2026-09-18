"""
§13.5 — the remaining two-copy helpers, and which of them are actually two copies.

D114: *"'One mechanism' is true of the decisions and not yet of the code around them."* The row
names five pairs. Read side by side, they are not the same kind of thing, and the row says so
itself — *"merging them needs a row abstraction over both that is a larger change than either
caller"* — so this item is as much about judging which is which as about deleting code.

**Byte-identical, and therefore where the next drift comes from:**

  `touch_metric` / `touch_correction`  — the market-fold loop is the same characters in both,
      down to `seen.add(fold_market(where))`. It is the fifth implementation of "which markets
      does this campaign count towards", which is the question D55 and D88 both record drifting.
  `campaigns_that_skipped` / `campaigns_that_skipped_correction` — the market-scoping SQL
      clause and its parameter packing are the same in both, differing only in a table alias.
  `embedding.each_string_embedded_once` / `core._each_record_read_once` — added by §13.3, both
      wrong in the same way (module globals under a threaded server), both fixed the same way.
      §13.3 named this as debt owed here rather than building a third copy.

**Shape-similar over different tables, and left alone deliberately:** `_newly_eligible`,
`graduate`, and the offered-once flags. What DECIDES in those is already shared — `learning.gate`,
`distinct_briefs`, `subject_markets`, `require_a_person`, the thresholds — and what remains is a
table name, a key column and an offer whose fields are different words for a reader. Merging
them needs a per-kind descriptor that is larger than either caller and would put a layer between
two callers and the SQL they each run once. That is a cost with no drift to prevent: the parts
that drifted were the parts that decided, and those were unified before this row was written.

So what this file pins is: the identical halves have one implementation, the behaviour they
carried is unchanged, and the parts that were left alone were left alone because their deciding
half is shared — not because nobody looked.
"""
import ast
import pathlib
import re

import core
import corrections
import metrics
import store

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _body(path, name):
    src = (ROOT / path).read_text()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return "".join(lines[node.lineno - 1:node.end_lineno])
    raise AssertionError(f"{name} not found in {path}")


def _campaign(conn, title, **kw):
    return core.ingest_campaign(conn, title=title, detail="A store opening.",
                                confirm=True, **kw)["campaign_id"]


# ---------------------------------------------------------------------------------------
# The identical halves


def test_which_markets_a_campaign_counts_towards_is_written_once():
    """The fold loop stood character-for-character in `touch_metric` and `touch_correction`.
    It is the fifth implementation of a question D55 and D88 both record drifting, and two
    copies of it is how the sixth arrives."""
    both = [_body("store.py", "touch_metric"), _body("store.py", "touch_correction")]

    for body in both:
        assert "_widen_markets" in body, "it does not go through the shared helper"

    # The merge that re-derives a correction's breadth is a THIRD caller the row does not
    # name, found by mutating the shared helper and watching a test that should have felt it
    # stay green. It had no test of its own, which is why nothing was watching it.
    assert "_widen_markets" in _body("store.py", "merge_correction")

    # And the DEDUPE inside the shared helper is `learning.fold_markets` — "one name per
    # market, first spelling kept", which is that function's own sentence. The first version
    # of `_widen_markets` open-coded it, making the extraction a fresh copy of the thing it
    # was extracting. A scan across the modules that answer this question is what catches the
    # next one.
    import ast as _ast

    open_coded = []
    for module in ("store.py", "metrics.py", "corrections.py", "replay.py", "core.py",
                   "feedback.py"):
        source = (ROOT / module).read_text()
        for node in _ast.walk(_ast.parse(source)):
            if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                continue
            body = _body(module, node.name)
            # The dedupe EXPRESSION, wherever it is spelled: a folded market tested for
            # membership of a set of folded markets. Matching on the word "seen" instead
            # caught comment prose and `times_seen`, which is a scan nobody can act on.
            if re.search(r"fold_market\([^)]*\)\s+not in\s*[{(]", body) \
                    or re.search(r"fold_market\(where\)\s+not in\s+seen", body):
                open_coded.append(f"{module}:{node.name}")

    assert not open_coded, (
        f"these open-code the market dedupe rather than folding through "
        f"`learning.fold_markets`: {open_coded}"
    )


def test_the_market_scoping_clause_is_written_once():
    """`campaigns_that_skipped` and `campaigns_that_skipped_correction` built the same SQL
    fragment and packed the same parameters, differing only in a table alias. A change to how
    a market is matched — and §12.2 changed exactly that — has to land in both or the two
    answers diverge silently."""
    both = [_body("store.py", "campaigns_that_skipped"),
            _body("store.py", "campaigns_that_skipped_correction")]

    for body in both:
        assert 'LOWER(c.markets) LIKE ?' not in body, (
            "the market-scoping clause is still spelled out here"
        )
        assert "_market_scope" in body, "it does not go through the shared helper"


def test_one_call_one_set_of_answers_is_one_mechanism():
    """§13.3 wrote this twice — a `ContextVar`, a context manager installing a dict only if
    none is installed, and a pass-through reader — and named it as debt owed here. Both copies
    had the same non-locality bug under a threaded server and both needed the same fix, which
    is that duplication made concrete."""
    import embedding
    import scoping

    # Neither is a function any more — both are `scoping.scoped_memo(...)`, which is the
    # point. A source scan would have kept passing on two `def`s that happened to share a
    # helper; what makes them ONE mechanism is that the context manager, the re-entrancy rule
    # and the reader all come from one place.
    for module, name in ((embedding, "each_string_embedded_once"),
                         (core, "_each_record_read_once")):
        scope = getattr(module, name)
        assert hasattr(scope, "remembering") and hasattr(scope, "is_open"), (
            f"{module.__name__}.{name} is not a `scoping.scoped_memo`"
        )

    # And they are separate scopes, not one shared dict — a record read must not be served
    # from the embedding memo or the other way round.
    with embedding.each_string_embedded_once():
        assert embedding.each_string_embedded_once.is_open()
        assert not core._each_record_read_once.is_open()


def test_the_shared_scope_is_re_entrant(conn):
    """`save_evaluation` opens both scopes and reaches `prepare_evaluation`, which opens one
    again. A nested block installing a FRESH dict would throw away what the outer one had
    already paid for — the memo quietly doing nothing on exactly the path that opens it
    twice."""
    import scoping

    scope = scoping.scoped_memo("a test scope")
    done = []

    with scope():
        scope.remembering("k", lambda: done.append(1))
        with scope():
            scope.remembering("k", lambda: done.append(1))
        scope.remembering("k", lambda: done.append(1))

    assert len(done) == 1, "a nested scope discarded what the outer one had remembered"


def test_outside_a_scope_the_work_is_simply_done(conn):
    """What makes it safe to reach from code that does not know whether a scope is open —
    and what stops it becoming a process-lifetime cache, which is how the first version of
    the embedding memo hid an embedder that had gone down between two calls."""
    import scoping

    scope = scoping.scoped_memo("another test scope")
    done = []

    scope.remembering("k", lambda: done.append(1))
    scope.remembering("k", lambda: done.append(1))

    assert len(done) == 2
    assert not scope.is_open()


# ---------------------------------------------------------------------------------------
# And the behaviour they carried is unchanged


def test_a_measure_still_counts_every_market_a_campaign_ran_in(conn):
    """What the fold loop is FOR. A campaign in MX and CO counts towards both, and reading
    `market or region` alone counted it as zero — the drift the loop was written to stop."""
    cid = _campaign(conn, "Bogota launch", markets=["Mexico", "Colombia"], status="concluded")
    metrics.record(conn, campaign_id=cid, key="roas", value=3.4, metric_type="actual")

    seen = metrics.describe(conn, "roas")["markets"]

    assert {m.lower() for m in seen} == {"mexico", "colombia"}


def test_a_correction_still_counts_every_market_a_campaign_ran_in(conn):
    """The same, through the other caller — which is the point of there being one loop."""
    cid = _campaign(conn, "Bogota launch", markets=["Mexico", "Colombia"], status="concluded")
    noted = corrections.note(conn, text="Creator captions name the product first.",
                             campaign_id=cid, provenance="Client call")

    seen = corrections.describe(conn, noted["new_correction"]["correction_id"])["markets"]

    assert {m.lower() for m in seen} == {"mexico", "colombia"}


def test_merging_two_rules_keeps_every_market_between_them(conn):
    """The third copy of the fold loop, which had no test of its own — which is why it was
    found by mutating the shared helper rather than by reading. Folding two rules together
    re-derives the breadth from the sightings that just moved, and §8.3's gate reads that
    number: losing a market here makes a rule look narrower than the evidence for it."""
    peru = _campaign(conn, "Peru launch", market="Peru", status="concluded")
    mexico = _campaign(conn, "Mexico launch", market="Mexico", status="concluded")
    first = corrections.note(conn, text="Creator captions name the product first.",
                             campaign_id=peru,
                             provenance="Client call")["new_correction"]["correction_id"]
    second = corrections.note(conn, text="Captions must lead with the product name always.",
                              campaign_id=mexico,
                              provenance="Client call")["new_correction"]["correction_id"]

    corrections.resolve(conn, correction_id=second, decision="same_rule", same_as=first)

    kept = corrections.describe(conn, first)["markets"]
    assert {m.lower() for m in kept} == {"peru", "mexico"}, (
        f"a market was lost when two rules were folded together: {kept}"
    )


def test_a_market_is_folded_once_however_it_was_typed(conn):
    """"LATAM" and "latam" are one market, and the first spelling seen is the one kept because
    it is what somebody typed."""
    first = _campaign(conn, "Bogota launch", market="LATAM", status="concluded")
    second = _campaign(conn, "Lima launch", market="latam", status="concluded")
    for cid in (first, second):
        metrics.record(conn, campaign_id=cid, key="roas", value=3.4, metric_type="actual")

    seen = metrics.describe(conn, "roas")["markets"]

    assert seen == ["LATAM"], seen


def test_the_market_scope_still_narrows_a_staleness_count(conn):
    """What the SQL clause is FOR: ten APAC campaigns must not retire a measure expected in
    LATAM. A market that never used it cannot be evidence that it fell out of use."""
    import time as _time

    peru = _campaign(conn, "Peru launch", market="Peru", status="concluded")
    metrics.record(conn, campaign_id=peru, key="roas", value=3.4, metric_type="actual")
    since = _time.time()
    for n in range(4):
        elsewhere = _campaign(conn, f"Hanoi launch {n}", market="Vietnam", status="concluded")
        metrics.record(conn, campaign_id=elsewhere, key="reach", value=1000,
                       metric_type="actual")

    # A MULTI-market campaign, which is what the JSON `LIKE` in the shared clause is for: an
    # exact match on `market` would miss a record that ran in Peru among others, and the count
    # would then say a measure had fallen out of use in a market that was still using it.
    both = _campaign(conn, "Andes launch", markets=["Peru", "Colombia"], status="concluded")
    metrics.record(conn, campaign_id=both, key="reach", value=900, metric_type="actual")

    assert store.campaigns_that_skipped(conn, "roas", since=since, markets=["Colombia"]) == 1, (
        "a campaign whose market is in its `markets` list was not counted"
    )
    assert store.campaigns_that_skipped(conn, "roas", since=since, markets=["Peru"]) == 1, (
        "a multi-market campaign that skipped it was not counted for Peru"
    )
    assert store.campaigns_that_skipped(conn, "roas", since=since, markets=["Vietnam"]) == 4, (
        "campaigns in another market were not counted for that market"
    )


def test_a_record_with_no_market_reaches_no_rule(conn):
    """`learning.reaches`'s falsy case, which is a decision and not a guard. A rule that
    graduated on LATAM does not apply to a record that says nowhere it ran, and answering
    otherwise would put every unmarked record on every rule's list the first time anything
    graduated anywhere."""
    import learning

    assert learning.reaches(["LATAM", "Peru"], "peru")
    assert not learning.reaches(["LATAM", "Peru"], "Vietnam")
    for nothing in (None, "", "   "):
        assert not learning.reaches(["LATAM", "Peru"], nothing), (
            f"a record whose market is {nothing!r} was said to reach a LATAM rule"
        )


def test_a_records_own_markets_are_deduped_however_they_were_typed(conn):
    """`markets_of` answers "every market this record counts towards", and it folds — a
    record whose `market` is "Peru" and whose `markets` list also says "peru" counts once,
    because §8.3's gate is "seen in at least two" and reading one market as two satisfies a
    gate whose entire purpose is that breadth is earned."""
    cid = _campaign(conn, "Andes launch", market="Peru", markets=["peru", " PERU ", "Chile"])

    counts = store.markets_of(store.get_campaign(conn, cid))

    folded = [store.fold_market(m) for m in counts]
    assert len(folded) == len(set(folded)), f"one market counted more than once: {counts}"
    assert set(folded) == {"peru", "chile"}


def test_the_market_scope_refuses_an_alias_that_is_not_one(conn):
    """`_market_scope` interpolates its alias into SQL — the first function in `store` to put
    a table alias in a query. Both callers pass a literal; the assert is what keeps them the
    only two that could."""
    import pytest

    assert store._market_scope(["Peru"], "v")[0].startswith(" AND EXISTS")

    with pytest.raises(AssertionError):
        store._market_scope(["Peru"], "v; DROP TABLE campaigns; --")


# ---------------------------------------------------------------------------------------
# What was left alone, and why


def test_the_deciding_half_of_graduation_is_already_one_implementation():
    """The row's own reasoning, pinned so "left alone" stays a judgment rather than an
    oversight. `_newly_eligible` and `graduate` differ in a table name and the words of an
    offer; what DECIDES is `learning.gate`, and both go through it. That is the half that
    drifted — §12.4 found the gate applying half of its own rule — and it is shared."""
    for module in ("metrics.py", "corrections.py"):
        assert "learning.gate(" in _body(module, "graduation"), (
            f"{module}'s graduation decides eligibility for itself"
        )
        # AND that the two callers reach it. Asserting only that `graduation` calls the gate
        # left `_newly_eligible` free to decide for itself — review replaced its gate check
        # with a local `times_seen < 2` and this file stayed green, which is the exact drift
        # the row describes passing the test written to forbid it.
        for caller in ("_newly_eligible", "graduate"):
            assert "graduation(" in _body(module, caller), (
                f"{module}'s {caller} does not go through the shared gate"
            )


def test_promoting_a_measure_is_audited_like_promoting_a_rule(conn):
    """The asymmetry the "left alone" judgment missed, and the reason it is worth checking a
    judgment rather than asserting one.

    §11.1 puts the account beside the name "on the write that puts a rule in front of every
    future brief in its markets" — and a graduated MEASURE is put in front of every future
    brief in exactly the same way. `corrections.graduate` had recorded it since §11.1;
    `metrics.graduate` never did, so a standing requirement kept only a free-text name. That
    is a decision made one way on one path and the other way on the other, which is not the
    table-shaped difference D114's row reduces them to."""
    peru = _campaign(conn, "Peru launch", market="Peru", status="concluded")
    mexico = _campaign(conn, "Mexico launch", market="Mexico", status="concluded")
    cusco = _campaign(conn, "Cusco launch", market="Peru", status="concluded")
    for cid in (peru, mexico, cusco):
        metrics.record(conn, campaign_id=cid, key="footfall_uplift_pct", value=4.0,
                       metric_type="actual")

    metrics.graduate(conn, "footfall_uplift", confirmed_by="R. Vega")

    said = store.authorship_for(conn, "metric", "footfall_uplift")
    assert said, "nothing recorded who put this measure in front of every future brief"
    assert said["on_behalf_of"]["name"] == "R. Vega"
    assert said["captured_by"]["method"], "the account the call was made from is not recorded"
