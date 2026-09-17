"""
§13.2 — computed facts stored beside the record instead of recomputed on every call.

D95: *"`gaps()` recomputes every campaign's facts on every call (500 decks ≈ 40 s, dominated
by the channel regexes) and does it before the empty-library early return. Facts are
deterministic on body text, so they belong beside the record, written at ingest and on edit."*

Measured before anything was changed, on 200 decks of 38,000 characters each — the shape the
review's 500 real decks had: **2.41 s per `gaps()` call, of which `facts.compute` is 2.3 s.**
The diagnosis is right and the cost is real.

**The row's premise is not.** "Deterministic on body text" is false: `facts.compute` reads the
customer's rulebook twice — `rulebook.vocabulary("channels")` decides the channel checklist
(`facts.py:399`) and `rulebook.rules()` supplies the guardrail `watch_for` words
(`facts.py:612`). Verified on one unchanged string: with no overlay it computes
`channels: partial` and `guardrails: nothing_to_check`; with a rulebook declared it computes
`channels: present` and `guardrails: contradicted`.

So a cache keyed on body text alone would, the moment a customer wrote their rulebook, go on
reporting `guardrails: nothing_to_check` for every record already stored — the product saying
"there are no rules to check" about a library whose rules had just been written, with
`nothing_to_check` doing the work of a pass. That is this codebase's cardinal sin, and it is
what caching the row as specified would have built.

The key is therefore (body text, rulebook version). `rulebook.version()` is the composite
stamp — `core-1.0+acme-3` — which already changes whenever either half does, and §12.1 built
it for exactly this kind of question.
"""
import time

import core
import facts
import pytest
import rulebook
import store


@pytest.fixture()
def declaring():
    """A customer rulebook in force for one test, cleared afterwards."""
    def write(yaml):
        path = rulebook.overlay_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml, encoding="utf-8")
        rulebook.load.cache_clear()

    yield write
    rulebook.load.cache_clear()


_A_RULEBOOK = (
    "version: 'acme-3'\n"
    "rules:\n"
    "  - id: no-ai-imagery\n"
    "    rule: Never use AI-generated imagery in client-facing creative.\n"
    "    severity: blocking\n"
    "    why: Two clients have asked in writing.\n"
    "    watch_for: ['midjourney']\n"
)

_BODY = ("Seeding brief for the Bogota opening. Concept boards were made in midjourney. "
         "Budget: USD 145,000. Engagement rate 3.2%. Runs 2026-03-01 to 2026-04-15.")


def _counting(monkeypatch):
    """How many times `facts.compute` actually runs."""
    calls = []
    real = facts.compute
    monkeypatch.setattr(facts, "compute",
                        lambda text: (calls.append(1), real(text))[1])
    return calls


# ---------------------------------------------------------------------------------------
# The cost D95 names


def test_a_stored_records_facts_are_not_recomputed_on_every_call(conn, monkeypatch):
    """The whole of D95. Reading the library must not re-scan every deck's text."""
    for n in range(4):
        core.ingest_campaign(conn, title=f"Peru launch {n}", market="Peru",
                             status="concluded", deck_text=_BODY, confirm=True)

    calls = _counting(monkeypatch)
    core.gaps(conn)

    assert not calls, (
        f"{len(calls)} full text scans to read a library nothing had changed"
    )


def test_the_facts_are_written_when_the_record_arrives(conn):
    """Beside the record, at ingest — the row's own prescription, and the reason the read
    path can stop scanning."""
    cid = core.ingest_campaign(conn, title="Peru launch", market="Peru", status="concluded",
                               deck_text=_BODY, confirm=True)["campaign_id"]

    stored = store.stored_facts(conn, cid)

    assert stored, "nothing was written beside the record"
    assert stored["budget"]["status"] == "present"
    # And it is the WHOLE answer, not a summary of it — the read path serves this dict, so
    # anything missing here is a fact the library would stop reporting.
    assert stored == facts.compute(_BODY)
    assert store.stored_facts_are_current(conn, cid, _BODY)


def test_the_cached_answer_is_the_computed_one(conn):
    """A cache that answers differently from the thing it caches is not a cache. This is the
    only assertion that makes the others worth anything."""
    cid = core.ingest_campaign(conn, title="Peru launch", market="Peru", status="concluded",
                               deck_text=_BODY, confirm=True)["campaign_id"]

    assert facts.for_campaign(conn, cid) == facts.compute(_BODY)


# ---------------------------------------------------------------------------------------
# Invalidation: the half the row's premise gets wrong


def test_editing_the_body_recomputes_the_facts(conn):
    """The half D95 does name. A brief whose budget was removed must not keep reporting one."""
    cid = core.ingest_campaign(conn, title="Peru launch", market="Peru", status="concluded",
                               deck_text=_BODY, confirm=True)["campaign_id"]
    assert facts.for_campaign(conn, cid)["budget"]["status"] == "present"

    core.update_campaign(conn, campaign_id=cid, deck_text=None,
                         detail="No numbers in this version at all.")
    # `deck_text` is not editable through `update_campaign` by design, so the body that
    # changed is `detail`. Either way the stored facts must follow it.
    conn.execute("UPDATE campaigns SET deck_text = ? WHERE id = ?",
                 ("No numbers in this version at all.", cid))
    conn.commit()
    core._rebuild_body_index(conn, cid)

    body = "\n\n".join(store.text_on_file(conn, cid)["body"])
    assert facts.for_campaign(conn, cid) == facts.compute(body)
    assert facts.for_campaign(conn, cid)["budget"]["status"] != "present", (
        "the brief no longer states a budget and the library still reports one"
    )


def test_a_body_changed_by_a_path_that_forgot_still_reads_correctly(conn):
    """Why the key carries a hash of the body and not just a flag somebody sets.

    A cache the write paths have to remember to refresh goes stale the first time somebody
    adds a seventh write path — this codebase's most repeated defect, and not something to
    leave to my having found all six. So the key is checked on every read: a body that does
    not match what the row was computed under is a MISS, however it came to differ."""
    cid = core.ingest_campaign(conn, title="Peru launch", market="Peru", status="concluded",
                               deck_text=_BODY, confirm=True)["campaign_id"]
    assert facts.for_campaign(conn, cid)["budget"]["status"] == "present"

    # Straight into the row, the way a write path that never learned about the cache would.
    conn.execute("UPDATE campaigns SET deck_text = ? WHERE id = ?",
                 ("A brief with no numbers in it at all.", cid))
    conn.commit()

    assert facts.for_campaign(conn, cid)["budget"]["status"] != "present", (
        "the stored facts describe a version of this brief that no longer exists"
    )
    body = "\n\n".join(store.text_on_file(conn, cid)["body"])
    assert facts.for_campaign(conn, cid) == facts.compute(body)


def test_rebuilding_the_body_leaves_the_facts_warm(conn, monkeypatch):
    """The read path repairs a stale cache by recomputing, so correctness never depends on a
    write path remembering. What the write path buys is that nobody PAYS for that scan while
    waiting for a report — the whole point of the item. Editing a brief is the moment the
    text is already in hand."""
    cid = core.ingest_campaign(conn, title="Peru launch", market="Peru", status="concluded",
                               deck_text=_BODY, confirm=True)["campaign_id"]
    conn.execute("UPDATE campaigns SET deck_text = ? WHERE id = ?",
                 ("A brief with no numbers in it at all.", cid))
    conn.commit()
    core._rebuild_body_index(conn, cid)

    calls = _counting(monkeypatch)
    read = facts.for_campaign(conn, cid)

    assert not calls, "the rebuild left the facts cold, so the next reader pays for the scan"
    assert read["budget"]["status"] != "present"


def test_declaring_a_rulebook_does_not_leave_stale_facts_behind(conn, declaring):
    """**The half the row's premise misses, and the reason a body-text key would be wrong.**

    `facts.compute` reads the rulebook twice — the channel checklist and the guardrail
    `watch_for` words — so one unchanged string computes different facts before and after a
    customer writes their rules. Cached on the text alone, every record stored before the
    rulebook existed would go on reporting `guardrails: nothing_to_check`: the product saying
    there are no rules to check about a library whose rules were just written, with
    `nothing_to_check` doing the work of a pass."""
    cid = core.ingest_campaign(conn, title="Peru launch", market="Peru", status="concluded",
                               deck_text=_BODY, confirm=True)["campaign_id"]
    assert facts.for_campaign(conn, cid)["guardrails"]["status"] == "nothing_to_check"

    declaring(_A_RULEBOOK)

    fresh = facts.for_campaign(conn, cid)
    assert fresh["guardrails"]["status"] != "nothing_to_check", (
        "the library reports no rules to check on a record whose rules now exist"
    )
    assert fresh == facts.compute(_BODY), "the cached answer is not the computed one"


def test_editing_a_rule_without_bumping_the_version_still_expires_the_cache(conn, declaring):
    """**The hole the first version of this cache had, found in review.**

    Nothing makes a customer bump `version:` when they edit a rule — the README asks, and
    that is all. Keyed on `rulebook.version()`, a rule edited in place kept its key, so the
    old answer was served forever. Before this cache every read recomputed, so a restart
    healed it; the cache is what made "it heals on restart" stop being true, and it owes the
    guard.

    The direction matters as much as the fact. Here the customer DELETES the rule: the
    library went on reporting `contradicted` — asserting a guardrail breach against a rule
    that no longer exists. Words ADDED without a bump fail the mirror way, reporting
    `checked_clean` about text that now breaches something."""
    declaring(_A_RULEBOOK)
    cid = core.ingest_campaign(conn, title="Peru launch", market="Peru", status="concluded",
                               deck_text=_BODY, confirm=True)["campaign_id"]
    assert facts.for_campaign(conn, cid)["guardrails"]["status"] == "contradicted"

    # The SAME version string. Only the rule's words changed.
    declaring(_A_RULEBOOK.replace("midjourney", "dall-e"))
    assert rulebook.version() == "core-1.0+acme-3", "the version did not move, by design"

    assert facts.for_campaign(conn, cid)["guardrails"]["status"] == "checked_clean", (
        "the library asserts a breach of a rule the customer has deleted"
    )
    assert facts.for_campaign(conn, cid) == facts.compute(_BODY)


def test_adding_a_watch_word_without_a_bump_is_noticed_too(conn, declaring):
    """The mirror, and the more dangerous direction: a rule the customer has just written
    reported as `checked_clean` against text that breaches it."""
    declaring(_A_RULEBOOK.replace("midjourney", "dall-e"))
    cid = core.ingest_campaign(conn, title="Peru launch", market="Peru", status="concluded",
                               deck_text=_BODY, confirm=True)["campaign_id"]
    assert facts.for_campaign(conn, cid)["guardrails"]["status"] == "checked_clean"

    declaring(_A_RULEBOOK)

    assert facts.for_campaign(conn, cid)["guardrails"]["status"] == "contradicted"


def test_a_change_to_the_checks_themselves_expires_the_cache(conn, monkeypatch):
    """`rulebook.version()`'s `core-1.0` half is the bundled YAML's version, NOT a code
    version — `facts.py` has changed six times while that string stood still. So a release
    that fixes a matcher would have left every stored row answering with the old one, on
    every install, with no way to notice.

    Before the cache an upgrade healed everything by recomputing. The cache is what removed
    that, so the key carries a stamp for the algorithm as well as its inputs."""
    cid = core.ingest_campaign(conn, title="Peru launch", market="Peru", status="concluded",
                               deck_text=_BODY, confirm=True)["campaign_id"]
    assert facts.for_campaign(conn, cid)["budget"]["status"] == "present"

    # A release that changes how a fact is computed, standing in for an edit to facts.py.
    monkeypatch.setattr(store, "_facts_algorithm_stamp", lambda: "a-later-release")

    assert not store.stored_facts_are_current(conn, cid, _BODY), (
        "a row computed by the old checks is served as though it were current"
    )


def test_a_row_that_is_not_a_fact_set_costs_a_scan_and_never_an_answer(conn):
    """The docstring promised that a corrupt row costs a scan and never an answer, and
    checking only that the JSON parsed made it false. `[]` and `"text"` parse and then fail
    in the caller with an `AttributeError`; `{}` parses, is truthy, and is returned as a
    record with NO facts at all — which is worse, because nothing raises and the record
    silently has nothing to say about itself."""
    cid = core.ingest_campaign(conn, title="Peru launch", market="Peru", status="concluded",
                               deck_text=_BODY, confirm=True)["campaign_id"]
    key = store.facts_key(_BODY)

    for junk in ("[]", '"text"', "{}", '{"budget": {}}', "null"):
        conn.execute("UPDATE campaigns SET computed_facts = ?, computed_facts_key = ? "
                     "WHERE id = ?", (junk, key, cid))
        conn.commit()

        answer = facts.for_campaign(conn, cid)

        assert answer["budget"]["status"] == "present", f"{junk} was served as a fact set"
        assert answer == facts.compute(_BODY)


# ---------------------------------------------------------------------------------------
# A READ that writes — the three hazards that buys, none of which had a test


def test_filling_the_cache_does_not_commit_the_callers_transaction(conn):
    """`facts.for_campaign` is a READ, and on a miss it writes. An unconditional `commit`
    there ends a transaction the caller opened and meant to control — so work they were
    about to roll back is made permanent by the act of reading. `store`'s metrics import
    documents this exact shape ("a rollback after it had nothing left to undo"); review
    demonstrated it here."""
    cid = core.ingest_campaign(conn, title="Peru launch", market="Peru", status="concluded",
                               deck_text=_BODY, confirm=True)["campaign_id"]
    conn.execute("UPDATE campaigns SET computed_facts_key = NULL WHERE id = ?", (cid,))
    conn.commit()

    conn.execute("BEGIN")
    conn.execute("UPDATE campaigns SET title = ? WHERE id = ?", ("NOT COMMITTED", cid))
    facts.for_campaign(conn, cid)          # cold: fills the cache
    conn.rollback()

    assert store.get_campaign(conn, cid)["title"] == "Peru launch", (
        "reading the library committed a write the caller rolled back"
    )


def test_reading_a_read_only_library_still_works(conn, tmp_path):
    """Before this cache `gaps()` was a pure read. A cold record on a read-only database —
    a replica, a backup being inspected, a file somebody chmod'd — raised
    `attempt to write a readonly database`. A cache that cannot be filled is a slow read,
    never a failed one."""
    import sqlite3

    cid = core.ingest_campaign(conn, title="Peru launch", market="Peru", status="concluded",
                               deck_text=_BODY, confirm=True)["campaign_id"]
    conn.execute("UPDATE campaigns SET computed_facts_key = NULL WHERE id = ?", (cid,))
    conn.commit()
    path = conn.execute("PRAGMA database_list").fetchone()[2]

    read_only = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    read_only.row_factory = sqlite3.Row
    try:
        answer = facts.for_campaign(read_only, cid)
    finally:
        read_only.close()

    assert answer["budget"]["status"] == "present"


def test_filling_the_cache_does_not_wait_on_another_writer(conn, tmp_path):
    """A rulebook edit makes EVERY record cold at once, so the first report afterwards tries
    to fill the whole cache — and with another connection mid-write each one sat out the full
    busy timeout inside a read somebody was waiting on. Review measured 5.49 s for a single
    record. Filling a cache is never worth blocking on."""
    import sqlite3
    import time as _time

    cid = core.ingest_campaign(conn, title="Peru launch", market="Peru", status="concluded",
                               deck_text=_BODY, confirm=True)["campaign_id"]
    conn.execute("UPDATE campaigns SET computed_facts_key = NULL WHERE id = ?", (cid,))
    conn.commit()
    path = conn.execute("PRAGMA database_list").fetchone()[2]

    blocker = sqlite3.connect(path)
    blocker.execute("BEGIN IMMEDIATE")
    blocker.execute("UPDATE campaigns SET region = 'held' WHERE id = ?", (cid,))
    try:
        started = _time.perf_counter()
        answer = facts.for_campaign(conn, cid)
        waited = _time.perf_counter() - started
    finally:
        blocker.rollback()
        blocker.close()

    assert answer["budget"]["status"] == "present", "the read failed rather than skipping"
    assert waited < 1.0, f"a read waited {waited:.2f}s for a cache it could have skipped"


def test_a_measured_metrics_words_are_part_of_the_body_and_rewarm_it(conn, monkeypatch):
    """`text_on_file` puts a MEASURED metric's `detail` in the body — in writing, because
    "CTR was 3.2 percent, well above the benchmark" is this product's most common real
    citation. So recording one changes what the checks read. The read path would recompute
    correctly either way; warming here is what stops a bulk import leaving the whole library
    cold for the next report."""
    cid = core.ingest_campaign(conn, title="Peru launch", market="Peru", status="concluded",
                               deck_text="A store opening in Bogota.", confirm=True)["campaign_id"]
    assert facts.for_campaign(conn, cid)["budget"]["status"] != "present"

    core.add_metrics(conn, campaign_id=cid, metric_type="actual", confirm=True,
                     detail="Spend came in at USD 145,000 against the plan.")

    calls = _counting(monkeypatch)
    after = facts.for_campaign(conn, cid)

    assert not calls, "recording a metric left the record cold"
    assert after["budget"]["status"] == "present", (
        "the metric's own words are in the body and the facts did not follow them"
    )


def test_an_operator_can_see_how_much_of_the_cache_is_warm(conn):
    """Under the staleness this cache could produce, the rulebook stamp was the only thing
    that would let anybody notice — and it was on disk, reachable from no surface at all.
    `health_check` is where "is this install answering from rows computed under an older
    rulebook" belongs, and a cold row is reported rather than failed, because it is a slower
    read and never a wrong answer."""
    for n in range(3):
        core.ingest_campaign(conn, title=f"Peru launch {n}", market="Peru",
                             status="concluded", deck_text=_BODY, confirm=True)

    said = core.health_check(conn, probe=False)["components"]["computed_facts"]

    assert said["ok"] and said["warm"] == 3 and said["records"] == 3
    assert said["checked_under"] == rulebook.version()


# ---------------------------------------------------------------------------------------
# And it has to be worth having


def test_reading_a_large_library_does_not_re_scan_every_deck(conn, monkeypatch):
    """The measurement that made this item real: 200 decks of 38,000 characters cost 2.41 s
    per `gaps()` call, essentially all of it inside `facts.compute`. The point is not the
    second saved on a small library; it is that the cost grows with the library on every
    single read, and a product whose reporting surface slows down as it is used is one people
    stop reading."""
    body = "\n\n".join("Bogota launch paid social influencer OOH reach impressions. " * 40
                       for _ in range(10))
    for n in range(12):
        core.ingest_campaign(conn, title=f"Peru launch {n}", market=f"M{n % 3}",
                             status="concluded", deck_text=body, confirm=True)

    calls = _counting(monkeypatch)
    started = time.perf_counter()
    core.gaps(conn)
    core.gaps(conn)
    elapsed = time.perf_counter() - started

    assert not calls
    assert elapsed < 1.0, f"two reads of a 12-deck library took {elapsed:.2f}s"
