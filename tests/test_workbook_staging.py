"""
§10.5 applied to the workbook: an offer that has to survive a round trip.

10.5's rule is about the feedback menu — *"numbers are not identifiers. If anything changes
between rendering the menu and the user answering, '3' silently means a different campaign"* —
and `menu_token` is the answer. Two tracker rows say the import has the same problem in two
different ways, and the same answer fixes both:

  D117  A workbook is resent in full to confirm it — a 500-row import crosses the wire twice.
        A `preview_id` staging the rows server-side would avoid it.
  D118  An import stopped by the time budget has no resume key, so resending the whole file
        re-imports the prefix. The result says how many rows were processed but not from
        where.

D118 is the sharper of the two, because the product currently makes a promise it cannot keep.
The truncated import returns, in writing:

    "imported 40 rows before the 25s time budget ran out; 460 rows were not processed. Send
     them again to continue — nothing already imported is duplicated by doing so."

Nothing enforces that. `add_metrics` appends; there is no key and no dedup, so a caller who
resends the workbook — the obvious reading of "send them again", and the only thing a client
holding a file can easily do — imports the first forty rows a second time and the library
then holds two of every figure. A sentence promising idempotency on a path that has none is
worse than no sentence: it is the product telling somebody the safe thing to do is the thing
that corrupts their numbers.

So the preview stages the rows and hands back a key. Confirming quotes the key rather than
the workbook, and resuming quotes the key too — and the key remembers how far it got.
"""
import pytest

import core
import store


def _campaign(conn, title):
    return core.ingest_campaign(conn, title=title, market="LATAM", status="concluded",
                                detail=f"A campaign called {title}.")["campaign_id"]


def _rows(conn, n):
    return [{"campaign_id": _campaign(conn, f"Campaign {i}"),
             "structured": {"roas": 1.0 + i}} for i in range(n)]


# ── D117: the workbook crosses the wire once ───────────────────────────────

def test_a_preview_hands_back_a_key(conn):
    """"A `preview_id` staging the rows server-side would avoid it." The preview already
    holds the rows; the only reason to send them again was that nothing kept them."""
    preview = store.bulk_import_metrics(conn, _rows(conn, 3), confirm=False)

    assert preview["preview"] is True
    assert preview["preview_id"]


def test_confirming_quotes_the_key_instead_of_the_workbook(conn):
    """A 500-row import crossing the wire twice is the cost; this is the fix. The rows are
    the ones that were previewed, by construction, rather than by a promise that the second
    send matched the first."""
    rows = _rows(conn, 3)
    preview = store.bulk_import_metrics(conn, rows, confirm=False)

    done = store.bulk_import_metrics(conn, None, confirm=True,
                                     preview_id=preview["preview_id"])

    assert done["imported"] == 3
    assert [len(store.get_campaign(conn, r["campaign_id"])["metrics"]) for r in rows] \
        == [1, 1, 1]


def test_the_preview_offers_the_import_it_is_the_consent_step_for(conn):
    """D116, and the reason this is worth doing at all: "send it again with confirm=True" was
    an instruction in prose, which is the form this project's own principle says models drop.
    With the rows staged the offer is genuinely ONE step — nothing is left for the caller to
    supply, so `needs` is empty and the user really can just say yes."""
    preview = store.bulk_import_metrics(conn, _rows(conn, 3), confirm=False)

    offer = next(a for a in preview["next_actions"] if a["tool"] == "bulk_import_metrics")

    assert offer["prefilled_args"]["preview_id"] == preview["preview_id"]
    assert offer["prefilled_args"]["confirm"] is True
    assert not offer.get("needs"), "the rows are staged, so accepting is one step"


def test_a_key_that_was_never_issued_is_refused(conn):
    """Never guessed at and never silently treated as an empty workbook: "imported 0 rows" for
    a key the server does not hold reads as a successful import of nothing."""
    with pytest.raises(ValueError, match="not a staged workbook"):
        store.bulk_import_metrics(conn, None, confirm=True, preview_id="preview_nonesuch")


def test_a_confirm_with_neither_rows_nor_key_says_which_to_send(conn):
    with pytest.raises(ValueError, match="rows"):
        store.bulk_import_metrics(conn, None, confirm=True)


def test_sending_rows_still_works(conn):
    """The old contract. A customer's script that sends the workbook twice is not broken by
    this — it is simply no longer the only way."""
    rows = _rows(conn, 2)

    done = store.bulk_import_metrics(conn, rows, confirm=True)

    assert done["imported"] == 2


# ── D118: an interrupted import knows where it stopped ─────────────────────

def test_an_interrupted_import_returns_the_key_to_resume_from(conn, monkeypatch):
    """"The result says how many rows were processed but not from where." A count is not a
    resume key: it tells the caller how much is left and nothing about what to send."""
    rows = _rows(conn, 6)
    preview = store.bulk_import_metrics(conn, rows, confirm=False)
    _stop_after(monkeypatch, 2)

    part = store.bulk_import_metrics(conn, None, confirm=True,
                                     preview_id=preview["preview_id"])

    assert part["imported"] == 2 and part["not_processed"] == 4
    assert part["preview_id"] == preview["preview_id"]
    assert any(a["tool"] == "bulk_import_metrics" for a in part["next_actions"])


def test_resuming_imports_only_what_is_left(conn, monkeypatch):
    """The whole of D118. Resuming on the key re-imports nothing, so the note's promise —
    "nothing already imported is duplicated" — is true because something enforces it rather
    than because it is written down."""
    rows = _rows(conn, 6)
    preview = store.bulk_import_metrics(conn, rows, confirm=False)
    _stop_after(monkeypatch, 2)
    store.bulk_import_metrics(conn, None, confirm=True, preview_id=preview["preview_id"])

    monkeypatch.undo()
    rest = store.bulk_import_metrics(conn, None, confirm=True,
                                     preview_id=preview["preview_id"])

    assert rest["imported"] == 4
    assert [len(store.get_campaign(conn, r["campaign_id"])["metrics"]) for r in rows] \
        == [1, 1, 1, 1, 1, 1], "a resumed import duplicated the rows it had already done"


def test_resuming_a_finished_import_imports_nothing_twice(conn):
    """The offer survives in a transcript, and a model re-reading one has no way to know the
    call was already made. Accepting it twice has to be safe, which is the same reason
    `menu_token` refreshes rather than writing."""
    rows = _rows(conn, 3)
    preview = store.bulk_import_metrics(conn, rows, confirm=False)
    store.bulk_import_metrics(conn, None, confirm=True, preview_id=preview["preview_id"])

    again = store.bulk_import_metrics(conn, None, confirm=True,
                                      preview_id=preview["preview_id"])

    assert again["imported"] == 0
    assert again["status"] == "already_imported"
    assert [len(store.get_campaign(conn, r["campaign_id"])["metrics"]) for r in rows] \
        == [1, 1, 1]


def test_the_note_no_longer_tells_the_caller_to_resend_the_file(conn, monkeypatch):
    """The false promise, removed. "Send them again — nothing already imported is duplicated"
    was untrue for the obvious reading of it, and a sentence telling somebody the safe move is
    the one that doubles their numbers is worse than saying nothing."""
    preview = store.bulk_import_metrics(conn, _rows(conn, 6), confirm=False)
    _stop_after(monkeypatch, 2)

    part = store.bulk_import_metrics(conn, None, confirm=True,
                                     preview_id=preview["preview_id"])

    assert "send them again" not in part["note"].lower()
    assert preview["preview_id"] in part["note"]


# ── what staging costs, and what pays it back ──────────────────────────────

def test_finished_batches_do_not_accumulate(conn):
    """A staged workbook is a copy of the customer's file. Ten previews of a 500-row workbook
    is 5,000 rows of JSON kept forever against a feature whose whole purpose is to send the
    file ONCE — the fix for a bandwidth cost, paid for with an unbounded disk cost.

    Finished batches are dropped: there is nothing to resume and re-confirming answers
    `already_imported` from a row that no longer needs the workbook attached to it."""
    for _ in range(store.MAX_STAGED_IMPORTS + 3):
        preview = store.bulk_import_metrics(conn, _rows(conn, 1), confirm=False)
        store.bulk_import_metrics(conn, None, confirm=True, preview_id=preview["preview_id"])

    held = conn.execute("SELECT COUNT(*) AS n FROM import_batches").fetchone()["n"]

    assert held <= store.MAX_STAGED_IMPORTS


def test_an_unfinished_batch_is_not_dropped_for_a_newer_one(conn):
    """The one thing pruning must never do. An import stopped by the time budget is the case
    the resume key exists FOR, and dropping it because somebody previewed something else in
    the meantime loses rows that were half-written — silently, since the offer to resume is
    in a transcript and the key it names has simply gone."""
    rows = _rows(conn, 4)
    unfinished = store.bulk_import_metrics(conn, rows, confirm=False)
    for _ in range(store.MAX_STAGED_IMPORTS + 3):
        done = store.bulk_import_metrics(conn, _rows(conn, 1), confirm=False)
        store.bulk_import_metrics(conn, None, confirm=True, preview_id=done["preview_id"])

    resumed = store.bulk_import_metrics(conn, None, confirm=True,
                                        preview_id=unfinished["preview_id"])

    assert resumed["imported"] == 4


def test_two_sessions_resuming_one_key_do_not_double_import(conn, monkeypatch):
    """The resume was a TOCTOU: the "already imported" guard read `imported_through` at entry
    and `_advance_import` wrote it at exit, with the whole import in between. Two sessions
    holding the same key both read 0, both imported every row, and the library ended with two
    of every figure — the exact corruption D118 exists to remove, arriving through the key
    that removed it.

    A check at entry and a write at exit is not a check; it is a gap with a write on each
    side of it."""
    rows = _rows(conn, 3)
    preview = store.bulk_import_metrics(conn, rows, confirm=False)
    other = store.connect()

    # The window is between reading the batch and opening the transaction: once this session
    # holds `BEGIN`, SQLite's own locking makes the other one wait. `diff_columns` is the
    # call that sits in that gap, so racing there is racing where the defect actually lives —
    # which is also exactly where the reviewer's probe reproduced it.
    import metrics as metrics_module

    real = metrics_module.diff_columns
    fired = {"done": False}

    def racing(*a, **kw):
        if not fired["done"]:
            fired["done"] = True
            store.bulk_import_metrics(other, None, confirm=True,
                                      preview_id=preview["preview_id"])
        return real(*a, **kw)

    monkeypatch.setattr(metrics_module, "diff_columns", racing)

    with pytest.raises(ValueError, match="resumed somewhere else"):
        store.bulk_import_metrics(conn, None, confirm=True,
                                  preview_id=preview["preview_id"])

    monkeypatch.undo()
    other.close()
    assert [len(store.get_campaign(conn, r["campaign_id"])["metrics"]) for r in rows] \
        == [1, 1, 1], "the losing session's rows were a second copy and must not survive"


def test_an_abandoned_preview_does_not_live_forever(conn):
    """An unfinished batch is never pruned by the count rule AND is subtracted from the cap,
    so twenty abandoned previews drove the limit to zero, deleted every finished batch, and
    then grew without bound themselves. Each one is a full copy of a customer's KPI workbook
    with no expiry and no route out through `delete_campaign`'s cascade."""
    stale = store.bulk_import_metrics(conn, _rows(conn, 2), confirm=False)["preview_id"]
    conn.execute("UPDATE import_batches SET created_at = ? WHERE id = ?",
                 (store._now() - store.STAGED_IMPORT_TTL_SECONDS - 1, stale))
    conn.commit()

    store.bulk_import_metrics(conn, _rows(conn, 1), confirm=False)

    with pytest.raises(ValueError, match="not a staged workbook"):
        store.bulk_import_metrics(conn, None, confirm=True, preview_id=stale)


def test_a_recent_unfinished_preview_is_kept(conn):
    """The line the TTL must not cross. An import stopped by the time budget an hour ago is
    exactly the case the resume key exists for."""
    recent = store.bulk_import_metrics(conn, _rows(conn, 2), confirm=False)["preview_id"]
    conn.execute("UPDATE import_batches SET created_at = ? WHERE id = ?",
                 (store._now() - 3600, recent))
    conn.commit()

    store.bulk_import_metrics(conn, _rows(conn, 1), confirm=False)

    assert store.bulk_import_metrics(conn, None, confirm=True,
                                     preview_id=recent)["imported"] == 2


def _stop_after(monkeypatch, n: int):
    """Make the time budget run out after `n` rows, the way a slow machine does."""
    import time as time_module
    calls = {"n": 0}
    real = time_module.monotonic

    def creeping():
        calls["n"] += 1
        # The loop checks once per row, plus once to set the deadline.
        return real() + (0 if calls["n"] <= n + 1 else 10_000)

    monkeypatch.setattr(store.time, "monotonic", creeping)


def test_rows_and_a_key_together_are_refused(conn):
    """The staged rows won silently, so a caller who edited a row and resent it with the old
    key had the edit discarded with nothing in the result saying so. Two sources for one
    input, and the quiet one winning, which is the worst arrangement of that shape."""
    rows = _rows(conn, 2)
    preview = store.bulk_import_metrics(conn, rows, confirm=False)
    edited = [{**rows[0], "structured": {"roas": 99.0}}]

    with pytest.raises(ValueError, match="not both"):
        store.bulk_import_metrics(conn, edited, confirm=True,
                                  preview_id=preview["preview_id"])


def test_an_empty_staged_workbook_is_not_a_successful_import(conn):
    """"already_imported — all 0 row(s) of it" for a preview of nothing reads as a job done.
    `nothing_to_check` is never a pass, and neither is this."""
    preview = store.bulk_import_metrics(conn, [], confirm=False)

    with pytest.raises(ValueError, match="empty workbook"):
        store.bulk_import_metrics(conn, None, confirm=True,
                                  preview_id=preview["preview_id"])
