"""
§11.7 — treat it as personal data. A live obligation, not a position paper.

    "Treat it as personal data — install disclosure, per-person view, deletion or
     anonymisation preserving the judgment, stated retention position."

Three tracker rows call this a LIVE obligation rather than a question:

  D14   `campaign_chunks.source.author` is the first field holding a person's name harvested
        from a file rather than typed by the operator — PDF `/T` and PowerPoint comment
        authors — returned as `matched_author` on search hits and through `get_campaign`.
        Deletable only by cascade when the whole campaign is deleted.
  D111  D14's scope has GROWN. `correction_sightings.provenance` routinely carries a person's
        name and `after_upload` auto-populates it from those same comment authors;
        `_checked_correction` copies it onto saved evaluation findings, where campaign-cascade
        deletion cannot reach it, and `correction_sightings.campaign_id` is `ON DELETE SET
        NULL`, so erasing a campaign silently weakens the evidence behind a standing rule
        without saying so.
  D22   `detail` interpolates exception text and asset paths, which on Windows embed
        `C:\\Users\\<name>\\` — and `detail` is explicitly "send this to support" text.

## The hard part is the second half of the item's own sentence

"Deletion **or anonymisation preserving the judgment**". Those pull against each other, and
the naive readings are both wrong:

- **Delete the rows** and a saved judgment that cited "R. Vega objected to the timeline"
  becomes a judgment citing nothing. The library then holds a finding whose evidence has
  silently vanished, which is worse than either keeping or removing it cleanly — it is the
  library asserting something it can no longer support.
- **Redact to a blank** and two findings citing the same person stop being connected. "Two
  reviewers objected" and "one reviewer objected twice" are different facts, and a blank
  makes them indistinguishable.

So: a stable pseudonym per person. The name is gone and nothing can recover it; the structure
is intact, so a judgment still reads as one person objecting twice; and every row that was
changed says it was changed, because a record silently rewritten is a record nobody can
trust.
"""
import pathlib
import unicodedata

import pytest

import core
import people
import store


def _stored_data(conn) -> str:
    """Every row in the database, without the schema.

    `iterdump()` includes the CREATE TABLE statements, and this file's own schema comments use
    "R. Vega" as the illustrative name — so a dump-wide scan finds the documentation and
    reports a name that was erased. The check has to be about DATA, which is what was asked to
    be erased.
    """
    return "\n".join(line for line in conn.iterdump() if line.startswith("INSERT"))


def _campaign(conn, title="Colombia", **kw):
    kw.setdefault("market", "LATAM")
    kw.setdefault("status", "concluded")
    kw.setdefault("detail", f"A campaign called {title}, which ran in a market.")
    return core.ingest_campaign(conn, title=title, **kw)["campaign_id"]


def _a_library_mentioning(conn, name):
    """One person's name in as many places as this library can put it."""
    import context

    cid = _campaign(conn, "Peru launch", market="Peru")
    context.record(conn, starts_on="2026-09-05", ends_on="2026-09-14", scope="market",
                   scope_value="Peru", kind="supply_chain",
                   description="The port was shut for nine days.", recorded_by=name)
    core.record_reaction(conn, campaign_id=cid, value="liked", said_by=name)
    store.insert_chunks(conn, cid, ["The timeline here looks short."], kind="commentary",
                        sources=[{"kind": "comment", "author": name, "anchor": "slide 3"}])
    store.add_feedback_note(conn, campaign_id=cid, note="Slide 23 should be the standard.",
                            said_by=name, said_at="2026-09-20T09:00:00+00:00")
    return cid


# ── per-person view ────────────────────────────────────────────────────────

def test_everything_this_library_holds_about_one_person(conn):
    """The question a data subject actually asks, and the one this library could not answer:
    the name is spread across a dozen tables and two JSON blobs, and nothing could gather it."""
    _a_library_mentioning(conn, "R. Vega")

    held = people.on_file(conn, "R. Vega")

    assert held["name"] == "R. Vega"
    assert held["mentions"] >= 4
    assert {w["table"] for w in held["where"]} >= {
        "context_events", "reactions", "feedback_notes", "campaign_chunks"}


def test_the_view_says_what_each_mention_is(conn):
    """A list of table names is a list only an engineer can read. Somebody asking what this
    product holds about them is owed what it MEANS — a view they recorded, a comment they left
    on a deck — because that is what they are deciding whether to have erased."""
    _a_library_mentioning(conn, "R. Vega")

    held = people.on_file(conn, "R. Vega")

    described = {w["what"] for w in held["where"]}
    assert any("recorded" in d for d in described)
    assert all(w.get("what") for w in held["where"])


def test_a_name_nobody_holds_says_so_rather_than_returning_nothing(conn):
    """`nothing_to_check` is never a pass. "We hold nothing about you" is a real answer to a
    subject access request and must be distinguishable from a search that failed."""
    _a_library_mentioning(conn, "R. Vega")

    held = people.on_file(conn, "A. Duarte")

    assert held["status"] == "nothing_on_file"
    assert held["mentions"] == 0


def test_the_view_is_case_and_spacing_insensitive(conn):
    """A name is not an identifier. Somebody asking about "r. vega" is asking about the same
    person, and a view that misses them because of a capital reports "we hold nothing"."""
    _a_library_mentioning(conn, "R. Vega")

    assert people.on_file(conn, "  r. vega ")["mentions"] >= 4


# ── erasure that preserves the judgment ────────────────────────────────────

def test_erasure_replaces_the_name_everywhere(conn):
    """The name is gone from every table and every blob. Anything less is a promise the
    product cannot keep, and this is the promise the install disclosure has to make."""
    _a_library_mentioning(conn, "R. Vega")

    people.erase(conn, "R. Vega", why="Subject access request, 12 October",
                 said_by="N. Bhattacharya")

    assert people.on_file(conn, "R. Vega")["mentions"] == 0
    assert "R. Vega" not in _stored_data(conn)


def test_the_name_is_gone_from_the_file_itself(conn):
    """SQLite leaves old page contents in the free list, so the name stayed readable in the
    raw file after a successful erasure — review read it straight out. `iterdump()` only sees
    live rows, so every test here passed on a file that still held it. `VACUUM` is what
    rewrites the file without those pages.

    The fixture keeps a DELETED record on purpose. An UPDATE that fits usually rewrites its
    cell in place, so a library built only of live rows loses the name from the bytes whether
    VACUUM runs or not — the first version of this test asserted exactly that and stayed
    green with the VACUUM deleted, which mutation found. Pages freed by a deletion are the
    case that bites, and it is not a contrived one: a withdrawn pitch is dropped, and every
    mention of its reviewer sits unreferenced in the file, invisible to the scan that erase
    does and present in any copy of the database.

    A distinctive fixture name on purpose: this file's own schema comments use "R. Vega" as
    the illustrative name, and `sqlite_master` stores the schema text — so asserting on that
    name would fail against documentation rather than data, which is a different thing and
    not one erasure should touch."""
    cid = _campaign(conn, "Peru launch")
    core.record_reaction(conn, campaign_id=cid, value="liked", said_by="Xylophone Quandary")

    withdrawn = _campaign(conn, "Withdrawn pitch")
    for slide in range(40):
        store.insert_chunks(conn, withdrawn, ["Filler. " * 200], kind="commentary",
                            sources=[{"kind": "comment", "author": "Xylophone Quandary",
                                      "anchor": f"slide {slide}"}])
    conn.commit()
    conn.execute("DELETE FROM campaign_chunks WHERE campaign_id = ?", (withdrawn,))
    conn.execute("DELETE FROM campaigns WHERE id = ?", (withdrawn,))
    conn.commit()

    raw = pathlib.Path(store.config.DB_PATH)
    assert b"Xylophone Quandary" in raw.read_bytes(), "the fixture has to be on disk first"
    assert conn.execute("PRAGMA freelist_count").fetchone()[0] > 0, (
        "and the deletion has to have left unreferenced pages, or this proves nothing"
    )

    people.erase(conn, "Xylophone Quandary", why="SAR", said_by="N. Bhattacharya")

    assert b"Xylophone Quandary" not in raw.read_bytes()
    assert conn.execute("PRAGMA freelist_count").fetchone()[0] == 0


def test_the_same_person_stays_the_same_person(conn):
    """"Anonymisation PRESERVING THE JUDGMENT." Redacting to a blank makes "two reviewers
    objected" and "one reviewer objected twice" indistinguishable — different facts, and the
    difference is often the whole finding."""
    cid = _a_library_mentioning(conn, "R. Vega")
    core.record_reaction(conn, campaign_id=cid, value="performed_well", said_by="R. Vega")

    erased = people.erase(conn, "R. Vega", why="Subject access request",
                          said_by="N. Bhattacharya")

    names = {v["said_by"] for v in store.reactions_for(conn, cid)}
    assert names == {erased["pseudonym"]}
    assert len(names) == 1, "one person before, one person after"


def test_two_different_people_stay_different(conn):
    """The other half. A pseudonym shared between two people would merge two voices into one
    — and §11.5's whole mechanism is two people disagreeing."""
    cid = _a_library_mentioning(conn, "R. Vega")
    core.record_reaction(conn, campaign_id=cid, value="not_liked", said_by="A. Duarte")

    first = people.erase(conn, "R. Vega", why="SAR", said_by="N. Bhattacharya")
    second = people.erase(conn, "A. Duarte", why="SAR", said_by="N. Bhattacharya")

    assert first["pseudonym"] != second["pseudonym"]
    assert store.get_campaign(conn, cid)["disagreement"], (
        "and the disagreement between them survives, which is the judgment being preserved"
    )


def test_the_pseudonym_says_it_is_one(conn):
    """A reader must never take it for a name. "Person 7" beside "R. Vega" in the same field
    is a record that reads as two colleagues, one of whom has an odd name."""
    _a_library_mentioning(conn, "R. Vega")

    erased = people.erase(conn, "R. Vega", why="SAR", said_by="N. Bhattacharya")

    assert erased["pseudonym"].startswith("erased-")
    assert erased["status"] == "pseudonymised"


def test_the_erasure_itself_is_on_the_record(conn):
    """A record silently rewritten is a record nobody can trust. What was erased, when, why
    and on whose authority — without the name, which would defeat the whole exercise."""
    _a_library_mentioning(conn, "R. Vega")

    erased = people.erase(conn, "R. Vega", why="Subject access request, 12 October",
                          said_by="N. Bhattacharya")

    log = people.erasures(conn)
    assert log[0]["pseudonym"] == erased["pseudonym"]
    assert log[0]["why"] == "Subject access request, 12 October"
    assert log[0]["said_by"] == "N. Bhattacharya"
    assert "R. Vega" not in str(log), "the log must not re-create what it records removing"


def test_erasure_needs_a_reason_and_a_person(conn):
    """It is irreversible and it rewrites records a judgment rests on. An erasure nobody's
    name is against is one nobody can question, on the one operation that cannot be undone."""
    _a_library_mentioning(conn, "R. Vega")

    with pytest.raises(ValueError, match="why"):
        people.erase(conn, "R. Vega", why="", said_by="N. Bhattacharya")
    with pytest.raises(ValueError, match="must be a PERSON"):
        people.erase(conn, "R. Vega", why="SAR", said_by="the system")


def test_erasing_a_name_nobody_holds_is_refused(conn):
    """A silent no-op would answer "done" to a data subject whose name is spelled differently
    in the library — the one place a false reassurance is a compliance failure rather than an
    inconvenience."""
    _a_library_mentioning(conn, "R. Vega")

    with pytest.raises(ValueError, match="nothing on file"):
        people.erase(conn, "Someone Else", why="SAR", said_by="N. Bhattacharya")


# ── D111: the paths campaign deletion never reached ────────────────────────

def test_a_name_inside_a_saved_finding_is_reached(conn):
    """D111's real path: `_checked_correction` joins every sighting's provenance — and the
    name of whoever confirmed the rule — onto the saved finding, where campaign-cascade
    deletion cannot reach it. A JSON blob is still personal data.

    Built through the actual mechanism rather than by passing `provenance` into a precedent,
    which `save_evaluation` strips: a fixture that fakes the route proves nothing about it.
    """
    import corrections

    rule = "Never put the logo on a dark background."
    noted = None
    for n, market in enumerate(("LATAM", "EMEA", "APAC")):
        cid = _campaign(conn, f"Campaign {n}", market=market)
        noted = corrections.note(conn, text=rule, campaign_id=cid,
                                 provenance=f"R. Vega, client email, item {n}")
    corrections.graduate(conn, noted["correction_id"], confirmed_by="R. Vega")

    judged = core.save_evaluation(
        conn, subject_title="A new brief", campaign_id=cid, verdict="reject",
        summary="It breaks a standing rule.",
        findings=[{"severity": "blocking", "kind": "guardrail_breach",
                   "finding": "The logo sits on a dark background.",
                   "fix": "Move it onto the light panel.",
                   "precedent": {"correction_id": noted["correction_id"], "quote": rule}}])
    stored = conn.execute("SELECT findings FROM evaluations WHERE id = ?",
                          (judged["evaluation_id"],)).fetchone()["findings"]
    assert "R. Vega" in stored, "the fixture has to reproduce D111's copy, or it tests nothing"

    people.erase(conn, "R. Vega", why="SAR", said_by="N. Bhattacharya")

    assert "R. Vega" not in _stored_data(conn)


def test_a_correction_sighting_keeps_its_evidence(conn):
    """D111's other half: erasing must not silently weaken the evidence behind a standing
    rule. The sighting stays — the rule still recurs across the markets it recurred in — and
    only the name inside its provenance is replaced."""
    import corrections

    cid = _campaign(conn, "Peru launch")
    corrections.note(conn, text="Never put the logo on a dark background.",
                     provenance="R. Vega, client email, 4 March", campaign_id=cid)
    before = len(corrections.sightings(conn, corrections.find(
        conn, "Never put the logo on a dark background.")["correction_id"]))

    people.erase(conn, "R. Vega", why="SAR", said_by="N. Bhattacharya")

    after = corrections.sightings(conn, corrections.find(
        conn, "Never put the logo on a dark background.")["correction_id"])
    assert len(after) == before
    assert "R. Vega" not in str(after)


# ── the stated position ────────────────────────────────────────────────────

def test_the_product_states_what_it_holds_and_for_how_long(conn):
    """"stated retention position" — the item's own words, and the thing an install
    disclosure has to be built from. A product that cannot say what it keeps cannot ask
    anybody to agree to it."""
    said = people.retention()

    assert said["personal_data"], "what it holds"
    assert said["retention"], "for how long"
    assert said["lawful_basis_note"]
    assert said["how_to_see_it"] and said["how_to_erase_it"]


def test_the_position_names_the_harvested_field(conn):
    """D14's own point: `campaign_chunks.source.author` is the first field holding a person's
    name harvested from a FILE rather than typed by the operator, and that is the one a
    customer will not expect."""
    said = people.retention()

    harvested = [h for h in said["personal_data"] if h.get("harvested")]
    assert harvested
    assert any("comment" in h["what"].lower() for h in harvested)


# ── D22: "send this to support" text that embeds a home directory ──────────

def test_a_warning_does_not_carry_a_home_directory(conn):
    """D22: `detail` interpolates exception text and asset paths, which on Windows embed
    `C:\\Users\\<name>\\` — and `detail` is explicitly the field this product tells people to
    send to support. A filesystem path is not obviously personal data until you notice that
    on two of three platforms it contains somebody's name."""
    import notices

    said = notices.notice(
        "image_not_stored",
        detail=r"could not read C:\Users\rvega\Desktop\Q3 deck.pptx: permission denied")

    assert "rvega" not in said["detail"]
    assert "Q3 deck.pptx" in said["detail"], "the filename is what makes it diagnosable"


def test_the_posix_home_is_redacted_too(conn):
    import notices

    said = notices.notice("image_not_stored",
                          detail="could not read /Users/rvega/Desktop/deck.pdf: no such file")

    assert "rvega" not in said["detail"]
    assert "deck.pdf" in said["detail"]


def test_a_path_with_no_person_in_it_is_left_alone(conn):
    """Redaction that fires on every path would make support text unreadable to solve a
    problem it does not have."""
    import notices

    said = notices.notice("image_not_stored",
                          detail="could not read /opt/campaign/models/weights.bin")

    assert said["detail"].endswith("/opt/campaign/models/weights.bin")


# ── D26: the disclosure has somewhere to come from ────────────────────────

def test_the_install_disclosure_is_generated_from_the_position(conn):
    """D26 asks for a disclosure at install, and D30 for a retention position on the config
    backup. Both have to be built from the SAME statement the product answers questions with —
    a disclosure maintained separately is the copy that drifts, and here the drift is between
    what a customer agreed to and what the product does."""
    text = people.install_disclosure()

    assert "PDF annotation" in text or "comment author" in text
    for held in people.retention()["personal_data"]:
        if held.get("harvested"):
            assert held["what"].split(":")[0][:30] in text


def test_the_disclosure_says_what_the_config_backup_holds(conn):
    """D30: the `.bak` of `claude_desktop_config.json` is a second copy of other connectors'
    environment — possibly their tokens — rewritten on every run, with no retention position.
    It is not this product's data and that is exactly why it needs saying."""
    text = people.install_disclosure()

    assert "claude_desktop_config" in text
    assert ".bak" in text


# ── the promise has to stay true as the schema grows ──────────────────────

def test_no_column_holding_a_name_is_unlisted():
    """`NAME_COLUMNS` claims to be exhaustive, and a list maintained by hand beside a schema
    is the copy that drifts — this codebase has been bitten by that shape six times. Here the
    drift is a name this product promised a data subject it would erase, and did not.

    So the schema is walked, and any column that looks like it holds a person has to be either
    listed or explicitly excused. The excuse list is short and each entry says why."""
    import re

    schema = store._SCHEMA
    listed = {(t, c) for t, c, _ in people.NAME_COLUMNS + people.NAME_IN_TEXT}
    listed |= {(t, c) for t, c, _ in people.NAME_IN_JSON}

    suspicious = set()
    for block in re.split(r"CREATE TABLE IF NOT EXISTS ", schema)[1:]:
        table = block.split("(")[0].strip()
        for line in block.split("\n"):
            m = re.match(r"\s*([a-z_]+)\s+TEXT", line)
            if m and any(w in m.group(1) for w in
                         ("_by", "author", "said_by", "provenance", "display", "account")):
                suspicious.add((table, m.group(1)))

    assert suspicious - listed <= EXCUSED, (
        f"these columns look like they hold a person and are neither erasable nor excused: "
        f"{sorted(suspicious - listed - EXCUSED)}"
    )


# Columns that match the shape and are not a person, each with the reason. Short on purpose:
# an excuse nobody has to justify is how the exhaustive list stops being exhaustive.
EXCUSED = {
    # A pseudonym and the person who PERFORMED an erasure. `said_by` here is the operator's
    # own name and is genuinely theirs to erase — but erasing it through this path would
    # rewrite the log of erasures while it is being written, so it is excluded and named in
    # the retention position instead.
    ("erasures", "pseudonym"), ("erasures", "said_by"),
    # Not a person: which of `verified`/`stated` a claim is, and how an account was
    # established. (`captured_host` IS reachable — it is in NAME_IN_TEXT, because a machine
    # name routinely contains a login. This guard found that the comment here claimed it was
    # listed while it was not, which is the drift the guard exists to catch, caught on its
    # first run against the file that asserts the list is exhaustive.)
    ("authorship", "captured_source"), ("authorship", "captured_method"),
    # A DATE: when this library stamped "we never looked" onto a record's authorship (§11.6).
    ("campaigns", "authorship_backfilled_at"),
    # A MEASURE's own name — "Sell-through at 60 days" — not a person's.
    ("metric_registry", "display_name"),
}


def test_a_name_inside_a_longer_word_is_left_alone(conn):
    """`_swap` is whole-word. A plain substring replace turns "Al" into a token inside "Also"
    and "Alameda" — mangling a sentence somebody has to read, which is a different way of
    losing the record than the one erasure is preventing.

    Tested on `correction_sightings.provenance`, which is a field erasure ACTUALLY visits. The
    first version of this test used a feedback note and passed for the wrong reason: free
    prose is deliberately never searched, so the sentence survived because nothing touched it
    — the test would have stayed green with the boundary deleted. Mutation found that."""
    import corrections

    cid = _campaign(conn, "Peru launch")
    corrections.note(conn, text="Never put the logo on a dark background.", campaign_id=cid,
                     provenance="Al, client call — also raised on the Alameda deck")

    people.erase(conn, "Al", why="SAR", said_by="N. Bhattacharya")

    sighting = corrections.sightings(conn, corrections.find(
        conn, "Never put the logo on a dark background.")["correction_id"])[0]
    assert "also raised on the Alameda deck" in sighting["provenance"], (
        "the rest of the sentence is the provenance a rule's evidence rests on"
    )
    assert sighting["provenance"].startswith("erased-"), "and the name itself IS replaced"


def test_a_name_with_regex_characters_in_it_is_handled(conn):
    """Names contain punctuation. `re.escape` is what keeps "D'Angelo (Design)" from being
    compiled as a pattern — and a crash here fails a subject access request."""
    cid = _campaign(conn, "Peru launch")
    core.record_reaction(conn, campaign_id=cid, value="liked", said_by="D'Angelo (Design)")

    erased = people.erase(conn, "D'Angelo (Design)", why="SAR", said_by="N. Bhattacharya")

    assert erased["mentions"] >= 1
    assert people.on_file(conn, "D'Angelo (Design)")["mentions"] == 0


# ── found by review: the promise that was not kept ────────────────────────

def test_the_tag_blob_is_erased_too(conn):
    """`campaigns.tags` is where the MENU writes `said_by` — the product's main feedback path
    — and it was not in the erasure list. Worse than a miss: `person_on_file` reported ZERO
    mentions while the name sat in the blob and came back from `get_campaign`, so a DPO got a
    clean bill with the name still on screen."""
    import feedback

    cid = _campaign(conn, "Peru launch")
    menu = feedback.queue(conn)
    row = next(r for r in menu["rows"] if r.get("campaign_id") == cid)
    feedback.record(conn, menu_token=menu["menu_token"], choice=row["number"],
                    said_by="R. Vega", reaction=1)
    assert "R. Vega" in str(store.get_campaign(conn, cid)["tags"])

    assert people.on_file(conn, "R. Vega")["mentions"] >= 1, (
        "the view has to SEE it before erasure can be believed"
    )
    people.erase(conn, "R. Vega", why="SAR", said_by="N. Bhattacharya")

    assert "R. Vega" not in str(store.get_campaign(conn, cid)["tags"])


def test_an_erased_name_cannot_come_back_through_an_edit(conn):
    """`_keep_the_view` re-reads `tag["said_by"]` on every `update_campaign`. With the blob
    unerased, a routine edit next week appends a fresh reaction row under the old name — the
    DPO's clean "0 mentions" undone by somebody fixing a title, and the erased person
    reappearing as a SECOND voice who can then disagree with their own erased self."""
    import feedback

    cid = _campaign(conn, "Peru launch")
    menu = feedback.queue(conn)
    row = next(r for r in menu["rows"] if r.get("campaign_id") == cid)
    feedback.record(conn, menu_token=menu["menu_token"], choice=row["number"],
                    said_by="R. Vega", reaction=1)
    people.erase(conn, "R. Vega", why="SAR", said_by="N. Bhattacharya")

    core.update_campaign(conn, campaign_id=cid,
                         tags=store.get_campaign(conn, cid)["tags"])

    assert people.on_file(conn, "R. Vega")["mentions"] == 0
    assert not any(v["said_by"] == "R. Vega" for v in store.reactions_for(conn, cid))


def test_the_guard_walks_json_columns_too():
    """The exhaustiveness guard matched on COLUMN NAMES, so `campaigns.tags` — a TEXT column
    holding JSON with `said_by` inside — passed it while the promise failed. A guard that
    cannot see the shape of the defect is a guard that reports the absence of the defects it
    can see."""
    import re

    listed = {(t, c) for t, c, _ in
              people.NAME_COLUMNS + people.NAME_IN_TEXT + people.NAME_IN_JSON}
    # Any column this product json.dumps a person's name into, found by looking for the
    # writers rather than by reading the schema.
    writes_names_as_json = {("campaigns", "tags"), ("campaign_chunks", "source"),
                            ("evaluations", "findings")}

    assert writes_names_as_json <= listed, (
        f"these hold a name inside JSON and are not erasable: "
        f"{sorted(writes_names_as_json - listed)}"
    )


# ── found by review: five ways a name survived "erasure" ──────────────────

def test_an_accented_name_is_found_and_erased(conn):
    """SQLite's `LOWER()` is ASCII-only — `LOWER('Ángel')` is `'Ángel'` — while Python's
    `casefold()` gives `'ángel'`, so the comparison could never match. And `json.dumps`
    defaults to `ensure_ascii=True`, so `José` is on disk as `Jos\\u00e9` and no LIKE on the
    literal finds it either.

    These are exactly the LATAM names harvested from PDF and PowerPoint comment authors —
    the field D14 is about. A data subject called Ángel was told this library held nothing."""
    cid = _campaign(conn, "Peru launch")
    core.record_reaction(conn, campaign_id=cid, value="liked", said_by="José Álvarez")
    store.insert_chunks(conn, cid, ["The timeline looks short."], kind="commentary",
                        sources=[{"kind": "comment", "author": "José Álvarez",
                                  "anchor": "slide 3"}])

    assert people.on_file(conn, "José Álvarez")["mentions"] >= 2
    assert people.on_file(conn, "josé álvarez")["mentions"] >= 2

    people.erase(conn, "José Álvarez", why="SAR", said_by="N. Bhattacharya")

    assert people.on_file(conn, "José Álvarez")["mentions"] == 0
    assert "José" not in _stored_data(conn) and "Jos\\u00e9" not in _stored_data(conn)


def test_the_same_name_spelled_with_different_spacing_is_one_person(conn):
    """`on_file` folded the name for its SQL and `_swap` escaped it unfolded, so "R.  Vega"
    with two spaces found the rows, rewrote the plain columns, silently failed on the JSON
    ones, and returned `erased` anyway."""
    cid = _campaign(conn, "Peru launch")
    store.insert_chunks(conn, cid, ["The timeline looks short."], kind="commentary",
                        sources=[{"kind": "comment", "author": "R. Vega", "anchor": "s3"}])
    core.record_reaction(conn, campaign_id=cid, value="liked", said_by="R. Vega")

    people.erase(conn, "R.  Vega", why="SAR", said_by="N. Bhattacharya")

    assert "R. Vega" not in _stored_data(conn)


def test_erasing_one_person_does_not_rewrite_another(conn):
    """Whole-word treats a space as a boundary, so erasing "Vega" rewrote the deck author
    "Ana Vega" into "Ana erased-…" — a SECOND person pseudonymised without being asked, and
    linked to the first person's token. `on_file` counted her mention as his, too."""
    cid = _campaign(conn, "Peru launch")
    store.insert_chunks(conn, cid, ["The timeline looks short."], kind="commentary",
                        sources=[{"kind": "comment", "author": "Ana Vega", "anchor": "s3"}])

    with pytest.raises(ValueError, match="is part of"):
        people.erase(conn, "Vega", why="SAR", said_by="N. Bhattacharya")

    assert "Ana Vega" in _stored_data(conn)


def test_a_name_with_a_wildcard_in_it_matches_only_itself(conn):
    """`%` and `_` were passed straight into LIKE, so `on_file('%')` reported every row as
    that person's and `erase('%')` sailed past the "nothing on file" refusal, changed nothing,
    and returned success — the worst possible answer to a subject access request."""
    cid = _campaign(conn, "Peru launch")
    core.record_reaction(conn, campaign_id=cid, value="liked", said_by="Ana Duarte")

    assert people.on_file(conn, "%")["mentions"] == 0
    assert people.on_file(conn, "A_a Duarte")["mentions"] == 0

    with pytest.raises(ValueError, match="nothing on file"):
        people.erase(conn, "%", why="SAR", said_by="N. Bhattacharya")


def test_the_view_lists_the_spellings_it_found(conn):
    """A name is not an identifier, and a refusal that just says "nothing on file" leaves a
    data subject guessing. What is on file is knowable, so the refusal says it."""
    cid = _campaign(conn, "Peru launch")
    core.record_reaction(conn, campaign_id=cid, value="liked", said_by="Ana Duarte")

    held = people.on_file(conn, "Ana Duarte")

    assert "Ana Duarte" in held["spellings"]


def test_the_disclosure_is_reachable_without_reading_the_source(conn):
    """D26 asks for a disclosure at install. It shipped as a string generator with ZERO call
    sites — closed by dead code, which is worse than not done, because the tracker said it was
    finished. A disclosure nobody is ever shown is not a disclosure, and this is the one
    artefact in the item whose value is entirely in WHEN it appears."""
    import subprocess
    import sys

    shown = subprocess.run([sys.executable, "-m", "main", "disclosure"],
                           capture_output=True, text=True, cwd=str(pathlib.Path(__file__).
                           resolve().parent.parent))

    assert shown.returncode == 0, shown.stderr
    assert "stores personal data on this machine" in shown.stdout
    assert "PDF annotation authors" in shown.stdout


def test_the_disclosure_says_where_the_words_actually_go(conn, monkeypatch):
    """It said "nothing is sent anywhere except the text you give an embedding provider,
    which never includes these fields" — true of the author COLUMN and false of the WORDS.
    §2.5 extracts each comment into its own chunk and every chunk is embedded, so with a
    cloud provider the client's own sentences leave the machine and the country. For a DPO
    that is the most consequential sentence in the document and it read as reassurance."""
    import config

    monkeypatch.setattr(config, "EMBED_PROVIDER", "voyage")
    remote = people.retention()["where_it_is"]

    assert "REMOTE" in remote
    assert "comment" in remote and "transfer" in remote

    monkeypatch.setattr(config, "EMBED_PROVIDER", "ollama")
    assert "no text leaves this machine" in people.retention()["where_it_is"]


def test_the_erasure_log_is_disclosed_rather_than_quietly_exempt(conn):
    """`erasures.said_by` holds a name permanently and is deliberately outside the erasure
    sweep. The test justifying that exemption said it was "named in the retention position
    instead" — and it was not. A test comment asserting a disclosure that does not exist is
    exactly the drift "generated, not maintained beside it" was meant to stop."""
    said = people.retention()["personal_data"]

    assert any("pseudonymised whom" in h["what"] for h in said)
    assert any("Delete the database" in h["why"] for h in said)


def test_two_people_never_share_a_token(conn):
    """A collision MERGES two people, and §11.5's whole mechanism is two voices being two.
    Eight hex characters is 2**32: about one collision in ten thousand at a thousand distinct
    names, which for a promise made to a data subject is not a rate, it is a defect waiting
    for a big enough library."""
    seen = {}
    for n in range(2000):
        name = f"Person Number {n}"
        token = people.pseudonym(conn, name)
        assert token not in seen, f"{name} collides with {seen.get(token)}"
        seen[token] = name


def test_the_salt_is_per_database_and_kept(conn):
    """Stable within a database, or one person's two mentions stop being one person's."""
    first = people.pseudonym(conn, "R. Vega")

    assert people.pseudonym(conn, "R. Vega") == first
    assert people.pseudonym(conn, "r.  vega") == first, "same person, different typing"


# ── what a DPO actually needs ─────────────────────────────────────────────

def test_the_view_returns_the_data_not_a_count_of_it(conn):
    """Art. 15 requires a copy of the personal data undergoing processing. "You appear 3 times
    in `reactions.said_by`" is not that — it is a receipt for data nobody can see, and a data
    subject cannot decide whether to ask for erasure without knowing what was said."""
    cid = _a_library_mentioning(conn, "R. Vega")
    core.record_reaction(conn, campaign_id=cid, value="liked", said_by="R. Vega")

    held = people.on_file(conn, "R. Vega")

    entries = [e for w in held["where"] for e in w["entries"]]
    assert entries, held
    assert any("Peru launch" in (e.get("about") or "") for e in entries), (
        "and each one says which record it is attached to, or it cannot be acted on"
    )
    assert all(e.get("recorded_at") for e in entries)


def test_a_misspelled_name_can_be_corrected_rather_than_erased(conn):
    """Art. 16. With rectification missing, the only remedy for a typo was the irreversible
    token — so somebody whose name was entered wrong had to choose between a wrong record and
    no record. It is the same sweep as pseudonymising with a different replacement."""
    cid = _a_library_mentioning(conn, "R. Vegga")

    corrected = people.rename(conn, "R. Vegga", to="R. Vega", why="Typo, corrected on request",
                              said_by="N. Bhattacharya")

    assert corrected["mentions"] >= 1
    assert people.on_file(conn, "R. Vegga")["mentions"] == 0
    assert people.on_file(conn, "R. Vega")["mentions"] >= 1


def test_a_rename_cannot_quietly_merge_two_people(conn):
    """The one thing rectification must not do. Renaming "A. Duarte" to a name somebody else
    already uses would fold two people into one — and §11.5's whole mechanism is two voices
    being two."""
    cid = _a_library_mentioning(conn, "R. Vega")
    core.record_reaction(conn, campaign_id=cid, value="not_liked", said_by="A. Duarte")

    with pytest.raises(ValueError, match="already on file"):
        people.rename(conn, "A. Duarte", to="R. Vega", why="Typo",
                      said_by="N. Bhattacharya")


def test_the_position_says_who_the_controller_is(conn):
    """The paragraph a DPO needs before any other: this runs on the customer's machine against
    their own files and sends nothing to the vendor, so the customer is the controller and
    this software is not a processor at all. It is the strongest fact the product has and it
    was the one thing the position did not say."""
    said = people.retention()

    assert "controller" in said["controller_note"]
    assert "not a processor" in said["controller_note"]


def test_a_name_stored_decomposed_is_found_when_typed_composed(conn):
    """The same name reaches this library two ways — typed by an operator and harvested from a
    file — and macOS hands out DECOMPOSED forms, so "José" from a PDF and "José" typed into
    the menu can be different byte strings for one person.

    Mutation found that the earlier accented test could not see this: it used one consistent
    spelling, so composed and decomposed never differed and the normalisation could be
    deleted with nothing noticing."""
    import unicodedata

    decomposed = unicodedata.normalize("NFD", "José Álvarez")
    composed = unicodedata.normalize("NFC", "José Álvarez")
    assert decomposed != composed, "the fixture has to actually differ in bytes"

    cid = _campaign(conn, "Peru launch")
    store.insert_chunks(conn, cid, ["The timeline looks short."], kind="commentary",
                        sources=[{"kind": "comment", "author": decomposed,
                                  "anchor": "slide 3"}])

    assert people.on_file(conn, composed)["mentions"] >= 1

    people.erase(conn, composed, why="SAR", said_by="N. Bhattacharya")

    assert people.on_file(conn, composed)["mentions"] == 0
    assert people.on_file(conn, decomposed)["mentions"] == 0


def test_a_name_stored_with_irregular_spacing_is_erased_inside_a_sentence(conn):
    """A name pasted out of a PDF carries whatever spacing the PDF had. `_swap` joins the
    parts of the folded name with `\\s+`, so "R.  Vega" sitting INSIDE a sentence is reached
    by "R. Vega" typed into the tool.

    Beside other text specifically. A value that is nothing but the name takes `_replaced`'s
    exact-match branch, which folds both sides and never consults the pattern — so an earlier
    version of this test erased the name through a path the `\\s+` join does not sit on, and
    stayed green with the join deleted. `correction_sightings.provenance` is the shape that
    does go through `_swap`: the comma-separated head is what `_scan` recognises as a person,
    and the rest of the line has to survive."""
    import corrections

    cid = _campaign(conn, "Peru launch")
    corrections.note(conn, text="Never put the logo on a dark background.", campaign_id=cid,
                     provenance="R.  Vega, client email, 4 March")

    people.erase(conn, "R. Vega", why="SAR", said_by="N. Bhattacharya")

    sighting = corrections.sightings(conn, corrections.find(
        conn, "Never put the logo on a dark background.")["correction_id"])[0]
    assert "Vega" not in sighting["provenance"], (
        "the stored spelling is the one nobody controls; the typed one is all a subject has"
    )
    assert "client email, 4 March" in sighting["provenance"]


def test_the_token_is_long_enough_and_salted(conn):
    """Both properties, asserted directly rather than inferred. The earlier pair could not
    fail: the bare-hash test compared against a 16-character slice while the mutation produced
    8, and 2,000 names do not collide at 2**32 often enough to catch a shortened token."""
    import hashlib

    token = people.pseudonym(conn, "R. Vega")
    digest = token[len("erased-"):]

    assert len(digest) == 16, "8 hex is 2**32 — a collision merges two people"
    assert digest != hashlib.sha256("r. vega".encode()).hexdigest()[:16], (
        "unsalted, anybody with a staff directory confirms a guess in microseconds"
    )


def test_the_salt_is_per_database(conn, tmp_path, monkeypatch):
    """C88's actual claim, which the test above cannot make: the salt defeats a guess for
    somebody holding the TOKEN without the database. One salt shared across installs would
    make every token comparable between them, and a token that identifies the same person in
    two customers' libraries is the thing pseudonymisation is for.

    (The earlier version of this file asserted `digest != sha256(...)[:8]` three lines after
    asserting `len(digest) == 16`. A 16-character string is never equal to an 8-character
    one — a tautology sitting in the test whose docstring is about a comparison against the
    wrong slice.)"""
    import config

    first = people.pseudonym(conn, "R. Vega")

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "second.db")
    store.init_db()
    second = store.connect()
    try:
        assert people.pseudonym(second, "R. Vega") != first
    finally:
        second.close()


def test_a_decomposed_name_inside_a_sentence_is_actually_replaced(conn):
    """C89 fixed the folding for FINDING a name and not for REPLACING one inside text.

    `_scan` folds to NFC and matches. `_swap` then built its pattern from that FOLDED string
    and ran it against the RAW stored bytes — so a macOS-decomposed "José" was found, not
    replaced, and the row was skipped. macOS hands out decomposed forms; `people._folded`'s
    own docstring says so."""
    import corrections

    decomposed = unicodedata.normalize("NFD", "José Álvarez")
    cid = _campaign(conn, "Peru launch")
    corrections.note(conn, text="Never put the logo on a dark background.", campaign_id=cid,
                     provenance=f"{decomposed}, client email, 4 March")

    people.erase(conn, "José Álvarez", why="SAR", said_by="N. Bhattacharya")

    assert people.on_file(conn, "José Álvarez")["mentions"] == 0
    sighting = corrections.sightings(conn, corrections.find(
        conn, "Never put the logo on a dark background.")["correction_id"])[0]
    assert "lvarez" not in sighting["provenance"]
    assert "client email, 4 March" in sighting["provenance"]


def test_an_erasure_that_missed_a_row_does_not_report_success(conn):
    """The worst sentence this product can produce. `changed` could be less than the number of
    rows found and nothing said so — no notice, no counter, no different `status` — so the
    observable result of a TOTAL failure was `status: pseudonymised` and a paragraph telling a
    data subject their name is gone.

    Whatever the cause (a folding this code has not learned yet, a row another process is
    holding), the guarantee has to be: either it is gone, or you are told it is not."""
    import corrections

    cid = _campaign(conn, "Peru launch")
    # In a SENTENCE: a value that is nothing but the name is replaced by `_replaced`'s
    # exact-match branch, which never reaches `_swap` at all.
    corrections.note(conn, text="Never put the logo on a dark background.", campaign_id=cid,
                     provenance="R. Vega, client email, 4 March")

    original = people._swap
    people._swap = lambda value, wanted, token: value     # a replace that quietly does nothing
    try:
        with pytest.raises(ValueError, match="still on file|could not"):
            people.erase(conn, "R. Vega", why="SAR", said_by="N. Bhattacharya")
    finally:
        people._swap = original

    assert people.on_file(conn, "R. Vega")["mentions"] >= 1, "and nothing was half-rewritten"


def test_the_controller_statement_does_not_claim_privacy_the_deployment_may_not_have(
        monkeypatch):
    """"a database only you can read" is true of the stdio install and false of the `serve`
    subcommand this product ships: `auth`'s default provider is `none`, every request is
    anonymous, and `identity.METHODS` carries `unattributed` precisely for "reachable over the
    network with no identity provider configured".

    `_where_it_is` next door was fixed to read the config rather than assert the default,
    and this sentence was not given the same treatment — while being, in its own words, the
    strongest fact this product has, and therefore the one a DPO will quote back."""
    import config

    monkeypatch.setattr(config, "SERVING_HTTP", True, raising=False)
    served = people.retention()["controller_note"]

    assert "only you can read" not in served
    assert "network" in served.lower() or "http" in served.lower()
    assert "not a processor" in served, "the actual controller fact still has to be there"

    monkeypatch.setattr(config, "SERVING_HTTP", False, raising=False)
    local = people.retention()["controller_note"]
    assert "on this machine" in local
    assert "network" not in local.lower()
    # Not "only you can read" either, which the local case also does not have: a machine with
    # two accounts on it has two people who can open the file. The replacement says who can,
    # rather than naming one person and hoping.
    assert "anyone with an account on it" in local


def test_the_install_disclosure_carries_the_controller_statement():
    """D26 wired the disclosure to the install moment; C98 called the controller statement
    the strongest fact this product has — and the disclosure did not contain it. It was
    reachable only by calling a tool, which is not the moment anybody is deciding whether to
    install this."""
    assert "not a processor" in people.install_disclosure()
