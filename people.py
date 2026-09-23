"""
§11.7 — the personal data in this library, and what a person can do about it.

A live obligation, not a position paper. Three tracker rows say so:

  D14   `campaign_chunks.source.author` is the first field holding a person's name harvested
        from a FILE rather than typed by the operator — PDF `/T` and PowerPoint comment
        authors — and it is returned as `matched_author` on search hits and through
        `get_campaign`. Deletable only by cascade, when the whole campaign is deleted.
  D111  That scope has grown. `correction_sightings.provenance` routinely carries a person's
        name and `after_upload` auto-populates it from those same comment authors;
        `_checked_correction` copies it onto saved evaluation findings, where campaign-cascade
        deletion cannot reach it at all.
  D22   Exception text and asset paths embed `C:\\Users\\<name>\\` on Windows, in a field
        whose whole purpose is "send this to support".

## Erasure that preserves the judgment

The item asks for "deletion **or anonymisation preserving the judgment**", and the two naive
readings are both wrong.

Delete the rows, and a saved judgment citing "R. Vega objected to the timeline" becomes a
judgment citing nothing — the library asserting a finding whose evidence has silently
vanished, which is worse than either keeping it or removing it cleanly.

Redact to a blank, and two findings citing one person stop being connected. "Two reviewers
objected" and "one reviewer objected twice" are different facts, and often the difference IS
the finding.

So: a stable pseudonym per person. The name is gone and nothing here can recover it — the
mapping is never stored, only the token — while the structure survives, so one person
objecting twice still reads as one person. Every changed row is recorded as changed, because
a record silently rewritten is a record nobody can trust.

## What this does not claim

It does not scrub free prose. A `note` or a `why` is somebody's own sentence and may mention
anyone; searching it for names would be a guess, and a guess that missed one would be worse
than the honest position, which is that those fields are user-authored and are named in the
disclosure as such. What IS exhaustive is every field this product PUTS a name in.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Optional

import identity

# Every column this product writes a person's name into. Exhaustive by construction — the
# test `test_no_column_holds_a_name_unlisted` walks the schema and fails when a new one
# appears, because a list maintained by hand beside a schema is the copy that drifts, and here
# a drift means a name this product promised to erase and did not.
NAME_COLUMNS = (
    ("context_events", "recorded_by", "recorded what else was going on in a market"),
    ("context_events", "withdrawn_by", "took a context event off the record"),
    ("context_attributions", "stated_by", "said what an event did to a campaign's numbers"),
    ("context_attributions", "withdrawn_by", "withdrew an account of an event"),
    ("reactions", "said_by", "recorded what they made of a campaign"),
    # §12.4/D15: whose sign-off a returned deck carries. A person, on a campaign row — and
    # §11.7's guard caught it the moment the column was added, which is what that guard is
    # for: a name in a column nothing scans is a name `person_on_file` reports as absent and
    # `pseudonymise_person` leaves behind.
    ("campaigns", "approval_by", "signed off on a deck"),
    ("answers", "said_by", "answered a question this library asked"),
    ("feedback_notes", "said_by", "left a note about a campaign"),
    ("corrections", "confirmed_by", "confirmed a standing rule"),
    ("metric_registry", "confirmed_by", "confirmed a measure"),
    ("drift_classifications", "classified_by", "judged a difference between brief and result"),
    ("authorship", "on_behalf_of", "is named as whose judgment something was"),
    ("authorship", "captured_display", "was the operator at the keyboard"),
    ("authorship", "captured_account", "was the account a call was made from"),
)

# Columns holding free text this product AUTO-POPULATES from a file, so a name in them got
# there because this library put it there (D111). Replaced by substring, not wholesale: the
# rest of the sentence is the provenance a rule's evidence rests on.
NAME_IN_TEXT = (
    ("correction_sightings", "provenance", "is named as where a client rule came from"),
    ("correction_sightings", "said_as", "is quoted stating a client rule"),
    # A machine name routinely contains a person's — "rvega-mbp" is the default on macOS and
    # common on Windows. Substring rather than exact, because the host is not the name: an
    # exact-match erasure would leave `rvega-mbp` standing beside every field that had been
    # cleaned, which is the promise kept everywhere except the one place nobody looks.
    ("authorship", "captured_host", "is named in the machine a call was made from"),
)

# JSON blobs this product writes names into. D14's original and D111's growth.
NAME_IN_JSON = (
    # The MENU's own path. `feedback.record` writes `said_by` into this blob on every answer,
    # and it was missing from this list — so `person_on_file` reported ZERO mentions while the
    # name sat in `campaigns.tags` and came straight back out of `get_campaign`. A DPO got a
    # clean bill with the name still on screen, and a routine `update_campaign` next week
    # would have re-read it and appended a fresh reaction row under the old name.
    ("campaigns", "tags", "recorded what they made of a campaign"),
    ("campaign_chunks", "source", "commented on or annotated a deck"),
    ("evaluations", "findings", "is cited as the source of a rule a judgment checked"),
    ("evaluations", "provenance", "is recorded in a judgment's provenance"),
)


def _folded(name: str) -> str:
    """One spelling of a name, for comparison only.

    NFC first, because the same name reaches this library two ways — typed by an operator and
    harvested from a PDF — and macOS hands out decomposed forms, so `José` and `José` can be
    different byte strings for the same person.

    Matching happens in PYTHON and never in SQL. SQLite's `LOWER()` is ASCII-only, so
    `LOWER('Ángel')` is `'Ángel'` while `casefold()` gives `'ángel'` — the comparison could
    not match, and a data subject called Ángel was told this library held nothing about them.
    `LIKE` was worse: `%` and `_` in a name were wildcards, so `on_file('%')` claimed every
    row and `erase('%')` reported success having changed nothing.
    """
    return " ".join(unicodedata.normalize("NFC", name or "").split()).casefold()


def _mentions(value, wanted: str) -> bool:
    """Whether this stored value is about the person we are looking for."""
    return wanted in _spellings(value)


def _spellings(value) -> set:
    """Every name-shaped string inside a stored value, folded.

    JSON is decoded rather than searched as text: `json.dumps` defaults to
    `ensure_ascii=True`, so `José` is on disk as `Jos\u00e9` and no search for the literal
    could ever find it. Decoding also means a name nested at any depth is reached, rather than
    only one the pattern happened to be written for.
    """
    out: set = set()
    if value is None:
        return out
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("{", "[")):
            try:
                return _spellings(json.loads(text))
            except ValueError:
                pass
        out.add(_folded(text))
        # A name inside a sentence — `correction_sightings.provenance` is "R. Vega, client
        # email, 4 March", and the comma-separated head is the part that is a person.
        for piece in re.split(r"[,;—|]", text):
            if piece.strip():
                out.add(_folded(piece))
        return out
    if isinstance(value, dict):
        for inner in value.values():
            out |= _spellings(inner)
        return out
    if isinstance(value, (list, tuple)):
        for inner in value:
            out |= _spellings(inner)
    return out


def _salt(conn) -> str:
    """One random secret per database, made once and kept (§11.7).

    Without it the token is `sha256(name)` — and the space of human names is small enough that
    anybody holding a staff directory can confirm a guess in microseconds. Review demonstrated
    exactly that against a four-name list. A per-database salt defeats that for anyone who has
    the token without the database.

    It does NOT make this anonymisation, and nothing here pretends it does: the salt lives in
    the same file, so a reader with the database and a candidate list can still confirm a
    match. That is what `pseudonymised` means, and why the tool says so.
    """
    import secrets
    import store

    held = store.get_state(conn, "pseudonym_salt")
    if not held:
        held = secrets.token_hex(16)
        store.set_state(conn, "pseudonym_salt", held)
    return held


def pseudonym(conn, name: str) -> str:
    """A stable token for one person (§11.7).

    Stable so that one person objecting twice still reads as one person — the "preserving the
    judgment" half of the item. Sixteen hex characters rather than eight: at 2**32 a thousand
    distinct names collide about one time in ten thousand, and a collision MERGES TWO PEOPLE,
    which is the one thing this must never do — §11.5's whole mechanism is two voices being
    two.

    No mapping is stored, so this product cannot reverse it. That is not the same as nobody
    being able to: see `_salt`, and see every user-facing string, which say `pseudonymised`
    rather than `erased` for that reason.
    """
    digest = hashlib.sha256(f"{_salt(conn)}:{_folded(name)}".encode("utf-8")).hexdigest()[:16]
    return f"erased-{digest}"


# How many entries to show per place. A subject access request over a large library should
# not return the whole library, and the count beside them says what was not shown.
_MAX_SHOWN = 20

# Where to look for the record a mention is attached to, per table. A row id is not something
# a person can act on; the campaign it belongs to is.
_ATTACHED_TO = {
    "reactions": "campaign_id", "feedback_notes": "campaign_id",
    "campaign_chunks": "campaign_id", "context_attributions": "campaign_id",
    "campaigns": "id",
}


def _readable(conn, table: str, column: str, rowid: int, value) -> dict:
    """One mention, in a form the person it is about can read."""
    entry: dict = {"value": value if isinstance(value, str) else json.dumps(value)}
    row = conn.execute(
        f"SELECT * FROM {table} WHERE rowid = ?", (rowid,)).fetchone()   # noqa: S608
    keys = set(row.keys()) if row else set()
    for stamp in ("said_at", "created_at", "captured_at", "erased_at", "first_seen"):
        if stamp in keys and row[stamp]:
            entry["recorded_at"] = str(row[stamp])
            break
    entry.setdefault("recorded_at", "")
    attached = _ATTACHED_TO.get(table)
    if attached and attached in keys and row[attached]:
        record = conn.execute("SELECT title FROM campaigns WHERE id = ?",
                              (row[attached],)).fetchone()
        if record:
            entry["about"] = record["title"]
    return entry


def _scan(conn, wanted: str):
    """Every stored value that is about this person, with where it came from.

    Reads rows and compares in Python. SQL comparison was wrong three separate ways — ASCII
    `LOWER`, unescaped `LIKE` wildcards, and `ensure_ascii` JSON — each of which made a name
    invisible to the view that is supposed to find it.
    """
    import store

    for table, column, what in NAME_COLUMNS + NAME_IN_TEXT + NAME_IN_JSON:
        if not store._columns(conn, table):
            continue
        rows = conn.execute(
            f"SELECT rowid AS _row, {column} AS value FROM {table} "   # noqa: S608 — fixed list
            f"WHERE {column} IS NOT NULL").fetchall()
        for row in rows:
            spellings = _spellings(row["value"])
            if wanted in spellings:
                yield table, column, what, row["_row"], row["value"], spellings


def on_file(conn, name: str) -> dict:
    """Everything this library holds about one person (§11.7).

    The question a data subject actually asks, and the one this product could not answer: the
    name is spread across a dozen tables and four JSON blobs, and nothing could gather it.
    """
    wanted = _folded(name)
    if not wanted:
        raise ValueError("`name` is required: whose data this is.")

    where: dict = {}
    spellings: set = set()
    for table, column, what, rowid, value, found in _scan(conn, wanted):
        key = (table, column)
        entry = where.setdefault(key, {"table": table, "column": column, "mentions": 0,
                                       "what": f"{name} {what}", "entries": []})
        entry["mentions"] += 1
        # Art. 15 asks for a COPY of the personal data undergoing processing. "You appear 3
        # times in reactions.said_by" is a receipt for data nobody can see, and a data subject
        # cannot decide whether to ask for erasure without knowing what was said about them.
        if len(entry["entries"]) < _MAX_SHOWN:
            entry["entries"].append(_readable(conn, table, column, rowid, value))
        if isinstance(value, str) and _folded(value) == wanted:
            spellings.add(value.strip())

    total = sum(e["mentions"] for e in where.values())
    if not total:
        return {
            "name": name, "status": "nothing_on_file", "mentions": 0, "where": [],
            "spellings": [],
            "what_it_means": (
                f"This library holds nothing under {name!r}. That is a real answer and not a "
                f"failed search — but a name is not an identifier, so check the spelling it "
                f"would have been entered under before telling anybody their data is not "
                f"here."),
        }
    return {
        "name": name, "status": "on_file", "mentions": total,
        "where": sorted(where.values(), key=lambda e: (e["table"], e["column"])),
        # The exact strings on file, so somebody whose name appears three ways can see all
        # three rather than guessing at a refusal.
        "spellings": sorted(spellings),
        "basis": "computed",
        "what_it_means": (
            f"{total} mention(s) of {name} across {len(where)} place(s). Free-text notes and "
            f"reasons are NOT searched: those are somebody's own sentences and may name "
            f"anyone, so this covers every field this product puts a name IN rather than "
            f"every field a name could appear in."),
    }


def erase(conn, name: str, *, why: str, said_by: str) -> dict:
    """Replace one person's name with a stable token, everywhere (§11.7).

    Irreversible by this product, and it rewrites records that judgments rest on — so it takes
    a reason and a person, like every other consequential write here. See `pseudonym` for what
    "irreversible" does and does not mean.
    """
    import store

    if not (why or "").strip():
        raise ValueError(
            "`why` is required: this rewrites records that saved judgments rest on, and this "
            "product cannot turn it back. An erasure nobody can account for later is the one "
            "write where that cannot be fixed afterwards.")
    said_by = identity.person(said_by, field="said_by")
    wanted = _folded(name)
    if not wanted:
        raise ValueError("`name` is required: whose data this is.")

    hits = list(_scan(conn, wanted))
    if not hits:
        # A name that is PART of somebody else's. Erasing "Vega" used to rewrite the deck
        # author "Ana Vega" into "Ana erased-…" — a second person pseudonymised without being
        # asked, and filed under the first person's token. Matching whole names stops that,
        # and saying WHICH longer names exist is what turns a safe refusal into a useful one:
        # a data subject asking about "Vega" can then say which of them they are.
        near = _longer_names_containing(conn, wanted)
        if near:
            raise ValueError(
                f"{name!r} is not a name on file, but it is part of "
                f"{', '.join(repr(n) for n in near[:3])}"
                + (f" and {len(near) - 3} more" if len(near) > 3 else "")
                + ". Erasing it as given would rewrite somebody else's record and file them "
                  "under this person's token. Give the full name as it appears.")
        raise ValueError(
            f"nothing on file under {name!r}, so there is nothing to erase. A silent success "
            f"here would tell a data subject their name is gone when it is on the record "
            f"under a different spelling.")

    token = pseudonym(conn, name)
    changed = 0
    missed = []
    for table, column, _what, rowid, value, _found in hits:
        replaced = _replaced(value, wanted, token)
        if replaced == value:
            # Every hit CONTAINS the name — that is what made it a hit — so a value that comes
            # back unchanged is not "nothing to do", it is a replace that failed.
            missed.append(f"{table}.{column}")
            continue
        conn.execute(f"UPDATE {table} SET {column} = ? WHERE rowid = ?",   # noqa: S608
                     (replaced, rowid))
        changed += 1
    # Every row that was found has to have been rewritten. `changed` could be less than the
    # number of hits and NOTHING said so — no notice, no counter, no different status — so
    # the observable result of a total failure was `status: pseudonymised` and a paragraph
    # telling a data subject their name is gone. That is the worst sentence this product can
    # produce, and it is the one the code produced when a folding it had not learned yet met
    # a name it could find and could not replace.
    #
    # Rolled back rather than reported-and-kept: a half-rewritten library is worse than an
    # untouched one, because the second attempt now has two spellings to find. The guarantee
    # is: either it is gone, or you are told it is not, and nothing moved.
    if missed:
        conn.rollback()
        raise ValueError(
            f"{name!r} is still on file in {', '.join(sorted(set(missed))[:3])} and could "
            f"not be rewritten, so nothing was changed. Reporting this as done would tell "
            f"somebody their name is gone while it is in the database. Send this message to "
            f"support — it means a spelling this library can find and cannot replace.")
    store.record_erasure(conn, pseudonym=token, why=why.strip(), said_by=said_by,
                         mentions=changed)
    # The old page contents sit in SQLite's free list after an UPDATE, so the name is still in
    # the file's bytes until the pages are reused. Review read it straight out of the raw
    # file after a successful erasure. VACUUM rewrites the database without them.
    conn.execute("VACUUM")
    conn.commit()
    return {
        "status": "pseudonymised", "pseudonym": token, "mentions": changed,
        "basis": "computed", "why": why.strip(), "said_by": said_by,
        "reversible_by": "somebody holding both this database and a list of candidate names",
        "what_it_means": (
            f"{changed} record(s) now read {token} where the name was, and this product "
            f"cannot turn it back — no mapping is kept. It is PSEUDONYMISATION rather than "
            f"anonymisation: the token is derived from the name, so somebody holding this "
            f"database AND a list of candidate names could confirm a match. Under GDPR that "
            f"means these records are still personal data. Say that plainly if somebody asks "
            f"whether their name has been deleted.\n\n"
            f"What it buys is the judgment: the rows are intact, so two findings about this "
            f"person are still about the same someone, where deleting them would leave a "
            f"judgment citing evidence that had vanished."),
    }


def rename(conn, name: str, *, to: str, why: str, said_by: str) -> dict:
    """Correct a misspelled name, everywhere (§11.7, GDPR Art. 16).

    The same sweep as `erase` with a different replacement. Without it the only remedy for a
    typo was the irreversible token — so somebody whose name was entered wrong had to choose
    between a wrong record and no record, which is not a choice rectification is supposed to
    involve.

    It refuses to merge. Renaming one person to a name somebody else already uses would fold
    two voices into one, and §11.5's whole mechanism is two voices being two — a merge here
    would silently turn a disagreement into a single view that nobody holds.
    """
    import store

    if not (why or "").strip():
        raise ValueError("`why` is required: this rewrites records a judgment rests on.")
    said_by = identity.person(said_by, field="said_by")
    corrected = identity.person(to, field="to")
    wanted, replacement = _folded(name), _folded(corrected)
    if not wanted:
        raise ValueError("`name` is required: whose data this is.")
    if wanted == replacement:
        raise ValueError(f"{name!r} and {to!r} are the same name; nothing to correct.")

    hits = list(_scan(conn, wanted))
    if not hits:
        raise ValueError(f"nothing on file under {name!r}, so there is nothing to correct.")
    if list(_scan(conn, replacement)):
        raise ValueError(
            f"{to!r} is already on file as somebody. Renaming {name!r} to it would fold two "
            f"people into one voice — and where they disagreed, that disagreement would "
            f"quietly become a single view nobody holds. Correct it to a spelling nobody "
            f"else uses, or delete the records if they are genuinely the same person.")

    changed = 0
    for table, column, _what, rowid, value, _found in hits:
        replaced = _replaced(value, wanted, corrected)
        if replaced == value:
            continue
        conn.execute(f"UPDATE {table} SET {column} = ? WHERE rowid = ?",   # noqa: S608
                     (replaced, rowid))
        changed += 1
    conn.commit()
    return {
        "status": "corrected", "from": name, "to": corrected, "mentions": changed,
        "why": why.strip(), "said_by": said_by, "basis": "computed",
        "what_it_means": (
            f"{changed} record(s) now read {corrected!r}. Everything they said is still on "
            f"file and still theirs — this changes the spelling, not the record."),
    }


def _longer_names_containing(conn, wanted: str) -> list:
    """Whole names on file that this one is a part of (§11.7).

    So a refusal can say which, rather than leaving a data subject to guess. Read from the
    same scan the view uses, so the two cannot disagree about what is on file.
    """
    import store

    found: set = set()
    for table, column, _what in NAME_COLUMNS + NAME_IN_TEXT + NAME_IN_JSON:
        if not store._columns(conn, table):
            continue
        for row in conn.execute(
                f"SELECT {column} AS value FROM {table} "               # noqa: S608
                f"WHERE {column} IS NOT NULL").fetchall():
            for spelling in _spellings(row["value"]):
                if (spelling != wanted
                        and re.search(rf"(?<!\w){re.escape(wanted)}(?!\w)", spelling)):
                    found.add(spelling)
    return sorted(found)


def _replaced(value, wanted: str, token: str):
    """The stored value with this person's name swapped for the token.

    JSON is decoded, walked and re-encoded rather than pattern-matched as text — that is what
    reaches a name at any depth, and what handles `ensure_ascii` escaping, which no regex over
    the stored string could.
    """
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("{", "[")):
            try:
                return json.dumps(_replaced(json.loads(text), wanted, token))
            except ValueError:
                pass
        if _folded(value) == wanted:
            return token
        return _swap(value, wanted, token)
    if isinstance(value, dict):
        return {k: _replaced(v, wanted, token) for k, v in value.items()}
    if isinstance(value, list):
        return [_replaced(v, wanted, token) for v in value]
    return value


def _swap(value: str, wanted: str, token: str) -> str:
    """Replace the name inside free text or JSON, case-insensitively, whole-word.

    Whole-word because a substring replace turns "Al" into a token inside "Also", and a
    judgment whose prose has been mangled is one nobody can read — which is a different way
    of losing the record than the one this is preventing.
    """
    # On the FOLDED name, so "R.  Vega" and "R. Vega" are one person — the old version folded
    # for the search and escaped the raw input for the replace, so a caller who typed an extra
    # space rewrote the plain columns, silently missed the JSON ones, and was told `erased`.
    pattern = r"\s+".join(re.escape(part) for part in wanted.split())
    # And the haystack is normalised too. `wanted` has been through NFC; the STORED text had
    # not, so a macOS-decomposed "José" was found by `_scan` and not replaced here — the row
    # skipped, the miss uncounted, and the subject told their name was gone. Rewriting the
    # value as NFC is not a change to what it says; it is the same text in the composed form,
    # and only rows being rewritten anyway are touched.
    return re.sub(rf"(?<!\w){pattern}(?!\w)", token,
                  unicodedata.normalize("NFC", value or ""), flags=re.IGNORECASE)


def erasures(conn) -> list[dict]:
    """Every erasure this library has performed (§11.7).

    Without the name. A log that recorded what it removed would be the personal data again, in
    the one table nobody would think to erase.
    """
    import store

    return store.erasures(conn)


# D22: home directories, on every platform this product runs on. `detail` is explicitly
# "send this to support" text, and it interpolates exception messages and asset paths — which
# on Windows embed `C:\\Users\\<name>\\` and on macOS `/Users/<name>/`. A filesystem path
# does not look like personal data until you notice that on two of three platforms it contains
# somebody's name, in the one field this product asks people to email elsewhere.
_HOME_PATHS = (
    re.compile(r"(?i)\b([A-Z]:\\Users\\)([^\\/:*?\"<>|\r\n]+)"),
    re.compile(r"(/(?:Users|home))/([^/\s:]+)"),
)


def redact_paths(text: str) -> str:
    """Replace the account name inside a home directory, keeping the rest of the path (D22).

    The filename is what makes a support message diagnosable, so it stays. Only the segment
    that is a person's login is replaced — a redaction that fired on every path would make
    support text unreadable to solve a problem it does not have.
    """
    out = text or ""
    for pattern in _HOME_PATHS:
        out = pattern.sub(lambda m: f"{m.group(1)}<user>", out)
    return out


def _controller_note() -> str:
    """Who the controller is, and who can read the data (§11.7, C98).

    Reads the deployment rather than asserting the local one, for the same reason
    `_where_it_is` does. The first version said "in a database only you can read" — true of
    the stdio install, and false of the `serve` subcommand this product ships, where `auth`'s
    default provider is `none`, every request is anonymous, and `identity.METHODS` carries
    `unattributed` for exactly that case. An unqualified version of the strongest fact this
    product has is the sentence a data protection officer will quote back.
    """
    import config

    served = bool(getattr(config, "SERVING_HTTP", False))
    return (
        "You are the controller of this data and this software is not a processor of it. It "
        "runs on your machine, against your own files; nothing about these people is sent to "
        "the people who wrote this product, who cannot see your library and have no access "
        "to it. That is the first thing a data protection officer needs to establish, and it "
        "is the strongest fact this product has. "
        + ("This server is currently reachable over the NETWORK, so who can read the library "
           "is whoever can reach this port and satisfy whatever identity provider is "
           "configured — which by default is none, meaning every request is anonymous. Access "
           "control for that deployment is yours to provide and this product does not claim "
           "to."
           if served else
           "It is a database on this machine, readable by anyone with an account on it and "
           "by nobody else."))


def install_disclosure() -> str:
    """The personal-data disclosure shown at install (D26), built from `retention()`.

    Generated rather than written beside it, because a disclosure maintained separately is
    the copy that drifts — and here the drift is between what a customer agreed to and what
    the product actually does, which is the one place this project's recurring
    two-implementations failure becomes a compliance problem rather than a bug.
    """
    position = retention()
    lines = ["This product stores personal data on this machine.", ""]
    for held in position["personal_data"]:
        mark = "  [harvested from your files] " if held.get("harvested") else "  "
        lines.append(f"{mark}{held['what']}")
        lines.append(f"      Why: {held['why']}")
    lines += [
        "",
        f"Where: {position['where_it_is']}",
        f"Kept: {position['retention']}",
        "",
        # D30: not this product's data, which is exactly why it needs saying.
        "It also writes claude_desktop_config.json.bak when it wires itself into Claude "
        "Desktop — a copy of that file as it stood, including the environment of any OTHER "
        "connectors configured there, which may include their credentials. It is rewritten "
        "on every run and never read by this product. Delete it once the install is working "
        "if that file holds anything sensitive.",
        "",
        f"To see what is held about somebody: {position['how_to_see_it']}",
        f"To remove it: {position['how_to_erase_it']}",
        "",
        # C98's controller statement, at the install moment rather than only behind a tool
        # call. It is the first thing a data protection officer needs, and the point of D26
        # is that somebody reads this while they can still decide not to install.
        position["controller_note"],
        "",
        position["lawful_basis_note"],
    ]
    return "\n".join(lines)


def _where_it_is() -> str:
    """Where the data lives, and what leaves the machine (§11.7).

    Reads the configured provider rather than asserting the default. The first version said
    "nothing is sent anywhere except the text you give an embedding provider, which never
    includes these fields" — true of the author COLUMN and false of the words: §2.5 extracts
    each comment into its own chunk and every chunk is embedded, so with a cloud provider the
    client's and colleagues' own sentences leave the machine and the country. For a DPO that
    is the most consequential sentence in the document, and it read as reassurance.
    """
    import config

    provider = (getattr(config, "EMBED_PROVIDER", "") or "").lower()
    local = provider in ("ollama", "hash")
    return (
        "One SQLite file in the data directory, on the machine running the server. "
        + (f"The embedding provider is {provider!r}, which runs locally, so no text leaves "
           f"this machine."
           if local else
           f"The embedding provider is {provider!r}, which is a REMOTE service: the text of "
           f"every brief and every comment extracted from a deck is sent to it to be "
           f"embedded. That includes the comments your clients and colleagues wrote, which is "
           f"a transfer you have to account for. Set CAMPAIGN_POC_EMBED_PROVIDER=ollama to "
           f"keep it local.")
        + " The author names themselves are never sent; the words they wrote are, wherever "
          "the provider runs.")


def retention() -> dict:
    """What this product holds about people, for how long, and what they can do (§11.7).

    The stated position the install disclosure is built from (D26). A product that cannot say
    what it keeps cannot ask anybody to agree to it.
    """
    return {
        "personal_data": [
            {"what": "Names typed by the operator when recording a decision — who said "
                     "something, who confirmed a rule, who withdrew a finding.",
             "harvested": False,
             "why": "A judgment nobody's name is against is one nobody can question later."},
            {"what": "Names harvested from uploaded decks: PDF annotation authors and "
                     "PowerPoint comment authors, stored with the comment they wrote.",
             "harvested": True,
             "why": "A client's objection and the agency's own speaker note carry different "
                    "weight, and the author is the only thing that tells them apart."},
            {"what": "The operating-system account the server runs as, and the machine name.",
             "harvested": True,
             "why": "So a decision records the account that made it, separately from whose "
                    "judgment it was."},
            {"what": "Free-text notes and reasons, which are somebody's own sentences and may "
                     "mention anyone.",
             "harvested": False,
             "why": "These are not searched for names: a guess that missed one would be worse "
                    "than saying plainly that they are user-authored."},
            {"what": "How and when each decision arrived: the channel, the time, and an id "
                     "for the run it belonged to.",
             "harvested": True,
             "why": "So a reader can tell eleven decisions made in one sitting from eleven "
                    "made over a month — the difference between somebody working through a "
                    "queue and somebody agreeing with everything."},
            {"what": "A permanent log of who pseudonymised whom, and why — recording the "
                     "token rather than the name it replaced, and the name of whoever "
                     "performed it.",
             "harvested": False,
             "why": "A record silently rewritten is a record nobody can trust. This is the "
                    "one name `pseudonymise_person` deliberately does NOT remove: erasing it "
                    "would rewrite the log of erasures while it was being written. Delete "
                    "the database to remove it."},
        ],
        "retention": (
            "For as long as the record they belong to. This library is a permanent memory by "
            "design — a judgment is worth something because the evidence behind it is still "
            "there — so nothing here expires on a timer. Deleting a campaign removes the "
            "names attached to it; `pseudonymise_person` replaces one person's name "
            "everywhere at once."),
        "controller_note": _controller_note(),
        "lawful_basis_note": (
            "This product ships with no lawful basis chosen: it runs on the customer's own "
            "machine against their own files, and which basis applies is theirs to decide "
            "with their own counsel. What it guarantees is that the data is enumerable and "
            "erasable on request, which is what makes any basis workable."),
        "how_to_see_it": "Call `person_on_file` with the name.",
        "how_to_erase_it": (
            "Call `pseudonymise_person`. The name is replaced everywhere by a stable token "
            "derived from it, so a judgment that rested on what they said stays explicable. "
            "This product keeps no mapping and cannot reverse it — but the token IS derived "
            "from the name, so somebody holding this database and a list of candidate names "
            "could confirm a match. That makes it pseudonymisation, not anonymisation, and "
            "the records remain personal data under GDPR. If a data subject needs true "
            "erasure, delete the campaigns their words are attached to."),
        "where_it_is": _where_it_is(),
        "basis": "computed",
    }
