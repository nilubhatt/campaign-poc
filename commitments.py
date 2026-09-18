"""
§9.3: the named commitments, checked one by one.

The review: *"The richest signal is not statistical. A brief makes specific, checkable
promises: Colombia v2 commits to a claw machine, a photo booth, a matcha cart and a DJ... Fix:
extract the commitment list at upload — it is mostly already in the deck as an experience or
floorplan list — and at delivery, check each one against the returned photos. 'Claw machine:
present. Photo booth: not visible in 14 delivered images.' That is a post-mortem a marketer can
act on, produced without anyone writing it."*

**"Not visible in 14 delivered images" is the review's own phrasing, and it is the whole
honesty of this item.** Not "absent", not "missing", not "was not delivered". The server can say
what it could not see in the photographs it was given; it cannot say the photo booth was not
there. Fourteen photographs of a launch do not show everything at a launch, and a product that
reports "absent" from their silence has asserted something it has no way to know — about a
supplier who may well have delivered exactly what was promised.

**Two claims, two bases.** That a LINE IS IN THE DECK is a fact: it is quoted, and the quote is
checkable, so `source_line` travels with every commitment. That the line is a COMMITMENT rather
than a passing mention is a reading, and that a photograph shows the thing is a resemblance
score — so the check is `heuristic`, §7.8's third basis, which exists for exactly this.
"""
from __future__ import annotations

import re
from typing import Optional

# A floorplan list is a handful of things. A deck with two hundred bullets is a deck, and
# reading it as a commitment list produces a post-mortem nobody finishes — the same "fires on
# everything" failure this phase has now guarded three times.
MAX_COMMITMENTS = 12
# How much a photograph has to resemble the phrase before the server will say it is there.
#
# TWO tests, and the second is the one doing the work. A single absolute cutoff cannot exist:
# CLIP's objective constrains only RELATIVE order, so the raw cosine scale is arbitrary and
# every phrase has its own floor — measured against the real weights, "Jan 12 — kick-off"
# scored 0.246 against a solid green square while a genuine promise scored 0.231 against noise.
# One number is necessarily wrong for at least one phrase.
#
# So a photograph has to beat the phrase's OWN median across the delivered set by a margin.
# That is per-phrase calibration from data already in hand, and it is what distinguishes "this
# phrase describes this photograph" from "this phrase scores high against everything".
VISIBLE_AT = 0.24
# How far above its own median the best photograph has to sit.
MARGIN = 0.03
# A promise a max-similarity search CANNOT answer, however good the model is.
#
# "Single colourway across all recipients" is the review's own third example, and running it
# through "which photograph most resembles this phrase" is guaranteed to CONFIRM it: more
# colourways means more hero images means a higher maximum. The verdict would read as the
# promise being kept, on exactly the evidence that it was broken. A negative, a universal or a
# count is a claim about the WHOLE SET and about quantities, and this instrument answers "does
# any one image look like this" — so it is refused rather than answered wrongly.
_NOT_ANSWERABLE = re.compile(
    r"\b(single|only|one|no|never|not|all|every|each|exclusively|solely|without|"
    r"at least|at most|minimum|maximum|up to|fewer|more than|"
    r"\d+\s*(x|×|%|mm|cm|m|k|people|influencers|recipients|units|days|hours|guests))\b", re.I)

# Slides that are about the deck rather than about the campaign. A commitment list read off an
# agenda produces "Welcome and introductions: not visible in 14 delivered images", which is a
# post-mortem line about a meeting; read off a team slide it produces one about a person.
_NOT_A_PROMISE_SLIDE = re.compile(
    r"\b(agenda|contents|table of contents|timeline|schedule|next steps|the team|our team|"
    r"who we are|introductions?|thank you|q&a|appendix|objectives?|kpis?|results?|"
    r"hashtags?|index|sources?)\b", re.I)

# Lines that look like a list item. A deck's promises are almost always written as one.
_BULLET = re.compile(
    r"^\s*(?:[-–—*+>‣·•▪▫◦●○➢➤]|\uf0b7|\uf0a7|\d+[.)]|\(\d+\))\s*(.{2,200})$")
# Words that make a line an instruction about the deck rather than a promise in it.
_NOT_A_PROMISE = re.compile(
    r"^(agenda|contents|next steps?|q&a|questions?|thank you|appendix|notes?|sources?|"
    r"timeline|index)\b", re.I)
# A line that is a date, a name and a title, a table row, or a hashtag is not a promise about
# what will be at the event.
_NOT_A_LINE = re.compile(
    r"^(#|\||\d{1,2}[/-]\d{1,2}|"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d)|"
    r"^[A-Z][a-z]+ [A-Z][a-z]+,\s|\|.*\|", re.I)


def _candidate_lines(text: str) -> list:
    """List items in the deck, which is where a brief writes its promises.

    Bullets only, deliberately. A deck is mostly prose, and reading every sentence as a promise
    is the shape this phase keeps finding: a list that fires on everything is one nobody works
    through, and here each false entry becomes a line in a post-mortem accusing somebody.
    """
    found, skipped, heading = [], 0, ""
    for line in (text or "").splitlines():
        match = _BULLET.match(line)
        if not match:
            # Not a list item, so it is the heading whatever follows sits under. An agenda's
            # bullets are an agenda; a team slide's are people. Reading them as promises
            # produces "Welcome and introductions: not visible in 14 delivered images".
            if line.strip():
                heading = line.strip()
            continue
        if _NOT_A_PROMISE_SLIDE.search(heading):
            skipped += 1
            continue
        said = match.group(1).strip().rstrip(".;,")
        if not said or _NOT_A_PROMISE.match(said) or _NOT_A_LINE.match(said):
            skipped += 1
            continue
        if said.lower() in {f["text"].lower() for f in found}:
            continue
        found.append({"text": said, "source_line": line.strip(), "heading": heading})
        if len(found) >= MAX_COMMITMENTS:
            skipped += 1
            break
    return found, skipped


def extract(conn, *, campaign_id: str, text: str) -> list:
    """Record the promises the deck makes, at upload (§9.3).

    Re-runnable. An edited brief promises something different, and leaving the first reading in
    place would check the new deck against the old one's promises — the "learning from the
    wrong document" failure this phase exists to stop, arriving inside the fix for it.

    What a PERSON decided survives: a commitment somebody added by hand stays, and one somebody
    dropped stays dropped rather than reappearing on the next edit. Only the extractor's own
    reading is replaced.
    """
    import store

    previous = {c["text"].lower(): c for c in
                store.commitments_for(conn, campaign_id, include_dropped=True)}
    for row in previous.values():
        if row["origin"] == "extracted" and row["status"] == "open":
            store.forget_commitment(conn, row["id"])

    candidates, _skipped = _candidate_lines(text)
    out = []
    for found in candidates:
        was = previous.get(found["text"].lower())
        if was and was["status"] == "dropped":
            continue                 # somebody said this is not a promise; do not re-add it
        if was and was["origin"] == "added":
            continue                 # already on file, by hand
        out.append(store.insert_commitment(conn, campaign_id=campaign_id, text=found["text"],
                                           source_line=found["source_line"]))
    return out


def summary_for(conn, campaign_id: str) -> dict:
    """What the brief was read as promising, offered for correction at upload (§9.3).

    Extraction is mechanical and will pick up lines that are not promises. Shown here so the
    list can be corrected BEFORE it is used to say something about a supplier — the same shape
    §8.2 uses for an unfamiliar measure: surfaced once, answered once.
    """
    import actions

    found = for_campaign(conn, campaign_id)
    if not found:
        return {}
    return {
        "count": len(found),
        "promises": [{"commitment_id": c["id"], "text": c["text"],
                      "source_line": c["source_line"]} for c in found[:5]],
        "more": max(0, len(found) - 5),
        "what_it_means": (
            f"The brief was read as promising {len(found)} specific "
            f"thing{'s' * (len(found) != 1)}, taken from its own list items. Once the "
            f"photographs come back, each one can be looked for in them — so a line here that "
            f"is not really a promise becomes a line in a post-mortem about somebody. Check "
            f"them now rather than then."),
        "next_actions": actions.trim([
            actions.action("Show everything the brief was read as promising",
                           "list_commitments",
                           why="Extraction is mechanical, and a wrong entry becomes a "
                               "post-mortem line accusing a supplier.",
                           consent="ask", campaign_id=campaign_id),
            actions.action("Add a promise written in a sentence rather than a list",
                           "add_commitment",
                           why="Only list items are read automatically; "
                               "\u201cthirty influencers in identical outfits\u201d is a "
                               "promise in prose.",
                           consent="ask",
                           needs=["text — the promise in the brief's own words",
                                  "source_line — where in the brief it says so"],
                           campaign_id=campaign_id)]),
    }


def for_campaign(conn, campaign_id: str) -> list:
    import store
    return [_public(c) for c in store.commitments_for(conn, campaign_id)]


def describe(conn, commitment_id: str) -> Optional[dict]:
    import store
    row = store.get_commitment(conn, commitment_id)
    return _public(row) if row else None


def _public(row: dict) -> dict:
    # `computed` is a claim about the QUOTE — this line is in the deck — and not about the
    # reading that it is a promise. The source line is what lets somebody disagree with the
    # reading, which is why it is never optional.
    return {**row, "basis": "computed"}


def add(conn, *, campaign_id: str, text: str, source_line: str) -> dict:
    """A promise the extractor did not find. "Thirty influencers in identical outfits" is a
    commitment in a sentence, not a bullet."""
    import store

    if not (text or "").strip():
        raise ValueError("A commitment needs the promise itself, in the brief's own words.")
    if not (source_line or "").strip():
        raise ValueError(
            "`source_line` is required: a commitment has to say where in the brief it comes "
            "from, or a post-mortem resting on it can say only “the library says so”.")
    if store.get_campaign(conn, campaign_id) is None:
        raise ValueError(f"campaign_id {campaign_id!r} is not a record in this library.")
    cid = store.insert_commitment(conn, campaign_id=campaign_id, text=text.strip(),
                                  source_line=source_line.strip(), origin="added")
    return describe(conn, cid)


def drop(conn, commitment_id: str, *, why: Optional[str] = None) -> dict:
    """Not a promise after all. Kept rather than deleted — the record of what the server read
    as a commitment is how somebody later understands why a post-mortem said what it said."""
    import store

    if store.get_commitment(conn, commitment_id) is None:
        raise ValueError(f"{commitment_id!r} is not a commitment on file")
    store.drop_commitment(conn, commitment_id, why=why)
    return describe(conn, commitment_id)


def check(conn, *, campaign_id: str) -> dict:
    """Each named promise against the photographs that came back (§9.3)."""
    import core
    import store

    record = store.get_campaign(conn, campaign_id)
    if record is None:
        raise ValueError(f"{campaign_id!r} is not a record in this library.")

    promises = for_campaign(conn, campaign_id)
    delivered = store.assets_in_phase(conn, campaign_id, "delivered")
    warnings: list = []

    if not promises:
        return _report(campaign_id, record, [], delivered, "none_named",
                       "There is nothing the brief named that the library can look for in a "
                       "photograph — no experience or floorplan list, and nothing added by "
                       "hand. Add the specific promises and this becomes a post-mortem.",
                       warnings)
    if not delivered:
        return _report(campaign_id, record,
                       [_unchecked(p, "Nothing has come back to look in.") for p in promises],
                       delivered, "nothing_to_check",
                       f"{len(promises)} promise{'s' * (len(promises) != 1)} on file and "
                       f"nothing has come back, so none of them has been checked. A campaign "
                       f"whose photographs have not arrived has not failed its commitments.",
                       warnings)

    vectors, missing = core._asset_vectors(conn, delivered)
    if not vectors:
        # §6.4, and the worst possible false negative in this product: an outage that reads as
        # "none of your commitments appeared" ACCUSES A SUPPLIER.
        warnings.append(core._vision_notice(
            "the delivered photographs are not visually indexed, so the brief's promises "
            "could not be looked for in them. Run finish_indexing — the files are stored."))
        return _report(campaign_id, record,
                       [_unchecked(p, "The photographs are not visually indexed.")
                        for p in promises], delivered, "nothing_to_check",
                       "The photographs could not be searched, so nothing is known about any "
                       "of these promises. That is not the same as not finding them.",
                       warnings)
    if missing:
        warnings.append(core._vision_notice(
            f"{len(missing)} of the delivered photographs {'are' if len(missing) != 1 else 'is'}"
            f" not visually indexed, so the promises were looked for in the rest."))

    assets = [a for a in delivered if a["id"] not in missing]

    # §13.6/D126: the photographs' own weights. A phrase encoded by THIS build and images
    # encoded by another are two different spaces, and a cosine across them is a number with no
    # meaning wearing the shape of one that has it. Re-encoding the phrase cannot fix this
    # half — the images are the stale side, and nothing in this product re-embeds them — so the
    # honest answer is that nothing is known, which is not the same as not finding them.
    mine = core.embedding_model_id("asset")
    others = sorted(m for m in asset_weights(conn, among=[a["id"] for a in assets])
                    if m != mine)
    if others:
        warnings.append(core._vision_notice(
            f"the delivered photographs were indexed by {', '.join(others)} and this build "
            f"embeds with {mine}. Similarities between the two are not comparable, so the "
            f"promises were not looked for. Re-index the images (finish_indexing after "
            f"clearing the index) and this becomes answerable again."))
        return _report(campaign_id, record,
                       [_unchecked(p, f"The photographs were indexed by {', '.join(others)} "
                                      f"and this build embeds with {mine}; a similarity "
                                      f"between two models is not a measurement.")
                        for p in promises], delivered, "nothing_to_check",
                       "The photographs and this build do not share a visual space, so "
                       "nothing is known about any of these promises. That is not the same "
                       "as not finding them.", warnings)

    items = [_look_for(conn, promise, assets, vectors) for promise in promises]
    present = [i for i in items if i["verdict"] == "present"]
    looked = [i for i in items if i["verdict"] != "unchecked"]
    unchecked = [i for i in items if i["verdict"] == "unchecked"]

    if not looked:
        # EVERY promise came back unchecked — the phrases could not be embedded at all, or
        # none of them is answerable this way. Reporting "0 of 4 can be seen... the rest were
        # not visible" over that is the supplier-accusing false negative this module's own
        # docstring names, produced by the headline sentence rather than by any verdict.
        if any("could not be embedded" in i["what_it_means"] or "no text/image space"
               in i["what_it_means"] for i in items):
            warnings.append(core._vision_notice(
                "the brief's promises could not be turned into something searchable, so none "
                "of them was looked for. Nothing is known about any of them."))
        return _report(campaign_id, record, items, delivered, "nothing_to_check",
                       f"None of the {len(items)} named promises could be checked — see each "
                       f"one for why. Nothing is known about any of them, which is not the "
                       f"same as not finding them.", warnings)

    said = (f"{len(present)} of {len(looked)} named promises can be seen in the "
            f"{len(assets)} photograph{'s' * (len(assets) != 1)} that came back. The rest "
            f"were not visible in them, which is not the same as not delivered — photographs "
            f"of a launch do not show everything at a launch.")
    if unchecked:
        said += (f" {len(unchecked)} could not be checked this way at all — see each one "
                 f"for why.")
    return _report(campaign_id, record, items, delivered, "checked", said, warnings,
                   searched=len(assets))


def _look_for(conn, promise: dict, assets: list, vectors: list) -> dict:
    """Which delivered photograph most resembles this phrase, and whether that is enough."""
    import clip_embed
    import embedding
    import vectorstore

    try:
        # Only a vector this build's own weights made. A cached phrase from other weights is
        # not a cheaper answer, it is a different space, and a similarity computed across two
        # spaces is a number that means nothing while looking exactly like one that does.
        staged = ({} if _made_by_other_weights(conn, promise["id"]) else
                  vectorstore.get_many(conn, [f"commitment:{promise['id']}"],
                                       space="commitment"))
    except Exception:
        # No commitment vectors have ever been written on this install, so the table does not
        # exist. Reading it raised `OperationalError` — not a `ValueError`, so it would reach
        # the model as "Error executing tool" with the reason discarded.
        staged = {}
    if _NOT_ANSWERABLE.search(promise["text"]):
        return _unchecked(
            promise,
            # §12.4/D125 settles what this is: an accepted limit, not work owed. Naming the
            # INSTRUMENT is what makes it honest — without it a reader cannot tell a boundary
            # of this product from something broken or unfinished, and that difference decides
            # whether they wait for a fix or go and look themselves.
            "this promise is about how MANY or about ALL of something, and looking for the "
            "photograph that most resembles a phrase cannot answer that — for “a "
            "single colourway”, four colourways in frame would make the match STRONGER. "
            "Counting what is in a photograph needs object DETECTION and attribute "
            "extraction, which is a different instrument from the retrieval and perceptual "
            "hashing this product ships; it is not a gap that will close by itself. Somebody "
            "has to look, and the honest report is “not checked” rather than a "
            "verdict.")
    query = staged.get(f"commitment:{promise['id']}")
    if query is None:
        try:
            query = clip_embed.embed_text(promise["text"])
        except Exception:
            return _unchecked(promise, "The phrase could not be embedded.")
        # Kept, so the same phrase is not re-encoded on every check. Written here rather than
        # at extraction because a text-only campaign with no images would otherwise pay a CLIP
        # model load at upload for a check nobody may ever run.
        if any(query):
            try:
                import core

                vectorstore.init(conn, space="commitment")
                # `core._add_vector`, not `vectorstore.add` — "one function so the two cannot
                # drift", and this was the one vector in the product that had drifted. Written
                # raw, it carried no record of WHICH MODEL made it, so after a CLIP weights
                # change a cached phrase was compared against images embedded by different
                # weights: §7.2's "a model upgrade is a visible migration rather than a silent
                # re-ranking", happening silently, in a cache nobody reads (§13.6/D126).
                core._add_vector(conn, f"commitment:{promise['id']}", query,
                                 space="commitment")
            except Exception:
                pass                 # a cache that cannot be written is still just a cache
    if not any(query):
        return _unchecked(promise, "This build has no text/image space to compare in.")

    scored = sorted(((embedding.cosine(query, vec), asset)
                     for asset, vec in zip(assets, vectors)),
                    key=lambda pair: -pair[0])
    best, asset = scored[0]
    median = sorted(score for score, _ in scored)[len(scored) // 2]
    # The closest few, always. The ranking is the thing a person can act on — "go and look at
    # these three" is true whatever the threshold does — and computing it and keeping only the
    # maximum is the same waste §9.2 was pulled up for one item ago.
    closest = [{"asset_id": a["id"], "file": a["file_path"], "similarity": round(sc, 4)}
               for sc, a in scored[:3]]

    if best >= VISIBLE_AT and (len(scored) < 2 or best - median >= MARGIN):
        return {**_base(promise),
                "verdict": "present", "similarity": round(best, 4),
                "seen_in": {"asset_id": asset["id"], "file": asset["file_path"],
                            "captured_on": asset["captured_on"]},
                "closest": closest,
                "what_it_means": (
                    f"{asset['file_path']} is the delivered photograph that most resembles "
                    f"“{promise['text']}”, and it resembles it more than the others do. "
                    f"Look at it to confirm — this is a resemblance between a phrase and an "
                    f"image, not a record that the thing was there.")}
    return {**_base(promise),
            "verdict": "not_visible", "similarity": round(best, 4),
            "closest": closest,
            "what_it_means": (
                f"“{promise['text']}” was not visible in {len(assets)} delivered "
                f"image{'s' * (len(assets) != 1)}. That is what the photographs show, not "
                f"what happened — photographs of a launch do not show everything at a "
                f"launch. The closest {len(closest)} "
                f"{'are' if len(closest) != 1 else 'is'} "
                f"{', '.join(c['file'] for c in closest)}, if you want to look.")}


def stale_vectors(conn, *, only: Optional[str] = None) -> list:
    """Cached phrases made by weights that do not match the images they are scored against.

    §13.6/D126. The reference is the ASSET vectors' model, NOT this build's — and the first
    version of this function got that wrong in the way that mattered. It compared the cached
    phrase against `embedding_model_id`, so after a weights swap it re-encoded the phrase with
    the new model and left the images on the old one: before the "fix", phrase and images were
    both stale and the comparison was at least internally consistent; after it, the comparison
    was guaranteed to be across two spaces. A check can be worse than no check when its
    reference is the wrong thing.

    Rebuildable — re-encoding a phrase costs one CLIP text pass — so a phrase that disagrees
    with the images is simply re-made. A phrase this build cannot make agree, because the
    IMAGES are the stale half, is not something re-encoding can fix, and `check` refuses the
    comparison out loud rather than reporting a number from two spaces.
    """
    import core
    import store

    made = store.vector_models(conn, space="commitment")
    # The images' own model, or this build's when nothing recorded one. A phrase cannot
    # disagree with an index that never said what made it.
    images = asset_weights(conn)
    reference = images.pop() if len(images) == 1 else core.embedding_model_id("asset")
    return sorted(vid for vid, model in made.items()
                  if vid.startswith("commitment:") and model != reference
                  and (only is None or vid == only))


def asset_weights(conn, *, among: Optional[list] = None) -> set:
    """Which models made the image vectors — the space a commitment vector has to agree with.

    ONE implementation, because it was briefly two: `check` computed it inline against
    `embedding_model_id` while this computed it from the recorded rows, in the module whose own
    comment cites two-implementations-of-one-rule as the shape to avoid. They agreed only
    because `check` refused before the other ran.

    Empty when nothing recorded an asset vector, which callers read as "no disagreement": an
    empty library has none, and a library old enough to have no provenance never recorded one
    either way. "Unknown" is not "different" — treating it as different would refuse every
    check on exactly the installs with no way to answer.
    """
    import store

    if not store.provenance_knows_spaces(conn):
        # Every row would answer for every space, so the text embedder would read as the model
        # that made the photographs and every check would refuse.
        return set()
    made = store.vector_models(conn, space="asset")
    return {model for vid, model in made.items() if among is None or vid in set(among)}


def _made_by_other_weights(conn, promise_id: str) -> bool:
    """Whether this promise's cached vector disagrees with the images it would be scored
    against, in which case it is re-encoded rather than read."""
    try:
        return bool(stale_vectors(conn, only=f"commitment:{promise_id}"))
    except Exception:                  # noqa: BLE001 - a cache check is never the answer
        return False


def _unchecked(promise: dict, why: str) -> dict:
    return {**_base(promise), "verdict": "unchecked", "similarity": None,
            "what_it_means": f"Not checked: {why}"}


def _base(promise: dict) -> dict:
    return {"commitment_id": promise["id"], "text": promise["text"],
            "source_line": promise["source_line"],
            # The reading and the resemblance are both inferences. §7.8's third basis.
            "basis": "heuristic"}


_VERDICT_WORD = {"present": "present", "not_visible": "not visible", "unchecked": "not checked"}


def _report(campaign_id, record, items, delivered, status, said, warnings,
            *, searched=None) -> dict:
    import notices

    return {
        "campaign_id": campaign_id,
        "title": record["title"],
        "status": status,
        "basis": "heuristic",
        "delivered_count": len(delivered),
        "counts": {verdict: sum(1 for i in items if i["verdict"] == verdict)
                   for verdict in ("present", "not_visible", "unchecked")},
        "items": items,
        # The review's own shape: one line per promise, in the brief's own words.
        # The count that was actually SEARCHED, not the count on file. With one of three
        # photographs unindexed the summary said "not visible in 3 delivered images" while the
        # item said 2 — and the summary is the line the review calls the actionable one.
        "searched": searched if searched is not None else 0,
        "summary": [f"{i['text']}: {_VERDICT_WORD[i['verdict']]}"
                    + (f" in {searched or 0} delivered image"
                       f"{'s' * ((searched or 0) != 1)}"
                       if i["verdict"] == "not_visible" else "")
                    for i in items],
        "what_it_means": said,
        # Every branch was a dead end the model had to navigate from memory: `none_named` told
        # the user to add promises and did not offer the tool, and a verdict on a line that was
        # never a promise did not offer to drop it.
        "next_actions": _offers(campaign_id, status, items),
        "warnings": notices.collapse(warnings),
    }


def _offers(campaign_id: str, status: str, items: list) -> list:
    import actions

    if status == "none_named":
        return actions.trim([actions.action(
            "Record what this brief promised", "add_commitment",
            why="Nothing in the brief was written as a list, so there is nothing to look for "
                "in the photographs yet.",
            consent="ask",
            needs=["text — the promise in the brief's own words",
                   "source_line — where in the brief it says so"],
            campaign_id=campaign_id)])
    wrong = next((i for i in items if i["verdict"] == "not_visible"), None)
    if wrong:
        return actions.trim([actions.action(
            f"\u201c{wrong['text'][:50]}\u201d was not a promise at all", "drop_commitment",
            why="Extraction is mechanical. A line that is not a deliverable should not be "
                "reported as one that did not appear.",
            consent="ask", commitment_id=wrong["commitment_id"])])
    return []
