"""
Warnings with a shape, one field per reader (§3.1, defect 09).

The reviewer was generous about the engineering and exact about the gap: "The
download-failure warning was precise enough to diagnose from — genuinely good engineering
output. But the intended user is a marketer, for whom 'Failed to download weights for tag
openai' carries no action at all." The asked-for fix was a stable `code`, a one-line human
`remedy`, and the raw `detail` — "so the surface can say 'Visual search is offline — ask IT
to run setup' while the detail stays available for support."

The first version of this file put three readers' text into one `remedy`, which review
caught: a marketer was being read Claude's own stage directions ("Tell the user that, and
offer to..."). One field per reader:

  code       a stable identifier. The part that must survive a reworded message, because
             support tooling and the installer's checks key off it.
  scope      WHO has to act — `machine` (an administrator, once, for everyone), `record`
             (the person who sent this), `call` (nobody; Claude finishes it). This is the
             marketer's actual question, and it is not the same axis as how bad it is.
  severity   how much it costs: `blocked`, `degraded`, `note`.
  affects    the consequence, in the user's terms. What they lose.
  remedy     what a PERSON does about it. Said out loud, verbatim.
  next_step  what CLAUDE does about it. Never read aloud. (§5.2 turns this into
             {tool, prefilled_args}; the string is the draft of that.)
  detail     the original engineering text, unchanged, for support.

Named `notices` rather than `warnings` because the latter is a standard library module, and
shadowing it from the project root would break any import of the real one.
"""
from __future__ import annotations

from typing import Optional

import actions
import people

# How much it costs. Ordered worst-first, and `collapse` sorts by it, so a surface reading
# the list in order leads with the right one — ordering as a property of the response rather
# than an instruction in a docstring that only one tool carried.
SEVERITIES = ("blocked", "degraded", "note")

# Who has to act. The marketer's real question is not "how bad is this" but "is this mine to
# fix, is it IT's, or is it already handled" — and the first version answered it only by
# accident, through severity, which put a blank title in the same class as a missing model.
SCOPES = ("machine", "record", "call")

# code -> (severity, scope, affects, remedy, next_step)
_REGISTRY: dict[str, tuple] = {
    # §7.2. Two embedding models in one index means the similarity numbers are not
    # comparable, so the ranking the whole evidence package rests on is arithmetic across
    # incompatible scales. Degraded rather than blocked: the search still returns records,
    # and they are still roughly the right ones — it is the ORDER that is unsound, which is
    # exactly the kind of quiet wrongness this registry exists to say out loud.
    "mixed_embedding_models": (
        "degraded", "machine",
        "This library was indexed by more than one embedding model, so the ranking of "
        "search results is not reliable — the scores from two models are not on the same "
        "scale. Results are still roughly right; their order is not.",
        "Re-index the library so every record is embedded by the current model.",
        "Say it once per session, not per search. Offer reembed."),
    # ── the review's own example ──
    # The registry supplies no remedy text for the vision model: clip_embed.WeightsResolution
    # already works out the right one per cause (bundled copy absent, configured path wrong,
    # load failed), and the first version discarded that and substituted "ask IT to run
    # setup" — naming a gesture that does not exist, since there is no setup script and
    # nothing in run.sh or run.ps1 fetches the vision weights. The caller passes the
    # computed one in.
    "visual_search_offline": (
        "blocked", "machine",
        "Visual search is off on this machine, so 'find me something that looks like this' "
        "comes back empty. Exact-reuse detection is unaffected and still running, as is "
        "everything to do with text.",
        "Ask whoever installed this to restore the vision model.",
        "Say this once, not per image. Offer finish_indexing after it is fixed — the images "
        "are saved and do not need uploading again.",
    ),
    "text_search_offline": (
        "blocked", "machine",
        "Search is off on this machine, so nothing sent now can be found afterwards. The "
        "records themselves are saved.",
        "Ask whoever installed this to re-run the installer. It starts the text model "
        "service and installs the embedding model, and its self-test confirms both before "
        "it reports success.",
        "Say this once, not per section. Do NOT offer finish_indexing until it is fixed — "
        "it would fail on every item for the same reason.",
    ),
    # ── partial work, recoverable without re-uploading ──
    "indexing_incomplete": (
        "degraded", "call",
        "",     # always supplied per call: it has to carry the counts to mean anything
        "Nothing — this finishes without you.",
        "",     # supplied per call, so it can name the campaign
    ),
    "chunk_not_embedded": (
        "degraded", "call",
        "Part of this deck will not come back in searches.",
        "Nothing — this finishes without you.",
        "Offer finish_indexing on the campaign; no re-upload needed.",
    ),
    # The file a caller named could not be read. Every one of these was a BARE STRING in the
    # warnings list, and `notices.collapse` reads dicts — so an ordinary wrong path crashed the
    # upload tool with `TypeError: string indices must be integers`, which is not a ValueError
    # and so reached the model as "Error executing tool" with the reason discarded.
    "asset_unreadable": (
        "degraded", "record",
        "The file could not be read, so the record was stored without it: {detail}",
        "Check the path or id and send the file again — the record is already on file, so "
        "only the attachment is missing.",
        "",
    ),
    # §9.2: the comparison ran, and some images were not in it. Left silent, the three lists
    # would add up to fewer images than the campaign has and nobody would know which were
    # missing — the totals lying about what was examined.
    "assets_not_fingerprinted": (
        "degraded", "record",
        "{count_phrase} not compared against the brief, because nothing was fingerprinted "
        "for {them}. The comparison covers the rest.",
        "Run finish_indexing — the files are stored, so nothing needs uploading again.",
        "",
    ),
    "image_not_fingerprinted": (
        "degraded", "record",
        # {count} is filled in when several folded together and dropped when it is one:
        # "1 images" is how a count that is always interpolated reads, and "An image" said
        # about six of them is how one that never is reads.
        "{count_phrase} not in exact-reuse detection, so 'have we used this before?' may "
        "answer no when the answer is yes.",
        "If reuse matters for this campaign, send the image again — fingerprinting happens "
        "on the way in.",
        "",
    ),
    # §9.6. `global` plus a market is a contradiction somebody probably did not mean, and
    # keeping the market silently would make the event match one market while reading as
    # global everywhere it is shown. Dropping it is the right call; dropping it without
    # saying so leaves the recorder believing they scoped something they did not.
    "scope_value_ignored": (
        "degraded", "call",
        "{affects}",
        "Record it again with scope `market` if the event was local to one market.",
        "Say it once, on the write. The event IS stored — global — so this is a correction "
        "offer, not a failure."),
    "image_not_embedded": (
        "degraded", "call",
        "{count_phrase} will not come back in 'looks like this' searches.",
        "Nothing — this finishes without you.",
        "Offer finish_indexing; the images are saved and do not need uploading again.",
    ),
    "image_not_stored": (
        "degraded", "record",
        "{count_phrase} could not be saved, so not searchable and not reuse-checked. The "
        "rest of the deck is unaffected.",
        "Nothing, unless that particular image matters — in which case send it on its own.",
        "",
    ),
    "reuse_check_failed": (
        "degraded", "record",
        "This image was stored but never compared against the library, so a reuse of it "
        "would not have been flagged.",
        "Nothing required.",
        "Offer check_image_provenance on it if reuse matters for this campaign.",
    ),
    "images_unreadable": (
        "degraded", "record",
        "The images in this deck could not be read, so none of them are searchable or "
        "reuse-checked. The deck's text is unaffected.",
        "Re-saving the file from PowerPoint, or re-exporting the PDF, usually fixes it.",
        "",
    ),
    "commentary_unreadable": (
        "degraded", "record",
        "Comments and speaker notes in this file could not be read, so they are not "
        "searchable. The deck itself was stored normally.",
        "Nothing required; re-saving the file from PowerPoint usually fixes it.",
        "",
    ),
    # ── caps: the product did what it was asked, up to a limit ──
    "images_capped": (
        # `degraded`, not a note: for a product whose job is reuse detection, "images 21 and
        # after were never checked" is a gap in this record, not a remark.
        "degraded", "record",
        "This deck has more images than are checked automatically, so the later ones were "
        "stored but never compared for reuse.",
        "Nothing required.",
        "Offer check_image_provenance on any specific image that matters.",
    ),
    "commentary_capped": (
        "degraded", "record",
        "This file carries more comments than are indexed automatically, so the later ones "
        "are not searchable.",
        "Nothing required.",
        "",
    ),
    # ── the upload itself had nothing in it ──
    "nothing_to_embed": (
        # NOT `blocked`. The product is working and the record saved; the gap is in what was
        # supplied. Ranking this above everything but a real outage had Claude leading with
        # an IT-flavoured alarm over a typo.
        "degraded", "record",
        "There was nothing to index — no description, and no readable text in the file — so "
        "the record is saved but will not come back in any search.",
        "Add a description, or send a file the text can be read from.",
        "Offer update_campaign with a description the user dictates.",
    ),
    "images_not_extractable": (
        # Distinct from unsupported_file_type on purpose: a PNG sent alongside deck_text is
        # a perfectly good upload whose TEXT is searchable, and the first version told that
        # user "nothing in it is searchable" and advised converting a PNG to PDF.
        "note", "call",
        "This file is not a deck, so it was not searched for embedded images.",
        "Nothing required.",
        "",
    ),
    "unsupported_file_type": (
        "degraded", "record",
        "The text in this file could not be read, so nothing in the file itself is "
        "searchable.",
        "PDF and PowerPoint both work — converting it and sending it again will fix it.",
        "",
    ),
    "legacy_ppt": (
        "degraded", "record",
        "Old .ppt files cannot be read, so the file is stored but nothing in it is "
        "searchable.",
        "Open it in PowerPoint, save it as .pptx, and send that instead.",
        "",
    ),
    "duplicate_title": (
        "note", "record",
        "Another campaign in the library already has this exact title, so anything that "
        "looks a record up by name — a results import, for one — cannot tell them apart.",
        "If they are different campaigns, give one a more specific name; if this is the "
        "same one again, the earlier record can be removed.",
        "",
    ),
    "version_order_unverified": (
        "note", "record",
        "Nothing in the library records which of these two versions came first, so the "
        "order you gave was assumed. If it is the wrong way round, every "
        "\u201cadopted\u201d and \u201craised again\u201d below is inverted.",
        "If one of these replaces the other, say so when uploading it, and the comparison "
        "will not have to assume.",
        "",
    ),
    "results_may_be_incomplete": (
        "note", "call",
        "",     # always supplied per call: it has to carry the counts
        "Nothing — this finishes without you.",
        "Mention it if the answer looks thin, and offer finish_indexing.",
    ),
}

CODES = tuple(_REGISTRY)
_FIELDS = ("severity", "scope", "affects", "remedy", "next_step")


def _entry(code: str) -> tuple:
    if code not in _REGISTRY:
        raise KeyError(f"unknown warning code {code!r}; add it to notices._REGISTRY with "
                       f"text written for somebody who does not read tracebacks")
    return _REGISTRY[code]


def remedy_for(code: str) -> str:
    """The registered remedy. Raises for an unknown code on purpose: a warning with no text
    is one that quietly reverted to being engineer-only, which is the defect."""
    return _entry(code)[_FIELDS.index("remedy")]


def _named_support(remedy: str) -> str:
    """"Ask whoever installed this" — with a name, where the customer has given one (D23).

    That phrase is the best a product shipping to strangers can do, and it is a shrug: the
    person reading it usually cannot act on it. The customer knows who IT is, and §12.2 gives
    them somewhere to say so. Appended rather than substituted, because the rest of the remedy
    is the part that says WHAT to ask for.
    """
    if not remedy or "whoever installed this" not in remedy:
        return remedy
    try:
        import rulebook

        who = rulebook.support()
    except ValueError:
        who = None
    return f"{remedy} (Here, that is: {who}.)" if who else remedy


def notice(code: str, *, detail: str, affects: Optional[str] = None,
           remedy: Optional[str] = None, next_step: Optional[str] = None,
           next_actions: Optional[list] = None, count: int = 1, **extra) -> dict:
    """One warning.

    `affects`, `remedy` and `next_step` can be supplied per call — for the entries that have
    to carry counts or name a campaign to mean anything, and for `visual_search_offline`,
    where the component that failed already knows the right remedy and the registry does not.
    """
    severity, scope, reg_affects, reg_remedy, reg_next = _entry(code)
    entry = {
        "code": code,
        "severity": severity,
        "scope": scope,
        "affects": affects or reg_affects,
        "remedy": _named_support(remedy or reg_remedy),
        # D22/§11.7: `detail` interpolates exception text and asset paths, and this field is
        # explicitly the one this product tells people to send to support. On Windows those
        # paths embed `C:\Users\<name>\` and on macOS `/Users/<name>/` — a filesystem path
        # does not look like personal data until you notice it carries somebody's login, in
        # the one field designed to leave the machine. The filename survives, because that is
        # what makes the message diagnosable.
        "detail": people.redact_paths(detail),
    }
    step = next_step if next_step is not None else reg_next
    if step:
        entry["next_step"] = step
    # §5.2: the same instruction as something the user can accept, rather than prose Claude
    # has to turn into a call. `next_step` stays for the cases with no single tool behind
    # them ("say this once, not per image").
    if next_actions:
        entry["next_actions"] = next_actions
    if count != 1:
        entry["count"] = count
    entry.update(extra)
    return entry


def collapse(entries: list[dict]) -> list[dict]:
    """Fold identical warnings, keep distinct ones, put the worst first.

    A deck with twenty unreadable images produced twenty near-identical lines, which is how
    the one warning that mattered got scrolled past. But folding on the code alone lost real
    information twice over, both found in review: two `indexing_incomplete` entries with
    different text — one about the images, one about the deck's sections — became the image
    one with a count of 2, on a response whose own counts said no section had been indexed;
    and four chunks failing for two different reasons reported one reason and a count of
    four, so the other cause was invisible to the reader `detail` exists for.

    So the fold key is the whole user-facing message, and distinct causes are appended rather
    than dropped. `finish_indexing` already collapsed by reason; this is that trade, made
    once, in one place.
    """
    folded: dict[tuple, dict] = {}
    order: list[tuple] = []
    for entry in entries:
        # The actions are part of the identity. Two warnings that differ only in which
        # campaign they would repair are two warnings; folding them kept the first one's
        # action and a count of two, so one campaign silently lost its offer.
        key = (entry["code"], entry.get("affects"), entry.get("remedy"),
               entry.get("next_step"),
               # `actions.identity`, not a fourth hand-written copy of the same rule: the
               # tupled version here was unhashable for any offer carrying a list argument,
               # which is the crash `trim` had already been fixed for and this had not.
               tuple(sorted(actions.identity(a)
                            for a in entry.get("next_actions", []))))
        if key not in folded:
            kept = dict(entry)
            kept["count"] = entry.get("count", 1)
            kept["_reasons"] = [entry["detail"]]
            folded[key] = kept
            order.append(key)
            continue
        kept = folded[key]
        kept["count"] += entry.get("count", 1)
        if entry["detail"] not in kept["_reasons"]:
            kept["_reasons"].append(entry["detail"])

    result = []
    for key in order:
        entry = folded[key]
        reasons = entry.pop("_reasons")
        entry["affects"] = _phrase(entry.get("affects", ""), entry["count"])
        if len(reasons) > 1:
            entry["detail"] = "; ".join(reasons[:5])
            if len(reasons) > 5:
                entry["detail"] += f"; and {len(reasons) - 5} more"
        if entry["count"] == 1:
            entry.pop("count")
        result.append(entry)

    # Worst first. Ordering belongs to the response: the first version left it to emission
    # order and told Claude, in one tool's docstring, to sort it out mentally.
    result.sort(key=lambda e: SEVERITIES.index(e.get("severity", "note")))
    return result


def _phrase(affects: str, count: int) -> str:
    """Fill `{count_phrase}` once the fold knows how many there were.

    A count that is stored and never spoken makes a folded warning read as though it were
    about one thing — "An image in the deck could not be saved", said about six of them. A
    count that is always spoken reads as "1 images". So the phrase is chosen by the number.
    """
    if "{count_phrase}" not in affects:
        return affects
    phrase = "An image in the deck" if count == 1 else f"{count} images in the deck"
    return affects.replace("{count_phrase}", phrase)


def leading(entries: list[dict]) -> Optional[dict]:
    """The one to say out loud. `collapse` already sorts, so this is the first — kept as a
    named call so callers do not have to know that."""
    return entries[0] if entries else None
