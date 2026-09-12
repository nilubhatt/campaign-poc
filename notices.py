"""
Warnings with a shape, for three audiences at once (§3.1, defect 09).

The reviewer was generous about the engineering and exact about the gap: "The
download-failure warning was precise enough to diagnose from — genuinely good engineering
output. But the intended user is a marketer, for whom 'Failed to download weights for tag
openai' carries no action at all." The asked-for fix was a stable `code`, a one-line human
`remedy`, and the raw `detail` — "so the surface can say 'Visual search is offline — ask IT
to run setup' while the detail stays available for support."

So every warning now serves all three readers without any of them reading the others' text:

  code      a stable identifier. The part that must survive a reworded message, because
            support tooling and the installer's own checks key off it.
  remedy    one line, written for somebody who does not know what a weight is, and phrased
            as what to do rather than what happened.
  detail    the original engineering text, kept rather than replaced.
  severity  whether anybody has to act. "Visual search is offline" and "one image out of
            forty was skipped" are not the same news, and a flat list of strings made every
            surface treat them identically.

Named `notices` rather than `warnings` because the latter is a standard library module, and
shadowing it from the project root would break any import of the real one.
"""
from __future__ import annotations

from typing import Optional

# Whether the user has to do something. `blocked` means a whole capability is unavailable
# until somebody acts; `degraded` means this record is incomplete but the product works;
# `note` is worth saying once.
SEVERITIES = ("blocked", "degraded", "note")

# code -> (severity, remedy). The remedy names the consequence in the user's terms and then
# the action — "X is off, ask IT to do Y" — never the mechanism.
_REGISTRY: dict[str, tuple[str, str]] = {
    # ── the review's own example ──
    "visual_search_offline": (
        "blocked",
        "Visual search is offline, so image similarity and reuse checks will miss this "
        "one. Ask IT to run setup on this machine — everything else works normally.",
    ),
    "text_search_offline": (
        "blocked",
        "Search is offline because the text model is not responding, so nothing uploaded "
        "now will be findable. Ask IT to check that Ollama is running, then re-run "
        "finish_indexing — no re-upload needed.",
    ),
    # ── partial work, recoverable without re-uploading ──
    "indexing_incomplete": (
        "degraded",
        # Always overridden per call: this one has to name the campaign and the counts to be
        # usable as written, and a generic version would be the "call again" advice that
        # sent a user round an infinite loop once already.
        "Part of this upload is not searchable yet. Run finish_indexing on the campaign to "
        "complete it; no re-upload needed.",
    ),
    "chunk_not_embedded": (
        "degraded",
        "Part of this deck will not come back in searches. Run finish_indexing on the "
        "campaign to complete it; no re-upload needed.",
    ),
    "image_not_fingerprinted": (
        "degraded",
        "Reuse detection will miss these images, so 'have we used this before?' may answer "
        "no when the answer is yes. Re-upload the deck if that matters for this campaign.",
    ),
    "image_not_embedded": (
        "degraded",
        "These images will not come back in visual similarity searches. If visual search is "
        "also reported offline, fixing that and re-uploading resolves both.",
    ),
    "image_not_stored": (
        "degraded",
        "An image in the deck could not be saved, so it is not searchable and not "
        "reuse-checked. The rest of the deck is unaffected.",
    ),
    "reuse_check_failed": (
        "degraded",
        "This image was stored but not compared against what is already in the library, so "
        "a reuse of it would not have been flagged. Ask for a provenance check on it "
        "directly if that matters.",
    ),
    "images_unreadable": (
        "degraded",
        "The images in this deck could not be read, so none of them are searchable or "
        "reuse-checked. The deck's text is unaffected. Re-saving the file from PowerPoint "
        "or re-exporting the PDF usually fixes it.",
    ),
    "commentary_unreadable": (
        "degraded",
        "Comments and speaker notes in this file could not be read, so they are not "
        "searchable. The deck itself was stored normally.",
    ),
    "results_may_be_incomplete": (
        "note",
        # Overridden per call to carry the counts, but never blank: a code with an empty
        # remedy is one that quietly reverted to being engineer-only, which is the defect.
        "Some records are only partly indexed, so these results may be incomplete. Offer to "
        "run finish_indexing.",
    ),
    # ── caps: the product did what it was asked, up to a limit ──
    "images_capped": (
        "note",
        "This deck has more images than are checked automatically. The rest were stored "
        "but not compared for reuse — ask for a provenance check on any specific one.",
    ),
    "commentary_capped": (
        "note",
        "This file carries more comments than are indexed automatically; the later ones "
        "are not searchable.",
    ),
    # ── the upload itself had nothing in it ──
    "nothing_to_embed": (
        "blocked",
        "There was nothing to index — no description and no readable text in the file. The "
        "record is saved but will not come back in any search. Add a description, or upload "
        "a file the text can be read from.",
    ),
    "unsupported_file_type": (
        "degraded",
        "This file type cannot be read, so nothing in it is searchable. PDF and PPTX both "
        "work; converting it and re-uploading will fix it.",
    ),
    "legacy_ppt": (
        "degraded",
        "Old .ppt files cannot be read. The file is stored, but nothing in it is "
        "searchable — re-save it as .pptx in PowerPoint and upload that.",
    ),
}

CODES = tuple(_REGISTRY)


def remedy_for(code: str) -> str:
    """The registered remedy. Raises for an unknown code on purpose: a warning with a blank
    remedy is one that quietly reverted to being engineer-only, which is the defect."""
    if code not in _REGISTRY:
        raise KeyError(f"unknown warning code {code!r}; add it to notices._REGISTRY with a "
                       f"remedy written for somebody who does not read tracebacks")
    return _REGISTRY[code][1]


def notice(code: str, *, detail: str, remedy: Optional[str] = None,
           count: int = 1, **extra) -> dict:
    """One warning. `remedy` overrides the registered line for the few that have to name a
    specific campaign or file to be usable as written."""
    severity, registered = _REGISTRY[code] if code in _REGISTRY else _unknown(code)
    entry = {
        "code": code,
        "severity": severity,
        "remedy": remedy or registered,
        "detail": detail,
    }
    if count != 1:
        entry["count"] = count
    entry.update(extra)
    return entry


def _unknown(code: str):
    raise KeyError(f"unknown warning code {code!r}; add it to notices._REGISTRY with a "
                   f"remedy written for somebody who does not read tracebacks")


def collapse(entries: list[dict]) -> list[dict]:
    """Fold repeats of the same code into one line carrying the count.

    A deck with twenty unreadable images produced twenty near-identical lines, which is how
    a genuinely important warning ends up scrolled past. The first occurrence's detail is
    kept — it is the one somebody will actually read — and the count says how far it went.
    """
    folded: dict[str, dict] = {}
    order: list[str] = []
    for entry in entries:
        code = entry["code"]
        if code not in folded:
            folded[code] = dict(entry)
            folded[code].setdefault("count", 0)
            order.append(code)
        folded[code]["count"] += entry.get("count", 1)
    result = []
    for code in order:
        entry = folded[code]
        if entry["count"] == 1:
            entry.pop("count")
        result.append(entry)
    return result


def leading(entries: list[dict]) -> Optional[dict]:
    """The one to say out loud. With several warnings a surface needs to know which leads,
    or it reads them all out — which is the version a marketer ignores."""
    if not entries:
        return None
    return min(entries, key=lambda e: SEVERITIES.index(e.get("severity", "note")))
