"""PDF / PPTX text extraction — the only file processing the POC needs."""
from __future__ import annotations

import mimetypes
from pathlib import Path

import config
import notices


def guess_mime(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


def extract_units(path: Path, mime: str | None = None) -> tuple[list[str], list[str]]:
    """
    Return (units, warnings) — one text unit per PDF page / PPTX slide, the natural chunk
    boundaries chunking.pack() uses for search (§6.1). Legacy .ppt is accepted upstream but
    yields no units (no reliable pure-Python extractor).
    """
    mime = mime or guess_mime(path.name)
    if mime == "application/pdf":
        return _read_pdf(path), []
    if mime == config.PPTX_MIME:
        return _read_pptx(path), []
    if mime == config.PPT_LEGACY_MIME:
        return [], [notices.notice(
            "legacy_ppt",
            detail="legacy .ppt stored but not text-extracted; convert to .pptx for "
                   "searchability")]
    return [], [notices.notice(
        "unsupported_file_type",
        detail=f"unsupported type {mime}; POC accepts PDF and PPTX")]


def extract_images(path: Path, mime: str | None = None) -> tuple[list[tuple[bytes, str, int]], list[str]]:
    """
    Return ([(image_bytes, extension, slide_or_page_number), ...], warnings) — images embedded
    in a PDF/PPTX. slide_or_page_number is 1-based (which slide/page a flagged image came
    from — needed to tell the user which one, not just that "an" image was reused). Identical
    images (by content, e.g. a logo pasted on every slide) are deduped to their first
    occurrence BEFORE the config.MAX_EXTRACTED_IMAGES_PER_DECK cap is applied, so a repeated
    image can't starve the cap and hide a later, distinct one. A single corrupt/unsupported
    embedded image is skipped (with a warning) rather than aborting extraction for the rest of
    the deck. Only works when the actual file reaches the server (asset_ref) — the LLM-first
    deck_text-only path never gives us bytes to extract images from, an inherent limit, not a
    bug. Extension includes the leading dot.
    """
    mime = mime or guess_mime(path.name)
    if mime == "application/pdf":
        return _read_pdf_images(path)
    if mime == config.PPTX_MIME:
        return _read_pptx_images(path)
    return [], [notices.notice(
        "unsupported_file_type", detail=f"image extraction not supported for {mime}")]


def _dedup_and_cap(
    raw: list[tuple[bytes, str, int]], warnings: list[str]
) -> list[tuple[bytes, str, int]]:
    import hashlib

    seen: set[bytes] = set()
    unique: list[tuple[bytes, str, int]] = []
    for data, ext, location in raw:
        digest = hashlib.sha256(data).digest()
        if digest in seen:
            continue
        seen.add(digest)
        unique.append((data, ext, location))

    if len(unique) > config.MAX_EXTRACTED_IMAGES_PER_DECK:
        warnings.append(notices.notice(
            "images_capped",
            detail=f"deck has more than {config.MAX_EXTRACTED_IMAGES_PER_DECK} distinct "
                   f"embedded images; only the first "
                   f"{config.MAX_EXTRACTED_IMAGES_PER_DECK} were checked"))
    return unique[: config.MAX_EXTRACTED_IMAGES_PER_DECK]


def _read_pdf_images(path: Path) -> tuple[list[tuple[bytes, str, int]], list[str]]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    warnings: list[str] = []
    raw: list[tuple[bytes, str, int]] = []
    for i, page in enumerate(reader.pages):
        if i >= config.MAX_PDF_PAGES:
            break
        page_num = i + 1
        try:
            n = len(page.images)
        except Exception as exc:
            warnings.append(notices.notice(
                "images_unreadable",
                detail=f"page {page_num}: could not read embedded images ({exc}); skipped"))
            continue
        for j in range(n):
            try:
                img = page.images[j]
                ext = Path(img.name).suffix or ".png"
                raw.append((img.data, ext, page_num))
            except Exception as exc:
                warnings.append(notices.notice(
                    "images_unreadable",
                    detail=f"page {page_num} image {j + 1}: could not be read ({exc}); "
                           f"skipped"))
    return _dedup_and_cap(raw, warnings), warnings


def _read_pptx_images(path: Path) -> tuple[list[tuple[bytes, str, int]], list[str]]:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    from pptx.shapes.picture import Picture
    prs = Presentation(str(path))
    warnings: list[str] = []
    raw: list[tuple[bytes, str, int]] = []

    def walk(shapes, slide_num: int) -> None:
        for shape in shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                walk(shape.shapes, slide_num)
            elif isinstance(shape, Picture):
                # Picture covers both a plain inserted picture (shape_type PICTURE) and a
                # picture-filled template placeholder (shape_type PLACEHOLDER, common in
                # corporate decks built from a layout) — both expose .image.
                try:
                    raw.append((shape.image.blob, f".{shape.image.ext}", slide_num))
                except Exception as exc:
                    warnings.append(notices.notice(
                        "images_unreadable",
                        detail=f"slide {slide_num}: image could not be read ({exc}); "
                               f"skipped"))

    for i, slide in enumerate(prs.slides):
        if i >= config.MAX_PPTX_SLIDES:
            break
        walk(slide.shapes, i + 1)

    return _dedup_and_cap(raw, warnings), warnings


def _read_pdf(path: Path) -> list[str]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    units: list[str] = []
    total = 0
    for i, page in enumerate(reader.pages):
        if i >= config.MAX_PDF_PAGES or total >= config.MAX_DOC_TEXT_CHARS:
            break
        text = (page.extract_text() or "").strip()
        if text:
            units.append(text)
            total += len(text)
    return units


def _read_pptx(path: Path) -> list[str]:
    from pptx import Presentation
    prs = Presentation(str(path))
    units: list[str] = []
    total = 0
    for i, slide in enumerate(prs.slides):
        if i >= config.MAX_PPTX_SLIDES or total >= config.MAX_DOC_TEXT_CHARS:
            break
        parts = [shape.text_frame.text for shape in slide.shapes
                 if shape.has_text_frame and shape.text_frame.text]
        text = "\n".join(parts).strip()
        if text:
            units.append(text)
            total += len(text)
    return units


# ── commentary: comments, annotations and speaker notes (§2.5, defect 08) ────
#
# A distinct layer from the deck body, deliberately. The review was explicit about why:
# commentary "is usually the internal reaction to the work rather than the work itself", so
# concatenating it into deck_text makes a reviewer's objection read back as something the
# brief itself claimed. It carries an author, a date and a page or slide anchor; the body
# carries none of those.

# Subtypes that hold somebody's words. /Link is excluded on purpose: the reviewer found that
# every other deck in the library carries /Link annotations only, and a hyperlink is not an
# opinion — indexing them would put URL fragments into the evidence a judgment cites.
_COMMENT_SUBTYPES = {"/Text", "/FreeText", "/Highlight", "/StrikeOut", "/Underline",
                     "/Square", "/Caret", "/Ink"}


def extract_commentary(path: Path, mime: str | None = None) -> tuple[list[dict], list[str]]:
    """Return ([{kind, text, author, date, page|slide, anchor}, ...], warnings).

    Never raises: commentary is the bonus layer, and a malformed comments part must not cost
    the user the deck they actually uploaded.
    """
    mime = mime or guess_mime(path.name)
    warnings: list[str] = []
    try:
        if mime == "application/pdf":
            items = _read_pdf_commentary(path, warnings)
        elif mime == config.PPTX_MIME:
            items = _read_pptx_commentary(path, warnings)
        else:
            return [], []
    except Exception as exc:                      # noqa: BLE001 - see docstring
        return [], [notices.notice(
            "commentary_unreadable",
            detail=f"comments and notes could not be read ({exc}); the deck itself was "
                   f"ingested normally")]

    if len(items) > config.MAX_COMMENTARY_ITEMS:
        warnings.append(notices.notice(
            "commentary_capped",
            detail=f"deck carries more than {config.MAX_COMMENTARY_ITEMS} comments and "
                   f"notes; only the first {config.MAX_COMMENTARY_ITEMS} were indexed"))
        items = items[:config.MAX_COMMENTARY_ITEMS]
    return items, warnings


def _pdf_date(raw) -> str | None:
    """PDF /M is `D:YYYYMMDDHHmmSS±HH'mm'`. Returned ISO-ish so it sorts and reads the same
    way as every other date in the library.

    Anything that is not a real date is kept VERBATIM rather than reformatted: the first
    version built the string before validating it, so `D:2026` became "2026--" and
    `D:99999999` became "9999-99-99" — a date no calendar contains, in a field a reader
    would take at face value. An unparseable value is still evidence of when somebody said
    something; an invented one is worse than none.
    """
    import datetime

    if not raw:
        return None
    text = str(raw).strip()
    digits = text[2:] if text.startswith("D:") else text
    if len(digits) < 8 or not digits[:8].isdigit():
        return text
    try:
        date = datetime.date(int(digits[0:4]), int(digits[4:6]), int(digits[6:8]))
    except ValueError:
        return text
    stamp = date.isoformat()
    if len(digits) >= 14 and digits[8:14].isdigit():
        try:
            time = datetime.time(int(digits[8:10]), int(digits[10:12]), int(digits[12:14]))
        except ValueError:
            return stamp
        stamp += f"T{time.isoformat()}"
    return stamp


def _read_pdf_commentary(path: Path, warnings: list[str]) -> list[dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    items: list[dict] = []
    for i, page in enumerate(reader.pages):
        if i >= config.MAX_PDF_PAGES:
            break
        try:
            annots = page.get("/Annots") or []
        except Exception as exc:                  # noqa: BLE001
            warnings.append(notices.notice(
                "commentary_unreadable",
                detail=f"page {i + 1}: comments could not be read ({exc}); skipped"))
            continue
        for ref in annots:
            try:
                annot = ref.get_object()
                if annot.get("/Subtype") not in _COMMENT_SUBTYPES:
                    continue
                # PDF uses a bare CR for a line break, which renders as one run-on line in
                # any excerpt a person reads.
                text = str(annot.get("/Contents") or "").replace("\r\n", "\n")
                text = text.replace("\r", "\n").strip()
                if not text:
                    continue
                items.append({
                    "kind": "annotation",
                    "text": text,
                    "author": str(annot["/T"]) if annot.get("/T") else None,
                    "date": _pdf_date(annot.get("/M")),
                    "page": i + 1,
                    "anchor": f"page {i + 1}",
                })
            except Exception as exc:              # noqa: BLE001
                warnings.append(notices.notice(
                    "commentary_unreadable",
                    detail=f"page {i + 1}: a comment could not be read ({exc}); skipped"))
    return items


def _read_pptx_commentary(path: Path, warnings: list[str]) -> list[dict]:
    items = _read_pptx_notes(path, warnings)
    items.extend(_read_pptx_comments(path, warnings))
    return items


def _read_pptx_notes(path: Path, warnings: list[str]) -> list[dict]:
    from pptx import Presentation

    prs = Presentation(str(path))
    items: list[dict] = []
    for i, slide in enumerate(prs.slides):
        if i >= config.MAX_PPTX_SLIDES:
            break
        try:
            # PowerPoint creates a notesSlide as soon as anything touches the slide, so
            # `has_notes_slide` is true far more often than there is a note — the common
            # case is an empty one, and a library of blank commentary rows dilutes every
            # retrieval it appears in.
            if not slide.has_notes_slide:
                continue
            text = (slide.notes_slide.notes_text_frame.text or "").strip()
            if not text:
                continue
            items.append({
                "kind": "speaker_note",
                "text": text,
                "author": None,          # a notesSlide records no author
                "date": None,
                "slide": i + 1,
                "anchor": f"slide {i + 1}",
            })
        except Exception as exc:                  # noqa: BLE001
            warnings.append(notices.notice(
                "commentary_unreadable",
                detail=f"slide {i + 1}: speaker notes could not be read ({exc}); skipped"))
    return items


def _read_pptx_comments(path: Path, warnings: list[str]) -> list[dict]:
    """Reviewer comments, read straight out of the package: python-pptx has no API for them.

    Two formats, because PowerPoint changed it. `ppt/comments/` is the classic one, keyed to
    an author list in `ppt/commentAuthors.xml`; `ppt/modernComments/` is what current
    PowerPoint writes, keyed to `ppt/authors.xml`. A deck can carry either, and a deck that
    has been round-tripped can carry both — which is the deck that matters here, since "a
    partner deck returned with tracked client comments is the feedback the library exists to
    remember".
    """
    import xml.etree.ElementTree as ET
    import zipfile

    items: list[dict] = []
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        authors = _pptx_authors(z, names, warnings)
        for name in sorted(n for n in names
                           if n.startswith(("ppt/comments/", "ppt/modernComments/"))
                           and n.endswith(".xml")):
            try:
                root = ET.fromstring(z.read(name))
            except ET.ParseError as exc:
                warnings.append(notices.notice(
                    "commentary_unreadable",
                    detail=f"{name}: comments could not be parsed ({exc}); skipped"))
                continue
            for node in root:
                items.extend(_comment_and_replies(node, authors, name))
    return items


def _comment_and_replies(node, authors: dict[str, str], part: str,
                         parent_id: str | None = None) -> list[dict]:
    """One item per utterance, never one per thread.

    A tracked thread on a returned deck is typically a client's remark and the agency's
    answer, and PowerPoint nests the replies inside the parent comment. Joining every run of
    text under the parent recorded the agency's response as the client's words — which is
    defect 08's own misattribution, one level down, inside the exact feature that exists to
    fix it.
    """
    items: list[dict] = []
    ident = node.get("id")
    text = _own_text(node)
    if text:
        items.append({
            "kind": "comment",
            "text": text,
            "author": authors.get(node.get("authorId")),
            "date": (node.get("created") or node.get("dt") or "").strip() or None,
            "slide": None,
            "anchor": _comment_anchor(part),
            **({"reply_to": parent_id} if parent_id else {}),
        })
    for child in node.iter():
        if child is node or not child.tag.endswith("}reply"):
            continue
        items.extend(_comment_and_replies(child, authors, part, parent_id=ident))
    return items


def _own_text(node) -> str:
    """This utterance's words only — the runs under it, minus anything belonging to a nested
    reply. Both element names are handled: modernComments carries drawingml `<a:t>` runs,
    while the classic `ppt/comments/` format uses a single `<p:text>`, which matched nothing
    and silently dropped every comment in the older format."""
    reply_descendants = {id(d) for child in node.iter()
                         if child is not node and child.tag.endswith("}reply")
                         for d in child.iter()}
    parts = []
    for element in node.iter():
        if id(element) in reply_descendants:
            continue
        if element.tag.endswith("}t") or element.tag.endswith("}text"):
            if element.text:
                parts.append(element.text)
    return " ".join(parts).strip()


def _pptx_authors(z, names: set[str], warnings: list[str]) -> dict[str, str]:
    import xml.etree.ElementTree as ET

    authors: dict[str, str] = {}
    for part in ("ppt/authors.xml", "ppt/commentAuthors.xml"):
        if part not in names:
            continue
        try:
            root = ET.fromstring(z.read(part))
        except ET.ParseError as exc:
            warnings.append(notices.notice(
                "commentary_unreadable",
                detail=f"{part}: comment authors could not be parsed ({exc}); the comments "
                       f"were kept without a name"))
            continue
        for node in root:
            ident, name = node.get("id"), node.get("name")
            if ident and name:
                authors[ident] = name
    return authors


def _comment_anchor(name: str) -> str:
    """"deck", not a guessed slide.

    A PDF page index is exact. The number in `commentN.xml` is the comment PART's ordinal,
    not the slide's — and it was being written into the same `anchor` field, read with the
    same confidence as the exact one. An anchor that is sometimes silently wrong is worse
    than one that admits it does not know: resolving a comment part to its slide needs the
    package relationships, which is work item 2.5 does not need to do to be correct.
    """
    return "deck"
