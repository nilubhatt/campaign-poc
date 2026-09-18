"""PDF / PPTX text extraction — the only file processing the POC needs."""
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Optional

import config
import notices


def guess_mime(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


def extract_units(path: Path, mime: str | None = None) -> tuple[list[str], list[dict]]:
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


def extract_images(path: Path, mime: str | None = None) -> tuple[list[tuple[bytes, str, int]], list[dict]]:
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
    # NOT unsupported_file_type: a PNG sent alongside deck_text is a perfectly good upload
    # whose text is searchable, and saying "nothing in it is searchable" about it — then
    # advising the user to convert a PNG to PDF — was a regression the old engineering
    # string did not have.
    return [], [notices.notice(
        "images_not_extractable", detail=f"image extraction not supported for {mime}")]


def _dedup_and_cap(
    raw: list[tuple[bytes, str, int]], warnings: list[dict]
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


def _read_pdf_images(path: Path) -> tuple[list[tuple[bytes, str, int]], list[dict]]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    warnings: list[dict] = []
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


def _read_pptx_images(path: Path) -> tuple[list[tuple[bytes, str, int]], list[dict]]:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    from pptx.shapes.picture import Picture
    prs = Presentation(str(path))
    warnings: list[dict] = []
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
        parts = [_frame_text(shape) for shape in slide.shapes
                 if shape.has_text_frame and shape.text_frame.text]
        text = "\n".join(parts).strip()
        if text:
            units.append(text)
            total += len(text)
    return units


_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


# Placeholder kinds whose paragraphs ARE a bulleted list unless a paragraph says otherwise.
# This is what PowerPoint does: a body or content placeholder inherits its bullet from the
# layout and master, so the bullet is nowhere in the slide's own XML.
_LIST_PLACEHOLDERS = {2, 3, 7, 14}     # BODY, CENTER_TITLE-adjacent body, OBJECT, VERTICAL_BODY


def _is_list_paragraph(paragraph, *, inherits_bullets: bool) -> bool:
    """Whether this paragraph is a bullet or a numbered item.

    In OOXML a bullet is PARAGRAPH FORMATTING — `<a:buChar>` or `<a:buAutoNum>` — and never a
    character in a run, so `text_frame.text` returns the words with no glyph in front of them.
    §9.3 reads a deck's experience or floorplan list to find what a brief promised, and against
    a real PPTX it found nothing at all: the list was there, the list-ness was not.

    And usually the bullet is not in the slide either — a body placeholder inherits it from the
    layout, so checking only for an explicit `buChar` still finds nothing on an ordinary deck.
    `inherits_bullets` is that case. An explicit `buNone` overrides it, which is how a
    deliberately unbulleted paragraph says so.
    """
    properties = paragraph._p.find(f"{_NS}pPr")
    if properties is not None:
        if properties.find(f"{_NS}buNone") is not None:
            return False
        if (properties.find(f"{_NS}buChar") is not None
                or properties.find(f"{_NS}buAutoNum") is not None):
            return True
        if properties.get("lvl") not in (None, "0"):
            return True
    return inherits_bullets


def _frame_text(shape) -> str:
    """A shape's words, with its list structure preserved as "- " prefixes.

    The marker is added rather than the original glyph because the glyph is not in the file —
    it is drawn from the formatting, and Office writes Symbol-font code points (\uf0b7,
    \uf0a7) that mean nothing outside the font.
    """
    paragraphs = [p for p in shape.text_frame.paragraphs if (p.text or "").strip()]
    inherits = False
    try:
        # TWO or more paragraphs. A body placeholder inherits a bullet from the layout whatever
        # is in it, so a single-paragraph placeholder would be marked as a list — and the
        # marker goes into the text this library stores, quotes against and searches. One line
        # under a heading is a statement; three are a list.
        if shape.is_placeholder and len(paragraphs) > 1:
            inherits = int(shape.placeholder_format.type) in _LIST_PLACEHOLDERS
    except Exception:                  # noqa: BLE001 — a shape with no usable placeholder info
        inherits = False
    lines = []
    for paragraph in paragraphs:
        said = paragraph.text.strip()
        lines.append(f"- {said}" if _is_list_paragraph(paragraph, inherits_bullets=inherits)
                     else said)
    return "\n".join(lines)


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


def extract_commentary(path: Path, mime: str | None = None) -> tuple[list[dict], list[dict]]:
    """Return ([{kind, text, author, date, page|slide, anchor}, ...], warnings).

    Never raises: commentary is the bonus layer, and a malformed comments part must not cost
    the user the deck they actually uploaded.
    """
    mime = mime or guess_mime(path.name)
    warnings: list[dict] = []
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


def _read_pdf_commentary(path: Path, warnings: list[dict]) -> list[dict]:
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


def _read_pptx_commentary(path: Path, warnings: list[dict]) -> list[dict]:
    items = _read_pptx_notes(path, warnings)
    items.extend(_read_pptx_comments(path, warnings))
    return items


def _read_pptx_notes(path: Path, warnings: list[dict]) -> list[dict]:
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


def _read_pptx_comments(path: Path, warnings: list[dict]) -> list[dict]:
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
        on_slide = _pptx_comment_slides(z, names)
        for name in sorted(n for n in names
                           if n.startswith(tuple(_COMMENT_PARTS.values()))
                           and n.endswith(".xml")):
            try:
                root = ET.fromstring(z.read(name))
            except ET.ParseError as exc:
                warnings.append(notices.notice(
                    "commentary_unreadable",
                    detail=f"{name}: comments could not be parsed ({exc}); skipped"))
                continue
            for node in root:
                items.extend(_comment_and_replies(node, authors, name,
                                                  slide=on_slide.get(name)))
    return items


def _comment_and_replies(node, authors: dict[str, dict[str, str]], part: str,
                         parent_id: str | None = None,
                         slide: Optional[int] = None) -> list[dict]:
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
            "author": authors.get(_comment_format(part), {}).get(node.get("authorId")),
            "date": (node.get("created") or node.get("dt") or "").strip() or None,
            "slide": slide,
            "anchor": _comment_anchor(part, slide),
            **({"reply_to": parent_id} if parent_id else {}),
        })
    for child in node.iter():
        if child is node or not child.tag.endswith("}reply"):
            continue
        items.extend(_comment_and_replies(child, authors, part, parent_id=ident,
                                          slide=slide))
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


# Which author list belongs to which comment format. PowerPoint changed the format, and a
# deck round-tripped through both versions carries both parts — with author ids that are
# LOCAL TO EACH. Merged into one dictionary keyed on the raw id, a `1` in one namespace
# silently overwrote the `1` in the other, and a client's words came back under whoever
# happened to share their number. Who said something decides whose rule enters the checklist
# and whose name a judgment cites, so this is not a cosmetic mix-up.
_AUTHOR_PARTS = {"modern": "ppt/authors.xml", "classic": "ppt/commentAuthors.xml"}
# The comment parts, by the same two names, so nothing has to re-derive the pairing. ONE
# mapping: a second place deciding which authors a part uses is the shape this codebase keeps
# being bitten by.
_COMMENT_PARTS = {"modern": "ppt/modernComments/", "classic": "ppt/comments/"}


def _comment_format(part: str) -> str:
    """Which format a comment part is in, by the folder PowerPoint puts it in."""
    for fmt, prefix in _COMMENT_PARTS.items():
        if part.startswith(prefix):
            return fmt
    return "classic"


def _pptx_authors(z, names: set[str], warnings: list[dict]) -> dict[str, dict[str, str]]:
    """`{format: {author_id: name}}` — one namespace per format, never merged."""
    import xml.etree.ElementTree as ET

    authors: dict[str, dict[str, str]] = {fmt: {} for fmt in _AUTHOR_PARTS}
    for fmt, part in _AUTHOR_PARTS.items():
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
                authors[fmt][ident] = name
    return authors


_RELS_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_OFFICE_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PML_NS = "{http://schemas.openxmlformats.org/presentationml/2006/main}"


def _resolve(base: str, target: str) -> str:
    """A relationship Target, resolved against the part that declared it.

    Targets are relative — `../comments/comment1.xml` from inside `ppt/slides/` — so they have
    to be walked rather than concatenated, or the key never matches the part name.
    """
    parts = base.split("/")[:-1]
    for step in target.replace("\\", "/").split("/"):
        if step in ("", "."):
            continue
        if step == "..":
            if parts:
                parts.pop()
        else:
            parts.append(step)
    return "/".join(parts)


def _slide_order(z, names: set[str]) -> list[str]:
    """The slide parts in the order a reader sees them, which is what "slide 3" means.

    `slide11.xml` is not necessarily the eleventh slide: the file number is an id, and the
    ORDER lives in `presentation.xml`'s `sldIdLst` resolved through the presentation's own
    relationships. Sorting the filenames instead would be the same class of guess this
    function exists to replace, just a tidier-looking one.
    """
    import xml.etree.ElementTree as ET

    rels_part = "ppt/_rels/presentation.xml.rels"
    if "ppt/presentation.xml" not in names or rels_part not in names:
        return []
    try:
        rels = {node.get("Id"): node.get("Target")
                for node in ET.fromstring(z.read(rels_part))}
        root = ET.fromstring(z.read("ppt/presentation.xml"))
    except ET.ParseError:
        return []
    ordered = []
    for lst in root.iter(f"{_PML_NS}sldIdLst"):
        for node in lst:
            target = rels.get(node.get(f"{_OFFICE_REL}id"))
            if target:
                ordered.append(_resolve("ppt/presentation.xml", target))
    return ordered


def _pptx_comment_slides(z, names: set[str]) -> dict[str, int]:
    """`{comment part: slide number}` — resolved through the package, never scraped.

    §2.5 recorded every comment as `slide: None`, `anchor: "deck"`, and the reasoning was
    sound as far as it went: the number in `commentN.xml` is the comment PART's ordinal, and
    presenting it as a slide number would be a guess wearing the same clothes as the exact
    page index a PDF gives. What it stopped short of was doing the resolution properly — the
    slide that OWNS a comment part is written down, in that slide's own relationships.

    Unresolved parts are simply absent, and the caller falls back to "deck". A comment whose
    slide genuinely cannot be determined still says so rather than naming one.
    """
    import xml.etree.ElementTree as ET

    owners: dict[str, int] = {}
    for position, slide_part in enumerate(_slide_order(z, names), start=1):
        folder, _, filename = slide_part.rpartition("/")
        rels_part = f"{folder}/_rels/{filename}.rels"
        if rels_part not in names:
            continue
        try:
            rels = ET.fromstring(z.read(rels_part))
        except ET.ParseError:
            continue
        for node in rels:
            target = node.get("Target") or ""
            resolved = _resolve(slide_part, target)
            if resolved.startswith(tuple(_COMMENT_PARTS.values())):
                owners[resolved] = position
    return owners


def _comment_anchor(name: str, slide: Optional[int]) -> str:
    """The slide it is on, once the package has been asked — otherwise "deck".

    A PDF page index is exact, and this now is too: it comes from the slide's own
    relationships rather than from the comment part's filename. Where the relationship is
    missing or unreadable the answer stays "deck", because an anchor that is sometimes
    silently wrong is worse than one that admits it does not know.
    """
    return f"slide {slide}" if slide else "deck"
