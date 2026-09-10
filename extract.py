"""PDF / PPTX text extraction — the only file processing the POC needs."""
from __future__ import annotations

import mimetypes
from pathlib import Path

import config


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
        return [], ["legacy .ppt stored but not text-extracted; convert to .pptx for searchability"]
    return [], [f"unsupported type {mime}; POC accepts PDF and PPTX"]


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
    return [], [f"image extraction not supported for {mime}"]


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
        warnings.append(
            f"deck has more than {config.MAX_EXTRACTED_IMAGES_PER_DECK} distinct embedded "
            f"images; only the first {config.MAX_EXTRACTED_IMAGES_PER_DECK} were checked"
        )
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
            warnings.append(f"page {page_num}: could not read embedded images ({exc}); skipped")
            continue
        for j in range(n):
            try:
                img = page.images[j]
                ext = Path(img.name).suffix or ".png"
                raw.append((img.data, ext, page_num))
            except Exception as exc:
                warnings.append(f"page {page_num} image {j + 1}: could not be read ({exc}); skipped")
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
                    warnings.append(f"slide {slide_num}: image could not be read ({exc}); skipped")

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
