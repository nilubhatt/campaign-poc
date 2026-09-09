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
