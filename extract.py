"""PDF / PPTX text extraction — the only file processing the POC needs."""
from __future__ import annotations

import mimetypes
from pathlib import Path

import config


def guess_mime(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


def extract_text(path: Path, mime: str | None = None) -> tuple[str, list[str]]:
    """
    Return (text, warnings). PDF and .pptx are extracted; legacy .ppt is accepted upstream
    but returns empty text with a warning (no reliable pure-Python extractor).
    """
    mime = mime or guess_mime(path.name)
    if mime == "application/pdf":
        return _read_pdf(path), []
    if mime == config.PPTX_MIME:
        return _read_pptx(path), []
    if mime == config.PPT_LEGACY_MIME:
        return "", ["legacy .ppt stored but not text-extracted; convert to .pptx for searchability"]
    return "", [f"unsupported type {mime}; POC accepts PDF and PPTX"]


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    out, total = [], 0
    for i, page in enumerate(reader.pages):
        if i >= config.MAX_PDF_PAGES:
            break
        chunk = page.extract_text() or ""
        out.append(chunk)
        total += len(chunk)
        if total >= config.MAX_DOC_TEXT_CHARS:
            break
    return "\n".join(out)[: config.MAX_DOC_TEXT_CHARS]


def _read_pptx(path: Path) -> str:
    from pptx import Presentation
    prs = Presentation(str(path))
    out, total = [], 0
    for i, slide in enumerate(prs.slides):
        if i >= config.MAX_PPTX_SLIDES:
            break
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text:
                out.append(shape.text_frame.text)
                total += len(shape.text_frame.text)
        if total >= config.MAX_DOC_TEXT_CHARS:
            break
    return "\n".join(out)[: config.MAX_DOC_TEXT_CHARS]
