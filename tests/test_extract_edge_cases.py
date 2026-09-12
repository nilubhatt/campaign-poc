from pathlib import Path

import config
import extract


def test_guess_mime_pdf_and_pptx():
    assert extract.guess_mime("deck.pdf") == "application/pdf"
    assert extract.guess_mime("deck.pptx") == config.PPTX_MIME


def test_guess_mime_unknown_extension_falls_back():
    assert extract.guess_mime("deck.notarealextension123") == "application/octet-stream"


def test_legacy_ppt_returns_no_units_with_warning():
    units, warnings = extract.extract_units(Path("whatever.ppt"), mime=config.PPT_LEGACY_MIME)
    assert units == []
    assert warnings[0]["code"] == "legacy_ppt"
    assert "legacy .ppt" in warnings[0]["detail"]
    assert ".pptx" in warnings[0]["remedy"], "tell them what to actually do about it"


def test_unsupported_mime_returns_no_units_with_warning():
    units, warnings = extract.extract_units(Path("whatever.docx"), mime="application/msword")
    assert units == []
    assert warnings[0]["code"] == "unsupported_file_type"
    assert "unsupported type" in warnings[0]["detail"]


class _FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class _FakeReader:
    def __init__(self, path):
        self.pages = [_FakePage(f"page {i} text") for i in range(10)]


def test_pdf_extraction_respects_max_pages(monkeypatch):
    monkeypatch.setattr(config, "MAX_PDF_PAGES", 3)
    monkeypatch.setattr(config, "MAX_DOC_TEXT_CHARS", 100_000)
    monkeypatch.setattr("pypdf.PdfReader", _FakeReader)

    units, warnings = extract.extract_units(Path("fake.pdf"), mime="application/pdf")
    assert warnings == []
    assert len(units) == 3
    assert units == ["page 0 text", "page 1 text", "page 2 text"]


def test_pdf_extraction_respects_char_cap(monkeypatch):
    monkeypatch.setattr(config, "MAX_PDF_PAGES", 100)
    # cap smaller than a single page (11 chars) -> the cap check (before adding a page)
    # trips on the *next* iteration, so exactly one page gets in.
    monkeypatch.setattr(config, "MAX_DOC_TEXT_CHARS", 5)
    monkeypatch.setattr("pypdf.PdfReader", _FakeReader)

    units, warnings = extract.extract_units(Path("fake.pdf"), mime="application/pdf")
    assert warnings == []
    assert units == ["page 0 text"]


def test_pdf_extraction_skips_blank_pages(monkeypatch):
    class BlankAwareReader:
        def __init__(self, path):
            self.pages = [_FakePage("real text"), _FakePage(""), _FakePage("   "), _FakePage("more text")]

    monkeypatch.setattr(config, "MAX_PDF_PAGES", 100)
    monkeypatch.setattr(config, "MAX_DOC_TEXT_CHARS", 100_000)
    monkeypatch.setattr("pypdf.PdfReader", BlankAwareReader)

    units, warnings = extract.extract_units(Path("fake.pdf"), mime="application/pdf")
    assert units == ["real text", "more text"]
