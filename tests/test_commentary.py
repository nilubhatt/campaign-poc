"""
Product review defect 08 (P1): "Comments and speaker notes are never ingested."

The reviewer tested this against the PDFs already in the library and measured the loss:

    "One of them carries eight substantive /Text annotations — PowerPoint speaker notes
    exported as PDF comments. None of it is indexed. Querying the library with that second
    note almost verbatim returns the right record at only 0.63, matching on the summary a
    human wrote — not on the note. Had the note been ingested, a near-verbatim query would
    score above 0.9."

And said why it matters more here than the defect number suggests: "a partner deck returned
with tracked client comments is the feedback the library exists to remember, and it
currently has to be retyped by hand into the detail field to be captured at all."

The fix is explicit that commentary is a DIFFERENT LAYER, not more deck text: "store them as
a distinct layer from the deck body — not concatenated into deck_text. They are a different
kind of evidence: commentary carries author, date, and page or slide anchor, and it is
usually the internal reaction to the work rather than the work itself."
"""
import datetime
import json
import zipfile

import pytest
from pptx import Presentation
from pypdf import PdfWriter
from pypdf.annotations import FreeText, Link, Text

import config
import core
import extract
import store


def _pdf_with_annotations(path, annots):
    """A PDF carrying real /Annots, built the way the reviewer's deck was: PowerPoint
    speaker notes exported as PDF comments."""
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=200, height=200)
    for page_index, annotation, author in annots:
        writer.add_annotation(page_number=page_index, annotation=annotation)
        # pypdf's helper does not write /T or /M, which is where author and date live.
        target = writer.pages[page_index]["/Annots"][-1].get_object()
        target[pypdf_name("/T")] = _pdf_string(author)
        target[pypdf_name("/M")] = _pdf_string("D:20260904143000+01'00'")
    with open(path, "wb") as fh:
        writer.write(fh)
    return path


def pypdf_name(value):
    from pypdf.generic import NameObject
    return NameObject(value)


def _pdf_string(value):
    from pypdf.generic import TextStringObject
    return TextStringObject(value)


NOTE_2 = ("Keep the leadership deck concise and put detail in the workbook. The workbook is "
          "the scalable input, and the deck is the summary of it.")
NOTE_3 = ("Call out the new competitor brand rule explicitly. These are compliance and brand "
          "safety rules, not preferences.")


# ── extraction ───────────────────────────────────────────────────────────────

def test_pdf_text_annotations_are_extracted_with_author_date_and_page(tmp_path):
    path = _pdf_with_annotations(tmp_path / "deck.pdf", [
        (1, Text(rect=(10, 10, 30, 30), text=NOTE_2), "Priya"),
        (2, Text(rect=(10, 10, 30, 30), text=NOTE_3), "Priya"),
    ])

    commentary, warnings = extract.extract_commentary(path)

    assert [c["text"] for c in commentary] == [NOTE_2, NOTE_3]
    assert [c["page"] for c in commentary] == [2, 3], "1-based, like every other anchor"
    assert all(c["author"] == "Priya" for c in commentary)
    assert commentary[0]["date"].startswith("2026-09-04"), "/M is a PDF date, not a string"
    assert commentary[0]["kind"] == "annotation"


def test_a_link_annotation_is_not_commentary(tmp_path):
    """"Every other deck in the library carries /Link annotations only, so no comment text
    is being silently dropped elsewhere" — a hyperlink is not somebody's opinion, and
    indexing one would put URL fragments into the evidence a judgment cites."""
    path = _pdf_with_annotations(tmp_path / "links.pdf", [
        (0, Link(rect=(10, 10, 30, 30), url="https://example.com/brief"), "Word"),
    ])

    commentary, _ = extract.extract_commentary(path)

    assert commentary == []


def test_a_freetext_annotation_is_commentary(tmp_path):
    path = _pdf_with_annotations(tmp_path / "ft.pdf", [
        (0, FreeText(text="Lose the third colourway", rect=(10, 10, 90, 30)), "Sam"),
    ])

    commentary, _ = extract.extract_commentary(path)

    assert commentary[0]["text"] == "Lose the third colourway"
    assert commentary[0]["author"] == "Sam"


def test_pptx_speaker_notes_are_extracted(tmp_path):
    prs = Presentation()
    for i in range(2):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.notes_slide.notes_text_frame.text = f"Speaker note {i + 1}: {NOTE_2}"
    path = tmp_path / "deck.pptx"
    prs.save(str(path))

    commentary, _ = extract.extract_commentary(path)

    assert len(commentary) == 2
    assert commentary[0]["kind"] == "speaker_note"
    assert commentary[0]["slide"] == 1
    assert NOTE_2 in commentary[0]["text"]


def test_an_empty_notes_slide_is_not_a_comment(tmp_path):
    """PowerPoint creates a notesSlide for every slide the moment anything touches it, so
    the common case is an empty one. A library full of blank commentary rows is noise that
    dilutes every retrieval."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.notes_slide.notes_text_frame.text = "   "
    path = tmp_path / "blank.pptx"
    prs.save(str(path))

    assert extract.extract_commentary(path)[0] == []


def test_pptx_reviewer_comments_are_extracted_with_author_and_timestamp(tmp_path):
    """The real prize: "a partner deck returned with tracked client comments is the feedback
    the library exists to remember". python-pptx has no API for these, so they are read out
    of the package parts directly — modernComments is what current PowerPoint writes."""
    prs = Presentation()
    prs.slides.add_slide(prs.slide_layouts[6])
    path = tmp_path / "commented.pptx"
    prs.save(str(path))

    _add_modern_comment(path, author="Dana Ruiz", when="2026-09-04T14:30:00.000",
                        text="This is the adidas-affiliated profile we flagged in Q1.")

    commentary, _ = extract.extract_commentary(path)

    comment = next(c for c in commentary if c["kind"] == "comment")
    assert comment["author"] == "Dana Ruiz"
    assert comment["date"].startswith("2026-09-04")
    assert "adidas-affiliated" in comment["text"]


def test_a_deck_with_no_commentary_yields_none_without_complaining(tmp_path):
    prs = Presentation()
    prs.slides.add_slide(prs.slide_layouts[6])
    path = tmp_path / "plain.pptx"
    prs.save(str(path))

    commentary, warnings = extract.extract_commentary(path)

    assert commentary == []
    assert warnings == []


def test_unreadable_commentary_is_a_warning_not_a_failed_upload(tmp_path):
    """Commentary is the bonus layer. A malformed comments part must never cost the user the
    deck itself — they uploaded a brief, not a comments file."""
    prs = Presentation()
    prs.slides.add_slide(prs.slide_layouts[6])
    path = tmp_path / "broken.pptx"
    prs.save(str(path))
    with zipfile.ZipFile(path, "a") as z:
        z.writestr("ppt/modernComments/modernComment_1.xml", "<not-xml")

    commentary, warnings = extract.extract_commentary(path)

    assert commentary == []
    assert any("comment" in w.lower() for w in warnings)


# ── it is a distinct layer, not more deck text ───────────────────────────────

def test_commentary_is_not_concatenated_into_the_deck_body(tmp_path, conn):
    """The review was explicit: "not concatenated into deck_text... usually the internal
    reaction to the work rather than the work itself". Merged in, a reviewer's objection
    reads back as something the brief itself claimed."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Colombia Q4 activation"
    slide.notes_slide.notes_text_frame.text = "I do not think the timeline is realistic."
    path = tmp_path / "deck.pptx"
    prs.save(str(path))

    result = core.ingest_campaign(conn, title="Colombia", asset_ref={"path": str(path)},
                                  confirm=True)
    campaign = store.get_campaign(conn, result["campaign_id"])

    assert "Colombia Q4 activation" in campaign["deck_text"]
    assert "not think the timeline" not in (campaign["deck_text"] or "")
    assert any("not think the timeline" in c["text"] for c in campaign["commentary"])


def test_commentary_carries_its_anchor_author_and_kind_into_storage(tmp_path, conn):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.notes_slide.notes_text_frame.text = NOTE_3
    path = tmp_path / "deck.pptx"
    prs.save(str(path))

    result = core.ingest_campaign(conn, title="Fabletics AI agent",
                                  asset_ref={"path": str(path)}, confirm=True)
    stored = store.get_campaign(conn, result["campaign_id"])["commentary"][0]

    assert stored["kind"] == "speaker_note"
    assert stored["anchor"] == "slide 1"
    assert stored["text"] == NOTE_3


def test_the_ingest_result_says_how_much_commentary_it_found(tmp_path, conn):
    """Silence is how defect 08 survived: nothing said the notes existed, so nothing said
    they were being dropped."""
    prs = Presentation()
    for _ in range(3):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.notes_slide.notes_text_frame.text = NOTE_2
    path = tmp_path / "deck.pptx"
    prs.save(str(path))

    result = core.ingest_campaign(conn, title="Noted", asset_ref={"path": str(path)},
                                  confirm=True)

    assert result["commentary_found"] == 3


# ── the reviewer's own measurement ───────────────────────────────────────────

def test_a_near_verbatim_query_on_a_note_finds_the_deck_that_carries_it(tmp_path, conn):
    """The review's numbers: querying near-verbatim on an unindexed note scored 0.63,
    matching the human-written summary instead. "Had the note been ingested, a near-verbatim
    query would score above 0.9." This is that measurement, as a test."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Fabletics AI agent — leadership update"
    slide.notes_slide.notes_text_frame.text = NOTE_2
    path = tmp_path / "deck.pptx"
    prs.save(str(path))

    core.ingest_campaign(conn, title="Fabletics AI agent", asset_ref={"path": str(path)},
                         confirm=True)

    # Near-verbatim, as the reviewer's probe was: the wording a person retypes from memory,
    # not a paraphrase. (A looser paraphrase of the same note lands around 0.90, which is
    # still the right record and still far above the 0.63 the un-indexed note managed.)
    hits = core.find_similar(conn, text="Keep the leadership deck concise and put detail in "
                                        "the workbook. The workbook is the scalable input "
                                        "and the deck summarises it.", top_k=3)

    assert hits, "the note is indexed, so it is findable"
    assert hits[0]["similarity"] > 0.9, f"scored {hits[0]['similarity']}"


def test_a_commentary_match_says_it_is_commentary_and_who_said_it(tmp_path, conn):
    """"tagged so retrieval can weigh or filter it" — a judgment that cites a reviewer's
    objection as if it were the brief's own claim is citing the wrong thing."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Colombia Q4"
    slide.notes_slide.notes_text_frame.text = ("The claw machine idea outperformed every "
                                               "other mechanic we tried in the UAE.")
    path = tmp_path / "deck.pptx"
    prs.save(str(path))

    core.ingest_campaign(conn, title="Colombia", asset_ref={"path": str(path)}, confirm=True)

    hit = core.find_similar(conn, text="claw machine mechanic performance", top_k=1)[0]

    assert hit["matched_kind"] == "commentary"
    assert hit["matched_anchor"] == "slide 1"


def test_commentary_can_be_excluded_from_retrieval(tmp_path, conn):
    """Weighing it means being able to not weigh it: "is this what the brief says, or what
    one person thought of it" is a question a marketer will ask."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Colombia Q4 influencer activation"
    slide.notes_slide.notes_text_frame.text = "A completely unrelated remark about logistics"
    path = tmp_path / "deck.pptx"
    prs.save(str(path))

    core.ingest_campaign(conn, title="Colombia", asset_ref={"path": str(path)}, confirm=True)

    hit = core.find_similar(conn, text="unrelated remark about logistics", top_k=1,
                            include_commentary=False)

    assert hit == [] or hit[0]["matched_kind"] != "commentary"


# ── it must not cost the deck ────────────────────────────────────────────────

def test_a_database_from_before_this_release_gains_the_commentary_layer(tmp_path):
    """Every schema change from here on gets this test — 2.4 shipped one that was fine on a
    fresh database and impossible on a real one."""
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE campaigns (id TEXT PRIMARY KEY, title TEXT NOT NULL, collection TEXT,
                                markets TEXT NOT NULL DEFAULT '[]');
        CREATE TABLE campaign_chunks (id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL,
                                      chunk_index INTEGER NOT NULL, text TEXT NOT NULL,
                                      embedded INTEGER NOT NULL DEFAULT 0,
                                      created_at REAL NOT NULL);
        CREATE TABLE evaluations (id TEXT PRIMARY KEY, campaign_id TEXT,
                                  subject_title TEXT NOT NULL, cited_ids TEXT,
                                  analysis TEXT NOT NULL, predictions TEXT,
                                  created_at REAL NOT NULL);
        CREATE TABLE reconciliations (id TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL,
                                      actual TEXT, comparison TEXT NOT NULL,
                                      created_at REAL NOT NULL);
    """)
    old.execute("INSERT INTO campaign_chunks VALUES ('ch1','c1',0,'existing body text',1,1.0)")
    old.commit()
    old.close()

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    store._migrate_schema(conn)

    row = conn.execute("SELECT * FROM campaign_chunks WHERE id='ch1'").fetchone()
    assert row["text"] == "existing body text"
    assert row["kind"] == "body", "an existing chunk is deck body, not commentary"


def test_a_deck_whose_commentary_cannot_be_read_still_ingests(tmp_path, conn, monkeypatch):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Colombia Q4"
    path = tmp_path / "deck.pptx"
    prs.save(str(path))

    def boom(*a, **k):
        raise RuntimeError("comments part is corrupt")

    monkeypatch.setattr(extract, "extract_commentary", boom)

    result = core.ingest_campaign(conn, title="Colombia", asset_ref={"path": str(path)},
                                  confirm=True)

    assert result["chunks_total"] > 0
    assert any("comment" in w.lower() for w in result["warnings"])


def test_commentary_is_capped_like_every_other_unbounded_input(tmp_path, monkeypatch):
    """A deck's comment count is controlled by whoever made the deck, and every other
    caller-sized loop in this server is bounded."""
    monkeypatch.setattr(config, "MAX_COMMENTARY_ITEMS", 5)
    prs = Presentation()
    for _ in range(config.MAX_COMMENTARY_ITEMS + 4):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.notes_slide.notes_text_frame.text = NOTE_2
    path = tmp_path / "many.pptx"
    prs.save(str(path))

    commentary, warnings = extract.extract_commentary(path)

    assert len(commentary) == config.MAX_COMMENTARY_ITEMS
    assert any(str(config.MAX_COMMENTARY_ITEMS) in w for w in warnings)


def _add_modern_comment(path, *, author, when, text):
    """Write a ppt/modernComments part into an existing .pptx, which is what PowerPoint
    does when a reviewer comments on a slide. python-pptx cannot author these."""
    import shutil

    src = str(path) + ".orig"
    shutil.move(str(path), src)
    author_id = "{6E9F5E3E-0000-0000-0000-000000000001}"
    comment_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p188:cmLst xmlns:p188="http://schemas.microsoft.com/office/powerpoint/2018/8/main">'
        f'<p188:cm id="1" authorId="{author_id}" created="{when}">'
        '<p188:txBody><a:bodyPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"/>'
        '<a:p xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        f'<a:r><a:t>{text}</a:t></a:r></a:p></p188:txBody></p188:cm></p188:cmLst>'
    )
    authors_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p188:cmAuthorLst xmlns:p188="http://schemas.microsoft.com/office/powerpoint/2018/8/main">'
        f'<p188:cmAuthor id="{author_id}" name="{author}" initials="DR"/></p188:cmAuthorLst>'
    )
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(path, "w") as zout:
        for item in zin.infolist():
            zout.writestr(item, zin.read(item.filename))
        zout.writestr("ppt/modernComments/modernComment_1.xml", comment_xml)
        zout.writestr("ppt/authors.xml", authors_xml)
