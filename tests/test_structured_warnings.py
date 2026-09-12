"""
Product review defect 09 (P2): "Warnings are engineer-readable, not operator-actionable."

The reviewer was generous about the engineering and precise about the gap:

    "The download-failure warning was precise enough to diagnose from — genuinely good
    engineering output. But the intended user is a marketer, for whom 'Failed to download
    weights for tag openai' carries no action at all."

    Fix: "Give warnings a structured shape — a stable `code`, a one-line human `remedy`, and
    the raw `detail` — so the surface can say 'Visual search is offline — ask IT to run
    setup' while the detail stays available for support."

Three audiences, one warning: the marketer needs the remedy, support needs the detail, and
anything programmatic needs the code — which is the part a reworded message must not break.
"""
import pytest
from pptx import Presentation

import clip_embed
import config
import core
import extract
import notices
import store


def test_a_warning_carries_a_code_a_remedy_and_the_raw_detail(conn):
    result = core.ingest_campaign(conn, title="   ", confirm=True)

    warning = result["warnings"][0]
    assert warning["code"], "stable identifier — the part a reworded message must not break"
    assert warning["remedy"], "one line the marketer can act on"
    assert warning["detail"], "what support needs, kept rather than replaced"


def test_the_remedy_is_written_for_a_marketer_not_an_engineer(monkeypatch, tmp_path, conn):
    """The review's own example. "Failed to download weights for tag 'openai'" is a fine
    diagnostic and a useless instruction."""
    image = tmp_path / "hero.png"
    _write_png(image)

    def unavailable(*a, **k):
        raise RuntimeError("Failed to download weights for tag 'openai': connection refused")

    monkeypatch.setattr(clip_embed, "embed_image", unavailable)
    # The machine the reviewer ran on: the model is not there at all, which is the
    # operator's problem to fix once — as opposed to one bad PNG on a working model.
    monkeypatch.setattr(clip_embed, "weights_status", lambda: clip_embed.WeightsResolution(
        ok=False, source="missing", reason="no checkpoint found"))

    result = core.ingest_image_asset(conn, campaign_id=_campaign(conn),
                                    asset_ref={"path": str(image)})

    warning = next(w for w in result["warnings"] if w["code"] == "visual_search_offline")
    assert "weights" not in warning["remedy"].lower(), "a marketer does not know what a weight is"
    assert "openai" not in warning["remedy"].lower()
    assert "Failed to download weights" in warning["detail"], "support still gets the real text"


def test_every_code_in_use_has_a_remedy_written_for_it(conn):
    """A code with no remedy is a warning that reverted to being engineer-only, which is the
    defect. The registry is what stops one being added quietly."""
    for code in notices.CODES:
        assert notices.remedy_for(code), code


def test_an_unregistered_code_is_a_programming_error_not_a_blank_remedy(conn):
    with pytest.raises(KeyError):
        notices.notice("no_such_code", detail="whatever")


def test_the_code_survives_a_reworded_message(conn):
    """The point of a stable code: support tooling and the installer's own checks key off it,
    so improving the wording must not be a breaking change."""
    first = notices.notice("nothing_to_embed", detail="no title/detail/deck_text")

    assert first["code"] == "nothing_to_embed"
    assert first["detail"] == "no title/detail/deck_text"


def test_warnings_of_the_same_kind_are_not_repeated_once_per_item(tmp_path, conn, monkeypatch):
    """A deck with twenty unreadable images produced twenty near-identical lines, which is
    how a genuinely important warning gets scrolled past. One entry, with the count."""
    deck = _deck_with_images(tmp_path, 6)
    monkeypatch.setattr(core.images, "phash",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad image")))

    result = core.ingest_campaign(conn, title="Broken images",
                                  asset_ref={"path": str(deck)}, confirm=True)

    repeated = [w for w in result["warnings"] if w["code"] == "image_not_fingerprinted"]
    assert len(repeated) == 1, f"{len(repeated)} near-identical lines"
    assert repeated[0]["count"] == 6
    assert "bad image" in repeated[0]["detail"], "the first real reason is still there"


def test_the_budget_warning_keeps_its_instruction(tmp_path, conn, monkeypatch):
    """This one was already operator-actionable and says exactly what to do next; the
    restructuring must not lose it."""
    monkeypatch.setattr(config, "TOOL_TIME_BUDGET_SECONDS", 0.0)
    deck = "\n\n".join(f"section {i} " + "word " * 200 for i in range(6))

    result = core.ingest_campaign(conn, title="Cut short", deck_text=deck, confirm=True)

    warning = next(w for w in result["warnings"] if w["code"] == "indexing_incomplete")
    # §5.2: the call is an offer with structured arguments now, not a sentence. It was never
    # something to read out to somebody who cannot call a tool.
    offer = warning["next_actions"][0]
    assert offer["tool"] == "finish_indexing"
    assert offer["prefilled_args"]["campaign_id"] == result["campaign_id"]


def test_an_extraction_warning_is_structured_too(tmp_path):
    """extract.py raises warnings that reach the same response, so a caller cannot be left
    handling two shapes depending on which layer produced the line."""
    import zipfile

    prs = Presentation()
    prs.slides.add_slide(prs.slide_layouts[6])
    path = tmp_path / "broken.pptx"
    prs.save(str(path))
    with zipfile.ZipFile(path, "a") as z:
        z.writestr("ppt/modernComments/modernComment_1.xml", "<not-xml")

    _, warnings = extract.extract_commentary(path)

    assert warnings[0]["code"] == "commentary_unreadable"
    assert warnings[0]["remedy"]


def test_a_severity_says_whether_the_user_has_to_do_anything(conn):
    """"Visual search is offline" and "one image out of forty was skipped" are not the same
    news, and a flat list made a surface treat them identically."""
    result = core.ingest_campaign(conn, title="   ", confirm=True)

    assert result["warnings"][0]["severity"] in notices.SEVERITIES


def test_the_response_says_what_the_user_should_be_told(tmp_path, conn, monkeypatch):
    """With several warnings, a surface needs to know which one leads — otherwise it reads
    them all out, which is what a marketer ignores."""
    image = tmp_path / "hero.png"
    _write_png(image)
    monkeypatch.setattr(clip_embed, "embed_image",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no weights")))
    monkeypatch.setattr(clip_embed, "weights_status", lambda: clip_embed.WeightsResolution(
        ok=False, source="missing", reason="no checkpoint found"))

    result = core.ingest_image_asset(conn, campaign_id=_campaign(conn),
                                    asset_ref={"path": str(image)})

    assert result["warnings"], "precondition"
    leading = notices.leading(result["warnings"])
    assert leading["code"] == "visual_search_offline"


# ── helpers ─────────────────────────────────────────────────────────────────

def _campaign(conn):
    return core.ingest_campaign(conn, title="Host", detail="a campaign", confirm=True)[
        "campaign_id"]


def _write_png(path, seed=0):
    import numpy as np
    from PIL import Image

    data = np.random.default_rng(seed).integers(0, 256, (8, 8, 3), dtype="uint8")
    Image.fromarray(data, mode="RGB").resize((120, 120), Image.BICUBIC).save(path)
    return path


def _deck_with_images(tmp_path, n):
    from pptx.util import Inches

    prs = Presentation()
    for i in range(n):
        img = _write_png(tmp_path / f"i{i}.png", seed=i)
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.shapes.add_picture(str(img), Inches(1), Inches(1), width=Inches(2))
    deck = tmp_path / "deck.pptx"
    prs.save(str(deck))
    return deck


# ══ adversarial + design review of 3.1 ═══════════════════════════════════════

def test_two_different_remedies_are_never_folded_into_one(tmp_path, conn, monkeypatch):
    """Both reviewers, independently. When the budget runs out on a deck with images AND
    text, both loops raise `indexing_incomplete` with different overrides — and folding on
    the code alone kept the image sentence and threw away the one saying the deck's TEXT is
    unsearchable, on a response whose counts said 0 of 3 sections embedded."""
    deck = _deck_with_images(tmp_path, 3)
    monkeypatch.setattr(config, "TOOL_TIME_BUDGET_SECONDS", 0.0)

    result = core.ingest_campaign(conn, title="Both loops", deck_text="a" * 4000,
                                  asset_ref={"path": str(deck)}, confirm=True)

    incomplete = [w for w in result["warnings"] if w["code"] == "indexing_incomplete"]
    assert len(incomplete) == 2, "one per thing that was cut short"
    assert any("section" in w["affects"] for w in incomplete), (
        "the text half has to survive: it is what makes the deck unfindable"
    )


def test_folding_keeps_every_distinct_reason(tmp_path, conn, monkeypatch):
    """Four chunks failing for two different reasons reported one reason and a count of
    four, so the other cause was invisible to the person the detail exists for.
    `finish_indexing` already collapses by reason; this is the same trade made once."""
    reasons = iter(["connection refused", "connection refused", "timeout after 30s",
                    "timeout after 30s"])

    def flaky(text, timeout=None):
        raise RuntimeError(next(reasons, "timeout after 30s"))

    monkeypatch.setattr(core.embedding, "embed", flaky)
    result = core.ingest_campaign(conn, title="Flaky", deck_text="\n\n".join(
        f"section {i} " + "word " * 200 for i in range(4)), confirm=True)

    failures = [w for w in result["warnings"] if w["code"] == "chunk_not_embedded"]
    details = " ".join(w["detail"] for w in failures)
    assert "connection refused" in details and "timeout" in details


def test_a_remedy_never_claims_something_the_response_contradicts(tmp_path, conn, monkeypatch):
    """"Visual search is offline, so image similarity and reuse checks will miss this one" —
    said in a response that had just flagged a reuse at hamming distance 0. Perceptual
    hashing does not need the vision model, and telling a marketer their reuse check failed
    while showing them the match it found is the defect pointing the other way."""
    image = tmp_path / "hero.png"
    _write_png(image)
    cid = _campaign(conn)
    core.ingest_image_asset(conn, campaign_id=cid, asset_ref={"path": str(image)})

    monkeypatch.setattr(clip_embed, "weights_status", lambda: clip_embed.WeightsResolution(
        ok=False, source="missing", reason="no checkpoint", remedy="Reinstall to restore it."))
    monkeypatch.setattr(clip_embed, "embed_image",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no weights")))

    second = core.ingest_image_asset(conn, campaign_id=_campaign(conn),
                                     asset_ref={"path": str(image)})

    assert second["fingerprinted"] is True, "precondition: pHash does not need CLIP"
    warning = next(w for w in second["warnings"] if w["code"] == "visual_search_offline")
    assert "reuse" not in warning["remedy"].lower(), (
        "reuse detection is pHash and is working — do not report it as broken"
    )


def test_the_remedy_for_a_missing_model_is_the_one_the_component_computed(tmp_path, conn,
                                                                         monkeypatch):
    """"Ask IT to run setup" named a gesture that does not exist — there is no setup script,
    and nothing in run.sh or run.ps1 fetches the vision weights. `WeightsResolution` already
    works out the right remedy for each cause (missing bundled copy, bad configured path,
    load failure), and the registry was replacing it with a wrong static line."""
    image = tmp_path / "hero.png"
    _write_png(image)
    monkeypatch.setattr(clip_embed, "weights_status", lambda: clip_embed.WeightsResolution(
        ok=False, source="missing", reason="the shipped copy is not there",
        remedy="Reinstall Campaign Intelligence to restore the vision model."))
    monkeypatch.setattr(clip_embed, "embed_image",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no weights")))

    result = core.ingest_image_asset(conn, campaign_id=_campaign(conn),
                                     asset_ref={"path": str(image)})

    warning = next(w for w in result["warnings"] if w["code"] == "visual_search_offline")
    assert "Reinstall" in warning["remedy"], warning["remedy"]
    assert "run setup" not in warning["remedy"].lower()


def test_the_text_embedder_being_down_is_reported_as_an_outage(conn, monkeypatch):
    """The image path splits "the model is missing" from "this one item failed"; the text
    path did not, so a dead Ollama produced per-chunk advice to run finish_indexing — which
    then fails every item and says "calling again will not help". `text_search_offline` was
    written for this and was never emitted from anywhere."""
    import embedding

    monkeypatch.setattr(core.embedding, "embed",
                        lambda *a, **k: (_ for _ in ()).throw(embedding.Unavailable(
                            "could not reach the ollama embedder at http://localhost:11434")))

    result = core.ingest_campaign(conn, title="No embedder", deck_text="a brief", confirm=True)

    codes = {w["code"] for w in result["warnings"]}
    assert "text_search_offline" in codes, codes
    assert "chunk_not_embedded" not in codes, (
        "one outage, said once — not one line per chunk that could never have worked"
    )


def test_a_remedy_does_not_claim_the_record_is_unsearchable_when_it_is_not(tmp_path, conn):
    """A PNG passed as asset_ref alongside deck_text produced "This file type cannot be read,
    so nothing in it is searchable" on a record that was fully searchable — and advised
    converting a PNG to PDF. The old engineering string was accurate; the remedy was a
    regression to a wrong and alarming claim."""
    image = tmp_path / "hero.png"
    _write_png(image)

    result = core.ingest_campaign(conn, title="With a picture", deck_text="a real brief",
                                  asset_ref={"path": str(image)}, confirm=True)

    assert result["embedded"] is True, "precondition: the text made it in"
    for warning in result["warnings"]:
        assert "nothing in it is searchable" not in warning["remedy"], warning


def test_instructions_to_claude_are_not_read_out_to_the_user(tmp_path, conn, monkeypatch):
    """"Tell the user that, and offer to finish it now" is stage direction. The docstring
    says to say the remedy verbatim, so a marketer heard Claude's own instructions read
    back at them. One reader per field."""
    monkeypatch.setattr(config, "TOOL_TIME_BUDGET_SECONDS", 0.0)
    result = core.ingest_campaign(conn, title="Cut short", deck_text="\n\n".join(
        f"section {i} " + "word " * 200 for i in range(4)), confirm=True)

    warning = next(w for w in result["warnings"] if w["code"] == "indexing_incomplete")

    assert "tell the user" not in warning["remedy"].lower()
    assert warning["next_actions"], "what Claude should do goes in its own field"
    assert warning["next_actions"][0]["tool"] == "finish_indexing"


def test_the_response_names_the_warning_to_lead_with(tmp_path, conn, monkeypatch):
    """Ordering was an instruction in one tool's docstring rather than a property of the
    response, and the blocked one arrived second. health_check already solved this with a
    headline."""
    image = tmp_path / "hero.png"
    _write_png(image)
    monkeypatch.setattr(clip_embed, "weights_status", lambda: clip_embed.WeightsResolution(
        ok=False, source="missing", reason="gone", remedy="Reinstall."))
    monkeypatch.setattr(clip_embed, "embed_image",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no weights")))
    monkeypatch.setattr(core.images, "phash",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad image")))

    deck = _deck_with_images(tmp_path, 2)
    result = core.ingest_campaign(conn, title="Several problems",
                                  asset_ref={"path": str(deck)}, confirm=True)

    assert result["warnings"][0]["severity"] == "blocked", (
        "most severe first, so a surface that reads the list in order leads correctly"
    )


def test_a_blank_upload_is_not_the_same_class_as_a_missing_model(conn):
    """`blocked` means a capability is off on this machine until somebody acts.
    An empty record is neither: the product works and the fix is the user's own input.
    Ranking it above everything but a real outage would have Claude lead with an
    IT-flavoured alarm over a typo."""
    result = core.ingest_campaign(conn, title="   ", confirm=True)

    warning = result["warnings"][0]
    assert warning["code"] == "nothing_to_embed"
    assert warning["severity"] != "blocked"
    assert warning["scope"] == "record", "who has to act, and about what"


def test_every_registered_code_is_actually_reachable():
    """The registry test passed against an empty registry, because an empty loop body is a
    passing test. And a code nobody emits is a remedy nobody proof-read."""
    import re

    assert len(notices.CODES) >= 10, "the loop below has to actually run"
    sources = "".join((Path(p).read_text(encoding="utf-8"))
                      for p in ("core.py", "extract.py", "clip_embed.py", "store.py"))
    for code in notices.CODES:
        assert re.search(rf'["\']{re.escape(code)}["\']', sources), (
            f"{code} is registered but emitted from nowhere"
        )


def test_rewording_a_remedy_is_not_a_breaking_change():
    """The whole point of a stable code. Any test that asserts a remedy's exact wording
    makes improving it a breaking change, which is the thing the code exists to prevent."""
    import re

    suite = "".join(Path(p).read_text(encoding="utf-8") for p in Path("tests").glob("*.py"))
    # Only POSITIVE substring assertions couple to wording. "x not in remedy" is the
    # opposite: it forbids a class of mistake and stays true however the line is rewritten,
    # which is exactly what this file wants more of.
    # Scoped to notice remedies (the dict form). WeightsResolution carries its own `remedy`
    # with no code behind it, and asserting on that one is a different bargain.
    positive = [line for line in suite.splitlines()
                if re.search(r'assert\s+(?!not\b)["\'][^"\']+["\']\s+in\s+[^\s]*\["remedy"\]',
                             line)]

    assert len(positive) <= 3, (
        f"{len(positive)} tests assert exact remedy wording: {positive}. Assert the code, or "
        f"assert what the line must NOT say."
    )


from pathlib import Path  # noqa: E402  (used by the two tests above)


def test_an_unreachable_embedder_is_a_different_exception_from_a_rejected_chunk(monkeypatch):
    """The distinction core relies on, made where it is actually known. Reported as a type
    rather than matched on the message, because a reworded message must not be a breaking
    change — which is the whole premise of the stable code."""
    import httpx

    import embedding

    monkeypatch.setattr(config, "EMBED_PROVIDER", "ollama")

    monkeypatch.setattr(httpx, "post",
                        lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    with pytest.raises(embedding.Unavailable):
        embedding.embed("anything")

    response = httpx.Response(400, request=httpx.Request("POST", "http://x"))
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(
        httpx.HTTPStatusError("too long", request=response.request, response=response)))
    with pytest.raises(ValueError) as exc:
        embedding.embed("anything")
    assert not isinstance(exc.value, embedding.Unavailable), (
        "the model rejecting one input is not the service being down — the rest of the "
        "deck should still be attempted"
    )


def test_a_folded_warning_reads_as_plural_when_it_is_plural(tmp_path, conn, monkeypatch):
    """`count` was stored and no text ever consumed it, so six unreadable images said "An
    image in the deck could not be saved" — singular, about one of six. The count is the
    only part of a folded warning that says how big the problem is."""
    deck = _deck_with_images(tmp_path, 4)
    monkeypatch.setattr(core.images, "phash",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad image")))

    result = core.ingest_campaign(conn, title="Broken", asset_ref={"path": str(deck)},
                                  confirm=True)

    warning = next(w for w in result["warnings"] if w["code"] == "image_not_fingerprinted")
    assert warning["count"] == 4
    assert "4" in warning["affects"], (
        f"the user is told how many: {warning['affects']}"
    )


def test_a_single_occurrence_does_not_say_a_number(tmp_path, conn, monkeypatch):
    """The other half: "1 images" is how a count that is always interpolated reads."""
    image = tmp_path / "hero.png"
    _write_png(image)
    monkeypatch.setattr(core.images, "phash",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad image")))

    result = core.ingest_image_asset(conn, campaign_id=_campaign(conn),
                                     asset_ref={"path": str(image)})

    warning = next(w for w in result["warnings"] if w["code"] == "image_not_fingerprinted")
    assert "count" not in warning
    assert "1 " not in warning["affects"]


def test_a_judgment_is_told_when_the_library_is_only_half_indexed(conn, monkeypatch):
    """`find_similar_campaigns` warns that results may be incomplete; `prepare_evaluation`
    calls the unwrapped `find_similar` and says nothing — so the one surface where it matters
    most, a verdict about to be saved against this evidence, was the one surface that stayed
    silent about the evidence being partial."""
    core.ingest_campaign(conn, title="Indexed", detail="a concluded campaign", confirm=True)
    unindexed = core.ingest_campaign(conn, title="Not indexed", detail="another campaign",
                                     confirm=True)
    conn.execute("UPDATE campaign_chunks SET embedded = 0 WHERE campaign_id = ?",
                 (unindexed["campaign_id"],))
    conn.commit()

    packaged = core.prepare_evaluation(conn, subject_title="New brief",
                                       proposal_text="a campaign like the others")

    assert any(w["code"] == "results_may_be_incomplete" for w in packaged["warnings"]), (
        packaged.get("warnings")
    )


def test_one_condition_produces_one_warning(tmp_path, conn):
    """A legacy .ppt reported both `legacy_ppt` and `unsupported_file_type` — two entries,
    slightly different advice, one problem. Which of the two a surface leads with is then
    arbitrary."""
    deck = tmp_path / "old.ppt"
    deck.write_bytes(b"\xd0\xcf\x11\xe0" + b"\x00" * 64)   # OLE2 magic

    result = core.ingest_campaign(conn, title="Old deck", asset_ref={"path": str(deck)},
                                  confirm=True)

    codes = [w["code"] for w in result["warnings"]]
    assert codes.count("legacy_ppt") + codes.count("unsupported_file_type") <= 1, codes
