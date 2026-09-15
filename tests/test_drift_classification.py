"""
§9.4: drift is not a synonym for failure — classify it.

The review: *"The UAE brief's claw machine was a departure from precedent that turned out
better than precedent. Treating every deviation as a defect would teach the library to punish
improvement. Fix: classify each drift item as improvement, neutral or degradation, and mark
that classification **judged** rather than **computed**. Presence and absence are facts;
whether a change was good is a judgment, and usually needs the outcome to settle it."*

**"Presence and absence are facts; whether a change was good is a judgment"** is the sentence
this item is built on, and it draws a line the server must not cross. §9.2 computes which
briefed images came back and §9.3 looks for the named promises — both mechanical. What none of
them can say is whether a thing not appearing was a loss. A claw machine replaced by something
better is drift; so is a claw machine that never turned up. The numbers are identical.

So the server records a classification and never makes one. There is no default, no inference
from the drift score, and no "high drift is probably bad" anywhere — that inference is exactly
what "would teach the library to punish improvement" describes.

**"Usually needs the outcome to settle it"** gets its own answer rather than a caveat. A fourth
value, `too_early`, is the honest classification before results exist; and every classification
is stamped with whether the outcome was known WHEN IT WAS MADE, so a later reader can weigh
"this looked like an improvement" against "this was an improvement" without having to
reconstruct which one it was.
"""
import pytest

import core
import drift
import store


def _campaign(conn, title="UAE launch"):
    return core.ingest_campaign(conn, title=title, market="EMEA", status="concluded",
                                detail="Plan:\n- A claw machine\n- A photo booth\n")[
        "campaign_id"]


def _claw(conn, campaign_id):
    """The commitment id for the claw machine — a subject somebody can read back, which is the
    point of keying on one."""
    import commitments
    return next(c["id"] for c in commitments.for_campaign(conn, campaign_id)
                if "claw" in c["text"].lower())


def _booth(conn, campaign_id):
    import commitments
    return next(c["id"] for c in commitments.for_campaign(conn, campaign_id)
                if "booth" in c["text"].lower())


def _png(tmp_path, name, seed=1):
    import random

    from PIL import Image

    rng = random.Random(seed)
    img = Image.new("RGB", (64, 64), (250, 250, 250))
    px = img.load()
    for _ in range(700):
        x, y = rng.randrange(64), rng.randrange(64)
        c = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        for dx in range(4):
            for dy in range(4):
                if x + dx < 64 and y + dy < 64:
                    px[x + dx, y + dy] = c
    path = tmp_path / name
    img.save(path)
    return str(path)


# ── the server records, and never decides ───────────────────────────────────

def test_a_classification_is_judged_never_computed(conn):
    """"Mark that classification judged rather than computed." §2.4 made `computed` unwritable
    from the MCP surface precisely so the word keeps meaning "the server worked this out"."""
    cid = _campaign(conn)

    saved = drift.classify(conn, campaign_id=cid, subject=_claw(conn, cid),
                           classification="improvement",
                           why="A claw machine drew a queue the photo booth never did.",
                           classified_by="R. Vega")

    assert saved["basis"] == "judged"


def test_the_server_classifies_nothing_on_its_own(conn, tmp_path):
    """There is no default and no inference from the drift score. "High drift is probably bad"
    is exactly the inference that "would teach the library to punish improvement"."""
    cid = _campaign(conn)
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": _png(tmp_path, "brief.png", seed=3)})
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": _png(tmp_path, "shot.png", seed=91)},
                            phase="delivered", captured_on="2026-03-14")

    out = core.compare_execution(conn, campaign_id=cid)

    assert out["drift"]["score"] > 0
    assert all(item.get("classification") is None
               for item in out["never_appeared"] + out["new"])
    assert "classified" in out["what_it_means"] or out["unclassified"] > 0


def test_a_classification_needs_a_reason(conn):
    """A judgment with no reason cannot be weighed later, and this one is meant to be read
    months afterwards by somebody deciding whether to repeat the change."""
    cid = _campaign(conn)

    with pytest.raises(ValueError) as e:
        drift.classify(conn, campaign_id=cid, subject=_claw(conn, cid),
                       classification="improvement", why="  ", classified_by="R. Vega")
    assert "why" in str(e.value).lower()


def test_a_classification_needs_a_person(conn):
    """The same rule as §8.3's gate: a judgment nobody's name is against is one nobody can
    question later."""
    cid = _campaign(conn)

    with pytest.raises(ValueError) as e:
        drift.classify(conn, campaign_id=cid, subject=_claw(conn, cid),
                       classification="improvement", why="It drew a queue.",
                       classified_by="")
    assert "person" in str(e.value).lower()


def test_a_classification_outside_the_vocabulary_is_refused_with_it(conn):
    cid = _campaign(conn)

    with pytest.raises(ValueError) as e:
        drift.classify(conn, campaign_id=cid, subject=_claw(conn, cid), classification="bad",
                       why="It was bad.", classified_by="R. Vega")
    assert "improvement" in str(e.value) and "degradation" in str(e.value)


# ── the UAE claw machine ────────────────────────────────────────────────────

def test_a_departure_can_be_an_improvement(conn):
    """"The UAE brief's claw machine was a departure from precedent that turned out better than
    precedent. Treating every deviation as a defect would teach the library to punish
    improvement.\""""
    cid = _campaign(conn)

    drift.classify(conn, campaign_id=cid, subject=_claw(conn, cid),
                   classification="improvement",
                   why="Unbriefed, and it drew the queue the photo booth was meant to.",
                   classified_by="R. Vega")

    summary = drift.for_campaign(conn, cid)
    assert summary["counts"]["improvement"] == 1
    assert summary["counts"]["degradation"] == 0


def test_high_drift_and_improvement_are_expressible_together(conn, tmp_path):
    """The whole point of the item. A campaign whose execution moved a long way from its brief
    and moved somewhere better has to be sayable, or the library learns that distance is bad."""
    cid = _campaign(conn)
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": _png(tmp_path, "brief.png", seed=3)})
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": _png(tmp_path, "shot.png", seed=91)},
                            phase="delivered", captured_on="2026-03-14")
    drift.classify(conn, campaign_id=cid, subject=_claw(conn, cid),
                   classification="improvement", why="It worked better.",
                   classified_by="R. Vega")

    out = core.compare_execution(conn, campaign_id=cid)
    assert out["drift"]["score"] > 0
    assert out["classified"]["counts"]["improvement"] == 1


# ── "usually needs the outcome to settle it" ────────────────────────────────

def test_too_early_is_an_answer(conn):
    """A fourth value rather than a caveat. Before the results exist, "it is too early to say"
    is the true classification and forcing one of the other three invents a verdict."""
    cid = _campaign(conn)

    saved = drift.classify(conn, campaign_id=cid, subject=_claw(conn, cid),
                           classification="too_early",
                           why="Nothing has been measured yet.", classified_by="R. Vega")

    assert saved["classification"] == "too_early"


def test_whether_the_outcome_was_known_is_stamped(conn):
    """So a later reader can weigh "this looked like an improvement" against "this WAS an
    improvement" without reconstructing which one it was."""
    cid = _campaign(conn)

    before = drift.classify(conn, campaign_id=cid, subject=_claw(conn, cid),
                            classification="improvement", why="It drew a queue.",
                            classified_by="R. Vega")
    assert before["outcome_known"] is False

    core.add_metrics(conn, campaign_id=cid, structured={"footfall_uplift_pct": 12.0},
                     confirm=True)
    after = drift.classify(conn, campaign_id=cid, subject=_booth(conn, cid),
                           classification="degradation", why="Footfall came from the claw.",
                           classified_by="R. Vega")
    assert after["outcome_known"] is True


def test_a_judgment_made_before_the_results_says_so_in_its_own_words(conn):
    cid = _campaign(conn)

    saved = drift.classify(conn, campaign_id=cid, subject=_claw(conn, cid),
                           classification="improvement", why="It drew a queue.",
                           classified_by="R. Vega")

    assert "no measured result" in saved["what_it_means"].lower()


def test_classifying_again_never_rewrites_the_first_one(conn):
    """§8.7's rule. "This looked like an improvement before the numbers came in, and like a
    degradation after" is the most interesting thing this table can hold, and overwriting
    destroys it."""
    cid = _campaign(conn)
    drift.classify(conn, campaign_id=cid, subject=_claw(conn, cid),
                   classification="improvement", why="It drew a queue.",
                   classified_by="R. Vega")
    core.add_metrics(conn, campaign_id=cid, structured={"footfall_uplift_pct": -3.0},
                     confirm=True)
    drift.classify(conn, campaign_id=cid, subject=_claw(conn, cid),
                   classification="degradation", why="Footfall fell.", classified_by="R. Vega")

    history = drift.history(conn, campaign_id=cid, subject=_claw(conn, cid))
    assert [h["classification"] for h in history] == ["improvement", "degradation"]
    assert drift.for_campaign(conn, cid)["counts"]["degradation"] == 1
    assert drift.for_campaign(conn, cid)["counts"]["improvement"] == 0, "the latest stands"


def test_the_summary_says_when_a_reading_changed(conn):
    cid = _campaign(conn)
    drift.classify(conn, campaign_id=cid, subject=_claw(conn, cid),
                   classification="improvement", why="It drew a queue.",
                   classified_by="R. Vega")
    core.add_metrics(conn, campaign_id=cid, structured={"footfall_uplift_pct": -3.0},
                     confirm=True)
    drift.classify(conn, campaign_id=cid, subject=_claw(conn, cid),
                   classification="degradation", why="Footfall fell.", classified_by="R. Vega")

    said = drift.for_campaign(conn, cid)["what_it_means"]
    assert "changed" in said.lower()


# ── it reaches the comparison ───────────────────────────────────────────────

def test_the_comparison_shows_what_has_been_classified_and_what_has_not(conn, tmp_path):
    cid = _campaign(conn)
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": _png(tmp_path, "brief.png", seed=3)})
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": _png(tmp_path, "shot.png", seed=91)},
                            phase="delivered", captured_on="2026-03-14")

    out = core.compare_execution(conn, campaign_id=cid)
    assert out["unclassified"] == 2
    assert out["next_actions"][0]["tool"] == "classify_drift"


def test_it_reaches_the_model_over_the_protocol(conn, tmp_path):
    import asyncio
    import json

    import mcp_server

    async def call(name, args):
        return json.loads((await mcp_server.mcp.call_tool(name, args)).content[0].text)

    cid = _campaign(conn)
    saved = asyncio.run(call("classify_drift", {
        "campaign_id": cid, "subject": _claw(conn, cid), "classification": "improvement",
        "why": "Unbriefed, and it drew the queue the photo booth was meant to.",
        "classified_by": "R. Vega"}))

    assert saved["basis"] == "judged"
    assert saved["outcome_known"] is False
    assert "claw" in saved["item"].lower(), "the record says what it is about, not a uuid"


def test_a_phrase_is_refused_with_what_to_pass_instead(conn):
    """The defect the review named: keyed on a free string, a person answering in their own
    words files against nothing, the original difference stays unjudged, and the offer re-fires
    on the same uuid forever."""
    cid = _campaign(conn)

    with pytest.raises(ValueError) as e:
        drift.classify(conn, campaign_id=cid, subject="A claw machine",
                       classification="improvement", why="It drew a queue.",
                       classified_by="R. Vega")
    said = str(e.value)
    assert "commitment_id" in said and "asset_id" in said
    assert drift.for_campaign(conn, cid)["counts"]["improvement"] == 0


# ── the offer and the count are the same list ───────────────────────────────

def _stage_all(conn, campaign_id):
    """Give every commitment a distinct text vector.

    The `hash` provider returns zeros from `embed_text` — honestly, since it has no shared
    text/image space — so without this every verdict is `unchecked`, no promise is ever drift,
    and the branch these tests are about is unreachable (D124).
    """
    import random

    import commitments
    import config
    import vectorstore

    vectorstore.init(conn, space="commitment")
    for n, promise in enumerate(commitments.for_campaign(conn, campaign_id)):
        rng = random.Random(1000 + n)
        vec = [rng.uniform(-1, 1) for _ in range(config.CLIP_EMBED_DIM)]
        vectorstore.add(conn, f"commitment:{promise['id']}", vec, space="commitment")


def _with_photographs(conn, tmp_path):
    cid = _campaign(conn)
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": _png(tmp_path, "brief.png", seed=3)})
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": _png(tmp_path, "shot.png", seed=91)},
                            phase="delivered", captured_on="2026-03-14")
    _stage_all(conn, cid)
    return cid


def test_answering_the_offer_removes_it_and_decrements_the_count(conn, tmp_path):
    """The count and the offer were built from different lists — events keyed on asset ids, the
    offer on commitment ids — so answering the question the server asked moved nothing and the
    next call asked it again with the same prefilled subject. A nag that cannot be satisfied is
    worse than no offer, because the number beside it reads as unexplained failure."""
    cid = _with_photographs(conn, tmp_path)

    before = core.compare_execution(conn, campaign_id=cid)
    offer = before["next_actions"][0]
    assert offer["tool"] == "classify_drift"

    # Answered with the offer's OWN prefilled arguments, which is the only way a caller who
    # takes the offer at its word would answer it.
    drift.classify(conn, classification="improvement",
                   why="The queue formed somewhere else and formed anyway.",
                   classified_by="R. Vega", **offer["prefilled_args"])

    after = core.compare_execution(conn, campaign_id=cid)
    assert after["unclassified"] == before["unclassified"] - 1
    assert offer["prefilled_args"]["subject"] not in {
        a["prefilled_args"]["subject"] for a in after["next_actions"]}


def test_the_count_and_the_offer_are_drawn_from_one_list(conn, tmp_path):
    cid = _with_photographs(conn, tmp_path)

    out = core.compare_execution(conn, campaign_id=cid)

    assert out["unclassified"] == len(out["unclassified_items"])
    assert out["next_actions"][0]["prefilled_args"]["subject"] == out["unclassified_items"][0][
        "subject"]


def test_a_promise_that_is_not_visible_is_drift_to_judge(conn, tmp_path):
    """§9.3 finds the promises; §9.4 asks what they meant. A promise the deck named and the
    photographs do not show is the most readable drift subject there is."""
    cid = _with_photographs(conn, tmp_path)

    out = core.compare_execution(conn, campaign_id=cid)

    claw = _claw(conn, cid)
    assert claw in {i["subject"] for i in out["unclassified_items"]}
    assert "claw" in out["next_actions"][0]["label"].lower()


def test_a_promise_that_was_visible_is_never_offered_as_drift(conn, tmp_path):
    """Offering to judge a difference that is not there asks for an explanation of nothing, and
    a question with no true answer gets a made-up one."""
    cid = _with_photographs(conn, tmp_path)
    delivered = [a for a in store.get_assets_for_campaign(conn, cid)
                 if a["phase"] == "delivered"]
    import vectorstore
    seen = vectorstore.get_many(conn, [delivered[0]["id"]], space="asset")[delivered[0]["id"]]
    vectorstore.add(conn, f"commitment:{_claw(conn, cid)}", seen, space="commitment")

    out = core.compare_execution(conn, campaign_id=cid)

    assert _claw(conn, cid) not in {i["subject"] for i in out["unclassified_items"]}
