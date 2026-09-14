"""
§9.2: `compare_execution` — what shipped, and how far it moved.

The review: *"The machinery for this is already built and verified. Fingerprints answer what
shipped: delivered assets that match a proposed one are **as briefed**; proposed assets with no
match **never appeared**; delivered assets matching nothing are **new**. Visual similarity
answers how far it moved: the distance between the proposed and delivered creative sets is a
drift score, not a vibe. Return all three lists plus the score, with the matching evidence
attached per item — the same evidence discipline as everything else in the product."*

**Two different instruments answering two different questions**, and keeping them apart is the
whole of it. A fingerprint is an identity claim — this photograph is that render — and it is
exact enough to state. A CLIP distance is a claim about resemblance, which is a number and not
a verdict. Reporting one as the other is how "the creative changed" becomes an assertion nobody
can check.

**The signature in the plan is `compare_execution(campaign_id, delivered_assets[])`, and this
takes only the campaign.** §9.1 said "everything below falls out of that one field", and it
does: once an asset carries its phase, which images are the brief and which came back is
already on file. A second inline upload path would be a second way to attach an image, with its
own resolution rules and its own bugs.

**"Nothing has come back" is not "nothing drifted".** A campaign with no delivered assets has
not been checked, and reporting zero drift for it would be the confident unfounded claim in its
purest form — a score computed over an empty set, with the server's authority behind it.
"""
import pytest

import core
import store


def _campaign(conn, title="Colombia v1"):
    return core.ingest_campaign(conn, title=title, market="LATAM", status="concluded",
                                detail="A launch.")["campaign_id"]


def _png(tmp_path, name, seed=1):
    """An image with STRUCTURE. A perceptual hash measures structure, so flat colour fields
    all hash identically — every fixture in the first version of this file was the same image
    as far as pHash was concerned, and the tests passed for the wrong reason."""
    import random

    from PIL import Image

    rng = random.Random(seed)
    img = Image.new("RGB", (64, 64), (250, 250, 250))
    pixels = img.load()
    for _ in range(700):
        x, y = rng.randrange(64), rng.randrange(64)
        block = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        for dx in range(4):
            for dy in range(4):
                if x + dx < 64 and y + dy < 64:
                    pixels[x + dx, y + dy] = block
    path = tmp_path / name
    img.save(path)
    return str(path)


def _brief(conn, cid, path):
    return core.ingest_image_asset(conn, campaign_id=cid, asset_ref={"path": path})


def _delivered(conn, cid, path, when="2026-03-14"):
    return core.ingest_image_asset(conn, campaign_id=cid, asset_ref={"path": path},
                                   phase="delivered", captured_on=when)


# ── the three lists ─────────────────────────────────────────────────────────

def test_a_delivered_photo_matching_a_briefed_render_is_as_briefed(conn, tmp_path):
    """A fingerprint is an IDENTITY claim — this photograph is that render — and it is exact
    enough to state plainly."""
    cid = _campaign(conn)
    same = _png(tmp_path, "hero.png")
    _brief(conn, cid, same)
    _delivered(conn, cid, same)

    out = core.compare_execution(conn, campaign_id=cid)

    assert [item["file"] for item in out["as_briefed"]]
    assert out["never_appeared"] == [] and out["new"] == []


def test_a_briefed_asset_nothing_matches_never_appeared(conn, tmp_path):
    cid = _campaign(conn)
    _brief(conn, cid, _png(tmp_path, "claw.png", seed=58))
    _delivered(conn, cid, _png(tmp_path, "other.png", seed=72))

    out = core.compare_execution(conn, campaign_id=cid)

    assert len(out["never_appeared"]) == 1
    assert len(out["new"]) == 1
    assert out["as_briefed"] == []


def test_a_delivered_photo_matching_nothing_is_new(conn, tmp_path):
    cid = _campaign(conn)
    _brief(conn, cid, _png(tmp_path, "a.png", seed=36))
    _delivered(conn, cid, _png(tmp_path, "b.png", seed=58))

    assert len(core.compare_execution(conn, campaign_id=cid)["new"]) == 1


def test_every_item_carries_the_evidence_it_rests_on(conn, tmp_path):
    """"With the matching evidence attached per item — the same evidence discipline as
    everything else in the product." A classification with nothing behind it is the assertion
    this library refuses everywhere else."""
    cid = _campaign(conn)
    same = _png(tmp_path, "hero.png")
    _brief(conn, cid, same)
    _delivered(conn, cid, same)

    matched = core.compare_execution(conn, campaign_id=cid)["as_briefed"][0]
    assert matched["matched_asset_id"]
    assert matched["distance"] == 0
    assert matched["threshold"] == pytest.approx(core.config.PHASH_MATCH_THRESHOLD)


# ── the drift score ─────────────────────────────────────────────────────────

def test_the_score_is_a_distance_not_a_vibe(conn, tmp_path):
    """"The distance between the proposed and delivered creative sets is a drift score, not a
    vibe." It is computed, so it says so."""
    cid = _campaign(conn)
    _brief(conn, cid, _png(tmp_path, "a.png", seed=36))
    _delivered(conn, cid, _png(tmp_path, "b.png", seed=58))

    out = core.compare_execution(conn, campaign_id=cid)

    assert 0.0 <= out["drift"]["score"] <= 1.0
    assert out["drift"]["basis"] == "computed"


def test_identical_creative_drifts_less_than_different_creative(conn, tmp_path):
    """The only claim a distance can carry on its own: this pair moved further than that one.
    What that MEANS is §9.4's, and it is a judgment."""
    same_file = _png(tmp_path, "same.png", seed=81)
    a = _campaign(conn, "Same")
    _brief(conn, a, same_file)
    _delivered(conn, a, same_file)

    b = _campaign(conn, "Different")
    _brief(conn, b, _png(tmp_path, "brief.png", seed=69))
    _delivered(conn, b, _png(tmp_path, "shot.png", seed=93))

    assert core.compare_execution(conn, campaign_id=a)["drift"]["score"] < \
        core.compare_execution(conn, campaign_id=b)["drift"]["score"]


def test_it_never_says_what_the_drift_means(conn, tmp_path):
    """§9.4 classifies drift as improvement / neutral / degradation and marks it `judged`.
    A distance cannot know which, and a server that guessed would be asserting a verdict from
    a cosine."""
    cid = _campaign(conn)
    _brief(conn, cid, _png(tmp_path, "a.png", seed=36))
    _delivered(conn, cid, _png(tmp_path, "b.png", seed=58))

    said = core.compare_execution(conn, campaign_id=cid)["drift"]["what_it_means"].lower()
    assert "worse" not in said and "better" not in said
    assert "how far" in said or "distance" in said


# ── what it refuses to claim ────────────────────────────────────────────────

def test_nothing_come_back_is_not_zero_drift(conn, tmp_path):
    """A score computed over an empty set, with the server's authority behind it, is the
    confident unfounded claim in its purest form."""
    cid = _campaign(conn)
    _brief(conn, cid, _png(tmp_path, "a.png"))

    out = core.compare_execution(conn, campaign_id=cid)

    assert out["drift"]["score"] is None
    assert out["status"] == "nothing_to_check"
    assert "has come back" in out["what_it_means"]


def test_a_campaign_with_no_brief_creative_says_so_too(conn, tmp_path):
    """Photographs with nothing to compare them against are not "all new" in any useful
    sense — there was never a brief to drift from."""
    cid = _campaign(conn)
    _delivered(conn, cid, _png(tmp_path, "a.png"))

    out = core.compare_execution(conn, campaign_id=cid)
    assert out["status"] == "nothing_to_check"
    assert "nothing briefed" in out["what_it_means"].lower()


def test_an_unfingerprinted_asset_is_said_rather_than_silently_dropped(conn, tmp_path):
    """§2.1's rule. An image that could not be hashed cannot be matched, and leaving it out of
    all three lists would make the totals lie about what was examined."""
    cid = _campaign(conn)
    _brief(conn, cid, _png(tmp_path, "a.png"))
    _delivered(conn, cid, _png(tmp_path, "b.png", seed=58))
    conn.execute("DELETE FROM asset_fingerprints")
    conn.commit()

    out = core.compare_execution(conn, campaign_id=cid)
    assert out["not_compared"]
    assert any(w["code"] == "assets_not_fingerprinted" for w in out["warnings"])


def test_the_score_is_absent_rather_than_zero_when_vision_is_off(conn, tmp_path, monkeypatch):
    """§6.4's lesson in the other half of this file: an outage that reads as a clean result is
    worse than an outage that says so."""
    cid = _campaign(conn)
    _brief(conn, cid, _png(tmp_path, "a.png", seed=36))
    _delivered(conn, cid, _png(tmp_path, "b.png", seed=58))
    conn.execute("DELETE FROM asset_vectors")
    conn.commit()

    out = core.compare_execution(conn, campaign_id=cid)
    assert out["drift"]["score"] is None
    assert out["drift"]["code"] == "not_visually_indexed"
    assert out["as_briefed"] is not None, "the fingerprint half still worked"


def test_a_campaign_that_does_not_exist_is_refused(conn):
    with pytest.raises(ValueError) as e:
        core.compare_execution(conn, campaign_id="camp_invented")
    assert "not a record" in str(e.value)


def test_it_reaches_the_model_over_the_protocol(conn, tmp_path):
    import asyncio
    import json

    import mcp_server

    async def call(name, args):
        return json.loads((await mcp_server.mcp.call_tool(name, args)).content[0].text)

    cid = _campaign(conn)
    _brief(conn, cid, _png(tmp_path, "claw.png", seed=58))
    same = _png(tmp_path, "booth.png", seed=27)
    _brief(conn, cid, same)
    _delivered(conn, cid, same)

    out = asyncio.run(call("compare_execution", {"campaign_id": cid}))

    assert len(out["as_briefed"]) == 1
    assert len(out["never_appeared"]) == 1
    assert out["drift"]["basis"] == "computed"


def test_the_score_is_given_a_yardstick(conn, tmp_path):
    """A raw visual distance is unreadable: two completely unrelated images measured 0.0009
    apart in a protocol run, and a marketer reading that as a percentage concludes "no drift"
    about creative that shares nothing. The brief's own internal spread is the only yardstick
    the library has that does not come from somewhere else's data."""
    cid = _campaign(conn)
    for n, seed in enumerate((11, 23, 37)):
        _brief(conn, cid, _png(tmp_path, f"b{n}.png", seed=seed))
    _delivered(conn, cid, _png(tmp_path, "shot.png", seed=91))

    drift = core.compare_execution(conn, campaign_id=cid)["drift"]

    assert drift["brief_spread"] > 0
    assert drift["relative_to_brief_spread"] is not None
    assert "each\nother" in drift["what_it_means"].replace(" ", "\n") or \
        "EACH" in drift["what_it_means"]


def test_one_briefed_image_has_no_spread_and_says_so(conn, tmp_path):
    """A single image cannot be spread out. Inventing a yardstick from it — or dividing by
    zero — would be worse than saying the number stands alone."""
    cid = _campaign(conn)
    _brief(conn, cid, _png(tmp_path, "only.png", seed=11))
    _delivered(conn, cid, _png(tmp_path, "shot.png", seed=91))

    drift = core.compare_execution(conn, campaign_id=cid)["drift"]

    assert drift["relative_to_brief_spread"] is None
    assert "stands alone" in drift["what_it_means"]


# ── reachable, which is the pattern this project keeps missing (D116) ────────

def test_the_photograph_that_makes_the_comparison_possible_offers_it(conn, tmp_path):
    """The fourth time this project would have shipped a tool referenced by nothing but its
    own definition (§8.3's gate, §8.6's `note_correction`, §8.7's `replay_rules`, this).
    Uploading a delivered photograph is the moment `compare_execution` stops being empty."""
    cid = _campaign(conn)
    _brief(conn, cid, _png(tmp_path, "claw.png", seed=11))

    landed = _delivered(conn, cid, _png(tmp_path, "shot.png", seed=23))

    offer = landed["next_actions"][0]
    assert offer["tool"] == "compare_execution"
    assert offer["prefilled_args"]["campaign_id"] == cid


def test_a_photograph_with_nothing_briefed_offers_no_comparison(conn, tmp_path):
    """An offer that leads to `nothing_to_check` is an offer that wastes the user's yes."""
    cid = _campaign(conn)
    landed = _delivered(conn, cid, _png(tmp_path, "shot.png", seed=23))
    assert landed["next_actions"] == []


def test_briefed_creative_offers_nothing(conn, tmp_path):
    cid = _campaign(conn)
    assert _brief(conn, cid, _png(tmp_path, "a.png"))["next_actions"] == []


def test_a_campaign_judged_on_a_brief_alone_is_a_gap(conn, tmp_path):
    """"A library that learns from briefs while measuring executions is learning from the wrong
    document, and has no way to notice." This is the noticing, on the surface that exists to
    say what the library is missing."""
    cid = _campaign(conn)
    _brief(conn, cid, _png(tmp_path, "claw.png", seed=11))
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.1}, confirm=True)

    found = [g for g in core.gaps(conn)["gaps"] if g["code"] == "execution_never_checked"]
    assert found
    assert "Colombia v1" in found[0]["what"]
    assert found[0]["next_actions"][0]["prefilled_args"]["phase"] == "delivered"


def test_a_campaign_whose_execution_was_checked_is_not_a_gap(conn, tmp_path):
    cid = _campaign(conn)
    _brief(conn, cid, _png(tmp_path, "claw.png", seed=11))
    _delivered(conn, cid, _png(tmp_path, "shot.png", seed=23))
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.1}, confirm=True)

    assert [g for g in core.gaps(conn)["gaps"]
            if g["code"] == "execution_never_checked"] == []


def test_a_campaign_with_no_briefed_creative_is_not_this_gap(conn, tmp_path):
    """There is nothing to compare against, so "nobody checked what ran" is not the thing
    that is missing — the creative is."""
    cid = _campaign(conn)
    core.add_metrics(conn, campaign_id=cid, structured={"roas": 3.1}, confirm=True)

    assert [g for g in core.gaps(conn)["gaps"]
            if g["code"] == "execution_never_checked"] == []


def test_two_photographs_of_one_briefed_element_count_it_once(conn, tmp_path):
    """A brief promising three things, with three photos all of the claw machine, read as
    "three briefed elements appeared" when one did. Counting the extra shots as `new` would be
    equally false — "matches nothing that was briefed" is not true of them."""
    cid = _campaign(conn)
    claw = _png(tmp_path, "claw.png", seed=11)
    _brief(conn, cid, claw)
    _brief(conn, cid, _png(tmp_path, "booth.png", seed=23))
    _delivered(conn, cid, claw)
    _delivered(conn, cid, claw)

    out = core.compare_execution(conn, campaign_id=cid)

    assert len(out["as_briefed"]) == 1
    assert len(out["another_view"]) == 1
    assert out["new"] == []
    assert len(out["never_appeared"]) == 1
    assert "already counted" in out["what_it_means"]


def test_the_brief_spread_is_derived_from_the_images(conn, tmp_path):
    """A constant would satisfy "greater than zero" and mean nothing. Three near-identical
    renders sit closer together than three unrelated ones, and the yardstick has to move
    with them or it is not a yardstick."""
    tight = _campaign(conn, "Tight")
    one = _png(tmp_path, "one.png", seed=11)
    for n in range(3):
        _brief(conn, tight, one)
    _delivered(conn, tight, _png(tmp_path, "t-shot.png", seed=91))

    loose = _campaign(conn, "Loose")
    for n, seed in enumerate((13, 47, 83)):
        _brief(conn, loose, _png(tmp_path, f"l{n}.png", seed=seed))
    _delivered(conn, loose, _png(tmp_path, "l-shot.png", seed=91))

    assert core.compare_execution(conn, campaign_id=tight)["drift"]["brief_spread"] < \
        core.compare_execution(conn, campaign_id=loose)["drift"]["brief_spread"]


def test_a_brief_whose_images_all_look_alike_is_not_a_brief_with_one_image(conn, tmp_path):
    """Zero spread is a MEASUREMENT — every briefed image looks the same — and no spread is
    the absence of one. Collapsing them made a tightly art-directed brief indistinguishable
    from a brief carrying a single render."""
    tight = _campaign(conn, "Tight")
    one = _png(tmp_path, "one.png", seed=11)
    for _ in range(3):
        _brief(conn, tight, one)
    _delivered(conn, tight, _png(tmp_path, "t.png", seed=91))

    single = _campaign(conn, "Single")
    _brief(conn, single, _png(tmp_path, "s.png", seed=11))
    _delivered(conn, single, _png(tmp_path, "s2.png", seed=91))

    assert core.compare_execution(conn, campaign_id=tight)["drift"]["brief_spread"] == 0.0
    assert core.compare_execution(conn, campaign_id=single)["drift"]["brief_spread"] is None


# ── what the reviews found ──────────────────────────────────────────────────

def test_the_score_does_not_change_because_more_photos_arrived(conn, tmp_path):
    """Measured: the centroid version halved — 0.84 to 0.47 — purely because somebody uploaded
    eight photographs of one shoot instead of one. A centroid contracts toward the mean as a
    set grows, so a score that falls when you supply more evidence is not a measurement."""
    seen = []
    for count in (1, 3, 8):
        cid = _campaign(conn, f"C{count}")
        for n, seed in enumerate((11, 23, 37)):
            _brief(conn, cid, _png(tmp_path, f"b{count}-{n}.png", seed=seed))
        for n in range(count):
            _delivered(conn, cid, _png(tmp_path, f"d{count}-{n}.png", seed=60 + n))
        seen.append(core.compare_execution(conn, campaign_id=cid)["drift"]["score"])

    assert max(seen) - min(seen) < 0.05, seen


def test_a_brief_of_near_identical_images_gives_no_yardstick(conn, tmp_path):
    """A vanishing spread drove the ratio to 442 — a number that looks like precision and is
    noise. And the sentence for it said "there is only one briefed image", which reproduced
    with two."""
    cid = _campaign(conn)
    one = _png(tmp_path, "hero.png", seed=11)
    _brief(conn, cid, one)
    _brief(conn, cid, one)
    _delivered(conn, cid, _png(tmp_path, "shot.png", seed=91))

    drift = core.compare_execution(conn, campaign_id=cid)["drift"]
    assert drift["relative_to_brief_spread"] is None
    assert "only one briefed image" not in drift["what_it_means"]
    assert "essentially the same" in drift["what_it_means"]


def test_what_did_not_match_says_what_it_nearly_matched(conn, tmp_path):
    """"With the matching evidence attached per item." Two of the three lists carried a bare
    sentence, while the matching loop had already computed every distance and thrown the
    near-misses away. A negative with no number behind it is the assertion this product
    refuses everywhere else."""
    cid = _campaign(conn)
    _brief(conn, cid, _png(tmp_path, "claw.png", seed=11))
    _delivered(conn, cid, _png(tmp_path, "crowd.png", seed=57))

    out = core.compare_execution(conn, campaign_id=cid)
    assert out["never_appeared"][0]["nearest"]["distance"] > \
        out["never_appeared"][0]["threshold"]
    assert out["new"][0]["nearest"]["file"]


def test_a_photograph_of_a_built_thing_is_offered_as_a_resemblance(conn, tmp_path):
    """A fingerprint answers "same image FILE" — that is what `images.py`'s docstring scopes it
    to. A photograph of a claw machine that was actually built will not land within Hamming 8
    of the render of it, so pHash alone reports "0 as briefed, 3 never appeared, 14 new" about
    a perfectly executed campaign and stamps `computed` on it."""
    cid = _campaign(conn)
    _brief(conn, cid, _png(tmp_path, "render.png", seed=11))
    # Near, not identical: the same subject photographed rather than rendered.
    _delivered(conn, cid, _png(tmp_path, "photo.png", seed=11 + 0))
    # A real hash, far from the render's — not `ffffffffffffffff`, which is all-bits-set and
    # is now correctly refused as carrying no structure.
    conn.execute("UPDATE asset_fingerprints SET phash = '0f0f0f0f0f0f0f0f' "
                 "WHERE asset_id IN (SELECT id FROM assets WHERE phase = 'delivered')")
    conn.commit()

    out = core.compare_execution(conn, campaign_id=cid)

    assert out["as_briefed"] == [], "the fingerprints genuinely do not match"
    suggested = out["never_appeared"][0].get("looks_like")
    assert suggested, "and resemblance is the only thing left that can say anything"
    assert suggested["basis"] == "heuristic", "which is a threshold somebody chose"
    assert "question for somebody" in suggested["what_it_means"]


def test_a_resemblance_never_becomes_a_match(conn, tmp_path):
    """Promoting one to `as_briefed` would put the server's `computed` authority behind a guess
    about whether a thing was built."""
    cid = _campaign(conn)
    _brief(conn, cid, _png(tmp_path, "render.png", seed=11))
    _delivered(conn, cid, _png(tmp_path, "photo.png", seed=11))
    # A real hash, far from the render's — not `ffffffffffffffff`, which is all-bits-set and
    # is now correctly refused as carrying no structure.
    conn.execute("UPDATE asset_fingerprints SET phash = '0f0f0f0f0f0f0f0f' "
                 "WHERE asset_id IN (SELECT id FROM assets WHERE phase = 'delivered')")
    conn.commit()

    out = core.compare_execution(conn, campaign_id=cid)
    assert len(out["never_appeared"]) == 1 and len(out["new"]) == 1
    assert out["as_briefed"] == []


def test_the_empty_answer_offers_the_fix_rather_than_describing_it(conn, tmp_path):
    """C12 replaced prose-instead-of-an-offer once already, and this is the best place in the
    product to make it: the user is looking at "nobody has checked what ran" at the moment they
    could fix it."""
    cid = _campaign(conn)
    _brief(conn, cid, _png(tmp_path, "a.png"))

    out = core.compare_execution(conn, campaign_id=cid)
    assert out["next_actions"][0]["tool"] == "upload_image_asset"
    assert out["next_actions"][0]["prefilled_args"]["phase"] == "delivered"
    assert out["title"] == "Colombia v1"


def test_several_photographs_go_up_in_one_call(conn, tmp_path):
    """Fourteen photographs were fourteen uploads and fourteen tool calls — and
    `ingest_campaign`'s own comment already records that a manual call per image "isn't a
    workflow anyone would actually use"."""
    cid = _campaign(conn)
    _brief(conn, cid, _png(tmp_path, "claw.png", seed=11))

    out = core.ingest_image_assets(
        conn, campaign_id=cid,
        asset_refs=[{"path": _png(tmp_path, f"s{n}.png", seed=40 + n)} for n in range(4)],
        phase="delivered", captured_on="2026-03-14")

    assert out["stored"] == 4
    assert len(store.assets_in_phase(conn, cid, "delivered")) == 4
    assert out["next_actions"][0]["tool"] == "compare_execution"


def test_one_unreadable_image_does_not_stop_the_others(conn, tmp_path):
    cid = _campaign(conn)
    out = core.ingest_image_assets(
        conn, campaign_id=cid,
        asset_refs=[{"path": "/nope/missing.png"},
                    {"path": _png(tmp_path, "good.png", seed=5)}],
        phase="delivered", captured_on="2026-03-14")

    assert out["stored"] == 1
    assert len(out["failed"]) == 1


def test_a_library_that_never_embedded_an_image_degrades_rather_than_crashing(conn, tmp_path):
    """Reading `asset_vectors` when no image was ever embedded raised OperationalError — not a
    ValueError, so it reached the model as "Error executing tool" with the reason discarded, on
    exactly the install where CLIP never ran."""
    cid = _campaign(conn)
    _brief(conn, cid, _png(tmp_path, "a.png", seed=11))
    _delivered(conn, cid, _png(tmp_path, "b.png", seed=57))
    conn.execute("DROP TABLE IF EXISTS asset_vectors")
    conn.commit()

    out = core.compare_execution(conn, campaign_id=cid)
    assert out["drift"]["score"] is None
    assert out["never_appeared"], "the fingerprint half still worked"


def test_a_half_indexed_set_says_so_instead_of_calling_itself_measured(conn, tmp_path):
    """Firing only when a side is ENTIRELY unindexed meant a half-indexed set computed a score
    from whatever subset had vectors and labelled it `measured`, with a count buried in
    `compared` as the only tell."""
    cid = _campaign(conn)
    _brief(conn, cid, _png(tmp_path, "a.png", seed=11))
    _brief(conn, cid, _png(tmp_path, "b.png", seed=23))
    _delivered(conn, cid, _png(tmp_path, "c.png", seed=57))
    one = store.assets_in_phase(conn, cid, "proposed")[0]["id"]
    conn.execute("DELETE FROM asset_vectors WHERE vector_id = ?", (one,))
    conn.commit()

    out = core.compare_execution(conn, campaign_id=cid)
    assert out["drift"]["code"] == "partly_measured"
    assert out["warnings"]


def test_a_blank_frame_is_not_reported_as_a_briefed_thing_built(conn, tmp_path):
    """A perceptual hash measures STRUCTURE, and an image with none hashes to the same value
    as every other image with none — flat red, flat blue and a blown-out frame all produce
    `8000000000000000`. Decks routinely carry solid-fill rectangles, so without this a blank
    wall in a delivered photograph reports a colour swatch as having been built."""
    from PIL import Image

    cid = _campaign(conn)
    swatch = tmp_path / "swatch.png"
    Image.new("RGB", (64, 64), (200, 30, 30)).save(swatch)
    blank = tmp_path / "blank.png"
    Image.new("RGB", (64, 64), (20, 20, 220)).save(blank)

    _brief(conn, cid, str(swatch))
    _delivered(conn, cid, str(blank))

    out = core.compare_execution(conn, campaign_id=cid)

    assert out["as_briefed"] == [], "two different flat fills are not the same image"
    assert len(out["not_compared"]) == 2
    assert any("flat fill" in w["detail"] for w in out["warnings"])


def test_one_briefed_image_on_two_slides_is_one_briefed_thing(conn, tmp_path):
    """A deck puts the hero on the cover and again on a detail slide, and the extractor dedupes
    by sha256 — two files, one image. Treated as two elements, a photograph of it satisfied one
    and the other read as NEVER APPEARED: the list whose entire claim is identity, saying a
    briefed element was missing when it demonstrably was not."""
    from PIL import Image

    cid = _campaign(conn)
    hero = _png(tmp_path, "hero.png", seed=11)
    smaller = tmp_path / "hero_small.png"
    with Image.open(hero) as img:
        img.resize((48, 48)).save(smaller)

    _brief(conn, cid, hero)
    _brief(conn, cid, str(smaller))
    _delivered(conn, cid, hero)

    out = core.compare_execution(conn, campaign_id=cid)

    assert len(out["as_briefed"]) == 1
    assert out["never_appeared"] == []
    assert out["as_briefed"][0]["also_briefed_as"], "and it says the brief showed it twice"


def test_the_answer_does_not_depend_on_which_photo_was_uploaded_first(conn, tmp_path):
    """Taking each photograph's closest unclaimed render in turn made the same four images
    produce opposite verdicts on the same briefed element, decided by upload order."""
    def run(order):
        cid = _campaign(conn, f"Order {order}")
        for n, seed in enumerate((11, 23)):
            _brief(conn, cid, _png(tmp_path, f"b{order}{n}.png", seed=seed))
        for n in order:
            _delivered(conn, cid, _png(tmp_path, f"d{order}{n}.png", seed=41 + n))
        out = core.compare_execution(conn, campaign_id=cid)
        return len(out["as_briefed"]), len(out["never_appeared"]), len(out["new"])

    assert run((0, 1)) == run((1, 0))


def test_a_match_at_exactly_the_threshold_is_a_match(conn, tmp_path):
    """The boundary the whole classification turns on, and it was untested in both
    directions."""
    cid = _campaign(conn)
    _brief(conn, cid, _png(tmp_path, "a.png", seed=11))
    _delivered(conn, cid, _png(tmp_path, "b.png", seed=23))
    briefed = store.assets_in_phase(conn, cid, "proposed")[0]["id"]
    shot = store.assets_in_phase(conn, cid, "delivered")[0]["id"]
    conn.execute("UPDATE asset_fingerprints SET phash = '0f0f0f0f0f0f0f00' WHERE asset_id = ?",
                 (briefed,))
    # Exactly PHASH_MATCH_THRESHOLD bits apart.
    conn.execute("UPDATE asset_fingerprints SET phash = '0f0f0f0f0f0f0fff' WHERE asset_id = ?",
                 (shot,))
    conn.commit()
    assert core.images.hamming_distance("0f0f0f0f0f0f0f00", "0f0f0f0f0f0f0fff") == \
        core.config.PHASH_MATCH_THRESHOLD

    assert len(core.compare_execution(conn, campaign_id=cid)["as_briefed"]) == 1


def test_a_match_one_past_the_threshold_is_not(conn, tmp_path):
    cid = _campaign(conn)
    _brief(conn, cid, _png(tmp_path, "a.png", seed=11))
    _delivered(conn, cid, _png(tmp_path, "b.png", seed=23))
    briefed = store.assets_in_phase(conn, cid, "proposed")[0]["id"]
    shot = store.assets_in_phase(conn, cid, "delivered")[0]["id"]
    conn.execute("UPDATE asset_fingerprints SET phash = '0f0f0f0f0f0f0f00' WHERE asset_id = ?",
                 (briefed,))
    conn.execute("UPDATE asset_fingerprints SET phash = '0f0f0f0f0f0f1fff' WHERE asset_id = ?",
                 (shot,))
    conn.commit()

    assert core.compare_execution(conn, campaign_id=cid)["as_briefed"] == []


def test_a_photograph_is_matched_to_its_closest_brief_not_its_first(conn, tmp_path):
    """"Its CLOSEST unclaimed" is the docstring's claim and was untested — first-match passes
    every other test in this file."""
    cid = _campaign(conn)
    far = _brief(conn, cid, _png(tmp_path, "far.png", seed=11))["asset_id"]
    near = _brief(conn, cid, _png(tmp_path, "near.png", seed=23))["asset_id"]
    shot = _delivered(conn, cid, _png(tmp_path, "shot.png", seed=37))["asset_id"]
    # Far enough apart to be two briefed ELEMENTS (11 bits), both within the threshold of the
    # shot — otherwise the clustering correctly treats them as one image and there is no
    # closest to pick.
    for aid, value in ((far, "0f0f0f0f00ff0000"), (near, "0f0f0f0f00000007"),
                       (shot, "0f0f0f0f00000000")):
        conn.execute("UPDATE asset_fingerprints SET phash = ? WHERE asset_id = ?", (value, aid))
    conn.commit()

    matched = core.compare_execution(conn, campaign_id=cid)["as_briefed"][0]
    assert matched["matched_asset_id"] == near


def test_the_protocol_default_is_proposed(conn, tmp_path):
    """The code's own comment calls this the load-bearing decision, and the protocol test
    passed `phase` explicitly — so the default the model actually meets was untested."""
    import asyncio
    import json

    import mcp_server

    cid = _campaign(conn)
    got = json.loads(asyncio.run(mcp_server.mcp.call_tool(
        "upload_image_asset",
        {"campaign_id": cid, "asset_ref": {"path": _png(tmp_path, "x.png")}})).content[0].text)

    assert got["phase"] == "proposed"


def test_the_evidence_fields_carry_the_real_values(conn, tmp_path):
    """Asserted only as truthy, `matched_asset_id` could be the shot's own id and `distance`
    could be a constant zero."""
    cid = _campaign(conn)
    same = _png(tmp_path, "hero.png", seed=11)
    briefed = _brief(conn, cid, same)["asset_id"]
    shot = _delivered(conn, cid, same)["asset_id"]

    out = core.compare_execution(conn, campaign_id=cid)
    item = out["as_briefed"][0]

    assert item["asset_id"] == shot
    assert item["matched_asset_id"] == briefed
    assert item["captured_on"] == "2026-03-14"
    assert out["delivered_count"] == 1 and out["briefed_count"] == 1


def test_the_ratio_is_drift_over_spread_not_the_other_way_round(conn, tmp_path):
    cid = _campaign(conn)
    for n, seed in enumerate((11, 23, 37)):
        _brief(conn, cid, _png(tmp_path, f"b{n}.png", seed=seed))
    _delivered(conn, cid, _png(tmp_path, "shot.png", seed=91))

    drift = core.compare_execution(conn, campaign_id=cid)["drift"]
    assert drift["relative_to_brief_spread"] == pytest.approx(
        drift["score"] / drift["brief_spread"], rel=0.02)


def test_it_speaks_the_same_result_vocabulary_as_the_other_diff(conn, tmp_path):
    """D60: "so 'what changed' means one thing at both levels." `match` says what a
    classification RESTS on, which is the distinction this whole item is built around — an
    identity claim and a resemblance are not the same kind of statement."""
    cid = _campaign(conn)
    same = _png(tmp_path, "hero.png", seed=11)
    _brief(conn, cid, same)
    _brief(conn, cid, _png(tmp_path, "claw.png", seed=57))
    _delivered(conn, cid, same)

    out = core.compare_execution(conn, campaign_id=cid)

    assert out["counts"] == {"as_briefed": 1, "never_appeared": 1, "new": 0,
                             "another_view": 0, "not_compared": 0}
    assert out["as_briefed"][0]["match"] == "fingerprint"
    assert out["never_appeared"][0]["match"] == "none"
