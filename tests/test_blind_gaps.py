"""
§13.6 — two things that cannot see the record they are about.

    D122  `execution_never_checked` fires only for campaigns that ALREADY have briefed
          creative. Measured before the change: a concluded campaign with results and no
          assets at all is invisible to it, while one with a board on file is reported. Those
          are the campaigns with the LEAST evidence about what ran, and the gap built to
          notice "what ran was not what was briefed" cannot see them.
    D126  Commitment vectors are written with a bare `vectorstore.add`, where every other
          vector in this product goes through `core._add_vector`, which also calls
          `store.record_vector_model`. So after a CLIP weights change they are compared
          across models — §7.2's "a model upgrade is a visible migration rather than a silent
          re-ranking", silently. And `count_unreadable_vectors` iterates `("campaign",
          "asset")`, so the commitment space is outside the one check written to notice a
          vector table this process cannot read.

Both were filed as latent. D122 is not latent — it is wrong on the ordinary text-only upload,
today. D126 is latent until somebody swaps weights, and the cache is rebuildable, so what it
costs is a silent wrong answer rather than lost data. The difference is worth keeping in view:
one is a gap that does not fire, the other is a check that does not cover.
"""
import zlib

import pytest

import config
import core
import store
import vectorstore


@pytest.fixture
def clip_text(monkeypatch):
    """A text/image space to compare in.

    The bundled `hash` CLIP returns an all-ZERO text vector — it has no shared space, and
    `_look_for` correctly refuses to score in one that does not exist. So under the default
    fixture no commitment vector is ever written, and a test about what is recorded ALONGSIDE
    that vector would pass while the vector it is about never existed. This supplies a
    deterministic stand-in so the write path is actually taken; nothing below reads the numbers.
    """
    import clip_embed

    monkeypatch.setattr(clip_embed, "embed_text", lambda text, **kw: [
        ((zlib.crc32(f"{text}:{i}".encode()) % 199) + 1) / 200 for i in range(512)])
    return clip_embed


def _measured(conn, title, **kw):
    cid = core.ingest_campaign(conn, title=title, market="Peru", status="concluded",
                               confirm=True, **kw)["campaign_id"]
    core.add_metrics(conn, campaign_id=cid, detail="ROAS came in at 3.4.",
                     structured={"roas": 3.4}, metric_type="actual", confirm=True)
    return cid


# A promise the instrument can actually answer. "one colourway per creator" is refused by
# `_NOT_ANSWERABLE` — it is a claim about how many — and refused BEFORE the phrase is encoded,
# so a fixture using it never reaches the vector these tests are about.
_ANSWERABLE = "Creator posts show the product against a navy backdrop."


def _png(tmp_path, name="board.png"):
    from PIL import Image

    path = tmp_path / name
    Image.new("RGB", (48, 48), "navy").save(path)
    return path


# ---------------------------------------------------------------------------------------
# D122 — the campaigns with the least evidence are the ones it cannot see


def test_a_concluded_campaign_with_no_creative_at_all_is_reported(conn):
    """The gap asks "did what ran match what was briefed", and answered it only where briefed
    creative was already on file. A campaign uploaded as text, concluded, with results and no
    images at either phase, has the LEAST evidence of what actually ran — and was the one case
    the gap could not see."""
    bare = _measured(conn, "Bogota, text only",
                     deck_text="A store opening. Two hero shots and one film.")

    gap = next(g for g in core.gaps(conn)["gaps"]
               if g["code"] == "execution_never_checked")

    assert bare in gap["campaign_ids"], (
        "the campaign with nothing on file at all is not among those reported"
    )


def test_it_still_reports_a_campaign_whose_boards_are_on_file(conn, tmp_path):
    """The case it already saw, kept — briefed creative and nothing delivered is still the
    clearest form of "nobody checked what ran"."""
    briefed = _measured(conn, "Lima, with boards", detail="A store opening.")
    core.ingest_image_asset(conn, campaign_id=briefed,
                            asset_ref={"path": str(_png(tmp_path))}, phase="proposed")

    gap = next(g for g in core.gaps(conn)["gaps"]
               if g["code"] == "execution_never_checked")

    assert gap["counts"]["campaigns"] >= 1
    assert briefed in gap["campaign_ids"]
    # The two states are counted apart, because they need different things: this one can be
    # compared the moment the photographs arrive, and a record with nothing on file cannot.
    assert gap["counts"]["no_creative_at_all"] == 0, gap["counts"]

    bare = _measured(conn, "Bogota, text only", deck_text="A store opening. Two hero shots.")
    again = next(g for g in core.gaps(conn)["gaps"]
                 if g["code"] == "execution_never_checked")
    assert again["counts"] == {"campaigns": 2, "no_creative_at_all": 1,
                               "nothing_briefed": 0, "nothing_delivered": 1,
                               "never_compared": 0, "named": 2}, again["counts"]
    assert {briefed, bare} <= set(again["campaign_ids"])


def test_a_campaign_whose_delivered_shots_are_on_file_is_not_reported(conn, tmp_path):
    """And the gap closes when it should. A campaign with what ran on file has been checked —
    reporting it forever is the permanent complaint every gap in this product is written
    against."""
    done = _measured(conn, "Cusco, checked", detail="A store opening.")
    core.ingest_image_asset(conn, campaign_id=done,
                            asset_ref={"path": str(_png(tmp_path, "brief.png"))},
                            phase="proposed")
    core.ingest_image_asset(conn, campaign_id=done,
                            asset_ref={"path": str(_png(tmp_path, "ran.png"))},
                            phase="delivered")

    gap = next((g for g in core.gaps(conn)["gaps"]
                if g["code"] == "execution_never_checked"), None)

    named = {a["prefilled_args"].get("campaign_id") for a in (gap or {}).get("next_actions", [])}
    assert done not in named


def test_what_it_offers_fits_what_the_record_actually_has(conn):
    """§5.2's rule: an offer has to be one the tool accepts and the record can take. A
    campaign with no creative at all cannot be asked to compare execution — there is nothing
    to compare — so what it needs first is the photographs."""
    _measured(conn, "Bogota, text only", deck_text="A store opening. Two hero shots.")

    gap = next(g for g in core.gaps(conn)["gaps"]
               if g["code"] == "execution_never_checked")

    assert {a["tool"] for a in gap["next_actions"]} <= {"upload_image_asset",
                                                        "compare_execution"}


def test_a_library_that_does_not_photograph_can_say_so(conn):
    """The cost of widening the gap, and the thing that makes it bearable. A shop that records
    campaigns from descriptions has no creative at either phase and never will, so before this
    the gap would have been a permanent complaint — the exact state D53's set-aside exists for.
    It was not set-asideable while it required briefed creative, because every record it could
    reach had somebody who photographs."""
    _measured(conn, "Bogota, text only", deck_text="A store opening. Two hero shots.")
    assert any(g["code"] == "execution_never_checked" for g in core.gaps(conn)["gaps"])

    core.answer_gap(conn, code="execution_never_checked", answer="not_applicable",
                    note="We work from descriptions; nobody photographs what ran.",
                    said_by="R. Vega")

    assert not any(g["code"] == "execution_never_checked" for g in core.gaps(conn)["gaps"])


def test_the_offer_names_the_record_it_prefills(conn, tmp_path):
    """The two halves of an offer have to be about the same campaign. The reader approves what
    the sentence says; the tool receives what the arguments hold — and this one picked the
    title from the first unchecked record and the id from the first with nothing on file, which
    are different records the moment both states are present."""
    with_boards = _measured(conn, "Lima, with boards", detail="A store opening.")
    core.ingest_image_asset(conn, campaign_id=with_boards,
                            asset_ref={"path": str(_png(tmp_path))}, phase="proposed")
    _measured(conn, "Bogota, text only", deck_text="A store opening. Two hero shots.")

    gap = next(g for g in core.gaps(conn)["gaps"]
               if g["code"] == "execution_never_checked")

    for offer in gap["next_actions"]:
        named = store.get_campaign(conn, offer["prefilled_args"]["campaign_id"])
        assert named["title"] in offer["label"], (
            f"the offer says {offer['label']!r} and prefills {named['title']!r}"
        )


# ---------------------------------------------------------------------------------------
# D126 — a vector whose model nothing recorded


def test_a_commitment_vector_records_which_model_made_it(conn, clip_text, tmp_path):
    """Every other vector in this product goes through `core._add_vector`, which writes the
    vector AND `store.record_vector_model`. Commitments wrote theirs raw, so after a weights
    change they were compared against vectors from a different model — §7.2's "a model upgrade
    is a visible migration rather than a silent re-ranking", happening silently."""
    import commitments

    cid = _measured(conn, "Bogota launch", detail="Creator posts show the product against a navy backdrop.")
    commitments.add(conn, campaign_id=cid, text=_ANSWERABLE,
                    source_line=f"Creative: {_ANSWERABLE}")
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": str(_png(tmp_path))}, phase="delivered")

    commitments.check(conn, campaign_id=cid)

    staged = [r["vector_id"] for r in conn.execute(
        "SELECT vector_id FROM vector_provenance WHERE vector_id LIKE 'commitment:%'")]
    assert staged, "a commitment vector was cached with no record of which model made it"


def test_a_commitment_vector_is_stamped_with_clips_name_not_the_text_embedders(conn, monkeypatch):
    """`_add_vector`'s own rule: "the model recorded is the one for THIS space". A commitment
    vector is a PHRASE encoded by CLIP so it can be compared against photographs — stamping the
    text embedder's name on it would make the provenance false for every one of them, and would
    report "mixed models" the moment the text embedder changed, on the strength of rows that had
    nothing to do with it."""
    monkeypatch.setattr(config, "CLIP_PROVIDER", "open_clip")
    monkeypatch.setattr(config, "EMBED_PROVIDER", "ollama")

    assert core.embedding_model_id("commitment") == core.embedding_model_id("asset")
    assert core.embedding_model_id("commitment") != core.embedding_model_id("campaign")


def test_a_commitment_vector_from_another_model_is_not_reused(conn, clip_text, tmp_path):
    """What the record is FOR. A cached vector made by the previous weights must not be
    compared against one made by the new ones — the answer would be a similarity between two
    different spaces, reported as though it meant something."""
    import commitments

    cid = _measured(conn, "Bogota launch", detail="Creator posts show the product against a navy backdrop.")
    commitments.add(conn, campaign_id=cid, text=_ANSWERABLE,
                    source_line=f"Creative: {_ANSWERABLE}")
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": str(_png(tmp_path))}, phase="delivered")
    commitments.check(conn, campaign_id=cid)

    conn.execute("UPDATE vector_provenance SET model = 'clip/some-older-weights' "
                 "WHERE vector_id LIKE 'commitment:%'")
    conn.commit()

    assert commitments.stale_vectors(conn), (
        "a vector made by different weights is not reported as needing rebuilding"
    )

    # And REPORTING it is not the point — not USING it is. The first version of this test
    # asserted only the line above, and a mutation that removed the skip entirely left it
    # green: it proved the record existed while the thing the record is for went on happening.
    encoded = []
    real = clip_text.embed_text
    clip_text.embed_text = lambda text, **kw: (encoded.append(text), real(text, **kw))[1]
    try:
        commitments.check(conn, campaign_id=cid)
    finally:
        clip_text.embed_text = real

    assert _ANSWERABLE in encoded, (
        "the phrase was read out of a cache made by weights this build no longer runs, and "
        "scored against images embedded by the ones it does"
    )


def test_the_unreadable_vector_check_covers_every_space(conn):
    """`count_unreadable_vectors` is the one check written to notice a vector table this
    process cannot read — "every row flagged embedded and every search returning nothing".
    It iterated two spaces by name, so a third was outside the only thing looking."""
    import inspect

    source = inspect.getsource(vectorstore.count_unreadable_vectors)

    assert '("campaign", "asset")' not in source, (
        "the spaces are listed by hand, so a new one is outside the check by default"
    )
    assert "SPACES" in source, "it does not read the one list of spaces"
    assert "commitment" in vectorstore.SPACES


def test_a_stranded_commitment_vector_is_counted(conn, clip_text, tmp_path):
    """The failure the check exists for, in the space it did not cover: rows written where the
    extension loaded, opened where it does not. Every row reads as embedded and every search
    comes back empty, which no coverage count can see."""
    import commitments

    cid = _measured(conn, "Bogota launch", detail="Creator posts show the product against a navy backdrop.")
    commitments.add(conn, campaign_id=cid, text=_ANSWERABLE,
                    source_line=f"Creative: {_ANSWERABLE}")
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": str(_png(tmp_path))}, phase="delivered")
    commitments.check(conn, campaign_id=cid)

    before = vectorstore.count_unreadable_vectors(conn)
    vec_table, fallback = vectorstore._table_names("commitment")
    unread = fallback if vectorstore._try_load_vec(conn) else vec_table
    conn.execute(f"CREATE TABLE IF NOT EXISTS {unread} (vector_id TEXT PRIMARY KEY, v BLOB)")
    conn.execute(f"INSERT OR REPLACE INTO {unread} (vector_id, v) VALUES ('commitment:x', x'00')")
    conn.commit()

    assert vectorstore.count_unreadable_vectors(conn) == before + 1, (
        "a commitment vector in the table this process cannot read was not counted"
    )


# ---------------------------------------------------------------------------------------
# What the first pass at §13.6 got wrong, and an adversarial review found.


def test_a_record_that_arrives_after_the_set_aside_reopens_the_gap(conn):
    """F1. `since` is what keeps "this will never be true here" a statement about the records
    somebody was looking at. Without it `execution_never_checked` became the one set-asideable
    code with no way back — the product agreeing to stop mentioning something it had not yet
    seen, in the exact place the mechanism exists to prevent that."""
    _measured(conn, "Bogota, text only", deck_text="A store opening.")
    core.answer_gap(conn, code="execution_never_checked", answer="not_applicable",
                    note="We work from descriptions; nobody photographs what ran.",
                    said_by="R. Vega")
    assert not any(g["code"] == "execution_never_checked" for g in core.gaps(conn)["gaps"])

    import time

    time.sleep(1.1)                    # created_at has one-second resolution
    later = _measured(conn, "Cusco, also text only", deck_text="Another store opening.")

    gap = next((g for g in core.gaps(conn)["gaps"]
                if g["code"] == "execution_never_checked"), None)
    assert gap, "a record uploaded after the set-aside did not bring the gap back"
    assert later in gap["campaign_ids"]
    assert gap["reopened"], "it came back without saying why"


def test_the_record_the_offer_acts_on_is_one_the_gap_named(conn, tmp_path):
    """F5. `what` and `campaign_ids` are both capped at `_MAX_NAMED`, and the target was read
    from a different ordering — so with seven records the sentence named five boards-only
    campaigns and the offer said "add the photographs from" a sixth, whose id appeared in
    neither list."""
    # The bare record LAST, so the library's own ordering puts it seventh — past `_MAX_NAMED`,
    # and behind six records the offer would otherwise land on. Uploaded first it is already
    # at position 0 and the ordering carries no weight, which is a fixture that cannot tell
    # whether the code sorts at all.
    for n in range(6):
        cid = _measured(conn, f"Boards {n}", detail="A store opening.")
        core.ingest_image_asset(conn, campaign_id=cid,
                                asset_ref={"path": str(_png(tmp_path, f"b{n}.png"))},
                                phase="proposed")
    bare = _measured(conn, "Bare, uploaded last", deck_text="A store opening.")

    gap = next(g for g in core.gaps(conn)["gaps"]
               if g["code"] == "execution_never_checked")

    for offer in gap["next_actions"]:
        cid = offer["prefilled_args"]["campaign_id"]
        assert cid in gap["campaign_ids"], (
            "the offer acts on a record the gap did not name"
        )
        assert store.get_campaign(conn, cid)["title"] in gap["what"]
        assert cid == bare, (
            "the offer acts on a record with more evidence than one the gap also counts"
        )
    assert gap["counts"]["named"] == len(gap["campaign_ids"]) < gap["counts"]["campaigns"]


def test_one_bare_record_is_not_described_as_a_fraction_of_itself(conn):
    """F8. "1 finished campaign ... (1 of them has no creative on file at all)" is one record
    described twice, the second time as a fraction of itself."""
    _measured(conn, "Bogota, text only", deck_text="A store opening.")

    gap = next(g for g in core.gaps(conn)["gaps"]
               if g["code"] == "execution_never_checked")

    assert "of them" not in gap["what"], gap["what"]
    assert "1 with no creative" not in gap["what"], (
        f"one record counted against itself: {gap['what']}"
    )
    assert "\u2014 no creative on file at all" in gap["what"], gap["what"]


def test_photographs_from_other_weights_are_not_scored_against_a_fresh_phrase(conn, clip_text,
                                                                             tmp_path):
    """F2, and the defect that mattered most: the first version of this check made the
    weights-swap case WORSE. It compared the cached phrase against the BUILD, so it re-encoded
    the phrase with the new model and left the images on the old one — before it, both halves
    were stale and the comparison was at least internally consistent; after it, the comparison
    was guaranteed to be across two spaces. Re-encoding cannot fix the image half, so the
    honest answer is that nothing is known."""
    import commitments

    cid = _measured(conn, "Bogota launch", detail=_ANSWERABLE)
    commitments.add(conn, campaign_id=cid, text=_ANSWERABLE,
                    source_line=f"Creative: {_ANSWERABLE}")
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": str(_png(tmp_path))}, phase="delivered")
    assert commitments.check(conn, campaign_id=cid)["status"] == "checked"

    conn.execute("UPDATE vector_provenance SET model = 'clip/some-older-weights' "
                 "WHERE space = 'asset'")
    conn.commit()

    out = commitments.check(conn, campaign_id=cid)

    assert out["status"] == "nothing_to_check", (
        "a phrase from this build was scored against photographs indexed by another model"
    )
    assert all(i["verdict"] == "unchecked" for i in out["items"])
    assert any("clip/some-older-weights" in (w.get("detail") or "") for w in out["warnings"]), (
        out["warnings"]
    )
    assert "not the same as not finding them" in out["what_it_means"]


def test_a_stale_phrase_is_measured_against_the_photographs_not_the_build(conn, clip_text,
                                                                         tmp_path):
    """The other half of F2. "Stale" means "disagrees with the images it will be scored
    against". A cached phrase whose model matches the photographs is usable even if neither
    matches this build; one that disagrees with them is not, whatever the build says."""
    import commitments

    cid = _measured(conn, "Bogota launch", detail=_ANSWERABLE)
    commitments.add(conn, campaign_id=cid, text=_ANSWERABLE,
                    source_line=f"Creative: {_ANSWERABLE}")
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": str(_png(tmp_path))}, phase="delivered")
    commitments.check(conn, campaign_id=cid)

    # Both halves moved together: a library indexed by an older model, consistently.
    conn.execute("UPDATE vector_provenance SET model = 'clip/older-weights'")
    conn.commit()
    assert commitments.stale_vectors(conn) == [], (
        "a phrase that agrees with the photographs was called stale because the BUILD moved"
    )

    # And now only the phrase moves.
    conn.execute("UPDATE vector_provenance SET model = 'clip/newer-weights' "
                 "WHERE space = 'commitment'")
    conn.commit()
    assert commitments.stale_vectors(conn), (
        "a phrase that disagrees with the photographs was not reported"
    )


def test_a_deleted_commitment_takes_its_provenance_row_with_it(conn, clip_text, tmp_path):
    """F3. While commitment vectors carried no provenance there was nothing to clean up; now
    there is, and leaving it is the leak `forget_vector_models` exists to prevent — a model
    reported for a library holding no such vector, and `stale_vectors` naming one that is
    gone."""
    import commitments

    cid = _measured(conn, "Bogota launch", detail=_ANSWERABLE)
    commitments.add(conn, campaign_id=cid, text=_ANSWERABLE,
                    source_line=f"Creative: {_ANSWERABLE}")
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": str(_png(tmp_path))}, phase="delivered")
    commitments.check(conn, campaign_id=cid)
    assert store.vector_models(conn, space="commitment")

    store.delete_campaign(conn, cid)

    assert store.vector_models(conn, space="commitment") == {}, (
        "the row saying which model made the vector outlived the vector"
    )
    assert store.embedding_models(conn, "commitment") == set()


def test_the_health_check_reports_an_index_left_behind_by_a_weights_change(conn, tmp_path):
    """F6. `stale_vectors` said in its own docstring that it "must never stay silent", and
    nothing reported it anywhere. `_mixed_model_warning` asks a narrower question — two models
    inside the campaign space — so a whole image index left behind by a weights change was
    uniform, incomparable, and invisible on every surface."""
    cid = _measured(conn, "Bogota launch", detail="A store opening.")
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": str(_png(tmp_path))}, phase="delivered")
    assert core.health_check(conn, probe=False)["components"]["database"]["ok"]

    conn.execute("UPDATE vector_provenance SET model = 'clip/older-weights' "
                 "WHERE space = 'asset'")
    conn.commit()

    db = core.health_check(conn, probe=False)["components"]["database"]

    assert db["ok"] is False and db["code"] == "vectors_from_other_weights", db
    assert "clip/older-weights" in db["detail"]
    assert "asset" in db["detail"]
    assert db["remedy"]


def test_the_space_column_a_database_may_not_have_yet(conn, clip_text, tmp_path):
    """F7. `vector_provenance` predates its own `space` column, so a query naming it RAISES on
    an older install — and the caller's bare `except` read the exception as "nothing is
    stale", which is the check silently not running on exactly the databases old enough to
    have lived through a model change."""
    import commitments

    cid = _measured(conn, "Bogota launch", detail=_ANSWERABLE)
    commitments.add(conn, campaign_id=cid, text=_ANSWERABLE,
                    source_line=f"Creative: {_ANSWERABLE}")
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": str(_png(tmp_path))}, phase="delivered")
    commitments.check(conn, campaign_id=cid)

    rows = [(r["vector_id"], r["model"]) for r in
            conn.execute("SELECT vector_id, model FROM vector_provenance")]
    conn.execute("DROP TABLE vector_provenance")
    conn.execute("CREATE TABLE vector_provenance (vector_id TEXT PRIMARY KEY, model TEXT, "
                 "created_at INTEGER)")
    conn.executemany("INSERT INTO vector_provenance (vector_id, model, created_at) "
                     "VALUES (?,?,0)", rows)
    conn.commit()

    assert store.provenance_knows_spaces(conn) is False
    # No exception, and no false claim either way: the schema cannot attribute a row to a
    # space, so nothing is reported rather than everything.
    assert commitments.stale_vectors(conn) == []
    assert core._vectors_from_other_weights(conn) == {}
    assert commitments.check(conn, campaign_id=cid)["status"] == "checked"


def test_one_list_of_which_spaces_are_clips(conn, monkeypatch):
    """F4. "Is this space CLIP's" had two hand-written answers — `_default_dim` sized two
    spaces by name and `embedding_model_id` named them again — so a space added to one and not
    the other would be SIZED by CLIP and STAMPED with the text embedder's name, making the
    provenance false for every vector in it. That is the two-implementations-of-one-rule shape
    D126 was itself filed against, reintroduced by the change that closed it."""
    import config

    # Apart, or both providers answer "hash" and the two stamps cannot be told apart.
    monkeypatch.setattr(config, "CLIP_PROVIDER", "open_clip")
    monkeypatch.setattr(config, "EMBED_PROVIDER", "ollama")

    for space in vectorstore.SPACES:
        clips = space in vectorstore.CLIP_SPACES
        assert vectorstore._default_dim(space) == (config.CLIP_EMBED_DIM if clips
                                                   else config.EMBED_DIM), space
        assert (core.embedding_model_id(space) == core.embedding_model_id("asset")) == clips, (
            f"{space} is sized and stamped by different models"
        )


# ---------------------------------------------------------------------------------------
# The offer that did not work, and the two surfaces it made disagree.


def test_each_offer_makes_progress_and_the_gap_closes_only_when_it_is_answerable(conn,
                                                                                  tmp_path):
    """The blocking finding. The offer asked a record with nothing briefed for the DELIVERED
    photographs, and `compare_execution` then refused from the other side — "1 delivered image
    on file and nothing briefed to compare them against" — while the upload CLOSED the gap.
    §5.2's rule is that an offer has to be one the tool accepts AND the record can use.

    A bare record needs both halves, so no single offer can make the comparison possible; what
    each one has to do is move the record to a state that is nearer and still reported. The
    gap closes when the comparison has actually been made, and not before."""
    bare = _measured(conn, "Bogota, text only", deck_text="A store opening.")
    seen = []

    for step in range(2):
        gap = next((g for g in core.gaps(conn)["gaps"]
                    if g["code"] == "execution_never_checked"), None)
        assert gap and bare in gap["campaign_ids"], (
            f"step {step}: the gap went quiet while the comparison still cannot run"
        )
        offer = gap["next_actions"][0]
        assert offer["prefilled_args"]["campaign_id"] == bare
        phase = offer["prefilled_args"]["phase"]
        assert phase not in seen, f"it asked twice for the {phase} half"
        seen.append(phase)
        core.ingest_image_asset(conn, campaign_id=bare,
                                asset_ref={"path": str(_png(tmp_path, f"{phase}.png"))},
                                phase=phase)

    assert seen == ["proposed", "delivered"], (
        f"the briefed half has to come first — asked in the order {seen}"
    )
    out = core.compare_execution(conn, campaign_id=bare)
    assert out["status"] != "nothing_to_check", (
        f"both offers were accepted and the comparison still cannot run: {out['what_it_means']}"
    )
    assert not any(g["code"] == "execution_never_checked" for g in core.gaps(conn)["gaps"]), (
        "the record has been checked and the gap still reports it"
    )


def test_accepting_the_offer_does_not_close_the_gap_while_the_citations_still_object(conn,
                                                                                     tmp_path):
    """D55's shape, manufactured by the product's own suggestion. The gap read "no delivered
    creative" while every citation read the stored drift status, so uploading one photograph to
    a brief-less record removed it from the report and left `never_checked` stamped on every
    citation of it forever."""
    bare = _measured(conn, "Bogota, text only", deck_text="A store opening.")
    core.ingest_image_asset(conn, campaign_id=bare,
                            asset_ref={"path": str(_png(tmp_path, "ran.png"))},
                            phase="delivered")

    assert store.execution_drift_for(conn, bare)["status"] == "never_checked"
    gap = next((g for g in core.gaps(conn)["gaps"]
                if g["code"] == "execution_never_checked"), None)
    assert gap and bare in gap["campaign_ids"], (
        "the gap went quiet about a record every citation of which still says nobody checked it"
    )
    assert gap["counts"]["nothing_briefed"] == 1
    assert gap["next_actions"][0]["prefilled_args"]["phase"] == "proposed", (
        "it asked again for the half that is already there"
    )


def test_a_record_with_both_halves_and_no_comparison_is_offered_the_comparison(conn, tmp_path):
    """The fourth state, which the old condition could not see at all: both halves on file and
    nothing ever compared. The citations say nobody checked it, because nobody has."""
    cid = _measured(conn, "Cusco", detail="A store opening.")
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": str(_png(tmp_path, "brief.png"))},
                            phase="proposed")
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": str(_png(tmp_path, "ran.png"))},
                            phase="delivered")
    conn.execute("DELETE FROM execution_drift WHERE campaign_id = ?", (cid,))
    conn.commit()

    gap = next(g for g in core.gaps(conn)["gaps"]
               if g["code"] == "execution_never_checked")

    assert gap["counts"]["never_compared"] == 1
    assert gap["next_actions"][0]["tool"] == "compare_execution"
    assert gap["next_actions"][0]["prefilled_args"] == {"campaign_id": cid}


def test_the_gap_and_the_caveat_are_one_claim(conn, tmp_path):
    """What the condition is now FOR. The gap reads the same stored row `_execution_note`
    stamps on every citation, so the report and the caveat cannot disagree about a record —
    reported exactly while the caveat is attached, closed exactly when it stops."""
    checked = _measured(conn, "Cusco, checked", detail="A store opening.")
    core.ingest_image_asset(conn, campaign_id=checked,
                            asset_ref={"path": str(_png(tmp_path, "brief.png"))},
                            phase="proposed")
    core.ingest_image_asset(conn, campaign_id=checked,
                            asset_ref={"path": str(_png(tmp_path, "ran.png"))},
                            phase="delivered")
    bare = _measured(conn, "Bogota, text only", deck_text="A store opening.")

    gap = next(g for g in core.gaps(conn)["gaps"]
               if g["code"] == "execution_never_checked")
    reported = set(gap["campaign_ids"])

    for cid in (checked, bare):
        caveats = store.execution_drift_for(conn, cid)["status"] == "never_checked"
        assert caveats == (cid in reported), (
            f"{cid}: citation says never_checked={caveats}, gap reports it={cid in reported}"
        )


def test_a_clip_weights_change_reaches_the_visible_migration_notice(conn, tmp_path):
    """§7.2 asks for "a model upgrade is a visible migration rather than a silent re-ranking",
    and the notice that delivers it read the campaign space only — so a CLIP change, which
    re-ranks every image similarity in the product, could not reach it. Two models ACROSS
    spaces stays silent: CLIP makes the image vectors and the text embedder the chunk ones,
    and reading that as a mixed index would warn on every correctly-indexed library."""
    cid = _measured(conn, "Bogota launch", detail="A store opening.")
    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": str(_png(tmp_path, "a.png"))}, phase="delivered")
    assert core._mixed_model_warning(conn) == [], (
        "a library with CLIP images and text chunks is not a mixed index"
    )

    core.ingest_image_asset(conn, campaign_id=cid,
                            asset_ref={"path": str(_png(tmp_path, "b.png"))}, phase="proposed")
    conn.execute("UPDATE vector_provenance SET model = 'clip/older-weights' "
                 "WHERE space = 'asset' AND vector_id = ("
                 "  SELECT vector_id FROM vector_provenance WHERE space = 'asset' LIMIT 1)")
    conn.commit()

    warned = core._mixed_model_warning(conn)

    assert warned, "half the image index came from other weights and nothing said so"
    assert "asset" in warned[0]["detail"] and "clip/older-weights" in warned[0]["detail"]
