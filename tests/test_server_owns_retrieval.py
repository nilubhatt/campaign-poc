"""
§7.2: stop letting the model author the retrieval query.

The review's words: *"`prepare_evaluation` takes `proposal_text` — which the model writes. A
200-word summary and a 2,000-word one retrieve different evidence from the same deck, and
different evidence is a different verdict. The same applies to the filters: the model
currently chooses region, status and tags, and each choice changes the evidence package. Fix:
the server extracts the query from the source file, not from the conversation. Same deck in,
same chunks out. Derive filters from the subject record's own attributes rather than accepting
them from the caller, or require the caller to pass the subject's `campaign_id` and derive
everything from it. Pin `top_k`, make tie-breaking on equal similarity stable and
deterministic, and record the embedding model version on every stored vector so a model
upgrade is a visible migration rather than a silent re-ranking."*

**"Same deck in, same chunks out"** is the acceptance test, and it is the whole phase in one
sentence. Everything below exists to make it true.

Five parts, and the fifth is the one that pays for the rest. Once the SERVER owns retrieval it
can write down what it retrieved — and a recorded evidence window is what D77, D82, D83 and
D86 have all been waiting for. §6.1 rejected a receipt as the *mechanism* for verifying a
quote (X7), because it answers a weaker question than "does the record contain this" and it
put a stateful handshake on a stateless tool. That objection was about 6.1. Here the server is
already doing the retrieval, so the receipt is a by-product rather than an imposition, and it
answers the question 6.1 could not: was this evidence in front of the reasoner at all.
"""
import pytest

import core
import store


@pytest.fixture
def library(conn):
    ids = {}
    for name, market in (("Peru launch", "LATAM"), ("Chile launch", "LATAM"),
                         ("Jakarta launch", "SEA")):
        ids[name] = core.ingest_campaign(
            conn, title=name, market=market, status="concluded",
            detail=f"A creator-led launch in {market} with four colourways and a "
                   f"compressed three-week flight.")["campaign_id"]
    return ids


# ── same deck in, same chunks out ───────────────────────────────────────────

def test_the_same_record_retrieves_the_same_evidence_however_it_is_described(conn, library):
    """The review's acceptance test. "A 200-word summary and a 2,000-word one retrieve
    different evidence from the same deck, and different evidence is a different verdict."""
    subject = core.ingest_campaign(
        conn, title="Colombia v1", market="LATAM",
        detail="A creator-led launch in Colombia with four colourways.")["campaign_id"]

    terse = core.prepare_evaluation(conn, subject_title="Colombia v1",
                                    proposal_text="A launch.", campaign_id=subject)
    verbose = core.prepare_evaluation(
        conn, subject_title="Colombia v1", campaign_id=subject,
        proposal_text="An extremely long description of an entirely different subject, "
                      "concerning out-of-home media in a market nobody mentioned, at "
                      "sufficient length to change any similarity ranking that read it. " * 8)

    assert [e["campaign_id"] for e in terse["evidence"]] == \
           [e["campaign_id"] for e in verbose["evidence"]]
    assert terse["retrieval"]["query"] == "subject_record"


def test_without_a_record_the_query_is_the_prose_and_the_package_says_so(conn, library):
    """The honest half. A brief that is not yet a record can only be described, and the
    variance is real — so it is named rather than hidden."""
    package = core.prepare_evaluation(conn, subject_title="Colombia v1",
                                      proposal_text="A creator-led launch.")

    assert package["retrieval"]["query"] == "caller_text"
    assert "campaign_id" in package["retrieval"]["what_it_means"]


# ── filters come from the record, not from the conversation ─────────────────

def test_filters_are_derived_from_the_subject_record(conn, library):
    """"The model currently chooses region, status and tags, and each choice changes the
    evidence package."""
    subject = core.ingest_campaign(
        conn, title="Colombia v1", market="LATAM",
        detail="A creator-led launch in LATAM.")["campaign_id"]

    package = core.prepare_evaluation(conn, subject_title="Colombia v1",
                                      proposal_text="x", campaign_id=subject)

    assert package["retrieval"]["filters"] == {"market": "LATAM"}
    assert package["retrieval"]["filters_from"] == "subject_record"
    assert library["Jakarta launch"] not in [e["campaign_id"] for e in package["evidence"]]


def test_a_caller_cannot_override_the_subjects_own_filters(conn, library):
    """Accepting them would put the variance back in the one call the item exists to make
    deterministic — and silently, because a caller cannot see which filters were used."""
    subject = core.ingest_campaign(conn, title="Colombia v1", market="LATAM",
                                   detail="A launch.")["campaign_id"]

    with pytest.raises(ValueError) as e:
        core.prepare_evaluation(conn, subject_title="Colombia v1", proposal_text="x",
                                campaign_id=subject, market="SEA")
    assert "subject" in str(e.value).lower()


def test_without_a_subject_record_caller_filters_are_allowed_and_recorded(conn, library):
    """A brief with no record has no attributes to derive from, so refusing filters would
    remove a capability and gain nothing. What must not happen is the choice being invisible."""
    package = core.prepare_evaluation(conn, subject_title="Colombia v1",
                                      proposal_text="A launch.", market="SEA")

    assert package["retrieval"]["filters"] == {"market": "SEA"}
    assert package["retrieval"]["filters_from"] == "caller"


# ── pinned, and deterministic ───────────────────────────────────────────────

def test_top_k_is_the_servers_to_choose(conn, library):
    """"Pin top_k." A caller who asks for twenty gets a different evidence package from one
    who asks for three, and neither of them chose the brief."""
    with pytest.raises(ValueError) as e:
        core.prepare_evaluation(conn, subject_title="X", proposal_text="A launch.", top_k=20)
    assert "top_k" in str(e.value)

    package = core.prepare_evaluation(conn, subject_title="X", proposal_text="A launch.")
    assert package["retrieval"]["top_k"] == core._PINNED_TOP_K


def test_equal_similarity_breaks_the_same_way_every_time(conn):
    """"Make tie-breaking on equal similarity stable and deterministic." Three identical
    decks score identically, and whichever order the rows happen to come back in becomes the
    evidence order — and the first one is what `closest_precedent` points at."""
    body = "A creator-led launch with four colourways and a compressed flight."
    for name in ("Gamma", "Alpha", "Beta"):
        core.ingest_campaign(conn, title=name, detail=body)

    orders = [[e["campaign_id"] for e in core.prepare_evaluation(
        conn, subject_title="X", proposal_text=body)["evidence"]] for _ in range(3)]

    assert orders[0] == orders[1] == orders[2]
    assert orders[0] == sorted(orders[0]), "ties break on the id, which is stable across runs"


# ── the embedding model is on the record ────────────────────────────────────

def test_every_stored_vector_records_the_model_that_made_it(conn):
    """"So a model upgrade is a visible migration rather than a silent re-ranking." Without
    it, changing the embedder re-ranks every judgment in the library and nothing says so."""
    cid = core.ingest_campaign(conn, title="Peru", detail="A launch.")["campaign_id"]

    models = store.embedding_models(conn)
    assert models == {core.embedding_model_id()}
    assert cid


def test_a_library_embedded_by_two_models_is_reported_as_such(conn):
    """The migration made visible. Two models in one index means the similarities are not
    comparable, and a ranking across them is arithmetic on incompatible numbers."""
    core.ingest_campaign(conn, title="Peru", detail="A launch.")
    store.set_embedding_model(conn, "some-other-model")
    core.ingest_campaign(conn, title="Chile", detail="A launch.")

    package = core.prepare_evaluation(conn, subject_title="X", proposal_text="A launch.")
    warning = [w for w in package["warnings"] if w["code"] == "mixed_embedding_models"]
    assert warning, [w["code"] for w in package["warnings"]]
    assert "re-index" in warning[0]["remedy"].lower() or "reembed" in warning[0]["remedy"].lower()


# ── the receipt, and the four rows waiting on it ────────────────────────────

def test_the_server_records_what_it_retrieved(conn, library):
    """Once the server owns retrieval it can write down what it returned — and §6.1 rejected
    a receipt as the MECHANISM for verifying a quote (X7) on the grounds that it answers a
    weaker question and puts a stateful handshake on a stateless tool. That objection was
    about 6.1. Here the server is already doing the retrieval, so the receipt is a by-product,
    and it answers the question 6.1 could not: was this in front of the reasoner at all."""
    package = core.prepare_evaluation(conn, subject_title="Colombia v1",
                                      proposal_text="A creator-led launch.")

    receipt = store.get_retrieval(conn, package["retrieval"]["receipt"])
    assert receipt["campaign_ids"] == [e["campaign_id"] for e in package["evidence"]]
    assert receipt["query"] == "caller_text"


def test_a_judgment_records_the_window_it_was_given(conn, library):
    """D77. §6.1 checks that the cited record contains the quote; this checks that the record
    was in the evidence at all — the half that catches cherry-picking and cross-session
    memory, and the half Phase 7 needs, because two users judging the same brief should rest
    on the same evidence."""
    package = core.prepare_evaluation(conn, subject_title="Colombia v1",
                                      proposal_text="A creator-led launch.")
    cited = package["evidence"][0]["campaign_id"]

    saved = core.save_evaluation(
        conn, subject_title="Colombia v1", verdict="revise", summary="A stretch.",
        approve_if="It is fixed.", cited_ids=[cited],
        retrieval=package["retrieval"]["receipt"],
        findings=[{"severity": "should_fix", "kind": "missing_information",
                   "finding": "No end date", "fix": "Add one"}])

    assert saved["evidence"]["from_the_window"] == [cited]
    assert saved["evidence"]["outside_the_window"] == []


def test_a_citation_from_outside_the_window_is_named(conn, library):
    """Cherry-picking, and the case §6.1's record check cannot see: the record really does
    contain the quote, and the reasoner was never shown it."""
    package = core.prepare_evaluation(conn, subject_title="Colombia v1",
                                      proposal_text="A creator-led launch.")
    shown = {e["campaign_id"] for e in package["evidence"]}
    unshown = [c for c in library.values() if c not in shown]
    if not unshown:                       # every record was retrieved; make one that was not
        unshown = [core.ingest_campaign(conn, title="Unrelated",
                                        detail="zzz qqq vvv.")["campaign_id"]]

    saved = core.save_evaluation(
        conn, subject_title="Colombia v1", verdict="revise", summary="A stretch.",
        approve_if="It is fixed.", cited_ids=[unshown[0]],
        retrieval=package["retrieval"]["receipt"],
        findings=[{"severity": "should_fix", "kind": "missing_information",
                   "finding": "No end date", "fix": "Add one"}])

    assert saved["evidence"]["outside_the_window"] == [unshown[0]]
    assert "outside" in saved["note"].lower()


def test_a_judgment_with_no_receipt_says_the_window_is_unknown(conn, library):
    """Not "everything was in the window", which is what an absent check reads as. The same
    distinction §6.4 needed separate codes for."""
    saved = core.save_evaluation(
        conn, subject_title="Colombia v1", verdict="revise", summary="A stretch.",
        approve_if="It is fixed.", cited_ids=[library["Peru launch"]],
        findings=[{"severity": "should_fix", "kind": "missing_information",
                   "finding": "No end date", "fix": "Add one"}])

    assert saved["evidence"]["window"] == "not_recorded"
    assert "from_the_window" not in saved["evidence"]


def test_a_receipt_that_does_not_exist_is_refused(conn, library):
    with pytest.raises(ValueError) as e:
        core.save_evaluation(
            conn, subject_title="X", verdict="revise", summary="A stretch.",
            approve_if="It is fixed.", retrieval="ret_nope",
            findings=[{"severity": "should_fix", "kind": "missing_information",
                       "finding": "No end date", "fix": "Add one"}])
    assert "ret_nope" in str(e.value)


def test_it_reaches_the_model_over_the_protocol(conn, library):
    import asyncio
    import json

    async def call(name, args):
        import mcp_server
        return json.loads((await mcp_server.mcp.call_tool(name, args)).content[0].text)

    package = asyncio.run(call("prepare_evaluation",
                               {"subject_title": "Colombia v1",
                                "proposal_text": "A creator-led launch."}))
    assert package["retrieval"]["top_k"] == core._PINNED_TOP_K

    refused = asyncio.run(call("prepare_evaluation",
                               {"subject_title": "X", "proposal_text": "A launch.",
                                "top_k": 20}))
    assert "top_k" in refused["error"]


# ── the three rows this item was blocking ───────────────────────────────────

def test_the_closest_precedent_is_the_servers_to_name(conn, library):
    """D8. It was whatever the model asserted — a claim about which record is nearest, made
    by the party that did not do the ranking. With the server owning retrieval it is the top
    of the window, which is a fact, and §6.6's `dominated_by` already leans on it."""
    package = core.prepare_evaluation(conn, subject_title="Colombia v1",
                                      proposal_text="A creator-led launch.")

    saved = core.save_evaluation(
        conn, subject_title="Colombia v1", verdict="revise", summary="A stretch.",
        approve_if="It is fixed.", retrieval=package["retrieval"]["receipt"],
        findings=[{"severity": "should_fix", "kind": "missing_information",
                   "finding": "No end date", "fix": "Add one"}])

    assert saved["closest_precedent"]["campaign_id"] == \
        package["evidence"][0]["campaign_id"]
    assert saved["closest_precedent"]["basis"] == "computed"


def test_a_model_cannot_assert_the_closest_precedent_when_the_server_knows(conn, library):
    package = core.prepare_evaluation(conn, subject_title="Colombia v1",
                                      proposal_text="A creator-led launch.")
    with pytest.raises(ValueError) as e:
        core.save_evaluation(
            conn, subject_title="Colombia v1", verdict="revise", summary="A stretch.",
            approve_if="It is fixed.", retrieval=package["retrieval"]["receipt"],
            closest_precedent={"campaign_id": library["Peru launch"]},
            findings=[{"severity": "should_fix", "kind": "missing_information",
                       "finding": "No end date", "fix": "Add one"}])
    assert "server" in str(e.value).lower()


def test_without_a_receipt_the_model_may_still_name_one(conn, library):
    """The server cannot compute what it did not retrieve, and refusing the field outright
    would remove it from every judgment made without `prepare_evaluation`."""
    saved = core.save_evaluation(
        conn, subject_title="Colombia v1", verdict="revise", summary="A stretch.",
        approve_if="It is fixed.",
        closest_precedent={"campaign_id": library["Peru launch"], "similarity": 0.9},
        findings=[{"severity": "should_fix", "kind": "missing_information",
                   "finding": "No end date", "fix": "Add one"}])
    assert saved["closest_precedent"]["campaign_id"] == library["Peru launch"]
    assert saved["closest_precedent"].get("basis") != "computed"


def test_editing_a_record_re_indexes_it(conn):
    """D80. `update_campaign` rewrote `title` and `detail` and left the chunks and vectors
    alone, so the search still matched the old wording — "same deck in, same chunks out" is
    false the moment a deck changes and the index does not. §6.1 stopped the stale text being
    QUOTABLE by reading the row columns; the index itself was still stale."""
    cid = core.ingest_campaign(
        conn, title="Peru launch",
        detail="A soap opera product placement in Lima.")["campaign_id"]

    core.update_campaign(conn, cid, detail="A creator-led influencer seeding in Lima.")

    hits = [h["campaign_id"] for h in
            core.find_similar(conn, text="creator-led influencer seeding", top_k=5)]
    assert cid in hits
    stored = "\n".join(store.text_on_file(conn, cid)["body"])
    assert "soap opera" not in stored


def test_an_edit_that_touches_no_content_does_not_re_embed(conn):
    """Re-indexing costs an embedding call per chunk. A status or tag change is not a content
    change, and paying for one would make every bulk edit a re-index of the library."""
    cid = core.ingest_campaign(conn, title="Peru", detail="A launch.")["campaign_id"]
    before = store.get_campaign(conn, cid)["chunks_embedded"]

    calls = []
    original = core.embedding.embed
    core.embedding.embed = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    try:
        core.update_campaign(conn, cid, status="concluded")
    finally:
        core.embedding.embed = original

    assert calls == []
    assert store.get_campaign(conn, cid)["chunks_embedded"] == before


def test_the_disconfirming_search_uses_the_subject_the_server_retrieved_for(conn, library):
    """D86. It queried the judgment's own prose whenever the subject was not a stored record
    — a claim about the brief made from a search over the complaint about it. The receipt
    records which subject the evidence was gathered for, so the check can use it."""
    package = core.prepare_evaluation(
        conn, subject_title="Colombia v1",
        proposal_text="A creator-led launch in LATAM with four colourways.")

    saved = core.save_evaluation(
        conn, subject_title="Colombia v1", verdict="revise",
        summary="Budget looks thin.",          # shares no words with the brief
        approve_if="It is fixed.", retrieval=package["retrieval"]["receipt"],
        findings=[{"severity": "should_fix", "kind": "missing_information",
                   "finding": "No end date", "fix": "Add one"}])

    assert saved["disconfirming"]["query"] == "retrieval_receipt"
