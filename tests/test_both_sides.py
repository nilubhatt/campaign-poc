"""
§11.5 — append-only reactions: keep both sides of a disagreement.

    "Append-only reactions — keep both sides of a disagreement, surface it in retrieval,
     authority order configured in the rulebook, never inferred."

## What is wrong today

`update_campaign` REPLACES the tag list. So R. Vega records `liked` in March, A. Duarte
records `not_liked` in June, and March is gone — not superseded, not outvoted, gone, with
nothing on the record saying it was ever there.

That is the most expensive thing this library can lose. A campaign two people disagreed about
is worth more as evidence than one everybody liked, because the disagreement is where the
client's actual taste lives — and the product's whole premise is reasoning from what this
client thinks. A later judgment retrieving that record is told "they liked it" with the same
confidence whether it was unanimous or two-to-one.

It is also how the queue lies. §10.1 calls a campaign open when nobody has said what they
made of the work; one person answering closes it for everybody, so the second opinion is
never asked for, and the first answer becomes the record by being first.

## What this does NOT do

**It does not decide who wins.** D10 is explicit and 11.5 repeats it: *authority order is
configured in the rulebook, never inferred.* The rulebook is Phase 12, so today the honest
answer is that both views are on file, the library will not rank them, and it says so. A
product that quietly preferred the newer, or the client's, or the one from somebody with a
grander job title would be inventing an authority nobody granted it — and §2.5 already
refused exactly this once, because a PDF export turns speaker notes into annotations, so even
the FORMAT cannot be trusted to say whose words carry more weight.
"""
import pytest

import core
import store


def _campaign(conn, title="Colombia", **kw):
    kw.setdefault("market", "LATAM")
    kw.setdefault("status", "concluded")
    kw.setdefault("detail", f"A campaign called {title}, which ran in a market.")
    return core.ingest_campaign(conn, title=title, **kw)["campaign_id"]


def _said(conn, cid, value, who, **kw):
    return core.record_reaction(conn, campaign_id=cid, value=value, said_by=who, **kw)


# ── the record keeps both ──────────────────────────────────────────────────

def test_a_second_opinion_does_not_erase_the_first(conn):
    """The defect. `update_campaign` replaces the tag list, so June destroys March — not
    superseded, not outvoted, gone, with nothing saying it was ever there."""
    cid = _campaign(conn)
    _said(conn, cid, "liked", "R. Vega")

    _said(conn, cid, "not_liked", "A. Duarte")

    voices = store.reactions_for(conn, cid)
    assert [(v["value"], v["said_by"]) for v in voices] == [
        ("liked", "R. Vega"), ("not_liked", "A. Duarte")]


def test_the_same_person_changing_their_mind_keeps_both(conn):
    """"This looked fine before the numbers came in" is the most interesting row this table
    can hold — §9.8 made exactly this argument about attributions. An update in place destroys
    the pair, and the pair is the point."""
    cid = _campaign(conn)
    _said(conn, cid, "liked", "R. Vega")

    _said(conn, cid, "not_liked", "R. Vega")

    voices = store.reactions_for(conn, cid)
    assert len(voices) == 2
    assert voices[-1]["supersedes_own_earlier_view"] is True


def test_a_reaction_records_who_and_when(conn):
    """§11.1's account beside §11.5's opinion: a view with nobody's name against it cannot be
    weighed against another one later, which is the entire mechanism here."""
    cid = _campaign(conn)

    _said(conn, cid, "liked", "R. Vega", role="Regional planner, LATAM")

    voice = store.reactions_for(conn, cid)[0]
    assert voice["said_by"] == "R. Vega"
    assert voice["said_at"]
    assert voice["role"] == "Regional planner, LATAM"
    assert voice["source"] == "stated"


def test_the_product_cannot_hold_an_opinion(conn):
    cid = _campaign(conn)

    with pytest.raises(ValueError, match="must be a PERSON"):
        _said(conn, cid, "liked", "Claude")


# ── the disagreement is reported, never resolved ───────────────────────────

def test_a_disagreement_is_named_on_the_record(conn):
    """A campaign two people disagreed about is worth MORE as evidence than one everybody
    liked — the disagreement is where the client's actual taste lives. Reporting "liked" flat
    is the library throwing away the most informative thing it holds."""
    cid = _campaign(conn)
    _said(conn, cid, "liked", "R. Vega")
    _said(conn, cid, "not_liked", "A. Duarte")

    record = store.get_campaign(conn, cid)

    assert record["disagreement"]["reaction"]["values"] == ["liked", "not_liked"]
    assert record["disagreement"]["basis"] == "computed"


def test_the_library_refuses_to_say_who_is_right(conn):
    """D10, verbatim: "authority order is configured in the rulebook, never inferred." The
    rulebook is Phase 12, so the honest answer today is that both are on file and this library
    will not rank them — and it has to SAY so, because a reader who is not told will assume
    the one shown first won."""
    cid = _campaign(conn)
    _said(conn, cid, "liked", "R. Vega")
    _said(conn, cid, "not_liked", "A. Duarte")

    said = store.get_campaign(conn, cid)["disagreement"]["what_it_means"]

    assert "not configured" in said
    assert "rulebook" in said


def test_neither_recency_nor_role_decides_it(conn):
    """The two things a product would reach for. A later view is not a better one — it is a
    view somebody happened to record second — and a grander job title is authority this
    library was never given."""
    cid = _campaign(conn)
    _said(conn, cid, "liked", "R. Vega", role="Head of strategy")
    _said(conn, cid, "not_liked", "A. Duarte", role="Intern")

    holding = store.get_campaign(conn, cid)["disagreement"]["reaction"]

    assert "winner" not in holding and "standing" not in holding
    assert sorted(v["said_by"] for v in holding["voices"]) == ["A. Duarte", "R. Vega"]


def test_agreement_is_not_reported_as_a_disagreement(conn):
    """Two people saying the same thing is corroboration, and reporting it under a heading
    that means "they disagreed" would be worse than saying nothing."""
    cid = _campaign(conn)
    _said(conn, cid, "liked", "R. Vega")
    _said(conn, cid, "liked", "A. Duarte")

    assert "disagreement" not in store.get_campaign(conn, cid)


def test_one_person_twice_on_the_same_axis_is_not_a_disagreement(conn):
    """Somebody changing their own mind is a sequence, not a split. Two voices are needed for
    a disagreement, and counting one person's revision as one would report a split that never
    happened to every judgment citing the record."""
    cid = _campaign(conn)
    _said(conn, cid, "liked", "R. Vega")
    _said(conn, cid, "not_liked", "R. Vega")

    assert "disagreement" not in store.get_campaign(conn, cid)


def test_different_axes_are_not_a_disagreement(conn):
    """"They liked it and it underperformed" is the most ordinary finding in marketing, and
    reading it as a contradiction would flag half the library."""
    cid = _campaign(conn)
    _said(conn, cid, "liked", "R. Vega")
    _said(conn, cid, "underperformed", "A. Duarte")

    assert "disagreement" not in store.get_campaign(conn, cid)


# ── surfaced in retrieval, which is where it changes a judgment ────────────

def test_the_evidence_row_says_they_disagreed(conn):
    """"surface it in retrieval" — the item's own words, and the half that matters. A
    judgment weighing a precedent needs to know the precedent was contested; told only
    "liked", it cites a two-to-one split as though it were unanimous."""
    cid = _campaign(conn, "Peru launch")
    _said(conn, cid, "liked", "R. Vega")
    _said(conn, cid, "not_liked", "A. Duarte")
    _campaign(conn, "Bogota launch")

    prepared = core.prepare_evaluation(conn, subject_title="A new brief",
                                       proposal_text="A retail activation in a market.")

    row = next(e for e in prepared["evidence"] if e["campaign_id"] == cid)
    assert row["disagreement"]["reaction"]["values"] == ["liked", "not_liked"]


def test_the_model_is_told_what_a_disagreement_is_worth(conn):
    """Structure alone does not survive a model's summary. §7.1's lesson: a fact the model has
    to infer the significance of is one it will flatten into "they liked it"."""
    cid = _campaign(conn, "Peru launch")
    _said(conn, cid, "liked", "R. Vega")
    _said(conn, cid, "not_liked", "A. Duarte")

    prepared = core.prepare_evaluation(conn, subject_title="A new brief",
                                       proposal_text="A retail activation in a market.")

    note = prepared["note"]
    assert "DISAGREED" in note
    # The two things the model has to be told, or the structure is decoration: cite the
    # split rather than a side, and this library will not rank the sides for you.
    assert "Cite the split" in note
    assert "authority order is configured in the rulebook" in note


def test_a_library_with_no_disagreements_says_nothing_about_them(conn):
    """The silence half. A paragraph about disagreement on every judgment is one nobody reads
    by the third time, including on the judgment where there is one."""
    _campaign(conn, "Peru launch")

    prepared = core.prepare_evaluation(conn, subject_title="A new brief",
                                       proposal_text="A retail activation in a market.")

    assert "DISAGREED" not in prepared["note"]


def test_the_split_is_between_people_because_the_collapse_says_so(conn):
    """What actually enforces "two people, not two values": the per-person collapse, which
    keeps each person's latest view before anything is compared. Mutation showed the explicit
    `len(standing) > 1` beside it could be deleted with no test noticing — because it was
    redundant, not because it was untested. A condition that cannot change the outcome is a
    dead guard, and this codebase has just spent a round removing two of those."""
    cid = _campaign(conn)
    for value in ("liked", "not_liked", "liked", "mixed_reaction"):
        _said(conn, cid, value, "R. Vega")

    assert "disagreement" not in store.get_campaign(conn, cid), (
        "four views from one person is a sequence, however much they changed their mind"
    )
    assert len(store.reactions_for(conn, cid)) == 4, "and all four are still on file"

    _said(conn, cid, "liked", "A. Duarte")

    split = store.get_campaign(conn, cid)["disagreement"]["reaction"]
    assert split["values"] == ["liked", "mixed_reaction"], (
        "and it is each person's LATEST that is compared, not everything they ever said"
    )


def test_an_opinion_arriving_through_update_campaign_reaches_the_record(conn):
    """`update_campaign` is the other door into a reaction — its own docstring calls it the
    way to upgrade a tag's provenance. A view arriving that way and not reaching the
    append-only table would be one the library silently lost, and mutation showed no test
    covered it: the stdio probe found it, one layer above where the unit tests look."""
    cid = _campaign(conn)
    _said(conn, cid, "liked", "R. Vega")

    core.update_campaign(conn, campaign_id=cid, tags=[
        {"value": "not_liked", "source": "stated", "said_by": "A. Duarte"}])

    assert {v["said_by"] for v in store.reactions_for(conn, cid)} == {"R. Vega", "A. Duarte"}
    assert store.get_campaign(conn, cid)["disagreement"]


def test_a_tag_with_nobody_against_it_is_not_filed_as_an_opinion(conn):
    """The silence half. A `liked` with no `said_by` is bookkeeping or an import, and filing
    it as somebody's view would put an opinion on the record with nobody's name against it —
    the one thing this table exists to prevent."""
    cid = _campaign(conn)

    core.update_campaign(conn, campaign_id=cid,
                         tags=[{"value": "liked", "source": "stated"}])

    assert store.reactions_for(conn, cid) == []


# ── the silence, not the sentence ─────────────────────────────────────────

def _split_precedent(conn):
    cid = _campaign(conn, "Peru launch")
    _said(conn, cid, "liked", "R. Vega")
    _said(conn, cid, "not_liked", "A. Duarte")
    return cid


def _judge(conn, cid, summary, **kw):
    return core.save_evaluation(
        conn, subject_title="A new brief", campaign_id=None, verdict="approve",
        summary=summary, findings=[], cited_ids=[cid], **kw)


def test_citing_a_contested_precedent_as_unanimous_raises_a_finding(conn):
    """§9.8 built `_confounded_silence` — a check that reads what the model actually WROTE and
    raises a finding when it leaned on a confounded outcome without saying so. §11.5 built the
    sentence telling the model about splits and no check at all.

    That asymmetry is the whole gap: the note carries fifteen other instructions, and this
    product has twice learned that an instruction in a payload gets dropped. A judgment citing
    a two-to-one split as though it were unanimous produced nothing — no finding, no flag —
    which is the exact loss 11.5 exists to prevent, and undetectable."""
    cid = _split_precedent(conn)

    judged = _judge(conn, cid, "Peru worked and they liked it, so this should too.")

    raised = [f for f in judged["findings"] if "different views" in (f.get("finding") or "")]
    assert raised, [f.get("finding") for f in judged["findings"]]
    assert raised[0]["severity"] == "should_fix"
    assert "Peru launch" in raised[0]["finding"]


def test_a_verdict_that_names_the_split_raises_nothing(conn):
    """The finding is the SILENCE, exactly as §9.7 and §9.8 framed it. A judgment that says
    the precedent was contested has said what there was to say, and raising one anyway is the
    wallpaper that teaches a reader to skip server findings."""
    cid = _split_precedent(conn)

    judged = _judge(conn, cid, "Peru is a split precedent — two people disagreed about it — "
                               "so it is weaker support than it looks.")

    assert not [f for f in judged["findings"]
                if "different views" in (f.get("finding") or "")]


def test_an_uncontested_precedent_raises_nothing(conn):
    """The other silence. A finding on every judgment that cites anything would be the
    always-present warning this product refuses everywhere else."""
    cid = _campaign(conn, "Peru launch")
    _said(conn, cid, "liked", "R. Vega")
    _said(conn, cid, "liked", "A. Duarte")

    judged = _judge(conn, cid, "Peru worked and they liked it, so this should too.")

    assert not [f for f in judged["findings"]
                if "different views" in (f.get("finding") or "")]


def test_the_finding_cannot_be_dropped_by_an_approve_with_no_findings(conn):
    """§7.8's machinery: appended after the model's list and exempt from the caps, so an
    `approve` that wrote nothing cannot make it disappear. That is what "impossible to drop"
    meant, and it is why this belongs in `_COMPUTED_FINDINGS` rather than in the prose."""
    cid = _split_precedent(conn)

    judged = _judge(conn, cid, "Fine, ship it.")

    assert [f for f in judged["findings"] if "different views" in (f.get("finding") or "")]
