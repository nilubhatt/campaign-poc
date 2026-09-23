"""§12.2 — the customer's own rulebook, layered over the product's.

The item: "Customer overlay file layering on the default, with the version stamped onto every
saved judgment (pairs with 7.6)."

§12.1 built the file and shipped it EMPTY on purpose: the product owns its judgment
discipline and the customer owns the rules about their briefs. That decision only pays off if
there is somewhere for the customer's half to live that a product upgrade cannot touch —
otherwise "write your rules in rulebook.yaml" means "write them in the file the next installer
overwrites", which is a worse trap than not offering it.

**The stamp is the half that cannot be deferred.** `core.compare_provenance` diffs four
fields and concludes "these two judgments were produced under the same conditions, so an
agreement between them is evidence rather than luck". With one scalar version, two judgments
made under two different overlays both stamp `core-1.0` and that sentence becomes false in the
tool whose entire purpose is explaining disagreement. Widening the field now is a field;
widening it after rows are in the field is a migration.
"""
import pytest

import config
import core
import enums
import facts
import rulebook
import store


@pytest.fixture(autouse=True)
def _forget_the_rulebook():
    rulebook.load.cache_clear()
    yield
    rulebook.load.cache_clear()


def _overlay(conn, text):
    """Write a customer overlay where the product looks for one."""
    path = rulebook.overlay_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    rulebook.load.cache_clear()
    return path


# ---------------------------------------------------------------------------------------
# Where it lives, and why there


def test_the_overlay_lives_where_an_upgrade_cannot_reach_it(conn):
    """In the DATA directory, beside the database — not beside the executable.

    The product's own rulebook sits in the install directory, which is Program Files on
    Windows and is replaced wholesale by the next installer. Telling a customer to write their
    rules there would be telling them to write in the file an upgrade overwrites, which is a
    worse trap than not offering the file at all. The data directory is the one this product
    already promises to keep: it holds the database, and the install disclosure names it.
    """
    assert rulebook.overlay_path().parent == config.DATA_DIR
    assert rulebook.overlay_path() != rulebook._bundled()


def test_with_no_overlay_the_product_behaves_exactly_as_before(conn):
    """A customer who never writes one must not be able to tell this shipped."""
    assert rulebook.version() == "core-1.0"
    assert rulebook.rules() == []
    assert rulebook.overlay() is None


# ---------------------------------------------------------------------------------------
# Layering


def test_an_overlay_rule_is_in_force(conn):
    _overlay(conn, "version: 'acme-3'\nrules:\n  - id: no-ai-imagery\n"
                   "    rule: Never use AI-generated imagery.\n"
                   "    severity: blocking\n    why: Two clients asked in writing.\n")

    rules = rulebook.rules()

    assert [rule["id"] for rule in rules] == ["no-ai-imagery"]
    assert rules[0]["source"] == "acme-3", (
        "whose rule it is, carried on the rule — a judgment citing it says which rulebook it "
        "came from, and once there are two that is the only thing that can"
    )


def test_an_overlay_rule_with_the_same_id_replaces_the_product_one(conn, monkeypatch,
                                                                   tmp_path):
    """Replaces rather than both-apply. Two rules with one id is the case the loader already
    refuses inside a single file, for the reason that a finding citing it names two different
    rules — and layering must not create through the back door what one file cannot express.

    Replacing is also the only useful answer: an overlay exists so a customer can disagree
    with the product, and a customer who cannot turn a shipped rule off has to work around it.
    """
    product = tmp_path / "shipped" / "rulebook.yaml"
    product.parent.mkdir(exist_ok=True)
    product.write_text("version: 'core-9'\nrules:\n  - id: shared\n"
                       "    rule: The product's wording.\n"
                       "    severity: should_fix\n    why: The product's reason.\n")
    monkeypatch.setattr(rulebook, "_bundled", lambda: product)
    _overlay(conn, "version: 'acme-3'\nrules:\n  - id: shared\n"
                   "    rule: The customer's wording.\n"
                   "    severity: blocking\n    why: The customer's reason.\n")

    rules = rulebook.rules()

    assert len(rules) == 1
    assert rules[0]["rule"] == "The customer's wording."
    assert rules[0]["severity"] == "blocking"
    assert rules[0]["source"] == "acme-3"


def test_a_product_rule_the_overlay_does_not_mention_survives(conn, monkeypatch, tmp_path):
    """Layering, not replacement of the whole file. An overlay that silently dropped every
    product rule would make writing one rule cost you all of them, and the customer would have
    no way to notice: the count they would check comes from the same list."""
    product = tmp_path / "shipped" / "rulebook.yaml"
    product.parent.mkdir(exist_ok=True)
    product.write_text("version: 'core-9'\nrules:\n  - id: kept\n    rule: A product rule.\n"
                       "    severity: should_fix\n    why: A reason.\n")
    monkeypatch.setattr(rulebook, "_bundled", lambda: product)
    _overlay(conn, "version: 'acme-3'\nrules:\n  - id: theirs\n    rule: A customer rule.\n"
                   "    severity: blocking\n    why: Another reason.\n")

    assert {rule["id"]: rule["source"] for rule in rulebook.rules()} == {
        "kept": "product", "theirs": "acme-3"}


def test_a_broken_overlay_refuses_and_says_it_is_the_overlay(conn):
    """The same rule as the product file, and the message has to name WHICH file — the
    customer has two, and only one of them is theirs to fix."""
    _overlay(conn, "version: 'acme-3'\nrules:\n  - id: one\n   rule: bad indent\n")

    with pytest.raises(ValueError) as raised:
        rulebook.load()

    assert str(rulebook.overlay_path()) in str(raised.value)


def test_an_overlay_with_no_version_is_refused(conn):
    """It is stamped onto every judgment made under it. An overlay that cannot be named makes
    every judgment under it indistinguishable from one made under the product default —
    which is the exact confusion the stamp exists to prevent."""
    _overlay(conn, "rules:\n  - id: one\n    rule: A rule.\n"
                   "    severity: blocking\n    why: A reason.\n")

    with pytest.raises(ValueError, match="version"):
        rulebook.load()


# ---------------------------------------------------------------------------------------
# The stamp — the half of the item that cannot be deferred


def test_the_stamp_names_both_rulebooks(conn):
    """One scalar, both versions. `compare_provenance` diffs `rulebook_version` and concludes
    two judgments were made "under the same conditions" — so with the overlay unnamed, two
    judgments under two different sets of the customer's own rules would both stamp `core-1.0`
    and be reported as comparable. That is a false statement in the tool whose purpose is
    explaining disagreement."""
    _overlay(conn, "version: 'acme-3'\nrules: []\n")

    assert rulebook.version() == "core-1.0+acme-3"


def test_a_judgment_records_the_overlay_it_was_made_under(conn):
    _overlay(conn, "version: 'acme-3'\nrules: []\n")

    saved = core.save_evaluation(
        conn, subject_title="Bogota launch", verdict="approve",
        summary="Nothing on file bears on this, so it is judged on its own terms.",
        findings=[])

    assert store.get_evaluation(conn, saved["evaluation_id"])["provenance"][
        "rulebook_version"] == "core-1.0+acme-3"


def test_two_judgments_under_different_overlays_are_not_called_comparable(conn):
    """The sentence that would have been false."""
    _overlay(conn, "version: 'acme-3'\nrules: []\n")
    first = core.save_evaluation(
        conn, subject_title="Bogota launch", verdict="approve",
        summary="Nothing on file bears on this, so it is judged on its own terms.",
        findings=[])

    _overlay(conn, "version: 'acme-4'\nrules: []\n")
    second = core.save_evaluation(
        conn, subject_title="Bogota launch", verdict="approve",
        summary="Nothing on file bears on this, so it is judged on its own terms.",
        findings=[])

    compared = core.compare_provenance(conn, first["evaluation_id"],
                                       second["evaluation_id"])

    assert "rulebook_version" in str(compared["differs"])


def test_the_parts_are_readable_without_parsing_the_string(conn):
    """A composite scalar keeps every existing reader correct; splitting it on `+` to find out
    whose rules were in force would be a second parser of this product's own field."""
    _overlay(conn, "version: 'acme-3'\nrules: []\n")

    applied = rulebook.applied()

    assert applied["product"] == "core-1.0"
    assert applied["overlay"] == "acme-3"
    assert applied["version"] == "core-1.0+acme-3"


# ---------------------------------------------------------------------------------------
# D36 — the customer's stage names, and the vocabularies that stay the product's


def test_a_customer_can_name_their_own_stages(conn):
    """D36. "A stage name is customer vocabulary": an agency that says `in_market` or `shipped`
    is describing their own process, and the product refusing it is the product telling them
    how to talk about their work."""
    # One shape for every vocabulary — canonical name, then the other spellings for it.
    # `channels` cannot be expressed the other way round (the channel list IS the
    # declaration), and two shapes to learn for one file is one more than anybody needs.
    _overlay(conn, "version: 'acme-3'\nrules: []\n"
                   "vocabulary:\n  statuses:\n"
                   "    concluded: ['shipped', 'out the door']\n"
                   "    in_flight: ['in the wild']\n")

    cid = core.ingest_campaign(conn, title="Bogota launch", market="LATAM",
                               status="shipped", detail="A store opening.",
                               confirm=True)["campaign_id"]

    assert store.get_campaign(conn, cid)["status"] == "concluded"


def test_a_customer_cannot_redefine_what_this_library_means_by_verified(conn):
    """D36's other half, and the more important one. `verified` has a hard definition here —
    backed by a `metric_type='actual'` row, enforced on write — and every judgment that weighs
    verified evidence more heavily depends on it. A customer mapping "confirmed" to `verified`
    would make "the client confirmed it worked" outweigh a measured result, silently, in every
    comparison the product makes.

    These are the library's epistemics, not the customer's vocabulary, and the refusal says
    so rather than ignoring the key."""
    _overlay(conn, "version: 'acme-3'\nrules: []\n"
                   "vocabulary:\n  tag_sources:\n    verified: ['confirmed']\n")

    with pytest.raises(ValueError) as raised:
        rulebook.load()

    assert "tag_sources" in str(raised.value)
    assert "verified" in str(raised.value)


@pytest.mark.parametrize("key", ["metric_types", "record_types", "tag_sources"])
def test_the_epistemic_vocabularies_are_refused_by_name(conn, key):
    """Named individually, because a customer who declares one is trying to do something
    reasonable-sounding and deserves to be told why not — silence would read as acceptance
    and the key would simply never apply."""
    _overlay(conn, f"version: 'acme-3'\nrules: []\nvocabulary:\n  {key}:\n"
                   f"    mine: ['theirs']\n")

    with pytest.raises(ValueError) as raised:
        rulebook.load()

    said = str(raised.value)
    assert key in said
    # The RATIONALE, not just the key. With the guard deleted the fall-through "not something
    # this product reads" message also names the key, so all three of these passed against a
    # product that had stopped refusing them — and only the `verified` test still failed.
    assert "cannot be declared" in said
    assert "Did you mean" not in said, "this is not a typo; it is a thing they may not do"


def test_an_unknown_vocabulary_key_is_refused_rather_than_ignored(conn):
    """A typo in a config file is the commonest way a declared rule silently does not apply,
    and the customer cannot tell: nothing they can see says the key was never read."""
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  statusses:\n"
                   "    a: ['b']\n")

    with pytest.raises(ValueError) as raised:
        rulebook.load()

    assert "statusses" in str(raised.value)
    assert "statuses" in str(raised.value), "and it names the one they meant"


# ---------------------------------------------------------------------------------------
# D72/D71 — canonical markets, and what region is


def test_a_declared_market_makes_a_spelling_a_synonym_rather_than_a_fold(conn):
    """D72. `latam` and `LATAM` are folded to one thing today by lowercasing, which is a
    stopgap: it cannot make "Latin America" and `LATAM` one market, and those are the same
    market in every library anyone actually has."""
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  markets:\n"
                   "    LATAM:\n      also: ['latin america', 'lat-am', 'latam region']\n")

    cid = core.ingest_campaign(conn, title="Bogota launch", market="Latin America",
                               status="concluded", detail="A store opening.",
                               confirm=True)["campaign_id"]

    # Their words in the row; the declaration says which words mean the same market.
    assert store.get_campaign(conn, cid)["market"] == "Latin America"
    assert store.filter_campaign_ids(conn, market="LATAM") == [cid]
    assert store.fold_market("Latin America") == store.fold_market("LATAM")


def test_an_undeclared_market_is_still_accepted(conn):
    """A declared vocabulary is not a closed one. Refusing an unlisted market would mean a
    customer cannot record a campaign in a country they have not got round to declaring —
    which turns a convenience into a gate on doing the work."""
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  markets:\n"
                   "    LATAM:\n      also: ['latin america']\n")

    cid = core.ingest_campaign(conn, title="Jakarta launch", market="Indonesia",
                               status="concluded", detail="A store opening.",
                               confirm=True)["campaign_id"]

    assert store.get_campaign(conn, cid)["market"] == "Indonesia"
    assert store.filter_campaign_ids(conn, market="Indonesia") == [cid]


def test_the_declared_market_says_which_region_it_is_in(conn):
    """D71: "whether `region` should feed the market grouping at all, or is a different axis".

    It is a different axis, and the declaration is what makes that sayable: a region is a
    GROUPING of markets, so it belongs beside the market list rather than in a free-text field
    that sometimes means a continent and sometimes means a country."""
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  markets:\n"
                   "    Peru:\n      region: LATAM\n    Indonesia:\n      region: APAC\n")

    assert rulebook.region_of("Peru") == "LATAM"
    assert rulebook.region_of("peru") == "LATAM", "the same folding as everywhere else"
    assert rulebook.region_of("Nowhere") is None


# ---------------------------------------------------------------------------------------
# D73 — canonical tag values


def test_a_declared_tag_spelling_is_one_value_not_two(conn):
    """D73. `"not liked"` and `not_liked` are two spellings the code has to know about, and
    the code knowing about them is the problem: every new spelling is a code change.

    The customer's own words stay in the row. A declaration says two words MEAN the same
    thing; it is not permission to rewrite what somebody typed, and rewriting it would make
    every record written before the declaration disagree with every record written after."""
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  tags:\n"
                   "    not_liked:\n      also: ['disliked', 'went down badly']\n")

    cid = core.ingest_campaign(conn, title="Bogota launch", market="LATAM",
                               status="concluded", detail="A store opening.",
                               tags=["went down badly"], confirm=True)["campaign_id"]

    assert [tag["value"] for tag in store.get_campaign(conn, cid)["tags"]] == [
        "went down badly"], "their words, kept"
    assert store.filter_campaign_ids(conn, tags=["not_liked"]) == [cid], (
        "and the product knows it is a not_liked"
    )


def test_a_declared_tag_does_not_move_an_opinion_off_its_axis(conn):
    """The danger in declaring tag words, and the reason the stored value is left alone.

    `_keep_the_view` files an opinion on the axis named by the tag VALUE, and only the seven
    `REACTION_AXES` words have one. Rewriting a stored `not_liked` into the customer's
    "went down badly" would have taken the opinion off the axis entirely: §11.5's append-only
    record would never hear it, which is the loss that whole item exists to prevent.

    So a declared word is a way of FINDING an opinion, never a way of relabelling one."""
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  tags:\n"
                   "    not_liked:\n      also: ['went down badly']\n")
    cid = core.ingest_campaign(conn, title="Bogota launch", market="LATAM",
                               status="concluded", detail="A store opening.",
                               confirm=True)["campaign_id"]

    core.update_campaign(conn, campaign_id=cid, tags=[
        {"value": "not_liked", "source": "stated", "said_by": "R. Vega"}])

    assert [v["value"] for v in store.reactions_for(conn, cid)] == ["not_liked"]
    assert store.filter_campaign_ids(conn, tags=["went down badly"]) == [cid]


# ---------------------------------------------------------------------------------------
# D93 — the channel checklist


def test_a_customer_can_declare_the_channels_they_actually_use(conn):
    """D93. The 360 checklist is the product's fixed list, and "fixed is the point" was the
    right call while nobody could declare one: two users judging the same brief must not
    disagree about which channels it covers. A DECLARED list keeps that property — it is fixed
    per library and per rulebook version — while letting an agency whose work is retail media
    and CTV stop being told it is missing cinema."""
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  channels:\n"
                   "    retail_media: ['retail media', 'onsite retail']\n"
                   "    ctv: ['ctv', 'connected tv']\n")

    computed = facts.compute("A six-week push on connected TV with onsite retail media.")

    assert computed["channels"]["status"] == "present"
    assert set(computed["channels"]["present"]) == {"retail_media", "ctv"}
    assert computed["channels"]["checklist"] == ["retail_media", "ctv"]


def test_the_checklist_says_whose_it_is(conn):
    """`channels.checklist` already names which list was used "so a reader is never guessing
    what missing was measured against". With two possible lists that stops being a nicety."""
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  channels:\n"
                   "    ctv: ['connected tv']\n")

    computed = facts.compute("A six-week push on connected TV.")

    assert computed["channels"]["checklist_source"] == "acme-3"


def test_with_no_declared_channels_the_product_list_still_applies(conn):
    computed = facts.compute("A six-week push on TV and paid social.")

    assert computed["channels"]["checklist_source"] == "product"
    assert len(computed["channels"]["checklist"]) > 2


# ---------------------------------------------------------------------------------------
# D101 — the scorecard


def test_a_declared_scorecard_reaches_the_judgment(conn):
    """D101. §7.4 asks for the scorecard's criteria in the shared procedure, and the product
    cannot ship them: they are one customer's rubric. Declared, they reach the model the same
    way the rules do — in full, on every judgment, not retrieved."""
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary: {}\n"
                   "scorecard:\n"
                   "  - name: Brand fit\n    asks: Does it sound like us?\n"
                   "  - name: Commercial case\n    asks: Is the spend justified by the goal?\n")

    prepared = core.prepare_evaluation(
        conn, subject_title="Bogota launch",
        proposal_text="A six-week push with influencers and paid social.")

    assert "Brand fit" in prepared["note"]
    assert "Is the spend justified by the goal?" in prepared["note"]
    assert prepared["rulebook"]["scorecard"] == 2


def test_the_procedure_no_longer_says_the_scorecard_is_missing(conn):
    """The procedure names the gap "so a reader is not left thinking the step was judged
    unnecessary" — and once a customer HAS declared one, still saying it is outstanding is the
    same defect the other way round."""
    _overlay(conn, "version: 'acme-3'\nrules: []\n"
                   "scorecard:\n  - name: Brand fit\n    asks: Does it sound like us?\n")

    note = core.prepare_evaluation(
        conn, subject_title="Bogota launch",
        proposal_text="A six-week push with influencers.")["note"]

    assert "NO SCORECARD IS DECLARED" not in note, (
        "the earlier version asserted a phrase no code path emits any more, so it passed "
        "whether or not a scorecard was declared"
    )
    assert "Brand fit" in note


def test_a_scorecard_criterion_needs_a_question_it_asks(conn):
    """A name with no question is a heading. The model would have to invent what "Brand fit"
    means, which is the variance the shared procedure exists to remove."""
    _overlay(conn, "version: 'acme-3'\nrules: []\nscorecard:\n  - name: Brand fit\n")

    with pytest.raises(ValueError, match="asks"):
        rulebook.load()


# ---------------------------------------------------------------------------------------
# It has to be discoverable, or it is a file nobody writes


def test_init_says_where_the_customer_rulebook_goes(conn, capsys, monkeypatch):
    """§10.6's rule, applied to a file rather than a tool: a capability nobody is told about
    is one nobody uses. The overlay is the entire point of §12.1 shipping empty, and nothing
    in the product would have mentioned it."""
    import sys

    monkeypatch.setattr(sys, "argv", ["campaign-intelligence", "init"])
    monkeypatch.setattr("config.ensure_dirs", lambda: None)
    monkeypatch.setattr("store.init_db", lambda: None)

    import main

    main.main()

    printed = capsys.readouterr().out
    assert str(rulebook.overlay_path()) in printed


def test_the_health_check_reports_the_overlay(conn):
    _overlay(conn, "version: 'acme-3'\nrules:\n  - id: one\n    rule: A rule.\n"
                   "    severity: blocking\n    why: A reason.\n")

    component = core.health_check(conn, probe=False)["components"]["rulebook"]

    assert "acme-3" in component["detail"]
    assert component["ok"] is True


def test_one_file_serving_as_both_is_not_layered_with_itself(conn, monkeypatch):
    """The data directory is configurable, so a customer can point it at the install
    directory — and then the product's rulebook and the customer's overlay are one file.

    Layered with itself every rule replaces its own twin and the stamp reads `acme-3+acme-3`:
    nonsense, and silent, carried on every judgment made under it. It is read once, as the
    product's, which is the only sane reading of one file."""
    monkeypatch.setattr(config, "app_dir", lambda: config.DATA_DIR)
    _overlay(conn, "version: 'only-one'\nrules:\n  - id: one\n    rule: A rule.\n"
                   "    severity: blocking\n    why: A reason.\n")

    assert rulebook.version() == "only-one"
    assert rulebook.overlay() is None
    assert len(rulebook.rules()) == 1


def test_a_multi_word_status_spelling_actually_applies(conn):
    """The fold on each side of the lookup has to be the SAME fold.

    `_declared` keyed the synonym table with spaces ("out the door") while `enums.normalise`
    looks the value up in underscore shape ("out_the_door"), so every multi-word spelling a
    customer declared was present in the table and unreachable — a declared rule that quietly
    does not apply, which is what this loader exists to refuse.

    The first version of `test_a_customer_can_name_their_own_stages` declared exactly this and
    exercised only the one-word spelling beside it, so its own fixture carried the bug."""
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  statuses:\n"
                   "    concluded: ['out the door', 'wrapped up']\n")

    cid = core.ingest_campaign(conn, title="Bogota launch", market="LATAM",
                               status="wrapped up", detail="A store opening.",
                               confirm=True)["campaign_id"]

    assert store.get_campaign(conn, cid)["status"] == "concluded"


def test_a_status_the_product_does_not_have_is_refused_at_load(conn):
    """D36 is "a customer can add spellings for the stages this product has", not "a customer
    can invent stages" — every gap check, every reconciliation and every "has this concluded"
    question is written against the three.

    Accepted at load, it refused every WRITE instead, citing a word the caller never sent:
    `ingest_campaign` mapped `done` to `shipped` and `insert_campaign` then refused `shipped`.
    A configuration error has to surface when the configuration is read."""
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  statuses:\n"
                   "    shipped: ['done']\n")

    with pytest.raises(ValueError) as raised:
        rulebook.load()

    assert "shipped" in str(raised.value)
    assert "proposed" in str(raised.value), "and it lists the stages there are"


def test_a_tag_spelling_cannot_be_another_axis_word(conn):
    """The epistemic guard's own reasoning, applied where it also holds.

    `tag_sources` is refused because "mapping another word onto it would make a stated
    impression outweigh a measured result". The seven `REACTION_AXES` values decide the `axis`
    column on the append-only record and drive every quadrant query — and `tags` could remap
    them freely, including inverting them. `not_liked: ['liked']` made a liked campaign read
    as disliked; `performed_well: ['client loved it']` put a stated opinion on the PERFORMANCE
    axis, where it answers "what performed well"."""
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  tags:\n"
                   "    not_liked: ['liked']\n")

    with pytest.raises(ValueError) as raised:
        rulebook.load()

    assert "liked" in str(raised.value)


def test_a_tag_can_still_be_given_the_customers_own_words(conn):
    """The other half: `not_liked: ['went down badly']` is the whole point of D73, and a guard
    that refused it would have removed the feature to fix the flaw."""
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  tags:\n"
                   "    not_liked: ['went down badly']\n")

    assert rulebook.canonical("tags", "went down badly") == "not_liked"


def test_one_spelling_cannot_mean_two_things(conn):
    """The loader refuses two rules with one id "because nobody reading it could tell which
    was breached". Two canonicals with one spelling is the same defect, and it resolved
    DIFFERENTLY in two places — `rulebook.canonical` took the first match and `store._declared`
    built a dict where the last won. One file, one spelling, two answers."""
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  tags:\n"
                   "    liked: ['x']\n    not_liked: ['x']\n")

    with pytest.raises(ValueError, match="'x'"):
        rulebook.load()


def test_a_duplicate_key_is_refused_rather_than_silently_dropping_the_first(conn):
    """PyYAML keeps the last of duplicate mapping keys and says nothing, so an overlay with
    two `rules:` blocks loses the first one entirely — and the customer's count comes from the
    same list that dropped it. Every other way of writing this file wrong is refused loudly;
    this one defeated all of it at the parse step."""
    _overlay(conn, "version: 'acme-3'\n"
                   "rules:\n  - id: a\n    rule: First.\n    severity: blocking\n"
                   "    why: A reason.\n"
                   "rules:\n  - id: b\n    rule: Second.\n    severity: blocking\n"
                   "    why: Another reason.\n")

    with pytest.raises(ValueError, match="rules"):
        rulebook.load()


def test_a_version_that_is_not_a_string_is_refused(conn):
    """`version: 1.10` is a YAML FLOAT, so it stamps `1.1` — and a customer bumping 1.1 to
    1.10 moves their rules under a stamp that did not move, which is the exact failure the
    file's own VERSIONS note warns about."""
    _overlay(conn, "version: 1.10\nrules: []\n")

    with pytest.raises(ValueError, match="quote"):
        rulebook.load()


def test_a_dangling_overlay_symlink_is_not_read_as_no_overlay(conn, tmp_path):
    """Every other unreadable shape — a directory, a mode-000 file, a bad tag — refuses
    loudly. A symlink whose target has moved returns False from `exists()` and fell into the
    no-overlay branch, so a customer whose checkout moved got every judgment stamped as though
    they had never written any rules."""
    target = tmp_path / "gone" / "rulebook.yaml"
    path = rulebook.overlay_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target)
    rulebook.load.cache_clear()

    with pytest.raises(ValueError) as raised:
        rulebook.load()

    assert "symlink" in str(raised.value) or "link" in str(raised.value)


# ---------------------------------------------------------------------------------------
# The read path
#
# Canonicalising on WRITE alone splits a library the moment a vocabulary is declared: rows
# written before it keep the old spelling, rows after get the new one, and a query in either
# spelling finds half of them. The declaration — which exists to say "these are the same
# thing" — was what created the split.


def _in_market(conn, title, market):
    return core.ingest_campaign(conn, title=title, market=market, status="concluded",
                                detail=f"A campaign called {title}, which ran somewhere.",
                                confirm=True)["campaign_id"]


def test_a_declaration_reaches_records_written_before_it(conn):
    """No migration, and none needed: the declaration says two spellings are one market, so
    a query for either finds both. Rewriting history would be the other answer and a worse
    one — it edits records a customer may have exported, and it cannot run on the rows
    somebody adds tomorrow from a spreadsheet that still says the old word."""
    before = _in_market(conn, "Bogota launch", "Latin America")
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  markets:\n"
                   "    LATAM:\n      also: ['latin america']\n")
    after = _in_market(conn, "Lima launch", "LATAM")

    assert set(store.filter_campaign_ids(conn, market="LATAM")) == {before, after}
    assert set(store.filter_campaign_ids(conn, market="Latin America")) == {before, after}


def test_the_same_holds_for_a_tag_spelling(conn):
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  tags:\n"
                   "    not_liked: ['went down badly']\n")
    conn.execute("UPDATE campaigns SET tags = ? WHERE id = ?",
                 ('[{"value": "went down badly", "source": "stated"}]',
                  _in_market(conn, "Bogota launch", "LATAM")))
    conn.commit()
    after = _in_market(conn, "Lima launch", "LATAM")
    core.update_campaign(conn, campaign_id=after, tags=["not_liked"])

    assert len(store.filter_campaign_ids(conn, tags=["not_liked"])) == 2
    assert len(store.filter_campaign_ids(conn, tags=["went down badly"])) == 2


def test_two_spellings_of_one_market_are_one_cell(conn):
    """The consequence the write-path fix was reaching for: `canonical_market`'s own docstring
    says the two "sat in separate cells, each with its own thin evidence, and nothing said
    they were the same place". That stays true of every pre-existing row unless the READ side
    knows about the declaration too."""
    _in_market(conn, "Bogota launch", "Latin America")
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  markets:\n"
                   "    LATAM:\n      also: ['latin america']\n")
    _in_market(conn, "Lima launch", "LATAM")

    cells = {store.fold_market(row["market"])
             for row in (store.get_campaign(conn, cid)
                         for cid in store.filter_campaign_ids(conn))}

    assert cells == {"latam"}, f"two spellings of one market made two cells: {cells}"


def test_the_product_files_vocabulary_and_scorecard_are_not_thrown_away(conn, monkeypatch,
                                                                        tmp_path):
    """Rules layered and the other two sections did not: with no overlay the product's own
    `vocabulary` and `scorecard` were replaced by empty defaults, and with one they were
    replaced wholesale — so an overlay declaring only `tags` erased a product `statuses`
    declaration, and the contract printed "NO SCORECARD IS DECLARED" while one sat in the file
    it had just read.

    It matters now rather than later: D36's row says `STATUS_SYNONYMS` MOVES INTO the bundled
    rulebook, which this branch would have silently discarded."""
    product = tmp_path / "shipped" / "rulebook.yaml"
    product.parent.mkdir(exist_ok=True)
    product.write_text("version: 'core-9'\nrules: []\n"
                       "vocabulary:\n  statuses:\n    concluded: ['wrapped up']\n"
                       "scorecard:\n  - name: Evidence\n    asks: Does it cite anything?\n")
    monkeypatch.setattr(rulebook, "_bundled", lambda: product)
    rulebook.load.cache_clear()

    assert rulebook.canonical("statuses", "wrapped up") == "concluded"
    assert [c["name"] for c in rulebook.scorecard()] == ["Evidence"]

    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  tags:\n"
                   "    not_liked: ['went down badly']\n")

    assert rulebook.canonical("statuses", "wrapped up") == "concluded", (
        "declaring tags must not erase the statuses the product declared"
    )
    assert rulebook.canonical("tags", "went down badly") == "not_liked"
    assert [c["name"] for c in rulebook.scorecard()] == ["Evidence"]


def test_an_overlay_can_replace_one_of_the_products_declarations(conn, monkeypatch, tmp_path):
    """Layering, with the same rule as rules: same key replaces, everything else survives."""
    product = tmp_path / "shipped" / "rulebook.yaml"
    product.parent.mkdir(exist_ok=True)
    product.write_text("version: 'core-9'\nrules: []\n"
                       "vocabulary:\n  statuses:\n    concluded: ['wrapped up']\n")
    monkeypatch.setattr(rulebook, "_bundled", lambda: product)
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  statuses:\n"
                   "    concluded: ['out the door']\n")

    assert rulebook.canonical("statuses", "out the door") == "concluded"
    assert rulebook.canonical("statuses", "wrapped up") is None


def test_a_customer_can_declare_their_collection_names(conn):
    """D72's other half — "canonical market AND COLLECTION names from the rulebook". It was
    absent from the allowed list, so declaring one refused the whole file and the server would
    not start: the row's own example, answered with a crash."""
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  collections:\n"
                   "    spring-drop-2026: ['spring drop', 'ss26 drop']\n")

    cid = core.ingest_campaign(conn, title="Bogota launch", market="LATAM",
                               status="concluded", detail="A store opening.",
                               collection="spring drop", confirm=True)["campaign_id"]

    assert store.filter_campaign_ids(conn, collection="spring-drop-2026") == [cid]


def test_a_declared_region_is_not_counted_as_a_market(conn):
    """D71, answered in code rather than in a docstring.

    `markets_of` folds `market`, `region` and `markets` into "every market this campaign
    counts towards" — so a campaign in Peru whose region is LATAM counted as TWO markets and
    satisfied §8.3's "seen in at least two markets" gate on its own. That gate exists so that
    breadth has to be earned.

    Only where the rulebook declares the region: undeclared, `region` is free text that means
    a continent on one record and a country on the next, which is why D71 was a question."""
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  markets:\n"
                   "    Peru:\n      region: LATAM\n")

    counts = store.markets_of({"market": "Peru", "region": "LATAM", "markets": []})

    assert counts == ["Peru"], f"a region counted as a second market: {counts}"


def test_an_undeclared_region_still_counts_as_it_always_did(conn):
    """The other half. Guessing would drop a real market from a library that never declared
    anything — plenty of records use `region` to mean the one place they ran."""
    counts = store.markets_of({"market": "Peru", "region": "LATAM", "markets": []})

    assert counts == ["Peru", "LATAM"]


# ---------------------------------------------------------------------------------------
# Discoverability — the whole payoff of §12.1 shipping empty
#
# Every in-product pointer named `rulebook.yaml` with no directory, which is the file the
# product's own header now says DO NOT EDIT. A customer following the only instruction the
# product gave them would have written their rules into the file the next installer replaces.


def test_readiness_names_the_file_the_customer_should_actually_write(conn):
    row = next(r for r in core.readiness(conn)["cannot"]
               if r["code"] == "check_against_rules")

    assert str(rulebook.overlay_path()) in row["needs"]


def test_the_health_check_names_the_overlay_and_where_it_would_go(conn):
    """It printed the INSTALL file's path even when the rules came from the overlay, so an
    administrator reading "0 rule(s) from /Applications/.../rulebook.yaml" had no lead at
    all."""
    without = core.health_check(conn, probe=False)["components"]["rulebook"]
    assert str(rulebook.overlay_path()) in without["detail"]

    _overlay(conn, "version: 'acme-3'\nrules:\n  - id: one\n    rule: A rule.\n"
                   "    severity: blocking\n    why: A reason.\n")
    with_one = core.health_check(conn, probe=False)["components"]["rulebook"]

    assert "acme-3" in with_one["detail"]
    assert str(rulebook.overlay_path()) in with_one["detail"]
    assert "1 rule" in with_one["detail"]


def test_a_broken_overlay_is_not_blamed_on_the_installer(conn):
    """The remedy said "re-run the installer, which restores the shipped copy" — false for an
    overlay, which lives in the data directory the installer never touches. A remedy that
    cannot work is worse than none: it costs somebody a reinstall before they look further."""
    _overlay(conn, "version: 'acme-3'\nrules:\n  - id: one\n   rule: bad indent\n")

    component = core.health_check(conn, probe=False)["components"]["rulebook"]

    assert component["ok"] is False
    assert str(rulebook.overlay_path()) in component["remedy"]
    assert "installer" not in component["remedy"]


def test_an_overlay_added_while_the_server_runs_is_reported_not_ignored(conn):
    """The rulebook is read once per process, which is right — a judgment must not depend on
    when in a session it was made. But the overlay is the file the product actively invites
    the customer to create, under a connector that stays up for days, so "written and not
    loaded" is the ordinary case rather than an edge one.

    The stamp stays honest either way; what was missing is that nothing told them."""
    core.health_check(conn, probe=False)   # load with no overlay present
    _overlay_path = rulebook.overlay_path()
    _overlay_path.parent.mkdir(parents=True, exist_ok=True)
    _overlay_path.write_text("version: 'acme-9'\nrules: []\n", encoding="utf-8")

    component = core.health_check(conn, probe=False)["components"]["rulebook"]

    assert component["degraded"], "a rulebook on disk that is not in force has to be said"
    assert "restart" in component["degraded"].lower()


def test_the_contract_says_where_a_scorecard_would_go(conn):
    """`readiness`' own discipline — "each entry names the record that would lift it, so it is
    a next step rather than a disclaimer" — applied to the one line that named a gap and no
    remedy. A model reading it learned not to invent a rubric and had nothing to tell the
    user."""
    note = core.prepare_evaluation(
        conn, subject_title="Bogota launch",
        proposal_text="A six-week push with influencers.")["note"]

    assert "NO SCORECARD IS DECLARED" in note
    assert str(rulebook.overlay_path()) in note


def test_a_declared_channel_is_found_by_its_own_name(conn):
    """The pattern set was `entry["also"] or [name]` — so giving a channel ANY other spelling
    EXCLUDED its canonical name, and the product told a customer their brief was missing a
    channel the brief names. `rulebook.canonical` matched name-or-spelling, which made two
    implementations of "which spellings count", disagreeing."""
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  channels:\n"
                   "    ctv: ['connected tv']\n    cinema: []\n")

    computed = facts.compute("A six-week push on CTV and in cinema.")

    assert set(computed["channels"]["present"]) == {"ctv", "cinema"}
    assert computed["channels"]["status"] == "present"


def test_a_declared_channel_matches_across_punctuation(conn):
    """The declaration is stored with punctuation folded out ("out-of-home" becomes "out of
    home"), and the folded string was then `re.escape`d against the brief as written — so the
    brief had to spell it the way the fold left it, which nobody can predict."""
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  channels:\n"
                   "    ooh: ['out-of-home']\n")

    assert facts.compute("A six-week out of home push.")["channels"]["present"] == ["ooh"]
    assert facts.compute("A six-week out-of-home push.")["channels"]["present"] == ["ooh"]


def test_an_unreadable_rulebook_does_not_claim_the_product_list_was_intended(conn):
    """Every write path refuses a broken rulebook and `health_check` reports it; this one
    swallowed the error and reported `checklist_source: "product"` — a false statement when a
    customer's rulebook exists and could not be read."""
    _overlay(conn, "version: 'acme-3'\nrules: []\nvocabulary:\n  channels:\n   ctv: bad\n")

    source = facts.compute("A push on TV.")["channels"]["checklist_source"]

    assert source != "product"
    assert "could not be read" in source


# ---------------------------------------------------------------------------------------
# D28, D23, D115 — the other three rows this item owed


def test_the_uninstaller_names_the_rulebook_before_deleting_it(conn):
    """D28. `--purge` prints what it is about to remove and asks for a typed confirmation,
    because "an uninstaller that silently deletes a year of campaign history is not a
    trade-off anyone agreed to". The customer's rulebook is now in that directory, and it is
    the one file in it they WROTE rather than accumulated — a rebuildable library of campaigns
    is a different loss from a set of rules somebody sat down and agreed.

    Read from the scripts rather than run: they delete a home directory."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for script in ("installer/macos/uninstall.sh", "installer/linux/uninstall.sh"):
        text = (root / script).read_text(encoding="utf-8")
        assert "rulebook.yaml" in text, f"{script} never mentions the rulebook"
        # In what the script SAYS, in the block that lists what is about to go — not
        # somewhere in the file, and not in a comment. The first version of this assertion
        # matched the comment above the line and passed with the line deleted; mutation
        # found it.
        warning = text[text.index("About to delete"):].split("Type DELETE")[0]
        said = "\n".join(line for line in warning.splitlines()
                         if "echo" in line and not line.strip().startswith("#"))
        assert "rulebook" in said.lower(), (
            f"{script} does not name the rulebook in what it prints before deleting"
        )


def test_the_overlay_can_name_who_support_is(conn):
    """D23. `notices` tells somebody to "ask whoever installed this", which is the best a
    product that ships to strangers can do — and the customer knows the answer. Declared, a
    remedy can name them."""
    _overlay(conn, "version: 'acme-3'\nrules: []\n"
                   "support: 'Ops team, #campaign-help on Slack'\n")

    assert rulebook.support() == "Ops team, #campaign-help on Slack"


def test_a_machine_scoped_remedy_names_them_instead_of_guessing(conn):
    """The point of declaring it: the sentence changes, everywhere that sentence appears."""
    import notices

    _overlay(conn, "version: 'acme-3'\nrules: []\nsupport: 'Ops team, #campaign-help'\n")

    said = notices.notice("visual_search_offline", detail="The weights are missing.")

    # The DECLARED value, read from the rulebook rather than typed here — this asserts the
    # wiring, not the wording, which is what `test_rewording_a_remedy_is_not_a_breaking_change`
    # asks of every test that touches a remedy.
    assert rulebook.support() in said["remedy"]


def test_with_no_support_declared_the_remedy_is_what_it_always_was(conn):
    import notices

    said = notices.notice("visual_search_offline", detail="The weights are missing.")

    assert rulebook.support() is None
    assert "Here, that is" not in said["remedy"]


def test_the_replay_notices_that_the_rulebook_itself_changed(conn):
    """D115. The replay is built from `confirmed_at` on registry and correction rows — and a
    rulebook's own rules carry no `confirmed_at`, so a v1→v2 rulebook change produced a replay
    reporting that nothing had changed. A silent false negative on precisely the axis §8.7
    was named for.

    It could not be built before there were rulebook versions to compare. There are now: the
    stamp on each judgment says which rulebook it was made under."""
    _overlay(conn, "version: 'acme-3'\nrules: []\n")
    first = core.save_evaluation(
        conn, subject_title="Bogota launch", verdict="approve",
        summary="Nothing on file bears on this, so it is judged on its own terms.",
        findings=[])
    _overlay(conn, "version: 'acme-4'\nrules: []\n")
    core.save_evaluation(
        conn, subject_title="Lima launch", verdict="approve",
        summary="Nothing on file bears on this, so it is judged on its own terms.",
        findings=[])

    import replay

    report = replay.run(conn)

    assert report["rulebook_changed"] is True
    assert {"core-1.0+acme-3", "core-1.0+acme-4"} <= set(report["rulebooks_seen"])
    # The note has to SAY it, and say which — a boolean nobody reads is the silent false
    # negative with the sign flipped. Both stamps appear, because "the rules moved" is only
    # actionable if you can see between which two.
    said = report["rulebook_note"]
    assert "changed" in said.lower()
    assert "core-1.0+acme-3" in said and "core-1.0+acme-4" in said
    assert first["evaluation_id"]


def test_one_rulebook_throughout_is_reported_as_one(conn):
    """The other half. A replay that cried "the rules changed" on every library would be as
    useless as one that never did."""
    import replay

    _overlay(conn, "version: 'acme-3'\nrules: []\n")
    core.save_evaluation(
        conn, subject_title="Bogota launch", verdict="approve",
        summary="Nothing on file bears on this, so it is judged on its own terms.",
        findings=[])

    report = replay.run(conn)

    assert report["rulebook_changed"] is False
    assert report["rulebooks_seen"] == ["core-1.0+acme-3"]
    assert "One rulebook throughout" in report["rulebook_note"]
