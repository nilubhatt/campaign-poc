"""§12.3 — an example customer config, and the two rows that could not land before it.

The item: "the full transcribed content (guardrails, tag taxonomy, influencer criteria + red
flags, partner feedback, 360 checklist, scorecard) and the ten standing corrections with
provenance, shipped as an **example/customer** file, not the product default."

**The transcribed content is not in this repository.** It lives in the 11 Sep product review,
which is the customer's own material; the plan's own decision says the product stays generic
and their rules ship as an example that layers on top. What this file tests is everything that
does NOT depend on having their words: that an example file exists, that it is loadable rather
than illustrative prose, that it is never mistaken for the product's default, and that the two
tracker rows waiting on it now close.

Written this way on purpose. An example nobody can load is a document; §11.7's install
disclosure shipped once as a string generator with no call sites, and the tracker called it
finished.
"""
from pathlib import Path

import pytest

import config
import core
import enums
import learning
import rulebook
import store

EXAMPLE = Path(__file__).resolve().parent.parent / "docs" / "example-rulebook.yaml"


@pytest.fixture(autouse=True)
def _forget_the_rulebook():
    rulebook.load.cache_clear()
    yield
    rulebook.load.cache_clear()


def _as_the_customers(conn):
    """Install the example where a customer's own rulebook goes."""
    path = rulebook.overlay_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    rulebook.load.cache_clear()
    return path


# ---------------------------------------------------------------------------------------
# It is an example, and it loads


def test_an_example_config_ships():
    assert EXAMPLE.exists(), f"no example at {EXAMPLE}"


def test_the_example_is_not_the_product_default(conn):
    """The product-owner decision, in as many words: "the product stays generic... never as
    the product's baked-in default". A customer who installs this product and writes nothing
    gets no rules, no vocabulary and no scorecard."""
    assert rulebook.rules() == []
    assert rulebook.scorecard() == []
    assert rulebook.overlay() is None
    assert EXAMPLE.parent != config.app_dir(), "it must not sit where the default is read"


def test_the_example_loads_as_a_real_rulebook(conn):
    """Loadable, not illustrative. An example that has drifted out of the format is worse
    than none: it is the first thing a customer copies, and it fails in their hands rather
    than in the build."""
    _as_the_customers(conn)

    loaded = rulebook.load()

    assert loaded["overlay_version"], "the example has to name its own version"
    assert loaded["rules"], "an example rulebook with no rules teaches nothing"


def test_every_part_the_item_names_is_present(conn):
    """The item lists what the transcription covers. Each part is a different SHAPE in the
    file, and an example that showed three of them would leave a customer guessing at the
    rest — which is what they are reading it for."""
    _as_the_customers(conn)
    loaded = rulebook.load()

    assert loaded["rules"], "guardrails"
    assert loaded["vocabulary"]["tags"], "tag taxonomy"
    assert loaded["vocabulary"]["channels"], "the 360 checklist"
    assert loaded["scorecard"], "the scorecard"
    assert loaded["expects"], "what a brief must carry"
    assert any(rule["watch_for"] for rule in loaded["rules"]), (
        "at least one rule the SERVER can check, or the example teaches only the half a "
        "model has to apply by reading"
    )


def test_the_example_says_it_is_an_example(conn):
    """A file copied into a data directory loses its path. It has to say what it is in its
    own text, because the version string is what lands on every judgment made under it."""
    _as_the_customers(conn)

    assert "example" in rulebook.load()["overlay_version"].lower()


def test_installing_it_does_not_change_what_the_product_claims_about_itself(conn):
    """It is somebody's rules, so it changes what can be CHECKED — and it must not change a
    word about the product's own discipline, which is `EVALUATION_PROCEDURE`'s."""
    before = core.EVALUATION_PROCEDURE
    _as_the_customers(conn)
    after = core.prepare_evaluation(
        conn, subject_title="Bogota launch",
        proposal_text="A six-week push with creators.")["note"]

    # The procedure reaches the model INSIDE the note, so this asserts what the model is
    # actually told rather than that a module constant equals itself, which is what the first
    # version compared.
    assert before in after
    assert core.EVALUATION_PROCEDURE == before


# ---------------------------------------------------------------------------------------
# D37 — "in market" is the customer's word, not the product's


def test_in_market_is_no_longer_a_product_synonym():
    """D37. `in_market` sat in `STATUS_SYNONYMS` as a product default — one agency's phrasing
    baked into a product that ships generic, which is the thing the rulebook decision rules
    out. It waited for D36 because it needed somewhere to go."""
    assert "in_market" not in enums.STATUS_SYNONYMS


def test_the_example_is_where_in_market_now_lives(conn):
    """Moved, not deleted: it is a real word real people use, and an agency that says it
    should find it in the example they are copying from."""
    _as_the_customers(conn)

    assert rulebook.canonical("statuses", "in market") == "in_flight"

    cid = core.ingest_campaign(conn, title="Bogota launch", market="LATAM",
                               status="in market", detail="A store opening.",
                               confirm=True)["campaign_id"]
    assert store.get_campaign(conn, cid)["status"] == "in_flight"


def test_without_the_example_in_market_is_refused_with_the_valid_set(conn):
    """And the refusal teaches, which is §5.1's whole point — a word the product no longer
    knows must come back with what it does know, not a bare rejection."""
    with pytest.raises(Exception) as raised:
        core.ingest_campaign(conn, title="Bogota launch", market="LATAM",
                             status="in market", detail="A store opening.", confirm=True)

    assert "in_flight" in str(raised.value)


# ---------------------------------------------------------------------------------------
# D108 — a rule that arrives already confirmed


def test_a_rule_from_the_customers_own_file_does_not_wait_for_the_gate(conn):
    """D108. The ten standing corrections §12.3 ships have provenance — a deck and a slide —
    and no `campaign_id` in this library, so the gate refuses them forever: seen in 0
    campaigns, needs 3. "The item built to give them somewhere to live and grow gives them
    somewhere they can be stored and never applied."

    An overlay-sourced rule is confirmed by the customer BY DEFINITION: it is in their own
    file, which they wrote. The gate exists to stop the LIBRARY promoting a rule it inferred
    from one partner's house style — a different question, and not one a customer's own
    written rule has to answer."""
    _as_the_customers(conn)

    decided = learning.gate(name="Never use AI-generated imagery", noun="rule",
                            campaigns=0, markets=[], status="provisional",
                            from_rulebook="example-1")

    assert decided["eligible"] is True
    assert decided["code"] == "declared_by_the_customer"
    assert "own rulebook" in decided["what_it_means"]


def test_a_rule_the_library_inferred_still_has_to_earn_it(conn):
    """The other half, and the one that matters: the anti-capture gate is untouched. A rule
    the library noticed for itself, in one market, is still refused — otherwise this would
    have removed the gate rather than routed around it for a case it was never about."""
    decided = learning.gate(name="Seed one colourway per creator", noun="rule",
                            campaigns=1, markets=["Peru"], status="provisional")

    assert decided["eligible"] is False
    assert decided["code"] == "not_yet"


def test_the_rule_still_needs_a_person_against_it(conn):
    """What the shortcut does NOT skip. §8.2 spent the loop's only human step on "who says
    so", and a rule arriving from a file is not a person confirming it — somebody still has
    to put their name to promoting it."""
    _as_the_customers(conn)

    with pytest.raises(ValueError, match="PERSON"):
        learning.require_a_person(None)

    assert learning.require_a_person("R. Vega") == "R. Vega"


def test_where_the_rule_came_from_is_recorded(conn):
    """A rule promoted without the gate has to say why it was allowed to skip it. Otherwise
    the audit trail reads identically to one that met three campaigns in two markets, and
    nobody later can tell a customer's declaration from the library's inference."""
    _as_the_customers(conn)

    decided = learning.gate(name="Never use AI-generated imagery", noun="rule",
                            campaigns=0, markets=[], status="provisional",
                            from_rulebook="example-1")

    assert decided["from_rulebook"] == "example-1"
    assert decided["basis"] == "stated"


# ---------------------------------------------------------------------------------------
# The ten standing corrections, and the route D108 says they need
#
# A rulebook RULE and a standing CORRECTION are different things and the product keeps them
# apart on purpose: a rule is something the customer wrote down, a correction is something the
# library watched recur until somebody confirmed it. The item ships both, because an agency
# arriving with this product has both — rules they have always had, and the ten things they
# find themselves saying on every deck.


def test_the_example_ships_standing_corrections_with_provenance(conn):
    """"the ten standing corrections WITH PROVENANCE". A correction with no provenance can
    only say "the library says so", which `corrections.note` refuses for the reason this whole
    product exists: it is the unfounded confident claim."""
    _as_the_customers(conn)

    declared = rulebook.corrections()

    assert len(declared) >= 10, f"the item says ten; the example has {len(declared)}"
    for entry in declared:
        assert entry["text"] and entry["provenance"], entry
        assert entry["provenance"] != entry["text"], "provenance is where it came FROM"


def test_a_correction_with_no_provenance_is_refused(conn, tmp_path):
    """The requirement `corrections.note` already makes, at the door the rulebook opens.
    Without provenance a judgment citing the rule can say only "the library says so", which is
    the unfounded confident claim this product is built against — and mutation showed the
    check was asserted nowhere."""
    path = rulebook.overlay_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("version: 'acme-3'\nrules: []\ncorrections:\n"
                    "  - text: Never put the logo on a dark background.\n", encoding="utf-8")
    rulebook.load.cache_clear()

    with pytest.raises(ValueError, match="provenance"):
        rulebook.load()


def test_the_declared_corrections_are_in_force_without_three_campaigns(conn):
    """D108, end to end. They have a deck and a slide against them and no `campaign_id` in
    this library, so the counting gate refused them forever — stored and never applied."""
    _as_the_customers(conn)

    loaded = core.load_declared_corrections(conn, confirmed_by="R. Vega")

    assert loaded["loaded"] >= 10
    standing = corrections_in_force(conn)
    assert len(standing) >= 10, "declared and not in force is the failure D108 names"


def corrections_in_force(conn):
    import corrections as corrections_module

    return [c for c in corrections_module.all_of_them(conn)
            if c.get("status") == "expected"]


def test_a_declared_correction_reaches_a_judgment(conn):
    """The point of loading them: a brief is judged against them. Stored-and-never-applied is
    the state D108 describes, and it is invisible from the row itself."""
    _as_the_customers(conn)
    core.load_declared_corrections(conn, confirmed_by="R. Vega")

    prepared = core.prepare_evaluation(
        conn, subject_title="Bogota launch", market="Peru",
        proposal_text="A six-week push with creators in Lima.")

    # `["standing"]`, not the dict. `standing_corrections` is the whole reply from
    # `corrections.standing_for` and always carries `market`, `basis`, `code` and a sentence —
    # so it is truthy when ZERO corrections apply, and the earlier version of this assertion
    # passed with the feature entirely broken. Its failure message was the true sentence.
    standing = prepared["standing_corrections"]["standing"]

    assert standing, (
        f"none of them reached the judgment: "
        f"{prepared['standing_corrections']['what_it_means']}"
    )
    assert len(standing) >= 9, "a house rule with no market applies everywhere, not nowhere"


def test_a_declared_correction_with_no_market_applies_in_every_market(conn):
    """"Empty means everywhere" is what both the loader and the example file say, and it is
    what a house rule means — "the logo never sits on a photographic background" is not a
    Peru rule. It was implemented as NOWHERE: `expected_in = []` and
    `standing_for` intersects the record's markets with that list, and an empty set
    intersects nothing.

    So nine of the ten corrections the tool reported "now in force" reached no judgment in any
    market. That is exactly the state D108 names — stored and never applied — shipped as the
    fix for D108."""
    _as_the_customers(conn)
    core.load_declared_corrections(conn, confirmed_by="R. Vega")

    import corrections as corrections_module

    for market in ("Peru", "Indonesia", "a market nobody declared"):
        reply = corrections_module.standing_for(conn, campaign_id=None, markets=[market])
        assert reply["standing"], f"nothing stood in {market}: {reply['what_it_means']}"


def test_a_correction_the_library_learned_is_still_scoped_to_its_markets(conn):
    """The other half, and the reason `expected_in = []` meant "no checklist" in the first
    place. A rule the library INFERRED is expected where it was actually seen — that is all
    the evidence supports, and §8.3's gate exists to keep it that way. Only a rule somebody
    DECLARED gets to say "everywhere", because they are the ones who know."""
    import corrections as corrections_module

    cid = core.ingest_campaign(conn, title="Lima launch", market="Peru", status="concluded",
                               detail="A creator push in Lima.", confirm=True)["campaign_id"]
    noted = corrections_module.note(conn, text="Seed one colourway per creator.",
                                    campaign_id=cid, provenance="Client call, 3 March")
    store.graduate_correction(conn, noted["correction_id"], markets=["Peru"],
                              confirmed_by="R. Vega")

    in_peru = corrections_module.standing_for(conn, campaign_id=None, markets=["Peru"])
    elsewhere = corrections_module.standing_for(conn, campaign_id=None, markets=["Indonesia"])

    assert [c["text"] for c in in_peru["standing"]] == ["Seed one colourway per creator."]
    assert elsewhere["standing"] == []


def test_loading_them_twice_does_not_double_them(conn):
    """The file is read at every start and a customer edits it; a loader that appended would
    turn one rule into five over a week of restarts, each looking like independent
    confirmation of the same thing."""
    _as_the_customers(conn)

    first = core.load_declared_corrections(conn, confirmed_by="R. Vega")
    again = core.load_declared_corrections(conn, confirmed_by="R. Vega")

    assert again["loaded"] == 0
    assert again["already"] == first["loaded"]
    assert len(corrections_in_force(conn)) == first["loaded"]


def test_a_declared_correction_says_it_came_from_the_rulebook(conn):
    """Its provenance is the customer's own file, and a judgment citing it has to be able to
    say so — otherwise it reads exactly like one the library inferred and a person confirmed
    after three campaigns, which is a different claim about how much is known."""
    _as_the_customers(conn)
    core.load_declared_corrections(conn, confirmed_by="R. Vega")

    import corrections as corrections_module

    entry = corrections_in_force(conn)[0]
    # On the SIGHTING, which is where a correction's provenance lives — the correction is the
    # rule, and the sightings are the occasions somebody said it.
    seen = corrections_module.sightings(conn, entry["id"])

    assert seen, "a declared correction with no sighting has no provenance at all"
    assert rulebook.overlay() in seen[0]["provenance"]
    assert "slide" in seen[0]["provenance"] or "email" in seen[0]["provenance"], (
        "and the customer's own provenance survives beside the file's name"
    )


def test_loading_needs_a_person(conn):
    """The half the shortcut does not skip."""
    _as_the_customers(conn)

    with pytest.raises(ValueError, match="PERSON"):
        core.load_declared_corrections(conn, confirmed_by="the system")

    # NOTHING WRITTEN. With the guard removed the refusal still arrived — from
    # `record_authorship`, after `graduate_correction` had committed a rule into force under
    # "the system". A raise that happens after the row exists is not a refusal; it is a
    # half-written record with a matching exception, and this product has now shipped that
    # shape twice.
    import corrections as corrections_module

    assert corrections_module.all_of_them(conn) == []


def test_with_no_declared_corrections_nothing_is_loaded_and_it_says_so(conn):
    loaded = core.load_declared_corrections(conn, confirmed_by="R. Vega")

    assert loaded["loaded"] == 0
    assert "no" in loaded["what_it_means"].lower()


# ---------------------------------------------------------------------------------------
# D116 — a capability nobody offers is one nobody calls


def test_loading_them_is_offered_when_the_rulebook_declares_some(conn):
    """§10.6's rule, and this product has hit it eight times. A customer writes ten
    corrections into their rulebook, restarts, and nothing happens: the rules sit in the file
    and the library never mentions them. That is the state D108 describes — stored and never
    applied — arriving one level up, through nobody being told there was something to do."""
    _as_the_customers(conn)

    offered = core.readiness(conn).get("next_actions", [])

    assert any(a["tool"] == "load_rulebook_corrections" for a in offered), (
        f"nothing offered it: {[a['tool'] for a in offered]}"
    )


def test_it_stops_being_offered_once_they_are_in_force(conn):
    """Guidance that never stops appearing is guidance nobody reads — the same reason the
    first-steps path empties once it is walked."""
    _as_the_customers(conn)
    core.load_declared_corrections(conn, confirmed_by="R. Vega")

    offered = core.readiness(conn).get("next_actions", [])

    assert not any(a["tool"] == "load_rulebook_corrections" for a in offered)


def test_the_offer_does_not_prefill_who_is_confirming(conn):
    """`actions`' own rule, and §10.2's: the one thing the server cannot work out is whose
    confirmation this is. Prefilled, it would put a rule in front of every future brief under
    a name nobody gave."""
    _as_the_customers(conn)

    offer = next(a for a in core.readiness(conn).get("next_actions", [])
                 if a["tool"] == "load_rulebook_corrections")

    assert "confirmed_by" not in (offer.get("arguments") or {})
    assert any("confirm" in need for need in offer.get("needs", []))


def test_with_no_declared_corrections_nothing_is_offered(conn):
    assert not any(a["tool"] == "load_rulebook_corrections"
                   for a in core.readiness(conn).get("next_actions", []))


def test_there_is_one_route_to_expected_not_two(conn):
    """`learning.py`'s own docstring: "'One learning mechanism' is the requirement, and a
    second one shaped like the first does not satisfy it. Two implementations agree on the day
    they are written and drift by the next item."

    D108 asked for a route that does not run the counting gate, and the first version built it
    TWICE — a `from_rulebook` branch inside `learning.gate` that nothing called, and a second
    path around the gate in `load_declared_corrections` that duplicated the back half of
    `corrections.graduate` and silently dropped what that function does: the replay entry, the
    next-actions, and the sentence a person reads."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    callers = []
    for path in sorted(root.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "graduate_correction"):
                callers.append(path.name)

    assert callers == ["store.py"] or set(callers) <= {"store.py", "corrections.py"}, (
        f"more than one path writes a correction to `expected`: {sorted(set(callers))}"
    )


def test_the_declared_route_still_produces_what_graduating_produces(conn):
    """What the second implementation dropped. `corrections.graduate` records the replay entry
    and returns the offers that follow a promotion; a path around it produced a row and none
    of that, which is invisible from the row."""
    _as_the_customers(conn)

    loaded = core.load_declared_corrections(conn, confirmed_by="R. Vega")

    assert loaded["loaded"] == 10
    assert loaded.get("next_actions"), "promoting ten rules offers nothing to do next"


def test_a_declared_rule_already_noted_from_a_deck_is_still_put_in_force(conn):
    """The likeliest case for a house rule, and it failed silently.

    If the library had already recorded the same rule PROVISIONALLY from a deck, the loader
    found it, counted it as "already on file", and never graduated it — so the rule stayed
    provisional forever and the offer went quiet, because the offer used the same lookup. The
    customer is told "1 were already on file", which is true, and the rule is never applied,
    which nothing said."""
    import corrections as corrections_module

    _as_the_customers(conn)
    cid = core.ingest_campaign(conn, title="Lima launch", market="Peru", status="concluded",
                               detail="A creator push.", confirm=True)["campaign_id"]
    declared = rulebook.corrections()[0]["text"]
    corrections_module.note(conn, text=declared, campaign_id=cid,
                            provenance="Client call, 3 March")

    core.load_declared_corrections(conn, confirmed_by="R. Vega")

    entry = corrections_module.find(conn, declared)
    assert entry["status"] == "expected", (
        "a rule the customer declared stayed provisional because the library had already "
        "heard it once"
    )


def test_the_example_does_not_declare_one_name_as_both_a_market_and_a_region(conn):
    """`markets_of` strips a value that is a declared REGION — that is D71's answer, and it is
    right. The example declared `LATAM` as a market AND as Peru's region, so a campaign stored
    with `market: "LATAM"` resolved to no market at all and was invisible to every
    market-scoped check: standing corrections, expected measures, coverage.

    The example is the first thing a customer copies, so a trap in it is a trap they inherit
    before they have written a line."""
    _as_the_customers(conn)

    declared = rulebook.vocabulary("markets")
    regions = {(entry.get("region") or "").strip().lower()
               for entry in declared.values() if entry.get("region")}
    names = {name.strip().lower() for name in declared}

    assert not (regions & names), (
        f"declared as both a market and a region: {sorted(regions & names)}"
    )


def test_a_campaign_in_a_declared_market_is_visible_to_the_checklists(conn):
    """The consequence, asserted rather than reasoned about."""
    _as_the_customers(conn)
    core.load_declared_corrections(conn, confirmed_by="R. Vega")

    for market in ("Peru", "Mexico", "Indonesia"):
        cid = core.ingest_campaign(conn, title=f"{market} launch", market=market,
                                   status="concluded", detail="A store opening.",
                                   confirm=True)["campaign_id"]
        record = store.get_campaign(conn, cid)
        assert store.markets_of(record) != [None], f"{market} resolves to no market at all"


def test_the_example_is_shipped_with_the_product(conn):
    """"shipped as an example/customer file" — and nothing shipped it. Not in the PyInstaller
    spec, not copied by either build script, not named by `init` or the README or the
    rulebook the product does ship. A customer who installs the built artefact had no example
    at all, while `enums.py`'s own comment asserted they would "find it in the file they are
    copying from"."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = (root / "campaign-poc.spec").read_text(encoding="utf-8")
    assert "example-rulebook.yaml" in spec, "the build does not collect it"

    for script in ("build.sh", "build.ps1"):
        assert "example-rulebook.yaml" in (root / script).read_text(encoding="utf-8"), (
            f"{script} does not place it beside the executable"
        )


def test_the_product_tells_somebody_the_example_exists(conn, capsys, monkeypatch):
    """§10.6 again, on a file. An example nobody is told about is one nobody reads."""
    import sys

    monkeypatch.setattr(sys, "argv", ["campaign-intelligence", "init"])
    monkeypatch.setattr("config.ensure_dirs", lambda: None)
    monkeypatch.setattr("store.init_db", lambda: None)

    import main

    main.main()

    assert "example-rulebook.yaml" in capsys.readouterr().out


def test_the_shipped_rulebook_names_the_example_by_filename(conn):
    """Its header said §12.3 "will ship a worked example" — future tense, no filename, in the
    file a customer opens first."""
    text = rulebook._bundled().read_text(encoding="utf-8")

    assert "example-rulebook.yaml" in text
    assert "will ship" not in text


def test_the_model_is_not_told_a_declared_rule_was_learned(conn):
    """The sentence in front of every judgment said standing corrections are "learned from
    their own feedback, NOT WRITTEN BY THEM". For a declared correction that is false twice
    over — they wrote it, and it recurred nowhere.

    `load_declared_corrections`' own comment says a judgment citing one "must not read
    identically to one citing a rule the library inferred". The only place that survived was
    inside a provenance string, while the framing sentence asserted the inferred story for all
    of them."""
    _as_the_customers(conn)
    core.load_declared_corrections(conn, confirmed_by="R. Vega")

    note = core.prepare_evaluation(
        conn, subject_title="Lima launch", market="Peru",
        proposal_text="A six-week creator push in Lima.")["note"]

    assert "not written by them" not in note.lower()
    assert "wrote" in note.lower() or "declared" in note.lower(), (
        "the note has to say these came from the customer's own rulebook"
    )


def test_in_market_is_gone_from_the_product_everywhere(conn):
    """D37, swept. `enums` lost it and `mcp_server`'s instructions still taught it — the
    vocabulary decision implemented in two places that disagreed, with one customer's phrasing
    still in the product's own model-facing text."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    import re

    # Whole words. The first version matched `for m in markets:` — a substring scan finding
    # "in market" inside "in markets", which is the same class of bug §11.2's name check had.
    teaches = re.compile(r"(?<!\w)in[ _]market(?!\w)", re.IGNORECASE)
    for name in ("enums.py", "mcp_server.py", "core.py", "store.py"):
        text = (root / name).read_text(encoding="utf-8")
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("#") or "D37" in line:
                continue
            assert not teaches.search(line), f"{name}:{n} still teaches it: {line}"


def test_editing_a_correction_in_the_rulebook_is_reported_not_silent(conn):
    """Idempotency held only for an UNCHANGED file. Edit a correction's text and the new one
    loads while the old one stays in force — eleven rules standing for ten declared, both
    citing the rulebook, neither wrong on its face. Delete one and it stays in force forever.
    The offer empties either way, so nothing surfaces the drift.

    Retiring them automatically would be worse: a typo in a file would silently withdraw a
    rule that judgments already cite. So it is reported, with what to do."""
    _as_the_customers(conn)
    core.load_declared_corrections(conn, confirmed_by="R. Vega")

    path = rulebook.overlay_path()
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("Logo never sits on a photographic background",
                                 "Logo never sits on any photographic background"),
                    encoding="utf-8")
    rulebook.load.cache_clear()

    again = core.load_declared_corrections(conn, confirmed_by="R. Vega")

    assert again["no_longer_declared"], (
        "the rule that was edited out is still in force and nothing says so"
    )
    assert "no longer in your rulebook" in again["what_it_means"].lower()
    assert "not withdrawn automatically" in again["what_it_means"]


def test_a_correction_too_long_to_store_is_refused_before_anything_is_written(conn):
    """The loader appended "(declared in rulebook X)" AFTER the file was validated, so a
    provenance near the limit passed the loader and was refused by the WRITE — mid-loop, with
    earlier rules already committed and in force, and a reply that said nothing about them.
    Every rerun then failed identically at the same entry.

    A configuration error has to surface when the configuration is read."""
    path = rulebook.overlay_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "version: 'acme-3'\nrules: []\ncorrections:\n"
        "  - text: Never put the logo on a dark background.\n"
        f"    provenance: {'deck ' * 90}\n", encoding="utf-8")
    rulebook.load.cache_clear()

    with pytest.raises(ValueError, match="provenance"):
        rulebook.load()


def test_there_is_one_route_to_writing_when_a_rule_started_applying(conn):
    """The same guard as `graduate_correction`'s, for the table that answers "did this rule
    apply to this brief on the day it was judged". Two writers exist on purpose — a
    confirmation and a withdrawal are different events — and both live in `store`, beside the
    one function that writes the scope itself. A third door in another module is how the two
    come to disagree about what "no history" means, which is exactly the defect review found
    here: the two seeds diverged, and the divergence WAS the bug."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    callers = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, (ast.Name, ast.Attribute))
                    and getattr(node.func, "id", getattr(node.func, "attr", ""))
                    in ("_write_correction_scope", "_record_correction_scope",
                        "withdraw_correction_scope")):
                callers.append(path.name)

    assert set(callers) == {"store.py"}, (
        f"the in-force history is written from more than one module: {sorted(set(callers))}"
    )
