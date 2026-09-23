"""
§12.3's content half — the Fabletics rulebook (D129).

The format half shipped long ago: `docs/example-rulebook.yaml` is a worked example of every
shape the file can carry, using an invented agency. The CONTENT half was recorded as awaiting
input, on the grounds that the transcription "lives in the 11 Sep review document, not in this
repository". That was true of the repository and beside the point — the document was the one
that kickstarted the workstream, and a reviewer was right that the content had been available
all along. It is transcribed now, section by section, from "Ship the rulebook with the
product": four guardrails, the tag taxonomy, the influencer criteria and red flags, the
recurring partner feedback, the 360° checklist, the six-criteria scorecard, and the ten
standing corrections with their provenance.

It is a CUSTOMER file, not the product default — which is §12.3's own decision, in its own
words: "shipped as an example/customer file, not the product default." Shipping it as the
default was tried and reverted: an empty default is what makes the first-run and empty-library
behaviour what it is, and 126 tests encode that.

These tests exist for the reason the example's do: a file that ships and is never loaded rots,
and the first anybody knows is a customer whose rules do nothing.
"""
import os
import re
from pathlib import Path

import pytest

import rulebook

FABLETICS = Path(__file__).resolve().parent.parent / "docs" / "fabletics-rulebook.yaml"


@pytest.fixture(autouse=True)
def _forget_the_rulebook():
    rulebook.load.cache_clear()
    yield
    rulebook.load.cache_clear()


def _as_the_customers(conn):
    """Install it where a customer's own rulebook goes — `<data>/rulebook.yaml`."""
    path = rulebook.overlay_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FABLETICS.read_text(encoding="utf-8"), encoding="utf-8")
    rulebook.load.cache_clear()
    return path


def _as_the_old_loader_left_it(conn):
    """The rows an upgraded install actually holds.

    Written through `store` rather than by re-running an old code path: the earlier loader
    threw the declared markets away and promoted with `applies_everywhere=False`, leaving
    `expected_in = []` — "no checklist", so the rule reached no market at all. `graduate` will
    not produce that state any more (it reads the declaration itself now), and a fixture that
    monkeypatches the function under test into behaving like last year's version stops
    reproducing anything the moment that function is fixed. What has to be reproduced is the
    DATABASE.
    """
    import core
    import store

    core.load_declared_corrections(conn, confirmed_by="R. Vega")
    for row in store.corrections(conn):
        if row["status"] == "expected":
            store.graduate_correction(conn, row["id"], markets=[],
                                      confirmed_by=row["confirmed_by"] or "R. Vega",
                                      applies_everywhere=False)
    # AND NO IN-FORCE HISTORY, which is the other half of "as the old loader left it" and the
    # half a fixture written through today's `store` cannot help writing. With those rows
    # present the upgrade takes the branch that READS a history; a real upgraded install takes
    # the branch that has to reconstruct one, and that is the branch worth testing.
    conn.execute("DELETE FROM correction_scope")
    conn.commit()
    assert all(r["expected_in"] == [] and not r["applies_everywhere"]
               for r in store.corrections(conn) if r["status"] == "expected"), (
        "the fixture did not reproduce the old state")
    assert not conn.execute("SELECT COUNT(*) FROM correction_scope").fetchone()[0], (
        "the fixture left an in-force history no upgraded install has")


def test_it_ships():
    assert FABLETICS.exists(), f"no customer rulebook at {FABLETICS}"


def test_it_is_not_the_product_default(conn):
    """§12.3's decision, in its own words. A customer who installs this product and writes
    nothing gets no rules — these are one customer's, and baking them in would make every
    install of a generic product theirs."""
    assert rulebook.rules() == []
    assert rulebook.version() == "core-1.0"


def test_it_loads_and_says_whose_rules_are_in_force(conn):
    """The stamp names BOTH, so a judgment made under these rules can be told from one made
    under the product's alone, years later."""
    _as_the_customers(conn)

    assert rulebook.version() == "core-1.0+fabletics-2026.09"
    assert rulebook.rules(), "the rules loaded as nothing"


def test_every_guardrail_the_source_calls_non_negotiable_is_blocking(conn):
    """"Non-negotiables — hard guardrails, not preferences." A breach of one is a
    `guardrail_breach`, which is a different class of finding from a departure from precedent:
    one says "this breaks your own rule", the other says "this differs from Peru"."""
    _as_the_customers(conn)
    by_id = {r["id"]: r for r in rulebook.rules()}

    for rule in ("no-licensed-music", "no-ai-imagery", "no-competitor-conflict",
                 "approval-before-publishing"):
        assert rule in by_id, f"{rule} is not in the rulebook"
        assert by_id[rule]["severity"] == "blocking", (
            f"{rule} is a non-negotiable in the source and is not blocking here"
        )


def test_the_competitor_rule_keeps_the_scope_the_source_insisted_on(conn):
    """The source calls the scope out specifically: it is about "the creator's own feed, not
    only about the Fabletics asset". A roster cleared on the asset alone has not been cleared,
    which is exactly how Peru left two adidas-affiliated profiles on it."""
    _as_the_customers(conn)
    rule = next(r for r in rulebook.rules() if r["id"] == "no-competitor-conflict")

    assert "own feed" in (rule["why"] + rule["rule"]).lower()
    assert "adidas" in " ".join(rule.get("watch_for", [])).lower()


def test_the_taxonomy_is_tags_and_overlap_is_possible(conn):
    """Source guidance, carried as a presenter note: "The category labels should be used as
    tags, not rigid buckets. This will help the model separate taste, brand fit, and
    performance." One campaign can be LIKED and UNDERPERFORMED at once — strong creative,
    weak numbers — and collapsing those into one verdict is what the taxonomy prevents."""
    _as_the_customers(conn)
    tags = rulebook.vocabulary("tags")

    for tag in ("liked", "performed_well", "not_liked", "underperformed"):
        assert tag in tags, f"{tag} is not in the taxonomy"


def test_the_360_checklist_is_what_a_brief_must_carry(conn):
    """6 · "Check every channel before Approve", and the six things the source says to ALWAYS
    flag when absent. `expects` is the surface that reports a missing input WITH the words
    that would show it had arrived — without those a gap is reported forever and never
    closed."""
    _as_the_customers(conn)
    by_id = {e["id"]: e for e in rulebook.expects()}

    for owed in ("call-to-action", "store-traffic-plan", "paid-or-organic-boost",
                 "partner-collab", "ecom-link-path", "kpi-plan"):
        assert owed in by_id, f"the checklist does not ask for {owed}"
        assert by_id[owed].get("looks_like"), (
            f"{owed} has no `looks_like`, so the gap could never be closed"
        )


def test_the_scorecard_is_the_same_six_criteria(conn):
    """7 · "The same six criteria, every time." Source guidance: "The scorecard makes the
    model explain its reasoning in a structured way. It also gives leadership confidence that
    the output is not arbitrary.\""""
    _as_the_customers(conn)
    names = [c["name"] for c in rulebook.scorecard()]

    assert len(names) == 6, names
    for criterion in ("Brand alignment", "Creative quality", "Audience relevance",
                      "Influencer fit", "Compliance", "Business objective"):
        assert criterion in names, f"{criterion} is missing from the scorecard"


def test_all_ten_standing_corrections_carry_their_provenance(conn):
    """The source calls these "the most valuable content the library holds, because they are
    the things this client repeats", and says each "needs its provenance shipped with it so a
    judgment can cite where the rule came from". Without it a judgment can say only "the
    library says so", which is the unfounded confident claim this product is built against."""
    import yaml

    declared = yaml.safe_load(FABLETICS.read_text(encoding="utf-8"))["corrections"]

    assert len(declared) == 10, f"the source lists ten; this has {len(declared)}"
    for correction in declared:
        assert correction.get("provenance", "").strip(), (
            f"no provenance on: {correction['text'][:60]}"
        )

def test_the_markets_the_source_names_are_kept_as_provenance(conn):
    """The source names markets in a column headed "Where it came from", and that is where
    they stay. This test used to assert the opposite — that three corrections carried a
    `markets` list — which is the provenance-as-scope error written down as a requirement."""
    import yaml

    declared = yaml.safe_load(FABLETICS.read_text(encoding="utf-8"))["corrections"]
    provenance = " ".join(c["provenance"] for c in declared)

    for market in ("Peru", "Australia", "Colombia", "Mexico"):
        assert market in provenance, f"{market} is named by the source and is nowhere on file"
    assert not [c for c in declared if c.get("markets")], (
        "where a rule came from is not where it applies"
    )



def test_the_build_ships_it_beside_the_binary():
    """Putting it in force has to be a copy, not a download — the networks this product
    targets are the ones that block things."""
    root = Path(__file__).resolve().parent.parent
    spec = (root / "campaign-poc.spec").read_text(encoding="utf-8")

    assert "fabletics-rulebook.yaml" in spec, "the build does not collect it"


# ── put in force by `init`, which is what every installer runs ──────────────────

def _init_into(tmp_path, *, shipped: bool, existing: str | None = None):
    """Run `init` with a bundle dir that may or may not carry the customer rulebook."""
    import shutil
    import subprocess
    import sys

    app = tmp_path / "app"
    data = tmp_path / "data"
    app.mkdir(parents=True, exist_ok=True)
    if shipped:
        shutil.copyfile(FABLETICS, app / "fabletics-rulebook.yaml")
    if existing is not None:
        data.mkdir(parents=True, exist_ok=True)
        (data / "rulebook.yaml").write_text(existing, encoding="utf-8")

    script = (
        "import pathlib, sys\n"
        "sys.argv = ['campaign-intelligence', 'init']\n"
        "import config\n"
        f"config.app_dir = lambda: pathlib.Path({str(app)!r})\n"
        "import main\n"
        "main.main()\n"
    )
    out = subprocess.run([sys.executable, "-c", script],
                         cwd=Path(__file__).resolve().parent.parent,
                         capture_output=True, text=True, timeout=180,
                         env={**os.environ, "CAMPAIGN_POC_DATA": str(data),
                              "CAMPAIGN_POC_DB": str(data / "c.db")})
    assert out.returncode == 0, out.stdout + out.stderr
    return data / "rulebook.yaml", out.stdout


def test_a_fresh_install_puts_the_shipped_rulebook_in_force(tmp_path):
    """The review's P1 in one line: "fresh installations do not enforce the customer's rules —
    the exact failure the rulebook requirement was intended to solve". A file sitting beside
    the binary that nobody copies is a rulebook nobody has.

    `init` is where it happens because `init` is what all three installers run, before the
    gate — so this is one implementation rather than three."""
    written, said = _init_into(tmp_path, shipped=True)

    assert written.exists(), "a fresh install left the customer with no rules"
    assert "fabletics" in written.read_text(encoding="utf-8").lower()
    assert "Installed the rulebook shipped with this build" in said, (
        "it installed the rules and did not say so"
    )


def test_a_rulebook_already_there_is_never_overwritten(tmp_path):
    """Yours is yours. An installer that replaces the rules somebody wrote is an installer
    that loses them silently, and an upgrade is when it would happen."""
    mine = "# my own rules\nversion: mine\nrules: []\n"

    written, said = _init_into(tmp_path, shipped=True, existing=mine)

    assert written.read_text(encoding="utf-8") == mine
    assert "Installed the rulebook shipped" not in said


def test_a_build_that_ships_no_customer_rulebook_stays_generic(tmp_path):
    """Presence of the shipped file is the switch. A build made without it installs nothing and
    the product stays generic — which is §12.1's decision, and still right for anybody who is
    not this customer."""
    written, said = _init_into(tmp_path, shipped=False)

    assert not written.exists()
    assert "Installed the rulebook shipped" not in said


# ── the rules actually reaching a judgment, which is the only point of shipping them ──

def test_the_guardrails_reach_an_evaluation(conn):
    """A rulebook that loads and is never applied is the state D108 describes, and it is
    invisible from the file itself. This is the whole point of shipping the content: a brief is
    judged against these rules, outside the similarity path, every time."""
    import core

    _as_the_customers(conn)

    prepared = core.prepare_evaluation(
        conn, subject_title="Bogota store opening", market="Peru",
        proposal_text="A six-week push with creators in Lima, using trending audio.")

    # The `rulebook` block the package actually carries, not a repr match with an `or` in it.
    book = prepared["rulebook"]

    assert book["rules_applied"] == 13, book
    assert book["scorecard"] == 6, book
    assert book["overlay"] == "fabletics-2026.09", book
    assert book["version"] == "core-1.0+fabletics-2026.09", (
        "the judgment does not say which rulebook it was made under — the stamp is the only "
        "thing that tells two rulebooks apart afterwards"
    )
    assert book["basis"] == "computed"


def test_the_standing_corrections_reach_a_judgment(conn):
    """Ten corrections with provenance, declared rather than inferred — so they skip the
    counting gate and not the person. Stored-and-never-applied is what D108 was reopened for."""
    import core

    _as_the_customers(conn)
    core.load_declared_corrections(conn, confirmed_by="R. Vega")

    prepared = core.prepare_evaluation(
        conn, subject_title="Bogota store opening", market="Peru",
        proposal_text="A six-week push with creators in Lima.")

    standing = prepared["standing_corrections"]["standing"]

    assert standing, (
        f"none of them reached the judgment: "
        f"{prepared['standing_corrections']['what_it_means']}"
    )
    # All ten. `>= 7` was the original assertion and it passed while three corrections were
    # dead in every market — a test that could not fail in the way that mattered. It then said
    # 8, which encoded the provenance-as-scope error the count was a symptom of.
    assert len(standing) == 10, (
        f"{len(standing)} of ten applied in Peru: {[c['text'][:40] for c in standing]}"
    )


@pytest.mark.parametrize("market,phrase", [
    ("Peru", "Seed a single colourway"),
    ("Australia", "Seed a single colourway"),
    ("Colombia", "Brand arrival before celebrity collection"),
    ("Mexico", "State duration and placement for any OOH"),
])
def test_a_correction_scoped_to_a_market_reaches_that_market(conn, market, phrase):
    """Three of the ten name the markets they came from, and the loader threw the list away:
    `graduate` was given the markets the library had SEEN the rule in, which for a declared
    rule is none, so `expected_in` stayed empty and `standing_for` excluded them from
    everywhere.

    A declared correction WITH markets was therefore strictly worse than one without — it
    applied nowhere, while a market-less one applied everywhere. That is the exact inversion
    D108's own fix shipped, recurring for scoped rules."""
    import core

    _as_the_customers(conn)
    core.load_declared_corrections(conn, confirmed_by="R. Vega")

    prepared = core.prepare_evaluation(conn, subject_title="X", market=market,
                                       proposal_text="A push.")
    texts = " ".join(c["text"] for c in prepared["standing_corrections"]["standing"])

    assert phrase in texts, (
        f"the rule scoped to {market} does not reach a {market} brief"
    )

def test_provenance_is_not_read_as_scope(conn):
    """The source's table has a column headed "Where it came from", and three of its cells name
    a market: "Peru (30 influencers, praised); instructed independently in Australia",
    "Colombia slide 6", "Mexico slides 9 and 10".

    An earlier version of this file read those as SCOPE and gave the three rules a `markets`
    list — and this test asserted the consequence, that a single-colourway brief in the UAE saw
    no such rule. Nothing in the review says that. Read as scope, a provenance column
    SUPPRESSES a generally worded client standard everywhere it was not first written down,
    which is a worse failure than applying it too widely and one nobody would ever see.

    `markets` still works and is carried correctly when a client sets it — that is their call.
    It is not a call to be made by reading a provenance column."""
    import core
    import yaml

    declared = yaml.safe_load(FABLETICS.read_text(encoding="utf-8"))["corrections"]
    assert not [c for c in declared if c.get("markets")], (
        "a correction is scoped to a market the source only names as where it came from"
    )

    _as_the_customers(conn)
    core.load_declared_corrections(conn, confirmed_by="R. Vega")

    for market in ("Peru", "UAE", "Colombia", "Mexico", "Australia"):
        prepared = core.prepare_evaluation(conn, subject_title="X", market=market,
                                           proposal_text="A push.")
        texts = " ".join(c["text"] for c in prepared["standing_corrections"]["standing"])
        assert "Seed a single colourway" in texts, (
            f"the single-colourway rule does not reach a brief in {market}"
        )
        assert "State duration and placement" in texts, (
            f"the OOH rule does not reach a brief in {market}"
        )



def test_the_shipped_rulebook_is_found_where_pyinstaller_actually_puts_it(tmp_path,
                                                                          monkeypatch):
    """PyInstaller 6 puts every `datas` entry under `_internal/` whatever destination the spec
    names — `rulebook._bundled` documents that and handles it for the product's own rulebook.
    The customer one looked beside the executable alone, so on a frozen build `init` found
    nothing, installed no rules, and reported success. A CI run rediscovered it.

    Both places, executable-side first so a build script that puts it somewhere an
    administrator can see and replace wins."""
    import sys

    import config
    import rulebook

    app = tmp_path / "app"
    internal = app / "_internal"
    internal.mkdir(parents=True)
    monkeypatch.setattr(config, "app_dir", lambda: app)

    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    assert rulebook.shipped_customer_rulebook() is None, "a generic build must stay generic"

    # Collected into _internal, which is what a real frozen build looks like.
    (internal / "fabletics-rulebook.yaml").write_text("version: x\n", encoding="utf-8")
    monkeypatch.setattr(sys, "_MEIPASS", str(internal), raising=False)
    assert rulebook.shipped_customer_rulebook() == internal / "fabletics-rulebook.yaml"

    # And a copy beside the executable wins, because that one can be seen and replaced.
    (app / "fabletics-rulebook.yaml").write_text("version: y\n", encoding="utf-8")
    assert rulebook.shipped_customer_rulebook() == app / "fabletics-rulebook.yaml"


def test_every_rationale_says_where_it_came_from():
    """The file once claimed "nothing here is invented" while carrying thirteen `why` fields
    that were the author's own — "a copyright strike takes the asset down mid-flight", "the
    audience reacts to it". Review checked each against the source document and found all
    thirteen absent from it.

    They are not decoration: a `why` reaches every evaluation as the CLIENT'S stated reasoning,
    which makes an invented one the confident unfounded claim this product exists to refuse,
    inside the file carrying the client's own rules. Every one must now name its source, and
    where the review gives a reason it is quoted rather than paraphrased."""
    import yaml

    book = yaml.safe_load(FABLETICS.read_text(encoding="utf-8"))

    # rules AND expects. The first version of this test checked `book["rules"]` alone, and six
    # invented rationales sat in `expects` untouched while it passed — the same blind spot it
    # was written to close, one key over. `missing.py` hands an expectation's `why` back as
    # `why_it_matters` and labels it the customer's own rule, so an invented one there is no
    # more inert than an invented one in a rule.
    for entry in book["rules"] + book["expects"]:
        why = entry["why"]
        assert "September 2026 product review" in why, (
            f"{entry['id']}'s rationale does not say where it came from: {why[:70]}"
        )
    # And where the source is silent, the file says so rather than filling the gap.
    silent = [r for r in book["rules"] if "none is invented here" in r["why"]]
    assert silent, (
        "every rule now claims a stated rationale; the source does not give one for all of "
        "them, and inventing the difference is the defect this test exists for"
    )

    # Naming the review is not enough on its own: an invented sentence appended to a true
    # provenance clause still reaches an evaluation as the client's reasoning, and the first
    # version of this test passed on exactly that. So outside quotation marks a `why` may say
    # where the rule came from and nothing else — no causal claim about what the rule is FOR.
    #
    # Crude, and deliberately so, in the way `negated()` is: it refuses a shape rather than
    # judging a sentence, and the cost of a false refusal is rewording a line in a YAML file.
    causal = ("cannot", "because", "so that", "reads as", "is how", "would be", "is where",
              "delivers", "is a hope", "expensive", "quietly lost", "takes the asset")
    for entry in book["rules"] + book["expects"]:
        outside = re.sub(r'"[^"]*"', " ", entry["why"]).lower()
        found = [c for c in causal if c in outside]
        assert not found, (
            f"{entry['id']}'s rationale argues {found} outside anything the review says — "
            f"that reaches an evaluation as the CLIENT'S reasoning: {entry['why'][:90]}"
        )


def test_a_symlinked_rulebook_is_left_alone(tmp_path):
    """`exists()` is False for a DANGLING symlink, so `init` read one as "no rulebook here" and
    `shutil.copyfile` — which follows symlinks — wrote the customer rulebook into the link's
    target. Reproduced: 22KB appeared at a path outside the data directory entirely, and the
    install reported success.

    The product already decided what a dangling symlink means. `rulebook._read` refuses to
    load one and says so, precisely because "the target moved" and "there is no rulebook" are
    different situations and silently treating the first as the second loses somebody's rules.
    `init` must not overrule that from the other side."""
    import shutil
    import subprocess
    import sys

    app, data, elsewhere = tmp_path / "app", tmp_path / "data", tmp_path / "elsewhere"
    for p in (app, data, elsewhere):
        p.mkdir(parents=True)
    shutil.copyfile(FABLETICS, app / "fabletics-rulebook.yaml")
    (data / "rulebook.yaml").symlink_to(elsewhere / "gone.yaml")

    script = ("import pathlib, sys\n"
              "sys.argv = ['campaign-intelligence', 'init']\n"
              "import config\n"
              f"config.app_dir = lambda: pathlib.Path({str(app)!r})\n"
              "import main\nmain.main()\n")
    out = subprocess.run([sys.executable, "-c", script],
                         cwd=Path(__file__).resolve().parent.parent,
                         capture_output=True, text=True, timeout=180,
                         env={**os.environ, "CAMPAIGN_POC_DATA": str(data),
                              "CAMPAIGN_POC_DB": str(data / "c.db")})

    assert not (elsewhere / "gone.yaml").exists(), (
        "init followed a dangling symlink and wrote the rulebook outside the data directory"
    )
    assert "Installed the rulebook shipped" not in out.stdout, (
        "it reported installing rules over a link it should not have touched"
    )


def test_an_install_that_ran_the_old_loader_is_repaired_on_upgrade(conn):
    """A fix that only works on a fresh database is not a fix. The old loader threw the
    declared markets away and promoted with `applies_everywhere=False`, leaving
    `expected_in = []` — which means "no checklist", so those rules reached no market at all.

    On the next start the loader saw them already on file and reported "already": 0 loaded, 10
    already, still broken, and nothing said so. Reproduced before fixing. The loader has to
    RECONCILE what is on file against what the rulebook now declares, not merely skip it."""
    import core
    import corrections
    import store

    _as_the_customers(conn)

    # Exactly what the previous implementation persisted.
    _as_the_old_loader_left_it(conn)

    broken = [r for r in store.corrections(conn) if "colourway" in r["text"]][0]
    assert broken["expected_in"] == [], "the fixture did not reproduce the old state"

    out = core.load_declared_corrections(conn, confirmed_by="R. Vega")

    prepared = core.prepare_evaluation(conn, subject_title="X", market="Peru",
                                       proposal_text="A push.")
    texts = " ".join(c["text"] for c in prepared["standing_corrections"]["standing"])
    assert "Seed a single colourway" in texts, (
        f"an upgraded install is still broken; the loader said "
        f"{out['loaded']} loaded, {out['already']} already"
    )
    assert out.get("repaired"), "it repaired rows and did not say so"


def _declared(conn, text, markets):
    """A rulebook rule promoted exactly as the loader promotes one."""
    import corrections

    cid = corrections.note(conn, text=text, campaign_id=None,
                           provenance="the house rules (declared in rulebook fab-1.0)",
                           )["correction_id"]
    corrections.graduate(conn, cid, confirmed_by="R. Vega", from_rulebook="fab-1.0",
                         everywhere=not markets, markets=markets or None)
    return cid


def test_a_rule_that_becomes_global_stops_claiming_the_market_it_left(conn):
    """The rulebook said Peru, then said everywhere. Keeping the old list because the new one
    is empty leaves the row saying two different things — in force in every market, seen in
    Peru — and `standing_for` prints the stale half back to the customer as `seen_in`. The
    declaration is the authority on a declared rule; there is nothing else it can be."""
    import corrections

    cid = _declared(conn, "Never promise a delivery date in creative.", ["Peru"])
    assert corrections.describe(conn, cid)["expected_in"] == ["Peru"]

    assert corrections.reconcile_declared(conn, cid, markets=[], everywhere=True,
                                          confirmed_by="R. Vega")
    entry = corrections.describe(conn, cid)
    assert entry["applies_everywhere"]
    assert entry["expected_in"] == [], (
        f"a rule the rulebook now declares everywhere still reports {entry['expected_in']}")

    # And what the customer reads back names the two facts apart: this rule is in force in
    # Japan (it is a house rule), and the library has never seen it in a campaign anywhere.
    standing = corrections.standing_for(conn, campaign_id="", markets=["Japan"])["standing"]
    assert [c["expected_in"] for c in standing] == [[]]
    assert [c["seen_in"] for c in standing] == [[]]


def test_a_rule_that_stops_being_global_takes_the_new_scope(conn):
    """The other direction, which the same line has to get right."""
    import corrections

    cid = _declared(conn, "Never promise a delivery date in creative.", [])
    assert corrections.reconcile_declared(conn, cid, markets=["Peru"], everywhere=False,
                                          confirmed_by="R. Vega")
    entry = corrections.describe(conn, cid)
    assert not entry["applies_everywhere"]
    assert entry["expected_in"] == ["Peru"]


def test_reconciling_nothing_writes_nothing(conn):
    """It runs on every start. A repair that reports itself every time is one nobody reads."""
    import corrections

    cid = _declared(conn, "Never promise a delivery date in creative.", ["Peru"])
    assert corrections.reconcile_declared(conn, cid, markets=["Peru"], everywhere=False,
                                          confirmed_by="R. Vega") is None


def test_a_repair_records_who_made_the_call(conn):
    """§11.1: the account beside the name, on the write that puts a rule in force in markets
    it was in force in nowhere. `graduate` records it and this write is the same write — going
    around it is how the one surface built to review human decisions comes to miss one."""
    import corrections
    import store

    cid = _declared(conn, "Never promise a delivery date in creative.", ["Peru"])
    before = conn.execute("SELECT COUNT(*) FROM authorship WHERE subject_key = ?",
                          (cid,)).fetchone()[0]

    corrections.reconcile_declared(conn, cid, markets=[], everywhere=True,
                                   confirmed_by="A. Okafor")

    rows = conn.execute("SELECT COUNT(*) FROM authorship WHERE subject_key = ?",
                        (cid,)).fetchone()[0]
    assert rows == before + 1, "the repair changed what a rule applies to and said who nowhere"
    who = store.authorship_for(conn, "correction", cid)
    assert who["on_behalf_of"]["name"] == "A. Okafor"
    # And the rule's own `confirmed_by` is untouched: whoever ran the upgrade did not confirm
    # this rule, and overwriting that name would rewrite the record of somebody's decision as
    # a side effect of a bug fix. Mutation found this assertion missing.
    assert corrections.describe(conn, cid)["confirmed_by"] == "R. Vega"
    # What they SAID travels with the name, which is where §11.2 keeps it.
    assert "rulebook" in who["on_behalf_of"]["why"].lower()
    assert "Peru" in who["on_behalf_of"]["why"], "it did not say what the row used to be"


def test_a_repaired_rule_offers_the_replay(conn):
    """The offer is the whole point of the repair: these rules were in force nowhere, they are
    in force everywhere now, and the judgments written before them are exactly what a person
    has to go and look at. A rule loaded fresh offers it; a rule repaired offered nothing."""
    import core
    import corrections

    _as_the_customers(conn)

    _as_the_old_loader_left_it(conn)

    out = core.load_declared_corrections(conn, confirmed_by="R. Vega")
    assert out.get("repaired")
    assert out["loaded"] == 0, "the fixture was meant to repair, not load"
    assert [a["tool"] for a in out.get("next_actions") or []] == ["replay_rules"], (
        "ten rules went from in force nowhere to in force everywhere and nothing offered "
        "the report that says which saved judgments they reach")



def test_the_half_repaired_state_the_LAST_fix_left_is_repaired_too(conn):
    """An upgrade path with two hops in it, and the middle one is a state this product
    actually shipped: the previous fix wrote `markets or on_file`, so a rule the rulebook had
    scoped to Peru and now declares everywhere came out of it applying in every market and
    still reporting `seen_in: ["Peru"]`. Both halves are on file and they disagree.

    The guard has to notice, which means comparing the market lists even when the new one is
    empty — `not markets or ...` reads that row as already reconciled and leaves it saying two
    things forever. Mutation found this: nothing else in the suite could tell the two guards
    apart, because every other case has either the flag or the list differing."""
    import corrections
    import store

    cid = _declared(conn, "Never promise a delivery date in creative.", ["Peru"])
    store.graduate_correction(conn, cid, markets=["Peru"], confirmed_by="R. Vega",
                              applies_everywhere=True)          # what the last fix left
    assert corrections.describe(conn, cid)["expected_in"] == ["Peru"]

    fixed = corrections.reconcile_declared(conn, cid, markets=[], everywhere=True,
                                           confirmed_by="R. Vega")
    assert fixed, "the row says it applies everywhere AND only in Peru, and nothing repaired it"
    assert corrections.describe(conn, cid)["expected_in"] == []


def test_the_repair_offers_a_replay_that_names_every_market(conn):
    """The offer is what a person reads before saying yes. "Briefs in these markets" about a
    rule that has no markets — because it applies to all of them — names nothing."""
    import corrections

    cid = _declared(conn, "Never promise a delivery date in creative.", ["Peru"])
    fixed = corrections.reconcile_declared(conn, cid, markets=[], everywhere=True,
                                           confirmed_by="R. Vega")
    why = fixed["next_actions"][0]["why"]
    assert "every market" in why, why
    assert "these markets" not in why


def test_a_house_rule_graduates_saying_it_applies_everywhere(conn):
    """Same sentence, on the path a fresh install takes. `', '.join([])` is the empty string,
    so both the offer and the confirmation read as though a market had gone missing."""
    import corrections

    cid = corrections.note(conn, text="No price promises in creative.", campaign_id=None,
                           provenance="the house rules (declared in rulebook fab-1.0)",
                           )["correction_id"]
    out = corrections.graduate(conn, cid, confirmed_by="R. Vega", from_rulebook="fab-1.0",
                               everywhere=True, markets=None)

    assert "Briefs in every market are now judged against this" in out["what_it_means"]
    assert "every market" in out["next_actions"][0]["why"]


def test_a_repair_refuses_to_switch_a_rule_off_by_accident(conn):
    """"No markets and not everywhere" is `expected_in = []` with the flag clear, which is the
    exact state D108 was reopened for: standing, and in force nowhere. The loader cannot ask
    for it, so a caller that does has made a mistake — and carrying it out quietly is how the
    rules went dead the first time."""
    import corrections
    import pytest as _pytest

    cid = _declared(conn, "Never promise a delivery date in creative.", ["Peru"])
    with _pytest.raises(ValueError, match="applies EVERYWHERE"):
        corrections.reconcile_declared(conn, cid, markets=[], everywhere=False,
                                       confirmed_by="R. Vega")
    assert corrections.describe(conn, cid)["expected_in"] == ["Peru"], "it wrote anyway"


def test_a_rule_somebody_set_aside_does_not_break_the_next_load(conn):
    """Reproduced before fixing: set aside ONE declared rule through the tool built for it,
    and the next load of the rulebook raised `ValueError` out of `graduate`'s gate — so the
    other nine were never reconciled, and a customer who had used a documented tool could
    never load their own file again.

    Setting one aside is a person's decision and the file is not allowed to overturn it
    silently, which is the same rule the other direction already follows: deleting a line does
    not withdraw a standing rule, it is reported. So this is reported too."""
    import core
    import corrections

    _as_the_customers(conn)
    core.load_declared_corrections(conn, confirmed_by="R. Vega")
    standing = [r for r in corrections.all_of_them(conn) if r["status"] == "expected"]
    corrections.set_aside(conn, standing[0]["id"], why="not ours any more")

    out = core.load_declared_corrections(conn, confirmed_by="R. Vega")

    assert out["already"] == 9, f"the other nine did not reconcile: {out}"
    assert out.get("set_aside") == [standing[0]["text"]], (
        f"nothing said the file and the library disagree about this rule: {out}")
    assert "set aside" in out["what_it_means"]
    assert corrections.describe(conn, standing[0]["id"])["status"] == "ignored", (
        "the file overturned a person's decision without asking"
    )
    assert "reopen_correction" in [a["tool"] for a in out.get("next_actions") or []]


def test_two_declared_rules_that_are_one_rule_on_file_do_not_ping_pong(conn):
    """Somebody folded two wordings into one rule (`same_rule`, which is §8.2's own answer),
    and the rulebook still declares both — with different scopes. `find` follows the merge, so
    both declarations land on the SAME row and each start reconciled it back and forth:
    `repaired` named one id twice, the sentence claimed a repair on every restart, and after
    the authorship fix the §11 history for that rule grew by two rows per start, forever.

    The loader cannot honour two scopes for one row. It honours the first and SAYS so."""
    import core
    import corrections
    import rulebook

    a = "Seeding boxes carry one colourway."
    b = "Only one colourway per seeding box."
    path = rulebook.overlay_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("version: fab-1.0\ndescribes: two wordings of one rule\ncorrections:\n"
                    f"  - text: {a}\n    provenance: Client call, 3 March\n"
                    "    markets: ['Peru']\n"
                    f"  - text: {b}\n    provenance: Client call, 9 May\n", encoding="utf-8")
    rulebook.load.cache_clear()

    cid = core.ingest_campaign(conn, title="Lima", market="Peru", status="concluded",
                               detail="A launch.", confirm=True)["campaign_id"]
    first = corrections.note(conn, text=a, campaign_id=cid,
                             provenance="Deck")["correction_id"]
    second = corrections.note(conn, text=b, campaign_id=cid,
                              provenance="Deck")["correction_id"]
    corrections.resolve(conn, first, decision="same_rule", same_as=second)

    core.load_declared_corrections(conn, confirmed_by="R. Vega")
    rows = lambda: conn.execute(                                       # noqa: E731
        "SELECT COUNT(*) FROM authorship WHERE subject_kind = 'correction'").fetchone()[0]
    settled = rows()

    out = core.load_declared_corrections(conn, confirmed_by="R. Vega")

    assert not out.get("repaired"), f"it repaired a rule it had just repaired: {out}"
    assert rows() == settled, "every start writes another authorship row for the same rule"
    assert out.get("same_rule_on_file"), (
        f"two declarations are one rule on file and nothing said so: {out}")
    assert "one rule" in out["what_it_means"]
    # Somebody DID answer `same_rule` here, so saying so is a fact rather than a guess.
    assert [m["basis"] for m in out["same_rule_on_file"]] == ["judged"]
    assert "somebody answered `same_rule` about them" in out["what_it_means"]


def test_a_rule_the_library_learned_becomes_a_rule_the_rulebook_declares(conn):
    """A rule confirmed from three decks, whose text the customer later writes into their
    rulebook. Widening it to everywhere is right — they wrote it down, which is the whole of
    what `applies_everywhere` means — but nothing recorded that it now comes from the file.

    Two consequences, both reproduced: the drift report keys on rulebook provenance, so
    deleting the line afterwards withdrew nothing and reported nothing; and the loader's own
    sentence claimed it had put the row where the rulebook says, about a row whose provenance
    still named three decks."""
    import core
    import corrections
    import rulebook

    text = "Captions name the product in the first line."
    for market in ("Peru", "Mexico", "Colombia"):
        cid = core.ingest_campaign(conn, title=f"{market} launch", market=market,
                                   status="concluded", detail="A launch.",
                                   confirm=True)["campaign_id"]
        corrections.note(conn, text=text, campaign_id=cid, provenance=f"{market} slide 9")
    learned = corrections.find(conn, text)["correction_id"]
    corrections.graduate(conn, learned, confirmed_by="A. Okafor")

    path = rulebook.overlay_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("version: fab-1.0\ndescribes: declared later\ncorrections:\n"
                    f"  - text: {text}\n    provenance: Client email, 19 May\n",
                    encoding="utf-8")
    rulebook.load.cache_clear()
    core.load_declared_corrections(conn, confirmed_by="R. Vega")

    assert "rulebook" in corrections._provenance_of(conn, learned), (
        "the rulebook widened this rule to every market and cited three decks for it")
    # The decks are not erased by the declaration: it applies everywhere AND the library
    # watched it recur in three markets, which are two facts and both are true.
    standing = corrections.standing_for(conn, campaign_id="", markets=["Japan"])["standing"]
    mine = next(c for c in standing if c["text"] == text)
    assert sorted(mine["seen_in"]) == ["Colombia", "Mexico", "Peru"], mine["seen_in"]
    assert mine["expected_in"] == []
    assert corrections.describe(conn, learned)["confirmed_by"] == "A. Okafor", (
        "the upgrade overwrote the name of the person who actually confirmed it")

    path.write_text("version: fab-1.1\ndescribes: dropped\ncorrections: []\n",
                    encoding="utf-8")
    rulebook.load.cache_clear()
    gone = core.load_declared_corrections(conn, confirmed_by="R. Vega")

    assert gone.get("no_longer_declared") == [text], (
        f"the customer deleted the line and nothing reported the rule still standing: {gone}")


def test_emptying_the_rulebook_reports_the_rules_it_leaves_standing(conn):
    """The largest edit a customer can make to this file, and the one nothing reported.
    Deleting one line of ten was reported as drift; deleting all ten returned early — "your
    rulebook declares no standing corrections" — while all ten went on being applied to every
    brief. Nothing is withdrawn automatically either way: saved judgments cite these."""
    import core
    import corrections
    import rulebook

    _as_the_customers(conn)
    core.load_declared_corrections(conn, confirmed_by="R. Vega")
    standing = len([r for r in corrections.all_of_them(conn) if r["status"] == "expected"])
    assert standing == 10

    rulebook.overlay_path().write_text(
        "version: fabletics-2026.10\ndescribes: emptied\ncorrections: []\n", encoding="utf-8")
    rulebook.load.cache_clear()

    out = core.load_declared_corrections(conn, confirmed_by="R. Vega")

    assert len(out.get("no_longer_declared") or []) == 10, (
        f"ten rules are still in force and the loader said: {out['what_it_means']}")
    assert "no longer declared" in out["what_it_means"]
    assert len([r for r in corrections.all_of_them(conn)
                if r["status"] == "expected"]) == 10, "a file edit withdrew a standing rule"


def test_a_house_rule_heard_from_a_deck_first_does_not_come_out_scoped_to_that_deck(conn):
    """The likeliest case for a house rule, on a FRESH database: the library heard it in Peru
    before the customer's file declared it. `graduate` fell back to `gate["seen_in"]` when the
    declaration named no markets, so the first load produced `applies_everywhere = 1` with
    `expected_in = ["Peru"]` — in force everywhere, reported as seen in Peru, which is the
    state the upgrade repair exists to clean up, manufactured one start earlier by the load.

    The fallback is right for a rule the library INFERRED and wrong for one the customer
    declared, and `everywhere` is exactly the difference."""
    import core
    import corrections

    _as_the_customers(conn)
    import yaml
    text = yaml.safe_load(FABLETICS.read_text(encoding="utf-8"))["corrections"][0]["text"]
    peru = core.ingest_campaign(conn, title="Lima flagship", market="Peru",
                                status="concluded", detail="A launch.",
                                confirm=True)["campaign_id"]
    cid = corrections.note(conn, text=text, campaign_id=peru,
                           provenance="Peru slide 9")["correction_id"]

    core.load_declared_corrections(conn, confirmed_by="R. Vega")

    row = corrections.describe(conn, cid)
    assert row["applies_everywhere"]
    assert row["expected_in"] == [], (
        f"a rule in force in every market says it graduated in {row['expected_in']}")
    # And the same branch is the one that never marked the row as coming from a rulebook, so
    # the drift report could not see it: declared by behaviour, learned by provenance.
    assert "rulebook" in corrections._provenance_of(conn, cid)


def test_the_preview_before_confirming_a_house_rule_counts_every_market(conn):
    """`if_confirmed` is the sentence somebody reads at the moment they decide, and it read
    the market list alone: for a rule the customer declares everywhere — which has no market
    list, because it has no market — it said confirming would affect the briefs of "no
    market", and 0 saved judgments. Then the report it opens afterwards lists them all.

    That is the disagreement this whole round is about, pointing the other way."""
    import core
    import corrections

    _as_the_customers(conn)
    import yaml
    text = yaml.safe_load(FABLETICS.read_text(encoding="utf-8"))["corrections"][0]["text"]
    for market in ("Peru", "Japan"):
        cid = core.ingest_campaign(conn, title=f"{market} v1", market=market,
                                   status="concluded", detail="A launch.",
                                   confirm=True)["campaign_id"]
        core.save_evaluation(conn, subject_title=f"{market} v1", campaign_id=cid,
                             verdict="approve", summary="Looks sound.", findings=[])
    heard = corrections.note(conn, text=text, campaign_id=cid,
                             provenance="Japan slide 9")["correction_id"]

    preview = corrections.graduation(conn, heard)["if_confirmed"]

    assert preview["judgments_affected"] == 2, (
        f"the rulebook declares this everywhere and the preview counted "
        f"{preview['judgments_affected']} of 2: {preview['what_it_means']}")
    assert "every market" in preview["what_it_means"], preview["what_it_means"]
    # And the payload says the same thing the sentence does: nothing is scoped, and where it
    # has been heard is a separate fact under its own name.
    assert preview["applies_everywhere"] and preview["markets"] == []
    assert preview["seen_in"] == ["Japan"]


def test_the_sentence_a_person_reads_counts_the_repairs(conn):
    """A key a reader has to go looking for is not a report. `repaired` rows are not "already
    on file" — the row CHANGED, and the count belongs in the sentence beside the loaded one.
    Its own test, because it was asserted inside a test named for the replay offer."""
    import core
    import corrections

    _as_the_customers(conn)
    _as_the_old_loader_left_it(conn)

    out = core.load_declared_corrections(conn, confirmed_by="R. Vega")

    assert "10 rule(s) already on file now apply where your rulebook says they do" in (
        out["what_it_means"]), out["what_it_means"]
    # And no claim about WHY they differed: "an earlier version of this loader left them" is
    # true of an upgrade and false of a rule this library learned and the file has just
    # widened, and the loader cannot tell those apart from here.
    assert "earlier version" not in out["what_it_means"]


def test_the_model_is_not_told_a_house_rule_applies_in_no_market(conn):
    """`correction_status` is model-facing, and the gate's answer for a rule already in force
    read "already expected of briefs in no market" — about a rule in force in all of them.
    The fourth copy of one sentence, in the module that owns the market question. A tool that
    hands the model a false fact about what a brief is checked against is worse than one that
    says nothing, because the model will repeat it."""
    import core
    import corrections

    _as_the_customers(conn)
    core.load_declared_corrections(conn, confirmed_by="R. Vega")
    standing = [r for r in corrections.all_of_them(conn) if r["status"] == "expected"][0]

    said = corrections.graduation(conn, standing["id"])["what_it_means"]

    assert "in every market" in said, said
    assert "no market" not in said


def test_a_quiet_house_rule_is_not_reported_as_quiet_in_no_market(conn):
    """The retirement question, asked about a rule that applies everywhere. `expected_in` is
    empty for one, so the sentence fell back to "in its markets" — vague where the answer is
    known, in the one place this loop asks somebody to stop applying a rule."""
    import core
    import corrections

    _as_the_customers(conn)
    core.load_declared_corrections(conn, confirmed_by="R. Vega")
    rule = [r for r in corrections.all_of_them(conn) if r["status"] == "expected"][0]

    # Enough campaigns recording feedback of their OWN, after this rule was last heard, for
    # the retirement question to be worth asking at all. The rule is not repeated in any of
    # them, which is what "gone quiet" means.
    # Recording feedback is what asks the question — the product asks it once, where somebody
    # is already looking, rather than waiting to be called.
    asked = []
    for n in range(corrections.RETIREMENT_AFTER + 1):
        cid = core.ingest_campaign(conn, title=f"Later {n}", market="Peru",
                                   status="concluded", detail="A launch.",
                                   confirm=True)["campaign_id"]
        asked += corrections.note(conn, text=f"Something else entirely, number {n}.",
                                  campaign_id=cid, provenance="Deck").get("gone_quiet") or []

    quiet = [q for q in asked if q["correction_id"] == rule["id"]]
    assert quiet, "the retirement question was never asked about a standing house rule"
    assert "in every market" in quiet[0]["what_it_means"], quiet[0]["what_it_means"]


def test_the_preview_does_not_call_a_declared_scope_a_sighting(conn):
    """The same conflation this round removed, in a key this round added. For a rule the file
    SCOPES to markets, what would be in force is the declaration and where it was heard is an
    observation, and the preview was reporting the first under the name of the second."""
    import corrections
    import core
    import replay
    import rulebook

    text = "Every asset naming a price carries the currency."
    path = rulebook.overlay_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("version: fab-1.0\ndescribes: a scoped rule\ncorrections:\n"
                    f"  - text: {text}\n    provenance: Client email, 19 May\n"
                    "    markets: ['Peru']\n", encoding="utf-8")
    rulebook.load.cache_clear()

    japan = core.ingest_campaign(conn, title="Japan v1", market="Japan", status="concluded",
                                 detail="A launch.", confirm=True)["campaign_id"]
    cid = corrections.note(conn, text=text, campaign_id=japan,
                           provenance="Japan slide 4")["correction_id"]

    preview = replay.if_graduated(conn, correction_id=cid, markets=["Peru"])

    assert preview["markets"] == ["Peru"], "what would be in force is what the file declares"
    assert preview["seen_in"] == ["Japan"], (
        f"the file's declaration was reported as where the rule was seen: "
        f"{preview['seen_in']}")


def test_confirming_a_declared_rule_by_hand_uses_the_scope_it_declares(conn):
    """`correction_status` previews what the rulebook declares, and `graduate_correction` is
    the tool it offers next. The tool ignored the declaration and fell back to where the rule
    had been overheard — so accepting the offer produced a Peru-only rule from a global
    declaration, and a Peru-only rule from a Mexico one. The preview and the act it leads to
    have to read the same source."""
    import corrections
    import core
    import rulebook

    path = rulebook.overlay_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("version: fab-1.0\ndescribes: two declarations\ncorrections:\n"
                    "  - text: A global house rule about dates.\n    provenance: Client call\n"
                    "  - text: A Mexico rule about currency.\n    provenance: Client email\n"
                    "    markets: ['Mexico']\n", encoding="utf-8")
    rulebook.load.cache_clear()

    peru = core.ingest_campaign(conn, title="Lima", market="Peru", status="concluded",
                                detail="A launch.", confirm=True)["campaign_id"]
    rules = {}
    for text in ("A global house rule about dates.", "A Mexico rule about currency."):
        cid = corrections.note(conn, text=text, campaign_id=peru,
                               provenance="Peru slide 2")["correction_id"]
        corrections.graduate(conn, cid, confirmed_by="R. Vega")
        rules[text] = corrections.describe(conn, cid)

    globally = rules["A global house rule about dates."]
    assert globally["applies_everywhere"], "a global declaration became a scoped rule"
    assert globally["expected_in"] == []

    mexico = rules["A Mexico rule about currency."]
    assert mexico["expected_in"] == ["Mexico"], (
        f"a Mexico declaration became {mexico['expected_in']}")
    assert not mexico["applies_everywhere"]


def test_a_declared_alias_of_a_merged_rule_is_not_reported_as_deleted(conn):
    """Somebody answered `same_rule` about two wordings, so the library holds one rule under
    the other's words. The file declares the alias; `find` follows the merge and puts the
    right rule in force — and the drift sweep, comparing literal text, then said in the same
    response that the rule it had just loaded is NO LONGER IN THE RULEBOOK and should be
    retired. One response, two answers."""
    import core
    import corrections
    import rulebook

    alias = "Seeding boxes carry one colourway."
    canonical = "Only one colourway per seeding box."
    path = rulebook.overlay_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("version: fab-1.0\ndescribes: an alias\ncorrections:\n"
                    f"  - text: {alias}\n    provenance: Client call, 3 March\n",
                    encoding="utf-8")
    rulebook.load.cache_clear()

    cid = core.ingest_campaign(conn, title="Lima", market="Peru", status="concluded",
                               detail="A launch.", confirm=True)["campaign_id"]
    first = corrections.note(conn, text=alias, campaign_id=cid,
                             provenance="Deck")["correction_id"]
    second = corrections.note(conn, text=canonical, campaign_id=cid,
                              provenance="Deck")["correction_id"]
    corrections.resolve(conn, first, decision="same_rule", same_as=second)

    out = core.load_declared_corrections(conn, confirmed_by="R. Vega")

    assert out["loaded"] == 1
    assert not out.get("no_longer_declared"), (
        f"it loaded the rule and reported it deleted in the same breath: "
        f"{out.get('no_longer_declared')}")
    assert "NO LONGER IN YOUR RULEBOOK" not in out["what_it_means"]


def test_a_judgment_written_while_the_rule_reached_nothing_is_listed_after_the_repair(conn):
    """The repair's whole offer, on the sequence an upgraded install actually has: the rules
    were in force NOWHERE, a brief was judged without them, and then the upgrade put them in
    force everywhere. `confirmed_at` is deliberately preserved across the repair — it is the
    audit field, and rewriting it would make this report claim a judgment that CITES a rule
    was never checked against it — so the replay, comparing against `confirmed_at` alone, saw
    a rule confirmed before the judgment and said nothing. The offer opened an empty report.

    When a rule STARTED APPLYING is a different fact from when somebody confirmed it, and it
    is the one this report needs."""
    import core
    import replay

    _as_the_customers(conn)
    _as_the_old_loader_left_it(conn)

    peru = core.ingest_campaign(conn, title="Lima flagship", market="Peru",
                                status="concluded", detail="A launch.",
                                confirm=True)["campaign_id"]
    core.save_evaluation(conn, subject_title="Lima flagship", campaign_id=peru,
                         verdict="approve", summary="Looks sound.", findings=[])

    out = core.load_declared_corrections(conn, confirmed_by="R. Vega")
    assert [a["tool"] for a in out.get("next_actions") or []] == ["replay_rules"]

    report = replay.run(conn)
    assert report["judgments_total"] == 1, (
        "the repair offered the replay and the replay it offered is empty")
    row = report["judgments"][0]
    assert row["consequence"] == "rule_not_applied"
    assert len(row["not_checked_against"]["corrections"]) == 10


def test_a_first_load_reports_nothing_as_no_longer_declared(conn):
    """The drift sweep asks which standing rules the file has stopped declaring, and it asks
    it of the rows THIS run reached. A row created by this very run is not one of those — and
    when the loop recorded only pre-existing rows, a first load of a rulebook reported all ten
    of its own rules as no longer in it, in the response that had just loaded them."""
    import core

    _as_the_customers(conn)

    out = core.load_declared_corrections(conn, confirmed_by="R. Vega")

    assert out["loaded"] == 10
    assert not out.get("no_longer_declared"), out.get("no_longer_declared")
    assert "NO LONGER IN YOUR RULEBOOK" not in out["what_it_means"]


def test_two_lines_folded_by_punctuation_are_not_reported_as_somebody_s_decision(conn):
    """The same report, on the other reason two lines can be one rule. `rulebook` refuses an
    exact duplicate, but `corrections._normalise` folds case and punctuation — so two lines
    nobody was ever asked about collapse onto one row, and the sentence credited a person with
    a decision they never made. A heuristic reported as a judgment is the one distinction this
    product is built on, inverted."""
    import core
    import rulebook

    path = rulebook.overlay_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("version: fab-1.0\ndescribes: two spellings\ncorrections:\n"
                    "  - text: Seeding boxes carry one colourway.\n"
                    "    provenance: Client call, 3 March\n    markets: ['Mexico']\n"
                    "  - text: seeding boxes carry one colourway\n"
                    "    provenance: Client call, 9 May\n    markets: ['Japan']\n",
                    encoding="utf-8")
    rulebook.load.cache_clear()

    out = core.load_declared_corrections(conn, confirmed_by="R. Vega")

    assert out["loaded"] == 1
    assert [m["basis"] for m in out["same_rule_on_file"]] == ["heuristic"]
    assert "somebody answered" not in out["what_it_means"]
    assert "differ only in case or punctuation" in out["what_it_means"]


def test_a_declaration_reaches_the_rule_it_was_folded_into(conn):
    """`same_rule` is an answer this product offers, so a customer can end up with their
    rulebook's wording living as an alias of another rule. The loader resolves each line
    through `find` and promotes the right rule; `correction_status` compared the file's text
    against the row's canonical text and found nothing — so it told the customer their own
    declared rule was "seen in 1 campaign, needs 3" and offered no preview at all. Two lookups
    of one fact, disagreeing."""
    import corrections
    import core
    import rulebook

    alias = "Seeding boxes carry one colourway."
    canonical = "Only one colourway per seeding box."
    path = rulebook.overlay_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("version: fab-1.0\ndescribes: an alias\ncorrections:\n"
                    f"  - text: {alias}\n    provenance: Client call, 3 March\n",
                    encoding="utf-8")
    rulebook.load.cache_clear()

    cid = core.ingest_campaign(conn, title="Lima", market="Peru", status="concluded",
                               detail="A launch.", confirm=True)["campaign_id"]
    first = corrections.note(conn, text=alias, campaign_id=cid,
                             provenance="Deck")["correction_id"]
    second = corrections.note(conn, text=canonical, campaign_id=cid,
                              provenance="Deck")["correction_id"]
    corrections.resolve(conn, first, decision="same_rule", same_as=second)
    live = corrections.find(conn, alias)["correction_id"]

    gate = corrections.graduation(conn, live)

    assert gate["eligible"], gate["what_it_means"]
    assert gate["code"] == "declared_by_the_customer"
    assert gate.get("if_confirmed"), "no preview for a rule the customer's own file declares"
    # And confirming it by hand applies what the file says, not where it was overheard.
    corrections.graduate(conn, live, confirmed_by="R. Vega")
    assert corrections.describe(conn, live)["applies_everywhere"]


def test_an_unrelated_old_merge_does_not_make_a_punctuation_fold_a_decision(conn):
    """The `basis` is about the PAIR of lines being reported, not about the row's history.
    Asked of the row — "has anything ever been merged into it" — two lines differing by a full
    stop were credited to a person because an unrelated alias had been folded in long before,
    and the response said somebody had answered `same_rule` about lines nobody was ever
    shown."""
    import core
    import corrections
    import rulebook

    text = "Seeding boxes carry one colourway."
    path = rulebook.overlay_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("version: fab-1.0\ndescribes: two spellings\ncorrections:\n"
                    f"  - text: {text}\n    provenance: Client call\n"
                    "    markets: ['Mexico']\n"
                    "  - text: seeding boxes carry one colourway\n"
                    "    provenance: Client email\n    markets: ['Japan']\n",
                    encoding="utf-8")
    rulebook.load.cache_clear()

    cid = core.ingest_campaign(conn, title="Lima", market="Peru", status="concluded",
                               detail="A launch.", confirm=True)["campaign_id"]
    unrelated = corrections.note(conn, text="Something else entirely about pricing.",
                                 campaign_id=cid, provenance="Deck")["correction_id"]
    target = corrections.note(conn, text=text, campaign_id=cid,
                              provenance="Deck")["correction_id"]
    corrections.resolve(conn, unrelated, decision="same_rule", same_as=target)

    out = core.load_declared_corrections(conn, confirmed_by="R. Vega")

    assert [m["basis"] for m in out["same_rule_on_file"]] == ["heuristic"], (
        "an old merge of a different rule made this pair somebody's decision")
    assert "somebody answered" not in out["what_it_means"]
