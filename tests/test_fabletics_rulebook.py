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
    real = corrections.graduate
    corrections.graduate = lambda c, cid, **kw: real(c, cid, **{**kw, "markets": None,
                                                                "everywhere": False})
    try:
        core.load_declared_corrections(conn, confirmed_by="R. Vega")
    finally:
        corrections.graduate = real

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
