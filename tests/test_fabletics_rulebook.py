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


def test_the_corrections_the_source_scoped_to_markets_say_so(conn):
    """Three of the ten came from particular markets and the source says which. A correction
    with no markets applies everywhere, which is right for a house rule and wrong for one
    learned in Colombia."""
    import yaml

    by_text = {c["text"][:40]: c for c in
               yaml.safe_load(FABLETICS.read_text(encoding="utf-8"))["corrections"]}
    scoped = {c["text"][:40]: c.get("markets") for c in by_text.values() if c.get("markets")}

    assert scoped, "every correction is declared as applying everywhere"
    assert any("Colombia" in m for m in scoped.values())
    assert any("Peru" in m for m in scoped.values())


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
    # Seven of the ten carry no market, and a house rule applies everywhere rather than
    # nowhere — the exact inversion D108's own fix shipped.
    assert len(standing) >= 7, f"only {len(standing)} of ten applied in Peru"


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
