"""§12.1 — the rulebook as versioned configuration, applied deterministically.

The item is one sentence with a hard clause in it: "applied deterministically **outside** the
similarity path (today the rubric is a `reference` row retrieved by similarity, truncatable
and deletable)".

That parenthesis is the whole defect. Brand guidelines live in the library as an ordinary
`reference` record, so they reach a judgment only if they happen to rank in the top five for
that brief — and `readiness` says so in writing, in the one row on its `cannot` list that
names §12.1 as the work that would fix it:

    "Guidelines on file are retrieved by similarity like anything else, so a guardrail that is
     not retrieved is not a guardrail — it can say "this differs from what you did in Peru",
     which invites an argument, but not "this breaks your own rule", which does not."

A rule that applies when it happens to be retrieved is not a rule. Worse, the failure is
silent and asymmetric: the briefs most likely to breach a guideline are the ones least like
the guideline document, so the case the rulebook exists for is the case it is least likely to
be retrieved for.

So the tests here are mostly about the PINNING, not about YAML parsing.
"""
import pytest

import config
import core
import facts
import rulebook
import store


# ---------------------------------------------------------------------------------------
# It exists, it is versioned, and the version is the one stamped onto judgments


def test_the_product_ships_a_rulebook():
    """Bundled, not "upload one and hope". `RULEBOOK_VERSION` read
    "none (no rulebook ships yet — §12.1)" and every saved judgment carried that string."""
    loaded = rulebook.load()

    assert loaded["version"], "a rulebook with no version cannot be stamped onto anything"


def test_the_product_ships_no_rules_of_its_own():
    """Deliberately, and the reason is this codebase's signature defect.

    The first version shipped six rules about this product's own judgment discipline — and
    four of them already exist word for word in `EVALUATION_PROCEDURE`, which reaches the
    model three ways, while two more are enforced by validators that refuse the write. A
    second copy of a rule that already has one is exactly what §7.4 was written to stop, and
    it would have been shipped in the file whose whole purpose is being the single versioned
    place a rule lives.

    What belongs here is a rule about the CUSTOMER's briefs, which nothing else in this
    product knows. §12.3 ships a worked example as a customer file for the same reason."""
    assert rulebook.rules() == []
    assert rulebook.expects() == []


def test_the_version_is_what_lands_on_a_judgment():
    """§7.6 stamps `rulebook_version` onto every saved judgment so two judgments made under
    different rules can be told apart later. That is worth nothing if the constant and the
    file can disagree — the hand-maintained-copy failure this project has hit four times."""
    assert core.rulebook_version() == rulebook.version()
    assert "no rulebook ships yet" not in core.rulebook_version()


def test_a_rule_carries_what_a_finding_needs_to_cite_it():
    """§6.1 requires a guardrail breach to cite a `rule_id`, and §6.2 refuses one that cites
    a campaign instead. A rulebook whose rules have no stable ids cannot be cited, so the
    finding class §6.1 built would have nothing to point at."""
    for rule in rulebook.load()["rules"]:
        assert rule["id"], rule
        assert rule["rule"], rule
        assert rule["severity"] in ("blocking", "should_fix", "note"), rule
        # WHY, in the rulebook rather than invented per judgment. A rule with no reason is
        # one nobody can argue with, and §8.6's whole loop is about rules being arguable.
        assert rule["why"], rule


# ---------------------------------------------------------------------------------------
# The pinning — the actual item


def test_every_rule_reaches_the_judgment_without_being_retrieved(conn):
    """The item, stated as a test. The rules are in the contract the model is holding when it
    judges, whatever the brief is about and whatever similarity returned.

    The brief here is deliberately about nothing the rulebook mentions: under retrieval that
    is precisely the brief whose guardrails never arrive."""
    prepared = core.prepare_evaluation(
        conn, subject_title="A supermarket sampling day in Lyon",
        proposal_text="Two stands, a Saturday, eight hundred sachets, no media at all.")

    # The contract is the head of `note` — §7.4 put the procedure where the model is
    # holding the evidence package, and §7.5's argument is about WHEN rather than what.
    contract = prepared["note"]
    for rule in rulebook.load()["rules"]:
        assert rule["id"] in contract, (
            f"{rule['id']} did not reach the model, so it is not a rule — it is a document "
            f"that might rank"
        )


def test_more_rules_than_the_evidence_cap_all_reach_the_judgment(conn, tmp_path,
                                                                 monkeypatch):
    """`_PINNED_TOP_K` is 5, and the cap exists to stop a caller choosing how much PRECEDENT a
    judgment rests on. Rules are not precedent, so nothing may cap them.

    Twelve rules, against a cap of five. The earlier version of this test asserted
    `applied()["rules_applied"] == len(load()["rules"])` — both sides reading the same list,
    so it was `len(A) == len(A)` and could not fail. Mutation found it; review found it
    independently. This counts the ids that actually arrive in the contract."""
    _loaded(tmp_path, monkeypatch,
            "version: '1.0'\nrules:\n" + "".join(
                f"  - id: rule-{n}\n    rule: Rule number {n}.\n"
                f"    severity: should_fix\n    why: Reason {n}.\n" for n in range(12)))

    prepared = core.prepare_evaluation(
        conn, subject_title="Anything at all",
        proposal_text="A short brief with no detail in it.")

    arrived = [f"rule-{n}" for n in range(12) if f"rule-{n}" in prepared["note"]]
    assert len(arrived) == 12, f"only {len(arrived)} of 12 rules reached the model"
    assert prepared["rulebook"]["rules_applied"] == 12
    assert prepared["rulebook"]["basis"] == "computed"


def test_the_whole_rule_reaches_the_model_not_just_its_id(conn, tmp_path, monkeypatch):
    """Mutation: `as_contract` emitting only `[id] (severity)` with no rule TEXT survived
    every test in this file, because they all searched the contract for ids. A model handed a
    list of identifiers has not been given any rules."""
    _loaded(tmp_path, monkeypatch,
            "version: '1.0'\nrules:\n  - id: no-ai-imagery\n"
            "    rule: Never use AI-generated imagery in client-facing creative.\n"
            "    severity: blocking\n    why: Two clients have asked in writing.\n")

    note = core.prepare_evaluation(
        conn, subject_title="Anything at all",
        proposal_text="A short brief with no detail in it.")["note"]

    assert "Never use AI-generated imagery in client-facing creative." in note
    assert "Two clients have asked in writing." in note


def test_a_judgment_records_which_rulebook_it_was_made_under(conn):
    """§7.6's stamp, end to end rather than as a constant: onto the SAVED judgment, which is
    what somebody reads years later when they are wondering which rules were in force."""
    core.prepare_evaluation(conn, subject_title="Anything at all",
                            proposal_text="A short brief with no detail in it.")
    saved = core.save_evaluation(
        conn, subject_title="Anything at all", verdict="approve",
        summary="Nothing in the library bears on this brief, so it is judged on its own.",
        findings=[])

    stamp = store.get_evaluation(conn, saved["evaluation_id"])["provenance"]
    assert stamp["rulebook_version"] == rulebook.version()
    assert "no rulebook ships yet" not in stamp["rulebook_version"]


# ---------------------------------------------------------------------------------------
# What the product may now claim


def test_the_limit_now_names_a_gesture_instead_of_a_planned_item(conn):
    """The row that named this item as its own remedy: "the rulebook to be pinned rather than
    retrieved, which is planned work". A limit whose remedy is a roadmap number is one the
    reader can do nothing about.

    It is still a `cannot` on a fresh install — no rules are written — but what it needs is
    now something the person reading it can go and do this afternoon."""
    row = next(r for r in core.readiness(conn)["cannot"]
               if r["code"] == "check_against_rules")

    assert "planned work" not in row["needs"]
    assert rulebook.BUNDLED_NAME in row["needs"]


def test_with_no_rules_written_the_product_does_not_claim_to_check_them(conn):
    """The overclaim, caught by review before it shipped.

    `check_against_rules` went onto `can` the moment the rulebook FILE existed — so a fresh
    install advertised "I can check this against your rules and say 'this breaks your own
    rule'" while holding no rules at all. That is precisely the confident first-run claim
    `readiness` exists to prevent, arriving through the item meant to remove it.

    The mechanism shipping is not the capability existing. With nothing written down, this is
    a `cannot` that names what would lift it."""
    told = core.readiness(conn)

    assert "check_against_rules" not in {row["code"] for row in told["can"]}
    row = next(r for r in told["cannot"] if r["code"] == "check_against_rules")
    assert "rulebook" in row["needs"]


def test_with_rules_written_it_claims_exactly_them(conn, tmp_path, monkeypatch):
    """And the other half, or the fix is just the old limit back. Rules written down ARE
    checkable, the claim names how many and which version, and it carries its boundary."""
    _loaded(tmp_path, monkeypatch,
            "version: 'acme-2'\nrules:\n  - id: no-ai-imagery\n    rule: No AI imagery.\n"
            "    severity: blocking\n    why: Two clients have asked in writing.\n")

    told = core.readiness(conn)

    row = next(r for r in told["can"] if r["code"] == "check_against_rules")
    assert "1 rule" in row["what"] and "acme-2" in row["what"]
    assert "written down" in row["bounded_by"] and "similarity" in row["bounded_by"]


# ---------------------------------------------------------------------------------------
# Failure modes a customer will actually hit


def test_a_broken_rulebook_says_which_line_and_refuses_to_run(tmp_path, monkeypatch):
    """A rulebook that fails to parse must not degrade into "no rules". Silently running with
    an empty rulebook is how a customer's guardrails stop applying without anybody noticing —
    the same shape as the `reference` row that was never retrieved, arriving through a typo.
    """
    broken = tmp_path / "rulebook.yaml"
    broken.write_text("version: 1\nrules:\n  - id: one\n   rule: bad indent\n")
    monkeypatch.setattr(rulebook, "_bundled", lambda: broken)
    rulebook.load.cache_clear()

    with pytest.raises(ValueError) as raised:
        rulebook.load()

    assert "rulebook" in str(raised.value).lower()
    assert "line" in str(raised.value).lower()
    rulebook.load.cache_clear()


def test_a_rulebook_missing_from_the_install_is_an_error_not_an_empty_one(
        tmp_path, monkeypatch):
    """Same rule, other cause. An installed copy ships this file; its absence means the
    install is broken, and answering "no rules apply" would be this product asserting
    something false about the customer's own guidelines."""
    monkeypatch.setattr(rulebook, "_bundled", lambda: tmp_path / "not-there.yaml")
    rulebook.load.cache_clear()

    with pytest.raises(ValueError, match="rulebook"):
        rulebook.load()

    rulebook.load.cache_clear()


def test_a_rule_with_no_id_is_refused_rather_than_skipped(tmp_path, monkeypatch):
    """Skipping it would mean a customer's rule quietly not applying, which is the failure
    this whole item is about — and they would have no way to tell, because the count they
    would compare it against comes from the same skipped list."""
    partial = tmp_path / "rulebook.yaml"
    partial.write_text(
        "version: '1.0'\nrules:\n"
        "  - id: colour\n    rule: Never on a dark background.\n"
        "    severity: blocking\n    why: It is unreadable.\n"
        "  - rule: A rule with no id.\n    severity: should_fix\n    why: Because.\n")
    monkeypatch.setattr(rulebook, "_bundled", lambda: partial)
    rulebook.load.cache_clear()

    with pytest.raises(ValueError, match="id"):
        rulebook.load()

    rulebook.load.cache_clear()


def test_the_file_is_read_from_beside_an_installed_binary(monkeypatch, tmp_path):
    """`config.app_dir()` rather than `__file__`, for the reason its own docstring gives: in
    a frozen app `__file__` points inside the bundle, where an administrator can neither see
    nor replace a file — and §12.2's whole premise is that a customer edits this."""
    monkeypatch.setattr(config, "app_dir", lambda: tmp_path)
    installed = tmp_path / "installed"
    installed.mkdir(exist_ok=True)
    monkeypatch.setattr(config, "app_dir", lambda: installed)
    (installed / "rulebook.yaml").write_text(
        "version: 'from-the-install-dir'\nrules:\n"
        "  - id: one\n    rule: A rule.\n    severity: should_fix\n    why: Because.\n")
    rulebook.load.cache_clear()
    try:
        assert rulebook.version() == "from-the-install-dir"
    finally:
        rulebook.load.cache_clear()


# ---------------------------------------------------------------------------------------
# Startup


def test_the_server_loads_the_rulebook_before_it_starts_serving(monkeypatch):
    """Loaded at startup, which the item says in as many words — and it is not a performance
    point. `load()` is the only thing that reads the file, so without this a broken rulebook
    is discovered by the first `prepare_evaluation` of the session, as a tool error, to a
    model in the middle of judging something. The person who edited the file is by then
    somewhere else entirely.

    Failing at startup puts the error where the gesture was."""
    import sys
    import types

    calls = []
    monkeypatch.setattr("config.ensure_dirs", lambda: calls.append("ensure_dirs"))
    monkeypatch.setattr("store.init_db", lambda: calls.append("init_db"))
    monkeypatch.setattr("clip_embed.warm_up", lambda: calls.append("warm_up"))
    monkeypatch.setattr("embedding.warm_up", lambda: calls.append("embed_warm_up"))
    monkeypatch.setattr("rulebook.version", lambda: calls.append("rulebook") or "core-1.0")
    monkeypatch.setattr("mcp_server.mcp",
                        types.SimpleNamespace(run=lambda transport=None:
                                              calls.append(f"run:{transport}")))
    monkeypatch.setattr(sys, "argv", ["campaign-intelligence", "stdio"])

    import main

    main.main()

    assert "rulebook" in calls, "a broken rulebook must not wait for the first judgment"
    assert calls.index("rulebook") < calls.index("run:stdio")


# ---------------------------------------------------------------------------------------
# The refusals, each of which is a rule that would otherwise quietly not apply
#
# Mutation found every one of these unasserted: the loader's checks could all be deleted and
# the suite stayed green. They are the checks whose absence is invisible — a customer whose
# rule was dropped sees a shorter list, and has nothing to compare it against.


def _loaded(tmp_path, monkeypatch, text):
    path = tmp_path / "rulebook.yaml"
    path.write_text(text)
    monkeypatch.setattr(rulebook, "_bundled", lambda: path)
    rulebook.load.cache_clear()
    return path


@pytest.fixture(autouse=True)
def _forget_the_rulebook():
    """Every test here either loads the real file or points at a temporary one, and `load` is
    cached for the process — so a stale parse would leak into the next test either way."""
    yield
    rulebook.load.cache_clear()


def test_a_rule_with_an_unknown_severity_is_refused(tmp_path, monkeypatch):
    """`severity` is what decides whether a breach blocks a verdict. A value nothing
    recognises would be dropped when the finding was written, so the rule would be checked
    and its breach would carry no weight — the worst of both."""
    _loaded(tmp_path, monkeypatch,
            "version: '1.0'\nrules:\n  - id: one\n    rule: A rule.\n"
            "    severity: critical\n    why: Because.\n")

    with pytest.raises(ValueError, match="severity"):
        rulebook.load()


def test_two_rules_cannot_share_an_id(tmp_path, monkeypatch):
    """A finding cites a rule by id (§6.1). Two rules under one id means a breach names both,
    and nobody reading the judgment can tell which was broken — while the product goes on
    presenting the citation as the thing that makes the finding checkable."""
    _loaded(tmp_path, monkeypatch,
            "version: '1.0'\nrules:\n"
            "  - id: colour\n    rule: Never on dark.\n    severity: blocking\n    why: A.\n"
            "  - id: colour\n    rule: Never on red.\n    severity: should_fix\n    why: B.\n")

    with pytest.raises(ValueError, match="id"):
        rulebook.load()


def test_a_rulebook_with_no_rules_loads_and_says_so(tmp_path, monkeypatch):
    """An empty rules list is allowed, because the PRODUCT ships one.

    The first version refused it, reasoning that an empty rulebook is the situation this file
    replaced. But that situation was rules on file that never APPLIED — a different thing from
    no rules having been written. Refusing it would have forced the product to ship rule
    content it has no business owning, and `readiness` is where "you have not written any
    rules" belongs: it is a fact about the customer's configuration, not a broken file."""
    _loaded(tmp_path, monkeypatch, "version: '1.0'\nrules: []\n")

    assert rulebook.load()["rules"] == []


def test_a_rulebook_with_no_version_is_refused(tmp_path, monkeypatch):
    """§7.6 stamps the version onto every judgment so two made under different rules can be
    told apart years later. A rulebook that cannot be named cannot be stamped, and the stamp
    is the entire reason this is a versioned file rather than a list in the code."""
    _loaded(tmp_path, monkeypatch,
            "rules:\n  - id: one\n    rule: A rule.\n    severity: should_fix\n    why: Because.\n")

    with pytest.raises(ValueError, match="version"):
        rulebook.load()


def test_the_model_is_given_the_reason_for_each_rule(conn):
    """A rule with no reason is one nobody can argue with, and §8.6's whole loop is about
    rules being arguable — a model that cannot see why a rule exists cannot tell a real breach
    from a technicality, and cannot say anything useful when it thinks the rule is wrong."""
    contract = rulebook.as_contract()

    for rule in rulebook.rules():
        assert rule["why"] in contract, (
            f"{rule['id']} reached the model as an instruction with no reason attached"
        )


# ---------------------------------------------------------------------------------------
# D27, D50, D74, D92 — the four rows that were owed to this item and could not be done
# before there was a rulebook to do them against.


def test_health_check_reports_the_rulebook(conn):
    """D27. The rulebook is a shipped payload like the CLIP weights, and `health_check`'s own
    contract is that every component fails independently and says so — "vision being
    unavailable is not the server being down".

    A missing or unparseable rulebook is exactly that kind of failure: the server answers, the
    database is fine, and every judgment it makes is missing its rules. Before this it was
    invisible to the diagnostic, so an install broken in the one way §12.1 cares about would
    pass its own gate."""
    report = core.health_check(conn, probe=False)

    assert "rulebook" in report["components"]
    component = report["components"]["rulebook"]
    assert component["ok"] is True
    assert rulebook.version() in component["detail"]


def test_a_broken_rulebook_fails_the_health_check_rather_than_the_call(conn, tmp_path,
                                                                      monkeypatch):
    """The second half of `health_check`'s contract: it never raises. A diagnostic that
    crashes on the thing it is diagnosing leaves the administrator exactly where they
    started — and this one is called by the installers' post-install gate."""
    broken = tmp_path / "rulebook.yaml"
    broken.write_text("version: '1.0'\nrules:\n  - id: one\n   rule: bad indent\n")
    monkeypatch.setattr(rulebook, "_bundled", lambda: broken)
    rulebook.load.cache_clear()

    report = core.health_check(conn, probe=False)

    component = report["components"]["rulebook"]
    assert component["ok"] is False
    assert component["code"] == "rulebook_unreadable"
    assert component["affects"] and component["remedy"]
    assert report["ok"] is False, (
        "a server judging with no rules is not a healthy server, and the installers' gate "
        "reads this field"
    )


def test_the_rulebook_declares_what_a_brief_is_expected_to_carry(conn):
    """D50. `gaps()`'s third case — "a named expected input that has never been supplied" —
    needed a rulebook that DECLARES expected inputs. The rulebook is the only place that can:
    an expectation nobody wrote down cannot be reported as missing without the product
    inventing a requirement on the customer's behalf."""
    expects = rulebook.expects()

    assert isinstance(expects, list)
    for expected in expects:
        assert expected["id"] and expected["input"] and expected["why"], expected


def test_a_declared_input_that_no_brief_carries_is_reported_as_a_gap(conn, tmp_path,
                                                                    monkeypatch):
    """D50 proper. The product ships no expectations — they are the customer's, and §12.2 is
    where they arrive — so this declares one and checks the machinery reads it."""
    declared = tmp_path / "rulebook.yaml"
    declared.write_text(
        "version: '1.0'\n"
        "rules:\n  - id: one\n    rule: A rule.\n    severity: should_fix\n    why: Because.\n"
        "expects:\n"
        "  - id: kpi-workbook\n    input: A KPI workbook\n"
        "    why: Nothing can be reconciled against a brief that never said what it was for.\n"
        "    looks_like: ['kpi', 'workbook', 'targets']\n")
    monkeypatch.setattr(rulebook, "_bundled", lambda: declared)
    rulebook.load.cache_clear()

    for title, detail in (("Bogota launch", "A store opening with influencers and OOH."),
                          ("Lima launch", "A second store opening, same creative.")):
        core.ingest_campaign(conn, title=title, market="LATAM", status="concluded",
                             detail=detail, confirm=True)

    reported = core.gaps(conn)
    codes = {gap["code"] for gap in reported["gaps"]}

    assert "expected_input_never_supplied" in codes
    gap = next(g for g in reported["gaps"] if g["code"] == "expected_input_never_supplied")
    assert "KPI workbook" in gap["what_it_means"]
    assert gap["basis"] == "computed"
    assert "kpi-workbook" in str(gap)


def test_a_reference_record_is_not_mistaken_for_the_rulebook(conn):
    """D74. `has_rulebook` meant "any `reference` record exists", so a competitor's deck filed
    as reference material made the product report that the customer's guidelines were on file
    — and `readiness` then told them the rulebook step of the shortest path was done.

    The rulebook is a declared artefact now, so the two questions are separable and both are
    answerable."""
    core.ingest_campaign(conn, title="Competitor teardown", record_type="reference",
                         detail="What the other agency shipped in Q3.", confirm=True)

    report = core.readiness(conn)

    assert report["has_rulebook"] is True, "the rulebook FILE ships, so it is always in force"
    assert report["has_reference_material"] is True, "and the teardown IS reference material"
    # The two are separate facts now. The step that used to say "add your brand guidelines as
    # the rulebook" ticked itself on this upload, telling the customer they had put rules in
    # force when they had filed a competitor's deck.
    assert not any("as the rulebook" in str(step) for step in report["shortest_path"])


# ---------------------------------------------------------------------------------------
# D92 — the sixth computed fact


def test_a_rule_can_declare_the_words_that_would_breach_it(tmp_path, monkeypatch):
    """D92, the sixth computed fact §7.1 asks for, which needed a rulebook that HAS keywords.
    The product ships none: which words breach a rule is the customer's own judgment, and
    inventing them would be the product asserting a guardrail nobody wrote."""
    _loaded(tmp_path, monkeypatch,
            "version: '1.0'\nrules:\n"
            "  - id: no-ai-imagery\n    rule: Never use AI-generated imagery.\n"
            "    severity: blocking\n    why: The client's audience reacts badly to it.\n"
            "    watch_for: ['ai-generated', 'midjourney', 'generative imagery']\n")

    rule = rulebook.by_id("no-ai-imagery")
    assert rule["watch_for"] == ["ai-generated", "midjourney", "generative imagery"]


def test_the_brief_is_checked_against_those_words_and_the_hit_is_computed(conn, tmp_path,
                                                                         monkeypatch):
    """§7.8's `basis` distinction on the one fact where it matters most: the server READ the
    word, so a `guardrails` finding carries the sentence it was read from and the rule it
    belongs to. A model asserting "this breaks your AI-imagery rule" is a claim; this is a
    quotation."""
    _loaded(tmp_path, monkeypatch,
            "version: '1.0'\nrules:\n"
            "  - id: no-ai-imagery\n    rule: Never use AI-generated imagery.\n"
            "    severity: blocking\n    why: The client's audience reacts badly to it.\n"
            "    watch_for: ['midjourney']\n")

    computed = facts.compute(
        "A six-week push in Bogota. Key visuals generated in Midjourney to save on the "
        "photo shoot, with paid social behind them.")

    guardrails = computed["guardrails"]
    assert guardrails["status"] == "contradicted"
    assert guardrails["basis"] == "computed"
    hit = guardrails["hits"][0]
    assert hit["rule_id"] == "no-ai-imagery"
    assert hit["severity"] == "blocking"
    assert "Midjourney" in hit["evidence"], "the sentence, not just the word"


def test_a_brief_that_breaches_nothing_says_checked_rather_than_silent(conn, tmp_path,
                                                                      monkeypatch):
    """`nothing_to_check` is never a pass, and neither is silence. A brief checked against
    one rule and found clean is a different fact from a brief nobody checked."""
    _loaded(tmp_path, monkeypatch,
            "version: '1.0'\nrules:\n"
            "  - id: no-ai-imagery\n    rule: Never use AI-generated imagery.\n"
            "    severity: blocking\n    why: The client's audience reacts badly to it.\n"
            "    watch_for: ['midjourney']\n")

    computed = facts.compute("A six-week push in Bogota with a photo shoot and paid social.")

    assert computed["guardrails"]["status"] == "checked_clean"
    assert computed["guardrails"]["rules_checked"] == 1


def test_with_no_watch_words_declared_the_check_says_it_did_not_run(conn):
    """The product's own rulebook declares no keywords, so out of the box this fact must read
    `nothing_to_check` — and must NOT read clean. A brief reported as breaching no guardrails
    when nothing was ever checked is the exact false pass §11.7 spent a whole item removing
    from the language checks."""
    computed = facts.compute("A six-week push in Bogota with a photo shoot and paid social.")

    assert computed["guardrails"]["status"] == "nothing_to_check"
    assert computed["guardrails"]["rules_checked"] == 0
    assert "no rule" in computed["guardrails"]["what_it_means"].lower()


# ---------------------------------------------------------------------------------------
# Citing a rule
#
# The single most important thing §12.1 claims, and until review walked it, the one thing the
# product refused. `as_contract` tells the model "a finding that a rule is breached cites its
# id" and `readiness` says "citing the rule by its id" — while `_verify_quote` resolved
# `rule_id` through `store.text_on_file`, a lookup in the CAMPAIGNS table, and demanded
# `record_type == "reference"`. So every real rulebook id was refused, the refusal steered the
# model to downgrade the finding to `missing_information`, and the only citable "rule" was the
# similarity-retrieved reference record §12.1 exists to replace.


def _with_a_rule(tmp_path, monkeypatch):
    return _loaded(tmp_path, monkeypatch,
                   "version: 'acme-2'\nrules:\n  - id: no-ai-imagery\n"
                   "    rule: Never use AI-generated imagery in client-facing creative.\n"
                   "    severity: blocking\n    why: Two clients have asked in writing.\n")


def test_a_finding_can_cite_a_rule_from_the_rulebook(conn, tmp_path, monkeypatch):
    """The claim, made good."""
    _with_a_rule(tmp_path, monkeypatch)

    saved = core.save_evaluation(
        conn, subject_title="Bogota launch", verdict="revise",
        summary="The key visuals breach the imagery rule.",
        approve_if="The AI-generated key visuals are replaced with shot photography.",
        findings=[{"severity": "blocking", "kind": "guardrail_breach",
                   "finding": "The key visuals are AI-generated.",
                   "fix": "Replace them with shot photography.",
                   "precedent": {"rule_id": "no-ai-imagery",
                                 "quote": "Never use AI-generated imagery in client-facing "
                                          "creative."}}])

    stored = store.get_evaluation(conn, saved["evaluation_id"])
    cited = stored["findings"][0]["precedent"]
    assert cited["rule_id"] == "no-ai-imagery"
    assert cited["rule_source"] == "product", "which rulebook the rule came from"
    assert cited["checked"] == ["rulebook", "wording"], (
        "and what the server actually checked: that the rule exists and that the quote is "
        "the rule's own words — not that the brief breaches it, which is the judgment"
    )
    assert stored["provenance"]["rulebook_version"] == "acme-2", (
        "and the judgment records WHICH rulebook the rule it cited came from"
    )


def test_a_quote_the_rule_does_not_contain_is_refused(conn, tmp_path, monkeypatch):
    """The same guarantee §6.1 gives for a campaign quote, on the citation the product calls
    not debatable. A breach finding quoting a rule that does not say that is worse than an
    uncited one: it reads as established."""
    _with_a_rule(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="no-ai-imagery"):
        core.save_evaluation(
            conn, subject_title="Bogota launch", verdict="revise",
            summary="The key visuals breach the imagery rule.",
            approve_if="The key visuals are replaced.",
            findings=[{"severity": "blocking", "kind": "guardrail_breach",
                       "finding": "The key visuals are AI-generated.",
                       "fix": "Replace them.",
                       "precedent": {"rule_id": "no-ai-imagery",
                                     "quote": "Never use purple on packaging."}}])


def test_citing_a_rule_that_is_not_in_the_rulebook_is_refused(conn, tmp_path, monkeypatch):
    """A rule id the rulebook does not contain is a rule nobody wrote. Before this, a
    freshly-ingested `reference` record's campaign id was accepted as a `rule_id` — so a
    judgment stamped `acme-2` could carry a breach of a rule `acme-2` does not contain."""
    _with_a_rule(tmp_path, monkeypatch)
    invented = core.ingest_campaign(
        conn, title="Brand guidelines", record_type="reference",
        detail="Never use purple on packaging.", confirm=True)["campaign_id"]

    with pytest.raises(ValueError) as raised:
        core.save_evaluation(
            conn, subject_title="Bogota launch", verdict="revise",
            summary="Purple packaging.", approve_if="It is not purple.",
            findings=[{"severity": "blocking", "kind": "guardrail_breach",
                       "finding": "The packaging is purple.", "fix": "Change it.",
                       "precedent": {"rule_id": invented,
                                     "quote": "Never use purple on packaging."}}])

    assert "rulebook" in str(raised.value)
    assert "acme-2" in str(raised.value), "and it says which rulebook was searched"


def test_the_refusal_lists_the_rules_there_actually_are(conn, tmp_path, monkeypatch):
    """A refusal that does not say what WOULD be accepted sends the model guessing, and §2.4's
    lesson is that the easy exit in a validation message is the one that gets taken — here
    that exit is downgrading a breach to `missing_information`."""
    _with_a_rule(tmp_path, monkeypatch)

    with pytest.raises(ValueError) as raised:
        core.save_evaluation(
            conn, subject_title="Bogota launch", verdict="revise",
            summary="Something.", approve_if="Something else.",
            findings=[{"severity": "blocking", "kind": "guardrail_breach",
                       "finding": "A breach.", "fix": "Fix it.",
                       "precedent": {"rule_id": "no-such-rule",
                                     "quote": "Never use AI-generated imagery."}}])

    assert "no-ai-imagery" in str(raised.value)


def test_the_http_server_loads_the_rulebook_before_it_serves(monkeypatch):
    """The startup load was in `main.py`'s `stdio` branch alone. `serve` never touched it, so
    an HTTP deployment discovered a broken rulebook as a tool error mid-request — and the
    first tool to crash is `readiness`, whose entire job is saying what the product cannot do.
    """
    import http_app

    calls = []
    monkeypatch.setattr("config.ensure_dirs", lambda: calls.append("ensure_dirs"))
    monkeypatch.setattr("store.init_db", lambda: calls.append("init_db"))
    monkeypatch.setattr("clip_embed.warm_up", lambda: calls.append("warm_up"))
    monkeypatch.setattr("embedding.warm_up", lambda: calls.append("embed_warm_up"))
    monkeypatch.setattr("rulebook.version", lambda: calls.append("rulebook") or "core-1.0")
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: calls.append("serving"))

    http_app.main()

    assert "rulebook" in calls
    assert calls.index("rulebook") < calls.index("serving")


def test_the_source_checkout_entry_point_loads_it_too(monkeypatch):
    """`stdio_server.py` is what `configure-desktop` wires for a source checkout, and it is a
    separate entry point that has silently missed a startup step before — §1.1's warm-up was
    in this file and NOT in `main.py stdio`, which was the whole of product-review defect 04.
    The same omission, the other way round."""
    import types

    import stdio_server

    calls = []
    monkeypatch.setattr("config.ensure_dirs", lambda: calls.append("ensure_dirs"))
    monkeypatch.setattr("store.init_db", lambda: calls.append("init_db"))
    monkeypatch.setattr("clip_embed.warm_up", lambda: calls.append("warm_up"))
    monkeypatch.setattr("embedding.warm_up", lambda: calls.append("embed_warm_up"))
    monkeypatch.setattr("rulebook.version", lambda: calls.append("rulebook") or "core-1.0")
    # On `stdio_server`, not `mcp_server`: this module does `from mcp_server import mcp` at
    # import time, so the name it calls is its own.
    monkeypatch.setattr(stdio_server, "mcp",
                        types.SimpleNamespace(run=lambda transport=None:
                                              calls.append(f"run:{transport}")))

    stdio_server.main()

    assert "rulebook" in calls
    assert calls.index("rulebook") < calls.index("run:stdio")


def test_a_frozen_build_missing_the_visible_copy_still_starts(monkeypatch, tmp_path):
    """The P0 review found: PyInstaller 6 puts every `datas` entry under the contents
    directory `_internal/` whatever relative destination the spec names, while
    `config.app_dir()` deliberately resolves BESIDE the executable — so the file the spec
    shipped and the file the code read were different paths, and every frozen install raised
    "the rulebook is missing" at startup. The Claude Desktop connector would have died on
    launch, on every installed copy.

    The build scripts now place the visible copy. This is the belt: a build that forgets it
    starts anyway, rather than a customer's install being bricked by a packaging slip."""
    import sys

    bundle = tmp_path / "_internal"
    bundle.mkdir()
    (bundle / "rulebook.yaml").write_text("version: 'from-the-bundle'\nrules: []\n")
    monkeypatch.setattr(config, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    rulebook.load.cache_clear()
    try:
        assert rulebook.version() == "from-the-bundle"
        assert rulebook.is_the_editable_copy() is False
    finally:
        monkeypatch.delattr(sys, "_MEIPASS", raising=False)
        rulebook.load.cache_clear()


def test_the_visible_copy_wins_and_the_health_check_says_which_is_in_force(
        conn, monkeypatch, tmp_path):
    """The braces. A silent fallback is its own failure: an administrator editing the file
    they can see, on an install reading the buried one, would change nothing and be told
    nothing — which is worse than either half alone, because it looks like it worked."""
    import sys

    installed = tmp_path / "installed"
    installed.mkdir(exist_ok=True)
    bundle = installed / "_internal"
    bundle.mkdir()
    (bundle / "rulebook.yaml").write_text("version: 'from-the-bundle'\nrules: []\n")
    (installed / "rulebook.yaml").write_text("version: 'the-one-you-can-edit'\nrules: []\n")
    monkeypatch.setattr(config, "app_dir", lambda: installed)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    rulebook.load.cache_clear()
    try:
        assert rulebook.version() == "the-one-you-can-edit"
        assert rulebook.is_the_editable_copy() is True

        (installed / "rulebook.yaml").unlink()
        rulebook.load.cache_clear()
        component = core.health_check(conn, probe=False)["components"]["rulebook"]
        assert component["ok"] is True, "it works, so it is not a failure"
        assert component["degraded"], "but it is not the file anybody can edit, and says so"
        assert "edit" in component["degraded"]
    finally:
        monkeypatch.delattr(sys, "_MEIPASS", raising=False)
        rulebook.load.cache_clear()


def test_the_finding_records_which_rulebook_the_rule_came_from(conn, tmp_path, monkeypatch):
    """`source` was carried on every rule from the first version of the loader and read by
    nothing — a field whose stated purpose ("so a judgment can say whether a breach was of the
    product's rule or the customer's") no caller could achieve. Review found it write-only;
    mutation confirmed that hard-coding it to `"product"` broke no test.

    It matters most once §12.2's overlays exist: two judgments citing `no-ai-imagery` may be
    citing two different rules, and the provenance stamp is a single scalar."""
    _loaded(tmp_path, monkeypatch,
            "version: 'acme-2'\nrules:\n  - id: no-ai-imagery\n"
            "    rule: Never use AI-generated imagery in client-facing creative.\n"
            "    severity: blocking\n    why: Two clients have asked in writing.\n"
            "    source: acme-house-rules\n")

    saved = core.save_evaluation(
        conn, subject_title="Bogota launch", verdict="revise",
        summary="The key visuals breach the imagery rule.",
        approve_if="The key visuals are replaced.",
        findings=[{"severity": "blocking", "kind": "guardrail_breach",
                   "finding": "The key visuals are AI-generated.",
                   "fix": "Replace them.",
                   "precedent": {"rule_id": "no-ai-imagery",
                                 "quote": "Never use AI-generated imagery in client-facing "
                                          "creative."}}])

    cited = store.get_evaluation(conn, saved["evaluation_id"])["findings"][0]["precedent"]
    assert cited["rule_source"] == "acme-house-rules", (
        "whose rule it was, not which file happened to be loaded"
    )


def test_a_brief_that_carries_the_declared_input_closes_the_gap(conn, tmp_path, monkeypatch):
    """The other half of D50, and the half that makes the gap worth reporting: a gap that
    cannot be closed is one everybody learns to ignore, which is §8.2's own lesson about
    re-asking a question somebody declined.

    Mutation found this unasserted — the `looks_like` search could be made to match NOTHING
    and every test still passed, because they all checked that the gap FIRES."""
    _loaded(tmp_path, monkeypatch,
            "version: '1.0'\nrules: []\nexpects:\n"
            "  - id: kpi-workbook\n    input: A KPI workbook\n"
            "    why: Nothing can be reconciled against a brief that never said what it was "
            "for.\n    looks_like: ['kpi', 'workbook']\n")
    core.ingest_campaign(conn, title="Bogota launch", market="LATAM", status="concluded",
                         detail="A store opening with influencers and OOH.", confirm=True)
    core.ingest_campaign(conn, title="Lima launch", market="LATAM", status="concluded",
                         detail="A second store opening. Targets are in the attached KPI "
                                "workbook.", confirm=True)

    codes = {gap["code"] for gap in core.gaps(conn)["gaps"]}

    assert "expected_input_never_supplied" not in codes, (
        "one brief carrying it is enough: the claim is that NO record on file mentions one"
    )


def test_an_expectation_with_nothing_to_look_for_is_refused(tmp_path, monkeypatch):
    """It used to be skipped — an expectation that quietly never applies, which is the exact
    failure this loader exists to refuse, arriving through the one field that makes the
    expectation checkable. The customer would see a shorter list and have nothing to compare
    it against."""
    _loaded(tmp_path, monkeypatch,
            "version: '1.0'\nrules: []\nexpects:\n"
            "  - id: kpi-workbook\n    input: A KPI workbook\n    why: Because.\n")

    with pytest.raises(ValueError, match="looks_like"):
        rulebook.load()
