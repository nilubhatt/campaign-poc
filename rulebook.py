"""The rulebook: the rules every judgment is made under, loaded from a versioned file (§12.1).

**The item is the word "deterministically", not the word "YAML".**

Before this, the rules lived in the library as an ordinary `reference` record, so they reached
a judgment only when they happened to rank in the top five most similar records for that
brief. `readiness` carried the consequence on its `cannot` list, naming this item by number:

    "Guidelines on file are retrieved by similarity like anything else, so a guardrail that is
     not retrieved is not a guardrail — it can say 'this differs from what you did in Peru',
     which invites an argument, but not 'this breaks your own rule', which does not."

Two things made that worse than a missing feature. The failure was silent: nothing recorded
that a rule had not been applied, so a judgment made with the guardrails absent is
indistinguishable from one made with them present. And it was the wrong way round: the briefs
least like the guidelines document are the ones least likely to retrieve it, and they are
exactly the briefs most likely to breach it.

So everything here is in service of one property — every rule reaches every judgment, is not
ranked, is not truncated, and cannot be deleted by anyone editing the library.

WHY NOT IN THE DATABASE. A rule in the database is a row, and rows are editable by whoever can
call `update_campaign`, deletable, and invisible to a diff. A rule in a file is something an
administrator can see, a customer can put in version control, and an upgrade can carry
forward. §12.2's overlay depends on that: it is a file the customer owns, layered over this
one, so an upgrade replacing the product's own rules cannot take theirs with it.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

import config

# What a rule must carry. `id` because §6.1 requires a guardrail breach to cite a `rule_id`
# and refuses one that cites a campaign instead — a rule with no stable id cannot be cited, so
# the finding class §6.1 built would have nothing to point at. `why` because §8.6's whole loop
# is about rules being arguable, and a rule with no reason is one nobody can argue with.
_REQUIRED = ("id", "rule", "severity", "why")

# A finding's vocabulary, minus `note`. Review found `note` was UNEXPRESSIBLE: `core.py`'s
# finding validator refuses a `guardrail_breach` below `should_fix` — "if the rule applies the
# finding is blocking" — so a rule shipped at `note` severity could be breached and the breach
# could not be written down. A value the rest of the product cannot express is worse than a
# missing one, because the file reads as though it were honoured.
SEVERITIES = ("blocking", "should_fix")

BUNDLED_NAME = "rulebook.yaml"


def _inside_the_bundle() -> Optional[Path]:
    """The copy PyInstaller collected, if this is a frozen build.

    PyInstaller 6 puts every `datas` entry under the contents directory `_internal/`,
    whatever relative destination the spec names — so the spec's `("rulebook.yaml", ".")`
    lands at `<app>/_internal/rulebook.yaml` and NOT beside the executable. Review built a
    onedir bundle and confirmed it. Without this, every frozen install raised "the rulebook is
    missing" at startup and the Claude Desktop connector died on launch.
    """
    import sys

    base = getattr(sys, "_MEIPASS", None)
    return Path(base) / BUNDLED_NAME if base else None


def _bundled() -> Path:
    """Where the rulebook is read from.

    `config.app_dir()` FIRST, for the reason its docstring gives: frozen, `__file__` points
    inside the bundle, where an administrator can neither see nor replace a file — and a
    rulebook nobody can edit is most of what this item was about. The build scripts copy it
    there beside the executable.

    The bundled copy is the fallback and not the preference. It exists so that a build which
    forgot the copy still starts; `health_check` says which one is in use, because an
    administrator editing the visible file and seeing nothing change would be worse than
    either failure on its own.
    """
    beside = config.app_dir() / BUNDLED_NAME
    if beside.exists():
        return beside
    inside = _inside_the_bundle()
    return inside if inside and inside.exists() else beside


def is_the_editable_copy() -> bool:
    """Whether the file in force is the one an administrator can see and change."""
    return _bundled() == config.app_dir() / BUNDLED_NAME


def _read(path: Path) -> dict:
    """Parse one rulebook file, or say what is wrong with it in terms of that file.

    A YAML error names a line, and that line number is the whole value of the message to
    somebody who has just edited the file — so it is passed through rather than replaced with
    a tidier sentence that loses it.
    """
    import yaml

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ValueError(
            f"the rulebook is missing: no file at {path}. This ships with the product, so its "
            f"absence means the install is incomplete — re-run the installer. It is NOT "
            f"treated as 'no rules apply': answering that would be this product asserting "
            f"something false about your own guidelines.") from None
    except OSError as bad:
        raise ValueError(f"the rulebook at {path} could not be read: {bad}") from None

    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as bad:
        # `yaml.YAMLError` stringifies with the line and column already in it.
        raise ValueError(
            f"the rulebook at {path} is not valid YAML, so no rules were loaded and nothing "
            f"will run under it:\n{bad}\n\nFix the file and start the server again. Running "
            f"with the rules quietly absent is the failure §12.1 exists to remove.") from None

    if loaded is None:
        raise ValueError(f"the rulebook at {path} is empty.")
    if not isinstance(loaded, dict):
        raise ValueError(
            f"the rulebook at {path} must be a mapping with `version` and `rules`, not a "
            f"{type(loaded).__name__}.")
    return loaded


def _checked(loaded: dict, *, path: Path, require_rules: bool = True) -> dict:
    """The shape, refused loudly rather than repaired quietly.

    Every refusal here is a rule that would otherwise have silently not applied — and the
    reader could not have noticed, because the count they would check it against comes from
    the same list that dropped it.
    """
    version = str(loaded.get("version") or "").strip()
    if not version:
        raise ValueError(
            f"the rulebook at {path} has no `version`. Every judgment is stamped with it so "
            f"two judgments made under different rules can be told apart later; a rulebook "
            f"that cannot be named cannot be stamped.")

    rules = loaded.get("rules")
    if rules is None:
        rules = []
    if not isinstance(rules, list):
        raise ValueError(f"the rulebook at {path}: `rules` must be a list, not a "
                         f"{type(rules).__name__}.")
    # An EMPTY rules list is allowed, and the product default is exactly that. The earlier
    # version refused it, reasoning that an empty rulebook is the situation this file replaced
    # — but that situation was rules on file that never APPLIED, which is a different thing
    # from no rules having been written. Refusing it would have forced the product to ship
    # content it has no business owning (§12.3: the customer's rules are a customer file), and
    # `readiness` says plainly when there are none rather than the loader refusing to start.

    seen: dict = {}
    checked = []
    for position, rule in enumerate(rules, 1):
        if not isinstance(rule, dict):
            raise ValueError(f"the rulebook at {path}: rule {position} is a "
                             f"{type(rule).__name__}, not a mapping.")
        missing = [field for field in _REQUIRED if not str(rule.get(field) or "").strip()]
        if missing:
            raise ValueError(
                f"the rulebook at {path}: rule {position} "
                f"({rule.get('id') or 'with no id'}) is missing {', '.join(missing)}. "
                f"Skipping it would mean a rule quietly not applying, which is the failure "
                f"this file exists to remove.")
        severity = str(rule["severity"]).strip()
        if severity not in SEVERITIES:
            raise ValueError(
                f"the rulebook at {path}: rule {rule['id']!r} has severity {severity!r}, "
                f"which is not one of {list(SEVERITIES)}. A severity nothing recognises would "
                f"be dropped when the finding was written.")
        rule_id = str(rule["id"]).strip()
        if rule_id in seen:
            raise ValueError(
                f"the rulebook at {path}: two rules share the id {rule_id!r} (rules "
                f"{seen[rule_id]} and {position}). A finding citing it would name two "
                f"different rules, and nobody reading it could tell which was breached.")
        seen[rule_id] = position
        # D92: the words that would BREACH this rule, so §7.1's sixth computed fact has
        # something to compute against. The product declares none — which words breach a rule
        # is the customer's own judgment, and inventing them would be this product asserting a
        # guardrail nobody wrote.
        watch_for = rule.get("watch_for") or []
        if not isinstance(watch_for, list) or not all(isinstance(w, str) for w in watch_for):
            raise ValueError(
                f"the rulebook at {path}: rule {rule_id!r} has a `watch_for` that is not a "
                f"list of strings. It is the words a brief would contain if it breached this "
                f"rule, and a malformed one would be read as 'nothing to look for'.")
        checked.append({"id": rule_id, "rule": " ".join(str(rule["rule"]).split()),
                        "watch_for": [w.strip().lower() for w in watch_for if w.strip()],
                        "severity": severity, "why": " ".join(str(rule["why"]).split()),
                        # Where this rule came from, carried on the rule itself so a judgment
                        # can say whether a breach was of the product's rule or the
                        # customer's. §12.2 sets it to the overlay's own name.
                        "source": str(rule.get("source") or "").strip() or "product"})

    return {"version": version,
            "describes": " ".join(str(loaded.get("describes") or "").split()),
            "rules": checked,
            "expects": _checked_expectations(loaded, path=path)}


# D50. What a brief is expected to CARRY, as opposed to what a judgment must do. The rubric in
# the review named a KPI workbook; `gaps()` could not report it missing because nothing
# declared it, and a product that invented the expectation itself would be imposing a
# requirement on the customer's behalf.
#
# The product ships NONE of these, deliberately: which inputs a brief must carry is the
# customer's own rule, and §12.2's overlay is where they arrive. The machinery is here so the
# overlay has something to arrive INTO — the alternative is a config file whose most important
# section is unread until a later item wires it up, which is how §11.7's disclosure shipped
# with no call sites.
_EXPECTED_REQUIRED = ("id", "input", "why")


def _checked_expectations(loaded: dict, *, path: Path) -> list:
    """`expects` entries, refused the same way rules are and for the same reason."""
    declared = loaded.get("expects")
    if declared is None:
        return []
    if not isinstance(declared, list):
        raise ValueError(f"the rulebook at {path}: `expects` must be a list, not a "
                         f"{type(declared).__name__}.")
    seen: set = set()
    out = []
    for position, entry in enumerate(declared, 1):
        if not isinstance(entry, dict):
            raise ValueError(f"the rulebook at {path}: expectation {position} is a "
                             f"{type(entry).__name__}, not a mapping.")
        missing = [f for f in _EXPECTED_REQUIRED if not str(entry.get(f) or "").strip()]
        if missing:
            raise ValueError(
                f"the rulebook at {path}: expectation {position} "
                f"({entry.get('id') or 'with no id'}) is missing {', '.join(missing)}.")
        entry_id = str(entry["id"]).strip()
        if entry_id in seen:
            raise ValueError(f"the rulebook at {path}: two expectations share the id "
                             f"{entry_id!r}.")
        seen.add(entry_id)
        # The words that would show it HAD been supplied. Without them the gap can only be
        # reported against every brief forever, which is a gap nobody can close and therefore
        # one everybody learns to ignore — §8.2's own lesson about re-asking a declined
        # question.
        looks_like = entry.get("looks_like") or []
        if not isinstance(looks_like, list) or not all(isinstance(w, str) for w in looks_like):
            raise ValueError(f"the rulebook at {path}: expectation {entry_id!r} has a "
                             f"`looks_like` that is not a list of strings.")
        if not [w for w in looks_like if w.strip()]:
            # REQUIRED, not optional. Without it `gaps()` had no way to tell a brief that
            # carried the input from one that did not, so the expectation was skipped — an
            # expectation that quietly never applies, which is the exact failure this loader
            # exists to refuse, arriving through the field that makes it checkable.
            raise ValueError(
                f"the rulebook at {path}: expectation {entry_id!r} has no `looks_like`. "
                f"Those are the words that would show the input HAD been supplied, and "
                f"without them the gap can be reported forever and never closed — which is a "
                f"gap everybody learns to ignore.")
        out.append({"id": entry_id, "input": " ".join(str(entry["input"]).split()),
                    "why": " ".join(str(entry["why"]).split()),
                    "looks_like": [w.strip().lower() for w in looks_like if w.strip()]})
    return out


@functools.lru_cache(maxsize=1)
def load() -> dict:
    """The rulebook in force, parsed once per process.

    Cached because it is read on every `prepare_evaluation` and the file does not change under
    a running server — and because a parse error must be the same error every time rather than
    an intermittent one depending on who touched the file mid-session. `load.cache_clear()` is
    what tests and a future reload command use.
    """
    path = _bundled()
    return _checked(_read(path), path=path)


def version() -> str:
    """The version stamped onto every judgment made under these rules (§7.6)."""
    return load()["version"]


def rules() -> list:
    """Every rule, in file order. Not ranked, not truncated, not filtered."""
    return list(load()["rules"])


def expects() -> list:
    """What a brief is declared to have to carry (D50). Empty until a customer declares one."""
    return list(load()["expects"])


def by_id(rule_id: str) -> Optional[dict]:
    """One rule, for a finding that cites it (§6.1)."""
    wanted = (rule_id or "").strip()
    return next((rule for rule in load()["rules"] if rule["id"] == wanted), None)


def as_contract() -> str:
    """The rules, as the model reads them when it is about to judge.

    Written out in full every time. The temptation is to summarise or to send only the rules
    that look relevant to this brief — which is the similarity path again, wearing a cheaper
    hat, and would restore exactly the silent, wrong-way-round failure this file removed.
    """
    loaded = load()
    lines = [f"RULEBOOK IN FORCE: {loaded['version']}. These apply to EVERY judgment, "
             f"including this one. A finding that a rule is breached cites its id."]
    for rule in loaded["rules"]:
        lines.append(f"  [{rule['id']}] ({rule['severity']}) {rule['rule']}")
        lines.append(f"      Why: {rule['why']}")
    return "\n".join(lines)


def applied() -> dict:
    """What was applied, for the response and for the record.

    `basis: computed` because it is: the server put these rules in front of the model itself,
    rather than the model reporting that it considered them. §7.8's distinction, on the one
    field where "the model says it used the rulebook" would be worth nothing.
    """
    loaded = load()
    return {
        "version": loaded["version"],
        "rules_applied": len(loaded["rules"]),
        "basis": "computed",
        "what_it_means": (
            f"All {len(loaded['rules'])} rule(s) in rulebook {loaded['version']} were put in "
            f"front of the judgment, in full, regardless of what this brief is about. They "
            f"were not retrieved by similarity and could not be ranked out, truncated, or "
            f"deleted by editing the library."),
    }
