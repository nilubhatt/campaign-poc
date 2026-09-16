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
import os
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


OVERLAY_NAME = "rulebook.yaml"


def _no_duplicate_keys():
    """A YAML loader that refuses a repeated mapping key.

    PyYAML keeps the LAST of duplicate keys and says nothing, so an overlay with two `rules:`
    blocks loses the first one entirely — and the customer cannot notice, because the count
    they would check it against comes from the same list that dropped it. Every other way of
    writing this file wrong is refused with a message naming the file; this one defeated all
    of that at the parse step, before any of it ran.
    """
    import yaml

    class Loader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            seen = set()
            for key_node, _ in node.value:
                key = self.construct_object(key_node, deep=deep)
                if key in seen:
                    raise yaml.constructor.ConstructorError(
                        None, None,
                        f"duplicate key {key!r} — the second one silently replaces the first, "
                        f"so everything under the first is lost", key_node.start_mark)
                seen.add(key)
            return super().construct_mapping(node, deep=deep)

    return Loader


_NoDuplicateKeys = _no_duplicate_keys()


def overlay_path() -> Path:
    """Where the CUSTOMER's rulebook lives (§12.2).

    The DATA directory, beside the database — not the install directory beside the product's
    own. The install directory is Program Files on Windows and is replaced wholesale by the
    next installer, so telling a customer to write their rules there would be telling them to
    write in the file an upgrade overwrites. That is a worse trap than not offering the file:
    they would lose work they had been told was safe.

    The data directory is the one this product already promises to keep. It holds the
    database, the install disclosure names it, and nothing in an upgrade touches it.
    """
    return Path(config.DATA_DIR) / OVERLAY_NAME


def _read(path: Path) -> dict:
    """Parse one rulebook file, or say what is wrong with it in terms of that file.

    A YAML error names a line, and that line number is the whole value of the message to
    somebody who has just edited the file — so it is passed through rather than replaced with
    a tidier sentence that loses it.
    """
    import yaml

    if path.is_symlink() and not path.exists():
        # Every other unreadable shape refuses loudly — a directory, a mode-000 file, a bad
        # tag. A symlink whose target has moved returns False from `exists()` and fell into
        # the "no overlay" branch, so a customer whose checkout moved had every judgment
        # stamped as though they had never written any rules, silently.
        raise ValueError(
            f"the rulebook at {path} is a symlink whose target is missing "
            f"({os.readlink(path)}). It is NOT treated as 'no rules': a broken link is a "
            f"rulebook you meant to have, and running without it would stamp every judgment "
            f"as though you had never written one.")

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
        loaded = yaml.load(text, _NoDuplicateKeys)
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


# What a customer may declare, and what they may not. The split is D36's and it is an
# epistemic one rather than a matter of taste.
#
# A STAGE NAME is the customer's vocabulary: an agency that says `shipped` or `in_the_wild` is
# describing their own process, and a product refusing it is a product telling them how to
# talk about their work.
#
# `verified`, `actual`, `reference` are THIS LIBRARY'S CLAIMS ABOUT EVIDENCE. `verified` has a
# hard definition — backed by a `metric_type='actual'` row, enforced on write — and every
# judgment that weighs verified evidence more heavily depends on it. A customer mapping
# "confirmed" onto it would make "the client confirmed it worked" outweigh a measured result,
# silently, in every comparison this product makes.
# `campaign_types` is §12.4/D102's: the review asks for "the checklist for a campaign TYPE"
# and market was standing in for it. It is declared for the same reason markets are — unfolded,
# "Store Launch" and `store_launch` are two checklists, which is C16's failure.
_MAY_DECLARE = ("statuses", "markets", "collections", "tags", "channels", "campaign_types")
_MAY_NOT_DECLARE = {
    "tag_sources": ("`verified` and `stated` are this library's claim about EVIDENCE, not "
                    "words for a stage. `verified` means a metric_type='actual' row exists "
                    "and is enforced on write; mapping another word onto it would make a "
                    "stated impression outweigh a measured result in every comparison this "
                    "product makes."),
    "metric_types": ("`actual`, `predicted` and `target` are what a number IS. Redefining "
                     "them would change what every reconciliation compares against."),
    "record_types": ("`campaign`, `reference` and `stub` are storage classes this product "
                     "reasons about — a `reference` record is excluded from precedent, for "
                     "one. They are not a way of describing your work."),
}


def _checked_vocabulary(loaded: dict, *, path: Path) -> dict:
    """The declared vocabulary, refused loudly where it is not the customer's to declare."""
    declared = loaded.get("vocabulary")
    if declared is None:
        return {key: {} for key in _MAY_DECLARE}
    if not isinstance(declared, dict):
        raise ValueError(f"the rulebook at {path}: `vocabulary` must be a mapping, not a "
                         f"{type(declared).__name__}.")
    out = {key: {} for key in _MAY_DECLARE}
    for key, value in declared.items():
        if key in _MAY_NOT_DECLARE:
            raise ValueError(
                f"the rulebook at {path}: `vocabulary.{key}` cannot be declared. "
                f"{_MAY_NOT_DECLARE[key]}")
        if key not in _MAY_DECLARE:
            # Refused rather than ignored: a typo in a config file is the commonest way a
            # declared rule silently does not apply, and nothing the customer can see would
            # say the key was never read.
            near = _closest(key, _MAY_DECLARE)
            raise ValueError(
                f"the rulebook at {path}: `vocabulary.{key}` is not something this product "
                f"reads. It reads {', '.join(_MAY_DECLARE)}."
                + (f" Did you mean `{near}`?" if near else ""))
        if not isinstance(value, dict):
            raise ValueError(f"the rulebook at {path}: `vocabulary.{key}` must be a mapping, "
                             f"not a {type(value).__name__}.")
        out[key] = _checked_entries(key, value, path=path)
    return out


def _closest(word: str, among) -> Optional[str]:
    import difflib

    near = difflib.get_close_matches(word, list(among), n=1, cutoff=0.6)
    return near[0] if near else None


def _canonical_must_be(key: str):
    """The values a customer may map their words ONTO, for the vocabularies that have a set.

    D36 is "a customer can add spellings for the stages this product has", not "a customer can
    invent stages": every gap check, every reconciliation and every "has this concluded"
    question is written against the three. Accepted at load, an invented stage refused every
    WRITE instead — `ingest_campaign` mapped the customer's word onto it and `insert_campaign`
    then refused, citing a word the caller never sent. A configuration error has to surface
    when the configuration is read.
    """
    import store

    return {"statuses": store.VALID_STATUSES}.get(key)


def _checked_entries(key: str, value: dict, *, path: Path) -> dict:
    """One vocabulary section, normalised into {canonical: {...}}.

    Two shapes are accepted because two are natural: `channels` is a list of words per
    channel, and `markets` carries a region as well. Both end up as a mapping so every reader
    has one shape to handle.
    """
    import store

    allowed = _canonical_must_be(key)
    out = {}
    spellings: dict = {}
    for canonical, detail in value.items():
        name = str(canonical).strip()
        if not name:
            raise ValueError(f"the rulebook at {path}: `vocabulary.{key}` has an empty name.")
        if allowed is not None and name not in allowed:
            raise ValueError(
                f"the rulebook at {path}: `vocabulary.{key}` cannot add {name!r}. These are "
                f"the stages this product reasons about — {', '.join(allowed)} — and every "
                f"gap check and reconciliation is written against them. What you CAN do is "
                f"give one of them your own words: `{allowed[-1]}: ['{name}']`.")
        if isinstance(detail, list):
            detail = {"also": detail}
        if detail is None:
            detail = {}
        if not isinstance(detail, dict):
            raise ValueError(
                f"the rulebook at {path}: `vocabulary.{key}.{name}` must be a list of other "
                f"spellings or a mapping, not a {type(detail).__name__}.")
        also = detail.get("also") or []
        if not isinstance(also, list) or not all(isinstance(w, str) for w in also):
            raise ValueError(f"the rulebook at {path}: `vocabulary.{key}.{name}.also` must "
                             f"be a list of strings.")
        folded = [_folded_word(w) for w in also if str(w).strip()]
        for word in folded:
            if word in spellings and spellings[word] != name:
                # The loader refuses two rules with one id "because nobody reading it could
                # tell which was breached". Two canonicals with one spelling is the same
                # defect, and it resolved DIFFERENTLY in two places — one took the first
                # match, the other built a dict where the last won.
                raise ValueError(
                    f"the rulebook at {path}: `vocabulary.{key}` gives {word!r} to both "
                    f"{spellings[word]!r} and {name!r}. One spelling cannot mean two things, "
                    f"and which one won would depend on where it was read.")
            spellings[word] = name
        if key == "tags":
            # The seven reaction axes decide the `axis` column on the append-only record and
            # drive every quadrant query. `tags` could remap them freely, including inverting
            # them: `not_liked: ['liked']` made a liked campaign read as disliked, and
            # `performed_well: ['client loved it']` put a stated opinion on the PERFORMANCE
            # axis, where it answers "what performed well". That is the reasoning the
            # epistemic guard gives for `tag_sources`, word for word.
            #
            # Only the SPELLINGS are restricted. `not_liked: ['went down badly']` is the whole
            # point of D73, so a canonical that is an axis word stays allowed.
            for word in folded:
                axis = word.replace(" ", "_")
                if axis in store.REACTION_AXES and axis != name:
                    raise ValueError(
                        f"the rulebook at {path}: `vocabulary.tags` gives {word!r} to "
                        f"{name!r}, and {axis!r} is one of this product's own reaction "
                        f"words. Remapping it would change which axis an opinion lands on "
                        f"and could invert it — a liked campaign reading as disliked. Give "
                        f"{name!r} words of your own instead.")
        entry = {"also": folded}
        region = detail.get("region")
        if region is not None:
            if key != "markets":
                raise ValueError(f"the rulebook at {path}: `region` means nothing under "
                                 f"`vocabulary.{key}` — only a market is in a region.")
            entry["region"] = str(region).strip()
        out[name] = entry
    return out


def _folded_word(word: str) -> str:
    """One spelling, for comparison only. Case, spacing and punctuation are not meaning —
    `enums`' first layer, applied to the declared vocabulary so the two agree."""
    import re as _re

    return _re.sub(r"[\s_\-]+", " ", str(word or "").strip()).casefold()


def _checked_corrections(loaded: dict, *, path: Path) -> list:
    """Standing corrections the customer has declared (§12.3/D108).

    A rulebook RULE and a standing CORRECTION are different things and this product keeps them
    apart deliberately: a rule is something the customer wrote down and applies to every brief;
    a correction is something the library watched RECUR until somebody confirmed it, and it
    applies in the markets it was seen in. An agency arriving with this product has both — the
    rules they have always had, and the ten things they find themselves saying on every deck.

    `provenance` is required for the same reason `corrections.note` requires it: without it a
    judgment citing one can say only "the library says so", which is the unfounded confident
    claim this product is built against.
    """
    declared = loaded.get("corrections")
    if declared is None:
        return []
    if not isinstance(declared, list):
        raise ValueError(f"the rulebook at {path}: `corrections` must be a list, not a "
                         f"{type(declared).__name__}.")
    out, seen = [], set()
    for position, entry in enumerate(declared, 1):
        if not isinstance(entry, dict):
            raise ValueError(f"the rulebook at {path}: correction {position} is a "
                             f"{type(entry).__name__}, not a mapping.")
        text = " ".join(str(entry.get("text") or "").split())
        provenance = " ".join(str(entry.get("provenance") or "").split())
        if not text or not provenance:
            raise ValueError(
                f"the rulebook at {path}: correction {position} needs both the rule itself "
                f"(`text`) and where it came from (`provenance`) — the deck and slide, or who "
                f"asked for it. Without provenance a judgment citing it can say only \"the "
                f"library says so\".")
        if text.casefold() in seen:
            raise ValueError(f"the rulebook at {path}: correction {position} repeats one "
                             f"already declared: {text!r}.")
        seen.add(text.casefold())
        markets = entry.get("markets") or []
        if not isinstance(markets, list) or not all(isinstance(m, str) for m in markets):
            raise ValueError(f"the rulebook at {path}: correction {position} has a `markets` "
                             f"that is not a list of strings.")
        # The limits the WRITE enforces, checked here where the file is read. The loader
        # appends "(declared in rulebook X)" to the provenance after this point, so the margin
        # is left for it — without that, a provenance near the limit passed validation and was
        # refused mid-loop by `corrections.note`, with earlier rules already committed and in
        # force and the reply saying nothing about them. Every rerun then failed identically.
        if len(text) > _MAX_CORRECTION_TEXT:
            raise ValueError(
                f"the rulebook at {path}: correction {position} is {len(text)} characters and "
                f"the limit is {_MAX_CORRECTION_TEXT}. A standing correction is a rule "
                f"somebody has to read on every judgment — if it needs a paragraph it is "
                f"probably two rules.")
        if len(provenance) > _MAX_CORRECTION_PROVENANCE:
            raise ValueError(
                f"the rulebook at {path}: correction {position} has {len(provenance)} "
                f"characters of provenance and the limit is {_MAX_CORRECTION_PROVENANCE} "
                f"(the rulebook's own name is added to it when it is loaded). Name the deck "
                f"and the slide, not the conversation.")
        out.append({"text": text, "provenance": provenance,
                    # Empty means everywhere. A correction the library LEARNED is expected in
                    # the markets it was seen in, because that is all the evidence supports; a
                    # correction the customer declares is theirs to scope, and most of them
                    # are house rules that hold everywhere.
                    "markets": [m.strip() for m in markets if m.strip()]})
    return out


def _checked_scorecard(loaded: dict, *, path: Path) -> list:
    """The customer's scorecard criteria (D101).

    §7.4 asks for them in the shared procedure and the product cannot ship them: they are one
    customer's rubric, and hard-coding it into a product that ships generic is what the
    product-owner decision rules out. Declared, they reach the model the same way the rules
    do — in full, on every judgment.
    """
    declared = loaded.get("scorecard")
    if declared is None:
        return []
    if not isinstance(declared, list):
        raise ValueError(f"the rulebook at {path}: `scorecard` must be a list, not a "
                         f"{type(declared).__name__}.")
    out = []
    for position, entry in enumerate(declared, 1):
        if not isinstance(entry, dict):
            raise ValueError(f"the rulebook at {path}: scorecard entry {position} is a "
                             f"{type(entry).__name__}, not a mapping.")
        name = str(entry.get("name") or "").strip()
        asks = " ".join(str(entry.get("asks") or "").split())
        if not name or not asks:
            # A name with no question is a heading. The model would have to invent what
            # "Brand fit" means, which is the variance a shared procedure exists to remove.
            raise ValueError(
                f"the rulebook at {path}: scorecard entry {position} "
                f"({name or 'with no name'}) needs both a `name` and what it `asks`. A "
                f"criterion with no question is a heading, and the model would have to "
                f"invent what it means.")
        out.append({"name": name, "asks": asks})
    return out


def _checked(loaded: dict, *, path: Path, default_source: str = "product") -> dict:
    """The shape, refused loudly rather than repaired quietly.

    Every refusal here is a rule that would otherwise have silently not applied — and the
    reader could not have noticed, because the count they would check it against comes from
    the same list that dropped it.
    """
    raw_version = loaded.get("version")
    if raw_version is not None and not isinstance(raw_version, str):
        # `version: 1.10` is a YAML FLOAT, and stamps `1.1`. A customer bumping 1.1 to 1.10
        # would move their rules under a stamp that did not move — the exact failure this
        # file's own VERSIONS note warns about, produced by writing the version the way
        # versions are normally written.
        raise ValueError(
            f"the rulebook at {path}: `version` must be quoted. {raw_version!r} is a "
            f"{type(raw_version).__name__} to YAML, so `1.10` becomes `1.1` and a version "
            f"bump can leave the stamp unchanged while the rules move. Write "
            f"`version: \"{raw_version}\"`.")
    version = str(raw_version or "").strip()
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
                        # customer's. Defaulted by the FILE it was read from — an overlay's
                        # rules are the overlay's unless they say otherwise — because
                        # defaulting to "product" here made every customer rule claim to be
                        # the product's, which is the one thing this field exists to tell
                        # apart.
                        "source": str(rule.get("source") or "").strip() or default_source})

    return {"version": version,
            "describes": " ".join(str(loaded.get("describes") or "").split()),
            "rules": checked,
            "expects": _checked_expectations(loaded, path=path),
            "vocabulary": _checked_vocabulary(loaded, path=path),
            "scorecard": _checked_scorecard(loaded, path=path),
            # D23: who to contact here. One line of free text: it goes into a remedy a person
            # reads, so validating its shape would be the product having opinions about the
            # customer's own support arrangements.
            "support": " ".join(str(loaded.get("support") or "").split()),
            "corrections": _checked_corrections(loaded, path=path)}


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

# What `corrections.note` will accept, checked where the FILE is read rather than discovered
# halfway through writing. The provenance margin leaves room for the rulebook's own name,
# which the loader appends.
_MAX_CORRECTION_TEXT = 400
_MAX_CORRECTION_PROVENANCE = 240


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


def _layered(product: dict, overlay: dict) -> dict:
    """The customer's rulebook over the product's (§12.2).

    Rules ADD, and a rule whose id the product also uses REPLACES it. Both halves matter.
    Adding is what stops writing one rule from costing you every shipped one — and the
    customer could not notice that loss, because the count they would check it against comes
    from the same list. Replacing is what makes the overlay worth having: an overlay exists so
    a customer can DISAGREE with the product, and one who cannot turn a shipped rule off has
    to work around it instead.

    Replacing is also the only layering that does not create, through the back door, the thing
    a single file already refuses: two rules under one id, where a finding citing it names two
    different rules and nobody reading it can tell which was breached.
    """
    by_id = {rule["id"]: rule for rule in product["rules"]}
    for rule in overlay["rules"]:
        by_id[rule["id"]] = rule
    expects = {entry["id"]: entry for entry in product["expects"]}
    for entry in overlay["expects"]:
        expects[entry["id"]] = entry
    # EVERY section layers, not only the rules. The first version replaced `vocabulary` and
    # `scorecard` wholesale, so an overlay declaring only `tags` erased a product `statuses`
    # declaration and the contract announced "no scorecard is declared" while one sat in the
    # file it had just read — the silent non-application this loader refuses loudly everywhere
    # else. It matters now rather than later: D36's row says `STATUS_SYNONYMS` MOVES INTO the
    # bundled rulebook, and this branch would have thrown it away on arrival.
    #
    # Per key, with the same rule as rules: the customer's declaration of `statuses` replaces
    # the product's `statuses` and leaves `markets` alone.
    vocabulary = {key: dict(product["vocabulary"].get(key) or {})
                  for key in _MAY_DECLARE}
    for key, declared in overlay["vocabulary"].items():
        if declared:
            vocabulary[key] = declared
    return {"rules": list(by_id.values()), "expects": list(expects.values()),
            "vocabulary": vocabulary,
            # A scorecard is a whole rubric rather than a set of independent entries — half
            # the product's criteria and half the customer's is a rubric nobody wrote — so an
            # overlay that declares one replaces it, and one that does not keeps the
            # product's.
            "scorecard": overlay["scorecard"] or product["scorecard"],
            "support": overlay["support"] or product["support"],
            # The customer's, replacing rather than adding to the product's — which ships
            # none, and would have no business shipping somebody's house rules.
            "corrections": overlay["corrections"] or product["corrections"]}


@functools.lru_cache(maxsize=1)
def load() -> dict:
    """The rulebook in force, parsed once per process.

    Cached because it is read on every `prepare_evaluation` and neither file changes under a
    running server — and because a parse error must be the same error every time rather than
    an intermittent one depending on who touched a file mid-session. `load.cache_clear()` is
    what tests and a future reload command use.
    """
    path = _bundled()
    product = _checked(_read(path), path=path)

    overlay_file = overlay_path()
    # THE SAME FILE. On a normal install these are two directories, but the data directory is
    # configurable and a customer who points it at the install directory would otherwise have
    # one file layered with itself: every rule replacing its own twin, and a version stamp
    # reading `acme-3+acme-3`. Nonsense, and silent — the judgment would carry it.
    same = False
    try:
        same = overlay_file.exists() and overlay_file.samefile(path)
    except OSError:
        same = False
    # A dangling symlink is not "no overlay" — see `_read`, which says why. It is checked
    # here as well because `exists()` is what decides whether `_read` is called at all.
    dangling = overlay_file.is_symlink() and not overlay_file.exists()
    if not dangling and (same or not overlay_file.exists()):
        # The product's own declarations, kept. Returning the empty defaults here discarded
        # anything the bundled file declared, which is the same silent loss as above with
        # nobody's overlay involved at all.
        return {**product, "product_version": product["version"], "overlay_version": None}

    # Read once to learn its version, then checked with that version as the default source —
    # a rule in the customer's file is the customer's unless it says otherwise.
    raw = _read(overlay_file)
    overlay = _checked(raw, path=overlay_file,
                       default_source=str(raw.get("version") or "").strip() or "overlay")
    merged = _layered(product, overlay)
    return {
        # ONE scalar carrying both, so every existing reader stays correct. `compare_provenance`
        # diffs `rulebook_version` and concludes two judgments were made "under the same
        # conditions, so an agreement between them is evidence rather than luck" — with the
        # overlay unnamed, two judgments under two different sets of the customer's own rules
        # would both stamp `core-1.0` and be called comparable. That is a false statement in
        # the tool whose whole purpose is explaining disagreement.
        "version": f"{product['version']}+{overlay['version']}",
        "product_version": product["version"],
        "overlay_version": overlay["version"],
        "describes": overlay["describes"] or product["describes"],
        **merged,
    }


def version() -> str:
    """The version stamped onto every judgment made under these rules (§7.6)."""
    return load()["version"]


def rules() -> list:
    """Every rule, in file order. Not ranked, not truncated, not filtered."""
    return list(load()["rules"])


def expects() -> list:
    """What a brief is declared to have to carry (D50). Empty until a customer declares one."""
    return list(load()["expects"])


def overlay() -> Optional[str]:
    """The customer's rulebook version, or None if they have not written one (§12.2)."""
    return load()["overlay_version"]


def vocabulary(key: str) -> dict:
    """One declared section, as {canonical: {"also": [...], ...}}. Empty when undeclared."""
    return load()["vocabulary"].get(key) or {}


def canonical(key: str, value: str) -> Optional[str]:
    """The declared spelling of this value, or None if nothing declares it.

    None rather than the input, deliberately: "the customer calls this LATAM" and "nobody has
    said" are different answers, and a caller that cannot tell them apart would report a
    folded guess as a declared vocabulary.
    """
    wanted = _folded_word(value)
    if not wanted:
        return None
    for name, entry in vocabulary(key).items():
        if wanted == _folded_word(name) or wanted in entry["also"]:
            return name
    return None


def region_of(market: str) -> Optional[str]:
    """Which region a declared market is in (D71).

    D71 asks "whether `region` should feed the market grouping at all, or is a different
    axis". It is a different axis: a region is a GROUPING of markets, so it belongs beside the
    market list rather than in a free-text field that means a continent on one record and a
    country on the next.
    """
    name = canonical("markets", market)
    return (vocabulary("markets").get(name) or {}).get("region") if name else None


def corrections() -> list:
    """Standing corrections the customer declared (§12.3/D108). Empty unless they wrote some."""
    return list(load()["corrections"])


def support() -> Optional[str]:
    """Who IT is, here (D23).

    `notices` tells somebody to "ask whoever installed this", which is the best a product that
    ships to strangers can do — and the customer knows the answer. Declared, a remedy can name
    them, which is the difference between a remedy and a shrug.
    """
    return load()["support"] or None


def scorecard() -> list:
    """The customer's scorecard criteria (D101). Empty until they declare some."""
    return list(load()["scorecard"])


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
    # D101: the customer's scorecard, which §7.4 asks for in the shared procedure and the
    # product cannot ship — it is one customer's rubric. Declared, it reaches the model the
    # same way the rules do: in full, on every judgment, not retrieved by similarity. It is
    # NOT a list of findings to produce; it is what this customer looks at, so a judgment that
    # ignores half of it is answering a different question from the one they asked.
    if loaded["scorecard"]:
        lines.append("THE SCORECARD THIS CUSTOMER JUDGES AGAINST. Cover each one or say why "
                     "it does not apply to this brief; do not invent a score.")
        for entry in loaded["scorecard"]:
            lines.append(f"  {entry['name']}: {entry['asks']}")
    else:
        # The gap, NAMED — "so a reader is not left thinking the step was judged
        # unnecessary". It moved here from `EVALUATION_PROCEDURE` because it is a fact about
        # THIS library rather than a universal statement of procedure: once a customer has
        # declared a scorecard, still saying it is outstanding is the same defect the other
        # way round, and a constant cannot tell the two apart.
        lines.append(
            f"NO SCORECARD IS DECLARED. This customer has not written down the criteria they "
            f"judge against, so judge on the evidence and the rules and do not invent a "
            f"rubric for them. If they ask for one, it goes in {overlay_path()} under "
            f"`scorecard:`, each criterion a `name` and what it `asks`.")
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
        # The parts, readable without splitting the scalar on `+` — which would be a second
        # parser of this product's own field, in every caller that wanted to know whose rules
        # were in force.
        "product": loaded["product_version"],
        "overlay": loaded["overlay_version"],
        "rules_applied": len(loaded["rules"]),
        "scorecard": len(loaded["scorecard"]),
        "basis": "computed",
        "what_it_means": (
            f"All {len(loaded['rules'])} rule(s) in rulebook {loaded['version']} were put in "
            f"front of the judgment, in full, regardless of what this brief is about. They "
            f"were not retrieved by similarity and could not be ranked out, truncated, or "
            f"deleted by editing the library."),
    }
