"""
Forgiving enums, teaching errors (§5.1, idea A).

    "Three enum values were rejected before the right one was found: `client_stated` for a
    tag source (valid: verified, stated), `target` for a metric type (valid: actual,
    predicted), and a bare string where a list was required. Each rejection said what was
    wrong, none said what was right.

    Fix: every enum error returns the valid set and the closest match. Better still,
    normalise on the way in — 'client stated', 'stated by client' and 'client_stated' all
    mean stated. The intended user is a marketer in a conversation, not an engineer reading
    a schema."

Three layers, in order, because they are three different claims:

1. **Shape.** Case, spacing and punctuation are not meaning. "In Flight", "in-flight" and
   "in_flight" are one value, and a synonym table that has to list all three is a table
   nobody keeps complete.
2. **Synonyms.** Different words for the same thing, listed explicitly. These are judgments
   about vocabulary and belong somewhere a person can read and argue with them.
3. **Teach.** Anything else is rejected with the valid set and — only when something really
   is close — the closest match. A suggestion that is not close is worse than none: it sends
   somebody to retype a word that will be rejected too.

What this deliberately does NOT do is guess between values that mean different things.
`target` was the reviewer's own example, and it is worth following what happened to it. A
target is what somebody wants to happen and a prediction is what this library expects to
happen; filing one as the other corrupts every reconciliation, which exists to compare what
was predicted against what occurred. So it was refused, with the distinction spelled out and
the advice to put the number in the campaign's prose.

§8.1/D33 then built the place it belonged — a target is a `metric_type` now, comparable
against the actual — and the refusal outlived it. It kept telling people to put a number into
freeform text for two more items, which is the drift this file exists to prevent, arriving in
the file itself. `target` normalises now; the DISTINCTION it was refused for is unchanged and
is why it is its own value rather than a synonym of `predicted`.
"""
from __future__ import annotations

import difflib
import re
from typing import Iterable, Optional

# How close a suggestion has to be before offering it is help rather than noise. Measured
# rather than guessed: real typos in this project's vocabularies score 0.82 and up
# ("predicated"/"predicted" 0.95, "in_flite"/"in_flight" 0.82), while the coincidences score
# below 0.7 — "approved"/"proposed" 0.63, which means something else entirely and would have
# been suggested at 0.6. ("cancelled"/"concluded" 0.67 was the other example, and §12.4/D38
# turned it into a real status: the word was being refused because there was nowhere to put
# it, and the nearest valid value was a different fact about the campaign.)
_SUGGESTION_CUTOFF = 0.75

# Prefixes that make a word the DENIAL of what follows. difflib rates "unverified" at 0.89
# against "verified" — the highest-scoring suggestion in the whole vocabulary, and the most
# harmful one possible: a caller retrying with it marks an unverified claim as measured
# evidence, on the single field this library weighs judgments by. A suggestion that inverts
# the input is never help.
_NEGATIONS = ("un", "non", "not_", "no_", "never_", "dis")

_SHAPE = re.compile(r"[\s\-]+")


def canonical_shape(value: str) -> str:
    """Lower-cased, trimmed, with spaces and hyphens as underscores. Not meaning — form."""
    return _SHAPE.sub("_", value.strip().lower()).strip("_")


# ── what is forgiving, and what is not ─────────────────────────────────────
#
# Only the vocabularies a MARKETER authors are forgiving: record_type, status, tag source,
# metric_type. Those words arrive from a person in conversation, or from a spreadsheet column
# header, and rejecting "live" because the column is called `in_flight` is the defect this
# item exists to fix.
#
# The vocabularies the MODEL authors stay strict: verdict, severity, finding kind, basis,
# precedent layer. A `Literal` retry costs the model nothing, and §2.4's lesson was that any
# easy exit offered in an error message gets taken — auto-mapping "minor" to `note` would be
# handing it a severity downgrade path, which is precisely what the golden set in §7.7 exists
# to detect.
#
# The tables below are the PRODUCT's, and they stay. §12.2 lets a customer ADD to the status
# table from their own rulebook — `store._declared("statuses")` is merged into the synonyms
# passed here — because a stage name is customer vocabulary and an agency that says "shipped"
# is describing their own process. What they may not do is invent a stage: the three canonical
# values are what every gap check and reconciliation is written against, so the rulebook
# loader refuses a mapping onto anything else.
#
# The others stay product-owned and non-overridable, and the rulebook refuses them by name:
# `verified`/`stated` and `actual`/`predicted` are this library's EPISTEMICS, and a customer
# redefining "confirmed" as measured would corrupt every judgment that weighs verified
# evidence more heavily.

# ── vocabulary. Explicit, so a person can read and disagree with it ─────────

TAG_SOURCE_SYNONYMS = {
    # The reviewer's own rejection, and everything anyone would type meaning the same.
    "client_stated": "stated",
    "stated_by_client": "stated",
    "client_said": "stated",
    "self_reported": "stated",
    "reported": "stated",
    "claimed": "stated",
    "anecdotal": "stated",
    "impression": "stated",
    # The other side: a claim with measurement behind it.
    # Measurement words only. `verified` has a hard definition in this library — backed by
    # a metric_type='actual' row, enforced on write — so "confirmed" does NOT belong here:
    # "the client confirmed it worked" is a stated claim. The write side would catch that
    # when no metrics exist; the filter side has no such guard, so a query for "confirmed"
    # would have silently narrowed to measured evidence.
    "measured": "verified",
    "from_metrics": "verified",
    "backed_by_data": "verified",
    "data_backed": "verified",
}

STATUS_SYNONYMS = {
    "draft": "proposed",
    "idea": "proposed",
    "pitch": "proposed",
    "a_pitch": "proposed",
    "proposal": "proposed",
    "planned": "proposed",
    "upcoming": "proposed",
    # D37: `in_market` is NOT here. It is one agency's phrasing, and a product that ships
    # generic does not carry one customer's vocabulary in its own tables — the rulebook
    # decision rules that out in as many words. It moved to `docs/example-rulebook.yaml`,
    # where an agency that says it will find it in the file they are copying from.
    "live": "in_flight",
    "running": "in_flight",
    "active": "in_flight",
    "ongoing": "in_flight",
    # §12.4/D38. `on_hold` is `paused` in somebody else's words, not a fifth state.
    "on_hold": "paused",
    "holding": "paused",
    "parked": "paused",
    "suspended": "paused",
    "killed": "cancelled",
    "dropped": "cancelled",
    "pulled": "cancelled",
    "shelved": "cancelled",
    "done": "concluded",
    "finished": "concluded",
    "complete": "concluded",
    "completed": "concluded",
    "ended": "concluded",
    # NOT "past": temporal, not lifecycle. A cancelled campaign is also past, and filing it
    # as concluded puts it into every later "what worked" query as though it had run.
}

# §9.1. What a marketer types for a photograph that came back.
ASSET_PHASE_SYNONYMS = {
    "brief": "proposed", "briefed": "proposed", "concept": "proposed",
    "mockup": "proposed", "mock_up": "proposed", "render": "proposed",
    "planned": "proposed", "intended": "proposed",
    "as_delivered": "delivered", "shot": "delivered", "photo": "delivered",
    "photos": "delivered", "final": "delivered", "actual": "delivered",
    "built": "delivered", "executed": "delivered", "on_site": "delivered",
    "post": "delivered", "after": "delivered",
}

RECORD_TYPE_SYNONYMS = {
    "guidelines": "reference",
    "rulebook": "reference",
    "brand_guidelines": "reference",
    "placeholder": "stub",
}

METRIC_TYPE_SYNONYMS = {
    "result": "actual",
    "results": "actual",
    "outcome": "actual",
    "outcomes": "actual",
    "measured": "actual",
    "forecast": "predicted",
    "projection": "predicted",
    "estimate": "predicted",
    "expected": "predicted",
    # §8.1/D33 made `target` a value of its own, so these now normalise to it rather than
    # being refused. They are NOT synonyms of `predicted`, and that distinction is the reason
    # the refusal existed: a prediction is what this library expects to happen and is what
    # reconciliation scores itself against, while a target is what somebody wanted. Filing one
    # as the other scores the library against an ambition.
    "goal": "target",
    "objective": "target",
    "kpi_target": "target",
    "aim": "target",
}

# Values that look like a near-miss but mean something else, with the distinction spelled
# out. Guessing these is how a library quietly fills up with claims nobody made.
_EXPLAIN = {
    # `target` is a value now (§8.1/D33), not a near-miss. The explanation that used to live
    # here sent people to put the number in `detail` "where freeform text belongs" — advice
    # that was right when there was nowhere else for it and wrong the moment D33 built the
    # place. It is kept below as a DISTINCTION rather than a refusal, because the reason it was
    # refused is still true: a target is not a prediction, and weighing one as the other scores
    # the library against somebody's ambition.
    # §12.4/D38: `cancelled` and `paused` ARE statuses now, so `killed` and `on_hold` are
    # near-misses for them rather than for nothing. The four entries that stood here told a
    # caller "this library has no status for a campaign that was called off — leave the status
    # unset", and an unset status is read as `concluded`: the advice steered a cancelled
    # campaign into the bucket it was written to keep it out of. A near-miss table that
    # outlives the values it describes is worse than none, because it is confident.
    ("status", "killed"): (
        "'cancelled' is the word this library uses for a campaign that was called off. It is "
        "a real status — it does NOT count as having run, so it is never a missing result."
    ),
    ("status", "on_hold"): (
        "'paused' is the word this library uses for a campaign stopped for now and expected "
        "to resume. It is a real status and is neither concluded nor cancelled."
    ),
    ("metric_type", "benchmark"): (
        "A benchmark is somebody else's number. This field records this campaign's own "
        "figures: its 'actual' results, its 'predicted' ones, or the 'target' it was aiming "
        "at. Somebody else's number belongs in the campaign's detail, as context."
    ),
}


def normalise(value: Optional[str], *, field: str, valid: Iterable[str],
              synonyms: Optional[dict] = None, allow_none: bool = True) -> Optional[str]:
    """Return the canonical value, or raise a ValueError that teaches.

    `field` is the PARAMETER's name. The message is read by Claude before it is read by
    anybody else, and Claude has to know which argument to correct — a prose paraphrase
    ("record type") reads better and tells it less.
    """
    valid = tuple(valid)
    if value is None:
        if allow_none:
            return None
        raise BadValue(f"{field} is required; it is one of {list(valid)}",
                       field=field, given="", valid=valid)
    if not isinstance(value, str):
        raise BadValue(f"{field} must be one of {list(valid)}, got "
                       f"{type(value).__name__}", field=field, given=str(value),
                       valid=valid)

    shaped = canonical_shape(value)
    if not shaped:
        if allow_none:
            return None
        raise BadValue(f"{field} is required; it is one of {list(valid)}",
                       field=field, given="", valid=valid)

    by_shape = {canonical_shape(v): v for v in valid}
    if shaped in by_shape:
        return by_shape[shaped]

    resolved = (synonyms or {}).get(shaped)
    if resolved is not None:
        return resolved

    raise _teach(value, shaped, field, valid)


class BadValue(ValueError):
    """A rejected enum value, carrying the retry as data rather than only as a sentence.

    Still a ValueError, because that is the only exception type the tool layer converts into
    something the caller can read — anything else becomes "Error executing tool X" with the
    detail discarded. But it carries `valid` and `suggestion` as attributes, so the surface
    can offer the retry (§5.2) instead of asking somebody to parse "Did you mean...?" out of
    prose (tracker D31).
    """

    def __init__(self, message: str, *, field: str, given: str, valid: tuple,
                 suggestion=None):
        super().__init__(message)
        self.field = field
        self.given = given
        self.valid = list(valid)
        self.suggestion = suggestion


def _teach(given: str, shaped: str, field: str, valid: tuple) -> BadValue:
    """The valid set always; the closest match only when something really is close; and the
    distinction spelled out for the near-misses that mean something else."""
    explanation = _EXPLAIN.get((field, shaped))
    message = f"{field} must be one of {list(valid)}, got {given!r}."
    if explanation:
        # A near-miss that means something else. No suggestion: the whole point is that the
        # obvious-looking value is the wrong one.
        return BadValue(f"{message} {explanation}", field=field, given=given, valid=valid)

    suggestion = None
    close = difflib.get_close_matches(shaped, [canonical_shape(v) for v in valid],
                                      n=1, cutoff=_SUGGESTION_CUTOFF)
    if close and not _is_denial_of(shaped, close[0]):
        suggestion = {canonical_shape(v): v for v in valid}[close[0]]
        message = f"{message} Did you mean {suggestion!r}?"
    return BadValue(message, field=field, given=given, valid=valid, suggestion=suggestion)


def _is_denial_of(shaped: str, candidate: str) -> bool:
    """True when the input is the candidate with a negation in front of it."""
    for prefix in _NEGATIONS:
        stripped = shaped[len(prefix):].lstrip("_-") if shaped.startswith(prefix) else None
        if stripped == candidate:
            return True
    return False
