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
`target` is the reviewer's own example: a target is what somebody wants to happen and a
prediction is what this library expects to happen. Filing one as the other corrupts every
later reconciliation, which exists to compare what was predicted against what occurred. So
that one teaches rather than normalises, and says why.
"""
from __future__ import annotations

import difflib
import re
from typing import Iterable, Optional

# How close a suggestion has to be before offering it is help rather than noise.
_SUGGESTION_CUTOFF = 0.6

_SHAPE = re.compile(r"[\s\-]+")


def canonical_shape(value: str) -> str:
    """Lower-cased, trimmed, with spaces and hyphens as underscores. Not meaning — form."""
    return _SHAPE.sub("_", value.strip().lower()).strip("_")


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
    "measured": "verified",
    "from_metrics": "verified",
    "backed_by_data": "verified",
    "data_backed": "verified",
    "confirmed": "verified",
}

STATUS_SYNONYMS = {
    "draft": "proposed",
    "idea": "proposed",
    "pitch": "proposed",
    "a_pitch": "proposed",
    "proposal": "proposed",
    "planned": "proposed",
    "upcoming": "proposed",
    "live": "in_flight",
    "running": "in_flight",
    "active": "in_flight",
    "in_market": "in_flight",
    "ongoing": "in_flight",
    "done": "concluded",
    "finished": "concluded",
    "complete": "concluded",
    "completed": "concluded",
    "ended": "concluded",
    "past": "concluded",
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
    # NOTE: "target" is deliberately absent. See _EXPLAIN below and the module docstring.
}

# Values that look like a near-miss but mean something else, with the distinction spelled
# out. Guessing these is how a library quietly fills up with claims nobody made.
_EXPLAIN = {
    ("metric_type", "target"): (
        "A target is what you want to happen; 'predicted' is what this library expects to "
        "happen, and reconciliation later compares predictions against actuals. Recording a "
        "target as a prediction would score the library against somebody's ambition. If it "
        "is a goal, put it in the campaign's detail; if it is a forecast, use 'predicted'."
    ),
    ("metric_type", "goal"): (
        "A goal is what you want to happen; 'predicted' is what this library expects to "
        "happen. Put the goal in the campaign's detail rather than in a metric."
    ),
    ("metric_type", "benchmark"): (
        "A benchmark is somebody else's number. This field records this campaign's own "
        "'actual' results or its 'predicted' ones."
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
        raise ValueError(f"{field} is required; it is one of {list(valid)}")
    if not isinstance(value, str):
        raise ValueError(f"{field} must be one of {list(valid)}, got "
                         f"{type(value).__name__}")

    shaped = canonical_shape(value)
    if not shaped:
        if allow_none:
            return None
        raise ValueError(f"{field} is required; it is one of {list(valid)}")

    by_shape = {canonical_shape(v): v for v in valid}
    if shaped in by_shape:
        return by_shape[shaped]

    resolved = (synonyms or {}).get(shaped)
    if resolved is not None:
        return resolved

    raise ValueError(_teach(value, shaped, field, valid))


def _teach(given: str, shaped: str, field: str, valid: tuple) -> str:
    """The valid set always; the closest match only when something really is close; and the
    distinction spelled out for the near-misses that mean something else."""
    explanation = _EXPLAIN.get((field, shaped))
    message = f"{field} must be one of {list(valid)}, got {given!r}."
    if explanation:
        return f"{message} {explanation}"
    close = difflib.get_close_matches(shaped, [canonical_shape(v) for v in valid],
                                      n=1, cutoff=_SUGGESTION_CUTOFF)
    if close:
        suggestion = {canonical_shape(v): v for v in valid}[close[0]]
        return f"{message} Did you mean {suggestion!r}?"
    return message
