"""
§7.1: the findings that do not need a language model.

The review counted them: *"Of the findings produced in two real evaluations today, roughly
two thirds were mechanical: no dated flighting, no engagement rate on any profile, no budget
figure, missing 360 channels, three internal date contradictions, a 'weekend' falling on a
Tuesday. None of that needs a language model, and anything a model decides is something a
model can decide differently tomorrow."*

That last clause is why this is Phase 7's first item rather than a convenience. The subject of
the phase is consistency across users, and the mechanism here is not a better prompt — it is
removing the question. Two thirds of the output stops being generated at all, so two thirds of
the variance has nowhere to come from.

Its own module because these are six independent text checks with no knowledge of campaigns,
evaluations or retrieval, and `core.py` is long enough that burying them there would make them
harder to argue with — which is the one thing a computed fact has to be.

**Three rules hold for all of them.**

*Body layer only* (D11). A reviewer's comment saying "make sure we never mention adidas" would
otherwise register as the brief mentioning adidas: the check reading somebody's warning as the
thing they were warning about. §2.5 built the layer separation and this is its first consumer.

*Evidence attached.* A fact that says "no budget" without showing what it looked at cannot be
argued with, and being checkable by a person is the entire difference between this and a model
saying the same words.

*Absent is a finding; unknown is not.* "This brief carries no dates" is mechanical. "This
brief carries no dates I could parse" is a fact about the parser, and reporting the second as
the first is exactly the confident unfounded claim the review is built around. Every check
that cannot run says so in its own code rather than returning the clean answer.
"""
from __future__ import annotations

import calendar
import datetime
import re
from typing import Optional

MAX_EVIDENCE = 3
_EVIDENCE_CHARS = 160

# ── dates ────────────────────────────────────────────────────────────────────
_MONTHS = {m.lower(): n for n, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): n for n, m in enumerate(calendar.month_abbr) if m})
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
# "3 March 2026", "3rd of March", "March 3", "2026-03-03". Deliberately not a general date
# parser: a wrong date read out of a brief is worse than an unread one, so this matches the
# forms a brief actually uses and leaves the rest to `nothing_to_check`.
_DATE_RE = re.compile(
    rf"""(?P<iso>\b\d{{4}}-\d{{2}}-\d{{2}}\b)
       | (?P<dmy>\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<m>{_MONTH_RE})\b
          (?:\s+(?P<y>\d{{4}}))?)
       | (?P<mdy>\b(?P<m2>{_MONTH_RE})\s+(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?\b
          (?:,?\s+(?P<y2>\d{{4}}))?)""",
    re.IGNORECASE | re.VERBOSE)
_WEEKDAYS = {d.lower(): n for n, d in enumerate(calendar.day_name)}
_WEEKDAY_RE = re.compile(r"\b(" + "|".join(_WEEKDAYS) + r")\b", re.IGNORECASE)
_WEEKEND_WORDS = re.compile(r"\bweekend\b", re.IGNORECASE)

# ── money ────────────────────────────────────────────────────────────────────
# A figure, not a mention. "Budget: TBC" is a brief with no budget in it, and counting the
# word would report the gap as filled by the label naming it.
_MONEY_RE = re.compile(
    r"(?:(?:[$£€]|\b(?:usd|eur|gbp|pen|cop|mxn|myr|idr)\b)\s*\d[\d,.]*\s*(?:k|m|bn)?"
    r"|\b\d[\d,.]*\s*(?:k|m|bn)?\s*(?:[$£€]|\b(?:usd|eur|gbp|pen|cop|mxn|myr|idr)\b))",
    re.IGNORECASE)

# ── creator profiles and engagement ──────────────────────────────────────────
_PROFILE_RE = re.compile(r"@[A-Za-z0-9_.]{2,}")
_RATE_RE = re.compile(r"(?:\b\d[\d.]*\s*%\s*(?:er\b|engagement)|"
                      r"\bengagement(?:\s+rate)?\b[^.\n]{0,20}?\d[\d.]*\s*%|"
                      r"\b\d[\d.]*\s*%\s*(?=[^.\n]{0,20}\bengagement\b))", re.IGNORECASE)

# ── the 360 checklist ────────────────────────────────────────────────────────
# A FIXED list, which is the point: two users judging the same brief cannot disagree about
# which channels it covers. §12.1 will let a customer's rulebook replace it; until then the
# list is the product's, and `channels.checklist` says which one was used so a reader is never
# guessing what "missing" was measured against.
_CHANNELS = {
    "paid_social": (r"paid social", r"\bmeta\b", r"\bfacebook\b", r"\binstagram ads\b",
                    r"\btiktok ads\b", r"\bpaid media\b"),
    "organic_social": (r"organic social", r"\borganic\b", r"brand handles?",
                       r"\bcommunity\b"),
    "influencer": (r"\binfluencers?\b", r"\bcreators?\b", r"\bugc\b", r"\bseeding\b"),
    "email": (r"\bemail\b", r"\bcrm\b", r"\bnewsletter\b"),
    "out_of_home": (r"\booh\b", r"out.of.home", r"\bbillboard", r"\bmetro\b", r"\btransit\b"),
    "retail": (r"\bretail\b", r"in.store", r"\bstore launch\b", r"\bpop.?up\b"),
    "pr": (r"\bpr\b", r"press", r"\beditorial\b", r"\bmedia relations\b"),
    "web": (r"\bwebsite\b", r"\becom", r"\be-commerce\b", r"\blanding page\b", r"\bsite\b"),
}


def _evidence(text: str, at: int, length: int) -> str:
    """The line the check was looking at, trimmed. Enough to argue with, not the whole deck."""
    start = max(0, at - _EVIDENCE_CHARS // 2)
    end = min(len(text), at + length + _EVIDENCE_CHARS // 2)
    return ("… " if start else "") + " ".join(text[start:end].split()) + (" …" if end < len(text) else "")


def _fact(code: str, status: str, what_it_means: str, evidence=None, **extra) -> dict:
    return {"code": code, "status": status, "basis": "computed", "layer": "body",
            "evidence": (evidence or [])[:MAX_EVIDENCE], "what_it_means": what_it_means,
            **extra}


def _parsed_dates(text: str) -> list:
    """Every date the checker recognised, with its span and whether a year was given."""
    out = []
    for m in _DATE_RE.finditer(text):
        try:
            if m.group("iso"):
                value = datetime.date.fromisoformat(m.group("iso"))
                out.append((m, value, True))
            elif m.group("dmy"):
                month = _MONTHS[m.group("m").lower()]
                year = int(m.group("y")) if m.group("y") else None
                out.append((m, datetime.date(year or 2000, month, int(m.group("d"))),
                            year is not None))
            else:
                month = _MONTHS[m.group("m2").lower()]
                year = int(m.group("y2")) if m.group("y2") else None
                out.append((m, datetime.date(year or 2000, month, int(m.group("d2"))),
                            year is not None))
        except (ValueError, KeyError):
            # An unparseable date is not a date the brief got wrong; it is one this checker
            # cannot read, and the two must not be reported the same way.
            continue
    return out


def _date_coverage(text: str) -> dict:
    found = _parsed_dates(text)
    if not found:
        return _fact("date_coverage", "absent",
                     "No date appears anywhere in this brief, so nothing in it can be "
                     "sequenced or held to a deadline.", dates_found=0)
    return _fact("date_coverage", "present",
                 f"{len(found)} date(s) appear in the brief.",
                 evidence=[_evidence(text, m.start(), len(m.group(0))) for m, _, _ in found],
                 dates_found=len(found))


def _date_consistency(text: str) -> dict:
    """Claims inside the brief that the calendar contradicts.

    Two kinds, both from the review's own list: a weekday or "weekend" attached to a date that
    is neither, and a range that ends before it starts. Only dates carrying a YEAR can be
    checked for a weekday — without one the day of the week is unknowable, and guessing a year
    to produce a contradiction would be inventing the finding.
    """
    found = _parsed_dates(text)
    checkable = [(m, d) for m, d, has_year in found if has_year]
    contradictions = []

    for m, value in checkable:
        # The word immediately before the date: "Saturday 3 March 2026", "the weekend of…".
        before = text[max(0, m.start() - 40):m.start()]
        claimed = _WEEKDAY_RE.findall(before)
        actual = calendar.day_name[value.weekday()]
        if claimed and _WEEKDAYS[claimed[-1].lower()] != value.weekday():
            contradictions.append({
                "detail": f"The brief calls this a {claimed[-1]}; "
                          f"{value:%-d %B %Y} was a {actual}.",
                "evidence": _evidence(text, m.start(), len(m.group(0)))})
        elif _WEEKEND_WORDS.search(before) and value.weekday() < 5:
            contradictions.append({
                "detail": f"The brief calls this a weekend; {value:%-d %B %Y} was a {actual}.",
                "evidence": _evidence(text, m.start(), len(m.group(0)))})

    # A range that runs backwards. Only between two dates that both carry a year, and only
    # when the text between them reads like a range.
    for (first, a), (second, b) in zip(checkable, checkable[1:]):
        between = text[first.end():second.start()]
        if re.fullmatch(r"[\s,]*(?:to|until|through|–|—|-)[\s,]*", between) and b < a:
            contradictions.append({
                "detail": f"The range runs backwards: {a:%-d %B %Y} to {b:%-d %B %Y}.",
                "evidence": _evidence(text, first.start(),
                                      second.end() - first.start())})

    if contradictions:
        return _fact("date_consistency", "contradicted",
                     f"{len(contradictions)} statement(s) about dates contradict the "
                     f"calendar.",
                     evidence=[c["evidence"] for c in contradictions],
                     contradictions=contradictions)
    if not checkable:
        # NOT "consistent". Nothing was checked, and a green tick on an unread page is the
        # collapse §6.4 needed three separate codes to avoid.
        return _fact("date_consistency", "nothing_to_check",
                     "No date in this brief carries a year, so no claim about which day it "
                     "falls on can be checked. This is not a clean result — it is an "
                     "unchecked one.")
    return _fact("date_consistency", "consistent",
                 f"The {len(checkable)} dated statement(s) that could be checked agree with "
                 f"the calendar.")


def _budget(text: str) -> dict:
    hits = list(_MONEY_RE.finditer(text))
    if not hits:
        return _fact("budget", "absent",
                     "No money figure appears in this brief. Whether that matters is a "
                     "judgment; that it is absent is not.")
    return _fact("budget", "present", f"{len(hits)} money figure(s) appear in the brief.",
                 evidence=[_evidence(text, m.start(), len(m.group(0))) for m in hits])


def _engagement_rate(text: str) -> dict:
    profiles = list(_PROFILE_RE.finditer(text))
    rates = list(_RATE_RE.finditer(text))
    if not profiles:
        # An absent engagement rate on a brief with no creators is not a gap, it is a
        # category that does not apply — and reporting it would be the complaint nobody can
        # answer that §5.3 wrote out.
        return _fact("engagement_rate", "not_applicable",
                     "No creator profiles appear in this brief, so there are no engagement "
                     "rates to be missing.", profiles_found=0, profiles_with_rate=0)
    # Counted per profile by proximity: a rate belongs to the nearest profile before it, which
    # is how a roster reads on a slide. Not exact, and it does not need to be — the mechanical
    # question is whether the roster carries rates at all.
    with_rate = 0
    for profile in profiles:
        window = text[profile.end():profile.end() + 80]
        if _RATE_RE.search(window):
            with_rate += 1
    if not with_rate:
        return _fact("engagement_rate", "absent",
                     f"{len(profiles)} creator profile(s) appear and none carries an "
                     f"engagement rate, so the roster cannot be judged on performance.",
                     evidence=[_evidence(text, p.start(), len(p.group(0)))
                               for p in profiles],
                     profiles_found=len(profiles), profiles_with_rate=0)
    status = "present" if with_rate == len(profiles) else "partial"
    return _fact("engagement_rate", status,
                 f"{with_rate} of {len(profiles)} creator profile(s) carry an engagement "
                 f"rate.",
                 evidence=[_evidence(text, m.start(), len(m.group(0))) for m in rates],
                 profiles_found=len(profiles), profiles_with_rate=with_rate)


def _channels(text: str) -> dict:
    present, evidence = [], []
    for channel, patterns in _CHANNELS.items():
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                present.append(channel)
                evidence.append(_evidence(text, m.start(), len(m.group(0))))
                break
    missing = [c for c in _CHANNELS if c not in present]
    if not present:
        return _fact("channels", "absent",
                     "No channel from the checklist is named in this brief.",
                     present=[], missing=missing, checklist=list(_CHANNELS))
    status = "present" if not missing else "partial"
    return _fact("channels", status,
                 f"{len(present)} of {len(_CHANNELS)} checklist channels are named"
                 + (f"; missing: {', '.join(missing)}." if missing else "."),
                 evidence=evidence, present=present, missing=missing,
                 checklist=list(_CHANNELS))


def compute(text: Optional[str]) -> dict:
    """Every mechanical check, against BODY text only.

    The caller is responsible for passing body text — see `for_campaign`, which is the only
    thing that knows about layers. Passing a whole record here would defeat D11, so the
    signature takes the text and not the record.
    """
    text = text or ""
    return {fact["code"]: fact for fact in (
        _date_coverage(text),
        _date_consistency(text),
        _budget(text),
        _engagement_rate(text),
        _channels(text),
    )}


def for_campaign(conn, campaign_id: str) -> dict:
    """The same checks against a stored record's BODY layer (D11).

    The commentary is deliberately not read. A reviewer's note saying "make sure we never
    mention a competitor budget of 90,000" would otherwise report a budget of 90,000 — the
    check reading somebody's warning as the thing they were warning about.
    """
    import store

    on_file = store.text_on_file(conn, campaign_id)
    if on_file is None:
        return {}
    return compute("\n\n".join(on_file["body"]))
