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

MAX_EVIDENCE = 8
_EVIDENCE_CHARS = 160

# ── dates ────────────────────────────────────────────────────────────────────
_MONTHS = {m.lower(): n for n, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): n for n, m in enumerate(calendar.month_abbr) if m})
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
# "3 March 2026", "3rd of March", "March 3", "2026-03-03". Deliberately not a general date
# parser: a wrong date read out of a brief is worse than an unread one, so this matches the
# forms a brief actually uses and leaves the rest to `nothing_to_check`.
# `may` is a modal verb before it is a month: "Budget line 3 may be deferred" was read as a
# date. See `_ambiguous` for the two ways a month name does ordinary work in a brief.
_AMBIGUOUS_MONTHS = {"may"}
_MONTH_ABBR = {m.lower() for m in calendar.month_abbr if m}
_DATE_RE = re.compile(
    rf"""(?P<iso>\b\d{{4}}-\d{{2}}-\d{{2}}\b)
       | (?P<dmy>\b(?P<d>\d{{1,2}})(?P<ord>st|nd|rd|th)?\s+(?:of\s+)?(?P<m>{_MONTH_RE})\b
          (?:,?\s+(?P<y>\d{{4}}))?)
       | (?P<mdy>\b(?P<m2>{_MONTH_RE})\s+(?P<d2>\d{{1,2}})(?P<ord2>st|nd|rd|th)?\b
          (?:,?\s+(?P<y2>\d{{4}}))?)""",
    re.IGNORECASE | re.VERBOSE)
# A month and a year with no day: "Go live March 2026". A real date reference, and reporting
# it as "no date appears anywhere in this brief" was the parser-failure-as-absence the module
# docstring forbids, committed by the module.
_MONTH_YEAR_RE = re.compile(rf"\b(?P<m>{_MONTH_RE})\s+(?P<y>\d{{4}})\b", re.IGNORECASE)
_WEEKDAYS = {d.lower(): n for n, d in enumerate(calendar.day_name)}
# Slides abbreviate, and a check that only knows "Tuesday" silently passes "Tue 7 March 2026".
_WEEKDAYS.update({d.lower(): n for n, d in enumerate(calendar.day_abbr)})
_WEEKDAYS.update({"tues": 1, "weds": 2, "thur": 3, "thurs": 4})
_WEEKDAY_RE = re.compile(r"\b(" + "|".join(sorted(_WEEKDAYS, key=len, reverse=True))
                         + r")\b\.?", re.IGNORECASE)
_WEEKEND_WORDS = re.compile(r"\bweekend\b", re.IGNORECASE)

# ── money ────────────────────────────────────────────────────────────────────
# A figure, not a mention. "Budget: TBC" is a brief with no budget in it, and counting the
# word would report the gap as filled by the label naming it.
# Currency codes are UPPERCASE only. Lower-cased, `pen` and `cop` are ordinary words, so
# "12 pen and paper" and "20 cop cars" were money.
_MONEY_RE = re.compile(
    r"(?:(?:[$£€]|\b(?:USD|EUR|GBP|PEN|COP|MXN|MYR|IDR)\b)\s*\d[\d,.]*\s*(?:k|m|bn)?\b"
    r"|\b\d[\d,.]*\s*(?:k|m|bn)?\s*(?:[$£€]|\b(?:USD|EUR|GBP|PEN|COP|MXN|MYR|IDR)\b))")
# A money figure is not a budget. A retail price, a ticket price and "$0 spend on paid" are
# all money and none of them is the thing the review means by "budget detected". The figure
# has to sit near a word that says it is one.
# D94: the languages this library's markets actually use. The amount itself is
# language-neutral — `_MONEY_RE` finds "45.000 EUR" perfectly well — so the ENGLISH WORD was
# the whole of what made this check English-only, and `Presupuesto: 45.000 EUR` was reported
# as no budget at all. The file already carries stopwords for five languages, so the shape
# was established; this is the same list for the word that matters.
_BUDGET_WORDS = re.compile(
    r"\b(?:budget|spend|investment|fee|cost|funding|media\s+spend|working\s+media"
    # es / pt
    r"|presupuesto|inversi[oó]n|gasto|coste|costo|or[cç]amento|investimento|verba"
    # fr
    r"|budget|co[uû]t|investissement|d[eé]penses?"
    # id / ms
    r"|anggaran|biaya|belanja)\b",
    re.IGNORECASE)
_BUDGET_WINDOW = 60

# ── creator profiles and engagement ──────────────────────────────────────────
# A handle, not an email address. `maria@brand.com` was two creator profiles, and almost every
# brief carries a contact address — so the commonest real brief got an authoritative "none of
# your creators has an engagement rate" about people who do not exist.
_PROFILE_RE = re.compile(r"(?<![\w.@])@(?![\w.]*@)[A-Za-z0-9_.]{2,}(?<![.])")
_RATE_RE = re.compile(r"(?:\b\d[\d.]*\s*%\s*(?:er\b|engagement)|"
                      r"\bengagement(?:\s+rate)?\b[^.\n]{0,20}?\d[\d.]*\s*%|"
                      r"\b\d[\d.]*\s*%\s*(?=[^.\n]{0,20}\bengagement\b))", re.IGNORECASE)

# ── the 360 checklist ────────────────────────────────────────────────────────
# A FIXED list, which is the point: two users judging the same brief cannot disagree about
# which channels it covers. §12.1 will let a customer's rulebook replace it; until then the
# list is the product's, and `channels.checklist` says which one was used so a reader is never
# guessing what "missing" was measured against.
_CHANNELS = {
    "paid_social": (r"paid social", r"\bMeta\b(?! description)", r"\bfacebook ads?\b",
                    r"\binstagram ads?\b", r"\btiktok ads?\b", r"\bpaid media\b"),
    "organic_social": (r"organic social", r"organic (?:posts?|content|channels?|feed)",
                       r"brand handles?", r"community (?:management|posts?|content)"),
    "influencer": (r"\binfluencers?\b", r"\bcreators?\b", r"\bugc\b", r"\bseeding\b"),
    "email": (r"\bemail\b", r"\bcrm\b", r"\bnewsletter\b"),
    "out_of_home": (r"\bOOH\b", r"out.of.home", r"\bbillboards?\b",
                    r"metro (?:stations?|network|takeover)", r"transit (?:media|ads?)"),
    "retail": (r"\bretail\b", r"in.store", r"\bstore launch\b", r"\bpop.?up\b"),
    "pr": (r"\bPR\b(?![-/.\w])", r"\bpress (?:release|office|coverage|kit|day)\b",
           r"\beditorial\b", r"\bmedia relations\b"),
    "web": (r"\bwebsite\b", r"\becomm?erce\b", r"\be-commerce\b", r"\blanding pages?\b",
            r"\bon.site\b(?! activation)", r"\bdotcom\b"),
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
        if not m.group("iso") and _ambiguous(m, text):
            continue
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
    # A month and a year is a date reference. "Go live March 2026" reported "no date appears
    # anywhere in this brief", which is the parser-failure-as-absence the docstring forbids.
    spans = {(m.start(), m.end()) for m, _, _ in found}
    month_years = [m for m in _MONTH_YEAR_RE.finditer(text)
                   if not any(a <= m.start() < b for a, b in spans)]
    found = found + [(m, None, True) for m in month_years]
    if not found:
        return _fact("date_coverage", "absent",
                     "No date appears anywhere in this brief, so nothing in it can be "
                     "sequenced or held to a deadline.", dates_found=0)
    return _fact("date_coverage", "present",
                 f"{len(found)} date(s) appear in the brief.",
                 evidence=[_evidence(text, m.start(), len(m.group(0))) for m, _, _ in found],
                 dates_found=len(found))


def _ambiguous(match, text: str) -> bool:
    """Is this a month name doing ordinary work rather than naming a date?

    Two ways it happens, both from review, both on ordinary brief prose:

    "Budget line 3 may be deferred" — `may` is a modal verb, and `\bmay\b` after a number
    looks exactly like a date. Capitalisation is the signal a brief actually carries: a month
    is written "May" and the verb is not. A year or an ordinal settles it either way.

    "Order 24 Dec-branded hoodies", "Reference 15 Jan-Feb split" — a month abbreviation
    hyphenated into a compound is naming a product line, not a day.
    """
    name = match.group("m") or match.group("m2") or ""
    after = text[match.end():match.end() + 2]
    if name.lower() in _MONTH_ABBR and after[:1] == "-" and after[1:2].isalpha():
        return True
    if name.lower() not in _AMBIGUOUS_MONTHS:
        return False
    if match.group("y") or match.group("y2") or match.group("ord") or match.group("ord2"):
        return False
    return not name[:1].isupper()


# Only these may sit between a weekday and the date it names. Anything else and the weekday
# belongs to a different clause.
_CLAIM_FILLER = re.compile(r"^[\s,]*(?:the\s+)?(?:of\s+)?(?:week\s+(?:commencing|of)\s+)?$",
                           re.IGNORECASE)
_CLAIM_WINDOW = 40


def _claim_window(text: str, date, previous) -> str:
    """The text that can legitimately be claiming a weekday for THIS date.

    A fixed 40-character lookback read across neighbouring dates and sentence ends, so
    "Saturday 7 March 2026 to 12 March 2026" — a correct brief — reported that the brief calls
    12 March a Saturday. It produced the very finding this item was sold on, out of nothing,
    with the server's authority behind it.

    So the window stops at the previous date, and at a sentence boundary, and the weekday has
    to be the last thing in it apart from filler like "the" or "of".
    """
    start = max(0, date.start() - _CLAIM_WINDOW)
    if previous is not None:
        # Defensive, and worth saying that no test can currently kill it: the filler rule
        # below already refuses any weekday that is not the last token before the date, which
        # covers every case a previous date creates. Kept because the two rules answer
        # different questions — this one says which text belongs to this date, that one says
        # whether the text is a claim — and a change to either should not silently widen the
        # other. Review's mutation of this line survives the suite; that is accurate.
        start = max(start, previous.end())
    window = text[start:date.start()]
    for boundary in (".", ";", "\n", " and ", " then ", " to ", " until ", " through "):
        cut = window.rfind(boundary)
        if cut >= 0:
            window = window[cut + len(boundary):]
    claimed = _WEEKDAY_RE.search(window)
    if claimed and not _CLAIM_FILLER.match(window[claimed.end():]):
        return ""
    return window


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

    for n, (m, value) in enumerate(checkable):
        before = _claim_window(text, m, checkable[n - 1][0] if n else None)
        claimed = _WEEKDAY_RE.search(before)
        actual = calendar.day_name[value.weekday()]
        if claimed:
            named = claimed.group(1)
            if _WEEKDAYS[named.lower()] != value.weekday():
                contradictions.append({
                    "detail": f"The brief calls this a {named}; "
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
    """A budget, not any money figure.

    A retail price, a ticket price and a unit cost are all money, and none of them is what the
    review means by "budget detected". `RRP $49.99 per unit` was reported as a budget, so the
    model was told the brief carries one and told not to contradict it. The figure has to sit
    near a word that says it is a budget.
    """
    money = list(_MONEY_RE.finditer(text))
    hits = [m for m in money
            if _BUDGET_WORDS.search(text[max(0, m.start() - _BUDGET_WINDOW):
                                         m.end() + _BUDGET_WINDOW])]
    if not hits:
        return _fact("budget", "absent",
                     "No budget figure appears in this brief"
                     + (f" ({len(money)} money figure(s) appear, none of them near a word "
                        f"like budget, spend or investment)." if money else "."),
                     money_figures=len(money))
    return _fact("budget", "present", f"{len(hits)} budget figure(s) appear in the brief.",
                 evidence=[_evidence(text, m.start(), len(m.group(0))) for m in hits],
                 money_figures=len(money))


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
    for n, profile in enumerate(profiles):
        # Bounded by the NEXT handle. A fixed 80 characters ran straight past it, so one
        # creator was credited with another's rate and a roster with one measured profile
        # read as fully rated.
        end = profiles[n + 1].start() if n + 1 < len(profiles) else len(text)
        window = text[profile.end():min(end, profile.end() + 80)]
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


# "No paid social", "we will not use influencers", "email is out of scope". A channel a brief
# rules OUT is not a channel it covers, and reporting it as present inverts the finding.
_NEGATION = re.compile(r"\b(?:no|not|without|excluding|minus)\b[^.\n]{0,20}$", re.IGNORECASE)
_NEGATED_AFTER = re.compile(r"^[^.\n]{0,30}?\b(?:is|are)\s+(?:out of scope|excluded|"
                            r"not (?:in scope|included|planned))\b", re.IGNORECASE)


def _named_here(text: str, m) -> bool:
    """Is this mention a plan, or a statement that the channel is not being used?"""
    before = text[max(0, m.start() - 40):m.start()]
    after = text[m.end():m.end() + 60]
    return not (_NEGATION.search(before) or _NEGATED_AFTER.match(after))


def _channels(text: str) -> dict:
    present, evidence, ruled_out = [], {}, []
    for channel, patterns in _CHANNELS.items():
        for pattern in patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                if _named_here(text, m):
                    present.append(channel)
                    # Keyed by channel: a flat list capped at three left five of eight
                    # channels with no way to see which snippet justified them.
                    evidence[channel] = _evidence(text, m.start(), len(m.group(0)))
                    break
                if channel not in ruled_out:
                    ruled_out.append(channel)
            if channel in present:
                break
    missing = [c for c in _CHANNELS if c not in present]
    ruled_out = [c for c in ruled_out if c not in present]
    if not present:
        return _fact("channels", "absent",
                     "No channel from the checklist is named in this brief.",
                     present=[], missing=missing, ruled_out=ruled_out,
                     checklist=list(_CHANNELS))
    status = "present" if not missing else "partial"
    return _fact("channels", status,
                 # "named", not "covered": the check reads words, and whether a channel that
                 # is named is actually planned is a judgment. Saying "covered" would be the
                 # server claiming something it did not establish.
                 f"{len(present)} of {len(_CHANNELS)} checklist channels are NAMED in the "
                 f"brief (named, not necessarily planned)"
                 + (f"; not named: {', '.join(missing)}." if missing else "."),
                 evidence=list(evidence.values()), evidence_by_channel=evidence,
                 present=present, missing=missing, ruled_out=ruled_out,
                 checklist=list(_CHANNELS))


# D94: the commonest words in the languages this library's markets actually use. Stopwords,
# because content words are the ones a marketing deck borrows from English — "brief",
# "engagement", "launch" and every channel name appear untranslated in a Spanish deck, so
# matching on those would report English for half of LATAM.
_STOPWORDS_BY_LANGUAGE = {
    "en": {"the", "and", "for", "with", "this", "that", "from", "will", "have", "are",
           "our", "their", "over", "into", "each", "than", "been", "were"},
    "es": {"de", "la", "el", "en", "los", "las", "una", "por", "con", "para", "del", "que",
           "se", "su", "al", "lo", "como", "más", "pero", "sus"},
    "pt": {"de", "da", "do", "em", "os", "as", "uma", "por", "com", "para", "que", "se",
           "sua", "ao", "mais", "mas", "seus", "nas", "nos", "pelo"},
    "fr": {"le", "la", "les", "des", "une", "pour", "avec", "dans", "que", "sur", "pas",
           "par", "plus", "sont", "ont", "leur", "aux", "ce"},
    "id": {"yang", "dan", "untuk", "dengan", "dari", "ini", "itu", "akan", "pada", "ke",
           "di", "adalah", "atau", "juga", "tidak", "sudah"},
}

# Below this many words there is no sample to judge, and guessing would suppress the checks on
# an English brief — reporting `unchecked` where the answer was sitting in the text, which is
# the same defect pointing the other way.
_ENOUGH_TO_TELL = 12

# How far ahead of English another language has to be before the checks stand down. A deck is
# a mix — a Spanish brief carries English channel names, an English one carries a market name
# — so a narrow lead is not a language, it is noise.
_CLEARLY_AHEAD = 2


def language_of(text: Optional[str]) -> dict:
    """Which language this reads as, and whether the English-only checks apply (D94).

    A HEURISTIC and it says so. This is stopword counting, not a language model, and a
    confident `language: es` would be a computed-looking claim resting on counting little
    words — which is the kind of claim this whole product exists to stop making. D61 settled
    the vocabulary: a threshold deciding something is `heuristic`, never `judged`.

    When it cannot tell, the checks RUN. They are English-only, so an unknown sample is more
    likely English than not, and a false `unchecked` hides a real finding — which is worse
    than the alternative, because a finding nobody sees cannot be argued with either.
    """
    words = [w for w in re.findall(r"[^\W\d_]+", (text or "").lower()) if len(w) > 1]
    if len(words) < _ENOUGH_TO_TELL:
        # Too little to judge — and `reads_as_english` is TRUE here, deliberately. This
        # product's users write English; a six-word snippet with no budget in it genuinely has
        # no budget, and reporting that as `unchecked` would turn every short brief into a
        # shrug. The suppression is for POSITIVE evidence of another language, or for enough
        # text to expect English words and none appearing — not for the absence of a sample.
        return {
            "code": "language", "status": "checked",
            "language": "unknown", "checks_apply": True, "reads_as_english": True,
            "basis": "heuristic",
            "evidence": [], "counts": {}, "what_it_means": (
                "Too little text to tell what language this is, so the checks ran and their "
                "results are reported as they stand. A sample this short is more likely "
                "English than not, and a false “could not check” on every short brief "
                "would hide findings that were there."),
        }
    counts = {code: sum(1 for w in words if w in stop)
              for code, stop in _STOPWORDS_BY_LANGUAGE.items()}
    english = counts.get("en", 0)
    best = max((c for code, c in counts.items() if code != "en"), default=0)
    winner = next((code for code, c in counts.items() if c == best and code != "en"), None)
    if best >= max(_CLEARLY_AHEAD, english * _CLEARLY_AHEAD) and winner:
        return {
            "code": "language", "status": "checked",
            "language": winner, "checks_apply": False, "reads_as_english": False,
            "basis": "heuristic",
            "evidence": [], "counts": counts, "what_it_means": (
                f"This reads as {winner!r} rather than English — {best} common {winner} words "
                f"against {english} English ones. The mechanical checks only read English, so "
                f"they did NOT run: what they would have reported as absent is unchecked, "
                f"which is a different thing. Read the brief yourself for the facts they "
                f"would have established."),
        }
    if not english:
        # No English words in a text long enough to have some. That is evidence AGAINST
        # English, not evidence for it — and falling through to "en" here is how Vietnamese
        # and Thai, the SEA markets D94 names explicitly, were read as English and told their
        # decks carried no budget. There is no stopword list for them and there does not need
        # to be: the absence of English is the whole of what this has to establish.
        return {
            "code": "language", "status": "checked", "evidence": [],
            "language": "unknown", "checks_apply": True, "reads_as_english": False,
            "basis": "heuristic", "counts": counts,
            "what_it_means": (
                "No common English words appear in this brief, and none of the other "
                "languages this recognises is clearly ahead either — so it is probably not "
                "English and this cannot say what it is. Anything the checks FOUND is still "
                "reported; anything they did not is reported as unchecked rather than as a "
                "finding that the brief lacks it."),
        }
    return {
        "code": "language", "status": "checked", "evidence": [],
        "language": "en", "checks_apply": True, "reads_as_english": True,
        "basis": "heuristic", "counts": counts,
        "what_it_means": "This reads as English, so the mechanical checks ran normally.",
    }


def _unchecked(code: str, language: dict) -> dict:
    """A check that did not run, said as itself (D94).

    `absent` on a Spanish brief is "we could not look" wearing "we looked and found nothing",
    with the server's authority attached — and §9.6 spent a whole round on exactly that
    distinction. `Presupuesto: 45.000 €` is a budget, and the library was telling marketers
    their decks had none.
    """
    return _fact(code, "unchecked", (
        f"Not checked: this brief does not appear to be in English "
        f"(reads as {language['language']!r}), and this check only reads English. That is "
        f"NOT a finding that the brief lacks it — read the brief for this fact."))


def compute(text: Optional[str]) -> dict:
    """Every mechanical check, against BODY text only.

    The caller is responsible for passing body text — see `for_campaign`, which is the only
    thing that knows about layers. Passing a whole record here would defeat D11, so the
    signature takes the text and not the record.
    """
    text = text or ""
    # D94: what language this is, and therefore whether the checks below mean anything. Every
    # one of them is an English regex, so on a Spanish deck they reported `absent` — "we could
    # not look" wearing "we looked and found nothing", with the server's authority attached.
    # `Presupuesto: 45.000 €` is a budget, and the library was telling marketers they had none.
    language = language_of(text)
    found = {fact["code"]: fact for fact in (
        _date_coverage(text),
        _date_consistency(text),
        _budget(text),
        _engagement_rate(text),
        _channels(text),
    )}
    # D94, and the asymmetry is the whole design: **finding something is trustworthy in any
    # language; finding nothing is only trustworthy if we could read it.**
    #
    # The first version suppressed every check on a non-English brief, which was wrong twice
    # over. It stood down checks whose matchers are language-NEUTRAL — the amount in
    # "Presupuesto: 45.000 EUR" is found by a currency regex, and the English word beside it
    # was the whole of what made that check English-only. And it made every check hostage to
    # a stopword count that misfires on bullet decks, which have almost no stopwords and are
    # the commonest deck shape there is: two market names ("Los Angeles", "Las Vegas")
    # outvoted zero English words and silenced a perfectly readable English brief.
    #
    # Under this rule a wrong guess costs at most a real `absent` reading `unchecked` —
    # conservative, and never the other way, which is the direction that invents findings.
    if not language["reads_as_english"]:
        for code, fact in found.items():
            if fact["status"] in ("absent", "nothing_to_check"):
                found[code] = _unchecked(code, language)
    return {"language": language, **found}


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
    computed = compute("\n\n".join(on_file["body"]))
    # §9.6/§9.7: a window somebody ENTERED is a date this record carries, and a stronger one
    # than any sentence — it is structured, and §9.7 names it twice in the adjacent line of
    # the same block. Reporting "no date appears anywhere in this brief" beside it faulted the
    # record for something the record does not lack, in the one place the model is told not to
    # re-derive what it reads.
    record = store.get_campaign(conn, campaign_id) or {}
    if computed.get("date_coverage", {}).get("status") == "absent" and record.get("starts_on"):
        computed["date_coverage"] = _fact(
            "date_coverage", "present",
            f"This record carries no date in its TEXT, but its window is on file as "
            f"{record['starts_on']} to {record.get('ends_on') or 'open-ended'}, which is what "
            f"the calendar check and every context event are matched against.",
            evidence=[f"{record['starts_on']} to {record.get('ends_on') or 'open-ended'}"],
            dates_found=0, from_window=True)
    return computed
