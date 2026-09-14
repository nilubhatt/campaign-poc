"""
§9.7: the fixed calendar this product ships with.

The review: *"seed the table with known fixed events per market — Ramadan, national holidays,
Golden Week, Black Friday, monsoon and rainy seasons, major sporting fixtures — and have
`prepare_evaluation` check every proposed window against it as a computed fact."*

**A shipped calendar is a claim about the world, and most of these claims are approximate.**
Ramadan begins on a sighting and starts a day apart in neighbouring countries. A monsoon has no
start date at all — it has an onset that moves by weeks. Golden Week's substitute working days
are announced each autumn. Shipping any of that as an exact fact would be the confident
unfounded claim this product is written against, so every row here carries how well it is known
and what would settle it, and that note travels with the event wherever it is shown.

**What is NOT approximate is the arithmetic.** Given a window and a date range, whether they
overlap is computed — and that is the half the review means by "a finding no model needs to be
clever to produce". The hedge belongs on the date, not on the overlap.

**A partial calendar is only better than none if it says what it covers.** `COVERS` is the span
these dates are good for and `covered_markets()` is the list they reach; §9.7 reports a window
outside the first as `calendar_expired` and a market outside the second as `not_covered`, rather
than as "nothing was happening". Without that, the commonest output of this feature would be a
clean bill for a market nobody ever wrote a row about — and it would arrive under the heading
telling the model not to re-derive it.

**Scoped to the markets an event is ABOUT.** Ramadan scoped globally told a Polish spring
retail push, as an established fact, that its plan failed to mention Ramadan. A nuisance list is
how a check stops being read, so `global` here means commercially global and nothing looser.

**This is a starting point, not a source of truth.** Every row can be withdrawn (§9.6) by a
customer whose market does not observe it, and every row can be replaced by a corrected one on
upgrade, because `seed_key` is a stable handle rather than a random id. A customer's own
recorded events sit in the same table and reach a proposal by the same arithmetic.

Dates are deliberately few. A large calendar nobody checked is worse than a small one somebody
did: every row below is one a marketer would recognise, and the product says plainly that it is
ours to be wrong about.
"""
from __future__ import annotations

# The span these dates are good for. Past it the shipped calendar has expired and says so —
# otherwise the first day of 2027 is the day this feature starts giving every window on earth a
# silent clean bill, with `absent` promoted into "what the server already established".
COVERS = ("2026-01-01", "2026-12-31")

# `certainty` is the honest half of a shipped date:
#   fixed       — a date on a Gregorian calendar that does not move (Black Friday is defined
#                 relative to US Thanksgiving; Singles' Day is 11/11).
#   announced   — set by a body and published in advance, but changed from year to year.
#   observed    — determined by sighting or local declaration, and genuinely differs by country.
#   seasonal    — a period with no start date at all, only an onset that moves by weeks.
_CERTAINTY_NOTE = {
    "fixed": "This date does not move.",
    "announced": "This is the published date and bodies do change them — check it against the "
                 "official calendar before relying on it.",
    "observed": "This date is determined by sighting or local declaration and differs by "
                "country, often by a day. Treat it as approximate and confirm it for the "
                "market you are planning in.",
    "seasonal": "This is a season rather than a date: the onset moves by weeks from year to "
                "year and by hundreds of miles within a market. Treat the range as indicative "
                "only.",
}

# `markets` is the list an event is ABOUT, including the spellings people actually type —
# `market` is freeform everywhere in this library, so "United States", "USA" and "US" are three
# names for one place and a calendar that knows only the first gives the other two a clean bill.
# `None` means commercially global, and is used sparingly for the same reason.
#
# (seed_key, markets, kind, starts_on, ends_on, certainty, subject, description)
EVENTS = (
    # ── observed: lunar, and a day apart between neighbours ──────────────────
    ("ramadan.2026",
     ("UAE", "United Arab Emirates", "Saudi Arabia", "KSA", "Qatar", "Kuwait", "Bahrain",
      "Oman", "Egypt", "Jordan", "Morocco", "Turkey", "Indonesia", "Malaysia", "Pakistan",
      "Bangladesh", "MENA", "GCC"),
     "fixed_calendar", "2026-02-18", "2026-03-19", "observed", "Ramadan",
     "Ramadan (approximate). Daytime trading, out-of-home and food and beverage patterns "
     "shift substantially; working hours are often shortened."),
    ("eid-al-fitr.2026",
     ("UAE", "United Arab Emirates", "Saudi Arabia", "KSA", "Qatar", "Kuwait", "Bahrain",
      "Oman", "Egypt", "Jordan", "Morocco", "Turkey", "Indonesia", "Malaysia", "Pakistan",
      "Bangladesh", "MENA", "GCC"),
     "fixed_calendar", "2026-03-20", "2026-03-22", "observed", "Eid",
     "Eid al-Fitr (approximate). A major gifting and travel period; many businesses close."),
    ("chinese-new-year.2026", ("China", "Hong Kong", "Taiwan", "Singapore", "PRC"),
     "fixed_calendar", "2026-02-15", "2026-02-23", "announced", "Chinese New Year",
     "Chinese New Year (Year of the Horse), across the published State Council holiday from "
     "New Year's Eve. Factories and logistics stop for roughly a fortnight around it, and the "
     "dates are announced each year."),
    ("diwali.2026", ("India", "Singapore", "Malaysia"),
     "fixed_calendar", "2026-10-25", "2026-11-10", "observed", "Diwali",
     "Diwali (approximate; the day itself is around 8 November 2026). The range given is the "
     "RETAIL window rather than the festival — the largest gifting period of the Indian year, "
     "with buying starting weeks before and the date itself set by local observance. A "
     "one-day row told a campaign running through late October that nothing was on."),

    # ── announced: published, and moved from year to year ────────────────────
    ("golden-week-cn.2026", ("China", "PRC"),
     "fixed_calendar", "2026-10-01", "2026-10-07", "announced", "Golden Week",
     "Golden Week (China, National Day). Travel peaks and the surrounding working days are "
     "rearranged by announcement."),
    ("golden-week-jp.2026", ("Japan",),
     "fixed_calendar", "2026-04-29", "2026-05-06", "announced", "Golden Week",
     "Golden Week (Japan). A run of national holidays; domestic travel peaks. Constitution "
     "Memorial Day falls on a Sunday in 2026, so 6 May is a substitute holiday."),
    ("fifa-world-cup.2026", None,
     "fixed_calendar", "2026-06-11", "2026-07-19", "announced", "World Cup",
     "FIFA World Cup, hosted across the United States, Mexico and Canada. Attention, media "
     "inventory and pricing move sharply in the host markets and in every competing one."),

    # ── fixed: a Gregorian date, or defined relative to one ──────────────────
    ("black-friday.2026",
     ("United States", "USA", "US", "U.S.", "Canada", "United Kingdom", "UK", "Germany"),
     "fixed_calendar", "2026-11-27", "2026-11-30", "fixed", "Black Friday",
     "Black Friday through Cyber Monday. Paid media pricing rises steeply for weeks either "
     "side, well beyond the dates themselves. Adoption is wider than the markets listed here."),
    ("singles-day.2026",
     ("China", "PRC", "Hong Kong", "Taiwan", "Singapore", "Malaysia", "Indonesia", "Thailand",
      "Vietnam", "Philippines", "SEA"),
     "fixed_calendar", "2026-11-11", "2026-11-11", "fixed", "Singles' Day",
     "Singles' Day (11.11). The largest single e-commerce day in China and across South-East "
     "Asia, with a build-up running through late October. Scoped to those markets rather than "
     "globally: a Norwegian retail plan being told it fails to mention 11.11 is the nuisance "
     "that teaches a reader to skip this check."),
    ("buen-fin.2026", ("Mexico", "MX"),
     "fixed_calendar", "2026-11-13", "2026-11-17", "announced", "Buen Fin",
     "El Buen Fin. Mexico's national discount period, announced each year; the 2026 edition "
     "runs five days."),
    ("dia-de-muertos.2026", ("Mexico", "MX"),
     "fixed_calendar", "2026-11-01", "2026-11-02", "fixed", "Día de Muertos",
     "Día de Muertos. A major cultural period in Mexico with strong expectations about how "
     "brands participate."),

    # ── seasonal: an onset, not a date ───────────────────────────────────────
    ("monsoon-in.2026", ("India",),
     "fixed_calendar", "2026-06-01", "2026-09-30", "seasonal", "monsoon",
     "South-west monsoon (indicative). Outdoor activation, footfall and out-of-home "
     "visibility are all affected, and the onset moves by weeks and by region."),
    # TWO rows, because South-East Asia has two opposite rainy seasons and shipping one of
    # them was wrong for most of the region: mainland SEA is wet from roughly May to October
    # and DRY from November to March, which is exactly when the single row used to fire. A
    # Bangkok campaign in December was told it clashed with the rainy season, and one in July
    # was told nothing was on.
    ("rainy-season-mainland-sea.2026",
     ("Thailand", "Vietnam", "Cambodia", "Laos", "Myanmar", "Philippines"),
     "fixed_calendar", "2026-05-15", "2026-10-31", "seasonal", "rainy season",
     "Rainy season across mainland South-East Asia (indicative). Outdoor activations and "
     "footfall are affected; onset and withdrawal move by weeks and differ by country."),
    ("rainy-season-maritime-sea.2026",
     ("Indonesia", "Singapore", "Malaysia", "SEA"),
     "fixed_calendar", "2026-11-01", "2027-03-31", "seasonal", "rainy season",
     "Rainy season across maritime South-East Asia — Indonesia, Singapore and Malaysia's east "
     "coast (indicative). The opposite half of the year from mainland South-East Asia."),
)

# What a plan would actually call each of these, plus the other names it might use. The silence
# test (§9.7's `unaddressed`) is a phrase match, so "we have pre-bought FIFA inventory around
# the fixtures" has to count as addressing the World Cup — a plan discusses a thing in its own
# words, not in the library's. Matched on word boundaries: "Eidos Media" is not Eid.
ALIASES = {
    "Ramadan": ("ramadan", "ramadhan", "ramzan", "ramadan season", "holy month"),
    # "Eid" is three letters and appears inside Heidi, Reid and Eidos — matched on word
    # boundaries for that reason, which is also why the longer forms are listed separately.
    "Eid": ("eid", "eid al-fitr", "eid-al-fitr", "eid ul-fitr", "eid-ul-fitr", "eid alfitr"),
    "Chinese New Year": ("chinese new year", "lunar new year", "spring festival", "cny"),
    "Diwali": ("diwali", "deepavali", "divali"),
    "Golden Week": ("golden week",),
    "World Cup": ("world cup", "worldcup", "world-cup", "fifa", "mundial", "the tournament",
                  "the fixtures"),
    "Black Friday": ("black friday", "cyber monday", "cyber week", "bfcm"),
    "Singles' Day": ("singles' day", "singles day", "11.11", "double 11", "11/11"),
    "Buen Fin": ("buen fin", "el buen fin"),
    "Día de Muertos": ("día de muertos", "dia de muertos", "day of the dead",
                       "día de los muertos", "dia de los muertos"),
    "monsoon": ("monsoon",),
    "rainy season": ("rainy season", "wet season", "monsoon"),
}


def rows() -> list:
    """The seed pack, one row per (event, market) — ready for `store.insert_context_event`.

    One row per market rather than one row with a list, because the overlap query matches a
    single `scope_key` and because a customer withdrawing "Ramadan in Turkey" should not remove
    it from the UAE. The `seed_key` carries the market so each row stays individually
    correctable on upgrade.
    """
    out = []
    for key, markets, kind, starts, ends, certainty, subject, description in EVENTS:
        for scope, value in ([("global", None)] if markets is None
                             else [("market", m) for m in markets]):
            out.append({
                "seed_key": f"{key}@{value or 'global'}",
                "scope": scope, "scope_value": value, "kind": kind,
                "starts_on": starts, "ends_on": ends, "certainty": certainty,
                "subject": subject, "description": description,
            })
    return out


def covered_markets() -> set:
    """Every market name the shipped calendar has a row for, folded for comparison.

    `global` is deliberately not in here: a customer whose only coverage is the World Cup and
    Singles' Day has not had their market checked, and saying otherwise is the clean bill this
    exists to prevent.
    """
    return {m.casefold()
            for _, markets, *_ in EVENTS if markets
            for m in markets}


def note_for(certainty: str) -> str:
    return _CERTAINTY_NOTE.get(certainty or "", "")


def subject_for(seed_key: str) -> str:
    """The handle a plan would use, from a row's key. Keys carry the market suffix."""
    base = (seed_key or "").split("@")[0]
    for key, _markets, _kind, _starts, _ends, _certainty, subject, _description in EVENTS:
        if key == base:
            return subject
    return ""
