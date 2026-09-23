"""
Who said it, and how much that claim is worth (§11.1, §11.2).

Nine write paths in this product take a person's name as a free string — `said_by`,
`recorded_by`, `stated_by`, `withdrawn_by`, `confirmed_by`, `linked_by` — and the server has
never known what it was receiving. D105 put it plainly about one of them: *"nothing
distinguishes a real confirmation from one the model wrote itself."*

That matters because of what those fields do. `confirmed_by` promotes a rule to the checklist
where it is applied to every future brief in its markets. `said_by` turns an impression into a
`stated` tag that later judgments weigh. `withdrawn_by` takes a finding off the record. Each
is a claim that a person decided something, recorded with identical confidence whether a
marketer typed their own name or a model supplied a plausible one.

## The distinction

The server knows exactly one thing for certain: **which account made the call.** It knows
nothing whatever about **whose opinion is being recorded.** Those are different facts, and
this product already has the vocabulary for that difference — so §11.2's instruction is to
reuse it rather than invent a second scheme:

    captured_by    the account. Derived here, never accepted from a caller. `verified`, with
                   a `method` that says how strong that is.
    on_behalf_of   whose judgment this is. Only a person can say, so it is `stated` and it is
                   REQUIRED. Deriving it would be the product inventing the one fact it
                   cannot know — and inventing it as the operator, who is usually the person
                   typing rather than the person whose judgment is being recorded.

**`verified` here is a narrow claim and the `method` is what keeps it narrow.** `stdio_local`
means "this is the desktop account the server runs as". Nobody authenticated; no identity
provider was consulted. Reporting that as `verified` with nothing beside it would inflate a
fact about a process into a fact about a person, which is the overstatement this whole review
was written against.

D35 is explicit that `enums.TAG_SOURCE_SYNONYMS` must not be reused for this. That table
accepts "measured" and "from metrics", which are meaningful about a performance tag and
meaningless about a person — and a synonym table that accepts them would let
`source: measured` through on an identity claim.
"""
from __future__ import annotations

import contextlib
import getpass
import os
import re
import socket
from typing import Optional

import auth
import config

# §11.2, and deliberately only two. An identity claim is either something this server
# established or something it was told; there is no third state, and D35 is why this is not
# `enums.TAG_SOURCE_SYNONYMS` — "measured" is a sentence about a number, not about a person.
AUTHOR_SOURCES = ("verified", "stated")

# How the account was established. The item names three; the fourth is the one it did not
# foresee and the one that matters most for a shared deployment.
METHODS = (
    # A local stdio server: the account is the desktop login the process runs as. True, and
    # weak — see the module docstring.
    "stdio_local",
    # An identity provider actually vouched for this caller.
    "sso",
    # A script or workbook load. Not a person at a keyboard, and a row that reads as one is
    # how a bulk import becomes evidence that somebody confirmed something.
    "import",
    # Reachable over the network with no auth configured. NOTHING is known about the caller,
    # and saying so is the only honest answer.
    "unattributed",
)

# §9.8's guard, promoted to every path. A model writing "the system" into a field that
# promotes a rule to the checklist is the library confirming its own suggestion and recording
# that a person did.
# The library wearing a person's clothes.
_IS_THE_PRODUCT = ("server", "system", "computed", "campaign-poc", "campaign intelligence",
                   "library", "automatic", "auto", "claude", "the model", "assistant")

# PLACEHOLDERS. §11.3's menu offers "2 = Someone else — who?" and `said_by` is a free string,
# so the model has to turn the number into a name itself — and the menu's own option LABEL was
# accepted verbatim as a person, along with "a colleague" and "the client". Those are names
# `person_on_file` can never find and `pseudonymise_person` can never erase: an opinion filed
# under one is outside every promise §11.7 makes.
#
# Kept apart from the product names because they are DIFFERENT MISTAKES and the fix differs.
# "Claude" is the library confirming its own suggestion. "the team" is a model relaying a real
# human answer it could not turn into a name, and telling that caller it is "naming itself"
# describes nothing that happened and suggests nothing to do about it.
_IS_A_PLACEHOLDER = ("n/a", "unknown", "anonymous", "nobody", "everyone",
                     "someone else", "somebody else", "a colleague", "the colleague",
                     "the client", "a client", "the team", "the agency", "they", "them",
                     "who?", "tbc", "tbd")

_NOT_A_PERSON = _IS_THE_PRODUCT + _IS_A_PLACEHOLDER

_IMPORTING = False


@contextlib.contextmanager
def importing():
    """Mark everything written inside as machine-driven rather than typed (§11.2).

    A workbook load writes hundreds of rows and not one of them is a person deciding
    something. Without this the bulk path produces identical attribution to a marketer
    answering a question, and "somebody confirmed this" becomes a thing a script can assert.
    """
    global _IMPORTING
    was = _IMPORTING
    _IMPORTING = True
    try:
        yield
    finally:
        _IMPORTING = was


def _over_http() -> bool:
    """Whether this process is serving the network rather than one desktop.

    The distinction decides whether `os_user` means anything. On a local stdio server the OS
    account IS the user; on a shared HTTP deployment it is whoever launched the service, and
    reporting it per-caller would file nine people's decisions under one name.
    """
    return bool(getattr(config, "SERVING_HTTP", False))


def _principal_for_this_call():
    """The authenticated caller, or anonymous. A seam, so `auth` stays the only thing that
    decides what counts as authenticated."""
    return getattr(auth, "CURRENT_PRINCIPAL", None) or auth.ANONYMOUS


def _os_user() -> Optional[str]:
    # `getpass.getuser` consults several environment variables before falling back to the
    # password database, and raises on a system where none of them answer. An attribution
    # that cannot be derived is absent, never guessed.
    try:
        return getpass.getuser() or None
    except Exception:                          # noqa: BLE001
        return os.environ.get("USER") or os.environ.get("USERNAME") or None


def _host() -> Optional[str]:
    try:
        return socket.gethostname() or None
    except Exception:                          # noqa: BLE001
        return None


# What each method actually establishes, keyed on the value that IS stored. §11.2 says in
# writing that reporting `verified` "with nothing beside it would inflate a fact about a
# process into a fact about a person" — and then the sentence lived only in the live dict
# `captured_by()` returns, which is never persisted. So the override log, the surface whose
# whole purpose is "where somebody argued and won", showed `source: verified` by a named
# person with the caveat stripped off. Keyed on the stored column, so it cannot be stripped
# again.
METHOD_MEANS = {
    "stdio_local": (
        "The desktop account this server is running as. That is not an authentication — "
        "nobody proved who they are, and anyone with access to this machine writes under "
        "this account."),
    "sso": (
        "An identity provider vouched for this account. This is a real authentication of the "
        "ACCOUNT — it still says nothing about whose judgment the content is, which is what "
        "`on_behalf_of` is for."),
    "import": (
        "Written by a bulk load rather than by somebody answering a question. No account was "
        "acting: nothing here says a person decided anything."),
    "unattributed": (
        "This server is reachable over the network and no identity provider is configured, so "
        "nothing at all is known about who made this call. The OS account is deliberately not "
        "recorded: it is whoever started the service, not whoever is using it."),
}


def captured_by() -> dict:
    """The account that made this call, as strong a claim as the evidence supports (§11.1)."""
    if _IMPORTING:
        return {
            "method": "import", "source": "stated",
            "what_it_means": (
                "Written by a bulk load rather than by somebody answering a question. No "
                "account is recorded because none was acting: this is not an authentication, "
                "and nothing here says a person decided anything."),
        }

    principal = _principal_for_this_call()
    if not principal.anonymous:
        who = {
            "method": "sso", "source": "verified", "subject": principal.subject,
            "what_it_means": (
                f"An identity provider vouched for {principal.subject}. This is a real "
                f"authentication of the ACCOUNT — it still says nothing about whose judgment "
                f"the content is, which is what `on_behalf_of` is for."),
        }
        if principal.email:
            who["email"] = principal.email
        return who

    if _over_http():
        # The failure this branch exists for: a shared server reporting the OS account it was
        # STARTED as for every caller, so nine people's decisions are filed under whoever
        # launched the service.
        return {
            "method": "unattributed", "source": "stated",
            "what_it_means": (
                "This server is reachable over the network and no identity provider is "
                "configured, so nothing at all is known about who made this call. The OS "
                "account is deliberately not recorded: it is whoever started the service, "
                "not whoever is using it."),
        }

    who = {"method": "stdio_local", "source": "verified",
           "what_it_means": (
               "The desktop account this server is running as. That is not an "
               "authentication — nobody proved who they are, and anyone with access to this "
               "machine writes under this account.")}
    user, host = _os_user(), _host()
    if user:
        who["os_user"] = user
    if host:
        who["host"] = host
    # Only when the operator SET one. Title-casing `nbhatt` into "Nbhatt" would make a field
    # that sometimes holds a real name and sometimes a prettied-up login, which is a field
    # nobody can read.
    display = (getattr(config, "OPERATOR_NAME", "") or "").strip()
    if display:
        who["display_name"] = display
    return who


def channel() -> str:
    """How this call arrived (§11.4). One of `stdio`, `http`, `import`."""
    if _IMPORTING:
        return "import"
    return "http" if _over_http() else "stdio"


def session_id() -> Optional[str]:
    """The run this call belongs to (§11.4), or None.

    Per PROCESS, which is what a session is for a stdio server: one Claude Desktop
    conversation starts one server and ends it. Not persisted anywhere and not derived from
    anything about the user — it exists so a reader can tell "these eleven decisions were all
    made in one sitting" from "these eleven were made over a month", which is the difference
    between somebody working through a queue and somebody agreeing with everything.
    """
    return _SESSION


_SESSION = f"run_{os.getpid()}_{int(__import__('time').time())}"


# Words that carry no identity of their own, so a value made only of these and disqualifying
# terms names nobody. Separate from `_NOT_A_PERSON` because "the" is not a claim about the
# product — it is what makes "the system" a phrase instead of two.
_FILLER = ("the", "a", "an", "our", "my", "this", "that", "of", "for", "at", "by", "and")

# Whole words, not substrings. Anchored on `\w` rather than `\b` so "campaign-poc" and "n/a"
# match across their punctuation.
_NOT_A_PERSON_PATTERN = re.compile(
    "|".join(rf"(?<!\w){re.escape(term)}(?!\w)"
             # Longest first, so "the model" is struck out whole rather than leaving "model"
             # behind after "the" goes.
             for term in sorted(_NOT_A_PERSON, key=len, reverse=True)))


def reads_as_the_product(name: str) -> bool:
    """Whether this name is the product wearing a person's clothes.

    Strike out every word that means "not a person" and every filler word. If nothing is
    left, nothing named anybody.

    **It used to be a substring scan**, and that refused real people: "Themba Nkosi" for
    containing "them", "Automne" for "auto", "Claudette" for "claude", plus Sautoy, Lautoka,
    Matthey, Serverin, Anthem and Autolycus. That is not a safe failure — it is a hard block
    on recording somebody's view, it falls hardest on names that are not Anglo, and the
    remedy the refusal offers ("give a spelling nobody else uses") does not exist for a
    person's own name.

    Striking out rather than matching the whole value is what keeps "Claude Monet" a person
    while "Claude" is not, and "the system admin" a person while "the system" is not. The
    residue is the test: a name is what is left when the words that name nobody are gone.
    """
    folded = " ".join((name or "").lower().split())
    if not folded:
        return False   # empty is "nobody said", which `person` refuses with its own sentence
    residue = _NOT_A_PERSON_PATTERN.sub(" ", folded)
    left = [word for word in re.split(r"[^\w']+", residue) if word]
    return all(word in _FILLER for word in left)


def _names_the_product(name: str) -> bool:
    """Which of the two mistakes this is — the library, or a placeholder for a person."""
    folded = " ".join((name or "").lower().split())
    return any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", folded)
               for term in _IS_THE_PRODUCT)


def person(name: str, *, field: str = "on_behalf_of") -> str:
    """A name that can stand as whose judgment something is, or a ValueError saying why not."""
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValueError(
            f"`{field}` is required and must name a PERSON: whose judgment this is, which is "
            f"the one thing this server cannot work out for itself. The account that made the "
            f"call is recorded separately and automatically — that is a different fact, and "
            f"filling this in from it would file every decision under whoever was typing.")
    # A one-character name is not one. `feedback` had this rule and the other four paths did
    # not, which is the same weakest-door arrangement as the product-name check: whichever
    # entry point a model reaches decides what the library will believe about who decided
    # something.
    if len(cleaned) < 2:
        raise ValueError(
            f"`{field}` is {cleaned!r}, which is not a name. This records whose judgment "
            f"something is; an initial nobody can resolve is the same as nothing.")
    if reads_as_the_product(cleaned):
        raise ValueError(
            f"`{field}` must be a PERSON, not {cleaned!r}. " + (
                "This records whose judgment something is, and the library naming itself "
                "there is it confirming its own suggestion and recording that somebody did."
                if _names_the_product(cleaned) else
                "This records whose judgment something is, and nothing can find, weigh or "
                "erase a view held by that. Ask WHO holds it and use their name — or leave "
                "the field out, which records honestly that nobody said."))
    return cleaned


def who_said_it(*, on_behalf_of: str, field: str = "on_behalf_of") -> dict:
    """Both halves of authorship, each carrying what it is worth (§11.2)."""
    return {
        "captured_by": captured_by(),
        "on_behalf_of": {"name": person(on_behalf_of, field=field), "source": "stated"},
    }
