"""
§13.7 — is this the same rule?

D109 asks for a semantic fallback so a paraphrase sharing no vocabulary is noticed. §13.4
attempted it and reverted it; §13.7 attempted it again, twice, and the answer is no — with a
measurement the row did not have and which says why rather than only that.

Against `nomic-embed-text`, over three independently-written populations (one mine, two written
by reviewers who had not seen mine):

    ONE RULE, said two ways        0.646 ... 0.972
    NOT one rule, same subject     0.763 ... 0.963

No floor exists in either direction — not to FIND a paraphrase, and not to REFUSE a bad lexical
match, which is the weaker job §13.7 briefly shipped before these numbers arrived. The pairs
that settle it:

    0.963  "…logo in the top left corner…"  /  "…logo in the bottom right corner…"
    0.963  "Seed one colourway per box."    /  "Box one seed per colourway."
    0.751  "Logo top left."  /  "Keep the logo in the top left corner of every asset."

A rule against its own OPPOSITE scores higher than 12 of 14 pairs that genuinely are one rule,
and `negated()` cannot catch it because neither side is a prohibition. One rule's words
shuffled into nonsense scores the same. One rule stated tersely and at length scores below
every false pair above — and holding the long side fixed while varying only the short one gives
0.751, 0.860, 0.864, so the score tracks the short side's LENGTH. The instrument measures
structural and topical similarity; rule identity is neither.

So §13.7's answer is the one its own row names as legitimate: nothing at all. Two things
remain, because the measurement turned them up on the way — a wrong refusal in the matcher that
SHIPS, with no embedder involved, and a sentence the product could never say.

Nothing here tests a cosine. What is tested is the lexical matcher this product actually has.
"""
import core
import corrections
import store


def _campaign(conn):
    return core.ingest_campaign(conn, title="Bogota launch", market="Peru", status="concluded",
                                detail="A store opening.", confirm=True)["campaign_id"]


def test_a_prohibition_stated_without_the_word_not_is_still_a_prohibition(conn):
    """`negated()` listed `not`/`no`/`never` and not `nothing`/`none`/`nobody`, and the cost was
    not a missed refusal but a WRONG one. "Do not schedule anything during Semana Santa." and
    "Nothing is scheduled during Semana Santa." are one rule; with `nothing` absent the second
    read as a PERMISSION, so `resembles` returned 0.0 and the pair was thrown away.

    The guard exists to stop a prohibition being merged with its permission. Here it discarded
    a prohibition stated two ways — the opposite error, in the one check written to prevent it,
    and reachable with no embedder involved at all."""
    prohibition = "Do not schedule anything during Semana Santa."
    same = "Nothing is scheduled during Semana Santa."
    permission = "Semana Santa posts are scheduled a month ahead."

    assert corrections.negated(prohibition)
    assert corrections.negated(same), (
        "“Nothing is scheduled” reads as a permission, so the guard refuses it a hearing"
    )
    assert not corrections.negated(permission)

    assert corrections.resembles(corrections._reading(prohibition),
                                 corrections._reading(same)) > 0.0, (
        "two statements of one prohibition score zero against each other"
    )
    assert corrections.resembles(corrections._reading(same),
                                 corrections._reading(permission)) == 0.0


def test_the_guard_still_keeps_a_prohibition_away_from_its_permission(conn):
    """The words added must not weaken what the guard is for. "Always show the logo" and "Never
    show the logo" share every content word and are opposite instructions."""
    for yes, no in (("Always show the logo on the final frame.",
                     "Never show the logo on the final frame."),
                    ("Always use AI imagery for background plates.",
                     "Nothing in the campaign uses AI imagery.")):
        assert corrections.resembles(corrections._reading(yes),
                                     corrections._reading(no)) == 0.0, (yes, no)


def test_a_prohibition_two_ways_is_offered_for_somebody_to_rule_on(conn):
    """End to end, and the reason the guard's word list matters. Two statements of one
    prohibition now reach the one surface that can settle them — a person's answer — where
    before they were two rules that each stayed at one market and never recurred."""
    cid = _campaign(conn)
    corrections.note(conn, text="Do not schedule anything during Semana Santa.",
                     campaign_id=cid, provenance="Client call, 3 March")

    second = corrections.note(conn, text="Nothing is scheduled during Semana Santa.",
                              campaign_id=cid, provenance="Client call, 9 March")

    looks_like = second["new_correction"]["looks_like"]
    assert looks_like, "the second statement of one prohibition resembles nothing on file"
    assert looks_like["text"] == "Do not schedule anything during Semana Santa."
    assert looks_like["basis"] == "heuristic"
    assert looks_like["how"] == "wording"


def test_the_product_never_claims_to_have_matched_on_meaning(conn):
    """`note`'s sentence carried a clause — "in what it MEANS rather than in its words" — that
    `_looks_like` could never trigger: it returns `wording` unconditionally, and has since
    §13.4's semantic matcher was reverted. A dead branch that describes a FEATURE is worse than
    one that does nothing, because the next reader believes it; and this one advertised exactly
    the capability §13.7 measured as unbuildable on the shipped embedder."""
    cid = _campaign(conn)
    corrections.note(conn, text="Creator captions name the product in the first line.",
                     campaign_id=cid, provenance="Client call, 3 March")

    second = corrections.note(conn, text="Creator captions name the product on the first line.",
                              campaign_id=cid, provenance="Client call, 9 March")

    said = second["new_correction"]["what_it_means"]
    assert "closely resembles one already on file" in said
    assert "MEANS rather than in its words" not in said, (
        "the product claimed a semantic match it has no instrument for"
    )
    assert "meaning" not in {c["how"] for c in [second["new_correction"]["looks_like"]]}


def test_recording_a_rule_reaches_no_model(conn, monkeypatch):
    """§13.7's hard constraint, and the one §13.4 broke by reading "blocked on D99" as
    permission. `note` is a WRITE — somebody is telling the library what a client said — and
    every attempt at a semantic matcher has turned one write into one embed per correction on
    file. Both attempts are gone; this is what must stay true of whatever replaces them."""
    import embedding

    cid = _campaign(conn)
    for n in range(6):
        corrections.note(conn, text=f"A standing rule number {n} about seeding boxes.",
                         campaign_id=cid, provenance="Client call")

    calls = []
    real = embedding.embed
    monkeypatch.setattr(embedding, "embed",
                        lambda text, timeout=None: (calls.append(text),
                                                    real(text, timeout=timeout))[1])

    corrections.note(conn, text="Something else entirely about packaging.",
                     campaign_id=cid, provenance="Client call")
    import feedback

    feedback.queue(conn)

    assert not calls, f"recording one rule and reading the queue reached the embedder {len(calls)} times"


def test_reading_the_queue_does_not_re_read_the_table_per_pair(conn):
    """`_rule_questions`' own docstring records the O(n²) disaster this surface already had —
    "100 corrections 0.37s, 200 corrections 1.74s, 400 corrections 8.39s, for a single upload" —
    and says the fix was ONE pass over the corrections. §13.7's veto put a full table read back
    inside the comparison loop and took 400 corrections to 44 seconds, the entire tool budget,
    inside the function whose comment explains why that was removed. The veto is gone; this is
    the guard that would have caught it."""
    import time

    import feedback

    cid = _campaign(conn)
    for n in range(120):
        corrections.note(conn, text=f"Standing rule {n} about how the seeding boxes are packed.",
                         campaign_id=cid, provenance="Client call")

    started = time.perf_counter()
    feedback._rule_questions(conn)
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, (
        f"pairing 120 corrections took {elapsed:.2f}s — the comparison is re-reading the table"
    )
