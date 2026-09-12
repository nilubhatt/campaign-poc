# Deferral tracker

Every piece of work that was consciously **not done** at the point it came up, and where it
went instead. Updated in the same commit as the item that creates the deferral — never
reconstructed afterwards, because a deferral nobody wrote down is indistinguishable from a
thing that was forgotten.

Three kinds of row, and the distinction matters:

- **Deferred** — real work, sequenced into a named later item. It is owed.
- **Decided against** — considered and rejected, with the reason. It is *not* owed, and the
  reason is recorded so it is not silently re-litigated or quietly reintroduced.
- **Accepted limit** — a known boundary of the design. No item will fix it.

`tests/test_deferral_tracker.py` asserts that every item this file points at exists in
`PRODUCT-REVIEW-PLAN.md`, and that nothing points at an item already marked done. A tracker
that drifts from the plan is the same hand-maintained-copy failure this project has now hit
three times (the Windows installer version, `TOOL_NAMES`, and the plan's own counts).

---

## Open — deferred

| # | Deferred | From | To | Why not then |
|---|---|---|---|---|
| D1 | `kind` required on findings; a guardrail breach must cite a `rule_id`; the two classes need genuinely different wording | 2.4 | 6.2 | The field was free to define with no callers; the *behaviour* is 6.2's subject |
| D2 | `approve_if` required on a revise, forbidden on an approve; `fix` required on blocking/should_fix | 2.4 | 6.5 | Same — the column landed early to avoid migrating stored judgments later |
| D3 | Server fills `evidence` (how much precedent a verdict rests on) | 2.4 | 6.6 | It is a server fact by definition; the model must not assert it |
| D4 | Server fills `provenance` (rulebook/model/server versions) | 2.4 | 7.6 | Same |
| D5 | Server stamps `basis: computed` on findings it worked out | 2.4 | 7.8 | §7.1 produces the computed findings; until then only `judged` is writable |
| D6 | `precedent.quote` verified against what was actually retrieved | 2.4 | 6.1 | Impossible while `prepare_evaluation` is stateless — needs 7.2's server-owned retrieval, or a receipt id |
| D7 | A verified quote must also match the chunk's **layer** — otherwise verification would bless a commentary quote as something the deck said | 2.5 | 6.1 | Same dependency as D6 |
| D8 | `closest_precedent` computed by the server rather than asserted by the model | 2.4 | 7.2 | Needs server-owned retrieval |
| D9 | Golden set + agreement measurement; count rule-2 rejections to detect systematic severity downgrading | 2.4 | 7.7 | Nothing to measure against yet |
| D10 | Weighting a client's comment above the deck author's own note | 2.5 | 11.5 | `kind` records the *format* the words arrived in, not their authority — a PDF export turns speaker notes into annotations. Authority order is configured in the rulebook, never inferred |
| D11 | Computed facts (guardrail keyword hits, date coverage) must run on the **body** layer only | 2.5 | 7.1 | Otherwise a comment saying "never mention adidas" registers as the brief mentioning adidas |
| D12 | The layer rule repeated in `prepare_evaluation`'s description and the server `instructions` | 2.5 | 7.4 / 7.5 | It is part of the evaluation procedure, which is those items' subject |
| D13 | `author: unknown` convention — a speaker note records no author and a PDF annotation often has none, so a null means "the file did not say" | 2.5 | 11.6 | 11.6 owns the convention across all records |
| D14 | Per-person view and erasure for `campaign_chunks.source.author` — the first personal data harvested from a file rather than typed by the operator | 2.5 | 11.7 | Deletion today is cascade-only. **Live compliance obligation, not cosmetic** |
| D15 | Decide whether tracked client comments *are* the `approval_notes` schema gap, or whether a separate structured field is still owed | 2.5 | 12.4 | Needs the rulebook's schema decisions |
| D16 | `next_step` strings become `{label, tool, prefilled_args}` | 3.1 | 5.2 | The strings are the draft; 5.2 is the structure |
| D17 | Persist notices per campaign so `gaps()` can report them after the call returns (`images_unreadable`, `commentary_capped` etc. are not reconstructible from counts) | 3.1 | 5.3 | Notices are ephemeral today |
| D18 | Notices raised during `prepare_evaluation` belong in the verdict stamp alongside `embedding_model` | 3.1 | 7.6 | The stamp does not exist yet |
| D19 | `compare_execution` must emit `visual_search_offline` when CLIP is down, not a new code; 9.8's "caveat travels with the metric" should use the notice shape | 3.1 | 9.2 / 9.8 | Those tools do not exist yet |
| D20 | Fold persisted `blocked`/`degraded` notices into the feedback queue — a notice addressed to a person is by definition "needs something from a human" | 3.1 | 10.1 / 10.6 | Depends on D17 |
| D21 | `scope: machine` says "ask whoever installed this"; when the current user *is* the administrator that is the wrong sentence | 3.1 | 11.1 / 11.4 | Needs the role attribution those items introduce |
| D22 | `detail` interpolates exception text and asset paths, which on Windows embed `C:\Users\<name>\` — and `detail` is explicitly "send this to support" text | 3.1 | 11.7 | Disclosure surface; 11.7 owns the position |
| D23 | The customer overlay should name *who IT is here* (a support contact), so a remedy can say "contact X" rather than the generic "ask whoever installed this" | 3.1 | 12.2 | No overlay file yet |

## Open — decided against

| # | Rejected | Raised in | Reason |
|---|---|---|---|
| X1 | Exposing `include_commentary` on `prepare_evaluation` | 2.5 design review | A judgment should weigh every recorded objection. A switch that drops the layer carrying the client's is an easy path to a cleaner-looking approval — 2.4's lesson was that an error message offering an easy exit gets taken, and a parameter is no different |
| X2 | Merging the health-check code vocabulary (`clip_weights_missing`) with the notice vocabulary (`visual_search_offline`) | 3.1 review | They are different axes: a health code names what is *wrong*, a notice code names what the user *loses*. Linked by `cause` instead, so support reading an upload warning and an installer gating on health_check can tell they are one problem |
| X3 | Reordering commentary ahead of body chunks in the embed loop | 2.5 design review | Commentary is the first thing a time budget drops, which is a real cost — but a deck with 200 comments and 10 body chunks would starve the body, and the deck is what the user uploaded. `finish_indexing` recovers either way |
| X4 | Removing `check-weights` now that `health-check` exists | 2.3 | CI runs it on a runner with no Ollama and no database, where a full health check fails on text search. They answer different questions |

## Accepted limits

| # | Limit | Why it stays |
|---|---|---|
| L1 | Commentary and embedded images can only be read when the actual file reaches the server (`asset_ref`) | The `deck_text`-only path never gives us bytes. Inherent, not a bug — surfaced as `commentary_checked: false` so it is never mistaken for "there were none" |
| L2 | A PPTX reviewer comment anchors to `"deck"`, not a slide | The number in `commentN.xml` is the comment part's ordinal, not the slide's. Resolving it needs the package relationships; an anchor that is sometimes silently wrong is worse than one that admits it does not know |
| L3 | Legacy `.ppt` is stored but never text-extracted | No reliable pure-Python extractor. Reported as `legacy_ppt` with the instruction to re-save as `.pptx` |
| L4 | Caps are `len()` on code points, so emoji and combining characters count oddly against the 120/240 limits | The caps exist to stop prose, not to measure typography |

---

## Closed

| # | Was | Closed by |
|---|---|---|
| C1 | Structured warning code for "comments could not be read" | 3.1 — `commentary_unreadable` |
| C2 | The `{code, severity, remedy, detail}` contract lived in one tool's docstring | 3.1 review round — moved to the server `instructions`, so it reaches every tool |
| C3 | `prepare_evaluation` said nothing about a half-indexed library | Closed in the carry-over pass — both surfaces share `_incompleteness_warnings` |
| C4 | Folded warnings read as singular (`count` stored, never spoken) | Same pass — `{count_phrase}` filled by the fold |
| C5 | A legacy `.ppt` emitted two warnings for one condition | Same pass |
| C6 | `version.py` and the Windows installer's `AppVersion` could disagree at release | CI fails a tagged build whose tag does not match `version.py` |
| C7 | `text_search_offline` said "start the local text model service", which names no gesture anyone performs | Caught by the tracker's own staleness test on its first run: it was owed to 4.3, and 4.3 had shipped. The remedy now names re-running the installer, which starts the service, installs the model, and self-tests both |
