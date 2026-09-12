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
| D18 | Notices raised during `prepare_evaluation` belong in the verdict stamp alongside `embedding_model` | 3.1 | 7.6 | The stamp does not exist yet |
| D19 | `compare_execution` must emit `visual_search_offline` when CLIP is down, not a new code; 9.8's "caveat travels with the metric" should use the notice shape | 3.1 | 9.2 / 9.8 | Those tools do not exist yet |
| D20 | Fold persisted `blocked`/`degraded` notices into the feedback queue — a notice addressed to a person is by definition "needs something from a human" | 3.1 | 10.1 / 10.6 | Needs notices persisted per record; D17's half of that closed as C14, the rest is this row |
| D21 | `scope: machine` says "ask whoever installed this"; when the current user *is* the administrator that is the wrong sentence | 3.1 | 11.1 / 11.4 | Needs the role attribution those items introduce |
| D22 | `detail` interpolates exception text and asset paths, which on Windows embed `C:\Users\<name>\` — and `detail` is explicitly "send this to support" text | 3.1 | 11.7 | Disclosure surface; 11.7 owns the position |
| D25 | Run the Windows installer end to end in CI — silently against a good bundle and a deliberately broken one — rather than only compiling it | 4.1 | 4.4 | The `[Code]` path has still never been executed anywhere; compiling it is new but is not the same thing |
| D26 | The install is a defined moment with a screen on all three platforms; the personal-data disclosure belongs there (a wizard page *before* install on Windows) | 4.1 | 11.7 | No disclosure text exists yet |
| D27 | `health_check` should gain a `rulebook` component, and the install gate should name it — the bundled rulebook is a shipped payload like the weights | 4.1 | 12.1 | No rulebook exists yet |
| D28 | The customer overlay must live in `DATA_DIR`, not the app directory: both Unix installers delete and replace the app directory on every run, and `--purge` must name it | 4.1 | 12.2 | Same |
| D29 | Pin the Ollama model by digest rather than by name — CLIP is hash-pinned, but any 768-dimension model currently satisfies the text check | 4.3 | 7.6 | Embedding identity is 7.6's subject |
| D30 | `.bak` of `claude_desktop_config.json` is a second copy of other connectors' environment (possibly tokens), rewritten on every run, with no retention position | 3.2 | 11.7 | Disclosure and retention are 11.7's |
| D32 | Gloss every marketer-facing enum value with its synonyms in the tool docstrings (`in_flight — live, running, in market`) so the model maps *meaning* before the server maps shape, and say in `instructions` that normalised values are echoed under `normalised` and should be spoken | 5.1 | 7.4 | The docstrings are the shared prompt, which is 7.4's subject |
| D33 | A target has no structured home — `detail` is freeform and can never be reconciled, yet "did we hit our number" is the comparison a marketer most wants. Add `target` as a metric type or registry attribute, then replace the refusal text | 5.1 | 8.1 | Needs the metric registry to exist |
| D34 | `enums.normalise` gains a provisional mode (or 8.2 composes `canonical_shape` + `_teach`) so an unknown metric key is accepted provisionally instead of raising — the same machinery, a different third layer | 5.1 | 8.2 | Two normalisers for one problem is how they drift apart |
| D35 | Author `verified`/`stated` (§11.2) needs its own synonym table — `TAG_SOURCE_SYNONYMS` must NOT be reused, since "measured" and "from metrics" are meaningless for an identity claim | 5.1 | 11.2 | The vocabulary is shared; the meaning is not |
| D36 | `STATUS_SYNONYMS` moves into the bundled rulebook with a customer overlay; the epistemic vocabularies (`metric_type`, tag source, `record_type`) stay product-owned and non-overridable | 5.1 | 12.1 | A stage name is customer vocabulary; "verified" is this library's claim about evidence |
| D37 | "in market" is a product default today and should move to the Fabletics example overlay once D36 lands | 5.1 | 12.3 | Depends on D36 |
| D38 | No lifecycle status for cancelled / paused / on hold — today they are refused, with the error explaining there is no home | 5.1 | 12.4 | Adding a status is a schema decision, and 12.4 is where the schema gaps are settled |
| D39 | No tool attaches a deck to an EXISTING campaign — `update_campaign` takes no `asset_ref` and `upload_image_asset` takes an image — so a record whose `commentary_checked` is false can only gain its comments by being uploaded again as a duplicate | 5.2 | 12.4 | Found by 5.2's own test that every offered action must name arguments its tool accepts: the obvious offer could not be made. Adding a tool is a surface decision, and 12.4 is where the schema and surface gaps are settled |
| D40 | An evaluation cannot be linked to a campaign after the fact, so the store-then-reconcile sequence cannot complete: accepting "add this to the library" leaves the judgment's `campaign_id` null and nothing can set it | 5.2 | 9.9 | Reconciliation is 9.9's subject, and the link is what makes it possible |
| D41 | Supersession offered only on evidence the schema actually holds — a non-empty `resolved`, a record that already carries `supersedes`, or (added by 5.4) a `comparable` diff between two unlinked records, which is stronger evidence than either since two judgments of the same brief exist. Never on `closest_precedent`, and with the replaced record named in the label. Needs a write path: `update_campaign` cannot set `supersedes` | 5.2, 5.4 | 6.3 | Both reviewers independently condemned the similarity-based version; 6.3 is where supersession and the prediction loop meet |
| D42 | `next_actions` carry no staleness token, so an offer accepted several turns later can act on a library that has changed underneath it | 5.2 | 10.5 | 10.5 is `menu_token`, which is exactly this problem |
| D43 | An offer whose arguments the user must supply needs a defined shape — `needs` is a list of prose today, and 5.6's guided first run is made almost entirely of such offers | 5.2 | 5.6 | 5.6 is the item that will actually depend on it |
| D46 | `supersedes` is never validated to exist — `upload_campaign(supersedes="camp_doesnotexist")` is accepted and stored. 5.4 now trusts that pointer to decide which version is earlier, so a dangling one silently inverts a diff | 5.2, 5.4 | 6.4 | Supersession integrity belongs with the item that makes the reasoning defensible |
| D47 | `bulk_import_metrics` still flattens `BadValue` to a string, so the structured retry (field/valid/suggestion) is available on the single-write path and not on the batch path — which the plan itself named as the likeliest place "Target" arrives | 5.2 | 8.8 | 8.8 is the bulk-import item |
| D49 | `gaps()` reports missing fields *within* records — "no LATAM store launch has ever carried a budget" is one of the review's own three examples and is about an absent field, not an absent record | 5.3 | 7.1 | Needs 7.1's computed facts (budget detected, on the body layer) to produce the per-record signal before anything can aggregate it |
| D50 | `gaps()` reports a named expected input that has never been supplied — the review's third example, the KPI workbook the rubric itself names | 5.3 | 12.1 | Needs a rulebook that *declares* expected inputs; today the rubric is a `reference` row retrieved by similarity |
| D51 | `commentary_never_read` reported as a gap once a deck can be attached to an existing record | 5.3 | 12.4 | Blocked on D39: the only offer available today creates a duplicate campaign, which is the offer 5.2 refused in writing |
| D52 | Gap ranking by judgments affected rather than a fixed rank per kind — one unindexed record currently outranks forty | 5.3 | 10.1 | "Ordered by value" is 10.1's subject, and there is not enough evaluation traffic to rank by yet |
| D53 | An "acknowledged" or "not applicable" state for a gap the user cannot close | 5.3 | 10.2 | Needs a human answer, which is what the numbered queue is for |
| D54 | `gaps()` offered at session start and after an upload that changes the top gap, rather than by a sentence in a docstring | 5.3 | 10.6 | Proactive offers are 10.6's; the instructions are 7.4's |
| D56 | Verdict agreement stratified by whether the missing-input line fired | 5.3 | 7.7 | No golden set yet |
| D57 | Predictions half of the supersession feeder: when a superseding record is judged, surface the prior evaluation's `predictions` and ask which held | 5.4 | 6.3 | 5.4 built the findings half; predictions are 6.3's subject |
| D58 | Fact-level "went stale" — dates, budgets, channels diffed across versions as computed facts, not only stale citations and structured record fields | 5.4 | 7.1 | Needs 7.1's computed facts on the body layer |
| D59 | A human-answerable path for `no_longer_raised`: append `resolved` to an existing evaluation, so "yes, we fixed that" can be recorded after the judgment was saved | 5.4 | 10.2 | Needs a write path onto a saved judgment (with D40) and a queue to ask in |
| D60 | `compare_execution` reuses this result vocabulary — `basis`, `match`, buckets named for the observation, `counts` — so "what changed" means one thing at both levels | 5.4 | 9.2 | 9.2 does not exist yet |
| D61 | Whether text-similarity matching qualifies as `basis: computed` at all, or needs a third value | 5.4 | 7.8 | 7.8 owns the vocabulary; 5.4 stamps such matches `judged` in the meantime |
| D62 | Commentary on the earlier deck treated as the corrections record — in the Colombia case "corrections" meant the client's tracked comments at least as much as the library's findings | 5.4 | 12.4 | Depends on the `approval_notes` decision (with D15) |
| D63 | Deck-level diff: added and removed passages between two versions, by layer, from the chunks already stored per version | 5.4 | 12.4 | Re-pointed from 5.5: a coverage view is about the library, a deck diff is about two records, and they share no machinery. It is a new capability on the diff surface, so it belongs with the other surface gaps |
| D64 | A supersession chain longer than one link — `diff(v1, v3)` reports a finding resolved in v2 as `no_longer_raised`, since only adjacent judgments are compared | 5.4 | 6.3 | Walking the chain is supersession logic, which 6.3 owns |
| D23 | The customer overlay should name *who IT is here* (a support contact), so a remedy can say "contact X" rather than the generic "ask whoever installed this" | 3.1 | 12.2 | No overlay file yet |

## Open — decided against

| # | Rejected | Raised in | Reason |
|---|---|---|---|
| X1 | Exposing `include_commentary` on `prepare_evaluation` | 2.5 design review | A judgment should weigh every recorded objection. A switch that drops the layer carrying the client's is an easy path to a cleaner-looking approval — 2.4's lesson was that an error message offering an easy exit gets taken, and a parameter is no different |
| X2 | Merging the health-check code vocabulary (`clip_weights_missing`) with the notice vocabulary (`visual_search_offline`) | 3.1 review | They are different axes: a health code names what is *wrong*, a notice code names what the user *loses*. Linked by `cause` instead, so support reading an upload warning and an installer gating on health_check can tell they are one problem |
| X3 | Reordering commentary ahead of body chunks in the embed loop | 2.5 design review | Commentary is the first thing a time budget drops, which is a real cost — but a deck with 200 comments and 10 body chunks would starve the body, and the deck is what the user uploaded. `finish_indexing` recovers either way |
| X5 | Making the model-authored vocabularies forgiving too (verdict, severity, finding `kind`, `basis`, precedent `layer`, `include_commentary`) | 5.1 review | A `Literal` retry costs the model nothing, and §2.4's lesson was that any easy exit offered in an error message gets taken. Auto-mapping "minor" to `note` would hand it a severity downgrade path — the exact thing §7.7's golden set exists to detect. The split is by authorship: what a person types is forgiven, what the model emits is not |
| X4 | Removing `check-weights` now that `health-check` exists | 2.3 | CI runs it on a runner with no Ollama and no database, where a full health check fails on text search. They answer different questions |

## Accepted limits

| # | Limit | Why it stays |
|---|---|---|
| L1 | Commentary and embedded images can only be read when the actual file reaches the server (`asset_ref`) | The `deck_text`-only path never gives us bytes. Inherent, not a bug — surfaced as `commentary_checked: false` so it is never mistaken for "there were none" |
| L2 | A PPTX reviewer comment anchors to `"deck"`, not a slide | The number in `commentN.xml` is the comment part's ordinal, not the slide's. Resolving it needs the package relationships; an anchor that is sometimes silently wrong is worse than one that admits it does not know |
| L3 | Legacy `.ppt` is stored but never text-extracted | No reliable pure-Python extractor. Reported as `legacy_ppt` with the instruction to re-save as `.pptx` |
| L5 | A failed Windows self-test cannot abort setup or set a non-zero exit code | Inno invokes `ssPostInstall` with `HandleExceptions = True`, which logs and swallows any exception, so execution reaches the Finished page regardless; only `ssInstall` re-raises, and an exception there rolls the files back — contradicting the advice to fix the problem and re-run against what was just installed. What is achievable is done: Claude Desktop is left unconnected and the Finished page says *"Installed, but not working"*. Claiming an abort would be the overstatement this exercise is about |
| L7 | A deck can be attached to an existing campaign only as an image asset, which stores the file but reads nothing from it | `upload_image_asset` accepts a `.pptx` path and returns an `asset_id`, but no text, images or commentary are extracted from it — so it is storage, not ingestion. D39 covers the missing capability; this records that the nearest existing path does not substitute for it |
| L6 | Installing on someone else's behalf on Windows still configures the installing account | With a per-user install there is no UAC prompt to run steps under a different identity, and with an elevated one every step acts on the administrator's profile. No installer can fix this from the other side; the honest answer is to run it signed in as the person who will use it, which the script now says |
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
| C8 | Every fresh install failed its own gate — `health_check_cli` opens the database read-only by design, and nothing created it | Phase 4 review round: `campaign-intelligence init` is an install step, which also proves the data location is writable by whoever is running the installer |
| C9 | A refused install still left Claude Desktop wired to the product it had just condemned | Same round: the gate runs before the wiring on all three platforms, and the failure message says so |
| C10 | A failed upgrade destroyed the working copy before verifying the new one | Same round: the Unix installers stage and swap only after the self-test passes |
| C12 | `next_step` prose in warnings (was D16) | 5.2 — `next_actions` with `{label, tool, prefilled_args}`; completed in 5.3, which covered `results_may_be_incomplete` (the last one where a single tool sat behind the advice) |
| C13 | Teaching errors carried the retry only inside a sentence (was D31) | 5.2 — `enums.BadValue` carries `field`, `valid` and `suggestion`, and `_catch_value_errors` passes them through the tool boundary |
| C14 | The one upload-time finding `gaps()` needed could not survive the call that produced it (was D17) | 5.3 — `campaigns.commentary_checked` is persisted and backfilled. **Only that one:** `images_unreadable`, `commentary_capped` and the rest are still ephemeral, which D20 covers |
| C15 | `results_may_be_incomplete` carried `finish_indexing` as prose (was D45) | 5.3 — it carries the action, `consent: "do"`, on every surface that raises it including `prepare_evaluation`. Note those actions are nested inside the warning rather than at the top level of the result |
| C16 | `gaps()` grouped markets itself, so it and the coverage view could describe one library differently — and it ignored the `markets` list (was D55) | 5.5 — both group through `_markets_of`; verified over the protocol that the two name the same thin markets |
| C11 | The Windows uninstaller never removed the Claude Desktop connector entry, while both Unix ones did | Same round: `unconfigure-desktop` |
| C7 | `text_search_offline` said "start the local text model service", which names no gesture anyone performs | Caught by the tracker's own staleness test on its first run: it was owed to 4.3, and 4.3 had shipped. The remedy now names re-running the installer, which starts the service, installs the model, and self-tests both |
