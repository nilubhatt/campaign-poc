# Product review response — sequenced work plan

Source: `Campaign-Intelligence-Product-Review.html` (11 Sep 2026, gathered against the
installed Windows build while preparing a leadership demo). Every item below traces to a
numbered defect or lettered idea in that document.

## How this work is executed

One item at a time, in the order below. For **each** item, in this order:

1. Write the failing test(s) first (TDD) — confirm red.
2. Implement; confirm green; run the **full** suite.
3. Verify over a real MCP protocol round-trip where the item touches the tool surface.
4. Two **independent** adversarial/design reviewers (fresh agents, fable) on that item alone.
5. Fix every real finding; re-review until clean.
6. Update this file's status, commit, push onto the **same** PR; then start the next item.

Isolating each item this way is deliberate: no cross-feature drift, no batch that can't be
reasoned about. Findings from a review round belong to the item under review.

**Platform rule:** the review was gathered on Windows 10, but every fix must hold on
Windows, macOS and Linux. Anything platform-specific (installer payload, script encoding,
path handling) gets verified per platform, not assumed from one.

**Decisions taken before starting** (confirmed with the product owner):
- **Weights:** bundle in the installer payload on all three platforms (option A), *and*
  honour a `CLIP_WEIGHTS_PATH` override (option C). Not the GitHub-Release fetch (option B) —
  the bar is "zero outbound network required", which B does not meet.
- **Rulebook:** the product stays generic. Ship a minimal/skeleton default `rulebook.yaml`;
  the transcribed Fabletics rules ship as a **separate example/customer config** that layers
  on top, never as the product's baked-in default.
- **Evaluation schema:** clean cutover to the structured findings array. No legacy free-text
  reading path — there are no production customers to preserve.

Status: `[ ]` not started · `[~]` in progress · `[x]` done (tested, reviewed, pushed)

---

## Phase 1 — P0: the product must work air-gapped

- [ ] **1.1 `CLIP_WEIGHTS_PATH` override** (defect 02, ship-route C). `clip_embed._load_model`
      passes a filesystem path straight through to `open_clip` when the env var/config key is
      set. Removes the runtime Hub dependency for an admin holding the file.
- [ ] **1.2 Bundle the weights in the installer payload** (defect 01, ship-route A). CI
      downloads the weights once, verifies SHA-256, ships them in the Windows/macOS/Linux
      payloads; the app resolves the bundled path by default. fp16 to halve size if accuracy
      holds for ranking.
- [ ] **1.3 No Hub dependency at runtime** (defect 03). Verified, not assumed: with weights
      present locally, assert **zero** network calls to huggingface.co. Install fails loudly
      (hash-verified) if the payload is absent or corrupt.

## Phase 2 — P1 defects

- [ ] **2.1 No first-use init or network I/O inside a tool handler** (defect 04). `warm_up()`
      exists but the reviewer still hit a 60s transport timeout inside `upload_image_asset` —
      audit **every** handler for lazy initialisation, bound any residual network call well
      under the transport ceiling, fail fast with a typed error.
- [ ] **2.2 `reembed(scope)` + partial-state visibility** (defect 05). Backfill any row
      missing a vector (text or image); surface counts in `list_campaigns`/`get_campaign` so
      "stored" and "searchable" are never conflated. Two assets are currently unrepairable.
- [ ] **2.3 `health_check` tool** (defect 06). Component status (ollama/clip/db) + coverage
      counts. Doubles as the installer's post-install self-test and a pre-demo preflight.
- [ ] **2.4 Structured findings array replaces free-text `analysis`** (defect 07). verdict,
      summary (≤240), closest_precedent, findings[] (severity enum ×3, category, finding
      ≤120, detail, precedent{id,quote}, fix ≤120), resolved[]. Caps and enums enforced
      server-side. Clean cutover.
- [ ] **2.5 Commentary layer: comments, annotations, speaker notes** (defect 08). PDF
      `/Annots` (Text/FreeText/Highlight/StrikeOut/Underline/Square/Caret/Ink) with
      `/Contents`, `/T`, `/M`, page index; PPTX `notesSlide` + `ppt/comments/` +
      `ppt/modernComments/`. Stored as `commentary[]` `{page, author, date, text, kind}` —
      a distinct layer from `deck_text`, embedded but tagged so retrieval can weigh it.

## Phase 3 — P2 defects

- [ ] **3.1 Structured warnings** (defect 09). Stable `code`, one-line human `remedy`, raw
      `detail` — so a surface can say "Visual search is offline — ask IT to run setup".
- [ ] **3.2 Version surface + stale-schema documentation** (defect 10). Build version in the
      server description; README documents that the host app must be fully quit and reopened
      after a rebuild (closing the window leaves the server running).
- [ ] **3.3 Shipped scripts must be encoding-safe** (defect 11). Any `.ps1` ASCII-only or
      UTF-8 **with** BOM, asserted in CI; audit the macOS/Linux shell scripts for the
      analogous trap rather than assuming it is Windows-only.

## Phase 4 — Installer acceptance criteria

- [ ] **4.1 Post-install self-test runs `health_check`** and a non-green result blocks the
      success screen, naming the failing component — all three platforms.
- [ ] **4.2 Zero-egress install verified** on a host with egress disabled (actually tested,
      not "degrades gracefully").
- [ ] **4.3 Ollama verified at install** to the same standard as the vision model — daemon
      reachable and `nomic-embed-text` present.

## Phase 5 — Making it intuitive (ideas A–F)

- [ ] **5.1 (A) Forgiving enums, teaching errors.** Every enum error returns the valid set
      and the closest match; normalise on the way in ("client stated" → `stated`).
- [ ] **5.2 (B) `next_actions`** on every result — `{label, tool, prefilled_args}`.
- [ ] **5.3 (C) `gaps()`** + a standing line on every evaluation naming the single most
      valuable missing input for that judgment.
- [ ] **5.4 (D) `diff_campaigns(a, b)`** — adopted / ignored / newly-introduced /
      carried-stale, computed against the earlier version's evaluation findings. Flagged by
      the reviewer as the highest-value feature not already on the list.
- [ ] **5.5 (E) Coverage view** — market × collection × stage, counts and evidence quality.
- [ ] **5.6 (F) Guided first run** on an empty or thin library.

## Phase 6 — Making the reasoning defensible (ideas G–L)

*Depends on 2.4.*

- [ ] **6.1 (G) `precedent.quote` required** on every finding, drawn from the retrieved chunk.
- [ ] **6.2 (H) Guardrail breach vs departure from precedent** — two classes, different
      vocabulary, only one is debatable.
- [ ] **6.3 (I) Close the prediction loop** — when a superseding record arrives, surface the
      prior evaluation's predictions and ask which held.
- [ ] **6.4 (J) Disconfirming search required** before a verdict is saved; record what came
      back, including "nothing".
- [ ] **6.5 (K) `approve_if`** — the testable exit condition that converts revise → approve.
- [ ] **6.6 (L) Evidence-strength line** on every judgment — how many precedents, concluded,
      verified; top similarity; whether one match dominates.

## Phase 7 — Consistency across users (ideas 1–8)

*Reviewer's own order: 1 and 2 first, then 3 and 4, then 6 and 7 before any prompt tuning.*

- [ ] **7.1 Compute what can be computed.** Date coverage, ER presence per profile, budget
      detected, channel checklist, internal date consistency, guardrail keyword hits —
      returned as **computed facts** with evidence. Two thirds of real findings were
      mechanical; this removes most cross-user variance.
- [ ] **7.2 Server owns the retrieval query.** Derive from the subject record/file, not
      model-authored `proposal_text`; derive filters from the subject's attributes; pin
      `top_k`; stable deterministic tie-breaking; record embedding-model version per vector.
- [ ] **7.3 Enforce the output shape server-side** — reject writes missing a precedent quote
      or exceeding caps, rather than accepting and hoping.
- [ ] **7.4 Tool descriptions as the shared prompt** — the evaluation procedure into
      `prepare_evaluation`'s description; set the MCP server-level `instructions` field.
- [ ] **7.5 Ship the procedure with the evidence** — `prepare_evaluation`'s `note` carries
      the pinned rulebook, computed facts, required output schema and weighting rules.
- [ ] **7.6 Stamp every verdict** with `rulebook_version`, `server_version`,
      `embedding_model`, `model_id`, and retrieved ids + similarities.
- [ ] **7.7 Golden set + agreement measurement** — verdict agreement and finding recall, three
      runs per brief. Harness built now; briefs to be supplied by the product owner.
- [ ] **7.8 Mark findings `computed` vs `judged`.**

## Phase 8 — Metrics that evolve

- [ ] **8.1 Metric registry** — canonical key, display name, unit, direction, aliases, first
      seen, times seen, campaign types, status. Values in typed storage, not an opaque JSON
      string (today's `structured` cannot answer "show me every ROAS on file").
- [ ] **8.2 Unknown metric → provisional + ask** (never reject, never silently accept).
- [ ] **8.3 Graduation gate** — seen in N campaigns, ≥2 partners/markets, confirmed once by a
      person, before a measure becomes expected.
- [ ] **8.4 Fixed template, growing data** — never a self-editing prompt.
- [ ] **8.5 Retirement** — track last seen, demote after M campaigns, never delete.
- [ ] **8.6 Same loop for standing corrections** — one learning mechanism for both.
- [ ] **8.7 Replay as a report** — which campaigns now fail a new expectation; which past
      verdicts would change. Never silently rewrite old verdicts.
- [ ] **8.8 `bulk_import_metrics` diffs columns against the registry before writing.**

## Phase 9 — Execution drift and context events

- [ ] **9.1 Asset `phase`** — `proposed` vs `delivered`, plus date.
- [ ] **9.2 `compare_execution(campaign_id, delivered_assets[])`** — as briefed / never
      appeared / new, plus a drift score from visual distance. Machinery already exists.
- [ ] **9.3 Named commitment checking** — extract the commitment list at upload, check each
      against delivered photos.
- [ ] **9.4 Classify drift** improvement / neutral / degradation, marked `judged`.
- [ ] **9.5 Attach drift to the outcome record**; high-drift campaigns are weaker precedent
      and retrieval says so.
- [ ] **9.6 `context_events` table** — date range, scope, kind, description, source, stated
      impact; auto-linked by market + window overlap.
- [ ] **9.7 Seed fixed calendar events per market**; `prepare_evaluation` checks every
      proposed window against them as a computed fact.
- [ ] **9.8 Record overlap, never assert cause** — mark outcomes `confounded`; the caveat
      travels with the metric wherever it is cited.
- [ ] **9.9 Make `reconcile_evaluation` do its job** — predicted vs delivered vs actual vs
      context, one record. It has never run.

## Phase 10 — Feedback capture

- [ ] **10.1 Define "open"** as *needs something from a human*, ordered by value.
- [ ] **10.2 `feedback_queue()`** with fixed numbering (1–9 campaigns, 10 concluded, 11 more).
- [ ] **10.3 Numbered all the way down**; free text invited, never required.
- [ ] **10.4 Server builds the menu, not the model.**
- [ ] **10.5 `menu_token`** so a stale selection refreshes instead of writing to the wrong row.
- [ ] **10.6 Offer the queue proactively** — after uploads/evaluations and on first
      interaction of a session.

## Phase 11 — Attribution

- [ ] **11.1 Environment-derived author** — `os_user`, `host`, config display name.
- [ ] **11.2 Reuse `verified` vs `stated`** for author rather than inventing a vocabulary;
      add `method` (stdio_local | sso | import).
- [ ] **11.3 `captured_by` vs `on_behalf_of`** — the field most products miss; asked as one
      numbered question.
- [ ] **11.4 Context** — `captured_at`, channel, session id, role **as stated at the time**.
- [ ] **11.5 Append-only reactions** — keep both sides of a disagreement, surface it in
      retrieval, authority order configured in the rulebook, never inferred.
- [ ] **11.6 Backfill existing records as `author: unknown`** with import date, explicitly.
- [ ] **11.7 Treat it as personal data** — install disclosure, per-person view, deletion or
      anonymisation preserving the judgment, stated retention position.

## Phase 12 — Rulebook as versioned configuration

- [ ] **12.1 Skeleton `rulebook.yaml` + loader** — bundled, versioned, loaded at startup,
      applied deterministically **outside** the similarity path (today the rubric is a
      `reference` row retrieved by similarity, truncatable and deletable).
- [ ] **12.2 Customer overlay file** layering on the default, with the version stamped onto
      every saved judgment (pairs with 7.6).
- [ ] **12.3 Fabletics example config** — the full transcribed content (guardrails, tag
      taxonomy, influencer criteria + red flags, partner feedback, 360 checklist, scorecard)
      and the ten standing corrections with provenance, shipped as an **example/customer**
      file, not the product default.
- [ ] **12.4 Schema gaps named in the review** — no field today for `asset_link`,
      `approval_notes`, or market-scoped feedback patterns.

---

## Regression baseline — must not break

The review explicitly verified these as working. Any change that breaks one is a regression:

- Perceptual fingerprinting (resized/recompressed/renamed slide matched at Hamming 0, own
  assets correctly excluded, cross-region flag raised).
- Aesthetic similarity once weights are present (sensible ordering, 0.66 vs 0.35).
- Graceful degradation on a failed embedding (asset stored, precise warning, no crash).
- Passage chunking (5/5 chunks embedded; resolved the old long-deck 500s).
- Bundled dependency set correctly collected into `_internal`.
