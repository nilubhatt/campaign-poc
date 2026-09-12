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
- **Durable partial state over transactional writes.** The review asked for this to be
  decided deliberately ("either is defensible, silent partial state is not"). Chosen:
  records are committed before they are indexed, so an interrupted run leaves rows rather
  than rolling back. Rolling back would throw away the fingerprinting and reuse-check that
  are the product's core question, and a deck too large for one time budget could then never
  be stored at all. The obligation that comes with the choice is that partial state is never
  silent: counts in `list_campaigns`/`get_campaign`, a warning on the upload that names the
  fix, a warning on search that results may be incomplete, and `finish_indexing` to close it.

Status: `[ ]` not started · `[~]` in progress · `[x]` done (tested, reviewed, pushed)

---

## Phase 1 — P0: the product must work air-gapped

- [x] **1.1 `CLIP_WEIGHTS_PATH` override** (defect 02, ship-route C). A local checkpoint —
      file *or* folder containing one — is passed straight through to `open_clip`.
      Resolution is separated from loading (`resolve_weights()` is cheap, never raises,
      returns source/ok/reason/remedy) so startup records it and `health_check` can report
      it later; a bad path does **not** kill a server whose text tools don't need CLIP.
      Reported via `/healthz` → `clip_weights` and a stderr line at startup.
      *Two review findings folded in:* the frozen binary's `stdio` entry point never called
      `warm_up()` at all — the actual root cause of defect 04, since the installers wire
      exactly that path — and the first error message told the admin to fall back to the
      network that defect 03 says is blocked.
      Empirically verified by the adversarial reviewer: with a local path, loading makes
      **zero** network calls and yields vectors bit-identical to the tag, while the tag path
      issues live requests even with a warm cache.
- [x] **1.2 Bundle the weights in the installer payload** (defect 01, ship-route A).
      `scripts/fetch_weights.py` fetches the checkpoint pinned to an immutable HF commit,
      verifies SHA-256, and writes a `.sha256` sidecar; CI, `build.sh` and `build.ps1` all
      stage it into `<bundle>/models/`. The app auto-discovers it at `config.app_dir() /
      "models"` — beside the executable, outside `_internal`, from `sys.executable` when
      frozen (symlink-safe, covered by a test since the Linux installer puts a symlink on
      PATH). Resolution order is env → bundled → tag.
      **An installed copy with no weights is now an error, not a network fallback** — both
      reviewers independently caught that the first version contradicted this plan's own
      rule and would have recreated defects 01/03 verbatim on the customer's machine while
      reporting itself healthy. Source checkouts still resolve the tag.
      CI's verification now unpacks the **shipped archive** and runs the frozen binary's new
      `check-weights` subcommand (which item 4.1's post-install self-test also calls) — the
      first version overrode `BUNDLED_WEIGHTS_DIR` on the source tree and proved nothing
      about packaging. `installer/linux/install.sh` verifies the sidecar after copying, so a
      truncated copy fails the install instead of surfacing later inside a tool call.
      Inno keeps the checkpoint out of the solid LZMA2 stream (`nocompression solidbreak`) —
      300MB of near-random tensor data compresses ~0% and costs minutes.
      Manually verified (no automated test yet — that is 1.3): bundled weights load and
      embed with the network hard-blocked; a truncated copy fails the sidecar check; an
      installed copy with weights removed reports `source: missing` with a reinstall remedy. Asset sizes land ~830-950MB, clear of the
      2GB release cap that bit v0.2.0/v0.2.1.
- [x] **1.2b Halve the payload, and stop depending on huggingface.co at build time.**
      The checkpoint is now **fp16, 302,588,458 bytes** (half of fp32, to within a 33KB header), mirrored as
      a release asset on this repo (`weights-v1`) and tried *before* upstream — the premise
      of this whole phase is that huggingface.co is blocked at the customer, so depending on
      it to *build* the fix was a dependency worth removing too. Upstream stays as a
      fallback, so a deleted or renamed release can't break a build; each source carries its
      own hash and size because they are genuinely different files.

      **The review's condition was "no meaningful accuracy cost for similarity ranking" —
      measured, not assumed** (`scripts/convert_fp16.py` reports it):

      Both rows below are reproducible from the checked-in script —
      `python scripts/convert_fp16.py <fp32> <fp16> [photo_dir]` prints them, and labels the
      probe set it used so an easy-case number can never be mistaken for a hard-case one:

      | probe set | similarity band | fp16 perturbation | tightest margin | headroom | top-1 / top-5 | full order |
      |---|---|---|---|---|---|---|
      | 12 synthetic shapes (easy) | 0.619–0.976 | 5.4e-04 | 7.1e-03 | 13× | identical | identical |
      | **13 real photographs (hard)** | **0.807–0.985** | 1.4e-04 | 8.3e-04 | **6.1×** | **identical** | **not identical** |

      Cosine between the fp32 and fp16 embedding of the same image is ≥ 0.999999, and the
      fp16 file loads into an **fp32 model** (`load_state_dict` upcasts — verified by
      inspecting parameter dtypes), so inference precision is unchanged; only the stored file
      is halved.

      **Three corrections this measurement went through, all worth keeping:**
      1. The first run used upscaled random noise and reported *ranking changed*. It does —
         but noise images sit within ~1e-5 of each other under CLIP, so any perturbation
         reorders near-ties. A fact about the probe set, not about fp16.
      2. The second used distinct synthetic shapes and reported a comfortable 13× headroom,
         which flatters the result by measuring the easy case.
      3. The third used hand-made "near-variants of one image" — which are *near-duplicates*,
         a case this product handles with pHash, not CLIP. Review caught that this still
         wasn't the real hard case. Real photographs are: distinct images that are
         nonetheless alike, which is what ranking one brand's campaign shoots actually is.

      **The honest limit:** top-1 and top-5 ordering are stable, but full ordering is not —
      on the real-photo set one adjacent pair sat inside the perturbation window. So:
      *ordering of two candidates whose similarity differs by less than ~1.5e-4 is not
      guaranteed identical between the two checkpoints.* That is below the 4th decimal the
      product displays, and such ties carry no information under fp32 either — but it is a
      behavioural difference, not "identical".
      (An earlier draft defended this by saying sub-threshold gaps are "near-duplicates,
      pHash's job". Both reviewers independently flagged that as a non-sequitur, and they are
      right: the tie is between two *candidates' distances to a query*, not between the
      candidates themselves — two images 0.82 similar to each other can still tie. Removed.)
      Note also that the review's "0.75–0.81 band" complaint was about the **text** embedding
      space (nomic), not CLIP; the CLIP bands above happen to overlap it, which is why the
      comparison is still worth making, but they are not the same measurement.
      A release build **requires** the mirror (`CAMPAIGN_WEIGHTS_REQUIRE_MIRROR=1` on tag
      builds) and asserts which checkpoint got staged by hash — review's point that a
      silently-missing mirror would ship a different, 300MB-larger model with different
      vectors and nothing going red. Local dev keeps the fallback.
      Manually verified: the mirror fetches and hash-verifies from the live release, and the
      fp16 checkpoint loads and embeds with the network hard-blocked. Assets drop ~300MB.

      **Follow-ups this raised, tracked rather than lost:**
      - *Vector comparability.* Pre-upgrade vectors came from fp32, post-upgrade from fp16.
        Ranking is unaffected (measured above), but ~45-55% of displayed similarities change
        in the 4th decimal, and a customer re-running yesterday's query has no explanation.
        Nothing records which checkpoint produced a vector. Pulled into **7.6** (stamp the
        embedding identity) and **2.2** (`reembed` on mismatch) — fp16 is benign, but the
        next real model change would silently mix incomparable vectors.
      - *Attestation.* The conversion is bit-for-bit reproducible (a reviewer independently
        re-derived `cbd90e47…` from the pinned upstream blob). Worth a CI job that rebuilds
        it from upstream and attaches build provenance, so "why trust this mirror" has an
        answer a customer can run themselves. Added as **1.2c**.
- [x] **1.2c Prove the mirror, don't ask to be trusted.** The fp16 conversion is
      bit-for-bit reproducible — a reviewer independently re-derived `cbd90e47…` from the
      pinned upstream blob using the checked-in script, which is a better answer to "why
      should I trust a checkpoint hosted in someone's GitHub repo" than asking for trust.
      `.github/workflows/weights-provenance.yml` now demonstrates it on every change to the
      weights tooling: fetch the pinned upstream fp32, convert, require the result to equal
      `MIRROR.sha256` (failing loudly with "do not ship until this is explained" if it ever
      diverges), then attach build provenance via `actions/attest-build-provenance` so a
      customer can run `gh attestation verify`. README gained a "weights provenance" section
      with both hashes, the upstream commit, and the one-line rebuild command.

- [x] **1.3 No Hub dependency at runtime** (defect 03). Made a **property**, not a claim:
      `HF_HUB_OFFLINE=1` is now set before `open_clip` is imported whenever the weights
      resolved locally (env or bundled). A test proving "we didn't call the network" only
      covers the paths it exercises; the variable covers the ones nobody thought of — a
      future library version checking for a model-card update, a transitive import phoning
      home — on precisely the network where that call fails as a confusing TLS error. It
      must be set before the import, since `huggingface_hub` reads it into a module constant
      at import time; an operator who set it themselves is left alone, and the tag path (a
      source checkout, which genuinely needs the Hub) is left online.
      Demonstrated against the real library, not a stub: a cache-gated integration test
      blocks `create_connection`, `getaddrinfo`, `socket.connect`, `urlopen` and
      `hf_hub_download`, loads a real checkpoint, and embeds an image — zero attempts,
      512-dim unit vector. Skips with an actionable reason on a machine with no checkpoint.
      Preprocess parity is pinned rather than assumed (review's point): loading by path
      skips the tag's preprocess metadata and falls back to model defaults, which are
      identical for `ViT-B-32-quickgelu`/`openai` today — but if that ever diverges the
      failure is silent, vectors keep computing and simply stop being comparable with
      everything already in the database.
      Install-time "fail loudly if absent or corrupt" is **4.1's** job, not this item's —
      ownership recorded there.

## Phase 2 — P1 defects

- [x] **2.1 No first-use init or network I/O inside a tool handler** (defect 04). The
      review asked for a sweep — "a pattern, not a single site" — and the sweep found two
      sites worse than the one the reviewer hit.
      **`embedding.embed()` had `timeout=60`, equal to the entire transport ceiling**, and
      `ingest_campaign` calls it once per chunk: a twelve-chunk deck against a stalled
      Ollama could block for twelve minutes inside one call while the client gave up at
      sixty seconds — the reported symptom exactly (timeout + half-written row + no way to
      tell which half). Now `EMBED_TIMEOUT_SECONDS` (15s), and each call is granted only the
      budget actually remaining, because checking a deadline at the top of a loop bounds
      when work *starts*, not when it *ends*: the first defaults came to 45 + 15 = 60, i.e.
      a budget that could still be cut off, which is no budget.
      **The text embedder was never warmed at all.** CLIP was warmed at startup; Ollama was
      not, and it unloads `nomic-embed-text` after its idle window — so the first chunk of
      every upload paid the model load inside a handler. That is this item's own definition
      of the defect, sitting in the code meant to be sweeping for it. Added
      `embedding.warm_up()` at all three entry points plus `keep_alive` so it stays resident.
      One wall-clock budget covers the whole handler, not one per loop — the image loop runs
      first and was silently spending what the text loop then measured against.
      **The image loop is now two passes**, which review argued for and is plainly right:
      pass 1 stores, fingerprints and reuse-checks every image (milliseconds, and reuse
      detection is the question this product exists to answer — never worth cutting); pass 2
      does the expensive CLIP embedding and is what yields. Every image therefore leaves a
      row that can be finished later, matching how text chunks already behaved, instead of
      being silently lost with nothing to recover.
      **A regression review caught before it shipped:** `images_checked` still reported
      `True` after a partial loop, and the tool docstring tells Claude that means reuse *was*
      checked — so it would have told a marketer "no reuse found" about slides nobody looked
      at. The response now carries `images_total`/`images_embedded` counts, and the warnings
      state the position plainly rather than naming `reembed`, a tool that does not exist
      until 2.2 (asserting that promise in a test was TDD against the wrong item).
      **Deferred to 2.2 with its scope sharpened:** `reembed` must handle three states, not
      one — unembedded chunk rows, unembedded asset rows, and (only if the two-pass change
      had not been made) images never extracted at all. It must also honour the same budget
      and be resumable. `campaigns.embedded` being a boolean is now actively misleading and
      should become counts.
      **A second review round found four more**, all fixed here: `embedded: True` on a
      campaign with 2 of 12 sections indexed (the boolean lied *by design* once partial
      became a designed outcome — now it means fully searchable, with
      `chunks_embedded`/`assets_embedded` counts in `list_campaigns` telling the partial
      story); an embedder timeout reaching the caller as a generic "Error executing tool X"
      because only `ValueError` is translated (now a sentence naming the component, the
      address and the likely cause); `bulk_import_metrics` as the last unbounded handler,
      committing once per caller-supplied row (now one transaction, same budget, reporting
      `not_processed`); and this item's own budget test passing against an implementation
      whose grant never shrank.
      Also: env names now carry their unit like every sibling, and a budget of `0`, `nan` or
      a non-number is rejected at startup rather than silently disabling the protection it
      configures.
      **Not yet measured:** the 45/15 defaults are reasoned, not observed. A per-ingest
      timing line (chunks, images, seconds) belongs in 3.2's version/diagnostics work so the
      next customer run produces the number rather than another estimate.

- [x] **2.2 `finish_indexing` + partial-state visibility** (defect 05). Repairs anything
      stored but not searchable — unembedded chunks and unembedded image assets alike —
      without re-uploading a thing, which is the review's acceptance line. Verified over a
      real MCP round-trip against the reviewer's own scenario: a deck whose budget ran out
      shows `chunks_embedded: 0`, is invisible to search, and comes back fully findable.
      **Named for the user's intent, not the mechanism.** "reembed" is jargon and the "re-"
      is false — these rows were never embedded. `reembed` is left free for 7.6's genuinely
      different operation, re-embedding when the model identity changes (moved wholly to 7.6;
      this item does NOT do model-mismatch detection).
      **The failure modes mattered more than the happy path.** With the embedder down, every
      item fails in milliseconds, so the first version reported a full backlog and advised
      "call again" — an invitation to loop forever against something that is not coming back.
      Now: progress made → resume; no progress → name the cause and say retrying will not
      help; items that can never succeed (their file is gone) are counted as `failed` rather
      than `remaining`, so `complete` can still become true. Errors are collapsed by cause
      with counts — two hundred copies of one Ollama message is not a report.
      Also from review: a nonexistent `campaign_id` used to report "complete, nothing to do"
      (now an error, like every sibling); a global run rewrote `updated_at` on every
      campaign including untouched ones; the backlog was worked oldest-first, so the deck
      someone just uploaded finished last *and* a permanently failing row was retried at the
      head of every run, able to starve everything behind it; `count_unembedded` dragged
      every chunk's full text out of the database to count it.
      **Search no longer stays quiet.** The review's second complaint was that a half-indexed
      record is invisible to search while looking fine in a listing. The listing was the easy
      half; the moment it misleads someone is when they ask a question and get a confident
      answer missing a deck. `find_similar_campaigns` now warns when anything is outstanding.
      **Deliberately not done:** background/automatic continuation. Warm-up is idempotent and
      read-only, so a background thread was right there; backfill is a write loop competing
      for the same Ollama model and SQLite connection in a process Claude Desktop kills on
      quit — which would manufacture exactly the silent partial state this item exists to
      remove. The product answer is this primitive plus proactivity: 5.2's `next_actions`
      carrying a prefilled call, 5.3's `gaps()`, 2.3's `health_check` backlog, and 7.4's
      server `instructions` offering to finish outstanding work at session start.
      **Known limit, for pre-2.1 data only:** images that were never *extracted* (a
      `deck_text`-only upload, a failed extraction, legacy `.ppt`) have no row and so cannot
      be repaired — re-upload with `asset_ref` is the only route. Everything stored since
      2.1's two-pass image loop leaves a row. The reviewer's two stranded assets were stored
      and fingerprinted, so they are the repairable kind, and v0.2.0's schema already carries
      the `embedded` columns — their existing database needs no migration.

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
      **Already confirmed present while working item 1.2** — `run.ps1` is BOM-less and
      contains em dashes, including **line 42 inside a double-quoted string**
      (`Write-Host "Ollama not found — downloading..."`), which is precisely the reported
      failure: PowerShell 5.1 decodes it as Windows-1252, the third byte becomes a closing
      curly quote, the string terminates mid-sentence and every brace after it mismatches.
      `installer/windows/campaign-intelligence.iss:1` also has one (in a comment, so
      harmless, but it should not survive the CI check either). `build.ps1` is clean.

## Phase 4 — Installer acceptance criteria

- [ ] **4.1 Post-install self-test runs `health_check`** and a non-green result blocks the
      success screen, naming the failing component — all three platforms. The binary's
      `check-weights` subcommand (added in 1.2) is the seam; Linux already verifies the
      weights sidecar at install, so what remains is Windows/macOS parity and widening it
      from weights to every component once 2.3 exists.
      **Ownership note:** "install fails loudly if weights are absent or corrupt" is
      delivered by 1.2 (CI-side + Linux installer) and completed here (all platforms,
      blocking the success screen) — 1.3 covers the zero-egress *proof*, not the install
      gate.
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
