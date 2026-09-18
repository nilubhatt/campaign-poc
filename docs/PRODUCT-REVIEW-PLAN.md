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
6. Record anything **not** done in [`DEFERRALS.md`](DEFERRALS.md) — deferred, decided
   against, or an accepted limit — in the same commit that creates it. **Every review
   finding gets exactly one of three fates: fixed, a tracker row, or a decided-against row
   with the reason.** No fourth option. This was added after two findings marked "worth
   noting" in a review were neither fixed nor recorded and surfaced only because somebody
   asked whether anything was outstanding — one of them was the first-run guidance
   disappearing before the user had done any of the three things it was telling them to do.
   A finding's severity decides how urgently it is handled, never whether it is tracked.
7. Update this file's status, commit, push onto the **same** PR; then start the next item.

Isolating each item this way is deliberate: no cross-feature drift, no batch that can't be
reasoned about. Findings from a review round belong to the item under review.

Step 6 is not bookkeeping. A review round routinely concludes that something real belongs to
a later item, and that judgment is only safe if the thing is written down at the moment it is
made — reconstructed later it is indistinguishable from having been forgotten. The tracker
separates *owed* work from *rejected* work from *accepted limits*, because collapsing those
three is how a decision gets silently re-litigated or a compliance obligation quietly lost.
`tests/test_deferral_tracker.py` fails the build if a row points at an item that does not
exist, or is still owed to one already marked done.

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

- [x] **2.3 `health_check`** (defect 06). One call, about a second, never raises, never
      hangs — the reviewer needed a 60-second timeout, a second upload, and a read of the
      source to learn that half the product was dead.
      **It paid for itself before review.** On its first run against a live Ollama it found
      a regression item 2.1 had introduced: `keep_alive` was sent as the string `"-1"`, which
      Ollama rejects (`time: missing unit in duration "-1"`), so every embed returned HTTP
      400 and text search was entirely dead. 363 tests were green at the time — they all use
      the offline provider or stub `httpx`, so nothing exercised the real request shape.
      Fixed (send a number, or a duration with a unit) with a regression test on the wire
      format. Logged here rather than buried: it is a 2.1 defect, found by 2.3.
      **Liveness and coverage are separate verdicts.** `ok` means every component is alive;
      `coverage.complete` means the library is fully searchable. Folding them — the first
      version did — meant a working machine failed its own health check because someone had
      just uploaded a big deck, and an installer gating on it would block a good install.
      Item 2.2 deliberately made partial records a normal recoverable state; the health check
      must not call that "broken".
      **Three audiences, three fields.** A stable `code` for installers and later automation
      (chosen now rather than in 3.1 — this surface has no callers yet, and otherwise 4.1
      gates by string-matching prose), an `affects` line in the user's terms, and a `remedy`
      for whoever administers the machine. One string could not serve a marketer, an admin
      and an exit code.
      **Adversarial review then found three ways it reported green on a broken system**,
      each reproduced: vectors written with sqlite-vec loaded are invisible to a build
      without it, so every record read as indexed, the backlog read as zero, and search
      returned nothing — now detected as `vector_index_mismatch`; an embedder returning
      wrong-width vectors passed because the probe discarded the answer; and the CLI ran
      `init_db()`, so a *deleted* database was recreated and reported "0 records, healthy"
      while a corrupt one crashed with a raw traceback. The CLI now opens read-only, reports
      absence as a finding, and loads the vision model for real, because "the weights
      resolved" is not "the weights work" — a corrupt checkpoint previously passed, which is
      exactly what 4.1's post-install self-test exists to catch.
      Also: the failure detail reported the embed timeout (15s) rather than the probe's own
      (3s) — the wrong number on the surface built to answer that question; a 404 from Ollama
      was explained as a context-limit problem instead of "the model was never pulled"; and
      `/healthz` now returns the same report as the tool and the CLI, so three surfaces
      cannot disagree about one machine.
      **Left alone deliberately:** `check-weights` stays. CI runs it on a runner with no
      Ollama and no database, where a full health check would fail on text search; the two
      answer different questions.

- [x] **2.4 Structured findings array replaces free-text `analysis`** (defect 07). verdict,
      summary (≤240), closest_precedent, findings[] (severity enum ×3, category, finding
      ≤120, detail, precedent{campaign_id|rule_id, quote}, fix ≤120), resolved[]. Caps and
      enums enforced
      server-side. Clean cutover.
      **Done:** a judgment is now `verdict` + `summary` + `findings[]`, with every field a
      model can write into bounded — `summary` 240, `finding` 120, `fix` 120, `detail` 600,
      `precedent.quote` 300, `approve_if` 240, `resolved.was/now` 120, `category` 40
      (lower-cased so it can be grouped), and at most 12 findings, because thirty capped
      lines is an essay built out of bricks. Three consistency rules the prose version could
      not enforce, since nothing could count the findings: `revise`/`reject` needs a finding
      above a note (three notes was the same hedge one level down), `approve` cannot carry a
      blocking one, and a `guardrail_breach` cannot be a note. Findings sort most-severe-
      first and the sort is stable, so deliberate ordering within a severity survives; each
      carries an id (`{evaluation_id}#n`) so a later version can say which finding it closed.
      `get_evaluation(evaluation_id, severity=, kind=)` is a new tool — "the detail is
      fetched on demand" had been true of the storage and false of the surface, so "show me
      the blocking items" worked only inside the session that produced them. The save
      response hands back the blocking and should_fix lines themselves rather than a note
      telling Claude to go and offer them, which competed with the user's actual request and
      cost a round trip. `list_evaluations` carries verdict and counts, so "which of these
      still need work" is answerable from the list.
      **Fields defined here, behaviour owned elsewhere:** `kind` (6.2), `approve_if` (6.5),
      `evidence` (6.6), `provenance` (7.6), `basis` (7.8). This schema had no callers, so
      the columns were free today and a migration of every stored judgment later. `basis`,
      `evidence` and `provenance` are *not* writable from the MCP surface: 7.8's premise —
      a computed finding is identical for every user, so a difference is a bug — holds only
      if the **server** computed it, and a model that read a missing date did not compute
      it. `precedent` takes `campaign_id` **or** `rule_id`, because a guardrail breach cites
      the rulebook (§12) and had nowhere to cite it.
      **The blocker both reviewers reproduced independently:** the cutover was fine on a
      fresh database and impossible on a real one. `ALTER TABLE ADD COLUMN` cannot relax the
      legacy `analysis TEXT NOT NULL`, so after migrating, every save died on an
      `IntegrityError` — which is not a `ValueError`, so it reached the marketer as "Error
      executing tool save_evaluation" with the reason discarded. Invisible to all 419 tests
      because every one of them starts from a fresh schema. `_migrate_schema` now rebuilds
      the table (taking the target DDL from a scratch database built by `_SCHEMA` itself, so
      the rebuilt table is by construction the one a fresh install gets), carrying the
      existing essays across: they are the evidence for defect 07. A pre-cutover judgment
      reads back as `original_analysis` + `schema: "legacy"` through both `get_evaluation`
      and `reconcile_evaluation`, which had been reconciling against an original it had
      silently dropped.
      **Also from review:** `"proceed/revise/reject"` in `prepare_evaluation`'s note taught
      a word the verdict enum rejects; caps were measured after stripping in one field and
      before it in another; a non-string `finding` escaped as an `AttributeError`; a dict
      passed as `findings` was iterated as keys and blamed the wrong thing; the approve-with-
      blocking message offered "or lower the severity" as an equal exit, which is one token
      against rewriting the summary.
      **Not done here, named so it is not assumed:** the quote is bounded but not checked at
      all — 6.1 checks it against the record it cites, and D77 still owes the separate
      question of whether it was inside the retrieval window. (This paragraph used to assert
      that 6.1 "needs 7.2's server-owned retrieval first, because `prepare_evaluation` is
      stateless". That was wrong, and 6.1 says why: the statelessness argument applies to the
      window question and not to the fabrication one.) `evidence` and `provenance` are still
      whatever the caller passes (7.6), and `closest_precedent` is model-asserted (D8/7.2) —
      6.1 checks that its id resolves and any quote in it is real, which is a different thing
      from the server choosing it; `findings` has no per-evaluation golden set yet (7.7).
- [x] **2.5 Commentary layer: comments, annotations, speaker notes** (defect 08). PDF
      `/Annots` (Text/FreeText/Highlight/StrikeOut/Underline/Square/Caret/Ink) with
      `/Contents`, `/T`, `/M`, page index; PPTX `notesSlide` + `ppt/comments/` +
      `ppt/modernComments/`. Stored as `commentary[]` `{page, author, date, text, kind}` —
      a distinct layer from `deck_text`, embedded but tagged so retrieval can weigh it.
      **Done:** `extract.extract_commentary()` reads all three sources; PPTX reviewer
      comments come straight out of the package (`ppt/comments/` and
      `ppt/modernComments/`, authors resolved from `ppt/commentAuthors.xml` /
      `ppt/authors.xml`) because python-pptx has no API for them. `/Link` is excluded on
      purpose — the reviewer noted every other deck in the library carries link annotations
      only, and a hyperlink is not an opinion; indexing them would put URL fragments into
      the evidence a judgment cites. An empty `notesSlide` is not a note: PowerPoint creates
      one the moment anything touches a slide, so the common case is blank, and a library of
      empty commentary rows dilutes every retrieval it appears in.
      `campaign_chunks` gains `kind` (`body`|`commentary`) and `source` (JSON `{kind, author,
      date, anchor, page|slide, reply_to}` — the position as a number as well as a display
      string, so something downstream can sort or group by where in a deck a remark sits). One chunk per comment, never merged: two notes packed together would
      share one author and one anchor, and the anchor is half of what makes a comment worth
      keeping. Every hit carries `matched_kind`, plus `matched_author`/`matched_anchor`/
      `matched_date` when commentary matched, so a reviewer's objection cannot be cited as
      something the brief itself claimed. `include_commentary=False` answers "what does the
      brief say" as opposed to "what did people think of it".
      **Measured against the review's own number.** The reviewer probed a note that was not
      indexed and got 0.63, matching the human-written summary instead: "had the note been
      ingested, a near-verbatim query would score above 0.9." Over the real MCP protocol, on
      a deck carrying two speaker notes and one tracked client comment: the near-verbatim
      note query returns **0.9929**, `matched_kind: commentary`, `anchor: slide 1`; the
      client-comment query returns 0.8836 attributed to *Dana Ruiz*; the same query with
      `include_commentary=False` drops to 0.4361 on a body chunk, which is what proves the
      layers are genuinely separate rather than nominally tagged.
      **From design review, in this item:** a finding can now cite a comment and has to say
      so — `precedent` carries `layer` (`body` | `commentary`, defaulting to body) plus the
      author and anchor. The layer rule had been stated on the browsing tool and absent on
      the judging one, and a commentary chunk IS a retrieved chunk, so a client's "I do not
      think the timeline is realistic" was a fully compliant citation whose stored form read
      as something that campaign's deck claimed. `commentary_checked` answers "did anyone
      look" separately from "were there any" — the exact ambiguity `images_checked` exists
      to fix, re-created in the same function. A threaded comment is now one item per
      utterance with a `reply_to`, because a returned deck's thread is typically a client
      remark and the agency's reply, and joining the runs recorded the agency's answer as
      the client's words. The classic `ppt/comments/` format uses `<p:text>`, not
      drawingml `<a:t>`, so the older format had been parsing to empty and dropping every
      comment while the docstring claimed both. A PPTX comment's anchor is now `"deck"`
      rather than a slide number scraped from the part filename, which is the comment part's
      ordinal and not the slide's — an anchor that is sometimes silently wrong, presented in
      the same field and with the same confidence as a PDF page index, is worse than one
      that admits it does not know. `include_commentary` also takes a list of kinds, so
      "what did the client say" is askable rather than only "what did anyone write".
      **Decided against:** exposing `include_commentary` on `prepare_evaluation`. A judgment
      should weigh every recorded objection, and a switch that drops the layer carrying the
      client's is an easy path to a cleaner-looking approval — 2.4's lesson was that an
      error message offering an easy exit gets taken, and the same is true of a parameter.
      **From adversarial review, in this item:** excluding commentary could delete a
      campaign from the results entirely. The layer filter ran after the vector search's
      over-fetch window and after the per-campaign rollup, so a deck whose 25 speaker notes
      filled the window lost its body chunk before the filter ever saw it — the same
      starvation the structured-filter path already carries a comment about, one layer down.
      Narrowing by layer is a filter, and now happens where the other filters happen: it
      narrows what can MATCH, it does not delete a record. The first version of the test
      asserted `hit == [] or ...`, which blessed the bug; the reviewer's reproduction
      replaced it. Also: a deck whose body extracted to nothing reported its notes as found
      and then threw them away, because the "nothing to embed" early return fired before the
      commentary was stored; `D:2026` became `2026--` and `D:99999999` became `9999-99-99`,
      a well-formed-looking date no calendar contains, because the string was built before
      being validated — an unparseable date is now kept verbatim, since it is still evidence
      of when somebody said something and an invented one is worse than none; and a bare CR
      in a PDF annotation rendered as one run-on line in the excerpt a person reads.
      **Not gated on `not deck_text`** — the trap the image extractor fell into, which
      silently skipped extraction on the documented demo flow. Claude passing deck_text says
      nothing about whether the notes were read, and the realistic call passes both.
      Extraction never raises: a malformed comments part costs the user a warning, never the
      deck they actually uploaded. Capped at `MAX_COMMENTARY_ITEMS` (200) like every other
      caller-sized loop.
      **Migration:** `campaign_chunks` gains both columns additively, existing chunks
      default to `body`, with the pre-2.5-database test 2.4 taught us to write. Fixed while
      here: `_migrate_schema` crashed on a table that does not exist — harmless in every real
      install, because `init_db` runs `_SCHEMA` first, which is exactly why nobody noticed.

## Phase 3 — P2 defects

- [x] **3.1 Structured warnings** (defect 09). Stable `code`, one-line human `remedy`, raw
      `detail` — so a surface can say "Visual search is offline — ask IT to run setup".
      **Done:** `notices.py` (named that way because `warnings` is a standard library module
      and shadowing it from the project root would break any import of the real one) holds a
      registry of `code -> (severity, remedy)`, and every warning site in `core.py` and
      `extract.py` now emits `{code, severity, remedy, detail}`. An unregistered code raises
      rather than producing a blank remedy — a warning with no remedy is one that quietly
      reverted to being engineer-only, which is the defect itself.
      `severity` was added on top of what the review asked for: `blocked` (a capability is
      off until somebody acts), `degraded` (this record is incomplete, the product works),
      `note`. "Visual search is offline" and "one image of forty was skipped" are not the
      same news, and a flat list of strings made every surface treat them identically.
      Three smaller review findings were carried for one round and closed here: a folded
      warning now reads as plural when it is plural (`count` was stored and no text ever
      consumed it, so six unreadable images said "An image in the deck could not be saved");
      `prepare_evaluation` now carries `results_may_be_incomplete`, since the one surface
      where a half-indexed library matters most — a verdict about to be saved against that
      evidence — was the only one staying silent about it; and a legacy `.ppt` no longer
      emits two warnings with slightly different advice for one condition.
      `notices.collapse()` folds repeats of one code into a single entry with a `count`,
      because a deck with twenty unreadable images produced twenty near-identical lines,
      which is how the one warning that mattered got scrolled past.
      Which code an image failure gets depends on WHY: the model being absent is the
      operator's problem to fix once (`visual_search_offline`), while one bad PNG on a
      working model is this record's problem (`image_not_embedded`). Telling a marketer to
      "ask IT to run setup" over a single corrupt image would be the same mistake pointing
      the other way.
      **From review, in this item.** The first version put three readers' text into one
      `remedy`, so a marketer was read Claude's own stage directions — "Tell the user that,
      and offer to..." — because the docstring said to say the remedy verbatim. One field per
      reader now: `affects` (what the user loses), `remedy` (what a person does), `next_step`
      (what Claude does, never read aloud), `detail` (support). `scope` was added alongside
      severity, because the marketer's real question is not how bad it is but whether it is
      theirs to fix, IT's, or already handled — and severity alone had put a blank title in
      the same class as a missing model.
      `collapse` folded on the code alone, which lost information twice: two
      `indexing_incomplete` entries — one about images, one about the deck's sections —
      became the image one with a count of 2, on a response whose own counts said no section
      was indexed; and four chunks failing for two reasons reported one reason and a count of
      four. It now folds on the whole user-facing message and appends distinct causes, and
      sorts worst-first so ordering is a property of the response rather than an instruction
      in one tool's docstring.
      Three remedies made claims the response contradicted. "Reuse checks will miss this
      one" was said while flagging a reuse at hamming distance 0 — perceptual hashing does
      not touch the vision model (`health_check` was making the same false claim, found
      transitively, and is fixed too). "Ask IT to run setup" named a gesture that exists
      nowhere: there is no setup script, and nothing in `run.sh` or `run.ps1` fetches the
      weights — `WeightsResolution` already computes the right remedy per cause and the
      registry was discarding it. And a PNG sent alongside `deck_text` was told "nothing in
      it is searchable", on a fully searchable record, with advice to convert a PNG to PDF.
      A dead text embedder was reported once per chunk, each line advising `finish_indexing`
      — which then fails every item and says calling again will not help. `embedding` now
      raises a distinguishable `Unavailable` for transport failures (a type, not a message
      match, because rewording a message must not be a breaking change), and the text path
      makes the same outage-vs-item split the image path already had.
      `find_similar_images` and `finish_indexing` were still handing raw exception text to a
      marketer — the sibling tools of the one defect 09 was reported against. Both now speak
      the shape. The contract itself moved from `upload_campaign`'s docstring to the server
      `instructions`, so it reaches every tool rather than only the one somebody read first.
      **Verified on a simulation of the reviewer's own machine** — weights path invalid,
      Hugging Face cache empty, `HF_HUB_OFFLINE=1` — the upload now returns "Visual search
      is offline, so image similarity and reuse checks will miss this one. Ask IT to run
      setup on this machine — everything else works normally.", with the original "Failed to
      download weights for tag 'openai'" text intact in `detail`.
- [x] **3.2 Version surface + stale-schema documentation** (defect 10). Build version in the
      server description; README documents that the host app must be fully quit and reopened
      after a rebuild (closing the window leaves the server running).
      **Done:** `version.py` is the single source — `VERSION` by hand at release, `BUILD`
      stamped by CI with the commit and date, because two builds of one version from
      different commits are not the same thing to anybody debugging one. It imports nothing
      and touches no I/O beyond one optional file read, so `--version` answers on a machine
      where the database is corrupt, the embedder is down and the model is missing, which is
      exactly when somebody asks. Surfaced in the server's own `instructions` (where a host
      shows it, next to the connector), in `health_check`, in `/healthz`, and as
      `campaign-intelligence --version`.
      **The half the review asked for is the weaker half.** A version string says the server
      changed; it does not say which of the parameters in your hand are missing, and the
      reported symptom was silent absence — "several parameters that were live and working
      were absent from the schemas in use, and had to be rediscovered by trial and error".
      So `health_check` also publishes `tools`: every tool mapped to the parameters it
      actually takes, read off the function signatures rather than maintained by hand, since
      a hand-written copy would drift from the tools exactly the way the client's cache did —
      the defect reproduced inside its own fix. `core.stale_schema_parameters()` diffs a
      caller's schema against it, so the gap gets named instead of guessed at, and the server
      `instructions` tell Claude to do that and then say to fully quit and reopen the app.
      The tool inventory is best-effort: an import failure costs the inventory, never the
      report that says which component is down.
      **From review, in this item.** The first version iterated a hand-typed `TOOL_NAMES`
      tuple — the hand-maintained copy this very item says it refuses to have. It matched on
      the day it was written, and a tool registered without editing it would have been
      omitted silently: defect 10 reproduced inside its own fix. The reviewer deleted a name
      from the tuple and the entire suite stayed green, because the test guarding it compared
      the registry against a set derived from the registry. The inventory now comes from the
      server's own tool registry, and the test compares against what MCP actually advertises.
      Two more version copies were wrong. `build.sh`/`build.ps1` never stamped, so two local
      builds were indistinguishable and a frozen PyInstaller build called itself a "source
      checkout"; both stamp now. And `AppVersion` was hard-coded in the Inno Setup script and
      had **already drifted** — 0.2.7 against `version.py`'s 0.3.0 — so a release would have
      shipped an installer whose Add/Remove Programs entry, uninstall display and upgrade
      detection all disagreed with the binary inside it. CI passes it in with `/DAppVersion`.
      **And defect 11's own bug class was still shipping.** `configure-desktop` read and
      wrote Claude Desktop's config with no `encoding=`, so on Windows it used cp1252 against
      a UTF-8 file: another connector pointing at `C:\Users\José` came back as
      `C:\Users\JosÃ©` and was written back corrupted, and a byte cp1252 cannot decode
      raised an uncaught `UnicodeDecodeError`. It runs on every Windows and Linux install.
      Fixed there and in both uninstallers' embedded Python, which had the same hole under a
      legacy locale.
- [x] **3.3 Shipped scripts must be encoding-safe** (defect 11). Any `.ps1` ASCII-only or
      UTF-8 **with** BOM, asserted in CI; audit the macOS/Linux shell scripts for the
      analogous trap rather than assuming it is Windows-only.
      **Already confirmed present while working item 1.2** — `run.ps1` is BOM-less and
      contains em dashes, including **line 42 inside a double-quoted string**
      (`Write-Host "Ollama not found — downloading..."`), which is precisely the reported
      failure: PowerShell 5.1 decodes it as Windows-1252, the third byte becomes a closing
      curly quote, the string terminates mid-sentence and every brace after it mismatches.
      `installer/windows/campaign-intelligence.iss:1` also has one (in a comment, so
      harmless, but it should not survive the CI check either). `build.ps1` is clean.
      **Done:** the mechanism was confirmed byte for byte before fixing anything — `e2 80 94`
      on `run.ps1:42`, decoded as cp1252, yields `â€”`, and the `”` is a character PowerShell
      accepts as a closing double quote. Both files are now ASCII, chosen over a BOM because
      a BOM makes the file correct while leaving the next person free to paste in a smart
      quote; ASCII is a property CI can state plainly. `tests/test_shipped_script_encoding.py`
      asserts it over every `.ps1`, `.cmd`, `.bat` and `.iss` in the tree, and runs in CI on
      every push — the review's own note that "this passes on a developer machine" is exactly
      why it is a test and not a habit.
      **The Unix audit the item asked for, rather than assuming Windows-only:** the encoding
      half genuinely is Windows-only, since sh reads bytes and a UTF-8 locale handles the
      rest. The SHAPE of the defect is not — a script that parses where it was written and
      not where it runs — and on Unix that is line endings: a CRLF turns the shebang into
      `/usr/bin/env bash\r`, and the kernel then reports "bad interpreter: no such file or
      directory" about a file that plainly exists. All four `.sh` files are clean and now
      tested, along with a shebang check.
      **`.gitattributes` added** for the way those bytes change with nobody editing the file:
      a checkout with `core.autocrlf=true` rewrites every LF, so the shipped scripts are
      pinned per-interpreter (`eol=lf` for `.sh`, `eol=crlf` for the Windows ones) rather
      than left to local git config.

## Phase 4 — Installer acceptance criteria

- [x] **4.1 Post-install self-test runs `health_check`** and a non-green result blocks the
      success screen, naming the failing component — all three platforms. The binary's
      `check-weights` subcommand (added in 1.2) is the seam; Linux already verifies the
      weights sidecar at install, so what remains is Windows/macOS parity and widening it
      from weights to every component once 2.3 exists.
      **Ownership note:** "install fails loudly if weights are absent or corrupt" is
      delivered by 1.2 (CI-side + Linux installer) and completed here (all platforms,
      blocking the success screen) — 1.3 covers the zero-egress *proof*, not the install
      gate.
      **Done:** `health-check` gained `--json`, because an installer parsing prose is an
      installer that breaks when the prose improves. All three platforms now run it as the
      last step and refuse to report success over a failure.
      *Linux* previously verified the weights checksum and nothing else; it now ends with
      the self-test and exits non-zero, which a test proves by executing the real script
      against a stub binary that fails the check.
      *macOS had no installer at all* — the release shipped a tar.gz and left the user to
      work out where to put it. It has one now, with the same checksum verification, the
      same gate, and the Gatekeeper quarantine flag stripped (left in place, the first
      launch is a "developer cannot be verified" dialog that nobody connects to the deck
      they just tried to upload). Found while writing it: CI was copying the **Linux**
      installer into the macOS archive, so a Mac user got systemd units and
      `~/.local/share` paths.
      *Windows* verified nothing. Ollama, the Claude Desktop wiring and the self-test moved
      into `[Code]` so the gate could run before the wiring — a `[Run]` entry's exit code is
      ignored by Inno, so nothing there can gate anything.
      **Two claims in the first version of this entry were wrong, and the review corrected
      both from Inno Setup's own source.** `[Run]` entries do *not* execute after
      `ssPostInstall` — `Setup.MainForm.pas` calls `ProcessRunEntries` (line 234) before
      `SetStep(ssPostInstall)` (line 241) — so ordering alone never required the move. And
      `RaiseException` at `ssPostInstall` does **not** abort setup: that step is invoked with
      `HandleExceptions = True`, which logs the exception and calls
      `Application.HandleException`, swallowing it. Execution continues to the Finished page
      with exit code 0 — precisely the "message dismissed, success page shown anyway"
      behaviour the gate was supposed to replace. Only `ssInstall` re-raises, and an
      exception there rolls the files back, contradicting the advice to fix the problem and
      re-run against what was just installed.
      What is achievable, and is what a marketer actually needs, is done instead: the failure
      is recorded, Claude Desktop is **not** connected, and the Finished page is rewritten to
      *"Installed, but not working"* naming the component. See **L5** in
      [`DEFERRALS.md`](DEFERRALS.md) — Inno has no seam for "keep the files, fail the run",
      and claiming one would be the overstatement this whole exercise is about.
      Also fixed: `RunAndCapture`'s `cmd /C` quoting closed the outer quote before the
      redirect, so `>` sat inside quotes, the exe received one argument, exited 2, and no log
      was written — after which `LoadStringsFromFile` returned False without raising and the
      dialog named nothing at all. And every step ran elevated under
      `PrivilegesRequired=admin`, so in the case this product is built for — IT installing for
      a marketer — the model was pulled into the *admin's* Ollama, the *admin's* Claude
      Desktop was wired, and the self-test passed against an environment the marketer would
      never see. The install is now per-user, like the rest of the product.
- [x] **4.2 Zero-egress install verified** on a host with egress disabled (actually tested,
      not "degrades gracefully").
      **Done, and the first two attempts at it were worthless.** Injecting a socket guard through
      `sitecustomize`/`PYTHONPATH` does nothing to a frozen binary — PyInstaller controls
      `sys.path`, so the guard never loads and the test passes by doing nothing. CI now uses
      real blocks against the packaged product: an empty network namespace (`unshare -rn`)
      on Linux, and every HTTP request routed to a closed port on macOS. The library-level
      proof runs in the suite too, blocking `socket.connect` and then LOADING the vision
      model — not merely resolving its path, which is the weaker check that passes in
      milliseconds without touching the network either way.
      The *second* worthless version was the CI half. It ran `check-weights`, which only
      stats a path, so it would have passed offline whether or not the product worked; the
      macOS step also set `HF_HUB_OFFLINE=1`, which tells the client not to try, making the
      proxy block prove nothing. And `unshare -rn` fails outright on `ubuntu-latest` (24.04
      restricts unprivileged user namespaces via AppArmor and the runner images do not relax
      it). All three platforms now run the packaged binary's full `health-check --json`,
      which loads the vision model, and assert on `visual_search`; Linux uses `sudo unshare
      -n`, and Windows — the platform the review was written against, on a network that
      blocks huggingface.co — had no offline check at all and now has one.
- [x] **4.4 macOS: sign, notarize, staple - and ship a `.pkg`** (raised by the Phase 4
      design review; not in the original review, and owed). Today the macOS archive is an
      unsigned binary whose quarantine flag the installer strips. That strip is a legitimate
      stopgap *inside an installer the user chose to run*, on the product's own directory —
      but it ships "we are not signed" as the plan of record, and on macOS 15+ an unsigned
      quarantined binary launched by Claude Desktop is blocked outright rather than offering
      "open anyway". The marketer never gets that far in any case: the tarball needs Terminal
      to extract, `install.sh` is itself quarantined, and Finder opens a `.sh` in a text
      editor. What is owed: Developer ID signing + notarization + stapling in CI, and a
      `.pkg` whose postinstall runs the same gate — after which the quarantine strip can go.
      (§4.1's entry above still says the quarantine flag is "stripped"; it was not, and §4.4
      is where that is established. Read them together.)

      **Done, and the quarantine strip did not work.** Measured before changing anything: the
      installer's `xattr -dr com.apple.quarantine "$DEST"` stripped the LIVE directory, while
      everything it runs — `init`, the `health-check` gate — runs out of `$STAGE`, which `mv`
      then puts in place. On a fresh install `$DEST` does not exist, so the strip was a no-op
      swallowed by `|| true`; on an upgrade it stripped the copy about to be deleted. Either
      way the installed binary and the CLIP weights beside it were still quarantined. So the
      position written down here — "a legitimate stopgap inside an installer the user chose to
      run" — was not the position the product was in. Fixed, and pinned by a test that installs
      a quarantined bundle and reads the attribute off the installed copy.

      **Done: the `.pkg`.** `installer/macos/pkg/` — payload staged through /usr/local, a
      postinstall that drops to the console user and runs `install.sh`, and nothing else. Thin
      on purpose: the gate is `install.sh`'s and a postinstall repeating any of it would be
      the second implementation of one rule, which is the shape this codebase keeps being bitten
      by. `require-scripts="true"`, so the gate cannot be skipped by a payload-only install;
      the staged copy is removed on success AND on failure, so a refused install does not leave
      2GB of weights as a souvenir.

      **Done: signing, notarization and stapling in CI** — `build-pkg.sh`, conditional on
      credentials and saying which it did. It signs every Mach-O in the payload and not only
      the package: a signed `.pkg` around an unsigned binary notarizes and then fails on
      launch, because the package's signature says nothing about what is inside it.
      `--options runtime` on every call, without which notarization is refused, and `stapler
      staple` after — unstapled, the first launch asks Apple whether the package is notarized,
      on a network §4.2 says may not allow it.

      **Done: D25, one platform wider than it asked.** The Windows installer was compiled in
      CI and never run, so its `[Code]` gate had never executed anywhere. Both installers are
      now installed end to end on a clean runner, the product they leave behind must pass its
      own self-test and must be wired to Claude Desktop; and both are built a second time
      around a deliberately corrupted bundle, which must be REFUSED and must leave Claude
      Desktop unwired. The Inno script gained a `SourceDir` define so that second build is
      possible at all.
      **Measured, not merely written.** Five dispatched runs later, both installers install
      end to end on a clean runner and both refuse a corrupted bundle:

          installer: The install was successful.
          installed and green: ['computed_facts','database','rulebook','text_search',
                                'visual_search']
          and wired to Claude Desktop
          ...
          FAILED: the CLIP weights are corrupt or incomplete. Visual search would be dead.
          refused, said why, and wired nothing

      That is §13.1's definition of measured, and D25's own sentence — "compiling it is new but
      is not the same thing" — finally satisfied for its own fix. What the runs cost was five
      defects, four of them mine: no diagnosis on failure; an expectation that Ollama could
      install on a machine with no GUI session; a refusal check reading the previous install's
      leftovers; and twice, an expected non-zero exit leaking out as the step's verdict. The
      real one was the installer running a SECOND installer inside a silent install, which is
      what hung the first dispatch for 99 minutes.
      What the first refusal check asserted could not have worked, and review caught both
      halves. On Windows a refused install exits ZERO by design: the `.iss` explains that Inno
      discards an exception at `ssPostInstall`, and that raising at `ssInstall` rolls back the
      files the user is told to fix and re-run against. Asserting a non-zero exit would have
      failed every release build while proving nothing. On macOS the check grepped for
      `FAILED`, which the postinstall prints whatever goes wrong — so a permission error would
      have passed as "refused the corrupt weights". Both now assert the promise the `.iss`
      actually makes: **never wire a product the self-test rejected**, with a positive control
      so an installer that wires nothing at all cannot pass either.

      **Signing is an ACCEPTED LIMIT, not outstanding work (L11).** It needs an Apple
      Developer Program membership and a certificate only the account holder can issue, and the
      owner has decided not to buy one for now. Nothing is owed on the code side: `build-pkg.sh`
      signs every Mach-O in the payload with the hardened runtime, notarizes and staples, each
      step conditional on its own credential — set `MACOS_SIGN_IDENTITY`,
      `MACOS_SIGN_IDENTITY_INSTALLER` and `AC_NOTARY_PROFILE` as repository secrets and the same
      build signs with no code change. Until then the release ships an unsigned `.pkg`, which
      installs, and README says what Gatekeeper shows and the three steps past it — strictly
      more than the tarball allowed, since Finder opens a `.sh` in a text editor. The quarantine
      strip stays while that holds; it is inert on the `.pkg` path either way" **on the tarball
      path**. On the `.pkg` path the strip is inert — a pkg payload is never quarantined, as
      the CI comment says — and the dialog a marketer meets there is on the package itself,
      before anything is installed, where no strip can reach. That is what D130 is for, and
      until then README says what they will see and what to click. *Closes D25; the signing
      half waits on D130.*

- [x] **4.3 Ollama verified at install** to the same standard as the vision model — daemon
      reachable and `nomic-embed-text` present.
      **Done:** `health_check` already separates "not reachable" from "the model is not
      installed" — a running Ollama with nothing pulled answers on the socket and 404s every
      embed, which is how text search was dead while everything looked fine. What was
      missing was anything acting on it: a failed `ollama pull` was logged as a warning and
      the install carried on. The pull may still fail; what has changed is that the
      self-test runs afterwards and turns it into a refused install. Pinned by a test that
      asserts the ordering, since a gate placed before the thing it gates is decoration.
      **From review:** the gate got exactly one shot at a daemon started three seconds
      earlier. A cold model load measured 0.7s here and has never been measured on the
      Windows laptop this review came from, with antivirus scanning a freshly written 274MB
      file — and refusing an install because a service was slow to warm up is a refusal
      nobody can act on. `health-check --wait 60` retries, but only while the *only* thing
      wrong is transient: a missing checkpoint does not appear by waiting for it, and taking
      a minute to say so again is worse than saying it at once.

## Phase 5 — Making it intuitive (ideas A–F)

- [x] **5.1 (A) Forgiving enums, teaching errors.** Every enum error returns the valid set
      and the closest match; normalise on the way in ("client stated" → `stated`).
      **Reproduced first — and my reading of one case was wrong, caught in review.** I
      claimed `Union[str, TagObject]` reported only its first branch and never named
      `source`. It names both. My own probe printed *"2 validation errors"* and I truncated
      the output to 300 characters before reading the second, then stated the conclusion as
      reproduced fact in four places. The change is still worth having — a 66-character
      message naming one field beats a two-branch dump whose first line tells a marketer
      their object should be a string — but for that reason, not the one I gave.
      The bare-string case was fixed on the FILTER only: writing `tags="liked"` still failed
      with *"Input should be a valid list"* until this round.
      **Done:** `enums.py`, in three layers because they are three different claims. *Shape*
      — case, spacing and punctuation are not meaning, so "In Flight", "in-flight" and
      "in_flight" are one value and no synonym table has to list all three. *Synonyms* — an
      explicit table a person can read and argue with, covering what a marketer actually
      types ("client stated", "live", "done", "a pitch"). *Teach* — the valid set always,
      plus the closest match only when something really is close, since a suggestion that is
      not close just sends somebody to retype a word that will also be rejected.
      The values are normalised identically on the **filter** side, because a value the
      library accepted that cannot then be used to search for itself is a trap rather than a
      kindness.
      **Deliberately not guessed:** `target` for a metric type — the reviewer's own example.
      A target is what somebody wants to happen and a prediction is what this library expects
      to happen; filing one as the other corrupts every later reconciliation, which exists to
      compare predictions against actuals. That one teaches, and says why. `goal` and
      `benchmark` get the same treatment.
      **The schema still advertises the enum** while the Python type is a plain string: the
      enum is what stops a well-behaved caller guessing in the first place, and the
      normalising is for when it guesses anyway. Without loosening the type, the value dies
      at pydantic's boundary before any of this code can see it.
      **Found while verifying over the protocol:** `add_metrics(confirm=False)` returned its
      preview *before* validating, so a marketer was shown `metric_type: "target"` as though
      it were about to be saved and the rejection arrived only after they agreed — and a
      normalised value was hidden from the one screen that exists for them to correct it.
      **From design review.** The same defect was still in the flagship flow: `upload_campaign`
      previewed the word the marketer typed while a different value went into the database,
      and the default status was computed from the RAW `record_type` while the write computed
      it from the normalised one — so `record_type="Campaign"` previewed `status: None` and
      committed `concluded`. The user agreed to one record and got another. Both preview and
      write now return `normalised: [{field, given, stored_as}]`, and the note tells Claude to
      say it: silent normalisation is acceptable only if every write says what it stored,
      because a guess nobody hears about is one nobody can correct. Shape changes are lossless
      and stay silent — "recording this as in flight" said to somebody who typed `in_flight`
      is the noise that stops the real ones being read.
      `bulk_import_metrics` was the one path still using the old strict check — the same
      vocabulary with two behaviours depending on how many rows you sent, on the path where a
      spreadsheet column headed "Results" or "Target" is most likely to arrive.
      Two synonyms were guessing meaning rather than shape and are gone: `past → concluded`
      is temporal, not lifecycle (a cancelled campaign is also past, and filing it as
      concluded counts it in every later "what worked" question), and `confirmed → verified`
      collided with the hard definition `verified` carries here — backed by a real `actual`
      metric row. "The client confirmed it worked" is a *stated* claim; the write path would
      have caught it, but the filter path has no such guard, so a query for "confirmed" would
      have silently narrowed to measured evidence.
      `cancelled` was getting *"Did you mean 'concluded'?"* — difflib measures string
      similarity, not meaning, and that suggestion would file an abandoned campaign as a
      finished one. It, `paused`, `on hold` and `killed` now get an explanation instead.
      **And the shared prompt was teaching the word the server refuses:** `add_metrics`'s
      docstring said "'predicted' (a forecast/**target** set before launch)", so Claude would
      map "our target is 2% CTR" to `predicted` on that authority and never reach the teaching
      error at all.
      **Scope, stated plainly:** this makes the vocabularies a *marketer* authors forgiving.
      The model-authored ones (verdict, severity, finding `kind`, `basis`, precedent `layer`,
      `include_commentary`) still return raw pydantic errors — deliberately, see below — so
      "every enum error returns the valid set and the closest match" is true of the ones this
      item is about, not of every enum on the surface.
      **The forgiving/strict split is now written down** rather than left looking like where
      the work stopped: vocabularies a *marketer* authors are forgiving; vocabularies the
      *model* authors (verdict, severity, kind, basis, layer) stay strict, because §2.4's
      lesson was that an easy exit in an error message gets taken, and auto-mapping "minor"
      to `note` would hand the model a severity downgrade path.
      **From adversarial review.** The suggestion layer was offering antonyms: difflib rated
      `unverified` at 0.89 against `verified` — the highest-scoring suggestion in the whole
      vocabulary and the most harmful one possible, since a caller retrying with it marks an
      unverified claim as measured evidence, on the single field this library weighs
      judgments by. The cutoff moved from 0.6 to 0.75 (measured: real typos score 0.82 and
      up, coincidences like `approved`/`proposed` 0.63 and `cancelled`/`concluded` 0.67 fall
      below it) and a negation guard suppresses any suggestion the input is the denial of.
      A blank value now consistently means "not saying" rather than clearing a set one — a
      spreadsheet row with an empty cell was erasing a status.
- [x] **5.2 (B) `next_actions`** on every result — `{label, tool, prefilled_args}`.
      **Done:** `actions.py`. The review named the three things anyone does after a judgment
      and all three are offered, prefilled: store the brief as a record, mark what it
      supersedes, and reconcile when the results land. Nothing *schedules* anything, because
      the product has no scheduler — so "schedule the reconciliation" is the reconciliation
      call itself, prefilled and labelled for when the numbers arrive, which is the honest
      version of the same offer.
      **Two rules, and most of the tests are about them.** An offer has to WORK: a suggestion
      whose argument name has drifted is one the user accepts and Claude then cannot carry
      out — an error they did not cause and cannot fix. Every offered tool is checked to
      exist, every prefilled argument to be one that tool takes, and one offer is executed
      end to end. And an offer has to be worth making: a list that is always there stops
      being read, and then the one that mattered is not read either. An empty list is a real
      answer — `find_similar_campaigns` returns one, because what somebody does after reading
      results depends entirely on why they searched. Three is the ceiling, which is the
      number the review itself used.
      Prefilled arguments are only ever values this library already holds. The user says yes
      to the *label*, so anything filled in is something they agreed to without being shown
      it; a placeholder or a guess would be written on their behalf.
      **Closes two tracker rows.** D16: warning `next_step` prose becomes structured actions
      (`next_step` survives only where no single tool sits behind the advice — "say this
      once, not per image"). D31: `enums.BadValue` carries `field`, `valid` and `suggestion`
      as data, and `_catch_value_errors` passes them through the tool boundary, so a retry is
      something a caller acts on rather than parses out of "Did you mean...?".
      **Opened one.** The obvious offer after a deck whose comments were never read — send
      the file — cannot be made: no tool attaches a deck to an existing campaign. The test
      that every prefilled argument must exist caught the attempt. Tracked as D39 rather than
      shipped as an offer that fails.
      **Found by the protocol round-trip:** the server `instructions` are an f-string, so the
      literal `{label, tool, prefilled_args}` in the new paragraph was read as an expression
      and the server would not start at all.
      **Both reviewers independently condemned the supersession offer, and it is gone.** It
      was prefilled from `closest_precedent.campaign_id` — a similarity match the *model*
      asserts, which the server never computes or validates. "Mark this as replacing the
      version it revises" is a claim about somebody's intent; a similarity score is a fact
      about text. The user hears only the label — "add this to the library" — while agreeing,
      unseen, to hide another record from every future search, and `update_campaign` cannot
      clear `supersedes`, so a wrong yes is undoable short of deleting the campaign. The gate
      was inverted too: it suppressed the offer on approvals and permitted it on **rejects**,
      so rejecting a proposal offered to file it as the replacement of the concluded campaign
      it happened to resemble — metrics and all. Demonstrated over the protocol: accepting it
      removed the precedent from `find_similar` entirely. The `campaign_id` branch was worse
      still — it offered `update_campaign(supersedes=…)`, an argument that tool does not take,
      and the SDK *drops* unknown keys, so the user got a success-shaped response and nothing
      happened. That branch existed because the test fixture never passed a `campaign_id`,
      which is precisely how an untested branch ships. Deferred to 6.3 on real evidence (D41).
      **The reconcile offer moved to where it can succeed.** Offered at save time it failed
      on acceptance — there are no actuals yet. An offer with a temporal precondition is a
      reminder, and this product has no reminder channel; it is offered when results are
      recorded against a campaign with an open judgment.
      **Every action now carries `consent`.** Three rules were in play for one action:
      `next_step` said "act on it, never read it out", `next_actions` said "only if the user
      accepts", and `finish_indexing`'s own docstring says "do not stop to ask". Writing a
      record (`ask`) and resuming an interrupted index (`do`) are not the same kind of act.
      Actions also carry `why` — the library fact that prompted the offer — and `needs`,
      where accepting genuinely is not one step.
      **An empty metrics row is refused.** The `add_metrics` offer prefills only
      `campaign_id`, so accepting it with `confirm=True` stored a content-free `actual` row —
      which then satisfied the gate that lets a performance tag be marked `verified`. One yes
      away from the provenance this library weighs judgments by, for a row measuring nothing.
      **And the commonest upload got nothing.** `after_upload` was handed the caller's
      `status`, which is None when omitted, while the store defaults it to `concluded` — so a
      deck uploaded without a status, the most ordinary case there is, was the one path that
      offered no results prompt. Read back from the record now.
      Also: a duplicate title warns at the moment it is created (the offer is what creates
      them, and the product's own title lookup then reports the record as ambiguous);
      `collapse` keys on the actions too, so two warnings differing only in which campaign
      they repair no longer fold into one and lose an offer; and the "Add to the library"
      label became "Review and add", because `upload_campaign` previews unless `confirm=True`
      and the preview *is* the consent step for a write.
      **Mutation testing found four tests that could not fail:** `trim` returning everything,
      `MAX_ACTIONS` at 30, `after_upload` always offering, and the broken `campaign_id`
      branch. All four now have tests that bite.
- [x] **5.3 (C) `gaps()`** + a standing line on every evaluation naming the single most
      valuable missing input for that judgment.
      **Done, as two halves that answer different questions.** `gaps()` is about the LIBRARY:
      what is missing across everything, ranked, so somebody can go and fix the biggest hole.
      `most_valuable_missing_input` on `prepare_evaluation` is about ONE judgment: of
      everything missing, the single thing that would most change *this* verdict. They
      routinely disagree, which is the point — a library that is mostly measured can still
      produce a verdict resting entirely on the part that is not.
      **Ranked, and the ranking is the design.** An unordered list of everything absent is
      exactly the thing nobody reads, which is the state the review was describing. The order
      is by how much closing the gap changes what the library can answer: empty library,
      then nothing measured anywhere, then a whole market with nothing measured, then records
      stored but not searchable, then decks whose commentary was never read.
      **"Single" is load-bearing on the judgment line.** Several things are always missing;
      naming them all is the behaviour being replaced. It is `None` when nothing is missing,
      which is what makes it worth reading when it appears.
      Every gap carries `what` (the fact), `why_it_matters` (what it costs) and
      `next_actions` that would close it — a gap with no action is a complaint.
      **An empty library reports one thing.** Every gap is present in an empty library, so
      listing them is useless; a new install has one thing to do and it is not "fix your
      LATAM coverage".
      **Only records that had a deck can have unread commentary.** A campaign entered as a
      title and a description never had comments to read, and reporting it would be the noise
      that stops the real ones landing.
      **Closes two tracker rows.** D17: warnings live for one response, so "this deck's
      comments were never read" was unrecoverable the moment the upload returned —
      `campaigns.commentary_checked` persists it. D45: `results_may_be_incomplete` carried
      `finish_indexing` as prose while a single tool sat behind it, and `prepare_evaluation`
      returned no actions at all.
      **Kept deliberately distinct from 6.6.** The evidence-strength line says what a
      judgment RESTS ON — how many precedents, how many concluded, top similarity. This says
      what would most improve it. One describes, the other asks.
      **Narrower than the idea, and that is now recorded rather than implied.** The review
      gave three examples; this covers one. "No LATAM store launch has ever carried a budget"
      is about a missing FIELD within records and nothing detects a budget (D49 → 7.1); "the
      KPI workbook named in its own rubric has never been supplied" needs a rulebook that
      declares its expected inputs (D50 → 12.1). The test that quoted the budget example was
      testing something else and has been re-attributed.
      **From design review.** The standing line lived only on `prepare_evaluation` — two
      calls before the verdict a user actually hears, with a note asking for it to be
      repeated. This project's own principle, stated three lines above that return, is that
      models mirror the shape of a tool result far more reliably than they follow
      instructions inside one; a line delivered early and asked to be carried forward is the
      thing that gets dropped. It is recomputed at `save_evaluation` from `cited_ids`,
      returned there, and persisted in `evidence` so `get_evaluation` and §7.6's stamp can
      recover it. Computed by the server, not accepted from the model, for the same reason
      `evidence` and `provenance` are.
      `commentary_never_read` is recorded but **no longer reported**: its only available
      offer was `upload_campaign`, which creates a second record and fires `duplicate_title`
      — the exact offer §5.2 refused in writing one commit earlier, because no tool attaches
      a deck to an existing record. A gap whose only action makes things worse is a
      complaint, which this item's own rule forbids (D51 holds it until D39 lands).
      A **forecast was counting as a measured outcome**: `has_metrics` counts any metric row,
      so a `predicted` figure silenced both the gap and the judgment line — and a forecast is
      the opposite of an outcome, being the thing reconciliation later scores against the
      actuals. And a **campaign that has not run was counted as missing its results**, which
      is a request nobody can satisfy, on the highest-ranked gap; `after_upload` already drew
      that line and this did not.
      The migration **manufactured a gap for every record it had already read** —
      `commentary_checked` defaults to 0, so everything ingested since §2.5 read as
      never-read, on the reviewer's own database first. Backfilled from the commentary chunks
      themselves, which are proof the file was read.
      And the one place magnitude decided anything decided it **alphabetically**: the market
      offered for repair was `barren[0]`, so Andorra's single campaign was offered ahead of
      LATAM's twenty.
      **From adversarial review, four more.** The guard `len(barren) < len(by_market)` was
      meant to avoid noise and instead **silenced the review's own example**: in a library
      that holds only LATAM campaigns, "no LATAM store launch has ever carried a budget" said
      nothing at all, because every market was barren. Gone.
      The first fix round filtered forecasts out of the `save_evaluation` path and left
      `prepare_evaluation` reading the raw evidence, where `find_similar` includes predicted
      rows — so the earlier of the two calls still reported a judgment resting entirely on a
      forecast as complete.
      Markets were grouped in a way the rest of the product does not recognise: a campaign
      reached only through the `markets` LIST was invisible, a whitespace market was reported
      as a market ("No campaign in    , NA has measured results"), and `LATAM` was called
      barren while `latam` had results — though `filter_campaign_ids` matches them
      case-insensitively and returns both. And 35 markets produced a 3,185-character
      sentence, now capped at five names plus a count.
      Superseded records were counted as library evidence, though `find_similar` excludes
      them — so a measured v1 replaced by an unmeasured v2 made the library look measured
      while no judgment could reach the measurement.
      Two gaps could send the user to the same campaign, so accepting both appended two
      identical metric rows. A gap that is closed by an earlier one's action now says
      `closed_by` and keeps the fact without repeating the offer.
      **Two mutations survived** the first round's tests — reversing the rank order, and
      deleting the empty-library early return — because the ranking test built a library with
      one gap in it and an empty library is naturally silent on every later branch. Both now
      have tests that bite, and `test_every_gap_says_what_would_close_it` was reaching one
      code of three.
- [x] **5.4 (D) `diff_campaigns(a, b)`** — adopted / ignored / newly-introduced /
      carried-stale, computed against the earlier version's evaluation findings. Flagged by
      the reviewer as the highest-value feature not already on the list.
      **Done, and "computed against the earlier version's evaluation findings" is what makes
      it a computation rather than a second judgment.** Once both versions have been
      evaluated the comparison is between two sets of findings, and every bucket is a fact
      about the record rather than an opinion about the decks. Everything carries
      `basis: "computed"`, so Claude can say it as fact.
      **A fifth bucket, because the review's four collapse a distinction that matters.**
      `no_longer_raised` — neither resolved nor raised again — was folded into "adopted",
      and it is not the same thing: the finding was either fixed without being recorded or
      not looked at the second time, and the record cannot tell which. Counting it as adopted
      would credit a brief for work nobody verified, on the one surface whose whole purpose
      is to say whether corrections were taken. It is reported with that caveat attached.
      **Matching is deliberately generous.** Two reviews of one defect rarely word it
      identically, so exact-text matching would report every ignored correction as a new
      one — the most flattering error available. Category agreement is a hint rather than a
      gate, since the same defect gets filed differently by two reviewers.
      **Nothing is guessed.** Where a version has never been evaluated, `comparable` is
      false and `why_not_comparable` says which side is missing: "you ignored my correction"
      is an accusation, and inferring it from deck text would be a judgment dressed as
      arithmetic. The tool offers `prepare_evaluation` on the unjudged version instead.
      **The argument order is corrected rather than trusted.** Which version is earlier
      decides what "adopted" means, so having them the wrong way round inverts the entire
      answer; supersession states it outright and creation order is the fallback, with
      `arguments_reordered` returned so the user is told.
      **It gets better as judgments accumulate.** `resolved` entries carrying a `finding_id`
      — the mechanism §2.4 built for precisely this — are what turn "probably fixed" into
      `adopted`. The feature 2.4 was accused of over-engineering is the one this depends on.
      Verified end to end over the protocol on a three-version Colombia chain: 1 adopted, 1
      raised again, 1 newly introduced, 1 no-longer-raised, and a `carried_stale` citation to
      a Peru record that had been superseded in between.
      **Both reviewers found the same thing, and it reframed the item: `adopted` had no
      feeder.** It is populated only from `resolved[].finding_id`, and nothing in the product
      ever put those ids in front of the model at the moment it was judging v2 —
      `prepare_evaluation` took a title and some text, and `find_similar` excludes superseded
      records by design, so v1's judgment was invisible in v2's evidence. In production every
      fixed finding would have landed in `no_longer_raised` and `adopted` would have been
      permanently empty. The suite missed it because the fixture reads the id out of the
      store and writes it in by hand. `prepare_evaluation` now takes an optional
      `campaign_id` and returns an `earlier_version` block with the prior findings and their
      ids, and the note tells Claude to resolve each by id or repeat it with `repeats`.
      **`ignored` is gone, and the measurements are why.** Character similarity scored
      "Two influencers are adidas-affiliated" against "...Nike-affiliated" at **0.89**,
      "Budget is over the approved ceiling" against "...under..." at **0.93**, and
      "Slide 4 has no posting dates" against "Slide 9..." at **0.96** — all different
      problems, all reported as a correction somebody ignored, on a surface the marketer
      carries to their agency. It scored the SAME problem reworded at **0.36**. So it
      measures phrasing, not meaning, in both directions. The bucket is now `raised_again`,
      every entry carries `match: "id" | "text"`, a text match is stamped `basis: "judged"`
      with its similarity and a caveat, and the docstring forbids the word "ignored" on
      anything but an id match. "Deliberately generous" was exactly wrong: it was generous to
      near-duplicates and strict on rewording.
      **Matching is no longer order-dependent.** Taking each earlier finding in turn and
      giving it the best remaining later one meant that with two competing for one, the one
      processed first won — at 0.79 against the other's 0.98. Which correction was accused
      therefore depended on the order somebody typed the findings, or on the severity sort
      that reorders them before ids are assigned. All pairs are now scored and assigned
      best-first, with ties broken on the text itself, so the answer is a property of the two
      sets rather than of their order.
      **The order is corrected only on recorded evidence.** Creation order is *upload* order,
      and an organisation seeding its archive uploads v1 after v2 as a matter of course —
      silently reversing their question with a JSON field they never see as the only tell.
      `order_basis` is now `"supersession"` or `"as_given"`, and the latter carries a warning.
      **A pre-2.4 judgment is not comparable.** It reads back with no verdict and no
      findings, so it was treated as comparable and every later finding came back as newly
      introduced — a v2 blamed for everything its own v1 review had found. The marker is the
      verdict rather than a non-empty findings list, because an approve with nothing wrong is
      a complete judgment that happens to have no findings, and recognising the version where
      everything got fixed is the whole point.
      **Also:** `carried_stale` looked only at the earlier judgment, though the later one is
      the one somebody is about to act on; `counts` was missing on the not-comparable branch,
      so the shape depended on which branch ran; `record_changes` was added because a v2 that
      drops a market from a LATAM brief is the largest change a version can carry and diffed
      as identical, as did a performance tag that lost its `verified` source; a `finding_id`
      pointing at nothing saved without complaint and then vanished from the diff, so "we
      fixed that" was recorded and silently lost; and the offer to evaluate an unjudged
      version sent `detail or title`, so for a deck-only upload accepting "evaluate this
      version" judged eleven characters.
      **Three mutations survived** the first round: a later finding matching twice, the
      category tie-break, and the creation-order fallback. All three now have tests.
      **And I broke the tracker's own rule** — 5.4 created eight deferrals and recorded none
      in the same commit. D57-D64, plus amendments to D41 and D46.
- [x] **5.5 (E) Coverage view** — market × collection × stage, counts and evidence quality.
      **Done.** `coverage()` answers the three questions the review asks, and the third is
      the one nothing else in the product answered: *"which have only one example carrying
      all the weight"*. A cell carried by a single campaign produces judgments that are
      really that one campaign's opinion, and the similarity score looks identical whether it
      came from one precedent or nine. `evidence` is `no_outcomes`, `single_example` or
      `measured` — worst first, with both counts present so nothing hides behind the marker.
      `no_outcomes` outranks `single_example` deliberately: two campaigns with nothing
      measured compare a proposal against what was *planned*, which is weaker than one
      measured example.
      **Cells overlap by construction** and the report says so. A campaign that ran in three
      markets belongs to three cells, because "what do I have in Colombia" is asked per
      market — so the counts do not sum to `campaigns_total`, and leaving somebody to add
      them up would be the quiet kind of wrong.
      **`thin` is the answer; the matrix is the evidence for it.** A matrix is something to
      browse, so the same cells come back ordered worst-first and the docstring tells Claude
      to read that instead.
      **Records with no market are included**, since that is most of a young library and
      dropping them would describe a library nobody has. Superseded and `reference` records
      are excluded, for the reason §5.3 needed: a superseded record is invisible to every
      search, so counting it as coverage describes evidence no judgment can reach.
      **Closes D55.** `gaps()` had grown its own market-or-region grouping, so the two
      surfaces could describe the same library differently — and one of them ignored the
      `markets` list entirely. Both now group through `_markets_of`, verified over the
      protocol: coverage reported MENA thin and `gaps()` named MENA.
      **Bounded**, because market × collection × stage is multiplicative and this goes inside
      a tool result somebody reads: 25 cells, with `cells_total` saying how many exist.
      Kept distinct from `gaps()`: this is "what do I have", that is "what do I fix first"
      with prefilled actions. Same grouping, different question.
      **From review — and the "cannot disagree" claim above was false when first written.**
      `_markets_of` folded case within a record while the bucket key used the raw spelling,
      so "LATAM" and "latam" became two cells — one reading `no_outcomes`, the other
      `single_example` — for a library `filter_campaign_ids` treats as one market of two with
      one measured. `gaps()` folded correctly and reported no barren market at all, so the
      two surfaces contradicted each other on the same library and C16 was not a closure.
      `collection` had the identical split, and the store matches it on `LOWER(collection)`.
      Both axes now fold, keyed lower-cased and displayed as first seen.
      **A campaign that has not run gets `not_yet_run`, not `no_outcomes`** — §5.3 wrote that
      lesson out and this surface repeated it: a proposed campaign cannot have results, so
      calling its cell thin is a complaint nobody can answer.
      **A `stub` is not coverage.** It is a placeholder by the product's own definition, so
      counting it as evidence describes content that is not there — and it arrived with a
      null stage that nothing explained.
      **Truncation can lose detail but never the shape.** With 25 weak cells, `cells` and
      `thin` came back byte-identical: no thick cell shown, ten hidden with no count of what,
      and `single_example` — this item's own headline — absent because `no_outcomes` filled
      the list. `cells` is now ordered for reading, `thin` carries the ranking, `hidden`
      counts what was cut by kind, and `evidence_summary` counts every cell whether shown or
      not, so "measured: 0" is stated rather than inferred from an absence.
      **Six mutations survived the first round** — any-metric-row, two sort mutations, the
      `campaign_ids` cap, the case dedupe, and `region` being ignored — because the only
      ordering test used two cells whose alphabetical order happened to match their ranking.
      Also: the empty-library offer was written out twice, in `gaps()` and here, and neither
      copy named the title among its `needs`, so accepting it literally raised
      `TypeError: missing 'title'`. One `actions.to_first_upload()` now, with a test that
      checks `needs` structurally against the tool's required parameters rather than looking
      for the word "title" in prose.
      And `health_check` already had a field called `coverage`, meaning indexing
      completeness, whose docstring now read as a pointer to this tool; disambiguated.
- [x] **5.6 (F) Guided first run** on an empty or thin library.
      **Done, in two halves.** The *shortest path* is the review's own prescription and is
      ordered rather than offered as a menu: one brief you liked, one you did not, the
      rulebook. The contrast is the point — two briefs somebody liked teach the library
      nothing about the axis it is being asked to judge on. Steps drop off as they are taken,
      because a path that still lists what you have done is a checklist nobody believes.
      **There is deliberately no "you need N records" number**, and the review asked for one
      ("how many records it takes before judgments become useful"). Giving a number would be
      the kind of confident, unfounded figure this whole review was written against:
      usefulness depends on WHAT is in the library, not how much. Two contrasting briefs make
      the like/dislike comparison work at two records; a hundred concluded campaigns with
      nothing measured still cannot say whether any of it worked. So readiness is stated as
      what the product **can** and **cannot** do given what is actually present, and every
      limit names the record that would lift it — a capability statement nobody can act on is
      a disclaimer.
      Each `can`/`cannot` carries a stable `code` beside its prose, for §3.1's reason: the
      sentence is for the user and will be rewritten, and nothing reading it should break
      when it is. Two of this item's own tests were coupled to wording before that existed.
      **It reaches somebody — and the first version of this sentence was false.** The claim
      was made on the strength of `core.list_campaigns_with_readiness`, which was written,
      tested, and never called by any tool: the guidance reached nobody, which is the exact
      failure the tracker row behind it described. The function also returned
      differently-shaped rows from the tool's own projection, so wiring it in naively would
      have changed the tool's output. Replaced by `readiness_for_listing`, which returns only
      the guidance, and the `list_campaigns` TOOL attaches it while the library is not yet
      working and stops once it is.
      **Closes three tracker rows.** D43: every step here is an offer whose arguments only
      the user has, so `needs` is what keeps "accepting is one step" honest. D66: `coverage`'s
      `thin` list collapses to one summary line plus the path when nothing in the library is
      measured at all — listing 25 weak cells in a wholly-weak library is the matrix again,
      and the answer at that size is not "fix LATAM". D68: `unmeasured_campaigns` groups the
      fix per campaign, so one campaign in three thin cells is offered once.
      Found while wiring D66: "nothing is measured" has to be a fact about the LIBRARY, not
      about the cell markers — five cells of one measured campaign each read `single_example`
      rather than `measured`, and that is a library with real evidence in it.
      **From review, and one finding is this item committing the mistake it was written to
      prevent.** `can: check_against_rules` claimed the product could "check a brief against
      your own guidelines" on the strength of `has_rulebook`, which is only "some `reference`
      record exists". Nothing pins, fetches or checks against it — `prepare_evaluation` is
      similarity retrieval, so the rulebook reaches the evidence only if it happens to rank,
      and the accompanying `cannot` text says in as many words that a guardrail which might
      not be retrieved is not a guardrail. So the item promising to say plainly what the
      product cannot do was promising the one thing it cannot. Now `rulebook_on_file`
      ("cite your guidelines when they happen to be retrieved") is the `can`, and
      `check_against_rules` stays in `cannot` until §7.5/§12.1 pin the rulebook.
      The D66 collapse handed over to `shortest_path`, which never mentions measurement — so
      once the three steps were done, the summary named measurement as the problem and
      offered nothing, while `gaps()` on the same library offered `add_metrics`. It offers
      the measurement now, falling back to the path only when nothing has concluded.
      Tags were matched case-sensitively while the store folds them, so a marketer who typed
      "Liked" was told forever to add a campaign they liked — the permanent complaint this
      codebase names three files over. `readiness` also ignored `status`, asserting "compare
      against what you have RUN before" for a library of proposals and asking for a concluded
      campaign's results when nothing had concluded; `gaps()` draws that line explicitly and
      this, written beside it, did not.
      `say_what_worked` appeared in `can` and `cannot` at once for any partly measured
      library, so a reader keying on the code could not tell which side won. And
      `mixed_reaction` counted as a dislike — a mutation removing it passed all 103 tests, so
      nothing pinned it, and the item's own rationale asks for "one you did NOT like", which
      a mixed reaction is not.

## Phase 6 — Making the reasoning defensible (ideas G–L)

*Depends on 2.4.*

- [x] **6.1 (G) `precedent.quote` required** on every finding, and verified.
      **The re-sequencing note this item carried was wrong, and it is worth saying why.** It
      read: "drawn from the retrieved chunk" cannot be enforced while `prepare_evaluation` is
      stateless, so this needs 7.2 or a receipt id. True of the literal wording, and
      irrelevant to the defect. The thing worth preventing is a quote attributed to a record
      that the record does not contain — and that is checkable against the CITED RECORD's own
      stored text, with no receipt, no retained window, and no dependency on who wrote the
      retrieval query. Three items of machinery were sequenced ahead of a check that needed
      none of them, because the note restated the requirement instead of the risk.
      **What it does.** A `precedent` must carry a quote (it was bounded but never required,
      so a finding could name a campaign and quote nothing). The cited id must resolve to a
      real record — this codebase's own fixtures had been citing `camp_jdsea`, which resolved
      to nothing, quoting a sentence nobody had written, and every test passed. The quote must
      be IN that record, and at the layer it claims: the text of a reviewer's objection is in
      the record, so verifying text alone would have returned a green tick on "the deck said
      the timeline is unrealistic". Refused rather than flagged, for consistency idea 3's own
      reason — a validation error is a retry, and an unverified quote stored beside verified
      ones is drift nothing downstream can undo.
      **Faithful, not byte-identical.** Case, wrapping and curly quotes are folded; an elision
      (… or ...) matches in order and within ONE stored unit, so a quote cannot be stitched
      out of two chunks that were never adjacent. Refusing a re-cased quote would not improve
      provenance — it would teach the model that quoting is a game it loses, and the way a
      model wins that game is by quoting less.
      **`rule_id` is not a way round it.** It must resolve to a `reference` record, or the
      check would be decorative: any finding could be saved unverified by writing `rule_id`
      where `campaign_id` would have been checked.
      **Deliberately not required on every finding.** `missing_information` and
      `internal_contradiction` are claims about the subject, which is not in the library.
      Demanding a precedent quote there would send the model looking for a campaign to quote
      at, and a citation produced to satisfy a validator is the invented evidence this item
      exists to stop. The two kinds that assert something about another record must cite one.
      The rule is in `save_evaluation`'s description AND in `prepare_evaluation`'s note,
      because a rule a model only meets as a rejection afterwards costs a retry every time.
      Closes D6 and D7.
      **From review, and the first version was wrong in both directions at once — it let
      unsupported citations through and refused faithful ones.**
      *Ways past it.* Filling BOTH `campaign_id` and `rule_id` made the campaign decorative:
      the quote was checked against the rule, and the campaign could be invented and still be
      stored beside a passing check. A `guardrail_breach` could anchor itself to an ordinary
      campaign simply by using the other slot — the thing the rule_id type check was written
      to prevent, reachable by a different door. And there was no floor: `"a"` and `"."`
      verified against every record in the library, so the tick certified nothing. The slots
      now have to match the kind, only one may be filled, and a quote is at least 12
      characters with at least 8 either side of an elision.
      *The elision was the worst of it.* The per-unit rule was described as stopping a quote
      being "stitched out of two chunks that were never adjacent" — but `chunking.pack` merges
      slides up to 1800 characters, so a short deck is ONE unit, and an unbounded elision
      inside it stitched two unrelated slides into a sentence the deck never contained, with
      the meaning inverted. Elisions are now bounded to 200 characters: leave out more than
      that and it is two quotes, which is two findings.
      *Ways it blocked honest work.* Verification read CHUNKS, and chunk boundaries are an
      1800-character packing artefact the model cannot see and the document does not have — a
      verbatim, contiguous quote across one was refused with "do not paraphrase into a quote",
      and no amount of re-copying could satisfy it. Chunks are also stale: `update_campaign`
      rewrites `detail` without re-chunking, so a citation of what the marketer had already
      corrected verified against the old wording. Body verification now reads the row's own
      `title`/`detail`/`deck_text` — current by definition, contiguous by construction, and a
      superset of every body chunk. Commentary keeps chunks, because there is no column, with
      consecutive pieces of ONE person's comment joined and two people's never.
      Review also ran real PDF and PPTX files through the extractor: a soft hyphen inside
      "require­ments", "sched-\nule" hyphenated across a justified line, a decomposed accent,
      U+2010, a zero-width space — each refused a quote a person would call verbatim, while
      accusing the model of paraphrase. Folding is now NFKC plus the invisibles plus
      line-break hyphenation. And `find_similar`'s own `... [truncated]` marker was held
      against a model quoting the tail of what we showed it.
      *The word.* `verified: true` became `basis: "computed"` + `checked: ["record", "layer"]`.
      "Verified" already means something exact in this product — a performance claim backed by
      real metric data — and on a finding the nearer reading is that the FINDING is verified,
      which is not what was checked. `basis` is the vocabulary this codebase already uses for
      "the server worked this out"; `checked` names what was actually established, so 7.6 can
      add `"window"` without a schema change. A model that sends either is refused, like every
      other server fact.
      *Two loops with no exit.* The refusal said "drop the citation and say it as an
      observation" — which `_CITING_KINDS` then refuses a second time, leaving deleting `kind`
      as the only way out, which no message mentioned. And the rule-type refusal said "doing
      it differently from a past campaign is a precedent_departure", an in-error offer to
      downgrade the one class the product calls not debatable — §2.4's own lesson is that an
      easy exit offered inside a validation message gets taken.
      *Reach.* `closest_precedent` carries the same id-and-quote shape and was outside the
      check entirely — an invented citation at the top of the judgment, where a summary is
      most likely to read it aloud. And the worked example in `save_evaluation`'s description
      — the shared prompt §7.4 is built on — hung a `camp_jdsea` precedent off a
      `missing_information` finding: an id that resolves to nothing, on the one kind the same
      docstring says should not go looking for a campaign to quote at.
      *One thing typed too strictly.* `quote: str` in the `Precedent` TypedDict made pydantic
      refuse the call at its own boundary, so the caller got "Field required" instead of this
      project's sentence about what an uncited assertion is — the exact failure `_enum`'s
      comment describes. Required in core, `NotRequired` in the type.
      **From the re-review of those fixes — three of them had moved the problem rather than
      closed it, which is the reason a fix round gets reviewed too.**
      *The elision bound was per hop.* Ten legal 130-character hops chained across a thirty-
      slide deck, because each one passed on its own; and a character budget alone never
      catches a SHORT deck, where four slides assemble into a sentence nobody wrote while
      leaving out barely fifty characters. Bounded now by the TOTAL elided (200) and by the
      NUMBER of gaps (2). A quotation has one gap, occasionally two; four is a composition.
      *The hyphen fold was asymmetric.* Deleting every `-\s+` deleted a spaced dash and left
      an unspaced one, so "3 - 28 March" and the quote "3-28 March" folded to different
      strings and a faithful quote came back as "do not paraphrase". The hyphenation join is
      now anchored to an actual line break, and dash spacing is collapsed on both sides.
      *Metric detail was not quotable* — the product's most common real citation. "CTR was
      3.2 percent, well above the benchmark" is what `find_similar` shows the model as
      evidence, and a metrics-only record was told it had "nothing but a title" while the
      numbers' own words were exactly what was being quoted. And the *title* was quotable
      whenever the record had any detail, so the same quote was evidence or not depending on
      whether a brief happened to exist. Title out, results in.
      *`closest_precedent` was echoed back raw* once it was validated at all: a model could
      assert its own `verified` beside a 5,000-character junk field, one field above where
      those keys are refused. It is rebuilt from known keys now — and its `layer` went into
      `on_file[layer]` unvalidated, coming back as a `KeyError`, which `_catch_value_errors`
      does not catch, so the caller got "Error executing tool" with the message discarded.
      *Two unattributed comments were joined* into one quotable block, because `None == None`
      — two strangers' remarks stitched into a single quotation, which is the misattribution
      the layer rule exists to stop, one level down. The comparison was also on JSON text, so
      key order decided whether one person's comment was one comment.
      **A third round, on those fixes.** *Greedy matching refused a faithful quote:* the span
      was measured from the LEFTMOST occurrence of the first phrase, so a recap slide
      restating the opening — ordinary in a deck — put the two halves of a verbatim quote at
      opposite ends of the file. Every starting occurrence is tried now, up to twenty.
      *The hyphen fold was still asymmetric,* the other way round: no regex can tell
      "sched-\nule" (one word the layout split) from "well-\nknown" (a real hyphen falling at
      the break), and typesetting breaks at an existing hyphen FIRST, so the second is the
      commoner. The stored text is read both ways.
      *Predicted metrics were quotable as body,* so a finding could record a campaign as
      stating an outcome it had only forecast — the layer misattribution wearing different
      clothes. Actual rows only.
      And the total-elided bound *had no test that could fail*: every case with more than one
      gap was refused by the gap COUNT before the arithmetic ran, so reverting it to per-hop
      left the suite green. The commit claiming "thirteen mutations, all killed" did not
      cover the one that mattered most — which is why a fix round gets its own mutation pass
      and not just its own review.
      *And `basis` was the wrong word after all.* The first fix replaced `verified: true` with
      `basis: "computed"`, which collides: a finding's `basis` says who produced the FINDING,
      so the same key one level down claims the server wrote the citation. It did not — the
      model wrote the quote and the id, and only the check is the server's. `checked` alone,
      which says that and no more.
- [x] **6.2 (H) Guardrail breach vs departure from precedent** — two classes, different
      vocabulary, only one is debatable.
      **`kind` is required.** Optional, it was the cheapest way past every rule attached to
      it: a finding with no kind needs no citation, cannot contradict its slot, and is exempt
      from the departure rule below. The 6.1 review found that escape and it went into the
      tracker as the cheaper half of D78. Required in core rather than in the type, for the
      reason 6.1 learned the hard way — typed required, pydantic refuses at its own boundary
      and the caller gets "Field required" instead of the sentence naming the four kinds.
      **A departure has to say which way it departs**, and this is the part the review
      actually asked for that neither 2.4 nor 6.1 delivered. "Does not match how Peru seeded"
      is not a finding until somebody says whether different is WORSE here — and forcing that
      choice is what makes the class genuinely different from a breach. A breach is a fact
      about a rule. A departure is a judgment about whether a difference matters, and that
      judgment is precisely what a partner is entitled to argue with. `departure` is
      `regression` | `unexplained` | `possible_improvement`, required on a
      `precedent_departure` and refused on anything else — a rule is not a matter of degree,
      and a gap in the brief is not a difference from anything.
      **`possible_improvement` must be a note**, the exact mirror of 2.4's "a guardrail
      breach cannot be a note". Asking the marketer to change something back while recording
      that it may be better contradicts the finding's own reading — and that is what this
      product did to the UAE brief, whose claw machine was a departure that turned out better
      than the precedent. The review named that case; until now there was nowhere to put it.
      **The two classes are voiced differently, not just labelled differently.** A class
      distinction nothing says out loud is a column in a database. `save_evaluation` returns
      `by_class` (`not_debatable` / `arguable` / `about_the_brief`) beside the severity
      counts, because "2 blocking" reads the same for a rule somebody broke and a preference
      they may have been right to depart from — which is the review's complaint in one line,
      that the two "come out of the same machinery". And the response `note` differs by what
      is actually in the judgment: state a breach and name the rule; ask about a departure and
      do not tell them to change it back until they have answered; say a possible improvement
      as good news. The tool description carries the same split, with the claw machine in it,
      because an abstract rule without a worked case is a rule nobody applies.
      **Not done here, named so it is not assumed** (the convention 2.4 set): the departure
      asks a question and the product cannot take the answer. When the marketer says "yes,
      four colourways was the client's call", nothing records it — there is no write path onto
      a saved judgment (D53, D59), and `resolved` is accepted only on a later version's
      judgment, where it records "we changed it", not "we kept it, and here is why". So the
      stored finding still reads `unexplained` after it has been explained, and the same
      question comes back next time. The response says that out loud rather than pretending
      otherwise, and D84 owes the write path to 10.2.
      The other thing named rather than fixed: `unexplained` is the cheap answer. It costs no
      claim, unlike `regression`, and no visibility, unlike `possible_improvement`. Expect
      mostly `unexplained` after a hundred evaluations; D78 predicts it and 7.7's agreement
      runs are the instrument, which D81 says does not exist yet.
      **From review — and two of the findings were this item repeating a mistake it had just
      been written to prevent.** The response's `note` branched on the CLASS, so a blocking
      `regression` was told to "ask whether it is deliberate, do not tell them to change it
      back" in the same response as its own `fix` line: the three readings exist so the
      voicing can differ, and two of the three were sharing one voice. And the `_how_to_say_it`
      guidance told the model to deliver the good news about a possible improvement whose text
      the response did not contain — the improvement is a `note` by rule, and `findings` keeps
      only blocking and should_fix, so the claw machine was the finding least likely to reach
      the marketer. It has its own `improvements` list now, and `departure` is in the findings
      projection so "worse" and "may be deliberate" do not need a second round trip to tell
      apart.
      *The rule was one-sided.* `possible_improvement` could not be blocking, but
      `unexplained` could — a question that by itself stops the brief is the same
      contradiction. And a `possible_improvement` could still carry a `fix`, which is the
      reversal the severity rule refuses, arriving in the other field.
      *The worked example the shared prompt teaches was refused by the server* — no
      `departure` — and its `fix` said "seed one colourway", telling the marketer to change
      back the thing the same docstring says to ask about four paragraphs above. 6.1's review
      found this same class of defect in this same docstring, so the test now SAVES the
      examples rather than checking that the right words appear in them.
      *`diff_campaigns` dropped `departure`,* so a first review calling a difference a
      `regression` and a second calling it a `possible_improvement` was reported as "raised
      again" — a correction not taken. That is the claw-machine story filed as a repeat
      defect by the tool built to tell them apart. It now carries `departure_now` and a
      `reread` code.
      *And `get_evaluation` leaked the raw vocabulary:* `save_evaluation` translated the
      classes into something a marketer can hear and the read path did not, which is the
      "worked only inside the session that produced it" failure 2.4 fixed for the findings
      themselves. It returns `by_class`, `improvements` and the same voicing note, and takes a
      `departure=` filter.
      Closes D1. The `rule_id` half of D1 was already done by 6.1.
- [x] **6.3 (I) Close the prediction loop** — when a superseding record arrives, surface the
      prior evaluation's predictions and ask which held.
      **The load-bearing sentence in the review is the second one:** "`reconcile_evaluation`
      has never once run". The tool exists, works, and is never called, because it needs
      somebody to decide to go back — and nobody does. So the fix is not a tool, it is a
      MOMENT: the person uploading v2 is looking at the thing that proves or refutes what was
      said about v1, and asking then costs one prompt. `upload_campaign(supersedes=…)` now
      returns `earlier_judgment` — the prior verdict, its predictions, its open findings, and
      an ask — plus a `reconcile_evaluation` offer prefilled with the id.
      **Not gated on `predictions`.** Most judgments carry none, and the case the review
      described was a RECOMMENDATION that came true ("the same structure returns rearranged
      unless the premise is settled"), not a CTR range. Gating on the forecast field would
      have dropped the example the item exists for. It IS gated on there being something to
      check: a record nobody judged has no claim to test, and a judgment already reconciled
      has been tested — asking again is the always-present prompt nobody reads.
      **Closes D41, the offer both reviewers condemned in 5.2.** What was wrong there was
      never the offer, it was the evidence: prefilled from `closest_precedent`, a similarity
      match asserted by the model, when "this replaces that" is a claim about somebody's
      intent and a similarity score is a fact about text. It comes back on the evidence 5.4
      identified as strongest — two judgments of the same brief that a diff compared to
      completion, which means two reviews already treated them as versions of one thing. The
      label names the record that would be hidden, because that is the consequence being
      agreed to and it is invisible in the arguments.
      **And it needed a write path, which is why D41 could not be closed before.**
      `update_campaign` could not set `supersedes` at all: a supersession could only be
      declared at upload, and one declared by mistake hid a record from every future search
      and was undoable only by deleting the campaign — that is why 5.2's reviewers called the
      offer dangerous rather than merely wrong. It can now be set, and cleared with `""`.
      Three ways of being nonsense are refused: a record superseding itself removes itself
      from the library, a record superseding nothing links nothing, and a cycle hides both
      ends and would make the chain walk below run forever.
      **Closes D64.** `diff(v1, v3)` compared only the two endpoints, so a finding v2 had
      explicitly resolved came back against v3 as `no_longer_raised` — "neither resolved nor
      repeated", when the library holds the record of it being resolved. It walks the
      supersession chain now and reports `through`, because a diff across a chain is a
      different claim from a diff between adjacent versions and a reader cannot tell
      otherwise. Closes D57.
      **From review, and the first finding was this item recreating the §5.2 failure it had
      quoted.** The prefilled offer FAILED WHEN ACCEPTED: `reconcile_evaluation` looks up
      measured results on the record being replaced, which is a brief that never ran, so
      accepting returned "no actual metrics on file" and told the model to record results for
      a proposal. §5.2's rule is not "the tool must exist", it is that accepting must work —
      and the test asserted the prefilled arguments instead of calling it. It offers
      `save_reconciliation` directly now, and the test calls the offer.
      *The D41 offer is accepted by calling `update_campaign`* — the one path where the loop
      did not fire. The flow this item constructs led straight to the gap in it. That user is
      the better case, not the worse one: they have both judgments on screen because they
      just compared them.
      *The link offer's evidence was still wrong.* It fired whenever a diff of two judged
      records completed, which is not a fact about the records — it is a fact about which two
      ids the CALLER passed, and the caller is the model 5.2 condemned. Two unrelated
      campaigns got an offer to hide one of them, in the same response whose warning says the
      library cannot tell which came first. The gate is a finding id now: a later finding
      whose `repeats` names an earlier one, or a recorded resolution. Both are statements
      somebody made, which is this module's own rule that a resemblance never outranks a
      statement. The guard also reads both endpoints and their `is_superseded`, because
      reading only the later one let a record already replaced by a third be re-parented.
      *`ingest_campaign(supersedes=)` had no validation at all* — the primary declaration
      path could create every state `update_campaign` refuses, and a typo left `{"", "  ",
      "camp_nope"}` in the set of superseded ids. A blank is now no supersession rather than
      an error, because whitespace is an empty form field and not a typo for an id.
      *The order check and the chain walk disagreed about "linked".* The check read the two
      `supersedes` columns while the walk read the chain, so `diff(v1, v3)` warned "no
      supersedes link" while using the link, and `diff(v3, v1)` was silently answered
      backwards with the chain never walked. Ancestry, not adjacency.
      *Chain-adopted is not adjacent-adopted.* Between adjacent versions "adopted" means the
      later reviewer confirmed it; across a chain it means confirmed once and not re-examined
      since. Entries carry `resolved_in` and a caveat saying so — the module caveats weaker
      claims than that one.
      *And the payload was a quiz.* Unfiltered, a twelve-finding judgment turned filing a
      document into an exam, and the findings are settled one by one when this version is
      evaluated anyway — where 5.4 records each by id rather than as free text. Capped at
      three with the total reported; the verdict and the predictions are not capped, because
      the forecast-shaped claim is the thing only this moment catches.
- [x] **6.4 (J) Disconfirming search required** before a verdict is saved; record what came
      back, including "nothing".
      **The server runs it, not the model**, and that is the decision worth writing down
      because the review does not say who. A model that has already reached a verdict, asked
      to go and find evidence against itself, is marking its own homework — and the failure
      the review names, "drift toward confirming whatever the first strong match suggests",
      is exactly what would shape the query. This codebase settled the same question twice
      already, in §2.4's `basis` rule and §6.1's `precedent.checked`: a check is worth
      something only if a difference in it is a bug, which holds only when the server did it.
      A model-supplied `evidence.disconfirming` is refused.
      **What contradicts depends on the verdict**, which is why it happens at save time and
      not at retrieval. A `revise` or `reject` is contradicted by precedent that looked like
      this and WORKED — Mexico, the library's own warning, a brief rated weak that performed.
      An `approve` is contradicted by precedent that looked like this and did not. `verified`
      tags only: a performance tag somebody typed is an impression, and an impression cannot
      be the counterweight to a verdict.
      **Three outcomes, not two.** "It could not be checked", "it was checked and nothing came
      back" and "it was checked and something did" are three conclusions, and collapsing any
      two turns an absence of evidence into evidence of absence. A library with nothing
      measured cannot argue back, and reporting that as a clean check would make an empty
      shelf a supporting vote. Each has its own code and its own sentence in the note.
      **`uncited` is the point, not `found`.** A contradicting campaign the judgment DID cite
      has already been weighed; throwing it back would train the reader to skip the field.
      The Mexico case is the uncited one — the reasoner never retrieved it, so it never had
      to explain it away.
      **And the save-time check cannot change a verdict, so there is a second half.**
      `prepare_evaluation` now splits the retrieved evidence into `outcomes.worked` /
      `did_not_work` / `unknown` by measured result, instead of handing over one ranked list
      in which the strongest match sets the tone. The save-time search records
      overconfidence; this is the part that can prevent it.
      **From review, and the first finding falsified the item's own headline sentence.** The
      search queried the JUDGMENT'S OWN PROSE — summary plus finding lines — and then called
      what came back "precedent resembling this one": a claim about the brief, made from a
      search over the complaint about it. A verdict reading "budget is thin" about a
      Mexico-shaped brief does not retrieve Mexico. It queries the subject record where the
      subject is a stored record, says `query: "judgment_text"` where it is not, and D86 owes
      7.2 the case where a brief is judged before it is filed.
      *And the preventive half was a partition, not a search.* `outcomes` sorted the top-5
      the reasoner had already seen — so on any real library the five nearest are five
      records nobody tagged, both poles come back empty, and the reasoner has seen nothing
      contradictory. That is the state the review describes, dressed as the fix for it. Each
      pole is now its own filtered search, because a filter runs before ranking and a
      partition cannot reach past it.
      *An embedder outage read as a verdict that survived scrutiny.* `embedding.Unavailable`
      is a `ValueError`, so it was swallowed into "nothing came back" and the note then said
      the verdict "was argued against and it held" — the exact collapse this item exists to
      prevent, arriving on the failure that actually happens, which is Ollama not running.
      *There was no similarity floor,* so a single-market print campaign came back at
      similarity 0.0 as precedent "resembling" a creator-led launch. Once a library holds one
      measured campaign per pole that fires on every save: the always-on field nobody reads.
      *And `top_k` truncated before the cited filter,* so three cited campaigns filled the
      window and a fourth, genuinely uncited, was never looked at — reported as "nothing
      contradicted it". The cited ones are excluded from the candidates now.
      *Five codes, not three.* Two of the original three covered states meaning opposite
      things. `nothing_contradicted_it` in particular claimed "the verdict was argued against
      and it held" on the strength of one filtered query — §6.1 was made to narrow `checked`
      to exactly what it established, and this was doing the reverse.
      *It was arguing with rules.* A `revise` resting only on `guardrail_breach` searched for
      "precedent that worked anyway" and told the model it was "most likely to change what
      the marketer does" — undoing §6.2 from the next field over. A rule is not a matter of
      precedent; whether some campaign broke it and did well is a question for whoever owns
      the rulebook. Exempt only when the verdict rests ENTIRELY on rules.
      *`reference` records were precedent,* which nothing else in this codebase allows.
      *`nothing_to_check_against` named no way out,* while every other `cannot` here names
      the record that would lift it — it now carries the concluded, measured campaigns that
      nobody has tagged, which is one `update_campaign` away.
      *`uncited` was bare ids,* so a model mirroring the shape says "camp_8701e96a" rather
      than "Mexico launch, performed_well, CTR 3.4% against a 1.8% benchmark".
      *And it was not on the read path.* "Recorded so it can be audited" was true of the
      table and false of the surface: `get_evaluation` did not return it, so from the tool
      side the check lived for exactly one response.
- [x] **6.5 (K) `approve_if`** — the testable exit condition that converts revise → approve.
      **Required on a revise, forbidden on an approve — and forbidden on a reject**, which the
      review does not say and which is the one part of "testable" a validator can genuinely
      enforce. If naming a set of changes would make a brief approvable, that is what "revise"
      means; letting a reject carry an exit condition lets the harsher word be used with the
      softer meaning. An `approve_if` on an approve is a reservation the verdict does not
      admit to — §2.4 removed the same hedge from a different field.
      **A `fix` on every finding above a note** (D2's other half). A finding at blocking or
      should_fix is a claim that something must change, and without a `fix` the reader has the
      complaint and not the remedy.
      **The checklist is what makes it testable.** The review asks for an exit condition that
      is "testable" and that "doubles as the note the partner receives", and those are one
      requirement: a sentence cannot be ticked off and a list can. `exit_checklist` is
      composed by the SERVER from the `fix` lines the findings already carry — not written
      separately, because a hand-written checklist beside a findings array is two sources of
      truth for one thing, which is the failure this project has hit four times. Each item
      points at the finding it came from, so a later version can close them by id. Absent
      rather than empty on an approve: an empty list reads as "nothing left to do, we
      checked", when the truth is there was never a list.
      **And it reaches the version that is supposed to meet it.** §6.3 built the moment a
      superseding record arrives; the exit condition and the checklist ride along with it, and
      both are on the read path. An exit condition nobody sees when the next version lands is
      a sentence in a database.
      *One tension this surfaced, worth recording:* §6.2 says an `unexplained` departure must
      be asked about rather than corrected, and §6.5 requires a `fix` on it. The resolution is
      that for an unexplained departure the action IS an answer — "say whether four colourways
      is deliberate" — which is a real, checkable thing to do and not an instruction to undo
      what the finding has just said may be deliberate. The worked example says so.
      **From review, and the finding is this item contradicting itself one line apart.** The
      checklist was composed for EVERY verdict. So an approve carrying a `should_fix` — which
      §2.4 permits — came back with a list of things to change: the server manufacturing the
      exact "reservation the verdict does not admit to" that the `approve_if` rule two
      functions up refuses, out of the `fix` it had just demanded. A reject got one too, while
      being refused an exit condition on the grounds that having one makes it a revise. The
      rule was enforced on the sentence and contradicted by the list. Only a revise has an
      exit, so only a revise gets a checklist; the fixes stay on the findings either way.
      *`_exit_checklist` was the one read-path consumer that indexed rather than `.get`,* so a
      stored judgment from before the `fix` rule raised `KeyError` out of `get_evaluation` —
      and out of §6.3's moment, which would have 500'd instead of asking.
      *The handoff broke two of its own neighbours' rules:* it emitted `approve_if: None` and
      `exit_checklist: []` where the response beside it says absent-not-empty, and it re-listed
      every finding uncapped under a second key, next to an `open_findings` capped at three
      because "a twelve-finding judgment turned an upload into a quiz".
      *And the checklist stripped `departure`,* which is the field that tells the person
      ticking it off whether an item is a change or an answer — the distinction §6.2 exists
      for and that this item's own worked example turns on.
      Closes D2.
- [x] **6.6 (L) Evidence-strength line** on every judgment — how many precedents, concluded,
      verified; top similarity; whether one match dominates.
      **The load-bearing word in the review is "identical":** "a verdict resting on five
      concluded campaigns with verified outcomes and one resting on a single proposed brief
      currently look identical". The complaint is not that the numbers are missing, it is
      that two judgments of very different worth are presented the same way — a shape problem,
      not a data problem, which decides the rest.
      **Counted from `cited_ids`, not from the evidence package.** What a judgment rests on is
      what it CITED, not what it was handed. A verdict shown eight precedents and citing one
      rests on one, and reporting eight would be the overstatement this field exists to
      prevent, committed by the field written to prevent it. An id that resolves to no record
      is not counted and is listed as unresolved — §6.1 verifies the precedent on a finding,
      and `cited_ids` is a separate list nothing had ever checked.
      **A number nobody reads is not a signal.** Five counts and a similarity are a row of
      digits; "one match is carrying this" is a sentence. `strength` is a stable code
      (`no_precedent` / `single_example` / `unmeasured` / `measured`) with a line saying what
      it means, and the note says to give it BEFORE the verdict — afterwards the verdict has
      landed and the caveat reads as hedging.
      **Dominance is a gap, not a rank.** Something is always top of the list; what the review
      asks for is the case where one match is far nearer than anything else cited, so the
      judgment is effectively resting on one record while appearing to rest on six.
      **Closes D3.** The whole `evidence` block is the server's now, which is what D3 asked
      for and what a 2.4 test had to be rewritten to reflect: the column landed early so no
      migration would be needed, and it was accepting whatever the caller passed. A model
      reporting the strength of its own evidence is not reporting a measure.
      **Closes D65** with the same question one level up: `coverage` reads a cell of six
      measured campaigns as `measured`, but if every verdict in it cites the same one record
      the library's real depth there is one. `cited_share_top` and `never_cited` come from
      `cited_ids`, which the schema has held all along with nothing reading it. `None` rather
      than 1.0 when nothing has been judged — zero judgments is not "one record carries
      everything", and that is the same distinction §6.4 needed separate codes for.
      **From review, and the first finding is §6.4's own lesson broken 240 lines below it.**
      `embedding.Unavailable` is a `ValueError`, and catching it with one reported
      `top_similarity: 0.0` — indistinguishable from "dissimilar" — in the same response where
      §6.4 correctly said `could_not_check`. It reports `similarity_checked: false` now.
      *The ladder's words outran its counts.* It branched on verified TAGS while the sentence
      claimed measured RESULTS, so five concluded campaigns with real metric rows were told
      "none of them has measured results" while `coverage` called the same five `measured` —
      D88's drift, contradicting itself in adjacent fields. The two are now counted
      separately (`with_results` and `verified`) because both are real and they are not the
      same number, and the ladder is built on the first. And five unmeasured briefs came back
      `single_example`, identical to one brief: the review's own complaint, one rung down,
      inside the item written to fix it. `unmeasured` and `partly_measured` separate them.
      *`top_similarity` was 0.0 for any cited record outside the retrieval window,* so "far
      nearer this brief than anything else cited" could be true only because the others were
      never retrieved. Unscored records are excluded rather than floored at zero.
      *And the two dominance thresholds masked each other.* Deleting either left the whole
      suite green, because in any realistic fixture a large gap comes with a top score above
      the floor. Pulling the decision into `_dominant` and testing it with explicit numbers
      killed both — and the first attempt at the floor case still had a gap under the
      threshold, so the gap refused it and the floor was free to be deleted again.
      Three things found on the way out and tracked, not fixed here: D89 (the migration adds
      three campaign columns while the readers assume the whole schema), D90 (the same string
      embedded twice per save), D91 (`coverage` scans the evaluations table once per cell).
      One fixed on the way out: `campaigns_with_actual_metrics` now returns empty rather than
      raising on a database with no `metrics` table — §5.3 had been crashing on an upgraded
      database too, unnoticed, because the legacy tests happened to cite nothing.

## Phase 7 — Consistency across users (ideas 1–8)

*Reviewer's own order: 1 and 2 first, then 3 and 4, then 6 and 7 before any prompt tuning.*

- [x] **7.1 Compute what can be computed.** Must run on the BODY layer only (§2.5):
      a reviewer's comment saying "make sure we never mention adidas" would otherwise
      register as the brief mentioning adidas. Date coverage, ER presence per profile, budget
      detected, channel checklist, internal date consistency, guardrail keyword hits —
      returned as **computed facts** with evidence. Two thirds of real findings were
      mechanical; this removes most cross-user variance.
      **The sentence that decides the design is "anything a model decides is something a model
      can decide differently tomorrow."** Phase 7's subject is consistency across users, and
      the mechanism here is not a better prompt — it is removing the question. Two thirds of
      the output stops being generated, so two thirds of the variance has nowhere to come
      from. In `facts.py`, its own module: six independent text checks that know nothing about
      campaigns or retrieval, and burying them in `core.py` would make them harder to argue
      with, which is the one thing a computed fact has to be.
      **Five of the six shipped; the sixth is deferred and named.** Guardrail keyword hits
      need a rulebook to have keywords, and today the "rulebook" is any `reference` record
      retrieved by similarity — D92 carries it to §12.1. Shipping a keyword check against
      whatever reference record happens to exist would be the guardrail-that-might-not-be-
      retrieved the review already condemned.
      **Absent is a finding; unknown is not.** "This brief carries no dates" is mechanical.
      "No date in it carries a year, so no claim about which day it falls on can be checked"
      is a different statement, and `date_consistency` returns `nothing_to_check` rather than
      `consistent` for it — a green tick on an unread page is the collapse §6.4 needed three
      codes to avoid. Same for `not_applicable`: a brief with no creators is not missing their
      engagement rates.
      **Closes D11** — the body layer, its first consumer. A reviewer's note saying "never
      mention a competitor budget of 90,000" is not the brief's budget.
      **Closes D49.** "No LATAM store launch has ever carried a budget" is one of the review's
      own three gap examples and the only one about an absent FIELD inside records; it needed
      this item because nothing could detect a budget. Two records minimum, because one record
      missing a budget is a fact about that record and calling it a coverage gap would turn
      every new market into a complaint the day it is added. Appended after the ranked gaps
      and carrying no offer, because closing one means editing several records and that write
      path is D84's — ranking an unofferable gap above an offerable one puts the thing nobody
      can act on first.
      **Closes D58.** `diff_campaigns` compared structured fields and citations; a budget
      dropped between versions, or a date contradiction introduced, was invisible. It reports
      the STATUS changing, not the value: "the budget went from 40,000 to 50,000" is content a
      reader can see, "the budget disappeared" is a change in what the brief can be judged on.
      **From review, and every one of the five was a confident false fact on ordinary brief
      text** — which is the specific thing this item must not do, because a wrong computed
      fact is worse than a model's guess: it carries the server's authority and the note told
      the model not to contradict it.
      *`maria@brand.com` was two creator profiles.* Almost every brief carries a contact
      address, so the commonest real brief got an authoritative "none of your creators has an
      engagement rate" about people who do not exist — and it poisoned the D49 market gap on
      top, because that filter excludes `not_applicable` and not a false `absent`.
      *"500k impressions" marked PR as covered.* `press` was an unanchored substring, so the
      checklist reported a channel as handled because another word contained it.
      *A correct brief was told its dates contradict the calendar.* The weekday lookback was a
      fixed forty characters, so "Saturday 7 March 2026 to 12 March 2026" — where 7 March
      really is a Saturday — reported that the brief calls 12 March one. It produced the very
      finding this item was sold on, out of nothing.
      *"Budget line 3 may be deferred" was a date,* because `may` is a modal verb before it is
      a month; and "Go live March 2026" was NO date at all, which is the parser-failure-as-
      absence the module docstring forbids, committed by the module.
      *"No paid social" reported paid social as present,* inverting the finding, and `RRP
      $49.99 per unit` was a budget.
      Fixed by precision rather than breadth: handles that are not email addresses, anchored
      channel patterns with a negation check and `ruled_out` beside `present`, a claim window
      bounded by the previous date and by sentence structure, capitalisation as the signal for
      `may`, month-year as a date, uppercase-only currency codes, and a budget figure that has
      to sit near a word saying it is one. The checklist now says NAMED rather than covered,
      because the check reads words and whether a named channel is planned is a judgment.
      **And the note got an escape hatch.** "Do not contradict them" removed the last check on
      a wrong fact — the model reading the evidence and seeing the "creator" is an email
      address. It now says to dispute a fact the evidence plainly does not support, naming the
      code and quoting the evidence, rather than deferring to it or silently re-deriving it.
- [x] **7.2 Server owns the retrieval query.** Derive from the subject record/file, not
      model-authored `proposal_text`; derive filters from the subject's attributes; pin
      `top_k`; stable deterministic tie-breaking; record embedding-model version per vector.
      **"Same deck in, same chunks out"** is the review's acceptance test and the whole phase
      in one sentence. Pass `campaign_id` and the query is the record's own text and the
      filters are its own attributes, so a terse description and a 2,000-word one retrieve
      the same evidence. Caller filters are REFUSED alongside a `campaign_id` — a filter the
      model picked is a filter nobody can see it picked — and allowed without one, where
      there are no attributes to derive from, with `filters_from` recording the difference.
      `top_k` is pinned: a judgment resting on three precedents and one resting on twenty are
      different judgments, and neither number is a fact about the brief.
      **Tie-breaking had to be fixed in three places,** which is the tell that it was never
      really decided anywhere: the ANN index returns equal distances in storage order, the
      brute-force fallback sorted on similarity alone, and the campaign-level rollup re-sorted
      and undid both. Ties break on the campaign id at every level — stable across machines,
      across an insert, and across the two backends.
      **The embedding model is recorded per vector,** and two models in one index raises a
      `degraded` notice: similarities from different models are not comparable, so the ranking
      the whole evidence package rests on would be arithmetic across incompatible scales.
      **The receipt is the part that pays for the rest.** §6.1 rejected a receipt as the
      MECHANISM for verifying a quote (X7) — it answers a weaker question than "does the
      record contain these words", and it would have put a stateful handshake on a stateless
      tool. That objection stands for 6.1. Here the server is already doing the retrieval, so
      writing it down costs one row, and it answers what 6.1 could not: **was this evidence in
      front of the reasoner at all.**
      **Closes D77.** A judgment carries `from_the_window` and `outside_the_window`. A cited
      record that was never retrieved is not shared evidence — the next person judging the
      same brief will not see it — which is precisely what Phase 7 is about. `not_recorded`
      rather than a clean result when no receipt is passed, because an absent check reads as a
      passed one.
      **Closes D8.** `closest_precedent` is the top of the window when there is a receipt, and
      marked `computed`; a model-supplied one is then refused. Without a receipt the server
      cannot compute what it did not retrieve, so the model may still name one and it is not
      marked computed.
      **Closes D86.** The disconfirming search reads the subject from the receipt, instead of
      querying the judgment's own prose and calling the results "precedent resembling this
      one" — a claim about the brief made from a search over the complaint about it.
      **Closes D80.** `update_campaign` rewrote `title` and `detail` and left the chunks and
      vectors alone, so search kept matching the old wording. §6.1 had stopped the stale text
      being QUOTABLE by reading the row columns; the index itself was still stale, and a
      marketer who corrects a brief and then cannot find it is looking at the same bug from
      the other end. Content edits re-index; a status or tag change does not, because
      re-embedding for one would make every bulk edit a re-index of the library.
      **From review, and the first finding hollows out the item's headline feature.** *The
      receipt was not bound to its subject.* A receipt taken for one brief laundered any
      citation into `from_the_window` for a DIFFERENT brief, and stamped a server-`computed`
      closest precedent onto a subject it was never about. The receipt's whole claim is "this
      evidence was in front of the reasoner for this brief", and half of it was unchecked.
      *A snapshot ages.* `closest_precedent` took the top of the window unconditionally, so it
      could assert as a computed fact a record since deleted or superseded — while
      `unresolved_citations` in the same block named the same id as missing. It now takes the
      first record a search would still return, and `superseded_since_retrieval` /
      `deleted_since_retrieval` say what changed underneath.
      *Two of the three tie-breaks were untested,* which is worth recording because the commit
      message boasted about all three: the one tie test had three records, so the rollup alone
      fixed their order and the two sorts underneath could be deleted with the suite green.
      *Provenance outlived its vectors, and asset vectors carried the text model's name.*
      After a real migration — delete, re-embed — the old model would be reported forever,
      leaving a permanent "this library is mixed" notice on a library that is not; and a text
      embedder change would have flagged CLIP vectors that had nothing to do with it.
      *The remedy named a tool that does not exist* ("run reembed"), which is L5's lesson: a
      remedy nobody can follow is worse than none, because it moves the blame to them.
      **And one design decision was simply too broad.** Refusing every caller filter removed a
      capability the docstring documents — "weigh only precedent whose performance claim is
      verified" is a deliberate narrowing somebody asks for, not the model quietly choosing a
      scope. The line is what the filter DOES: `market`, `region`, `collection` and `markets`
      say which brief this is and come from the record; `tags`, `status` and `record_type`
      narrow within it and stay the caller's, recorded on the receipt so the choice is
      reproducible rather than invisible — which was the actual complaint.
- [x] **7.3 Enforce the output shape server-side** — reject writes missing a precedent quote
      or exceeding caps, rather than accepting and hoping. *Caps done by 2.4; the
      missing-quote half done by 6.1, which also verifies the quote rather than only
      requiring one. Completed by 6.2, which made `kind` required — until then a finding
      with no `kind` at all needed no citation.*
- [x] **7.4 Tool descriptions as the shared prompt** — the evaluation procedure into
      `prepare_evaluation`'s description; set the MCP server-level `instructions` field.
      **The load-bearing decision is not what the procedure says — it is that there is ONE of
      it.** The phase's subject is consistency across users, and a procedure only produces
      consistency if every user gets the same one. A copy in the server instructions and a
      copy in the tool description is two procedures that agree today, and this project has
      watched a hand-maintained copy drift four times (the Windows installer version,
      `TOOL_NAMES`, this plan's own counts, and D75's three definitions of "measured").
      `EVALUATION_PROCEDURE` is one constant referenced twice, and a test fails if the opening
      line ever appears in the source more than once.
      **The description is built rather than taken from the docstring.** A docstring is for
      somebody reading the file; what reaches a model is the registered description, and those
      are not the same string unless something makes them so.
      **Two of the review's six items could not ship as written, which is part of the item
      rather than an omission.** The *scorecard's six criteria* belong to the customer's
      rulebook: hard-coding Fabletics' rubric into a product that ships generic is exactly
      what the product-owner decision forbids, so the procedure says so and D101 carries it to
      §12.1. And *"the instruction to run a disconfirming search"* changed hands — §6.4
      decided the SERVER runs it, because a model asked to find evidence against a verdict it
      has already reached is marking its own homework. The instruction that exists is to read
      the result, and to read the `code` rather than the absence of rows.
      **Closes D12** — the layer rule now lives in the shared procedure rather than only on
      the browsing tool.
      **There is a length ceiling, and the number is a budget decision rather than a
      measurement.** The review's own position is that restating the rules alongside the data
      is "the highest-leverage prompt real estate the product has", so it is deliberately
      generous; what the ceiling catches is the accretion that turns a procedure into a
      manual, by forcing every future addition to displace something.
- [x] **7.5 Ship the procedure with the evidence** — `prepare_evaluation`'s `note` carries
      the pinned rulebook, computed facts, required output schema and weighting rules.
      **The argument is about WHEN, not what.** §7.4 put the procedure where it loads once; a
      rule read at connect time has been competing with an hour of conversation by the moment
      it matters, which is when the model is holding the evidence and about to judge.
      **The note carries the same constant, not a summary of it.** Summarising would produce a
      second, shorter procedure — and §7.4's whole point was that two versions of one
      procedure is the drift this project has watched four times. The procedure moved into
      `core` so the note can reference the same object; it is now written once and referenced
      from three places, and the once-only test scans every module rather than the one that
      happens to hold it.
      **What goes first is what cannot be static:** the rulebook version in force and THIS
      brief's computed facts, each with what it means. A model reads a long field from the
      top, and the procedure is the half it has already been given.
      **The rulebook line is the honest gap.** §12.1 has not built one, so "the rulebook at
      its pinned version" is a version of nothing — and it says so rather than omitting the
      line, because a judgment made with no rulebook is a different judgment from one made
      under a rulebook that happened to say nothing. §7.6 stamps it onto the record so the two
      can be told apart later.
- [x] **7.6 Stamp every verdict** with `rulebook_version`, `server_version`,
      `embedding_model`, `model_id`, and retrieved ids + similarities.
      **The clause that decides the design is the last one:** "when they agree, you have
      evidence the agreement is real rather than luck". A stamp is not an audit trail for its
      own sake — it is what lets AGREEMENT be read as evidence, which is the premise §7.7's
      golden set rests on. Two runs that agree while differing in embedding model and rulebook
      version agree about nothing in particular.
      **Four of the five are facts the server holds. `model_id` is not** — the judging model
      is on the other side of the protocol — so it is recorded as unknown unless the caller
      says, and marked `stated` when they do. A stamp with a wrong field in it is worse than
      one with a missing field, because the diff that is meant to explain a disagreement would
      then explain it wrongly.
      **`compare_provenance` produces the diff,** because "the diff of those five fields
      usually explains it in seconds" only works if something actually produces it. It names
      what the two had in COMMON as well as what differed — agreement is the claim the review
      cares about, and it is a claim about the stamp rather than about the verdicts.
      **Closes D4.** The block is the server's. A provenance record the model writes is a
      record of what the model says produced it — and this is the second field to make that
      journey after `evidence`, which means rewriting a second 2.4 test that had asserted the
      column would store whatever the caller passed.
      **Closes D18.** The warnings raised while the evidence was gathered are stamped: a
      judgment made over a half-indexed library is a different judgment from one made over a
      whole one, and the warning that said so lived for exactly one response.
      **Closes D81.** Refusals are counted by REASON, not totalled — "thirty writes were
      refused" says nothing, and a rise in citation refusals means something different from a
      rise in severity downgrades. The quietest outcome is the one this makes visible: a real
      finding dropped because its citation would not verify used to leave no trace at all.
      **Closes D85.** A reconciliation says what it was checked against, `results` or
      `superseding_version`. §6.3 made the version-based one the common case, so "v2 shows the
      structure came back" was landing in the same column as a CTR figure — and anything
      computing calibration later would have read both as measured outcomes.
      **Closes D29.** The stamp names a digest when the embedder will say what it loaded.
      `ollama/nomic-embed-text` names a TAG and a tag moves, so two judgments could be stamped
      with the same embedder while using different weights — and the stamp's whole claim is
      that it explains a disagreement. Absent rather than invented when the embedder will not
      answer.
      **Closes D89, and the fix is the class rather than the instances.** The columns to add
      are derived from `_SCHEMA` itself, because a hand-kept list of migrations is a second
      copy of the schema and a second copy drifts — three campaign columns were listed while
      `get_campaign` read the whole thing. Writing the test surfaced a second, sharper
      problem: `_SCHEMA` creates indexes on columns the migration has not yet added, so on an
      upgraded database it failed on an index before the migration that would have made it
      valid could run. Tables, then columns, then indexes — `store.upgrade()` is now the only
      correct order and the only entry point.
- [x] **7.7 Golden set + agreement measurement** — verdict agreement and finding recall, three
      runs per brief. Harness built now; briefs to be supplied by the product owner.
      **"Everything above is a hypothesis until it is measured"** is the sentence that places
      this item. Phases 6 and 7 are twenty-odd changes made on the argument that they reduce
      variance, and not one of them has been measured — every claim in them is an argument
      until this exists. `agreement.py`, runnable as `python -m agreement`, because "run them
      on every release" is the whole point and a harness nobody can invoke is a module.
      **The harness ships; the briefs do not.** Ten to twenty briefs with CLIENT-AGREED
      verdicts is the definition, and only the product owner can supply them. Inventing them
      would measure agreement with my own guesses, which is worse than measuring nothing
      because it produces a number somebody will quote.
      **The judging is supplied too** — the model is on the other side of the protocol — so
      `run` takes a callable. That is also what makes the measurement testable: a stub judge
      with known behaviour is the only way to check the arithmetic is right.
      **Three figures, not two.** The review asks for verdict agreement and finding recall.
      Three runs per brief measures a third thing neither can see — whether the product
      answers the same question the same way twice — and without it a brief that matches the
      client two runs in three reads as a partial success rather than as instability. When
      anything is unstable the report says to read that FIRST, because a product that answers
      one question two ways has not agreed with anybody.
      **An empty golden set reports nothing, not a perfect score.** Zero agreeing out of zero
      is the confident-shape-with-nothing-in-it failure this project keeps finding, and it
      would have been reported on every release until somebody noticed. The CLI exits 0 —
      an absent measurement is not a failing build, and failing CI over it would make the
      first person to add a brief the person who broke the pipeline.
      **Recall, not precision,** exactly as the review asks: a judgment that raises the two
      expected problems and one more has not failed. And a judge that raises is counted as an
      error rather than a disagreement — a model that failed to answer did not disagree with
      the client, and counting a crash as a wrong verdict would make the product look
      inconsistent when it was merely unavailable.
      **Stamped with §7.6's provenance.** A figure with no record of the conditions it was
      measured under cannot be compared with last release's, and a change in it cannot be
      attributed to anything.
      **It also forced a fourth kind of tracker row.** Seven rows pointed at 7.7 saying "the
      golden set will measure this" — D9, D56, D69, D78, D82, D83, D96 — and the harness
      existing does not answer any of them. Filing them as still deferred would make a
      finished item look incomplete; filing them as accepted limits would say nothing will
      ever fix them, which is false. They are **awaiting input**, and the distinction the new
      section records is who the next move belongs to.
- [x] **7.8 Mark findings `computed` vs `judged`.**
      **§2.4's premise said nothing until now.** "A computed finding is identical for every
      user, so a difference there is a bug" is a claim about findings the SERVER produced —
      and the server produced none, so `computed` was an unwritable value on an empty set.
      **Closes D5.** §7.1 established the facts; this turns the ones that are problems into
      findings. A fact is not a finding: "no date appears in this brief" becomes one when
      somebody says it is a problem, and until now only the model could take that step, which
      put a `judged` label on the most mechanical half of the output — so §7.7 would have been
      measuring a model's consistency at reading a regex's result.
      **Three facts become findings, not all of them.** `channels: partial` is the normal state
      of every brief, and a finding that fires on everything is one nobody reads. The three
      are the review's own examples, and each is a thing simply absent or simply wrong.
      **They never change the verdict.** The server establishes facts and the reasoner judges;
      a server finding that flipped an approve would be the product overruling the reasoner on
      the strength of a regex, and §6.4 already settled that the server ARGUES with a verdict
      rather than replacing it.
      **`counts_by_basis` says which half of the output is the model's,** because §7.7
      measures finding recall against what the product raised — and without it the figure
      would improve every time a regex fired, with nobody able to tell why.
      **Closes D61 with a third value.** §5.4 matches findings across versions by character
      similarity and had to call the result `judged` with a caveat, because it is neither: a
      similarity score is not a human judgment, and it is not identical for every user in the
      sense `computed` means, since it is a heuristic whose threshold somebody chose.
      `heuristic` says exactly that, and each of the three values carries a sentence — which
      is what stops `heuristic` being read as a softer `computed`.

## Phase 8 — Metrics that evolve

- [x] **8.1 Metric registry** — canonical key, display name, unit, direction, aliases, first
      seen, times seen, campaign types, status. Values in typed storage, not an opaque JSON
      string (today's `structured` cannot answer "show me every ROAS on file").
      **Reading the review's list of collisions closely gives three problems, not one.**
      *The keys collide* — `reach` / `crm_reach` / `stated_combined_influencer_reach` are one
      measure with three names. The registry canonicalises, and `raw_key` is kept beside the
      canonical one because canonicalising is a CLAIM about what somebody meant, and keeping
      what they wrote is what makes the claim checkable.
      *Some of what is in those keys is not a name at all.* `_mxn` is a UNIT and
      `_upper_funnel` is a SCOPE. Folding them into the key is what made two budgets unaddable
      and impressions unqueryable, so they are pulled into their own columns rather than being
      spelled away — the registry knows the kind of unit and the key knows the instance.
      *And the last pair is not a naming problem.* `planned_roas_july_stated` versus
      `_recomputed` is "the partner's figure is wrong and here is the right one" with nowhere
      to live, so it became a key name — the review calls it the clearest signal the schema is
      failing. A registry that only canonicalised names would rename both halves to `roas` and
      lose exactly what distinguishes them. A value row carries its own `source`, a correction
      is a second VALUE rather than a second measure, and `best_value` says which one to weigh
      — keeping both and saying nothing records a disagreement without resolving it.
      **Typed storage is the other half.** "Show me every ROAS on file" is the question a
      metric store exists to answer, and a JSON string cannot be asked it.
      **The registry is seeded into the database, not read from the constant,** because §8.2
      adds provisional entries and §8.3 graduates them — a registry half in code and half in a
      table is two registries, which is the drift this project has now watched five times.
      The seed is the PRODUCT's canonical names; §12.1 lets a rulebook extend it.
      **Closes D33.** A target is a `metric_type` beside `actual`, not a separate thing: it is
      the same measure and the only difference is whether it had happened yet. `against_target`
      answers "did we hit our number" with THREE answers rather than two — `met` is None when
      there is no result yet, because a campaign that has not concluded has not missed
      anything, and None again when the measure has no recorded `direction`, because under
      budget is not automatically good and over is not automatically bad. That is what
      `direction` was in the registry for. And a target is never weighed as a result: a
      library that does that reports what somebody hoped for as what happened.
- [x] **8.2 Unknown metric → provisional + ask** (never reject, never silently accept).
      **Both halves fail in opposite directions.** Rejecting an unfamiliar key loses the
      measurement: the marketer has the number, the product refuses it, and it goes into
      freeform prose where nothing can ever compare it. Silently accepting is how two campaigns
      produced 25 keys — every one was accepted, and nobody was ever asked whether `crm_reach`
      was the `reach` already on file.
      **"Surface it once" is what makes the loop bearable.** The same unknown key arriving in
      ten campaigns asks once, not ten times, and once answered it never asks again —
      including when the answer was "ignore", because re-asking a question somebody has
      declined is how a product teaches people to dismiss it. `unanswered()` holds the ones
      nobody answered, or "surface once" quietly becomes "surface once and lose".
      **The three answers do different things.** `same_thing` folds the key in
      RETROSPECTIVELY — the values already recorded under the provisional name belong to the
      measure it turned out to be, and an answer that fixes the vocabulary while losing the
      data has fixed nothing. `different_measure` keeps it and stops asking, but leaves it
      `provisional`: becoming part of what a brief is EXPECTED to carry is §8.3's gate, and one
      partner's house metric must not become a standing requirement because somebody said it
      was real. `ignore` is a decision about the vocabulary and not the data — the values stay,
      because a dismissive click must not destroy a measurement somebody recorded.
      **The suggestion is never the default.** §5.1's cutoff, and its reasoning: a wrong alias
      silently merges two measures that are not the same thing, and the answer is one click
      away. A key that resembles nothing suggests nothing.
      *Found while building it:* `footfall_uplift_pct` and `footfall_uplift_percent` would have
      been two measures, because only currencies were being read as units in a key. That is
      the exact collision this item exists to stop, reintroduced by the thing meant to prevent
      it.
- [x] **8.3 Graduation gate** — seen in N campaigns, ≥2 partners/markets, confirmed once by a
      person, before a measure becomes expected.
      **"Count alone is not enough" is the sentence that shapes the gate.** Three conditions,
      all required, and the second is the one doing the work: fifteen sightings in one market
      is one partner's habit, and promoting it makes every other market fail a checklist it
      never agreed to. The third is a person, deliberately not automatable — §8.2 called its
      question "the only human step in the loop" and this is where it is spent.
      **CAMPAIGNS, not writes.** `times_seen` counts every value recorded, so a brief logging a
      stated figure and a recomputed correction would count twice towards a gate meant to say
      "three different briefs carried this". One brief's opinion recorded three times is one
      brief.
      **A shipped name is not a standing requirement.** The seed registry's twelve measures
      register as `known`, not `expected`: `status` is where a measure sits in the VOCABULARY
      and `expected_in` is which checklists it is on, and collapsing the two would have put
      twelve measures on every checklist the day the product is installed. A check that fires
      on everything is one nobody reads.
      **A measure is expected only where it graduated.** `expected_in` holds the markets it was
      actually seen in, so a measure that earned its place on LATAM and SEA is not a gap in a
      market nobody has used it in — otherwise every new market fails a checklist on its first
      brief.
      **The checklist is keyed on MARKET, and the review asked for campaign TYPE** — "Expected
      for a store launch". No field carries that: `record_type` is a storage class, `collection`
      is an instance family, `objective` does not exist, and `tags` has no controlled
      vocabulary (D73 owes that to 12.1). Market is a **stand-in**, recorded as D102 rather
      than presented as the reading; "partners", which the review names first, has no field at
      all (D103).
      *Found by review:* every one of these reproduced. **One market typed three ways** —
      "SEA", "sea", "Sea" — satisfied the two-market gate, which is exactly the capture the
      gate exists to stop, and C16 had already closed this bug elsewhere. A campaign that
      **literally ran in MX and CO counted as zero markets**, because the gate read
      `market or region` and ignored `markets` — the fifth implementation of one question
      (D55/D88), now moved into `store.markets_of` where `core` and `metrics` share it.
      **v1/v2/v3 of one brief counted as three campaigns.** A brief with **no market was held
      to the union of every checklist**, and a `reference` record was told it was missing
      footfall uplift. A **target counted as a result**, so a concluded campaign holding only
      targets was told it "carries all of them". The gate **refused the only names a user
      has** (`footfall_uplift_pct`, `crm_reach`), and on an already-graduated measure it said
      "collect more data" while a second `graduate` **overwrote `confirmed_by`** — the one
      audit field the gate exists to create. An **`ignore`d measure was told to collect more
      data** too, re-asking a question the user had declined, which §8.2 named one item ago.
      And `resolve_measure` could **demote or DELETE a graduated measure** — `same_thing` runs
      `merge_metric`, which drops the registry row and `confirmed_by` with it: §8.5's "never
      delete", broken from the one direction nothing was watching.
- [x] **8.4 Fixed template, growing data** — never a self-editing prompt.
      The expected set is read from the registry at call time and rendered into
      `prepare_evaluation`'s `expected_measures`, so a measure graduating changes what every
      subsequent brief is checked against **without anybody touching a string** — the review's
      own test: *"No prompt was edited to make that appear."* The per-brief note names what
      THIS brief is missing and is silent when nothing has graduated, and it says missing
      rather than wrong: the server establishes the gap, the model decides whether it matters
      (§6.5).
      **The human step has to be reachable, or none of this ever fires.** The gate's third
      condition is a person, and nothing surfaced eligibility — which left `measure_status`, a
      tool a user would have to know exists, think to call, and name the measure by its
      canonical stem to use. No measure would ever have graduated, so §8.4's headline sentence
      could never have appeared under any data. §8.2 had already solved this: the question
      rides on the write that creates it. `add_metrics` now returns `newly_eligible` once, with
      a prefilled offer whose `confirmed_by` is deliberately **not** filled in — prefilled with
      a name nobody gave, it would manufacture the very confirmation the gate requires (§6.5);
      left out silently it would fail when accepted (§5.2). `needs` is how this codebase
      already says "one more thing, and it is yours to supply".
      **Documenting the field is not editing the prompt.** `expected_measures` was in the
      package and in no description, which most plausibly means the model turns every miss into
      an uncited `missing_information` finding — the pressure D78 already tracks.
- [x] **8.5 Retirement** — track last seen, demote after M campaigns, never delete.
      Campaigns, not months: a library nobody has opened for six months has not retired
      anything. A retired measure keeps its row, its `expected_in` and every value under it —
      "we used to track this" is an answer somebody will need, and a deleted row can only say
      "we never did". One that is recorded again is expected again without a second
      confirmation, because it graduated once and a person confirmed it.
      *Found while building it:* `_add_missing_columns` iterated a hand-kept TUPLE of tables,
      and §8.1 added `metric_registry` and `metric_values` without adding them to it — so every
      column §8.3 declares would have reached a fresh install and no upgraded one. That is
      exactly the D89 defect, which was closed "as a class rather than the three instances
      somebody happened to notice", reopened by the one list still holding a second copy of the
      schema. The tuple is now derived from `_SCHEMA`, and the D89 test — which named four
      tables by hand — now iterates every declared one.
      *Also found:* a real protocol round-trip printed **"This brief carries none of the 1."**
      The review's sentence is "none of the four" and the code had hard-coded that word.
      *And by review:* the retirement unit was wrong in three separate ways, each of which
      retires a measure the library is actively using. It counted **writes**, so one campaign
      logging ten figures aged out everything; it counted **every campaign**, so a bulk import
      of 200 historical briefs retired the measure carried by 53 of the 63 on file; and it
      counted **every market**, so ten APAC campaigns retired a measure expected in LATAM, SEA
      and EMEA. It is now campaigns, in the markets where the measure is expected, that
      reported measurements and **skipped this one** — which is what "stopped appearing" means.
      It also ran **inside a read**: `prepare_evaluation` silently mutated the registry, the
      list of what it demoted was discarded, and *who read* decided *what was retired*. It runs
      on the write that makes it true and is reported on `add_metrics`, per §2.1.
      *Upgrade:* §8.1 wrote the twelve seed measures as `status='expected'` when the word meant
      "a measure this product recognises". §8.3 gave it its real meaning, so on a database from
      the earlier release all twelve would have become standing requirements confirmed by
      nobody. A column migration cannot see a change of MEANING, so
      `_demote_unconfirmed_seed_metrics` migrates it — touching only rows with no `expected_in`
      and no `confirmed_by`, so anything a person really graduated survives.
- [x] **8.6 Same loop for standing corrections** — one learning mechanism for both.
      **"One learning mechanism" is the requirement, and a second one shaped like the first
      does not satisfy it.** The loop lives in `learning.py` — thresholds, gate, the order
      status is asked in, the market fold, the version fold, the three reasons a record has no
      checklist — and `metrics` and `corrections` both read it. Every §8.3 defect is therefore
      fixed for corrections by construction, because there is only one place it could be wrong.
      A test asserts the constants are the same OBJECTS and that neither caller re-declares
      them.
      **What differs is the data.** A measure is a key with numeric values; a correction is a
      sentence. `provenance` is required — *"so a judgment can cite where the rule came from"* —
      and every mention keeps its own, because "praised in Peru; instructed independently in
      Australia" is one rule with two origins and the second is what makes it more than one
      client's house style.
      **A judgment can now actually cite one.** `guardrail_breach` demanded a `rule_id`
      pointing at a `reference` record, so a graduated correction was shown to the model as a
      rule and could not be cited by it — §5.2's "an offer whose write path does not exist",
      one item after it was named. `precedent: {correction_id, quote}` is the third slot, the
      quote is checked against the rule's own words, and the provenance and confirmer travel
      onto the stored finding.
      **The ten corrections are not seeded.** They are one customer's rules; §12.3 ships them
      as an example overlay. The point of this item is the loop, so they can grow.
      *Found by review, all reproduced:*
      **The loop could not fire at all, in two independent places.** `note_correction` was
      referenced nowhere in the product but its own definition — a tool the model would have to
      know existed and spontaneously call. That is §8.3's unreachable human step one stage
      earlier and strictly worse: nothing would ever be RECORDED for the gate to act on. The
      input was already in hand, because §2.5 ingests tracked client comments with author and
      anchor, which is the whole of what a correction needs; `after_upload` now offers it.
      And **the gate was inverted**: matching was exact-string, so three markets independently
      saying "seed a single colourway", "seeding boxes should carry one colourway" and "only
      one colourway per box" made THREE corrections, each stuck at one campaign in one market.
      It was satisfiable only by the identical string arriving three times — one person
      copy-pasting, the single-source case the two-market rule exists to reject. Easy to
      satisfy illegitimately, impossible to satisfy legitimately. §5.1 and §8.2 both settled
      this as *suggest, never auto-merge*, and keeping only the refusal half misread both.
      **A standing correction was given authority nobody granted it.** `not_debatable` voicing
      says "a rule **they wrote** was broken" — false of an inference from repetition plus one
      confirmation. Worse, a verdict resting only on rules skips the disconfirming search
      entirely, which is right for a rule the customer wrote and backwards for one the library
      inferred: a past campaign that did the opposite and did well is the only evidence that
      could ever catch a wrong inference. And there was no inverse — `learning.gate` has
      carried a `set_aside` branch since §8.3 and nothing could reach it, so a correction
      promoted in error blocked approvals with no way back.
      **Retirement is the one place this loop deliberately differs from §8.5's**, because the
      same signal means opposite things. For a measure, absence is disuse. For a rule, absence
      of repetition is usually COMPLIANCE — a client stops restating a rule exactly when the
      agency starts following it, so retiring on silence demotes the rules that are working and
      keeps the ones being ignored. No threshold fixes that. The question is asked once, with
      keep/set-aside offers, and nothing is ever demoted unattended. There is no revival path
      either: a measure is demoted automatically so a sighting can undo it, while a correction
      only leaves the checklist because a person put it there.
      *Also:* a proposal that was not yet a record saw no standing corrections even when the
      caller named the market — the flagship "judge this pitch" flow, missing the client's own
      rules. `times_seen` was the only count on the surface and is not the one the gate reads
      (five mentions with no `campaign_id` showed 5 while the gate read 0). The rendered rule
      list and the joined provenance were both unbounded. `list_corrections(status="expectd")`
      returned an empty list with no error.
- [x] **8.7 Replay as a report** — which campaigns now fail a new expectation; which past
      verdicts would change. Never silently rewrite old verdicts.
      **"The replay is a report" is the whole design.** Nothing is written, nothing is marked,
      and running it twice changes nothing — so "never silently rewritten" is true by
      construction rather than by discipline, and there is no stored judgment carrying a
      verdict about itself that a reader has to trust.
      **"Which past verdicts would change" is a claim this server cannot honestly make**, and
      saying it would be the confident unfounded assertion the whole review is written
      against — it cannot re-run the model, and §7.7 exists precisely because a model asked
      twice does not always answer the same way. What it knows exactly is what judging again
      would MECHANICALLY meet, and each row carries that as a `consequence` from a closed
      vocabulary: `stated_basis_withdrawn` (the verdict rested only on rules somebody has
      since set aside — the most urgent row this report can produce), `rests_on_withdrawn`,
      `gap_appears` (the subject has no measured value for something that became expected
      after it was judged), `rule_not_applied`. Each is a fact about the EVIDENCE, computed
      identically for every reader. A check that would simply pass is not listed at all.
      **The review's premise does not hold yet and the implementation does not rest on it.**
      *"Every verdict is stamped with its rulebook version"* — and `RULEBOOK_VERSION` is one
      literal string, identical on every judgment, until §12.1 ships a rulebook. The report is
      built from WHEN each measure and rule was confirmed instead, which says *which* rule
      arrived and when rather than only that something did. D115 records what breaks when a
      real rulebook lands.
      **The backlog is grouped by who to ask and what for**, because one conversation per
      partner per measure is the unit of work the review describes. Campaigns that have not
      concluded are counted separately rather than listed: you cannot ask a partner for a
      number that does not exist yet, and dropping them silently would confuse "nothing to
      chase" with "I did not look".
      Closes **D104** — the blast radius, on the offer where somebody actually decides.
      *Found by review:* **`replay.run()` wrote twelve rows on its first call.** The metric
      registry seeded itself on first read, so a report whose entire claim is that it writes
      nothing wrote — on exactly the database a customer meets first, an upgraded library
      where the registry is empty until something touches it. Seeding moved to
      `store.upgrade`, where a shipped payload belongs.
      **Re-confirming a rule made the server assert something false about a judgment that
      CITED it.** `confirmed_at` was overwritten on every graduation, so a rule graduated,
      cited by a blocking finding, set aside and re-graduated looked newer than the judgment
      citing it — and the report said it had never been checked against it. `COALESCE` keeps
      the first confirmation, because that is when the thing became a rule.
      Also: a judgment whose subject ALREADY carried the measure was listed with the same
      sentence as a real gap (a non-event, and `expected_check` was right there); a verdict
      resting entirely on withdrawn rules read identically to one mentioning one in passing;
      superseded versions made one conversation look like two; a rule reopened but not
      re-confirmed vanished from the report entirely; and the payload was unbounded — the
      third time this phase has had to bound a model-facing list.
      **And `replay_rules` was reachable from nothing but its own definition** — the third
      instance this phase of a tool the model would have to know existed and spontaneously
      call (§8.3's gate, §8.6's `note_correction`, this). Graduation is the exact instant the
      report stops being empty, so that is where it is offered.
- [x] **8.8 `bulk_import_metrics` diffs columns against the registry before writing.**
      **"Before writing anything" is the shape of the whole item.** A workbook carries a COLUMN
      VOCABULARY, and the moment to look at a vocabulary is once, as a vocabulary — not forty
      times, one key at a time, after the writes. The import previews by default and writes on
      `confirm`, which is the pattern `upload_campaign` and `add_metrics` already use for
      exactly this reason: the preview IS the consent step.
      **Four answers, because the review asks for four.** What is already known (and which
      measure); what it THINKS are aliases — a suggestion, never a merge, because §5.1 and §8.2
      both settled that a wrong alias silently folds two different measures together and a
      workbook is the worst place to get that wrong; what is genuinely new; and what it cannot
      type, which is otherwise discovered row by row as the import half-fails.
      *Found while reading the code:* **the bulk path called `store.add_metrics` directly**, so
      a workbook import never reached `metrics.record` and never touched the registry at all.
      The item the review calls "where the registry gets seeded" seeded nothing, and forty
      columns went into a JSON blob exactly as the review says they must not. It routes through
      `core.add_metrics` now, inside a per-row SAVEPOINT with the commits batched — a commit per
      row is an fsync per row, which is why `commit=False` existed, and routing through the
      registry would have reintroduced four hundred of them.
      **`target` became a real `metric_type`.** §5.1 refused it deliberately, with the
      distinction spelled out and advice to put the number in the campaign's prose. §8.1/D33
      then built the place it belonged — a `metric_type` of its own, comparable against the
      actual — and the refusal outlived it by two items, still telling people to put a number
      into freeform text that nothing can compare. The DISTINCTION is unchanged and is why it
      is its own value rather than a synonym of `predicted`: reconciliation scores the library
      against what it PREDICTED, and scoring it against somebody's ambition would make every
      calibration figure meaningless. A vocabulary's advice has to be retired when the thing it
      routed around gets built.
      *That change has a blast radius, and it found one:* `has_metrics` counts any row, so a
      campaign carrying only a target looked measured and stopped being asked for its results —
      the §8.1 lesson ("a target is not a result") one layer out, on a path that had only ever
      seen two metric types. `has_actual_metrics` is separate now.
      Closes **D47** — the structured retry (`field`, `valid`, `suggestion`) on the batch path,
      which the plan itself named as the likeliest place "Target" arrives, and where it was
      flattened to a string.
      *Found by review, all reproduced:*
      **The import consumed forty questions and showed none of them.** `metrics.record` marks a
      measure surfaced and offered as a SIDE EFFECT, and the batch discarded `core.add_metrics`'s
      return — so §8.2's "is this a new measure?" and §8.3's graduation offer were burned for
      every column at once and could never be asked again. The item whose headline is "never
      silently accept" made forty acceptances unaskable, permanently.
      **"Target Reach" was stored as a measured reach.** `target_reach` decorates to a stem
      ending in `_reach`, so the sibling-column shape every KPI workbook uses filed a target as
      a result — §8.1's founding distinction, through the door this item opened. A metric type
      spelled into a column name is read as one now, and CONFLICTS with the row's rather than
      overriding it silently.
      **Targets counted towards the graduation gate**, so three target rows could make a
      measure nobody had ever measured a standing requirement — and `expected_check`, which
      correctly counts actuals only, then reported those same three campaigns as missing it.
      Two halves of one item disagreeing about what counts. A target no longer revives a
      retired measure either.
      **Two new columns in one workbook were silently merged.** `dwell` and `queue_dwell` both
      classified `new`, then the suffix rule matched the second against the provisional entry
      the first had just created, with dict ordering deciding which survived. A provisional
      measure is no longer an alias root: nobody has confirmed it, so treating it as one
      asserts a relationship nobody agreed to.
      **A workbook's Month, Store # and Campaign columns became KPIs.** They are numeric, so
      each accrued sightings and could graduate — the "forty new keys" drift, produced by the
      tool built to stop it. A fifth class says so and they are not recorded.
      **The preview previewed the wrong half.** `errors: []` was asserted rather than computed,
      so "nothing has been stored, send with confirm=True" could be followed by three hundred
      of five hundred rows failing on an unmatched title. Rows are validated now — identity,
      `metric_type`, shape — without writing anything. A `structured` that was not an object
      also killed the whole batch with an `AttributeError`, which is not a `ValueError` and so
      reached the model as "Error executing tool" with no row index.
      **A skipped cell was silent.** The preview said a column would not be imported and the
      write skipped it without a word, which is the same silent acceptance wearing its other
      face. `skipped` says which cell and why.
      **And the batching was illusory.** Without an explicit `BEGIN` the per-row SAVEPOINT was
      the outermost one, and releasing the outermost savepoint COMMITS — so the import fsynced
      once per row exactly as it had before, while the comment claimed otherwise. A false claim
      about cost is worse than none, because it stops anybody measuring.

## Phase 9 — Execution drift and context events

- [x] **9.1 Asset `phase`** — `proposed` vs `delivered`, plus date.
      **The default is the load-bearing decision.** Every asset in a library today came out of
      a deck, so an upgraded database has to read `proposed`; defaulting the other way would
      have §9.2 compare a library of briefs against itself and report zero drift with total
      confidence. `captured_on` belongs to `delivered` and is refused on briefed creative — a
      capture date on a render says a photograph exists of something that has not happened —
      and it is stored normalised, because `fromisoformat` also accepts `20260314` and a later
      comparison against a campaign window is a string comparison.
- [x] **9.2 `compare_execution(campaign_id, delivered_assets[])`** — as briefed / never
      appeared / new, plus a drift score from visual distance. Machinery already exists.
      **Two instruments, two questions, kept apart.** A fingerprint is an identity claim — this
      photograph is that render — and is exact enough to state. A visual distance is a claim
      about resemblance, which is a number and not a verdict.
      **It takes only the campaign**, not the plan's `delivered_assets[]`: §9.1 said everything
      falls out of the phase field and it does, and a second inline upload path would be a
      second way to attach an image with its own resolution rules. `upload_image_assets` is the
      bulk path instead — fourteen photographs were fourteen uploads and fourteen tool calls,
      and `ingest_campaign`'s own comment already recorded that a call per image "isn't a
      workflow anyone would actually use".
      *Found by review, all reproduced:*
      **A fingerprint cannot do the job the item assigns it.** `images.py` scopes pHash to the
      same FILE resized or recompressed, and that is what its regression baseline verified. A
      photograph of a thing physically built will not land within Hamming 8 of the render of
      it — so on the review's own motivating case, a perfectly executed campaign reported
      "0 as briefed, 3 never appeared, 14 new" with `computed` stamped on it. A visual second
      pass now offers `looks_like` on the unmatched, marked `heuristic` (§7.8's third basis,
      which exists for exactly this), and it never moves an item between lists: a resemblance
      is not an identity claim.
      **One briefed image on two slides read as two briefed things.** The deck puts the hero on
      the cover and again on a detail slide, and the extractor dedupes by sha256 — two files,
      one image. A photograph of it satisfied one and the other reported NEVER APPEARED, on the
      list whose entire claim is identity. Briefed images are clustered first.
      **And the answer depended on upload order.** Taking each photograph's closest unclaimed
      render in turn made the same four images produce opposite verdicts on the same briefed
      element. Pairs are assigned globally from the closest, so the result is a property of the
      images rather than of the sequence.
      **A blank frame matched a colour swatch at distance 0.** A perceptual hash measures
      structure and an image with none hashes identically to every other image with none — flat
      red, flat blue and a blown-out frame are all `8000000000000000`. Decks routinely carry
      solid-fill rectangles, so a blank wall in a delivered photograph reported a swatch as
      having been built.
      **The drift score fell when you supplied more evidence.** Measured: 0.84 to 0.47, purely
      from uploading eight photographs of one shoot instead of one, because a centroid contracts
      toward the mean as a set grows. It is the mean PAIRWISE distance now, which is also the
      same kind of quantity as the yardstick it is divided by. The yardstick itself exploded to
      442 on a brief of two near-identical renders, and said "there is only one briefed image"
      about a brief with two.
      **Two of the three lists carried no evidence at all**, against the item's own "matching
      evidence attached per item", while the matcher had already computed every distance and
      thrown the near-misses away. And a half-indexed set computed a score from whatever subset
      had vectors and called it `measured`; a library that never embedded an image crashed with
      `OperationalError` rather than degrading.
- [x] **9.3 Named commitment checking** — extract the commitment list at upload, check each
      against delivered photos.
      **"Not visible in 14 delivered images" is the review's own phrasing and it is the whole
      honesty of the item.** Not "absent", not "missing", not "was not delivered". The server
      can say what it could not see in the photographs it was given; it cannot say the photo
      booth was not there, and a product that reports "absent" from their silence accuses a
      supplier who may well have delivered exactly what was promised.
      **Two claims, two bases.** That a LINE IS IN THE DECK is a fact, so `source_line` travels
      with every commitment and the reading can be disagreed with. That the line is a promise,
      and that a photograph shows the thing, are inferences — `heuristic` throughout.
      *Found by review, all reproduced:*
      **The feature returned NOTHING on the one path it was written for.** In OOXML a bullet is
      paragraph formatting, and on an ordinary deck it is inherited from the layout rather than
      written on the paragraph at all — so `text_frame.text` hands back the words with no
      list-ness, and a real PPTX experience list produced zero commitments. It worked only when
      somebody typed markdown into `detail`. `extract` now reads the list structure, and a
      single-paragraph placeholder is not a list, because the marker goes into the text this
      library stores, quotes against and searches.
      **A promise about how MANY confirms itself.** "Single colourway across all recipients" —
      the review's own third example — run through "which photograph most resembles this
      phrase" is guaranteed to come back `present`, because more colourways means more hero
      images means a higher maximum. The verdict would have read as the promise being kept, on
      exactly the evidence that it was broken. Quantified, universal and negative claims are
      refused rather than answered wrongly.
      **One absolute threshold cannot exist.** CLIP constrains only RELATIVE order, so every
      phrase has its own floor: measured against the real weights, "Jan 12 — kick-off" scored
      0.246 against a solid green square while a genuine promise scored 0.231 against noise. A
      photograph now has to beat the phrase's own median across the delivered set, which is
      per-phrase calibration from data already in hand — and every verdict carries the closest
      three photographs, because "go and look at these" is true whatever the threshold does.
      **An agenda slide produced six commitments**, a team slide four — "Maria Gonzalez,
      Account Director: not visible in 14 delivered images" is a post-mortem line about a
      person — and the cap took the first twelve in document order, so fifteen objective
      bullets on slide two used every slot before the floorplan on slide twenty.
      **And the headline sentence accused a supplier on its own.** With every item `unchecked`,
      the report still said "0 of 4 named promises can be seen… the rest were not visible in
      them". The summary also claimed a wider search than happened when a photograph was
      unindexed.
      *Also:* a title typo re-ran extraction, deleting and reinserting every open commitment,
      so the `commitment_id`s the model had just shown went stale. And the extracted list first
      reached a human as verdicts in a post-mortem — D116's sixth occurrence, in the item after
      the one that recorded the process gap — so it is shown at upload now, where it can be
      corrected before it says anything about anybody.
      *Found separately, outside this item:* `_resolve_asset` returned bare strings in a
      warnings list that `notices.collapse` reads as dicts, so an ordinary wrong path crashed
      `upload_campaign` with `TypeError` — not a `ValueError`, so the model got "Error
      executing tool" with the reason discarded, in the one case where saying "that path does
      not exist" is the entire job.
- [x] **9.4 Classify drift** improvement / neutral / degradation, marked `judged`.
      **"Presence and absence are facts; whether a change was good is a judgment."** That
      sentence draws the line the server must not cross, so `drift.py` RECORDS a classification
      and never makes one: no default, no inference from the drift score, nothing anywhere that
      reads a high score as a bad sign. A claw machine replaced by something better and a claw
      machine that never turned up produce identical numbers.
      **"Usually needs the outcome to settle it" is a value, not a caveat.** `too_early` is the
      honest answer before results, every classification is stamped with whether the outcome was
      known WHEN IT WAS MADE, and nothing is overwritten — "this looked like an improvement
      before the numbers and a degradation after" is the most interesting thing the record
      holds. `drift_to_revisit` re-asks the `too_early` ones once a result lands, because the
      classification offer fires when the photographs arrive and that is normally before the
      numbers.
      *Found by review, all reproduced:*
      **§9.4 punished improvement at the one layer where drift changes a verdict.** The citation
      caveat was keyed on the computed status alone, so a departure a named person had judged an
      *improvement* carried the identical "the brief may not have been it" discount as one
      judged a degradation. The item exists to stop the library punishing improvement and that
      is precisely where it was punished; `_execution_note` now reads the classifications.
      **The offer could not be satisfied.** Classifications were keyed on a free string that the
      offer prefilled with `asset.file_path` — a uuid by design. The question read "say whether
      a3f9c21d88e04b17.png not matching was a loss", a person answering in their own words filed
      against a different item, the count never fell and the same offer re-fired forever. Keyed
      on a resolvable `subject` (`asset_id` or `commitment_id`) with a human `about` line, and
      an unresolvable subject is refused with what to pass instead.
      **The count and the offer were different lists.** `unclassified` counted asset-keyed
      events; the offer was drawn from commitments. Answering the server's own question moved
      nothing. One population now feeds both, a `never_appeared`/`new` pair counts once, and a
      promise that WAS visible is never offered — a question with no true answer gets a made-up
      one.
      **The computed downgrade was uncorrectable.** §9.2's own comments say a photograph of a
      built claw machine will not fingerprint-match the briefed render, so every campaign with
      site photography is stamped `drifted` — and all four judgments on offer began "the
      execution moved away from the brief". A person looking at the photographs had no way to
      say "it did not; you just couldn't tell". `not_drift` is that correction: the one
      classification that disputes a fact rather than weighing one, which is why it is a
      person's and is visible as such. The computed status stays what the instrument saw.
- [x] **9.5 Attach drift to the outcome record**; high-drift campaigns are weaker precedent
      and retrieval says so.
      **The caveat travels WITH the citation.** Every evidence row carries an `execution` note
      saying whether that campaign ran as briefed, because a result from a campaign that drifted
      is evidence that something worked and the brief may not have been it — and a result from
      one nobody checked is neither. It says WEAKER, never disqualified: dropping high-drift
      campaigns throws away the evidence this phase most wants kept.
      **Snapshotted at every moment the answer can change** — results arriving, photographs
      arriving, a classification being made. Taken only on `add_metrics`, it froze every real
      campaign at `never_checked`, since the wrap deck lands weeks after the numbers.
      *Found by review, all reproduced:*
      **`never_checked` and `as_briefed` are different facts and the stale snapshot said the
      wrong one.** Stale is worse than absent here, because `never_checked`'s own sentence
      asserts that nobody looked.
      **Partitioning the ranked evidence cannot answer "did anything run as briefed".** Same
      argument as §6.4's poles, one axis over: on a library where the five nearest all drifted,
      partitioning those five reports "nothing here ran as briefed", which is a statement about
      the RANKING dressed as one about the library. `ran_as_briefed` is a filtered retrieval
      pass, and it is never silent on empty.
      **`performed_well` was the one pole without the caveat** — "this one worked" is the line a
      reasoner leans on hardest, and it arrived as a bare endorsement of the brief.
      **A bulk import ran the whole comparison once per row** — fifty workbook rows for one
      campaign meant fifty identical comparisons and forty-nine discarded snapshots. Once per
      campaign, after the loop.
      **The snapshot swallowed every exception**, leaving the campaign reading `never_checked`
      forever — indistinguishable from nobody having looked, which is the distinction the whole
      item rests on. It still does not raise (a result must file), and it no longer disappears.
- [x] **9.6 `context_events` table** — date range, scope, kind, description, source, stated
      impact; auto-linked by market + window overlap.
      **"No one should have to remember to connect them" is the item**, so there is no join
      table — a maintained link IS the remembering. The overlap is computed from (scope, range)
      against (market, window) on every read, because both halves keep arriving: an earthquake
      is recorded weeks after the campaigns it overlapped, and a campaign is uploaded months
      after its market's calendar was seeded. A link written at either moment is wrong about
      the other.
      **A campaign had no window, and that was the load-bearing gap.** Dates existed only as
      prose `facts.py` parses for contradictions, so there was nothing for a date range to
      overlap WITH. `starts_on`/`ends_on` is what this item turns on, the way `phase` was
      §9.1's — settable at upload, at update, and from a workbook's own date columns (D119,
      closed here: a `Month` column was being dropped as "not a measurement").
      **The link is an overlap and nothing more.** "Ran during" is a fact; "affected by" is an
      assertion, and a model asked to explain a disappointing number will reach for whatever
      is nearby. §9.8 owns `confounded`; this owns not getting there early.
      *Found by review, all reproduced:*
      **Three routes to a false clean bill, in the class the module was built to prevent.** The
      `nothing_to_check`/`checked` split exists so an empty list never reads as a claim — and a
      window read out of prose defeated it (a stray asset-spec date became a one-day window,
      found nothing, and the product said "that is a real answer"), as did a record naming no
      market (only global events could reach it), as did a window inverted across two calls
      (validated only when both ends arrived together, so a one-field typo stored 2026-12-01 to
      2026-03-31 and printed it backwards inside the sentence asserting nothing had happened).
      **The two directions were two implementations and gave two answers.** SQLite's
      `COLLATE NOCASE` folds ASCII only and Python's `.lower()` folds Unicode, so recording an
      event said it reached a campaign and opening that campaign said nothing had. One stored
      fold now, matched identically from both sides.
      **The item's own headline was false in practice.** The link is automatic, but the WINDOW
      it trades for was asked for nowhere: not at upload, not in `after_upload`, and absent
      from every gap — so the fraction of campaigns with no context forever was whatever
      fraction nobody made a second deliberate call for, silently. §9.5 gave itself
      `execution_never_checked` for exactly this reason; this now has `no_window`.
      **The caveat reached nobody.** §9.5 wires `_execution_note` into four read paths; §9.6
      was wired into none, so a reader looking at an evidence row saw nothing about the port
      closure that ran through half the flight. `context` now travels beside `execution`.
      **Truncation hid the one event that mattered.** Ordered by date and cut at eight,
      "earliest eight" becomes "eight holidays" the moment §9.7 seeds a calendar — and the
      flood with a stated fourteen-day delay, precisely what §9.8 needs linked, is what
      disappears. Ordered by salience now, with the kinds surviving the cut.
      **A wrong event was permanent and contagious** — worse than the hand-maintained join it
      replaces, which at least lets you unlink. `withdraw_context_event` is §8.5's shape: kept,
      not deleted, and it says how many campaigns stop carrying it.
      **A monthly workbook made a twelve-month campaign a January one**, labelled `stated` so
      nothing downstream doubted it. A workbook window widens as its rows arrive, and never
      over dates a person typed.
      *Built for what comes next:* `context.for_window` takes a window and markets with no
      campaign involved — §9.7 checks a PROPOSED window that may not be stored, and §9.8 needs
      a measurement period narrower than an always-on campaign's year. `seed_key` makes §9.7's
      seeder idempotent and lets a corrected Ramadan date replace a wrong one on upgrade.
- [x] **9.7 Seed fixed calendar events per market**; `prepare_evaluation` checks every
      proposed window against them as a computed fact.
      **"Impossible to drop" is the phrase the item turns on**, and landing in `computed` is
      only half of it. §2.4's write protection stops the model FORGING a computed fact; what
      the review asked for is §7.8's persistence — so a clash the plan never names joins
      `_COMPUTED_FINDINGS` and is appended after the model's own list, exempt from the caps. An
      `approve` with no findings cannot make it disappear, and the figure is stamped onto the
      evaluation so §9.9 can reconcile against what the calendar actually said.
      **`should_fix`, never `blocking`.** The instruction everywhere else in this item is "do
      not turn it into a blocking finding on its own", and the server holds itself to the rule
      it gives the model.
      **The finding is the SILENCE, not the clash.** Mexico's deck DID flag the World Cup. A
      plan that names what it runs into has said what there was to say; raising one anyway is
      the nuisance that teaches a reader to skip server findings.
      **A shipped calendar is a claim about the world, and most of these claims are
      approximate.** Every row carries a `certainty` — `fixed`, `announced`, `observed`,
      `seasonal` — with the sentence that says what would settle it, and says it came from the
      product rather than from the customer. What is not hedged is the arithmetic.
      *Found by review, all reproduced:*
      **Five of the first thirteen shipped dates were wrong** — Ramadan by a day (against the
      review's own "18 Feb"), Chinese New Year by two (the row promised the published State
      Council dates and shipped others), Japan's Golden Week by one (3 May 2026 is a Sunday, so
      6 May is the substitute), Buen Fin by one, and the South-East Asian rainy season by half
      a year: mainland SEA is wet May–October and DRY November–March, so a Bangkok campaign in
      December was told it clashed and one in July was told nothing was on. Nothing tested any
      date; shifting the World Cup by seven weeks passed the suite.
      **The server asserted a falsehood as a computed fact.** "The proposal names all of them,
      so this is a timing it chose" was emitted over clash sets made entirely of customer
      records, which have no short name to look for — an affirmative claim about a document
      nobody read. It fired for the whole of §9.6's contribution.
      **Truncation deleted the check.** The clash reused §9.6's display list, which caps at
      eight by salience — and salience sorts seeded rows LAST, correctly for a list somebody
      reads and exactly wrong here. Eight customer records silently removed the World Cup,
      Ramadan and Buen Fin, and the fact then stated "8 thing(s)" as a count of what it saw.
      **`absent` was mostly "we do not cover your market".** Thirteen events is not the wrong
      size, it is the wrong claim: Brazil got "nothing on record was happening" on every
      window, under the heading telling the model not to re-derive it, with a hedge that blamed
      the customer. `not_covered` and `calendar_expired` are now separate answers — and without
      the second, 1 January 2027 was the day this feature started giving every window on earth
      a silent clean bill.
      **Ramadan was scoped globally**, so a Polish spring retail push was told as an established
      fact that its plan failed to mention it. Scoped to the markets an event is about.
      **The silence test was a substring match on one handle per event.** "Timed to the FIFA
      tournament" read as silent; "Heidi Reid" silenced Eid. Word boundaries, accent folding and
      per-event aliases now, and the affirmative half of the sentence is gone entirely — a false
      positive lands as a neutral question, a false negative used to land as an endorsement.
      **A heuristic window was relabelled `stated`** on the record path, so `campaign_context`
      hedged "may have run during" while `prepare_evaluation` asserted a clash on the same
      record from the same dates.
      **`date_coverage: absent` fired beside a fact naming the window twice** — the brief was
      formally faulted for carrying no dates in the same block that quoted them.
      **Re-seeding rewrote all thirteen rows on every start**, churning `created_at` and
      silently reverting any local edit; and `seed_key UNIQUE` did not survive the migration,
      so upgraded databases had none of the uniqueness the schema comment promised.
- [x] **9.8 Record overlap, never assert cause** — mark outcomes `confounded`; the caveat
      travels with the metric wherever it is cited.
      **"Confounded outcomes still count" decides the design.** Dropping them would throw away
      most of what a real library holds — every campaign that ran through a holiday, a port
      closure or an election. So nothing filters, excludes, down-ranks or reweights; the number
      is the number, and what changes is that a reader can no longer be told it was clean.
      **"Everywhere it is cited" means one choke point, not a list of call sites.** Metrics load
      in exactly one function, so the caveat attaches there and every reader gets it, including
      readers nobody has written yet. A hand-maintained list of places to attach it is a copy of
      the codebase, and this project has been bitten by that shape four times.
      **Attribution is a person's, always.** The server records that a metric and an event
      overlapped; whether the event moved the number is `stated`, needs a name against it, is
      refused when the event never overlapped, and is refused when the name reads as the
      product itself.
      *Found by review, all reproduced:*
      **The escape hatch the whole design rests on was unreachable.** A recurring date confounds
      only when somebody says it did — and `attribute` validated against the DISPLAY list, which
      caps at eight by a salience that sorts seeded rows LAST. So the only events it could cut
      were exactly the ones needing an attribution, the server refused with "it did not run in
      its market" about an event that plainly did, and the hatch closed as soon as a library got
      rich. Third instance of this bug shape in three consecutive items.
      **The product contradicted itself on the review's own example.** §9.7 raises a `should_fix`
      that a window runs into the World Cup; §9.8 then told the reasoner to read the result as
      clean. Every row in the shipped calendar is `fixed_calendar`, so keying on kind put a
      once-in-a-generation home tournament in the same bucket as Black Friday. The axis is
      RECURRENCE, and `competitor_launch` moved off the unconditional list — competitor launches
      are continuous background, so a diligent customer would have turned every outcome
      confounded through their own diligence.
      **The caveat certified cleanliness.** A campaign with no overlaps was silent while one with
      two was told to "read it as a clean result" — so the more the library knew was going on,
      the more affirmatively it said nothing was. The false clean bill, inverted.
      **January's sell-through was confounded by a September earthquake**, on a row that
      literally carries `month: 2026-01` — a column the importer already parses and was
      discarding. Metrics now carry their own measurement period, and the fallback to the
      campaign window says it is one. It failed hardest for the customers with the most data.
      **A 120-row workbook produced a 742 KB response**: the same event list and the same
      858-character sentence copied onto every row. §6.8 exists in this codebase because a match
      with many metric rows was already heavy; this re-created it by another route. The row now
      carries the label and the count; the argument is said once per package.
      **`reconcile_evaluation` — the surface §9.9 is named after — cited the number with no
      caveat at all**, and had a latent `NameError` on the branch where the figures are passed
      in by hand.
      **"Never quoted as clean evidence" was hoped for rather than true.** Every signal was an
      input the model could ignore: an `approve` leaning on a confounded number and never
      mentioning it was accepted and left no trace. It now raises a server finding on §9.7's
      pattern — the finding is the SILENCE, `should_fix` and never blocking.
      **Targets and forecasts were marked "an outcome that ran through" something** — a category
      error the metrics schema comment already guards, and one that would corrupt §9.9.
      **`confounded: false` collapsed four states into one word**, including "this campaign has
      no window" and "it names no market" — the exact collapse its two neighbours on every
      evidence row were each written to prevent.
      **An attribution overwrote the last one**, against §9.4's explicit precedent four items
      earlier; a zero stated impact ("somebody looked and found no delay") was read as somebody
      saying it changed things; and `stated_by="campaign-poc server (computed)"` was accepted
      and rendered as though the library had worked it out.
- [x] **9.9 Make `reconcile_evaluation` do its job** — predicted vs delivered vs actual vs
      context, one record. It has never run.
      **Four columns, and three of them already existed.** §2.4 recorded what was said,
      §9.2/§9.3 what shipped, §6.5 what it did, §9.6–§9.8 what else was going on — each built
      for its own reasons and never once put beside the others. That independence is most of
      the value: a reconciliation assembled out of one subsystem's opinion of itself proves
      nothing.
      **The server lines up and scores arithmetic; it does not mark its own homework.** Whether
      3.4 falls inside 3.0–4.0 is computed. Whether "the six-week timeline is unrealistic" held
      is a judgment, and comes back `yours_to_judge`. `not_comparable` is a THIRD answer and
      never a quiet pass — a calibration figure built from unscorable rows scored as held would
      be this product awarding itself marks.
      **`calibration` is the only thing in the product that grades the product**, and it says
      what it does not cover: how many judgments could have been reconciled, how many were,
      how many ran through something else, and that one judgment landing is not a record.
      *Found by review, all reproduced. Every one of these flattered or punished the product's
      own score, which is the failure mode this item is uniquely exposed to:*
      **The range parser was wrong on most real formats.** `"3.0-4.0"` read the hyphen as a
      MINUS, giving −4 to 3 — so every prediction written with the commonest separator scored
      as a miss, in the direction nobody audits, because a product reporting its own judgments
      as worse than they were reads as modesty. Fixing that exposed six more: `"2%-3%"` brought
      the bug back whenever a unit sat before the hyphen, `"Q3 2026: 3.0-4.0"` made the year the
      upper bound so 1500 "held", `"3,5-4,0"` read a European decimal as 3 to 5,
      `"5,000-6,000"` became 0 to 5, `"up to 4"` became the point 4, and a single figure could
      hold only on exact equality. Every ambiguity now resolves to `not_comparable`.
      **Predictions were scored against the wrong measure.** The registry's prefix rule routes
      `ctr_lift` into the CTR family — right for storing it, wrong for deciding which number
      answers a prediction. A predicted 10–20% CTR *lift* was scored against a measured CTR
      *level* of 2.3 and reported "missed, below by 7.7": arithmetic on two different
      quantities, wearing the server's authority.
      **The one automatic route in was closed by the other automatic route.** §6.3's
      version-based reconciliation wrote a row, and "no reconciliation exists" then closed the
      RESULTS loop permanently — before any results existed. D85 created `basis` for exactly
      this distinction and nothing read it.
      **The same judgment could be counted as many times as it was saved**, and re-saving was
      also the only way to refresh a stale tally, so the product pushed users into inflating
      its own score.
      **The stored tally went stale and `calibration` read it**, so a corrected figure that
      turned a held prediction into a missed one left the pass standing. The figure is
      recomputed live now; the stored record remains as what the LESSON rested on, and
      `get_reconciliation` shows both with a `changed_since`.
      **Nothing said "you have fourteen judgments nobody has checked"** while `calibration`
      reported a perfect record beside them — the item whose whole diagnosis is "nobody decides
      to go back" gave itself no gap, which is the reason §9.6 gave itself one.
      **The bulk import dropped the offer**, on the path a customer actually loads a workbook —
      D116's shape for the seventh time.
      **It was a view, not a record.** The four columns were assembled and thrown away; three
      of them are recomputed over state that keeps moving. §9.5 solved this for itself with
      `execution_at_save` and wrote down why.
      **§9.5 and §9.8 both stamped evidence naming §9.9 in their docstrings, and §9.9 read
      neither** — so the then-versus-now delta existed for the calendar and nothing else.
      **A campaign-level prediction was scored against one arbitrary month** — insertion order,
      on a twelve-row workbook.
      **The lesson could be "ok", "." or an emoji**, and the repo's own fixtures wrote
      "Recorded." into the one field carrying what the product learned about itself. §9.8 one
      item earlier refuses a `stated_by` that reads as the product; adjacent items, opposite
      rigour.
      *Closes D40, D120 and D127, all three re-pointed here:* a judgment made about a pitch
      nobody had stored could never be attached to the record it became (`link_evaluation`),
      and reconciliation did not walk the supersession chain — so a re-briefed campaign
      reported `nothing_to_check` and offered to record results against a brief that never ran.

## Phase 10 — Feedback capture

- [x] **10.1 Define "open"** as *needs something from a human*, ordered by value.
      Eleven reasons in `feedback._REASONS`, each with a rank and the sentence that says why
      the row is open. *Closes D20, re-pointed here:* `campaign_notices` persists the
      `blocked` and `degraded` warnings, so the thing the library most wants somebody to act
      on stopped being the thing it forgot fastest — it is now the top-ranked reason.
      *Closes D70 and D87*: a market resting on one campaign, and results on file beside a
      claim that is still somebody's impression.
- [x] **10.2 `feedback_queue()`** with fixed numbering (1–9 campaigns, 10 concluded, 11 more).
      **Demanded a name and discarded it** while saying it had been kept — found by the first
      review round on the write path.
      *Closes D53, D59, D84 and D110, all four re-pointed here,* and they turned out to be one
      thing: **a question this product asks and cannot hear the answer to.** "Is this
      departure deliberate?" was asked on every judgment and the reply died in a chat window,
      so the same question came back on the next version — and the UN-answer was stored:
      `departure: unexplained` is a fact every later judgment reads to decide severity, and it
      is false the moment somebody explains it. A question nobody can answer decays into a
      wrong answer, which is worse than not asking. `answer_finding` and `answer_gap`
      are where the answer goes, attached at `store.get_evaluation` so it reaches every
      reader; D110 needed no write path at all, only to be *asked again*, which is the
      queue's job. *Closes D52:* the ranking could not see magnitude, so one unindexed record
      outranked forty.
- [x] **10.3 Numbered all the way down**; free text invited, never required.
      The free-text box is where "slide 23 should be the standard for every market" gets
      captured — the highest-value sentence in the system — so it is invited and never
      required, and it is stored in the speaker's words rather than folded into a tag.
- [x] **10.4 Server builds the menu, not the model.**
      Including the rendered `line`, which is the point: handing the model a title, a market
      and a paragraph and hoping two operators format them the same way is the variance this
      item exists to remove.
- [x] **10.5 `menu_token`** so a stale selection refreshes instead of writing to the wrong row.
      *Closes D42, D117 and D118, re-pointed here.* The token is over the ROWS as rendered, so
      anything that would change what a number means changes it. D117/D118 are the same
      problem with a bigger argument — an offer that has to survive a round trip — and
      `import_batches` answers both: the preview stages the workbook and hands back a
      `preview_id`, so confirming quotes a key instead of resending 500 rows, and the same key
      resumes an import the time budget cut short. The old note promised "nothing already
      imported is duplicated by doing so" and **nothing enforced it**: the obvious reading of
      "send them again" doubled every row already written.
- [x] **10.6 Offer the queue proactively** — after uploads/evaluations and on first
      interaction of a session. *"The menu is worthless if the user has to know it exists"* —
      which is true of the menu and not only of it. *Closes D54, D67, D76, D98 and D116.*
      **D116 is the one that matters**, because it had happened seven times and each was found
      by a reviewer rather than by anything in the code: `tests/test_every_tool_is_offered.py`
      is that checklist executed instead of remembered, and it found **nine** unoffered tools
      on its first run — `save_evaluation` among them, which the whole second half of the
      product starts at. It then caught 10.2's two new tools the moment they were added.
      **A tool nobody offers is a tool nobody calls, and a feature nobody calls cannot fail,
      so nothing reports it broken** — §8.6's gate had nothing to act on for exactly this
      reason. `gaps` and `coverage` are offered where they are cheap; D76 folded the two
      copies of the first-steps path into one, since they could disagree about the same
      library while each stayed internally consistent; D98's partial re-index raised no
      notice at all, so an edit that half-failed was observably a successful edit and the
      record was unfindable by its own new wording.

**Two review rounds followed, and between them found nineteen defects the 2,032-test suite
could not see.** The shape that recurs is *a new thing quietly switching off an old one*,
with nothing to notice because each half works perfectly alone:

- **One pair of similarly-worded corrections disabled every proactive offer in the product.**
  `waiting()` filtered on a key the new rule row does not have; the KeyError went into a bare
  `except` and came back as `0`. 10.6's entire headline, switched off by 10.2's own new row,
  permanently and invisibly, in the ordinary state of any agency library.
- **`diff_campaigns` crashed BECAUSE the user answered the question the product asked** —
  rewriting `departure` to `explained` hit `_reread`'s positional index. The same rewrite made
  the finding unreachable by `get_evaluation(departure="unexplained")`, and `explained` could
  not be passed through the MCP schema at all. The product ate the finding. The rewrite is
  gone; the predicate moved to the reader instead.
- **The import resume was a TOCTOU**: the guard read `imported_through` at entry and wrote it
  at exit with the whole import between, so two sessions on one key both imported every row —
  the exact corruption D118 removed, arriving through the key that removed it.
- **`share` divided judgments by campaigns**, so one record with three unchecked judgments
  reported `share: 3.0` and claimed to outweigh its kind. A ratio of two units is a number
  that cannot be wrong because it means nothing, and it was published beside
  `rank_basis: computed`.
- **The set-aside whitelist gated the offer and not the write**, so `answer_gap` accepted
  `library_is_empty` and `gaps()` then returned `[]` — the product saying nothing is missing
  about a library holding nothing.
- **`chunk_not_embedded` was retracted only by `finish_indexing`**, so a partial edit followed
  by a successful one left a wholly searchable record permanently occupying `needs_attention`,
  the top-ranked reason a campaign waits on a person.
- **`_rule_questions` was O(n²) on every upload** — 8.4s at 400 corrections, a third of the
  tool's whole time budget, on a library that is not large.
- **The offer that says "a question needs your answer" prefilled the answer**, against
  `actions`' own written rule, so "we fixed that" could be recorded as "we kept it on purpose"
  under the user's name.
- **`attribute_outcome` asked a yes/no question with no way to say no** — D84's failure
  reintroduced in new code, in the same diff that fixed it. `bears_on=False` now exists.
- **The override log was write-only.** §10.2 gave a person's answer somewhere to go and
  nothing to read it back with; for a product whose pitch is that its judgments can be argued
  with, the record of where somebody argued and won is the most valuable thing it holds.
  `answers` is that read, and `stale_answers_offer` is the re-ask the queue's own founding
  insight demanded and never applied to itself.

## Phase 11 — Attribution

- [x] **11.1 Environment-derived author** — `os_user`, `host`, config display name.
      Ships with 11.2, because an author the server derives and does not label is exactly the
      unlabelled claim this product refuses. `identity.captured_by()` never accepts anything:
      a parameter would make the one fact this server can establish for itself into one more
      claim it has to take on trust.
      **The case the item did not foresee, and the one that matters most:** `unattributed`. A
      shared HTTP deployment with no identity provider would otherwise report the OS account
      it was STARTED as for every caller, filing nine people's decisions under whoever
      launched the service.
- [x] **11.2 Reuse `verified` vs `stated`** for author rather than inventing a vocabulary;
      add `method` (stdio_local | sso | import). *Closes D35:* `TAG_SOURCE_SYNONYMS` is NOT
      reused — "measured" is a sentence about a number and meaningless about a person, and a
      synonym table accepting it would let `source: measured` through on an identity claim.
      **`method` is what keeps `verified` honest.** `stdio_local` means "the desktop account
      this runs as" — nobody authenticated, and reporting that as `verified` with nothing
      beside it would inflate a fact about a process into a fact about a person.
      *Closes D105:* sweeping for the guard found **five** implementations of "is this a
      person", two of which checked only for an empty string — so
      `withdraw_context_event(withdrawn_by="the system")` took an event off the record, and
      `link_evaluation(linked_by="Claude")` decided what a judgment was a judgment OF. The
      weakest door was deciding what the library believed about who decided things.
      A **sixth** door turned up when mutation testing removed the guard from `_keep_the_view`
      and no test noticed: `store.normalize_tags`, which is what actually decides what a tag
      STORES, had no check at all. So `said_by: "the team"` was refused from the append-only
      record and written onto the campaign anyway — `get_campaign` telling every reader the
      team held an opinion the table built to be the authority on opinions had never heard of.
      Two implementations of one rule, disagreeing, with the reader-facing one being the
      wrong half. The guard now sits at the single door both paths pass through and refuses
      the write naming the tag, and the copy in `_keep_the_view` is DELETED rather than kept
      as defence in depth: it was unreachable, and a second copy is how this drifts again.
      **Why the earlier sweeps could not have found it.** They grepped for the marker strings
      of a duplicate implementation, and a door with NO check leaves no marker. That is now
      `test_every_write_that_takes_a_name_reaches_the_one_guard`, which walks the AST for
      every function taking a name-shaped parameter and asserts it reaches `identity.person`.
      `store.record_authorship` deliberately does not count as the guard: it fires AFTER the
      row is written, so a caller relying on it raises the right error having already stored
      the thing — which is what `drift.classify` did, refusing "the system" with the judgment
      already on file under it.
      *And the check itself was refusing real people.* `reads_as_the_product` was a substring
      scan, so "Themba Nkosi" was the library because of "them" — along with Claudette,
      Automne, Sautoy, Lautoka, Matthey, Serverin and Autolycus. That is not a safe failure:
      it is a hard block on recording somebody's view, it falls hardest on names that are not
      Anglo, and the remedy the message offers — give another spelling — does not exist for a
      person's own name. It strikes the disqualifying words out and asks what is left, which
      keeps "Claude Monet" a person and "Claude" not one.
- [x] **11.3 `captured_by` vs `on_behalf_of`** — the field most products miss; asked as one
      numbered question. They miss it because it only matters later: on the day the note is
      written everybody knows who was in the room, and two years on the record says a name
      with nothing to say whether that person held the view or merely typed it.
      Numbered per §10.3 — the operator is by far the commonest answer and the server already
      knows their name, so spelling it was the one bit of typing the menu had left. Offered
      only when a display name is CONFIGURED: `os_user` is an account, and "1 = nbhatt" would
      make the easy answer the wrong one. `speaking_for_themselves` is computed from the two
      names, and is `None` rather than `False` where the account has no name to compare —
      answering "no, somebody else" out of an absence would be a claim made from ignorance.
- [x] **11.4 Context** — `captured_at`, channel, session id, role **as stated at the time**.
      The emphasis is the whole point: a role looked up later is the role somebody holds
      TODAY, so a planner who becomes head of strategy would retroactively have made every
      past decision as head of strategy, the record silently gaining authority nobody granted
      it. Stored with `role_basis: "as stated at the time"` beside it, optional, never
      inferred. Session is per PROCESS, which is what a session is for a stdio server — it
      makes "eleven decisions in one sitting" distinguishable from "eleven over a month",
      which is the difference between working a queue and agreeing with everything.
      *Closes D21:* `scope: machine` can now say whether the current user IS the account it
      is telling them to go and ask.
- [x] **11.5 Append-only reactions** — keep both sides of a disagreement, surface it in
      retrieval, authority order configured in the rulebook, never inferred. *Closes D10.*
      `update_campaign` REPLACED the tag list, so R. Vega recording `liked` in March and A.
      Duarte recording `not_liked` in June left only June — not superseded, not outvoted,
      gone, with nothing saying March was ever there. **That is the most expensive thing this
      library can lose:** a campaign two people disagreed about is stronger evidence about
      this client's taste than one everybody liked, and the whole premise here is reasoning
      from what this client thinks.
      **It does not decide who wins**, and `disagreement` has no `winner` field at all.
      Preferring the later view, or the client's, or the grander job title would be authority
      nobody granted it — and §2.5 refused exactly this once, because a PDF export turns
      speaker notes into annotations, so even the FORMAT cannot say whose words weigh more.
      Three distinctions, each with a test for the silence: two PEOPLE rather than two values
      (one person changing their mind is a revision); per AXIS ("they liked it and it
      underperformed" is the most ordinary finding in marketing); and agreement is
      corroboration, not a split.
- [x] **11.6 Backfill existing records as `author: unknown`** with import date, explicitly.
      *Closes D13.* A null was saying FOUR things at once, with different consequences for a
      judgment citing the words: `format_carries_none` (a speaker note has no author field —
      the agency talking to itself), `file_did_not_say` (a PDF annotation CAN carry `/T` and
      this one does not — somebody commented anonymously), `not_captured` (stored before this
      library read commentary; a fact about US, and reporting it as the document's silence
      would blame the customer's deck for our gap), and `never_claimed` (body text is not a
      comment). Told `null`, a judgment citing "a reviewer objected" cannot tell a client's
      objection from the deck's own note — §2.5's distinction, erased at the last step.
      The backfill never touches a name: overwriting one would destroy the personal data 11.7
      must be able to show and erase, while making the library look as though it never knew.
- [x] **11.7 Treat it as personal data** — install disclosure, per-person view, deletion or
      anonymisation preserving the judgment, stated retention position. *Closes D14, D22, D26,
      D30 and D111.*
      **The hard part is "or anonymisation PRESERVING THE JUDGMENT", and both naive readings
      are wrong.** Delete the rows and a judgment citing "R. Vega objected" cites nothing —
      the library asserting a finding whose evidence silently vanished. Redact to a blank and
      "two reviewers objected" becomes indistinguishable from "one reviewer objected twice",
      which is often the whole finding. So: a stable pseudonym per person, and the erasure
      itself on a permanent record — without the name, which would defeat it.
      **"Irreversible" is what this item originally said, and it was not true.** The token is
      derived from the name, so anybody holding this database and a list of candidate names
      can confirm a match — Art. 4(5) pseudonymisation, and pseudonymised data is still
      personal data. A per-database salt defeats the guess for somebody holding the token
      WITHOUT the database, and nothing more, because the salt lives in the same file. The
      tool is named `pseudonymise_person` and every user-facing string says so, which is the
      one claim here it would be worst to get wrong: a data subject told their name is gone.
      *D111's care:* the swap inside `correction_sightings.provenance` is whole-word, because
      replacing "R. Vega, client email, 4 March" wholesale would remove WHEN and WHERE a
      client said something — weakening a standing rule's evidence to remove a name, which
      D111 says explicitly must not happen silently.
      *D22:* a filesystem path does not look like personal data until you notice that on two
      of three platforms it carries somebody's login — in `detail`, the field this product
      explicitly tells people to send to support. The filename survives; the login does not.
      *D26/D30:* the disclosure is GENERATED from `retention()`, because a disclosure
      maintained separately is the copy that drifts — and here the drift is between what a
      customer agreed to and what the product does.
      **The worst sentence this product can produce, and it produced it.** `_scan` folded a
      name to NFC and found it; `_swap` then matched that folded pattern against the RAW
      stored bytes, so a macOS-decomposed "José" was found and not replaced — and the skip was
      never counted, so `status: pseudonymised` came back with a paragraph telling a data
      subject their name was gone. There was no partial-failure state at all. Both halves are
      fixed: the haystack is normalised too, and an erasure that leaves any row it FOUND
      unchanged rolls back and refuses. The guarantee is that either it is gone, or you are
      told it is not, and nothing moved.
      *The controller statement reads the deployment.* "A database only you can read" is true
      of the stdio install and false of the `serve` subcommand this product ships, where
      `auth`'s default provider is `none` and every request is anonymous — and it is, in its
      own words, the strongest fact this product has, which makes it the sentence a DPO will
      quote back. It is also IN the install disclosure now, rather than reachable only behind
      a tool call, which is the whole point of D26 naming a moment.

## Phase 12 — Rulebook as versioned configuration

- [x] **12.1 Skeleton `rulebook.yaml` + loader** — bundled, versioned, loaded at startup,
      applied deterministically **outside** the similarity path (today the rubric is a
      `reference` row retrieved by similarity, truncatable and deletable).
      **The parenthesis is the whole item.** Guidelines lived in the library as an ordinary
      record, so they reached a judgment only if they happened to rank in the top five for
      that brief — and the failure was silent AND the wrong way round: the briefs least like
      the guidelines document are the ones least likely to retrieve it, and they are exactly
      the briefs most likely to breach it. `readiness` said so on its `cannot` list, naming
      this item by number as the work that would fix it.
      Every rule is now put in front of every judgment in full, whatever the brief is about.
      Not ranked, not truncated, and not deletable by anyone editing the library: a rule in a
      row is editable by whoever can call `update_campaign`, and a rule in a file is something
      an administrator can see and a customer can put in version control — which is also what
      §12.2's overlay needs.
      **`RULEBOOK_VERSION` is gone**, replaced by `core.rulebook_version()` reading the file.
      A constant beside a file that can disagree is the hand-maintained-copy failure this
      project has hit four times, and here it would have made §7.6's stamp worse than useless:
      a judgment labelled with a version it was not made under.
      **What it does NOT claim.** `check_against_rules` moved onto `can` and carries a
      `bounded_by` saying what it is bounded by: only rules actually written in the rulebook.
      A guideline nobody wrote down is not checkable by anything, and an uploaded guidelines
      DOCUMENT is still retrieved by similarity like any other record — `rulebook_on_file` is
      still a separate, weaker row, and still says so. Replacing one overclaim with another in
      the one tool whose job is saying what this product cannot do would have been worse than
      leaving the limit in place.
      **It ships EMPTY, and that is the considered answer.** The first draft carried six
      rules about this product's own judgment discipline — and four of them already exist word
      for word in `EVALUATION_PROCEDURE`, which reaches the model three ways, while two more
      are enforced by validators that refuse the write. The file whose entire purpose is being
      the single versioned home of a rule would have shipped as a second copy of six. The
      product owns the discipline in the procedure; the customer owns the rules about their
      briefs, which is also why §12.3's content is an example file rather than the default.
      *Closes D27, D50, D74, D92.* `health_check` gains a `rulebook` component, because the
      rulebook is a shipped payload like the CLIP weights and fails the same way — the server
      answers, the database is fine, and every judgment is missing its rules. `expects`
      declares what a brief must CARRY, so `gaps()` can report a named input nothing supplies.
      `has_rulebook` no longer means "any reference record exists", which had been telling
      customers who filed a competitor teardown that their own rules were in force. And
      `watch_for` gives §7.1 its sixth computed fact: the server reads the brief for the words
      a rule forbids and attaches the sentence — reporting `nothing_to_check` where no words
      are declared, never `checked_clean`, because a brief said to breach no guardrails when
      none were looked for is the false pass §11.7 spent a whole item removing.
      **What review caught before it shipped**, and each was the item failing in its own
      terms: a finding could not cite a rulebook rule at all (`rule_id` resolved through the
      CAMPAIGNS table and demanded a `reference` record — the arrangement §12.1 exists to
      replace), so the product handed the model rule ids and refused every one; the startup
      load was on one entry point of three; `readiness` advertised rule-checking on an install
      with no rules written; and every frozen install would have failed to start, because
      PyInstaller puts `datas` under `_internal/` while `config.app_dir()` looks beside the
      executable.
- [x] **12.2 Customer overlay file** layering on the default, with the version stamped onto
      every saved judgment (pairs with 7.6). *Closes D36, D71, D72, D73, D93, D101.*
      In the DATA directory beside the database, not beside the executable: the install
      directory is Program Files on Windows and an installer replaces it wholesale, so
      "write your rules in rulebook.yaml" would have meant "write them in the file the next
      upgrade overwrites" — a worse trap than not offering the file at all.
      **The stamp could not be deferred.** `compare_provenance` diffs `rulebook_version` and
      concludes two judgments were made "under the same conditions, so an agreement between
      them is evidence rather than luck". With one scalar, two judgments under two different
      sets of the customer's own rules both stamped `core-1.0` and were called comparable —
      false, in the tool whose purpose is explaining disagreement. Widening the field now was
      a field; after rows are in the field it is a migration.
      **A vocabulary is a SYNONYM, not a rewrite** — D72's own words. The first version
      canonicalised on write, and review showed that made declaring one actively harmful:
      rows written before kept the old spelling, rows after got the new one, and a query in
      either found half of them. Two spellings that had at least folded to one cell became
      two cells with thin evidence each — the precise harm the feature exists to remove,
      inflicted by the feature, on exactly the libraries that already had data. Resolved when
      two values are COMPARED, a declaration reaches every row ever written, rows added
      tomorrow from a spreadsheet that still says the old word are found too, and withdrawing
      it puts everything back.
      *D36's line:* stage names are the customer's and `verified`/`actual`/`reference` are
      not. A customer mapping "confirmed" onto `verified` would make a stated impression
      outweigh a measured result in every comparison this product makes, silently.
      *D71 was a question* — "whether `region` should feed the market grouping at all, or is a
      different axis" — and it is a different axis. A campaign in Peru whose region is LATAM
      counted as TWO markets and satisfied §8.3's "seen in at least two markets" gate alone.
      Only where the rulebook declares the region: undeclared, `region` means a continent on
      one record and a country on the next, which is why the row was a question.
- [x] **12.3 Fabletics example config** — the full transcribed content (guardrails, tag
      taxonomy, influencer criteria + red flags, partner feedback, 360 checklist, scorecard)
      and the ten standing corrections with provenance, shipped as an **example/customer**
      file, not the product default.
      **Split, because the two halves need different things.** The FORMAT half is done:
      `docs/example-rulebook.yaml` is a worked example of every shape the file can carry,
      shipped beside the binary, named by `init` and by the product's own rulebook, and
      loaded by the tests so it cannot rot silently. It carries an invented agency and says
      so in its own header — the rules in it are illustrative and the product treats them as
      nobody's.
      **The CONTENT half is done, and the row was wrong about why it could not be.** It said
      the transcription "lives in the 11 Sep review document, not in this repository" — true of
      the repository, and beside the point. The document is the one that kickstarted this
      workstream and was sitting in the customer's Downloads the whole time; a reviewer said so
      and was right. Searching the repo and its full history was the wrong search, and stating
      the result as "only the customer has this" made a blocked row out of an available one.
      `docs/fabletics-rulebook.yaml` now carries it, transcribed section by section from "Ship
      the rulebook with the product": four non-negotiable guardrails as `blocking` rules, the
      four-way tag taxonomy, the influencer criteria and red flags, the recurring partner
      feedback as evaluation logic, the 360° checklist as the six things a brief must carry
      with the words that would show each had arrived, the six-criteria scorecard, and all ten
      standing corrections with their provenance and market scope. The presenter notes carried
      as PDF annotations are transcribed with the sections they annotate.
      **Shipped as a customer file, which is this row's own decision.** Putting it in the
      product default was tried and reverted: 126 tests encode the empty default, because it is
      what makes the first-run and empty-library behaviour what it is, and one customer's rules
      in a generic product's default would make every install theirs. It ships beside the
      binary, so putting it in force is one `cp` into the data directory and no download. The
      source document asks for the defaults to ship populated, which for a Fabletics build is
      right — making the INSTALLER do that copy is one line and a product decision rather than
      a technical one, and it has not been made here. *Closes D129.*
      *Closes D37 and D108.* D37 moved "in market" out of the product's own synonym table —
      one agency's phrasing does not belong in a product that ships generic — and review
      caught the sweep half done, with the model-facing gloss still teaching the word `enums`
      had dropped.
      **D108 was the sharp one.** A correction declared in a customer's rulebook arrives with
      provenance and no `campaign_id`, so the counting gate refused it forever: seen in 0
      campaigns, needs 3. That gate is for a rule the library INFERRED, where breadth is what
      makes the guess safe; a rule somebody wrote down is not a guess, and no number of
      campaigns makes their own rule truer. It skips the counting and not the person.
      **And then shipped D108's own failure as its fix:** `expected_in = []` already meant "no
      checklist", so nine of the ten declared corrections reached no judgment in any market
      while the tool reported them in force — stored and never applied, which is D108's own
      sentence. Review found it; a column carries "everywhere" now, and the test that should
      have caught it had asserted a dict that is always truthy.
- [x] **12.4 Schema gaps named in the review** — no field today for `asset_link`,
      `approval_notes`, or market-scoped feedback patterns. *Decide explicitly whether the
      tracked client comments 2.5 now ingests ARE the `approval_notes` gap — in most agency
      flows a returned deck's comments are exactly that — or whether a separate structured
      field is still owed.*
      **The decision: they ARE.** A returned deck's tracked comments are what the client wrote
      when they sent it back, and they arrive with an author, an anchor to the slide, a date
      and a reply thread — because they came out of the file rather than being retyped into a
      box. A separate `approval_notes` field would be a second place for the same thing and
      the worse copy of it. What was genuinely missing is not the notes but the VERDICT:
      `approval`, `approval_note` and `approval_by`, which is a PERSON because "the client
      approved it" with nobody's name against it is an opinion this library cannot attribute.
      *Thirteen tracker rows had accumulated here*, because every "this needs a field that
      does not exist" answer for seven phases was sent to this item. They were four different
      kinds of thing, and settling them meant deciding which kind each was rather than
      building thirteen features.
      **Fields, and the readers that make them fields rather than columns.** The first round
      of this item added all five and wired two, and both reviewers opened on the same
      sentence: a field added because a review named it and read by nothing is the defect this
      project has hit most. So — `asset_link` (where the work actually lives, which
      `asset_path` is not); `campaign_type`, which now KEYS the checklist (D102: a measure
      whose evidence was type-coherent is asked of that type in every market, and of no other
      type in its own — market was standing in for the word "launch" and got both halves
      wrong); `partner`, which `learning.gate` now counts through `store.breadth_of` (D103:
      §8.3's gate is NAMED for partners — "across at least two PARTNERS or markets" — and two
      agencies writing the same note in one market, which is exactly not one shop's house
      style, was refused); the statuses a campaign can actually be in (D38 — `cancelled` and
      `paused` were refused, and "we stopped this one" is an outcome), with the WIRE pinned to
      `store.VALID_STATUSES` by a test and every reader swept, because the schema published
      three while the core accepted five and `coverage` called a cancelled campaign
      `not_yet_run`; and retire/revive history (D106: one timestamp cannot say "twice"), read
      on `measure_status` and filled with the observation `retire_stale` computes rather than
      a default sentence true of every retirement.
      **Surfaces** — `attach_deck` (D39), which unblocked D51's gap: a record whose comments
      were never read could only gain them by being uploaded again as a duplicate, which §5.2
      refused in writing, so the gap went unreported for seven phases rather than recommend
      it. It reads a deck through the same paths an upload does — body, commentary, §9.3
      promises, the images inside it — because a record repaired the way the gap recommends
      must not come out thinner than one uploaded whole; a file nothing can be read from is
      refused rather than written, so the record stays repairable; and the gap counts only
      records with NO deck, which is the population the offer actually fits. The first round
      reported it on pasted-text briefs too, whose only remedy `attach_deck` rejects — §5.2's
      complaint arriving back through the door the fix for it opened. And `update_asset`
      (D123), because a model that guessed `proposed` on fourteen photographs had produced an
      unrepairable record; its `why` is stored on `authorship.note` rather than echoed back
      and dropped.
      **Meaning** — "correction" meant three unrelated things (D113): a diff's corrections
      taken, a recomputed metric value, and a standing correction. Renaming the stored
      vocabulary would break every saved row, so each surface says which it means.
      **Limits** — D125 is settled as an accepted limit rather than work owed. Refusing a
      promise about a COUNT is right, because a max-similarity search confirms a universal
      claim by construction: four colourways in frame make "a single colourway" match
      STRONGER. Counting what is in a photograph needs object detection, a different
      instrument from the retrieval this product ships — and the refusal now NAMES that, so a
      reader can tell a boundary from a defect. D109 moved to §13.4, which is where the
      missing timeout it is actually blocked on gets fixed — and §13.4's own line was updated
      to name it, because a re-point written on one side only aims a row at an item that does
      not know it is owed anything.

      **Already built** — the third thing the item's own line names, *"market-scoped feedback
      patterns"*, was not a missing field: §8.6 built it and nothing here said so, which is
      how work that is done still reads as owed. A standing correction carries `markets`
      (where it was seen) and `expected_in` (the markets it graduated into), and
      `corrections.in_force` applies it only in those — with `applies_everywhere` for a rule
      the customer declared in their own rulebook, which §12.3 added because a declared rule
      is not scoped by where the library happened to notice it. Verified at
      `corrections.py:610`, not assumed.


## Phase 13 — The engineering work filed into a bucket

Thirteen tracker rows say the same thing in their own words: *"Re-pointed from 10.1, which was
being used as a bucket for work filed while it was the next unstarted item."* They were then
re-pointed to 12.1 for the same reason, which would have made the rulebook item the second
bucket. None of them is about a rulebook.

They are named here as their own phase because that is what they are — real work, sequenced,
with an item to close them against — rather than because a phase number makes them optional.
Every one is either a cost that is fine at the sizes anyone has today and will not stay fine,
or a second copy of something that already exists once.

- [x] **13.1 One definition of "measured"** — `readiness`, `gaps`, `coverage` and
      `disconfirming` each decide separately what a measured library is, and two of them now
      disagree OUT LOUD: a library can be told all its campaigns have measured results and, in
      the next response, that nothing in it has a measured verdict. *Closes D75, D88.*

      **The fix is not one predicate, and that is the whole finding.** Reproduced first, over
      stdio, before anything changed — and read closely the two sentences are not one claim
      made twice. They are two different claims wearing one word: `with_results` is numbers on
      file, `with_verdicts` is somebody having said whether those numbers were good and stood
      behind it (a `performed_well` / `underperformed` tag, `source: verified`). A campaign can
      have every number anybody asked for and no verdict at all — that is the ordinary state of
      a library nobody has been back to — so collapsing them would make the product either
      claim a verdict it does not have or deny results it does. §6.6 had worked this out for
      the evidence ladder and written it down; the other three surfaces never got it.

      **That distinction is not the unification the rows asked for, and the first round of
      this item stopped there.** D75 asks for "one shared library-state helper"; D88 says four
      surfaces "decide separately what measured means". Neither is about how many PREDICATES
      exist — they are about four surfaces applying four different record lines and status
      tests to one word. Naming two facts explains why there are two names; it does not touch
      membership. Review caught it, and caught that giving `coverage` a status test while
      leaving `readiness` without one had produced a SECOND out-loud contradiction where the
      first had been: *"all 3 campaign(s) here have measured results"* beside *"none with
      results on file"*, both halves now about the same fact.

      So membership is decided once: `measured = with_results & ran`, and `readiness`, `gaps`
      and `coverage` all read it — they agree by construction rather than by three status
      tests somebody has to keep in step. `with_results` stays beside it as a fact about DATA,
      because `_evidence_strength` weighs the records a judgment CITED and a citation of a
      superseded version is still a citation of something with numbers on it. `has_results`
      and `has_a_verdict` are the per-record predicates for exactly that case.

      **The regression the adversarial pass caught, which both the design review and I
      missed.** `has_a_verdict` matched the tag value as an exact string, and §12.2 settled
      that a declared vocabulary is a synonym resolved WHEN TWO VALUES ARE COMPARED. An agency
      that declares `underperformed: ['flopped']` and tags a record `flopped` has recorded a
      verdict — `store.filter_campaign_ids` says so, because retrieval folds tags at read
      time — and `core` said it had not. This was not cosmetic: the disconfirming search HID a
      contradicting precedent that the code before §13.1 found, and reported "no verdict
      recorded against them" about a record whose verdict was written in the customer's own
      word. §13.1 made the product worse than it was until this was fixed. Both predicates
      fold through `store.fold_vocabulary` now, and `carries_verdict` exists because the
      narrower "does this record carry THIS verdict" was written out inline in the one place
      where getting it wrong cost a precedent.

      Three further findings from the same pass: a `performed_as_expected` tag is an ANSWER,
      so naming that record in `could_be_checked_if` as having "no verdict recorded" invited
      somebody to contradict what was already recorded — `with_neutral_verdicts` is the third
      bucket that fixes it. A verdict on a CANCELLED campaign was counted as precedent while
      the same helper said it never ran; `with_verdicts` applies the status rule too. And the
      remedy list was sorted by uuid and then truncated, so which three records it named was
      arbitrary and unstable — oldest first now, which is also the honest order. The disconfirming search
      now reads *"3 campaign(s) here have measured results and no verdict recorded against
      them"* instead of *"no campaign in the library is tagged X with measured results behind
      it"* — the same fact, said in a way that does not contradict the sentence before it, and
      naming the records that are one `update_campaign` away from lifting it — the full count,
      not the length of the capped list, which said "3 campaign(s)" about seven.

      It distinguishes the THREE states it can be in — verdicts exist but none of this kind,
      results with no verdict recorded, nothing measured at all — because the first round
      collapsed them into one sentence that was false in two of them: on a library where every
      campaign was tagged `performed_well`, a search for `underperformed` reported that
      nothing carried a verdict, and the string it replaced had been TRUE there. A fix that
      makes a sentence false where it used to be true is worse than the defect. The state is
      decided once as `why_not` and rendered twice — in `what_it_means` and in
      `_say_the_disconfirming_check`, which is the sentence the model says ALOUD and which the
      first round left on the old wording entirely.

      **What the inventory turned up beyond the two rows**, none of which the item names:
      `readiness` had no status test at all, so a `proposed` campaign with numbers counted as
      measured — named as a finding in the first round and not actually fixed until the
      second, which is a row closed with its own statement still true, this project's most
      repeated defect; `gaps` counted stubs and the other two did not; `_campaigns_that_could_be_tagged`
      offered SUPERSEDED records as the remedy, which tagging would never make reachable;
      `missing_input_for_citations` computed a measured set and never used it; `coverage` held
      two answers, reporting zero measured cells while refusing to say the library was
      unmeasured; and `store.list_campaigns` carried `has_metrics` (any row, a forecast
      included) with no `has_actual_metrics` beside it — published raw to clients, so a campaign
      holding nothing but what somebody HOPED for reported as having metrics. The listing also had no
      legacy-schema guard. §13.1 put it on the save path and it was added there — but review
      established that the claim first written here, "the fourth function to crash on an
      upgraded v0.2.0 database", overstates it: `upgrade()` creates every missing TABLE before
      anything else, so a live install cannot reach that state, and what actually breaks an
      upgraded database is missing COLUMNS (D89). The guard is defensive and now says so.
- [x] **13.2 Computed facts cached rather than recomputed** — `gaps()` recomputes every
      campaign's facts on every call (500 decks ≈ 40 s, dominated by the channel regexes) and
      does it before the empty-library early return. Facts are deterministic on body text, so
      they belong beside the record, written at ingest and on edit. *Closes D95.*

      **The cost is real and was measured first**, on 200 decks of ~38,000 characters:
      **2.41 s per `gaps()` call, of which `facts.compute` is 2.3 s.** Warm afterwards:
      **0.13 s and zero text scans**, and the same on `coverage`, `readiness` and
      `diff_campaigns`, all of which walked the same path.

      Two honest qualifications, both raised in review, because a performance item that
      reports only its best case is advertised rather than measured. **This is not the
      review's corpus.** D95's 500 decks ≈ 40 s implies 80 ms a deck; these measure 12 ms,
      and a reviewer's own channel-dense deck measured 26 ms. Whatever the review's 500 were,
      they were three times heavier than anything reproduced here — so the shape of the
      finding is confirmed and its magnitude is not, and the earlier draft of this note
      asserted the corpora matched while flagging a *different* D95 discrepancy one paragraph
      later. **And the cold pass is now slower than before the change**: the same full scan
      plus a write, measured at 2.86 s for 200 cold records against 2.41 s before. That is
      paid once per (body, rulebook, release) and never inside a loop, but it is paid, and
      two things cool the whole library at once — editing the rulebook, and importing metrics
      whose `detail` is part of the body. Ingest, re-index and `add_metrics` all warm, so the
      scan lands where somebody is already waiting for a file rather than inside a report.

      **The row's premise is wrong, and building it as written would have shipped this
      product's cardinal sin.** "Deterministic on body text" is false: `facts.compute` reads
      the customer's rulebook twice — `rulebook.vocabulary("channels")` decides the channel
      checklist and `rulebook.rules()` supplies the guardrail `watch_for` words. Verified on
      one unchanged string: with no overlay it computes `channels: partial` and `guardrails:
      nothing_to_check`; with a rulebook declared, `channels: present` and `guardrails:
      contradicted`. Keyed on the text alone, every record stored before the rulebook existed
      would go on reporting `guardrails: nothing_to_check` — the product saying there are no
      rules to check about a library whose rules were just written, with `nothing_to_check`
      doing the work of a pass. The key is (body, `rulebook.version()`), which §12.1 built as
      a composite stamp that moves whenever either half does.

      **The key is CONTENT, not a declared version — and the first attempt got that wrong in
      exactly the way it was correcting.** Keyed on `rulebook.version()`, a customer who
      edited a rule without bumping their version string kept the old answer forever:
      reproduced, with the library reporting `contradicted` against a rule the customer had
      just DELETED. Before the cache a restart healed that, because every read recomputed —
      the cache is what removed the healing, so it owes the guard. Worse, `rulebook.version()`'s
      `core-1.0` half is the bundled YAML's version and not a code version, and `facts.py`
      has changed six times while that string stood still, so a release fixing a matcher would
      have left every install answering with the old one. The key is now a digest of the body,
      a digest of the rulebook's CONTENTS, and a stamp for `facts.py` itself — its source
      hash, falling back to the product version in a frozen build. Content-addressed
      throughout, so nothing rests on anybody remembering to bump a number.

      **Read-through, and the read must never cost the reader anything.** The key is
      re-checked on every read and a miss recomputes, so a body changed by a path that never
      learned about the cache still reads correctly. But a read that WRITES brought three
      hazards, none of which had a test until review found them: it committed a transaction
      its caller meant to roll back, it raised on a read-only database, and it sat out the
      full five-second busy timeout when another connection held a write — per record, and a
      rulebook edit makes every record cold at once. Filling a cache is never worth any of
      that: the write now never raises, never commits somebody else's transaction, and gives
      up on a lock after 50 ms.

      **`health_check` reports how warm the cache is and under which rulebook.** Under the
      staleness the first attempt could produce, that stamp was the only thing that would let
      anybody notice — and it was on disk, reachable from no surface at all.

      One claim in the row I could not reproduce: `_fields_never_recorded` runs AFTER
      `_ranked_gaps` and over an empty list on an empty library, so "before the empty-library
      early return" describes an arrangement that is no longer there.
- [x] **13.3 The embedding and scan costs** — the same string embedded twice per save and
      three times per prepare; `store.citations` scanned once per coverage cell; quote
      verification re-reading every chunk per finding; a content edit re-embedding a whole
      deck for a title change; `retire_stale` scanning the registry on every metric write.
      *Closes D79, D90, D91, D97, D107.*

      **Measured before any of them was touched, and counted as CALLS rather than wall time**
      — which is the part that does not depend on the provider: against the bundled hash
      embedder a repeated embed is microseconds, and against Ollama it is a network
      round-trip for a vector already in hand. Two of the five rows turned out to be wrong
      about their own subject, in opposite directions, which is the argument both for the
      rows existing and for measuring one before fixing it.

      | | claimed | measured | now |
      |---|---|---|---|
      | D79 | one full read per finding | 12 findings, 12 reads | **1** |
      | D90 | save 2×, prepare 3× | save already 1×; **prepare 4×** | **1** |
      | D91 | one scan per cell | confirmed | **1** per report |
      | D97 | re-embeds the summary chunk | **all 17 chunks**, on a TITLE edit | **1**† |
      | D107 | a registry scan per metric write | 5 scans | **2**‡ |

      † **One when the pack boundaries hold**, which a title edit guarantees and a detail
      edit does not: `chunking.pack` merges adjacent units, so an edit that changes the
      summary's LENGTH moves every boundary after it and the whole deck genuinely differs at
      every position. Review measured a 200-short-section deck re-embedding all of it on a
      detail edit. The comparison is doing the right thing; the deck really did change.

      ‡ **Scans, not queries.** The five full registry reads become two full reads plus two
      single-row lookups and one filtered read — the win is WIDTH, not round-trips. And it is
      not the term that grows: `retire_stale` still calls `store.campaigns_that_skipped` once
      per expected measure, so the real scaling cost of a metric write is the size of the
      CHECKLIST, which this did not touch. Left as it is because the checklist is small by
      construction — §8.3's gate is what keeps it so — and named here rather than left for
      somebody to discover.

      The unit is not the same for all five, and saying "calls" for all of them flatters three.
      D90 and D97 count NETWORK round-trips to an embedder; D79, D91 and D107 count local
      SQLite reads, which are sub-millisecond each. The first two are the real wins; the other
      three are shape — work that grows with the library on every read — and D79's defence is
      specifically that `text_on_file` reads every CHUNK, so twelve findings against a
      200-chunk deck is some thousands of row reads rather than twelve.

      **The mechanism is the same in three of the five: compute once, pass it down.** D91
      fetches the citation list once per report; D79 reads each cited record once per save;
      D107 gets a single-row `store.metric_entry` for what was a full table read per name,
      and `retire_stale` asks the database for the `expected` rows instead of filtering them
      in Python. D97 compares the new chunk texts to the stored ones POSITION BY POSITION and
      rewrites only what differs — position rather than set, because §13.4/D100 is about
      chunk order being load-bearing, and a set match would silently reorder a deck whose
      paragraphs repeat. A rewritten chunk keeps its id, which is the half with a correctness
      cost rather than a speed one: `finish_indexing` names chunk ids, and §6.1 verifies
      quotes against the row's own columns precisely so that chunk ids need not be anchors.

      **D90's memo is scoped to a call, and the first version of it was not.** A
      process-lifetime LRU keyed on (model, text) is sound about vectors and wrong about
      everything else: a vector computed in one call was served in another, so an embedder
      that went down in between was never noticed. Eight tests of partial-failure and outage
      behaviour failed, each of them describing a real thing this product is supposed to
      report. The waste D90 names is INSIDE one call, so a call is the right scope; outside
      one, `embed_once` is `embed`. Keyed on the model even within a scope, because §7.2
      established that a model change is a visible migration rather than a silent re-ranking.
      Ingest is scoped too: a deck repeats itself, and 7 of 17 embeds on a deck with repeated
      sections were for a vector already in hand.

      **Two defects of my own, found in review.** The first version wrote each chunk's text
      and embedded it in the SAME loop and broke out on an embedder failure — so the positions
      after the break kept the old deck's words while the row held the new ones, with their
      `embedded` flag still set from before, and `finish_indexing` reported the record
      complete. That is D80's defect reintroduced by the fix for D97, hidden behind a comment
      claiming the row was correct and only the index behind. Text first, vectors second: a
      partial failure now lands where §2.1 says it should. And both memos were module globals
      while the MCP server runs sync tools on worker threads, so two overlapping calls shared
      one — and interleaved, they left a dict behind after both scopes exited, turning a
      call-scoped memo into exactly the process-lifetime cache it was written to avoid. Both
      are `contextvars.ContextVar` now, and both are pinned.

      **Two of the same shape that §13.3 had walked past.** `_expected_inputs_never_supplied`
      read every record's whole text once PER EXPECTATION and re-folded each body on every
      pass — D79's shape at a worse exponent, in the same file as the memo that fixes it. And
      `coverage` hoisted its citation scan ABOVE the empty-library early return, making the
      one path that previously did no work more expensive. `gaps()` now opens one read scope
      for the whole report, which `facts.for_campaign` shares, so the two parts that each walk
      every record read each one once between them rather than once apiece.

      **Acknowledged debt.** `each_string_embedded_once` and `_each_record_read_once` are the
      same mechanism written twice — a `ContextVar`, a context manager that installs a dict
      only if none is installed, and a pass-through wrapper. Both had the same non-locality
      bug and needed the same fix, which is that duplication made concrete. It is D114's
      subject and §13.5's item, and it is named there rather than being made a third mechanism
      here.

      One test elsewhere had to be re-fixtured rather than fixed: `test_a_partial_reindex_
      raises_a_notice` produced its partial state by letting an embedder die part-way through
      a full rebuild, which no longer happens. Its intent is unchanged and the partial state
      is now a smaller one — the changed chunk failing while the deck's sections stay
      embedded from before.
- [x] **13.4 The re-index's three loose ends** — `embedding.embed` called with no timeout on
      the re-index path alone, so a hung embedder hangs the handler per chunk; chunk ordering
      after a re-index, which silently broke "chunk 0 is the summary"; and D109, re-pointed
      here from §12.4 because the thing it is blocked on is D99's missing timeout and fixing
      it anywhere else would be fixing the symptom. *Closes D99, D100, D109.*

      **§13.3 made body `chunk_index` order load-bearing, and this item must not break it.**
      `_rebuild_body_index` matches `store.body_chunks(... ORDER BY chunk_index)` positionally
      against `chunking.pack`'s output to decide what to re-embed. Any renumbering here that
      does not preserve the body's RELATIVE order makes every position mismatch — and the
      failure is silent, because a total mismatch just re-embeds the whole deck, which is the
      old behaviour. The only guard is a test asserting a title edit costs one embed.

      §13.3 also narrowed D100 without closing it: the common edit no longer re-inserts, so it
      no longer trips `insert_chunks`' `MAX(chunk_index) + 1` allocation across ALL kinds. A
      body that GAINS a position still does, so the allocation is still the thing to fix — and
      it now has two callers to keep consistent.

      **D99 — the one embed path without a deadline.** Ingest passes `timeout=remaining` and
      `finish_indexing` passes `timeout=remaining_time`; the re-index passed nothing, so a
      hung embedder hung the handler once per chunk — on the path that runs over every chunk
      of a deck, which is the worst place for it. It now embeds against the remaining budget
      rather than a fixed per-call timeout, which is what bounds when the handler ENDS: a
      fixed one bounds when each call starts, and the last can begin just inside the budget
      and run the full timeout on top. The commentary half of `_reindex` gets its own budget
      rather than the body's leftovers, because sharing one would make the commentary's share
      depend on how long the deck happened to be, which is nobody's decision.

      **D100 — measured before the change, and worse than the row says.** A three-section deck
      with one comment, grown to six sections, came back as `body 0-6, commentary 7, body
      8-13`: one layer's sequence interrupted by another's, not merely commentary before body.
      And a record that gained its commentary BEFORE its body — which is what attaching a deck
      to something somebody had already annotated looks like — put a client's remark at index
      0, which is the invariant the row says was silently lost. `insert_chunks` now allocates
      per KIND. Safe because every reader orders within a kind and nothing treats the index as
      unique across them, so an existing database whose numbering overlaps between layers
      reads exactly as before — and §13.3's positional rebuild keeps working by construction
      rather than by accident.

      **D109 — attempted, measured, and left OPEN. The row was right and I read it wrong.**
      It defers a semantic match because `note` is a WRITE path, and I treated "blocked on
      D99" as permission: fixing the timeout removes *hung*, and the objection was *a model
      call on a write*. Both reviewers said so independently, and the codebase had already
      settled it — `feedback.py` moved this very pairing OFF the per-write path into
      `waiting()`, with measurements, for exactly this reason. What I built embedded every
      correction on file per write: 205 calls on a library of 204, each with its own timeout,
      which is the "bounds when each call starts, not when the handler ends" pattern this same
      item fixes in `core`.

      **And it does not work.** Measured against `nomic-embed-text`, the embedder this product
      ships, over five pairs that ARE one rule and six that are not:

      | | scores |
      |---|---|
      | one rule | 0.651 · 0.771 · 0.872 · 0.880 · 0.896 |
      | different rules | 0.378 · 0.391 · 0.418 · 0.446 · 0.495 · 0.695 |

      The populations OVERLAP, so no threshold separates them. The row's own motivating pair —
      "photography before training" / "shoot before the workout" — scores 0.771, under the
      0.86 I had picked; and a pair differing ONLY by negation, "Always show the logo" /
      "Never show the logo", scores 0.933, over it. A threshold catching the first fires on
      the second, which is the prohibition merged with its permission that `negated()` exists
      to prevent. Cosine on this model measures topical similarity, and two rules about one
      subject are topically similar whether or not they say the same thing.

      Two guards were also half-carried, which the measurement makes moot but which say
      something about how the attempt was made: the negation half of `_reading` was dropped
      entirely, and `_MIN_CONTENT` was applied to the incoming text and not to the candidates,
      so a fragment already on file was offered as the rule it came from.

      So the fallback is reverted and **D109 stays open, with a measurement it did not have** —
      one that says what the next attempt must not be: not a cosine threshold on this embedder,
      and not on the write path. What is kept is the label: a resemblance now says
      `basis: heuristic` and `how: wording`, because the only matcher this product ships reads
      words and should say so.

      **One budget per tool call.** `attach_deck` rebuilds the body, embeds the commentary and
      indexes the images; each had its own deadline, so one call could spend three times the
      number `TOOL_TIME_BUDGET_SECONDS` names. `ingest_campaign` already threads a single
      deadline through exactly those three phases — the question was settled and this had
      quietly answered it differently. Both reviewers measured it.
- [x] **13.5 The remaining two-copy helpers** — `_newly_eligible`/`graduate`, the offered-once
      flags, `touch_metric`/`touch_correction` (byte-identical market-fold loops) and
      `campaigns_that_skipped`/`campaigns_that_skipped_correction`. "One mechanism" is true of
      the decisions and not of the code under them. *Closes D114.*

      **The five pairs are not the same kind of thing, and the row says so itself** —
      *"merging them needs a row abstraction over both that is a larger change than either
      caller"*. So the work was as much judging which is which as deleting code.

      **Byte-identical, and therefore where the next drift comes from.** The market-fold loop
      in `touch_metric` and `touch_correction` is the same characters in both, down to
      `seen.add(fold_market(where))` — the fifth implementation of "which markets does this
      campaign count towards", which is the question D55 and D88 both record drifting. The
      market-scoping SQL in `campaigns_that_skipped` and its correction twin is the same
      fragment and the same parameter packing, differing only in a table alias — so §12.2's
      change to how a market is matched had to land twice or the two answers diverged in
      silence. Both are now `store._widen_markets` and `store._market_scope`.

      And a THIRD copy of the fold loop that the row does not name, in the merge that
      re-derives a correction's breadth from the sightings that just moved. It was found by
      mutating the shared helper and watching a test that should have felt it stay green — it
      had no test of its own, so nothing was watching it. That is the argument for
      the row, made by the row's own work.

      **The memo mechanism §13.3 wrote twice** is `scoping.scoped_memo` now. Both copies were
      module globals under a threaded server, both were therefore shared between overlapping
      tool calls, and both needed the same `contextvars` fix — D114's argument in miniature,
      inside one item. §13.3 named it as debt owed here rather than writing a third.

      **Left alone, deliberately:** `_newly_eligible`, `graduate` and the offered-once flags.
      What DECIDES in those is already shared — `learning.gate`, `distinct_briefs`,
      `subject_markets`, `require_a_person`, the thresholds — and that is the half that
      drifted, which §12.4 found applying half of its own rule. What remains is a table name,
      a key column and an offer whose fields are different words for a reader. Merging them
      needs a per-kind descriptor larger than either caller, putting a layer between two
      callers and the SQL they each run once: a cost with no drift to prevent. A test pins
      that BOTH callers reach the shared gate — an earlier version checked only that
      `graduation` calls it, and review proved a mutant deciding eligibility locally passed —
      so "left alone" stays a judgment rather than an oversight.

      **And the judgment was wrong in one place, which is the argument for checking one.**
      `corrections.graduate` records authorship and `metrics.graduate` did not, so a graduated
      MEASURE kept only a free-text `confirmed_by` while a graduated RULE kept the account the
      call was made from. §11.1's own words cover both — "the account beside the name, on the
      write that puts a rule in front of every future brief in its markets" — and a graduated
      measure is put in front of every future brief in exactly that way. That is not a table
      name or an offer's wording; it is a decision about whether a write is audited, made one
      way on one path and the other way on the other. Both reviewers found it independently.
      Fixed, and pinned.

      **What this did NOT exhaust**, so C155 is not read as closing the class: the same folded
      membership test stood in four more places — twice in `replay`, and as the market half of
      `metrics.expected_for` and `corrections.standing_for`, which are the two functions that
      decide what a brief is CHECKED AGAINST on the two paths D114 is about. All four now use
      `learning.reaches`. And the first version of `_widen_markets` open-coded the dedupe that
      IS `learning.fold_markets` — an extraction adding a copy of the thing it was extracting,
      which `store.markets_of` had a fourth time. The guard that catches the next one scans
      every module for the dedupe EXPRESSION rather than one file for one spelling.
- [x] **13.7 Is this the same rule?** — **no semantic matcher was built.** The row's own
      licensed outcome, "nothing at all", is the outcome — reached with a measurement strong
      enough to say why rather than only that, and with the part of the ask that turned out to
      be a lexical defect fixed.

      Measured on three independently-written populations against the shipped
      `nomic-embed-text`: pairs that ARE one rule score 0.646–0.972, pairs that are NOT one
      rule on the same subject score 0.763–0.963, and the two are interleaved in the wrong
      order. The row's own example ("photography before training" / "shoot before the workout")
      scores 0.793, and any floor admitting it admits seven of eight false pairs — among them
      "…logo in the top left corner…" against "…in the bottom right corner…" at 0.963 and
      "Talent is confirmed before the shoot" against "…after the shoot" at 0.960. Each is a
      rule against its own opposite and neither is catchable by `negated()`, because neither
      side is phrased as a prohibition. Meanwhile every terse-vs-long statement of one rule
      scores BELOW the row's own example: hold the long side fixed, vary only the short one,
      and the score climbs with its LENGTH (0.751, 0.860, 0.864). The instrument measures
      structural and topical similarity, and rule identity is neither — so it is wrong on both
      axes at once, which no threshold and no direction repairs.

      Two attempts are recorded against this, both reverted. §13.4 tried a semantic matcher.
      §13.7 tried the weaker inverse — the embedder used only to REFUSE a lexical match, the
      way `negated()` refuses — and shipped it before the numbers were in: it refused true
      pairs at 0.646–0.751 while passing false ones up to 0.963, reintroduced the O(n²) table
      read `feedback._rule_questions` documents removing (400 corrections, 44s), compared
      vectors across models without checking provenance, and depended on vectors that no path
      the product suggests would ever create. Reverted whole.

      What DID ship, because the measurement turned it up: `negated()` listed `not`/`no`/`never`
      and not `nothing`/`none`/`nobody`, so "Nothing is scheduled during Semana Santa." read as
      a PERMISSION and scored 0.0 against "Do not schedule anything during Semana Santa." — one
      rule stated two ways, discarded by the guard written to protect it, with no embedder
      involved anywhere. It now pairs lexically at 0.60, with prohibition-against-permission
      still refused at 0.00. And `note`'s dead `how: "meaning"` branch is gone: it advertised
      "resembles one already on file in what it MEANS rather than in its words", a claim
      `_looks_like` has never been able to make.

      To revisit needs a DIFFERENT instrument, not a different threshold. Asking a model the
      question is not ruled out by any of this — but the server cannot ask one, the judging
      model being on the other side of the protocol, and cosine cannot even generate the
      candidates, because its top-k fails toward the antonym. *Closes D109.*
- [x] **13.6 Two gaps that cannot see the record they are about** — `execution_never_checked`
      fired only where briefed creative already existed, so a concluded campaign uploaded as
      text with no assets was invisible to it; and commitment vectors were cached without
      `record_vector_model`, so a CLIP weights change compared across models and
      `count_unreadable_vectors` did not cover the space. Both rows named the population and
      neither named what widening it would cost.

      **D122.** The gap's own `why_it_matters` — "every outcome on those campaigns is being read
      as though the brief caused it" — never mentioned briefed creative, and is strictly MORE
      true of a record with nothing on file: the old condition made the gap false of the
      population it excluded. Widening it made the gap reach the shop that works from
      descriptions, for whom it is permanent, so it had to become set-asideable — a question
      that had never come up, because while it required briefed creative every record it
      reached had somebody who photographs. It also needed `since`, without which it was the
      one set-asideable code with no way back. And the first widening produced an offer that
      DID NOT WORK: asked for the delivered photographs, a record with nothing briefed got
      `compare_execution`'s refusal from the other side, and the upload closed the gap while
      every citation of that campaign went on saying nobody had checked what it ran — D55's
      two-surfaces shape, manufactured by the product's own suggestion. The condition is now
      the stored status the citations read, so the gap and the caveat are one claim by
      construction, and the offer asks for whichever half is missing, briefed first. What keeps
      the set-aside honest is `_execution_note`, attached per cited record; it is NOT
      `set_aside`, which nothing that builds a judgment reads.

      **D126.** Routed through `core._add_vector` — the one function that writes a vector and
      its model together — and `vectorstore.SPACES` puts the commitment space inside the check
      written to notice a vector table this process cannot read. Which model fills which space
      now has one answer, beside `_default_dim`, the fact that has to agree with it. The row's
      framing of the risk was wrong in the way that mattered: "stale" cannot mean "made by
      weights this BUILD no longer runs". Measured against that reference the check made the
      weights-swap case WORSE, re-encoding the phrase with the new model and leaving the images
      on the old one. The reference is the asset vectors' recorded model; a phrase that
      disagrees is re-encoded, and images that disagree are not rebuildable from here, so
      `check` refuses out loud rather than reporting a cosine between two spaces. *Closes D122,
      D126.*

---

## Regression baseline — must not break

The review explicitly verified these as working. Any change that breaks one is a regression:

- Perceptual fingerprinting (resized/recompressed/renamed slide matched at Hamming 0, own
  assets correctly excluded, cross-region flag raised).
- Aesthetic similarity once weights are present (sensible ordering, 0.66 vs 0.35).
- Graceful degradation on a failed embedding (asset stored, precise warning, no crash).
- Passage chunking (5/5 chunks embedded; resolved the old long-deck 500s).
- Bundled dependency set correctly collected into `_internal`.
