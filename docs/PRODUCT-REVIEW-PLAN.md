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
   against, or an accepted limit — in the same commit that creates it.
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
      ≤120, detail, precedent{id,quote}, fix ≤120), resolved[]. Caps and enums enforced
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
      **Not done here, named so it is not assumed:** the quote is bounded but not *checked
      against what was retrieved* (6.1, which needs 7.2's server-owned retrieval first —
      `prepare_evaluation` is stateless, so the server cannot today tell a real quote from a
      plausible one); `closest_precedent`, `evidence` and `provenance` are still whatever the
      caller passes (7.2/7.6); `findings` has no per-evaluation golden set yet (7.7).
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
- [ ] **4.4 macOS: sign, notarize, staple - and ship a `.pkg`** (raised by the Phase 4
      design review; not in the original review, and owed). Today the macOS archive is an
      unsigned binary whose quarantine flag the installer strips. That strip is a legitimate
      stopgap *inside an installer the user chose to run*, on the product's own directory —
      but it ships "we are not signed" as the plan of record, and on macOS 15+ an unsigned
      quarantined binary launched by Claude Desktop is blocked outright rather than offering
      "open anyway". The marketer never gets that far in any case: the tarball needs Terminal
      to extract, `install.sh` is itself quarantined, and Finder opens a `.sh` in a text
      editor. What is owed: Developer ID signing + notarization + stapling in CI, and a
      `.pkg` whose postinstall runs the same gate — after which the quarantine strip can go.

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
      **It reaches somebody.** `list_campaigns` — the surface the review named, "returns
      eight rows" with nothing to say whether eight is enough — carries the guidance while
      the library is not yet working, and stops once it is, because guidance that never goes
      away is the thing nobody reads.
      **Closes three tracker rows.** D43: every step here is an offer whose arguments only
      the user has, so `needs` is what keeps "accepting is one step" honest. D66: `coverage`'s
      `thin` list collapses to one summary line plus the path when nothing in the library is
      measured at all — listing 25 weak cells in a wholly-weak library is the matrix again,
      and the answer at that size is not "fix LATAM". D68: `unmeasured_campaigns` groups the
      fix per campaign, so one campaign in three thin cells is offered once.
      Found while wiring D66: "nothing is measured" has to be a fact about the LIBRARY, not
      about the cell markers — five cells of one measured campaign each read `single_example`
      rather than `measured`, and that is a library with real evidence in it.

## Phase 6 — Making the reasoning defensible (ideas G–L)

*Depends on 2.4.*

- [ ] **6.1 (G) `precedent.quote` required** on every finding, drawn from the retrieved chunk.
      *Re-sequenced after review: 2.4 bounds the quote and requires a `campaign_id`/`rule_id`,
      but "drawn from the retrieved chunk" cannot be enforced while `prepare_evaluation` is
      stateless — the server does not retain what it returned. Needs 7.2 first, or a receipt
      id from `prepare_evaluation` that `save_evaluation` requires. 7.3 is otherwise
      subsumed by 2.4; what remains of it is this. A verified quote must also match the
      chunk's LAYER (§2.5): a commentary quote is valid evidence, but only when the
      precedent is marked `layer: "commentary"` — verifying the text alone would confirm a
      reviewer's objection as something the deck itself said, and bless the misattribution
      with a green tick.*
- [ ] **6.2 (H) Guardrail breach vs departure from precedent** — two classes, different
      vocabulary, only one is debatable. *2.4 defined `kind` and the rule that a guardrail
      breach cannot be a note; what remains is making `kind` required, requiring a `rule_id`
      citation on a breach, and giving the two classes genuinely different wording.*
- [ ] **6.3 (I) Close the prediction loop** — when a superseding record arrives, surface the
      prior evaluation's predictions and ask which held.
- [ ] **6.4 (J) Disconfirming search required** before a verdict is saved; record what came
      back, including "nothing".
- [ ] **6.5 (K) `approve_if`** — the testable exit condition that converts revise → approve.
      *2.4 defined and bounded the field; what remains is requiring it on a revise, forbidding
      it on an approve, and requiring a `fix` on every blocking/should_fix finding.*
- [ ] **6.6 (L) Evidence-strength line** on every judgment — how many precedents, concluded,
      verified; top similarity; whether one match dominates.

## Phase 7 — Consistency across users (ideas 1–8)

*Reviewer's own order: 1 and 2 first, then 3 and 4, then 6 and 7 before any prompt tuning.*

- [ ] **7.1 Compute what can be computed.** Must run on the BODY layer only (§2.5):
      a reviewer's comment saying "make sure we never mention adidas" would otherwise
      register as the brief mentioning adidas. Date coverage, ER presence per profile, budget
      detected, channel checklist, internal date consistency, guardrail keyword hits —
      returned as **computed facts** with evidence. Two thirds of real findings were
      mechanical; this removes most cross-user variance.
- [ ] **7.2 Server owns the retrieval query.** Derive from the subject record/file, not
      model-authored `proposal_text`; derive filters from the subject's attributes; pin
      `top_k`; stable deterministic tie-breaking; record embedding-model version per vector.
- [x] **7.3 Enforce the output shape server-side** — reject writes missing a precedent quote
      or exceeding caps, rather than accepting and hoping. *Done by 2.4, except the "missing
      a precedent quote" half, which is 6.1 and depends on 7.2.*
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
      retrieval, authority order configured in the rulebook, never inferred. *Now also
      covers commentary: 2.5 stores a `kind` (speaker_note / comment / annotation) and makes
      it filterable, but deliberately does NOT weigh a client's comment above an author's
      own note — the kind records the format the words arrived in, not their authority (a
      PDF export turns speaker notes into annotations), and authority order is configured
      here, never inferred there.*
- [ ] **11.6 Backfill existing records as `author: unknown`** with import date, explicitly.
      *Applies to `campaign_chunks.source.author`: a speaker note carries no author field at
      all and a PDF annotation often has no `/T`, so a null there means "the file did not
      say", which is a different statement from "nobody said it". Documented in the
      docstrings by 2.5; the explicit convention is this item's.*
- [ ] **11.7 Treat it as personal data** — install disclosure, per-person view, deletion or
      anonymisation preserving the judgment, stated retention position. ***Live obligation
      as of 2.5:*** *`campaign_chunks.source.author` is the first field in the library
      holding a person's name harvested from a file rather than typed by the operator — PDF
      `/T` and PowerPoint comment authors — and it is returned as `matched_author` on search
      hits and through `get_campaign`. Today it is deletable only by cascade when the
      campaign is deleted. Per-person view and erasure are owed here, and the names are in
      scope for the install disclosure.*

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
      `approval_notes`, or market-scoped feedback patterns. *Decide explicitly whether the
      tracked client comments 2.5 now ingests ARE the `approval_notes` gap — in most agency
      flows a returned deck's comments are exactly that — or whether a separate structured
      field is still owed.*

---

## Regression baseline — must not break

The review explicitly verified these as working. Any change that breaks one is a regression:

- Perceptual fingerprinting (resized/recompressed/renamed slide matched at Hamming 0, own
  assets correctly excluded, cross-region flag raised).
- Aesthetic similarity once weights are present (sensible ordering, 0.66 vs 0.35).
- Graceful degradation on a failed embedding (asset stored, precise warning, no crash).
- Passage chunking (5/5 chunks embedded; resolved the old long-deck 500s).
- Bundled dependency set correctly collected into `_internal`.
