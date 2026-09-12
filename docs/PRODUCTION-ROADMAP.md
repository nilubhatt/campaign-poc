# Production Roadmap — Campaign Intelligence (full product)

Status of this document: **planning / deferred.** The immediate priority is getting the
product working **end-to-end on a local Windows laptop** (Claude Desktop + local server +
Ollama). Everything below — central deployment, OAuth, TLS, multi-user — is captured here so
it isn't lost, to be picked up **after** local end-to-end is solid.

> This started as a lightweight PoC. The decision (Sept 2026) is to **extend it into the full
> product** — no more "lean vs heavy" split. This repo (`campaign-poc`) is the product going
> forward; the data layer moves from SQLite to Postgres for the central deployment (local runs
> keep working unchanged).

---

## 1. Vision

A **central marketing-campaign intelligence server** on a company VM. Every user calls it from
**their own Claude** (Web or Desktop) as a custom connector. It accumulates institutional
memory of past campaigns + real outcomes, judges new proposals against that evidence (Claude
does the reasoning), and flags reused/derivative creative across regions.

**Cost model (confirm):** reasoning runs on **each user's own Claude subscription** — no central
Anthropic API key or per-call cost. The VM runs only the free local pieces (Postgres, Ollama,
CLIP). Central cost ≈ the VM itself.

---

## 2. Decisions locked

- **Full product**, not a PoC. One codebase (this repo).
- **Reload all campaigns in v0.2** — schema may change freely; no migration of the existing ~10
  records required (chunking re-embeds anyway).
- **Skip external stock-photo lookup** (reverse-image search against Getty/Shutterstock/the web).
  We detect **reuse within our own library**; "is this an unknown web stock photo" is out of scope.
- **Auth = Azure AD / Entra ID** for this org — but the **auth provider must be pluggable** so the
  same product can be deployed for a different org with a different IdP (see §5).
- **Data reload OK**, so v0.2 is free to change the schema.

## 3. Open questions (status)

| # | Question | Status / recommendation |
|---|---|---|
| 1 | Which codebase is the product? | **Decided:** evolve this repo; move data layer to Postgres + pgvector. |
| 2 | Can Claude **Web** reach the VM? (public HTTPS) | **Open — blocking for central deploy.** Is it an **Azure VM with public DNS + inbound 443**? If yes → Azure Front Door / App Gateway for DNS + managed TLS. If internal-only → named tunnel or VPN-only Desktop access. |
| 3 | Identity provider | **Decided: Azure AD / Entra ID** (pluggable — see §5). |
| 4 | Access model — global vs scoped | **Open.** Rec: start **global-read within the company**; add region/team scoping via Entra **group claims** later. |
| 5 | VM specs & admin | **Open.** GPU or CPU-only? (CPU fine for text + CLIP at this scale, slower ingest.) Who has root? |
| 6 | Cost model | Rec: users' own Claude accounts do reasoning (no central key). **Confirm.** |
| 7 | Metrics ownership | **Open — highest data-value item.** Who loads KPI results (the workbook) centrally, how often? A bulk import can make it one step. |

---

## 4. Target architecture

```
 Users' Claude (Web / Desktop)  ──HTTPS + OAuth──►  Reverse proxy / Azure Front Door (TLS)
                                                        │
                                                        ▼
                                       MCP server (Streamable HTTP /mcp) + REST /upload
                                                        │
     ┌──────────────────────────────┬──────────────────┴───────────────┬─────────────────┐
     ▼                              ▼                                    ▼                 ▼
 Postgres + pgvector          Ollama (text embeddings)          CLIP (image vectors)   Asset store
  campaigns · chunks ·         nomic-embed-text                  visual space +         (originals,
  assets · metrics ·                                             perceptual hashes      thumbnails)
  evaluations · reconciliations
```

- **Database:** PostgreSQL + **pgvector** (concurrent, durable, real ANN) — replaces SQLite/sqlite-vec.
- **Text vectors:** chunked (per slide/section), one vector per chunk (see §6.1).
- **Image vectors:** CLIP visual embeddings **+ perceptual hashes** (see §6.6).
- **Transport:** MCP over Streamable HTTP (already built) + REST `/upload`.
- **Auth:** pluggable provider, Entra for this org (§5).
- **Deploy:** Docker Compose on the VM; existing GitHub Actions CI extends to build/publish.

---

## 5. Authentication (pluggable; Entra first)

**Design (implemented as a seam now, no-op default):** `auth.authenticate(header)` is the single
hook; the active provider is chosen by `CAMPAIGN_POC_AUTH_PROVIDER`. Adding an IdP = implement
`AuthProvider` + `auth.register("<name>", provider)` + set the env. No call sites change. The
`Principal` carries `subject`, `email`, and **`groups`** (IdP group claims) so access scoping can
key off them later. See `auth.py`.

**Entra specifics (to build when we do OAuth):**
- **App registration** in Entra for the server; expose an API scope; validate incoming JWTs
  (verify signature via the tenant JWKS, check `iss` / `aud` / `exp`, extract `oid`/`preferred_username`/`groups`).
- Claude's connector runs OAuth authorization-code + **PKCE** → Entra SSO (with MFA / conditional
  access) → Bearer token → `/mcp`.
- ⚠️ **Real risk to verify early:** enterprise Entra usually **disables dynamic client
  registration (DCR)**. Confirm Claude's remote-MCP connector supports a **pre-registered
  `client_id`** and register **Claude's redirect URI** in the Entra app. If the connector only
  does DCR, front it with a thin OAuth broker that Claude registers against and that federates to
  Entra. **Test this with a throwaway connector before building the full flow.**
- **Group claims → access model (§ open #4):** Entra groups map to region/team scope.

**Pluggability payoff:** a different customer just implements e.g. `OktaProvider` / `GoogleProvider`
/ generic `OIDCProvider` and sets the env — the product ships unchanged.

---

## 6. v0.2 feature backlog

Driven by real-use feedback (the memory was "a well-organised brief archive, not a performance
memory") plus the SVP's creative-reuse question.

### 6.1 Chunk `deck_text` server-side  *(highest leverage — one change fixes three problems)* — **Done (2026-09-08)**
Split each deck by slide/section, embed **each chunk**, store multiple vectors per campaign,
search chunks and roll up to campaigns. Fixes: (a) the **silent embedding failure** (long deck →
Ollama 500 → record saved but `embedded:false` hidden in warnings → unsearchable orphan; former
ceiling ~6.8k–9k chars), (b) the size limit, (c) delivers the **per-slide granularity** asked for.

Implemented on `v0.2-feature-backlog`: `extract.py` returns one unit per PDF page/PPTX slide
(natural chunks); `chunking.py` packs any text (natural units, or a flat `deck_text` string from
the LLM-first path with no boundaries) into pieces ≤ `MAX_CHUNK_CHARS` (default 1800, env
`CAMPAIGN_POC_MAX_CHUNK_CHARS`); new `campaign_chunks` table, one row per chunk; `vectorstore.py`
now keys vectors by chunk id, not campaign id; `core.ingest_campaign` embeds each chunk
independently so one oversized/rejected chunk no longer silently drops the whole campaign — it's
reported per-chunk in `warnings`, with `chunks_total`/`chunks_embedded` on the response;
`core.find_similar` searches chunks (over-fetching 4x `top_k`) and rolls up to the
best-matching chunk per campaign, returning `matched_excerpt` (the actual matched slide/section,
not a naive text prefix). 14 pytest tests added (`tests/`, offline `hash` provider + a real
generated `.pptx` for the extraction path) and verified end-to-end against live Ollama too. CI
gained a `test.yml` workflow (pytest on push/PR — there was no test workflow before).

**Note:** this changed the SQLite schema (new table + a renamed vector-store column) — an
existing local `campaigns.db` from before this change needs to be deleted/reloaded (consistent
with the locked "reload OK" decision above), not migrated.

### 6.2 Filter **before** similarity  *(fixes "can't discriminate")* — **Done (2026-09-08)**
On a single-brand corpus, pure vector search returns noise (observed band 0.67–0.84, everything
"similar"). Add structured fields and filter first, then rank: `market`/`region`, `tags`, `status`.
Query becomes "concluded seeding events in APAC" → then similarity.

Implemented alongside §6.3 (same commit — filtering needs the fields §6.3 adds). `store.py` gains
`filter_campaign_ids()` (record_type/status/tags-any-match/region/market, region+market matched
case-insensitively, tags freeform no fixed taxonomy per the locked decision below). `core.find_similar`
branches: no filters → the existing fast ANN `vectorstore.search`; any filter given → restrict to
the matching campaigns' chunks first (`vectorstore.get_many`, new) and brute-force-rank only that
candidate set (`embedding.rank`, already existed, now used) — correct at POC scale, no ANN
infrastructure needed for an already-narrow set. `find_similar_campaigns` and `prepare_evaluation`
both expose the filter params. `channel`/`campaign_type`/`brand_stage` from the original idea list
were **not** added as separate columns — folded into freeform `tags` instead (see §6.3).

### 6.3 Schema: split `kind`, add tags & region — **Done (2026-09-08)**
`kind` (concluded|proposal) is too binary — doesn't fit reference docs, stubs, finished campaigns
without results. Split into **`status`** + **`record_type`**, add a **`tags[]`** array, and
**`region`/`market`** as first-class.

**Decisions confirmed when building this:** `record_type` ∈ `campaign | reference | stub`;
`status` ∈ `proposed | in_flight | concluded`, defaulting to `concluded` only when
`record_type='campaign'` (reference/stub records have no lifecycle status unless explicitly set —
a `reference` record with status stuck at "concluded" made no sense). `tags[]` is **freeform, no
fixed taxonomy** — the "four-category taxonomy" mentioned in earlier notes wasn't available to
build against, so tags are just strings the user types; a controlled vocabulary can be added
later once real tag values are seen. `region`/`market` are freeform text, not a controlled list,
matching the conversational-intake style (§6.9) — a fixed region list can be added later too.

Schema change: `campaigns.kind` column removed entirely, replaced by `record_type`/`status`/
`tags` (JSON array)/`region`/`market`. No back-compat shim — consistent with the "reload OK"
decision; this was already true after §6.1's chunk-table change, so the same local DB reload
covers both.

### 6.4 Lifecycle: CRUD + supersede — **Done (2026-09-08)**
`delete_campaign`, `update_campaign`, and a **`supersedes`** field. Today every mistake is
permanent (the Mexico deck exists twice, dead + live, with nothing saying one replaces the other).

`campaigns` gains `supersedes` (the id this record replaces) and an auto-maintained reverse
pointer `superseded_by`; `find_similar`/`filter_campaign_ids` always exclude superseded records
from search evidence (both the filtered and the unfiltered ANN path) — direct `get_campaign`/
`list_campaigns` lookups still show them, only evidence ranking hides them. `delete_campaign`
cascades chunks/vectors/metrics, detaches (not deletes) any evaluation that cited it, and
restores an old record to active if the thing that superseded it gets deleted. `update_campaign`
is metadata-only (title/detail/record_type/status/tags/region/market) — NOT deck_text/chunks;
content changes go through a new upload + `supersedes`, by design (avoids the complexity of
re-chunking/re-embedding in place). `supersedes` is set only at creation, not editable via
`update_campaign` — kept simple; changing what a record supersedes after the fact is deferred.

### 6.5 Metrics as first-class + bulk import — **Done (2026-09-08)**
Distinguish **prediction vs actual**; `list_campaigns` should show whether a record has
metrics/evaluations; add a **bulk metrics/CSV import** so loading the KPI workbook is one step.
This is what makes `reconcile_evaluation` actually functional.

`metrics.metric_type` ('actual' default | 'predicted'). `get_campaign`/`list_campaigns` both
report `has_metrics`/`has_evaluations`. `bulk_import_metrics` (new tool) takes rows identifying
their campaign by `campaign_id` or exact case-insensitive `title` — LLM-first, matching the
product's existing pattern: Claude reads the workbook (CSV, pasted table, whatever form it's in)
and passes structured rows, rather than the server parsing a file format itself; ambiguous/
unmatched titles are per-row errors, never guessed, and valid rows still import when others fail.
`reconcile_evaluation` now auto-pulls a campaign's `metric_type='actual'` metrics when `actual`
isn't passed explicitly — this is the actual fix that makes it functional; before, the user had
to retype numbers that were already on file every time they wanted to reconcile.

### 6.6 Image vectorization + creative-reuse detection  *(the SVP question)* — **Done (2026-09-08, pHash + CLIP)**
Two techniques, two problems:
- **Perceptual hashing (pHash/dHash — `imagehash` + Pillow):** catches the **same/near-same photo**
  (reused, incl. across regions) even after resize/recompress/light crop. Cheap, no ML. **Ship
  first** — this is the demo: *"⚠ this hero image is identical to one used in the Mexico launch."*
- **CLIP visual embeddings:** catches **aesthetic/regional similarity** — same product, different
  photo; same styling; "looks like the APAC shoot." Needs the same region filtering as text.
- **Region-aware flag** (the SVP's real point): the flag is not "this image exists" but
  **"this image / its look belongs to a *different* region"** — regions are expected to differ.
- Tool: `check_image_provenance(image)` + auto-flag on proposal upload.
- **Boundary:** internal reuse ✅; unknown web stock photo ❌ (needs external reverse-image API — skipped).

**pHash layer implemented:** `images.py` (phash + Hamming distance, wrapped to plain Python
types — imagehash returns numpy scalars, which don't JSON-serialize cleanly through an MCP tool
response, caught in testing). New `upload_image_asset` and `check_image_provenance` tools;
`check_image_provenance` works on an image **before** it's stored (checked pre-upload, as the
doc asks), excludes the target campaign's own assets, and flags a match whose campaign has a
*different* `region` than the one passed in — delivering the actual SVP demo without needing
CLIP, since exact/near-duplicate reuse is a pHash match by definition. New deps: `Pillow` +
`imagehash` (pulls in `numpy`/`scipy`/`PyWavelets` transitively) — small, no GPU/ML, matches the
doc's own "cheap, no ML, ship first" framing, added to `requirements.txt` without a separate
confirmation gate.

**CLIP layer implemented (2026-09-08), after explicit confirmation to carry the dependency
weight.** New `clip_embed.py` — `open_clip_torch` + `torch`, model
`ViT-B-32-quickgelu`/`openai` (the `-quickgelu` variant matters: open_clip warns of an
activation-function mismatch against plain `ViT-B-32` with `openai` weights, which would
subtly degrade embedding quality — verified no warning with the correct name). Real
dependency cost, measured, not guessed: `torch` alone is a **121MB** wheel (macOS arm64 CPU
build), plus a **605MB** checkpoint (`ViT-B-32`) — shipped in the installer payload since
the 11 Sep product review (see docs/PRODUCT-REVIEW-PLAN.md 1.2), not downloaded; Linux/Windows
CPU wheels run similarly large. `vectorstore.py` gained a `space` parameter (default
`"campaign"`, backward compatible) so CLIP's 512-dim vectors and text's 768-dim
(nomic-embed-text) chunk vectors can coexist without sharing a fixed-width `vec0` column — a
new `"asset"` space, initialized in `store.init_db()`. New `find_similar_images` tool
(aesthetic/regional similarity, ranked by cosine similarity, same region-filter-first and
region-mismatch-flag pattern as `check_image_provenance`/`find_similar` — the flag logic is
shared via one helper, not duplicated). `ingest_image_asset` now does both pHash AND CLIP on
upload, independently — one failing doesn't block the other.

**Kept offline-testable the same way `embedding.py`'s text embedder already is:** a
`CAMPAIGN_POC_CLIP_PROVIDER=hash` test provider (`clip_embed.py`, pixel-hash based, NOT
semantically meaningful) means the automated test suite never downloads a model or imports
`torch`'s heavy paths at runtime, matching the existing `EMBED_PROVIDER=hash` convention.
Real `openclip` provider verified end-to-end manually (actual model download from the
`timm/vit_base_patch32_clip_224.openai` HF checkpoint, real embedding, correct region-flag
output, confirmed JSON-serializable through the MCP tool layer) — not exercised by CI.

README/requirements.txt updated — the "no CLIP/torch" line is gone; replaced with the actual
size tradeoff stated plainly.

**Packaging: was BROKEN, now FIXED and verified (2026-09-08).** An actual PyInstaller build
with torch included (removed the earlier blanket `excludes=["torch"]`, added `collect_all`
for torch/open_clip/timm) built (717MB) but the packaged binary crashed on startup:
`RuntimeError: operator torchvision::nms does not exist`.

Root cause, confirmed by direct inspection (not guessed): torchvision loads its ops
extension via `torch.ops.load_library(explicit_path)` (`torchvision/extension.py`) — a
runtime `dlopen`-by-path, not a Python `import` — so PyInstaller's static import-graph
analysis never sees it needs bundling. Neither `collect_all`'s `collect_dynamic_libs` (only
picks up `torchvision/.dylibs/*`, its vendored transitive deps like libpng/libjpeg) nor
`collect_data_files` (excludes binary-looking files) collects the extension itself
(`_C_stable.so`, sitting directly in the `torchvision/` package dir — confirmed empty-handed
by calling both directly). The bundled `_pyinstaller_hooks_contrib` hook for torchvision is
stale: it declares `hiddenimports = ['torchvision._C']`, the pre-0.29 name (torchvision
renamed the extension to `_C_stable.so` under a newer ABI-stable scheme) — a no-op against a
name that no longer exists (hence the build's "Hidden import torchvision._C not found!"
warning), and wouldn't have copied the binary even if the name were current, since
hiddenimports only affects the import-graph, not binary collection.

**Fix:** `campaign-poc.spec` now explicitly globs torchvision's own top-level `*.so` files
and adds them to `binaries` at the exact relative path torchvision's own extension-loader
expects (`os.path.dirname(__file__)` — right alongside `torchvision/__init__.py`).
Verified with an actual rebuild-and-run: `/healthz` responds with `clip_provider: openclip`
(proving the model loaded — `warm_up()` runs the exact `open_clip.create_model_and_transforms`
call that used to crash the whole process before startup), and a full MCP round-trip through
the live packaged binary (`upload_campaign` → `upload_image_asset` → `find_similar_images`,
via a real `mcp` streamable-HTTP client) returned `visually_embedded: true` and a real CLIP
similarity match. Source/dev usage was never affected either way.

### 6.7 Extensible asset pipeline (audio/video pluggable later) — **Done (2026-09-08) for image; audio/video not started**
Generic `assets` (modality) + `asset_fingerprints` (pHash) + `asset_vectors` (CLIP), keyed by
asset. **Video = keyframes → the image pipeline** (reused clips share keyframes — nearly free once
images work). **Audio later** = audio fingerprint + embedding, same shape.

The generic shape is fully in place for `image`: `assets.modality` column, `asset_fingerprints`
(pHash), and CLIP vectors live in `vectorstore`'s `"asset"` space (the module's generic
`space`/`dim` params ARE the "asset_vectors, keyed by asset" the doc asked for — no separate
table needed). Adding `video`/`audio` later is: extract representative frames/audio segments,
reuse `images.phash`/`clip_embed.embed_image` unchanged on keyframes, same `assets` row shape
with `modality='video'`/`'audio'`. Not built — no video/audio ingestion exists yet to feed it.

### 6.8 Minor — **Done (2026-09-08), one item unreproducible**
Trim `find_similar` payload (summary by default, full detail on request); fix `&` stored as
`&amp;`; **align the port** (binary/config default 8080 vs `run.ps1` default 8086 — make them match).

- **Port aligned to 8086** (not 8080) — `run.sh`/`run.ps1`/`LINUX.md`/`WINDOWS.md` and the
  actual local dev setup already standardized on 8086; `config.py`'s raw default and two
  README lines were the outliers and are now fixed to match, rather than the other way
  around (would have broken the already-wired Claude Desktop `mcp-remote` config).
- **`find_similar` trimmed**: each evidence row's freeform `detail` is cut to
  `config.EVIDENCE_DETAIL_SUMMARY_CHARS` (default 300) with a `detail_truncated` flag, opt out
  via `full_detail=True`. `prepare_evaluation` defaults `full_detail=True` instead — a real
  judgment over a short evidence list shouldn't work from trimmed briefs, only a browsing
  search (`find_similar_campaigns`) should default to trimmed.
- **`&` → `&amp;` — could NOT reproduce.** Checked `extract.py`'s PPTX text extraction directly
  (python-pptx correctly returns unescaped `&`) and grepped the whole codebase for any
  HTML/XML-escaping code — none exists here; nothing in this repo writes XML/HTML. Left as an
  open item rather than guessing at a fix for a bug with no locatable cause in this codebase —
  it may be a source-file artifact (a PPTX whose own XML has literal `&amp;` text) or from a
  display layer outside this repo, not a code defect here.

### 6.9 Conversational intake + confirm-before-write (uploads & feedback) — **Done (2026-09-08)**
Users are **non-technical marketers**, not people filling out a form. Both entry points —
uploading a campaign and recording feedback/outcomes — should be a guided conversation, not a
schema dump. Mostly a tool-description / system-prompt change (Claude already sits in front of
every MCP tool as the conversational layer); the new part is a **confirm-before-write step** so a
non-technical user can actually verify what got saved, rather than trusting the parse silently.

**Upload flow (`upload_campaign`).** Guide with questions like: is this a **finished campaign or
a future/proposed one**? What do you **like** about it? What do you **not like** / what's
missing? What are you trying to **achieve** (the goal)? Map answers onto the structured fields
from §6.2/§6.3 (`status`/`record_type`, `region`/`market`, `tags`, goal/notes) as the conversation
goes.

**Feedback flow (evaluation/outcome capture).** Guide with: **which campaign** (look it up /
disambiguate by name+region if ambiguous — ties to §6.4 lifecycle so the right record gets
updated), **how did it go**, **how was the response**, and **what metrics do you have** —
impressions, likes/engagement, footfall, sales, etc. (feeds §6.5 metrics-as-first-class).

**Both flows share one pattern:** if the user answers with a free-text paragraph instead of
answering field-by-field, **parse it into the structured breakdown** and **present it back**
("Here's what I got: type=future, goal=awareness, likes=X, concerns=Y, metrics={impressions: …,
footfall: …} — anything to fix?"). Only call the write tool (`upload_campaign` / `save_evaluation`
/ `add_metrics`) **after** the user confirms or edits. Never write silently from a raw parse.

Depends on §6.2/6.3 (structured fields to map onto) and §6.5 (metrics fields) landing first, or
at least in the same pass — the conversation needs somewhere structured to put the answers.

**Implemented as two things:** (1) tool docstrings on `upload_campaign` and `add_metrics`
rewritten to explicitly instruct the guided-conversation + parse-free-text + show-the-breakdown
pattern — this is genuinely a prompt-engineering change, since Claude *is* the conversational
layer, there's no separate Python "conversation engine" to build. (2) **A structural
confirm-before-write gate**, not just a prompted convention: both tools take `confirm` (MCP
tool default `False`); `confirm=False` returns a preview of exactly what would be stored/recorded
— echoing back the parsed fields — **without writing anything**, and only `confirm=True` (a
second, explicit call) commits. This makes "don't write silently from a raw parse" enforceable
rather than something an LLM could skip under time pressure. `core.ingest_campaign`/
`core.add_metrics` default `confirm=True` instead (backward-compatible for direct/programmatic
callers, e.g. tests, that already know what they want stored) — only the MCP tool layer defaults
to the safe preview-first behavior. Scoped to `upload_campaign`/`add_metrics` only — not
`bulk_import_metrics` (already-structured tabular data, not a free-text answer to parse) or
`save_evaluation`/`save_reconciliation` (Claude's own judgment, not the user's answer).

---

## 7. The data gap no code fixes

The library currently teaches the agent **what you like**, not **what works** — because there is
**no good-brief-that-underperformed** example, and nearly all outcome data is impressions, not
numbers. Loading the KPI workbook (§6.5) and capturing at least one strong-brief/weak-result case
is the highest-value non-engineering action.

---

## 8. Ops & compliance (later)

- **Backups:** Postgres (dumps) + the asset store; test restore.
- **Deploy/versioning:** Docker Compose on the VM; extend CI; tagged releases.
- **Compliance lens:** campaign decks may contain PII → access logging, least-privilege, and
  **region data-residency** (relevant given the regional model). Apply the SOC2/GDPR review at
  design time, not after.

---

## 8.5 Adversarial + design review round (2026-09-08)

After §6.1–6.9 landed, two independent fresh reviewers (no prior context on this session)
reviewed the full diff — one adversarial code review, one architecture/design review. Real,
verified findings were fixed across three follow-up commits (each TDD — a failing test
reproducing the bug, confirmed red, before the fix): the `_keep_asset` filename-collision
data-loss bug, `delete_campaign` leaking CLIP vectors + files, the `supersedes`/
`superseded_by` design flaw (a maintained reverse-pointer breaks on supersession chains and
fan-in — replaced with a live-derived query), two real search bugs (over-fetch starvation
before the per-campaign rollup, and a sqlite-vec `k`-limit crash from folding superseded
campaigns into the ANN exclude set), inconsistent nonexistent-id handling across five entry
points, `bulk_import_metrics` crashing on a malformed row, `reconcile_evaluation` dropping
structured-only actuals, missing `record_type`/`status`/`tags` validation, a `clip_embed`
model-loading race, two tool-surface gaps (`list_evaluations` missing, `get_campaign` not
showing image assets), and `POST /upload` silently rejecting images its own docs said it
accepted.

**Deferred, not fixed** (real observations, judged lower-value or higher-risk to fix
reactively than to plan properly):
- ~~PyInstaller + torchvision packaging~~ — **fixed same session, see §6.6.**
- `asset_fingerprints` is a 1:1 table with one column — neither a plain column on `assets`
  nor a proper multi-fingerprint table `(asset_id, kind, value)` that audio fingerprints
  (§6.7) would eventually want. Revisit when audio is actually built.
- Decks (`campaigns.asset_path`) and images (`assets` table) are two different asset
  models — unifying them into one generic asset table is a real simplification but a bigger
  schema change than this round should absorb reactively.
- `metrics.metric_type='predicted'` has no reader anywhere (only `'actual'` is consulted by
  `reconcile_evaluation`) — not wrong, just unused; revisit if a predicted-vs-actual variance
  report is ever built, or drop the option.
- `update_campaign`'s `tags` replace is arguably as deserving of the §6.9 confirm-before-write
  gate as `add_metrics` is (design review's point) — not added, to keep the gate scoped to the
  two tools actually fed by free-text conversational parsing.
- The confirm=False preview on `upload_campaign` only echoes the fields given; it doesn't
  resolve `asset_ref` or validate a `supersedes` target, so a preview can look fine and still
  fail (differently) on the `confirm=True` call. A fuller preview (cheap existence checks
  without doing the full extract/chunk/embed work) would close that gap.

## 8.6 Closing the delta against the original feedback (2026-09-09)

After §8.5's review round, went back to the **original feedback document** (the one that
actually drove §6.1–6.9 — its exact wording, not the paraphrase in this doc) and checked each
item against what was actually built. Most was addressed; three real gaps were found and closed:

- **"No relationships. Same collection different market... buried in prose."** New
  `campaigns.collection` (freeform, like region/market) — market/version variants of the same
  creative now share an explicit, queryable value instead of only being findable by
  title-guessing. `supersedes` already covered v1→v2 (replacement, asymmetric);
  `collection` covers siblings (variants, symmetric) — a different relationship, not a
  duplicate of supersedes.
- **The four-category taxonomy, defined:** `liked` / `not_liked` / `mixed_reaction`
  (creative reaction) and `performed_well` / `underperformed` / `performed_as_expected` /
  `no_data_yet` (performance) — two independent axes that commonly co-occur on one campaign
  (documented as `SUGGESTED_TAGS` in `mcp_server.py`, not schema-enforced — tags stay
  freeform). Discovering this needed the query capability below, which didn't exist yet:
  **tags matching was OR-only** ("has any of the given tags"), which cannot answer "show me
  campaigns that are BOTH liked AND underperformed" — the actual quadrant query the taxonomy
  exists to enable. Added `match_all_tags=True` (AND-match) to `filter_campaign_ids`/
  `find_similar`/`find_similar_campaigns`/`prepare_evaluation`.
- **Tag provenance — verified vs. stated:** "Four of our five performable tags are
  currently impression, not measurements. Without that distinction, the tag reads as
  evidence when it isn't, and the agent will weight it as if it were." Tags are no longer
  plain strings — each is `{"value": str, "source": "verified"|"stated"}` (a plain string
  still works for ergonomics and defaults to `"stated"`, the conservative assumption).

  **First version of this was flawed** — two more independent reviews (adversarial + design,
  same fresh-agent process as §8.5) on this specific change caught it: `source='verified'`
  had no structural connection to anything (anyone could claim it with zero metrics behind
  it — the exact trust problem the feature exists to fix, just moved one level down), and a
  global `verified_tags_only` flag couldn't express the real quadrant query at all (a
  creative-reaction tag like "liked" can never itself be "verified" the way a performance
  tag can — a flag requiring every queried tag to be verified made "liked AND verified-
  underperformed" unanswerable). Both fixed: `source='verified'` is now REJECTED unless the
  campaign already has a `metric_type='actual'` record on file (enforced in
  `store.normalize_tags`, not just documented — a brand-new campaign can never satisfy this,
  so verified tags can only be set via `update_campaign`, after `add_metrics`, never at
  creation). `verified_tags_only` is gone; tag queries now carry per-tag source instead
  (mirroring the storage shape) — `tags=["liked", {"value": "underperformed", "source":
  "verified"}]` expresses "liked, any evidence, AND underperformed, but only if verified" —
  the actual quadrant query, precisely.

  Also fixed in this pass: a database from before `collection` existed (including the
  already-shipped v0.2.0 release) would have failed on every operation — `CREATE TABLE IF
  NOT EXISTS` does nothing for a column added to an existing table. `init_db()` now runs an
  additive migration (`ALTER TABLE ... ADD COLUMN` when missing, never destructive).
  Pre-existing plain-string tags (from before provenance existed) are now tolerated on read
  rather than crashing the new filtering code. `get_campaign` gained a derived
  `collection_siblings` (mirroring `superseded_by`) — a collection value on its own wasn't
  usable from a single record without a way to find the other members. Metrics trimming in
  evidence now keeps the most decision-relevant rows (`actual` before `predicted`, most
  recent within each) instead of the oldest by insertion order. The `confirm=False` preview
  now normalizes (and validates) tags the same way `confirm=True` would, so what the user
  reviews actually matches what gets stored.

Verified end-to-end, including through a real MCP round-trip: the literal "Mexico record"
scenario from the feedback (liked, stated + underperformed, verified) is retrievable via
`tags=["liked", {"value":"underperformed","source":"verified"}], match_all_tags=True`, and
attempting to mark a tag verified on a campaign with no actual metrics is correctly
rejected. 199 tests passing (was 169 before this round; 187 after the first, flawed pass;
199 after the fixes).

**A third review round** (verification-focused, same fresh-agent process) confirmed all of
the above actually holds up under tracing — plus caught one more real bug: a bare
`ValueError` raised from inside a tool body (exactly what the new tag-provenance validation
raises) was being swallowed by the MCP framework and replaced with a generic "Error
executing tool X" — none of the carefully-written validation messages anywhere in this
codebase were reaching the caller, verified by reproducing it over a real MCP streamable-
HTTP client. Fixed with a `_catch_value_errors` decorator applied to every `@mcp.tool()`
function, converting to the same `{"error": str(exc)}` convention already used for
not-found cases. Re-verified over the live protocol afterward — the actual message now
reaches the client. 204 tests passing.

Minor items from the third round intentionally left as-is (documented, not silent): a
`'verified'` tag is a one-time write gate, not a live invariant — if a `delete_metrics` tool
is ever added, a tag could theoretically outlive the metric that justified it (no such tool
exists today); `_keep_asset` still copies a file before tag validation runs on
`confirm=True`, leaving an orphan on rejection (pre-existing pattern, not introduced here);
`collection_siblings` doesn't exclude superseded records the way `filter_campaign_ids` does
(mirrors `superseded_by`'s own behavior, arguable either way); an update on a nonexistent
campaign_id with an invalid verified tag reports the tag error before the not-found error.

**Still open, not fixed:**
- **`&` → `&amp;`** — still unreproduced; no HTML/XML-escaping code found anywhere in the
  repo across two attempts. Needs a specific record to chase it in.
- **Outcome data itself** — the mechanisms now exist (`bulk_import_metrics`, `metric_type`,
  tag provenance), but no software fix produces real KPI data that hasn't been loaded yet.
  §7's "missing quadrant" data gap is unchanged until the actual workbook is loaded.

## 8.7 Automatic image extraction from uploaded decks (2026-09-10)

§6.6's `check_image_provenance`/`upload_image_asset` pair only ever worked as a **standalone**
path — a user (or Claude) had to upload each image separately, one call per image. Real
decks carry their images baked in; the user's own correction: *"Image reuse detection can't
just be outside of the deck as an independent functionality... you really want users to be
doing this for every image? who'll use it then?"* A day before a live demo, chose to build
this now rather than after, despite it touching `core.ingest_campaign` — the single most
load-bearing function in the product, with no time left for the multi-round review cycle
that caught real bugs in every prior pass (§8.5, §8.6).

- New `extract.extract_images(path)` (PDF via `pypdf`'s `page.images`, PPTX via
  `python-pptx`'s `shape.image.blob`/`.ext` on `MSO_SHAPE_TYPE.PICTURE` shapes), capped at
  `config.MAX_EXTRACTED_IMAGES_PER_DECK` (default 20, bounds per-upload phash/CLIP cost on a
  large deck).
- `core.ingest_campaign` now runs this automatically whenever `asset_ref` resolves to a real
  file — including **alongside** `deck_text` (see the fixed-bug note below), not only on the
  file-only path. The pure LLM-first path with no `asset_ref` at all still has no bytes to
  extract images from — an inherent limit, not a bug. Each extracted image is stored,
  pHash-fingerprinted, CLIP-embedded, and checked for reuse against every other campaign's
  image, reusing `_phash_matches` (shared with `check_image_provenance`). The fingerprint/
  CLIP-embed steps are NOT shared with `_store_and_fingerprint_image` (`upload_image_asset`'s
  helper) — that helper doesn't compute reuse flags inline, so this path re-implements those
  two steps rather than bolt reuse-checking on after calling it; an earlier version of this
  doc claimed it was shared, which a design review caught as doc-code drift. New
  `core._keep_asset_bytes` writes extracted image bytes straight to `ASSET_DIR` (no source
  file exists to copy, unlike `_keep_asset`).
- Response gains `image_assets: list[dict]` (`asset_id`, `fingerprinted`, `visually_embedded`,
  `reuse_flags` per image) and `images_checked: bool`. Each per-image step fails independently
  into `warnings` (matching the existing chunk-embedding-failure philosophy) rather than
  failing the whole upload. The standalone `upload_image_asset`/`check_image_provenance`
  tools are unchanged and remain available as a secondary, explicit path.
- Verified with the literal "Mexico record" scenario as a test (`tests/test_deck_image_extraction.py`):
  the same hero image embedded in decks uploaded for `region="LATAM"` then `region="APAC"`
  produces an automatic `reuse_flags` entry on the second upload (`hamming_distance: 0`,
  flagged as a different-region reuse) with no manual per-image step — verified both directly
  and over a real MCP `streamable_http_client` round-trip.

**Independent review round** (adversarial + design, two parallel fresh agents, same process
as §8.5/§8.6, run despite the timeline because the standing instruction has no time-pressure
carve-out) — both found real, demo-relevant bugs, all fixed and covered by new regression
tests in the same pass:

- **Design review — the feature was invisible on the demo's own documented flow.** This doc's
  own Run-it section says the demo is "attach a campaign PDF... Claude reads the deck and
  calls upload_campaign" — text Claude read itself, i.e. `deck_text`. The code only ran image
  extraction when `asset_ref` was given **and `deck_text` was NOT**
  (`if asset_ref and not deck_text:`), while the docstring claimed "instead of/alongside
  deck_text" — the realistic call shape (Claude passing both: text it already read, plus the
  file for storage) silently skipped extraction entirely. **Fixed:** extraction now runs
  whenever `asset_ref` resolves, independent of `deck_text` (the caller's `deck_text`, when
  given, is kept as-is, not overwritten by server-side extraction). Also fixed:
  `image_assets: []` was indistinguishable between "deck had no images" and "never checked"
  (no file, unsupported type, or extraction failed) — added `images_checked: bool`, and
  rewrote `upload_campaign`'s docstring to tell the calling LLM to check it before reporting
  "no reuse found," and to treat any non-empty `reuse_flags` entry as reuse worth mentioning
  (not only entries whose `flag` field is non-null — a null `flag` means same-region reuse,
  still real reuse, just not itself the cross-region signal).
- **Adversarial review — three bugs that would each make a real reused image go undetected,**
  reproduced with real corrupt/placeholder/grouped/repeated images before fixing:
  1. One unsupported or corrupt embedded image (e.g. WebP — `python-pptx`'s `shape.image.ext`
     raises for any format outside its known map) aborted extraction for the **entire deck**,
     not just that image. Fixed: each image is now extracted inside its own `try`/`except`;
     one bad image is skipped with a warning, the rest of the deck still gets checked.
  2. A picture inserted into a template's picture placeholder (`shape_type == PLACEHOLDER`,
     how corporate decks built from a layout typically place images) and a picture nested
     inside a grouped shape were both silently skipped — `shape_type == PICTURE` doesn't
     match either. Fixed: `isinstance(shape, pptx.shapes.picture.Picture)` (covers
     `PlaceholderPicture`, a `Picture` subclass) plus recursion into `GROUP` shapes.
  3. The same image repeated across slides (e.g. a per-slide logo) was extracted and
     cap-checked once per occurrence — on a deck with ≥`MAX_EXTRACTED_IMAGES_PER_DECK` slides
     carrying a footer logo, the cap fills with logo copies and a genuinely distinct hero
     image later in the deck is **never extracted at all**, the exact failure mode that would
     hide the demo's headline flag. Fixed: images are deduped by content hash to their first
     occurrence, *before* the cap is applied, in `extract._dedup_and_cap`.
  4. Two related correctness/consistency fixes taken in the same pass: `_phash_matches` is
     now called only after `store.set_asset_fingerprint` succeeds (previously the other way
     round — a match-query failure would discard a fingerprint that was in fact already
     computed, hiding this image from every future upload's reuse check, not just this one);
     and an image file written by `_keep_asset_bytes` is now unlinked if the following
     `store.insert_asset` fails, closing an orphaned-file leak.
  5. Each `extract_images` tuple now carries the 1-based slide/page number
     (`image_assets[].location`) so a flagged image can be pointed to — the natural
     follow-up question ("which slide?") a multi-image deck couldn't previously answer.

Verified with the literal "Mexico record" scenario (same hero image, decks uploaded for
`region="LATAM"` then `region="APAC"`, `deck_text` passed alongside `asset_ref` on both — the
exact shape the design review flagged as broken) both directly and over a real MCP
`streamable_http_client` round-trip. 227 tests passing (was 216 before this round; 218 after
the first, still-flawed pass; 227 after the fixes) — 20 of them new to this specific round
(dedup, placeholder/group pictures, per-image resilience, cap-starvation, fingerprint-before-
match ordering, orphaned-file cleanup, `images_checked` semantics under legacy/unsupported
file types).

**Deferred, flagged rather than silently skipped, given the timeline:** no independent
verification of real-Windows behavior for this feature or the §6.6 PyInstaller torchvision
`.pyd` fix it depends on; both remain inspection-verified only. Recommend testing on the
actual Windows box before the demo, ideally tonight.

## 8.8 `markets`: multi-country activation, found in live testing (2026-09-10)

Real usage testing the night before a demo, on freshly re-tagged data (the "collection/tag-
provenance" work landing meant `region`/`market`/`collection` were finally populated on
every record) surfaced a genuine gap the collection/tag-provenance work didn't anticipate:
`region`/`market` are single-value, exact-match strings. A campaign whose actual activation
spanned several countries (e.g. a SEA launch that ran in Malaysia AND Indonesia) can only be
tagged with ONE of them as `region` — querying `region="Indonesia"` for a record tagged
`region="Malaysia"` returns nothing, even though the activation genuinely included
Indonesia. That reads as "no such campaign," which is worse than no filter at all.

Confirmed via live testing, not assumed: filtering itself works correctly once fields are
populated (a filtered query on the real corpus went from a flat 0.75–0.81 similarity band to
a real 0.81/0.75/0.64 spread, correctly excluding unrelated campaigns and ranking a
no-campaign "stub" record last) — the gap was specifically that a multi-country activation
has no single truthful value for a single-value field. Storing a delimited composite
("Malaysia, Indonesia") doesn't work either (confirmed by testing it): exact-match filtering
needs atomic values, so a human-readable composite matches nothing.

- New `markets` column: `list[str]`, additive (mirrors `tags`' storage shape/philosophy, no
  provenance concept needed here), matched by case-insensitive **membership** rather than
  exact-match — the opposite of region/market on purpose. `region`/`market` are unchanged in
  meaning: `region` stays the single primary country/geo tag, `market` stays the freeform
  grouping label (e.g. "SEA"); `markets` is additive, carrying the full list of countries/
  sub-markets an activation actually touched.
- `store.normalize_markets`/`_parse_stored_markets`, additive `_migrate_schema` column (NOT
  NULL with a `'[]'` default — SQLite allows this on `ADD COLUMN`, backfilling existing
  rows, verified against a reconstructed pre-`markets` database), threaded through
  `insert_campaign`/`update_campaign`/`get_campaign`/`list_campaigns`/`filter_campaign_ids`
  and up through `core.find_similar`/`prepare_evaluation` and the
  `upload_campaign`/`update_campaign`/`find_similar_campaigns`/`prepare_evaluation` tools.
- **Independent review round** (adversarial + design, same process as prior rounds) found no
  correctness bugs in the storage/migration/filtering logic itself, but caught one real
  ergonomics bug before merge: the write-side `markets` param is a list
  (`markets=["Malaysia","Indonesia"]`) but the query-side param was typed as a single `str`
  — an LLM that just wrote a list will plausibly query with one too, which failed at the
  MCP/pydantic schema layer with a generic input-validation error. Also identified as the
  answer to a real design question: an "Indonesia OR Thailand" query was otherwise
  unreachable in one call. Fixed by accepting `Union[str, list[str]]` on the query side
  (`filter_campaign_ids`/`find_similar`/`prepare_evaluation`/the two search tools) with
  ANY-match semantics for a list — verified both directly and over a real MCP
  `streamable_http_client` round-trip (single-string and list-of-two-countries queries both
  return the correct campaigns). 244 tests passing (was 227 before this round).
- **Deferred, flagged rather than silently fixed:** `region`/`market`/`collection` still
  apply no normalization (no whitespace trim) on write, a pre-existing inconsistency the
  review surfaced (not introduced by this change) — `region=" Malaysia "` is stored verbatim
  and unreachable by `region="Malaysia"`, while `markets` now holds a stricter standard
  (strips, rejects empty entries). `find_similar_images`'s region-mismatch flag (§6.6) still
  compares only the single `region` field, not `markets` — a campaign tagged
  `region="Malaysia", markets=["Malaysia","Indonesia"]` versus one tagged
  `region="Indonesia"` still flags as "different region" even though both activations
  touched Indonesia. Neither blocks tonight's use case; both are real follow-ups.

## 9. Sequence

1. **Now:** local Windows end-to-end green (Desktop + server + Ollama), campaigns loaded. **Done**
   (2026-09-08) — ran end-to-end locally; §6 below is the feedback from that run.
2. **v0.2:** the §6 features (chunking, filters, schema, CRUD, metrics, image pHash + CLIP,
   conversational intake) — data reloaded fresh. **§6.1–6.9 together are considered the bar for a
   complete v1 product** — not a partial cut of them. **Status (2026-09-08): §6.1–6.9 all done**,
   including CLIP (confirmed by the user to carry the torch/model-size dependency — see §6.6).
   169 tests passing (`tests/`) after an adversarial + design review round (§8.5) — CI
   (`test.yml`) runs them on every push/PR.
3. **Central deploy:** Postgres, Docker on the VM, TLS (Front Door / App Gateway), **Entra OAuth**
   via the pluggable provider, access scoping via group claims.
