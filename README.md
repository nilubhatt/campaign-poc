# Campaign Intelligence (lean)

An **LLM-first** marketing-campaign memory. Python owns the files, database, and semantic
search; **Claude does the judgment**. You call it from **Claude Web / cowork** as a custom
connector (remote MCP server) — or any MCP client.

The loop it enables:
1. **Build memory** — upload past campaigns (deck text + whatever detail you have) and their
   metrics.
2. **Analyze** — for a new proposal, Claude gets the most similar prior campaigns *with their
   outcomes*, judges it, and **cites specific priors**.
3. **Reconcile** — after the campaign concludes, feed the real metrics; Claude compares its
   prediction to reality and stores the lesson, so future judgments get calibrated.

## Stack — local + free

| Concern | Choice |
|---|---|
| Database + **vector search** | SQLite + **`sqlite-vec`** (real KNN; falls back to pure-Python cosine if the extension is absent) |
| File processing | `pypdf` (PDF), `python-pptx` (PPTX) |
| Embeddings | **Ollama** `nomic-embed-text` (local, free) — default. `voyage` (paid, better) or `hash` (offline smoke-test only) are swaps via `CAMPAIGN_POC_EMBED_PROVIDER` |
| Reasoning | **Claude** (the intelligence you're paying for) |
| Transport | MCP over **Streamable HTTP** (`/mcp`) — what Claude Web connectors use |

No Postgres, no Docker required. CLIP (`torch` + `open_clip_torch`) IS a dependency, for
aesthetic/regional image-similarity detection — a deliberate size tradeoff (~150-250MB of
deps + a 303MB fp16 model checkpoint); see `docs/PRODUCTION-ROADMAP.md` §6.6.

**The installers ship the checkpoint.** An installed copy finds it beside the executable
(`<install dir>/models/`) and needs no network for it — including on a fully air-gapped
machine, with nothing configured. A **source checkout** has no bundled copy and resolves the
model by tag from huggingface.co on first use; run `python scripts/fetch_weights.py models`
once if you want the offline behaviour locally too.

### Weights provenance — verify rather than trust

The shipped checkpoint is vendored in this repo's [`weights-v1`](https://github.com/nilubhatt/campaign-poc/releases/tag/weights-v1)
release, because the networks this product targets block huggingface.co. You should not have
to take that file on trust:

| | |
|---|---|
| shipped (fp16) | `cbd90e47b939016c1cb2e3dc63e3d5ee3d663da57dc929c6ebb3067eedd0c452` · 302,588,458 bytes |
| upstream (fp32) | `e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31` · 605,143,284 bytes |
| upstream source | `timm/vit_base_patch32_clip_224.openai` @ `a6f597a30f7b82c51704746581f9a4e41421e878` |

The conversion is deterministic, so you can rebuild the exact bytes:

```bash
python scripts/convert_fp16.py upstream_fp32.safetensors rebuilt.safetensors
shasum -a 256 rebuilt.safetensors   # matches the fp16 hash above
```

CI does this on every change to the weights tooling (`.github/workflows/weights-provenance.yml`)
and fails if the rebuild stops matching, then attaches build provenance — so
`gh attestation verify open_clip_model.safetensors --repo nilubhatt/campaign-poc` confirms
the bytes came from that workflow rather than from someone's laptop.

### Offline / restricted networks — pointing at local CLIP weights

Resolving the model by tag goes to **huggingface.co**, which plenty of corporate networks
block outright (endpoint filters answering :443 in plaintext, so it fails as a TLS error
rather than an obvious block). Installed copies don't do this at all. For a source checkout,
or weights supplied out of band, point at the file and the product never touches the network:

```jsonc
// claude_desktop_config.json — the env block is how a local server gets settings
"env": { "CAMPAIGN_POC_CLIP_WEIGHTS_PATH": "C:\\path\\where\\you\\put\\the\\weights" }
```

Point it at the checkpoint file (`open_clip_model.safetensors` or
`open_clip_pytorch_model.bin`) or the folder containing it. `CLIP_WEIGHTS_PATH` works too;
the prefixed name wins if both are set. Verified to make **zero** network calls, producing
vectors identical to the tag-resolved model.

Nothing here is fatal: if the weights are missing, unreadable or corrupt, the server still
starts and text search, upload and evaluation work normally — only visual similarity is
off. It says which: an `[campaign-intelligence]` line on stderr at startup (in Claude
Desktop's MCP log for a local stdio server), and `clip_weights` in `GET /healthz` when
running over HTTP.

## Run it

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# real semantic search (local + free): install Ollama, then
ollama pull nomic-embed-text        # one time

python -m http_app                  # serves http://0.0.0.0:8086  (/mcp, /upload, /healthz)
```

Check it: `curl http://localhost:8086/healthz` →
`{"status":"ok","vector_backend":"sqlite-vec","embed_provider":"ollama",...}`

Offline smoke test (no Ollama): `CAMPAIGN_POC_EMBED_PROVIDER=hash python -m http_app`
(the `hash` embedder is not semantic — for pipeline testing only).

## Connect from Claude Web / cowork

1. Expose the server on a reachable HTTPS URL — locally, a tunnel:
   `cloudflared tunnel --url http://localhost:8086` (or `ngrok http 8086`).
2. Claude Web → **Settings → Connectors → Add custom connector** → point at `<public-url>/mcp`.
3. In a chat: attach a campaign PDF and say *"upload this as a concluded campaign, detail: …"*;
   Claude reads the deck and calls `upload_campaign`. Then *"analyze this new proposal against
   our history"* → Claude calls `prepare_evaluation`, judges, and `save_evaluation`.

> **Attaching a file in chat is not the same as sending the file.** Claude reads the deck and
> passes the text, so the server never sees the bytes — and comments, speaker notes and
> embedded images can only be read from the file itself. To capture those, POST it to
> `/upload`, or use Claude Desktop where a local path can be passed as `asset_ref`. The
> upload result says which happened: `commentary_checked: false` means nothing was read, not
> that the deck had no comments.

> Auth: `auth.py` is a no-op seam today (`CAMPAIGN_POC_AUTH_MODE=none`). claude.ai connectors
> generally want OAuth — implement it in `auth.authenticate()` and set the mode to `oauth`.

## Tools (what Claude calls)

`upload_campaign` · `update_campaign` · `delete_campaign` · `add_metrics` ·
`bulk_import_metrics` · `upload_image_asset` · `check_image_provenance` ·
`find_similar_images` · `upload_image_assets` · `compare_execution` · `check_commitments` ·
`list_commitments` · `add_commitment` · `drop_commitment` ·
`classify_drift` · `drift_readings` · `record_context_event` · `campaign_context` · `withdraw_context_event` · `attribute_outcome` · `withdraw_attribution` ·
`list_campaigns` · `get_campaign` · `find_similar_campaigns` ·
`diff_campaigns` · `coverage` · `gaps` · `getting_started` · `prepare_evaluation` · `save_evaluation` · `get_evaluation` ·
`list_evaluations` · `reconcile_evaluation` · `save_reconciliation` · `resolve_measure` ·
`measure_status` · `graduate_measure` · `note_correction` · `list_corrections` ·
`correction_status` · `resolve_correction` · `graduate_correction` ·
`set_aside_correction` · `reopen_correction` · `keep_correction` · `replay_rules` ·
`finish_indexing` ·
`health_check`

## Files

```
config.py        env-driven config
store.py         SQLite schema + CRUD (campaigns, campaign_chunks, assets, asset_fingerprints, metrics, evaluations, reconciliations)
vectorstore.py   sqlite-vec vector table, keyed by chunk id (+ pure-Python cosine fallback)
embedding.py     ollama | voyage | hash embedders + cosine
extract.py       PDF / PPTX text, embedded images, and commentary (comments, annotations,
                 speaker notes) — one text unit per page/slide
chunking.py      packs text units into embeddable chunks (server-side, per slide/section)
images.py        perceptual hashing (pHash) for exact/near-duplicate creative-reuse detection
clip_embed.py    CLIP visual embeddings for aesthetic/regional similarity (torch, heavy)
core.py          ingest + chunk + semantic retrieval + evidence packaging (LLM-first)
facts.py         the mechanical checks the SERVER establishes about a brief (dates, budget,
                 engagement rates, channels) — computed facts, not judgments
enums.py         the vocabularies, and refusals that name the valid set and the near miss
notices.py       one shape for "this is degraded and here is what you lose"
actions.py       the offers a result carries — every one a call that can actually be made
learning.py      the shared loop: how a measure or a rule becomes expected, and of whom
metrics.py       the metric registry and typed values (canonical names, units, targets)
corrections.py   standing corrections — client feedback on the same loop, with provenance
commitments.py   the promises a brief names, and whether the photographs show them
drift.py         what somebody made of the difference — recorded, never inferred
context.py       what else was going on: events by market and date, linked by overlap
calendar_seed.py the fixed calendar shipped with the product, hedged by certainty
replay.py        what changed when the rules changed, as a report; never a rewrite
agreement.py     the golden-set harness: verdict agreement and self-consistency
mcp_server.py    MCPServer tools (the API Claude calls)
http_app.py      Streamable-HTTP app (/mcp) + /upload + /healthz
auth.py          no-op auth seam (OAuth goes here)
main.py          unified binary entry (serve | stdio)
stdio_server.py  stdio entrypoint for a local Claude Desktop connector
```

### After rebuilding: fully quit the host app

An MCP host caches each tool's schema when it connects. Rebuild the server mid-session and
the client keeps the old schemas — parameters that are live and working simply are not
there, with no error to say so. This cost a reviewer an afternoon of rediscovering
`markets`, `status`, tag `source`, `match_all_tags` and `confirm` by trial and error against
a server that already supported all five.

**Closing the window is not enough** — the host keeps the server process running. Fully quit
the app (`Cmd-Q` / File → Exit) and reopen it.

To check what the running server actually accepts: `health_check` returns `version`, `build`
and a `tools` map of every tool to its real parameter list, read off the functions
themselves. Compare that against the schema in hand and the gap is named rather than
guessed at. `campaign-intelligence --version` answers the same question from a terminal, and
works on a machine where the database, the model and the embedder are all broken.

## Install / packaging

Run from source, or ship a self-contained bundle (no Python on the target):

- **Windows** — `run.ps1 -Setup` (venv + deps + **auto-installs Ollama** + pulls the model),
  then `run.ps1` (on demand) or `run.ps1 -Mode service` (background). See **[WINDOWS.md](WINDOWS.md)**.
- **Linux / macOS** — `./run.sh --setup`, then `./run.sh` or `./run.sh --service` (systemd
  `--user`). See **[LINUX.md](LINUX.md)**.
- **Prebuilt bundles** — `campaign-poc.spec` builds a one-folder bundle via PyInstaller
  (the `sqlite-vec` native lib is bundled). `build.sh` / `build.ps1` build for the current OS;
  CI (`.github/workflows/build.yml`) builds Linux + Windows + macOS bundles on a `v*` tag.

### Installing a bundle

Each archive carries the installer for its own platform: `installer/linux/install.sh`,
`installer/macos/install.sh`, and the Inno Setup `.exe` for Windows. All three ensure Ollama
and the embedding model, create the data directory, and then **run a post-install self-test**
that names the failing component and refuses to connect Claude Desktop if anything is wrong —
rather than leaving it to be discovered later by a tool call that times out. The Unix
installers additionally verify the shipped CLIP weights against a `.sha256` sidecar before
anything else; on Windows, Inno Setup's own per-file integrity check covers the same ground.

On Unix the new copy is staged and only swapped in once the self-test passes, so a failed
upgrade leaves the working install untouched. On Windows the files are installed first (Inno
has no seam for "keep the files but fail the run"), so a failed self-test shows a *"Installed,
but not working"* finish page naming the component, with Claude Desktop left unconnected.

To re-run that check at any time:

```
campaign-intelligence health-check          # human-readable; exit 1 if anything is wrong
campaign-intelligence health-check --json   # the same report, for a script
```

The install needs no network for the product itself: the weights ship in the archive, and CI
proves it on all three platforms by running the packaged binary's full health check — which
loads the vision model, the step that would reach for the Hub — inside an empty network
namespace on Linux, and with every HTTP request routed to a closed port on macOS and Windows.
Ollama's own installer does need internet; `--no-ollama` skips it if the machine already
has it.

Both **Claude Web** (custom connector → `<tunnel-url>/mcp`) and **Claude Desktop** (stdio, or
`mcp-remote` to the local HTTP server) are supported.

## Roadmap

This is being extended into the full product — a central multi-user server (Postgres + pgvector,
CLIP image vectors + perceptual-hash creative-reuse detection, region-scoped metadata, pluggable
OAuth). Server-side chunking (per slide/section, one vector per chunk) already ships locally —
the central move swaps the vector store for pgvector, same chunking. Current priority is local
end-to-end; production/OAuth/deploy is
captured in **[docs/PRODUCTION-ROADMAP.md](docs/PRODUCTION-ROADMAP.md)**. Auth is already a
**pluggable provider** (`auth.py`) — no-op today, drop in Azure AD / any OIDC without touching call sites.

## Status

End-to-end verified against a live server with the real `sqlite-vec` backend: full
memory→judge→reconcile loop over MCP-over-HTTP, plus `/upload` → ingest-by-`asset_id` with
server-side PPTX/PDF extraction. Embeddings need Ollama running for real semantic quality
(the pipeline itself is proven with the offline `hash` provider).

Not built yet (deliberately): OAuth, multi-user isolation, cleanup/retention, a migration to
Postgres+pgvector (the `vectorstore` interface is the single swap point when you outgrow
single-node).
