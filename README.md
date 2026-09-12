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
deps + a 605MB model checkpoint); see `docs/PRODUCTION-ROADMAP.md` §6.6.

**The installers ship the checkpoint.** An installed copy finds it beside the executable
(`<install dir>/models/`) and needs no network for it — including on a fully air-gapped
machine, with nothing configured. A **source checkout** has no bundled copy and resolves the
model by tag from huggingface.co on first use; run `python scripts/fetch_weights.py models`
once if you want the offline behaviour locally too.

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

> Auth: `auth.py` is a no-op seam today (`CAMPAIGN_POC_AUTH_MODE=none`). claude.ai connectors
> generally want OAuth — implement it in `auth.authenticate()` and set the mode to `oauth`.

## Tools (what Claude calls)

`upload_campaign` · `update_campaign` · `delete_campaign` · `add_metrics` ·
`bulk_import_metrics` · `upload_image_asset` · `check_image_provenance` ·
`find_similar_images` · `list_campaigns` · `get_campaign` · `find_similar_campaigns` ·
`prepare_evaluation` · `save_evaluation` · `list_evaluations` · `reconcile_evaluation` ·
`save_reconciliation`

## Files

```
config.py        env-driven config
store.py         SQLite schema + CRUD (campaigns, campaign_chunks, assets, asset_fingerprints, metrics, evaluations, reconciliations)
vectorstore.py   sqlite-vec vector table, keyed by chunk id (+ pure-Python cosine fallback)
embedding.py     ollama | voyage | hash embedders + cosine
extract.py       PDF / PPTX text extraction, one unit per page/slide
chunking.py      packs text units into embeddable chunks (server-side, per slide/section)
images.py        perceptual hashing (pHash) for exact/near-duplicate creative-reuse detection
clip_embed.py    CLIP visual embeddings for aesthetic/regional similarity (torch, heavy)
core.py          ingest + chunk + semantic retrieval + evidence packaging (LLM-first)
mcp_server.py    MCPServer tools (the API Claude calls)
http_app.py      Streamable-HTTP app (/mcp) + /upload + /healthz
auth.py          no-op auth seam (OAuth goes here)
main.py          unified binary entry (serve | stdio)
stdio_server.py  stdio entrypoint for a local Claude Desktop connector
```

## Install / packaging

Run from source, or ship a self-contained bundle (no Python on the target):

- **Windows** — `run.ps1 -Setup` (venv + deps + **auto-installs Ollama** + pulls the model),
  then `run.ps1` (on demand) or `run.ps1 -Mode service` (background). See **[WINDOWS.md](WINDOWS.md)**.
- **Linux / macOS** — `./run.sh --setup`, then `./run.sh` or `./run.sh --service` (systemd
  `--user`). See **[LINUX.md](LINUX.md)**.
- **Prebuilt bundles** — `campaign-poc.spec` builds a one-folder bundle via PyInstaller
  (the `sqlite-vec` native lib is bundled). `build.sh` / `build.ps1` build for the current OS;
  CI (`.github/workflows/build.yml`) builds Linux + Windows + macOS bundles on a `v*` tag.

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
