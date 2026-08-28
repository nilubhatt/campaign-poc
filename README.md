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

No Postgres, no CLIP/torch, no Docker required.

## Run it

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# real semantic search (local + free): install Ollama, then
ollama pull nomic-embed-text        # one time

python -m http_app                  # serves http://0.0.0.0:8080  (/mcp, /upload, /healthz)
```

Check it: `curl http://localhost:8080/healthz` →
`{"status":"ok","vector_backend":"sqlite-vec","embed_provider":"ollama",...}`

Offline smoke test (no Ollama): `CAMPAIGN_POC_EMBED_PROVIDER=hash python -m http_app`
(the `hash` embedder is not semantic — for pipeline testing only).

## Connect from Claude Web / cowork

1. Expose the server on a reachable HTTPS URL — locally, a tunnel:
   `cloudflared tunnel --url http://localhost:8080` (or `ngrok http 8080`).
2. Claude Web → **Settings → Connectors → Add custom connector** → point at `<public-url>/mcp`.
3. In a chat: attach a campaign PDF and say *"upload this as a concluded campaign, detail: …"*;
   Claude reads the deck and calls `upload_campaign`. Then *"analyze this new proposal against
   our history"* → Claude calls `prepare_evaluation`, judges, and `save_evaluation`.

> Auth: `auth.py` is a no-op seam today (`CAMPAIGN_POC_AUTH_MODE=none`). claude.ai connectors
> generally want OAuth — implement it in `auth.authenticate()` and set the mode to `oauth`.

## Tools (what Claude calls)

`upload_campaign` · `add_metrics` · `list_campaigns` · `get_campaign` ·
`find_similar_campaigns` · `prepare_evaluation` · `save_evaluation` ·
`reconcile_evaluation` · `save_reconciliation`

## Files

```
config.py        env-driven config
store.py         SQLite schema + CRUD (campaigns, metrics, evaluations, reconciliations)
vectorstore.py   sqlite-vec vector table (+ pure-Python cosine fallback)
embedding.py     ollama | voyage | hash embedders + cosine
extract.py       PDF / PPTX text extraction
core.py          ingest + semantic retrieval + evidence packaging (LLM-first)
mcp_server.py    MCPServer tools (the API Claude calls)
http_app.py      Streamable-HTTP app (/mcp) + /upload + /healthz
auth.py          no-op auth seam (OAuth goes here)
```

## Status

End-to-end verified against a live server with the real `sqlite-vec` backend: full
memory→judge→reconcile loop over MCP-over-HTTP, plus `/upload` → ingest-by-`asset_id` with
server-side PPTX/PDF extraction. Embeddings need Ollama running for real semantic quality
(the pipeline itself is proven with the offline `hash` provider).

Not built yet (deliberately): OAuth, multi-user isolation, cleanup/retention, a migration to
Postgres+pgvector (the `vectorstore` interface is the single swap point when you outgrow
single-node).
