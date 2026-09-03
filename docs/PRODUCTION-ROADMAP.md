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

### 6.1 Chunk `deck_text` server-side  *(highest leverage — one change fixes three problems)*
Split each deck by slide/section, embed **each chunk**, store multiple vectors per campaign,
search chunks and roll up to campaigns. Fixes: (a) the **silent embedding failure** (long deck →
Ollama 500 → record saved but `embedded:false` hidden in warnings → unsearchable orphan; current
ceiling ~6.8k–9k chars), (b) the size limit, (c) delivers the **per-slide granularity** asked for.

### 6.2 Filter **before** similarity  *(fixes "can't discriminate")*
On a single-brand corpus, pure vector search returns noise (observed band 0.67–0.84, everything
"similar"). Add structured fields and filter first, then rank: `market`/`region`, `channel`,
`campaign_type`, `brand_stage`, `tags`, `status`. Query becomes "concluded seeding events in APAC"
→ then similarity.

### 6.3 Schema: split `kind`, add tags & region
`kind` (concluded|proposal) is too binary — doesn't fit reference docs, stubs, finished campaigns
without results. Split into **`status`** + **`record_type`**, add a **`tags[]`** array (the
four-category taxonomy currently lives in prose), and **`region`/`market`** as first-class.

### 6.4 Lifecycle: CRUD + supersede
`delete_campaign`, `update_campaign`, and a **`supersedes`** field. Today every mistake is
permanent (the Mexico deck exists twice, dead + live, with nothing saying one replaces the other).

### 6.5 Metrics as first-class + bulk import
Distinguish **prediction vs actual**; `list_campaigns` should show whether a record has
metrics/evaluations; add a **bulk metrics/CSV import** so loading the KPI workbook is one step.
This is what makes `reconcile_evaluation` actually functional.

### 6.6 Image vectorization + creative-reuse detection  *(the SVP question)*
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

### 6.7 Extensible asset pipeline (audio/video pluggable later)
Generic `assets` (modality) + `asset_fingerprints` (pHash) + `asset_vectors` (CLIP), keyed by
asset. **Video = keyframes → the image pipeline** (reused clips share keyframes — nearly free once
images work). **Audio later** = audio fingerprint + embedding, same shape.

### 6.8 Minor
Trim `find_similar` payload (summary by default, full detail on request); fix `&` stored as
`&amp;`; **align the port** (binary/config default 8080 vs `run.ps1` default 8086 — make them match).

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

## 9. Sequence

1. **Now:** local Windows end-to-end green (Desktop + server + Ollama), campaigns loaded.
2. **v0.2:** the §6 features (chunking, filters, schema, CRUD, metrics, image pHash + CLIP) — data reloaded fresh.
3. **Central deploy:** Postgres, Docker on the VM, TLS (Front Door / App Gateway), **Entra OAuth**
   via the pluggable provider, access scoping via group claims.
