# Engram

**Long-term memory for AI agents: fast, low-token, model-agnostic.**

AI agents forget everything between sessions. Teams either resend the whole conversation
history on every call, which is slow and expensive and makes answers worse as context grows,
or add memory tools that return noisy, outdated context. Engram is a memory layer any agent
can plug into. It stores what agents learn about users and companies, and returns only the
facts each query needs, inside a token budget you choose.

> **Status: prototype.** The core memory pipeline works and is tested. Nothing has been
> benchmarked yet; every performance figure below is either cited research or a design target.

## How it works: memory in layers, like a brain

| Brain function | In Engram | Status |
|---|---|---|
| Sensory log | Every message stored unchanged as an *episode* (the ground truth) | ✅ |
| Semantic memory | Small, self-contained *facts* with a valid-from date, extracted add-only | ✅ |
| Associative memory | Entity links (people, projects, IDs) between facts | ✅ stored; graph search planned |
| Self-model | A short user *profile*, rebuilt during sleep, meant for the cached prompt prefix | ✅ |
| Attention | A router that skips retrieval when a message needs no memory | ✅ rules; learned router planned |
| Sleep | Consolidation: retire duplicates and facts that are no longer true (kept as history, never recalled) | ✅ |
| Immune system | Trust tiers: external content can add facts but can never overwrite what the user said | ✅ |
| Forgetting | True erasure that follows each fact back to its source episode | ✅ |
| Working memory | Per-session buffer | planned |
| Procedural memory | Learned playbooks ("for this customer, always do X") | planned |

### Read path (what happens on `recall`)

```
query → router ──"none"──► empty packet (0 tokens)
          │
          └─"lookup"─► tenant + user filter
                       → dense search (cosine) + BM25 keyword search
                       → weighted fusion
                       → MMR selection until the token budget is full
                       → compact packet:  - [2026-06-20] The user lives in Bangalore (source: user)
```

No LLM is called on the read path, so it is fast and costs no tokens. The packet may be
empty: sending nothing beats sending noise.

### Write path (what happens on `remember` and `sleep`)

```
message → episode log (immutable) → LLM extracts facts (add-only) → embed → vector + BM25 + entities
sleep   → find similar fact pairs → retire duplicates → LLM decides which old facts are outdated
        → retire them with an end date (history kept) → rebuild the profile
```

## Quick start

```bash
git clone https://github.com/snehaa-eth/Engram.git && cd Engram
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Free run, no API keys (rule-based stand-ins for the LLM and embeddings)
pytest -q
ENGRAM_LLM=fake ENGRAM_EMBEDDER=hash python examples/demo.py

# Real run: copy .env.example to .env and add GEMINI_API_KEY (or ANTHROPIC_API_KEY with ENGRAM_LLM=claude)
cp .env.example .env
python examples/demo.py
```

### Command line

```bash
engram --user sneha remember "I moved to Bangalore last week."
engram --user sneha recall "Where do I live?" --budget 300
engram --user sneha sleep
engram --user sneha profile
```

### HTTP API

```bash
engram serve   # http://127.0.0.1:8000/docs for interactive docs

curl -X POST localhost:8000/v1/remember -H "X-Tenant-Id: acme" -H "Content-Type: application/json" \
  -d '{"user_id": "sneha", "content": "I prefer short answers with Python examples."}'

curl -X POST localhost:8000/v1/recall -H "X-Tenant-Id: acme" -H "Content-Type: application/json" \
  -d '{"user_id": "sneha", "query": "how should I format answers?", "token_budget": 300}'
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/remember` | Store a message (`speaker`: user / assistant / tool / document; `trust`: user / assistant / external) |
| `POST /v1/recall` | Get the memory packet and profile for a query |
| `POST /v1/sleep` | Run consolidation for a user |
| `GET /v1/profile/{user_id}` | Read the profile |
| `DELETE /v1/episodes/{id}` | Erase one message and every fact derived from it |
| `DELETE /v1/users/{user_id}` | Erase everything about a user |

**Multi-tenancy:** set `ENGRAM_API_KEYS="keyA:acme,keyB:globex"` and the tenant comes from the
`X-API-Key` header, so a client can never read another company's data. Shared company
knowledge is stored under `user_id = "_shared"` and is visible to every user in that tenant.

## Model-agnostic

| Setting | Options |
|---|---|
| `ENGRAM_LLM` | `gemini` (default), `claude`, `fake` |
| `ENGRAM_EMBEDDER` | `gemini` (default), `hash` |

LLM and embedding calls go through a record-and-replay cache (`.engram_cache/`), so re-running
the same tests or demos does not spend API credits twice.

## Why this design

Each choice follows published 2025–2026 results (full write-ups with sources in [`reports/`](reports/)):

- **Atomic facts instead of chunks.** Mem0 cut context from about 26k to about 1.8k tokens per query this way.
- **Add-only writes plus offline consolidation.** Mem0 dropped in-line ADD/UPDATE/DELETE in 2026; LightMem reports 20–38x fewer tokens with an offline "sleep" step.
- **Retire facts with dates instead of deleting them.** Zep/Graphiti's approach; history stays auditable.
- **Hybrid BM25 + dense search.** Keyword search catches exact names and IDs that embeddings miss.
- **Skip retrieval when it is not needed.** Gating studies (TARG, RAGRouter-Bench) show many turns need no memory.
- **Small context beats big context.** Chroma's context-rot study found focused ~300-token prompts beat ~113k-token prompts.
- **Trust tiers.** Memory-poisoning attacks (MINJA, SpAIware) work by getting untrusted content into memory.

## Design targets (not yet measured)

| | Tokens per query | Brain-side p95 latency |
|---|---|---|
| Engram target | ~700–1,300 (incl. cached profile) | < 250 ms |
| Mem0 (published) | ~1,764 | ~200 ms search |
| Full history | 26k–115k | — |

## Roadmap

1. **Prototype** (this repo): layers, hybrid search, sleep, trust tiers, erasure, API, tests.
2. **Quality:** Qwen3-Embedding and Qwen3-Reranker, contextual retrieval, learned router, entity-graph search for multi-hop questions, MCP server.
3. **Proof:** an open benchmark (LongMemEval, BEAM, MemoryAgentBench) with audited judges, reporting accuracy, tokens, latency and cost against Mem0, Zep, Letta and full context.
4. **Production:** Postgres + pgvector with row-level security, background workers, per-tenant encryption, observability, SDKs, billing.

## Project layout

```
src/engram/
  brain.py      remember / recall / sleep / forget: ties the layers together
  store.py      storage for every layer (SQLite + FTS5 in the prototype)
  retrieve.py   hybrid search, fusion, MMR packing
  router.py     decides whether memory is needed
  llm.py        Gemini / Claude / fake providers + replay cache
  embed.py      Gemini / hash embedders + replay cache
  prompts.py    extraction, supersede and profile prompts
  api.py        FastAPI server
  cli.py        command line
tests/          free tests (fake LLM, hash embeddings)
reports/        research behind the design
```
