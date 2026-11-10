"""Retrieval: find the right facts and pack them into a small token budget.

Pipeline:
  1. Dense search   - exact cosine similarity over the tenant's valid facts.
                      (Exact search is fast enough below ~1M facts per tenant; an ANN index
                      such as HNSW replaces it at larger scale.)
  2. Sparse search  - BM25 keyword search (catches exact names, IDs, codes).
  3. Fusion         - weighted blend of the two scores (research: a tuned convex blend beats
                      plain Reciprocal Rank Fusion; dense weight ~0.6-0.8).
  4. MMR selection  - pick facts one by one, rewarding relevance and penalising repeats,
                      until the token budget is full. The result may be empty: sending
                      nothing beats sending noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engram.embed import Embedder
from engram.store import Fact, Store
from engram.tokens import approx_tokens


@dataclass
class Scored:
    fact: Fact
    score: float


def hybrid_search(store: Store, embedder: Embedder, tenant: str, users: list[str], query: str,
                  k: int = 40, dense_weight: float = 0.7) -> list[Scored]:
    facts = store.valid_facts(tenant, users)
    if not facts:
        return []
    by_id = {f.id: i for i, f in enumerate(facts)}
    matrix = np.stack([f.embedding for f in facts])
    qvec = embedder.embed([query], "query")[0]
    dense = matrix @ qvec  # cosine similarity, because all vectors are unit length

    candidates = set(np.argsort(-dense)[:k].tolist())
    sparse_raw = store.bm25_search(tenant, users, query, k)
    sparse: dict[int, float] = {}
    if sparse_raw:
        top = max(s for _, s in sparse_raw) or 1.0
        for fid, s in sparse_raw:
            if fid in by_id:
                sparse[by_id[fid]] = s / top  # scale to 0..1
                candidates.add(by_id[fid])

    out = []
    for i in candidates:
        score = dense_weight * max(float(dense[i]), 0.0) + (1 - dense_weight) * sparse.get(i, 0.0)
        out.append(Scored(facts[i], score))
    out.sort(key=lambda s: s.score, reverse=True)
    return out


def render_fact(f: Fact) -> str:
    """One compact line per fact. Dates and source let the model judge freshness and trust."""
    return f"- [{f.valid_from[:10]}] {f.text} (source: {f.trust})"


def select_mmr(scored: list[Scored], token_budget: int, min_score: float = 0.15,
               relative_floor: float = 0.4, lam: float = 0.7, dup_sim: float = 0.95) -> list[Fact]:
    if not scored:
        return []
    best = scored[0].score
    pool = [s for s in scored if s.score >= max(min_score, best * relative_floor)]
    chosen: list[Fact] = []
    used = 0
    while pool:
        def mmr(s: Scored) -> float:
            if not chosen:
                return s.score
            redundancy = max(float(s.fact.embedding @ c.embedding) for c in chosen)
            return lam * s.score - (1 - lam) * redundancy

        pick = max(pool, key=mmr)
        pool.remove(pick)
        if chosen and max(float(pick.fact.embedding @ c.embedding) for c in chosen) >= dup_sim:
            continue  # near-duplicate: costs tokens, adds nothing
        cost = approx_tokens(render_fact(pick.fact)) + 1
        if used + cost > token_budget:
            continue  # try smaller facts that may still fit
        chosen.append(pick.fact)
        used += cost
    return chosen
