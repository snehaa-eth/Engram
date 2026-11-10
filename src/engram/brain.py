"""The Brain: one object that ties every memory layer together.

    remember()  write path: log the episode -> extract facts (add-only) -> index
    recall()    read path:  route -> hybrid search -> MMR pack into a token budget
    sleep()     consolidation: retire duplicates and outdated facts, rebuild the profile
    forget_*()  true erasure, following each fact back to its source episode

In the prototype the write path runs inline. In production it runs in background
workers, so storing a memory never slows down a query.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from engram import prompts
from engram.cache import DiskCache
from engram.config import Settings
from engram.embed import Embedder, build_embedder
from engram.llm import LLM, build_llm
from engram.retrieve import hybrid_search, render_fact, select_mmr
from engram.router import NONE, route
from engram.store import SHARED_USER, Fact, Store, now_iso
from engram.tokens import approx_tokens

# Trust tiers ("immune system"). Content from outside sources (web pages, emails, tool
# output) may add facts, but can never retire a fact the user stated themselves.
# This blocks memory-poisoning attacks such as MINJA and SpAIware.
TRUST_BY_SPEAKER = {"user": "user", "assistant": "assistant"}
TRUST_RANK = {"external": 0, "assistant": 1, "user": 2}


@dataclass
class RememberResult:
    episode_id: str
    facts: list[dict]


@dataclass
class RecallResult:
    route: str
    context: str                 # the packet to put in the prompt (may be empty)
    profile: str                 # stable profile; put it in the cached prompt prefix
    facts: list[dict] = field(default_factory=list)
    context_tokens: int = 0
    profile_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class SleepReport:
    duplicates_retired: int
    outdated_retired: int
    profile: str


class Brain:
    def __init__(self, settings: Settings | None = None, *, llm: LLM | None = None,
                 embedder: Embedder | None = None, store: Store | None = None):
        self.settings = settings or Settings()
        s = self.settings
        cache = DiskCache(s.cache_dir) if s.use_cache else None
        self.llm = llm or build_llm(s.llm, s.gemini_model, s.claude_model, cache)
        self.embedder = embedder or build_embedder(s.embedder, s.gemini_embed_model, s.embed_dim, cache)
        self.store = store or Store(s.db_path)

    # ------------------------------------------------------------------ write path

    def remember(self, tenant: str, user: str, content: str, *, speaker: str = "user",
                 source: str = "chat", trust: str | None = None,
                 occurred_at: str | None = None) -> RememberResult:
        trust = trust or TRUST_BY_SPEAKER.get(speaker, "external")
        if trust not in TRUST_RANK:
            raise ValueError(f"trust must be one of {sorted(TRUST_RANK)}")
        occurred_at = occurred_at or now_iso()

        # 1. Sensory log: keep the raw message forever (until the user asks to erase it).
        episode_id = self.store.add_episode(tenant, user, speaker, source, trust, content, occurred_at)

        # 2. Extract small, self-contained facts. Add-only: nothing is edited here;
        #    reconciling contradictions is the job of sleep().
        payload = json.dumps({"speaker": speaker, "timestamp": occurred_at, "message": content})
        raw = self.llm.complete_json("extract", prompts.EXTRACT_SYSTEM, payload).get("facts", [])
        facts = [f for f in raw if isinstance(f, dict) and str(f.get("text", "")).strip()]
        if not facts:
            return RememberResult(episode_id, [])

        # 3. Index: embed all facts in one batch, then store vector + BM25 + entity links.
        vectors = self.embedder.embed([f["text"].strip() for f in facts], "document")
        stored = []
        for f, vec in zip(facts, vectors):
            kind = f.get("kind") if f.get("kind") in prompts.VALID_KINDS else "knowledge"
            entities = [str(e) for e in f.get("entities", []) if e]
            fid = self.store.add_fact(tenant, user, episode_id, f["text"].strip(), kind, speaker,
                                      trust, occurred_at, vec, entities)
            stored.append({"id": fid, "text": f["text"].strip(), "kind": kind, "entities": entities})
        return RememberResult(episode_id, stored)

    # ------------------------------------------------------------------ read path

    def recall(self, tenant: str, user: str, query: str, *, token_budget: int | None = None,
               include_shared: bool = True) -> RecallResult:
        t0 = time.perf_counter()
        budget = token_budget or self.settings.token_budget
        profile = self.store.get_profile(tenant, user)
        result = RecallResult(route=route(query), context="", profile=profile,
                              profile_tokens=approx_tokens(profile))
        if result.route != NONE:
            users = [user, SHARED_USER] if include_shared else [user]
            scored = hybrid_search(self.store, self.embedder, tenant, users, query)
            chosen = select_mmr(scored, budget)
            if chosen:
                result.context = "Relevant memory:\n" + "\n".join(render_fact(f) for f in chosen)
                result.facts = [f.as_dict() for f in chosen]
        result.context_tokens = approx_tokens(result.context)
        result.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return result

    # ------------------------------------------------------------------ consolidation

    def sleep(self, tenant: str, user: str, *, similarity: float = 0.75,
              duplicate: float = 0.97, batch: int = 25) -> SleepReport:
        facts = self.store.valid_facts(tenant, [user])
        retired: set[str] = set()
        dups = 0
        candidates: list[tuple[Fact, Fact]] = []

        if len(facts) > 1:
            sims = np.stack([f.embedding for f in facts]) @ np.stack([f.embedding for f in facts]).T
            for j in range(len(facts)):            # j = newer (facts are sorted by time)
                for i in range(j):                 # i = older
                    old, new = facts[i], facts[j]
                    if old.id in retired:
                        continue
                    same_text = old.text.strip().lower() == new.text.strip().lower()
                    if sims[i, j] >= duplicate or same_text:
                        self.store.invalidate(tenant, old.id, new.valid_from, new.id)
                        retired.add(old.id)
                        dups += 1
                    elif sims[i, j] >= similarity and TRUST_RANK[new.trust] >= TRUST_RANK[old.trust]:
                        candidates.append((old, new))

        outdated = 0
        candidates = [(o, n) for o, n in candidates if o.id not in retired]
        for start in range(0, len(candidates), batch):
            chunk = candidates[start : start + batch]
            pairs = [{"index": k, "old": o.text, "old_date": o.valid_from, "new": n.text,
                      "new_date": n.valid_from} for k, (o, n) in enumerate(chunk)]
            answer = self.llm.complete_json("supersede", prompts.SUPERSEDE_SYSTEM,
                                            json.dumps({"pairs": pairs}))
            for k in answer.get("superseded", []):
                if isinstance(k, int) and 0 <= k < len(chunk):
                    old, new = chunk[k]
                    if old.id not in retired:
                        self.store.invalidate(tenant, old.id, new.valid_from, new.id)
                        retired.add(old.id)
                        outdated += 1

        profile = self._rebuild_profile(tenant, user)
        return SleepReport(dups, outdated, profile)

    def _rebuild_profile(self, tenant: str, user: str) -> str:
        facts = [f for f in self.store.valid_facts(tenant, [user]) if f.kind in prompts.PROFILE_KINDS]
        if not facts:
            return self.store.get_profile(tenant, user)
        recent = [f.text for f in facts[-60:]]
        text = self.llm.complete_json("profile", prompts.PROFILE_SYSTEM,
                                      json.dumps({"facts": recent})).get("profile", "")
        if text:
            self.store.set_profile(tenant, user, text.strip())
        return text.strip()

    # ------------------------------------------------------------------ other

    def profile(self, tenant: str, user: str) -> str:
        return self.store.get_profile(tenant, user)

    def forget_episode(self, tenant: str, episode_id: str) -> int:
        return self.store.forget_episode(tenant, episode_id)

    def forget_user(self, tenant: str, user: str) -> int:
        return self.store.forget_user(tenant, user)

    def graph(self, tenant: str, user: str, *, neighbors: int = 4, min_link: float = 0.3) -> dict:
        """The brain as a network, for the visual map.

        Nodes: facts ("neurons") and the entities they mention ("concept hubs").
        Links: similar meaning (each fact to its closest neighbours), mentions
        (fact -> entity) and replaced (an outdated fact -> the fact that replaced it).
        """
        facts = self.store.all_facts(tenant, [user, SHARED_USER])
        ids = [f.id for f in facts]
        ents = self.store.entities_for(tenant, ids)
        eps = self.store.get_episodes(tenant, list({f.episode_id for f in facts}))

        nodes: list[dict] = []
        links: list[dict] = []
        for f in facts:
            nodes.append({**f.as_dict(), "type": "fact", "active": f.valid_to is None,
                          "superseded_by": f.superseded_by, "shared": f.user_id == SHARED_USER,
                          "entities": ents.get(f.id, []), "episode": eps.get(f.episode_id)})
            if f.superseded_by:
                links.append({"source": f.id, "target": f.superseded_by, "type": "replaced", "weight": 1.0})

        mentions: dict[str, int] = {}
        for fid, names in ents.items():
            for name in names:
                mentions[name] = mentions.get(name, 0) + 1
                links.append({"source": fid, "target": f"ent:{name}", "type": "mentions", "weight": 0.5})
        nodes += [{"id": f"ent:{n}", "type": "entity", "name": n, "count": c} for n, c in mentions.items()]

        # Meaning links between facts that are still true. Scores are rescaled so that
        # "unrelated" is 0 for every embedding model (same calibration as recall).
        active = [f for f in facts if f.valid_to is None]
        if len(active) > 1:
            floor = getattr(self.embedder, "floor", 0.0)
            m = np.stack([f.embedding for f in active])
            sims = np.clip((m @ m.T - floor) / (1.0 - floor), 0.0, 1.0)
            np.fill_diagonal(sims, 0.0)
            seen: set[tuple[int, int]] = set()
            for i in range(len(active)):
                for j in np.argsort(-sims[i])[:neighbors]:
                    j = int(j)
                    if sims[i, j] < min_link or (min(i, j), max(i, j)) in seen:
                        continue
                    seen.add((min(i, j), max(i, j)))
                    links.append({"source": active[i].id, "target": active[j].id,
                                  "type": "similar", "weight": round(float(sims[i, j]), 3)})

        return {"user_id": user, "nodes": nodes, "links": links,
                "profile": self.store.get_profile(tenant, user),
                "stats": {"neurons": len(active), "retired": len(facts) - len(active),
                          "concepts": len(mentions), "connections": len(links),
                          "messages": len(eps)}}

    def brains(self, tenant: str) -> list[dict]:
        return [b for b in self.store.list_users(tenant) if b["user_id"] != SHARED_USER]

    def stats(self, tenant: str) -> dict:
        return self.store.stats(tenant)
