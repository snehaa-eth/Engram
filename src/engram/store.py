"""Storage for every brain layer (SQLite for the prototype).

Layers and tables:
  episodes        - sensory/episodic log: every message, never edited (ground truth)
  facts           - semantic memory: small self-contained facts with validity times
  facts_fts       - BM25 keyword index over facts (SQLite FTS5)
  entities,
  fact_entities   - associative memory: which facts mention which people/things
  profiles        - self-model: a short, always-loaded summary per user

Every query is scoped by tenant_id (the company) and user_id. Shared company
knowledge uses user_id = "_shared". There is no code path that searches across tenants.

Production plan: the same schema moves to Postgres (pgvector + row-level security).
"""

from __future__ import annotations

import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

SHARED_USER = "_shared"

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes(
  id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  speaker TEXT NOT NULL,
  source TEXT NOT NULL,
  trust TEXT NOT NULL,
  content TEXT NOT NULL,
  occurred_at TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_episodes_scope ON episodes(tenant_id, user_id, occurred_at);

CREATE TABLE IF NOT EXISTS facts(
  id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  episode_id TEXT NOT NULL,
  text TEXT NOT NULL,
  kind TEXT NOT NULL,
  speaker TEXT NOT NULL,
  trust TEXT NOT NULL,
  valid_from TEXT NOT NULL,
  valid_to TEXT,
  superseded_by TEXT,
  created_at TEXT NOT NULL,
  embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_facts_scope ON facts(tenant_id, user_id, valid_to);
CREATE INDEX IF NOT EXISTS ix_facts_episode ON facts(episode_id);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
  fact_id UNINDEXED, tenant_id UNINDEXED, user_id UNINDEXED, body,
  tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS entities(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tenant_id TEXT NOT NULL,
  name TEXT NOT NULL,
  UNIQUE(tenant_id, name)
);
CREATE TABLE IF NOT EXISTS fact_entities(
  fact_id TEXT NOT NULL,
  entity_id INTEGER NOT NULL,
  PRIMARY KEY(fact_id, entity_id)
);

CREATE TABLE IF NOT EXISTS profiles(
  tenant_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  text TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(tenant_id, user_id)
);
"""

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "what", "which", "who",
    "where", "when", "how", "why", "of", "to", "in", "on", "for", "and", "or", "i", "me", "my",
    "you", "your", "it", "this", "that", "be", "about", "with", "at", "by", "from", "can",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass
class Fact:
    id: str
    tenant_id: str
    user_id: str
    episode_id: str
    text: str
    kind: str
    speaker: str
    trust: str
    valid_from: str
    valid_to: str | None
    superseded_by: str | None
    embedding: np.ndarray

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "kind": self.kind,
            "speaker": self.speaker,
            "trust": self.trust,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "episode_id": self.episode_id,
        }


_FACT_COLS = (
    "id, tenant_id, user_id, episode_id, text, kind, speaker, trust, "
    "valid_from, valid_to, superseded_by, embedding"
)


def _row_to_fact(r: tuple) -> Fact:
    return Fact(*r[:11], embedding=np.frombuffer(r[11], dtype=np.float32))


class Store:
    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._lock = threading.RLock()

    # ---------- writes ----------

    def add_episode(self, tenant: str, user: str, speaker: str, source: str, trust: str,
                    content: str, occurred_at: str) -> str:
        eid = new_id("ep")
        with self._lock:
            self._conn.execute(
                "INSERT INTO episodes VALUES(?,?,?,?,?,?,?,?,?)",
                (eid, tenant, user, speaker, source, trust, content, occurred_at, now_iso()),
            )
            self._conn.commit()
        return eid

    def add_fact(self, tenant: str, user: str, episode_id: str, text: str, kind: str,
                 speaker: str, trust: str, valid_from: str, embedding: np.ndarray,
                 entities: list[str]) -> str:
        fid = new_id("f")
        with self._lock:
            self._conn.execute(
                "INSERT INTO facts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (fid, tenant, user, episode_id, text, kind, speaker, trust, valid_from,
                 None, None, now_iso(), embedding.astype(np.float32).tobytes()),
            )
            self._conn.execute(
                "INSERT INTO facts_fts(fact_id, tenant_id, user_id, body) VALUES(?,?,?,?)",
                (fid, tenant, user, text),
            )
            for name in {e.strip() for e in entities if e and e.strip()}:
                self._conn.execute(
                    "INSERT OR IGNORE INTO entities(tenant_id, name) VALUES(?,?)", (tenant, name)
                )
                (ent_id,) = self._conn.execute(
                    "SELECT id FROM entities WHERE tenant_id=? AND name=?", (tenant, name)
                ).fetchone()
                self._conn.execute("INSERT OR IGNORE INTO fact_entities VALUES(?,?)", (fid, ent_id))
            self._conn.commit()
        return fid

    def invalidate(self, tenant: str, fact_id: str, valid_to: str, superseded_by: str) -> None:
        """Retire a fact without deleting it: history is kept, but it is never recalled again."""
        with self._lock:
            self._conn.execute(
                "UPDATE facts SET valid_to=?, superseded_by=? "
                "WHERE id=? AND tenant_id=? AND valid_to IS NULL",
                (valid_to, superseded_by, fact_id, tenant),
            )
            self._conn.commit()

    def set_profile(self, tenant: str, user: str, text: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO profiles VALUES(?,?,?,?)", (tenant, user, text, now_iso())
            )
            self._conn.commit()

    # ---------- true deletion (right to be forgotten) ----------

    def _delete_facts(self, where: str, args: tuple) -> int:
        ids = [r[0] for r in self._conn.execute(f"SELECT id FROM facts WHERE {where}", args)]
        for fid in ids:
            self._conn.execute("DELETE FROM facts_fts WHERE fact_id=?", (fid,))
            self._conn.execute("DELETE FROM fact_entities WHERE fact_id=?", (fid,))
            self._conn.execute("DELETE FROM facts WHERE id=?", (fid,))
        return len(ids)

    def forget_episode(self, tenant: str, episode_id: str) -> int:
        """Delete an episode and every fact derived from it (lineage-based erasure)."""
        with self._lock:
            n = self._delete_facts("tenant_id=? AND episode_id=?", (tenant, episode_id))
            self._conn.execute("DELETE FROM episodes WHERE tenant_id=? AND id=?", (tenant, episode_id))
            self._conn.commit()
        return n

    def forget_user(self, tenant: str, user: str) -> int:
        with self._lock:
            n = self._delete_facts("tenant_id=? AND user_id=?", (tenant, user))
            self._conn.execute("DELETE FROM episodes WHERE tenant_id=? AND user_id=?", (tenant, user))
            self._conn.execute("DELETE FROM profiles WHERE tenant_id=? AND user_id=?", (tenant, user))
            self._conn.commit()
        return n

    # ---------- reads ----------

    def valid_facts(self, tenant: str, users: list[str]) -> list[Fact]:
        marks = ",".join("?" * len(users))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_FACT_COLS} FROM facts WHERE tenant_id=? AND user_id IN ({marks}) "
                "AND valid_to IS NULL ORDER BY valid_from, created_at",
                (tenant, *users),
            ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def get_facts(self, tenant: str, ids: list[str]) -> dict[str, Fact]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_FACT_COLS} FROM facts WHERE tenant_id=? AND id IN ({marks})", (tenant, *ids)
            ).fetchall()
        return {r[0]: _row_to_fact(r) for r in rows}

    def bm25_search(self, tenant: str, users: list[str], query: str, k: int) -> list[tuple[str, float]]:
        """Keyword search. Returns (fact_id, score) with higher = better."""
        words = [w for w in re.findall(r"\w+", query.lower())
                 if len(w) > 1 and w not in _STOPWORDS]
        if not words:
            return []
        match = " OR ".join(f'"{w}"' for w in dict.fromkeys(words))
        marks = ",".join("?" * len(users))
        with self._lock:
            rows = self._conn.execute(
                "SELECT facts_fts.fact_id, bm25(facts_fts) AS s FROM facts_fts "
                "JOIN facts ON facts.id = facts_fts.fact_id "
                f"WHERE facts_fts MATCH ? AND facts_fts.tenant_id=? AND facts_fts.user_id IN ({marks}) "
                "AND facts.valid_to IS NULL ORDER BY s LIMIT ?",
                (match, tenant, *users, k),
            ).fetchall()
        # SQLite's bm25() is "lower is better" (negative numbers); flip the sign.
        return [(fid, -s) for fid, s in rows]

    def get_profile(self, tenant: str, user: str) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT text FROM profiles WHERE tenant_id=? AND user_id=?", (tenant, user)
            ).fetchone()
        return row[0] if row else ""

    # ---------- visualisation reads (the brain map) ----------

    def all_facts(self, tenant: str, users: list[str]) -> list[Fact]:
        """Every fact, including retired ones (the map shows history too)."""
        marks = ",".join("?" * len(users))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_FACT_COLS} FROM facts WHERE tenant_id=? AND user_id IN ({marks}) "
                "ORDER BY valid_from, created_at",
                (tenant, *users),
            ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def entities_for(self, tenant: str, fact_ids: list[str]) -> dict[str, list[str]]:
        if not fact_ids:
            return {}
        marks = ",".join("?" * len(fact_ids))
        out: dict[str, list[str]] = {}
        with self._lock:
            rows = self._conn.execute(
                "SELECT fe.fact_id, e.name FROM fact_entities fe JOIN entities e ON e.id = fe.entity_id "
                f"WHERE e.tenant_id=? AND fe.fact_id IN ({marks})",
                (tenant, *fact_ids),
            ).fetchall()
        for fid, name in rows:
            out.setdefault(fid, []).append(name)
        return out

    def get_episodes(self, tenant: str, ids: list[str]) -> dict[str, dict]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, speaker, source, trust, content, occurred_at FROM episodes "
                f"WHERE tenant_id=? AND id IN ({marks})",
                (tenant, *ids),
            ).fetchall()
        keys = ("id", "speaker", "source", "trust", "content", "occurred_at")
        return {r[0]: dict(zip(keys, r)) for r in rows}

    def list_users(self, tenant: str) -> list[dict]:
        """One row per user ("brain") in the tenant, most recently active first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT e.user_id, COUNT(DISTINCT e.id), MAX(e.occurred_at), "
                "(SELECT COUNT(*) FROM facts f WHERE f.tenant_id=e.tenant_id AND f.user_id=e.user_id "
                " AND f.valid_to IS NULL) "
                "FROM episodes e WHERE e.tenant_id=? GROUP BY e.user_id ORDER BY MAX(e.occurred_at) DESC",
                (tenant,),
            ).fetchall()
        return [{"user_id": u, "messages": m, "last_active": t, "facts": f} for u, m, t, f in rows]

    def stats(self, tenant: str) -> dict:
        with self._lock:
            ep = self._conn.execute("SELECT COUNT(*) FROM episodes WHERE tenant_id=?", (tenant,)).fetchone()[0]
            valid = self._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE tenant_id=? AND valid_to IS NULL", (tenant,)
            ).fetchone()[0]
            retired = self._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE tenant_id=? AND valid_to IS NOT NULL", (tenant,)
            ).fetchone()[0]
        return {"episodes": ep, "valid_facts": valid, "retired_facts": retired}
