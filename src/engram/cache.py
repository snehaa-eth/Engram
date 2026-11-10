"""Record-and-replay cache for LLM and embedding calls.

Every call is keyed by a hash of (provider, task, input). The first time, the real API
is called and the answer is saved; afterwards the saved answer is returned for free.
This keeps development and benchmark re-runs cheap.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading


class DiskCache:
    def __init__(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self._conn = sqlite3.connect(os.path.join(directory, "cache.db"), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value BLOB NOT NULL)")
        self._lock = threading.Lock()

    @staticmethod
    def key(*parts: str) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: bytes) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO kv(key, value) VALUES(?,?)", (key, value))
            self._conn.commit()
