"""Embedders: turn text into unit-length vectors so similar meanings are close together.

All vectors are L2-normalised, so cosine similarity is just a dot product.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol

import numpy as np

from engram.cache import DiskCache

_WORD = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        """kind is "document" (when storing) or "query" (when searching)."""
        ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class HashEmbedder:
    """Free bag-of-words embedder (the "hashing trick"). No model, no API.

    It only matches shared words, not meaning ("car" will not match "vehicle"),
    so it is for tests and offline demos, not production.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash:{dim}"

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in _WORD.findall(t.lower()):
                h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalise(out)


class GeminiEmbedder:
    def __init__(self, model: str, dim: int):
        from google import genai
        from google.genai import types

        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set (add it to .env)")
        self._client = genai.Client(api_key=key)
        self._types = types
        self.model = model
        self.dim = dim
        self.name = f"gemini:{model}:{dim}"

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        rows: list[list[float]] = []
        for start in range(0, len(texts), 100):  # API batch limit
            resp = self._client.models.embed_content(
                model=self.model,
                contents=texts[start : start + 100],
                config=self._types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim),
            )
            rows.extend(e.values for e in resp.embeddings)
        # Truncated (Matryoshka) Gemini vectors must be re-normalised.
        return _normalise(np.asarray(rows, dtype=np.float32))


class CachedEmbedder:
    def __init__(self, inner: Embedder, cache: DiskCache):
        self._inner = inner
        self._cache = cache
        self.name = inner.name
        self.dim = inner.dim

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        keys = [DiskCache.key("emb", self.name, kind, t) for t in texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        missing = []
        for i, k in enumerate(keys):
            hit = self._cache.get(k)
            if hit is None:
                missing.append(i)
            else:
                out[i] = np.frombuffer(hit, dtype=np.float32)
        if missing:
            fresh = self._inner.embed([texts[i] for i in missing], kind)
            for row, i in zip(fresh, missing):
                out[i] = row
                self._cache.put(keys[i], row.astype(np.float32).tobytes())
        return out


def build_embedder(provider: str, model: str, dim: int, cache: DiskCache | None) -> Embedder:
    if provider == "hash":
        return HashEmbedder()
    if provider == "gemini":
        emb: Embedder = GeminiEmbedder(model, dim)
        return CachedEmbedder(emb, cache) if cache else emb
    raise ValueError(f"unknown ENGRAM_EMBEDDER {provider!r} (use gemini or hash)")
