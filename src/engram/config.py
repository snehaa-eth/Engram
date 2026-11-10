"""Settings, read from environment variables (or a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    # Where memories are stored (SQLite file for the prototype).
    db_path: str = field(default_factory=lambda: _env("ENGRAM_DB", "engram.db"))

    # Which LLM extracts facts and runs consolidation: "gemini", "claude" or "fake".
    # "fake" is a rule-based stand-in that costs nothing (used by tests).
    llm: str = field(default_factory=lambda: _env("ENGRAM_LLM", "gemini"))

    # Which embedding model turns text into vectors: "gemini" or "hash".
    # "hash" is a free bag-of-words embedder for tests and offline demos.
    embedder: str = field(default_factory=lambda: _env("ENGRAM_EMBEDDER", "gemini"))

    gemini_model: str = field(default_factory=lambda: _env("ENGRAM_GEMINI_MODEL", "gemini-2.5-flash"))
    gemini_embed_model: str = field(
        default_factory=lambda: _env("ENGRAM_GEMINI_EMBED_MODEL", "gemini-embedding-001")
    )
    # Fact extraction is a bulk, high-volume job, so the cheap current Claude model is the default.
    claude_model: str = field(default_factory=lambda: _env("ENGRAM_CLAUDE_MODEL", "claude-haiku-5-5"))
    embed_dim: int = field(default_factory=lambda: int(_env("ENGRAM_EMBED_DIM", "768")))

    # Record-and-replay cache: identical LLM/embedding calls are answered from disk,
    # so re-running tests or demos does not spend API credits twice.
    cache_dir: str = field(default_factory=lambda: _env("ENGRAM_CACHE_DIR", ".engram_cache"))
    use_cache: bool = field(default_factory=lambda: _env("ENGRAM_CACHE", "1") == "1")

    # Default size of the context packet returned by recall().
    token_budget: int = field(default_factory=lambda: int(_env("ENGRAM_TOKEN_BUDGET", "600")))

    # Optional API keys for the HTTP server, as "key1:tenantA,key2:tenantB".
    api_keys: str = field(default_factory=lambda: _env("ENGRAM_API_KEYS", ""))

    def api_key_map(self) -> dict[str, str]:
        pairs = [p.split(":", 1) for p in self.api_keys.split(",") if ":" in p]
        return {k.strip(): t.strip() for k, t in pairs}
