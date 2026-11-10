"""LLM providers behind one small interface.

The brain only needs one capability from an LLM: "given instructions and an input,
return JSON". Keeping the interface this small is what makes Engram model-agnostic:
switch providers with the ENGRAM_LLM setting, no code changes.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

from engram.cache import DiskCache


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def complete_json(self, task: str, system: str, user: str) -> dict: ...


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating ```json fences or stray text."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise LLMError(f"model did not return JSON: {text[:200]!r}")
    return json.loads(text[start : end + 1])


class GeminiLLM:
    def __init__(self, model: str):
        from google import genai
        from google.genai import types

        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set (add it to .env)")
        self._client = genai.Client(api_key=key)
        self._types = types
        self.model = model
        self.name = f"gemini:{model}"

    def complete_json(self, task: str, system: str, user: str) -> dict:
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=self._types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
            ),
        )
        return parse_json(resp.text or "{}")


class ClaudeLLM:
    def __init__(self, model: str):
        import anthropic

        # Credentials come from ANTHROPIC_API_KEY (or an `ant auth login` profile).
        self._client = anthropic.Anthropic()
        self.model = model
        self.name = f"claude:{model}"

    def complete_json(self, task: str, system: str, user: str) -> dict:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=8000,
            system=system,
            # Extraction is simple; low effort keeps thinking (and cost) small.
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": user}],
        )
        if resp.stop_reason == "refusal":
            raise LLMError("model declined the request")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return parse_json(text)


class CachedLLM:
    """Wraps any LLM with the record-and-replay disk cache."""

    def __init__(self, inner: LLM, cache: DiskCache):
        self._inner = inner
        self._cache = cache
        self.name = inner.name

    def complete_json(self, task: str, system: str, user: str) -> dict:
        key = DiskCache.key("llm", self.name, task, system, user)
        hit = self._cache.get(key)
        if hit is not None:
            return json.loads(hit)
        result = self._inner.complete_json(task, system, user)
        self._cache.put(key, json.dumps(result).encode("utf-8"))
        return result


class FakeLLM:
    """A free, deterministic, rule-based stand-in for an LLM.

    It is NOT smart. It exists so the whole pipeline can be tested without API
    credits. Each task gets a crude heuristic.
    """

    name = "fake"

    def complete_json(self, task: str, system: str, user: str) -> dict:
        data = json.loads(user)
        if task == "extract":
            return {"facts": self._extract(data["message"])}
        if task == "supersede":
            out = []
            for p in data["pairs"]:
                a, b = p["old"].lower().split(), p["new"].lower().split()
                if a[:3] == b[:3] and a != b:
                    out.append(p["index"])
            return {"superseded": out}
        if task == "profile":
            return {"profile": "\n".join(f"- {f}" for f in data["facts"])}
        raise LLMError(f"FakeLLM has no rule for task {task!r}")

    @staticmethod
    def _extract(message: str) -> list[dict]:
        facts = []
        for sentence in re.split(r"(?<=[.!?])\s+", message.strip()):
            s = sentence.strip().rstrip(".!?")
            if len(s.split()) < 3:
                continue
            s = re.sub(r"\bI'm\b", "The user is", s)
            s = re.sub(r"\bI've\b", "The user has", s)
            s = re.sub(r"\bI\b", "The user", s)
            s = re.sub(r"\bmy\b", "the user's", s, flags=re.I)
            low = s.lower()
            kind = "preference" if ("prefer" in low or "like" in low) else (
                "identity" if "live" in low or "work" in low or "name" in low else "knowledge"
            )
            entities = [w.strip(",") for w in s.split()[1:] if w[:1].isupper()]
            facts.append({"text": s, "kind": kind, "entities": entities})
        return facts


def build_llm(provider: str, gemini_model: str, claude_model: str, cache: DiskCache | None) -> LLM:
    if provider == "fake":
        return FakeLLM()
    if provider == "gemini":
        llm: LLM = GeminiLLM(gemini_model)
    elif provider == "claude":
        llm = ClaudeLLM(claude_model)
    else:
        raise LLMError(f"unknown ENGRAM_LLM provider {provider!r} (use gemini, claude or fake)")
    return CachedLLM(llm, cache) if cache else llm
