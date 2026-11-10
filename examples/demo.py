"""Engram demo: a short conversation, a consolidation ("sleep"), and recalls.

Free run (no API keys, rule-based stand-ins):
    ENGRAM_LLM=fake ENGRAM_EMBEDDER=hash python examples/demo.py

Real run (uses your Gemini key from .env):
    python examples/demo.py
"""

import os
import tempfile

from engram import Brain, Settings

db = os.path.join(tempfile.mkdtemp(), "demo.db")
brain = Brain(Settings(db_path=db))
T, U = "acme", "sneha"

conversation = [
    ("2026-01-10T09:00:00+00:00", "user", "Hi! I'm Sneha, I lead the payments team at Acme. I live in Delhi."),
    ("2026-01-10T09:01:00+00:00", "user", "I prefer short answers with code examples in Python."),
    ("2026-02-02T10:00:00+00:00", "user", "Our Q2 launch is code-named Bluebird and ships on 15 June."),
    ("2026-06-20T08:30:00+00:00", "user", "Big news: I moved to Bangalore last week."),
]
for when, speaker, text in conversation:
    r = brain.remember(T, U, text, speaker=speaker, occurred_at=when)
    print(f"remembered {len(r.facts)} fact(s) from: {text}")

report = brain.sleep(T, U)
print(f"\nsleep: {report.duplicates_retired} duplicate(s), {report.outdated_retired} outdated fact(s) retired")
print("profile:\n" + report.profile)

for q in ["Where does Sneha live now?", "When does Bluebird ship?", "thanks!"]:
    r = brain.recall(T, U, q, token_budget=300)
    print(f"\nQ: {q}\nroute={r.route}  context_tokens={r.context_tokens}  latency={r.latency_ms}ms")
    print(r.context or "(no memory needed)")
