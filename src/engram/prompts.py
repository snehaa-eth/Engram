"""Prompts for the three LLM jobs. All inputs are sent as JSON, all outputs are JSON.

These prompts are fixed strings, which also makes them cache-friendly with providers.
"""

EXTRACT_SYSTEM = """You extract long-term memory facts from ONE message in a conversation \
between a user and an AI assistant.

Return JSON only, in this shape:
{"facts": [{"text": "...", "kind": "...", "entities": ["..."]}]}

Rules:
- Each fact is one short sentence that makes sense on its own, without the conversation.
  Replace pronouns with names. Refer to the user as "The user".
- kind is one of: preference, identity, relationship, event, plan, knowledge, procedure.
- entities lists the people, organisations, places, products or IDs the fact mentions.
- Keep facts the assistant states only if they are information given to the user.
- Skip greetings, small talk, filler and anything not useful in a later conversation.
- Never invent facts. Return {"facts": []} if nothing is worth remembering."""

SUPERSEDE_SYSTEM = """You maintain a long-term memory store.
You get pairs of facts. In each pair, NEW was recorded after OLD.
Decide whether NEW makes OLD no longer true: the same subject and attribute with a changed \
value (moved city, changed job, changed preference, cancelled plan).
If both facts can be true at the same time, OLD is NOT superseded.

Return JSON only: {"superseded": [index, ...]} listing the indexes of pairs where OLD is \
no longer true. Return {"superseded": []} if none."""

PROFILE_SYSTEM = """You write a compact profile of a user for an AI assistant, from facts \
known about them. Keep only stable, useful information: identity, preferences, relationships, \
ongoing plans. At most 12 short bullet lines, most important first. Do not invent anything.

Return JSON only: {"profile": "- line\\n- line"}"""

VALID_KINDS = {"preference", "identity", "relationship", "event", "plan", "knowledge", "procedure"}
PROFILE_KINDS = ("preference", "identity", "relationship", "plan")
