"""Attention: decide whether a query needs memory at all.

The cheapest retrieval is the one you skip. Research (TARG, RAGRouter-Bench) shows a large
share of turns need no retrieval. The prototype uses simple rules with zero tokens and
microsecond latency; a trained classifier and multi-hop/global routes come later.
"""

from __future__ import annotations

import re

NONE = "none"        # answer without memory (0 tokens)
LOOKUP = "lookup"    # hybrid fact lookup (default path)

_SMALL_TALK = re.compile(
    r"^\s*(hi|hello|hey|yo|thanks|thank you|thx|ok|okay|k|cool|great|nice|awesome|bye|"
    r"good (morning|afternoon|evening|night)|lol|haha|sure|yes|no|yep|nope)[\s!.?]*$",
    re.I,
)


def route(query: str) -> str:
    if not query or not query.strip() or _SMALL_TALK.match(query):
        return NONE
    return LOOKUP
