"""Token counting.

The prototype uses the common ~4 characters per token approximation so it needs no
tokenizer download and no API call. Real billing differs per model; the benchmark
harness will use provider token counts instead.
"""


def approx_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0
