"""
src/utils/token_counter.py

Utility for counting tokens accurately using tiktoken, instead of
approximating from character count. This matters because the whole
point of TORQ is managing a TOKEN budget precisely — an approximation
that's off by 20% defeats the purpose of the budget system.

Used by:
- budget_check_node.py (to check if retrieved_chunks exceed token_budget)
- compression_node.py (to greedily select chunks until budget is filled)
- metrics_tracker.py (to log actual token usage per query)
"""

import tiktoken

# cl100k_base is the encoding used by GPT-4/GPT-3.5-turbo and is a
# reasonable general-purpose approximation even if your reasoning LLM
# is a different model (e.g. Gemini) — exact tokenizers differ slightly
# across providers, but cl100k_base gives a consistent, defensible
# reference point for budgeting decisions across your whole pipeline.
ENCODING_NAME = "cl100k_base"

_encoding = None


def _get_encoding():
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding(ENCODING_NAME)
    return _encoding


def count_tokens(text: str) -> int:
    """Returns the exact token count for a single string."""
    encoding = _get_encoding()
    return len(encoding.encode(text))


def count_tokens_for_chunks(chunks: list, text_key: str = "text") -> int:
    """
    Returns the total token count across a list of chunk dicts.
    Used by budget_check_node to see if retrieved_chunks together
    exceed the budget.
    """
    encoding = _get_encoding()
    total = 0
    for chunk in chunks:
        total += len(encoding.encode(chunk[text_key]))
    return total


def count_tokens_per_chunk(chunks: list, text_key: str = "text") -> list:
    """
    Returns per-chunk token counts (same order as input), rather than
    a single total. Used by compression_node when greedily selecting
    chunks up to the budget — it needs to know each chunk's individual
    cost, not just the aggregate.
    """
    encoding = _get_encoding()
    return [len(encoding.encode(chunk[text_key])) for chunk in chunks]


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """
    Hard-truncates a string to at most max_tokens tokens.
    Used as a last-resort safety net in compression_node if even the
    LLM-summarized output somehow still exceeds budget — better to
    cut cleanly at a token boundary than to send an oversized request
    that errors out at the reasoning LLM.
    """
    encoding = _get_encoding()
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)