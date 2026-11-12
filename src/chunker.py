"""Split plain text into translation-sized chunks.

Splits text on paragraph boundaries (blank lines) and greedily batches
consecutive paragraphs up to a token budget. Paragraphs are never split across
chunks, so every chunk ends on a semantic boundary (avoiding mid-sentence
fragments that translate poorly). This module is source-agnostic — any
format-specific preprocessing (e.g. stripping Gutenberg boilerplate) happens
before the text reaches here.
"""

from __future__ import annotations

import re

__all__ = ["estimate_tokens", "chunk_text"]

# Rough token estimate: English averages ~4 characters per token. This is only
# used for conservative budgeting, so an approximation is sufficient.
_CHARS_PER_TOKEN = 4

# Default per-chunk input-token budget. The real ceiling is per-request latency
# vs the request timeout (not the context window): empirically ~3K tokens
# translates in ~65s with safe margin, ~4K in ~81s, and ~8K exceeds the 120s
# timeout. 3K balances fewer API calls (vs the 10 RPM limit) against that margin.
DEFAULT_CHUNK_TOKENS = 3_000

# Paragraphs are separated by one or more blank lines.
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in ``text`` (chars / 4 heuristic)."""
    return len(text) // _CHARS_PER_TOKEN


def chunk_text(text: str, max_tokens: int = DEFAULT_CHUNK_TOKENS) -> list[str]:
    """Split ``text`` into token-budgeted chunks on paragraph boundaries.

    Args:
        text: Plain text to chunk (already preprocessed by the caller).
        max_tokens: Target maximum estimated input tokens per chunk. A single
            paragraph larger than this becomes its own (oversized) chunk.

    Returns:
        Chunks of one or more whole paragraphs, in reading order, each joined
        internally with blank lines.
    """
    paragraphs = _split_paragraphs(text)
    return _batch_paragraphs(paragraphs, max_tokens)


def _split_paragraphs(text: str) -> list[str]:
    """Split ``text`` into non-empty paragraphs on blank lines."""
    parts = _PARAGRAPH_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _batch_paragraphs(paragraphs: list[str], max_tokens: int) -> list[str]:
    """Greedily pack paragraphs into chunks up to ``max_tokens`` each."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        tokens = estimate_tokens(paragraph)
        # Start a new chunk if this paragraph would overflow the budget,
        # unless the current chunk is empty (a lone oversized paragraph).
        if current and current_tokens + tokens > max_tokens:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0
        current.append(paragraph)
        current_tokens += tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks
