"""Pure translation-policy decisions, free of I/O.

These helpers make the translator's choices — how many output tokens to allow,
whether a failure is worth retrying, how long to back off, how to split an
oversized chunk, and what system prompt to send — without performing any network
or disk I/O. Keeping them here (separate from the async orchestration in
:mod:`src.translator`) makes each decision unit-testable without touching the
SEA-LION API.
"""

from __future__ import annotations

import openai

from src.config import Config

__all__ = [
    "TRANSIENT_ERRORS",
    "backoff_seconds",
    "build_system_prompt",
    "is_transient",
    "plan_output_budget",
    "split_chunk",
]

# Transient errors worth retrying with another pass. APITimeoutError subclasses
# APIConnectionError, so it is covered; RateLimitError is 429; InternalServer
# Error covers 5xx. Anything else (auth, bad request, a paragraph that cannot be
# split) is treated as permanent and aborts immediately.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)

# Between retry passes, wait ``base * 2 ** (attempt - 1)`` seconds, capped, to
# let a brief provider outage clear before slamming it again.
_RETRY_BACKOFF_BASE = 3.0
_RETRY_BACKOFF_CAP = 30.0

# Output tokens are bounded at input_tokens * this factor (headroom for scripts
# that expand relative to English), floored at _MIN_OUTPUT_TOKENS.
_OUTPUT_EXPANSION = 3.0
_MIN_OUTPUT_TOKENS = 256

# Rough token allowance for the system prompt and chat formatting overhead.
_PROMPT_OVERHEAD_TOKENS = 500


def is_transient(error: BaseException) -> bool:
    """Return True if ``error`` is worth retrying with another pass."""
    return isinstance(error, TRANSIENT_ERRORS)


def backoff_seconds(attempt: int) -> float:
    """Seconds to wait before the pass following a failed ``attempt``."""
    delay: float = _RETRY_BACKOFF_BASE * 2 ** (attempt - 1)
    return min(delay, _RETRY_BACKOFF_CAP)


def plan_output_budget(input_tokens: int, context_window: int) -> int | None:
    """Decide the output-token cap for translating a chunk.

    Args:
        input_tokens: Estimated token count of the chunk to translate.
        context_window: Assumed model context window in tokens.

    Returns:
        The ``max_tokens`` cap to request, or ``None`` if even a minimal
        response could not fit the context window (the caller must split the
        chunk instead of sending it).
    """
    output_cap = context_window - input_tokens - _PROMPT_OVERHEAD_TOKENS
    if output_cap <= _MIN_OUTPUT_TOKENS:
        return None
    return min(int(input_tokens * _OUTPUT_EXPANSION) + _MIN_OUTPUT_TOKENS, output_cap)


def build_system_prompt(config: Config) -> str:
    """Build the translation system prompt for the configured language."""
    language = config.language_name
    return (
        f"You are a professional literary translator. Translate the user's "
        f"text into {language}. Preserve the original meaning, tone, paragraph "
        "structure, and line breaks. Output only the translation, with no "
        "explanations, notes, or commentary."
    )


def split_chunk(chunk: str) -> list[str]:
    """Split a chunk into two halves on paragraph boundaries.

    Returns a single-element list when the chunk is one paragraph (cannot be
    split without breaking a semantic unit).
    """
    paragraphs = chunk.split("\n\n")
    if len(paragraphs) == 1:
        return [chunk]
    mid = len(paragraphs) // 2
    return ["\n\n".join(paragraphs[:mid]), "\n\n".join(paragraphs[mid:])]
