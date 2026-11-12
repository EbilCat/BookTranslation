"""Translate plain text with the SEA-LION API.

Chunks the text (see :mod:`src.chunker`), then translates the chunks
concurrently with ``AsyncOpenAI``. Concurrency is bounded by a semaphore and
all API calls pass through a shared rate limiter that respects SEA-LION's
10 requests-per-minute limit. If a chunk's translation is truncated (or can't
fit the context window), the chunk is split and retried ("degrade on failure").
Results carry their index so the text is reassembled in order.
"""

from __future__ import annotations

import asyncio
import sys
from collections import deque

from src.chunker import chunk_text, estimate_tokens
from src.config import SUPPORTED_LANGUAGES, Config
from src.inference import (
    TranslationTruncatedError,
    build_client,
    run_inference,
)

__all__ = ["translate_text"]

# SEA-LION API rate limit (requests per minute per user).
_RATE_LIMIT_RPM = 10

# Maximum concurrent in-flight requests (the rate limiter still caps starts).
_MAX_CONCURRENCY = 10

# Output tokens are bounded at input_tokens * this factor (headroom for scripts
# that expand relative to English), floored at _MIN_OUTPUT_TOKENS.
_OUTPUT_EXPANSION = 3.0
_MIN_OUTPUT_TOKENS = 256

# Rough token allowance for the system prompt and chat formatting overhead.
_PROMPT_OVERHEAD_TOKENS = 500


async def translate_text(text: str, config: Config) -> str:
    """Translate plain text and return the reassembled translation.

    Args:
        text: Plain text to translate (already preprocessed by the caller).
        config: Resolved application settings (model, target language, limits).

    Returns:
        The translated text, chunks rejoined in order with blank lines.
    """
    chunks = chunk_text(text)
    client = build_client(config)
    system = _system_prompt(config)
    limiter = _RateLimiter(_RATE_LIMIT_RPM)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
    progress = _Progress(len(chunks))

    async def worker(index: int, chunk: str) -> tuple[int, str]:
        async with semaphore:
            translated = await _translate_chunk(
                client, chunk, config, system, limiter
            )
        await progress.tick()
        return index, translated

    try:
        results = await asyncio.gather(
            *(worker(i, chunk) for i, chunk in enumerate(chunks))
        )
    finally:
        await client.close()

    results = sorted(results)
    return "\n\n".join(translated for _, translated in results)


class _RateLimiter:
    """Async limiter capping starts to ``rpm`` within any rolling 60s window."""

    def __init__(self, rpm: int) -> None:
        self._rpm = rpm
        self._window = 60.0
        self._starts: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until starting another request stays within the limit."""
        async with self._lock:
            loop = asyncio.get_running_loop()
            self._evict(loop.time())
            if len(self._starts) >= self._rpm:
                await asyncio.sleep(self._window - (loop.time() - self._starts[0]))
                self._evict(loop.time())
            self._starts.append(loop.time())

    def _evict(self, now: float) -> None:
        """Drop start timestamps older than the rolling window."""
        while self._starts and now - self._starts[0] >= self._window:
            self._starts.popleft()


class _Progress:
    """Thread-safe-ish async counter that prints completion progress."""

    def __init__(self, total: int) -> None:
        self._total = total
        self._done = 0
        self._lock = asyncio.Lock()

    async def tick(self) -> None:
        """Record one completed chunk and print progress to stderr."""
        async with self._lock:
            self._done += 1
            print(
                f"  translated {self._done}/{self._total} chunks",
                end="\r",
                file=sys.stderr,
                flush=True,
            )
            if self._done == self._total:
                print(file=sys.stderr)


def _system_prompt(config: Config) -> str:
    """Build the translation system prompt for the configured language."""
    language = SUPPORTED_LANGUAGES.get(
        config.target_language, config.target_language
    )
    return (
        f"You are a professional literary translator. Translate the user's "
        f"text into {language}. Preserve the original meaning, tone, paragraph "
        "structure, and line breaks. Output only the translation, with no "
        "explanations, notes, or commentary."
    )


async def _translate_chunk(
    client,
    chunk: str,
    config: Config,
    system: str,
    limiter: _RateLimiter,
) -> str:
    """Translate one chunk, splitting and retrying if it fails to fit.

    Raises:
        RuntimeError: if a single, unsplittable paragraph cannot fit the
            context window or keeps getting truncated.
    """
    input_tokens = estimate_tokens(chunk)
    output_cap = config.context_window - input_tokens - _PROMPT_OVERHEAD_TOKENS

    # Guard: if even a minimal response can't fit the context window, the chunk
    # is too big — split it (or fail if it's a single paragraph).
    if output_cap <= _MIN_OUTPUT_TOKENS:
        return await _split_and_retry(
            client, chunk, config, system, limiter,
            reason="exceeds the context window",
        )

    max_tokens = min(
        int(input_tokens * _OUTPUT_EXPANSION) + _MIN_OUTPUT_TOKENS, output_cap
    )
    await limiter.acquire()
    try:
        return await run_inference(
            client, chunk, config, system=system, max_tokens=max_tokens
        )
    except TranslationTruncatedError:
        return await _split_and_retry(
            client, chunk, config, system, limiter,
            reason="was truncated",
        )


async def _split_and_retry(
    client,
    chunk: str,
    config: Config,
    system: str,
    limiter: _RateLimiter,
    *,
    reason: str,
) -> str:
    """Split a chunk into smaller pieces and translate each, else give up."""
    pieces = _split_chunk(chunk)
    if len(pieces) == 1:
        raise RuntimeError(
            f"A single paragraph {reason} and cannot be split further."
        )
    parts = [
        await _translate_chunk(client, piece, config, system, limiter)
        for piece in pieces
    ]
    return "\n\n".join(parts)


def _split_chunk(chunk: str) -> list[str]:
    """Split a chunk into two halves on paragraph boundaries.

    Returns a single-element list when the chunk is one paragraph (cannot be
    split without breaking a semantic unit).
    """
    paragraphs = chunk.split("\n\n")
    if len(paragraphs) == 1:
        return [chunk]
    mid = len(paragraphs) // 2
    return ["\n\n".join(paragraphs[:mid]), "\n\n".join(paragraphs[mid:])]
