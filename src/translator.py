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
from pathlib import Path

from src.cache import ChunkCache
from src.chunker import chunk_text, estimate_tokens
from src.config import SUPPORTED_LANGUAGES, Config
from src.inference import (
    TranslationTruncatedError,
    build_client,
    run_inference,
)
from src.ratelimit import RateLimiter

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


async def translate_text(
    text: str,
    config: Config,
    *,
    cache_dir: str | Path | None = None,
) -> str:
    """Translate plain text and return the reassembled translation.

    When ``cache_dir`` is given, each chunk's translation is written to disk as
    it completes and reused on a later run, so an interrupted translation can
    resume instead of re-translating (and re-paying for) chunks already done.

    Args:
        text: Plain text to translate (already preprocessed by the caller).
        config: Resolved application settings (model, target language, limits).
        cache_dir: Directory for the resumable per-chunk translation cache.
            ``None`` disables caching (every chunk is translated afresh).

    Returns:
        The translated text, chunks rejoined in order with blank lines.
    """
    chunks = chunk_text(text)
    client = build_client(config)
    system = _system_prompt(config)
    limiter = RateLimiter(_RATE_LIMIT_RPM)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
    progress = _Progress(len(chunks))
    cache = (
        ChunkCache(
            cache_dir,
            language=config.target_language,
            model=config.model,
        )
        if cache_dir is not None
        else None
    )

    async def worker(index: int, chunk: str) -> tuple[int, str]:
        if cache is not None:
            cached = cache.get(chunk)
            if cached is not None:
                await progress.tick(cached=True)
                return index, cached
        async with semaphore:
            translated = await _translate_chunk(client, chunk, config, system, limiter)
        if cache is not None:
            cache.set(chunk, translated)
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


class _Progress:
    """Thread-safe-ish async counter that prints completion progress."""

    def __init__(self, total: int) -> None:
        self._total = total
        self._done = 0
        self._cached = 0
        self._lock = asyncio.Lock()

    async def tick(self, *, cached: bool = False) -> None:
        """Record one finished chunk and print progress to stderr.

        Args:
            cached: True if the chunk was served from the resume cache rather
                than freshly translated; tallied and shown in the progress line.
        """
        async with self._lock:
            self._done += 1
            if cached:
                self._cached += 1
            suffix = f" ({self._cached} resumed from cache)" if self._cached else ""
            print(
                f"  translated {self._done}/{self._total} chunks{suffix}",
                end="\r",
                file=sys.stderr,
                flush=True,
            )
            if self._done == self._total:
                print(file=sys.stderr)


def _system_prompt(config: Config) -> str:
    """Build the translation system prompt for the configured language."""
    language = SUPPORTED_LANGUAGES.get(config.target_language, config.target_language)
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
    limiter: RateLimiter,
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
            client,
            chunk,
            config,
            system,
            limiter,
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
            client,
            chunk,
            config,
            system,
            limiter,
            reason="was truncated",
        )


async def _split_and_retry(
    client,
    chunk: str,
    config: Config,
    system: str,
    limiter: RateLimiter,
    *,
    reason: str,
) -> str:
    """Split a chunk into smaller pieces and translate each, else give up."""
    pieces = _split_chunk(chunk)
    if len(pieces) == 1:
        raise RuntimeError(f"A single paragraph {reason} and cannot be split further.")
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
