"""Translate plain text with the SEA-LION API.

Chunks the text (see :mod:`src.chunker`), then translates the chunks
concurrently with ``AsyncOpenAI``. Concurrency is bounded by a semaphore and
all API calls pass through a shared rate limiter that respects SEA-LION's
10 requests-per-minute limit. If a chunk's translation is truncated (or can't
fit the context window), the chunk is split and retried ("degrade on failure").
Results carry their index so the text is reassembled in order.

Chunk failures are handled in whole-book passes: a pass lets every chunk run to
completion (one flaky chunk no longer aborts the book), and if any chunk failed
on a *transient* error (timeout, rate limit, 5xx) the whole book is retried, up
to ``max_attempts`` passes. Chunks that already succeeded are skipped on retry
(kept in memory and, when enabled, on disk via the cache), so a retry pass only
re-attempts the stragglers. A *permanent* failure (e.g. a single paragraph that
cannot be split, or an auth error) aborts immediately rather than wasting
passes. When chunks still fail after the final pass, ``TranslationFailedError``
is raised and no output is written.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import openai

from src.cache import ChunkCache
from src.chunker import chunk_text, estimate_tokens
from src.config import SUPPORTED_LANGUAGES, Config
from src.inference import (
    TranslationTruncatedError,
    build_client,
    run_inference,
)
from src.ratelimit import RateLimiter

__all__ = ["translate_text", "TranslationFailedError"]

# Default number of whole-book translation passes (initial + retries).
_DEFAULT_MAX_ATTEMPTS = 3

# Between retry passes, wait ``base * 2 ** (attempt - 1)`` seconds, capped, to
# let a brief provider outage clear before slamming it again.
_RETRY_BACKOFF_BASE = 3.0
_RETRY_BACKOFF_CAP = 30.0

# Transient errors worth retrying with another pass. APITimeoutError subclasses
# APIConnectionError, so it is covered; RateLimitError is 429; InternalServer
# Error covers 5xx. Anything else (auth, bad request, a paragraph that cannot be
# split) is treated as permanent and aborts immediately.
_TRANSIENT_ERRORS = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)

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


class TranslationFailedError(RuntimeError):
    """Raised when some chunks could not be translated after all passes.

    Attributes:
        language: Target language name (for the message).
        failed_indices: Sorted chunk indices that never succeeded.
        attempts: Number of whole-book passes attempted.
        recoverable: True if every remaining failure was transient (a later
            rerun may succeed); False if at least one failure was permanent.
        last_error: A representative underlying exception, if any.
    """

    def __init__(
        self,
        language: str,
        failed_indices: list[int],
        attempts: int,
        *,
        recoverable: bool,
        last_error: BaseException | None = None,
    ) -> None:
        """Build the error and its human-readable message."""
        self.language = language
        self.failed_indices = sorted(failed_indices)
        self.attempts = attempts
        self.recoverable = recoverable
        self.last_error = last_error
        shown = ", ".join(f"#{i}" for i in self.failed_indices)
        kind = "transient" if recoverable else "unrecoverable"
        detail = (
            f"{type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "unknown error"
        )
        super().__init__(
            f"Translation to {language} failed after {attempts} attempt(s): "
            f"{len(self.failed_indices)} chunk(s) could not be translated "
            f"({kind}): {shown}. Last error: {detail}."
        )


async def translate_text(
    text: str,
    config: Config,
    *,
    cache_dir: str | Path | None = None,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> str:
    """Translate plain text and return the reassembled translation.

    The book is translated in whole-book passes. Within a pass every chunk runs
    to completion, so a single flaky chunk no longer aborts the book. If chunks
    fail only on transient errors, the book is retried (up to ``max_attempts``
    passes); chunks already translated are skipped, so a retry pass re-attempts
    only the stragglers. A permanent failure aborts immediately.

    When ``cache_dir`` is given, each chunk's translation is written to disk as
    it completes and reused on a later run, so an interrupted translation can
    resume instead of re-translating (and re-paying for) chunks already done.

    Args:
        text: Plain text to translate (already preprocessed by the caller).
        config: Resolved application settings (model, target language, limits).
        cache_dir: Directory for the resumable per-chunk translation cache.
            ``None`` disables caching (every chunk is translated afresh).
        max_attempts: Maximum number of whole-book passes (initial + retries).

    Returns:
        The translated text, chunks rejoined in order with blank lines.

    Raises:
        ValueError: if ``max_attempts`` is less than 1.
        TranslationFailedError: if any chunk is still untranslated after the
            final pass, or on the first permanent (non-transient) failure.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    chunks = chunk_text(text)
    system = _system_prompt(config)
    language = SUPPORTED_LANGUAGES.get(config.target_language, config.target_language)
    cache = (
        ChunkCache(
            cache_dir,
            language=config.target_language,
            model=config.model,
        )
        if cache_dir is not None
        else None
    )

    translations: dict[int, str] = {}
    pending = list(range(len(chunks)))

    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            print(
                f"Attempt {attempt}/{max_attempts}: retrying "
                f"{len(pending)} failed chunk(s) ...",
                file=sys.stderr,
            )
        failures = await _run_pass(chunks, pending, translations, config, system, cache)
        if not failures:
            break

        permanent = [(i, e) for i, e in failures if not _is_transient(e)]
        if permanent:
            raise TranslationFailedError(
                language,
                [i for i, _ in failures],
                attempt,
                recoverable=False,
                last_error=permanent[-1][1],
            )

        pending = [i for i, _ in failures]
        if attempt == max_attempts:
            raise TranslationFailedError(
                language,
                pending,
                attempt,
                recoverable=True,
                last_error=failures[-1][1],
            )
        await asyncio.sleep(_backoff_seconds(attempt))

    return "\n\n".join(translations[i] for i in range(len(chunks)))


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


async def _run_pass(
    chunks: list[str],
    pending: list[int],
    translations: dict[int, str],
    config: Config,
    system: str,
    cache: ChunkCache | None,
) -> list[tuple[int, BaseException]]:
    """Translate the pending chunks once, concurrently.

    Successful translations are stored in ``translations`` (keyed by chunk
    index). Each pass gets a fresh client, rate limiter, and concurrency gate.

    Args:
        chunks: All chunks of the book, in reading order.
        pending: Indices into ``chunks`` to translate on this pass.
        translations: Accumulator; completed chunks are added here.
        config: Resolved application settings.
        system: The translation system prompt.
        cache: Optional per-chunk cache (``None`` disables it).

    Returns:
        A list of ``(index, exception)`` for chunks that failed this pass.
    """
    client = build_client(config)
    limiter = RateLimiter(_RATE_LIMIT_RPM)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
    progress = _Progress(len(pending))

    async def worker(index: int) -> tuple[int, str]:
        chunk = chunks[index]
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
            *(worker(i) for i in pending), return_exceptions=True
        )
    finally:
        await client.close()

    failures: list[tuple[int, BaseException]] = []
    for index, result in zip(pending, results):
        if isinstance(result, BaseException):
            failures.append((index, result))
        else:
            translations[index] = result[1]
    return failures


def _is_transient(error: BaseException) -> bool:
    """Return True if ``error`` is worth retrying with another pass."""
    return isinstance(error, _TRANSIENT_ERRORS)


def _backoff_seconds(attempt: int) -> float:
    """Seconds to wait before the pass following a failed ``attempt``."""
    return min(_RETRY_BACKOFF_BASE * 2 ** (attempt - 1), _RETRY_BACKOFF_CAP)


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
