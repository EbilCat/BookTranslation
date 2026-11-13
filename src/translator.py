"""Translate plain text with the SEA-LION API.

Chunks the text (see :mod:`src.chunker`), then translates the chunks
concurrently with ``AsyncOpenAI``. Concurrency is bounded by a semaphore and
all API calls pass through a shared rate limiter that respects SEA-LION's
10 requests-per-minute limit. If a chunk's translation is truncated (or can't
fit the context window), the chunk is split and retried ("degrade on failure").
Results carry their index so the text is reassembled in order.

Chunk failures are handled in whole-book passes: a pass lets every chunk run to
completion (so one flaky chunk does not abort the whole book), and if any chunk
failed on a *transient* error (timeout, rate limit, 5xx) the whole book is
retried, up to ``max_attempts`` passes. Chunks that already succeeded are
skipped on retry
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

from src.cache import ChunkCache
from src.chunker import chunk_text, estimate_tokens
from src.config import Config
from src.inference import (
    TranslationTruncatedError,
    build_client,
    run_inference,
)
from src.ratelimit import RateLimiter
from src.translation_policy import (
    backoff_seconds,
    build_system_prompt,
    is_transient,
    plan_output_budget,
    split_chunk,
)

__all__ = ["translate_text", "TranslationFailedError"]

# Default number of whole-book translation passes (initial + retries).
_DEFAULT_MAX_ATTEMPTS = 3

# SEA-LION API rate limit (requests per minute per user).
_RATE_LIMIT_RPM = 10

# Maximum concurrent in-flight requests (the rate limiter still caps starts).
_MAX_CONCURRENCY = 10


async def translate_text(
    text: str,
    config: Config,
    *,
    cache_dir: str | Path | None = None,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> str:
    """Translate plain text and return the reassembled translation.

    The book is translated in whole-book passes. Within a pass every chunk runs
    to completion, so a single flaky chunk does not abort the whole book. If chunks
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
    system = build_system_prompt(config)
    language = config.language_name
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

        pending = _plan_retry(failures, language, attempt, max_attempts)
        await asyncio.sleep(backoff_seconds(attempt))

    return "\n\n".join(translations[i] for i in range(len(chunks)))


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


def _plan_retry(
    failures: list[tuple[int, BaseException]],
    language: str,
    attempt: int,
    max_attempts: int,
) -> list[int]:
    """Decide what to do after a pass left some chunks untranslated.

    Args:
        failures: ``(index, exception)`` for each chunk that failed this pass.
        language: Target language name (for the error message).
        attempt: The pass number that just ran (1-based).
        max_attempts: Maximum number of whole-book passes allowed.

    Returns:
        The chunk indices to re-attempt on the next pass.

    Raises:
        TranslationFailedError: if translation must stop — either a permanent
            (non-transient) failure, or transient failures on the final pass.
    """
    failed_indices = [index for index, _ in failures]
    permanent_errors = [
        (index, error) for index, error in failures if not is_transient(error)
    ]
    if permanent_errors:
        raise TranslationFailedError(
            language,
            failed_indices,
            attempt,
            recoverable=False,
            last_error=permanent_errors[-1][1],
        )
    if attempt == max_attempts:
        raise TranslationFailedError(
            language,
            failed_indices,
            attempt,
            recoverable=True,
            last_error=failures[-1][1],
        )
    return failed_indices


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
    budget = plan_output_budget(estimate_tokens(chunk), config.context_window)

    # If even a minimal response can't fit the context window, the chunk is too
    # big — split it (or fail if it's a single paragraph).
    if budget is None:
        return await _split_and_retry(
            client,
            chunk,
            config,
            system,
            limiter,
            reason="exceeds the context window",
        )

    await limiter.acquire()
    try:
        return await run_inference(
            client, chunk, config, system=system, max_tokens=budget
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
    pieces = split_chunk(chunk)
    if len(pieces) == 1:
        raise RuntimeError(f"A single paragraph {reason} and cannot be split further.")
    parts = [
        await _translate_chunk(client, piece, config, system, limiter)
        for piece in pieces
    ]
    return "\n\n".join(parts)
