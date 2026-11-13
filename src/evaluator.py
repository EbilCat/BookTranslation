"""LLM-as-a-judge: reference-free translation-quality evaluation.

A separate judge model (Google Gemini, via its OpenAI-compatible endpoint)
scores a sample of translated chunks against the English source on a
per-dimension rubric. Using a *different* provider than the translator keeps the
judge off SEA-LION's 10 RPM budget and avoids a model grading its own output
(self-evaluation bias). The per-chunk scores and rationales are aggregated into
an :class:`~src.evaluation_report.EvaluationReport` (serialisable to JSON and
Markdown).

There is no gold-standard reference translation, so this is *quality
estimation*: the judge sees ``(source, translation)`` and rates the translation
on its own merits.

Source/translation alignment is free: the judge re-chunks the source with the
same chunker the translator used and looks up each chunk's translation by
content in the content-addressed :class:`~src.cache.ChunkCache`. So judging runs
against an already-translated book and relies on the per-chunk cache being
populated (the default translation path; a ``--no-cache`` run leaves nothing to
judge).

This module holds the async orchestration only; the pure grading spec and reply
parsing live in :mod:`src.judge_rubric`, and the aggregation/reporting in
:mod:`src.evaluation_report`.
"""

from __future__ import annotations

import asyncio
import random
from pathlib import Path

from openai import AsyncOpenAI

from src.cache import ChunkCache
from src.chunker import chunk_text
from src.config import Config, JudgeConfig
from src.evaluation_report import ChunkEvaluation, EvaluationReport
from src.judge_rubric import judge_prompt, judge_system, parse_judgement
from src.ratelimit import RateLimiter

__all__ = [
    "ChunkEvaluation",
    "EvaluationReport",
    "JudgeError",
    "build_judge_client",
    "collect_pairs",
    "evaluate_translation",
]

# Judge request limits. The judge runs on its own provider/budget; a small
# sample means the rolling-minute cap is rarely the binding constraint.
_JUDGE_RATE_LIMIT_RPM = 10
_JUDGE_MAX_CONCURRENCY = 5

# Output-token cap for one judgement. Generous so a thinking-enabled judge model
# can reason briefly and still emit the full JSON object.
_JUDGE_MAX_TOKENS = 2048

# One retry if the judge returns unparseable / empty content (transient).
_JUDGE_PARSE_RETRIES = 1

# Default number of chunks to sample when the caller does not specify.
DEFAULT_SAMPLE_SIZE = 8


def build_judge_client(judge: JudgeConfig) -> AsyncOpenAI:
    """Create an AsyncOpenAI client pointed at the judge provider (Gemini)."""
    return AsyncOpenAI(
        api_key=judge.api_key,
        base_url=judge.base_url,
        timeout=judge.timeout,
        max_retries=_JUDGE_PARSE_RETRIES + 2,
    )


def collect_pairs(
    source_text: str,
    config: Config,
    cache_dir: str | Path,
) -> list[tuple[int, str, str]]:
    """Pair each source chunk with its cached translation.

    Re-chunks ``source_text`` exactly as the translator did and looks each chunk
    up by content in the content-addressed cache, so the returned translation is
    guaranteed to correspond to its source chunk.

    Args:
        source_text: The preprocessed source text that was translated.
        config: Resolved settings (supplies the target language and model that
            key the cache).
        cache_dir: Directory of the per-chunk translation cache.

    Returns:
        ``(index, source_chunk, translation)`` tuples for every chunk that has a
        cached translation, in reading order. Chunks with no cache entry (e.g.
        an interrupted or ``--no-cache`` run) are omitted.
    """
    chunks = chunk_text(source_text)
    cache = ChunkCache(cache_dir, language=config.target_language, model=config.model)
    pairs: list[tuple[int, str, str]] = []
    for index, chunk in enumerate(chunks):
        translation = cache.get(chunk)
        if translation is not None:
            pairs.append((index, chunk, translation))
    return pairs


async def evaluate_translation(
    source_text: str,
    config: Config,
    judge: JudgeConfig,
    *,
    cache_dir: str | Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int | None = None,
) -> EvaluationReport:
    """Judge a random sample of translated chunks and aggregate the scores.

    Args:
        source_text: The preprocessed source text that was translated.
        config: Resolved translation settings (language, model; keys the cache).
        judge: Resolved judge settings (provider, model).
        cache_dir: Directory of the per-chunk translation cache to read from.
        sample_size: Number of chunks to judge (capped at the number available).
        seed: Optional RNG seed for a reproducible sample.

    Returns:
        An :class:`~src.evaluation_report.EvaluationReport` over the sampled
        chunks.

    Raises:
        JudgeError: if no translated chunks are available to judge.
    """
    total_chunks = len(chunk_text(source_text))
    pairs = collect_pairs(source_text, config, cache_dir)
    if not pairs:
        raise JudgeError(
            "No cached translations found to judge. Run a translation with the "
            "cache enabled (the default; do not pass --no-cache) first."
        )

    sample = _sample(pairs, sample_size, seed)
    language = config.language_name
    client = build_judge_client(judge)
    limiter = RateLimiter(_JUDGE_RATE_LIMIT_RPM)
    semaphore = asyncio.Semaphore(_JUDGE_MAX_CONCURRENCY)

    async def worker(index: int, source: str, translation: str) -> ChunkEvaluation:
        async with semaphore:
            return await _judge_pair(
                client, index, source, translation, judge, language, limiter
            )

    try:
        evaluations = await asyncio.gather(
            *(worker(i, src, tr) for i, src, tr in sample)
        )
    finally:
        await client.close()

    return EvaluationReport(
        language=language,
        judge_model=judge.model,
        translation_model=config.model,
        total_chunks=total_chunks,
        evaluations=list(evaluations),
    )


class JudgeError(RuntimeError):
    """Raised when the judge cannot produce a usable evaluation."""


def _sample(
    pairs: list[tuple[int, str, str]],
    sample_size: int,
    seed: int | None,
) -> list[tuple[int, str, str]]:
    """Pick up to ``sample_size`` pairs at random, returned in reading order."""
    count = min(sample_size, len(pairs))
    chosen = random.Random(seed).sample(pairs, count)
    return sorted(chosen, key=lambda pair: pair[0])


async def _judge_pair(
    client: AsyncOpenAI,
    index: int,
    source: str,
    translation: str,
    judge: JudgeConfig,
    language: str,
    limiter: RateLimiter,
) -> ChunkEvaluation:
    """Send one (source, translation) pair to the judge and parse the result."""
    system = judge_system(language)
    prompt = judge_prompt(source, translation, language)
    last_error: Exception | None = None
    for _ in range(_JUDGE_PARSE_RETRIES + 1):
        await limiter.acquire()
        response = await client.chat.completions.create(
            model=judge.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=_JUDGE_MAX_TOKENS,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        try:
            scores, notes = parse_judgement(content)
            return ChunkEvaluation(index=index, scores=scores, notes=notes)
        except (ValueError, KeyError) as error:
            last_error = error
    raise JudgeError(f"Could not parse a judgement for chunk {index}: {last_error}")
