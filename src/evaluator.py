"""LLM-as-a-judge: reference-free translation-quality evaluation.

A separate judge model (Google Gemini, via its OpenAI-compatible endpoint)
scores a sample of translated chunks against the English source on a
per-dimension rubric. Using a *different* provider than the translator keeps the
judge off SEA-LION's 10 RPM budget and avoids a model grading its own output
(self-evaluation bias). The per-chunk scores and rationales are aggregated into
an :class:`EvaluationReport` (serialisable to JSON and Markdown).

There is no gold-standard reference translation, so this is *quality
estimation*: the judge sees ``(source, translation)`` and rates the translation
on its own merits.

Source/translation alignment is free: the judge re-chunks the source with the
same chunker the translator used and looks up each chunk's translation by
content in the content-addressed :class:`~src.cache.ChunkCache`. So judging runs
against an already-translated book and relies on the per-chunk cache being
populated (the default translation path; a ``--no-cache`` run leaves nothing to
judge).
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

from src.cache import ChunkCache
from src.chunker import chunk_text
from src.config import Config, JudgeConfig
from src.ratelimit import RateLimiter

__all__ = [
    "ChunkEvaluation",
    "EvaluationReport",
    "JudgeError",
    "build_judge_client",
    "collect_pairs",
    "evaluate_translation",
]

# Rubric dimensions (key -> what the judge is asked to assess). Order is the
# display order in the report.
_DIMENSIONS: dict[str, str] = {
    "accuracy": (
        "Adequacy — does the translation faithfully convey the full meaning of "
        "the source, with no mistranslations, additions, or distortions?"
    ),
    "fluency": (
        "Fluency — is the translation natural, grammatical, idiomatic prose in "
        "the target language (not awkward or obviously machine-translated)?"
    ),
    "completeness": (
        "Completeness — is every sentence of the source rendered, with nothing "
        "dropped, truncated, or left untranslated?"
    ),
    "tone": (
        "Tone & style — are the register, literary style, and tone of the "
        "source preserved?"
    ),
}

# Inclusive integer score range applied to every dimension.
_SCORE_MIN = 1
_SCORE_MAX = 5

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


@dataclass(frozen=True)
class ChunkEvaluation:
    """One judged (source, translation) chunk pair.

    Attributes:
        index: The chunk's position in reading order (0-based).
        scores: Dimension key -> integer score in ``[_SCORE_MIN, _SCORE_MAX]``.
        notes: Dimension key -> the judge's one-line justification.
    """

    index: int
    scores: dict[str, int]
    notes: dict[str, str]

    @property
    def mean(self) -> float:
        """Mean score across all dimensions for this chunk."""
        return sum(self.scores.values()) / len(self.scores)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view of this evaluation."""
        return {
            "index": self.index,
            "mean": round(self.mean, 2),
            "dimensions": {
                dimension: {
                    "score": self.scores[dimension],
                    "note": self.notes[dimension],
                }
                for dimension in self.scores
            },
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregated results of judging a sample of translated chunks.

    Attributes:
        language: Full target-language name (e.g. ``"Thai"``).
        judge_model: The model that produced the judgements.
        translation_model: The model whose translations were judged.
        total_chunks: Number of chunks the whole book was split into.
        evaluations: The per-chunk evaluations, in sampled order.
    """

    language: str
    judge_model: str
    translation_model: str
    total_chunks: int
    evaluations: list[ChunkEvaluation]

    @property
    def sample_size(self) -> int:
        """Number of chunks actually judged."""
        return len(self.evaluations)

    def dimension_means(self) -> dict[str, float]:
        """Mean score per dimension across the sampled chunks."""
        if not self.evaluations:
            return {dimension: 0.0 for dimension in _DIMENSIONS}
        return {
            dimension: sum(e.scores[dimension] for e in self.evaluations)
            / len(self.evaluations)
            for dimension in _DIMENSIONS
        }

    def overall_mean(self) -> float:
        """Grand mean across every dimension of every sampled chunk."""
        means = self.dimension_means().values()
        return sum(means) / len(means) if means else 0.0

    def weakest_chunks(self, limit: int = 3) -> list[ChunkEvaluation]:
        """Return the lowest-scoring sampled chunks (most in need of review)."""
        return sorted(self.evaluations, key=lambda e: e.mean)[:limit]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable summary of the whole report."""
        return {
            "language": self.language,
            "judge_model": self.judge_model,
            "translation_model": self.translation_model,
            "total_chunks": self.total_chunks,
            "sample_size": self.sample_size,
            "overall_mean": round(self.overall_mean(), 2),
            "dimension_means": {
                dimension: round(value, 2)
                for dimension, value in self.dimension_means().items()
            },
            "evaluations": [e.to_dict() for e in self.evaluations],
        }

    def to_markdown(self) -> str:
        """Render a human-readable Markdown report."""
        return _render_markdown(self)


class JudgeError(RuntimeError):
    """Raised when the judge cannot produce a usable evaluation."""


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
        An :class:`EvaluationReport` over the sampled chunks.

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
    system = _judge_system(language)
    prompt = _judge_prompt(source, translation, language)
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
            scores, notes = _parse_judgement(content)
            return ChunkEvaluation(index=index, scores=scores, notes=notes)
        except (ValueError, KeyError) as error:
            last_error = error
    raise JudgeError(f"Could not parse a judgement for chunk {index}: {last_error}")


def _judge_system(language: str) -> str:
    """Build the judge system prompt for the configured target language."""
    rubric = "\n".join(f"- {key}: {desc}" for key, desc in _DIMENSIONS.items())
    keys = ", ".join(f'"{key}"' for key in _DIMENSIONS)
    return (
        f"You are a meticulous, impartial translation-quality evaluator for "
        f"English-to-{language} literary translation. You are given a source "
        f"passage and its translation. Score the translation on each dimension "
        f"from {_SCORE_MIN} (unacceptable) to {_SCORE_MAX} (excellent):\n"
        f"{rubric}\n\n"
        f"Respond with ONLY a JSON object (no markdown fences, no prose) whose "
        f"keys are exactly {keys}. Each value is an object "
        f'{{"score": <int>, "note": "<one concise sentence>"}}. '
        f"Base every score strictly on the texts shown; do not reward or "
        f"penalise length. Keep each note under 25 words."
    )


def _judge_prompt(source: str, translation: str, language: str) -> str:
    """Build the judge user message carrying the source/translation pair."""
    return f"SOURCE (English):\n{source}\n\nTRANSLATION ({language}):\n{translation}"


def _parse_judgement(content: str) -> tuple[dict[str, int], dict[str, str]]:
    """Parse the judge's JSON reply into score and note maps.

    Raises:
        ValueError: if no JSON object can be extracted or a dimension/score is
            malformed.
        KeyError: if a required dimension is absent.
    """
    data = _extract_json(content)
    scores: dict[str, int] = {}
    notes: dict[str, str] = {}
    for dimension in _DIMENSIONS:
        entry = data[dimension]  # KeyError if a dimension is missing.
        if not isinstance(entry, dict) or "score" not in entry:
            raise ValueError(f"Malformed entry for dimension '{dimension}'.")
        scores[dimension] = _clamp_score(entry["score"])
        notes[dimension] = str(entry.get("note", "")).strip()
    return scores, notes


def _extract_json(content: str) -> dict[str, object]:
    """Extract the first JSON object from ``content`` (tolerating code fences)."""
    text = content.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in the judge response.")
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in the judge response: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError("The judge response was not a JSON object.")
    return parsed


def _clamp_score(value: object) -> int:
    """Coerce ``value`` to an int clamped to the valid score range."""
    try:
        score = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Score {value!r} is not an integer.") from error
    return max(_SCORE_MIN, min(_SCORE_MAX, score))


def _render_markdown(report: EvaluationReport) -> str:
    """Render ``report`` as a Markdown document."""
    lines = [
        f"# Translation quality report — {report.language}",
        "",
        f"- **Judge model:** `{report.judge_model}`",
        f"- **Translation model:** `{report.translation_model}`",
        f"- **Chunks judged:** {report.sample_size} of {report.total_chunks}",
        f"- **Overall score:** {report.overall_mean():.2f} / {_SCORE_MAX}",
        "",
        "## Scores by dimension",
        "",
        "| Dimension | Mean score |",
        "| --- | --- |",
    ]
    for dimension, value in report.dimension_means().items():
        lines.append(f"| {dimension} | {value:.2f} |")

    lines += ["", "## Weakest sampled chunks", ""]
    for evaluation in report.weakest_chunks():
        detail = ", ".join(f"{dim} {evaluation.scores[dim]}" for dim in _DIMENSIONS)
        lines.append(
            f"- **Chunk {evaluation.index}** (mean {evaluation.mean:.2f}): {detail}"
        )
        for dimension in _DIMENSIONS:
            note = evaluation.notes[dimension]
            if note:
                lines.append(f"  - _{dimension}:_ {note}")

    lines.append("")
    return "\n".join(lines)
