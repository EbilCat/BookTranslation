"""Evaluation data model and reporting.

Pure and I/O-free: the per-chunk judgement (:class:`ChunkEvaluation`), the
aggregated :class:`EvaluationReport` (per-dimension and overall means, weakest
chunks), and its JSON/Markdown renderings. The report aggregates over the
rubric's dimensions, so it imports the grading spec from :mod:`src.judge_rubric`.
Kept separate from the async judging in :mod:`src.evaluator` so the aggregation
math is unit-testable without a live judge call.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.judge_rubric import DIMENSIONS, SCORE_MAX

__all__ = ["ChunkEvaluation", "EvaluationReport"]


@dataclass(frozen=True)
class ChunkEvaluation:
    """One judged (source, translation) chunk pair.

    Attributes:
        index: The chunk's position in reading order (0-based).
        scores: Dimension key -> integer score in ``[SCORE_MIN, SCORE_MAX]``.
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
            return {dimension: 0.0 for dimension in DIMENSIONS}
        return {
            dimension: sum(e.scores[dimension] for e in self.evaluations)
            / len(self.evaluations)
            for dimension in DIMENSIONS
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


def _render_markdown(report: EvaluationReport) -> str:
    """Render ``report`` as a Markdown document."""
    lines = [
        f"# Translation quality report — {report.language}",
        "",
        f"- **Judge model:** `{report.judge_model}`",
        f"- **Translation model:** `{report.translation_model}`",
        f"- **Chunks judged:** {report.sample_size} of {report.total_chunks}",
        f"- **Overall score:** {report.overall_mean():.2f} / {SCORE_MAX}",
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
        detail = ", ".join(f"{dim} {evaluation.scores[dim]}" for dim in DIMENSIONS)
        lines.append(
            f"- **Chunk {evaluation.index}** (mean {evaluation.mean:.2f}): {detail}"
        )
        for dimension in DIMENSIONS:
            note = evaluation.notes[dimension]
            if note:
                lines.append(f"  - _{dimension}:_ {note}")

    lines.append("")
    return "\n".join(lines)
