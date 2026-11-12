"""Tests for src.evaluation_report (pure data model and rendering)."""

from __future__ import annotations

from src.evaluation_report import ChunkEvaluation, EvaluationReport
from src.judge_rubric import DIMENSIONS


def _evaluation(index: int, score: int, note: str = "") -> ChunkEvaluation:
    """A ChunkEvaluation giving every dimension the same ``score``."""
    return ChunkEvaluation(
        index=index,
        scores={dim: score for dim in DIMENSIONS},
        notes={dim: note for dim in DIMENSIONS},
    )


def _report(*evaluations: ChunkEvaluation, total_chunks: int = 10) -> EvaluationReport:
    return EvaluationReport(
        language="Thai",
        judge_model="judge-model",
        translation_model="translation-model",
        total_chunks=total_chunks,
        evaluations=list(evaluations),
    )


class TestChunkEvaluation:
    def test_mean_averages_the_dimensions(self) -> None:
        ev = ChunkEvaluation(
            index=0,
            scores={"accuracy": 4, "fluency": 2, "completeness": 5, "tone": 1},
            notes={},
        )
        assert ev.mean == 3.0

    def test_to_dict_rounds_mean_and_nests_dimensions(self) -> None:
        ev = _evaluation(2, 5, note="good")
        out = ev.to_dict()
        assert out["index"] == 2
        assert out["mean"] == 5.0
        assert out["dimensions"]["accuracy"] == {"score": 5, "note": "good"}


class TestEvaluationReport:
    def test_sample_size_counts_evaluations(self) -> None:
        assert _report(_evaluation(0, 4), _evaluation(1, 2)).sample_size == 2

    def test_dimension_and_overall_means(self) -> None:
        report = _report(_evaluation(0, 4), _evaluation(1, 2))
        assert report.dimension_means() == {dim: 3.0 for dim in DIMENSIONS}
        assert report.overall_mean() == 3.0

    def test_empty_report_means_are_zero(self) -> None:
        report = _report()
        assert report.dimension_means() == {dim: 0.0 for dim in DIMENSIONS}
        assert report.overall_mean() == 0.0

    def test_weakest_chunks_sorted_ascending_and_limited(self) -> None:
        report = _report(_evaluation(0, 5), _evaluation(1, 1), _evaluation(2, 3))
        weakest = report.weakest_chunks(limit=2)
        assert [e.index for e in weakest] == [1, 2]

    def test_to_dict_summary_fields(self) -> None:
        out = _report(_evaluation(0, 4), total_chunks=10).to_dict()
        assert out["language"] == "Thai"
        assert out["sample_size"] == 1
        assert out["total_chunks"] == 10
        assert out["overall_mean"] == 4.0
        assert set(out["dimension_means"]) == set(DIMENSIONS)

    def test_to_markdown_contains_header_table_and_weakest(self) -> None:
        md = _report(_evaluation(0, 4, note="solid")).to_markdown()
        assert "# Translation quality report — Thai" in md
        assert "| Dimension | Mean score |" in md
        assert "## Weakest sampled chunks" in md
        assert "solid" in md
