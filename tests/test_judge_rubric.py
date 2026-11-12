"""Tests for src.judge_rubric (grading spec, prompts, reply parsing)."""

from __future__ import annotations

import json

import pytest

from src.judge_rubric import (
    DIMENSIONS,
    SCORE_MAX,
    SCORE_MIN,
    clamp_score,
    extract_json,
    judge_prompt,
    judge_system,
    parse_judgement,
)


def _full_judgement(score: int = 4, note: str = "ok") -> str:
    """Serialise a well-formed judgement covering every rubric dimension."""
    return json.dumps({dim: {"score": score, "note": note} for dim in DIMENSIONS})


class TestParseJudgement:
    def test_parses_a_well_formed_object(self) -> None:
        scores, notes = parse_judgement(_full_judgement(score=5, note="great"))
        assert scores == {dim: 5 for dim in DIMENSIONS}
        assert notes == {dim: "great" for dim in DIMENSIONS}

    def test_tolerates_code_fences(self) -> None:
        fenced = f"```json\n{_full_judgement(score=3)}\n```"
        scores, _ = parse_judgement(fenced)
        assert scores == {dim: 3 for dim in DIMENSIONS}

    def test_clamps_out_of_range_scores(self) -> None:
        scores, _ = parse_judgement(_full_judgement(score=99))
        assert all(v == SCORE_MAX for v in scores.values())

    def test_missing_note_becomes_empty_string(self) -> None:
        payload = json.dumps({dim: {"score": 4} for dim in DIMENSIONS})
        _, notes = parse_judgement(payload)
        assert all(note == "" for note in notes.values())

    def test_missing_dimension_raises_keyerror(self) -> None:
        partial = json.dumps({"accuracy": {"score": 4, "note": "x"}})
        with pytest.raises(KeyError):
            parse_judgement(partial)

    def test_malformed_entry_raises_valueerror(self) -> None:
        # "accuracy" present but not a {score: ...} object.
        bad = json.dumps({dim: "nope" for dim in DIMENSIONS})
        with pytest.raises(ValueError):
            parse_judgement(bad)


class TestExtractJson:
    def test_plain_object(self) -> None:
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_object_embedded_in_prose_and_fences(self) -> None:
        assert extract_json('Here you go:\n```\n{"a": 1}\n```') == {"a": 1}

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="No JSON object"):
            extract_json("there is no object here")

    def test_non_object_json_raises(self) -> None:
        with pytest.raises(ValueError, match="not a JSON object"):
            extract_json("[1, 2, 3]")


class TestClampScore:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("4", 4), (4, 4), (9, SCORE_MAX), (0, SCORE_MIN), (-3, SCORE_MIN), (3.7, 3)],
    )
    def test_coerces_and_clamps(self, value: object, expected: int) -> None:
        assert clamp_score(value) == expected

    @pytest.mark.parametrize("value", ["abc", None, "3.7"])
    def test_non_integer_raises(self, value: object) -> None:
        with pytest.raises(ValueError):
            clamp_score(value)


class TestPrompts:
    def test_judge_system_embeds_language_and_rubric(self) -> None:
        system = judge_system("Thai")
        assert "Thai" in system
        assert all(dim in system for dim in DIMENSIONS)
        assert "JSON" in system

    def test_judge_prompt_carries_source_and_translation(self) -> None:
        prompt = judge_prompt("hello", "สวัสดี", "Thai")
        assert "hello" in prompt
        assert "สวัสดี" in prompt
        assert "Thai" in prompt
