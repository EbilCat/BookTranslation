"""Tests for src.translation_policy (pure translator decisions)."""

from __future__ import annotations

import openai
import pytest

from src.config import Config
from src.translation_policy import (
    backoff_seconds,
    build_system_prompt,
    is_transient,
    plan_output_budget,
    split_chunk,
)


class TestPlanOutputBudget:
    def test_small_input_uses_expansion_floor(self) -> None:
        # min(10*3 + 256, 128000-10-500) = min(286, 127490) = 286
        assert plan_output_budget(10, 128_000) == 286

    def test_zero_input_is_min_output(self) -> None:
        assert plan_output_budget(0, 128_000) == 256

    def test_cap_binds_when_window_is_tight(self) -> None:
        # cap = 128000-40000-500 = 87500 < 40000*3+256 = 120256
        assert plan_output_budget(40_000, 128_000) == 87_500

    def test_returns_none_at_the_fit_boundary(self) -> None:
        # output_cap == 256 (<= _MIN_OUTPUT_TOKENS) -> cannot fit -> None
        assert plan_output_budget(244, 1_000) is None
        # one token smaller -> output_cap == 257 -> fits
        assert plan_output_budget(243, 1_000) == 257

    def test_oversized_chunk_returns_none(self) -> None:
        assert plan_output_budget(200_000, 128_000) is None


class TestIsTransient:
    @pytest.mark.parametrize(
        "exc_type",
        [
            openai.APIConnectionError,
            openai.APITimeoutError,  # subclass of APIConnectionError
            openai.RateLimitError,
            openai.InternalServerError,
        ],
    )
    def test_transient_errors_are_retryable(self, exc_type: type) -> None:
        assert is_transient(exc_type.__new__(exc_type)) is True

    @pytest.mark.parametrize(
        "error",
        [
            ValueError("bad"),
            RuntimeError("unsplittable paragraph"),
            KeyError("missing"),
            openai.BadRequestError.__new__(openai.BadRequestError),
            openai.AuthenticationError.__new__(openai.AuthenticationError),
        ],
    )
    def test_permanent_errors_are_not_retryable(self, error: BaseException) -> None:
        assert is_transient(error) is False


class TestBackoffSeconds:
    def test_doubles_each_attempt(self) -> None:
        assert [backoff_seconds(a) for a in (1, 2, 3, 4)] == [3.0, 6.0, 12.0, 24.0]

    def test_caps_at_thirty(self) -> None:
        assert backoff_seconds(5) == 30.0
        assert backoff_seconds(10) == 30.0


class TestSplitChunk:
    def test_single_paragraph_is_not_split(self) -> None:
        assert split_chunk("one whole paragraph") == ["one whole paragraph"]

    def test_even_split_on_paragraph_boundaries(self) -> None:
        assert split_chunk("a\n\nb\n\nc\n\nd") == ["a\n\nb", "c\n\nd"]

    def test_odd_split_puts_remainder_in_second_half(self) -> None:
        assert split_chunk("a\n\nb\n\nc") == ["a", "b\n\nc"]


class TestBuildSystemPrompt:
    def test_names_the_target_language(self) -> None:
        prompt = build_system_prompt(Config(api_key="k", target_language="th"))
        assert "Thai" in prompt
        assert "literary translator" in prompt
