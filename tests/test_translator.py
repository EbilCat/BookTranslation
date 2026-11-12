"""Tests for the pure stop-or-retry decision in src.translator."""

from __future__ import annotations

import openai
import pytest

from src.translator import TranslationFailedError, _plan_retry


def _transient() -> BaseException:
    """A transient (retryable) error instance without invoking __init__."""
    return openai.RateLimitError.__new__(openai.RateLimitError)


class TestPlanRetry:
    def test_transient_before_final_pass_returns_indices_to_retry(self) -> None:
        failures = [(0, _transient()), (2, _transient())]
        assert _plan_retry(failures, "Thai", attempt=1, max_attempts=3) == [0, 2]

    def test_transient_on_final_pass_raises_recoverable(self) -> None:
        failures = [(1, _transient())]
        with pytest.raises(TranslationFailedError) as excinfo:
            _plan_retry(failures, "Thai", attempt=3, max_attempts=3)
        error = excinfo.value
        assert error.recoverable is True
        assert error.failed_indices == [1]

    def test_permanent_failure_raises_unrecoverable_immediately(self) -> None:
        failures = [(1, _transient()), (4, ValueError("unsplittable"))]
        with pytest.raises(TranslationFailedError) as excinfo:
            _plan_retry(failures, "Thai", attempt=1, max_attempts=3)
        error = excinfo.value
        assert error.recoverable is False
        assert error.failed_indices == [1, 4]  # sorted
