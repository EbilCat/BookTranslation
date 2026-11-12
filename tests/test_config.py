"""Tests for src.config (settings, validation, env loading)."""

from __future__ import annotations

import pytest

from src.config import (
    DEFAULT_MODEL,
    Config,
    JudgeConfig,
    load_config,
    load_judge_config,
)


class TestConfigValidation:
    def test_default_language_is_indonesian(self) -> None:
        assert Config(api_key="k").language_name == "Indonesian"

    def test_language_name_resolves_supported_code(self) -> None:
        assert Config(api_key="k", target_language="th").language_name == "Thai"

    def test_unsupported_language_raises_with_valid_codes(self) -> None:
        with pytest.raises(ValueError, match="Unsupported target language"):
            Config(api_key="k", target_language="xx")

    def test_error_message_lists_the_supported_codes(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            Config(api_key="k", target_language="xx")
        message = str(excinfo.value)
        assert "th" in message and "vi" in message


class TestLoadConfig:
    def test_reads_api_key_and_applies_cli_overrides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEA_LION_API_KEY", "secret")
        monkeypatch.delenv("SEA_LION_MODEL", raising=False)
        config = load_config(target_language="vi", book_url="http://example/book")
        assert config.api_key == "secret"
        assert config.target_language == "vi"
        assert config.book_url == "http://example/book"
        assert config.model == DEFAULT_MODEL

    def test_env_overrides_are_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEA_LION_API_KEY", "secret")
        monkeypatch.setenv("SEA_LION_MODEL", "custom-model")
        monkeypatch.setenv("SEA_LION_TIMEOUT", "42")
        config = load_config()
        assert config.model == "custom-model"
        assert config.timeout == 42.0
        assert config.target_language == "id"  # default when None

    def test_missing_api_key_raises_runtimeerror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SEA_LION_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="SEA_LION_API_KEY"):
            load_config()


class TestLoadJudgeConfig:
    def test_reads_gemini_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "gkey")
        judge = load_judge_config()
        assert isinstance(judge, JudgeConfig)
        assert judge.api_key == "gkey"

    def test_missing_gemini_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            load_judge_config()
