"""Application configuration loaded from the environment and the .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

__all__ = [
    "Config",
    "JudgeConfig",
    "load_config",
    "load_judge_config",
    "SUPPORTED_LANGUAGES",
]

# Supported target languages, mapping the configurable short code to the full
# language name (used for prompts, CLI help, and validation).
SUPPORTED_LANGUAGES = {
    "id": "Indonesian",
    "fil": "Filipino",
    "ta": "Tamil",
    "th": "Thai",
    "vi": "Vietnamese",
}

# Default target language code (must be a key in SUPPORTED_LANGUAGES).
DEFAULT_TARGET_LANGUAGE = "id"

# SEA-LION API connection details.
DEFAULT_BASE_URL = "https://api.sea-lion.ai/v1"
DEFAULT_MODEL = "aisingapore/Gemma-SEA-LION-v4-27B-IT"
DEFAULT_TIMEOUT = 120.0

# Assumed model context window (tokens). Optimistic default; the translator
# degrades (splits chunks) when a call fails, so an over-estimate is safe.
DEFAULT_CONTEXT_WINDOW = 128_000

# Source book to translate (Project Gutenberg plain-text URL).
DEFAULT_BOOK_URL = "https://www.gutenberg.org/cache/epub/16317/pg16317.txt"

# LLM-as-a-judge settings. The translation-quality judge runs on a *separate*
# provider (Google Gemini) so it never competes with SEA-LION's 10 RPM budget
# and never grades a model's own output. Gemini exposes an OpenAI-compatible
# endpoint, so the same AsyncOpenAI client works (see Decision 1).
JUDGE_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
JUDGE_DEFAULT_MODEL = "gemini-2.5-flash"
JUDGE_DEFAULT_TIMEOUT = 120.0

# Load variables from a .env file in the project root into os.environ.
# Real environment variables take precedence over .env (override=False).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Config:
    """Resolved, immutable application settings."""

    api_key: str
    target_language: str = DEFAULT_TARGET_LANGUAGE
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    context_window: int = DEFAULT_CONTEXT_WINDOW
    book_url: str = DEFAULT_BOOK_URL


@dataclass(frozen=True)
class JudgeConfig:
    """Resolved settings for the translation-quality judge (Gemini)."""

    api_key: str
    base_url: str = JUDGE_DEFAULT_BASE_URL
    model: str = JUDGE_DEFAULT_MODEL
    timeout: float = JUDGE_DEFAULT_TIMEOUT


def load_config(
    target_language: str | None = None,
    book_url: str | None = None,
) -> Config:
    """Build a Config from the environment plus command-line arguments.

    The API key and operational settings come from the environment (`.env`);
    the target language and source book are supplied on the command line.

    Args:
        target_language: Language code from the CLI; falls back to the default.
        book_url: Source book URL from the CLI; falls back to the default.

    Returns:
        A populated Config instance.

    Raises:
        RuntimeError: if SEA_LION_API_KEY is missing.
    """
    api_key = os.getenv("SEA_LION_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SEA_LION_API_KEY is not set. Add it to your .env file (see .env.example)."
        )

    return Config(
        api_key=api_key,
        target_language=target_language or DEFAULT_TARGET_LANGUAGE,
        base_url=os.getenv("SEA_LION_BASE_URL") or DEFAULT_BASE_URL,
        model=os.getenv("SEA_LION_MODEL") or DEFAULT_MODEL,
        timeout=float(os.getenv("SEA_LION_TIMEOUT") or DEFAULT_TIMEOUT),
        context_window=int(
            os.getenv("SEA_LION_CONTEXT_WINDOW") or DEFAULT_CONTEXT_WINDOW
        ),
        book_url=book_url or DEFAULT_BOOK_URL,
    )


def load_judge_config() -> JudgeConfig:
    """Build a JudgeConfig for the LLM-as-a-judge evaluation step.

    Reads the judge API key from the environment (`.env`), with optional env
    overrides for the base URL (`JUDGE_BASE_URL`), model (`JUDGE_MODEL`), and
    timeout (`JUDGE_TIMEOUT`).

    Returns:
        A populated JudgeConfig instance.

    Raises:
        RuntimeError: if GEMINI_API_KEY is missing.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env file "
            "(see .env.example) to use the --judge evaluation step."
        )

    return JudgeConfig(
        api_key=api_key,
        base_url=os.getenv("JUDGE_BASE_URL") or JUDGE_DEFAULT_BASE_URL,
        model=os.getenv("JUDGE_MODEL") or JUDGE_DEFAULT_MODEL,
        timeout=float(os.getenv("JUDGE_TIMEOUT") or JUDGE_DEFAULT_TIMEOUT),
    )
