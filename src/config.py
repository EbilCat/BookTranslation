"""Application configuration loaded from the environment and the .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

__all__ = ["Config", "load_config"]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_TARGET_LANGUAGE = "id"

# SEA-LION API connection details.
DEFAULT_BASE_URL = "https://api.sea-lion.ai/v1"
DEFAULT_MODEL = "aisingapore/Gemma-SEA-LION-v4-27B-IT"
DEFAULT_TIMEOUT = 120.0

# Source book to translate (Project Gutenberg plain-text URL).
DEFAULT_BOOK_URL = "https://www.gutenberg.org/cache/epub/16317/pg16317.txt"

# Load variables from a .env file in the project root into os.environ.
# Real environment variables take precedence over .env (override=False).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    """Resolved, immutable application settings."""

    api_key: str
    target_language: str = DEFAULT_TARGET_LANGUAGE
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    book_url: str = DEFAULT_BOOK_URL


def load_config(target_language: str | None = None) -> Config:
    """Build a Config, reading secrets and settings from the environment.

    Args:
        target_language: Language code from the CLI; falls back to the
            TARGET_LANGUAGE env var, then the default.

    Returns:
        A populated Config instance.

    Raises:
        RuntimeError: if SEA_LION_API_KEY is missing.
    """
    api_key = os.getenv("SEA_LION_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SEA_LION_API_KEY is not set. Add it to your .env file "
            "(see .env.example)."
        )

    language = (
        target_language
        or os.getenv("TARGET_LANGUAGE")
        or DEFAULT_TARGET_LANGUAGE
    )
    return Config(
        api_key=api_key,
        target_language=language,
        base_url=os.getenv("SEA_LION_BASE_URL") or DEFAULT_BASE_URL,
        model=os.getenv("SEA_LION_MODEL") or DEFAULT_MODEL,
        timeout=float(os.getenv("SEA_LION_TIMEOUT") or DEFAULT_TIMEOUT),
        book_url=os.getenv("BOOK_URL") or DEFAULT_BOOK_URL,
    )
