"""Judge rubric: the grading spec, prompt construction, and reply parsing.

Pure and I/O-free. Defines the scoring dimensions and range, builds the judge's
system and user prompts, and parses the judge's JSON reply back into score and
note maps (tolerating code fences and flagging malformed output). Kept separate
from the async judging in :mod:`src.evaluator` so every grading decision is
unit-testable without a live judge call.
"""

from __future__ import annotations

import json
import re

__all__ = [
    "DIMENSIONS",
    "SCORE_MAX",
    "SCORE_MIN",
    "clamp_score",
    "extract_json",
    "judge_prompt",
    "judge_system",
    "parse_judgement",
]

# Rubric dimensions (key -> what the judge is asked to assess). Order is the
# display order in the report.
DIMENSIONS: dict[str, str] = {
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
SCORE_MIN = 1
SCORE_MAX = 5


def judge_system(language: str) -> str:
    """Build the judge system prompt for the configured target language."""
    rubric = "\n".join(f"- {key}: {desc}" for key, desc in DIMENSIONS.items())
    keys = ", ".join(f'"{key}"' for key in DIMENSIONS)
    return (
        f"You are a meticulous, impartial translation-quality evaluator for "
        f"English-to-{language} literary translation. You are given a source "
        f"passage and its translation. Score the translation on each dimension "
        f"from {SCORE_MIN} (unacceptable) to {SCORE_MAX} (excellent):\n"
        f"{rubric}\n\n"
        f"Respond with ONLY a JSON object (no markdown fences, no prose) whose "
        f"keys are exactly {keys}. Each value is an object "
        f'{{"score": <int>, "note": "<one concise sentence>"}}. '
        f"Base every score strictly on the texts shown; do not reward or "
        f"penalise length. Keep each note under 25 words."
    )


def judge_prompt(source: str, translation: str, language: str) -> str:
    """Build the judge user message carrying the source/translation pair."""
    return f"SOURCE (English):\n{source}\n\nTRANSLATION ({language}):\n{translation}"


def parse_judgement(content: str) -> tuple[dict[str, int], dict[str, str]]:
    """Parse the judge's JSON reply into score and note maps.

    Raises:
        ValueError: if no JSON object can be extracted or a dimension/score is
            malformed.
        KeyError: if a required dimension is absent.
    """
    data = extract_json(content)
    scores: dict[str, int] = {}
    notes: dict[str, str] = {}
    for dimension in DIMENSIONS:
        entry = data[dimension]  # KeyError if a dimension is missing.
        if not isinstance(entry, dict) or "score" not in entry:
            raise ValueError(f"Malformed entry for dimension '{dimension}'.")
        scores[dimension] = clamp_score(entry["score"])
        notes[dimension] = str(entry.get("note", "")).strip()
    return scores, notes


def extract_json(content: str) -> dict[str, object]:
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


def clamp_score(value: object) -> int:
    """Coerce ``value`` to an int clamped to the valid score range."""
    try:
        # ``value`` is an arbitrary JSON value; int() raises on anything not
        # coercible, which the except turns into a clear ValueError.
        score = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError) as error:
        raise ValueError(f"Score {value!r} is not an integer.") from error
    return max(SCORE_MIN, min(SCORE_MAX, score))
