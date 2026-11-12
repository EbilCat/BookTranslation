"""Async SEA-LION inference helpers built on the OpenAI SDK.

The SEA-LION API is OpenAI-compatible, so we use ``AsyncOpenAI`` pointed at the
SEA-LION base URL. The async client lets the translator run many calls
concurrently (bounded by a rate limiter), and its built-in, Retry-After-aware
backoff handles transient failures.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from src.config import Config

__all__ = ["build_client", "run_inference", "TranslationTruncatedError"]

# Client-side retries for transient errors (429/5xx); the SDK honours
# Retry-After and backs off exponentially between attempts.
_MAX_RETRIES = 5


class TranslationTruncatedError(RuntimeError):
    """Raised when the model stops because it hit the output token cap."""


def build_client(config: Config) -> AsyncOpenAI:
    """Create an AsyncOpenAI client pointed at the SEA-LION API."""
    return AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout,
        max_retries=_MAX_RETRIES,
    )


async def run_inference(
    client: AsyncOpenAI,
    prompt: str,
    config: Config,
    *,
    system: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Send one chat completion and return the assistant's reply text.

    Args:
        client: An AsyncOpenAI client (see :func:`build_client`).
        prompt: The user message to send.
        config: Resolved application settings (provides the model).
        system: Optional system message prepended to the conversation.
        max_tokens: Optional cap on output tokens. When the response is cut off
            at this cap, a TranslationTruncatedError is raised.

    Returns:
        The text content of the model's response.

    Raises:
        TranslationTruncatedError: if the response was truncated at max_tokens.
        openai.OpenAIError: on an API error the client could not recover from.
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = await client.chat.completions.create(
        model=config.model,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=max_tokens,
    )
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise TranslationTruncatedError(
            f"Response truncated at max_tokens={max_tokens}."
        )
    return choice.message.content or ""
