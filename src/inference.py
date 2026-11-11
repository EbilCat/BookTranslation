"""Minimal SEA-LION inference helpers built directly on httpx.

The SEA-LION API is OpenAI-compatible, so we POST to the standard
``/chat/completions`` endpoint with a Bearer token and read the assistant
message back out of the JSON response.
"""

from __future__ import annotations

import httpx

from src.config import Config, load_config

__all__ = ["run_inference"]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def run_inference(prompt: str, config: Config) -> str:
    """Send a single prompt to SEA-LION and return the assistant's reply.

    Args:
        prompt: The user message to send.
        config: Resolved application settings (API key, model, base URL,
            request timeout).

    Returns:
        The text content of the model's response.

    Raises:
        httpx.HTTPStatusError: if the API returns a non-2xx status.
    """
    response = httpx.post(
        f"{config.base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=config.timeout,
    )
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def test_inference() -> str:
    """Smoke test: load config, query the model, and print the reply."""
    config = load_config()
    reply = run_inference("How many days are there in a year?", config)
    print(reply)
    return reply


if __name__ == "__main__":
    test_inference()
