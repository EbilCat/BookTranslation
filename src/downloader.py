"""General-purpose text downloader.

Fetches a plain-text resource over HTTP and saves it to a given path, with an
optional on-disk cache so repeated runs don't re-download the same URL.
"""

from __future__ import annotations

from pathlib import Path

import httpx

__all__ = ["download", "download_to_file"]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def download(url: str, timeout: float = 60.0) -> str:
    r"""Download a plain-text resource and return its contents.

    Line endings are normalized to ``\n`` so the result is independent of the
    source's line-ending style.

    Args:
        url: URL to fetch.
        timeout: Per-request timeout in seconds.

    Returns:
        The decoded text of the response body, with LF line endings.

    Raises:
        httpx.HTTPStatusError: if the server returns a non-2xx status.
    """
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return _normalize_newlines(response.text)


def download_to_file(
    url: str,
    dest: str | Path,
    *,
    force: bool = False,
    timeout: float = 60.0,
) -> str:
    """Download ``url`` to ``dest``, caching the result on disk.

    If ``dest`` already exists and ``force`` is False, the cached copy is read
    from disk instead of downloading again.

    Args:
        url: URL to fetch.
        dest: Local path to read from / write the downloaded text to.
        force: Re-download even if the cached file already exists.
        timeout: Per-request timeout in seconds.

    Returns:
        The decoded text, whether freshly downloaded or read from cache.

    Raises:
        httpx.HTTPStatusError: if a download is attempted and fails.
    """
    path = Path(dest)
    if path.exists() and not force:
        # newline="" disables read-time translation so we control normalization.
        cached = path.read_text(encoding="utf-8", newline="")
        return _normalize_newlines(cached)

    text = download(url, timeout=timeout)
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" prevents the OS from rewriting LF to CRLF on write.
    path.write_text(text, encoding="utf-8", newline="\n")
    return text


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _normalize_newlines(text: str) -> str:
    r"""Normalize CRLF and lone CR line endings to LF (``\n``)."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    """Download the configured source book into ``output/source.txt``."""
    from src.config import load_config

    config = load_config()
    text = download_to_file(
        config.book_url, "output/source.txt", timeout=config.timeout
    )
    print(f"Downloaded {len(text):,} characters from {config.book_url}")


if __name__ == "__main__":
    main()
