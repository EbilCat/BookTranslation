"""Command-line entry point: download a book and translate it via SEA-LION.

Pipeline: parse args -> load config -> download (cached) -> translate the
chunks concurrently -> write the translated text to a local file.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from src.config import SUPPORTED_LANGUAGES, load_config
from src.downloader import download_to_file
from src.gutenberg import strip_boilerplate
from src.translator import TranslationFailedError, translate_text

# Where the downloaded source book is cached, and the default output template.
_SOURCE_CACHE = "output/source.txt"
_OUTPUT_TEMPLATE = "output/{language}/translated_{language}.txt"

# Directory for the resumable per-chunk translation cache (git-ignored).
_TRANSLATION_CACHE_DIR = "processing/translation_cache"

# Default number of whole-book translation passes (initial + retries).
_DEFAULT_MAX_ATTEMPTS = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    language_help = "Target language: " + ", ".join(
        f"{code} ({name})" for code, name in SUPPORTED_LANGUAGES.items()
    )
    parser = argparse.ArgumentParser(
        description="Translate a public-domain book using the SEA-LION API.",
    )
    parser.add_argument(
        "-l",
        "--language",
        choices=sorted(SUPPORTED_LANGUAGES),
        default=None,
        metavar="CODE",
        help=f"{language_help} (default: id).",
    )
    parser.add_argument(
        "-u",
        "--book-url",
        default=None,
        help="Source book URL (default: the configured Gutenberg book).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path (default: output/<language>/translated_<language>.txt).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the book even if a cached copy exists.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Translate the text as-is, without stripping Gutenberg boilerplate.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the resumable per-chunk translation cache "
        "(translate every chunk afresh).",
    )
    parser.add_argument(
        "--max-attempts",
        type=_positive_int,
        default=_DEFAULT_MAX_ATTEMPTS,
        metavar="N",
        help="Whole-book translation passes before giving up "
        f"(initial + retries; default: {_DEFAULT_MAX_ATTEMPTS}).",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Run the book-translation pipeline from command-line arguments."""
    args = parse_args()
    try:
        asyncio.run(_run(args))
    except TranslationFailedError as error:
        print(f"\nError: {error}", file=sys.stderr)
        print(
            "No output file written. Cached progress is preserved — rerun to resume.",
            file=sys.stderr,
        )
        raise SystemExit(1) from error


async def _run(args: argparse.Namespace) -> None:
    """Execute the download -> translate -> write pipeline."""
    config = load_config(target_language=args.language, book_url=args.book_url)
    language = SUPPORTED_LANGUAGES[config.target_language]

    print(f"Downloading book from {config.book_url} ...")
    text = download_to_file(
        config.book_url,
        _SOURCE_CACHE,
        force=args.force_download,
        timeout=config.timeout,
    )
    if not args.raw:
        text = strip_boilerplate(text)
    print(f"Translating {len(text):,} characters to {language} ...")

    cache_dir = None if args.no_cache else _TRANSLATION_CACHE_DIR
    translated = await translate_text(
        text, config, cache_dir=cache_dir, max_attempts=args.max_attempts
    )

    output_path = Path(
        args.output or _OUTPUT_TEMPLATE.format(language=config.target_language)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(translated, encoding="utf-8", newline="\n")
    print(f"Wrote translation to {output_path}")


def _positive_int(value: str) -> int:
    """Parse a CLI integer that must be >= 1 (argparse ``type``)."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer.") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1.")
    return number


if __name__ == "__main__":
    main()
