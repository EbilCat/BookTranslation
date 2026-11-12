"""Command-line entry point: download a book and translate it via SEA-LION.

Pipeline: parse args -> load config -> download (cached) -> translate the
chunks concurrently -> write the translated text to a local file. With
``--judge``, a final LLM-as-a-judge step scores a sample of the translated
chunks (see :mod:`src.evaluator`) and writes a quality report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from src.config import SUPPORTED_LANGUAGES, load_config, load_judge_config
from src.downloader import download_to_file
from src.evaluator import DEFAULT_SAMPLE_SIZE, evaluate_translation
from src.gutenberg import strip_boilerplate
from src.translator import TranslationFailedError, translate_text

# Where the downloaded source book is cached, and the default output template.
_SOURCE_CACHE = "output/source.txt"
_OUTPUT_TEMPLATE = "output/{language}/translated_{language}.txt"

# Default location (without extension) for the judge's quality report.
_JUDGE_REPORT_TEMPLATE = "output/{language}/judge_report"

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
    parser.add_argument(
        "--judge",
        action="store_true",
        help="After translating, score a sample of chunks with the Gemini "
        "LLM-as-a-judge and write a quality report (needs GEMINI_API_KEY).",
    )
    parser.add_argument(
        "--judge-sample",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        metavar="N",
        help=f"Number of chunks the judge scores (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--judge-seed",
        type=int,
        default=None,
        metavar="SEED",
        help="RNG seed for a reproducible judge sample (default: random).",
    )
    args = parser.parse_args(argv)
    if args.judge and args.no_cache:
        parser.error(
            "--judge reads translations from the per-chunk cache, so it cannot "
            "be combined with --no-cache."
        )
    return args


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

    if args.judge:
        await _judge(args, config, text)


async def _judge(args: argparse.Namespace, config, text: str) -> None:
    """Run the LLM-as-a-judge evaluation and write a quality report."""
    judge_config = load_judge_config()
    language = SUPPORTED_LANGUAGES[config.target_language]
    print(f"Judging up to {args.judge_sample} chunks with {judge_config.model} ...")
    report = await evaluate_translation(
        text,
        config,
        judge_config,
        cache_dir=_TRANSLATION_CACHE_DIR,
        sample_size=args.judge_sample,
        seed=args.judge_seed,
    )

    report_base = Path(_JUDGE_REPORT_TEMPLATE.format(language=config.target_language))
    report_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = report_base.with_suffix(".json")
    md_path = report_base.with_suffix(".md")
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    md_path.write_text(report.to_markdown(), encoding="utf-8", newline="\n")

    print(
        f"Judge: overall {report.overall_mean():.2f}/5 "
        f"across {report.sample_size} chunks ({language}). "
        f"Reports: {md_path}, {json_path}"
    )


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
