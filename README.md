# Book Translation

Translate a public-domain book into a Southeast Asian language using the
[SEA-LION API](https://docs.sea-lion.ai/guides/inferencing/api).

The app downloads a plain-text book (defaults to a
[Project Gutenberg](https://www.gutenberg.org/cache/epub/16317/pg16317.txt)
title), strips the Gutenberg boilerplate, splits it into paragraph-aligned
chunks, translates the chunks concurrently (rate-limited and resumable), and
writes the translated book to a local file. An optional
[LLM-as-a-judge](#optional-translation-quality-evaluation) step scores a sample
of the translation for quality.

Supported target languages: **Indonesian** (`id`, default), **Filipino**
(`fil`), **Tamil** (`ta`), **Thai** (`th`), **Vietnamese** (`vi`).

## Requirements

- **Python 3.13+** (pinned in `pyproject.toml`)
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- A **SEA-LION API key** — get one at
  https://docs.sea-lion.ai/guides/inferencing/api

## Setup

```bash
# 1. Install dependencies into a local virtual environment
uv sync

# 2. Create your .env from the template and add your key(s)
cp .env.example .env
#   then edit .env and set SEA_LION_API_KEY=...
```

`.env` is git-ignored — never commit real secrets.

## Usage

Run with `uv run`, which uses the project's virtual environment:

```bash
# Translate the default book into the default language (Indonesian)
uv run main.py

# Choose a target language (configurable, as required)
uv run main.py --language th

# Translate a different plain-text book
uv run main.py --language vi --book-url https://www.gutenberg.org/cache/epub/1342/pg1342.txt
```

The translated book is written to
`output/<language>/translated_<language>.txt` (e.g. `output/th/translated_th.txt`).

### Options

| Flag | Default | Description |
| --- | --- | --- |
| `-l`, `--language CODE` | `id` | Target language: `id`, `fil`, `ta`, `th`, `vi`. |
| `-u`, `--book-url URL` | the Gutenberg default | Source book (plain-text URL). |
| `-o`, `--output PATH` | `output/<language>/translated_<language>.txt` | Output file path. |
| `--force-download` | off | Re-download the book even if a cached copy exists. |
| `--raw` | off | Translate as-is, without stripping Gutenberg boilerplate. |
| `--no-cache` | off | Disable the resumable per-chunk cache (translate every chunk afresh). |
| `--max-attempts N` | `3` | Whole-book translation passes before giving up (retries transient failures). |
| `--judge` | off | After translating, score a sample of chunks (see below). |
| `--judge-sample N` | `8` | Number of chunks the judge scores. |
| `--judge-seed SEED` | random | RNG seed for a reproducible judge sample. |

Translation is **resumable**: each chunk's result is cached on disk as it
completes, so an interrupted run (crash, Ctrl-C, rate-limit give-up) resumes
instead of re-translating — and re-paying for — chunks already done. If chunks
still fail after `--max-attempts` passes, no output file is written and the
cached progress is preserved for a later rerun.

### Where files go

| Path | Contents |
| --- | --- |
| `output/<language>/translated_<language>.txt` | The translated book (deliverable). |
| `output/<language>/judge_report.{json,md}` | Quality report, with `--judge`. |
| `processing/source_<hash>.txt` | The downloaded source book, cached per URL. |
| `processing/translation_cache/` | Per-chunk resume cache. |

Both `output/` and `processing/` are git-ignored.

## Optional: translation-quality evaluation

With `--judge`, a separate judge model (Google Gemini, via its
OpenAI-compatible endpoint) scores a random sample of the translated chunks
against the English source on a per-dimension rubric (accuracy, fluency,
completeness, tone; 1–5). Using a different provider avoids self-evaluation
bias and keeps off SEA-LION's rate budget.

It requires a **`GEMINI_API_KEY`** in `.env` (free tier works; get one at
https://aistudio.google.com/apikey) and reads the translations from the chunk
cache, so it cannot be combined with `--no-cache`.

```bash
uv run main.py --language th --judge --judge-sample 10
```

The report is written to `output/<language>/judge_report.json` and
`.md`.

## Configuration

Secrets and operational settings are read from the environment (`.env`). The
two keys above are all you need; the rest are optional overrides with sensible
defaults. See [`.env.example`](.env.example) for the full list (base URLs,
timeouts, model, context window, and the judge model/endpoint). Target language
and book URL are **CLI arguments**, not env vars.

## Project structure

```text
BookTranslation/
├── main.py                    # CLI entry point: download → translate → write (+ optional judge)
├── src/
│   ├── config.py              # Settings & secrets from env/.env; target-language validation
│   ├── downloader.py          # Fetch a plain-text book, with an on-disk cache
│   ├── gutenberg.py           # Strip Project Gutenberg START/END boilerplate
│   ├── chunker.py             # Split text into paragraph-aligned, token-budgeted chunks
│   ├── ratelimit.py           # Async limiter capping request starts per rolling minute
│   ├── inference.py           # AsyncOpenAI client + one SEA-LION chat completion
│   ├── translation_policy.py  # Pure translator decisions (token budget, retry/backoff, split)
│   ├── translator.py          # Orchestrates concurrent, resumable, retrying translation
│   ├── cache.py               # Content-addressed on-disk cache of translated chunks
│   ├── judge_rubric.py        # Judge grading spec, prompts & reply parsing (pure)
│   ├── evaluation_report.py   # Evaluation data model + JSON/Markdown rendering (pure)
│   └── evaluator.py           # LLM-as-a-judge orchestration (Gemini)
├── tests/                     # Offline unit tests for the pure logic (pytest)
├── .env.example               # Template for required/optional environment variables
├── pyproject.toml             # Project metadata, dependencies & tooling config
└── uv.lock                    # Pinned dependency lockfile
```

Created at runtime and git-ignored:

```text
output/        # translated book + judge reports (the deliverables)
processing/    # downloaded source book + per-chunk resume cache
```

## How it works

The code follows a *pure core, imperative shell* split so the decision logic is
testable without the network:

| Module | Responsibility | I/O? |
| --- | --- | --- |
| `src/config.py` | Resolve settings & secrets from env/`.env`; validate the target language | pure |
| `src/downloader.py` | Fetch a plain-text book, with an on-disk cache | network / disk |
| `src/gutenberg.py` | Strip Project Gutenberg START/END boilerplate | pure |
| `src/chunker.py` | Split text into paragraph-aligned, token-budgeted chunks | pure |
| `src/ratelimit.py` | Cap request starts per rolling minute (shared limiter) | pure (async) |
| `src/inference.py` | One SEA-LION chat completion via `AsyncOpenAI` | network |
| `src/translation_policy.py` | Token budget, retry/backoff, and chunk-split decisions | pure |
| `src/translator.py` | Orchestrate concurrent, resumable, retrying translation | shell |
| `src/cache.py` | Content-addressed on-disk cache of translated chunks | disk |
| `src/judge_rubric.py` | Judge grading spec, prompts, and reply parsing | pure |
| `src/evaluation_report.py` | Evaluation data model + JSON/Markdown rendering | pure |
| `src/evaluator.py` | LLM-as-a-judge orchestration (Gemini) | network |
| `main.py` | Wire the pipeline together | shell |

## Design decisions

A few choices worth calling out, with the reasoning behind them:

- **The rate limit, not the context window, is the real bottleneck.** SEA-LION
  allows ~10 requests/minute, and translating a ~240K-token book one paragraph
  per call would take hours. The chunker greedily batches whole paragraphs
  (never splitting one mid-sentence) up to a token budget to minimise call
  count, and the translator runs chunks concurrently under a shared limiter that
  caps request *starts* at 10/min — saturating the budget instead of wasting it
  on slow sequential calls.

- **Resumable, content-addressed cache.** Each chunk's translation is cached on
  disk keyed by `sha256(cache_version, language, model, chunk_text)` — not by
  position. A hit is therefore only possible for a byte-identical request, so
  changing the language, model, or chunking misses and re-translates rather than
  silently reusing a stale result. Writes are atomic (temp file + `os.replace`),
  so an interrupted run resumes cleanly and never trusts a half-written file.

- **Degrade on failure, and fail loud — never a silently-incomplete book.** A
  chunk that truncates or won't fit is split and retried; the whole book runs in
  passes so one flaky chunk doesn't sink it, retrying only transient failures
  (timeout / 429 / 5xx) and aborting immediately on permanent ones (auth, a
  single unsplittable paragraph). If any chunk is still missing after the final
  pass, the run exits non-zero and writes **no** output file — a loud failure is
  safer than a translation that looks complete but has gaps.

- **Reference-free LLM-as-a-judge on a separate provider.** The optional
  `--judge` step scores quality with Google Gemini rather than SEA-LION: a
  different provider avoids self-evaluation bias (a model grading its own output
  skews lenient) and keeps judging off the translation rate budget. It samples
  chunks and scores a per-dimension rubric (accuracy, fluency, completeness,
  tone), which is more diagnostic than a single number.

## Development

Run the test suite (all tests are offline — no API key or network required):

```bash
uv run pytest
```

The suite covers the pure, I/O-free logic (chunking, token budgeting, retry
classification, judge-reply parsing, score aggregation, config validation,
caching).

Lint, format, and type-check:

```bash
uv run ruff check .      # lint + import sort
uv run ruff format .     # format
uv run mypy              # type-checks the shipped code (src/ and main.py)
```

Continuous integration (`.github/workflows/ci.yml`) runs `ruff check`,
`ruff format --check`, mypy, and pytest on every push and pull request against
Python 3.13.
