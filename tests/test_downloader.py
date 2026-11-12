"""Tests for src.downloader pure helpers (URL-keyed source cache path)."""

from __future__ import annotations

from pathlib import Path

from src.downloader import source_cache_path


class TestSourceCachePath:
    def test_is_deterministic_for_the_same_url(self) -> None:
        a = source_cache_path("https://example.com/book.txt", "processing")
        b = source_cache_path("https://example.com/book.txt", "processing")
        assert a == b

    def test_distinct_urls_get_distinct_files(self) -> None:
        a = source_cache_path("https://example.com/a.txt", "processing")
        b = source_cache_path("https://example.com/b.txt", "processing")
        assert a != b

    def test_lives_under_the_cache_dir_with_expected_name(self) -> None:
        path = source_cache_path("https://example.com/book.txt", "processing")
        assert path.parent == Path("processing")
        assert path.name.startswith("source_")
        assert path.suffix == ".txt"

    def test_accepts_a_path_cache_dir(self, tmp_path: Path) -> None:
        path = source_cache_path("https://example.com/book.txt", tmp_path)
        assert path.parent == tmp_path
