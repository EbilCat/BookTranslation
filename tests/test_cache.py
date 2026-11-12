"""Tests for src.cache (content-addressed resumable chunk cache)."""

from __future__ import annotations

from pathlib import Path

from src.cache import ChunkCache


class TestChunkCache:
    def test_miss_returns_none(self, tmp_path: Path) -> None:
        cache = ChunkCache(tmp_path, language="th", model="m")
        assert cache.get("unseen chunk") is None

    def test_set_then_get_round_trips(self, tmp_path: Path) -> None:
        cache = ChunkCache(tmp_path, language="th", model="m")
        cache.set("source chunk", "คำแปล")
        assert cache.get("source chunk") == "คำแปล"

    def test_set_overwrites_previous_value(self, tmp_path: Path) -> None:
        cache = ChunkCache(tmp_path, language="th", model="m")
        cache.set("chunk", "first")
        cache.set("chunk", "second")
        assert cache.get("chunk") == "second"

    def test_different_chunk_text_misses(self, tmp_path: Path) -> None:
        cache = ChunkCache(tmp_path, language="th", model="m")
        cache.set("chunk a", "A")
        assert cache.get("chunk b") is None

    def test_key_is_sensitive_to_language(self, tmp_path: Path) -> None:
        ChunkCache(tmp_path, language="th", model="m").set("chunk", "thai")
        other = ChunkCache(tmp_path, language="vi", model="m")
        assert other.get("chunk") is None

    def test_key_is_sensitive_to_model(self, tmp_path: Path) -> None:
        ChunkCache(tmp_path, language="th", model="m1").set("chunk", "v1")
        other = ChunkCache(tmp_path, language="th", model="m2")
        assert other.get("chunk") is None

    def test_persists_across_cache_instances(self, tmp_path: Path) -> None:
        ChunkCache(tmp_path, language="th", model="m").set("chunk", "value")
        reopened = ChunkCache(tmp_path, language="th", model="m")
        assert reopened.get("chunk") == "value"

    def test_creates_missing_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "cache"
        cache = ChunkCache(nested, language="th", model="m")
        cache.set("chunk", "value")
        assert nested.is_dir()
        assert cache.get("chunk") == "value"
