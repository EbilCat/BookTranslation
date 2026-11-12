"""Tests for src.chunker (paragraph-aware token batching)."""

from __future__ import annotations

from src.chunker import chunk_text, estimate_tokens


class TestEstimateTokens:
    def test_four_chars_per_token(self) -> None:
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("a" * 20) == 5

    def test_empty_text_is_zero(self) -> None:
        assert estimate_tokens("") == 0


class TestChunkText:
    def test_empty_text_yields_no_chunks(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("   \n\n  ") == []

    def test_paragraphs_are_stripped(self) -> None:
        assert chunk_text("  hello  ", max_tokens=1000) == ["hello"]

    def test_batches_whole_paragraphs_up_to_budget(self) -> None:
        # Each 4-char paragraph is ~1 token; budget 2 packs two per chunk.
        text = "aaaa\n\nbbbb\n\ncccc\n\ndddd"
        assert chunk_text(text, max_tokens=2) == ["aaaa\n\nbbbb", "cccc\n\ndddd"]

    def test_never_splits_a_single_paragraph(self) -> None:
        # One oversized paragraph (2 tokens) exceeds the 1-token budget but is
        # kept whole as its own chunk.
        assert chunk_text("aaaaaaaa", max_tokens=1) == ["aaaaaaaa"]

    def test_preserves_reading_order(self) -> None:
        # Paragraphs of >= 4 chars each estimate to >= 1 token, so a 1-token
        # budget puts each in its own chunk, in order.
        text = "alfa\n\nbravo\n\ndelta"
        assert chunk_text(text, max_tokens=1) == ["alfa", "bravo", "delta"]

    def test_joins_paragraphs_in_a_chunk_with_blank_line(self) -> None:
        assert chunk_text("aaaa\n\nbbbb", max_tokens=1000) == ["aaaa\n\nbbbb"]
