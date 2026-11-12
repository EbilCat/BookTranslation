"""Tests for src.gutenberg (boilerplate stripping)."""

from __future__ import annotations

from src.gutenberg import strip_boilerplate

_START = "*** START OF THE PROJECT GUTENBERG EBOOK FOO ***"
_END = "*** END OF THE PROJECT GUTENBERG EBOOK FOO ***"


class TestStripBoilerplate:
    def test_returns_text_between_markers(self) -> None:
        text = f"front matter\n{_START}\nTHE BODY\n{_END}\nlicense text"
        assert strip_boilerplate(text) == "THE BODY"

    def test_is_a_no_op_without_markers(self) -> None:
        text = "just some plain text\n\nwith paragraphs"
        assert strip_boilerplate(text) == text

    def test_accepts_this_etext_and_mixed_case_variants(self) -> None:
        start = "*** start of this project gutenberg etext bar ***"
        end = "*** end of this project gutenberg etext bar ***"
        assert strip_boilerplate(f"x\n{start}\nBODY\n{end}\ny") == "BODY"

    def test_strips_surrounding_whitespace_of_body(self) -> None:
        text = f"{_START}\n\n   BODY   \n\n{_END}"
        assert strip_boilerplate(text) == "BODY"

    def test_missing_end_marker_falls_back_to_rest_of_text(self) -> None:
        # No END marker: body runs from after START to the end.
        assert strip_boilerplate(f"pre\n{_START}\nBODY tail") == "BODY tail"
