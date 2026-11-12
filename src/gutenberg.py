"""Project Gutenberg-specific preprocessing.

Isolated from the generic chunking/translation core so the rest of the app can
translate any plain text. Only the application layer (main.py) opts into this.
"""

from __future__ import annotations

import re

__all__ = ["strip_boilerplate"]

# Project Gutenberg wraps the actual work between START/END marker lines. The
# format has varied over time, so accept: optional space after the asterisks,
# "THE" or "THIS", and "EBOOK" or (legacy) "ETEXT"; match case-insensitively.
# Staying specific to "... GUTENBERG EBOOK/ETEXT" avoids matching the
# "START: FULL LICENSE" line inside the trailing license boilerplate.
_START_MARKER = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG (?:EBOOK|ETEXT).*?\*\*\*",
    re.IGNORECASE,
)
_END_MARKER = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG (?:EBOOK|ETEXT).*?\*\*\*",
    re.IGNORECASE,
)


def strip_boilerplate(text: str) -> str:
    """Return the text between the Gutenberg START and END markers.

    Falls back to the full (stripped) text if either marker is absent, so this
    is safe to call on non-Gutenberg input (it becomes a no-op). Legacy
    pre-2004 "small print" formats are not handled.

    Args:
        text: Full plain-text document, possibly with Gutenberg boilerplate.

    Returns:
        The body text with surrounding Gutenberg boilerplate removed.
    """
    start = _START_MARKER.search(text)
    end = _END_MARKER.search(text)
    begin = start.end() if start else 0
    finish = end.start() if end else len(text)
    return text[begin:finish].strip()
