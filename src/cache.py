"""On-disk cache of translated chunks for resumable translation.

Each source chunk's translation is written to its own file as soon as it
completes, so a run that is interrupted part-way (crash, Ctrl-C, rate-limit
give-up) can be restarted and will re-use the work already done instead of
re-spending the API budget on chunks it already translated.

The cache is *content-addressed*: a chunk's key is derived from its text plus
the translation parameters (target language and model), not from its position
in the book. This makes cache reuse correct by construction -- a hit is only
ever returned for byte-identical text translated under the same language and
model, so changing the language, the model, or the chunking (which changes the
text of each chunk) simply misses and re-translates rather than returning a
stale result. Reassembly stays trivial because the caller keeps the chunks in
order and looks each one up by its content.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

__all__ = ["ChunkCache"]

# Encoding for all cache files.
_ENCODING = "utf-8"

# Bump to invalidate every existing entry after a breaking change to how a
# translation is produced (e.g. a prompt rewrite that changes outputs).
_CACHE_VERSION = "1"

# Byte separator between key components so distinct inputs cannot collide by
# concatenation (e.g. "a" + "bc" vs "ab" + "c").
_KEY_SEPARATOR = b"\x00"


class ChunkCache:
    """Content-addressed disk cache mapping a source chunk to its translation.

    Writes are atomic (temp file in the same directory, then ``os.replace``),
    so a process killed mid-write never leaves a half-written file that a later
    run would mistake for a complete translation.

    Attributes are private; interact through :meth:`get` and :meth:`set`.
    """

    def __init__(self, directory: str | Path, *, language: str, model: str) -> None:
        """Open (creating if needed) a cache rooted at ``directory``.

        Args:
            directory: Folder to store cache files in; created if absent.
            language: Target language code; part of the cache key.
            model: Model identifier; part of the cache key.
        """
        self._dir = Path(directory)
        self._language = language
        self._model = model
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, chunk: str) -> str | None:
        """Return the cached translation for ``chunk``.

        Args:
            chunk: The source text whose translation is wanted.

        Returns:
            The previously cached translation, or ``None`` if there is no
            entry (or it could not be read).
        """
        path = self._path(chunk)
        try:
            return path.read_text(encoding=_ENCODING)
        except FileNotFoundError:
            return None
        except OSError:
            # An unreadable entry is treated as a miss; the chunk is re-done.
            return None

    def set(self, chunk: str, translation: str) -> None:
        """Atomically store ``translation`` as the result for ``chunk``.

        Args:
            chunk: The source text that was translated.
            translation: The translated text to cache.
        """
        path = self._path(chunk)
        # Write to a uniquely named temp file in the same directory, then
        # atomically rename it into place (atomic on POSIX and Windows when
        # source and destination share a filesystem, which they do here).
        fd, tmp_name = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding=_ENCODING, newline="\n") as handle:
                handle.write(translation)
            os.replace(tmp_name, path)
        except BaseException:
            # Clean up the temp file on any failure (including cancellation).
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def _key(self, chunk: str) -> str:
        """Derive the content-addressed cache key (hex digest) for ``chunk``."""
        digest = hashlib.sha256()
        for part in (_CACHE_VERSION, self._language, self._model, chunk):
            digest.update(part.encode(_ENCODING))
            digest.update(_KEY_SEPARATOR)
        return digest.hexdigest()

    def _path(self, chunk: str) -> Path:
        """Return the on-disk path of the cache file for ``chunk``."""
        return self._dir / f"{self._key(chunk)}.txt"
