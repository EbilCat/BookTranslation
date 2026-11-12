"""Async rate limiter shared by the translator and the judge.

Both the SEA-LION translation calls and the Gemini judge calls go through a
provider that caps requests per minute, so each caller keeps its own limiter
instance (a separate rolling window per provider) built from this one class.
"""

from __future__ import annotations

import asyncio
from collections import deque

__all__ = ["RateLimiter"]


class RateLimiter:
    """Async limiter capping starts to ``rpm`` within any rolling 60s window."""

    def __init__(self, rpm: int) -> None:
        """Create a limiter allowing at most ``rpm`` starts per rolling minute."""
        self._rpm = rpm
        self._window = 60.0
        self._starts: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until starting another request stays within the limit."""
        async with self._lock:
            loop = asyncio.get_running_loop()
            self._evict(loop.time())
            if len(self._starts) >= self._rpm:
                await asyncio.sleep(self._window - (loop.time() - self._starts[0]))
                self._evict(loop.time())
            self._starts.append(loop.time())

    def _evict(self, now: float) -> None:
        """Drop start timestamps older than the rolling window."""
        while self._starts and now - self._starts[0] >= self._window:
            self._starts.popleft()
