"""
live/rate_limiter.py
====================
Async rate limiter + exchange-info cache for Binance API calls.

Prevents exceeding Binance's 1200-weight/min limit by enforcing a
sliding-window throttle on all outbound API requests.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

from utils.logger import setup_logger

log = setup_logger("RateLimiter")


class AsyncRateLimiter:
    """Token-bucket style rate limiter (async-safe)."""

    def __init__(self, max_per_minute: int = 1000):
        self._max = max_per_minute
        self._tokens: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self, weight: int = 1):
        """Wait until *weight* request tokens are available."""
        async with self._lock:
            now = time.monotonic()
            # Purge tokens older than 60 s
            cutoff = now - 60.0
            self._tokens = [t for t in self._tokens if t > cutoff]

            if len(self._tokens) + weight > self._max:
                # Wait until oldest token expires
                wait = self._tokens[0] - cutoff
                if wait > 0:
                    log.debug("Rate limit: sleeping %.2fs", wait)
                    await asyncio.sleep(wait)
                self._tokens = [t for t in self._tokens if t > time.monotonic() - 60.0]

            for _ in range(weight):
                self._tokens.append(time.monotonic())


class ExchangeInfoCache:
    """Caches futures_exchange_info with a configurable TTL."""

    def __init__(self, ttl_sec: int = 300):
        self._ttl = ttl_sec
        self._data: Optional[Dict[str, Any]] = None
        self._ts: float = 0.0
        self._lock = asyncio.Lock()

    async def get(self, fetch_fn) -> Dict[str, Any]:
        """Return cached data or call *fetch_fn* if stale."""
        async with self._lock:
            now = time.monotonic()
            if self._data is not None and (now - self._ts) < self._ttl:
                return self._data
            self._data = await fetch_fn()
            self._ts = now
            return self._data

    def invalidate(self):
        self._data = None
