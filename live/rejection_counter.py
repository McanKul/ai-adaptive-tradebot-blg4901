"""
live/rejection_counter.py
=========================
Rolling rejection counter for the live broker.

When N rejections happen inside a moving ``window_seconds``, the
counter trips a callback (typically
``LiveGlobalRisk.trip_kill_switch``).  This stops the bot from
hammering the exchange with bad orders and protects the account from
silent error storms — e.g. an exchange flapping rate limits, a bug
that constantly fails MIN_NOTIONAL, or a wallet with too little
margin to back any new entry.

Defaults are deliberately conservative for canary trading: 5 errors
in 5 minutes is plenty of evidence something is wrong.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Callable, Deque, Optional


class RejectionCounter:
    """Sliding-window rejection counter with kill-switch callback.

    Args:
        max_count: Number of rejections to allow inside the window
            before tripping (default 5).
        window_seconds: Window length (default 300 = 5 min).
        on_trip: Callback invoked when the threshold is crossed.
            Receives the human-readable reason; idempotent — the
            counter does not call again until the window decays
            below the threshold.
    """

    def __init__(
        self,
        max_count: int = 5,
        window_seconds: float = 300.0,
        on_trip: Optional[Callable[[str], None]] = None,
    ):
        if max_count < 1:
            raise ValueError("max_count must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self.max_count = int(max_count)
        self.window_seconds = float(window_seconds)
        self.on_trip = on_trip
        self._events: Deque[tuple[float, str]] = deque()
        self._tripped: bool = False

    # ------------------------------------------------------------------

    def record(self, reason: str = "rejection") -> bool:
        """Add an event to the window.  Returns True iff this call tripped."""
        now = time.time()
        self._evict(now)
        self._events.append((now, reason))
        if not self._tripped and len(self._events) >= self.max_count:
            self._tripped = True
            if self.on_trip is not None:
                msg = (
                    f"rejection_storm: {len(self._events)} rejections in last "
                    f"{int(self.window_seconds)}s "
                    f"(latest: {reason})"
                )
                try:
                    self.on_trip(msg)
                except Exception:  # pragma: no cover — defensive
                    pass
            return True
        return False

    def reset(self) -> None:
        """Clear the window manually (used by tests)."""
        self._events.clear()
        self._tripped = False

    # ------------------------------------------------------------------

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
        # Once the window decays below the trip threshold the counter
        # rearms, so a transient spike can recover without restart.
        if self._tripped and len(self._events) < self.max_count:
            self._tripped = False

    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        self._evict(time.time())
        return len(self._events)

    @property
    def tripped(self) -> bool:
        self._evict(time.time())
        return self._tripped
