"""
Backtest/multi_symbol_sync.py
=============================
Synchronize bars from multiple symbols into a single ``MultiBar`` event.

Used by multi-symbol strategies (arbitrage, pairs, etc.) that need the
latest closed bar from every leg before they decide.

Modes:
* ``"strict"``       — emit only when every symbol has a bar at the
                        same timestamp_ns.  Misaligned bars are dropped
                        from the older symbols and a metric is emitted.
* ``"forward_fill"`` — emit on every update; missing legs reuse their
                        last seen bar (caller decides if the staleness
                        is acceptable via ``MultiBar.staleness_ns``).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from Interfaces.market_data import Bar


@dataclass
class MultiBar:
    """A snapshot of the latest closed bar for every subscribed symbol."""
    timestamp_ns: int
    bars: Dict[str, "Bar"]
    staleness_ns: Dict[str, int] = field(default_factory=dict)

    def __getitem__(self, symbol: str) -> "Bar":
        return self.bars[symbol]


class BarSyncBuffer:
    """Buffer last-seen bar per symbol and emit synchronized MultiBars.

    Args:
        symbols: List of symbols to track.
        mode: ``"strict"`` or ``"forward_fill"``.
        tolerance_ns: Strict-mode tolerance for "same timestamp" — bars
            within this window are considered aligned.  Default 0.
    """

    def __init__(
        self,
        symbols: List[str],
        mode: str = "strict",
        tolerance_ns: int = 0,
    ):
        if not symbols:
            raise ValueError("BarSyncBuffer requires at least one symbol")
        if mode not in ("strict", "forward_fill"):
            raise ValueError(f"unknown mode '{mode}'")
        if tolerance_ns < 0:
            raise ValueError("tolerance_ns must be >= 0")

        self.symbols = list(symbols)
        self.mode = mode
        self.tolerance_ns = tolerance_ns
        self._latest: Dict[str, "Bar"] = {}
        self._last_emit_ts: int = -1
        self.dropped: int = 0
        self.emitted: int = 0

    def update(self, bar: "Bar") -> None:
        """Record ``bar`` against its symbol.  No-op for unknown symbols."""
        if bar.symbol not in self.symbols:
            return
        self._latest[bar.symbol] = bar

    def try_emit(self) -> Optional[MultiBar]:
        """Return a MultiBar if one is ready, else None."""
        if len(self._latest) < len(self.symbols):
            return None  # still warming up

        if self.mode == "strict":
            timestamps = [b.timestamp_ns for b in self._latest.values()]
            tmin, tmax = min(timestamps), max(timestamps)
            if (tmax - tmin) > self.tolerance_ns:
                # Bars not aligned.  Don't emit yet; the lagging symbol
                # will catch up on a subsequent update.
                return None
            ts = tmax
        else:  # forward_fill
            ts = max(b.timestamp_ns for b in self._latest.values())

        # Avoid double-emitting for the same alignment
        if ts <= self._last_emit_ts:
            self.dropped += 1
            return None

        bars = {s: self._latest[s] for s in self.symbols if s in self._latest}
        if len(bars) != len(self.symbols):
            return None  # forward_fill but a symbol never seen yet

        staleness = {s: ts - bars[s].timestamp_ns for s in bars}
        self._last_emit_ts = ts
        self.emitted += 1
        return MultiBar(timestamp_ns=ts, bars=bars, staleness_ns=staleness)

    def reset(self) -> None:
        self._latest.clear()
        self._last_emit_ts = -1
        self.dropped = 0
        self.emitted = 0
