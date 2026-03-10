"""
Backtest/disk_streamer.py
=========================
Disk-based tick streamer for deterministic backtest replay.

Design decisions:
- Implements ITickIterator protocol (NOT live IStreamer)
- NO network usage - purely disk-based
- Yields ticks in timestamp order deterministically
- Supports multiple symbols merged by timestamp
- Optional replay speed control (for debugging/visualization)
"""
from __future__ import annotations
import heapq
from dataclasses import dataclass
from typing import Iterator, List, Optional, Dict, Any
import logging

from Interfaces.market_data import Tick
from Backtest.tick_store import TickStore, TickStoreConfig

log = logging.getLogger(__name__)


@dataclass
class DiskStreamerConfig:
    """Configuration for DiskTickStreamer."""
    symbols: List[str]  # List of symbols to stream
    start_ts_ns: Optional[int] = None  # Start timestamp (inclusive)
    end_ts_ns: Optional[int] = None  # End timestamp (inclusive)
    replay_speed: float = 0.0  # 0 = instant, >0 = realtime multiplier (for debug)


class DiskTickStreamer:
    """
    Disk-based tick streamer that replays historical ticks deterministically.
    
    Implements ITickIterator protocol for backtest engine consumption.
    Merges multiple symbols' ticks in timestamp order using a heap.
    
    Usage:
        store = TickStore(store_config)
        streamer = DiskTickStreamer(store, stream_config)
        
        for tick in streamer.iter_ticks():
            bar_builder.on_tick(tick)
    """
    
    def __init__(self, tick_store: TickStore, config: DiskStreamerConfig):
        """
        Initialize DiskTickStreamer.
        
        Args:
            tick_store: TickStore for loading tick data
            config: DiskStreamerConfig with symbols and time range
        """
        self.tick_store = tick_store
        self.config = config
        self._iterators: Dict[str, Iterator[Tick]] = {}
        self._heap: List[tuple[int, int, str, Tick]] = []  # (ts, counter, symbol, tick)
        self._counter = 0  # For stable heap ordering
        self._initialized = False
        self._total_ticks = 0
    
    def _initialize(self) -> None:
        """Initialize iterators and heap for all symbols."""
        self._heap = []
        self._counter = 0
        self._iterators = {}
        
        for symbol in self.config.symbols:
            if not self.tick_store.file_exists(symbol):
                log.warning(f"No tick data found for symbol: {symbol}")
                continue
            
            it = self.tick_store.iter_ticks(
                symbol,
                self.config.start_ts_ns,
                self.config.end_ts_ns
            )
            self._iterators[symbol] = it
            
            # Prime the heap with first tick from each symbol
            self._advance_iterator(symbol)
        
        self._initialized = True
        log.info(
            f"DiskTickStreamer initialized: {len(self._iterators)} symbols, "
            f"range [{self.config.start_ts_ns}, {self.config.end_ts_ns}]"
        )
    
    def _advance_iterator(self, symbol: str) -> bool:
        """
        Get next tick from symbol's iterator and add to heap.
        
        Returns:
            True if a tick was added, False if iterator exhausted
        """
        it = self._iterators.get(symbol)
        if it is None:
            return False
        
        try:
            tick = next(it)
            # Heap entry: (timestamp, counter, symbol, tick)
            # Counter ensures stable ordering for same-timestamp ticks
            heapq.heappush(self._heap, (tick.timestamp_ns, self._counter, symbol, tick))
            self._counter += 1
            return True
        except StopIteration:
            del self._iterators[symbol]
            return False
    
    def iter_ticks(self) -> Iterator[Tick]:
        """
        Iterate over all ticks in timestamp order.
        
        Yields:
            Tick objects merged from all symbols in ascending timestamp order
        """
        if not self._initialized:
            self._initialize()
        
        while self._heap:
            ts, _, symbol, tick = heapq.heappop(self._heap)
            self._total_ticks += 1
            
            # Advance the iterator for this symbol
            self._advance_iterator(symbol)
            
            yield tick
    
    def reset(self) -> None:
        """Reset streamer for a new iteration."""
        self._initialized = False
        self._heap = []
        self._iterators = {}
        self._counter = 0
        self._total_ticks = 0
        log.debug("DiskTickStreamer reset")
    
    @property
    def total_ticks_yielded(self) -> int:
        """Number of ticks yielded so far."""
        return self._total_ticks
    
    def peek_next_timestamp(self) -> Optional[int]:
        """
        Peek at the next tick's timestamp without consuming it.
        
        Returns:
            Next timestamp in nanoseconds, or None if exhausted
        """
        if not self._initialized:
            self._initialize()
        
        if self._heap:
            return self._heap[0][0]
        return None


class SingleSymbolDiskStreamer:
    """
    Simplified single-symbol streamer (no heap merging needed).
    
    More efficient when backtesting a single symbol.
    """
    
    def __init__(
        self,
        tick_store: TickStore,
        symbol: str,
        start_ts_ns: Optional[int] = None,
        end_ts_ns: Optional[int] = None
    ):
        self.tick_store = tick_store
        self.symbol = symbol
        self.start_ts_ns = start_ts_ns
        self.end_ts_ns = end_ts_ns
        self._total_ticks = 0
    
    def iter_ticks(self) -> Iterator[Tick]:
        """Iterate over ticks for the single symbol."""
        self._total_ticks = 0
        for tick in self.tick_store.iter_ticks(
            self.symbol, self.start_ts_ns, self.end_ts_ns
        ):
            self._total_ticks += 1
            yield tick
    
    def reset(self) -> None:
        """Reset for new iteration."""
        self._total_ticks = 0
    
    @property
    def total_ticks_yielded(self) -> int:
        return self._total_ticks
