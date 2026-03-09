"""
Backtest/bar_builder.py
=======================
Bar builders that aggregate ticks into OHLCV bars.

Design decisions:
- Base BarBuilder class with common logic
- TimeBarBuilder: Standard time-based bars (1m, 5m, etc.) - MANDATORY
- TickBarBuilder: Bars based on tick count (AFML-inspired)
- VolumeBarBuilder: Bars based on volume threshold (AFML-inspired)
- DollarBarBuilder: Bars based on dollar volume threshold (AFML-inspired)

All builders:
- Receive ticks via on_tick()
- Return completed Bar when a bar closes
- Maintain internal state for partial bars
- Are deterministic (no random behavior)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List
import logging

from Interfaces.market_data import Tick, Bar

log = logging.getLogger(__name__)


# Standard timeframe mappings
TIMEFRAME_SECONDS: Dict[str, int] = {
    "1s": 1,
    "5s": 5,
    "10s": 10,
    "30s": 30,
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}

TIMEFRAME_NS: Dict[str, int] = {k: v * 1_000_000_000 for k, v in TIMEFRAME_SECONDS.items()}


@dataclass
class PartialBar:
    """Internal state for a bar being built."""
    symbol: str
    timeframe: str
    open_ts_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int
    dollar_volume: float


class BarBuilder(ABC):
    """
    Abstract base class for bar builders.
    
    Subclasses implement _should_close_bar() to determine when to emit a bar.
    """
    
    def __init__(self, symbol: str, timeframe: str):
        """
        Initialize bar builder.
        
        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe identifier
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self._partial: Optional[PartialBar] = None
        self._bar_count = 0
    
    @abstractmethod
    def _should_close_bar(self, tick: Tick) -> bool:
        """
        Determine if current tick should close the partial bar.
        
        Args:
            tick: The incoming tick
            
        Returns:
            True if bar should be closed before processing this tick
        """
        ...
    
    @abstractmethod
    def _get_bar_open_ts(self, tick: Tick) -> int:
        """
        Get the opening timestamp for a new bar.
        
        Args:
            tick: The first tick of the new bar
            
        Returns:
            Bar open timestamp in nanoseconds
        """
        ...
    
    def on_tick(self, tick: Tick) -> Optional[Bar]:
        """
        Process a tick, potentially completing a bar.
        
        Args:
            tick: The incoming tick
            
        Returns:
            Completed Bar if one was finalized, None otherwise
        """
        if tick.symbol != self.symbol:
            return None
        
        completed_bar: Optional[Bar] = None
        
        # Check if we should close the current bar
        if self._partial is not None and self._should_close_bar(tick):
            completed_bar = self._finalize_bar()
        
        # Update or create partial bar
        if self._partial is None:
            self._start_new_bar(tick)
        else:
            self._update_bar(tick)
        
        return completed_bar
    
    def _start_new_bar(self, tick: Tick) -> None:
        """Start a new partial bar with the given tick."""
        self._partial = PartialBar(
            symbol=self.symbol,
            timeframe=self.timeframe,
            open_ts_ns=self._get_bar_open_ts(tick),
            open=tick.price,
            high=tick.price,
            low=tick.price,
            close=tick.price,
            volume=tick.volume,
            tick_count=1,
            dollar_volume=tick.price * tick.volume
        )
    
    def _update_bar(self, tick: Tick) -> None:
        """Update the partial bar with a new tick."""
        if self._partial is None:
            return
        
        self._partial.high = max(self._partial.high, tick.price)
        self._partial.low = min(self._partial.low, tick.price)
        self._partial.close = tick.price
        self._partial.volume += tick.volume
        self._partial.tick_count += 1
        self._partial.dollar_volume += tick.price * tick.volume
    
    def _finalize_bar(self) -> Bar:
        """Finalize and return the current partial bar."""
        if self._partial is None:
            raise RuntimeError("Cannot finalize bar: no partial bar exists")
        
        bar = Bar(
            symbol=self._partial.symbol,
            timeframe=self._partial.timeframe,
            timestamp_ns=self._partial.open_ts_ns,
            open=self._partial.open,
            high=self._partial.high,
            low=self._partial.low,
            close=self._partial.close,
            volume=self._partial.volume,
            tick_count=self._partial.tick_count,
            dollar_volume=self._partial.dollar_volume,
            closed=True
        )
        
        self._partial = None
        self._bar_count += 1
        return bar
    
    def flush(self) -> Optional[Bar]:
        """
        Force-close the current partial bar (e.g., at end of data).
        
        Returns:
            The finalized bar, or None if no partial bar exists
        """
        if self._partial is not None:
            return self._finalize_bar()
        return None
    
    def reset(self) -> None:
        """Reset builder state for a new backtest run."""
        self._partial = None
        self._bar_count = 0
    
    @property
    def bar_count(self) -> int:
        """Number of bars completed."""
        return self._bar_count
    
    @property
    def has_partial(self) -> bool:
        """Whether there's a partial bar in progress."""
        return self._partial is not None


class TimeBarBuilder(BarBuilder):
    """
    Time-based bar builder (standard OHLCV bars).
    
    Closes bars at fixed time intervals (1m, 5m, 1h, etc.).
    This is the MANDATORY implementation.
    """
    
    def __init__(self, symbol: str, timeframe: str = "1m"):
        """
        Initialize TimeBarBuilder.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string (e.g., '1m', '5m', '1h')
        """
        super().__init__(symbol, timeframe)
        
        if timeframe not in TIMEFRAME_NS:
            raise ValueError(
                f"Unknown timeframe '{timeframe}'. "
                f"Valid options: {list(TIMEFRAME_NS.keys())}"
            )
        
        self._period_ns = TIMEFRAME_NS[timeframe]
    
    def _should_close_bar(self, tick: Tick) -> bool:
        """Close bar when tick crosses into a new time bucket."""
        if self._partial is None:
            return False
        
        current_bucket = self._partial.open_ts_ns
        tick_bucket = (tick.timestamp_ns // self._period_ns) * self._period_ns
        
        return tick_bucket > current_bucket
    
    def _get_bar_open_ts(self, tick: Tick) -> int:
        """Get the aligned bar open timestamp."""
        return (tick.timestamp_ns // self._period_ns) * self._period_ns


class TickBarBuilder(BarBuilder):
    """
    Tick count-based bar builder (AFML-inspired information-driven bars).
    
    Closes bars after a fixed number of ticks.
    Idea: Each bar contains approximately equal information (number of transactions).
    """
    
    def __init__(self, symbol: str, tick_threshold: int = 100):
        """
        Initialize TickBarBuilder.
        
        Args:
            symbol: Trading symbol
            tick_threshold: Number of ticks per bar
        """
        super().__init__(symbol, f"tick_{tick_threshold}")
        self._tick_threshold = tick_threshold
    
    def _should_close_bar(self, tick: Tick) -> bool:
        """Close bar when tick count reaches threshold."""
        if self._partial is None:
            return False
        return self._partial.tick_count >= self._tick_threshold
    
    def _get_bar_open_ts(self, tick: Tick) -> int:
        """Use tick timestamp as bar open."""
        return tick.timestamp_ns


class VolumeBarBuilder(BarBuilder):
    """
    Volume-based bar builder (AFML-inspired information-driven bars).
    
    Closes bars when cumulative volume exceeds threshold.
    Idea: Each bar contains approximately equal volume traded.
    """
    
    def __init__(self, symbol: str, volume_threshold: float = 1000.0):
        """
        Initialize VolumeBarBuilder.
        
        Args:
            symbol: Trading symbol
            volume_threshold: Volume per bar
        """
        super().__init__(symbol, f"vol_{volume_threshold:.0f}")
        self._volume_threshold = volume_threshold
    
    def _should_close_bar(self, tick: Tick) -> bool:
        """Close bar when volume exceeds threshold."""
        if self._partial is None:
            return False
        return self._partial.volume >= self._volume_threshold
    
    def _get_bar_open_ts(self, tick: Tick) -> int:
        """Use tick timestamp as bar open."""
        return tick.timestamp_ns


class DollarBarBuilder(BarBuilder):
    """
    Dollar volume-based bar builder (AFML-inspired information-driven bars).
    
    Closes bars when cumulative dollar volume (price * volume) exceeds threshold.
    Idea: Each bar contains approximately equal value traded.
    """
    
    def __init__(self, symbol: str, dollar_threshold: float = 100000.0):
        """
        Initialize DollarBarBuilder.
        
        Args:
            symbol: Trading symbol
            dollar_threshold: Dollar volume per bar
        """
        super().__init__(symbol, f"dollar_{dollar_threshold:.0f}")
        self._dollar_threshold = dollar_threshold
    
    def _should_close_bar(self, tick: Tick) -> bool:
        """Close bar when dollar volume exceeds threshold."""
        if self._partial is None:
            return False
        return self._partial.dollar_volume >= self._dollar_threshold
    
    def _get_bar_open_ts(self, tick: Tick) -> int:
        """Use tick timestamp as bar open."""
        return tick.timestamp_ns


def create_bar_builder(
    symbol: str,
    bar_type: str = "time",
    timeframe: str = "1m",
    tick_threshold: int = 100,
    volume_threshold: float = 1000.0,
    dollar_threshold: float = 100000.0
) -> BarBuilder:
    """
    Factory function to create bar builders.
    
    Args:
        symbol: Trading symbol
        bar_type: Type of bar ('time', 'tick', 'volume', 'dollar')
        timeframe: Timeframe for time bars
        tick_threshold: Threshold for tick bars
        volume_threshold: Threshold for volume bars
        dollar_threshold: Threshold for dollar bars
        
    Returns:
        Appropriate BarBuilder instance
    """
    bar_type = bar_type.lower()
    
    if bar_type == "time":
        return TimeBarBuilder(symbol, timeframe)
    elif bar_type == "tick":
        return TickBarBuilder(symbol, tick_threshold)
    elif bar_type == "volume":
        return VolumeBarBuilder(symbol, volume_threshold)
    elif bar_type == "dollar":
        return DollarBarBuilder(symbol, dollar_threshold)
    else:
        raise ValueError(
            f"Unknown bar type '{bar_type}'. "
            f"Valid options: time, tick, volume, dollar"
        )
