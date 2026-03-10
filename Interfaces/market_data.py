"""
Interfaces/market_data.py
=========================
Core market data types: Tick and Bar dataclasses.
These are the fundamental data structures used throughout the backtest system.

Design decisions:
- Tick: Represents a single trade/price update at tick level (used for realistic bar building)
- Bar: Represents OHLCV data aggregated over a time period (used by strategies)
- All timestamps are in nanoseconds (int) for precision and determinism
- Prices and volumes are floats for compatibility with numpy/pandas
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True, slots=True)
class Tick:
    """
    A single tick (trade) from historical data.
    
    Attributes:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        timestamp_ns: Unix timestamp in nanoseconds
        price: Trade price
        volume: Trade volume/quantity
        side: 'buy' or 'sell' if available, None otherwise
        trade_id: Optional unique trade identifier
    """
    symbol: str
    timestamp_ns: int
    price: float
    volume: float
    side: Optional[str] = None
    trade_id: Optional[int] = None
    
    @property
    def timestamp_ms(self) -> int:
        """Timestamp in milliseconds."""
        return self.timestamp_ns // 1_000_000
    
    @property
    def timestamp_s(self) -> float:
        """Timestamp in seconds (float)."""
        return self.timestamp_ns / 1_000_000_000


@dataclass(slots=True)
class Bar:
    """
    A completed OHLCV bar.
    
    Attributes:
        symbol: Trading pair symbol
        timeframe: Bar timeframe string (e.g., '1m', '5m', '1h')
        timestamp_ns: Bar open timestamp in nanoseconds
        open: Open price
        high: High price
        low: Low price
        close: Close price
        volume: Total volume in bar
        tick_count: Number of ticks that formed this bar (for tick/volume bars)
        dollar_volume: Total dollar volume (price * volume) if computed
        closed: Whether the bar is finalized/closed
    """
    symbol: str
    timeframe: str
    timestamp_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int = 0
    dollar_volume: float = 0.0
    closed: bool = True
    
    @property
    def timestamp_ms(self) -> int:
        """Bar open timestamp in milliseconds."""
        return self.timestamp_ns // 1_000_000
    
    @property
    def timestamp_s(self) -> float:
        """Bar open timestamp in seconds."""
        return self.timestamp_ns / 1_000_000_000
    
    @property
    def mid_price(self) -> float:
        """Mid price of high and low."""
        return (self.high + self.low) / 2.0
    
    @property
    def typical_price(self) -> float:
        """Typical price: (high + low + close) / 3."""
        return (self.high + self.low + self.close) / 3.0
    
    @property
    def vwap(self) -> float:
        """VWAP approximation using typical price (actual VWAP needs tick data)."""
        return self.typical_price
    
    def to_dict(self) -> dict:
        """Convert to dict format compatible with existing BarStore."""
        return {
            "o": self.open,
            "h": self.high,
            "l": self.low,
            "c": self.close,
            "v": self.volume,
            "t": self.timestamp_ms,
            "i": self.timeframe,
            "x": self.closed,
        }
