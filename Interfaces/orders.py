"""
Interfaces/orders.py
====================
Order and Fill dataclasses for the trading system.

Design decisions:
- Order: Represents intent to trade (submitted by strategy)
- Fill: Represents executed trade (returned by execution model)
- OrderType: MARKET orders fill at bar close price (default), LIMIT orders may or may not fill
- All IDs are strings for flexibility
- Timestamps in nanoseconds for precision
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import uuid


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = auto()
    LIMIT = auto()
    STOP_MARKET = auto()
    TAKE_PROFIT = auto()


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = auto()
    SELL = auto()
    
    @classmethod
    def from_string(cls, s: str) -> "OrderSide":
        """Parse from string ('buy', 'sell', '+1', '-1')."""
        s = str(s).lower().strip()
        if s in ("buy", "+1", "1", "long"):
            return cls.BUY
        elif s in ("sell", "-1", "short"):
            return cls.SELL
        raise ValueError(f"Unknown order side: {s}")


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = auto()
    FILLED = auto()
    PARTIALLY_FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()


@dataclass(slots=True)
class Order:
    """
    An order submitted by a strategy.
    
    Attributes:
        symbol: Trading pair symbol
        side: BUY or SELL
        order_type: MARKET, LIMIT, etc.
        quantity: Order quantity (positive)
        price: Limit price (required for LIMIT orders, ignored for MARKET)
        stop_price: Stop price for STOP_MARKET orders
        order_id: Unique order identifier
        timestamp_ns: Order creation timestamp in nanoseconds
        strategy_id: Identifier of the strategy that created this order
        reduce_only: If True, order can only reduce position (for exits/liquidations)
        metadata: Optional metadata dict for custom fields
    """
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_ns: int = 0
    strategy_id: str = ""
    reduce_only: bool = False
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {self.quantity}")
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("LIMIT orders require a price")
    
    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        return self.side == OrderSide.SELL
    
    @property
    def notional(self) -> Optional[float]:
        """Notional value if price is set."""
        if self.price is not None:
            return self.quantity * self.price
        return None


@dataclass(slots=True)
class Fill:
    """
    A filled order (or partial fill).
    
    Attributes:
        order_id: Reference to the original order
        symbol: Trading pair symbol
        side: BUY or SELL
        fill_price: Actual execution price (after slippage/spread)
        fill_quantity: Quantity filled
        fee: Transaction fee paid
        slippage: Slippage cost (difference from intended price)
        timestamp_ns: Fill timestamp in nanoseconds
        fill_id: Unique fill identifier
        metadata: Optional metadata dict
    """
    order_id: str
    symbol: str
    side: OrderSide
    fill_price: float
    fill_quantity: float
    fee: float = 0.0
    slippage: float = 0.0
    timestamp_ns: int = 0
    fill_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict = field(default_factory=dict)
    
    @property
    def notional(self) -> float:
        """Notional value of the fill."""
        return self.fill_quantity * self.fill_price
    
    @property
    def total_cost(self) -> float:
        """Total cost including fees and slippage."""
        return self.fee + abs(self.slippage)
    
    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        return self.side == OrderSide.SELL
