"""
Interfaces/cost_interface.py
============================
Cost model interfaces for realistic transaction cost simulation.

Design decisions:
- Separate interfaces for different cost components
- All costs are additive: total_cost = fee + slippage + spread_cost
- Latency is modeled but can be no-op in backtest
- All methods can use seeded RNG for deterministic randomness
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable
from random import Random

from Interfaces.orders import Order, OrderSide
from Interfaces.market_data import Bar


@runtime_checkable
class ICostModel(Protocol):
    """
    Combined cost model interface.
    Implementations should combine fee, slippage, and spread models.
    """
    
    def calculate_fee(
        self,
        notional: float,
        order_side: OrderSide,
        is_maker: bool = False
    ) -> float:
        """
        Calculate transaction fee.
        
        Args:
            notional: Trade notional value (price * quantity)
            order_side: BUY or SELL
            is_maker: Whether order was maker (limit) or taker (market)
            
        Returns:
            Fee amount in quote currency
        """
        ...
    
    def calculate_slippage(
        self,
        price: float,
        quantity: float,
        order_side: OrderSide,
        bar: Bar,
        rng: Random
    ) -> float:
        """
        Calculate slippage (price impact).
        
        Args:
            price: Intended execution price
            quantity: Order quantity
            order_side: BUY or SELL
            bar: Current bar for volatility estimation
            rng: Seeded RNG for deterministic randomness
            
        Returns:
            Slippage amount (positive = unfavorable)
        """
        ...
    
    def calculate_spread_cost(
        self,
        price: float,
        order_side: OrderSide
    ) -> float:
        """
        Calculate half-spread cost.
        
        Args:
            price: Mid price
            order_side: BUY or SELL
            
        Returns:
            Half-spread cost (always positive)
        """
        ...
    
    def get_latency_ns(self, rng: Random) -> int:
        """
        Get simulated latency in nanoseconds.
        
        Args:
            rng: Seeded RNG for deterministic randomness
            
        Returns:
            Latency in nanoseconds (can be 0 for instant execution)
        """
        ...
    
    def total_cost(
        self,
        order: Order,
        fill_price: float,
        bar: Bar,
        rng: Random,
        is_maker: bool = False
    ) -> tuple[float, float, float]:
        """
        Calculate total transaction costs.
        
        Args:
            order: The order being filled
            fill_price: Execution price
            bar: Current bar
            rng: Seeded RNG
            is_maker: Whether maker order
            
        Returns:
            Tuple of (fee, slippage, spread_cost)
        """
        ...


@runtime_checkable
class IFeeModel(Protocol):
    """Protocol for fee-only model."""
    
    def calculate(
        self,
        notional: float,
        order_side: OrderSide,
        is_maker: bool = False
    ) -> float:
        """Calculate fee for given notional value."""
        ...


@runtime_checkable  
class ISlippageModel(Protocol):
    """Protocol for slippage model."""
    
    def calculate(
        self,
        price: float,
        quantity: float,
        order_side: OrderSide,
        bar: Bar,
        rng: Random
    ) -> float:
        """Calculate slippage for given order."""
        ...


@runtime_checkable
class ISpreadModel(Protocol):
    """Protocol for bid-ask spread model."""
    
    def calculate(
        self,
        price: float,
        order_side: OrderSide
    ) -> float:
        """Calculate half-spread cost."""
        ...


@runtime_checkable
class ILatencyModel(Protocol):
    """Protocol for latency simulation."""
    
    def get_latency_ns(self, rng: Random) -> int:
        """Get latency in nanoseconds."""
        ...
