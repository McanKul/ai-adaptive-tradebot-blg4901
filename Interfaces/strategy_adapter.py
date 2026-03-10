"""
Interfaces/strategy_adapter.py
==============================
Strategy adapter for unifying different strategy interfaces.

This module provides the bridge between:
- Legacy generate_signal() strategies (returns "+1"/"-1"/None)
- Modern on_bar() strategies (returns List[Order] or StrategyDecision)
- Unified interface for both live and backtest

WHY THIS EXISTS:
================
1. generate_signal() = Simple, legacy interface for quick strategies
2. on_bar() = Advanced interface with full control
3. This adapter converts between them automatically

The adapter is used by both:
- BacktestEngine: calls adapt_strategy_output() after each bar
- LiveEngine: calls adapt_strategy_output() after each bar

SIZING SUPPORT:
===============
Strategies output "intent" (buy/sell). Actual position size is computed
at runtime using SizingConfig:

- FIXED_QTY: Use explicit quantity (e.g. 10000 DOGE)
- NOTIONAL_USD: qty = notional / current_price
- MARGIN_USD: qty = (margin * leverage) / current_price

This ensures realistic position sizing in backtests.

DOCUMENTATION FOR FUTURE:
=========================
Why does generate_signal exist?
- It's the legacy/easy path for simple binary strategies
- Returns just a signal, no order details
- The adapter converts it to proper orders

Why does on_bar exist?
- Full control over orders, sizing, TP/SL metadata
- Returns StrategyDecision with rich information
- Used for advanced strategies
"""
from __future__ import annotations
from typing import List, Optional, Any, Union, Protocol, runtime_checkable, Dict, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

from Interfaces.market_data import Bar
from Interfaces.orders import Order, OrderType, OrderSide
from Interfaces.IStrategy import IStrategy, StrategyDecision

if TYPE_CHECKING:
    from utils.bar_store import BarStore


# =============================================================================
# StrategyContext - Provides market state to strategies
# =============================================================================
@dataclass
class StrategyContext:
    """
    Context object passed to strategy.on_bar().
    
    Provides the strategy with all the market state it needs:
    - Current symbol and timeframe
    - Historical bar data via bar_store
    - Portfolio snapshot (positions, equity)
    - Current position in the symbol
    - Metadata for additional context
    
    Usage in strategy:
        def on_bar(self, bar: Bar, ctx: StrategyContext) -> StrategyDecision:
            # Get OHLCV history
            ohlcv = ctx.get_ohlcv()
            
            # Check current position
            if ctx.position > 0:
                # Already long
                ...
    """
    symbol: str
    timeframe: str
    bar_store: Optional["BarStore"] = None
    portfolio: Optional[Dict[str, Any]] = None
    position: float = 0.0
    equity: float = 0.0
    cash: float = 0.0
    timestamp_ns: int = 0
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_ohlcv(self, limit: int = 500) -> Optional[Dict[str, List[float]]]:
        """
        Get OHLCV data from bar store as a dict of arrays.
        
        Args:
            limit: Maximum number of bars to return
            
        Returns:
            Dict with keys ['open', 'high', 'low', 'close', 'volume'],
            each containing a list of values, or None if bar_store unavailable.
            
        Example:
            ohlcv = ctx.get_ohlcv()
            closes = ohlcv.get("close", [])  # [1.0, 1.1, 1.2, ...]
        """
        if self.bar_store is None:
            return None
        
        # Get list of bar dicts from bar_store
        bars = self.bar_store.get_recent(self.symbol, self.timeframe, limit=limit)
        
        if not bars:
            return {"open": [], "high": [], "low": [], "close": [], "volume": []}
        
        # Transform list of dicts to dict of lists
        return {
            "open": [b.get("open", 0.0) for b in bars],
            "high": [b.get("high", 0.0) for b in bars],
            "low": [b.get("low", 0.0) for b in bars],
            "close": [b.get("close", 0.0) for b in bars],
            "volume": [b.get("volume", 0.0) for b in bars],
        }


# =============================================================================
# IBacktestStrategy - Alias for backward compatibility
# =============================================================================
# For code that imports IBacktestStrategy from strategy_interface
IBacktestStrategy = IStrategy


class SizingMode(Enum):
    """Position sizing mode."""
    FIXED_QTY = "fixed_qty"      # Use explicit quantity
    NOTIONAL_USD = "notional_usd"  # qty = notional / price
    MARGIN_USD = "margin_usd"    # qty = (margin * leverage) / price


@dataclass
class SizingConfig:
    """
    Configuration for position sizing.
    
    Determines how order quantity is calculated at entry time.
    The engine uses this to convert strategy intent to actual orders.
    
    Attributes:
        mode: Sizing mode (FIXED_QTY, NOTIONAL_USD, MARGIN_USD)
        fixed_qty: Explicit quantity (for FIXED_QTY mode)
        notional_usd: Target USD notional (for NOTIONAL_USD mode)
        margin_usd: Margin in USD (for MARGIN_USD mode)
        leverage: Leverage multiplier (used with MARGIN_USD)
        leverage_mode: "spot" or "margin"
    """
    mode: SizingMode = SizingMode.MARGIN_USD
    fixed_qty: Optional[float] = None
    notional_usd: Optional[float] = None
    margin_usd: float = 100.0
    leverage: float = 1.0
    leverage_mode: str = "margin"
    
    def compute_qty(self, price: float) -> float:
        """
        Compute order quantity based on sizing mode and current price.
        
        Args:
            price: Current asset price (e.g. 0.08 for DOGE)
            
        Returns:
            Quantity in base asset units (e.g. 12500 DOGE)
        """
        if price <= 0:
            return 0.0
        
        if self.mode == SizingMode.FIXED_QTY:
            return self.fixed_qty or 1.0
        
        elif self.mode == SizingMode.NOTIONAL_USD:
            notional = self.notional_usd or 100.0
            return notional / price
        
        elif self.mode == SizingMode.MARGIN_USD:
            # For spot: no leverage, notional = margin
            # For margin: notional = margin * leverage
            if self.leverage_mode == "spot":
                notional = self.margin_usd
            else:
                notional = self.margin_usd * self.leverage
            return notional / price
        
        return 1.0  # Fallback
    
    def get_target_notional(self) -> float:
        """Get target notional USD for display purposes."""
        if self.mode == SizingMode.NOTIONAL_USD:
            return self.notional_usd or 100.0
        elif self.mode == SizingMode.MARGIN_USD:
            if self.leverage_mode == "spot":
                return self.margin_usd
            return self.margin_usd * self.leverage
        return 0.0  # Unknown for FIXED_QTY without price


@runtime_checkable
class HasOnBar(Protocol):
    """Protocol for strategies with on_bar method."""
    def on_bar(self, bar: Bar, ctx: Any) -> Union[List[Order], StrategyDecision]: ...


@runtime_checkable
class HasGenerateSignal(Protocol):
    """Protocol for strategies with generate_signal method."""
    def generate_signal(self, symbol: str) -> Optional[str]: ...


@dataclass
class AdaptedOutput:
    """
    Output from strategy adaptation.
    
    Contains:
    - orders: Orders to execute
    - decision: Full StrategyDecision if available
    - source: How the output was generated ('on_bar', 'generate_signal', 'none')
    """
    orders: List[Order]
    decision: Optional[StrategyDecision] = None
    source: str = "none"


def adapt_strategy_output(
    strategy: Any,
    bar: Bar,
    ctx: Any,
    position_size: float = 1.0,
    strategy_id: str = "",
) -> AdaptedOutput:
    """
    Adapt strategy output to a list of orders.
    
    This is the SINGLE adapter function used by both live and backtest.
    It handles all strategy interface variations:
    
    1. Strategy.on_bar(bar, ctx) -> StrategyDecision
    2. Strategy.on_bar(bar, ctx) -> List[Order]
    3. Strategy.generate_signal(symbol) -> "+1"/"-1"/None
    
    Args:
        strategy: Strategy instance (IStrategy or compatible)
        bar: Current completed bar
        ctx: Strategy context
        position_size: Default position size for signal-based orders
        strategy_id: ID to attach to generated orders
        
    Returns:
        AdaptedOutput with orders and metadata
    """
    orders: List[Order] = []
    decision: Optional[StrategyDecision] = None
    source = "none"
    
    # Try on_bar first (advanced interface)
    if hasattr(strategy, 'on_bar') and callable(getattr(strategy, 'on_bar')):
        try:
            result = strategy.on_bar(bar, ctx)
            
            # Handle different return types
            if isinstance(result, StrategyDecision):
                decision = result
                orders = result.orders.copy()
                source = "on_bar"
                
                # If decision has signal but no orders, convert signal to order
                if not orders and result.has_signal:
                    signal_orders = _signal_to_orders(
                        result.signal, 
                        bar, 
                        position_size, 
                        strategy_id,
                        result.metadata
                    )
                    orders = signal_orders
                    
            elif isinstance(result, list):
                # Direct order list return
                orders = result
                decision = StrategyDecision(orders=orders)
                source = "on_bar"
                
            elif result is None:
                # No action
                decision = StrategyDecision.no_action()
                source = "on_bar"
                
        except NotImplementedError:
            # on_bar not implemented, fall through to generate_signal
            pass
        except TypeError as e:
            # Wrong signature, try simpler call
            if "argument" in str(e).lower():
                pass  # Fall through
            else:
                raise
    
    # Fall back to generate_signal (legacy interface)
    if source == "none" and hasattr(strategy, 'generate_signal') and callable(getattr(strategy, 'generate_signal')):
        signal = strategy.generate_signal(bar.symbol)
        if signal is not None:
            orders = _signal_to_orders(signal, bar, position_size, strategy_id)
            decision = StrategyDecision.from_signal(signal)
            decision.orders = orders
            source = "generate_signal"
        else:
            decision = StrategyDecision.no_action()
            source = "generate_signal"
    
    return AdaptedOutput(orders=orders, decision=decision, source=source)


def _signal_to_orders(
    signal: str,
    bar: Bar,
    position_size: float,
    strategy_id: str = "",
    metadata: Optional[dict] = None,
) -> List[Order]:
    """
    Convert a signal string to an order list.
    
    Args:
        signal: "+1" for buy, "-1" for sell
        bar: Current bar for symbol/timestamp
        position_size: Order quantity
        strategy_id: Strategy identifier
        metadata: Optional order metadata (TP/SL params, etc.)
        
    Returns:
        List with one order, or empty list
    """
    if signal not in ("+1", "-1", "1", "-1"):
        return []
    
    side = OrderSide.BUY if signal in ("+1", "1") else OrderSide.SELL
    
    order = Order(
        symbol=bar.symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=position_size,
        timestamp_ns=bar.timestamp_ns,
        strategy_id=strategy_id,
        metadata=metadata or {},
    )
    
    return [order]


def is_advanced_strategy(strategy: Any) -> bool:
    """
    Check if strategy implements the advanced on_bar interface.
    
    This helps determine if the strategy needs the full context
    or just symbol information.
    """
    if not hasattr(strategy, 'on_bar'):
        return False
    
    # Check if on_bar is the default IStrategy implementation
    # by comparing to the base class
    if isinstance(strategy, IStrategy):
        # If it's the base implementation, it just calls generate_signal
        # so we treat it as non-advanced
        method = getattr(strategy.__class__, 'on_bar', None)
        base_method = getattr(IStrategy, 'on_bar', None)
        if method is base_method:
            return False
    
    return True


def get_strategy_exit_params(strategy: Any, order: Order) -> dict:
    """
    Extract exit parameters from strategy and order.
    
    Looks for TP/SL parameters in:
    1. Order metadata (highest priority)
    2. Strategy.get_exit_params()
    3. Strategy.exit_manager.config
    
    Args:
        strategy: Strategy instance
        order: The order being placed
        
    Returns:
        Dict with exit parameters (tp_pct, sl_pct, tp_price, sl_price, etc.)
    """
    params = {}
    
    # From strategy
    if hasattr(strategy, 'get_exit_params') and callable(strategy.get_exit_params):
        params.update(strategy.get_exit_params())
    
    # From exit manager
    exit_mgr = getattr(strategy, 'exit_manager', None)
    if exit_mgr and hasattr(exit_mgr, 'config'):
        cfg = exit_mgr.config
        if hasattr(cfg, 'take_profit_pct') and cfg.take_profit_pct:
            params['tp_pct'] = cfg.take_profit_pct
        if hasattr(cfg, 'stop_loss_pct') and cfg.stop_loss_pct:
            params['sl_pct'] = cfg.stop_loss_pct
        if hasattr(cfg, 'take_profit_usd') and cfg.take_profit_usd:
            params['tp_usd'] = cfg.take_profit_usd
        if hasattr(cfg, 'stop_loss_usd') and cfg.stop_loss_usd:
            params['sl_usd'] = cfg.stop_loss_usd
    
    # From order metadata (highest priority)
    if order.metadata:
        for key in ('tp_pct', 'sl_pct', 'tp_price', 'sl_price', 'tp_usd', 'sl_usd'):
            if key in order.metadata:
                params[key] = order.metadata[key]
    
    return params


def apply_sizing_to_orders(
    orders: List[Order],
    sizing_config: SizingConfig,
    current_price: float,
    log_first_n: int = 3,
) -> List[Order]:
    """
    Apply sizing configuration to orders, computing qty at current price.
    
    This function is called by the engine AFTER strategy produces orders
    but BEFORE risk checks and execution. It replaces placeholder quantities
    with properly computed quantities based on sizing mode.
    
    Args:
        orders: Orders from strategy (may have placeholder qty)
        sizing_config: Sizing configuration
        current_price: Current market price for qty calculation
        log_first_n: Log details for first N orders (for debugging)
        
    Returns:
        Orders with updated quantities
    
    Example:
        # Strategy outputs intent with qty=1.0 placeholder
        # Engine calls apply_sizing_to_orders with price=0.08
        # SizingConfig(mode=MARGIN_USD, margin=100, leverage=10)
        # Result: qty = (100 * 10) / 0.08 = 12500 DOGE
    """
    import logging
    log = logging.getLogger(__name__)
    
    updated_orders = []
    _sizing_log_count = getattr(apply_sizing_to_orders, '_log_count', 0)
    
    for order in orders:
        # Skip reduce-only orders - they should use actual position size
        if order.reduce_only:
            updated_orders.append(order)
            continue
        
        # Compute quantity based on sizing config
        computed_qty = sizing_config.compute_qty(current_price)
        
        # Log first few for visibility
        if _sizing_log_count < log_first_n:
            target_notional = sizing_config.get_target_notional()
            log.info(
                f"SIZING: {order.symbol} {order.side.value} | "
                f"mode={sizing_config.mode.value} | "
                f"price={current_price:.6f} | "
                f"qty={computed_qty:.6f} | "
                f"notional=${computed_qty * current_price:,.2f}"
            )
            _sizing_log_count += 1
            apply_sizing_to_orders._log_count = _sizing_log_count
        
        # Create new order with computed qty
        updated_order = Order(
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=computed_qty,
            price=order.price,
            timestamp_ns=order.timestamp_ns,
            strategy_id=order.strategy_id,
            reduce_only=order.reduce_only,
            metadata=order.metadata.copy() if order.metadata else {},
        )
        
        # Store sizing info in metadata for debugging
        updated_order.metadata['_sizing_mode'] = sizing_config.mode.value
        updated_order.metadata['_entry_price'] = current_price
        updated_order.metadata['_computed_qty'] = computed_qty
        updated_order.metadata['_target_notional'] = computed_qty * current_price
        
        updated_orders.append(updated_order)
    
    return updated_orders


def reset_sizing_log_counter():
    """Reset the sizing log counter for a new backtest run."""
    apply_sizing_to_orders._log_count = 0
