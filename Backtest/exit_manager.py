"""
Backtest/exit_manager.py
========================
Reusable exit management for strategies.

Provides standardized TP/SL logic that strategies can opt into.
The ExitManager checks current position against exit rules and
generates reduce-only orders when conditions are met.

Exit types:
- Take Profit (absolute USD or percentage)
- Stop Loss (absolute USD or percentage)
- Trailing Stop (percentage from peak)
- Max Holding Bars (time-based exit)

Usage:
    exit_mgr = ExitManager(
        take_profit_usd=2.0,
        stop_loss_usd=1.5,
        trailing_stop_pct=0.02
    )
    
    # In strategy.on_bar():
    exit_order = exit_mgr.check_exit(bar, ctx, avg_entry_price, entry_bar)
    if exit_order:
        return [exit_order]
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import logging

from Interfaces.market_data import Bar
from Interfaces.orders import Order, OrderSide, OrderType

log = logging.getLogger(__name__)


class ExitReason(Enum):
    """Reason for exit signal."""
    TAKE_PROFIT_USD = "TP_USD"
    TAKE_PROFIT_PCT = "TP_PCT"
    STOP_LOSS_USD = "SL_USD"
    STOP_LOSS_PCT = "SL_PCT"
    TRAILING_STOP = "TRAILING"
    MAX_HOLDING_BARS = "MAX_BARS"
    LIQUIDATION = "LIQUIDATION"


@dataclass
class ExitConfig:
    """Configuration for exit rules."""
    # Take profit
    take_profit_usd: Optional[float] = None  # Absolute USD profit target
    take_profit_pct: Optional[float] = None  # Percentage from entry (0.02 = 2%)
    
    # Stop loss
    stop_loss_usd: Optional[float] = None  # Absolute USD loss limit
    stop_loss_pct: Optional[float] = None  # Percentage from entry (0.02 = 2%)
    
    # Trailing stop
    trailing_stop_pct: Optional[float] = None  # Trail by this % from peak profit
    
    # Time-based
    max_holding_bars: Optional[int] = None  # Exit after N bars
    
    # Behavior
    reduce_only: bool = True  # Only generate reduce-only orders
    use_market_orders: bool = True  # Use market orders for exits
    
    # NEW: Intra-bar exit checks (feature-flagged, default OFF for backward compat)
    # When enabled, TP/SL checks use bar high/low instead of just close
    # This reduces optimistic bias but may change signal timing
    use_intrabar_checks: bool = False
    
    # NEW: Leverage-aware P&L calculation (feature-flagged, default OFF)
    # When enabled, P&L is adjusted based on leverage factor
    # This affects TP/SL threshold calculations
    leverage: float = 1.0  # 1.0 = spot (no leverage adjustment)
    
    def has_any_rule(self) -> bool:
        """Check if any exit rule is configured."""
        return any([
            self.take_profit_usd is not None,
            self.take_profit_pct is not None,
            self.stop_loss_usd is not None,
            self.stop_loss_pct is not None,
            self.trailing_stop_pct is not None,
            self.max_holding_bars is not None,
        ])


@dataclass
class PositionState:
    """Tracks state needed for exit calculations."""
    symbol: str
    quantity: float  # Signed: positive=long, negative=short
    avg_entry_price: float
    entry_bar_index: int = 0
    entry_timestamp_ns: int = 0
    
    # Trailing stop tracking
    peak_pnl: float = 0.0
    peak_price: float = 0.0  # For longs: highest price seen; for shorts: lowest
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < 1e-10
    
    @property
    def notional(self) -> float:
        """Position notional at entry."""
        return abs(self.quantity) * self.avg_entry_price
    
    def unrealized_pnl(self, current_price: float, leverage: float = 1.0) -> float:
        """Calculate current unrealized P&L.
        
        Args:
            current_price: Current market price
            leverage: Leverage multiplier (1.0 = spot, >1.0 = leveraged)
                      In leverage mode, P&L is amplified relative to margin used
        
        Returns:
            Unrealized P&L in USD
        """
        if self.is_flat:
            return 0.0
        # Base P&L calculation (price change * quantity)
        base_pnl = (current_price - self.avg_entry_price) * self.quantity
        # NOTE: Leverage does NOT multiply P&L - it affects margin requirement
        # The P&L is the same, but relative to margin used, the % return is higher
        return base_pnl
    
    def unrealized_pnl_pct(self, current_price: float, leverage: float = 1.0) -> float:
        """Calculate unrealized P&L as percentage of entry notional.
        
        Args:
            current_price: Current market price
            leverage: Leverage multiplier - when >1.0, the percentage return
                      is calculated relative to margin used, not full notional
        
        Returns:
            P&L as fraction (0.02 = 2%)
        """
        if self.is_flat or self.notional == 0:
            return 0.0
        # In leveraged mode, calculate return relative to margin (notional / leverage)
        effective_capital = self.notional / leverage if leverage > 0 else self.notional
        return self.unrealized_pnl(current_price) / effective_capital
    
    def update_peak(self, current_price: float) -> None:
        """Update peak price/pnl for trailing stop."""
        pnl = self.unrealized_pnl(current_price)
        
        if self.is_long:
            if current_price > self.peak_price:
                self.peak_price = current_price
                self.peak_pnl = pnl
        elif self.is_short:
            if self.peak_price == 0 or current_price < self.peak_price:
                self.peak_price = current_price
                self.peak_pnl = pnl


class ExitManager:
    """
    Manages exit rules for a position.
    
    Call check_exit() on each bar to see if an exit should be triggered.
    The ExitManager maintains state for trailing stops across bars.
    """
    
    def __init__(self, config: Optional[ExitConfig] = None, **kwargs):
        """
        Initialize ExitManager.
        
        Args:
            config: ExitConfig object, or pass individual params as kwargs
        """
        if config is not None:
            self.config = config
        else:
            self.config = ExitConfig(**kwargs)
        
        # Per-symbol position state for trailing stops
        self._states: Dict[str, PositionState] = {}
        
        # Statistics
        self.exits_triggered = 0
        self.exit_reasons: Dict[str, int] = {}
    
    def register_entry(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        bar_index: int,
        timestamp_ns: int = 0
    ) -> None:
        """
        Register a new position entry.
        
        Call this when opening a position to initialize exit tracking.
        
        Args:
            symbol: Trading symbol
            quantity: Position size (signed)
            entry_price: Average entry price
            bar_index: Current bar index
            timestamp_ns: Entry timestamp
        """
        self._states[symbol] = PositionState(
            symbol=symbol,
            quantity=quantity,
            avg_entry_price=entry_price,
            entry_bar_index=bar_index,
            entry_timestamp_ns=timestamp_ns,
            peak_pnl=0.0,
            peak_price=entry_price,
        )
    
    def update_position(
        self,
        symbol: str,
        quantity: float,
        avg_entry_price: float
    ) -> None:
        """
        Update position state (e.g., after scaling in/out).
        
        Args:
            symbol: Trading symbol
            quantity: New position quantity
            avg_entry_price: New average entry price
        """
        if symbol in self._states:
            state = self._states[symbol]
            state.quantity = quantity
            state.avg_entry_price = avg_entry_price
        else:
            # Create new state if not exists
            self._states[symbol] = PositionState(
                symbol=symbol,
                quantity=quantity,
                avg_entry_price=avg_entry_price,
            )
    
    def close_position(self, symbol: str) -> None:
        """Clear position state when position is closed."""
        if symbol in self._states:
            del self._states[symbol]
    
    def check_exit(
        self,
        bar: Bar,
        position: float,
        avg_entry_price: float,
        bar_index: int,
        entry_bar_index: Optional[int] = None,
        strategy_id: str = "exit_mgr"
    ) -> Optional[tuple[Order, ExitReason]]:
        """
        Check if any exit condition is met.
        
        Args:
            bar: Current completed bar
            position: Current position quantity (signed)
            avg_entry_price: Position's average entry price
            bar_index: Current bar index
            entry_bar_index: Bar index when position was opened (for max_holding_bars)
            strategy_id: ID to tag on generated orders
            
        Returns:
            Tuple of (Order, ExitReason) if exit triggered, None otherwise
            
        FEATURE FLAGS:
        - use_intrabar_checks: When True, uses bar high/low for TP/SL checks
          This reduces optimistic bias from only checking close prices
        - leverage: When >1.0, P&L percentages are calculated relative to margin
        """
        symbol = bar.symbol
        current_price = bar.close
        
        # No exit if flat
        if abs(position) < 1e-10:
            return None
        
        # Get or create position state
        if symbol not in self._states:
            self._states[symbol] = PositionState(
                symbol=symbol,
                quantity=position,
                avg_entry_price=avg_entry_price,
                entry_bar_index=entry_bar_index or bar_index,
            )
        
        state = self._states[symbol]
        
        # Sync state with actual position
        state.quantity = position
        state.avg_entry_price = avg_entry_price
        
        # Update peak for trailing stop
        state.update_peak(current_price)
        
        # Get leverage from config (default 1.0 = spot)
        leverage = self.config.leverage
        
        # ================================================================
        # INTRA-BAR PRICE SELECTION (feature-flagged)
        # When enabled, use bar high/low to check if TP/SL would have
        # triggered during the bar, not just at close
        # ================================================================
        if self.config.use_intrabar_checks:
            # For longs: TP checks high, SL checks low
            # For shorts: TP checks low, SL checks high
            if position > 0:  # Long position
                tp_check_price = bar.high  # Best price for TP
                sl_check_price = bar.low   # Worst price for SL
            else:  # Short position
                tp_check_price = bar.low   # Best price for TP (lower is better)
                sl_check_price = bar.high  # Worst price for SL (higher is worse)
        else:
            # DEFAULT: Use close price for all checks (backward compatible)
            tp_check_price = current_price
            sl_check_price = current_price
        
        # Calculate P&L at different prices for different checks
        pnl_at_close = state.unrealized_pnl(current_price, leverage)
        pnl_pct_at_close = state.unrealized_pnl_pct(current_price, leverage)
        
        # For TP/SL checks, use appropriate prices
        pnl_for_tp = state.unrealized_pnl(tp_check_price, leverage)
        pnl_pct_for_tp = state.unrealized_pnl_pct(tp_check_price, leverage)
        pnl_for_sl = state.unrealized_pnl(sl_check_price, leverage)
        pnl_pct_for_sl = state.unrealized_pnl_pct(sl_check_price, leverage)
        
        # Use close-based P&L for trailing stop and other checks
        pnl = pnl_at_close
        pnl_pct = pnl_pct_at_close
        
        # Check exit conditions in priority order
        exit_reason = None
        
        # 1. Stop Loss USD - use SL-specific price (worst case)
        if self.config.stop_loss_usd is not None:
            if pnl_for_sl <= -self.config.stop_loss_usd:
                exit_reason = ExitReason.STOP_LOSS_USD
        
        # 2. Stop Loss Percentage - use SL-specific price
        if exit_reason is None and self.config.stop_loss_pct is not None:
            if pnl_pct_for_sl <= -self.config.stop_loss_pct:
                exit_reason = ExitReason.STOP_LOSS_PCT
        
        # 3. Take Profit USD - use TP-specific price (best case)
        if exit_reason is None and self.config.take_profit_usd is not None:
            if pnl_for_tp >= self.config.take_profit_usd:
                exit_reason = ExitReason.TAKE_PROFIT_USD
        
        # 4. Take Profit Percentage - use TP-specific price
        if exit_reason is None and self.config.take_profit_pct is not None:
            if pnl_pct_for_tp >= self.config.take_profit_pct:
                exit_reason = ExitReason.TAKE_PROFIT_PCT
        
        # 5. Trailing Stop - uses close price (pnl/pnl_pct)
        if exit_reason is None and self.config.trailing_stop_pct is not None:
            if state.peak_pnl > 0:  # Only trail after being in profit
                notional = state.notional
                if notional > 0:
                    peak_pnl_pct = state.peak_pnl / notional
                    current_pnl_pct = pnl / notional
                    drawdown_from_peak = peak_pnl_pct - current_pnl_pct
                    
                    if drawdown_from_peak >= self.config.trailing_stop_pct:
                        exit_reason = ExitReason.TRAILING_STOP
        
        # 6. Max Holding Bars
        if exit_reason is None and self.config.max_holding_bars is not None:
            bars_held = bar_index - state.entry_bar_index
            if bars_held >= self.config.max_holding_bars:
                exit_reason = ExitReason.MAX_HOLDING_BARS
        
        # Generate exit order if triggered
        if exit_reason is not None:
            order = self._create_exit_order(
                symbol=symbol,
                position=position,
                current_price=current_price,
                timestamp_ns=bar.timestamp_ns,
                strategy_id=strategy_id,
                reason=exit_reason,
            )
            
            self.exits_triggered += 1
            self.exit_reasons[exit_reason.value] = self.exit_reasons.get(exit_reason.value, 0) + 1
            
            log.debug(
                f"Exit triggered for {symbol}: {exit_reason.value}, "
                f"pnl=${pnl:.2f}, pnl_pct={pnl_pct:.2%}"
            )
            
            return order, exit_reason
        
        return None
    
    def _create_exit_order(
        self,
        symbol: str,
        position: float,
        current_price: float,
        timestamp_ns: int,
        strategy_id: str,
        reason: ExitReason,
    ) -> Order:
        """Create a reduce-only exit order."""
        # Opposite side to close position
        if position > 0:
            side = OrderSide.SELL
            quantity = abs(position)
        else:
            side = OrderSide.BUY
            quantity = abs(position)
        
        order_type = OrderType.MARKET if self.config.use_market_orders else OrderType.LIMIT
        
        return Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=current_price if order_type == OrderType.LIMIT else None,
            timestamp_ns=timestamp_ns,
            strategy_id=strategy_id,
            metadata={
                "exit_reason": reason.value,
                "reduce_only": self.config.reduce_only,
            },
        )
    
    def create_liquidation_order(
        self,
        symbol: str,
        position: float,
        current_price: float,
        timestamp_ns: int,
    ) -> Order:
        """
        Create a forced liquidation order.
        
        Called by engine when margin/maintenance requirements are breached.
        """
        if position > 0:
            side = OrderSide.SELL
        else:
            side = OrderSide.BUY
        
        return Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=abs(position),
            timestamp_ns=timestamp_ns,
            strategy_id="LIQUIDATION",
            metadata={
                "exit_reason": ExitReason.LIQUIDATION.value,
                "reduce_only": True,
            },
        )
    
    def check_exit_tick(
        self,
        symbol: str,
        tick_price: float,
        tick_timestamp_ns: int,
        position: float,
        avg_entry_price: float,
        strategy_id: str = "exit_mgr_tick"
    ) -> Optional[tuple[Order, ExitReason]]:
        """
        Check if TP/SL should trigger at the current tick price.
        
        This is the TICK-LEVEL exit check for intrabar exits.
        Called by the backtest engine on every tick when a position is open.
        
        Unlike check_exit() which uses bar data, this uses the raw tick price
        for immediate exit detection. This enables true intrabar exit simulation.
        
        Args:
            symbol: Trading symbol
            tick_price: Current tick price
            tick_timestamp_ns: Tick timestamp
            position: Current position quantity (signed)
            avg_entry_price: Position's average entry price
            strategy_id: ID to tag on generated orders
            
        Returns:
            Tuple of (Order, ExitReason) if exit triggered, None otherwise
        """
        # No exit if flat
        if abs(position) < 1e-10:
            return None
        
        # Get or create position state
        if symbol not in self._states:
            self._states[symbol] = PositionState(
                symbol=symbol,
                quantity=position,
                avg_entry_price=avg_entry_price,
            )
        
        state = self._states[symbol]
        
        # Sync state
        state.quantity = position
        state.avg_entry_price = avg_entry_price
        
        # Update peak for trailing stop
        state.update_peak(tick_price)
        
        # Get leverage
        leverage = self.config.leverage
        
        # Calculate P&L at tick price
        pnl = state.unrealized_pnl(tick_price, leverage)
        pnl_pct = state.unrealized_pnl_pct(tick_price, leverage)
        
        exit_reason = None
        
        # Check stop loss first (more important to exit on loss)
        if self.config.stop_loss_usd is not None:
            if pnl <= -self.config.stop_loss_usd:
                exit_reason = ExitReason.STOP_LOSS_USD
        
        if exit_reason is None and self.config.stop_loss_pct is not None:
            if pnl_pct <= -self.config.stop_loss_pct:
                exit_reason = ExitReason.STOP_LOSS_PCT
        
        # Check take profit
        if exit_reason is None and self.config.take_profit_usd is not None:
            if pnl >= self.config.take_profit_usd:
                exit_reason = ExitReason.TAKE_PROFIT_USD
        
        if exit_reason is None and self.config.take_profit_pct is not None:
            if pnl_pct >= self.config.take_profit_pct:
                exit_reason = ExitReason.TAKE_PROFIT_PCT
        
        # Check trailing stop
        if exit_reason is None and self.config.trailing_stop_pct is not None:
            if state.peak_pnl > 0:
                notional = state.notional
                if notional > 0:
                    peak_pnl_pct = state.peak_pnl / notional
                    current_pnl_pct = pnl / notional
                    drawdown_from_peak = peak_pnl_pct - current_pnl_pct
                    
                    if drawdown_from_peak >= self.config.trailing_stop_pct:
                        exit_reason = ExitReason.TRAILING_STOP
        
        # Generate exit order if triggered
        if exit_reason is not None:
            order = self._create_exit_order(
                symbol=symbol,
                position=position,
                current_price=tick_price,
                timestamp_ns=tick_timestamp_ns,
                strategy_id=strategy_id,
                reason=exit_reason,
            )
            
            # Add intrabar marker to metadata
            order.metadata["intrabar_exit"] = True
            order.metadata["exit_tick_price"] = tick_price
            
            self.exits_triggered += 1
            self.exit_reasons[exit_reason.value] = self.exit_reasons.get(exit_reason.value, 0) + 1
            
            log.debug(
                f"INTRABAR exit triggered for {symbol} at tick price {tick_price}: "
                f"{exit_reason.value}, pnl=${pnl:.2f}, pnl_pct={pnl_pct:.2%}"
            )
            
            return order, exit_reason
        
        return None
    
    def reset(self) -> None:
        """Reset exit manager state."""
        self._states.clear()
        self.exits_triggered = 0
        self.exit_reasons.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get exit statistics."""
        return {
            "exits_triggered": self.exits_triggered,
            "exit_reasons": dict(self.exit_reasons),
            "active_positions": len(self._states),
        }
