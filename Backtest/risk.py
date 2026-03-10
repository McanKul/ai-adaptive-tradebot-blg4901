"""
Backtest/risk.py
================
Risk management for backtest simulation.

Components:
- BasicRiskManager: Simple but real risk controls

Features:
- Maximum position size limit
- Maximum daily loss limit
- Kill switch on excessive drawdown
- Per-symbol and portfolio-wide limits
- NEW: Proper leverage calculation (symbol-aware)
- NEW: Margin availability checks for margin mode
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import logging

from Interfaces.orders import Order, OrderSide
from Interfaces.market_data import Bar
from Interfaces.risk_interface import IRiskManager

log = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limit configuration."""
    max_position_size: float = 10.0  # Maximum position quantity per symbol
    max_position_notional: float = 100000.0  # Maximum position value
    max_daily_loss: float = 1000.0  # Maximum daily loss in currency
    max_daily_loss_pct: float = 0.1  # Maximum daily loss as fraction of equity
    max_drawdown: float = 0.2  # Kill switch drawdown threshold (20%)
    max_orders_per_bar: int = 10  # Maximum orders per bar (prevent spam)
    max_leverage: float = 10.0  # Maximum leverage
    
    # Margin mode specific
    enforce_margin_check: bool = True  # Reject orders if insufficient margin


class BasicRiskManager:
    """
    Basic risk manager with essential controls.
    
    Validates orders before execution and can trigger kill switch.
    """
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        """
        Initialize BasicRiskManager.
        
        Args:
            limits: RiskLimits configuration (uses defaults if None)
        """
        self.limits = limits or RiskLimits()
        
        # Daily tracking
        self._daily_pnl = 0.0
        self._daily_start_equity = 0.0
        self._orders_this_bar = 0
        
        # Kill switch
        self._kill_switch_active = False
        self._kill_switch_reason = ""
        
        # Statistics
        self._rejected_orders = 0
        self._approved_orders = 0
    
    def pre_trade_check(
        self,
        order: Order,
        portfolio: "Any",
        bar: Bar
    ) -> bool:
        """
        Validate an order before execution.
        
        Args:
            order: Order to validate
            portfolio: Current portfolio state
            bar: Current bar (for price reference)
            
        Returns:
            True if order passes risk checks, False to reject
        """
        # Kill switch check first
        if self._kill_switch_active:
            self._rejected_orders += 1
            log.warning(f"Order rejected: kill switch active ({self._kill_switch_reason})")
            return False
        
        # Order count limit
        if self._orders_this_bar >= self.limits.max_orders_per_bar:
            self._rejected_orders += 1
            log.debug(f"Order rejected: max orders per bar reached")
            return False
        
        # Position size check
        current_position = portfolio.position_quantity(order.symbol)
        if order.side == OrderSide.BUY:
            new_position = current_position + order.quantity
        else:
            new_position = current_position - order.quantity
        
        if abs(new_position) > self.limits.max_position_size:
            self._rejected_orders += 1
            log.debug(
                f"Order rejected: position size {abs(new_position):.4f} > "
                f"limit {self.limits.max_position_size}"
            )
            return False
        
        # Position notional check
        price = bar.close
        new_notional = abs(new_position) * price
        if new_notional > self.limits.max_position_notional:
            self._rejected_orders += 1
            log.debug(
                f"Order rejected: notional {new_notional:.2f} > "
                f"limit {self.limits.max_position_notional}"
            )
            return False
        
        # FIXED: Leverage check - correctly handle reducing positions
        # Compute new total exposure by updating only the symbol being traded
        equity = portfolio.equity()
        if equity > 0:
            # Get current total exposure
            total_exposure = portfolio.total_exposure()
            
            # Compute old and new exposure for this symbol specifically
            old_sym_exposure = portfolio.symbol_exposure(order.symbol)
            new_sym_exposure = abs(new_position) * price
            
            # New total exposure = total - old_symbol + new_symbol
            new_total_exposure = total_exposure - old_sym_exposure + new_sym_exposure
            
            leverage = new_total_exposure / equity
            
            if leverage > self.limits.max_leverage:
                self._rejected_orders += 1
                log.debug(
                    f"Order rejected: leverage {leverage:.2f} > "
                    f"limit {self.limits.max_leverage}"
                )
                return False
        
        # NEW: Margin availability check (for margin mode)
        if self.limits.enforce_margin_check and hasattr(portfolio, 'is_margin_mode') and portfolio.is_margin_mode:
            delta_margin = portfolio.margin_for_order(
                symbol=order.symbol,
                order_qty=order.quantity,
                order_price=price,
                is_buy=(order.side == OrderSide.BUY)
            )
            
            if delta_margin > 0:  # Order requires additional margin
                available = portfolio.available_margin()
                if delta_margin > available:
                    self._rejected_orders += 1
                    log.debug(
                        f"Order rejected: insufficient margin. "
                        f"Need {delta_margin:.2f}, available {available:.2f}"
                    )
                    return False
        
        # Daily loss check
        if self._daily_pnl < -self.limits.max_daily_loss:
            self._activate_kill_switch("daily loss limit exceeded")
            self._rejected_orders += 1
            return False
        
        if self._daily_start_equity > 0:
            daily_loss_pct = -self._daily_pnl / self._daily_start_equity
            if daily_loss_pct > self.limits.max_daily_loss_pct:
                self._activate_kill_switch("daily loss % limit exceeded")
                self._rejected_orders += 1
                return False
        
        # All checks passed
        self._orders_this_bar += 1
        self._approved_orders += 1
        return True
    
    def check_drawdown(self, portfolio: "Any") -> bool:
        """
        Check if drawdown exceeds limit.
        
        Args:
            portfolio: Portfolio to check
            
        Returns:
            True if within limits, False if kill switch triggered
        """
        drawdown = portfolio.max_drawdown
        
        if drawdown > self.limits.max_drawdown:
            self._activate_kill_switch(
                f"drawdown {drawdown:.1%} > limit {self.limits.max_drawdown:.1%}"
            )
            return False
        
        return True
    
    def _activate_kill_switch(self, reason: str) -> None:
        """Activate the kill switch."""
        if not self._kill_switch_active:
            self._kill_switch_active = True
            self._kill_switch_reason = reason
            log.warning(f"KILL SWITCH ACTIVATED: {reason}")
    
    def is_kill_switch_active(self) -> bool:
        """Check if kill switch has been triggered."""
        return self._kill_switch_active
    
    def update_daily_pnl(self, pnl: float) -> None:
        """Update daily P&L tracking."""
        self._daily_pnl += pnl
    
    def set_daily_start_equity(self, equity: float) -> None:
        """Set the starting equity for daily loss calculation."""
        self._daily_start_equity = equity
    
    def on_new_bar(self) -> None:
        """Called at start of each bar to reset per-bar counters."""
        self._orders_this_bar = 0
    
    def reset_daily(self) -> None:
        """Reset daily metrics (called at day boundary)."""
        self._daily_pnl = 0.0
        self._orders_this_bar = 0
    
    def reset(self) -> None:
        """Full reset for new backtest run."""
        self._daily_pnl = 0.0
        self._daily_start_equity = 0.0
        self._orders_this_bar = 0
        self._kill_switch_active = False
        self._kill_switch_reason = ""
        self._rejected_orders = 0
        self._approved_orders = 0
    
    def get_state(self) -> Dict[str, Any]:
        """Get current risk state for logging/metrics."""
        return {
            "kill_switch_active": self._kill_switch_active,
            "kill_switch_reason": self._kill_switch_reason,
            "daily_pnl": self._daily_pnl,
            "daily_start_equity": self._daily_start_equity,
            "orders_this_bar": self._orders_this_bar,
            "rejected_orders": self._rejected_orders,
            "approved_orders": self._approved_orders,
        }
