"""
Interfaces/risk_interface.py
============================
Risk management interface for pre-trade checks.

Design decisions:
- Risk manager performs pre-trade validation
- Can reject orders that violate risk limits
- Tracks cumulative metrics (daily loss, etc.)
- Kill-switch capability for emergency stops
"""
from __future__ import annotations
from typing import Protocol, Optional, Dict, Any, runtime_checkable

from Interfaces.orders import Order
from Interfaces.market_data import Bar


@runtime_checkable
class IRiskManager(Protocol):
    """
    Protocol for risk management.
    
    Risk managers validate orders before execution and can:
    - Reject orders that exceed position limits
    - Enforce daily loss limits
    - Trigger kill-switch on excessive drawdown
    """
    
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
        ...
    
    def is_kill_switch_active(self) -> bool:
        """
        Check if kill switch has been triggered.
        
        Returns:
            True if trading should be halted
        """
        ...
    
    def update_daily_pnl(self, pnl: float) -> None:
        """
        Update daily P&L tracking.
        
        Args:
            pnl: P&L change to add
        """
        ...
    
    def reset_daily(self) -> None:
        """Reset daily metrics (called at day boundary)."""
        ...
    
    def reset(self) -> None:
        """Full reset for new backtest run."""
        ...
    
    def get_state(self) -> Dict[str, Any]:
        """Get current risk state for logging/metrics."""
        ...


# Forward reference
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass
