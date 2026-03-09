"""
Interfaces/IStrategy.py
=======================
Core strategy interface - the SINGLE contract for both Live and Backtest.

DESIGN PHILOSOPHY:
==================
This is the ONE strategy interface to rule them all. Both live trading and
backtesting use strategies implementing this interface. The interface supports
two modes of operation:

1. SIGNAL MODE (Legacy/Simple):
   - Implement generate_signal(symbol) -> Optional[str]
   - Returns "+1" (buy), "-1" (sell), or None (no action)
   - Good for simple strategies that don't need fine-grained control

2. DECISION MODE (Advanced):
   - Implement on_bar(bar, ctx) -> StrategyDecision
   - Returns a StrategyDecision with orders, features, and metadata
   - Good for complex strategies with multiple signals, TP/SL, etc.

Strategies can implement EITHER or BOTH methods. The strategy adapter will:
- Try on_bar() first if it exists
- Fall back to generate_signal() if on_bar() is not implemented
- Convert signals to orders automatically

BACKWARD COMPATIBILITY:
=======================
- Existing BinaryBaseStrategy.generate_signal() still works
- Existing IBacktestStrategy.on_bar() still works
- No breaking changes to existing strategies
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from Interfaces.market_data import Bar
    from Interfaces.orders import Order


@dataclass
class StrategyDecision:
    """
    Output from a strategy decision.
    
    This standardizes what a strategy can return:
    - orders: List of orders to execute (the primary output)
    - signal: Legacy signal value (for backward compatibility)
    - features: Strategy features/indicators for debugging/analysis
    - metadata: Additional data (entry price, TP/SL params, etc.)
    - regime_tags: Market regime tags (e.g., "trend_up", "range")
    
    Usage:
        # Simple order
        return StrategyDecision(orders=[buy_order])
        
        # With TP/SL metadata
        return StrategyDecision(
            orders=[buy_order],
            metadata={"tp_pct": 0.02, "sl_pct": 0.01}
        )
        
        # Just a signal (legacy mode)
        return StrategyDecision(signal="+1")
    """
    orders: List["Order"] = field(default_factory=list)
    signal: Optional[str] = None  # "+1", "-1", None for legacy compatibility
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    regime_tags: List[str] = field(default_factory=list)
    
    @classmethod
    def from_signal(cls, signal: Optional[str]) -> "StrategyDecision":
        """Create a decision from a legacy signal."""
        return cls(signal=signal)
    
    @classmethod
    def no_action(cls) -> "StrategyDecision":
        """Create an empty decision (no trade)."""
        return cls()
    
    @property
    def has_orders(self) -> bool:
        """Check if decision has any orders."""
        return len(self.orders) > 0
    
    @property
    def has_signal(self) -> bool:
        """Check if decision has a signal."""
        return self.signal is not None


class IStrategy(ABC):
    """
    Core strategy interface for both live and backtest.
    
    Strategies should implement AT LEAST ONE of:
    - generate_signal(symbol) -> Optional[str]  (legacy/simple)
    - on_bar(bar, ctx) -> StrategyDecision  (advanced)
    
    The system will automatically detect which methods are implemented
    and use the appropriate one.
    
    For backward compatibility:
    - If only generate_signal() is implemented, it will be wrapped
    - If only on_bar() is implemented, it will be used directly
    - If both are implemented, on_bar() takes precedence
    """
    
    def generate_signal(self, symbol: str) -> Optional[str]:
        """
        Generate a trading signal for the given symbol.
        
        Legacy method - implement this for simple signal-based strategies.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            "+1" for buy, "-1" for sell, None for no action
        """
        return None
    
    def on_bar(self, bar: "Bar", ctx: Any) -> "StrategyDecision":
        """
        Process a bar and return a trading decision.
        
        Advanced method - implement this for complex strategies.
        
        Args:
            bar: The completed bar
            ctx: Strategy context with market state
            
        Returns:
            StrategyDecision with orders, signals, and metadata
        """
        # Default implementation: use generate_signal if available
        signal = self.generate_signal(bar.symbol if hasattr(bar, 'symbol') else "")
        return StrategyDecision.from_signal(signal)
    
    def reset(self) -> None:
        """
        Reset strategy state for a new run.
        
        Called before each backtest iteration.
        Override this to reset internal state.
        """
        pass
    
    def get_exit_params(self) -> Dict[str, Any]:
        """
        Get exit parameters (TP/SL) for the strategy.
        
        Override to provide strategy-specific exit parameters.
        These will be used by the ExitManager for intrabar exit checks.
        
        Returns:
            Dict with keys like 'tp_pct', 'sl_pct', 'tp_price', 'sl_price'
        """
        return {}
    
    def get_exit_manager(self) -> Optional[Any]:
        """
        Get the strategy's exit manager if it has one.
        
        Returns:
            ExitManager instance or None
        """
        return getattr(self, 'exit_manager', None)
