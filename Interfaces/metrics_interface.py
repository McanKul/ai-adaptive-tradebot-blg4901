"""
Interfaces/metrics_interface.py
===============================
Metrics sink interface for collecting backtest results.

Design decisions:
- Collects data incrementally during backtest
- finalize() computes final metrics and returns structured result
- Stores trades list for analysis
- Tracks equity curve for drawdown/return calculations
"""
from __future__ import annotations
from typing import Protocol, Dict, List, Any, Optional, runtime_checkable
from dataclasses import dataclass, field

from Interfaces.market_data import Bar
from Interfaces.orders import Order, Fill


@dataclass
class BacktestResult:
    """
    Structured backtest result.
    
    Contains all metrics and data from a single backtest run.
    """
    # Identification
    strategy_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    data_config: Dict[str, Any] = field(default_factory=dict)
    
    # Time range
    start_timestamp_ns: int = 0
    end_timestamp_ns: int = 0
    
    # Capital
    initial_capital: float = 0.0
    final_equity: float = 0.0
    
    # Returns
    total_return: float = 0.0  # (final - initial) / initial
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0  # As fraction (0.1 = 10%)
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    
    # Costs
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_costs: float = 0.0
    
    # Turnover
    turnover: float = 0.0  # Total traded notional / avg equity
    
    # Time series data
    equity_curve: List[tuple[int, float]] = field(default_factory=list)  # (timestamp_ns, equity)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "params": self.params,
            "data_config": self.data_config,
            "start_timestamp_ns": self.start_timestamp_ns,
            "end_timestamp_ns": self.end_timestamp_ns,
            "initial_capital": self.initial_capital,
            "final_equity": self.final_equity,
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "annualized_return": self.annualized_return,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_return": self.avg_trade_return,
            "total_fees": self.total_fees,
            "total_slippage": self.total_slippage,
            "total_costs": self.total_costs,
            "turnover": self.turnover,
            "equity_curve": self.equity_curve,
            "trades": self.trades,
            "metadata": self.metadata,
        }


@runtime_checkable
class IMetricsSink(Protocol):
    """
    Protocol for metrics collection during backtest.
    
    Called by backtest engine at various points to collect data.
    finalize() computes final metrics.
    """
    
    def on_bar(self, bar: Bar, equity: float, position: float) -> None:
        """
        Called after each bar is processed.
        
        Args:
            bar: The completed bar
            equity: Current portfolio equity
            position: Current position quantity
        """
        ...
    
    def on_order(self, order: Order) -> None:
        """
        Called when an order is submitted.
        
        Args:
            order: The submitted order
        """
        ...
    
    def on_fill(self, fill: Fill, portfolio_equity: float) -> None:
        """
        Called when an order is filled.
        
        Args:
            fill: The fill details
            portfolio_equity: Portfolio equity after fill
        """
        ...
    
    def finalize(
        self,
        initial_capital: float,
        final_equity: float,
        start_ts: int,
        end_ts: int
    ) -> BacktestResult:
        """
        Compute final metrics and return result.
        
        Args:
            initial_capital: Starting capital
            final_equity: Ending equity
            start_ts: Start timestamp (ns)
            end_ts: End timestamp (ns)
            
        Returns:
            Complete BacktestResult
        """
        ...
    
    def reset(self) -> None:
        """Reset metrics for new run."""
        ...
