"""
Backtest/metrics.py
===================
Metrics collection and computation for backtest results.

Design decisions:
- Collects data incrementally during backtest
- Computes standard metrics: return, Sharpe, drawdown, etc.
- Stores equity curve and trade list
- Returns BacktestResult dataclass
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import math
import logging

from Interfaces.market_data import Bar
from Interfaces.orders import Order, Fill, OrderSide
from Interfaces.metrics_interface import IMetricsSink, BacktestResult

log = logging.getLogger(__name__)


# Seconds per year for annualization
SECONDS_PER_YEAR = 365.25 * 24 * 3600
NS_PER_YEAR = SECONDS_PER_YEAR * 1e9


class MetricsSink:
    """
    Metrics collector for backtest.
    
    Collects data during backtest and computes final metrics.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.0,
        annualization_factor: float = 252.0  # Trading days per year
    ):
        """
        Initialize MetricsSink.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            annualization_factor: Factor for annualizing returns (252 for daily)
        """
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
        
        # Data collection
        self._equity_curve: List[tuple[int, float]] = []  # (timestamp_ns, equity)
        self._returns: List[float] = []  # Bar-to-bar returns
        self._orders: List[Dict[str, Any]] = []
        self._fills: List[Dict[str, Any]] = []
        self._trades: List[Dict[str, Any]] = []  # Round-trip trades
        
        # Aggregates
        self._total_fees = 0.0
        self._total_slippage = 0.0
        self._total_spread_cost = 0.0
        self._total_slippage_cost = 0.0
        self._total_funding_cost = 0.0
        self._total_borrow_cost = 0.0
        self._bar_count = 0
        self._peak_equity = 0.0
        self._max_drawdown = 0.0
        
        # For round-trip tracking
        self._open_trades: Dict[str, Dict[str, Any]] = {}  # symbol -> trade info
        
        # Strategy info (set by runner)
        self.strategy_name = ""
        self.params: Dict[str, Any] = {}
        self.data_config: Dict[str, Any] = {}
    
    def on_bar(self, bar: Bar, equity: float, position: float) -> None:
        """
        Called after each bar is processed.
        
        Args:
            bar: The completed bar
            equity: Current portfolio equity
            position: Current position quantity
        """
        # Record equity
        self._equity_curve.append((bar.timestamp_ns, equity))
        
        # Compute return
        if len(self._equity_curve) > 1:
            prev_equity = self._equity_curve[-2][1]
            if prev_equity > 0:
                ret = (equity - prev_equity) / prev_equity
                self._returns.append(ret)
        
        # Update peak and drawdown
        if equity > self._peak_equity:
            self._peak_equity = equity
        
        if self._peak_equity > 0:
            dd = (self._peak_equity - equity) / self._peak_equity
            self._max_drawdown = max(self._max_drawdown, dd)
        
        self._bar_count += 1
    
    def on_order(self, order: Order) -> None:
        """Called when an order is submitted."""
        self._orders.append({
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.name,
            "type": order.order_type.name,
            "quantity": order.quantity,
            "price": order.price,
            "timestamp_ns": order.timestamp_ns,
            "strategy_id": order.strategy_id,
        })
    
    def on_fill(self, fill: Fill, portfolio_equity: float) -> None:
        """Called when an order is filled."""
        self._fills.append({
            "fill_id": fill.fill_id,
            "order_id": fill.order_id,
            "symbol": fill.symbol,
            "side": fill.side.name,
            "price": fill.fill_price,
            "quantity": fill.fill_quantity,
            "fee": fill.fee,
            "slippage": fill.slippage,
            "timestamp_ns": fill.timestamp_ns,
            "portfolio_equity": portfolio_equity,
        })
        
        self._total_fees += fill.fee
        self._total_slippage += abs(fill.slippage)
        
        # Decomposed cost tracking (A3) – read from fill.metadata if present
        cb = fill.metadata.get("cost_breakdown") if fill.metadata else None
        if cb:
            self._total_spread_cost += cb.get("spread_cost_quote", 0.0)
            self._total_slippage_cost += cb.get("slippage_cost_quote", 0.0)
        else:
            # Legacy fills: attribute everything to slippage bucket
            self._total_slippage_cost += abs(fill.slippage)
        
        # Track round-trip trades
        self._update_trades(fill)
    
    def _update_trades(self, fill: Fill) -> None:
        """
        Track round-trip trades (bets) using SIGN-CHANGE detection.
        
        AFML Definition: A bet ends when position is FLATTENED or FLIPPED.
        - Flattened: position goes to 0 (close bet)
        - Flipped: position sign changes (close old bet, open new bet for remaining)
        
        This replaces the old reversal logic that incorrectly used entry_side comparison.
        """
        symbol = fill.symbol
        EPS = 1e-10
        
        # Helper to get sign: +1, -1, or 0
        def sign(x: float) -> int:
            if x > EPS:
                return 1
            elif x < -EPS:
                return -1
            return 0
        
        if symbol not in self._open_trades:
            # New bet - only open if fill creates a position
            new_qty = fill.fill_quantity if fill.is_buy else -fill.fill_quantity
            if abs(new_qty) > EPS:
                self._open_trades[symbol] = {
                    "symbol": symbol,
                    "entry_side": "LONG" if new_qty > 0 else "SHORT",
                    "entry_price": fill.fill_price,
                    "entry_quantity": abs(new_qty),
                    "entry_timestamp_ns": fill.timestamp_ns,
                    "entry_fee": fill.fee,
                    "quantity": new_qty,  # Signed quantity
                    "total_fees": fill.fee,
                    "fills": [fill.fill_quantity],
                    "prices": [fill.fill_price],
                }
        else:
            trade = self._open_trades[symbol]
            prev_qty = trade["quantity"]
            prev_sign = sign(prev_qty)
            
            # Update quantity based on fill side
            if fill.is_buy:
                new_qty = prev_qty + fill.fill_quantity
            else:
                new_qty = prev_qty - fill.fill_quantity
            
            new_sign = sign(new_qty)
            
            # Accumulate fees and track fills
            trade["total_fees"] = trade.get("total_fees", trade["entry_fee"]) + fill.fee
            trade["fills"] = trade.get("fills", [trade["entry_quantity"]]) + [fill.fill_quantity]
            trade["prices"] = trade.get("prices", [trade["entry_price"]]) + [fill.fill_price]
            
            # CASE 1: Position FLATTENED (new_qty ≈ 0) -> close bet
            if abs(new_qty) < EPS:
                self._close_bet(trade, fill.fill_price, fill.timestamp_ns)
                del self._open_trades[symbol]
            
            # CASE 2: Position FLIPPED (sign changed) -> close old bet, open new bet
            elif prev_sign != 0 and new_sign != 0 and prev_sign != new_sign:
                # Close the old bet at the fill price
                self._close_bet(trade, fill.fill_price, fill.timestamp_ns)
                
                # Open new bet for the remaining quantity (the flip amount)
                self._open_trades[symbol] = {
                    "symbol": symbol,
                    "entry_side": "LONG" if new_qty > 0 else "SHORT",
                    "entry_price": fill.fill_price,
                    "entry_quantity": abs(new_qty),
                    "entry_timestamp_ns": fill.timestamp_ns,
                    "entry_fee": fill.fee,  # Allocate this fill's fee to new bet
                    "quantity": new_qty,
                    "total_fees": fill.fee,
                    "fills": [abs(new_qty)],
                    "prices": [fill.fill_price],
                }
            
            # CASE 3: Same sign (scale in/out) -> continue bet
            else:
                trade["quantity"] = new_qty
    
    def _close_bet(self, trade: Dict[str, Any], exit_price: float, exit_ts: int) -> None:
        """Close a bet and record the round-trip trade."""
        # Calculate P&L based on entry side
        entry_qty = trade["entry_quantity"]
        entry_price = trade["entry_price"]
        total_fees = trade.get("total_fees", trade["entry_fee"])
        
        if trade["entry_side"] == "LONG":
            gross_pnl = (exit_price - entry_price) * entry_qty
        else:  # SHORT
            gross_pnl = (entry_price - exit_price) * entry_qty
        
        net_pnl = gross_pnl - total_fees
        
        self._trades.append({
            "symbol": trade["symbol"],
            "entry_side": trade["entry_side"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": entry_qty,
            "entry_timestamp_ns": trade["entry_timestamp_ns"],
            "exit_timestamp_ns": exit_ts,
            "gross_pnl": gross_pnl,
            "pnl": net_pnl,
            "fees": total_fees,
            "return_pct": net_pnl / (entry_price * entry_qty) if entry_price > 0 else 0,
        })
    
    def finalize(
        self,
        initial_capital: float,
        final_equity: float,
        start_ts: int,
        end_ts: int,
        portfolio_max_drawdown: Optional[float] = None,
        leverage_stats: Optional[Dict[str, float]] = None,
        leverage_mode: str = "spot",
        liquidation_count: int = 0,
    ) -> BacktestResult:
        """
        Compute final metrics and return result.
        
        Args:
            initial_capital: Starting capital
            final_equity: Ending equity
            start_ts: Start timestamp (ns)
            end_ts: End timestamp (ns)
            portfolio_max_drawdown: Max drawdown from portfolio (for consistency with kill-switch)
            
        Returns:
            Complete BacktestResult
        """
        result = BacktestResult(
            strategy_name=self.strategy_name,
            params=self.params.copy(),
            data_config=self.data_config.copy(),
            start_timestamp_ns=start_ts,
            end_timestamp_ns=end_ts,
            initial_capital=initial_capital,
            final_equity=final_equity,
        )
        
        # Returns
        # BUG FIX: total_return is the dollar amount, total_return_pct is PERCENTAGE (multiply by 100)
        result.total_return = final_equity - initial_capital
        if initial_capital > 0:
            result.total_return_pct = ((final_equity / initial_capital) - 1.0) * 100.0  # As percentage
        else:
            result.total_return_pct = 0.0
        
        # Annualized return
        duration_ns = end_ts - start_ts
        if duration_ns > 0:
            years = duration_ns / NS_PER_YEAR
            if years > 0 and final_equity > 0 and initial_capital > 0:
                result.annualized_return = (final_equity / initial_capital) ** (1 / years) - 1
        
        # Drawdown - use portfolio's max_drawdown if provided (for consistency with kill-switch)
        # Otherwise fall back to metrics' computed drawdown
        if portfolio_max_drawdown is not None:
            result.max_drawdown = portfolio_max_drawdown
        else:
            result.max_drawdown = self._max_drawdown
        result.max_drawdown_pct = result.max_drawdown * 100
        
        # Sharpe ratio
        if len(self._returns) > 1:
            avg_return = sum(self._returns) / len(self._returns)
            variance = sum((r - avg_return) ** 2 for r in self._returns) / (len(self._returns) - 1)
            std_return = math.sqrt(variance) if variance > 0 else 0
            
            if std_return > 0:
                # Annualize
                excess_return = avg_return - self.risk_free_rate / self.annualization_factor
                result.sharpe_ratio = excess_return / std_return * math.sqrt(self.annualization_factor)
            
            # Sortino ratio (downside deviation)
            downside_returns = [r for r in self._returns if r < 0]
            if len(downside_returns) > 1:
                downside_variance = sum(r ** 2 for r in downside_returns) / len(downside_returns)
                downside_std = math.sqrt(downside_variance)
                if downside_std > 0:
                    result.sortino_ratio = avg_return / downside_std * math.sqrt(self.annualization_factor)
        
        # Calmar ratio
        if result.max_drawdown > 0 and result.annualized_return != 0:
            result.calmar_ratio = result.annualized_return / result.max_drawdown
        
        # Trade statistics
        # NOTE: total_trades counts ROUND-TRIP trades (entry + exit), not individual fills.
        # Use metadata["fill_count"] for total fill events.
        result.total_trades = len(self._trades)
        winning_trades = [t for t in self._trades if t["pnl"] > 0]
        losing_trades = [t for t in self._trades if t["pnl"] <= 0]
        
        result.winning_trades = len(winning_trades)
        result.losing_trades = len(losing_trades)
        result.win_rate = len(winning_trades) / len(self._trades) if self._trades else 0
        
        total_profit = sum(t["pnl"] for t in winning_trades)
        total_loss = abs(sum(t["pnl"] for t in losing_trades))
        result.profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
        
        if self._trades:
            result.avg_trade_return = sum(t["return_pct"] for t in self._trades) / len(self._trades)
        
        # Costs
        result.total_fees = self._total_fees
        result.total_slippage = self._total_slippage
        result.total_costs = self._total_fees + self._total_slippage
        
        # Turnover (total traded notional / average equity)
        total_traded = sum(f["price"] * f["quantity"] for f in self._fills)
        if self._equity_curve:
            avg_equity = sum(e[1] for e in self._equity_curve) / len(self._equity_curve)
            result.turnover = total_traded / avg_equity if avg_equity > 0 else 0
        
        # Time series data
        result.equity_curve = self._equity_curve.copy()
        result.trades = self._trades.copy()
        
        # Metadata - clear reporting to avoid confusion
        # - fill_count: Individual fill events (orders executed)
        # - closed_bets: Round-trip trades (entry -> exit, position flattened/flipped)
        # - open_bets: Positions still open at end of backtest
        result.metadata = {
            "bar_count": self._bar_count,
            "order_count": len(self._orders),
            "fill_count": len(self._fills),
            "closed_bets": len(self._trades),
            "open_bets": len(self._open_trades),
            # Backwards compatibility
            "round_trip_count": len(self._trades),
            "open_trades": len(self._open_trades),
            # NEW: Leverage mode info
            "leverage_mode": leverage_mode,
            "liquidation_count": liquidation_count,
        }
        
        # Decomposed cost buckets (A3) — placed AFTER metadata dict init to avoid overwrite
        result.metadata["total_fee_cost"] = self._total_fees
        result.metadata["total_spread_cost"] = self._total_spread_cost
        result.metadata["total_slippage_cost"] = self._total_slippage_cost
        result.metadata["total_funding_cost"] = self._total_funding_cost
        result.metadata["total_borrow_cost"] = self._total_borrow_cost
        
        # NEW: Add leverage statistics (AFML metrics)
        if leverage_stats:
            result.metadata["avg_exposure"] = leverage_stats.get("avg_exposure", 0.0)
            result.metadata["max_exposure"] = leverage_stats.get("max_exposure", 0.0)
            result.metadata["avg_equity"] = leverage_stats.get("avg_equity", 0.0)
            result.metadata["avg_leverage_afml"] = leverage_stats.get("avg_leverage_afml", 0.0)
            result.metadata["max_leverage"] = leverage_stats.get("max_leverage", 0.0)
        
        return result
    
    def reset(self, initial_capital: float = 0.0) -> None:
        """Reset metrics for new run."""
        self._equity_curve.clear()
        self._returns.clear()
        self._orders.clear()
        self._fills.clear()
        self._trades.clear()
        self._open_trades.clear()
        self._total_fees = 0.0
        self._total_slippage = 0.0
        self._total_spread_cost = 0.0
        self._total_slippage_cost = 0.0
        self._total_funding_cost = 0.0
        self._total_borrow_cost = 0.0
        self._bar_count = 0
        # Initialize peak_equity to initial_capital to match portfolio behavior
        self._peak_equity = initial_capital if initial_capital > 0 else 0.0
        self._max_drawdown = 0.0
