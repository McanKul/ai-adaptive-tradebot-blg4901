"""
Backtest/portfolio.py
=====================
Portfolio management for backtest simulation.

Design decisions:
- Tracks cash, positions, and average entry prices
- Applies fills from execution model
- Computes equity using last close price
- Tracks realized and unrealized P&L
- Records fee and slippage totals
- Supports both long and short positions
- NEW: Supports spot mode (default) and margin mode for leverage simulation
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
import logging

from Interfaces.orders import Fill, OrderSide
from Interfaces.market_data import Bar

log = logging.getLogger(__name__)


class LeverageMode(Enum):
    """Portfolio leverage mode."""
    SPOT = "spot"  # Full notional paid from cash (traditional)
    MARGIN = "margin"  # Collateral-based with margin requirements


@dataclass
class MarginConfig:
    """Configuration for margin mode."""
    leverage: float = 10.0  # Max leverage (e.g., 10x)
    maintenance_margin_ratio: float = 0.5  # Maintenance = initial * this ratio
    liquidation_buffer: float = 0.01  # Extra buffer before liquidation (1%)
    
    @property
    def initial_margin_rate(self) -> float:
        """Initial margin requirement as fraction of notional."""
        return 1.0 / self.leverage
    
    @property
    def maintenance_margin_rate(self) -> float:
        """Maintenance margin requirement as fraction of notional."""
        return self.initial_margin_rate * self.maintenance_margin_ratio


@dataclass
class Position:
    """
    A position in a single symbol.
    
    Positive quantity = long position
    Negative quantity = short position
    """
    symbol: str
    quantity: float = 0.0  # Positive = long, negative = short
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    entry_count: int = 0  # Number of entries
    exit_count: int = 0  # Number of exits
    
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
        """Notional value using average entry price."""
        return abs(self.quantity) * self.avg_entry_price
    
    def current_notional(self, current_price: float) -> float:
        """Notional value at current price."""
        return abs(self.quantity) * current_price
    
    def unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L at current price.
        
        For long: (current - avg_entry) * quantity
        For short: (avg_entry - current) * abs(quantity)
        """
        if self.is_flat:
            return 0.0
        return (current_price - self.avg_entry_price) * self.quantity
    
    def apply_fill(self, fill: Fill) -> float:
        """
        Apply a fill to this position.
        
        Args:
            fill: The fill to apply
            
        Returns:
            Realized P&L from this fill (if closing/reducing position)
        """
        if fill.symbol != self.symbol:
            raise ValueError(f"Fill symbol {fill.symbol} doesn't match position {self.symbol}")
        
        realized_pnl = 0.0
        fill_qty = fill.fill_quantity if fill.is_buy else -fill.fill_quantity
        
        # Track costs
        self.total_fees += fill.fee
        self.total_slippage += abs(fill.slippage)
        
        # Case 1: Flat position, opening new
        if self.is_flat:
            self.quantity = fill_qty
            self.avg_entry_price = fill.fill_price
            self.entry_count += 1
        
        # Case 2: Adding to existing position (same direction)
        elif (self.is_long and fill.is_buy) or (self.is_short and fill.is_sell):
            # Weighted average entry price
            old_notional = abs(self.quantity) * self.avg_entry_price
            new_notional = fill.fill_quantity * fill.fill_price
            new_qty = abs(self.quantity) + fill.fill_quantity
            self.avg_entry_price = (old_notional + new_notional) / new_qty
            self.quantity += fill_qty
            self.entry_count += 1
        
        # Case 3: Reducing or closing position (opposite direction)
        else:
            close_qty = min(abs(self.quantity), fill.fill_quantity)
            
            # Realize P&L
            if self.is_long:
                realized_pnl = (fill.fill_price - self.avg_entry_price) * close_qty
            else:  # is_short
                realized_pnl = (self.avg_entry_price - fill.fill_price) * close_qty
            
            self.realized_pnl += realized_pnl
            self.exit_count += 1
            
            # Update position
            remaining_fill = fill.fill_quantity - close_qty
            self.quantity += fill_qty  # This reduces or flips the position
            
            # If flipped to opposite side, reset avg price to fill price
            if (self.is_long and fill.is_sell and remaining_fill > 0) or \
               (self.is_short and fill.is_buy and remaining_fill > 0):
                self.avg_entry_price = fill.fill_price
                self.entry_count += 1
        
        return realized_pnl


class Portfolio:
    """
    Portfolio tracker for backtesting.
    
    Manages cash, positions, and computes equity/P&L.
    
    Supports two modes:
    - SPOT (default): Cash pays full notional. Traditional spot trading.
    - MARGIN: Cash is collateral. Only fees affect cash immediately.
              P&L is realized when positions are closed.
    """
    
    def __init__(
        self,
        initial_cash: float = 10000.0,
        margin_requirement: float = 0.1,  # Legacy: 10x leverage max
        leverage_mode: LeverageMode = LeverageMode.SPOT,
        margin_config: Optional[MarginConfig] = None,
    ):
        """
        Initialize portfolio.
        
        Args:
            initial_cash: Starting cash/collateral balance
            margin_requirement: Legacy fraction of position value required as margin
            leverage_mode: SPOT or MARGIN mode
            margin_config: Configuration for margin mode
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash  # In MARGIN mode, this is "collateral"
        self.margin_requirement = margin_requirement
        self.leverage_mode = leverage_mode
        self.margin_config = margin_config or MarginConfig(leverage=1.0 / margin_requirement)
        
        self._positions: Dict[str, Position] = {}
        self._last_prices: Dict[str, float] = {}
        self._trade_history: List[Dict[str, Any]] = []
        
        # Aggregate metrics
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.total_realized_pnl = 0.0
        self.peak_equity = initial_cash
        self.max_drawdown = 0.0
        self.total_traded_notional = 0.0
        
        # Margin mode tracking
        self._liquidations: List[Dict[str, Any]] = []
        
        # Leverage statistics for AFML metrics
        self._exposure_samples: List[float] = []  # Total exposure at each bar
        self._equity_samples: List[float] = []  # Equity at each bar (for avg AUM)
    
    @property
    def is_margin_mode(self) -> bool:
        """Check if running in margin mode."""
        return self.leverage_mode == LeverageMode.MARGIN
    
    @property
    def collateral(self) -> float:
        """Collateral balance (alias for cash in margin mode)."""
        return self.cash
    
    def get_position(self, symbol: str) -> Position:
        """Get or create position for symbol."""
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)
        return self._positions[symbol]
    
    def position_quantity(self, symbol: str) -> float:
        """Get position quantity for symbol (0 if no position)."""
        pos = self._positions.get(symbol)
        return pos.quantity if pos else 0.0
    
    def position_avg_entry(self, symbol: str) -> float:
        """Get position average entry price for symbol."""
        pos = self._positions.get(symbol)
        return pos.avg_entry_price if pos else 0.0
    
    def update_price(self, symbol: str, price: float) -> None:
        """Update last known price for a symbol."""
        self._last_prices[symbol] = price
    
    def apply_fill(self, fill: Fill) -> float:
        """
        Apply a fill to the portfolio.
        
        In SPOT mode: Cash is debited/credited for full notional.
        In MARGIN mode: Only fees affect collateral. P&L realized on close.
        
        Args:
            fill: The fill to apply
            
        Returns:
            Realized P&L from this fill
        """
        position = self.get_position(fill.symbol)
        old_qty = position.quantity
        
        # Apply to position
        realized_pnl = position.apply_fill(fill)
        
        if self.is_margin_mode:
            # MARGIN MODE: Only fees affect collateral immediately
            # Realized P&L is added to collateral when position reduces/closes
            self.cash -= fill.fee  # Fees always deducted
            self.cash += realized_pnl  # Realized P&L added to collateral
        else:
            # SPOT MODE: Cash is debited/credited for full notional
            if fill.is_buy:
                self.cash -= fill.notional + fill.fee
            else:
                self.cash += fill.notional - fill.fee
        
        # Track aggregates
        self.total_fees += fill.fee
        self.total_slippage += abs(fill.slippage)
        self.total_realized_pnl += realized_pnl
        self.total_traded_notional += fill.notional
        
        # Update price
        self.update_price(fill.symbol, fill.fill_price)
        
        # Record trade
        self._trade_history.append({
            "fill_id": fill.fill_id,
            "order_id": fill.order_id,
            "symbol": fill.symbol,
            "side": fill.side.name,
            "price": fill.fill_price,
            "quantity": fill.fill_quantity,
            "notional": fill.notional,
            "fee": fill.fee,
            "slippage": fill.slippage,
            "timestamp_ns": fill.timestamp_ns,
            "position_before": old_qty,
            "position_after": position.quantity,
            "realized_pnl": realized_pnl,
            "leverage_mode": self.leverage_mode.value,
        })
        
        return realized_pnl
    
    def equity(self) -> float:
        """
        Calculate total portfolio equity.
        
        SPOT mode: equity = cash + sum of position values at current prices
        MARGIN mode: equity = collateral + total unrealized P&L
        """
        if self.is_margin_mode:
            # Margin mode: collateral + unrealized P&L
            return self.cash + self.total_unrealized_pnl()
        else:
            # Spot mode: cash + position values
            equity = self.cash
            for symbol, position in self._positions.items():
                if position.is_flat:
                    continue
                price = self._last_prices.get(symbol, position.avg_entry_price)
                position_value = position.quantity * price
                equity += position_value
            return equity
    
    def total_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L across all positions."""
        total = 0.0
        for symbol, position in self._positions.items():
            price = self._last_prices.get(symbol, position.avg_entry_price)
            total += position.unrealized_pnl(price)
        return total
    
    # Alias for backwards compatibility
    def unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L across all positions."""
        return self.total_unrealized_pnl()
    
    def total_exposure(self) -> float:
        """Calculate total absolute exposure across all positions."""
        total = 0.0
        for symbol, position in self._positions.items():
            if position.is_flat:
                continue
            price = self._last_prices.get(symbol, position.avg_entry_price)
            total += abs(position.quantity) * price
        return total
    
    def symbol_exposure(self, symbol: str) -> float:
        """Calculate absolute exposure for a single symbol."""
        pos = self._positions.get(symbol)
        if pos is None or pos.is_flat:
            return 0.0
        price = self._last_prices.get(symbol, pos.avg_entry_price)
        return abs(pos.quantity) * price
    
    def current_leverage(self) -> float:
        """
        Calculate current leverage ratio.
        
        Leverage = total_exposure / equity
        """
        eq = self.equity()
        if eq <= 0:
            return 0.0
        return self.total_exposure() / eq
    
    def update_drawdown(self) -> float:
        """
        Update peak equity and max drawdown.
        
        Returns:
            Current drawdown as fraction
        """
        current_equity = self.equity()
        
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, drawdown)
            return drawdown
        
        return 0.0
    
    def margin_used(self) -> float:
        """Calculate total margin used by positions (initial margin)."""
        return self.initial_margin_required()
    
    def initial_margin_required(self, symbol: Optional[str] = None) -> float:
        """
        Calculate initial margin required for positions.
        
        Initial margin = notional / leverage = notional * initial_margin_rate
        
        Args:
            symbol: Specific symbol, or None for total
        """
        if symbol is not None:
            pos = self._positions.get(symbol)
            if pos is None or pos.is_flat:
                return 0.0
            price = self._last_prices.get(symbol, pos.avg_entry_price)
            notional = abs(pos.quantity) * price
            return notional * self.margin_config.initial_margin_rate
        
        # Total across all positions
        total = 0.0
        for sym, pos in self._positions.items():
            if pos.is_flat:
                continue
            price = self._last_prices.get(sym, pos.avg_entry_price)
            notional = abs(pos.quantity) * price
            total += notional * self.margin_config.initial_margin_rate
        return total
    
    def maintenance_margin_required(self, symbol: Optional[str] = None) -> float:
        """
        Calculate maintenance margin required for positions.
        
        Maintenance margin = initial_margin * maintenance_ratio
        
        Args:
            symbol: Specific symbol, or None for total
        """
        return self.initial_margin_required(symbol) * self.margin_config.maintenance_margin_ratio
    
    def available_margin(self) -> float:
        """
        Calculate available margin for new positions.
        
        Available = equity - initial_margin_used
        """
        return max(0.0, self.equity() - self.initial_margin_required())
    
    def margin_for_order(
        self,
        symbol: str,
        order_qty: float,
        order_price: float,
        is_buy: bool
    ) -> float:
        """
        Calculate additional margin required for an order.
        
        Args:
            symbol: Trading symbol
            order_qty: Order quantity (unsigned)
            order_price: Order price
            is_buy: True for buy, False for sell
            
        Returns:
            Delta margin required (positive = need more, negative = releases margin)
        """
        current_pos = self.position_quantity(symbol)
        current_price = self._last_prices.get(symbol, order_price)
        
        # Calculate new position
        if is_buy:
            new_pos = current_pos + order_qty
        else:
            new_pos = current_pos - order_qty
        
        # Old margin for this symbol
        old_notional = abs(current_pos) * current_price
        old_margin = old_notional * self.margin_config.initial_margin_rate
        
        # New margin for this symbol
        new_notional = abs(new_pos) * order_price
        new_margin = new_notional * self.margin_config.initial_margin_rate
        
        return new_margin - old_margin
    
    def is_liquidation_triggered(self) -> bool:
        """
        Check if liquidation should be triggered.
        
        Liquidation when: equity < maintenance_margin_required
        
        Returns:
            True if positions should be liquidated
        """
        if not self.is_margin_mode:
            return False
        
        equity = self.equity()
        maintenance = self.maintenance_margin_required()
        
        # Add buffer to prevent oscillation
        threshold = maintenance * (1 + self.margin_config.liquidation_buffer)
        
        return equity < threshold
    
    def get_liquidation_symbols(self) -> List[str]:
        """
        Get list of symbols to liquidate.
        
        Returns all symbols with open positions when liquidation is triggered.
        """
        symbols = []
        for symbol, pos in self._positions.items():
            if not pos.is_flat:
                symbols.append(symbol)
        return symbols
    
    def record_liquidation(self, symbol: str, price: float, timestamp_ns: int) -> None:
        """Record a liquidation event."""
        pos = self._positions.get(symbol)
        if pos is None:
            return
        
        self._liquidations.append({
            "symbol": symbol,
            "quantity": pos.quantity,
            "price": price,
            "timestamp_ns": timestamp_ns,
            "equity_at_liquidation": self.equity(),
            "maintenance_margin": self.maintenance_margin_required(),
        })
    
    # =========================================================================
    # LEVERAGE STATISTICS (for AFML metrics)
    # =========================================================================
    
    def sample_leverage_stats(self) -> None:
        """
        Record current exposure and equity for leverage stats.
        
        Call this once per bar to build statistics for AFML leverage metric.
        """
        self._exposure_samples.append(self.total_exposure())
        self._equity_samples.append(self.equity())
    
    def get_leverage_stats(self) -> Dict[str, float]:
        """
        Calculate leverage statistics.
        
        Returns AFML-style leverage: avg_exposure / avg_equity
        """
        if not self._exposure_samples or not self._equity_samples:
            return {
                "avg_exposure": 0.0,
                "max_exposure": 0.0,
                "avg_equity": 0.0,
                "avg_leverage_afml": 0.0,
                "max_leverage": 0.0,
            }
        
        avg_exposure = sum(self._exposure_samples) / len(self._exposure_samples)
        max_exposure = max(self._exposure_samples)
        avg_equity = sum(self._equity_samples) / len(self._equity_samples)
        
        # AFML leverage = avg dollar position / avg AUM
        avg_leverage_afml = avg_exposure / avg_equity if avg_equity > 0 else 0.0
        
        # Max leverage seen
        max_leverage = 0.0
        for exp, eq in zip(self._exposure_samples, self._equity_samples):
            if eq > 0:
                max_leverage = max(max_leverage, exp / eq)
        
        return {
            "avg_exposure": avg_exposure,
            "max_exposure": max_exposure,
            "avg_equity": avg_equity,
            "avg_leverage_afml": avg_leverage_afml,
            "max_leverage": max_leverage,
        }
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of current portfolio state."""
        return {
            "cash": self.cash,
            "equity": self.equity(),
            "unrealized_pnl": self.unrealized_pnl(),
            "realized_pnl": self.total_realized_pnl,
            "total_fees": self.total_fees,
            "total_slippage": self.total_slippage,
            "max_drawdown": self.max_drawdown,
            "leverage_mode": self.leverage_mode.value,
            "total_exposure": self.total_exposure(),
            "current_leverage": self.current_leverage(),
            "initial_margin_required": self.initial_margin_required() if self.is_margin_mode else 0.0,
            "available_margin": self.available_margin(),
            "positions": {
                sym: {
                    "quantity": pos.quantity,
                    "avg_entry": pos.avg_entry_price,
                    "unrealized_pnl": pos.unrealized_pnl(
                        self._last_prices.get(sym, pos.avg_entry_price)
                    ),
                }
                for sym, pos in self._positions.items()
                if not pos.is_flat
            },
        }
    
    @property
    def trade_history(self) -> List[Dict[str, Any]]:
        """Get list of all trades."""
        return self._trade_history
    
    @property
    def positions(self) -> Dict[str, Position]:
        """Get all positions."""
        return self._positions
    
    @property
    def liquidations(self) -> List[Dict[str, Any]]:
        """Get list of liquidation events."""
        return self._liquidations
    
    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_cash
        self._positions.clear()
        self._last_prices.clear()
        self._trade_history.clear()
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.total_realized_pnl = 0.0
        self.peak_equity = self.initial_cash
        self.max_drawdown = 0.0
        self.total_traded_notional = 0.0
        self._liquidations.clear()
        self._exposure_samples.clear()
        self._equity_samples.clear()
