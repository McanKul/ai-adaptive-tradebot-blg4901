"""
Tests for leverage mode, margin accounting, exit manager, and support/resistance.
"""
import pytest
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

# Import modules under test
from Backtest.portfolio import Portfolio, LeverageMode, MarginConfig
from Backtest.risk import BasicRiskManager as RiskManager, RiskLimits
from Backtest.exit_manager import ExitManager, ExitConfig, ExitReason, PositionState
from Interfaces.orders import Fill, OrderSide, Order, OrderType
from Interfaces.market_data import Bar
from utils.levels import (
    compute_pivot_levels, 
    detect_swing_levels, 
    cluster_levels,
    SupportResistanceTracker,
    SupportResistanceResult,
)


def create_fill(symbol: str, side: str, qty: float, price: float, fee: float = 0.0, ts: int = 0) -> Fill:
    """Helper to create Fill objects for testing."""
    order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    return Fill(
        order_id="test-order",
        symbol=symbol,
        side=order_side,
        fill_price=price,
        fill_quantity=qty,
        fee=fee,
        slippage=0.0,
        timestamp_ns=ts,
    )


def create_bar(symbol: str, open_: float, high: float, low: float, close: float, ts: int = 0) -> Bar:
    """Helper to create Bar objects for testing."""
    return Bar(
        symbol=symbol,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1000.0,
        timestamp_ns=ts,
        timeframe="1m",  # Required field
    )


# -----------------------------------------------------------------------------
# Portfolio Margin Mode Tests
# -----------------------------------------------------------------------------
class TestPortfolioMarginMode:
    """Test margin mode accounting in Portfolio."""
    
    def test_spot_mode_subtracts_full_notional(self):
        """In spot mode, buying should subtract full notional from cash."""
        portfolio = Portfolio(
            initial_cash=10000.0,
            leverage_mode=LeverageMode.SPOT,
        )
        
        # Buy 10 units at $100 = $1000 notional
        fill = create_fill("TEST", "buy", 10.0, 100.0, fee=1.0, ts=1000)
        portfolio.apply_fill(fill)
        
        # Cash should be reduced by notional + fee
        assert portfolio.cash == pytest.approx(10000 - 1000 - 1, rel=1e-6)
        assert portfolio.total_exposure() == pytest.approx(1000.0, rel=1e-6)
    
    def test_margin_mode_only_deducts_fee(self):
        """In margin mode, buying should only deduct fee, not full notional."""
        margin_config = MarginConfig(leverage=10, maintenance_margin_ratio=0.5)
        portfolio = Portfolio(
            initial_cash=10000.0,
            leverage_mode=LeverageMode.MARGIN,
            margin_config=margin_config,
        )
        
        # Buy 10 units at $100 = $1000 notional
        fill = create_fill("TEST", "buy", 10.0, 100.0, fee=1.0, ts=1000)
        portfolio.apply_fill(fill)
        
        # Cash should only be reduced by fee (margin mode: cash = collateral)
        assert portfolio.cash == pytest.approx(10000 - 1, rel=1e-6)
        assert portfolio.total_exposure() == pytest.approx(1000.0, rel=1e-6)
    
    def test_margin_mode_pnl_on_close(self):
        """In margin mode, P&L should be realized on position close."""
        margin_config = MarginConfig(leverage=10)
        portfolio = Portfolio(
            initial_cash=10000.0,
            leverage_mode=LeverageMode.MARGIN,
            margin_config=margin_config,
        )
        
        # Open long: Buy 10 @ $100
        fill = create_fill("TEST", "buy", 10.0, 100.0, fee=1.0, ts=1000)
        portfolio.apply_fill(fill)
        cash_after_open = portfolio.cash
        
        # Close with profit: Sell 10 @ $110 (profit = 10 * 10 = $100)
        fill = create_fill("TEST", "sell", 10.0, 110.0, fee=1.0, ts=2000)
        portfolio.apply_fill(fill)
        
        # Cash should include realized P&L minus fees
        expected_cash = cash_after_open + 100.0 - 1.0  # +profit -fee
        assert portfolio.cash == pytest.approx(expected_cash, rel=1e-6)
        assert portfolio.total_exposure() == pytest.approx(0.0, rel=1e-6)
    
    def test_margin_mode_pnl_on_close_loss(self):
        """In margin mode, loss should be deducted on position close."""
        margin_config = MarginConfig(leverage=10)
        portfolio = Portfolio(
            initial_cash=10000.0,
            leverage_mode=LeverageMode.MARGIN,
            margin_config=margin_config,
        )
        
        # Open long: Buy 10 @ $100
        fill = create_fill("TEST", "buy", 10.0, 100.0, fee=1.0, ts=1000)
        portfolio.apply_fill(fill)
        cash_after_open = portfolio.cash
        
        # Close with loss: Sell 10 @ $90 (loss = 10 * -10 = -$100)
        fill = create_fill("TEST", "sell", 10.0, 90.0, fee=1.0, ts=2000)
        portfolio.apply_fill(fill)
        
        # Cash should have loss deducted
        expected_cash = cash_after_open - 100.0 - 1.0  # -loss -fee
        assert portfolio.cash == pytest.approx(expected_cash, rel=1e-6)
    
    def test_initial_margin_required(self):
        """Test initial margin calculation for open positions."""
        margin_config = MarginConfig(leverage=10)  # 10x leverage
        portfolio = Portfolio(
            initial_cash=10000.0,
            leverage_mode=LeverageMode.MARGIN,
            margin_config=margin_config,
        )
        
        # No position = 0 margin required
        assert portfolio.initial_margin_required() == pytest.approx(0.0, rel=1e-6)
        
        # Open $1000 position, at 10x leverage = $100 margin required
        fill = create_fill("TEST", "buy", 10.0, 100.0, fee=0.0, ts=1000)
        portfolio.apply_fill(fill)
        
        assert portfolio.initial_margin_required() == pytest.approx(100.0, rel=1e-6)
        assert portfolio.initial_margin_required("TEST") == pytest.approx(100.0, rel=1e-6)
    
    def test_maintenance_margin_required(self):
        """Test maintenance margin calculation."""
        margin_config = MarginConfig(leverage=10, maintenance_margin_ratio=0.5)
        portfolio = Portfolio(
            initial_cash=10000.0,
            leverage_mode=LeverageMode.MARGIN,
            margin_config=margin_config,
        )
        
        # Open position
        fill = create_fill("TEST", "buy", 10.0, 100.0, fee=0.0, ts=1000)
        portfolio.apply_fill(fill)
        
        # Maintenance = initial_margin * ratio = 100 * 0.5 = 50
        assert portfolio.maintenance_margin_required() == pytest.approx(50.0, rel=1e-6)
    
    def test_liquidation_triggered(self):
        """Test liquidation detection when equity < maintenance margin."""
        margin_config = MarginConfig(leverage=10, maintenance_margin_ratio=0.5, liquidation_buffer=0.0)
        portfolio = Portfolio(
            initial_cash=100.0,  # Very small capital
            leverage_mode=LeverageMode.MARGIN,
            margin_config=margin_config,
        )
        
        # Open large position: $1000 notional, requires $100 initial margin
        # With only $100 capital, we're at exactly 100% margin usage
        fill = create_fill("TEST", "buy", 10.0, 100.0, fee=0.0, ts=1000)
        portfolio.apply_fill(fill)
        
        # Equity = cash + unrealized_pnl = 100 + 0 = 100
        # Maintenance = 100 * 0.5 = 50
        # Not liquidated yet (100 > 50)
        assert not portfolio.is_liquidation_triggered()
        
        # Simulate price drop: update position's unrealized P&L
        # This is normally done through update_price
        portfolio.update_price("TEST", 50.0)
        
        # Now equity = cash + unrealized_pnl = 100 + 10*(50-100) = 100 - 500 = -400
        # -400 < 50, liquidation triggered
        assert portfolio.is_liquidation_triggered()
    
    def test_current_leverage_calculation(self):
        """Test current leverage = exposure / equity."""
        margin_config = MarginConfig(leverage=10)
        portfolio = Portfolio(
            initial_cash=1000.0,
            leverage_mode=LeverageMode.MARGIN,
            margin_config=margin_config,
        )
        
        # No position = 0 leverage
        assert portfolio.current_leverage() == pytest.approx(0.0, rel=1e-6)
        
        # Open $1000 position with $1000 equity = 1x leverage
        fill = create_fill("TEST", "buy", 10.0, 100.0, fee=0.0, ts=1000)
        portfolio.apply_fill(fill)
        assert portfolio.current_leverage() == pytest.approx(1.0, rel=1e-6)
        
        # Open another $1000 position = 2x leverage
        fill = create_fill("TEST2", "buy", 10.0, 100.0, fee=0.0, ts=2000)
        portfolio.apply_fill(fill)
        assert portfolio.current_leverage() == pytest.approx(2.0, rel=1e-6)


# -----------------------------------------------------------------------------
# Risk Manager Margin Tests
# -----------------------------------------------------------------------------
class TestRiskManagerMargin:
    """Test risk manager margin checks."""
    
    def test_margin_for_order_increase_position(self):
        """Test margin calculation when increasing position."""
        margin_config = MarginConfig(leverage=10)
        portfolio = Portfolio(
            initial_cash=100.0,
            leverage_mode=LeverageMode.MARGIN,
            margin_config=margin_config,
        )
        
        # New position of $1000 notional = $100 margin required
        delta_margin = portfolio.margin_for_order("TEST", 10.0, 100.0, is_buy=True)
        assert delta_margin == pytest.approx(100.0, rel=1e-6)
    
    def test_margin_for_order_close_position(self):
        """Test margin calculation when closing position."""
        margin_config = MarginConfig(leverage=10)
        portfolio = Portfolio(
            initial_cash=1000.0,
            leverage_mode=LeverageMode.MARGIN,
            margin_config=margin_config,
        )
        
        # Open position first
        fill = create_fill("TEST", "buy", 10.0, 100.0, fee=0.0, ts=1000)
        portfolio.apply_fill(fill)
        
        # Selling entire position should release margin
        delta_margin = portfolio.margin_for_order("TEST", 10.0, 100.0, is_buy=False)
        assert delta_margin == pytest.approx(-100.0, rel=1e-6)  # Releases $100 margin


# -----------------------------------------------------------------------------
# Exit Manager Tests
# -----------------------------------------------------------------------------
class TestExitManager:
    """Test exit manager TP/SL/trailing stop logic."""
    
    def test_take_profit_pct_long(self):
        """Test take profit by percentage for long position."""
        config = ExitConfig(take_profit_pct=0.02)  # 2% TP
        exit_mgr = ExitManager(config)
        
        # Register entry: Long at $100
        exit_mgr.register_entry("TEST", 10.0, 100.0, bar_index=0)
        
        # Create bar at $101 (1%) - no exit
        bar = create_bar("TEST", 100.5, 101.5, 100.0, 101.0, ts=1000)
        result = exit_mgr.check_exit(bar, position=10.0, avg_entry_price=100.0, bar_index=1)
        assert result is None
        
        # Create bar at $102.01 (>2%) - should trigger TP
        bar = create_bar("TEST", 101.0, 102.5, 101.0, 102.01, ts=2000)
        result = exit_mgr.check_exit(bar, position=10.0, avg_entry_price=100.0, bar_index=2)
        assert result is not None
        assert result[1] == ExitReason.TAKE_PROFIT_PCT
    
    def test_stop_loss_pct_long(self):
        """Test stop loss by percentage for long position."""
        config = ExitConfig(stop_loss_pct=0.01)  # 1% SL
        exit_mgr = ExitManager(config)
        
        exit_mgr.register_entry("TEST", 10.0, 100.0, bar_index=0)
        
        # Create bar at $99.5 (0.5% loss) - no exit
        bar = create_bar("TEST", 100.0, 100.0, 99.0, 99.5, ts=1000)
        result = exit_mgr.check_exit(bar, position=10.0, avg_entry_price=100.0, bar_index=1)
        assert result is None
        
        # Create bar at $98.99 (>1% loss) - should trigger SL
        bar = create_bar("TEST", 99.5, 99.5, 98.5, 98.99, ts=2000)
        result = exit_mgr.check_exit(bar, position=10.0, avg_entry_price=100.0, bar_index=2)
        assert result is not None
        assert result[1] == ExitReason.STOP_LOSS_PCT
    
    def test_take_profit_pct_short(self):
        """Test take profit by percentage for short position."""
        config = ExitConfig(take_profit_pct=0.02)  # 2% TP
        exit_mgr = ExitManager(config)
        
        # Register entry: Short at $100 (negative qty)
        exit_mgr.register_entry("TEST", -10.0, 100.0, bar_index=0)
        
        # Create bar at $97.99 (>2% profit for short) - should trigger TP
        bar = create_bar("TEST", 99.0, 99.0, 97.5, 97.99, ts=1000)
        result = exit_mgr.check_exit(bar, position=-10.0, avg_entry_price=100.0, bar_index=1)
        assert result is not None
        assert result[1] == ExitReason.TAKE_PROFIT_PCT
    
    def test_stop_loss_pct_short(self):
        """Test stop loss by percentage for short position."""
        config = ExitConfig(stop_loss_pct=0.01)  # 1% SL
        exit_mgr = ExitManager(config)
        
        exit_mgr.register_entry("TEST", -10.0, 100.0, bar_index=0)
        
        # Create bar at $101.01 (>1% loss for short) - should trigger SL
        bar = create_bar("TEST", 100.5, 101.5, 100.5, 101.01, ts=1000)
        result = exit_mgr.check_exit(bar, position=-10.0, avg_entry_price=100.0, bar_index=1)
        assert result is not None
        assert result[1] == ExitReason.STOP_LOSS_PCT
    
    def test_take_profit_usd(self):
        """Test take profit by USD amount."""
        config = ExitConfig(take_profit_usd=50.0)  # $50 TP
        exit_mgr = ExitManager(config)
        
        # Long 10 units at $100, need $5/unit profit = $105
        exit_mgr.register_entry("TEST", 10.0, 100.0, bar_index=0)
        
        # Create bar at $104 (P&L = $40) - no exit
        bar = create_bar("TEST", 102.0, 104.5, 102.0, 104.0, ts=1000)
        result = exit_mgr.check_exit(bar, position=10.0, avg_entry_price=100.0, bar_index=1)
        assert result is None
        
        # Create bar at $105.01 (P&L > $50) - should trigger TP
        bar = create_bar("TEST", 104.0, 106.0, 104.0, 105.01, ts=2000)
        result = exit_mgr.check_exit(bar, position=10.0, avg_entry_price=100.0, bar_index=2)
        assert result is not None
        assert result[1] == ExitReason.TAKE_PROFIT_USD
    
    def test_stop_loss_usd(self):
        """Test stop loss by USD amount."""
        config = ExitConfig(stop_loss_usd=30.0)  # $30 SL
        exit_mgr = ExitManager(config)
        
        # Long 10 units at $100
        exit_mgr.register_entry("TEST", 10.0, 100.0, bar_index=0)
        
        # Create bar at $97.01 (loss = $29.9) - no exit
        bar = create_bar("TEST", 98.0, 98.0, 96.5, 97.01, ts=1000)
        result = exit_mgr.check_exit(bar, position=10.0, avg_entry_price=100.0, bar_index=1)
        assert result is None
        
        # Create bar at $96.99 (loss > $30) - should trigger SL
        bar = create_bar("TEST", 97.0, 97.0, 96.5, 96.99, ts=2000)
        result = exit_mgr.check_exit(bar, position=10.0, avg_entry_price=100.0, bar_index=2)
        assert result is not None
        assert result[1] == ExitReason.STOP_LOSS_USD
    
    def test_trailing_stop(self):
        """Test trailing stop updates and triggers."""
        config = ExitConfig(trailing_stop_pct=0.02)  # 2% trailing
        exit_mgr = ExitManager(config)
        
        exit_mgr.register_entry("TEST", 10.0, 100.0, bar_index=0)
        
        # Price rises to $105 - no exit yet (no prior peak)
        bar = create_bar("TEST", 100.0, 105.5, 100.0, 105.0, ts=1000)
        result = exit_mgr.check_exit(bar, position=10.0, avg_entry_price=100.0, bar_index=1)
        assert result is None
        
        # Price rises to $110 - still no exit
        bar = create_bar("TEST", 105.0, 110.5, 105.0, 110.0, ts=2000)
        result = exit_mgr.check_exit(bar, position=10.0, avg_entry_price=100.0, bar_index=2)
        assert result is None
        
        # Price drops to $107.5 - check if trailing triggers
        # Peak was at $110, entry at $100, peak P&L = $100
        # At $107.5, P&L = $75
        # Drawdown from peak P&L = ($100 - $75) / $1000 (notional) = 2.5% > 2%
        bar = create_bar("TEST", 110.0, 110.0, 107.0, 107.5, ts=3000)
        result = exit_mgr.check_exit(bar, position=10.0, avg_entry_price=100.0, bar_index=3)
        assert result is not None
        assert result[1] == ExitReason.TRAILING_STOP
    
    def test_max_holding_bars(self):
        """Test max holding bars exit."""
        config = ExitConfig(max_holding_bars=5)
        exit_mgr = ExitManager(config)
        
        exit_mgr.register_entry("TEST", 10.0, 100.0, bar_index=0)
        
        # At bar 4 (held for 4 bars) - no exit
        bar = create_bar("TEST", 100.0, 101.0, 99.0, 100.0, ts=4000)
        result = exit_mgr.check_exit(bar, position=10.0, avg_entry_price=100.0, bar_index=4)
        assert result is None
        
        # At bar 5 (held for 5 bars) - should exit
        bar = create_bar("TEST", 100.0, 101.0, 99.0, 100.0, ts=5000)
        result = exit_mgr.check_exit(bar, position=10.0, avg_entry_price=100.0, bar_index=5)
        assert result is not None
        assert result[1] == ExitReason.MAX_HOLDING_BARS
    
    def test_exit_order_generation(self):
        """Test that exit generates proper order."""
        config = ExitConfig(take_profit_pct=0.02)
        exit_mgr = ExitManager(config)
        
        exit_mgr.register_entry("TEST", 10.0, 100.0, bar_index=0)
        
        # Trigger exit
        bar = create_bar("TEST", 101.0, 103.0, 101.0, 102.01, ts=1000)
        result = exit_mgr.check_exit(bar, position=10.0, avg_entry_price=100.0, bar_index=1)
        
        assert result is not None
        order, reason = result
        assert isinstance(order, Order)
        assert order.symbol == "TEST"
        assert order.side == OrderSide.SELL  # Closing long
        assert order.quantity == 10.0
    
    def test_reset_exit_manager(self):
        """Test resetting exit manager state."""
        config = ExitConfig(take_profit_pct=0.02)
        exit_mgr = ExitManager(config)
        
        exit_mgr.register_entry("TEST", 10.0, 100.0, bar_index=0)
        assert len(exit_mgr._states) == 1
        
        exit_mgr.reset()
        assert len(exit_mgr._states) == 0


# -----------------------------------------------------------------------------
# Support/Resistance Tests
# -----------------------------------------------------------------------------
class TestSupportResistance:
    """Test support/resistance level detection."""
    
    def test_pivot_levels_classic(self):
        """Test classic pivot point calculation."""
        result = compute_pivot_levels(
            high=110.0, low=90.0, close=100.0, method="classic"
        )
        
        # Should have 7 levels: S3, S2, S1, Pivot, R1, R2, R3
        assert len(result.levels) == 7
        
        # Classic pivot = (H + L + C) / 3 = 300 / 3 = 100
        # The pivot should be in the middle of the list (index 3)
        assert 100.0 in result.levels
    
    def test_pivot_levels_fibonacci(self):
        """Test Fibonacci pivot point calculation."""
        result = compute_pivot_levels(
            high=110.0, low=90.0, close=100.0, method="fibonacci"
        )
        
        # Should have 7 levels
        assert len(result.levels) == 7
        
        # Pivot is still (H+L+C)/3 = 100
        assert 100.0 in result.levels
    
    def test_detect_swing_levels(self):
        """Test swing high/low detection."""
        # Create price series with clear swing points
        high = np.array([100, 105, 110, 105, 100, 95, 90, 95, 100, 105, 110], dtype=float)
        low = np.array([95, 100, 105, 100, 95, 90, 85, 90, 95, 100, 105], dtype=float)
        close = np.array([98, 103, 108, 102, 97, 92, 87, 93, 98, 103, 108], dtype=float)
        
        result = detect_swing_levels(high, low, close, window=2, num_levels=3)
        
        # Should return levels
        assert isinstance(result, SupportResistanceResult)
        assert len(result.levels) > 0
    
    def test_cluster_levels(self):
        """Test level clustering."""
        levels = [100.0, 100.5, 101.0, 150.0, 150.2, 200.0]
        
        # Cluster with 2% tolerance (0.02)
        clustered = cluster_levels(levels, tolerance=0.02)
        
        # Should cluster similar levels together
        assert len(clustered) <= len(levels)
        assert len(clustered) >= 3  # At least 3 distinct clusters
    
    def test_support_resistance_tracker(self):
        """Test SupportResistanceTracker updates."""
        tracker = SupportResistanceTracker(window=10, swing_window=3, num_levels=3)
        
        # Add some bars using on_bar method
        for i in range(15):
            price = 100 + i + np.sin(i) * 5  # Add some oscillation
            tracker.on_bar(
                high=price + 2,
                low=price - 2,
                close=price,
            )
        
        # Get levels
        levels = tracker.get_current_levels()
        
        # Should return some result after enough bars
        assert levels is not None
        assert isinstance(levels, SupportResistanceResult)


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------
class TestLeverageIntegration:
    """Integration tests combining multiple components."""
    
    def test_margin_mode_with_exit_manager(self):
        """Test margin mode portfolio with exit manager."""
        margin_config = MarginConfig(leverage=10)
        portfolio = Portfolio(
            initial_cash=10000.0,
            leverage_mode=LeverageMode.MARGIN,
            margin_config=margin_config,
        )
        
        exit_config = ExitConfig(take_profit_pct=0.05, stop_loss_pct=0.02)
        exit_mgr = ExitManager(exit_config)
        
        # Open position
        fill = create_fill("TEST", "buy", 100.0, 100.0, fee=10.0, ts=1000)
        portfolio.apply_fill(fill)
        exit_mgr.register_entry("TEST", 100.0, 100.0, bar_index=0)
        
        # Check exit at $105 (5% profit) - should trigger TP
        bar = create_bar("TEST", 104.0, 106.0, 104.0, 105.0, ts=2000)
        result = exit_mgr.check_exit(bar, position=100.0, avg_entry_price=100.0, bar_index=1)
        assert result is not None
        assert result[1] == ExitReason.TAKE_PROFIT_PCT
        
        # Close position with profit
        fill = create_fill("TEST", "sell", 100.0, 105.0, fee=10.0, ts=2000)
        portfolio.apply_fill(fill)
        
        # Verify P&L realized
        # P&L = 100 * (105 - 100) = $500
        # Fees = $20 total
        # Final equity should be 10000 + 500 - 20 = 10480
        assert portfolio.equity() == pytest.approx(10480.0, rel=1e-6)
    
    def test_leverage_stats_sampling(self):
        """Test leverage stats sampling for AFML metrics."""
        margin_config = MarginConfig(leverage=10)
        portfolio = Portfolio(
            initial_cash=10000.0,
            leverage_mode=LeverageMode.MARGIN,
            margin_config=margin_config,
        )
        
        # Open position
        fill = create_fill("TEST", "buy", 50.0, 100.0, fee=0.0, ts=1000)
        portfolio.apply_fill(fill)
        
        # Sample stats
        portfolio.sample_leverage_stats()
        
        # Get stats
        stats = portfolio.get_leverage_stats()
        
        assert "avg_exposure" in stats
        assert "avg_equity" in stats
        assert "avg_leverage_afml" in stats
        assert stats["avg_exposure"] == pytest.approx(5000.0, rel=1e-6)
        assert stats["avg_equity"] == pytest.approx(10000.0, rel=1e-6)
        assert stats["avg_leverage_afml"] == pytest.approx(0.5, rel=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
