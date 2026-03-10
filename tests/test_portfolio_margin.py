"""
tests/test_portfolio_margin.py
==============================
Tests for margin/leverage mode in Portfolio.

These tests verify:
- Spot mode: cash pays full notional, no leverage
- Margin mode: collateral-based with leverage multiplier
- Equity calculation under both modes
- Liquidation warning/handling in margin mode
"""
import pytest
from decimal import Decimal
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from Backtest.portfolio import Portfolio, LeverageMode, MarginConfig
from Interfaces.orders import Order, OrderSide, OrderType, Fill


class TestSpotMode:
    """Test spot (cash) mode - no leverage."""
    
    def test_spot_requires_full_cash(self):
        """In spot mode, buying requires full notional in cash."""
        portfolio = Portfolio(
            initial_cash=10000.0,
            leverage_mode=LeverageMode.SPOT,
        )
        
        # Buy $10,000 worth of BTC (0.2 BTC at $50,000)
        fill = Fill(
            order_id="1",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            fill_price=50000.0,
            fill_quantity=0.2,  # 0.2 BTC = $10,000
            fee=0.0,
            timestamp_ns=1000000000
        )
        
        portfolio.apply_fill(fill)
        
        assert portfolio.cash == 0.0  # All cash used
        assert portfolio.position_quantity("BTCUSDT") == 0.2
    
    def test_spot_equity_is_cash_plus_positions(self):
        """Spot equity = cash + sum(position * price)."""
        portfolio = Portfolio(
            initial_cash=10000.0,
            leverage_mode=LeverageMode.SPOT,
        )
        
        # Buy 0.1 BTC at $50,000 = $5,000
        fill = Fill(
            order_id="1",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            fill_price=50000.0,
            fill_quantity=0.1,
            fee=0.0,
            timestamp_ns=1000000000
        )
        portfolio.apply_fill(fill)
        
        # Cash: $5,000, Position: 0.1 BTC
        # If price stays at $50,000, equity = $5,000 + 0.1*$50,000 = $10,000
        equity = portfolio.equity()
        
        assert abs(equity - 10000.0) < 0.01
        
        # If price goes to $60,000, equity = $5,000 + 0.1*$60,000 = $11,000
        portfolio.update_price("BTCUSDT", 60000.0)
        equity_up = portfolio.equity()
        
        assert abs(equity_up - 11000.0) < 0.01


class TestMarginMode:
    """Test margin (leverage) mode."""
    
    def test_margin_allows_leveraged_position(self):
        """In margin mode, can open position larger than cash (up to leverage)."""
        margin_config = MarginConfig(leverage=10.0)
        portfolio = Portfolio(
            initial_cash=10000.0,
            leverage_mode=LeverageMode.MARGIN,
            margin_config=margin_config,
        )
        
        # Buy 2 BTC at $50,000 = $100,000 notional
        # With $10,000 collateral and 10x leverage, max is $100,000
        fill = Fill(
            order_id="1",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            fill_price=50000.0,
            fill_quantity=2.0,  # 2 BTC = $100,000 notional
            fee=0.0,
            timestamp_ns=1000000000
        )
        
        portfolio.apply_fill(fill)
        
        assert portfolio.position_quantity("BTCUSDT") == 2.0
    
    def test_margin_equity_with_unrealized_pnl(self):
        """Margin equity = collateral + unrealized PnL."""
        margin_config = MarginConfig(leverage=10.0)
        portfolio = Portfolio(
            initial_cash=10000.0,
            leverage_mode=LeverageMode.MARGIN,
            margin_config=margin_config,
        )
        
        # Buy 1 BTC at $50,000
        entry_price = 50000.0
        fill = Fill(
            order_id="1",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            fill_price=entry_price,
            fill_quantity=1.0,
            fee=0.0,
            timestamp_ns=1000000000
        )
        portfolio.apply_fill(fill)
        
        # Price goes up 10% to $55,000
        # Unrealized PnL = 1.0 * ($55,000 - $50,000) = $5,000
        # Equity = $10,000 + $5,000 = $15,000
        portfolio.update_price("BTCUSDT", 55000.0)
        equity_up = portfolio.equity()
        
        assert abs(equity_up - 15000.0) < 0.01
        
        # Price goes down 10% to $45,000
        # Unrealized PnL = 1.0 * ($45,000 - $50,000) = -$5,000
        # Equity = $10,000 - $5,000 = $5,000
        portfolio.update_price("BTCUSDT", 45000.0)
        equity_down = portfolio.equity()
        
        assert abs(equity_down - 5000.0) < 0.01
    
    def test_leverage_amplifies_returns(self):
        """Leverage should amplify returns (both gains and losses)."""
        # Create two portfolios: spot and 5x margin
        spot = Portfolio(
            initial_cash=10000.0, 
            leverage_mode=LeverageMode.SPOT,
        )
        
        margin_config = MarginConfig(leverage=5.0)
        margin = Portfolio(
            initial_cash=10000.0, 
            leverage_mode=LeverageMode.MARGIN,
            margin_config=margin_config,
        )
        
        # Both buy as much as they can
        # Spot: $10,000 / $50,000 = 0.2 BTC
        # Margin: $50,000 / $50,000 = 1.0 BTC (5x leverage on $10,000)
        
        spot_fill = Fill(
            order_id="1", symbol="BTCUSDT", side=OrderSide.BUY,
            fill_price=50000.0, fill_quantity=0.2, fee=0.0, timestamp_ns=1000000000
        )
        spot.apply_fill(spot_fill)
        
        margin_fill = Fill(
            order_id="2", symbol="BTCUSDT", side=OrderSide.BUY,
            fill_price=50000.0, fill_quantity=1.0, fee=0.0, timestamp_ns=1000000000
        )
        margin.apply_fill(margin_fill)
        
        # Price goes up 20% to $60,000
        spot.update_price("BTCUSDT", 60000.0)
        margin.update_price("BTCUSDT", 60000.0)
        
        spot_equity = spot.equity()
        margin_equity = margin.equity()
        
        spot_return = (spot_equity - 10000.0) / 10000.0
        margin_return = (margin_equity - 10000.0) / 10000.0
        
        # Spot: 0.2 BTC * $10,000 gain = $2,000 = 20% return
        # Margin: 1.0 BTC * $10,000 gain = $10,000 = 100% return (5x)
        
        # Margin return should be roughly 5x spot return
        assert margin_return > spot_return
        assert abs(margin_return / spot_return - 5.0) < 0.1


class TestCurrentLeverage:
    """Test current leverage calculation."""
    
    def test_leverage_calculation(self):
        """Current leverage = total_exposure / equity."""
        margin_config = MarginConfig(leverage=10.0)
        portfolio = Portfolio(
            initial_cash=10000.0,
            leverage_mode=LeverageMode.MARGIN,
            margin_config=margin_config,
        )
        
        # Buy 1 BTC at $50,000 = $50,000 exposure
        fill = Fill(
            order_id="1", symbol="BTCUSDT", side=OrderSide.BUY,
            fill_price=50000.0, fill_quantity=1.0, fee=0.0, timestamp_ns=1000000000
        )
        portfolio.apply_fill(fill)
        
        # Exposure = $50,000, Equity = $10,000
        # Leverage = 5x
        current_lev = portfolio.current_leverage()
        assert abs(current_lev - 5.0) < 0.01
    
    def test_no_position_no_leverage(self):
        """With no positions, leverage should be 0."""
        portfolio = Portfolio(
            initial_cash=10000.0,
            leverage_mode=LeverageMode.MARGIN,
        )
        
        assert portfolio.current_leverage() == 0.0


class TestReduceOnlyOrders:
    """Test position closing."""
    
    def test_close_position_realizes_pnl(self):
        """Closing position should realize PnL."""
        margin_config = MarginConfig(leverage=10.0)
        portfolio = Portfolio(
            initial_cash=10000.0,
            leverage_mode=LeverageMode.MARGIN,
            margin_config=margin_config,
        )
        
        # Open long position
        open_fill = Fill(
            order_id="1", symbol="BTCUSDT", side=OrderSide.BUY,
            fill_price=50000.0, fill_quantity=1.0, fee=0.0, timestamp_ns=1000000000
        )
        portfolio.apply_fill(open_fill)
        
        assert portfolio.position_quantity("BTCUSDT") == 1.0
        
        # Close with profit
        close_fill = Fill(
            order_id="2", symbol="BTCUSDT", side=OrderSide.SELL,
            fill_price=55000.0, fill_quantity=1.0, fee=0.0, timestamp_ns=2000000000
        )
        pnl = portfolio.apply_fill(close_fill)
        
        assert portfolio.position_quantity("BTCUSDT") == 0.0
        assert abs(pnl - 5000.0) < 0.01  # $5,000 profit
        
        # Collateral should include realized PnL
        assert abs(portfolio.cash - 15000.0) < 0.01


class TestPortfolioInvariants:
    """Test portfolio maintains accounting invariants."""
    
    def test_spot_no_negative_cash_after_fill(self):
        """Spot mode cash after fill should be >= 0 if fill within means."""
        portfolio = Portfolio(
            initial_cash=10000.0,
            leverage_mode=LeverageMode.SPOT,
        )
        
        # Buy within means
        fill = Fill(
            order_id="1", symbol="BTCUSDT", side=OrderSide.BUY,
            fill_price=50000.0, fill_quantity=0.1,  # $5,000
            fee=0.0, timestamp_ns=1000000000
        )
        portfolio.apply_fill(fill)
        
        assert portfolio.cash >= 0
        assert abs(portfolio.cash - 5000.0) < 0.01
    
    def test_pnl_realized_on_close(self):
        """Closing a position should realize PnL."""
        portfolio = Portfolio(
            initial_cash=10000.0,
            leverage_mode=LeverageMode.SPOT,
        )
        
        # Buy 0.1 BTC at $50,000 = $5,000
        buy_fill = Fill(
            order_id="1", symbol="BTCUSDT", side=OrderSide.BUY,
            fill_price=50000.0, fill_quantity=0.1, fee=0.0, timestamp_ns=1000000000
        )
        portfolio.apply_fill(buy_fill)
        
        initial_cash = portfolio.cash
        assert abs(initial_cash - 5000.0) < 0.01
        
        # Sell 0.1 BTC at $60,000 = $6,000 (profit of $1,000)
        sell_fill = Fill(
            order_id="2", symbol="BTCUSDT", side=OrderSide.SELL,
            fill_price=60000.0, fill_quantity=0.1, fee=0.0, timestamp_ns=2000000000
        )
        portfolio.apply_fill(sell_fill)
        
        # Cash should be initial + proceeds = $5,000 + $6,000 = $11,000
        assert abs(portfolio.cash - 11000.0) < 0.01
        assert portfolio.position_quantity("BTCUSDT") == 0.0
    
    def test_fees_reduce_equity(self):
        """Fees should reduce equity."""
        portfolio = Portfolio(
            initial_cash=10000.0,
            leverage_mode=LeverageMode.SPOT,
        )
        
        fill = Fill(
            order_id="1", symbol="BTCUSDT", side=OrderSide.BUY,
            fill_price=50000.0, fill_quantity=0.1,
            fee=10.0,  # $10 fee
            timestamp_ns=1000000000
        )
        portfolio.apply_fill(fill)
        
        # Initial: $10,000
        # Spent: $5,000 + $10 fee = $5,010
        # Cash remaining: $4,990
        # Position value: $5,000
        # Total equity: $9,990
        
        equity = portfolio.equity()
        assert abs(equity - 9990.0) < 0.01
        assert portfolio.total_fees == 10.0


class TestDrawdownTracking:
    """Test max drawdown calculation."""
    
    def test_drawdown_updated(self):
        """Drawdown should be tracked as equity declines."""
        portfolio = Portfolio(
            initial_cash=10000.0,
            leverage_mode=LeverageMode.MARGIN,
        )
        
        # Open position
        fill = Fill(
            order_id="1", symbol="BTCUSDT", side=OrderSide.BUY,
            fill_price=50000.0, fill_quantity=1.0, fee=0.0, timestamp_ns=1000000000
        )
        portfolio.apply_fill(fill)
        
        # Price goes up - new peak
        portfolio.update_price("BTCUSDT", 55000.0)
        portfolio.update_drawdown()
        
        assert portfolio.peak_equity == 15000.0  # $10k + $5k unrealized
        
        # Price drops - drawdown
        portfolio.update_price("BTCUSDT", 50000.0)
        dd = portfolio.update_drawdown()
        
        # Equity = $10,000, Peak = $15,000
        # Drawdown = ($15,000 - $10,000) / $15,000 = 33.3%
        expected_dd = 5000.0 / 15000.0
        assert abs(portfolio.max_drawdown - expected_dd) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
