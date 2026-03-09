"""
tests/test_backtest_accounting.py
=================================
Tests for backtest accounting invariants.

These tests verify the CRITICAL INVARIANTS:
1. No orders => no equity change, no fees
2. Rejected orders => no equity change, no fees  
3. Fill events update cash/equity correctly
4. Trade count matches fills (or round-trips)
5. Return calculation is correct
"""
import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
import numpy as np

from Interfaces.market_data import Bar, Tick
from Interfaces.orders import Order, OrderType, OrderSide
from Interfaces.strategy_adapter import StrategyContext

from Backtest.engine import BacktestEngine, EngineConfig
from Backtest.portfolio import Portfolio
from Backtest.metrics import MetricsSink
from Backtest.execution_models import SimpleExecutionModel
from Backtest.cost_models import create_cost_model
from utils.bar_store import BarStore


def create_test_tick_data(tmp_path, symbol="BTCUSDT", prices=None, num_ticks=100):
    """Create test tick data with controlled prices."""
    tick_file = tmp_path / f"{symbol}_ticks.csv"
    
    if prices is None:
        prices = [100.0] * num_ticks  # Flat price
    
    ns_per_tick = 60_000_000_000  # 1 minute
    
    with open(tick_file, 'w') as f:
        f.write("timestamp_ns,symbol,price,volume\n")
        for i, price in enumerate(prices):
            ts = 1000000000000 + i * ns_per_tick
            f.write(f"{ts},{symbol},{price},1.0\n")
    
    return str(tmp_path)


class NoOrderStrategy:
    """Strategy that generates NO orders - for testing baseline."""
    
    def __init__(self):
        self.bar_count = 0
    
    def on_bar(self, bar: Bar, ctx: StrategyContext) -> List[Order]:
        self.bar_count += 1
        return []  # No orders ever
    
    def reset(self):
        self.bar_count = 0


class SingleBuyStrategy:
    """Strategy that buys once on bar 5."""
    
    def __init__(self, quantity: float = 1.0):
        self.quantity = quantity
        self.bar_count = 0
        self.bought = False
    
    def on_bar(self, bar: Bar, ctx: StrategyContext) -> List[Order]:
        self.bar_count += 1
        orders = []
        
        if self.bar_count == 5 and not self.bought:
            orders.append(Order(
                symbol=bar.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=self.quantity,
            ))
            self.bought = True
        
        return orders
    
    def reset(self):
        self.bar_count = 0
        self.bought = False


class BuySellStrategy:
    """Strategy that buys on bar 5, sells on bar 10."""
    
    def __init__(self, quantity: float = 1.0):
        self.quantity = quantity
        self.bar_count = 0
        self.bought = False
        self.sold = False
    
    def on_bar(self, bar: Bar, ctx: StrategyContext) -> List[Order]:
        self.bar_count += 1
        orders = []
        
        if self.bar_count == 5 and not self.bought:
            orders.append(Order(
                symbol=bar.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=self.quantity,
            ))
            self.bought = True
        
        elif self.bar_count == 10 and self.bought and not self.sold:
            orders.append(Order(
                symbol=bar.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=self.quantity,
            ))
            self.sold = True
        
        return orders
    
    def reset(self):
        self.bar_count = 0
        self.bought = False
        self.sold = False


class TestNoOrdersInvariant:
    """Test 1: Strategy returns no orders => equity unchanged, fees=0, trades=0."""
    
    def test_no_orders_no_change(self, tmp_path):
        """With no orders, equity should stay at initial capital."""
        # Create flat price data
        data_dir = create_test_tick_data(tmp_path, prices=[100.0] * 100)
        
        config = EngineConfig(
            tick_data_dir=data_dir,
            symbols=["BTCUSDT"],
            timeframe="1m",
            initial_capital=10000.0,
            random_seed=42,
        )
        
        engine = BacktestEngine(config)
        strategy = NoOrderStrategy()
        
        result = engine.run(strategy)
        
        # INVARIANT 1: No orders => equity unchanged
        assert result.final_equity == result.initial_capital, \
            f"Equity changed without orders: {result.initial_capital} -> {result.final_equity}"
        
        # INVARIANT 2: No orders => fees = 0
        assert result.total_fees == 0.0, \
            f"Fees charged without orders: {result.total_fees}"
        
        # INVARIANT 3: No orders => trades = 0
        assert result.total_trades == 0, \
            f"Trades counted without orders: {result.total_trades}"
        
        # INVARIANT 4: Return should be 0%
        assert abs(result.total_return_pct) < 0.001, \
            f"Return != 0 without orders: {result.total_return_pct}%"
        
        # INVARIANT 5: Drawdown should be 0
        assert result.max_drawdown == 0.0, \
            f"Drawdown without equity change: {result.max_drawdown}"
        
        # Check metadata
        assert result.metadata["fill_count"] == 0


class TestFillsUpdateEquity:
    """Test 2: Fills update cash/equity correctly."""
    
    def test_buy_reduces_cash(self, tmp_path):
        """Buying should reduce cash by notional + fee."""
        data_dir = create_test_tick_data(tmp_path, prices=[100.0] * 20)
        
        config = EngineConfig(
            tick_data_dir=data_dir,
            symbols=["BTCUSDT"],
            timeframe="1m",
            initial_capital=10000.0,
            taker_fee_bps=10.0,  # 0.1% fee
            slippage_bps=0.0,
            spread_bps=0.0,
            random_seed=42,
        )
        
        engine = BacktestEngine(config)
        strategy = SingleBuyStrategy(quantity=1.0)
        
        result = engine.run(strategy)
        
        # Should have 1 fill
        assert result.metadata["fill_count"] == 1
        
        # Cash should be reduced by notional (~100) + fee (~0.1)
        # Final equity = cash + position_value
        # Since price stayed at 100, position_value = 1 * 100 = 100
        # Cash = 10000 - 100 - fee
        # Equity = cash + 100 = 10000 - fee
        expected_equity = 10000.0 - result.total_fees
        assert abs(result.final_equity - expected_equity) < 1.0, \
            f"Expected equity ~{expected_equity}, got {result.final_equity}"
    
    def test_round_trip_trade_counted(self, tmp_path):
        """Buy then sell should count as 1 round-trip trade."""
        data_dir = create_test_tick_data(tmp_path, prices=[100.0] * 20)
        
        config = EngineConfig(
            tick_data_dir=data_dir,
            symbols=["BTCUSDT"],
            timeframe="1m",
            initial_capital=10000.0,
            taker_fee_bps=10.0,
            slippage_bps=0.0,
            spread_bps=0.0,
            random_seed=42,
        )
        
        engine = BacktestEngine(config)
        strategy = BuySellStrategy(quantity=1.0)
        
        result = engine.run(strategy)
        
        # Should have 2 fills (buy + sell)
        assert result.metadata["fill_count"] == 2, \
            f"Expected 2 fills, got {result.metadata['fill_count']}"
        
        # Should have 1 round-trip trade
        assert result.total_trades == 1, \
            f"Expected 1 round-trip trade, got {result.total_trades}"
        
        # Position should be closed
        assert result.metadata["open_trades"] == 0


class TestReturnCalculation:
    """Test 3: Return calculation is correct."""
    
    def test_return_percentage_correct(self, tmp_path):
        """Return percentage should match equity change."""
        # Price goes from 100 to 110 (+10%)
        prices = [100.0] * 10 + [110.0] * 10
        data_dir = create_test_tick_data(tmp_path, prices=prices)
        
        config = EngineConfig(
            tick_data_dir=data_dir,
            symbols=["BTCUSDT"],
            timeframe="1m",
            initial_capital=10000.0,
            taker_fee_bps=0.0,  # No fees for clean calculation
            slippage_bps=0.0,
            spread_bps=0.0,
            random_seed=42,
        )
        
        engine = BacktestEngine(config)
        strategy = SingleBuyStrategy(quantity=10.0)  # Buy 10 units
        
        result = engine.run(strategy)
        
        # Manual calculation:
        # Buy at 100, position = 10 units
        # Price moves to 110
        # Position value = 10 * 110 = 1100
        # Cash after buy = 10000 - 1000 = 9000
        # Final equity = 9000 + 1100 = 10100
        # Return = (10100 / 10000 - 1) * 100 = 1%
        
        expected_equity = 10000.0 + 10.0 * (110.0 - 100.0)  # 10100
        expected_return_pct = ((expected_equity / 10000.0) - 1.0) * 100.0  # 1%
        
        assert abs(result.final_equity - expected_equity) < 1.0, \
            f"Expected equity {expected_equity}, got {result.final_equity}"
        
        assert abs(result.total_return_pct - expected_return_pct) < 0.1, \
            f"Expected return {expected_return_pct}%, got {result.total_return_pct}%"


class TestDeterminism:
    """Test 4: Same seed produces same results."""
    
    def test_deterministic_results(self, tmp_path):
        """Two runs with same seed should produce identical results."""
        # Use varying prices to make non-trivial
        prices = [100 + i * 0.1 for i in range(50)]
        data_dir = create_test_tick_data(tmp_path, prices=prices)
        
        config = EngineConfig(
            tick_data_dir=data_dir,
            symbols=["BTCUSDT"],
            timeframe="1m",
            initial_capital=10000.0,
            random_seed=42,
        )
        
        results = []
        for _ in range(3):
            engine = BacktestEngine(config)
            strategy = BuySellStrategy(quantity=1.0)
            result = engine.run(strategy)
            results.append(result)
        
        # All runs should be identical
        for i in range(1, len(results)):
            assert results[0].final_equity == results[i].final_equity, \
                f"Run 0 equity {results[0].final_equity} != Run {i} equity {results[i].final_equity}"
            assert results[0].total_fees == results[i].total_fees
            assert results[0].total_trades == results[i].total_trades


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
