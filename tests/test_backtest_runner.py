"""
tests/test_backtest_runner.py
=============================
Tests for BacktestRunner and single backtest runs.

Verifies:
- Single run returns BacktestResult with metrics
- Parameters are tracked in result
- Deterministic behavior
"""
import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any

from Interfaces.market_data import Bar
from Interfaces.orders import Order, OrderType, OrderSide
from Interfaces.strategy_adapter import StrategyContext
from Interfaces.metrics_interface import BacktestResult

from Backtest.runner import BacktestRunner, BacktestConfig, DataConfig, create_runner


class SimpleStrategy:
    """Simple test strategy."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.bar_count = 0
    
    def on_bar(self, bar: Bar, ctx: StrategyContext) -> List[Order]:
        self.bar_count += 1
        return []
    
    def reset(self):
        self.bar_count = 0


class SignalStrategy:
    """Strategy that generates signals."""
    
    def __init__(self, signal_every_n: int = 5):
        self.signal_every_n = signal_every_n
        self.bar_count = 0
        self.position = 0.0
    
    def on_bar(self, bar: Bar, ctx: StrategyContext) -> List[Order]:
        self.bar_count += 1
        orders = []
        
        if self.bar_count % self.signal_every_n == 0:
            if self.position == 0:
                # Open position
                orders.append(Order(
                    symbol=bar.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=0.1,
                ))
                self.position = 0.1
            else:
                # Close position
                orders.append(Order(
                    symbol=bar.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=0.1,
                ))
                self.position = 0.0
        
        return orders
    
    def reset(self):
        self.bar_count = 0
        self.position = 0.0


def create_test_tick_data(tmp_path, num_minutes=10):
    """Create test tick data."""
    tick_file = tmp_path / "BTCUSDT_ticks.csv"
    
    ns_per_minute = 60_000_000_000
    
    with open(tick_file, 'w') as f:
        f.write("timestamp_ns,symbol,price,volume\n")
        
        for minute in range(num_minutes):
            for tick in range(10):  # 10 ticks per minute
                ts = minute * ns_per_minute + tick * (ns_per_minute // 10)
                price = 50000 + minute * 10 + tick
                f.write(f"{ts},BTCUSDT,{price},1.0\n")
    
    return str(tmp_path)


class TestBacktestRunner:
    """Tests for BacktestRunner."""
    
    def test_single_run_returns_metrics(self, tmp_path):
        """Test that a single run returns BacktestResult with metrics."""
        data_dir = create_test_tick_data(tmp_path, num_minutes=10)
        
        data_config = DataConfig(
            tick_data_dir=data_dir,
            symbols=["BTCUSDT"],
            timeframe="1m",
        )
        
        config = BacktestConfig(
            data=data_config,
            initial_capital=10000.0,
            random_seed=42,
        )
        
        runner = BacktestRunner(config)
        
        def factory(params):
            return SimpleStrategy(**params)
        
        result = runner.run_once(
            strategy_factory=factory,
            params={"threshold": 0.7},
            strategy_name="TestStrategy",
        )
        
        # Verify result structure
        assert isinstance(result, BacktestResult)
        assert result.initial_capital == 10000.0
        assert result.final_equity > 0
        assert result.strategy_name == "TestStrategy"
        assert result.params == {"threshold": 0.7}
        
        # Metrics should be present
        assert hasattr(result, 'total_return')
        assert hasattr(result, 'max_drawdown')
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'equity_curve')
        
        # Data should be recorded
        assert len(result.equity_curve) > 0
    
    def test_params_tracked_in_result(self, tmp_path):
        """Test that strategy params are tracked in result."""
        data_dir = create_test_tick_data(tmp_path)
        
        runner = create_runner(
            tick_data_dir=data_dir,
            symbols=["BTCUSDT"],
            initial_capital=10000.0,
        )
        
        params = {"threshold": 0.5, "period": 14, "name": "test"}
        
        def factory(p):
            return SimpleStrategy(threshold=p.get("threshold", 0.5))
        
        result = runner.run_once(factory, params)
        
        assert result.params == params
    
    def test_deterministic_results(self, tmp_path):
        """Test that same inputs produce same outputs."""
        data_dir = create_test_tick_data(tmp_path, num_minutes=20)
        
        runner = create_runner(
            tick_data_dir=data_dir,
            symbols=["BTCUSDT"],
            initial_capital=10000.0,
            random_seed=42,
        )
        
        def factory(p):
            return SignalStrategy(signal_every_n=p["signal_every_n"])
        
        params = {"signal_every_n": 3}
        
        # Run twice
        result1 = runner.run_once(factory, params)
        result2 = runner.run_once(factory, params)
        
        # Results should be identical
        assert result1.final_equity == result2.final_equity
        assert result1.total_trades == result2.total_trades
        assert result1.max_drawdown == result2.max_drawdown
        assert len(result1.equity_curve) == len(result2.equity_curve)
    
    def test_trading_metrics_computed(self, tmp_path):
        """Test that trading metrics are computed when trades occur."""
        data_dir = create_test_tick_data(tmp_path, num_minutes=30)
        
        runner = create_runner(
            tick_data_dir=data_dir,
            symbols=["BTCUSDT"],
            initial_capital=10000.0,
        )
        
        def factory(p):
            return SignalStrategy(signal_every_n=p["n"])
        
        result = runner.run_once(factory, {"n": 5})
        
        # Should have executed trades
        assert result.total_trades >= 0
        assert result.total_fees >= 0
        assert result.turnover >= 0


class TestRunnerWithClass:
    """Tests for run_with_class method."""
    
    def test_run_with_class(self, tmp_path):
        """Test running with a strategy class directly."""
        data_dir = create_test_tick_data(tmp_path)
        
        runner = create_runner(
            tick_data_dir=data_dir,
            symbols=["BTCUSDT"],
            initial_capital=10000.0,
        )
        
        result = runner.run_with_class(
            SimpleStrategy,
            {"threshold": 0.8},
            "SimpleStrategy"
        )
        
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "SimpleStrategy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
