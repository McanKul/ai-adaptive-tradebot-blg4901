"""
tests/test_execution_realism.py
==============================
Tests for execution realism features:
- Partial fills with DOLLAR-VOLUME-based liquidity proxy
- Deterministic latency simulation
- Close positions at end
"""
import pytest
from dataclasses import dataclass
from random import Random

from Backtest.execution_models import (
    SimpleExecutionModel,
    PartialFillConfig,
    LatencyConfig,
    ExecutionStats,
)
from Interfaces.orders import Order, Fill, OrderType, OrderSide
from Interfaces.market_data import Bar


class MockCostModel:
    """Mock cost model for testing."""
    def calculate_slippage(self, price, qty, side, bar, rng):
        return 0.0
    
    def calculate_spread_cost(self, price, side):
        return 0.0
    
    def calculate_fee(self, notional, side, is_maker=False):
        return 0.0


def make_bar(symbol="TEST", timestamp_ns=1_000_000_000, close=100.0, volume=1000.0):
    """Helper to create test bars."""
    return Bar(
        symbol=symbol,
        timeframe="1m",
        timestamp_ns=timestamp_ns,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=volume,
    )


class TestPartialFillsDollarVolume:
    """Tests for partial fill execution using DOLLAR-VOLUME formula."""
    
    def test_no_partial_fills_by_default(self):
        """By default, orders fill completely."""
        model = SimpleExecutionModel()
        rng = Random(42)
        cost_model = MockCostModel()
        
        bar = make_bar()
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET,
        )
        
        fills = model.process_orders([order], bar, None, cost_model, rng)
        
        assert len(fills) == 1
        assert fills[0].fill_quantity == 100.0
        assert fills[0].fill_price == 100.0
    
    def test_partial_fill_dollar_volume_formula(self):
        """
        Partial fill uses dollar-volume formula:
        fill_ratio = min(1.0, (bar.volume * bar.close) / (order.qty * bar.close * liquidity_scale))
        """
        config = PartialFillConfig(
            enable_partial_fills=True,
            liquidity_scale=10.0,
            min_fill_ratio=0.0,
        )
        model = SimpleExecutionModel(partial_fill_config=config)
        rng = Random(42)
        cost_model = MockCostModel()
        
        # bar: close=100, volume=500 => dollar_volume = 50,000
        # order: qty=100, close=100 => notional = 10,000
        # fill_ratio = 50,000 / (10,000 * 10) = 0.5
        bar = make_bar(close=100.0, volume=500.0)
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET,
        )
        
        fills = model.process_orders([order], bar, None, cost_model, rng)
        
        assert len(fills) == 1
        assert fills[0].fill_quantity == 50.0  # 50% fill
    
    def test_full_fill_with_high_liquidity(self):
        """High liquidity bars result in full fills."""
        config = PartialFillConfig(
            enable_partial_fills=True,
            liquidity_scale=10.0,
            min_fill_ratio=0.0,
        )
        model = SimpleExecutionModel(partial_fill_config=config)
        rng = Random(42)
        cost_model = MockCostModel()
        
        # bar: close=100, volume=100,000 => dollar_volume = 10,000,000
        # order: qty=100, close=100 => notional = 10,000
        # fill_ratio = 10,000,000 / (10,000 * 10) = 100 => capped at 1.0
        bar = make_bar(close=100.0, volume=100_000.0)
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET,
        )
        
        fills = model.process_orders([order], bar, None, cost_model, rng)
        
        assert len(fills) == 1
        assert fills[0].fill_quantity == 100.0  # Full fill
    
    def test_partial_fill_min_ratio_rejection(self):
        """Orders below min_fill_ratio don't fill."""
        config = PartialFillConfig(
            enable_partial_fills=True,
            liquidity_scale=100.0,  # Very strict
            min_fill_ratio=0.25,
        )
        model = SimpleExecutionModel(partial_fill_config=config)
        rng = Random(42)
        cost_model = MockCostModel()
        
        # fill_ratio = 100,000 / (10,000 * 100) = 0.1 < 0.25 => rejected
        bar = make_bar(close=100.0, volume=100.0)
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET,
        )
        
        fills = model.process_orders([order], bar, None, cost_model, rng)
        
        assert len(fills) == 0  # Rejected
    
    def test_partial_fill_determinism(self):
        """Same inputs produce same fill ratio (no randomness)."""
        config = PartialFillConfig(
            enable_partial_fills=True,
            liquidity_scale=10.0,
            min_fill_ratio=0.0,
        )
        
        results = []
        for _ in range(5):
            model = SimpleExecutionModel(partial_fill_config=config)
            rng = Random(42)  # Different RNG but should not affect fill ratio
            cost_model = MockCostModel()
            
            bar = make_bar(close=100.0, volume=500.0)
            order = Order(
                symbol="TEST",
                side=OrderSide.BUY,
                quantity=100.0,
                order_type=OrderType.MARKET,
            )
            
            fills = model.process_orders([order], bar, None, cost_model, rng)
            results.append(fills[0].fill_quantity)
        
        # All fills should be identical
        assert all(r == results[0] for r in results)
        assert results[0] == 50.0  # Expected fill
    
    def test_partial_fill_limit_order(self):
        """Limit orders also support partial fills."""
        config = PartialFillConfig(
            enable_partial_fills=True,
            liquidity_scale=10.0,
            min_fill_ratio=0.0,
        )
        model = SimpleExecutionModel(partial_fill_config=config)
        rng = Random(42)
        cost_model = MockCostModel()
        
        # Limit buy at 99 should fill since bar low is 99
        bar = make_bar(close=100.0, volume=500.0)  # low=99
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.LIMIT,
            price=99.0,
        )
        
        fills = model.process_orders([order], bar, None, cost_model, rng)
        
        # Should partially fill based on liquidity
        assert len(fills) == 1
        assert fills[0].fill_quantity < 100.0  # Partial
    
    def test_partial_fill_stats_tracked(self):
        """Execution stats properly track partial fills."""
        config = PartialFillConfig(enable_partial_fills=True, liquidity_scale=10.0)
        model = SimpleExecutionModel(partial_fill_config=config)
        rng = Random(42)
        cost_model = MockCostModel()
        
        bar = make_bar(close=100.0, volume=500.0)
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET,
        )
        
        fills = model.process_orders([order], bar, None, cost_model, rng)
        
        stats = model._stats
        assert stats.total_fills == 1
        assert stats.partial_fills == 1


class TestLatencySimulation:
    """Tests for deterministic latency simulation."""
    
    def test_no_latency_by_default(self):
        """By default, no latency is applied."""
        model = SimpleExecutionModel()
        rng = Random(42)
        cost_model = MockCostModel()
        
        bar = make_bar()
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET,
        )
        
        fills = model.process_orders([order], bar, None, cost_model, rng)
        
        assert fills[0].timestamp_ns == bar.timestamp_ns
    
    def test_fixed_latency(self):
        """Fixed latency adds constant delay."""
        config = LatencyConfig(
            enable_latency=True,
            base_latency_ns=100_000,  # 100 microseconds
            max_jitter_ns=0,
        )
        model = SimpleExecutionModel(latency_config=config)
        rng = Random(42)
        cost_model = MockCostModel()
        
        bar = make_bar()
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET,
        )
        
        fills = model.process_orders([order], bar, None, cost_model, rng)
        
        assert fills[0].timestamp_ns == bar.timestamp_ns + 100_000
    
    def test_latency_determinism_same_seed(self):
        """Same seed produces same latency sequence."""
        config = LatencyConfig(
            enable_latency=True,
            base_latency_ns=100_000,
            max_jitter_ns=50_000,
        )
        
        latencies1 = []
        latencies2 = []
        
        for run in range(2):
            rng = Random(42)  # Reset seed each run
            model = SimpleExecutionModel(latency_config=config)
            cost_model = MockCostModel()
            latencies = []
            
            for i in range(10):
                bar = make_bar(timestamp_ns=1_000_000_000 + i * 1_000_000)
                order = Order(
                    symbol="TEST",
                    side=OrderSide.BUY,
                    quantity=1.0,
                    order_type=OrderType.MARKET,
                )
                
                fills = model.process_orders([order], bar, None, cost_model, rng)
                latency = fills[0].timestamp_ns - bar.timestamp_ns
                latencies.append(latency)
            
            if run == 0:
                latencies1 = latencies
            else:
                latencies2 = latencies
        
        assert latencies1 == latencies2
    
    def test_latency_stats_tracked(self):
        """Execution stats properly track latency."""
        config = LatencyConfig(
            enable_latency=True,
            base_latency_ns=100_000,
            max_jitter_ns=0,
        )
        model = SimpleExecutionModel(latency_config=config)
        rng = Random(42)
        cost_model = MockCostModel()
        
        for i in range(5):
            bar = make_bar(timestamp_ns=1_000_000_000 + i * 1_000_000)
            order = Order(
                symbol="TEST",
                side=OrderSide.BUY,
                quantity=1.0,
                order_type=OrderType.MARKET,
            )
            model.process_orders([order], bar, None, cost_model, rng)
        
        stats = model._stats
        assert stats.total_fills == 5
        assert stats.avg_latency_ns == 100_000  # Fixed latency, no jitter


class TestExecutionStats:
    """Tests for execution statistics."""
    
    def test_stats_to_dict(self):
        """Stats can be converted to dict for serialization."""
        stats = ExecutionStats(
            total_fills=10,
            partial_fills=2,
            rejected_orders=1,
            total_latency_ns=500_000,
            max_latency_ns=100_000,
            latency_samples=5,
        )
        
        d = stats.to_dict()
        
        assert d["total_fills"] == 10
        assert d["partial_fills"] == 2
        assert d["avg_latency_ns"] == 100_000


class TestClosePositionsAtEnd:
    """Tests for close_positions_at_end functionality."""
    
    def test_engine_config_has_close_positions_flag(self):
        """EngineConfig includes close_positions_at_end."""
        from Backtest.engine import EngineConfig
        
        config = EngineConfig(
            tick_data_dir="data/ticks",
            symbols=["TEST"],
            close_positions_at_end=True,
        )
        assert config.close_positions_at_end is True
        
        config2 = EngineConfig(
            tick_data_dir="data/ticks",
            symbols=["TEST"],
            close_positions_at_end=False,
        )
        assert config2.close_positions_at_end is False
    
    def test_runner_config_has_close_positions_flag(self):
        """BacktestConfig includes close_positions_at_end."""
        from Backtest.runner import BacktestConfig, DataConfig
        
        data = DataConfig(
            tick_data_dir="data/ticks",
            symbols=["TEST"],
        )
        config = BacktestConfig(
            data=data,
            close_positions_at_end=True,
        )
        
        engine_config = config.to_engine_config()
        assert engine_config.close_positions_at_end is True
