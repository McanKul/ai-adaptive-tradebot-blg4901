"""
tests/test_backtest_strategy_interface.py
=========================================
Tests verifying that strategies only receive bars, never ticks.

CRITICAL TEST: This verifies the core architectural constraint that
strategies operate ONLY on bars.
"""
import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any
from dataclasses import dataclass

from Interfaces.market_data import Tick, Bar
from Interfaces.orders import Order, OrderType, OrderSide
from Interfaces.strategy_adapter import IBacktestStrategy, StrategyContext

from Backtest.tick_store import TickStore, TickStoreConfig
from Backtest.disk_streamer import DiskTickStreamer, DiskStreamerConfig
from Backtest.bar_builder import TimeBarBuilder
from Backtest.engine import BacktestEngine, EngineConfig
from utils.bar_store import BarStore


class BarOnlyTestStrategy:
    """
    Test strategy that tracks what it receives.
    
    CRITICAL: This strategy should ONLY receive Bar objects via on_bar().
    It should NEVER receive Tick objects directly.
    """
    
    def __init__(self):
        self.bars_received: List[Bar] = []
        self.contexts_received: List[StrategyContext] = []
        self.ticks_received: List[Tick] = []  # Should always be empty!
        self.on_tick_called = False  # Should always be False!
    
    def on_bar(self, bar: Bar, ctx: StrategyContext) -> List[Order]:
        """
        Process a bar. This should be the ONLY entry point.
        """
        self.bars_received.append(bar)
        self.contexts_received.append(ctx)
        return []
    
    def on_tick(self, tick: Tick) -> None:
        """
        This method exists only to verify it's NEVER called.
        """
        self.on_tick_called = True
        self.ticks_received.append(tick)
    
    def reset(self) -> None:
        self.bars_received.clear()
        self.contexts_received.clear()
        self.ticks_received.clear()
        self.on_tick_called = False


class TestStrategyReceivesBarsOnly:
    """Tests that verify strategies only receive bars, never ticks."""
    
    def _create_tick_data(self, tmp_path, ticks_per_minute=10, num_minutes=3):
        """Create tick data that spans multiple bars."""
        tick_file = tmp_path / "BTCUSDT_ticks.csv"
        
        ns_per_minute = 60_000_000_000
        tick_interval = ns_per_minute // ticks_per_minute
        
        with open(tick_file, 'w') as f:
            f.write("timestamp_ns,symbol,price,volume\n")
            
            for minute in range(num_minutes):
                for tick_idx in range(ticks_per_minute):
                    ts = minute * ns_per_minute + tick_idx * tick_interval
                    price = 50000 + minute * 100 + tick_idx
                    f.write(f"{ts},BTCUSDT,{price},1.0\n")
        
        return str(tmp_path)
    
    def test_strategy_receives_only_bars_not_ticks(self, tmp_path):
        """CRITICAL TEST: Verify strategy receives bars, never ticks."""
        data_dir = self._create_tick_data(tmp_path, ticks_per_minute=10, num_minutes=3)
        
        config = EngineConfig(
            tick_data_dir=data_dir,
            symbols=["BTCUSDT"],
            timeframe="1m",
            initial_capital=10000.0,
            random_seed=42,
        )
        
        engine = BacktestEngine(config)
        strategy = BarOnlyTestStrategy()
        
        result = engine.run(strategy)
        
        # CRITICAL ASSERTIONS
        assert len(strategy.bars_received) > 0, "Strategy should receive bars"
        assert len(strategy.ticks_received) == 0, "Strategy should NOT receive ticks"
        assert strategy.on_tick_called == False, "Strategy.on_tick should NEVER be called"
        
        # Verify bars have correct structure
        for bar in strategy.bars_received:
            assert isinstance(bar, Bar)
            assert bar.symbol == "BTCUSDT"
            assert bar.timeframe == "1m"
            assert bar.open > 0
            assert bar.high >= bar.open
            assert bar.low <= bar.open
            assert bar.close > 0
            assert bar.volume > 0
    
    def test_context_provides_bar_store_not_ticks(self, tmp_path):
        """Verify context provides BarStore access, not tick access."""
        data_dir = self._create_tick_data(tmp_path, ticks_per_minute=5, num_minutes=2)
        
        config = EngineConfig(
            tick_data_dir=data_dir,
            symbols=["BTCUSDT"],
            timeframe="1m",
            initial_capital=10000.0,
            random_seed=42,
        )
        
        engine = BacktestEngine(config)
        strategy = BarOnlyTestStrategy()
        
        engine.run(strategy)
        
        # Verify contexts
        assert len(strategy.contexts_received) > 0
        
        for ctx in strategy.contexts_received:
            assert isinstance(ctx, StrategyContext)
            assert ctx.symbol == "BTCUSDT"
            assert ctx.bar_store is not None
            
            # Can get OHLCV data (bars)
            ohlcv = ctx.get_ohlcv()
            assert "open" in ohlcv
            assert "close" in ohlcv
            
            # No tick-level access in context
            assert not hasattr(ctx, 'ticks')
            assert not hasattr(ctx, 'get_ticks')


class TestStrategyInterfaceCompliance:
    """Tests for IBacktestStrategy interface compliance."""
    
    def test_on_bar_returns_orders(self):
        """Test that on_bar returns a list of orders."""
        
        class OrderGeneratingStrategy:
            def __init__(self):
                pass
            
            def on_bar(self, bar: Bar, ctx: StrategyContext) -> List[Order]:
                return [Order(
                    symbol=bar.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=1.0,
                )]
            
            def reset(self):
                pass
        
        strategy = OrderGeneratingStrategy()
        
        bar = Bar(
            symbol="BTCUSDT",
            timeframe="1m",
            timestamp_ns=1000000000000,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=10.0,
        )
        
        ctx = StrategyContext(
            symbol="BTCUSDT",
            timeframe="1m",
            bar_store=BarStore(),
        )
        
        orders = strategy.on_bar(bar, ctx)
        
        assert isinstance(orders, list)
        assert len(orders) == 1
        assert isinstance(orders[0], Order)
    
    def test_on_bar_can_return_empty_list(self):
        """Test that on_bar can return empty list (no signal)."""
        
        class NoSignalStrategy:
            def on_bar(self, bar: Bar, ctx: StrategyContext) -> List[Order]:
                return []
            
            def reset(self):
                pass
        
        strategy = NoSignalStrategy()
        
        bar = Bar("BTCUSDT", "1m", 1000, 100.0, 101.0, 99.0, 100.5, 10.0)
        ctx = StrategyContext("BTCUSDT", "1m", BarStore())
        
        orders = strategy.on_bar(bar, ctx)
        
        assert isinstance(orders, list)
        assert len(orders) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
