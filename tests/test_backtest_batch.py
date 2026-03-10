"""
tests/test_backtest_batch.py
============================
Tests for batch backtesting and result ranking.

Verifies:
- Multiple parameter combinations run correctly
- Results are ranked by score
- Top-k selection works
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any

from Interfaces.market_data import Bar
from Interfaces.orders import Order, OrderType, OrderSide
from Interfaces.strategy_adapter import StrategyContext
from Interfaces.metrics_interface import BacktestResult

from Backtest.runner import BacktestConfig, DataConfig
from Backtest.scoring.search_space import ParameterGrid, SearchSpace
from Backtest.scoring.scorer import Scorer, create_scorer
from Backtest.scoring.batch import BatchBacktest, BatchResult
from Backtest.scoring.selector import Selector, select_top_k


class ParameterizedStrategy:
    """Strategy with tunable parameters for testing."""
    
    def __init__(self, period: int = 10, multiplier: float = 1.0):
        self.period = period
        self.multiplier = multiplier
        self.bar_count = 0
        self.position = 0.0
    
    def on_bar(self, bar: Bar, ctx: StrategyContext) -> List[Order]:
        self.bar_count += 1
        orders = []
        
        # Simple signal based on parameters
        if self.bar_count % self.period == 0:
            if self.position == 0:
                orders.append(Order(
                    symbol=bar.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=0.1 * self.multiplier,
                ))
                self.position = 0.1 * self.multiplier
            else:
                orders.append(Order(
                    symbol=bar.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=self.position,
                ))
                self.position = 0.0
        
        return orders
    
    def reset(self):
        self.bar_count = 0
        self.position = 0.0


def create_test_data(tmp_path, num_minutes=30):
    """Create test tick data."""
    tick_file = tmp_path / "BTCUSDT_ticks.csv"
    ns_per_minute = 60_000_000_000
    
    with open(tick_file, 'w') as f:
        f.write("timestamp_ns,symbol,price,volume\n")
        for minute in range(num_minutes):
            for tick in range(10):
                ts = minute * ns_per_minute + tick * (ns_per_minute // 10)
                price = 50000 + minute * 5 + tick
                f.write(f"{ts},BTCUSDT,{price},1.0\n")
    
    return str(tmp_path)


class TestParameterGrid:
    """Tests for ParameterGrid."""
    
    def test_generates_all_combinations(self):
        """Test that all combinations are generated."""
        grid = ParameterGrid({
            "a": [1, 2],
            "b": ["x", "y", "z"],
        })
        
        combos = grid.get_combinations()
        
        assert len(combos) == 6  # 2 * 3
        assert {"a": 1, "b": "x"} in combos
        assert {"a": 2, "b": "z"} in combos
    
    def test_iteration(self):
        """Test iteration over grid."""
        grid = ParameterGrid({"x": [1, 2], "y": [3, 4]})
        
        count = 0
        for params in grid:
            assert "x" in params
            assert "y" in params
            count += 1
        
        assert count == 4
    
    def test_from_ranges(self):
        """Test creating grid from ranges."""
        grid = ParameterGrid.from_ranges(
            period=(10, 20, 5),  # [10, 15, 20]
        )
        
        combos = grid.get_combinations()
        assert len(combos) == 3


class TestSearchSpace:
    """Tests for SearchSpace with constraints."""
    
    def test_with_constraint(self):
        """Test that constraints filter invalid combinations."""
        space = SearchSpace()
        space.add("min_val", [10, 20, 30])
        space.add("max_val", [20, 30, 40])
        space.add_constraint(lambda p: p["min_val"] < p["max_val"])
        
        combos = space.get_combinations()
        
        # Verify all combos satisfy constraint
        for combo in combos:
            assert combo["min_val"] < combo["max_val"]
        
        # Should filter out invalid ones
        assert len(combos) < 9  # 9 total without constraint
    
    def test_chaining(self):
        """Test method chaining."""
        space = (SearchSpace()
            .add("a", [1, 2])
            .add("b", [3, 4])
            .add_constraint(lambda p: p["a"] < p["b"]))
        
        assert len(space) == 4  # All pass constraint
    
    def test_info(self):
        """Test info method."""
        space = SearchSpace()
        space.add("x", [1, 2, 3])
        space.add("y", [4, 5])
        
        info = space.info()
        assert info["total_unconstrained"] == 6


class TestBatchBacktest:
    """Tests for BatchBacktest."""
    
    def test_batch_runs_all_params(self, tmp_path):
        """Test that batch runs all parameter combinations."""
        data_dir = create_test_data(tmp_path, num_minutes=20)
        
        data_config = DataConfig(
            tick_data_dir=data_dir,
            symbols=["BTCUSDT"],
            timeframe="1m",
        )
        
        config = BacktestConfig(
            data=data_config,
            initial_capital=10000.0,
        )
        
        def factory(p):
            return ParameterizedStrategy(**p)
        
        batch = BatchBacktest(config, factory)
        
        grid = ParameterGrid({
            "period": [3, 5],
            "multiplier": [1.0, 2.0],
        })
        
        result = batch.run(grid)
        
        assert isinstance(result, BatchResult)
        assert len(result.results) == 4  # 2 * 2 combinations
        assert len(result.scores) == 4
        assert len(result.rankings) == 4
    
    def test_results_are_ranked(self, tmp_path):
        """Test that results are ranked by score."""
        data_dir = create_test_data(tmp_path, num_minutes=30)
        
        data_config = DataConfig(
            tick_data_dir=data_dir,
            symbols=["BTCUSDT"],
            timeframe="1m",
        )
        
        config = BacktestConfig(
            data=data_config,
            initial_capital=10000.0,
        )
        
        def factory(p):
            return ParameterizedStrategy(**p)
        
        batch = BatchBacktest(config, factory)
        
        grid = ParameterGrid({
            "period": [3, 5, 7],
            "multiplier": [0.5, 1.0],
        })
        
        result = batch.run(grid)
        
        # Rankings should be indices sorted by score descending
        scores_by_rank = [result.scores[i] for i in result.rankings]
        assert scores_by_rank == sorted(result.scores, reverse=True)
    
    def test_get_ranked_results(self, tmp_path):
        """Test getting ranked results."""
        data_dir = create_test_data(tmp_path, num_minutes=20)
        
        data_config = DataConfig(
            tick_data_dir=data_dir,
            symbols=["BTCUSDT"],
            timeframe="1m",
        )
        
        config = BacktestConfig(
            data=data_config,
            initial_capital=10000.0,
        )
        
        def factory(p):
            return ParameterizedStrategy(**p)
        
        batch = BatchBacktest(config, factory)
        
        grid = ParameterGrid({"period": [3, 5]})
        batch_result = batch.run(grid)
        
        ranked = batch_result.get_ranked_results()
        
        assert len(ranked) == 2
        
        # First result should have highest score
        _, first_score, _ = ranked[0]
        _, second_score, _ = ranked[1]
        assert first_score >= second_score


class TestSelector:
    """Tests for Selector."""
    
    def test_select_top_k(self, tmp_path):
        """Test selecting top k results."""
        data_dir = create_test_data(tmp_path, num_minutes=30)
        
        data_config = DataConfig(
            tick_data_dir=data_dir,
            symbols=["BTCUSDT"],
            timeframe="1m",
        )
        
        config = BacktestConfig(
            data=data_config,
            initial_capital=10000.0,
        )
        
        def factory(p):
            return ParameterizedStrategy(**p)
        
        batch = BatchBacktest(config, factory)
        grid = ParameterGrid({"period": [3, 5, 7, 10]})
        batch_result = batch.run(grid)
        
        selector = Selector()
        top_2 = selector.select_top_k(batch_result, k=2, apply_filters=False)
        
        assert len(top_2) == 2
    
    def test_convenience_function(self, tmp_path):
        """Test select_top_k convenience function."""
        data_dir = create_test_data(tmp_path, num_minutes=20)
        
        data_config = DataConfig(
            tick_data_dir=data_dir,
            symbols=["BTCUSDT"],
            timeframe="1m",
        )
        
        config = BacktestConfig(
            data=data_config,
            initial_capital=10000.0,
        )
        
        def factory(p):
            return ParameterizedStrategy(**p)
        
        batch = BatchBacktest(config, factory)
        grid = ParameterGrid({"period": [3, 5, 7]})
        batch_result = batch.run(grid)
        
        top = select_top_k(batch_result, k=1)
        
        assert len(top) <= 1  # May be filtered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
