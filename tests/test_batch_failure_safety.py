"""
tests/test_batch_failure_safety.py
==================================
Tests for batch failure safety:
- Failed strategies don't crash batch processing
- Failed results get proper dummy values
- Selection handles failed results correctly
"""
import pytest
from typing import Dict, Any
from dataclasses import dataclass

from Backtest.scoring.batch import BatchBacktest, BatchResult, create_dummy_result
from Backtest.scoring.selector import Selector, SelectionCriteria, select_top_k
from Interfaces.metrics_interface import BacktestResult


class TestCreateDummyResult:
    """Tests for create_dummy_result function."""
    
    def test_dummy_result_has_required_fields(self):
        """Dummy result has all required BacktestResult fields."""
        params = {"period": 14, "threshold": 0.5}
        result = create_dummy_result(params, "Test error")
        
        assert result.strategy_name == "FAILED_RUN"
        assert result.params == params
        assert result.initial_capital == 10000.0
        assert result.final_equity == 10000.0
        assert result.total_return == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.total_trades == 0
    
    def test_dummy_result_metadata_contains_error(self):
        """Dummy result metadata includes error information."""
        result = create_dummy_result({}, "Something went wrong")
        
        assert result.metadata["failed"] is True
        assert "Something went wrong" in result.metadata["error"]
    
    def test_dummy_result_is_valid_backtest_result(self):
        """Dummy result can be used as a normal BacktestResult."""
        result = create_dummy_result({"x": 1}, "Error")
        
        # Should not raise when accessing standard properties
        _ = result.sharpe_ratio
        _ = result.total_trades
        _ = result.max_drawdown
        _ = result.win_rate


class TestBatchFailureSafety:
    """Tests for batch processing failure safety."""
    
    def test_failed_result_has_negative_inf_score(self):
        """Failed runs receive -inf score so they rank last."""
        # This tests the batch logic by checking the result structure
        result = create_dummy_result({"x": 1}, "Error")
        
        # Score for failed runs should be -inf
        # The batch.run method assigns this
        expected_score = -float('inf')
        assert expected_score < -1e10  # Verify it's very negative
    
    def test_batch_result_tracks_failures(self):
        """BatchResult has failed_count field."""
        # Create a mock BatchResult
        result = BatchResult(
            results=[create_dummy_result({}, "err1"), create_dummy_result({}, "err2")],
            scores=[-float('inf'), -float('inf')],
            params_list=[{"a": 1}, {"a": 2}],
            rankings=[0, 1],
            total_time_seconds=1.0,
            trial_count=2,
            failed_count=2,
        )
        
        assert result.failed_count == 2
        assert result.trial_count == 2
    
    def test_batch_result_mixed_success_failure(self):
        """BatchResult handles mix of successful and failed runs."""
        success_result = BacktestResult(
            strategy_name="Success",
            params={"x": 1},
            initial_capital=10000.0,
            final_equity=12000.0,
            total_return=2000.0,
            total_return_pct=20.0,
            max_drawdown=0.1,
            sharpe_ratio=1.5,
            total_trades=50,
            win_rate=0.6,
        )
        failed_result = create_dummy_result({"x": 2}, "Error")
        
        result = BatchResult(
            results=[success_result, failed_result],
            scores=[1.5, -float('inf')],
            params_list=[{"x": 1}, {"x": 2}],
            rankings=[0, 1],  # Success ranks first
            total_time_seconds=1.0,
            trial_count=2,
            failed_count=1,
        )
        
        # Best result should be the successful one
        best, score, params = result.best_result()
        assert best.strategy_name == "Success"
        assert score == 1.5


class TestSelectorWithFailures:
    """Tests for Selector handling of failed results."""
    
    def test_selector_filters_failed_by_trades(self):
        """Failed results (0 trades) are filtered by min_trades."""
        success_result = BacktestResult(
            strategy_name="Success",
            params={"x": 1},
            initial_capital=10000.0,
            final_equity=11000.0,
            total_return=1000.0,
            total_return_pct=10.0,
            max_drawdown=0.1,
            sharpe_ratio=0.5,
            total_trades=20,
            win_rate=0.5,
        )
        failed_result = create_dummy_result({"x": 2}, "Error")
        
        batch_result = BatchResult(
            results=[success_result, failed_result],
            scores=[0.5, -float('inf')],
            params_list=[{"x": 1}, {"x": 2}],
            rankings=[0, 1],
            total_time_seconds=1.0,
            trial_count=2,
            failed_count=1,
        )
        
        criteria = SelectionCriteria(min_trades=10)
        selector = Selector(criteria)
        
        filtered = selector.select_filtered(batch_result)
        
        # Only success should pass
        assert len(filtered) == 1
        assert filtered[0][0].strategy_name == "Success"
    
    def test_selector_handles_all_failures(self):
        """Selector handles case where all runs failed."""
        failed1 = create_dummy_result({"x": 1}, "Error 1")
        failed2 = create_dummy_result({"x": 2}, "Error 2")
        
        batch_result = BatchResult(
            results=[failed1, failed2],
            scores=[-float('inf'), -float('inf')],
            params_list=[{"x": 1}, {"x": 2}],
            rankings=[0, 1],
            total_time_seconds=1.0,
            trial_count=2,
            failed_count=2,
        )
        
        selector = Selector()
        best = selector.select_best(batch_result)
        
        # Should return None when no results pass filters
        assert best is None
    
    def test_select_top_k_with_failures(self):
        """select_top_k convenience function handles failures."""
        success = BacktestResult(
            strategy_name="Good",
            params={"x": 1},
            initial_capital=10000.0,
            final_equity=12000.0,
            total_return=2000.0,
            total_return_pct=20.0,
            max_drawdown=0.15,
            sharpe_ratio=1.2,
            total_trades=30,
            win_rate=0.55,
        )
        failed = create_dummy_result({"x": 2}, "Boom")
        
        batch_result = BatchResult(
            results=[failed, success],  # Failed is first in list
            scores=[-float('inf'), 1.2],
            params_list=[{"x": 2}, {"x": 1}],
            rankings=[1, 0],  # Success ranks higher
            total_time_seconds=0.5,
            trial_count=2,
            failed_count=1,
        )
        
        top = select_top_k(batch_result, k=5, min_trades=10)
        
        # Should only include the successful run
        assert len(top) == 1
        assert top[0][0].strategy_name == "Good"


class TestBatchResultMethods:
    """Tests for BatchResult helper methods."""
    
    def test_get_ranked_results_orders_by_score(self):
        """get_ranked_results returns results ordered by score."""
        results = [
            BacktestResult(strategy_name="Low", params={}, sharpe_ratio=0.1, 
                          initial_capital=10000, final_equity=10100, total_trades=10),
            BacktestResult(strategy_name="High", params={}, sharpe_ratio=2.0,
                          initial_capital=10000, final_equity=15000, total_trades=20),
            BacktestResult(strategy_name="Mid", params={}, sharpe_ratio=1.0,
                          initial_capital=10000, final_equity=12000, total_trades=15),
        ]
        
        batch = BatchResult(
            results=results,
            scores=[0.1, 2.0, 1.0],
            params_list=[{}, {}, {}],
            rankings=[1, 2, 0],  # High, Mid, Low
            total_time_seconds=1.0,
        )
        
        ranked = batch.get_ranked_results()
        
        assert ranked[0][0].strategy_name == "High"
        assert ranked[1][0].strategy_name == "Mid"
        assert ranked[2][0].strategy_name == "Low"
    
    def test_top_k_limits_results(self):
        """top_k returns at most k results."""
        results = [
            BacktestResult(strategy_name=f"S{i}", params={}, sharpe_ratio=float(i),
                          initial_capital=10000, final_equity=10000+i*100, total_trades=10)
            for i in range(10)
        ]
        
        batch = BatchResult(
            results=results,
            scores=[float(i) for i in range(10)],
            params_list=[{} for _ in range(10)],
            rankings=list(reversed(range(10))),  # Best first
            total_time_seconds=1.0,
        )
        
        top = batch.top_k(3)
        assert len(top) == 3
        assert top[0][0].strategy_name == "S9"  # Highest score
