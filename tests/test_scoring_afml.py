"""
tests/test_scoring_afml.py
===========================
Tests for AFML-aligned scoring components:
- TrialAwareScorer with Deflated Sharpe Ratio
- BatchBacktest error handling
- Selection bias reporting

These tests verify the AFML discipline for multiple comparisons:
- DSR calculation accounts for number of trials
- Failed runs don't crash the batch
- Selection bias warnings are properly generated
"""
import pytest
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from Backtest.scoring.scorer import (
    Scorer,
    TrialAwareScorer,
    compute_deflated_sharpe,
    selection_bias_warning,
)
from Backtest.scoring.batch import BatchResult, create_dummy_result
from Backtest.scoring.search_space import SearchSpace


def create_mock_result(
    sharpe_ratio: float = 1.0,
    total_return: float = 0.1,
    max_drawdown: float = 0.1,
    total_trades: int = 10,
    turnover: float = 1.0,
    total_costs: float = 0.0,
    initial_capital: float = 10000.0,
    win_rate: float = 0.5,
    profit_factor: float = 1.5,
    calmar_ratio: float = 1.0,
):
    """Create a mock BacktestResult-like object for testing."""
    return type('MockResult', (), {
        'sharpe_ratio': sharpe_ratio,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'turnover': turnover,
        'total_costs': total_costs,
        'initial_capital': initial_capital,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'calmar_ratio': calmar_ratio,
    })()


class TestDeflatedSharpeRatio:
    """Test Deflated Sharpe Ratio calculation."""
    
    def test_dsr_decreases_with_more_trials(self):
        """DSR should decrease as number of trials increases (more luck involved)."""
        sharpe = 2.0
        
        dsr_10 = compute_deflated_sharpe(
            sharpe=sharpe,
            n_trials=10,
        )
        
        dsr_100 = compute_deflated_sharpe(
            sharpe=sharpe,
            n_trials=100,
        )
        
        dsr_1000 = compute_deflated_sharpe(
            sharpe=sharpe,
            n_trials=1000,
        )
        
        assert dsr_10 > dsr_100 > dsr_1000, "DSR should decrease with more trials"
    
    def test_dsr_is_less_than_raw_sharpe(self):
        """DSR should be less than raw Sharpe when trials > 1.
        
        The point of DSR is to penalize selection bias - after trying many
        parameter combinations, your "best" Sharpe is inflated by luck.
        DSR corrects for this inflation.
        """
        sharpe = 1.5
        n_trials = 100
        
        dsr = compute_deflated_sharpe(
            sharpe=sharpe,
            n_trials=n_trials,
        )
        
        # DSR should be less than raw Sharpe due to selection bias penalty
        assert dsr < sharpe, "DSR should be less than raw Sharpe (selection bias adjustment)"
    
    def test_dsr_formula_is_deterministic(self):
        """DSR calculation should be deterministic."""
        sharpe = 1.0
        n_trials = 50
        
        dsr1 = compute_deflated_sharpe(sharpe=sharpe, n_trials=n_trials)
        dsr2 = compute_deflated_sharpe(sharpe=sharpe, n_trials=n_trials)
        
        assert dsr1 == dsr2, "DSR should be deterministic"
    
    def test_dsr_increases_with_more_data(self):
        """More data (years) should give more confidence, higher DSR."""
        sharpe = 1.5
        n_trials = 100
        
        dsr_short = compute_deflated_sharpe(
            sharpe=sharpe,
            n_trials=n_trials,
            t_years=0.5,  # 6 months
        )
        
        dsr_long = compute_deflated_sharpe(
            sharpe=sharpe,
            n_trials=n_trials,
            t_years=5.0,  # 5 years
        )
        
        assert dsr_long > dsr_short, "More data should give higher DSR"
    
    def test_dsr_handles_edge_cases(self):
        """DSR should handle edge cases gracefully."""
        # Single trial - should return original sharpe
        dsr_single = compute_deflated_sharpe(
            sharpe=1.0,
            n_trials=1,
        )
        assert dsr_single == 1.0, "Single trial should return original Sharpe"
        
        # Zero sharpe
        dsr_zero = compute_deflated_sharpe(
            sharpe=0.0,
            n_trials=100,
        )
        assert dsr_zero < 0, "Zero Sharpe with many trials should be negative (inflated by luck)"


class TestSelectionBiasWarning:
    """Test selection bias warning generation."""
    
    def test_warning_generated(self):
        """Should generate warning string with relevant info."""
        warning = selection_bias_warning(
            n_trials=100,
            best_sharpe=1.5,
        )
        
        assert warning is not None
        assert "100" in warning  # Trial count should be mentioned
        assert "1.5" in warning or "1.50" in warning  # Best sharpe
    
    def test_warning_mentions_dsr(self):
        """Warning should mention deflated sharpe ratio."""
        warning = selection_bias_warning(
            n_trials=500,
            best_sharpe=1.0,
        )
        
        assert "DSR" in warning or "deflated" in warning.lower()


class TestTrialAwareScorer:
    """Test TrialAwareScorer class."""
    
    def test_tracks_trial_count(self):
        """Should track number of trials scored."""
        scorer = TrialAwareScorer()
        
        # Score some results
        for i in range(5):
            result = create_mock_result(sharpe_ratio=1.0 + i * 0.1)
            scorer.score(result)
        
        assert scorer.trial_count == 5
    
    def test_tracks_best_sharpe(self):
        """Should track best Sharpe ratio seen."""
        scorer = TrialAwareScorer()
        
        sharpes = [0.5, 2.0, 1.5, 0.8, 1.2]
        for sharpe in sharpes:
            result = create_mock_result(sharpe_ratio=sharpe)
            scorer.score(result)
        
        assert scorer.best_sharpe == 2.0
    
    def test_generates_trial_report(self):
        """Should generate trial report."""
        scorer = TrialAwareScorer()
        
        for i in range(100):
            result = create_mock_result(sharpe_ratio=0.5 + (i % 10) * 0.1)
            scorer.score(result)
        
        report = scorer.get_trial_report()
        
        assert 'trial_count' in report
        assert report['trial_count'] == 100
        assert 'best_sharpe' in report
        assert 'deflated_sharpe' in report
    
    def test_reset_trials(self):
        """Should be able to reset trial tracking."""
        scorer = TrialAwareScorer()
        
        # Score some results
        for i in range(10):
            result = create_mock_result(sharpe_ratio=1.0)
            scorer.score(result)
        
        assert scorer.trial_count == 10
        
        scorer.reset_trials()
        assert scorer.trial_count == 0


class TestBatchErrorHandling:
    """Test BatchBacktest error handling."""
    
    def test_create_dummy_result(self):
        """create_dummy_result should return valid placeholder result."""
        params = {"lookback": 20, "threshold": 0.5}
        error_msg = "Strategy crashed: division by zero"
        
        dummy = create_dummy_result(params, error_msg)
        
        # Should have all required fields
        assert dummy.params == params
        # Error is stored in metadata
        assert dummy.metadata.get("failed") == True
        assert dummy.metadata.get("error") == error_msg
        
        # Should have safe default values
        assert dummy.total_return == 0.0
        assert dummy.total_trades == 0
    
    def test_batch_result_basics(self):
        """BatchResult should track trial counts in selection_bias_report."""
        # Create a minimal BatchResult
        from Interfaces.metrics_interface import BacktestResult
        
        # Create a dummy result for the list
        dummy = BacktestResult(
            strategy_name="test",
            params={},
            initial_capital=10000.0,
            final_equity=10000.0,
            total_return=0.0,
            total_return_pct=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            total_trades=0,
            win_rate=0.0,
        )
        
        batch_result = BatchResult(
            results=[dummy],
            scores=[0.0],
            params_list=[{}],
            rankings=[0],
            total_time_seconds=1.0,
            trial_count=100,
            failed_count=5,
            selection_bias_report={'total_trials': 100, 'failed_trials': 5}
        )
        
        assert batch_result.trial_count == 100
        assert batch_result.failed_count == 5
        assert batch_result.selection_bias_report['failed_trials'] == 5


class TestSearchSpaceConstraints:
    """Test SearchSpace constraint validation."""
    
    def test_require_less_than(self):
        """Test less-than constraint."""
        space = SearchSpace()
        space.add("fast_period", [5, 10, 20])
        space.add("slow_period", [20, 50, 100])
        space.require_less_than("fast_period", "slow_period")
        
        combos = list(space)
        
        # Should only have combos where fast_period < slow_period
        for params in combos:
            assert params["fast_period"] < params["slow_period"]
    
    def test_require_range(self):
        """Test range constraint."""
        space = SearchSpace()
        space.add("lookback", [5, 10, 20, 50, 100])
        space.require_range("lookback", min_val=10, max_val=50)
        
        combos = list(space)
        
        for params in combos:
            assert 10 <= params["lookback"] <= 50
    
    def test_require_max_leverage(self):
        """Test max leverage constraint (applies to 'leverage' parameter)."""
        space = SearchSpace()
        space.add("leverage", [1, 2, 3, 5, 10])
        space.require_max_leverage(max_leverage=3)
        
        combos = list(space)
        
        for params in combos:
            assert params["leverage"] <= 3
    
    def test_multiple_constraints(self):
        """Test multiple constraints together."""
        space = SearchSpace()
        space.add("fast", [5, 10, 20])
        space.add("slow", [10, 20, 50])
        space.add("size", [0.1, 0.5, 1.0])
        
        space.require_less_than("fast", "slow")
        space.require_range("size", min_val=0.1, max_val=0.5)
        
        combos = list(space)
        
        for params in combos:
            assert params["fast"] < params["slow"]
            assert 0.1 <= params["size"] <= 0.5
    
    def test_constraint_reduces_space(self):
        """Constraints should reduce the parameter space."""
        space_unconstrained = SearchSpace()
        space_unconstrained.add("a", [1, 2, 3, 4, 5])
        space_unconstrained.add("b", [1, 2, 3, 4, 5])
        
        space_constrained = SearchSpace()
        space_constrained.add("a", [1, 2, 3, 4, 5])
        space_constrained.add("b", [1, 2, 3, 4, 5])
        space_constrained.require_less_than("a", "b")
        
        unconstrained_count = len(list(space_unconstrained))
        constrained_count = len(list(space_constrained))
        
        assert constrained_count < unconstrained_count
        # Unconstrained: 5*5 = 25
        # Constrained: pairs where a < b = 10 (1<2,3,4,5; 2<3,4,5; 3<4,5; 4<5)
        assert constrained_count == 10


class TestScorerIntegration:
    """Integration tests for scorer components."""
    
    def test_scorer_score_function(self):
        """Test basic scorer score calculation."""
        scorer = Scorer()
        
        result = create_mock_result(
            sharpe_ratio=1.5,
            total_return=0.2,
            max_drawdown=0.1,
            total_trades=50,
        )
        
        score = scorer.score(result)
        
        assert isinstance(score, float)
    
    def test_negative_sharpe_gets_low_score(self):
        """Negative Sharpe should result in lower score than positive."""
        scorer = Scorer()
        
        positive_result = create_mock_result(
            sharpe_ratio=1.0,
            total_return=0.1,
            max_drawdown=0.1,
            total_trades=10,
        )
        
        negative_result = create_mock_result(
            sharpe_ratio=-1.0,
            total_return=-0.1,
            max_drawdown=0.3,
            total_trades=10,
        )
        
        positive_score = scorer.score(positive_result)
        negative_score = scorer.score(negative_result)
        
        assert positive_score > negative_score


class TestCVSplits:
    """Tests for cross-validation split functionality."""
    
    def test_purged_kfold_split_count(self):
        """PurgedKFold should generate n_splits folds."""
        from Backtest.scoring.splits import PurgedKFold
        
        start_ns = 1704067200_000_000_000  # 2024-01-01
        end_ns = start_ns + 10 * 86400_000_000_000  # 10 days
        
        splitter = PurgedKFold(
            start_ns=start_ns,
            end_ns=end_ns,
            n_splits=5,
            embargo_pct=0.01,
        )
        
        splits = list(splitter.split())
        assert len(splits) == 5
    
    def test_purged_kfold_no_overlap(self):
        """Test ranges should not overlap with each other."""
        from Backtest.scoring.splits import PurgedKFold
        
        start_ns = 1704067200_000_000_000
        end_ns = start_ns + 10 * 86400_000_000_000
        
        splitter = PurgedKFold(
            start_ns=start_ns,
            end_ns=end_ns,
            n_splits=5,
        )
        
        test_ranges = splitter.get_test_ranges()
        
        for i, r1 in enumerate(test_ranges):
            for j, r2 in enumerate(test_ranges):
                if i != j:
                    assert not r1.overlaps(r2), f"Fold {i} and {j} overlap"
    
    def test_embargo_split_creates_gap(self):
        """Embargo split should create gap between train and test."""
        from Backtest.scoring.splits import embargo_split
        
        start_ns = 1704067200_000_000_000
        end_ns = start_ns + 100 * 86400_000_000_000  # 100 days
        
        train_range, test_range = embargo_split(
            start_ns=start_ns,
            end_ns=end_ns,
            test_pct=0.2,
            embargo_pct=0.05,
        )
        
        # Gap should exist between train end and test start
        gap = test_range.start_ns - train_range.end_ns
        expected_gap = int((end_ns - start_ns) * 0.05)
        
        assert gap >= expected_gap - 1  # Allow small rounding
    
    def test_walk_forward_sliding_window(self):
        """Walk forward should produce sliding windows."""
        from Backtest.scoring.splits import walk_forward_splits
        
        start_ns = 1704067200_000_000_000
        end_ns = start_ns + 100 * 86400_000_000_000
        train_ns = 30 * 86400_000_000_000  # 30 days
        test_ns = 10 * 86400_000_000_000   # 10 days
        
        splits = list(walk_forward_splits(
            start_ns=start_ns,
            end_ns=end_ns,
            train_duration_ns=train_ns,
            test_duration_ns=test_ns,
        ))
        
        assert len(splits) >= 1
        
        # Check that train windows slide
        if len(splits) >= 2:
            assert splits[1][0].start_ns > splits[0][0].start_ns
    
    def test_expanding_window_anchored(self):
        """Expanding window should keep train anchored at start."""
        from Backtest.scoring.splits import expanding_window_splits
        
        start_ns = 1704067200_000_000_000
        end_ns = start_ns + 100 * 86400_000_000_000
        initial_train_ns = 30 * 86400_000_000_000
        test_ns = 10 * 86400_000_000_000
        
        splits = list(expanding_window_splits(
            start_ns=start_ns,
            end_ns=end_ns,
            initial_train_ns=initial_train_ns,
            test_duration_ns=test_ns,
        ))
        
        assert len(splits) >= 1
        
        # All train ranges should start at the same point
        for train_range, _ in splits:
            assert train_range.start_ns == start_ns
        
        # Train ranges should expand
        if len(splits) >= 2:
            assert splits[1][0].end_ns > splits[0][0].end_ns
    
    def test_time_range_properties(self):
        """Test TimeRange helper methods."""
        from Backtest.scoring.splits import TimeRange
        
        r = TimeRange(start_ns=1000, end_ns=2000)
        
        assert r.duration_ns == 1000
        assert r.contains(1500)
        assert not r.contains(500)
        assert not r.contains(2500)
        
        r2 = TimeRange(start_ns=1500, end_ns=2500)
        assert r.overlaps(r2)
        
        r3 = TimeRange(start_ns=3000, end_ns=4000)
        assert not r.overlaps(r3)
    
    def test_cv_result_statistics(self):
        """Test CVResult aggregation."""
        from Backtest.scoring.splits import CVResult
        
        cv_result = CVResult(
            params={"a": 1},
            metric="sharpe_ratio",
            fold_scores=[1.0, 1.5, 2.0, 0.5, 1.0],
            fold_results=[],
            n_splits=5,
        )
        
        assert cv_result.valid_folds == 5
        assert abs(cv_result.mean_score - 1.2) < 0.01
        assert cv_result.min_score == 0.5
        assert cv_result.max_score == 2.0
        assert cv_result.is_consistent  # std < mean for positive


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
