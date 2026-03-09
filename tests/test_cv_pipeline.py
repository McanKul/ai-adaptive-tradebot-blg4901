"""
tests/test_cv_pipeline.py
=========================
Acceptance tests for Part C — CV splits integrated into the scoring pipeline.

Covers:
  C1  RealismConfig.cv_* fields wired through to run_with_cv
  C2  run_cv_from_config() convenience
  C3  _create_fold_config propagates realism
  C4  CV-aware Selector (stability penalty + max_cv_std filter)
  C5  Fold breakdown metadata in BatchResult / BacktestResult
  C6  No-leakage verification (embargo works)
"""
import pytest
import math
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any

from Interfaces.market_data import Bar
from Interfaces.orders import Order, OrderType, OrderSide
from Interfaces.strategy_adapter import StrategyContext
from Interfaces.metrics_interface import BacktestResult

from Backtest.runner import BacktestConfig, DataConfig
from Backtest.realism_config import RealismConfig, TransactionCostConfig
from Backtest.scoring.batch import BatchBacktest, BatchResult, create_dummy_result
from Backtest.scoring.scorer import Scorer, create_scorer
from Backtest.scoring.selector import Selector, SelectionCriteria
from Backtest.scoring.splits import (
    PurgedKFold,
    WalkForwardSplit,
    CombinatorialPurgedCV,
    TimeRange,
)
from Backtest.scoring.search_space import ParameterGrid


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
DAY_NS = 86_400_000_000_000
START_NS = 1_700_000_000_000_000_000
END_NS = START_NS + 100 * DAY_NS  # 100 days


class TrivialStrategy:
    """Deterministic strategy that trades on every Nth bar."""

    def __init__(self, period: int = 3, qty: float = 0.01):
        self.period = period
        self.qty = qty
        self._n = 0
        self._pos = 0.0

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> List[Order]:
        self._n += 1
        orders: List[Order] = []
        if self._n % self.period == 0:
            if self._pos == 0:
                orders.append(
                    Order(symbol=bar.symbol, side=OrderSide.BUY,
                          order_type=OrderType.MARKET, quantity=self.qty)
                )
                self._pos = self.qty
            else:
                orders.append(
                    Order(symbol=bar.symbol, side=OrderSide.SELL,
                          order_type=OrderType.MARKET, quantity=self._pos)
                )
                self._pos = 0.0
        return orders

    def reset(self):
        self._n = 0
        self._pos = 0.0


def _make_tick_csv(tmp_path, n_minutes: int = 60):
    """Create a tiny tick CSV and return the data directory path."""
    tick_file = tmp_path / "BTCUSDT_ticks.csv"
    ns_per_min = 60_000_000_000
    with open(tick_file, "w") as f:
        f.write("timestamp_ns,symbol,price,volume\n")
        for m in range(n_minutes):
            for t in range(5):
                ts = m * ns_per_min + t * (ns_per_min // 5)
                price = 50_000 + m * 2 + t
                f.write(f"{ts},BTCUSDT,{price},1.0\n")
    return str(tmp_path)


def _base_config(data_dir: str, start_ns=None, end_ns=None, realism=None):
    data = DataConfig(
        tick_data_dir=data_dir,
        symbols=["BTCUSDT"],
        timeframe="1m",
        start_ts_ns=start_ns,
        end_ts_ns=end_ns,
    )
    return BacktestConfig(
        data=data,
        initial_capital=10_000.0,
        realism=realism or RealismConfig(),
    )


def _factory(p: Dict[str, Any]):
    return TrivialStrategy(**p)


# ===================================================================
# C1 + C2  — run_cv_from_config reads RealismConfig.cv_*
# ===================================================================

class TestRunCvFromConfig:
    """run_cv_from_config() delegates to run_with_cv using RealismConfig.cv_*."""

    def test_cv_disabled_falls_back_to_run(self, tmp_path):
        """When cv_enabled=False, run_cv_from_config delegates to plain run()."""
        data_dir = _make_tick_csv(tmp_path)
        rc = RealismConfig(cv_enabled=False)
        config = _base_config(data_dir, realism=rc)
        batch = BatchBacktest(config, _factory)

        grid = ParameterGrid({"period": [3]})
        result = batch.run_cv_from_config(grid)

        # Plain run → no cv metadata on first result
        assert "cv_scores" not in result.results[0].metadata
        assert result.cv_method is None

    def test_cv_enabled_purged_kfold(self, tmp_path):
        """cv_enabled=True + purged_kfold runs CV pipeline."""
        data_dir = _make_tick_csv(tmp_path, n_minutes=120)
        ns_per_min = 60_000_000_000
        rc = RealismConfig(
            cv_enabled=True,
            cv_method="purged_kfold",
            cv_n_splits=3,
            cv_embargo_pct=0.01,
        )
        config = _base_config(data_dir, start_ns=0, end_ns=120 * ns_per_min, realism=rc)
        batch = BatchBacktest(config, _factory)

        grid = ParameterGrid({"period": [3]})
        result = batch.run_cv_from_config(grid)

        r0 = result.results[0]
        assert "cv_scores" in r0.metadata
        assert r0.metadata["cv_n_folds"] == 3
        assert result.cv_method == "purged_kfold"

    def test_cv_enabled_walk_forward(self, tmp_path):
        """cv_method='walk_forward' with explicit durations."""
        data_dir = _make_tick_csv(tmp_path, n_minutes=120)
        ns_per_min = 60_000_000_000
        rc = RealismConfig(
            cv_enabled=True,
            cv_method="walk_forward",
            cv_n_splits=3,
            cv_embargo_pct=0.01,
            cv_train_duration_ns=40 * ns_per_min,
            cv_test_duration_ns=20 * ns_per_min,
        )
        config = _base_config(data_dir, start_ns=0, end_ns=120 * ns_per_min, realism=rc)
        batch = BatchBacktest(config, _factory)

        grid = ParameterGrid({"period": [3]})
        result = batch.run_cv_from_config(grid)

        r0 = result.results[0]
        assert "cv_scores" in r0.metadata
        assert result.cv_method == "walk_forward"

    def test_cv_enabled_cpcv(self, tmp_path):
        """cv_method='combinatorial_purged' maps to cpcv."""
        data_dir = _make_tick_csv(tmp_path, n_minutes=120)
        ns_per_min = 60_000_000_000
        rc = RealismConfig(
            cv_enabled=True,
            cv_method="combinatorial_purged",
            cv_n_splits=4,
            cv_embargo_pct=0.01,
        )
        config = _base_config(data_dir, start_ns=0, end_ns=120 * ns_per_min, realism=rc)
        batch = BatchBacktest(config, _factory)

        grid = ParameterGrid({"period": [3]})
        result = batch.run_cv_from_config(grid)

        r0 = result.results[0]
        assert "cv_scores" in r0.metadata
        assert result.cv_method == "cpcv"


# ===================================================================
# C3  — _create_fold_config propagates realism
# ===================================================================

class TestFoldConfigPropagation:
    """Realism config is forwarded to each fold runner."""

    def test_fold_config_has_realism(self, tmp_path):
        data_dir = _make_tick_csv(tmp_path)
        rc = RealismConfig(
            transaction_costs=TransactionCostConfig(slippage_model="volume_sqrt"),
        )
        config = _base_config(data_dir, realism=rc)
        batch = BatchBacktest(config, _factory)

        tr = TimeRange(start_ns=0, end_ns=60_000_000_000)
        fold_cfg = batch._create_fold_config(tr)

        assert fold_cfg.realism is rc
        assert fold_cfg.realism.transaction_costs.slippage_model == "volume_sqrt"


# ===================================================================
# C4  — CV-aware Selector (stability penalty + max_cv_std)
# ===================================================================

class TestCVAwareSelector:
    """Selector uses cv_score_std for stability-aware ranking."""

    @staticmethod
    def _make_result(score: float, cv_std: float, **kwargs) -> BacktestResult:
        md = {
            "cv_score_std": cv_std,
            "cv_scores": [score - cv_std, score + cv_std],
        }
        md.update(kwargs)
        return BacktestResult(
            sharpe_ratio=score,
            total_trades=20,
            max_drawdown=0.1,
            win_rate=0.5,
            total_return_pct=score * 10,
            initial_capital=10_000,
            total_costs=10,
            turnover=5,
            metadata=md,
        )

    def test_stability_weight_reorders(self):
        """Higher cv_stability_weight should prefer lower cv_std."""
        # Result A: high score (2.0) but high std (1.5)
        # Result B: lower score (1.5) but low std (0.1)
        rA = self._make_result(2.0, cv_std=1.5)
        rB = self._make_result(1.5, cv_std=0.1)

        batch = BatchResult(
            results=[rA, rB],
            scores=[2.0, 1.5],
            params_list=[{"a": 1}, {"a": 2}],
            rankings=[0, 1],
            total_time_seconds=0,
        )

        # Without stability penalty → A wins
        sel_no = Selector(SelectionCriteria(min_trades=0, cv_stability_weight=0.0))
        top_no = sel_no.select_top_k(batch, k=2, apply_filters=False)
        assert top_no[0][2] == {"a": 1}  # A comes first

        # With stability penalty → B wins
        # effective_A = 2.0 - 1.0*1.5 = 0.5
        # effective_B = 1.5 - 1.0*0.1 = 1.4
        sel_yes = Selector(SelectionCriteria(min_trades=0, cv_stability_weight=1.0))
        top_yes = sel_yes.select_top_k(batch, k=2, apply_filters=False)
        assert top_yes[0][2] == {"a": 2}  # B now first

    def test_max_cv_std_filter(self):
        """Results with cv_score_std > max_cv_std are filtered out."""
        rA = self._make_result(2.0, cv_std=2.0)
        rB = self._make_result(1.5, cv_std=0.3)

        batch = BatchResult(
            results=[rA, rB],
            scores=[2.0, 1.5],
            params_list=[{"a": 1}, {"a": 2}],
            rankings=[0, 1],
            total_time_seconds=0,
        )

        sel = Selector(SelectionCriteria(min_trades=0, max_cv_std=0.5))
        top = sel.select_top_k(batch, k=10, apply_filters=True)
        # Only B passes
        assert len(top) == 1
        assert top[0][2] == {"a": 2}


# ===================================================================
# C5  — Fold breakdown metadata
# ===================================================================

class TestFoldBreakdownMetadata:
    """BatchResult and BacktestResult carry per-fold detail."""

    def test_cv_metadata_fields(self, tmp_path):
        """run_with_cv produces cv_score_mean/std/min/max + cv_fold_details."""
        data_dir = _make_tick_csv(tmp_path, n_minutes=120)
        ns_per_min = 60_000_000_000
        config = _base_config(data_dir, start_ns=0, end_ns=120 * ns_per_min)
        batch = BatchBacktest(config, _factory)

        grid = ParameterGrid({"period": [3]})
        result = batch.run_with_cv(grid, split_mode="purged_kfold", n_splits=3)

        r0 = result.results[0]
        md = r0.metadata

        # Required metadata keys
        assert "cv_scores" in md
        assert "cv_score_mean" in md
        assert "cv_score_std" in md
        assert "cv_score_min" in md
        assert "cv_score_max" in md
        assert "cv_n_folds" in md
        assert "cv_fold_details" in md

        # fold_details has one entry per fold
        assert len(md["cv_fold_details"]) == md["cv_n_folds"]

        # Each fold detail has required fields
        for d in md["cv_fold_details"]:
            assert "fold_idx" in d
            assert "score" in d
            assert "sharpe" in d
            assert "total_return_pct" in d
            assert "max_drawdown" in d
            assert "total_trades" in d
            assert "total_costs" in d

    def test_batch_result_cv_fold_details_list(self, tmp_path):
        """BatchResult.cv_fold_details is a list parallel to results."""
        data_dir = _make_tick_csv(tmp_path, n_minutes=120)
        ns_per_min = 60_000_000_000
        config = _base_config(data_dir, start_ns=0, end_ns=120 * ns_per_min)
        batch = BatchBacktest(config, _factory)

        grid = ParameterGrid({"period": [3, 5]})
        result = batch.run_with_cv(grid, split_mode="purged_kfold", n_splits=3)

        assert result.cv_fold_details is not None
        assert len(result.cv_fold_details) == 2  # one per param combo
        assert result.cv_method == "purged_kfold"


# ===================================================================
# C6  — No-leakage verification
# ===================================================================

class TestNoLeakage:
    """Verify that train/test ranges in CV have no overlap (embargo respected)."""

    def test_purged_kfold_no_overlap(self):
        """PurgedKFold: train ranges never overlap test (with embargo)."""
        splitter = PurgedKFold(
            start_ns=START_NS, end_ns=END_NS,
            n_splits=5, embargo_pct=0.02,
        )
        for train_ranges, test_range in splitter.split():
            for tr in train_ranges:
                # No overlap at all
                assert tr.end_ns <= test_range.start_ns or tr.start_ns >= test_range.end_ns, (
                    f"Overlap detected: train {tr} vs test {test_range}"
                )

    def test_walk_forward_no_overlap(self):
        """WalkForward: embargo separates train from test."""
        total = END_NS - START_NS
        splitter = WalkForwardSplit(
            start_ns=START_NS, end_ns=END_NS,
            train_duration_ns=total // 3,
            test_duration_ns=total // 6,
            embargo_pct=0.02,
        )
        for train_ranges, test_range in splitter.split():
            for tr in train_ranges:
                assert tr.end_ns <= test_range.start_ns, (
                    f"Train end {tr.end_ns} > test start {test_range.start_ns}"
                )

    def test_cpcv_no_overlap(self):
        """CPCV: train ranges never include test fold timestamps."""
        splitter = CombinatorialPurgedCV(
            start_ns=START_NS, end_ns=END_NS,
            n_splits=5, n_test_splits=2, embargo_pct=0.02,
        )
        for train_ranges, test_ranges in splitter.split():
            for tr in train_ranges:
                for te in test_ranges:
                    overlap_start = max(tr.start_ns, te.start_ns)
                    overlap_end = min(tr.end_ns, te.end_ns)
                    assert overlap_start >= overlap_end, (
                        f"Overlap: train {tr} vs test {te}"
                    )


# ===================================================================
# Integration: run_with_cv + selector end-to-end
# ===================================================================

class TestEndToEndCVSelection:
    """Full pipeline: batch CV → selector picks stable winner."""

    def test_selector_picks_from_cv_batch(self, tmp_path):
        """Selector can pick best from CV batch result."""
        data_dir = _make_tick_csv(tmp_path, n_minutes=120)
        ns_per_min = 60_000_000_000
        config = _base_config(data_dir, start_ns=0, end_ns=120 * ns_per_min)
        batch = BatchBacktest(config, _factory)

        grid = ParameterGrid({"period": [3, 5, 7]})
        br = batch.run_with_cv(grid, split_mode="purged_kfold", n_splits=3)

        sel = Selector(SelectionCriteria(min_trades=0, min_sharpe=-999))
        best = sel.select_best(br)

        assert best is not None
        result, score, params = best
        assert "period" in params
        assert "cv_scores" in result.metadata

    def test_cv_aggregate_modes(self, tmp_path):
        """mean / median / min aggregation each run without error."""
        data_dir = _make_tick_csv(tmp_path, n_minutes=120)
        ns_per_min = 60_000_000_000
        config = _base_config(data_dir, start_ns=0, end_ns=120 * ns_per_min)
        grid = ParameterGrid({"period": [3]})

        for agg in ("mean", "median", "min"):
            batch = BatchBacktest(config, _factory)
            br = batch.run_with_cv(grid, aggregate=agg, split_mode="purged_kfold", n_splits=3)
            assert br.results[0].metadata["cv_aggregate"] == agg
