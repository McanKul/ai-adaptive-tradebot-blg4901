"""
Backtest/scoring/__init__.py
============================
Scoring and parameter optimization package.

This package provides:
- ParameterGrid / SearchSpace for generating parameter combinations
- Scorer / TrialAwareScorer for evaluating backtest results with AFML DSR
- BatchBacktest for running multiple parameter sets
- Selector for choosing best parameters
- Splits for leakage-safe evaluation (PurgedKFold + Embargo)
- DSR (Deflated Sharpe Ratio) utilities for selection bias correction
"""

from Backtest.scoring.search_space import ParameterGrid, SearchSpace
from Backtest.scoring.scorer import (
    Scorer, 
    create_scorer,
    TrialAwareScorer,
    compute_deflated_sharpe,
    selection_bias_warning,
)
from Backtest.scoring.batch import BatchBacktest, BatchResult, create_dummy_result
from Backtest.scoring.selector import Selector
from Backtest.scoring.splits import (
    PurgedKFold,
    embargo_split,
    walk_forward_splits,
    expanding_window_splits,
    combinatorial_purged_cv,
    TimeRange,
    CVResult,
)

__all__ = [
    # Search Space
    "ParameterGrid",
    "SearchSpace",
    # Scoring
    "Scorer",
    "create_scorer",
    "TrialAwareScorer",
    "compute_deflated_sharpe",
    "selection_bias_warning",
    # Batch
    "BatchBacktest",
    "BatchResult",
    "create_dummy_result",
    # Selection
    "Selector",
    # Splits
    "PurgedKFold",
    "embargo_split",
    "walk_forward_splits",
    "expanding_window_splits",
    "combinatorial_purged_cv",
    "TimeRange",
    "CVResult",
]
