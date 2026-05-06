"""
Backtest/scoring/batch.py
=========================
Batch backtesting for parameter sweeps.

Design decisions:
- Runs multiple parameter combinations
- Returns ranked results
- Supports leakage-safe evaluation via splits
- Robust error handling (failed runs get dummy results)
- Trial-aware reporting for AFML selection bias discipline
- Can optionally run in parallel (future extension)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Iterator, Callable, Tuple, Union
import logging
import time
import traceback

from Interfaces.strategy_adapter import IBacktestStrategy
from Interfaces.metrics_interface import BacktestResult

from Backtest.runner import BacktestRunner, BacktestConfig, DataConfig
from Backtest.realism_config import RealismConfig
from Backtest.scoring.search_space import ParameterGrid, SearchSpace
from Backtest.scoring.scorer import Scorer, create_scorer, TrialAwareScorer, selection_bias_warning
from Backtest.scoring.splits import PurgedKFold, WalkForwardSplit, CombinatorialPurgedCV, TimeRange

log = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result from a batch backtest."""
    results: List[BacktestResult]
    scores: List[float]
    params_list: List[Dict[str, Any]]
    rankings: List[int]  # Indices sorted by score (descending)
    total_time_seconds: float
    
    # Trial-aware reporting (AFML selection bias)
    trial_count: int = 0
    failed_count: int = 0
    selection_bias_report: Optional[Dict[str, Any]] = None
    
    # CV fold breakdown (Part C)
    cv_fold_details: Optional[List[Dict[str, Any]]] = None
    cv_method: Optional[str] = None
    
    def get_ranked_results(self) -> List[Tuple[BacktestResult, float, Dict[str, Any]]]:
        """Get results sorted by score (best first)."""
        return [
            (self.results[i], self.scores[i], self.params_list[i])
            for i in self.rankings
        ]
    
    def best_result(self) -> Tuple[BacktestResult, float, Dict[str, Any]]:
        """Get the best result."""
        idx = self.rankings[0]
        return self.results[idx], self.scores[idx], self.params_list[idx]
    
    def top_k(self, k: int) -> List[Tuple[BacktestResult, float, Dict[str, Any]]]:
        """Get top k results."""
        return self.get_ranked_results()[:k]
    
    def print_selection_bias_warning(self) -> None:
        """Print selection bias warning to log."""
        if self.selection_bias_report:
            log.warning(self.selection_bias_report.get("selection_bias_warning", ""))


def create_dummy_result(params: Dict[str, Any], error_msg: str) -> BacktestResult:
    """
    Create a safe dummy BacktestResult for failed runs.
    
    AFML Discipline: Failed runs should not crash batch processing.
    They receive minimal valid results with very low scores.
    
    Args:
        params: Parameters that caused the failure
        error_msg: Error message for debugging
        
    Returns:
        BacktestResult with minimal required fields
    """
    return BacktestResult(
        strategy_name="FAILED_RUN",
        params=params,
        initial_capital=10000.0,
        final_equity=10000.0,  # No change
        total_return=0.0,
        total_return_pct=0.0,
        max_drawdown=0.0,
        sharpe_ratio=0.0,
        total_trades=0,
        win_rate=0.0,
        metadata={
            "failed": True,
            "error": error_msg,
        }
    )


# Strategy factory type
StrategyFactory = Callable[[Dict[str, Any]], IBacktestStrategy]


class BatchBacktest:
    """
    Batch backtesting for parameter optimization.
    
    Runs multiple parameter combinations and ranks results.
    
    Usage:
        batch = BatchBacktest(config, factory)
        
        # Simple grid search
        grid = ParameterGrid({
            "rsi_period": [14, 21],
            "threshold": [0.5, 0.7],
        })
        result = batch.run(grid)
        
        # Get best parameters
        best_result, best_score, best_params = result.best_result()
    """
    
    def __init__(
        self,
        config: BacktestConfig,
        strategy_factory: StrategyFactory,
        scorer: Optional[Scorer] = None,
        strategy_name: str = "",
    ):
        """
        Initialize BatchBacktest.
        
        Args:
            config: BacktestConfig for runner
            strategy_factory: Factory to create strategies from params
            scorer: Scorer for ranking results (default scorer if None)
            strategy_name: Name for logging
        """
        self.config = config
        self.strategy_factory = strategy_factory
        self.scorer = scorer or create_scorer()
        self.strategy_name = strategy_name
        self._run_count = 0
    
    def run(
        self,
        param_space: ParameterGrid | SearchSpace | List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult:
        """
        Run batch backtest over parameter space.
        
        Args:
            param_space: ParameterGrid, SearchSpace, or list of param dicts
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            BatchResult with all results and rankings
        """
        start_time = time.time()
        
        # Convert to list
        if isinstance(param_space, (ParameterGrid, SearchSpace)):
            params_list = list(param_space)
        else:
            params_list = list(param_space)
        
        total = len(params_list)
        log.info(f"Starting batch backtest: {total} parameter combinations")
        
        results: List[BacktestResult] = []
        scores: List[float] = []
        failed_count = 0
        
        runner = BacktestRunner(self.config)
        
        # Use trial-aware scorer if available
        trial_scorer = TrialAwareScorer() if isinstance(self.scorer, Scorer) else None
        
        for i, params in enumerate(params_list):
            self._run_count += 1
            
            try:
                result = runner.run_once(
                    self.strategy_factory,
                    params,
                    self.strategy_name,
                )
                score = self.scorer.score(result)
                
                # Track for trial-aware reporting
                if trial_scorer:
                    trial_scorer.score(result)
                    
            except Exception as e:
                error_msg = f"{e.__class__.__name__}: {e}"
                log.error(f"Backtest failed for params {params}: {error_msg}")
                log.debug(traceback.format_exc())
                
                # Create a safe dummy result instead of crashing
                result = create_dummy_result(params, error_msg)
                score = -float('inf')
                failed_count += 1
            
            results.append(result)
            scores.append(score)
            
            if progress_callback:
                progress_callback(i + 1, total)
            
            if (i + 1) % max(1, total // 10) == 0:
                log.info(f"Progress: {i + 1}/{total} ({100 * (i + 1) / total:.0f}%)")
        
        # Rank by score (descending)
        rankings = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        elapsed = time.time() - start_time
        
        # Get trial-aware report if available
        selection_bias_report = None
        if trial_scorer:
            selection_bias_report = trial_scorer.get_trial_report()
        
        log.info(
            f"Batch complete: {total} runs in {elapsed:.1f}s "
            f"({elapsed / total:.2f}s per run), {failed_count} failed"
        )
        
        # Warn about selection bias
        if total > 10:
            best_sharpe = max((r.sharpe_ratio for r in results if r.sharpe_ratio > -999), default=0)
            log.warning(selection_bias_warning(total, best_sharpe))
        
        return BatchResult(
            results=results,
            scores=scores,
            params_list=params_list,
            rankings=rankings,
            total_time_seconds=elapsed,
            trial_count=total,
            failed_count=failed_count,
            selection_bias_report=selection_bias_report,
        )
    
    def run_with_cv(
        self,
        param_space: ParameterGrid | SearchSpace | List[Dict[str, Any]],
        splitter: Optional[Union[PurgedKFold, WalkForwardSplit, CombinatorialPurgedCV]] = None,
        aggregate: str = "mean",  # 'mean', 'median', 'min'
        split_mode: str = "purged_kfold",  # 'purged_kfold', 'walk_forward', 'cpcv'
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        train_pct: float = 0.6,  # For walk_forward
        n_test_splits: int = 2,  # For cpcv
    ) -> BatchResult:
        """
        Run batch backtest with cross-validation.
        
        For each parameter combination, runs on multiple folds
        and aggregates scores for leakage-safe evaluation.
        
        Args:
            param_space: Parameter space to search
            splitter: Pre-configured splitter (if None, creates based on split_mode)
            aggregate: How to aggregate fold scores ('mean', 'median', 'min')
            split_mode: Type of split - 'purged_kfold', 'walk_forward', 'cpcv'
            n_splits: Number of folds (for purged_kfold and cpcv)
            embargo_pct: Embargo gap as fraction of total period
            train_pct: Training fraction for walk_forward mode
            n_test_splits: Number of test folds for cpcv mode
            
        Returns:
            BatchResult with aggregated scores
        """
        start_time = time.time()
        
        if isinstance(param_space, (ParameterGrid, SearchSpace)):
            params_list = list(param_space)
        else:
            params_list = list(param_space)
        
        total = len(params_list)
        
        # Create splitter if not provided.  Earlier versions defaulted
        # to ``start=0, end=1 year`` whenever ``data.start_ts_ns`` was
        # not pre-populated — every fold landed outside the actual
        # tick data range and produced 0 trades, so the aggregate
        # scores were meaningless.  Inferring from disk is the right
        # behaviour and we now refuse to fall back to fictitious
        # bounds when no data is found.
        if splitter is None:
            start_ns = self.config.data.start_ts_ns
            end_ns = self.config.data.end_ts_ns
            # Use ``is None`` rather than truthiness so 0 (a legitimate
            # start_ts in synthetic test data) doesn't trigger the
            # inference path.
            if start_ns is None or end_ns is None:
                inferred = self._infer_tick_range()
                if inferred is None:
                    raise ValueError(
                        "BatchBacktest CV needs an explicit time range.  "
                        "Set DataConfig.start_ts_ns/end_ts_ns or backfill "
                        "tick data so the bounds can be inferred from "
                        "filenames in `data.tick_data_dir`."
                    )
                start_ns, end_ns = inferred
                log.info(
                    "CV time range inferred from tick data: "
                    "%d -> %d", start_ns, end_ns,
                )
            total_duration = end_ns - start_ns
            
            if split_mode == "purged_kfold":
                splitter = PurgedKFold(
                    start_ns=start_ns,
                    end_ns=end_ns,
                    n_splits=n_splits,
                    embargo_pct=embargo_pct,
                )
            elif split_mode == "walk_forward":
                train_duration = int(total_duration * train_pct)
                test_duration = int(total_duration * (1 - train_pct) / n_splits)
                splitter = WalkForwardSplit(
                    start_ns=start_ns,
                    end_ns=end_ns,
                    train_duration_ns=train_duration,
                    test_duration_ns=test_duration,
                    embargo_pct=embargo_pct,
                )
            elif split_mode == "cpcv":
                splitter = CombinatorialPurgedCV(
                    start_ns=start_ns,
                    end_ns=end_ns,
                    n_splits=n_splits,
                    n_test_splits=n_test_splits,
                    embargo_pct=embargo_pct,
                )
            else:
                raise ValueError(f"Unknown split_mode: {split_mode}. Use 'purged_kfold', 'walk_forward', or 'cpcv'")
        
        n_folds = splitter.get_n_splits() if hasattr(splitter, 'get_n_splits') else splitter.n_splits
        log.info(
            f"Starting CV batch backtest: {total} params x {n_folds} folds (mode={split_mode})"
        )
        
        # Get fold time ranges
        folds = list(splitter.split())
        
        # Detect if this is CPCV mode (test_data is a list of TimeRange)
        is_cpcv = len(folds) > 0 and isinstance(folds[0][1], list)
        if is_cpcv:
            log.info(
                "CPCV mode: Running each test fold window separately "
                "and aggregating (no merging of test ranges)"
            )
        
        final_results: List[BacktestResult] = []
        final_scores: List[float] = []
        
        for i, params in enumerate(params_list):
            fold_scores: List[float] = []
            fold_results: List[BacktestResult] = []
            
            for fold_idx, (train_ranges, test_data) in enumerate(folds):
                # test_data can be a single TimeRange or a list of TimeRange (CPCV)
                if isinstance(test_data, list):
                    # CombinatorialPurgedCV: run EACH test range separately
                    # This avoids incorrectly merging disjoint test windows
                    range_scores: List[float] = []
                    range_results: List[BacktestResult] = []
                    
                    for range_idx, test_range in enumerate(test_data):
                        fold_config = self._create_fold_config(test_range)
                        runner = BacktestRunner(fold_config)
                        
                        try:
                            result = runner.run_once(
                                self.strategy_factory,
                                params,
                                self.strategy_name,
                            )
                            score = self.scorer.score(result)
                        except Exception as e:
                            log.error(f"CPCV split {fold_idx} range {range_idx} failed: {e}")
                            result = create_dummy_result(params, str(e))
                            score = -float('inf')
                        
                        range_scores.append(score)
                        range_results.append(result)
                    
                    # Aggregate scores across ranges within this split
                    if aggregate == "mean":
                        split_score = sum(range_scores) / len(range_scores) if range_scores else 0
                    elif aggregate == "median":
                        sorted_scores = sorted(range_scores)
                        mid = len(sorted_scores) // 2
                        split_score = sorted_scores[mid] if sorted_scores else 0
                    elif aggregate == "min":
                        split_score = min(range_scores) if range_scores else 0
                    else:
                        split_score = sum(range_scores) / len(range_scores) if range_scores else 0
                    
                    # Store aggregated score for this CPCV split
                    fold_scores.append(split_score)
                    
                    # Use last range's result but add all range scores to metadata
                    if range_results:
                        repr_result = range_results[-1]
                        repr_result.metadata["cpcv_range_scores"] = range_scores
                        fold_results.append(repr_result)
                else:
                    # Standard single test range (PurgedKFold or WalkForward)
                    test_range = test_data
                    
                    fold_config = self._create_fold_config(test_range)
                    runner = BacktestRunner(fold_config)
                    
                    try:
                        result = runner.run_once(
                            self.strategy_factory,
                            params,
                            self.strategy_name,
                        )
                        score = self.scorer.score(result)
                    except Exception as e:
                        log.error(f"CV fold {fold_idx} failed: {e}")
                        result = create_dummy_result(params, str(e))
                        score = -float('inf')
                    
                    fold_scores.append(score)
                    fold_results.append(result)
            
            # Aggregate scores across all folds/splits
            if aggregate == "mean":
                agg_score = sum(fold_scores) / len(fold_scores) if fold_scores else 0
            elif aggregate == "median":
                sorted_scores = sorted(fold_scores)
                mid = len(sorted_scores) // 2
                agg_score = sorted_scores[mid] if sorted_scores else 0
            elif aggregate == "min":
                agg_score = min(fold_scores) if fold_scores else 0
            else:
                agg_score = sum(fold_scores) / len(fold_scores) if fold_scores else 0
            
            # Use last fold's result as representative (with aggregated metadata)
            if fold_results:
                final_result = fold_results[-1]
            else:
                final_result = create_dummy_result(params, "No fold results")
            
            final_result.metadata["cv_scores"] = fold_scores
            final_result.metadata["cv_aggregate"] = aggregate
            final_result.metadata["cv_score_std"] = (
                (sum((s - agg_score) ** 2 for s in fold_scores) / len(fold_scores)) ** 0.5
                if len(fold_scores) > 1 else 0.0
            )
            final_result.metadata["cv_score_mean"] = agg_score
            final_result.metadata["cv_n_folds"] = len(fold_scores)
            final_result.metadata["cv_score_min"] = min(fold_scores) if fold_scores else 0.0
            final_result.metadata["cv_score_max"] = max(fold_scores) if fold_scores else 0.0
            
            # Per-fold detail breakdown
            param_fold_details = []
            for fi, (fr, fs) in enumerate(zip(fold_results, fold_scores)):
                detail = {
                    "fold_idx": fi,
                    "score": fs,
                    "sharpe": fr.sharpe_ratio,
                    "total_return_pct": fr.total_return_pct,
                    "max_drawdown": fr.max_drawdown,
                    "total_trades": fr.total_trades,
                    "total_costs": fr.total_costs,
                }
                param_fold_details.append(detail)
            final_result.metadata["cv_fold_details"] = param_fold_details
            
            final_results.append(final_result)
            final_scores.append(agg_score)
            
            if (i + 1) % max(1, total // 10) == 0:
                log.info(f"CV Progress: {i + 1}/{total}")
        
        rankings = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
        
        elapsed = time.time() - start_time
        log.info(f"CV batch complete in {elapsed:.1f}s")
        
        # Warn about selection bias
        if total > 10:
            best_sharpe = max((r.sharpe_ratio for r in final_results if r.sharpe_ratio > -999), default=0)
            log.warning(selection_bias_warning(total, best_sharpe))
        
        return BatchResult(
            results=final_results,
            scores=final_scores,
            params_list=params_list,
            rankings=rankings,
            total_time_seconds=elapsed,
            trial_count=total,
            failed_count=0,  # CV handles failures per-fold
            selection_bias_report=None,  # CV results have different interpretation
            cv_fold_details=[
                r.metadata.get("cv_fold_details", []) for r in final_results
            ],
            cv_method=split_mode,
        )

    def run_with_cv_hyperband(
        self,
        param_space: ParameterGrid | SearchSpace | List[Dict[str, Any]],
        splitter: Optional[Union[PurgedKFold, WalkForwardSplit]] = None,
        aggregate: str = "mean",
        split_mode: str = "walk_forward",
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        train_pct: float = 0.6,
        halving_factor: int = 2,
        min_active: int = 2,
    ) -> BatchResult:
        """
        Successive-halving CV (Hyperband-style early termination).

        After each fold, drop the bottom ``1 / halving_factor`` of params
        by mean score so far.  Surviving combos progress to the next
        rung.  Floor at ``min_active`` params so the final rung always
        has something to compare.

        CPCV is not supported here — the per-split aggregation in CPCV
        already wraps multiple test ranges, which collides with the
        rung-based pruning logic.  Use :meth:`run_with_cv` for CPCV.

        Returns the same :class:`BatchResult` shape as :meth:`run_with_cv`,
        with each result's ``cv_pruned_at_fold`` metadata recording the
        rung at which a combo was dropped (``None`` if it survived).
        """
        start_time = time.time()

        if isinstance(param_space, (ParameterGrid, SearchSpace)):
            params_list = list(param_space)
        else:
            params_list = list(param_space)

        total = len(params_list)
        if total == 0:
            log.warning("Hyperband CV called with empty param space")
            return BatchResult(
                results=[], scores=[], params_list=[], rankings=[],
                total_time_seconds=0.0,
            )

        # Splitter setup mirrors run_with_cv (excluding CPCV)
        if splitter is None:
            start_ns = self.config.data.start_ts_ns
            end_ns = self.config.data.end_ts_ns
            if start_ns is None or end_ns is None:
                inferred = self._infer_tick_range()
                if inferred is None:
                    raise ValueError(
                        "Hyperband CV needs an explicit time range. "
                        "Set DataConfig.start_ts_ns/end_ts_ns or backfill "
                        "tick data so the bounds can be inferred."
                    )
                start_ns, end_ns = inferred
                log.info("Hyperband CV time range inferred: %d -> %d", start_ns, end_ns)
            total_duration = end_ns - start_ns

            if split_mode == "purged_kfold":
                splitter = PurgedKFold(
                    start_ns=start_ns, end_ns=end_ns,
                    n_splits=n_splits, embargo_pct=embargo_pct,
                )
            elif split_mode == "walk_forward":
                train_duration = int(total_duration * train_pct)
                test_duration = int(total_duration * (1 - train_pct) / n_splits)
                splitter = WalkForwardSplit(
                    start_ns=start_ns, end_ns=end_ns,
                    train_duration_ns=train_duration,
                    test_duration_ns=test_duration,
                    embargo_pct=embargo_pct,
                )
            else:
                raise ValueError(
                    f"Hyperband does not support split_mode={split_mode!r}; "
                    f"use 'walk_forward' or 'purged_kfold'."
                )

        folds = list(splitter.split())
        if any(isinstance(test_data, list) for _, test_data in folds):
            raise ValueError(
                "Hyperband does not support CPCV (list-valued test ranges). "
                "Use run_with_cv() instead."
            )

        n_folds = len(folds)
        log.info(
            "Hyperband CV: %d params x %d rungs (halving_factor=%d, min_active=%d)",
            total, n_folds, halving_factor, min_active,
        )

        # Per-param storage
        fold_scores: Dict[int, List[float]] = {i: [] for i in range(total)}
        fold_results: Dict[int, List[BacktestResult]] = {i: [] for i in range(total)}
        pruned_at_fold: Dict[int, Optional[int]] = {i: None for i in range(total)}

        active_indices: List[int] = list(range(total))
        n_evaluations = 0

        for fold_idx, (_train_ranges, test_data) in enumerate(folds):
            log.info(
                "  Rung %d/%d: %d active params",
                fold_idx + 1, n_folds, len(active_indices),
            )

            for orig_idx in active_indices:
                params = params_list[orig_idx]
                fold_config = self._create_fold_config(test_data)
                runner = BacktestRunner(fold_config)

                try:
                    result = runner.run_once(
                        self.strategy_factory, params, self.strategy_name,
                    )
                    score = self.scorer.score(result)
                except Exception as e:
                    log.error(
                        "Hyperband fold %d param idx %d failed: %s",
                        fold_idx, orig_idx, e,
                    )
                    result = create_dummy_result(params, str(e))
                    score = -float("inf")

                fold_scores[orig_idx].append(score)
                fold_results[orig_idx].append(result)
                n_evaluations += 1

            # Successive halving (skip after the last fold)
            if fold_idx < n_folds - 1 and len(active_indices) > min_active:
                ranked = sorted(
                    active_indices,
                    key=lambda i: sum(fold_scores[i]) / len(fold_scores[i]),
                    reverse=True,
                )
                keep_n = max(min_active, len(active_indices) // halving_factor)
                kept = ranked[:keep_n]
                pruned = ranked[keep_n:]
                for i in pruned:
                    pruned_at_fold[i] = fold_idx
                log.info(
                    "    pruned %d params, kept %d (cutoff mean_score=%.3f)",
                    len(pruned), len(kept),
                    sum(fold_scores[kept[-1]]) / len(fold_scores[kept[-1]])
                    if kept else float("nan"),
                )
                active_indices = kept

        # Build BatchResult with every param represented (pruned ones included)
        final_results: List[BacktestResult] = []
        final_scores: List[float] = []
        for i, params in enumerate(params_list):
            scores_i = fold_scores[i]
            results_i = fold_results[i]
            if not scores_i:
                final_results.append(create_dummy_result(params, "Pruned before any rung"))
                final_scores.append(-float("inf"))
                continue

            if aggregate == "mean":
                agg = sum(scores_i) / len(scores_i)
            elif aggregate == "median":
                sorted_s = sorted(scores_i)
                agg = sorted_s[len(sorted_s) // 2]
            elif aggregate == "min":
                agg = min(scores_i)
            else:
                agg = sum(scores_i) / len(scores_i)

            repr_result = results_i[-1]
            repr_result.metadata["cv_scores"] = scores_i
            repr_result.metadata["cv_aggregate"] = aggregate
            repr_result.metadata["cv_score_mean"] = agg
            repr_result.metadata["cv_n_folds"] = len(scores_i)
            repr_result.metadata["cv_pruned_at_fold"] = pruned_at_fold[i]
            repr_result.metadata["cv_score_min"] = min(scores_i)
            repr_result.metadata["cv_score_max"] = max(scores_i)
            repr_result.metadata["cv_score_std"] = (
                (sum((s - agg) ** 2 for s in scores_i) / len(scores_i)) ** 0.5
                if len(scores_i) > 1 else 0.0
            )
            repr_result.metadata["cv_fold_details"] = [
                {
                    "fold_idx": fi, "score": s,
                    "sharpe": r.sharpe_ratio,
                    "total_return_pct": r.total_return_pct,
                    "max_drawdown": r.max_drawdown,
                    "total_trades": r.total_trades,
                    "total_costs": r.total_costs,
                }
                for fi, (s, r) in enumerate(zip(scores_i, results_i))
            ]
            final_results.append(repr_result)
            final_scores.append(agg)

        rankings = sorted(
            range(len(final_scores)),
            key=lambda i: final_scores[i],
            reverse=True,
        )
        elapsed = time.time() - start_time
        full_grid_evals = total * n_folds
        log.info(
            "Hyperband CV complete in %.1fs: %d evaluations (vs %d full-grid, saved %.1f%%)",
            elapsed, n_evaluations, full_grid_evals,
            100.0 * (1.0 - n_evaluations / max(1, full_grid_evals)),
        )

        if total > 10:
            best_sharpe = max(
                (r.sharpe_ratio for r in final_results if r.sharpe_ratio > -999),
                default=0,
            )
            log.warning(selection_bias_warning(total, best_sharpe))

        return BatchResult(
            results=final_results,
            scores=final_scores,
            params_list=params_list,
            rankings=rankings,
            total_time_seconds=elapsed,
            trial_count=total,
            failed_count=0,
            selection_bias_report=None,
            cv_fold_details=[
                r.metadata.get("cv_fold_details", []) for r in final_results
            ],
            cv_method=f"{split_mode}+hyperband",
        )

    def run_cv_from_config(
        self,
        param_space: ParameterGrid | SearchSpace | List[Dict[str, Any]],
        aggregate: str = "mean",
    ) -> BatchResult:
        """
        Run CV backtest using cv_* settings from ``self.config.realism``.

        This is a convenience wrapper around :meth:`run_with_cv` that reads
        split_mode, n_splits, embargo_pct, expanding window settings, etc.
        from the ``RealismConfig`` attached to ``BacktestConfig.realism``.

        If ``realism.cv_enabled`` is False the call falls back to a plain
        :meth:`run` (no CV).

        Args:
            param_space: Parameter space to search.
            aggregate: How to aggregate fold scores ('mean', 'median', 'min').

        Returns:
            BatchResult with aggregated CV scores.
        """
        rc: RealismConfig = self.config.realism

        if not rc.cv_enabled:
            log.info("RealismConfig.cv_enabled=False → falling back to plain run()")
            return self.run(param_space)

        # Map config cv_method → split_mode expected by run_with_cv
        method_map = {
            "purged_kfold": "purged_kfold",
            "walk_forward": "walk_forward",
            "combinatorial_purged": "cpcv",
            "cpcv": "cpcv",
        }
        split_mode = method_map.get(rc.cv_method, rc.cv_method)

        log.info(
            f"run_cv_from_config: method={split_mode}, n_splits={rc.cv_n_splits}, "
            f"embargo_pct={rc.cv_embargo_pct}"
        )

        # Build a custom splitter if walk-forward with explicit durations
        splitter = None
        if split_mode == "walk_forward" and rc.cv_train_duration_ns > 0:
            start_ns = self.config.data.start_ts_ns or 0
            end_ns = self.config.data.end_ts_ns or int(365 * 24 * 60 * 60 * 1e9)
            splitter = WalkForwardSplit(
                start_ns=start_ns,
                end_ns=end_ns,
                train_duration_ns=rc.cv_train_duration_ns,
                test_duration_ns=rc.cv_test_duration_ns,
                embargo_pct=rc.cv_embargo_pct,
                expanding=rc.cv_expanding,
            )

        return self.run_with_cv(
            param_space=param_space,
            splitter=splitter,
            aggregate=aggregate,
            split_mode=split_mode,
            n_splits=rc.cv_n_splits,
            embargo_pct=rc.cv_embargo_pct,
        )

    def _create_fold_config(self, time_range: TimeRange) -> BacktestConfig:
        """Create a config for a specific time range fold."""
        # Copy the original config but update time range
        fold_data = DataConfig(
            tick_data_dir=self.config.data.tick_data_dir,
            symbols=self.config.data.symbols,
            start_ts_ns=time_range.start_ns,
            end_ts_ns=time_range.end_ns,
            bar_type=self.config.data.bar_type,
            timeframe=self.config.data.timeframe,
            tick_threshold=self.config.data.tick_threshold,
            volume_threshold=self.config.data.volume_threshold,
            dollar_threshold=self.config.data.dollar_threshold,
        )
        
        return BacktestConfig(
            data=fold_data,
            initial_capital=self.config.initial_capital,
            margin_requirement=self.config.margin_requirement,
            taker_fee_bps=self.config.taker_fee_bps,
            maker_fee_bps=self.config.maker_fee_bps,
            slippage_bps=self.config.slippage_bps,
            spread_bps=self.config.spread_bps,
            max_position_size=self.config.max_position_size,
            max_position_notional=self.config.max_position_notional,
            max_daily_loss=self.config.max_daily_loss,
            max_drawdown=self.config.max_drawdown,
            use_bar_close=self.config.use_bar_close,
            random_seed=self.config.random_seed,
            bar_store_maxlen=self.config.bar_store_maxlen,
            # Execution realism fields
            latency_ns=getattr(self.config, 'latency_ns', 0),
            latency_jitter_ns=getattr(self.config, 'latency_jitter_ns', 0),
            enable_partial_fills=getattr(self.config, 'enable_partial_fills', False),
            liquidity_scale=getattr(self.config, 'liquidity_scale', 1.0),
            min_fill_ratio=getattr(self.config, 'min_fill_ratio', 0.0),
            close_positions_at_end=getattr(self.config, 'close_positions_at_end', True),
            # Leverage settings — previously dropped, which silently
            # demoted CV folds to spot mode while the single backtest
            # ran in margin mode (CV scores then diverged from the
            # full-period score for the same params).
            leverage_mode=getattr(self.config, "leverage_mode", "spot"),
            leverage=getattr(self.config, "leverage", 1),
            maintenance_margin_ratio=getattr(
                self.config, "maintenance_margin_ratio", 0.5,
            ),
            # Tick-level exit + per-trade exit rules — same reason: the
            # full backtest had these on; CV folds were running with
            # the dataclass defaults, which let trailing stops and
            # max-holding behave differently between the two paths.
            enable_tick_exit=getattr(self.config, "enable_tick_exit", True),
            tp_pct=getattr(self.config, "tp_pct", None),
            sl_pct=getattr(self.config, "sl_pct", None),
            trailing_stop_pct=getattr(self.config, "trailing_stop_pct", None),
            max_holding_bars=getattr(self.config, "max_holding_bars", None),
            # Realism config passthrough (Part A/C)
            realism=self.config.realism,
        )

    # ------------------------------------------------------------------
    # Tick-range inference (Bug 4 — silent default removal)
    # ------------------------------------------------------------------
    def _infer_tick_range(self) -> Optional[Tuple[int, int]]:
        """Scan ``data.tick_data_dir`` for partitioned daily files and
        return ``(start_ns, end_ns)`` if any are found.

        Tick files are named ``YYYY-MM-DD.csv`` per symbol directory.
        We use the earliest start-of-day across symbols and the latest
        end-of-day — that bounds the entire data the backtest engine
        could possibly load.  Returning None forces the caller to
        fail loud (see ``run_with_cv``).
        """
        import os
        import datetime as _dt
        try:
            tick_dir = self.config.data.tick_data_dir
            symbols = self.config.data.symbols or []
        except AttributeError:
            return None
        if not tick_dir or not symbols:
            return None
        if not os.path.isdir(tick_dir):
            return None

        all_dates: List[_dt.date] = []
        for sym in symbols:
            sym_dir = os.path.join(tick_dir, sym)
            if not os.path.isdir(sym_dir):
                continue
            for fname in os.listdir(sym_dir):
                stem, _, ext = fname.partition(".")
                if ext.lower() not in ("csv", "csv.gz", "parquet"):
                    continue
                try:
                    all_dates.append(_dt.date.fromisoformat(stem))
                except ValueError:
                    continue
        if not all_dates:
            return None
        all_dates.sort()
        first = _dt.datetime.combine(all_dates[0], _dt.time.min,
                                     tzinfo=_dt.timezone.utc)
        last = _dt.datetime.combine(all_dates[-1], _dt.time.max,
                                    tzinfo=_dt.timezone.utc)
        return int(first.timestamp() * 1e9), int(last.timestamp() * 1e9)

    @property
    def run_count(self) -> int:
        """Total backtests run."""
        return self._run_count
