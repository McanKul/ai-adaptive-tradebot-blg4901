"""
Backtest/scoring/selector.py
============================
Selection utilities for choosing best parameters/strategies.

Design decisions:
- Selector ranks and filters results
- Supports tie-breakers (drawdown, turnover)
- Can apply additional filters (minimum metrics)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging

from Interfaces.metrics_interface import BacktestResult
from Backtest.scoring.batch import BatchResult

log = logging.getLogger(__name__)


@dataclass
class SelectionCriteria:
    """Criteria for selection and tie-breaking."""
    # Minimum requirements
    min_trades: int = 10
    min_sharpe: float = 0.0
    max_drawdown: float = 1.0  # 100%
    min_win_rate: float = 0.0
    
    # CV-aware selection (Part C)
    cv_stability_weight: float = 0.0  # Penalty multiplier for cv_score_std
    max_cv_std: float = float("inf")  # Hard filter: reject if cv_score_std >
    
    # Tie-breakers (in order of priority)
    # Each is (metric_name, prefer_higher)
    tie_breakers: List[Tuple[str, bool]] = None
    
    def __post_init__(self):
        if self.tie_breakers is None:
            self.tie_breakers = [
                ("max_drawdown", False),  # Prefer lower drawdown
                ("turnover", False),  # Prefer lower turnover
                ("total_costs", False),  # Prefer lower costs
            ]


class Selector:
    """
    Selector for choosing best results from batch backtest.
    
    Usage:
        selector = Selector()
        top_results = selector.select_top_k(batch_result, k=5)
        
        # With custom criteria
        criteria = SelectionCriteria(min_sharpe=0.5, max_drawdown=0.2)
        selector = Selector(criteria)
        filtered = selector.select_filtered(batch_result)
    """
    
    def __init__(self, criteria: Optional[SelectionCriteria] = None):
        """
        Initialize Selector.
        
        Args:
            criteria: SelectionCriteria for filtering and tie-breaking
        """
        self.criteria = criteria or SelectionCriteria()
    
    def select_top_k(
        self,
        batch_result: BatchResult,
        k: int,
        apply_filters: bool = True,
    ) -> List[Tuple[BacktestResult, float, Dict[str, Any]]]:
        """
        Select top k results.
        
        Args:
            batch_result: BatchResult from batch backtest
            k: Number of results to return
            apply_filters: Whether to apply minimum criteria filters
            
        Returns:
            List of (result, score, params) tuples, sorted by score
        """
        candidates = batch_result.get_ranked_results()
        
        if apply_filters:
            candidates = [
                (r, s, p) for r, s, p in candidates
                if self._passes_filters(r)
            ]
        
        # Apply tie-breaking for equal scores
        candidates = self._apply_tie_breakers(candidates)
        
        return candidates[:k]
    
    def select_filtered(
        self,
        batch_result: BatchResult,
    ) -> List[Tuple[BacktestResult, float, Dict[str, Any]]]:
        """
        Select all results that pass filters.
        
        Args:
            batch_result: BatchResult from batch backtest
            
        Returns:
            Filtered and sorted list of (result, score, params) tuples
        """
        candidates = batch_result.get_ranked_results()
        filtered = [
            (r, s, p) for r, s, p in candidates
            if self._passes_filters(r)
        ]
        return self._apply_tie_breakers(filtered)
    
    def select_best(
        self,
        batch_result: BatchResult,
    ) -> Optional[Tuple[BacktestResult, float, Dict[str, Any]]]:
        """
        Select the single best result.
        
        Args:
            batch_result: BatchResult from batch backtest
            
        Returns:
            (result, score, params) tuple or None if no results pass filters
        """
        top = self.select_top_k(batch_result, k=1, apply_filters=True)
        return top[0] if top else None
    
    def _passes_filters(self, result: BacktestResult) -> bool:
        """Check if result passes all filter criteria."""
        c = self.criteria
        
        if result.total_trades < c.min_trades:
            return False
        
        if result.sharpe_ratio < c.min_sharpe:
            return False
        
        if result.max_drawdown > c.max_drawdown:
            return False
        
        if result.win_rate < c.min_win_rate:
            return False
        
        # CV stability hard filter
        cv_std = result.metadata.get("cv_score_std")
        if cv_std is not None and cv_std > c.max_cv_std:
            return False
        
        return True
    
    def _apply_tie_breakers(
        self,
        candidates: List[Tuple[BacktestResult, float, Dict[str, Any]]],
    ) -> List[Tuple[BacktestResult, float, Dict[str, Any]]]:
        """
        Sort candidates using tie-breakers for equal scores.
        
        When ``cv_stability_weight > 0`` the effective score is adjusted:
            effective_score = score - cv_stability_weight * cv_score_std
        
        This penalises parameter sets whose performance varies widely
        across CV folds, promoting robust configurations.
        """
        w = self.criteria.cv_stability_weight
        
        def sort_key(item: Tuple[BacktestResult, float, Dict[str, Any]]) -> tuple:
            result, score, params = item
            
            # Adjust score by CV-stability penalty if available
            effective_score = score
            if w > 0:
                cv_std = result.metadata.get("cv_score_std", 0.0)
                if cv_std is not None:
                    effective_score = score - w * cv_std
            
            # Primary key: effective score (descending, so negate)
            key = [-effective_score]
            
            # Tie-breaker keys
            for metric_name, prefer_higher in self.criteria.tie_breakers:
                value = getattr(result, metric_name, 0.0)
                if prefer_higher:
                    key.append(-value)  # Negate so higher values come first
                else:
                    key.append(value)  # Lower values come first
            
            return tuple(key)
        
        return sorted(candidates, key=sort_key)
    
    def rank_by_metric(
        self,
        batch_result: BatchResult,
        metric: str,
        ascending: bool = False,
    ) -> List[Tuple[BacktestResult, float, Dict[str, Any]]]:
        """
        Rank results by a specific metric.
        
        Args:
            batch_result: BatchResult from batch backtest
            metric: Metric name to rank by
            ascending: If True, lower values rank higher
            
        Returns:
            Sorted list of (result, score, params) tuples
        """
        candidates = batch_result.get_ranked_results()
        
        def get_metric(item):
            result, _, _ = item
            return getattr(result, metric, 0.0)
        
        return sorted(candidates, key=get_metric, reverse=not ascending)
    
    def pareto_frontier(
        self,
        batch_result: BatchResult,
        objectives: List[Tuple[str, bool]],  # (metric_name, maximize)
    ) -> List[Tuple[BacktestResult, float, Dict[str, Any]]]:
        """
        Find the Pareto frontier of non-dominated solutions.
        
        Args:
            batch_result: BatchResult from batch backtest
            objectives: List of (metric_name, maximize) tuples
            
        Returns:
            List of Pareto-optimal (result, score, params) tuples
        """
        candidates = batch_result.get_ranked_results()
        
        def dominates(a: BacktestResult, b: BacktestResult) -> bool:
            """Check if a dominates b (a >= b in all objectives, a > b in at least one)."""
            dominated = False
            strictly_better = False
            
            for metric, maximize in objectives:
                val_a = getattr(a, metric, 0.0)
                val_b = getattr(b, metric, 0.0)
                
                if maximize:
                    if val_a < val_b:
                        return False
                    elif val_a > val_b:
                        strictly_better = True
                else:
                    if val_a > val_b:
                        return False
                    elif val_a < val_b:
                        strictly_better = True
            
            return strictly_better
        
        # Find non-dominated solutions
        frontier = []
        for i, (result_i, score_i, params_i) in enumerate(candidates):
            is_dominated = False
            for j, (result_j, _, _) in enumerate(candidates):
                if i != j and dominates(result_j, result_i):
                    is_dominated = True
                    break
            
            if not is_dominated:
                frontier.append((result_i, score_i, params_i))
        
        return frontier


def select_top_k(
    batch_result: BatchResult,
    k: int,
    min_trades: int = 10,
    max_drawdown: float = 1.0,
) -> List[Tuple[BacktestResult, float, Dict[str, Any]]]:
    """
    Convenience function for simple top-k selection.
    
    Args:
        batch_result: BatchResult from batch backtest
        k: Number of results to return
        min_trades: Minimum trades filter
        max_drawdown: Maximum drawdown filter
        
    Returns:
        Top k results as (result, score, params) tuples
    """
    criteria = SelectionCriteria(
        min_trades=min_trades,
        max_drawdown=max_drawdown,
    )
    selector = Selector(criteria)
    return selector.select_top_k(batch_result, k)
