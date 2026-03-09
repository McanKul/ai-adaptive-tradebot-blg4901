"""
Backtest/scoring/scorer.py
==========================
Scoring functions for evaluating backtest results.

Design decisions:
- Scorer computes a single score from BacktestResult
- Supports multi-objective scoring (risk-adjusted returns)
- Configurable weights for different components
- Default formula documented and tunable

DEFAULT SCORING FORMULA:
========================
score = sharpe_ratio 
        - 0.5 * max_drawdown 
        - 0.1 * turnover 
        - 0.2 * (total_costs / initial_capital)

Components:
- sharpe_ratio: Risk-adjusted return (higher is better)
- max_drawdown: Maximum peak-to-trough decline (penalized)
- turnover: Trading activity (penalized for overtrading)
- costs: Fees and slippage relative to capital (penalized)

This formula prefers:
- Higher Sharpe ratios
- Lower drawdowns
- Moderate trading frequency
- Lower transaction costs
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List
import math
import logging

from Interfaces.metrics_interface import BacktestResult

log = logging.getLogger(__name__)


@dataclass
class ScorerWeights:
    """Weights for scoring components."""
    sharpe: float = 1.0
    max_drawdown: float = 0.5
    turnover: float = 0.1
    costs: float = 0.2
    win_rate: float = 0.0  # Optional: reward high win rate
    profit_factor: float = 0.0  # Optional: reward high profit factor
    calmar: float = 0.0  # Optional: use Calmar ratio


class Scorer:
    """
    Scorer for evaluating backtest results.
    
    Computes a single score from multiple metrics.
    Higher score = better strategy.
    
    Usage:
        scorer = Scorer()
        score = scorer.score(result)
        
        # Custom weights
        scorer = Scorer(ScorerWeights(sharpe=1.5, max_drawdown=0.8))
    """
    
    def __init__(
        self,
        weights: Optional[ScorerWeights] = None,
        min_trades: int = 10,  # Minimum trades for valid score
        min_sharpe: float = -999.0,  # Floor for Sharpe contribution
        max_sharpe: float = 10.0,  # Cap for Sharpe contribution
    ):
        """
        Initialize Scorer.
        
        Args:
            weights: ScorerWeights for component weights
            min_trades: Minimum trades required (else penalized)
            min_sharpe: Minimum Sharpe to use (floor)
            max_sharpe: Maximum Sharpe to use (cap)
        """
        self.weights = weights or ScorerWeights()
        self.min_trades = min_trades
        self.min_sharpe = min_sharpe
        self.max_sharpe = max_sharpe
    
    def score(self, result: BacktestResult) -> float:
        """
        Compute score for a backtest result.
        
        Args:
            result: BacktestResult to score
            
        Returns:
            Score (higher is better)
        """
        # Check minimum trades
        if result.total_trades < self.min_trades:
            log.debug(
                f"Insufficient trades ({result.total_trades} < {self.min_trades}), "
                f"applying penalty"
            )
            return -1000.0 + result.total_trades  # Heavily penalized but differentiable
        
        # Clamp Sharpe
        sharpe = max(self.min_sharpe, min(self.max_sharpe, result.sharpe_ratio))
        if math.isnan(sharpe) or math.isinf(sharpe):
            sharpe = 0.0
        
        # Compute components
        sharpe_component = self.weights.sharpe * sharpe
        
        drawdown_component = self.weights.max_drawdown * result.max_drawdown
        
        turnover_component = self.weights.turnover * min(result.turnover, 100.0)  # Cap turnover
        
        if result.initial_capital > 0:
            cost_ratio = result.total_costs / result.initial_capital
        else:
            cost_ratio = 0.0
        costs_component = self.weights.costs * cost_ratio
        
        # Optional components
        win_rate_component = self.weights.win_rate * result.win_rate
        
        pf = result.profit_factor if result.profit_factor < float('inf') else 10.0
        profit_factor_component = self.weights.profit_factor * min(pf, 10.0)
        
        calmar = result.calmar_ratio if not math.isnan(result.calmar_ratio) else 0.0
        calmar_component = self.weights.calmar * min(max(calmar, -10), 10)
        
        # Final score
        score = (
            sharpe_component
            - drawdown_component
            - turnover_component
            - costs_component
            + win_rate_component
            + profit_factor_component
            + calmar_component
        )
        
        return score
    
    def score_breakdown(self, result: BacktestResult) -> Dict[str, float]:
        """
        Get detailed score breakdown.
        
        Args:
            result: BacktestResult to analyze
            
        Returns:
            Dict with component scores
        """
        sharpe = max(self.min_sharpe, min(self.max_sharpe, result.sharpe_ratio))
        if math.isnan(sharpe) or math.isinf(sharpe):
            sharpe = 0.0
        
        if result.initial_capital > 0:
            cost_ratio = result.total_costs / result.initial_capital
        else:
            cost_ratio = 0.0
        
        pf = result.profit_factor if result.profit_factor < float('inf') else 10.0
        calmar = result.calmar_ratio if not math.isnan(result.calmar_ratio) else 0.0
        
        return {
            "sharpe_raw": result.sharpe_ratio,
            "sharpe_clamped": sharpe,
            "sharpe_component": self.weights.sharpe * sharpe,
            "drawdown_component": -self.weights.max_drawdown * result.max_drawdown,
            "turnover_component": -self.weights.turnover * min(result.turnover, 100.0),
            "costs_component": -self.weights.costs * cost_ratio,
            "win_rate_component": self.weights.win_rate * result.win_rate,
            "profit_factor_component": self.weights.profit_factor * min(pf, 10.0),
            "calmar_component": self.weights.calmar * min(max(calmar, -10), 10),
            "total_score": self.score(result),
            "trade_count": result.total_trades,
            "min_trades_met": result.total_trades >= self.min_trades,
        }


def create_scorer(
    sharpe_weight: float = 1.0,
    drawdown_weight: float = 0.5,
    turnover_weight: float = 0.1,
    cost_weight: float = 0.2,
    min_trades: int = 10,
) -> Scorer:
    """
    Factory function for creating a Scorer.
    
    Args:
        sharpe_weight: Weight for Sharpe ratio (positive = reward)
        drawdown_weight: Weight for max drawdown (positive = penalty)
        turnover_weight: Weight for turnover (positive = penalty)
        cost_weight: Weight for costs (positive = penalty)
        min_trades: Minimum trades required
        
    Returns:
        Configured Scorer
    """
    return Scorer(
        weights=ScorerWeights(
            sharpe=sharpe_weight,
            max_drawdown=drawdown_weight,
            turnover=turnover_weight,
            costs=cost_weight,
        ),
        min_trades=min_trades,
    )


# =============================================================================
# TRIAL-AWARE REPORTING (AFML Selection Bias Discipline)
# =============================================================================

def compute_deflated_sharpe(
    sharpe: float,
    n_trials: int,
    variance: float = 1.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    t_years: float = 1.0,
) -> float:
    """
    Compute Deflated Sharpe Ratio (DSR) accounting for multiple trials.
    
    AFML Insight: When optimizing over many parameter combinations, the best
    Sharpe ratio is inflated by selection bias. DSR adjusts for this.
    
    Simplified formula based on Lopez de Prado's work:
    DSR ≈ Sharpe - sqrt(Variance(Sharpe)) * E[max(Z_1, ..., Z_n)]
    
    Where E[max(Z)] ≈ sqrt(2 * log(n)) for large n (Gumbel approximation).
    
    Args:
        sharpe: Observed Sharpe ratio
        n_trials: Number of parameter combinations tested
        variance: Variance of returns (default 1.0 for standardized)
        skewness: Skewness of returns
        kurtosis: Kurtosis of returns (3.0 for normal)
        t_years: Time period in years
        
    Returns:
        Deflated Sharpe Ratio
    """
    if n_trials <= 1:
        return sharpe
    
    # Expected maximum of n standard normal variables
    # E[max(Z_1, ..., Z_n)] ≈ sqrt(2 * ln(n)) for large n
    expected_max = math.sqrt(2 * math.log(n_trials))
    
    # Variance of Sharpe ratio estimate
    # Var(SR) ≈ (1 + 0.5 * SR^2 - skew * SR + (kurtosis - 3) * SR^2 / 4) / T
    sr_variance = (1 + 0.5 * sharpe**2 - skewness * sharpe + 
                   (kurtosis - 3) * sharpe**2 / 4) / max(t_years, 0.1)
    sr_std = math.sqrt(sr_variance)
    
    # Deflated Sharpe
    dsr = sharpe - sr_std * expected_max
    
    return dsr


def selection_bias_warning(n_trials: int, best_sharpe: float) -> str:
    """
    Generate a warning message about selection bias.
    
    AFML Discipline: Always report the number of trials alongside
    the best result to contextualize the selection bias risk.
    
    Args:
        n_trials: Number of parameter combinations tested
        best_sharpe: Best observed Sharpe ratio
        
    Returns:
        Warning message string
    """
    if n_trials <= 5:
        risk = "LOW"
        msg = "Few trials tested; selection bias minimal."
    elif n_trials <= 20:
        risk = "MODERATE"
        msg = "Moderate number of trials; some selection bias possible."
    elif n_trials <= 100:
        risk = "HIGH"
        msg = "Many trials tested; significant selection bias likely."
    else:
        risk = "VERY HIGH"
        msg = "Very many trials tested; reported Sharpe is likely inflated."
    
    dsr = compute_deflated_sharpe(best_sharpe, n_trials)
    
    return (
        f"SELECTION BIAS WARNING [{risk}]\n"
        f"  Trials tested: {n_trials}\n"
        f"  Best Sharpe: {best_sharpe:.3f}\n"
        f"  Deflated Sharpe (DSR): {dsr:.3f}\n"
        f"  {msg}"
    )


class TrialAwareScorer(Scorer):
    """
    Scorer that tracks number of trials for selection bias reporting.
    
    AFML Discipline: The more parameter combinations you test, the more
    likely your best result is due to luck (selection bias). This scorer
    tracks trial counts and can compute deflated metrics.
    
    Usage:
        scorer = TrialAwareScorer()
        for params in param_grid:
            result = run_backtest(params)
            score = scorer.score(result)
        
        # After all trials
        report = scorer.get_trial_report()
    """
    
    def __init__(
        self,
        weights: Optional[ScorerWeights] = None,
        min_trades: int = 10,
        min_sharpe: float = -999.0,
        max_sharpe: float = 10.0,
    ):
        super().__init__(weights, min_trades, min_sharpe, max_sharpe)
        self._trial_count = 0
        self._scores: List[float] = []
        self._sharpes: List[float] = []
        self._best_score = -float('inf')
        self._best_sharpe = -float('inf')
    
    @property
    def trial_count(self) -> int:
        """Number of trials scored so far."""
        return self._trial_count
    
    @property
    def best_sharpe(self) -> float:
        """Best Sharpe ratio seen across all trials."""
        return self._best_sharpe
    
    @property
    def best_score(self) -> float:
        """Best composite score seen across all trials."""
        return self._best_score
    
    def score(self, result: BacktestResult) -> float:
        """Score result and track trial statistics."""
        score = super().score(result)
        
        self._trial_count += 1
        self._scores.append(score)
        
        sharpe = result.sharpe_ratio
        if not math.isnan(sharpe) and not math.isinf(sharpe):
            self._sharpes.append(sharpe)
            self._best_sharpe = max(self._best_sharpe, sharpe)
        
        self._best_score = max(self._best_score, score)
        
        return score
    
    def get_trial_report(self) -> Dict[str, Any]:
        """
        Get report on trial statistics including selection bias metrics.
        
        Returns:
            Dict with trial count, best scores, and deflated metrics
        """
        if not self._sharpes:
            return {
                "trial_count": self._trial_count,
                "best_score": self._best_score,
                "best_sharpe": None,
                "deflated_sharpe": None,
                "selection_bias_warning": "No valid Sharpe ratios computed",
            }
        
        dsr = compute_deflated_sharpe(self._best_sharpe, self._trial_count)
        
        return {
            "trial_count": self._trial_count,
            "best_score": self._best_score,
            "best_sharpe": self._best_sharpe,
            "deflated_sharpe": dsr,
            "sharpe_inflation": self._best_sharpe - dsr,
            "avg_sharpe": sum(self._sharpes) / len(self._sharpes),
            "selection_bias_warning": selection_bias_warning(
                self._trial_count, self._best_sharpe
            ),
        }
    
    def reset_trials(self) -> None:
        """Reset trial tracking for a new sweep."""
        self._trial_count = 0
        self._scores.clear()
        self._sharpes.clear()
        self._best_score = -float('inf')
        self._best_sharpe = -float('inf')


class MultiObjectiveScorer:
    """
    Multi-objective scorer that returns multiple scores.
    
    Useful for Pareto optimization where you don't want to
    combine objectives into a single number.
    """
    
    def __init__(self):
        """Initialize MultiObjectiveScorer."""
        pass
    
    def score(self, result: BacktestResult) -> Dict[str, float]:
        """
        Compute multiple objective scores.
        
        Args:
            result: BacktestResult to score
            
        Returns:
            Dict with objective scores (to maximize)
        """
        sharpe = result.sharpe_ratio
        if math.isnan(sharpe) or math.isinf(sharpe):
            sharpe = 0.0
        
        return {
            "return": result.total_return_pct,
            "sharpe": sharpe,
            "neg_drawdown": -result.max_drawdown,  # Negative so higher is better
            "neg_turnover": -result.turnover,
            "win_rate": result.win_rate,
        }
    
    def is_pareto_dominant(
        self,
        scores_a: Dict[str, float],
        scores_b: Dict[str, float]
    ) -> bool:
        """
        Check if scores_a Pareto dominates scores_b.
        
        A dominates B if A is >= B in all objectives and > B in at least one.
        
        Args:
            scores_a: First score dict
            scores_b: Second score dict
            
        Returns:
            True if A dominates B
        """
        dominated = False
        strictly_better = False
        
        for key in scores_a:
            if key not in scores_b:
                continue
            
            if scores_a[key] < scores_b[key]:
                return False  # A is worse in some objective
            elif scores_a[key] > scores_b[key]:
                strictly_better = True
        
        return strictly_better
