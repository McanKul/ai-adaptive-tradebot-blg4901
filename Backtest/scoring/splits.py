"""
Backtest/scoring/splits.py
==========================
Time series cross-validation splits for leakage-safe evaluation.

Implements:
- PurgedKFold: K-fold CV with purging to prevent lookahead bias
- Embargo: Gap between train and test to prevent information leakage
- WalkForwardSplit: Walk-forward optimization with sliding/expanding windows
- CombinatorialPurgedCV: Combinatorial purged cross-validation (CPCV-inspired)

AFML-INSPIRED (but AI-free):
============================
These techniques come from Marcos López de Prado's work on preventing
data leakage in financial ML. However, we use them here purely for
fair parameter evaluation, NOT for ML training.

The key insight: when optimizing strategy parameters on historical data,
there's a risk of overfitting if you don't properly separate train/test
periods. Purging and embargo help ensure that:
1. No future information leaks into "training" (parameter selection)
2. Results reflect realistic out-of-sample performance

IMPORTANT: All splitters return consistent structure:
- train_ranges: List[TimeRange] - list of training time ranges
- test_range: TimeRange - single test time range

Usage:
    # PurgedKFold
    splitter = PurgedKFold(start_ns, end_ns, n_splits=5, embargo_pct=0.01)
    for train_ranges, test_range in splitter.split():
        # Run backtest on test_range
    
    # WalkForwardSplit
    splitter = WalkForwardSplit(
        start_ns, end_ns,
        train_duration_ns=30*DAY_NS, test_duration_ns=7*DAY_NS,
        embargo_pct=0.01
    )
    for train_ranges, test_range in splitter.split():
        # Run backtest on test_range
    
    # CombinatorialPurgedCV
    splitter = CombinatorialPurgedCV(start_ns, end_ns, n_splits=6, n_test_splits=2)
    for train_ranges, test_ranges in splitter.split():
        # test_ranges is a list (multiple test folds combined)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Iterator, Tuple, Optional, Dict, Any, TYPE_CHECKING
import logging
from statistics import mean, stdev
from itertools import combinations

if TYPE_CHECKING:
    from Backtest.scoring.batch import BatchBacktest
    from Interfaces.metrics_interface import BacktestResult

log = logging.getLogger(__name__)

# Time constants
SECOND_NS = 1_000_000_000
MINUTE_NS = 60 * SECOND_NS
HOUR_NS = 60 * MINUTE_NS
DAY_NS = 24 * HOUR_NS


@dataclass(frozen=True)
class TimeRange:
    """A time range with start and end timestamps (nanoseconds)."""
    start_ns: int
    end_ns: int
    
    @property
    def duration_ns(self) -> int:
        """Duration in nanoseconds."""
        return self.end_ns - self.start_ns
    
    @property
    def duration_days(self) -> float:
        """Duration in days."""
        return self.duration_ns / (86400 * 1e9)
    
    def overlaps(self, other: "TimeRange") -> bool:
        """Check if this range overlaps with another."""
        return self.start_ns < other.end_ns and other.start_ns < self.end_ns
    
    def contains(self, timestamp_ns: int) -> bool:
        """Check if timestamp is within this range."""
        return self.start_ns <= timestamp_ns < self.end_ns


class PurgedKFold:
    """
    Purged K-Fold cross-validation for time series.
    
    Divides the time range into K folds. Each fold is used as the test set
    once, with the remaining folds (minus purge/embargo) as training.
    
    Purging: Removes training samples that could leak information into test.
    Embargo: Adds a gap after the test period before using as training.
    
    Note: In backtest context, we primarily use test folds for evaluation.
    The "training" concept here means "parameter selection period" - not ML training.
    
    Example with 5 folds and embargo:
    
    Fold 0 as test:
    [TEST][embargo][----TRAIN----|----TRAIN----|----TRAIN----|----TRAIN----]
    
    Fold 2 as test:
    [TRAIN][TRAIN][embargo][TEST][embargo][----TRAIN----|----TRAIN----]
    """
    
    def __init__(
        self,
        start_ns: int,
        end_ns: int,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.0,
    ):
        """
        Initialize PurgedKFold.
        
        Args:
            start_ns: Start timestamp in nanoseconds
            end_ns: End timestamp in nanoseconds
            n_splits: Number of folds
            embargo_pct: Fraction of total time to use as embargo gap
            purge_pct: Fraction of test period to purge from adjacent training
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not 0 <= embargo_pct < 1:
            raise ValueError("embargo_pct must be in [0, 1)")
        if not 0 <= purge_pct < 1:
            raise ValueError("purge_pct must be in [0, 1)")
        
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct
        
        self._total_duration = end_ns - start_ns
        self._fold_duration = self._total_duration // n_splits
        self._embargo_ns = int(self._total_duration * embargo_pct)
        
        log.debug(
            f"PurgedKFold: {n_splits} folds, "
            f"fold_duration={self._fold_duration / 1e9:.0f}s, "
            f"embargo={self._embargo_ns / 1e9:.0f}s"
        )
    
    def split(self) -> Iterator[Tuple[List[TimeRange], TimeRange]]:
        """
        Generate train/test splits.
        
        Yields:
            Tuple of (train_ranges, test_range) for each fold
        """
        for fold_idx in range(self.n_splits):
            test_range = self._get_fold_range(fold_idx)
            train_ranges = self._get_train_ranges(fold_idx, test_range)
            yield train_ranges, test_range
    
    def _get_fold_range(self, fold_idx: int) -> TimeRange:
        """Get the time range for a specific fold."""
        start = self.start_ns + fold_idx * self._fold_duration
        end = start + self._fold_duration
        
        # Last fold goes to end
        if fold_idx == self.n_splits - 1:
            end = self.end_ns
        
        return TimeRange(start_ns=start, end_ns=end)
    
    def _get_train_ranges(
        self,
        test_fold_idx: int,
        test_range: TimeRange
    ) -> List[TimeRange]:
        """
        Get training time ranges for a given test fold.
        
        Excludes test fold and applies embargo.
        """
        train_ranges = []
        
        # Compute purge distance (remove training samples near test)
        purge_ns = int(test_range.duration_ns * self.purge_pct)
        
        for fold_idx in range(self.n_splits):
            if fold_idx == test_fold_idx:
                continue
            
            fold_range = self._get_fold_range(fold_idx)
            
            # Apply embargo: if this fold is after test, add gap
            if fold_range.start_ns >= test_range.end_ns:
                # Fold is after test - apply embargo
                adjusted_start = fold_range.start_ns + self._embargo_ns
                if adjusted_start < fold_range.end_ns:
                    train_ranges.append(TimeRange(
                        start_ns=adjusted_start,
                        end_ns=fold_range.end_ns
                    ))
            
            # Apply purge: if fold is adjacent to test, remove overlap
            elif fold_range.end_ns <= test_range.start_ns:
                # Fold is before test - apply purge to end
                adjusted_end = fold_range.end_ns - purge_ns
                if adjusted_end > fold_range.start_ns:
                    train_ranges.append(TimeRange(
                        start_ns=fold_range.start_ns,
                        end_ns=adjusted_end
                    ))
            else:
                # Fold doesn't need adjustment
                train_ranges.append(fold_range)
        
        return train_ranges
    
    def get_test_ranges(self) -> List[TimeRange]:
        """Get all test fold ranges."""
        return [self._get_fold_range(i) for i in range(self.n_splits)]
    
    def info(self) -> dict:
        """Get information about the split configuration."""
        return {
            "n_splits": self.n_splits,
            "total_duration_days": self._total_duration / (86400 * 1e9),
            "fold_duration_days": self._fold_duration / (86400 * 1e9),
            "embargo_days": self._embargo_ns / (86400 * 1e9),
            "embargo_pct": self.embargo_pct,
            "purge_pct": self.purge_pct,
        }
    
    def cross_validate(
        self,
        batch: "BatchBacktest",
        params: Dict[str, Any],
        metric: str = "sharpe_ratio",
    ) -> "CVResult":
        """
        Run cross-validation with the given parameters.
        
        Args:
            batch: BatchBacktest instance to use for running
            params: Strategy parameters to evaluate
            metric: Metric to extract from results
            
        Returns:
            CVResult with aggregated scores across folds
        """
        fold_scores = []
        fold_results = []
        
        for fold_idx, (train_ranges, test_range) in enumerate(self.split()):
            # Run backtest on test range
            result = batch.run_single(
                params,
                start_ns=test_range.start_ns,
                end_ns=test_range.end_ns,
            )
            
            fold_results.append(result)
            
            # Extract metric
            score = getattr(result, metric, None)
            if score is not None and not (isinstance(score, float) and (score != score)):  # nan check
                fold_scores.append(score)
        
        return CVResult(
            params=params,
            metric=metric,
            fold_scores=fold_scores,
            fold_results=fold_results,
            n_splits=self.n_splits,
        )


@dataclass
class CVResult:
    """
    Result of cross-validation.
    
    Contains scores from each fold and aggregated statistics.
    """
    params: Dict[str, Any]
    metric: str
    fold_scores: List[float]
    fold_results: List["BacktestResult"]
    n_splits: int
    
    @property
    def mean_score(self) -> float:
        """Mean score across folds."""
        return mean(self.fold_scores) if self.fold_scores else 0.0
    
    @property
    def std_score(self) -> float:
        """Standard deviation of scores."""
        if len(self.fold_scores) < 2:
            return 0.0
        return stdev(self.fold_scores)
    
    @property
    def min_score(self) -> float:
        """Minimum score across folds."""
        return min(self.fold_scores) if self.fold_scores else 0.0
    
    @property
    def max_score(self) -> float:
        """Maximum score across folds."""
        return max(self.fold_scores) if self.fold_scores else 0.0
    
    @property
    def valid_folds(self) -> int:
        """Number of folds with valid scores."""
        return len(self.fold_scores)
    
    @property
    def is_consistent(self) -> bool:
        """
        Check if results are consistent across folds.
        
        Returns True if:
        - All folds have valid scores
        - Standard deviation is reasonable (< mean for positive metrics)
        """
        if self.valid_folds < self.n_splits:
            return False
        if self.mean_score > 0 and self.std_score > self.mean_score:
            return False
        return True
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "params": self.params,
            "metric": self.metric,
            "n_folds": self.n_splits,
            "valid_folds": self.valid_folds,
            "mean": self.mean_score,
            "std": self.std_score,
            "min": self.min_score,
            "max": self.max_score,
            "is_consistent": self.is_consistent,
        }


class WalkForwardSplit:
    """
    Walk-forward optimization with consistent class-based interface.
    
    Provides two modes:
    - Sliding window: train window moves with test window
    - Expanding window: train window anchored at start, grows over time
    
    Example (sliding):
        Split 0: [TRAIN|embargo|TEST]....................
        Split 1: ......[TRAIN|embargo|TEST]..............
        Split 2: ............[TRAIN|embargo|TEST]........
    
    Example (expanding):
        Split 0: [TRAIN|embargo|TEST]....................
        Split 1: [TRAIN....|embargo|TEST]................
        Split 2: [TRAIN........|embargo|TEST]............
    """
    
    def __init__(
        self,
        start_ns: int,
        end_ns: int,
        train_duration_ns: int,
        test_duration_ns: int,
        step_ns: Optional[int] = None,
        embargo_pct: float = 0.01,
        expanding: bool = False,
    ):
        """
        Initialize WalkForwardSplit.
        
        Args:
            start_ns: Start timestamp in nanoseconds
            end_ns: End timestamp in nanoseconds
            train_duration_ns: Training window duration (or initial if expanding)
            test_duration_ns: Test window duration
            step_ns: Step size between splits (defaults to test_duration_ns)
            embargo_pct: Fraction of total period for embargo gap
            expanding: If True, use expanding window (anchored training)
        """
        if train_duration_ns <= 0:
            raise ValueError("train_duration_ns must be positive")
        if test_duration_ns <= 0:
            raise ValueError("test_duration_ns must be positive")
        if not 0 <= embargo_pct < 1:
            raise ValueError("embargo_pct must be in [0, 1)")
        
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.train_duration_ns = train_duration_ns
        self.test_duration_ns = test_duration_ns
        self.step_ns = step_ns if step_ns is not None else test_duration_ns
        self.embargo_pct = embargo_pct
        self.expanding = expanding
        
        self._total_duration = end_ns - start_ns
        self._embargo_ns = int(self._total_duration * embargo_pct)
        
        log.debug(
            f"WalkForwardSplit: train={train_duration_ns / 1e9:.0f}s, "
            f"test={test_duration_ns / 1e9:.0f}s, "
            f"step={self.step_ns / 1e9:.0f}s, "
            f"embargo={self._embargo_ns / 1e9:.0f}s, "
            f"expanding={expanding}"
        )
    
    def split(self) -> Iterator[Tuple[List[TimeRange], TimeRange]]:
        """
        Generate train/test splits.
        
        Yields:
            Tuple of (train_ranges, test_range) for each window position.
            train_ranges is a list with single TimeRange for consistency with PurgedKFold.
        """
        if self.expanding:
            yield from self._expanding_split()
        else:
            yield from self._sliding_split()
    
    def _sliding_split(self) -> Iterator[Tuple[List[TimeRange], TimeRange]]:
        """Generate sliding window splits."""
        current_train_start = self.start_ns
        
        while True:
            train_end = current_train_start + self.train_duration_ns
            test_start = train_end + self._embargo_ns
            test_end = test_start + self.test_duration_ns
            
            if test_end > self.end_ns:
                break
            
            train_range = TimeRange(start_ns=current_train_start, end_ns=train_end)
            test_range = TimeRange(start_ns=test_start, end_ns=test_end)
            
            yield [train_range], test_range
            
            current_train_start += self.step_ns
    
    def _expanding_split(self) -> Iterator[Tuple[List[TimeRange], TimeRange]]:
        """Generate expanding window splits."""
        train_end = self.start_ns + self.train_duration_ns
        
        while True:
            test_start = train_end + self._embargo_ns
            test_end = test_start + self.test_duration_ns
            
            if test_end > self.end_ns:
                break
            
            # Training always anchored at start
            train_range = TimeRange(start_ns=self.start_ns, end_ns=train_end)
            test_range = TimeRange(start_ns=test_start, end_ns=test_end)
            
            yield [train_range], test_range
            
            train_end += self.step_ns
    
    def get_n_splits(self) -> int:
        """Get the number of splits that will be generated."""
        count = 0
        if self.expanding:
            train_end = self.start_ns + self.train_duration_ns
            while True:
                test_end = train_end + self._embargo_ns + self.test_duration_ns
                if test_end > self.end_ns:
                    break
                count += 1
                train_end += self.step_ns
        else:
            current_train_start = self.start_ns
            while True:
                test_end = current_train_start + self.train_duration_ns + self._embargo_ns + self.test_duration_ns
                if test_end > self.end_ns:
                    break
                count += 1
                current_train_start += self.step_ns
        return count
    
    def info(self) -> dict:
        """Get information about the split configuration."""
        return {
            "mode": "expanding" if self.expanding else "sliding",
            "train_duration_days": self.train_duration_ns / DAY_NS,
            "test_duration_days": self.test_duration_ns / DAY_NS,
            "step_days": self.step_ns / DAY_NS,
            "embargo_days": self._embargo_ns / DAY_NS,
            "embargo_pct": self.embargo_pct,
            "n_splits": self.get_n_splits(),
        }


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV) with class-based interface.
    
    Instead of using single folds as test (like PurgedKFold), CPCV uses
    combinations of multiple folds as the test set. This provides:
    - More diverse test scenarios
    - Better estimation of generalization error variance
    - Reduced chance of lucky/unlucky single-fold results
    
    IMPORTANT: split() yields (train_ranges, test_ranges) where test_ranges is
    a LIST of TimeRange objects (multiple disjoint test windows). Do NOT merge
    these ranges! Run backtest on each test range separately and aggregate scores.
    
    Example with n_splits=4, n_test_splits=2:
        Fold indices: [0, 1, 2, 3]
        Test combinations: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        
        For test=(0,1): test_ranges = [fold0_range, fold1_range]  (2 disjoint windows!)
        For test=(1,3): test_ranges = [fold1_range, fold3_range]  (gaps between them!)
        
        WRONG: merge into single (min_start, max_end) - includes untested data!
        RIGHT: run backtest on each range, aggregate scores.
    
    Reference: AFML Chapter 7 - Combinatorial Purged Cross-Validation
    """
    
    def __init__(
        self,
        start_ns: int,
        end_ns: int,
        n_splits: int = 6,
        n_test_splits: int = 2,
        embargo_pct: float = 0.01,
    ):
        """
        Initialize CombinatorialPurgedCV.
        
        Args:
            start_ns: Start timestamp in nanoseconds
            end_ns: End timestamp in nanoseconds
            n_splits: Total number of folds to divide time into
            n_test_splits: Number of folds to use as test in each combination
            embargo_pct: Fraction of total period for embargo gap
        """
        if n_splits < 3:
            raise ValueError("n_splits must be at least 3 for CPCV")
        if n_test_splits < 1 or n_test_splits >= n_splits:
            raise ValueError("n_test_splits must be in [1, n_splits)")
        if not 0 <= embargo_pct < 1:
            raise ValueError("embargo_pct must be in [0, 1)")
        
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct
        
        self._total_duration = end_ns - start_ns
        self._fold_duration = self._total_duration // n_splits
        self._embargo_ns = int(self._total_duration * embargo_pct)
        
        # Calculate number of combinations
        from math import comb
        self._n_combinations = comb(n_splits, n_test_splits)
        
        log.debug(
            f"CombinatorialPurgedCV: {n_splits} folds, "
            f"{n_test_splits} test folds per split, "
            f"{self._n_combinations} combinations, "
            f"embargo={self._embargo_ns / 1e9:.0f}s"
        )
    
    def _get_fold_range(self, fold_idx: int) -> TimeRange:
        """Get the time range for a specific fold."""
        start = self.start_ns + fold_idx * self._fold_duration
        end = start + self._fold_duration if fold_idx < self.n_splits - 1 else self.end_ns
        return TimeRange(start_ns=start, end_ns=end)
    
    def split(self) -> Iterator[Tuple[List[TimeRange], List[TimeRange]]]:
        """
        Generate train/test splits.
        
        Yields:
            Tuple of (train_ranges, test_ranges).
            Note: test_ranges is a LIST (multiple folds combined).
        """
        # Generate all combinations of test fold indices
        for test_indices in combinations(range(self.n_splits), self.n_test_splits):
            test_ranges = [self._get_fold_range(i) for i in test_indices]
            train_ranges = self._get_train_ranges(test_indices, test_ranges)
            yield train_ranges, test_ranges
    
    def _get_train_ranges(
        self,
        test_indices: Tuple[int, ...],
        test_ranges: List[TimeRange]
    ) -> List[TimeRange]:
        """
        Get training ranges with embargo applied.
        
        Training excludes test folds and applies embargo after each test fold.
        """
        train_ranges = []
        
        for fold_idx in range(self.n_splits):
            if fold_idx in test_indices:
                continue
            
            fold_range = self._get_fold_range(fold_idx)
            adjusted_range = fold_range
            
            # Check relationship with each test fold
            for test_range in test_ranges:
                # If fold is immediately after test, apply embargo
                if adjusted_range.start_ns >= test_range.end_ns:
                    embargo_end = test_range.end_ns + self._embargo_ns
                    if adjusted_range.start_ns < embargo_end:
                        # Need to shrink this fold due to embargo
                        new_start = max(adjusted_range.start_ns, embargo_end)
                        if new_start < adjusted_range.end_ns:
                            adjusted_range = TimeRange(
                                start_ns=new_start,
                                end_ns=adjusted_range.end_ns
                            )
                        else:
                            adjusted_range = None
                            break
            
            if adjusted_range is not None and adjusted_range.duration_ns > 0:
                train_ranges.append(adjusted_range)
        
        return train_ranges
    
    def get_n_splits(self) -> int:
        """Get the number of splits (combinations)."""
        return self._n_combinations
    
    def info(self) -> dict:
        """Get information about the split configuration."""
        return {
            "n_folds": self.n_splits,
            "n_test_folds": self.n_test_splits,
            "n_combinations": self._n_combinations,
            "fold_duration_days": self._fold_duration / DAY_NS,
            "embargo_days": self._embargo_ns / DAY_NS,
            "embargo_pct": self.embargo_pct,
        }


def embargo_split(
    start_ns: int,
    end_ns: int,
    test_pct: float = 0.2,
    embargo_pct: float = 0.02,
) -> Tuple[TimeRange, TimeRange]:
    """
    Simple train/test split with embargo.
    
    Creates a single split:
    [-------TRAIN-------][embargo][---TEST---]
    
    Args:
        start_ns: Start timestamp
        end_ns: End timestamp
        test_pct: Fraction of data for test (from end)
        embargo_pct: Fraction of data for embargo gap
        
    Returns:
        Tuple of (train_range, test_range)
    """
    total = end_ns - start_ns
    test_duration = int(total * test_pct)
    embargo_duration = int(total * embargo_pct)
    
    test_start = end_ns - test_duration
    train_end = test_start - embargo_duration
    
    train_range = TimeRange(start_ns=start_ns, end_ns=train_end)
    test_range = TimeRange(start_ns=test_start, end_ns=end_ns)
    
    return train_range, test_range


def walk_forward_splits(
    start_ns: int,
    end_ns: int,
    train_duration_ns: int,
    test_duration_ns: int,
    step_ns: Optional[int] = None,
    embargo_ns: int = 0,
) -> Iterator[Tuple[TimeRange, TimeRange]]:
    """
    Walk-forward optimization splits (sliding window).
    
    Creates sliding window splits where train window moves forward:
    [TRAIN][TEST] -> step -> [TRAIN][TEST] -> step -> ...
    
    Args:
        start_ns: Start timestamp
        end_ns: End timestamp
        train_duration_ns: Duration of training window
        test_duration_ns: Duration of test window
        step_ns: Step size between windows (defaults to test_duration)
        embargo_ns: Gap between train and test
        
    Yields:
        (train_range, test_range) tuples
    """
    if step_ns is None:
        step_ns = test_duration_ns
    
    current_train_start = start_ns
    
    while True:
        train_end = current_train_start + train_duration_ns
        test_start = train_end + embargo_ns
        test_end = test_start + test_duration_ns
        
        if test_end > end_ns:
            break
        
        train_range = TimeRange(start_ns=current_train_start, end_ns=train_end)
        test_range = TimeRange(start_ns=test_start, end_ns=test_end)
        
        yield train_range, test_range
        
        current_train_start += step_ns


def expanding_window_splits(
    start_ns: int,
    end_ns: int,
    initial_train_ns: int,
    test_duration_ns: int,
    step_ns: Optional[int] = None,
    embargo_ns: int = 0,
) -> Iterator[Tuple[TimeRange, TimeRange]]:
    """
    Expanding window (anchored) walk-forward splits.
    
    Training window always starts from the same point but expands:
    [TRAIN][TEST] -> step -> [TRAIN....][TEST] -> step -> [TRAIN........][TEST]
    
    This is often preferred for financial data as it uses all available history.
    
    Args:
        start_ns: Start timestamp (anchor point)
        end_ns: End timestamp
        initial_train_ns: Minimum initial training duration
        test_duration_ns: Duration of test window
        step_ns: Step size between tests (defaults to test_duration)
        embargo_ns: Gap between train and test
        
    Yields:
        (train_range, test_range) tuples
    """
    if step_ns is None:
        step_ns = test_duration_ns
    
    train_end = start_ns + initial_train_ns
    
    while True:
        test_start = train_end + embargo_ns
        test_end = test_start + test_duration_ns
        
        if test_end > end_ns:
            break
        
        train_range = TimeRange(start_ns=start_ns, end_ns=train_end)  # Anchored at start
        test_range = TimeRange(start_ns=test_start, end_ns=test_end)
        
        yield train_range, test_range
        
        train_end += step_ns  # Expand training window


def combinatorial_purged_cv(
    start_ns: int,
    end_ns: int,
    n_splits: int,
    n_test_splits: int = 2,
    embargo_pct: float = 0.01,
) -> Iterator[Tuple[List[TimeRange], List[TimeRange]]]:
    """
    Combinatorial purged cross-validation.
    
    Instead of using single folds as test, uses combinations of folds.
    This provides more diverse test scenarios.
    
    Args:
        start_ns: Start timestamp
        end_ns: End timestamp
        n_splits: Total number of folds
        n_test_splits: Number of folds to use as test
        embargo_pct: Embargo fraction
        
    Yields:
        (train_ranges, test_ranges) tuples
    """
    from itertools import combinations
    
    fold_duration = (end_ns - start_ns) // n_splits
    embargo_ns = int((end_ns - start_ns) * embargo_pct)
    
    def get_fold_range(idx: int) -> TimeRange:
        start = start_ns + idx * fold_duration
        end = start + fold_duration if idx < n_splits - 1 else end_ns
        return TimeRange(start_ns=start, end_ns=end)
    
    # Generate all combinations of test folds
    for test_indices in combinations(range(n_splits), n_test_splits):
        test_ranges = [get_fold_range(i) for i in test_indices]
        
        # Train ranges are all other folds with embargo applied
        train_ranges = []
        for i in range(n_splits):
            if i in test_indices:
                continue
            
            fold_range = get_fold_range(i)
            
            # Check if adjacent to any test fold
            for test_range in test_ranges:
                if fold_range.end_ns <= test_range.start_ns:
                    # Before test - OK
                    pass
                elif fold_range.start_ns >= test_range.end_ns:
                    # After test - apply embargo
                    adjusted_start = max(fold_range.start_ns, test_range.end_ns + embargo_ns)
                    if adjusted_start < fold_range.end_ns:
                        fold_range = TimeRange(start_ns=adjusted_start, end_ns=fold_range.end_ns)
            
            train_ranges.append(fold_range)
        
        yield train_ranges, test_ranges
