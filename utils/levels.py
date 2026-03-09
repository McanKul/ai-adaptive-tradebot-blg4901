"""
utils/levels.py
===============
Support and Resistance level detection utilities.

Provides helpers for computing S/R levels from bar history:
- Pivot Points (classic, Fibonacci, Woodie, Camarilla)
- Swing High/Low detection
- Volume profile levels (optional)

These are STRATEGY HELPERS - they don't affect execution.
Strategies can use these levels for entry/exit decisions.

Usage:
    from utils.levels import compute_pivot_levels, detect_swing_levels
    
    # Classic pivot points from yesterday's OHLC
    pivots = compute_pivot_levels(high, low, close, method="classic")
    
    # Swing highs/lows from bar history
    swings = detect_swing_levels(highs, lows, window=5, num_levels=3)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Tuple
import numpy as np
import logging

log = logging.getLogger(__name__)


@dataclass
class SupportResistanceResult:
    """Result from S/R level computation."""
    levels: List[float]  # All levels sorted
    supports: List[float]  # Levels below reference price
    resistances: List[float]  # Levels above reference price
    method: str  # Method used
    metadata: Dict = field(default_factory=dict)


def compute_pivot_levels(
    high: float,
    low: float,
    close: float,
    method: Literal["classic", "fibonacci", "woodie", "camarilla"] = "classic",
    open_price: Optional[float] = None,
) -> SupportResistanceResult:
    """
    Compute pivot point levels from OHLC data.
    
    Typically uses previous period's OHLC (e.g., yesterday's daily bar).
    
    Args:
        high: Previous period high
        low: Previous period low
        close: Previous period close
        method: Pivot calculation method
        open_price: Previous period open (required for Woodie)
        
    Returns:
        SupportResistanceResult with pivot levels
    """
    levels = []
    
    if method == "classic":
        # Classic floor trader pivots
        pivot = (high + low + close) / 3
        
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
        levels = [s3, s2, s1, pivot, r1, r2, r3]
        
    elif method == "fibonacci":
        # Fibonacci pivot points
        pivot = (high + low + close) / 3
        range_ = high - low
        
        r1 = pivot + 0.382 * range_
        r2 = pivot + 0.618 * range_
        r3 = pivot + 1.0 * range_
        s1 = pivot - 0.382 * range_
        s2 = pivot - 0.618 * range_
        s3 = pivot - 1.0 * range_
        
        levels = [s3, s2, s1, pivot, r1, r2, r3]
        
    elif method == "woodie":
        # Woodie pivot points (emphasizes close)
        pivot = (high + low + 2 * close) / 4
        
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        
        levels = [s2, s1, pivot, r1, r2]
        
    elif method == "camarilla":
        # Camarilla pivot points (uses multipliers)
        range_ = high - low
        
        s1 = close - range_ * 1.1 / 12
        s2 = close - range_ * 1.1 / 6
        s3 = close - range_ * 1.1 / 4
        s4 = close - range_ * 1.1 / 2
        r1 = close + range_ * 1.1 / 12
        r2 = close + range_ * 1.1 / 6
        r3 = close + range_ * 1.1 / 4
        r4 = close + range_ * 1.1 / 2
        
        pivot = (high + low + close) / 3
        levels = [s4, s3, s2, s1, pivot, r1, r2, r3, r4]
    
    else:
        raise ValueError(f"Unknown pivot method: {method}")
    
    # Sort and separate
    levels = sorted(levels)
    supports = [l for l in levels if l < close]
    resistances = [l for l in levels if l > close]
    
    return SupportResistanceResult(
        levels=levels,
        supports=supports,
        resistances=resistances,
        method=f"pivot_{method}",
        metadata={"high": high, "low": low, "close": close},
    )


def detect_swing_levels(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    window: int = 5,
    num_levels: int = 3,
    min_touches: int = 1,
    tolerance_bps: float = 10.0,
    reference_price: Optional[float] = None,
) -> SupportResistanceResult:
    """
    Detect support/resistance levels from swing highs/lows.
    
    A swing high is a local maximum where the high is greater than
    the highs of `window` bars on either side.
    A swing low is a local minimum similarly.
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        window: Number of bars on each side to confirm swing
        num_levels: Maximum number of S/R levels to return
        min_touches: Minimum times a level must be touched to qualify
        tolerance_bps: Tolerance in basis points for grouping levels
        reference_price: Price to use for S/R classification (default: last close)
        
    Returns:
        SupportResistanceResult with detected levels
    """
    n = len(highs)
    if n < 2 * window + 1:
        return SupportResistanceResult(
            levels=[], supports=[], resistances=[],
            method="swing", metadata={"error": "insufficient data"}
        )
    
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    
    swing_highs = []
    swing_lows = []
    
    # Detect swings
    for i in range(window, n - window):
        # Check swing high
        is_swing_high = True
        for j in range(1, window + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs.append(highs[i])
        
        # Check swing low
        is_swing_low = True
        for j in range(1, window + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break
        if is_swing_low:
            swing_lows.append(lows[i])
    
    # Cluster nearby levels
    all_swings = swing_highs + swing_lows
    if not all_swings:
        ref_price = reference_price or closes[-1]
        return SupportResistanceResult(
            levels=[], supports=[], resistances=[],
            method="swing", metadata={"swing_highs": 0, "swing_lows": 0}
        )
    
    tolerance = tolerance_bps / 10000.0
    clusters = cluster_levels(all_swings, tolerance)
    
    # Filter by min_touches
    filtered_clusters = [c for c in clusters if c["count"] >= min_touches]
    
    # Sort by touch count (most significant first)
    filtered_clusters.sort(key=lambda x: x["count"], reverse=True)
    
    # Take top N levels
    top_levels = [c["level"] for c in filtered_clusters[:num_levels * 2]]
    top_levels = sorted(top_levels)
    
    # Classify as support or resistance
    ref_price = reference_price or closes[-1]
    supports = [l for l in top_levels if l < ref_price][:num_levels]
    resistances = [l for l in top_levels if l > ref_price][:num_levels]
    
    return SupportResistanceResult(
        levels=sorted(supports + resistances),
        supports=supports,
        resistances=resistances,
        method="swing",
        metadata={
            "window": window,
            "swing_highs_found": len(swing_highs),
            "swing_lows_found": len(swing_lows),
            "clusters_formed": len(clusters),
            "reference_price": ref_price,
        },
    )


def cluster_levels(
    levels: List[float],
    tolerance: float
) -> List[Dict]:
    """
    Cluster nearby price levels together.
    
    Args:
        levels: List of price levels
        tolerance: Fraction tolerance for grouping (0.001 = 0.1%)
        
    Returns:
        List of dicts with 'level' (average) and 'count'
    """
    if not levels:
        return []
    
    sorted_levels = sorted(levels)
    clusters = []
    current_cluster = [sorted_levels[0]]
    
    for i in range(1, len(sorted_levels)):
        level = sorted_levels[i]
        cluster_avg = sum(current_cluster) / len(current_cluster)
        
        # Check if within tolerance
        if abs(level - cluster_avg) / cluster_avg <= tolerance:
            current_cluster.append(level)
        else:
            # Save current cluster and start new one
            clusters.append({
                "level": sum(current_cluster) / len(current_cluster),
                "count": len(current_cluster),
                "min": min(current_cluster),
                "max": max(current_cluster),
            })
            current_cluster = [level]
    
    # Don't forget last cluster
    if current_cluster:
        clusters.append({
            "level": sum(current_cluster) / len(current_cluster),
            "count": len(current_cluster),
            "min": min(current_cluster),
            "max": max(current_cluster),
        })
    
    return clusters


def find_nearest_level(
    price: float,
    levels: List[float],
    direction: Literal["above", "below", "nearest"] = "nearest"
) -> Optional[float]:
    """
    Find the nearest S/R level to a given price.
    
    Args:
        price: Reference price
        levels: List of S/R levels
        direction: Search direction
        
    Returns:
        Nearest level or None if no levels
    """
    if not levels:
        return None
    
    if direction == "above":
        above = [l for l in levels if l > price]
        return min(above) if above else None
    elif direction == "below":
        below = [l for l in levels if l < price]
        return max(below) if below else None
    else:  # nearest
        return min(levels, key=lambda l: abs(l - price))


def compute_level_distance(
    price: float,
    level: float,
    as_bps: bool = True
) -> float:
    """
    Compute distance from price to a level.
    
    Args:
        price: Current price
        level: S/R level
        as_bps: Return as basis points (True) or fraction (False)
        
    Returns:
        Distance (positive if level above price, negative if below)
    """
    if price == 0:
        return 0.0
    
    diff = (level - price) / price
    return diff * 10000 if as_bps else diff


def get_level_strength(
    level: float,
    highs: np.ndarray,
    lows: np.ndarray,
    tolerance_bps: float = 10.0
) -> int:
    """
    Count how many times a level was touched (tested).
    
    Args:
        level: The S/R level to test
        highs: Array of high prices
        lows: Array of low prices
        tolerance_bps: Tolerance in basis points
        
    Returns:
        Number of touches
    """
    tolerance = level * tolerance_bps / 10000
    touches = 0
    
    for h, l in zip(highs, lows):
        # Level was within bar's range
        if l - tolerance <= level <= h + tolerance:
            touches += 1
    
    return touches


class SupportResistanceTracker:
    """
    Tracks S/R levels over time for a symbol.
    
    Maintains rolling window of bars and recomputes levels periodically.
    """
    
    def __init__(
        self,
        window: int = 50,
        swing_window: int = 5,
        num_levels: int = 3,
        update_frequency: int = 1,  # Update every N bars
        pivot_method: str = "classic",
    ):
        self.window = window
        self.swing_window = swing_window
        self.num_levels = num_levels
        self.update_frequency = update_frequency
        self.pivot_method = pivot_method
        
        self._bars_since_update = 0
        self._last_result: Optional[SupportResistanceResult] = None
        
        # Rolling bar data
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._closes: List[float] = []
    
    def on_bar(self, high: float, low: float, close: float) -> Optional[SupportResistanceResult]:
        """
        Update with new bar and optionally recompute levels.
        
        Args:
            high: Bar high
            low: Bar low
            close: Bar close
            
        Returns:
            Updated S/R result if recomputed, else last result
        """
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)
        
        # Trim to window
        if len(self._highs) > self.window:
            self._highs = self._highs[-self.window:]
            self._lows = self._lows[-self.window:]
            self._closes = self._closes[-self.window:]
        
        self._bars_since_update += 1
        
        # Recompute if needed
        if self._bars_since_update >= self.update_frequency:
            self._bars_since_update = 0
            self._last_result = self._compute()
        
        return self._last_result
    
    def _compute(self) -> SupportResistanceResult:
        """Compute S/R levels from current bar history."""
        if len(self._highs) < 2 * self.swing_window + 1:
            return SupportResistanceResult(
                levels=[], supports=[], resistances=[],
                method="combined", metadata={"error": "insufficient data"}
            )
        
        # Get swing levels
        swing_result = detect_swing_levels(
            highs=np.array(self._highs),
            lows=np.array(self._lows),
            closes=np.array(self._closes),
            window=self.swing_window,
            num_levels=self.num_levels,
        )
        
        # Get pivot levels from most recent complete period
        if len(self._highs) >= 2:
            pivot_result = compute_pivot_levels(
                high=max(self._highs[-24:]) if len(self._highs) >= 24 else max(self._highs),
                low=min(self._lows[-24:]) if len(self._lows) >= 24 else min(self._lows),
                close=self._closes[-1],
                method=self.pivot_method,
            )
        else:
            pivot_result = SupportResistanceResult(
                levels=[], supports=[], resistances=[],
                method="pivot", metadata={}
            )
        
        # Combine and deduplicate
        all_levels = swing_result.levels + pivot_result.levels
        if all_levels:
            clusters = cluster_levels(all_levels, 0.001)  # 0.1% tolerance
            combined_levels = sorted([c["level"] for c in clusters])
        else:
            combined_levels = []
        
        ref_price = self._closes[-1] if self._closes else 0
        supports = [l for l in combined_levels if l < ref_price]
        resistances = [l for l in combined_levels if l > ref_price]
        
        return SupportResistanceResult(
            levels=combined_levels,
            supports=supports[-self.num_levels:],  # Nearest supports
            resistances=resistances[:self.num_levels],  # Nearest resistances
            method="combined",
            metadata={
                "swing_levels": swing_result.levels,
                "pivot_levels": pivot_result.levels,
                "bar_count": len(self._highs),
            },
        )
    
    def get_current_levels(self) -> Optional[SupportResistanceResult]:
        """Get the most recently computed levels."""
        return self._last_result
    
    def reset(self) -> None:
        """Clear all state."""
        self._highs.clear()
        self._lows.clear()
        self._closes.clear()
        self._bars_since_update = 0
        self._last_result = None
