"""
Backtest/scoring/search_space.py
================================
Parameter grid and search space generation for parameter sweeps.

Design decisions:
- ParameterGrid: Simple grid of all combinations
- SearchSpace: Advanced with constraints and filtering
- Generates parameter dicts for use with BacktestRunner
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Iterator, Optional, Callable
import itertools
import logging

log = logging.getLogger(__name__)


@dataclass
class ParameterSpec:
    """Specification for a single parameter."""
    name: str
    values: List[Any]
    
    def __post_init__(self):
        if not self.values:
            raise ValueError(f"Parameter '{self.name}' must have at least one value")


class ParameterGrid:
    """
    Simple parameter grid generator.
    
    Generates all combinations of parameter values.
    
    Usage:
        grid = ParameterGrid({
            "rsi_period": [14, 21, 28],
            "rsi_overbought": [70, 75, 80],
            "rsi_oversold": [20, 25, 30],
        })
        
        for params in grid:
            result = runner.run_once(factory, params)
    """
    
    def __init__(self, param_dict: Dict[str, List[Any]]):
        """
        Initialize ParameterGrid.
        
        Args:
            param_dict: Dict mapping parameter names to lists of values
        """
        self.param_dict = param_dict
        self._validate()
        
        # Compute total combinations
        self._size = 1
        for values in param_dict.values():
            self._size *= len(values)
    
    def _validate(self) -> None:
        """Validate parameter specification."""
        for name, values in self.param_dict.items():
            if not values:
                raise ValueError(f"Parameter '{name}' has no values")
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"Parameter '{name}' values must be a list")
    
    def __len__(self) -> int:
        """Number of parameter combinations."""
        return self._size
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all parameter combinations."""
        keys = list(self.param_dict.keys())
        values = [self.param_dict[k] for k in keys]
        
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get parameter combination by index."""
        if idx < 0 or idx >= self._size:
            raise IndexError(f"Index {idx} out of range [0, {self._size})")
        
        # Convert index to combination
        keys = list(self.param_dict.keys())
        result = {}
        
        for key in reversed(keys):
            values = self.param_dict[key]
            result[key] = values[idx % len(values)]
            idx //= len(values)
        
        return result
    
    def get_combinations(self) -> List[Dict[str, Any]]:
        """Get all combinations as a list."""
        return list(self)
    
    @classmethod
    def from_ranges(
        cls,
        **kwargs: tuple
    ) -> "ParameterGrid":
        """
        Create grid from numeric ranges.
        
        Args:
            **kwargs: Parameter name to (start, stop, step) tuples
            
        Example:
            grid = ParameterGrid.from_ranges(
                rsi_period=(10, 30, 5),  # [10, 15, 20, 25, 30]
                threshold=(0.5, 1.0, 0.1),
            )
        """
        param_dict = {}
        for name, (start, stop, step) in kwargs.items():
            values = []
            current = start
            while current <= stop + 1e-10:  # Handle float precision
                values.append(current)
                current += step
            param_dict[name] = values
        
        return cls(param_dict)


# Constraint function type
Constraint = Callable[[Dict[str, Any]], bool]


# =============================================================================
# PREDEFINED CONSTRAINTS
# =============================================================================

def less_than_constraint(param_a: str, param_b: str) -> Constraint:
    """Create constraint: param_a < param_b"""
    def constraint(params: Dict[str, Any]) -> bool:
        if param_a not in params or param_b not in params:
            return True  # Pass if params not present
        return params[param_a] < params[param_b]
    return constraint


def less_equal_constraint(param_a: str, param_b: str) -> Constraint:
    """Create constraint: param_a <= param_b"""
    def constraint(params: Dict[str, Any]) -> bool:
        if param_a not in params or param_b not in params:
            return True
        return params[param_a] <= params[param_b]
    return constraint


def range_constraint(param: str, min_val: float, max_val: float) -> Constraint:
    """Create constraint: min_val <= param <= max_val"""
    def constraint(params: Dict[str, Any]) -> bool:
        if param not in params:
            return True
        return min_val <= params[param] <= max_val
    return constraint


def leverage_constraint(max_leverage: float) -> Constraint:
    """Create constraint: leverage <= max_leverage"""
    def constraint(params: Dict[str, Any]) -> bool:
        lev = params.get("leverage", 1.0)
        return lev <= max_leverage
    return constraint


class SearchSpace:
    """
    Advanced search space with constraints.
    
    Supports:
    - Parameter constraints (e.g., oversold < overbought)
    - Conditional parameters
    - Filtering invalid combinations
    - Predefined constraint helpers
    
    Usage:
        space = SearchSpace()
        space.add("rsi_period", [14, 21, 28])
        space.add("rsi_overbought", [70, 75, 80])
        space.add("rsi_oversold", [20, 25, 30])
        
        # Use predefined constraint
        space.require_less_than("rsi_oversold", "rsi_overbought")
        
        # Or custom lambda
        space.add_constraint(lambda p: p["rsi_oversold"] < p["rsi_overbought"])
        
        for params in space:
            # Only valid combinations
            result = runner.run_once(factory, params)
    """
    
    def __init__(self):
        """Initialize SearchSpace."""
        self._params: Dict[str, List[Any]] = {}
        self._constraints: List[Constraint] = []
        self._cached_valid: Optional[List[Dict[str, Any]]] = None
    
    def add(self, name: str, values: List[Any]) -> "SearchSpace":
        """
        Add a parameter to the search space.
        
        Args:
            name: Parameter name
            values: List of possible values
            
        Returns:
            self for chaining
        """
        if not values:
            raise ValueError(f"Parameter '{name}' must have at least one value")
        
        self._params[name] = list(values)
        self._cached_valid = None
        return self
    
    def add_range(
        self,
        name: str,
        start: float,
        stop: float,
        step: float
    ) -> "SearchSpace":
        """
        Add a numeric range parameter.
        
        Args:
            name: Parameter name
            start: Start value (inclusive)
            stop: Stop value (inclusive)
            step: Step size
            
        Returns:
            self for chaining
        """
        values = []
        current = start
        while current <= stop + 1e-10:
            values.append(current)
            current += step
        
        return self.add(name, values)
    
    def add_constraint(self, constraint: Constraint) -> "SearchSpace":
        """
        Add a constraint function.
        
        Args:
            constraint: Function that takes params dict, returns True if valid
            
        Returns:
            self for chaining
        """
        self._constraints.append(constraint)
        self._cached_valid = None
        return self
    
    # =========================================================================
    # CONVENIENCE CONSTRAINT METHODS
    # =========================================================================
    
    def require_less_than(self, param_a: str, param_b: str) -> "SearchSpace":
        """
        Add constraint: param_a < param_b
        
        Common use case: rsi_oversold < rsi_overbought
        """
        return self.add_constraint(less_than_constraint(param_a, param_b))
    
    def require_less_equal(self, param_a: str, param_b: str) -> "SearchSpace":
        """Add constraint: param_a <= param_b"""
        return self.add_constraint(less_equal_constraint(param_a, param_b))
    
    def require_range(self, param: str, min_val: float, max_val: float) -> "SearchSpace":
        """Add constraint: min_val <= param <= max_val"""
        return self.add_constraint(range_constraint(param, min_val, max_val))
    
    def require_max_leverage(self, max_leverage: float) -> "SearchSpace":
        """Add constraint: leverage <= max_leverage"""
        return self.add_constraint(leverage_constraint(max_leverage))
    
    def _is_valid(self, params: Dict[str, Any]) -> bool:
        """Check if parameter combination satisfies all constraints."""
        return all(c(params) for c in self._constraints)
    
    def _compute_valid(self) -> List[Dict[str, Any]]:
        """Compute all valid parameter combinations."""
        if not self._params:
            return []
        
        keys = list(self._params.keys())
        values = [self._params[k] for k in keys]
        
        valid = []
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            if self._is_valid(params):
                valid.append(params)
        
        return valid
    
    def __len__(self) -> int:
        """Number of valid parameter combinations."""
        if self._cached_valid is None:
            self._cached_valid = self._compute_valid()
        return len(self._cached_valid)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over valid parameter combinations."""
        if self._cached_valid is None:
            self._cached_valid = self._compute_valid()
        return iter(self._cached_valid)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get valid combination by index."""
        if self._cached_valid is None:
            self._cached_valid = self._compute_valid()
        return self._cached_valid[idx]
    
    def get_combinations(self) -> List[Dict[str, Any]]:
        """Get all valid combinations as a list."""
        if self._cached_valid is None:
            self._cached_valid = self._compute_valid()
        return self._cached_valid.copy()
    
    def total_unconstrained(self) -> int:
        """Total combinations without constraints."""
        total = 1
        for values in self._params.values():
            total *= len(values)
        return total
    
    def info(self) -> Dict[str, Any]:
        """Get info about the search space."""
        return {
            "parameters": {k: len(v) for k, v in self._params.items()},
            "constraints": len(self._constraints),
            "total_unconstrained": self.total_unconstrained(),
            "valid_combinations": len(self),
            "filter_rate": 1 - len(self) / self.total_unconstrained() if self.total_unconstrained() > 0 else 0,
        }
