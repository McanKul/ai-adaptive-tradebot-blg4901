"""
Interfaces/strategy_slot.py
===========================
Per-strategy allocation slot used by ``CompositeStrategy``.

A slot bundles a child strategy with the per-strategy knobs the user
asked for: entry coefficient, sizing override, exit policy, risk
sub-limits, and regime gating.

Composite handles aggregation; the engine still sees a single
``IStrategy``.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from Interfaces.IStrategy import IStrategy
from Interfaces.strategy_adapter import SizingConfig
from Backtest.exit_manager import ExitConfig


@dataclass
class StrategySlot:
    """One slot in a ``CompositeStrategy``.

    Attributes:
        id: Unique slot identifier.  Tagged onto every order via
            ``order.strategy_id`` for routing and per-slot accounting.
        strategy: Child strategy instance.
        weight: Allocator weight (used by regime_weight policy).
        entry_coefficient: Multiplies the per-slot order quantity.
            Lets the user dial different "entry coefficients" per regime
            without rewriting the child strategy.
        sizing: Per-slot ``SizingConfig``.  When ``None`` the composite's
            ``default_sizing`` is used.
        exit: Per-slot ``ExitConfig`` for an isolated ``ExitManager``.
        risk_limits: Optional per-slot risk overrides
            (e.g. ``{"max_position_notional": 5_000.0}``).
        regimes: Tags the slot is active in.  ``None`` ⇒ always active.
            Honored by composite ``policy="regime_gate"``/``"regime_weight"``.
        enabled: Hard kill switch.
    """
    id: str
    strategy: IStrategy
    weight: float = 1.0
    entry_coefficient: float = 1.0
    sizing: Optional[SizingConfig] = None
    exit: Optional[ExitConfig] = None
    risk_limits: Optional[Dict[str, Any]] = None
    regimes: Optional[List[str]] = None
    enabled: bool = True

    def __post_init__(self):
        if not self.id or not isinstance(self.id, str):
            raise ValueError("StrategySlot.id must be a non-empty string")
        if self.weight < 0:
            raise ValueError(f"StrategySlot.weight must be >= 0, got {self.weight}")
        if self.entry_coefficient < 0:
            raise ValueError(
                f"StrategySlot.entry_coefficient must be >= 0, got {self.entry_coefficient}"
            )
