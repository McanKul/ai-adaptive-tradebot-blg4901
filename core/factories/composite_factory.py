"""
core/factories/composite_factory.py
===================================
Factory for building ``CompositeStrategy`` from a YAML/JSON spec.

Spec shape (YAML)::

    name: trend_plus_meanrev
    policy: regime_gate          # fixed | regime_gate | regime_weight
    default_sizing:
      mode: margin_usd
      margin_usd: 100
      leverage: 5
    slots:
      - id: trend_v1
        strategy: EMACrossMACDTrend
        params: {fast_ema_period: 12, slow_ema_period: 26}
        weight: 1.0
        entry_coefficient: 1.0
        sizing:
          mode: margin_usd
          margin_usd: 150
          leverage: 5
        exit:
          tp_pct: 0.02
          sl_pct: 0.01
          trailing_pct: 0.005
        regimes: [trend_up, trend_down]
      - id: meanrev_v1
        strategy: RSIThreshold
        params: {rsi_period: 14, lower: 30, upper: 70}
        entry_coefficient: 0.5
        regimes: [range, vol_low]

Reuses :class:`StrategyFactory` for child instantiation.
"""
from __future__ import annotations
import json
import os
from typing import Any, Dict, Optional

from Backtest.exit_manager import ExitConfig
from Interfaces.strategy_adapter import SizingConfig, SizingMode
from Interfaces.strategy_slot import StrategySlot
from Strategy.composite_strategy import CompositeStrategy
from core.factories.strategy_factory import StrategyFactory


def _load_dict(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"composite spec not found: {path}")
    if path.endswith((".yml", ".yaml")):
        try:
            import yaml  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("PyYAML required for YAML composite specs") from e
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_sizing(d: Optional[Dict[str, Any]]) -> Optional[SizingConfig]:
    if not d:
        return None
    mode_str = d.get("mode", "margin_usd")
    valid = {m.value for m in SizingMode}
    if mode_str not in valid:
        raise ValueError(f"unknown sizing mode '{mode_str}', valid: {sorted(valid)}")
    return SizingConfig(
        mode=SizingMode(mode_str),
        fixed_qty=d.get("fixed_qty"),
        notional_usd=d.get("notional_usd"),
        margin_usd=float(d.get("margin_usd", 100.0)),
        leverage=float(d.get("leverage", 1.0)),
        leverage_mode=d.get("leverage_mode", "margin"),
    )


def _build_exit(d: Optional[Dict[str, Any]]) -> Optional[ExitConfig]:
    if not d:
        return None
    return ExitConfig(
        take_profit_usd=d.get("tp_usd"),
        take_profit_pct=d.get("tp_pct"),
        stop_loss_usd=d.get("sl_usd"),
        stop_loss_pct=d.get("sl_pct"),
        trailing_stop_pct=d.get("trailing_pct"),
        max_holding_bars=d.get("max_bars"),
        use_intrabar_checks=bool(d.get("use_intrabar_checks", False)),
        leverage=float(d.get("leverage", 1.0)),
    )


class CompositeFactory:
    """Builds ``CompositeStrategy`` from a dict / YAML / JSON spec."""

    @staticmethod
    def from_spec(
        spec: Dict[str, Any],
        regime_detector: Any = None,
    ) -> CompositeStrategy:
        slots_raw = spec.get("slots") or []
        if not slots_raw:
            raise ValueError("composite spec must contain at least one slot")

        slots = []
        for s in slots_raw:
            if "id" not in s or "strategy" not in s:
                raise ValueError(f"slot missing 'id' or 'strategy': {s}")
            cls = StrategyFactory.resolve_class(s["strategy"])
            inst = cls(**(s.get("params") or {}))
            slots.append(StrategySlot(
                id=s["id"],
                strategy=inst,
                weight=float(s.get("weight", 1.0)),
                entry_coefficient=float(s.get("entry_coefficient", 1.0)),
                sizing=_build_sizing(s.get("sizing")),
                exit=_build_exit(s.get("exit")),
                risk_limits=s.get("risk_limits"),
                regimes=s.get("regimes"),
                enabled=bool(s.get("enabled", True)),
            ))

        return CompositeStrategy(
            slots=slots,
            default_sizing=_build_sizing(spec.get("default_sizing")),
            regime_detector=regime_detector,
            policy=spec.get("policy", "fixed"),
            name=spec.get("name", "composite"),
        )

    @staticmethod
    def from_path(path: str, regime_detector: Any = None) -> CompositeStrategy:
        return CompositeFactory.from_spec(
            _load_dict(path), regime_detector=regime_detector
        )
