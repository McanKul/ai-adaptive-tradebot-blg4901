"""
core/factories/regime_factory.py
================================
Registry-based factory for regime detectors.

Same shape as :class:`StrategyFactory`: a name → module-path map.
Bootstrap pre-registers the built-in detectors (ADX, ATR percentile).
"""
from __future__ import annotations
import importlib
from typing import Dict, Any, List

from Interfaces.IRegimeDetector import IRegimeDetector


class RegimeFactory:
    """Registry-based factory for ``IRegimeDetector`` implementations."""

    _registry: Dict[str, str] = {}

    @classmethod
    def register(cls, name: str, dotted_path: str) -> None:
        cls._registry[name] = dotted_path

    @classmethod
    def resolve_class(cls, name: str):
        if name in cls._registry:
            dotted = cls._registry[name]
        else:
            dotted = name  # allow callers to pass full dotted path
        module_path, _, attr = dotted.rpartition(".")
        if not module_path:
            raise ValueError(
                f"unknown regime detector '{name}' and not a dotted path"
            )
        try:
            mod = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(f"cannot import regime module '{module_path}': {e}") from e
        klass = getattr(mod, attr, None)
        if klass is None or not isinstance(klass, type):
            raise AttributeError(
                f"'{attr}' not a class in module '{module_path}'"
            )
        if not issubclass(klass, IRegimeDetector):
            raise TypeError(f"{klass} does not implement IRegimeDetector")
        return klass

    @classmethod
    def create(cls, name: str, **kwargs) -> IRegimeDetector:
        klass = cls.resolve_class(name)
        return klass(**kwargs)

    @classmethod
    def from_spec(cls, spec: Any) -> IRegimeDetector:
        """Build from a dict ({"type": "...", "params": {...}}) or a list
        of such dicts (auto-wraps in CompositeRegimeDetector)."""
        if isinstance(spec, list):
            from Strategy.regime.composite_regime import CompositeRegimeDetector
            detectors = [cls.from_spec(s) for s in spec]
            return CompositeRegimeDetector(detectors=detectors)
        if not isinstance(spec, dict) or "type" not in spec:
            raise ValueError(
                f"regime spec must be a dict with 'type' or a list, got {spec!r}"
            )
        return cls.create(spec["type"], **(spec.get("params") or {}))

    @classmethod
    def registered(cls) -> List[str]:
        return sorted(cls._registry.keys())
