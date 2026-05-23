"""
core/factories/strategy_factory.py
===================================
Factory for resolving and instantiating strategy classes.

Replaces the former STRATEGY_MAP dict + resolve_strategy_class().
"""
from __future__ import annotations

import importlib
import logging
from typing import Any

log = logging.getLogger(__name__)


class StrategyFactory:
    """
    Registry-based factory for trading strategies.

    Strategies are registered by friendly name (e.g. "EMACrossMACDTrend")
    mapped to their module path (e.g. "Strategy.EMACrossMACDTrend").
    All strategy modules export a class named ``Strategy``.
    """

    _registry: dict[str, str] = {}

    @classmethod
    def register(cls, name: str, module_path: str) -> None:
        """Register a strategy name -> module path mapping."""
        cls._registry[name] = module_path
        log.debug("Registered strategy: %s -> %s", name, module_path)

    @classmethod
    def resolve_class(cls, name: str) -> type:
        """
        Import and return the strategy class.

        Resolution order:
        1. Check the registry for a known name.
        2. Treat *name* as a dotted import path (``module.ClassName``).

        Raises:
            ValueError: If the name cannot be resolved.
        """
        module_path = cls._registry.get(name)

        if module_path is not None:
            mod = importlib.import_module(module_path)
            return getattr(mod, "Strategy")

        # Fallback: treat as dotted path  e.g. "Strategy.RSIThreshold.Strategy"
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            mod = importlib.import_module(parts[0])
            return getattr(mod, parts[1])

        raise ValueError(
            f"Cannot resolve strategy class: '{name}'. "
            f"Available strategies: {cls.list_available()}"
        )

    @classmethod
    def create(cls, name: str, **kwargs: Any):
        """Resolve the strategy class and instantiate it with *kwargs*."""
        strategy_cls = cls.resolve_class(name)
        return strategy_cls(**kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        """Return all registered strategy names."""
        return list(cls._registry.keys())
