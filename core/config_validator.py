"""
core/config_validator.py
========================
Validates live trading config files before use.
"""
from __future__ import annotations

import logging
import os
from typing import List

from core.factories.strategy_factory import StrategyFactory

log = logging.getLogger(__name__)


class ConfigValidator:
    """Validate a YAML/JSON live config file and report errors."""

    def validate(self, config_path: str) -> List[str]:
        """
        Validate the given config file.

        Returns:
            List of error strings. Empty list means the config is valid.
        """
        errors: List[str] = []

        # File existence
        if not os.path.exists(config_path):
            return [f"Config file not found: {config_path}"]

        # Parse
        try:
            from live.live_config import LiveConfig
            if config_path.endswith(".json"):
                cfg = LiveConfig.from_json(config_path)
            else:
                cfg = LiveConfig.from_yaml(config_path)
        except Exception as e:
            return [f"Failed to parse config: {e}"]

        # Strategy
        if not cfg.strategy_class:
            errors.append("strategy.class is missing or empty")
        else:
            try:
                StrategyFactory.resolve_class(cfg.strategy_class)
            except (ValueError, ImportError, AttributeError) as e:
                errors.append(
                    f"Cannot resolve strategy '{cfg.strategy_class}': {e}. "
                    f"Available: {StrategyFactory.list_available()}"
                )

        # Symbols
        if not cfg.symbols:
            errors.append("symbols list is empty")

        # Timeframe
        if not cfg.timeframe:
            errors.append("timeframe is missing or empty")

        # Sizing
        sizing = cfg.sizing
        if sizing.leverage < 1:
            errors.append(f"sizing.leverage must be >= 1, got {sizing.leverage}")
        if sizing.margin_usd <= 0:
            errors.append(f"sizing.margin_usd must be > 0, got {sizing.margin_usd}")
        valid_modes = {"fixed_qty", "notional_usd", "margin_usd"}
        if sizing.mode not in valid_modes:
            errors.append(f"sizing.mode must be one of {valid_modes}, got '{sizing.mode}'")

        # Risk
        risk = cfg.risk
        if risk.max_concurrent_positions < 1:
            errors.append(f"risk.max_concurrent_positions must be >= 1, got {risk.max_concurrent_positions}")
        if risk.max_daily_loss <= 0:
            errors.append(f"risk.max_daily_loss must be > 0, got {risk.max_daily_loss}")

        # News (if enabled)
        if cfg.news.enabled:
            provider = cfg.news.sentiment_provider.lower()
            from core.factories.news_factory import NewsFactory
            if provider not in NewsFactory._providers:
                errors.append(
                    f"Unknown news sentiment provider: '{provider}'. "
                    f"Available: {list(NewsFactory._providers.keys())}"
                )

        return errors
