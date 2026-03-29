"""
core/factories/news_factory.py
===============================
Factory for creating news sentiment components.

Replaces the create_news_components() function from live_runner.py.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional

from Interfaces.INewsSource import INewsSource
from Interfaces.ISentimentAnalyzer import ISentimentAnalyzer
from Interfaces.ISignalCombiner import ISignalCombiner

log = logging.getLogger(__name__)


@dataclass
class NewsComponents:
    """Container for the three news-sentiment objects."""
    news_source: Optional[INewsSource] = None
    sentiment_analyzer: Optional[ISentimentAnalyzer] = None
    signal_combiner: Optional[ISignalCombiner] = None


class NewsFactory:
    """
    Registry-based factory for news sentiment providers.

    Each registered provider is a callable that receives an API key
    and returns ``(INewsSource, ISentimentAnalyzer)``.
    """

    # name -> callable(api_key) -> (INewsSource, ISentimentAnalyzer)
    _providers: dict[str, Callable] = {}

    @classmethod
    def register_provider(
        cls,
        name: str,
        creator: Callable[[str], tuple],
    ) -> None:
        """Register a sentiment provider factory."""
        cls._providers[name] = creator
        log.debug("Registered news provider: %s", name)

    @classmethod
    def create(cls, cfg) -> NewsComponents:
        """
        Create news components from a ``LiveConfig``.

        Returns an all-None ``NewsComponents`` when news is disabled,
        the API key is missing, or the provider is unknown.
        """
        news_cfg = cfg.news

        if not news_cfg.enabled:
            log.info("News sentiment disabled by config")
            return NewsComponents()

        provider = news_cfg.sentiment_provider.lower()
        creator = cls._providers.get(provider)

        if creator is None:
            log.warning("Unknown sentiment provider: %s", provider)
            return NewsComponents()

        # Resolve API key from config or environment
        api_key = news_cfg.api_key
        if not api_key:
            env_map = {"gemini": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY"}
            env_var = env_map.get(provider, "")
            api_key = os.getenv(env_var, "") if env_var else ""

        if not api_key:
            log.warning(
                "%s sentiment enabled but no API key found (config or env)",
                provider,
            )
            return NewsComponents()

        news_source, sentiment_analyzer = creator(api_key)

        from news.signal_combiner import BinarySignalCombiner
        signal_combiner = BinarySignalCombiner(
            buy_threshold=news_cfg.buy_threshold,
            sell_threshold=news_cfg.sell_threshold,
        )

        log.info(
            "News sentiment enabled: provider=%s, refresh=%ds, "
            "thresholds=(buy=%.2f, sell=%.2f)",
            provider, news_cfg.refresh_interval,
            news_cfg.buy_threshold, news_cfg.sell_threshold,
        )

        return NewsComponents(
            news_source=news_source,
            sentiment_analyzer=sentiment_analyzer,
            signal_combiner=signal_combiner,
        )
