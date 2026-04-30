"""
core/bootstrap.py
=================
Composition root — registers all default strategies and news providers.

Call ``register_defaults()`` once at application startup before using factories.
"""
from __future__ import annotations

from core.factories.strategy_factory import StrategyFactory
from core.factories.news_factory import NewsFactory
from core.factories.regime_factory import RegimeFactory


def _create_gemini_components(api_key: str):
    from news.ddg_news_source import DDGNewsSource
    from news.gemini_sentiment import GeminiSentimentAnalyzer
    return DDGNewsSource(), GeminiSentimentAnalyzer(api_key=api_key)


def _create_openai_components(api_key: str):
    from news.ddg_news_source import DDGNewsSource
    from news.openai_sentiment import OpenAISentimentAnalyzer
    return DDGNewsSource(), OpenAISentimentAnalyzer(api_key=api_key)


def register_defaults() -> None:
    """Register all built-in strategies and news sentiment providers."""
    # Strategies
    StrategyFactory.register("RSIThreshold", "Strategy.RSIThreshold")
    StrategyFactory.register("EMACrossMACDTrend", "Strategy.EMACrossMACDTrend")
    StrategyFactory.register("DonchianATRVolTarget", "Strategy.DonchianATRVolTarget")

    # News sentiment providers
    NewsFactory.register_provider("gemini", _create_gemini_components)
    NewsFactory.register_provider("openai", _create_openai_components)

    # Regime detectors (Phase 2)
    RegimeFactory.register("ADXRegimeDetector", "Strategy.regime.adx_regime.ADXRegimeDetector")
    RegimeFactory.register("ATRPercentileRegime", "Strategy.regime.atr_vol_regime.ATRPercentileRegime")
    RegimeFactory.register("CompositeRegimeDetector",
                           "Strategy.regime.composite_regime.CompositeRegimeDetector")
