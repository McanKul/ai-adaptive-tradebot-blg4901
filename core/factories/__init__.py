"""
core.factories
==============
Factory classes for creating trading system components.
"""
from core.factories.strategy_factory import StrategyFactory
from core.factories.broker_factory import BrokerFactory
from core.factories.news_factory import NewsFactory, NewsComponents

__all__ = ["StrategyFactory", "BrokerFactory", "NewsFactory", "NewsComponents"]
