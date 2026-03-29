"""
tests/test_news_factory.py
===========================
Tests for NewsFactory: enabled/disabled, providers, missing keys.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.factories.news_factory import NewsFactory, NewsComponents
from core.bootstrap import register_defaults


def _make_news_cfg(enabled=False, provider="gemini", api_key=None):
    """Create a minimal mock LiveConfig for NewsFactory."""
    cfg = MagicMock()
    cfg.news.enabled = enabled
    cfg.news.sentiment_provider = provider
    cfg.news.api_key = api_key
    cfg.news.buy_threshold = 0.6
    cfg.news.sell_threshold = 0.4
    cfg.news.refresh_interval = 300
    return cfg


class TestNewsFactory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        NewsFactory._providers.clear()
        register_defaults()

    def test_disabled_returns_empty(self):
        cfg = _make_news_cfg(enabled=False)
        result = NewsFactory.create(cfg)
        self.assertIsInstance(result, NewsComponents)
        self.assertIsNone(result.news_source)
        self.assertIsNone(result.sentiment_analyzer)
        self.assertIsNone(result.signal_combiner)

    def test_unknown_provider_returns_empty(self):
        cfg = _make_news_cfg(enabled=True, provider="nonexistent")
        result = NewsFactory.create(cfg)
        self.assertIsNone(result.news_source)

    @patch.dict(os.environ, {"GOOGLE_API_KEY": ""}, clear=False)
    def test_missing_api_key_returns_empty(self):
        cfg = _make_news_cfg(enabled=True, provider="gemini", api_key=None)
        result = NewsFactory.create(cfg)
        self.assertIsNone(result.news_source)

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key-123"}, clear=False)
    def test_gemini_provider_creates_components(self):
        cfg = _make_news_cfg(enabled=True, provider="gemini", api_key=None)
        result = NewsFactory.create(cfg)
        self.assertIsNotNone(result.news_source)
        self.assertIsNotNone(result.sentiment_analyzer)
        self.assertIsNotNone(result.signal_combiner)

    def test_explicit_api_key_used(self):
        cfg = _make_news_cfg(enabled=True, provider="gemini", api_key="explicit-key")
        result = NewsFactory.create(cfg)
        self.assertIsNotNone(result.news_source)
        self.assertIsNotNone(result.sentiment_analyzer)

    def test_custom_provider_registration(self):
        def _custom_creator(api_key):
            return MagicMock(), MagicMock()

        NewsFactory.register_provider("custom_test", _custom_creator)
        cfg = _make_news_cfg(enabled=True, provider="custom_test", api_key="key")
        result = NewsFactory.create(cfg)
        self.assertIsNotNone(result.news_source)
        self.assertIsNotNone(result.signal_combiner)
        # Cleanup
        del NewsFactory._providers["custom_test"]


if __name__ == "__main__":
    unittest.main()
