"""
Integration test for news sentiment with trading strategy.
Uses all mock components to verify the full margin-adjustment flow.
"""
import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

# Mock binance before importing modules that use it
sys.modules["binance"] = MagicMock()
sys.modules["binance.client"] = MagicMock()
sys.modules["binance.exceptions"] = MagicMock()
sys.modules["binance.enums"] = MagicMock()

sys.modules["binance.enums"].SIDE_BUY = "BUY"
sys.modules["binance.enums"].SIDE_SELL = "SELL"
sys.modules["binance.enums"].FUTURE_ORDER_TYPE_MARKET = "MARKET"
sys.modules["binance.enums"].FUTURE_ORDER_TYPE_STOP_MARKET = "STOP_MARKET"
sys.modules["binance.enums"].FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from news.crypto_news_source import MockNewsSource
from news.signal_combiner import SentimentMarginAdjuster
from news.news_engine import NewsEngine
from Interfaces.ISentimentAnalyzer import ISentimentAnalyzer


class MockSentimentAnalyzer(ISentimentAnalyzer):
    """Mock sentiment analyzer that returns predefined scores."""

    def __init__(self, score: float = 0.5):
        self.score = score
        self.call_count = 0

    async def analyze(self, texts, symbol):
        self.call_count += 1
        return self.score


class TestNewsIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for news sentiment → margin adjustment pipeline."""

    async def test_full_pipeline_long_bullish_boosts_margin(self):
        """LONG + strongly bullish news → margin multiplier > 1.0."""
        # Setup
        news_source = MockNewsSource()
        sentiment = MockSentimentAnalyzer(score=0.90)  # Strongly bullish
        engine = NewsEngine(news_source, sentiment)
        adjuster = SentimentMarginAdjuster()

        # Get sentiment
        score = await engine.get_sentiment("BTCUSDT")

        # Compute margin multiplier for LONG
        multiplier = adjuster.compute_margin_multiplier(direction=+1, sentiment_score=score)

        # Should boost margin
        self.assertGreater(multiplier, 1.0)
        self.assertEqual(sentiment.call_count, 1)

    async def test_full_pipeline_short_bearish_boosts_margin(self):
        """SHORT + strongly bearish news → margin multiplier > 1.0."""
        news_source = MockNewsSource()
        sentiment = MockSentimentAnalyzer(score=0.10)  # Strongly bearish
        engine = NewsEngine(news_source, sentiment)
        adjuster = SentimentMarginAdjuster()

        score = await engine.get_sentiment("BTCUSDT")
        multiplier = adjuster.compute_margin_multiplier(direction=-1, sentiment_score=score)

        self.assertGreater(multiplier, 1.0)

    async def test_full_pipeline_neutral_sentiment_no_boost(self):
        """Neutral sentiment → margin multiplier = 1.0 (no boost)."""
        news_source = MockNewsSource()
        sentiment = MockSentimentAnalyzer(score=0.50)  # Neutral
        engine = NewsEngine(news_source, sentiment)
        adjuster = SentimentMarginAdjuster()

        score = await engine.get_sentiment("BTCUSDT")
        multiplier = adjuster.compute_margin_multiplier(direction=+1, sentiment_score=score)

        self.assertEqual(multiplier, 1.0)  # No boost for neutral

    async def test_full_pipeline_contrary_sentiment_no_reduction(self):
        """LONG + bearish sentiment → multiplier stays 1.0 (no reduction)."""
        news_source = MockNewsSource()
        sentiment = MockSentimentAnalyzer(score=0.10)  # Bearish
        engine = NewsEngine(news_source, sentiment)
        adjuster = SentimentMarginAdjuster()

        score = await engine.get_sentiment("BTCUSDT")
        multiplier = adjuster.compute_margin_multiplier(direction=+1, sentiment_score=score)

        self.assertEqual(multiplier, 1.0)  # No reduction, just default

    async def test_caching_prevents_duplicate_api_calls(self):
        """Verify that caching prevents redundant sentiment API calls."""
        news_source = MockNewsSource()
        sentiment = MockSentimentAnalyzer(score=0.7)
        engine = NewsEngine(news_source, sentiment, refresh_interval=300)

        # Multiple calls to same symbol
        await engine.get_sentiment("BTCUSDT")
        await engine.get_sentiment("BTCUSDT")
        await engine.get_sentiment("BTCUSDT")

        # Should only analyze once due to caching
        self.assertEqual(sentiment.call_count, 1)

    async def test_force_refresh_bypasses_cache(self):
        """Force refresh should bypass the cache."""
        news_source = MockNewsSource()
        sentiment = MockSentimentAnalyzer(score=0.7)
        engine = NewsEngine(news_source, sentiment, refresh_interval=300)

        await engine.get_sentiment("BTCUSDT")
        await engine.get_sentiment("BTCUSDT", force_refresh=True)

        self.assertEqual(sentiment.call_count, 2)

    async def test_prefetch_multiple_symbols(self):
        """Prefetch should analyze multiple symbols."""
        news_source = MockNewsSource()
        sentiment = MockSentimentAnalyzer(score=0.5)
        engine = NewsEngine(news_source, sentiment)

        await engine.prefetch_symbols(["BTCUSDT", "ETHUSDT"])

        # Should have analyzed both (though ETH may return 0.5 due to no mock news)
        self.assertIsNotNone(engine.get_cached_sentiment("BTCUSDT"))

    async def test_unknown_symbol_returns_neutral(self):
        """Unknown symbol with no news should return neutral sentiment."""
        news_source = MockNewsSource()
        sentiment = MockSentimentAnalyzer(score=0.5)
        engine = NewsEngine(news_source, sentiment)

        # Patch to return empty for unknown symbol
        score = await engine.get_sentiment("UNKNOWNUSDT")

        # MockSentimentAnalyzer returns 0.5, or if no articles, engine returns 0.5
        self.assertAlmostEqual(score, 0.5, places=1)


if __name__ == '__main__':
    unittest.main()
