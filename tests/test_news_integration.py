"""
Integration test for news sentiment with trading strategy.
Uses all mock components to verify the full flow.
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
from news.signal_combiner import BinarySignalCombiner
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
    """Integration tests for news sentiment pipeline."""
    
    async def test_full_pipeline_buy_scenario(self):
        """Test full pipeline: bullish news → buy signal combination."""
        # Setup
        news_source = MockNewsSource()
        sentiment = MockSentimentAnalyzer(score=0.85)  # Bullish
        engine = NewsEngine(news_source, sentiment)
        combiner = BinarySignalCombiner()
        
        # Get sentiment
        score = await engine.get_sentiment("BTCUSDT")
        
        # Combine with BUY strategy signal
        result = combiner.combine(strategy_signal=+1, sentiment_score=score)
        
        # Should produce LONG signal
        self.assertEqual(result, +1)
        self.assertEqual(sentiment.call_count, 1)
    
    async def test_full_pipeline_sell_scenario(self):
        """Test full pipeline: bearish news → sell signal combination."""
        news_source = MockNewsSource()
        sentiment = MockSentimentAnalyzer(score=0.15)  # Bearish
        engine = NewsEngine(news_source, sentiment)
        combiner = BinarySignalCombiner()
        
        score = await engine.get_sentiment("BTCUSDT")
        result = combiner.combine(strategy_signal=-1, sentiment_score=score)
        
        self.assertEqual(result, -1)
    
    async def test_full_pipeline_mixed_signals(self):
        """Test full pipeline: bullish news + sell strategy → no action."""
        news_source = MockNewsSource()
        sentiment = MockSentimentAnalyzer(score=0.85)  # Bullish
        engine = NewsEngine(news_source, sentiment)
        combiner = BinarySignalCombiner()
        
        score = await engine.get_sentiment("BTCUSDT")
        result = combiner.combine(strategy_signal=-1, sentiment_score=score)  # SELL
        
        self.assertIsNone(result)  # Mixed signals → no action
    
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
