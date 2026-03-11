import asyncio
from typing import Dict, Optional

from Interfaces.INewsSource import INewsSource
from Interfaces.ISentimentAnalyzer import ISentimentAnalyzer
from utils.logger import setup_logger

log = setup_logger("NewsEngine")


class NewsEngine:
    """
    Orchestrates news fetching and sentiment analysis.
    Provides cached sentiment scores for symbols with configurable refresh intervals.
    """
    
    def __init__(
        self,
        news_source: INewsSource,
        sentiment_analyzer: ISentimentAnalyzer,
        refresh_interval: int = 300,  # 5 minutes default
        news_limit: int = 5
    ):
        """
        Initialize the news engine.
        
        Args:
            news_source: Implementation of INewsSource
            sentiment_analyzer: Implementation of ISentimentAnalyzer
            refresh_interval: Seconds between sentiment refreshes per symbol
            news_limit: Maximum news articles to analyze
        """
        self.news_source = news_source
        self.sentiment_analyzer = sentiment_analyzer
        self.refresh_interval = refresh_interval
        self.news_limit = news_limit
        
        # Cache: {symbol: {"score": float, "last_updated": float}}
        self._cache: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
    
    async def get_sentiment(self, symbol: str, force_refresh: bool = False) -> float:
        """
        Get sentiment score for a symbol. Uses cache if available and fresh.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            force_refresh: If True, bypass cache and fetch new data
            
        Returns:
            Sentiment score in [0, 1]
        """
        import time
        
        async with self._lock:
            current_time = time.time()
            
            # Check cache
            if not force_refresh and symbol in self._cache:
                cached = self._cache[symbol]
                age = current_time - cached["last_updated"]
                if age < self.refresh_interval:
                    log.debug("Using cached sentiment for %s: %.2f (age: %.0fs)", 
                             symbol, cached["score"], age)
                    return cached["score"]
            
            # Fetch fresh news
            try:
                articles = await self.news_source.fetch_news(symbol, limit=self.news_limit)
                
                if not articles:
                    log.warning("No news found for %s, returning neutral (0.5)", symbol)
                    score = 0.5
                else:
                    # Combine title and content for analysis
                    texts = [
                        f"{article.title}. {article.content[:200]}" 
                        for article in articles
                    ]
                    
                    score = await self.sentiment_analyzer.analyze(texts, symbol)
                    log.info("Fresh sentiment for %s: %.2f (from %d articles)", 
                            symbol, score, len(articles))
                
                # Update cache
                self._cache[symbol] = {
                    "score": score,
                    "last_updated": current_time
                }
                
                return score
                
            except Exception as e:
                log.error("Error getting sentiment for %s: %s", symbol, e)
                # Return cached value if available, otherwise neutral
                if symbol in self._cache:
                    return self._cache[symbol]["score"]
                return 0.5
    
    async def prefetch_symbols(self, symbols: list):
        """
        Prefetch sentiment for multiple symbols in parallel.
        
        Args:
            symbols: List of trading symbols
        """
        log.info("Prefetching sentiment for %d symbols", len(symbols))
        tasks = [self.get_sentiment(symbol) for symbol in symbols]
        await asyncio.gather(*tasks, return_exceptions=True)
        log.info("Sentiment prefetch complete")
    
    def get_cached_sentiment(self, symbol: str) -> Optional[float]:
        """
        Get cached sentiment without fetching new data.
        
        Returns:
            Cached score or None if not cached
        """
        if symbol in self._cache:
            return self._cache[symbol]["score"]
        return None
    
    def clear_cache(self):
        """Clear the sentiment cache."""
        self._cache.clear()
        log.info("News engine cache cleared")
