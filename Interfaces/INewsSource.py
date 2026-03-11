from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass


@dataclass
class NewsArticle:
    """Represents a single news article for sentiment analysis."""
    title: str
    content: str
    source: str
    published_at: str
    symbol: str  # e.g., "BTCUSDT" or "BTC"


class INewsSource(ABC):
    """Interface for news data sources. Implementations can fetch from various APIs."""
    
    @abstractmethod
    async def fetch_news(self, symbol: str, limit: int = 10) -> List[NewsArticle]:
        """
        Fetch recent news articles for a given symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT" or "BTC")
            limit: Maximum number of articles to return
            
        Returns:
            List of NewsArticle objects
        """
        ...
