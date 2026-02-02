from abc import ABC, abstractmethod
from typing import List


class ISentimentAnalyzer(ABC):
    """Interface for sentiment analysis engines. Implementations can use various LLMs."""
    
    @abstractmethod
    async def analyze(self, texts: List[str], symbol: str) -> float:
        """
        Analyze sentiment of given texts for a trading symbol.
        
        Args:
            texts: List of text content (news headlines/articles) to analyze
            symbol: Trading symbol for context (e.g., "BTCUSDT")
            
        Returns:
            Sentiment score in range [0, 1]:
            - 0.0 = Very bearish (strong sell signal)
            - 0.5 = Neutral
            - 1.0 = Very bullish (strong buy signal)
        """
        ...
