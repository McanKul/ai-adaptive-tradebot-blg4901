from abc import ABC, abstractmethod
from typing import Optional


class ISignalCombiner(ABC):
    """Interface for combining strategy signals with sentiment analysis."""
    
    @abstractmethod
    def combine(self, strategy_signal: Optional[int], sentiment_score: float) -> Optional[int]:
        """
        Combine a strategy signal with a sentiment score to produce a final trading signal.
        
        Args:
            strategy_signal: Signal from technical strategy (+1 for buy, -1 for sell, None for no signal)
            sentiment_score: Sentiment score from news analysis in range [0, 1]
            
        Returns:
            Combined trading signal:
            - +1: Open long position (both signals agree on buy)
            - -1: Open short position (both signals agree on sell)
            - None: No action (signals disagree or are neutral)
        """
        ...
