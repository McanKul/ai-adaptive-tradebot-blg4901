from typing import Optional

from Interfaces.ISignalCombiner import ISignalCombiner
from utils.logger import setup_logger

log = setup_logger("SignalCombiner")


class BinarySignalCombiner(ISignalCombiner):
    """
    Combines strategy signals with sentiment analysis using binary logic.
    
    Both signals must agree for a position to be opened:
    - Strategy BUY + Sentiment BUY (>threshold) = LONG
    - Strategy SELL + Sentiment SELL (<threshold) = SHORT
    - Mixed signals = NO ACTION
    """
    
    def __init__(self, buy_threshold: float = 0.6, sell_threshold: float = 0.4):
        """
        Initialize the signal combiner.
        
        Args:
            buy_threshold: Sentiment score above this is considered bullish (default: 0.6)
            sell_threshold: Sentiment score below this is considered bearish (default: 0.4)
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
    
    def combine(self, strategy_signal: Optional[int], sentiment_score: float) -> Optional[int]:
        """
        Combine strategy signal with sentiment score.
        
        Args:
            strategy_signal: +1 (buy), -1 (sell), or None (no signal)
            sentiment_score: Score in [0, 1]
            
        Returns:
            +1 for long, -1 for short, None for no action
        """
        # No strategy signal = no action
        if strategy_signal is None:
            log.info("No strategy signal, no action taken")
            return None
        
        # Convert strategy signal to int if it's a string
        if isinstance(strategy_signal, str):
            strategy_signal = int(strategy_signal)
        
        # Determine sentiment signal
        if sentiment_score > self.buy_threshold:
            sentiment_signal = +1  # Bullish
        elif sentiment_score < self.sell_threshold:
            sentiment_signal = -1  # Bearish
        else:
            sentiment_signal = 0   # Neutral
            log.info("Neutral sentiment (%.2f), no action taken", sentiment_score)
            return None
        
        # Combine signals
        if strategy_signal == +1 and sentiment_signal == +1:
            log.info("BUY + Bullish sentiment (%.2f) → LONG position", sentiment_score)
            return +1
        elif strategy_signal == -1 and sentiment_signal == -1:
            log.info("SELL + Bearish sentiment (%.2f) → SHORT position", sentiment_score)
            return -1
        else:
            log.info(
                "Mixed signals: strategy=%d, sentiment=%d (%.2f) → No action",
                strategy_signal, sentiment_signal, sentiment_score
            )
            return None
