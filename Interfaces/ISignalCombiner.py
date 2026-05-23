from abc import ABC, abstractmethod
from typing import Optional


class ISignalCombiner(ABC):
    """Interface for combining strategy signals with sentiment analysis.

    .. deprecated::
        Replaced by :class:`IMarginAdjuster`.  Kept only for backward
        compatibility with external code that may reference it.
    """

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


class IMarginAdjuster(ABC):
    """Interface for adjusting position margin based on news sentiment.

    Instead of gating trades (agree/disagree), implementations of this
    interface compute a *margin multiplier* that scales the position size
    when sentiment strongly confirms the strategy direction.
    """

    @abstractmethod
    def compute_margin_multiplier(
        self, direction: int, sentiment_score: float,
    ) -> float:
        """
        Compute a margin multiplier based on trade direction and sentiment.

        Args:
            direction: +1 for LONG, -1 for SHORT
            sentiment_score: Sentiment score from news analysis in range [0, 1]

        Returns:
            Multiplier >= 1.0.  1.0 means use the default margin;
            values > 1.0 increase the margin (and thus position size).
        """
        ...
