from typing import Optional

from Interfaces.ISignalCombiner import ISignalCombiner, IMarginAdjuster
from utils.logger import setup_logger

log = setup_logger("SignalCombiner")


class BinarySignalCombiner(ISignalCombiner):
    """
    Combines strategy signals with sentiment analysis using binary logic.

    .. deprecated::
        Replaced by :class:`SentimentMarginAdjuster`.  Kept for backward
        compatibility only.

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


class SentimentMarginAdjuster(IMarginAdjuster):
    """
    Adjusts position margin based on news sentiment analysis.

    Instead of combining signals to gate trades (agree/disagree), this class
    computes a *margin multiplier* that increases the position size when
    sentiment strongly confirms the strategy direction:

    - LONG  + strongly bullish news  → margin increased (up to margin_boost_max)
    - SHORT + strongly bearish news  → margin increased (up to margin_boost_max)
    - Neutral / weak / contrary news → default margin (multiplier = 1.0)

    The boost scales linearly from 1.0 at the threshold to
    ``1 + margin_boost_max`` at the extreme (score = 1.0 or 0.0).
    """

    def __init__(
        self,
        high_sentiment_threshold: float = 0.75,
        low_sentiment_threshold: float = 0.25,
        margin_boost_max: float = 0.5,
    ):
        """
        Initialize the margin adjuster.

        Args:
            high_sentiment_threshold: Sentiment score above this triggers
                a bullish margin boost for LONG positions (default: 0.75).
            low_sentiment_threshold: Sentiment score below this triggers
                a bearish margin boost for SHORT positions (default: 0.25).
            margin_boost_max: Maximum fractional increase applied to the
                base margin.  0.5 means +50% at peak sentiment
                (e.g. $10 → $15).  Default: 0.5.
        """
        self.high_sentiment_threshold = high_sentiment_threshold
        self.low_sentiment_threshold = low_sentiment_threshold
        self.margin_boost_max = margin_boost_max

    def compute_margin_multiplier(
        self, direction: int, sentiment_score: float,
    ) -> float:
        """
        Compute margin multiplier from trade direction and sentiment score.

        Args:
            direction: +1 for LONG, -1 for SHORT
            sentiment_score: Score in [0, 1] from news sentiment analysis.

        Returns:
            A float >= 1.0.
            - 1.0 means use the default margin (no boost).
            - Values > 1.0 linearly scale the margin up.
        """
        # Clamp sentiment to [0, 1] for safety
        sentiment_score = max(0.0, min(1.0, sentiment_score))

        # ── LONG direction: boost when sentiment is strongly bullish ──
        if direction == +1 and sentiment_score > self.high_sentiment_threshold:
            # Distance above threshold, normalised to [0, 1]
            headroom = 1.0 - self.high_sentiment_threshold
            if headroom <= 0:
                return 1.0
            excess = sentiment_score - self.high_sentiment_threshold
            ratio = excess / headroom  # 0 → 1
            multiplier = 1.0 + self.margin_boost_max * ratio
            log.info(
                "LONG + bullish sentiment (%.2f) → margin multiplier %.2f",
                sentiment_score, multiplier,
            )
            return multiplier

        # ── SHORT direction: boost when sentiment is strongly bearish ──
        if direction == -1 and sentiment_score < self.low_sentiment_threshold:
            headroom = self.low_sentiment_threshold  # distance from 0 to threshold
            if headroom <= 0:
                return 1.0
            excess = self.low_sentiment_threshold - sentiment_score
            ratio = excess / headroom  # 0 → 1
            multiplier = 1.0 + self.margin_boost_max * ratio
            log.info(
                "SHORT + bearish sentiment (%.2f) → margin multiplier %.2f",
                sentiment_score, multiplier,
            )
            return multiplier

        # ── No boost ──
        log.info(
            "Sentiment (%.2f) does not strongly confirm direction (%+d) → no margin boost",
            sentiment_score, direction,
        )
        return 1.0
