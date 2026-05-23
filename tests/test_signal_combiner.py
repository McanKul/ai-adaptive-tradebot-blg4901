import pytest
from news.signal_combiner import SentimentMarginAdjuster


class TestSentimentMarginAdjuster:
    """Test cases for the SentimentMarginAdjuster class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adjuster = SentimentMarginAdjuster(
            high_sentiment_threshold=0.75,
            low_sentiment_threshold=0.25,
            margin_boost_max=0.5,
        )

    # --- LONG + bullish sentiment (boost) ---

    def test_long_strongly_bullish_returns_boost(self):
        """LONG + very high sentiment → margin multiplier > 1.0."""
        result = self.adjuster.compute_margin_multiplier(direction=+1, sentiment_score=0.9)
        assert result > 1.0
        assert result <= 1.5  # max boost is 50%

    def test_long_max_bullish_returns_max_boost(self):
        """LONG + sentiment = 1.0 → maximum margin multiplier."""
        result = self.adjuster.compute_margin_multiplier(direction=+1, sentiment_score=1.0)
        assert result == pytest.approx(1.5, abs=0.01)

    def test_long_at_threshold_returns_no_boost(self):
        """LONG + sentiment exactly at threshold → no boost (1.0)."""
        result = self.adjuster.compute_margin_multiplier(direction=+1, sentiment_score=0.75)
        assert result == 1.0

    def test_long_above_threshold_returns_boost(self):
        """LONG + sentiment just above threshold → slight boost."""
        result = self.adjuster.compute_margin_multiplier(direction=+1, sentiment_score=0.76)
        assert result > 1.0
        assert result < 1.1  # small boost for barely above threshold

    # --- SHORT + bearish sentiment (boost) ---

    def test_short_strongly_bearish_returns_boost(self):
        """SHORT + very low sentiment → margin multiplier > 1.0."""
        result = self.adjuster.compute_margin_multiplier(direction=-1, sentiment_score=0.1)
        assert result > 1.0
        assert result <= 1.5

    def test_short_max_bearish_returns_max_boost(self):
        """SHORT + sentiment = 0.0 → maximum margin multiplier."""
        result = self.adjuster.compute_margin_multiplier(direction=-1, sentiment_score=0.0)
        assert result == pytest.approx(1.5, abs=0.01)

    def test_short_at_threshold_returns_no_boost(self):
        """SHORT + sentiment exactly at threshold → no boost."""
        result = self.adjuster.compute_margin_multiplier(direction=-1, sentiment_score=0.25)
        assert result == 1.0

    def test_short_below_threshold_returns_boost(self):
        """SHORT + sentiment just below threshold → slight boost."""
        result = self.adjuster.compute_margin_multiplier(direction=-1, sentiment_score=0.24)
        assert result > 1.0
        assert result < 1.1

    # --- Neutral sentiment (no boost) ---

    def test_long_neutral_sentiment_no_boost(self):
        """LONG + neutral sentiment → no boost."""
        result = self.adjuster.compute_margin_multiplier(direction=+1, sentiment_score=0.5)
        assert result == 1.0

    def test_short_neutral_sentiment_no_boost(self):
        """SHORT + neutral sentiment → no boost."""
        result = self.adjuster.compute_margin_multiplier(direction=-1, sentiment_score=0.5)
        assert result == 1.0

    # --- Contrary sentiment (no boost, no reduction) ---

    def test_long_bearish_sentiment_no_change(self):
        """LONG + bearish sentiment → no boost (not reduced either)."""
        result = self.adjuster.compute_margin_multiplier(direction=+1, sentiment_score=0.1)
        assert result == 1.0

    def test_short_bullish_sentiment_no_change(self):
        """SHORT + bullish sentiment → no boost (not reduced either)."""
        result = self.adjuster.compute_margin_multiplier(direction=-1, sentiment_score=0.9)
        assert result == 1.0

    # --- Linear scaling verification ---

    def test_linear_scaling_long(self):
        """Verify linear scaling for LONG positions."""
        # At threshold (0.75) → 1.0
        # At 0.875 (midpoint between 0.75 and 1.0) → 1.25
        # At 1.0 → 1.5
        mid = self.adjuster.compute_margin_multiplier(direction=+1, sentiment_score=0.875)
        assert mid == pytest.approx(1.25, abs=0.01)

    def test_linear_scaling_short(self):
        """Verify linear scaling for SHORT positions."""
        # At threshold (0.25) → 1.0
        # At 0.125 (midpoint between 0.0 and 0.25) → 1.25
        # At 0.0 → 1.5
        mid = self.adjuster.compute_margin_multiplier(direction=-1, sentiment_score=0.125)
        assert mid == pytest.approx(1.25, abs=0.01)

    # --- Custom thresholds ---

    def test_custom_thresholds(self):
        """Test with custom thresholds and boost."""
        custom = SentimentMarginAdjuster(
            high_sentiment_threshold=0.80,
            low_sentiment_threshold=0.20,
            margin_boost_max=1.0,  # 100% boost
        )

        # Max bullish → 2.0x margin
        result = custom.compute_margin_multiplier(direction=+1, sentiment_score=1.0)
        assert result == pytest.approx(2.0, abs=0.01)

        # Max bearish → 2.0x margin
        result = custom.compute_margin_multiplier(direction=-1, sentiment_score=0.0)
        assert result == pytest.approx(2.0, abs=0.01)

        # Score that's bullish with default but not with custom
        # 0.76 is above 0.75 (default) but below 0.80 (custom)
        assert self.adjuster.compute_margin_multiplier(direction=+1, sentiment_score=0.76) > 1.0
        assert custom.compute_margin_multiplier(direction=+1, sentiment_score=0.76) == 1.0

    # --- Edge cases ---

    def test_sentiment_clamped_above_one(self):
        """Sentiment scores > 1.0 should be clamped."""
        result = self.adjuster.compute_margin_multiplier(direction=+1, sentiment_score=1.5)
        assert result == pytest.approx(1.5, abs=0.01)  # same as score=1.0

    def test_sentiment_clamped_below_zero(self):
        """Sentiment scores < 0.0 should be clamped."""
        result = self.adjuster.compute_margin_multiplier(direction=-1, sentiment_score=-0.5)
        assert result == pytest.approx(1.5, abs=0.01)  # same as score=0.0

    def test_result_always_at_least_one(self):
        """Multiplier should never be below 1.0."""
        for direction in [+1, -1]:
            for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
                result = self.adjuster.compute_margin_multiplier(direction, score)
                assert result >= 1.0, f"direction={direction}, score={score} → {result}"
