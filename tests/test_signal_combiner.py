import pytest
from news.signal_combiner import BinarySignalCombiner


class TestBinarySignalCombiner:
    """Test cases for the BinarySignalCombiner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.combiner = BinarySignalCombiner(buy_threshold=0.6, sell_threshold=0.4)
    
    # --- Both signals agree ---
    
    def test_both_buy_signals_returns_long(self):
        """Strategy BUY + Bullish sentiment → LONG."""
        result = self.combiner.combine(strategy_signal=+1, sentiment_score=0.8)
        assert result == +1
    
    def test_both_sell_signals_returns_short(self):
        """Strategy SELL + Bearish sentiment → SHORT."""
        result = self.combiner.combine(strategy_signal=-1, sentiment_score=0.2)
        assert result == -1
    
    # --- Mixed signals ---
    
    def test_buy_strategy_bearish_sentiment_returns_none(self):
        """Strategy BUY + Bearish sentiment → No action."""
        result = self.combiner.combine(strategy_signal=+1, sentiment_score=0.2)
        assert result is None
    
    def test_sell_strategy_bullish_sentiment_returns_none(self):
        """Strategy SELL + Bullish sentiment → No action."""
        result = self.combiner.combine(strategy_signal=-1, sentiment_score=0.8)
        assert result is None
    
    # --- Neutral sentiment ---
    
    def test_buy_strategy_neutral_sentiment_returns_none(self):
        """Strategy BUY + Neutral sentiment → No action."""
        result = self.combiner.combine(strategy_signal=+1, sentiment_score=0.5)
        assert result is None
    
    def test_sell_strategy_neutral_sentiment_returns_none(self):
        """Strategy SELL + Neutral sentiment → No action."""
        result = self.combiner.combine(strategy_signal=-1, sentiment_score=0.5)
        assert result is None
    
    # --- No strategy signal ---
    
    def test_no_strategy_signal_returns_none(self):
        """No strategy signal → No action regardless of sentiment."""
        assert self.combiner.combine(strategy_signal=None, sentiment_score=0.9) is None
        assert self.combiner.combine(strategy_signal=None, sentiment_score=0.1) is None
        assert self.combiner.combine(strategy_signal=None, sentiment_score=0.5) is None
    
    # --- Threshold boundary tests ---
    
    def test_sentiment_at_buy_threshold_is_neutral(self):
        """Sentiment exactly at buy threshold is neutral."""
        result = self.combiner.combine(strategy_signal=+1, sentiment_score=0.6)
        assert result is None  # 0.6 is not > 0.6
    
    def test_sentiment_above_buy_threshold_is_bullish(self):
        """Sentiment above buy threshold is bullish."""
        result = self.combiner.combine(strategy_signal=+1, sentiment_score=0.61)
        assert result == +1
    
    def test_sentiment_at_sell_threshold_is_neutral(self):
        """Sentiment exactly at sell threshold is neutral."""
        result = self.combiner.combine(strategy_signal=-1, sentiment_score=0.4)
        assert result is None  # 0.4 is not < 0.4
    
    def test_sentiment_below_sell_threshold_is_bearish(self):
        """Sentiment below sell threshold is bearish."""
        result = self.combiner.combine(strategy_signal=-1, sentiment_score=0.39)
        assert result == -1
    
    # --- String signal handling ---
    
    def test_string_buy_signal_is_handled(self):
        """String '+1' signal should be converted to int."""
        result = self.combiner.combine(strategy_signal="+1", sentiment_score=0.8)
        assert result == +1
    
    def test_string_sell_signal_is_handled(self):
        """String '-1' signal should be converted to int."""
        result = self.combiner.combine(strategy_signal="-1", sentiment_score=0.2)
        assert result == -1
    
    # --- Custom threshold tests ---
    
    def test_custom_thresholds(self):
        """Test with custom thresholds."""
        custom = BinarySignalCombiner(buy_threshold=0.7, sell_threshold=0.3)
        
        # 0.65 is bullish with default but neutral with custom
        assert self.combiner.combine(strategy_signal=+1, sentiment_score=0.65) == +1
        assert custom.combine(strategy_signal=+1, sentiment_score=0.65) is None
        
        # 0.35 is bearish with default but neutral with custom
        assert self.combiner.combine(strategy_signal=-1, sentiment_score=0.35) == -1
        assert custom.combine(strategy_signal=-1, sentiment_score=0.35) is None
