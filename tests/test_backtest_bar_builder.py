"""
tests/test_backtest_bar_builder.py
==================================
Tests for bar builders (TimeBarBuilder, etc.).

Verifies:
- Time bars close at correct boundaries
- OHLCV values are computed correctly
- Different bar types work as expected
- Flush handles partial bars
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Interfaces.market_data import Tick, Bar
from Backtest.bar_builder import (
    TimeBarBuilder, TickBarBuilder, VolumeBarBuilder, DollarBarBuilder,
    create_bar_builder, TIMEFRAME_NS
)


class TestTimeBarBuilder:
    """Tests for TimeBarBuilder."""
    
    def test_emits_bar_at_boundary(self):
        """Test that bar is emitted when crossing time boundary."""
        builder = TimeBarBuilder("BTCUSDT", "1m")
        
        # 1 minute = 60 seconds = 60_000_000_000 nanoseconds
        ns_per_minute = 60_000_000_000
        
        # First bar: ticks at t=0, 30, 59 seconds
        bar = builder.on_tick(Tick("BTCUSDT", 0, 50000.0, 1.0))
        assert bar is None  # Not complete yet
        
        bar = builder.on_tick(Tick("BTCUSDT", 30_000_000_000, 50100.0, 0.5))
        assert bar is None
        
        bar = builder.on_tick(Tick("BTCUSDT", 59_000_000_000, 49900.0, 0.3))
        assert bar is None
        
        # Tick at t=60s should close the first bar
        bar = builder.on_tick(Tick("BTCUSDT", ns_per_minute, 50200.0, 0.2))
        
        assert bar is not None
        assert bar.symbol == "BTCUSDT"
        assert bar.timeframe == "1m"
        assert bar.open == 50000.0
        assert bar.high == 50100.0
        assert bar.low == 49900.0
        assert bar.close == 49900.0  # Last tick in bar
        assert bar.volume == 1.8  # 1.0 + 0.5 + 0.3
        assert bar.tick_count == 3
        assert bar.closed == True
    
    def test_ohlcv_calculation(self):
        """Test correct OHLCV calculation."""
        builder = TimeBarBuilder("BTCUSDT", "1m")
        ns_per_minute = 60_000_000_000
        
        # Ticks: open=100, high=120, low=90, close=110
        builder.on_tick(Tick("BTCUSDT", 0, 100.0, 1.0))  # Open
        builder.on_tick(Tick("BTCUSDT", 10_000_000_000, 90.0, 0.5))  # Low
        builder.on_tick(Tick("BTCUSDT", 20_000_000_000, 120.0, 0.3))  # High
        builder.on_tick(Tick("BTCUSDT", 50_000_000_000, 110.0, 0.2))  # Close
        
        # Close bar
        bar = builder.on_tick(Tick("BTCUSDT", ns_per_minute, 105.0, 0.1))
        
        assert bar.open == 100.0
        assert bar.high == 120.0
        assert bar.low == 90.0
        assert bar.close == 110.0
        assert bar.volume == pytest.approx(2.0)  # 1.0 + 0.5 + 0.3 + 0.2
    
    def test_multiple_bars(self):
        """Test building multiple consecutive bars."""
        builder = TimeBarBuilder("BTCUSDT", "1m")
        ns_per_minute = 60_000_000_000
        
        bars = []
        
        # Bar 1 tick
        builder.on_tick(Tick("BTCUSDT", 0, 100.0, 1.0))
        
        # Bar 2 tick closes bar 1
        bar = builder.on_tick(Tick("BTCUSDT", ns_per_minute, 101.0, 1.0))
        if bar:
            bars.append(bar)
        
        # Bar 3 tick closes bar 2
        bar = builder.on_tick(Tick("BTCUSDT", 2 * ns_per_minute, 102.0, 1.0))
        if bar:
            bars.append(bar)
        
        assert len(bars) == 2
        assert bars[0].close == 100.0
        assert bars[1].close == 101.0
    
    def test_flush_partial_bar(self):
        """Test flushing a partial bar."""
        builder = TimeBarBuilder("BTCUSDT", "1m")
        
        builder.on_tick(Tick("BTCUSDT", 0, 100.0, 1.0))
        builder.on_tick(Tick("BTCUSDT", 10_000_000_000, 105.0, 0.5))
        
        assert builder.has_partial
        
        bar = builder.flush()
        
        assert bar is not None
        assert bar.open == 100.0
        assert bar.close == 105.0
        assert bar.volume == 1.5
        assert not builder.has_partial
    
    def test_5m_bars(self):
        """Test 5-minute bars."""
        builder = TimeBarBuilder("BTCUSDT", "5m")
        ns_per_5min = 5 * 60_000_000_000
        
        builder.on_tick(Tick("BTCUSDT", 0, 100.0, 1.0))
        
        # Tick at 1min shouldn't close bar
        bar = builder.on_tick(Tick("BTCUSDT", 60_000_000_000, 101.0, 1.0))
        assert bar is None
        
        # Tick at 5min should close bar
        bar = builder.on_tick(Tick("BTCUSDT", ns_per_5min, 102.0, 1.0))
        assert bar is not None
        assert bar.timeframe == "5m"


class TestTickBarBuilder:
    """Tests for TickBarBuilder."""
    
    def test_emits_after_threshold(self):
        """Test bar emitted after tick count threshold."""
        builder = TickBarBuilder("BTCUSDT", tick_threshold=3)
        
        bar = builder.on_tick(Tick("BTCUSDT", 1000, 100.0, 1.0))
        assert bar is None
        
        bar = builder.on_tick(Tick("BTCUSDT", 2000, 101.0, 1.0))
        assert bar is None
        
        bar = builder.on_tick(Tick("BTCUSDT", 3000, 102.0, 1.0))
        assert bar is None
        
        # 4th tick should close the 3-tick bar
        bar = builder.on_tick(Tick("BTCUSDT", 4000, 103.0, 1.0))
        
        assert bar is not None
        assert bar.tick_count == 3
        assert bar.close == 102.0


class TestVolumeBarBuilder:
    """Tests for VolumeBarBuilder."""
    
    def test_emits_after_volume_threshold(self):
        """Test bar emitted after volume threshold."""
        builder = VolumeBarBuilder("BTCUSDT", volume_threshold=5.0)
        
        bar = builder.on_tick(Tick("BTCUSDT", 1000, 100.0, 2.0))
        assert bar is None
        
        bar = builder.on_tick(Tick("BTCUSDT", 2000, 101.0, 2.0))
        assert bar is None
        
        bar = builder.on_tick(Tick("BTCUSDT", 3000, 102.0, 1.0))  # Total: 5.0
        assert bar is None
        
        # Next tick closes the bar
        bar = builder.on_tick(Tick("BTCUSDT", 4000, 103.0, 0.5))
        
        assert bar is not None
        assert bar.volume == 5.0


class TestDollarBarBuilder:
    """Tests for DollarBarBuilder."""
    
    def test_emits_after_dollar_threshold(self):
        """Test bar emitted after dollar volume threshold."""
        builder = DollarBarBuilder("BTCUSDT", dollar_threshold=1000.0)
        
        # Tick 1: $500 (100 * 5)
        bar = builder.on_tick(Tick("BTCUSDT", 1000, 100.0, 5.0))
        assert bar is None
        
        # Tick 2: $500 more (100 * 5)
        bar = builder.on_tick(Tick("BTCUSDT", 2000, 100.0, 5.0))
        assert bar is None
        
        # Tick 3 closes bar
        bar = builder.on_tick(Tick("BTCUSDT", 3000, 100.0, 1.0))
        
        assert bar is not None
        assert bar.dollar_volume == pytest.approx(1000.0)


class TestBarBuilderFactory:
    """Tests for bar builder factory function."""
    
    def test_create_time_bar_builder(self):
        """Test creating time bar builder."""
        builder = create_bar_builder("BTCUSDT", bar_type="time", timeframe="5m")
        assert isinstance(builder, TimeBarBuilder)
    
    def test_create_tick_bar_builder(self):
        """Test creating tick bar builder."""
        builder = create_bar_builder("BTCUSDT", bar_type="tick", tick_threshold=50)
        assert isinstance(builder, TickBarBuilder)
    
    def test_create_volume_bar_builder(self):
        """Test creating volume bar builder."""
        builder = create_bar_builder("BTCUSDT", bar_type="volume", volume_threshold=100)
        assert isinstance(builder, VolumeBarBuilder)
    
    def test_invalid_bar_type(self):
        """Test error on invalid bar type."""
        with pytest.raises(ValueError, match="Unknown bar type"):
            create_bar_builder("BTCUSDT", bar_type="invalid")
    
    def test_invalid_timeframe(self):
        """Test error on invalid timeframe."""
        with pytest.raises(ValueError, match="Unknown timeframe"):
            TimeBarBuilder("BTCUSDT", "invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
