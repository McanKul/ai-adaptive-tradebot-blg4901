"""
tests/test_backtest_disk_streamer.py
====================================
Tests for DiskTickStreamer and TickStore.

Verifies:
- Ticks are yielded in timestamp order
- Multiple symbols are merged correctly
- Time range filtering works
- Reset functionality works
"""
import sys
import os
import tempfile
import pytest

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Interfaces.market_data import Tick
from Backtest.tick_store import TickStore, TickStoreConfig
from Backtest.disk_streamer import DiskTickStreamer, DiskStreamerConfig, SingleSymbolDiskStreamer


def create_tick_csv(filepath: str, symbol: str, ticks: list):
    """Helper to create a tick CSV file."""
    with open(filepath, 'w', newline='') as f:
        f.write("timestamp_ns,symbol,price,volume\n")
        for ts, price, volume in ticks:
            f.write(f"{ts},{symbol},{price},{volume}\n")


class TestTickStore:
    """Tests for TickStore."""
    
    def test_load_ticks_from_csv(self, tmp_path):
        """Test loading ticks from CSV file."""
        # Create test data
        tick_file = tmp_path / "BTCUSDT_ticks.csv"
        create_tick_csv(str(tick_file), "BTCUSDT", [
            (1000000000000, 50000.0, 1.0),
            (1000001000000, 50001.0, 0.5),
            (1000002000000, 50002.0, 0.3),
        ])
        
        config = TickStoreConfig(data_dir=str(tmp_path))
        store = TickStore(config)
        
        ticks = list(store.iter_ticks("BTCUSDT"))
        
        assert len(ticks) == 3
        assert ticks[0].price == 50000.0
        assert ticks[1].price == 50001.0
        assert ticks[2].price == 50002.0
    
    def test_time_range_filter(self, tmp_path):
        """Test filtering ticks by time range."""
        tick_file = tmp_path / "BTCUSDT_ticks.csv"
        # Use realistic nanosecond timestamps (2024-01-01 00:00:XX)
        # 1704067200 = 2024-01-01 00:00:00 UTC in seconds
        base_ns = 1704067200_000_000_000  # 2024-01-01 in nanoseconds
        create_tick_csv(str(tick_file), "BTCUSDT", [
            (base_ns + 1_000_000_000, 50000.0, 1.0),  # +1 sec
            (base_ns + 2_000_000_000, 50001.0, 0.5),  # +2 sec
            (base_ns + 3_000_000_000, 50002.0, 0.3),  # +3 sec
            (base_ns + 4_000_000_000, 50003.0, 0.2),  # +4 sec
        ])
        
        config = TickStoreConfig(data_dir=str(tmp_path))
        store = TickStore(config)
        
        # Filter middle range: 1.5s to 3.5s
        ticks = list(store.iter_ticks(
            "BTCUSDT",
            start_ts_ns=base_ns + 1_500_000_000,
            end_ts_ns=base_ns + 3_500_000_000
        ))
        
        assert len(ticks) == 2
        assert ticks[0].price == 50001.0
        assert ticks[1].price == 50002.0
    
    def test_file_not_found(self, tmp_path):
        """Test handling of missing file - should raise TickDataNotFoundError."""
        from Backtest.tick_store import TickDataNotFoundError
        
        config = TickStoreConfig(data_dir=str(tmp_path))
        store = TickStore(config)
        
        with pytest.raises(TickDataNotFoundError):
            list(store.iter_ticks("NONEXISTENT"))


class TestDiskTickStreamer:
    """Tests for DiskTickStreamer."""
    
    def test_single_symbol_streaming(self, tmp_path):
        """Test streaming single symbol."""
        tick_file = tmp_path / "BTCUSDT_ticks.csv"
        create_tick_csv(str(tick_file), "BTCUSDT", [
            (1000000000000, 50000.0, 1.0),
            (1000001000000, 50001.0, 0.5),
            (1000002000000, 50002.0, 0.3),
        ])
        
        store = TickStore(TickStoreConfig(data_dir=str(tmp_path)))
        config = DiskStreamerConfig(symbols=["BTCUSDT"])
        streamer = DiskTickStreamer(store, config)
        
        ticks = list(streamer.iter_ticks())
        
        assert len(ticks) == 3
        assert all(t.symbol == "BTCUSDT" for t in ticks)
    
    def test_multi_symbol_merging_in_time_order(self, tmp_path):
        """Test that multiple symbols are merged in timestamp order."""
        # BTC ticks at t=1, 3, 5
        btc_file = tmp_path / "BTCUSDT_ticks.csv"
        create_tick_csv(str(btc_file), "BTCUSDT", [
            (1000000000000, 50000.0, 1.0),
            (3000000000000, 50002.0, 0.5),
            (5000000000000, 50004.0, 0.3),
        ])
        
        # ETH ticks at t=2, 4
        eth_file = tmp_path / "ETHUSDT_ticks.csv"
        create_tick_csv(str(eth_file), "ETHUSDT", [
            (2000000000000, 3000.0, 2.0),
            (4000000000000, 3001.0, 1.5),
        ])
        
        store = TickStore(TickStoreConfig(data_dir=str(tmp_path)))
        config = DiskStreamerConfig(symbols=["BTCUSDT", "ETHUSDT"])
        streamer = DiskTickStreamer(store, config)
        
        ticks = list(streamer.iter_ticks())
        
        # Should be 5 ticks total, in time order
        assert len(ticks) == 5
        
        # Verify time ordering
        timestamps = [t.timestamp_ns for t in ticks]
        assert timestamps == sorted(timestamps)
        
        # Verify interleaving
        symbols = [t.symbol for t in ticks]
        assert symbols == ["BTCUSDT", "ETHUSDT", "BTCUSDT", "ETHUSDT", "BTCUSDT"]
    
    def test_deterministic_ordering(self, tmp_path):
        """Test that ordering is deterministic across multiple runs."""
        btc_file = tmp_path / "BTCUSDT_ticks.csv"
        create_tick_csv(str(btc_file), "BTCUSDT", [
            (1000000000000, 50000.0, 1.0),
            (2000000000000, 50001.0, 0.5),
        ])
        
        eth_file = tmp_path / "ETHUSDT_ticks.csv"
        create_tick_csv(str(eth_file), "ETHUSDT", [
            (1000000000000, 3000.0, 2.0),  # Same timestamp as BTC
            (2000000000000, 3001.0, 1.5),
        ])
        
        store = TickStore(TickStoreConfig(data_dir=str(tmp_path)))
        config = DiskStreamerConfig(symbols=["BTCUSDT", "ETHUSDT"])
        
        # Run multiple times
        results = []
        for _ in range(3):
            streamer = DiskTickStreamer(store, config)
            ticks = list(streamer.iter_ticks())
            results.append([(t.timestamp_ns, t.symbol) for t in ticks])
        
        # All runs should produce identical results
        assert results[0] == results[1] == results[2]
    
    def test_reset(self, tmp_path):
        """Test reset functionality."""
        tick_file = tmp_path / "BTCUSDT_ticks.csv"
        create_tick_csv(str(tick_file), "BTCUSDT", [
            (1000000000000, 50000.0, 1.0),
            (2000000000000, 50001.0, 0.5),
        ])
        
        store = TickStore(TickStoreConfig(data_dir=str(tmp_path)))
        config = DiskStreamerConfig(symbols=["BTCUSDT"])
        streamer = DiskTickStreamer(store, config)
        
        # First iteration
        ticks1 = list(streamer.iter_ticks())
        assert len(ticks1) == 2
        
        # Second iteration without reset - should be empty
        ticks2 = list(streamer.iter_ticks())
        assert len(ticks2) == 0
        
        # After reset
        streamer.reset()
        ticks3 = list(streamer.iter_ticks())
        assert len(ticks3) == 2


class TestSingleSymbolStreamer:
    """Tests for SingleSymbolDiskStreamer."""
    
    def test_single_symbol_streamer(self, tmp_path):
        """Test simplified single-symbol streamer."""
        tick_file = tmp_path / "BTCUSDT_ticks.csv"
        create_tick_csv(str(tick_file), "BTCUSDT", [
            (1000000000000, 50000.0, 1.0),
            (2000000000000, 50001.0, 0.5),
            (3000000000000, 50002.0, 0.3),
        ])
        
        store = TickStore(TickStoreConfig(data_dir=str(tmp_path)))
        streamer = SingleSymbolDiskStreamer(store, "BTCUSDT")
        
        ticks = list(streamer.iter_ticks())
        
        assert len(ticks) == 3
        assert streamer.total_ticks_yielded == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
