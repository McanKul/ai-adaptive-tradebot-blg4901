"""
tests/test_tick_store_dedup.py
==============================
Tests for tick store deduplication using (timestamp_ns, trade_id) key.

These tests verify the AFML-aligned deduplication approach:
- Correct key: (timestamp_ns, trade_id) - NOT (timestamp, price, volume)
- Proper handling of duplicate entries
- Memory-efficient sorting for out-of-order data
"""
import csv
import tempfile
from datetime import datetime
from pathlib import Path
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from Backtest.tick_store import TickStore, TickStoreConfig


class TestDeduplicationKey:
    """Test that deduplication uses correct (timestamp_ns, trade_id) key."""
    
    def create_csv_with_duplicates(self, file_path: Path, ticks: list):
        """Helper to create CSV with specified ticks (may contain duplicates)."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp_ns', 'symbol', 'price', 'volume', 'side', 'trade_id'])
            for tick in ticks:
                writer.writerow(tick)
    
    def test_exact_duplicate_removed(self):
        """Exact duplicate rows (same timestamp + trade_id) should be removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            file_path = data_dir / "BTCUSDT" / "2024-01-01.csv"
            
            # Same tick appears twice
            ticks = [
                (1000000000, 'BTCUSDT', 100.0, 1.0, 'buy', 12345),
                (1000000000, 'BTCUSDT', 100.0, 1.0, 'buy', 12345),  # Exact duplicate
                (2000000000, 'BTCUSDT', 101.0, 2.0, 'sell', 12346),
            ]
            self.create_csv_with_duplicates(file_path, ticks)
            
            config = TickStoreConfig(data_dir=str(data_dir), deduplicate=True)
            store = TickStore(config)
            
            result = list(store.iter_ticks("BTCUSDT"))
            assert len(result) == 2, "Duplicate should be removed"
    
    def test_same_timestamp_different_trade_id_kept(self):
        """
        Same timestamp but different trade_id should be kept.
        This is correct behavior - multiple trades CAN happen at same nanosecond.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            file_path = data_dir / "BTCUSDT" / "2024-01-01.csv"
            
            # Same timestamp, different trade IDs (valid - different trades)
            ticks = [
                (1000000000, 'BTCUSDT', 100.0, 1.0, 'buy', 12345),
                (1000000000, 'BTCUSDT', 100.1, 2.0, 'sell', 12346),  # Different trade_id
                (1000000000, 'BTCUSDT', 100.2, 0.5, 'buy', 12347),   # Different trade_id
            ]
            self.create_csv_with_duplicates(file_path, ticks)
            
            config = TickStoreConfig(data_dir=str(data_dir), deduplicate=True)
            store = TickStore(config)
            
            result = list(store.iter_ticks("BTCUSDT"))
            assert len(result) == 3, "Different trade IDs at same timestamp should ALL be kept"
    
    def test_same_trade_id_different_timestamp_kept(self):
        """
        Same trade_id but different timestamp should be kept.
        (This shouldn't happen in practice but tests the key correctly)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            file_path = data_dir / "BTCUSDT" / "2024-01-01.csv"
            
            ticks = [
                (1000000000, 'BTCUSDT', 100.0, 1.0, 'buy', 12345),
                (2000000000, 'BTCUSDT', 100.1, 2.0, 'sell', 12345),  # Same trade_id, different time
            ]
            self.create_csv_with_duplicates(file_path, ticks)
            
            config = TickStoreConfig(data_dir=str(data_dir), deduplicate=True)
            store = TickStore(config)
            
            result = list(store.iter_ticks("BTCUSDT"))
            assert len(result) == 2, "Different timestamps should be kept even with same trade_id"
    
    def test_dedup_disabled_keeps_all(self):
        """When deduplicate=False, all rows should be kept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            file_path = data_dir / "BTCUSDT" / "2024-01-01.csv"
            
            ticks = [
                (1000000000, 'BTCUSDT', 100.0, 1.0, 'buy', 12345),
                (1000000000, 'BTCUSDT', 100.0, 1.0, 'buy', 12345),  # Exact duplicate
                (1000000000, 'BTCUSDT', 100.0, 1.0, 'buy', 12345),  # Triple
            ]
            self.create_csv_with_duplicates(file_path, ticks)
            
            config = TickStoreConfig(data_dir=str(data_dir), deduplicate=False)
            store = TickStore(config)
            
            result = list(store.iter_ticks("BTCUSDT"))
            assert len(result) == 3, "With dedup disabled, all rows should be kept"


class TestOrdering:
    """Test that ticks are properly ordered by timestamp."""
    
    def create_csv_with_data(self, file_path: Path, ticks: list):
        """Helper to create CSV with specified ticks."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp_ns', 'symbol', 'price', 'volume', 'side', 'trade_id'])
            for tick in ticks:
                writer.writerow(tick)
    
    def test_out_of_order_sorted(self):
        """Out-of-order ticks should be sorted when sort_if_needed=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            file_path = data_dir / "BTCUSDT" / "2024-01-01.csv"
            
            # Ticks are out of order (using nanoseconds)
            base_ts = int(datetime(2024, 1, 1).timestamp()) * 1_000_000_000
            ticks = [
                (base_ts + 3_000_000_000, 'BTCUSDT', 103.0, 3.0, 'buy', 3),
                (base_ts + 1_000_000_000, 'BTCUSDT', 101.0, 1.0, 'buy', 1),
                (base_ts + 2_000_000_000, 'BTCUSDT', 102.0, 2.0, 'sell', 2),
            ]
            self.create_csv_with_data(file_path, ticks)
            
            config = TickStoreConfig(data_dir=str(data_dir), sort_if_needed=True)
            store = TickStore(config)
            
            result = list(store.iter_ticks("BTCUSDT"))
            timestamps = [t.timestamp_ns for t in result]
            
            expected = sorted([base_ts + 1_000_000_000, base_ts + 2_000_000_000, base_ts + 3_000_000_000])
            assert timestamps == expected, "Should be sorted"
    
    def test_sort_disabled_preserves_order(self):
        """When sort_if_needed=False and data is out of order, should raise error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            file_path = data_dir / "BTCUSDT" / "2024-01-01.csv"
            
            # Ticks are out of order (using nanoseconds)
            base_ts = int(datetime(2024, 1, 1).timestamp()) * 1_000_000_000
            ticks = [
                (base_ts + 3_000_000_000, 'BTCUSDT', 103.0, 3.0, 'buy', 3),
                (base_ts + 1_000_000_000, 'BTCUSDT', 101.0, 1.0, 'buy', 1),
                (base_ts + 2_000_000_000, 'BTCUSDT', 102.0, 2.0, 'sell', 2),
            ]
            self.create_csv_with_data(file_path, ticks)
            
            config = TickStoreConfig(data_dir=str(data_dir), sort_if_needed=False)
            store = TickStore(config)
            
            # Should raise ValueError because data is out of order and sort is disabled
            with pytest.raises(ValueError, match="out-of-order"):
                list(store.iter_ticks("BTCUSDT"))
    
    def test_already_sorted_data(self):
        """Already sorted data should pass through efficiently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            file_path = data_dir / "BTCUSDT" / "2024-01-01.csv"
            
            # Ticks are already in order (using nanoseconds)
            base_ts = int(datetime(2024, 1, 1).timestamp()) * 1_000_000_000
            ticks = [
                (base_ts + 1_000_000_000, 'BTCUSDT', 101.0, 1.0, 'buy', 1),
                (base_ts + 2_000_000_000, 'BTCUSDT', 102.0, 2.0, 'sell', 2),
                (base_ts + 3_000_000_000, 'BTCUSDT', 103.0, 3.0, 'buy', 3),
            ]
            self.create_csv_with_data(file_path, ticks)
            
            config = TickStoreConfig(data_dir=str(data_dir), sort_if_needed=True)
            store = TickStore(config)
            
            result = list(store.iter_ticks("BTCUSDT"))
            timestamps = [t.timestamp_ns for t in result]
            
            expected = [base_ts + 1_000_000_000, base_ts + 2_000_000_000, base_ts + 3_000_000_000]
            assert timestamps == expected


class TestCombinedDedupAndSort:
    """Test deduplication and sorting work together."""
    
    def create_csv_with_data(self, file_path: Path, ticks: list):
        """Helper to create CSV with specified ticks."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp_ns', 'symbol', 'price', 'volume', 'side', 'trade_id'])
            for tick in ticks:
                writer.writerow(tick)
    
    def test_dedup_and_sort_combined(self):
        """Should deduplicate and sort in one pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            file_path = data_dir / "BTCUSDT" / "2024-01-01.csv"
            
            # Out of order with duplicates (using nanoseconds)
            base_ts = int(datetime(2024, 1, 1).timestamp()) * 1_000_000_000
            ticks = [
                (base_ts + 3_000_000_000, 'BTCUSDT', 103.0, 3.0, 'buy', 3),
                (base_ts + 1_000_000_000, 'BTCUSDT', 101.0, 1.0, 'buy', 1),
                (base_ts + 3_000_000_000, 'BTCUSDT', 103.0, 3.0, 'buy', 3),  # Duplicate
                (base_ts + 2_000_000_000, 'BTCUSDT', 102.0, 2.0, 'sell', 2),
                (base_ts + 1_000_000_000, 'BTCUSDT', 101.0, 1.0, 'buy', 1),  # Duplicate
            ]
            self.create_csv_with_data(file_path, ticks)
            
            config = TickStoreConfig(
                data_dir=str(data_dir), 
                deduplicate=True,
                sort_if_needed=True
            )
            store = TickStore(config)
            
            result = list(store.iter_ticks("BTCUSDT"))
            
            assert len(result) == 3, "Should have 3 unique ticks"
            timestamps = [t.timestamp_ns for t in result]
            expected = [base_ts + 1_000_000_000, base_ts + 2_000_000_000, base_ts + 3_000_000_000]
            assert timestamps == expected, "Should be sorted"
    
    def test_real_world_scenario(self):
        """
        Real-world scenario: multiple trades at same timestamp (aggregated trades)
        plus some duplicates from multiple data sources.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            file_path = data_dir / "ETHUSDT" / "2024-01-01.csv"
            
            # Using realistic nanosecond timestamps
            base_ts = int(datetime(2024, 1, 1).timestamp()) * 1_000_000_000
            t1 = base_ts + 1_000_000_000
            t2 = base_ts + 2_000_000_000
            t3 = base_ts + 3_000_000_000
            
            ticks = [
                # Multiple trades at exact same timestamp (normal for exchange)
                (t1, 'ETHUSDT', 2500.00, 1.0, 'buy', 1001),
                (t1, 'ETHUSDT', 2500.01, 0.5, 'buy', 1002),
                (t1, 'ETHUSDT', 2500.02, 2.0, 'sell', 1003),
                
                # Later trades
                (t2, 'ETHUSDT', 2501.00, 1.0, 'buy', 1004),
                (t3, 'ETHUSDT', 2500.50, 1.5, 'sell', 1005),
                
                # Duplicates (e.g., from overlapping data downloads)
                (t1, 'ETHUSDT', 2500.00, 1.0, 'buy', 1001),  # Duplicate
                (t2, 'ETHUSDT', 2501.00, 1.0, 'buy', 1004),  # Duplicate
            ]
            self.create_csv_with_data(file_path, ticks)
            
            config = TickStoreConfig(
                data_dir=str(data_dir),
                deduplicate=True,
                sort_if_needed=True
            )
            store = TickStore(config)
            
            result = list(store.iter_ticks("ETHUSDT"))
            
            # Should have 5 unique ticks (3 at t=1, 1 at t=2, 1 at t=3)
            assert len(result) == 5
            
            # Check trades at same timestamp are preserved
            t1_trades = [t for t in result if t.timestamp_ns == t1]
            assert len(t1_trades) == 3, "All 3 different trades at same timestamp should be kept"
            
            # Verify trade IDs
            trade_ids = {t.trade_id for t in result}
            assert trade_ids == {1001, 1002, 1003, 1004, 1005}


class TestMaxSortSize:
    """Test max_sort_size limit."""
    
    def create_csv_with_data(self, file_path: Path, num_ticks: int, reverse_order: bool = True):
        """Helper to create CSV with many ticks."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        base_ts = int(datetime(2024, 1, 1).timestamp()) * 1_000_000_000
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp_ns', 'symbol', 'price', 'volume', 'side', 'trade_id'])
            for i in range(num_ticks):
                # Reverse order to force sorting
                if reverse_order:
                    ts = base_ts + (num_ticks - i) * 1_000_000_000
                else:
                    ts = base_ts + i * 1_000_000_000
                writer.writerow([ts, 'BTCUSDT', 100.0 + i, 1.0, 'buy', i])
    
    def test_exceeds_max_sort_size_error(self):
        """Should raise error when file exceeds max_sort_size and needs sorting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            file_path = data_dir / "BTCUSDT" / "2024-01-01.csv"
            
            # Create 100 ticks out of order but set max to 50
            self.create_csv_with_data(file_path, 100, reverse_order=True)
            
            config = TickStoreConfig(
                data_dir=str(data_dir),
                sort_if_needed=True,
                max_sort_size=50  # Limit to 50 ticks
            )
            store = TickStore(config)
            
            # Should raise ValueError because it needs sorting but exceeds limit
            with pytest.raises(ValueError, match="too large to sort"):
                list(store.iter_ticks("BTCUSDT"))
    
    def test_within_max_sort_size_works(self):
        """Should sort successfully when within max_sort_size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            file_path = data_dir / "BTCUSDT" / "2024-01-01.csv"
            
            # Create 50 ticks out of order, max is 100
            self.create_csv_with_data(file_path, 50, reverse_order=True)
            
            config = TickStoreConfig(
                data_dir=str(data_dir),
                sort_if_needed=True,
                max_sort_size=100
            )
            store = TickStore(config)
            
            result = list(store.iter_ticks("BTCUSDT"))
            assert len(result) == 50
            
            # Verify sorted
            timestamps = [t.timestamp_ns for t in result]
            assert timestamps == sorted(timestamps)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
