"""
tests/test_tick_store_partitioned.py
====================================
Tests for tick store with partitioned layout and data fetcher.

Tests:
1. Missing data raises TickDataNotFoundError
2. Fetcher saves files correctly
3. TickStore loads data in order across multiple files
4. Legacy format still works
"""
import csv
import os
import tempfile
import pytest
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from Backtest.tick_store import TickStore, TickStoreConfig, TickDataNotFoundError
from Interfaces.market_data import Tick


class TestMissingDataRaisesError:
    """Test that missing data raises clear error."""
    
    def test_missing_data_raises_error(self):
        """When data is missing and allow_synthetic=False, should raise error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TickStoreConfig(
                data_dir=tmpdir,
                allow_synthetic=False
            )
            store = TickStore(config)
            
            with pytest.raises(TickDataNotFoundError) as exc_info:
                list(store.iter_ticks("BTCUSDT"))
            
            # Check error message contains helpful instructions
            error_msg = str(exc_info.value)
            assert "BTCUSDT" in error_msg
            assert "fetch_ticks.py" in error_msg
            assert "allow_synthetic" in error_msg
    
    def test_missing_data_with_synthetic_allowed(self):
        """When data is missing but allow_synthetic=True, should not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TickStoreConfig(
                data_dir=tmpdir,
                allow_synthetic=True
            )
            store = TickStore(config)
            
            # Should not raise, just return empty iterator
            ticks = list(store.iter_ticks("BTCUSDT"))
            assert ticks == []
    
    def test_error_contains_symbol_and_path(self):
        """Error message should contain symbol and data path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TickStoreConfig(data_dir=tmpdir, allow_synthetic=False)
            store = TickStore(config)
            
            try:
                list(store.iter_ticks("DOGEUSDT"))
                assert False, "Should have raised TickDataNotFoundError"
            except TickDataNotFoundError as e:
                assert e.symbol == "DOGEUSDT"
                assert tmpdir in e.data_dir


class TestPartitionedLayout:
    """Test partitioned date-based file layout."""
    
    def create_partitioned_data(self, data_dir: Path, symbol: str, dates: list):
        """Helper to create partitioned test data."""
        symbol_dir = data_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        ticks_created = []
        
        for date in dates:
            file_path = symbol_dir / f"{date.strftime('%Y-%m-%d')}.csv"
            
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp_ns', 'symbol', 'price', 'volume', 'side', 'trade_id'])
                
                # Create 10 ticks per day
                base_ts = int(date.timestamp()) * 1_000_000_000
                for i in range(10):
                    ts = base_ts + i * 1_000_000_000  # 1 second apart
                    tick = (ts, symbol, 100.0 + i * 0.01, 1.0, 'buy', i + 1)
                    writer.writerow(tick)
                    ticks_created.append(tick)
        
        return ticks_created
    
    def test_loads_partitioned_data(self):
        """Should load data from partitioned layout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            dates = [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
            ]
            
            expected_ticks = self.create_partitioned_data(data_dir, "BTCUSDT", dates)
            
            config = TickStoreConfig(data_dir=str(data_dir))
            store = TickStore(config)
            
            assert store.get_storage_layout("BTCUSDT") == "partitioned"
            assert store.file_exists("BTCUSDT")
            
            ticks = list(store.iter_ticks("BTCUSDT"))
            assert len(ticks) == 30  # 10 ticks * 3 days
    
    def test_data_in_timestamp_order(self):
        """Ticks should be yielded in timestamp order across files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            dates = [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
            ]
            
            self.create_partitioned_data(data_dir, "ETHUSDT", dates)
            
            config = TickStoreConfig(data_dir=str(data_dir))
            store = TickStore(config)
            
            ticks = list(store.iter_ticks("ETHUSDT"))
            
            # Verify timestamps are in order
            timestamps = [t.timestamp_ns for t in ticks]
            assert timestamps == sorted(timestamps), "Ticks should be in timestamp order"
    
    def test_time_range_filter(self):
        """Should filter by time range across partitioned files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            dates = [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
            ]
            
            self.create_partitioned_data(data_dir, "BTCUSDT", dates)
            
            config = TickStoreConfig(data_dir=str(data_dir))
            store = TickStore(config)
            
            # Only get ticks from day 2
            start_ts = int(datetime(2024, 1, 2).timestamp()) * 1_000_000_000
            end_ts = int(datetime(2024, 1, 3).timestamp()) * 1_000_000_000 - 1
            
            ticks = list(store.iter_ticks("BTCUSDT", start_ts, end_ts))
            assert len(ticks) == 10  # Only day 2
    
    def test_get_available_dates(self):
        """Should return list of available dates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            dates = [
                datetime(2024, 1, 1),
                datetime(2024, 1, 3),
                datetime(2024, 1, 5),
            ]
            
            self.create_partitioned_data(data_dir, "BTCUSDT", dates)
            
            config = TickStoreConfig(data_dir=str(data_dir))
            store = TickStore(config)
            
            available = store.get_available_dates("BTCUSDT")
            assert len(available) == 3
            assert available[0] == datetime(2024, 1, 1)
            assert available[1] == datetime(2024, 1, 3)
            assert available[2] == datetime(2024, 1, 5)


class TestLegacyLayout:
    """Test legacy single-file layout still works."""
    
    def create_legacy_data(self, data_dir: Path, symbol: str, num_ticks: int = 100):
        """Helper to create legacy test data."""
        data_dir.mkdir(parents=True, exist_ok=True)
        file_path = data_dir / f"{symbol}_ticks.csv"
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp_ns', 'symbol', 'price', 'volume', 'side', 'trade_id'])
            
            base_ts = int(datetime(2024, 1, 1).timestamp()) * 1_000_000_000
            for i in range(num_ticks):
                ts = base_ts + i * 1_000_000_000
                writer.writerow([ts, symbol, 100.0 + i * 0.01, 1.0, 'buy', i + 1])
    
    def test_loads_legacy_data(self):
        """Should load data from legacy single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            self.create_legacy_data(data_dir, "BTCUSDT", 50)
            
            config = TickStoreConfig(data_dir=str(data_dir))
            store = TickStore(config)
            
            assert store.get_storage_layout("BTCUSDT") == "legacy"
            assert store.file_exists("BTCUSDT")
            
            ticks = list(store.iter_ticks("BTCUSDT"))
            assert len(ticks) == 50
    
    def test_partitioned_takes_priority(self):
        """If both layouts exist, partitioned should be preferred."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            
            # Create legacy file
            self.create_legacy_data(data_dir, "BTCUSDT", 50)
            
            # Create partitioned data (should take priority)
            symbol_dir = data_dir / "BTCUSDT"
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            with open(symbol_dir / "2024-01-01.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp_ns', 'symbol', 'price', 'volume', 'side', 'trade_id'])
                base_ts = int(datetime(2024, 1, 1).timestamp()) * 1_000_000_000
                for i in range(10):
                    ts = base_ts + i * 1_000_000_000
                    writer.writerow([ts, "BTCUSDT", 200.0, 1.0, 'buy', i + 1])
            
            config = TickStoreConfig(data_dir=str(data_dir))
            store = TickStore(config)
            
            assert store.get_storage_layout("BTCUSDT") == "partitioned"
            
            ticks = list(store.iter_ticks("BTCUSDT"))
            assert len(ticks) == 10  # From partitioned, not legacy


class TestAvailableSymbols:
    """Test get_available_symbols with both layouts."""
    
    def test_finds_all_symbols(self):
        """Should find symbols in both layouts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            
            # Create legacy file
            (data_dir / "BTCUSDT_ticks.csv").write_text("timestamp_ns,symbol,price,volume\n")
            
            # Create partitioned directory
            (data_dir / "ETHUSDT").mkdir()
            (data_dir / "ETHUSDT" / "2024-01-01.csv").write_text("timestamp_ns,symbol,price,volume\n")
            
            config = TickStoreConfig(data_dir=str(data_dir))
            store = TickStore(config)
            
            symbols = store.get_available_symbols()
            assert "BTCUSDT" in symbols
            assert "ETHUSDT" in symbols


class TestDataFetcher:
    """Test data fetcher functionality."""
    
    def test_synthetic_generator(self):
        """Test synthetic tick generator."""
        from data_fetcher.binance_vision import SyntheticTickGenerator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_ticks.csv"
            
            tick_count = SyntheticTickGenerator.generate(
                symbol="TESTUSDT",
                output_path=output_path,
                num_days=1,
                seed=42
            )
            
            assert output_path.exists()
            assert tick_count == 86400  # 1 day = 86400 ticks
            
            # Verify file format
            with open(output_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                assert 'timestamp_ns' in header
                assert 'symbol' in header
                assert 'price' in header
                
                first_row = next(reader)
                assert len(first_row) >= 4
    
    def test_synthetic_determinism(self):
        """Synthetic generator should be deterministic with same seed."""
        from data_fetcher.binance_vision import SyntheticTickGenerator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "run1.csv"
            path2 = Path(tmpdir) / "run2.csv"
            
            SyntheticTickGenerator.generate("TESTUSDT", path1, num_days=1, seed=123)
            SyntheticTickGenerator.generate("TESTUSDT", path2, num_days=1, seed=123)
            
            # Files should be identical
            content1 = path1.read_text()
            content2 = path2.read_text()
            assert content1 == content2


class TestFetchConfig:
    """Test FetchConfig defaults."""
    
    def test_default_config(self):
        """FetchConfig should have sensible defaults."""
        from data_fetcher.binance_vision import FetchConfig
        
        config = FetchConfig()
        assert config.output_dir == "data/ticks"
        assert config.data_type == "aggTrades"
        assert config.market_type == "spot"
        assert config.overwrite is False
        assert config.max_retries == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
