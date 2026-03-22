"""
Backtest/tick_store.py
======================
Tick storage and retrieval from disk.

Design decisions:
- Supports CSV format (primary), designed for easy Parquet extension
- Validates tick ordering (warns on out-of-order)
- Optional deduplication of identical ticks
- Basic gap detection (counts gaps)
- Memory-efficient iteration (doesn't load all ticks at once for large files)

Supports TWO storage layouts:
1. LEGACY: Single file per symbol: {data_dir}/{SYMBOL}_ticks.csv
2. PARTITIONED: Files by date: {data_dir}/{SYMBOL}/YYYY-MM-DD.csv

The loader automatically detects which layout is in use.

CSV Format expected:
    timestamp_ns,symbol,price,volume[,side][,trade_id]
    
    OR with headers:
    timestamp_ns,symbol,price,volume,side,trade_id
"""
from __future__ import annotations
import csv
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional, List, Dict, Any, Callable
import logging

from Interfaces.market_data import Tick

log = logging.getLogger(__name__)


class TickDataNotFoundError(Exception):
    """Raised when required tick data is not found."""
    
    def __init__(self, symbol: str, data_dir: str, message: str = None):
        self.symbol = symbol
        self.data_dir = data_dir
        
        if message is None:
            message = self._build_message()
        
        super().__init__(message)
    
    def _build_message(self) -> str:
        return f"""
================================================================================
ERROR: No tick data found for {self.symbol}
================================================================================

Tick data is missing. To run a backtest, you need real historical data.

OPTION 1: Download real data from Binance Vision (recommended):
    python tools/fetch_ticks.py --symbol {self.symbol} --start 2024-01-01 --end 2024-01-07

OPTION 2: Use synthetic data for testing ONLY (not recommended for real backtests):
    Set allow_synthetic=True in config OR use --synthetic flag

Expected data location: {self.data_dir}/{self.symbol}/
    OR legacy format: {self.data_dir}/{self.symbol}_ticks.csv

For more information, see docs/data_setup.md
================================================================================
"""


@dataclass
class TickStoreConfig:
    """Configuration for TickStore."""
    data_dir: str  # Base directory containing tick data files
    file_pattern: str = "{symbol}_ticks.csv"  # Pattern for LEGACY tick files
    validate_order: bool = True  # Validate ticks are in time order
    deduplicate: bool = True  # Remove duplicate ticks (by timestamp + trade_id)
    sort_if_needed: bool = True  # Sort ticks if out of order (small files only)
    max_sort_size: int = 10_000_000  # Max ticks to sort in memory
    gap_threshold_ns: int = 60_000_000_000  # 60 seconds - gaps larger than this are counted
    allow_synthetic: bool = False  # Allow synthetic data generation (dev/test only)


class TickStore:
    """
    Loads and iterates tick data from disk.
    
    Supports both LEGACY (single file) and PARTITIONED (date-based) layouts.
    Automatically detects which format is available.
    
    Usage:
        store = TickStore(config)
        for tick in store.iter_ticks("BTCUSDT", start_ts, end_ts):
            process(tick)
    """
    
    def __init__(self, config: TickStoreConfig):
        """
        Initialize TickStore.
        
        Args:
            config: TickStoreConfig with data directory and options
        """
        self.config = config
        self.data_dir = Path(config.data_dir)
        self._file_cache: Dict[str, Path] = {}
        self._gap_counts: Dict[str, int] = {}
        
        if not self.data_dir.exists():
            log.warning(f"Data directory does not exist: {self.data_dir}")
    
    def get_legacy_file_path(self, symbol: str) -> Path:
        """Get the LEGACY file path for a symbol's tick data."""
        filename = self.config.file_pattern.format(symbol=symbol)
        return self.data_dir / filename
    
    def get_file_path(self, symbol: str) -> Path:
        """Get the file path for a symbol's tick data (legacy compatibility)."""
        return self.get_legacy_file_path(symbol)
    
    def get_partitioned_dir(self, symbol: str) -> Path:
        """Get the PARTITIONED directory for a symbol."""
        return self.data_dir / symbol.upper()
    
    def get_storage_layout(self, symbol: str) -> str:
        """
        Detect which storage layout is available for a symbol.
        
        Returns:
            'partitioned': Date-partitioned files in {symbol}/ directory
            'legacy': Single {symbol}_ticks.csv file
            'none': No data found
        """
        # Check partitioned first (preferred)
        partitioned_dir = self.get_partitioned_dir(symbol)
        if partitioned_dir.exists():
            csv_files = list(partitioned_dir.glob("*.csv"))
            if csv_files:
                return "partitioned"
        
        # Check legacy
        legacy_path = self.get_legacy_file_path(symbol)
        if legacy_path.exists():
            return "legacy"
        
        return "none"
    
    def file_exists(self, symbol: str) -> bool:
        """Check if tick data exists for symbol (any layout)."""
        return self.get_storage_layout(symbol) != "none"
    
    def get_available_dates(self, symbol: str) -> List[datetime]:
        """
        Get list of available dates for a symbol (partitioned layout only).
        
        Returns sorted list of dates with data.
        """
        partitioned_dir = self.get_partitioned_dir(symbol)
        if not partitioned_dir.exists():
            return []
        
        dates = []
        for f in partitioned_dir.glob("*.csv"):
            try:
                date_str = f.stem  # e.g., "2024-01-15"
                date = datetime.strptime(date_str, "%Y-%m-%d")
                dates.append(date)
            except ValueError:
                continue
        
        return sorted(dates)
    
    def iter_ticks(
        self,
        symbol: str,
        start_ts_ns: Optional[int] = None,
        end_ts_ns: Optional[int] = None
    ) -> Iterator[Tick]:
        """
        Iterate over ticks for a symbol within time range.
        
        Automatically handles both legacy and partitioned layouts.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            start_ts_ns: Start timestamp in nanoseconds (inclusive), None for beginning
            end_ts_ns: End timestamp in nanoseconds (inclusive), None for end
            
        Yields:
            Tick objects in timestamp order
            
        Raises:
            TickDataNotFoundError: If no data found and allow_synthetic=False
        """
        layout = self.get_storage_layout(symbol)
        
        if layout == "none":
            if not self.config.allow_synthetic:
                raise TickDataNotFoundError(symbol, str(self.data_dir))
            else:
                log.warning(f"No tick data for {symbol}, allow_synthetic=True but no data to iterate")
                return
        
        if layout == "partitioned":
            yield from self._iter_partitioned(symbol, start_ts_ns, end_ts_ns)
        else:
            # Legacy single file
            file_path = self.get_legacy_file_path(symbol)
            yield from self._iter_csv(file_path, symbol, start_ts_ns, end_ts_ns)
    
    def _iter_partitioned(
        self,
        symbol: str,
        start_ts_ns: Optional[int],
        end_ts_ns: Optional[int]
    ) -> Iterator[Tick]:
        """Iterate over partitioned date files."""
        partitioned_dir = self.get_partitioned_dir(symbol)
        
        # Get all date files
        date_files = sorted(partitioned_dir.glob("*.csv"))
        
        if not date_files:
            log.warning(f"No CSV files found in {partitioned_dir}")
            return
        
        # Filter files by date range if timestamps provided
        if start_ts_ns is not None or end_ts_ns is not None:
            filtered_files = []
            for f in date_files:
                try:
                    file_date = datetime.strptime(f.stem, "%Y-%m-%d")
                    file_start_ns = int(file_date.timestamp()) * 1_000_000_000
                    file_end_ns = file_start_ns + 86400 * 1_000_000_000  # +1 day
                    
                    # Check overlap with requested range
                    if end_ts_ns is not None and file_start_ns > end_ts_ns:
                        continue
                    if start_ts_ns is not None and file_end_ns < start_ts_ns:
                        continue
                    
                    filtered_files.append(f)
                except ValueError:
                    # Include non-date files just in case
                    filtered_files.append(f)
            
            date_files = filtered_files
        
        log.debug(f"Loading {len(date_files)} date files for {symbol}")
        
        # Iterate through files in order
        for file_path in date_files:
            yield from self._iter_csv(file_path, symbol, start_ts_ns, end_ts_ns)
    
    def _iter_csv(
        self,
        file_path: Path,
        symbol: str,
        start_ts_ns: Optional[int],
        end_ts_ns: Optional[int]
    ) -> Iterator[Tick]:
        """
        Iterate over CSV tick file.
        
        Handles multiple CSV formats:
        1. With headers: timestamp_ns,symbol,price,volume,side,trade_id
        2. Without headers: assumed column order
        3. Minimal: timestamp_ns,price,volume (symbol from filename)
        
        Deduplication: Uses (timestamp_ns, trade_id) as unique key.
        Ordering: Validates non-decreasing timestamps; sorts if needed.
        """
        last_ts: Optional[int] = None
        seen_keys: set = set()  # For deduplication: (timestamp_ns, trade_id)
        gap_count = 0
        out_of_order_count = 0
        duplicate_count = 0
        
        # First pass: collect ticks and check ordering
        ticks_buffer: List[Tick] = []
        needs_sort = False
        
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            # Detect if first line is header
            first_line = f.readline().strip()
            f.seek(0)
            
            has_header = 'timestamp' in first_line.lower() or 'price' in first_line.lower()
            
            reader = csv.reader(f)
            if has_header:
                headers = next(reader)
                col_map = {h.lower().strip(): i for i, h in enumerate(headers)}
            else:
                # Assume standard order
                col_map = {
                    'timestamp_ns': 0, 'symbol': 1, 'price': 2,
                    'volume': 3, 'side': 4, 'trade_id': 5
                }
            
            for row in reader:
                if not row or len(row) < 3:
                    continue
                
                try:
                    tick = self._parse_row(row, col_map, symbol)
                except (ValueError, IndexError) as e:
                    log.warning(f"Failed to parse row: {row}, error: {e}")
                    continue
                
                # Time filter
                if start_ts_ns is not None and tick.timestamp_ns < start_ts_ns:
                    continue
                if end_ts_ns is not None and tick.timestamp_ns > end_ts_ns:
                    continue
                
                # Deduplication check using (timestamp_ns, trade_id)
                if self.config.deduplicate:
                    dedup_key = (tick.timestamp_ns, tick.trade_id)
                    if dedup_key in seen_keys:
                        duplicate_count += 1
                        continue
                    seen_keys.add(dedup_key)
                
                # Ordering check
                if self.config.validate_order and last_ts is not None:
                    if tick.timestamp_ns < last_ts:
                        out_of_order_count += 1
                        needs_sort = True
                    
                    # Gap detection
                    gap = tick.timestamp_ns - last_ts
                    if gap > self.config.gap_threshold_ns:
                        gap_count += 1
                
                last_ts = tick.timestamp_ns
                ticks_buffer.append(tick)
        
        # Handle out-of-order data
        if needs_sort and out_of_order_count > 0:
            if len(ticks_buffer) <= self.config.max_sort_size and self.config.sort_if_needed:
                log.warning(
                    f"Symbol {symbol}: {out_of_order_count} out-of-order ticks found. "
                    f"Sorting {len(ticks_buffer)} ticks in memory."
                )
                ticks_buffer.sort(key=lambda t: t.timestamp_ns)
            else:
                raise ValueError(
                    f"Symbol {symbol}: {out_of_order_count} out-of-order ticks found. "
                    f"Data too large to sort in memory ({len(ticks_buffer)} ticks > {self.config.max_sort_size}). "
                    f"Please sort the CSV file externally or fix the data source."
                )
        
        # Log summary
        if duplicate_count > 0:
            log.info(f"Symbol {symbol}: removed {duplicate_count} duplicate ticks")
        if gap_count > 0:
            log.info(f"Symbol {symbol}: detected {gap_count} gaps > {self.config.gap_threshold_ns/1e9:.0f}s")
        if out_of_order_count > 0 and not needs_sort:
            log.warning(f"Symbol {symbol}: {out_of_order_count} out-of-order ticks (not sorted)")
        
        self._gap_counts[symbol] = gap_count
        
        # Yield ticks in order
        for tick in ticks_buffer:
            yield tick
    
    def _parse_row(
        self,
        row: List[str],
        col_map: Dict[str, int],
        default_symbol: str
    ) -> Tick:
        """Parse a CSV row into a Tick object."""
        def get_col(name: str, default=None):
            idx = col_map.get(name)
            if idx is not None and idx < len(row):
                val = row[idx].strip()
                return val if val else default
            return default
        
        # Parse timestamp - handle both ns and ms
        ts_raw = get_col('timestamp_ns') or get_col('timestamp') or row[0]
        ts = int(ts_raw)
        # If timestamp looks like milliseconds (13 digits), convert to ns
        if ts < 1e15:  # Less than year 2001 in ns
            ts = ts * 1_000_000  # Convert ms to ns
        elif ts > 1e21:  # Looks like picoseconds (22+ digits), convert to ns
            ts = ts // 1_000
        
        # Symbol
        sym = get_col('symbol', default_symbol) or default_symbol
        
        # Price and volume
        price = float(get_col('price') or row[min(2, len(row)-1)])
        volume = float(get_col('volume') or row[min(3, len(row)-1)] if len(row) > 3 else 0)
        
        # Optional fields
        side = get_col('side')
        trade_id_raw = get_col('trade_id')
        trade_id = int(trade_id_raw) if trade_id_raw else None
        
        return Tick(
            symbol=sym.upper(),
            timestamp_ns=ts,
            price=price,
            volume=volume,
            side=side,
            trade_id=trade_id
        )
    
    def get_gap_count(self, symbol: str) -> int:
        """Get number of detected gaps for a symbol (after iteration)."""
        return self._gap_counts.get(symbol, 0)
    
    def get_available_symbols(self) -> List[str]:
        """List symbols with available tick data."""
        symbols = set()
        
        if not self.data_dir.exists():
            return []
        
        # Check legacy files
        for f in self.data_dir.glob("*_ticks.csv"):
            sym = f.stem.replace("_ticks", "")
            symbols.add(sym.upper())
        
        # Check partitioned directories
        for d in self.data_dir.iterdir():
            if d.is_dir() and any(d.glob("*.csv")):
                symbols.add(d.name.upper())
        
        return sorted(symbols)
    
    def get_time_range(self, symbol: str) -> Optional[tuple[int, int]]:
        """
        Get the time range of available data for a symbol.
        
        Returns:
            Tuple of (start_ts_ns, end_ts_ns) or None if no data
        """
        layout = self.get_storage_layout(symbol)
        
        if layout == "none":
            return None
        
        if layout == "partitioned":
            dates = self.get_available_dates(symbol)
            if not dates:
                return None
            
            # Get first tick from first file
            first_file = self.get_partitioned_dir(symbol) / f"{dates[0].strftime('%Y-%m-%d')}.csv"
            last_file = self.get_partitioned_dir(symbol) / f"{dates[-1].strftime('%Y-%m-%d')}.csv"
            
            first_ts = self._get_first_timestamp(first_file)
            last_ts = self._get_last_timestamp(last_file)
            
            if first_ts and last_ts:
                return (first_ts, last_ts)
            return None
        
        # Legacy single file
        file_path = self.get_legacy_file_path(symbol)
        first_ts = self._get_first_timestamp(file_path)
        last_ts = self._get_last_timestamp(file_path)
        
        if first_ts and last_ts:
            return (first_ts, last_ts)
        return None
    
    def _get_first_timestamp(self, file_path: Path) -> Optional[int]:
        """Get first timestamp from a CSV file."""
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            first_line = f.readline().strip()
            has_header = 'timestamp' in first_line.lower()
            
            if has_header:
                second_line = f.readline().strip()
                if second_line:
                    try:
                        ts = int(second_line.split(',')[0])
                        if ts < 1e15:
                            ts = ts * 1_000_000
                        return ts
                    except (ValueError, IndexError):
                        pass
            else:
                try:
                    ts = int(first_line.split(',')[0])
                    if ts < 1e15:
                        ts = ts * 1_000_000
                    return ts
                except (ValueError, IndexError):
                    pass
        return None
    
    def _get_last_timestamp(self, file_path: Path) -> Optional[int]:
        """Get last timestamp from a CSV file (reads last lines)."""
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'rb') as f:
                # Seek to end and read backwards
                f.seek(0, 2)
                file_size = f.tell()
                
                # Read last 1KB
                chunk_size = min(1024, file_size)
                f.seek(-chunk_size, 2)
                chunk = f.read().decode('utf-8', errors='ignore')
                
                lines = chunk.strip().split('\n')
                for line in reversed(lines):
                    if line and not 'timestamp' in line.lower():
                        try:
                            ts = int(line.split(',')[0])
                            if ts < 1e15:
                                ts = ts * 1_000_000
                            return ts
                        except (ValueError, IndexError):
                            continue
        except Exception:
            pass
        
        return None


# Placeholder for future Parquet support
class ParquetTickStore(TickStore):
    """
    Parquet-based tick store (future implementation).
    
    Would provide:
    - Faster random access
    - Better compression
    - Column-based filtering
    """
    
    def __init__(self, config: TickStoreConfig):
        super().__init__(config)
        # Would use pyarrow or fastparquet here
        raise NotImplementedError(
            "ParquetTickStore not yet implemented. Use TickStore with CSV files."
        )
