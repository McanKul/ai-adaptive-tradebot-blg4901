"""
data_fetcher/binance_vision.py
==============================
Download real historical tick data from Binance Vision public data dumps.

Binance Vision provides free historical market data at:
https://data.binance.vision/

Supported data types:
- aggTrades: Aggregated trades (recommended for tick-level backtests)
- trades: Individual trades (larger files)

Data is organized by:
- spot/um/cm (spot, USD-M futures, COIN-M futures)
- daily/monthly
- Symbol
- Date

Example URL:
https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-15.zip

This module:
1. Downloads ZIP files from Binance Vision
2. Extracts and normalizes to our Tick schema
3. Saves partitioned by symbol/date in CSV format
4. Handles resume/checkpointing for large downloads
"""
from __future__ import annotations

import csv
import gzip
import io
import logging
import os
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
from urllib.parse import urljoin

import requests

log = logging.getLogger(__name__)

# Binance Vision base URLs
BINANCE_VISION_BASE = "https://data.binance.vision/"
SPOT_AGGTRADES_DAILY = "data/spot/daily/aggTrades/{symbol}/"
SPOT_AGGTRADES_MONTHLY = "data/spot/monthly/aggTrades/{symbol}/"
SPOT_TRADES_DAILY = "data/spot/daily/trades/{symbol}/"


@dataclass
class FetchConfig:
    """Configuration for data fetching."""
    output_dir: str = "data/ticks"
    data_type: str = "aggTrades"  # aggTrades or trades
    market_type: str = "spot"  # spot, um (USD-M futures), cm (COIN-M futures)
    use_monthly: bool = False  # Use monthly files instead of daily (for longer ranges)
    overwrite: bool = False  # Overwrite existing files
    max_retries: int = 3
    retry_delay: float = 1.0
    request_timeout: int = 60
    chunk_size: int = 8192
    verify_ssl: bool = True


class BinanceVisionFetcher:
    """
    Fetches historical tick data from Binance Vision public data dumps.
    
    Usage:
        fetcher = BinanceVisionFetcher(FetchConfig(output_dir="data/ticks"))
        fetcher.fetch_range("DOGEUSDT", start_date, end_date)
    """
    
    def __init__(self, config: Optional[FetchConfig] = None):
        self.config = config or FetchConfig()
        self.output_dir = Path(self.config.output_dir)
        self._session: Optional[requests.Session] = None
        
        # Statistics
        self.files_downloaded = 0
        self.files_skipped = 0
        self.total_ticks = 0
        self.errors: List[str] = []
    
    @property
    def session(self) -> requests.Session:
        """Lazy-initialize requests session."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": "BacktestTickFetcher/1.0"
            })
        return self._session
    
    def close(self):
        """Close the session."""
        if self._session:
            self._session.close()
            self._session = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def get_daily_url(self, symbol: str, date: datetime) -> str:
        """Get URL for daily aggTrades/trades file."""
        date_str = date.strftime("%Y-%m-%d")
        
        if self.config.data_type == "aggTrades":
            path = SPOT_AGGTRADES_DAILY.format(symbol=symbol)
            filename = f"{symbol}-aggTrades-{date_str}.zip"
        else:
            path = SPOT_TRADES_DAILY.format(symbol=symbol)
            filename = f"{symbol}-trades-{date_str}.zip"
        
        return urljoin(BINANCE_VISION_BASE, path + filename)
    
    def get_monthly_url(self, symbol: str, year: int, month: int) -> str:
        """Get URL for monthly aggTrades file."""
        month_str = f"{year}-{month:02d}"
        path = SPOT_AGGTRADES_MONTHLY.format(symbol=symbol)
        filename = f"{symbol}-aggTrades-{month_str}.zip"
        return urljoin(BINANCE_VISION_BASE, path + filename)
    
    def get_output_path(self, symbol: str, date: datetime) -> Path:
        """Get output path for a date's tick data."""
        symbol_dir = self.output_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        date_str = date.strftime("%Y-%m-%d")
        return symbol_dir / f"{date_str}.csv"
    
    def file_exists(self, symbol: str, date: datetime) -> bool:
        """Check if output file already exists."""
        return self.get_output_path(symbol, date).exists()
    
    def download_file(self, url: str, dest_path: Path) -> bool:
        """
        Download a file with retry logic.
        
        Returns:
            True if successful, False otherwise
        """
        for attempt in range(self.config.max_retries):
            try:
                log.debug(f"Downloading {url} (attempt {attempt + 1})")
                
                response = self.session.get(
                    url,
                    stream=True,
                    timeout=self.config.request_timeout,
                    verify=self.config.verify_ssl
                )
                
                if response.status_code == 404:
                    log.warning(f"File not found (404): {url}")
                    return False
                
                response.raise_for_status()
                
                # Write to temp file first
                temp_path = dest_path.with_suffix(".tmp")
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=self.config.chunk_size):
                        if chunk:
                            f.write(chunk)
                
                # Move to final path
                shutil.move(temp_path, dest_path)
                return True
                
            except requests.RequestException as e:
                log.warning(f"Download failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    self.errors.append(f"Failed to download {url}: {e}")
                    return False
            except Exception as e:
                log.error(f"Unexpected error downloading {url}: {e}")
                self.errors.append(f"Error downloading {url}: {e}")
                return False
        
        return False
    
    def extract_and_normalize(
        self,
        zip_path: Path,
        symbol: str,
        date: datetime
    ) -> int:
        """
        Extract ZIP file and normalize to our tick schema.
        
        Binance aggTrades CSV format:
            agg_trade_id, price, quantity, first_trade_id, last_trade_id, 
            timestamp_ms, is_buyer_maker, is_best_match
        
        Returns:
            Number of ticks processed
        """
        output_path = self.get_output_path(symbol, date)
        tick_count = 0
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Find the CSV file inside
                csv_files = [n for n in zf.namelist() if n.endswith('.csv')]
                if not csv_files:
                    log.error(f"No CSV file found in {zip_path}")
                    return 0
                
                csv_name = csv_files[0]
                
                with zf.open(csv_name) as csv_file:
                    # Read and normalize
                    reader = csv.reader(io.TextIOWrapper(csv_file, encoding='utf-8'))
                    
                    with open(output_path, 'w', newline='', encoding='utf-8') as out_f:
                        writer = csv.writer(out_f)
                        # Write header matching our Tick schema
                        writer.writerow([
                            'timestamp_ns', 'symbol', 'price', 'volume', 
                            'side', 'trade_id'
                        ])
                        
                        for row in reader:
                            if len(row) < 6:
                                continue
                            
                            try:
                                # Parse Binance format
                                agg_trade_id = int(row[0])
                                price = float(row[1])
                                quantity = float(row[2])
                                timestamp_ms = int(row[5])
                                is_buyer_maker = row[6].lower() == 'true' if len(row) > 6 else False
                                
                                # Convert to our schema
                                timestamp_ns = timestamp_ms * 1_000_000
                                side = 'sell' if is_buyer_maker else 'buy'
                                
                                writer.writerow([
                                    timestamp_ns,
                                    symbol,
                                    f"{price:.8f}",
                                    f"{quantity:.8f}",
                                    side,
                                    agg_trade_id
                                ])
                                tick_count += 1
                                
                            except (ValueError, IndexError) as e:
                                log.debug(f"Skipping malformed row: {row}")
                                continue
            
            log.info(f"Extracted {tick_count:,} ticks to {output_path}")
            return tick_count
            
        except zipfile.BadZipFile:
            log.error(f"Bad ZIP file: {zip_path}")
            self.errors.append(f"Bad ZIP file: {zip_path}")
            return 0
        except Exception as e:
            log.error(f"Error extracting {zip_path}: {e}")
            self.errors.append(f"Error extracting {zip_path}: {e}")
            return 0
    
    def fetch_date(self, symbol: str, date: datetime) -> int:
        """
        Fetch tick data for a single date.
        
        Returns:
            Number of ticks fetched, 0 if failed or skipped
        """
        symbol = symbol.upper()
        
        # Check if already exists
        if not self.config.overwrite and self.file_exists(symbol, date):
            log.debug(f"Skipping {symbol} {date.date()}: already exists")
            self.files_skipped += 1
            return 0
        
        # Create temp directory for downloads
        temp_dir = self.output_dir / ".temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Download ZIP
        url = self.get_daily_url(symbol, date)
        zip_path = temp_dir / f"{symbol}-{date.strftime('%Y-%m-%d')}.zip"
        
        if not self.download_file(url, zip_path):
            return 0
        
        # Extract and normalize
        tick_count = self.extract_and_normalize(zip_path, symbol, date)
        
        # Cleanup temp file
        try:
            zip_path.unlink()
        except:
            pass
        
        if tick_count > 0:
            self.files_downloaded += 1
            self.total_ticks += tick_count
        
        return tick_count
    
    def fetch_range(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        progress_callback=None
    ) -> Tuple[int, int]:
        """
        Fetch tick data for a date range.
        
        Args:
            symbol: Trading symbol (e.g., 'DOGEUSDT')
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            progress_callback: Optional callback(current_date, total_dates, ticks)
        
        Returns:
            Tuple of (files_downloaded, total_ticks)
        """
        symbol = symbol.upper()
        
        # Reset statistics
        self.files_downloaded = 0
        self.files_skipped = 0
        self.total_ticks = 0
        self.errors = []
        
        # Generate date list
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)
        
        log.info(f"Fetching {symbol} from {start_date.date()} to {end_date.date()} ({len(dates)} days)")
        
        for i, date in enumerate(dates):
            tick_count = self.fetch_date(symbol, date)
            
            if progress_callback:
                progress_callback(date, len(dates), tick_count)
            
            # Rate limiting - be nice to Binance servers
            if tick_count > 0:
                time.sleep(0.2)
        
        log.info(
            f"Completed: {self.files_downloaded} files downloaded, "
            f"{self.files_skipped} skipped, {self.total_ticks:,} total ticks"
        )
        
        if self.errors:
            log.warning(f"Encountered {len(self.errors)} errors")
        
        return self.files_downloaded, self.total_ticks
    
    def verify_data(self, symbol: str, start_date: datetime, end_date: datetime) -> dict:
        """
        Verify downloaded data completeness and integrity.
        
        Returns dict with:
            - dates_expected: Total dates in range
            - dates_present: Dates with data
            - dates_missing: Missing dates
            - total_ticks: Total tick count
        """
        symbol = symbol.upper()
        
        dates_expected = []
        current = start_date
        while current <= end_date:
            dates_expected.append(current)
            current += timedelta(days=1)
        
        dates_present = []
        dates_missing = []
        total_ticks = 0
        
        for date in dates_expected:
            path = self.get_output_path(symbol, date)
            if path.exists():
                dates_present.append(date)
                # Count lines (excluding header)
                with open(path, 'r') as f:
                    total_ticks += sum(1 for _ in f) - 1
            else:
                dates_missing.append(date)
        
        return {
            "symbol": symbol,
            "dates_expected": len(dates_expected),
            "dates_present": len(dates_present),
            "dates_missing": [d.strftime("%Y-%m-%d") for d in dates_missing],
            "total_ticks": total_ticks,
            "coverage_pct": len(dates_present) / len(dates_expected) * 100 if dates_expected else 0
        }


class SyntheticTickGenerator:
    """
    Generate synthetic tick data for testing/development ONLY.
    
    This should NOT be used for actual backtests - use real data!
    Clearly marked as synthetic to prevent accidental use.
    """
    
    @staticmethod
    def generate(
        symbol: str,
        output_path: Path,
        num_days: int = 7,
        seed: int = 42,
        initial_price: float = 0.08,
        volatility: float = 0.015
    ) -> int:
        """
        Generate synthetic tick data.
        
        Returns number of ticks generated.
        """
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        
        ns_per_second = 1_000_000_000
        ticks_per_day = 86400
        total_ticks = ticks_per_day * num_days
        
        # Start timestamp: 2025-01-01 00:00:00 UTC
        start_ts = 1735689600 * ns_per_second
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp_ns', 'symbol', 'price', 'volume', 'side', 'trade_id'])
            
            price = initial_price
            
            for i in range(total_ticks):
                jitter_ns = np.random.randint(-500_000_000, 500_000_000)
                ts = start_ts + i * ns_per_second + jitter_ns
                
                trend = 0.00001 * np.sin(2 * np.pi * i / (ticks_per_day * 3))
                hour_of_day = (i % ticks_per_day) // 3600
                vol_mult = 1.0 + 0.3 * np.sin(2 * np.pi * hour_of_day / 24)
                
                daily_vol = volatility / np.sqrt(ticks_per_day)
                change = np.random.normal(trend, daily_vol * vol_mult)
                price = price * (1 + change)
                price = max(0.05, min(0.15, price))
                
                base_volume = np.random.exponential(1000)
                volume = base_volume * (1 + abs(change) * 50)
                
                side = 'buy' if random.random() > 0.5 else 'sell'
                
                writer.writerow([
                    ts,
                    symbol,
                    f"{price:.8f}",
                    f"{volume:.2f}",
                    side,
                    i + 1  # trade_id
                ])
        
        return total_ticks
