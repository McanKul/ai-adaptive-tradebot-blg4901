#!/usr/bin/env python3
"""
tools/fetch_ticks.py
====================
CLI tool to download real historical tick data from Binance Vision.

Usage:
    python tools/fetch_ticks.py --symbol DOGEUSDT --start 2024-01-01 --end 2024-01-07
    python tools/fetch_ticks.py --symbol BTCUSDT --start 2024-01-01 --end 2024-01-31 --output data/ticks
    python tools/fetch_ticks.py --symbol DOGEUSDT --verify --start 2024-01-01 --end 2024-01-07

Options:
    --symbol        Trading symbol (e.g., DOGEUSDT, BTCUSDT)
    --start         Start date (YYYY-MM-DD)
    --end           End date (YYYY-MM-DD)
    --output        Output directory (default: data/ticks)
    --overwrite     Overwrite existing files
    --verify        Verify existing data instead of downloading
    --verbose       Enable verbose logging
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_fetcher.binance_vision import BinanceVisionFetcher, FetchConfig


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def progress_bar(current: int, total: int, width: int = 40) -> str:
    """Create a simple progress bar string."""
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = current / total * 100
    return f"[{bar}] {pct:5.1f}%"


def main():
    parser = argparse.ArgumentParser(
        description="Download historical tick data from Binance Vision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Download DOGE tick data for a week:
    python tools/fetch_ticks.py --symbol DOGEUSDT --start 2024-01-01 --end 2024-01-07

  Download with custom output directory:
    python tools/fetch_ticks.py --symbol BTCUSDT --start 2024-01-01 --end 2024-01-31 --output ./my_data

  Verify existing data:
    python tools/fetch_ticks.py --symbol DOGEUSDT --start 2024-01-01 --end 2024-01-07 --verify

Note: Data is downloaded from https://data.binance.vision/ (free, no API key needed)
        """
    )
    
    parser.add_argument(
        "--symbol", "-s",
        required=True,
        help="Trading symbol (e.g., DOGEUSDT, BTCUSDT)"
    )
    parser.add_argument(
        "--start",
        required=True,
        type=parse_date,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        required=True,
        type=parse_date,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/ticks",
        help="Output directory (default: data/ticks)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing data instead of downloading"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Validate dates
    if args.end < args.start:
        print(f"Error: End date ({args.end.date()}) must be >= start date ({args.start.date()})")
        sys.exit(1)
    
    today = datetime.now()
    if args.end > today:
        print(f"Warning: End date is in the future. Using yesterday's date instead.")
        from datetime import timedelta
        args.end = today - timedelta(days=1)
    
    # Calculate number of days
    num_days = (args.end - args.start).days + 1
    
    print("=" * 60)
    print("BINANCE VISION TICK DATA FETCHER")
    print("=" * 60)
    print(f"Symbol:     {args.symbol.upper()}")
    print(f"Date Range: {args.start.date()} to {args.end.date()} ({num_days} days)")
    print(f"Output Dir: {args.output}")
    print("=" * 60)
    
    config = FetchConfig(
        output_dir=args.output,
        overwrite=args.overwrite
    )
    
    with BinanceVisionFetcher(config) as fetcher:
        if args.verify:
            # Verify mode
            print("\nVerifying existing data...")
            result = fetcher.verify_data(args.symbol, args.start, args.end)
            
            print(f"\nVerification Results:")
            print(f"  Dates Expected: {result['dates_expected']}")
            print(f"  Dates Present:  {result['dates_present']}")
            print(f"  Coverage:       {result['coverage_pct']:.1f}%")
            print(f"  Total Ticks:    {result['total_ticks']:,}")
            
            if result['dates_missing']:
                print(f"\nMissing Dates ({len(result['dates_missing'])}):")
                for d in result['dates_missing'][:10]:
                    print(f"    {d}")
                if len(result['dates_missing']) > 10:
                    print(f"    ... and {len(result['dates_missing']) - 10} more")
            
            sys.exit(0 if result['coverage_pct'] == 100 else 1)
        
        # Download mode
        print("\nDownloading tick data...")
        print()
        
        dates_processed = [0]
        
        def on_progress(date, total, tick_count):
            dates_processed[0] += 1
            bar = progress_bar(dates_processed[0], total)
            status = f"✓ {tick_count:,} ticks" if tick_count > 0 else "⊘ skipped/missing"
            print(f"\r{bar} {date.date()} {status}".ljust(80), end="", flush=True)
        
        try:
            files, ticks = fetcher.fetch_range(
                args.symbol,
                args.start,
                args.end,
                progress_callback=on_progress
            )
        except KeyboardInterrupt:
            print("\n\nDownload interrupted by user.")
            sys.exit(1)
        
        print("\n")
        print("=" * 60)
        print("DOWNLOAD COMPLETE")
        print("=" * 60)
        print(f"Files Downloaded: {files}")
        print(f"Files Skipped:    {fetcher.files_skipped}")
        print(f"Total Ticks:      {ticks:,}")
        
        if fetcher.errors:
            print(f"\nErrors ({len(fetcher.errors)}):")
            for err in fetcher.errors[:5]:
                print(f"  - {err}")
            if len(fetcher.errors) > 5:
                print(f"  ... and {len(fetcher.errors) - 5} more")
        
        # Show output location
        output_path = Path(args.output) / args.symbol.upper()
        print(f"\nData saved to: {output_path}/")
        
        # Verify what we got
        result = fetcher.verify_data(args.symbol, args.start, args.end)
        if result['dates_missing']:
            print(f"\n⚠ Warning: {len(result['dates_missing'])} dates missing (may be weekends or no data)")


if __name__ == "__main__":
    main()
