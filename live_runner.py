"""
live_runner.py
==============
Entry-point for live trading.

Usage:
    python live_runner.py                              # uses example_live_config.yaml
    python live_runner.py --config my_config.yaml      # custom config
"""
import asyncio
import argparse
import os
import sys

from binance.client import AsyncClient

from live.live_config import LiveConfig
from live.live_engine import LiveEngine
from live.broker_binance import BinanceBroker
from live.binance_client import BinanceClient
from strategy.RSIThreshold import Strategy  # Example strategy import

# Optional: News sentiment components
from news.ddg_news_source import DDGNewsSource
from news.gemini_sentiment import GeminiSentimentAnalyzer
from news.signal_combiner import BinarySignalCombiner

# Example Configuration
# TODO: for future - load from external file

CONFIG = {
    "coins": ["DOGEUSDT"],
    "timeframe": "1m",
    "name": "RSI_Threshold_Strategy",
    "params": {
        "timeframe": "1m",
    },
    "execution": {
        "leverage": 5,
        "sl_pct": 2.0,
        "tp_pct": 4.0,
    }
}


async def main():
    # Load API keys from env or secure source
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    if not api_key:
        print("WARNING: BINANCE_API_KEY not set. Client may fail or be read-only.")

    testnet = cfg.testnet
    raw_client = await AsyncClient.create(api_key, api_secret, testnet=testnet)
    client = BinanceClient(raw_client)
    broker = BinanceBroker(client)

    # ── News Sentiment Setup (optional) ──────────────────────────────────
    news_source = None
    sentiment_analyzer = None
    signal_combiner = None

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        print("✔ GOOGLE_API_KEY found — enabling news sentiment integration")
        news_source = DDGNewsSource()
        sentiment_analyzer = GeminiSentimentAnalyzer(api_key=google_api_key)
        signal_combiner = BinarySignalCombiner(buy_threshold=0.6, sell_threshold=0.4)
    else:
        print("ℹ GOOGLE_API_KEY not set — running without news sentiment")

    # ── Create Engine ────────────────────────────────────────────────────
    engine = LiveEngine(
        CONFIG,
        broker,
        Strategy,
        news_source=news_source,
        sentiment_analyzer=sentiment_analyzer,
        signal_combiner=signal_combiner,
        news_refresh_interval=300,
    )

    try:
        await engine.run()
    except KeyboardInterrupt:
        print("\nStopping (KeyboardInterrupt)...")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.close_connection()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live trading runner")
    parser.add_argument(
        "--config",
        type=str,
        default="example_live_config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    asyncio.run(main(args.config))
