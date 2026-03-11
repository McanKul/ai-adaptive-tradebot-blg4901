import asyncio
import os
from binance.client import AsyncClient
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
        print("WARNING: BINANCE_API_KEY not found in env. Client might fail or be read-only.")

    raw_client = await AsyncClient.create(api_key, api_secret)
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
        print("Stopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close_connection()

if __name__ == "__main__":
    asyncio.run(main())
