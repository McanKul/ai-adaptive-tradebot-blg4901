import asyncio
import os
from binance.client import AsyncClient
from live.live_engine import LiveEngine
from live.broker_binance import BinanceBroker
from live.binance_client import BinanceClient
from strategy.RSIThreshold import Strategy # Example strategy import

# Example Configuration
# TODO: for future - load from external file

CONFIG = {
    "coins": ["DOGEUSDT"],
    "timeframe": "1m",
    "name": "RSI_Threshold_Strategy",
    "params": { 
        "timeframe": "1m", # Pass to strategy init too if needed
    },
    "execution": {
        "leverage": 5,
        "sl_pct": 2.0,
        "tp_pct": 4.0,
        # "expire_sec": 3600 # Removed from logic
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
    
    engine = LiveEngine(CONFIG, broker, Strategy)
    
    try:
        await engine.run()
    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close_connection() # Calls wrapper close which calls raw close

if __name__ == "__main__":
    asyncio.run(main())
