import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock
import sys
import os

# Mock binance before importing modules that use it
sys.modules["binance"] = MagicMock()
sys.modules["binance.client"] = MagicMock()
sys.modules["binance.exceptions"] = MagicMock()
sys.modules["binance.enums"] = MagicMock()

# Define enums needed
sys.modules["binance.enums"].SIDE_BUY = "BUY"
sys.modules["binance.enums"].SIDE_SELL = "SELL"
sys.modules["binance.enums"].FUTURE_ORDER_TYPE_MARKET = "MARKET"
sys.modules["binance.enums"].FUTURE_ORDER_TYPE_STOP_MARKET = "STOP_MARKET"
sys.modules["binance.enums"].FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"

# Adjust path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from live.live_engine import LiveEngine
from live.live_config import LiveConfig, SizingConfig, ExitConfig, RiskConfig
from Interfaces.IBroker import IBroker
from Strategy.binary_base_strategy import BinaryBaseStrategy

# Mock Strategy
class MockStrategy(BinaryBaseStrategy):
    def _live_signal(self, o, h, l, c, v):
        # Always return BUY signal if close > 100
        if c[-1] > 100:
            return "+1"
        return None

class TestLiveEngine(unittest.IsolatedAsyncioTestCase):
    async def test_engine_flow(self):
        # 1. Mock Broker & Client
        mock_client = AsyncMock()
        mock_client.futures_exchange_info.return_value = {
            "symbols": [{"symbol": "BTCUSDT", "quoteAsset": "USDT", "status": "TRADING"}]
        }
        # Mock klines for preload
        klines_data = [
            [1600000000000, "100", "110", "90", "105", "1000", 1600000059999, "0", 0, "0", "0", "0"]
        ]
        # Ensure futures_klines is an AsyncMock and set return_value
        mock_client.futures_klines = AsyncMock(return_value=klines_data)

        
        # Mock Socket Manager
        mock_bsm = MagicMock()
        mock_client.futures_socket.return_value = AsyncMock() # socket context mgr

        broker = AsyncMock(spec=IBroker)
        broker.client = mock_client
        broker.balance = AsyncMock(return_value=1000.0)

        # 2. Config (using LiveConfig)
        cfg = LiveConfig(
            strategy_class="MockStrategy",
            strategy_params={"timeframe": "1m"},
            symbols=["BTCUSDT"],
            timeframe="1m",
            name="TestStrategy",
            sizing=SizingConfig(leverage=1, margin_usd=10.0),
            exit=ExitConfig(stop_loss_pct=0.01, take_profit_pct=0.01),
            risk=RiskConfig(max_concurrent_positions=1),
        )

        # 3. Create Engine
        engine = LiveEngine(cfg, broker, MockStrategy)
        
        # Mock Streamer.get() to return one item then raise Cancelled or Stop
        # We need to monkeypatch the streamer created inside engine, 
        # but engine creates it in run().
        # So we will rely on mocking the Streamer class or just let it run briefly.
        
        # Easier: Mock Streamer class used in live_engine
        # But we imported it inside live_engine. Let's just mock resolve_symbols first.
        
        # 4. Run loop briefly
        # We start the engine task, wait a bit, then cancel.
        task = asyncio.create_task(engine.run())
        
        await asyncio.sleep(1) # Let it initialize and preload
        
        # Now we want to simulate a live tick.
        # Since we can't easily inject into the real Streamer without more mocking,
        # we check if preload worked (BarStore should have data).
        
        data = engine.bar_store.get_ohlcv("BTCUSDT", "1m")
        self.assertTrue(len(data["close"]) > 0, "BarStore should have preloaded data")
        self.assertEqual(data["close"][-1], 105.0)

        # Clean up
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

if __name__ == '__main__':
    unittest.main()
