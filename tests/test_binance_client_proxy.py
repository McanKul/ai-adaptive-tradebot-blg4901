"""
tests/test_binance_client_proxy.py
===================================
Pin the IClient → AsyncClient proxy surface for BinanceClient.

Background
----------
`BinanceBroker.get_24h_volume` calls `self._client.futures_ticker(...)`.
The wrapper `live.binance_client.BinanceClient` was previously missing
that method, so on the canary's first run the call raised
`AttributeError: 'BinanceClient' object has no attribute 'futures_ticker'`.
The Phase-D volume-gate test caught nothing because it pokes a raw
`MagicMock` (which auto-attributes everything) — these tests use the
*real* wrapper class against a stubbed AsyncClient so the missing-method
regression cannot creep back in.
"""
import os
import sys
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from live.binance_client import BinanceClient


class TestBinanceClientFuturesTickerProxy(unittest.IsolatedAsyncioTestCase):
    async def test_futures_ticker_delegates_to_raw_client(self):
        raw = MagicMock()
        raw.futures_ticker = AsyncMock(return_value={
            "symbol": "BTCUSDT",
            "quoteVolume": "999.0",
        })
        wrapper = BinanceClient(raw)
        result = await wrapper.futures_ticker(symbol="BTCUSDT")
        raw.futures_ticker.assert_awaited_once_with(symbol="BTCUSDT")
        self.assertEqual(result["quoteVolume"], "999.0")

    async def test_futures_ticker_method_exists_on_wrapper(self):
        # Direct attribute check — guards against someone deleting the
        # delegating method while leaving raw_client intact.
        self.assertTrue(hasattr(BinanceClient, "futures_ticker"))
        # And it should be async.
        import inspect
        self.assertTrue(
            inspect.iscoroutinefunction(BinanceClient.futures_ticker)
        )


class TestBinanceBrokerVolumeOverWrapper(unittest.IsolatedAsyncioTestCase):
    """End-to-end through the real wrapper, not a bare MagicMock."""

    async def test_get_24h_volume_through_real_wrapper(self):
        from live.broker_binance import BinanceBroker

        raw = MagicMock()
        raw.futures_ticker = AsyncMock(return_value={
            "symbol": "BTCUSDT",
            "quoteVolume": "12345678901.23",
        })
        wrapper = BinanceClient(raw)
        broker = BinanceBroker(wrapper)
        vol = await broker.get_24h_volume("BTCUSDT")
        self.assertAlmostEqual(vol, 12345678901.23, places=2)

    async def test_iclient_subclass_must_implement_futures_ticker(self):
        """Sanity: instantiating an IClient subclass that omits
        futures_ticker should fail at construction time so silent
        breakage cannot slip in via a copy-paste."""
        from Interfaces.IClient import IClient

        class IncompleteClient(IClient):
            async def futures_exchange_info(self): return {}
            async def futures_mark_price(self, symbol): return {}
            async def futures_create_order(self, **kw): return {}
            async def futures_cancel_all_open_orders(self, symbol): return None
            async def futures_cancel_order(self, symbol, orderId): return None
            async def futures_get_open_orders(self, symbol): return []
            async def futures_position_information(self, symbol): return []
            async def futures_account_balance(self): return []
            async def futures_klines(self, symbol, interval, limit): return []
            async def futures_change_margin_type(self, symbol, marginType): return None
            async def futures_change_leverage(self, symbol, leverage): return None
            # Intentionally missing: futures_ticker
            async def close_connection(self): return None

        with self.assertRaises(TypeError):
            IncompleteClient()


if __name__ == "__main__":
    unittest.main()
