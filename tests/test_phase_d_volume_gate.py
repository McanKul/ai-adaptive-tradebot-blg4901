"""
tests/test_phase_d_volume_gate.py
=================================
Phase D — 24h volume liquidity gate at engine startup.

Covers:
* ``IBroker.get_24h_volume`` default returns ``+inf`` so existing
  brokers without the override admit every symbol.
* ``BinanceBroker.get_24h_volume`` reads ``quoteVolume`` from the
  exchange's ticker endpoint.
* ``LiveConfig.min_24h_volume_usd`` field is plumbed and defaults to
  the canary-friendly $100M.
"""
from __future__ import annotations
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from live.dry_broker import DryBroker
from live.live_config import LiveConfig


class TestIBrokerDefault(unittest.IsolatedAsyncioTestCase):

    async def test_dry_broker_admits_all(self):
        # DryBroker inherits the IBroker default of +inf
        broker = DryBroker()
        self.assertEqual(await broker.get_24h_volume("BTCUSDT"), float("inf"))


class TestBinanceBrokerVolume(unittest.IsolatedAsyncioTestCase):

    async def test_pulls_quote_volume(self):
        from live.broker_binance import BinanceBroker

        client = MagicMock()
        client.futures_ticker = AsyncMock(return_value={
            "symbol": "BTCUSDT",
            "quoteVolume": "12345678901.23",
        })
        broker = BinanceBroker(client)
        vol = await broker.get_24h_volume("BTCUSDT")
        self.assertAlmostEqual(vol, 12345678901.23, places=2)

    async def test_failure_returns_zero(self):
        from live.broker_binance import BinanceBroker

        client = MagicMock()
        client.futures_ticker = AsyncMock(side_effect=Exception("timeout"))
        broker = BinanceBroker(client)
        self.assertEqual(await broker.get_24h_volume("BTCUSDT"), 0.0)


class TestLiveConfigField(unittest.TestCase):

    def test_default_threshold_disabled(self):
        # Default 0 keeps the gate disabled to preserve backwards
        # compatibility with existing fixtures and dry-run smoke
        # tests.  Canary profile YAML re-enables it at 100M.
        cfg = LiveConfig()
        self.assertEqual(cfg.min_24h_volume_usd, 0.0)

    def test_yaml_override(self):
        cfg = LiveConfig.from_dict({"min_24h_volume_usd": 50_000_000})
        self.assertEqual(cfg.min_24h_volume_usd, 50_000_000.0)


if __name__ == "__main__":
    unittest.main()
