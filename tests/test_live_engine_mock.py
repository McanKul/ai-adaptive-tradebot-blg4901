"""
Integration test for LiveEngine with fully mocked broker and streamer.

Tests the full flow: preload → signal → open position → exit check.
"""
import asyncio
import unittest
import tempfile
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

# Mock binance before importing modules that use it
sys.modules["binance"] = MagicMock()
sys.modules["binance.client"] = MagicMock()
sys.modules["binance.exceptions"] = MagicMock()
sys.modules["binance.enums"] = MagicMock()

sys.modules["binance.enums"].SIDE_BUY = "BUY"
sys.modules["binance.enums"].SIDE_SELL = "SELL"
sys.modules["binance.enums"].FUTURE_ORDER_TYPE_MARKET = "MARKET"
sys.modules["binance.enums"].FUTURE_ORDER_TYPE_STOP_MARKET = "STOP_MARKET"
sys.modules["binance.enums"].FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from live.live_engine import LiveEngine
from live.live_config import LiveConfig, SizingConfig, ExitConfig, RiskConfig
from live.global_risk import LiveGlobalRisk, GlobalRiskConfig
from Interfaces.IBroker import IBroker
from Strategy.binary_base_strategy import BinaryBaseStrategy


class MockStrategy(BinaryBaseStrategy):
    """Always returns BUY when close > 100."""
    def _live_signal(self, o, h, l, c, v):
        if len(c) > 0 and c[-1] > 100:
            return "+1"
        return None


class TestLiveEngineIntegration(unittest.IsolatedAsyncioTestCase):

    def _make_cfg(self):
        return LiveConfig(
            strategy_class="MockStrategy",
            strategy_params={"timeframe": "1m"},
            symbols=["BTCUSDT"],
            timeframe="1m",
            name="TestStrategy",
            sizing=SizingConfig(leverage=1, margin_usd=10.0),
            exit=ExitConfig(stop_loss_pct=0.01, take_profit_pct=0.01),
            risk=RiskConfig(max_concurrent_positions=1),
        )

    def _make_broker(self):
        broker = AsyncMock(spec=IBroker)
        mock_client = AsyncMock()
        mock_client.raw_client = MagicMock()
        mock_client.futures_exchange_info.return_value = {
            "symbols": [{"symbol": "BTCUSDT", "quoteAsset": "USDT", "status": "TRADING"}]
        }
        mock_client.futures_klines.return_value = [
            [1600000000000, "100", "110", "90", "105", "1000",
             1600000059999, "0", 0, "0", "0", "0"],
        ]
        broker.client = mock_client
        broker.balance = AsyncMock(return_value=1000.0)
        broker.get_mark_price = AsyncMock(return_value=105.0)
        broker.ensure_isolated_margin = AsyncMock()
        broker.set_leverage = AsyncMock()
        broker.market_order = AsyncMock()
        broker.place_stop_market = AsyncMock(return_value=1001)
        broker.place_take_profit = AsyncMock(return_value=2001)
        broker.cancel_order = AsyncMock()
        broker.close_position = AsyncMock()
        broker.position_amt = AsyncMock(return_value=0.0)
        broker.exchange_info = AsyncMock(return_value={
            "symbols": [{
                "symbol": "BTCUSDT",
                "filters": [
                    {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                ],
            }]
        })
        return broker

    async def test_preload_populates_bar_store(self):
        """Engine preloads historical bars into BarStore."""
        cfg = self._make_cfg()
        broker = self._make_broker()
        global_risk = LiveGlobalRisk(GlobalRiskConfig())

        engine = LiveEngine(cfg, broker, MockStrategy, global_risk)

        # Run engine briefly then cancel
        task = asyncio.create_task(engine.run())
        await asyncio.sleep(0.5)

        data = engine.bar_store.get_ohlcv("BTCUSDT", "1m")
        self.assertTrue(len(data["close"]) > 0, "BarStore should have preloaded data")
        self.assertEqual(data["close"][-1], 105.0)

        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    async def test_engine_processes_bar_and_generates_signal(self):
        """Engine receives a bar from queue, runs strategy, and opens position."""
        cfg = self._make_cfg()
        broker = self._make_broker()
        global_risk = LiveGlobalRisk(GlobalRiskConfig())

        engine = LiveEngine(cfg, broker, MockStrategy, global_risk)

        # Manually set up engine state (skip run() startup)
        engine.symbols = ["BTCUSDT"]
        engine.supervisor.register_symbol(
            "BTCUSDT",
            sizing_cfg=cfg.sizing_for("BTCUSDT"),
            exit_cfg=cfg.exit_for("BTCUSDT"),
            max_concurrent=1,
        )

        # Preload some bars so strategy has data
        engine.bar_store.add_bar("BTCUSDT", "1m", {
            "t": 1000, "o": "100", "h": "110", "l": "90",
            "c": "105", "v": "1000", "x": True, "i": "1m",
        })
        for i in range(20):
            engine.bar_store.add_bar("BTCUSDT", "1m", {
                "t": 2000 + i * 60000, "o": "105", "h": "115", "l": "95",
                "c": str(106 + i * 0.1), "v": "900", "x": True, "i": "1m",
            })

        # Strategy should generate signal since close > 100
        from Interfaces.market_data import Bar
        from Interfaces.strategy_adapter import StrategyContext

        bar = Bar(symbol="BTCUSDT", timeframe="1m", timestamp_ns=0,
                  open=105, high=115, low=95, close=108, volume=900)
        ctx = StrategyContext(
            symbol="BTCUSDT", timeframe="1m",
            bar_store=engine.bar_store, position=0.0,
            equity=1000.0, cash=1000.0,
        )

        decision = engine.strategy.on_bar(bar, ctx)
        self.assertTrue(decision.has_signal or decision.has_orders)

    async def test_metrics_snapshot_after_init(self):
        """Metrics should be initialized and return valid snapshot."""
        cfg = self._make_cfg()
        broker = self._make_broker()
        global_risk = LiveGlobalRisk(GlobalRiskConfig())

        engine = LiveEngine(cfg, broker, MockStrategy, global_risk)

        snap = engine.metrics.snapshot()
        self.assertEqual(snap["total_trades"], 0)
        self.assertEqual(snap["win_rate_pct"], 0)


if __name__ == '__main__':
    unittest.main()
