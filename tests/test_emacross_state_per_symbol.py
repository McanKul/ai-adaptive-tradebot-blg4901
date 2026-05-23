"""
tests/test_emacross_state_per_symbol.py
========================================
Regression-lock for the EMA cross / MACD trend strategy.

EMACrossMACDTrend already keeps per-symbol state in `self._state` and
`_exit_order` already accepts a SIGNED position; these tests pin the
contract so a future refactor cannot silently regress to instance-level
state or to passing `abs(position)` (which would force every short exit
to ship a SELL reduce_only — i.e. it would NOT close the short).
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from Strategy.EMACrossMACDTrend import Strategy as EMACrossStrategy
from Interfaces.market_data import Bar
from Interfaces.orders import OrderSide

from test_emacross_strategy import make_bar, MockCtx  # noqa: E402


def _flat_ohlcv(n: int = 30, center: float = 100.0):
    closes = [center for _ in range(n)]
    highs = [c + 0.05 for c in closes]
    lows = [c - 0.05 for c in closes]
    opens = closes[:]
    volumes = [1000.0] * n
    return {"open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes}


class TestEMACrossStatePerSymbol(unittest.TestCase):
    def test_state_dict_starts_empty(self):
        strategy = EMACrossStrategy()
        self.assertEqual(strategy._state, {})

    def test_two_symbols_create_distinct_state_dicts(self):
        strategy = EMACrossStrategy(volume_filter=False)
        ohlcv = _flat_ohlcv()

        for sym in ("BTCUSDT", "ETHUSDT"):
            bar = make_bar(sym, ohlcv)
            strategy.on_bar(bar, MockCtx(sym, ohlcv, position=0.0))

        self.assertIn("BTCUSDT", strategy._state)
        self.assertIn("ETHUSDT", strategy._state)
        self.assertIsNot(strategy._state["BTCUSDT"],
                         strategy._state["ETHUSDT"])

    def test_btc_bar_does_not_pollute_eth_state(self):
        strategy = EMACrossStrategy(volume_filter=False)
        ohlcv = _flat_ohlcv(n=30)

        # Drive BTC for many bars.
        for i in range(10, 30):
            partial = {k: v[: i + 1] for k, v in ohlcv.items()}
            bar = make_bar("BTCUSDT", partial)
            strategy.on_bar(bar, MockCtx("BTCUSDT", partial, position=0.0))

        btc_count = strategy._state["BTCUSDT"]["bar_count"]
        self.assertGreater(btc_count, 0)

        # One ETH bar.
        bar_eth = make_bar("ETHUSDT", ohlcv)
        strategy.on_bar(bar_eth, MockCtx("ETHUSDT", ohlcv, position=0.0))

        eth = strategy._state["ETHUSDT"]
        self.assertEqual(eth["bar_count"], 1)
        self.assertEqual(eth["entry_bar_index"], 0)
        self.assertEqual(eth["entry_price"], 0.0)
        self.assertEqual(eth["highest_since_entry"], -math.inf)
        self.assertEqual(eth["lowest_since_entry"], math.inf)


class TestEMACrossExitOrderDirection(unittest.TestCase):
    def _bar(self) -> Bar:
        return Bar(symbol="BTCUSDT", timeframe="5m", timestamp_ns=0,
                   open=100, high=101, low=99, close=100, volume=1.0)

    def test_long_exit_returns_sell(self):
        order = EMACrossStrategy._exit_order(self._bar(), 5.0)
        self.assertEqual(order.side, OrderSide.SELL)
        self.assertEqual(order.quantity, 5.0)
        self.assertTrue(order.reduce_only)
        self.assertEqual(order.strategy_id, "EMACROSS_EXIT")

    def test_short_exit_returns_buy(self):
        order = EMACrossStrategy._exit_order(self._bar(), -3.0)
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.quantity, 3.0)
        self.assertTrue(order.reduce_only)


if __name__ == "__main__":
    unittest.main()
