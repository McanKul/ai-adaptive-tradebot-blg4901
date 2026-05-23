"""
tests/test_donchian_state_per_symbol.py
========================================
Locks the per-symbol state isolation contract for the Donchian breakout
strategy and verifies the signed-position exit-order semantics.

Multi-coin live/dry-run runs feed the same Strategy instance with bars
from different symbols.  If state is global (instance-level) a BTC bar
can clobber ETH's trailing-stop or entry-price tracking, which can mask
bugs in backtest until real money is on the line.
"""
import math
import os
import sys
import unittest
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from Strategy.DonchianATRVolTarget import Strategy as DonchianStrategy
from Interfaces.market_data import Bar
from Interfaces.orders import OrderSide

# Reuse the existing OHLCV builders and MockCtx from the EMA cross tests.
from test_emacross_strategy import make_bar, MockCtx  # noqa: E402


def _flat_ohlcv(n: int = 60, center: float = 100.0) -> Dict[str, List[float]]:
    """Tight flat market — no breakout, just keeps state machinery alive."""
    closes = [center for _ in range(n)]
    highs = [c + 0.05 for c in closes]
    lows = [c - 0.05 for c in closes]
    opens = closes[:]
    volumes = [1000.0] * n
    return {"open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes}


class TestDonchianStatePerSymbol(unittest.TestCase):
    """State must be a dict-of-dicts keyed by symbol."""

    def test_state_dict_starts_empty(self):
        strategy = DonchianStrategy(filter_type="none")
        self.assertEqual(strategy._state, {})

    def test_two_symbols_create_two_state_dicts(self):
        strategy = DonchianStrategy(filter_type="none", dc_period=5,
                                    atr_period=5)
        ohlcv = _flat_ohlcv(n=30)

        bar_btc = make_bar("BTCUSDT", ohlcv)
        strategy.on_bar(bar_btc, MockCtx("BTCUSDT", ohlcv, position=0.0))

        bar_eth = make_bar("ETHUSDT", ohlcv)
        strategy.on_bar(bar_eth, MockCtx("ETHUSDT", ohlcv, position=0.0))

        self.assertIn("BTCUSDT", strategy._state)
        self.assertIn("ETHUSDT", strategy._state)
        # State dicts must be distinct objects, not shared references.
        self.assertIsNot(strategy._state["BTCUSDT"], strategy._state["ETHUSDT"])

    def test_btc_bar_does_not_pollute_eth_state(self):
        strategy = DonchianStrategy(filter_type="none", dc_period=5,
                                    atr_period=5)

        # Drive BTC through enough bars to bump bar_count and possibly
        # arm trailing-stop tracking via a long position.
        ohlcv = _flat_ohlcv(n=30)
        for i in range(10, 30):
            partial = {k: v[: i + 1] for k, v in ohlcv.items()}
            bar = make_bar("BTCUSDT", partial)
            strategy.on_bar(bar, MockCtx("BTCUSDT", partial, position=1.0))

        btc_state = strategy._state["BTCUSDT"]
        self.assertGreater(btc_state["bar_count"], 0)

        # Now feed a single ETH bar — its state must be pristine.
        bar_eth = make_bar("ETHUSDT", ohlcv)
        strategy.on_bar(bar_eth, MockCtx("ETHUSDT", ohlcv, position=0.0))

        eth_state = strategy._state["ETHUSDT"]
        self.assertEqual(eth_state["entry_bar_index"], 0)
        self.assertEqual(eth_state["entry_price"], 0.0)
        self.assertEqual(eth_state["highest_since_entry"], -math.inf)
        self.assertEqual(eth_state["lowest_since_entry"], math.inf)
        # ETH only saw 1 bar; BTC saw 20.
        self.assertEqual(eth_state["bar_count"], 1)
        self.assertGreaterEqual(btc_state["bar_count"], 20)

    def test_reset_clears_all_symbols(self):
        strategy = DonchianStrategy(filter_type="none", dc_period=5,
                                    atr_period=5)
        ohlcv = _flat_ohlcv(n=20)

        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            bar = make_bar(sym, ohlcv)
            strategy.on_bar(bar, MockCtx(sym, ohlcv, position=0.0))

        self.assertEqual(set(strategy._state.keys()),
                         {"BTCUSDT", "ETHUSDT", "SOLUSDT"})

        strategy.reset()
        self.assertEqual(strategy._state, {})

    def test_register_entry_writes_into_per_symbol_state(self):
        """Entry helper must update only the supplied symbol's dict."""
        strategy = DonchianStrategy(filter_type="none")

        st_btc = strategy._get_state("BTCUSDT")
        st_eth = strategy._get_state("ETHUSDT")

        st_btc["bar_count"] = 42
        strategy._register_entry(st_btc, price=12345.0)

        self.assertEqual(st_btc["entry_bar_index"], 42)
        self.assertEqual(st_btc["entry_price"], 12345.0)
        self.assertEqual(st_btc["highest_since_entry"], 12345.0)
        self.assertEqual(st_btc["lowest_since_entry"], 12345.0)

        # ETH untouched.
        self.assertEqual(st_eth["entry_bar_index"], 0)
        self.assertEqual(st_eth["entry_price"], 0.0)


class TestDonchianExitOrderDirection(unittest.TestCase):
    """`_exit_order` must take SIGNED position and pick the correct side."""

    def _bar(self) -> Bar:
        return Bar(symbol="BTCUSDT", timeframe="5m", timestamp_ns=0,
                   open=100, high=101, low=99, close=100, volume=1.0)

    def test_long_exit_returns_sell(self):
        order = DonchianStrategy._exit_order(self._bar(), 5.0)
        self.assertEqual(order.side, OrderSide.SELL)
        self.assertEqual(order.quantity, 5.0)
        self.assertTrue(order.reduce_only)

    def test_short_exit_returns_buy(self):
        order = DonchianStrategy._exit_order(self._bar(), -3.0)
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.quantity, 3.0)
        self.assertTrue(order.reduce_only)


if __name__ == "__main__":
    unittest.main()
