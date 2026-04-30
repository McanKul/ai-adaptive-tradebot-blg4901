"""
tests/test_risk_order_set.py
============================
Tests for ``BasicRiskManager.validate_order_set`` — atomic multi-leg
risk validation used by paired-arbitrage strategies.
"""
from __future__ import annotations
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Backtest.risk import BasicRiskManager, RiskLimits
from Interfaces.market_data import Bar
from Interfaces.orders import Order, OrderSide, OrderType


def _make_bar(close=100.0):
    return Bar(
        symbol="BTCUSDT", timeframe="1h",
        timestamp_ns=1_000_000_000,
        open=99.0, high=101.0, low=98.0, close=close, volume=1_000.0,
    )


def _make_order(symbol, side, qty=0.1):
    return Order(
        symbol=symbol, side=side, order_type=OrderType.MARKET,
        quantity=qty, timestamp_ns=1_000_000_000,
    )


def _portfolio_stub(equity=10_000.0):
    """Minimal portfolio mock for risk checks."""
    p = MagicMock()
    p.position_quantity.return_value = 0.0
    p.equity.return_value = equity
    p.total_exposure.return_value = 0.0
    p.symbol_exposure.return_value = 0.0
    p.is_margin_mode = False
    return p


class TestValidateOrderSet(unittest.TestCase):

    def test_empty_set_is_trivially_valid(self):
        rm = BasicRiskManager()
        self.assertTrue(rm.validate_order_set([], _portfolio_stub(), _make_bar()))

    def test_all_legs_pass_returns_true_and_commits(self):
        rm = BasicRiskManager(RiskLimits(max_position_size=10.0,
                                          max_position_notional=10_000.0))
        orders = [
            _make_order("BTCUSDT", OrderSide.BUY, qty=0.05),
            _make_order("ETHUSDT", OrderSide.SELL, qty=1.0),
        ]
        ok = rm.validate_order_set(orders, _portfolio_stub(), _make_bar())
        self.assertTrue(ok)
        # Counters reflect commit (2 orders approved)
        self.assertEqual(rm._approved_orders, 2)
        self.assertEqual(rm._orders_this_bar, 2)
        self.assertEqual(rm._rejected_orders, 0)

    def test_one_leg_fails_rolls_back_and_returns_false(self):
        # Notional cap forces the SECOND leg to fail
        rm = BasicRiskManager(RiskLimits(
            max_position_size=10.0,
            max_position_notional=20.0,   # so 0.05 BTC * 100 = 5 OK; 1.0 ETH * 100 = 100 fails
        ))
        good = _make_order("BTCUSDT", OrderSide.BUY, qty=0.05)
        bad = _make_order("ETHUSDT", OrderSide.BUY, qty=1.0)
        ok = rm.validate_order_set([good, bad], _portfolio_stub(), _make_bar())
        self.assertFalse(ok)
        # Approved counter NOT incremented (rollback)
        self.assertEqual(rm._approved_orders, 0)
        self.assertEqual(rm._orders_this_bar, 0)
        # Single rejection bump for the whole set (not per leg)
        self.assertEqual(rm._rejected_orders, 1)

    def test_kill_switch_rejects_set_immediately(self):
        rm = BasicRiskManager()
        rm._activate_kill_switch("test")
        orders = [_make_order("BTCUSDT", OrderSide.BUY, qty=0.01)]
        self.assertFalse(rm.validate_order_set(orders, _portfolio_stub(), _make_bar()))


if __name__ == "__main__":
    unittest.main()
