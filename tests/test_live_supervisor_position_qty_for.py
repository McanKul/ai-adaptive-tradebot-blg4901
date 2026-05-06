"""
tests/test_live_supervisor_position_qty_for.py
================================================
Pin the per-(symbol, strategy_name) read-side of position state.

Composite strategies open multiple positions on the same symbol with
distinct ``strategy_name`` keys (one per slot.id).  ``position_qty_for``
is the slot-aware accessor that LiveEngine's order-level dispatch uses
to read a specific slot's qty without being confused by sibling slots
sharing the same symbol.
"""
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock binance + submodules — same pattern as the order-level dispatch
# tests; needed because position_manager imports binance.enums at
# module load time.
_binance_stub = types.SimpleNamespace(
    BinanceSocketManager=MagicMock(return_value=MagicMock()),
    AsyncClient=MagicMock(),
    Client=MagicMock(),
)
sys.modules["binance"] = _binance_stub
sys.modules["binance.exceptions"] = types.SimpleNamespace(
    BinanceAPIException=type("BinanceAPIException", (Exception,), {}),
)
sys.modules["binance.enums"] = types.SimpleNamespace(
    SIDE_BUY="BUY",
    SIDE_SELL="SELL",
    FUTURE_ORDER_TYPE_MARKET="MARKET",
    FUTURE_ORDER_TYPE_STOP_MARKET="STOP_MARKET",
    FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET="TAKE_PROFIT_MARKET",
)

from live.position_manager import LiveSupervisor, PositionManager, Position  # noqa: E402


def _bare_position_manager() -> PositionManager:
    """Construct a PositionManager without going through __init__'s
    factory plumbing (sizing/exit/broker setup).  We only need the
    open_positions dict for the read-side accessors under test."""
    with patch.object(PositionManager, "__init__", lambda self, *a, **kw: None):
        pm = PositionManager()
    pm.open_positions = {}
    pm.history = []
    return pm


def _bare_supervisor() -> LiveSupervisor:
    with patch.object(LiveSupervisor, "__init__", lambda self, *a, **kw: None):
        sup = LiveSupervisor()
    sup._managers = {}
    return sup


def _open_position(symbol: str, side: str, qty: float, entry: float = 50_000.0) -> Position:
    """Helper to build a Position whose `is_long` reflects the side."""
    return Position(
        symbol=symbol, side=side, qty=qty, entry_price=entry,
        strategy="testfixture",
    )


class TestPositionManagerPositionQtyFor(unittest.TestCase):
    def test_returns_zero_when_no_position_open(self):
        pm = _bare_position_manager()
        self.assertEqual(0.0, pm.position_qty_for("BTCUSDT", "slot_a"))

    def test_returns_signed_qty_for_long(self):
        pm = _bare_position_manager()
        pm.open_positions[("BTCUSDT", "slot_a")] = _open_position(
            "BTCUSDT", side="BUY", qty=2.5,
        )
        self.assertEqual(2.5, pm.position_qty_for("BTCUSDT", "slot_a"))

    def test_returns_negative_qty_for_short(self):
        pm = _bare_position_manager()
        pm.open_positions[("BTCUSDT", "slot_b")] = _open_position(
            "BTCUSDT", side="SELL", qty=1.5,
        )
        self.assertEqual(-1.5, pm.position_qty_for("BTCUSDT", "slot_b"))

    def test_does_not_leak_across_strategy_names(self):
        pm = _bare_position_manager()
        pm.open_positions[("BTCUSDT", "slot_a")] = _open_position(
            "BTCUSDT", side="BUY", qty=2.5,
        )
        # Sibling slot has nothing open → must read 0, not slot_a's qty.
        self.assertEqual(0.0, pm.position_qty_for("BTCUSDT", "slot_b"))

    def test_closed_position_treated_as_zero(self):
        pm = _bare_position_manager()
        pos = _open_position("BTCUSDT", side="BUY", qty=2.5)
        pos.closed = True
        pm.open_positions[("BTCUSDT", "slot_a")] = pos
        self.assertEqual(0.0, pm.position_qty_for("BTCUSDT", "slot_a"))

    def test_legacy_position_qty_still_returns_first_open(self):
        # Single-strategy callers keep working — verify the
        # backward-compat aggregate accessor still finds the open
        # position when one bucket exists.
        pm = _bare_position_manager()
        pm.open_positions[("BTCUSDT", "canary_v1")] = _open_position(
            "BTCUSDT", side="SELL", qty=1.0,
        )
        self.assertEqual(-1.0, pm.position_qty("BTCUSDT"))


class TestLiveSupervisorPositionQtyFor(unittest.TestCase):
    def test_returns_zero_when_symbol_has_no_manager(self):
        sup = _bare_supervisor()
        self.assertEqual(0.0, sup.position_qty_for("BTCUSDT", "slot_a"))

    def test_delegates_to_per_symbol_manager(self):
        sup = _bare_supervisor()
        pm = _bare_position_manager()
        pm.open_positions[("BTCUSDT", "slot_a")] = _open_position(
            "BTCUSDT", side="BUY", qty=3.0,
        )
        sup._managers["BTCUSDT"] = pm
        self.assertEqual(3.0, sup.position_qty_for("BTCUSDT", "slot_a"))

    def test_two_slots_same_symbol_isolated(self):
        sup = _bare_supervisor()
        pm = _bare_position_manager()
        pm.open_positions[("BTCUSDT", "slot_a")] = _open_position(
            "BTCUSDT", side="BUY", qty=2.0,
        )
        pm.open_positions[("BTCUSDT", "slot_b")] = _open_position(
            "BTCUSDT", side="SELL", qty=1.5,
        )
        sup._managers["BTCUSDT"] = pm
        self.assertEqual(2.0, sup.position_qty_for("BTCUSDT", "slot_a"))
        self.assertEqual(-1.5, sup.position_qty_for("BTCUSDT", "slot_b"))
        # Composite-unaware aggregate accessor returns the first match;
        # this test pins that we did NOT change its behaviour.
        first = sup.position_qty("BTCUSDT")
        self.assertIn(first, (2.0, -1.5))


if __name__ == "__main__":
    unittest.main()
