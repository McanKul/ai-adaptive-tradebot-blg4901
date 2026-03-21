import sys, os, tempfile, types
from unittest.mock import MagicMock

# Mock binance
sys.modules.setdefault("binance", types.SimpleNamespace())
sys.modules.setdefault("binance.client", MagicMock())
sys.modules.setdefault("binance.exceptions", MagicMock())
enums = types.SimpleNamespace(
    SIDE_BUY="BUY", SIDE_SELL="SELL", FUTURE_ORDER_TYPE_MARKET="MARKET",
    FUTURE_ORDER_TYPE_STOP_MARKET="STOP_MARKET",
    FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET="TAKE_PROFIT_MARKET",
)
sys.modules["binance.enums"] = enums

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from live.position_store import PositionStore
from live.position_manager import Position


def _make_pos(symbol="BTCUSDT", side="BUY", qty=0.01, entry=50000.0):
    pos = Position(symbol=symbol, side=side, qty=qty, entry_price=entry,
                   strategy="test_strat", timeframe="1m")
    pos.sl_order_id = 1001
    pos.tp_order_id = 2001
    pos.bars_held = 5
    pos.peak_price = 51000.0
    return pos


def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "positions.json")
        store = PositionStore(path=path)

        positions = {
            ("BTCUSDT", "test_strat"): _make_pos(),
            ("ETHUSDT", "test_strat"): _make_pos("ETHUSDT", "SELL", 1.0, 3000.0),
        }

        store.save(positions)
        records = store.load()

        assert len(records) == 2
        btc = next(r for r in records if r["symbol"] == "BTCUSDT")
        assert btc["side"] == "BUY"
        assert btc["qty"] == 0.01
        assert btc["entry_price"] == 50000.0
        assert btc["sl_order_id"] == 1001
        assert btc["tp_order_id"] == 2001
        assert btc["bars_held"] == 5
        assert btc["peak_price"] == 51000.0


def test_load_empty_when_no_file():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "nonexistent.json")
        store = PositionStore(path=path)
        assert store.load() == []


def test_clear_removes_file():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "positions.json")
        store = PositionStore(path=path)
        store.save({("BTC", "s"): _make_pos()})
        assert os.path.exists(path)
        store.clear()
        assert not os.path.exists(path)
