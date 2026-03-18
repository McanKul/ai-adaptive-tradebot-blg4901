import sys, os, asyncio, types
from unittest.mock import MagicMock

sys.modules.setdefault("binance", types.SimpleNamespace())
sys.modules.setdefault("binance.enums", types.SimpleNamespace(
    SIDE_BUY="BUY", SIDE_SELL="SELL", FUTURE_ORDER_TYPE_MARKET="MARKET",
))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from live.dry_broker import DryBroker


def test_market_order_updates_position():
    broker = DryBroker(initial_balance=1000.0)
    broker.set_price("BTCUSDT", 50000.0)

    asyncio.run(broker.market_order("BTCUSDT", "BUY", 0.01))
    amt = asyncio.run(broker.position_amt("BTCUSDT"))
    assert amt == 0.01

    # Close
    asyncio.run(broker.close_position("BTCUSDT"))
    amt = asyncio.run(broker.position_amt("BTCUSDT"))
    assert amt == 0.0


def test_balance_returns_initial():
    broker = DryBroker(initial_balance=5000.0)
    bal = asyncio.run(broker.balance("USDT"))
    assert bal == 5000.0


def test_short_position():
    broker = DryBroker()
    broker.set_price("ETHUSDT", 3000.0)

    asyncio.run(broker.market_order("ETHUSDT", "SELL", 1.0))
    amt = asyncio.run(broker.position_amt("ETHUSDT"))
    assert amt == -1.0


def test_mark_price_returns_set_price():
    broker = DryBroker()
    broker.set_price("DOGEUSDT", 0.08)
    price = asyncio.run(broker.get_mark_price("DOGEUSDT"))
    assert price == 0.08


def test_sl_tp_return_order_ids():
    broker = DryBroker()
    oid1 = asyncio.run(broker.place_stop_market("BTC", "SELL", 49000))
    oid2 = asyncio.run(broker.place_take_profit("BTC", "SELL", 52000))
    assert oid1 != oid2
    assert oid1 > 0
