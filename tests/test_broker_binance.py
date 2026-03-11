import sys, os, types
from unittest.mock import AsyncMock

# Mock binance modules
class _Ex(Exception):
    def __init__(self, code):
        super().__init__(f"code {code}")
        self.code = code

sys.modules.setdefault("binance", types.SimpleNamespace())
sys.modules["binance.exceptions"] = types.SimpleNamespace(BinanceAPIException=_Ex)
sys.modules["binance.enums"] = types.SimpleNamespace(
    SIDE_BUY="BUY", SIDE_SELL="SELL",
    FUTURE_ORDER_TYPE_MARKET="MARKET",
    FUTURE_ORDER_TYPE_STOP_MARKET="STOP_MARKET",
    FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET="TAKE_PROFIT_MARKET",
)

# Ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from live.broker_binance import BinanceBroker


def _client_with_exchange_info(tick_size="0.01"):
    c = AsyncMock()
    c.futures_exchange_info.return_value = {
        "symbols": [{
            "symbol": "BTCUSDT",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": tick_size}
            ]
        }]
    }
    return c


def test_market_order_calls_underlying():
    c = _client_with_exchange_info()
    broker = BinanceBroker(c)
    c.futures_create_order = AsyncMock()

    import asyncio
    asyncio.run(broker.market_order("BTCUSDT", "BUY", 0.001))
    c.futures_create_order.assert_awaited()


def test_place_stop_market_formats_price():
    c = _client_with_exchange_info("0.01")
    c.futures_create_order = AsyncMock(return_value={"orderId": 12345})
    broker = BinanceBroker(c)

    import asyncio
    oid = asyncio.run(broker.place_stop_market("BTCUSDT", "SELL", 123.4567))
    call = c.futures_create_order.await_args
    kwargs = call.kwargs
    # With tick 0.01 price should be rounded to 2 decimals
    assert kwargs["stopPrice"].count(".") == 1
    assert len(kwargs["stopPrice"].split(".")[-1]) == 2
    assert oid == 12345


def test_ensure_isolated_ignores_already_isolated():
    c = _client_with_exchange_info()
    c.futures_change_margin_type = AsyncMock(side_effect=sys.modules["binance.exceptions"].BinanceAPIException(-4046))
    broker = BinanceBroker(c)

    import asyncio
    # should not raise
    asyncio.run(broker.ensure_isolated_margin("BTCUSDT"))


def test_position_amt_and_balance():
    c = _client_with_exchange_info()
    c.futures_position_information = AsyncMock(return_value=[{"positionAmt": "0.5"}])
    c.futures_account_balance = AsyncMock(return_value=[{"asset": "USDT", "balance": "123.45"}])
    broker = BinanceBroker(c)

    import asyncio
    amt = asyncio.run(broker.position_amt("BTCUSDT"))
    bal = asyncio.run(broker.balance("USDT"))
    assert amt == 0.5
    assert bal == 123.45
