import sys, os
import asyncio
import types
from unittest.mock import AsyncMock, MagicMock

# Mock binance modules used by position_manager
sys.modules.setdefault("binance", types.SimpleNamespace())
sys.modules["binance.client"] = MagicMock()
sys.modules["binance.exceptions"] = MagicMock()
enums = types.SimpleNamespace(
    SIDE_BUY="BUY", SIDE_SELL="SELL",
    FUTURE_ORDER_TYPE_MARKET="MARKET",
    FUTURE_ORDER_TYPE_STOP_MARKET="STOP_MARKET",
    FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET="TAKE_PROFIT_MARKET",
)
sys.modules["binance.enums"] = enums

# Ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Interfaces.IBroker import IBroker
from live.position_manager import PositionManager


def make_filters(step_size: str = "0.001", tick_size: str = "0.01"):
    return {
        "symbols": [{
            "symbol": "BTCUSDT",
            "filters": [
                {"filterType": "LOT_SIZE", "stepSize": step_size},
                {"filterType": "PRICE_FILTER", "tickSize": tick_size},
            ],
        }]
    }


async def _open_pos(pm: PositionManager):
    broker = pm.broker
    broker.client.futures_mark_price.return_value = {"markPrice": "100.0"}
    broker.client.futures_exchange_info.return_value = make_filters()
    broker.ensure_isolated_margin = AsyncMock()
    broker.set_leverage = AsyncMock()
    broker.market_order = AsyncMock()
    broker.place_stop_market = AsyncMock()
    broker.place_take_profit = AsyncMock()

    ok = await pm.open_position(
        symbol="BTCUSDT", side=1, strategy_name="TestStrat",
        leverage=10, sl_pct=1.0, tp_pct=1.0, timeframes="1m"
    )
    return ok


def test_open_position_happy_path(event_loop=None):
    broker = AsyncMock(spec=IBroker)
    broker.client = AsyncMock()
    pm = PositionManager(broker, base_capital=10.0, max_concurrent=2)

    ok = asyncio.run(_open_pos(pm))
    assert ok is True
    # Should have one open position
    assert len(pm.open_positions) == 1


def test_update_all_triggers_close_on_tp():
    broker = AsyncMock(spec=IBroker)
    broker.client = AsyncMock()
    pm = PositionManager(broker, base_capital=10.0, max_concurrent=2)

    asyncio.run(_open_pos(pm))
    # Mark price above TP to trigger close
    broker.get_mark_price = AsyncMock(return_value=1e9)
    broker.close_position = AsyncMock()

    asyncio.run(pm.update_all())
    broker.close_position.assert_awaited()
    assert len(pm.open_positions) == 0


def test_force_close_all_clears_positions():
    broker = AsyncMock(spec=IBroker)
    broker.client = AsyncMock()
    pm = PositionManager(broker, base_capital=10.0, max_concurrent=2)

    asyncio.run(_open_pos(pm))
    assert len(pm.open_positions) == 1

    broker.close_position = AsyncMock()
    asyncio.run(pm.force_close_all())
    assert len(pm.open_positions) == 0
    assert len(pm.history) == 1
