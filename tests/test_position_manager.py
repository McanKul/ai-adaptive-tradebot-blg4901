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
from live.live_config import SizingConfig, ExitConfig


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


def _make_pm(max_concurrent=2, leverage=10, margin_usd=10.0,
             sl_pct=0.01, tp_pct=0.01):
    broker = AsyncMock(spec=IBroker)
    broker.client = AsyncMock()
    sizing = SizingConfig(leverage=leverage, margin_usd=margin_usd)
    exit_cfg = ExitConfig(stop_loss_pct=sl_pct, take_profit_pct=tp_pct,
                          use_exchange_orders=True)
    pm = PositionManager(broker, sizing_cfg=sizing, exit_cfg=exit_cfg,
                         max_concurrent=max_concurrent)
    return pm


async def _open_pos(pm: PositionManager, leverage: int = 10):
    broker = pm.broker
    filters = make_filters()
    broker.client.futures_mark_price.return_value = {"markPrice": "100.0"}
    broker.client.futures_exchange_info.return_value = filters
    broker.exchange_info = AsyncMock(return_value=filters)
    broker.ensure_isolated_margin = AsyncMock()
    broker.set_leverage = AsyncMock()
    broker.market_order = AsyncMock()
    broker.place_stop_market = AsyncMock(return_value=1001)
    broker.place_take_profit = AsyncMock(return_value=2001)
    broker.cancel_order = AsyncMock()
    broker.get_open_orders = AsyncMock(return_value=[])

    ok = await pm.open_position(
        symbol="BTCUSDT", side=1, strategy_name="TestStrat",
        leverage=leverage, timeframe="1m"
    )
    return ok


def test_open_position_happy_path():
    pm = _make_pm()
    ok = asyncio.run(_open_pos(pm))
    assert ok is True
    assert len(pm.open_positions) == 1


def test_update_all_triggers_close_on_tp():
    pm = _make_pm()
    asyncio.run(_open_pos(pm))
    # Mark price above TP to trigger close
    pm.broker.get_mark_price = AsyncMock(return_value=1e9)
    pm.broker.position_amt = AsyncMock(return_value=10.0)  # still open on exchange
    pm.broker.close_position = AsyncMock()
    pm.broker.cancel_order = AsyncMock()
    pm.broker.client.futures_cancel_all_open_orders = AsyncMock()

    asyncio.run(pm.update_all())
    pm.broker.close_position.assert_awaited()
    assert len(pm.open_positions) == 0


def test_force_close_all_clears_positions():
    pm = _make_pm()
    asyncio.run(_open_pos(pm))
    assert len(pm.open_positions) == 1

    pm.broker.close_position = AsyncMock()
    pm.broker.cancel_order = AsyncMock()
    pm.broker.client.futures_cancel_all_open_orders = AsyncMock()
    asyncio.run(pm.force_close_all())
    assert len(pm.open_positions) == 0
    assert len(pm.history) == 1
