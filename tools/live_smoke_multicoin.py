"""
tools/live_smoke_multicoin.py
=============================
Smoke-test for multi-coin live engine with a mock broker.

Run:
    python -m tools.live_smoke_multicoin

Verifies:
- LiveConfig multi-coin parsing (symbol_routes, global_risk, rate_limit)
- LiveSupervisor creates per-symbol managers
- Startup reconciliation against mock exchange state
- Order ID tracking in Position
- Global risk persistence round-trip
- Rate limiter + exchange info cache
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import sys

# ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live.live_config import (
    LiveConfig, SizingConfig, ExitConfig, GlobalRiskConfig,
    RateLimitConfig, SymbolRoute,
)
from live.position_manager import PositionManager, LiveSupervisor, Position
from live.global_risk import LiveGlobalRisk
from live.rate_limiter import AsyncRateLimiter, ExchangeInfoCache


# ── Minimal mock broker ──────────────────────────────────────────────
class _MockClient:
    """Minimal mock IClient."""
    async def futures_exchange_info(self):
        return {"symbols": [
            {"symbol": "BTCUSDT", "filters": [
                {"filterType": "LOT_SIZE", "stepSize": "0.0001"},
                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
            ]},
            {"symbol": "ETHUSDT", "filters": [
                {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
            ]},
        ]}
    async def futures_mark_price(self, symbol):
        prices = {"BTCUSDT": "65000.00", "ETHUSDT": "3500.00"}
        return {"markPrice": prices.get(symbol, "100.0")}
    async def futures_create_order(self, **kw):
        return {"orderId": 12345}
    async def futures_cancel_all_open_orders(self, symbol): pass
    async def futures_cancel_order(self, symbol, orderId): pass
    async def futures_get_open_orders(self, symbol): return []
    async def futures_position_information(self, symbol):
        # simulate orphan BTC position
        if symbol == "BTCUSDT":
            return [{"positionAmt": "0.010", "symbol": "BTCUSDT"}]
        return [{"positionAmt": "0", "symbol": symbol}]
    async def futures_account_balance(self):
        return [{"asset": "USDT", "balance": "1000.0"}]
    async def futures_change_margin_type(self, symbol, marginType): pass
    async def futures_change_leverage(self, symbol, leverage): pass
    async def futures_klines(self, symbol, interval, limit): return []
    async def close_connection(self): pass


class _MockBroker:
    """Minimal mock IBroker wrapping _MockClient."""
    def __init__(self):
        self._client = _MockClient()
        self._orders_placed: list[dict] = []

    @property
    def client(self):
        return self._client

    async def get_mark_price(self, symbol):
        d = await self._client.futures_mark_price(symbol=symbol)
        return float(d["markPrice"])

    async def market_order(self, symbol, side, qty):
        self._orders_placed.append({"type": "MARKET", "symbol": symbol,
                                     "side": side, "qty": qty})
        return {"orderId": len(self._orders_placed)}

    async def close_position(self, symbol):
        self._orders_placed.append({"type": "CLOSE", "symbol": symbol})

    async def position_amt(self, symbol):
        info = await self._client.futures_position_information(symbol=symbol)
        p = next((x for x in info if float(x["positionAmt"]) != 0), None)
        return float(p["positionAmt"]) if p else 0.0

    async def ensure_isolated_margin(self, symbol): pass
    async def set_leverage(self, symbol, leverage): pass

    async def place_stop_market(self, symbol, side, stop_price):
        self._orders_placed.append({"type": "SL", "symbol": symbol,
                                     "stopPrice": stop_price})
        return 100 + len(self._orders_placed)

    async def place_take_profit(self, symbol, side, stop_price):
        self._orders_placed.append({"type": "TP", "symbol": symbol,
                                     "stopPrice": stop_price})
        return 200 + len(self._orders_placed)

    async def cancel_order(self, symbol, order_id): pass
    async def get_open_orders(self, symbol): return []
    async def balance(self, asset="USDT"): return 1000.0

    async def exchange_info(self):
        return await self._client.futures_exchange_info()


# ── Helpers ───────────────────────────────────────────────────────────
def _ok(label: str):
    print(f"  ✓ {label}")

def _fail(label: str, detail: str = ""):
    print(f"  ✗ {label} — {detail}")


# ── Tests ─────────────────────────────────────────────────────────────
def test_config_multicoin():
    print("\n[1] LiveConfig multi-coin parsing")
    d = {
        "strategy": {"class": "RSIThreshold", "params": {}},
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "timeframe": "1m",
        "sizing": {"mode": "margin_usd", "margin_usd": 10.0, "leverage": 5},
        "exit": {"stop_loss_pct": 0.01, "take_profit_pct": 0.02},
        "symbol_routes": [
            {"symbol": "BTCUSDT", "leverage": 3, "margin_usd": 20.0,
             "stop_loss_pct": 0.005},
            {"symbol": "ETHUSDT", "trailing_stop_pct": 0.01},
        ],
        "global_risk": {
            "max_account_drawdown_pct": 0.15,
            "max_total_exposure_usd": 30000,
            "persist_path": "logs/test_risk.json",
        },
        "rate_limit": {"requests_per_minute": 800},
    }
    cfg = LiveConfig.from_dict(d)

    assert cfg.symbols == ["BTCUSDT", "ETHUSDT"], "symbols"
    _ok("symbols parsed")

    # Per-symbol overrides
    btc_sz = cfg.sizing_for("BTCUSDT")
    assert btc_sz.leverage == 3, f"BTC leverage={btc_sz.leverage}"
    assert btc_sz.margin_usd == 20.0, f"BTC margin={btc_sz.margin_usd}"
    _ok("BTCUSDT sizing override (leverage=3, margin=20)")

    btc_ex = cfg.exit_for("BTCUSDT")
    assert btc_ex.stop_loss_pct == 0.005, f"BTC SL={btc_ex.stop_loss_pct}"
    assert btc_ex.take_profit_pct == 0.02  # falls back to global
    _ok("BTCUSDT exit override (SL=0.5%, TP fallback=2%)")

    eth_ex = cfg.exit_for("ETHUSDT")
    assert eth_ex.trailing_stop_pct == 0.01
    assert eth_ex.stop_loss_pct == 0.01  # global
    _ok("ETHUSDT exit override (trailing=1%, SL=global)")

    # Fallback for unregistered symbol
    doge_sz = cfg.sizing_for("DOGEUSDT")
    assert doge_sz.leverage == 5
    _ok("Unregistered symbol falls back to global")

    # Global risk
    assert cfg.global_risk.max_account_drawdown_pct == 0.15
    _ok("GlobalRiskConfig parsed")

    # Rate limit
    assert cfg.rate_limit.requests_per_minute == 800
    _ok("RateLimitConfig parsed")


async def test_supervisor():
    print("\n[2] LiveSupervisor per-symbol isolation")
    broker = _MockBroker()
    sv = LiveSupervisor(broker)

    sv.register_symbol("BTCUSDT",
                        SizingConfig(margin_usd=20, leverage=3),
                        ExitConfig(stop_loss_pct=0.005, take_profit_pct=0.015),
                        max_concurrent=1)
    sv.register_symbol("ETHUSDT",
                        SizingConfig(margin_usd=10, leverage=5),
                        ExitConfig(stop_loss_pct=0.01, take_profit_pct=0.02),
                        max_concurrent=1)
    _ok("Registered 2 symbols")

    # Open BTC
    ok = await sv.open_position("BTCUSDT", 1, "test", 3, "1m")
    assert ok, "BTC open failed"
    _ok(f"BTCUSDT opened (orders={len(broker._orders_placed)})")

    # Open ETH
    ok = await sv.open_position("ETHUSDT", -1, "test", 5, "1m")
    assert ok, "ETH open failed"
    _ok(f"ETHUSDT opened (orders={len(broker._orders_placed)})")

    # Check isolation
    btc_qty = sv.position_qty("BTCUSDT")
    eth_qty = sv.position_qty("ETHUSDT")
    assert btc_qty > 0, f"BTC qty={btc_qty}"
    assert eth_qty < 0, f"ETH qty={eth_qty}"
    _ok(f"Isolation OK: BTC={btc_qty:.4f}, ETH={eth_qty:.4f}")

    # Check order IDs tracked
    btc_pos = list(sv.get("BTCUSDT").open_positions.values())[0]
    assert btc_pos.sl_order_id is not None, "SL order ID missing"
    assert btc_pos.tp_order_id is not None, "TP order ID missing"
    _ok(f"Order IDs tracked: SL={btc_pos.sl_order_id}, TP={btc_pos.tp_order_id}")

    # Merged view
    assert len(sv.open_positions) == 2
    _ok("Merged open_positions = 2")


async def test_reconcile():
    print("\n[3] Startup reconciliation")
    broker = _MockBroker()
    sv = LiveSupervisor(broker)

    sv.register_symbol("BTCUSDT",
                        SizingConfig(margin_usd=20, leverage=3),
                        ExitConfig(),
                        max_concurrent=1)
    sv.register_symbol("ETHUSDT",
                        SizingConfig(margin_usd=10, leverage=5),
                        ExitConfig(),
                        max_concurrent=1)

    await sv.reconcile_all("reconciled")

    # BTC had orphan position_amt=0.010
    btc_positions = sv.get("BTCUSDT").open_positions
    assert len(btc_positions) == 1, f"BTC should have 1 reconciled pos, got {len(btc_positions)}"
    _ok("BTCUSDT orphan adopted")

    # ETH had position_amt=0
    eth_positions = sv.get("ETHUSDT").open_positions
    assert len(eth_positions) == 0, f"ETH should have 0 pos, got {len(eth_positions)}"
    _ok("ETHUSDT no orphan (correct)")


def test_global_risk_persistence():
    print("\n[4] Global risk persistence")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "risk_state.json")
        cfg = GlobalRiskConfig(persist_path=path, max_account_drawdown_pct=0.10)

        # First instance
        gr = LiveGlobalRisk(cfg)
        gr.set_start_equity(1000.0)
        gr.record_pnl(-25.0)
        _ok(f"Recorded PnL: daily_pnl={gr.daily_pnl}")

        # Verify file exists
        assert os.path.exists(path), "State file not created"
        with open(path) as f:
            state = json.load(f)
        assert state["daily_pnl"] == -25.0
        _ok("State persisted to JSON")

        # Second instance (simulates restart)
        gr2 = LiveGlobalRisk(cfg)
        assert gr2.daily_pnl == -25.0, f"Loaded pnl={gr2.daily_pnl}"
        _ok("State loaded on restart")

        # Drawdown check
        ok, reason = gr2.check_account_risk(
            current_equity=850.0,
            total_exposure_usd=1000.0,
            open_position_count=1,
        )
        assert not ok, "Should fail drawdown"
        assert "drawdown" in reason.lower()
        _ok(f"Drawdown kill switch: {reason}")


async def test_rate_limiter():
    print("\n[5] Rate limiter + exchange info cache")
    rl = AsyncRateLimiter(max_per_minute=100)
    await rl.acquire(weight=1)
    _ok("Rate limiter acquire works")

    cache = ExchangeInfoCache(ttl_sec=60)
    call_count = 0
    async def _fetch():
        nonlocal call_count
        call_count += 1
        return {"symbols": []}

    await cache.get(_fetch)
    await cache.get(_fetch)
    assert call_count == 1, f"Cache missed: {call_count} calls"
    _ok("Exchange info cached (1 fetch for 2 gets)")

    cache.invalidate()
    await cache.get(_fetch)
    assert call_count == 2
    _ok("Cache invalidate + re-fetch works")


# ── Main ──────────────────────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("  LIVE MULTI-COIN SMOKE TEST")
    print("=" * 60)

    test_config_multicoin()
    await test_supervisor()
    await test_reconcile()
    test_global_risk_persistence()
    await test_rate_limiter()

    print("\n" + "=" * 60)
    print("  ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
