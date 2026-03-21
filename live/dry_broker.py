"""
live/dry_broker.py
==================
Paper-trading broker that simulates all exchange operations locally.

Implements IBroker without hitting any real API. All orders are filled
instantly at current mark price. Useful for testing the full live pipeline
without risking real money.

Usage:
    broker = DryBroker(initial_balance=1000.0)
    engine = LiveEngine(cfg, broker, strategy_cls, global_risk)
"""
from __future__ import annotations

from typing import Dict, Optional
from utils.logger import setup_logger
from Interfaces.IBroker import IBroker

log = setup_logger("DryBroker")


class _DryClient:
    """Minimal IClient-like stub for DryBroker."""

    def __init__(self, broker: "DryBroker"):
        self._broker = broker

    @property
    def raw_client(self):
        return self

    async def futures_exchange_info(self):
        return self._broker._exchange_info

    async def futures_mark_price(self, symbol: str):
        return {"markPrice": str(self._broker._prices.get(symbol, 0))}

    async def futures_klines(self, symbol: str, interval: str, limit: int):
        return []

    async def futures_create_order(self, **kw):
        return {"orderId": self._broker._next_oid()}

    async def futures_cancel_order(self, symbol: str, orderId: int):
        pass

    async def futures_cancel_all_open_orders(self, symbol: str):
        pass

    async def futures_get_open_orders(self, symbol: str):
        return []

    async def futures_position_information(self, symbol: str):
        amt = self._broker._positions.get(symbol, 0.0)
        return [{"positionAmt": str(amt)}] if amt != 0 else []

    async def futures_account_balance(self):
        return [{"asset": "USDT", "balance": str(self._broker._balance)}]

    async def futures_change_margin_type(self, symbol: str, marginType: str):
        pass

    async def futures_change_leverage(self, symbol: str, leverage: int):
        pass

    async def close_connection(self):
        pass


class DryBroker(IBroker):
    """
    Paper-trading broker — no real API calls.

    Tracks positions and balance locally. Orders fill instantly at mark price.
    Pass market_client to enable real mark-price lookups for accurate sizing.
    """

    def __init__(self, initial_balance: float = 10_000.0, market_client=None):
        self._balance = initial_balance
        self._positions: Dict[str, float] = {}    # symbol → signed qty
        self._prices: Dict[str, float] = {}       # symbol → last known price
        self._market_client = market_client       # Real client for mark price
        self._client_stub = _DryClient(self)
        self._order_counter = 9000
        self._exchange_info: dict = {"symbols": []}

        log.info("DryBroker initialized (paper trading) — balance=%.2f USDT", initial_balance)

    def _next_oid(self) -> int:
        self._order_counter += 1
        return self._order_counter

    # ── IBroker interface ────────────────────────────────────────────

    @property
    def client(self):
        return self._client_stub

    async def get_mark_price(self, symbol: str) -> float:
        # Use cached price if available (set by engine via bar data)
        if symbol in self._prices:
            return self._prices[symbol]
        # Fall back to real API if market_client provided
        if self._market_client is not None:
            try:
                data = await self._market_client.raw_client.futures_mark_price(symbol=symbol)
                price = float(data["markPrice"])
                self._prices[symbol] = price
                return price
            except Exception as e:
                log.warning("DryBroker: mark price fetch failed for %s: %s", symbol, e)
        return 0.0

    def set_price(self, symbol: str, price: float):
        """Set simulated mark price (called by engine or test harness)."""
        self._prices[symbol] = price

    async def market_order(self, symbol: str, side: str, qty: float):
        price = self._prices.get(symbol, 0.0)
        signed = qty if side == "BUY" else -qty
        self._positions[symbol] = self._positions.get(symbol, 0.0) + signed

        # Simulate P&L on close (simple)
        notional = price * qty
        log.info(
            "[DRY] %s %s qty=%.6f @ %.8f (notional=$%.2f)",
            symbol, side, qty, price, notional,
        )
        return {"orderId": self._next_oid()}

    async def close_position(self, symbol: str):
        amt = self._positions.get(symbol, 0.0)
        if amt == 0:
            return
        side = "SELL" if amt > 0 else "BUY"
        await self.market_order(symbol, side, abs(amt))

    async def position_amt(self, symbol: str) -> float:
        return self._positions.get(symbol, 0.0)

    async def ensure_isolated_margin(self, symbol: str):
        pass

    async def set_leverage(self, symbol: str, leverage: int):
        pass

    async def place_stop_market(self, symbol: str, side: str, stop_price: float) -> int:
        log.info("[DRY] SL order: %s %s @ %.8f", symbol, side, stop_price)
        return self._next_oid()

    async def place_take_profit(self, symbol: str, side: str, stop_price: float) -> int:
        log.info("[DRY] TP order: %s %s @ %.8f", symbol, side, stop_price)
        return self._next_oid()

    async def cancel_order(self, symbol: str, order_id: int):
        pass

    async def get_open_orders(self, symbol: str) -> list:
        return []

    async def balance(self, asset: str = "USDT") -> float:
        return self._balance

    async def exchange_info(self) -> dict:
        return self._exchange_info

    def set_exchange_info(self, info: dict):
        """Inject exchange info for symbol filters."""
        self._exchange_info = info
