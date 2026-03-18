"""
live/notifier.py
================
Async Telegram notification sender for critical live trading events.

Sends alerts for:
- Position opened / closed (with P&L)
- Kill switch activated
- WebSocket disconnected / reconnected
- Daily summary

Setup:
    1. Create a bot via @BotFather → get BOT_TOKEN
    2. Send /start to your bot, then get your CHAT_ID via
       https://api.telegram.org/bot<TOKEN>/getUpdates
    3. Set env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

Usage:
    notifier = TelegramNotifier()          # reads from env
    await notifier.send("Hello world!")
    notifier.try_send("sync context msg")  # fire-and-forget from sync code
"""
from __future__ import annotations

import asyncio
import os
from typing import Optional
from urllib.request import urlopen, Request
from urllib.parse import quote

from utils.logger import setup_logger

log = setup_logger("Notifier")


class TelegramNotifier:
    """Lightweight Telegram bot notifier — no external deps (uses urllib)."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self._token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self._token and self._chat_id)

        if self._enabled:
            log.info("Telegram notifications enabled (chat_id=%s)", self._chat_id)
        else:
            log.info("Telegram notifications disabled (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)")

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def send(self, text: str) -> bool:
        """Send a message asynchronously. Returns True on success."""
        if not self._enabled:
            return False
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._send_sync, text)
        except Exception as e:
            log.debug("Telegram send failed: %s", e)
            return False

    def _send_sync(self, text: str) -> bool:
        """Blocking send via urllib (no aiohttp dependency)."""
        try:
            encoded = quote(text)
            url = (
                f"https://api.telegram.org/bot{self._token}"
                f"/sendMessage?chat_id={self._chat_id}"
                f"&text={encoded}&parse_mode=HTML"
            )
            req = Request(url, method="GET")
            with urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            log.debug("Telegram HTTP error: %s", e)
            return False

    def try_send(self, text: str) -> None:
        """Fire-and-forget send (safe from sync context)."""
        if not self._enabled:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.send(text))
        except RuntimeError:
            # No running loop — skip
            pass

    # ── Convenience methods for common events ────────────────────────

    async def position_opened(
        self, symbol: str, side: str, qty: float, price: float, leverage: int
    ):
        await self.send(
            f"📈 <b>OPEN</b> {symbol}\n"
            f"{side} qty={qty:.6f} @ {price:.4f}\n"
            f"Leverage: {leverage}x"
        )

    async def position_closed(
        self, symbol: str, side: str, pnl: float, pnl_pct: float, exit_type: str
    ):
        emoji = "✅" if pnl >= 0 else "🔴"
        await self.send(
            f"{emoji} <b>CLOSE</b> {symbol} ({exit_type})\n"
            f"{side} P&L: ${pnl:.4f} ({pnl_pct:+.2f}%)"
        )

    async def engine_started(self):
        await self.send("🚀 <b>Trading engine started</b>")
        
    async def kill_switch(self, reason: str):
        await self.send(f"🚨 <b>KILL SWITCH ACTIVATED</b>\n{reason}")

    async def ws_disconnected(self, attempt: int, error: str):
        await self.send(
            f"⚠️ <b>WS Disconnected</b> (attempt #{attempt})\n{error}"
        )

    async def daily_summary(self, stats: dict):
        await self.send(
            f"📊 <b>Daily Summary</b>\n"
            f"Trades: {stats.get('total_trades', 0)}\n"
            f"Win rate: {stats.get('win_rate_pct', 0):.1f}%\n"
            f"P&L: ${stats.get('total_pnl_usd', 0):.2f}\n"
            f"Max DD: {stats.get('max_drawdown_pct', 0):.2f}%"
        )

if __name__ == "__main__":
    # Quick test
    notifier = TelegramNotifier()
    asyncio.run(notifier.send("Hello from TelegramNotifier!"))