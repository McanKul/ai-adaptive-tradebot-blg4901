"""
live/global_risk.py
===================
Account-level risk manager with JSON state persistence.

Tracks daily P&L, drawdown, and total exposure across all symbols.
State is saved to disk so that restarts don't lose the daily loss counter.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from utils.logger import setup_logger
from live.live_config import GlobalRiskConfig

log = setup_logger("GlobalRisk")


class LiveGlobalRisk:
    """
    Persistent global risk manager.

    State file (JSON):
        {
            "day": "2025-01-15",
            "daily_pnl": -12.5,
            "start_equity": 1000.0,
            "peak_equity": 1050.0,
            "kill_switch": false,
            "kill_reason": ""
        }
    """

    def __init__(self, cfg: GlobalRiskConfig):
        self.cfg = cfg
        self._path = cfg.persist_path
        self._day: str = ""
        self._daily_pnl: float = 0.0
        self._start_equity: float = 0.0
        self._peak_equity: float = 0.0
        self._kill_switch: bool = False
        self._kill_reason: str = ""
        # Phase A3 — consecutive-loss cooldown state (persisted)
        self._consecutive_losses: int = 0
        self._cooldown_until_ts: float = 0.0
        self._load()

    # ---- persistence ----
    def _load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                d = json.load(f)
            today = time.strftime("%Y-%m-%d")
            if d.get("day") == today:
                self._day = today
                self._daily_pnl = d.get("daily_pnl", 0.0)
                self._start_equity = d.get("start_equity", 0.0)
                self._peak_equity = d.get("peak_equity", 0.0)
                self._kill_switch = d.get("kill_switch", False)
                self._kill_reason = d.get("kill_reason", "")
                self._consecutive_losses = int(d.get("consecutive_losses", 0))
                self._cooldown_until_ts = float(d.get("cooldown_until_ts", 0.0))
                log.info("Loaded risk state: day=%s pnl=%.2f cooldown_until=%s",
                         today, self._daily_pnl,
                         self._cooldown_until_ts if self._cooldown_until_ts else "-")
            else:
                log.info("Risk state is from %s, resetting for today.", d.get("day"))
        except Exception as e:
            log.warning("Could not load risk state: %s", e)

    def _save(self):
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        d = {
            "day": self._day,
            "daily_pnl": round(self._daily_pnl, 4),
            "start_equity": round(self._start_equity, 4),
            "peak_equity": round(self._peak_equity, 4),
            "kill_switch": self._kill_switch,
            "kill_reason": self._kill_reason,
            "consecutive_losses": self._consecutive_losses,
            "cooldown_until_ts": round(self._cooldown_until_ts, 3),
        }
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2)
        except Exception as e:
            log.warning("Could not save risk state: %s", e)

    # ---- daily reset ----
    def set_start_equity(self, equity: float):
        today = time.strftime("%Y-%m-%d")
        if today != self._day:
            self._day = today
            self._daily_pnl = 0.0
            self._start_equity = equity
            self._peak_equity = max(self._peak_equity, equity)
            self._kill_switch = False
            self._kill_reason = ""
            self._save()
        if equity > self._peak_equity:
            self._peak_equity = equity
            self._save()

    # ---- record ----
    def record_pnl(self, pnl: float):
        """Record a closed-trade P&L.

        Updates the daily counter and the consecutive-loss state used
        by the cooldown circuit-breaker (Phase A3):

        * Loss (``pnl < 0``) increments the streak counter.  When the
          streak hits ``cfg.cooldown_after_losses`` the timer is set
          to ``now + cfg.cooldown_seconds``.
        * Win (``pnl > 0``) resets the streak counter (a single
          winner ends the cooldown's accrual; the live timer keeps
          ticking until expiry).
        """
        self._daily_pnl += pnl
        if pnl < 0:
            self._consecutive_losses += 1
            if (self.cfg.cooldown_after_losses > 0
                    and self._consecutive_losses >= self.cfg.cooldown_after_losses):
                self._cooldown_until_ts = time.time() + float(self.cfg.cooldown_seconds)
                log.warning(
                    "Cooldown armed: %d consecutive losses → halt entries for %ds",
                    self._consecutive_losses, self.cfg.cooldown_seconds,
                )
        elif pnl > 0:
            if self._consecutive_losses:
                log.info("Loss streak reset (was %d)", self._consecutive_losses)
            self._consecutive_losses = 0
        self._save()

    # ---- checks ----
    @property
    def is_kill_switch_active(self) -> bool:
        return self._kill_switch

    def trip_kill_switch(self, reason: str) -> None:
        """Manually activate the kill switch from any external safety check
        (drift detection, anomaly monitor, etc.).  Idempotent: a second
        call with a different reason keeps the original reason intact —
        the *first* trip is the one we want recorded."""
        if self._kill_switch:
            return
        self._kill_switch = True
        self._kill_reason = reason
        self._save()
        log.warning("KILL SWITCH TRIPPED externally: %s", reason)

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    def check_account_risk(
        self,
        current_equity: float,
        total_exposure_usd: float,
        open_position_count: int,
    ) -> tuple[bool, str]:
        """Return (ok, reason). ok=True means trading is allowed."""
        if self._kill_switch:
            return False, f"kill switch: {self._kill_reason}"

        # ── Phase A1: daily-loss circuit breaker ──────────────────────
        # USD threshold trips the kill switch so the bot stays halted
        # for the rest of the UTC day (counter resets in set_start_equity
        # on the next day's first heartbeat).
        if self.cfg.max_daily_loss > 0 and -self._daily_pnl >= self.cfg.max_daily_loss:
            self._kill_switch = True
            self._kill_reason = (
                f"daily loss {-self._daily_pnl:.2f} USD >= "
                f"{self.cfg.max_daily_loss:.2f} USD limit"
            )
            self._save()
            return False, self._kill_reason
        if (self.cfg.max_daily_loss_pct > 0 and self._start_equity > 0
                and -self._daily_pnl >= self._start_equity * self.cfg.max_daily_loss_pct):
            pct = -self._daily_pnl / self._start_equity
            self._kill_switch = True
            self._kill_reason = (
                f"daily loss {pct:.2%} >= "
                f"{self.cfg.max_daily_loss_pct:.2%} of start equity"
            )
            self._save()
            return False, self._kill_reason

        # ── Phase A3: consecutive-loss cooldown ───────────────────────
        # Soft block — does NOT trip kill switch.  When the timer
        # expires this branch is silently skipped → automatic resume
        # without manual intervention (user requirement).
        if self._cooldown_until_ts and time.time() < self._cooldown_until_ts:
            remaining = int(self._cooldown_until_ts - time.time())
            return False, (
                f"cooldown active: {self._consecutive_losses} consecutive losses, "
                f"{remaining}s remaining"
            )

        # Drawdown from peak equity
        if self._peak_equity > 0:
            dd = (self._peak_equity - current_equity) / self._peak_equity
            if dd > self.cfg.max_account_drawdown_pct:
                self._kill_switch = True
                self._kill_reason = f"drawdown {dd:.2%} > {self.cfg.max_account_drawdown_pct:.2%}"
                self._save()
                return False, self._kill_reason

        # Total exposure
        if total_exposure_usd > self.cfg.max_total_exposure_usd:
            return False, (
                f"total exposure {total_exposure_usd:.2f} > "
                f"{self.cfg.max_total_exposure_usd:.2f}"
            )

        # Concurrent positions cap (legacy field name was max_correlated_*)
        if open_position_count >= self.cfg.max_concurrent_positions:
            return False, (
                f"concurrent positions {open_position_count} >= "
                f"{self.cfg.max_concurrent_positions}"
            )

        return True, ""
