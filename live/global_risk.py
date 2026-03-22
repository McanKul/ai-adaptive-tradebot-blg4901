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
                log.info("Loaded risk state: day=%s pnl=%.2f", today, self._daily_pnl)
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
        self._daily_pnl += pnl
        self._save()

    # ---- checks ----
    @property
    def is_kill_switch_active(self) -> bool:
        return self._kill_switch

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

        # Daily loss USD (checked by LiveRiskChecker too, but this is the persistent version)
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

        # Correlated positions
        if open_position_count >= self.cfg.max_correlated_positions:
            return False, (
                f"correlated positions {open_position_count} >= "
                f"{self.cfg.max_correlated_positions}"
            )

        return True, ""
