"""
core/config_validator.py
========================
Validates live trading config files before use.

Two modes:
* ``validate(path)`` — basic structural checks, used by ``app.py validate``.
* ``validate(path, real_money=True)`` — adds real-money safety checks
  (leverage cap, liquidity gate, no FTM/MATIC outdated tickers,
  reversal off, daily-loss enforced).  These rules embody the same
  defaults as ``config/profiles/canary.yaml``; if you fork the canary
  profile this validator catches divergences before the bot opens
  its first real-money position.
"""
from __future__ import annotations

import logging
import os
from typing import List

from core.factories.strategy_factory import StrategyFactory

log = logging.getLogger(__name__)


# Tickers Binance has renamed/delisted on USDT-M perpetuals — leaving
# them in the YAML lights up symbol-not-found rejections at order time.
# Keep updated as the exchange announces rebrandings.
DEPRECATED_TICKERS = {
    "FTMUSDT": "renamed to S in 2024 — use SUSDT once liquidity stabilises",
    "MATICUSDT": "renamed to POL in 2024 — use POLUSDT",
}

# Real-money default thresholds.  These mirror the canary profile and
# the canary_promotion_checklist.md.  They are intentionally tight.
REAL_MONEY_DEFAULTS = {
    "max_leverage": 10,
    "min_24h_volume_usd": 50_000_000.0,
    "min_max_daily_loss_pct": 0.05,  # require at least one loss-pct or USD cap
}


class ConfigValidator:
    """Validate a YAML/JSON live config file and report errors."""

    def validate(self, config_path: str, real_money: bool = False) -> List[str]:
        """
        Validate the given config file.

        Args:
            config_path: Path to the YAML/JSON live config.
            real_money: When True, also apply the strict real-money
                checks (leverage cap, liquidity gate, no deprecated
                tickers, etc.).  ``app.py validate`` exposes this via
                ``--real-money`` for the canary promotion gate.

        Returns:
            List of error strings. Empty list means the config is valid.
        """
        errors: List[str] = []

        # File existence
        if not os.path.exists(config_path):
            return [f"Config file not found: {config_path}"]

        # Parse
        try:
            from live.live_config import LiveConfig
            if config_path.endswith(".json"):
                cfg = LiveConfig.from_json(config_path)
            else:
                cfg = LiveConfig.from_yaml(config_path)
        except Exception as e:
            return [f"Failed to parse config: {e}"]

        # Strategy
        if not cfg.strategy_class:
            errors.append("strategy.class is missing or empty")
        else:
            try:
                StrategyFactory.resolve_class(cfg.strategy_class)
            except (ValueError, ImportError, AttributeError) as e:
                errors.append(
                    f"Cannot resolve strategy '{cfg.strategy_class}': {e}. "
                    f"Available: {StrategyFactory.list_available()}"
                )

        # Symbols
        if not cfg.symbols:
            errors.append("symbols list is empty")

        # Timeframe
        if not cfg.timeframe:
            errors.append("timeframe is missing or empty")

        # Sizing
        sizing = cfg.sizing
        if sizing.leverage < 1:
            errors.append(f"sizing.leverage must be >= 1, got {sizing.leverage}")
        if sizing.margin_usd <= 0:
            errors.append(f"sizing.margin_usd must be > 0, got {sizing.margin_usd}")
        valid_modes = {"fixed_qty", "notional_usd", "margin_usd"}
        if sizing.mode not in valid_modes:
            errors.append(f"sizing.mode must be one of {valid_modes}, got '{sizing.mode}'")

        # Risk
        risk = cfg.risk
        if risk.max_concurrent_positions < 1:
            errors.append(f"risk.max_concurrent_positions must be >= 1, got {risk.max_concurrent_positions}")
        if risk.max_daily_loss <= 0:
            errors.append(f"risk.max_daily_loss must be > 0, got {risk.max_daily_loss}")

        # News (if enabled)
        if cfg.news.enabled:
            provider = cfg.news.sentiment_provider.lower()
            from core.factories.news_factory import NewsFactory
            if provider not in NewsFactory._providers:
                errors.append(
                    f"Unknown news sentiment provider: '{provider}'. "
                    f"Available: {list(NewsFactory._providers.keys())}"
                )

        # ── Real-money strict checks ───────────────────────────────────
        if real_money:
            errors.extend(self._real_money_checks(cfg))

        return errors

    @staticmethod
    def _real_money_checks(cfg) -> List[str]:
        """Apply the canary-profile contract.

        Each rule maps to a specific real-money failure mode; the
        message names which one so the user can fix it without
        guessing.
        """
        errors: List[str] = []

        # 1. Leverage cap (5× preferred for canary; 10× hard ceiling).
        if cfg.sizing.leverage > REAL_MONEY_DEFAULTS["max_leverage"]:
            errors.append(
                f"[real-money] leverage {cfg.sizing.leverage}x exceeds "
                f"hard cap {REAL_MONEY_DEFAULTS['max_leverage']}x — "
                f"first real-money runs should stay at 3-5x"
            )

        # 2. Liquidity gate must be active.
        min_vol = float(getattr(cfg, "min_24h_volume_usd", 0.0) or 0.0)
        if min_vol < REAL_MONEY_DEFAULTS["min_24h_volume_usd"]:
            errors.append(
                f"[real-money] min_24h_volume_usd is "
                f"${min_vol:,.0f} (< required "
                f"${REAL_MONEY_DEFAULTS['min_24h_volume_usd']:,.0f}); "
                f"micro-cap whitelist will admit thin pairs"
            )

        # 3. Deprecated tickers — Binance rebranded these and the old
        # symbols may not exist on the margin pair anymore.
        for sym in cfg.symbols or []:
            if sym in DEPRECATED_TICKERS:
                errors.append(
                    f"[real-money] symbol '{sym}' is deprecated: "
                    f"{DEPRECATED_TICKERS[sym]}"
                )

        # 4. Sembol sayısı + concurrent positions hesabı — bir senkron
        # dump'da kontrol edilebilir bir sınır.
        n_syms = len(cfg.symbols or [])
        if n_syms > 8:
            errors.append(
                f"[real-money] symbol count {n_syms} > 8 — "
                f"correlation risk is high during alt-season; trim to "
                f"a smaller liquid whitelist (BTC/ETH/SOL/AVAX/LINK is the "
                f"canary baseline)"
            )

        # 5. allow_reversal must be off until Phase B5 ships.
        params = (cfg.strategy_params or {})
        if params.get("allow_reversal", False):
            errors.append(
                "[real-money] strategy.params.allow_reversal=true — "
                "close+open is two non-atomic legs; keep this off until "
                "Phase B5 (limit_with_timeout + reversal-fill confirmation) "
                "lands"
            )

        # 6. Daily-loss circuit-breaker must be configured.
        gr = getattr(cfg, "global_risk", None)
        if gr is not None:
            usd_cap = float(getattr(gr, "max_daily_loss", 0.0) or 0.0)
            pct_cap = float(getattr(gr, "max_daily_loss_pct", 0.0) or 0.0)
            if usd_cap <= 0 and pct_cap <= 0:
                errors.append(
                    "[real-money] global_risk daily-loss circuit-breaker "
                    "disabled — set max_daily_loss (USD) or "
                    "max_daily_loss_pct"
                )

        # 7. Liquidation guard must be on.
        liq = getattr(cfg, "liquidation_guard", None)
        if liq is not None and not getattr(liq, "enabled", False):
            errors.append(
                "[real-money] liquidation_guard.enabled=false — pre-emptive "
                "close is the cheapest insurance you have; turn it on"
            )

        # 8. Reconciliation must be on.
        rec = getattr(cfg, "reconciliation", None)
        if rec is not None and not getattr(rec, "enabled", False):
            errors.append(
                "[real-money] reconciliation.enabled=false — drift "
                "detection catches manual / ADL fills the engine missed; "
                "turn it on"
            )

        # 9. risk.max_leverage must dominate every effective per-symbol
        # leverage.  symbol_routes can quietly bump leverage above the
        # account-level tavan; the only way to catch that today is to
        # walk every symbol explicitly.
        risk_max_lev = float(getattr(cfg.risk, "max_leverage", 0.0) or 0.0)
        if risk_max_lev > 0.0 and hasattr(cfg, "leverage_for"):
            for sym in cfg.symbols or []:
                try:
                    eff_lev = float(cfg.leverage_for(sym))
                except Exception:
                    continue
                if eff_lev > risk_max_lev:
                    errors.append(
                        f"[real-money] {sym}: effective leverage "
                        f"{eff_lev:g}x exceeds risk.max_leverage "
                        f"{risk_max_lev:g}x (check symbol_routes overrides)"
                    )

        # 10. Spread filter must be armed.  PositionManager treats
        # ``max_entry_spread_bps=0`` as "filter disabled", and when the
        # filter is disabled a book-streamer outage cannot default-deny
        # new entries.  Real money requires the filter on.
        exec_cfg = getattr(cfg, "execution", None)
        max_spread = float(getattr(exec_cfg, "max_entry_spread_bps", 0.0) or 0.0)
        if max_spread <= 0.0:
            errors.append(
                "[real-money] execution.max_entry_spread_bps must be > 0 "
                "— a disabled spread filter cannot default-deny entries "
                "when the bookTicker stream is down"
            )

        return errors
