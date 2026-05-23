"""Smoke sanity for the live config YAMLs shipped at the repo root.

These tests do not exercise behaviour — they just guard against typos
and "wrong unit" mistakes that have bitten the project before (e.g. a
comment claiming ``96 bars x 5m = 8h`` next to a 15m timeframe).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from live.live_config import LiveConfig


REPO_ROOT = Path(__file__).resolve().parents[1]

# Binance USDT-M canonical timeframes.  Reject typos like ``15min`` or
# ``1H`` early — the engine would fail later, but we want a friendlier
# pre-flight signal.
VALID_TIMEFRAMES = {
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
}


@pytest.mark.parametrize(
    "config_name",
    ["live_config.yaml", "live_config_emacross.yaml", "example_live_config.yaml"],
)
def test_live_config_loads_with_canonical_timeframe(config_name: str) -> None:
    path = REPO_ROOT / config_name
    if not path.exists():
        pytest.skip(f"{config_name} not present in this checkout")
    cfg = LiveConfig.from_yaml(str(path))
    assert cfg.timeframe in VALID_TIMEFRAMES, (
        f"{config_name}: timeframe={cfg.timeframe!r} is not a canonical "
        f"Binance USDT-M interval — fix it or extend VALID_TIMEFRAMES."
    )
    assert cfg.symbols, f"{config_name}: symbols list is empty"
    assert cfg.strategy_class, f"{config_name}: strategy.class is missing"
