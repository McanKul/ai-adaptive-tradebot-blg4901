"""
live/position_store.py
======================
JSON-based persistence for open positions.

Saves open positions to disk so that a restart can recover:
- Entry price, qty, side, SL/TP prices
- Exchange order IDs (SL/TP)
- Bars held, strategy name, timeframe
- Trailing stop peak tracking

On startup, LiveSupervisor loads persisted positions and adopts them.
On every position open/close, the file is rewritten atomically.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from utils.logger import setup_logger

if TYPE_CHECKING:
    from live.position_manager import Position

log = setup_logger("PositionStore")

_DEFAULT_PATH = "logs/live_positions.json"


class PositionStore:
    """Persist open positions to a JSON file."""

    def __init__(self, path: str = _DEFAULT_PATH):
        self._path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # ── Save ─────────────────────────────────────────────────────────

    def save(self, positions: Dict[tuple, "Position"]) -> None:
        """Serialize all open positions to JSON."""
        records = []
        for (symbol, strategy), pos in positions.items():
            records.append({
                "symbol": pos.symbol,
                "side": pos.side,
                "qty": pos.qty,
                "entry_price": pos.entry,
                "sl_price": pos.sl,
                "tp_price": pos.tp,
                "open_ts": pos.open_ts,
                "tick": pos.tick,
                "strategy": pos.strategy,
                "timeframe": pos.timeframe,
                "bars_held": pos.bars_held,
                "sl_order_id": pos.sl_order_id,
                "tp_order_id": pos.tp_order_id,
                "peak_price": pos.peak_price,
                "peak_pnl": pos.peak_pnl,
            })

        data = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "count": len(records),
            "positions": records,
        }

        tmp = self._path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._path)
        except Exception as e:
            log.warning("Failed to save positions: %s", e)

    # ── Load ─────────────────────────────────────────────────────────

    def load(self) -> List[Dict[str, Any]]:
        """Load position records from JSON. Returns list of dicts."""
        if not os.path.exists(self._path):
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data.get("positions", [])
            log.info(
                "Loaded %d persisted positions (saved %s)",
                len(records), data.get("saved_at", "?"),
            )
            return records
        except Exception as e:
            log.warning("Failed to load positions: %s", e)
            return []

    # ── Clear ────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove the persistence file (e.g. after clean shutdown)."""
        try:
            if os.path.exists(self._path):
                os.remove(self._path)
        except Exception:
            pass
