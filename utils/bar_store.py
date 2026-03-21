from collections import defaultdict
from typing import Dict, List, Any

class BarStore:
    """
    Centralized OHLCV buffer for all symbol-timeframe combinations.
    ▸ add_bar(...)   : Adds a closed bar from Streamer
    ▸ get_ohlcv(...) : Strategies fetch data from here

    Timestamp-based dedup: if a bar with the same open timestamp arrives
    again (e.g. preload vs WebSocket overlap), the last values are replaced
    instead of appending a duplicate.
    """

    def __init__(self, maxlen: int = 600):
        self._maxlen = maxlen
        # data[(symbol, timeframe)] = {"open": [...], "high": [...], ...}
        self._data: Dict[tuple[str, str], Dict[str, List[float]]] = defaultdict(
            lambda: {"open": [], "high": [], "low": [], "close": [], "volume": []}
        )
        # last open-timestamp per (symbol, tf) for dedup
        self._last_ts: Dict[tuple[str, str], int] = {}

    # ---------------- Called by Streamer -----------------
    def add_bar(self, symbol: str, tf: str, k: Dict[str, Any]) -> None:
        """Add closed candle from Binance kline JSON."""
        # Ensure bar is closed
        if "x" in k and not k["x"]:
            return

        # Open timestamp: REST uses "t", old streamer used "start"
        ts = k.get("t") or k.get("start")
        if ts is not None:
            ts = int(ts)

        key = (symbol, tf)
        buf = self._data[key]

        # Dedup: same open timestamp as last bar → replace instead of append
        if ts is not None and self._last_ts.get(key) == ts and len(buf["open"]) > 0:
            buf["open"][-1] = float(k["o"])
            buf["high"][-1] = float(k["h"])
            buf["low"][-1] = float(k["l"])
            buf["close"][-1] = float(k["c"])
            buf["volume"][-1] = float(k["v"])
            return

        buf["open"].append(float(k["o"]))
        buf["high"].append(float(k["h"]))
        buf["low"].append(float(k["l"]))
        buf["close"].append(float(k["c"]))
        buf["volume"].append(float(k["v"]))

        if ts is not None:
            self._last_ts[key] = ts

        # maxlen protection
        for arr in buf.values():
            if len(arr) > self._maxlen:
                del arr[: len(arr) - self._maxlen]

    # ---------------- Called by Strategies --------------
    def get_ohlcv(self, symbol: str, tf: str) -> Dict[str, List[float]]:
        """Returns reference, not copy – strategy can use directly."""
        return self._data[(symbol, tf)]

    def get_recent(self, symbol: str, tf: str, limit: int) -> List[Dict[str, float]]:
        """
        Returns the most recent `limit` bars as a list of dictionaries.
        Each dictionary contains keys: 'open', 'high', 'low', 'close', 'volume'.
        """
        buf = self._data[(symbol, tf)]
        return [
            {
                "open": buf["open"][i],
                "high": buf["high"][i],
                "low": buf["low"][i],
                "close": buf["close"][i],
                "volume": buf["volume"][i],
            }
            for i in range(max(0, len(buf["open"]) - limit), len(buf["open"]))
        ]
