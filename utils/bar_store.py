from collections import defaultdict
from typing import Dict, List, Any

class BarStore:
    """
    Centralized OHLCV buffer for all symbol-timeframe combinations.
    ▸ add_bar(...)   : Adds a closed bar from Streamer
    ▸ get_ohlcv(...) : Strategies fetch data from here
    """

    def __init__(self, maxlen: int = 600):
        self._maxlen = maxlen
        # data[(symbol, timeframe)] = {"open": [...], "high": [...], ...}
        self._data: Dict[tuple[str, str], Dict[str, List[float]]] = defaultdict(
            lambda: {"open": [], "high": [], "low": [], "close": [], "volume": []}
        )

    # ---------------- Called by Streamer -----------------
    def add_bar(self, symbol: str, tf: str, k: Dict[str, Any]) -> None:
        """Add closed candle from Binance kline JSON."""
        # Ensure bar is closed (though streamer typically checks this too)
        # If 'x' key is missing, assume it's a closed bar provided by historical fetch
        if "x" in k and not k["x"]:   
            return
            
        buf = self._data[(symbol, tf)]
        buf["open"].append(float(k["o"]))
        buf["high"].append(float(k["h"]))
        buf["low"].append(float(k["l"]))
        buf["close"].append(float(k["c"]))
        buf["volume"].append(float(k["v"]))

        # maxlen protection
        for arr in buf.values():
            if len(arr) > self._maxlen:
                del arr[: len(arr) - self._maxlen]

    # ---------------- Called by Strategies --------------
    def get_ohlcv(self, symbol: str, tf: str) -> Dict[str, List[float]]:
        """Returns reference, not copy – strategy can use directly."""
        return self._data[(symbol, tf)]
