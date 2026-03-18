import sys, os

# Ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.bar_store import BarStore


def test_bar_store_add_and_trim():
    bs = BarStore(maxlen=3)
    sym, tf = "BTCUSDT", "1m"

    # Add three closed bars
    for o,h,l,c,v in [(100,110,90,105,1000), (105,115,95,110,900), (110,120,100,115,800)]:
        bs.add_bar(sym, tf, {"o":o, "h":h, "l":l, "c":c, "v":v, "x":True})

    data = bs.get_ohlcv(sym, tf)
    assert data["close"] == [105.0, 110.0, 115.0]

    # Non-closed bar should be ignored
    bs.add_bar(sym, tf, {"o":116, "h":121, "l":111, "c":117, "v":700, "x":False})
    assert bs.get_ohlcv(sym, tf)["close"] == [105.0, 110.0, 115.0]

    # Add another closed bar -> trims to maxlen=3
    bs.add_bar(sym, tf, {"o":116, "h":121, "l":111, "c":117, "v":700, "x":True})
    assert bs.get_ohlcv(sym, tf)["close"] == [110.0, 115.0, 117.0]


def test_bar_store_dedup_by_timestamp():
    """Same open timestamp should replace the last bar, not duplicate."""
    bs = BarStore(maxlen=10)
    sym, tf = "ETHUSDT", "1m"

    # Preload adds a bar with timestamp t=1000
    bs.add_bar(sym, tf, {"t": 1000, "o": 50, "h": 55, "l": 48, "c": 52, "v": 100, "x": True})
    assert len(bs.get_ohlcv(sym, tf)["close"]) == 1
    assert bs.get_ohlcv(sym, tf)["close"][-1] == 52.0

    # WebSocket closes the SAME bar (t=1000) with updated final values
    bs.add_bar(sym, tf, {"t": 1000, "o": 50, "h": 56, "l": 47, "c": 54, "v": 120, "x": True})
    # Should still be 1 bar, not 2 — values updated
    assert len(bs.get_ohlcv(sym, tf)["close"]) == 1
    assert bs.get_ohlcv(sym, tf)["close"][-1] == 54.0
    assert bs.get_ohlcv(sym, tf)["high"][-1] == 56.0
    assert bs.get_ohlcv(sym, tf)["volume"][-1] == 120.0

    # New bar with different timestamp appends normally
    bs.add_bar(sym, tf, {"t": 2000, "o": 54, "h": 58, "l": 53, "c": 57, "v": 80, "x": True})
    assert len(bs.get_ohlcv(sym, tf)["close"]) == 2
    assert bs.get_ohlcv(sym, tf)["close"] == [54.0, 57.0]
