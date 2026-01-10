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
