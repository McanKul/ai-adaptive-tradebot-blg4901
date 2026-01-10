import sys, os
import numpy as np
import pandas as pd

# Ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Strategy.RSIThreshold import Strategy as RSIStrategy


def make_df(closes):
    n = len(closes)
    return pd.DataFrame({
        "open":   closes,
        "high":   [c+1 for c in closes],
        "low":    [c-1 for c in closes],
        "close":  closes,
        "volume": [1000]*n,
    })


def test_rsi_strategy_overbought_sells():
    # Monotonically increasing closes -> RSI high -> expect -1
    closes = list(range(1, 60))
    df = make_df(closes)
    strat = RSIStrategy(bars=df, rsi_period=14, rsi_overbought=70, rsi_oversold=30)
    sig = strat.generate_signal()
    assert sig in (-1, +1, None)
    assert sig == -1


def test_rsi_strategy_oversold_buys():
    # Monotonically decreasing closes -> RSI low -> expect +1
    closes = list(range(100, 40, -1))
    df = make_df(closes)
    strat = RSIStrategy(bars=df, rsi_period=14, rsi_overbought=70, rsi_oversold=30)
    sig = strat.generate_signal()
    assert sig in (-1, +1, None)
    assert sig == +1
