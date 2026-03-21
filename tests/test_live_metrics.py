import sys, os, time, tempfile

# Mock binance enums before importing
import types
binance_mod = types.SimpleNamespace()
binance_mod.enums = types.SimpleNamespace(
    SIDE_BUY="BUY", SIDE_SELL="SELL", FUTURE_ORDER_TYPE_MARKET="MARKET"
)
sys.modules.setdefault("binance", binance_mod)
sys.modules.setdefault("binance.enums", binance_mod.enums)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from live.live_metrics import LiveMetrics
from live.position_manager import Position


def _make_pos(symbol, side, qty, entry, exit_price, exit_type="TP",
              bars_held=5, hold_sec=300, strategy="test"):
    pos = Position(symbol=symbol, side=side, qty=qty, entry_price=entry)
    pos.exit = exit_price
    pos.exit_type = exit_type
    pos.exit_ts = pos.open_ts + hold_sec
    pos.bars_held = bars_held
    pos.closed = True
    pos.strategy = strategy
    return pos


def test_record_winning_trade():
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "trades.csv")
        m = LiveMetrics(csv_path=csv_path, start_equity=1000.0)

        pos = _make_pos("BTCUSDT", "BUY", 0.01, 50000.0, 51000.0)
        m.record(pos)

        snap = m.snapshot()
        assert snap["total_trades"] == 1
        assert snap["winning_trades"] == 1
        assert snap["losing_trades"] == 0
        assert snap["win_rate_pct"] == 100.0
        assert snap["total_pnl_usd"] == 10.0  # (51000-50000)*0.01

        # CSV should have header + 1 data row
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 2


def test_record_losing_trade():
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "trades.csv")
        m = LiveMetrics(csv_path=csv_path, start_equity=1000.0)

        pos = _make_pos("ETHUSDT", "BUY", 1.0, 3000.0, 2900.0, exit_type="SL")
        m.record(pos)

        snap = m.snapshot()
        assert snap["total_trades"] == 1
        assert snap["winning_trades"] == 0
        assert snap["losing_trades"] == 1
        assert snap["total_pnl_usd"] == -100.0


def test_short_trade_pnl():
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "trades.csv")
        m = LiveMetrics(csv_path=csv_path, start_equity=1000.0)

        pos = _make_pos("DOGEUSDT", "SELL", 10000.0, 0.10, 0.09, exit_type="TP")
        m.record(pos)

        snap = m.snapshot()
        assert snap["total_pnl_usd"] == 100.0  # (0.10-0.09)*10000


def test_multiple_trades_stats():
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "trades.csv")
        m = LiveMetrics(csv_path=csv_path, start_equity=1000.0)

        # Win
        m.record(_make_pos("BTCUSDT", "BUY", 0.01, 50000, 52000, bars_held=3))
        # Loss
        m.record(_make_pos("ETHUSDT", "BUY", 1.0, 3000, 2800, bars_held=7))
        # Win
        m.record(_make_pos("DOGEUSDT", "SELL", 10000, 0.10, 0.08, bars_held=2))

        snap = m.snapshot()
        assert snap["total_trades"] == 3
        assert snap["winning_trades"] == 2
        assert snap["losing_trades"] == 1
        assert snap["win_rate_pct"] == round(2/3*100, 2)
        assert snap["best_trade_usd"] == 200.0   # DOGE short: (0.10-0.08)*10000
        assert snap["worst_trade_usd"] == -200.0  # ETH long: (2800-3000)*1


def test_drawdown_tracking():
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "trades.csv")
        m = LiveMetrics(csv_path=csv_path, start_equity=1000.0)

        m.update_equity(1100.0)  # new peak
        m.update_equity(990.0)   # drawdown from 1100

        snap = m.snapshot()
        assert snap["peak_equity"] == 1100.0
        expected_dd = (1100 - 990) / 1100 * 100
        assert abs(snap["max_drawdown_pct"] - expected_dd) < 0.01


def test_per_symbol_pnl():
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "trades.csv")
        m = LiveMetrics(csv_path=csv_path, start_equity=1000.0)

        m.record(_make_pos("BTCUSDT", "BUY", 0.01, 50000, 51000))  # +10
        m.record(_make_pos("BTCUSDT", "BUY", 0.01, 51000, 50500))  # -5
        m.record(_make_pos("ETHUSDT", "SELL", 1.0, 3000, 2900))    # +100

        snap = m.snapshot()
        assert snap["symbol_pnl"]["BTCUSDT"] == 5.0    # 10 - 5
        assert snap["symbol_pnl"]["ETHUSDT"] == 100.0
