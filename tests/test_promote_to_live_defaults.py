"""Tests for the LiveConfig-driven defaults and guards in
``tools.promote_to_live``.

The gate must not be able to pass with backtest parameters that
disagree with the live config it is supposed to clear.  These tests
exercise the small pure helpers — ``_resolve_backtest_inputs`` and
``_count_live_trades`` — so we don't need to spin up the full
subprocess chain.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.promote_to_live import (  # noqa: E402
    _count_live_trades,
    _resolve_backtest_inputs,
    compute_config_sha256,
)


def _write(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(content))


def _ns(**kwargs) -> argparse.Namespace:
    """Build an args namespace pre-populated with the gate's defaults."""
    defaults = dict(
        config=None,
        strategy=None,
        symbol=None,
        timeframe=None,
        strategy_params=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


SINGLE_SYM_CONFIG = """
    strategy:
      class: EMACrossMACDTrend
      params:
        fast_ema_period: 12
        slow_ema_period: 21
    symbols:
      - BTCUSDT
    timeframe: "15m"
    sizing:
      mode: margin_usd
      margin_usd: 10.0
      leverage: 5
"""

MULTI_SYM_CONFIG = """
    strategy:
      class: EMACrossMACDTrend
      params: {}
    symbols:
      - BTCUSDT
      - ETHUSDT
    timeframe: "15m"
"""


class TestResolveBacktestInputs(unittest.TestCase):
    def test_defaults_populated_from_live_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "canary.yaml")
            _write(cfg_path, SINGLE_SYM_CONFIG)
            args = _ns(config=cfg_path)
            rc = _resolve_backtest_inputs(args)
            self.assertIsNone(rc)
            self.assertEqual(args.strategy, "EMACrossMACDTrend")
            self.assertEqual(args.symbol, "BTCUSDT")
            self.assertEqual(args.timeframe, "15m")
            # strategy_params should be a JSON string matching the YAML
            import json
            self.assertEqual(
                json.loads(args.strategy_params),
                {"fast_ema_period": 12, "slow_ema_period": 21},
            )

    def test_matching_cli_overrides_pass_through(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "canary.yaml")
            _write(cfg_path, SINGLE_SYM_CONFIG)
            args = _ns(
                config=cfg_path,
                strategy="EMACrossMACDTrend",
                symbol="BTCUSDT",
                timeframe="15m",
            )
            self.assertIsNone(_resolve_backtest_inputs(args))

    def test_strategy_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "canary.yaml")
            _write(cfg_path, SINGLE_SYM_CONFIG)
            args = _ns(config=cfg_path, strategy="RSIThreshold")
            self.assertEqual(_resolve_backtest_inputs(args), 1)

    def test_symbol_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "canary.yaml")
            _write(cfg_path, SINGLE_SYM_CONFIG)
            args = _ns(config=cfg_path, symbol="ETHUSDT")
            self.assertEqual(_resolve_backtest_inputs(args), 1)

    def test_timeframe_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "canary.yaml")
            _write(cfg_path, SINGLE_SYM_CONFIG)
            args = _ns(config=cfg_path, timeframe="5m")
            self.assertEqual(_resolve_backtest_inputs(args), 1)

    def test_multi_symbol_config_is_rejected_with_5(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "canary.yaml")
            _write(cfg_path, MULTI_SYM_CONFIG)
            args = _ns(config=cfg_path)
            self.assertEqual(_resolve_backtest_inputs(args), 5)

    def test_malformed_strategy_params_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "canary.yaml")
            _write(cfg_path, SINGLE_SYM_CONFIG)
            args = _ns(config=cfg_path, strategy_params="not-json")
            self.assertEqual(_resolve_backtest_inputs(args), 1)


class TestCountLiveTrades(unittest.TestCase):
    def test_zero_when_file_missing(self) -> None:
        self.assertEqual(_count_live_trades("/nonexistent/path.csv"), 0)

    def test_zero_when_only_header(self) -> None:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False, encoding="utf-8", newline=""
        ) as f:
            csv.writer(f).writerow(["ts", "symbol", "pnl"])
            path = f.name
        try:
            self.assertEqual(_count_live_trades(path), 0)
        finally:
            os.unlink(path)

    def test_counts_non_header_rows(self) -> None:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False, encoding="utf-8", newline=""
        ) as f:
            w = csv.writer(f)
            w.writerow(["ts", "symbol", "pnl"])
            for i in range(7):
                w.writerow([i, "BTCUSDT", 0.1])
            path = f.name
        try:
            self.assertEqual(_count_live_trades(path), 7)
        finally:
            os.unlink(path)


class TestComputeConfigSha256(unittest.TestCase):
    def test_returns_64_hex_chars(self) -> None:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("foo: bar\n")
            path = f.name
        try:
            digest = compute_config_sha256(path)
            self.assertEqual(len(digest), 64)
            int(digest, 16)  # hex check
        finally:
            os.unlink(path)

    def test_changes_when_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "a.yaml")
            p2 = os.path.join(tmp, "b.yaml")
            _write(p1, "foo: bar\n")
            _write(p2, "foo: baz\n")
            self.assertNotEqual(
                compute_config_sha256(p1), compute_config_sha256(p2),
            )


if __name__ == "__main__":
    unittest.main()
