"""
tests/test_promotion_stamp.py
==============================
Pin the contract of the promotion-gate stamp written by
`tools/promote_to_live.py:write_promotion_stamp`.

`app.py live` reads only the stamp's existence today, but the file is
also the auditable record of "which artefacts cleared which
thresholds" — its key set is part of the contract with downstream
operators.  Any change to the keys should be a deliberate, tested
change here as well.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.promote_to_live import write_promotion_stamp


def _common_kwargs(config_path: str):
    return dict(
        config=config_path,
        strategy="EMACrossMACDTrend",
        symbol="BTCUSDT",
        timeframe="15m",
        live_trades="logs/live_trades_canary.csv",
        backtest_export="logs/backtest_canary.json",
        eod_rollup="logs/eod_2026-05-06_canary.csv",
        thresholds={
            "match_rate": 0.80,
            "max_side_mismatch": 0,
            "max_pnl_diff_avg": 1.0,
            "min_profit_factor": 1.10,
            "max_drawdown_pct": 3.0,
        },
    )


class TestPromotionStamp(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.out_dir = self._tmp.name
        # Stamp now embeds the config's sha256; write_promotion_stamp
        # opens this path to hash it, so it must exist.
        self.config_path = os.path.join(self.out_dir, "canary.yaml")
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write("strategy:\n  class: EMACrossMACDTrend\nsymbols: [BTCUSDT]\n")

    def tearDown(self):
        self._tmp.cleanup()

    def test_creates_file_with_run_id_in_name(self):
        path = write_promotion_stamp(self.out_dir, "canary",
                                     **_common_kwargs(self.config_path))
        self.assertTrue(os.path.exists(path))
        self.assertEqual(os.path.basename(path), "promotion_gate_canary.json")

    def test_file_is_valid_json(self):
        path = write_promotion_stamp(self.out_dir, "canary",
                                     **_common_kwargs(self.config_path))
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)

    def test_contains_required_top_level_keys(self):
        path = write_promotion_stamp(self.out_dir, "canary",
                                     **_common_kwargs(self.config_path))
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        expected = {
            "run_id", "config", "config_sha256", "strategy", "symbol",
            "timeframe", "passed_at_utc", "live_trades",
            "backtest_export", "eod_rollup", "thresholds",
        }
        self.assertEqual(expected, set(data.keys()))

    def test_thresholds_dict_complete(self):
        path = write_promotion_stamp(self.out_dir, "canary",
                                     **_common_kwargs(self.config_path))
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(
            set(data["thresholds"].keys()),
            {"match_rate", "max_side_mismatch", "max_pnl_diff_avg",
             "min_profit_factor", "max_drawdown_pct"},
        )

    def test_passed_at_utc_iso_format(self):
        from datetime import datetime
        path = write_promotion_stamp(self.out_dir, "canary",
                                     **_common_kwargs(self.config_path))
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Should round-trip through fromisoformat.
        parsed = datetime.fromisoformat(data["passed_at_utc"])
        self.assertIsNotNone(parsed.tzinfo, "passed_at_utc must be timezone-aware")

    def test_creates_out_dir_if_missing(self):
        nested = os.path.join(self.out_dir, "does", "not", "exist")
        path = write_promotion_stamp(nested, "canary",
                                     **_common_kwargs(self.config_path))
        self.assertTrue(os.path.exists(path))

    def test_run_id_propagates_into_payload(self):
        path = write_promotion_stamp(self.out_dir, "smoke_run",
                                     **_common_kwargs(self.config_path))
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["run_id"], "smoke_run")
        self.assertTrue(path.endswith("promotion_gate_smoke_run.json"))

    def test_config_sha256_is_hex_and_matches_file(self):
        path = write_promotion_stamp(self.out_dir, "canary",
                                     **_common_kwargs(self.config_path))
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        digest = data["config_sha256"]
        self.assertIsInstance(digest, str)
        self.assertEqual(len(digest), 64)
        # All hex
        int(digest, 16)  # raises if not hex
        # Hash recomputed against the same file bytes must match.
        import hashlib
        h = hashlib.sha256()
        with open(self.config_path, "rb") as f:
            h.update(f.read())
        self.assertEqual(digest, h.hexdigest())


if __name__ == "__main__":
    unittest.main()
