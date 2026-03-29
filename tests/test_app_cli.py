"""
tests/test_app_cli.py
======================
Tests for the unified CLI entrypoint argument parsing.
"""
import os
import sys
import subprocess
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

APP_PY = os.path.join(os.path.dirname(__file__), '..', 'app.py')


class TestAppCLI(unittest.TestCase):
    """Test CLI argument parsing via subprocess."""

    def _run(self, *args, expect_rc=0):
        result = subprocess.run(
            [sys.executable, APP_PY] + list(args),
            capture_output=True, text=True, timeout=30,
        )
        if expect_rc is not None:
            self.assertEqual(
                result.returncode, expect_rc,
                f"Expected rc={expect_rc}, got {result.returncode}.\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        return result

    # ── Help ──────────────────────────────────────────────────────────

    def test_no_args_shows_help(self):
        r = self._run(expect_rc=0)
        self.assertIn("backtest", r.stdout)
        self.assertIn("live", r.stdout)
        self.assertIn("dry-run", r.stdout)
        self.assertIn("sweep", r.stdout)
        self.assertIn("validate", r.stdout)

    def test_help_flag(self):
        r = self._run("--help", expect_rc=0)
        self.assertIn("Unified CLI", r.stdout)

    def test_backtest_help(self):
        r = self._run("backtest", "--help", expect_rc=0)
        self.assertIn("--strategy", r.stdout)
        self.assertIn("--symbol", r.stdout)
        self.assertIn("--strategy-params", r.stdout)

    def test_sweep_help(self):
        r = self._run("sweep", "--help", expect_rc=0)
        self.assertIn("--param-grid", r.stdout)
        self.assertIn("--csv-output", r.stdout)

    def test_validate_help(self):
        r = self._run("validate", "--help", expect_rc=0)
        self.assertIn("--config", r.stdout)

    # ── Validate subcommand ──────────────────────────────────────────

    def test_validate_valid_config(self):
        config = os.path.join(os.path.dirname(__file__), '..', 'example_live_config.yaml')
        if not os.path.exists(config):
            self.skipTest("example_live_config.yaml not found")
        r = self._run("validate", "--config", config, expect_rc=0)
        self.assertIn("valid", r.stdout.lower())

    def test_validate_missing_config(self):
        r = self._run("validate", "--config", "does_not_exist.yaml", expect_rc=1)

    # ── Missing required args ────────────────────────────────────────

    def test_backtest_missing_strategy_fails(self):
        r = self._run("backtest", expect_rc=2)  # argparse exits with 2

    def test_live_missing_config_fails(self):
        r = self._run("live", expect_rc=2)

    def test_sweep_missing_args_fails(self):
        r = self._run("sweep", expect_rc=2)


if __name__ == "__main__":
    unittest.main()
