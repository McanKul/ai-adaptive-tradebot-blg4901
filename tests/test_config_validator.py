"""
tests/test_config_validator.py
===============================
Tests for ConfigValidator.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config_validator import ConfigValidator
from core.bootstrap import register_defaults
from core.factories.strategy_factory import StrategyFactory


class TestConfigValidator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        StrategyFactory._registry.clear()
        register_defaults()
        cls.validator = ConfigValidator()

    def test_missing_file(self):
        errors = self.validator.validate("nonexistent_file_xyz.yaml")
        self.assertEqual(len(errors), 1)
        self.assertIn("not found", errors[0])

    def test_valid_config(self):
        # Use the existing example config
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'example_live_config.yaml'
        )
        if not os.path.exists(config_path):
            self.skipTest("example_live_config.yaml not found")
        errors = self.validator.validate(config_path)
        self.assertEqual(errors, [], f"Expected no errors, got: {errors}")

    def test_malformed_yaml(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("{{invalid yaml content::")
            f.flush()
            errors = self.validator.validate(f.name)
        os.unlink(f.name)
        self.assertTrue(len(errors) > 0)
        self.assertIn("parse", errors[0].lower())

    def test_invalid_strategy_name(self):
        """Config with an unresolvable strategy should report error."""
        yaml_content = """
strategy:
  class: "TotallyBogusStrategy"
  params: {}
symbols:
  - BTCUSDT
timeframe: "15m"
sizing:
  mode: margin_usd
  margin_usd: 10
  leverage: 10
risk:
  max_concurrent_positions: 2
  max_daily_loss: 50
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            errors = self.validator.validate(f.name)
        os.unlink(f.name)
        strategy_errors = [e for e in errors if "strategy" in e.lower() or "resolve" in e.lower()]
        self.assertTrue(len(strategy_errors) > 0, f"Expected strategy error, got: {errors}")


if __name__ == "__main__":
    unittest.main()
