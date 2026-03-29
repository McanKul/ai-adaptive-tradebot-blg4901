"""
tests/test_strategy_factory.py
===============================
Tests for StrategyFactory: registry CRUD, resolution, instantiation.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.factories.strategy_factory import StrategyFactory
from core.bootstrap import register_defaults


class TestStrategyFactory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Clear and re-register to get a clean state
        StrategyFactory._registry.clear()
        register_defaults()

    def test_default_registry_has_three_strategies(self):
        available = StrategyFactory.list_available()
        self.assertIn("RSIThreshold", available)
        self.assertIn("EMACrossMACDTrend", available)
        self.assertIn("DonchianATRVolTarget", available)
        self.assertEqual(len(available), 3)

    def test_resolve_class_rsi(self):
        cls = StrategyFactory.resolve_class("RSIThreshold")
        self.assertEqual(cls.__name__, "Strategy")

    def test_resolve_class_emacross(self):
        cls = StrategyFactory.resolve_class("EMACrossMACDTrend")
        self.assertEqual(cls.__name__, "Strategy")

    def test_resolve_class_donchian(self):
        cls = StrategyFactory.resolve_class("DonchianATRVolTarget")
        self.assertEqual(cls.__name__, "Strategy")

    def test_resolve_class_dotted_path(self):
        """Fallback: treat name as dotted import path."""
        cls = StrategyFactory.resolve_class("Strategy.RSIThreshold.Strategy")
        self.assertEqual(cls.__name__, "Strategy")

    def test_resolve_class_unknown_raises(self):
        with self.assertRaises(ValueError) as ctx:
            StrategyFactory.resolve_class("NonExistentStrategy")
        self.assertIn("Cannot resolve", str(ctx.exception))

    def test_create_instantiates(self):
        """create() should return an instance with the given params."""
        strategy = StrategyFactory.create("RSIThreshold", rsi_period=7)
        # The strategy should have the param stored
        self.assertEqual(strategy.rsi_period, 7)

    def test_custom_register(self):
        """register() should add a new entry resolvable by name."""
        StrategyFactory.register("CustomTest", "Strategy.RSIThreshold")
        cls = StrategyFactory.resolve_class("CustomTest")
        self.assertEqual(cls.__name__, "Strategy")
        self.assertIn("CustomTest", StrategyFactory.list_available())
        # Cleanup
        del StrategyFactory._registry["CustomTest"]

    def test_list_available_returns_list(self):
        result = StrategyFactory.list_available()
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(n, str) for n in result))


if __name__ == "__main__":
    unittest.main()
