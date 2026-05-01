"""
tests/test_live_engine_safety.py
================================
Regression tests for three pre-canary safety fixes:

1. ``live/live_engine.py`` actually imports ``asyncio`` (it uses
   ``asyncio.create_task`` and ``asyncio.sleep`` at runtime — without
   the import, the engine throws ``NameError`` the first time the
   reconciliation loop or shutdown path runs).

2. The ``risk_block_entries`` flag inside the bar-processing loop is
   defaulted to ``True`` BEFORE the try/except that calls
   ``check_account_risk``.  A risk-check exception used to leave the
   name unbound; later code path then read it and raised a second
   ``NameError`` (or, if the name *was* bound from a previous bar,
   silently let entries through during a degraded state).

3. ``EMACrossMACDTrend.allow_reversal`` defaults to ``False``.
   Reversal == close+open as two non-atomic orders; the close-leg
   fill is not awaited before the open-leg goes out, so a rejected
   close can leave the bot doubled up.  Real-money default must be
   conservative.
"""
from __future__ import annotations
import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Fix 1 — asyncio import present
# ---------------------------------------------------------------------------

class TestAsyncioImport(unittest.TestCase):

    def test_asyncio_is_in_module_namespace(self):
        from live import live_engine
        # If the import is missing, attribute access here would fail
        self.assertTrue(hasattr(live_engine, "asyncio"))
        # And the symbol must be the real asyncio module
        import asyncio as real_asyncio
        self.assertIs(live_engine.asyncio, real_asyncio)

    def test_runtime_path_uses_asyncio(self):
        """Sanity: confirm the engine actually exercises asyncio.* so the
        import is not dead weight."""
        from live import live_engine
        src = inspect.getsource(live_engine.LiveEngine)
        # Calls that would hit NameError without the import
        self.assertIn("asyncio.create_task", src)
        self.assertIn("asyncio.sleep", src)


# ---------------------------------------------------------------------------
# Fix 2 — risk_block_entries default-deny
# ---------------------------------------------------------------------------

class TestRiskBlockEntriesDefault(unittest.TestCase):

    def test_default_set_before_try_block(self):
        """The source must contain a ``risk_block_entries = True`` line that
        precedes the ``try:`` that calls ``check_account_risk``.  This is
        the structural guarantee against the previous unbound-on-exception
        bug.  Brittle on whitespace, but the alternative (spinning up
        the whole engine) is heavier and slower.
        """
        from live import live_engine
        src = inspect.getsource(live_engine.LiveEngine.run)
        assign_idx = src.find("risk_block_entries = True")
        try_idx = src.find("check_account_risk")
        self.assertNotEqual(assign_idx, -1,
                            "missing risk_block_entries = True default")
        self.assertNotEqual(try_idx, -1,
                            "could not locate check_account_risk call")
        self.assertLess(assign_idx, try_idx,
                        "default-deny assignment must precede the try block")


# ---------------------------------------------------------------------------
# Fix 3 — EMACrossMACDTrend conservative defaults
# ---------------------------------------------------------------------------

class TestEMACrossMACDTrendDefaults(unittest.TestCase):

    def test_allow_reversal_defaults_false(self):
        from Strategy.EMACrossMACDTrend import Strategy as EMACrossMACDTrend
        sig = inspect.signature(EMACrossMACDTrend.__init__)
        param = sig.parameters.get("allow_reversal")
        self.assertIsNotNone(param, "allow_reversal parameter missing")
        self.assertEqual(param.default, False,
                         "allow_reversal must default to False for canary")

    def test_instance_honors_default(self):
        from Strategy.EMACrossMACDTrend import Strategy as EMACrossMACDTrend
        s = EMACrossMACDTrend()
        self.assertFalse(s.allow_reversal,
                         "fresh instance must have allow_reversal=False")


if __name__ == "__main__":
    unittest.main()
