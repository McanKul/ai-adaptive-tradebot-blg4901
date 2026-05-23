"""
tests/test_phase_e_heartbeat_gate.py
====================================
Phase E1 — stale-feed gate exposed by ``Streamer.seconds_since_last_message``
and consumed by the entry path in ``LiveEngine``.

The full LiveEngine integration is exercised by ``test_live_engine_full.py``;
here we only verify:

* The streamer property reports +inf before the first message and a
  monotonic delta afterwards.
* ``ExecutionConfig.max_tick_age_seconds`` round-trips through YAML.
"""
from __future__ import annotations
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from live.live_config import ExecutionConfig, LiveConfig


class TestExecutionConfigField(unittest.TestCase):

    def test_default_value(self):
        cfg = ExecutionConfig()
        self.assertEqual(cfg.max_tick_age_seconds, 30.0)

    def test_yaml_round_trip(self):
        cfg = LiveConfig.from_dict({
            "execution": {"max_tick_age_seconds": 12.5}
        })
        self.assertEqual(cfg.execution.max_tick_age_seconds, 12.5)


class TestStreamerHeartbeatProperty(unittest.TestCase):

    def test_returns_inf_before_any_message(self):
        from live.streamer import Streamer
        # Build a streamer without starting it; only the property logic
        # matters here.
        s = Streamer.__new__(Streamer)
        s._last_msg_time = 0.0
        self.assertEqual(s.seconds_since_last_message, float("inf"))

    def test_returns_positive_delta_after_message(self):
        from live.streamer import Streamer
        s = Streamer.__new__(Streamer)
        s._last_msg_time = time.monotonic()
        # Spin briefly; the delta should be non-negative
        delta = s.seconds_since_last_message
        self.assertGreaterEqual(delta, 0.0)
        self.assertLess(delta, 1.0)


if __name__ == "__main__":
    unittest.main()
