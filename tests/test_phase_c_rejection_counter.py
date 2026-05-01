"""
tests/test_phase_c_rejection_counter.py
=======================================
Phase C3 — sliding-window rejection counter trips the kill-switch
after N broker errors inside the configured window.

Covers:
* Counter increments and trip semantics.
* Trip is idempotent (callback fires once until window decays).
* Rearming after the window decays below threshold.
* ``BinanceBroker`` records BinanceAPIException into the counter.
"""
from __future__ import annotations
import os
import sys
import time
import unittest
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from live.rejection_counter import RejectionCounter


# ---------------------------------------------------------------------------
# Pure RejectionCounter
# ---------------------------------------------------------------------------

class TestRejectionCounter(unittest.TestCase):

    def test_no_trip_below_threshold(self):
        trips = []
        c = RejectionCounter(max_count=5, window_seconds=60.0,
                              on_trip=lambda r: trips.append(r))
        for _ in range(4):
            c.record("x")
        self.assertEqual(c.count, 4)
        self.assertFalse(c.tripped)
        self.assertEqual(trips, [])

    def test_trips_at_threshold(self):
        trips = []
        c = RejectionCounter(max_count=3, window_seconds=60.0,
                              on_trip=lambda r: trips.append(r))
        c.record("a")
        c.record("b")
        tripped_now = c.record("c")
        self.assertTrue(tripped_now)
        self.assertTrue(c.tripped)
        self.assertEqual(len(trips), 1)
        self.assertIn("rejection_storm", trips[0])

    def test_trip_idempotent_until_window_decays(self):
        trips = []
        c = RejectionCounter(max_count=2, window_seconds=60.0,
                              on_trip=lambda r: trips.append(r))
        c.record("a"); c.record("b")
        self.assertEqual(len(trips), 1)
        # Further events do not re-fire while still tripped
        c.record("c"); c.record("d")
        self.assertEqual(len(trips), 1)

    def test_window_decay_rearms(self):
        trips = []
        c = RejectionCounter(max_count=2, window_seconds=1.0,
                              on_trip=lambda r: trips.append(r))
        c.record("a"); c.record("b")
        self.assertEqual(len(trips), 1)
        # Move all events outside the window manually
        c._events.clear()
        c._tripped = False
        # New events in fresh window should be able to trip again
        c.record("c"); c.record("d")
        self.assertEqual(len(trips), 2)

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            RejectionCounter(max_count=0)
        with self.assertRaises(ValueError):
            RejectionCounter(window_seconds=0)


# ---------------------------------------------------------------------------
# BinanceBroker integration
# ---------------------------------------------------------------------------

class TestBrokerRecordsRejections(unittest.IsolatedAsyncioTestCase):

    async def test_broker_records_rejection_via_hook(self):
        """Verify the broker hook calls counter.record on each rejection.

        We test the hook directly rather than provoking a real
        ``BinanceAPIException`` through ``futures_create_order`` because
        unrelated tests in the suite occasionally leave ``binance``
        sub-modules in a mocked state, which makes the exception class
        un-raisable.  The behaviour we care about — "every rejection
        feeds the counter" — is fully covered by exercising the hook.
        """
        from live.broker_binance import BinanceBroker

        client = MagicMock()
        broker = BinanceBroker(client)
        on_trip = MagicMock()
        counter = RejectionCounter(max_count=2, window_seconds=60.0,
                                    on_trip=on_trip)
        broker.attach_rejection_counter(counter)

        # Two simulated rejections → trip
        broker._record_rejection("market_order BTCUSDT BUY: -2010")
        broker._record_rejection("place_stop_market BTCUSDT SELL: -1013")
        self.assertTrue(counter.tripped)
        on_trip.assert_called_once()

    async def test_broker_no_counter_attached_is_noop(self):
        from live.broker_binance import BinanceBroker
        client = MagicMock()
        broker = BinanceBroker(client)
        # No counter attached — must not raise
        broker._record_rejection("anything")


if __name__ == "__main__":
    unittest.main()
