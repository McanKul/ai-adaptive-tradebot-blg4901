"""
tests/test_app_live_gate.py
============================
Pin the promotion-gate enforcement on `app.py live`.

The check is delegated to a small helper (`_check_promotion_stamp`) so
this test does not need to spin up a broker, websocket, or asyncio loop
to exercise the gate logic.
"""
import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module  # noqa: E402


@contextmanager
def _chdir(path: str):
    original = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original)


def _write_stamp(
    logs_dir: str,
    name: str,
    *,
    config: str = "config/canary.yaml",
    strategy: str = "EMACrossMACDTrend",
    age_days: float = 0.0,
) -> str:
    """Write a stamp whose contents pass the validator by default.

    Tests can dial individual fields off-spec to exercise specific
    failure paths.
    """
    from datetime import datetime, timezone, timedelta
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, f"promotion_gate_{name}.json")
    ts = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": name,
            "config": config,
            "strategy": strategy,
            "passed_at_utc": ts,
        }, f)
    return path


class TestCheckPromotionStamp(unittest.TestCase):
    """Direct tests on `_check_promotion_stamp` — no CLI involved."""

    def test_missing_stamp_exits_with_code_2(self):
        with tempfile.TemporaryDirectory() as tmp, _chdir(tmp):
            os.makedirs("config", exist_ok=True)
            cfg = "config/dummy.yaml"
            open(cfg, "w").close()
            with self.assertRaises(SystemExit) as ctx:
                app_module._check_promotion_stamp(cfg, run_id="canary")
            self.assertEqual(ctx.exception.code, 2)

    def test_present_stamp_returns_path(self):
        with tempfile.TemporaryDirectory() as tmp, _chdir(tmp):
            stamp = _write_stamp("logs", "canary", config="canary.yaml")
            cfg = "config/canary.yaml"
            os.makedirs("config", exist_ok=True)
            open(cfg, "w").close()
            result = app_module._check_promotion_stamp(cfg, run_id="canary")
            self.assertEqual(os.path.normpath(result), os.path.normpath(stamp))

    def test_run_id_takes_precedence_over_config_basename(self):
        with tempfile.TemporaryDirectory() as tmp, _chdir(tmp):
            # Stamp uses the run_id, NOT the config basename.
            _write_stamp("logs", "canary_v2", config="canary.yaml")
            cfg = "config/profiles/canary.yaml"  # basename "canary"
            os.makedirs(os.path.dirname(cfg), exist_ok=True)
            open(cfg, "w").close()
            result = app_module._check_promotion_stamp(cfg, run_id="canary_v2")
            self.assertTrue(result.endswith("promotion_gate_canary_v2.json"))

    def test_falls_back_to_config_basename_when_run_id_missing(self):
        with tempfile.TemporaryDirectory() as tmp, _chdir(tmp):
            # matches Path("canary.yaml").stem AND its config basename.
            _write_stamp("logs", "canary", config="canary.yaml")
            cfg = "config/profiles/canary.yaml"
            os.makedirs(os.path.dirname(cfg), exist_ok=True)
            open(cfg, "w").close()
            result = app_module._check_promotion_stamp(cfg, run_id=None)
            self.assertTrue(result.endswith("promotion_gate_canary.json"))


class TestPromotionStampContentValidation(unittest.TestCase):
    """Validate the stamp's contents, not just its existence."""

    def test_unreadable_json_exits(self):
        with tempfile.TemporaryDirectory() as tmp, _chdir(tmp):
            os.makedirs("logs", exist_ok=True)
            path = os.path.join("logs", "promotion_gate_canary.json")
            with open(path, "w") as f:
                f.write("{not valid json")
            os.makedirs("config", exist_ok=True)
            open("config/canary.yaml", "w").close()
            with self.assertRaises(SystemExit) as ctx:
                app_module._check_promotion_stamp(
                    "config/canary.yaml", run_id="canary"
                )
            self.assertEqual(ctx.exception.code, 2)

    def test_config_basename_mismatch_exits(self):
        with tempfile.TemporaryDirectory() as tmp, _chdir(tmp):
            _write_stamp("logs", "canary",
                         config="config/profiles/canary.yaml")
            os.makedirs("config", exist_ok=True)
            open("config/aggressive.yaml", "w").close()
            with self.assertRaises(SystemExit) as ctx:
                app_module._check_promotion_stamp(
                    "config/aggressive.yaml", run_id="canary"
                )
            self.assertEqual(ctx.exception.code, 2)

    def test_config_basename_match_with_different_path_passes(self):
        # Stamp says config="config/profiles/canary.yaml", caller used
        # "canary.yaml" — same basename → accept.
        with tempfile.TemporaryDirectory() as tmp, _chdir(tmp):
            _write_stamp("logs", "canary",
                         config="config/profiles/canary.yaml")
            open("canary.yaml", "w").close()
            result = app_module._check_promotion_stamp(
                "canary.yaml", run_id="canary"
            )
            self.assertTrue(result.endswith("promotion_gate_canary.json"))

    def test_stale_stamp_exits(self):
        with tempfile.TemporaryDirectory() as tmp, _chdir(tmp):
            _write_stamp("logs", "canary", config="canary.yaml",
                         age_days=10.0)
            open("canary.yaml", "w").close()
            with self.assertRaises(SystemExit) as ctx:
                app_module._check_promotion_stamp(
                    "canary.yaml", run_id="canary", max_age_days=7.0
                )
            self.assertEqual(ctx.exception.code, 2)

    def test_stale_window_can_be_widened(self):
        with tempfile.TemporaryDirectory() as tmp, _chdir(tmp):
            _write_stamp("logs", "canary", config="canary.yaml",
                         age_days=10.0)
            open("canary.yaml", "w").close()
            # Caller widens window beyond 10 days — should pass.
            result = app_module._check_promotion_stamp(
                "canary.yaml", run_id="canary", max_age_days=30.0
            )
            self.assertTrue(result.endswith("promotion_gate_canary.json"))

    def test_strategy_mismatch_exits(self):
        with tempfile.TemporaryDirectory() as tmp, _chdir(tmp):
            _write_stamp("logs", "canary", config="canary.yaml",
                         strategy="EMACrossMACDTrend")
            open("canary.yaml", "w").close()
            with self.assertRaises(SystemExit) as ctx:
                app_module._check_promotion_stamp(
                    "canary.yaml", run_id="canary",
                    expected_strategy="DonchianATRVolTarget",
                )
            self.assertEqual(ctx.exception.code, 2)

    def test_strategy_match_passes(self):
        with tempfile.TemporaryDirectory() as tmp, _chdir(tmp):
            _write_stamp("logs", "canary", config="canary.yaml",
                         strategy="EMACrossMACDTrend")
            open("canary.yaml", "w").close()
            result = app_module._check_promotion_stamp(
                "canary.yaml", run_id="canary",
                expected_strategy="EMACrossMACDTrend",
            )
            self.assertTrue(result.endswith("promotion_gate_canary.json"))

    def test_missing_passed_at_utc_exits(self):
        with tempfile.TemporaryDirectory() as tmp, _chdir(tmp):
            os.makedirs("logs", exist_ok=True)
            path = os.path.join("logs", "promotion_gate_canary.json")
            with open(path, "w") as f:
                json.dump({"run_id": "canary", "config": "canary.yaml",
                           "strategy": "EMACrossMACDTrend"}, f)
            open("canary.yaml", "w").close()
            with self.assertRaises(SystemExit) as ctx:
                app_module._check_promotion_stamp(
                    "canary.yaml", run_id="canary"
                )
            self.assertEqual(ctx.exception.code, 2)


class TestCmdLiveGateEnforcement(unittest.TestCase):
    """Verify `_cmd_live` honours --force-live and the stamp check."""

    def _make_args(self, **overrides):
        ns = mock.MagicMock()
        ns.config = overrides.get("config")
        ns.run_id = overrides.get("run_id")
        ns.sentiment = overrides.get("sentiment")
        ns.force_live = overrides.get("force_live", False)
        ns.max_stamp_age_days = overrides.get("max_stamp_age_days", 7.0)
        return ns

    def test_missing_config_exits_before_gate(self):
        args = self._make_args(config="/nonexistent/path.yaml")
        with self.assertRaises(SystemExit) as ctx:
            app_module._cmd_live(args)
        self.assertEqual(ctx.exception.code, 1)

    def test_no_stamp_no_force_exits_with_2(self):
        with tempfile.TemporaryDirectory() as tmp, _chdir(tmp):
            os.makedirs("config", exist_ok=True)
            cfg = "config/canary.yaml"
            open(cfg, "w").close()
            args = self._make_args(config=cfg, run_id="canary",
                                   force_live=False)
            with self.assertRaises(SystemExit) as ctx:
                app_module._cmd_live(args)
            self.assertEqual(ctx.exception.code, 2)

    def test_force_live_bypasses_gate_and_enters_service(self):
        """--force-live skips the stamp check and proceeds to LiveService."""
        with tempfile.TemporaryDirectory() as tmp, _chdir(tmp):
            os.makedirs("config", exist_ok=True)
            cfg = "config/canary.yaml"
            open(cfg, "w").close()
            args = self._make_args(config=cfg, run_id="canary",
                                   force_live=True)

            # Stub out LiveService.run via the import-path used in _cmd_live.
            with mock.patch(
                "core.services.live_service.LiveService"
            ) as svc_cls:
                instance = svc_cls.return_value

                async def _noop(*a, **kw):
                    return None

                instance.run.side_effect = _noop
                app_module._cmd_live(args)

            instance.run.assert_called_once()

    def test_present_stamp_proceeds_without_force(self):
        with tempfile.TemporaryDirectory() as tmp, _chdir(tmp):
            os.makedirs("config", exist_ok=True)
            cfg = "config/canary.yaml"
            open(cfg, "w").close()
            # Empty YAML → LiveConfig.strategy_class default "RSIThreshold".
            _write_stamp("logs", "canary", config="canary.yaml",
                         strategy="RSIThreshold")
            args = self._make_args(config=cfg, run_id="canary",
                                   force_live=False)

            # _cmd_live now also runs the real-money validator; this
            # test focuses on the stamp path, so stub the validator to
            # report a clean config.  A dedicated test below covers the
            # case where the validator surfaces errors.
            with mock.patch(
                "core.config_validator.ConfigValidator"
            ) as cv_cls, mock.patch(
                "core.services.live_service.LiveService"
            ) as svc_cls:
                cv_cls.return_value.validate.return_value = []
                instance = svc_cls.return_value

                async def _noop(*a, **kw):
                    return None

                instance.run.side_effect = _noop
                app_module._cmd_live(args)

            instance.run.assert_called_once()

    def test_real_money_validator_failure_exits_with_2(self):
        """ConfigValidator(real_money=True) errors must short-circuit
        live launch even when a valid promotion-gate stamp exists."""
        with tempfile.TemporaryDirectory() as tmp, _chdir(tmp):
            os.makedirs("config", exist_ok=True)
            cfg = "config/canary.yaml"
            open(cfg, "w").close()
            _write_stamp("logs", "canary", config="canary.yaml",
                         strategy="RSIThreshold")
            args = self._make_args(config=cfg, run_id="canary",
                                   force_live=False)

            with mock.patch(
                "core.config_validator.ConfigValidator"
            ) as cv_cls, mock.patch(
                "core.services.live_service.LiveService"
            ) as svc_cls:
                cv_cls.return_value.validate.return_value = [
                    "[real-money] leverage 25x exceeds hard cap 10x"
                ]
                with self.assertRaises(SystemExit) as ctx:
                    app_module._cmd_live(args)
                self.assertEqual(ctx.exception.code, 2)
                # LiveService must not have been constructed/used.
                svc_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
