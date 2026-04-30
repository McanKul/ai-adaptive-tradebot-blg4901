#!/usr/bin/env python3
"""
app.py
======
Unified CLI entrypoint for the trading system.

Subcommands:
    backtest    Run a single backtest
    live        Start live trading
    dry-run     Start paper trading (no real orders)
    sweep       Run a parameter sweep
    validate    Validate a config file

Usage:
    python app.py backtest --strategy EMACrossMACDTrend --symbol AVAXUSDT \\
                           --strategy-params '{"fast_ema_period":12,"slow_ema_period":26}'

    python app.py live    --config live_config.yaml
    python app.py dry-run --config live_config.yaml

    python app.py sweep --strategy RSIThreshold --symbol AVAXUSDT \\
                        --param-grid grid.yaml --csv-output results.csv

    python app.py validate --config live_config.yaml
"""
import argparse
import asyncio
import json
import logging
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.bootstrap import register_defaults


# ── Helpers ──────────────────────────────────────────────────────────


def _load_param_grid(path: str) -> dict:
    """Load a parameter grid from a YAML or JSON file."""
    if not os.path.exists(path):
        print(f"ERROR: Param-grid file not found: {path}")
        sys.exit(1)

    if path.endswith(".json"):
        with open(path) as f:
            return json.load(f)

    # YAML
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_strategy_params(raw: str | None) -> dict:
    """Parse strategy params from a JSON string or return empty dict."""
    if not raw:
        return {}
    try:
        params = json.loads(raw)
        if not isinstance(params, dict):
            print(f"ERROR: --strategy-params must be a JSON object, got {type(params).__name__}")
            sys.exit(1)
        return params
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in --strategy-params: {e}")
        sys.exit(1)


# ── Subcommand: backtest ─────────────────────────────────────────────


def _add_backtest_parser(subparsers):
    p = subparsers.add_parser("backtest", help="Run a single backtest")

    # Either --strategy or --composite-spec is required (composite skips --strategy)
    p.add_argument("--strategy", default=None,
                   help="Strategy name (e.g. EMACrossMACDTrend). "
                        "Ignored when --composite-spec is set.")
    p.add_argument("--composite-spec", default=None,
                   help="Path to composite YAML/JSON for multi-strategy run")
    p.add_argument("--symbol", default="AVAXUSDT", help="Trading symbol")
    p.add_argument("--timeframe", default="15m", help="Bar timeframe")
    p.add_argument("--strategy-params", default=None,
                   help='Strategy constructor params as JSON (e.g. \'{"rsi_period":14}\')')

    # Sizing
    p.add_argument("--margin-usd", type=float, default=100.0, help="Margin in USD (default)")
    p.add_argument("--notional-usd", type=float, default=None, help="Explicit notional USD")
    p.add_argument("--qty", type=float, default=None, help="Explicit asset quantity")
    p.add_argument("--leverage", type=float, default=10.0, help="Leverage multiplier")
    p.add_argument("--leverage-mode", default="margin", choices=["spot", "margin"])

    # Capital / exit
    p.add_argument("--capital", type=float, default=10_000.0, help="Initial capital")
    p.add_argument("--tp-pct", type=float, default=None, help="Take profit %%")
    p.add_argument("--sl-pct", type=float, default=None, help="Stop loss %%")

    # Tick exit
    p.add_argument("--tick-exit", action="store_true", default=True)
    p.add_argument("--no-tick-exit", action="store_false", dest="tick_exit")

    # Data / realism
    p.add_argument("--data-dir", default="./data/ticks", help="Tick data directory")
    p.add_argument("--realism-config", default=None, help="Path to realism YAML")

    # Cross-validation
    p.add_argument("--cv-method", default="none",
                   choices=["none", "purged_kfold", "walk_forward", "cpcv"],
                   help="Run CV instead of a single backtest")
    p.add_argument("--cv-n-splits", type=int, default=5)
    p.add_argument("--cv-embargo-pct", type=float, default=0.01)
    p.add_argument("--cv-train-pct", type=float, default=0.6,
                   help="Walk-forward training-window fraction")
    p.add_argument("--cv-n-test-splits", type=int, default=2,
                   help="CPCV test folds per split")
    p.add_argument("--cv-aggregate", default="mean", choices=["mean", "median", "min"])
    p.add_argument("--cv-expanding", action="store_true",
                   help="Walk-forward expanding window")

    p.set_defaults(func=_cmd_backtest)


def _cmd_backtest(args):
    from core.services.backtest_service import BacktestService

    composite_spec = getattr(args, "composite_spec", None)
    if not composite_spec and not args.strategy:
        print("ERROR: --strategy is required (or pass --composite-spec).")
        sys.exit(2)

    strategy_params = _parse_strategy_params(args.strategy_params)
    strategy_arg = args.strategy or "composite"

    svc = BacktestService()

    if getattr(args, "cv_method", "none") != "none":
        if composite_spec:
            print("ERROR: --composite-spec is not yet supported with --cv-method.")
            sys.exit(2)
        batch = svc.run_with_cv(
            strategy_name=strategy_arg,
            strategy_params=strategy_params,
            symbol=args.symbol,
            timeframe=args.timeframe,
            capital=args.capital,
            leverage=args.leverage,
            leverage_mode=args.leverage_mode,
            margin_usd=args.margin_usd,
            notional_usd=args.notional_usd,
            fixed_qty=args.qty,
            tp_pct=args.tp_pct,
            sl_pct=args.sl_pct,
            tick_exit=args.tick_exit,
            data_dir=args.data_dir,
            realism_config_path=args.realism_config,
            cv_method=args.cv_method,
            cv_n_splits=args.cv_n_splits,
            cv_embargo_pct=args.cv_embargo_pct,
            cv_train_pct=args.cv_train_pct,
            cv_n_test_splits=args.cv_n_test_splits,
            cv_aggregate=args.cv_aggregate,
            cv_expanding=args.cv_expanding,
        )
        svc.print_cv_result(batch, symbol=args.symbol)
        return

    result = svc.run_single(
        strategy_name=strategy_arg,
        strategy_params=strategy_params,
        symbol=args.symbol,
        timeframe=args.timeframe,
        capital=args.capital,
        leverage=args.leverage,
        leverage_mode=args.leverage_mode,
        margin_usd=args.margin_usd,
        notional_usd=args.notional_usd,
        fixed_qty=args.qty,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        tick_exit=args.tick_exit,
        data_dir=args.data_dir,
        realism_config_path=args.realism_config,
        composite_spec_path=composite_spec,
    )
    svc.print_result(result, symbol=args.symbol)


# ── Subcommand: live ─────────────────────────────────────────────────


def _add_live_parser(subparsers):
    p = subparsers.add_parser("live", help="Start live trading")
    p.add_argument("--config", required=True, help="Path to YAML/JSON config")
    p.set_defaults(func=_cmd_live)


def _cmd_live(args):
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    from core.services.live_service import LiveService
    svc = LiveService()
    asyncio.run(svc.run(args.config, dry_run=False))


# ── Subcommand: dry-run ─────────────────────────────────────────────


def _add_dry_run_parser(subparsers):
    p = subparsers.add_parser("dry-run", help="Start paper trading (no real orders)")
    p.add_argument("--config", required=True, help="Path to YAML/JSON config")
    p.set_defaults(func=_cmd_dry_run)


def _cmd_dry_run(args):
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    from core.services.live_service import LiveService
    svc = LiveService()
    asyncio.run(svc.run(args.config, dry_run=True))


# ── Subcommand: sweep ───────────────────────────────────────────────


def _add_sweep_parser(subparsers):
    p = subparsers.add_parser("sweep", help="Run a parameter sweep")

    p.add_argument("--strategy", required=True, help="Strategy name")
    p.add_argument("--symbol", default="AVAXUSDT", help="Trading symbol")
    p.add_argument("--timeframe", default="15m", help="Bar timeframe")
    p.add_argument("--param-grid", required=True,
                   help="Path to YAML/JSON file with parameter grid")

    # Sizing
    p.add_argument("--margin-usd", type=float, default=100.0)
    p.add_argument("--leverage", type=float, default=10.0)
    p.add_argument("--capital", type=float, default=10_000.0)

    # Exit
    p.add_argument("--tp-pct", type=float, default=None)
    p.add_argument("--sl-pct", type=float, default=None)

    # Data / output
    p.add_argument("--data-dir", default="./data/ticks")
    p.add_argument("--csv-output", default=None, help="CSV output path")
    p.add_argument("--top-n", type=int, default=15, help="Top N results to print")

    p.set_defaults(func=_cmd_sweep)


def _cmd_sweep(args):
    from core.services.sweep_service import SweepService

    param_grid = _load_param_grid(args.param_grid)

    svc = SweepService()
    svc.run_sweep(
        strategy_name=args.strategy,
        param_grid=param_grid,
        symbol=args.symbol,
        timeframe=args.timeframe,
        capital=args.capital,
        leverage=args.leverage,
        margin_usd=args.margin_usd,
        data_dir=args.data_dir,
        csv_output=args.csv_output,
        top_n=args.top_n,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
    )


# ── Subcommand: walk-forward (alias for backtest --cv-method walk_forward) ──


def _add_walk_forward_parser(subparsers):
    p = subparsers.add_parser(
        "walk-forward",
        help="Walk-forward cross-validation (alias for backtest --cv-method walk_forward)",
    )
    p.add_argument("--strategy", required=True)
    p.add_argument("--symbol", default="AVAXUSDT")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--strategy-params", default=None)
    p.add_argument("--margin-usd", type=float, default=100.0)
    p.add_argument("--notional-usd", type=float, default=None)
    p.add_argument("--qty", type=float, default=None)
    p.add_argument("--leverage", type=float, default=10.0)
    p.add_argument("--leverage-mode", default="margin", choices=["spot", "margin"])
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--tp-pct", type=float, default=None)
    p.add_argument("--sl-pct", type=float, default=None)
    p.add_argument("--tick-exit", action="store_true", default=True)
    p.add_argument("--no-tick-exit", action="store_false", dest="tick_exit")
    p.add_argument("--data-dir", default="./data/ticks")
    p.add_argument("--realism-config", default=None)
    p.add_argument("--n-splits", type=int, default=5, dest="cv_n_splits")
    p.add_argument("--embargo-pct", type=float, default=0.01, dest="cv_embargo_pct")
    p.add_argument("--train-pct", type=float, default=0.6, dest="cv_train_pct")
    p.add_argument("--aggregate", default="mean", choices=["mean", "median", "min"],
                   dest="cv_aggregate")
    p.add_argument("--expanding", action="store_true", dest="cv_expanding")
    p.set_defaults(func=_cmd_walk_forward)


def _cmd_walk_forward(args):
    args.cv_method = "walk_forward"
    args.cv_n_test_splits = 2  # unused for walk_forward but keeps sig consistent
    _cmd_backtest(args)


# ── Subcommand: validate ────────────────────────────────────────────


def _add_validate_parser(subparsers):
    p = subparsers.add_parser("validate", help="Validate a config file")
    p.add_argument("--config", required=True, help="Path to YAML/JSON config")
    p.set_defaults(func=_cmd_validate)


def _cmd_validate(args):
    from core.config_validator import ConfigValidator

    validator = ConfigValidator()
    errors = validator.validate(args.config)

    if errors:
        print(f"\nConfig validation FAILED ({len(errors)} error(s)):\n")
        for i, e in enumerate(errors, 1):
            print(f"  {i}. {e}")
        sys.exit(1)

    print(f"\nConfig is valid: {args.config}")


# ── Main ─────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="app",
        description="Unified CLI for the AI-Adaptive Trading Bot",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _add_backtest_parser(subparsers)
    _add_walk_forward_parser(subparsers)
    _add_live_parser(subparsers)
    _add_dry_run_parser(subparsers)
    _add_sweep_parser(subparsers)
    _add_validate_parser(subparsers)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Register all default factories before dispatching
    register_defaults()

    args.func(args)


if __name__ == "__main__":
    main()
