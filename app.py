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


def _load_param_grid(path: str) -> tuple[dict, list]:
    """Load a parameter grid + optional constraints from a YAML/JSON file.

    Two formats are supported:

    * **Flat** (legacy):
        ``{rsi_period: [14, 21], threshold: [0.5, 0.7]}``  → grid only.
    * **Structured**:
        ``{grid: {...}, constraints: [{type: less_than, a: x, b: y}, ...]}``
        → grid + SearchSpace constraints (rsi_oversold < rsi_overbought etc.).
    """
    if not os.path.exists(path):
        print(f"ERROR: Param-grid file not found: {path}")
        sys.exit(1)

    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
    else:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)

    if isinstance(data, dict) and "grid" in data:
        return data["grid"], data.get("constraints") or []
    return data, []


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

    # Trade export — feeds tools/compare_backtest_live.py as a divergence gate
    p.add_argument("--export-trades", default=None, metavar="PATH",
                   help="Write the round-trip trade list to JSON for the "
                        "backtest-vs-live divergence harness")

    # Partial fills — backtest defaults to clean fills.  Turning these
    # on with ``--enable-partial-fills`` exposes the strategy to the
    # liquidity-ratio fill model already implemented in the engine
    # (see ``Backtest/execution_models.py``).  The same knobs can be
    # set in the realism YAML; CLI flags win for one-off comparisons.
    p.add_argument("--enable-partial-fills", action="store_true", default=None,
                   help="Turn on partial-fill simulation (engine default off)")
    p.add_argument("--liquidity-scale", type=float, default=None,
                   help="Fill ratio = bar.volume / (qty * liquidity_scale); "
                        "lower = stricter liquidity (default 10.0)")
    p.add_argument("--min-fill-ratio", type=float, default=None,
                   help="Below this ratio orders are rejected (default 0.1)")

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
        enable_partial_fills=args.enable_partial_fills,
        liquidity_scale=args.liquidity_scale,
        min_fill_ratio=args.min_fill_ratio,
    )
    svc.print_result(result, symbol=args.symbol)

    # --export-trades: serialise the trade list for the divergence
    # harness.  This is the missing rung between backtest and live
    # promotion gates — without it ``compare_backtest_live.py`` has
    # nothing to consume on the backtest side.  The runtime context
    # (data type, partial-fill state, exit knobs, realism YAML) is
    # passed alongside so the JSON tells you WHICH backtest produced
    # the trades — without it the divergence run is comparing
    # apples to oranges.
    export_path = getattr(args, "export_trades", None)
    if export_path:
        svc.export_trades(
            result, export_path,
            strategy_name=args.strategy or "composite",
            symbol=args.symbol,
            run_metadata={
                "timeframe": args.timeframe,
                "data_dir": args.data_dir,
                "realism_config": args.realism_config,
                "tick_exit_enabled": bool(args.tick_exit),
                "partial_fills_cli": args.enable_partial_fills,
                "liquidity_scale_cli": args.liquidity_scale,
                "min_fill_ratio_cli": args.min_fill_ratio,
                "leverage": args.leverage,
                "leverage_mode": args.leverage_mode,
                "tp_pct_cli": args.tp_pct,
                "sl_pct_cli": args.sl_pct,
            },
        )


# ── Subcommand: live ─────────────────────────────────────────────────


def _add_live_parser(subparsers):
    p = subparsers.add_parser("live", help="Start live trading")
    p.add_argument("--config", required=True, help="Path to YAML/JSON config")
    p.add_argument("--run-id", default=None,
                   help="Tag log/state files with this id (default: config.name)")
    p.add_argument("--sentiment", choices=["on", "off"], default=None,
                   help="Force sentiment on/off regardless of YAML")
    p.add_argument("--force-live", action="store_true",
                   help="Bypass the promotion-gate stamp check.  Dangerous: "
                        "use only when the gate has been verified manually "
                        "out-of-band.")
    p.add_argument("--max-stamp-age-days", type=float, default=7.0,
                   help="Reject promotion-gate stamps older than this many "
                        "days (default 7).  A long-stale stamp likely means "
                        "the gate was passed against a different code or "
                        "data state.")
    p.set_defaults(func=_cmd_live)


def _check_promotion_stamp(
    config_path: str,
    run_id: str | None,
    *,
    expected_strategy: str | None = None,
    max_age_days: float = 7.0,
) -> str:
    """Resolve the promotion-gate stamp path and validate its contents.

    Exits with code 2 on any failure with a precise reason so the
    operator knows whether to re-promote or rebuild.

    Checks:
      1. Stamp file exists at ``logs/promotion_gate_<run_id>.json``
         (or the config basename when ``--run-id`` is omitted).
      2. Stamp parses as JSON.
      3. ``config`` field's basename matches the ``--config`` basename
         (path-relative invocations on the same canary YAML pass).
      4. ``passed_at_utc`` is within ``max_age_days`` from now.
      5. ``strategy`` matches ``expected_strategy`` when provided
         (i.e. the strategy_class loaded from the config).
      6. ``config_sha256`` matches the hash of the live config bytes
         (defends against content edits with the same file name).
         Stamps written by an older release without this field emit a
         warning instead of failing — a one-release transition window.

    Returns the resolved stamp path on success.
    """
    import hashlib
    import json
    from datetime import datetime, timezone, timedelta
    from pathlib import Path

    effective = run_id or Path(config_path).stem
    stamp_path = os.path.join("logs", f"promotion_gate_{effective}.json")

    if not os.path.exists(stamp_path):
        print(
            f"ERROR: Promotion gate stamp not found at {stamp_path}.\n"
            f"  Run tools/promote_to_live.py first or pass --force-live "
            f"to bypass intentionally."
        )
        sys.exit(2)

    try:
        with open(stamp_path, "r", encoding="utf-8") as f:
            stamp = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR: Promotion gate stamp at {stamp_path} is unreadable: {e}")
        sys.exit(2)

    # 3) config basename match — tolerant to path-relative invocation.
    stamp_cfg = stamp.get("config", "")
    if Path(stamp_cfg).name != Path(config_path).name:
        print(
            f"ERROR: Stamp at {stamp_path} was issued for config "
            f"'{stamp_cfg}', but live was invoked with '{config_path}'.\n"
            f"  Re-run tools/promote_to_live.py against the current "
            f"config or pass --force-live to bypass intentionally."
        )
        sys.exit(2)

    # 4) freshness window.
    raw_ts = stamp.get("passed_at_utc")
    if not isinstance(raw_ts, str):
        print(f"ERROR: Stamp at {stamp_path} has no passed_at_utc timestamp.")
        sys.exit(2)
    try:
        ts = datetime.fromisoformat(raw_ts)
    except ValueError as e:
        print(f"ERROR: Stamp at {stamp_path} has malformed passed_at_utc "
              f"({raw_ts!r}): {e}")
        sys.exit(2)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - ts
    if age > timedelta(days=max_age_days):
        print(
            f"ERROR: Promotion gate stamp at {stamp_path} is "
            f"{age.days} days old (max {max_age_days:.1f}).\n"
            f"  Re-run tools/promote_to_live.py or pass --force-live "
            f"to bypass intentionally."
        )
        sys.exit(2)

    # 5) strategy match.
    if expected_strategy is not None:
        stamp_strategy = stamp.get("strategy")
        if stamp_strategy != expected_strategy:
            print(
                f"ERROR: Stamp at {stamp_path} cleared "
                f"strategy='{stamp_strategy}', but config requests "
                f"strategy='{expected_strategy}'.\n"
                f"  Re-run tools/promote_to_live.py against the current "
                f"strategy or pass --force-live to bypass intentionally."
            )
            sys.exit(2)

    # 6) config content hash.  Basename match (step 3) catches renames;
    # this catches in-place content edits that keep the same filename.
    stamp_hash = stamp.get("config_sha256")
    if stamp_hash is None:
        print(
            f"WARNING: Stamp at {stamp_path} has no config_sha256 (legacy "
            f"stamp). Re-run tools/promote_to_live.py to get a hashed "
            f"stamp; this fallback will be removed in a future release."
        )
    else:
        h = hashlib.sha256()
        try:
            with open(config_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
        except OSError as e:
            print(f"ERROR: cannot hash config {config_path}: {e}")
            sys.exit(2)
        live_hash = h.hexdigest()
        if live_hash != stamp_hash:
            print(
                f"ERROR: Stamp at {stamp_path} was issued against config "
                f"sha256={stamp_hash[:12]}…, but current "
                f"'{config_path}' hashes to {live_hash[:12]}….\n"
                f"  The config has been edited since the gate passed. "
                f"Re-run tools/promote_to_live.py or pass --force-live "
                f"to bypass intentionally."
            )
            sys.exit(2)

    return stamp_path


def _cmd_live(args):
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    if args.force_live:
        print("WARNING: --force-live set — promotion gate bypassed. "
              "Real money at risk.")
    else:
        # Run real-money strict validation up front. The validate
        # subcommand exposes the same checks, but live trading must not
        # depend on the operator remembering to run it first.
        from core.config_validator import ConfigValidator
        rm_errors = ConfigValidator().validate(args.config, real_money=True)
        if rm_errors:
            print(
                f"\nREAL-MONEY config validation FAILED "
                f"({len(rm_errors)} error(s)):\n"
            )
            for i, e in enumerate(rm_errors, 1):
                print(f"  {i}. {e}")
            print(
                "\nFix the above before launching live, or pass --force-live "
                "to bypass intentionally."
            )
            sys.exit(2)

        # Pre-parse the config so the stamp's strategy field can be
        # checked against what live is actually about to run.  Cheap —
        # YAML parse is sub-millisecond — and prevents the "stamped
        # against EMACross, launched RSIThreshold" foot-gun.
        from live.live_config import LiveConfig
        if args.config.endswith(".json"):
            cfg_for_validation = LiveConfig.from_json(args.config)
        else:
            cfg_for_validation = LiveConfig.from_yaml(args.config)
        stamp = _check_promotion_stamp(
            args.config,
            args.run_id,
            expected_strategy=cfg_for_validation.strategy_class,
            max_age_days=args.max_stamp_age_days,
        )
        print(f"Promotion gate stamp verified: {stamp}")

    from core.services.live_service import LiveService
    svc = LiveService()
    asyncio.run(svc.run(
        args.config, dry_run=False,
        run_id=args.run_id, sentiment_override=args.sentiment,
    ))


# ── Subcommand: dry-run ─────────────────────────────────────────────


def _add_dry_run_parser(subparsers):
    p = subparsers.add_parser("dry-run", help="Start paper trading (no real orders)")
    p.add_argument("--config", required=True, help="Path to YAML/JSON config")
    p.add_argument("--run-id", default=None,
                   help="Tag log/state files with this id. Use distinct ids "
                        "when running A/B sessions in parallel "
                        "(e.g. --run-id sentiment_on vs --run-id sentiment_off).")
    p.add_argument("--sentiment", choices=["on", "off"], default=None,
                   help="Force sentiment on/off regardless of YAML "
                        "(handy for the thesis demo A/B recipe)")
    p.set_defaults(func=_cmd_dry_run)


def _cmd_dry_run(args):
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    from core.services.live_service import LiveService
    svc = LiveService()
    asyncio.run(svc.run(
        args.config, dry_run=True,
        run_id=args.run_id, sentiment_override=args.sentiment,
    ))


# ── Subcommand: sweep ───────────────────────────────────────────────


def _add_sweep_parser(subparsers):
    p = subparsers.add_parser("sweep", help="Run a parameter sweep")

    p.add_argument("--strategy", required=True, help="Strategy name")
    p.add_argument("--symbol", default="AVAXUSDT", help="Trading symbol")
    p.add_argument("--timeframe", default="15m", help="Bar timeframe")
    p.add_argument("--param-grid", required=True,
                   help="Path to YAML/JSON with parameter grid (flat dict or "
                        "{grid: ..., constraints: [...]})")

    # Sizing / exit
    p.add_argument("--margin-usd", type=float, default=100.0)
    p.add_argument("--leverage", type=float, default=10.0)
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--tp-pct", type=float, default=None)
    p.add_argument("--sl-pct", type=float, default=None)

    # Data / output / realism
    p.add_argument("--data-dir", default="./data/ticks")
    p.add_argument("--csv-output", default=None, help="CSV output path")
    p.add_argument("--top-n", type=int, default=15, help="Top N results to print")
    p.add_argument("--realism-config", default=None,
                   help="Path to realism YAML (latency/slippage/funding/borrow)")

    # Cross-validation (parameter selection with leakage-safe scoring)
    p.add_argument("--cv-method", default="none",
                   choices=["none", "purged_kfold", "walk_forward", "cpcv"],
                   help="Score each parameter combo through CV instead of "
                        "a single full-period run")
    p.add_argument("--cv-n-splits", type=int, default=5)
    p.add_argument("--cv-embargo-pct", type=float, default=0.01)
    p.add_argument("--cv-train-pct", type=float, default=0.6,
                   help="Walk-forward training-window fraction")
    p.add_argument("--cv-n-test-splits", type=int, default=2,
                   help="CPCV test folds per split")
    p.add_argument("--cv-aggregate", default="mean", choices=["mean", "median", "min"])
    p.add_argument("--cv-expanding", action="store_true")

    # Hyperband (successive halving) — drop bottom 1/halving_factor of params
    # after each fold so bad combos don't keep burning compute through every
    # rung.  Incompatible with cv_method='cpcv'.
    p.add_argument("--cv-hyperband", action="store_true",
                   help="Enable Hyperband-style successive halving across CV folds")
    p.add_argument("--cv-halving-factor", type=int, default=2,
                   help="Keep top 1/halving_factor of params after each rung "
                        "(default 2 = keep top half)")
    p.add_argument("--cv-min-active", type=int, default=2,
                   help="Floor on active params at each rung (default 2)")

    # Selector — minimum-quality filters
    p.add_argument("--min-trades", type=int, default=None,
                   help="Drop combos with fewer than N trades")
    p.add_argument("--min-sharpe", type=float, default=None)
    p.add_argument("--max-dd", type=float, default=None,
                   help="Drop combos whose max drawdown exceeds this fraction "
                        "(0.20 = 20%%)")
    p.add_argument("--min-win-rate", type=float, default=None)
    p.add_argument("--cv-stability-weight", type=float, default=None,
                   help="Penalty multiplier on cv_score_std (only with --cv-method)")
    p.add_argument("--max-cv-std", type=float, default=None,
                   help="Reject combos with cv_score_std above this (--cv-method)")

    p.set_defaults(func=_cmd_sweep)


def _cmd_sweep(args):
    from core.services.sweep_service import SweepService

    param_grid, constraints = _load_param_grid(args.param_grid)

    svc = SweepService()
    svc.run_sweep(
        strategy_name=args.strategy,
        param_grid=param_grid,
        constraints=constraints,
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
        realism_config_path=args.realism_config,
        cv_method=args.cv_method,
        cv_n_splits=args.cv_n_splits,
        cv_embargo_pct=args.cv_embargo_pct,
        cv_train_pct=args.cv_train_pct,
        cv_n_test_splits=args.cv_n_test_splits,
        cv_aggregate=args.cv_aggregate,
        cv_expanding=args.cv_expanding,
        cv_hyperband=args.cv_hyperband,
        cv_halving_factor=args.cv_halving_factor,
        cv_min_active=args.cv_min_active,
        min_trades=args.min_trades,
        min_sharpe=args.min_sharpe,
        max_drawdown=args.max_dd,
        min_win_rate=args.min_win_rate,
        cv_stability_weight=args.cv_stability_weight,
        max_cv_std=args.max_cv_std,
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
    p.add_argument("--real-money", action="store_true",
                   help="Apply strict real-money checks (leverage cap, "
                        "liquidity gate, no deprecated tickers, daily-loss "
                        "enforced, allow_reversal off, etc.).  Required "
                        "before promoting a config to live trading.")
    p.set_defaults(func=_cmd_validate)


def _cmd_validate(args):
    from core.config_validator import ConfigValidator

    validator = ConfigValidator()
    errors = validator.validate(args.config, real_money=args.real_money)

    if errors:
        scope = "REAL-MONEY" if args.real_money else "BASIC"
        print(f"\n{scope} config validation FAILED ({len(errors)} error(s)):\n")
        for i, e in enumerate(errors, 1):
            print(f"  {i}. {e}")
        sys.exit(1)

    scope = "real-money checks passed" if args.real_money else "basic checks passed"
    print(f"\nConfig is valid ({scope}): {args.config}")


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
