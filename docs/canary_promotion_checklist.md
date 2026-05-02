# Canary Promotion Checklist

This is the **only** path to enabling real-money live trading. It is
manual on purpose — automating "ship to prod" when prod loses money
is a footgun.

## Pre-flight (must all be GREEN)

1. **Tests** — `pytest -x tests/` is fully green on the deployment
   branch. No skipped real-money tests, no pre-existing fails.

2. **Backtest realism config** is the same one you intend to run
   live. The `--realism-config` YAML must have:
   * `slippage_model` calibrated to recent live data
   * `latency.distribution` matching the geographic broker setup
   * `partial_fill.enabled: true`

3. **Walk-forward on the strategy** — at least 5 folds with
   `--cv-method walk_forward`, aggregate Sharpe ≥ 1.0, max drawdown
   ≤ 5% across folds.

4. **24h dry-run** with the canary YAML
   (`config/profiles/canary.yaml`) on the same instance, same
   network, same broker credentials (`testnet: true` or live
   read-only API key). The 24h must be free of:
   * any drift events (Phase A2 / reconciliation loop)
   * any missing-stop trips (Phase A2 entry guard)
   * any rejection-storm trips (Phase C3)
   * any heartbeat-lost notifications (Phase E1)
   * any liquidation-guard fires (Phase A liquidation guard)

5. **Backtest-vs-live divergence harness** — run
   `python tools/compare_backtest_live.py` over the same time window
   as the dry-run. Required:
   * `match_rate >= 0.80`
   * `side_mismatch == 0`
   * `|pnl_diff_avg_matched| < $1`
   * VERDICT line says "agree within tolerance"

6. **EOD rollup** of the dry-run window
   (`python tools/eod_rollup.py --trade-csv logs/live_trades_canary.csv --date <date> --run-id canary`):
   * `profit_factor >= 1.1`
   * `max_drawdown_pct <= 3.0`
   * `avg_slippage_bps <= 10`
   * `fee / pnl_gross_usd <= 25%` (if it's higher, the strategy is
     not viable at the canary's leverage — **don't fix it by raising
     leverage**)

## Configuration sanity (read the YAML once more)

* `paper: false` is **NOT a config field** — `app.py live` is the
  flag. `app.py dry-run` for paper. Don't fat-finger.
* `testnet: false` ⇒ real money. Verify your `BINANCE_API_KEY` is
  pointed at the production endpoint.
* `liquidation_guard.enabled: true`
* `reconciliation.enabled: true`
* `risk.max_daily_loss` is set (Phase A1 enforces it)
* `global_risk.cooldown_after_losses > 0`
* `execution.max_entry_spread_bps > 0`
* `execution.max_slippage_bps > 0`
* `execution.max_tick_age_seconds > 0`
* `min_24h_volume_usd >= 100_000_000` for canary symbols

## Launch sequence

```bash
# 1) Validate YAML — REAL-MONEY MODE (strict)
#    Catches: leverage > 10, deprecated tickers (FTM/MATIC), volume gate
#    off, allow_reversal=true, daily-loss disabled, liquidation guard
#    off, reconciliation off, > 8 symbols.
python app.py validate --config config/profiles/canary.yaml --real-money

# 2) Final dry-run smoke (24h, must be silent)
python app.py dry-run --config config/profiles/canary.yaml --run-id canary

# 3) **Promotion gate** — runs validate + backtest export + divergence
#    in strict mode + EOD rollup + thresholds.  Refuses to return 0
#    unless every numeric check passes.  This is the single command
#    you actually run before flipping to live.
python tools/promote_to_live.py \
  --config config/profiles/canary.yaml \
  --strategy EMACrossMACDTrend --symbol BTCUSDT --timeframe 15m \
  --realism-config example_realism_config.yaml \
  --live-trades logs/live_trades_canary.csv \
  --date $(date -u +%F) \
  --window "$(date -u -d 'yesterday' +%FT00:00),$(date -u +%FT00:00)" \
  --run-id canary

# 4) Only if step 3 returned 0:
python app.py live --config config/profiles/canary.yaml --run-id canary
```

The promotion gate exits with codes ``1/2/3/4`` (validate / backtest /
divergence / EOD); each stage prints the exact failure reasons before
exiting.  Don't override its decision by hand.

## First 24 hours — watch list

| Metric | Threshold | If breached |
|---|---|---|
| `slippage_bps` per trade | avg ≤ 10 | Verify spread filter active. Don't switch to market mode. |
| `fee / pnl_gross` | ≤ 25% | Strategy not viable at this leverage. Halt. Do not increase leverage. |
| Drift events | 0 | Kill switch. Manual investigation. No auto-resume. |
| Rejection counter | < 1/hour | Symbol filters or rate limits misconfigured. |
| Heartbeat-lost events | 0 | Verify positions on Binance UI; resume only after clean reconcile. |
| Cooldown trigger count | ≤ 1 | More than one strongly suggests bad market timing. Inspect EOD report. |
| Liquidation guard fires | 0 | One = lower leverage; >1 = halt strategy. |
| Daily loss | within `max_daily_loss` | Kill switch trips by design. Wait for next UTC day. |

## 24h decision matrix

After 24h run `tools/eod_rollup.py`:

| EOD outcome | Action |
|---|---|
| `profit_factor < 1.1` OR `max_drawdown_pct > 3` | Halt. Re-evaluate strategy. |
| `match_rate < 0.80` (vs same-window backtest) | Halt. Execution model is wrong; tune `--realism-config`. |
| Above + clean watch list | Continue 48h. Double sermaye. |

## Things to NEVER do during canary

* Raise leverage to "fix" low PnL. The bot is fee-bleeding, not
  capital-starved. Increasing leverage makes a worse trade bigger,
  not better.
* Add new symbols mid-run. Wait for next launch window. Each new
  symbol changes correlation structure and `max_concurrent_positions`
  semantics.
* Disable any of the safety guards. Every one of them has a
  specific failure mode behind it. Disable = walking into that
  failure with a blindfold.
* Force-resume a kill switch. The persistence is intentional. If
  the kill switch tripped, something concrete is wrong — find it
  before re-enabling.

## When to escalate from canary to "real" deployment

Canary's job is to catch the things you can't see in dry-run:
network jitter, exchange rate-limit edges, fill-quality reality,
funding-rate surprises. After the canary returns clean numbers
across 7 consecutive days:

1. Lift `min_24h_volume_usd` to 50M (admit a few more symbols).
2. Lift `leverage` to 5 (still per-symbol capped at 10).
3. Increase `margin_usd` per trade to 100.
4. Add Telegram + log monitoring on a separate machine.
5. Plan broker-fallback (currently only Binance — single point of
   failure for real money).
