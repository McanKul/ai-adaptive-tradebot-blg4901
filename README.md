# AI-Adaptive Trading Bot

Tick-level backtesting, multi-strategy live trading, news sentiment integration, and AFML-inspired model selection — built for Binance USDT-M Futures.

> **Risk Disclaimer:** This software is for educational and research purposes. Algorithmic trading carries substantial financial risk. Past backtest performance does **not** guarantee future returns. Never trade with money you cannot afford to lose. The authors accept no liability for financial losses.

---

## Quickstart

### 1. Prerequisites

- Python 3.10+
- [TA-Lib C library](https://ta-lib.org/) installed on your system
- Binance Futures account (for live/dry-run trading)

### 2. Install

```bash
# Clone
git clone <repo-url> && cd bitirme

# (Recommended) Create a virtual environment
python -m venv .venv && source .venv/bin/activate  # Linux/Mac
python -m venv .venv && .venv\Scripts\activate      # Windows

# Install pinned dependencies (reproducible)
pip install -r requirements.txt
```

### 3. Environment Variables

```bash
cp .env.example .env
# Edit .env and fill in your Binance API keys
```

| Variable | Required | Description |
|---|---|---|
| `BINANCE_API_KEY` | Yes (live) | Binance API key |
| `BINANCE_API_SECRET` | Yes (live) | Binance API secret |
| `GOOGLE_API_KEY` | No | Gemini sentiment provider |
| `OPENAI_API_KEY` | No | OpenAI sentiment provider |
| `TELEGRAM_BOT_TOKEN` | No | Telegram notifications |
| `TELEGRAM_CHAT_ID` | No | Telegram chat ID |

### 4. Run a Backtest

```bash
# Single backtest — EMACrossMACDTrend on AVAXUSDT 15m
python run_emacross_backtest.py

# Parameter sweep (108 combinations, ~30 min)
python run_emacross_sweep.py

# Flexible CLI backtest
python run_unified_backtest.py --symbol DOGEUSDT --leverage 10 --margin-usd 100
```

### 5. Run Live Trading (Dry-Run First!)

```bash
# Paper trading — no real orders, uses live market data
python live_runner.py --config live_config_emacross.yaml --dry-run

# Real trading (use with extreme caution)
python live_runner.py --config live_config_emacross.yaml
```

### 6. Docker

```bash
# Build
docker build -t tradebot .

# Dry-run
docker run --env-file .env tradebot python live_runner.py --config live_config_emacross.yaml --dry-run

# Production
docker run -d --env-file .env --restart unless-stopped tradebot
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Entry Points                             │
│  live_runner.py (production)  │  run_*_backtest.py (research)   │
└───────────────┬───────────────┴──────────────┬──────────────────┘
                │                              │
     ┌──────────▼──────────┐        ┌──────────▼──────────┐
     │    Live Trading      │        │     Backtesting      │
     │  live/live_engine.py │        │  Backtest/engine.py  │
     │  live/broker_binance │        │  Backtest/runner.py  │
     │  live/position_mgr   │        │  Backtest/metrics.py │
     │  live/global_risk    │        │  Backtest/scoring/   │
     └──────────┬───────────┘        └──────────┬──────────┘
                │                              │
     ┌──────────▼──────────────────────────────▼──────────┐
     │                  Shared Layer                       │
     │  Interfaces/  — IStrategy, IBroker, IClient         │
     │  Strategy/    — EMACrossMACDTrend, Donchian, RSI    │
     │  news/        — DDG source, Gemini/OpenAI sentiment │
     │  utils/       — logger, bar_store, leverage_utils   │
     └────────────────────────────────────────────────────┘
```

### Key Design Principles

- **Unified Strategy Interface** — Same `IStrategy.on_bar()` works in backtest and live.
- **Tick-level backtesting** — Strategies only see bars; intrabar TP/SL checked on raw ticks.
- **Deterministic replay** — Fixed seeds + disk tick data = reproducible results.
- **AFML methodology** — Purged K-fold CV, Deflated Sharpe Ratio, selection bias warnings.
- **Dual-layer risk** — Exchange-side SL/TP orders + local software checks as safety net.
- **Realistic costs** — Taker/maker fees, slippage, spread, funding rates, borrow costs.

### Strategies

| Strategy | Description | Key Indicators |
|---|---|---|
| `EMACrossMACDTrend` | Trend-following with multi-confirmation | EMA cross, MACD histogram, ADX, ATR trailing stop |
| `DonchianATRVolTarget` | Breakout with volatility targeting | Donchian channel, ATR, EMA/ADX filter |
| `RSIThreshold` | Mean-reversion / overbought-oversold | RSI with configurable TP/SL |

---

## Safe Rollout Path

Follow this progression — **never skip steps**:

```
1. BACKTEST          →  Validate strategy on historical data
   python run_emacross_backtest.py

2. DRY-RUN (paper)   →  Verify live data flow, no real money
   python live_runner.py --config config/profiles/safe.yaml --dry-run

3. TESTNET           →  Real orders on Binance testnet
   # Set api.testnet: true in config

4. LIMITED LIVE      →  Small margin ($5-10), single coin, safe profile
   python live_runner.py --config config/profiles/safe.yaml

5. SCALED LIVE       →  Increase margin/coins gradually
   python live_runner.py --config config/profiles/standard.yaml
```

---

## Config Profiles

Pre-built risk profiles in `config/profiles/`:

| Profile | Margin | Leverage | Max Concurrent | Daily Loss Limit | Use Case |
|---|---|---|---|---|---|
| `safe.yaml` | $5 | 5x | 1 | $10 | First live deployment, learning |
| `standard.yaml` | $10 | 10x | 2 | $30 | Normal operation |
| `aggressive.yaml` | $25 | 15x | 3 | $75 | Experienced, high-conviction |

```bash
python live_runner.py --config config/profiles/safe.yaml --dry-run
```

---

## Execution Realism Notes

### Intra-bar price model (`intrabar_price_model`)

When order latency keeps a fill inside the same bar (no bar-boundary crossing),
the `intrabar_price_model` setting controls whether the fill price is perturbed:

| Value | Behaviour |
|-------|-----------|
| `"none"` **(default)** | Fill stays at the bar's open (or the field set by `price_latency_bar_field`). No intra-bar perturbation — deterministic and bias-free. |
| `"gaussian_clamped"` | A Gaussian shift proportional to `latency / bar_duration` is applied, clamped to `[low, high]`. **This is an approximation** — it uses OHLC bounds that are only known after the bar closes, so it introduces a mild look-ahead. Use for stress-testing, not for unbiased PnL estimation. |

Recommendation: leave the default (`"none"`) for all research runs.
Switch to `"gaussian_clamped"` only when you want to study sensitivity to
intra-bar price uncertainty — and document that the results carry a small
optimistic bias from the OHLC clamp.

---

## Troubleshooting

### API Keys

- **"Binance client could not be created"** — Check `BINANCE_API_KEY` and `BINANCE_API_SECRET` in `.env`. Ensure the key has Futures trading permission enabled.
- **Testnet vs Mainnet** — Set `api.testnet: true` in your config YAML for testnet. Testnet and mainnet use different API keys.

### TA-Lib Installation

- **Linux:** `sudo apt-get install -y build-essential wget && wget https://github.com/TA-Lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && tar -xzf ta-lib-0.6.4-src.tar.gz && cd ta-lib-0.6.4 && ./configure --prefix=/usr && make && sudo make install`
- **Windows:** Download wheel from [ta-lib unofficial](https://github.com/cgohlke/talib-build/releases) matching your Python version, then `pip install TA_Lib‑<version>.whl`
- **Mac:** `brew install ta-lib`

### Symbol Filters

- **"Symbol not found"** — Ensure you use USDT-M Futures symbols (e.g., `BTCUSDT`, `ETHUSDT`). Check Binance Futures for available pairs.
- **Lot size errors** — The bot auto-rounds to exchange precision. If you get errors, check `max_position_notional` is reasonable.

### Rate Limits

- Default: 1000 requests/minute (Binance allows 1200).
- If you see 429 errors, reduce `rate_limit.requests_per_minute` in config.
- `exchange_info_ttl_sec: 300` caches symbol info for 5 minutes.

### Common Issues

- **"No tick data found"** — Fetch tick data first: `python tools/fetch_ticks.py --symbol AVAXUSDT`
- **Low win rate in backtest** — Check timeframe (15m recommended for EMACross, 5m has too much noise).
- **Position not closing** — Verify `exit_on_macd_cross: true` in strategy params (critical: 66% vs 23% win rate).

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_emacross_strategy.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing
```

---

## Project Structure

```
├── Backtest/           # Backtesting engine, metrics, scoring, CV splits
├── Strategy/           # Strategy implementations (EMACross, Donchian, RSI)
├── Interfaces/         # Abstract interfaces (IStrategy, IBroker, etc.)
├── live/               # Live trading (engine, broker, positions, risk)
├── news/               # News sentiment (DDG, Gemini, OpenAI)
├── utils/              # Shared utilities (logger, bar_store, leverage)
├── config/profiles/    # Risk profiles (safe, standard, aggressive)
├── data/               # Tick data storage
├── tests/              # Test suite
├── tools/              # Utility scripts (fetch ticks, smoke tests)
├── live_runner.py      # Production entry-point
├── run_*_backtest.py   # Backtest scripts
└── Dockerfile          # Container deployment
```

---

## License

This project is part of a university thesis (BLG4901). See repository for details.
