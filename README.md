# botipman1000 — Quant-grade Bybit Futures Bot

A two-phase systematic futures bot for Bybit USDT perpetuals built around
modern quant-finance practice (López de Prado *Advances in Financial Machine
Learning*): information-driven sampling, triple-barrier labeling, meta-labeling,
CPCV, deflated Sharpe, and vol-targeted fractional-Kelly sizing.

> **Phase 1** — find the best parameter combination via CPCV-aware Optuna
> search on historical data.
> **Phase 2** — run the winning combination in shadow (paper) and then live
> mode against Bybit v5 with full risk controls.

## Indicators & features
- **MACD** → histogram z-score, slope, divergence vs price z-score, crossovers
- **Stochastic** → smoothed %K, %K/%D spread, overbought/oversold duration
- **Donchian breakout channel** → ATR-normalized breakout strength + **Hawkes**
  self-exciting intensity of breakout events
- **Fibonacci** → auto ZigZag swings, distance-to-nearest level (ATR units),
  golden-zone 0.5–0.618 flag
- **Elliott waves** → rule-engine on ZigZag pivots using Fib ratios
  (W2 retrace ∈ 0.382–0.786, W3 ≥ 1.618·W1 and not shortest, W4 no overlap,
  W5 ∈ 0.382–1.618·W1). Output is a wave-confidence feature, not an oracle.
- **Regime** → HMM on returns + |returns|, rolling Hurst exponent
- **Price** → fractional differentiation (FFD) to get a stationary series that
  preserves long-range memory
- **Sampling** → symmetric CUSUM filter on adaptive volatility threshold

## Pipeline

```
raw klines → dollar/CUSUM events → feature matrix →
    primary rule model → triple-barrier labels →
        meta-labeler (LightGBM) → sized orders → risk engine → Bybit
```

## Layout

```
quant/
  config.py          central settings
  data/              Bybit loader, dollar bars, CUSUM, frac-diff
  features/          indicators, zigzag, fibonacci, elliott, hawkes, regime
  labeling/          triple barrier
  models/            primary rules, meta-labeling LightGBM
  sizing/            vol-target + fractional Kelly
  risk/              circuit breakers, VaR, correlation cap
  backtest/          event-driven engine, CPCV, deflated Sharpe, PBO, MC bootstrap
  optimize/          Optuna runner with stability selection
  execution/         Bybit v5 client, drift monitor
  live/              shadow & live runners
main.py              typer CLI
```

## Install (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # edit with your Bybit keys (testnet first)
```

## Quick start (local CLI)

```bash
# 1. fetch history
python main.py fetch --symbol BTCUSDT --interval 5 --days 365

# 2. phase-1 optimization (CPCV + Optuna)
python main.py optimize --symbol BTCUSDT --trials 200 --out artifacts/best_params.json

# 3. train meta-labeling classifier
python main.py train-meta --symbol BTCUSDT --params artifacts/best_params.json

# 4. backtest with winning params
python main.py backtest --symbol BTCUSDT --params artifacts/best_params.json --meta artifacts/meta_model.joblib

# 5. shadow (paper) mode — required before live
python main.py shadow --symbols BTCUSDT,ETHUSDT

# 6. live (testnet by default — requires BYBIT_TESTNET=true in .env)
python main.py live --symbols BTCUSDT --confirm --once
```

## Deploy on Railway

The project ships with a `Dockerfile`, `railway.toml`, `Procfile` and a single
unified entrypoint (`quant/entrypoint.py`) that reads `RUN_MODE` from env.

### 1. Generate artifacts locally first
Run optimization + meta-training on your laptop (fast, no Railway cost) and
commit the resulting files so the Railway worker starts trading immediately:
```bash
python main.py fetch    --symbol BTCUSDT --days 365
python main.py optimize --symbol BTCUSDT --trials 300
python main.py train-meta --symbol BTCUSDT
git add -f artifacts/best_params.json artifacts/meta_model.joblib
git commit -m "chore: trained artifacts"
```

### 2. Create the Railway project
- `railway init`, connect your GitHub repo
- Add a **volume** mounted at `/app/data` (parquet cache) and a second at
  `/app/artifacts` if you prefer not to commit artifacts
- Set environment variables:

| Variable           | Example                 | Notes                                   |
| ------------------ | ----------------------- | --------------------------------------- |
| `RUN_MODE`         | `shadow`                | `shadow` / `live` / `optimize` / `backtest` / `fetch` |
| `BYBIT_API_KEY`    | `...`                   | Bybit v5 key                            |
| `BYBIT_API_SECRET` | `...`                   | Bybit v5 secret                         |
| `BYBIT_TESTNET`    | `true`                  | Flip to `false` only after validation   |
| `SYMBOLS`          | `BTCUSDT,ETHUSDT`       | comma-separated                         |
| `INTERVAL`         | `5`                     | kline minutes                           |
| `POLL_SECONDS`     | `30`                    | loop interval                           |
| `MAX_LEVERAGE`     | `3`                     |                                         |
| `RISK_PER_TRADE`   | `0.01`                  |                                         |
| `DAILY_LOSS_LIMIT` | `0.03`                  |                                         |
| `MAX_DRAWDOWN`     | `0.15`                  |                                         |
| `KELLY_FRACTION`   | `0.25`                  |                                         |
| `CONFIRM_LIVE`     | unset                   | must equal `yes` to enable live orders  |
| `LOG_LEVEL`        | `INFO`                  |                                         |

### 3. Deploy
```bash
railway up
```
Railway builds the Dockerfile. The container runs `python -m quant.entrypoint`
which dispatches based on `RUN_MODE`.

### 4. Runbook
- **Easiest path**: deploy with `RUN_MODE=web` (the Docker default) and drive
  everything from the browser control panel — Fetch → Optimize → Train Meta →
  Backtest → Start runner. See *Web control panel* below.
- Or run headless with `RUN_MODE=shadow` and watch logs for ≥ 1 week
- Re-run `RUN_MODE=optimize` periodically (e.g. weekly) to refresh params
- Switch to `RUN_MODE=live` **only** after setting `CONFIRM_LIVE=yes` AND
  testnet paper-trading looks sane
- Flip `BYBIT_TESTNET=false` only when you are sure

## Web control panel

Set `RUN_MODE=web` (the Docker default) and the container starts a FastAPI +
Jinja control panel on `$PORT` (Railway sets this automatically; defaults to
`8000` locally). It lets you:

- **Dashboard** — runner status, last signals per symbol, risk state, start /
  stop the shadow or live runner, reset risk halts
- **Config** — read & edit `.env` from the browser (secrets masked, leave blank
  to keep current value)
- **Params** — read & edit `artifacts/best_params.json` directly
- **Fetch / Optimize / Train Meta / Backtest** — kick off long-running jobs in
  background threads and watch their progress + results
- **Trades** — paper-trade ledger from the active shadow runner
- **Logs** — tail the latest log file
- **Artifacts** — list & download files under `artifacts/`

### Auth
Set `WEB_PASSWORD` (and optionally `WEB_USERNAME`, default `admin`) before
exposing the panel to the internet. If `WEB_PASSWORD` is unset the panel runs
unauthenticated and a warning is logged on startup — useful for local dev only.

### Run locally
```bash
pip install -r requirements.txt
WEB_PASSWORD=changeme python -m quant.entrypoint  # → http://localhost:8000
```

## Env-var driven entrypoint (same for Docker / Railway / Fly / k8s)

```bash
docker build -t botipman1000 .
docker run --rm -e RUN_MODE=backtest -e SYMBOLS=BTCUSDT \
    -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts \
    botipman1000
```

## Risk defaults
- `risk_per_trade = 1%`, `max_leverage = 3x`, `vol_target = 15%` annualized
- Daily loss kill-switch at −3%, max drawdown circuit breaker at −15%
- Soft breaker at 60% of limits halves size
- Correlation cap 0.75 between concurrent positions
- Historical VaR 95% limit at 5%

## Backtesting — why this isn't toy code
- **Combinatorial Purged K-Fold CV** (embargoed) instead of naive walk-forward
- **Triple-barrier labels** on CUSUM events with ATR-scaled targets
- **Deflated Sharpe Ratio** penalizing multi-trial selection bias
- **Monte Carlo bootstrap** on trade sequences → distribution of Sharpe/DD
- **Probability of Backtest Overfitting** (PBO) estimator
- Realistic execution model: taker/maker fees, slippage, funding-rate drag

## Safety
- Live mode refuses to start without `--confirm`
- Default env sets `BYBIT_TESTNET=true`
- Drift monitor (KS + z-test) halts trading if the live feature distribution
  diverges from the training distribution
- Risk halts persist in the runner's state until manually reset

## Disclaimer
This is a research framework. Crypto perpetual futures can and will liquidate
your account. Run on testnet. Read every function. Nothing here is financial
advice.
