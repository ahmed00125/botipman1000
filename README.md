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

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # edit with your Bybit keys (testnet first)
```

## Quick start

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
