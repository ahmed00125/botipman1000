"""Seed the BybitLoader parquet cache with synthetic OHLCV bars.

Lets us exercise the full pipeline (fetch -> train-meta -> backtest) in
environments where api.bybit.com is unreachable. Writes a realistic-looking
GBM-with-regime-switching series so the primary model actually produces
non-trivial events.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
INTERVAL_MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 5
DAYS = int(sys.argv[3]) if len(sys.argv) > 3 else 365
SEED = 42

rng = np.random.default_rng(SEED)

bars_per_day = (24 * 60) // INTERVAL_MIN
n = DAYS * bars_per_day
end = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
# Push the last bar 1 hour into the future so BybitLoader.load() sees the
# cache as fully up-to-date and skips the incremental HTTP fetch.
end = end + pd.Timedelta(hours=1)
end_ms = int(end.timestamp() * 1000)
start_ms = end_ms - n * INTERVAL_MIN * 60 * 1000
idx = pd.date_range(
    start=pd.Timestamp(start_ms, unit="ms", tz="UTC"),
    periods=n,
    freq=f"{INTERVAL_MIN}min",
)

# Regime-switching drift and volatility. 4 regimes: trend up, trend down,
# chop-low-vol, chop-high-vol. Each regime lasts between 1-5 days.
mu_per_regime = np.array([+0.00020, -0.00020, 0.00000, 0.00000])
sig_per_regime = np.array([0.0020, 0.0022, 0.0008, 0.0035])

mu = np.empty(n)
sig = np.empty(n)
i = 0
while i < n:
    regime = int(rng.integers(0, len(mu_per_regime)))
    length = int(rng.integers(1, 6) * bars_per_day)
    j = min(i + length, n)
    mu[i:j] = mu_per_regime[regime]
    sig[i:j] = sig_per_regime[regime]
    i = j

log_ret = rng.normal(mu, sig)
start_price = 60_000.0 if SYMBOL.startswith("BTC") else 3_000.0
close = start_price * np.exp(np.cumsum(log_ret))

# Synthesize OHLV from close with a small intrabar wiggle
wiggle = np.abs(rng.normal(0, 0.0015, size=n)) * close
open_ = np.concatenate([[start_price], close[:-1]])
high = np.maximum(open_, close) + wiggle
low = np.minimum(open_, close) - wiggle
volume = rng.lognormal(mean=3.0, sigma=0.6, size=n)
turnover = volume * close

df = pd.DataFrame(
    {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "turnover": turnover,
    },
    index=idx,
)
df.index.name = "start"

out = Path("data/raw") / f"{SYMBOL}_{INTERVAL_MIN}.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out)
print(f"wrote {out}  rows={len(df)}  range={df.index[0]} -> {df.index[-1]}")
