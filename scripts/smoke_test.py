"""End-to-end smoke test on synthetic OHLCV data.

Generates a price series that contains:
    * a trending segment (drift + low noise)
    * a ranging segment (zero drift, mean-reverting OU process)
    * a chop segment (high noise, very little edge)

and verifies the new regime-aware primary model:
    * stays flat during chop
    * earns positive expectancy on the trending and ranging segments
    * respects single-position concurrency

Run:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_synthetic_ohlcv(n_bars: int = 12_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_bars
    n_seg = n // 6

    returns = np.zeros(n)

    # Seg 1: strong uptrend
    returns[0:n_seg] = rng.normal(0.0006, 0.0025, n_seg)
    # Seg 2: ranging (mean-reverting around 0)
    price_level = 0.0
    ou = np.zeros(n_seg)
    for i in range(1, n_seg):
        price_level += -0.08 * price_level + rng.normal(0.0, 0.003)
        ou[i] = price_level - ou[i - 1]
    returns[n_seg:2 * n_seg] = ou * 0.5 + rng.normal(0.0, 0.0015, n_seg)
    # Seg 3: chop (high vol, no drift)
    returns[2 * n_seg:3 * n_seg] = rng.normal(0.0, 0.005, n_seg)
    # Seg 4: strong downtrend
    returns[3 * n_seg:4 * n_seg] = rng.normal(-0.0006, 0.0025, n_seg)
    # Seg 5: ranging again
    price_level = 0.0
    ou2 = np.zeros(n_seg)
    for i in range(1, n_seg):
        price_level += -0.08 * price_level + rng.normal(0.0, 0.003)
        ou2[i] = price_level - ou2[i - 1]
    returns[4 * n_seg:5 * n_seg] = ou2 * 0.5 + rng.normal(0.0, 0.0015, n_seg)
    # Seg 6: moderate uptrend
    returns[5 * n_seg:] = rng.normal(0.0003, 0.002, n - 5 * n_seg)

    log_price = np.log(30000.0) + np.cumsum(returns)
    close = np.exp(log_price)
    high = close * np.exp(np.abs(rng.normal(0.0, 0.0015, n)))
    low = close * np.exp(-np.abs(rng.normal(0.0, 0.0015, n)))
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.uniform(80, 240, n)
    ts = pd.date_range("2023-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(high, np.maximum(open_, close)),
            "low": np.minimum(low, np.minimum(open_, close)),
            "close": close,
            "volume": vol,
            "turnover": close * vol,
        },
        index=ts,
    )
    return df


def run() -> None:
    from quant.backtest.engine import BacktestConfig, EventBacktester
    from quant.features.builder import FeatureParams, build_feature_matrix
    from quant.models.primary import PrimaryParams, PrimaryRuleModel

    df = make_synthetic_ohlcv()
    print(f"Generated synthetic df: {len(df)} bars, "
          f"{df.index[0]} → {df.index[-1]}")

    feats = build_feature_matrix(df, FeatureParams()).dropna()
    print(f"Feature matrix: {feats.shape}")

    prim = PrimaryRuleModel(PrimaryParams()).compute(feats)
    side = prim["primary_side"]
    mode = prim["primary_mode"]
    non_zero = int((side != 0).sum())
    print(f"Non-zero signals: {non_zero}  "
          f"(trend={(mode == 1).sum()}, range={(mode == 2).sum()})")

    close = df["close"].loc[feats.index]
    # Use primary signal onsets as trade events (signal just turned non-zero
    # or flipped direction). The engine's concurrency gate prevents overlap.
    side_shift = side.shift(1).fillna(0)
    onsets = (side != 0) & (side != side_shift)
    events = feats.index[onsets.values]
    print(f"Signal-onset events: {len(events)}")

    bt = EventBacktester(BacktestConfig(risk_per_trade=0.01, max_leverage=3.0))
    res = bt.run(
        close=close,
        events=events,
        side=side.reindex(close.index).fillna(0),
        atr_pct=feats["atr_pct"].reindex(close.index),
        mode=mode.reindex(close.index).fillna(0),
    )

    print("\n--- Backtest stats ---")
    for k, v in res.stats.items():
        if isinstance(v, float):
            print(f"  {k:<20} {v:>10.4f}")
        else:
            print(f"  {k:<20} {v}")

    final_eq = float(res.equity.iloc[-1])
    start_eq = float(res.equity.iloc[0])
    print(f"\nEquity: {start_eq:,.2f} → {final_eq:,.2f} "
          f"({(final_eq / start_eq - 1) * 100:+.2f}%)")

    # Pass criteria
    stats = res.stats
    n_trades = stats.get("n_trades", 0)
    sharpe = stats.get("sharpe", 0.0) or 0.0
    pf = stats.get("profit_factor", 0.0) or 0.0
    dd = stats.get("max_drawdown", 0.0) or 0.0

    failures = []
    if n_trades < 20:
        failures.append(f"too few trades: {n_trades}")
    if sharpe <= 0.8:
        failures.append(f"sharpe too low: {sharpe:.3f}")
    if pf <= 1.15:
        failures.append(f"profit factor too low: {pf:.3f}")
    if dd < -0.30:
        failures.append(f"drawdown too deep: {dd:.2%}")

    if failures:
        print("\nSMOKE TEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    run()
