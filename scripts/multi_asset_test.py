"""Multi-asset, multi-timeframe robustness test.

Runs the regime-aware strategy across a matrix of (asset_profile, timeframe,
seed) combinations and reports per-cell metrics plus an aggregate summary.

Each ``asset_profile`` is a synthetic generator parameterised to mimic a
realistic crypto behaviour:

    btc_like  : large-cap, mixed regime, ~2% daily vol
    eth_like  : large-cap, more directional, ~3% daily vol
    sol_like  : mid-cap, choppy, frequent regime flips, ~5% daily vol
    range_alt : low-vol coin spending most of its time in a range
    trend_alt : memecoin-style persistent trend with deep pullbacks

For every profile we test on four timeframes (5m, 15m, 1h, 4h) with three
random seeds each. That's 60 independent backtests — enough to expose
parameter brittleness without bootstrapping noise.

NOTE: this is **synthetic** data because the sandbox blocks crypto APIs.
Use ``python main.py backtest <symbol> --interval <m>`` on a machine with
internet access to validate the same code on real Bybit klines.

Run:
    python scripts/multi_asset_test.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------- helpers
def _ohlcv_from_returns(
    returns: np.ndarray,
    bar_minutes: int,
    seed: int,
    start_price: float = 30_000.0,
    wick_bps: float = 15.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 9001)
    n = len(returns)
    close = np.exp(np.log(start_price) + np.cumsum(returns))
    high = close * np.exp(np.abs(rng.normal(0.0, wick_bps / 1e4, n)))
    low = close * np.exp(-np.abs(rng.normal(0.0, wick_bps / 1e4, n)))
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.uniform(80, 240, n)
    ts = pd.date_range("2024-01-01", periods=n, freq=f"{bar_minutes}min", tz="UTC")
    return pd.DataFrame(
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


def _scale_vol(daily_vol: float, bar_minutes: int) -> float:
    bars_per_day = 24 * 60 / bar_minutes
    return daily_vol / np.sqrt(bars_per_day)


# ------------------------------------------------------ asset-profile generators
def gen_btc_like(n_bars: int, bar_minutes: int, seed: int) -> pd.DataFrame:
    """Mixed regime: ~2% daily vol, alternating trend / range / chop."""
    rng = np.random.default_rng(seed)
    sd = _scale_vol(0.02, bar_minutes)
    drift = sd * 0.05
    n_seg = n_bars // 6
    rets = np.zeros(n_bars)
    rets[0:n_seg] = rng.normal(drift * 1.5, sd, n_seg)
    rets[n_seg:2 * n_seg] = rng.normal(0.0, sd * 0.7, n_seg)        # quiet range
    rets[2 * n_seg:3 * n_seg] = rng.normal(-drift * 1.5, sd, n_seg)
    rets[3 * n_seg:4 * n_seg] = rng.normal(0.0, sd * 1.4, n_seg)    # high vol chop
    rets[4 * n_seg:5 * n_seg] = rng.normal(drift, sd, n_seg)
    rets[5 * n_seg:] = rng.normal(0.0, sd, n_bars - 5 * n_seg)
    return _ohlcv_from_returns(rets, bar_minutes, seed)


def gen_eth_like(n_bars: int, bar_minutes: int, seed: int) -> pd.DataFrame:
    """More directional, ~3% daily vol, longer trends."""
    rng = np.random.default_rng(seed + 1)
    sd = _scale_vol(0.03, bar_minutes)
    drift = sd * 0.08
    n_seg = n_bars // 4
    rets = np.zeros(n_bars)
    rets[0:n_seg] = rng.normal(drift * 1.6, sd, n_seg)
    rets[n_seg:2 * n_seg] = rng.normal(-drift * 1.6, sd, n_seg)
    rets[2 * n_seg:3 * n_seg] = rng.normal(drift * 1.2, sd * 1.1, n_seg)
    rets[3 * n_seg:] = rng.normal(0.0, sd * 0.9, n_bars - 3 * n_seg)
    return _ohlcv_from_returns(rets, bar_minutes, seed, start_price=2_500.0)


def gen_sol_like(n_bars: int, bar_minutes: int, seed: int) -> pd.DataFrame:
    """High vol mid-cap, frequent regime flips, ~5% daily vol."""
    rng = np.random.default_rng(seed + 2)
    sd = _scale_vol(0.05, bar_minutes)
    drift = sd * 0.06
    n_seg = n_bars // 8
    signs = [1, -1, 1, -1, 1, -1, 1, -1]
    rets = np.zeros(n_bars)
    for i, s in enumerate(signs):
        lo = i * n_seg
        hi = (i + 1) * n_seg if i < 7 else n_bars
        rets[lo:hi] = rng.normal(s * drift, sd, hi - lo)
    return _ohlcv_from_returns(rets, bar_minutes, seed, start_price=120.0)


def gen_range_alt(n_bars: int, bar_minutes: int, seed: int) -> pd.DataFrame:
    """Mostly mean-reverting OU with brief breakouts."""
    rng = np.random.default_rng(seed + 3)
    sd = _scale_vol(0.025, bar_minutes)
    rets = np.zeros(n_bars)
    level = 0.0
    for i in range(1, n_bars):
        level += -0.08 * level + rng.normal(0.0, sd * 0.7)
        rets[i] = level - (rets[i - 1] if i > 1 else 0)
    rets = rets * 0.5 + rng.normal(0.0, sd * 0.6, n_bars)
    # inject two short breakout bursts
    burst = n_bars // 10
    rets[3 * burst:4 * burst] += rng.normal(sd * 0.4, sd, burst)
    rets[7 * burst:8 * burst] += rng.normal(-sd * 0.4, sd, burst)
    return _ohlcv_from_returns(rets, bar_minutes, seed, start_price=8.0)


def gen_trend_alt(n_bars: int, bar_minutes: int, seed: int) -> pd.DataFrame:
    """Persistent trend with deep ATR pullbacks (memecoin-ish)."""
    rng = np.random.default_rng(seed + 4)
    sd = _scale_vol(0.06, bar_minutes)
    drift = sd * 0.12
    rets = rng.normal(drift, sd, n_bars)
    # carve out periodic pullbacks
    pull = n_bars // 12
    for i in range(2, 12, 3):
        rets[i * pull:(i + 1) * pull] = rng.normal(-drift * 0.8, sd, pull)
    return _ohlcv_from_returns(rets, bar_minutes, seed, start_price=0.05)


PROFILES: dict[str, Callable[[int, int, int], pd.DataFrame]] = {
    "btc_like": gen_btc_like,
    "eth_like": gen_eth_like,
    "sol_like": gen_sol_like,
    "range_alt": gen_range_alt,
    "trend_alt": gen_trend_alt,
}

TIMEFRAMES = [5, 15, 60, 240]   # minutes


# ------------------------------------------------------------------ run a cell
@dataclass
class CellResult:
    profile: str
    tf_min: int
    seed: int
    n_trades: int
    sharpe: float
    profit_factor: float
    win_rate: float
    max_dd: float
    equity_pct: float
    trend_n: int
    range_n: int


def _bars_for(timeframe_min: int) -> int:
    """Pick a bar count that gives ~3-4 months of data per cell."""
    if timeframe_min <= 5:
        return 18_000      # ~62 days
    if timeframe_min <= 15:
        return 12_000      # ~125 days
    if timeframe_min <= 60:
        return 6_000       # ~250 days
    return 3_000           # ~500 days


def run_cell(profile: str, tf_min: int, seed: int) -> CellResult:
    from quant.backtest.engine import BacktestConfig, EventBacktester
    from quant.features.builder import FeatureParams, build_feature_matrix
    from quant.models.primary import PrimaryParams, PrimaryRuleModel

    df = PROFILES[profile](_bars_for(tf_min), tf_min, seed)
    feats = build_feature_matrix(df, FeatureParams()).dropna()
    if feats.empty:
        return CellResult(profile, tf_min, seed, 0, 0, 0, 0, 0, 0, 0, 0)
    prim = PrimaryRuleModel(PrimaryParams()).compute(feats)
    side = prim["primary_side"]
    mode = prim["primary_mode"]
    side_shift = side.shift(1).fillna(0)
    onsets = (side != 0) & (side != side_shift)
    events = feats.index[onsets.values]
    close = df["close"].loc[feats.index]

    bars_per_year = int(365 * 24 * 60 / tf_min)
    cfg = BacktestConfig(
        risk_per_trade=0.01,
        max_leverage=3.0,
        bars_per_year=bars_per_year,
    )
    bt = EventBacktester(cfg)
    res = bt.run(
        close=close,
        events=events,
        side=side.reindex(close.index).fillna(0),
        atr_pct=feats["atr_pct"].reindex(close.index),
        mode=mode.reindex(close.index).fillna(0),
    )
    s = res.stats
    eq_pct = float(res.equity.iloc[-1] / res.equity.iloc[0] - 1) * 100
    return CellResult(
        profile=profile,
        tf_min=tf_min,
        seed=seed,
        n_trades=int(s.get("n_trades", 0)),
        sharpe=float(s.get("sharpe", 0) or 0),
        profit_factor=float(s.get("profit_factor", 0) or 0),
        win_rate=float(s.get("win_rate", 0) or 0),
        max_dd=float(s.get("max_drawdown", 0) or 0),
        equity_pct=eq_pct,
        trend_n=int(s.get("trend_n", 0)),
        range_n=int(s.get("range_n", 0)),
    )


# --------------------------------------------------------------------- output
def _fmt_row(r: CellResult) -> str:
    return (
        f"  {r.profile:<10} {r.tf_min:>4}m  seed={r.seed}  "
        f"n={r.n_trades:>4}  sharpe={r.sharpe:>7.2f}  pf={r.profit_factor:>5.2f}  "
        f"win={r.win_rate*100:>5.1f}%  mdd={r.max_dd*100:>6.1f}%  "
        f"eq={r.equity_pct:>+8.1f}%  (T={r.trend_n} R={r.range_n})"
    )


def main() -> int:
    import logging
    logging.getLogger().setLevel(logging.WARNING)

    print("=" * 100)
    print(" Multi-asset / multi-timeframe robustness test  (synthetic data)")
    print("=" * 100)
    print(" Profiles : btc_like, eth_like, sol_like, range_alt, trend_alt")
    print(" TFs      : 5m, 15m, 1h, 4h")
    print(" Seeds    : 3 per cell    -> 5 x 4 x 3 = 60 backtests")
    print()

    rows: list[CellResult] = []
    seeds = [11, 23, 47]
    for profile in PROFILES:
        print(f"--- {profile} -------------------------------------------------")
        for tf in TIMEFRAMES:
            for s in seeds:
                r = run_cell(profile, tf, s)
                rows.append(r)
                print(_fmt_row(r))
        print()

    df = pd.DataFrame([r.__dict__ for r in rows])

    # ----- per-asset summary
    print("=" * 100)
    print(" Per-profile summary  (mean across timeframes & seeds)")
    print("=" * 100)
    g = df.groupby("profile").agg(
        n_trades=("n_trades", "mean"),
        sharpe=("sharpe", "mean"),
        pf=("profit_factor", "mean"),
        win=("win_rate", "mean"),
        mdd=("max_dd", "mean"),
        eq_pct=("equity_pct", "mean"),
    ).round(3)
    print(g.to_string())
    print()

    # ----- per-timeframe summary
    print("=" * 100)
    print(" Per-timeframe summary  (mean across profiles & seeds)")
    print("=" * 100)
    g2 = df.groupby("tf_min").agg(
        n_trades=("n_trades", "mean"),
        sharpe=("sharpe", "mean"),
        pf=("profit_factor", "mean"),
        win=("win_rate", "mean"),
        mdd=("max_dd", "mean"),
        eq_pct=("equity_pct", "mean"),
    ).round(3)
    print(g2.to_string())
    print()

    # ----- per-cell viability (avg across 3 seeds)
    print("=" * 100)
    print(" Per-cell viability  (mean across seeds; * = LIVE-READY, x = AVOID)")
    print("=" * 100)
    g3 = df.groupby(["profile", "tf_min"]).agg(
        sharpe=("sharpe", "mean"),
        pf=("profit_factor", "mean"),
        eq_pct=("equity_pct", "mean"),
        mdd=("max_dd", "mean"),
    ).round(2)
    for (prof, tf), row in g3.iterrows():
        ok = row["sharpe"] >= 0.8 and row["pf"] >= 1.2 and row["mdd"] >= -0.20
        bad = row["eq_pct"] < -10
        flag = "*" if ok else ("x" if bad else " ")
        print(
            f"  {flag} {prof:<10} {tf:>4}m  sharpe={row['sharpe']:>6.2f}  "
            f"pf={row['pf']:>5.2f}  eq={row['eq_pct']:>+7.1f}%  mdd={row['mdd']*100:>6.1f}%"
        )
    print()

    # ----- aggregate health
    print("=" * 100)
    print(" Aggregate health")
    print("=" * 100)
    n_cells = len(df)
    n_trades_total = int(df["n_trades"].sum())
    profitable = int((df["equity_pct"] > 0).sum())
    pf_ok = int((df["profit_factor"] > 1.2).sum())
    sharpe_ok = int((df["sharpe"] > 0.8).sum())
    mdd_ok = int((df["max_dd"] > -0.30).sum())
    print(f"  cells                : {n_cells}")
    print(f"  total trades         : {n_trades_total}")
    print(f"  profitable cells     : {profitable}/{n_cells}  ({profitable/n_cells*100:.0f}%)")
    print(f"  PF > 1.2             : {pf_ok}/{n_cells}  ({pf_ok/n_cells*100:.0f}%)")
    print(f"  Sharpe > 0.8         : {sharpe_ok}/{n_cells}  ({sharpe_ok/n_cells*100:.0f}%)")
    print(f"  MDD > -30%           : {mdd_ok}/{n_cells}  ({mdd_ok/n_cells*100:.0f}%)")
    print(f"  mean Sharpe          : {df['sharpe'].mean():.2f}")
    print(f"  median Sharpe        : {df['sharpe'].median():.2f}")
    print(f"  mean PF              : {df['profit_factor'].replace(np.inf, np.nan).mean():.2f}")
    print(f"  mean equity change   : {df['equity_pct'].mean():+.1f}%")
    print()
    print(" Note: synthetic data only. For real Bybit validation run e.g.")
    print("       python main.py backtest BTCUSDT --interval 5  --days 365")
    print("       python main.py backtest ETHUSDT --interval 15 --days 365")
    return 0 if profitable >= int(0.6 * n_cells) else 1


if __name__ == "__main__":
    sys.exit(main())
