"""Information-driven bars and event sampling (López de Prado, AFML ch.2, ch.5).

- ``dollar_bars``: aggregate ticks/klines by fixed USD volume traded.
- ``cusum_events``: symmetric CUSUM filter returning the timestamps where the
  cumulative signed log-return breaks the threshold h (adaptive to rolling vol).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def dollar_bars(df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Resample OHLCV klines into dollar bars.

    Parameters
    ----------
    df : DataFrame with [open, high, low, close, volume, turnover] indexed by ts.
    threshold : USD volume per bar. If None, use median daily turnover / 50.
    """
    if "turnover" not in df.columns:
        df = df.copy()
        df["turnover"] = df["close"] * df["volume"]

    if threshold is None:
        daily = df["turnover"].resample("1D").sum()
        threshold = float(daily.median() / 50.0) if len(daily) else 1e7
        threshold = max(threshold, 1e5)

    ts = df.index.to_numpy()
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    v = df["volume"].to_numpy()
    dv = df["turnover"].to_numpy()

    out_ts, out_o, out_h, out_l, out_c, out_v, out_dv = [], [], [], [], [], [], []
    cum = 0.0
    bar_o = o[0] if len(o) else np.nan
    bar_h = -np.inf
    bar_l = np.inf
    bar_v = 0.0
    bar_dv = 0.0
    bar_start = ts[0] if len(ts) else None

    for i in range(len(df)):
        if cum == 0:
            bar_o = o[i]
            bar_start = ts[i]
            bar_h = -np.inf
            bar_l = np.inf
            bar_v = 0.0
            bar_dv = 0.0
        bar_h = max(bar_h, h[i])
        bar_l = min(bar_l, l[i])
        bar_v += v[i]
        bar_dv += dv[i]
        cum += dv[i]
        if cum >= threshold:
            out_ts.append(ts[i])
            out_o.append(bar_o)
            out_h.append(bar_h)
            out_l.append(bar_l)
            out_c.append(c[i])
            out_v.append(bar_v)
            out_dv.append(bar_dv)
            cum = 0.0

    bars = pd.DataFrame(
        {
            "open": out_o,
            "high": out_h,
            "low": out_l,
            "close": out_c,
            "volume": out_v,
            "turnover": out_dv,
        },
        index=pd.DatetimeIndex(out_ts, name="ts"),
    )
    return bars


def cusum_events(
    close: pd.Series,
    h: float | pd.Series | None = None,
    vol_lookback: int = 100,
    h_mult: float = 2.0,
) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter on log returns.

    If ``h`` is None the threshold becomes ``h_mult * rolling_std(log_returns)``,
    so the filter adapts to current volatility.
    """
    log_ret = np.log(close).diff().dropna()
    if h is None:
        vol = log_ret.rolling(vol_lookback, min_periods=20).std()
        h_series = (vol * h_mult).reindex(log_ret.index).bfill()
    elif isinstance(h, (int, float)):
        h_series = pd.Series(float(h), index=log_ret.index)
    else:
        h_series = h.reindex(log_ret.index).ffill().bfill()

    s_pos = 0.0
    s_neg = 0.0
    events: list[pd.Timestamp] = []
    for t, r in log_ret.items():
        thr = float(h_series.loc[t])
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        if s_pos > thr:
            s_pos = 0.0
            events.append(t)
        elif s_neg < -thr:
            s_neg = 0.0
            events.append(t)
    return pd.DatetimeIndex(events, name="event")
