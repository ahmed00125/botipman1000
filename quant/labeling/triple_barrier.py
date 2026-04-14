"""Triple Barrier labeling (AFML ch.3).

For each event (candidate entry) we set:
  - upper barrier = entry + pt_mult * target (profit take)
  - lower barrier = entry - sl_mult * target (stop loss)
  - vertical barrier = entry + max_hold
The label is the sign of the first barrier hit (or 0 if vertical hit first).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def get_daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
    """Exponentially-weighted daily return std used as target volatility."""
    idx = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    idx = idx[idx > 0]
    prev = pd.Series(close.index[idx - 1], index=close.index[close.shape[0] - len(idx) :])
    ret = (close.loc[prev.index] / close.loc[prev.values].values - 1).rename("dret")
    return ret.ewm(span=span).std().rename("target_vol")


def get_vertical_barriers(
    events: pd.DatetimeIndex, close: pd.Series, num_bars: int = 48
) -> pd.Series:
    """Vertical barrier N bars after each event."""
    idx = close.index
    t1 = pd.Series(pd.NaT, index=events)
    for t in events:
        loc = idx.searchsorted(t)
        tgt = loc + num_bars
        if tgt < len(idx):
            t1.loc[t] = idx[tgt]
    return t1


def apply_triple_barrier(
    close: pd.Series,
    events: pd.DatetimeIndex,
    target_vol: pd.Series,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_hold_bars: int = 48,
    side: pd.Series | None = None,
    min_target: float = 0.0005,
) -> pd.DataFrame:
    """Return a DataFrame indexed by event timestamp with columns:
        [t1, ret, bin, pt, sl, side]
    """
    t1 = get_vertical_barriers(events, close, num_bars=max_hold_bars)
    if side is None:
        side = pd.Series(1.0, index=events, name="side")
    else:
        side = side.reindex(events).fillna(1.0)

    out = pd.DataFrame(
        index=events, columns=["t1", "ret", "bin", "pt", "sl", "side"], dtype=float
    )
    out["t1"] = t1
    out["side"] = side

    for t in events:
        if t not in close.index:
            continue
        tv = float(target_vol.reindex([t]).ffill().iloc[0]) if t in target_vol.index or not target_vol.empty else min_target
        tv = max(tv, min_target)
        upper = pt_mult * tv
        lower = -sl_mult * tv
        horizon_end = out.at[t, "t1"] if pd.notna(out.at[t, "t1"]) else close.index[-1]
        path = close.loc[t:horizon_end]
        if len(path) < 2:
            continue
        s = float(out.at[t, "side"])
        rets = (path / path.iloc[0] - 1) * s
        hit_up = rets[rets >= upper]
        hit_dn = rets[rets <= lower]
        first_up = hit_up.index[0] if len(hit_up) else None
        first_dn = hit_dn.index[0] if len(hit_dn) else None

        if first_up is not None and (first_dn is None or first_up <= first_dn):
            out.at[t, "t1"] = first_up
            out.at[t, "ret"] = float(rets.loc[first_up])
            out.at[t, "bin"] = 1.0
        elif first_dn is not None:
            out.at[t, "t1"] = first_dn
            out.at[t, "ret"] = float(rets.loc[first_dn])
            out.at[t, "bin"] = -1.0
        else:
            last = rets.index[-1]
            out.at[t, "t1"] = last
            out.at[t, "ret"] = float(rets.iloc[-1])
            out.at[t, "bin"] = 0.0
        out.at[t, "pt"] = upper
        out.at[t, "sl"] = lower

    return out.dropna(subset=["bin"])
