"""Triple Barrier labeling (AFML ch.3).

For each event (candidate entry) we set:
  - upper barrier = entry + pt_mult * target (profit take)
  - lower barrier = entry - sl_mult * target (stop loss)
  - vertical barrier = entry + max_hold
The label is the sign of the first barrier hit (or 0 if vertical hit first).
"""
from __future__ import annotations

import numpy as np
import pandas as pd  # noqa: F401


def get_daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
    """Exponentially-weighted return std used as target volatility.

    Uses the AFML pattern: map each bar to the bar closest to 1 day earlier,
    compute the return across that window, then EWMA its std.
    """
    idx = close.index
    prior = idx.searchsorted(idx - pd.Timedelta(days=1))
    prior = prior[prior > 0]
    if len(prior) == 0:
        ret = np.log(close).diff()
    else:
        current_idx = idx[len(idx) - len(prior) :]
        ret = pd.Series(
            close.loc[current_idx].values / close.iloc[prior - 1].values - 1.0,
            index=current_idx,
        )
    return ret.ewm(span=span, min_periods=min(span, 20)).std().rename("target_vol")


def get_vertical_barriers(
    events: pd.DatetimeIndex, close: pd.Series, num_bars: int = 48
) -> pd.Series:
    """Vertical barrier N bars after each event."""
    idx = close.index
    out = []
    for t in events:
        loc = idx.searchsorted(t)
        tgt = loc + num_bars
        if tgt < len(idx):
            out.append(idx[tgt])
        else:
            out.append(idx[-1])
    return pd.Series(pd.DatetimeIndex(out, tz=idx.tz), index=events)


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
