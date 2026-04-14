"""Fibonacci retracement features derived from ZigZag swings."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.features.indicators import atr
from quant.features.zigzag import zigzag_pivots

FIB_LEVELS = np.array([0.236, 0.382, 0.5, 0.618, 0.786])


def fibonacci_features(df: pd.DataFrame, atr_mult: float = 3.0) -> pd.DataFrame:
    """For each bar, compute distance (in ATR units) to the nearest Fib level
    of the latest completed swing, and which level is nearest.
    """
    pivots = zigzag_pivots(df, atr_mult=atr_mult)
    a = atr(df, 14).bfill()

    nearest_dist = pd.Series(np.nan, index=df.index, name="fib_nearest_dist_atr")
    nearest_lvl = pd.Series(np.nan, index=df.index, name="fib_nearest_level")
    in_golden = pd.Series(0.0, index=df.index, name="fib_in_golden_zone")

    if len(pivots) < 2:
        return pd.concat([nearest_dist, nearest_lvl, in_golden], axis=1)

    piv_idx = pivots.index
    piv_price = pivots["price"].values

    # For each bar, find the most recent two pivots forming a swing.
    piv_pos = np.searchsorted(df.index.values, piv_idx.values, side="left")
    # map each bar to the index of the 2nd-latest pivot (swing start)
    for bar_i in range(len(df)):
        # index of latest pivot strictly before bar_i
        j = np.searchsorted(piv_pos, bar_i, side="right") - 1
        if j < 1:
            continue
        p_end = piv_price[j]
        p_start = piv_price[j - 1]
        if p_end == p_start:
            continue
        rng = p_end - p_start
        levels = p_end - FIB_LEVELS * rng  # retrace back from the swing end
        price = df["close"].iloc[bar_i]
        dists = np.abs(levels - price) / max(a.iloc[bar_i], 1e-9)
        k = int(np.argmin(dists))
        nearest_dist.iloc[bar_i] = float(dists[k])
        nearest_lvl.iloc[bar_i] = float(FIB_LEVELS[k])
        # golden zone = 0.5–0.618 retracement
        lo, hi = sorted([p_end - 0.618 * rng, p_end - 0.5 * rng])
        in_golden.iloc[bar_i] = 1.0 if lo <= price <= hi else 0.0

    return pd.concat([nearest_dist, nearest_lvl, in_golden], axis=1)
