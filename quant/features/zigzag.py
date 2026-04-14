"""ATR-adaptive ZigZag pivot detector.

Returns a DataFrame of pivots with columns [price, kind] where kind in
{+1 (high), -1 (low)}. The threshold is expressed in ATR units so the
detector adapts to volatility instead of using a fixed percentage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.features.indicators import atr


def zigzag_pivots(df: pd.DataFrame, atr_mult: float = 3.0, atr_n: int = 14) -> pd.DataFrame:
    a = atr(df, atr_n).bfill().values
    h = df["high"].values
    l = df["low"].values
    idx = df.index

    pivots_i: list[int] = []
    pivots_p: list[float] = []
    pivots_k: list[int] = []

    direction = 0  # +1 up-leg, -1 down-leg
    last_pivot_i = 0
    last_pivot_p = df["close"].iloc[0]

    for i in range(1, len(df)):
        thr = atr_mult * a[i]
        if direction >= 0:
            # tracking an up-leg → look for reversal down
            if h[i] > last_pivot_p:
                last_pivot_p = h[i]
                last_pivot_i = i
            elif last_pivot_p - l[i] > thr:
                pivots_i.append(last_pivot_i)
                pivots_p.append(last_pivot_p)
                pivots_k.append(+1)
                direction = -1
                last_pivot_p = l[i]
                last_pivot_i = i
        if direction <= 0:
            if l[i] < last_pivot_p:
                last_pivot_p = l[i]
                last_pivot_i = i
            elif h[i] - last_pivot_p > thr:
                pivots_i.append(last_pivot_i)
                pivots_p.append(last_pivot_p)
                pivots_k.append(-1)
                direction = +1
                last_pivot_p = h[i]
                last_pivot_i = i

    if not pivots_i:
        return pd.DataFrame(columns=["price", "kind"])
    out = pd.DataFrame(
        {"price": pivots_p, "kind": pivots_k},
        index=idx[pivots_i],
    )
    out = out[~out.index.duplicated(keep="first")]
    return out
