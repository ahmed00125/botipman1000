"""Fractional differentiation with Fixed-Width Window (AFML ch.5).

Produces a stationary series that preserves memory — unlike log returns which
erase long-range dependence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _get_weights_ffd(d: float, thres: float = 1e-5, max_size: int = 10_000) -> np.ndarray:
    w = [1.0]
    for k in range(1, max_size):
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thres:
            break
        w.append(w_)
    return np.array(w[::-1]).reshape(-1, 1)


def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
    """Apply FFD to a series. Returns a Series aligned to the input index."""
    w = _get_weights_ffd(d, thres)
    width = len(w) - 1
    out = np.full(len(series), np.nan)
    values = series.values.astype(float)
    for i in range(width, len(series)):
        window = values[i - width : i + 1]
        if np.isnan(window).any():
            continue
        out[i] = float(np.dot(w.T, window)[0, 0])
    return pd.Series(out, index=series.index, name=f"ffd_{d:.2f}")


def find_min_d(
    series: pd.Series, d_values: np.ndarray | None = None, adf_level: float = 0.05
) -> float:
    """Find the smallest d in [0,1] that makes the series stationary (ADF p<level).

    Imports adfuller lazily to keep statsmodels optional.
    """
    from statsmodels.tsa.stattools import adfuller  # type: ignore

    if d_values is None:
        d_values = np.linspace(0.0, 1.0, 11)
    for d in d_values:
        fd = frac_diff_ffd(series, d).dropna()
        if len(fd) < 100:
            continue
        stat, pval, *_ = adfuller(fd, maxlag=1, regression="c", autolag=None)
        if pval < adf_level:
            return float(d)
    return 1.0
