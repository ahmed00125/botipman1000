"""Classic indicators → feature engineering.

We expose **features** (slopes, z-scores, distances, durations), not raw
indicator values, because raw values are non-stationary and regime-dependent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------- helpers
def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def zscore(x: pd.Series, window: int = 100) -> pd.Series:
    mu = x.rolling(window, min_periods=window // 2).mean()
    sd = x.rolling(window, min_periods=window // 2).std(ddof=0)
    return (x - mu) / sd.replace(0, np.nan)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat(
        [(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


# ------------------------------------------------------------------------- MACD
def macd_features(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    ema_f = _ema(close, fast)
    ema_s = _ema(close, slow)
    macd = ema_f - ema_s
    sig = _ema(macd, signal)
    hist = macd - sig

    hist_slope = hist.diff()
    hist_z = zscore(hist, 100)
    # bullish/bearish divergence proxy: price vs macd z-score disagreement
    price_z = zscore(close, 100)
    divergence = (price_z - hist_z).fillna(0.0)
    cross_up = ((macd > sig) & (macd.shift(1) <= sig.shift(1))).astype(float)
    cross_dn = ((macd < sig) & (macd.shift(1) >= sig.shift(1))).astype(float)

    return pd.DataFrame(
        {
            "macd_hist_z": hist_z,
            "macd_hist_slope": hist_slope,
            "macd_divergence": divergence,
            "macd_cross_up": cross_up,
            "macd_cross_dn": cross_dn,
        }
    )


# -------------------------------------------------------------------- Stochastic
def stoch_features(
    df: pd.DataFrame, k: int = 14, d: int = 3, smooth: int = 3
) -> pd.DataFrame:
    low_k = df["low"].rolling(k).min()
    high_k = df["high"].rolling(k).max()
    rng = (high_k - low_k).replace(0, np.nan)
    k_fast = 100 * (df["close"] - low_k) / rng
    k_slow = k_fast.rolling(smooth).mean()
    d_slow = k_slow.rolling(d).mean()
    spread = k_slow - d_slow

    oversold = (k_slow < 20).astype(float)
    overbought = (k_slow > 80).astype(float)
    os_duration = oversold.groupby((oversold != oversold.shift()).cumsum()).cumsum()
    ob_duration = overbought.groupby((overbought != overbought.shift()).cumsum()).cumsum()

    return pd.DataFrame(
        {
            "stoch_k": k_slow / 100.0,
            "stoch_spread": spread,
            "stoch_oversold_dur": os_duration,
            "stoch_overbought_dur": ob_duration,
        }
    )


# ---------------------------------------------------------------------- Donchian
def donchian_features(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    upper = df["high"].rolling(n).max()
    lower = df["low"].rolling(n).min()
    mid = (upper + lower) / 2.0
    width = (upper - lower)
    _atr = atr(df, 14)

    breakout_up = ((df["close"] > upper.shift(1)) & (df["close"].shift(1) <= upper.shift(1))).astype(float)
    breakout_dn = ((df["close"] < lower.shift(1)) & (df["close"].shift(1) >= lower.shift(1))).astype(float)
    pos_in_chan = (df["close"] - lower) / width.replace(0, np.nan)
    breakout_strength = (df["close"] - upper.shift(1)) / _atr

    return pd.DataFrame(
        {
            "donch_pos": pos_in_chan,
            "donch_width_atr": width / _atr,
            "donch_break_up": breakout_up,
            "donch_break_dn": breakout_dn,
            "donch_break_strength": breakout_strength,
            "donch_mid_dist_atr": (df["close"] - mid) / _atr,
        }
    )
