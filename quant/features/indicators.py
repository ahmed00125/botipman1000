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


def _wilder(x: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing — equivalent to EMA with alpha = 1/n."""
    return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


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


# --------------------------------------------------------------- EMA trend stack
def ema_stack(close: pd.Series) -> pd.DataFrame:
    """Multi-horizon EMAs + slope features used for regime classification."""
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    slope20 = ema20.diff(5) / ema20.shift(5).replace(0, np.nan)
    slope50 = ema50.diff(10) / ema50.shift(10).replace(0, np.nan)
    slope200 = ema200.diff(20) / ema200.shift(20).replace(0, np.nan)
    trend_align = (
        np.sign(ema20 - ema50).fillna(0)
        + np.sign(ema50 - ema200).fillna(0)
        + np.sign(close - ema200).fillna(0)
    ) / 3.0
    return pd.DataFrame(
        {
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
            "ema20_slope": slope20,
            "ema50_slope": slope50,
            "ema200_slope": slope200,
            "trend_align": trend_align,
        }
    )


# ----------------------------------------------------------------------- ADX
def adx_features(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Average Directional Index (Wilder).

    Returns ``adx`` plus ``plus_di``/``minus_di`` and an ``adx_norm`` in [0,1].
    """
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = ((up > dn) & (up > 0)) * up
    minus_dm = ((dn > up) & (dn > 0)) * dn
    prev_c = c.shift(1)
    tr = pd.concat(
        [(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1
    ).max(axis=1)
    atr_n = _wilder(tr, n)
    plus_di = 100 * _wilder(plus_dm, n) / atr_n.replace(0, np.nan)
    minus_di = 100 * _wilder(minus_dm, n) / atr_n.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _wilder(dx.fillna(0), n)
    return pd.DataFrame(
        {
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "adx_norm": (adx / 100.0).clip(0, 1),
            "di_spread": (plus_di - minus_di),
        }
    )


# ----------------------------------------------------------------------- RSI
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _wilder(gain, n)
    avg_loss = _wilder(loss, n)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f"rsi_{n}")


# ------------------------------------------------------------------ Bollinger
def bollinger_features(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(n, min_periods=n // 2).mean()
    sd = close.rolling(n, min_periods=n // 2).std(ddof=0)
    upper = mid + k * sd
    lower = mid - k * sd
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    z = (close - mid) / sd.replace(0, np.nan)
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_width": width,
            "bb_pct_b": pct_b,
            "bb_z": z,
        }
    )


# ---------------------------------------------------------- Regime score combo
def regime_score(
    adx: pd.Series,
    bb_width: pd.Series,
    trend_align: pd.Series,
    atr_pct: pd.Series,
    adx_trend_th: float = 22.0,
    adx_range_th: float = 18.0,
    width_pctl_lookback: int = 250,
) -> pd.DataFrame:
    """Classify each bar into TREND_UP / TREND_DOWN / RANGE / CHOP.

    Codes:
        +2 strong trend up
        +1 weak trend up
         0 range / neutral
        -1 weak trend down
        -2 strong trend down
        NaN where not enough history
    ``chop`` is a boolean feature: ADX < range_th AND bb_width in lowest quartile.
    """
    width_pctl = bb_width.rolling(width_pctl_lookback, min_periods=50).rank(pct=True)
    atr_pctl = atr_pct.rolling(width_pctl_lookback, min_periods=50).rank(pct=True)

    # A real trend needs *both* a strong ADX and a perfect EMA stack that has
    # been aligned for at least ``align_min_bars`` bars in a row. This filter
    # rejects the rising leg of a long oscillation that happens to push ADX up.
    align_sign = np.sign(trend_align).fillna(0)
    same_run = (align_sign != align_sign.shift(1)).cumsum()
    persist = align_sign.groupby(same_run).cumcount() + 1
    align_persistent = persist >= 30  # ~2.5 hours on 5m bars

    trending_strong = (
        (adx >= adx_trend_th + 10)
        & (trend_align.abs() >= 1.0)
        & align_persistent
    )
    trending_weak = (
        (adx >= adx_trend_th + 3)
        & (trend_align.abs() >= 2 / 3)
        & align_persistent
    )
    # Range: low ADX OR trend_align uncertain — and BB width not in top decile
    ranging = (
        ((adx <= adx_range_th + 4) | (trend_align.abs() <= 1 / 3))
        & (width_pctl <= 0.75)
    )
    # Chop = either tight low-vol noise OR high-vol noise where bands are wide
    # AND adx is weak. High-vol chop is what historically eats the range module.
    chop_lo = (adx <= adx_range_th - 2) & (width_pctl <= 0.20) & (atr_pctl <= 0.30)
    chop_hi = (adx <= adx_range_th + 2) & (atr_pctl >= 0.85) & (~align_persistent)
    chop = chop_lo | chop_hi

    dir_sign = align_sign
    regime = pd.Series(0.0, index=adx.index)
    regime = regime.where(~trending_weak, dir_sign * 1.0)
    regime = regime.where(~trending_strong, dir_sign * 2.0)
    regime = regime.where(~ranging, 0.0)

    is_trend_up = (regime > 0).astype(float)
    is_trend_dn = (regime < 0).astype(float)
    # Range = ranging context AND not trending AND not chop
    is_range = (
        ranging & (~trending_weak) & (~trending_strong) & (~chop.fillna(False))
    ).astype(float)
    is_chop = chop.fillna(False).astype(float)

    return pd.DataFrame(
        {
            "regime_code": regime,
            "is_trend_up": is_trend_up,
            "is_trend_dn": is_trend_dn,
            "is_range": is_range,
            "is_chop": is_chop,
            "bb_width_pctl": width_pctl,
        }
    )


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
