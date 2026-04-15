"""Assemble the full feature matrix from OHLCV bars."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant.data.frac_diff import frac_diff_ffd
from quant.features.elliott import elliott_features
from quant.features.fibonacci import fibonacci_features
from quant.features.hawkes import hawkes_intensity
from quant.features.indicators import (
    adx_features,
    atr,
    bollinger_features,
    donchian_features,
    ema_stack,
    macd_features,
    regime_score,
    rsi,
    stoch_features,
    zscore,
)
from quant.features.regime import hmm_regimes, rolling_hurst


@dataclass
class FeatureParams:
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_smooth: int = 3
    donch_n: int = 20
    zigzag_atr_mult: float = 3.0
    hawkes_decay: float = 0.1
    frac_d: float = 0.4
    regime_states: int = 3
    # Regime / trend detection
    adx_n: int = 14
    rsi_n: int = 14
    bb_n: int = 20
    bb_k: float = 2.0
    adx_trend_th: float = 22.0
    adx_range_th: float = 18.0


def build_feature_matrix(df: pd.DataFrame, params: FeatureParams | None = None) -> pd.DataFrame:
    """Compute the full feature matrix.

    Input must be an OHLCV DataFrame indexed by UTC timestamp.
    Output is aligned to the input index; rows with NaNs at the head should be
    dropped by downstream consumers.
    """
    p = params or FeatureParams()
    df = df.copy()
    df["turnover"] = df.get("turnover", df["close"] * df["volume"])

    feats: list[pd.DataFrame | pd.Series] = []

    # Base price / return features
    feats.append(df["close"].rename("close"))
    log_ret = np.log(df["close"]).diff().rename("log_ret")
    feats.append(log_ret)
    feats.append(log_ret.rolling(20).std().rename("vol_20"))
    feats.append(log_ret.rolling(64).std().rename("vol_64"))
    feats.append(frac_diff_ffd(df["close"], d=p.frac_d).rename(f"ffd_{p.frac_d}"))
    feats.append(zscore(df["close"], 100).rename("price_z"))

    # ATR
    _atr = atr(df, 14).rename("atr_14")
    feats.append(_atr)
    feats.append((_atr / df["close"]).rename("atr_pct"))

    # MACD / Stoch / Donchian
    feats.append(macd_features(df["close"], p.macd_fast, p.macd_slow, p.macd_signal))
    feats.append(stoch_features(df, p.stoch_k, p.stoch_d, p.stoch_smooth))
    donch = donchian_features(df, p.donch_n)
    feats.append(donch)

    # Hawkes on breakouts
    break_events = (donch["donch_break_up"] + donch["donch_break_dn"]).clip(0, 1)
    feats.append(hawkes_intensity(break_events, decay=p.hawkes_decay))

    # Trend stack / ADX / RSI / Bollinger  → foundation for regime-aware model
    stack = ema_stack(df["close"])
    feats.append(stack)
    adx_df = adx_features(df, n=p.adx_n)
    feats.append(adx_df)
    feats.append(rsi(df["close"], n=p.rsi_n))
    bb = bollinger_features(df["close"], n=p.bb_n, k=p.bb_k)
    feats.append(bb)

    rs = regime_score(
        adx=adx_df["adx"],
        bb_width=bb["bb_width"],
        trend_align=stack["trend_align"],
        atr_pct=(_atr / df["close"]),
        adx_trend_th=p.adx_trend_th,
        adx_range_th=p.adx_range_th,
    )
    feats.append(rs)

    # Fibonacci + Elliott
    feats.append(fibonacci_features(df, atr_mult=p.zigzag_atr_mult))
    feats.append(elliott_features(df, atr_mult=p.zigzag_atr_mult))

    # Regime (HMM kept as a side feature — no longer the filter driver)
    feats.append(rolling_hurst(df["close"], window=256, max_lag=64))
    feats.append(hmm_regimes(log_ret.fillna(0), n_states=p.regime_states))

    out = pd.concat(feats, axis=1)
    # Align and forward-fill sparse/categorical features
    categorical = ["regime", "ew_wave_id"]
    for c in categorical:
        if c in out.columns:
            out[c] = out[c].ffill()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
