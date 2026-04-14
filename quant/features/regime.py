"""Regime detection: Hurst exponent + HMM on returns & realized volatility."""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def hurst_exponent(series: pd.Series, max_lag: int = 64) -> float:
    """Rescaled-range style Hurst estimator. H≈0.5 random, >0.5 trending,
    <0.5 mean-reverting.
    """
    s = series.dropna().values
    if len(s) < max_lag * 2:
        return 0.5
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        diff = s[lag:] - s[:-lag]
        sd = np.std(diff)
        if sd <= 0 or np.isnan(sd):
            continue
        tau.append(sd)
    if len(tau) < 5:
        return 0.5
    log_lags = np.log(list(range(2, 2 + len(tau))))
    log_tau = np.log(tau)
    slope, _ = np.polyfit(log_lags, log_tau, 1)
    return float(slope)


def rolling_hurst(series: pd.Series, window: int = 256, max_lag: int = 64) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, name="hurst")
    arr = series.values
    for i in range(window, len(series)):
        out.iloc[i] = hurst_exponent(pd.Series(arr[i - window : i]), max_lag=max_lag)
    return out


def hmm_regimes(returns: pd.Series, n_states: int = 3) -> pd.Series:
    """Fit a Gaussian HMM on (returns, |returns|) and return the Viterbi path.

    Falls back to a vol-quantile classifier if hmmlearn is unavailable.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception as exc:
        logger.warning(f"hmmlearn unavailable, falling back to vol-quantile: {exc}")
        vol = returns.rolling(64).std()
        q = vol.rank(pct=True)
        bins = pd.cut(q, bins=[-0.01, 0.33, 0.66, 1.01], labels=[0, 1, 2]).astype(
            "float"
        )
        return bins.rename("regime")

    r = returns.fillna(0).values.reshape(-1, 1)
    feats = np.hstack([r, np.abs(r)])
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=50,
        random_state=42,
    )
    try:
        model.fit(feats)
        states = model.predict(feats)
    except Exception as exc:
        logger.warning(f"HMM fit failed: {exc}")
        return pd.Series(0, index=returns.index, name="regime")
    return pd.Series(states, index=returns.index, name="regime")
