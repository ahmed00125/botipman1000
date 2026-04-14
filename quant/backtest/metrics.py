"""Risk-adjusted metrics — Probabilistic & Deflated Sharpe, PBO, MC bootstrap.

References:
- Bailey, López de Prado (2012, 2014): "The Sharpe ratio efficient frontier"
- Bailey et al. (2015): "The Probability of Backtest Overfitting"
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy import stats


def sharpe_ratio(returns: pd.Series, ann_factor: float) -> float:
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(ann_factor))


def probabilistic_sharpe_ratio(
    sr: float, n: int, skew: float, kurt: float, sr_benchmark: float = 0.0
) -> float:
    """PSR: probability that the observed SR > benchmark given sampling noise.

    sr / sr_benchmark are per-period (not annualized).
    """
    if n <= 1:
        return 0.0
    num = (sr - sr_benchmark) * np.sqrt(n - 1)
    den = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-9))
    return float(stats.norm.cdf(num / den))


def deflated_sharpe_ratio(
    sr: float,
    n: int,
    skew: float,
    kurt: float,
    n_trials: int,
    var_of_trials: float,
) -> float:
    """Deflated Sharpe Ratio — PSR with a benchmark adjusted for multiple testing.

    Uses the approximation in Bailey & López de Prado (2014).
    """
    if n_trials < 1:
        return probabilistic_sharpe_ratio(sr, n, skew, kurt, sr_benchmark=0.0)
    emc = 0.5772156649  # Euler-Mascheroni
    max_z = (1 - emc) * stats.norm.ppf(1 - 1.0 / n_trials) + emc * stats.norm.ppf(
        1 - 1.0 / (n_trials * np.e)
    )
    sr_benchmark = np.sqrt(max(var_of_trials, 1e-12)) * max_z
    return probabilistic_sharpe_ratio(sr, n, skew, kurt, sr_benchmark=sr_benchmark)


def monte_carlo_bootstrap(
    trade_returns: np.ndarray, n_sims: int = 2000, horizon: int | None = None
) -> dict:
    """Bootstrap trade sequences to get a distribution of cumulative return and DD."""
    if len(trade_returns) == 0:
        return {"mean_ret": 0.0, "var_ret": 0.0, "p5_dd": 0.0, "p95_ret": 0.0}
    horizon = horizon or len(trade_returns)
    rng = np.random.default_rng(42)
    final_rets = np.empty(n_sims)
    max_dds = np.empty(n_sims)
    for s in range(n_sims):
        sample = rng.choice(trade_returns, size=horizon, replace=True)
        eq = np.cumprod(1 + sample)
        peak = np.maximum.accumulate(eq)
        dd = (eq / peak - 1).min()
        final_rets[s] = eq[-1] - 1
        max_dds[s] = dd
    return dict(
        mean_ret=float(final_rets.mean()),
        median_ret=float(np.median(final_rets)),
        p5_ret=float(np.quantile(final_rets, 0.05)),
        p95_ret=float(np.quantile(final_rets, 0.95)),
        p5_dd=float(np.quantile(max_dds, 0.05)),
        mean_dd=float(max_dds.mean()),
    )


def pbo(matrix: pd.DataFrame, score_fn: Callable[[pd.Series], float] | None = None) -> float:
    """Probability of Backtest Overfitting via Combinatorially Symmetric CV.

    ``matrix`` has shape (T, N) where each column is a strategy's return stream
    over the same time index. We split time into 2S equal parts, pick all C(2S,S)
    combinations, score train/test, and count how often the train-best strategy
    underperforms the median on test.
    """
    if score_fn is None:
        score_fn = lambda r: r.mean() / (r.std() + 1e-9)
    from itertools import combinations

    T, N = matrix.shape
    if N < 2 or T < 16:
        return 0.5
    S = 4  # 2S=8 blocks
    block_size = T // (2 * S)
    blocks = [
        matrix.iloc[i * block_size : (i + 1) * block_size] for i in range(2 * S)
    ]
    combos = list(combinations(range(2 * S), S))
    losses = 0
    total = 0
    for train_idx in combos:
        test_idx = tuple(i for i in range(2 * S) if i not in train_idx)
        train = pd.concat([blocks[i] for i in train_idx])
        test = pd.concat([blocks[i] for i in test_idx])
        train_scores = train.apply(score_fn)
        test_scores = test.apply(score_fn)
        best_col = train_scores.idxmax()
        rank_on_test = (test_scores <= test_scores[best_col]).sum()
        w = rank_on_test / N
        if w <= 0.5:
            losses += 1
        total += 1
    return losses / total if total else 0.5
