"""Position sizing helpers: fractional Kelly and volatility targeting."""
from __future__ import annotations

import numpy as np


def fractional_kelly(
    win_prob: float, win_loss_ratio: float, fraction: float = 0.25
) -> float:
    """Classic Kelly fraction scaled by ``fraction`` (e.g. 0.25).

    f* = p - (1-p)/b   where b = win/loss ratio.
    Clipped to [0, 1].
    """
    p = float(np.clip(win_prob, 0.0, 1.0))
    b = max(float(win_loss_ratio), 1e-6)
    f_full = p - (1.0 - p) / b
    f_full = max(f_full, 0.0)
    return float(np.clip(fraction * f_full, 0.0, 1.0))


def vol_target_size(
    equity: float,
    price: float,
    realized_vol_ann: float,
    target_vol_ann: float = 0.15,
    max_leverage: float = 3.0,
) -> float:
    """Size in base-units to target ``target_vol_ann`` portfolio volatility.

    realized_vol_ann : the annualized vol of the instrument's returns.
    """
    if realized_vol_ann <= 0 or price <= 0:
        return 0.0
    leverage = target_vol_ann / realized_vol_ann
    leverage = float(np.clip(leverage, 0.0, max_leverage))
    notional = equity * leverage
    return notional / price


def combined_size(
    equity: float,
    price: float,
    realized_vol_ann: float,
    win_prob: float,
    win_loss_ratio: float,
    target_vol_ann: float = 0.15,
    max_leverage: float = 3.0,
    kelly_fraction: float = 0.25,
    min_prob: float = 0.52,
) -> float:
    """Combine vol-target sizing with fractional Kelly.

    If win_prob < min_prob → 0. Else: qty = vol_target * kelly_scale.
    """
    if win_prob < min_prob:
        return 0.0
    base_qty = vol_target_size(equity, price, realized_vol_ann, target_vol_ann, max_leverage)
    k = fractional_kelly(win_prob, win_loss_ratio, kelly_fraction)
    # Normalize: k ranges roughly [0, 0.25*1] → scale to [0,1]
    k_scale = min(k / max(kelly_fraction, 1e-6), 1.0)
    return base_qty * k_scale
