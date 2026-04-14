"""Hawkes-process intensity proxy for breakout self-excitation.

A full MLE-fit Hawkes is overkill for features. We approximate the intensity
via an exponentially-decayed count of recent "events" (e.g. Donchian
breakouts). Intensity I_t = alpha * sum_{s<=t} exp(-beta*(t-s)) 1[event_s].
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def hawkes_intensity(
    events: pd.Series, decay: float = 0.1, alpha: float = 1.0
) -> pd.Series:
    """Compute decayed event intensity.

    ``events`` must be a 0/1 indicator aligned to a regular index. ``decay`` is
    the discrete-time decay factor in (0,1).
    """
    if not (0 < decay < 1):
        raise ValueError("decay must be in (0,1)")
    x = events.fillna(0).values.astype(float)
    out = np.zeros_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = acc * (1 - decay) + alpha * x[i]
        out[i] = acc
    return pd.Series(out, index=events.index, name="hawkes_intensity")
