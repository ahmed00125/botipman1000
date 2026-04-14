"""Feature/return distribution drift detector — halts trading if live stats
deviate materially from training distribution (KS + ADWIN-style).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DriftConfig:
    window: int = 500
    ks_p_threshold: float = 0.01
    mean_z_threshold: float = 3.0


class DriftMonitor:
    def __init__(self, reference: pd.Series, config: DriftConfig | None = None):
        self.cfg = config or DriftConfig()
        self.ref = reference.dropna().values
        self.ref_mean = float(np.mean(self.ref))
        self.ref_std = float(np.std(self.ref) + 1e-9)
        self.live: deque[float] = deque(maxlen=self.cfg.window)

    def update(self, x: float) -> dict:
        self.live.append(float(x))
        if len(self.live) < 50:
            return {"drift": False, "reason": "warming up"}
        live_arr = np.fromiter(self.live, dtype=float)
        ks_stat, ks_p = stats.ks_2samp(self.ref, live_arr)
        z = (live_arr.mean() - self.ref_mean) / self.ref_std * np.sqrt(len(live_arr))
        drift = (ks_p < self.cfg.ks_p_threshold) or (abs(z) > self.cfg.mean_z_threshold)
        return {
            "drift": bool(drift),
            "ks_p": float(ks_p),
            "z": float(z),
            "live_mean": float(live_arr.mean()),
            "ref_mean": self.ref_mean,
        }
