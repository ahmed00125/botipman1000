"""Primary rule-based confluence model.

Combines MACD, Stochastic, Donchian, Fibonacci and Elliott *features* into a
single directional signal. This model is intentionally simple and
interpretable — the meta-labeler filters its false positives.

Per bar output:
    side : {-1, 0, +1}
    score: [-1, 1]   raw confluence score (useful as a feature for the meta model)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PrimaryParams:
    w_macd: float = 1.0
    w_stoch: float = 1.0
    w_donch: float = 1.2
    w_fib: float = 0.8
    w_ew: float = 0.8
    min_abs_score: float = 0.35
    regime_trending: int = 2  # HMM state idx we consider "trending"
    regime_neutral: int = 1


class PrimaryRuleModel:
    def __init__(self, params: PrimaryParams | None = None):
        self.p = params or PrimaryParams()

    def compute(self, feats: pd.DataFrame) -> pd.DataFrame:
        p = self.p
        idx = feats.index
        zeros = pd.Series(0.0, index=idx)

        # --- MACD vote ------------------------------------------------------
        macd_vote = zeros.copy()
        if "macd_hist_z" in feats.columns:
            mh = feats["macd_hist_z"].fillna(0)
            macd_vote = np.tanh(mh / 1.5)
            if "macd_cross_up" in feats.columns:
                macd_vote += 0.5 * feats["macd_cross_up"].fillna(0)
                macd_vote -= 0.5 * feats["macd_cross_dn"].fillna(0)
            macd_vote = macd_vote.clip(-1, 1)

        # --- Stochastic vote -----------------------------------------------
        stoch_vote = zeros.copy()
        if "stoch_k" in feats.columns:
            k = feats["stoch_k"].fillna(0.5)
            spread = feats.get("stoch_spread", zeros).fillna(0)
            # buy when rising out of oversold, sell when falling from overbought
            stoch_vote = np.where(
                (k < 0.25) & (spread > 0), 1.0,
                np.where((k > 0.75) & (spread < 0), -1.0, np.tanh(spread * 20)),
            )
            stoch_vote = pd.Series(stoch_vote, index=idx)

        # --- Donchian breakout ---------------------------------------------
        donch_vote = zeros.copy()
        if "donch_break_strength" in feats.columns:
            bs = feats["donch_break_strength"].fillna(0)
            donch_vote = np.tanh(bs)
            donch_vote += 0.4 * feats.get("donch_break_up", zeros).fillna(0)
            donch_vote -= 0.4 * feats.get("donch_break_dn", zeros).fillna(0)
            donch_vote = donch_vote.clip(-1, 1)

        # --- Fibonacci (buy pullbacks to golden zone when uptrend) ---------
        fib_vote = zeros.copy()
        if "fib_in_golden_zone" in feats.columns:
            gold = feats["fib_in_golden_zone"].fillna(0)
            price_z = feats.get("price_z", zeros).fillna(0)
            fib_vote = gold * np.sign(price_z).replace(0, np.nan).fillna(1.0)

        # --- Elliott (wave 3 and wave 5 add directional bias) --------------
        ew_vote = zeros.copy()
        if "ew_in_wave3" in feats.columns:
            price_z = feats.get("price_z", zeros).fillna(0)
            bias = np.sign(price_z).replace(0, np.nan).fillna(0)
            ew_vote = bias * (
                feats["ew_in_wave3"].fillna(0) + 0.6 * feats["ew_in_wave5"].fillna(0)
            )
            ew_vote = ew_vote.clip(-1, 1)

        score = (
            p.w_macd * macd_vote
            + p.w_stoch * stoch_vote
            + p.w_donch * donch_vote
            + p.w_fib * fib_vote
            + p.w_ew * ew_vote
        )
        norm = p.w_macd + p.w_stoch + p.w_donch + p.w_fib + p.w_ew
        score = (score / max(norm, 1e-9)).clip(-1, 1).astype(float)

        side = pd.Series(0, index=idx, dtype=int)
        side[score >= p.min_abs_score] = 1
        side[score <= -p.min_abs_score] = -1

        # regime filter: reduce signals in chop (regime==neutral)
        if "regime" in feats.columns:
            regime = feats["regime"].astype(float).fillna(0)
            chop_mask = regime == p.regime_neutral
            side = side.where(~chop_mask, 0)

        return pd.DataFrame({"primary_side": side, "primary_score": score})
