"""Regime-aware primary rule model.

The previous model blended trend-following and mean-reverting votes on every
bar, which cancels in trends and whipsaws in ranges. This version picks a
*different* play book for each market regime:

* **TREND**  — enter *with* the trend on pullbacks. Only long in uptrends,
  only short in downtrends. Confirms with MACD histogram and Donchian
  breakout strength. Filters out counter-trend mean-reversion noise.
* **RANGE**  — fade the edges. Long when price tags the lower Bollinger band
  with RSI turning up; short at the upper band with RSI rolling over.
  Requires a compressed ADX to avoid catching falling-knife trend starts.
* **CHOP / UNCLASSIFIED** — no trade. Capital preservation beats forcing.

Per bar output:
    primary_side  : {-1, 0, +1}
    primary_score : [-1, 1]   continuous strength (input feature for meta)
    primary_mode  : 0=none  1=trend  2=range
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PrimaryParams:
    # Trend module
    trend_macd_min: float = 0.0      # min MACD hist z-score with trend
    trend_pullback_k: float = 1.2    # acceptable pullback depth vs BB (in sd)
    trend_donch_conf: float = 0.0    # min Donchian breakout strength to reinforce
    trend_align_min: float = 0.33    # min |trend_align| to trade trend

    # Range module
    range_rsi_lo: float = 38.0
    range_rsi_hi: float = 62.0
    range_pctb_lo: float = 0.20      # fade when below this pct-B
    range_pctb_hi: float = 0.80      # fade when above this pct-B
    range_max_adx: float = 24.0      # never fade in a strong trend
    range_confirm_reversal: bool = True  # require a candle reversal

    # Shared
    min_abs_score: float = 0.35
    cooldown_bars: int = 3           # skip this many bars after each signal
    # Kept for config backward-compat (unused now but optuna may set them)
    w_macd: float = 1.0
    w_stoch: float = 1.0
    w_donch: float = 1.2
    w_fib: float = 0.8
    w_ew: float = 0.8
    regime_trending: int = 2
    regime_neutral: int = 1


class PrimaryRuleModel:
    """Regime-aware directional model."""

    def __init__(self, params: PrimaryParams | None = None):
        self.p = params or PrimaryParams()

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _col(feats: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
        if name in feats.columns:
            return feats[name].astype(float).fillna(default)
        return pd.Series(default, index=feats.index, dtype=float)

    # -------------------------------------------------------------- per-module
    def _trend_signal(self, feats: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Return (side, score) for the trend-following module.

        Long setup: uptrend (EMA stack aligned, ADX raised, +DI dominant),
        price pulled back *to* the EMA20 from above (not overextended), and
        RSI/MACD turning back up. Short is the symmetric.
        """
        p = self.p
        close = self._col(feats, "close")
        ema20 = self._col(feats, "ema20")
        ema50 = self._col(feats, "ema50")
        atr_abs = (self._col(feats, "atr_pct") * close).replace(0, np.nan)
        trend_align = self._col(feats, "trend_align")
        rsi14 = self._col(feats, "rsi_14")
        macd_slope = self._col(feats, "macd_hist_slope")
        macd_z = self._col(feats, "macd_hist_z")
        donch_bs = self._col(feats, "donch_break_strength")
        plus_di = self._col(feats, "plus_di")
        minus_di = self._col(feats, "minus_di")
        adx = self._col(feats, "adx")

        is_up = self._col(feats, "is_trend_up")
        is_dn = self._col(feats, "is_trend_dn")

        # Pullback depth: how far below EMA20 the price is, in ATR units.
        dist_ema20 = (close - ema20) / atr_abs
        dist_ema20 = dist_ema20.fillna(0)

        # --- LONG --------------------------------------------------------
        long_ok = (
            (is_up > 0)
            & (trend_align >= p.trend_align_min)
            & (plus_di > minus_di)
            & (adx >= 20.0)
            & (close >= ema50)               # above 50-EMA (big-picture up)
        )
        # Price has dipped to or slightly below EMA20 and is about to bounce
        long_pullback = (
            (dist_ema20 <= 0.3)
            & (dist_ema20 >= -p.trend_pullback_k)
            & ((macd_slope > 0) | (rsi14 >= 40))
        )

        # --- SHORT -------------------------------------------------------
        short_ok = (
            (is_dn > 0)
            & (trend_align <= -p.trend_align_min)
            & (minus_di > plus_di)
            & (adx >= 20.0)
            & (close <= ema50)
        )
        short_pullback = (
            (dist_ema20 >= -0.3)
            & (dist_ema20 <= p.trend_pullback_k)
            & ((macd_slope < 0) | (rsi14 <= 60))
        )

        conf = np.tanh(donch_bs).clip(-1, 1)

        score = pd.Series(0.0, index=feats.index)
        long_mask = long_ok & long_pullback
        short_mask = short_ok & short_pullback
        long_score = (0.55 + 0.20 * np.tanh(macd_z / 1.5) + 0.15 * conf.clip(0, None)).clip(0, 1)
        short_score = (0.55 + 0.20 * np.tanh(-macd_z / 1.5) + 0.15 * (-conf).clip(0, None)).clip(0, 1)
        score = score.mask(long_mask, long_score)
        score = score.mask(short_mask, -short_score)

        side = pd.Series(0, index=feats.index, dtype=int)
        side[score > p.min_abs_score] = 1
        side[score < -p.min_abs_score] = -1
        return side, score

    def _range_signal(self, feats: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Fade the band extremes inside a low-ADX range.

        Long: price < lower BB (pct_b < 0.10), RSI < 32, stochastic oversold,
        and MACD histogram slope just turned positive (reversal confirmation).
        Also require that the previous bar was at an even deeper pct_b — i.e.
        we buy the bounce, not the first touch.
        """
        p = self.p
        is_range = self._col(feats, "is_range")
        adx = self._col(feats, "adx")
        rsi14 = self._col(feats, "rsi_14")
        pct_b = self._col(feats, "bb_pct_b")
        bb_z = self._col(feats, "bb_z")
        macd_slope = self._col(feats, "macd_hist_slope")
        close = self._col(feats, "close")
        bb_mid = self._col(feats, "bb_mid")

        # Previous bar's stretch — we want a recovery FROM deeper
        pct_b_prev = pct_b.shift(1).fillna(0.5)

        in_range_context = (is_range > 0) & (adx < p.range_max_adx)

        long_setup = (
            in_range_context
            & (pct_b <= p.range_pctb_lo)
            & (pct_b_prev <= p.range_pctb_lo + 0.05)
            & (pct_b >= pct_b_prev - 0.02)           # bouncing up (not still falling)
            & (rsi14 <= p.range_rsi_lo)
            & (bb_z < -1.0)                          # genuinely stretched
        )
        short_setup = (
            in_range_context
            & (pct_b >= p.range_pctb_hi)
            & (pct_b_prev >= p.range_pctb_hi - 0.05)
            & (pct_b <= pct_b_prev + 0.02)
            & (rsi14 >= p.range_rsi_hi)
            & (bb_z > 1.0)
        )
        if p.range_confirm_reversal:
            long_setup &= (macd_slope > 0) | (rsi14.diff().fillna(0) > 0)
            short_setup &= (macd_slope < 0) | (rsi14.diff().fillna(0) < 0)

        long_strength = (p.range_pctb_lo - pct_b).clip(lower=0) * 4.0
        short_strength = (pct_b - p.range_pctb_hi).clip(lower=0) * 4.0

        score = pd.Series(0.0, index=feats.index)
        score = score.mask(long_setup, (0.55 + long_strength).clip(0, 1))
        score = score.mask(short_setup, -(0.55 + short_strength).clip(0, 1))

        side = pd.Series(0, index=feats.index, dtype=int)
        side[score > p.min_abs_score] = 1
        side[score < -p.min_abs_score] = -1
        return side, score

    # ------------------------------------------------------------- orchestrate
    def compute(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Combine trend + range signals, apply chop filter and cool-down."""
        p = self.p
        idx = feats.index

        trend_side, trend_score = self._trend_signal(feats)
        range_side, range_score = self._range_signal(feats)

        # Prefer trend signal when both fire (shouldn't normally, but
        # regime flags are soft boolean), fall through to range.
        side = trend_side.copy()
        score = trend_score.copy()
        mode = pd.Series(0, index=idx, dtype=int)
        mode[trend_side != 0] = 1

        mask_range = (side == 0) & (range_side != 0)
        side[mask_range] = range_side[mask_range]
        score[mask_range] = range_score[mask_range]
        mode[mask_range] = 2

        # Chop filter — never trade when explicit chop flag is on
        is_chop = self._col(feats, "is_chop")
        chop_mask = is_chop > 0
        side = side.where(~chop_mask, 0)
        score = score.where(~chop_mask, 0.0)
        mode = mode.where(~chop_mask, 0)

        # Cool-down: after any non-zero side, blank the next ``cooldown_bars``
        if p.cooldown_bars and p.cooldown_bars > 0:
            cd = np.zeros(len(side), dtype=int)
            remaining = 0
            side_arr = side.values
            for i in range(len(side_arr)):
                if remaining > 0:
                    cd[i] = 1
                    remaining -= 1
                    continue
                if side_arr[i] != 0:
                    remaining = p.cooldown_bars
            cd_mask = pd.Series(cd.astype(bool), index=idx)
            side = side.where(~cd_mask, 0)
            score = score.where(~cd_mask, 0.0)
            mode = mode.where(~cd_mask, 0)

        return pd.DataFrame(
            {
                "primary_side": side.astype(int),
                "primary_score": score.astype(float).clip(-1, 1),
                "primary_mode": mode.astype(int),
            }
        )
