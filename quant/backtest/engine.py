"""Event-driven backtest engine.

Improvements over the previous version:
    * **ATR-based barriers** — targets are in ATR units converted to fractional
      returns, not daily-vol units. On 5m bars this makes pt/sl meaningful.
    * **Regime-aware barriers** — tighter stops + smaller targets for range
      signals, wider stops + 2R targets for trend signals.
    * **Trade concurrency** — only one open position at a time; events during
      an open trade are skipped, eliminating double-counting.
    * **Realistic costs** — taker fees + slippage per side + funding drag.
    * **Single-source sizing** — R per trade is exactly ``risk_per_trade``
      (capped by max_leverage).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    fee_taker: float = 0.00055
    fee_maker: float = 0.0002
    slippage_bps: float = 3.0        # 0.03% per side
    funding_bps_per_8h: float = 1.0
    starting_equity: float = 10_000.0
    max_leverage: float = 3.0
    risk_per_trade: float = 0.01
    use_maker: bool = False

    # --- Regime-aware ATR barriers (multiples of ATR) ---
    trend_pt_atr: float = 3.0
    trend_sl_atr: float = 1.5
    trend_max_hold_bars: int = 72
    trend_trail_atr: float = 2.0      # trailing stop distance once in profit

    range_pt_atr: float = 1.5
    range_sl_atr: float = 1.0
    range_max_hold_bars: int = 24

    # Legacy fallback
    pt_mult: float = 2.0
    sl_mult: float = 1.0
    max_hold_bars: int = 48
    meta_threshold: float = 0.55
    min_target_frac: float = 0.003    # floor TP to 0.3% so costs are covered
    vol_target_ann: float = 0.15
    bars_per_year: int = 365 * 24 * 12   # ~5m bars


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    returns: pd.Series
    stats: dict = field(default_factory=dict)


class EventBacktester:
    def __init__(self, config: BacktestConfig):
        self.cfg = config

    # --------------------------------------------------------------- helpers
    def _resolve_barriers(
        self,
        close_arr: np.ndarray,
        entry_i: int,
        s: int,
        atr_frac: float,
        pt_mult: float,
        sl_mult: float,
        max_hold: int,
        trail_atr: float | None = None,
    ) -> tuple[int, float, int]:
        """Walk the price path after entry until a barrier is hit.

        Returns (exit_i, realized_return_signed, barrier_label).
        """
        atr_frac = max(atr_frac, 1e-6)
        upper = pt_mult * atr_frac
        lower = -sl_mult * atr_frac
        entry_p = close_arr[entry_i]
        peak_ret = 0.0
        last_i = min(entry_i + max_hold, len(close_arr) - 1)
        for j in range(entry_i + 1, last_i + 1):
            cur_p = close_arr[j]
            ret = s * (cur_p / entry_p - 1)
            peak_ret = max(peak_ret, ret)
            # Fixed take profit
            if ret >= upper:
                return j, upper, 1
            # Stop loss
            if ret <= lower:
                return j, lower, -1
            # Trailing stop once in profit (trend mode only)
            if trail_atr is not None and peak_ret > atr_frac * 0.8:
                trail_level = peak_ret - trail_atr * atr_frac
                if ret <= trail_level and trail_level > 0:
                    return j, ret, 1  # locked profit
        # Vertical barrier
        final_ret = s * (close_arr[last_i] / entry_p - 1)
        return last_i, float(final_ret), 0

    # -------------------------------------------------------------------- run
    def run(
        self,
        close: pd.Series,
        events: pd.DatetimeIndex,
        side: pd.Series,
        meta_proba: pd.Series | None = None,
        atr_pct: pd.Series | None = None,
        mode: pd.Series | None = None,
    ) -> BacktestResult:
        cfg = self.cfg
        close = close.astype(float)
        idx = close.index

        # Default ATR if not supplied: 14-bar true-range as fraction
        if atr_pct is None:
            rets = np.log(close).diff().abs().fillna(0)
            atr_pct = rets.ewm(span=14, adjust=False).mean().bfill()
        atr_pct = atr_pct.reindex(idx).ffill().bfill().clip(lower=1e-4)

        side = side.reindex(idx).fillna(0).astype(int)
        if mode is None:
            mode = pd.Series(1, index=idx, dtype=int)
        mode = mode.reindex(idx).fillna(0).astype(int)

        events = events.intersection(idx)
        events = events[side.loc[events] != 0]
        if meta_proba is not None:
            mp = meta_proba.reindex(events).fillna(0)
            events = events[mp >= cfg.meta_threshold]
        if len(events) == 0:
            eq = pd.Series([cfg.starting_equity], index=idx[:1])
            return BacktestResult(
                trades=pd.DataFrame(),
                equity=eq,
                returns=pd.Series(dtype=float),
                stats={"n_trades": 0},
            )

        fee = cfg.fee_maker if cfg.use_maker else cfg.fee_taker
        slip = cfg.slippage_bps / 1e4

        close_arr = close.values
        atr_arr = atr_pct.values

        records: list[dict] = []
        equity = cfg.starting_equity
        equity_curve = pd.Series(np.nan, index=idx, dtype=float)
        equity_curve.iloc[0] = equity

        next_free_i = 0  # index after which a new trade may start
        last_fill_i = 0

        for ts in events.sort_values():
            entry_i = idx.searchsorted(ts)
            if entry_i >= len(close_arr) - 2:
                continue
            if entry_i < next_free_i:
                # Already in a position — skip this event (concurrency cap)
                continue
            s = int(side.iloc[entry_i])
            if s == 0:
                continue
            m = int(mode.iloc[entry_i])
            atr_frac = float(atr_arr[entry_i])

            if m == 1:  # trend mode
                pt_mult = cfg.trend_pt_atr
                sl_mult = cfg.trend_sl_atr
                max_hold = cfg.trend_max_hold_bars
                trail = cfg.trend_trail_atr
            elif m == 2:  # range mode
                pt_mult = cfg.range_pt_atr
                sl_mult = cfg.range_sl_atr
                max_hold = cfg.range_max_hold_bars
                trail = None
            else:  # legacy fallback
                pt_mult = cfg.pt_mult
                sl_mult = cfg.sl_mult
                max_hold = cfg.max_hold_bars
                trail = None

            # Ensure TP target clears round-trip cost
            rt_cost = 2 * (fee + slip)
            pt_frac = pt_mult * atr_frac
            if pt_frac < max(rt_cost * 2.0, cfg.min_target_frac):
                # Widen barriers proportionally rather than silently skip,
                # so we only take setups with genuine edge.
                needed = max(rt_cost * 2.0, cfg.min_target_frac)
                scale = needed / max(pt_frac, 1e-9)
                pt_mult *= scale
                sl_mult *= scale

            exit_i, gross_ret, barrier = self._resolve_barriers(
                close_arr=close_arr,
                entry_i=entry_i,
                s=s,
                atr_frac=atr_frac,
                pt_mult=pt_mult,
                sl_mult=sl_mult,
                max_hold=max_hold,
                trail_atr=trail,
            )

            hold_hours = max(
                (idx[exit_i] - idx[entry_i]).total_seconds() / 3600.0, 0
            )
            funding_cost = (hold_hours / 8.0) * (cfg.funding_bps_per_8h / 1e4)
            cost = 2 * (fee + slip) + funding_cost
            net_ret = gross_ret - cost

            # Risk-normalized sizing. R per trade ≡ risk_per_trade × equity
            # size_frac = risk_per_trade / stop_distance_frac, capped by leverage
            sl_frac = sl_mult * atr_frac
            size_frac = cfg.risk_per_trade / max(sl_frac, 1e-4)
            size_frac = min(size_frac, cfg.max_leverage)

            trade_pnl = equity * size_frac * net_ret
            equity += trade_pnl
            equity_curve.iloc[last_fill_i:exit_i + 1] = equity
            last_fill_i = exit_i
            next_free_i = exit_i + 1

            records.append(
                dict(
                    entry_time=idx[entry_i],
                    exit_time=idx[exit_i],
                    side=s,
                    mode=m,
                    entry=float(close_arr[entry_i]),
                    exit=float(close_arr[exit_i]),
                    atr_frac=atr_frac,
                    pt_frac=pt_mult * atr_frac,
                    sl_frac=sl_frac,
                    gross_ret=gross_ret,
                    net_ret=net_ret,
                    size_frac=size_frac,
                    pnl=trade_pnl,
                    equity_after=equity,
                    barrier=barrier,
                    hold_hours=hold_hours,
                )
            )

        equity_curve = equity_curve.ffill().fillna(cfg.starting_equity)
        trades = pd.DataFrame(records)
        ret = equity_curve.pct_change().fillna(0)
        stats = self._compute_stats(ret, trades, cfg)
        return BacktestResult(trades=trades, equity=equity_curve, returns=ret, stats=stats)

    # --------------------------------------------------------------- metrics
    @staticmethod
    def _compute_stats(ret: pd.Series, trades: pd.DataFrame, cfg: BacktestConfig) -> dict:
        if len(trades) == 0 or ret.std() == 0:
            return {"n_trades": int(len(trades))}
        ann_factor = np.sqrt(cfg.bars_per_year)
        sharpe = ret.mean() / ret.std() * ann_factor
        downside = ret[ret < 0].std()
        sortino = ret.mean() / downside * ann_factor if downside and downside > 0 else np.nan
        equity = (1 + ret).cumprod()
        peak = equity.cummax()
        dd = (equity / peak - 1).min()
        cagr = equity.iloc[-1] ** (cfg.bars_per_year / max(len(ret), 1)) - 1
        win_rate = float((trades["net_ret"] > 0).mean())
        pf_gains = trades.loc[trades["net_ret"] > 0, "net_ret"].sum()
        pf_losses = -trades.loc[trades["net_ret"] < 0, "net_ret"].sum()
        profit_factor = pf_gains / pf_losses if pf_losses > 0 else np.inf
        calmar = cagr / abs(dd) if dd != 0 else np.nan
        # Regime-mode breakdown
        by_mode = {}
        if "mode" in trades.columns:
            for m_id, m_name in {1: "trend", 2: "range"}.items():
                sub = trades[trades["mode"] == m_id]
                if len(sub):
                    by_mode[f"{m_name}_n"] = int(len(sub))
                    by_mode[f"{m_name}_win"] = float((sub["net_ret"] > 0).mean())
                    by_mode[f"{m_name}_mean_ret"] = float(sub["net_ret"].mean())
        return dict(
            n_trades=int(len(trades)),
            sharpe=float(sharpe),
            sortino=float(sortino) if not np.isnan(sortino) else None,
            max_drawdown=float(dd),
            cagr=float(cagr),
            calmar=float(calmar) if not np.isnan(calmar) else None,
            win_rate=win_rate,
            profit_factor=float(profit_factor),
            avg_trade_ret=float(trades["net_ret"].mean()),
            **by_mode,
        )
