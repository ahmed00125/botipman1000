"""Event-driven backtest engine.

- Takes a sequence of CUSUM events + primary model side + (optional) meta proba
- Simulates execution with fees, slippage and funding cost
- Uses triple-barrier outcomes to resolve each trade
- Aggregates PnL, equity curve, per-trade records, and risk-adjusted metrics
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from quant.labeling.triple_barrier import apply_triple_barrier, get_daily_vol


@dataclass
class BacktestConfig:
    fee_taker: float = 0.00055
    fee_maker: float = 0.0002
    slippage_bps: float = 1.0       # 0.01%
    funding_bps_per_8h: float = 1.0  # rough avg
    starting_equity: float = 10_000.0
    max_leverage: float = 3.0
    risk_per_trade: float = 0.01
    use_maker: bool = False
    pt_mult: float = 2.0
    sl_mult: float = 1.0
    max_hold_bars: int = 48
    meta_threshold: float = 0.55
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

    def run(
        self,
        close: pd.Series,
        events: pd.DatetimeIndex,
        side: pd.Series,
        meta_proba: pd.Series | None = None,
    ) -> BacktestResult:
        cfg = self.cfg
        # Target vol scale from daily vol (AFML)
        target_vol = get_daily_vol(close, span=100).reindex(close.index).ffill().bfill()
        target_vol = target_vol.clip(lower=1e-4)

        # Apply triple barrier to get outcomes
        events = events.intersection(side.index)
        events = events[side.loc[events] != 0]
        if meta_proba is not None:
            mp = meta_proba.reindex(events).fillna(0)
            events = events[mp >= cfg.meta_threshold]
        if len(events) == 0:
            eq = pd.Series([cfg.starting_equity], index=close.index[:1])
            return BacktestResult(
                trades=pd.DataFrame(),
                equity=eq,
                returns=pd.Series(dtype=float),
                stats={"n_trades": 0},
            )

        labels = apply_triple_barrier(
            close=close,
            events=events,
            target_vol=target_vol,
            pt_mult=cfg.pt_mult,
            sl_mult=cfg.sl_mult,
            max_hold_bars=cfg.max_hold_bars,
            side=side.reindex(events).fillna(0),
        )

        fee = cfg.fee_maker if cfg.use_maker else cfg.fee_taker
        slip = cfg.slippage_bps / 1e4

        records = []
        equity = cfg.starting_equity
        equity_curve = pd.Series(index=close.index, dtype=float)
        equity_curve.iloc[0] = equity
        last_update_i = 0

        close_idx = close.index
        close_arr = close.values

        for ts, row in labels.sort_index().iterrows():
            entry_i = close_idx.searchsorted(ts)
            exit_i = close_idx.searchsorted(row["t1"])
            if entry_i >= len(close_arr) or exit_i >= len(close_arr):
                continue
            entry_p = close_arr[entry_i]
            exit_p = close_arr[exit_i]
            s = int(row["side"])
            gross_ret = s * (exit_p / entry_p - 1)

            # Funding cost: 8-hour window count
            hold_hours = max((row["t1"] - ts).total_seconds() / 3600.0, 0)
            funding_periods = hold_hours / 8.0
            funding_cost = funding_periods * (cfg.funding_bps_per_8h / 1e4)
            # Entry + exit fees (both sides) and slippage both sides
            cost = 2 * (fee + slip) + funding_cost
            net_ret = gross_ret - cost

            # Sizing: vol-target capped by risk_per_trade and max leverage
            realized_vol = float(target_vol.loc[ts]) if ts in target_vol.index else 0.01
            daily_vol_ann = realized_vol * np.sqrt(cfg.bars_per_year / 24)
            if daily_vol_ann <= 0:
                leverage = 1.0
            else:
                leverage = min(cfg.vol_target_ann / max(daily_vol_ann, 1e-4), cfg.max_leverage)
            risk_cap = cfg.risk_per_trade / max(abs(row["sl"]), 1e-4)
            size_frac = min(leverage, risk_cap, cfg.max_leverage)

            trade_pnl = equity * size_frac * net_ret
            equity += trade_pnl
            # Fill the equity curve flat until this exit point
            equity_curve.iloc[last_update_i:exit_i + 1] = equity
            last_update_i = exit_i

            records.append(
                dict(
                    entry_time=ts,
                    exit_time=row["t1"],
                    side=s,
                    entry=entry_p,
                    exit=exit_p,
                    gross_ret=gross_ret,
                    net_ret=net_ret,
                    size_frac=size_frac,
                    pnl=trade_pnl,
                    equity_after=equity,
                    barrier=row["bin"],
                    hold_hours=hold_hours,
                )
            )

        equity_curve = equity_curve.ffill().fillna(cfg.starting_equity)
        trades = pd.DataFrame(records)
        ret = equity_curve.pct_change().fillna(0)
        stats = self._compute_stats(ret, trades, cfg)
        return BacktestResult(trades=trades, equity=equity_curve, returns=ret, stats=stats)

    @staticmethod
    def _compute_stats(ret: pd.Series, trades: pd.DataFrame, cfg: BacktestConfig) -> dict:
        if len(trades) == 0 or ret.std() == 0:
            return {"n_trades": len(trades)}
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
        )
