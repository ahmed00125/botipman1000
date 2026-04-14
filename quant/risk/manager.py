"""Risk engine: circuit breakers, VaR kill-switch, correlation cap."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class RiskLimits:
    max_leverage: float = 3.0
    risk_per_trade: float = 0.01
    daily_loss_limit: float = 0.03
    max_drawdown: float = 0.15
    max_concurrent_positions: int = 4
    correlation_cap: float = 0.75   # block new pos if corr > cap vs existing
    var_limit: float = 0.05          # 95% historical VaR over 1 day
    liquidation_buffer_atr: float = 2.0


@dataclass
class RiskState:
    equity: float
    peak_equity: float
    daily_pnl: float = 0.0
    day: date = field(default_factory=lambda: datetime.utcnow().date())
    halted: bool = False
    halt_reason: str = ""
    size_multiplier: float = 1.0   # 1.0 normal, 0.5 after soft breach

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.equity - self.peak_equity) / self.peak_equity


class RiskManager:
    def __init__(self, limits: RiskLimits, starting_equity: float):
        self.limits = limits
        self.state = RiskState(equity=starting_equity, peak_equity=starting_equity)
        self.open_positions: Dict[str, dict] = {}
        self.returns_history: Dict[str, pd.Series] = {}  # per-symbol return history

    # ---------------------------------------------------------- bookkeeping
    def update_equity(self, new_equity: float) -> None:
        today = datetime.utcnow().date()
        if today != self.state.day:
            self.state.day = today
            self.state.daily_pnl = 0.0
            if not self.state.halted:
                self.state.size_multiplier = 1.0  # reset soft breaker
        pnl_delta = new_equity - self.state.equity
        self.state.daily_pnl += pnl_delta
        self.state.equity = new_equity
        self.state.peak_equity = max(self.state.peak_equity, new_equity)
        self._check_breakers()

    def set_return_history(self, symbol: str, rets: pd.Series) -> None:
        self.returns_history[symbol] = rets.dropna()

    # ----------------------------------------------------------- breakers
    def _check_breakers(self) -> None:
        dd = self.state.drawdown
        daily_r = self.state.daily_pnl / max(self.state.peak_equity, 1e-9)

        # Soft daily loss → halve size
        if daily_r <= -self.limits.daily_loss_limit * 0.6 and not self.state.halted:
            self.state.size_multiplier = 0.5
            logger.warning(f"soft daily loss breach ({daily_r:.2%}) → size x0.5")

        if daily_r <= -self.limits.daily_loss_limit:
            self._halt(f"daily loss limit: {daily_r:.2%}")

        if dd <= -self.limits.max_drawdown * 0.66:
            self.state.size_multiplier = min(self.state.size_multiplier, 0.5)

        if dd <= -self.limits.max_drawdown:
            self._halt(f"max drawdown: {dd:.2%}")

    def _halt(self, reason: str) -> None:
        self.state.halted = True
        self.state.halt_reason = reason
        self.state.size_multiplier = 0.0
        logger.error(f"RISK HALT: {reason}")

    def reset_halt(self) -> None:
        self.state.halted = False
        self.state.halt_reason = ""
        self.state.size_multiplier = 1.0

    # ------------------------------------------------------- trade gating
    def can_open(self, symbol: str, side: int) -> tuple[bool, str]:
        if self.state.halted:
            return False, self.state.halt_reason
        if len(self.open_positions) >= self.limits.max_concurrent_positions:
            return False, "max concurrent positions"
        if symbol in self.open_positions:
            return False, "position already open"
        # Correlation cap
        if len(self.open_positions) > 0 and symbol in self.returns_history:
            for other in self.open_positions:
                if other not in self.returns_history:
                    continue
                a = self.returns_history[symbol].tail(500)
                b = self.returns_history[other].tail(500)
                joined = pd.concat([a, b], axis=1).dropna()
                if len(joined) >= 50:
                    corr = float(joined.corr().iloc[0, 1])
                    if abs(corr) >= self.limits.correlation_cap:
                        same_side = self.open_positions[other]["side"] == side
                        if same_side:
                            return False, f"correlation cap vs {other}: {corr:.2f}"
        # Historical VaR
        if symbol in self.returns_history:
            rets = self.returns_history[symbol].tail(500)
            if len(rets) >= 100:
                var = -float(np.quantile(rets, 0.05))
                if var > self.limits.var_limit:
                    return False, f"VaR>{self.limits.var_limit:.2%}: {var:.2%}"
        return True, "ok"

    def register_open(self, symbol: str, side: int, qty: float, entry: float) -> None:
        self.open_positions[symbol] = dict(side=side, qty=qty, entry=entry, opened=datetime.utcnow())

    def register_close(self, symbol: str, pnl: float) -> None:
        self.open_positions.pop(symbol, None)
        self.update_equity(self.state.equity + pnl)

    # --------------------------------------------------------- sizing hook
    def effective_risk(self) -> float:
        return self.limits.risk_per_trade * self.state.size_multiplier
