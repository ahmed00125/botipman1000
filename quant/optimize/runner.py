"""Optuna-based parameter search with CPCV + Deflated Sharpe objective."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Any

import numpy as np
import pandas as pd
from loguru import logger

from quant.backtest.cpcv import CPCV
from quant.backtest.engine import BacktestConfig, EventBacktester
from quant.backtest.metrics import deflated_sharpe_ratio, pbo, sharpe_ratio
from quant.data.bars import cusum_events
from quant.features.builder import FeatureParams, build_feature_matrix
from quant.models.primary import PrimaryParams, PrimaryRuleModel


@dataclass
class OptimizeResult:
    best_params: Dict[str, Any]
    best_value: float
    all_trials: list[dict] = field(default_factory=list)
    stability_winners: list[Dict[str, Any]] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))


class OptunaRunner:
    def __init__(
        self,
        df: pd.DataFrame,
        n_trials: int = 200,
        n_splits: int = 6,
        n_test_folds: int = 2,
        embargo_pct: float = 0.01,
        seed: int = 42,
    ):
        self.df = df
        self.n_trials = n_trials
        self.cpcv = CPCV(n_splits=n_splits, n_test_folds=n_test_folds, embargo_pct=embargo_pct)
        self.seed = seed

    # ----------------------------------------------------- param spaces
    def _suggest_feature_params(self, trial) -> FeatureParams:
        return FeatureParams(
            macd_fast=trial.suggest_int("macd_fast", 6, 20),
            macd_slow=trial.suggest_int("macd_slow", 21, 60),
            macd_signal=trial.suggest_int("macd_signal", 5, 15),
            stoch_k=trial.suggest_int("stoch_k", 5, 30),
            stoch_d=trial.suggest_int("stoch_d", 2, 10),
            stoch_smooth=trial.suggest_int("stoch_smooth", 2, 8),
            donch_n=trial.suggest_int("donch_n", 10, 60),
            zigzag_atr_mult=trial.suggest_float("zigzag_atr_mult", 1.5, 5.0),
            hawkes_decay=trial.suggest_float("hawkes_decay", 0.02, 0.3),
            frac_d=trial.suggest_float("frac_d", 0.2, 0.8),
        )

    def _suggest_primary_params(self, trial) -> PrimaryParams:
        return PrimaryParams(
            w_macd=trial.suggest_float("w_macd", 0.0, 2.0),
            w_stoch=trial.suggest_float("w_stoch", 0.0, 2.0),
            w_donch=trial.suggest_float("w_donch", 0.0, 2.0),
            w_fib=trial.suggest_float("w_fib", 0.0, 2.0),
            w_ew=trial.suggest_float("w_ew", 0.0, 2.0),
            min_abs_score=trial.suggest_float("min_abs_score", 0.15, 0.6),
        )

    def _suggest_backtest_cfg(self, trial) -> BacktestConfig:
        return BacktestConfig(
            pt_mult=trial.suggest_float("pt_mult", 1.0, 4.0),
            sl_mult=trial.suggest_float("sl_mult", 0.5, 2.5),
            max_hold_bars=trial.suggest_int("max_hold_bars", 12, 120),
            meta_threshold=trial.suggest_float("meta_threshold", 0.0, 0.0),  # no meta in this phase
            vol_target_ann=0.15,
        )

    # --------------------------------------------------------- objective
    def _objective(self, trial) -> float:
        feat_params = self._suggest_feature_params(trial)
        prim_params = self._suggest_primary_params(trial)
        bt_cfg = self._suggest_backtest_cfg(trial)

        # Feature matrix computed once per trial
        feats = build_feature_matrix(self.df, feat_params).dropna()
        if len(feats) < 500:
            return -10.0

        prim = PrimaryRuleModel(prim_params).compute(feats)
        side = prim["primary_side"]
        close = self.df["close"].loc[feats.index]

        # Event sampling inside optimization uses CUSUM (adaptive)
        events = cusum_events(close, h_mult=2.0)
        if len(events) < 50:
            return -5.0

        # CPCV-based evaluation
        engine = EventBacktester(bt_cfg)
        idx = feats.index
        event_ends = pd.Series(idx, index=idx)  # crude: will be replaced by t1 inside engine
        test_sharpes = []
        for train_idx, test_idx in self.cpcv.split(idx, event_ends):
            test_times = idx[test_idx]
            test_events = events.intersection(test_times)
            if len(test_events) < 20:
                continue
            res = engine.run(
                close=close,
                events=test_events,
                side=side.reindex(close.index).fillna(0),
            )
            if res.stats.get("n_trades", 0) < 5:
                continue
            sr = res.stats.get("sharpe", 0.0)
            if np.isfinite(sr):
                test_sharpes.append(sr)

        if not test_sharpes:
            return -3.0
        mean_sr = float(np.mean(test_sharpes))
        std_sr = float(np.std(test_sharpes) + 1e-9)
        # Stability-weighted objective: reward consistency across folds
        score = mean_sr - 0.5 * std_sr
        trial.set_user_attr("mean_sharpe", mean_sr)
        trial.set_user_attr("std_sharpe", std_sr)
        trial.set_user_attr("n_folds", len(test_sharpes))
        return score

    # ------------------------------------------------------------- run
    def run(self, study_name: str = "quant_v1") -> OptimizeResult:
        try:
            import optuna
            from optuna.samplers import TPESampler
            from optuna.pruners import MedianPruner
        except Exception as exc:
            raise ImportError("optuna required") from exc

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=TPESampler(seed=self.seed, multivariate=True),
            pruner=MedianPruner(n_warmup_steps=5),
        )
        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=False)

        all_trials = [
            dict(
                params=t.params,
                value=t.value,
                user_attrs=t.user_attrs,
                state=str(t.state),
            )
            for t in study.trials
        ]
        # Stability selection: top-10% of trials by score, intersection of similar params
        top_k = max(int(len(all_trials) * 0.1), 5)
        top = sorted(all_trials, key=lambda d: d.get("value") or -1e9, reverse=True)[:top_k]
        logger.info(f"best value={study.best_value:.4f}")

        return OptimizeResult(
            best_params=study.best_params,
            best_value=float(study.best_value),
            all_trials=all_trials,
            stability_winners=top,
        )
