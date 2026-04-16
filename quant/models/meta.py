"""Meta-labeling classifier (LightGBM).

Given the primary model's candidate signals and the feature matrix, train a
binary classifier that predicts whether the trade will hit TP before SL. The
classifier uses the triple-barrier labels generated in quant/labeling.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class MetaParams:
    num_leaves: int = 31
    max_depth: int = -1
    learning_rate: float = 0.05
    n_estimators: int = 400
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    threshold: float = 0.55


class MetaLabeler:
    def __init__(self, params: MetaParams | None = None):
        self.p = params or MetaParams()
        self.model = None
        self.feature_names: list[str] = []

    # ---------------------------------------------------------------- train
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> None:
        try:
            import lightgbm as lgb
        except Exception as exc:
            raise ImportError(
                "lightgbm required for meta-labeling. pip install lightgbm"
            ) from exc

        # Guard against empty / degenerate training sets. Without this,
        # LightGBM + sklearn's compute_class_weight(class_weight="balanced")
        # crashes with a cryptic "IndexError: arrays used as indices must be
        # of integer (or boolean) type" when y has zero samples, because
        # np.unique on an empty array returns an empty float64 array.
        y_arr = np.asarray(y).astype(int, copy=False)
        if len(X) == 0 or len(y_arr) == 0:
            raise ValueError(
                "MetaLabeler.fit: no training samples available. "
                "The primary model produced no events to label — try widening "
                "the date range, loosening primary-signal thresholds, or "
                "lowering CUSUM filter sensitivity."
            )
        unique_classes = np.unique(y_arr)
        if unique_classes.size < 2:
            only = int(unique_classes[0]) if unique_classes.size else None
            raise ValueError(
                f"MetaLabeler.fit: training labels contain only one class "
                f"({only!r}); cannot fit a binary classifier. Collect more "
                f"data or adjust triple-barrier parameters so both TP and "
                f"SL outcomes are represented."
            )

        self.feature_names = list(X.columns)
        self.model = lgb.LGBMClassifier(
            num_leaves=self.p.num_leaves,
            max_depth=self.p.max_depth,
            learning_rate=self.p.learning_rate,
            n_estimators=self.p.n_estimators,
            min_child_samples=self.p.min_child_samples,
            subsample=self.p.subsample,
            colsample_bytree=self.p.colsample_bytree,
            reg_alpha=self.p.reg_alpha,
            reg_lambda=self.p.reg_lambda,
            objective="binary",
            class_weight="balanced",
            verbose=-1,
        )
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight.loc[X.index].values
        # Pass DataFrame so LightGBM records feature names (prevents mismatch
        # warnings at predict time).
        self.model.fit(X, y_arr, **fit_kwargs)

    # --------------------------------------------------------------- predict
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MetaLabeler not fitted")
        missing = [f for f in self.feature_names if f not in X.columns]
        if missing:
            raise KeyError(f"missing meta features at predict time: {missing[:5]}")
        return self.model.predict_proba(X[self.feature_names])[:, 1]

    def filter_signals(
        self, side: pd.Series, X: pd.DataFrame, threshold: float | None = None
    ) -> pd.DataFrame:
        thr = threshold if threshold is not None else self.p.threshold
        events = side[side != 0].index
        if len(events) == 0:
            return pd.DataFrame(columns=["side", "prob"])
        proba = self.predict_proba(X.loc[events])
        out = pd.DataFrame(
            {"side": side.loc[events].astype(int), "prob": proba}, index=events
        )
        out["take"] = out["prob"] >= thr
        return out

    # ----------------------------------------------------------------- I/O
    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump({"model": self.model, "features": self.feature_names, "params": self.p}, path)
        logger.info(f"saved meta-model to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "MetaLabeler":
        import joblib

        obj = joblib.load(path)
        inst = cls(params=obj["params"])
        inst.model = obj["model"]
        inst.feature_names = obj["features"]
        return inst
