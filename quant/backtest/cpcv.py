"""Combinatorial Purged Cross-Validation (AFML ch.12).

Provides two things:
- ``purged_kfold_indices``: simple purged k-fold with embargo
- ``CPCV``: combinatorial purged CV generating all (N choose k) test-fold
  combinations, ensuring leakage-free train/test splits for time series.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterator

import numpy as np
import pandas as pd


def _embargo_window(idx: pd.DatetimeIndex, embargo_pct: float) -> int:
    n = len(idx)
    return max(int(np.ceil(n * embargo_pct)), 1)


def purged_kfold_indices(
    idx: pd.DatetimeIndex,
    event_ends: pd.Series,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) for each fold.

    ``event_ends`` is a Series mapping each event timestamp to its exit time
    (e.g. from the triple-barrier ``t1``). Training observations whose
    event label window overlaps a test fold are purged.
    """
    n = len(idx)
    fold_size = n // n_splits
    embargo = _embargo_window(idx, embargo_pct)
    folds = []
    for k in range(n_splits):
        start = k * fold_size
        stop = (k + 1) * fold_size if k < n_splits - 1 else n
        test_idx = np.arange(start, stop)
        test_times_start = idx[start]
        test_times_end = idx[stop - 1]
        # Purge: drop training points whose event end is inside the test window
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        if event_ends is not None and len(event_ends):
            ee = event_ends.reindex(idx)
            overlap = (ee >= test_times_start) & (idx <= test_times_end)
            train_mask &= ~overlap.values
        # Embargo
        emb_stop = min(stop + embargo, n)
        train_mask[stop:emb_stop] = False
        folds.append((np.where(train_mask)[0], test_idx))
    return folds


class CPCV:
    """Combinatorial Purged CV.

    Generates C(N, k) combinations of test folds. For each combination, returns
    the concatenated test indices and the purged training indices.
    """

    def __init__(self, n_splits: int = 6, n_test_folds: int = 2, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.n_test_folds = n_test_folds
        self.embargo_pct = embargo_pct

    def split(
        self, idx: pd.DatetimeIndex, event_ends: pd.Series | None = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(idx)
        fold_size = n // self.n_splits
        bounds = [
            (k * fold_size, (k + 1) * fold_size if k < self.n_splits - 1 else n)
            for k in range(self.n_splits)
        ]
        embargo = _embargo_window(idx, self.embargo_pct)
        all_idx = np.arange(n)

        for combo in combinations(range(self.n_splits), self.n_test_folds):
            test_idx_parts = [np.arange(*bounds[k]) for k in combo]
            test_idx = np.concatenate(test_idx_parts)
            test_mask = np.zeros(n, dtype=bool)
            test_mask[test_idx] = True

            train_mask = ~test_mask
            # Purge overlap via event_ends
            if event_ends is not None and len(event_ends):
                ee = event_ends.reindex(idx)
                for k in combo:
                    s, e = bounds[k]
                    test_start = idx[s]
                    test_end = idx[e - 1]
                    overlap = (ee >= test_start) & (idx <= test_end)
                    train_mask &= ~overlap.values
            # Embargo after each test segment
            for k in combo:
                s, e = bounds[k]
                emb_stop = min(e + embargo, n)
                train_mask[e:emb_stop] = False
            yield all_idx[train_mask], test_idx
