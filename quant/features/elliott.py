"""Rule-based Elliott Wave tagger.

We do NOT attempt to count the full fractal tree — that's a rabbit hole. We
look at the last N pivots and check classic impulsive-wave rules:

    - W1 up leg, W2 retrace 38.2–78.6% of W1
    - W3 >= 1.618 * W1 and is NOT the shortest of (W1,W3,W5)
    - W4 does not overlap W1 territory
    - W5 typically 0.618–1.618 of W1 (or 1:1 of W0→W3)

Output per bar:
    ew_wave_id   : 0=unknown, 1..5=impulse waves, -1..-3=corrective a/b/c
    ew_confidence: 0..1, sum of rules satisfied
    ew_in_wave3  : 1 if current bar is inside a confirmed wave-3
    ew_in_wave5  : 1 if current bar is inside a confirmed wave-5
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.features.zigzag import zigzag_pivots


def _check_impulse(p: list[float], k: list[int]) -> tuple[int, float]:
    """Given 6 pivots P0..P5 (alternating), return (wave_id, confidence 0..1).

    wave_id in {1,2,3,4,5} describes which wave the most recent segment is in,
    0 if the structure doesn't look like an impulse.
    """
    if len(p) < 6:
        return 0, 0.0
    P0, P1, P2, P3, P4, P5 = p[-6:]
    K0, K1, K2, K3, K4, K5 = k[-6:]
    # Expect alternating pivots. Orient up-impulse if K0 == -1 (low) at start.
    direction = +1 if K0 == -1 else -1
    w1 = (P1 - P0) * direction
    w2 = (P1 - P2) * direction  # retrace
    w3 = (P3 - P2) * direction
    w4 = (P3 - P4) * direction  # retrace
    w5 = (P5 - P4) * direction
    if min(w1, w3, w5) <= 0 or min(w2, w4) <= 0:
        return 0, 0.0

    rules = 0
    total = 5
    # Rule 1: W2 retraces 0.382–0.786 of W1
    r2 = w2 / w1 if w1 else 0
    if 0.382 <= r2 <= 0.786:
        rules += 1
    # Rule 2: W3 >= 1.618 * W1
    if w3 >= 1.618 * w1:
        rules += 1
    # Rule 3: W3 not the shortest
    if w3 >= w1 and w3 >= w5:
        rules += 1
    # Rule 4: W4 does not overlap W1 territory
    p1_level = P0 + direction * w1
    p4_level = P3 - direction * w4
    if (direction == +1 and p4_level > p1_level) or (
        direction == -1 and p4_level < p1_level
    ):
        rules += 1
    # Rule 5: W5 in 0.382–1.618 of W1
    r5 = w5 / w1 if w1 else 0
    if 0.382 <= r5 <= 1.618:
        rules += 1

    conf = rules / total
    return (5 if conf >= 0.6 else 0), conf


def elliott_features(df: pd.DataFrame, atr_mult: float = 3.0) -> pd.DataFrame:
    pivots = zigzag_pivots(df, atr_mult=atr_mult)
    n = len(df)
    wave_id = np.zeros(n, dtype=int)
    conf = np.zeros(n, dtype=float)
    in_w3 = np.zeros(n, dtype=float)
    in_w5 = np.zeros(n, dtype=float)

    if len(pivots) < 6:
        return pd.DataFrame(
            {
                "ew_wave_id": wave_id,
                "ew_confidence": conf,
                "ew_in_wave3": in_w3,
                "ew_in_wave5": in_w5,
            },
            index=df.index,
        )

    piv_pos = np.searchsorted(df.index.values, pivots.index.values, side="left")
    prices = pivots["price"].tolist()
    kinds = pivots["kind"].tolist()

    # For each bar, look back at the last 6 pivots completed before this bar.
    for i in range(n):
        j = np.searchsorted(piv_pos, i, side="right")
        if j < 6:
            continue
        wid, c = _check_impulse(prices[:j], kinds[:j])
        wave_id[i] = wid
        conf[i] = c
        # Wave 3 is between pivot[j-4] and pivot[j-3]; we are beyond wave 5 now,
        # but we label the *current* segment after pivot[j-1]:
        #   segment 0..1 => wave1, 1..2 => wave2, ..., last (j-1..current) => wave5+
        seg = (j - 1) % 5 + 1  # crude but useful tag
        if c >= 0.6:
            if seg == 3:
                in_w3[i] = 1.0
            if seg == 5:
                in_w5[i] = 1.0

    return pd.DataFrame(
        {
            "ew_wave_id": wave_id,
            "ew_confidence": conf,
            "ew_in_wave3": in_w3,
            "ew_in_wave5": in_w5,
        },
        index=df.index,
    )
