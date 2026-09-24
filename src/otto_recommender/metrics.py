"""Evaluation helpers for multi-objective Recall@20."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


METRIC_WEIGHTS = np.asarray([0.10, 0.30, 0.60], dtype=np.float64)


def aggregate_recall(
    hits: Sequence[int] | np.ndarray,
    denominators: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Return objective-level recall from aggregate hit counts."""
    hit_array = np.asarray(hits, dtype=np.float64)
    denominator_array = np.asarray(denominators, dtype=np.float64)
    if hit_array.shape != (3,) or denominator_array.shape != (3,):
        raise ValueError("hits and denominators must contain clicks, carts, and orders")
    if np.any(hit_array < 0) or np.any(denominator_array < 0):
        raise ValueError("hits and denominators must be non-negative")
    return np.divide(
        hit_array,
        denominator_array,
        out=np.zeros(3, dtype=np.float64),
        where=denominator_array > 0,
    )


def weighted_recall(recalls: Sequence[float] | np.ndarray) -> float:
    """Combine click, cart, and order recall using 10/30/60 weights."""
    values = np.asarray(recalls, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError("recalls must contain clicks, carts, and orders")
    return float(values @ METRIC_WEIGHTS)
