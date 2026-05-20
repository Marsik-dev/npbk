from __future__ import annotations

import numpy as np


def safe_std(x: np.ndarray, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    """Standard deviation with small epsilon to avoid division by zero."""
    return np.std(x, axis=axis) + eps


def binary_forward(
    vectors: np.ndarray,  # (N, D)
    mu: np.ndarray,       # (D,)
    signs: np.ndarray,    # (D,) int8
    thresholds: np.ndarray,  # (D,)
) -> np.ndarray:
    """Apply a statistical neural layer: output_i = 1 if sign_i*mu_i*v_i >= t_i."""
    weighted = signs * mu * vectors  # (N, D)
    return (weighted >= thresholds).astype(np.int8)


def bit_stability(bits: np.ndarray) -> np.ndarray:
    """
    Compute stability index ω_i = 2*|0.5 - P(bit_i=1)| per GOST formula.
    bits: (N, K) binary array. Returns (K,) stability indices in [0, 1].
    """
    p1 = bits.mean(axis=0)
    return 2.0 * np.abs(0.5 - p1)
