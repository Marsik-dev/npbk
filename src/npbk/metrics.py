"""Quality metrics per GOST R 52633.5-2011."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class EvaluationReport:
    far: float
    frr: float
    eer: float
    roc_thresholds: np.ndarray
    roc_far: np.ndarray
    roc_tar: np.ndarray
    hamming_hist_own: np.ndarray
    hamming_hist_other: np.ndarray


def stability_index(bits: np.ndarray) -> np.ndarray:
    """ω_i = 2*|0.5 - P(bit_i=1)|. bits: (N, K). Returns (K,)."""
    p1 = bits.mean(axis=0)
    return 2.0 * np.abs(0.5 - p1)


def predict_far(stability_indices: np.ndarray, n_bits: int | None = None) -> float:
    """
    Estimate FAR from the stability distribution.
    Models each bit as independent Bernoulli with probability 0.5*(1 - ω_i) of error.
    FAR ≈ P(Hamming(random, key) <= threshold).
    Uses a Gaussian approximation of the Binomial sum.
    """
    if n_bits is None:
        n_bits = len(stability_indices)
    # For Other images, each bit is ~Bernoulli(0.5) → error prob = 0.5
    # Mean and std of Hamming distance for random vector vs key
    p_err = 0.5  # Other images are random
    mu_h = n_bits * p_err
    sigma_h = np.sqrt(n_bits * p_err * (1 - p_err))
    threshold = optimal_hamming_threshold(stability_indices)
    return float(norm.cdf(threshold, loc=mu_h, scale=sigma_h))


def predict_frr(stability_indices: np.ndarray, hamming_threshold: int) -> float:
    """
    Estimate FRR from stability distribution.
    Each stable bit has error prob ≈ (1 - ω_i) / 2.
    """
    p_err = (1.0 - stability_indices) / 2.0  # per-bit error probability for Own
    mu_h = p_err.sum()
    sigma_h = float(np.sqrt((p_err * (1 - p_err)).sum())) + 1e-8
    return float(1.0 - norm.cdf(hamming_threshold, loc=mu_h, scale=sigma_h))


def optimal_hamming_threshold(
    stability_indices: np.ndarray,
    target_far: float = 1e-6,
    n_bits: int | None = None,
) -> int:
    """Find minimum Hamming threshold that keeps predicted FAR ≤ target_far."""
    if n_bits is None:
        n_bits = len(stability_indices)
    p_err = 0.5
    mu_h = n_bits * p_err
    sigma_h = float(np.sqrt(n_bits * p_err * (1 - p_err)))
    # FAR = Φ((t - mu_h) / sigma_h) ≤ target_far  →  t ≤ mu_h + z*sigma_h
    z = norm.ppf(target_far)
    threshold = int(np.floor(mu_h + z * sigma_h))
    return max(0, threshold)


def evaluate(
    own_codes: np.ndarray,    # (N_own, K) binary
    other_codes: np.ndarray,  # (N_other, K) binary
    reference_code: np.ndarray,  # (K,) binary
) -> EvaluationReport:
    """Empirical FAR/FRR evaluation."""
    own_dists = (own_codes != reference_code).sum(axis=1)
    other_dists = (other_codes != reference_code).sum(axis=1)

    n_bits = reference_code.shape[0]
    thresholds = np.arange(0, n_bits + 1)

    roc_far = np.array([(other_dists <= t).mean() for t in thresholds])
    roc_tar = np.array([(own_dists <= t).mean() for t in thresholds])
    roc_frr = 1.0 - roc_tar

    # EER: point where FAR ≈ FRR
    diffs = np.abs(roc_far - roc_frr)
    eer_idx = int(np.argmin(diffs))
    eer = float((roc_far[eer_idx] + roc_frr[eer_idx]) / 2)

    # Use first threshold where FAR drops below target
    tgt_idx = int(np.searchsorted(-roc_far, -1e-6))
    far = float(roc_far[min(tgt_idx, len(roc_far) - 1)])
    frr = float(roc_frr[min(tgt_idx, len(roc_frr) - 1)])

    return EvaluationReport(
        far=far,
        frr=frr,
        eer=eer,
        roc_thresholds=thresholds,
        roc_far=roc_far,
        roc_tar=roc_tar,
        hamming_hist_own=own_dists,
        hamming_hist_other=other_dists,
    )
