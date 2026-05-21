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


def robust_hamming_threshold(
    own_final: np.ndarray,       # (N_own, K) binary
    reference_code: np.ndarray,  # (K,) binary
    other_holdout_final: np.ndarray | None = None,  # (N_holdout, K) binary
) -> int:
    """
    Compute Hamming threshold as min(5σ_own, 1%-percentile of other holdout - 1).
    The own-based bound covers ~all genuine users (FRR≈0).
    The other-based bound ensures empirical FAR ≤ 1% when holdout is available.
    """
    K = len(reference_code)
    if K == 0:
        return 0
    own_dists = (own_final != reference_code).sum(axis=1).astype(float)
    d_mean = float(own_dists.mean())
    d_std = float(own_dists.std()) + 1.0
    threshold_own = min(int(d_mean + 5.0 * d_std), K)

    if other_holdout_final is not None and len(other_holdout_final) >= 10:
        other_dists = (other_holdout_final != reference_code).sum(axis=1)
        pct1 = int(np.percentile(other_dists, 1))
        threshold_other = max(0, pct1 - 1)
        return min(threshold_own, threshold_other)

    return threshold_own


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


def make_layer_report(
    layer_num: int,
    own_codes: np.ndarray,       # (N_own, K) binary — selected neurons only
    other_codes: np.ndarray,     # (N_other, K) binary — training other
    n_total: int,                # neurons before stability filter
    stability_own: np.ndarray,   # (K,)
    stability_other: np.ndarray, # (K,)
    reference_code: np.ndarray | None = None,    # if None — computed from own_codes
    other_codes_holdout: np.ndarray | None = None,  # holdout Other for unbiased FAR
) -> "LayerReport":  # type: ignore[name-defined]
    """
    Build a LayerReport with both predicted and empirical FAR/FRR.

    Empirical FAR is computed on other_codes_holdout when provided — this avoids
    the training bias where FAR=0 because the same Other data was used to fit weights.
    """
    from .types import LayerReport

    K = own_codes.shape[1] if own_codes.ndim == 2 else 0

    # Use provided reference_code or derive from Own training majority vote
    if reference_code is not None:
        ref = reference_code
    else:
        ref = (own_codes.mean(axis=0) >= 0.5).astype(np.int8)

    # Hamming threshold from own distribution
    own_dists = (own_codes != ref).sum(axis=1).astype(float)
    d_mean = float(own_dists.mean()) if K else 0.0
    d_std = float(own_dists.std()) + 1.0
    threshold = min(int(d_mean + 5.0 * d_std), K)

    # Empirical FRR on training Own
    emp_frr = float((own_dists > threshold).mean()) if K else 1.0

    # Empirical FAR: prefer holdout Other (unbiased) over training Other (biased)
    eval_other = (
        other_codes_holdout
        if other_codes_holdout is not None and len(other_codes_holdout) > 0
        else other_codes
    )
    other_dists = (eval_other != ref).sum(axis=1)
    emp_far = float((other_dists <= threshold).mean())

    return LayerReport(
        layer_num=layer_num,
        n_neurons_total=n_total,
        n_neurons_selected=K,
        stability_own=stability_own,
        stability_other=stability_other,
        mean_stability_own=float(stability_own.mean()) if K else 0.0,
        mean_stability_other=float(stability_other.mean()) if K else 0.0,
        predicted_far=predict_far(stability_own, n_bits=K),
        predicted_frr=predict_frr(stability_own, threshold),
        empirical_far=emp_far,
        empirical_frr=emp_frr,
        hamming_threshold=threshold,
    )
