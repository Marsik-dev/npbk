"""
Layer 1 of NPBK per GOST R 52633.5-2011.

Each neuron i independently computes one output bit:
  mu_i      = E_own[v_i] / std_own[v_i]          (weight magnitude)
  sign_i    = sign(E_own[v_i] - E_other[v_i])     (weight sign)
  threshold_i = mu_i * E_own[v_i]                  (decision threshold)
  output_i  = 1  if  sign_i * mu_i * v_i >= threshold_i  else 0
"""
from __future__ import annotations

import numpy as np

from .constants import MIN_STABILITY_INDEX
from .types import Layer1Weights
from .utils import binary_forward, bit_stability, safe_std


class Layer1:
    def fit(self, own: np.ndarray, other: np.ndarray) -> Layer1Weights:
        """
        Compute weights analytically from Own/Other training sets.

        own:   (N_own,  D)
        other: (N_other, D)
        """
        e_own = own.mean(axis=0)       # (D,)
        e_other = other.mean(axis=0)   # (D,)
        std_own = safe_std(own, axis=0)  # (D,)

        mu = e_own / std_own                         # (D,)
        signs = np.sign(e_own - e_other).astype(np.int8)
        signs[signs == 0] = 1                        # break ties → positive
        # Threshold at E_other so that Own is consistently ≥ threshold
        # and Other is centered at threshold (random output → stability ≈ 0).
        thresholds = signs * mu * e_other            # (D,)

        # Evaluate stability on training Own set to identify useful neurons
        bits_own = binary_forward(own, mu, signs, thresholds)    # (N_own, D)
        bits_other = binary_forward(other, mu, signs, thresholds) # (N_other, D)
        omega_own = bit_stability(bits_own)                       # (D,)

        # A neuron is useful if it is stable on Own AND noisy on Other
        omega_other = bit_stability(bits_other)
        selected = (omega_own >= MIN_STABILITY_INDEX) & (omega_other < MIN_STABILITY_INDEX)

        return Layer1Weights(
            mu=mu,
            signs=signs,
            thresholds=thresholds,
            selected=selected,
        )

    def forward(self, vectors: np.ndarray, weights: Layer1Weights) -> np.ndarray:
        """
        vectors: (N, D) or (D,)
        Returns binary array (N, K_selected) or (K_selected,).
        """
        scalar = vectors.ndim == 1
        if scalar:
            vectors = vectors[np.newaxis, :]
        bits = binary_forward(vectors, weights.mu, weights.signs, weights.thresholds)
        result = bits[:, weights.selected]
        return result[0] if scalar else result

    def stability_on(self, vectors: np.ndarray, weights: Layer1Weights) -> np.ndarray:
        """Return ω_i for all neurons (before selection filter). Shape (D,)."""
        bits = binary_forward(vectors, weights.mu, weights.signs, weights.thresholds)
        return bit_stability(bits)
