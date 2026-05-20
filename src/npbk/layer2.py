"""
Layer 2 of NPBK per GOST R 52633.5-2011.

Operates on binary outputs of Layer 1. Uses the same statistical fitting
approach as Layer 1 but on binary inputs, acting as an error-correction layer.
"""
from __future__ import annotations

import numpy as np

from .constants import MIN_STABILITY_INDEX
from .types import Layer2Weights
from .utils import binary_forward, bit_stability, safe_std


class Layer2:
    def fit(self, own_codes: np.ndarray, other_codes: np.ndarray) -> Layer2Weights:
        """
        own_codes:   (N_own,  K1) binary — Layer 1 outputs on Own set
        other_codes: (N_other, K1) binary — Layer 1 outputs on Other set
        """
        # Cast to float for statistical computations
        own_f = own_codes.astype(np.float64)
        other_f = other_codes.astype(np.float64)

        e_own = own_f.mean(axis=0)
        e_other = other_f.mean(axis=0)
        std_own = safe_std(own_f, axis=0)

        mu = e_own / std_own
        signs = np.sign(e_own - e_other).astype(np.int8)
        signs[signs == 0] = 1
        thresholds = signs * mu * e_other

        bits_own = binary_forward(own_f, mu, signs, thresholds)
        bits_other = binary_forward(other_f, mu, signs, thresholds)
        omega_own = bit_stability(bits_own)
        omega_other = bit_stability(bits_other)

        selected = (omega_own >= MIN_STABILITY_INDEX) & (omega_other < MIN_STABILITY_INDEX)

        return Layer2Weights(
            mu=mu,
            signs=signs,
            thresholds=thresholds,
            selected=selected,
        )

    def forward(self, codes: np.ndarray, weights: Layer2Weights) -> np.ndarray:
        """
        codes: (N, K1) or (K1,) binary
        Returns (N, K2_selected) or (K2_selected,).
        """
        scalar = codes.ndim == 1
        if scalar:
            codes = codes[np.newaxis, :]
        codes_f = codes.astype(np.float64)
        bits = binary_forward(codes_f, weights.mu, weights.signs, weights.thresholds)
        result = bits[:, weights.selected]
        return result[0] if scalar else result
