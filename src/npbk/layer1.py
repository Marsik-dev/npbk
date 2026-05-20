"""
Layer 1 of NPBK per GOST R 52633.5-2011.

Each neuron i independently computes one output bit:
  mu_i        = E_own[v_i] / std_own[v_i]         (weight magnitude)
  sign_i      = sign(E_own[v_i] - E_other[v_i])   (weight sign)
  threshold_i = sign_i * mu_i * E_other[v_i]       (decision threshold)
  output_i    = 1  if  sign_i * mu_i * v_i >= threshold_i  else 0

Neuron selection strategies (experiments showed discrimination is best):
  "gost"           — omega_own >= t AND omega_other < t  (original GOST)
  "discrimination" — omega_own >= t AND |P(own=1) - P(other=1)| >= disc_t
  "own_only"       — omega_own >= t  (no constraint on Other)
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from .constants import MIN_STABILITY_INDEX
from .types import Layer1Weights
from .utils import binary_forward, bit_stability, safe_std

SelectionStrategy = Literal["gost", "discrimination", "own_only"]


class Layer1:
    def __init__(
        self,
        selection: SelectionStrategy = "discrimination",
        disc_threshold: float = 0.25,
    ) -> None:
        """
        selection:      neuron selection strategy
        disc_threshold: min |P(own=1) - P(other=1)| for "discrimination" mode
        """
        self.selection = selection
        self.disc_threshold = disc_threshold

    def fit(self, own: np.ndarray, other: np.ndarray) -> Layer1Weights:
        e_own   = own.mean(axis=0)
        e_other = other.mean(axis=0)
        std_own = safe_std(own, axis=0)

        mu        = e_own / std_own
        signs     = np.sign(e_own - e_other).astype(np.int8)
        signs[signs == 0] = 1
        thresholds = signs * mu * e_other

        bits_own   = binary_forward(own,   mu, signs, thresholds)
        bits_other = binary_forward(other, mu, signs, thresholds)
        omega_own   = bit_stability(bits_own)
        omega_other = bit_stability(bits_other)

        own_stable = omega_own >= MIN_STABILITY_INDEX

        if self.selection == "gost":
            selected = own_stable & (omega_other < MIN_STABILITY_INDEX)

        elif self.selection == "discrimination":
            p_own   = bits_own.mean(axis=0)
            p_other = bits_other.mean(axis=0)
            discriminates = np.abs(p_own - p_other) >= self.disc_threshold
            selected = own_stable & discriminates

        else:  # "own_only"
            selected = own_stable

        return Layer1Weights(
            mu=mu,
            signs=signs,
            thresholds=thresholds,
            selected=selected,
        )

    def forward(self, vectors: np.ndarray, weights: Layer1Weights) -> np.ndarray:
        scalar = vectors.ndim == 1
        if scalar:
            vectors = vectors[np.newaxis, :]
        bits = binary_forward(vectors, weights.mu, weights.signs, weights.thresholds)
        result = bits[:, weights.selected]
        return result[0] if scalar else result

    def stability_on(self, vectors: np.ndarray, weights: Layer1Weights) -> np.ndarray:
        bits = binary_forward(vectors, weights.mu, weights.signs, weights.thresholds)
        return bit_stability(bits)
