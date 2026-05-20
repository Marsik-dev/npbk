from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TrainingSet:
    """Input data for NPBK training. own: (N≥11, D), other: (N≥64, D)."""
    own_vectors: np.ndarray
    other_vectors: np.ndarray


@dataclass
class Layer1Weights:
    """Analytically computed Layer 1 weights per GOST formula."""
    mu: np.ndarray        # (K,) weight magnitudes
    signs: np.ndarray     # (K,) int8: +1 or -1
    thresholds: np.ndarray  # (K,) decision thresholds
    selected: np.ndarray  # (K,) bool mask — neurons that passed stability check

    @property
    def n_neurons(self) -> int:
        return int(self.selected.sum())


@dataclass
class Layer2Weights:
    """Layer 2 error-correction weights."""
    mu: np.ndarray        # (K2,)
    signs: np.ndarray     # (K2,) int8
    thresholds: np.ndarray  # (K2,)
    selected: np.ndarray  # (K2,) bool mask

    @property
    def n_neurons(self) -> int:
        return int(self.selected.sum())


@dataclass
class QualityReport:
    mean_stability: float
    min_stability: float
    neuron_pass_rate: float
    predicted_far: float
    predicted_frr: float
    n_own_samples: int
    n_other_samples: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class TrainingResult:
    layer1: Layer1Weights
    layer2: Layer2Weights
    stability_own: np.ndarray   # (K,) ω_i for each Layer 1 neuron on Own set
    stability_other: np.ndarray # (K,) ω_i on Other set
    reference_code: np.ndarray  # (K_selected,) binary reference code from mean Own
    hamming_threshold: int
    quality: QualityReport
    passed: bool


@dataclass
class AuthResult:
    accepted: bool
    hamming_distance: int
    confidence: float           # 1 - hamming_distance / code_length
    code: np.ndarray            # produced binary code
    details: dict[str, Any] = field(default_factory=dict)
