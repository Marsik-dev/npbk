from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TrainingSet:
    """Input data for NPBK training. own: (N≥11, D), other: (N≥64, D)."""
    own_vectors: np.ndarray
    other_vectors: np.ndarray
    emotion: str | None = None  # when set, other must contain only this emotion


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
class LayerReport:
    """FAR/FRR + stability metrics after a single training layer."""
    layer_num: int               # 1 or 2
    n_neurons_total: int         # before selection filter
    n_neurons_selected: int      # after stability filter
    stability_own: np.ndarray    # (K,) ω_i on Own
    stability_other: np.ndarray  # (K,) ω_i on Other
    mean_stability_own: float
    mean_stability_other: float
    predicted_far: float
    predicted_frr: float
    empirical_far: float         # actual FAR on training Other set
    empirical_frr: float         # actual FRR on training Own set
    hamming_threshold: int


@dataclass
class TrainingResult:
    layer1: Layer1Weights
    layer2: Layer2Weights
    stability_own: np.ndarray   # (K,) ω_i for final selected neurons on Own
    stability_other: np.ndarray # (K,) ω_i on Other
    reference_code: np.ndarray  # binary reference code from mean Own
    hamming_threshold: int
    quality: QualityReport
    passed: bool
    # Per-layer diagnostics (new)
    layer1_report: "LayerReport | None" = None
    layer2_report: "LayerReport | None" = None


@dataclass
class AuthResult:
    accepted: bool
    hamming_distance: int
    confidence: float           # 1 - hamming_distance / code_length
    code: np.ndarray            # produced binary code
    details: dict[str, Any] = field(default_factory=dict)
