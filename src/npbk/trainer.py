"""Full two-layer NPBK training procedure per GOST R 52633.5-2011."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    MIN_NEURON_PASS_RATE,
    MIN_OTHER_SAMPLES,
    MIN_OWN_SAMPLES,
    MIN_STABILITY_INDEX,
    TARGET_FAR,
)
from .layer1 import Layer1
from .layer2 import Layer2
from .metrics import make_layer_report, optimal_hamming_threshold, predict_far, predict_frr
from .types import QualityReport, TrainingResult, TrainingSet
from .utils import binary_forward, bit_stability


@dataclass
class TrainerConfig:
    min_stability: float = MIN_STABILITY_INDEX
    target_far: float = TARGET_FAR
    # "discrimination" confirmed best by experiments on RAVDESS
    selection: str = "discrimination"
    disc_threshold: float = 0.25


class NPBKTrainer:
    def __init__(self, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig()

    def validate(self, ts: TrainingSet) -> list[str]:
        errors: list[str] = []
        if len(ts.own_vectors) < MIN_OWN_SAMPLES:
            errors.append(
                f"Own samples: {len(ts.own_vectors)} < required {MIN_OWN_SAMPLES}"
            )
        if len(ts.other_vectors) < MIN_OTHER_SAMPLES:
            errors.append(
                f"Other samples: {len(ts.other_vectors)} < required {MIN_OTHER_SAMPLES}"
            )
        if ts.own_vectors.ndim != 2 or ts.other_vectors.ndim != 2:
            errors.append("Vectors must be 2-dimensional arrays (N, D)")
        elif ts.own_vectors.shape[1] != ts.other_vectors.shape[1]:
            errors.append("Own and Other vectors must have the same dimensionality")
        return errors

    def train(self, ts: TrainingSet) -> TrainingResult:
        errors = self.validate(ts)
        if errors:
            raise ValueError("Training set validation failed:\n" + "\n".join(errors))

        own = ts.own_vectors.astype(np.float64)
        other = ts.other_vectors.astype(np.float64)

        # --- Layer 1 ---
        l1 = Layer1(
            selection=self.config.selection,       # type: ignore[arg-type]
            disc_threshold=self.config.disc_threshold,
        )
        w1 = l1.fit(own, other)

        own_codes_l1 = binary_forward(own, w1.mu, w1.signs, w1.thresholds)
        other_codes_l1 = binary_forward(other, w1.mu, w1.signs, w1.thresholds)

        stability_own_full = bit_stability(own_codes_l1)    # all neurons
        stability_other_full = bit_stability(other_codes_l1)

        # Select stable neurons for Layer 1
        own_codes_sel = own_codes_l1[:, w1.selected]
        other_codes_sel = other_codes_l1[:, w1.selected]

        # Layer 1 report (before correction)
        l1_stability_own = bit_stability(own_codes_sel)
        l1_stability_other = bit_stability(other_codes_sel)
        layer1_report = make_layer_report(
            layer_num=1,
            own_codes=own_codes_sel,
            other_codes=other_codes_sel,
            n_total=own_codes_l1.shape[1],
            stability_own=l1_stability_own,
            stability_other=l1_stability_other,
        )

        # --- Layer 2 ---
        l2 = Layer2()
        w2 = l2.fit(own_codes_sel, other_codes_sel)

        own_codes_l2 = binary_forward(
            own_codes_sel.astype(np.float64), w2.mu, w2.signs, w2.thresholds
        )
        other_codes_l2 = binary_forward(
            other_codes_sel.astype(np.float64), w2.mu, w2.signs, w2.thresholds
        )

        # Final selected neurons from Layer 2
        own_final = own_codes_l2[:, w2.selected]
        other_final = other_codes_l2[:, w2.selected]

        # Reference code = majority vote over Own training samples
        reference_code = (own_final.mean(axis=0) >= 0.5).astype(np.int8)

        # Stability on final selected neurons
        stability_own = bit_stability(own_final)
        stability_other = bit_stability(other_final)

        pass_rate = float((stability_own >= self.config.min_stability).mean())
        hamming_threshold = optimal_hamming_threshold(
            stability_own, target_far=self.config.target_far
        )

        warnings: list[str] = []
        if pass_rate < MIN_NEURON_PASS_RATE:
            warnings.append(
                f"Only {pass_rate:.1%} of neurons pass stability threshold "
                f"(required {MIN_NEURON_PASS_RATE:.0%})"
            )
        if len(reference_code) < 8:
            warnings.append(f"Final code length {len(reference_code)} is very short")

        quality = QualityReport(
            mean_stability=float(stability_own.mean()),
            min_stability=float(stability_own.min()) if len(stability_own) else 0.0,
            neuron_pass_rate=pass_rate,
            predicted_far=predict_far(stability_own),
            predicted_frr=predict_frr(stability_own, hamming_threshold),
            n_own_samples=len(own),
            n_other_samples=len(other),
            warnings=warnings,
        )

        passed = len(warnings) == 0 and len(reference_code) >= 8

        # Layer 2 report (after correction)
        layer2_report = make_layer_report(
            layer_num=2,
            own_codes=own_final,
            other_codes=other_final,
            n_total=own_codes_l2.shape[1],
            stability_own=stability_own,
            stability_other=stability_other,
        )

        return TrainingResult(
            layer1=w1,
            layer2=w2,
            stability_own=stability_own,
            stability_other=stability_other,
            reference_code=reference_code,
            hamming_threshold=hamming_threshold,
            quality=quality,
            passed=passed,
            layer1_report=layer1_report,
            layer2_report=layer2_report,
        )
