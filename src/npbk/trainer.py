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
from .metrics import make_layer_report, optimal_hamming_threshold, robust_hamming_threshold
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

        warnings: list[str] = []

        # Split Other into train (≥80%) and holdout (≤20%) for unbiased FAR evaluation.
        # We need at least MIN_OTHER_SAMPLES in the training split.
        n_other = len(other)
        n_holdout = max(10, int(n_other * 0.2))
        n_train = n_other - n_holdout
        if n_train >= MIN_OTHER_SAMPLES:
            rng = np.random.default_rng(42)
            idx = rng.permutation(n_other)
            other_holdout: np.ndarray | None = other[idx[:n_holdout]]
            other_train = other[idx[n_holdout:]]
        else:
            # Not enough samples for a proper holdout — use all for training.
            # Empirical FAR will be biased (over-optimistic) in this case.
            other_train = other
            other_holdout = None
            warnings.append(
                f"Only {n_other} Other samples — need ≥{MIN_OTHER_SAMPLES + 10} for holdout FAR. "
                "Reported FAR may be over-optimistic."
            )

        # --- Layer 1 ---
        l1 = Layer1(
            selection=self.config.selection,       # type: ignore[arg-type]
            disc_threshold=self.config.disc_threshold,
        )
        w1 = l1.fit(own, other_train)

        own_codes_l1 = binary_forward(own, w1.mu, w1.signs, w1.thresholds)
        other_codes_l1_train = binary_forward(other_train, w1.mu, w1.signs, w1.thresholds)
        other_codes_l1_holdout = (
            binary_forward(other_holdout, w1.mu, w1.signs, w1.thresholds)
            if other_holdout is not None else None
        )

        # Select stable neurons for Layer 1
        own_codes_sel = own_codes_l1[:, w1.selected]
        other_codes_sel_train = other_codes_l1_train[:, w1.selected]
        other_codes_sel_holdout = (
            other_codes_l1_holdout[:, w1.selected]
            if other_codes_l1_holdout is not None else None
        )

        # Layer 1 report (before error correction) — FAR on holdout
        l1_stability_own = bit_stability(own_codes_sel)
        l1_stability_other = bit_stability(other_codes_sel_train)
        ref_l1 = (own_codes_sel.mean(axis=0) >= 0.5).astype(np.int8)
        layer1_report = make_layer_report(
            layer_num=1,
            own_codes=own_codes_sel,
            other_codes=other_codes_sel_train,
            n_total=own_codes_l1.shape[1],
            stability_own=l1_stability_own,
            stability_other=l1_stability_other,
            reference_code=ref_l1,
            other_codes_holdout=other_codes_sel_holdout,
        )

        # --- Layer 2 ---
        l2 = Layer2()
        w2 = l2.fit(own_codes_sel, other_codes_sel_train)

        own_codes_l2 = binary_forward(
            own_codes_sel.astype(np.float64), w2.mu, w2.signs, w2.thresholds
        )
        other_codes_l2_train = binary_forward(
            other_codes_sel_train.astype(np.float64), w2.mu, w2.signs, w2.thresholds
        )
        other_codes_l2_holdout = (
            binary_forward(
                other_codes_sel_holdout.astype(np.float64), w2.mu, w2.signs, w2.thresholds
            )
            if other_codes_sel_holdout is not None else None
        )

        # Final selected neurons from Layer 2
        own_final = own_codes_l2[:, w2.selected]
        other_final_train = other_codes_l2_train[:, w2.selected]
        other_final_holdout = (
            other_codes_l2_holdout[:, w2.selected]
            if other_codes_l2_holdout is not None else None
        )

        # Reference code = majority vote over Own training samples
        reference_code = (own_final.mean(axis=0) >= 0.5).astype(np.int8)

        # Stability on final selected neurons
        stability_own = bit_stability(own_final)
        stability_other = bit_stability(other_final_train)

        pass_rate = float((stability_own >= self.config.min_stability).mean())

        # Hamming threshold: min(5σ_own, 1%-percentile of other holdout - 1).
        # This covers ~all genuine users while bounding empirical FAR ≤ 1%.
        K = len(reference_code)
        if K > 0:
            hamming_threshold = robust_hamming_threshold(
                own_final, reference_code, other_final_holdout
            )
        else:
            hamming_threshold = optimal_hamming_threshold(
                stability_own, target_far=self.config.target_far
            )

        if pass_rate < MIN_NEURON_PASS_RATE:
            warnings.append(
                f"Only {pass_rate:.1%} of neurons pass stability threshold "
                f"(required {MIN_NEURON_PASS_RATE:.0%})"
            )
        if len(reference_code) < 8:
            warnings.append(f"Final code length {len(reference_code)} is very short")

        # Empirical FAR/FRR — use holdout for unbiased FAR
        if K > 0:
            own_dists_final = (own_final != reference_code).sum(axis=1)
            empirical_frr = float((own_dists_final > hamming_threshold).mean())
            eval_other_final = other_final_holdout if other_final_holdout is not None else other_final_train
            other_dists_final = (eval_other_final != reference_code).sum(axis=1)
            empirical_far = float((other_dists_final <= hamming_threshold).mean())
        else:
            empirical_frr = 1.0
            empirical_far = 1.0

        quality = QualityReport(
            mean_stability=float(stability_own.mean()),
            min_stability=float(stability_own.min()) if len(stability_own) else 0.0,
            neuron_pass_rate=pass_rate,
            predicted_far=empirical_far,
            predicted_frr=empirical_frr,
            n_own_samples=len(own),
            n_other_samples=len(other),
            warnings=warnings,
        )

        passed = len(warnings) == 0 and len(reference_code) >= 8

        # Layer 2 report (after error correction) — FAR on holdout
        layer2_report = make_layer_report(
            layer_num=2,
            own_codes=own_final,
            other_codes=other_final_train,
            n_total=own_codes_l2.shape[1],
            stability_own=stability_own,
            stability_other=stability_other,
            reference_code=reference_code,
            other_codes_holdout=other_final_holdout,
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
