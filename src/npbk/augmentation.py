"""
Synthetic biometric sample generation per GOST R 52633.2-2010.

Generates additional "Own" feature vectors from a small real sample set
when the number of real recordings is below the required minimum.

Techniques used (in order of preference):
1. Convex interpolation between two existing Own samples
2. Gaussian noise perturbation within measured intra-class variance
3. Affine jitter (small scale + shift in feature space)
"""
from __future__ import annotations

import numpy as np


def augment_own(
    own_vectors: np.ndarray,
    target_count: int,
    noise_scale: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Augment Own-class vectors to reach target_count.

    own_vectors:  (N, D) — real Own samples
    target_count: desired number of samples (≥ N)
    noise_scale:  Gaussian noise magnitude as fraction of per-feature std

    Returns (target_count, D) array — original samples first, then synthetic.
    """
    n, d = own_vectors.shape
    if n >= target_count:
        return own_vectors

    rng = rng or np.random.default_rng()
    std = own_vectors.std(axis=0) + 1e-8
    augmented: list[np.ndarray] = list(own_vectors)

    while len(augmented) < target_count:
        method = rng.integers(0, 3)

        if method == 0 or n < 2:
            # Technique 1: interpolation between two random samples
            i, j = rng.choice(n, size=2, replace=(n < 2))
            alpha = rng.uniform(0.25, 0.75)
            new = alpha * own_vectors[i] + (1 - alpha) * own_vectors[j]
            # Small noise on top
            new = new + rng.standard_normal(d) * std * noise_scale

        elif method == 1:
            # Technique 2: pure Gaussian noise around a random sample
            i = rng.integers(n)
            scale = rng.uniform(0.03, 0.08)
            new = own_vectors[i] + rng.standard_normal(d) * std * scale

        else:
            # Technique 3: affine jitter (scale + shift)
            i = rng.integers(n)
            scale = rng.uniform(0.95, 1.05)
            shift = rng.standard_normal(d) * std * 0.03
            new = own_vectors[i] * scale + shift

        augmented.append(new.astype(np.float32))

    return np.array(augmented[:target_count], dtype=np.float32)


def augmentation_info(original_count: int, target_count: int) -> str:
    """Human-readable description of how many samples will be generated."""
    if original_count >= target_count:
        return f"{original_count} образцов — достаточно (ГОСТ требует ≥{target_count})"
    n_gen = target_count - original_count
    return (
        f"{original_count} реальных + {n_gen} синтетических (ГОСТ Р 52633.2) "
        f"= {target_count} образцов"
    )
