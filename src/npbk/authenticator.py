"""NPBK authentication (inference) procedure."""
from __future__ import annotations

import numpy as np

from .types import AuthResult
from .utils import binary_forward


class NPBKAuthenticator:
    def __init__(self, container: "NBKContainer") -> None:  # type: ignore[name-defined]
        self.container = container

    def authenticate(self, vector: np.ndarray) -> AuthResult:
        """
        vector: (D,) feature vector (same dimensionality as training set).
        Returns AuthResult with accept/reject decision.
        """
        c = self.container
        v = vector.astype(np.float64)

        # Layer 1 forward pass
        bits_l1_full = binary_forward(
            v[np.newaxis, :], c.layer1.mu, c.layer1.signs, c.layer1.thresholds
        )[0]
        bits_l1 = bits_l1_full[c.layer1.selected]

        # Layer 2 forward pass
        bits_l2_full = binary_forward(
            bits_l1.astype(np.float64)[np.newaxis, :],
            c.layer2.mu,
            c.layer2.signs,
            c.layer2.thresholds,
        )[0]
        code = bits_l2_full[c.layer2.selected]

        # Hamming distance to reference
        hamming = int((code != c.reference_code).sum())
        code_len = len(c.reference_code)
        accepted = hamming <= c.hamming_threshold
        confidence = 1.0 - hamming / max(code_len, 1)

        return AuthResult(
            accepted=accepted,
            hamming_distance=hamming,
            confidence=confidence,
            code=code,
            details={
                "hamming_threshold": c.hamming_threshold,
                "code_length": code_len,
            },
        )
