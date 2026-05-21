"""
NBK Container — secure serialization of NPBK parameters.

Stores only trained weights and the reference code.
Never stores raw biometric data.
Format: MessagePack payload, optionally AES-256-GCM encrypted.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .constants import CONTAINER_VERSION
from .types import Layer1Weights, Layer2Weights, QualityReport, TrainingResult


def _arr_to_list(a: np.ndarray) -> dict[str, Any]:
    return {"dtype": str(a.dtype), "shape": list(a.shape), "data": a.tobytes()}


def _list_to_arr(d: dict[str, Any]) -> np.ndarray:
    return np.frombuffer(d["data"], dtype=d["dtype"]).reshape(d["shape"])


@dataclass
class NBKContainer:
    version: str
    created_at: str
    layer1: Layer1Weights
    layer2: Layer2Weights
    reference_code: np.ndarray
    hamming_threshold: int
    quality: QualityReport
    feature_dim: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(
        cls,
        result: TrainingResult,
        feature_dim: int,
        metadata: dict[str, Any] | None = None,
    ) -> "NBKContainer":
        return cls(
            version=CONTAINER_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(),
            layer1=result.layer1,
            layer2=result.layer2,
            reference_code=result.reference_code,
            hamming_threshold=result.hamming_threshold,
            quality=result.quality,
            feature_dim=feature_dim,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "feature_dim": self.feature_dim,
            "hamming_threshold": self.hamming_threshold,
            "reference_code": _arr_to_list(self.reference_code),
            "layer1": {
                "mu": _arr_to_list(self.layer1.mu),
                "signs": _arr_to_list(self.layer1.signs),
                "thresholds": _arr_to_list(self.layer1.thresholds),
                "selected": _arr_to_list(self.layer1.selected),
            },
            "layer2": {
                "mu": _arr_to_list(self.layer2.mu),
                "signs": _arr_to_list(self.layer2.signs),
                "thresholds": _arr_to_list(self.layer2.thresholds),
                "selected": _arr_to_list(self.layer2.selected),
            },
            "quality": {
                "mean_stability": self.quality.mean_stability,
                "min_stability": self.quality.min_stability,
                "neuron_pass_rate": self.quality.neuron_pass_rate,
                "predicted_far": self.quality.predicted_far,
                "predicted_frr": self.quality.predicted_frr,
                "n_own_samples": self.quality.n_own_samples,
                "n_other_samples": self.quality.n_other_samples,
                "warnings": self.quality.warnings,
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NBKContainer":
        l1 = d["layer1"]
        l2 = d["layer2"]
        q = d["quality"]
        return cls(
            version=d["version"],
            created_at=d["created_at"],
            feature_dim=d["feature_dim"],
            hamming_threshold=d["hamming_threshold"],
            reference_code=_list_to_arr(d["reference_code"]),
            layer1=Layer1Weights(
                mu=_list_to_arr(l1["mu"]),
                signs=_list_to_arr(l1["signs"]),
                thresholds=_list_to_arr(l1["thresholds"]),
                selected=_list_to_arr(l1["selected"]),
            ),
            layer2=Layer2Weights(
                mu=_list_to_arr(l2["mu"]),
                signs=_list_to_arr(l2["signs"]),
                thresholds=_list_to_arr(l2["thresholds"]),
                selected=_list_to_arr(l2["selected"]),
            ),
            quality=QualityReport(
                mean_stability=q["mean_stability"],
                min_stability=q["min_stability"],
                neuron_pass_rate=q["neuron_pass_rate"],
                predicted_far=q["predicted_far"],
                predicted_frr=q["predicted_frr"],
                n_own_samples=q["n_own_samples"],
                n_other_samples=q["n_other_samples"],
                warnings=q["warnings"],
            ),
            metadata=d.get("metadata", {}),
        )

    def to_bytes(self, password: bytes | None = None) -> bytes:
        """Serialize to bytes (for DB storage). Optionally encrypt."""
        payload = msgpack.packb(self.to_dict(), use_bin_type=True)
        if password is not None:
            key = _derive_key(password)
            nonce = os.urandom(12)
            aesgcm = AESGCM(key)
            ciphertext = aesgcm.encrypt(nonce, payload, None)
            return b"NBK1ENC" + nonce + ciphertext
        return b"NBK1RAW" + payload

    @classmethod
    def from_bytes(cls, data: bytes, password: bytes | None = None) -> "NBKContainer":
        """Deserialize from bytes (from DB storage)."""
        magic = data[:7]
        body = data[7:]
        if magic == b"NBK1ENC":
            if password is None:
                raise ValueError("Container is encrypted — provide password")
            key = _derive_key(password)
            nonce, ciphertext = body[:12], body[12:]
            aesgcm = AESGCM(key)
            payload = aesgcm.decrypt(nonce, ciphertext, None)
        elif magic == b"NBK1RAW":
            payload = body
        else:
            raise ValueError(f"Unknown container format magic: {magic!r}")
        d = msgpack.unpackb(payload, raw=False)
        return cls.from_dict(d)

    def save(self, path: str | Path, password: bytes | None = None) -> None:
        payload = msgpack.packb(self.to_dict(), use_bin_type=True)
        if password is not None:
            key = _derive_key(password)
            nonce = os.urandom(12)
            aesgcm = AESGCM(key)
            ciphertext = aesgcm.encrypt(nonce, payload, None)
            data = b"NBK1ENC" + nonce + ciphertext
        else:
            data = b"NBK1RAW" + payload
        Path(path).write_bytes(data)

    @classmethod
    def load(cls, path: str | Path, password: bytes | None = None) -> "NBKContainer":
        data = Path(path).read_bytes()
        magic = data[:7]
        body = data[7:]
        if magic == b"NBK1ENC":
            if password is None:
                raise ValueError("Container is encrypted — provide password")
            key = _derive_key(password)
            nonce, ciphertext = body[:12], body[12:]
            aesgcm = AESGCM(key)
            payload = aesgcm.decrypt(nonce, ciphertext, None)
        elif magic == b"NBK1RAW":
            payload = body
        else:
            raise ValueError(f"Unknown container format magic: {magic!r}")
        d = msgpack.unpackb(payload, raw=False)
        return cls.from_dict(d)


def _derive_key(password: bytes) -> bytes:
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=b"npbk-salt-v1", iterations=100_000)
    return kdf.derive(password)
