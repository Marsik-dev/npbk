import tempfile
from pathlib import Path

import numpy as np
import pytest

from npbk import NBKContainer, NPBKTrainer, TrainingSet


def _make_container(rng: np.random.Generator) -> NBKContainer:
    own = rng.normal(1.0, 0.2, (20, 32))
    other = rng.normal(0.0, 0.5, (100, 32))
    result = NPBKTrainer().train(TrainingSet(own, other))
    return NBKContainer.from_result(result, feature_dim=32)


def test_save_load_roundtrip():
    rng = np.random.default_rng(7)
    c = _make_container(rng)
    with tempfile.NamedTemporaryFile(suffix=".nbk") as f:
        c.save(f.name)
        loaded = NBKContainer.load(f.name)
    np.testing.assert_array_equal(c.reference_code, loaded.reference_code)
    assert c.hamming_threshold == loaded.hamming_threshold
    assert c.feature_dim == loaded.feature_dim
    assert c.version == loaded.version


def test_encrypted_roundtrip():
    rng = np.random.default_rng(8)
    c = _make_container(rng)
    password = b"test-password-123"
    with tempfile.NamedTemporaryFile(suffix=".nbk") as f:
        c.save(f.name, password=password)
        loaded = NBKContainer.load(f.name, password=password)
    np.testing.assert_array_equal(c.reference_code, loaded.reference_code)


def test_encrypted_wrong_password_fails():
    rng = np.random.default_rng(9)
    c = _make_container(rng)
    with tempfile.NamedTemporaryFile(suffix=".nbk") as f:
        c.save(f.name, password=b"correct")
        with pytest.raises(Exception):
            NBKContainer.load(f.name, password=b"wrong")


def test_no_raw_biometrics_in_file():
    """The serialized container must not contain raw own feature vectors."""
    rng = np.random.default_rng(10)
    own = rng.normal(1.0, 0.2, (20, 32))
    other = rng.normal(0.0, 0.5, (100, 32))
    result = NPBKTrainer().train(TrainingSet(own, other))
    c = NBKContainer.from_result(result, feature_dim=32)
    with tempfile.NamedTemporaryFile(suffix=".nbk") as f:
        c.save(f.name)
        raw = Path(f.name).read_bytes()
    # Own vectors should not appear verbatim in the file
    for row in own:
        assert row.tobytes() not in raw
