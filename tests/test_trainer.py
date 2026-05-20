import numpy as np
import pytest

from npbk import NPBKTrainer, TrainingSet
from npbk.constants import MIN_OTHER_SAMPLES, MIN_OWN_SAMPLES


def test_train_returns_result(synthetic_training_set):
    own, other = synthetic_training_set
    ts = TrainingSet(own_vectors=own, other_vectors=other)
    result = NPBKTrainer().train(ts)
    assert result.reference_code.ndim == 1
    assert len(result.reference_code) > 0
    assert 0 <= result.quality.mean_stability <= 1.0


def test_train_rejects_too_few_own():
    rng = np.random.default_rng(0)
    ts = TrainingSet(
        own_vectors=rng.random((MIN_OWN_SAMPLES - 1, 16)),
        other_vectors=rng.random((MIN_OTHER_SAMPLES, 16)),
    )
    with pytest.raises(ValueError, match="Own samples"):
        NPBKTrainer().train(ts)


def test_train_rejects_too_few_other():
    rng = np.random.default_rng(0)
    ts = TrainingSet(
        own_vectors=rng.random((MIN_OWN_SAMPLES, 16)),
        other_vectors=rng.random((MIN_OTHER_SAMPLES - 1, 16)),
    )
    with pytest.raises(ValueError, match="Other samples"):
        NPBKTrainer().train(ts)


def test_train_rejects_dim_mismatch():
    rng = np.random.default_rng(0)
    ts = TrainingSet(
        own_vectors=rng.random((12, 16)),
        other_vectors=rng.random((64, 32)),
    )
    with pytest.raises(ValueError, match="dimensionality"):
        NPBKTrainer().train(ts)


def test_reference_code_is_binary(synthetic_training_set):
    own, other = synthetic_training_set
    result = NPBKTrainer().train(TrainingSet(own, other))
    assert set(result.reference_code.tolist()).issubset({0, 1})
