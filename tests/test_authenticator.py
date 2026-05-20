import numpy as np

from npbk import NBKContainer, NPBKAuthenticator, NPBKTrainer, TrainingSet


def _setup(rng: np.random.Generator):
    own = rng.normal(1.0, 0.15, (20, 32))
    other = rng.normal(0.0, 0.5, (100, 32))
    result = NPBKTrainer().train(TrainingSet(own, other))
    container = NBKContainer.from_result(result, feature_dim=32)
    return own, other, container


def test_own_sample_accepted():
    rng = np.random.default_rng(42)
    own, _, container = _setup(rng)
    auth = NPBKAuthenticator(container)
    # Most Own samples should be accepted
    accepted = sum(auth.authenticate(v).accepted for v in own)
    assert accepted / len(own) >= 0.7


def test_other_sample_rejected():
    rng = np.random.default_rng(42)
    _, other, container = _setup(rng)
    auth = NPBKAuthenticator(container)
    # Most Other samples should be rejected
    rejected = sum(not auth.authenticate(v).accepted for v in other)
    assert rejected / len(other) >= 0.8


def test_auth_result_fields():
    rng = np.random.default_rng(99)
    own, _, container = _setup(rng)
    result = NPBKAuthenticator(container).authenticate(own[0])
    assert 0.0 <= result.confidence <= 1.0
    assert result.hamming_distance >= 0
    assert result.code.ndim == 1
    assert set(result.code.tolist()).issubset({0, 1})
