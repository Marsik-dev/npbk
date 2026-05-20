import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_training_set(rng: np.random.Generator):
    """Simple separable training set: Own centered at +1, Other at 0."""
    D = 32
    own = rng.normal(loc=1.0, scale=0.2, size=(20, D))
    other = rng.normal(loc=0.0, scale=0.5, size=(100, D))
    return own, other
