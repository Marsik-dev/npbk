import numpy as np

from npbk.layer1 import Layer1
from npbk.utils import bit_stability


def test_fit_produces_weights(synthetic_training_set):
    own, other = synthetic_training_set
    w = Layer1().fit(own, other)
    assert w.mu.shape == (own.shape[1],)
    assert w.signs.shape == (own.shape[1],)
    assert set(w.signs).issubset({-1, 1})
    assert w.thresholds.shape == (own.shape[1],)


def test_stability_on_separable_data(synthetic_training_set):
    own, other = synthetic_training_set
    l1 = Layer1()
    w = l1.fit(own, other)
    omega = l1.stability_on(own, w)
    # On a clearly separable dataset most neurons should be highly stable
    assert (omega >= 0.6).mean() > 0.7


def test_forward_shape(synthetic_training_set):
    own, other = synthetic_training_set
    l1 = Layer1()
    w = l1.fit(own, other)
    codes = l1.forward(own, w)
    assert codes.shape[0] == len(own)
    assert codes.shape[1] == w.n_neurons


def test_forward_scalar(synthetic_training_set):
    own, other = synthetic_training_set
    l1 = Layer1()
    w = l1.fit(own, other)
    code = l1.forward(own[0], w)
    assert code.ndim == 1
    assert len(code) == w.n_neurons


def test_gost_weight_formula():
    """Hand-computed reference: verify GOST formula implementation."""
    rng = np.random.default_rng(0)
    own = rng.normal(2.0, 0.1, (15, 4))
    other = rng.normal(0.0, 1.0, (100, 4))

    e_own = own.mean(axis=0)
    std_own = own.std(axis=0) + 1e-8
    e_other = other.mean(axis=0)

    expected_mu = e_own / std_own
    expected_signs = np.sign(e_own - e_other).astype(np.int8)
    expected_signs[expected_signs == 0] = 1
    expected_thresholds = expected_signs * expected_mu * e_other

    w = Layer1().fit(own, other)
    np.testing.assert_allclose(w.mu, expected_mu, rtol=1e-6)
    np.testing.assert_array_equal(w.signs, expected_signs)
    np.testing.assert_allclose(w.thresholds, expected_thresholds, rtol=1e-6)
