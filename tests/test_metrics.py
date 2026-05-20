import numpy as np

from npbk.metrics import evaluate, stability_index


def test_stability_index_perfect():
    # All 1s → ω = 1.0
    bits = np.ones((100, 8), dtype=np.int8)
    omega = stability_index(bits)
    np.testing.assert_allclose(omega, 1.0)


def test_stability_index_random():
    # ~50/50 bits → ω ≈ 0
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, (10000, 16), dtype=np.int8)
    omega = stability_index(bits)
    assert (omega < 0.1).all()


def test_evaluate_shapes():
    rng = np.random.default_rng(5)
    ref = rng.integers(0, 2, (32,), dtype=np.int8)
    own_codes = rng.integers(0, 2, (20, 32), dtype=np.int8)
    other_codes = rng.integers(0, 2, (100, 32), dtype=np.int8)
    report = evaluate(own_codes, other_codes, ref)
    assert 0 <= report.eer <= 1
    assert len(report.roc_far) == len(report.roc_tar)
