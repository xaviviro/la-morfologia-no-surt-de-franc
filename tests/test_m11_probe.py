import numpy as np

from scripts.m11_probe import _probe_accuracy


def test_probe_perfect_when_separable():
    rng = np.random.default_rng(0)
    # two clearly separated clusters, 10 lemmas (groups) per class
    a = rng.normal(+3, 0.3, size=(20, 8))
    b = rng.normal(-3, 0.3, size=(20, 8))
    X = np.vstack([a, b])
    y = np.array([0] * 20 + [1] * 20)
    groups = np.concatenate([np.arange(20), np.arange(20)])  # paired lemmas
    assert _probe_accuracy(X, y, groups) > 0.95


def test_probe_chance_when_random():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 8))
    y = np.array([0, 1] * 20)
    groups = np.repeat(np.arange(20), 2)
    assert _probe_accuracy(X, y, groups) < 0.75  # near chance


def test_probe_nan_single_class():
    X = np.zeros((6, 4))
    y = np.zeros(6, dtype=int)
    groups = np.arange(6)
    assert np.isnan(_probe_accuracy(X, y, groups))
