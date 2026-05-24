import numpy as np
import pytest

from scripts.geom_lib import (
    direction_consistency,
    pc1_variance_ratio,
    analogy_accuracy_loo,
    analogy_correct_per_pair,
    paired_bootstrap_ci,
    bootstrap_delta_ci,
)


def test_direction_consistency_aligned_is_high():
    offsets = np.array([[1.0, 0.0], [0.9, 0.1], [1.1, -0.05], [1.0, 0.02]])
    assert direction_consistency(offsets) > 0.98


def test_direction_consistency_random_is_low():
    rng = np.random.default_rng(0)
    offsets = rng.normal(size=(200, 50))
    # mean cosine of random high-dim vectors to their mean ≈ small
    assert abs(direction_consistency(offsets)) < 0.3


def test_pc1_variance_ratio_single_axis_is_one():
    offsets = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [-1.0, 0.0]])
    assert pc1_variance_ratio(offsets) == pytest.approx(1.0, abs=1e-6)


def test_pc1_variance_ratio_isotropic_is_about_half():
    rng = np.random.default_rng(1)
    offsets = rng.normal(size=(500, 2))
    assert 0.4 < pc1_variance_ratio(offsets) < 0.6


def test_analogy_accuracy_perfect_when_offset_constant():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(6, 8))
    direction = np.ones(8)
    derived = base + direction  # perfectly constant offset
    assert analogy_accuracy_loo(base, derived) == pytest.approx(1.0)


def test_analogy_accuracy_chance_when_unrelated():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(20, 8))
    derived = rng.normal(size=(20, 8))  # no consistent offset
    acc = analogy_accuracy_loo(base, derived)
    assert acc < 0.5


def test_paired_bootstrap_ci_brackets_mean_difference():
    rng = np.random.default_rng(4)
    native = rng.normal(0.4, 0.05, size=40)
    # add small noise so CI is non-degenerate while mean stays ~0.1
    morphemic = native + 0.1 + rng.normal(0, 0.01, size=40)
    lo, hi = paired_bootstrap_ci(morphemic, native, n=1000, seed=0)
    assert lo > 0.0 and hi > lo
    assert lo < 0.1 < hi


def test_bootstrap_delta_ci_positive_when_morph_cleaner():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(30, 8))
    der_native = base + rng.normal(size=(30, 8))  # noisy per-pair direction
    der_morph = base + np.ones(8)                 # clean constant direction

    def stat(b, d):
        return direction_consistency(d - b)

    lo, hi = bootstrap_delta_ci(stat, base, der_native, der_morph, n=400, seed=1)
    assert lo > 0.0  # morphemic clearly more consistent; CI excludes 0


def test_analogy_correct_per_pair_mean_equals_accuracy():
    rng = np.random.default_rng(7)
    base = rng.normal(size=(12, 8))
    derived = base + np.ones(8) + rng.normal(0, 0.1, size=(12, 8))
    per_pair = analogy_correct_per_pair(base, derived)
    assert per_pair.shape == (12,)
    assert set(np.unique(per_pair)) <= {0.0, 1.0}
    assert per_pair.mean() == pytest.approx(analogy_accuracy_loo(base, derived))


def test_bootstrap_delta_ci_brackets_zero_when_identical():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(30, 8))
    der = base + rng.normal(size=(30, 8))

    def stat(b, d):
        return direction_consistency(d - b)

    lo, hi = bootstrap_delta_ci(stat, base, der, der, n=400, seed=3)
    assert lo <= 0.0 <= hi  # identical conditions → delta ≈ 0
