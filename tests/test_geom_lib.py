import numpy as np
import pytest

from scripts.geom_lib import (
    direction_consistency,
    pc1_variance_ratio,
    analogy_accuracy_loo,
    analogy_correct_per_pair,
    paired_bootstrap_ci,
    bootstrap_delta_ci,
    bootstrap_delta_ci_p,
    paired_bootstrap_ci_p,
    bootstrap_mean_ci,
    benjamini_hochberg,
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


def test_bootstrap_delta_ci_p_small_when_morph_cleaner():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(30, 8))
    der_a = base + np.ones(8)                 # clean direction
    der_b = base + rng.normal(size=(30, 8))   # noisy direction

    def stat(b, d):
        return direction_consistency(d - b)

    lo, hi, p = bootstrap_delta_ci_p(stat, base, der_a, der_b, n=400, seed=1)
    assert lo > 0.0 and p < 0.05


def test_bootstrap_delta_ci_p_large_when_identical():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(30, 8))
    der = base + rng.normal(size=(30, 8))

    def stat(b, d):
        return direction_consistency(d - b)

    lo, hi, p = bootstrap_delta_ci_p(stat, base, der, der, n=400, seed=3)
    assert lo <= 0.0 <= hi and p > 0.2


def test_paired_bootstrap_ci_p_small_when_shifted():
    rng = np.random.default_rng(4)
    a = rng.normal(0.5, 0.05, size=40)
    b = a - 0.1 + rng.normal(0, 0.01, size=40)
    lo, hi, p = paired_bootstrap_ci_p(a, b, n=1000, seed=0)
    assert lo > 0.0 and p < 0.05


def test_bootstrap_mean_ci_brackets_mean():
    rng = np.random.default_rng(5)
    vals = rng.normal(0.1, 0.05, size=12)
    mean, lo, hi = bootstrap_mean_ci(vals, n=1000, seed=0)
    assert mean == pytest.approx(vals.mean())
    assert lo < mean < hi


def test_benjamini_hochberg_monotone_and_ge_p():
    p = np.array([0.001, 0.01, 0.02, 0.5, 0.8])
    q = benjamini_hochberg(p)
    assert np.all(q >= p - 1e-9)          # q-values are >= raw p
    assert np.all(np.diff(q) >= -1e-9)    # non-decreasing in p order
    assert np.all((q >= 0) & (q <= 1))


def test_benjamini_hochberg_preserves_nan():
    q = benjamini_hochberg([0.01, np.nan, 0.04])
    assert np.isnan(q[1]) and not np.isnan(q[0])


def test_spearman_perfect_monotone():
    from scripts.geom_lib import spearman_r
    assert spearman_r([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman_r([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_handles_ties_and_nan():
    from scripts.geom_lib import spearman_r
    import numpy as np
    r = spearman_r([1, 1, 2, 3], [1, 2, 2, 3])
    assert -1.0 <= r <= 1.0
    assert not np.isnan(spearman_r([1, 2, np.nan, 3], [3, 2, 5, 1]))


def test_bootstrap_diff_ci_positive_when_a_higher():
    from scripts.geom_lib import bootstrap_diff_ci
    rng = np.random.default_rng(0)
    a = rng.normal(0.6, 0.05, size=6)
    b = rng.normal(0.5, 0.05, size=4)
    mean, lo, hi, p = bootstrap_diff_ci(a, b, n=500, seed=1)
    assert mean > 0 and lo < mean < hi
