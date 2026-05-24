"""Pure geometry metrics for morphological offset vectors.

An "offset" is v(derived) - v(base) for a (base, derived) word pair. A clean,
compositional morphological operation shows up as a *consistent linear
direction* across a family of such offsets.
"""

from __future__ import annotations

import numpy as np


def _l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)


def direction_consistency(offsets: np.ndarray) -> float:
    """Mean cosine of each offset to the family's mean direction.

    1.0 = every pair moves the same way (a clean morpheme direction);
    ~0.0 = no shared direction.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    unit = _l2norm(offsets, axis=1)
    mean_dir = unit.mean(axis=0)
    mean_dir = mean_dir / np.clip(np.linalg.norm(mean_dir), 1e-12, None)
    return float((unit @ mean_dir).mean())


def pc1_variance_ratio(offsets: np.ndarray) -> float:
    """Fraction of variance explained by the first principal component of the
    centred offsets. 1.0 = the offsets lie on a single line (maximally linear).
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    centred = offsets - offsets.mean(axis=0, keepdims=True)
    # SVD on centred data; squared singular values are proportional to variance.
    s = np.linalg.svd(centred, compute_uv=False)
    var = s**2
    total = var.sum()
    if total <= 0:
        return 0.0
    return float(var[0] / total)


def analogy_correct_per_pair(base: np.ndarray, derived: np.ndarray) -> np.ndarray:
    """Per-pair leave-one-out correctness (1.0/0.0).

    For each pair j, predict derived[j] ≈ base[j] + mean(offset over i != j) and
    check whether the true derived[j] is the nearest (cosine) among all derived
    candidates. The mean of this array is ``analogy_accuracy_loo``. Bootstrapping
    these per-pair outcomes (rather than re-running LOO on resampled indices)
    avoids the duplicate-index self-collision that would bias a nearest-neighbour
    statistic."""
    base = np.asarray(base, dtype=np.float64)
    derived = np.asarray(derived, dtype=np.float64)
    n = len(base)
    out = np.zeros(n)
    if n < 2:
        return out
    offsets = derived - base
    derived_unit = _l2norm(derived, axis=1)
    for j in range(n):
        loo_mean = (offsets.sum(axis=0) - offsets[j]) / (n - 1)
        pred = base[j] + loo_mean
        pred_unit = pred / np.clip(np.linalg.norm(pred), 1e-12, None)
        sims = derived_unit @ pred_unit
        out[j] = 1.0 if int(np.argmax(sims)) == j else 0.0
    return out


def analogy_accuracy_loo(base: np.ndarray, derived: np.ndarray) -> float:
    """Leave-one-out top-1 analogy accuracy in [0, 1] (mean per-pair correctness)."""
    if len(base) < 2:
        return 0.0
    return float(analogy_correct_per_pair(base, derived).mean())


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Paired bootstrap CI for mean(a - b). Resamples pair indices with
    replacement so shared per-pair noise cancels (mirrors the coca study)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, "paired inputs must align"
    diff = a - b
    rng = np.random.default_rng(seed)
    m = len(diff)
    means = np.empty(n)
    for k in range(n):
        idx = rng.integers(0, m, size=m)
        means[k] = diff[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def bootstrap_delta_ci(stat_fn, base, derived_native, derived_morph,
                       n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Paired bootstrap CI for the morphemic−native delta of a family statistic.

    `stat_fn(base, derived) -> float` is recomputed on each resample of pair
    indices for both conditions (base is shared), and the per-resample delta
    `stat_fn(base, derived_morph) − stat_fn(base, derived_native)` is collected.
    Returns (lo, hi) at the given confidence level; (nan, nan) if < 2 pairs."""
    base = np.asarray(base, dtype=np.float64)
    dn = np.asarray(derived_native, dtype=np.float64)
    dm = np.asarray(derived_morph, dtype=np.float64)
    m = len(base)
    if m < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n)
    for k in range(n):
        idx = rng.integers(0, m, size=m)
        deltas[k] = stat_fn(base[idx], dm[idx]) - stat_fn(base[idx], dn[idx])
    return float(np.quantile(deltas, alpha / 2)), float(np.quantile(deltas, 1 - alpha / 2))


def _two_sided_p(samples: np.ndarray) -> float:
    """Bootstrap two-sided p that the resampled statistic differs from 0."""
    n = len(samples)
    p = 2.0 * min(float(np.mean(samples <= 0.0)), float(np.mean(samples >= 0.0)))
    return float(min(1.0, max(p, 1.0 / n)))  # floor at 1/n, cap at 1


def bootstrap_delta_ci_p(stat_fn, base, derived_a, derived_b,
                         n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Like bootstrap_delta_ci but also returns a two-sided bootstrap p-value for
    delta = stat_fn(base, derived_a) − stat_fn(base, derived_b).
    Returns (lo, hi, p); (nan, nan, nan) if < 2 pairs."""
    base = np.asarray(base, dtype=np.float64)
    da = np.asarray(derived_a, dtype=np.float64)
    db = np.asarray(derived_b, dtype=np.float64)
    m = len(base)
    if m < 2:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n)
    for k in range(n):
        idx = rng.integers(0, m, size=m)
        deltas[k] = stat_fn(base[idx], da[idx]) - stat_fn(base[idx], db[idx])
    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    return lo, hi, _two_sided_p(deltas)


def paired_bootstrap_ci_p(a, b, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Paired bootstrap CI + two-sided p-value for mean(a − b)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, "paired inputs must align"
    diff = a - b
    m = len(diff)
    if m < 2:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for k in range(n):
        means[k] = diff[rng.integers(0, m, size=m)].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi, _two_sided_p(means)


def bootstrap_mean_ci(values, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Bootstrap CI for the mean of `values` (e.g. per-family deltas aggregated
    into a per-model number). Returns (mean, lo, hi); NaNs are dropped."""
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    m = len(values)
    if m == 0:
        return (float("nan"), float("nan"), float("nan"))
    if m == 1:
        return (float(values[0]), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for k in range(n):
        means[k] = values[rng.integers(0, m, size=m)].mean()
    return (float(values.mean()),
            float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def benjamini_hochberg(pvalues) -> np.ndarray:
    """Benjamini–Hochberg FDR-adjusted q-values for a 1D array of p-values.
    NaNs are preserved (excluded from the correction)."""
    p = np.asarray(pvalues, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    pv = p[valid]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]  # enforce monotonicity
    q = np.clip(q, 0.0, 1.0)
    adj = np.empty(m)
    adj[order] = q
    out[valid] = adj
    return out
