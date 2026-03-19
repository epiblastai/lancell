"""Vectorized Welch's t-test across columns of two dense matrices."""

import math
from collections import namedtuple

import numba as nb
import numpy as np
from scipy.stats import t as t_dist

TTestResult = namedtuple("TTestResult", ("statistic", "pvalue"))


@nb.njit(parallel=True)
def _welch_t_stats(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch's t-statistic and degrees of freedom per column.

    Parameters
    ----------
    x, y
        Dense 2-D float64 arrays with shape ``(n_samples, n_features)``.

    Returns
    -------
    t_stats : ndarray of float64
    dfs : ndarray of float64
    """
    n_a = x.shape[0]
    n_b = y.shape[0]
    n_features = x.shape[1]
    t_stats = np.empty(n_features, dtype=np.float64)
    dfs = np.empty(n_features, dtype=np.float64)

    for j in nb.prange(n_features):  # type: ignore[attr-defined]
        mean_a = 0.0
        for i in range(n_a):
            mean_a += x[i, j]
        mean_a /= n_a

        mean_b = 0.0
        for i in range(n_b):
            mean_b += y[i, j]
        mean_b /= n_b

        var_a = 0.0
        for i in range(n_a):
            d = x[i, j] - mean_a
            var_a += d * d
        var_a /= n_a - 1

        var_b = 0.0
        for i in range(n_b):
            d = y[i, j] - mean_b
            var_b += d * d
        var_b /= n_b - 1

        se = var_a / n_a + var_b / n_b
        if se > 0:
            t_stats[j] = (mean_a - mean_b) / math.sqrt(se)
            # Welch-Satterthwaite degrees of freedom
            dfs[j] = (se * se) / ((var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1))
        else:
            t_stats[j] = 0.0
            dfs[j] = n_a + n_b - 2

    return t_stats, dfs


def welch_ttest(x: np.ndarray, y: np.ndarray) -> TTestResult:
    """Run Welch's t-test for each feature (column) of two dense matrices.

    Parameters
    ----------
    x, y
        Dense 2-D arrays with shape ``(n_samples, n_features)``.
        Must have the same number of columns.

    Returns
    -------
    TTestResult with ``statistic`` and ``pvalue`` arrays of length ``n_features``.
    """
    assert x.shape[1] == y.shape[1], (
        f"x and y must have the same number of columns, got {x.shape[1]} and {y.shape[1]}"
    )
    assert x.shape[0] >= 2, "x must have at least 2 rows"
    assert y.shape[0] >= 2, "y must have at least 2 rows"

    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)

    t_stats, dfs = _welch_t_stats(x, y)
    pvalues = 2.0 * t_dist.sf(np.abs(t_stats), dfs)
    np.clip(pvalues, 0.0, 1.0, out=pvalues)

    return TTestResult(t_stats, pvalues)
