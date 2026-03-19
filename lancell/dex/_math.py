"""Math functions for differential expression: pseudobulk, fold change, normalization.

Adapted from pdex (https://github.com/ArcInstitute/pdex/) with the addition
of normalize_log1p_sparse().
"""

import numba as nb
import numpy as np
from scipy.sparse import csr_matrix

from lancell.dex._numba_mwu import (
    MannWhitneyUResult,
    SparseColumnIndex,
    mannwhitneyu_dense,
    mannwhitneyu_sparse,
)


@nb.njit(parallel=True)
def _log1p_col_mean(matrix: np.ndarray) -> np.ndarray:
    """Mean of log1p(X) across rows (axis=0) for a dense 2-D array."""
    n_rows, n_cols = matrix.shape
    result = np.zeros(n_cols)
    for j in nb.prange(n_cols):  # type: ignore[attr-defined]
        s = 0.0
        for i in range(n_rows):
            s += np.log1p(matrix[i, j])
        result[j] = s / n_rows
    return result


@nb.njit(parallel=True)
def _expm1_vec(x: np.ndarray) -> np.ndarray:
    """Element-wise expm1 over a 1-D array."""
    result = np.empty_like(x)
    for i in nb.prange(len(x)):  # type: ignore[attr-defined]
        result[i] = np.expm1(x[i])
    return result


@nb.njit(parallel=True)
def _expm1_vec_mean(matrix: np.ndarray) -> np.ndarray:
    """Mean of expm1(X) across rows (axis=0) for a dense 2-D array."""
    n_rows, n_cols = matrix.shape
    result = np.zeros(n_cols)
    for j in nb.prange(n_cols):  # type: ignore[attr-defined]
        s = 0.0
        for i in range(n_rows):
            s += np.expm1(matrix[i, j])
        result[j] = s / n_rows
    return result


def bulk_matrix_arithmetic(matrix: np.ndarray | csr_matrix, is_log1p: bool, axis=0) -> np.ndarray:
    """Arithmetic mean across cells (axis=0), returned in natural (count) space."""
    if is_log1p:
        if isinstance(matrix, csr_matrix):
            m = matrix.copy()
            np.expm1(m.data, out=m.data)
            return np.array(m.mean(axis=axis)).flatten()
        return _expm1_vec_mean(np.asarray(matrix, dtype=np.float64))
    return np.array(matrix.mean(axis=axis)).flatten()


def bulk_matrix_geometric(matrix: np.ndarray | csr_matrix, is_log1p: bool, axis=0) -> np.ndarray:
    """Geometric mean of expression values, back-transformed to count space."""
    if is_log1p:
        log_mean = np.array(matrix.mean(axis=axis)).flatten()
    elif isinstance(matrix, csr_matrix):
        m = matrix.copy()
        np.log1p(m.data, out=m.data)
        log_mean = np.array(m.mean(axis=axis)).flatten()
    else:
        log_mean = _log1p_col_mean(np.asarray(matrix, dtype=np.float64))
    return _expm1_vec(log_mean)


def pseudobulk(matrix: np.ndarray | csr_matrix, geometric_mean: bool, is_log1p: bool) -> np.ndarray:
    """Compute pseudobulk summary across cells (axis=0).

    Always returns values in natural (count) space.
    """
    if geometric_mean:
        return bulk_matrix_geometric(matrix, is_log1p=is_log1p)
    return bulk_matrix_arithmetic(matrix, is_log1p=is_log1p)


@nb.njit(parallel=True)
def fold_change(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculates the log2-fold change between two arrays."""
    return np.log2(x / y)


@nb.njit(parallel=True)
def percent_change(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculates the change between two arrays."""
    return (x - y) / y


def mwu(
    x: np.ndarray | csr_matrix | SparseColumnIndex,
    y: np.ndarray | csr_matrix | SparseColumnIndex,
) -> MannWhitneyUResult:
    """Route to sparse or dense MWU based on input type."""
    from scipy.sparse import issparse

    if isinstance(x, SparseColumnIndex) or issparse(x):
        return mannwhitneyu_sparse(x, y)
    return mannwhitneyu_dense(x, y)


def normalize_log1p_sparse(X: csr_matrix, target_sum: float) -> csr_matrix:
    """In-place library-size normalization + log1p on a CSR matrix.

    For each row: x = log1p(x * target_sum / row_sum).
    Operates on ``.data`` array directly using CSR indptr for row boundaries.
    Only touches nonzero entries.
    """
    data = X.data.astype(np.float64, copy=False)
    indptr = X.indptr

    for i in range(X.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        if start == end:
            continue
        row_sum = data[start:end].sum()
        if row_sum > 0:
            data[start:end] *= target_sum / row_sum

    np.log1p(data, out=data)
    X.data = data.astype(np.float32)
    return X
