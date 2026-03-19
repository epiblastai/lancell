"""Numba-accelerated Mann-Whitney U test for sparse CSR matrices.

Adapted from pdex (https://github.com/ArcInstitute/pdex/), consolidated into
a single file since we only need the sparse path. Works directly with CSR
format by building a lightweight column index (permutation array + column
pointers) without copying data values.
"""

import math
from collections import namedtuple

import numba as nb
import numpy as np

# Alternative hypothesis encoding
TWO_SIDED = 0
LESS = 1
GREATER = 2

_ALTERNATIVE_MAP = {
    "two-sided": TWO_SIDED,
    "less": LESS,
    "greater": GREATER,
}

MannWhitneyUResult = namedtuple("MannWhitneyUResult", ("statistic", "pvalue"))
SparseColumnIndex = namedtuple(
    "SparseColumnIndex", ("data", "col_indptr", "col_order", "n_rows", "n_cols")
)


# ---------------------------------------------------------------------------
# Core kernels
# ---------------------------------------------------------------------------


@nb.njit
def _ndtr(x: float) -> float:
    """Normal CDF using math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


@nb.njit
def _build_col_index(csr_indptr, csr_indices, n_cols):
    """Build column pointers and a permutation from CSR arrays.

    Returns
    -------
    col_indptr : int64 array (n_cols + 1)
    col_order : int64 array (nnz)
    """
    nnz = csr_indices.shape[0]

    col_counts = np.zeros(n_cols, dtype=np.int64)
    for k in range(nnz):
        col_counts[csr_indices[k]] += 1

    col_indptr = np.empty(n_cols + 1, dtype=np.int64)
    col_indptr[0] = 0
    for j in range(n_cols):
        col_indptr[j + 1] = col_indptr[j] + col_counts[j]

    pos = np.empty(n_cols, dtype=np.int64)
    for j in range(n_cols):
        pos[j] = col_indptr[j]

    col_order = np.empty(nnz, dtype=np.int64)
    n_rows = csr_indptr.shape[0] - 1
    for i in range(n_rows):
        for k in range(csr_indptr[i], csr_indptr[i + 1]):
            c = csr_indices[k]
            col_order[pos[c]] = k
            pos[c] += 1

    return col_indptr, col_order


@nb.njit
def _gather_col_vals(csr_data, col_order, start, end):
    """Gather nonzero values for a single column from CSR data via col_order."""
    n = end - start
    vals = np.empty(n, dtype=np.float64)
    for k in range(n):
        vals[k] = csr_data[col_order[start + k]]
    return vals


@nb.njit
def _sparse_mwu_column(vals_a, vals_b, n_a, n_b, use_continuity, alternative):
    """Compute Mann-Whitney U for a single gene from two groups' nonzero values.

    Zeros are treated analytically — they form a contiguous block at the
    start of the sorted order (requires non-negative data).
    """
    n = n_a + n_b
    nnz_a = vals_a.shape[0]
    nnz_b = vals_b.shape[0]
    nnz = nnz_a + nnz_b
    nz = n - nnz
    nz_a = n_a - nnz_a

    if nnz == 0:
        return n_a * n_b / 2.0, 1.0

    all_vals = np.empty(nnz, dtype=np.float64)
    is_a = np.empty(nnz, dtype=np.int8)
    for k in range(nnz_a):
        all_vals[k] = vals_a[k]
        is_a[k] = 1
    for k in range(nnz_b):
        all_vals[nnz_a + k] = vals_b[k]
        is_a[nnz_a + k] = 0

    sort_idx = np.argsort(all_vals)

    tie_term_nz = 0.0
    sum_global_ranks_a = 0.0

    i = 0
    while i < nnz:
        j = i
        while j < nnz - 1 and all_vals[sort_idx[j]] == all_vals[sort_idx[j + 1]]:
            j += 1

        tie_count = float(j - i + 1)
        tie_term_nz += tie_count * tie_count * tie_count - tie_count

        local_avg_rank = (i + j) / 2.0 + 1.0
        global_avg_rank = nz + local_avg_rank

        for k in range(i, j + 1):
            if is_a[sort_idx[k]] == 1:
                sum_global_ranks_a += global_avg_rank

        i = j + 1

    zero_avg_rank = (nz + 1.0) / 2.0
    R1 = nz_a * zero_avg_rank + sum_global_ranks_a

    U1 = R1 - n_a * (n_a + 1.0) / 2.0
    U2 = n_a * n_b - U1

    tie_term = (float(nz) * float(nz) * float(nz) - float(nz)) + tie_term_nz

    mu = n_a * n_b / 2.0
    denom = float(n) * float(n - 1)
    var_inner = (n + 1.0) - tie_term / denom
    s_sq = n_a * n_b / 12.0 * var_inner

    if s_sq <= 0.0:
        return U1, 1.0

    s = math.sqrt(s_sq)

    if alternative == GREATER:
        U = U1
        f = 1.0
    elif alternative == LESS:
        U = U2
        f = 1.0
    else:  # TWO_SIDED
        U = max(U1, U2)
        f = 2.0

    numerator = U - mu
    if use_continuity:
        numerator -= 0.5

    z = numerator / s
    p = _ndtr(-z) * f

    if p > 1.0:
        p = 1.0
    if p < 0.0:
        p = 0.0

    return U1, p


@nb.njit(parallel=True)
def _sparse_mwu_batch(
    data_a,
    col_indptr_a,
    col_order_a,
    n_a,
    data_b,
    col_indptr_b,
    col_order_b,
    n_b,
    use_continuity,
    alternative,
):
    """Run sparse MWU test on each gene using two CSR matrices' column indices."""
    n_genes = col_indptr_a.shape[0] - 1
    U_out = np.empty(n_genes, dtype=np.float64)
    p_out = np.empty(n_genes, dtype=np.float64)

    for j in nb.prange(n_genes):  # type: ignore
        vals_a = _gather_col_vals(data_a, col_order_a, col_indptr_a[j], col_indptr_a[j + 1])
        vals_b = _gather_col_vals(data_b, col_order_b, col_indptr_b[j], col_indptr_b[j + 1])
        U_out[j], p_out[j] = _sparse_mwu_column(
            vals_a, vals_b, n_a, n_b, use_continuity, alternative
        )

    return U_out, p_out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _validate_alternative(alternative):
    alt = alternative.lower()
    if alt not in _ALTERNATIVE_MAP:
        raise ValueError(
            f"`alternative` must be one of {set(_ALTERNATIVE_MAP)}, got {alternative!r}"
        )
    return _ALTERNATIVE_MAP[alt]


def _validate_csr(X, name):
    from scipy.sparse import issparse, isspmatrix_csr

    if not issparse(X):
        raise TypeError(f"`{name}` must be a scipy sparse matrix.")
    if not (isspmatrix_csr(X) or X.format == "csr"):
        raise TypeError(f"`{name}` must be in CSR format. Convert with `{name}.tocsr()` if needed.")
    if X.data.size > 0 and X.data.min() < 0:
        raise ValueError(
            f"Sparse MWU requires non-negative data in `{name}`. "
            "For data with negative values, convert to dense and use "
            "mannwhitneyu_columns."
        )
    return X


def sparse_column_index(X) -> SparseColumnIndex:
    """Precompute a column index for a CSR sparse matrix.

    The returned ``SparseColumnIndex`` can be passed to
    ``mannwhitneyu_sparse`` in place of a raw CSR matrix, avoiding
    redundant index construction when the same matrix is reused
    across many comparisons.
    """
    X = _validate_csr(X, "X")
    if X.shape[0] == 0:
        raise ValueError("`X` must have at least one row.")
    data = np.ascontiguousarray(X.data, dtype=np.float64)
    indptr = np.ascontiguousarray(X.indptr)
    indices = np.ascontiguousarray(X.indices)
    col_indptr, col_order = _build_col_index(indptr, indices, X.shape[1])
    return SparseColumnIndex(data, col_indptr, col_order, X.shape[0], X.shape[1])


def _resolve_sparse(arg, name):
    """Convert a CSR matrix or SparseColumnIndex to a SparseColumnIndex."""
    if isinstance(arg, SparseColumnIndex):
        return arg
    return sparse_column_index(arg)


def mannwhitneyu_sparse(X, Y, use_continuity=True, alternative="two-sided"):
    """Run Mann-Whitney U test for each gene (column) of two sparse matrices.

    Both ``X`` and ``Y`` can be either a CSR matrix or a precomputed
    ``SparseColumnIndex`` (from ``sparse_column_index``).
    """
    idx_a = _resolve_sparse(X, "X")
    idx_b = _resolve_sparse(Y, "Y")

    if idx_a.n_cols != idx_b.n_cols:
        raise ValueError(
            f"`X` and `Y` must have the same number of columns, "
            f"got {idx_a.n_cols} and {idx_b.n_cols}."
        )

    alt = _validate_alternative(alternative)

    stats, pvals = _sparse_mwu_batch(
        idx_a.data,
        idx_a.col_indptr,
        idx_a.col_order,
        idx_a.n_rows,
        idx_b.data,
        idx_b.col_indptr,
        idx_b.col_order,
        idx_b.n_rows,
        use_continuity,
        alt,
    )
    return MannWhitneyUResult(stats, pvals)


# ---------------------------------------------------------------------------
# Dense path
# ---------------------------------------------------------------------------


@nb.njit
def _dense_mwu_column(col_a, col_b, use_continuity, alternative):
    """Compute Mann-Whitney U for one column from two dense groups."""
    n_a = col_a.shape[0]
    n_b = col_b.shape[0]
    n = n_a + n_b

    all_vals = np.empty(n, dtype=np.float64)
    is_a = np.empty(n, dtype=np.int8)
    for i in range(n_a):
        all_vals[i] = col_a[i]
        is_a[i] = 1
    for i in range(n_b):
        all_vals[n_a + i] = col_b[i]
        is_a[n_a + i] = 0

    sort_idx = np.argsort(all_vals)

    tie_term = 0.0
    R1 = 0.0

    i = 0
    while i < n:
        j = i
        while j < n - 1 and all_vals[sort_idx[j]] == all_vals[sort_idx[j + 1]]:
            j += 1

        tie_count = float(j - i + 1)
        tie_term += tie_count * tie_count * tie_count - tie_count

        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if is_a[sort_idx[k]] == 1:
                R1 += avg_rank

        i = j + 1

    U1 = R1 - n_a * (n_a + 1.0) / 2.0

    mu = n_a * n_b / 2.0
    denom = float(n) * float(n - 1)
    var_inner = (n + 1.0) - tie_term / denom
    s_sq = n_a * n_b / 12.0 * var_inner

    if s_sq <= 0.0:
        return U1, 1.0

    s = math.sqrt(s_sq)

    if alternative == GREATER:
        U = U1
        f = 1.0
    elif alternative == LESS:
        U = n_a * n_b - U1
        f = 1.0
    else:
        U = max(U1, n_a * n_b - U1)
        f = 2.0

    numerator = U - mu
    if use_continuity:
        numerator -= 0.5

    z = numerator / s
    p = _ndtr(-z) * f

    if p > 1.0:
        p = 1.0
    if p < 0.0:
        p = 0.0

    return U1, p


@nb.njit(parallel=True)
def _dense_mwu_batch(x, y, use_continuity, alternative):
    """Run dense MWU test on each column of two 2-D arrays."""
    n_features = x.shape[1]
    U_out = np.empty(n_features, dtype=np.float64)
    p_out = np.empty(n_features, dtype=np.float64)

    for j in nb.prange(n_features):  # type: ignore
        U_out[j], p_out[j] = _dense_mwu_column(x[:, j], y[:, j], use_continuity, alternative)

    return U_out, p_out


def mannwhitneyu_dense(X, Y, use_continuity=True, alternative="two-sided"):
    """Run Mann-Whitney U test for each column of two dense matrices.

    Parameters
    ----------
    X, Y
        Dense 2-D arrays with shape ``(n_samples, n_features)``.
        Must have the same number of columns.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("`X` and `Y` must be 2-D arrays.")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"`X` and `Y` must have the same number of columns, got {X.shape[1]} and {Y.shape[1]}."
        )
    if X.shape[0] == 0 or Y.shape[0] == 0:
        raise ValueError("`X` and `Y` must each have at least one row.")

    alt = _validate_alternative(alternative)
    stats, pvals = _dense_mwu_batch(X, Y, use_continuity, alt)
    return MannWhitneyUResult(stats, pvals)
