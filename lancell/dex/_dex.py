"""Core differential expression: MWU or Welch's t-test on sparse or dense data."""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import anndata as ad
import numba
import numpy as np
import polars as pl
from scipy.sparse import issparse

from lancell.dex._math import fold_change, mwu, normalize_log1p_sparse, percent_change, pseudobulk
from lancell.dex._numba_mwu import sparse_column_index
from lancell.dex._ttest import welch_ttest
from lancell.group_specs import PointerKind, get_spec

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_numba_threadpool(threads: int = 0):
    available_threads = os.cpu_count() or 1
    if threads == 0:
        threads = available_threads
    else:
        threads = min(threads, available_threads)
    numba.set_num_threads(threads)


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    if n == 0:
        return np.array([], dtype=np.float64)

    sorted_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_idx]

    fdr = np.empty(n, dtype=np.float64)
    fdr[sorted_idx[-1]] = sorted_pvals[-1]
    cummin = sorted_pvals[-1]
    for i in range(n - 2, -1, -1):
        rank = i + 1
        corrected = sorted_pvals[i] * n / rank
        cummin = min(cummin, corrected)
        fdr[sorted_idx[i]] = cummin

    np.clip(fdr, 0.0, 1.0, out=fdr)
    return fdr


_EMPTY_DEX_SCHEMA = {
    "feature": pl.String,
    "target_mean": pl.Float64,
    "ref_mean": pl.Float64,
    "target_n": pl.Int64,
    "ref_n": pl.Int64,
    "fold_change": pl.Float64,
    "percent_change": pl.Float64,
    "p_value": pl.Float64,
    "statistic": pl.Float64,
    "fdr": pl.Float64,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _group_where(groupby: str, value: str) -> str:
    """Build a WHERE clause for a single group value."""
    escaped = value.replace("'", "''")
    return f"{groupby} = '{escaped}'"


def _load_adata(
    atlas,
    where: str,
    feature_space: str,
    max_records: int | None = None,
) -> ad.AnnData:
    """Load cells from the atlas as AnnData."""
    q = atlas.query().where(where).feature_spaces(feature_space)
    if max_records is not None:
        q = q.limit(max_records)
    return q.to_anndata()


def _extract_matrix(adata: ad.AnnData, pointer_kind: PointerKind):
    """Return CSR for SPARSE, float64 ndarray for DENSE."""
    X = adata.X
    if pointer_kind == PointerKind.SPARSE:
        if not issparse(X):
            from scipy.sparse import csr_matrix

            X = csr_matrix(X)
        elif X.format != "csr":
            X = X.tocsr()
        return X
    else:
        if issparse(X):
            return np.asarray(X.todense(), dtype=np.float64)
        return np.asarray(X, dtype=np.float64)


def _run_test(
    test: Literal["mwu", "ttest"],
    target_matrix,
    control_matrix,
    control_idx_cache=None,
):
    """Dispatch to the requested statistical test."""
    if test == "mwu":
        if control_idx_cache is not None:
            return mwu(target_matrix, control_idx_cache)
        return mwu(target_matrix, control_matrix)
    # t-test requires dense
    if issparse(target_matrix):
        target_dense = np.asarray(target_matrix.todense(), dtype=np.float64)
    else:
        target_dense = np.asarray(target_matrix, dtype=np.float64)
    if issparse(control_matrix):
        control_dense = np.asarray(control_matrix.todense(), dtype=np.float64)
    else:
        control_dense = np.asarray(control_matrix, dtype=np.float64)
    return welch_ttest(target_dense, control_dense)


def _compare(
    target_matrix,
    control_matrix,
    pointer_kind: PointerKind,
    test: Literal["mwu", "ttest"],
    target_sum: float,
    geometric_mean: bool,
    feature_names: np.ndarray,
    control_mean_cache: np.ndarray | None = None,
    control_idx_cache=None,
) -> pl.DataFrame:
    """Run the full stats pipeline for one comparison. Returns a DataFrame."""
    eps = 1e-9
    is_log1p = pointer_kind == PointerKind.SPARSE

    if is_log1p:
        normalize_log1p_sparse(target_matrix, target_sum)

    target_mean = pseudobulk(target_matrix, geometric_mean=geometric_mean, is_log1p=is_log1p)

    if control_mean_cache is not None:
        ref_mean = control_mean_cache
    else:
        if is_log1p:
            normalize_log1p_sparse(control_matrix, target_sum)
        ref_mean = pseudobulk(control_matrix, geometric_mean=geometric_mean, is_log1p=is_log1p)

    fc = fold_change(target_mean + eps, ref_mean + eps)
    pc = percent_change(target_mean + eps, ref_mean + eps)

    result = _run_test(test, target_matrix, control_matrix, control_idx_cache)

    fdr = _benjamini_hochberg(result.pvalue)
    target_n = target_matrix.shape[0]
    control_n = (
        control_idx_cache.n_rows if control_idx_cache is not None else control_matrix.shape[0]
    )

    return pl.DataFrame(
        {
            "feature": feature_names.tolist(),
            "target_mean": target_mean,
            "ref_mean": ref_mean,
            "target_n": np.full(len(feature_names), target_n, dtype=np.int64),
            "ref_n": np.full(len(feature_names), control_n, dtype=np.int64),
            "fold_change": fc,
            "percent_change": pc,
            "p_value": result.pvalue,
            "statistic": result.statistic,
            "fdr": fdr,
        }
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dex(
    atlas,
    *,
    groupby: str,
    target: list[str],
    control: str,
    feature_space: str,
    test: Literal["mwu", "ttest"],
    target_sum: float = 1e4,
    threads: int = 0,
    geometric_mean: bool = True,
    max_records: int | None = None,
) -> pl.DataFrame:
    """Differential expression between target and control groups.

    Follows the scanpy ``rank_genes_groups`` pattern: specify a grouping
    column, which groups to test, and which group is the reference.

    Parameters
    ----------
    atlas : RaggedAtlas
        A checked-out lancell atlas.
    groupby : str
        Obs column that defines the groups (e.g. ``"tissue"``).
    target : list[str]
        Group values to test (e.g. ``["liver", "blood"]``). Each group
        is compared independently against *control*.
    control : str
        Reference group value (e.g. ``"brain"``).
    feature_space : str
        Feature space name (e.g. ``"gene_expression"``).
    test : {"mwu", "ttest"}
        Statistical test to use. ``"mwu"`` runs Mann-Whitney U;
        ``"ttest"`` runs Welch's t-test.
    target_sum : float
        Library-size normalization target (sparse path only).
    threads : int
        Number of numba threads (0 = all available).
    geometric_mean : bool
        Use geometric mean for pseudobulk (default True).
    max_records : int, optional
        Cap on cells loaded per group (target and control alike).

    Returns
    -------
    pl.DataFrame
        Columns: feature, target_mean, ref_mean, target_n, ref_n,
        fold_change, percent_change, p_value, statistic, fdr, target.
    """
    t_total = time.perf_counter()
    _set_numba_threadpool(threads)

    spec = get_spec(feature_space)
    pointer_kind = spec.pointer_kind

    # -- Load all groups in parallel ----------------------------------------
    with ThreadPoolExecutor(max_workers=len(target) + 1) as pool:
        fut_control = pool.submit(
            _load_adata,
            atlas,
            _group_where(groupby, control),
            feature_space,
            max_records,
        )
        fut_targets = {
            label: pool.submit(
                _load_adata,
                atlas,
                _group_where(groupby, label),
                feature_space,
                max_records,
            )
            for label in target
        }
        control_adata = fut_control.result()
        target_adatas = {label: fut.result() for label, fut in fut_targets.items()}

    assert control_adata.n_obs > 0, f"No control cells found for {groupby}={control!r}"
    if test == "ttest":
        assert control_adata.n_obs >= 2, "Welch's t-test requires at least 2 control cells"

    # -- Filter empty / too-small target groups -----------------------------
    valid_targets: dict[str, ad.AnnData] = {}
    for label, adata in target_adatas.items():
        if adata.n_obs == 0:
            log.warning("[dex] skipping group %r: no cells found", label)
        elif test == "ttest" and adata.n_obs < 2:
            log.warning("[dex] skipping group %r: only %d cell(s)", label, adata.n_obs)
        else:
            valid_targets[label] = adata

    if not valid_targets:
        return pl.DataFrame(schema={**_EMPTY_DEX_SCHEMA, "target": pl.String})

    # -- Align features across all groups -----------------------------------
    common = control_adata.var.index
    for adata in valid_targets.values():
        common = common.intersection(adata.var.index)
    if len(common) == 0:
        raise ValueError(
            "Target and control share no features. "
            "Ensure all groups have overlapping feature spaces."
        )
    feature_names = np.asarray(common)
    control_adata = control_adata[:, common]
    valid_targets = {k: v[:, common] for k, v in valid_targets.items()}

    # -- Prepare control once -----------------------------------------------
    control_matrix = _extract_matrix(control_adata, pointer_kind)
    is_log1p = pointer_kind == PointerKind.SPARSE

    if is_log1p:
        normalize_log1p_sparse(control_matrix, target_sum)

    if test == "mwu" and pointer_kind == PointerKind.SPARSE:
        control_idx = sparse_column_index(control_matrix)
    else:
        control_idx = None

    control_mean = pseudobulk(control_matrix, geometric_mean=geometric_mean, is_log1p=is_log1p)

    # -- Run comparisons ----------------------------------------------------
    all_results: list[pl.DataFrame] = []
    for label in sorted(valid_targets):
        target_matrix = _extract_matrix(valid_targets[label], pointer_kind)

        df = _compare(
            target_matrix,
            control_matrix,
            pointer_kind,
            test,
            target_sum,
            geometric_mean,
            feature_names,
            control_mean_cache=control_mean,
            control_idx_cache=control_idx,
        )
        all_results.append(df.with_columns(pl.lit(label).alias("target")))

    log.info(
        "[dex] done: %d groups, %d features — %.2fs",
        len(all_results),
        len(feature_names),
        time.perf_counter() - t_total,
    )
    return pl.concat(all_results)
