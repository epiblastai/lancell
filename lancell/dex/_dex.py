"""Core differential expression: unified sparse (MWU) + dense (Welch t-test)."""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

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
# Helpers carried over from original
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


def _build_result_df(
    feature_names: np.ndarray,
    target_mean: np.ndarray,
    ref_mean: np.ndarray,
    target_n: int,
    ref_n: int,
    fc: np.ndarray,
    pc: np.ndarray,
    statistics: np.ndarray,
    pvalues: np.ndarray,
    fdr: np.ndarray,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": feature_names.tolist(),
            "target_mean": target_mean,
            "ref_mean": ref_mean,
            "target_n": np.full(len(feature_names), target_n, dtype=np.int64),
            "ref_n": np.full(len(feature_names), ref_n, dtype=np.int64),
            "fold_change": fc,
            "percent_change": pc,
            "p_value": pvalues,
            "statistic": statistics,
            "fdr": fdr,
        }
    )


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
# New helpers for atlas-based data loading
# ---------------------------------------------------------------------------


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


def _align_features(
    target_adata: ad.AnnData,
    control_adata: ad.AnnData,
) -> tuple[ad.AnnData, ad.AnnData, np.ndarray]:
    """Intersect features so both AnnData objects have identical columns.

    Returns (target_adata, control_adata, feature_names) where feature_names
    are the UIDs of the shared features in order.
    """
    common = target_adata.var.index.intersection(control_adata.var.index)
    if len(common) == 0:
        raise ValueError(
            "Target and control share no features. "
            "Ensure both queries return cells with overlapping feature spaces."
        )
    target_adata = target_adata[:, common]
    control_adata = control_adata[:, common]
    feature_names = np.asarray(common)
    return target_adata, control_adata, feature_names


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


def _compare(
    target_matrix,
    control_matrix,
    pointer_kind: PointerKind,
    target_sum: float,
    geometric_mean: bool,
    feature_names: np.ndarray,
    control_mean_cache: np.ndarray | None = None,
    control_idx_cache=None,
) -> pl.DataFrame:
    """Run the full stats pipeline for one comparison. Returns a DataFrame."""
    eps = 1e-9

    if pointer_kind == PointerKind.SPARSE:
        normalize_log1p_sparse(target_matrix, target_sum)
        target_mean = pseudobulk(target_matrix, geometric_mean=geometric_mean, is_log1p=True)

        if control_mean_cache is not None:
            ref_mean = control_mean_cache
        else:
            normalize_log1p_sparse(control_matrix, target_sum)
            ref_mean = pseudobulk(control_matrix, geometric_mean=geometric_mean, is_log1p=True)

        fc = fold_change(target_mean + eps, ref_mean + eps)
        pc = percent_change(target_mean + eps, ref_mean + eps)

        if control_idx_cache is not None:
            result = mwu(target_matrix, control_idx_cache)
        else:
            result = mwu(target_matrix, control_matrix)
    else:
        target_mean = pseudobulk(target_matrix, geometric_mean=geometric_mean, is_log1p=False)

        if control_mean_cache is not None:
            ref_mean = control_mean_cache
        else:
            ref_mean = pseudobulk(control_matrix, geometric_mean=geometric_mean, is_log1p=False)

        fc = fold_change(target_mean + eps, ref_mean + eps)
        pc = percent_change(target_mean + eps, ref_mean + eps)

        result = welch_ttest(target_matrix, control_matrix)

    fdr = _benjamini_hochberg(result.pvalue)
    target_n = target_matrix.shape[0]
    control_n = (
        control_idx_cache.n_rows if control_idx_cache is not None else control_matrix.shape[0]
    )

    return _build_result_df(
        feature_names,
        target_mean,
        ref_mean,
        target_n,
        control_n,
        fc,
        pc,
        result.statistic,
        result.pvalue,
        fdr,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dex(
    atlas,
    *,
    target: str,
    control: str,
    feature_space: str,
    groupby: str | None = None,
    target_sum: float = 1e4,
    threads: int = 0,
    geometric_mean: bool = True,
    max_records: int | None = None,
) -> pl.DataFrame:
    """Differential expression between target and control cells.

    Dispatches automatically: MWU for sparse feature spaces (e.g.
    ``gene_expression``), Welch's t-test for dense (e.g.
    ``protein_abundance``).

    Parameters
    ----------
    atlas : RaggedAtlas
        A checked-out lancell atlas.
    target : str
        SQL WHERE clause selecting target (perturbation) cells.
    control : str
        SQL WHERE clause selecting control cells.
    feature_space : str
        Feature space name (e.g. ``"gene_expression"``).
    groupby : str, optional
        If set, split target cells by this obs column and run a
        separate comparison per group. Adds a ``"target"`` column
        to the result.
    target_sum : float
        Library-size normalization target (sparse path only).
    threads : int
        Number of numba threads (0 = all available).
    geometric_mean : bool
        Use geometric mean for pseudobulk (default True).
    max_records : int, optional
        Cap on cells per query (or per group when ``groupby`` is set).

    Returns
    -------
    pl.DataFrame
        Columns: feature, target_mean, ref_mean, target_n, ref_n,
        fold_change, percent_change, p_value, statistic, fdr.
        With ``groupby``, an additional ``target`` column is included.
    """
    t_total = time.perf_counter()
    _set_numba_threadpool(threads)

    spec = get_spec(feature_space)
    pointer_kind = spec.pointer_kind

    if groupby is None:
        return _dex_single(
            atlas,
            target,
            control,
            feature_space,
            pointer_kind,
            target_sum,
            geometric_mean,
            max_records,
            t_total,
        )
    else:
        return _dex_grouped(
            atlas,
            target,
            control,
            feature_space,
            pointer_kind,
            groupby,
            target_sum,
            geometric_mean,
            max_records,
            t_total,
        )


def _dex_single(
    atlas,
    target: str,
    control: str,
    feature_space: str,
    pointer_kind: PointerKind,
    target_sum: float,
    geometric_mean: bool,
    max_records: int | None,
    t_total: float,
) -> pl.DataFrame:
    """Single comparison: all target cells vs all control cells."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_target = pool.submit(_load_adata, atlas, target, feature_space, max_records)
        fut_control = pool.submit(_load_adata, atlas, control, feature_space, max_records)
        target_adata = fut_target.result()
        control_adata = fut_control.result()

    assert target_adata.n_obs > 0, "No target cells found for the given query"
    assert control_adata.n_obs > 0, "No control cells found for the given query"
    if pointer_kind == PointerKind.DENSE:
        assert target_adata.n_obs >= 2, "Welch's t-test requires at least 2 target cells"
        assert control_adata.n_obs >= 2, "Welch's t-test requires at least 2 control cells"

    target_adata, control_adata, feature_names = _align_features(target_adata, control_adata)

    target_matrix = _extract_matrix(target_adata, pointer_kind)
    control_matrix = _extract_matrix(control_adata, pointer_kind)

    df = _compare(
        target_matrix,
        control_matrix,
        pointer_kind,
        target_sum,
        geometric_mean,
        feature_names,
    )

    log.info(
        "[dex] done: target=%d cells, control=%d cells, %d features — %.2fs",
        target_adata.n_obs,
        control_adata.n_obs,
        len(feature_names),
        time.perf_counter() - t_total,
    )
    return df


def _dex_grouped(
    atlas,
    target: str,
    control: str,
    feature_space: str,
    pointer_kind: PointerKind,
    groupby: str,
    target_sum: float,
    geometric_mean: bool,
    max_records: int | None,
    t_total: float,
) -> pl.DataFrame:
    """Per-group comparison: split target by groupby column."""
    # Load control and all target cells
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_target = pool.submit(_load_adata, atlas, target, feature_space)
        fut_control = pool.submit(_load_adata, atlas, control, feature_space, max_records)
        target_adata = fut_target.result()
        control_adata = fut_control.result()

    assert target_adata.n_obs > 0, "No target cells found for the given query"
    assert control_adata.n_obs > 0, "No control cells found for the given query"

    target_adata, control_adata, feature_names = _align_features(target_adata, control_adata)

    # Prepare control once
    control_matrix = _extract_matrix(control_adata, pointer_kind)

    if pointer_kind == PointerKind.SPARSE:
        normalize_log1p_sparse(control_matrix, target_sum)
        control_idx = sparse_column_index(control_matrix)
        control_mean = pseudobulk(control_matrix, geometric_mean=geometric_mean, is_log1p=True)
    else:
        control_idx = None
        control_mean = pseudobulk(control_matrix, geometric_mean=geometric_mean, is_log1p=False)

    # Split target by groupby column
    assert groupby in target_adata.obs.columns, (
        f"groupby column {groupby!r} not found in obs. Available: {list(target_adata.obs.columns)}"
    )

    all_results: list[pl.DataFrame] = []
    groups = target_adata.obs[groupby].unique()

    for label in sorted(groups):
        mask = target_adata.obs[groupby] == label
        group_adata = target_adata[mask]

        if max_records is not None:
            group_adata = group_adata[:max_records]

        if group_adata.n_obs == 0:
            continue
        if pointer_kind == PointerKind.DENSE and group_adata.n_obs < 2:
            log.warning("[dex] skipping group %r: only %d cell(s)", label, group_adata.n_obs)
            continue

        group_matrix = _extract_matrix(group_adata, pointer_kind)

        df = _compare(
            group_matrix,
            control_matrix,
            pointer_kind,
            target_sum,
            geometric_mean,
            feature_names,
            control_mean_cache=control_mean,
            control_idx_cache=control_idx,
        )
        all_results.append(df.with_columns(pl.lit(str(label)).alias("target")))

    if not all_results:
        return pl.DataFrame(schema={**_EMPTY_DEX_SCHEMA, "target": pl.String})

    log.info(
        "[dex] done: %d groups, %d features — %.2fs",
        len(all_results),
        len(feature_names),
        time.perf_counter() - t_total,
    )
    return pl.concat(all_results)
