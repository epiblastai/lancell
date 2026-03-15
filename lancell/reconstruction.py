"""Reconstruction helpers for building AnnData from atlas query results."""

import asyncio
import functools
from typing import TYPE_CHECKING, Literal

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
from zarr.core.sync import sync

from lancell.batch_array import BatchAsyncArray
from lancell.group_specs import ZarrGroupSpec
from lancell.obs_alignment import PointerFieldInfo
from lancell.protocols import Reconstructor

if TYPE_CHECKING:
    from lancell.atlas import RaggedAtlas

# Re-export for downstream convenience
__all__ = [
    "Reconstructor",
    "SparseCSRReconstructor",
    "DenseReconstructor",
    "FeatureCSCReconstructor",
    "_get_pointer_columns",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare_sparse_cells(
    cells_pl: pl.DataFrame,
    pf: PointerFieldInfo,
) -> tuple[pl.DataFrame, list[str]]:
    """Unnest sparse pointer struct, filter empty, return (filtered_df, unique_groups).

    Adds internal columns ``_zg``, ``_start``, ``_end``, ``_zarr_row``.
    """
    col = pf.field_name
    struct_df = cells_pl[col].struct.unnest()
    cells_pl = cells_pl.with_columns(
        struct_df["zarr_group"].alias("_zg"),
        struct_df["start"].alias("_start"),
        struct_df["end"].alias("_end"),
        struct_df["zarr_row"].alias("_zarr_row"),
    )
    cells_pl = cells_pl.filter(pl.col("_zg") != "")
    groups = cells_pl["_zg"].unique().to_list() if not cells_pl.is_empty() else []
    return cells_pl, groups


def _prepare_dense_cells(
    cells_pl: pl.DataFrame,
    pf: PointerFieldInfo,
) -> tuple[pl.DataFrame, list[str]]:
    """Unnest dense pointer struct, filter empty, return (filtered_df, unique_groups).

    Adds internal columns ``_zg``, ``_pos``.
    """
    col = pf.field_name
    struct_df = cells_pl[col].struct.unnest()
    cells_pl = cells_pl.with_columns(
        struct_df["zarr_group"].alias("_zg"),
        struct_df["position"].alias("_pos"),
    )
    cells_pl = cells_pl.filter(pl.col("_zg") != "")
    groups = cells_pl["_zg"].unique().to_list() if not cells_pl.is_empty() else []
    return cells_pl, groups


def _load_remaps_and_features(
    atlas: "RaggedAtlas",
    groups: list[str],
    spec: ZarrGroupSpec,
    feature_join: Literal["union", "intersection"] = "union",
    wanted_globals: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], int]:
    """Load remaps for groups, build joined feature space.

    When *wanted_globals* is provided, skip the union/intersection step and
    use the requested global indices directly, applying intersection-style
    masking for each group.

    Returns (group_remaps, joined_globals, group_remap_to_joined, n_features).
    """
    group_remaps: dict[str, np.ndarray] = {}
    if spec.has_var_df:
        for zg in groups:
            group_remaps[zg] = atlas._get_remap(zg, spec.feature_space)

    if wanted_globals is not None:
        joined_globals = wanted_globals
        group_remap_to_joined = {
            zg: _apply_wanted_globals_remap(remap, wanted_globals)
            for zg, remap in group_remaps.items()
        }
        n_features = len(wanted_globals)
    elif group_remaps:
        joined_globals, group_remap_to_joined = _build_feature_space(group_remaps, feature_join)
        n_features = len(joined_globals)
    else:
        joined_globals = np.array([], dtype=np.int32)
        group_remap_to_joined = {}
        n_features = 0

    return group_remaps, joined_globals, group_remap_to_joined, n_features


def _build_feature_space(
    remaps: dict[str, np.ndarray],
    join: Literal["union", "intersection"] = "union",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute union or intersection of global indices and per-group local-to-joined mappings.

    Parameters
    ----------
    remaps:
        ``{zarr_group: remap_array}`` where ``remap[local_i] = global_index``.
    join:
        ``"union"`` to include all features across groups, ``"intersection"``
        to include only features present in every group.

    Returns
    -------
    (joined_globals, group_remap_to_joined)
        ``joined_globals``: sorted array of unique global indices in the joined space.
        ``group_remap_to_joined[zg]``: array where ``arr[local_i]`` is the
        column position in the joined-space matrix. For intersection mode,
        local features not in the joined space are mapped to ``-1``.
    """
    if join == "union":
        reduce_fn = np.union1d
    elif join == "intersection":
        reduce_fn = np.intersect1d
    else:
        raise ValueError(f"feature_join must be 'union' or 'intersection', got '{join}'")

    # functools.reduce with a single-element iterable returns that element unchanged
    # (reduce_fn is never called), so the result may be unsorted. np.unique ensures
    # sorted unique output in all cases, which searchsorted requires.
    joined_globals = np.unique(functools.reduce(reduce_fn, remaps.values())).astype(np.int32)

    group_remap_to_joined: dict[str, np.ndarray] = {}
    for group, remap in remaps.items():
        positions = np.searchsorted(joined_globals, remap).astype(np.int32)
        if join == "intersection":
            # searchsorted can return out-of-bounds or wrong-match indices;
            # mark features not in the intersection as -1
            mask = np.isin(remap, joined_globals)
            positions[~mask] = -1
        group_remap_to_joined[group] = positions

    return joined_globals, group_remap_to_joined


def _apply_wanted_globals_remap(remap: np.ndarray, wanted_globals: np.ndarray) -> np.ndarray:
    """Map local feature indices to positions in wanted_globals; -1 if absent.

    Parameters
    ----------
    remap:
        Array where remap[local_i] = global_index.
    wanted_globals:
        Sorted int32 array of desired global indices.

    Returns
    -------
    np.ndarray
        int32 array; result[local_i] = position in wanted_globals, or -1.
    """
    positions = np.searchsorted(wanted_globals, remap).astype(np.int32)
    mask = np.isin(remap, wanted_globals)
    positions[~mask] = -1
    return positions


def _build_obs_df(cells_pl: pl.DataFrame) -> pd.DataFrame:
    """Build an obs DataFrame from query results, excluding pointer/internal columns."""
    # Drop struct columns (pointer fields) and internal helper columns
    keep_cols = [
        c for c in cells_pl.columns if cells_pl[c].dtype != pl.Struct and not c.startswith("_")
    ]
    obs = cells_pl.select(keep_cols).to_pandas()
    if "uid" in obs.columns:
        obs = obs.set_index("uid")
    return obs


def _get_pointer_columns(cells_pl: pl.DataFrame) -> list[str]:
    """Return the names of zarr pointer struct columns.

    Inverse of :func:`_build_obs_only_anndata` which strips pointer columns
    and keeps only obs. This is used to ensure pointer columns are always
    loaded from the database even when a user-level ``select`` restricts
    the returned metadata columns.
    """
    return [c for c in cells_pl.columns if cells_pl[c].dtype == pl.Struct]


def _build_obs_only_anndata(cells_pl: pl.DataFrame) -> ad.AnnData:
    """Build an AnnData with only obs, no X."""
    keep_cols = [c for c in cells_pl.columns if cells_pl[c].dtype != pl.Struct]
    obs = cells_pl.select(keep_cols).to_pandas()
    if "uid" in obs.columns:
        obs = obs.set_index("uid")
    return ad.AnnData(obs=obs)


def _build_var(
    atlas: "RaggedAtlas",
    feature_space: str,
    joined_globals: np.ndarray,
) -> pd.DataFrame:
    """Build a var DataFrame from the feature registry."""
    if feature_space not in atlas._registry_tables:
        raise ValueError(
            f"No registry table for feature space '{feature_space}'. "
            f"Available: {sorted(atlas._registry_tables.keys())}"
        )
    if len(joined_globals) == 0:
        return pd.DataFrame(index=pd.RangeIndex(0))

    registry_table = atlas._registry_tables[feature_space]
    registry_df = registry_table.search().to_polars()

    # Filter to joined globals
    registry_df = registry_df.filter(pl.col("global_index").is_in(joined_globals.tolist())).sort(
        "global_index"
    )

    var = registry_df.to_pandas()
    # uid is mandatory via FeatureBaseSchema
    var = var.set_index("uid")
    return var


# ---------------------------------------------------------------------------
# Async read helpers
# ---------------------------------------------------------------------------


async def _read_sparse_group(
    index_reader: "BatchAsyncArray",
    layer_readers: "list[BatchAsyncArray]",
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Read index array and layer arrays concurrently for one zarr group."""
    coros = [index_reader.read_ranges(starts, ends)]
    coros.extend(r.read_ranges(starts, ends) for r in layer_readers)

    results = await asyncio.gather(*coros)
    return results[0], list(results[1:])


async def _read_dense_group(
    readers: "list[BatchAsyncArray]",
    starts: np.ndarray,
    ends: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read all dense arrays concurrently for one zarr group."""
    coros = [r.read_ranges(starts, ends) for r in readers]
    return list(await asyncio.gather(*coros))


# ---------------------------------------------------------------------------
# Built-in reconstructor implementations
# ---------------------------------------------------------------------------


class SparseCSRReconstructor:
    """Reconstruct sparse CSR data (e.g. gene expression) across zarr groups."""

    def as_anndata(
        self,
        atlas: "RaggedAtlas",
        cells_pl: pl.DataFrame,
        pf: PointerFieldInfo,
        spec: ZarrGroupSpec,
        layer_overrides: list[str] | None = None,
        feature_join: Literal["union", "intersection"] = "union",
        wanted_globals: np.ndarray | None = None,
    ) -> ad.AnnData:
        if wanted_globals is not None:
            if feature_join != "union":
                raise ValueError(
                    "feature_join has no effect when wanted_globals is provided; "
                    "the feature space is pinned to the requested globals."
                )
            return FeatureCSCReconstructor().as_anndata(
                atlas, cells_pl, pf, spec, layer_overrides, feature_join, wanted_globals
            )

        # Determine index array name from spec's required_arrays
        if len(spec.required_arrays) != 1:
            raise NotImplementedError(
                f"Sparse reconstruction for feature space '{pf.feature_space}' "
                f"is not yet supported (requires {len(spec.required_arrays)} "
                f"primary arrays: {[a.array_name for a in spec.required_arrays]})"
            )
        index_array_name = spec.required_arrays[0].array_name

        cells_pl, groups = _prepare_sparse_cells(cells_pl, pf)
        if not groups:
            return ad.AnnData()

        _, joined_globals, group_remap_to_joined, n_features = _load_remaps_and_features(
            atlas, groups, spec, feature_join, wanted_globals
        )

        # Determine which layers to read
        if layer_overrides is not None:
            layers_to_read = layer_overrides
        else:
            layers_to_read = list(spec.required_layers)
            if not layers_to_read:
                raise ValueError(
                    f"No layers specified and spec for '{pf.feature_space}' has no required layers"
                )

        # Prepare per-group cell data and pre-create readers (must happen
        # outside the async context to avoid nested sync() calls)
        group_data: list[
            tuple[str, pl.DataFrame, np.ndarray, np.ndarray, BatchAsyncArray, list[BatchAsyncArray]]
        ] = []
        for zg in groups:
            group_cells = cells_pl.filter(pl.col("_zg") == zg)
            starts = group_cells["_start"].to_numpy().astype(np.int64)
            ends = group_cells["_end"].to_numpy().astype(np.int64)
            gr = atlas._get_group_reader(zg, pf.feature_space)
            idx_reader = gr.get_array_reader(index_array_name)
            lyr_readers = [gr.get_array_reader(f"csr/layers/{ln}") for ln in layers_to_read]
            group_data.append((zg, group_cells, starts, ends, idx_reader, lyr_readers))

        # Dispatch all groups concurrently
        async def _read_all():
            return await asyncio.gather(
                *[
                    _read_sparse_group(idx_reader, lyr_readers, starts, ends)
                    for _, _, starts, ends, idx_reader, lyr_readers in group_data
                ]
            )

        all_results = sync(_read_all())

        # Assemble CSRs
        all_csrs: dict[str, list[sp.csr_matrix]] = {ln: [] for ln in layers_to_read}
        obs_parts: list[pl.DataFrame] = []

        for (zg, group_cells, _, _, _, _), (index_result, layer_results) in zip(
            group_data, all_results, strict=True
        ):
            flat_indices, lengths = index_result
            n_cells_group = len(group_cells)

            # Remap local indices -> joined positions
            if zg in group_remap_to_joined:
                joined_remap = group_remap_to_joined[zg]
                joined_indices = joined_remap[flat_indices.astype(np.intp)]
            else:
                joined_indices = flat_indices.astype(np.int32)

            # For intersection or feature filter, filter out features not in the joined space
            if (
                feature_join == "intersection" or wanted_globals is not None
            ) and zg in group_remap_to_joined:
                keep_mask = joined_indices >= 0
                joined_indices = joined_indices[keep_mask]
                # Recompute per-cell lengths after filtering
                cell_ids = np.repeat(np.arange(n_cells_group), lengths)
                lengths = np.bincount(cell_ids[keep_mask], minlength=n_cells_group).astype(np.int64)
            else:
                keep_mask = None

            # Build indptr from lengths
            indptr = np.zeros(n_cells_group + 1, dtype=np.int64)
            np.cumsum(lengths, out=indptr[1:])

            # Build CSR for each layer
            for ln, (flat_values, _) in zip(layers_to_read, layer_results, strict=True):
                if keep_mask is not None:
                    flat_values = flat_values[keep_mask]
                csr = sp.csr_matrix(
                    (flat_values, joined_indices, indptr),
                    shape=(n_cells_group, n_features),
                )
                all_csrs[ln].append(csr)

            obs_parts.append(group_cells)

        # Stack CSRs
        stacked: dict[str, sp.csr_matrix] = {}
        for ln, csr_list in all_csrs.items():
            if csr_list:
                stacked[ln] = sp.vstack(csr_list, format="csr")

        # Build obs
        obs_pl = pl.concat(obs_parts, how="diagonal_relaxed")
        obs = _build_obs_df(obs_pl)

        # Build var from registry
        var = _build_var(atlas, pf.feature_space, joined_globals)

        # First layer becomes X, rest go to layers
        first_layer = layers_to_read[0]
        X = stacked.get(first_layer)
        extra_layers = {ln: stacked[ln] for ln in layers_to_read[1:] if ln in stacked}

        return ad.AnnData(X=X, obs=obs, var=var, layers=extra_layers if extra_layers else None)


class DenseReconstructor:
    """Reconstruct dense data (e.g. protein abundance) across zarr groups."""

    def as_anndata(
        self,
        atlas: "RaggedAtlas",
        cells_pl: pl.DataFrame,
        pf: PointerFieldInfo,
        spec: ZarrGroupSpec,
        layer_overrides: list[str] | None = None,
        feature_join: Literal["union", "intersection"] = "union",
        wanted_globals: np.ndarray | None = None,
    ) -> ad.AnnData:
        cells_pl, groups = _prepare_dense_cells(cells_pl, pf)
        if not groups:
            return ad.AnnData()

        _, joined_globals, group_remap_to_joined, n_features = _load_remaps_and_features(
            atlas, groups, spec, feature_join, wanted_globals
        )

        # Determine which layers to read
        if layer_overrides is not None:
            layers_to_read = layer_overrides
        else:
            layers_to_read = list(spec.required_layers)

        # Resolve array names: "layers/{ln}" for layered specs, "data" for plain
        array_names = [f"layers/{ln}" for ln in layers_to_read] if layers_to_read else ["data"]
        output_keys = layers_to_read if layers_to_read else ["data"]

        n_total_cells = cells_pl.height
        all_layers: dict[str, np.ndarray] = {
            k: np.zeros((n_total_cells, n_features), dtype=np.float32) for k in output_keys
        }

        # Prepare per-group cell data, pre-create readers, and compute offsets
        group_data: list[
            tuple[str, pl.DataFrame, np.ndarray, np.ndarray, int, list[BatchAsyncArray]]
        ] = []
        offset = 0
        for zg in groups:
            group_cells = cells_pl.filter(pl.col("_zg") == zg)
            positions = group_cells["_pos"].to_numpy().astype(np.int64)
            starts = positions
            ends = positions + 1
            gr = atlas._get_group_reader(zg, pf.feature_space)
            readers = [gr.get_array_reader(an) for an in array_names]
            group_data.append((zg, group_cells, starts, ends, offset, readers))
            offset += len(positions)

        # Dispatch all groups concurrently
        async def _read_all():
            return await asyncio.gather(
                *[
                    _read_dense_group(readers, starts, ends)
                    for _, _, starts, ends, _, readers in group_data
                ]
            )

        all_results = sync(_read_all())

        # Assemble into pre-allocated arrays
        obs_parts: list[pl.DataFrame] = []

        for (zg, group_cells, _, _, offset, _), group_results in zip(
            group_data, all_results, strict=True
        ):
            n_cells_group = group_cells.height

            for out_key, (flat_data, _) in zip(output_keys, group_results, strict=True):
                n_local_features = flat_data.shape[0] // n_cells_group
                local_data = flat_data.reshape(n_cells_group, n_local_features)

                if zg in group_remap_to_joined:
                    joined_cols = group_remap_to_joined[zg]
                    if feature_join == "intersection" or wanted_globals is not None:
                        valid = joined_cols >= 0
                        all_layers[out_key][offset : offset + n_cells_group][
                            :, joined_cols[valid]
                        ] = local_data[:, valid]
                    else:
                        all_layers[out_key][offset : offset + n_cells_group][:, joined_cols] = (
                            local_data
                        )
                else:
                    all_layers[out_key][offset : offset + n_cells_group, :n_local_features] = (
                        local_data
                    )

            obs_parts.append(group_cells)

        obs_pl = pl.concat(obs_parts, how="diagonal_relaxed")
        obs = _build_obs_df(obs_pl)
        var = _build_var(atlas, pf.feature_space, joined_globals)

        # First layer/array -> X, rest -> adata.layers
        first_key = output_keys[0]
        X = all_layers[first_key]
        extra_layers = {k: all_layers[k] for k in output_keys[1:]}

        return ad.AnnData(X=X, obs=obs, var=var, layers=extra_layers if extra_layers else None)


class FeatureCSCReconstructor:
    """Reconstruct sparse data using CSC for groups that have it, CSR otherwise.

    Intended for feature-filtered queries (few features, many cells).
    When a group has CSC data (populated ``csc_start``/``csc_end`` in var.parquet),
    reads O(nnz for wanted features) instead of O(nnz per cell × n_cells).
    Falls back to CSR for groups that have not been post-processed by ``add_csc``.
    """

    def as_anndata(
        self,
        atlas: "RaggedAtlas",
        cells_pl: pl.DataFrame,
        pf: PointerFieldInfo,
        spec: ZarrGroupSpec,
        layer_overrides: list[str] | None = None,
        feature_join: Literal["union", "intersection"] = "union",
        wanted_globals: np.ndarray | None = None,
    ) -> ad.AnnData:
        if wanted_globals is None:
            return SparseCSRReconstructor().as_anndata(
                atlas, cells_pl, pf, spec, layer_overrides, feature_join, wanted_globals
            )

        if feature_join != "union":
            raise ValueError(
                "feature_join has no effect when wanted_globals is provided; "
                "the feature space is pinned to the requested globals."
            )

        if len(spec.required_arrays) != 1:
            raise NotImplementedError(
                f"CSC reconstruction for '{pf.feature_space}' requires exactly one primary array"
            )
        csr_index_name = spec.required_arrays[0].array_name  # e.g. "csr/indices"

        cells_pl, groups = _prepare_sparse_cells(cells_pl, pf)
        if not groups:
            return ad.AnnData()

        n_features = len(wanted_globals)
        layers_to_read = (
            layer_overrides if layer_overrides is not None else list(spec.required_layers)
        )
        if not layers_to_read:
            raise ValueError(f"No layers specified for feature space '{pf.feature_space}'")

        _, _, group_remap_to_joined, _ = _load_remaps_and_features(
            atlas, groups, spec, "intersection", wanted_globals
        )

        # Per-group preparation: one read coroutine per group (CSC or CSR)
        group_info: list[dict] = []
        read_coroutines = []

        for zg in groups:
            group_cells = cells_pl.filter(pl.col("_zg") == zg)
            gr = atlas._get_group_reader(zg, spec.feature_space)

            if gr.has_csc:
                var_df = gr.var_df
                remap = gr.get_remap()

                # Build global_index -> local_index inverse map
                remap_inv = np.full(
                    int(remap.max()) + 1 if len(remap) > 0 else 0, -1, dtype=np.int64
                )
                for local_i, global_i in enumerate(remap):
                    remap_inv[int(global_i)] = local_i

                # Find CSC ranges for each wanted global feature present in this group
                csc_starts_list: list[int] = []
                csc_ends_list: list[int] = []
                feat_col_indices: list[int] = []

                for col_idx, global_f in enumerate(wanted_globals):
                    gf = int(global_f)
                    if gf >= len(remap_inv) or remap_inv[gf] == -1:
                        continue
                    local_f = int(remap_inv[gf])
                    row = var_df.row(local_f, named=True)
                    cs = row.get("csc_start")
                    ce = row.get("csc_end")
                    if cs is None or ce is None:
                        continue
                    csc_starts_list.append(int(cs))
                    csc_ends_list.append(int(ce))
                    feat_col_indices.append(col_idx)

                # Build zarr_row -> rank-within-group lookup (vectorized)
                zarr_rows_arr = group_cells["_zarr_row"].to_numpy().astype(np.int64)
                max_zr = int(zarr_rows_arr.max()) + 1 if len(zarr_rows_arr) > 0 else 0
                zr_to_rank = np.full(max_zr, -1, dtype=np.int64)
                for rank, zr in enumerate(zarr_rows_arr):
                    zr_to_rank[int(zr)] = rank

                starts = np.array(csc_starts_list, dtype=np.int64)
                ends = np.array(csc_ends_list, dtype=np.int64)
                idx_reader = gr.get_array_reader("csc/indices")
                lyr_readers = [gr.get_array_reader(f"csc/layers/{ln}") for ln in layers_to_read]
                read_coroutines.append(_read_sparse_group(idx_reader, lyr_readers, starts, ends))
                group_info.append(
                    {
                        "mode": "csc",
                        "group_cells": group_cells,
                        "feat_col_indices": feat_col_indices,
                        "zr_to_rank": zr_to_rank,
                    }
                )

            else:
                starts = group_cells["_start"].to_numpy().astype(np.int64)
                ends = group_cells["_end"].to_numpy().astype(np.int64)
                idx_reader = gr.get_array_reader(csr_index_name)
                lyr_readers = [gr.get_array_reader(f"csr/layers/{ln}") for ln in layers_to_read]
                read_coroutines.append(_read_sparse_group(idx_reader, lyr_readers, starts, ends))
                group_info.append(
                    {
                        "mode": "csr",
                        "group_cells": group_cells,
                        "zg": zg,
                    }
                )

        async def _read_all():
            return await asyncio.gather(*read_coroutines)

        all_results = sync(_read_all())

        # Assemble COO entries across all groups
        rows_parts: list[np.ndarray] = []
        cols_parts: list[np.ndarray] = []
        layer_vals_parts: dict[str, list[np.ndarray]] = {ln: [] for ln in layers_to_read}
        obs_parts: list[pl.DataFrame] = []
        cell_offset = 0

        for info, (idx_result, layer_results) in zip(group_info, all_results, strict=True):
            group_cells = info["group_cells"]
            n_cells_group = group_cells.height
            flat_indices, lengths = idx_result

            if info["mode"] == "csc":
                feat_col_indices = info["feat_col_indices"]
                zr_to_rank = info["zr_to_rank"]

                offset = 0
                for length, col_idx in zip(lengths, feat_col_indices, strict=True):
                    if length == 0:
                        offset += length
                        continue
                    zr_seg = flat_indices[offset : offset + length].astype(np.int64)
                    # Two-step: numpy & doesn't short-circuit, so indexing zr_to_rank
                    # with out-of-bounds zr_seg values would raise even if the bounds
                    # mask would have excluded them.
                    in_bounds = zr_seg < len(zr_to_rank)
                    valid_mask = in_bounds.copy()
                    valid_mask[in_bounds] = zr_to_rank[zr_seg[in_bounds]] >= 0
                    kept_zr = zr_seg[valid_mask]
                    if len(kept_zr) > 0:
                        ranks = zr_to_rank[kept_zr]
                        rows_parts.append((cell_offset + ranks).astype(np.int64))
                        cols_parts.append(np.full(len(kept_zr), col_idx, dtype=np.int64))
                        for ln_i, ln in enumerate(layers_to_read):
                            flat_vals, _ = layer_results[ln_i]
                            val_seg = flat_vals[offset : offset + length]
                            layer_vals_parts[ln].append(val_seg[valid_mask])
                    offset += length

            else:
                zg = info["zg"]
                joined_remap = group_remap_to_joined.get(zg)
                if joined_remap is not None:
                    joined_indices = joined_remap[flat_indices.astype(np.intp)]
                    keep_mask = joined_indices >= 0
                    joined_indices_kept = joined_indices[keep_mask]
                else:
                    keep_mask = None
                    joined_indices_kept = flat_indices.astype(np.int64)

                if keep_mask is not None:
                    cell_ids = np.repeat(np.arange(n_cells_group, dtype=np.int64), lengths)
                    lengths_filtered = np.bincount(
                        cell_ids[keep_mask], minlength=n_cells_group
                    ).astype(np.int64)
                else:
                    lengths_filtered = lengths

                cell_local_ids = np.repeat(
                    np.arange(n_cells_group, dtype=np.int64), lengths_filtered
                )
                rows_parts.append(cell_offset + cell_local_ids)
                cols_parts.append(joined_indices_kept.astype(np.int64))
                for ln_i, ln in enumerate(layers_to_read):
                    flat_vals, _ = layer_results[ln_i]
                    layer_vals_parts[ln].append(
                        flat_vals[keep_mask] if keep_mask is not None else flat_vals
                    )

            obs_parts.append(group_cells)
            cell_offset += n_cells_group

        n_total_cells = cell_offset

        rows = np.concatenate(rows_parts) if rows_parts else np.array([], dtype=np.int64)
        cols = np.concatenate(cols_parts) if cols_parts else np.array([], dtype=np.int64)

        stacked: dict[str, sp.csr_matrix] = {}
        for ln in layers_to_read:
            vals_list = layer_vals_parts[ln]
            vals = np.concatenate(vals_list) if vals_list else np.array([], dtype=np.float32)
            stacked[ln] = sp.coo_matrix(
                (vals, (rows, cols)), shape=(n_total_cells, n_features)
            ).tocsr()

        obs_pl = pl.concat(obs_parts, how="diagonal_relaxed")
        obs = _build_obs_df(obs_pl)
        var = _build_var(atlas, pf.feature_space, wanted_globals)

        first_layer = layers_to_read[0]
        X = stacked.get(first_layer)
        extra_layers = {ln: stacked[ln] for ln in layers_to_read[1:] if ln in stacked}

        return ad.AnnData(X=X, obs=obs, var=var, layers=extra_layers if extra_layers else None)
