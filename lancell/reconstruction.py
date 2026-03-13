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

from lancell.atlas import PointerFieldInfo
from lancell.batch_array import BatchAsyncArray
from lancell.group_specs import ZarrGroupSpec
from lancell.protocols import Reconstructor

if TYPE_CHECKING:
    from lancell.atlas import RaggedAtlas

# Re-export for downstream convenience
__all__ = [
    "Reconstructor",
    "SparseCSRReconstructor",
    "DenseReconstructor",
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

    Adds internal columns ``_zg``, ``_start``, ``_end``.
    """
    col = pf.field_name
    struct_df = cells_pl[col].struct.unnest()
    cells_pl = cells_pl.with_columns(
        struct_df["zarr_group"].alias("_zg"),
        struct_df["start"].alias("_start"),
        struct_df["end"].alias("_end"),
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
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], int]:
    """Load remaps for groups, build joined feature space.

    Returns (group_remaps, joined_globals, group_remap_to_joined, n_features).
    """
    group_remaps: dict[str, np.ndarray] = {}
    if spec.has_var_df:
        for zg in groups:
            group_remaps[zg] = atlas._get_remap(zg, spec.feature_space)

    if group_remaps:
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

    joined_globals = functools.reduce(reduce_fn, remaps.values()).astype(np.int32)

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
    ) -> ad.AnnData:
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
            atlas, groups, spec, feature_join
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
            idx_reader = atlas._get_batch_reader(zg, index_array_name)
            lyr_readers = [atlas._get_batch_reader(zg, f"layers/{ln}") for ln in layers_to_read]
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
            group_data, all_results, strict=False
        ):
            flat_indices, lengths = index_result
            n_cells_group = len(group_cells)

            # Remap local indices -> joined positions
            if zg in group_remap_to_joined:
                joined_remap = group_remap_to_joined[zg]
                joined_indices = joined_remap[flat_indices.astype(np.intp)]
            else:
                joined_indices = flat_indices.astype(np.int32)

            # For intersection, filter out features not in the joined space
            if feature_join == "intersection" and zg in group_remap_to_joined:
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
            for ln, (flat_values, _) in zip(layers_to_read, layer_results, strict=False):
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
    ) -> ad.AnnData:
        cells_pl, groups = _prepare_dense_cells(cells_pl, pf)
        if not groups:
            return ad.AnnData()

        _, joined_globals, group_remap_to_joined, n_features = _load_remaps_and_features(
            atlas, groups, spec, feature_join
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
            readers = [atlas._get_batch_reader(zg, an) for an in array_names]
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
            group_data, all_results, strict=False
        ):
            n_cells_group = group_cells.height

            for out_key, (flat_data, _) in zip(output_keys, group_results, strict=False):
                n_local_features = flat_data.shape[0] // n_cells_group
                local_data = flat_data.reshape(n_cells_group, n_local_features)

                if zg in group_remap_to_joined:
                    joined_cols = group_remap_to_joined[zg]
                    if feature_join == "intersection":
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
