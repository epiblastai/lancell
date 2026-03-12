"""Reconstruction helpers for building AnnData from atlas query results."""

import functools
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp

from lancell.atlas import PointerFieldInfo
from lancell.group_specs import Reconstructor, ZarrGroupSpec

if TYPE_CHECKING:
    from lancell.atlas import RaggedAtlas

# Re-export for downstream convenience
__all__ = [
    "Reconstructor",
    "SparseCSRReconstructor",
    "DenseReconstructor",
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


def _load_remaps_and_union(
    atlas: "RaggedAtlas",
    groups: list[str],
    spec: ZarrGroupSpec,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], int]:
    """Load remaps for groups, build union feature space.

    Returns (group_remaps, union_globals, group_remap_to_union, n_features).
    """
    group_remaps: dict[str, np.ndarray] = {}
    if spec.has_var_df:
        for zg in groups:
            group_remaps[zg] = atlas._get_remap(zg, spec.feature_space)

    if group_remaps:
        union_globals, group_remap_to_union = _build_union_feature_space(group_remaps)
        n_features = len(union_globals)
    else:
        union_globals = np.array([], dtype=np.int32)
        group_remap_to_union = {}
        n_features = 0

    return group_remaps, union_globals, group_remap_to_union, n_features


def _build_union_feature_space(
    remaps: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute union of global indices and per-group local-to-union mappings.

    Parameters
    ----------
    remaps:
        ``{zarr_group: remap_array}`` where ``remap[local_i] = global_index``.

    Returns
    -------
    (union_globals, group_remap_to_union)
        ``union_globals``: sorted array of unique global indices in the union.
        ``group_remap_to_union[zg]``: array where ``arr[local_i]`` is the
        column position in the union-space matrix.
    """
    union_globals = functools.reduce(np.union1d, remaps.values()).astype(np.int32)

    group_remap_to_union = {
        group: np.searchsorted(union_globals, remap).astype(np.int32)
        for group, remap in remaps.items()
    }
    return union_globals, group_remap_to_union


def _build_obs_df(cells_pl: pl.DataFrame) -> pd.DataFrame:
    """Build an obs DataFrame from query results, excluding pointer/internal columns."""
    # Drop struct columns (pointer fields) and internal helper columns
    keep_cols = [
        c for c in cells_pl.columns
        if cells_pl[c].dtype != pl.Struct and not c.startswith("_")
    ]
    obs = cells_pl.select(keep_cols).to_pandas()
    if "uid" in obs.columns:
        obs = obs.set_index("uid")
    return obs


def _build_obs_only_anndata(cells_pl: pl.DataFrame) -> ad.AnnData:
    """Build an AnnData with only obs, no X."""
    keep_cols = [
        c for c in cells_pl.columns
        if cells_pl[c].dtype != pl.Struct
    ]
    obs = cells_pl.select(keep_cols).to_pandas()
    if "uid" in obs.columns:
        obs = obs.set_index("uid")
    return ad.AnnData(obs=obs)


def _build_var(
    atlas: "RaggedAtlas",
    feature_space: str,
    union_globals: np.ndarray,
) -> pd.DataFrame:
    """Build a var DataFrame from the feature registry."""
    if feature_space not in atlas._registry_tables:
        raise ValueError(
            f"No registry table for feature space '{feature_space}'. "
            f"Available: {sorted(atlas._registry_tables.keys())}"
        )
    if len(union_globals) == 0:
        return pd.DataFrame(index=pd.RangeIndex(0))

    registry_table = atlas._registry_tables[feature_space]
    registry_df = registry_table.search().to_polars()

    # Filter to union globals
    registry_df = registry_df.filter(
        pl.col("global_index").is_in(union_globals.tolist())
    ).sort("global_index")

    var = registry_df.to_pandas()
    # uid is mandatory via FeatureBaseSchema
    var = var.set_index("uid")
    return var


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

        _, union_globals, group_remap_to_union, n_features = _load_remaps_and_union(
            atlas, groups, spec
        )

        # Determine which layers to read
        if layer_overrides is not None:
            layers_to_read = layer_overrides
        else:
            layers_to_read = list(spec.required_layers)
            if not layers_to_read:
                raise ValueError(
                    f"No layers specified and spec for '{pf.feature_space}' "
                    f"has no required layers"
                )

        # Process each zarr group
        all_csrs: dict[str, list[sp.csr_matrix]] = {ln: [] for ln in layers_to_read}
        obs_parts: list[pl.DataFrame] = []

        for zg in groups:
            group_cells = cells_pl.filter(pl.col("_zg") == zg)
            starts = group_cells["_start"].to_numpy().astype(np.int64)
            ends = group_cells["_end"].to_numpy().astype(np.int64)
            n_cells_group = len(starts)

            # Batch-read index array via Rust reader
            indices_reader = atlas._get_batch_reader(zg, index_array_name)
            flat_indices, lengths = indices_reader.read_ranges(starts, ends)

            # Remap local indices -> union positions
            if zg in group_remap_to_union:
                union_remap = group_remap_to_union[zg]
                union_indices = union_remap[flat_indices.astype(np.intp)]
            else:
                union_indices = flat_indices.astype(np.int32)

            # Build indptr from lengths
            indptr = np.zeros(n_cells_group + 1, dtype=np.int64)
            np.cumsum(lengths, out=indptr[1:])

            # Batch-read each layer
            for ln in layers_to_read:
                layer_reader = atlas._get_batch_reader(
                    zg, f"layers/{ln}"
                )
                flat_values, _ = layer_reader.read_ranges(starts, ends)

                csr = sp.csr_matrix(
                    (flat_values, union_indices, indptr),
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
        var = _build_var(atlas, pf.feature_space, union_globals)

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
    ) -> ad.AnnData:
        cells_pl, groups = _prepare_dense_cells(cells_pl, pf)
        if not groups:
            return ad.AnnData()

        _, union_globals, group_remap_to_union, n_union_features = _load_remaps_and_union(
            atlas, groups, spec
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
            k: np.zeros((n_total_cells, n_union_features), dtype=np.float32)
            for k in output_keys
        }

        obs_parts: list[pl.DataFrame] = []
        offset = 0

        for zg in groups:
            group_cells = cells_pl.filter(pl.col("_zg") == zg)
            positions = group_cells["_pos"].to_numpy().astype(np.int64)
            n_cells_group = len(positions)

            # Build axis-0 ranges: each position is a single row [pos, pos+1)
            starts = positions
            ends = positions + 1

            for array_name, out_key in zip(array_names, output_keys):
                reader = atlas._get_batch_reader(zg, array_name)
                flat_data, _ = reader.read_ranges(starts, ends)
                n_local_features = flat_data.shape[0] // n_cells_group
                local_data = flat_data.reshape(n_cells_group, n_local_features)

                if zg in group_remap_to_union:
                    union_cols = group_remap_to_union[zg]
                    all_layers[out_key][offset : offset + n_cells_group][:, union_cols] = local_data
                else:
                    all_layers[out_key][offset : offset + n_cells_group, :n_local_features] = local_data

            obs_parts.append(group_cells)
            offset += n_cells_group

        obs_pl = pl.concat(obs_parts, how="diagonal_relaxed")
        obs = _build_obs_df(obs_pl)
        var = _build_var(atlas, pf.feature_space, union_globals)

        # First layer/array -> X, rest -> adata.layers
        first_key = output_keys[0]
        X = all_layers[first_key]
        extra_layers = {k: all_layers[k] for k in output_keys[1:]}

        return ad.AnnData(X=X, obs=obs, var=var, layers=extra_layers if extra_layers else None)
