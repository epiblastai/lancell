"""Post-processing tool: add CSC layout alongside an existing CSR zarr group.

Usage::

    from lancell.tools.add_csc import add_csc
    add_csc(atlas, zarr_group="datasets/my_dataset", feature_space="gene_expression")

After running, ``_dataset_vars`` gains ``csc_start``/``csc_end`` values and a new
``{zarr_group}/csc/`` subgroup appears alongside the existing ``{zarr_group}/csr/``.
Subsequent feature-filtered queries will automatically use the CSC path.
"""

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import zarr

from lancell.batch_array import BatchArray
from lancell.dataset_vars import read_dataset_vars

if TYPE_CHECKING:
    from lancell.atlas import RaggedAtlas


def add_csc(
    atlas: "RaggedAtlas",
    zarr_group: str,
    feature_space: str,
    layer_name: str = "counts",
    chunk_size: int = 4096,
    shard_size: int = 65536,
) -> None:
    """Read existing CSR group and write CSC alongside it.

    Reads the full CSR flat arrays from ``{zarr_group}/csr/``, transposes
    to CSC order sorted by feature index, writes ``{zarr_group}/csc/``, and
    updates ``_dataset_vars`` with ``csc_start``/``csc_end`` values.

    Parameters
    ----------
    atlas:
        The atlas whose zarr store and cell table to use.
    zarr_group:
        Path of the zarr group to process (relative to atlas store root).
    feature_space:
        The feature space this zarr group belongs to.
    layer_name:
        Which layer to transpose (e.g. ``"counts"``).
    chunk_size:
        Chunk size for the new CSC zarr arrays.
    shard_size:
        Shard size for the new CSC zarr arrays.

    Raises
    ------
    ValueError
        If no cells or no dataset record are found for this group, or if
        ``zarr_row`` is not sequential.
    """
    if atlas._dataset_vars_table is None:
        raise ValueError(
            "_dataset_vars table not found. This atlas may have been created "
            "before _dataset_vars was introduced."
        )

    # Look up dataset_uid for this zarr_group + feature_space
    datasets_df = (
        atlas._dataset_table.search()
        .to_polars()
        .filter((pl.col("zarr_group") == zarr_group) & (pl.col("feature_space") == feature_space))
        .select(["uid"])
    )
    if datasets_df.is_empty():
        raise ValueError(
            f"No dataset record found for zarr_group='{zarr_group}', "
            f"feature_space='{feature_space}'"
        )
    dataset_uid = datasets_df["uid"][0]

    # Query all cells in this zarr group
    cells_df = atlas.cell_table.search().to_polars()
    ptr_struct = cells_df[feature_space].struct.unnest()
    cells_df = cells_df.with_columns(
        ptr_struct["zarr_group"].alias("_zg"),
        ptr_struct["zarr_row"].alias("_zarr_row"),
        ptr_struct["start"].alias("_start"),
        ptr_struct["end"].alias("_end"),
    ).filter(pl.col("_zg") == zarr_group)

    if cells_df.is_empty():
        raise ValueError(
            f"No cells found for zarr_group='{zarr_group}', feature_space='{feature_space}'"
        )

    cells_df = cells_df.sort("_zarr_row")
    zarr_rows = cells_df["_zarr_row"].to_numpy()
    starts = cells_df["_start"].to_numpy()
    ends = cells_df["_end"].to_numpy()
    n_cells = len(zarr_rows)

    if not np.array_equal(zarr_rows, np.arange(n_cells)):
        raise ValueError(
            f"zarr_rows for group '{zarr_group}' are not sequential 0..{n_cells - 1}. "
            f"Was zarr_row populated correctly during ingest?"
        )

    # Read the full flat CSR arrays
    csr_index_arr = BatchArray.from_array(atlas._root[f"{zarr_group}/csr/indices"])
    csr_layer_arr = BatchArray.from_array(atlas._root[f"{zarr_group}/csr/layers/{layer_name}"])

    flat_indices, lengths = csr_index_arr.read_ranges(
        starts.astype(np.int64), ends.astype(np.int64)
    )
    flat_values, _ = csr_layer_arr.read_ranges(starts.astype(np.int64), ends.astype(np.int64))

    # Reconstruct (zarr_row, feature_idx) for every non-zero element
    cell_ids = np.repeat(np.arange(n_cells, dtype=np.int64), lengths)
    feature_indices = flat_indices.astype(np.int64)

    # Sort by feature index (stable preserves zarr_row order within each feature)
    sort_order = np.argsort(feature_indices, kind="stable")
    sorted_cell_ids = cell_ids[sort_order]
    sorted_features = feature_indices[sort_order]
    sorted_values = flat_values[sort_order]

    # Get n_features from _dataset_vars
    rows = read_dataset_vars(atlas._dataset_vars_table, dataset_uid)
    n_features = len(rows)

    feature_range = np.arange(n_features, dtype=np.int64)
    csc_start = np.searchsorted(sorted_features, feature_range, side="left").astype(np.int64)
    csc_end = np.searchsorted(sorted_features, feature_range, side="right").astype(np.int64)

    # Write CSC zarr arrays (open writable root so this works even when atlas was opened read-only)
    _writable_root = zarr.open_group(zarr.storage.ObjectStore(atlas._store), mode="a")
    csc_group = _writable_root.require_group(f"{zarr_group}/csc")
    csc_group.create_array(
        "indices",
        data=sorted_cell_ids.astype(np.uint32),
        chunks=(chunk_size,),
        shards=(shard_size,),
    )
    layers_group = csc_group.create_group("layers")
    layers_group.create_array(
        layer_name,
        data=sorted_values,
        chunks=(chunk_size,),
        shards=(shard_size,),
    )

    # Update _dataset_vars with csc_start/csc_end via merge_insert
    rows = rows.with_columns(
        pl.Series("csc_start", csc_start),
        pl.Series("csc_end", csc_end),
    )
    (
        atlas._dataset_vars_table.merge_insert(on=["feature_uid", "dataset_uid"])
        .when_matched_update_all()
        .execute(rows)
    )
    # Cache invalidation is automatic: GroupReader checks _dataset_vars_table.version
    # on next access to var_df or has_csc.
    atlas._group_readers.pop((zarr_group, feature_space), None)
