"""Ingestion utilities for the multimodal perturbation atlas.

Includes multimodal batch ingestion (gene_expression + protein_abundance
+ chromatin_accessibility in one pass). Fragment-specific ingestion
functions have been moved to :mod:`lancell.fragments.ingestion`.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.sparse as sp

from lancell.atlas import RaggedAtlas
from lancell.group_specs import PointerKind, get_spec
from lancell.ingestion import (
    _CHUNK_ELEMS,
    _SHARD_ELEMS,
    _write_dense_batched,
    _write_sparse_batched,
    write_feature_layout,
)
from lancell.obs_alignment import _schema_obs_fields
from lancell.schema import DatasetRecord, make_uid

_INTEGER_DTYPES = {np.dtype("int32"), np.dtype("int64"), np.dtype("uint32"), np.dtype("uint64")}

_CHUNKS_PER_SHARD = 1024


def add_multimodal_batch(
    atlas: RaggedAtlas,
    modalities: dict[str, ad.AnnData],
    *,
    obs_df: pd.DataFrame,
    zarr_layer: str,
    dataset_records: dict[str, DatasetRecord],
) -> int:
    """Ingest aligned multimodal data, creating one cell record per cell.

    Unlike ``add_anndata_batch`` (which fills a single pointer per call),
    this writes zarr arrays for all modalities and creates cell records
    with ALL pointer fields populated in a single insert.

    Parameters
    ----------
    atlas
        Open RaggedAtlas.
    modalities
        ``{feature_space: AnnData}`` — each AnnData must have the same
        number of cells in the same barcode order. ``adata.var`` must
        have a ``global_feature_uid`` column.
    obs_df
        Shared obs DataFrame for all modalities (validated, schema-aligned).
    zarr_layer
        Zarr layer name (e.g. ``"counts"``).
    dataset_records
        ``{feature_space: DatasetRecord}`` — one per modality.

    Returns
    -------
    int
        Number of cells ingested.
    """
    n_cells = len(obs_df)
    for fs, adata in modalities.items():
        assert adata.n_obs == n_cells, f"Modality {fs} has {adata.n_obs} cells, expected {n_cells}"

    arrow_schema = atlas._cell_schema.to_arrow_schema()
    schema_fields = _schema_obs_fields(atlas._cell_schema)

    # Write dataset records and zarr arrays; collect pointer data
    pointer_data: dict[str, pa.StructArray] = {}

    for fs, adata in modalities.items():
        spec = get_spec(fs)
        ds = dataset_records[fs]
        zarr_group = ds.zarr_group

        # Add dataset record
        ds_arrow = pa.Table.from_pylist([ds.model_dump()], schema=type(ds).to_arrow_schema())
        atlas._dataset_table.add(ds_arrow)

        # Write zarr
        group = atlas._root.create_group(zarr_group)

        if spec.pointer_kind is PointerKind.SPARSE:
            data_dtype = (
                np.dtype(adata.X.dtype)
                if not sp.issparse(adata.X)
                else np.dtype(adata.X.data.dtype)
            )
            use_bitpacking = data_dtype in _INTEGER_DTYPES
            starts, ends = _write_sparse_batched(
                group,
                adata,
                zarr_layer,
                (_CHUNK_ELEMS,),
                (_CHUNKS_PER_SHARD * _CHUNK_ELEMS,),
                use_bitpacking,
                spec,
            )
            pointer_data[fs] = pa.StructArray.from_arrays(
                [
                    pa.array([fs] * n_cells, type=pa.string()),
                    pa.array([zarr_group] * n_cells, type=pa.string()),
                    pa.array(starts.astype(np.int64), type=pa.int64()),
                    pa.array(ends.astype(np.int64), type=pa.int64()),
                    pa.array(np.arange(n_cells, dtype=np.int64), type=pa.int64()),
                ],
                names=["feature_space", "zarr_group", "start", "end", "zarr_row"],
            )
        else:
            n_vars = adata.n_vars
            chunk_shape = (max(1, _CHUNK_ELEMS // n_vars), n_vars)
            shard_shape = (max(1, _SHARD_ELEMS // n_vars), n_vars)
            _write_dense_batched(group, adata, zarr_layer, chunk_shape, shard_shape, spec)
            pointer_data[fs] = pa.StructArray.from_arrays(
                [
                    pa.array([fs] * n_cells, type=pa.string()),
                    pa.array([zarr_group] * n_cells, type=pa.string()),
                    pa.array(np.arange(n_cells, dtype=np.int64), type=pa.int64()),
                ],
                names=["feature_space", "zarr_group", "position"],
            )

        # Write feature layout
        if spec.has_var_df:
            write_feature_layout(atlas, adata, fs, zarr_group, ds.uid)

    # Build cell records with all pointers
    columns = {
        "uid": pa.array([make_uid() for _ in range(n_cells)], type=pa.string()),
        "dataset_uid": pa.array(
            [dataset_records[next(iter(modalities))].uid] * n_cells, type=pa.string()
        ),
    }

    # Fill pointer fields — real data for modalities we have, zero-fill for others
    for pf_name, pf in atlas._pointer_fields.items():
        if pf.feature_space in pointer_data:
            columns[pf_name] = pointer_data[pf.feature_space]
        elif pf.pointer_kind is PointerKind.SPARSE:
            columns[pf_name] = pa.StructArray.from_arrays(
                [
                    pa.array([""] * n_cells, type=pa.string()),
                    pa.array([""] * n_cells, type=pa.string()),
                    pa.array([0] * n_cells, type=pa.int64()),
                    pa.array([0] * n_cells, type=pa.int64()),
                    pa.array([0] * n_cells, type=pa.int64()),
                ],
                names=["feature_space", "zarr_group", "start", "end", "zarr_row"],
            )
        else:
            columns[pf_name] = pa.StructArray.from_arrays(
                [
                    pa.array([""] * n_cells, type=pa.string()),
                    pa.array([""] * n_cells, type=pa.string()),
                    pa.array([0] * n_cells, type=pa.int64()),
                ],
                names=["feature_space", "zarr_group", "position"],
            )

    # Add obs columns
    for col in schema_fields:
        if col in obs_df.columns:
            columns[col] = pa.array(obs_df[col].values, type=arrow_schema.field(col).type)
    for col in schema_fields:
        if col not in columns:
            columns[col] = pa.nulls(n_cells, type=arrow_schema.field(col).type)

    arrow_table = pa.table(columns, schema=arrow_schema)
    atlas.cell_table.add(arrow_table)
    return n_cells
