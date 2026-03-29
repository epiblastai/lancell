"""Ingestion utilities for the multimodal perturbation atlas.

Includes multimodal batch ingestion (gene_expression + protein_abundance
+ chromatin_accessibility in one pass). Fragment-specific ingestion
functions have been moved to :mod:`lancell.fragments.ingestion`.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import scipy.sparse as sp

from lancell.atlas import RaggedAtlas
from lancell.fragments.ingestion import (
    build_chrom_order,
    build_end_max,
    parse_bed_fragments,
    sort_fragments_by_cell,
    sort_fragments_by_genome,
    write_fragment_arrays,
    write_genome_sorted_arrays,
)
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
            chunk_rows = max(1, _CHUNK_ELEMS // n_vars)
            chunk_shape = (chunk_rows, n_vars)
            shard_shape = (chunk_rows * _CHUNKS_PER_SHARD, n_vars)
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


# ---------------------------------------------------------------------------
# Metadata-only ingestion (no zarr data)
# ---------------------------------------------------------------------------


def add_metadata_only_batch(
    atlas: RaggedAtlas,
    obs_df: pd.DataFrame,
    *,
    dataset_record: DatasetRecord,
) -> int:
    """Ingest cell records without any matrix data.

    Use this for datasets where the actual feature matrices are not available
    (e.g., scATAC-seq QC-only datasets where fragment files haven't been
    processed into count matrices). All zarr pointers are zero-filled.

    Parameters
    ----------
    atlas
        Open RaggedAtlas.
    obs_df
        Validated obs DataFrame with schema-aligned columns.
    dataset_record
        Dataset record to register. ``zarr_group`` is still required
        but no zarr group is created on disk.

    Returns
    -------
    int
        Number of cells ingested.
    """
    n_cells = len(obs_df)

    # Add dataset record
    ds_arrow = pa.Table.from_pylist(
        [dataset_record.model_dump()],
        schema=type(dataset_record).to_arrow_schema(),
    )
    atlas._dataset_table.add(ds_arrow)

    # Build cell records — all pointers zero-filled
    arrow_schema = atlas._cell_schema.to_arrow_schema()
    schema_fields = _schema_obs_fields(atlas._cell_schema)

    columns = {
        "uid": pa.array([make_uid() for _ in range(n_cells)], type=pa.string()),
        "dataset_uid": pa.array([dataset_record.uid] * n_cells, type=pa.string()),
    }

    # Zero-fill all pointer fields
    for pf_name, pf in atlas._pointer_fields.items():
        if pf.pointer_kind is PointerKind.SPARSE:
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


# ---------------------------------------------------------------------------
# Fragment-based ingestion (chromatin accessibility)
# ---------------------------------------------------------------------------


def add_fragment_batch(
    atlas: RaggedAtlas,
    bed_path: Path | None = None,
    *,
    obs_df: pd.DataFrame,
    chrom_uids: dict[str, str],
    feature_space: str,
    dataset_record: DatasetRecord,
    barcode_col: str = "barcode",
    fragments: pl.DataFrame | None = None,
) -> int:
    """Ingest fragment data into the atlas.

    Accepts either a BED file path or a pre-parsed polars DataFrame of
    fragments (but not both). Writes cell-sorted and genome-sorted
    fragment arrays to zarr, writes the feature layout, and inserts cell
    records with ``SparseZarrPointer`` values.

    Parameters
    ----------
    atlas
        Open RaggedAtlas.
    bed_path
        Path to a (possibly gzipped) BED fragment file (4- or 5-column).
        Mutually exclusive with *fragments*.
    obs_df
        Validated obs DataFrame. Its index must be cell barcodes that
        appear in the fragment data. Order determines cell record order.
    chrom_uids
        ``{chromosome_name: global_feature_uid}`` mapping for all
        chromosomes that may appear in the fragment data. Chromosomes
        not in this dict are silently dropped.
    feature_space
        Feature space name (``"chromatin_accessibility"``).
    dataset_record
        Dataset record to register.
    barcode_col
        Name of the barcode column used internally. Defaults to ``"barcode"``.
    fragments
        Pre-parsed polars DataFrame with columns ``chrom`` (str),
        ``start`` (uint32), ``length`` (uint16), and ``<barcode_col>``
        (str) — the same schema returned by
        :func:`~lancell.fragments.ingestion.parse_bed_fragments`.
        Mutually exclusive with *bed_path*.

    Returns
    -------
    int
        Number of cells ingested.
    """
    if (bed_path is None) == (fragments is None):
        raise ValueError("Exactly one of bed_path or fragments must be provided.")

    if atlas._cell_schema is None:
        raise ValueError(
            "Cannot ingest data into an atlas opened without a cell schema."
        )

    spec = get_spec(feature_space)
    zarr_group = dataset_record.zarr_group

    # --- Parse and filter fragments ---
    if fragments is None:
        print(f"  Parsing BED file: {bed_path.name} ...")
        fragments = parse_bed_fragments(bed_path, barcode_col=barcode_col)

    # Keep only chromosomes we have registered features for
    known_chroms = set(chrom_uids.keys())
    n_before = len(fragments)
    fragments = fragments.filter(pl.col("chrom").is_in(known_chroms))
    n_after = len(fragments)
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after:,} fragments on unregistered chromosomes")

    # Keep only barcodes that are in obs_df
    obs_barcodes = set(obs_df.index.tolist())
    fragments = fragments.filter(pl.col(barcode_col).is_in(obs_barcodes))
    print(f"  {len(fragments):,} fragments for {len(obs_barcodes):,} cells")

    # --- Sort by cell ---
    chrom_order = build_chrom_order(fragments)
    chromosomes, starts, lengths, offsets, cell_ids = sort_fragments_by_cell(
        fragments, chrom_order, barcode_col=barcode_col
    )
    print(f"  Sorted {len(chromosomes):,} fragments by cell ({len(cell_ids):,} cells)")

    # Align obs_df to the cell_ids order from sort_fragments_by_cell
    obs_df = obs_df.loc[cell_ids]
    n_cells = len(cell_ids)

    # --- Write dataset record ---
    dataset_record.n_cells = n_cells
    ds_arrow = pa.Table.from_pylist(
        [dataset_record.model_dump()],
        schema=type(dataset_record).to_arrow_schema(),
    )
    atlas._dataset_table.add(ds_arrow)

    # --- Write zarr arrays ---
    group = atlas._root.create_group(zarr_group)
    write_fragment_arrays(group, chromosomes, starts, lengths)

    # Genome-sorted arrays for range queries
    gs_cell_ids, gs_starts, gs_lengths, chrom_offsets = sort_fragments_by_genome(
        fragments, chrom_order, cell_ids, barcode_col=barcode_col
    )
    end_max = build_end_max(gs_starts, gs_lengths)
    write_genome_sorted_arrays(
        group, gs_cell_ids, gs_starts, gs_lengths, chrom_offsets, end_max
    )
    print(f"  Wrote cell-sorted and genome-sorted fragment arrays")

    # --- Write feature layout ---
    var_df = pl.DataFrame({
        "global_feature_uid": [chrom_uids[c] for c in chrom_order],
    })
    atlas.add_or_reuse_layout(var_df, dataset_record.uid, feature_space)

    # --- Build and insert cell records ---
    arrow_schema = atlas._cell_schema.to_arrow_schema()
    schema_fields = _schema_obs_fields(atlas._cell_schema)

    pointer_field = None
    for pf in atlas._pointer_fields.values():
        if pf.feature_space == feature_space:
            pointer_field = pf
            break
    assert pointer_field is not None

    pointer_starts = offsets[:-1].astype(np.int64)
    pointer_ends = offsets[1:].astype(np.int64)

    pointer_struct = pa.StructArray.from_arrays(
        [
            pa.array([feature_space] * n_cells, type=pa.string()),
            pa.array([zarr_group] * n_cells, type=pa.string()),
            pa.array(pointer_starts, type=pa.int64()),
            pa.array(pointer_ends, type=pa.int64()),
            pa.array(np.arange(n_cells, dtype=np.int64), type=pa.int64()),
        ],
        names=["feature_space", "zarr_group", "start", "end", "zarr_row"],
    )

    columns = {
        "uid": pa.array([make_uid() for _ in range(n_cells)], type=pa.string()),
        "dataset_uid": pa.array([dataset_record.uid] * n_cells, type=pa.string()),
        pointer_field.field_name: pointer_struct,
    }

    # Zero-fill other pointer fields
    for other_name, other_pf in atlas._pointer_fields.items():
        if other_name == pointer_field.field_name:
            continue
        if other_pf.pointer_kind is PointerKind.SPARSE:
            columns[other_name] = pa.StructArray.from_arrays(
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
            columns[other_name] = pa.StructArray.from_arrays(
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
