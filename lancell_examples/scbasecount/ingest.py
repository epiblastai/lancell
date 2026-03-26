"""Ingest a single scBaseCount h5ad file into a lancell ragged atlas.

Usage:
    python -m lancell_examples.scbasecount.ingest \
        --h5ad /path/to/SRX12345.h5ad \
        --atlas-dir ./atlas/scbasecount \
        --sample-metadata ./data/scbasecount/sample_metadata.parquet \
        --no-csc
"""

import argparse
import os
from pathlib import Path

import anndata as ad
import numpy as np
import obstore.store
import polars as pl
import pyarrow as pa
import scipy.sparse as sp

from lancell.atlas import RaggedAtlas
from lancell.codecs.bitpacking import BitpackingCodec
from lancell.ingestion import add_csc
from lancell.obs_alignment import PointerFieldInfo, _schema_obs_fields
from lancell.schema import make_uid
from lancell_examples.scbasecount.schema import (
    CellObs,
    GeneFeatureSpace,
    ScBasecountDatasetRecord,
)

FEATURE_SPACE = "genefull_expression"
CHUNK_SIZE = 40_960
SHARD_SIZE = 1024 * CHUNK_SIZE

OBS_RENAME = {
    "gene_count_Unique": "gene_count_unique",
    "umi_count_Unique": "umi_count_unique",
    "SRX_accession": "srx_accession",
}


def make_store(atlas_dir: str) -> obstore.store.ObjectStore:
    """Build an obstore for the zarr data, choosing backend from the path."""
    if atlas_dir.startswith("s3://"):
        from urllib.parse import urlparse

        parsed = urlparse(atlas_dir)
        bucket = parsed.netloc
        prefix = os.path.join(parsed.path.strip("/"), "zarr_store")
        region = os.environ.get("AWS_REGION")
        if not region:
            raise ValueError("AWS_REGION environment variable must be set for S3 access")
        return obstore.store.S3Store(bucket, prefix=prefix, region=region)
    else:
        zarr_path = Path(atlas_dir) / "zarr_store"
        zarr_path.mkdir(parents=True, exist_ok=True)
        return obstore.store.LocalStore(str(zarr_path))


def db_uri_for(atlas_dir: str) -> str:
    """Return the LanceDB URI for the given atlas directory."""
    if atlas_dir.startswith("s3://"):
        return atlas_dir.rstrip("/") + "/lance_db"
    return str(Path(atlas_dir) / "lance_db")


def atlas_exists(atlas_dir: str) -> bool:
    """Check whether an atlas already exists at the given path."""
    if atlas_dir.startswith("s3://"):
        import lancedb

        db = lancedb.connect(db_uri_for(atlas_dir))
        return "cells" in db.list_tables().tables
    return (Path(atlas_dir) / "lance_db").exists()


def create_atlas(atlas_dir: str) -> RaggedAtlas:
    """Create a new atlas at the given directory."""
    if not atlas_dir.startswith("s3://"):
        Path(atlas_dir).mkdir(parents=True, exist_ok=True)
    store = make_store(atlas_dir)
    return RaggedAtlas.create(
        db_uri=db_uri_for(atlas_dir),
        cell_table_name="cells",
        cell_schema=CellObs,
        dataset_table_name="datasets",
        dataset_schema=ScBasecountDatasetRecord,
        store=store,
        registry_schemas={FEATURE_SPACE: GeneFeatureSpace},
    )


def open_atlas(atlas_dir: str) -> RaggedAtlas:
    """Open an existing atlas."""
    store = make_store(atlas_dir)
    return RaggedAtlas.open(
        db_uri=db_uri_for(atlas_dir),
        cell_table_name="cells",
        cell_schema=CellObs,
        dataset_table_name="datasets",
        store=store,
        registry_tables={FEATURE_SPACE: "genefull_expression_registry"},
    )


def register_genes(atlas: RaggedAtlas, adata: ad.AnnData, organism: str) -> dict[str, str]:
    """Register gene features and return gene_id -> registry uid mapping.

    Deduplication is strictly by ``adata.var.index`` (the gene_id), never by
    the ``gene_symbols`` column which can collide across organisms.
    """
    var = adata.var
    # Deduplicate var by index so each unique gene_id is registered once
    var_deduped = var[~var.index.duplicated(keep="first")]
    gene_ids = list(var_deduped.index)

    registry_table = atlas._registry_tables[FEATURE_SPACE]
    existing_df = registry_table.search().select(["uid", "gene_id"]).to_polars()
    existing_gene_to_uid = dict(
        zip(existing_df["gene_id"].to_list(), existing_df["uid"].to_list(), strict=False)
    )

    gene_to_uid: dict[str, str] = {}
    new_features: list[GeneFeatureSpace] = []
    for i, gene_id in enumerate(gene_ids):
        if gene_id in existing_gene_to_uid:
            gene_to_uid[gene_id] = existing_gene_to_uid[gene_id]
        else:
            gene_name = (
                str(var_deduped.iloc[i]["gene_symbols"])
                if "gene_symbols" in var_deduped.columns
                else gene_id
            )
            feature = GeneFeatureSpace(gene_id=gene_id, gene_name=gene_name, organism=organism)
            new_features.append(feature)
            gene_to_uid[gene_id] = feature.uid

    if new_features:
        n_new = atlas.register_features(FEATURE_SPACE, new_features)
        print(
            f"  Registered {n_new} new genes ({len(gene_ids)} unique in this file, "
            f"{len(existing_gene_to_uid)} already existed)"
        )
    else:
        print(f"  All {len(gene_ids)} genes already registered")

    return gene_to_uid


def _union_sparsity(
    matrices: list[sp.csr_matrix],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Compute union sparsity pattern across multiple CSR matrices.

    Returns
    -------
    indptr : np.ndarray
        CSR indptr for the union pattern.
    indices : np.ndarray
        Flat column indices for the union pattern.
    reindexed_values : list[np.ndarray]
        One flat values array per input matrix, reindexed to the union pattern
        (zeros where that matrix had no entry).
    """
    n_rows = matrices[0].shape[0]

    all_indptr = [m.indptr for m in matrices]
    all_indices = [m.indices for m in matrices]
    all_data = [m.data for m in matrices]

    union_indices_parts = []
    union_indptr = np.empty(n_rows + 1, dtype=np.int64)
    union_indptr[0] = 0

    reindexed_parts: list[list[np.ndarray]] = [[] for _ in matrices]

    for i in range(n_rows):
        # Collect column indices from all matrices for this row
        row_col_sets = []
        for k in range(len(matrices)):
            s, e = int(all_indptr[k][i]), int(all_indptr[k][i + 1])
            row_col_sets.append(all_indices[k][s:e])

        # Union of column indices, sorted
        union_cols = (
            np.unique(np.concatenate(row_col_sets))
            if any(len(c) > 0 for c in row_col_sets)
            else np.array([], dtype=np.int32)
        )
        union_indices_parts.append(union_cols)
        union_indptr[i + 1] = union_indptr[i] + len(union_cols)

        # Reindex each matrix's values into the union pattern
        for k in range(len(matrices)):
            s, e = int(all_indptr[k][i]), int(all_indptr[k][i + 1])
            vals = np.zeros(len(union_cols), dtype=all_data[k].dtype)
            if e > s:
                col_k = all_indices[k][s:e]
                data_k = all_data[k][s:e]
                # Find positions of col_k in union_cols
                pos = np.searchsorted(union_cols, col_k)
                vals[pos] = data_k
            reindexed_parts[k].append(vals)

    indices = (
        np.concatenate(union_indices_parts) if union_indices_parts else np.array([], dtype=np.int32)
    )
    reindexed_values = [
        np.concatenate(parts) if parts else np.array([], dtype=matrices[k].data.dtype)
        for k, parts in enumerate(reindexed_parts)
    ]
    return union_indptr, indices, reindexed_values


def ingest_genefull(
    atlas: RaggedAtlas,
    adata: ad.AnnData,
    h5ad_path: str,
    gene_to_uid: dict[str, str],
    sample_row: dict,
    feature_type: str,
    release_date: str,
) -> tuple[int, str]:
    """Ingest an h5ad with GeneFull_Ex50pAS layers into the atlas."""
    n_cells = adata.n_obs
    print(f"  n_cells={n_cells:,}")

    srx_accession = Path(h5ad_path).stem
    zarr_group = make_uid()

    # Rename obs columns
    adata.obs = adata.obs.rename(columns=OBS_RENAME)

    # Attach global_feature_uid to adata.var
    gene_ids = list(adata.var.index)
    adata.var["global_feature_uid"] = [gene_to_uid[gid] for gid in gene_ids]

    # Get sparse matrices for all 3 layers
    unique = adata.X if isinstance(adata.X, sp.csr_matrix) else sp.csr_matrix(adata.X)
    em = (
        adata.layers["UniqueAndMult-EM"]
        if isinstance(adata.layers["UniqueAndMult-EM"], sp.csr_matrix)
        else sp.csr_matrix(adata.layers["UniqueAndMult-EM"])
    )
    uniform = (
        adata.layers["UniqueAndMult-Uniform"]
        if isinstance(adata.layers["UniqueAndMult-Uniform"], sp.csr_matrix)
        else sp.csr_matrix(adata.layers["UniqueAndMult-Uniform"])
    )

    # Cast float to int32 if values are raw integer counts
    for mat in (unique, em, uniform):
        if mat.data.dtype.kind == "f" and np.array_equal(mat.data, np.floor(mat.data)):
            mat.data = mat.data.astype(np.int32)

    # Compute union sparsity and reindex
    union_indptr, union_indices, reindexed = _union_sparsity([unique, em, uniform])
    nnz = len(union_indices)

    # Determine if data is integer for bitpacking
    data_dtype = unique.data.dtype
    use_bitpacking = data_dtype in {
        np.dtype("int32"),
        np.dtype("int64"),
        np.dtype("uint32"),
        np.dtype("uint64"),
    }
    indices_kwargs: dict = {"compressors": BitpackingCodec(transform="delta")}
    layer_kwargs: dict = {}
    if use_bitpacking:
        layer_kwargs["compressors"] = BitpackingCodec(transform="none")

    chunk_shape = (CHUNK_SIZE,)
    shard_shape = (SHARD_SIZE,)

    # Write zarr group
    group = atlas._root.create_group(zarr_group)
    csr_group = group.create_group("csr")

    zarr_indices = csr_group.create_array(
        "indices",
        shape=(nnz,),
        dtype=np.uint32,
        chunks=chunk_shape,
        shards=shard_shape,
        **indices_kwargs,
    )
    layers_group = csr_group.create_group("layers")

    layer_names = ["Unique", "UniqueAndMult-EM", "UniqueAndMult-Uniform"]
    for k, layer_name in enumerate(layer_names):
        zarr_layer = layers_group.create_array(
            layer_name,
            shape=(nnz,),
            dtype=data_dtype,
            chunks=chunk_shape,
            shards=shard_shape,
            **layer_kwargs,
        )
        # Write in batches
        batch_size = shard_shape[0]
        written = 0
        while written < nnz:
            end = min(written + batch_size, nnz)
            if k == 0:
                # Write indices only once (with first layer)
                zarr_indices[written:end] = union_indices[written:end].astype(np.uint32, copy=False)
            zarr_layer[written:end] = reindexed[k][written:end]
            written = end

    # Build starts/ends from union indptr
    starts = union_indptr[:-1].astype(np.int64)
    ends = union_indptr[1:].astype(np.int64)

    # Build dataset record from sample metadata (must exist before add_or_reuse_layout)
    dataset_kwargs = {
        "uid": zarr_group,
        "zarr_group": zarr_group,
        "feature_space": FEATURE_SPACE,
        "n_cells": n_cells,
        "srx_accession": srx_accession,
        "feature_type": feature_type,
        "release_date": release_date,
    }
    # Copy sample metadata fields that exist
    sample_fields = [
        "lib_prep",
        "tech_10x",
        "cell_prep",
        "organism",
        "tissue",
        "tissue_ontology_term_id",
        "disease",
        "disease_ontology_term_id",
        "perturbation",
        "cell_line",
        "antibody_derived_tag",
        "czi_collection_id",
        "czi_collection_name",
    ]
    for field in sample_fields:
        if field in sample_row:
            dataset_kwargs[field] = sample_row[field]

    dataset_record = ScBasecountDatasetRecord(**dataset_kwargs)
    dataset_arrow = pa.Table.from_pylist(
        [dataset_record.model_dump()],
        schema=ScBasecountDatasetRecord.to_arrow_schema(),
    )
    atlas._dataset_table.add(dataset_arrow)

    # Register feature layout (updates dataset record with layout_uid)
    var_pl = pl.DataFrame({"global_feature_uid": adata.var["global_feature_uid"].tolist()})
    atlas.add_or_reuse_layout(var_pl, zarr_group, FEATURE_SPACE)

    # Insert cell records
    pointer_field: PointerFieldInfo | None = None
    for pf in atlas._pointer_fields.values():
        if pf.feature_space == FEATURE_SPACE:
            pointer_field = pf
            break

    arrow_schema = CellObs.to_arrow_schema()
    obs_df = adata.obs
    schema_fields = _schema_obs_fields(CellObs)

    pointer_struct = pa.StructArray.from_arrays(
        [
            pa.array([FEATURE_SPACE] * n_cells, type=pa.string()),
            pa.array([zarr_group] * n_cells, type=pa.string()),
            pa.array(starts, type=pa.int64()),
            pa.array(ends, type=pa.int64()),
            pa.array(np.arange(n_cells, dtype=np.int64), type=pa.int64()),
        ],
        names=["feature_space", "zarr_group", "start", "end", "zarr_row"],
    )

    columns = {
        "uid": pa.array([make_uid() for _ in range(n_cells)], type=pa.string()),
        "dataset_uid": pa.array([dataset_record.uid] * n_cells, type=pa.string()),
        pointer_field.field_name: pointer_struct,
    }

    for col in schema_fields:
        if col in obs_df.columns:
            columns[col] = pa.array(obs_df[col].values, type=arrow_schema.field(col).type)

    for col in schema_fields:
        if col not in columns:
            columns[col] = pa.nulls(n_cells, type=arrow_schema.field(col).type)

    arrow_table = pa.table(columns, schema=arrow_schema)
    atlas.cell_table.add(arrow_table)

    print(f"    Inserted {n_cells:,} cell records")
    return n_cells, zarr_group


def main():
    parser = argparse.ArgumentParser(description="Ingest a scBaseCount h5ad into a lancell atlas")
    parser.add_argument("--h5ad", required=True, help="Path to .h5ad file")
    parser.add_argument("--atlas-dir", required=True, help="Path to atlas directory")
    parser.add_argument("--sample-metadata", required=True, help="Path to sample_metadata.parquet")
    parser.add_argument(
        "--feature-type",
        default="GeneFull_Ex50pAS",
        help="Feature type (default: GeneFull_Ex50pAS)",
    )
    parser.add_argument(
        "--release-date", default="2026-01-12", help="Release date (default: 2026-01-12)"
    )
    parser.add_argument("--no-csc", action="store_true", help="Skip CSC layout")
    args = parser.parse_args()

    h5ad_path = args.h5ad
    atlas_dir = args.atlas_dir

    # Read sample metadata and find row for this h5ad
    srx_accession = Path(h5ad_path).stem
    metadata_df = pl.read_parquet(args.sample_metadata)
    srx_col = "SRX_accession" if "SRX_accession" in metadata_df.columns else "srx_accession"
    sample_rows = metadata_df.filter(pl.col(srx_col) == srx_accession).to_dicts()
    if not sample_rows:
        raise ValueError(f"No sample metadata found for {srx_accession}")
    sample_row = sample_rows[0]

    print(f"Loading {h5ad_path}...")
    adata = ad.read_h5ad(h5ad_path)

    if atlas_exists(atlas_dir):
        print(f"Opening existing atlas at {atlas_dir}")
        atlas = open_atlas(atlas_dir)
    else:
        print(f"Creating new atlas at {atlas_dir}")
        atlas = create_atlas(atlas_dir)

    print("Registering genes...")
    gene_to_uid = register_genes(atlas, adata, organism=sample_row["organism"])

    print("Ingesting genefull data...")
    n_cells, zarr_group = ingest_genefull(
        atlas,
        adata,
        h5ad_path,
        gene_to_uid,
        sample_row,
        args.feature_type,
        args.release_date,
    )

    if not args.no_csc:
        print("Adding CSC layout for Unique layer...")
        add_csc(
            atlas,
            zarr_group=zarr_group,
            feature_space=FEATURE_SPACE,
            layer_name="Unique",
            chunk_size=CHUNK_SIZE,
            shard_size=SHARD_SIZE,
        )

    print(f"Done! Ingested {n_cells:,} cells from {Path(h5ad_path).name}")


if __name__ == "__main__":
    main()
