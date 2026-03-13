"""Ingest a single cellxgene census h5ad file into a lancell ragged atlas.

Usage:
    python -m examples.cellxgene_census.ingest \
        --h5ad /path/to/dataset.h5ad \
        --atlas-dir /path/to/atlas

Designed to be run in parallel for multiple h5ad files against the same atlas.
"""

import argparse
import os
from pathlib import Path

import anndata as ad
import numpy as np
import obstore.store
import polars as pl
import pyarrow as pa
import zarr

from examples.cellxgene_census.schema import (
    CellObs,
    CensusDatasetRecord,
    GeneFeatureSpace,
)
from lancell.atlas import RaggedAtlas, _schema_obs_fields
from lancell.codecs.bitpacking import BitpackingCodec
from lancell.schema import make_uid
from lancell.var_df import build_remap, reindex_registry, write_remap, write_var_df

FEATURE_SPACE = "gene_expression"
LAYER_NAME = "counts"
CHUNK_SIZE = 5_000
SHARD_SIZE = 50_000_000


def make_store(atlas_dir: str) -> obstore.store.ObjectStore:
    """Build an obstore for the zarr data, choosing backend from the path."""
    if atlas_dir.startswith("s3://"):
        # e.g. "s3://bucket/prefix/atlas"
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
        dataset_schema=CensusDatasetRecord,
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
        registry_tables={FEATURE_SPACE: "gene_expression_registry"},
    )


def register_genes(atlas: RaggedAtlas, adata: ad.AnnData) -> dict[str, str]:
    """Register gene features and return ensembl_id -> registry uid mapping."""
    var = adata.var
    ensembl_ids = list(var.index)

    # Load existing registry to find genes already registered
    registry_table = atlas._registry_tables[FEATURE_SPACE]
    existing_df = registry_table.search().select(["uid", "ensembl_id"]).to_polars()
    existing_ensembl_to_uid = dict(
        zip(existing_df["ensembl_id"].to_list(), existing_df["uid"].to_list(), strict=False)
    )

    # Only create new features for genes not already in the registry
    ensembl_to_uid: dict[str, str] = {}
    new_features: list[GeneFeatureSpace] = []
    for ensembl_id in ensembl_ids:
        if ensembl_id in existing_ensembl_to_uid:
            ensembl_to_uid[ensembl_id] = existing_ensembl_to_uid[ensembl_id]
        elif ensembl_id in ensembl_to_uid:
            # Already seen in this file (duplicate var index), skip
            pass
        else:
            row = var.loc[ensembl_id]
            feature = GeneFeatureSpace(
                ensembl_id=ensembl_id,
                feature_name=str(row["feature_name"]),
                feature_reference=str(row["feature_reference"]),
                feature_biotype=str(row["feature_biotype"]),
                feature_length=int(row["feature_length"]),
                feature_type=str(row["feature_type"]),
                feature_is_filtered=bool(row["feature_is_filtered"]),
            )
            new_features.append(feature)
            ensembl_to_uid[ensembl_id] = feature.uid

    if new_features:
        n_new = atlas.register_features(FEATURE_SPACE, new_features)
        print(
            f"  Registered {n_new} new genes ({len(ensembl_ids)} total in this file, "
            f"{len(existing_ensembl_to_uid)} already existed)"
        )
    else:
        print(f"  All {len(ensembl_ids)} genes already registered")

    reindex_registry(registry_table)
    return ensembl_to_uid


def ingest_backed(
    atlas: RaggedAtlas,
    adata: ad.AnnData,
    h5ad_path: str,
    ensembl_to_uid: dict[str, str],
) -> int:
    """Ingest a backed h5ad into the atlas using streaming writes.

    1. Read nnz from the HDF5 file without loading data
    2. Pre-allocate zarr arrays with shape=(nnz,)
    3. Stream shard-sized batches of values/indices from the backed CSR
    4. Build cell records with correct start/end pointers
    """
    h5_file = adata.file._file
    h5_data = h5_file["X/data"]
    h5_indices = h5_file["X/indices"]
    h5_indptr = h5_file["X/indptr"]

    nnz = h5_data.shape[0]
    n_cells = adata.n_obs
    print(f"  nnz={nnz:,}, n_cells={n_cells:,}")

    # Create dataset record
    cellxgene_dataset_id = Path(h5ad_path).stem
    zarr_group = make_uid()
    dataset_record = CensusDatasetRecord(
        uid=zarr_group,
        zarr_group=zarr_group,
        feature_space=FEATURE_SPACE,
        n_cells=n_cells,
        cellxgene_dataset_id=cellxgene_dataset_id,
    )

    # Write dataset record
    dataset_arrow = pa.Table.from_pylist(
        [dataset_record.model_dump()],
        schema=CensusDatasetRecord.to_arrow_schema(),
    )
    atlas._dataset_table.add(dataset_arrow)

    # Create zarr group and pre-allocate arrays
    # Use a writable zarr store wrapping the same obstore (atlas._root may be read-only)
    writable_store = zarr.storage.ObjectStore(atlas._store)
    group = zarr.open_group(writable_store, path=zarr_group, mode="w")
    group.create_array(
        "indices",
        shape=(nnz,),
        dtype=np.uint32,
        chunks=(CHUNK_SIZE,),
        shards=(SHARD_SIZE,),
        compressors=BitpackingCodec(transform="delta"),
    )
    layers = group.create_group("layers")
    layers.create_array(
        LAYER_NAME,
        shape=(nnz,),
        dtype=np.uint32,
        chunks=(CHUNK_SIZE,),
        shards=(SHARD_SIZE,),
        compressors=BitpackingCodec(transform="none"),
    )

    # Read indptr fully (small: n_cells+1 int64 values)
    indptr = h5_indptr[:]
    starts = indptr[:-1]
    ends = indptr[1:]

    # Stream data in shard-sized batches
    zarr_indices = group["indices"]
    zarr_counts = group["layers"][LAYER_NAME]

    written = 0
    while written < nnz:
        batch_end = min(written + SHARD_SIZE, nnz)
        chunk_indices = h5_indices[written:batch_end].astype(np.uint32)
        chunk_values = h5_data[written:batch_end].astype(np.uint32)
        zarr_indices[written:batch_end] = chunk_indices
        zarr_counts[written:batch_end] = chunk_values
        written = batch_end
        print(f"    Written {written:,}/{nnz:,} nnz values")

    # Write var_df sidecar with global_feature_uid mapped to registry uids
    ensembl_ids = list(adata.var.index)
    global_feature_uids = [ensembl_to_uid[eid] for eid in ensembl_ids]
    var_df = pl.from_pandas(adata.var.reset_index())
    var_df = var_df.with_columns(pl.Series("global_feature_uid", global_feature_uids))
    write_var_df(atlas._store, zarr_group, var_df)

    # Build and write remap
    registry_table = atlas._registry_tables[FEATURE_SPACE]
    remap = build_remap(var_df, registry_table)
    write_remap(atlas._store, group, remap, registry_version=registry_table.version)

    # Build and insert cell records directly from the obs table
    arrow_schema = CellObs.to_arrow_schema()
    schema_fields = _schema_obs_fields(CellObs)
    obs_df = adata.obs

    # Build pointer struct array
    pointer_struct = pa.StructArray.from_arrays(
        [
            pa.array([FEATURE_SPACE] * n_cells, type=pa.string()),
            pa.array([zarr_group] * n_cells, type=pa.string()),
            pa.array(starts.astype(np.int64), type=pa.int64()),
            pa.array(ends.astype(np.int64), type=pa.int64()),
        ],
        names=["feature_space", "zarr_group", "start", "end"],
    )

    # Start with auto-generated columns
    uids = [make_uid() for _ in range(n_cells)]
    columns = {
        "uid": pa.array(uids, type=pa.string()),
        "dataset_uid": pa.array([dataset_record.uid] * n_cells, type=pa.string()),
        "gene_expression": pointer_struct,
    }

    # Add obs columns that match the schema
    available_obs_cols = [c for c in schema_fields if c in obs_df.columns]
    for col in available_obs_cols:
        series = obs_df[col]
        # Convert categorical to string for Arrow compatibility
        if hasattr(series, "cat"):
            series = series.astype(str)
        arrow_field = arrow_schema.field(col)
        columns[col] = pa.array(series.values, type=arrow_field.type)

    # Add None columns for schema fields not in obs
    for field_name in schema_fields:
        if field_name not in columns:
            arrow_field = arrow_schema.field(field_name)
            columns[field_name] = pa.nulls(n_cells, type=arrow_field.type)

    arrow_table = pa.table(columns, schema=arrow_schema)

    atlas.cell_table.add(arrow_table)
    print(f"    Inserted {n_cells:,} cell records")

    return n_cells


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a cellxgene census h5ad into a lancell atlas"
    )
    parser.add_argument("--h5ad", required=True, help="Path to .h5ad file")
    parser.add_argument("--atlas-dir", required=True, help="Path to atlas directory")
    args = parser.parse_args()

    h5ad_path = args.h5ad
    atlas_dir = args.atlas_dir

    print(f"Loading {h5ad_path} (backed mode)...")
    adata = ad.read_h5ad(h5ad_path, backed="r")

    if atlas_exists(atlas_dir):
        print(f"Opening existing atlas at {atlas_dir}")
        atlas = open_atlas(atlas_dir)
    else:
        print(f"Creating new atlas at {atlas_dir}")
        atlas = create_atlas(atlas_dir)

    print("Registering genes...")
    ensembl_to_uid = register_genes(atlas, adata)

    print("Ingesting data...")
    n_cells = ingest_backed(atlas, adata, h5ad_path, ensembl_to_uid)

    print(f"Done! Ingested {n_cells:,} cells from {Path(h5ad_path).name}")


if __name__ == "__main__":
    main()
