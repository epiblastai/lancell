"""Ingest a CellxGene Census TileDB-SOMA store into a lancell ragged atlas.

Reads batches of cells directly from the on-disk TileDB-SOMA experiment
(one zarr group per cellxgene dataset_id), ingesting only the raw counts layer.

Usage:
    python -m lancell_examples.cellxgene_census_tiledb.ingest \
        --soma-path ~/datasets/mus_musculus \
        --atlas-dir /path/to/atlas \
        [--batch-size 50000] \
        [--no-csc]

Each cellxgene dataset_id becomes a separate zarr group so that the atlas
mirrors the original per-study structure. The script is resumable: datasets
whose zarr group already exists in the atlas are skipped.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import obstore.store
import pandas as pd
import polars as pl
import pyarrow as pa
import tiledbsoma

from lancell.atlas import RaggedAtlas
from lancell.codecs.bitpacking import BitpackingCodec
from lancell.ingestion import add_csc
from lancell.obs_alignment import PointerFieldInfo, _schema_obs_fields
from lancell.schema import make_uid
from lancell_examples.cellxgene_census_tiledb.schema import (
    CellObs,
    CensusDatasetRecord,
    GeneFeatureSpace,
)

FEATURE_SPACE = "gene_expression"
LAYER_NAME = "counts"
CHUNK_SIZE = 5_000
SHARD_SIZE = 50_000_000
BLOCKWISE_SIZE = 50_000  # cells per tiledbsoma read block


# ---------------------------------------------------------------------------
# Atlas helpers (same pattern as the h5ad example)
# ---------------------------------------------------------------------------


def make_store(atlas_dir: str) -> obstore.store.ObjectStore:
    if atlas_dir.startswith("s3://"):
        from urllib.parse import urlparse

        parsed = urlparse(atlas_dir)
        bucket = parsed.netloc
        prefix = os.path.join(parsed.path.strip("/"), "zarr_store")
        region = os.environ.get("AWS_REGION")
        if not region:
            raise ValueError("AWS_REGION environment variable must be set for S3 access")
        return obstore.store.S3Store(bucket, prefix=prefix, region=region)
    zarr_path = Path(atlas_dir) / "zarr_store"
    zarr_path.mkdir(parents=True, exist_ok=True)
    return obstore.store.LocalStore(str(zarr_path))


def db_uri_for(atlas_dir: str) -> str:
    if atlas_dir.startswith("s3://"):
        return atlas_dir.rstrip("/") + "/lance_db"
    return str(Path(atlas_dir) / "lance_db")


def atlas_exists(atlas_dir: str) -> bool:
    if atlas_dir.startswith("s3://"):
        import lancedb

        db = lancedb.connect(db_uri_for(atlas_dir))
        return "cells" in db.list_tables().tables
    return (Path(atlas_dir) / "lance_db").exists()


def create_atlas(atlas_dir: str) -> RaggedAtlas:
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
    store = make_store(atlas_dir)
    return RaggedAtlas.open(
        db_uri=db_uri_for(atlas_dir),
        cell_table_name="cells",
        cell_schema=CellObs,
        dataset_table_name="datasets",
        store=store,
        registry_tables={FEATURE_SPACE: "gene_expression_registry"},
    )


# ---------------------------------------------------------------------------
# Gene registration
# ---------------------------------------------------------------------------


def register_genes(atlas: RaggedAtlas, var_df: pd.DataFrame) -> dict[int, str]:
    """Register genes from the census var DataFrame.

    Returns a mapping of soma_joinid -> registry uid for all genes.
    """
    registry_table = atlas._registry_tables[FEATURE_SPACE]
    existing_df = registry_table.search().select(["uid", "ensembl_id"]).to_polars()
    existing_ensembl_to_uid: dict[str, str] = dict(
        zip(existing_df["ensembl_id"].to_list(), existing_df["uid"].to_list(), strict=False)
    )

    joinid_to_uid: dict[int, str] = {}
    new_features: list[GeneFeatureSpace] = []

    for _, row in var_df.iterrows():
        ensembl_id = row["feature_id"]
        joinid = int(row["soma_joinid"])
        if ensembl_id in existing_ensembl_to_uid:
            joinid_to_uid[joinid] = existing_ensembl_to_uid[ensembl_id]
        elif ensembl_id not in {f.ensembl_id for f in new_features}:
            feature = GeneFeatureSpace(
                ensembl_id=ensembl_id,
                feature_name=str(row["feature_name"]),
                feature_type=str(row["feature_type"]),
                feature_length=int(row["feature_length"]),
            )
            new_features.append(feature)
            joinid_to_uid[joinid] = feature.uid
        else:
            # Already queued in new_features
            for f in new_features:
                if f.ensembl_id == ensembl_id:
                    joinid_to_uid[joinid] = f.uid
                    break

    if new_features:
        n_new = atlas.register_features(FEATURE_SPACE, new_features)
        print(
            f"  Registered {n_new} new genes "
            f"({len(var_df)} total, {len(existing_ensembl_to_uid)} already existed)"
        )
    else:
        print(f"  All {len(var_df)} genes already registered")

    return joinid_to_uid


# ---------------------------------------------------------------------------
# Per-dataset ingestion from TileDB-SOMA
# ---------------------------------------------------------------------------


def _get_existing_datasets(atlas: RaggedAtlas) -> set[str]:
    """Return the set of cellxgene_dataset_ids already ingested."""
    try:
        df = atlas._dataset_table.search().select(["cellxgene_dataset_id"]).to_polars()
        return set(df["cellxgene_dataset_id"].to_list())
    except Exception:
        return set()


def ingest_dataset(
    atlas: RaggedAtlas,
    experiment: tiledbsoma.Experiment,
    dataset_id: str,
    obs_joinids: np.ndarray,
    var_joinids: np.ndarray,
    joinid_to_uid: dict[int, str],
    no_csc: bool = False,
    blockwise_size: int = BLOCKWISE_SIZE,
) -> int:
    """Ingest all cells for one cellxgene dataset_id from the SOMA store."""
    n_cells = len(obs_joinids)
    zarr_group = make_uid()
    print(f"  Dataset {dataset_id}: {n_cells:,} cells -> zarr_group {zarr_group}")

    # --- Read obs metadata for these cells ---
    obs_table = experiment.obs.read(coords=(obs_joinids,)).concat().to_pandas()
    obs_table = obs_table.sort_values("soma_joinid").reset_index(drop=True)

    # --- Create zarr group and arrays ---
    group = atlas._root.create_group(zarr_group)
    csr_group = group.create_group("csr")
    layers_group = csr_group.create_group("layers")

    # We need to accumulate CSR data across blockwise reads, then write.
    # Read the X data for this dataset's cells in blocks.
    query = experiment.axis_query(
        "RNA",
        obs_query=tiledbsoma.AxisQuery(coords=(obs_joinids,)),
    )
    sparse_read = query.X("raw")
    # blockwise along axis=0 (obs), keep var soma_joinids as-is for column indices
    blockwise = sparse_read.blockwise(
        axis=0,
        size=blockwise_size,
        reindex_disable_on_axis=1,
    )

    all_indices = []
    all_data = []
    all_indptr = [np.array([0], dtype=np.int64)]
    running_nnz = 0

    for block_csr, (_block_obs_joinids, _) in blockwise.scipy():
        # block_csr is a CSR matrix with shape (block_n_cells, n_vars_full)
        # reindex columns to local var index (0-based contiguous)
        # var_joinids are the soma_joinids we care about
        block_csr = block_csr[:, var_joinids]
        block_csr.sort_indices()

        all_indices.append(block_csr.indices.astype(np.uint32))
        all_data.append(block_csr.data)
        # Shift indptr by running offset
        block_indptr = block_csr.indptr.astype(np.int64)
        all_indptr.append(block_indptr[1:] + running_nnz)
        running_nnz += block_csr.nnz

    query.close()

    flat_indices = np.concatenate(all_indices) if all_indices else np.array([], dtype=np.uint32)
    flat_data = np.concatenate(all_data) if all_data else np.array([], dtype=np.float32)
    indptr = np.concatenate(all_indptr)
    nnz = len(flat_indices)

    assert len(indptr) == n_cells + 1, f"indptr length {len(indptr)} != n_cells+1 {n_cells + 1}"

    # Write zarr arrays
    chunk_shape = (CHUNK_SIZE,)
    shard_shape = (SHARD_SIZE,)

    if nnz > 0:
        zarr_indices = csr_group.create_array(
            "indices",
            shape=(nnz,),
            dtype=np.uint32,
            chunks=chunk_shape,
            shards=shard_shape,
            compressors=BitpackingCodec(transform="delta"),
        )
        # Use bitpacking for integer data
        layer_kwargs = {}
        if flat_data.dtype in (np.int32, np.int64, np.uint32, np.uint64):
            layer_kwargs["compressors"] = BitpackingCodec(transform="none")
        zarr_values = layers_group.create_array(
            LAYER_NAME,
            shape=(nnz,),
            dtype=flat_data.dtype,
            chunks=chunk_shape,
            shards=shard_shape,
            **layer_kwargs,
        )

        # Write in shard-sized batches
        written = 0
        while written < nnz:
            end = min(written + shard_shape[0], nnz)
            zarr_indices[written:end] = flat_indices[written:end]
            zarr_values[written:end] = flat_data[written:end]
            written = end
    else:
        # Empty dataset - create zero-length arrays
        csr_group.create_array(
            "indices",
            shape=(0,),
            dtype=np.uint32,
            chunks=(1,),
            shards=(1,),
        )
        layers_group.create_array(
            LAYER_NAME,
            shape=(0,),
            dtype=np.float32,
            chunks=(1,),
            shards=(1,),
        )

    starts = indptr[:-1]
    ends = indptr[1:]

    # --- Write dataset record ---
    dataset_record = CensusDatasetRecord(
        uid=zarr_group,
        zarr_group=zarr_group,
        feature_space=FEATURE_SPACE,
        n_cells=n_cells,
        cellxgene_dataset_id=dataset_id,
    )
    dataset_arrow = pa.Table.from_pylist(
        [dataset_record.model_dump()],
        schema=CensusDatasetRecord.to_arrow_schema(),
    )
    atlas._dataset_table.add(dataset_arrow)

    # --- Write dataset vars (feature mapping) ---
    var_uids = [joinid_to_uid[int(jid)] for jid in var_joinids]
    var_pl = pl.DataFrame({"global_feature_uid": var_uids})
    atlas.add_or_reuse_layout(var_pl, zarr_group, FEATURE_SPACE)

    # --- Write cell records ---
    pointer_field: PointerFieldInfo | None = None
    for pf in atlas._pointer_fields.values():
        if pf.feature_space == FEATURE_SPACE:
            pointer_field = pf
            break

    arrow_schema = CellObs.to_arrow_schema()
    schema_fields = _schema_obs_fields(CellObs)

    pointer_struct = pa.StructArray.from_arrays(
        [
            pa.array([FEATURE_SPACE] * n_cells, type=pa.string()),
            pa.array([zarr_group] * n_cells, type=pa.string()),
            pa.array(starts.astype(np.int64), type=pa.int64()),
            pa.array(ends.astype(np.int64), type=pa.int64()),
            pa.array(np.arange(n_cells, dtype=np.int64), type=pa.int64()),
        ],
        names=["feature_space", "zarr_group", "start", "end", "zarr_row"],
    )

    columns: dict[str, pa.Array] = {
        "uid": pa.array([make_uid() for _ in range(n_cells)], type=pa.string()),
        "dataset_uid": pa.array([zarr_group] * n_cells, type=pa.string()),
        pointer_field.field_name: pointer_struct,
    }

    # Map obs columns from the census to the schema
    for col in schema_fields:
        if col in obs_table.columns:
            values = obs_table[col].values
            target_type = arrow_schema.field(col).type
            columns[col] = pa.array(values, type=target_type)

    # Fill any missing schema fields with nulls
    for col in schema_fields:
        if col not in columns:
            columns[col] = pa.nulls(n_cells, type=arrow_schema.field(col).type)

    arrow_table = pa.table(columns, schema=arrow_schema)
    atlas.cell_table.add(arrow_table)

    # --- Optionally build CSC index ---
    if not no_csc and nnz > 0:
        add_csc(
            atlas,
            zarr_group=zarr_group,
            feature_space=FEATURE_SPACE,
            layer_name=LAYER_NAME,
            chunk_size=CHUNK_SIZE,
            shard_size=SHARD_SIZE,
        )

    print(f"    Inserted {n_cells:,} cells, {nnz:,} nnz")
    return n_cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Ingest CellxGene Census TileDB-SOMA store into a lancell atlas"
    )
    parser.add_argument(
        "--soma-path",
        required=True,
        help="Path to the local TileDB-SOMA experiment (e.g. ~/datasets/mus_musculus)",
    )
    parser.add_argument("--atlas-dir", required=True, help="Path to atlas directory")
    parser.add_argument("--no-csc", action="store_true", help="Skip adding CSC layout")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BLOCKWISE_SIZE,
        help=f"Cells per tiledbsoma read block (default: {BLOCKWISE_SIZE})",
    )
    args = parser.parse_args()

    soma_path = os.path.expanduser(args.soma_path)
    atlas_dir = args.atlas_dir
    blockwise_size = args.batch_size

    print(f"Opening TileDB-SOMA experiment at {soma_path}...")
    experiment = tiledbsoma.open(soma_path)

    # --- Read var (genes) ---
    print("Reading var (gene features)...")
    var_df = experiment.ms["RNA"]["var"].read().concat().to_pandas()
    var_joinids = var_df["soma_joinid"].values
    n_genes = len(var_df)
    print(f"  {n_genes:,} genes")

    # --- Read obs to partition by dataset_id ---
    print("Reading obs metadata...")
    obs_df = experiment.obs.read().concat().to_pandas()
    n_total_cells = len(obs_df)
    print(f"  {n_total_cells:,} total cells")

    dataset_groups = obs_df.groupby("dataset_id")["soma_joinid"].apply(lambda x: np.sort(x.values))
    n_datasets = len(dataset_groups)
    print(f"  {n_datasets} datasets")

    # --- Create or open atlas ---
    if atlas_exists(atlas_dir):
        print(f"Opening existing atlas at {atlas_dir}")
        atlas = open_atlas(atlas_dir)
    else:
        print(f"Creating new atlas at {atlas_dir}")
        atlas = create_atlas(atlas_dir)

    # --- Register genes ---
    print("Registering genes...")
    joinid_to_uid = register_genes(atlas, var_df)

    # --- Check which datasets are already ingested ---
    existing_datasets = _get_existing_datasets(atlas)
    if existing_datasets:
        print(f"  {len(existing_datasets)} datasets already ingested, will skip")

    # --- Ingest per dataset ---
    total_ingested = 0
    for i, (dataset_id, obs_joinids) in enumerate(dataset_groups.items()):
        if dataset_id in existing_datasets:
            print(f"[{i + 1}/{n_datasets}] Skipping {dataset_id} (already ingested)")
            continue

        print(f"[{i + 1}/{n_datasets}] Ingesting {dataset_id}...")
        n = ingest_dataset(
            atlas,
            experiment,
            dataset_id,
            obs_joinids,
            var_joinids,
            joinid_to_uid,
            no_csc=args.no_csc,
            blockwise_size=blockwise_size,
        )
        total_ingested += n

    experiment.close()
    print(f"\nDone! Ingested {total_ingested:,} cells from {n_datasets} datasets")


if __name__ == "__main__":
    main()
