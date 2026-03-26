"""Ingest an entire CellxGene Census TileDB-SOMA store into ONE zarr group.

All ~44M cells share a single zarr group with a single feature remapper,
eliminating per-dataset partitioning overhead. This is the "regular" baseline
for benchmarking against the ragged (per-dataset) layout in ``ingest.py``.

Usage:
    python -m lancell_examples.cellxgene_census_tiledb.ingest_single_group \
        --soma-path ~/datasets/mus_musculus \
        --atlas-dir /path/to/atlas_single \
        [--batch-size 50000] \
        [--no-csc]

The obs `nnz` column is used to pre-allocate zarr arrays so data can be
streamed without holding the full matrix in memory.
"""

import argparse
import os
import time
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
CHUNK_SIZE = 40_960
SHARD_SIZE = 1024 * CHUNK_SIZE
BLOCKWISE_SIZE = 50_000


# ---------------------------------------------------------------------------
# Atlas helpers (identical to ingest.py)
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


# ---------------------------------------------------------------------------
# Gene registration (reused from ingest.py)
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
# Single-group streaming ingestion
# ---------------------------------------------------------------------------


def ingest_all(
    atlas: RaggedAtlas,
    experiment: tiledbsoma.Experiment,
    obs_df: pd.DataFrame,
    var_joinids: np.ndarray,
    joinid_to_uid: dict[int, str],
    *,
    no_csc: bool = False,
    blockwise_size: int = BLOCKWISE_SIZE,
) -> int:
    """Ingest the entire experiment into a single zarr group.

    Uses obs['nnz'] to pre-allocate zarr arrays, then streams blocks from
    tiledbsoma directly into the pre-allocated arrays.
    """
    n_cells = len(obs_df)
    zarr_group = make_uid()

    # Pre-compute total nnz from obs metadata
    total_nnz = int(obs_df["nnz"].sum())
    print(f"  {n_cells:,} cells, {total_nnz:,} total nnz -> zarr_group {zarr_group}")

    # Sort obs by soma_joinid so cell ordering matches blockwise iteration
    obs_df = obs_df.sort_values("soma_joinid").reset_index(drop=True)
    all_obs_joinids = obs_df["soma_joinid"].values

    # --- Pre-allocate zarr arrays ---
    chunk_shape = (CHUNK_SIZE,)
    shard_shape = (SHARD_SIZE,)

    group = atlas._root.create_group(zarr_group)
    csr_group = group.create_group("csr")
    layers_group = csr_group.create_group("layers")

    zarr_indices = csr_group.create_array(
        "indices",
        shape=(total_nnz,),
        dtype=np.uint32,
        chunks=chunk_shape,
        shards=shard_shape,
        compressors=BitpackingCodec(transform="delta"),
    )
    # Raw counts are float32 from tiledbsoma; not integer so no bitpacking on values
    zarr_values = layers_group.create_array(
        LAYER_NAME,
        shape=(total_nnz,),
        dtype=np.float32,
        chunks=chunk_shape,
        shards=shard_shape,
    )

    # --- Stream blocks from tiledbsoma, buffer per-shard, then flush ---
    query = experiment.axis_query(
        "RNA",
        obs_query=tiledbsoma.AxisQuery(coords=(all_obs_joinids,)),
    )
    sparse_read = query.X("raw")
    blockwise = sparse_read.blockwise(
        axis=0,
        size=blockwise_size,
        reindex_disable_on_axis=1,
    )

    # We'll build indptr incrementally
    # indptr[i] = offset in flat arrays where cell i's data starts
    indptr = np.empty(n_cells + 1, dtype=np.int64)
    indptr[0] = 0
    nnz_written = 0  # global nnz offset (includes flushed + buffered)
    cells_written = 0

    # Shard-aligned write buffer: accumulate data until we can write
    # complete shards, so each shard file is only opened once.
    _buf_indices: list[np.ndarray] = []
    _buf_values: list[np.ndarray] = []
    _buf_nnz = 0  # nnz currently sitting in the buffer
    _flush_offset = 0  # global nnz offset of the buffer start
    _shards_flushed = 0
    _total_shards = (total_nnz + SHARD_SIZE - 1) // SHARD_SIZE

    def _flush_complete_shards() -> None:
        """Write complete shards from the buffer to zarr."""
        nonlocal _buf_indices, _buf_values, _buf_nnz, _flush_offset, _shards_flushed

        if _buf_nnz < SHARD_SIZE:
            return

        all_idx = np.concatenate(_buf_indices)
        all_val = np.concatenate(_buf_values)

        n_complete = _buf_nnz // SHARD_SIZE
        write_end = n_complete * SHARD_SIZE

        t0 = time.perf_counter()
        zarr_indices[_flush_offset : _flush_offset + write_end] = all_idx[:write_end]
        zarr_values[_flush_offset : _flush_offset + write_end] = all_val[:write_end]
        elapsed = time.perf_counter() - t0

        _shards_flushed += n_complete
        print(
            f"    Flushed {n_complete} shard(s) "
            f"({_shards_flushed}/{_total_shards}) in {elapsed:.1f}s"
        )

        if write_end < _buf_nnz:
            _buf_indices = [all_idx[write_end:]]
            _buf_values = [all_val[write_end:]]
        else:
            _buf_indices = []
            _buf_values = []

        _flush_offset += write_end
        _buf_nnz -= write_end

    t_stream_start = time.perf_counter()
    last_progress = 0

    for block_csr, (_block_obs_joinids, _) in blockwise.scipy():
        block_n = block_csr.shape[0]
        # Reindex to contiguous var columns
        block_csr = block_csr[:, var_joinids]
        block_csr.sort_indices()

        block_nnz = block_csr.nnz
        if block_nnz > 0:
            _buf_indices.append(block_csr.indices.astype(np.uint32))
            _buf_values.append(block_csr.data)
            _buf_nnz += block_nnz

        # Fill indptr for this block's cells
        block_indptr = block_csr.indptr.astype(np.int64)
        indptr[cells_written + 1 : cells_written + block_n + 1] = block_indptr[1:] + nnz_written

        nnz_written += block_nnz
        cells_written += block_n

        _flush_complete_shards()

        # Progress every 100k cells
        if cells_written - last_progress >= 100_000:
            elapsed = time.perf_counter() - t_stream_start
            rate = cells_written / elapsed if elapsed > 0 else 0
            pct = 100.0 * cells_written / n_cells
            print(
                f"    [{pct:5.1f}%] {cells_written:,} / {n_cells:,} cells "
                f"({nnz_written:,} nnz) | {rate:,.0f} cells/s | "
                f"buf {_buf_nnz:,} nnz"
            )
            last_progress = cells_written

    # Flush any remaining partial shard
    if _buf_nnz > 0:
        t0 = time.perf_counter()
        remaining_idx = np.concatenate(_buf_indices) if len(_buf_indices) > 1 else _buf_indices[0]
        remaining_val = np.concatenate(_buf_values) if len(_buf_values) > 1 else _buf_values[0]
        zarr_indices[_flush_offset : _flush_offset + _buf_nnz] = remaining_idx
        zarr_values[_flush_offset : _flush_offset + _buf_nnz] = remaining_val
        elapsed = time.perf_counter() - t0
        print(f"    Flushed final partial shard ({_buf_nnz:,} nnz) in {elapsed:.1f}s")

    query.close()

    stream_elapsed = time.perf_counter() - t_stream_start
    print(
        f"  Streaming complete: {cells_written:,} cells, "
        f"{nnz_written:,} nnz in {stream_elapsed:.1f}s"
    )

    assert cells_written == n_cells, f"Expected {n_cells} cells, wrote {cells_written}"
    assert nnz_written == total_nnz, f"Expected {total_nnz} nnz, wrote {nnz_written}"

    starts = indptr[:-1]
    ends = indptr[1:]

    # --- Dataset record ---
    dataset_record = CensusDatasetRecord(
        uid=zarr_group,
        zarr_group=zarr_group,
        feature_space=FEATURE_SPACE,
        n_cells=n_cells,
        cellxgene_dataset_id="all",
    )
    dataset_arrow = pa.Table.from_pylist(
        [dataset_record.model_dump()],
        schema=CensusDatasetRecord.to_arrow_schema(),
    )
    atlas._dataset_table.add(dataset_arrow)

    # --- Dataset vars (single set of feature mappings) ---
    var_uids = [joinid_to_uid[int(jid)] for jid in var_joinids]
    var_pl = pl.DataFrame({"global_feature_uid": var_uids})
    atlas.add_or_reuse_layout(var_pl, zarr_group, FEATURE_SPACE)

    # --- Write cell records in batches to avoid huge Arrow allocations ---
    pointer_field: PointerFieldInfo | None = None
    for pf in atlas._pointer_fields.values():
        if pf.feature_space == FEATURE_SPACE:
            pointer_field = pf
            break

    arrow_schema = CellObs.to_arrow_schema()
    schema_fields = _schema_obs_fields(CellObs)

    print("  Writing cell records...")
    t_cells_start = time.perf_counter()
    CELL_BATCH = 500_000
    for batch_start in range(0, n_cells, CELL_BATCH):
        batch_end = min(batch_start + CELL_BATCH, n_cells)
        batch_n = batch_end - batch_start
        batch_obs = obs_df.iloc[batch_start:batch_end]

        pointer_struct = pa.StructArray.from_arrays(
            [
                pa.array([FEATURE_SPACE] * batch_n, type=pa.string()),
                pa.array([zarr_group] * batch_n, type=pa.string()),
                pa.array(starts[batch_start:batch_end], type=pa.int64()),
                pa.array(ends[batch_start:batch_end], type=pa.int64()),
                pa.array(
                    np.arange(batch_start, batch_end, dtype=np.int64),
                    type=pa.int64(),
                ),
            ],
            names=["feature_space", "zarr_group", "start", "end", "zarr_row"],
        )

        columns: dict[str, pa.Array] = {
            "uid": pa.array([make_uid() for _ in range(batch_n)], type=pa.string()),
            "dataset_uid": pa.array([zarr_group] * batch_n, type=pa.string()),
            pointer_field.field_name: pointer_struct,
        }

        for col in schema_fields:
            if col in batch_obs.columns:
                columns[col] = pa.array(
                    batch_obs[col].values,
                    type=arrow_schema.field(col).type,
                )

        for col in schema_fields:
            if col not in columns:
                columns[col] = pa.nulls(batch_n, type=arrow_schema.field(col).type)

        atlas.cell_table.add(pa.table(columns, schema=arrow_schema))

        pct = 100.0 * batch_end / n_cells
        elapsed = time.perf_counter() - t_cells_start
        print(f"    [{pct:5.1f}%] Cell records: {batch_end:,} / {n_cells:,} ({elapsed:.1f}s)")

    # --- Optionally build CSC ---
    if not no_csc:
        print("  Building CSC index...")
        t_csc = time.perf_counter()
        add_csc(
            atlas,
            zarr_group=zarr_group,
            feature_space=FEATURE_SPACE,
            layer_name=LAYER_NAME,
            chunk_size=CHUNK_SIZE,
            shard_size=SHARD_SIZE,
        )
        print(f"  CSC index built in {time.perf_counter() - t_csc:.1f}s")

    print(f"  Done: {n_cells:,} cells, {total_nnz:,} nnz in single group")
    return n_cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Ingest CellxGene Census TileDB-SOMA into a single-group lancell atlas"
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

    print(f"Opening TileDB-SOMA experiment at {soma_path}...")
    experiment = tiledbsoma.open(soma_path)

    # --- Read var ---
    print("Reading var (gene features)...")
    var_df = experiment.ms["RNA"]["var"].read().concat().to_pandas()
    var_joinids = var_df["soma_joinid"].values
    print(f"  {len(var_df):,} genes")

    # --- Read obs ---
    print("Reading obs metadata...")
    obs_df = experiment.obs.read().concat().to_pandas()
    print(f"  {len(obs_df):,} total cells")

    # --- Create atlas ---
    print(f"Creating new atlas at {atlas_dir}")
    atlas = create_atlas(atlas_dir)

    # --- Register genes ---
    print("Registering genes...")
    joinid_to_uid = register_genes(atlas, var_df)

    # --- Ingest everything into one group ---
    print("Ingesting all cells into single zarr group...")
    n = ingest_all(
        atlas,
        experiment,
        obs_df,
        var_joinids,
        joinid_to_uid,
        no_csc=args.no_csc,
        blockwise_size=args.batch_size,
    )

    experiment.close()
    print(f"\nDone! Ingested {n:,} cells into single zarr group")


if __name__ == "__main__":
    main()
