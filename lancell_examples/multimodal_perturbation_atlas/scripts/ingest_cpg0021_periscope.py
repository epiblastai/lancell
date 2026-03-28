"""Ingest cpg0021-periscope (Ramezani et al. 2025) into a RaggedAtlas.

Genome-wide CRISPRko optical pooled screen (Cell Painting) in HeLa cells.
Two-stage streaming pipeline:

  Stage 1 — Feature buffering
      Stream CSV.gz shards through a process pool into a temporary dense zarr
      array.  Collect lightweight per-cell metadata (site, coordinates, obs
      fields) in memory.  Discard cells with invalid coordinates.

  Stage 2 — Site-grouped tiling + final write
      Group cell metadata by imaging site.  For each site: download the
      5-channel TIFF, crop tiles, read back buffered features via BatchArray,
      and accumulate into shard-sized batches.  When a batch is full, write
      features + tiles to the final zarr groups and append cell records to
      LanceDB.

Design rationale:
  - Features CSVs are per-guide, but tiles are per-site.  We cannot group
    by site during feature streaming.
  - 15M cells × 3,766 features won't fit in memory (~210 GB float32).
  - The temp zarr acts as a random-access buffer; BatchArray.read_ranges
    provides efficient batched reads from sharded zarr.
  - Each site image is downloaded exactly once; tiles are cropped instantly.

Prerequisites:
  - Feature CSV.gz shards in --features-dir
  - GeneticPerturbationSchema.parquet, PublicationSchema.parquet,
    ImageFeatureSchema.parquet, publication.json in --data-dir

Usage:
    python -m lancell_examples.multimodal_perturbation_atlas.scripts.ingest_cpg0021_periscope \\
        --atlas-path /tmp/atlas/perturbation_atlas \\
        --features-dir /tmp/geo_agent/cpg0021-periscope/features_full \\
        --data-dir /tmp/geo_agent/cpg0021-periscope \\
        [--max-per-shard 1000] [--csv-workers 8] [--flush-every 10000]
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import anndata as ad
import lancedb
import numpy as np
import obstore
import obstore.store
import pandas as pd
import pyarrow as pa
import tifffile
import zarr
from obstore.store import S3Store
from tqdm import tqdm

from lancell.atlas import RaggedAtlas, create_or_open_atlas
from lancell.batch_array import BatchArray
from lancell.group_specs import PointerKind
from lancell.ingestion import write_feature_layout
from lancell.obs_alignment import _schema_obs_fields
from lancell.schema import make_uid

from lancell_examples.multimodal_perturbation_atlas.schema import (
    CellIndex,
    DatasetSchema,
    GeneticPerturbationSchema,
    ImageFeatureSchema,
    PublicationSchema,
    PublicationSectionSchema,
    REGISTRY_SCHEMAS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCESSION = "cpg0021-periscope"
FEATURE_SPACE = "image_features"
TILE_FEATURE_SPACE = "image_tiles"

CHANNEL_NAMES = ["DNA", "ER", "Mito", "Phalloidin", "WGA"]
N_CHANNELS = len(CHANNEL_NAMES)
TILE_SIZE = 72

# CSV column names
META_SITE = "Metadata_Foci_site"
META_PLATE = "Metadata_Foci_plate"
META_WELL = "Metadata_Foci_well"
META_SITE_LOC = "Metadata_Foci_site_location"
META_OBJ_NUM = "Metadata_Cells_ObjectNumber"
CENTER_X = "Cells_AreaShape_Center_X"
CENTER_Y = "Cells_AreaShape_Center_Y"
ALIGN_X = "Align_Xshift_ConA"
ALIGN_Y = "Align_Yshift_ConA"

CONTROL_GENE_CODE = "nontargeting"
FEATURE_PREFIXES = ("Nuclei", "Cells", "Cytoplasm")


# ===================================================================
# Stage 1 — Feature buffering to temp zarr
# ===================================================================


def discover_feature_columns(sample_csv: Path) -> list[str]:
    """Read header of a sample CSV to discover CellProfiler feature columns."""
    header = pd.read_csv(sample_csv, nrows=0).columns.tolist()
    return [
        c for c in header
        if c.startswith(FEATURE_PREFIXES) and not c.startswith("Metadata_")
    ]


def process_shard(
    csv_path: Path,
    feature_columns: list[str],
    max_rows: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Read one CSV shard → (feature_matrix, metadata_df).

    Drops cells with invalid coordinates.
    """
    df = pd.read_csv(csv_path, nrows=max_rows)
    if df.empty:
        return np.empty((0, len(feature_columns)), dtype=np.float32), pd.DataFrame()

    # Parse guide/gene from filename
    fname = csv_path.name
    parts = fname.split("__")[-1].removesuffix(".csv.gz")
    guide_seq = parts.split("_", 1)[0]
    gene_name = parts.split("_", 1)[1]

    # Compute adjusted coordinates and filter invalid
    cx = df[CENTER_X].values + df.get(ALIGN_X, pd.Series(0, index=df.index)).values
    cy = df[CENTER_Y].values + df.get(ALIGN_Y, pd.Series(0, index=df.index)).values
    valid = np.isfinite(cx) & np.isfinite(cy)
    df = df[valid].reset_index(drop=True)
    cx = cx[valid]
    cy = cy[valid]

    if df.empty:
        return np.empty((0, len(feature_columns)), dtype=np.float32), pd.DataFrame()

    # Extract features — some shards have merge-artifact columns (e.g. _x suffix);
    # fill missing columns with 0.0 so the output width is always n_features.
    present = [c for c in feature_columns if c in df.columns]
    feat = np.zeros((len(df), len(feature_columns)), dtype=np.float32)
    if present:
        col_idx = [feature_columns.index(c) for c in present]
        feat[:, col_idx] = df[present].values.astype(np.float32)
    bad = ~np.isfinite(feat)
    if bad.any():
        feat[bad] = 0.0

    meta = pd.DataFrame({
        "original_cell_id": (
            df[META_PLATE].astype(str) + "_" +
            df[META_WELL].astype(str) + "_" +
            df[META_SITE_LOC].astype(int).astype(str) + "_" +
            df[META_OBJ_NUM].astype(int).astype(str)
        ),
        "metadata_site": df[META_SITE].astype(str),
        "center_x": cx,
        "center_y": cy,
        "guide_sequence": guide_seq,
        "gene_symbol": gene_name,
        "batch_id": df[META_PLATE].astype(str),
        "well_position": df[META_WELL].astype(str),
    })
    return feat, meta


def _count_shard(csv_path: Path, max_rows: int) -> int:
    """Count cells with valid coordinates in a CSV shard (for parallel use)."""
    df = pd.read_csv(csv_path, usecols=[CENTER_X], nrows=max_rows)
    return int(np.isfinite(df[CENTER_X].values).sum())


def buffer_features_to_temp_zarr(
    features_dir: Path,
    feature_columns: list[str],
    max_per_shard: int,
    csv_workers: int,
    total_cells: int | None = None,
) -> tuple[Path, pd.DataFrame, int]:
    """Stream CSV shards into a temp zarr, return (temp_dir, metadata, n_features).

    The temp zarr is at ``{temp_dir}/features.zarr`` as a 2D array
    (n_cells, n_features).  The metadata DataFrame has a ``temp_position``
    column with the row index into this array.
    """
    csv_paths = sorted(features_dir.glob("*.csv.gz"))
    assert csv_paths, f"No CSV.gz files in {features_dir}"
    n_features = len(feature_columns)

    if total_cells is None:
        # Count cells per shard in parallel
        print(f"  Counting cells across {len(csv_paths):,} shards ({csv_workers} workers)...")
        with ProcessPoolExecutor(max_workers=csv_workers) as executor:
            shard_counts = list(tqdm(
                executor.map(_count_shard, csv_paths, [max_per_shard] * len(csv_paths)),
                total=len(csv_paths),
                desc="Counting",
                leave=False,
            ))
        total_cells = sum(shard_counts)
    print(f"  Cells with valid coordinates: {total_cells:,}")

    # Create temp zarr (use ObjectStore backend so BatchArray can use it)
    temp_dir = Path(tempfile.mkdtemp(prefix="cpg0021_buffer_"))
    zarr_path = temp_dir / "features.zarr"
    zarr_path.mkdir()
    chunk_rows = max(1, 40_960 // n_features)
    shard_rows = max(1, (1024 * 40_960) // n_features)
    shard_rows = max(chunk_rows, (shard_rows // chunk_rows) * chunk_rows)

    store = zarr.storage.ObjectStore(obstore.store.LocalStore(str(zarr_path)))
    zarr_arr = zarr.create_array(
        store,
        shape=(total_cells, n_features),
        dtype=np.float32,
        chunks=(chunk_rows, n_features),
        shards=(shard_rows, n_features),
    )

    # Process shards in parallel, write to zarr in order as results arrive.
    # Uses a sliding window of futures so memory holds at most ~csv_workers
    # shard results at a time.  Rows are accumulated in a memory buffer and
    # flushed to zarr only when a full shard is ready, so each shard is
    # written exactly once (avoids costly read-modify-write cycles).
    print(f"  Streaming {len(csv_paths):,} shards ({csv_workers} workers)...")
    offset = 0
    all_meta: list[pd.DataFrame] = []

    # Shard-aligned write buffer (~168 MB for 3766 features)
    shard_buf = np.empty((shard_rows, n_features), dtype=np.float32)
    buf_count = 0  # rows currently in shard_buf

    def _flush_buf() -> None:
        nonlocal offset, buf_count
        if buf_count == 0:
            return
        zarr_arr[offset : offset + buf_count] = shard_buf[:buf_count]
        offset += buf_count
        buf_count = 0

    with ProcessPoolExecutor(max_workers=csv_workers) as executor:
        from collections import deque

        window: deque[tuple[int, "Future"]] = deque()
        submitted = 0
        pbar = tqdm(total=len(csv_paths), desc="Processing")

        # Fill initial window
        for i in range(min(csv_workers * 2, len(csv_paths))):
            fut = executor.submit(process_shard, csv_paths[i], feature_columns, max_per_shard)
            window.append((i, fut))
            submitted = i + 1

        consumed = 0
        while consumed < len(csv_paths):
            # Pop the next in-order result
            idx, fut = window.popleft()
            assert idx == consumed
            feat, meta = fut.result()
            consumed += 1
            pbar.update(1)

            # Submit next shard to keep window full
            if submitted < len(csv_paths):
                nfut = executor.submit(
                    process_shard, csv_paths[submitted], feature_columns, max_per_shard
                )
                window.append((submitted, nfut))
                submitted += 1

            if len(meta) == 0:
                continue

            # Assign temp_position before buffering (based on current offset + buf_count)
            n = feat.shape[0]
            meta["temp_position"] = np.arange(
                offset + buf_count, offset + buf_count + n, dtype=np.int64
            )
            all_meta.append(meta)

            # Copy into shard buffer, flushing whenever it fills up
            src_offset = 0
            while src_offset < n:
                space = shard_rows - buf_count
                take = min(space, n - src_offset)
                shard_buf[buf_count : buf_count + take] = feat[src_offset : src_offset + take]
                buf_count += take
                src_offset += take
                if buf_count == shard_rows:
                    _flush_buf()

            del feat  # free immediately

        # Flush any remaining rows in the buffer
        _flush_buf()
        pbar.close()

    meta_df = pd.concat(all_meta, ignore_index=True)
    meta_df.to_parquet(temp_dir / "meta.parquet", index=False)
    print(f"  Buffered {offset:,} × {n_features} to {zarr_path}")
    print(f"  Metadata cached to {temp_dir / 'meta.parquet'}")
    print(f"  To resume from this point: --resume-dir {temp_dir}")
    return temp_dir, meta_df, n_features


# ===================================================================
# Stage 2 — Site-grouped tiling + final write
# ===================================================================


def construct_s3_keys(metadata_site: str) -> list[str]:
    """Construct S3 keys for the 5 channel TIFFs of a site."""
    base = "cpg0021-periscope/broad/images/"
    if "CP257" in metadata_site:
        experiment_dir = "20210422_6W_CP257"
    elif "CP228" in metadata_site:
        experiment_dir = "20210124_6W_CP228"
    else:
        raise ValueError(f"Unknown experiment in site: {metadata_site}")
    plate, well, site = metadata_site.split("-")
    well_prefix = f"{base}{experiment_dir}/images_corrected_cropped/{plate}_{well}"
    return [
        f"{well_prefix}/Corr{ch}/Corr{ch}_Site_{site}.tiff"
        for ch in CHANNEL_NAMES
    ]


async def download_site_images(store: S3Store, site_name: str) -> np.ndarray:
    """Download 5 channel TIFFs for a site → stacked C×H×W uint16 array."""
    keys = construct_s3_keys(site_name)
    results = await asyncio.gather(*[obstore.get_async(store, k) for k in keys])
    channels = []
    for r in results:
        data = await r.bytes_async()
        channels.append(tifffile.imread(io.BytesIO(bytes(data))))
    return np.stack(channels)


def crop_tile(image: np.ndarray, cx: float, cy: float) -> np.ndarray | None:
    """Crop TILE_SIZE × TILE_SIZE from C×H×W image. None if OOB."""
    half = TILE_SIZE // 2
    x0, y0 = int(cx) - half, int(cy) - half
    x1, y1 = x0 + TILE_SIZE, y0 + TILE_SIZE
    if x0 < 0 or y0 < 0 or x1 > image.shape[2] or y1 > image.shape[1]:
        return None
    return image[:, y0:y1, x0:x1]


def build_obs_batch(
    meta_batch: pd.DataFrame,
    guide_to_uid: dict[str, str],
) -> pd.DataFrame:
    """Build schema-aligned obs DataFrame from streaming metadata."""
    n = len(meta_batch)
    obs = pd.DataFrame(index=range(n))

    obs["assay"] = "high content screen"
    obs["organism"] = "Homo sapiens"
    obs["cell_line"] = "HeLa"
    obs["cell_type"] = None
    obs["development_stage"] = None
    obs["disease"] = None
    obs["tissue"] = None
    obs["donor_uid"] = None
    obs["days_in_vitro"] = pd.array([pd.NA] * n, dtype=pd.Float64Dtype())
    obs["replicate"] = pd.array([pd.NA] * n, dtype=pd.Int64Dtype())
    obs["batch_id"] = meta_batch["batch_id"].values
    obs["well_position"] = meta_batch["well_position"].values

    obs["additional_metadata"] = [
        json.dumps({"original_cell_id": cid, "site": site})
        for cid, site in zip(meta_batch["original_cell_id"], meta_batch["metadata_site"])
    ]

    is_control = meta_batch["gene_symbol"].values == CONTROL_GENE_CODE
    obs["is_negative_control"] = is_control
    obs["negative_control_type"] = np.where(is_control, "nontargeting", None)

    uids, types, conc, dur = [], [], [], []
    for guide in meta_batch["guide_sequence"]:
        uid = guide_to_uid.get(guide)
        if uid:
            uids.append([uid])
            types.append(["genetic_perturbation"])
            conc.append([-1.0])
            dur.append([-1.0])
        else:
            uids.append(None)
            types.append(None)
            conc.append(None)
            dur.append(None)

    obs["perturbation_uids"] = uids
    obs["perturbation_types"] = types
    obs["perturbation_concentrations_um"] = conc
    obs["perturbation_durations_hr"] = dur
    obs["perturbation_additional_metadata"] = None

    obs = CellIndex.compute_auto_fields(obs)
    return obs


def flush_batch(
    atlas: RaggedAtlas,
    batch_meta: pd.DataFrame,
    batch_features: np.ndarray,
    batch_tiles: np.ndarray,
    feat_zarr_arr: zarr.Array,
    tile_zarr_arr: zarr.Array,
    feat_offset: int,
    tile_offset: int,
    feat_zarr_group: str,
    tile_zarr_group: str,
    guide_to_uid: dict[str, str],
) -> tuple[int, int]:
    """Write one batch of features + tiles to final zarr and insert cell records.

    Returns (new_feat_offset, new_tile_offset).
    """
    n = len(batch_meta)

    # Write features
    feat_zarr_arr[feat_offset : feat_offset + n] = batch_features

    # Write tiles
    tile_zarr_arr[tile_offset : tile_offset + n] = batch_tiles

    # Build cell records
    arrow_schema = atlas._cell_schema.to_arrow_schema()
    schema_fields = _schema_obs_fields(atlas._cell_schema)
    obs = build_obs_batch(batch_meta.reset_index(drop=True), guide_to_uid)

    columns: dict[str, pa.Array] = {
        "uid": pa.array([make_uid() for _ in range(n)], type=pa.string()),
        "dataset_uid": pa.array([feat_zarr_group] * n, type=pa.string()),
    }

    for pf_name, pf in atlas._pointer_fields.items():
        if pf.feature_space == FEATURE_SPACE:
            columns[pf_name] = pa.StructArray.from_arrays(
                [
                    pa.array([FEATURE_SPACE] * n, type=pa.string()),
                    pa.array([feat_zarr_group] * n, type=pa.string()),
                    pa.array(
                        np.arange(feat_offset, feat_offset + n, dtype=np.int64),
                        type=pa.int64(),
                    ),
                ],
                names=["feature_space", "zarr_group", "position"],
            )
        elif pf.feature_space == TILE_FEATURE_SPACE:
            columns[pf_name] = pa.StructArray.from_arrays(
                [
                    pa.array([TILE_FEATURE_SPACE] * n, type=pa.string()),
                    pa.array([tile_zarr_group] * n, type=pa.string()),
                    pa.array(
                        np.arange(tile_offset, tile_offset + n, dtype=np.int64),
                        type=pa.int64(),
                    ),
                ],
                names=["feature_space", "zarr_group", "position"],
            )
        elif pf.pointer_kind is PointerKind.SPARSE:
            columns[pf_name] = pa.StructArray.from_arrays(
                [
                    pa.array([""] * n, type=pa.string()),
                    pa.array([""] * n, type=pa.string()),
                    pa.array([0] * n, type=pa.int64()),
                    pa.array([0] * n, type=pa.int64()),
                    pa.array([0] * n, type=pa.int64()),
                ],
                names=["feature_space", "zarr_group", "start", "end", "zarr_row"],
            )
        else:
            columns[pf_name] = pa.StructArray.from_arrays(
                [
                    pa.array([""] * n, type=pa.string()),
                    pa.array([""] * n, type=pa.string()),
                    pa.array([0] * n, type=pa.int64()),
                ],
                names=["feature_space", "zarr_group", "position"],
            )

    for col in schema_fields:
        if col in obs.columns:
            columns[col] = pa.array(obs[col].values, type=arrow_schema.field(col).type)
    for col in schema_fields:
        if col not in columns:
            columns[col] = pa.nulls(n, type=arrow_schema.field(col).type)

    atlas.cell_table.add(pa.table(columns, schema=arrow_schema))
    return feat_offset + n, tile_offset + n


def run_stage2(
    atlas: RaggedAtlas,
    meta_df: pd.DataFrame,
    temp_zarr_path: Path,
    n_features: int,
    guide_to_uid: dict[str, str],
    publication_uid: str,
    data_dir: Path,
    flush_every: int,
    prefetch_ahead: int = 25,
) -> tuple[int, int]:
    """Site-grouped tiling + final writes. Returns (n_cells, n_tiles)."""

    total_cells = len(meta_df)
    pub_json = json.loads((data_dir / "publication.json").read_text())

    # Create final zarr groups
    feat_ds_uid = make_uid()
    tile_ds_uid = make_uid()

    # Pre-allocate final feature zarr
    feat_group = atlas._root.create_group(feat_ds_uid)
    layers_group = feat_group.create_group("layers")
    chunk_rows = max(1, 40_960 // n_features)
    shard_rows = max(1, (1024 * 40_960) // n_features)
    shard_rows = max(chunk_rows, (shard_rows // chunk_rows) * chunk_rows)
    feat_zarr = layers_group.create_array(
        "ctrl_standardized",
        shape=(total_cells, n_features),
        dtype=np.float32,
        chunks=(chunk_rows, n_features),
        shards=(shard_rows, n_features),
    )

    # Pre-allocate final tile zarr
    tile_group = atlas._root.create_group(tile_ds_uid)
    tile_chunk = (4, N_CHANNELS, TILE_SIZE, TILE_SIZE)
    tile_shard = (min(4096, total_cells), N_CHANNELS, TILE_SIZE, TILE_SIZE)
    tile_zarr = tile_group.create_array(
        "data",
        shape=(total_cells, N_CHANNELS, TILE_SIZE, TILE_SIZE),
        dtype=np.uint16,
        chunks=tile_chunk,
        shards=tile_shard,
    )

    # Write feature layout
    var_csv = data_dir / ACCESSION / f"{FEATURE_SPACE}_standardized_var.csv"
    if var_csv.exists():
        var_df = pd.read_csv(var_csv, index_col=0)
    else:
        img_df = pd.read_parquet(data_dir / "ImageFeatureSchema.parquet")
        var_df = pd.DataFrame(
            {"global_feature_uid": img_df["uid"].values},
            index=img_df["feature_name"].values,
        )
        var_df.index.name = "feature_name"
    dummy_adata = ad.AnnData(X=np.empty((1, len(var_df)), dtype=np.float32), var=var_df)
    dummy_adata.var.index = dummy_adata.var.index.astype(str)
    write_feature_layout(atlas, dummy_adata, FEATURE_SPACE, feat_ds_uid, feat_ds_uid)

    # Register dataset records
    for ds_uid, fs, nc in [(feat_ds_uid, FEATURE_SPACE, total_cells),
                           (tile_ds_uid, TILE_FEATURE_SPACE, total_cells)]:
        ds = DatasetSchema(
            uid=ds_uid, zarr_group=ds_uid, feature_space=fs, n_cells=nc,
            publication_uid=publication_uid,
            accession_database="cellpainting-gallery", accession_id=ACCESSION,
            dataset_description=pub_json.get("title"),
            organism=["Homo sapiens"], tissue=None, cell_line=["HeLa"], disease=None,
        )
        atlas._dataset_table.add(
            pa.Table.from_pylist([ds.model_dump()], schema=DatasetSchema.to_arrow_schema())
        )

    # Open temp zarr for batched reads (obstore-backed for BatchArray)
    temp_store = zarr.storage.ObjectStore(obstore.store.LocalStore(str(temp_zarr_path)))
    temp_arr = zarr.open_array(temp_store, mode="r")
    batch_reader = BatchArray.from_array(temp_arr)

    # Group by site
    site_groups = list(meta_df.groupby("metadata_site"))
    print(f"  Sites: {len(site_groups):,}, cells: {total_cells:,}")

    s3_store = S3Store("cellpainting-gallery", region="us-east-1", skip_signature=True)
    loop = asyncio.new_event_loop()

    site_names = [name for name, _ in site_groups]
    prefetch_tasks: dict[str, asyncio.Task] = {}
    for name in site_names[:prefetch_ahead]:
        prefetch_tasks[name] = loop.create_task(download_site_images(s3_store, name))

    # Accumulation buffers
    acc_meta: list[pd.DataFrame] = []
    acc_features: list[np.ndarray] = []
    acc_tiles: list[np.ndarray] = []
    acc_count = 0

    feat_offset = 0
    tile_offset = 0
    total_written = 0
    skipped_oob = 0

    for i, (site_name, site_df) in enumerate(tqdm(site_groups, desc="Tiling")):
        # Download site image
        stacked = loop.run_until_complete(prefetch_tasks.pop(site_name))

        # Prefetch next
        next_idx = i + prefetch_ahead
        if next_idx < len(site_names):
            prefetch_tasks[site_names[next_idx]] = loop.create_task(
                download_site_images(s3_store, site_names[next_idx])
            )

        # Crop tiles, filter OOB
        valid_rows = []
        site_tiles = []
        for _, row in site_df.iterrows():
            tile = crop_tile(stacked, row["center_x"], row["center_y"])
            if tile is None:
                skipped_oob += 1
                continue
            valid_rows.append(row)
            site_tiles.append(tile)

        if not valid_rows:
            continue

        site_meta = pd.DataFrame(valid_rows)
        site_tile_arr = np.stack(site_tiles)

        # Read features from temp zarr for these cells
        positions = site_meta["temp_position"].values.astype(np.int64)
        starts = positions
        ends = positions + 1
        flat_data, lengths = batch_reader.read_ranges(starts, ends)
        site_features = flat_data.reshape(len(positions), n_features)

        # Accumulate
        acc_meta.append(site_meta)
        acc_features.append(site_features)
        acc_tiles.append(site_tile_arr)
        acc_count += len(site_meta)

        # Flush when we have enough
        if acc_count >= flush_every:
            batch_meta = pd.concat(acc_meta, ignore_index=True)
            batch_features = np.concatenate(acc_features)
            batch_tiles = np.concatenate(acc_tiles)

            feat_offset, tile_offset = flush_batch(
                atlas, batch_meta, batch_features, batch_tiles,
                feat_zarr, tile_zarr, feat_offset, tile_offset,
                feat_ds_uid, tile_ds_uid, guide_to_uid,
            )
            total_written += len(batch_meta)
            print(f"  Flushed {total_written:,} / {total_cells:,} cells")
            acc_meta, acc_features, acc_tiles, acc_count = [], [], [], 0

    loop.close()

    # Final flush
    if acc_count > 0:
        batch_meta = pd.concat(acc_meta, ignore_index=True)
        batch_features = np.concatenate(acc_features)
        batch_tiles = np.concatenate(acc_tiles)
        feat_offset, tile_offset = flush_batch(
            atlas, batch_meta, batch_features, batch_tiles,
            feat_zarr, tile_zarr, feat_offset, tile_offset,
            feat_ds_uid, tile_ds_uid, guide_to_uid,
        )
        total_written += len(batch_meta)

    # Trim zarr arrays if cells were skipped due to OOB
    if feat_offset < total_cells:
        feat_zarr.resize((feat_offset, n_features))
        tile_zarr.resize((tile_offset, N_CHANNELS, TILE_SIZE, TILE_SIZE))
        # Update dataset records
        db = lancedb.connect(str(atlas._db_uri))
        ds_table = db.open_table("datasets")
        ds_table.update(where=f"uid = '{feat_ds_uid}'", values={"n_cells": feat_offset})
        ds_table.update(where=f"uid = '{tile_ds_uid}'", values={"n_cells": tile_offset})

    print(f"  Written: {total_written:,} cells, skipped {skipped_oob} OOB")
    return total_written, skipped_oob


# ===================================================================
# FK tables & feature registration
# ===================================================================


def populate_fk_tables(db_uri: str, data_dir: Path) -> str:
    """Create publication, publication_sections, and genetic_perturbation tables."""
    db = lancedb.connect(db_uri)
    existing = db.list_tables().tables

    pub_df = pd.read_parquet(data_dir / "PublicationSchema.parquet")
    publication_uid = pub_df["uid"].iloc[0]

    if "publications" not in existing:
        t = db.create_table("publications", schema=PublicationSchema.to_arrow_schema())
    else:
        t = db.open_table("publications")
    t.merge_insert(on="uid").when_not_matched_insert_all().execute(
        pa.Table.from_pandas(pub_df, schema=PublicationSchema.to_arrow_schema())
    )

    section_pq = data_dir / "PublicationSectionSchema.parquet"
    if section_pq.exists():
        sec_df = pd.read_parquet(section_pq)
        if "publication_sections" not in existing:
            st = db.create_table(
                "publication_sections", schema=PublicationSectionSchema.to_arrow_schema()
            )
            st.add(
                pa.Table.from_pandas(sec_df, schema=PublicationSectionSchema.to_arrow_schema())
            )
        else:
            st = db.open_table("publication_sections")
            existing_pubs = set(
                st.search().select(["publication_uid"]).to_pandas()["publication_uid"]
            )
            new_secs = sec_df[~sec_df["publication_uid"].isin(existing_pubs)]
            if not new_secs.empty:
                st.add(
                    pa.Table.from_pandas(
                        new_secs, schema=PublicationSectionSchema.to_arrow_schema()
                    )
                )

    gp_df = pd.read_parquet(data_dir / "GeneticPerturbationSchema.parquet")
    if "genetic_perturbations" not in existing:
        gt = db.create_table(
            "genetic_perturbations", schema=GeneticPerturbationSchema.to_arrow_schema()
        )
    else:
        gt = db.open_table("genetic_perturbations")
    gt.merge_insert(on="uid").when_not_matched_insert_all().execute(
        pa.Table.from_pandas(gp_df, schema=GeneticPerturbationSchema.to_arrow_schema())
    )

    return publication_uid


def register_features(atlas: RaggedAtlas, data_dir: Path) -> None:
    """Register ImageFeatureSchema from parquet."""
    df = pd.read_parquet(data_dir / "ImageFeatureSchema.parquet")
    records = [ImageFeatureSchema(**row.to_dict()) for _, row in df.iterrows()]
    n_new = atlas.register_features("image_features", records)
    print(f"  Registered {n_new} new features ({len(records)} total)")


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest cpg0021-periscope image features + tiles into a RaggedAtlas"
    )
    parser.add_argument("--atlas-path", type=str, required=True)
    parser.add_argument("--features-dir", type=str, required=True,
                        help="Directory with CSV.gz feature shards")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with resolved parquets and publication.json")
    parser.add_argument("--max-per-shard", type=int, default=1000,
                        help="Max cells per CSV shard (default: 1000)")
    parser.add_argument("--csv-workers", type=int, default=8,
                        help="Parallel workers for CSV processing (default: 8)")
    parser.add_argument("--flush-every", type=int, default=10_000,
                        help="Cells per flush to final zarr (default: 10000)")
    parser.add_argument("--total-cells", type=int, default=None,
                        help="Pre-known cell count to skip counting pass")
    parser.add_argument("--resume-dir", type=str, default=None,
                        help="Temp dir from a previous stage-1 run to skip directly to stage 2")
    args = parser.parse_args()

    atlas_path = Path(args.atlas_path)
    features_dir = Path(args.features_dir)
    data_dir = Path(args.data_dir)

    print(f"Dataset: cpg0021-periscope (Ramezani et al. 2025)")
    print(f"Atlas: {atlas_path}")
    print(f"Features: {features_dir}")
    print(f"Flush every: {args.flush_every:,} cells")

    # Setup atlas
    atlas = create_or_open_atlas(
        str(atlas_path),
        cell_table_name="cells",
        cell_schema=CellIndex,
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
        registry_schemas=REGISTRY_SCHEMAS,
    )
    db_uri = str(atlas_path / "lance_db")

    print(f"\n{'='*60}")
    print("FK tables & feature registration")
    print(f"{'='*60}")
    publication_uid = populate_fk_tables(db_uri, data_dir)
    register_features(atlas, data_dir)

    guide_to_uid = dict(zip(
        pd.read_parquet(data_dir / "GeneticPerturbationSchema.parquet")["guide_sequence"],
        pd.read_parquet(data_dir / "GeneticPerturbationSchema.parquet")["uid"],
    ))
    print(f"  Perturbation lookup: {len(guide_to_uid):,} guides")

    # Stage 1: Buffer features (or resume from a previous run)
    if args.resume_dir:
        temp_dir = Path(args.resume_dir)
        meta_df = pd.read_parquet(temp_dir / "meta.parquet")
        temp_zarr_path = temp_dir / "features.zarr"
        temp_store = zarr.storage.ObjectStore(obstore.store.LocalStore(str(temp_zarr_path)))
        n_features = zarr.open_array(temp_store, mode="r").shape[1]
        print(f"\n{'='*60}")
        print("Stage 1: RESUMED from cache")
        print(f"{'='*60}")
        print(f"  Temp dir: {temp_dir}")
        print(f"  Cells: {len(meta_df):,}, features: {n_features}")
    else:
        print(f"\n{'='*60}")
        print("Stage 1: Buffer features to temp zarr")
        print(f"{'='*60}")
        feature_columns = discover_feature_columns(next(features_dir.glob("*.csv.gz")))
        print(f"  Feature columns: {len(feature_columns)}")

        temp_dir, meta_df, n_features = buffer_features_to_temp_zarr(
            features_dir, feature_columns, args.max_per_shard, args.csv_workers,
            total_cells=args.total_cells,
        )
        temp_zarr_path = temp_dir / "features.zarr"

    # Stage 2: Site-grouped tiling
    print(f"\n{'='*60}")
    print("Stage 2: Site-grouped tiling + final write")
    print(f"{'='*60}")
    n_written, n_skipped = run_stage2(
        atlas, meta_df, temp_zarr_path, n_features,
        guide_to_uid, publication_uid, data_dir, args.flush_every,
    )

    # Clean up temp dir only on success
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"  Cleaned up temp dir: {temp_dir}")

    # Summary
    print(f"\n{'='*60}")
    print("Ingestion complete")
    print(f"{'='*60}")
    print(f"  Total cells: {n_written:,}")
    print(f"  Skipped OOB: {n_skipped}")
    print(f"  Features: {n_written:,} × {n_features}")
    print(f"  Tiles: {n_written:,} × {N_CHANNELS}ch × {TILE_SIZE}×{TILE_SIZE}")
    print(f"  Atlas: {atlas_path}")


if __name__ == "__main__":
    main()
