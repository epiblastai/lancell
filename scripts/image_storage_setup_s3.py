"""Setup script: generate synthetic image tile data and write to three storage backends on S3.

Image equivalent of sparse_storage_setup_s3.py.

Approach 1 -- Blob column: metadata + raw image bytes in a single LanceDB row.
Approach 2 -- Two tables: metadata in one LanceDB table, blobs in another.
             Row offsets stored in the metadata table for O(1) blob lookups.
Approach 3 -- Zarr pointer: metadata in LanceDB, image tiles in a 4D Zarr array
             (N, C, H, W) with sharding. LanceDB stores the row index into the
             Zarr array.

Run this once to create the data, then use image_storage_benchmark_s3.py to benchmark.
"""

import time

import lancedb
import numpy as np
import pyarrow as pa
import s3fs
import zarr
from obstore.store import S3Store

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_IMAGES = 40_000
N_CHANNELS = 5
IMG_H = 72
IMG_W = 72
TILE_SHAPE = (N_CHANNELS, IMG_H, IMG_W)
TILE_SIZE = N_CHANNELS * IMG_H * IMG_W  # 25,920 bytes per image

S3_BASE = "s3://epiblast/lancell_image_test"
S3_BUCKET = "epiblast"
S3_PREFIX = "lancell_image_test"
S3_REGION = "us-east-2"
SEED = 42

_s3 = s3fs.S3FileSystem()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_s3_size_mb(s3_path: str) -> float:
    """Return total size in MB of all objects under an S3 prefix."""
    prefix = s3_path.replace("s3://", "")
    files = _s3.find(prefix, detail=True)
    total = sum(info["size"] for info in files.values())
    return total / (1024 * 1024)


def s3_rm(s3_path: str):
    """Recursively delete all objects under an S3 prefix."""
    prefix = s3_path.replace("s3://", "")
    if _s3.exists(prefix):
        _s3.rm(prefix, recursive=True)


def _dict_to_arrow_table(data: dict) -> pa.Table:
    """Convert a dict of columns to a PyArrow table, handling bytes columns as large_binary."""
    arrays = {}
    for k, v in data.items():
        if len(v) > 0 and isinstance(v[0], bytes):
            arrays[k] = pa.array(v, type=pa.large_binary())
        else:
            arrays[k] = pa.array(v)
    return pa.table(arrays)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_all_data():
    """Generate synthetic image tile data for all cells."""
    print(
        f"Generating {N_IMAGES:,} images of shape {TILE_SHAPE} "
        f"(~{N_IMAGES * TILE_SIZE / 1024**2:.0f} MB)..."
    )
    rng = np.random.default_rng(SEED)

    t0 = time.perf_counter()
    images = rng.integers(0, 256, size=(N_IMAGES, *TILE_SHAPE), dtype=np.uint8)
    t_gen = time.perf_counter() - t0
    print(f"  Generated in {t_gen:.1f}s")

    tissues = ["brain", "lung", "liver", "heart", "kidney", "spleen", "blood", "skin"]
    cell_types = [
        "T cell",
        "B cell",
        "macrophage",
        "fibroblast",
        "epithelial",
        "neuron",
        "astrocyte",
        "endothelial",
    ]
    metadata = {
        "cell_uid": [f"cell_{i:07d}" for i in range(N_IMAGES)],
        "tissue": rng.choice(tissues, size=N_IMAGES).tolist(),
        "cell_type": rng.choice(cell_types, size=N_IMAGES).tolist(),
        "organism": ["human"] * N_IMAGES,
    }
    return images, metadata


# ---------------------------------------------------------------------------
# Approach 1: Single table with blob columns
# ---------------------------------------------------------------------------


def setup_approach1(images, metadata):
    path = f"{S3_BASE}/approach1_blob"
    db = lancedb.connect(path)

    table_data = {
        **metadata,
        "image": [images[i].tobytes() for i in range(N_IMAGES)],
    }

    print("  Writing approach 1 (blob column)...")
    t0 = time.perf_counter()
    tbl = db.create_table("cells", data=_dict_to_arrow_table(table_data))
    tbl.optimize()
    t_write = time.perf_counter() - t0
    print(f"  Written in {t_write:.1f}s, size: {get_s3_size_mb(path):.1f} MB")


# ---------------------------------------------------------------------------
# Approach 2: Two tables (metadata + blobs), row offset lookup
# ---------------------------------------------------------------------------


def setup_approach2(images, metadata):
    path = f"{S3_BASE}/approach2_two_tables"
    db = lancedb.connect(path)

    meta_data = {
        **metadata,
        "blob_row_offset": list(range(N_IMAGES)),
    }
    blob_data = {
        "image": [images[i].tobytes() for i in range(N_IMAGES)],
    }

    print("  Writing approach 2 (two tables)...")
    t0 = time.perf_counter()
    db.create_table("metadata", data=_dict_to_arrow_table(meta_data)).optimize()
    db.create_table("blobs", data=_dict_to_arrow_table(blob_data)).optimize()
    t_write = time.perf_counter() - t0
    print(f"  Written in {t_write:.1f}s, size: {get_s3_size_mb(path):.1f} MB")


# ---------------------------------------------------------------------------
# Approach 3: Zarr pointer (4D sharded array)
# ---------------------------------------------------------------------------


def setup_approach3(images, metadata):
    lance_path = f"{S3_BASE}/approach3_lance"
    db = lancedb.connect(lance_path)

    print("  Writing approach 3 (4D Zarr array)...")
    t0 = time.perf_counter()

    zarr_store = zarr.storage.ObjectStore(
        S3Store(
            S3_BUCKET,
            prefix=f"{S3_PREFIX}/approach3_images.zarr/",
            region=S3_REGION,
        ),
    )
    zarr.create_array(
        zarr_store,
        data=images,
        chunks=(10, N_CHANNELS, IMG_H, IMG_W),
        shards=(10_000, N_CHANNELS, IMG_H, IMG_W),
        overwrite=True,
    )
    t_zarr_write = time.perf_counter() - t0

    meta_data = {
        **metadata,
        "zarr_row_index": list(range(N_IMAGES)),
    }

    t1 = time.perf_counter()
    db.create_table("metadata", data=_dict_to_arrow_table(meta_data)).optimize()
    t_lance_write = time.perf_counter() - t1

    zarr_path = f"{S3_BASE}/approach3_images.zarr"
    zarr_size = get_s3_size_mb(zarr_path)
    lance_size = get_s3_size_mb(lance_path)
    total_size = lance_size + zarr_size
    print(
        f"  Written in {t_zarr_write + t_lance_write:.1f}s "
        f"(zarr: {zarr_size:.1f} MB, lance: {lance_size:.1f} MB, "
        f"total: {total_size:.1f} MB)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"Clearing existing data at {S3_BASE} ...")
    s3_rm(S3_BASE)

    images, metadata = generate_all_data()

    print("\n--- Setting up storage approaches ---")
    setup_approach1(images, metadata)
    setup_approach2(images, metadata)
    setup_approach3(images, metadata)

    print("\nStorage sizes:")
    print(f"  Approach 1 (blob column):  {get_s3_size_mb(f'{S3_BASE}/approach1_blob'):.1f} MB")
    print(
        f"  Approach 2 (two tables):   {get_s3_size_mb(f'{S3_BASE}/approach2_two_tables'):.1f} MB"
    )
    a3_lance = get_s3_size_mb(f"{S3_BASE}/approach3_lance")
    a3_zarr = get_s3_size_mb(f"{S3_BASE}/approach3_images.zarr")
    print(
        f"  Approach 3 (zarr pointer): {a3_lance + a3_zarr:.1f} MB "
        f"(lance: {a3_lance:.1f}, zarr: {a3_zarr:.1f})"
    )
    print(f"\nData written to {S3_BASE}")


if __name__ == "__main__":
    main()
