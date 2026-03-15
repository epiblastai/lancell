"""Benchmark: metadata-filtered query + image tile loading for storage strategies on S3.

Image equivalent of sparse_storage_benchmark_s3.py.
Reads data already uploaded by image_storage_setup_s3.py.
"""

import argparse
import asyncio
import time

import lancedb
import numpy as np
import pandas as pd
import zarr
from obstore.store import S3Store
from zarr.storage import ObjectStore

from lancell.batch_array import BatchAsyncArray

ALL_METHODS = ["one_table", "two_table", "zarr_obstore"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_CHANNELS = 5
IMG_H = 72
IMG_W = 72
TILE_SHAPE = (N_CHANNELS, IMG_H, IMG_W)

S3_BASE = "s3://epiblast/lancell_image_test"
S3_BUCKET = "epiblast"
S3_PREFIX = "lancell_image_test"
S3_REGION = "us-east-2"
N_REPEATS = 1

METADATA_COLS = ["cell_uid", "tissue", "cell_type", "organism"]
QUERIES = [
    ("tissue = 'brain' AND cell_type = 'neuron'", "1 tissue+type"),
    ("tissue = 'brain'", "1 tissue"),
    ("tissue IN ('brain', 'lung', 'liver')", "3 tissues"),
]


# ---------------------------------------------------------------------------
# Query functions -- each returns (obs_df, images_array)
# ---------------------------------------------------------------------------


def query_blob_col(tbl, where: str) -> tuple[pd.DataFrame, np.ndarray]:
    result = tbl.search().where(where).select(METADATA_COLS + ["image"]).to_arrow().to_pydict()
    obs = pd.DataFrame({c: result[c] for c in METADATA_COLS})
    images = np.array(
        [np.frombuffer(b, dtype=np.uint8).reshape(TILE_SHAPE) for b in result["image"]]
    )
    return obs, images


def query_two_tbl(meta_tbl, blob_tbl, where: str) -> tuple[pd.DataFrame, np.ndarray]:
    meta = (
        meta_tbl.search()
        .where(where)
        .select(METADATA_COLS + ["blob_row_offset"])
        .to_arrow()
        .to_pydict()
    )
    obs = pd.DataFrame({c: meta[c] for c in METADATA_COLS})
    blob = blob_tbl.to_lance().take(meta["blob_row_offset"], columns=["image"]).to_pydict()
    images = np.array([np.frombuffer(b, dtype=np.uint8).reshape(TILE_SHAPE) for b in blob["image"]])
    return obs, images


def query_zarr_obstore(meta_tbl, reader_images, where: str) -> tuple[pd.DataFrame, np.ndarray]:
    meta = (
        meta_tbl.search()
        .where(where)
        .select(METADATA_COLS + ["zarr_row_index"])
        .to_arrow()
        .to_pydict()
    )
    obs = pd.DataFrame({c: meta[c] for c in METADATA_COLS})

    row_indices = np.array(meta["zarr_row_index"], dtype=np.int64)
    starts = row_indices
    ends = row_indices + 1

    async def _fetch():
        return await reader_images.read_ranges(starts, ends)

    flat_data, lengths = asyncio.run(_fetch())

    images = flat_data.reshape(-1, *TILE_SHAPE)
    return obs, images


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def bench(name: str, fn, queries: list[tuple[str, str]]):
    results = {}
    for where, label in queries:
        times = []
        n = 0
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            obs, images = fn(where)
            t_total = time.perf_counter() - t0
            times.append(t_total)
            n = len(obs)
        med = np.median(times)
        results[label] = {"total": med, "n": n, "images": images}
        print(f"  {name:<22s} | {label:<16s} | n={n:>6d} | total={med:.4f}s")
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark image storage strategies on S3")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        default=ALL_METHODS,
        help=f"Methods to benchmark (default: all). Choices: {', '.join(ALL_METHODS)}",
    )
    args = parser.parse_args()
    methods = set(args.methods)

    print(f"Running methods: {', '.join(sorted(methods))}")
    print("Opening tables from S3...")

    all_benches = []

    if "one_table" in methods:
        tbl1 = lancedb.connect(f"{S3_BASE}/approach1_blob").open_table("cells")
        r1 = bench("blob_column", lambda w: query_blob_col(tbl1, w), QUERIES)
        all_benches.append(("blob_col", r1))
        print()

    if "two_table" in methods:
        db2 = lancedb.connect(f"{S3_BASE}/approach2_two_tables")
        meta2, blob2 = db2.open_table("metadata"), db2.open_table("blobs")
        r2 = bench("two_tables", lambda w: query_two_tbl(meta2, blob2, w), QUERIES)
        all_benches.append(("two_tbl", r2))
        print()

    if "zarr_obstore" in methods:
        meta3 = lancedb.connect(f"{S3_BASE}/approach3_lance").open_table("metadata")
        images_store = ObjectStore(
            S3Store(S3_BUCKET, prefix=f"{S3_PREFIX}/approach3_images.zarr/", region=S3_REGION),
            read_only=True,
        )
        images_arr = zarr.open_array(store=images_store, mode="r")
        print(
            f"Zarr images: shape={images_arr.shape}, chunks={images_arr.chunks}, shards={images_arr.shards}"
        )

        reader_images = BatchAsyncArray.from_array(images_arr)
        r3 = bench("zarr_obstore", lambda w: query_zarr_obstore(meta3, reader_images, w), QUERIES)
        all_benches.append(("zarr_obstore", r3))
        print()

    # Verify consistency across approaches that ran
    if len(all_benches) > 1:
        print("--- Verifying result consistency across all approaches ---")
        for _, label in QUERIES:
            ref_name, ref = all_benches[0]
            ref_imgs = ref[label]["images"]
            for cmp_name, cmp in all_benches[1:]:
                cmp_imgs = cmp[label]["images"]
                try:
                    assert ref_imgs.shape == cmp_imgs.shape, (
                        f"[{label}] shape mismatch: {ref_name} {ref_imgs.shape} vs {cmp_name} {cmp_imgs.shape}"
                    )
                    assert np.array_equal(ref_imgs, cmp_imgs), (
                        f"[{label}] data mismatch: {ref_name} vs {cmp_name}"
                    )
                    print(f"  {label}: {ref_name} == {cmp_name} OK ({ref_imgs.shape[0]} images)")
                except AssertionError as e:
                    print(f"  {label}: {ref_name} != {cmp_name} FAIL: {e}")

    # Summary table
    if not all_benches:
        return

    labels = [label for _, label in QUERIES]
    col_width = 14
    header_names = [name for name, _ in all_benches]

    print(f"\n{'=' * (26 + (col_width + 3) * len(all_benches))}")
    print(f"{'':>26s}", end="")
    for name in header_names:
        print(f" | {name:>{col_width}s}", end="")
    print()
    print(f"{'Query':<18s} {'N':>6s}", end="")
    for _ in all_benches:
        print(f" | {'total':>{col_width}s}", end="")
    print()
    print(f"{'-' * (26 + (col_width + 3) * len(all_benches))}")
    for label in labels:
        n = all_benches[0][1][label]["n"]
        print(f"{label:<18s} {n:>6d}", end="")
        for _, r in all_benches:
            d = r[label]
            print(f" | {d['total']:>{col_width - 1}.4f}s", end="")
        print()


if __name__ == "__main__":
    main()
