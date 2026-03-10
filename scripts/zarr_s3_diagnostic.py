"""Quick diagnostic: test zarr read speed from S3 with different store backends."""

import time

import zarr
from obstore.store import S3Store

ZARR_PATH = "s3://epiblast/lancell_data_structure_test/approach3_coo.zarr"

print("=== Test 1: ObjectStore (obstore S3Store) ===")
store1 = zarr.storage.ObjectStore(
    S3Store("epiblast", prefix="lancell_data_structure_test/approach3_coo.zarr/", region="us-east-2"),
    read_only=True,
)
arr1 = zarr.open_array(store=store1, mode="r")
print(f"shape={arr1.shape}, dtype={arr1.dtype}, chunks={arr1.chunks}")
print(f"metadata chunk_grid: {arr1.metadata.chunk_grid}")

t0 = time.perf_counter()
chunk = arr1[0:100, :]
print(f"Read 100 rows: {time.perf_counter() - t0:.3f}s, shape={chunk.shape}")

t0 = time.perf_counter()
chunk = arr1[0:5000, :]
print(f"Read 5000 rows: {time.perf_counter() - t0:.3f}s, shape={chunk.shape}")

t0 = time.perf_counter()
chunk = arr1[0:100_000, :]
print(f"Read 100k rows (1 shard): {time.perf_counter() - t0:.3f}s, shape={chunk.shape}")

print("\n=== Test 2: FsspecStore (s3fs) ===")
import s3fs
fs = s3fs.S3FileSystem()
store2 = zarr.storage.FsspecStore(fs=fs, path="epiblast/lancell_data_structure_test/approach3_coo.zarr")
arr2 = zarr.open_array(store=store2, mode="r")
print(f"shape={arr2.shape}, dtype={arr2.dtype}")

t0 = time.perf_counter()
chunk = arr2[0:100, :]
print(f"Read 100 rows: {time.perf_counter() - t0:.3f}s, shape={chunk.shape}")

t0 = time.perf_counter()
chunk = arr2[0:5000, :]
print(f"Read 5000 rows: {time.perf_counter() - t0:.3f}s, shape={chunk.shape}")

t0 = time.perf_counter()
chunk = arr2[0:100_000, :]
print(f"Read 100k rows (1 shard): {time.perf_counter() - t0:.3f}s, shape={chunk.shape}")

print("\n=== Test 3: URL string (zarr auto-resolve) ===")
arr3 = zarr.open_array(ZARR_PATH, mode="r")
print(f"shape={arr3.shape}, dtype={arr3.dtype}, store type: {type(arr3.store)}")

t0 = time.perf_counter()
chunk = arr3[0:100, :]
print(f"Read 100 rows: {time.perf_counter() - t0:.3f}s, shape={chunk.shape}")

t0 = time.perf_counter()
chunk = arr3[0:5000, :]
print(f"Read 5000 rows: {time.perf_counter() - t0:.3f}s, shape={chunk.shape}")

t0 = time.perf_counter()
chunk = arr3[0:100_000, :]
print(f"Read 100k rows (1 shard): {time.perf_counter() - t0:.3f}s, shape={chunk.shape}")
