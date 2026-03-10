"""Pure zarr retrieval timing from S3. No lance, no CSR, no zarrs."""

import time

import numpy as np
import zarr
from obstore.store import S3Store

zarr_store = zarr.storage.ObjectStore(
    S3Store("epiblast", prefix="lancell_data_structure_test/approach3_coo.zarr/", region="us-east-2"),
    read_only=True,
)
arr = zarr.open_array(store=zarr_store, mode="r")
print(f"shape={arr.shape}, dtype={arr.dtype}, shard={arr.metadata.chunk_grid.chunk_shape}")

# Simulate scattered cell reads: 1000 cells, ~1000 nnz each, random positions
rng = np.random.default_rng(42)
n_cells = 1000
nnz_per_cell = 1000
total_rows = arr.shape[0]
starts = np.sort(rng.choice(total_rows - nnz_per_cell, size=n_cells, replace=False))
ends = starts + nnz_per_cell

# --- Test 1: per-cell loop ---
print(f"\n--- Per-cell loop: {n_cells} cells ---")
t0 = time.perf_counter()
for s, e in zip(starts, ends):
    _ = arr[s:e, :]
t1 = time.perf_counter()
print(f"  {t1 - t0:.3f}s  ({(t1 - t0) / n_cells * 1000:.1f}ms per cell)")

# --- Test 2: single contiguous read (best case) ---
print(f"\n--- Single contiguous read: rows 0:{n_cells * nnz_per_cell} ---")
t0 = time.perf_counter()
_ = arr[0:n_cells * nnz_per_cell, :]
t1 = time.perf_counter()
print(f"  {t1 - t0:.3f}s")

# --- Test 3: sorted block-merged reads ---
print(f"\n--- Block-merged reads: {n_cells} cells ---")
shard_rows = arr.metadata.chunk_grid.chunk_shape[0]
# Merge ranges in the same shard
blocks = []
cur_s, cur_e = int(starts[0]), int(ends[0])
for i in range(1, len(starts)):
    s, e = int(starts[i]), int(ends[i])
    if s <= cur_e or s // shard_rows == (cur_e - 1) // shard_rows:
        cur_e = max(cur_e, e)
    else:
        blocks.append((cur_s, cur_e))
        cur_s, cur_e = s, e
blocks.append((cur_s, cur_e))
print(f"  {len(blocks)} blocks from {n_cells} cells")

t0 = time.perf_counter()
for bs, be in blocks:
    _ = arr[bs:be, :]
t1 = time.perf_counter()
print(f"  {t1 - t0:.3f}s  ({(t1 - t0) / len(blocks) * 1000:.1f}ms per block)")
