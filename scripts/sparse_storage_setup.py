"""Setup script: generate synthetic sparse single-cell data and write to three storage backends.

Approach 1 — Blob column: metadata + raw sparse data in a single LanceDB row.
Approach 2 — Two tables: metadata in one LanceDB table, blobs in another.
             Row offsets stored in the metadata table for O(1) blob lookups.
Approach 3 — Zarr pointer: metadata in LanceDB, sparse COO data in a Zarr array
             of shape (N, 3) where columns are (cell_id, gene_id, value).
             LanceDB stores (start, end) positions into the Zarr array.

Run this once to create the data, then use sparse_storage_benchmark.py to benchmark.
"""

import os
import shutil
import time

import lancedb
import numpy as np
import pyarrow as pa
import zarr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_CELLS = 500_000
N_GENES = 20_000
DENSITY = 0.05  # ~5% nonzero entries per cell
DATA_DIR = "/tmp/sparse_storage_benchmark"
SEED = 42

NNZ_PER_CELL = int(N_GENES * DENSITY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_dir_size_mb(path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)


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

def generate_sparse_cell(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate one sparse cell: sorted gene indices and float32 counts."""
    indices = np.sort(rng.choice(N_GENES, size=NNZ_PER_CELL, replace=False)).astype(np.int32)
    values = rng.exponential(2.0, size=NNZ_PER_CELL).astype(np.float32)
    return indices, values


def generate_all_data():
    """Generate synthetic sparse expression data for all cells."""
    print(f"Generating {N_CELLS:,} cells x {N_GENES:,} genes @ {DENSITY:.0%} density "
          f"({NNZ_PER_CELL} nnz/cell)...")
    rng = np.random.default_rng(SEED)

    all_gene_indices = []
    all_counts = []

    t0 = time.perf_counter()
    for _ in range(N_CELLS):
        gi, cv = generate_sparse_cell(rng)
        all_gene_indices.append(gi)
        all_counts.append(cv)
    t_gen = time.perf_counter() - t0
    print(f"  Generated in {t_gen:.1f}s")

    tissues = ["brain", "lung", "liver", "heart", "kidney", "spleen", "blood", "skin"]
    cell_types = ["T cell", "B cell", "macrophage", "fibroblast", "epithelial",
                  "neuron", "astrocyte", "endothelial"]
    metadata = {
        "cell_uid": [f"cell_{i:07d}" for i in range(N_CELLS)],
        "tissue": rng.choice(tissues, size=N_CELLS).tolist(),
        "cell_type": rng.choice(cell_types, size=N_CELLS).tolist(),
        "organism": ["human"] * N_CELLS,
    }
    return all_gene_indices, all_counts, metadata


# ---------------------------------------------------------------------------
# Approach 1: Single table with blob columns
# ---------------------------------------------------------------------------

def setup_approach1(all_gene_indices, all_counts, metadata):
    path = os.path.join(DATA_DIR, "approach1_blob")
    db = lancedb.connect(path)

    table_data = {
        **metadata,
        "gene_indices": [gi.tobytes() for gi in all_gene_indices],
        "counts": [cv.tobytes() for cv in all_counts],
    }

    print("  Writing approach 1 (blob column)...")
    t0 = time.perf_counter()
    tbl = db.create_table("cells", data=_dict_to_arrow_table(table_data))
    tbl.optimize()
    t_write = time.perf_counter() - t0
    print(f"  Written in {t_write:.1f}s, size: {get_dir_size_mb(path):.1f} MB")


# ---------------------------------------------------------------------------
# Approach 2: Two tables (metadata + blobs), row offset lookup
# ---------------------------------------------------------------------------

def setup_approach2(all_gene_indices, all_counts, metadata):
    path = os.path.join(DATA_DIR, "approach2_two_tables")
    db = lancedb.connect(path)

    meta_data = {
        **metadata,
        "blob_row_offset": list(range(N_CELLS)),
    }
    blob_data = {
        "gene_indices": [gi.tobytes() for gi in all_gene_indices],
        "counts": [cv.tobytes() for cv in all_counts],
    }

    print("  Writing approach 2 (two tables)...")
    t0 = time.perf_counter()
    db.create_table("metadata", data=_dict_to_arrow_table(meta_data)).optimize()
    db.create_table("blobs", data=_dict_to_arrow_table(blob_data)).optimize()
    t_write = time.perf_counter() - t0
    print(f"  Written in {t_write:.1f}s, size: {get_dir_size_mb(path):.1f} MB")


# ---------------------------------------------------------------------------
# Approach 3: Zarr COO pointer
# ---------------------------------------------------------------------------

def setup_approach3(all_gene_indices, all_counts, metadata):
    lance_path = os.path.join(DATA_DIR, "approach3_lance")
    zarr_path = os.path.join(DATA_DIR, "approach3_coo.zarr")
    db = lancedb.connect(lance_path)

    total_nnz = sum(len(gi) for gi in all_gene_indices)
    print(f"  Total NNZ entries: {total_nnz:,}")

    # Pre-allocate COO arrays
    coo_cell_ids = np.empty(total_nnz, dtype=np.int32)
    coo_gene_ids = np.empty(total_nnz, dtype=np.int32)
    coo_values = np.empty(total_nnz, dtype=np.float32)

    starts = np.empty(N_CELLS, dtype=np.int64)
    ends = np.empty(N_CELLS, dtype=np.int64)
    offset = 0
    for i in range(N_CELLS):
        n = len(all_gene_indices[i])
        starts[i] = offset
        ends[i] = offset + n
        coo_cell_ids[offset:offset + n] = i
        coo_gene_ids[offset:offset + n] = all_gene_indices[i]
        coo_values[offset:offset + n] = all_counts[i]
        offset += n

    print("  Writing approach 3 (Zarr COO)...")
    t0 = time.perf_counter()

    store = zarr.storage.LocalStore(zarr_path)
    zarr_arr = zarr.create_array(
        store=store,
        shape=(total_nnz, 3),
        chunks=(5_000, 3),
        shards=(100_000, 3),
        dtype=np.float32,
        overwrite=True,
    )

    WRITE_BATCH = 10_000_000
    for batch_start in range(0, total_nnz, WRITE_BATCH):
        batch_end = min(batch_start + WRITE_BATCH, total_nnz)
        batch_data = np.column_stack([
            coo_cell_ids[batch_start:batch_end].astype(np.float32),
            coo_gene_ids[batch_start:batch_end].astype(np.float32),
            coo_values[batch_start:batch_end],
        ])
        zarr_arr[batch_start:batch_end, :] = batch_data
    t_zarr_write = time.perf_counter() - t0

    meta_data = {
        **metadata,
        "zarr_start": starts.tolist(),
        "zarr_end": ends.tolist(),
    }

    t1 = time.perf_counter()
    db.create_table("metadata", data=_dict_to_arrow_table(meta_data)).optimize()
    t_lance_write = time.perf_counter() - t1

    total_size = get_dir_size_mb(lance_path) + get_dir_size_mb(zarr_path)
    print(f"  Written in {t_zarr_write + t_lance_write:.1f}s "
          f"(zarr: {get_dir_size_mb(zarr_path):.1f} MB, lance: {get_dir_size_mb(lance_path):.1f} MB, "
          f"total: {total_size:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)

    all_gene_indices, all_counts, metadata = generate_all_data()

    print("\n--- Setting up storage approaches ---")
    setup_approach1(all_gene_indices, all_counts, metadata)
    setup_approach2(all_gene_indices, all_counts, metadata)
    setup_approach3(all_gene_indices, all_counts, metadata)

    print(f"\nStorage sizes:")
    print(f"  Approach 1 (blob column):  {get_dir_size_mb(os.path.join(DATA_DIR, 'approach1_blob')):.1f} MB")
    print(f"  Approach 2 (two tables):   {get_dir_size_mb(os.path.join(DATA_DIR, 'approach2_two_tables')):.1f} MB")
    a3_lance = get_dir_size_mb(os.path.join(DATA_DIR, "approach3_lance"))
    a3_zarr = get_dir_size_mb(os.path.join(DATA_DIR, "approach3_coo.zarr"))
    print(f"  Approach 3 (zarr pointer): {a3_lance + a3_zarr:.1f} MB (lance: {a3_lance:.1f}, zarr: {a3_zarr:.1f})")
    print(f"\nData written to {DATA_DIR}")


if __name__ == "__main__":
    main()
