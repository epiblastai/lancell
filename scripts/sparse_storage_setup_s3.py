"""Setup script: generate synthetic sparse single-cell data and write to three storage backends on S3.

S3 equivalent of sparse_storage_setup.py.

Approach 1 — Blob column: metadata + raw sparse data in a single LanceDB row.
Approach 2 — Two tables: metadata in one LanceDB table, blobs in another.
             Row offsets stored in the metadata table for O(1) blob lookups.
Approach 3 — Zarr pointer: metadata in LanceDB, sparse COO data in two separate
             1D Zarr arrays (gene_indices as int32, counts as float32).
             LanceDB stores (start, end) positions into the Zarr arrays.

Run this once to create the data, then use sparse_storage_benchmark_s3.py to benchmark.
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
N_CELLS = 200_000
N_GENES = 20_000
DENSITY = 0.05  # ~5% nonzero entries per cell
S3_BASE = "s3://epiblast/lancell_data_structure_test"
SEED = 42

MEAN_NNZ_PER_CELL = int(N_GENES * DENSITY)
NNZ_STD = int(MEAN_NNZ_PER_CELL * 0.3)  # 30% CV around the mean

_s3 = s3fs.S3FileSystem()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_s3_size_mb(s3_path: str) -> float:
    """Return total size in MB of all objects under an S3 prefix."""
    # strip protocol for s3fs.find
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

def generate_sparse_cell(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate one sparse cell with variable sparsity: sorted gene indices and float32 counts."""
    nnz = int(rng.normal(MEAN_NNZ_PER_CELL, NNZ_STD))
    nnz = max(1, min(nnz, N_GENES))  # clamp to [1, N_GENES]
    indices = np.sort(rng.choice(N_GENES, size=nnz, replace=False)).astype(np.int32)
    values = rng.exponential(2.0, size=nnz).astype(np.float32)
    return indices, values


def generate_all_data():
    """Generate synthetic sparse expression data for all cells."""
    print(f"Generating {N_CELLS:,} cells x {N_GENES:,} genes @ ~{DENSITY:.0%} density "
          f"(mean {MEAN_NNZ_PER_CELL} nnz/cell, std {NNZ_STD})...")
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
    path = f"{S3_BASE}/approach1_blob"
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
    print(f"  Written in {t_write:.1f}s, size: {get_s3_size_mb(path):.1f} MB")


# ---------------------------------------------------------------------------
# Approach 2: Two tables (metadata + blobs), row offset lookup
# ---------------------------------------------------------------------------

def setup_approach2(all_gene_indices, all_counts, metadata):
    path = f"{S3_BASE}/approach2_two_tables"
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
    print(f"  Written in {t_write:.1f}s, size: {get_s3_size_mb(path):.1f} MB")


# ---------------------------------------------------------------------------
# Approach 3: Zarr COO pointer
# ---------------------------------------------------------------------------

def setup_approach3(all_gene_indices, all_counts, metadata):
    lance_path = f"{S3_BASE}/approach3_lance"
    db = lancedb.connect(lance_path)

    total_nnz = sum(len(gi) for gi in all_gene_indices)
    print(f"  Total NNZ entries: {total_nnz:,}")

    # Pre-allocate flat arrays
    coo_gene_ids = np.empty(total_nnz, dtype=np.int32)
    coo_values = np.empty(total_nnz, dtype=np.float32)

    starts = np.empty(N_CELLS, dtype=np.int64)
    ends = np.empty(N_CELLS, dtype=np.int64)
    offset = 0
    for i in range(N_CELLS):
        n = len(all_gene_indices[i])
        starts[i] = offset
        ends[i] = offset + n
        coo_gene_ids[offset:offset + n] = all_gene_indices[i]
        coo_values[offset:offset + n] = all_counts[i]
        offset += n

    print("  Writing approach 3 (Zarr COO, separate 1D arrays)...")
    t0 = time.perf_counter()

    indices_store = zarr.storage.ObjectStore(
        S3Store("epiblast", prefix="lancell_data_structure_test/approach3_indices.zarr/", region="us-east-2"),
    )
    counts_store = zarr.storage.ObjectStore(
        S3Store("epiblast", prefix="lancell_data_structure_test/approach3_counts.zarr/", region="us-east-2"),
    )
    zarr.create_array(
        indices_store,
        data=coo_gene_ids,
        chunks=(5_000,),
        shards=(50_000_000,),
        overwrite=True,
    )
    zarr.create_array(
        counts_store,
        data=coo_values,
        chunks=(5_000,),
        shards=(50_000_000,),
        overwrite=True,
    )
    t_zarr_write = time.perf_counter() - t0

    meta_data = {
        **metadata,
        "zarr_start": starts.tolist(),
        "zarr_end": ends.tolist(),
    }

    t1 = time.perf_counter()
    db.create_table("metadata", data=_dict_to_arrow_table(meta_data)).optimize()
    t_lance_write = time.perf_counter() - t1

    indices_zarr_path = f"{S3_BASE}/approach3_indices.zarr"
    counts_zarr_path = f"{S3_BASE}/approach3_counts.zarr"
    indices_size = get_s3_size_mb(indices_zarr_path)
    counts_size = get_s3_size_mb(counts_zarr_path)
    lance_size = get_s3_size_mb(lance_path)
    total_size = lance_size + indices_size + counts_size
    print(f"  Written in {t_zarr_write + t_lance_write:.1f}s "
          f"(indices zarr: {indices_size:.1f} MB, counts zarr: {counts_size:.1f} MB, "
          f"lance: {lance_size:.1f} MB, total: {total_size:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Clearing existing data at {S3_BASE} ...")
    s3_rm(S3_BASE)

    all_gene_indices, all_counts, metadata = generate_all_data()

    print("\n--- Setting up storage approaches ---")
    setup_approach1(all_gene_indices, all_counts, metadata)
    setup_approach2(all_gene_indices, all_counts, metadata)
    setup_approach3(all_gene_indices, all_counts, metadata)

    print(f"\nStorage sizes:")
    print(f"  Approach 1 (blob column):  {get_s3_size_mb(f'{S3_BASE}/approach1_blob'):.1f} MB")
    print(f"  Approach 2 (two tables):   {get_s3_size_mb(f'{S3_BASE}/approach2_two_tables'):.1f} MB")
    a3_lance = get_s3_size_mb(f"{S3_BASE}/approach3_lance")
    a3_idx = get_s3_size_mb(f"{S3_BASE}/approach3_indices.zarr")
    a3_cnt = get_s3_size_mb(f"{S3_BASE}/approach3_counts.zarr")
    print(f"  Approach 3 (zarr pointer): {a3_lance + a3_idx + a3_cnt:.1f} MB "
          f"(lance: {a3_lance:.1f}, indices: {a3_idx:.1f}, counts: {a3_cnt:.1f})")
    print(f"\nData written to {S3_BASE}")


if __name__ == "__main__":
    main()
