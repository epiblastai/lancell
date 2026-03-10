"""Benchmark: metadata-filtered query + sparse matrix loading for storage strategies.

Self-contained: generates data, writes all backends, and benchmarks in one run.
Designed to complete in under 60s for fast iteration.
"""

import os
import shutil
import time

import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.sparse as sp
import zarr
from zarrs.utils import ChunkItem
from zarrs.pipeline import get_codec_pipeline_impl

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_CELLS = 50_000
N_GENES = 20_000
MEDIAN_NNZ = 1000  # median nnz per cell (variable across cells)
DATA_DIR = "/tmp/sparse_bench"
N_REPEATS = 3
SEED = 42

METADATA_COLS = ["cell_uid", "tissue", "cell_type", "organism"]
QUERIES = [
    ("tissue = 'brain' AND cell_type = 'neuron'", "1 tissue+type"),
    ("tissue = 'brain'", "1 tissue"),
    ("tissue IN ('brain', 'lung', 'liver')", "3 tissues"),
]


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
    arrays = {}
    for k, v in data.items():
        if len(v) > 0 and isinstance(v[0], bytes):
            arrays[k] = pa.array(v, type=pa.large_binary())
        else:
            arrays[k] = pa.array(v)
    return pa.table(arrays)


def reconstruct_csr(all_indices: list[np.ndarray], all_values: list[np.ndarray]) -> sp.csr_matrix:
    indptr = np.zeros(len(all_indices) + 1, dtype=np.int64)
    for i, idx in enumerate(all_indices):
        indptr[i + 1] = indptr[i] + len(idx)
    return sp.csr_matrix(
        (np.concatenate(all_values), np.concatenate(all_indices), indptr),
        shape=(len(all_indices), N_GENES),
    )


# ---------------------------------------------------------------------------
# Batched zarr read — bypasses per-cell Python overhead
# ---------------------------------------------------------------------------

def batch_read_zarr(impl, starts, ends, shard_shape, n_cols=2):
    """Read multiple [start:end, :] ranges in a single Rust call.

    Handles ranges that span shard boundaries by splitting into
    multiple ChunkItems per cell.
    """
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    lengths = ends - starts
    total_rows = int(lengths.sum())
    shard_rows = shard_shape[0]

    items = []
    out_offset = 0
    for i in range(len(starts)):
        s = int(starts[i])
        e = int(ends[i])
        while s < e:
            shard_idx = s // shard_rows
            local_start = s % shard_rows
            chunk_len = min(e, (shard_idx + 1) * shard_rows) - s
            items.append(ChunkItem(
                key=f"c/{shard_idx}/0",
                chunk_subset=[slice(local_start, local_start + chunk_len), slice(0, n_cols)],
                chunk_shape=shard_shape,
                subset=[slice(out_offset, out_offset + chunk_len), slice(0, n_cols)],
                shape=(total_rows, n_cols),
            ))
            out_offset += chunk_len
            s += chunk_len

    out = np.empty((total_rows, n_cols), dtype=np.float32)
    impl.retrieve_chunks_and_apply_index(items, out)
    return out, lengths


# ---------------------------------------------------------------------------
# Data generation + setup
# ---------------------------------------------------------------------------

def generate_and_setup():
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)

    rng = np.random.default_rng(SEED)

    # Variable NNZ per cell: log-normal, median=MEDIAN_NNZ, range ~200-4000
    nnz_counts = np.clip(
        rng.lognormal(mean=np.log(MEDIAN_NNZ), sigma=0.5, size=N_CELLS).astype(int),
        100, N_GENES,
    )
    print(f"Generating {N_CELLS:,} cells x {N_GENES:,} genes, "
          f"variable NNZ (median={int(np.median(nnz_counts))}, "
          f"range={nnz_counts.min()}-{nnz_counts.max()})...")

    all_gi = []
    all_cv = []
    for nnz in nnz_counts:
        indices = np.sort(rng.choice(N_GENES, size=nnz, replace=False)).astype(np.int32)
        values = rng.exponential(2.0, size=nnz).astype(np.float32)
        all_gi.append(indices)
        all_cv.append(values)

    tissues = ["brain", "lung", "liver", "heart", "kidney", "spleen", "blood", "skin"]
    cell_types = ["T cell", "B cell", "macrophage", "fibroblast", "epithelial",
                  "neuron", "astrocyte", "endothelial"]
    metadata = {
        "cell_uid": [f"cell_{i:06d}" for i in range(N_CELLS)],
        "tissue": rng.choice(tissues, size=N_CELLS).tolist(),
        "cell_type": rng.choice(cell_types, size=N_CELLS).tolist(),
        "organism": ["human"] * N_CELLS,
    }

    gi_bytes = [g.tobytes() for g in all_gi]
    cv_bytes = [c.tobytes() for c in all_cv]

    # --- Approach 1: blob column ---
    print("  Writing approach 1 (blob column)...")
    db1 = lancedb.connect(os.path.join(DATA_DIR, "a1"))
    db1.create_table("cells", data=_dict_to_arrow_table({
        **metadata, "gene_indices": gi_bytes, "counts": cv_bytes,
    })).optimize()

    # --- Approach 2: two tables ---
    print("  Writing approach 2 (two tables)...")
    db2 = lancedb.connect(os.path.join(DATA_DIR, "a2"))
    db2.create_table("metadata", data=_dict_to_arrow_table({
        **metadata, "blob_row_offset": list(range(N_CELLS)),
    })).optimize()
    db2.create_table("blobs", data=_dict_to_arrow_table({
        "gene_indices": gi_bytes, "counts": cv_bytes,
    })).optimize()

    # --- Approach 3: zarr (gene_id, value) — variable NNZ per cell ---
    cell_lengths = np.array([len(g) for g in all_gi], dtype=np.int64)
    ends = np.cumsum(cell_lengths)
    starts = ends - cell_lengths
    total_nnz = int(ends[-1])

    coo_gene_ids = np.concatenate(all_gi)
    coo_values = np.concatenate(all_cv)

    print("  Writing approach 3 (zarr Nx2, zstd)...")
    store = zarr.storage.LocalStore(os.path.join(DATA_DIR, "a3.zarr"))
    zarr_arr = zarr.create_array(
        store=store,
        shape=(total_nnz, 2),
        chunks=(5_000, 2),
        shards=(100_000, 2),
        dtype=np.float32,
        overwrite=True,
    )
    batch_data = np.column_stack([coo_gene_ids.astype(np.float32), coo_values])
    BATCH = 10_000_000
    for i in range(0, total_nnz, BATCH):
        zarr_arr[i:i+BATCH, :] = batch_data[i:i+BATCH]

    # Lance metadata for zarr approach
    db3 = lancedb.connect(os.path.join(DATA_DIR, "a3_lance"))
    db3.create_table("metadata", data=_dict_to_arrow_table({
        **metadata, "zarr_start": starts.tolist(), "zarr_end": ends.tolist(),
    })).optimize()

    print(f"\nStorage sizes:")
    print(f"  Approach 1 (blob column):     {get_dir_size_mb(os.path.join(DATA_DIR, 'a1')):.1f} MB")
    print(f"  Approach 2 (two tables):      {get_dir_size_mb(os.path.join(DATA_DIR, 'a2')):.1f} MB")
    a3l = get_dir_size_mb(os.path.join(DATA_DIR, "a3_lance"))
    a3z = get_dir_size_mb(os.path.join(DATA_DIR, "a3.zarr"))
    print(f"  Approach 3 (zarr+lance):      {a3l + a3z:.1f} MB (lance: {a3l:.1f}, zarr: {a3z:.1f})")


# ---------------------------------------------------------------------------
# Query functions — each returns (obs_df, gi_list, cv_list, t_load)
# Load phase: query metadata + fetch raw sparse data
# CSR reconstruction is timed separately by the benchmark harness
# ---------------------------------------------------------------------------

def query_blob_col(tbl, where: str) -> tuple[pd.DataFrame, list, list]:
    result = tbl.search().where(where).select(METADATA_COLS + ["gene_indices", "counts"]).to_arrow().to_pydict()
    obs = pd.DataFrame({c: result[c] for c in METADATA_COLS})
    gi = [np.frombuffer(b, dtype=np.int32) for b in result["gene_indices"]]
    cv = [np.frombuffer(b, dtype=np.float32) for b in result["counts"]]
    return obs, gi, cv


def query_two_tbl(meta_tbl, blob_tbl, where: str) -> tuple[pd.DataFrame, list, list]:
    meta = meta_tbl.search().where(where).select(METADATA_COLS + ["blob_row_offset"]).to_arrow().to_pydict()
    obs = pd.DataFrame({c: meta[c] for c in METADATA_COLS})
    blob = blob_tbl.to_lance().take(meta["blob_row_offset"], columns=["gene_indices", "counts"]).to_pydict()
    gi = [np.frombuffer(b, dtype=np.int32) for b in blob["gene_indices"]]
    cv = [np.frombuffer(b, dtype=np.float32) for b in blob["counts"]]
    return obs, gi, cv


def query_zarr_loop(meta_tbl, zarr_arr, where: str) -> tuple[pd.DataFrame, list, list]:
    """Zarr with per-cell loop (baseline)."""
    meta = meta_tbl.search().where(where).select(METADATA_COLS + ["zarr_start", "zarr_end"]).to_arrow().to_pydict()
    obs = pd.DataFrame({c: meta[c] for c in METADATA_COLS})
    gi_list, cv_list = [], []
    for s, e in zip(meta["zarr_start"], meta["zarr_end"]):
        chunk = zarr_arr[s:e, :]
        gi_list.append(chunk[:, 0].astype(np.int32))
        cv_list.append(chunk[:, 1])
    return obs, gi_list, cv_list


def query_zarr_batch(meta_tbl, zarr_impl, zarr_shard_shape, where: str) -> tuple[pd.DataFrame, list, list]:
    """Zarr with batched Rust read — bypasses per-cell Python overhead."""
    meta = meta_tbl.search().where(where).select(METADATA_COLS + ["zarr_start", "zarr_end"]).to_arrow().to_pydict()
    obs = pd.DataFrame({c: meta[c] for c in METADATA_COLS})

    starts = np.array(meta["zarr_start"], dtype=np.int64)
    ends = np.array(meta["zarr_end"], dtype=np.int64)
    data, lengths = batch_read_zarr(zarr_impl, starts, ends, zarr_shard_shape, n_cols=2)

    # Split into per-cell arrays
    splits = np.cumsum(lengths[:-1])
    gi_list = [g.astype(np.int32) for g in np.split(data[:, 0], splits)]
    cv_list = list(np.split(data[:, 1], splits))
    return obs, gi_list, cv_list


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench(name: str, fn, queries: list[tuple[str, str]]):
    results = {}
    for where, label in queries:
        load_times = []
        csr_times = []
        n = 0
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            obs, gi, cv = fn(where)
            t_load = time.perf_counter() - t0

            t1 = time.perf_counter()
            mat = reconstruct_csr(gi, cv)
            t_csr = time.perf_counter() - t1

            load_times.append(t_load)
            csr_times.append(t_csr)
            n = len(obs)
        med_load = np.median(load_times)
        med_csr = np.median(csr_times)
        med_total = med_load + med_csr
        results[label] = {"load": med_load, "csr": med_csr, "total": med_total, "n": n}
        print(f"  {name:<22s} | {label:<16s} | n={n:>6d} | load={med_load:.4f}s | csr={med_csr:.4f}s | total={med_total:.4f}s")
    return results


def main():
    generate_and_setup()

    print("\nOpening tables...")
    tbl1 = lancedb.connect(os.path.join(DATA_DIR, "a1")).open_table("cells")
    db2 = lancedb.connect(os.path.join(DATA_DIR, "a2"))
    meta2, blob2 = db2.open_table("metadata"), db2.open_table("blobs")
    meta3 = lancedb.connect(os.path.join(DATA_DIR, "a3_lance")).open_table("metadata")

    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
    zarr_arr = zarr.open_array(os.path.join(DATA_DIR, "a3.zarr"), mode="r")
    zarr_impl = get_codec_pipeline_impl(zarr_arr.metadata, zarr_arr.store, strict=True)
    zarr_shard_shape = tuple(zarr_arr.metadata.chunk_grid.chunk_shape)

    print(f"\n--- Benchmark ({N_REPEATS} repeats, median) ---\n")
    r1 = bench("blob_column", lambda w: query_blob_col(tbl1, w), QUERIES)
    print()
    r2 = bench("two_tables", lambda w: query_two_tbl(meta2, blob2, w), QUERIES)
    print()
    r3 = bench("zarr_loop", lambda w: query_zarr_loop(meta3, zarr_arr, w), QUERIES)
    print()
    r4 = bench("zarr_batch", lambda w: query_zarr_batch(meta3, zarr_impl, zarr_shard_shape, w), QUERIES)

    # Summary
    labels = [label for _, label in QUERIES]
    all_results = [("blob_col", r1), ("two_tbl", r2), ("zarr_loop", r3), ("zarr_batch", r4)]

    print(f"\n{'='*100}")
    print(f"{'':>26s} | {'blob_col':>24s} | {'two_tbl':>24s} | {'zarr_loop':>24s} | {'zarr_batch':>24s}")
    print(f"{'Query':<18s} {'N':>6s} | {'load':>8s} {'csr':>8s} {'total':>8s}| {'load':>8s} {'csr':>8s} {'total':>8s}| {'load':>8s} {'csr':>8s} {'total':>8s}| {'load':>8s} {'csr':>8s} {'total':>8s}")
    print(f"{'-'*134}")
    for label in labels:
        n = r1[label]["n"]
        parts = []
        for _, r in all_results:
            d = r[label]
            parts.append(f"{d['load']:>7.4f}s {d['csr']:>7.4f}s {d['total']:>7.4f}s")
        print(f"{label:<18s} {n:>6d} | {'| '.join(parts)}")


if __name__ == "__main__":
    main()
