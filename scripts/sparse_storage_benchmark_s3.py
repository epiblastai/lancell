"""Benchmark: metadata-filtered query + sparse matrix loading for storage strategies on S3.

S3 equivalent of sparse_storage_benchmark.py.
Reads data already uploaded by sparse_storage_setup_s3.py.
"""

import argparse
import asyncio
import time

import lancedb
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from obstore.store import S3Store
from zarr.storage import ObjectStore

from lancell.batch_selection import BatchAsyncArray

ALL_METHODS = ["one_table", "two_table", "zarr_obstore"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_GENES = 20_000
S3_BASE = "s3://epiblast/lancell_data_structure_test"
N_REPEATS = 1

METADATA_COLS = ["cell_uid", "tissue", "cell_type", "organism"]
QUERIES = [
    ("tissue = 'brain' AND cell_type = 'neuron'", "1 tissue+type"),
    ("tissue = 'brain'", "1 tissue"),
    ("tissue IN ('brain', 'lung', 'liver')", "3 tissues"),
]


def reconstruct_csr(all_indices: list[np.ndarray], all_values: list[np.ndarray]) -> sp.csr_matrix:
    indptr = np.zeros(len(all_indices) + 1, dtype=np.int64)
    for i, idx in enumerate(all_indices):
        indptr[i + 1] = indptr[i] + len(idx)
    return sp.csr_matrix(
        (np.concatenate(all_values), np.concatenate(all_indices), indptr),
        shape=(len(all_indices), N_GENES),
    )


# ---------------------------------------------------------------------------
# Query functions — each returns (obs_df, gi_list, cv_list)
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


def query_zarr_obstore(meta_tbl, reader_indices, reader_counts, where: str) -> tuple[pd.DataFrame, list, list]:
    meta = meta_tbl.search().where(where).select(
        METADATA_COLS + ["zarr_start", "zarr_end"]
    ).to_arrow().to_pydict()
    obs = pd.DataFrame({c: meta[c] for c in METADATA_COLS})

    starts = np.array(meta["zarr_start"], dtype=np.int64)
    ends = np.array(meta["zarr_end"], dtype=np.int64)

    async def _fetch():
        return await asyncio.gather(
            reader_indices.read_ranges(starts, ends),
            reader_counts.read_ranges(starts, ends),
        )
    (gi_flat, gi_lengths), (cv_flat, cv_lengths) = asyncio.run(_fetch())

    gi_list = np.split(gi_flat, np.cumsum(gi_lengths)[:-1])
    cv_list = np.split(cv_flat, np.cumsum(cv_lengths)[:-1])
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
        results[label] = {"load": med_load, "csr": med_csr, "total": med_total, "n": n, "mat": mat}
        print(f"  {name:<22s} | {label:<16s} | n={n:>6d} | load={med_load:.4f}s | csr={med_csr:.4f}s | total={med_total:.4f}s")
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark sparse storage strategies on S3")
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
        indices_store = ObjectStore(
            S3Store("epiblast", prefix="lancell_data_structure_test/approach3_indices.zarr/", region="us-east-2"),
            read_only=True,
        )
        counts_store = ObjectStore(
            S3Store("epiblast", prefix="lancell_data_structure_test/approach3_counts.zarr/", region="us-east-2"),
            read_only=True,
        )
        indices_arr = zarr.open_array(store=indices_store, mode="r")
        counts_arr = zarr.open_array(store=counts_store, mode="r")
        print(f"Zarr indices: shape={indices_arr.shape}, shards={indices_arr.shards}")
        print(f"Zarr counts:  shape={counts_arr.shape}, shards={counts_arr.shards}")

        reader_indices = BatchAsyncArray.from_array(indices_arr)
        reader_counts = BatchAsyncArray.from_array(counts_arr)
        r3 = bench("zarr_obstore", lambda w: query_zarr_obstore(meta3, reader_indices, reader_counts, w), QUERIES)
        all_benches.append(("zarr_obstore", r3))
        print()

    # Verify consistency across approaches that ran
    if len(all_benches) > 1:
        print("--- Verifying result consistency across all approaches ---")
        for _, label in QUERIES:
            ref_name, ref = all_benches[0]
            ref_mat = ref[label]["mat"]
            for cmp_name, cmp in all_benches[1:]:
                cmp_mat = cmp[label]["mat"]
                try:
                    assert ref_mat.shape == cmp_mat.shape, (
                        f"[{label}] shape mismatch: {ref_name} {ref_mat.shape} vs {cmp_name} {cmp_mat.shape}"
                    )
                    diff = ref_mat - cmp_mat
                    assert diff.nnz == 0, (
                        f"[{label}] data mismatch: {ref_name} vs {cmp_name} ({diff.nnz} differing elements)"
                    )
                    print(f"  {label}: {ref_name} == {cmp_name} OK ({ref_mat.shape[0]} cells)")
                except AssertionError as e:
                    print(f"  {label}: {ref_name} != {cmp_name} FAIL: {e}")

    # Summary table
    if not all_benches:
        return

    labels = [label for _, label in QUERIES]
    col_width = 24
    header_names = [name for name, _ in all_benches]

    print(f"\n{'=' * (26 + (col_width + 3) * len(all_benches))}")
    print(f"{'':>26s}", end="")
    for name in header_names:
        print(f" | {name:>{col_width}s}", end="")
    print()
    print(f"{'Query':<18s} {'N':>6s}", end="")
    for _ in all_benches:
        print(f" | {'load':>8s} {'csr':>8s} {'total':>8s}", end="")
    print()
    print(f"{'-' * (26 + (col_width + 3) * len(all_benches))}")
    for label in labels:
        n = all_benches[0][1][label]["n"]
        print(f"{label:<18s} {n:>6d}", end="")
        for _, r in all_benches:
            d = r[label]
            print(f" | {d['load']:>7.4f}s {d['csr']:>7.4f}s {d['total']:>7.4f}s", end="")
        print()


if __name__ == "__main__":
    main()
