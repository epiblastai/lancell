import numpy as np
import lancedb
import zarr
import numpy as np
import pandas as pd
import uuid
from zarr.storage import ObjectStore
from obstore.store import S3Store
from scipy import sparse
from zarrs.utils import ChunkItem
from time import time
from zarrs.pipeline import get_codec_pipeline_impl


def batch_read_zarr(impl, starts, ends, shard_shape):
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
                chunk_subset=[slice(local_start, local_start + chunk_len)],
                chunk_shape=shard_shape,
                subset=[slice(out_offset, out_offset + chunk_len)],
                shape=(total_rows,),
            ))
            out_offset += chunk_len
            s += chunk_len

    out = np.empty((total_rows,), dtype=np.int32)
    impl.retrieve_chunks_and_apply_index(items, out)
    return out, lengths


def setup_data():
    db = lancedb.connect("s3://epiblast/lancezarr_testing/v0_lance/")
    num_cells = 1000
    median_counts_per_cell = 2_000
    num_genes_per_cell = np.random.normal(median_counts_per_cell, scale=(median_counts_per_cell * 0.1), size=num_cells).astype(int)

    cell_ranges = np.cumsum(num_genes_per_cell)
    cell_ranges = np.insert(cell_ranges, 0, 0)
    cell_ranges = np.stack([cell_ranges[:-1], cell_ranges[1:]], axis=1)

    dummy_coo_counts = np.random.randint(0, 1000, size=np.sum(num_genes_per_cell).item()).astype(np.int32)
    dummy_coo_indices = np.concatenate([
        np.arange(0, n, dtype=np.int32) for n in num_genes_per_cell
    ], axis=0)

    group_store = ObjectStore(
        S3Store("epiblast", prefix="lancezarr_testing/v0/", region="us-east-2")
    )
    zarr_group = zarr.create_group(group_store, overwrite=True)

    counts_zarray = zarr_group.create_array(
        "counts",
        data=dummy_coo_counts,
        chunks=(5_000,),
        shards=(100_000,),
    )
    indices_zarray = zarr_group.create_array(
        "indices",
        data=dummy_coo_indices,
        chunks=(5_000,),
        shards=(100_000,),
    )

    cell_type_choices = ["A-549", "K-562", "HeLa", "Hep-G2", "MCF7", "HNIL", "NGN2"]
    metadata_df = pd.DataFrame(
        dict(
            cell_uid=[str(uuid.uuid4()) for _ in range(num_cells)],
            cell_type=np.random.choice(cell_type_choices, size=num_cells),
            start_pos=cell_ranges[:, 0],
            end_pos=cell_ranges[:, 1],
        )
    )
    cell_table = db.create_table("cells", data=metadata_df)

    cell_table.optimize()
    cell_table.create_scalar_index("cell_type", replace=True)


def benchmark_load_time():
    db = lancedb.connect("s3://epiblast/lancezarr_testing/v0_lance/")
    cell_table = db.open_table("cells")

    group_store = ObjectStore(
        S3Store("epiblast", prefix="lancezarr_testing/v0/", region="us-east-2")
    )
    zarr_group = zarr.open(group_store, mode="r")
    counts_zarray = zarr_group["counts"]
    indices_zarray = zarr_group["indices"]

    t0 = time()
    cell_query_metadata = cell_table.search().where(
        "cell_type IN ('A-549', 'Hep-G2', 'HeLa', 'MCF7', 'NGN2')"
    ).to_pandas()
    print(f"Queried metadata for {len(cell_query_metadata)} cells in {time() - t0:.4f}s")

    t1 = time()
    counts_zarr_impl = get_codec_pipeline_impl(counts_zarray.metadata, counts_zarray.store, strict=True)
    out_counts, _ = batch_read_zarr(
        counts_zarr_impl,
        starts=cell_query_metadata.start_pos.values,
        ends=cell_query_metadata.end_pos.values,
        shard_shape=counts_zarray.shards,
    )
    print(f"Loaded counts for {len(out_counts)} entries in {time() - t1:.4f}s")

    t2 = time()
    indices_zarr_impl = get_codec_pipeline_impl(indices_zarray.metadata, indices_zarray.store, strict=True)
    out_indices, lengths = batch_read_zarr(
        indices_zarr_impl,
        starts=cell_query_metadata.start_pos.values,
        ends=cell_query_metadata.end_pos.values,
        shard_shape=indices_zarray.shards,
    )
    print(f"Loaded indices for {len(out_indices)} entries in {time() - t2:.4f}s")
    
    return cell_query_metadata, out_counts, out_indices, lengths


if __name__ == "__main__":
    print("Setting up data...")
    setup_data()

    print("\nBenchmarking load time...")
    cell_query_metadata, out_counts, out_indices, lengths = benchmark_load_time()
    print(f"Loaded data for {len(cell_query_metadata)} cells, total counts: {len(out_counts)}")