import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import asyncio
    import time
    from collections import defaultdict

    import os
    import marimo as mo
    import numpy as np
    import lancedb
    import obstore
    import zarr
    from obstore.store import S3Store
    from zarr.storage import ObjectStore
    return (
        ObjectStore,
        S3Store,
        asyncio,
        defaultdict,
        lancedb,
        mo,
        np,
        obstore,
        time,
        zarr,
    )


@app.cell
def _(np, zarr):
    zarr_md = zarr.open("s3://epiblast/lancell_data_structure_test/image_tiles.zarr", mode="w")
    image_tile_data = np.random.randint(0, 255, size=(1000, 5, 72, 72)).astype(np.uint8)
    zarray = zarr_md.create_array(data=image_tile_data, name="tiles", chunks=(20, 1, 72, 72), shards=(500, 1, 72, 72))
    return


@app.cell
def _():
    # os.listdir("/tmp/example_md.zarr/tiles/c/0/0/0/0")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Read a Zarr shard index from S3

    Open the counts zarr array, compute the shard index byte range,
    and extract the index for shard `c/0` using `obstore.get_range`.
    """)
    return


@app.cell
def _(np):
    MAX_UINT64 = 2**64 - 1

    def parse_shard_index(raw_bytes, chunks_per_shard, index_raw_bytes):
        """Parse raw shard index bytes (strip crc32c) into (N, 2) uint64 array."""
        return np.frombuffer(
            bytes(raw_bytes[:index_raw_bytes]), dtype="<u8"
        ).reshape(chunks_per_shard, 2)
    return MAX_UINT64, parse_shard_index


@app.cell
def _(ObjectStore, S3Store, zarr):
    store = S3Store(
        "epiblast",
        prefix="lancell_data_structure_test/image_tiles.zarr/tiles/",
        region="us-east-2",
    )
    arr = zarr.open_array(store=ObjectStore(store, read_only=True), mode="r")

    chunk_size = arr.chunks[0]
    chunks_per_shard = arr.shards[0] // chunk_size

    # Index = 2 x uint64 per chunk = 16 bytes/chunk
    # + 4 bytes crc32c (default index_codecs: BytesCodec + Crc32cCodec)
    INDEX_RAW_BYTES = chunks_per_shard * 2 * 8
    INDEX_TOTAL_BYTES = INDEX_RAW_BYTES + 4

    print(f"array shape={arr.shape}, dtype={arr.dtype}")
    print(f"shards={arr.shards}, chunks={arr.chunks} → chunks_per_shard={chunks_per_shard:,}")
    print(f"index size = {INDEX_RAW_BYTES:,} + 4 crc = {INDEX_TOTAL_BYTES:,} bytes")
    return (
        INDEX_RAW_BYTES,
        INDEX_TOTAL_BYTES,
        chunk_size,
        chunks_per_shard,
        store,
    )


@app.cell
def _(
    INDEX_RAW_BYTES,
    INDEX_TOTAL_BYTES,
    MAX_UINT64,
    chunks_per_shard,
    obstore,
    parse_shard_index,
    store,
):
    # 1D array → shard key is just "c/{shard_index}"
    _shard_key = "c/0/0/0/0"

    _meta = obstore.head(store, _shard_key)
    _file_size = _meta["size"]
    print(f"shard file size: {_file_size:,} bytes")

    # Index is at the END of the shard (default index_location="end")
    _idx_bytes = obstore.get_range(
        store, _shard_key,
        start=_file_size - INDEX_TOTAL_BYTES,
        end=_file_size,
    )
    print(f"fetched index bytes: {len(_idx_bytes):,}")

    _index = parse_shard_index(_idx_bytes, chunks_per_shard, INDEX_RAW_BYTES)

    _populated = _index[:, 0] != MAX_UINT64
    print(f"populated chunks: {int(_populated.sum()):,} / {chunks_per_shard:,}  "
          f"(empty: {chunks_per_shard - int(_populated.sum()):,})")

    print("\nFirst 10 index entries (offset, length):")
    for _i in range(min(10, chunks_per_shard)):
        _off, _ln = _index[_i]
        if _off == MAX_UINT64:
            print(f"  chunk {_i}: EMPTY")
        else:
            print(f"  chunk {_i}: offset={_off:,}  length={_ln:,}  end={_off + _ln:,}")
    return


@app.cell
def _(lancedb, np):
    _S3_BASE = "s3://epiblast/lancell_data_structure_test"
    _METADATA_COLS = ["cell_uid", "tissue", "cell_type", "organism"]

    _meta_tbl = lancedb.connect(f"{_S3_BASE}/approach3_lance").open_table("metadata")
    _where = "tissue = 'brain' AND cell_type = 'neuron'"

    _result = _meta_tbl.search().where(_where).select(
        _METADATA_COLS + ["zarr_start", "zarr_end"]
    ).to_arrow().to_pydict()

    starts = np.array(_result["zarr_start"], dtype=np.int64)
    ends = np.array(_result["zarr_end"], dtype=np.int64)
    print(f"query: {_where}")
    print(f"matched {len(starts):,} cells, total elements: {int((ends - starts).sum()):,}")
    print(f"zarr range span: [{starts.min():,}, {ends.max():,})")
    return ends, starts


@app.cell
async def _(
    INDEX_RAW_BYTES,
    INDEX_TOTAL_BYTES,
    MAX_UINT64,
    asyncio,
    chunk_size,
    chunks_per_shard,
    defaultdict,
    ends,
    obstore,
    parse_shard_index,
    starts,
    store,
):
    # Step 1: map each zarr range to the (shard, local_chunk) pairs it touches
    _shard_chunks = defaultdict(set)
    for _s, _e in zip(starts, ends):
        _first = int(_s) // chunk_size
        _last = (int(_e) - 1) // chunk_size
        for _abs in range(_first, _last + 1):
            _shard_chunks[_abs // chunks_per_shard].add(_abs % chunks_per_shard)

    print(f"shards touched: {len(_shard_chunks)}")

    # Step 2: fetch all shard indexes concurrently, then look up byte ranges
    async def _fetch_shard_index(sid):
        _key = f"c/{sid}"
        _meta = await obstore.head_async(store, _key)
        _fsize = _meta["size"]
        _idx_raw = await obstore.get_range_async(
            store, _key, start=_fsize - INDEX_TOTAL_BYTES, end=_fsize,
        )
        return sid, _key, parse_shard_index(_idx_raw, chunks_per_shard, INDEX_RAW_BYTES)

    _index_results = await asyncio.gather(
        *[_fetch_shard_index(sid) for sid in sorted(_shard_chunks)]
    )

    shard_ranges = {}  # shard_idx -> (shard_key, byte_starts, byte_ends)
    for _sid, _key, _index in _index_results:
        _local = sorted(_shard_chunks[_sid])
        _bstarts = []
        _bends = []
        for _lc in _local:
            _off, _ln = _index[_lc]
            if _off == MAX_UINT64:
                continue
            _bstarts.append(int(_off))
            _bends.append(int(_off + _ln))
        shard_ranges[_sid] = (_key, _bstarts, _bends)
        print(f"  shard {_key}: {len(_local):,} chunks → "
              f"{len(_bstarts):,} non-empty byte ranges")
    return (shard_ranges,)


@app.cell
async def _(asyncio, obstore, shard_ranges, store, time):
    # Fetch all chunk data from all shards concurrently
    async def _fetch_shard_data(shard_key, byte_starts, byte_ends):
        buffers = await obstore.get_ranges_async(
            store, shard_key, starts=byte_starts, ends=byte_ends,
        )
        return shard_key, buffers

    _t0 = time.perf_counter()
    _results = await asyncio.gather(*[
        _fetch_shard_data(key, bstarts, bends)
        for key, bstarts, bends in shard_ranges.values()
    ])
    _elapsed = time.perf_counter() - _t0

    shard_data = {key: bufs for key, bufs in _results}

    _total_bytes = sum(sum(len(b) for b in bufs) for bufs in shard_data.values())
    _total_ranges = sum(len(bufs) for bufs in shard_data.values())
    print(f"fetched {_total_ranges:,} ranges across {len(shard_data)} shards "
          f"({_total_bytes / 1e6:.1f} MB) in {_elapsed:.3f}s")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
