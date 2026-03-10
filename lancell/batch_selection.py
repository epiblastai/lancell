from __future__ import annotations

import asyncio
import concurrent.futures
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import obstore
from numcodecs.zstd import Zstd

from zarr.core.array import Array, AsyncArray
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.core.indexing import BasicIndexer, BasicSelection, ChunkProjection, Indexer

from zarr.core.buffer import BufferPrototype, NDArrayLikeOrScalar
from zarr.core.chunk_grids import ChunkGrid

try:
    from lancell._rust import RustShardReader
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchIndexer(Indexer):
    """Indexer for a batch of basic selections.

    Wraps multiple :class:`BasicIndexer` instances and yields their
    :class:`ChunkProjection` objects with ``out_selection`` offsets adjusted
    so that all results land in a single concatenated output buffer
    (concatenated along axis 0).

    Use ``split_indices`` with :func:`numpy.split` to separate the combined
    result into individual arrays afterward.

    Parameters
    ----------
    selections
        Sequence of basic selections (slices, ints, or tuples thereof).
    array_shape
        Shape of the source array.
    chunk_grid
        Chunk grid of the source array.
    """

    sub_indexers: list[BasicIndexer]
    shape: tuple[int, ...]
    drop_axes: tuple[int, ...]
    split_indices: tuple[int, ...]

    def __init__(
        self,
        selections: Sequence[BasicSelection],
        array_shape: tuple[int, ...],
        chunk_grid: ChunkGrid,
    ) -> None:
        if len(selections) == 0:
            raise ValueError("selections must be non-empty")

        sub_indexers = [BasicIndexer(sel, array_shape, chunk_grid) for sel in selections]

        # All sub-indexers must produce the same dimensionality and drop the
        # same axes so the results can be concatenated along axis 0.
        first_shape = sub_indexers[0].shape
        first_drop = sub_indexers[0].drop_axes
        for i, idx in enumerate(sub_indexers[1:], 1):
            if idx.drop_axes != first_drop:
                raise IndexError(
                    f"all selections must drop the same axes; "
                    f"selection 0 drops {first_drop}, selection {i} drops {idx.drop_axes}"
                )
            if len(idx.shape) != len(first_shape):
                raise IndexError(
                    f"all selections must produce the same number of dimensions; "
                    f"selection 0 has {len(first_shape)}, selection {i} has {len(idx.shape)}"
                )
            if len(first_shape) > 1 and idx.shape[1:] != first_shape[1:]:
                raise IndexError(
                    f"all selections must have matching trailing dimensions; "
                    f"selection 0 has {first_shape[1:]}, selection {i} has {idx.shape[1:]}"
                )

        # Combined shape: sum along axis 0, trailing dims unchanged.
        axis0_sizes = [idx.shape[0] if len(idx.shape) > 0 else 1 for idx in sub_indexers]
        total_axis0 = sum(axis0_sizes)
        trailing = first_shape[1:] if len(first_shape) > 1 else ()
        combined_shape = (total_axis0, *trailing)

        # Cumulative axis-0 sizes (excluding final total) for np.split.
        cumulative: list[int] = []
        running = 0
        for size in axis0_sizes[:-1]:
            running += size
            cumulative.append(running)

        object.__setattr__(self, "sub_indexers", sub_indexers)
        object.__setattr__(self, "shape", combined_shape)
        object.__setattr__(self, "drop_axes", first_drop)
        object.__setattr__(self, "split_indices", tuple(cumulative))

    def __iter__(self) -> Iterator[ChunkProjection]:
        offset = 0
        for idx in self.sub_indexers:
            for chunk_coords, chunk_selection, out_selection, is_complete_chunk in idx:
                if offset > 0:
                    out_selection = _offset_out_selection(out_selection, offset)
                yield ChunkProjection(
                    chunk_coords, chunk_selection, out_selection, is_complete_chunk
                )
            axis0_size = idx.shape[0] if len(idx.shape) > 0 else 1
            offset += axis0_size


def _offset_out_selection(
    out_selection: tuple | slice,
    offset: int,
) -> tuple | slice:
    """Shift the first axis of *out_selection* by *offset*."""
    if isinstance(out_selection, tuple) and len(out_selection) > 0:
        first = out_selection[0]
        if isinstance(first, slice):
            adjusted = slice(first.start + offset, first.stop + offset, first.step)
        else:
            adjusted = first + offset
        return (adjusted, *out_selection[1:])
    if isinstance(out_selection, slice):
        return slice(
            out_selection.start + offset,
            out_selection.stop + offset,
            out_selection.step,
        )
    return out_selection


# ---------------------------------------------------------------------------
# Async array subclass
# ---------------------------------------------------------------------------


class BatchAsyncArray(AsyncArray):
    """AsyncArray subclass with :meth:`get_batch_selection`."""

    async def get_batch_selection(
        self,
        selections: Sequence[BasicSelection],
        *,
        prototype: BufferPrototype | None = None,
    ) -> list[NDArrayLikeOrScalar]:
        """Read multiple basic selections in a single batched operation.

        All chunk I/O across all selections is submitted to the codec pipeline
        in one call, enabling concurrent fetching of all needed chunks.

        Parameters
        ----------
        selections
            A sequence of basic selections (slices, ints, or tuples thereof).
        prototype
            Buffer prototype to use for output data.

        Returns
        -------
        list of array-like or scalar
            One result per selection, in the same order.
        """
        if prototype is None:
            prototype = default_buffer_prototype()
        indexer = BatchIndexer(selections, self.metadata.shape, self.metadata.chunk_grid)
        combined = await self._get_selection(indexer, prototype=prototype)
        return np.split(combined, indexer.split_indices, axis=0)  # type: ignore[arg-type]


class BatchArray(Array):
    """Array subclass with :meth:`get_batch_selection`.

    Create via :meth:`from_array` to wrap an existing :class:`zarr.Array`::

        batch_arr = BatchArray.from_array(zarr.open_array("data.zarr"))
        results = batch_arr.get_batch_selection([slice(0, 100), slice(500, 600)])
    """

    @classmethod
    def from_array(cls, array: Array) -> "BatchArray":
        """Wrap an existing :class:`zarr.Array`."""
        return cls(array._async_array)

    def get_batch_selection(
        self,
        selections: Sequence[BasicSelection],
        *,
        prototype: BufferPrototype | None = None,
    ) -> list[NDArrayLikeOrScalar]:
        """Read multiple basic selections in a single batched operation.

        For sharded arrays whose selections are all 1-D slices, this builds
        ChunkItem objects directly at shard granularity and calls the Rust
        codec pipeline without going through the async zarr read path.
        This avoids the inner-chunk-level addressing that causes redundant
        shard fetches.

        Falls back to the generic async path for non-sharded arrays or
        non-slice selections.
        """
        if prototype is None:
            prototype = default_buffer_prototype()
        indexer = BatchIndexer(selections, self.shape, self.metadata.chunk_grid)
        combined = sync(
            self.async_array._get_selection(indexer, prototype=prototype)
        )
        return np.split(combined, indexer.split_indices, axis=0)


# ---------------------------------------------------------------------------
# Obstore-based shard reader
# ---------------------------------------------------------------------------

_MAX_UINT64 = 2**64 - 1


class ObstoreShardReader:
    """Reads zarr sharded array chunks via obstore, bypassing zarr's store layer."""

    def __init__(self, s3_store, arr: Array):
        """
        Parameters
        ----------
        s3_store : obstore S3Store pointing at the zarr array root
        arr : opened zarr.Array (used only for metadata)
        """
        md = arr.metadata
        self.s3_store = s3_store
        self.dtype = np.dtype(md.dtype.to_native_dtype())

        sharding_codec = md.codecs[0]
        self.chunk_size: int = sharding_codec.chunk_shape[0]
        self.shard_size: int = arr.shards[0]
        self.chunks_per_shard: int = self.shard_size // self.chunk_size

        # Index sizing: 2 x uint64 per chunk + 4 bytes crc32c
        self.index_raw_bytes: int = self.chunks_per_shard * 2 * 8
        self.index_total_bytes: int = self.index_raw_bytes + 4

        # Direct zstd codec for fast decompression (bypasses zarr pipeline overhead)
        self.zstd_codec = Zstd()
        self.chunk_bytes = self.chunk_size * self.dtype.itemsize

        # Create Rust reader if available (extracts S3 config from arr.store internally)
        self._rust_reader = None
        if _HAS_RUST:
            try:
                self._rust_reader = RustShardReader(arr)
            except Exception as e:
                print(f"    [warn] Failed to create RustShardReader, using Python fallback: {e}")

    def parse_shard_index(self, raw_bytes: bytes) -> np.ndarray:
        """Parse raw shard index bytes into (N, 2) uint64 array."""
        return np.frombuffer(
            raw_bytes[: self.index_raw_bytes], dtype="<u8"
        ).reshape(self.chunks_per_shard, 2)

    async def read_ranges(
        self, starts: np.ndarray, ends: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Read element ranges from the sharded array via obstore.

        Parameters
        ----------
        starts, ends : 1-D int64 arrays of element start/end positions.

        Returns
        -------
        (flat_data, lengths) where flat_data is the concatenated result and
        lengths[i] = ends[i] - starts[i].
        """
        if self._rust_reader is not None:
            t0 = time.perf_counter()
            loop = asyncio.get_running_loop()
            raw_bytes, lengths = await loop.run_in_executor(
                None,
                self._rust_reader.read_ranges,
                starts.astype(np.int64),
                ends.astype(np.int64),
            )
            flat_data = np.frombuffer(raw_bytes, dtype=self.dtype)
            t1 = time.perf_counter()
            print(
                f"    read_ranges (rust): {len(starts)} cells, "
                f"total={t1 - t0:.4f}s"
            )
            return flat_data, lengths

        t_start = time.perf_counter()
        chunk_size = self.chunk_size
        chunks_per_shard = self.chunks_per_shard

        # Step 1: Map each range to (absolute_chunk_idx) and track slicing info.
        # For each chunk we need to fetch, record which cell ranges need slices.
        # chunk_requests: abs_chunk_idx -> list of (cell_idx, local_start, local_end)
        chunk_requests: dict[int, list[tuple[int, int, int]]] = defaultdict(list)

        for cell_idx in range(len(starts)):
            s, e = int(starts[cell_idx]), int(ends[cell_idx])
            pos = s
            while pos < e:
                abs_chunk = pos // chunk_size
                local_start = pos % chunk_size
                chunk_end = min(e, (abs_chunk + 1) * chunk_size)
                local_end = local_start + (chunk_end - pos)
                chunk_requests[abs_chunk].append((cell_idx, local_start, local_end))
                pos = chunk_end

        # Step 2: Group chunks by shard
        # shard_chunks: shard_idx -> set of local_chunk_idx
        shard_chunks: dict[int, set[int]] = defaultdict(set)
        for abs_chunk in chunk_requests:
            shard_idx = abs_chunk // chunks_per_shard
            local_chunk = abs_chunk % chunks_per_shard
            shard_chunks[shard_idx].add(local_chunk)

        t_mapping = time.perf_counter()
        n_chunks = len(chunk_requests)
        n_shards = len(shard_chunks)

        # Step 3: Fetch shard indexes concurrently
        async def fetch_shard_index(shard_idx: int):
            key = f"c/{shard_idx}"
            meta = await obstore.head_async(self.s3_store, key)
            fsize = meta["size"]
            idx_raw = await obstore.get_range_async(
                self.s3_store,
                key,
                start=fsize - self.index_total_bytes,
                end=fsize,
            )
            return shard_idx, key, self.parse_shard_index(bytes(idx_raw))

        index_results = await asyncio.gather(
            *[fetch_shard_index(sid) for sid in sorted(shard_chunks)]
        )

        t_index = time.perf_counter()

        # Step 4: Look up byte ranges and fetch compressed data per shard
        # Build mapping: abs_chunk_idx -> decoded numpy array (filled after decode)
        decoded_chunks: dict[int, np.ndarray] = {}
        fetch_bytes_total = 0
        fetch_times = []
        decode_times = []

        async def fetch_and_decode_shard(
            shard_idx: int, shard_key: str, index: np.ndarray
        ):
            nonlocal fetch_bytes_total
            local_chunks = sorted(shard_chunks[shard_idx])
            byte_starts = []
            byte_ends = []
            valid_locals = []
            for lc in local_chunks:
                off, ln = index[lc]
                if off == _MAX_UINT64:
                    # Empty chunk - store zeros
                    abs_chunk = shard_idx * chunks_per_shard + lc
                    decoded_chunks[abs_chunk] = np.zeros(chunk_size, dtype=self.dtype)
                    continue
                byte_starts.append(int(off))
                byte_ends.append(int(off + ln))
                valid_locals.append(lc)

            if not valid_locals:
                return

            t_f0 = time.perf_counter()
            buffers = await obstore.get_ranges_async(
                self.s3_store,
                shard_key,
                starts=byte_starts,
                ends=byte_ends,
            )
            t_f1 = time.perf_counter()

            fetch_bytes_total += sum(len(b) for b in buffers)
            fetch_times.append(t_f1 - t_f0)

            # Decode chunks: zstd decompress directly via numcodecs (bypasses zarr pipeline)
            t_d0 = time.perf_counter()
            zstd = self.zstd_codec
            dtype = self.dtype

            def _decode_one(buf):
                return np.frombuffer(zstd.decode(bytes(buf)), dtype=dtype)

            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                decoded = list(await asyncio.gather(
                    *[loop.run_in_executor(pool, _decode_one, buf) for buf in buffers]
                ))
            t_d1 = time.perf_counter()

            decode_times.append(t_d1 - t_d0)

            for lc, chunk_arr in zip(valid_locals, decoded):
                abs_chunk = shard_idx * chunks_per_shard + lc
                decoded_chunks[abs_chunk] = chunk_arr

        await asyncio.gather(
            *[
                fetch_and_decode_shard(sid, key, idx)
                for sid, key, idx in index_results
            ]
        )

        t_fetch_decode = time.perf_counter()
        total_fetch = sum(fetch_times)
        total_decode = sum(decode_times)

        # Step 6: Slice decoded chunks to extract requested elements
        lengths = ends - starts
        total = int(lengths.sum())
        out = np.empty(total, dtype=self.dtype)
        out_pos = 0
        for cell_idx in range(len(starts)):
            s, e = int(starts[cell_idx]), int(ends[cell_idx])
            pos = s
            while pos < e:
                abs_chunk = pos // chunk_size
                local_start = pos % chunk_size
                chunk_end = min(e, (abs_chunk + 1) * chunk_size)
                local_end = local_start + (chunk_end - pos)
                n = local_end - local_start
                out[out_pos : out_pos + n] = decoded_chunks[abs_chunk][
                    local_start:local_end
                ]
                out_pos += n
                pos = chunk_end

        t_slice = time.perf_counter()

        print(
            f"    read_ranges: {len(starts)} cells, {n_chunks} chunks, {n_shards} shards, "
            f"{fetch_bytes_total/1e6:.1f}MB | "
            f"map={t_mapping - t_start:.4f}s  "
            f"idx={t_index - t_mapping:.4f}s  "
            f"fetch={total_fetch:.4f}s({len(fetch_times)}shards)  "
            f"decode={total_decode:.4f}s  "
            f"fetch+decode_wall={t_fetch_decode - t_index:.4f}s  "
            f"slice={t_slice - t_fetch_decode:.4f}s  "
            f"total={t_slice - t_start:.4f}s"
        )

        return out, lengths
