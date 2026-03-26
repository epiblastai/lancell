"""Genomic range queries on genome-sorted chromatin accessibility fragments."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import zarr

from lancell.batch_array import BatchArray

_END_MAX_BLOCK_SIZE = 128


class _RangeReader(Protocol):
    def read_range(self, start: int, end: int) -> np.ndarray: ...


# TODO: Why the fallback?
class _ZarrSliceReader:
    """Fallback reader using plain zarr slicing."""

    def __init__(self, arr: zarr.Array) -> None:
        self._arr = arr

    def read_range(self, start: int, end: int) -> np.ndarray:
        return self._arr[start:end]


class _BatchRangeReader:
    """Reader using RustBatchReader for fast batched I/O."""

    def __init__(self, arr: zarr.Array) -> None:
        self._ba = BatchArray.from_array(arr)

    def read_range(self, start: int, end: int) -> np.ndarray:
        s = np.array([start], dtype=np.int64)
        e = np.array([end], dtype=np.int64)
        data, _ = self._ba.read_ranges(s, e)
        return data


@dataclass
class RegionResult:
    """Fragments overlapping a genomic region.

    Attributes
    ----------
    cell_ids
        uint32 cell indices (into the cell ordering used at ingestion).
    starts
        uint32 genomic start positions.
    lengths
        uint16 fragment lengths (end = start + length).
    chrom_name
        Chromosome name for this query.
    """

    cell_ids: np.ndarray
    starts: np.ndarray
    lengths: np.ndarray
    chrom_name: str


def seek_region(
    chrom_offsets: np.ndarray,
    end_max: np.ndarray,
    chrom_idx: int,
    start: int,
    block_size: int = _END_MAX_BLOCK_SIZE,
) -> tuple[int, int]:
    """Find the element index range that may contain overlapping fragments.

    Uses binary search on the ``end_max`` seek index to skip blocks whose
    maximum end coordinate is below the query start.

    Parameters
    ----------
    chrom_offsets
        int64 boundary array (n_chroms + 1).
    end_max
        uint32 seek index (one value per *block_size* fragments).
    chrom_idx
        Chromosome index to query.
    start
        Query region start coordinate.
    block_size
        Number of fragments per end_max block (must match ingestion).

    Returns
    -------
    (begin_idx, end_idx)
        Element index range within the flat fragment arrays. All fragments
        that could overlap ``[start, ...)`` on this chromosome fall within
        this range. The caller must still filter by end coordinate.
    """
    chr_start = int(chrom_offsets[chrom_idx])
    chr_end = int(chrom_offsets[chrom_idx + 1])
    if chr_start >= chr_end:
        return chr_start, chr_end

    # end_max blocks covering this chromosome
    block_start = chr_start // block_size
    block_end = -(-chr_end // block_size)  # ceil division

    # Binary search for first block where end_max >= start
    chr_end_max = end_max[block_start:block_end]
    rel_block = np.searchsorted(chr_end_max, start, side="left")
    abs_block = block_start + rel_block

    begin_idx = max(chr_start, abs_block * block_size)
    return begin_idx, chr_end


class GenomeSortedReader:
    """Read interface for genome-sorted fragment data.

    Loads the small index arrays (``chrom_offsets``, ``end_max``) eagerly
    and lazily creates ``BatchArray`` readers for the large arrays on
    first query.

    Parameters
    ----------
    group
        Zarr group containing a ``genome_sorted/`` subgroup.
    chrom_names
        Ordered chromosome names matching the ``chrom_offsets`` array.
    """

    def __init__(self, group: zarr.Group, chrom_names: list[str]) -> None:
        gs = group["genome_sorted"]
        self._gs = gs
        self._chrom_names = list(chrom_names)
        self._chrom_to_idx = {name: i for i, name in enumerate(chrom_names)}
        self._chrom_offsets: np.ndarray = gs["chrom_offsets"][:]
        self._end_max: np.ndarray = gs["end_max"][:]

        # Lazy readers
        self._reader_cell_ids: _RangeReader | None = None
        self._reader_starts: _RangeReader | None = None
        self._reader_lengths: _RangeReader | None = None

    def _make_reader(self, arr: zarr.Array) -> _RangeReader:
        try:
            store = arr.store
            if hasattr(store, "store"):
                return _BatchRangeReader(arr)
        except Exception:
            pass
        return _ZarrSliceReader(arr)

    def _ensure_readers(self) -> tuple[_RangeReader, _RangeReader, _RangeReader]:
        if self._reader_cell_ids is None:
            self._reader_cell_ids = self._make_reader(self._gs["cell_ids"])
            self._reader_starts = self._make_reader(self._gs["starts"])
            self._reader_lengths = self._make_reader(self._gs["lengths"])
        return self._reader_cell_ids, self._reader_starts, self._reader_lengths

    @property
    def chrom_names(self) -> list[str]:
        return self._chrom_names

    @property
    def n_fragments(self) -> int:
        return int(self._chrom_offsets[-1])

    def query_region(
        self,
        chrom_name: str,
        start: int,
        end: int,
    ) -> RegionResult:
        """Read all fragments overlapping ``[start, end)`` on a chromosome.

        Parameters
        ----------
        chrom_name
            Chromosome name (e.g. ``"chr1"``).
        start
            Query region start (inclusive).
        end
            Query region end (exclusive).

        Returns
        -------
        RegionResult
            Fragments whose genomic interval ``[frag_start, frag_start + length)``
            overlaps ``[start, end)``.
        """
        chrom_idx = self._chrom_to_idx[chrom_name]
        begin_idx, end_idx = seek_region(
            self._chrom_offsets,
            self._end_max,
            chrom_idx,
            start,
        )

        if begin_idx >= end_idx:
            return RegionResult(
                cell_ids=np.array([], dtype=np.uint32),
                starts=np.array([], dtype=np.uint32),
                lengths=np.array([], dtype=np.uint16),
                chrom_name=chrom_name,
            )

        r_cell_ids, r_starts, r_lengths = self._ensure_readers()

        raw_cell_ids = r_cell_ids.read_range(begin_idx, end_idx)
        raw_starts = r_starts.read_range(begin_idx, end_idx)
        raw_lengths = r_lengths.read_range(begin_idx, end_idx)

        # Filter: keep fragments where frag_end > start AND frag_start < end
        frag_ends = raw_starts.astype(np.int64) + raw_lengths.astype(np.int64)
        mask = (frag_ends > start) & (raw_starts < end)

        return RegionResult(
            cell_ids=raw_cell_ids[mask],
            starts=raw_starts[mask],
            lengths=raw_lengths[mask],
            chrom_name=chrom_name,
        )
