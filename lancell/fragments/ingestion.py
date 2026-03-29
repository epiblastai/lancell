"""Fragment ingestion utilities for chromatin accessibility data.

Parses gzipped BED files, sorts fragments into cell order and genome order,
and writes the parallel zarr arrays expected by ``CHROMATIN_ACCESSIBILITY_SPEC``:

Cell-sorted (per-cell access via ``SparseZarrPointer``):

- ``cell_sorted/chromosomes`` (uint8)
- ``cell_sorted/starts`` (uint32)
- ``cell_sorted/lengths`` (uint16)

Genome-sorted (fast genomic range queries):

- ``genome_sorted/cell_ids`` (uint32)
- ``genome_sorted/starts`` (uint32)
- ``genome_sorted/lengths`` (uint16)
- ``genome_sorted/chrom_offsets`` (int64)
- ``genome_sorted/end_max`` (uint32)
"""

import re
from pathlib import Path

import numpy as np
import polars as pl
import zarr

from lancell.codecs.bitpacking import BitpackingCodec
from lancell.ingestion import _CHUNK_ELEMS, _SHARD_ELEMS

FEATURE_SPACE = "chromatin_accessibility"

# Regex for extracting the numeric suffix from chromosome names like "chr12"
_CHR_NUM_RE = re.compile(r"^chr(\d+)$")

# Named chromosomes that sort after numbered autosomes
_NAMED_CHROM_RANK = {"chrX": 0, "chrY": 1, "chrM": 2}

_END_MAX_BLOCK_SIZE = 128


def parse_bed_fragments(
    path: Path,
    barcode_col: str = "barcode",
) -> pl.DataFrame:
    """Parse a gzipped BED fragment file into a polars DataFrame.

    Supports both 4-column BED (``chrom, start, end, barcode``) and
    5-column 10x format (``chrom, start, end, barcode, count``). The
    column count is auto-detected; any columns beyond the fourth are
    dropped.

    Parameters
    ----------
    path
        Path to a (possibly gzipped) BED fragment file.
    barcode_col
        Name to assign to the fourth column (cell barcode / identifier).
        Defaults to ``"barcode"``.

    Returns
    -------
    pl.DataFrame
        Columns: ``chrom`` (str), ``start`` (uint32), ``length`` (uint16),
        ``<barcode_col>`` (str).
    """
    df = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        # Read only the first 4 columns regardless of how many the file has
        # These are (chrom, start, end, barcode) in both 4-col and 5-col formats
        # the fifth column in 10x format is a count. Currently we are choosing
        # to ignore the counts, which are usually 1, and only occasionally higher
        columns=[0, 1, 2, 3],
        new_columns=["chrom", "start", "end", barcode_col],
        schema_overrides={"start": pl.UInt32, "end": pl.UInt32},
    )
    df = df.with_columns(
        (pl.col("end") - pl.col("start")).cast(pl.UInt16).alias("length"),
    ).drop("end")
    return df


def build_chrom_order(fragments: pl.DataFrame) -> list[str]:
    """Build a deterministic chromosome ordering from fragment data.

    Sorts chromosomes as: numbered autosomes (chr1-chr22) in numeric
    order, then chrX, chrY, chrM, then any remaining scaffolds or
    contigs in alphabetical order.

    Parameters
    ----------
    fragments
        DataFrame from :func:`parse_bed_fragments` with a ``chrom`` column.

    Returns
    -------
    list[str]
        Ordered chromosome names. Length is guaranteed <= 256 (uint8 range).
    """
    unique_chroms: list[str] = fragments["chrom"].unique().sort().to_list()

    numbered: list[tuple[int, str]] = []
    named: list[tuple[int, str]] = []
    other: list[str] = []

    for chrom in unique_chroms:
        m = _CHR_NUM_RE.match(chrom)
        if m:
            numbered.append((int(m.group(1)), chrom))
        elif chrom in _NAMED_CHROM_RANK:
            named.append((_NAMED_CHROM_RANK[chrom], chrom))
        else:
            other.append(chrom)

    numbered.sort(key=lambda t: t[0])
    named.sort(key=lambda t: t[0])
    other.sort()

    result = [c for _, c in numbered] + [c for _, c in named] + other

    if len(result) > 256:
        raise ValueError(
            f"Too many unique chromosomes ({len(result)}) to encode as uint8. "
            "Filter the fragments to primary assembly sequences."
        )
    return result


def sort_fragments_by_cell(
    fragments: pl.DataFrame,
    chrom_order: list[str],
    barcode_col: str = "barcode",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Sort fragments into cell order and return flat arrays with offsets.

    Fragments are sorted by ``(cell_idx, chrom_idx, start)`` so that
    within each cell, fragments are ordered by genomic position. This
    makes the ``starts`` array highly compressible with delta encoding.

    Parameters
    ----------
    fragments
        DataFrame from :func:`parse_bed_fragments`.
    chrom_order
        Chromosome ordering from :func:`build_chrom_order`.
    barcode_col
        Name of the barcode column in *fragments*. Defaults to ``"barcode"``.

    Returns
    -------
    tuple of (chromosomes, starts, lengths, offsets, cell_ids)
        chromosomes : np.ndarray[uint8]
            Chromosome index for each fragment.
        starts : np.ndarray[uint32]
            Genomic start position for each fragment.
        lengths : np.ndarray[uint16]
            Fragment length (end - start) for each fragment.
        offsets : np.ndarray[int64]
            CSR-style indptr array of length ``n_cells + 1``.
            Cell *j*'s fragments are at indices ``offsets[j]:offsets[j+1]``.
        cell_ids : list[str]
            Ordered list of barcode strings (one per cell).
    """
    chrom_to_idx = {name: np.uint8(i) for i, name in enumerate(chrom_order)}

    # Encode chromosomes
    df = fragments.with_columns(
        pl.col("chrom").replace_strict(chrom_to_idx, return_dtype=pl.UInt8).alias("chrom_idx"),
    ).drop("chrom")

    # Build deterministic cell ordering from unique barcodes
    cell_ids: list[str] = df[barcode_col].unique().sort().to_list()
    n_cells = len(cell_ids)
    barcode_to_idx = {g: np.uint32(i) for i, g in enumerate(cell_ids)}

    # Map barcode → cell_idx and sort
    df = (
        df.with_columns(
            pl.col(barcode_col)
            .replace_strict(barcode_to_idx, return_dtype=pl.UInt32)
            .alias("cell_idx"),
        )
        .drop(barcode_col)
        .sort("cell_idx", "chrom_idx", "start")
    )

    # Extract flat arrays
    chromosomes = df["chrom_idx"].to_numpy()
    starts = df["start"].to_numpy()
    lengths = df["length"].to_numpy()

    # Build CSR-style offsets from per-cell fragment counts
    counts_df = df.group_by("cell_idx").len().sort("cell_idx")
    # Some cells may have 0 fragments -- fill in the gaps
    cell_counts = np.zeros(n_cells, dtype=np.int64)
    cell_indices = counts_df["cell_idx"].to_numpy()
    cell_counts[cell_indices] = counts_df["len"].to_numpy()

    offsets = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(cell_counts, out=offsets[1:])

    return chromosomes, starts, lengths, offsets, cell_ids


def write_fragment_arrays(
    group: zarr.Group,
    chromosomes: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    chunk_shape: tuple[int] | None = None,
    shard_shape: tuple[int] | None = None,
) -> None:
    """Write the three parallel fragment arrays to a zarr group.

    Creates arrays at ``cell_sorted/chromosomes``, ``cell_sorted/starts``,
    and ``cell_sorted/lengths``, matching the layout expected by
    ``CHROMATIN_ACCESSIBILITY_SPEC`` and ``IntervalReconstructor``.

    Parameters
    ----------
    group
        Zarr group to write into (e.g. ``atlas._root.create_group(uid)``).
    chromosomes
        uint8 flat array of chromosome indices.
    starts
        uint32 flat array of genomic start positions.
    lengths
        uint16 flat array of fragment lengths.
    chunk_shape
        Zarr chunk shape. Defaults to ``(_CHUNK_ELEMS,)``.
    shard_shape
        Zarr shard shape. Defaults to ``(_SHARD_ELEMS,)``.
    """
    chunk_shape = chunk_shape or (_CHUNK_ELEMS,)
    shard_shape = shard_shape or (_SHARD_ELEMS,)

    n_fragments = len(chromosomes)
    batch_size = shard_shape[0]

    cell_sorted = group.create_group("cell_sorted")

    zarr_chroms = cell_sorted.create_array(
        "chromosomes",
        shape=(n_fragments,),
        dtype=np.uint8,
        chunks=chunk_shape,
        shards=shard_shape,
    )
    zarr_starts = cell_sorted.create_array(
        "starts",
        shape=(n_fragments,),
        dtype=np.uint32,
        chunks=chunk_shape,
        shards=shard_shape,
        compressors=BitpackingCodec(transform="delta"),
    )
    zarr_lengths = cell_sorted.create_array(
        "lengths",
        shape=(n_fragments,),
        dtype=np.uint16,
        chunks=chunk_shape,
        shards=shard_shape,
    )

    # Write in shard-sized batches
    written = 0
    while written < n_fragments:
        end = min(written + batch_size, n_fragments)
        zarr_chroms[written:end] = chromosomes[written:end]
        zarr_starts[written:end] = starts[written:end]
        zarr_lengths[written:end] = lengths[written:end]
        written = end


# ---------------------------------------------------------------------------
# Genome-sorted storage
# ---------------------------------------------------------------------------


def sort_fragments_by_genome(
    fragments: pl.DataFrame,
    chrom_order: list[str],
    cell_ids: list[str],
    barcode_col: str = "barcode",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort fragments into genome order and return flat arrays.

    Fragments are sorted by ``(chrom_idx, start)`` globally so that
    sequential reads sweep through genomic coordinates. This enables
    fast region queries via the ``end_max`` seek index.

    Parameters
    ----------
    fragments
        DataFrame from :func:`parse_bed_fragments`.
    chrom_order
        Chromosome ordering from :func:`build_chrom_order`.
    cell_ids
        Ordered cell id list from :func:`sort_fragments_by_cell`
        (ensures consistent cell numbering with cell-sorted storage).
    barcode_col
        Name of the barcode column in *fragments*. Defaults to ``"barcode"``.

    Returns
    -------
    tuple of (cell_id_indices, starts, lengths, chrom_offsets)
        cell_id_indices : np.ndarray[uint32]
            Index into *cell_ids* for each fragment.
        starts : np.ndarray[uint32]
            Genomic start position, sorted within each chromosome.
        lengths : np.ndarray[uint16]
            Fragment length (end - start).
        chrom_offsets : np.ndarray[int64]
            Boundary array of length ``len(chrom_order) + 1``.
            Fragments for chromosome *c* are at
            ``chrom_offsets[c]:chrom_offsets[c+1]``.
    """
    chrom_to_idx = {name: np.uint8(i) for i, name in enumerate(chrom_order)}
    barcode_to_idx = {g: np.uint32(i) for i, g in enumerate(cell_ids)}

    df = (
        fragments.with_columns(
            pl.col("chrom").replace_strict(chrom_to_idx, return_dtype=pl.UInt8).alias("chrom_idx"),
            pl.col(barcode_col)
            .replace_strict(barcode_to_idx, return_dtype=pl.UInt32)
            .alias("cell_idx"),
        )
        .drop("chrom", barcode_col)
        .sort("chrom_idx", "start")
    )

    cell_id_indices = df["cell_idx"].to_numpy()
    starts = df["start"].to_numpy()
    lengths = df["length"].to_numpy()

    # Build chromosome boundary array
    n_chroms = len(chrom_order)
    chrom_offsets = np.zeros(n_chroms + 1, dtype=np.int64)
    chrom_indices = df["chrom_idx"].to_numpy()
    counts = np.bincount(chrom_indices, minlength=n_chroms)
    np.cumsum(counts, out=chrom_offsets[1:])

    return cell_id_indices, starts, lengths, chrom_offsets


def build_end_max(
    starts: np.ndarray,
    lengths: np.ndarray,
    block_size: int = _END_MAX_BLOCK_SIZE,
) -> np.ndarray:
    """Compute the seek index for genome-sorted fragments.

    For each block of *block_size* consecutive fragments, stores the
    maximum end coordinate (``start + length``). This allows binary
    search to skip blocks that cannot overlap a query region.

    Parameters
    ----------
    starts
        uint32 array of genomic start positions (genome-sorted).
    lengths
        uint16 array of fragment lengths.
    block_size
        Number of fragments per block. Defaults to 128 (matching the
        BitpackingCodec internal block size).

    Returns
    -------
    np.ndarray[uint32]
        Array of length ``ceil(len(starts) / block_size)``.
    """
    n = len(starts)
    if n == 0:
        return np.array([], dtype=np.uint32)

    ends = starts.astype(np.uint32) + lengths.astype(np.uint32)

    n_blocks = -(-n // block_size)  # ceil division
    # Pad to a multiple of block_size for reshape
    if n % block_size != 0:
        padded = np.zeros(n_blocks * block_size, dtype=np.uint32)
        padded[:n] = ends
        ends = padded

    return ends.reshape(n_blocks, block_size).max(axis=1).astype(np.uint32)


def write_genome_sorted_arrays(
    group: zarr.Group,
    cell_id_indices: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    chrom_offsets: np.ndarray,
    end_max: np.ndarray,
    chunk_shape: tuple[int] | None = None,
    shard_shape: tuple[int] | None = None,
) -> None:
    """Write genome-sorted fragment arrays to a zarr group.

    Creates arrays under ``genome_sorted/`` for fast genomic range
    queries. The three large arrays (``cell_ids``, ``starts``,
    ``lengths``) are sharded, while the small index arrays
    (``chrom_offsets``, ``end_max``) are stored as simple chunked arrays.

    Parameters
    ----------
    group
        Zarr group to write into (same group used for cell-sorted data).
    cell_id_indices
        uint32 flat array of cell indices.
    starts
        uint32 flat array of genomic start positions (genome-sorted).
    lengths
        uint16 flat array of fragment lengths.
    chrom_offsets
        int64 chromosome boundary array (length = n_chroms + 1).
    end_max
        uint32 seek index (length = ceil(n_fragments / 128)).
    chunk_shape
        Zarr chunk shape for large arrays. Defaults to ``(_CHUNK_ELEMS,)``.
    shard_shape
        Zarr shard shape for large arrays. Defaults to ``(_SHARD_ELEMS,)``.
    """
    chunk_shape = chunk_shape or (_CHUNK_ELEMS,)
    shard_shape = shard_shape or (_SHARD_ELEMS,)

    n_fragments = len(starts)
    batch_size = shard_shape[0]

    genome_sorted = group.create_group("genome_sorted")

    # Large sharded arrays
    zarr_cell_ids = genome_sorted.create_array(
        "cell_ids",
        shape=(n_fragments,),
        dtype=np.uint32,
        chunks=chunk_shape,
        shards=shard_shape,
        compressors=BitpackingCodec(transform="none"),
    )
    zarr_starts = genome_sorted.create_array(
        "starts",
        shape=(n_fragments,),
        dtype=np.uint32,
        chunks=chunk_shape,
        shards=shard_shape,
        compressors=BitpackingCodec(transform="delta"),
    )
    zarr_lengths = genome_sorted.create_array(
        "lengths",
        shape=(n_fragments,),
        dtype=np.uint16,
        chunks=chunk_shape,
        shards=shard_shape,
    )

    # Write large arrays in shard-sized batches
    written = 0
    while written < n_fragments:
        end = min(written + batch_size, n_fragments)
        zarr_cell_ids[written:end] = cell_id_indices[written:end]
        zarr_starts[written:end] = starts[written:end]
        zarr_lengths[written:end] = lengths[written:end]
        written = end

    # Small index arrays (not sharded)
    genome_sorted.create_array(
        "chrom_offsets",
        data=chrom_offsets.astype(np.int64),
    )
    genome_sorted.create_array(
        "end_max",
        data=end_max.astype(np.uint32),
    )
