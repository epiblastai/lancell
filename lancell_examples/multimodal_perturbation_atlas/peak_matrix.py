"""Convert chromatin accessibility fragments into a cells × ranges count matrix.

Given a set of genomic ranges (peaks, bins, tiles) and a
:class:`~lancell_examples.multimodal_perturbation_atlas.interval_reconstruction.FragmentResult`,
produces a sparse count matrix where entry ``(i, j)`` is the number of
fragments from cell *i* that overlap range *j*.

A fragment ``[frag_start, frag_start + frag_length)`` overlaps a range
``[range_start, range_end)`` when both conditions hold:

- ``frag_start < range_end``
- ``frag_start + frag_length > range_start``
"""

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from lancell_examples.multimodal_perturbation_atlas.interval_reconstruction import (
    FragmentResult,
)


@dataclass(frozen=True)
class GenomicRange:
    """A single genomic interval.

    Parameters
    ----------
    chrom
        Chromosome name (e.g. ``"chr1"``).
    start
        0-based start position (inclusive).
    end
        0-based end position (exclusive).
    name
        Optional identifier for the range. When empty, ``to_anndata``
        auto-generates names as ``"chrom:start-end"``.
    """

    chrom: str
    start: int
    end: int
    name: str = ""


@dataclass
class _PeakSet:
    """Vectorized, per-chromosome representation of sorted genomic ranges."""

    chrom_order: list[str]
    starts: dict[str, np.ndarray]  # {chrom: uint32 sorted start positions}
    ends: dict[str, np.ndarray]  # {chrom: uint32 sorted end positions}
    chrom_offsets: dict[str, int]  # {chrom: global index offset for this chrom's peaks}
    names: list[str]
    n_peaks: int


def _build_peak_set(ranges: list[GenomicRange]) -> _PeakSet:
    """Group, sort, and vectorize genomic ranges by chromosome."""
    if not ranges:
        return _PeakSet(
            chrom_order=[],
            starts={},
            ends={},
            chrom_offsets={},
            names=[],
            n_peaks=0,
        )

    # Group by chromosome
    by_chrom: dict[str, list[GenomicRange]] = {}
    for r in ranges:
        by_chrom.setdefault(r.chrom, []).append(r)

    # Deterministic chromosome order (sorted)
    chrom_order = sorted(by_chrom)

    starts: dict[str, np.ndarray] = {}
    ends: dict[str, np.ndarray] = {}
    chrom_offsets: dict[str, int] = {}
    names: list[str] = []
    offset = 0

    for chrom in chrom_order:
        chrom_ranges = sorted(by_chrom[chrom], key=lambda r: r.start)
        chrom_offsets[chrom] = offset

        starts[chrom] = np.array([r.start for r in chrom_ranges], dtype=np.int64)
        ends[chrom] = np.array([r.end for r in chrom_ranges], dtype=np.int64)
        names.extend(r.name if r.name else f"{r.chrom}:{r.start}-{r.end}" for r in chrom_ranges)
        offset += len(chrom_ranges)

    return _PeakSet(
        chrom_order=chrom_order,
        starts=starts,
        ends=ends,
        chrom_offsets=chrom_offsets,
        names=names,
        n_peaks=offset,
    )


def _count_chromosome(
    frag_starts: np.ndarray,
    frag_ends: np.ndarray,
    frag_cell_ids: np.ndarray,
    peak_starts: np.ndarray,
    peak_ends: np.ndarray,
    peak_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Count fragment-peak overlaps for one chromosome.

    Uses double searchsorted: for each fragment, the overlapping peaks
    are the contiguous index range ``[k, j)`` where::

        j = searchsorted(peak_starts, frag_end, side='left')
        k = searchsorted(peak_ends, frag_start, side='right')

    Returns
    -------
    (row_indices, col_indices)
        COO components for the sparse count matrix. ``row_indices`` are
        cell indices, ``col_indices`` are global peak indices.
    """
    j = np.searchsorted(peak_starts, frag_ends, side="left")
    k = np.searchsorted(peak_ends, frag_starts, side="right")

    counts = np.maximum(j - k, 0)

    if counts.max() <= 1:
        # Fast path: each fragment overlaps at most one peak
        mask = counts == 1
        if not mask.any():
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        return frag_cell_ids[mask], k[mask] + peak_offset
    else:
        # General path: some fragments overlap multiple peaks
        has_overlap = counts > 0
        frag_idx = np.repeat(np.where(has_overlap)[0], counts[has_overlap])
        peak_idx = np.concatenate(
            [np.arange(ki, ji) for ki, ji in zip(k[has_overlap], j[has_overlap], strict=False)]
        )
        return frag_cell_ids[frag_idx], peak_idx + peak_offset


class FragmentCounter:
    """Count fragments overlapping user-provided genomic ranges.

    Given a set of genomic ranges and a
    :class:`~lancell_examples.multimodal_perturbation_atlas.interval_reconstruction.FragmentResult`,
    produces a sparse count matrix of shape ``(n_cells, n_ranges)``.

    Parameters
    ----------
    ranges
        Genomic ranges defining the regions to count. Within each
        chromosome, ranges are sorted by start internally. Non-overlapping
        ranges (the typical case for MACS2 peaks or fixed-width bins)
        enable a fast path in the counting algorithm.

    Examples
    --------
    >>> peaks = [GenomicRange("chr1", 1000, 2000), GenomicRange("chr1", 5000, 6000)]
    >>> counter = FragmentCounter(peaks)
    >>> matrix = counter.count_fragments(fragment_result)
    """

    def __init__(self, ranges: list[GenomicRange]) -> None:
        self._peak_set = _build_peak_set(ranges)
        self._ranges = ranges

    @property
    def n_ranges(self) -> int:
        """Total number of genomic ranges."""
        return self._peak_set.n_peaks

    @property
    def var(self) -> pd.DataFrame:
        """Peak annotations as a DataFrame (chrom, start, end), indexed by name."""
        ps = self._peak_set
        if ps.n_peaks == 0:
            return pd.DataFrame(columns=["chrom", "start", "end"])

        chroms: list[str] = []
        starts_list: list[int] = []
        ends_list: list[int] = []
        for chrom in ps.chrom_order:
            n = len(ps.starts[chrom])
            chroms.extend([chrom] * n)
            starts_list.extend(ps.starts[chrom].tolist())
            ends_list.extend(ps.ends[chrom].tolist())

        return pd.DataFrame(
            {"chrom": chroms, "start": starts_list, "end": ends_list},
            index=pd.Index(ps.names, name="name"),
        )

    def count_fragments(self, fragments: FragmentResult) -> sp.csr_matrix:
        """Count fragment overlaps per cell per range.

        Parameters
        ----------
        fragments
            Fragment data from
            :meth:`~lancell_examples.multimodal_perturbation_atlas.interval_reconstruction.IntervalReconstructor.as_fragments`.

        Returns
        -------
        scipy.sparse.csr_matrix
            Shape ``(n_cells, n_ranges)``, dtype int32. Entry ``(i, j)``
            is the number of fragments from cell *i* overlapping range *j*.
        """
        ps = self._peak_set
        n_cells = len(fragments.offsets) - 1

        if ps.n_peaks == 0 or n_cells == 0:
            return sp.csr_matrix((n_cells, ps.n_peaks), dtype=np.int32)

        # Derive per-fragment cell IDs from CSR offsets
        cell_ids = np.repeat(np.arange(n_cells, dtype=np.int64), np.diff(fragments.offsets))

        # Pre-compute fragment ends once
        frag_ends = fragments.starts.astype(np.int64) + fragments.lengths.astype(np.int64)

        # Build chrom_name → chrom_idx map for fragments
        frag_chrom_to_idx = {name: i for i, name in enumerate(fragments.chrom_names)}

        row_parts: list[np.ndarray] = []
        col_parts: list[np.ndarray] = []

        for chrom in ps.chrom_order:
            if chrom not in frag_chrom_to_idx:
                continue

            chrom_idx = frag_chrom_to_idx[chrom]
            mask = fragments.chromosomes == chrom_idx

            if not mask.any():
                continue

            rows, cols = _count_chromosome(
                frag_starts=fragments.starts[mask].astype(np.int64),
                frag_ends=frag_ends[mask],
                frag_cell_ids=cell_ids[mask],
                peak_starts=ps.starts[chrom],
                peak_ends=ps.ends[chrom],
                peak_offset=ps.chrom_offsets[chrom],
            )

            if len(rows) > 0:
                row_parts.append(rows)
                col_parts.append(cols)

        if not row_parts:
            return sp.csr_matrix((n_cells, ps.n_peaks), dtype=np.int32)

        all_rows = np.concatenate(row_parts)
        all_cols = np.concatenate(col_parts)
        data = np.ones(len(all_rows), dtype=np.int32)

        matrix = sp.coo_matrix((data, (all_rows, all_cols)), shape=(n_cells, ps.n_peaks))
        return matrix.tocsr()

    def to_anndata(self, fragments: FragmentResult) -> ad.AnnData:
        """Count fragments and wrap in AnnData with obs and var metadata.

        Parameters
        ----------
        fragments
            Fragment data from ``IntervalReconstructor.as_fragments()``.

        Returns
        -------
        anndata.AnnData
            ``.X`` is the count matrix, ``.obs`` from ``fragments.obs``,
            ``.var`` has columns ``chrom``, ``start``, ``end``.
        """
        matrix = self.count_fragments(fragments)
        return ad.AnnData(X=matrix, obs=fragments.obs, var=self.var)
