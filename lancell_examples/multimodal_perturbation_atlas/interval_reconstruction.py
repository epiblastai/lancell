"""Interval-based reconstruction for chromatin accessibility fragments."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import polars as pl

from lancell.group_specs import ZarrGroupSpec
from lancell.obs_alignment import PointerFieldInfo
from lancell.read import (
    _prepare_sparse_cells,
    _read_parallel_arrays,
    _sync_gather,
)
from lancell.reconstruction import (
    _build_obs_df,
    _load_remaps_and_features,
)

if TYPE_CHECKING:
    import anndata as ad

    from lancell.atlas import RaggedAtlas


@dataclass
class FragmentResult:
    """Cell-sorted chromatin accessibility fragments.

    The three flat arrays (``chromosomes``, ``starts``, ``lengths``) are
    parallel — element *i* across all three describes one fragment.
    ``offsets`` is a CSR-style indptr of length ``n_cells + 1``:
    fragments for cell *j* are at indices ``offsets[j]:offsets[j+1]``.
    """

    chromosomes: np.ndarray  # uint8, flat — position in chrom_names
    starts: np.ndarray  # uint32, flat — genomic start positions
    lengths: np.ndarray  # uint16, flat — fragment length (end - start)
    offsets: np.ndarray  # int64, CSR-style indptr (n_cells + 1)
    chrom_names: list[str]  # chrom_names[idx] = sequence_name
    obs: pd.DataFrame  # cell metadata


def _resolve_chrom_names(
    atlas: "RaggedAtlas",
    feature_space: str,
    joined_globals: np.ndarray,
) -> list[str]:
    """Look up chromosome sequence_name values for joined global indices."""
    if len(joined_globals) == 0:
        return []
    registry_table = atlas._registry_tables[feature_space]
    indices_sql = ", ".join(str(i) for i in joined_globals.tolist())
    registry_df = (
        registry_table.search()
        .where(f"global_index IN ({indices_sql})", prefilter=True)
        .select(["global_index", "sequence_name"])
        .to_polars()
        .sort("global_index")
    )
    return registry_df["sequence_name"].to_list()


class IntervalReconstructor:
    """Reconstruct chromatin accessibility data as raw genomic fragments.

    Fragments are stored as three parallel 1D arrays (chromosomes, starts,
    lengths) with sparse pointers giving per-cell ranges.  This data cannot
    be represented as a cell-by-feature AnnData matrix, so :meth:`as_anndata`
    raises :class:`NotImplementedError`.  Use :meth:`as_fragments` instead.
    """

    def as_anndata(
        self,
        atlas: "RaggedAtlas",
        cells_pl: "pl.DataFrame",
        pf: "PointerFieldInfo",
        spec: "ZarrGroupSpec",
        layer_overrides: "list[str] | None" = None,
        feature_join: "Literal['union', 'intersection']" = "union",
        wanted_globals: "np.ndarray | None" = None,
    ) -> "ad.AnnData":
        # TODO: Consider returning this in the SnapATAC2 format with fragments stored in
        # obsm and the dimenionsality of the array is num_cells x len(genome). Non-zero values
        # are run lengths (positive or negative depending on strand), therefore `indices` of the
        # array are the global genomic positions.
        raise NotImplementedError(
            "Chromatin accessibility fragments cannot be represented as AnnData. "
            "Use IntervalReconstructor.as_fragments() instead."
        )

    def as_fragments(
        self,
        atlas: "RaggedAtlas",
        cells_pl: pl.DataFrame,
        pf: PointerFieldInfo,
        spec: ZarrGroupSpec,
    ) -> FragmentResult:
        """Read cell-sorted fragment arrays and return raw intervals.

        Parameters
        ----------
        atlas:
            The atlas to read from.
        cells_pl:
            Polars DataFrame of cell rows (must include the chromatin
            accessibility zarr pointer column).
        pf:
            Pointer field info for chromatin_accessibility.
        spec:
            The ``CHROMATIN_ACCESSIBILITY_SPEC`` zarr group spec.

        Returns
        -------
        FragmentResult
            Flat fragment arrays with CSR-style offsets and chromosome names.
        """
        cells_pl_original = cells_pl
        cells_pl, groups = _prepare_sparse_cells(cells_pl, pf)
        if not groups:
            return FragmentResult(
                chromosomes=np.array([], dtype=np.uint8),
                starts=np.array([], dtype=np.uint32),
                lengths=np.array([], dtype=np.uint16),
                offsets=np.zeros(cells_pl_original.height + 1, dtype=np.int64),
                chrom_names=[],
                obs=_build_obs_df(cells_pl_original),
            )

        # Build unified chromosome space across groups
        _, joined_globals, group_remap_to_joined, _ = _load_remaps_and_features(
            atlas,
            groups,
            spec,
            feature_join="union",
        )
        chrom_names = _resolve_chrom_names(atlas, spec.feature_space, joined_globals)

        # Array names from spec (chromosomes, starts, lengths)
        array_names = [a.array_name for a in spec.required_arrays]

        # Prepare per-group readers and ranges
        group_data: list[tuple[str, pl.DataFrame, np.ndarray, np.ndarray, list]] = []
        for zg in groups:
            group_cells = cells_pl.filter(pl.col("_zg") == zg)
            starts = group_cells["_start"].to_numpy().astype(np.int64)
            ends = group_cells["_end"].to_numpy().astype(np.int64)
            gr = atlas._get_group_reader(zg, spec.feature_space)
            readers = [gr.get_array_reader(name) for name in array_names]
            group_data.append((zg, group_cells, starts, ends, readers))

        # Dispatch all groups concurrently
        all_results = _sync_gather(
            [
                _read_parallel_arrays(readers, starts, ends)
                for _, _, starts, ends, readers in group_data
            ]
        )

        # Assemble across groups
        chrom_parts: list[np.ndarray] = []
        start_parts: list[np.ndarray] = []
        length_parts: list[np.ndarray] = []
        cell_length_parts: list[np.ndarray] = []
        obs_parts: list[pl.DataFrame] = []

        for (zg, group_cells, _, _, _), group_results in zip(group_data, all_results, strict=True):
            # group_results: [(flat_data, per_cell_lengths), ...] for each array
            # All 3 arrays share the same ranges so per_cell_lengths are identical
            chroms_flat, cell_lengths = group_results[0]
            starts_flat, _ = group_results[1]
            lengths_flat, _ = group_results[2]

            # Remap local chromosome indices to unified positions
            if zg in group_remap_to_joined:
                joined_remap = group_remap_to_joined[zg]
                chroms_flat = joined_remap[chroms_flat.astype(np.intp)].astype(np.uint8)

            chrom_parts.append(chroms_flat)
            start_parts.append(starts_flat)
            length_parts.append(lengths_flat)
            cell_length_parts.append(cell_lengths)
            obs_parts.append(group_cells)

        # Concatenate flat arrays
        chromosomes = np.concatenate(chrom_parts)
        starts_out = np.concatenate(start_parts)
        lengths_out = np.concatenate(length_parts)

        # Build CSR-style offsets from per-cell fragment counts
        all_cell_lengths = np.concatenate(cell_length_parts)
        offsets = np.zeros(len(all_cell_lengths) + 1, dtype=np.int64)
        np.cumsum(all_cell_lengths, out=offsets[1:])

        obs_pl = pl.concat(obs_parts, how="diagonal_relaxed")
        obs = _build_obs_df(obs_pl)

        return FragmentResult(
            chromosomes=chromosomes,
            starts=starts_out,
            lengths=lengths_out,
            offsets=offsets,
            chrom_names=chrom_names,
            obs=obs,
        )
