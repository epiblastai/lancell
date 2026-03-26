"""Zarr group read primitives and cell preparation helpers."""

import asyncio

import numpy as np
import polars as pl
from zarr.core.sync import sync

from lancell.batch_array import BatchAsyncArray
from lancell.obs_alignment import PointerFieldInfo


def _prepare_sparse_cells(
    cells_pl: pl.DataFrame,
    pf: PointerFieldInfo,
) -> tuple[pl.DataFrame, list[str]]:
    """Unnest sparse pointer struct, filter empty, return (filtered_df, unique_groups).

    Adds internal columns ``_zg``, ``_start``, ``_end``, ``_zarr_row``.
    """
    col = pf.field_name
    struct_df = cells_pl[col].struct.unnest()
    cells_pl = cells_pl.with_columns(
        struct_df["zarr_group"].alias("_zg"),
        struct_df["start"].alias("_start"),
        struct_df["end"].alias("_end"),
        struct_df["zarr_row"].alias("_zarr_row"),
    )
    cells_pl = cells_pl.filter(pl.col("_zg") != "")
    groups = cells_pl["_zg"].unique().to_list() if not cells_pl.is_empty() else []
    return cells_pl, groups


def _prepare_dense_cells(
    cells_pl: pl.DataFrame,
    pf: PointerFieldInfo,
) -> tuple[pl.DataFrame, list[str]]:
    """Unnest dense pointer struct, filter empty, return (filtered_df, unique_groups).

    Adds internal columns ``_zg``, ``_pos``.
    """
    col = pf.field_name
    struct_df = cells_pl[col].struct.unnest()
    cells_pl = cells_pl.with_columns(
        struct_df["zarr_group"].alias("_zg"),
        struct_df["position"].alias("_pos"),
    )
    cells_pl = cells_pl.filter(pl.col("_zg") != "")
    groups = cells_pl["_zg"].unique().to_list() if not cells_pl.is_empty() else []
    return cells_pl, groups


def _apply_wanted_globals_remap(remap: np.ndarray, wanted_globals: np.ndarray) -> np.ndarray:
    """Map local feature indices to positions in wanted_globals; -1 if absent.

    Parameters
    ----------
    remap:
        Array where remap[local_i] = global_index.
    wanted_globals:
        Sorted int32 array of desired global indices.

    Returns
    -------
    np.ndarray
        int32 array; result[local_i] = position in wanted_globals, or -1.
    """
    positions = np.searchsorted(wanted_globals, remap).astype(np.int32)
    mask = np.isin(remap, wanted_globals)
    positions[~mask] = -1
    return positions


async def _read_sparse_group(
    index_reader: BatchAsyncArray,
    layer_readers: list[BatchAsyncArray],
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Read index array and layer arrays concurrently for one zarr group."""
    coros = [index_reader.read_ranges(starts, ends)]
    coros.extend(r.read_ranges(starts, ends) for r in layer_readers)

    results = await asyncio.gather(*coros)
    return results[0], list(results[1:])


async def _read_dense_group(
    readers: list[BatchAsyncArray],
    starts: np.ndarray,
    ends: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read all dense arrays concurrently for one zarr group."""
    coros = [r.read_ranges(starts, ends) for r in readers]
    return list(await asyncio.gather(*coros))


async def _read_parallel_arrays(
    readers: list[BatchAsyncArray],
    starts: np.ndarray,
    ends: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read N arrays concurrently with shared start/end ranges.

    Returns [(flat_data, lengths), ...] for each reader.
    Unlike :func:`_read_sparse_group`, does not assume a 1-index + N-layers
    structure — all arrays are treated symmetrically.
    """
    return list(await asyncio.gather(*(r.read_ranges(starts, ends) for r in readers)))


def _sync_gather(coroutines: list) -> list:
    """Run coroutines concurrently on a zarr-managed event loop and return results."""

    async def _inner():
        return list(await asyncio.gather(*coroutines))

    return sync(_inner())
