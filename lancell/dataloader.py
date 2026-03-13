"""Fast batch dataloader for ML training from lancell atlases.

Separates planning (once: resolve remaps, create readers) from execution
(per batch: just fetch + remap + return raw arrays). Designed for the
``query -> CellDataset -> SparseBatch -> collate_fn -> GPU`` pipeline.
"""

import asyncio
import threading
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import polars as pl

from lancell.atlas import RaggedAtlas
from lancell.batch_array import BatchAsyncArray
from lancell.group_specs import PointerKind, get_spec
from lancell.reconstruction import _prepare_sparse_cells


@dataclass
class SparseBatch:
    """Minimal sparse batch for ML training.

    Represents a batch of cells as flat CSR-style arrays, avoiding
    the overhead of full AnnData/scipy/var DataFrame construction.

    Attributes
    ----------
    indices:
        int32, flat global feature indices (remapped from local).
    values:
        Native dtype, flat expression values.
    offsets:
        int64, CSR-style indptr (length = n_cells + 1).
    n_features:
        Global feature space width (registry size).
    metadata:
        Optional dict of obs columns as numpy arrays, aligned to cells.
    """

    indices: np.ndarray
    values: np.ndarray
    offsets: np.ndarray
    n_features: int
    metadata: dict[str, np.ndarray] | None = None


# ---------------------------------------------------------------------------
# Async primitives
# ---------------------------------------------------------------------------


async def _take_group_sparse(
    index_reader: BatchAsyncArray,
    layer_reader: BatchAsyncArray,
    remap: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read indices and values for one zarr group concurrently.

    Dispatches two concurrent ``run_in_executor`` calls (indices + values)
    for maximum I/O overlap with GIL released.
    """
    (flat_indices, lengths), (flat_values, _) = await asyncio.gather(
        index_reader.read_ranges(starts, ends),
        layer_reader.read_ranges(starts, ends),
    )
    return remap[flat_indices.astype(np.intp)], flat_values, lengths


async def _take_sparse(
    batch_cell_indices: np.ndarray,
    groups: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    unique_groups: list[str],
    index_readers: dict[str, BatchAsyncArray],
    layer_readers: dict[str, BatchAsyncArray],
    remaps: dict[str, np.ndarray],
    n_features: int,
    metadata_arrays: dict[str, np.ndarray] | None,
) -> SparseBatch:
    """Fetch a sparse batch, dispatching across zarr groups concurrently."""
    batch_groups = groups[batch_cell_indices]
    batch_starts = starts[batch_cell_indices]
    batch_ends = ends[batch_cell_indices]

    # Sort by group for ordered concatenation
    sort_order = np.argsort(batch_groups, kind="stable")
    batch_groups = batch_groups[sort_order]
    batch_starts = batch_starts[sort_order]
    batch_ends = batch_ends[sort_order]

    # Dispatch one task per unique group
    tasks = []
    for gid in np.unique(batch_groups):
        mask = batch_groups == gid
        zg = unique_groups[gid]
        tasks.append(
            _take_group_sparse(
                index_readers[zg],
                layer_readers[zg],
                remaps[zg],
                batch_starts[mask],
                batch_ends[mask],
            )
        )

    results = await asyncio.gather(*tasks)

    # Assemble: concatenate in group order
    # REVIEW: In principle, we have all the lengths, right?
    # Can't we pre-allocate and write into the right spots instead of concat?
    all_indices = []
    all_values = []
    all_lengths = []
    for remapped_indices, values, lengths in results:
        all_indices.append(remapped_indices)
        all_values.append(values)
        all_lengths.append(lengths)

    flat_indices = np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int32)
    flat_values = np.concatenate(all_values) if all_values else np.array([], dtype=np.float32)
    lengths = np.concatenate(all_lengths) if all_lengths else np.array([], dtype=np.int64)

    # Build CSR-style offsets
    offsets = np.zeros(len(batch_cell_indices) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])

    # Metadata: reorder to match sorted-by-group cell order
    metadata = None
    if metadata_arrays:
        sorted_cell_indices = batch_cell_indices[sort_order]
        metadata = {col: arr[sorted_cell_indices] for col, arr in metadata_arrays.items()}

    return SparseBatch(
        indices=flat_indices,
        values=flat_values,
        offsets=offsets,
        n_features=n_features,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# CellDataset
# ---------------------------------------------------------------------------


class CellDataset:
    """Pre-bound view for fast batch iteration over an atlas query.

    Separates planning (done once in ``__init__``) from execution
    (per-batch in ``__iter__``), yielding :class:`SparseBatch` objects
    with minimal overhead.

    Parameters
    ----------
    atlas:
        The atlas to read from.
    cells_pl:
        Polars DataFrame of cell records (from a query).
    feature_space:
        Which feature space to read.
    layer:
        Which layer to read within the feature space.
    batch_size:
        Number of cells per batch.
    shuffle:
        Whether to shuffle cells each epoch.
    seed:
        Random seed for reproducibility. Combined with epoch number.
    drop_last:
        Whether to drop the last incomplete batch.
    metadata_columns:
        Obs column names to include as metadata on each SparseBatch.
    """

    def __init__(
        self,
        atlas: RaggedAtlas,
        cells_pl: pl.DataFrame,
        feature_space: str = "gene_expression",
        layer: str = "counts",
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int | None = None,
        drop_last: bool = False,
        metadata_columns: list[str] | None = None,
    ) -> None:
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._drop_last = drop_last
        self._epoch = 0

        # Resolve feature space
        pf = atlas._pointer_fields[feature_space]
        spec = get_spec(feature_space)

        if spec.pointer_kind is not PointerKind.SPARSE:
            raise NotImplementedError(
                f"CellDataset only supports sparse feature spaces, "
                f"got {spec.pointer_kind.value} for '{feature_space}'"
            )

        if len(spec.required_arrays) != 1:
            raise NotImplementedError(
                f"CellDataset requires exactly 1 index array, "
                f"got {len(spec.required_arrays)} for '{feature_space}'"
            )
        index_array_name = spec.required_arrays[0].array_name

        # Unnest pointers and filter empty cells
        cells_pl, groups = _prepare_sparse_cells(cells_pl, pf)
        groups = sorted(groups)  # Deterministic group ordering

        self._n_cells = cells_pl.height
        if self._n_cells == 0:
            self._unique_groups: list[str] = []
            self._groups_np = np.array([], dtype=np.int32)
            self._starts = np.array([], dtype=np.int64)
            self._ends = np.array([], dtype=np.int64)
            self._index_readers: dict[str, BatchAsyncArray] = {}
            self._layer_readers: dict[str, BatchAsyncArray] = {}
            self._remaps: dict[str, np.ndarray] = {}
            self._n_features = 0
            self._metadata_arrays: dict[str, np.ndarray] | None = None
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._loop_thread.start()
            return

        # Map group strings to integer ids for fast numpy operations
        self._unique_groups = groups
        group_to_id = {g: i for i, g in enumerate(groups)}
        self._groups_np = np.array(
            [group_to_id[v] for v in cells_pl["_zg"].to_list()], dtype=np.int32
        )
        self._starts = cells_pl["_start"].to_numpy().astype(np.int64)
        self._ends = cells_pl["_end"].to_numpy().astype(np.int64)

        # Per-group: load remap (local->global), create async readers
        self._remaps: dict[str, np.ndarray] = {}
        self._index_readers: dict[str, BatchAsyncArray] = {}
        self._layer_readers: dict[str, BatchAsyncArray] = {}

        for zg in groups:
            # REVIEW: Should we save these as memmaps to avoid taking up too much RAM?
            self._remaps[zg] = atlas._get_remap(zg, feature_space)
            self._index_readers[zg] = atlas._get_batch_reader(zg, index_array_name)
            self._layer_readers[zg] = atlas._get_batch_reader(zg, f"layers/{layer}")

        # Global feature count from registry (stable across batches/epochs)
        registry_table = atlas._registry_tables[feature_space]
        self._n_features = registry_table.count_rows()

        # Extract metadata as numpy arrays
        # REVIEW: Ditto about memmaps, either as numpy arrays or as pyarrow
        self._metadata_arrays: dict[str, np.ndarray] | None = None
        if metadata_columns:
            self._metadata_arrays = {}
            for col in metadata_columns:
                if col in cells_pl.columns:
                    self._metadata_arrays[col] = cells_pl[col].to_numpy()

        # Background event loop for async dispatch — works even when the
        # caller is already inside a running loop (marimo, Jupyter, etc.)
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

    @property
    def n_cells(self) -> int:
        return self._n_cells

    @property
    def n_features(self) -> int:
        return self._n_features

    def __len__(self) -> int:
        """Number of batches per epoch."""
        if self._n_cells == 0:
            return 0
        if self._drop_last:
            return self._n_cells // self._batch_size
        return (self._n_cells + self._batch_size - 1) // self._batch_size

    def __iter__(self) -> Iterator[SparseBatch]:
        """Yield SparseBatch objects for one epoch."""
        if self._n_cells == 0:
            return

        # Cell-level permutation for this epoch
        if self._shuffle:
            rng = np.random.default_rng(
                self._seed + self._epoch if self._seed is not None else None
            )
            cell_order = rng.permutation(self._n_cells)
        else:
            cell_order = np.arange(self._n_cells)

        self._epoch += 1

        # Chunk into batches
        n_batches = len(self)
        for i in range(n_batches):
            start = i * self._batch_size
            end = min(start + self._batch_size, self._n_cells)
            if self._drop_last and (end - start) < self._batch_size:
                break

            batch_indices = cell_order[start:end]

            future = asyncio.run_coroutine_threadsafe(
                _take_sparse(
                    batch_indices,
                    self._groups_np,
                    self._starts,
                    self._ends,
                    self._unique_groups,
                    self._index_readers,
                    self._layer_readers,
                    self._remaps,
                    self._n_features,
                    self._metadata_arrays,
                ),
                self._loop,
            )
            batch = future.result()
            yield batch

    def __del__(self):
        if hasattr(self, "_loop") and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5)
            self._loop.close()


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------


def sparse_to_dense_collate(batch: SparseBatch) -> dict:
    """Convert a SparseBatch to a dense float32 tensor via scatter.

    Returns ``{"X": dense_tensor, **metadata_tensors}``.
    """
    import torch

    n_cells = len(batch.offsets) - 1
    X = torch.zeros(n_cells, batch.n_features, dtype=torch.float32)

    lengths = np.diff(batch.offsets)
    row_indices = np.repeat(np.arange(n_cells), lengths)

    X[row_indices, batch.indices] = torch.from_numpy(batch.values.astype(np.float32))

    result: dict = {"X": X}
    if batch.metadata:
        for col, arr in batch.metadata.items():
            if arr.dtype.kind in ("i", "u", "f"):
                result[col] = torch.from_numpy(arr)
            else:
                result[col] = arr
    return result


def sparse_to_csr_collate(batch: SparseBatch) -> dict:
    """Convert a SparseBatch to a sparse CSR tensor.

    Returns ``{"X": sparse_csr_tensor, **metadata_tensors}``.
    """
    import torch

    n_cells = len(batch.offsets) - 1
    X = torch.sparse_csr_tensor(
        crow_indices=torch.from_numpy(batch.offsets),
        col_indices=torch.from_numpy(batch.indices.astype(np.int64)),
        values=torch.from_numpy(batch.values.astype(np.float32)),
        size=(n_cells, batch.n_features),
    )

    result: dict = {"X": X}
    if batch.metadata:
        for col, arr in batch.metadata.items():
            if arr.dtype.kind in ("i", "u", "f"):
                result[col] = torch.from_numpy(arr)
            else:
                result[col] = arr
    return result


# ---------------------------------------------------------------------------
# Torch integration
# ---------------------------------------------------------------------------


class TorchCellDataset:
    """Torch IterableDataset wrapper around CellDataset.

    Use with ``torch.DataLoader(batch_size=None)`` since CellDataset
    already handles batching internally. ``num_workers=0`` for v1 — the
    Rust reader's async I/O + rayon decode already saturate network and CPU.

    Returns a ``torch.utils.data.IterableDataset`` instance via ``__new__``.
    """

    def __new__(cls, cell_dataset: CellDataset):
        from torch.utils.data import IterableDataset

        class _Wrapper(IterableDataset):
            def __init__(self, dataset: CellDataset):
                super().__init__()
                self._dataset = dataset

            def __iter__(self):
                return iter(self._dataset)

            def __len__(self):
                return len(self._dataset)

        return _Wrapper(cell_dataset)
