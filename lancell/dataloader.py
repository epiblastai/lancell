"""Fast batch dataloader for ML training from lancell atlases.

Separates planning (once: resolve remaps, plan batches) from execution
(per batch: just fetch + remap + return raw arrays). Designed for the
``query -> CellDataset -> SparseBatch -> collate_fn -> GPU`` pipeline.

CellDataset is a map-style dataset (``__getitem__`` + ``__len__``) so
PyTorch's DataLoader can distribute indices to workers automatically.
Reader initialization is deferred to the worker process, making the
dataset safely picklable for spawn-based multiprocessing.

Usage::

    dataset = CellDataset(atlas, cells_pl, ..., num_workers=4)
    for epoch in range(n_epochs):
        dataset.set_epoch(epoch)
        loader = make_loader(dataset)
        for batch in loader:
            ...
"""

import asyncio
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
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


@dataclass
class BatchDescriptor:
    """Descriptor for a single pre-planned batch.

    Attributes
    ----------
    cell_indices:
        int64, indices into the global ``_starts``/``_ends``/``_groups_np``
        arrays stored on ``CellDataset``.
    """

    cell_indices: np.ndarray


# ---------------------------------------------------------------------------
# Batch planning
# ---------------------------------------------------------------------------


def _plan_batches(
    epoch: int,
    num_workers: int,
    batch_size: int,
    drop_last: bool,
    shuffle: bool,
    seed: int | None,
    groups_np: np.ndarray,
    n_cells: int,
) -> list[BatchDescriptor]:
    """Plan all batches for one epoch, partitioned across workers.

    Groups are bin-packed across workers by cell count so each worker
    warms a small, stable reader cache. Batches are then interleaved
    column-major so PyTorch's round-robin distribution sends consecutive
    batches from the same group to the same worker.

    Parameters
    ----------
    epoch:
        Epoch index, mixed into the RNG seed.
    num_workers:
        Number of DataLoader workers. 0 is treated as 1.
    batch_size:
        Cells per batch.
    drop_last:
        Drop the trailing incomplete batch for each worker.
    shuffle:
        Shuffle cell order within each worker's partition.
    seed:
        Base random seed. ``seed + epoch`` is used per epoch.
        ``None`` means non-reproducible shuffle.
    groups_np:
        Integer group id for each cell (length = n_cells).
    n_cells:
        Total number of cells.

    Returns
    -------
    list[BatchDescriptor]
        Flat list of batch descriptors in interleaved worker order.
    """
    if n_cells == 0:
        return []

    effective_workers = max(1, num_workers)

    # Step 1: Bin-pack groups across workers (greedy, largest-first)
    unique_gids, counts = np.unique(groups_np, return_counts=True)
    sort_idx = np.argsort(-counts)  # descending
    unique_gids = unique_gids[sort_idx]
    counts = counts[sort_idx]

    worker_cell_lists: list[list[np.ndarray]] = [[] for _ in range(effective_workers)]
    worker_totals = np.zeros(effective_workers, dtype=np.int64)

    for gid, count in zip(unique_gids, counts, strict=False):
        w = int(np.argmin(worker_totals))
        cell_indices = np.where(groups_np == gid)[0].astype(np.int64)
        worker_cell_lists[w].append(cell_indices)
        worker_totals[w] += count

    # Step 2 & 3: Shuffle (if requested) and chunk into batches per worker
    rng_seed = (seed + epoch) if seed is not None else None
    rng = np.random.default_rng(rng_seed)

    worker_batches: list[list[BatchDescriptor]] = []
    for w in range(effective_workers):
        if not worker_cell_lists[w]:
            worker_batches.append([])
            continue

        cells = np.concatenate(worker_cell_lists[w])
        if shuffle:
            rng.shuffle(cells)

        batches: list[BatchDescriptor] = []
        for start in range(0, len(cells), batch_size):
            chunk = cells[start : start + batch_size]
            if drop_last and len(chunk) < batch_size:
                continue
            batches.append(BatchDescriptor(cell_indices=chunk))
        worker_batches.append(batches)

    # Step 4: Interleave across workers (column-major)
    # Batch b from worker w is at position b * effective_workers + w, so
    # PyTorch's round-robin assignment routes it to worker w.
    result: list[BatchDescriptor] = []
    max_batches = max((len(wb) for wb in worker_batches), default=0)
    for b in range(max_batches):
        for w in range(effective_workers):
            if b < len(worker_batches[w]):
                result.append(worker_batches[w][b])

    return result


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
    remapped = remap[flat_indices.astype(np.intp)]
    mask = remapped >= 0
    if not mask.all():
        cell_ids = np.repeat(np.arange(len(lengths)), lengths)
        remapped = remapped[mask]
        flat_values = flat_values[mask]
        lengths = np.bincount(cell_ids[mask], minlength=len(lengths)).astype(np.int64)
    return remapped, flat_values, lengths


async def _take_sparse(
    batch_cell_indices: np.ndarray,
    groups: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    unique_groups: list[str],
    local_readers: dict[str, tuple[BatchAsyncArray, BatchAsyncArray]],
    remaps: dict[str, np.ndarray],
    n_features: int,
    metadata_arrays: dict[str, np.ndarray] | None,
    zarr_root,
    index_array_name: str,
    layer: str,
) -> SparseBatch:
    """Fetch a sparse batch, dispatching across zarr groups concurrently.

    Readers are created lazily on first access and cached in
    ``local_readers`` for the lifetime of the worker process.
    """
    batch_groups = groups[batch_cell_indices]
    batch_starts = starts[batch_cell_indices]
    batch_ends = ends[batch_cell_indices]

    # Sort by group for ordered concatenation
    sort_order = np.argsort(batch_groups, kind="stable")
    batch_groups = batch_groups[sort_order]
    batch_starts = batch_starts[sort_order]
    batch_ends = batch_ends[sort_order]

    # Dispatch one task per unique group; create readers lazily
    tasks = []
    for gid in np.unique(batch_groups):
        mask = batch_groups == gid
        zg = unique_groups[gid]
        if zg not in local_readers:
            local_readers[zg] = (
                BatchAsyncArray.from_array(zarr_root[f"{zg}/{index_array_name}"]),
                BatchAsyncArray.from_array(zarr_root[f"{zg}/layers/{layer}"]),
            )
        index_reader, layer_reader = local_readers[zg]
        tasks.append(
            _take_group_sparse(
                index_reader,
                layer_reader,
                remaps[zg],
                batch_starts[mask],
                batch_ends[mask],
            )
        )

    results = await asyncio.gather(*tasks)

    # Assemble: concatenate in group order
    # REVIEW: We know lengths upfront, so we could pre-allocate and write into the right spots instead of concatenating?
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
    """Map-style dataset for fast batch access over an atlas query.

    Separates planning (done once in ``__init__`` and ``set_epoch``) from
    execution (per-batch in ``__getitem__``), yielding :class:`SparseBatch`
    objects with minimal overhead.

    Groups are bin-packed across workers so each worker's reader cache
    stays warm. Use :class:`GroupLocalSampler` with PyTorch's DataLoader
    to preserve this locality.

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
    num_workers:
        Number of DataLoader workers this dataset will be used with.
        Used to partition groups across workers during batch planning.
    """

    def __init__(
        self,
        atlas: "RaggedAtlas",
        cells_pl: pl.DataFrame,
        feature_space: str = "gene_expression",
        layer: str = "counts",
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int | None = None,
        drop_last: bool = False,
        metadata_columns: list[str] | None = None,
        wanted_globals: np.ndarray | None = None,
        num_workers: int = 1,
    ) -> None:
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._drop_last = drop_last
        self._num_workers = num_workers

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
        self._index_array_name = spec.required_arrays[0].array_name
        self._layer = layer

        # Store the obstore ObjectStore (picklable via __getnewargs_ex__)
        # Workers reconstruct the zarr root lazily from this store.
        self._store = atlas._store

        # Unnest pointers and filter empty cells
        cells_pl, groups = _prepare_sparse_cells(cells_pl, pf)
        groups = sorted(groups)  # Deterministic group ordering

        self._n_cells = cells_pl.height
        if self._n_cells == 0:
            self._unique_groups: list[str] = []
            self._groups_np = np.array([], dtype=np.int32)
            self._starts = np.array([], dtype=np.int64)
            self._ends = np.array([], dtype=np.int64)
            self._remaps: dict[str, np.ndarray] = {}
            self._n_features = len(wanted_globals) if wanted_globals is not None else 0
            self._metadata_arrays: dict[str, np.ndarray] | None = None
            self._batches: list[BatchDescriptor] = []
            self._local_readers: dict | None = None
            self._loop: asyncio.AbstractEventLoop | None = None
            self._loop_thread: threading.Thread | None = None
            self._zarr_root = None
            self._iter_epoch = 0
            return

        # Map group strings to integer ids for fast numpy operations
        self._unique_groups = groups
        group_to_id = {g: i for i, g in enumerate(groups)}
        self._groups_np = np.array(
            [group_to_id[v] for v in cells_pl["_zg"].to_list()], dtype=np.int32
        )
        self._starts = cells_pl["_start"].to_numpy().astype(np.int64)
        self._ends = cells_pl["_end"].to_numpy().astype(np.int64)

        # Per-group: load remap (local->global)
        self._remaps = {}
        for zg in groups:
            raw_remap = atlas._get_remap(zg, feature_space)
            if wanted_globals is not None:
                positions = np.searchsorted(wanted_globals, raw_remap).astype(np.int32)
                mask = np.isin(raw_remap, wanted_globals)
                positions[~mask] = -1
                self._remaps[zg] = positions
            else:
                self._remaps[zg] = raw_remap

        # Global feature count from registry (stable across batches/epochs)
        if wanted_globals is not None:
            self._n_features = len(wanted_globals)
        else:
            registry_table = atlas._registry_tables[feature_space]
            self._n_features = registry_table.count_rows()

        # Extract metadata as numpy arrays
        self._metadata_arrays = None
        if metadata_columns:
            self._metadata_arrays = {}
            for col in metadata_columns:
                if col in cells_pl.columns:
                    self._metadata_arrays[col] = cells_pl[col].to_numpy()

        # Plan batches for epoch 0
        self._batches = _plan_batches(
            epoch=0,
            num_workers=self._num_workers,
            batch_size=self._batch_size,
            drop_last=self._drop_last,
            shuffle=self._shuffle,
            seed=self._seed,
            groups_np=self._groups_np,
            n_cells=self._n_cells,
        )

        # Worker-local state — initialized lazily in _ensure_initialized()
        self._local_readers = None
        self._loop = None
        self._loop_thread = None
        self._zarr_root = None
        # Epoch counter used by __iter__ for simple for-loop usage
        self._iter_epoch = 0

    @property
    def n_cells(self) -> int:
        return self._n_cells

    @property
    def n_features(self) -> int:
        return self._n_features

    def set_epoch(self, epoch: int) -> None:
        """Re-plan batches for a new epoch (new shuffle).

        Must be called before creating a new DataLoader iterator each epoch.
        Incompatible with ``persistent_workers=True``.

        Parameters
        ----------
        epoch:
            Epoch index, mixed into the RNG seed for deterministic shuffles.
        """
        self._batches = _plan_batches(
            epoch=epoch,
            num_workers=self._num_workers,
            batch_size=self._batch_size,
            drop_last=self._drop_last,
            shuffle=self._shuffle,
            seed=self._seed,
            groups_np=self._groups_np,
            n_cells=self._n_cells,
        )

    def __len__(self) -> int:
        """Number of batches in the current epoch plan."""
        return len(self._batches)

    def __iter__(self):
        """Iterate over all batches for one epoch, advancing the epoch counter.

        Convenience for simple ``for batch in dataset:`` usage. Calls
        :meth:`set_epoch` with an internal counter so successive iterations
        produce differently-shuffled epochs.

        For multi-worker DataLoader training, prefer :class:`GroupLocalSampler`
        and explicit :meth:`set_epoch` calls instead.
        """
        self.set_epoch(self._iter_epoch)
        self._iter_epoch += 1
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> SparseBatch:
        """Fetch batch ``idx`` as a :class:`SparseBatch`."""
        self._ensure_initialized()
        desc = self._batches[idx]
        future = asyncio.run_coroutine_threadsafe(
            _take_sparse(
                desc.cell_indices,
                self._groups_np,
                self._starts,
                self._ends,
                self._unique_groups,
                self._local_readers,
                self._remaps,
                self._n_features,
                self._metadata_arrays,
                self._zarr_root,
                self._index_array_name,
                self._layer,
            ),
            self._loop,
        )
        return future.result()

    def _ensure_initialized(self) -> None:
        """Start the background event loop and reader cache if not yet done.

        Safe to call multiple times; subsequent calls are no-ops.
        Called automatically on the first ``__getitem__`` in each process,
        including spawned worker processes.
        """
        if self._local_readers is not None:
            return
        import zarr

        self._zarr_root = zarr.open_group(zarr.storage.ObjectStore(self._store), mode="r")
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()
        self._local_readers = {}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Drop worker-local state so the dataset is safely picklable for spawn.
        # Workers call _ensure_initialized() on their first __getitem__.
        state["_local_readers"] = None
        state["_loop"] = None
        state["_loop_thread"] = None
        state["_zarr_root"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def __del__(self) -> None:
        if hasattr(self, "_loop") and self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
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


class GroupLocalSampler:
    """Sequential sampler for use with :class:`CellDataset` and PyTorch.

    Emits indices ``0..N-1`` in order. Since :func:`_plan_batches`
    already interleaves batches column-major across workers, sequential
    emission causes PyTorch's round-robin distribution to route each
    batch to the worker that owns its groups.

    Parameters
    ----------
    dataset:
        A :class:`CellDataset` instance.
    """

    def __init__(self, dataset: CellDataset) -> None:
        if not hasattr(dataset, "_num_workers"):
            raise TypeError("dataset must be a CellDataset")
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self):
        return iter(range(len(self._dataset)))


class TorchCellDataset:
    """Picklable map-style dataset wrapper for use with torch DataLoader.

    PyTorch's DataLoader only checks ``isinstance(dataset, IterableDataset)``
    to select the iteration strategy; any object with ``__getitem__`` and
    ``__len__`` works as a map-style dataset. Being a plain module-level class
    (not a local class) makes this safely picklable for
    ``multiprocessing_context="spawn"``.

    Use with :func:`make_loader` (or manually with ``batch_size=None``,
    ``sampler=GroupLocalSampler(dataset)``, ``multiprocessing_context="spawn"``,
    ``persistent_workers=False``).
    """

    def __init__(self, dataset: CellDataset) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> SparseBatch:
        return self._dataset[idx]


def make_loader(dataset: CellDataset, **kwargs):
    """Create a DataLoader with the right defaults for CellDataset.

    Sets ``batch_size=None``, ``sampler=GroupLocalSampler``,
    ``num_workers=dataset._num_workers``, ``collate_fn=lambda x: x``,
    ``multiprocessing_context="spawn"``, and ``persistent_workers=False``.
    Any of these can be overridden via ``kwargs``.

    Parameters
    ----------
    dataset:
        A :class:`CellDataset` instance.
    **kwargs:
        Forwarded to ``torch.utils.data.DataLoader``, overriding defaults.

    Returns
    -------
    torch.utils.data.DataLoader
    """
    from torch.utils.data import DataLoader

    defaults = dict(
        batch_size=None,
        sampler=GroupLocalSampler(dataset),
        num_workers=dataset._num_workers,
        collate_fn=None,
        multiprocessing_context="spawn",
        persistent_workers=False,
    )
    defaults.update(kwargs)
    # Recompute multiprocessing_context after kwargs merge: spawn is invalid
    # when num_workers=0 and the caller didn't explicitly override it.
    if defaults["num_workers"] == 0 and "multiprocessing_context" not in kwargs:
        defaults["multiprocessing_context"] = None
    torch_dataset = TorchCellDataset(dataset)
    return DataLoader(torch_dataset, **defaults)
