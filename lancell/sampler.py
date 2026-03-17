"""Batch samplers for CellDataset-based ML training.

Implements ``torch.utils.data.Sampler`` subclasses that own all batch
planning logic (which cells go in which batch, shuffle, worker-locality,
balancing).  :class:`~lancell.dataloader.CellDataset` is left as a pure
data-access object; samplers compose on top of it.

Usage::

    dataset = atlas.query().to_cell_dataset("gene_expression", "counts", metadata_columns=["cell_type"])

    # Standard sampler: group-local, worker-aware
    sampler = CellSampler(dataset.groups_np, batch_size=256, shuffle=True,
                          seed=42, num_workers=4)

    # Balanced sampler: equal cells per category per batch
    sampler = BalancedCellSampler.from_column(
        dataset.cells_pl, "cell_type", batch_size=256, num_workers=4
    )

    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)
        loader = make_loader(dataset, sampler)
        for batch in loader:
            X = sparse_to_dense_collate(batch)["X"]
"""

import itertools
import math

import numpy as np
import polars as pl

try:
    from torch.utils.data import Sampler as _TorchSampler
except ImportError:

    class _TorchSampler:  # type: ignore[no-redef]
        """Stub so class definitions succeed when torch is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "torch is required for lancell samplers. Install it with: pip install lancell[ml]"
            )


class CellSampler(_TorchSampler):
    """Batch sampler that groups cells by zarr group for I/O locality.

    Groups are bin-packed across workers so each worker warms a small,
    stable reader cache.  Batches are interleaved column-major so
    PyTorch's round-robin distribution sends consecutive batches from
    the same group to the same worker.

    Bin-packing runs once at construction; :meth:`set_epoch` only records
    the epoch so that :meth:`__iter__` can apply a fresh shuffle without
    any pre-allocation.

    Parameters
    ----------
    groups_np:
        Integer group id for each cell (length = n_cells). Use
        :attr:`~lancell.dataloader.CellDataset.groups_np`.
    batch_size:
        Cells per batch.
    shuffle:
        Whether to shuffle cells each epoch.
    seed:
        Base random seed. ``seed + epoch`` is used per epoch.
        ``None`` means non-reproducible shuffle.
    drop_last:
        Drop the trailing incomplete batch for each worker.
    num_workers:
        Number of DataLoader workers.  0 is treated as 1 for planning.
    """

    def __init__(
        self,
        groups_np: np.ndarray,
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int | None = None,
        drop_last: bool = False,
        num_workers: int = 1,
    ) -> None:
        self.groups_np = groups_np
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.num_workers = num_workers
        self._n_cells = len(groups_np)
        self._epoch = 0

        effective_workers = max(1, self.num_workers)

        if self._n_cells == 0:
            self._worker_cell_arrays: list[np.ndarray] = []
        else:
            # Sort cell indices by group to enable O(n log n) groupby
            # instead of O(n × n_groups) repeated np.where scans.
            sort_idx = np.argsort(groups_np, kind="stable")
            sorted_groups = groups_np[sort_idx]
            unique_gids, counts = np.unique(sorted_groups, return_counts=True)
            group_starts = np.concatenate(([0], np.cumsum(counts)[:-1]))

            # Bin-pack groups across workers (greedy, largest-first), this ensures that
            # each worker only keeps a subset of zarr group readers cached instead of potentially
            # all workers having all zarr groups in cache.
            # Each worker receives one contiguous int64 array of cell indices
            # in natural group order; shuffle is applied per-epoch in __iter__.
            order = np.argsort(-counts)
            worker_cell_lists: list[list[np.ndarray]] = [[] for _ in range(effective_workers)]
            worker_totals = np.zeros(effective_workers, dtype=np.int64)

            for i in order:
                w = int(np.argmin(worker_totals))
                cell_indices = sort_idx[group_starts[i] : group_starts[i] + counts[i]]
                worker_cell_lists[w].append(cell_indices)
                worker_totals[w] += counts[i]

            self._worker_cell_arrays = [
                np.concatenate(cell_lists) if cell_lists else np.array([], dtype=np.int64)
                for cell_lists in worker_cell_lists
            ]

    def set_epoch(self, epoch: int) -> None:
        """Record the epoch so the next :meth:`__iter__` uses a fresh shuffle."""
        self._epoch = epoch

    def __len__(self) -> int:
        if self.drop_last:
            return sum(len(arr) // self.batch_size for arr in self._worker_cell_arrays)
        return sum(math.ceil(len(arr) / self.batch_size) for arr in self._worker_cell_arrays)

    def __iter__(self):
        rng_seed = (self.seed + self._epoch) if self.seed is not None else None
        rng = np.random.default_rng(rng_seed)

        worker_iters = []
        for arr in self._worker_cell_arrays:
            cells = arr.copy() if self.shuffle else arr
            if self.shuffle:
                rng.shuffle(cells)
            worker_iters.append(itertools.batched(cells, self.batch_size))

        # Column-major interleave: yield w0_b0, w1_b0, ..., w0_b1, w1_b1, ...
        # so PyTorch's round-robin dispatch routes consecutive batches from
        # the same worker back to that worker, preserving reader-cache locality.
        for batch_group in itertools.zip_longest(*worker_iters, fillvalue=None):
            for batch in batch_group:
                if batch is not None:
                    if self.drop_last and len(batch) < self.batch_size:
                        continue
                    yield batch


# TODO: BalancedCellSampler doesn't respect the requested batch_size exactly —
# it rounds down to cells_per_cat * n_cats.  The real goal is more equal
# representation across an epoch, not perfectly balanced individual batches.
# Rework to always emit the requested batch_size (e.g. top up with extra
# cells from larger categories).
class BalancedCellSampler(_TorchSampler):
    """Batch sampler that draws equal cells per category each batch.

    Each batch is assembled by drawing ``batch_size // n_cats`` cells from
    each unique category.  The epoch length is bounded by the smallest
    category: ``n_batches = min_cat_size // cells_per_cat`` (plus one
    partial batch if ``not drop_last``).  Cells from larger categories
    beyond ``cells_per_cat * n_batches`` are skipped each epoch.

    The batch plan is recomputed on each :meth:`set_epoch` call and
    cached; :meth:`__iter__` returns from the cache.

    Parameters
    ----------
    balance_values:
        Integer category id for each cell (length = n_cells).
    batch_size:
        Cells per batch.
    shuffle:
        Whether to shuffle cells within each category each epoch.
    seed:
        Base random seed. ``seed + epoch`` is used per epoch.
        ``None`` means non-reproducible shuffle.
    drop_last:
        Drop the last incomplete batch if the smallest category is not
        a multiple of ``cells_per_cat``.
    num_workers:
        Number of DataLoader workers (stored for use by
        :func:`~lancell.dataloader.make_loader`).
    """

    def __init__(
        self,
        balance_values: np.ndarray,
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int | None = None,
        drop_last: bool = False,
        num_workers: int = 1,
    ) -> None:
        self.balance_values = balance_values
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.num_workers = num_workers
        self._n_cells = len(balance_values)
        self._batches: list[list[int]] = []
        self._effective_workers = max(1, self.num_workers)

        self._unique_cats: list[int] = np.unique(balance_values).tolist()
        self._n_cats = len(self._unique_cats)
        self._cells_per_cat = max(1, self.batch_size // self._n_cats)

        # Cache per-category index arrays (computed once; shuffled per-epoch)
        self._cat_indices_base: dict[int, np.ndarray] = {
            cat: np.where(balance_values == cat)[0].astype(np.int64) for cat in self._unique_cats
        }

        self.set_epoch(0)

    @classmethod
    def from_column(
        cls,
        cells_pl: pl.DataFrame,
        column: str,
        batch_size: int = 1024,
        **kwargs,
    ) -> "BalancedCellSampler":
        """Construct from a Polars DataFrame column.

        Sorts unique values, integer-encodes them, and passes the result
        to ``__init__``.

        Parameters
        ----------
        cells_pl:
            DataFrame containing the balance column (e.g.
            ``dataset.cells_pl``).
        column:
            Column name to balance on.
        batch_size:
            Cells per batch.
        **kwargs:
            Forwarded to ``__init__`` (``shuffle``, ``seed``,
            ``drop_last``, ``num_workers``).
        """
        if column not in cells_pl.columns:
            raise ValueError(
                f"Column {column!r} not found in cells_pl; available columns: {cells_pl.columns}"
            )
        series = cells_pl[column]
        unique_sorted = series.unique().sort()
        balance_values = series.replace_strict(
            unique_sorted,
            pl.Series(range(len(unique_sorted)), dtype=pl.Int32),
        ).to_numpy()
        return cls(balance_values, batch_size=batch_size, **kwargs)

    def set_epoch(self, epoch: int) -> None:
        """Re-plan batches for a new epoch (new shuffle)."""
        self._batches = self._plan(epoch)

    def _plan(self, epoch: int) -> list[list[int]]:
        if self._n_cells == 0:
            return []

        rng = np.random.default_rng((self.seed + epoch) if self.seed is not None else None)

        # Copy and optionally shuffle the cached per-category index arrays
        cat_indices: dict[int, np.ndarray] = {}
        for cat, arr in self._cat_indices_base.items():
            idx = arr.copy()
            if self.shuffle:
                rng.shuffle(idx)
            cat_indices[cat] = idx

        # Epoch length is bounded by the smallest category: every category
        # must contribute cells_per_cat cells to each batch, so we can only
        # produce as many batches as the smallest category allows.
        min_size = min(len(v) for v in cat_indices.values())
        n_batches = min_size // self._cells_per_cat
        if not self.drop_last and min_size % self._cells_per_cat != 0:
            n_batches += 1
        if n_batches == 0:
            return []

        # Walk through each category's index array in lock-step, advancing
        # by cells_per_cat each batch.  Cells beyond n_batches * cells_per_cat
        # are skipped for this epoch (they'll get a different random slot next epoch).
        positions: dict[int, int] = {cat: 0 for cat in cat_indices}
        batches: list[list[int]] = []

        for _ in range(n_batches):
            parts: list[np.ndarray] = []
            for cat in self._unique_cats:
                arr = cat_indices[cat]
                pos = positions[cat]
                end = min(pos + self._cells_per_cat, len(arr))
                if pos < len(arr):
                    parts.append(arr[pos:end])
                positions[cat] = end

            # Concatenate the per-category slices, then optionally shuffle
            # within the batch so category ordering is not preserved.
            batch_indices = np.concatenate(parts)
            if self.shuffle:
                rng.shuffle(batch_indices)
            batches.append(batch_indices.tolist())

        # Distribute batches across workers in column-major interleave.
        # Round-robin assignment (batch 0→w0, batch 1→w1, …) followed by
        # column-major read-back ensures PyTorch's round-robin dispatch
        # routes worker-w batches back to worker w.
        if self._effective_workers <= 1:
            return batches

        worker_batches: list[list[list[int]]] = [[] for _ in range(self._effective_workers)]
        for i, b in enumerate(batches):
            worker_batches[i % self._effective_workers].append(b)

        result: list[list[int]] = []
        max_wb = max(len(wb) for wb in worker_batches)
        for bi in range(max_wb):
            for w in range(self._effective_workers):
                if bi < len(worker_batches[w]):
                    result.append(worker_batches[w][bi])
        return result

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)
