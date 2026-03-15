"""Batch samplers for CellDataset-based ML training.

Implements ``torch.utils.data.Sampler`` subclasses that own all batch
planning logic (which cells go in which batch, shuffle, worker-locality,
balancing).  :class:`~lancell.dataloader.CellDataset` is left as a pure
data-access object; samplers compose on top of it.

Usage::

    dataset = atlas.query().to_cell_dataset(metadata_columns=["cell_type"])

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

    The batch plan is recomputed on each :meth:`set_epoch` call and
    cached; :meth:`__iter__` returns from the cache.

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
        self._batches: list[list[int]] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        """Re-plan batches for a new epoch (new shuffle)."""
        self._batches = self._plan(epoch)

    def _plan(self, epoch: int) -> list[list[int]]:
        if self._n_cells == 0:
            return []

        effective_workers = max(1, self.num_workers)

        # Bin-pack groups across workers (greedy, largest-first)
        unique_gids, counts = np.unique(self.groups_np, return_counts=True)
        sort_idx = np.argsort(-counts)
        unique_gids = unique_gids[sort_idx]
        counts = counts[sort_idx]

        worker_cell_lists: list[list[np.ndarray]] = [[] for _ in range(effective_workers)]
        worker_totals = np.zeros(effective_workers, dtype=np.int64)

        for gid, count in zip(unique_gids, counts, strict=False):
            w = int(np.argmin(worker_totals))
            cell_indices = np.where(self.groups_np == gid)[0].astype(np.int64)
            worker_cell_lists[w].append(cell_indices)
            worker_totals[w] += count

        # Shuffle (if requested) and chunk into batches per worker
        rng_seed = (self.seed + epoch) if self.seed is not None else None
        rng = np.random.default_rng(rng_seed)

        worker_batches: list[list[list[int]]] = []
        for w in range(effective_workers):
            if not worker_cell_lists[w]:
                worker_batches.append([])
                continue

            cells = np.concatenate(worker_cell_lists[w])
            if self.shuffle:
                rng.shuffle(cells)

            batches: list[list[int]] = []
            for start in range(0, len(cells), self.batch_size):
                chunk = cells[start : start + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                batches.append(chunk.tolist())
            worker_batches.append(batches)

        # Interleave across workers (column-major) so PyTorch's round-robin
        # routes batch b from worker w to that worker.
        result: list[list[int]] = []
        max_batches = max((len(wb) for wb in worker_batches), default=0)
        for b in range(max_batches):
            for w in range(effective_workers):
                if b < len(worker_batches[w]):
                    result.append(worker_batches[w][b])
        return result

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)


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
        raw_vals = cells_pl[column].to_list()
        unique_cats = sorted(set(raw_vals))
        cat_to_id = {c: i for i, c in enumerate(unique_cats)}
        balance_values = np.array([cat_to_id[v] for v in raw_vals], dtype=np.int32)
        return cls(balance_values, batch_size=batch_size, **kwargs)

    def set_epoch(self, epoch: int) -> None:
        """Re-plan batches for a new epoch (new shuffle)."""
        self._batches = self._plan(epoch)

    def _plan(self, epoch: int) -> list[list[int]]:
        if self._n_cells == 0:
            return []

        rng = np.random.default_rng((self.seed + epoch) if self.seed is not None else None)
        unique_cats = np.unique(self.balance_values)
        n_cats = len(unique_cats)
        cells_per_cat = max(1, self.batch_size // n_cats)

        # Build per-category index arrays, optionally shuffled
        cat_indices: dict[int, np.ndarray] = {}
        for cat in unique_cats:
            idx = np.where(self.balance_values == cat)[0].astype(np.int64)
            if self.shuffle:
                rng.shuffle(idx)
            cat_indices[int(cat)] = idx

        min_size = min(len(v) for v in cat_indices.values())
        n_batches = min_size // cells_per_cat
        if not self.drop_last and min_size % cells_per_cat != 0:
            n_batches += 1
        if n_batches == 0:
            return []

        positions: dict[int, int] = {int(cat): 0 for cat in unique_cats}
        batches: list[list[int]] = []

        for _ in range(n_batches):
            parts: list[np.ndarray] = []
            for cat in unique_cats:
                c = int(cat)
                arr = cat_indices[c]
                pos = positions[c]
                end = min(pos + cells_per_cat, len(arr))
                if pos < len(arr):
                    parts.append(arr[pos:end])
                positions[c] = end

            batch_indices = np.concatenate(parts)
            if self.shuffle:
                rng.shuffle(batch_indices)
            batches.append(batch_indices.tolist())

        # Distribute batches across workers in column-major interleave
        effective_workers = max(1, self.num_workers)
        if effective_workers <= 1:
            return batches

        worker_batches: list[list[list[int]]] = [[] for _ in range(effective_workers)]
        for i, b in enumerate(batches):
            worker_batches[i % effective_workers].append(b)

        result: list[list[int]] = []
        max_wb = max(len(wb) for wb in worker_batches)
        for bi in range(max_wb):
            for w in range(effective_workers):
                if bi < len(worker_batches[w]):
                    result.append(worker_batches[w][bi])
        return result

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)
