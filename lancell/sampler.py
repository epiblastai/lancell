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

    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)
        loader = make_loader(dataset, sampler)
        for batch in loader:
            X = sparse_to_dense_collate(batch)["X"]
"""

import itertools
import math

import numpy as np

try:
    from torch.utils.data import Sampler as _TorchSampler
except ImportError:

    class _TorchSampler:  # type: ignore[no-redef]
        """Stub so class definitions succeed when torch is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "torch is required for lancell samplers. Install it with: pip install lancell[ml]"
            )


# TODO: Bin-packing by zarr group degenerates when most cells live in one
# large group (e.g. a single group with ~2000 shards): every worker gets a
# random slice with no shard locality.  We should partition by (group, shard)
# instead.  Shard ID per cell is cheap to compute at init time from the
# pointer `start` value and the zarr array's shard shape:
#   shard_id = start // (subchunks_per_shard * subchunk_size)
# Use a composite key (group_id * n_shards + shard_id) as the bin-packing
# unit so the existing algorithm applies unchanged with finer granularity.
# The shard shape is available from the zarr array metadata via GroupReader.
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
