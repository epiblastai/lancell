# PyTorch Data Loading

## Introduction

Lancell provides `CellDataset` and `MultimodalCellDataset` as map-style PyTorch datasets. This distinction matters: a map-style dataset exposes a `__getitem__` interface, so PyTorch's `DataLoader` can dispatch any index to any worker without coordination. There is no shared producer thread, no queue to saturate, and no global lock — each worker fetches its assigned cells independently from zarr.

The alternative, iterable datasets, require a single producer to generate batches and push them into a queue. No matter how many workers are configured, throughput is bounded by that one producer. Lancell avoids this pattern entirely.

Combined with `multiprocessing_context="spawn"`, all zarr I/O runs in parallel across worker processes. Spawn starts clean processes that re-open zarr handles from scratch, which sidesteps the deadlocks that zarr's async I/O and obstore's background threads can cause under the default `fork` context. Lancell's dataset classes are fully picklable so that workers can deserialise them after spawning.

---

## Creating a dataset

The recommended entry point is through `AtlasQuery.to_cell_dataset()` or `to_multimodal_dataset()`. These methods load the cell table and wire up zarr readers; the resulting dataset object is ready to hand to a `DataLoader`.

```python
from lancell.atlas import RaggedAtlas

atlas_r = RaggedAtlas.checkout_latest("/path/to/db", CellSchema, store)

dataset = (
    atlas_r.query()
    .where("split = 'train'")
    .to_cell_dataset(
        feature_space="gene_expression",
        layer="counts",
        metadata_columns=["cell_type", "batch"],
    )
)

print(dataset.n_cells)     # number of cells in the query result
print(dataset.n_features)  # width of the feature space (global index range)
```

`n_features` reflects the full global feature index for the selected feature space, not just the features present in the filtered cells. This ensures that feature indices are stable across training runs and dataset subsets, which matters when a model's input layer is tied to a fixed vocabulary.

### Feature-filtered datasets

When training on a fixed gene panel — a set of marker genes, a pre-selected HVG list, or a model-specific vocabulary — pass the feature UIDs to `.features()` before calling the terminal method. The dataset will only load and return those features, and `n_features` will equal the length of the list.

```python
dataset = (
    atlas_r.query()
    .features(
        ["ENSG00000010610", "ENSG00000156738", "ENSG00000105369"],
        feature_space="gene_expression",
    )
    .to_cell_dataset(feature_space="gene_expression", layer="counts")
)

print(dataset.n_features)  # 3
```

`.features()` accepts the same UID strings stored in the feature registry (Ensembl IDs, gene symbols, or whatever canonical identifier your schema uses). Internally it calls [`resolve_feature_uids_to_global_indices`](feature_layouts.md#resolve_feature_uids_to_global_indices) to translate them into the integer positions used by the zarr reader — no coordinate translation happens at batch time.

---

## Multimodal datasets

`to_multimodal_dataset()` covers atlases that store more than one assay per cell. It returns a `MultimodalCellDataset` whose batches contain one entry per feature space.

Not every cell in a multimodal atlas will have been measured by every assay — a cell from a CITE-seq experiment has both RNA and protein, but a cell from a 10x 3' experiment has RNA only. `MultimodalCellDataset` tracks this with per-modality `present` masks.

```python
dataset = (
    atlas_r.query()
    .to_multimodal_dataset(
        feature_spaces=["gene_expression", "protein_abundance"],
        layers={"gene_expression": "counts", "protein_abundance": "raw"},
        metadata_columns=["cell_type"],
    )
)
```

Batches are `MultimodalBatch` objects with the following fields:

- `modalities: dict[str, SparseBatch | DenseBatch]` — one entry per feature space, keyed by name
- `present: dict[str, np.ndarray]` — boolean array of shape `(n_cells,)` indicating which cells have data for each modality
- `metadata: dict[str, np.ndarray] | None` — requested metadata columns, one array per column

A cell that is absent for a modality still occupies a row in that modality's data array; the row will be zeros. The `present` mask lets downstream code distinguish true zeros from missing measurements.

---

## Samplers

Samplers control the order in which cells are batched. Both samplers implement PyTorch's `BatchSampler` interface, so they slot directly into `DataLoader` as `batch_sampler`. They also carry `num_workers`, which `make_loader` reads to configure the `DataLoader`.

### `CellSampler`

The default sampler. Cells in a lancell atlas are stored in zarr arrays grouped by dataset of origin. Fetching cells that span many groups in a single batch forces the zarr reader to open many array handles and load many chunks, most of which will not be reused. `CellSampler` addresses this by bin-packing groups across workers using a greedy largest-first strategy, then interleaving batches so that consecutive batches assigned to the same worker draw from the same group. This maximises zarr chunk cache locality and reduces redundant reads.

```python
from lancell.sampler import CellSampler

sampler = CellSampler(
    groups_np=dataset.groups_np,   # integer group ID per cell, from the dataset
    batch_size=1024,
    shuffle=True,
    seed=42,
    num_workers=4,
)
```

`groups_np` is exposed directly by `CellDataset` and `MultimodalCellDataset` — you do not need to construct it manually.

For epoch-reproducible shuffling, call `set_epoch` at the start of each epoch. The sampler derives its shuffle from `seed + epoch`, so the sequence is deterministic given the same seed but varies across epochs:

```python
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in loader:
        ...
```

### `BalancedCellSampler`

Draws an equal number of cells from each category per batch. This is useful when rare cell types would otherwise be underrepresented in a uniformly-sampled training run — a T-regulatory cell population at 0.5% of the atlas would appear in fewer than one in two hundred batches by default.

```python
from lancell.sampler import BalancedCellSampler

sampler = BalancedCellSampler.from_column(
    cells_pl=dataset.cells_pl,     # Polars DataFrame of cells, from the dataset
    column="cell_type",
    batch_size=512,
    shuffle=True,
    seed=0,
    num_workers=4,
)
```

`from_column` reads the specified column from the cell table, sorts the unique values, integer-encodes them, and constructs the sampler. Each batch contains `batch_size // n_categories` cells from each category. Epoch length is bounded by the smallest category: once any category is exhausted, the epoch ends and cells from other categories may go unused. This is intentional — the alternative of oversampling rare categories introduces duplicates, which can cause overfitting on small populations.

---

## Building the DataLoader

`make_loader` wraps `torch.utils.data.DataLoader` with sensible defaults for lancell datasets:

```python
from lancell.dataloader import make_loader, sparse_to_dense_collate

loader = make_loader(
    dataset,
    sampler,
    collate_fn=sparse_to_dense_collate,
)
```

`make_loader` sets `num_workers` from `sampler.num_workers`, passes `sampler` as `batch_sampler`, sets `multiprocessing_context="spawn"`, and uses an identity collate function by default (since `__getitems__` already returns an assembled batch object). Pass `collate_fn` to override the collation step. Any additional keyword arguments are forwarded to `DataLoader`.

---

## Collate functions

PyTorch's `DataLoader` calls the collate function on the output of `__getitems__` before yielding a batch to training code. Lancell's `__getitems__` returns a pre-assembled `SparseBatch` or `MultimodalBatch` — the collate function's job is to convert that into tensors.

### `sparse_to_dense_collate`

Scatters CSR sparse data into a dense `float32` tensor. This is the right default for models that expect a dense input matrix.

```python
from lancell.dataloader import sparse_to_dense_collate

loader = make_loader(dataset, sampler, collate_fn=sparse_to_dense_collate)

for batch in loader:
    X = batch["X"]          # torch.Tensor, shape (batch_size, n_features), float32
    cell_type = batch["cell_type"]  # torch.Tensor, present if metadata_columns was set
```

### `sparse_to_csr_collate`

Returns a sparse CSR tensor rather than a dense one. Use this for models that natively accept sparse input and where the data is sparse enough that materialising a dense matrix would be wasteful.

```python
from lancell.dataloader import sparse_to_csr_collate

loader = make_loader(dataset, sampler, collate_fn=sparse_to_csr_collate)

for batch in loader:
    X = batch["X"]  # torch.sparse_csr_tensor, shape (batch_size, n_features)
```

### `multimodal_to_dense_collate`

Converts a `MultimodalBatch` to a nested dictionary of dense tensors plus presence masks. Each modality becomes a dense `float32` tensor; presence masks become boolean tensors.

```python
from lancell.dataloader import multimodal_to_dense_collate

loader = make_loader(multimodal_dataset, sampler, collate_fn=multimodal_to_dense_collate)

for batch in loader:
    rna = batch["gene_expression"]["X"]        # (n_cells, n_rna_features), float32
    protein = batch["protein_abundance"]["X"]  # (n_cells, n_protein_features), float32
    rna_present = batch["present"]["gene_expression"]  # bool tensor (n_cells,)
    cell_type = batch["metadata"]["cell_type"]
```

---

## End-to-end example

```python
import torch
from lancell.atlas import RaggedAtlas
from lancell.sampler import CellSampler
from lancell.dataloader import make_loader, sparse_to_dense_collate

# Open a checked-out atlas
atlas_r = RaggedAtlas.checkout_latest("/path/to/db", CellSchema, store)

# Build the training dataset from a query
dataset = (
    atlas_r.query()
    .where("split = 'train'")
    .to_cell_dataset(
        feature_space="gene_expression",
        layer="counts",
        metadata_columns=["cell_type"],
    )
)

# Construct the sampler
sampler = CellSampler(
    groups_np=dataset.groups_np,
    batch_size=1024,
    shuffle=True,
    seed=0,
    num_workers=4,
)

# Build the DataLoader
loader = make_loader(dataset, sampler, collate_fn=sparse_to_dense_collate)

model = MyModel(n_features=dataset.n_features)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    sampler.set_epoch(epoch)
    for batch in loader:
        X = batch["X"].to("cuda")
        loss = model(X)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## Import reference

```python
from lancell.dataloader import (
    CellDataset,
    MultimodalCellDataset,
    SparseBatch,
    DenseBatch,
    MultimodalBatch,
    sparse_to_dense_collate,
    sparse_to_csr_collate,
    multimodal_to_dense_collate,
    make_loader,
)
from lancell.sampler import CellSampler, BalancedCellSampler
```
