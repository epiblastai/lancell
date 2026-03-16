# Array Storage (CSR and CSC)

Array data in lancell is stored in zarr, organized by dataset and feature space. For sparse assays (gene expression, chromatin accessibility), two complementary layouts exist:

- **CSR (row-sorted)** — the primary storage format. Every ingest call writes CSR. Appending a new dataset is a pure array append with no read-modify-write on existing data.
- **CSC (column-sorted)** — an optional transpose index. Enables efficient feature-oriented reads (e.g., "give me all cells' values for gene X") by providing direct byte ranges per feature column. Built on-demand via `add_csc()`.

The key design principle: **CSC is not required**. Reconstructors always fall back to CSR and filter the relevant columns when CSC is unavailable. You can add CSC incrementally to existing groups without disrupting reads on groups that do not yet have it.

---

## Ingesting data: `add_from_anndata`

`add_from_anndata` is the primary entry point for writing a dataset into the atlas. It accepts either an in-memory `AnnData` object or a path to an `.h5ad` file.

```python
from lancell.ingestion import add_from_anndata

n = add_from_anndata(
    atlas,
    adata,                          # AnnData object or path to .h5ad file
    feature_space="gene_expression",
    zarr_layer="counts",            # destination layer name within the zarr group
    dataset_record=dataset_record,
)
print(f"ingested {n} cells")
```

### What happens

1. Validates `zarr_layer` against the spec's `allowed_layers` and validates obs columns against the cell schema.
2. Locates the pointer field for this feature space in the cell schema.
3. Pre-allocates zarr arrays with the configured chunk and shard shapes, then streams data in shard-sized batches.
4. For backed `.h5ad` files, reads directly from HDF5 `indptr`, `indices`, and `data` datasets without materializing the full matrix into memory.
5. Writes the `_dataset_vars` feature mapping (one row per feature, recording `local_index` and `global_index`).
6. Inserts cell records with `SparseZarrPointer` fields (`start`, `end`, `zarr_row`) derived from the CSR `indptr` array.

For sparse assays, ingest produces the CSR layout under the zarr group:

```
<zarr_group>/
└── csr/
    ├── indices          # (N_entries,)  uint32 — local feature indices
    └── layers/
        └── counts       # (N_entries,)  dtype  — values
```

For dense assays (e.g., image feature vectors, protein panels):

```
<zarr_group>/
└── layers/
    └── counts           # (N_cells, N_features)  float32
```

### Chunking and sharding

Sharding is important for object-store performance: a shard is a single file, so large shards reduce the number of HTTP requests required to read a dataset. The defaults are:

- Sparse: chunk shape `(40960,)`, shard shape `(41943040,)` — one shard holds 1024 chunks.
- Dense: chunk shape `(max(1, 40960 // n_vars), n_vars)`, shard shape `(max(1, 41943040 // n_vars), n_vars)`.

These can be overridden at ingest time:

```python
n = add_from_anndata(
    atlas, adata,
    feature_space="gene_expression",
    zarr_layer="counts",
    dataset_record=dataset_record,
    chunk_shape=(4096,),       # zarr chunk shape for 1D CSR arrays
    shard_shape=(4194304,),    # outer shard; must be a multiple of chunk_shape
)
```

For sparse feature spaces, `chunk_shape` and `shard_shape` must be 1-element tuples. For dense feature spaces they must be 2-element tuples `(n_cells_per_chunk, n_features)`. Chunk sizes that are multiples of 128 align well with BP-128 bitpacking.

### Integer compression

When the source data has an integer dtype (`int32`, `int64`, `uint32`, `uint64`), `add_from_anndata` automatically applies BP-128 bitpacking with delta encoding on the `indices` array and BP-128 (no delta) on the values layer. This is a lossless codec that typically halves storage for typical single-cell count matrices. Float data is stored uncompressed at the zarr level (outer compression from the zarr store is still applied).

### Backed `.h5ad` files

If you pass an `.h5ad` path (or an `AnnData` opened with `backed="r"`), `add_from_anndata` streams CSR data directly from disk without loading the full matrix into memory. This is the recommended approach for large datasets.

```python
n = add_from_anndata(
    atlas,
    "/path/to/large_dataset.h5ad",  # opened backed="r" automatically
    feature_space="gene_expression",
    zarr_layer="counts",
    dataset_record=dataset_record,
)
```

For dense arrays, AnnData handles backed vs in-memory transparently — `adata.X[start:end]` streams from disk regardless.

---

## The `_dataset_vars` feature mapping

At ingest time, `add_from_anndata` writes one `DatasetVar` row per feature in the dataset into the `_dataset_vars` LanceDB table. Each row records:

| Field | Description |
|---|---|
| `feature_uid` | Stable global feature identifier (from `adata.var["global_feature_uid"]`) |
| `dataset_uid` | The `DatasetRecord.uid` for this dataset |
| `local_index` | 0-based column index in this dataset's zarr array — i.e. the value stored in `csr/indices` |
| `global_index` | The feature's position in the shared global feature space; used for scatter/gather at query time |
| `csc_start` | `null` until `add_csc()` is called |
| `csc_end` | `null` until `add_csc()` is called |

The `local_index → global_index` mapping is what allows the reconstruction layer to correctly align features from different datasets into a single output matrix. Each dataset may have measured a different subset of features, in a different order. The remap array (`remap[local_i] = global_index`) derived from `_dataset_vars` drives the scatter step.

### Prerequisite: `global_index` assignment

Before `add_from_anndata` can write `_dataset_vars`, features must have a non-null `global_index` in the registry. `global_index` is assigned by `atlas.optimize()`, which runs `reindex_registry` as part of its maintenance pass. If any feature in `adata.var` has `global_index = None` in the registry, `add_from_anndata` raises a `ValueError` reporting the offending UIDs.

The typical pattern is to batch-register features across all datasets first, call `optimize()` once to assign indices, and then ingest:

```python
# Register features for each dataset (idempotent — safe to call multiple times)
atlas.register_features("gene_expression", features_dataset_a)
atlas.register_features("gene_expression", features_dataset_b)

# optimize() assigns global_index to all newly registered features
atlas.optimize()

# Now ingest
n_a = add_from_anndata(atlas, adata_a, ...)
n_b = add_from_anndata(atlas, adata_b, ...)
```

---

## Building the CSC index: `add_csc`

CSC is a transposed view of the CSR data, sorted by local feature index. Where CSR stores entries in cell order (all non-zeros for cell 0, then all for cell 1, …), CSC stores them in feature order (all non-zeros for feature 0 across all cells, then feature 1, …).

After CSC is built, `_dataset_vars` gains byte-range pointers (`csc_start`/`csc_end`) for each feature. A feature-filtered query can then read exactly `csc/indices[csc_start:csc_end]` to get the cell row IDs that expressed that feature, without touching any other data.

```python
from lancell.ingestion import add_csc

add_csc(
    atlas,
    zarr_group="pbmc3k",
    feature_space="gene_expression",
    layer_name="counts",
)
```

### What happens

1. Looks up the `dataset_uid` for this `(zarr_group, feature_space)` pair.
2. Queries all cell records for this group; sorts them by `zarr_row` and validates that `zarr_row` is a contiguous 0..N-1 sequence (a prerequisite for the CSC index to be internally consistent).
3. Reads the full `csr/indices` and `csr/layers/<layer>` flat arrays via `BatchArray.read_ranges`.
4. Reconstructs `(cell_row, feature_idx)` pairs for every non-zero entry, then sorts by `feature_idx` (stable sort, so cell order is preserved within each feature column).
5. Writes `csc/indices` (sorted cell row IDs, `uint32`) and `csc/layers/<layer>` (corresponding values) as sharded zarr arrays.
6. Computes `csc_start`/`csc_end` for each feature via `np.searchsorted`, then updates `_dataset_vars` with a `merge_insert`.

After this call, the zarr group contains:

```
<zarr_group>/
├── csr/
│   ├── indices          # row-sorted: entry order matches cell order
│   └── layers/
│       └── counts
└── csc/
    ├── indices          # col-sorted: cell row IDs in feature order
    └── layers/
        └── counts
```

The `chunk_size` and `shard_size` parameters (defaulting to 4096 and 65536 respectively) control the CSC arrays' zarr layout independently of the CSR layout.

### When to run `add_csc`

- After bulk ingestion is complete for a dataset, before snapshotting.
- When queries frequently filter by specific feature UIDs (e.g., marker gene panels, pathway genes, guide sequences).
- When the atlas is large — the I/O savings are proportional to `1 - (n_wanted_features / n_total_features)`. For a dataset with 20,000 genes where you want 50, CSC reads ~0.25% of the data that full-CSR would read.

### When CSC is not needed

- Small atlases where full-CSR reads complete in acceptable time.
- Workloads that always load all features (`.to_anndata()` with union join and no feature filter).
- Prototyping and development — CSC can be added at any point without modifying existing CSR data or cell records.

---

## CSR vs CSC access patterns

| | CSR | CSC |
|---|---|---|
| Storage | Always present | Optional; built by `add_csc()` |
| Sorted by | Cell row (entry order matches cell order) | Feature column (entry order matches feature order) |
| Efficient for | Fetching all features for a set of cells | Fetching all cells' values for a specific feature |
| Reconstruction cost | O(nnz for queried cells) | O(nnz for wanted features) |
| Setup cost | None — written at ingest time | Full CSR read + sort + zarr write |
| Space overhead | Baseline | ~2× storage for the transposed copy |
| When querying N features from F total | Reads all nnz for those cells, then filters columns | Reads only the nnz belonging to those N features |

---

## CSC fallback behavior

`FeatureCSCReconstructor` (invoked automatically by `SparseCSRReconstructor` when `wanted_globals` is provided) checks each group independently:

- Groups **with** CSC: reads `csc/indices[csc_start:csc_end]` and the corresponding layer slice for each wanted feature. The result is assembled as a COO matrix and converted to CSR.
- Groups **without** CSC: reads the full `csr/indices` slice for each cell, remaps local indices to global positions, and filters to the wanted columns. Equivalent to a standard CSR read followed by column selection.

Both paths produce identical output. The fallback is not a degraded mode — it is the full-fidelity CSR reconstruction path. You can run `add_csc()` on your largest or most-queried groups while leaving smaller groups CSR-only.

```python
# Only add CSC to the two largest groups
add_csc(atlas, zarr_group="large_dataset_1", feature_space="gene_expression", layer_name="counts")
add_csc(atlas, zarr_group="large_dataset_2", feature_space="gene_expression", layer_name="counts")
# small_dataset_3 stays CSR-only — reconstructors fall back automatically
```

`GroupReader.has_csc` is the property that drives this decision at read time. It returns `True` only when all `csc_start`/`csc_end` values in `_dataset_vars` for the group are non-null — meaning `add_csc()` completed successfully for that group.

---

## Typical workflow

```python
from lancell.ingestion import add_from_anndata, add_csc

# 1. Register features
atlas.register_features("gene_expression", features)

# 2. Assign global_index and compact — optimize() handles reindex internally
atlas.optimize()

# 3. Ingest
n = add_from_anndata(
    atlas, adata,
    feature_space="gene_expression",
    zarr_layer="counts",
    dataset_record=record,
)

# 4. Build CSC for feature-filtered queries (optional but recommended for large groups)
add_csc(
    atlas,
    zarr_group=record.zarr_group,
    feature_space="gene_expression",
    layer_name="counts",
)

# 5. Compact and snapshot
atlas.optimize()
v = atlas.snapshot()
```

---

## Imports

```python
from lancell.ingestion import add_from_anndata, add_csc
```
