# Versioning

A `RaggedAtlas` is designed to support two workflows that happen concurrently and with different requirements:

- **Ingestion** — parallel writers adding datasets to a shared atlas, where throughput matters more than consistency.
- **Analysis and ML dataloading** — reads that require a stable, reproducible, validated view of the data.

Versioning bridges these two worlds. Writers accumulate data freely into the mutable "tip" of the atlas. When they are ready, they call `snapshot()` to commit a validated, immutable view. Readers and dataloaders then call `checkout()` to pin to a specific snapshot and are guaranteed that the data they see will not change.

---

## The lifecycle at a glance

```
RaggedAtlas.create()   ← one time, by the atlas owner
        │
        ▼
RaggedAtlas.open()     ← writers open the mutable tip
        │
        ├── register_features()
        ├── write zarr arrays + cell rows
        └── add_or_reuse_layout()
        │
        ▼
atlas.optimize()       ← compact tables, assign global_index
        │
        ▼
atlas.snapshot()       ← validate + commit a versioned snapshot
        │
        ▼
RaggedAtlas.checkout(version=N)   ← readers pin to a snapshot
        │
        └── atlas.query()...      ← stable, read-only queries
```

---

## Creating an atlas

`RaggedAtlas.create()` initialises all LanceDB tables and the zarr root group. This is a one-time operation performed by whoever owns the atlas.

```python
import obstore.store
import lancell
from lancell.atlas import RaggedAtlas
from lancell.schema import LancellBaseSchema, SparseZarrPointer
from lancell.schema import DatasetRecord
from my_project.schemas import GeneSchema  # a FeatureBaseSchema subclass

store = obstore.store.S3Store("s3://my-bucket/my-atlas/zarr")

atlas = RaggedAtlas.create(
    db_uri="s3://my-bucket/my-atlas/lancedb",
    cell_table_name="cells",
    cell_schema=MyCellSchema,
    dataset_table_name="datasets",
    dataset_schema=DatasetRecord,
    store=store,
    registry_schemas={
        "gene_expression": GeneSchema,
    },
)
```

This creates:

| Table | Purpose |
|---|---|
| `cells` | One row per cell, with pointer fields into zarr |
| `datasets` | One row per dataset/zarr group |
| `gene_expression_registry` | Global feature registry for gene expression |
| `_feature_layouts` | Feature ordering layouts referenced by datasets |
| `atlas_versions` | Snapshot history |

The zarr root group is created in write mode (`mode="w"`).

---

## Adding data

Any process that wants to add datasets calls `RaggedAtlas.open()`, which connects to the existing LanceDB tables and opens the zarr root in append mode (`mode="a"`).

```python
atlas = RaggedAtlas.open(
    db_uri="s3://my-bucket/my-atlas/lancedb",
    cell_table_name="cells",
    cell_schema=MyCellSchema,
    dataset_table_name="datasets",
    store=store,
    registry_tables={"gene_expression": "gene_expression_registry"},
)
```

A typical ingestion pipeline for a single dataset looks like this:

```python
# 1. Register the features in this dataset
n_new = atlas.register_features("gene_expression", var_df)
print(f"Registered {n_new} new genes")

# 2. Write the zarr arrays (CSR data, indptr, indices, data arrays)
write_dataset_to_zarr(atlas.root, dataset_uid, adata)

# 3. Insert cell rows into the cell table
atlas.cell_table.add(cell_rows)

# 4. Record the feature ordering for this dataset
atlas.add_or_reuse_layout(var_df, dataset_uid, "gene_expression")
```

**Newly added data is not yet queryable.** The atlas is in a partially consistent state after ingestion: features have been registered but have not yet been assigned a `global_index`, and no snapshot has captured this state. Attempting to call `atlas.query()` on an `open()`-ed atlas will raise:

```
RuntimeError: query() is only available on a versioned atlas.
After ingestion, call atlas.snapshot() then
RaggedAtlas.checkout(db_uri, version, schema, store) to pin to a
validated snapshot. For convenience, use RaggedAtlas.checkout_latest(...).
```

---

## Preparing for a snapshot: `optimize()`

Before calling `snapshot()`, you must call `optimize()`. This does three things:

1. **Compacts Lance fragments** — multiple small write batches get merged into larger fragments for efficient reads.
2. **Assigns `global_index`** — features are registered with `global_index = None`. `optimize()` calls `reindex_registry()` which assigns a stable integer index to every unindexed feature, starting from `max(existing_index) + 1`:

    ```python
    # reindex_registry() assigns indices like this:
    # uid="ENSG00000139618"  global_index=0
    # uid="ENSG00000141510"  global_index=1
    # uid="ENSG00000157764"  global_index=2
    # ...
    ```

3. **Propagates indices to `_feature_layouts`** — calls `sync_layouts_global_index()` to fill in the `global_index` column in every feature layout row that references a newly indexed feature.

```python
atlas.optimize()
```

`global_index` is the stable coordinate used at query time to align features across datasets. It is intentionally not assigned during registration — this allows multiple writers to register features independently without needing to coordinate on a shared counter. All the indexing happens in one place, in one writer's call to `optimize()`, eliminating the race condition.

---

## Committing a snapshot: `snapshot()`

`snapshot()` runs `validate()` and, if it passes, records the current Lance version numbers of all tables into the `atlas_versions` table.

```python
version = atlas.snapshot()
print(f"Created snapshot v{version}")
```

Under the hood, a snapshot record looks like this:

```python
AtlasVersionRecord(
    version=0,
    cell_table_name="cells",
    cell_table_version=4,           # Lance version at time of snapshot
    dataset_table_name="datasets",
    dataset_table_version=2,
    registry_table_names='{"gene_expression": "gene_expression_registry"}',
    registry_table_versions='{"gene_expression": 3}',
    feature_layouts_table_version=5,
    total_cells=1_234_567,
)
```

Every field in every LanceDB table is an append-only log of Lance fragments. Storing a version number for each table is sufficient to reconstruct the exact state of all tables as they existed at snapshot time — `checkout()` uses these numbers to call `.checkout(version)` on each table, effectively time-travelling to that state.

Validation checks that must pass before `snapshot()` succeeds:

- All features in every registry have `global_index` assigned (i.e., `optimize()` has been run).
- Every zarr group matches its registered `ZarrGroupSpec` (correct arrays, correct dtypes).
- Every feature layout is internally consistent: no duplicate feature UIDs, all UIDs present in the registry, no missing `global_index` values.

If any check fails, `snapshot()` raises a `ValueError` listing all errors:

```
ValueError: Atlas validation failed — fix errors before snapshotting:
  • Registry 'gene_expression': 142 row(s) have no global_index. Run reindex_registry(table) to fix.
```

---

## Accessing a snapshot: `checkout()`

`checkout()` is a class method that opens a read-only, validated copy of the atlas pinned to a specific version. The zarr root is opened in read-only mode (`mode="r"`), and each LanceDB table is checked out at the exact Lance version recorded in the snapshot.

```python
# Pin to a specific version
atlas_v0 = RaggedAtlas.checkout(
    db_uri="s3://my-bucket/my-atlas/lancedb",
    version=0,
    cell_schema=MyCellSchema,
    store=store,
)

# Or just grab the latest snapshot
atlas_latest = RaggedAtlas.checkout_latest(
    db_uri="s3://my-bucket/my-atlas/lancedb",
    cell_schema=MyCellSchema,
    store=store,
)
```

Once checked out, the atlas is fully queryable:

```python
adata = (
    atlas_latest.query()
    .where("tissue = 'liver'")
    .feature_spaces("gene_expression")
    .to_anndata()
)
```

To see all available snapshots:

```python
RaggedAtlas.list_versions("s3://my-bucket/my-atlas/lancedb")
# shape: (N, 7)
# ┌─────────┬────────────────────┬──────────────────────┬───┐
# │ version ┆ cell_table_version ┆ total_cells          ┆ … │
# │ 0       ┆ 4                  ┆ 1_234_567            ┆ … │
# │ 1       ┆ 9                  ┆ 2_891_034            ┆ … │
# └─────────┴────────────────────┴──────────────────────┴───┘
```

---

## Parallel writes and feature registry fragility

`RaggedAtlas` is designed so that multiple ingestion processes can run in parallel without coordination. LanceDB's `merge_insert` operations are used throughout:

- `register_features()` uses `merge_insert(on="uid").when_not_matched_insert_all()` — a feature UID that already exists in the registry is silently skipped.
- `add_or_reuse_layout()` uses `merge_insert(on=["layout_uid", "feature_uid"]).when_not_matched_insert_all()` — two workers computing the same layout produce identical rows; the second insert is a no-op.

However, **feature registries are the one place where parallel writes can produce fragile state**. The `merge_insert` insert-if-absent semantics prevent strict duplicates (same `uid` appearing twice), but if two workers race on the same feature UID and the LanceDB merge is not atomic, that feature could end up with two rows in the registry. This would cause `optimize()` to assign two different `global_index` values to the same feature, which would make cross-dataset feature queries return inconsistent results.

In practice, the risk is highest when many workers all ingest datasets that share a large common feature set (e.g., a canonical gene panel). The recommended mitigation is to **pre-register features in a single serial step** before launching parallel ingestion:

```python
# Serial: register all features once before parallel ingestion
atlas = RaggedAtlas.open(...)
for dataset in all_datasets:
    atlas.register_features("gene_expression", dataset.var_df)

# Parallel: write zarr arrays and cell rows concurrently
with multiprocessing.Pool() as pool:
    pool.map(ingest_dataset, all_datasets)
```

We plan to add a reconciliation step to `optimize()` that detects and merges duplicate registry rows, making the registry robust to concurrent registration. Until then, treating feature registration as a serial pre-flight step is the safest approach.

---

## Zarr arrays are not versioned

The LanceDB tables that store cell metadata, feature registries, and layouts are fully versioned — every `snapshot()` records exact table versions and `checkout()` restores them precisely. The zarr arrays that store the actual expression data are **not** versioned in the same way.

This is intentional: zarr arrays are append-only in practice. When a new dataset is ingested, new zarr groups are written to new paths under the root. Old groups are never modified.

The versioning story for zarr is maintained implicitly through the **feature layouts and dataset table**. When you check out version N, the `_feature_layouts` table is restored to its state at version N. Any zarr groups added after that snapshot will not have corresponding entries in `_feature_layouts` or the cell table, so the reconstruction layer will never attempt to read them. In this sense, the cell and layout tables act as an index into zarr — the snapshot tells you which groups exist and what their feature ordering is.

### Handling removed features and cells

If a feature is removed from a dataset (e.g., a low-quality gene is dropped during curation), the `global_index` for that feature will not appear in any `_feature_layouts` row for the updated dataset. At query time, the reconstruction layer builds a **remap array** that maps each local zarr position to its global feature index:

```python
# remap[local_position] = global_index
# A value of -1 means "this local feature is not in the query's feature space"
remap = [0, 1, -1, 3, 4, ...]
```

During reconstruction, indices that map to `-1` are masked out and their corresponding values are dropped before building the output matrix. Removed features therefore produce no output columns — they are silently skipped. The same logic applies to intersection queries across datasets with different feature sets: features present in one dataset but not another get `-1` in that dataset's remap and are excluded from the intersection result.

### Future: icechunk

True array-level versioning — where you could snapshot the zarr arrays themselves and roll them back independently of the metadata tables — is not currently supported. We are exploring [icechunk](https://icechunk.io) as a potential solution. Icechunk provides transactional, versioned object storage for zarr arrays with a snapshot model that would align naturally with lancell's `snapshot()`/`checkout()` workflow. This would close the remaining gap where zarr data written after a snapshot is technically visible at the storage level, even though it is unreachable through a checked-out atlas.
