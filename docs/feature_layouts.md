# Feature Layouts

The `lancell.feature_layouts` module is the Python API for the `_feature_layouts` LanceDB table and the global feature index system. It provides functions to compute layout identifiers, build and insert layout rows, validate layouts against the registry, assign global indices, and resolve feature UIDs to the dense integer positions used by the reconstruction and training pipeline.

For the conceptual overview of how layouts fit into the atlas data model — including the ER diagram and field-level schema — see [Data Structure](data_structure.md). For the `FeatureLayout` Pydantic model definition, see [Schemas](schemas.md#featurelayout). For how layouts are written during ingestion, see [Array Storage](array_storage.md#the-_feature_layouts-feature-mapping).

This page focuses on function signatures, parameters, error conditions, and usage patterns.

---

## How layouts work

Each unique feature ordering is stored once as a "layout", identified by a content-hash `layout_uid` (SHA-256 of the ordered feature list, truncated to 16 hex chars). Datasets with identical feature orderings share the same `layout_uid`, so row count scales with the number of distinct feature orderings rather than the number of datasets.

Each row in the `_feature_layouts` table maps a `(layout_uid, feature_uid, local_index)` triple to a `global_index`. The `local_index` is the column position in the dataset's zarr array; the `global_index` is the position in the shared global feature space used for scatter/gather at reconstruction and training time.

For the full schema and ER diagram, see [Data Structure](data_structure.md#_feature_layouts-table).

---

## API Reference

### `compute_layout_uid`

Compute a deterministic layout identifier from an ordered list of feature UIDs.

```python
def compute_layout_uid(feature_uids: list[str]) -> str
```

| Name | Type | Description |
|------|------|-------------|
| `feature_uids` | `list[str]` | Ordered list of feature UIDs. The order matters — the same features in a different order produce a different hash. |

**Returns:** `str` — SHA-256 hash of the ordered feature list, truncated to 16 hex characters.

```python
from lancell.feature_layouts import compute_layout_uid

uid = compute_layout_uid(["ENSG00000141510", "ENSG00000012048", "ENSG00000171862"])
print(uid)  # e.g. "a3f8c1d09b2e4f67"
```

---

### `build_feature_layout_df`

Build a Polars DataFrame ready for inserting into the `_feature_layouts` table.

```python
def build_feature_layout_df(
    var_df: pl.DataFrame,
    registry_table: lancedb.table.Table,
) -> tuple[str, pl.DataFrame]
```

| Name | Type | Description |
|------|------|-------------|
| `var_df` | `pl.DataFrame` | One row per local feature, in local feature order. Must contain a `global_feature_uid` column. |
| `registry_table` | `lancedb.table.Table` | The feature registry table. Used to look up `global_index` for each feature UID. |

**Returns:** `tuple[str, pl.DataFrame]` — `(layout_uid, df)` where `df` has columns: `layout_uid`, `feature_uid`, `local_index`, `global_index`.

**Raises:** `ValueError` — if any `global_feature_uid` value in `var_df` is missing from the registry.

!!! note
    `global_index` values in the returned DataFrame can be `None` for features that have not yet been indexed by `reindex_registry`. This is expected when building layouts before the first reindexing pass.

```python
from lancell.feature_layouts import build_feature_layout_df

layout_uid, layout_df = build_feature_layout_df(adata.var, registry_table)
print(layout_uid)       # "a3f8c1d09b2e4f67"
print(layout_df.shape)  # (n_features, 4)
```

---

### `layout_exists`

Check whether a layout with the given `layout_uid` already exists in the `_feature_layouts` table.

```python
def layout_exists(table: lancedb.table.Table, layout_uid: str) -> bool
```

| Name | Type | Description |
|------|------|-------------|
| `table` | `lancedb.table.Table` | The `_feature_layouts` LanceDB table. |
| `layout_uid` | `str` | The layout identifier to check. |

**Returns:** `bool` — `True` if at least one row with this `layout_uid` exists.

```python
from lancell.feature_layouts import layout_exists

if not layout_exists(layouts_table, layout_uid):
    layouts_table.add(layout_df)
```

---

### `read_feature_layout`

Read all rows for a layout, sorted by `local_index`.

```python
def read_feature_layout(
    table: lancedb.table.Table,
    layout_uid: str,
) -> pl.DataFrame
```

| Name | Type | Description |
|------|------|-------------|
| `table` | `lancedb.table.Table` | The `_feature_layouts` LanceDB table. |
| `layout_uid` | `str` | The layout identifier to read. |

**Returns:** `pl.DataFrame` — columns `layout_uid`, `feature_uid`, `local_index`, `global_index`, sorted by `local_index`.

```python
from lancell.feature_layouts import read_feature_layout

layout_df = read_feature_layout(layouts_table, "a3f8c1d09b2e4f67")
print(layout_df.columns)  # ['layout_uid', 'feature_uid', 'local_index', 'global_index']
```

---

### `validate_feature_layout`

Run validation checks on the `_feature_layouts` rows for a single layout. Returns a list of error strings — an empty list means the layout is valid.

```python
def validate_feature_layout(
    layouts_table: lancedb.table.Table,
    layout_uid: str,
    *,
    spec: ZarrGroupSpec,
    group: zarr.Group | None = None,
    expected_feature_count: int | None = None,
    registry_table: lancedb.table.Table | None = None,
) -> list[str]
```

| Name | Type | Description |
|------|------|-------------|
| `layouts_table` | `lancedb.table.Table` | The `_feature_layouts` LanceDB table. |
| `layout_uid` | `str` | The layout identifier to validate. |
| `spec` | `ZarrGroupSpec` | The group spec for this feature space. Used to derive expected feature count from the zarr group when `group` is provided. |
| `group` | `zarr.Group \| None` | Optional zarr group. If provided and `expected_feature_count` is `None`, the feature count is derived from the group's array shape. |
| `expected_feature_count` | `int \| None` | Expected number of features. If provided, the row count is checked against this value. Takes precedence over `group`. |
| `registry_table` | `lancedb.table.Table \| None` | If provided, each `feature_uid` is checked against the registry. Missing UIDs are reported as errors. |

**Returns:** `list[str]` — validation error messages. Empty means valid.

**Validation checks performed:**

1. **Row count** — if `expected_feature_count` or `group` is provided, verifies the number of layout rows matches
2. **Null feature UIDs** — checks for null values in the `feature_uid` column
3. **Duplicate feature UIDs** — checks that all `feature_uid` values are unique
4. **Registry membership** — if `registry_table` is provided, verifies every `feature_uid` exists in the registry

```python
from lancell.feature_layouts import validate_feature_layout

errors = validate_feature_layout(
    layouts_table,
    layout_uid="a3f8c1d09b2e4f67",
    spec=spec,
    expected_feature_count=33538,
    registry_table=registry_table,
)
if errors:
    for e in errors:
        print(f"  - {e}")
```

---

### `reindex_registry`

Assign `global_index` to any features in the registry that do not yet have one.

```python
def reindex_registry(table: lancedb.table.Table) -> int
```

| Name | Type | Description |
|------|------|-------------|
| `table` | `lancedb.table.Table` | The feature registry table (a `FeatureBaseSchema` subclass table). |

**Returns:** `int` — number of features newly indexed. `0` if all features already have a `global_index`.

Only unindexed rows (`global_index IS NULL`) are modified. Each is assigned a unique integer starting from `max(existing_global_index) + 1`, or `0` if the registry is empty. Features that already have a `global_index` are never changed.

!!! note
    `atlas.optimize()` calls `reindex_registry` internally, so you typically do not need to call this function directly. See [Building an Atlas](atlas.md) for the recommended workflow.

```python
# Typically called via atlas.optimize(), but can be used standalone:
from lancell.feature_layouts import reindex_registry

n_new = reindex_registry(registry_table)
print(f"Assigned global_index to {n_new} features")
```

---

### `sync_layouts_global_index`

Propagate updated `global_index` values from the registry into the `_feature_layouts` table.

```python
def sync_layouts_global_index(
    layouts_table: lancedb.table.Table,
    registry_table: lancedb.table.Table,
) -> int
```

| Name | Type | Description |
|------|------|-------------|
| `layouts_table` | `lancedb.table.Table` | The `_feature_layouts` LanceDB table. |
| `registry_table` | `lancedb.table.Table` | The feature registry table. |

**Returns:** `int` — number of rows updated.

After `reindex_registry()` assigns new `global_index` values, the `_feature_layouts` table still holds the old (possibly null) values. This function joins the two tables on `feature_uid` and writes updated `global_index` values back into `_feature_layouts` via `merge_insert`.

!!! note
    `atlas.optimize()` calls both `reindex_registry` and `sync_layouts_global_index` internally, so you typically do not need to call this function directly. See [Building an Atlas](atlas.md) for the recommended workflow.

```python
from lancell.feature_layouts import sync_layouts_global_index

n_updated = sync_layouts_global_index(layouts_table, registry_table)
print(f"Updated {n_updated} layout rows")
```

---

### `resolve_feature_uids_to_global_indices`

Resolve a list of feature UIDs to their sorted global indices. This is the bridge between human-readable feature identifiers (Ensembl IDs, gene symbols, etc.) and the integer positions used by the reconstruction and training pipeline.

```python
def resolve_feature_uids_to_global_indices(
    registry_table: lancedb.table.Table,
    feature_uids: list[str],
) -> np.ndarray
```

| Name | Type | Description |
|------|------|-------------|
| `registry_table` | `lancedb.table.Table` | The feature registry table with `uid` and `global_index` columns. |
| `feature_uids` | `list[str]` | List of feature UIDs to resolve. |

**Returns:** `numpy.ndarray` — sorted `int32` array of global indices.

**Raises:** `ValueError` — if any UID is missing from the registry, or if any matched UID has `global_index = None` (run `atlas.optimize()` first).

For how this connects to feature-filtered training, see [PyTorch Data Loading](dataloader.md#feature-filtered-datasets).

```python
# Use with a feature-filtered query
adata = (
    atlas_r.query()
    .features(
        ["ENSG00000141510", "ENSG00000012048", "ENSG00000171862"],
        feature_space="gene_expression",
    )
    .to_anndata()
)
```

If you already have global indices (e.g. from a prior `resolve_feature_uids_to_global_indices` call), pass the UIDs directly to `.features()` — it handles the registry lookup internally.

---

## Typical workflows

### Ingestion workflow

During ingestion, `add_from_anndata` handles layout creation internally. The underlying sequence is:

1. `atlas.optimize()` — assigns `global_index` to newly registered features (via `reindex_registry` internally)
2. Annotate `var` with `global_feature_uid`
3. `build_feature_layout_df` — build the layout DataFrame
4. Insert into `_feature_layouts` (skipped if `layout_exists` returns `True`)

```python
# You don't call these directly — add_from_anndata does it for you:
from lancell.ingestion import add_from_anndata

atlas.optimize()  # assigns global_index to new features
add_from_anndata(atlas, adata, feature_space="gene_expression",
                 zarr_layer="counts", dataset_record=record)
```

For the full ingestion API, see [Array Storage](array_storage.md).

### Post-ingestion maintenance

After ingesting new datasets, `optimize()` ensures global indices are up to date. Internally it runs `reindex_registry` and `sync_layouts_global_index`:

```python
# atlas.optimize() handles both steps:
atlas.optimize()
atlas.snapshot()
```

For the full atlas lifecycle, see [Building an Atlas](atlas.md).

### Feature-filtered queries

To query a specific set of features by UID:

1. `resolve_feature_uids_to_global_indices` — convert UIDs to global indices
2. `.features()` — pass the indices to the query builder

```python
from lancell.feature_layouts import resolve_feature_uids_to_global_indices

wanted = resolve_feature_uids_to_global_indices(
    atlas_r._registry_tables["gene_expression"],
    feature_uids=["ENSG00000141510", "ENSG00000012048"],
)

adata = (
    atlas_r.query()
    .features(wanted, feature_space="gene_expression")
    .to_anndata()
)
```

For querying details, see [Querying](querying.md). For training with feature filters, see [PyTorch Data Loading](dataloader.md#feature-filtered-datasets).

---

## Imports

```python
from lancell.feature_layouts import (
    compute_layout_uid,
    build_feature_layout_df,
    layout_exists,
    read_feature_layout,
    validate_feature_layout,
    reindex_registry,
    sync_layouts_global_index,
    resolve_feature_uids_to_global_indices,
)
```
