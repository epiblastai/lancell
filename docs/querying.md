# Querying the Atlas

## Introduction

Once you have a checked-out atlas, `atlas.query()` returns an `AtlasQuery` — a lazy, fluent query builder that lets you express complex data retrieval in a single readable chain. Methods like `.where()`, `.feature_join()`, and `.layers()` accumulate parameters without touching the database; execution is deferred until a terminal method (`.to_anndata()`, `.count()`, etc.) is called.

```python
from lancell.atlas import RaggedAtlas

atlas_r = RaggedAtlas.checkout_latest("/path/to/db", CellSchema, store)
q = atlas_r.query()  # returns AtlasQuery
```

`AtlasQuery` is returned by `atlas.query()` — it is not imported directly.

### Checked-out atlases only

`query()` is only available on a checked-out atlas, opened via `checkout()` or `checkout_latest()`. Calling it on a writable atlas raises `RuntimeError`. This constraint exists by design: queries always execute against a stable, versioned snapshot of the cell table. A writable atlas may be in the middle of ingesting new cells, so the cell table is not guaranteed to be consistent.

```python
# Correct: open a read-only snapshot first
atlas_r = RaggedAtlas.checkout_latest("/path/to/db", CellSchema, store)
q = atlas_r.query()

# Wrong: writable atlas raises RuntimeError
atlas_w = RaggedAtlas.open("/path/to/db", CellSchema, store)
q = atlas_w.query()  # RuntimeError
```

---

## Filtering cells

### `.where(condition: str)`

Filter cells with a SQL predicate string using LanceDB syntax. The predicate is evaluated against the flat cell table, so any column in your schema is available.

```python
# String equality
atlas_r.query().where("cell_type = 'CD4 T cells'").to_anndata()

# Numeric filter
atlas_r.query().where("n_genes > 500").to_anndata()

# Compound predicate
atlas_r.query().where("cell_type = 'B cells' AND dataset = 'pbmc3k'").to_anndata()
```

Predicates are forwarded directly to LanceDB, so any expression that LanceDB's SQL dialect supports — `IN`, `IS NOT NULL`, `BETWEEN`, nested `AND`/`OR` — works here.

### `.limit(n: int)`

Cap the number of cells returned. Useful for exploratory work or when you want a quick representative sample without loading the full result set.

```python
atlas_r.query().limit(100).to_anndata()
```

`limit` applies after filtering, so `.where(...).limit(100)` returns the first 100 cells matching the predicate rather than 100 random cells from the full atlas.

### `.balanced_limit(n: int, column: str)`

Cap the number of cells returned, drawing equally from each unique value of `column`. The result contains at most `n` cells split evenly across each unique value of `column` that passes any `.where()` filter.

```python
# Return at most 1000 cells, ~equal numbers per cell_type
atlas_r.query().balanced_limit(1000, "cell_type").to_anndata()
```

This is useful for quickly building balanced evaluation sets without manually querying each category. Cannot be combined with `.limit()` — using both on the same query raises a `ValueError`.

### `.search(...)`

Run a vector similarity search or full-text search, forwarded directly to LanceDB's `Table.search()`. Pair with `.limit()` to control how many nearest neighbors are retrieved.

```python
# Retrieve the 50 cells most similar to a query embedding
atlas_r.query().search(embedding_vector, vector_column_name="embedding").limit(50).to_anndata()
```

Vector search and scalar filtering can be combined: `.where()` applies as a post-filter on the ANN results.

---

## Controlling feature reconstruction

By default, every feature space with a pointer in the cell schema is reconstructed and included in the output. For multimodal atlases or atlases with many layers, the default can pull far more data than you need. The methods below let you scope reconstruction precisely.

### `.feature_spaces(*spaces: str)`

Restrict reconstruction to a named subset of feature spaces. Any space not listed is skipped entirely — no array I/O is performed for it.

```python
# Only reconstruct gene expression; skip protein in a multimodal atlas
atlas_r.query().feature_spaces("gene_expression").to_anndata()
```

### `.layers(feature_space: str, names: list[str])`

Override which layers are loaded for a given feature space. By default, reconstruction loads the layers defined as defaults in the schema (usually raw counts). Use this method when you want a different representation.

```python
# Load log_normalized instead of counts for gene expression
atlas_r.query().layers("gene_expression", ["log_normalized"]).to_anndata()
```

Multiple calls to `.layers()` for different feature spaces are cumulative and independent.

### `.feature_join(join: Literal["union", "intersection"])`

Control how the reconstruction layer handles cells from datasets with different feature panels. This is the core of what makes `RaggedAtlas` practical for heterogeneous collections.

- `"union"` (default) — the output matrix includes every feature from any dataset. Cells that were not profiled for a given feature receive zero in that column.
- `"intersection"` — the output matrix includes only features measured in every dataset that contributes cells to the result.

```python
# Union (default): result contains all genes from both PBMC panels
atlas_r.query().to_anndata()  # n_vars = 2395

# Intersection: only the genes shared by both panels
atlas_r.query().feature_join("intersection").to_anndata()  # n_vars = 208
```

Union is the right default for most exploratory queries because no information is discarded. Intersection is useful when downstream analysis requires a consistent feature space across all cells, such as running a single PCA across heterogeneous datasets.

### `.features(uids: list[str], feature_space: str)`

Restrict output to a specific list of features, identified by their registry UIDs. When `.features()` is set, it overrides `feature_join`: the output matrix contains exactly those features, filled with zeros for cells whose dataset did not measure them.

```python
# Load only a handful of marker genes
atlas_r.query().features(["CD3D", "CD19", "MS4A1"], "gene_expression").to_anndata()
```

This is the most targeted way to load data when you only care about a known gene or protein panel.

---

## Terminal methods

Terminal methods execute the query and return data. Calling a terminal method is what triggers LanceDB lookups and zarr reads.

### Counting

**`.count(group_by=None)`** performs a cheap cell count without loading any array data. Because it only touches the cell table, it is fast even for very large atlases.

```python
atlas_r.query().count()  # → int: total cells

atlas_r.query().count(group_by="cell_type")  # → pl.DataFrame with value_counts

atlas_r.query().count(group_by=["cell_type", "dataset_uid"])  # multi-column grouping
```

When `group_by` is provided, the result is a Polars DataFrame with one row per group and a `count` column.

### Metadata only

**`.to_polars()`** returns the cell metadata as a Polars DataFrame without performing any array reconstruction. Use this when you only need the cell table — for inspecting distributions, debugging a filter predicate, or feeding cell IDs into a separate pipeline.

```python
meta = atlas_r.query().where("cell_type IS NOT NULL").to_polars()
```

**`.select(columns: list[str])`** restricts which metadata columns appear in the output of `.to_polars()`. Pointer columns (the per-dataset zarr pointers stored in the cell table) are always fetched internally and stripped from the output regardless of what you pass to `.select()`.

```python
atlas_r.query().select(["cell_type", "n_genes"]).to_polars()
```

### AnnData and MuData

**`.to_anndata()`** reconstructs a single `AnnData` object. It uses the first sparse feature space in the schema, falling back to the first dense space if no sparse space is present. For unimodal atlases this is the natural terminal; for multimodal atlases, consider `.to_mudata()` instead.

```python
adata = atlas_r.query().where("cell_type = 'NK cells'").to_anndata()
```

**`.to_mudata()`** reconstructs one `AnnData` per feature space and wraps them in a `MuData` object. Each modality is keyed by its feature space name.

```python
mdata = atlas_r.query().to_mudata()
mdata["gene_expression"]    # AnnData for RNA
mdata["protein_abundance"]  # AnnData for protein
```

**`.to_batches(batch_size: int = 1024)`** returns a streaming iterator of `AnnData` objects. Each batch contains at most `batch_size` cells. Use this for large queries that would exhaust memory if materialised all at once.

```python
for batch in atlas_r.query().where("tissue = 'lung'").to_batches(batch_size=2048):
    process(batch)  # each batch is a small AnnData
```

The iterator respects all other query parameters — filters, feature spaces, layers, and feature join mode apply to every batch identically.

### ML training datasets

**`.to_cell_dataset(...)`** returns a `CellDataset` for PyTorch training. See the PyTorch Data Loading page for full details.

**`.to_multimodal_dataset(...)`** returns a `MultimodalCellDataset` covering all selected feature spaces. See the PyTorch Data Loading page.

---

## Chaining example

Every builder method returns `self`, so you can compose an entire query in a single expression. This example loads log-normalized gene expression for bone marrow cells, intersecting the feature sets across all contributing datasets, capped at 5000 cells:

```python
adata = (
    atlas_r
    .query()
    .where("tissue = 'bone marrow'")
    .feature_spaces("gene_expression")
    .layers("gene_expression", ["log_normalized"])
    .feature_join("intersection")
    .limit(5000)
    .to_anndata()
)
```

The call order of builder methods does not matter — parameters accumulate independently and are resolved at execution time.
