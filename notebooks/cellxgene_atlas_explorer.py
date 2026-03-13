# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "lancell",
#     "lancedb",
#     "obstore",
#     "polars",
#     "anndata",
#     "scanpy",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys

    import marimo as mo
    from tqdm.auto import tqdm

    # Allow imports from the repo root (lancell + examples)
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    return mo, os, tqdm


@app.cell
def _(mo):
    mo.md(r"""
    # CellxGene Census Atlas Explorer

    Open a lancell ragged atlas built from cellxgene census h5ad files,
    browse metadata, query cells, and reconstruct AnnData objects.
    """)
    return


@app.cell
def _(mo):
    atlas_dir_input = mo.ui.text(
        value="s3://epiblast/ragged_atlases/cellxgene_mini_bp/",
        label="Atlas directory",
        full_width=True,
    )
    atlas_dir_input
    return (atlas_dir_input,)


@app.cell
def _(atlas_dir_input, os):
    import obstore.store
    from pathlib import Path

    atlas_dir = atlas_dir_input.value

    def _make_store(atlas_dir: str) -> obstore.store.ObjectStore:
        if atlas_dir.startswith("s3://"):
            from urllib.parse import urlparse

            parsed = urlparse(atlas_dir)
            bucket = parsed.netloc
            prefix = os.path.join(parsed.path.strip("/"), "zarr_store")
            region = os.environ.get("AWS_REGION", "us-east-2")
            return obstore.store.S3Store(bucket, prefix=prefix, region=region)
        zarr_path = Path(atlas_dir) / "zarr_store"
        return obstore.store.LocalStore(str(zarr_path))

    def _db_uri(atlas_dir: str) -> str:
        if atlas_dir.startswith("s3://"):
            return atlas_dir.rstrip("/") + "/lance_db"
        return str(Path(atlas_dir) / "lance_db")

    store = _make_store(atlas_dir)
    db_uri = _db_uri(atlas_dir)
    return db_uri, store


@app.cell
def _(db_uri, store):
    from examples.cellxgene_census.schema import CellObs
    from lancell.atlas import RaggedAtlas

    atlas = RaggedAtlas.open(
        db_uri=db_uri,
        cell_table_name="cells",
        cell_schema=CellObs,
        dataset_table_name="datasets",
        store=store,
        registry_tables={"gene_expression": "gene_expression_registry"},
    )
    return (atlas,)


@app.cell
def _(mo):
    mo.md("""
    ## Datasets
    """)
    return


@app.cell
def _(atlas):
    datasets_df = atlas.list_datasets()
    datasets_df
    return


@app.cell
def _(mo):
    mo.md("""
    ## Gene Registry
    """)
    return


@app.cell
def _(atlas):
    registry = atlas._registry_tables["gene_expression"]
    genes_df = registry.search().to_polars()
    genes_df
    return (genes_df,)


@app.cell
def _(atlas, genes_df, mo):
    mo.md(f"""
    **{genes_df.height:,}** genes registered across
    **{atlas.list_datasets().height}** datasets.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Query Cells
    """)
    return


@app.cell
def _(mo):
    where_input = mo.ui.text(
        value="cell_type = 'CD4-positive, alpha-beta T cell'",
        label="WHERE clause (LanceDB SQL)",
        full_width=True,
    )
    limit_input = mo.ui.slider(
        start=10,
        stop=5000,
        value=100,
        step=10,
        label="Limit",
    )
    mo.hstack([where_input, limit_input])
    return limit_input, where_input


@app.cell
def _(atlas, limit_input, where_input):
    cells_df = (
        atlas.query()
        .where(where_input.value)
        .limit(limit_input.value)
        .to_polars()
    )
    cells_df
    return (cells_df,)


@app.cell
def _(cells_df, mo):
    mo.md(f"""
    Returned **{cells_df.height:,}** cells.
    """)
    return


@app.cell
def _(atlas):
    cell_type_df = (
        atlas.query()
        .select(["cell_type"])
        .to_polars()
    )
    cell_type_df["cell_type"].value_counts()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reconstruct AnnData

    Reconstruct a full AnnData object (obs + sparse X matrix) from the
    queried cells. This reads the zarr shards on S3 and remaps features
    to the global gene registry.
    """)
    return


@app.cell
def _(atlas, limit_input, where_input):
    adata = (
        atlas.query()
        .where(where_input.value)
        .limit(limit_input.value)
        .to_anndata()
    )
    adata
    return (adata,)


@app.cell
def _(adata, mo):
    mo.md(f"""
    **Reconstructed AnnData**: {adata.n_obs:,} cells x {adata.n_vars:,} genes

    - **X shape**: `{adata.X.shape}`
    - **X nnz**: `{adata.X.nnz:,}`
    - **obs columns**: `{list(adata.obs.columns)}`
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Top expressed genes
    """)
    return


@app.cell
def _(adata):
    import numpy as np
    import polars as pl

    mean_expr = np.asarray(adata.X.mean(axis=0)).ravel()
    top_genes = pl.DataFrame({
        "gene": adata.var.index.tolist(),
        "mean_expression": mean_expr,
    }).sort("mean_expression", descending=True).head(20)
    top_genes
    return np, pl


@app.cell
def _(mo):
    mo.md("""
    ### Cell type distribution in query results
    """)
    return


@app.cell
def _(adata):
    adata.obs["cell_type"].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Batch loading
    """)
    return


@app.cell
def _(atlas, limit_input, where_input):
    batch_iterator = (
        atlas.query()
        .where(where_input.value)
        .limit(limit_input.value)
        .to_batches(batch_size=512)
    )
    for batch_adata in batch_iterator:
        print("Loaded batch", batch_adata.shape)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Fast ML Dataloader

    The `to_cell_dataset()` path skips AnnData/scipy reconstruction entirely,
    yielding lightweight `SparseBatch` objects (flat CSR-style arrays).
    Concurrent async dispatch across zarr groups means cells spanning
    multiple datasets are fetched in parallel.
    """)
    return


@app.cell
def _(atlas, tqdm):
    import time

    q = (
        atlas.query()
        # .where(where_input.value)
        .where("sex == 'male'")
        # .limit(10_000)
    )

    cell_dataset = q.to_cell_dataset(
        feature_space="gene_expression",
        layer="counts",
        batch_size=256,
        shuffle=True,
        seed=42,
        metadata_columns=["cell_type", "tissue"],
    )

    print(f"CellDataset: {cell_dataset.n_cells:,} cells, "
          f"{cell_dataset.n_features:,} features, "
          f"{len(cell_dataset)} batches")

    t0 = time.perf_counter()
    for batch in tqdm(cell_dataset):
        pass  # consume one epoch
    elapsed = time.perf_counter() - t0

    print(f"Epoch time: {elapsed:.3f}s "
          f"({elapsed / max(len(cell_dataset), 1) * 1000:.1f} ms/batch)")
    return cell_dataset, q


@app.cell
def _(cell_dataset, mo):
    # Grab first batch for inspection
    first_batch = next(iter(cell_dataset))
    n_cells = len(first_batch.offsets) - 1
    nnz = int(first_batch.offsets[-1])

    mo.md(f"""
    ### SparseBatch structure

    | Field | Shape / Value |
    |-------|---------------|
    | `indices` | `{first_batch.indices.shape}` ({first_batch.indices.dtype}) |
    | `values` | `{first_batch.values.shape}` ({first_batch.values.dtype}) |
    | `offsets` | `{first_batch.offsets.shape}` ({first_batch.offsets.dtype}) |
    | `n_features` | `{first_batch.n_features:,}` |
    | cells in batch | `{n_cells}` |
    | total nnz | `{nnz:,}` |
    | metadata keys | `{list(first_batch.metadata.keys()) if first_batch.metadata else None}` |
    """)
    return (first_batch,)


@app.cell
def _(first_batch, mo, np, pl):
    lengths = np.diff(first_batch.offsets)
    nnz_stats = pl.DataFrame({
        "stat": ["min", "median", "mean", "max"],
        "nnz_per_cell": [
            float(lengths.min()),
            float(np.median(lengths)),
            round(float(lengths.mean()), 1),
            float(lengths.max()),
        ],
    })

    mo.md(f"""
    ### Per-cell sparsity

    {mo.as_html(nnz_stats)}

    Average density: **{lengths.mean() / first_batch.n_features * 100:.1f}%**
    ({lengths.mean():.0f} / {first_batch.n_features:,} features)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Dense collate

    `sparse_to_dense_collate` converts a `SparseBatch` into a dense
    float32 torch tensor via scatter — ready for a standard MLP or
    transformer encoder.
    """)
    return


@app.cell
def _(first_batch, pl):
    from lancell.dataloader import sparse_to_dense_collate

    dense_result = sparse_to_dense_collate(first_batch)

    X = dense_result["X"]
    print(f"X tensor: {X.shape}, dtype={X.dtype}")
    print(f"Sparsity: {(X == 0).float().mean():.1%} zeros")

    if "cell_type" in dense_result:
        cell_types = dense_result["cell_type"]
        print(f"\nCell types in batch:")
        print(pl.Series("cell_type", cell_types).value_counts())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Comparison: `to_batches()` vs `to_cell_dataset()`

    `to_batches()` reconstructs a full AnnData per batch (obs DataFrame,
    var DataFrame, scipy CSR, AnnData object). `to_cell_dataset()` skips
    all of that and returns flat numpy arrays.
    """)
    return


@app.cell
def _(q):
    import time as _time

    # AnnData path
    _t0 = _time.perf_counter()
    for _adata_batch in q.to_batches(batch_size=256):
        pass
    anndata_time = _time.perf_counter() - _t0

    # CellDataset path
    _ds = q.to_cell_dataset(batch_size=256, shuffle=False)
    _t0 = _time.perf_counter()
    for _batch in _ds:
        pass
    dataset_time = _time.perf_counter() - _t0

    speedup = anndata_time / max(dataset_time, 1e-9)
    print(f"to_batches() (AnnData):    {anndata_time:.3f}s")
    print(f"to_cell_dataset() (raw):   {dataset_time:.3f}s")
    print(f"Speedup:                   {speedup:.1f}x")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
