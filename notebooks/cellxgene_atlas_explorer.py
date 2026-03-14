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
#     "torch",
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
    from lancell.schema import AtlasVersionRecord

    # Allow imports from the repo root (lancell + examples)
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    return mo, os, tqdm


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(atlas, genes_df, mo):
    mo.md(f"""
    **{genes_df.height:,}** genes registered across
    **{atlas.list_datasets().height}** datasets.
    """)
    return


@app.cell(hide_code=True)
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
        value=1500,
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(adata, mo):
    mo.md(f"""
    **Reconstructed AnnData**: {adata.n_obs:,} cells x {adata.n_vars:,} genes

    - **X shape**: `{adata.X.shape}`
    - **X nnz**: `{adata.X.nnz:,}`
    - **obs columns**: `{list(adata.obs.columns)}`
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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
    ## Feature-Filtered Query

    `.features(uids, feature_space)` requests a specific subset of genes by
    global UID. The filter composes naturally with `.where()` and `.limit()`.
    Reconstruction memory is proportional to the requested feature set, not
    the full gene space — I/O is unchanged (full cell ranges are still read,
    the filter is applied during remapping).
    """)
    return


@app.cell
def _(adata, np):
    # Seed the text area with the top-5 expressed genes from the full query
    _mean_expr = np.asarray(adata.X.mean(axis=0)).ravel()
    _top_idx = np.argsort(_mean_expr)[::-1][:5]
    default_feature_uids = [adata.var.index[i] for i in _top_idx]
    return (default_feature_uids,)


@app.cell
def _(default_feature_uids, mo):
    feature_uid_input = mo.ui.text_area(
        value="\n".join(default_feature_uids),
        label="Feature UIDs to request (one per line)",
        rows=6,
        full_width=True,
    )
    feature_uid_input
    return (feature_uid_input,)


@app.cell
def _(atlas, feature_uid_input, limit_input, where_input):
    selected_uids = [
        u.strip()
        for u in feature_uid_input.value.strip().splitlines()
        if u.strip()
    ]
    filtered_adata = (
        atlas.query()
        .where(where_input.value)
        .limit(limit_input.value)
        .features(selected_uids, "gene_expression")
        .to_anndata()
    )
    filtered_adata
    return filtered_adata, selected_uids


@app.cell(hide_code=True)
def _(adata, filtered_adata, mo, selected_uids):
    mo.md(f"""
    **Feature filter result**

    | | Value |
    |---|---|
    | Requested UIDs | `{len(selected_uids)}` |
    | `filtered_adata.n_vars` | `{filtered_adata.n_vars}` |
    | Full atlas `n_vars` | `{adata.n_vars:,}` |
    | `filtered_adata.X.shape` | `{filtered_adata.X.shape}` |
    | `filtered_adata.X.nnz` | `{filtered_adata.X.nnz:,}` |

    `var` index: `{filtered_adata.var.index.tolist()}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Feature filter + ML dataloader

    The same `.features()` call works on `to_cell_dataset()`.
    `CellDataset.n_features` reflects the filtered count and batch
    `indices` are bounded by that value.
    """)
    return


@app.cell
def _(atlas, feature_uid_input, where_input):
    from lancell.sampler import CellSampler as _CellSampler

    _selected = [
        u.strip()
        for u in feature_uid_input.value.strip().splitlines()
        if u.strip()
    ]
    filtered_ds = (
        atlas.query()
        .where(where_input.value)
        .features(_selected, "gene_expression")
        .to_cell_dataset(layer="counts")
    )
    _sampler = _CellSampler(filtered_ds.groups_np, batch_size=256, shuffle=False)
    _first_batch = filtered_ds.__getitems__(next(iter(_sampler)))
    print(f"n_features (filtered) : {filtered_ds.n_features}")
    print(f"n_batches             : {len(_sampler)}")
    print(f"indices range         : [{_first_batch.indices.min() if len(_first_batch.indices) else 'n/a'}, "
          f"{_first_batch.indices.max() if len(_first_batch.indices) else 'n/a'}]")
    print(f"indices (first 20)    : {_first_batch.indices[:20]}")
    print(f"values  (first 20)    : {_first_batch.values[:20]}")
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


@app.cell(hide_code=True)
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
def _(atlas, tqdm, where_input):
    import time
    from lancell.sampler import CellSampler as _CellSampler2

    q = (
        atlas.query()
        .where(where_input.value)
        .limit(10_000)
    )

    cell_dataset = q.to_cell_dataset(
        feature_space="gene_expression",
        layer="counts",
        metadata_columns=["cell_type", "tissue"],
    )
    cell_sampler = _CellSampler2(
        cell_dataset.groups_np, batch_size=256, shuffle=True, seed=42
    )

    print(f"CellDataset: {cell_dataset.n_cells:,} cells, "
          f"{cell_dataset.n_features:,} features, "
          f"{len(cell_sampler)} batches")

    t0 = time.perf_counter()
    for _indices in tqdm(cell_sampler):
        cell_dataset.__getitems__(_indices)  # consume one epoch
    elapsed = time.perf_counter() - t0

    print(f"Epoch time: {elapsed:.3f}s "
          f"({elapsed / max(len(cell_sampler), 1) * 1000:.1f} ms/batch)")
    return cell_dataset, cell_sampler, q


@app.cell
def _(cell_dataset, cell_sampler, mo):
    # Grab first batch for inspection
    first_batch = cell_dataset.__getitems__(next(iter(cell_sampler)))
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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
    from lancell.sampler import CellSampler as _CellSampler3

    # AnnData path
    _t0 = _time.perf_counter()
    for _adata_batch in q.to_batches(batch_size=256):
        pass
    anndata_time = _time.perf_counter() - _t0

    # CellDataset path
    _ds = q.to_cell_dataset()
    _s = _CellSampler3(_ds.groups_np, batch_size=256, shuffle=False)
    _t0 = _time.perf_counter()
    for _idx in _s:
        _ds.__getitems__(_idx)
    dataset_time = _time.perf_counter() - _t0

    speedup = anndata_time / max(dataset_time, 1e-9)
    print(f"to_batches() (AnnData):    {anndata_time:.3f}s")
    print(f"to_cell_dataset() (raw):   {dataset_time:.3f}s")
    print(f"Speedup:                   {speedup:.1f}x")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Map-style DataLoader (multi-worker)

    `CellDataset` owns data access; `CellSampler` owns batch planning.
    Use `make_loader(dataset, sampler)` to wire them into a DataLoader.

    Key APIs:
    - **`CellSampler(groups_np, batch_size, num_workers)`** — bin-packs zarr
      groups across workers, keeping each worker's reader cache warm.
    - **`sampler.set_epoch(epoch)`** — re-shuffles the batch plan for the new
      epoch.  Call it before each DataLoader iteration.
    - **`make_loader(dataset, sampler)`** — sets `batch_sampler`, `collate_fn`,
      `multiprocessing_context="spawn"`, and `persistent_workers=False`.
    """)
    return


@app.cell
def _(atlas, mo):
    from lancell.dataloader import CellDataset, make_loader
    from lancell.sampler import CellSampler

    _num_workers = 2
    mw_dataset = CellDataset(
        atlas=atlas,
        cells_pl=(
            atlas.query()
            .where("sex == 'male'")
            .limit(10_000)
            .to_polars()
        ),
        feature_space="gene_expression",
        layer="counts",
        metadata_columns=["cell_type"],
    )
    mw_sampler = CellSampler(
        mw_dataset.groups_np,
        batch_size=256,
        shuffle=True,
        seed=0,
        num_workers=_num_workers,
    )

    mo.md(f"""
    **Dataset summary**

    | | |
    |---|---|
    | cells | `{mw_dataset.n_cells:,}` |
    | features | `{mw_dataset.n_features:,}` |
    | batches (epoch 0) | `{len(mw_sampler)}` |
    | `num_workers` | `{_num_workers}` |
    """)
    return make_loader, mw_dataset, mw_sampler


@app.cell
def _(make_loader, mo, mw_dataset, mw_sampler):
    import time as _time

    _n_epochs = 2
    _epoch_times = []

    for _epoch in range(_n_epochs):
        mw_sampler.set_epoch(_epoch)
        # For this notebook we override num_workers=0 (no subprocess spawn needed)
        _loader = make_loader(mw_dataset, mw_sampler, num_workers=0)

        _t0 = _time.perf_counter()
        _total_cells = 0
        for _batch in _loader:
            _total_cells += len(_batch.offsets) - 1
        _epoch_times.append(_time.perf_counter() - _t0)

        print(f"epoch {_epoch}: {_total_cells:,} cells in {_epoch_times[-1]:.3f}s "
              f"({_epoch_times[-1] / max(len(mw_sampler), 1) * 1000:.1f} ms/batch)")

    mo.md(f"""
    ### Multi-epoch results

    | Epoch | Time (s) | ms/batch |
    |-------|----------|----------|
    {"".join(f"| {e} | {t:.3f} | {t / max(len(mw_sampler), 1) * 1000:.1f} |" + chr(10) for e, t in enumerate(_epoch_times))}

    Each epoch covers all **{mw_dataset.n_cells:,}** cells exactly once.
    `set_epoch` re-shuffles with a new seed (`seed + epoch`) so the cell
    order differs every epoch.
    """)
    return


@app.cell
def _(mo, mw_dataset, mw_sampler):
    import pickle

    # Show that epoch 0 and epoch 1 produce different orderings
    _s = mw_sampler
    _s.set_epoch(0)
    _batch0_e0 = mw_dataset.__getitems__(next(iter(_s)))
    _first_nnz_e0 = int(_batch0_e0.offsets[-1])

    _s.set_epoch(1)
    _batch0_e1 = mw_dataset.__getitems__(next(iter(_s)))
    _first_nnz_e1 = int(_batch0_e1.offsets[-1])

    # Verify pickle safety (required for spawn-based multiprocessing)
    _pickled = pickle.dumps(mw_dataset)
    _restored = pickle.loads(_pickled)
    _pickle_ok = _restored._local_readers is None

    mo.md(f"""
    ### Epoch shuffle verification

    Batch 0 nnz — epoch 0: **{_first_nnz_e0:,}** / epoch 1: **{_first_nnz_e1:,}**
    *(different batches contain different cells)*

    **Pickle safety**: `{'✓ dataset pickles cleanly (worker-local state stripped)' if _pickle_ok else '✗ pickle failed'}`

    ---

    ### Usage with real multi-worker training

    ```python
    dataset = atlas.query().to_cell_dataset(metadata_columns=["cell_type"])
    sampler = CellSampler(dataset.groups_np, batch_size=256,
                          shuffle=True, seed=42, num_workers=4)

    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)
        loader = make_loader(dataset, sampler)   # sets spawn, batch_sampler, etc.
        for batch in loader:
            X = sparse_to_dense_collate(batch)["X"]   # (B, n_features) float32
            loss = model(X).loss
            loss.backward()
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Balanced Sampling

    By default, `CellSampler` shuffles cells randomly — batches reflect the
    natural class imbalance in the data.  Rare cell types can be swamped by
    common ones and may barely appear in any single batch.

    `BalancedCellSampler` fixes this: every batch draws exactly
    `batch_size // n_categories` cells from each unique value.  The epoch
    length is bounded by the smallest category; cells from larger categories
    beyond `cells_per_cat × n_batches` are skipped and sampled again in
    future epochs.
    """)
    return


@app.cell
def _(mo):
    balance_col_input = mo.ui.dropdown(
        options=["cell_type", "dataset_uid", "tissue"],
        value="cell_type",
        label="Balance on column",
    )
    balance_batch_size_input = mo.ui.slider(
        64, 512, value=128, step=64, label="Batch size"
    )
    mo.hstack([balance_col_input, balance_batch_size_input])
    return balance_batch_size_input, balance_col_input


@app.cell
def _(atlas, balance_col_input):
    _balance_col = balance_col_input.value
    bal_cells_pl = atlas.query().where("sex == 'male'").to_polars()

    category_counts = (
        bal_cells_pl[_balance_col].value_counts().sort("count", descending=True)
    )
    category_counts
    return (bal_cells_pl,)


@app.cell
def _(atlas, bal_cells_pl, balance_batch_size_input, balance_col_input, mo):
    from lancell.dataloader import CellDataset as _CellDataset
    from lancell.sampler import BalancedCellSampler

    _balance_col = balance_col_input.value
    _batch_size = balance_batch_size_input.value
    balanced_ds = _CellDataset(
        atlas=atlas,
        cells_pl=bal_cells_pl,
        feature_space="gene_expression",
        layer="counts",
        metadata_columns=[_balance_col],
    )
    balanced_sampler = BalancedCellSampler.from_column(
        balanced_ds.cells_pl,
        _balance_col,
        batch_size=_batch_size,
        shuffle=True,
        seed=42,
        drop_last=True,
    )

    _n_cats = len(bal_cells_pl[_balance_col].unique())
    _per_cat = max(1, _batch_size // _n_cats)

    mo.md(f"""
    **Balanced `BalancedCellSampler`** — `column={_balance_col!r}`

    | | |
    |---|---|
    | Categories (`n_cats`) | `{_n_cats}` |
    | Cells / category / batch | `{_per_cat}` (`{_batch_size}` ÷ `{_n_cats}`) |
    | Batches / epoch | `{len(balanced_sampler)}` |

    *Epoch bounded by the smallest category.  Larger categories contribute
    only `{_per_cat} × {len(balanced_sampler)}` = `{_per_cat * len(balanced_sampler):,}` cells.*
    """)
    return balanced_ds, balanced_sampler


@app.cell
def _(balance_col_input, balanced_ds, balanced_sampler, mo, tqdm):
    import numpy as _np
    import polars as _pl

    _balance_col = balance_col_input.value

    # Collect per-batch category counts over one epoch
    _rows = []
    for _i, _indices in tqdm(enumerate(balanced_sampler)):
        _batch = balanced_ds.__getitems__(_indices)
        if _batch.metadata is None:
            break
        _vals = _batch.metadata[_balance_col]
        for _cat, _cnt in zip(*_np.unique(_vals, return_counts=True)):
            _rows.append({"batch": _i, "category": str(_cat), "count": int(_cnt)})

    _stats = (
        _pl.DataFrame(_rows)
        .group_by("category")
        .agg(
            _pl.col("count").min().alias("min / batch"),
            _pl.col("count").mean().round(1).alias("mean / batch"),
            _pl.col("count").max().alias("max / batch"),
        )
        .sort("mean / batch", descending=True)
    )

    mo.md(f"""
    ### Per-category representation across {len(balanced_sampler)} batches

    {mo.as_html(_stats)}

    `min` ≈ `max` confirms every batch saw a balanced draw.
    Variation in the last batch (`drop_last=False`) accounts for any difference.
    """)
    return


@app.cell
def _(bal_cells_pl, balance_col_input, balanced_sampler, mo):
    _balance_col = balance_col_input.value

    # Unbalanced: natural frequency of each category
    _natural = bal_cells_pl[_balance_col].value_counts().sort("count", descending=True)
    _total = bal_cells_pl.height
    _n_cats2 = _natural.height

    # Balanced: equal cells per category
    _cells_per_cat = max(1, balanced_sampler.batch_size // _n_cats2)
    _balanced_epoch_cells = _cells_per_cat * len(balanced_sampler)

    _rows2 = []
    for _row in _natural.iter_rows(named=True):
        _cat = _row[_balance_col]
        _nat_pct = _row["count"] / _total * 100
        _bal_pct = 100.0 / _n_cats2
        _rows2.append({
            "category": str(_cat),
            "natural %": round(_nat_pct, 1),
            "balanced %": round(_bal_pct, 1),
        })

    import polars as _pl2
    _comparison = _pl2.DataFrame(_rows2)

    mo.md(f"""
    ### Natural vs balanced per-batch frequency

    {mo.as_html(_comparison)}

    With `column={_balance_col!r}`, every category appears in
    **{round(100.0 / _n_cats2, 1)}%** of each batch regardless of its natural
    frequency in the atlas.

    ---

    ### Usage

    ```python
    dataset = atlas.query().to_cell_dataset(metadata_columns=["cell_type"])
    sampler = BalancedCellSampler.from_column(
        dataset.cells_pl, "cell_type", batch_size=256
    )

    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)
        loader = make_loader(dataset, sampler)
        for batch in loader:
            # batch.metadata["cell_type"] has equal counts per type
            X = sparse_to_dense_collate(batch)["X"]
    ```
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
