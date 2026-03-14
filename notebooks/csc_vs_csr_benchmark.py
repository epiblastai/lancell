# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "lancell",
#     "lancedb",
#     "obstore",
#     "polars",
#     "anndata",
#     "pyarrow",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys
    import time

    import marimo as mo
    import numpy as np
    import polars as pl

    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    return mo, np, os, pl, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CSC vs CSR Feature Slicing Benchmark

    Two lancell atlases built from the same two CellxGene Census h5ad files (~26,600 cells,
    ~20,000 genes). The only difference is the zarr layout:

    | Atlas | Layout | S3 path |
    |-------|--------|---------|
    | **CSC** | CSR + CSC (feature-contiguous copy) | `s3://…/cellxgene_mini_csc/` |
    | **CSR** | CSR only (cell-contiguous) | `s3://…/cellxgene_mini_csr/` |

    **Key question**: how much faster is `atlas.query().features(uids).to_anndata()` when the
    CSC layout is present?

    ### Why the paths differ

    For a feature-filtered query on **N cells** and **F genes** out of **G total genes**:

    | Path | I/O cost | Scales with |
    |------|----------|-------------|
    | **CSC** | `sum(csc_end[f] - csc_start[f])` per requested gene | nnz for those F genes |
    | **CSR fallback** | `sum(end[i] - start[i])` per queried cell | nnz for all N cells × avg sparsity |

    When F ≪ G the CSC path reads a tiny fraction of the data the CSR path must read.
    """)
    return


@app.cell
def _(os):
    import obstore.store

    from examples.cellxgene_census.schema import CellObs
    from lancell.atlas import RaggedAtlas

    _region = os.environ.get("AWS_REGION", "us-east-2")

    def _open(atlas_dir: str) -> RaggedAtlas:
        from urllib.parse import urlparse

        _parsed = urlparse(atlas_dir)
        _store = obstore.store.S3Store(
            _parsed.netloc,
            prefix=os.path.join(_parsed.path.strip("/"), "zarr_store"),
            region=_region,
        )
        return RaggedAtlas.open(
            db_uri=atlas_dir.rstrip("/") + "/lance_db",
            cell_table_name="cells",
            cell_schema=CellObs,
            dataset_table_name="datasets",
            store=_store,
            registry_tables={"gene_expression": "gene_expression_registry"},
        )

    atlas_csc = _open("s3://epiblast/ragged_atlases/cellxgene_mini_csc/")
    atlas_csr = _open("s3://epiblast/ragged_atlases/cellxgene_mini_csr/")
    return atlas_csc, atlas_csr


@app.cell
def _(atlas_csc, atlas_csr, mo):
    _n_cells_csc = atlas_csc.cell_table.count_rows()
    _n_cells_csr = atlas_csr.cell_table.count_rows()
    _n_genes = atlas_csc._registry_tables["gene_expression"].count_rows()
    _groups = atlas_csc.list_datasets()["zarr_group"].to_list()
    _csc_ok = [atlas_csc._has_csc(g) for g in _groups]

    mo.md(f"""
    ## Atlas Stats

    | | CSC atlas | CSR atlas |
    |---|---|---|
    | Cells | `{_n_cells_csc:,}` | `{_n_cells_csr:,}` |
    | Genes (registry) | `{_n_genes:,}` | `{_n_genes:,}` |
    | Datasets | `{len(_groups)}` | `{len(_groups)}` |
    | CSC groups | `{sum(_csc_ok)}/{len(_groups)}` | `0/{len(_groups)}` |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Benchmark Gene Selection

    Pull a small batch of cells and rank genes by nnz count.
    We store **ensembl IDs** (stable across atlases) and resolve to per-atlas UIDs
    below, since each atlas generated its own random UIDs at ingest time.
    """)
    return


@app.cell
def _(atlas_csc, np):
    import scipy.sparse as _sp

    _batch = next(atlas_csc.query().limit(500).to_batches(batch_size=500))
    _X = _sp.csr_matrix(_batch.X) if not _sp.issparse(_batch.X) else _batch.X
    _nnz_per_gene = np.array(_X.getnnz(axis=0)).ravel()
    _top_idx = np.argsort(_nnz_per_gene)[::-1]
    # Store ensembl IDs — stable identifiers shared between both atlases
    benchmark_ensembl_ids = [_batch.var["ensembl_id"].iloc[i] for i in _top_idx if _nnz_per_gene[i] > 0]
    print(f"Expressed genes in sample: {len(benchmark_ensembl_ids)}")
    print(f"Top 10 (by nnz): {benchmark_ensembl_ids[:10]}")
    return (benchmark_ensembl_ids,)


@app.cell
def _(atlas_csc, atlas_csr, benchmark_ensembl_ids):
    import polars as _pl

    def _ensembl_to_uids(atlas, ensembl_ids: list[str]) -> list[str]:
        """Resolve ensembl IDs to registry UIDs for the given atlas."""
        _df = atlas._registry_tables["gene_expression"].search().select(["uid", "ensembl_id"]).to_polars()
        _mapping = dict(zip(_df["ensembl_id"].to_list(), _df["uid"].to_list()))
        return [_mapping[e] for e in ensembl_ids if e in _mapping]

    csc_uids = _ensembl_to_uids(atlas_csc, benchmark_ensembl_ids)
    csr_uids = _ensembl_to_uids(atlas_csr, benchmark_ensembl_ids)
    print(f"Resolved {len(csc_uids)} UIDs in CSC atlas, {len(csr_uids)} in CSR atlas")
    return csc_uids, csr_uids


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Benchmark 1 — Full Cell-Wise Query (all features)

    No feature filter → both atlases use the CSR path.
    Expected: near-identical performance.
    """)
    return


@app.cell
def _(atlas_csc, atlas_csr, mo, time):
    def _bench_full(atlas, n_cells, n_reps=2):
        best = float("inf")
        for _ in range(n_reps):
            t0 = time.perf_counter()
            _a = atlas.query().limit(n_cells).to_anndata()
            best = min(best, time.perf_counter() - t0)
        return best, _a.X.shape, _a.X.nnz

    _t_csc, _shape, _nnz = _bench_full(atlas_csc, 5_000)
    _t_csr, _, _ = _bench_full(atlas_csr, 5_000)

    mo.md(f"""
    ### 5,000 cells, all ~{atlas_csc._registry_tables["gene_expression"].count_rows():,} genes

    | Atlas | Best time (s) | Matrix shape | NNZ read |
    |-------|--------------|-------------|---------|
    | CSC   | `{_t_csc:.3f}` | `{_shape}` | `{_nnz:,}` |
    | CSR   | `{_t_csr:.3f}` | `{_shape}` | `{_nnz:,}` |
    | Ratio (CSC/CSR) | `{_t_csc / max(_t_csr, 1e-9):.2f}x` | | |

    Both read the same data — ratio should be ≈ 1.0.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Benchmark 2 — Feature Filter: Sweep Over Number of Genes

    `atlas.query().features(uids, "gene_expression").to_anndata()` — **all cells, no limit**.

    - **CSC atlas**: reads `csc/indices[csc_start[f]:csc_end[f]]` for each requested gene
    - **CSR atlas**: reads every queried cell's full row span, discards unrequested genes
    """)
    return


@app.cell
def _(atlas_csc, atlas_csr, csc_uids, csr_uids, pl, time):
    _n_features_to_test = [1, 5, 10, 25, 50, 100, 250, 500]
    _N_REPS = 2

    _rows = []
    for _n in _n_features_to_test:
        _best_csc = float("inf")
        for _ in range(_N_REPS):
            _t0 = time.perf_counter()
            atlas_csc.query().features(csc_uids[:_n], "gene_expression").to_anndata()
            _best_csc = min(_best_csc, time.perf_counter() - _t0)

        _best_csr = float("inf")
        for _ in range(_N_REPS):
            _t0 = time.perf_counter()
            atlas_csr.query().features(csr_uids[:_n], "gene_expression").to_anndata()
            _best_csr = min(_best_csr, time.perf_counter() - _t0)

        _rows.append({
            "n_features": _n,
            "csc_s": round(_best_csc, 3),
            "csr_s": round(_best_csr, 3),
            "speedup": round(_best_csr / max(_best_csc, 1e-9), 2),
        })
        print(
            f"n={_n:4d}  CSC={_best_csc:.3f}s  CSR={_best_csr:.3f}s  "
            f"speedup={_best_csr / max(_best_csc, 1e-9):.1f}x"
        )

    feature_sweep_df = pl.DataFrame(_rows)
    feature_sweep_df
    return (feature_sweep_df,)


@app.cell
def _(feature_sweep_df, mo):
    _peak = feature_sweep_df.filter(
        feature_sweep_df["speedup"] == feature_sweep_df["speedup"].max()
    )
    _at_500 = feature_sweep_df.filter(feature_sweep_df["n_features"] == 500)

    mo.md(f"""
    ### Observations

    - Peak speedup: **{_peak["speedup"][0]:.1f}x** at `n_features = {_peak["n_features"][0]}`
    - At 500 features: **{_at_500["speedup"][0]:.1f}x** (paths begin to converge)

    The CSC gain is largest when you request very few genes from a large atlas.
    For the CSR path the cost is roughly constant: it reads every queried cell's full nnz
    regardless of how many features you ultimately keep.
    As the requested feature set grows toward the full gene space the two paths converge
    because the CSC path must read proportionally more of the flat arrays.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Benchmark 3 — Feature Filter: Sweep Over Cell Count

    Fixed **5 genes**; vary how many cells are queried.

    The CSC path reads the full feature range for a gene across **all cells in each zarr group**,
    then filters to the queried subset by `zarr_row`. So CSC latency is nearly flat
    until a second zarr group (dataset) is added to the result. CSR latency scales
    linearly with the number of cells returned by LanceDB.
    """)
    return


@app.cell
def _(atlas_csc, atlas_csr, csc_uids, csr_uids, mo, pl, time):
    _cell_counts = [100, 500, 1000, 2500, 5000, 10_000, 26_000]
    _N_REPS2 = 2

    _rows2 = []
    for _nc in _cell_counts:
        _best_csc2 = float("inf")
        for _ in range(_N_REPS2):
            _t0 = time.perf_counter()
            atlas_csc.query().limit(_nc).features(csc_uids[:5], "gene_expression").to_anndata()
            _best_csc2 = min(_best_csc2, time.perf_counter() - _t0)

        _best_csr2 = float("inf")
        for _ in range(_N_REPS2):
            _t0 = time.perf_counter()
            atlas_csr.query().limit(_nc).features(csr_uids[:5], "gene_expression").to_anndata()
            _best_csr2 = min(_best_csr2, time.perf_counter() - _t0)

        _rows2.append({
            "n_cells": _nc,
            "csc_s": round(_best_csc2, 3),
            "csr_s": round(_best_csr2, 3),
            "speedup": round(_best_csr2 / max(_best_csc2, 1e-9), 2),
        })
        print(
            f"n_cells={_nc:6d}  CSC={_best_csc2:.3f}s  CSR={_best_csr2:.3f}s  "
            f"speedup={_best_csr2 / max(_best_csc2, 1e-9):.1f}x"
        )

    cell_sweep_df = pl.DataFrame(_rows2)

    mo.md(f"""
    ### Cell count sweep (5 genes)

    {mo.as_html(cell_sweep_df.rename({
        "n_cells": "# cells",
        "csc_s": "CSC (s)",
        "csr_s": "CSR (s)",
        "speedup": "Speedup (CSR/CSC)",
    }))}

    CSR time grows approximately linearly with cell count (each extra cell adds ~avg nnz).
    CSC time is nearly flat across cell counts — it reads the 5 genes' full feature slices once
    per zarr group, independent of how many cells LanceDB returns from that group.
    The CSC advantage is therefore **largest when querying a small fraction of all cells**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Usage Patterns

    ### Pattern 1: Fetch one gene for all cells (peak CSC use case)

    Request a single gene across the entire atlas.
    CSC reads only the nnz for that gene; CSR reads all cell spans and discards the rest.
    """)
    return


@app.cell
def _(
    atlas_csc,
    atlas_csr,
    benchmark_ensembl_ids,
    csc_uids,
    csr_uids,
    mo,
    time,
):
    _t0 = time.perf_counter()
    _a_csc = atlas_csc.query().features(csc_uids[:1], "gene_expression").to_anndata()
    _t_csc = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    _a_csr = atlas_csr.query().features(csr_uids[:1], "gene_expression").to_anndata()
    _t_csr = time.perf_counter() - _t0

    mo.md(f"""
    Gene `{benchmark_ensembl_ids[0]}` — **{_a_csc.n_obs:,} cells**

    | Atlas | Time (s) | Shape | NNZ |
    |-------|---------|-------|-----|
    | CSC | `{_t_csc:.3f}` | `{_a_csc.X.shape}` | `{_a_csc.X.nnz:,}` |
    | CSR | `{_t_csr:.3f}` | `{_a_csr.X.shape}` | `{_a_csr.X.nnz:,}` |
    | **Speedup** | **{_t_csr / max(_t_csc, 1e-9):.1f}x** | | |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Pattern 2: Feature filter + cell type restriction

    `.features()` composes naturally with `.where()` and `.limit()`.
    The CSC path still applies when `atlas._has_csc()` is true for a group.
    """)
    return


@app.cell
def _(atlas_csc):
    # Pick the most common cell type that is actually present in the mini atlas
    _ct_df = atlas_csc.cell_table.search().select(["cell_type"]).to_polars()
    _top_ct = (
        _ct_df["cell_type"]
        .drop_nulls()
        .value_counts()
        .sort("count", descending=True)
        .head(1)["cell_type"][0]
    )
    most_common_cell_type = _top_ct
    print(f"Most common cell type: {most_common_cell_type!r}")
    return (most_common_cell_type,)


@app.cell
def _(atlas_csc, csc_uids, most_common_cell_type):
    adata_filtered = (
        atlas_csc.query()
        .where(f"cell_type = '{most_common_cell_type}'")
        .features(csc_uids[:20], "gene_expression")
        .to_anndata()
    )
    adata_filtered
    return (adata_filtered,)


@app.cell
def _(adata_filtered, mo, most_common_cell_type):
    mo.md(f"""
    **{adata_filtered.n_obs:,} `{most_common_cell_type}` cells × {adata_filtered.n_vars} genes**
    `X.nnz = {adata_filtered.X.nnz:,}`,
    `obs` columns: `{list(adata_filtered.obs.columns)}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Pattern 3: Streaming AnnData batches (CSR path, both atlases)

    `to_batches()` reconstructs a full AnnData per batch — obs DataFrame,
    var DataFrame, scipy CSR matrix. Useful for out-of-core processing.
    """)
    return


@app.cell
def _(atlas_csc, time):
    _t0 = time.perf_counter()
    _n_batches, _total_cells = 0, 0
    for _batch in atlas_csc.query().limit(5_000).to_batches(batch_size=512):
        _n_batches += 1
        _total_cells += _batch.n_obs
    _elapsed = time.perf_counter() - _t0
    print(
        f"Streamed {_total_cells:,} cells in {_n_batches} batches — "
        f"{_elapsed:.3f}s ({_elapsed / _n_batches * 1000:.1f} ms/batch)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Pattern 4: ML training dataloader (CellDataset, CSR path)

    `to_cell_dataset()` skips AnnData/scipy entirely and yields flat `SparseBatch`
    objects (CSR-style numpy arrays). Batches are cell-centric so the CSR layout
    is always used here regardless of whether CSC data is present.
    """)
    return


@app.cell
def _(atlas_csc, mo, time):
    from lancell.sampler import CellSampler as _CellSampler

    _ds = atlas_csc.query().limit(5_000).to_cell_dataset(
        feature_space="gene_expression",
        layer="counts",
        metadata_columns=["cell_type"],
    )
    _s = _CellSampler(_ds.groups_np, batch_size=256, shuffle=True, seed=42)

    _t0 = time.perf_counter()
    for _idx in _s:
        _ds.__getitems__(_idx)
    _elapsed = time.perf_counter() - _t0

    _first = _ds.__getitems__(next(iter(_CellSampler(_ds.groups_np, batch_size=256, shuffle=False))))

    mo.md(f"""
    **CellDataset**: {_ds.n_cells:,} cells × {_ds.n_features:,} features

    | | |
    |---|---|
    | Epoch ({len(_s)} batches) | `{_elapsed:.3f}s` |
    | ms / batch | `{_elapsed / len(_s) * 1000:.1f}` |
    | First batch — `indices` shape | `{_first.indices.shape}` |
    | First batch — `values` shape | `{_first.values.shape}` |
    | First batch — `offsets` shape | `{_first.offsets.shape}` |
    | First batch — `n_features` | `{_first.n_features:,}` |
    | Metadata keys | `{list(_first.metadata.keys()) if _first.metadata else None}` |

    For multi-worker training use `make_loader(dataset, sampler)` which sets
    `multiprocessing_context="spawn"` and `batch_sampler` automatically:

    ```python
    from lancell.dataloader import make_loader, sparse_to_dense_collate
    from lancell.sampler import CellSampler

    dataset = atlas.query().to_cell_dataset(metadata_columns=["cell_type"])
    sampler = CellSampler(dataset.groups_np, batch_size=256,
                          shuffle=True, seed=42, num_workers=4)
    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)
        for batch in make_loader(dataset, sampler):
            X = sparse_to_dense_collate(batch)["X"]  # (B, n_features) float32
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Storage Layout

    The zarr store structure shows how CSC and CSR data coexist per dataset group.
    """)
    return


@app.cell
def _(atlas_csc, atlas_csr, mo):
    from lancell.var_df import read_var_df as _read_var_df

    _g_csc = atlas_csc.list_datasets()["zarr_group"][0]
    _g_csr = atlas_csr.list_datasets()["zarr_group"][0]
    _var_csc = _read_var_df(atlas_csc._store, _g_csc)
    _var_csr = _read_var_df(atlas_csr._store, _g_csr)

    _csc_nnz_total = (_var_csc["csc_end"] - _var_csc["csc_start"]).sum()
    _n_expressed = (_var_csc["csc_end"] > _var_csc["csc_start"]).sum()

    mo.md(f"""
    ### CSC atlas — dataset group `{_g_csc[:16]}…`

    `var.parquet` columns: `{_var_csc.columns}`

    - Total CSC nnz: **{_csc_nnz_total:,}** *(equals CSR nnz — same non-zeros, different order)*
    - Genes with ≥1 expressing cell: **{_n_expressed:,}** / {len(_var_csc):,}

    ### CSR atlas — dataset group `{_g_csr[:16]}…`

    `var.parquet` columns: `{_var_csr.columns}`

    *No `csc_start` / `csc_end` columns — `atlas._has_csc()` returns `False`.*

    ---

    ```
    {{zarr_group}}/
    ├── csr/
    │   ├── indices              # feature indices, cell-contiguous (uint32, bitpacked+delta)
    │   └── layers/
    │       └── counts           # expression values, cell-contiguous (uint32)
    ├── csc/                     # CSC atlas only
    │   ├── indices              # zarr_row (cell) indices, feature-contiguous (uint32)
    │   └── layers/
    │       └── counts           # expression values, feature-contiguous
    ├── var.parquet              # local feature metadata
    │                            #   + csc_start / csc_end  (CSC atlas only)
    └── local_to_global_index.parquet   # local→registry remap
    ```

    `csc/indices[csc_start[f] : csc_end[f]]` = zarr_rows of cells expressing gene *f*.
    `atlas._has_csc(zarr_group)` checks whether `var.parquet` has non-null `csc_start`/`csc_end`.
    The feature-filtered reconstructor (`FeatureCSCReconstructor`) checks this per group and
    falls back to the CSR path for any group that lacks CSC data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    | Query pattern | Layout | I/O cost | When to use |
    |---|---|---|---|
    | `query().to_anndata()` | CSR | O(cells × avg_nnz) | Most cells, all features |
    | `query().features(few_uids).to_anndata()` | **CSC** | O(nnz for those genes) | Few features, many cells |
    | `query().features(uids).to_anndata()` on CSR atlas | CSR fallback | O(cells × avg_nnz) | (no CSC available) |
    | `to_batches()` | CSR | O(cells × avg_nnz) | Out-of-core / streaming |
    | `to_cell_dataset()` → ML training | CSR | O(cells × avg_nnz) | Always cell-centric |

    **When does CSC win?**

    1. You request **few features** relative to the total gene space (F ≪ G)
    2. You want **all or most cells** (not a tiny random subset — CSR overhead is then low anyway)
    3. Features are **moderately expressed** (nnz-per-feature ≪ total nnz)

    **When does CSC not help?**

    - Full-atlas queries with no feature filter (CSR is the only path)
    - ML training dataloaders (batches are always cell-centric)
    - Very small cell subsets where CSR read overhead is already negligible
    """)
    return


if __name__ == "__main__":
    app.run()
