# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "lancell",
#     "lancedb",
#     "obstore",
#     "polars",
#     "numpy",
#     "torch",
#     "tiledbsoma",
#     "tiledbsoma-ml",
#     "pyarrow",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import time

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    import tiledbsoma
    from tqdm.auto import tqdm

    return alt, mo, np, os, pl, tiledbsoma, time, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TileDB-SOMA vs lancell: Dataloader & Query Benchmark

    Compares ML data loading throughput and AnnData query performance between:

    | System | Dataset path |
    |--------|-------------|
    | **TileDB-SOMA** (`tiledbsoma-ml`) | `s3://epiblast-public/cellxgene_mouse_tiledb/` |
    | **lancell** (`CellDataset` + `CellSampler`) | `s3://epiblast-public/cellxgene_mouse_lancell/` |

    Both atlases contain the same ~44M cell mouse atlas from CellxGene Census.
    We benchmark random-shuffle streaming throughput for single-worker and
    multi-worker configurations.
    """)
    return


@app.cell
def _(mo):
    batch_size_slider = mo.ui.slider(
        start=128, stop=8192, step=128, value=512, label="Batch size"
    )
    n_cells_slider = mo.ui.slider(
        start=10_000,
        stop=1_000_000,
        step=10_000,
        value=50_000,
        label="Cells to stream",
    )
    mo.vstack([batch_size_slider, n_cells_slider])
    return batch_size_slider, n_cells_slider


@app.cell
def _(batch_size_slider, n_cells_slider):
    BATCH_SIZE = batch_size_slider.value
    N_CELLS = n_cells_slider.value
    SEED = 42
    return BATCH_SIZE, N_CELLS, SEED


@app.cell
def _(tiledbsoma, time):
    TILEDB_URI = "s3://epiblast-public/cellxgene_mouse_tiledb/"

    _t0 = time.perf_counter()
    experiment = tiledbsoma.Experiment.open(TILEDB_URI)
    tiledb_open_s = time.perf_counter() - _t0

    n_obs_tiledb = experiment.obs.count
    n_vars_tiledb = experiment.ms["RNA"].var.count
    print(
        f"TileDB-SOMA: {n_obs_tiledb:,} cells x {n_vars_tiledb:,} genes "
        f"(opened in {tiledb_open_s:.2f}s)"
    )
    return experiment, n_obs_tiledb, n_vars_tiledb, tiledb_open_s


@app.cell
def _(os, time):
    import obstore.store

    from lancell.atlas import RaggedAtlas
    from lancell.schema import LancellBaseSchema, SparseZarrPointer

    class CellObs(LancellBaseSchema):
        gene_expression: SparseZarrPointer | None = None
        assay: str | None = None
        cell_type: str | None = None
        disease: str | None = None
        sex: str | None = None
        tissue: str | None = None
        self_reported_ethnicity: str | None = None
        development_stage: str | None = None
        tissue_type: str | None = None
        tissue_general: str | None = None
        suspension_type: str | None = None
        donor_id: str | None = None
        is_primary_data: bool | None = None
        observation_joinid: str | None = None
        assay_ontology_term_id: str | None = None
        cell_type_ontology_term_id: str | None = None
        disease_ontology_term_id: str | None = None
        sex_ontology_term_id: str | None = None
        tissue_ontology_term_id: str | None = None
        self_reported_ethnicity_ontology_term_id: str | None = None
        development_stage_ontology_term_id: str | None = None
        tissue_general_ontology_term_id: str | None = None
        raw_sum: float | None = None
        nnz: int | None = None
        raw_mean_nnz: float | None = None
        raw_variance_nnz: float | None = None
        n_measured_vars: int | None = None

    LANCELL_DIR = "s3://epiblast-public/cellxgene_mouse_lancell/"

    store = obstore.store.S3Store.from_url(
        os.path.join(LANCELL_DIR, "zarr_store"),
        config={"skip_signature": True, "region": "us-east-2"},
    )
    _t0 = time.perf_counter()
    atlas = RaggedAtlas.checkout_latest(
        db_uri=os.path.join(LANCELL_DIR, "lance_db"),
        cell_schema=CellObs,
        store=store,
    )
    lancell_open_s = time.perf_counter() - _t0

    n_obs_lancell = atlas.cell_table.count_rows()
    n_vars_lancell = atlas._registry_tables["gene_expression"].count_rows()
    print(
        f"lancell: {n_obs_lancell:,} cells x {n_vars_lancell:,} genes "
        f"(opened in {lancell_open_s:.2f}s)"
    )
    return atlas, lancell_open_s, n_obs_lancell, n_vars_lancell


@app.cell(hide_code=True)
def _(
    lancell_open_s,
    mo,
    n_obs_lancell,
    n_obs_tiledb,
    n_vars_lancell,
    n_vars_tiledb,
    tiledb_open_s,
):
    mo.md(f"""
    ## Atlas Summary

    | | TileDB-SOMA | lancell |
    |---|---|---|
    | Cells | `{n_obs_tiledb:,}` | `{n_obs_lancell:,}` |
    | Genes | `{n_vars_tiledb:,}` | `{n_vars_lancell:,}` |
    | Open time | `{tiledb_open_s:.2f}s` | `{lancell_open_s:.2f}s` |
    """)
    return


@app.cell
def _(np, tiledbsoma, time, tqdm):
    def run_tiledb_epoch(experiment, n_cells, batch_size, seed, num_workers):
        """Stream one epoch through tiledbsoma-ml and return timing stats."""
        from tiledbsoma_ml import ExperimentDataset, experiment_dataloader

        obs_ids = (
            experiment.obs.read(column_names=["soma_joinid"])
            .concat()
            .to_pandas()["soma_joinid"]
            .values
        )
        rng = np.random.default_rng(seed)
        selected_ids = rng.choice(obs_ids, size=min(n_cells, len(obs_ids)), replace=False)
        selected_ids = np.sort(selected_ids)

        with experiment.axis_query(
            measurement_name="RNA",
            obs_query=tiledbsoma.AxisQuery(coords=(selected_ids,)),
        ) as query:
            ds = ExperimentDataset(
                query,
                layer_name="raw",
                obs_column_names=["soma_joinid", "cell_type"],
                batch_size=batch_size,
                shuffle=True,
                seed=seed,
                return_sparse_X=False,
                io_batch_size=batch_size * 8,
                shuffle_chunk_size=batch_size,
            )
            dl = experiment_dataloader(ds, num_workers=num_workers)

            warmup = 10
            batch_times = []
            total_cells = 0
            for step, (X_batch, obs_batch) in enumerate(tqdm(dl)):
                if step < warmup:
                    t_start = time.perf_counter()
                    continue
                t_end = time.perf_counter()
                batch_times.append(t_end - t_start)
                total_cells += X_batch.shape[0]
                t_start = time.perf_counter()

        total_s = sum(batch_times)
        return {
            "total_s": total_s,
            "total_cells": total_cells,
            "n_batches": len(batch_times),
            "med_batch_ms": float(np.median(batch_times)) * 1000 if batch_times else 0,
            "p95_batch_ms": float(np.percentile(batch_times, 95)) * 1000
            if batch_times
            else 0,
            "cells_per_s": total_cells / total_s if total_s > 0 else 0,
        }

    return


@app.cell
def _(np, time, tqdm):
    def run_lancell_epoch(atlas, n_cells, batch_size, seed, num_workers):
        """Stream one epoch through lancell CellDataset and return timing stats."""
        from lancell.dataloader import make_loader, sparse_to_dense_collate
        from lancell.sampler import CellSampler

        ds = atlas.query().limit(n_cells).to_cell_dataset(
            feature_space="gene_expression",
            layer="counts",
            metadata_columns=["cell_type"],
        )
        sampler = CellSampler(
            ds.groups_np,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            num_workers=num_workers,
        )
        loader = make_loader(ds, sampler)

        warmup = 10
        batch_times = []
        total_cells = 0
        for step, batch in enumerate(tqdm(loader)):
            dense = sparse_to_dense_collate(batch)
            if step < warmup:
                t_start = time.perf_counter()
                continue
            t_end = time.perf_counter()
            batch_times.append(t_end - t_start)
            total_cells += dense["X"].shape[0]
            t_start = time.perf_counter()

        total_s = sum(batch_times)
        return {
            "total_s": total_s,
            "total_cells": total_cells,
            "n_batches": len(batch_times),
            "med_batch_ms": float(np.median(batch_times)) * 1000 if batch_times else 0,
            "p95_batch_ms": float(np.percentile(batch_times, 95)) * 1000
            if batch_times
            else 0,
            "cells_per_s": total_cells / total_s if total_s > 0 else 0,
        }

    return (run_lancell_epoch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Benchmark 1 — Worker Scaling

    Sweep over worker counts (including 0 for single in-process) to compare
    I/O throughput and how each system scales with parallel data loading.
    Both systems use `multiprocessing_context="spawn"` for multi-worker runs.
    """)
    return


@app.cell
def _(mo):
    worker_counts_input = mo.ui.array(
        [
            mo.ui.checkbox(value=True, label="0 (in-process)"),
            mo.ui.checkbox(value=False, label="2 workers"),
            mo.ui.checkbox(value=True, label="4 workers"),
            mo.ui.checkbox(value=False, label="8 workers"),
        ]
    )
    mo.hstack([mo.md("**Worker counts to benchmark:**"), worker_counts_input])
    return (worker_counts_input,)


@app.cell
def _(worker_counts_input):
    ALL_WORKER_COUNTS = [0, 2, 4, 8]
    worker_counts = [
        w for w, checked in zip(ALL_WORKER_COUNTS, worker_counts_input.value) if checked
    ]
    print(f"Will benchmark: {worker_counts}")
    return (worker_counts,)


@app.cell
def _(
    BATCH_SIZE,
    N_CELLS,
    SEED,
    atlas,
    mo,
    pl,
    run_lancell_epoch,
    worker_counts,
):
    multi_rows = []
    for nw in worker_counts:
        label = f"{nw} workers" if nw > 0 else "0 (in-process)"
        print(f"\n--- {label} ---")

        # lancell
        print(f"  lancell     ...", end=" ", flush=True)
        res_l = run_lancell_epoch(atlas, N_CELLS, BATCH_SIZE, SEED, num_workers=nw)
        print(
            f"{res_l['total_s']:.2f}s, {res_l['cells_per_s']:,.0f} cells/s"
        )

        # TileDB
        print(f"  TileDB-SOMA ...", end=" ", flush=True)
        res_t = res_l # run_tiledb_epoch(experiment, N_CELLS, BATCH_SIZE, SEED, num_workers=nw)
        print(
            f"{res_t['total_s']:.2f}s, {res_t['cells_per_s']:,.0f} cells/s"
        )

        multi_rows.append({
            "workers": label,
            "tiledb_epoch_s": round(res_t["total_s"], 2),
            "tiledb_cells_per_s": round(res_t["cells_per_s"]),
            "tiledb_med_batch_ms": round(res_t["med_batch_ms"], 1),
            "lancell_epoch_s": round(res_l["total_s"], 2),
            "lancell_cells_per_s": round(res_l["cells_per_s"]),
            "lancell_med_batch_ms": round(res_l["med_batch_ms"], 1),
            "speedup": round(
                res_l["cells_per_s"] / res_t["cells_per_s"]
                if res_t["cells_per_s"] > 0
                else 0,
                2,
            ),
        })

    multi_worker_df = pl.DataFrame(multi_rows)

    mo.vstack([
        mo.md("### Multi-Worker Scaling"),
        multi_worker_df,
    ])
    return (multi_worker_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    The chart below shows cells/sec throughput for each system across
    worker counts. Higher is better.
    """)
    return


@app.cell
def _(alt, mo, multi_worker_df, pl):
    _tiledb_plot = multi_worker_df.select(
        pl.col("workers"),
        pl.col("tiledb_cells_per_s").alias("cells_per_s"),
        pl.lit("TileDB-SOMA").alias("system"),
    )
    _lancell_plot = multi_worker_df.select(
        pl.col("workers"),
        pl.col("lancell_cells_per_s").alias("cells_per_s"),
        pl.lit("lancell").alias("system"),
    )
    _plot_df = pl.concat([_tiledb_plot, _lancell_plot])

    _chart = (
        alt.Chart(_plot_df.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("workers:N", title="Workers"),
            y=alt.Y("cells_per_s:Q", title="Cells / sec"),
            color=alt.Color("system:N", title="System"),
            xOffset="system:N",
        )
        .properties(width=400, height=300, title="Dataloader Throughput")
    )

    mo.ui.altair_chart(_chart)
    return


@app.cell(hide_code=True)
def _(mo, multi_worker_df):
    best_lancell = multi_worker_df["lancell_cells_per_s"].max()
    best_tiledb = multi_worker_df["tiledb_cells_per_s"].max()
    peak_speedup = best_lancell / best_tiledb if best_tiledb > 0 else 0

    mo.md(f"""
    ### Dataloader Takeaways

    - **Peak lancell throughput**: `{best_lancell:,.0f}` cells/s
    - **Peak TileDB-SOMA throughput**: `{best_tiledb:,.0f}` cells/s
    - **Peak speedup (lancell / TileDB)**: `{peak_speedup:.2f}x`

    lancell's `CellSampler` bin-packs zarr groups across workers for reader
    cache locality, while `tiledbsoma-ml` uses shuffle-chunk + IO-batch
    pipelining. The architectures differ fundamentally:

    | | TileDB-SOMA | lancell |
    |---|---|---|
    | Dataset type | `IterableDataset` | Map-style (`__getitems__`) |
    | Shuffle | Chunk-level + IO-batch | Full per-group, epoch-level |
    | Worker dispatch | Round-robin by PyTorch | Bin-packed by `CellSampler` |
    | Sparse output | `scipy.sparse` or dense | `SparseBatch` (flat CSR arrays) |
    | I/O backend | TileDB core (C++) | zarr + async Python |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Benchmark 3 — AnnData Query Performance

    Beyond ML dataloading, both systems reconstruct **AnnData** objects from
    filtered queries. We compare three access patterns:

    | Pattern | Description |
    |---------|-------------|
    | **Cell-oriented** | Filter cells by `cell_type`, full expression matrix |
    | **Feature-oriented** | Subset of genes across a cell population |
    | **Combined** | Filter cells AND select specific genes |
    """)
    return


@app.cell
def _(atlas, pl):
    cell_type_counts = (
        atlas.query()
        .count(group_by="cell_type")
        .filter(pl.col("cell_type").is_not_null())
        .sort("count", descending=True)
    )
    cell_type_counts.head(10)
    return (cell_type_counts,)


@app.cell
def _(cell_type_counts, mo):
    cell_type_dropdown = mo.ui.dropdown(
        options=cell_type_counts["cell_type"].to_list(),
        value=cell_type_counts["cell_type"][0],
        label="Cell type to query",
    )
    n_query_cells_slider = mo.ui.slider(
        start=1_000, stop=50_000, step=1_000, value=10_000,
        label="Max cells per query",
    )
    n_genes_slider = mo.ui.slider(
        start=5, stop=100, step=5, value=20,
        label="Genes for feature query",
    )
    mo.vstack([cell_type_dropdown, n_query_cells_slider, n_genes_slider])
    return cell_type_dropdown, n_genes_slider, n_query_cells_slider


@app.cell
def _(atlas, experiment, n_genes_slider, np, pl):
    lancell_gene_reg = (
        atlas._registry_tables["gene_expression"]
        .search()
        .select(["uid", "feature_name"])
        .to_polars()
    )
    tiledb_var_names = set(
        experiment.ms["RNA"].var
        .read(column_names=["feature_name"])
        .concat()
        .to_pandas()["feature_name"]
        .values
    )
    common_genes = sorted(
        set(lancell_gene_reg["feature_name"].to_list()) & tiledb_var_names
    )
    rng = np.random.default_rng(42)
    selected_gene_names = list(
        rng.choice(
            common_genes,
            size=min(n_genes_slider.value, len(common_genes)),
            replace=False,
        )
    )
    selected_gene_uids = (
        lancell_gene_reg
        .filter(pl.col("feature_name").is_in(selected_gene_names))
        ["uid"]
        .to_list()
    )
    print(f"{len(common_genes):,} genes in common; selected {len(selected_gene_names)} for benchmarks")
    print(f"Sample genes: {selected_gene_names[:5]}")
    return selected_gene_names, selected_gene_uids


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3a. Cell-oriented query

    Filter cells by a metadata column and reconstruct the full expression
    matrix. lancell pushes the filter and limit into LanceDB; TileDB-SOMA
    requires a two-step ID lookup then `axis_query`.
    """)
    return


@app.cell
def _(
    atlas,
    cell_type_dropdown,
    experiment,
    mo,
    n_query_cells_slider,
    np,
    tiledbsoma,
    time,
):
    def _run():
        cell_type = cell_type_dropdown.value
        max_cells = n_query_cells_slider.value

        print(f"Cell-oriented: cell_type = '{cell_type}', limit = {max_cells:,}")

        lancell_times = []
        for i in range(3):
            t0 = time.perf_counter()
            adata_l = (
                atlas.query()
                .where(f"cell_type = '{cell_type}'")
                .limit(max_cells)
                .to_anndata()
            )
            lancell_times.append(time.perf_counter() - t0)
            print(f"  lancell     run {i+1}: {lancell_times[-1]:.2f}s -> {adata_l.shape}")

        tiledb_times = []
        for i in range(3):
            t0 = time.perf_counter()
            obs_ids = (
                experiment.obs.read(
                    column_names=["soma_joinid"],
                    value_filter=f"cell_type == '{cell_type}'",
                )
                .concat()
                .to_pandas()["soma_joinid"]
                .values
            )
            obs_ids = np.sort(obs_ids[:max_cells])
            with experiment.axis_query(
                measurement_name="RNA",
                obs_query=tiledbsoma.AxisQuery(coords=(obs_ids,)),
            ) as q:
                adata_t = q.to_anndata(X_name="raw")
            tiledb_times.append(time.perf_counter() - t0)
            print(f"  TileDB-SOMA run {i+1}: {tiledb_times[-1]:.2f}s -> {adata_t.shape}")

        l_med = float(np.median(lancell_times))
        t_med = float(np.median(tiledb_times))
        return {
            "query": "Cell-oriented",
            "lancell_shape": f"{adata_l.n_obs:,} x {adata_l.n_vars:,}",
            "lancell_s": round(l_med, 2),
            "tiledb_shape": f"{adata_t.n_obs:,} x {adata_t.n_vars:,}",
            "tiledb_s": round(t_med, 2),
            "speedup": round(t_med / l_med, 1) if l_med > 0 else 0,
        }

    cell_query_result = _run()

    mo.md(f"""
    | | lancell | TileDB-SOMA |
    |---|---|---|
    | Shape | `{cell_query_result['lancell_shape']}` | `{cell_query_result['tiledb_shape']}` |
    | Median time | `{cell_query_result['lancell_s']}s` | `{cell_query_result['tiledb_s']}s` |
    | **Speedup** | **{cell_query_result['speedup']}x** | — |
    """)
    return (cell_query_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3b. Feature-oriented query

    Select a subset of genes and retrieve their expression across a cell
    population. lancell uses `.features(uids)` to filter by global feature
    UID; TileDB-SOMA uses a `var_query` with `value_filter` on gene name.
    """)
    return


@app.cell
def _(
    atlas,
    experiment,
    mo,
    np,
    selected_gene_names,
    selected_gene_uids,
    tiledbsoma,
    time,
):
    max_cells = 1_000_000
    N_RUNS = 3
    feature_obs_ids = np.arange(max_cells, dtype=np.int64)
    gene_filter = "feature_name in ('" + "', '".join(selected_gene_names) + "')"
    print(gene_filter)

    print(f"Feature-oriented: {len(selected_gene_names)} genes, {max_cells:,} cells")

    lancell_times = []
    for _i in range(N_RUNS):
        t0 = time.perf_counter()
        adata_l = (
            atlas.query()
            .limit(max_cells)
            .features(selected_gene_uids, "gene_expression")
            .to_anndata()
        )
        lancell_times.append(time.perf_counter() - t0)
        print(f"  lancell     run {_i+1}: {lancell_times[-1]:.2f}s -> {adata_l.shape}")

    tiledb_times = []
    for _i in range(N_RUNS):
        t0 = time.perf_counter()
        with experiment.axis_query(
            measurement_name="RNA",
            obs_query=tiledbsoma.AxisQuery(coords=(feature_obs_ids,)),
            var_query=tiledbsoma.AxisQuery(value_filter=gene_filter.replace("(", "[").replace(")", "]")),
        ) as q:
            adata_t = q.to_anndata(X_name="raw")
        tiledb_times.append(time.perf_counter() - t0)
        print(f"  TileDB-SOMA run {_i+1}: {tiledb_times[-1]:.2f}s -> {adata_t.shape}")

    l_med = float(np.median(lancell_times))
    t_med = float(np.median(tiledb_times))
    feat_query_result = {
        "query": "Feature-oriented",
        "lancell_shape": f"{adata_l.n_obs:,} x {adata_l.n_vars:,}",
        "lancell_s": round(l_med, 2),
        "tiledb_shape": f"{adata_t.n_obs:,} x {adata_t.n_vars:,}",
        "tiledb_s": round(t_med, 2),
        "speedup": round(t_med / l_med, 1) if l_med > 0 else 0,
    }

    mo.md(f"""
    | | lancell | TileDB-SOMA |
    |---|---|---|
    | Shape | `{adata_l.n_obs:,} x {adata_l.n_vars:,}` | `{adata_t.n_obs:,} x {adata_t.n_vars:,}` |
    | Median time | `{l_med:.2f}s` | `{t_med:.2f}s` |
    | **Speedup** | **{feat_query_result['speedup']}x** | — |
    """)
    return (feat_query_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3c. Combined query

    Filter cells by metadata AND select specific genes. This is the most
    common bioinformatics pattern: "give me marker gene expression for
    a specific cell population."
    """)
    return


@app.cell
def _(
    atlas,
    cell_type_dropdown,
    experiment,
    mo,
    n_query_cells_slider,
    np,
    selected_gene_names,
    selected_gene_uids,
    tiledbsoma,
    time,
):
    def _run():
        cell_type = cell_type_dropdown.value
        max_cells = n_query_cells_slider.value
        N_RUNS = 3
        gene_filter = "feature_name in ('" + "', '".join(selected_gene_names) + "')"

        print(f"Combined: cell_type = '{cell_type}', {len(selected_gene_names)} genes, limit = {max_cells:,}")

        lancell_times = []
        for _i in range(N_RUNS):
            t0 = time.perf_counter()
            adata_l = (
                atlas.query()
                .where(f"cell_type = '{cell_type}'")
                .limit(max_cells)
                .features(selected_gene_uids, "gene_expression")
                .to_anndata()
            )
            lancell_times.append(time.perf_counter() - t0)
            print(f"  lancell     run {_i+1}: {lancell_times[-1]:.2f}s -> {adata_l.shape}")

        tiledb_times = []
        for _i in range(N_RUNS):
            t0 = time.perf_counter()
            obs_ids = (
                experiment.obs.read(
                    column_names=["soma_joinid"],
                    value_filter=f"cell_type == '{cell_type}'",
                )
                .concat()
                .to_pandas()["soma_joinid"]
                .values
            )
            obs_ids = np.sort(obs_ids[:max_cells])
            with experiment.axis_query(
                measurement_name="RNA",
                obs_query=tiledbsoma.AxisQuery(coords=(obs_ids,)),
                var_query=tiledbsoma.AxisQuery(value_filter=gene_filter.replace("(", "[").replace(")", "]")),
            ) as q:
                adata_t = q.to_anndata(X_name="raw")
            tiledb_times.append(time.perf_counter() - t0)
            print(f"  TileDB-SOMA run {_i+1}: {tiledb_times[-1]:.2f}s -> {adata_t.shape}")

        return lancell_times, tiledb_times, adata_l, adata_t

    _lancell_times, _tiledb_times, _adata_l, _adata_t = _run()

    _l_med = float(np.median(_lancell_times))
    _t_med = float(np.median(_tiledb_times))
    comb_query_result = {
        "query": "Combined",
        "lancell_shape": f"{_adata_l.n_obs:,} x {_adata_l.n_vars:,}",
        "lancell_s": round(_l_med, 2),
        "tiledb_shape": f"{_adata_t.n_obs:,} x {_adata_t.n_vars:,}",
        "tiledb_s": round(_t_med, 2),
        "speedup": round(_t_med / _l_med, 1) if _l_med > 0 else 0,
    }

    mo.md(f"""
    | | lancell | TileDB-SOMA |
    |---|---|---|
    | Shape | `{_adata_l.n_obs:,} x {_adata_l.n_vars:,}` | `{_adata_t.n_obs:,} x {_adata_t.n_vars:,}` |
    | Median time | `{_l_med:.2f}s` | `{_t_med:.2f}s` |
    | **Speedup** | **{comb_query_result['speedup']}x** | — |
    """)
    return (comb_query_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Query Benchmark Summary
    """)
    return


@app.cell
def _(alt, cell_query_result, comb_query_result, feat_query_result, mo, pl):
    query_df = pl.DataFrame([cell_query_result, feat_query_result, comb_query_result])

    _tiledb_plot = query_df.select(
        pl.col("query"),
        pl.col("tiledb_s").alias("seconds"),
        pl.lit("TileDB-SOMA").alias("system"),
    )
    _lancell_plot = query_df.select(
        pl.col("query"),
        pl.col("lancell_s").alias("seconds"),
        pl.lit("lancell").alias("system"),
    )
    _chart_df = pl.concat([_tiledb_plot, _lancell_plot])

    _chart = (
        alt.Chart(_chart_df.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X(
                "query:N",
                title="Query Pattern",
                sort=["Cell-oriented", "Feature-oriented", "Combined"],
            ),
            y=alt.Y("seconds:Q", title="Time (seconds)"),
            color=alt.Color("system:N", title="System"),
            xOffset="system:N",
        )
        .properties(width=400, height=300, title="Query -> AnnData Latency")
    )

    mo.vstack([query_df, mo.ui.altair_chart(_chart)])
    return


@app.cell(hide_code=True)
def _(cell_query_result, comb_query_result, feat_query_result, mo):
    avg_speedup = (
        cell_query_result["speedup"]
        + feat_query_result["speedup"]
        + comb_query_result["speedup"]
    ) / 3

    mo.md(f"""
    ### Query Takeaways

    - **Cell-oriented speedup**: `{cell_query_result['speedup']}x`
    - **Feature-oriented speedup**: `{feat_query_result['speedup']}x`
    - **Combined speedup**: `{comb_query_result['speedup']}x`
    - **Average speedup**: `{avg_speedup:.1f}x`

    lancell's query API provides the same data access patterns as TileDB-SOMA
    while reconstructing AnnData objects faster. Key differences:

    | | TileDB-SOMA | lancell |
    |---|---|---|
    | Cell filter | `AxisQuery(value_filter=...)` | `.where(...)` (SQL push-down) |
    | Cell limit | Manual: read IDs, slice, coords | `.limit(n)` (built-in) |
    | Feature filter | `AxisQuery(value_filter=...)` on var | `.features(uids, space)` |
    | Reconstruction | Single `to_anndata()` path | Reconstructor per storage layout |
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
