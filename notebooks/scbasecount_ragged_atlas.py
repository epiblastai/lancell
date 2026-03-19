# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "lancell",
#     "polars",
#     "anndata",
#     "numba",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Exploring a lancell Atlas

    This notebook walks through the core features of **lancell** using a
    73 M-cell atlas built from [scBaseCount](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11885935/)
    data spanning *Homo sapiens* and *C. elegans*.

    We'll cover:

    1. Opening & versioning
    2. Metadata queries and filtering
    3. Working with **ragged feature spaces** (union vs. intersection joins)
    4. Feature selection via registry lookup
    5. AnnData / batch reconstruction
    6. ML training with `CellDataset` + `CellSampler`
    7. Differential expression with `dex`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Open the atlas

    A lancell atlas is a LanceDB database (cell metadata + feature registries)
    paired with a zarr store (expression matrices). We open a read-only
    **snapshot** — a point-in-time view that guarantees reproducible queries
    even while new data is being ingested.
    """)
    return


@app.cell
def _():
    import polars as pl
    from tqdm.auto import tqdm

    from lancell.atlas import RaggedAtlas
    from lancell.group_specs import (
        ArraySpec,
        DTypeKind,
        LayersSpec,
        PointerKind,
        ZarrGroupSpec,
        register_spec,
    )
    from lancell.reconstruction import SparseCSRReconstructor

    GENEFULL_EXPRESSION_SPEC = ZarrGroupSpec(
        feature_space="genefull_expression",
        pointer_kind=PointerKind.SPARSE,
        has_var_df=True,
        required_arrays=[
            ArraySpec(array_name="csr/indices", ndim=1, dtype_kind=DTypeKind.UNSIGNED_INTEGER),
        ],
        layers=LayersSpec(
            prefix="csr",
            uniform_shape=True,
            match_shape_of="csr/indices",
            required=["Unique"],
            allowed=["Unique", "UniqueAndMult-EM", "UniqueAndMult-Uniform"],
        ),
        reconstructor=SparseCSRReconstructor(),
    )
    register_spec(GENEFULL_EXPRESSION_SPEC)
    return RaggedAtlas, pl, tqdm


@app.cell
def _():
    db_uri = "s3://epiblast-public/scbasecount_mini_lancell/lance_db"
    S3_KWARGS = {"config": {"skip_signature": True, "region": "us-east-2"}}
    return S3_KWARGS, db_uri


@app.cell
def _(RaggedAtlas, db_uri):
    versions = RaggedAtlas.list_versions(db_uri)
    versions
    return


@app.cell
def _(RaggedAtlas, S3_KWARGS, db_uri):
    atlas = RaggedAtlas.checkout_latest(
        db_uri=db_uri,
        store_kwargs=S3_KWARGS,
    )
    return (atlas,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Explore metadata

    Metadata queries hit LanceDB only — no zarr I/O — so they're fast even
    at 73 M cells. The fluent `atlas.query()` API returns an `AtlasQuery`
    builder; terminal methods like `.count()`, `.to_polars()`, and
    `.to_anndata()` execute the query.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Total cell count

    `.count()` fetches a single cheap column and counts rows — no
    expression data is touched.
    """)
    return


@app.cell
def _(atlas):
    total_cells = atlas.query().count()
    total_cells
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Cells per dataset

    Pass `group_by=` to `.count()` to get per-group tallies without
    materialising the full cell table.
    """)
    return


@app.cell
def _(atlas):
    counts_by_dataset = atlas.query().count(group_by="dataset_uid")
    counts_by_dataset
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Dataset-level metadata

    `list_datasets()` returns a Polars DataFrame of dataset-level
    metadata (one row per ingested dataset). This is useful for
    discovering what's in the atlas before querying cells.
    """)
    return


@app.cell
def _(atlas):
    datasets = atlas.list_datasets()
    datasets
    return (datasets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Filtering cells with `.where()`

    LanceDB SQL syntax lets us push filters down before any data is
    materialised. Here we grab a small sample from a single experiment.
    """)
    return


@app.cell
def _(atlas, datasets):
    sample_srx = datasets["srx_accession"][0]

    filtered_cells = (
        atlas.query()
        .where(f"srx_accession = '{sample_srx}'")
        .select(["srx_accession", "cell_barcode", "gene_count_unique", "umi_count_unique"])
        .limit(10)
        .to_polars()
    )
    filtered_cells
    return (sample_srx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Ragged feature spaces

    Different datasets in this atlas measure **different subsets of genes**.
    lancell tracks a global feature registry and per-dataset *feature layouts*
    that map local indices to global positions. When querying across datasets,
    you choose how to reconcile these ragged feature sets:

    - **`"union"`** (default) — include every gene measured by *any* dataset.
      Cells that didn't measure a gene get zeros in that column.
    - **`"intersection"`** — include only genes measured by *every* dataset
      in the query. The resulting matrix is narrower but fully observed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Gene registry

    Each feature space has a **global feature registry** — a LanceDB table
    that assigns every unique feature (gene) a stable `global_index`. The
    registry grows as new datasets introduce previously unseen genes but
    existing indices never change.
    """)
    return


@app.cell
def _(atlas, pl):
    gene_registry = (
        atlas._registry_tables["genefull_expression"]
        .search()
        .to_polars()
        .sort("global_index")
    )
    n_genes = gene_registry.height
    mo_text = f"**{n_genes:,}** unique genes in the global registry"

    # show organism breakdown
    organism_counts = gene_registry.group_by("organism").agg(pl.len().alias("n_genes")).sort("organism")

    gene_registry
    return gene_registry, mo_text, organism_counts


@app.cell
def _(mo, mo_text, organism_counts):
    mo.vstack([
        mo.md(mo_text),
        organism_counts,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Union vs. intersection join

    To see the difference we need cells from datasets with **different
    feature sets** — e.g. one human and one *C. elegans* experiment.

    We first pick one dataset per organism, then use `.where()` to
    scope the query to just those two datasets. `.balanced_limit(5000,
    "dataset_uid")` draws an equal number of cells from each dataset
    so neither dominates the result.
    """)
    return


@app.cell
def _(datasets, pl):
    # Pick the smallest dataset per organism so the demo stays fast
    ragged_pair = (
        datasets
        .filter(pl.col("organism").is_not_null())
        .group_by("organism")
        .first()
        .select("uid", "organism", "n_cells")
    )
    ragged_uids = ragged_pair["uid"].to_list()
    ragged_where = f"dataset_uid IN ('{ragged_uids[0]}', '{ragged_uids[1]}')"
    ragged_pair
    return (ragged_where,)


@app.cell
def _(atlas, ragged_where):
    adata_union = (
        atlas.query()
        .where(ragged_where)
        .balanced_limit(5000, "dataset_uid")
        .feature_spaces("genefull_expression")
        .feature_join("union")
        .to_anndata()
    )
    adata_union
    return (adata_union,)


@app.cell
def _(atlas, ragged_where):
    adata_intersection = (
        atlas.query()
        .where(ragged_where)
        .balanced_limit(5000, "dataset_uid")
        .feature_spaces("genefull_expression")
        .feature_join("intersection")
        .to_anndata()
    )
    adata_intersection
    return (adata_intersection,)


@app.cell(hide_code=True)
def _(adata_intersection, adata_union, mo):
    mo.md(f"""
    | Join mode | Genes (vars) | Cells (obs) |
    |-----------|-------------|-------------|
    | **union** | {adata_union.n_vars:,} | {adata_union.n_obs:,} |
    | **intersection** | {adata_intersection.n_vars:,} | {adata_intersection.n_obs:,} |

    The union matrix is wider — it includes every gene measured by *either*
    organism, with structural zeros where a dataset didn't measure a gene.
    The intersection matrix only keeps genes measured by *both* datasets,
    giving a narrower but fully-observed matrix.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Feature selection

    To query specific genes, look them up in the registry by name to get
    their UIDs, then pass those UIDs to `.features()`. The reconstructed
    AnnData will contain only the requested columns — useful when you need
    a handful of marker genes instead of the full transcriptome.
    """)
    return


@app.cell
def _(gene_registry, pl):
    # Search for some well-known marker genes by name (case-insensitive partial match)
    marker_names = ["ACTB", "GAPDH", "MALAT1", "TP53", "BRCA1"]

    marker_genes = gene_registry.filter(
        pl.col("gene_name").is_in(marker_names)
    )
    marker_genes
    return (marker_genes,)


@app.cell
def _(atlas, marker_genes):
    marker_uids = marker_genes["uid"].to_list()

    adata_markers = (
        atlas.query()
        .feature_spaces("genefull_expression")
        .features(marker_uids, feature_space="genefull_expression")
        .limit(10_000)
        .to_anndata()
    )
    adata_markers
    return (adata_markers,)


@app.cell(hide_code=True)
def _(adata_markers, mo):
    mo.md(f"""
    Reconstructed AnnData with **{adata_markers.n_vars}** selected genes
    and **{adata_markers.n_obs:,}** cells. The `.var` DataFrame shows the
    gene metadata from the registry:
    """)
    return


@app.cell
def _(adata_markers):
    adata_markers.var
    return


@app.cell
def _(adata_markers):
    adata_markers.X
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Streaming AnnData batches

    For workloads that don't need all cells in memory at once —
    e.g. iterative QC, per-batch normalization, or writing to disk —
    `.to_batches()` streams AnnData objects one chunk at a time.
    Each batch is a fully-formed AnnData with `.obs`, `.var`, and `.X`.
    """)
    return


@app.cell
def _(atlas, sample_srx):
    batch_iter = (
        atlas.query()
        .where(f"srx_accession = '{sample_srx}'")
        .feature_spaces("genefull_expression")
        .limit(2048)
        .to_batches(batch_size=1024)
    )

    for i, batch_adata in enumerate(batch_iter):
        print(f"Batch {i}: {batch_adata.n_obs} cells × {batch_adata.n_vars} genes")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. ML training with `CellDataset` + `CellSampler`

    lancell provides a purpose-built dataloader pipeline for training on
    sparse single-cell data:

    ```
    AtlasQuery → CellDataset + CellSampler → DataLoader → SparseBatch → collate_fn → GPU
    ```

    - **`CellDataset`** — a `torch.utils.data.Dataset` that maps cell
      indices to sparse zarr reads. It owns no batching logic.
    - **`CellSampler`** — a `torch.utils.data.Sampler` that plans which
      cells go in each batch. It groups cells by zarr group for I/O
      locality and bin-packs groups across DataLoader workers so each
      worker warms a small, stable reader cache.
    - **`make_loader`** — a convenience function that wires a dataset and
      sampler into a standard `DataLoader` with the right collation.
    - **`sparse_to_dense_collate`** — converts the list of `SparseBatch`
      objects from the loader into a dense `(batch_size, n_features)` tensor.
    """)
    return


@app.cell
def _():
    import torch

    from lancell.dataloader import make_loader, sparse_to_dense_collate
    from lancell.sampler import CellSampler

    return CellSampler, make_loader, sparse_to_dense_collate, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Create a `CellDataset`

    `.to_cell_dataset()` materialises the query into a `CellDataset`.
    You specify which feature space and layer to read — here we use the
    `"Unique"` layer (uniquely-mapped UMI counts) from the
    `"genefull_expression"` feature space.
    """)
    return


@app.cell
def _(atlas):
    dataset = (
        atlas.query()
        .feature_spaces("genefull_expression")
        .limit(500_000)
        .to_cell_dataset(
            feature_space="genefull_expression",
            layer="Unique",
        )
    )
    print(f"CellDataset: {dataset.n_cells:,} cells, {dataset.n_features:,} features")
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Configure the `CellSampler`

    The sampler takes `dataset.groups_np` — an integer array that maps
    each cell to its zarr group. At construction it bin-packs groups
    across workers (largest-first greedy) so each worker touches a
    minimal subset of zarr groups, keeping reader caches warm.

    `drop_last=True` discards the trailing incomplete batch so every
    batch has exactly `batch_size` cells — convenient for fixed-size
    GPU kernels.
    """)
    return


@app.cell
def _(CellSampler, dataset):
    BATCH_SIZE = 1024
    NUM_WORKERS = 4  # 0 for in-process (notebook-friendly); use 4+ in real training

    sampler = CellSampler(
        dataset.groups_np,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    print(f"Sampler: {len(sampler)} batches per epoch")
    return (sampler,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Run one epoch

    `make_loader` wires the dataset and sampler into a standard
    `torch.utils.data.DataLoader`. Each iteration yields a list of
    `SparseBatch` objects (one per cell in the batch), which
    `sparse_to_dense_collate` stacks into a dense
    `(batch_size, n_features)` float32 tensor.

    Call `sampler.set_epoch(epoch)` before each epoch to get a fresh
    shuffle while keeping reproducibility (the RNG seed is
    `seed + epoch`).
    """)
    return


@app.cell
def _(dataset, make_loader, sampler, sparse_to_dense_collate, torch, tqdm):
    sampler.set_epoch(0)
    loader = make_loader(dataset, sampler)

    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
        result = sparse_to_dense_collate(batch)
        X = result["X"]  # (batch_size, n_features) float32 tensor

        if batch_idx == 0:
            print(f"First batch X shape: {X.shape}, dtype: {X.dtype}")
            print(f"  Non-zero fraction: {(X != 0).float().mean():.4f}")
            print(f"  Row sums (UMI counts): min={X.sum(1).min():.0f}, "
                  f"median={X.sum(1).median():.0f}, max={X.sum(1).max():.0f}")

    print(f"\nProcessed {batch_idx + 1} batches, last X shape: {X.shape}")
    _ = torch  # keep import alive
    return (X,)


@app.cell
def _(X):
    X.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    That's the core training loop. In a real training script you'd wrap
    this in an epoch loop and feed `X` into your model.

    lancell also provides a `BalancedCellSampler` that draws equal cells
    per category (e.g. cell type or dataset) each epoch — useful when
    dataset sizes span orders of magnitude and you want more equal
    representation during training.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. Differential expression with `dex`

    lancell includes a built-in differential expression module that
    compares two groups of cells directly from the atlas. It auto-dispatches
    **Mann-Whitney U** or **Welch's t-test** on any feature space.

    `dex()` follows the scanpy `rank_genes_groups` pattern: specify a
    `groupby` column, a list of `target` group values, and a `control`
    group value. It handles per-group loading, feature alignment,
    normalization, and multiple-testing correction internally.
    """)
    return


@app.cell
def _():
    from lancell.dex import dex

    return (dex,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Discover comparison groups

    We look up datasets by tissue and pick one `dataset_uid` per tissue.
    The cell table stores `dataset_uid`, so that's the column we group by.
    """)
    return


@app.cell
def _(datasets, pl):
    tissue_datasets = (
        datasets
        .filter(pl.col("tissue").is_not_null())
        .group_by("tissue")
        .agg(
            pl.col("uid").first().alias("dataset_uid"),
            pl.col("n_cells").first().alias("n_cells"),
        )
        .sort("n_cells", descending=True)
    )
    tissue_datasets
    return (tissue_datasets,)


@app.cell
def _(tissue_datasets):
    target_uid = tissue_datasets["dataset_uid"][0]
    control_uid = tissue_datasets["dataset_uid"][2]
    target_tissue = tissue_datasets["tissue"][0]
    control_tissue = tissue_datasets["tissue"][2]

    print(f"Target:  {target_tissue} (dataset_uid={target_uid})")
    print(f"Control: {control_tissue} (dataset_uid={control_uid})")
    return control_tissue, control_uid, target_tissue, target_uid


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Run `dex()`

    We compare cells from two datasets (one per tissue), grouping by
    `dataset_uid` and limiting each side to 1 000 cells with
    `max_records` so the demo runs quickly.
    """)
    return


@app.cell
def _(atlas, control_uid, dex, target_uid):
    dex_result = dex(
        atlas,
        groupby="dataset_uid",
        target=[target_uid],
        control=control_uid,
        feature_space="genefull_expression",
        test="mwu",
        max_records=1_000,
    )
    dex_result.sort("fdr")
    return (dex_result,)


@app.cell(hide_code=True)
def _(control_tissue, dex_result, mo, pl, target_tissue):
    _sig = dex_result.filter(pl.col("fdr") < 0.05).height
    mo.md(f"""
    **{target_tissue}** vs **{control_tissue}**: **{_sig:,}** significant
    genes (FDR < 0.05) out of **{dex_result.height:,}** tested.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Top differentially expressed genes

    The `feature` column contains internal UIDs. Use
    `atlas.join_feature_metadata()` to bring in human-readable
    gene names and Ensembl IDs from the feature registry.
    """)
    return


@app.cell
def _(atlas, dex_result):
    dex_annotated = atlas.join_feature_metadata(
        dex_result,
        feature_space="genefull_expression",
        columns=["gene_name", "gene_id"],
    )
    dex_annotated.sort("fdr").head(20)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
