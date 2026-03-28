# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "lancell",
#     "polars",
#     "anndata",
#     "mudata",
#     "numpy",
#     "matplotlib",
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
    # Exploring the Multimodal Perturbation Atlas

    This notebook walks through the core features of **lancell** using a
    perturbation atlas built from multiple GEO datasets

    We'll cover:

    1. Opening, optimizing & snapshotting
    2. Metadata queries and filtering
    3. Feature registry exploration
    4. AnnData reconstruction
    5. MuData reconstruction (multimodal queries)
    6. Unified multimodal queries with `to_multimodal()`
    7. Perturbation-aware queries (`PerturbationQuery`)
    8. ML training with `CellDataset` + `CellSampler`
    9. Chromatin accessibility (ATAC-seq fragments)
    10. Image features and tiles (Cell Painting)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Open, optimize & snapshot

    After ingestion, the atlas needs to be **optimized** (compact Lance fragments
    and assign global feature indices) and **snapshotted** (record a consistent
    point-in-time version). Run this section once, then comment it out and use
    `checkout_latest` for all subsequent work.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import obstore.store
    import polars as pl
    import matplotlib.pyplot as plt

    from lancell.atlas import RaggedAtlas
    from lancell_examples.multimodal_perturbation_atlas.atlas import PerturbationAtlas
    from lancell_examples.multimodal_perturbation_atlas.schema import CellIndex

    ATLAS_DIR = Path.home() / "multimodal_perturbation_atlas"
    # ATLAS_DIR = Path("/tmp/atlas/cpg0021_test")
    DB_URI = str(ATLAS_DIR / "lance_db")
    ZARR_PATH = str(ATLAS_DIR / "zarr_store")
    return (
        CellIndex,
        DB_URI,
        PerturbationAtlas,
        RaggedAtlas,
        ZARR_PATH,
        obstore,
        pl,
        plt,
    )


@app.cell
def _(RaggedAtlas):
    _atlas_restore = RaggedAtlas.restore(
      db_uri="/home/ubuntu/multimodal_perturbation_atlas/lance_db",
      version=3,  # or whatever your snapshot version is
    )
    return


@app.cell
def _(CellIndex, DB_URI, RaggedAtlas, ZARR_PATH, obstore):
    # # --- Run once after ingestion, then comment out ---
    _store = obstore.store.LocalStore(ZARR_PATH)
    _atlas_rw = RaggedAtlas.open(
        db_uri=DB_URI,
        cell_table_name="cells",
        cell_schema=CellIndex,
        store=_store,
    )
    print(_atlas_rw.cell_table.count_rows())
    # _atlas_rw.optimize()
    # version = _atlas_rw.snapshot()
    # print(f"Optimized and snapshotted: version {version}")
    return


@app.cell
def _(DB_URI, RaggedAtlas):
    versions = RaggedAtlas.list_versions(DB_URI)
    versions
    return


@app.cell
def _(DB_URI, PerturbationAtlas):
    # --- Use this for read-only access after snapshotting ---
    # PerturbationAtlas inherits from RaggedAtlas and adds perturbation-aware
    # query methods. checkout/checkout_latest/restore all work unchanged.
    atlas_rw = PerturbationAtlas.checkout_latest(db_uri=DB_URI)
    atlas_rw
    return (atlas_rw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Explore metadata

    Metadata queries hit LanceDB only — no zarr I/O — so they're fast.
    The fluent `atlas.query()` API returns a `PerturbationQuery` builder
    (extends `AtlasQuery` with perturbation-specific methods); terminal
    methods like `.count()`, `.to_polars()`, and `.to_anndata()` execute
    the query.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Total cell count
    """)
    return


@app.cell
def _(atlas_rw):
    total_cells = atlas_rw.query().count()
    total_cells
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Cells per dataset
    """)
    return


@app.cell
def _(atlas_rw):
    counts_by_dataset = atlas_rw.query().count(group_by="dataset_uid")
    counts_by_dataset
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Dataset-level metadata
    """)
    return


@app.cell
def _(atlas_rw):
    datasets = atlas_rw.list_datasets()
    datasets
    return (datasets,)


@app.cell
def _(atlas_rw):
    atlas_rw.db.list_tables()
    return


@app.cell
def _(atlas_rw):
    # PerturbationAtlas provides cached_property accessors for FK tables
    atlas_rw.publications_table.search().to_pandas()
    return


@app.cell
def _(atlas_rw):
    atlas_rw.genetic_perturbations_table.search().where("intended_gene_name == 'PPARG'").to_pandas()
    return


@app.cell
def _():
    # atlas_rw.small_molecules_table.search().to_pandas()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Filtering cells with `.where()`

    LanceDB SQL syntax pushes filters down before any data is materialised.
    Here we grab a small sample of HepG2 cells.
    """)
    return


@app.cell
def _(atlas_rw, datasets):
    sample_uid = datasets["uid"][0]

    filtered_cells = (
        atlas_rw.query()
        .where(f"dataset_uid = '{sample_uid}'")
        .select(["organism", "cell_line", "assay", "is_negative_control"])
        .limit(10)
        .to_polars()
    )
    filtered_cells
    return (sample_uid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Feature registry

    Each feature space has a **global feature registry** — a LanceDB table
    that assigns every unique feature a stable `global_index`. The registry
    grows as new datasets introduce previously unseen genes but existing
    indices never change.
    """)
    return


@app.cell
def _(atlas_rw):
    gene_registry = (
        atlas_rw._registry_tables["gene_expression"]
        .search()
        .to_polars()
        .sort("global_index")
    )
    n_genes = gene_registry.height
    print(f"{n_genes:,} genes in the global registry")
    gene_registry.head(10)
    return (gene_registry,)


@app.cell
def _(atlas_rw):
    protein_registry = (
        atlas_rw._registry_tables["protein_abundance"]
        .search()
        .to_polars()
        .sort("global_index")
    )
    n_proteins = protein_registry.height
    print(f"{n_proteins:,} proteins in the global registry")
    protein_registry.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Feature selection

    Look up specific genes by name, then pass their UIDs to `.features()`
    for targeted reconstruction.
    """)
    return


@app.cell
def _(gene_registry, pl):
    marker_names = ["ACTB", "GAPDH", "MALAT1", "TP53", "BRCA1"]

    marker_genes = gene_registry.filter(pl.col("gene_name").is_in(marker_names))
    marker_genes
    return (marker_genes,)


@app.cell
def _(atlas_rw, marker_genes):
    marker_uids = marker_genes["uid"].to_list()

    adata_markers = (
        atlas_rw.query()
        .feature_spaces("gene_expression")
        .features(marker_uids, feature_space="gene_expression")
        .limit(10_000)
        .to_anndata()
    )
    adata_markers
    return (adata_markers,)


@app.cell
def _(adata_markers):
    adata_markers.var
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. AnnData reconstruction

    Reconstruct a full AnnData for a single dataset. This reads the sparse
    expression data from zarr and joins it with obs metadata from LanceDB.
    """)
    return


@app.cell
def _(atlas_rw, sample_uid):
    adata_sample = (
        atlas_rw.query()
        .where(f"dataset_uid = '{sample_uid}'")
        .feature_spaces("gene_expression")
        .limit(5_000)
        .to_anndata()
    )
    adata_sample
    return (adata_sample,)


@app.cell
def _(adata_sample):
    adata_sample.obs[["organism", "cell_line", "assay", "is_negative_control", "perturbation_uids"]].head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. MuData reconstruction

    When cells have measurements across multiple feature spaces (e.g. gene
    expression + protein abundance from CITE-seq or ECCITE-seq), `.to_mudata()`
    reconstructs a **MuData** object with one modality per feature space.

    Note: `.to_mudata()` only includes AnnData-compatible modalities. For
    heterogeneous modalities (fragments, image tiles), use `.to_multimodal()`
    instead (section 6).
    """)
    return


@app.cell
def _(atlas_rw):
    mdata = (
        atlas_rw.query()
        .feature_spaces("gene_expression", "protein_abundance")
        .where("cell_line == 'THP-1'")
        .limit(5_000)
        .to_mudata()
    )
    mdata
    return (mdata,)


@app.cell
def _(mdata):
    # Each modality is an AnnData keyed by feature space name
    for mod_name, mod_adata in mdata.mod.items():
        print(f"  {mod_name}: {mod_adata.n_obs:,} cells × {mod_adata.n_vars:,} features")
    return


@app.cell
def _(mdata):
    # Access individual modalities — obs is shared across modalities
    mdata["gene_expression"].var.head()
    return


@app.cell
def _(mdata):
    mdata["protein_abundance"].var.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Unified multimodal queries with `to_multimodal()`

    `.to_multimodal()` returns a `MultimodalResult` — a unified container
    where each modality is stored in its **native format**:

    | Modality | Format | Example |
    |----------|--------|---------|
    | Gene expression | AnnData | Sparse cell x gene matrix |
    | Protein abundance | AnnData | Dense cell x protein matrix |
    | Image features | AnnData | Dense cell x feature matrix |
    | Chromatin accessibility | FragmentResult | Raw genomic intervals with CSR offsets |
    | Image tiles | ndarray | 4D array (cells x channels x H x W) |

    The container has a shared `obs` DataFrame for all cells, and per-modality
    `present` masks showing which cells have each modality.
    """)
    return


@app.cell
def _(atlas_rw):
    # Query all modalities at once
    multimodal = (
        atlas_rw.query()
        .balanced_limit(10_000, "dataset_uid")
        .to_multimodal()
    )
    multimodal
    return (multimodal,)


@app.cell
def _(mo, multimodal):
    mo.md(f"""
    ### Shared obs

    `multimodal.obs` has **{multimodal.n_cells}** rows — one per queried cell,
    regardless of which modalities are present.
    """)
    return


@app.cell
def _(multimodal):
    multimodal.obs.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Accessing individual modalities

    Each modality is accessed by feature space name. The type depends on
    the data — AnnData, FragmentResult, or ndarray.
    """)
    return


@app.cell
def _(multimodal):
    for fs, data in multimodal.mod.items():
        n_present = int(multimodal.present[fs].sum())
        print(f"  {fs}: {type(data).__name__}, {n_present} cells present")
    return


@app.cell
def _(multimodal):
    # AnnData modalities work as expected
    multimodal["gene_expression"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Converting to MuData

    `.to_mudata()` extracts only the AnnData-compatible modalities.
    Non-AnnData modalities (fragments, raw arrays) are dropped with a warning.
    """)
    return


@app.cell
def _(multimodal):
    mdata_from_multi = multimodal.to_mudata()
    mdata_from_multi
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. Perturbation-aware queries

    `PerturbationAtlas.query()` returns a `PerturbationQuery` — an extended
    query builder that can resolve human-readable identifiers (gene names,
    compound names, accessions) into the appropriate foreign-key UIDs and
    filter cells accordingly.

    All `by_*` methods compose via AND: calling `.by_gene().by_compound()`
    finds cells with both perturbations (combinatorial).
    """)
    return


@app.cell
def _(atlas_rw):
    atlas_rw.db.open_table("genetic_perturbations").search().to_pandas()
    return


@app.cell
def _():
    return


@app.cell
def _(atlas_rw, pl):
    # Count controls vs perturbed cells
    control_counts = (
        atlas_rw.query()
        .select(["is_negative_control"])
        .to_polars()
        .group_by("is_negative_control")
        .agg(pl.len().alias("n_cells"))
        .sort("is_negative_control")
    )
    control_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Find cells by gene target

    `by_gene()` looks up the `genetic_perturbations` table, finds matching
    UIDs, and filters cells whose `perturbation_search_string` contains
    those UIDs. No manual UID lookup needed.
    """)
    return


@app.cell
def _(atlas_rw):
    pparg_cells = (
        atlas_rw.query()
        .by_gene("PPARG")
        .select(["cell_line", "assay", "perturbation_uids", "perturbation_types"])
        .limit(500)
        .to_polars()
    )
    print(f"{pparg_cells.height} cells with PPARG perturbation")
    pparg_cells.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Find cells by small molecule

    `by_compound()` looks up the `small_molecules` table by name, SMILES,
    or PubChem CID. Here we find cells treated with dexamethasone
    (from the sciPlex2 screen, GSM4150377).
    """)
    return


@app.cell
def _(atlas_rw):
    dex_cells = (
        atlas_rw.query()
        .by_compound(name="Amisulpride")
        .select(["cell_line", "assay", "perturbation_uids", "perturbation_types",
                 "perturbation_concentrations_um", "perturbation_durations_hr"])
        .limit(500)
        .to_polars()
    )
    print(f"{dex_cells.height} cells treated with Amisulpride")
    dex_cells.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Find cells by biologic agent

    `by_biologic()` looks up the `biologic_perturbations` table by agent
    name. Here we find cells treated with IFN-gamma (from the
    Perturb-CITE-seq dataset, SCP1064).
    """)
    return


@app.cell
def _(atlas_rw):
    ifng_cells = (
        atlas_rw.query()
        .by_biologic("IFN-gamma")
        .select(["cell_line", "assay", "perturbation_uids", "perturbation_types"])
        .limit(500)
        .to_polars()
    )
    print(f"{ifng_cells.height} cells treated with IFN-gamma")
    ifng_cells.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Find cells by accession

    `by_accession()` resolves a GEO accession to dataset UIDs and
    filters cells from those datasets.
    """)
    return


@app.cell
def _(atlas_rw):
    gse153056_cells = (
        atlas_rw.query()
        .by_accession("GSE153056")
        .select(["cell_line", "assay", "is_negative_control"])
        .limit(500)
        .to_polars()
    )
    print(f"{gse153056_cells.height} cells from GSE153056")
    gse153056_cells.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Controls only

    `controls_only()` filters to negative control cells, optionally by
    control type (e.g. `"nontargeting"`, `"DMSO"`).
    """)
    return


@app.cell
def _(atlas_rw):
    controls = (
        atlas_rw.query()
        .by_accession("GSE153056")
        .controls_only()
        .select(["cell_line", "negative_control_type"])
        .to_polars()
    )
    print(f"{controls.height} control cells from GSE153056")
    controls.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Enrich results with perturbation metadata

    `with_perturbation_metadata()` joins full records from the FK tables
    back onto the result, adding list columns like `perturbation_intended_gene_name`.
    """)
    return


@app.cell
def _(atlas_rw):
    enriched = (
        atlas_rw.query()
        .by_gene("PPARG")
        .with_perturbation_metadata()
        .select([
            "cell_line", "assay",
            "perturbation_uids", "perturbation_types",
        ])
        .limit(100)
        .to_polars()
    )
    enriched.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Composing perturbation queries with reconstruction

    `by_*` methods compose naturally with the full `AtlasQuery` API —
    feature spaces, layers, limits, balanced sampling, and all output
    formats (AnnData, MuData, batches, etc.).
    """)
    return


@app.cell
def _(atlas_rw):
    adata_pparg = (
        atlas_rw.query()
        .by_gene("PPARG")
        .feature_spaces("gene_expression")
        .limit(5_000)
        .to_anndata()
    )
    adata_pparg
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Streaming AnnData batches

    For workloads that don't need all cells in memory at once,
    `.to_batches()` streams AnnData objects one chunk at a time.
    """)
    return


@app.cell
def _(atlas_rw, sample_uid):
    batch_iter = (
        atlas_rw.query()
        .where(f"dataset_uid = '{sample_uid}'")
        .feature_spaces("gene_expression")
        .limit(2048)
        .to_batches(batch_size=1024)
    )

    for i, batch_adata in enumerate(batch_iter):
        print(f"Batch {i}: {batch_adata.n_obs} cells × {batch_adata.n_vars} genes")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8. ML training with `CellDataset` + `CellSampler`

    lancell provides a purpose-built dataloader pipeline:

    ```
    AtlasQuery → CellDataset + CellSampler → DataLoader → SparseBatch → collate_fn → GPU
    ```

    - **`CellDataset`** maps cell indices to sparse zarr reads
    - **`CellSampler`** groups cells by zarr group for I/O locality
    - **`make_loader`** wires them into a standard `DataLoader`
    - **`sparse_to_dense_collate`** converts sparse batches to dense tensors
    """)
    return


@app.cell
def _():
    import torch

    from lancell.dataloader import make_loader, sparse_to_dense_collate
    from lancell.sampler import CellSampler

    return CellSampler, make_loader, sparse_to_dense_collate, torch


@app.cell
def _(atlas_rw):
    dataset = (
        atlas_rw.query()
        .feature_spaces("gene_expression")
        .limit(50_000)
        .to_cell_dataset(
            feature_space="gene_expression",
            layer="counts",
        )
    )
    print(f"CellDataset: {dataset.n_cells:,} cells, {dataset.n_features:,} features")
    return (dataset,)


@app.cell
def _(CellSampler, dataset):
    BATCH_SIZE = 1024
    NUM_WORKERS = 4

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


@app.cell
def _(dataset, make_loader, sampler, sparse_to_dense_collate, torch):
    sampler.set_epoch(0)
    loader = make_loader(dataset, sampler)

    for batch_idx, batch in enumerate(loader):
        result = sparse_to_dense_collate(batch)
        X = result["X"]

        if batch_idx == 0:
            print(f"First batch X shape: {X.shape}, dtype: {X.dtype}")
            print(f"  Non-zero fraction: {(X != 0).float().mean():.4f}")
            print(f"  Row sums (UMI counts): min={X.sum(1).min():.0f}, "
                  f"median={X.sum(1).median():.0f}, max={X.sum(1).max():.0f}")

    print(f"\nProcessed {batch_idx + 1} batches, last X shape: {X.shape}")
    _ = torch
    return (X,)


@app.cell
def _(X):
    X.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 9. Chromatin accessibility (ATAC-seq fragments)

    The atlas also stores ATAC-seq fragment data from
    [GSE161002](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE161002)
    (CRISPR-sciATAC in K-562 cells). Unlike gene expression or protein
    abundance, chromatin accessibility is stored as **raw genomic fragments**
    — three parallel 1D arrays (chromosomes, starts, lengths) — rather than
    a cell-by-feature matrix.

    Fragments are accessed via `.to_fragments()` on the query builder, which
    returns a `FragmentResult` with CSR-style offsets for per-cell access.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Chromatin accessibility feature registry

    The chromatin accessibility registry stores reference sequences
    (chromosomes) rather than genes or proteins.
    """)
    return


@app.cell
def _(atlas_rw):
    chrom_registry = (
        atlas_rw._registry_tables["chromatin_accessibility"]
        .search()
        .to_polars()
        .sort("global_index")
    )
    n_chroms = chrom_registry.height
    print(f"{n_chroms} reference sequences in the chromatin accessibility registry")
    chrom_registry.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Reading fragments from the atlas

    We query K-562 cells (the ATAC-seq cell line), then use `.to_fragments()`
    to read their chromatin accessibility data from zarr.
    """)
    return


@app.cell
def _():
    import numpy as np

    return (np,)


@app.cell
def _(atlas_rw, np):
    # Query 500 K-562 cells and reconstruct fragments directly
    frag_result = (
        atlas_rw.query()
        .where("cell_line = 'K-562'")
        .where("chromatin_accessibility.zarr_group != ''")
        .limit(500)
        .to_fragments()
    )
    per_cell_counts = np.diff(frag_result.offsets)

    print(f"{len(frag_result.obs)} cells, {frag_result.offsets[-1]:,} total fragments")
    print(f"Per-cell: min={per_cell_counts.min()}, median={int(np.median(per_cell_counts))}, max={per_cell_counts.max()}")
    print(f"Chromosomes: {frag_result.chrom_names[:5]}...")
    return frag_result, per_cell_counts


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Fragment length distribution

    ATAC-seq fragments show a characteristic nucleosomal periodicity:
    a sub-nucleosomal peak near ~150 bp and mono-nucleosomal peak
    near ~350 bp.
    """)
    return


@app.cell
def _(frag_result, np, plt):
    fig_frag, ax_frag = plt.subplots(figsize=(8, 3))
    ax_frag.hist(frag_result.lengths, bins=np.arange(0, 1001, 10), edgecolor="none", alpha=0.7)
    ax_frag.set_xlabel("Fragment length (bp)")
    ax_frag.set_ylabel("Count")
    ax_frag.set_title("Fragment length distribution (500 K-562 cells)")
    fig_frag.tight_layout()
    fig_frag
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Per-cell fragment count distribution
    """)
    return


@app.cell
def _(np, per_cell_counts, plt):
    fig_pc, ax_pc = plt.subplots(figsize=(8, 3))
    ax_pc.hist(per_cell_counts, bins=np.arange(0, per_cell_counts.max() + 100, 100), edgecolor="none", alpha=0.7)
    ax_pc.set_xlabel("Fragments per cell")
    ax_pc.set_ylabel("Number of cells")
    ax_pc.set_title("Per-cell fragment count distribution")
    fig_pc.tight_layout()
    fig_pc
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Extracting a single cell's fragments

    With CSR-style offsets, extracting one cell's fragments is a simple
    array slice.
    """)
    return


@app.cell
def _(frag_result, np, pl):
    cell_idx = 0
    s, e = frag_result.offsets[cell_idx], frag_result.offsets[cell_idx + 1]

    single_cell_df = pl.DataFrame(
        {
            "chrom": [frag_result.chrom_names[c] for c in frag_result.chromosomes[s:e]],
            "start": frag_result.starts[s:e],
            "length": frag_result.lengths[s:e],
            "end": frag_result.starts[s:e].astype(np.int64) + frag_result.lengths[s:e].astype(np.int64),
        }
    )
    print(f"Cell {cell_idx}: {e - s} fragments across {single_cell_df['chrom'].n_unique()} chromosomes")
    single_cell_df.head(15)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Peak count matrix from fragments

    `FragmentCounter` converts raw fragments into a **cells x peaks**
    sparse count matrix. Given a list of `GenomicRange` objects (peaks,
    bins, or arbitrary intervals), it counts overlapping fragments per cell.
    """)
    return


@app.cell
def _(frag_result):
    from lancell.fragments.peak_matrix import (
        FragmentCounter,
        GenomicRange,
    )

    # Example peaks across several chromosomes
    example_peaks = [
        GenomicRange("chr1", 1_000_000, 1_010_000, name="chr1_peak1"),
        GenomicRange("chr1", 1_500_000, 1_510_000, name="chr1_peak2"),
        GenomicRange("chr2", 500_000, 510_000, name="chr2_peak1"),
        GenomicRange("chr2", 1_000_000, 1_010_000, name="chr2_peak2"),
        GenomicRange("chr5", 100_000, 110_000, name="chr5_peak1"),
        GenomicRange("chr17", 7_500_000, 7_600_000, name="chr17_TP53_locus"),
    ]

    counter = FragmentCounter(example_peaks)
    peak_matrix = counter.count_fragments(frag_result)
    peak_adata = counter.to_anndata(frag_result)

    print(f"Peak matrix: {peak_matrix.shape[0]} cells x {peak_matrix.shape[1]} peaks")
    print(f"Non-zero entries: {peak_matrix.nnz}")
    print(f"Total fragment count: {peak_matrix.sum()}")
    peak_adata
    return (peak_adata,)


@app.cell
def _(peak_adata):
    peak_adata.var
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Perturbation context for ATAC cells

    The ATAC-seq cells carry the same perturbation metadata as other
    modalities, enabling perturbation-aware chromatin accessibility
    analyses.
    """)
    return


@app.cell
def _(frag_result):
    atac_pert_summary = (
        frag_result.obs[["is_negative_control"]]
        .value_counts()
        .sort_index()
    )
    atac_pert_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 10. Image features and tiles (Cell Painting)

    The atlas contains Cell Painting data from
    [cpg0021-periscope](https://github.com/broadinstitute/cellpainting-gallery)
    (Feldman et al. 2025) — genome-wide CRISPRko optical pooled screening in
    HeLa cells. Two dense modalities with partially overlapping cells:

    - **image_features** — 4,822 cells × 3,745 CellProfiler features (2D dense)
    - **image_tiles** — 46 cells × 5ch × 96×96 uint16 raw tiles (4D dense, no var)

    Since HeLa is the only cell line with image data, we can filter on it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Image feature registry

    The image feature registry stores CellProfiler feature names (Nuclei, Cells,
    Cytoplasm compartments) with their global indices.
    """)
    return


@app.cell
def _(atlas_rw):
    image_feature_registry = (
        atlas_rw._registry_tables["image_features"]
        .search()
        .to_polars()
        .sort("global_index")
    )
    n_image_features = image_feature_registry.height
    print(f"{n_image_features:,} features in the image feature registry")
    image_feature_registry.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Loading image features as AnnData

    Image features use a standard 2D dense layout with a feature registry,
    so `.to_anndata()` works directly — just like gene expression but dense.
    """)
    return


@app.cell
def _(atlas_rw):
    adata_image = (
        atlas_rw.query()
        .where("cell_line = 'HeLa'")
        .feature_spaces("image_features")
        .to_anndata()
    )
    adata_image
    return (adata_image,)


@app.cell
def _(adata_image):
    adata_image.obs[["cell_line", "assay", "is_negative_control", "perturbation_uids"]].head(10)
    return


@app.cell
def _(adata_image, np, pl, plt):
    # Feature magnitude distribution across compartments
    compartments = ["Nuclei", "Cells", "Cytoplasm"]
    feature_names = adata_image.var["feature_name"].tolist()

    fig_comp, axes_comp = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    for _ax, compartment in zip(axes_comp, compartments):
        mask = [f.startswith(f"{compartment}_") for f in feature_names]
        col_indices = np.where(mask)[0]
        if len(col_indices) > 0:
            means = np.nanmean(adata_image.X[:, col_indices], axis=0)
            _ax.hist(means, bins=50, edgecolor="none", alpha=0.7)
            _ax.set_title(f"{compartment} ({len(col_indices)} features)")
            _ax.set_xlabel("Mean value")
    axes_comp[0].set_ylabel("Count")
    fig_comp.suptitle("CellProfiler feature means by compartment", y=1.02)
    fig_comp.tight_layout()
    _ = pl  # keep import alive
    fig_comp
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Loading image tiles (4D dense arrays)

    Image tiles are stored as 4D arrays `(n_cells, channels, height, width)`.
    Since there's no feature registry (`has_var_df=False`), `.to_array()`
    returns the raw NumPy array preserving all dimensions.
    """)
    return


@app.cell
def _(atlas_rw):
    tiles, tiles_obs = (
        atlas_rw.query()
        .where("cell_line = 'HeLa'")
        .where("image_tiles.zarr_group != ''")
        .to_array(feature_space="image_tiles")
    )
    print(f"Tile array shape: {tiles.shape} ({tiles.dtype})")
    print(f"Cells: {len(tiles_obs)}")
    return (tiles,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Displaying tiles

    Each tile has 5 channels from the Cell Painting assay. We show a grid
    of cells with a false-color composite (channels 0, 1, 2 mapped to RGB).
    """)
    return


@app.cell
def _(np, plt, tiles):
    n_show = min(12, tiles.shape[0])
    ncols = min(6, n_show)
    nrows = (n_show + ncols - 1) // ncols

    fig_tiles, axes_tiles = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes_flat = [axes_tiles] if n_show == 1 else axes_tiles.ravel()

    for _i in range(n_show):
        _ax = axes_flat[_i]
        # False-color composite: map channels 0,1,2 to R,G,B
        rgb = np.stack([tiles[_i, c] for c in range(3)], axis=-1).astype(np.float32)
        rgb = rgb / np.percentile(rgb, 99.5)  # contrast stretch
        rgb = np.clip(rgb, 0, 1)
        _ax.imshow(rgb)
        _ax.set_title(f"Cell {_i}", fontsize=8)
        _ax.axis("off")

    # Hide unused axes
    for _i in range(n_show, len(axes_flat)):
        axes_flat[_i].axis("off")

    fig_tiles.suptitle("Cell Painting tiles (false-color composite: ch0=R, ch1=G, ch2=B)", y=1.02)
    fig_tiles.tight_layout()
    fig_tiles
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Individual channels

    Each of the 5 Cell Painting channels captures different cellular
    structures. Here we show all channels for one cell.
    """)
    return


@app.cell
def _(np, plt, tiles):
    channel_names = ["DNA", "ER", "RNA", "AGP", "Mito"]
    cell_idx_show = 0

    fig_ch, axes_ch = plt.subplots(1, 5, figsize=(15, 3))
    for _c, (_ax, _name) in enumerate(zip(axes_ch, channel_names)):
        _img = tiles[cell_idx_show, _c].astype(np.float32)
        _ax.imshow(_img, cmap="gray", vmin=0, vmax=np.percentile(_img, 99.5))
        _ax.set_title(_name, fontsize=10)
        _ax.axis("off")
    fig_ch.suptitle(f"Cell {cell_idx_show} — all 5 Cell Painting channels", y=1.02)
    fig_ch.tight_layout()
    fig_ch
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    That covers the core workflow: ingest → optimize → snapshot → query →
    reconstruct (AnnData, MuData, fragments, or tiles) → train. For subsequent
    sessions, comment out the `open` + `optimize` + `snapshot` cell and
    uncomment the `checkout_latest` cell for read-only access.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
