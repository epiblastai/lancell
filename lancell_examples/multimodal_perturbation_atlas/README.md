# Multimodal Perturbation Atlas

A unified, queryable atlas of single-cell perturbation data built with [lancell](../../README.md). Five feature spaces. Eighteen datasets spanning gene expression, chromatin accessibility, protein abundance, and cell imaging. Thousands of genetic and chemical perturbations — including combinatorial screens. All standardized, all queryable, all streamable for ML training.

| | |
|---|---|
| **Cells** | ~100M+ (gene expression), ~257K (chromatin), ~255K (protein), ~15K (imaging) |
| **Datasets** | 18 published studies |
| **Feature spaces** | `gene_expression`, `chromatin_accessibility`, `protein_abundance`, `image_features`, `image_tiles` |
| **Perturbation types** | CRISPR knockout, CRISPRi, CRISPRa, small molecules, biologics, combinations |
| **Cell lines** | K562, A549, HeLa, MCF7, RPE1, THP-1, A375, GM12878, Jurkat, EndoC-betaH1, 50+ cancer lines |

---

## Why this atlas exists

Perturbation biology data is fragmented. Each lab uses different assays, gene panels, file formats, and metadata conventions. CRISPR screens use different guide libraries. Chemical screens use different compound naming. There is no existing resource that unifies gene expression, chromatin accessibility, protein abundance, and cell imaging from perturbation experiments into a single queryable structure.

ML foundation models need large-scale, heterogeneous training data that can be streamed efficiently without materializing everything into memory. This atlas delivers exactly that: a single `RaggedAtlas` that holds every dataset in its native feature space, standardized by AI agents, queryable through SQL + vector search + full-text search, and streamable directly into PyTorch dataloaders.

---

## Why lancell and the RaggedAtlas

Real-world perturbation datasets were never designed to be compatible. One Perturb-seq experiment measures 33,000 genes; another measures 1,100 targeted genes. An ATAC-seq screen produces genomic fragments, not count matrices. A Cell Painting screen produces 3,700 CellProfiler features and 5-channel image tiles. Conventional approaches either pad to a union matrix (wasteful) or intersect to shared features (lossy).

Lancell's `RaggedAtlas` takes a different approach: each dataset occupies its own Zarr group with its own feature ordering. Every cell carries a pointer into its group. At query time, the reconstruction layer handles union, intersection, or feature-filtered reads — no padding is stored, no information is discarded.

```
Cell table (LanceDB)                  Zarr store (per-dataset groups)
─────────────────────                 ──────────────────────────────
cell A  gene_expression → GSE90546/   GSE90546/    32,738 genes, 86K cells
cell B  gene_expression → Norman19/   Norman19/    33,694 genes, 111K cells
cell C  protein_abundance → SCP1064/  SCP1064/     24 proteins, 218K cells
cell D  chromatin_access. → GSE168/   GSE168851/   fragments, 119K cells
cell E  image_features → cpg0021/     cpg0021/     3,725 features, 15K cells
```

**Multimodal by design.** A single cell row can carry pointers to multiple feature spaces simultaneously — sparse gene expression, dense protein measurements, chromatin fragments, and image tiles — all in one record. Query across modalities with `.to_multimodal()` or `.to_mudata()`.

**LanceDB metadata layer.** Cell metadata is a LanceDB table: SQL predicates, vector similarity search, full-text search, and scalar indexing come out of the box. Filter by cell type, perturbation, tissue, or embedding similarity without custom loaders.

**Versioned snapshots.** Every `optimize()` + `snapshot()` pins a reproducible, read-only view. Training runs execute against a frozen version while ingestion continues into the live atlas.

---

## Agent-powered standardization

Every dataset in this atlas was curated by AI agents equipped with specialized resolver skills. Raw GEO metadata — inconsistent gene symbols, free-text cell type labels, vendor compound names — is resolved to canonical identifiers before ingestion.

| Resolver | Input | Output |
|----------|-------|--------|
| **Gene** | Gene symbols, Ensembl IDs | Canonical Ensembl gene ID + NCBI cross-reference |
| **Guide RNA** | 20bp DNA sequence | Genomic coordinates via BLAT alignment + Ensembl overlap annotation |
| **Molecule** | Compound names, SMILES | PubChem CID + canonical SMILES + InChI key |
| **Protein** | Protein names, aliases | UniProt accession + sequence metadata |
| **Ontology** | Free-text cell type, tissue, disease | Canonical ontology CURIEs (CL, UBERON, MONDO, EFO) |
| **Publication** | GEO accession, PMID, DOI | Structured metadata + full text sections from PubMed/PMC |

All resolver results are cached in a local LanceDB reference database for offline reproducibility. The same resolvers are available as standalone tools via `lancell.standardization`.

---

## Genome-anchored genetic perturbations

Most perturbation databases store only gene names. This is lossy: enhancer-targeting screens, intergenic guides, and multi-gene effects are invisible when all you record is "TP53 knockout."

This atlas anchors every genetic perturbation to **genomic coordinates** — chromosome, start, end, strand — via BLAT alignment of guide RNA sequences against the reference genome. Each perturbation record stores both the coordinates (ground truth) and the intended gene name (annotation):

```python
GeneticPerturbationSchema(
    perturbation_type="crispr_ko",
    guide_sequence="GACTTCACCTGGAATCAGAT",
    target_chromosome="CM000684.2",     # chr22 (GenBank accession)
    target_start=28_695_869,
    target_end=28_695_889,
    target_strand="+",
    target_context="exon",              # classified by Ensembl overlap
    intended_gene_name="CHEK2",
    intended_ensembl_gene_id="ENSG00000183765",
)
```

This enables re-annotation against updated gene models, liftover to other assemblies, and correct handling of non-coding perturbations that don't map cleanly to a single gene.

---

## Querying the atlas

```python
from lancell_examples.multimodal_perturbation_atlas.atlas import PerturbationAtlas

atlas = PerturbationAtlas.checkout_latest("/path/to/atlas")
```

**Find all cells with a TP53 knockout:**

```python
adata = atlas.query().by_gene("TP53").limit(5000).to_anndata()
```

**Find all cells treated with dexamethasone:**

```python
adata = atlas.query().by_compound(name="dexamethasone").to_anndata()
```

**Multimodal query — gene expression + protein for CITE-seq datasets:**

```python
result = atlas.query().by_accession("SCP1064").to_multimodal()
result.mod["gene_expression"]     # AnnData (sparse)
result.mod["protein_abundance"]   # AnnData (dense, 24 proteins)
result.present["gene_expression"] # boolean mask: which cells have each modality
```

**Chromatin accessibility fragments:**

```python
fragments = atlas.query().by_accession("GSE168851").to_fragments()
fragments.chromosomes  # uint8 per fragment
fragments.starts       # uint32, delta-encoded BP-128
fragments.lengths      # uint16
```

**Feature-filtered read — only load specific genes via CSC index:**

```python
adata = (
    atlas.query()
    .features(["TP53", "BRCA1", "KRAS"], "gene_expression")
    .to_anndata()
)
```

**Controls for a specific dataset:**

```python
controls = atlas.query().by_accession("GSE133344").controls_only().to_anndata()
```

**Stream into a PyTorch dataloader for foundation model training:**

```python
from lancell.dataloader import CellDataset, CellSampler, make_loader

dataset = CellDataset(atlas, feature_spaces=["gene_expression"])
loader = make_loader(dataset, batch_size=4096, num_workers=4)
for batch in loader:
    # batch.sparse["gene_expression"] → (offsets, indices, values)
    pass
```

All query methods compose: chain `.by_gene()`, `.by_compound()`, `.where()`, `.controls_only()`, `.features()`, `.limit()`, and `.balanced_limit()` in any order.

---

## Extending the atlas

Adding a new dataset requires no schema migrations and no reindexing of existing data:

1. **Register features** — new genes, proteins, or sequences merge into the global registry via `merge_insert` (deduplicates automatically)
2. **Ingest cells** — `add_from_anndata()` or `add_anndata_batch()` writes arrays to a new Zarr group and inserts cell records
3. **Optimize + snapshot** — compact Lance fragments, assign global indices, pin a new version

Adding a new feature space is equally straightforward: add a pointer column to the cell schema and a registry schema. All existing search, query, streaming, and reconstruction capabilities apply to the new space automatically.

Foreign key tables — `publications`, `genetic_perturbations`, `small_molecules`, `biologic_perturbations` — grow incrementally. Each table deduplicates on natural keys, so the same compound or gene resolved from different datasets converges to a single record.

---

## Datasets

### Gene expression

| Accession | Reference | Assay | Cell line(s) | Perturbation | Cells |
|-----------|-----------|-------|-------------|--------------|-------|
| GSE90546 | Dixit et al. 2016 | Perturb-seq | K562 | CRISPR-KO (TFs, epistasis, UPR) | 86K |
| GSE92872 | Datlinger et al. 2017 | CROP-seq | Jurkat | CRISPR-KO (TCR pathway) | 6K |
| GSE120861 | Gasperini et al. 2019 | Perturb-seq | K562 | CRISPR (multiplex screens) | 296K |
| GSE133344 | Norman et al. 2019 | Perturb-seq | K562 | CRISPRa (genetic interactions) | 111K |
| GSE135497 | Schraivogel et al. 2020 | TAP-seq | K562 | CRISPR (chr8, chr11 targets) | 293K |
| GSE149383 | Multiple | 10x / Drop-seq | PC9, M14, A549 | Small molecules (erlotinib, vemurafenib, etc.) | 255K |
| GSE161824 | Ursu et al. 2022 | Perturb-seq | A549 | CRISPR (TP53, KRAS variants) | 176K |
| GSE273677 | Nan et al. 2026 | Perturb-seq | EndoC-betaH1 | CRISPR (GWAS loci, RQC) | 62K |
| GSM4150377 | Srivatsan et al. 2019 | sciPlex | A549 | Small molecules (4 compounds) | 24K |
| GSM4150378 | Srivatsan et al. 2019 | sciPlex | A549, MCF7, K562 | Small molecules (188 compounds) | 763K |
| GSM4150379 | Srivatsan et al. 2019 | sciPlex | A549, MCF7 | Small molecules (HDAC inhibitors) | 31K |
| figshare 20029387 | Replogle et al. 2022 | CRISPRi | K562, RPE1 | CRISPRi (genome-wide) | 2.5M |
| Tahoe-100M | Tahoe Bio 2025 | scRNA-seq | 50 cancer lines | Small molecules (~379 compounds x 3 doses) | 96M+ |

### Protein abundance (CITE-seq)

| Accession | Reference | Assay | Cell line(s) | Perturbation | Cells | Proteins |
|-----------|-----------|-------|-------------|--------------|-------|----------|
| GSE153056 | Papalexi et al. 2021 | ECCITE-seq | THP-1 | CRISPR (PD-L1) | 37K | 4 |
| SCP1064 | Frangieh et al. 2021 | Perturb-CITE-seq | A375 | CRISPR + IFN-gamma | 218K | 24 |

### Chromatin accessibility (ATAC-seq)

| Accession | Reference | Assay | Cell line(s) | Perturbation | Cells |
|-----------|-----------|-------|-------------|--------------|-------|
| GSE161002 | Liscovitch-Brauer et al. 2021 | CRISPR-sciATAC | K562 | CRISPR (chromatin modifiers) | 138K |
| GSE168851 | Pierce et al. 2021 | Spear-ATAC | GM12878, K562, MCF7 | CRISPR (chromatin remodeling) | 119K |

### Image features and tiles (Cell Painting)

| Accession | Reference | Assay | Cell line | Perturbation | Cells | Features |
|-----------|-----------|-------|----------|--------------|-------|----------|
| cpg0021-periscope | Feldman et al. 2025 | Optical pooled screen | HeLa | CRISPR-KO (genome-wide) | ~15K | 3,725 CellProfiler + 5ch tiles |

---

## Getting started

```bash
pip install lancell[all]
```

The interactive exploration notebook covers all atlas capabilities — metadata queries, feature registries, AnnData/MuData reconstruction, perturbation-aware queries, ML streaming, chromatin fragments, and image tiles:

- [`explore_perturbation_atlas.py`](explore_perturbation_atlas.py) (marimo notebook)

For lancell documentation and the core API:

- [lancell docs](https://epiblastai.github.io/lancell/)
- [lancell README](../../README.md)
