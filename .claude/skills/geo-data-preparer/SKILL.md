---
name: geo-data-preparer
description: Use when a user provides a GEO accession and a target schema file and whats to prepare a dataset for ingestion. Covers listing and downloading GEO files as well a file classification, metadata creation, and delegation to resolver sub-agents for metadata resolution to ontologies and databases.
---

# GEO Data Preparer

## Scope

This skill handles the full pre-ingestion pipeline:

1. **Listing** supplementary files for a GEO accession
2. **Downloading** selected files via FTP
3. **Classifying** files (e.g., h5ad vs matrix + companion files)
4. **Writing metadata.json** that stores GEO series or sample metadata
5. **Creating global tables** for feature registries and foreign keys
6. **Delegating** resolution to resolver sub-agents (accession-level global tables)

It does NOT handle assembling standardized CSVs, adding data to LanceDB, or writing Zarr arrays. Those responsibilities belong to the `geo-data-curator` skill. Mapping resolved perturbation UIDs back to per-experiment obs is handled by the perturbation resolver subagents (e.g., `genetic-perturbation-resolver` writes `{fs}_fragment_perturbation_obs.csv`).

> **HARD BOUNDARY: The preparer is a logistics role, not an interpretation role.** You download files, extract raw dataframes, and hand them to resolvers. You do NOT:
> - Read, parse, or inspect supplementary library files (guide libraries, reagent libraries, compound libraries). Pass the file path to the resolver.
> - Make design decisions about how resolvers should interpret data (e.g., how dual-guide pairs map to schema rows, how control labels are detected, how transcript IDs map to target context). The resolver skills already encode this logic.
> - Ask the user questions that a resolver would answer (e.g., "should each guide be its own row?", "what does this column mean for the perturbation schema?"). If you don't know, the resolver does.
>
> Your job is to relay column names, file paths, and delimiters to resolvers — not to understand what's inside them. When in doubt, pass more context to the resolver rather than trying to interpret it yourself.

## Scripts

You have access to scripts that can be used for common tasks. Run these via Bash. All paths are relative to this skill directory.

| Script | Usage | Purpose |
|--------|-------|---------|
| `scripts/list_geo_files.py` | `python scripts/list_geo_files.py GSE123456` | List supplementary files for any GEO accession (GSE or GSM) |
| `scripts/download_geo_file.py` | `python scripts/download_geo_file.py GSE123456 file.h5ad [dest_dir]` | Download a supplementary file via FTP (default dest: `/tmp/geo_agent/<accession>/`) |
| `scripts/write_metadata_json.py` | `python scripts/write_metadata_json.py <experiment_dir> <accession>` | Fetch GEO metadata and write metadata.json in the experiment directory |
| (see **publication-resolver** skill) | `python scripts/write_publication_json.py <data_dir> [--pmid PMID] [--title TITLE]` | Fetch publication metadata from PubMed/PMC and write publication.json (delegated to publication-resolver skill) |
| `scripts/reconcile_barcodes.py` | `python scripts/reconcile_barcodes.py <experiment_dir>` | Reconcile barcodes across modalities; writes `multimodal_barcode` to each feature space's preparer fragment |

## Workflow

### 1. List and identify data files for the provided GEO accession

Check the available files. If the user provides a series or super-series record from GEO, you may need to look for files at the sample-level. If the series record has aggregated and preprocessed files or a large tar file, those are generally preferable. However, if the series level has no files or only summary statistics, then check the sample-level for real data. If there are many sample records for a series its best to process them one at a time to avoid confusion. When this is the case, ask the user how they want to proceed.

Currently, we support the following file formats:

| Format | Action |
|--------|--------|
| `.h5ad` | Already AnnData — keep as-is, set `anndata` field |
| `.h5` (10x HDF5) | Set `matrix_file` field; can be read with `scanpy.read_10x_h5()` for validation |
| `.mtx` / `.mtx.gz` (Market Matrix) | Set `matrix_file` field; companions go to `cell_metadata`/`var_metadata` |
| `.tsv` / `.tsv.gz` | Sometimes used for protein abundance which is not sparse |
| `_fragments.tsv.gz` / `.bed.gz` / `.bed` | Fragment files — per-cell chromatin accessibility regions. Columns: `(chrom, start, end, barcode)` (4-col) or `(chrom, start, end, barcode, count)` (5-col, 10x format) |
| `.bw` (bigWig) | Not supported. Per-sample coverage tracks, not per-cell data. Skip and note in output. |
| Peak matrices (cells × peaks) | Not supported for chromatin accessibility ingestion. Skip and note in output. |
| `.rds` | Not supported. Skip and note in output. |

If the file formats present on the GEO record fall outside of this list, raise it to the user.

**mtx bundles:** When you see `.mtx.gz` files, look for companion `barcodes.tsv.gz` and `features.tsv.gz` (or `genes.tsv.gz`) files. These form a single dataset. If the mtx bundle files are in a tar/gz archive, download and extract it first.

**Multimodal datasets:** Watch out for file naming patterns that indicate multiple modalities from the same experiment (e.g., `*_cDNA_*` and `*_ADT_*` for CITE-seq, `*_RNA_*` and `*_ATAC_*` for multiome). We will want to group these files together later.

### 2. Read the schema file

This skill's validation workflow is driven by a **user-provided Python schema file**, which is of type lancedb.pydantic.LanceModel, a subclass of a pydantic BaseModel. The schema defines the tables and fields to populate with GEO data.

The user must provide the schema file path as part of their task prompt. Example:

> "Prepare GSE123456 using the schema at `some/path/schema.py`"

If no schema was provided, ask the user for the path before going any further. Read the Python file and identify:

1. **The obs schema class** — This inherits from `LancellBaseSchema`, verify that there is only one table in the schema file matching this.
2. **Feature registry classes** — These inherit from `FeatureBaseSchema` and correspond to var-level fields per feature space supported by an atlas.
3. **Foreign key classes** — These inherit directly from `LanceModel`. These tables are referenced by either the obs table or a feature registry table through a foreign key.

Our goal is to fill out each of the schemas and fields that apply to the provided GEO dataset, which will always include the obs class but may involve only a subset of the feature registry and foreign key classes in the schema file. If a field's purpose is unclear from its name, type, docstring or in-line comments, ask the user.

### 3. Download and read GEO metadata

Download the metadata from the GEO series or sample records:

```
python scripts/write_metadata_json.py /tmp/geo_agent/<accession> <accession>
```

You may need to run this multiple times. Sometimes when the data is stored at the series level, it still references a sample record (e.g., the filename contains a GSM id). In this case, download the metadata from the series and from the referenced sample ids.

Read the relevant json files. These often include helpful information about how to use the files and high-level metadata like organism or assay.

### 4. Download and parse the publication

Launch a subagent with `publication-resolver` skill to create `publication.json`. Provide it with a publication title, PMID, DOI, or author names and search terms and it will do the work of finding the publication on pubmed and downloading and parsing it. Often the requisite information will be found in the GEO metadata json files that you just downloaded in the previous step.

### 5. Download and organize files by experiment

Download the necessary files from GEO (be sure to use long enough timeouts for large files):

```
python scripts/download_geo_file.py <accession> <filename> [dest_dir]
```

Default destination: `/tmp/geo_agent/<accession>/`. Some GEO datasets have multiple files in a single tar archive -- extract it. If there are multiple versions of the same dataset, possibly indicated by terms like "filtered", "processed", or "validated", prefer these analysis-ready artifacts to the raw version. Ask the user if unsure.

Next group the files into subdirectories by experiment. Simply use `mv` to move the files into the correct subdirectory, no `cp` or `ln -s` for symlinks. Depending on the file formats and whether the assay is unimodal or multimodal, we may have multiple files bundled together in the same subdirectory. Do not create separate subdirectories for modalities captured in the same experiment.

### 6. Create raw obs and var dataframes

Each of the subdirectories should have dataframes that correspond to obs-level and typically var-level metadata as well. These dataframe might be csv or tsv or inside of an h5ad file. In either case, write new csv files with suffix `_raw_obs.csv` and `_raw_var.csv`, where the feature space might be "gene_expression", "chromatin_accessibility, "protein_abundance", etc. There shouldn't be more than 1 obs or var csv per modality.

For the most part you should not remove any columns from the original dataframes, but you may add additional fields that were discovered from the GEO metadata or the downloaded publication text. For example, the raw dataframes associated might not include global metadata like organism, cell type, or donor information. If that information is in the metadata or publication, create new columns relevant to the schema. Do not worry about standardizing the terms that you find because that is delegated to the resolver subagents.

For any obs fields that need only pass-through or type coercion (e.g., batch_id, replicate, well_position, days_in_vitro), write them to `{fs}_fragment_preparer_obs.csv` using the schema field names directly. For multimodal experiments, also run barcode reconciliation:

```
python scripts/reconcile_barcodes.py <experiment_dir>
```

### 7. Create global feature and foreign key tables

Before launching resolvers, create accession-level `_raw.csv` files that consolidate data across all experiments for entities that need global resolution.

**For each feature registry schema** (e.g., `GenomicFeatureSchema`):

1. Concatenate per-experiment `{fs}_raw_var.csv` files
2. Add columns: `var_index` (the var index value), `experiment_subdir`, `source_var_column`
3. Deduplicate on `var_index`
4. Write `{SchemaClassName}_raw.csv` at accession level (e.g., `GenomicFeatureSchema_raw.csv`)

**For each foreign key schema** (e.g., `GeneticPerturbationSchema`, `SmallMoleculeSchema`):

1. Extract relevant columns from obs across all experiments
2. Add a key column (e.g., `reagent_id`) for mapping back
3. Deduplicate on key
4. Write `{SchemaClassName}_raw.csv` at accession level

Column misalignment across datasets is OK — union of columns with NaN fills.

**Enrich `_raw.csv` with supplementary data.** Before handing off to resolvers, add supplementary info (e.g., publication metadata, global experimental variables, etc.) into `_raw.csv`. **`_raw.csv` contains all available information in unstandardized form.** The preparer never calls resolution functions; the resolver never hunts for supplementary files.

> **STOP: Do not read, parse, inspect, or make decisions about supplementary library files (guide libraries, reagent libraries, compound libraries, etc.).** Do not ask the user questions about their contents or format. Do not try to understand the data format or make schema-mapping decisions based on them. Pass the file path directly to the appropriate resolver subagent in its prompt. The resolver skills already know how to handle these files. Your only job is to relay the path. This applies equally to perturbation columns in obs — extract them into `_raw.csv` but do not interpret their structure (e.g., delimiters, ID formats, control labels). Tell the resolver what the column is called and let it do the rest.

**Naming convention:** Use the full schema class name: `GenomicFeatureSchema`, `GeneticPerturbationSchema`, `SmallMoleculeSchema`, `BiologicPerturbationSchema`, `ProteinSchema`.

### 8. Delegate resolution to resolver subagents

Feature registries (var) and foreign key tables are resolved across ALL experiments in one pass. Same entity in multiple experiments gets one UID.

Launch relevant resolvers for each global `_raw.csv`:

| Input | Resolver Skill | Output |
|-------|---------------|--------|
| `GenomicFeatureSchema_raw.csv` | `gene-resolver` | `GenomicFeatureSchema_resolved.csv` |
| `ProteinSchema_raw.csv` | `protein-resolver` | `ProteinSchema_resolved.csv` |
| `GeneticPerturbationSchema_raw.csv` | `genetic-perturbation-resolver` | `GeneticPerturbationSchema_resolved.csv` |
| `SmallMoleculeSchema_raw.csv` | `molecule-resolver` | `SmallMoleculeSchema_resolved.csv` |
| `BiologicPerturbationSchema_raw.csv` | `protein-resolver` | `BiologicPerturbationSchema_resolved.csv` |

**Prompt template for resolvers:**

```
Agent tool call:
  prompt: |
    Read the skill file at .claude/skills/<resolver-name>/SKILL.md and follow its workflow.

    Context:
    - Accession directory: <accession_dir>
    - Schema file: <schema_path>
    - Input: <SchemaClassName>_raw.csv
    - Output: <SchemaClassName>_resolved.csv (with UIDs assigned via make_uid())
```

Avoid giving the resolver skill any instructions about how to resolve the data. It already knows the correct procedure, such instructions in your prompt might contradict the skill.

**For the genetic-perturbation-resolver specifically**, also provide the obs-level mapping context so it can write perturbation obs fragments (see B1–B4 in the resolver skill):

```
    Additional context for obs-level fragment writing:
    - Experiment directories: [list of experiment subdirectory paths]
    - Perturbation column in obs: <column_name> (e.g., "sgID_AB")
    - Feature space: <feature_space> (e.g., "gene_expression")
    - Delimiter for multi-guide: <delimiter> (e.g., "|" for pipe-separated dual guides)
    - Dose column: <column_name or None>
    - Duration column: <column_name or None>
```

All resolvers can run in parallel.

**Note:** The ontology resolver operates per-experiment (writing `{fs}_fragment_ontology_obs.csv` directly in each experiment directory), unlike other resolvers which write global accession-level tables.

### 9. Verification

After all resolvers complete, verify that the expected output files exist:

- Finalized global tables: `{SchemaClassName}.csv` for each feature registry and foreign key schema
- Per-experiment: raw obs/var CSVs, resolver fragment obs CSVs (e.g., ontology fragments), preparer fragment obs CSVs
- Accession-level: `metadata.json`, `publication.json`

The preparer is now complete. Hand off to the `geo-data-curator` skill for assembly, validation, and ingestion.

## Directory Layout

```
/tmp/geo_agent/GSE264667/
├── GenomicFeatureSchema_raw.csv                        # resolver input
├── GenomicFeatureSchema_resolved.csv                   # resolver intermediate (with UIDs + raw columns)
├── GenomicFeatureSchema.csv                            # finalized, schema-validated
├── GeneticPerturbationSchema_raw.csv                   # resolver input
├── GeneticPerturbationSchema_resolved.csv              # resolver intermediate (with UIDs + raw columns)
├── GeneticPerturbationSchema.csv                       # finalized, schema-validated
├── publication.json
├── GSE264667_metadata.json
├── HepG2/
│   ├── GSE264667_HepG2.h5ad
│   ├── gene_expression_raw_obs.csv                     # all obs columns from the h5ad + metadata
│   ├── gene_expression_raw_var.csv                     # all var columns from the h5ad
│   ├── gene_expression_fragment_preparer_obs.csv       # pass-through fields (batch_id, etc.)
│   ├── gene_expression_fragment_ontology_obs.csv       # ontology-resolved fields (organism, assay, etc.)
│   ├── gene_expression_fragment_perturbation_obs.csv   # perturbation UIDs, control flags (from genetic-perturbation-resolver)
├── Jurkat/
│   └── ...
```

## Column Naming Convention

All resolvers output schema field names directly — no `validated_` prefix:
- `cell_type` not `validated_cell_type`
- `gene_name` not `validated_gene_name`
- `organism` as resolved scientific name (e.g., "Homo sapiens", "Mus musculus")
