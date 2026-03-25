---
name: protein-resolver
description: Use this skill when tasked with standardizing protein identifiers (aliases, gene names, UniProt accessions) in feature dataframes and looking up metadata to fill out missing information in a LanceDB table schema (e.g., ProteinSchema). Requires dataframes with at minimum the protein identifiers to standardize and a target schema specifying missing metadata to lookup. Handles isotype control detection. For biologic perturbation resolution, use a dedicated biologic-perturbation-resolver skill.
---

# Protein Resolver

Resolve protein identifiers in feature dataframes — typically the var index of an ADT/CITE-seq protein abundance matrix. Maps protein aliases, gene names, and UniProt accessions to canonical identifiers using the `lancell.standardization` suite.

For biologic perturbation resolution (cytokines, growth factors, antibodies applied to cells), use a dedicated **biologic-perturbation-resolver** skill. For genetic perturbation targets (CRISPR, siRNA, shRNA), use the **genetic-perturbation-resolver** skill.

## Interface

**Input:**
- `Protein_raw.csv` — consolidated var data across all experiments, at the accession level. Contains `var_index`, `experiment_subdir`, and protein identifiers (aliases, gene names, or UniProt accessions).
- A user-specified target schema describing which output columns to produce.

**Output:**
- `Protein_resolved.csv` — all raw columns plus resolved columns (`feature_name`, `uniprot_id`, `protein_name`, `gene_name`, `organism`, `sequence`, `sequence_length`, `resolved`, `uid`). This is the full intermediate output for inspection and debugging.
- `ProteinSchema.parquet` — finalized against the target schema with correct types. Contains exactly the schema fields, no `resolved` column, no raw columns. Parquet preserves types (nullable ints, lists, bools) so the file can be loaded directly into LanceDB.
- `{fs}_standardized_var.csv` — per-experiment var CSV with the original var index and a `global_feature_uid` column mapping each feature to its resolved UID. Written in each experiment subdirectory.
- `resolver_reports/protein-resolver.md` — markdown report written in the working directory. Summarize inputs, output paths, resolved/unresolved counts, control detection summary, and any fields left blank in the finalized schema output.

## Reporting

Each run must write a markdown report to `resolver_reports/` in the working directory.

- Create the directory if it does not exist.
- Default report path: `resolver_reports/protein-resolver.md`
- Overwrite the report for the current run unless the caller asks for a different naming scheme.
- Include:
  - input file path(s)
  - output file path(s)
  - row counts and resolved/unresolved counts
  - isotype control detection summary
  - correction mappings or fallback logic used
  - any finalized schema fields left blank, with reasons

## Scripts

### `resolve_proteins.py` — Resolution

Auto-detects protein identifier column, separates isotype controls (IgG1, IgG2a, etc.) and other controls, resolves actual proteins via `resolve_proteins()`, and maps organism to scientific name via `resolve_organisms()`.

```bash
python .claude/skills/protein-resolver/scripts/resolve_proteins.py \
    <input_csv> <output_csv> \
    [--protein-col COL] [--organism ORG] \
    [--index-col COL] [--dry-run]
```

- `--protein-col`: Column with protein identifiers. Default: auto-detect from columns (`var_index`, `feature_name`, `protein`, etc.) then index.
- `--organism`: Organism for resolution. Default: `human`.
- `--index-col`: Override which CSV column becomes the DataFrame index. Use `--index-col none` to disable index handling.
- `--dry-run`: Print detected columns and planned operations without writing output.

#### Columns produced

| Column | Source |
|---|---|
| `feature_name` | Original protein alias from input |
| `uniprot_id` | `ProteinResolution.uniprot_id` or None |
| `protein_name` | Resolved protein name, or original value if unresolved |
| `gene_name` | `ProteinResolution.gene_name` or None |
| `organism` | Scientific name via `resolve_organisms()` (e.g., `"Homo sapiens"`) |
| `sequence` | `ProteinResolution.sequence` or None |
| `sequence_length` | `ProteinResolution.sequence_length` or None |
| `resolved` | Boolean — `True` if resolution succeeded |
| `uid` | Unique ID via `make_uid()` |

### `write_standardized_var.py` — Per-experiment standardized var CSVs

Reads the resolved CSV to build a `var_index → uid` mapping, then writes `{fs}_standardized_var.csv` in each experiment subdirectory containing the original var index and a `global_feature_uid` column.

```bash
python .claude/skills/protein-resolver/scripts/write_standardized_var.py \
    <accession_dir> \
    [--resolved-csv Protein_resolved.csv] \
    [--feature-space protein]
```

- `--resolved-csv`: Filename of the resolved CSV in the accession directory. Default: `GenomicFeatureSchema_resolved.csv` (override to `Protein_resolved.csv`).
- `--feature-space`: Feature space name used to find `{fs}_raw_var.csv` per experiment and name the output. Default: `gene_expression` (override to `protein`).

Experiment directories are auto-discovered by scanning for `{fs}_raw_var.csv` files under the accession directory.

### `finalize_features.py` — Schema finalization with type coercion

Takes the resolved CSV, adds any schema-specific columns, drops everything not in the schema (including `resolved`), coerces types (JSON lists, bools, numerics), and writes parquet with correct types so the output can be loaded directly into LanceDB without further manipulation.

Does NOT do per-row pydantic validation — type coercion + parquet schema enforcement is sufficient. Type errors surface at LanceDB insertion time.

```bash
python .claude/skills/protein-resolver/scripts/finalize_features.py \
    <resolved_csv> <output_parquet> <schema_module> <schema_class> \
    [--column KEY=VALUE ...]
```

- `--column KEY=VALUE`: Add a column. If VALUE is an existing column name, copies that column. If VALUE is `None` or `null` (case-insensitive), sets actual Python None. Otherwise uses VALUE as a constant for all rows.

Example:

```bash
python .claude/skills/protein-resolver/scripts/finalize_features.py \
    /tmp/GSE123/Protein_resolved.csv \
    /tmp/GSE123/ProteinSchema.parquet \
    lancell_examples.multimodal_perturbation_atlas.schema \
    ProteinSchema \
    --column feature_type=protein \
    --column feature_id=uniprot_id
```

---

## Workflow

### 1. Run the resolution script

```bash
python .claude/skills/protein-resolver/scripts/resolve_proteins.py \
    /path/to/Protein_raw.csv \
    /path/to/Protein_resolved.csv
```

Review the output for resolved/unresolved counts and isotype control detection.

### 2. Finalize against the target schema

Read the target schema to determine which columns need to be added, then run:

```bash
python .claude/skills/protein-resolver/scripts/finalize_features.py \
    /path/to/Protein_resolved.csv \
    /path/to/ProteinSchema.parquet \
    <schema_module> <schema_class> \
    --column feature_type=protein \
    --column feature_id=uniprot_id
```

The script coerces types and writes parquet — the output is ready for direct LanceDB ingestion.

### 3. Write per-experiment standardized var CSVs

After finalization, write per-experiment `{fs}_standardized_var.csv` files that map each experiment's var index to the resolved UIDs:

```bash
python .claude/skills/protein-resolver/scripts/write_standardized_var.py \
    /path/to/accession_dir \
    --resolved-csv Protein_resolved.csv \
    --feature-space protein
```

This requires the caller (preparer) to provide the accession directory path and feature space name.

### 4. Write the markdown report

After finalization, write `resolver_reports/protein-resolver.md` in the working directory with the run summary, control detection details, and blank-field audit.

Common cases:
- Use `--column feature_id=uniprot_id` when the schema wants a stable protein foreign key.
- Use literal constants such as `--column feature_type=protein` for schema-wide values.
- `sequence` and `sequence_length` are populated from the SwissProt reference DB when a UniProt ID resolves.
- `FeatureBaseSchema.global_index` is auto-generated at ingestion time, not during resolution.

---

## Resolution Strategy

All resolved columns follow the same principle: **never NaN unless there is genuinely no value**, and **always flag resolution status with a boolean `resolved` column.**

1. **Resolution succeeds** (`resolved_value` is not None) → use canonical values from `ProteinResolution` (e.g., `.uniprot_id`, `.protein_name`). Set `resolved=True`.
2. **Resolution fails** (`resolved_value` is None) → keep the original value for name fields (do not set to NaN), but set `resolved=False`. ID fields (`uniprot_id`) can be None when no mapping exists.
3. **Isotype controls** → `resolved=False`, protein identity fields (`uniprot_id`, `gene_name`, `sequence`, `sequence_length`) set to None. `protein_name` keeps the original alias.
4. **NaN only when no value exists** — e.g., a protein has no known gene name at all.

## Rules

- **Accession-level only.** This resolver operates at the accession level on `Protein_raw.csv`, not per-experiment. The raw CSV is already deduplicated on `var_index` by the preparer — the same protein appearing in multiple experiment subdirs should only have one row.
- **No `validated_` prefix.** Output columns use schema field names directly (e.g., `protein_name` not `validated_protein_name`). This is the project-wide convention for all resolver output.
- **Organism as scientific name.** Use `resolve_organisms()` to map common names to scientific names. Do not hardcode organism mappings.
- **Assign UIDs via `make_uid()`.** Every unique feature row gets a UID in the output.
- **One-step resolution.** Use `resolve_proteins()` directly. Do not attempt the old two-step alias→gene symbol→UniProt approach.
- **Isotype controls are NOT caught by `is_control_label()`.** Use the explicit isotype patterns defined in the resolve script (IgG1, IgG2a, IgG2b, IgG2c, IgM, IgA, IgD, IgE, plus "isotype", "mouse-igg*", "rat-igg*" prefixes).
- **Assume human** unless the dataset metadata specifies another organism.
- **Never set name columns to NaN for failed resolution.** Use the original value and set `resolved=False`.
- **Output columns may overwrite raw columns.** In particular, resolved `organism` replaces any raw `organism` column.
- **Index collisions are renamed.** If the input index name collides with an output column such as `feature_name`, the script renames the index to `raw_<name>` before writing.
- **Column names follow the user's schema.** Do not assume specific column names — use whatever the user's target schema specifies.
- **Two output files.** `Protein_resolved.csv` retains `resolved` for inspection. `ProteinSchema.parquet` is type-coerced and production-ready for direct LanceDB ingestion.
