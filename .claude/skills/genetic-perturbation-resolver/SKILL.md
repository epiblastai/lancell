---
name: genetic-perturbation-resolver
description: Use this skill to standardize genetic perturbation targets in dataframes — gene names, guide RNA sequences, or genomic coordinates - and to lookup missing metadata. Expects a dataframe with details about genetic perturbations. Handles control detection, combinatorial splitting, perturbation method classification, and guide RNA alignment via BLAT.
---

# Genetic Perturbation Resolver

Resolve genetic perturbation targets and schema fields at the accession level: resolve `GeneticPerturbationSchema_raw.csv` → enrich missing schema fields from available sources → assign UIDs → write `GeneticPerturbationSchema_resolved.csv`.

Handles three input types that may co-exist in a single dataset:

1. **Gene names/symbols** — Target gene names (e.g., "TP53", "BRCA1").
2. **Guide RNA sequences** — Raw guide sequences from CRISPR screens. Aligns via BLAT to get genomic coordinates, then annotates with overlapping genes and target context. 20bp is the minimum for BLAT resolution and generally works great for resolving guide RNAs. This is because guide RNAs are chosen to exactly match the reference genome and to be unique within it.
3. **Genomic coordinates** — Pre-computed target regions (e.g., enhancer/promoter-targeting screens). Annotates with overlapping genes and target context without BLAT.

## Interface

**Input:**
- `GeneticPerturbationSchema_raw.csv` — consolidated perturbation data across all experiments, at the accession level. It may be sparse and does not need to be fully enriched by the preparer.
- A user-specified target schema.
- Experiment directories with per-experiment obs CSVs, the column linking cells to perturbations, and the feature space name.
- Any supplementary files or file paths available for enrichment, such as guide libraries, reagent manifests, vendor sheets, or publication-derived tables.

**Output:**

*Accession-level (global foreign key table):*
- `GeneticPerturbationSchema_resolved.csv` — all raw columns plus resolved columns, UIDs, `resolved` boolean. Full intermediate output for inspection and debugging.
- `GeneticPerturbationSchema.parquet` — finalized against the target schema with correct types. Contains exactly the schema fields, no `resolved` column, no raw columns. Parquet preserves types so the file can be loaded directly into LanceDB.
- `resolver_reports/genetic-perturbation-resolver.md` — markdown report written in the working directory. Summarize enrichment sources used, outputs written, counts, unresolved items, and a field-by-field blank justification audit.

*Per-experiment (obs-level fragment):*
- `{feature_space}_fragment_perturbation_obs.csv` — one per experiment directory. Contains perturbation-related obs columns: `is_negative_control`, `negative_control_type`, `perturbation_uids`, `perturbation_types`, `perturbation_concentrations_um`, `perturbation_durations_hr`. The cell barcode column is preserved as the index for joining.

**Column naming:** No `validated_` prefix. Use schema field names directly.

## Ownership

This resolver owns completion of the target perturbation schema. Do not assume the preparer has already filled dataset-specific fields.

The resolver must:

1. Inspect the target schema and determine every field that must be populated.
2. Inspect the raw CSV, experiment directories, and any supplementary files or paths provided by the caller.
3. Fill every schema field that can be derived from the available evidence.
4. Leave a field blank only after attempting resolution and documenting why the value is unavailable or unjustified.

The preparer may pass helpful context, but the resolver is responsible for the final content of `GeneticPerturbationSchema_resolved.csv` and `GeneticPerturbationSchema.parquet`.

## Reporting

Each run must write a markdown report to `resolver_reports/` in the working directory.

- Create the directory if it does not exist.
- Default report path: `resolver_reports/genetic-perturbation-resolver.md`
- Overwrite the report for the current run unless the caller asks for a different naming scheme.
- Include:
  - input file path(s)
  - supplementary files inspected
  - output file path(s)
  - row counts and resolved/unresolved counts
  - enrichment decisions and join keys used
  - schema field completeness audit, including reasons for blanks

## Imports

```python
from lancell.standardization import (
    resolve_genes,
    resolve_guide_sequences,
    annotate_genomic_coordinates,
    is_control_label,
    detect_control_labels,
    detect_negative_control_type,
    parse_combinatorial_perturbations,
    classify_perturbation_method,
    GeneticPerturbationType,
)
from lancell.standardization.assemblies import get_assembly_report
from lancell.standardization.types import GeneResolution, GuideRnaResolution, ResolutionReport
from lancell.schema import make_uid
```

## Scripts

### `scripts/resolve_genes.py`

Handles the standard gene-name workflow: optional reagent splitting, control detection, method classification, gene resolution via `resolve_genes`, Ensembl ID cross-checking, UID assignment, and CSV output.

```
python .claude/skills/genetic-perturbation-resolver/scripts/resolve_genes.py \
    <input_csv> <gene_column> <method> \
    [--organism human] \
    [--ensembl-column ensembl_gene_id] \
    [--split-column reagent_id] \
    [--split-delimiter "|"] \
    [--output-dir <dir>]
```

| Argument | Description |
|---|---|
| `input_csv` | Path to `GeneticPerturbationSchema_raw.csv` (must have `index_col=0`) |
| `gene_column` | Column containing gene names / control labels |
| `method` | Perturbation method string (e.g. `CRISPRi`, `CRISPRko`, `siRNA`) |
| `--organism` | Organism for gene resolution (default: `human`) |
| `--ensembl-column` | Column with existing Ensembl IDs; mismatches are reported |
| `--split-column` | Column to split into one reagent per row before resolution. If omitted, no row-splitting is performed. |
| `--split-delimiter` | Delimiter for `--split-column` (default: `|`) |
| `--output-dir` | Output directory (default: same as input) |

The script writes `GeneticPerturbationSchema_resolved.csv` with these columns populated: `perturbation_type`, `intended_gene_name`, `intended_ensembl_gene_id`, `reagent_id` (from index), `uid`, `resolved`. It may leave placeholder `None` columns for fields that still require dataset-specific enrichment.

**After running the script**, the resolver must continue enrichment until every target schema field is either populated or explicitly justified as blank. Placeholder `None` values are not a stopping point.

### `finalize_perturbations.py` — Schema finalization with type coercion (shared)

Uses the shared `gene-resolver/scripts/finalize_features.py` script. Takes the resolved CSV, drops everything not in the schema (including `resolved` and raw columns), coerces types, and writes parquet with correct types for direct LanceDB ingestion.

```bash
python .claude/skills/gene-resolver/scripts/finalize_features.py \
    <resolved_csv> <output_parquet> <schema_module> <schema_class> \
    [--column KEY=VALUE ...]
```

- `--column KEY=VALUE`: Add a column. If VALUE is an existing column name, copies that column. If VALUE is `None` or `null` (case-insensitive), sets actual Python None. Otherwise uses VALUE as a constant for all rows.

Example:

```bash
python .claude/skills/gene-resolver/scripts/finalize_features.py \
    /tmp/GSE123/GeneticPerturbationSchema_resolved.csv \
    /tmp/GSE123/GeneticPerturbationSchema.parquet \
    lancell_examples.multimodal_perturbation_atlas.schema \
    GeneticPerturbationSchema
```

---

## Core Constraints

- **One perturbation per row.** Each accession-level row must represent exactly one reagent.
- **Controls are not perturbations.** Control labels map to `None` in perturbation target fields and drive `is_negative_control` at the obs level.
- **Do not guess required guide-level fields.** If the schema requires `guide_sequence`, coordinates, or strand and the data is missing, stop and ask the user unless they explicitly approve nulls.
- **Use schema field names directly.** Do not introduce `validated_` prefixes or ad hoc column names.
- **Resolver owns enrichment.** If a field can be filled from supplementary files, raw identifiers, publication text, or deterministic parsing, do that work here rather than assuming the preparer already did it.
- **Every blank needs a reason.** In the final report, enumerate any schema fields left blank and justify why they were not be filled.

## Resolution Workflow

### A0. Inspect schema and enrichment sources first

Before running any resolution step:

1. Read the target `GeneticPerturbationSchema`.
2. List its fields and identify which are already present in `GeneticPerturbationSchema_raw.csv`.
3. Inspect any supplementary files or file paths provided by the caller, especially guide libraries and reagent manifests.
4. Determine which fields require:
   - direct pass-through from raw columns
   - deterministic parsing from reagent identifiers
   - lookup from supplementary files
   - biological resolution via `resolve_genes`, `resolve_guide_sequences`, or `annotate_genomic_coordinates`
5. Keep notes so the final report can justify any remaining blanks.

### A1–A5, A8: Gene resolution (use the script)

For the standard gene-name workflow, run `resolve_genes.py` as above. Use `--split-column` when reagent identifiers are paired or pipe-delimited. In paired-guide screens, split on reagent IDs, not just gene names, because the gene column can stay constant across both guides.

1. **Load & inspect** — reads the raw CSV, identifies columns
2. **Control detection** — `detect_control_labels` on the gene column, plus numbered-prefix check
3. **Optional row splitting** — if a reagent column contains paired entries such as `guideA|guideB`, split that column first so the output becomes one reagent per row
4. **Classify perturbation method** — `classify_perturbation_method` on the method string
5. **Resolve genes** — `resolve_genes` on unique non-control targets; if `--ensembl-column` is present, report mismatches and let the resolver's current Ensembl IDs take precedence unless the dataset explicitly requires a pinned release
6. **Build output** — maps results to schema fields, assigns UIDs, writes CSV

If you split rows first, expect the row count to increase. For dual-guide pairs it will usually double. If the accession spans multiple experiments, deduplicate on the reagent key after splitting so the accession-level FK table has one row per unique reagent.

### A6. Perform resolver-owned enrichment for all remaining schema fields

After the base gene-resolution script runs, inspect the partially resolved CSV and fill the remaining schema fields from the best available evidence source:

- `guide_sequence`:
  - Prefer a supplementary guide library or reagent manifest.
  - Join on `reagent_id`, guide ID, or other dataset-specific reagent keys.
  - If multiple possible joins exist, prefer the one that preserves one reagent per row and document the join key.
- `library_name`:
  - Prefer the library metadata file itself, then raw columns, then publication text if needed.
- `target_chromosome`:
  - BLAT and `GuideRnaResolution` return UCSC chromosome names (e.g., `chr1`). The schema may expect a different representation such as a GenBank accession (e.g., `CM000663.2`). Use `get_assembly_report()` from `lancell.standardization.assemblies` to convert:
    ```python
    from lancell.standardization.assemblies import get_assembly_report
    report = get_assembly_report("human", "GRCh38")
    seq = report.lookup("chr1")  # accepts UCSC, bare, GenBank, or RefSeq names
    seq.genbank_accession  # "CM000663.2"
    seq.ucsc_name          # "chr1"
    seq.sequence_name      # "1"
    ```
  - Check the target schema's docstring/comment for the expected naming convention. Populate `target_chromosome` accordingly using the appropriate `AssemblySequence` field.
- `target_start`, `target_end`, `target_strand`:
  - Prefer explicit columns from a guide library or manifest.
  - If absent, deterministically parse coordinates from reagent IDs when the identifier format encodes them.
- `target_context`:
  - Prefer explicit annotation from the library.
  - Otherwise infer from `resolve_guide_sequences()` or `annotate_genomic_coordinates()`.
  - For transcript-targeted CRISPRi screens, `promoter` is an acceptable fallback only when supported by the dataset design.
- `target_sequence_uid`:
  - Populate when the target sequence can be mapped unambiguously to a `ReferenceSequenceSchema` record already available to the workflow.
  - Otherwise leave null and justify it in the report.

Do not finalize while a schema field is still blank merely because the preparer omitted it. The resolver must inspect the available evidence itself.

### A7. Resolve by guide RNA sequence (if applicable)

```python
guide_col = "<guide_sequence_column>"
unique_guides = raw_df[guide_col].dropna().unique().tolist()
report = resolve_guide_sequences(unique_guides, organism="human")
print(f"Resolved: {report.resolved}/{report.total}, Ambiguous: {report.ambiguous}")
```

Deduplicate guide sequences before BLAT-backed resolution because guides are reused across many cells and BLAT is rate-limited. After inferring coordinates or target context, spot-check 3-5 guides with `resolve_guide_sequences()`. 

### A8. Resolve by genomic coordinates (if applicable)

```python
coordinates = []
for _, row in raw_df[raw_df["<chr_col>"].notna()].iterrows():
    coordinates.append({
        "chromosome": row["<chr_col>"],
        "start": int(row["<start_col>"]),
        "end": int(row["<end_col>"]),
        "strand": row.get("<strand_col>"),
    })

report = annotate_genomic_coordinates(coordinates, organism="human")
```

For transcript-targeted CRISPRi guides, use `target_context=promoter` unless the dataset provides stronger evidence for another context.

### A9. Validate completeness before finalization

Before finalizing:

1. Compare the resolved CSV against the target schema field list.
2. For each schema field, confirm one of:
   - populated for all applicable rows
   - intentionally null for control rows only
   - partially or fully blank with a documented justification
3. If a required field is blank and you have not yet inspected the obvious enrichment source for it, go back and do that work.

### A10. Finalize against the target schema

After resolution and any dataset-specific enrichment (A6/A7), run the finalize script:

```bash
python .claude/skills/gene-resolver/scripts/finalize_features.py \
    /path/to/GeneticPerturbationSchema_resolved.csv \
    /path/to/GeneticPerturbationSchema.parquet \
    <schema_module> <schema_class>
```

The script coerces types and writes parquet — the output is ready for direct LanceDB ingestion.

### A11. Write the markdown report

After finalization and obs-fragment writing, write `resolver_reports/genetic-perturbation-resolver.md` in the working directory with the run summary and field-completeness audit.

---

## Resolution Strategy

All resolved columns follow the same principle: **never NaN unless there is genuinely no value**, and **always flag resolution status with a boolean `resolved` column.**

1. **Resolution succeeds** → use canonical values. Set `resolved=True`.
2. **Resolution fails** (gene name unresolved, guide fails BLAT, coordinates have no gene overlap) → keep original values where possible, set `resolved=False`.
3. **NaN only when no value exists** — e.g., a cell has no perturbation target.
4. **Control labels → None** — "non-targeting", "NegCtrl0", etc. become None in perturbation columns (they inform `is_negative_control`, not the gene field).

## Obs-Level Fragment Workflow (B1–B4)

After the global foreign key table is resolved and finalized (A1–A10), write per-experiment obs fragments that map each cell to its perturbation UIDs and control status.

The preparer provides:
- A list of experiment directories (each containing `{feature_space}_raw_obs.csv`)
- The obs column that links cells to perturbations (e.g., `sgID_AB`, `guide_id`, `perturbation`)
- The feature space name (e.g., `gene_expression`)
- Dose/duration columns and their units, if applicable

### B1. Load the resolved foreign key table

Load `GeneticPerturbationSchema.parquet` (the finalized table with UIDs). Build a lookup from reagent identifiers such as `reagent_id`, `guide_sequence`, or `intended_gene_name` to `uid` values.

### B2. Map cells to perturbation UIDs

For each experiment directory:

1. Read `{feature_space}_raw_obs.csv`
2. Parse the perturbation column. Handle:
   - **Pipe-delimited dual/multi-guide pairs** (e.g., `guideA|guideB`) — split and look up each independently
   - **Single perturbation per cell** — direct lookup
   - **Combinatorial perturbations** — split by delimiter, look up each
3. For each cell, build:
   - `perturbation_uids`: list of UIDs (one per reagent acting on the cell)
   - `perturbation_types`: list of `"genetic_perturbation"` (matching length)
   - `perturbation_concentrations_um`: list of concentrations if available, else `[-1]` per reagent
   - `perturbation_durations_hr`: list of durations if available, else `[-1]` per reagent

### B3. Detect controls at the cell level

Use `detect_control_labels` and `is_control_label` on the perturbation column values:

- `is_negative_control = True` only if **all** perturbations for that cell are control-type (non-targeting, intergenic, etc.)
- `negative_control_type`: the control label (e.g., `"non-targeting"`) if `is_negative_control` is True, else None
- For control cells, `perturbation_uids` and `perturbation_types` should be None (controls are not perturbations)

NaN or missing perturbation values do not imply control.

### B4. Write the fragment

Write `{feature_space}_fragment_perturbation_obs.csv` in the experiment directory with columns:
- The cell barcode column (as index or first column, for joining)
- `is_negative_control`
- `negative_control_type`
- `perturbation_uids` (JSON-serialized list)
- `perturbation_types` (JSON-serialized list)
- `perturbation_concentrations_um` (JSON-serialized list)
- `perturbation_durations_hr` (JSON-serialized list)

Lists should be serialized as JSON strings so they survive CSV round-tripping.

---

## Notes

- Save the accession-level CSV after each added column if the environment is fragile or the table is expensive to rebuild.
- `GuideRnaResolution` objects expose `intended_gene_name`, `intended_ensembl_gene_id`, and `target_context`.
- The skill produces two accession-level files: `GeneticPerturbationSchema_resolved.csv` for inspection and `GeneticPerturbationSchema.parquet` for type-coerced, LanceDB-ready output.
- If delimiters or control labels are ambiguous, ask the user instead of guessing.
- Final reports must include a short field-completeness audit for the target schema. For each schema field, state the source used to populate it or the reason it remains blank.
