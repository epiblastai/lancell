---
name: ontology-resolver
description: Resolve free-text biological metadata (cell type, tissue, disease, organism, assay, development stage, ethnicity, sex, cell line) to canonical ontology terms and CURIEs using lancell.standardization.resolve_ontology_terms(). Handles control detection, OLS4 fallback for cell lines, organism-aware development stages, and hardcoded sex terms.
---

# Ontology Resolver

Resolve free-text biological metadata values to canonical ontology terms with CELLxGENE-compatible IDs. Covers 9 entity types across 8 ontologies:

| Entity | Ontology | Prefix |
|---|---|---|
| cell_type | Cell Ontology | CL |
| tissue | UBERON | UBERON |
| disease | MONDO | MONDO |
| organism | NCBITaxon | NCBITaxon |
| assay | EFO | EFO |
| development_stage | HsapDv / MmusDv | HsapDv, MmusDv |
| ethnicity | HANCESTRO | HANCESTRO |
| sex | PATO | PATO |
| cell_line | CLO | CLO |

## Interface

This resolver operates in **Phase B** (per-experiment obs resolution). It reads raw obs CSVs from an experiment directory and writes a fragment CSV with resolved ontology columns.

**Input:**
- Raw obs CSV (`{fs}_raw_obs.csv`) — **read-only**, do not modify this file
- A mapping of column names to `OntologyEntity` types (provided by the caller or derived from schema field classification)
- Organism context (required for `development_stage`, helpful for others)
- Output fragment path (`{fs}_fragment_ontology_obs.csv`)

**Output:**
- A new fragment CSV file (`{fs}_fragment_ontology_obs.csv`) indexed by the same index as the raw obs, containing:
  - One column per schema field the resolver is responsible for, using the exact schema field name (e.g., `cell_type`, `tissue`, `disease`) — resolved to canonical ontology term names
  - `ontology_resolved` boolean indicating whether all ontology fields resolved successfully
- A markdown report at `resolver_reports/ontology-resolver.md` in the working directory summarizing the experiment, fields resolved, unresolved values, and any omitted output columns with reasons.

**Column naming:** Output columns must exactly match the schema field names the resolver is told to fill. No `_ontology_id` suffixes, no `validated_` prefixes. For organism fields, output the resolved scientific name (e.g., `"Homo sapiens"`, `"Mus musculus"`) — do not convert to common names. Ontology CURIEs are recorded in the markdown report for debugging but do not appear in the fragment CSV.

## Reporting

Each run must write a markdown report to `resolver_reports/` in the working directory.

- Create the directory if it does not exist.
- Default report path: `resolver_reports/ontology-resolver.md`
- Overwrite the report for the current run unless the caller asks for a different naming scheme.
- Include:
  - experiment directory and input file path
  - output fragment path
  - ontology fields attempted
  - resolved/unresolved counts per field
  - correction mappings or manual normalizations used
  - any omitted or blank output fields, with reasons

## Imports

```python
from lancell.standardization import (
    OntologyEntity,
    resolve_ontology_terms,
    is_control_label,
    detect_control_labels,
    detect_negative_control_type,
)
from lancell.standardization.types import OntologyResolution, ResolutionReport
```

## Scripts

### `scripts/resolve_ontology.py`

Handles the standard ontology resolution workflow: control detection, resolution via `resolve_ontology_terms`, column writing, and report generation.

```
python .claude/skills/ontology-resolver/scripts/resolve_ontology.py \
    <input_csv> <output_csv> \
    --field <obs_col>:<schema_field>:<entity> [...] \
    [--organism human] \
    [--corrections corrections.json] \
    [--report-dir resolver_reports]
```

| Argument | Description |
|---|---|
| `input_csv` | Path to `{fs}_raw_obs.csv` (must have `index_col=0`) |
| `output_csv` | Path to fragment output (`{fs}_fragment_ontology_obs.csv`) |
| `--field` | Repeatable. Format: `obs_column:schema_field:ENTITY_TYPE`. Entity is the `OntologyEntity` enum name (e.g., `CELL_TYPE`, `TISSUE`, `DISEASE`). |
| `--organism` | Organism for development_stage resolution (default: `human`) |
| `--corrections` | JSON file with correction mappings (see below) |
| `--report-dir` | Output directory for markdown report (default: `resolver_reports` in input dir) |

The script writes the fragment CSV with one column per schema field (exact name match) plus the `ontology_resolved` boolean. No `_ontology_id` suffix columns are written — ontology CURIEs are included in the markdown report only. It prints per-field resolution stats and unresolved values to stdout.

### Corrections format

A JSON file mapping obs column names to `{original: corrected}` dictionaries:

```json
{
    "cell_type_annotation": {
        "T-cell": "T cell",
        "B-cell": "B cell",
        "Monocyte/Macrophage": "monocyte"
    }
}
```

Corrections are applied before resolution. Build these after reviewing unresolved values from a first pass.

### `finalize_features.py` — Schema finalization with type coercion (shared)

Uses the shared `gene-resolver/scripts/finalize_features.py` script if the ontology fragment needs to be written to parquet for direct LanceDB ingestion.

```bash
python .claude/skills/gene-resolver/scripts/finalize_features.py \
    <resolved_csv> <output_parquet> <schema_module> <schema_class> \
    [--column KEY=VALUE ...]
```

---

## Agent Workflow

### 1. Inspect raw obs and determine field mappings

Read the raw obs CSV. Identify which columns contain ontology-resolvable metadata and map each to a schema field name and `OntologyEntity`:

```python
import pandas as pd
raw_obs = pd.read_csv("<experiment_dir>/<fs>_raw_obs.csv", index_col=0)
for col in raw_obs.columns:
    unique = raw_obs[col].dropna().unique()
    print(f"{col}: {len(unique)} unique — {list(unique[:5])}")
```

Not every dataset has every ontology field. If a column is absent or entirely null, skip it.

### 2. Determine organism context

Identify the organism from the dataset metadata or a dedicated organism column. Pass it to `--organism` for correct development_stage dispatch (HsapDv for human, MmusDv for mouse).

### 3. Run the script

```bash
python .claude/skills/ontology-resolver/scripts/resolve_ontology.py \
    /path/to/{fs}_raw_obs.csv \
    /path/to/{fs}_fragment_ontology_obs.csv \
    --field cell_type_annotation:cell_type:CELL_TYPE \
    --field donor_tissue:tissue:TISSUE \
    --field diagnosis:disease:DISEASE \
    --organism human
```

### 4. Review unresolved values

The script prints unresolved values to stdout. For each:

- **Case/whitespace issues** — already handled by `resolve_ontology_terms`, but check for unusual Unicode
- **Abbreviations** — `"T cell"` vs `"T-cell"` vs `"T lymphocyte"`. Build corrections.
- **Concatenated annotations** — `"CD4+ T cell (activated)"` may need qualifier stripping
- **Dataset-specific labels** — `"Cluster_5"`, `"Unknown"`, `"Other"` are not ontology terms; accept as unresolved
- **Near-misses** — use hierarchy navigation to find the correct term:

```python
from lancell.standardization import get_ontology_descendants
descendants = get_ontology_descendants("CL:0000084", OntologyEntity.CELL_TYPE, max_depth=2)
```

### 5. Re-run with corrections (if needed)

Build a corrections JSON and re-run:

```bash
python .claude/skills/ontology-resolver/scripts/resolve_ontology.py \
    /path/to/{fs}_raw_obs.csv \
    /path/to/{fs}_fragment_ontology_obs.csv \
    --field cell_type_annotation:cell_type:CELL_TYPE \
    --corrections /path/to/corrections.json \
    --organism human
```

### 6. Verify fragment output

Confirm the fragment has the expected columns and the `ontology_resolved` counts are acceptable.

---

## Entity-Specific Notes

**organism** — Common values: `"Homo sapiens"`, `"Mus musculus"`. Use the resolved scientific name directly — do not convert to common names.

**development_stage** — Pass `--organism` to select the correct ontology (HsapDv for human, MmusDv for mouse). Without it, both are searched and may produce wrong matches.

**sex** — Only 3 canonical values: `"female"` (PATO:0000383), `"male"` (PATO:0000384), `"unknown"` (PATO:0000461). `"other"` maps to `"unknown sex"`. Resolution is hardcoded (not from the local DB).

**cell_line** — Uses Cellosaurus local DB with FTS fuzzy search fallback. Exact matches get confidence 1.0; fuzzy matches get confidence 0.7. Flag fuzzy matches for user review.

**assay** — Free-text assay names (e.g., `"10x 3' v3"`, `"Smart-seq2"`) must match EFO terms. Many GEO assay descriptions don't match EFO exactly — investigate failures carefully.

---

## Rules

- **Read-only input.** Never modify the `_raw_obs.csv` file. Write all output to the fragment file.
- **Exact schema field names only.** Output columns must match schema field names exactly. No `_ontology_id` suffixes, no `validated_` prefixes.
- **Organism as scientific name.** Do not convert to common names.
- **Use `resolve_ontology_terms()` for all entities.** It handles dispatch to DB lookup, OLS4, or hardcoded tables internally.
- **Pass `organism` for development_stage.** Without it, wrong matches are possible.
- **Do not derive control fields from ontology columns.** `is_negative_control` and `negative_control_type` are perturbation-level concepts populated by perturbation resolvers.
- **Use `detect_control_labels()` for control detection.** Do not hardcode control label sets.
- **Never set output columns to NaN for failed resolution.** Keep the original value in the schema field.
- **Always write `ontology_resolved` boolean.** Named `ontology_resolved` to avoid collision during assembly.
- **Save after each column pair** to prevent losing work.
- **Never modify h5ad files.** All validated data goes into the fragment CSV only.
- **Investigate failures before giving up.** Build correction mappings for close-but-not-exact values.
- **Dataset-specific labels are acceptable as unresolved.** Cluster IDs, "Unknown", "Other" are not ontology terms.
- **Ask before guessing.** If a column's entity type is ambiguous, ask the user.

## Resolution Strategy

1. **Resolution succeeds** — use the canonical ontology name in the schema field. Row is resolved. CURIE is recorded in the report.
2. **Resolution fails** — keep the original value in the schema field. Row is unresolved.
3. **NaN only when no value exists** — schema field is NaN.
4. **Control labels → None** — Control values map to None in the schema field.
