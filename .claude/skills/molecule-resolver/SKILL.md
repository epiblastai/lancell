---
name: molecule-resolver
description: Resolve chemical compound names, SMILES, or CIDs to canonical structures for SmallMoleculeSchema using lancell.standardization.resolve_molecules(). Handles name cleanup, control label detection, PubChem/ChEMBL resolution, and SMILES canonicalization. Use when a dataset has small molecule perturbation columns.
---

# Molecule Resolver

Resolve chemical compound identifiers and populate `SmallMoleculeSchema` registry records for downstream ingestion. Resolve `SmallMolecule_raw.csv` → enrich missing schema fields from available sources → assign UIDs → write `SmallMolecule_resolved.csv` and `SmallMoleculeSchema.parquet`.

Handles two input types that may co-exist in a single dataset:

1. **Compound names** — Common or trade names (e.g., "Imatinib", "Dexamethasone").
2. **SMILES strings** — Structural representations for compounds that fail name resolution.

## Interface

**Input:**
- `SmallMolecule_raw.csv` — consolidated compound data across all experiments, at the accession level. It may be sparse and does not need to be fully enriched by the preparer.
- A user-specified target schema.
- Any supplementary files or file paths available for enrichment, such as vendor catalogs, SMILES tables, or publication-derived tables.

**Output:**

*Accession-level (global foreign key table):*
- `SmallMolecule_resolved.csv` — all raw columns plus resolved columns, UIDs, `resolved` boolean. Full intermediate output for inspection and debugging.
- `SmallMoleculeSchema.parquet` — finalized against the target schema with correct types. Contains exactly the schema fields, no `resolved` column, no raw columns. Parquet preserves types so the file can be loaded directly into LanceDB.
- `resolver_reports/molecule-resolver.md` — markdown report written in the working directory. Summarize enrichment sources used, outputs written, counts, unresolved items, and a field-by-field blank justification audit.

*Per-experiment (obs-level fragment):*
- `{fs}_fragment_molecule_obs.csv` — one per experiment directory. Contains perturbation-related obs columns using the `|` convention: `perturbation_uids|SmallMolecule`, `perturbation_types|SmallMolecule`, `perturbation_concentrations_um|SmallMolecule`, `is_negative_control|SmallMolecule`, `negative_control_type|SmallMolecule`. The cell barcode column is preserved as the index for joining.

**Column naming:** No `validated_` prefix. Use schema field names directly.

## Ownership

This resolver owns completion of the target small molecule schema. Do not assume the preparer has already filled dataset-specific fields.

The resolver must:

1. Inspect the target schema and determine every field that must be populated.
2. Inspect the raw CSV and any supplementary files or paths provided by the caller.
3. Fill every schema field that can be derived from the available evidence.
4. Leave a field blank only after attempting resolution and documenting why the value is unavailable or unjustified.

The preparer may pass helpful context, but the resolver is responsible for the final content of `SmallMolecule_resolved.csv` and `SmallMoleculeSchema.parquet`.

## Reporting

Each run must write a markdown report to `resolver_reports/` in the working directory.

- Create the directory if it does not exist.
- Default report path: `resolver_reports/molecule-resolver.md`
- Overwrite the report for the current run unless the caller asks for a different naming scheme.
- Include:
  - input file path(s)
  - supplementary files inspected
  - output file path(s)
  - row counts and resolved/unresolved counts
  - control labels detected
  - correction mappings or fallback logic used
  - schema field completeness audit, including reasons for blanks

## Imports

```python
from lancell.standardization import (
    resolve_molecules,
    is_control_compound,
    is_control_label,
    detect_control_labels,
    detect_negative_control_type,
)
from lancell.standardization.types import MoleculeResolution, ResolutionReport
from lancell.schema import make_uid
```

## Scripts

### `scripts/resolve_molecules.py`

Handles the standard compound-name workflow: control detection, molecule resolution via `resolve_molecules`, UID assignment, and CSV output.

```
python .claude/skills/molecule-resolver/scripts/resolve_molecules.py \
    <input_csv> <compound_column> \
    [--smiles-column COL] \
    [--vendor-column COL] \
    [--catalog-column COL] \
    [--output-dir <dir>]
```

| Argument | Description |
|---|---|
| `input_csv` | Path to `SmallMolecule_raw.csv` (must have `index_col=0`) |
| `compound_column` | Column containing compound names / control labels |
| `--smiles-column` | Column with SMILES strings to carry through |
| `--vendor-column` | Column with vendor names to carry through |
| `--catalog-column` | Column with catalog numbers to carry through |
| `--output-dir` | Output directory (default: same as input) |

The script writes `SmallMolecule_resolved.csv` with these columns populated: `name`, `smiles`, `pubchem_cid`, `iupac_name`, `inchi_key`, `chembl_id`, `uid`, `resolved`. It may leave placeholder `None` for fields that still require dataset-specific enrichment (e.g., `vendor`, `catalog_number`).

**After running the script**, the resolver must continue enrichment until every target schema field is either populated or explicitly justified as blank. Placeholder `None` values are not a stopping point.

### `finalize_features.py` — Schema finalization with type coercion (shared)

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
    /tmp/GSE123/SmallMolecule_resolved.csv \
    /tmp/GSE123/SmallMoleculeSchema.parquet \
    lancell_examples.multimodal_perturbation_atlas.schema \
    SmallMoleculeSchema
```

---

## Core Constraints

- **One compound per row.** Each accession-level row must represent exactly one compound.
- **Controls are not compounds.** Control labels map to `None` in compound identity fields and drive `is_negative_control` at the obs level.
- **Use schema field names directly.** Do not introduce `validated_` prefixes or ad hoc column names.
- **Resolver owns enrichment.** If a field can be filled from supplementary files, raw identifiers, publication text, or deterministic parsing, do that work here rather than assuming the preparer already did it.
- **Every blank needs a reason.** In the final report, enumerate any schema fields left blank and justify why they could not be filled safely.

## Resolution Workflow

### A0. Inspect schema and enrichment sources first

Before running any resolution step:

1. Read the target `SmallMoleculeSchema`.
2. List its fields and identify which are already present in `SmallMolecule_raw.csv`.
3. Inspect any supplementary files or file paths provided by the caller (vendor catalogs, SMILES tables, etc.).
4. Determine which fields require:
   - direct pass-through from raw columns
   - lookup from supplementary files
   - resolution via `resolve_molecules`
5. Keep notes so the final report can justify any remaining blanks.

### A1–A3: Molecule resolution (use the script)

For the standard compound-name workflow, run `resolve_molecules.py` as above.

1. **Load & inspect** — reads the raw CSV, identifies columns
2. **Control detection** — `detect_control_labels` on the compound column
3. **Resolve molecules** — `resolve_molecules` on unique non-control compounds; prints resolved/unresolved counts

If the dataset provides compounds by CID or SMILES rather than name, the resolver should handle this in a follow-up step (A4) rather than via the script.

### A4. Perform resolver-owned enrichment for all remaining schema fields

After the base resolution script runs, inspect the partially resolved CSV and fill the remaining schema fields from the best available evidence source:

**Investigate failures.** Since `resolve_molecules` already tries cleaned names, PubChem, and ChEMBL in sequence, unresolved names are genuinely problematic. Common issues:

- **Stray characters:** `Glesatinib?(MGCD265)` -> `Glesatinib`
- **Parenthetical aliases:** `Abexinostat (PCI-24781)` -> `Abexinostat`
- **Underscore-joined identifiers:** `Drug_123` -> `Drug`

Build a correction mapping and re-resolve:

```python
corrections = {
    "Glesatinib?(MGCD265)": "Glesatinib",
    "Tucidinostat (Chidamide)": "Tucidinostat",
}

corrected_names = list(set(corrections.values()))
correction_report = resolve_molecules(corrected_names, input_type="name")

resolution_map = {res.input_value: res for res in report.results if res.resolved_value is not None}
for orig, fixed in corrections.items():
    for res in correction_report.results:
        if res.input_value == fixed and res.resolved_value is not None:
            resolution_map[orig] = res
            break
```

**SMILES fallback:** If the dataset provides SMILES strings and some names remain unresolved:

```python
smiles_for_unresolved = [smiles_map[name] for name in still_unresolved if name in smiles_map]
if smiles_for_unresolved:
    smiles_report = resolve_molecules(smiles_for_unresolved, input_type="smiles")
```

- `vendor`:
  - Prefer supplementary vendor metadata or raw columns.
  - Use `--vendor-column` in the script if available.
- `catalog_number`:
  - Prefer supplementary catalog data or raw columns.
  - Use `--catalog-column` in the script if available.

Do not finalize while a schema field is still blank merely because the preparer omitted it. The resolver must inspect the available evidence itself.

### A5. Validate completeness before finalization

Before finalizing:

1. Compare the resolved CSV against the target schema field list.
2. For each schema field, confirm one of:
   - populated for all applicable rows
   - intentionally null for control rows only
   - partially or fully blank with a documented justification
3. If a required field is blank and you have not yet inspected the obvious enrichment source for it, go back and do that work.

### A6. Finalize against the target schema

After resolution and any dataset-specific enrichment (A4), run the finalize script:

```bash
python .claude/skills/gene-resolver/scripts/finalize_features.py \
    /path/to/SmallMolecule_resolved.csv \
    /path/to/SmallMoleculeSchema.parquet \
    <schema_module> <schema_class>
```

The script coerces types and writes parquet — the output is ready for direct LanceDB ingestion.

### A7. Write the markdown report

After finalization and obs-fragment writing, write `resolver_reports/molecule-resolver.md` in the working directory with the run summary and field-completeness audit.

---

## Phase B: Per-Experiment Obs Fragments

### B1. Load the resolved foreign key table

Load `SmallMoleculeSchema.parquet` (the finalized table with UIDs). Build a lookup from compound identifiers such as `name` or `pubchem_cid` to `uid` values.

```python
accession_dir = Path("<accession_dir>")
experiment_dir = Path("<experiment_dir>")

resolved = pd.read_parquet(accession_dir / "SmallMoleculeSchema.parquet")
raw_obs = pd.read_csv(experiment_dir / f"{fs}_raw_obs.csv", index_col=0)

# Build lookup: compound key → uid
uid_map = dict(zip(resolved["<key_column>"], resolved["uid"]))

fragment = pd.DataFrame(index=raw_obs.index)
```

### B2. Build perturbation list columns with `|` convention

```python
import json

def build_perturbation_lists(row):
    compound = row[compound_col]
    concentration = row.get(concentration_col)
    if pd.isna(compound):
        return None, None, None
    if is_control_label(str(compound)):
        return None, None, None

    uid = uid_map.get(str(compound))
    if uid is None:
        return None, None, None

    conc = float(concentration) if pd.notna(concentration) else -1.0
    return json.dumps([uid]), json.dumps(["small_molecule"]), json.dumps([conc])

results = raw_obs.apply(build_perturbation_lists, axis=1)
fragment["perturbation_uids|SmallMolecule"] = results.apply(lambda x: x[0])
fragment["perturbation_types|SmallMolecule"] = results.apply(lambda x: x[1])
fragment["perturbation_concentrations_um|SmallMolecule"] = results.apply(lambda x: x[2])
```

### B3. Derive control columns with `|` convention

```python
fragment["is_negative_control|SmallMolecule"] = raw_obs[compound_col].apply(
    lambda v: is_control_label(str(v)) if pd.notna(v) else False
)

fragment["negative_control_type|SmallMolecule"] = raw_obs[compound_col].apply(
    lambda v: detect_negative_control_type(str(v)) if pd.notna(v) and is_control_label(str(v)) else None
)
```

**Critical rule:** `is_negative_control=True` ONLY when the dataset explicitly labels a cell as a control (DMSO, vehicle, etc.). Cells with NaN/None compound (e.g., unassigned wells) must have `is_negative_control=False`.

### B4. Write fragment

```python
fragment_path = experiment_dir / f"{fs}_fragment_molecule_obs.csv"
fragment.to_csv(fragment_path)
print(f"Wrote {fragment_path.name}: {len(fragment)} rows")
```

---

## Resolution Strategy

All resolved columns follow the same principle: **never NaN unless there is genuinely no value**, and **always flag resolution status with a boolean `resolved` column.**

1. **Resolution succeeds** (`resolved_value` is not None) — use canonical values from `MoleculeResolution`. Set `resolved=True`.
2. **Resolution fails** (`resolved_value` is None) — keep the original value for name fields. Structural fields (`pubchem_cid`, `smiles`, `inchi_key`, `chembl_id`) can be None. Set `resolved=False`.
3. **Controls** — map to None in compound identity fields. They inform `is_negative_control` / `negative_control_type` on the obs fragment.
4. **NaN only when no value exists.**

## Rules

- **Two-phase workflow.** Phase A resolves globally and assigns UIDs. Phase B maps UIDs to per-experiment obs.
- **Two output files.** `SmallMolecule_resolved.csv` retains `resolved` for inspection. `SmallMoleculeSchema.parquet` is type-coerced and production-ready for direct LanceDB ingestion.
- **No `validated_` prefix.** Output columns use schema field names directly.
- **Use `|` convention in Phase B.** All obs columns that could also be written by other perturbation resolvers use `{field}|SmallMolecule` naming.
- **Assign UIDs via `make_uid()` in Phase A.** Every unique compound gets a UID.
- **One-step resolution.** Use `resolve_molecules()` directly. Do not use `resolve_pubchem_cids()` or any epiblast imports.
- **Use `is_control_label()` for control detection.** It checks both chemical and genetic controls. Do not hardcode control label sets.
- **`is_negative_control=True` ONLY for explicit controls.** NaN/None compound does NOT imply control.
- **Controls map to None** in compound identity fields.
- **Resolver owns enrichment.** If a field can be filled from supplementary files, raw identifiers, or deterministic parsing, do that work here rather than assuming the preparer already did it.
- **Every blank needs a reason.** In the final report, enumerate any schema fields left blank and justify why they could not be filled safely.
- **Name failures must be investigated.** Build correction mappings for unresolved names and re-resolve.
- **SMILES failures are acceptable.** Not all compounds are in PubChem. SMILES may still canonicalize via RDKit (confidence 0.5).
- **Never set name columns to NaN for failed resolution.** Use the original value. Only structural ID fields can be None.
- **Always write a `resolved` boolean column.**
- **Save after each column** to prevent losing work on interruption.
- **Column names follow the user's schema.** Do not assume specific column names.
- **Registry vs. obs:** `SmallMoleculeSchema` is a perturbation registry. Control fields (`is_negative_control`, `negative_control_type`) belong on the obs fragment, not the molecule registry.
- **Never modify h5ad files.** All validated data goes into the CSV only.
- **Flag remaining unresolved names** for user review. Do not silently drop them.
