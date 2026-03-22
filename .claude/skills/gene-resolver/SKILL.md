---
name: gene-resolver
description: Use this skill when tasked with standardizing gene identifiers (symbols, Ensembl IDs) in feature dataframes and looking up metadata to fill out missing information in a LanceDB table schema (e.g., GenomicFeatureSchema). Requires dataframes with at minimum the gene identifiers to standardize and a target schema specifying missing metadata to lookup. For genetic perturbation resolution, use the genetic-perturbation-resolver skill instead.
---

# Gene Resolver

Resolve gene identifiers in feature dataframes — typically the var index of a gene expression or chromatin accessibility matrix. Maps gene symbols and Ensembl IDs to canonical identifiers using the `lancell.standardization` suite.

For genetic perturbation target resolution (obs-level: control detection, combinatorial splitting, guide RNA alignment, perturbation method classification), use the **genetic-perturbation-resolver** skill.

## Interface

**Input:**
- One or more dataframes (CSV, pandas, polars) containing gene identifiers — symbols, Ensembl IDs, or a mix
- A user-specified target schema describing which output columns to produce and how to name them

**Output:**
- The same dataframe(s) with resolution columns added, named per the user's target schema

When invoked from the geo-data-preparer pipeline, the input will be `{key}_{feature_space}_standardized_var.csv`, and output columns should use the `validated_` prefix convention from that pipeline.

**Rule:** Save the CSV after adding each column to prevent losing work.

## Imports

```python
from lancell.standardization import (
    resolve_genes,
    detect_organism_from_ensembl_ids,
    is_placeholder_symbol,
)
from lancell.standardization.types import GeneResolution, ResolutionReport
```

---

## Workflow

### 1. Load the dataframe

```python
import pandas as pd
from pathlib import Path

var_df = pd.read_csv(var_csv_path, index_col=0)
```

Also load the original var dataframe (from the h5ad or matrix companions) to access gene identifiers and symbols that may be in separate columns.

### 2. Detect identifier format

Determine whether the var index contains Ensembl IDs or gene symbols:

```python
var_index_sample = var_df.index[:10].tolist()
is_ensembl = any(str(v).startswith("ENS") for v in var_index_sample)
```

If the index is Ensembl IDs, gene symbols may be in a separate column (e.g., `gene_symbols`, `gene_name`, `feature_name`). If the index is gene symbols, Ensembl IDs may be in a column like `gene_ids`.

Note: `resolve_genes(values, input_type="auto")` auto-detects per-value, but knowing the dominant format helps with organism detection and fallback strategies.

### 3. Detect organisms from Ensembl prefixes (barnyard detection)

```python
# Get Ensembl IDs (from index or column), strip version suffixes
ensembl_ids = [str(eid).split(".")[0] for eid in ensembl_id_source]

id_to_organism = detect_organism_from_ensembl_ids(ensembl_ids)
unique_organisms = set(v for v in id_to_organism.values() if v != "unknown")

print(f"Organisms detected: {unique_organisms}")
for org in unique_organisms:
    count = sum(1 for v in id_to_organism.values() if v == org)
    print(f"  {org}: {count} genes")

unknown_count = sum(1 for v in id_to_organism.values() if v == "unknown")
if unknown_count:
    print(f"  unknown prefix: {unknown_count} genes")
```

If multiple organisms are detected, this is a **barnyard experiment**. Report the finding and proceed with per-organism resolution.

### 4. Resolve Ensembl IDs (per organism)

Strip version suffixes (e.g., `ENSG00000141510.16` -> `ENSG00000141510`) before resolution. The `.16` is the annotation version number (incremented when the gene model changes), not an isoform identifier.

```python
for organism in unique_organisms:
    org_ids = [eid for eid in ensembl_ids if id_to_organism.get(eid) == organism]

    report = resolve_genes(org_ids, organism=organism, input_type="ensembl_id")
    print(f"{organism}: {report.total} genes, {report.resolved} resolved, {report.unresolved} unresolved")
    if report.unresolved_values:
        print(f"  Unresolved sample: {report.unresolved_values[:10]}")
```

**Old Ensembl versions:** If a large fraction fails (suggesting GRCh37/hg19 vs GRCh38/hg38 mismatch), attempt recovery via gene symbols:

```python
# Get symbols for unresolved IDs, then resolve by symbol
unresolved_symbols = [sym for eid, sym in zip(ensembl_ids, gene_symbols) if eid in report.unresolved_values]
fallback_report = resolve_genes(unresolved_symbols, organism=organism, input_type="symbol")
```

**Resolution strategy:** Use `GeneResolution.ensembl_gene_id` when resolved. When unresolved (`resolved_value is None`), keep the original stripped Ensembl ID as-is — do not set to NaN. Flag it with `resolved=False` so it can be filtered or manually investigated later.

```python
for res in report.results:
    validated_id = res.ensembl_gene_id if res.ensembl_gene_id else res.input_value
    is_resolved = res.resolved_value is not None
```

### 5. Resolve gene symbols (per organism)

```python
for organism in unique_organisms:
    org_symbols = [sym for sym, org in zip(gene_symbols, gene_organisms) if org == organism]

    report = resolve_genes(org_symbols, organism=organism, input_type="symbol")
    for res in report.results:
        # Use canonical symbol when resolved, original symbol when not
        validated_symbol = res.symbol if res.symbol else res.input_value
        is_resolved = res.resolved_value is not None
```

Unresolved symbols are commonly GenBank/EMBL accession-based placeholders (e.g., `AC000061.1`, `AL590822.2`) assigned by GENCODE to lncRNAs, pseudogenes, and antisense RNAs that lack a proper HGNC symbol. Riken clones like `1700049J03Rik` are similar. Use `is_placeholder_symbol(symbol)` to detect these patterns. Keep their original names but flag `resolved=False`.

### 6. Write the `resolved` column

**Always** write a boolean `resolved` column alongside the resolved identifiers. This flags genes that could not be matched in the reference DB (which covers the two latest Ensembl releases plus GENCODE). Unresolved genes are candidates for removal or manual follow-up.

```python
var_df["<resolved_column>"] = [res.resolved_value is not None for res in all_results]
var_df.to_csv(var_csv_path)
```

---

## Resolution Strategy

All resolved columns follow the same principle: **never NaN unless there is genuinely no value**, and **always flag resolution status with a boolean `resolved` column.**

1. **Resolution succeeds** (`confidence > 0`, `resolved_value` is not None) → use the canonical value from `GeneResolution` (e.g., `.symbol`, `.ensembl_gene_id`). Set `resolved=True`.
2. **Resolution fails** (`confidence == 0.0`, `resolved_value` is None) → keep the original `input_value` as-is (do not set to NaN), but set `resolved=False`. The reference DB covers the two latest Ensembl releases plus GENCODE, so unresolved genes are likely deprecated IDs, accession-based placeholders (AC/AL-prefixed lncRNAs, pseudogenes), or errors. These are candidates for removal or manual investigation.
3. **NaN only when no value exists** — e.g., a gene has no symbol at all.

The `resolved` column ensures downstream consumers can distinguish "resolution failed" from "no data" and decide whether to keep, remove, or manually resolve flagged genes.

## Rules

- **Strip version suffixes** from Ensembl IDs before resolution (split on `.`).
- **Always write a `resolved` boolean column.** This flags genes that could not be matched. Unresolved genes are likely deprecated IDs or accession-based placeholders (AC/AL-prefixed lncRNAs, pseudogenes) and are candidates for removal or manual follow-up.
- **Resolve per organism** when multiple organisms are detected (barnyard experiments).
- **Old Ensembl versions:** If a large fraction of Ensembl IDs fail, attempt recovery via gene symbols.
- **Never set resolved columns to NaN for failed resolution.** Use the original value and set `resolved=False`.
- **Save after each column** to prevent losing work on interruption.
- **Column names follow the user's schema.** Do not assume specific column names — use whatever the user's target schema specifies.
