---
name: genetic-perturbation-resolver
description: Resolve genetic perturbation targets in dataframes — gene names, guide RNA sequences, or genomic coordinates — and fill GeneticPerturbationSchema fields. Handles control detection, combinatorial splitting, perturbation method classification, and guide RNA alignment via BLAT. Use when a dataset has genetic perturbation columns (CRISPR, siRNA, shRNA, ORF, ASO) that need standardization.
---

# Genetic Perturbation Resolver

Resolve genetic perturbation targets and fill `GeneticPerturbationSchema` fields. Handles three input types that may co-exist in a single dataset:

1. **Gene names/symbols** — Target gene names (e.g., "TP53", "BRCA1"). Resolves to canonical symbols and Ensembl IDs.
2. **Guide RNA sequences** — Raw 20bp guide sequences from CRISPR screens. Aligns via BLAT to get genomic coordinates, then annotates with overlapping genes and target context.
3. **Genomic coordinates** — Pre-computed target regions (e.g., enhancer/promoter-targeting screens). Annotates with overlapping genes and target context without BLAT.

## Interface

**Input:**
- Dataframe(s) with genetic perturbation information — gene names, guide sequences, and/or genomic coordinates
- A user-specified target schema describing which output columns to produce

**Output:**
- The same dataframe(s) with resolution columns added, named per the user's target schema

When invoked from the geo-data-preparer pipeline, the input will be `{key}_standardized_obs.csv`, and output columns should use the `validated_` prefix convention.

**Rule:** Save the CSV after adding each column to prevent losing work.

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
from lancell.standardization.types import GeneResolution, GuideRnaResolution, ResolutionReport
```

---

## Step 1: Load & inspect

```python
import pandas as pd

obs_df = pd.read_csv(obs_csv_path, index_col=0)
```

Identify which columns contain perturbation information and what input type(s) are available:

- **Gene name column** — e.g., `"gene"`, `"target_gene"`, `"sgRNA_target"`, `"perturbation"`
- **Guide sequence column** — e.g., `"guide_seq"`, `"sgRNA_sequence"`, `"protospacer"` — typically 20bp DNA strings
- **Coordinate columns** — e.g., `"target_chr"`, `"target_start"`, `"target_end"` — or a combined format like `"chr17:7687490-7687510"`

Print unique values for the user's review before proceeding.

---

## Step 2: Control detection

```python
target_col = "<target_column>"
unique_targets = obs_df[target_col].dropna().unique().tolist()

control_mask = detect_control_labels(unique_targets)
control_labels = [t for t, is_ctrl in zip(unique_targets, control_mask) if is_ctrl]
actual_targets = [t for t, is_ctrl in zip(unique_targets, control_mask) if not is_ctrl]
print(f"Control labels: {control_labels}")
print(f"Actual targets: {len(actual_targets)}")
```

**Check for numbered control prefixes** not caught by `detect_control_labels`:

```python
for t in actual_targets:
    v = t.strip().lower()
    if v.startswith("negctrl") or v.startswith("neg_ctrl") or v.startswith("neg-ctrl"):
        print(f"  Possible missed control: '{t}'")
```

### Derive `is_negative_control` and `negative_control_type`

```python
def derive_is_control(value) -> bool:
    if pd.isna(value):
        return False  # NaN perturbation does NOT imply control
    return is_control_label(str(value))

obs_df["<is_control_column>"] = obs_df[target_col].apply(derive_is_control)

# Also derive the canonical control type
obs_df["<control_type_column>"] = obs_df[target_col].apply(
    lambda v: detect_negative_control_type(str(v)) if not pd.isna(v) else None
)
obs_df.to_csv(obs_csv_path)
```

**Combinatorial screens:** A cell is only a control if **all** of its perturbations are control type. If a cell received two sgRNAs where one targets a gene and the other is non-targeting, that cell is **not** a control.

---

## Step 3: Combinatorial splitting

```python
sample_parts = [parse_combinatorial_perturbations(t) for t in actual_targets[:20]]
max_parts = max(len(p) for p in sample_parts)
if max_parts > 1:
    print(f"Combinatorial perturbations detected (max targets: {max_parts})")
```

`parse_combinatorial_perturbations` uses `+`, `&`, `;`, `|`, and comma-space as delimiters. It does **not** use `_` because underscores are common in perturbation column values for non-delimiter reasons (sample IDs, compound names, free-text labels). If the dataset uses `_` as a combinatorial delimiter (e.g., `"AHR_KLF1"`), investigate manually — resolve a sample of split parts as gene symbols to confirm splitting produces valid genes before proceeding.

---

## Step 4: Classify perturbation method

```python
method_string = "<method>"  # from GEO metadata or obs column
method_result = classify_perturbation_method(method_string)
if method_result is not None:
    validated_method = method_result.value
    print(f"Classified method: {validated_method}")
else:
    print(f"WARNING: Could not classify method '{method_string}'")
    validated_method = method_string  # keep original

obs_df["<method_column>"] = validated_method
obs_df.to_csv(obs_csv_path)
```

---

## Step 5: Resolve by gene name

Use when perturbation targets are gene symbols (the most common case).

```python
report = resolve_genes(actual_targets, organism="human", input_type="symbol")
target_map = {}
for res in report.results:
    target_map[res.input_value] = res.symbol if res.symbol else res.input_value

if report.unresolved_values:
    print(f"{len(report.unresolved_values)} targets unresolved: {report.unresolved_values[:10]}")
```

This fills `intended_gene_name` and `intended_ensembl_gene_id` but provides no genomic coordinates or target context.

For combinatorial datasets, split and resolve each part independently:

```python
all_individual_targets = set()
for target in actual_targets:
    for part in parse_combinatorial_perturbations(target):
        part = part.strip()
        if part and not is_control_label(part):
            all_individual_targets.add(part)

report = resolve_genes(list(all_individual_targets), organism="human", input_type="symbol")
individual_map = {}
for res in report.results:
    individual_map[res.input_value] = res.symbol if res.symbol else res.input_value
```

### Write resolved perturbation columns

For single-target datasets:

```python
for label in control_labels:
    target_map[label] = None

obs_df["<perturbation_column>"] = obs_df[target_col].map(target_map)
obs_df.to_csv(obs_csv_path)
```

For combinatorial datasets:

```python
for i in range(max_parts):
    col_name = f"<perturbation_column>_{i + 1}"
    def get_part(value, idx=i):
        if pd.isna(value) or is_control_label(str(value)):
            return None
        parts = parse_combinatorial_perturbations(str(value))
        if idx < len(parts):
            part = parts[idx].strip()
            if is_control_label(part):
                return None
            return individual_map.get(part, part)
        return None
    obs_df[col_name] = obs_df[target_col].apply(get_part)
    obs_df.to_csv(obs_csv_path)
```

---

## Step 6: Resolve by guide RNA sequence

Use when the dataset provides raw guide sequences (typically 20bp DNA). This is slower than gene name resolution — BLAT is rate-limited to ~1 request/second.

```python
guide_col = "<guide_sequence_column>"
unique_guides = obs_df[guide_col].dropna().unique().tolist()
print(f"Unique guide sequences: {len(unique_guides)}")

# Deduplicate before resolution — guides are shared across many cells
report = resolve_guide_sequences(unique_guides, organism="human")
print(f"Resolved: {report.resolved}/{report.total}, Ambiguous: {report.ambiguous}")
```

Each `GuideRnaResolution` provides:

| Field | Description |
|---|---|
| `chromosome` | e.g., `"chr17"` |
| `target_start` | Genomic start coordinate |
| `target_end` | Genomic end coordinate |
| `target_strand` | `"+"` or `"-"` |
| `intended_gene_name` | Closest protein-coding gene |
| `intended_ensembl_gene_id` | Ensembl ID of the intended gene |
| `target_context` | `"exon"`, `"intron"`, `"promoter"`, `"5_UTR"`, `"3_UTR"`, `"intergenic"`, or `"other"` |
| `assembly` | e.g., `"hg38"` |
| `blat_pct_match` | BLAT alignment quality (0–100) |

```python
guide_map = {res.input_value: res for res in report.results}

# Map results back to obs
for field in ["chromosome", "target_start", "target_end", "target_strand",
              "intended_gene_name", "intended_ensembl_gene_id", "target_context"]:
    obs_df[f"<{field}_column>"] = obs_df[guide_col].map(
        lambda g: getattr(guide_map.get(g, None), field, None) if not pd.isna(g) else None
    )
obs_df.to_csv(obs_csv_path)
```

---

## Step 7: Resolve by genomic coordinates

Use when the dataset provides pre-computed target coordinates (e.g., enhancer/promoter-targeting CRISPRi/a screens). Skips BLAT and goes directly to Ensembl overlap annotation.

```python
# Build coordinate dicts from obs columns
coordinates = []
for _, row in obs_df[obs_df["<chr_col>"].notna()].drop_duplicates(subset=["<chr_col>", "<start_col>", "<end_col>"]).iterrows():
    coordinates.append({
        "chromosome": row["<chr_col>"],
        "start": int(row["<start_col>"]),
        "end": int(row["<end_col>"]),
        "strand": row.get("<strand_col>"),
    })

report = annotate_genomic_coordinates(coordinates, organism="human")
```

This fills `intended_gene_name`, `intended_ensembl_gene_id`, and `target_context`. Regions with no overlapping protein-coding gene get `target_context="intergenic"` and `intended_gene_name=None`.

---

## Resolution Strategy

All resolved columns follow the same principle: **never NaN unless there is genuinely no value**, and **always flag resolution status with a boolean `resolved` column.**

1. **Resolution succeeds** → use canonical values. Set `resolved=True`.
2. **Resolution fails** (gene name unresolved, guide fails BLAT, coordinates have no gene overlap) → keep original values where possible, set `resolved=False`.
3. **NaN only when no value exists** — e.g., a cell has no perturbation target.
4. **Control labels → None** — "non-targeting", "NegCtrl0", etc. become None in perturbation columns (they inform `is_negative_control`, not the gene field).

## Rules

- **`is_negative_control=True` ONLY for explicit controls.** NaN/None perturbation does NOT imply control.
- **Combinatorial screens:** A cell is only a control if **all** perturbations are control type.
- **Control labels map to None in perturbation columns.** They inform `is_negative_control`, not the gene target.
- **Watch for multiple control label variants.** Inspect unique values for numbered controls.
- **Resolve each combinatorial part independently.** Split targets and resolve each as its own gene symbol.
- **Deduplicate guide sequences before BLAT.** Guides are shared across many cells; BLAT is rate-limited (~1 req/s).
- **Save after each column** to prevent losing work on interruption.
- **Column names follow the user's schema.** Do not assume specific column names — use whatever the user's target schema specifies.
- **Ask before guessing.** If the delimiter or control labels are ambiguous, ask the user.
