# Protein Resolver — API Reference

## Protein Resolution

```python
from lancell.standardization import resolve_proteins
from lancell.standardization.types import ProteinResolution, ResolutionReport
```

### `resolve_proteins(values, organism="human") -> ResolutionReport`

Resolve protein names, gene names, or UniProt accessions to canonical UniProt IDs using local LanceDB reference tables (`proteins` and `protein_aliases`).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `values` | `list[str]` | — | Protein names, gene names, UniProt accessions, or a mix |
| `organism` | `str` | `"human"` | Organism context (common name: `"human"`, `"mouse"`, etc.) |

Resolution is one-step: input values are lowercased and matched against the `protein_aliases` table, then enriched with `protein_name` and `gene_name` from the `proteins` table.

---

## Result Types

### `ProteinResolution`

| Field | Type | Description |
|---|---|---|
| `input_value` | `str` | Original input string |
| `resolved_value` | `str \| None` | Best UniProt ID, or `None` if unresolved |
| `confidence` | `float` | `1.0` = single canonical, `0.9` = single non-canonical, `0.7` = multiple IDs, `0.0` = unresolved |
| `source` | `str` | `"lancedb"` or `"none"` |
| `uniprot_id` | `str \| None` | Canonical UniProt accession (e.g., `"P04637"`) |
| `protein_name` | `str \| None` | Recommended protein name (e.g., `"Cellular tumor antigen p53"`) |
| `gene_name` | `str \| None` | Primary gene name (e.g., `"TP53"`) |
| `organism` | `str \| None` | Organism common name |
| `sequence` | `str \| None` | Amino acid sequence from SwissProt |
| `sequence_length` | `int \| None` | Length of the amino acid sequence in residues |
| `alternatives` | `list[str]` | Other UniProt IDs if ambiguous |

### `ResolutionReport`

| Field | Type | Description |
|---|---|---|
| `total` | `int` | Number of input values |
| `resolved` | `int` | Count of successfully resolved values |
| `unresolved` | `int` | Count of values with no match |
| `ambiguous` | `int` | Count of values with multiple matches |
| `results` | `list[ProteinResolution]` | One result per input value, aligned with input order |

**Properties:**
- `.unresolved_values` — `list[str]` of input values that could not be resolved
- `.ambiguous_values` — `list[str]` of input values with multiple alternatives

**Methods:**
- `.to_dataframe()` — Returns a `pandas.DataFrame` with columns for all resolution fields

---

## Control Detection

```python
from lancell.standardization import (
    is_control_label,
    detect_control_labels,
    detect_negative_control_type,
)
```

| Function | Signature | Description |
|---|---|---|
| `is_control_label` | `(value: str) -> bool` | Check if a value is a control label (genetic or chemical) |
| `detect_control_labels` | `(values: list[str]) -> list[bool]` | Vectorized version — returns boolean list |
| `detect_negative_control_type` | `(value: str) -> str \| None` | Returns canonical control type string or `None` |

Recognized controls: `nontargeting`, `scramble`, `DMSO`, `vehicle`, `untreated`, `PBS`, `control`, etc.

**Important:** These functions do **NOT** detect isotype controls (IgG1, IgG2a, IgG2b, IgM, etc.). Isotype control detection requires the explicit patterns defined in the `resolve_proteins.py` script.

---

## ProteinSchema Field Mapping

Shows which resolution outputs fill which `ProteinSchema` fields:

| Schema Field | Source | Notes |
|---|---|---|
| `uid` | Auto-generated | `FeatureBaseSchema` |
| `global_index` | Auto-assigned | `FeatureBaseSchema` |
| `feature_name` | Input value | `FeatureBaseSchema` |
| `uniprot_id` | `ProteinResolution.uniprot_id` | |
| `protein_name` | `ProteinResolution.protein_name` | |
| `gene_name` | `ProteinResolution.gene_name` | |
| `organism` | `ProteinResolution.organism` or metadata | |
| `sequence` | `ProteinResolution.sequence` | From SwissProt reference DB |
| `sequence_length` | `ProteinResolution.sequence_length` | From SwissProt reference DB |
