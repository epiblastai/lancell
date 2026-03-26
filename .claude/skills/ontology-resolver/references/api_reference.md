# Ontology Resolver — API Reference

## Ontology Resolution

```python
from lancell.standardization import resolve_ontology_terms, OntologyEntity
from lancell.standardization.types import OntologyResolution, ResolutionReport
```

### `resolve_ontology_terms(values, entity, organism=None, min_similarity=0.8) -> ResolutionReport`

Resolve free-text values to ontology terms with CELLxGENE-compatible IDs.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `values` | `list[str]` | — | Free-text metadata values |
| `entity` | `OntologyEntity` | — | Which ontology entity to resolve against |
| `organism` | `str \| None` | `None` | Organism context (required for `DEVELOPMENT_STAGE`) |
| `min_similarity` | `float` | `0.8` | Reserved for future fuzzy matching; currently unused |

**Resolution strategy (for DB-backed entities):**
1. Exact name match (case-insensitive) against `ontology_terms` table → confidence 1.0
2. Synonym match (pipe-delimited synonyms, case-insensitive) → confidence 0.9
3. No match → unresolved (confidence 0.0)

**Special entity handling:**
- `SEX` — Hardcoded lookup (not from DB): female→PATO:0000383, male→PATO:0000384, unknown/other→PATO:0000461
- `CELL_LINE` — OLS4 API: exact match → confidence 1.0, fuzzy match → confidence 0.8 (with alternatives)
- `DEVELOPMENT_STAGE` — Organism-aware prefix: human→HsapDv, mouse→MmusDv

---

## OntologyEntity Enum

```python
from lancell.standardization import OntologyEntity
```

| Value | Ontology | Prefix(es) | Backend |
|---|---|---|---|
| `CELL_TYPE` | Cell Ontology | CL | Local DB |
| `TISSUE` | UBERON | UBERON | Local DB |
| `DISEASE` | MONDO | MONDO | Local DB |
| `ORGANISM` | NCBITaxon | NCBITaxon | Local DB |
| `ASSAY` | EFO | EFO | Local DB |
| `DEVELOPMENT_STAGE` | HsapDv / MmusDv | HsapDv, MmusDv | Local DB (organism-aware) |
| `ETHNICITY` | HANCESTRO | HANCESTRO | Local DB |
| `SEX` | PATO | PATO | Hardcoded |
| `CELL_LINE` | CLO | CLO | OLS4 API |

---

## Result Types

### `OntologyResolution`

Inherits from `Resolution` base class.

| Field | Type | Description |
|---|---|---|
| `input_value` | `str` | Original input string |
| `resolved_value` | `str \| None` | Canonical ontology term name, or `None` if unresolved |
| `confidence` | `float` | `1.0` = exact name match, `0.9` = synonym match, `0.8` = OLS4 fuzzy, `0.0` = unresolved |
| `source` | `str` | `"reference_db"`, `"reference_db_synonym"`, `"ols4_clo"`, `"pato_hardcoded"`, or `"none"` |
| `ontology_term_id` | `str \| None` | CURIE (e.g., `"CL:0000540"`, `"UBERON:0002048"`) |
| `ontology_name` | `str \| None` | Display name (e.g., `"Cell Ontology"`, `"UBERON"`) |
| `alternatives` | `list[str]` | Other candidate term IDs if ambiguous (mainly for OLS4 fuzzy matches) |

### `ResolutionReport`

| Field | Type | Description |
|---|---|---|
| `total` | `int` | Number of input values |
| `resolved` | `int` | Count of successfully resolved values |
| `unresolved` | `int` | Count of values with no match |
| `ambiguous` | `int` | Count of values with multiple matches |
| `results` | `list[OntologyResolution]` | One result per input value, aligned with input order |

**Properties:**
- `.unresolved_values` — `list[str]` of input values that could not be resolved
- `.ambiguous_values` — `list[str]` of input values with multiple alternatives

**Methods:**
- `.to_dataframe()` — Returns a `pandas.DataFrame` with columns for all resolution fields

---

## Convenience Wrappers

```python
from lancell.standardization import (
    resolve_cell_types,
    resolve_tissues,
    resolve_diseases,
    resolve_organisms,
    resolve_assays,
    resolve_cell_lines,
)
```

Each is a thin wrapper around `resolve_ontology_terms` with the entity pre-set:

| Function | Signature | Entity |
|---|---|---|
| `resolve_cell_types` | `(values: list[str]) -> ResolutionReport` | `CELL_TYPE` |
| `resolve_tissues` | `(values: list[str]) -> ResolutionReport` | `TISSUE` |
| `resolve_diseases` | `(values: list[str]) -> ResolutionReport` | `DISEASE` |
| `resolve_organisms` | `(values: list[str]) -> ResolutionReport` | `ORGANISM` |
| `resolve_assays` | `(values: list[str]) -> ResolutionReport` | `ASSAY` |
| `resolve_cell_lines` | `(values: list[str]) -> ResolutionReport` | `CELL_LINE` |

Note: No convenience wrapper for `SEX`, `ETHNICITY`, or `DEVELOPMENT_STAGE` — use `resolve_ontology_terms()` directly.

---

## Single-Value Helper

```python
from lancell.standardization import get_ontology_term_id
```

| Function | Signature | Description |
|---|---|---|
| `get_ontology_term_id` | `(value: str, entity: OntologyEntity, organism: str \| None = None) -> str \| None` | Resolve a single value and return just the CURIE, or `None` |

---

## Hierarchy Navigation

```python
from lancell.standardization import (
    get_ontology_ancestors,
    get_ontology_descendants,
    get_ontology_siblings,
)
```

| Function | Signature | Returns |
|---|---|---|
| `get_ontology_ancestors` | `(term_id: str, entity: OntologyEntity, organism: str \| None = None, max_depth: int \| None = None)` | `list[tuple[str, str]]` — `(ontology_id, name)` pairs, closest first |
| `get_ontology_descendants` | `(term_id: str, entity: OntologyEntity, organism: str \| None = None, max_depth: int \| None = None)` | `list[tuple[str, str]]` — `(ontology_id, name)` pairs, closest first |
| `get_ontology_siblings` | `(term_id: str, entity: OntologyEntity, organism: str \| None = None)` | `list[tuple[str, str]]` — `(ontology_id, name)` pairs for sibling terms |

These operate on the local `ontology_terms` table (not available for `SEX` or `CELL_LINE`). Useful for investigating near-misses during failure analysis.

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

**Note:** Control labels in ontology columns are uncommon but possible (e.g., `"control"` in a cell_type column from a perturbation screen). These are not ontology terms — map to `None` in validated columns.

---

## Output Column Mapping

For each ontology field in the schema, the resolver produces two columns:

| Output Column | Source | Notes |
|---|---|---|
| `validated_{field}` | `OntologyResolution.resolved_value` or `input_value` | Canonical name; keep original on failure |
| `validated_{field}_ontology_id` | `OntologyResolution.ontology_term_id` | CURIE; `None` on failure |
| `resolved` | Derived | `True` if all ontology fields for that row resolved |
