# Gene Resolver — API Reference

## Gene Resolution

```python
from lancell.standardization import resolve_genes, detect_organism_from_ensembl_ids, is_placeholder_symbol
from lancell.standardization.types import GeneResolution, ResolutionReport
```

### `resolve_genes(values, organism="human", input_type="auto") -> ResolutionReport`

Resolve gene symbols or Ensembl IDs to canonical identifiers using local LanceDB reference tables.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `values` | `list[str]` | — | Gene symbols, Ensembl IDs, or a mix |
| `organism` | `str` | `"human"` | Organism context (common name: `"human"`, `"mouse"`, `"rat"`, etc.) |
| `input_type` | `"symbol" \| "ensembl_id" \| "auto"` | `"auto"` | `"auto"` detects per-value based on `ENS*G` prefix |

### `detect_organism_from_ensembl_ids(ids) -> dict[str, str]`

Infer organism for each Ensembl ID from its prefix. Returns mapping from Ensembl ID to organism common name. Unknown prefixes map to `"unknown"`. The prefix table is loaded dynamically from the reference DB — not hardcoded.

Common prefixes (informational):

| Prefix | Organism |
|---|---|
| `ENSG` | human |
| `ENSMUSG` | mouse |
| `ENSRNOG` | rat |
| `ENSDARG` | zebrafish |
| `ENSGALG` | chicken |
| `ENSSSOG` | pig |

### `is_placeholder_symbol(symbol: str) -> bool`

Check if a gene symbol is an accession-based placeholder or Riken clone — provisional identifiers assigned by GENCODE/RIKEN to genes lacking a proper HGNC/MGI symbol (typically lncRNAs, pseudogenes, antisense RNAs). Matches patterns like `AC134879.3`, `AL590822.2` (two uppercase letters + 6 digits + version) and `1700049J03Rik` (Riken clone format).

---

## Result Types

### `ResolutionReport`

| Field | Type | Description |
|---|---|---|
| `total` | `int` | Number of input values |
| `resolved` | `int` | Count of successfully resolved values |
| `unresolved` | `int` | Count of values with no match |
| `ambiguous` | `int` | Count of values with multiple matches |
| `results` | `list[GeneResolution]` | One result per input value, aligned with input order |

**Properties:**
- `.unresolved_values` — `list[str]` of input values that could not be resolved
- `.ambiguous_values` — `list[str]` of input values with multiple alternatives

**Methods:**
- `.to_dataframe()` — Returns a `pandas.DataFrame` with columns for all resolution fields

### `GeneResolution`

| Field | Type | Description |
|---|---|---|
| `input_value` | `str` | Original input string |
| `resolved_value` | `str \| None` | Canonical resolved ID, or `None` if unresolved |
| `confidence` | `float` | `1.0` = exact match, `0.9` = synonym, `0.7` = ambiguous, `0.0` = unresolved |
| `source` | `str` | `"lancedb"`, `"mygene"`, `"ensembl_rest"`, or `"none"` |
| `ensembl_gene_id` | `str \| None` | Resolved Ensembl gene ID (version-stripped) |
| `symbol` | `str \| None` | Canonical gene symbol (HGNC/MGI) |
| `organism` | `str \| None` | Organism common name |
| `ncbi_gene_id` | `int \| None` | NCBI gene ID |
| `alternatives` | `list[str]` | Other possible matches (Ensembl IDs) |
