# Genetic Perturbation Resolver — API Reference

## Gene Target Resolution

```python
from lancell.standardization import resolve_genes
from lancell.standardization.types import GeneResolution, ResolutionReport
```

### `resolve_genes(values, organism="human", input_type="symbol") -> ResolutionReport`

Resolve gene symbols to canonical identifiers. Returns a `ResolutionReport` containing `GeneResolution` objects. See the gene-resolver API reference for full details.

Key fields on `GeneResolution`: `symbol`, `ensembl_gene_id`, `organism`, `ncbi_gene_id`, `confidence`.

---

## Guide RNA Resolution

```python
from lancell.standardization import resolve_guide_sequences
from lancell.standardization.types import GuideRnaResolution, ResolutionReport
```

### `resolve_guide_sequences(sequences, organism="human") -> ResolutionReport`

Resolve guide RNA sequences to genomic coordinates and gene annotations. Uses UCSC BLAT to align each sequence, then queries Ensembl REST for overlapping protein-coding genes and classifies target context.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sequences` | `list[str]` | — | Guide RNA sequences (typically 20bp DNA strings) |
| `organism` | `str` | `"human"` | Currently supports `"human"` and `"mouse"` |

**Caching:** Results are cached locally in the LanceDB reference database (`guide_rnas` table). Subsequent calls with the same sequences skip BLAT. The cache is keyed on `(guide_sequence, organism)`.

**Rate limiting:** BLAT is limited to ~1 request/second for cache misses. Deduplicate sequences before calling — guides are shared across many cells.

---

## Coordinate Annotation

```python
from lancell.standardization import annotate_genomic_coordinates
```

### `annotate_genomic_coordinates(coordinates, organism="human") -> ResolutionReport`

Annotate pre-resolved genomic coordinates with gene and context information. Skips BLAT and goes directly to Ensembl overlap queries.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `coordinates` | `list[dict]` | — | Dicts with keys: `chromosome` (str), `start` (int), `end` (int), optionally `strand` (str), `guide_sequence` (str) |
| `organism` | `str` | `"human"` | Currently supports `"human"` and `"mouse"` |

---

## `GuideRnaResolution`

Returned by both `resolve_guide_sequences` and `annotate_genomic_coordinates`.

| Field | Type | Description |
|---|---|---|
| `input_value` | `str` | Original guide sequence or coordinate string |
| `resolved_value` | `str \| None` | Gene name or locus string, `None` if unresolved |
| `confidence` | `float` | `1.0` = single gene overlap, `0.9` = multiple genes, `0.5` = coordinates but no gene, `0.0` = failed |
| `source` | `str` | `"blat+ensembl"`, `"ensembl"`, or `"blat"` |
| `chromosome` | `str \| None` | e.g., `"chr17"` |
| `target_start` | `int \| None` | Genomic start coordinate |
| `target_end` | `int \| None` | Genomic end coordinate |
| `target_strand` | `str \| None` | `"+"` or `"-"` |
| `intended_gene_name` | `str \| None` | Closest protein-coding gene symbol |
| `intended_ensembl_gene_id` | `str \| None` | Ensembl ID of the intended gene |
| `target_context` | `str \| None` | Where the guide lands relative to gene structure |
| `assembly` | `str \| None` | e.g., `"hg38"`, `"mm39"` |
| `blat_pct_match` | `float \| None` | BLAT alignment quality (0–100), only for guide resolution |
| `alternatives` | `list[str]` | Other overlapping gene names |

### Target Context Values

| Value | Meaning |
|---|---|
| `"exon"` | Overlaps a CDS exon |
| `"intron"` | Within a transcript but not an exon |
| `"promoter"` | Within 2kb upstream of the TSS |
| `"5_UTR"` | In the 5' UTR |
| `"3_UTR"` | In the 3' UTR |
| `"intergenic"` | No overlapping protein-coding gene |
| `"other"` | Overlaps gene region but context is ambiguous |

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
| `is_control_label` | `(value: str) -> bool` | Check if a single value is a control label (genetic or chemical) |
| `detect_control_labels` | `(values: list[str]) -> list[bool]` | Vectorized version — returns boolean list |
| `detect_negative_control_type` | `(value: str) -> str \| None` | Returns canonical control type string or `None` |

Recognized genetic controls: `nontargeting`, `scramble`, `intergenic`, `safe-targeting`, `luciferase`, `lacz`, `gfp`, `rfp`, `empty_vector`, `control`, etc.

---

## Combinatorial Perturbation Parsing

```python
from lancell.standardization import parse_combinatorial_perturbations
```

`parse_combinatorial_perturbations(value: str) -> list[str]`

Splits on `+`, `&`, `;`, `|`, and comma-space. Does **not** split on `_`.

---

## Perturbation Method Classification

```python
from lancell.standardization import classify_perturbation_method, GeneticPerturbationType
```

`classify_perturbation_method(value: str) -> GeneticPerturbationType | None`

Regex-based classification of free-text method descriptions. Returns `None` if no known method matches. Bare "CRISPR" without qualifier defaults to `CRISPR_KO`.

### `GeneticPerturbationType` Enum

| Member | Value |
|---|---|
| `CRISPR_KO` | `"CRISPRko"` |
| `CRISPR_I` | `"CRISPRi"` |
| `CRISPR_A` | `"CRISPRa"` |
| `SI_RNA` | `"siRNA"` |
| `SH_RNA` | `"shRNA"` |
| `ASO` | `"ASO"` |
| `OVEREXPRESSION` | `"overexpression"` |
| `OTHER` | `"other"` |

---

## GeneticPerturbationSchema Field Mapping

Shows which resolution outputs fill which `GeneticPerturbationSchema` fields:

| Schema Field | Gene Name Mode | Guide Sequence Mode | Coordinate Mode |
|---|---|---|---|
| `perturbation_type` | `classify_perturbation_method()` | `classify_perturbation_method()` | `classify_perturbation_method()` |
| `guide_sequence` | — | input sequence | from input dict |
| `target_start` | — | `GuideRnaResolution.target_start` | from input |
| `target_end` | — | `GuideRnaResolution.target_end` | from input |
| `target_strand` | — | `GuideRnaResolution.target_strand` | from input |
| `intended_gene_name` | `GeneResolution.symbol` | `GuideRnaResolution.intended_gene_name` | `GuideRnaResolution.intended_gene_name` |
| `intended_ensembl_gene_id` | `GeneResolution.ensembl_gene_id` | `GuideRnaResolution.intended_ensembl_gene_id` | `GuideRnaResolution.intended_ensembl_gene_id` |
| `target_context` | — | `GuideRnaResolution.target_context` | `GuideRnaResolution.target_context` |
| `library_name` | from metadata | from metadata | from metadata |
| `reagent_id` | from metadata | from metadata | from metadata |
