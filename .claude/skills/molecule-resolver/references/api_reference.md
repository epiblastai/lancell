# Molecule Resolver — API Reference

## Molecule Resolution

```python
from lancell.standardization import resolve_molecules
from lancell.standardization.types import MoleculeResolution, ResolutionReport
```

### `resolve_molecules(values, input_type="name") -> ResolutionReport`

Resolve small molecule identifiers to canonical structures using a 3-tier fallback chain: local LanceDB → PubChem API → ChEMBL API. For name inputs, `clean_compound_name()` is applied automatically before lookup.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `values` | `list[str]` | — | Compound names, SMILES strings, or PubChem CID strings |
| `input_type` | `Literal["name", "smiles", "cid"]` | `"name"` | Type of input identifier |

**Resolution chain for `input_type="name"`:**
1. Control detection — DMSO, vehicle, PBS, etc. bypass resolution
2. Name cleaning — strip salt suffixes, parenthetical form info
3. Local LanceDB batch lookup (`compound_synonyms` table) — confidence 1.0 (title match) or 0.9 (synonym)
4. PubChem API (rate-limited 5/s) — confidence 0.9
5. ChEMBL API fallback — confidence 0.85

**Resolution chain for `input_type="smiles"`:**
1. RDKit canonicalization
2. PubChem lookup by SMILES — confidence 0.9
3. If PubChem fails but RDKit succeeds — confidence 0.5, source "rdkit"

---

## Result Types

### `MoleculeResolution`

| Field | Type | Description |
|---|---|---|
| `input_value` | `str` | Original input string |
| `resolved_value` | `str \| None` | Canonical name/SMILES, or `None` if unresolved |
| `confidence` | `float` | `1.0` = exact title, `0.9` = synonym/PubChem, `0.85` = ChEMBL, `0.7` = CID only, `0.5` = RDKit only, `0.0` = unresolved |
| `source` | `str` | `"lancedb"`, `"pubchem"`, `"chembl"`, `"rdkit"`, `"control_detection"`, or `"none"` |
| `pubchem_cid` | `int \| None` | PubChem Compound ID |
| `canonical_smiles` | `str \| None` | RDKit-canonicalized SMILES |
| `inchi_key` | `str \| None` | InChIKey identifier |
| `iupac_name` | `str \| None` | IUPAC systematic name |
| `chembl_id` | `str \| None` | ChEMBL molecule ID (e.g., `"CHEMBL941"`) |
| `alternatives` | `list[str]` | Other candidates if ambiguous |

### `ResolutionReport`

| Field | Type | Description |
|---|---|---|
| `total` | `int` | Number of input values |
| `resolved` | `int` | Count of successfully resolved values |
| `unresolved` | `int` | Count of values with no match |
| `ambiguous` | `int` | Count of values with multiple matches |
| `results` | `list[MoleculeResolution]` | One result per input value, aligned with input order |

**Properties:**
- `.unresolved_values` — `list[str]` of input values that could not be resolved
- `.ambiguous_values` — `list[str]` of input values with multiple alternatives

**Methods:**
- `.to_dataframe()` — Returns a `pandas.DataFrame` with columns for all resolution fields

---

## Helper Functions

```python
from lancell.standardization import (
    clean_compound_name,
    is_control_compound,
    canonicalize_smiles,
)
```

| Function | Signature | Description |
|---|---|---|
| `clean_compound_name` | `(name: str) -> str` | Strip whitespace, salt suffixes (hydrochloride, mesylate, etc.), parenthetical form info. Called internally by `resolve_molecules` for name inputs. |
| `is_control_compound` | `(name: str) -> bool` | Check if name is a chemical negative control (DMSO, vehicle, PBS, etc.). Narrower than `is_control_label`. |
| `canonicalize_smiles` | `(smiles: str) -> str \| None` | RDKit SMILES canonicalization. Returns `None` if invalid. |

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

Recognized chemical controls: `DMSO`, `vehicle`, `PBS`, `untreated`, `unperturbed`, `mock`, `media`, `water`, `ethanol`, `saline`, `control`, etc.

Recognized genetic controls: `nontargeting`, `scramble`, `safe-targeting`, `luciferase`, `intergenic`, `empty_vector`, etc.

**Note:** `is_control_label()` checks BOTH sets. `is_control_compound()` checks only chemical controls.

---

## SmallMoleculeSchema Field Mapping

Shows which resolution outputs fill which `SmallMoleculeSchema` fields:

| Schema Field | Source | Notes |
|---|---|---|
| `uid` | Auto-generated | At ingestion time |
| `smiles` | `MoleculeResolution.canonical_smiles` | RDKit-canonicalized |
| `pubchem_cid` | `MoleculeResolution.pubchem_cid` | |
| `iupac_name` | `MoleculeResolution.iupac_name` | |
| `inchi_key` | `MoleculeResolution.inchi_key` | |
| `chembl_id` | `MoleculeResolution.chembl_id` | |
| `name` | `MoleculeResolution.resolved_value` or `input_value` | Common/trade name; keep original on failure |
| `vendor` | From metadata | If available |
| `catalog_number` | From metadata | If available |
