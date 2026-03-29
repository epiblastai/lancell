"""Small molecule name → structure resolution.

Strategy:
1. Name cleanup: strip whitespace, normalize, remove salt suffixes
2. Control detection: DMSO, vehicle, PBS, etc. → skip resolution
3. Local LanceDB lookup (compounds + compound_synonyms tables)
4. PubChem API fallback via pubchempy
5. ChEMBL API fallback
6. RDKit SMILES canonicalization
"""

import re
from typing import Literal

import polars as pl
import requests

from lancell.standardization._rate_limit import rate_limited
from lancell.standardization.metadata_table import (
    COMPOUND_SYNONYMS_TABLE,
    COMPOUNDS_TABLE,
    get_reference_db,
)
from lancell.standardization.perturbations import _CHEMICAL_NEGATIVE_CONTROLS
from lancell.standardization.types import MoleculeResolution, ResolutionReport
from lancell.util import sql_escape

# Salt suffixes to strip from compound names
_SALT_SUFFIXES = re.compile(
    r"\s*\b("
    r"hydrochloride|hcl|dihydrochloride"
    r"|sodium|potassium|calcium"
    r"|sulfate|sulphate"
    r"|phosphate"
    r"|acetate"
    r"|citrate"
    r"|tartrate"
    r"|fumarate"
    r"|maleate|malate"
    r"|mesylate|methanesulfonate"
    r"|tosylate"
    r"|trifluoroacetate|tfa"
    r"|bromide|chloride|iodide"
    r"|nitrate"
    r"|succinate"
    r"|besylate|benzenesulfonate"
    r"|hemisulfate"
    r"|monohydrate|dihydrate|trihydrate|hydrate"
    r"|salt"
    r")\b.*$",
    re.IGNORECASE,
)

# Parenthetical salt/form info
_PAREN_SUFFIX = re.compile(
    r"\s*\([^)]*(?:salt|form|hydrate|anhydrous|free\s+base)[^)]*\)\s*$", re.I
)


def clean_compound_name(name: str) -> str:
    """Normalize a compound name for PubChem lookup.

    Strips whitespace, removes salt suffixes and parenthetical form info.
    """
    cleaned = name.strip()
    # Remove parenthetical salt/form info
    cleaned = _PAREN_SUFFIX.sub("", cleaned)
    # Remove salt suffixes
    cleaned = _SALT_SUFFIXES.sub("", cleaned)
    return cleaned.strip()


def is_control_compound(name: str) -> bool:
    """Check if a compound name is a known negative control."""
    return name.strip().lower() in _CHEMICAL_NEGATIVE_CONTROLS


def canonicalize_smiles(smiles: str) -> str | None:
    """Canonicalize a SMILES string using RDKit. Returns None if invalid."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


@rate_limited("pubchem", max_per_second=5)
def _pubchem_get_cids(identifier: str, namespace: str = "name") -> list[int]:
    """Rate-limited PubChem CID lookup."""
    import pubchempy as pcp

    if not identifier or not identifier.strip():
        return []

    try:
        return pcp.get_cids(identifier, namespace=namespace)
    except (pcp.BadRequestError, ValueError):
        return []


@rate_limited("pubchem", max_per_second=5)
def _pubchem_get_compound(cid: int) -> dict | None:
    """Rate-limited PubChem compound property fetch."""
    import pubchempy as pcp

    try:
        compounds = pcp.get_compounds(cid, namespace="cid")
        if compounds:
            c = compounds[0]
            return {
                "cid": c.cid,
                "canonical_smiles": c.connectivity_smiles,
                "isomeric_smiles": c.smiles,
                "inchikey": c.inchikey,
                "iupac_name": c.iupac_name,
            }
    except Exception:
        pass
    return None


_CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"


@rate_limited("chembl")
def _chembl_search_by_name(name: str) -> dict | None:
    """Search ChEMBL for a molecule by name. Returns first hit with structures, or None."""
    try:
        resp = requests.get(
            f"{_CHEMBL_API_BASE}/molecule/search.json",
            params={"q": name, "limit": 5},
            headers={"Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        for mol in data.get("molecules", []):
            if mol.get("molecule_structures") is not None:
                return mol
    except Exception:
        pass
    return None


def _resolve_chembl_fallback(name: str, cleaned: str) -> MoleculeResolution | None:
    """Try ChEMBL as a fallback when PubChem finds nothing."""
    mol = _chembl_search_by_name(cleaned)
    if mol is None and cleaned != name.strip():
        mol = _chembl_search_by_name(name.strip())

    if mol is None:
        return None

    chembl_id = mol.get("molecule_chembl_id")
    pref_name = mol.get("pref_name")
    structures = mol.get("molecule_structures", {}) or {}
    raw_smiles = structures.get("canonical_smiles")
    inchi_key = structures.get("standard_inchi_key")

    canon_smiles = None
    if raw_smiles:
        canon_smiles = canonicalize_smiles(raw_smiles) or raw_smiles

    return MoleculeResolution(
        input_value=name,
        resolved_value=pref_name or cleaned,
        confidence=0.85,
        source="chembl",
        chembl_id=chembl_id,
        canonical_smiles=canon_smiles,
        inchi_key=inchi_key,
    )


def _has_compound_tables() -> bool:
    """Check if the compound LanceDB tables are populated."""
    try:
        db = get_reference_db()
        tables = db.list_tables().tables
        return COMPOUND_SYNONYMS_TABLE in tables
    except (RuntimeError, Exception):
        return False


def _resolve_name_local(cleaned: str, original: str) -> MoleculeResolution | None:
    """Try to resolve a compound name via the local compound_synonyms table.

    Returns None if the DB is not populated or no match is found,
    allowing the caller to fall through to API resolution.
    """
    if not _has_compound_tables():
        return None

    db = get_reference_db()
    table = db.open_table(COMPOUND_SYNONYMS_TABLE)
    lower_name = cleaned.lower()

    df = (
        table.search()
        .where(f"synonym = '{sql_escape(lower_name)}'", prefilter=True)
        .select(["synonym", "synonym_original", "pubchem_cid", "is_title"])
        .to_polars()
    )

    if df.is_empty():
        return None

    # Prefer title matches over synonym matches
    title_matches = df.filter(pl.col("is_title"))
    if not title_matches.is_empty():
        row = title_matches.row(0, named=True)
        confidence = 1.0
    else:
        row = df.row(0, named=True)
        confidence = 0.9

    cid = row["pubchem_cid"]
    resolved_name = row["synonym_original"]

    # Look up SMILES from the compounds table
    canonical_smiles = None
    if COMPOUNDS_TABLE in db.list_tables().tables:
        compounds_table = db.open_table(COMPOUNDS_TABLE)
        comp_df = (
            compounds_table.search()
            .where(f"pubchem_cid = {cid}", prefilter=True)
            .select(["canonical_smiles"])
            .to_polars()
        )
        if not comp_df.is_empty():
            canonical_smiles = comp_df.row(0, named=True)["canonical_smiles"]

    return MoleculeResolution(
        input_value=original,
        resolved_value=resolved_name,
        confidence=confidence,
        source="lancedb",
        pubchem_cid=cid,
        canonical_smiles=canonical_smiles,
    )


def _resolve_batch_names_local(names: list[str]) -> dict[str, MoleculeResolution]:
    """Batch-resolve compound names via the local compound_synonyms table.

    Returns a dict mapping input name → MoleculeResolution for names that
    were successfully resolved. Names not found are omitted from the result.
    """
    if not names or not _has_compound_tables():
        return {}

    db = get_reference_db()
    table = db.open_table(COMPOUND_SYNONYMS_TABLE)

    # Build mapping from lowercased cleaned name → original input name
    cleaned_to_original: dict[str, str] = {}
    for name in names:
        cleaned = clean_compound_name(name)
        cleaned_to_original[cleaned.lower()] = name

    # Batch query in groups of 500
    lower_names = list(cleaned_to_original.keys())
    syn_frames: list[pl.DataFrame] = []
    for i in range(0, len(lower_names), 500):
        batch = lower_names[i : i + 500]
        in_clause = ", ".join(f"'{sql_escape(n)}'" for n in batch)
        df = (
            table.search()
            .where(f"synonym IN ({in_clause})", prefilter=True)
            .select(["synonym", "synonym_original", "pubchem_cid", "is_title"])
            .to_polars()
        )
        syn_frames.append(df)

    if not syn_frames:
        return {}

    all_syns = pl.concat(syn_frames)
    if all_syns.is_empty():
        return {}

    # Look up SMILES for all matched CIDs
    smiles_map: dict[int, str | None] = {}
    if COMPOUNDS_TABLE in db.list_tables().tables:
        matched_cids = all_syns.get_column("pubchem_cid").unique().to_list()
        compounds_table = db.open_table(COMPOUNDS_TABLE)
        for i in range(0, len(matched_cids), 500):
            batch = matched_cids[i : i + 500]
            in_clause = ", ".join(str(c) for c in batch)
            comp_df = (
                compounds_table.search()
                .where(f"pubchem_cid IN ({in_clause})", prefilter=True)
                .select(["pubchem_cid", "canonical_smiles"])
                .to_polars()
            )
            for row in comp_df.iter_rows(named=True):
                smiles_map[row["pubchem_cid"]] = row["canonical_smiles"]

    # Group by synonym and pick best match per name
    results: dict[str, MoleculeResolution] = {}
    grouped = all_syns.group_by("synonym").agg(pl.all())

    for row in grouped.iter_rows(named=True):
        synonym_lower = row["synonym"]
        original_name = cleaned_to_original.get(synonym_lower)
        if original_name is None:
            continue

        cids = row["pubchem_cid"]
        is_title_flags = row["is_title"]
        originals = row["synonym_original"]

        # Prefer title matches
        best_idx = 0
        best_confidence = 0.9
        for idx, is_title in enumerate(is_title_flags):
            if is_title:
                best_idx = idx
                best_confidence = 1.0
                break

        cid = cids[best_idx]
        resolved_name = originals[best_idx]

        results[original_name] = MoleculeResolution(
            input_value=original_name,
            resolved_value=resolved_name,
            confidence=best_confidence,
            source="lancedb",
            pubchem_cid=cid,
            canonical_smiles=smiles_map.get(cid),
        )

    return results


def _resolve_single_name(name: str) -> MoleculeResolution:
    """Resolve a single compound name to a MoleculeResolution."""
    # Check if it's a control
    if is_control_compound(name):
        return MoleculeResolution(
            input_value=name,
            resolved_value=name.strip().upper(),
            confidence=1.0,
            source="control_detection",
        )

    cleaned = clean_compound_name(name)

    # Try local DB first
    local_result = _resolve_name_local(cleaned, name)
    if local_result is not None:
        return local_result

    # PubChem lookup
    cids = _pubchem_get_cids(cleaned, namespace="name")
    if not cids:
        # Try original name as fallback
        if cleaned != name.strip():
            cids = _pubchem_get_cids(name.strip(), namespace="name")

    if not cids:
        chembl_result = _resolve_chembl_fallback(name, cleaned)
        if chembl_result is not None:
            return chembl_result
        return MoleculeResolution(
            input_value=name,
            resolved_value=None,
            confidence=0.0,
            source="none",
        )

    cid = cids[0]
    compound_data = _pubchem_get_compound(cid)
    if compound_data is None:
        # Got CID but couldn't fetch properties
        return MoleculeResolution(
            input_value=name,
            resolved_value=cleaned,
            confidence=0.7,
            source="pubchem",
            pubchem_cid=cid,
        )

    canon_smiles = compound_data.get("canonical_smiles")
    if canon_smiles:
        rdkit_canon = canonicalize_smiles(canon_smiles)
        if rdkit_canon:
            canon_smiles = rdkit_canon

    return MoleculeResolution(
        input_value=name,
        resolved_value=compound_data.get("iupac_name") or cleaned,
        confidence=0.9,
        source="pubchem",
        pubchem_cid=cid,
        canonical_smiles=canon_smiles,
        inchi_key=compound_data.get("inchikey"),
        iupac_name=compound_data.get("iupac_name"),
    )


def _resolve_single_smiles(smiles: str) -> MoleculeResolution:
    """Resolve a single SMILES string."""
    # Canonicalize first
    canonical = canonicalize_smiles(smiles)
    lookup_smiles = canonical or smiles

    # PubChem lookup by SMILES
    cids = _pubchem_get_cids(lookup_smiles, namespace="smiles")
    if not cids:
        # Still have a valid canonical SMILES even without PubChem
        if canonical:
            return MoleculeResolution(
                input_value=smiles,
                resolved_value=canonical,
                confidence=0.5,
                source="rdkit",
                canonical_smiles=canonical,
            )
        return MoleculeResolution(
            input_value=smiles,
            resolved_value=None,
            confidence=0.0,
            source="none",
        )

    cid = cids[0]
    compound_data = _pubchem_get_compound(cid)
    result_smiles = canonical or lookup_smiles

    if compound_data:
        return MoleculeResolution(
            input_value=smiles,
            resolved_value=compound_data.get("canonical_smiles") or result_smiles,
            confidence=0.9,
            source="pubchem",
            pubchem_cid=cid,
            canonical_smiles=compound_data.get("canonical_smiles") or result_smiles,
            inchi_key=compound_data.get("inchikey"),
            iupac_name=compound_data.get("iupac_name"),
        )
    else:
        return MoleculeResolution(
            input_value=smiles,
            resolved_value=result_smiles,
            confidence=0.7,
            source="pubchem",
            pubchem_cid=cid,
            canonical_smiles=result_smiles,
        )


def _resolve_single_cid(cid_str: str) -> MoleculeResolution:
    """Resolve a single PubChem CID (passed as string)."""
    try:
        cid = int(cid_str)
    except ValueError:
        return MoleculeResolution(
            input_value=cid_str,
            resolved_value=None,
            confidence=0.0,
            source="none",
        )

    compound_data = _pubchem_get_compound(cid)
    if compound_data is None:
        return MoleculeResolution(
            input_value=cid_str,
            resolved_value=None,
            confidence=0.0,
            source="none",
        )

    canon_smiles = compound_data.get("canonical_smiles")
    if canon_smiles:
        rdkit_canon = canonicalize_smiles(canon_smiles)
        if rdkit_canon:
            canon_smiles = rdkit_canon

    return MoleculeResolution(
        input_value=cid_str,
        resolved_value=compound_data.get("iupac_name") or str(cid),
        confidence=0.95,
        source="pubchem",
        pubchem_cid=cid,
        canonical_smiles=canon_smiles,
        inchi_key=compound_data.get("inchikey"),
        iupac_name=compound_data.get("iupac_name"),
    )


def resolve_molecules(
    values: list[str],
    input_type: Literal["name", "smiles", "cid"] = "name",
) -> ResolutionReport:
    """Resolve small molecule identifiers to canonical structures.

    Parameters
    ----------
    values
        Compound names, SMILES strings, or PubChem CID strings.
    input_type
        Type of input: ``"name"``, ``"smiles"``, or ``"cid"``.

    Returns
    -------
    ResolutionReport
        One ``MoleculeResolution`` per input value.
    """
    if input_type == "name":
        # Batch-resolve via local DB first, then fall back per-name for unresolved
        non_controls = [v for v in values if not is_control_compound(v)]
        local_results = _resolve_batch_names_local(non_controls)

        results: list[MoleculeResolution] = []
        for v in values:
            if v in local_results:
                results.append(local_results[v])
            else:
                results.append(_resolve_single_name(v))
    else:
        resolver = {
            "smiles": _resolve_single_smiles,
            "cid": _resolve_single_cid,
        }[input_type]
        results = [resolver(v) for v in values]

    resolved_count = sum(1 for r in results if r.resolved_value is not None)
    ambiguous_count = sum(1 for r in results if len(r.alternatives) > 1)

    return ResolutionReport(
        total=len(values),
        resolved=resolved_count,
        unresolved=len(values) - resolved_count,
        ambiguous=ambiguous_count,
        results=results,
    )
