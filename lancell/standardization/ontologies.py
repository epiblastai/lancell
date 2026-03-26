"""Ontology term resolution against local LanceDB reference tables.

Covers: cell_type (CL), tissue (UBERON), disease (MONDO), organism (NCBITaxon),
assay (EFO), development_stage (HsapDv/MmusDv), ethnicity (HANCESTRO),
sex (PATO), cell_line (Cellosaurus).

Strategy:
1. Exact name match (case-insensitive) against ontology_terms table → confidence 1.0
2. Synonym match (pipe-delimited, case-insensitive) → confidence 0.9
3. No match → unresolved (confidence 0.0)

Cell lines use a local Cellosaurus table (cell_lines + cell_line_synonyms).
"""

import functools
from collections import defaultdict
from enum import Enum

import polars as pl

from lancell.standardization.metadata_table import (
    CELL_LINE_SYNONYMS_TABLE,
    CELL_LINES_TABLE,
    ONTOLOGY_TERMS_TABLE,
    get_reference_db,
)
from lancell.standardization.types import CellLineResolution, OntologyResolution, ResolutionReport
from lancell.util import sql_escape


class OntologyEntity(str, Enum):
    """Supported ontology entity types for CELLxGENE-compatible resolution."""

    CELL_TYPE = "cell_type"
    CELL_LINE = "cell_line"
    TISSUE = "tissue"
    DISEASE = "disease"
    ORGANISM = "organism"
    ASSAY = "assay"
    DEVELOPMENT_STAGE = "development_stage"
    ETHNICITY = "ethnicity"
    SEX = "sex"


# Mapping from OntologyEntity → ontology prefix(es) in the reference DB
_ENTITY_TO_PREFIXES: dict[OntologyEntity, list[str]] = {
    OntologyEntity.CELL_TYPE: ["CL"],
    OntologyEntity.TISSUE: ["UBERON"],
    OntologyEntity.DISEASE: ["MONDO"],
    OntologyEntity.ORGANISM: ["NCBITaxon"],
    OntologyEntity.ASSAY: ["EFO"],
    OntologyEntity.DEVELOPMENT_STAGE: ["HsapDv", "MmusDv"],
    OntologyEntity.ETHNICITY: ["HANCESTRO"],
}

# Mapping from OntologyEntity → display name
_ENTITY_TO_ONTOLOGY_NAME: dict[OntologyEntity, str] = {
    OntologyEntity.CELL_TYPE: "Cell Ontology",
    OntologyEntity.CELL_LINE: "Cellosaurus",
    OntologyEntity.TISSUE: "UBERON",
    OntologyEntity.DISEASE: "MONDO",
    OntologyEntity.ORGANISM: "NCBITaxon",
    OntologyEntity.ASSAY: "EFO",
    OntologyEntity.DEVELOPMENT_STAGE: "HsapDv",
    OntologyEntity.ETHNICITY: "HANCESTRO",
    OntologyEntity.SEX: "PATO",
}

# Development stage prefix selection by organism
_DEVELOPMENT_STAGE_ORGANISM_PREFIX: dict[str, str] = {
    "human": "HsapDv",
    "homo_sapiens": "HsapDv",
    "mouse": "MmusDv",
    "mus_musculus": "MmusDv",
}

# Hard-coded sex terms (PATO terms are not in the OBO download)
_SEX_TERMS: dict[str, tuple[str, str]] = {
    "female": ("PATO:0000383", "female"),
    "male": ("PATO:0000384", "male"),
    "unknown": ("PATO:0000461", "unknown sex"),
    "other": ("PATO:0000461", "unknown sex"),
}


# ---------------------------------------------------------------------------
# Ontology data loading (cached)
# ---------------------------------------------------------------------------


def _get_prefixes(entity: OntologyEntity, organism: str | None = None) -> list[str]:
    """Get the ontology prefix(es) for an entity, considering organism for dev stage."""
    if entity == OntologyEntity.DEVELOPMENT_STAGE and organism:
        prefix = _DEVELOPMENT_STAGE_ORGANISM_PREFIX.get(organism.lower())
        if prefix:
            return [prefix]
    prefixes = _ENTITY_TO_PREFIXES.get(entity)
    if prefixes is None:
        raise ValueError(f"No ontology prefix mapping for entity {entity}")
    return prefixes


@functools.lru_cache(maxsize=32)
def _load_ontology_terms(prefix: str) -> pl.DataFrame:
    """Load all non-obsolete terms for a given ontology prefix. Cached."""
    db = get_reference_db()
    table = db.open_table(ONTOLOGY_TERMS_TABLE)
    return (
        table.search()
        .where(
            f"ontology_prefix = '{sql_escape(prefix)}' AND is_obsolete = false",
            prefilter=True,
        )
        .select(["ontology_term_id", "name", "synonyms", "parent_ids"])
        .to_polars()
    )


@functools.lru_cache(maxsize=32)
def _build_name_index(prefix: str) -> dict[str, tuple[str, str]]:
    """Build lowercased name → (ontology_term_id, canonical_name) index."""
    df = _load_ontology_terms(prefix)
    index: dict[str, tuple[str, str]] = {}
    for row in df.iter_rows(named=True):
        key = row["name"].strip().lower()
        if key not in index:
            index[key] = (row["ontology_term_id"], row["name"])
    return index


@functools.lru_cache(maxsize=32)
def _build_synonym_index(prefix: str) -> dict[str, tuple[str, str, str]]:
    """Build lowercased synonym → (ontology_term_id, canonical_name, synonym_original) index."""
    df = _load_ontology_terms(prefix)
    index: dict[str, tuple[str, str, str]] = {}
    for row in df.iter_rows(named=True):
        synonyms = row["synonyms"]
        if not synonyms:
            continue
        for syn in synonyms.split(" | "):
            syn_stripped = syn.strip()
            key = syn_stripped.lower()
            if key not in index:
                index[key] = (row["ontology_term_id"], row["name"], syn_stripped)
    return index


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _resolve_sex(value: str) -> OntologyResolution:
    """Resolve sex terms to PATO ontology IDs."""
    v = value.strip().lower()
    if v in _SEX_TERMS:
        term_id, label = _SEX_TERMS[v]
        return OntologyResolution(
            input_value=value,
            resolved_value=label,
            confidence=1.0,
            source="pato_hardcoded",
            ontology_term_id=term_id,
            ontology_name="PATO",
        )
    return OntologyResolution(
        input_value=value,
        resolved_value=None,
        confidence=0.0,
        source="none",
        ontology_name="PATO",
    )


@functools.lru_cache(maxsize=1)
def _load_cell_lines() -> pl.DataFrame:
    """Load all cell lines from the reference DB. Cached."""
    db = get_reference_db()
    table = db.open_table(CELL_LINES_TABLE)
    return (
        table.search()
        .select(
            [
                "cellosaurus_id",
                "cell_line_name",
                "species",
                "ncbi_taxonomy_id",
                "disease",
                "sex",
                "category",
            ]
        )
        .to_polars()
    )


@functools.lru_cache(maxsize=1)
def _build_cell_line_name_index() -> dict[str, str]:
    """Build lowercased cell line name → cellosaurus_id index."""
    df = _load_cell_lines()
    index: dict[str, str] = {}
    for row in df.iter_rows(named=True):
        key = row["cell_line_name"].strip().lower()
        if key not in index:
            index[key] = row["cellosaurus_id"]
    return index


@functools.lru_cache(maxsize=1)
def _build_cell_line_synonym_index() -> dict[str, str]:
    """Build lowercased synonym → cellosaurus_id index (non-primary names only)."""
    db = get_reference_db()
    table = db.open_table(CELL_LINE_SYNONYMS_TABLE)
    df = (
        table.search()
        .where("is_primary_name = false", prefilter=True)
        .select(["synonym", "cellosaurus_id"])
        .to_polars()
    )
    index: dict[str, str] = {}
    for row in df.iter_rows(named=True):
        key = row["synonym"]  # already lowercased at ingestion
        if key not in index:
            index[key] = row["cellosaurus_id"]
    return index


@functools.lru_cache(maxsize=1)
def _build_cell_line_record_lookup() -> dict[str, dict]:
    """Build cellosaurus_id → row dict for fast metadata retrieval."""
    df = _load_cell_lines()
    return {row["cellosaurus_id"]: row for row in df.iter_rows(named=True)}


def _make_cell_line_resolution(
    input_value: str,
    cellosaurus_id: str,
    confidence: float,
    source: str,
    record_lookup: dict[str, dict],
) -> CellLineResolution:
    """Build a CellLineResolution from a matched cellosaurus_id."""
    rec = record_lookup.get(cellosaurus_id, {})
    return CellLineResolution(
        input_value=input_value,
        resolved_value=rec.get("cell_line_name"),
        confidence=confidence,
        source=source,
        cellosaurus_id=cellosaurus_id,
        cell_line_name=rec.get("cell_line_name"),
        species=rec.get("species"),
        disease=rec.get("disease"),
        sex=rec.get("sex"),
        category=rec.get("category"),
    )


def _resolve_cell_lines(values: list[str]) -> list[CellLineResolution]:
    """Resolve cell line names against the local Cellosaurus reference DB.

    Resolution cascade:
    1. Exact name match (confidence 1.0)
    2. Synonym match (confidence 0.9)
    3. FTS fuzzy search (confidence 0.7)
    4. Unresolved (confidence 0.0)
    """
    name_index = _build_cell_line_name_index()
    synonym_index = _build_cell_line_synonym_index()
    record_lookup = _build_cell_line_record_lookup()

    results: list[CellLineResolution] = []
    for val in values:
        key = val.strip().lower()

        if not key:
            results.append(
                CellLineResolution(
                    input_value=val,
                    resolved_value=None,
                    confidence=0.0,
                    source="none",
                )
            )
            continue

        # Step 1: exact name match
        if key in name_index:
            results.append(
                _make_cell_line_resolution(val, name_index[key], 1.0, "reference_db", record_lookup)
            )
            continue

        # Step 2: synonym match
        if key in synonym_index:
            results.append(
                _make_cell_line_resolution(
                    val, synonym_index[key], 0.9, "reference_db_synonym", record_lookup
                )
            )
            continue

        # Step 3: FTS fuzzy search
        db = get_reference_db()
        fts_table = db.open_table(CELL_LINE_SYNONYMS_TABLE)
        fts_results = fts_table.search(val.strip(), query_type="fts").limit(5).to_polars()
        if not fts_results.is_empty():
            top_id = fts_results.row(0, named=True)["cellosaurus_id"]
            results.append(
                _make_cell_line_resolution(val, top_id, 0.7, "reference_db_fts", record_lookup)
            )
            continue

        # Step 4: unresolved
        results.append(
            CellLineResolution(
                input_value=val,
                resolved_value=None,
                confidence=0.0,
                source="none",
            )
        )

    return results


def _resolve_against_db(
    values: list[str],
    entity: OntologyEntity,
    organism: str | None = None,
) -> list[OntologyResolution]:
    """Resolve values against the local ontology_terms table."""
    ontology_name = _ENTITY_TO_ONTOLOGY_NAME.get(entity, entity.value)
    prefixes = _get_prefixes(entity, organism)

    # Build combined name and synonym indices across all relevant prefixes
    name_index: dict[str, tuple[str, str]] = {}
    synonym_index: dict[str, tuple[str, str, str]] = {}
    for prefix in prefixes:
        name_index.update(_build_name_index(prefix))
        synonym_index.update(_build_synonym_index(prefix))

    results: list[OntologyResolution] = []
    for val in values:
        key = val.strip().lower()

        # Step 1: exact name match
        if key in name_index:
            term_id, canonical_name = name_index[key]
            results.append(
                OntologyResolution(
                    input_value=val,
                    resolved_value=canonical_name,
                    confidence=1.0,
                    source="reference_db",
                    ontology_term_id=term_id,
                    ontology_name=ontology_name,
                )
            )
            continue

        # Step 2: synonym match
        if key in synonym_index:
            term_id, canonical_name, _syn_original = synonym_index[key]
            results.append(
                OntologyResolution(
                    input_value=val,
                    resolved_value=canonical_name,
                    confidence=0.9,
                    source="reference_db_synonym",
                    ontology_term_id=term_id,
                    ontology_name=ontology_name,
                )
            )
            continue

        # Step 3: unresolved
        results.append(
            OntologyResolution(
                input_value=val,
                resolved_value=None,
                confidence=0.0,
                source="none",
                ontology_name=ontology_name,
            )
        )

    return results


def resolve_ontology_terms(
    values: list[str],
    entity: OntologyEntity,
    organism: str | None = None,
    min_similarity: float = 0.8,
) -> ResolutionReport:
    """Resolve free-text values to ontology terms with CELLxGENE-compatible IDs.

    Parameters
    ----------
    values
        Free-text metadata values.
    entity
        Which ontology entity to resolve against.
    organism
        Organism context (required for development_stage, ignored for most others).
    min_similarity
        Minimum fuzzy match score (0-1) to accept a match. Currently unused
        since resolution is exact name/synonym only, kept for API compatibility.

    Returns
    -------
    ResolutionReport
        One ``OntologyResolution`` per input value.
    """
    if entity == OntologyEntity.SEX:
        results = [_resolve_sex(v) for v in values]
    elif entity == OntologyEntity.CELL_LINE:
        results = _resolve_cell_lines(values)
    else:
        results = _resolve_against_db(values, entity, organism)

    resolved_count = sum(1 for r in results if r.resolved_value is not None)
    ambiguous_count = sum(1 for r in results if len(r.alternatives) > 1)

    return ResolutionReport(
        total=len(values),
        resolved=resolved_count,
        unresolved=len(values) - resolved_count,
        ambiguous=ambiguous_count,
        results=results,
    )


def get_ontology_term_id(
    value: str,
    entity: OntologyEntity,
    organism: str | None = None,
) -> str | None:
    """Convenience: resolve a single value and return just the ontology term ID."""
    report = resolve_ontology_terms([value], entity, organism)
    r = report.results[0]
    if isinstance(r, OntologyResolution):
        return r.ontology_term_id
    return None


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def resolve_cell_types(values: list[str]) -> ResolutionReport:
    """Resolve cell type names to Cell Ontology (CL) terms."""
    return resolve_ontology_terms(values, OntologyEntity.CELL_TYPE)


def resolve_tissues(values: list[str]) -> ResolutionReport:
    """Resolve tissue names to UBERON terms."""
    return resolve_ontology_terms(values, OntologyEntity.TISSUE)


def resolve_diseases(values: list[str]) -> ResolutionReport:
    """Resolve disease names to MONDO terms."""
    return resolve_ontology_terms(values, OntologyEntity.DISEASE)


def resolve_organisms(values: list[str]) -> ResolutionReport:
    """Resolve organism names to NCBITaxon terms."""
    return resolve_ontology_terms(values, OntologyEntity.ORGANISM)


def resolve_assays(values: list[str]) -> ResolutionReport:
    """Resolve assay names to EFO terms."""
    return resolve_ontology_terms(values, OntologyEntity.ASSAY)


def resolve_cell_lines(values: list[str]) -> ResolutionReport:
    """Resolve cell line names to Cellosaurus cell line records."""
    return resolve_ontology_terms(values, OntologyEntity.CELL_LINE)


# ---------------------------------------------------------------------------
# Ontology hierarchy navigation
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _build_term_lookup(prefix: str) -> dict[str, dict]:
    """Build term_id → {name, parent_ids} lookup from cached terms."""
    df = _load_ontology_terms(prefix)
    lookup: dict[str, dict] = {}
    for row in df.iter_rows(named=True):
        lookup[row["ontology_term_id"]] = {
            "name": row["name"],
            "parent_ids": row["parent_ids"],
        }
    return lookup


@functools.lru_cache(maxsize=32)
def _build_children_index(prefix: str) -> dict[str, list[str]]:
    """Build reverse index: parent_id → [child_ids]."""
    lookup = _build_term_lookup(prefix)
    children: dict[str, list[str]] = defaultdict(list)
    for term_id, info in lookup.items():
        for pid in info["parent_ids"]:
            children[pid].append(term_id)
    return dict(children)


def _prefix_from_term_id(term_id: str) -> str:
    """Extract ontology prefix from a CURIE, e.g. 'CL:0000540' → 'CL'."""
    if ":" not in term_id:
        raise ValueError(
            f"Invalid ontology term ID '{term_id}' — expected CURIE format (e.g. 'CL:0000540')"
        )
    return term_id.split(":")[0]


def get_ontology_ancestors(
    term_id: str,
    entity: OntologyEntity,
    organism: str | None = None,
    max_depth: int | None = None,
) -> list[tuple[str, str]]:
    """Walk up the ontology hierarchy and return ancestors (closest first).

    Parameters
    ----------
    term_id
        Ontology term ID (e.g., ``"CL:0000540"``).
    entity
        Which ontology to query.
    organism
        Organism context (usually ``None``).
    max_depth
        Maximum number of hops upward. ``None`` means no limit.

    Returns
    -------
    list[tuple[str, str]]
        ``(ontology_id, name)`` pairs, closest ancestors first.

    Raises
    ------
    ValueError
        If *term_id* is not found in the ontology.
    """
    prefix = _prefix_from_term_id(term_id)
    lookup = _build_term_lookup(prefix)
    if term_id not in lookup:
        raise ValueError(f"Term '{term_id}' not found in {entity.value} ontology")

    ancestors: list[tuple[str, str]] = []
    visited: set[str] = {term_id}
    frontier: list[str] = list(lookup[term_id]["parent_ids"])
    depth = 0

    while frontier:
        if max_depth is not None and depth >= max_depth:
            break
        next_frontier: list[str] = []
        for pid in frontier:
            if pid in visited or pid not in lookup:
                continue
            visited.add(pid)
            ancestors.append((pid, lookup[pid]["name"]))
            next_frontier.extend(lookup[pid]["parent_ids"])
        frontier = next_frontier
        depth += 1

    return ancestors


def get_ontology_descendants(
    term_id: str,
    entity: OntologyEntity,
    organism: str | None = None,
    max_depth: int | None = None,
) -> list[tuple[str, str]]:
    """Walk down the ontology hierarchy and return descendants (closest first).

    Parameters
    ----------
    term_id
        Ontology term ID.
    entity
        Which ontology to query.
    organism
        Organism context (usually ``None``).
    max_depth
        Maximum number of hops downward. ``None`` means no limit.

    Returns
    -------
    list[tuple[str, str]]
        ``(ontology_id, name)`` pairs, closest descendants first.

    Raises
    ------
    ValueError
        If *term_id* is not found in the ontology.
    """
    prefix = _prefix_from_term_id(term_id)
    lookup = _build_term_lookup(prefix)
    if term_id not in lookup:
        raise ValueError(f"Term '{term_id}' not found in {entity.value} ontology")

    children_index = _build_children_index(prefix)

    descendants: list[tuple[str, str]] = []
    visited: set[str] = {term_id}
    frontier: list[str] = children_index.get(term_id, [])
    depth = 0

    while frontier:
        if max_depth is not None and depth >= max_depth:
            break
        next_frontier: list[str] = []
        for cid in frontier:
            if cid in visited or cid not in lookup:
                continue
            visited.add(cid)
            descendants.append((cid, lookup[cid]["name"]))
            next_frontier.extend(children_index.get(cid, []))
        frontier = next_frontier
        depth += 1

    return descendants


def get_ontology_siblings(
    term_id: str,
    entity: OntologyEntity,
    organism: str | None = None,
) -> list[tuple[str, str]]:
    """Return siblings — other children of the same parent(s), excluding self.

    Parameters
    ----------
    term_id
        Ontology term ID.
    entity
        Which ontology to query.
    organism
        Organism context (usually ``None``).

    Returns
    -------
    list[tuple[str, str]]
        ``(ontology_id, name)`` pairs for sibling terms.

    Raises
    ------
    ValueError
        If *term_id* is not found in the ontology.
    """
    prefix = _prefix_from_term_id(term_id)
    lookup = _build_term_lookup(prefix)
    if term_id not in lookup:
        raise ValueError(f"Term '{term_id}' not found in {entity.value} ontology")

    children_index = _build_children_index(prefix)
    parents = lookup[term_id]["parent_ids"]

    siblings: dict[str, str] = {}
    for pid in parents:
        for cid in children_index.get(pid, []):
            if cid != term_id and cid not in siblings and cid in lookup:
                siblings[cid] = lookup[cid]["name"]

    return list(siblings.items())
