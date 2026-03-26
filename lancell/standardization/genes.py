"""Gene name/ID resolution against local LanceDB reference tables.

Primary resolution path uses local LanceDB tables (organisms, genomic_features,
genomic_feature_aliases) for fast, offline, deterministic resolution. The mygene
and ensembl REST utilities are retained as standalone helpers for enrichment.
"""

import re
from typing import Literal

import polars as pl
import requests

from lancell.standardization._rate_limit import rate_limited
from lancell.standardization.metadata_table import (
    GENOMIC_FEATURE_ALIASES_TABLE,
    GENOMIC_FEATURES_TABLE,
    ORGANISMS_TABLE,
    get_reference_db,
)
from lancell.standardization.types import GeneResolution, ResolutionReport
from lancell.util import sql_escape

_ENSEMBL_ID_RE = re.compile(r"^ENS[A-Z]*G\d+(\.\d+)?$")

# Accession-based placeholder symbols assigned by GENCODE (e.g., AC134879.3, AL590822.2)
_ACCESSION_PLACEHOLDER_RE = re.compile(r"^[A-Z]{2}\d{6}\.\d+$")

# Riken clone symbols from mouse datasets (e.g., 1700049J03Rik, 2410002F23Rik)
_RIKEN_CLONE_RE = re.compile(r"^\d+[A-Z]\d+Rik$")


def is_placeholder_symbol(symbol: str) -> bool:
    """Check if a gene symbol is an accession-based placeholder or Riken clone.

    These are provisional identifiers assigned by GENCODE or RIKEN to genes
    that lack a proper HGNC/MGI symbol — typically lncRNAs, pseudogenes, and
    antisense RNAs. They are valid identifiers but often fail resolution
    against canonical gene databases.
    """
    return bool(_ACCESSION_PLACEHOLDER_RE.match(symbol) or _RIKEN_CLONE_RE.match(symbol))


# ---------------------------------------------------------------------------
# Organism cache
# ---------------------------------------------------------------------------

_organism_list: list[dict] | None = None
_organism_by_common: dict[str, dict] | None = None
_organism_by_scientific: dict[str, dict] | None = None


def _load_all_organisms() -> list[dict]:
    """Load all organism records from the DB."""
    global _organism_list, _organism_by_common, _organism_by_scientific
    if _organism_list is not None:
        return _organism_list
    db = get_reference_db()
    table = db.open_table(ORGANISMS_TABLE)
    df = table.search().to_polars()
    _organism_list = list(df.iter_rows(named=True))
    _organism_by_common = {row["common_name"]: row for row in _organism_list}
    _organism_by_scientific = {row["scientific_name"]: row for row in _organism_list}
    return _organism_list


def _get_organism_record(organism: str) -> dict:
    """Look up an organism by common_name or scientific_name. Raises ValueError if unknown."""
    _load_all_organisms()
    if organism in _organism_by_common:
        return _organism_by_common[organism]
    if organism in _organism_by_scientific:
        return _organism_by_scientific[organism]
    raise ValueError(
        f"Unknown organism '{organism}'. Pass a common_name (e.g. 'human') "
        f"or scientific_name (e.g. 'homo_sapiens')."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_ensembl_id(value: str) -> bool:
    return bool(_ENSEMBL_ID_RE.match(value.split(".")[0]))


def _batch_lookup_features(ensembl_gene_ids: list[str], scientific_name: str) -> pl.DataFrame:
    """Batch lookup genomic_features by ensembl_gene_id, returning a polars DataFrame."""
    if not ensembl_gene_ids:
        return pl.DataFrame(
            schema={
                "ensembl_gene_id": pl.Utf8,
                "symbol": pl.Utf8,
                "ncbi_gene_id": pl.Int64,
                "organism": pl.Utf8,
            }
        )
    db = get_reference_db()
    table = db.open_table(GENOMIC_FEATURES_TABLE)
    frames: list[pl.DataFrame] = []
    for i in range(0, len(ensembl_gene_ids), 500):
        batch = ensembl_gene_ids[i : i + 500]
        in_clause = ", ".join(f"'{sql_escape(eid)}'" for eid in batch)
        df = (
            table.search()
            .where(
                f"ensembl_gene_id IN ({in_clause}) AND organism = '{sql_escape(scientific_name)}'",
                prefilter=True,
            )
            .select(["ensembl_gene_id", "symbol", "ncbi_gene_id", "organism", "assembly"])
            .to_polars()
        )
        frames.append(df)
    if not frames:
        return pl.DataFrame(
            schema={
                "ensembl_gene_id": pl.Utf8,
                "symbol": pl.Utf8,
                "ncbi_gene_id": pl.Int64,
                "organism": pl.Utf8,
            }
        )
    result = pl.concat(frames)
    # Prefer current assembly over older assemblies when gene exists in multiple
    if result.height > result.get_column("ensembl_gene_id").n_unique():
        result = result.sort("assembly", descending=True, nulls_last=True).unique(
            subset=["ensembl_gene_id"], keep="first"
        )
    return result.drop("assembly")


# ---------------------------------------------------------------------------
# Primary resolution: symbols via alias table
# ---------------------------------------------------------------------------


def _resolve_symbols(
    symbols: list[str],
    organism: str,
) -> dict[str, GeneResolution]:
    """Resolve gene symbols via the genomic_feature_aliases table."""
    if not symbols:
        return {}

    org_record = _get_organism_record(organism)
    scientific_name = org_record["scientific_name"]

    db = get_reference_db()
    table = db.open_table(GENOMIC_FEATURE_ALIASES_TABLE)

    # Lowercase all input symbols for matching
    lower_to_original: dict[str, str] = {}
    for sym in symbols:
        lower_to_original[sym.lower()] = sym

    # Batch query aliases
    alias_frames: list[pl.DataFrame] = []
    lower_symbols = list(lower_to_original.keys())
    for i in range(0, len(lower_symbols), 500):
        batch = lower_symbols[i : i + 500]
        in_clause = ", ".join(f"'{sql_escape(a)}'" for a in batch)
        df = (
            table.search()
            .where(
                f"alias IN ({in_clause}) AND organism = '{sql_escape(scientific_name)}'",
                prefilter=True,
            )
            .select(["alias", "alias_original", "ensembl_gene_id", "is_canonical"])
            .to_polars()
        )
        alias_frames.append(df)

    if not alias_frames:
        return {}

    aliases_df = pl.concat(alias_frames)
    if aliases_df.is_empty():
        return {}

    # Group by alias to handle disambiguation
    results: dict[str, GeneResolution] = {}
    grouped = aliases_df.group_by("alias").agg(pl.all())

    for row in grouped.iter_rows(named=True):
        alias_lower = row["alias"]
        original_sym = lower_to_original.get(alias_lower)
        if original_sym is None:
            continue

        ensembl_ids = row["ensembl_gene_id"]
        is_canonical_flags = row["is_canonical"]

        # Deduplicate by ensembl_gene_id
        seen: dict[str, bool] = {}
        for eid, is_can in zip(ensembl_ids, is_canonical_flags, strict=True):
            if eid not in seen:
                seen[eid] = is_can
            else:
                # Keep the canonical flag if any alias is canonical
                seen[eid] = seen[eid] or is_can

        unique_ids = list(seen.keys())
        canonical_ids = [eid for eid, is_can in seen.items() if is_can]

        if len(canonical_ids) == 1:
            best_id = canonical_ids[0]
            confidence = 1.0
        elif len(unique_ids) == 1:
            best_id = unique_ids[0]
            confidence = 0.9
        else:
            # Multiple matches — pick first by sorted ensembl_gene_id
            if canonical_ids:
                sorted_ids = sorted(canonical_ids)
            else:
                sorted_ids = sorted(unique_ids)
            best_id = sorted_ids[0]
            confidence = 0.7

        alternatives = sorted(eid for eid in unique_ids if eid != best_id)

        results[original_sym] = GeneResolution(
            input_value=original_sym,
            resolved_value=best_id,
            confidence=confidence,
            source="lancedb",
            ensembl_gene_id=best_id,
            organism=organism,
            alternatives=alternatives,
        )

    # Batch look up feature records for symbol and ncbi_gene_id
    resolved_ids = [r.ensembl_gene_id for r in results.values() if r.ensembl_gene_id]
    if resolved_ids:
        features_df = _batch_lookup_features(resolved_ids, scientific_name)
        feature_map = {row["ensembl_gene_id"]: row for row in features_df.iter_rows(named=True)}
        for res in results.values():
            feat = feature_map.get(res.ensembl_gene_id)
            if feat:
                res.symbol = feat["symbol"]
                res.ncbi_gene_id = feat["ncbi_gene_id"]

    return results


# ---------------------------------------------------------------------------
# Primary resolution: Ensembl IDs via features table
# ---------------------------------------------------------------------------


def _resolve_ensembl_ids(
    ensembl_ids: list[str],
    organism: str,
) -> dict[str, GeneResolution]:
    """Resolve Ensembl gene IDs via the genomic_features table."""
    if not ensembl_ids:
        return {}

    org_record = _get_organism_record(organism)
    scientific_name = org_record["scientific_name"]

    db = get_reference_db()
    table = db.open_table(GENOMIC_FEATURES_TABLE)

    # Strip version suffixes
    id_to_base: dict[str, str] = {}
    for eid in ensembl_ids:
        id_to_base[eid] = eid.split(".")[0]

    base_ids = list(set(id_to_base.values()))

    # Batch query features
    feature_frames: list[pl.DataFrame] = []
    for i in range(0, len(base_ids), 500):
        batch = base_ids[i : i + 500]
        in_clause = ", ".join(f"'{sql_escape(eid)}'" for eid in batch)
        df = (
            table.search()
            .where(
                f"ensembl_gene_id IN ({in_clause}) AND organism = '{sql_escape(scientific_name)}'",
                prefilter=True,
            )
            .select(["ensembl_gene_id", "symbol", "ncbi_gene_id", "assembly"])
            .to_polars()
        )
        feature_frames.append(df)

    feature_map: dict[str, dict] = {}
    if feature_frames:
        features_df = pl.concat(feature_frames)
        # Prefer current assembly over older assemblies
        if features_df.height > features_df.get_column("ensembl_gene_id").n_unique():
            features_df = features_df.sort("assembly", descending=True, nulls_last=True).unique(
                subset=["ensembl_gene_id"], keep="first"
            )
        features_df = features_df.drop("assembly")
        for row in features_df.iter_rows(named=True):
            feature_map[row["ensembl_gene_id"]] = row

    results: dict[str, GeneResolution] = {}
    for eid in ensembl_ids:
        base = id_to_base[eid]
        feat = feature_map.get(base)
        if feat:
            results[eid] = GeneResolution(
                input_value=eid,
                resolved_value=base,
                confidence=1.0,
                source="lancedb",
                symbol=feat["symbol"],
                ensembl_gene_id=base,
                organism=organism,
                ncbi_gene_id=feat["ncbi_gene_id"],
            )

    return results


# ---------------------------------------------------------------------------
# Organism detection from Ensembl ID prefixes
# ---------------------------------------------------------------------------


def detect_organism_from_ensembl_ids(ids: list[str]) -> dict[str, str]:
    """Infer organism for each Ensembl ID from its prefix.

    Returns a mapping from Ensembl ID -> organism common_name.
    Unknown prefixes map to ``"unknown"``.
    """
    orgs = _load_all_organisms()
    # Build prefix -> common_name map, sorted longest-first (skip organisms without prefix)
    prefix_map: list[tuple[str, str]] = sorted(
        [
            (rec["ensembl_prefix"], rec["common_name"])
            for rec in orgs
            if rec["ensembl_prefix"] is not None
        ],
        key=lambda x: len(x[0]),
        reverse=True,
    )

    result: dict[str, str] = {}
    for eid in ids:
        base_id = eid.split(".")[0]
        matched = False
        for prefix, common_name in prefix_map:
            if base_id.startswith(prefix):
                result[eid] = common_name
                matched = True
                break
        if not matched:
            result[eid] = "unknown"
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_genes(
    values: list[str],
    organism: str = "human",
    input_type: Literal["symbol", "ensembl_id", "auto"] = "auto",
) -> ResolutionReport:
    """Resolve gene symbols or Ensembl IDs to canonical identifiers.

    Parameters
    ----------
    values
        Gene symbols, Ensembl IDs, or a mix.
    organism
        Organism context for resolution (default ``"human"``).
    input_type
        ``"symbol"`` for gene symbols, ``"ensembl_id"`` for Ensembl IDs,
        ``"auto"`` to detect per-value.

    Returns
    -------
    ResolutionReport
        One ``GeneResolution`` per input value.
    """
    results: dict[str, GeneResolution] = {}

    # Classify inputs
    if input_type == "auto":
        symbols = [v for v in values if not _is_ensembl_id(v)]
        ensembl_ids = [v for v in values if _is_ensembl_id(v)]
    elif input_type == "symbol":
        symbols = list(values)
        ensembl_ids = []
    else:
        symbols = []
        ensembl_ids = list(values)

    # Resolve symbols via alias table
    if symbols:
        symbol_results = _resolve_symbols(symbols, organism)
        results.update(symbol_results)

    # Resolve Ensembl IDs — detect organism per-ID from prefix
    if ensembl_ids:
        id_organisms = detect_organism_from_ensembl_ids(ensembl_ids)
        ids_by_organism: dict[str, list[str]] = {}
        for eid in ensembl_ids:
            org = id_organisms.get(eid, organism)
            if org == "unknown":
                org = organism  # fall back to caller-specified organism
            ids_by_organism.setdefault(org, []).append(eid)
        for org, org_ids in ids_by_organism.items():
            ensembl_results = _resolve_ensembl_ids(org_ids, org)
            results.update(ensembl_results)

    # Build final results list aligned with input
    final_results: list[GeneResolution] = []
    for v in values:
        if v in results:
            final_results.append(results[v])
        else:
            final_results.append(
                GeneResolution(
                    input_value=v,
                    resolved_value=None,
                    confidence=0.0,
                    source="none",
                    organism=organism,
                )
            )

    resolved_count = sum(1 for r in final_results if r.resolved_value is not None)
    ambiguous_count = sum(1 for r in final_results if len(r.alternatives) > 0)

    return ResolutionReport(
        total=len(values),
        resolved=resolved_count,
        unresolved=len(values) - resolved_count,
        ambiguous=ambiguous_count,
        results=final_results,
    )


# ---------------------------------------------------------------------------
# Standalone utilities (not in primary resolution path)
# ---------------------------------------------------------------------------

_ENSEMBL_REST_BASE = "https://rest.ensembl.org"


def _resolve_symbols_mygene(
    symbols: list[str],
    organism: str,
) -> dict[str, GeneResolution]:
    """Alias resolution: query MyGene.info for symbols."""
    import mygene

    results: dict[str, GeneResolution] = {}
    if not symbols:
        return results

    org_record = _get_organism_record(organism)
    species = org_record["common_name"]
    mg = mygene.MyGeneInfo()
    response = mg.querymany(
        symbols,
        scopes="symbol,alias",
        fields="symbol,ensembl.gene,entrezgene",
        species=species,
        returnall=True,
    )

    hits_by_query: dict[str, list[dict]] = {}
    for hit in response.get("out", []):
        query = hit.get("query", "")
        hits_by_query.setdefault(query, []).append(hit)

    for sym in symbols:
        hits = hits_by_query.get(sym, [])
        valid_hits = [h for h in hits if not h.get("notfound", False)]

        if not valid_hits:
            continue

        best = valid_hits[0]
        canonical_symbol = best.get("symbol")
        ensembl_data = best.get("ensembl", {})
        if isinstance(ensembl_data, list):
            ensembl_data = ensembl_data[0]
        ensembl_id = ensembl_data.get("gene") if isinstance(ensembl_data, dict) else None
        ncbi_id = best.get("entrezgene")
        if ncbi_id is not None:
            ncbi_id = int(ncbi_id)

        alternatives = [h.get("symbol", "") for h in valid_hits[1:] if h.get("symbol")]

        confidence = 0.8 if len(valid_hits) == 1 else 0.6
        results[sym] = GeneResolution(
            input_value=sym,
            resolved_value=canonical_symbol,
            confidence=confidence,
            source="mygene",
            alternatives=alternatives,
            symbol=canonical_symbol,
            ensembl_gene_id=ensembl_id,
            organism=organism,
            ncbi_gene_id=ncbi_id,
        )

    return results


@rate_limited("ensembl")
def _ensembl_rest_post_symbols(species: str, symbols: list[str]) -> dict:
    """POST batch symbol lookup to Ensembl REST. Raises on HTTP errors for retry."""
    resp = requests.post(
        f"{_ENSEMBL_REST_BASE}/lookup/symbol/{species}",
        json={"symbols": symbols},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _resolve_symbols_ensembl_rest(
    symbols: list[str],
    organism: str,
) -> dict[str, GeneResolution]:
    """Resolve gene symbols via Ensembl REST batch lookup."""
    results: dict[str, GeneResolution] = {}
    if not symbols:
        return results

    org_record = _get_organism_record(organism)
    species = org_record["ensembl_species_name"]

    # Batch in chunks of 1000
    for i in range(0, len(symbols), 1000):
        batch = symbols[i : i + 1000]
        try:
            response = _ensembl_rest_post_symbols(species, batch)
        except Exception:
            continue

        for sym in batch:
            hit = response.get(sym)
            if hit is None or not isinstance(hit, dict):
                continue

            ensembl_id = hit.get("id")
            display_name = hit.get("display_name") or sym

            results[sym] = GeneResolution(
                input_value=sym,
                resolved_value=display_name,
                confidence=0.85,
                source="ensembl_rest",
                symbol=display_name,
                ensembl_gene_id=ensembl_id,
                organism=organism,
            )

    return results


def _resolve_ensembl_ids_mygene(
    ensembl_ids: list[str],
    organism: str,
) -> dict[str, GeneResolution]:
    """Validate and resolve Ensembl IDs via MyGene.info."""
    import mygene

    results: dict[str, GeneResolution] = {}
    if not ensembl_ids:
        return results

    # Strip version suffixes for lookup
    id_to_base: dict[str, str] = {}
    for eid in ensembl_ids:
        id_to_base[eid] = eid.split(".")[0]

    base_ids = [id_to_base[eid] for eid in ensembl_ids]
    mg = mygene.MyGeneInfo()
    response = mg.getgenes(base_ids, fields="symbol,ensembl.gene,entrezgene")

    for eid, hit in zip(ensembl_ids, response, strict=False):
        if hit is None or hit.get("notfound", False):
            continue

        symbol = hit.get("symbol")
        ncbi_id = hit.get("entrezgene")
        if ncbi_id is not None:
            ncbi_id = int(ncbi_id)
        base = id_to_base[eid]

        results[eid] = GeneResolution(
            input_value=eid,
            resolved_value=symbol,
            confidence=0.95,
            source="mygene",
            symbol=symbol,
            ensembl_gene_id=base,
            organism=organism,
            ncbi_gene_id=ncbi_id,
        )

    return results
