"""Protein name/ID resolution against local LanceDB reference tables.

Resolves protein names, gene names, and UniProt accessions to canonical
UniProt IDs using the proteins and protein_aliases tables.
"""

import polars as pl

from lancell.standardization.genes import _get_organism_record
from lancell.standardization.metadata_table import (
    PROTEIN_ALIASES_TABLE,
    PROTEINS_TABLE,
    get_reference_db,
)
from lancell.standardization.types import ProteinResolution, ResolutionReport
from lancell.util import sql_escape


def _batch_lookup_proteins(uniprot_ids: list[str]) -> dict[str, dict]:
    """Batch lookup protein records by uniprot_id, returning a map of id -> record."""
    if not uniprot_ids:
        return {}
    db = get_reference_db()
    table = db.open_table(PROTEINS_TABLE)
    frames: list[pl.DataFrame] = []
    for i in range(0, len(uniprot_ids), 500):
        batch = uniprot_ids[i : i + 500]
        in_clause = ", ".join(f"'{sql_escape(uid)}'" for uid in batch)
        df = (
            table.search()
            .where(f"uniprot_id IN ({in_clause})", prefilter=True)
            .select(
                [
                    "uniprot_id",
                    "protein_name",
                    "gene_name",
                    "organism",
                    "sequence",
                    "sequence_length",
                ]
            )
            .to_polars()
        )
        frames.append(df)
    if not frames:
        return {}
    result = pl.concat(frames)
    return {row["uniprot_id"]: row for row in result.iter_rows(named=True)}


def resolve_proteins(
    values: list[str],
    organism: str = "human",
) -> ResolutionReport:
    """Resolve protein names or UniProt accessions to canonical UniProt IDs.

    Parameters
    ----------
    values
        Protein names, gene names, UniProt accessions, or a mix.
    organism
        Organism context for resolution (default ``"human"``).

    Returns
    -------
    ResolutionReport
        One ``ProteinResolution`` per input value.
    """
    if not values:
        return ResolutionReport(total=0, resolved=0, unresolved=0, ambiguous=0, results=[])

    org_record = _get_organism_record(organism)
    scientific_name = org_record["scientific_name"]

    db = get_reference_db()
    alias_table = db.open_table(PROTEIN_ALIASES_TABLE)

    # Lowercase all input values for matching
    lower_to_original: dict[str, str] = {}
    for v in values:
        lower_to_original[v.lower()] = v

    # Batch query aliases
    alias_frames: list[pl.DataFrame] = []
    lower_values = list(lower_to_original.keys())
    for i in range(0, len(lower_values), 500):
        batch = lower_values[i : i + 500]
        in_clause = ", ".join(f"'{sql_escape(a)}'" for a in batch)
        df = (
            alias_table.search()
            .where(
                f"alias IN ({in_clause}) AND organism = '{sql_escape(scientific_name)}'",
                prefilter=True,
            )
            .select(["alias", "alias_original", "uniprot_id", "is_canonical"])
            .to_polars()
        )
        alias_frames.append(df)

    # Build resolution results from alias matches
    results: dict[str, ProteinResolution] = {}

    if alias_frames:
        aliases_df = pl.concat(alias_frames)
        if not aliases_df.is_empty():
            grouped = aliases_df.group_by("alias").agg(pl.all())

            for row in grouped.iter_rows(named=True):
                alias_lower = row["alias"]
                original_val = lower_to_original.get(alias_lower)
                if original_val is None:
                    continue

                uniprot_ids = row["uniprot_id"]
                is_canonical_flags = row["is_canonical"]

                # Deduplicate by uniprot_id
                seen: dict[str, bool] = {}
                for uid, is_can in zip(uniprot_ids, is_canonical_flags, strict=True):
                    if uid not in seen:
                        seen[uid] = is_can
                    else:
                        seen[uid] = seen[uid] or is_can

                unique_ids = list(seen.keys())
                canonical_ids = [uid for uid, is_can in seen.items() if is_can]

                if len(canonical_ids) == 1:
                    best_id = canonical_ids[0]
                    confidence = 1.0
                elif len(unique_ids) == 1:
                    best_id = unique_ids[0]
                    confidence = 0.9
                else:
                    if canonical_ids:
                        sorted_ids = sorted(canonical_ids)
                    else:
                        sorted_ids = sorted(unique_ids)
                    best_id = sorted_ids[0]
                    confidence = 0.7

                alternatives = sorted(uid for uid in unique_ids if uid != best_id)

                results[original_val] = ProteinResolution(
                    input_value=original_val,
                    resolved_value=best_id,
                    confidence=confidence,
                    source="lancedb",
                    uniprot_id=best_id,
                    organism=organism,
                    alternatives=alternatives,
                )

    # Batch lookup protein records to enrich with protein_name and gene_name
    resolved_ids = [r.uniprot_id for r in results.values() if r.uniprot_id]
    if resolved_ids:
        protein_map = _batch_lookup_proteins(list(set(resolved_ids)))
        for res in results.values():
            prot = protein_map.get(res.uniprot_id)
            if prot:
                res.protein_name = prot["protein_name"]
                res.gene_name = prot["gene_name"]
                res.sequence = prot["sequence"]
                res.sequence_length = prot["sequence_length"]

    # Build final results list aligned with input order
    final_results: list[ProteinResolution] = []
    for v in values:
        if v in results:
            final_results.append(results[v])
        else:
            final_results.append(
                ProteinResolution(
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
