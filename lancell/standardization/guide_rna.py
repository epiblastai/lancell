"""Cache-aware guide RNA resolution.

Wraps the BLAT + Ensembl pipeline in ``gget.py`` with a local LanceDB
cache (``guide_rnas`` table). Previously resolved sequences are returned
instantly; cache misses go through BLAT and are saved for future reuse.
"""

import logging

from lancell.standardization.gget import _ASSEMBLY_MAP, _SPECIES_MAP, _resolve_single
from lancell.standardization.metadata_table import (
    DEFAULT_REFERENCE_DB_PATH,
    GUIDE_RNAS_TABLE,
    GuideRnaRecord,
    _custom_db_path,
    open_reference_db,
)
from lancell.standardization.types import GuideRnaResolution, ResolutionReport
from lancell.util import sql_escape

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB access
# ---------------------------------------------------------------------------


def _get_cache_db():
    """Get DB connection for the guide RNA cache, creating the directory if needed.

    Unlike ``get_reference_db()``, this does not raise if the DB directory
    does not exist — the guide RNA cache is populated lazily during
    resolution rather than by a download script.
    """
    db_path = _custom_db_path or DEFAULT_REFERENCE_DB_PATH
    return open_reference_db(db_path)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _record_to_resolution(row: dict, original_input: str) -> GuideRnaResolution:
    """Convert a cached DB row to a GuideRnaResolution."""
    alternatives = row["alternatives"].split("|") if row["alternatives"] else []
    return GuideRnaResolution(
        input_value=original_input,
        resolved_value=row["resolved_value"],
        confidence=row["confidence"],
        source="lancedb_cache",
        alternatives=alternatives,
        chromosome=row["chromosome"],
        target_start=row["target_start"],
        target_end=row["target_end"],
        target_strand=row["target_strand"],
        intended_gene_name=row["intended_gene_name"],
        intended_ensembl_gene_id=row["intended_ensembl_gene_id"],
        target_context=row["target_context"],
        assembly=row["assembly"],
        blat_pct_match=row["blat_pct_match"],
    )


def _resolution_to_record(res: GuideRnaResolution, species: str) -> dict:
    """Convert a fresh GuideRnaResolution to a dict for DB insertion."""
    return {
        "guide_sequence": res.input_value.upper(),
        "organism": species,
        "chromosome": res.chromosome,
        "target_start": res.target_start,
        "target_end": res.target_end,
        "target_strand": res.target_strand,
        "intended_gene_name": res.intended_gene_name,
        "intended_ensembl_gene_id": res.intended_ensembl_gene_id,
        "target_context": res.target_context,
        "assembly": res.assembly,
        "blat_pct_match": res.blat_pct_match,
        "confidence": res.confidence,
        "resolved_value": res.resolved_value,
        "alternatives": "|".join(res.alternatives) if res.alternatives else None,
    }


# ---------------------------------------------------------------------------
# Cache read / write
# ---------------------------------------------------------------------------


def _lookup_cached(sequences: list[str], species: str) -> dict[str, dict]:
    """Batch lookup cached guide RNA results.

    Returns ``{uppercase_sequence: row_dict}`` for sequences found in cache.
    """
    db = _get_cache_db()
    if GUIDE_RNAS_TABLE not in db.list_tables().tables:
        return {}

    table = db.open_table(GUIDE_RNAS_TABLE)
    cached: dict[str, dict] = {}

    for i in range(0, len(sequences), 500):
        batch = sequences[i : i + 500]
        in_clause = ", ".join(f"'{sql_escape(seq)}'" for seq in batch)
        df = (
            table.search()
            .where(
                f"guide_sequence IN ({in_clause}) AND organism = '{sql_escape(species)}'",
                prefilter=True,
            )
            .to_polars()
        )
        for row in df.iter_rows(named=True):
            cached[row["guide_sequence"]] = row

    return cached


def _save_to_cache(records: list[dict]) -> None:
    """Append new resolution results to the guide RNA cache table."""
    if not records:
        return
    db = _get_cache_db()
    if GUIDE_RNAS_TABLE in db.list_tables().tables:
        table = db.open_table(GUIDE_RNAS_TABLE)
        table.add(records)
    else:
        db.create_table(GUIDE_RNAS_TABLE, data=records, schema=GuideRnaRecord)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_guide_sequences(
    sequences: list[str],
    organism: str = "human",
) -> ResolutionReport:
    """Resolve guide RNA sequences to genomic coordinates and gene annotations.

    Checks the local LanceDB cache first. Cache misses are resolved via
    UCSC BLAT + Ensembl REST, and results are saved for future reuse.

    Parameters
    ----------
    sequences
        List of guide RNA sequences (typically 20bp DNA strings).
    organism
        Target organism. Currently supports ``"human"`` and ``"mouse"``.

    Returns
    -------
    ResolutionReport
        One ``GuideRnaResolution`` per input sequence.
    """
    assembly = _ASSEMBLY_MAP.get(organism)
    species = _SPECIES_MAP.get(organism)
    if assembly is None or species is None:
        raise ValueError(
            f"Unsupported organism '{organism}'. Supported: {list(_ASSEMBLY_MAP.keys())}"
        )

    # Deduplicate, preserving original casing for input_value
    upper_to_original: dict[str, str] = {}
    for seq in sequences:
        upper = seq.upper()
        if upper not in upper_to_original:
            upper_to_original[upper] = seq
    unique_upper = list(upper_to_original.keys())

    # 1. Cache lookup
    cached_rows = _lookup_cached(unique_upper, species)
    logger.info(
        "Guide RNA cache: %d/%d sequences found",
        len(cached_rows),
        len(unique_upper),
    )

    # 2. Resolve uncached via BLAT + Ensembl
    resolution_map: dict[str, GuideRnaResolution] = {}
    new_records: list[dict] = []

    for upper_seq in unique_upper:
        original = upper_to_original[upper_seq]
        if upper_seq in cached_rows:
            resolution_map[upper_seq] = _record_to_resolution(cached_rows[upper_seq], original)
        else:
            res = _resolve_single(upper_seq, organism, assembly, species)
            # Preserve original casing in input_value
            res.input_value = original
            resolution_map[upper_seq] = res
            new_records.append(_resolution_to_record(res, species))

    # 3. Save new results to cache
    _save_to_cache(new_records)

    # 4. Map back to input order
    results: list[GuideRnaResolution] = [resolution_map[seq.upper()] for seq in sequences]

    resolved_count = sum(1 for r in results if r.resolved_value is not None)
    ambiguous_count = sum(1 for r in results if len(r.alternatives) > 0)

    return ResolutionReport(
        total=len(sequences),
        resolved=resolved_count,
        unresolved=len(sequences) - resolved_count,
        ambiguous=ambiguous_count,
        results=results,
    )
