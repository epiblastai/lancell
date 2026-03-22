"""Guide RNA resolution via UCSC BLAT and Ensembl overlap annotation.

Resolves raw guide RNA sequences (typically 20bp) to genomic coordinates and
overlapping gene annotations, populating fields needed by GeneticPerturbationSchema:
target_start, target_end, target_strand, intended_gene_name, intended_ensembl_gene_id,
and target_context.

Two public entry points:
  - ``resolve_guide_sequences`` — full pipeline: BLAT alignment → Ensembl overlap
  - ``annotate_genomic_coordinates`` — Ensembl overlap only (for pre-computed coords)
"""

import logging
import re

import pandas as pd
import requests

from lancell.standardization._rate_limit import rate_limited
from lancell.standardization.types import GuideRnaResolution, ResolutionReport

logger = logging.getLogger(__name__)

_ENSEMBL_REST_BASE = "https://rest.ensembl.org"
_PRIMARY_CHROM_RE = re.compile(r"^chr(\d{1,2}|[XY])$")
_PROMOTER_UPSTREAM_BP = 2000

_ASSEMBLY_MAP: dict[str, str] = {
    "human": "hg38",
    "mouse": "mm39",
}
_SPECIES_MAP: dict[str, str] = {
    "human": "homo_sapiens",
    "mouse": "mus_musculus",
}


# ---------------------------------------------------------------------------
# BLAT helpers
# ---------------------------------------------------------------------------


@rate_limited("ucsc_blat", max_per_second=1)
def _blat_guide(sequence: str, assembly: str) -> pd.DataFrame | None:
    """Run gget.blat for a single guide sequence. Returns None on failure."""
    try:
        import gget
    except ImportError:
        raise ImportError(
            "gget is required for guide RNA resolution. Install it with: pip install lancell[bio]"
        ) from None

    result = gget.blat(sequence, assembly=assembly)
    if result is None or (isinstance(result, pd.DataFrame) and result.empty):
        return None
    return result


def _pick_best_blat_hit(blat_df: pd.DataFrame) -> dict | None:
    """Filter BLAT results to primary chromosomes and return the best hit."""
    mask = blat_df["chromosome"].str.match(_PRIMARY_CHROM_RE)
    primary = blat_df[mask]
    if primary.empty:
        return None

    # Sort by match quality — '%_matched' is the key quality metric
    sorted_df = primary.sort_values("%_matched", ascending=False)
    best = sorted_df.iloc[0]
    return {
        "chromosome": best["chromosome"],
        "start": int(best["start"]),
        "end": int(best["end"]),
        "strand": best.get("strand", "+"),
        "pct_matched": float(best["%_matched"]),
    }


# ---------------------------------------------------------------------------
# Ensembl overlap helpers
# ---------------------------------------------------------------------------


@rate_limited("ensembl")
def _ensembl_overlap_genes(species: str, chromosome: str, start: int, end: int) -> list[dict]:
    """Query Ensembl REST for protein-coding genes overlapping a region."""
    # Ensembl uses bare chromosome names (no 'chr' prefix)
    chrom_bare = chromosome.removeprefix("chr")
    url = f"{_ENSEMBL_REST_BASE}/overlap/region/{species}/{chrom_bare}:{start}-{end}"
    resp = requests.get(
        url,
        headers={"Content-Type": "application/json"},
        params={"feature": "gene"},
        timeout=30,
    )
    resp.raise_for_status()

    genes = resp.json()
    return [g for g in genes if g.get("biotype") == "protein_coding"]


@rate_limited("ensembl")
def _ensembl_overlap_features(species: str, chromosome: str, start: int, end: int) -> list[dict]:
    """Query Ensembl REST for exon/CDS/transcript features overlapping a region."""
    chrom_bare = chromosome.removeprefix("chr")
    url = f"{_ENSEMBL_REST_BASE}/overlap/region/{species}/{chrom_bare}:{start}-{end}"
    resp = requests.get(
        url,
        headers={"Content-Type": "application/json"},
        params={"feature": ["exon", "cds", "transcript"]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Target context classification
# ---------------------------------------------------------------------------


def _classify_target_context(
    guide_start: int,
    guide_end: int,
    genes: list[dict],
    features: list[dict],
) -> str:
    """Classify where a guide lands relative to gene structure.

    Returns a TargetContext-compatible string: "exon", "intron", "promoter",
    "5_UTR", "3_UTR", "intergenic", or "other".
    """
    guide_mid = (guide_start + guide_end) // 2

    if not genes:
        # Check if we're in a promoter region of any nearby gene
        # (would need a wider query — for now, return intergenic)
        return "intergenic"

    # Separate features by type
    cds_features = [f for f in features if f.get("feature_type") == "cds"]
    exon_features = [f for f in features if f.get("feature_type") == "exon"]
    transcript_features = [f for f in features if f.get("feature_type") == "transcript"]

    # Check CDS overlap → exon (coding)
    for cds in cds_features:
        if cds["start"] <= guide_mid <= cds["end"]:
            return "exon"

    # Check exon overlap without CDS → UTR
    for exon in exon_features:
        if exon["start"] <= guide_mid <= exon["end"]:
            # Determine 5' vs 3' UTR by comparing to CDS boundaries
            # Find CDS features from the same transcript
            parent_id = exon.get("Parent")
            transcript_cds = [c for c in cds_features if c.get("Parent") == parent_id]
            if transcript_cds:
                cds_starts = [c["start"] for c in transcript_cds]
                cds_min = min(cds_starts)
                strand = exon.get("strand", 1)
                if strand == 1 or strand == "+":
                    return "5_UTR" if guide_mid < cds_min else "3_UTR"
                else:
                    return "3_UTR" if guide_mid < cds_min else "5_UTR"
            # Exon but no CDS in this transcript — could be non-coding transcript
            # overlapping a coding gene. Still call it "other" for UTR ambiguity.
            return "other"

    # Check transcript overlap without exon → intron
    for tx in transcript_features:
        if tx["start"] <= guide_mid <= tx["end"]:
            return "intron"

    # Guide overlaps a gene region but not its transcripts — check promoter
    best_gene = _pick_closest_gene(guide_mid, genes)
    if best_gene:
        strand = best_gene.get("strand", 1)
        tss = best_gene["start"] if strand in (1, "+") else best_gene["end"]
        if strand in (1, "+"):
            if tss - _PROMOTER_UPSTREAM_BP <= guide_mid < tss:
                return "promoter"
        else:
            if tss < guide_mid <= tss + _PROMOTER_UPSTREAM_BP:
                return "promoter"

    return "other"


def _pick_closest_gene(guide_mid: int, genes: list[dict]) -> dict | None:
    """Pick the protein-coding gene whose TSS is closest to the guide midpoint."""
    if not genes:
        return None

    def tss_distance(gene: dict) -> int:
        strand = gene.get("strand", 1)
        tss = gene["start"] if strand in (1, "+") else gene["end"]
        return abs(guide_mid - tss)

    return min(genes, key=tss_distance)


# ---------------------------------------------------------------------------
# Single-item resolvers
# ---------------------------------------------------------------------------


def _resolve_single(
    sequence: str,
    organism: str,
    assembly: str,
    species: str,
) -> GuideRnaResolution:
    """Full BLAT → Ensembl overlap pipeline for a single guide sequence."""
    # Step 1: BLAT
    blat_df = _blat_guide(sequence, assembly)
    if blat_df is None:
        return GuideRnaResolution(
            input_value=sequence,
            resolved_value=None,
            confidence=0.0,
            source="blat",
        )

    hit = _pick_best_blat_hit(blat_df)
    if hit is None:
        return GuideRnaResolution(
            input_value=sequence,
            resolved_value=None,
            confidence=0.0,
            source="blat",
        )

    return _annotate_from_coordinates(
        chromosome=hit["chromosome"],
        start=hit["start"],
        end=hit["end"],
        strand=hit.get("strand"),
        species=species,
        organism=organism,
        input_value=sequence,
        source="blat+ensembl",
        assembly=assembly,
        blat_pct_match=hit["pct_matched"],
    )


def _annotate_from_coordinates(
    chromosome: str,
    start: int,
    end: int,
    strand: str | None,
    species: str,
    organism: str,
    input_value: str,
    source: str,
    assembly: str | None = None,
    blat_pct_match: float | None = None,
) -> GuideRnaResolution:
    """Ensembl overlap annotation from pre-resolved coordinates."""
    # Query Ensembl for overlapping genes and features
    try:
        genes = _ensembl_overlap_genes(species, chromosome, start, end)
    except requests.HTTPError as exc:
        logger.warning("Ensembl gene overlap failed for %s:%d-%d: %s", chromosome, start, end, exc)
        genes = []

    try:
        features = _ensembl_overlap_features(species, chromosome, start, end)
    except requests.HTTPError as exc:
        logger.warning(
            "Ensembl feature overlap failed for %s:%d-%d: %s", chromosome, start, end, exc
        )
        features = []

    # Classify target context
    target_context = _classify_target_context(start, end, genes, features)

    # Pick the best gene
    guide_mid = (start + end) // 2
    best_gene = _pick_closest_gene(guide_mid, genes)

    gene_name = best_gene.get("external_name") if best_gene else None
    gene_id = best_gene.get("gene_id") if best_gene else None

    # Determine confidence
    if best_gene and len(genes) == 1:
        confidence = 1.0
    elif best_gene and len(genes) > 1:
        confidence = 0.9
    else:
        confidence = 0.5  # coordinates resolved but no protein-coding gene

    # resolved_value: gene name or locus string
    resolved_value = gene_name if gene_name else f"{chromosome}:{start}-{end}"

    return GuideRnaResolution(
        input_value=input_value,
        resolved_value=resolved_value,
        confidence=confidence,
        source=source,
        alternatives=[
            g.get("external_name", g.get("gene_id", ""))
            for g in genes
            if g is not best_gene and g.get("external_name")
        ],
        chromosome=chromosome,
        target_start=start,
        target_end=end,
        target_strand=strand,
        intended_gene_name=gene_name,
        intended_ensembl_gene_id=gene_id,
        target_context=target_context,
        assembly=assembly,
        blat_pct_match=blat_pct_match,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_guide_sequences(
    sequences: list[str],
    organism: str = "human",
) -> ResolutionReport:
    """Resolve guide RNA sequences to genomic coordinates and gene annotations.

    Uses UCSC BLAT to align each sequence to the reference genome, then queries
    the Ensembl REST API for overlapping protein-coding genes and classifies the
    target context (exon, intron, promoter, UTR, intergenic).

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

    # Deduplicate sequences — guides are shared across many cells
    unique_sequences = list(dict.fromkeys(sequences))
    cache: dict[str, GuideRnaResolution] = {}

    for seq in unique_sequences:
        cache[seq] = _resolve_single(seq, organism, assembly, species)

    # Map results back to full input list
    results: list[GuideRnaResolution] = [cache[seq] for seq in sequences]

    resolved_count = sum(1 for r in results if r.resolved_value is not None)
    ambiguous_count = sum(1 for r in results if len(r.alternatives) > 0)

    return ResolutionReport(
        total=len(sequences),
        resolved=resolved_count,
        unresolved=len(sequences) - resolved_count,
        ambiguous=ambiguous_count,
        results=results,
    )


def annotate_genomic_coordinates(
    coordinates: list[dict],
    organism: str = "human",
) -> ResolutionReport:
    """Annotate pre-resolved genomic coordinates with gene and context information.

    Skips BLAT alignment and goes directly to Ensembl overlap queries. Useful when
    coordinates are already known from library files or other alignment tools.

    Parameters
    ----------
    coordinates
        List of dicts, each with keys: ``chromosome`` (str, e.g. "chr17"),
        ``start`` (int), ``end`` (int), and optionally ``strand`` (str),
        ``guide_sequence`` (str).
    organism
        Target organism. Currently supports ``"human"`` and ``"mouse"``.

    Returns
    -------
    ResolutionReport
        One ``GuideRnaResolution`` per input coordinate.
    """
    species = _SPECIES_MAP.get(organism)
    assembly = _ASSEMBLY_MAP.get(organism)
    if species is None:
        raise ValueError(
            f"Unsupported organism '{organism}'. Supported: {list(_SPECIES_MAP.keys())}"
        )

    results: list[GuideRnaResolution] = []
    for coord in coordinates:
        chromosome = coord["chromosome"]
        start = coord["start"]
        end = coord["end"]
        strand = coord.get("strand")
        guide_seq = coord.get("guide_sequence", f"{chromosome}:{start}-{end}")

        result = _annotate_from_coordinates(
            chromosome=chromosome,
            start=start,
            end=end,
            strand=strand,
            species=species,
            organism=organism,
            input_value=guide_seq,
            source="ensembl",
            assembly=assembly,
        )
        results.append(result)

    resolved_count = sum(1 for r in results if r.resolved_value is not None)
    ambiguous_count = sum(1 for r in results if len(r.alternatives) > 0)

    return ResolutionReport(
        total=len(coordinates),
        resolved=resolved_count,
        unresolved=len(coordinates) - resolved_count,
        ambiguous=ambiguous_count,
        results=results,
    )
