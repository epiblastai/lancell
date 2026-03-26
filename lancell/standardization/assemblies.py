"""Genome assembly report resolution.

Downloads and parses NCBI assembly reports, mapping chromosome/contig names
to standardized metadata (GenBank accession, RefSeq accession, sequence role,
etc.) for populating ``ReferenceSequenceSchema``.

Supports lookup by any naming convention: UCSC (``chr1``), bare (``1``),
GenBank (``CM000663.2``), or RefSeq (``NC_000001.11``).

Assembly reports are cached locally at ``~/.cache/lancell/assembly_reports/``.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "lancell" / "assembly_reports"

# ---------------------------------------------------------------------------
# Known assemblies — maps (organism, assembly_name) to RefSeq FTP path
# ---------------------------------------------------------------------------

_ASSEMBLY_REPORT_URLS: dict[tuple[str, str], str] = {
    ("human", "GRCh38"): (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/"
        "GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_assembly_report.txt"
    ),
    ("mouse", "GRCm39"): (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/635/"
        "GCF_000001635.27_GRCm39/GCF_000001635.27_GRCm39_assembly_report.txt"
    ),
    ("mouse", "GRCm38"): (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/635/"
        "GCF_000001635.26_GRCm38.p6/GCF_000001635.26_GRCm38.p6_assembly_report.txt"
    ),
    ("human", "GRCh37"): (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/"
        "GCF_000001405.25_GRCh37.p13/GCF_000001405.25_GRCh37.p13_assembly_report.txt"
    ),
}

# Map NCBI Sequence-Role + Location/Type to SequenceRole enum values
_ROLE_MAP: dict[tuple[str, str | None], str] = {
    ("assembled-molecule", "Chromosome"): "chromosome",
    ("assembled-molecule", "Mitochondrion"): "mitochondrial",
    ("unlocalized-scaffold", None): "unlocalized",
    ("unplaced-scaffold", None): "scaffold",
    ("alt-scaffold", None): "alt_locus",
    ("fix-patch", None): "patch",
    ("novel-patch", None): "patch",
}


def _classify_role(sequence_role: str, location_type: str) -> str:
    """Map NCBI assembly report role + location to SequenceRole value."""
    # Try exact match first
    key = (sequence_role, location_type)
    if key in _ROLE_MAP:
        return _ROLE_MAP[key]
    # Fall back to role-only match
    key_no_loc = (sequence_role, None)
    if key_no_loc in _ROLE_MAP:
        return _ROLE_MAP[key_no_loc]
    return "other"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AssemblySequence:
    """A single sequence/contig from an NCBI assembly report."""

    sequence_name: str  # Column 1: bare name, e.g. "1", "MT"
    sequence_role: str  # Mapped to SequenceRole enum value
    assigned_molecule: str  # Column 3, e.g. "1", "X", "MT", "na"
    location_type: str  # Column 4, e.g. "Chromosome", "Mitochondrion"
    genbank_accession: str | None  # Column 5
    refseq_accession: str | None  # Column 7
    assembly_unit: str  # Column 8, e.g. "Primary Assembly"
    sequence_length: int  # Column 9
    ucsc_name: str | None  # Column 10
    is_primary_assembly: bool


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_assembly_report(text: str) -> list[AssemblySequence]:
    """Parse raw NCBI assembly report text into AssemblySequence records."""
    records: list[AssemblySequence] = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 10:
            logger.warning("Skipping malformed line with %d columns: %s", len(cols), line[:80])
            continue

        seq_name = cols[0]
        ncbi_role = cols[1]
        assigned_mol = cols[2]
        loc_type = cols[3]
        genbank = cols[4] if cols[4] != "na" else None
        refseq = cols[6] if cols[6] != "na" else None
        assembly_unit = cols[7]
        seq_length = int(cols[8])
        ucsc = cols[9] if cols[9] != "na" else None

        role = _classify_role(ncbi_role, loc_type)
        is_primary = assembly_unit == "Primary Assembly" or assembly_unit == "non-nuclear"

        records.append(
            AssemblySequence(
                sequence_name=seq_name,
                sequence_role=role,
                assigned_molecule=assigned_mol,
                location_type=loc_type,
                genbank_accession=genbank,
                refseq_accession=refseq,
                assembly_unit=assembly_unit,
                sequence_length=seq_length,
                ucsc_name=ucsc,
                is_primary_assembly=is_primary,
            )
        )
    return records


# ---------------------------------------------------------------------------
# Download / cache
# ---------------------------------------------------------------------------


def _cache_path(organism: str, assembly: str) -> Path:
    return _CACHE_DIR / f"{organism}_{assembly}_assembly_report.txt"


def _download_report(organism: str, assembly: str) -> str:
    """Download an assembly report, caching locally."""
    cached = _cache_path(organism, assembly)
    if cached.exists():
        logger.info("Using cached assembly report: %s", cached)
        return cached.read_text()

    key = (organism, assembly)
    if key not in _ASSEMBLY_REPORT_URLS:
        raise ValueError(
            f"Unknown assembly ({organism!r}, {assembly!r}). "
            f"Known assemblies: {list(_ASSEMBLY_REPORT_URLS.keys())}"
        )

    url = _ASSEMBLY_REPORT_URLS[key]
    logger.info("Downloading assembly report from %s", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(resp.text)
    return resp.text


# ---------------------------------------------------------------------------
# Lookup index
# ---------------------------------------------------------------------------


class AssemblyReport:
    """Parsed assembly report with multi-key lookup.

    Supports lookup by:
    - UCSC name (``chr1``, ``chrM``, ``chrUn_GL000220v1``)
    - Bare sequence name (``1``, ``MT``, ``HSCHR6_MHC_COX_CTG1``)
    - GenBank accession (``CM000663.2``)
    - RefSeq accession (``NC_000001.11``)
    """

    def __init__(self, sequences: list[AssemblySequence], organism: str, assembly: str):
        self.sequences = sequences
        self.organism = organism
        self.assembly = assembly

        # Build lookup indices
        self._by_ucsc: dict[str, AssemblySequence] = {}
        self._by_name: dict[str, AssemblySequence] = {}
        self._by_genbank: dict[str, AssemblySequence] = {}
        self._by_refseq: dict[str, AssemblySequence] = {}

        for seq in sequences:
            if seq.ucsc_name:
                self._by_ucsc[seq.ucsc_name] = seq
            self._by_name[seq.sequence_name] = seq
            if seq.genbank_accession:
                self._by_genbank[seq.genbank_accession] = seq
            if seq.refseq_accession:
                self._by_refseq[seq.refseq_accession] = seq

    def lookup(self, name: str) -> AssemblySequence | None:
        """Look up a sequence by any naming convention.

        Tries UCSC name first, then bare name, then GenBank, then RefSeq.
        Returns None if not found.
        """
        return (
            self._by_ucsc.get(name)
            or self._by_name.get(name)
            or self._by_genbank.get(name)
            or self._by_refseq.get(name)
        )

    def lookup_batch(self, names: list[str]) -> dict[str, AssemblySequence | None]:
        """Look up multiple sequence names. Returns {name: AssemblySequence | None}."""
        return {name: self.lookup(name) for name in names}

    @property
    def primary_sequences(self) -> list[AssemblySequence]:
        """Return only primary assembly sequences (chromosomes + MT)."""
        return [s for s in self.sequences if s.is_primary_assembly]

    @property
    def chromosomes(self) -> list[AssemblySequence]:
        """Return only assembled-molecule chromosomes (excl. MT, scaffolds, etc.)."""
        return [s for s in self.sequences if s.sequence_role == "chromosome"]


# ---------------------------------------------------------------------------
# In-memory cache of parsed reports
# ---------------------------------------------------------------------------

_report_cache: dict[tuple[str, str], AssemblyReport] = {}


def get_assembly_report(organism: str, assembly: str) -> AssemblyReport:
    """Get a parsed assembly report, downloading if needed.

    Parameters
    ----------
    organism
        Common name: ``"human"`` or ``"mouse"``.
    assembly
        Assembly name: ``"GRCh38"``, ``"GRCm39"``, ``"GRCm38"``.

    Returns
    -------
    AssemblyReport
        Parsed report with multi-key lookup.
    """
    key = (organism, assembly)
    if key not in _report_cache:
        text = _download_report(organism, assembly)
        sequences = _parse_assembly_report(text)
        _report_cache[key] = AssemblyReport(sequences, organism, assembly)
        logger.info(
            "Parsed %d sequences for %s %s (%d primary)",
            len(sequences),
            organism,
            assembly,
            len(_report_cache[key].primary_sequences),
        )
    return _report_cache[key]


# ---------------------------------------------------------------------------
# Convenience: resolve chromosome names to schema-ready dicts
# ---------------------------------------------------------------------------


def resolve_sequence_names(
    names: list[str],
    organism: str = "human",
    assembly: str = "GRCh38",
) -> list[dict]:
    """Resolve a list of sequence names to ReferenceSequenceSchema-compatible dicts.

    Unrecognized names are included with ``sequence_role="other"`` and null
    accessions, so the caller always gets one output per input.

    Parameters
    ----------
    names
        Sequence names in any convention (UCSC, bare, GenBank, RefSeq).
    organism
        Common name, e.g. ``"human"``, ``"mouse"``.
    assembly
        Assembly name, e.g. ``"GRCh38"``, ``"GRCm39"``.

    Returns
    -------
    list[dict]
        One dict per unique input name with keys matching
        ``ReferenceSequenceSchema`` fields.
    """
    report = get_assembly_report(organism, assembly)
    seen: set[str] = set()
    results: list[dict] = []

    for name in names:
        if name in seen:
            continue
        seen.add(name)

        seq = report.lookup(name)
        if seq is not None:
            results.append(
                {
                    "sequence_name": seq.ucsc_name or name,
                    "sequence_role": seq.sequence_role,
                    "organism": organism,
                    "assembly": assembly,
                    "genbank_accession": seq.genbank_accession,
                    "refseq_accession": seq.refseq_accession,
                    "is_primary_assembly": seq.is_primary_assembly,
                }
            )
        else:
            logger.warning(
                "Sequence %r not found in %s %s assembly report", name, organism, assembly
            )
            results.append(
                {
                    "sequence_name": name,
                    "sequence_role": "other",
                    "organism": organism,
                    "assembly": assembly,
                    "genbank_accession": None,
                    "refseq_accession": None,
                    "is_primary_assembly": False,
                }
            )

    return results
