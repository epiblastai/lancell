"""LanceModel schemas and DB helpers for self-hosted reference databases.

Nine tables: organisms, genomic features, genomic feature aliases, ontology terms,
compounds, compound synonyms, proteins, protein aliases, and guide RNAs.
Stored in a single LanceDB at ``~/.cache/lancell/reference_db/``.
"""

from collections.abc import Iterator
from pathlib import Path

import lancedb
from lancedb.pydantic import LanceModel

# Table name constants
ORGANISMS_TABLE = "organisms"
GENOMIC_FEATURES_TABLE = "genomic_features"
GENOMIC_FEATURE_ALIASES_TABLE = "genomic_feature_aliases"
ONTOLOGY_TERMS_TABLE = "ontology_terms"
COMPOUNDS_TABLE = "compounds"
COMPOUND_SYNONYMS_TABLE = "compound_synonyms"
PROTEINS_TABLE = "proteins"
PROTEIN_ALIASES_TABLE = "protein_aliases"
GUIDE_RNAS_TABLE = "guide_rnas"

DEFAULT_REFERENCE_DB_PATH = Path.home() / ".cache" / "lancell" / "reference_db"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class OrganismRecord(LanceModel):
    """One row per supported organism. Replaces hardcoded dicts in genes.py.

    Parameters
    ----------
    common_name:
        Human-readable name, e.g. ``"human"``, ``"mouse"``.
    scientific_name:
        Binomial name in lowercase, e.g. ``"homo_sapiens"``. Used as
        the foreign key by other tables because scientific names are
        guaranteed globally unique by taxonomic convention.
    ncbi_taxonomy_id:
        NCBI Taxonomy ID, e.g. ``9606`` for human.
    ensembl_prefix:
        Ensembl gene ID prefix, e.g. ``"ENSG"`` for human.
        ``None`` until genomic features have been downloaded for this
        organism (the prefix is detected from actual gene IDs).
    ensembl_species_name:
        Species name used in Ensembl BioMart dataset names,
        e.g. ``"homo_sapiens"``.
    """

    common_name: str
    scientific_name: str
    ncbi_taxonomy_id: int
    ensembl_prefix: str | None = None
    ensembl_species_name: str


class GenomicFeatureRecord(LanceModel):
    """One row per Ensembl feature (genes, lncRNAs, miRNAs, pseudogenes, etc.).

    Parameters
    ----------
    ensembl_gene_id:
        Primary key, e.g. ``"ENSG00000141510"`` for TP53.
    symbol:
        Canonical symbol, e.g. ``"TP53"``, ``"HOTAIR"``.
    ncbi_gene_id:
        Entrez/NCBI gene ID, if available.
    biotype:
        Ensembl biotype, e.g. ``"protein_coding"``, ``"lncRNA"``, ``"miRNA"``,
        ``"pseudogene"``.
    chromosome:
        Chromosome or scaffold name, if available.
    organism:
        FK to ``OrganismRecord.scientific_name``,
        e.g. ``"homo_sapiens"``.
    assembly:
        Genome assembly, e.g. ``"GRCh38"``, ``"GRCh37"``.
        ``None`` for species where assembly is not tracked.
    """

    ensembl_gene_id: str
    symbol: str
    ncbi_gene_id: int | None
    biotype: str
    chromosome: str | None
    organism: str
    assembly: str | None = None


class GenomicFeatureAliasRecord(LanceModel):
    """Flattened alias table for fast exact-match lookup.

    The ``alias`` column is lowercased at ingestion time so that lookups can
    use a scalar index with ``WHERE alias = lower(input) AND organism = ?``.
    A scalar index is preferred over FTS here because gene symbols contain
    punctuation and digits (e.g. ``"il-6"``, ``"tp53"``) that FTS tokenizers
    would split or mangle.

    Parameters
    ----------
    alias:
        Lowercased alias string for case-insensitive exact match.
    alias_original:
        Original casing of the alias, e.g. ``"TP53"``, ``"IL-6"``.
    ensembl_gene_id:
        FK to ``GenomicFeatureRecord.ensembl_gene_id``.
    organism:
        FK to ``OrganismRecord.scientific_name``,
        e.g. ``"homo_sapiens"``.
    is_canonical:
        ``True`` if this alias is the HGNC/MGI canonical symbol.
    source:
        Which authority provided this alias. ``"gencode"`` for names
        from GENCODE GTFs (human/mouse), ``"biomart"`` for Ensembl
        BioMart synonyms and names.
    assembly:
        Genome assembly this alias was sourced from,
        e.g. ``"GRCh38"``, ``"GRCh37"``. ``None`` if not tracked.
    """

    alias: str
    alias_original: str
    ensembl_gene_id: str
    organism: str
    is_canonical: bool
    source: str = "biomart"
    assembly: str | None = None


class OntologyTermRecord(LanceModel):
    """Unified table for all ontologies (CL, UBERON, MONDO, EFO, etc.).

    Parameters
    ----------
    ontology_term_id:
        CURIE primary key, e.g. ``"CL:0000540"``.
    ontology_prefix:
        Ontology namespace prefix, e.g. ``"CL"``, ``"UBERON"``.
    name:
        Human-readable term name, e.g. ``"neuron"``.
    definition:
        Term definition text from the ontology, if available.
    synonyms:
        Pipe-delimited synonym text for FTS indexing,
        e.g. ``"nerve cell | neuronal cell | neurone"``.
    parent_ids:
        ``is_a`` parent term IDs for hierarchy traversal.
    is_obsolete:
        Whether this term is marked obsolete in the ontology.
    """

    ontology_term_id: str
    ontology_prefix: str
    name: str
    definition: str | None
    synonyms: str | None
    parent_ids: list[str]
    is_obsolete: bool


class CompoundRecord(LanceModel):
    """One row per PubChem compound.

    Parameters
    ----------
    pubchem_cid:
        PubChem Compound ID (primary key).
    name:
        Preferred compound name from CID-Title.
    canonical_smiles:
        Canonical SMILES from CID-SMILES, if available.
    """

    pubchem_cid: int
    name: str
    canonical_smiles: str | None = None


class CompoundSynonymRecord(LanceModel):
    """Flattened synonym table for fast name → CID lookup.

    The ``synonym`` column is lowercased at ingestion time so that lookups
    can use a scalar index with ``WHERE synonym IN (...)``.

    Parameters
    ----------
    synonym:
        Lowercased synonym string for case-insensitive exact match.
    synonym_original:
        Original casing of the synonym.
    pubchem_cid:
        FK to ``CompoundRecord.pubchem_cid``.
    is_title:
        ``True`` if this synonym is the preferred title from CID-Title.
    """

    synonym: str
    synonym_original: str
    pubchem_cid: int
    is_title: bool


class ProteinRecord(LanceModel):
    """One row per primary UniProt accession.

    Parameters
    ----------
    uniprot_id:
        Primary accession, e.g. ``"P04637"``.
    protein_name:
        RecName Full, e.g. ``"Cellular tumor antigen p53"``.
    gene_name:
        Primary GN Name, e.g. ``"TP53"``. ``None`` for viral ORFs etc.
    organism:
        Normalized scientific name, e.g. ``"homo_sapiens"``.
    ncbi_taxonomy_id:
        From OX line, e.g. ``9606`` for human.
    """

    uniprot_id: str
    protein_name: str
    gene_name: str | None = None
    organism: str
    ncbi_taxonomy_id: int


class ProteinAliasRecord(LanceModel):
    """Flattened alias table for fast exact-match protein lookup.

    The ``alias`` column is lowercased at ingestion time so that lookups
    can use a scalar index with ``WHERE alias IN (...) AND organism = ?``.

    Parameters
    ----------
    alias:
        Lowercased alias string for case-insensitive exact match.
    alias_original:
        Original casing of the alias.
    uniprot_id:
        FK to ``ProteinRecord.uniprot_id``.
    organism:
        Same organism format as ``ProteinRecord``.
    is_canonical:
        ``True`` for RecName Full and primary GN Name.
    source:
        Origin of the alias: ``"rec_name"``, ``"alt_name"``,
        ``"alt_name_short"``, ``"gene_name"``, ``"gene_synonym"``,
        ``"orf_name"``, or ``"secondary_accession"``.
    """

    alias: str
    alias_original: str
    uniprot_id: str
    organism: str
    is_canonical: bool
    source: str


class GuideRnaRecord(LanceModel):
    """Cached guide RNA resolution result.

    One row per unique (guide_sequence, organism) pair. Populated
    lazily as guide sequences are resolved via BLAT + Ensembl.

    Parameters
    ----------
    guide_sequence:
        Uppercase DNA sequence (typically 20bp). Lookup key.
    organism:
        Scientific name FK (e.g., ``"homo_sapiens"``). Lookup key.
    chromosome:
        BLAT-aligned chromosome, e.g. ``"chr17"``.
    target_start:
        Genomic start coordinate.
    target_end:
        Genomic end coordinate.
    target_strand:
        ``"+"`` or ``"-"``.
    intended_gene_name:
        Symbol of the closest protein-coding gene.
    intended_ensembl_gene_id:
        Ensembl gene ID of the intended gene.
    target_context:
        Where the guide lands relative to gene structure.
    assembly:
        Genome assembly, e.g. ``"hg38"``, ``"mm39"``.
    blat_pct_match:
        BLAT alignment quality percentage (0–100).
    confidence:
        Resolution confidence (1.0=single gene, 0.9=multiple,
        0.5=no gene, 0.0=failed).
    resolved_value:
        Gene name or locus string, ``None`` if unresolved.
    alternatives:
        Pipe-delimited alternative overlapping gene names.
    """

    guide_sequence: str
    organism: str
    chromosome: str | None = None
    target_start: int | None = None
    target_end: int | None = None
    target_strand: str | None = None
    intended_gene_name: str | None = None
    intended_ensembl_gene_id: str | None = None
    target_context: str | None = None
    assembly: str | None = None
    blat_pct_match: float | None = None
    confidence: float = 0.0
    resolved_value: str | None = None
    alternatives: str | None = None


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _is_remote_path(path: str | Path) -> bool:
    """Check if a path is a remote URI (S3, GCS, Azure)."""
    return str(path).startswith(("s3://", "gs://", "az://"))


def open_reference_db(db_path: str | Path | None = None) -> lancedb.DBConnection:
    """Open (or create) the reference LanceDB."""
    if db_path is None:
        db_path = DEFAULT_REFERENCE_DB_PATH
    if not _is_remote_path(db_path):
        db_path = Path(db_path)
        db_path.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(db_path))


def ensure_table(
    db: lancedb.DBConnection,
    table_name: str,
    schema: type[LanceModel],
    data: list[dict],
    mode: str = "overwrite",
) -> lancedb.table.Table:
    """Create or overwrite a table with the given data."""
    return db.create_table(table_name, data=data, schema=schema, mode=mode)


def ensure_table_chunked(
    db: lancedb.DBConnection,
    table_name: str,
    schema: type[LanceModel],
    chunks: Iterator[list[dict]],
) -> lancedb.table.Table:
    """Create a table from the first chunk, then append subsequent chunks.

    Needed for tables too large to materialize as a single ``list[dict]``
    (e.g. 116M+ compound rows).
    """
    table: lancedb.table.Table | None = None
    for chunk in chunks:
        if not chunk:
            continue
        if table is None:
            table = db.create_table(table_name, data=chunk, schema=schema, mode="overwrite")
        else:
            table.add(chunk)
    if table is None:
        raise ValueError(f"No data provided for table '{table_name}'")
    return table


def reference_db_exists(db_path: str | Path | None = None) -> bool:
    """Check if the reference DB is populated (has at least the organisms table)."""
    if db_path is None:
        db_path = _custom_db_path or DEFAULT_REFERENCE_DB_PATH
    if _is_remote_path(db_path):
        db = lancedb.connect(str(db_path))
        return ORGANISMS_TABLE in db.table_names()
    db_path = Path(db_path)
    if not db_path.exists():
        return False
    db = lancedb.connect(str(db_path))
    return ORGANISMS_TABLE in db.table_names()


# ---------------------------------------------------------------------------
# Centralized DB connection (lazy singleton with configurable path)
# ---------------------------------------------------------------------------

_custom_db_path: str | Path | None = None
_shared_db_connection: lancedb.DBConnection | None = None


def set_reference_db_path(db_path: str | Path) -> None:
    """Set a custom path for the reference DB (local or remote).

    Call this before any resolution functions to point at a non-default
    location (e.g. ``"s3://bucket/ontology_resolver/"``). Resets the
    cached connection so the next call to ``get_reference_db()`` connects
    to the new path.
    """
    global _custom_db_path, _shared_db_connection
    _custom_db_path = db_path
    _shared_db_connection = None


def get_reference_db() -> lancedb.DBConnection:
    """Return a cached LanceDB connection to the reference DB.

    Uses the path set by ``set_reference_db_path()`` if called, otherwise
    falls back to ``DEFAULT_REFERENCE_DB_PATH``.

    Raises
    ------
    RuntimeError
        If the reference DB does not exist at the configured path.
    """
    global _shared_db_connection
    if _shared_db_connection is not None:
        return _shared_db_connection
    db_path = _custom_db_path or DEFAULT_REFERENCE_DB_PATH
    if not _is_remote_path(db_path) and not Path(db_path).exists():
        raise RuntimeError(
            f"Reference database not found at {db_path}. "
            "Run `python scripts/download_references.py` to populate it, "
            "or call `set_reference_db_path()` to point at a remote DB."
        )
    _shared_db_connection = open_reference_db(db_path)
    return _shared_db_connection
