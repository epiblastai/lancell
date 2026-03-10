"""Library utilities imported by GEO ingestion scripts.

Generated ingestion scripts import from this module for gene/molecule resolution,
sparse count extraction, and metadata construction.
"""

import json
import xml.etree.ElementTree as ET
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

import lancedb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from Bio import Entrez

from lancell.schema import (
    CellIndex,
    ChromatinAccessibilityRecord,
    ChromosomeSchema,
    GeneExpressionRecord,
    GeneSchema,
    ImageFeatureVectorRecord,
    ImageTileRecord,
    ImageFeatureSchema,
    MoleculeSchema,
    ProteinAbundanceRecord,
    ProteinSchema,
    PublicationSchema,
)

Entrez.email = "ryan@epiblast.ai"
CellDataRecord = (
    GeneExpressionRecord
    | ProteinAbundanceRecord
    | ChromatinAccessibilityRecord
    | ImageFeatureVectorRecord
    | ImageTileRecord
)


def _escape_lance_value(value: str) -> str:
    """Escape single quotes in a string value for LanceDB WHERE clauses."""
    return value.replace("'", "''")


class OntologyEntity(Enum):
    GENE: str = "Gene"
    PROTEIN: str = "Protein"
    ORGANISM: str = "Organism"
    CELL_LINE: str = "CellLine"
    CELL_TYPE: str = "CellType"
    TISSUE: str = "Tissue"
    DISEASE: str = "Disease"
    DEVELOPMENT_STAGE: str = "DevelopmentalStage"


# ---------------------------------------------------------------------------
# Publication metadata utilities
# ---------------------------------------------------------------------------


def _parse_publication_date(value: str | None) -> datetime | None:
    if not value:
        return None

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%b-%d", "%Y-%m", "%Y/%m", "%Y-%b", "%Y"):
        try:
            dt = datetime.strptime(value, fmt)
            if fmt in ("%Y-%m", "%Y/%m", "%Y-%b"):
                return dt.replace(day=1)
            if fmt == "%Y":
                return dt.replace(month=1, day=1)
            return dt
        except ValueError:
            continue

    raise ValueError(f"Could not parse publication date: {value!r}")


def fetch_publication_metadata(pmid: str) -> dict:
    """Fetch publication metadata from PubMed and optionally full text from PMC.

    Returns a dict with keys: pmid, doi, title, journal, publication_date, full_text.
    """
    with Entrez.efetch(db="pubmed", id=pmid, rettype="xml") as handle:
        xml_data = handle.read()
    root = ET.fromstring(xml_data)

    article = root.find(".//MedlineCitation/Article")
    assert article is not None, f"No article found for PMID {pmid}"

    title = article.findtext("ArticleTitle", default="")
    journal = article.findtext("Journal/Title", default="")
    abstract_parts = article.findall(".//Abstract/AbstractText")
    abstract = " ".join(part.text or "" for part in abstract_parts) if abstract_parts else ""

    # DOI
    doi_elem = root.find('.//ArticleId[@IdType="doi"]')
    doi = doi_elem.text if doi_elem is not None else ""

    # Publication date — try MedlineDate, then Year/Month/Day from Article
    pub_date_elem = article.find(".//Journal/JournalIssue/PubDate")
    pub_date = None
    if pub_date_elem is not None:
        medline_date = pub_date_elem.findtext("MedlineDate")
        if medline_date:
            # MedlineDate is freeform like "2020 Jan-Feb", take the year
            pub_date = _parse_publication_date(medline_date.split()[0])
        else:
            year = pub_date_elem.findtext("Year", default="")
            month = pub_date_elem.findtext("Month", default="")
            day = pub_date_elem.findtext("Day", default="")
            date_str = "-".join(part for part in [year, month, day] if part)
            pub_date = _parse_publication_date(date_str)

    # PMC ID
    pmc_elem = root.find('.//ArticleId[@IdType="pmc"]')
    pmc_id = pmc_elem.text if pmc_elem is not None else None

    # Try to get full text from PMC
    full_text = abstract
    if pmc_id:
        try:
            with Entrez.efetch(db="pmc", id=pmc_id, rettype="xml") as handle:
                pmc_xml = handle.read()
            pmc_root = ET.fromstring(pmc_xml)
            paragraphs = []
            for p in pmc_root.iter("p"):
                text = "".join(p.itertext()).strip()
                if text:
                    paragraphs.append(text)
            if paragraphs:
                full_text = "\n\n".join(paragraphs)
        except Exception:
            pass  # Fall back to abstract

    return {
        "pmid": pmid,
        "doi": doi,
        "title": title,
        "journal": journal,
        "publication_date": pub_date.isoformat() if pub_date else None,
        "full_text": full_text,
    }


def search_pubmed_by_title(title: str) -> str | None:
    """Search PubMed for a paper by title. Returns the first matching PMID or None."""
    with Entrez.esearch(db="pubmed", term=f'"{title}"[Title]') as handle:
        results = Entrez.read(handle)
    id_list = results.get("IdList", [])
    return id_list[0] if id_list else None


def extract_nonzero_counts(matrix_row) -> tuple[np.ndarray, np.ndarray]:
    """Extract nonzero gene indices and values from a single cell's count vector.

    Returns (gene_indices, gene_values) as np.int32 and np.float32.
    """
    if sp.issparse(matrix_row):
        row_csr = sp.csr_matrix(matrix_row)
        indices = row_csr.indices.astype(np.int32)
        values = row_csr.data.astype(np.float32)
    else:
        arr = np.asarray(matrix_row).ravel()
        nonzero_mask = arr > 0
        indices = np.where(nonzero_mask)[0].astype(np.int32)
        values = arr[nonzero_mask].astype(np.float32)
    return indices, values


def remap_and_sort_indices(
    local_indices: np.ndarray,
    local_values: np.ndarray,
    positional_to_global: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remap positional feature indices to global indices and sort both arrays.

    Takes local (positional) indices from a single cell's sparse vector, maps them
    to global feature indices via ``positional_to_global``, and sorts both the
    indices and values arrays by the global index order.

    Returns (global_indices, sorted_values) as np.int32 and np.float32.
    """
    global_indices = positional_to_global[local_indices]
    sort_order = np.argsort(global_indices)
    return global_indices[sort_order], local_values[sort_order]


def build_additional_metadata(row, columns: list[str]) -> str | None:
    """Build additional_metadata JSON string from selected obs columns.

    Skips NaN values so different cells may have different keys.
    """
    metadata = {}
    for col in columns:
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            metadata[col] = str(val)
    if not metadata:
        return None
    return json.dumps(metadata)


def upsert_table(
    db: lancedb.DBConnection,
    table_name: str,
    data,
    *,
    schema=None,
) -> lancedb.table.Table:
    """Add data to an existing table or create it.

    If the table already exists, opens it and adds ``data``.  Otherwise creates
    a new table.  When ``data`` is falsy (empty list / None) and ``schema`` is
    provided, creates an empty table from the schema.

    Returns the opened or newly created table.
    """
    if table_name in db.list_tables().tables:
        table = db.open_table(table_name)
        if data:
            table.add(data)
        return table

    if data:
        return db.create_table(table_name, data=data)

    if schema is not None:
        return db.create_table(table_name, schema=schema)

    raise ValueError(f"Cannot create table '{table_name}': no data and no schema provided")


def load_publication(data_dir: Path) -> PublicationSchema | None:
    """Load ``publication.json`` from *data_dir* and return a PublicationSchema.

    Returns ``None`` when the file does not exist.
    """
    path = data_dir / "publication.json"
    if not path.exists():
        return None
    with open(path) as f:
        pub_data = json.load(f)
    pub_date_str = pub_data.get("publication_date")
    return PublicationSchema(
        pmid=pub_data["pmid"],
        doi=pub_data["doi"],
        title=pub_data["title"],
        journal=pub_data["journal"],
        publication_date=datetime.fromisoformat(pub_date_str) if pub_date_str else None,
        full_text=pub_data["full_text"],
    )


def upsert_publication(db: lancedb.DBConnection, data_dir: Path) -> None:
    """Load publication.json and write it to the publications table if not already present.

    Skips silently when publication.json doesn't exist or the PMID is already in the table.
    """
    publication = load_publication(data_dir)
    if publication is None:
        return

    if "publications" in db.list_tables().tables:
        existing = db.open_table("publications").search().select(["pmid"]).to_pandas()
        if publication.pmid in existing["pmid"].values:
            print(f"Publication PMID {publication.pmid} already in table, skipping.")
            return

    print("Writing publication record...")
    upsert_table(db, "publications", [publication])


# ---------------------------------------------------------------------------
# Lookup utilities for linking to existing genes and molecules tables
# ---------------------------------------------------------------------------


def get_max_gene_index(genes_table: lancedb.Table) -> int:
    """Get current max gene_index. Returns -1 if empty (so start_index = 0)."""
    df = genes_table.search().select(["gene_index"]).to_pandas()
    if df.empty:
        return -1
    return int(df["gene_index"].max())


def lookup_gene_indices_from_table(
    genes_table: lancedb.Table,
    organism: str,
    ensembl_ids: list[str] | None = None,
    gene_names: list[str] | None = None,
) -> dict[str, int]:
    """Look up gene_index values for Ensembl IDs already in the genes table."""
    assert (ensembl_ids is not None) ^ (gene_names is not None), (
        "Must provide either ensembl_ids or gene_names for lookup exclusively"
    )
    escaped_organism = _escape_lance_value(organism)
    if ensembl_ids:
        ensembl_ids_str = ", ".join(f"'{_escape_lance_value(eid)}'" for eid in ensembl_ids)
        where_clause = f"organism = '{escaped_organism}' AND ensembl_id IN ({ensembl_ids_str})"
        select_column = "ensembl_id"
    else:
        gene_names_str = ", ".join(f"'{_escape_lance_value(name)}'" for name in gene_names)
        where_clause = f"organism = '{escaped_organism}' AND gene_name IN ({gene_names_str})"
        select_column = "gene_name"

    df = genes_table.search().where(where_clause).select([select_column, "gene_index"]).to_pandas()
    return dict(zip(df[select_column], df["gene_index"], strict=False))


def create_new_gene_records_for_table(
    genes_table: lancedb.Table,
    organism: str,
    gene_names: set[str] | None = None,
    ensembl_ids: set[str] | None = None,
) -> None:
    assert (gene_names is not None) ^ (ensembl_ids is not None), (
        "Must provide either gene_names or ensembl_ids for lookup exclusively"
    )

    gene_records_to_add = []
    next_gene_index = get_max_gene_index(genes_table) + 1
    if gene_names is not None:
        assert isinstance(gene_names, set), "gene_names should be a set for efficient lookup"
        gene_name_mapping = lookup_gene_indices_from_table(
            genes_table, organism, gene_names=gene_names
        )
        missing_gene_names = [name for name in gene_names if name not in gene_name_mapping]
        missing_gene_name_ensembl_list = standardize_metadata_to_ontology(
            missing_gene_names,
            OntologyEntity.GENE,
            organism=organism,
            field="symbol",
            return_field="ensembl_gene_id",
        )
        missing_gene_name_ensembl = dict(
            zip(missing_gene_names, missing_gene_name_ensembl_list, strict=False)
        )
        for gene_name in missing_gene_names:
            ensembl_id = missing_gene_name_ensembl[gene_name]
            # If standardization returned the original name, it means no Ensembl ID was found
            if ensembl_id == gene_name:
                ensembl_id = None
            record = GeneSchema(
                gene_index=next_gene_index,
                gene_name=gene_name,
                ensembl_id=ensembl_id,
                ensembl_version=None,
                organism=organism,
            )
            gene_records_to_add.append(record)
            next_gene_index += 1

    if ensembl_ids is not None:
        assert isinstance(ensembl_ids, set), "ensembl_ids should be a set for efficient lookup"
        ensembl_id_mapping = lookup_gene_indices_from_table(
            genes_table, organism, ensembl_ids=ensembl_ids
        )
        missing_ensembl_ids = [eid for eid in ensembl_ids if eid not in ensembl_id_mapping]
        missing_ensembl_id_gene_names_list = standardize_metadata_to_ontology(
            missing_ensembl_ids,
            OntologyEntity.GENE,
            organism=organism,
            field="ensembl_gene_id",
            return_field="symbol",
        )
        missing_ensembl_id_gene_names = dict(
            zip(missing_ensembl_ids, missing_ensembl_id_gene_names_list, strict=False)
        )
        for ensembl_id in missing_ensembl_ids:
            gene_name = missing_ensembl_id_gene_names.get(ensembl_id) or ensembl_id
            record = GeneSchema(
                gene_index=next_gene_index,
                gene_name=gene_name,
                ensembl_id=ensembl_id,
                ensembl_version=None,
                organism=organism,
            )
            gene_records_to_add.append(record)
            next_gene_index += 1

    return gene_records_to_add


def register_genes_two_stage(
    db: lancedb.DBConnection,
    organism: str,
    *,
    measured_ensembl_ids: set[str] | None = None,
    measured_gene_names: set[str] | None = None,
    perturbation_gene_names: set[str] | None = None,
    measured_ensembl_ids_by_organism: dict[str, set[str]] | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Register measured genes and perturbation targets, returning lookup dicts.

    Encapsulates the two-stage gene registration pattern used by all ingestion
    scripts: first register measured genes (by Ensembl ID or gene symbol), then
    register perturbation targets (by gene name) as a separate step to avoid
    duplicates.

    Exactly one of ``measured_ensembl_ids``, ``measured_gene_names``, or
    ``measured_ensembl_ids_by_organism`` must be provided for measured genes.

    Parameters
    ----------
    db
        LanceDB connection.
    organism
        Organism string (e.g. ``"human"``).  Ignored when
        ``measured_ensembl_ids_by_organism`` is used (organisms come from the dict
        keys).
    measured_ensembl_ids
        Set of Ensembl gene IDs for measured genes (single organism).
    measured_gene_names
        Set of gene symbols for measured genes (when no Ensembl IDs available).
    perturbation_gene_names
        Set of gene symbols for perturbation targets (optional).
    measured_ensembl_ids_by_organism
        Dict mapping organism → set of Ensembl IDs for barnyard/multi-organism
        datasets.

    Returns
    -------
    measured_lookup : dict[str, int]
        Mapping from measured gene identifier (Ensembl ID or gene symbol) to
        ``gene_index``.
    perturbation_lookup : dict[str, int]
        Mapping from perturbation target gene name to ``gene_index``.
        Empty dict when ``perturbation_gene_names`` is not provided.
    """
    n_measured_args = sum(
        x is not None
        for x in [measured_ensembl_ids, measured_gene_names, measured_ensembl_ids_by_organism]
    )
    assert n_measured_args == 1, (
        "Must provide exactly one of measured_ensembl_ids, measured_gene_names, "
        "or measured_ensembl_ids_by_organism"
    )

    genes_table = upsert_table(db, "genes", None, schema=GeneSchema)

    # Stage 1: measured genes
    if measured_ensembl_ids_by_organism is not None:
        # Barnyard: register per organism
        for org, ensembl_ids in measured_ensembl_ids_by_organism.items():
            new_records = create_new_gene_records_for_table(
                genes_table, org, ensembl_ids=ensembl_ids
            )
            if new_records:
                print(f"  Registering {len(new_records)} {org} measured gene records...")
                genes_table.add(new_records)

        # Build combined lookup across all organisms
        measured_lookup: dict[str, int] = {}
        for org, ensembl_ids in measured_ensembl_ids_by_organism.items():
            measured_lookup.update(
                lookup_gene_indices_from_table(genes_table, org, ensembl_ids=ensembl_ids)
            )
    elif measured_ensembl_ids is not None:
        new_records = create_new_gene_records_for_table(
            genes_table, organism, ensembl_ids=measured_ensembl_ids
        )
        if new_records:
            print(f"  Registering {len(new_records)} measured gene records...")
            genes_table.add(new_records)
        measured_lookup = lookup_gene_indices_from_table(
            genes_table, organism, ensembl_ids=measured_ensembl_ids
        )
    else:
        new_records = create_new_gene_records_for_table(
            genes_table, organism, gene_names=measured_gene_names
        )
        if new_records:
            print(f"  Registering {len(new_records)} measured gene records...")
            genes_table.add(new_records)
        measured_lookup = lookup_gene_indices_from_table(
            genes_table, organism, gene_names=measured_gene_names
        )

    # Stage 2: perturbation targets by gene name
    perturbation_lookup: dict[str, int] = {}
    if perturbation_gene_names:
        new_records = create_new_gene_records_for_table(
            genes_table, organism, gene_names=perturbation_gene_names
        )
        if new_records:
            print(f"  Registering {len(new_records)} perturbation target gene records...")
            genes_table.add(new_records)
        perturbation_lookup = lookup_gene_indices_from_table(
            genes_table, organism, gene_names=perturbation_gene_names
        )

    return measured_lookup, perturbation_lookup


def resolve_pubchem_cids(
    names: list[str] | None = None,
    smiles: list[str] | None = None,
) -> tuple[dict[str, int], set[str]]:
    """Resolve molecule names or SMILES to PubChem CIDs via the PubChem API.

    Standalone function — does not require a LanceDB table. Intended for use
    during data preparation to validate and standardize compound identifiers
    before ingestion.

    Accepts exactly one of ``names`` or ``smiles``.

    Returns
    -------
    resolved : dict[str, int]
        Mapping from each input value to its PubChem CID (only resolved entries).
    unresolved : set[str]
        Input values for which no CID could be found.
    """
    from time import sleep

    import pubchempy as pcp

    assert (names is not None) ^ (smiles is not None), "Must provide exactly one of names or smiles"

    values = list(filter(lambda x: isinstance(x, str), set(names or smiles)))
    namespace = "name" if names is not None else "smiles"

    resolved: dict[str, int] = {}
    for value in values:
        try:
            cids = pcp.get_cids(value, namespace=namespace)
        except pcp.BadRequestError:
            cids = []
        if cids:
            resolved[value] = cids[0]
        sleep(0.2)  # PubChem rate limit: max 5 req/s

    unresolved = {v for v in values if v not in resolved}
    return resolved, unresolved


def lookup_pubchem_cids(
    molecules_table: lancedb.Table,
    names: list[str] | None = None,
    smiles: list[str] | None = None,
) -> dict[str, int]:
    """Look up molecule sample_uids by PubChem CID. Returns an empty dict if not found."""
    assert (names is not None) ^ (smiles is not None), (
        "Must provide either names or smiles for molecule lookup exclusively"
    )
    if names is not None:
        names = list(filter(lambda x: isinstance(x, str), set(names)))
        query_str = ", ".join(f"'{_escape_lance_value(name)}'" for name in names)
        field = "iupac_name"
    else:
        smiles = list(filter(lambda x: isinstance(x, str), set(smiles)))
        query_str = ", ".join(f"'{_escape_lance_value(s)}'" for s in smiles)
        field = "smiles"

    df = (
        molecules_table.search()
        .where(f"{field} IN ({query_str})")
        .select([field, "pubchem_cid"])
        .to_pandas()
    )
    field_to_cid_mapping = dict(zip(df[field], df["pubchem_cid"], strict=False))

    # Get molecules that we couldn't find CIDs for and look them
    # up with the PubChem API. This is slower because of rate limits but should
    # be needed less often as we build up the molecules table.
    missing_values = [v for v in (names or smiles) if v not in field_to_cid_mapping]
    if missing_values:
        from time import sleep

        import pubchempy as pcp

        for value in missing_values:
            if not isinstance(value, str):
                # There might be a NaN or None in the list
                continue

            result = pcp.get_cids(value, namespace=("name" if names is not None else "smiles"))
            if result:
                field_to_cid_mapping[value] = result[0]

            # Max of 5 requests per second to PubChem API to avoid rate limiting
            sleep(0.2)

    return field_to_cid_mapping


def lookup_molecule_uid(
    molecules_table: lancedb.Table,
    values: list[str] | list[int],
    field: Literal["pubchem_cid", "name", "smiles"] = "pubchem_cid",
) -> dict[str | int, str]:
    """Look up a molecule's sample_uid by PubChem CID. Returns an empty dict if not found."""
    if field == "pubchem_cid":
        # This field is numeric
        values_to_search = ", ".join(str(v) for v in values)
    else:
        # These fields are strings
        values_to_search = ", ".join(f"'{_escape_lance_value(str(v))}'" for v in values)

    df = (
        molecules_table.search()
        .where(f"{field} IN ({values_to_search})")
        .select(["sample_uid", field])
        .to_pandas()
    )
    if df.empty:
        return {}

    return dict(zip(df[field], df["sample_uid"], strict=False))


def resolve_molecule_uids_by_pubchem_cid(
    molecules_table: lancedb.Table,
    pubchem_cid: list[int] | None = None,
    name: list[str] | None = None,
    smiles: list[str] | None = None,
) -> tuple[list[str | None], list[MoleculeSchema], set]:
    """Resolve molecules to sample UIDs, creating new records as needed.

    Accepts exactly one of ``pubchem_cid``, ``name``, or ``smiles`` as a list.

    Returns
    -------
    sample_uids
        List of ``sample_uid`` strings (or ``None`` for unresolved) aligned with
        the input list.
    new_records
        List of new ``MoleculeSchema`` records. Caller must add them to the table.
    unresolved
        Set of input values that could not be resolved.
    """
    assert sum(x is not None for x in [pubchem_cid, name, smiles]) == 1, (
        "Must provide exactly one of pubchem_cid, name, or smiles for molecule lookup"
    )
    if name is not None:
        field_to_cid_mapping = lookup_pubchem_cids(molecules_table, names=name)
        cid_to_sample_uid = lookup_molecule_uid(
            molecules_table, list(field_to_cid_mapping.values()), field="pubchem_cid"
        )
    elif smiles is not None:
        field_to_cid_mapping = lookup_pubchem_cids(molecules_table, smiles=smiles)
        cid_to_sample_uid = lookup_molecule_uid(
            molecules_table, list(field_to_cid_mapping.values()), field="pubchem_cid"
        )
    else:
        field_to_cid_mapping = {cid: cid for cid in pubchem_cid}
        cid_to_sample_uid = lookup_molecule_uid(molecules_table, pubchem_cid, field="pubchem_cid")

    for cid in field_to_cid_mapping.values():
        if cid not in cid_to_sample_uid:
            cid_to_sample_uid[cid] = None

    new_cids = [cid for cid, uid in cid_to_sample_uid.items() if uid is None]
    new_records = []
    if new_cids:
        import pubchempy as pcp

        for compound in pcp.get_compounds(new_cids, namespace="cid"):
            record = MoleculeSchema(
                pubchem_cid=compound.cid,
                iupac_name=compound.iupac_name,
                smiles=compound.connectivity_smiles,
            )
            new_records.append(record)
            cid_to_sample_uid[compound.cid] = record.sample_uid

    # Convert the original input values to sample_uids using the conversion mapping
    if pubchem_cid is not None:
        sample_uids = [cid_to_sample_uid.get(field_to_cid_mapping.get(v)) for v in pubchem_cid]
        unresolved = set(
            [v for v, uid in zip(pubchem_cid, sample_uids, strict=False) if uid is None]
        )
    elif name is not None:
        sample_uids = [cid_to_sample_uid.get(field_to_cid_mapping.get(v)) for v in name]
        unresolved = set([v for v, uid in zip(name, sample_uids, strict=False) if uid is None])
    else:
        sample_uids = [cid_to_sample_uid.get(field_to_cid_mapping.get(v)) for v in smiles]
        unresolved = set([v for v, uid in zip(smiles, sample_uids, strict=False) if uid is None])

    return sample_uids, new_records, unresolved


def build_positional_to_gene_index(
    gene_names_or_ensembl_ids: list[str],
    gene_index_lookup: dict[str, int],
) -> np.ndarray:
    """Build measured_gene_expression_indices bytes for DatasetSchema.

    Maps gene names or Ensembl IDs to gene_index values, preserving order.
    """
    indices = np.array(
        [gene_index_lookup[gene] for gene in gene_names_or_ensembl_ids],
        dtype=np.int32,
    )
    return indices


# ---------------------------------------------------------------------------
# Image feature registration
# ---------------------------------------------------------------------------


def register_image_features(
    db: lancedb.DBConnection,
    feature_names: list[str],
) -> dict[str, int]:
    """Register image features and return name -> feature_index mapping."""
    table_names = db.list_tables().tables
    if "image_features" in table_names:
        table = db.open_table("image_features")
        existing_df = table.search().select(["feature_name", "feature_index"]).to_pandas()
        existing_map = dict(
            zip(existing_df["feature_name"], existing_df["feature_index"], strict=False)
        )
        next_index = max(existing_map.values()) + 1 if existing_map else 0
    else:
        existing_map = {}
        next_index = 0
        table = None

    new_records = []
    for name in feature_names:
        if name not in existing_map:
            existing_map[name] = next_index
            new_records.append(
                ImageFeatureSchema(
                    feature_index=next_index,
                    feature_name=name,
                    description=None,
                )
            )
            next_index += 1

    if new_records:
        if table is None:
            db.create_table("image_features", data=new_records)
        else:
            table.add(new_records)
        print(f"  Registered {len(new_records)} new image features (total: {next_index})")
    else:
        print(f"  All {len(feature_names)} features already registered")

    return existing_map


# ---------------------------------------------------------------------------
# Chromosome registration
# ---------------------------------------------------------------------------


def register_chromosomes(
    db: lancedb.DBConnection,
    organism: str,
    chromosome_names: list[str],
    assembly: str,
    chromosome_sizes: dict[str, int] | None = None,
) -> dict[str, int]:
    """Register chromosomes for an organism and return name -> chromosome_index mapping.

    Parameters
    ----------
    assembly
        Genome assembly name (e.g., ``"GRCh38"``, ``"GRCm39"``).
    chromosome_sizes
        Mapping from chromosome name to size in base pairs. Required for new
        chromosomes; ignored for chromosomes that already exist in the table.
    """
    table_names = db.list_tables().tables
    if "chromosomes" in table_names:
        table = db.open_table("chromosomes")
        existing_df = (
            table.search()
            .where(
                f"organism = '{_escape_lance_value(organism)}' and assembly = '{_escape_lance_value(assembly)}'"
            )
            .select(["chromosome_name", "chromosome_index"])
            .to_pandas()
        )
        existing_map = dict(
            zip(existing_df["chromosome_name"], existing_df["chromosome_index"], strict=False)
        )
        # Next index must be global across all organisms
        all_df = table.search().select(["chromosome_index"]).to_pandas()
        next_index = int(all_df["chromosome_index"].max()) + 1 if not all_df.empty else 0
    else:
        existing_map = {}
        next_index = 0
        table = None

    new_records = []
    for name in chromosome_names:
        if name not in existing_map:
            assert chromosome_sizes is not None and name in chromosome_sizes, (
                f"chromosome_sizes must include '{name}' when registering new chromosomes"
            )
            assert assembly is not None, "assembly is required when registering new chromosomes"
            existing_map[name] = next_index
            new_records.append(
                ChromosomeSchema(
                    chromosome_index=next_index,
                    chromosome_name=name,
                    chromosome_size=chromosome_sizes[name],
                    organism=organism,
                    assembly=assembly,
                )
            )
            next_index += 1

    if new_records:
        if table is None:
            db.create_table("chromosomes", data=new_records)
        else:
            table.add(new_records)
        print(
            f"  Registered {len(new_records)} new chromosomes for {organism} (total: {next_index})"
        )
    else:
        print(f"  All {len(chromosome_names)} chromosomes already registered for {organism}")

    return existing_map


def fetch_chromosome_sizes(assembly_name: str) -> dict[str, int]:
    """Fetch chromosome sizes from NCBI for a genome assembly.

    Uses the Entrez API to resolve the assembly name to a RefSeq accession,
    then queries the NCBI Datasets API for sequence reports. Only returns
    assembled molecules (chromosomes), not unplaced scaffolds or patches.

    Parameters
    ----------
    assembly_name
        NCBI assembly name, e.g. ``"GRCh38"`` or ``"GRCm39"``.

    Returns
    -------
    dict[str, int]
        Mapping from UCSC-style chromosome names (e.g., ``"chr1"``) to sizes
        in base pairs.
    """
    import requests

    # 1. Find the assembly UID via Entrez
    handle = Entrez.esearch(db="assembly", term=f"{assembly_name}[Assembly Name]")
    search_results = Entrez.read(handle)
    handle.close()
    id_list = search_results["IdList"]
    assert len(id_list) > 0, f"No assembly found for '{assembly_name}'"
    uid = id_list[0]

    # 2. Get the RefSeq accession from the assembly summary
    handle = Entrez.esummary(db="assembly", id=uid)
    summary = Entrez.read(handle)
    handle.close()
    doc = summary["DocumentSummarySet"]["DocumentSummary"][0]
    accession = doc.get("AssemblyAccession", "")
    assert accession, f"No accession found for assembly '{assembly_name}' (UID {uid})"

    # 3. Fetch sequence reports from the NCBI Datasets API
    datasets_url = (
        f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha"
        f"/genome/accession/{accession}/sequence_reports"
    )
    resp = requests.get(datasets_url, timeout=30)
    resp.raise_for_status()
    seq_data = resp.json()

    # 4. Filter to assembled-molecule sequences and build the mapping
    chrom_sizes: dict[str, int] = {}
    for report in seq_data.get("reports", []):
        if report.get("role") != "assembled-molecule":
            continue
        ucsc_name = report.get("ucsc_style_name", "")
        length = report.get("length", 0)
        if not ucsc_name or not length:
            continue
        chrom_sizes[ucsc_name] = int(length)

    assert len(chrom_sizes) > 0, (
        f"No assembled-molecule chromosomes found for assembly '{assembly_name}'"
    )
    return chrom_sizes


# ---------------------------------------------------------------------------
# Protein registration
# ---------------------------------------------------------------------------


def register_proteins(
    db: lancedb.DBConnection,
    uniprot_ids: set[str],
    gene_names: dict[str, str],
    organism: str,
) -> dict[str, int]:
    """Register proteins and return gene_name -> protein_index mapping.

    Fetches protein metadata (name, sequence) from the UniProt REST API for
    any proteins not already in the table. Only new proteins are fetched.

    Parameters
    ----------
    db
        LanceDB connection.
    uniprot_ids
        Set of UniProt accession IDs to register.
    gene_names
        Mapping from UniProt ID to gene symbol (e.g., ``{"Q9NZQ7": "CD274"}``).
    organism
        Organism string, e.g., ``"human"``.

    Returns
    -------
    dict[str, int]
        Mapping from gene_name to protein_index for all requested proteins.
    """
    import requests

    table_names = db.list_tables().tables
    if "proteins" in table_names:
        proteins_table = db.open_table("proteins")
        existing_df = (
            proteins_table.search().select(["uniprot_id", "gene_name", "protein_index"]).to_pandas()
        )
        existing_uid_map = dict(
            zip(existing_df["uniprot_id"], existing_df["protein_index"], strict=False)
        )
        existing_gene_map = dict(
            zip(existing_df["gene_name"], existing_df["protein_index"], strict=False)
        )
        all_df = proteins_table.search().select(["protein_index"]).to_pandas()
        next_index = int(all_df["protein_index"].max()) + 1 if not all_df.empty else 0
    else:
        existing_uid_map = {}
        existing_gene_map = {}
        next_index = 0
        proteins_table = None

    new_records = []
    for uid in uniprot_ids:
        if uid in existing_uid_map:
            continue

        resp = requests.get(
            f"https://rest.uniprot.org/uniprotkb/{uid}.json",
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        protein_name = (
            data.get("proteinDescription", {})
            .get("recommendedName", {})
            .get("fullName", {})
            .get("value")
        )
        if not protein_name:
            submitted = data.get("proteinDescription", {}).get("submittedName", [])
            protein_name = (
                submitted[0]["fullName"]["value"] if submitted else gene_names.get(uid, uid)
            )

        sequence_data = data.get("sequence", {})
        sequence = sequence_data.get("value", "")
        sequence_length = sequence_data.get("length", len(sequence))

        gene_name = gene_names.get(uid, uid)
        record = ProteinSchema(
            protein_index=next_index,
            uniprot_id=uid,
            protein_name=protein_name,
            gene_name=gene_name,
            organism=organism,
            sequence=sequence,
            sequence_length=sequence_length,
        )
        new_records.append(record)
        existing_uid_map[uid] = next_index
        existing_gene_map[gene_name] = next_index
        next_index += 1

    if new_records:
        if proteins_table is None:
            db.create_table("proteins", data=new_records)
        else:
            proteins_table.add(new_records)
        print(f"  Registered {len(new_records)} new proteins (total: {next_index})")
    else:
        print(f"  All {len(uniprot_ids)} proteins already registered")

    return existing_gene_map


def build_cell_metadata_kwargs(
    *,
    # Primary/Foreign keys
    cell_uid: str,
    dataset_uid: str,
    # Standard metadata fields
    assay: str,
    barcode: str | None = None,
    cell_line: str | None = None,
    development_stage: str | None = None,
    disease: str | None = None,
    organism: str | None = None,
    cell_type: str | None = None,
    tissue: str | None = None,
    additional_metadata: str | None = None,
    # Spatial/temporal metadata fields
    coord_t: float | None = None,
    coord_z: float | None = None,
    coord_y: float | None = None,
    coord_x: float | None = None,
    time_unit: str | None = None,
    spatial_unit: str | None = None,
    # Perturbation metadata fields
    is_control: bool | None = None,
    chemical_perturbation_uid: list[str] | None = None,
    chemical_perturbation_concentration: list[float] | None = None,
    chemical_perturbation_additional_metadata: list[str] | None = None,
    genetic_perturbation_gene_index: list[int] | None = None,
    genetic_perturbation_method: list[str] | None = None,
    genetic_perturbation_concentration: list[float] | None = None,
    genetic_perturbation_additional_metadata: list[str] | None = None,
) -> dict:
    """Build a kwargs dict for constructing CellIndex and modality records.

    Returns a dict with all _CellMetadataBase fields populated so it can be
    splatted into both ``CellIndex(...)`` and any modality record constructor.

    The ``barcode`` parameter is automatically merged into
    ``additional_metadata`` under the ``"barcode"`` key. If
    ``additional_metadata`` is already provided (as a JSON string), the barcode
    is added to the existing dict.
    """
    # Merge barcode into additional_metadata
    if barcode is not None:
        meta_dict = json.loads(additional_metadata) if additional_metadata else {}
        meta_dict["barcode"] = barcode
        additional_metadata = json.dumps(meta_dict)

    return {
        "cell_uid": cell_uid,
        "dataset_uid": dataset_uid,
        "assay": assay,
        "cell_line": cell_line,
        "development_stage": development_stage,
        "disease": disease,
        "organism": organism,
        "cell_type": cell_type,
        "tissue": tissue,
        "additional_metadata": additional_metadata,
        "coord_t": coord_t,
        "coord_z": coord_z,
        "coord_y": coord_y,
        "coord_x": coord_x,
        "time_unit": time_unit,
        "spatial_unit": spatial_unit,
        "is_control": is_control,
        "chemical_perturbation_uid": chemical_perturbation_uid,
        "chemical_perturbation_concentration": chemical_perturbation_concentration,
        "chemical_perturbation_additional_metadata": chemical_perturbation_additional_metadata,
        "genetic_perturbation_gene_index": genetic_perturbation_gene_index,
        "genetic_perturbation_method": genetic_perturbation_method,
        "genetic_perturbation_concentration": genetic_perturbation_concentration,
        "genetic_perturbation_additional_metadata": genetic_perturbation_additional_metadata,
    }


def write_cell_batch(
    db: lancedb.DBConnection,
    cell_index_records: list[CellIndex],
    modality_batches: dict[str, list[CellDataRecord]],
) -> None:
    """Write a batch of cell records to CellIndex and modality tables.

    Parameters
    ----------
    db
        LanceDB connection.
    cell_index_records
        List of CellIndex records to write.
    modality_batches
        Dict mapping table name to list of modality records.
        E.g. ``{"gene_expression": [GeneExpressionRecord(...), ...]}``
    """
    if cell_index_records:
        upsert_table(db, "cell_index", cell_index_records)
    for table_name, records in modality_batches.items():
        if records:
            upsert_table(db, table_name, records)


def standardize_metadata_to_ontology(
    values: list[str] | np.ndarray,
    entity: OntologyEntity,
    field: str,
    organism: str | None = None,
    return_field: str | None = None,
) -> list[str]:
    """
    Takes a list of metadata values and tries to standardize them to a public
    ontology based on the entity type. Returns standardized values when possible,
    otherwise it returns the original value.
    """
    import bionty as bt

    entity_str = entity.value
    if entity_str in ["CellLine", "CellType", "Tissue", "Disease"] or entity_str is None:
        organism = "all"

    ontology = getattr(bt, entity.value).public(organism=organism or "all")
    standard_values = ontology.standardize(values, field=field, return_field=return_field)
    return standard_values


def validate_metadata_against_ontology(
    values: list[str] | np.ndarray,
    entity: OntologyEntity,
    field: str,
    organism: str | None = None,
    return_field: str | None = None,
) -> list[str]:
    """
    Takes a list of metadata values and validates them against a public ontology after
    standardization. Returns a list of values that failed validation. This function
    should be used to make sense that values will resolve correctly before adding
    them to records.
    """
    import bionty as bt

    entity_str = entity.value
    if entity_str in ["CellLine", "CellType", "Tissue", "Disease"] or entity_str is None:
        organism = "all"

    ontology = getattr(bt, entity.value).public(organism=organism or "all")
    standard_values = ontology.standardize(values, field=field, return_field=field)
    validated = ontology.validate(standard_values, field=field)

    return np.array(standard_values)[~validated].tolist()


def search_metadata_in_ontology(
    query: str,
    entity: OntologyEntity,
    organism: str | None = None,
) -> pd.DataFrame:
    """
    Searches a query term in the specified ontology and returns a dataframe of results.
    This can be used to inspect why certain metadata values may not be validating or standardizing
    correctly and map them to the closest valid term in the ontology.
    """
    import bionty as bt

    entity_str = entity.value
    if entity_str in ["CellLine", "CellType", "Tissue", "Disease"] or entity_str is None:
        organism = "all"

    ontology = getattr(bt, entity.value).public(organism=organism or "all")
    return ontology.search(query)
