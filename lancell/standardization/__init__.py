"""Biomedical Data Standardization Suite.

Modular, fully independent standardization library for resolving messy
metadata to canonical identifiers and CELLxGENE-compatible ontology term IDs.
No coupling to ingestion_utils.py or LanceDB.
"""

from lancell.standardization.genes import (
    detect_organism_from_ensembl_ids,
    is_placeholder_symbol,
    resolve_genes,
)
from lancell.standardization.gget import annotate_genomic_coordinates
from lancell.standardization.guide_rna import resolve_guide_sequences
from lancell.standardization.metadata_table import get_reference_db, set_reference_db_path
from lancell.standardization.molecules import (
    canonicalize_smiles,
    clean_compound_name,
    is_control_compound,
    resolve_molecules,
)
from lancell.standardization.ncbi import (
    BioProjectMetadata,
    BioSampleMetadata,
    GeoSampleMetadata,
    GeoSeriesMetadata,
    PublicationFullText,
    PublicationMetadata,
    PublicationSection,
    fetch_bioproject,
    fetch_biosample,
    fetch_geo_biosample_attrs,
    fetch_geo_metadata,
    fetch_geo_sample,
    fetch_geo_series,
    fetch_publication,
    fetch_publication_metadata,
    fetch_publication_text,
    link_accessions,
    search_pubmed_by_title,
)
from lancell.standardization.ontologies import (
    OntologyEntity,
    get_ontology_ancestors,
    get_ontology_descendants,
    get_ontology_siblings,
    get_ontology_term_id,
    resolve_assays,
    resolve_cell_lines,
    resolve_cell_types,
    resolve_diseases,
    resolve_ontology_terms,
    resolve_organisms,
    resolve_tissues,
)
from lancell.standardization.perturbations import (
    GeneticPerturbationType,
    classify_perturbation_method,
    detect_control_labels,
    detect_negative_control_type,
    is_control_label,
    parse_combinatorial_perturbations,
)
from lancell.standardization.proteins import resolve_proteins
from lancell.standardization.types import (
    CellLineResolution,
    GeneResolution,
    GuideRnaResolution,
    MoleculeResolution,
    OntologyResolution,
    ProteinResolution,
    Resolution,
    ResolutionReport,
)

__all__ = [
    # Types
    "Resolution",
    "CellLineResolution",
    "GeneResolution",
    "GuideRnaResolution",
    "MoleculeResolution",
    "ProteinResolution",
    "OntologyResolution",
    "ResolutionReport",
    # Reference DB
    "set_reference_db_path",
    "get_reference_db",
    # Genes
    "resolve_genes",
    "detect_organism_from_ensembl_ids",
    "is_placeholder_symbol",
    # Guide RNAs
    "resolve_guide_sequences",
    "annotate_genomic_coordinates",
    # Proteins
    "resolve_proteins",
    # Molecules
    "resolve_molecules",
    "clean_compound_name",
    "is_control_compound",
    "canonicalize_smiles",
    # Ontologies
    "OntologyEntity",
    "resolve_ontology_terms",
    "get_ontology_term_id",
    "resolve_cell_types",
    "resolve_cell_lines",
    "resolve_tissues",
    "resolve_diseases",
    "resolve_organisms",
    "resolve_assays",
    # Ontology hierarchy
    "get_ontology_ancestors",
    "get_ontology_descendants",
    "get_ontology_siblings",
    # Perturbations
    "GeneticPerturbationType",
    "detect_control_labels",
    "is_control_label",
    "detect_negative_control_type",
    "parse_combinatorial_perturbations",
    "classify_perturbation_method",
    # NCBI metadata
    "GeoSeriesMetadata",
    "GeoSampleMetadata",
    "BioSampleMetadata",
    "BioProjectMetadata",
    "fetch_geo_metadata",
    "fetch_geo_series",
    "fetch_geo_sample",
    "fetch_biosample",
    "fetch_bioproject",
    "link_accessions",
    "fetch_geo_biosample_attrs",
    # Publications
    "PublicationMetadata",
    "PublicationSection",
    "PublicationFullText",
    "fetch_publication",
    "fetch_publication_metadata",
    "fetch_publication_text",
    "search_pubmed_by_title",
]
