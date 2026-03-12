"""Biomedical Data Standardization Suite.

Modular, fully independent standardization library for resolving messy
metadata to canonical identifiers and CELLxGENE-compatible ontology term IDs.
No coupling to ingestion_utils.py or LanceDB.

Caches API results at ``~/.cache/lancell/``.
"""

from lancell.standardization.cache import StandardizationCache, get_cache
from lancell.standardization.genes import detect_organism_from_ensembl_ids, resolve_genes
from lancell.standardization.molecules import (
    canonicalize_smiles,
    clean_compound_name,
    is_control_compound,
    resolve_molecules,
)
from lancell.standardization.ontologies import (
    OntologyEntity,
    get_ontology_ancestors,
    get_ontology_descendants,
    get_ontology_siblings,
    get_ontology_term_id,
    resolve_assays,
    resolve_cell_types,
    resolve_diseases,
    resolve_ontology_terms,
    resolve_organisms,
    resolve_tissues,
)
from lancell.standardization.ncbi import (
    BioProjectMetadata,
    BioSampleMetadata,
    GeoSampleMetadata,
    GeoSeriesMetadata,
    fetch_bioproject,
    fetch_biosample,
    fetch_geo_biosample_attrs,
    fetch_geo_metadata,
    fetch_geo_sample,
    fetch_geo_series,
    link_accessions,
)
from lancell.standardization.perturbations import (
    classify_perturbation_method,
    detect_control_labels,
    detect_negative_control_type,
    is_control_label,
    parse_combinatorial_perturbations,
)
from lancell.standardization.types import (
    GeneResolution,
    MoleculeResolution,
    OntologyResolution,
    ProteinResolution,
    Resolution,
    ResolutionReport,
)

__all__ = [
    # Types
    "Resolution",
    "GeneResolution",
    "MoleculeResolution",
    "ProteinResolution",
    "OntologyResolution",
    "ResolutionReport",
    # Cache
    "StandardizationCache",
    "get_cache",
    # Genes
    "resolve_genes",
    "detect_organism_from_ensembl_ids",
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
    "resolve_tissues",
    "resolve_diseases",
    "resolve_organisms",
    "resolve_assays",
    # Ontology hierarchy
    "get_ontology_ancestors",
    "get_ontology_descendants",
    "get_ontology_siblings",
    # Perturbations
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
]
