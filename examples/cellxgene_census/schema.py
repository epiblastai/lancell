"""Schemas for a CellxGene Census atlas built on lancell's ragged atlas framework.

Defines:
- GeneFeatureSpace: feature registry schema for genes (var metadata)
- CensusDatasetRecord: dataset-level metadata with cellxgene provenance
- CellObs: cell-level observation schema with gene_expression pointer
"""

from lancell.schema import (
    DatasetRecord,
    FeatureBaseSchema,
    LancellBaseSchema,
    SparseZarrPointer,
)


class GeneFeatureSpace(FeatureBaseSchema):
    """Gene feature registry entry, matching cellxgene census var columns."""

    ensembl_id: str
    feature_name: str
    feature_reference: str
    feature_biotype: str
    feature_length: int
    feature_type: str
    feature_is_filtered: bool


class CensusDatasetRecord(DatasetRecord):
    """Dataset record with cellxgene census provenance."""

    cellxgene_dataset_id: str
    census_release_date: str = "2025-11-17"


class CellObs(LancellBaseSchema):
    """Cell-level observation schema for cellxgene census data.

    All 23 obs columns from the harmonized cellxgene h5ad files are included.
    """

    # Pointer to sparse gene expression data
    gene_expression: SparseZarrPointer | None = None

    # Human-readable metadata
    assay: str | None = None
    cell_type: str | None = None
    disease: str | None = None
    sex: str | None = None
    tissue: str | None = None
    self_reported_ethnicity: str | None = None
    development_stage: str | None = None
    tissue_type: str | None = None
    disease_state: str | None = None
    author_cell_type: str | None = None
    genotype: str | None = None
    suspension_type: str | None = None
    donor_id: str | None = None
    sample: str | None = None
    is_primary_data: bool | None = None

    # Ontology term IDs
    assay_ontology_term_id: str | None = None
    cell_type_ontology_term_id: str | None = None
    disease_ontology_term_id: str | None = None
    sex_ontology_term_id: str | None = None
    tissue_ontology_term_id: str | None = None
    self_reported_ethnicity_ontology_term_id: str | None = None
    development_stage_ontology_term_id: str | None = None
