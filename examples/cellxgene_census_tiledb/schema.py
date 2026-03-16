"""Schemas for ingesting the CellxGene Census mouse TileDB-SOMA store into lancell.

The obs/var columns match the census 2025-11-17 release for mus_musculus.
Unlike the h5ad-based census example, var metadata here is limited to what
tiledbsoma exposes (no feature_reference, feature_biotype, feature_is_filtered).
"""

from lancell.schema import (
    DatasetRecord,
    FeatureBaseSchema,
    LancellBaseSchema,
    SparseZarrPointer,
)


class GeneFeatureSpace(FeatureBaseSchema):
    """Gene feature registry entry matching tiledbsoma census var columns."""

    ensembl_id: str  # feature_id in the census (e.g. ENSMUSG00000021124)
    feature_name: str
    feature_type: str  # e.g. "protein_coding", "lncRNA"
    feature_length: int


class CensusDatasetRecord(DatasetRecord):
    """Dataset record with cellxgene census provenance."""

    cellxgene_dataset_id: str
    census_release_date: str = "2025-11-17"


class CellObs(LancellBaseSchema):
    """Cell-level observation schema for the mouse census TileDB-SOMA store.

    Columns match the obs DataFrame in the 2025-11-17 census release for
    mus_musculus. Excludes soma_joinid (used only as a join key during ingest).
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
    tissue_general: str | None = None
    suspension_type: str | None = None
    donor_id: str | None = None
    is_primary_data: bool | None = None
    observation_joinid: str | None = None

    # Ontology term IDs
    assay_ontology_term_id: str | None = None
    cell_type_ontology_term_id: str | None = None
    disease_ontology_term_id: str | None = None
    sex_ontology_term_id: str | None = None
    tissue_ontology_term_id: str | None = None
    self_reported_ethnicity_ontology_term_id: str | None = None
    development_stage_ontology_term_id: str | None = None
    tissue_general_ontology_term_id: str | None = None

    # QC summary stats from the census
    raw_sum: float | None = None
    nnz: int | None = None
    raw_mean_nnz: float | None = None
    raw_variance_nnz: float | None = None
    n_measured_vars: int | None = None
