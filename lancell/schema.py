import uuid
from datetime import datetime
from typing import Self

from lancedb.pydantic import LanceModel
from pydantic import Field, model_validator

# Table name constants
CELL_INDEX_TABLE = "cell_index"
GENE_EXPRESSION_TABLE = "gene_expression"
PROTEIN_ABUNDANCE_TABLE = "protein_abundance"
CHROMATIN_ACCESSIBILITY_TABLE = "chromatin_accessibility"
IMAGE_FEATURE_VECTORS_TABLE = "image_feature_vectors"
IMAGE_TILES_TABLE = "image_tiles"

FEATURE_SPACE_TO_TABLE: dict[str, str] = {
    "gene_expression": GENE_EXPRESSION_TABLE,
    "protein_abundance": PROTEIN_ABUNDANCE_TABLE,
    "chromatin_accessibility": CHROMATIN_ACCESSIBILITY_TABLE,
    "image_features": IMAGE_FEATURE_VECTORS_TABLE,
    "image_tiles": IMAGE_TILES_TABLE,
}


class OpenPMCRecord(LanceModel):
    PMC_ID: str
    image: str
    sub_caption: str | None
    full_caption: str | None
    intext_refs_summary: str | None
    intext_refs: str | None
    modality: str | None
    image_data: bytes

    # Auto-generated UID
    sample_uid: str = Field(default_factory=lambda: str(uuid.uuid4()))


class PublicationSchema(LanceModel):
    # PubMed id for the paper
    pmid: str
    # The doi for the paper
    doi: str
    # The title of the paper
    title: str
    # The journal that the paper was published in, if applicable
    journal: str
    # The year that the paper was published, if applicable
    publication_date: datetime

    # Full text for the publication from PMC; if not available, then
    # this is just the abstract.
    full_text: str


class DatasetSchema(LanceModel):
    # PubMed id for the paper associated with the dataset, if applicable.
    # Should match pmid in PublicationSchema
    pmid: str | None
    # The doi for the dataset if availble.
    doi: str | None
    # The number of cells determined from the AnnData
    cell_count: int

    # Valid features spaces are gene_expression, protein_abundance,
    # chromatin_accessibility, image statistics, etc. This determines
    # what table we search for setting the var index
    feature_space: str

    # The indices of features that were measured in this dataset. These features
    # depend on the feature space. For gene expression, these are indices in the
    # GeneSchema table. For images these are image features by name.
    measured_feature_indices: bytes | None

    # Database from which the dataset was downloaded, if applicable
    accession_database: str | None
    accession_id: str | None

    # Dataset description, for example the sample preparation and experimental
    # protocol text from the GEO series or sample record.
    dataset_description: str | None

    # Unique dataset uid for the dataset, this is generated automatically and should not be set
    # Multimodal datasets where two or more modalities are measured on the same cells
    # should have the same dataset_uid across the modalities to allow linking them together
    dataset_uid: str = Field(default_factory=lambda: str(uuid.uuid4()))


class _CellMetadataMixin(LanceModel):
    """Shared cell metadata fields present on CellIndex and all data tables."""

    cell_uid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_uid: str
    assay: str
    cell_line: str | None
    development_stage: str | None
    disease: str | None
    organism: str | None
    cell_type: str | None
    tissue: str | None
    additional_metadata: str | None

    # Spatio-temporal coordinates
    coord_t: float | None
    coord_z: float | None
    coord_y: float | None
    coord_x: float | None
    time_unit: str | None
    spatial_unit: str | None

    is_control: bool | None
    chemical_perturbation_uid: list[str] | None
    chemical_perturbation_concentration: list[float] | None
    chemical_perturbation_additional_metadata: list[str] | None
    genetic_perturbation_gene_index: list[int] | None
    genetic_perturbation_method: list[str] | None
    genetic_perturbation_concentration: list[float] | None
    genetic_perturbation_additional_metadata: list[str] | None

    # Auto-filled field
    perturbation_search_string: str = ""

    @model_validator(mode="after")
    def generate_search_string(self) -> Self:
        self.perturbation_search_string = _generate_perturbation_search_tokens(self)
        return self


class CellIndex(_CellMetadataMixin):
    """Lightweight index table for multimodal cell discovery.

    One row per physical cell. Links to per-modality tables via cell_uid.
    """

    modalities: list[str]


class GeneExpressionRecord(_CellMetadataMixin):
    """Per-cell gene expression data stored in the gene_expression table."""

    # This schema is designed for sparse matrices
    gene_indices: bytes
    counts: bytes


class ProteinAbundanceRecord(_CellMetadataMixin):
    """Per-cell protein abundance data stored in the protein_abundance table."""

    # This schema is designed for sparse matrices
    protein_indices: bytes
    counts: bytes


class ChromatinAccessibilityRecord(_CellMetadataMixin):
    """Per-cell chromatin accessibility data stored in the chromatin_accessibility table."""

    chromosome_indices: bytes
    fragment_start_positions: bytes
    fragment_lengths: bytes


class ImageFeatureVectorRecord(_CellMetadataMixin):
    """Per-cell image feature vectors stored in the image_feature_vectors table."""

    # Dense feature values for a fixed feature space per dataset
    feature_values: bytes


class ImageTileRecord(_CellMetadataMixin):
    """Per-cell image tiles stored in the image_tiles table."""

    image: bytes
    image_shape: list[int]
    image_dtype: str
    channel_names: list[str]


class GeneSchema(LanceModel):
    # NOTE: Unlike the other records, this one does not have a sample_uid. This is because
    # the gene table is expected to stay relatively small and does not support any kind of
    # parallel writes. It should be updated infrequently and the gene_index is rigorously
    # enforced to be sequential and unique.

    # The unique index of the gene, used for feature_indices in GeneExpressionRecord
    gene_index: int
    # The perturbed gene name, e.g., TP53
    gene_name: str
    # The gene ensembl id, e.g., ENSG00000141510
    # Some gene names may not resolve to an ensembl id, in which case this field should be None
    ensembl_id: str | None
    # The version of ensembl used for the gene annotations. Only set this if known,
    # otherwise leave as None
    ensembl_version: str | None

    # The organism, e.g., human, mouse, etc. that this gene corresponds to
    organism: str


class ImageFeatureSchema(LanceModel):
    # The unique index of the image feature
    feature_index: int
    # The name of the image feature, e.g., "mean_intensity_DAPI", "texture_feature_1", etc.
    feature_name: str
    # A description of the image feature and how it was calculated, if available
    description: str | None


class ChromosomeSchema(LanceModel):
    # The unique index of the chromosome, used for feature_indices in ChromatinAccessibilityRecord
    chromosome_index: int
    # The chromosome name, e.g., "chr1", "chr2", ..., "chrX", "chrY", "chrM"
    chromosome_name: str
    # The size of the chromosome in base pairs (from the reference genome assembly)
    chromosome_size: int
    # The organism, e.g., human, mouse, etc.
    organism: str
    # The genome assembly name, e.g., "GRCh38", "GRCm39"
    assembly: str


class ProteinSchema(LanceModel):
    # The unique index of the protein, used for feature_indices in ProteinAbundanceRecord
    protein_index: int
    # The UniProt accession ID, e.g., "P04637"
    uniprot_id: str
    # The recommended protein name from UniProt, e.g., "Cellular tumor antigen p53"
    protein_name: str
    # The primary gene name encoding this protein, e.g., "TP53"
    gene_name: str | None
    # The organism, e.g., "human", "mouse"
    organism: str
    # The amino acid sequence
    sequence: str
    # Length of the amino acid sequence
    sequence_length: int


class MoleculeSchema(LanceModel):
    # The smiles string for the molecule
    smiles: str | None
    # PubChem CID for the molecule
    pubchem_cid: int | None
    # Standard name for the molecule
    iupac_name: str | None

    # Auto-generated UID, this is generated automatically and should not be set
    sample_uid: str = Field(default_factory=lambda: str(uuid.uuid4()))


def _generate_perturbation_search_tokens(record: _CellMetadataMixin) -> str:
    """Build perturbation search tokens from a record's perturbation fields."""
    tokens: list[str] = []
    if record.chemical_perturbation_uid:
        tokens.extend([f"SM:{uid}" for uid in record.chemical_perturbation_uid if uid])
    if record.genetic_perturbation_gene_index:
        tokens.extend([f"GENE_ID:{gid}" for gid in record.genetic_perturbation_gene_index if gid])
    if record.genetic_perturbation_method:
        tokens.extend([f"METHOD:{m}" for m in record.genetic_perturbation_method if m])
    return " ".join(tokens)


FEATURE_SPACE_TO_SCHEMA: dict[str, LanceModel] = {
    "gene_expression": GeneExpressionRecord,
    "protein_abundance": ProteinAbundanceRecord,
    "chromatin_accessibility": ChromatinAccessibilityRecord,
    "image_features": ImageFeatureVectorRecord,
}
