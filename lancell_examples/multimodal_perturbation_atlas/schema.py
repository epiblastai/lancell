import hashlib
from datetime import datetime
from enum import Enum
from typing import Self

from lancedb.pydantic import LanceModel
from pydantic import Field, model_validator

from lancell.schema import (
    DatasetRecord,
    DenseZarrPointer,
    FeatureBaseSchema,
    LancellBaseSchema,
    SparseZarrPointer,
    make_uid,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FeatureSpace(str, Enum):
    GENE_EXPRESSION = "gene_expression"
    PROTEIN_ABUNDANCE = "protein_abundance"
    CHROMATIN_ACCESSIBILITY = "chromatin_accessibility"
    IMAGE_FEATURES = "image_features"
    IMAGE_TILES = "image_tiles"


class FeatureType(str, Enum):
    """The level of resolution a genomic feature represents."""

    GENE = "gene"
    TRANSCRIPT = "transcript"
    EXON = "exon"
    CRYPTIC_EXON = "cryptic_exon"
    PROBE = "probe"
    OTHER = "other"


class SequenceRole(str, Enum):
    """The role of a sequence in a reference genome assembly."""

    CHROMOSOME = "chromosome"
    MITOCHONDRIAL = "mitochondrial"
    SCAFFOLD = "scaffold"
    UNLOCALIZED = "unlocalized"
    ALT_LOCUS = "alt_locus"
    PATCH = "patch"
    DECOY = "decoy"
    VIRAL = "viral"
    OTHER = "other"


class GeneticPerturbationType(str, Enum):
    """The class of genetic perturbation reagent."""

    CRISPR_KO = "CRISPRko"
    CRISPR_I = "CRISPRi"
    CRISPR_A = "CRISPRa"
    SI_RNA = "siRNA"
    SH_RNA = "shRNA"
    ASO = "ASO"
    OVEREXPRESSION = "overexpression"
    OTHER = "other"


class TargetContext(str, Enum):
    """Where a genetic perturbation reagent lands relative to gene structure."""

    EXON = "exon"
    INTRON = "intron"
    PROMOTER = "promoter"
    ENHANCER = "enhancer"
    UTR_5 = "5_UTR"
    UTR_3 = "3_UTR"
    INTERGENIC = "intergenic"
    OTHER = "other"


class BiologicPerturbationType(str, Enum):
    """The class of biologic perturbation agent."""

    CYTOKINE = "cytokine"
    GROWTH_FACTOR = "growth_factor"
    ANTIBODY = "antibody"
    LIGAND = "ligand"
    RECEPTOR_AGONIST = "receptor_agonist"
    RECEPTOR_ANTAGONIST = "receptor_antagonist"
    OTHER = "other"


class PerturbationType(str, Enum):
    SMALL_MOLECULE = "small_molecule"
    GENETIC_PERTURBATION = "genetic_perturbation"
    BIOLOGIC_PERTURBATION = "biologic_perturbation"


# ---------------------------------------------------------------------------
# Publications
# ---------------------------------------------------------------------------


class PublicationSchema(LanceModel):
    # Primary key
    uid: str = Field(default_factory=make_uid)

    # The doi for the paper, there is almost always one
    doi: str
    # PubMed id for the paper, there is almost always one
    pmid: int | None
    # The title of the paper
    title: str
    # The journal that the paper was published in, if applicable
    journal: str | None
    # The year that the paper was published, if applicable
    publication_date: datetime | None


class PublicationSectionSchema(LanceModel):
    publication_uid: str  # Foreign key to PublicationSchema.uid

    # Section-level fields (one row per section)
    # The text content of this section
    section_text: str
    # The heading / title of the section, e.g. "Abstract", "Introduction",
    # "Methods", "Results", "Discussion", "References", etc.
    section_title: str | None


# ---------------------------------------------------------------------------
# Datasets & donors
# ---------------------------------------------------------------------------


class DatasetSchema(DatasetRecord):
    # Foreign key: The uid for an associated publication
    publication_uid: str | None
    # Database from which the dataset was downloaded, if applicable
    accession_database: str | None
    accession_id: str | None
    # Dataset description, for example the sample preparation and experimental
    # protocol text from the GEO series or sample record.
    dataset_description: str | None

    # High-level metadata fields that are useful for searching and grouping datasets.
    organism: list[str] | None  # ["human", "mouse"] for barnyard
    tissue: list[str] | None  # ["cortex", "hippocampus"] for multi-region
    cell_line: list[str] | None  # ["A549", "MCF7", "K562"] for village-in-a-dish
    disease: list[str] | None  # ["ALS", "healthy"] for case-control


class DonorSchema(LanceModel):
    # Primary key
    uid: str = Field(default_factory=make_uid)
    age_years: float | None = None
    sex: str | None = None
    ethnicity: str | None = None
    cause_of_death: str | None = None  # for postmortem tissue
    pmi_hours: float | None = None  # postmortem interval in hours
    clinical_diagnosis: str | None = None
    pathological_diagnosis: str | None = None

    # Free-text notes about the donor
    description: str | None = None


# ---------------------------------------------------------------------------
# Feature registries (var tables)
# ---------------------------------------------------------------------------


class GenomicFeatureSchema(FeatureBaseSchema):
    """A single measurable genomic feature in a dataset's var space.

    This schema is designed to serve as a feature registry across datasets
    that may operate at different levels of resolution (gene, transcript,
    isoform, etc.). Within a dataset, `feature_index` is the positional
    index into the expression matrix. Across datasets, `ensembl_gene_id`
    and `feature_type` enable joins and roll-ups to a shared feature space.

    Multiple rows may share the same `ensembl_gene_id` when a dataset
    contains sub-gene resolution features (e.g., isoforms). To collapse
    back to gene-level, group by `ensembl_gene_id` and aggregate (e.g., sum).
    """

    uid: str = Field(default_factory=make_uid)

    # The canonical gene this feature maps to, if applicable
    gene_name: str | None
    ensembl_gene_id: str | None

    # The specific feature identity.
    # For gene-level features this equals ensembl_gene_id.
    # For transcripts this would be e.g. ENST00000269305.
    feature_id: str

    # What level of resolution this feature represents
    feature_type: FeatureType

    # For transcript/isoform-level features, e.g. ENST00000269305.7
    transcript_id: str | None = None

    # Free-text or controlled vocabulary for edge cases,
    # e.g. "STMN2 cryptic exon", "UNC13A cryptic exon"
    feature_annotation: str | None = None

    # The version of Ensembl used for annotations.
    # Only set if known, otherwise leave as None.
    ensembl_version: str | None = None

    # The organism this feature belongs to, e.g. "human", "mouse"
    organism: str


class ReferenceSequenceSchema(FeatureBaseSchema):
    """A single contig or sequence in a reference genome assembly.
    Intended as the feature table for chromatin accessibility peaks.

    Covers chromosomes as well as non-chromosomal sequences commonly
    found in reference genomes: unplaced/unlocalized scaffolds, alt loci,
    patches, decoys, and viral sequences (e.g., EBV in GRCh38).
    """

    # The sequence name as used in alignment, e.g. "chr1", "chrUn_GL000220v1",
    # "chr6_GL000256v2_alt", "chrEBV"
    sequence_name: str

    start: int
    end: int

    # The role this sequence plays in the assembly
    sequence_role: SequenceRole

    # The organism, e.g. "human", "mouse"
    organism: str
    # The genome assembly name, e.g. "GRCh38", "GRCm39"
    assembly: str

    # Unambiguous accession — stable across naming conventions
    # (e.g. "CM000663.2" for chr1 in GRCh38)
    genbank_accession: str | None = None
    refseq_accession: str | None = None

    # Whether this sequence is part of the primary assembly,
    # i.e. the set of sequences most analyses restrict to
    is_primary_assembly: bool = True


class ProteinSchema(FeatureBaseSchema):
    # The UniProt accession ID, e.g., "P04637"
    uniprot_id: str | None
    # The recommended protein name from UniProt, e.g., "Cellular tumor antigen p53"
    protein_name: str | None
    # The primary gene name encoding this protein, e.g., "TP53"
    gene_name: str | None
    # The organism, e.g., "human", "mouse"
    organism: str | None
    # The amino acid sequence
    sequence: str | None
    # Length of the amino acid sequence
    sequence_length: int | None


class ImageFeatureSchema(FeatureBaseSchema):
    # The name of the image feature, e.g., "mean_intensity_DAPI", "texture_feature_1", etc.
    feature_name: str
    # A description of the image feature and how it was calculated, if available
    description: str | None


# ---------------------------------------------------------------------------
# Perturbation registries
# ---------------------------------------------------------------------------


class SmallMoleculeSchema(LanceModel):
    """Small molecule data, either perturbations or features in themselves."""

    # Primary key
    uid: str = Field(default_factory=make_uid)

    # The smiles string for the molecule
    smiles: str | None
    # PubChem CID for the molecule
    pubchem_cid: int | None
    # Standard name for the molecule
    iupac_name: str | None
    inchi_key: str | None
    chembl_id: str | None
    # Common name for the molecule
    name: str | None

    # Provenance
    vendor: str | None = None
    catalog_number: str | None = None

    @model_validator(mode="after")
    def validate_identifiers(self) -> Self:
        if not any([self.smiles, self.pubchem_cid, self.iupac_name, self.name]):
            raise ValueError(
                "At least one identifier (smiles, pubchem_cid, iupac_name, name) must be provided"
            )
        return self


class GeneticPerturbationSchema(LanceModel):
    """A single genetic perturbation reagent and its genomic target.

    Perturbations are anchored to genomic coordinates rather than gene
    names, because the relationship between a reagent and a gene is an
    annotation (reflecting design intent), not ground truth. Storing
    coordinates allows re-annotation against updated gene models,
    liftover to other assemblies, and correct handling of cases where
    a single reagent affects multiple genes (e.g., enhancer-targeting
    screens).

    The assignment of perturbations to cells (obs) is a separate
    relationship and should not be stored here.
    """

    uid: str = Field(default_factory=make_uid)

    # Reagent type
    perturbation_type: GeneticPerturbationType

    # The actual reagent sequence, e.g. the 20bp guide or siRNA duplex
    guide_sequence: str | None = None

    # Foreign key to ReferenceSequenceSchema.uid, if applicable
    target_sequence_uid: str | None = None
    # Genomic target coordinates — where the reagent physically acts
    target_start: int | None = None
    target_end: int | None = None
    target_strand: str | None = None  # "+" or "-"

    # The intended gene target — this is annotation, not ground truth.
    # A guide near a promoter "targets" a gene by convention, but a guide
    # in an enhancer might affect multiple genes.
    intended_gene_name: str | None = None
    intended_ensembl_gene_id: str | None = None

    # Where the guide lands relative to gene structure
    target_context: TargetContext | None = None

    # Reagent provenance
    library_name: str | None = None  # e.g. "Brunello", "CROPseq"
    reagent_id: str | None = None  # e.g. "BRD_KO_1", "CROPseq_A1"


class BiologicPerturbationSchema(LanceModel):
    """A biologic agent (protein, cytokine, antibody, etc.) applied to cells.

    Biologic perturbations are identified by the agent's name and, where
    possible, a UniProt accession for the protein involved.
    """

    uid: str = Field(default_factory=make_uid)

    # Biologic identity
    biologic_name: str
    biologic_type: BiologicPerturbationType

    # Protein identity from ProteinSchema.uid, if applicable
    protein_uid: str | None = None

    # Provenance
    vendor: str | None = None
    catalog_number: str | None = None
    lot_number: str | None = None


# ---------------------------------------------------------------------------
# Cell index (obs table)
# ---------------------------------------------------------------------------


class CellIndex(LancellBaseSchema):
    # Assay used like Perturb-seq, Cell Painting, snATAC-seq, Drop-seq, etc.
    # TODO: Validate this against a controlled vocabulary, EFO
    assay: str
    # The organism that the cells in this sample come from, e.g. human, mouse, etc.
    organism: str
    # Cell line used, e.g. A549, HeLa, etc. (if applicable), this is distinct from cell type
    cell_line: str | None
    # Annotated cell type, does not apply to immortalized cell lines or iPSC-derived cells
    # Generally should only be used for primary cells or well-annotated cell lines like PBMCs
    cell_type: str | None
    # Development stage, disease, and tissue only apply to primary cells. For example, `disease`
    # should be null even for a "cancer cell line".
    development_stage: str | None
    disease: str | None
    tissue: str | None
    donor_uid: str | None  # Foreign key to a DonorSchema.uid if available
    # Number of days the cells were cultured in vitro before profiling, if applicable.
    days_in_vitro: float | None
    # Json dump string with additional metadata that doesn't fit in the schema
    additional_metadata: str | None

    # Batch information
    replicate: int | None
    batch_id: str | None
    well_position: str | None

    # Perturbation-specific columns
    # Whether this cell is a negative control
    is_negative_control: bool | None
    # If it is a control, what kind? For genetic perturbations with might be `nontargeting`
    # or `intergenic`, as in the guide RNA type. For a small molecule it might be `DMSO` or
    # `vehicle`.
    negative_control_type: str | None

    # Cumulative lists of all the perturbations effected on a cell. Could be
    # combinatorial CRISPR guides, or a small molecule and a CRISPR guide, or
    # any other such combination. Lists must have exactly matching lengths
    # UIDs and types go together to specify foreign keys. The uid is a foreign
    # key value and the perturbation type determines which table it is a key in.
    perturbation_uids: list[str] | None
    perturbation_types: list[PerturbationType] | None
    # Concentrations for the perturbation in micromolar, if applicable, else use -1
    # to keep the lists equally long
    perturbation_concentrations_um: list[float] | None
    # Time durations for the perturbation in hours, if applicable, else use -1
    perturbation_durations_hr: list[float] | None
    # List of json dump with additional metadata for each perturbation
    perturbation_additional_metadata: list[str] | None

    # Pointers for each of the feature spaces. These all have a corresponding
    # feature registry table
    gene_expression: SparseZarrPointer | None = None  # GenomicFeatureSchema
    chromatin_accessibility: SparseZarrPointer | None = None  # ReferenceSequenceSchema
    protein_abundance: DenseZarrPointer | None = None  # ProteinSchema
    image_features: DenseZarrPointer | None = None  # ImageFeatureSchema

    # Image tiles don't have a schema because they aren't features!
    # TODO: For image data we might want to define a concept like "axis annotations"
    # that are alternatives to the feature registry. Here for example, the axis annotations
    # would be channel names probably.
    image_tiles: DenseZarrPointer | None = None

    # Auto-filled field
    perturbation_search_string: str = ""

    @model_validator(mode="after")
    def validate_perturbation_lists(self) -> Self:
        lists = [
            self.perturbation_uids,
            self.perturbation_types,
            self.perturbation_concentrations_um,
            self.perturbation_durations_hr,
            self.perturbation_additional_metadata,
        ]
        non_none = [lst for lst in lists if lst is not None]
        if non_none and len(set(len(lst) for lst in non_none)) > 1:
            raise ValueError("All perturbation lists must have the same length")
        return self

    @model_validator(mode="after")
    def generate_search_string(self) -> Self:
        self.perturbation_search_string = self.generate_perturbation_search_tokens(self)
        return self

    @staticmethod
    def generate_perturbation_search_tokens(record: Self) -> str:
        """Build perturbation search tokens from a record's perturbation fields."""
        tokens: list[str] = []
        for uid, ptype in zip(
            record.perturbation_uids or [], record.perturbation_types or [], strict=False
        ):
            if ptype == PerturbationType.SMALL_MOLECULE:
                tokens.append(f"SM:{uid}")
            elif ptype == PerturbationType.GENETIC_PERTURBATION:
                tokens.append(f"GP:{uid}")
            elif ptype == PerturbationType.BIOLOGIC_PERTURBATION:
                tokens.append(f"BIO:{uid}")
        return " ".join(tokens)


# ---------------------------------------------------------------------------
# Dataset-perturbation index (materialized summary)
# ---------------------------------------------------------------------------


class DatasetPerturbationIndex(LanceModel):
    """Materialized summary linking datasets to their perturbations.

    Built at ingestion time. Enables queries like 'find all datasets
    where TP53 was perturbed' without scanning CellIndex.
    """

    dataset_uid: str
    perturbation_uid: str
    perturbation_type: PerturbationType

    # Denormalized for search convenience — avoids a join to the
    # perturbation tables for the most common query patterns
    intended_gene_name: str | None = None  # for genetic
    compound_name: str | None = None  # for small molecule
    agent_name: str | None = None  # for biologic

    # Summary stats
    cell_count: int | None = None  # how many cells got this perturbation
    control_cell_count: int | None = None  # how many matched controls

    # Autofilled
    uid: str = ""

    @model_validator(mode="after")
    def generate_uid(self) -> Self:
        # Deterministic short hash of dataset and perturbation uids
        key = f"{self.dataset_uid}_{self.perturbation_uid}".encode()
        self.uid = hashlib.blake2b(key, digest_size=8).hexdigest()
        return self
