"""Schemas for an scBaseCount atlas built on lancell's ragged atlas framework.

Defines:
- GENEFULL_EXPRESSION_SPEC: custom zarr group spec for Unique/EM/Uniform layers
- GeneFeatureSpace: feature registry schema for genes (var metadata)
- ScBasecountDatasetRecord: dataset-level metadata with scBaseCount provenance
- CellObs: cell-level observation schema with genefull_expression pointer
"""

from lancell.group_specs import (
    ArraySpec,
    DTypeKind,
    LayersSpec,
    PointerKind,
    ZarrGroupSpec,
    register_spec,
)
from lancell.reconstruction import SparseCSRReconstructor
from lancell.schema import (
    DatasetRecord,
    FeatureBaseSchema,
    LancellBaseSchema,
    SparseZarrPointer,
)

# ---------------------------------------------------------------------------
# Custom feature space spec for GeneFull_Ex50pAS layers
# ---------------------------------------------------------------------------

GENEFULL_EXPRESSION_SPEC = ZarrGroupSpec(
    feature_space="genefull_expression",
    pointer_kind=PointerKind.SPARSE,
    has_var_df=True,
    required_arrays=[
        ArraySpec(array_name="csr/indices", ndim=1, dtype_kind=DTypeKind.UNSIGNED_INTEGER),
    ],
    layers=LayersSpec(
        prefix="csr",
        uniform_shape=True,
        match_shape_of="csr/indices",
        required=["Unique"],
        allowed=["Unique", "UniqueAndMult-EM", "UniqueAndMult-Uniform"],
    ),
    reconstructor=SparseCSRReconstructor(),
)
register_spec(GENEFULL_EXPRESSION_SPEC)


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------


class GeneFeatureSpace(FeatureBaseSchema):
    """Gene feature registry entry for scBaseCount data."""

    gene_id: str
    gene_name: str
    organism: str


# ---------------------------------------------------------------------------
# Dataset record
# ---------------------------------------------------------------------------


class ScBasecountDatasetRecord(DatasetRecord):
    """Dataset record with scBaseCount provenance."""

    srx_accession: str
    feature_type: str = "GeneFull_Ex50pAS"
    release_date: str = "2026-01-12"
    lib_prep: str | None = None
    tech_10x: str | None = None
    cell_prep: str | None = None
    organism: str | None = None
    tissue: str | None = None
    tissue_ontology_term_id: str | None = None
    disease: str | None = None
    disease_ontology_term_id: str | None = None
    perturbation: str | None = None
    cell_line: str | None = None
    antibody_derived_tag: str | None = None
    czi_collection_id: str | None = None
    czi_collection_name: str | None = None


# ---------------------------------------------------------------------------
# Cell observation schema
# ---------------------------------------------------------------------------


class CellObs(LancellBaseSchema):
    """Cell-level observation schema for scBaseCount data."""

    genefull_expression: SparseZarrPointer | None = None

    cell_barcode: str | None = None
    srx_accession: str | None = None
    gene_count_unique: int | None = None
    umi_count_unique: int | None = None
    cell_type: str | None = None
    cell_ontology_term_id: str | None = None
