"""Feature space specs for the multimodal perturbation atlas.

Registers specs for protein_abundance, chromatin_accessibility, and
image_tiles. Gene expression and image features reuse the built-in specs
from lancell.builtins.
"""

from lancell.group_specs import (
    ArraySpec,
    DTypeKind,
    LayersSpec,
    PointerKind,
    ZarrGroupSpec,
    register_spec,
)
from lancell.reconstruction import DenseReconstructor
from lancell_examples.multimodal_perturbation_atlas.interval_reconstruction import (
    IntervalReconstructor,
)

# ---------------------------------------------------------------------------
# Protein abundance (CITE-seq / ADT)
# ---------------------------------------------------------------------------

PROTEIN_ABUNDANCE_SPEC = ZarrGroupSpec(
    feature_space="protein_abundance",
    pointer_kind=PointerKind.DENSE,
    has_var_df=True,
    layers=LayersSpec(
        uniform_shape=True,
        required=["counts"],
        allowed=["counts", "clr_normalized", "dsb_normalized"],
    ),
    reconstructor=DenseReconstructor(),
)

# ---------------------------------------------------------------------------
# Chromatin accessibility (cell-sorted fragments)
# ---------------------------------------------------------------------------

CHROMATIN_ACCESSIBILITY_SPEC = ZarrGroupSpec(
    feature_space="chromatin_accessibility",
    pointer_kind=PointerKind.SPARSE,
    has_var_df=True,
    required_arrays=[
        ArraySpec(
            array_name="cell_sorted/chromosomes", ndim=1, dtype_kind=DTypeKind.UNSIGNED_INTEGER
        ),
        ArraySpec(array_name="cell_sorted/starts", ndim=1, dtype_kind=DTypeKind.UNSIGNED_INTEGER),
        ArraySpec(array_name="cell_sorted/lengths", ndim=1, dtype_kind=DTypeKind.UNSIGNED_INTEGER),
    ],
    layers=LayersSpec(),
    reconstructor=IntervalReconstructor(),
)

# ---------------------------------------------------------------------------
# Image tiles
# ---------------------------------------------------------------------------

IMAGE_TILES_SPEC = ZarrGroupSpec(
    feature_space="image_tiles",
    pointer_kind=PointerKind.DENSE,
    has_var_df=False,
    required_arrays=[
        ArraySpec(array_name="data", ndim=4, dtype_kind=DTypeKind.UNSIGNED_INTEGER),
    ],
    reconstructor=DenseReconstructor(),
)

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

for _spec in [
    PROTEIN_ABUNDANCE_SPEC,
    CHROMATIN_ACCESSIBILITY_SPEC,
    IMAGE_TILES_SPEC,
]:
    register_spec(_spec)
