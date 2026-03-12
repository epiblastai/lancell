"""Built-in feature space specs and their reconstructors.

Imported at package init time to register all built-in specs before
user code defines schema subclasses or runs queries.
"""

from lancell.group_specs import (
    ArraySpec,
    DTypeKind,
    PointerKind,
    SubgroupSpec,
    ZarrGroupSpec,
    register_spec,
)
from lancell.reconstruction import DenseReconstructor, SparseCSRReconstructor

# ---------------------------------------------------------------------------
# Built-in specs
# ---------------------------------------------------------------------------

GENE_EXPRESSION_SPEC = ZarrGroupSpec(
    feature_space="gene_expression",
    pointer_kind=PointerKind.SPARSE,
    has_var_df=True,
    required_arrays=[
        ArraySpec(array_name="indices", ndim=1, dtype_kind=DTypeKind.UNSIGNED_INTEGER),
    ],
    required_subgroups=[
        SubgroupSpec(
            subgroup_name="layers",
            uniform_shape=True,
            match_shape_of="indices",
        ),
    ],
    required_layers=["counts"],
    allowed_layers=["counts", "log_normalized", "tpm"],
    reconstructor=SparseCSRReconstructor(),
)

CHROMATIN_PEAK_SPEC = ZarrGroupSpec(
    feature_space="chromatin_peak",
    pointer_kind=PointerKind.SPARSE,
    has_var_df=True,
    required_arrays=[
        ArraySpec(array_name="indices", ndim=1, dtype_kind=DTypeKind.UNSIGNED_INTEGER),
        ArraySpec(array_name="peak_starts", ndim=1, dtype_kind=DTypeKind.UNSIGNED_INTEGER),
        ArraySpec(array_name="peak_ends", ndim=1, dtype_kind=DTypeKind.UNSIGNED_INTEGER),
    ],
    required_layers=["counts"],
    allowed_layers=["counts", "log_normalized", "tpm"],
    reconstructor=SparseCSRReconstructor(),
)

PROTEIN_ABUNDANCE_SPEC = ZarrGroupSpec(
    feature_space="protein_abundance",
    pointer_kind=PointerKind.DENSE,
    has_var_df=True,
    required_subgroups=[
        SubgroupSpec(
            subgroup_name="layers",
            uniform_shape=True,
        ),
    ],
    required_layers=["counts"],
    allowed_layers=["counts", "clr", "dsb", "log_normalized"],
    reconstructor=DenseReconstructor(),
)

IMAGE_FEATURES_SPEC = ZarrGroupSpec(
    feature_space="image_features",
    pointer_kind=PointerKind.DENSE,
    has_var_df=True,
    required_subgroups=[
        SubgroupSpec(
            subgroup_name="layers",
            uniform_shape=True,
        ),
    ],
    required_layers=["raw"],
    allowed_layers=["raw", "log_normalized", "ctrl_standardized"],
    reconstructor=DenseReconstructor(),
)


for _spec in [
    GENE_EXPRESSION_SPEC,
    CHROMATIN_PEAK_SPEC,
    PROTEIN_ABUNDANCE_SPEC,
    IMAGE_FEATURES_SPEC,
]:
    register_spec(_spec)
