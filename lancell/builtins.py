"""Built-in feature space specs and their reconstructors.

Imported at package init time to register all built-in specs before
user code defines schema subclasses or runs queries.
"""

from lancell.group_specs import (
    ArraySpec,
    DTypeKind,
    LayersSpec,
    PointerKind,
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
        ArraySpec(array_name="csr/indices", ndim=1, dtype_kind=DTypeKind.UNSIGNED_INTEGER),
    ],
    layers=LayersSpec(
        prefix="csr",
        uniform_shape=True,
        match_shape_of="csr/indices",
        required=["counts"],
        allowed=["counts", "log_normalized", "tpm"],
    ),
    reconstructor=SparseCSRReconstructor(),
)

IMAGE_FEATURES_SPEC = ZarrGroupSpec(
    feature_space="image_features",
    pointer_kind=PointerKind.DENSE,
    has_var_df=True,
    layers=LayersSpec(
        uniform_shape=True,
        required=["raw"],
        allowed=["raw", "log_normalized", "ctrl_standardized"],
    ),
    reconstructor=DenseReconstructor(),
)


for _spec in [
    GENE_EXPRESSION_SPEC,
    IMAGE_FEATURES_SPEC,
]:
    register_spec(_spec)
