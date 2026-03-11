from enum import Enum

import zarr
from pydantic import BaseModel


class FeatureSpace(str, Enum):
    GENE_EXPRESSION = "gene_expression"
    PROTEIN_ABUNDANCE = "protein_abundance"
    CHROMATIN_FRAGMENT = "chromatin_fragment"
    CHROMATIN_PEAK = "chromatin_peak"
    IMAGE_FEATURES = "image_features"
    IMAGE_TILES = "image_tiles"


class DTypeKind(str, Enum):
    """Structured dtype.kind values used by NumPy and Zarr arrays."""

    BOOL = "b"
    SIGNED_INTEGER = "i"
    UNSIGNED_INTEGER = "u"
    FLOAT = "f"


class PointerKind(str, Enum):
    SPARSE = "sparse"
    DENSE = "dense"


class ArraySpec(BaseModel):
    """Expected properties of a single zarr array."""

    array_name: str
    dtype_kind: DTypeKind | None = None
    ndim: int | None = None


class SubgroupSpec(BaseModel):
    """Expected properties of a zarr subgroup with multiple arrays."""

    subgroup_name: str
    # If None, any arrays are allowed. If provided, these are the
    # minimum required arrays.
    required_arrays: list[ArraySpec] | None = None
    # All arrays in this subgroup must share the same shape
    uniform_shape: bool = False
    # All arrays must match shape of a named sibling array
    # at the parent group level (e.g. "indices")
    match_shape_of: str | None = None


class ZarrGroupSpec(BaseModel):
    """Declarative spec for the expected layout of a zarr group."""

    feature_space: FeatureSpace
    pointer_kind: PointerKind
    has_var_df: bool = False
    required_arrays: list[ArraySpec] = []
    required_subgroups: list[SubgroupSpec] = []

    def validate_group(self, group: zarr.Group) -> list[str]:
        """Validate a zarr group against this spec. Returns a list of errors."""
        errors: list[str] = []

        # Check required top-level arrays
        reference_shapes: dict[str, tuple] = {}
        for array_spec in self.required_arrays:
            if array_spec.array_name not in group:
                errors.append(f"Missing required array '{array_spec.array_name}'")
                continue
            arr = group[array_spec.array_name]
            if not isinstance(arr, zarr.Array):
                errors.append(f"'{array_spec.array_name}' is not an array")
                continue
            reference_shapes[array_spec.array_name] = arr.shape
            if array_spec.ndim is not None and arr.ndim != array_spec.ndim:
                errors.append(
                    f"'{array_spec.array_name}' has ndim={arr.ndim}, expected {array_spec.ndim}"
                )
            if array_spec.dtype_kind is not None and arr.dtype.kind != array_spec.dtype_kind.value:
                errors.append(
                    f"'{array_spec.array_name}' has dtype.kind='{arr.dtype.kind}', "
                    f"expected '{array_spec.dtype_kind.value}'"
                )

        # Check required subgroups
        for sg_spec in self.required_subgroups:
            if sg_spec.subgroup_name not in group:
                errors.append(f"Missing required subgroup '{sg_spec.subgroup_name}'")
                continue
            subgroup = group[sg_spec.subgroup_name]
            if not isinstance(subgroup, zarr.Group):
                errors.append(f"'{sg_spec.subgroup_name}' is not a group")
                continue

            # Check required arrays within subgroup
            sub_arrays = {k: v for k, v in subgroup.arrays()}
            if sg_spec.required_arrays:
                for arr_spec in sg_spec.required_arrays:
                    if arr_spec.array_name not in sub_arrays:
                        errors.append(
                            f"Subgroup '{sg_spec.subgroup_name}' missing "
                            f"required array '{arr_spec.array_name}'"
                        )

            # Uniform shape check across all arrays in subgroup
            if sg_spec.uniform_shape and sub_arrays:
                shapes = {name: arr.shape for name, arr in sub_arrays.items()}
                unique_shapes = set(shapes.values())
                if len(unique_shapes) > 1:
                    errors.append(
                        f"Subgroup '{sg_spec.subgroup_name}' arrays have "
                        f"inconsistent shapes: {shapes}"
                    )

            # Shape must match a sibling array at parent level
            if sg_spec.match_shape_of and sg_spec.match_shape_of in reference_shapes:
                expected = reference_shapes[sg_spec.match_shape_of]
                for name, arr in sub_arrays.items():
                    if arr.shape != expected:
                        errors.append(
                            f"'{sg_spec.subgroup_name}/{name}' shape {arr.shape} "
                            f"doesn't match '{sg_spec.match_shape_of}' "
                            f"shape {expected}"
                        )

        return errors


GENE_EXPRESSION_SPEC = ZarrGroupSpec(
    feature_space=FeatureSpace.GENE_EXPRESSION,
    pointer_kind=PointerKind.SPARSE,
    has_var_df=True,
    required_arrays=[
        ArraySpec(array_name="indices", ndim=1, dtype_kind=DTypeKind.UNSIGNED_INTEGER),
    ],
    required_subgroups=[
        SubgroupSpec(
            subgroup_name="layers",
            required_arrays=[
                ArraySpec(array_name="counts", ndim=1, dtype_kind=DTypeKind.UNSIGNED_INTEGER),
            ],
            uniform_shape=True,
            match_shape_of="indices",
        ),
    ],
)

CHROMATIN_FRAGMENT_SPEC = ZarrGroupSpec(
    feature_space=FeatureSpace.CHROMATIN_FRAGMENT,
    pointer_kind=PointerKind.SPARSE,
    required_arrays=[
        ArraySpec(array_name="fragment_starts", ndim=1, dtype_kind=DTypeKind.UNSIGNED_INTEGER),
        ArraySpec(array_name="fragment_ends", ndim=1, dtype_kind=DTypeKind.UNSIGNED_INTEGER),
    ],
)


CHROMATIN_PEAK_SPEC = ZarrGroupSpec(
    feature_space=FeatureSpace.CHROMATIN_PEAK,
    pointer_kind=PointerKind.SPARSE,
    has_var_df=True,
    required_arrays=[
        ArraySpec(array_name="peak_starts", ndim=1, dtype_kind=DTypeKind.UNSIGNED_INTEGER),
        ArraySpec(array_name="peak_ends", ndim=1, dtype_kind=DTypeKind.UNSIGNED_INTEGER),
    ],
)

PROTEIN_ABUNDANCE_SPEC = ZarrGroupSpec(
    feature_space=FeatureSpace.PROTEIN_ABUNDANCE,
    pointer_kind=PointerKind.DENSE,
    has_var_df=True,
    required_arrays=[
        ArraySpec(array_name="data", ndim=2, dtype_kind=DTypeKind.FLOAT),
    ],
)

IMAGE_FEATURES_SPEC = ZarrGroupSpec(
    feature_space=FeatureSpace.IMAGE_FEATURES,
    pointer_kind=PointerKind.DENSE,
    has_var_df=True,
    required_arrays=[
        ArraySpec(array_name="data", ndim=2, dtype_kind=DTypeKind.FLOAT),
    ],
)

IMAGE_TILES_SPEC = ZarrGroupSpec(
    feature_space=FeatureSpace.IMAGE_TILES,
    pointer_kind=PointerKind.DENSE,
    required_arrays=[
        # Tile can be any dtype
        ArraySpec(array_name="data", ndim=4),  # (N, C, H, W)
    ],
)

# Registry for lookup by feature space
ZARR_SPECS: dict[FeatureSpace, ZarrGroupSpec] = {
    spec.feature_space: spec
    for spec in [
        GENE_EXPRESSION_SPEC,
        CHROMATIN_FRAGMENT_SPEC,
        CHROMATIN_PEAK_SPEC,
        PROTEIN_ABUNDANCE_SPEC,
        IMAGE_FEATURES_SPEC,
        IMAGE_TILES_SPEC,
    ]
}
