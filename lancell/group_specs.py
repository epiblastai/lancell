from enum import Enum

import zarr
from pydantic import BaseModel

from lancell.protocols import Reconstructor


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

    # Needed so Pydantic accepts the Reconstructor protocol type.
    model_config = {"arbitrary_types_allowed": True}

    feature_space: str
    pointer_kind: PointerKind
    reconstructor: Reconstructor
    has_var_df: bool = False
    required_arrays: list[ArraySpec] = []
    required_subgroups: list[SubgroupSpec] = []
    required_layers: list[str] = []
    allowed_layers: list[str] = []

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

        # Check required layers
        if self.required_layers:
            if "layers" not in group or not isinstance(group["layers"], zarr.Group):
                errors.append(
                    f"Missing required 'layers' subgroup "
                    f"(required layers: {self.required_layers})"
                )
            else:
                layers_group = group["layers"]
                for layer_name in self.required_layers:
                    if layer_name not in layers_group:
                        errors.append(f"Missing required layer '{layer_name}'")

        # Check allowed layers (flag unknown arrays in layers/)
        if self.allowed_layers and "layers" in group and isinstance(group["layers"], zarr.Group):
            allowed_values = set(self.allowed_layers)
            for name, _ in group["layers"].arrays():
                if name not in allowed_values:
                    errors.append(
                        f"Unknown layer '{name}' in layers/ subgroup. "
                        f"Allowed: {sorted(allowed_values)}"
                    )

        return errors


# ---------------------------------------------------------------------------
# Spec registry
# ---------------------------------------------------------------------------

_SPEC_REGISTRY: dict[str, ZarrGroupSpec] = {}


def register_spec(spec: ZarrGroupSpec) -> None:
    """Register a new ZarrGroupSpec. Raises if already registered."""
    if spec.feature_space in _SPEC_REGISTRY:
        raise ValueError(
            f"Feature space '{spec.feature_space}' is already registered"
        )
    _SPEC_REGISTRY[spec.feature_space] = spec


def get_spec(feature_space: str) -> ZarrGroupSpec:
    """Look up a spec by feature space name."""
    if feature_space not in _SPEC_REGISTRY:
        raise KeyError(
            f"No spec registered for feature space '{feature_space}'. "
            f"Registered: {sorted(_SPEC_REGISTRY.keys())}"
        )
    return _SPEC_REGISTRY[feature_space]


def registered_feature_spaces() -> set[str]:
    """Return the set of all registered feature space names."""
    return set(_SPEC_REGISTRY.keys())
