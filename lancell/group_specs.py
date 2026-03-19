from enum import Enum

import zarr
from pydantic import BaseModel

from lancell.protocols import Reconstructor


# TODO: This seems totally unnecessary. We're just using the zarr dtype, I don't
# believe this is required anywhere aside from validation, but then we're not validating
# layer or assigning them a dtype, so it all seems a bit useless.
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


class LayersSpec(BaseModel):
    """Spec for the layers zarr subgroup."""

    prefix: str = ""
    uniform_shape: bool = False
    match_shape_of: str | None = None
    required: list[str] = []
    allowed: list[str] = []

    @property
    def path(self) -> str:
        return f"{self.prefix}/layers" if self.prefix else "layers"


class ZarrGroupSpec(BaseModel):
    """Declarative spec for the expected layout of a zarr group."""

    # Needed so Pydantic accepts the Reconstructor protocol type.
    model_config = {"arbitrary_types_allowed": True}

    feature_space: str
    pointer_kind: PointerKind
    reconstructor: Reconstructor
    has_var_df: bool = False
    required_arrays: list[ArraySpec] = []
    layers: LayersSpec = LayersSpec()

    def _check_top_level_arrays(self, group: zarr.Group) -> tuple[list[str], dict[str, tuple]]:
        errors: list[str] = []
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
        return errors, reference_shapes

    def find_layers_path(self) -> str:
        """Return the layers group path — may be top-level or nested (e.g. 'csr/layers')."""
        return self.layers.path

    def _check_layers(self, group: zarr.Group, reference_shapes: dict[str, tuple]) -> list[str]:
        errors: list[str] = []
        layers_path = self.layers.path

        try:
            layers_candidate = group[layers_path]
            layers_group: zarr.Group | None = (
                layers_candidate if isinstance(layers_candidate, zarr.Group) else None
            )
        except Exception:
            layers_group = None

        if self.layers.required:
            if layers_group is None:
                errors.append(
                    f"Missing required '{layers_path}' subgroup "
                    f"(required layers: {self.layers.required})"
                )
            else:
                for layer_name in self.layers.required:
                    if layer_name not in layers_group:
                        errors.append(f"Missing required layer '{layer_name}'")

        if self.layers.allowed and layers_group is not None:
            allowed_values = set(self.layers.allowed)
            for name, _ in layers_group.arrays():
                if name not in allowed_values:
                    errors.append(
                        f"Unknown layer '{name}' in {layers_path}/ subgroup. "
                        f"Allowed: {sorted(allowed_values)}"
                    )

        if layers_group is not None:
            sub_arrays = {k: v for k, v in layers_group.arrays()}

            if self.layers.uniform_shape and sub_arrays:
                shapes = {name: arr.shape for name, arr in sub_arrays.items()}
                if len(set(shapes.values())) > 1:
                    errors.append(f"'{layers_path}' arrays have inconsistent shapes: {shapes}")

            if self.layers.match_shape_of and self.layers.match_shape_of in reference_shapes:
                expected = reference_shapes[self.layers.match_shape_of]
                for name, arr in sub_arrays.items():
                    if arr.shape != expected:
                        errors.append(
                            f"'{layers_path}/{name}' shape {arr.shape} "
                            f"doesn't match '{self.layers.match_shape_of}' "
                            f"shape {expected}"
                        )

        return errors

    def validate_group(self, group: zarr.Group) -> list[str]:
        """Validate a zarr group against this spec. Returns a list of errors."""
        errors, reference_shapes = self._check_top_level_arrays(group)
        errors += self._check_layers(group, reference_shapes)
        return errors


# ---------------------------------------------------------------------------
# Spec registry
# ---------------------------------------------------------------------------

_SPEC_REGISTRY: dict[str, ZarrGroupSpec] = {}


def register_spec(spec: ZarrGroupSpec) -> None:
    """Register a new ZarrGroupSpec. Raises if already registered."""
    if spec.feature_space in _SPEC_REGISTRY:
        raise ValueError(f"Feature space '{spec.feature_space}' is already registered")
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
