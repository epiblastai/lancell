import datetime
import uuid
from types import UnionType
from typing import Union, get_args, get_origin

from lancedb.pydantic import LanceModel
from pydantic import Field, model_validator

from lancell.group_specs import ZARR_SPECS, FeatureSpace, PointerKind


class SparseZarrPointer(LanceModel):
    feature_space: str  # FeatureSpace value; stored as str for Arrow compat
    zarr_group: str
    start: int
    end: int

    @model_validator(mode="after")
    def _require_sparse_feature_space(self):
        fs = FeatureSpace(self.feature_space)
        spec = ZARR_SPECS[fs]
        if spec.pointer_kind is not PointerKind.SPARSE:
            raise ValueError(
                f"feature_space '{self.feature_space}' requires a "
                f"{spec.pointer_kind.value} pointer, not a sparse pointer"
            )
        return self


class DenseZarrPointer(LanceModel):
    feature_space: str  # FeatureSpace value; stored as str for Arrow compat
    zarr_group: str
    position: int

    @model_validator(mode="after")
    def _require_dense_feature_space(self):
        fs = FeatureSpace(self.feature_space)
        spec = ZARR_SPECS[fs]
        if spec.pointer_kind is not PointerKind.DENSE:
            raise ValueError(
                f"feature_space '{self.feature_space}' requires a "
                f"{spec.pointer_kind.value} pointer, not a dense pointer"
            )
        return self


# Placeholder for now, logic for loading hasn't been implemented yet
# Eventually we can use this for nifty things like RTree indexes
# class SpatialZarrPointer(LanceModel):
#     feature_space: FeatureSpace
#     zarr_group: str
#     bounding_box: list[float]  # [x_min, y_min, x_max, y_max]


ZarrPointer = SparseZarrPointer | DenseZarrPointer


class LancellBaseSchema(LanceModel):
    """
    Base schema for all lancell datasets. The only requirements are a uid string
    that allows for safe parallel-write scenarios, and at least one ZarrPointer
    into a feature space.
    """

    uid: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    dataset_uid: str = ""

    def __init_subclass__(cls, **kwargs):
        """Class-definition-time: subclass must declare at least one pointer field."""
        super().__init_subclass__(**kwargs)
        for name, annotation in cls.__annotations__.items():
            if name in ("uid", "dataset_uid"):
                continue
            origin = get_origin(annotation)
            if origin is Union or isinstance(annotation, UnionType):
                inner_types = get_args(annotation)
            else:
                inner_types = (annotation,)
            # TODO: The name of the field must be a value in `FeatureSpace`, because we
            # need this when parsing the schema to know which registry to look up in.
            if any(
                t is SparseZarrPointer or t is DenseZarrPointer
                for t in inner_types
                if t is not type(None)
            ):
                return  # found one, we're good
        raise TypeError(
            f"{cls.__name__} must declare at least one SparseZarrPointer or DenseZarrPointer field"
        )

    @model_validator(mode="after")
    def _require_at_least_one_pointer(self):
        """Instance-time: at least one pointer must be non-None."""
        for name in self.model_fields:
            if isinstance(getattr(self, name), SparseZarrPointer | DenseZarrPointer):
                return self
        raise ValueError(
            f"{type(self).__name__} requires at least one populated zarr pointer field"
        )


class FeatureBaseSchema(LanceModel):
    """
    Minimal schema for a global feature registry entry.

    Each feature space (e.g. genes, proteins) maintains its own registry where
    every row is a unique feature. Subclass this to add modality-specific fields.

    Fields:
        uid: Canonical stable identifier. Safe to preserve across registry rebuilds.
        global_index: Dense integer for compute paths (gather/scatter in NumPy,
            PyTorch, Arrow, Rust). Unique within one feature registry. May be
            reassigned on registry rebuild — use uid for durable references.
    """

    uid: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    global_index: int | None = None


class DatasetRecord(LanceModel):
    """Metadata for a single ingested dataset."""

    uid: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    zarr_group: str
    feature_space: str  # FeatureSpace value
    n_cells: int
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
