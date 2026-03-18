import datetime
import uuid
from types import UnionType
from typing import Union, get_args, get_origin

from lancedb.pydantic import LanceModel
from pydantic import Field, model_validator

from lancell.group_specs import PointerKind, get_spec, registered_feature_spaces


class SparseZarrPointer(LanceModel):
    feature_space: str  # FeatureSpace value; stored as str for Arrow compat
    zarr_group: str
    start: int
    end: int
    zarr_row: int  # cell's 0-indexed position within this zarr group (for CSC lookup)

    @model_validator(mode="after")
    def _require_sparse_feature_space(self):
        spec = get_spec(self.feature_space)
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
        spec = get_spec(self.feature_space)
        if spec.pointer_kind is not PointerKind.DENSE:
            raise ValueError(
                f"feature_space '{self.feature_space}' requires a "
                f"{spec.pointer_kind.value} pointer, not a dense pointer"
            )
        return self


ZarrPointer = SparseZarrPointer | DenseZarrPointer


def make_uid() -> str:
    """Generate a random 16-character hex uid."""
    return uuid.uuid4().hex[:16]


class LancellBaseSchema(LanceModel):
    """
    Base schema for all lancell datasets. The only requirements are a uid string
    that allows for safe parallel-write scenarios, and at least one ZarrPointer
    into a feature space.
    """

    uid: str = Field(default_factory=make_uid)
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
            if any(
                t is SparseZarrPointer or t is DenseZarrPointer
                for t in inner_types
                if t is not type(None)
            ):
                # Validate that the field name is a valid FeatureSpace value,
                # because we use the field name to look up the registry.
                valid_values = registered_feature_spaces()
                if name not in valid_values:
                    raise TypeError(
                        f"{cls.__name__}.{name}: pointer field name must match "
                        f"a registered feature space. Valid values: {sorted(valid_values)}"
                    )
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


# Fields set automatically by the atlas — never expected in user-provided obs.
AUTO_FIELDS: frozenset[str] = frozenset(LancellBaseSchema.model_fields)


class FeatureBaseSchema(LanceModel):
    """
    Minimal schema for a global feature registry entry.

    Each feature space (e.g. genes, proteins) maintains its own registry where
    every row is a unique feature. Subclass this to add modality-specific fields.

    Fields:
        uid: Canonical stable identifier. Safe to preserve across registry rebuilds.
        global_index: Unique stable integer, assigned incrementally (new features get
            max(existing) + 1). Used as a scatter/gather key in compute paths. Never
            reassigned once set — use uid for durable references.
    """

    uid: str = Field(default_factory=make_uid)
    global_index: int | None = None


class DatasetRecord(LanceModel):
    """Metadata for a single ingested dataset."""

    uid: str = Field(default_factory=make_uid)
    zarr_group: str
    feature_space: str  # FeatureSpace value
    n_cells: int
    # TODO: Layout UID is updated automatically by add_or_reuse_layout. If a user forgets
    # to call that method during ingestion, this will break. add_or_reuse_layout should
    # probably be called automatically somewhere to avoid mistakes.
    layout_uid: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


class FeatureLayout(LanceModel):
    """Per-feature-per-layout index row for the ``_feature_layouts`` table.

    Each row maps a single feature within a unique feature ordering (layout)
    to its local position and its global position in the feature registry.
    Multiple datasets sharing the same feature ordering reference the same
    layout_uid, dramatically reducing row count.

    Parameters
    ----------
    layout_uid:
        Content-hash of the ordered feature list (shared across datasets
        with identical feature orderings).
    feature_uid:
        Global feature UID (FTS indexed for feature-to-layout lookups).
    local_index:
        0-based position of the feature within this layout.
    global_index:
        Position in the global feature registry. Denormalized from the registry and
        kept in sync by ``sync_layouts_global_index``.
    """

    layout_uid: str
    feature_uid: str
    local_index: int
    global_index: int | None = None


class AtlasVersionRecord(LanceModel):
    """One row per atlas snapshot created by ``RaggedAtlas.snapshot()``.

    Captures the Lance table versions for every table in the atlas at the
    time of the snapshot, enabling reproducible point-in-time queries via
    ``RaggedAtlas.checkout(version)``.

    Parameters
    ----------
    version:
        Monotonically increasing snapshot version number.
    cell_table_name:
        Name of the cells Lance table.
    cell_table_version:
        Lance version of the cells table at snapshot time.
    dataset_table_name:
        Name of the datasets Lance table.
    dataset_table_version:
        Lance version of the datasets table at snapshot time.
    registry_table_names:
        JSON-encoded mapping of ``{feature_space: table_name}`` for feature registries.
    registry_table_versions:
        JSON-encoded mapping of ``{feature_space: version_int}`` for feature registries.
    feature_layouts_table_version:
        Lance version of the ``_feature_layouts`` table at snapshot time.
    total_cells:
        Total number of cells across all datasets at snapshot time.
    created_at:
        ISO-8601 UTC timestamp of when the snapshot was created.
    """

    version: int
    cell_table_name: str
    cell_table_version: int
    dataset_table_name: str
    dataset_table_version: int
    registry_table_names: str
    registry_table_versions: str
    feature_layouts_table_version: int
    total_cells: int
    zarr_store_uri: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
