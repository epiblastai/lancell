"""Schema introspection and obs-alignment utilities.

Used at both ingestion time and atlas query time to validate and align
AnnData obs DataFrames against a LancellBaseSchema.
"""

import dataclasses
from types import UnionType
from typing import Union, get_args, get_origin

import anndata as ad
import pandas as pd
import pyarrow as pa

from lancell.group_specs import PointerKind, get_spec
from lancell.schema import AUTO_FIELDS, DenseZarrPointer, LancellBaseSchema, SparseZarrPointer

# ---------------------------------------------------------------------------
# PointerFieldInfo — metadata about a schema's pointer fields
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PointerFieldInfo:
    """Metadata extracted from a single pointer field on a cell schema.

    Attributes
    ----------
    field_name:
        The attribute name on the schema class (e.g. ``"gene_expression"``).
        By convention this doubles as the feature space name.
    feature_space:
        The feature space this pointer references (e.g. ``"gene_expression"``).
        Used to look up the matching ``GroupSpec`` via ``get_spec()``.
    pointer_kind:
        Whether the feature data is stored as a sparse or dense zarr array.
        Must agree with what the registered ``GroupSpec`` expects.
    pointer_type:
        The concrete pointer class — either ``SparseZarrPointer`` or
        ``DenseZarrPointer``.
    """

    field_name: str
    feature_space: str
    pointer_kind: PointerKind
    pointer_type: type  # SparseZarrPointer or DenseZarrPointer


def _extract_pointer_fields(
    schema_cls: type[LancellBaseSchema],
) -> dict[str, PointerFieldInfo]:
    """Introspect a schema class and return info for each pointer field.

    Pointer fields (``SparseZarrPointer`` / ``DenseZarrPointer``) are not
    obs metadata — they are references to external zarr arrays that hold the
    actual feature matrices.  They must be excluded from obs alignment logic
    and tracked separately so the atlas knows which feature spaces a schema
    supports.

    We can't rely on the raw annotation alone because:

    * Fields may be optional (``SparseZarrPointer | None``), requiring union
      unpacking to find the concrete pointer type.
    * We need to verify that the pointer kind (sparse vs. dense) matches what
      the registered ``GroupSpec`` for that feature space expects, and this
      cross-check lives here rather than on the pointer class itself.
    """
    result: dict[str, PointerFieldInfo] = {}
    for name, annotation in schema_cls.__annotations__.items():
        if name == "uid":
            continue
        origin = get_origin(annotation)
        if origin is Union or isinstance(annotation, UnionType):
            inner_types = get_args(annotation)
        else:
            inner_types = (annotation,)

        for t in inner_types:
            if t is type(None):
                continue
            if t is SparseZarrPointer or t is DenseZarrPointer:
                # Convention: field name == feature space name.
                # Enforced at class-definition time in LancellBaseSchema.__init_subclass__.
                feature_space = name
                spec = get_spec(feature_space)
                pointer_kind = PointerKind.SPARSE if t is SparseZarrPointer else PointerKind.DENSE
                if pointer_kind is not spec.pointer_kind:
                    raise TypeError(
                        f"Field '{name}' uses {pointer_kind.value} pointer but "
                        f"feature space '{feature_space}' requires {spec.pointer_kind.value}"
                    )
                result[name] = PointerFieldInfo(
                    field_name=name,
                    feature_space=feature_space,
                    pointer_kind=pointer_kind,
                    pointer_type=t,
                )
                break
    return result


# ---------------------------------------------------------------------------
# Arrow-schema-based pointer inference (schema-less read path)
# ---------------------------------------------------------------------------

_SPARSE_SUBFIELDS = {"feature_space", "zarr_group", "start", "end", "zarr_row"}
_DENSE_SUBFIELDS = {"feature_space", "zarr_group", "position"}


def _infer_pointer_fields_from_arrow(
    arrow_schema: pa.Schema,
) -> dict[str, PointerFieldInfo]:
    """Infer pointer fields from a cell table's Arrow schema.

    Detects struct columns whose sub-field names match the signatures of
    ``SparseZarrPointer`` or ``DenseZarrPointer``.  This allows read-path
    code to work without the original Python schema class.
    """
    result: dict[str, PointerFieldInfo] = {}
    for i in range(len(arrow_schema)):
        field = arrow_schema.field(i)
        if not pa.types.is_struct(field.type):
            continue
        sub_names = {field.type.field(j).name for j in range(field.type.num_fields)}
        if sub_names == _SPARSE_SUBFIELDS:
            pointer_kind = PointerKind.SPARSE
            pointer_type = SparseZarrPointer
        elif sub_names == _DENSE_SUBFIELDS:
            pointer_kind = PointerKind.DENSE
            pointer_type = DenseZarrPointer
        else:
            continue

        feature_space = field.name
        spec = get_spec(feature_space)
        if pointer_kind is not spec.pointer_kind:
            raise TypeError(
                f"Arrow field '{feature_space}' looks like a {pointer_kind.value} pointer "
                f"but the registered spec requires {spec.pointer_kind.value}"
            )
        result[feature_space] = PointerFieldInfo(
            field_name=feature_space,
            feature_space=feature_space,
            pointer_kind=pointer_kind,
            pointer_type=pointer_type,
        )
    return result


# ---------------------------------------------------------------------------
# Pre-flight schema alignment
# ---------------------------------------------------------------------------


def _schema_obs_fields(
    cell_schema: type[LancellBaseSchema],
) -> dict[str, bool]:
    """Return {field_name: required} for user-supplied obs fields.

    Excludes auto-generated fields (uid, dataset_uid) and pointer fields.
    """
    pointer_fields = _extract_pointer_fields(cell_schema)
    result: dict[str, bool] = {}
    for name, field_info in cell_schema.model_fields.items():
        if name in AUTO_FIELDS or name in pointer_fields:
            continue
        required = field_info.is_required()
        result[name] = required
    return result


def validate_obs_columns(
    obs: pd.DataFrame,
    cell_schema: type[LancellBaseSchema],
    obs_to_schema: dict[str, str] | None = None,
) -> list[str]:
    """Validate that obs columns match the cell schema.

    Parameters
    ----------
    obs:
        The obs DataFrame from an AnnData.
    cell_schema:
        The schema class to validate against.
    obs_to_schema:
        Optional mapping from obs column names to schema field names.
        Use this when obs columns have different names than the schema
        expects, e.g. ``{"donor_id": "donor", "cell_type_ontology": "cell_type"}``.

    Returns
    -------
    list[str]
        List of error strings. Empty list means valid.
    """
    errors: list[str] = []
    schema_fields = _schema_obs_fields(cell_schema)
    obs_to_schema = obs_to_schema or {}

    # Build the set of schema field names reachable from obs columns
    # (either directly or via the mapping)
    reverse_map = {v: k for k, v in obs_to_schema.items()}
    obs_cols = set(obs.columns)

    for field_name, required in schema_fields.items():
        # Field is satisfied if obs has it directly or via mapping
        obs_col = reverse_map.get(field_name, field_name)
        if required and obs_col not in obs_cols:
            errors.append(f"Missing required column '{field_name}'")

    return errors


def align_obs_to_schema(
    adata: ad.AnnData,
    cell_schema: type[LancellBaseSchema],
    *,
    obs_to_schema: dict[str, str] | None = None,
    inplace: bool = False,
) -> ad.AnnData:
    """Align an AnnData's obs to match a cell schema.

    - Renames columns according to ``obs_to_schema``.
    - Raises if required fields are missing (after renaming).
    - Adds ``None`` columns for optional fields not present.
    - Drops extra columns not in the schema.

    Parameters
    ----------
    adata:
        The AnnData to align.
    cell_schema:
        The schema class to align to.
    obs_to_schema:
        Optional mapping from obs column names to schema field names.
        Use this when obs columns have different names than the schema
        expects, e.g. ``{"donor_id": "donor", "cell_type_ontology": "cell_type"}``.
    inplace:
        If True, modify ``adata`` in place. Otherwise return a copy.

    Returns
    -------
    ad.AnnData
        The aligned AnnData.
    """
    errors = validate_obs_columns(adata.obs, cell_schema, obs_to_schema)
    if errors:
        raise ValueError(f"Cannot align obs to schema: {errors}")

    if not inplace:
        adata = adata.copy()

    # Rename obs columns according to mapping
    if obs_to_schema:
        adata.obs = adata.obs.rename(columns=obs_to_schema)

    schema_fields = _schema_obs_fields(cell_schema)
    obs_cols = set(adata.obs.columns)

    # Add None columns for optional fields not present
    for field_name, required in schema_fields.items():
        if not required and field_name not in obs_cols:
            adata.obs[field_name] = None

    # Drop extra columns not in schema
    keep = [c for c in adata.obs.columns if c in schema_fields]
    adata.obs = adata.obs[keep]

    return adata
